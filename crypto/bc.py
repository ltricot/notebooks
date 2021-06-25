#!/usr/bin/env python
# coding: utf-8

# # The simplest blockchain
# 
# Says it all.
# 
# We start by finding a difficulty giving 1s block time. We don't care to hash super efficiently since we aren't securing a trillion dollars. Must loop $2^{22}$ times on MBP M1 to get a reasonable block time with one worker.

# In[ ]:

from __future__ import annotations

from hashlib import sha256
from array import array
from random import randint
import sys


def h(bts):
    m = sha256()
    m.update(bts)
    return m.digest()

def good(difficulty):
    def inner(_h):
        i = int.from_bytes(_h, 'big')
        return i < 2 ** (8 * 32 - difficulty)
    return inner

# don't care about being efficient, we are alone :)
def find(bts, difficulty):
    nonce = randint(0, 2 << 8 * 4)
    a = array('Q')
    a.frombytes((nonce).to_bytes(8, sys.byteorder) + bts)  # space for nonce
    
    g = good(difficulty)
    _h = h(a)
    while not g(_h):
        a[0] += 1
        _h = h(a)
    
    return a[0]

difficulty = 23


# We now work bottom up to create a blockchain.

# In[ ]:


from dataclasses import dataclass, field
from typing import List
import struct


zero = b'\0' * 32

Addr  = bytes  # address on the network
THash = bytes  # transaction hash
BHash = bytes  # block hash

@dataclass
class Tx:
    utxo: THash
    to:   Addr
    ix:   int  # block identifier

    def __bytes__(self):
        return struct.pack('32s32sQ', self.utxo, self.to, self.ix)

    @classmethod
    def frombytes(cls, bts):
        return cls(*struct.unpack('32s32sQ', bts))

@dataclass
class Block:
    txs:   List[Tx]  # notion of order
    last:  BHash
    ix:    int       # index of block for consensus
    nonce: bytes     # len 8

    def __bytes__(self):
        return (
            self.nonce + self.last + self.ix.to_bytes(8, sys.byteorder)
            + b''.join(bytes(tx) for tx in self.txs)
        )

    @classmethod
    def frombytes(cls, bts):
        nonce, last = bts[:8], bts[8:40]
        ix   = int.from_bytes(bts[40:48], sys.byteorder)
        m = memoryview(bts[48:])
        txs = [Tx.frombytes(m[o:o + 72]) for o in range(0, len(m), 72)]
        return cls(txs, last, ix, nonce)

    # for min-heap sorting
    def __le__(self, o: Block):
        return self.ix >= o.ix


# In[ ]:


from dataclasses import field
from typing import Dict, Set


@dataclass
class Blockchain:
    blocks: List[Block]
    txs:    Set[THash]
    spent:  Set[THash]
    pool:   Dict[THash, Tx] = field(default_factory=dict)

    @property
    def head(self) -> Block:
        return self.blocks[-1]

    @classmethod
    def genesis(cls, qty: int) -> Blockchain:
        txs = list(
            Tx(utxo=zero, to=i.to_bytes(32, sys.byteorder), ix=0)
            for i in range(qty)
        )

        genesis = Block(txs, last=zero, ix=0, nonce=zero[:8])
        genesis.nonce = find(bytes(genesis)[8:], difficulty).to_bytes(8, sys.byteorder)
        return cls([genesis], set(h(bytes(tx)) for tx in txs), set())

    def verify(self, b: Block) -> bool:
        return (
            b.ix == self.head.ix + 1                             # correct block ordering
            and good(difficulty)(h(bytes(b)))                    # respect difficulty
            and all(tx.utxo in self.pool for tx in b.txs)        # transactions in mempool
            and len(set(tx.utxo for tx in b.txs)) == len(b.txs)  # no double spend
            and all(tx.utxo not in self.spent for tx in b.txs)   # certainly no double spend
            and all(tx.utxo in self.txs for tx in b.txs)         # spend money which exists
            and all(tx.ix == b.ix for tx in b.txs)               # valid transaction data
            and b.last == h(bytes(self.head))                    # valid block hash
        )

    def tx(self, utxo: THash, to: Addr):
        self.pool[utxo] = to

    async def mine(self, executor) -> Block:
        # transaction data
        txs = [
            Tx(utxo=utxo, to=to, ix=self.head.ix + 1)
            for utxo, to in self.pool.items()
        ]

        # mine hash in other process
        b = Block(txs, last=h(bytes(self.head)), ix=self.head.ix + 1, nonce=zero[:8])
        bts = bytes(b)[8:]

        loop = aio.get_running_loop()
        t = loop.run_in_executor(executor, find, bts, difficulty)
        await t

        b.nonce = t.result().to_bytes(8, sys.byteorder)
        return b

    def add(self, b: Block):
        assert self.verify(b)
        self.blocks.append(b)

        # add txs to spent, remove from pool, add to unspent
        for tx in b.txs:
            _h = h(bytes(tx))
            self.spent.add(tx.utxo)
            del self.pool[tx.utxo]
            self.txs.add(_h)

        print(f'added block: {b.ix}: {h(bytes(b))} {b.last}')


# Now that we have a functioning blockchain, we can work on a toy network layer. It has two responsibilities: propagating transactions and blocks, and forming consensus (longest chain). All clients are full clients. All clients broadcast to all (known) clients.

# In[ ]:


import asyncio as aio


@dataclass
class Msg:
    tp:  int
    bts: bytes

    def __bytes__(self):
        return (
            self.tp.to_bytes(8, sys.byteorder)
            + self.bts
        )
    
    @classmethod
    def frombytes(cls, bts: bytes):
        tp = int.from_bytes(bts[:8], sys.byteorder)
        return cls(tp, bts[8:])

@dataclass
class Peer:
    ip:   str
    port: int
    
    def __bytes__(self):
        return self.port.to_bytes(8, sys.byteorder) + self.ip.encode()

    @classmethod
    def frombytes(cls, bts):
        port = int.from_bytes(bts[:8], sys.byteorder)
        ip   = bts[8:].decode()
        return cls(ip, port)

@dataclass
class Protocol:
    bc: Blockchain
    heads: Dict[BHash, Blockchain]

    peers = set()

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        self.addpeer(addr, addr)

        msg = Msg.frombytes(data)

        if msg.tp == 0:    # tx
            tx = Tx.frombytes(msg.bts)
            self.bc.tx(tx.utxo, tx.to)

        elif msg.tp == 1:  # block
            b = Block.frombytes(msg.bts)
            print(f'get block {b.ix} {h(bytes(b))} {b.last}')

            for head, _bc in self.heads.items():
                if b.last == head:
                    if _bc.verify(b):
                        _bc.add(b)
                        del self.heads[head]
                        self.heads[h(bytes(b))] = _bc
                    else:
                        print('failed verification')

                    break
            else:  # head not found, add new chain
                if b.ix == 0:
                    print('discover new chain')
                    _bc = Blockchain([b], set(h(bytes(tx)) for tx in b.txs), set())
                    self.heads[h(bytes(b))] = _bc
                else:
                    print('unknown block')
                    _bc = self.bc

            if len(_bc.blocks) > len(self.bc.blocks):
                self.bc = _bc

            for peer in self.__class__.peers:
                if peer == addr:
                    continue

                self.sendblock(peer, b)

        elif msg.tp == 2:  # request chain
            print('send chain')
            ix = struct.unpack('Q', msg.bts)

            # send all blocks of longest blockchain
            for b in self.bc.blocks:
                self.sendblock(addr, b)

        elif msg.tp == 3:
            p = Peer.frombytes(msg.bts)
            p = (p.ip, p.addr)
            self.addpeer(addr, p)
        
        else:
            ...

    def addpeer(self, addr, p):
        if p not in self.__class__.peers:
            self.__class__.peers.add(p)
            print(f'add peer {p}')

            for op in self.__class__.peers:
                if op not in (p, addr):
                    self.sendpeer(addr, op)

                    print(f'send {op} to {addr}')
    
    def connection_lost(self, exc):
        ...

    def sendblock(self, addr, b: Block):
        self.transport.sendto(bytes(Msg(1, bytes(b))), addr)
    
    def getchain(self, ix: int):
        self.transport.sendto(bytes(Msg(2, ix.to_bytes(8, sys.byteorder))))

    def sendpeer(self, addr, p):
        self.transport.sendto(bytes(Msg(3, bytes(Peer(*p)))), addr)


# In[ ]:

from concurrent.futures import ProcessPoolExecutor as PPE


async def main(*peers):
    ts = []
    loop = aio.get_running_loop()

    bc = Blockchain.genesis(1)
    heads = {h(bytes(bc.blocks[0])): bc}

    if not peers:
        t, protocol = await loop.create_datagram_endpoint(
            lambda: Protocol(bc, heads),
            local_addr=('127.0.0.1', 9999))
        
        print('server up')

    else:
        for addr, port in peers:
            t, protocol = await loop.create_datagram_endpoint(
                lambda: Protocol(bc, heads),
                remote_addr=(addr, port))
            
            ts.append(t)
        
        print('get chain')
        protocol.getchain(0)

    try:
        ex = PPE(max_workers=1)
        while True:
            head = max(heads, key=lambda _h: len(heads[_h].blocks))
            bc = heads[head]
            b = await bc.mine(ex)

            if bc.verify(b):
                bc.add(b)

                del heads[head]
                heads[h(bytes(b))] = bc

                for addr in Protocol.peers:
                    protocol.sendblock(addr, b)

            print(f'MINED {b.ix}')

    finally:
        for t in ts:
            t.close()
