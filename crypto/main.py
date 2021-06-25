from bc import *
import sys


if __name__ == '__main__':
    peers = []
    if len(sys.argv) == 2:
        peers.append(('127.0.0.1', int(sys.argv[1])))

    aio.run(main(*peers))
