# Reconstructing the order flow with optimal transport

Exchanges receive three types of orders:
- **Limit orders**, making liquidity
- **Market orders**, taking liquidity
- **Cancel orders**, taking liquidity

Limit orders are standing offers to buy (sell) a defined amount at a defined price. They can be cancelled using Cancel orders. Market orders are orders to buy a defined amount at market price: the best price which can be achieved using standing offers. Limit orders are accumulated and aggregated into a tick level order book, which contains the size of aggregated standing offers at each price. Market orders trigger one or more trades when they are matched with one or more Limit orders.

---

![Order book GIF](https://upload.wikimedia.org/wikipedia/commons/1/14/Order_book_depth_chart.gif)

---

Exchanges stream the order book updates and the trades flow to market participants, but never stream the order flow. Order book updates and trades are usually not aligned (w.r.t. time) and sometimes presented in aggregate form. The order flow is valuable data which can be used to study agents' behavior on the market, and possibly design algorithmic trading strategies. We attempt to reconstruct the order flow from the order book updates and trades flow.