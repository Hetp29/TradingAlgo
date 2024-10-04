#include <iostream>
#include "OrderBook.h"

int main() {
    OrderBook orderBook;

    // Adding limit orders
    Order buyOrder1(1, Order::Type::BUY, 100.50, 10);  // Buy 10 shares at $100.50
    Order sellOrder1(2, Order::Type::SELL, 100.00, 5); // Sell 5 shares at $100.00

    orderBook.addOrder(buyOrder1);
    orderBook.addOrder(sellOrder1);

    // Adding market orders
    Order marketBuyOrder(3, Order::Type::MARKET_BUY, 7);  // Buy 7 shares at market price

    orderBook.addOrder(marketBuyOrder);

    // Display the order book
    orderBook.displayOrders();

    // Display trade history (optional)
    orderBook.displayTradeHistory();

    return 0;
}
