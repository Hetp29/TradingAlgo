#include <iostream>
#include "OrderBook.h"

int main() {
    OrderBook orderBook;


    Order buyOrder1(1, Order::Type::BUY, 100.50, 10);  // Buy 10 shares at $100.50
    Order sellOrder1(2, Order::Type::SELL, 100.00, 5); // Sell 5 shares at $100.00
    Order buyOrder2(3, Order::Type::BUY, 99.75, 20);   // Buy 20 shares at $99.75
    Order sellOrder2(4, Order::Type::SELL, 99.50, 15); //sell 15 shares at $99.50


    orderBook.addOrder(buyOrder1);
    orderBook.addOrder(sellOrder1);
    orderBook.addOrder(buyOrder2);
    orderBook.addOrder(sellOrder2);

    orderBook.displayOrders();

    return 0;
}