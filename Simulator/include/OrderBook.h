#ifndef ORDERBOOK_H
#define ORDERBOOK_H

#include <vector>
#include "Order.h"

class OrderBook {
public:
    void addOrder(const Order& order);
    void matchOrders(); 
    void displayOrders() const;

private:
    std::vector<Order> buyOrders;
    std::vector<Order> sellOrders;

    void executeTrade(Order& buyOrder , Order& sellOrder);
};

#endif
