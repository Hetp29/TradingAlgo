#ifndef ORDERBOOK_H
#define ORDERBOOK_H

#include <vector>
#include "Order.h"

class OrderBook {
public:
    void addOrder(const Order& order);
    void displayOrders() const;

private:
    std::vector<Order> buyOrders;
    std::vector<Order> sellOrders;
};

#endif
