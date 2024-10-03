#include "OrderBook.h"
#include <iostream>

void OrderBook::addOrder(const Order& order) {
    if (order.getType() == Order::Type::BUY) {
        buyOrders.push_back(order);
    } else {
        sellOrders.push_back(order);
    }
}
//add order to book

void OrderBook::displayOrders() const {
    std::cout << "Buy Orders:" << std::endl;
    for (const auto& order : buyOrders) {
        std::cout << "ID: " << order.getId() << ", Price: " << order.getPrice() 
                << ", Quantity: " << order.getQuantity() << std::endl;
    }

    std::cout << "Sell Orders:" << std::endl;
    for (const auto& order : sellOrders) {
        std::cout << "ID: " << order.getId() << ", Price: " << order.getPrice() 
                << ", Quantity: " << order.getQuantity() << std::endl;
    }
}
//display orders in order book