#include "OrderBook.h"
#include <iostream>
#include <algorithm>

void OrderBook::addOrder(const Order& order) {
    if (order.getType() == Order::Type::BUY || order.getType() == Order::Type::MARKET_BUY) {
        buyOrders.push_back(order);
    } else if (order.getType() == Order::Type::SELL || order.getType() == Order::Type::MARKET_SELL) {
        sellOrders.push_back(order);
    }

    matchOrders(); //match after adding each new order
}
//add order to book

void OrderBook:matchOrders() {
    std::sort(buyOrders.begin(), buyOrders.end(), [](const Order& a, const Order& b) {
        return a.getPrice() > b.getPrice();
    });

    std::sort(sellOrders.begin(), sellOrders.end(), [](const Order& a, const Order& b) {
        return a.getPrice() < b.getPrice();
    });

    while(!buyOrders.empty() && !sellOrders.empty()) {
        Order& buyOrder = buyOrders.front(); //first element
        Order& sellOrder = sellOrders.front();

        if(buyOrder.getPrice() >= sellOrder.getPrice()) {
            executeTrade(buyOrder, sellOrder);

            if (buyOrder.getQuantity() == 0) buyOrders.erase(buyOrders.begin());
            if (sellOrder.getQuantity() == 0) sellOrders.erase(sellOrders.begin());
            continue;
        }

        if(sellOrder.getType() == Order::Type::MARKET_SELL) {
            executeTrade(buyOrder, sellOrder);
            if(buyOrder.getQuantity() == 0) buyOrders.erase(buyOrders.begin());
            if(sellOrder.getQuantity() == 0) sellOrders.erase(sellOrders.begin());
            continue;
        }

        if(buyOrder.getPrice() >= sellOrder.getPrice()) {
            executeTrade(buyOrder, sellOrder);

            if(buyOrder.getQuantity() == 0) buyOrders.erase(buyOrders.begin());
            if(sellOrder.getQuantity() == 0) sellOrders.erase(sellOrders.begin());
        }

        else {
            break;
        }
    }
}

void OrderBook::executeTrade(Order& buyOrder, Order& sellOrder) {
    int tradeQuantity = std::min(buyOrder.getQuantity(), sellOrder.getQuantity());
    double tradePrice = (buyOrder.getType() == Order::Type::MARKET_BUY || sellOrder.getType() == Order::Type::MARKET_SELL)
                        ? sellOrder.getPrice() : sellOrder.getPrice();

    std::cout << "Trade executed: " << tradeQuantity << " shares at $" << tradePrice << std::endl;
        
    buyOrder.setQuantity(buyOrder.getQuantity() - tradeQuantity);
    sellOrder.setQuantity(sellOrder.getQuantity() - tradeQuantity);
}

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

void OrderBook::displayTradeHistory() const {
    std::cout << "Trade History:" <<std::endl;
}