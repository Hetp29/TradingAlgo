#include "Trade.h"
#include <iostream>

Trade::Trade(int buyOrderId, int sellOrderId, double price, int quantity)
    : buyOrderId(buyOrderId), sellOrderId(sellOrderId), price(price), quantity(quantity) {}

int Trade::getBuyOrderId() const {
    return buyOrderId;
}

int Trade::getSellOrderId() const {
    return sellOrderId;
}

double Trade::getPrice() const {
    return price;
}

int Trade::getQuantity() const {
    return quantity;
}

void Trade::displayTrade() const {
    std::cout << "Trade - Buy Order ID: " << buyOrderId 
              << ", Sell Order ID: " << sellOrderId
              << ", Price: " << price
              << ", Quantity: " << quantity << std::endl;
}
