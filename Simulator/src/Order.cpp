#include "Order.h"

Order::Order(int id)

Order::Order(int id, Type type, double price, int quantity)
    : id(id), type(type), price(price), quantity(quantity) {}

int Order::getId() const {
    return id; //getter
}

Order::Type Order::getType() const {
    return type;
}

double Order::getPrice() const {
    return price;
}

int Order::getQuantity() const {
    return quantity;
}

void Order::setQuantity(int newQuantity) {
    quantity = newQuantity;
}