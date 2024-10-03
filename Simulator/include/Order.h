//this represents a class for Order that represents buy and sell orders
#indef ORDER_H
#define ORDER_H

#include <string>

class Order {
    public:
        enum class Type { BUY, SELL };

        Order(int id, Type type, double price, int quantity);

        int getId() const;
        Type getType() const;
        double getPrice() const;
        int getQuantity() const;

        void setQuantity(int quantity);
    
    private:
        int id; //ID for order
        Type type; //BUY or SELL order type
        double price; //price per share
        int quantity; //number of shares
};

#endif

//declarations of functions, classes and constructs 
//data and function of class