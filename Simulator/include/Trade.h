#ifndef TRADE_H
#define TRADE_H

class Trade {
public:
    Trade(int buyOrderId, int sellOrderId, double price, int quantity);


    int getBuyOrderId() const;
    int getSellOrderId() const;
    double getPrice() const;
    int getQuantity() const;


    void displayTrade() const;

private:
    int buyOrderId;  
    int sellOrderId; 
    double price;    
    int quantity;   
};

#endif
