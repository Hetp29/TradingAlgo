import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Portfolio:
    def __init__(self, initial_balance=100, risk_tolerance=0.05):
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.positions = {}  #number of shares held
        self.trade_history = []  #List of tuples (date, ticker, action, price, shares)
        self.risk_tolerance = risk_tolerance  #Max % loss you're willing to accept per trade
        self.equity_curve = []  #Track portfolio value over time

    def buy(self, ticker, price, shares, date):
        total_cost = price * shares
        if self.cash_balance >= total_cost:
            self.cash_balance -= total_cost
            if ticker in self.positions:
                self.positions[ticker] += shares
            else:
                self.positions[ticker] = shares

        
            self.trade_history.append((date, ticker, 'buy', price, shares))
            print(f"Bought {shares} shares of {ticker} at {price} on {date}")
        else:
            print(f"Not enough cash to buy {shares} shares of {ticker} at {price} on {date}")

    def sell(self, ticker, price, shares, date):
        
        if ticker in self.positions and self.positions[ticker] >= shares:
            self.positions[ticker] -= shares
            total_revenue = price * shares
            self.cash_balance += total_revenue

        
            self.trade_history.append((date, ticker, 'sell', price, shares))
            print(f"Sold {shares} shares of {ticker} at {price} on {date}")

            
            if self.positions[ticker] == 0:
                del self.positions[ticker]
        else:
            print(f"Not enough shares to sell {shares} of {ticker} on {date}")

    def value(self, stock_prices):
        """Calculate the current portfolio value based on cash and open positions."""
        stock_value = sum([shares * stock_prices[ticker] for ticker, shares in self.positions.items()])
        total_value = self.cash_balance + stock_value
        self.equity_curve.append(total_value)
        return total_value

    def risk_management(self, price_data, ticker):
        """Implement risk management strategies, like stop loss or trailing stop."""
        if ticker not in self.positions:
            return  

        current_price = price_data['Close'].iloc[-1]
        initial_price = self.trade_history[-1][3]  # Price at the last trade
        position_value = current_price * self.positions[ticker]

        
        if (initial_price - current_price) / initial_price > self.risk_tolerance:
            
            self.sell(ticker, current_price, self.positions[ticker], price_data.index[-1])
            print(f"Stop loss triggered for {ticker}")

    def plot_equity_curve(self):
        """Plot the equity curve of the portfolio over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.equity_curve, label="Portfolio Value")
        plt.title("Portfolio Equity Curve")
        plt.xlabel("Trades")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.show()

    def optimize_positions(self, stock_forecasts):
        """Optimize positions based on expected returns and risk."""

        for ticker, forecast in stock_forecasts.items():
            if forecast > 0:  
                shares_to_buy = (self.cash_balance * 0.1) // forecast  # Buy 10% of portfolio value
                self.buy(ticker, forecast, shares_to_buy, pd.Timestamp.now())
            elif ticker in self.positions:  
                self.sell(ticker, forecast, self.positions[ticker], pd.Timestamp.now())

    def performance_report(self):
        """Generate a performance report of the portfolio."""
        total_value = self.value({ticker: 1 for ticker in self.positions})  # Set prices to 1 for total value
        print(f"Initial Balance: ${self.initial_balance}")
        print(f"Current Balance: ${self.cash_balance}")
        print(f"Portfolio Value: ${total_value}")
        print(f"Open Positions: {self.positions}")
        print(f"Trade History:")
        for trade in self.trade_history:
            print(trade)
    
    def trade_summary(self):
        """Summarize wins/losses and key stats for your trades."""
        trade_data = pd.DataFrame(self.trade_history, columns=["Date", "Ticker", "Action", "Price", "Shares"])
        buys = trade_data[trade_data['Action'] == 'buy']
        sells = trade_data[trade_data['Action'] == 'sell']

        print("Summary of Buys:")
        print(buys.describe())
        print("\nSummary of Sells:")
        print(sells.describe())
