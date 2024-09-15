import pandas as pd
import numpy as np 
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def backtest_strategy(data, initial_balance=10000, risk_per_trade=0.02):
    balance = initial_balance
    balance_history = [balance]
    positions = 0 #shares held
    trade_log = []
    
    for i in range(1, len(data)):
        if data['Signal'][i] == 1 and positions == 0:
            positions = (balance * risk_per_trade) / data['Adj Close'][i]
            balance -= positions * data['Adj Close'][i]
            trade_log.append(f"Buy {positions} shares at {data['Adj Close'][i]} on {data.index[i]}")
            
        elif data['Signal'][i] == -1 and positions > 0:
            balance += positions * data['Adj Close'][i]
            trade_log.append(f"Sell {positions} shares at {data['Adj Close'][i]} on {data.index[i]}")
            positions = 0
        
        balance_history.append(balance + (positions * data['Adj Close'][i]))
    
    balance_df = pd.DataFrame({'Balance': balance_history}, index=data.index)
    
    return balance_df, trade_log

if __name__ == "__main__":
    stock_data = pd.read_csv("../data/NVDA_data.csv", index_col='Date', parse_dates=True)
    
    from strategies.moving_average_crossover import moving_average_crossover_strategy
    
    stock_data = moving_average_crossover_strategy(stock_data)
    
    balance_df, trade_log = backtest_strategy(stock_data)
    
    print(f"Final Balance: ${balance_df['Balance'].iloc[-1]:.2f}")
    print("Trade Log:")
    for trade in trade_log:
        print(trade)