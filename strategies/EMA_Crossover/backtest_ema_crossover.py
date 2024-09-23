import pandas as pd
import numpy as np

def backtest_ema_crossover(df, initial_balance=10000, risk_per_trade=0.02):
    balance = initial_balance
    balance_history = [balance]
    positions = 0  
    trade_log = []  
    
    for i in range(1, len(df)):
        if df['Signal'].iloc[i] == 1 and positions == 0: 
            shares_bought = (balance * risk_per_trade) / df['Adj Close'].loc[i]
            balance -= shares_bought * df['Adj Close'].loc[i]
            positions = shares_bought
            trade_log.append(f"Buy {shares_bought} shares at {df['Adj Close'].loc[i]} on {df.index[i]}")
        
        elif df['Signal'].iloc[i] == -1 and positions > 0:  
            balance += positions * df['Adj Close'].loc[i]
            trade_log.append(f"Sell {positions} shares at {df['Adj Close'].loc[i]} on {df.index[i]}")
            positions = 0 
        
        balance_history.append(balance + (positions * df['Adj Close'].loc[i]))
        
    balance_df = pd.DataFrame({'Balance': balance_history}, index=df.index)
    
    total_return = (balance_history[-1] - initial_balance) / initial_balance * 100
    num_trades = len(trade_log)
    max_drawdown = (np.max(balance_history) - np.min(balance_history)) / np.max(balance_history) * 100
    
    print(f"Final Balance: ${balance_history[-1]:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of Trades: {num_trades}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")

    return balance_df, trade_log

if __name__ == "__main__":
    df = pd.read_csv("../../data/NVDA_data.csv")
    from ema_crossover import ema_crossover_strategy
    df = ema_crossover_strategy(df)
    
    balance_df, trade_log = backtest_ema_crossover(df)
    
    print("Trade Log:")
    for trade in trade_log:
        print(trade)

