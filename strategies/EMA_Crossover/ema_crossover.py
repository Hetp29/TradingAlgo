import pandas as pd
import numpy as np

def ema_crossover_strategy(df, short_window=12, long_window=26, rsi_window=14, volume_confirmation=True, stop_loss_pct=0.02, take_profit_pct=0.05):
    #first, we calculate short-term and long-term EMAs
    df['Short_EMA'] = df['Adj Close'].ewm(span=short_window, adjust=False).mean()
    df['Long_EMA'] = df['Adj Close'].ewm(span=long_window, adjust=False).mean()
    
    #RSI calculation
    #relative strength index is momentum indication used in technical analysis 
    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    
    #buy/sell signals based on EMA cross and RSI
    df['Signal'] = 0
    buy_signal = (df['Short_EMA'] > df['Long_EMA']) & (df['RSI'] < 70)
    sell_signal = (df['Short_EMA'] < df['Long_EMA']) & (df['RSI'] > 30)
    
    if volume_confirmation:
        avg_volume = df['Volume'].rolling(window=20).mean()
        buy_signal &= (df['Volume'] > avg_volume)  
        sell_signal &= (df['Volume'] > avg_volume)
        
    df.loc[buy_signal, 'Signal'] = 1  
    df.loc[sell_signal, 'Signal'] = -1  
    
    #risk management with stop-loss and take-profit
    df['Stop_Loss'] = np.nan
    df['Take_Profit'] = np.nan
    df['Position'] = 0
    
    for i in range(1, len(df)):
        if df['Signal'].loc[i] == 1:
            df['Position'].loc[i] = 1
            entry_price = df['Adj Close'].loc[i]
            df['Stop_Loss'].loc[i] = entry_price * (1 - stop_loss_pct)
            df['Take_Profit'].loc[i] = entry_price * (1 + take_profit_pct)
        
        elif df['Signal'].loc[i] == -1:
            df['Position'].loc[i] = -1
            
        if df['Position'].loc[i-1] == 1:  
            if df['Adj Close'].loc[i] <= df['Stop_Loss'].iloc[i-1]:
                df['Signal'].loc[i] = -1  #Sell
                df['Position'].loc[i] = -1
            elif df['Adj Close'].loc[i] >= df['Take_Profit'].iloc[i-1]:
                df['Signal'].loc[i] = -1
                df['Position'].loc[i] = -1  
    return df

if __name__ == "__main__":
    df = pd.read_csv("../../data/NVDA_data.csv")
    
    df = ema_crossover_strategy(df)
    
    print(df[['Adj Close', 'Short_EMA', 'Long_EMA', 'RSI', 'Signal', 'Stop_Loss', 'Take_Profit']].tail(20))  # print last 20 rows
        