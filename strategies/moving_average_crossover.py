#short term moving average (50 day) reacts quickly to price changes and represents price trends
#long term moving average (200 day) reacts slowly and reflects broader market trend
#buy signal, when short-term moving average crosses above long-term moving average
#sell signal, when short-term moving average crosses below long-term moving average

def moving_average_crossover_strategy(df, short_window=50,long_window = 200):
    df['Short_MA'] = df['Adj Close'].rolling(window=short_window).mean()
    df['Long_MA'] = df('Adj Close').rolling(window=long_window).mean()
    
    df['Signal'] = 0 #default is no signal
    
    df.loc[df['Short_MA'] > df['Long_MA'], 'Signal'] = 1 #Buy
    df.loc[df['Short_MA'] < df['Long_MA'], 'Signal'] = -1 #Sell
    
    return df

#buy as price moves above moving average and sell when it drops below
#returns a dataframe with buy and sell signals