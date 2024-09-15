#Mean reversion:
#buy when stock prices is significantly below rolling mean (z-score is below negative threshold)
#sell when stock prices is significantly above rolling mean (z-score is above positive threshold)
#z-score is number of standard deviations a value is from mean of distribution
import pandas as pd

def mean_reversion_strategy(df, window=20, z_score_threshold=1.5):
    df['Rolling_Mean'] = df['Adj Close'].rolling(window=window).mean() #calculate rolling mean
    df['Rolling_Std'] = df['Adj Close'].rolling(window=window).std() #calculate rolling standard deviation
    
    df['Z-Score'] = (df['Adj Close'] - df['Rolling_Mean']) / df['Rolling_Std']
    
    df['Signal'] = 0
    df.loc[df['Z-Score'] < -z_score_threshold, 'Signal'] = 1 #buy signal
    df.loc[df['Z-Score'] > z_score_threshold, 'Signal'] = -1 #sell signal
    
    return df

##rolling mean is average price over a given number of past days (20 by default), current price versus recent average
#rolling standard deviation is how much price tends to fluctuate around average
#z score is how far current price is from rolling mean 
#buy when z score is less than -1.5 (price is lower than average)
#sell when z score is greater than 1.5 (price is higher than average)