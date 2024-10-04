import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def calculate_technical_indicators(df):
    """Add technical indicators to the dataframe."""
    
    #Moving Averages
    df['MA_50'] = df['Adj Close'].rolling(window=50).mean()  # 50-day Moving Average
    df['MA_200'] = df['Adj Close'].rolling(window=200).mean()  # 200-day Moving Average

    #Exponential Moving Averages
    df['EMA_50'] = df['Adj Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Adj Close'].ewm(span=200, adjust=False).mean()

    #Bollinger Bands
    df['BB_upper'] = df['MA_50'] + (df['Adj Close'].rolling(window=50).std() * 2)
    df['BB_lower'] = df['MA_50'] - (df['Adj Close'].rolling(window=50).std() * 2)

    #Relative Strength Index (RSI)
    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.fillna(0, inplace=True) 

    return df

def prepare_data(filename, window_size=60, features=['Adj Close']):
    df = pd.read_csv(filename)
    

    df = calculate_technical_indicators(df)
    

    selected_features = ['Adj Close', 'MA_50', 'MA_200', 'EMA_50', 'EMA_200', 'BB_upper', 'BB_lower', 'RSI']
    data = df[selected_features].values
    
    #Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, Y = [], []
    
    #Prepare the dataset for LSTM
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        Y.append(scaled_data[i, 0])  #Predict 'Adj Close' 
    
    X, Y = np.array(X), np.array(Y)
    

    X = np.reshape(X, (X.shape[0], X.shape[1], len(selected_features)))
    

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=False)
    
    return X_train, X_val, Y_train, Y_val, scaler

def save_scaler(scaler, scaler_filename):
    """Save the scaler to a file."""
    import joblib
    joblib.dump(scaler, scaler_filename)
