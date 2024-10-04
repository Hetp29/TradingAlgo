import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import joblib

def load_data(data_filename):
    data = pd.read_csv(data_filename)
    logging.info(f"Loaded data from {data_filename}")
    logging.info(f"Data type after loading: {type(data)}")
    return data

def scale_data(data):
    logging.info(f"Data type before scaling: {type(data)}, shape: {data.shape}")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    logging.info("Data scaled using MinMaxScaler.")
    logging.info(f"Scaled data type: {type(scaled_data)}, shape: {scaled_data.shape}")
    return scaled_data, scaler

def create_dataset(data, window_size=60):
    data = np.array(data)  # Ensure data is a NumPy array
    X, Y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        Y.append(data[i, 0])  # Predicting 'Adj Close'
    return np.array(X), np.array(Y)

def split_data(X, Y, train_size=0.8):
    split_index = int(len(X) * train_size)
    X_train, X_val = X[:split_index], X[split_index:]
    Y_train, Y_val = Y[:split_index], Y[split_index:]
    logging.info("Data split into training and validation sets.")
    return X_train, X_val, Y_train, Y_val

def calculate_technical_indicators(df):
    df['MA_50'] = df['Adj Close'].rolling(window=50).mean()
    df['MA_200'] = df['Adj Close'].rolling(window=200).mean()
    df['EMA_50'] = df['Adj Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Adj Close'].ewm(span=200, adjust=False).mean()
    df['BB_upper'] = df['MA_50'] + (df['Adj Close'].rolling(window=50).std() * 2)
    df['BB_lower'] = df['MA_50'] - (df['Adj Close'].rolling(window=50).std() * 2)
    
    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df.fillna(0, inplace=True)
    return df

def prepare_data(data_filename, window_size=60, features=['Adj Close']):
    df = load_data(data_filename)
    df = calculate_technical_indicators(df)
    
    # Select relevant features
    selected_features = ['Adj Close', 'MA_50', 'MA_200', 'EMA_50', 'EMA_200', 'BB_upper', 'BB_lower', 'RSI']
    data = df[selected_features].values
    
    # Scale data
    scaled_data, scaler = scale_data(data)

    # Ensure scaled_data is a NumPy array
    scaled_data = np.array(scaled_data)

    # Prepare the dataset for LSTM
    X, Y = create_dataset(scaled_data, window_size)
    
    # Split the dataset into training and validation sets
    X_train, X_val, Y_train, Y_val = split_data(X, Y)

    logging.info(f"Shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, Y_train: {Y_train.shape}, Y_val: {Y_val.shape}")

    return X_train, X_val, Y_train, Y_val, scaler

def save_scaler(scaler, scaler_filename):
    """Save the scaler to a file."""
    joblib.dump(scaler, scaler_filename)
    logging.info(f"Scaler saved to {scaler_filename}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
