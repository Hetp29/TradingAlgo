import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def prepare_data(filename, window_size=60):
    df = pd.read_csv(filename)
    
    data = df['Adj Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X_train = []
    Y_train = []
    
    for i in range(window_size, len(scaled_data)):
        X_train.append(scaled_data[i - window_size:i, 0])
        Y_train.append(scaled_data[i, 0])
    
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, Y_train, scaler