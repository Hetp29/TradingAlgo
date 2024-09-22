import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data(filename, window_size=60, features=['Adj Close']):
    df = pd.read_csv(filename)
    
    
    data = df[features].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, Y = [], []
    
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        Y.append(scaled_data[i, 0])  
    
    X, Y = np.array(X), np.array(Y)
    
    
    X = np.reshape(X, (X.shape[0], X.shape[1], len(features)))
    
    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=False)
    
    return X_train, X_val, Y_train, Y_val, scaler

def save_scaler(scaler, scaler_filename):
    """Save the scaler to a file."""
    import joblib
    joblib.dump(scaler, scaler_filename)
