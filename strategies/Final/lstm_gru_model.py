#Long Short-term memory is type of neural network to remember long-term dependencies in time-series data
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, LSTM, Guru # type: ignore

def load_data(file_path):
    data = pd.read_csv(file_path)
    close_prices = data['Close'].values.reshape(-1, 1) #Extract "Close" prices and scale between 0 and 1
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)
    return scaled_data, scaler

#slice scaled data into sequences, store X(input) as stock price from i to i + time_step and y(target) as stock market price at i + time_step 
def create_data(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)
    