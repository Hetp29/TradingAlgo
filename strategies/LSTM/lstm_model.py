#create LSTM model that will be trained on stock data
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout # type: ignore

def build_lstm_model(input_shape):
    model = Sequential()
    
    #LSTM layer with 50 units
    model.add(LSTM(units = 50, return_sequence=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units = 50, return_sequence = False))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model
