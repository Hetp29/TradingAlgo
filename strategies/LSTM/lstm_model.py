#create LSTM model that will be trained on stock data
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization # type: ignore

def build_lstm_model(input_shape):
    model = Sequential()
    
    
    model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    

    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    
    model.add(LSTM(units=128, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    
    model.add(Dense(64, activation='relu'))
    

    model.add(Dense(units=1))
    
    
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=[tf.keras.metrics.MeanAbsoluteError(), 'mse'])

    return model


#first layer has 50 units and returns sequences to be passed to next layer
#next layer has 50 units and does not return since it is final

#epoch is one pass through training data set bu algorithm
#one epoch is when you feed model batches and model processes it all from training dataset