import numpy as np
import pandas as pd
import prepare_data
import tensorflow as tf
import matplotlib.pyplot as plt

def predict_stock_price(model_filename, data_filename, scaler, window_size=60):
    model = tf.keras.models.load_model(model_filename)
    
    df = pd.read_csv(data_filename)
    
    scaled_data = scaler.transform(df['Adj Close'].values.reshape(-1, 1))
    
    X_test = []
    for i in range(window_size, len(scaled_data)):
        X_test.append(scaled_data[i - window_size: i, 0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predict_stock_price = model.predict(X_test)
    
    predict_stock_price = scaler.inverse_transform(predict_stock_price)
    
    return predict_stock_price

if __name__ == "__main__":
    X_train, Y_train, scaler = prepare_data.prepare_data("../../data/NVDA_data.csv")
    
    predict_stock_price = predict_stock_price("lstm_stock_model.h5", "../../data/NVDA_data.csv", scaler)
    
    df = pd.read_csv("../../data/NVDA_data.csv")
    true_prices = df['Adj Close'].values[60:]
    
    plt.plot(true_prices, color='blue', label='Actual Stock Price')
    plt.plot(predict_stock_price, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction (LSTM)')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()