import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

def predict_stock_price(model_filename, data_filename, scaler_filename, window_size=60, features=['Adj Close']):
    # Load the trained model and scaler
    model = tf.keras.models.load_model(model_filename)
    scaler = joblib.load(scaler_filename)
    
    df = pd.read_csv(data_filename)
    
    
    scaled_data = scaler.transform(df[features].values)
    
    X_test = []
    for i in range(window_size, len(scaled_data)):
        X_test.append(scaled_data[i - window_size:i])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(features)))
    
    predict_stock_price = model.predict(X_test)
    

    scaled_predictions = np.zeros((len(predict_stock_price), len(features)))
    scaled_predictions[:, 0] = predict_stock_price[:, 0]
    predict_stock_price = scaler.inverse_transform(scaled_predictions)[:, 0]
    
    return predict_stock_price

if __name__ == "__main__":
    
    predict_stock_price = predict_stock_price("enhanced_lstm_stock_model.h5", 
                                              "../../data/NVDA_data.csv", 
                                              "scaler.save", 
                                              window_size=60, 
                                              features=['Open', 'High', 'Low', 'Adj Close'])
    
    
    df = pd.read_csv("../../data/NVDA_data.csv")
    true_prices = df['Adj Close'].values[60:]
    
    
    plt.plot(true_prices, color='blue', label='Actual Stock Price')
    plt.plot(predict_stock_price, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction (LSTM)')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
