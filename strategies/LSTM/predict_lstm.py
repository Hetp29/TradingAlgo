import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from statsmodels.tsa.seasonal import seasonal_decompose

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_confidence_interval(predictions, confidence=0.95):
    mean_prediction = np.mean(predictions)
    stderr = np.std(predictions) / np.sqrt(len(predictions))
    margin_of_error = stderr * 1.96
    lower_bound = mean_prediction - margin_of_error
    upper_bound = mean_prediction + margin_of_error
    return lower_bound, upper_bound

def calculate_exponential_moving_average(df, span=20):
    return df['Adj Close'].ewm(span=span, adjust=False).mean()

def calculate_moving_average(df, window=20):
    return df['Adj Close'].rolling(window=window).mean()

def explore_data(data_filename):
    df = pd.read_csv(data_filename)
    print("Data Overview:")
    print(df.describe())
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Date', y='Adj Close')
    plt.title("Stock Price Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.xticks(rotation=45)
    plt.show()
    
    decomposition = seasonal_decompose(df['Adj Close'], model='additive', period=30)
    decomposition.plot()
    plt.show()

    df['EMA'] = calculate_exponential_moving_average(df)
    df['MA'] = calculate_moving_average(df)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Adj Close'], label='Actual Price', color='blue')
    plt.plot(df['Date'], df['EMA'], label='20-Day EMA', color='orange')
    plt.plot(df['Date'], df['MA'], label='20-Day MA', color='green')
    plt.title("Actual Price vs Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def plot_error_metrics(true_prices, predicted_prices):
    mae = mean_absolute_error(true_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(true_prices, predicted_prices))
    r2 = r2_score(true_prices, predicted_prices)
    mape = mean_absolute_percentage_error(true_prices, predicted_prices)

    plt.figure(figsize=(10, 10))
    plt.subplot(3, 1, 1)
    plt.plot(true_prices, label='Actual Prices', color='blue')
    plt.plot(predicted_prices, label='Predicted Prices', color='red')
    plt.title('Stock Price Prediction (LSTM)')
    plt.ylabel('Price')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    metrics = ['MAE', 'RMSE', 'R2', 'MAPE']
    values = [mae, rmse, r2, mape]
    plt.bar(metrics, values, color=['orange', 'green', 'blue', 'purple'])
    plt.title('Error Metrics')
    plt.ylabel('Error Value / %')
    
    plt.subplot(3, 1, 3)
    residuals = true_prices - predicted_prices
    plt.scatter(predicted_prices, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals Plot')
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals')
    
    plt.tight_layout()
    plt.show()

def save_predictions_to_csv(true_prices, predicted_prices, filename='predicted_vs_actual.csv'):
    results = pd.DataFrame({'True Prices': true_prices, 'Predicted Prices': predicted_prices})
    results.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

def predict_stock_price(model_filename, data_filename, scaler_filename, window_size=60, features=['Adj Close']):
    model = tf.keras.models.load_model(model_filename)
    scaler = joblib.load(scaler_filename)
    df = pd.read_csv(data_filename)
    scaled_data = scaler.transform(df[features].values)

    X_test = []
    for i in range(window_size, len(scaled_data)):
        X_test.append(scaled_data[i - window_size:i])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(features)))

    predicted_prices = model.predict(X_test)
    scaled_predictions = np.zeros((len(predicted_prices), len(features)))
    scaled_predictions[:, 0] = predicted_prices[:, 0]
    predicted_prices = scaler.inverse_transform(scaled_predictions)[:, 0]
    
    return predicted_prices

def log_performance(true_prices, predicted_prices, log_filename="performance_log.txt"):
    mae = mean_absolute_error(true_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(true_prices, predicted_prices))
    r2 = r2_score(true_prices, predicted_prices)
    mape = mean_absolute_percentage_error(true_prices, predicted_prices)

    with open(log_filename, "a") as log_file:
        log_file.write(f"\nMAE: {mae:.4f}\n")
        log_file.write(f"RMSE: {rmse:.4f}\n")
        log_file.write(f"R2: {r2:.4f}\n")
        log_file.write(f"MAPE: {mape:.2f}%\n")
    
    print(f"Performance metrics logged in {log_filename}")

def preprocess_data_with_nan_handling(data_filename, features=['Adj Close'], window_size=60):
    df = pd.read_csv(data_filename)
    df[features] = df[features].fillna(df[features].mean())
    
    for feature in features:
        df[f'{feature}_pct_change'] = df[feature].pct_change()
        df[f'{feature}_lag1'] = df[feature].shift(1)
        df[f'{feature}_lag2'] = df[feature].shift(2)
    
    df.dropna(inplace=True)
    return df

def tune_hyperparameters(model, X_train, y_train):
    param_grid = {
        'units': [32, 64, 128],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [16, 32, 64]
    }
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Best Hyperparameters: {best_params}")

def calculate_additional_metrics(true_prices, predicted_prices):
    mean_squared_error_value = mean_squared_error(true_prices, predicted_prices)
    mean_absolute_error_value = mean_absolute_error(true_prices, predicted_prices)
    r2 = r2_score(true_prices, predicted_prices)

    print(f"Mean Squared Error: {mean_squared_error_value:.4f}")
    print(f"Mean Absolute Error: {mean_absolute_error_value:.4f}")
    print(f"R^2 Score: {r2:.4f}")

def run_predictions_and_visualizations():
    model_filename = "enhanced_lstm_stock_model.h5"
    data_filename = "../../data/NVDA_data.csv"
    scaler_filename = "scaler.save"

    df = preprocess_data_with_nan_handling(data_filename, features=['Open', 'High', 'Low', 'Adj Close'])
    explore_data(data_filename)
    
    predicted_prices = predict_stock_price(model_filename, data_filename, scaler_filename, window_size=60, 
                                           features=['Open', 'High', 'Low', 'Adj Close'])

    true_prices = df['Adj Close'].values[60:]
    
    plt.figure(figsize=(12, 6))
    plt.plot(true_prices, color='blue', label='Actual Stock Price')
    plt.plot(predicted_prices, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction (LSTM)')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    
    plot_error_metrics(true_prices, predicted_prices)
    
    lower_bound, upper_bound = calculate_confidence_interval(predicted_prices)
    print(f"95% Confidence Interval for Predicted Prices: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    calculate_additional_metrics(true_prices, predicted_prices)
    
    save_predictions_to_csv(true_prices, predicted_prices)
    log_performance(true_prices, predicted_prices)

if __name__ == "__main__":
    run_predictions_and_visualizations()
