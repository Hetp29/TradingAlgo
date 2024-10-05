import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import logging

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
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError(), 'mse'])
    return model

def preprocess_data(df, window_size=60):
    scaler = joblib.load("scaler.pkl")
    scaled_data = scaler.transform(df)
    X, Y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        Y.append(scaled_data[i, 0])
    X, Y = np.array(X), np.array(Y)
    return X, Y, scaler

def data_augmentation(X_train, noise_factor=0.1):
    noise = np.random.normal(loc=0, scale=noise_factor, size=X_train.shape)
    return X_train + noise

def visualize_predictions(Y_true, Y_pred):
    plt.plot(Y_true, label='True')
    plt.plot(Y_pred, label='Predicted')
    plt.legend()
    plt.show()

def log_training_metrics(history):
    metrics_df = pd.DataFrame(history.history)
    metrics_df.to_csv("training_metrics.csv", index=False)

def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logging.basicConfig(filename='logs/training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def learning_rate_schedule(epoch, lr):
    if epoch > 10:
        return lr * np.exp(-0.1)
    return lr

def model_callbacks():
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint("best_lstm_model.h5", monitor='val_loss', save_best_only=True)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule)
    return [reduce_lr, early_stopping, model_checkpoint, lr_scheduler]

def evaluate_model(model, X_val, Y_val):
    predictions = model.predict(X_val)
    mae = mean_absolute_error(Y_val, predictions)
    mse = mean_squared_error(Y_val, predictions)
    return predictions, mae, mse

def plot_training_history(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

def save_scaler(scaler, filename):
    joblib.dump(scaler, filename)

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def grid_search_lstm(X_train, Y_train):
    from keras.wrappers.scikit_learn import KerasRegressor
    from sklearn.model_selection import GridSearchCV
    def create_model(units=64, dropout_rate=0.2):
        model = Sequential()
        model.add(LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units, return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    model = KerasRegressor(build_fn=create_model, epochs=10, batch_size=32, verbose=0)
    param_grid = {'units': [64, 128, 256], 'dropout_rate': [0.2, 0.3], 'batch_size': [32, 64], 'epochs': [10, 50]}
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(X_train, Y_train)
    return grid_result.best_params_

def ensemble_predictions(models, X_val):
    predictions = np.zeros((X_val.shape[0], len(models)))
    for i, model in enumerate(models):
        predictions[:, i] = model.predict(X_val).flatten()
    return np.mean(predictions, axis=1)

def augment_ensemble(model, X_train, Y_train, X_val):
    augmented_X_train = data_augmentation(X_train)
    augmented_Y_train = Y_train
    model.fit(augmented_X_train, augmented_Y_train, epochs=50, batch_size=32, validation_data=(X_val, Y_val), verbose=0)
    return model

def save_model(model, path="best_model.h5"):
    model.save(path)

def train_lstm_model(X_train, Y_train, X_val, Y_val):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_val, Y_val), callbacks=model_callbacks())
    log_training_metrics(history)
    return model

def augment_and_train(X_train, Y_train, X_val, Y_val):
    augmented_X_train = data_augmentation(X_train)
    model = train_lstm_model(augmented_X_train, Y_train, X_val, Y_val)
    return model

def main():
    setup_logging()
    df = pd.read_csv("stock_data.csv")
    X, Y, scaler = preprocess_data(df)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=False)
    save_scaler(scaler, "scaler.pkl")
    best_params = grid_search_lstm(X_train, Y_train)
    model = train_lstm_model(X_train, Y_train, X_val, Y_val)
    save_model(model, "best_lstm_model.h5")
    predictions, mae, mse = evaluate_model(model, X_val, Y_val)
    logging.info(f"MAE: {mae}, MSE: {mse}")
    visualize_predictions(Y_val, predictions)
    plot_training_history(model.history)

if __name__ == "__main__":
    main()
