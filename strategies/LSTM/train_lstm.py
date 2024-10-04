import lstm_model
import prepare_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import logging
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logging.basicConfig(filename='logs/training.log', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')


def log_training_metrics(history):
    metrics_df = pd.DataFrame(history.history)
    metrics_df.to_csv("training_metrics.csv", index=False)
    logging.info("Training metrics saved to training_metrics.csv")


def evaluate_model(model, X_val, Y_val):
    predictions = model.predict(X_val)
    mae = mean_absolute_error(Y_val, predictions)
    rmse = np.sqrt(mean_squared_error(Y_val, predictions))
    r2 = r2_score(Y_val, predictions)
    logging.info(f'Validation MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}')
    return predictions


def additional_evaluation_metrics(Y_val, predictions):
    mape = np.mean(np.abs((Y_val - predictions) / Y_val)) * 100
    logging.info(f'MAPE: {mape:.2f}%')
    print(f'MAPE: {mape:.2f}%')

    plt.figure(figsize=(12, 6))
    plt.hist(predictions - Y_val, bins=50, alpha=0.75, color='blue')
    plt.title('Residual Histogram')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig("residuals_histogram.png")
    plt.close()
    logging.info("Residual histogram saved to residuals_histogram.png")


def visualize_metrics(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy During Training')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
    plt.tight_layout()
    plt.savefig("metrics_plot.png")
    plt.close()
    logging.info("Metrics plot saved to metrics_plot.png")


def save_model(model, model_path):
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")


def load_model(model_path):
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    logging.info(f"Model loaded from {model_path}")
    return model


def data_augmentation(X_train, Y_train, noise_factor=0.1):
    noise = np.random.normal(loc=0, scale=noise_factor, size=X_train.shape)
    augmented_X = X_train + noise
    augmented_Y = Y_train  
    return augmented_X, augmented_Y


def preprocess_data(data_filename, features, noise_factor=0.1):
    X_train, X_val, Y_train, Y_val, scaler = prepare_data.prepare_data(data_filename, features)
    X_train, Y_train = data_augmentation(X_train, Y_train, noise_factor)
    return X_train, X_val, Y_train, Y_val, scaler


def save_scaler(scaler, filename):
    joblib.dump(scaler, filename)
    logging.info(f"Scaler saved to {filename}")


def learning_rate_schedule(epoch, lr):
    if epoch > 10:
        lr = lr * np.exp(0.1)
    return lr


def grid_search_hyperparameters(X_train, Y_train):
    from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

    def create_model(units=32, dropout_rate=0.0, learning_rate=0.001):
        model = lstm_model.build_lstm_model((X_train.shape[1], X_train.shape[2]), 
                                             units=units, 
                                             dropout_rate=dropout_rate)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                      loss='mean_squared_error')
        return model

    model = KerasRegressor(build_fn=create_model, verbose=0)
    param_grid = {
        'units': [32, 64, 128],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [16, 32],
        'epochs': [50, 100]
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', 
                               cv=tscv)
    grid_search.fit(X_train, Y_train)

    logging.info(f"Best Hyperparameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def ensemble_predictions(models, X_val):
    predictions = np.zeros((X_val.shape[0], len(models)))
    for i, model in enumerate(models):
        predictions[:, i] = model.predict(X_val)
    return np.mean(predictions, axis=1)


def main(epochs, batch_size):
    setup_logging()
    logging.info("Starting training process...")

    data_filename = "../../data/NVDA_data.csv"
    features = ['Open', 'High', 'Low', 'Adj Close']
    
    X_train, X_val, Y_train, Y_val, scaler = preprocess_data(data_filename, features)

    logging.info("Performing hyperparameter tuning...")
    best_model = grid_search_hyperparameters(X_train, Y_train)


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint("best_lstm_model.h5", monitor='val_loss', save_best_only=True)
    lr_scheduler = LearningRateScheduler(learning_rate_schedule)

    logging.info("Training the model...")
    history = best_model.fit(X_train, Y_train, 
                              epochs=epochs, 
                              batch_size=batch_size, 
                              validation_data=(X_val, Y_val),
                              callbacks=[reduce_lr, early_stopping, model_checkpoint, lr_scheduler])

    save_model(best_model, "enhanced_lstm_stock_model.h5")
    save_scaler(scaler, "scaler.save")

    log_training_metrics(history)
    predictions = evaluate_model(best_model, X_val, Y_val)
    plot_predictions(Y_val, predictions)
    visualize_metrics(history)
    additional_evaluation_metrics(Y_val, predictions)


    if isinstance(best_model, list):
        ensemble_pred = ensemble_predictions(best_model, X_val)
        logging.info("Ensemble predictions completed.")
        additional_evaluation_metrics(Y_val, ensemble_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an LSTM model for stock price prediction.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    args = parser.parse_args()
    
    main(args.epochs, args.batch_size)
