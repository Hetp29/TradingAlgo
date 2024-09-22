import lstm_model
import prepare_data
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping # type: ignore

if __name__ == "__main__":
    
    X_train, X_val, Y_train, Y_val, scaler = prepare_data.prepare_data("../../data/NVDA_data.csv", features=['Open', 'High', 'Low', 'Adj Close'])
    
    
    model = lstm_model.build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    
    history = model.fit(X_train, Y_train, 
                        epochs=50, 
                        batch_size=32, 
                        validation_data=(X_val, Y_val),
                        callbacks=[reduce_lr, early_stopping])
    

    model.save("enhanced_lstm_stock_model.h5")
    prepare_data.save_scaler(scaler, "scaler.save")
    
    
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
