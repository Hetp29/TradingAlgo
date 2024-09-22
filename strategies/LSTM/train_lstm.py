import lstm_model
import prepare_data
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X_train, Y_train, scaler = prepare_data.prepare_data("../../data/NVDA_data.csv")
    
    model = lstm_model.build_lstm_model((X_train.shape[1], 1))
    
    history = model.fit(X_train, Y_train, epochs=20, batch_size=32)
    
    model.save("lstm_stock_model.h5") #saving trained model
    
    plt.plot(history.history['loss'])
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show