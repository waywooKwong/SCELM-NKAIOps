import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Generate sample data
def generate_data(length):
    timestamp = np.arange(length)
    value = np.sin(timestamp * 0.1) + np.random.normal(0, 0.1, length)
    return pd.DataFrame({'timestamp': timestamp, 'value': value})

# Prepare data for LSTM
def prepare_data(df, window_size):
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df['value'].iloc[i:i+window_size].values)
        y.append(df['value'].iloc[i+window_size])
    return np.array(X), np.array(y)

# Build LSTM model
def build_model(window_size):
    model = Sequential()
    model.add(LSTM(50, input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Train LSTM model
def train_model(model, X_train, y_train, epochs, batch_size):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

# Evaluate model on test data
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

# Find best matching window size
def find_best_window_size(df):
    min_window_size = 1
    max_window_size = len(df) // 2
    best_window_size = None
    best_mse = float('inf')

    for window_size in range(min_window_size, max_window_size + 1, 100):#设置步长
        X, y = prepare_data(df, window_size)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = build_model(window_size)
        train_model(model, X, y, epochs=50, batch_size=32)

        mse = evaluate_model(model, X, y)
        if mse < best_mse:
            best_mse = mse
            best_window_size = window_size

    return best_window_size

# Main function
def main():
    # Generate sample data
    # data_length = 1000
    # df = generate_data(data_length)
    df = pd.read_csv('/home/sunyongqian/liuheng/Time-LLM/weihua/all-zero.csv') 

    # Find best window size
    best_window_size = find_best_window_size(df)
    print("Best window size:", best_window_size)

if __name__ == "__main__":
    main()
