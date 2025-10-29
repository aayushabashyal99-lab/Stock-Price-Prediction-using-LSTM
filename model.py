import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def fetch_data(symbol, start_date="2020-01-01", end_date="2024-08-01"):
    # Download historical close price data from Yahoo Finance
    data = yf.download(symbol, start=start_date, end=end_date)
    return data['Close'].values.reshape(-1, 1)

def prepare_data(prices, look_back=60):
    # Normalize prices between 0 and 1
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_model(input_shape):
    # Define stacked LSTM architecture
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_predict(symbol):
    prices = fetch_data(symbol)
    if len(prices) < 61:
        raise ValueError("Not enough data — try another symbol or a longer time range.")
    X, y, scaler = prepare_data(prices)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build and train the model
    model = build_model((X.shape[1], 1))
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # Predict next day based on last 60 days
    last_sequence = X[-1].reshape(1, X.shape[1], 1)
    predicted_scaled = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(predicted_scaled)
    return float(predicted_price[0][0])
