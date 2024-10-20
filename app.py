from flask import Flask, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import datetime

app = Flask(_name_)

# Load the pre-trained Bi-LSTM model
bilstm_model = tf.keras.models.load_model('bilstm_model.keras')

# Fetch and preprocess the Ethereum data
def fetch_and_preprocess_data():
    eth = yf.Ticker("ETH-USD")
    df = eth.history(period="max")  # Fetch the historical data
    df.drop(columns=['Dividends', 'Stock Splits'], inplace=True)

    # Use 'High', 'Low', 'Open', 'Volume' as features and 'Close' as target
    x = df[['High', 'Low', 'Open', 'Volume']].values
    y = df['Close'].values.reshape(-1, 1)

    # Separate scalers for features and 'Close'
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale features and target
    x_scaled = scaler.fit_transform(x)
    y_scaled = scaler.fit_transform(y)

    # Create time series data with 60 time steps
    time_steps = 60
    X, Y = create_time_series_data(x_scaled, y_scaled, time_steps)
    
    return df, X, Y, scaler

# Function to create time series data for LSTM
def create_time_series_data(x, y, time_steps=60):
    X, Y = [], []
    for i in range(len(x) - time_steps):
        X.append(x[i:i + time_steps])
        Y.append(y[i + time_steps])
    return np.array(X), np.array(Y)

# Function to predict the next 7 days using the trained Bi-LSTM model
def predict_next_days(model, last_sequence, n_days=7):
    predictions = []
    input_sequence = last_sequence.reshape(1, 60, last_sequence.shape[1])  # Reshape to 3D: (1, time_steps, features)

    for _ in range(n_days):
        # Predict the next day's close price
        next_day_pred = model.predict(input_sequence)[0]  # Prediction for the next day (single value)
        # Append the predicted close price
        predictions.append(next_day_pred)

        # Prepare the new input sequence (sliding window):
        # Remove the oldest time step and add the new prediction for the close price
        next_day_pred = np.array([[next_day_pred[0], input_sequence[0, -1, 1], input_sequence[0, -1, 2], input_sequence[0, -1, 3]]])
        input_sequence = np.append(input_sequence[:, 1:, :], next_day_pred[np.newaxis, :, :], axis=1)  # Shift the window

    return np.array(predictions)

# Fetch and preprocess data on API startup
df, X_train, Y_train, scaler = fetch_and_preprocess_data()

# API endpoint to predict the next 7 days and return last 7 days
@app.route('/', methods=['GET'])
def predict():
    # Get the last sequence of 60 days from the training data for prediction
    last_sequence = X_train[-1]  # Shape: (time_steps, features)
    
    # Predict the next 7 days
    predictions_scaled = predict_next_days(bilstm_model, last_sequence, n_days=7)

    # Inverse transform the predictions to get the original scale
    predictions_original = scaler.inverse_transform(np.concatenate([predictions_scaled, np.zeros((7, 3))], axis=1))[:, 0]

    # Get the actual 'Close' prices for the last 7 days + today
    last_7_days_actual = df['Close'][-8:].values  # Last 7 days + current day

    # Generate future dates for the predicted prices
    last_date = df.index[-1]
    past_dates = [last_date - datetime.timedelta(days=i) for i in range(7, -1, -1)]
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 8)]
    closing_prices = last_7_days_actual.tolist() + predictions_original.flatten().tolist()
    combined_dates = [f"{date.day:02d}-{date.month:02d}-{date.year}" for date in past_dates] + [f"{date.day:02d}-{date.month:02d}-{date.year}" for date in future_dates]

    # Return predictions and actuals in JSON format
    response = {
        'combined_dates': combined_dates,
        'future_dates': [f"{date.day:02d}-{date.month:02d}-{date.year}" for date in future_dates],
        'past_dates': [f"{date.day:02d}-{date.month:02d}-{date.year}" for date in past_dates],
        'last_7_days_actual': last_7_days_actual.tolist(),  # Last 7 actual days (including current)
        'predicted_prices': predictions_original.flatten().tolist(),  # Predicted prices for the next 7 days
        'closing_prices': closing_prices
    }
    return jsonify(response)
