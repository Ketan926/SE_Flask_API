from flask import Flask, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import datetime

app = Flask(__name__)

# Function to load the Bi-LSTM model
def load_bilstm_model():
    try:
        # Attempt to load the model with custom_objects
        custom_objects = {
            'InputLayer': tf.keras.layers.InputLayer,
            'LSTM': tf.keras.layers.LSTM,
            'Bidirectional': tf.keras.layers.Bidirectional,
            'Dense': tf.keras.layers.Dense
        }
        model = tf.keras.models.load_model('bilstm_model.keras', custom_objects=custom_objects, compile=False)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Fetch and preprocess the Ethereum data
def fetch_and_preprocess_data():
    eth = yf.Ticker("ETH-USD")
    df = eth.history(period="max")  # Fetch the historical data
    df.drop(columns=['Dividends', 'Stock Splits'], inplace=True)

    # Use 'High', 'Low', 'Open', 'Volume' as features and 'Close' as target
    x = df[['High', 'Low', 'Open', 'Volume']].values
    y = df['Close'].values.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
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
    input_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])  # Reshape for LSTM input

    for _ in range(n_days):
        next_day_pred = model.predict(input_sequence)[0]  # Predict the next day
        predictions.append(next_day_pred)

        # Prepare the next input (sliding window)
        next_pred_features = np.array([[next_day_pred[0], input_sequence[0, -1, 1], input_sequence[0, -1, 2], input_sequence[0, -1, 3]]])
        next_pred_scaled = MinMaxScaler().fit_transform(next_pred_features)
        next_pred_scaled = next_pred_scaled.reshape(1, 1, 4)

        input_sequence = np.append(input_sequence[:, 1:, :], next_pred_scaled, axis=1)

    return np.array(predictions)

# Load the Bi-LSTM model
bilstm_model = load_bilstm_model()

if bilstm_model is None:
    print("Failed to load the model. The application may not function correctly.")

# Fetch and preprocess data on API startup
df, X_train, Y_train, scaler = fetch_and_preprocess_data()

# API endpoint to predict the next 7 days and return last 7 days
@app.route('/predict', methods=['GET'])
def predict():
    if bilstm_model is None:
        return jsonify({"error": "Model failed to load. Unable to make predictions."}), 500

    # Get the last sequence of 60 days from the training data for prediction
    last_sequence = X_train[-1]  # Shape: (time_steps, features)
    
    # Predict the next 7 days
    predictions_scaled = predict_next_days(bilstm_model, last_sequence, n_days=7)

    # Inverse transform the predicted values back to the original scale
    predictions_original = scaler.inverse_transform(np.concatenate([predictions_scaled, np.zeros((7, 3))], axis=1))[:, 0]

    # Get the actual 'Close' prices for the last 7 days + today
    last_7_days_actual = df['Close'][-8:].values  # Last 7 days + current day

    # Generate future dates for the predicted prices
    last_date = df.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 8)]

    # Return predictions and actuals in JSON format
    response = {
        'last_7_days_actual': last_7_days_actual.tolist(),  # Last 7 actual days (including current)
        'future_dates': [str(date.date()) for date in future_dates],
        'predicted_prices': predictions_original.tolist()  # Predicted prices for the next 7 days
    }
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run()
