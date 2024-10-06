#####STREAMLIT CODEE#######

import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import date

# Set start and end dates
START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Load a pre-trained model (assuming you've already trained one)
MODEL_PATH = 'my_model.keras'

# Load the LSTM model
model = load_model(MODEL_PATH)

# Streamlit title and description
st.title("Stock Price Prediction App")
st.write("Enter the stock symbol (e.g., AAPL for Apple, TSLA for Tesla) and get predictions.")

# Input field to accept stock symbol from user
ticker = st.text_input("Enter stock symbol:", value='AAPL')

# Button to trigger prediction
if st.button("Predict"):
    # Function to load stock data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    # Load stock data
    st.write(f"Fetching data for {ticker}...")
    data = load_data(ticker)

    # Display raw stock data in a table
    st.subheader(f"Raw Data for {ticker}")
    st.write(data.tail())  # Show last few rows of data

    # Plot the closing price
    st.subheader("Closing Price vs Time")
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data['Close'], label="Closing Price")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(True)
    st.pyplot(fig)

    # Prepare data for the model
    df = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)

    # Prepare input for model prediction (using 10 days lookback as in your code)
    def prepare_input(scaled_data, lookback=10):
        X = []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
        return np.array(X)

    # Prepare the test data
    X_input = prepare_input(scaled_data)
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))

    # Predict the future prices
    st.write("Predicting future prices...")
    predicted_prices = model.predict(X_input)

    # Inverse transform the predictions back to original scale
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot the predictions
    st.subheader("Predicted vs Actual Prices")
    fig2, ax2 = plt.subplots()
    ax2.plot(data['Date'][-len(predicted_prices):], predicted_prices, color='red', label="Predicted Prices")
    ax2.plot(data['Date'], data['Close'], color='blue', label="Actual Prices")
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price (USD)')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)