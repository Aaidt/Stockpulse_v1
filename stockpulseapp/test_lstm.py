import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta.momentum import RSIIndicator
from newsapi import NewsApiClient
from datetime import date, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import plotly.graph_objects as go
import nltk
nltk.download('vader_lexicon')

# Initialize News API Client (Replace with your actual API Key)
newsapi = NewsApiClient(api_key='ee93de27f75f49dd93997f391a741e7b')

# Set start and end dates
START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Streamlit title and description
st.title("STOCK PREDICTION")
st.write("Enter the stock symbol (e.g., AAPL for Apple, TSLA for Tesla) and get predictions, along with sentiment analysis and technical indicators.")

# Input field to accept stock symbol from user
ticker = st.text_input("Enter stock symbol:", value='AAPL')

# Input for number of days to predict into the future
days_ahead = st.number_input("Enter the number of days to predict:", min_value=1, max_value=365, value=30)

# Function to load stock data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Function to calculate sentiment from news
def get_sentiment(ticker):
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    articles = newsapi.get_everything(q=ticker, from_param=start_date, to=end_date, language='en')

    if articles['totalResults'] == 0:
        return 0  # Neutral sentiment if no articles

    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    sentiment_score = 0

    for article in articles['articles']:
        title = article['title']
        description = article['description'] or ""
        combined_text = title + " " + description

        # Get the sentiment scores
        score = sia.polarity_scores(combined_text)
        sentiment_score += score['compound']  # Use the compound score

    # Calculate average sentiment score
    average_sentiment = sentiment_score / articles['totalResults']

    # Return sentiment as -1, 0, or 1
    if average_sentiment > 0.05:
        return 1  # Positive sentiment
    elif average_sentiment < -0.05:
        return -1  # Negative sentiment
    else:
        return 0  # Neutral sentiment

# Button to trigger prediction
if st.button("Predict"):
    st.write(f"Fetching data for {ticker}...")
    
    # Load stock data
    data = load_data(ticker)

    # Calculate moving averages and RSI
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA30'] = data['Close'].rolling(window=30).mean()
    rsi = RSIIndicator(data['Close'])
    data['RSI'] = rsi.rsi()

    # Fill missing values
    data.fillna(method='bfill', inplace=True)

    # Add sentiment analysis as a feature
    sentiment = get_sentiment(ticker)
    data['Sentiment'] = sentiment  # Use the sentiment from news analysis

    # Display the updated data, including volume and sentiment
    st.subheader("Raw Data with Indicators")
    st.write(data[['Date', 'Close', 'Volume', 'MA10', 'MA30', 'RSI', 'Sentiment']].tail())

    # Prepare the data for the model
    features = ['Close', 'Volume', 'MA10', 'MA30', 'RSI', 'Sentiment']
    df = data[features]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Prepare input for model prediction (using a 10-day lookback)
    def prepare_input(data, lookback=10):
        X = []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])  # Append the last 'lookback' days of data
        return np.array(X)

    # Prepare the test data
    X_input = prepare_input(scaled_data)
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], X_input.shape[2]))  # Ensure 3D shape

    # Build a new LSTM model (with adjusted input dimensions)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_input.shape[1], X_input.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer for predicting stock prices

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model (for demonstration, we use a small number of epochs)
    model.fit(X_input, scaled_data[10:, 0], epochs=5, batch_size=32)  # Train on 'Close' prices (1st column)

    # Predict the future prices for the given number of days
    st.write(f"Predicting stock prices for the next {days_ahead} days...")

    future_predictions = []
    last_input = X_input[-1]  # Last input to base future predictions on
    last_input = last_input.reshape(1, last_input.shape[0], last_input.shape[1])  # Reshape for prediction

    for _ in range(days_ahead):
        prediction = model.predict(last_input)
        future_predictions.append(prediction[0][0])

        # Prepare the new input based on the previous last_input and add the predicted price
        new_features = last_input[0][-1].copy()
        new_features[0] = prediction[0][0]  # Replace the Close price with the predicted price

        # Append the new input for prediction with the previous features
        new_input = np.append(last_input[0][1:], [new_features], axis=0)

        # Reshape new_input to be 3D
        last_input = new_input.reshape(1, new_input.shape[0], new_input.shape[1])

    # Convert future predictions to a DataFrame for inverse transformation
    last_known_features = scaled_data[-1].copy()  # Get the last known features from scaled_data
    future_predictions_full = []

    for predicted_price in future_predictions:
        new_entry = last_known_features.copy()
        new_entry[0] = predicted_price  # Set the 'Close' price
        future_predictions_full.append(new_entry)

    future_predictions_full = np.array(future_predictions_full)

    # Inverse transform the predictions back to original scale
    future_predictions_full = scaler.inverse_transform(future_predictions_full)

    # Create DataFrame for future predictions
    future_dates = pd.date_range(start=TODAY, periods=days_ahead)
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions_full[:, 0]})

    # NEW SECTION: Checking model accuracy using the past 30 days for comparison
    # Number of days for testing (past 30 days)
    test_days = 30  # You can also make this dynamic based on user input

    # Extract the test data (last 30 days for comparison)
    test_data = data[-test_days:]  # Last 'test_days' worth of data
    test_dates = test_data['Date'].values  # Get the dates for the last 'test_days'

    # Extract actual closing prices
    actual_prices = test_data['Close'].values

    # Prepare the input for testing on the last 30 days
    test_scaled_data = scaled_data[-test_days:]  # Last 'test_days' of scaled data
    X_test = prepare_input(test_scaled_data)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # Predict on the last 30 days
    test_predictions = model.predict(X_test)
    test_predictions_full = []

    # Inverse transform the test predictions back to original scale
    for prediction in test_predictions:
        new_entry = last_known_features.copy()
        new_entry[0] = prediction[0]
        test_predictions_full.append(new_entry)

    test_predictions_full = np.array(test_predictions_full)
    test_predictions_full = scaler.inverse_transform(test_predictions_full)

    # Create a DataFrame to compare actual vs predicted
    predicted_vs_actual = pd.DataFrame({
        'Date': test_dates[-len(test_predictions):],  # Trim the dates to match the length of predictions
        'Predicted Price': test_predictions_full[:, 0],  # Predictions array
        'Actual Price': actual_prices[-len(test_predictions):]  # Trim the actual prices to match
    })

    # Display comparison
    st.write("Predicted vs Actual Prices (Past 30 Days)")
    st.write(predicted_vs_actual)

    # Plot the predicted vs actual prices for the last 30 days
    # fig_compare = go.Figure()
    # fig_compare.add_trace(go.Scatter(x=predicted_vs_actual['Date'], y=predicted_vs_actual['Actual Price'],
    #                                  mode='lines', name='Actual Price'))
    # fig_compare.add_trace(go.Scatter(x=predicted_vs_actual['Date'], y=predicted_vs_actual['Predicted Price'],
    #                                  mode='lines', name='Predicted Price'))

    # st.plotly_chart(fig_compare)
    
    
        # Plot the predicted vs actual prices for the last 30 days
    fig_compare = go.Figure()

    # Change the color of the 'Actual Price' line (e.g., blue)
    fig_compare.add_trace(go.Scatter(
        x=predicted_vs_actual['Date'], 
        y=predicted_vs_actual['Actual Price'],
        mode='lines', 
        name='Actual Price',
        line=dict(color='blue')  # Set the line color for actual prices
    ))

    # Change the color of the 'Predicted Price' line (e.g., red)
    fig_compare.add_trace(go.Scatter(
        x=predicted_vs_actual['Date'], 
        y=predicted_vs_actual['Predicted Price'],
        mode='lines', 
        name='Predicted Price',
        line=dict(color='red')  # Set the line color for predicted prices
    ))

    st.plotly_chart(fig_compare)


    # # Display the future predictions
    # st.write("Predicted Stock Prices for the Next 30 Days")
    # st.write(future_df)

    # # Plot future predictions
    # fig_future = go.Figure()
    # fig_future.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted Price'], mode='lines', name='Predicted Price'))
    # st.plotly_chart(fig_future)
