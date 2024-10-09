# import pandas as pd
# import datetime as dt
# from datetime import date
# import matplotlib.pyplot as plt
# import yfinance as yf
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.layers import Dense, Dropout, LSTM
# from tensorflow.keras.models import Sequential
# from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    
    
    #OG CODE
    
    # # Set start and end dates
    # START = "2010-01-01"
    # TODAY = date.today().strftime("%Y-%m-%d")

    # # Function to load the dataset
    # def load_data(ticker):
    #     data = yf.download(ticker, START, TODAY)
    #     data.reset_index(inplace=True)
    #     return data

    # # Load data
    # data = load_data('AAPL')
    # df = data.copy()

    # # Drop unnecessary columns if they exist
    # df = df.drop(columns=['Date', 'Adj Close'], errors='ignore')

    # # Plot the closing price
    # plt.figure(figsize=(12, 6))
    # plt.plot(df['Close'])
    # plt.title("Apple Stock Price")
    # plt.xlabel("Date")
    # plt.ylabel("Price (USD)")
    # plt.grid(True)
    # plt.show()

    # # Calculate and plot 100-day moving average
    # ma10 = df['Close'].rolling(window=10).mean()
    # plt.figure(figsize=(12, 6))
    # plt.plot(df['Close'], label='Close Price')
    # plt.plot(ma10, 'r', label='10-Day MA')
    # plt.title('10-Day Moving Average')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # # Calculate and plot 100-day and 200-day moving averages
    # ma30 = df['Close'].rolling(window=30).mean()
    # plt.figure(figsize=(12, 6))
    # plt.plot(df['Close'], label='Close Price')
    # plt.plot(ma10, 'r', label='10-Day MA')
    # plt.plot(ma30, 'g', label='30-Day MA')
    # plt.title('Comparison of 10-Day and 30-Day Moving Averages')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # # Splitting the data into training and testing sets
    # train_size = int(len(df) * 0.70)
    # train, test = df[:train_size], df[train_size:]

    # # Normalize the data using MinMaxScaler
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # train_close = train[['Close']].values
    # test_close = test[['Close']].values

    # train_scaled = scaler.fit_transform(train_close)
    # test_scaled = scaler.transform(test_close)

    # # Prepare the training data
    # x_train, y_train = [], []
    # for i in range(10, len(train_scaled)):
    #     x_train.append(train_scaled[i-10:i])
    #     y_train.append(train_scaled[i, 0])

    # x_train, y_train = np.array(x_train), np.array(y_train)

    # # Build the LSTM model
    # model = Sequential()
    # model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=60, activation='relu', return_sequences=True))
    # model.add(Dropout(0.3))
    # model.add(LSTM(units=90, activation='relu', return_sequences=True))
    # model.add(Dropout(0.4))
    # model.add(LSTM(units=120, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(units=1))

    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    # model.summary()

    # # Train the model
    # model.fit(x_train, y_train, epochs=10)

    # # Save the model
    # model.save('my_model.keras')

    # # Prepare the testing data
    # past_100_days = train_close[-100:]
    # final_df = np.concatenate((past_100_days, test_close), axis=0)

    # input_data = scaler.transform(final_df)

    # x_test, y_test = [], []
    # for i in range(10, len(input_data)):
    #     x_test.append(input_data[i-10:i])
    #     y_test.append(input_data[i, 0])

    # x_test, y_test = np.array(x_test), np.array(y_test)

    # # Make predictions
    # y_pred = model.predict(x_test)

    # # Inverse transform the predictions and actual values
    # y_pred = scaler.inverse_transform(y_pred)
    # y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # # Plot the actual vs predicted prices
    # plt.figure(figsize=(12, 6))
    # plt.plot(y_test, 'b', label="Original Price")
    # plt.plot(y_pred, 'r', label="Predicted Price")
    # plt.xlabel('Time')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # Calculate and print the Mean Absolute Error
    # mae = mean_absolute_error(y_test, y_pred)
    # mae_percentage = (mae / np.mean(y_test)) * 50
    # print("Mean absolute error on test set: {:.2f}%".format(mae_percentage))




    # FIRST MOD

    # # stockpulseapp/lstm.py
    # import pandas as pd
    # import yfinance as yf
    # import numpy as np
    # from sklearn.preprocessing import MinMaxScaler
    # from tensorflow.keras.models import load_model
    # import matplotlib.pyplot as plt

    # # Load the pre-trained model (Ensure you have saved it beforehand)
    # model = load_model('my_model.keras')

    # def load_data(ticker):
    #     START = "2010-01-01"
    #     TODAY = date.today().strftime("%Y-%m-%d")
    #     data = yf.download(ticker, START, TODAY)
    #     data.reset_index(inplace=True)
    #     return data

    # def make_predictions(ticker):
    #     data = load_data(ticker)
    #     df = data[['Close']]

    #     # Preprocess data as in the training process
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     close_data = df[['Close']].values
    #     close_scaled = scaler.fit_transform(close_data)

    #     # Prepare input data for prediction
    #     x_input = []
    #     for i in range(10, len(close_scaled)):
    #         x_input.append(close_scaled[i-10:i])
        
    #     x_input = np.array(x_input)
        
    #     # Make predictions using the pre-trained model
    #     predicted_scaled = model.predict(x_input)

    #     # Inverse transform the predictions
    #     predicted = scaler.inverse_transform(predicted_scaled)

    #     # Return the actual vs predicted values (last 100)
    #     return {
    #         'actual': close_data[-100:].flatten(),
    #         'predicted': predicted.flatten()
    #     }

    # import pandas as pd
    # import yfinance as yf
    # import numpy as np
    # from sklearn.preprocessing import MinMaxScaler
    # from tensorflow.keras.models import Sequential, load_model
    # from tensorflow.keras.layers import LSTM, Dropout, Dense
    # import tensorflow as tf
    # from datetime import date
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import mean_absolute_error
    # # from .forms import StockForm
    # import forms
    
    # # Set start and end dates for data
    # START = "2010-01-01"
    # TODAY = date.today().strftime("%Y-%m-%d")
    # stocksymbol = forms.stock_symbol
    # # Function to load the dataset
    # def load_data(stocksymbol):
    #     data = yf.download(stocksymbol, START, TODAY)
    #     data.reset_index(inplace=True)
    #     return data

    # # Load the data
    # # ticker = 'AAPL'
    # data = load_data(forms.stock_symbol)
    # df = data.copy()

    # # Drop unnecessary columns if they exist
    # df = df.drop(columns=['Date', 'Adj Close'], errors='ignore')

    # # Plot the closing price
    # plt.figure(figsize=(12, 6))
    # plt.plot(df['Close'])
    # plt.title(f"{forms.stock_symbol} Stock Price")
    # plt.xlabel("Date")
    # plt.ylabel("Price (USD)")
    # plt.grid(True)
    # plt.show()

    # # Calculate and plot 10-day moving average
    # ma10 = df['Close'].rolling(window=10).mean()
    # plt.figure(figsize=(12, 6))
    # plt.plot(df['Close'], label='Close Price')
    # plt.plot(ma10, 'r', label='10-Day MA')
    # plt.title('10-Day Moving Average')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # # Calculate and plot 10-day and 30-day moving averages
    # ma30 = df['Close'].rolling(window=30).mean()
    # plt.figure(figsize=(12, 6))
    # plt.plot(df['Close'], label='Close Price')
    # plt.plot(ma10, 'r', label='10-Day MA')
    # plt.plot(ma30, 'g', label='30-Day MA')
    # plt.title('Comparison of 10-Day and 30-Day Moving Averages')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # # Split the data into training and testing sets
    # train_size = int(len(df) * 0.70)
    # train, test = df[:train_size], df[train_size:]

    # # Normalize the data using MinMaxScaler
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # train_close = train[['Close']].values
    # test_close = test[['Close']].values

    # train_scaled = scaler.fit_transform(train_close)
    # test_scaled = scaler.transform(test_close)

    # # Prepare the training data
    # x_train, y_train = [], []
    # for i in range(10, len(train_scaled)):
    #     x_train.append(train_scaled[i-10:i])
    #     y_train.append(train_scaled[i, 0])

    # x_train, y_train = np.array(x_train), np.array(y_train)

    # # Build the LSTM model
    # model = Sequential()
    # model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=60, activation='relu', return_sequences=True))
    # model.add(Dropout(0.3))
    # model.add(LSTM(units=90, activation='relu', return_sequences=True))
    # model.add(Dropout(0.4))
    # model.add(LSTM(units=120, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(units=1))

    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    # model.summary()

    # # Train the model
    # model.fit(x_train, y_train, epochs=10)

    # # Save the trained model
    # model.save('my_model.keras')

    # # Function to make predictions using the saved model
    # def make_predictions(stock_symbol):
    #     # Load the pre-trained model
    #     model = load_model('my_model.keras')

    #     # Reload the stock data for the selected ticker
    #     data = load_data(stock_symbol)
    #     df = data[['Close']]

    #     # Preprocess the data using the same scaler
    #     close_data = df[['Close']].values
    #     close_scaled = scaler.transform(close_data)

    #     # Prepare input data for prediction (last 100 days of data)
    #     x_input = []
    #     for i in range(10, len(close_scaled)):
    #         x_input.append(close_scaled[i-10:i])

    #     x_input = np.array(x_input)

    #     # Make predictions
    #     predicted_scaled = model.predict(x_input)

    #     # Inverse transform the predictions
    #     predicted = scaler.inverse_transform(predicted_scaled)

    #     # Return the last 100 actual and predicted values
    #     return {
    #         'actual': close_data[-100:].flatten(),
    #         'predicted': predicted.flatten()
    #     }

    # # Make predictions using the trained model
    # predictions = make_predictions(forms.stock_symbol)

    # # Plot the actual vs predicted prices
    # plt.figure(figsize=(12, 6))
    # plt.plot(predictions['actual'], 'b', label="Original Price (Actual)")
    # plt.plot(predictions['predicted'], 'r', label="Predicted Price")
    # plt.title(f"Actual vs Predicted Prices for {forms.stock_symbol}")
    # plt.xlabel('Days')
    # plt.ylabel('Price (USD)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # Calculate the mean absolute error
    # mae = mean_absolute_error(predictions['actual'], predictions['predicted'])
    # mae_percentage = (mae / np.mean(predictions['actual'])) * 100
    # print(f"Mean absolute error on test set: {mae_percentage:.2f}%")

    
    ####STREAMLIT CODE#####
    # import streamlit as st
    # import pandas as pd
    # import yfinance as yf
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import tensorflow as tf
    # from sklearn.preprocessing import MinMaxScaler
    # from tensorflow.keras.models import load_model
    # from datetime import date, timedelta

    # # Set start and end dates
    # START = "2010-01-01"
    # TODAY = date.today().strftime("%Y-%m-%d")

    # # Load a pre-trained model (assuming you've already trained one)
    # MODEL_PATH = 'my_model.keras'

    # # Load the LSTM model
    # model = load_model(MODEL_PATH)

    # # Streamlit title and description
    # st.title("Stock Price Prediction App")
    # st.write("Enter the stock symbol (e.g., AAPL for Apple, TSLA for Tesla) and get predictions.")

    # # Input field to accept stock symbol from user
    # ticker = st.text_input("Enter stock symbol:", value='AAPL')

    # # Input field to enter number of days to predict into the future
    # n_days = st.number_input("Enter number of days to predict into the future:", min_value=1, value=10)

    # # Button to trigger prediction
    # if st.button("Predict"):
    #     # Function to load stock data
    #     def load_data(ticker):
    #         data = yf.download(ticker, START, TODAY)
    #         data.reset_index(inplace=True)
    #         return data

    #     # Load stock data
    #     st.write(f"Fetching data for {ticker}...")
    #     data = load_data(ticker)

    #     # Display raw stock data in a table
    #     st.subheader(f"Raw Data for {ticker}")
    #     st.write(data.tail())  # Show last few rows of data

    #     # Plot the closing price
    #     st.subheader("Closing Price vs Time")
    #     fig, ax = plt.subplots()
    #     ax.plot(data['Date'], data['Close'], label="Closing Price")
    #     ax.set_xlabel('Date')
    #     ax.set_ylabel('Price (USD)')
    #     ax.grid(True)
    #     st.pyplot(fig)

    #     # Prepare data for the model
    #     df = data[['Close']]
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     scaled_data = scaler.fit_transform(df.values)

    #     # Prepare input for model prediction (using 10 days lookback as in your code)
    #     def prepare_input(scaled_data, lookback=10):
    #         X = []
    #         for i in range(lookback, len(scaled_data)):
    #             X.append(scaled_data[i-lookback:i, 0])
    #         return np.array(X)

    #     # Prepare the test data
    #     X_input = prepare_input(scaled_data)
    #     X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))

    #     # Predict the future prices
    #     st.write("Predicting future prices...")
    #     predicted_prices = model.predict(X_input)

    #     # Inverse transform the predictions back to original scale
    #     predicted_prices = scaler.inverse_transform(predicted_prices)

    #     # Prepare future predictions (based on the last known input)
    #     future_predictions = []
    #     last_input = X_input[-1]  # Last 10 days of scaled data

    #     for i in range(n_days):
    #         next_prediction = model.predict(last_input.reshape(1, -1, 1))
    #         future_predictions.append(next_prediction[0, 0])

    #         # Append the predicted value and remove the oldest value (to simulate a rolling window)
    #         last_input = np.append(last_input[1:], next_prediction)

    #     # Inverse transform future predictions
    #     future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    #     # Create future dates for plotting
    #     future_dates = [data['Date'].max() + timedelta(days=i+1) for i in range(n_days)]

    #     # Plot the predictions
    #     st.subheader("Predicted vs Actual Prices")
    #     fig2, ax2 = plt.subplots()
    #     ax2.plot(data['Date'][-len(predicted_prices):], predicted_prices, color='red', label="Predicted Prices")
    #     ax2.plot(data['Date'], data['Close'], color='blue', label="Actual Prices")

    #     # Plot future predictions
    #     ax2.plot(future_dates, future_predictions, color='green', label="Future Predictions")
    #     ax2.set_xlabel('Date')
    #     ax2.set_ylabel('Price (USD)')
    #     ax2.legend()
    #     ax2.grid(True)
    #     st.pyplot(fig2)

    #     # Show future predictions in a table
    #     future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions.flatten()})
    #     st.subheader(f"Future Predictions for {ticker}")
    #     st.write(future_df)



    #####STREAMLIT CODE WITH SENTIMENT ANALYSIS, TADING VOLUMES#######
    # 

    ###### UPDATED CODE FOR INPUT VALUES #####
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
    st.title("STOCK PREDICTION ")
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

        #  # Improved plot for actual closing price along with moving averages
        # st.subheader("Closing Price and Moving Averages")
        # fig2, ax2 = plt.subplots(figsize=(12, 6))
        # ax2.plot(data['Date'], data['Close'], color='blue', label="Actual Closing Price", linewidth=2)
        # ax2.plot(data['Date'], data['MA10'], color='orange', label="10 Day MA", linestyle='--')
        # ax2.plot(data['Date'], data['MA30'], color='green', label="30 Day MA", linestyle='--')
        # ax2.set_xlabel('Date')
        # ax2.set_ylabel('Price (USD)')
        # ax2.set_title('Closing Price and Moving Averages')
        # ax2.legend()
        # ax2.grid()
        # st.pyplot(fig2)
        
        

        # Closing Price and Moving Averages - Use Plotly for interactive chart
        st.subheader("Closing Price and Moving Averages")

        fig_ma = go.Figure()

        # Plot the actual closing price
        fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Actual Closing Price',
                                    line=dict(color='blue', width=2)))

        # Plot 10 Day Moving Average
        fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['MA10'], mode='lines', name='10 Day MA',
                                    line=dict(color='orange', dash='dash')))

        # Plot 30 Day Moving Average
        fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['MA30'], mode='lines', name='30 Day MA',
                                    line=dict(color='green', dash='dash')))
 
        # Set layout for the chart
        fig_ma.update_layout(title="Closing Price and Moving Averages over time",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            hovermode="x unified",  # Allows hover over the entire x-axis
                            legend=dict(x=0, y=1.1, orientation='h'),
                            template='plotly_white')  # White background for a cleaner look

        st.plotly_chart(fig_ma)


        # # Plot the trading volume as a larger bar chart
        # st.subheader("Trading Volume")
        # fig_volume, ax_volume = plt.subplots(figsize=(12, 6))
        # ax_volume.bar(data['Date'], data['Volume'], color='lightblue', width=1)  # Increased width for better visibility
        # ax_volume.set_xlabel('Date')
        # ax_volume.set_ylabel('Volume')
        # ax_volume.set_title('Trading Volume Over Time')
        # ax_volume.grid(axis='y')
        # st.pyplot(fig_volume)

       

        # # Plot sentiment analysis as a line plot
        # st.subheader("Sentiment Analysis")
        # fig_sentiment, ax_sentiment = plt.subplots(figsize=(12, 6))
        # ax_sentiment.plot(data['Date'], data['Sentiment'], color='purple', marker='o', linestyle='-')  # Line plot with markers
        # ax_sentiment.set_ylabel('Sentiment Value')
        # ax_sentiment.set_title('Sentiment Analysis Over Time')
        # ax_sentiment.set_ylim(-1, 1)  # Limit Y-axis to range [-1, 1]
        # ax_sentiment.grid()
        # st.pyplot(fig_sentiment)
        
        

        # Trading Volume - Use Plotly for interactive chart
        st.subheader("Trading Volume")
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(x=data['Date'], y=data['Volume'], marker_color='lightblue', name='Volume'))
        fig_volume.update_layout(title="Trading Volume Over Time", xaxis_title="Date", yaxis_title="Volume",
                                hovermode="x unified")
        st.plotly_chart(fig_volume)
        
        

        # # Sentiment Analysis - Use Plotly for interactive chart
        # st.subheader("Sentiment Analysis")
        # fig_sentiment = go.Figure()
        # fig_sentiment.add_trace(go.Scatter(x=data['Date'], y=data['Sentiment'], mode='lines+markers', name='Sentiment',
        #                                 marker=dict(color='purple')))
        # fig_sentiment.update_layout(title="Sentiment Analysis Over Time", xaxis_title="Date", yaxis_title="Sentiment",
        #                             yaxis_range=[-1, 1], hovermode="x unified")
        # st.plotly_chart(fig_sentiment)
        
        
        # Combine Sentiment and Closing Price in a Dual Axis Plot
        st.subheader("Closing Price and Sentiment Analysis")
        fig_combined = go.Figure()

        # Plot Closing Price on the primary y-axis
        fig_combined.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price',
                                        line=dict(color='blue', width=2)))

        # Plot Sentiment on the secondary y-axis
        fig_combined.add_trace(go.Scatter(x=data['Date'], y=data['Sentiment'], mode='lines+markers', name='Sentiment',
                                        marker=dict(color='purple')))

        # Update layout for dual y-axes
        fig_combined.update_layout(title="Closing Price and Sentiment Analysis Over Time",
                                xaxis_title="Date",
                                yaxis_title="Closing Price (USD)",
                                yaxis2=dict(title="Sentiment", overlaying='y', side='right', range=[-1, 1]),
                                hovermode="x unified")

        st.plotly_chart(fig_combined)

                
        # # Plot the predictions
        # st.subheader(f"Predicted Stock Prices for {ticker} for the next {days_ahead} days")
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.plot(future_df['Date'], future_df['Predicted Price'], color='red', label="Predicted Prices", marker='o')
        # ax.set_xlabel('Date')
        # ax.set_ylabel('Price (USD)')
        # ax.set_title('Future Stock Price Predictions')
        # ax.legend()
        # ax.grid(True)
        # st.pyplot(fig)
        
        # Future Stock Price Predictions - Use Plotly for interactive chart
        st.subheader(f"Predicted Stock Prices for {ticker} for the next {days_ahead} days")

        fig_predictions = go.Figure()

        # Plot the predicted prices
        fig_predictions.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted Price'], mode='lines+markers',
                                            name='Predicted Prices', line=dict(color='red'), marker=dict(size=8)))

        # Set layout for the chart
        fig_predictions.update_layout(title=f"Future Stock Price Predictions for {ticker}",
                                    xaxis_title="Date",
                                    yaxis_title="Price (USD)",
                                    hovermode="x unified",  # Allows hover over the entire x-axis
                                    legend=dict(x=0, y=1.1, orientation='h'),
                                    template='plotly_white')

        st.plotly_chart(fig_predictions)


        # Show future predictions in a table
        st.subheader("Future Predictions")
        st.write(future_df)


######CODE FOR CHECKING NEWS API#########
    # from newsapi import NewsApiClient

    # # Initialize with your API key
    # newsapi = NewsApiClient(api_key='ee93de27f75f49dd93997f391a741e7b')

    # # Fetch general stock market news
    # articles = newsapi.get_everything(q='stocks', language='en')

    # # Display the articles fetched
    # if articles['totalResults'] > 0:
    #     for article in articles['articles']:
    #         print(article['title'], "-", article['source']['name'])
    # else:
    #     print("No articles found.")
