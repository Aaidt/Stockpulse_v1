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

    import pandas as pd
    import yfinance as yf
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dropout, Dense
    import tensorflow as tf
    from datetime import date
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error
    # from .forms import StockForm
    import forms
    
    # Set start and end dates for data
    START = "2010-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    stocksymbol = forms.stock_symbol
    # Function to load the dataset
    def load_data(stocksymbol):
        data = yf.download(stocksymbol, START, TODAY)
        data.reset_index(inplace=True)
        return data

    # Load the data
    # ticker = 'AAPL'
    data = load_data(forms.stock_symbol)
    df = data.copy()

    # Drop unnecessary columns if they exist
    df = df.drop(columns=['Date', 'Adj Close'], errors='ignore')

    # Plot the closing price
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'])
    plt.title(f"{forms.stock_symbol} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.show()

    # Calculate and plot 10-day moving average
    ma10 = df['Close'].rolling(window=10).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(ma10, 'r', label='10-Day MA')
    plt.title('10-Day Moving Average')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Calculate and plot 10-day and 30-day moving averages
    ma30 = df['Close'].rolling(window=30).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(ma10, 'r', label='10-Day MA')
    plt.plot(ma30, 'g', label='30-Day MA')
    plt.title('Comparison of 10-Day and 30-Day Moving Averages')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Split the data into training and testing sets
    train_size = int(len(df) * 0.70)
    train, test = df[:train_size], df[train_size:]

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_close = train[['Close']].values
    test_close = test[['Close']].values

    train_scaled = scaler.fit_transform(train_close)
    test_scaled = scaler.transform(test_close)

    # Prepare the training data
    x_train, y_train = [], []
    for i in range(10, len(train_scaled)):
        x_train.append(train_scaled[i-10:i])
        y_train.append(train_scaled[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=90, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    model.summary()

    # Train the model
    model.fit(x_train, y_train, epochs=10)

    # Save the trained model
    model.save('my_model.keras')

    # Function to make predictions using the saved model
    def make_predictions(stock_symbol):
        # Load the pre-trained model
        model = load_model('my_model.keras')

        # Reload the stock data for the selected ticker
        data = load_data(stock_symbol)
        df = data[['Close']]

        # Preprocess the data using the same scaler
        close_data = df[['Close']].values
        close_scaled = scaler.transform(close_data)

        # Prepare input data for prediction (last 100 days of data)
        x_input = []
        for i in range(10, len(close_scaled)):
            x_input.append(close_scaled[i-10:i])

        x_input = np.array(x_input)

        # Make predictions
        predicted_scaled = model.predict(x_input)

        # Inverse transform the predictions
        predicted = scaler.inverse_transform(predicted_scaled)

        # Return the last 100 actual and predicted values
        return {
            'actual': close_data[-100:].flatten(),
            'predicted': predicted.flatten()
        }

    # Make predictions using the trained model
    predictions = make_predictions(forms.stock_symbol)

    # Plot the actual vs predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(predictions['actual'], 'b', label="Original Price (Actual)")
    plt.plot(predictions['predicted'], 'r', label="Predicted Price")
    plt.title(f"Actual vs Predicted Prices for {forms.stock_symbol}")
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate the mean absolute error
    mae = mean_absolute_error(predictions['actual'], predictions['predicted'])
    mae_percentage = (mae / np.mean(predictions['actual'])) * 100
    print(f"Mean absolute error on test set: {mae_percentage:.2f}%")
