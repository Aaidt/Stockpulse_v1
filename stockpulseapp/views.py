
from django.shortcuts import render,redirect
from django.contrib import messages
from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout




# stockpulseapp/views.py
from django.shortcuts import render
from .forms import StockForm
import stockpulseapp.forms as forms
# from django.http import HttpResponse
import django.http as http 
# import stockpulseapp.lstm as lstm




def predictions_view(request):
    predictions = None  # Initialize predictions variable
    stock_symbol = None

    if request.method == 'POST':
        form = forms.StockForm(request.POST)
        if form.is_valid():
            
            stock_symbol = form.cleaned_data['stock_symbol']
            #stock_symbol = forms.stock_symbol
            # Call the function in lstm.py to make the prediction
            # predictions = lstm.make_predictions(forms.stock_symbol)
            predictions = make_predictions(stock_symbol)
            # predictions = lstm.predictions
            
            
    else:
        form = forms.StockForm()
                          
    context = {
        'form': form,
        'predictions': predictions,
        'stock_symbol': stock_symbol,
    }
    return render(request, 'core/predictions.html', context)

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
# import forms
# predictions = None





def make_predictions(stock_symbol):
        

        
    # Set start and end dates for data
    START = "2010-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    # stocksymbol = stock_symbol
    # Function to load the dataset
    def load_data(stock_symbol):
        data = yf.download(stock_symbol, START, TODAY)
        data.reset_index(inplace=True)
        return data 

    # Load the data
    # ticker = 'AAPL'
    data = load_data(stock_symbol)
    df = data.copy()

    # Drop unnecessary columns if they exist
    df = df.drop(columns=['Date', 'Adj Close'], errors='ignore')

    # Plot the closing price
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'])
    plt.title(f"{stock_symbol} Stock Price")
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
    test = scaler.transform(test_close)

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
    data= close_data[-100:].flatten()
    data_predicted = predicted.flatten()


    
    

    
    # Return the last 100 actual and predicted values
    return {
        'actual': close_data[-100:].flatten(),
        'predicted': predicted.flatten()
            }
        
   

def plot_predictions(predictions):
        # Plot the actual vs predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(predictions['actual'], 'b', label="Original Price (Actual)")
    plt.plot(predictions['predicted'], 'r', label="Predicted Price")
    plt.title(f"Actual vs Predicted Prices for {stock_symbol}")
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate the mean absolute error
    mae = mean_absolute_error(predictions['actual'], predictions['predicted'])
    mae_percentage = (mae / np.mean(predictions['actual'])) * 100
    print(f"Mean absolute error on test set: {mae_percentage:.2f}%")

    
predictions = make_predictions(stock_symbol)   
plotgraph = plot_predictions(predictions)
    













def index(request):
    context = {
        'show_navbar': True,
        'show_footer': True,
    }
    return render(request, 'core/index.html', context)


@login_required
def crypto(request):
    context = {
        'show_navbar': True,
        'show_footer': True,
    }
    return render(request, 'core/cryptocurrency.html', context)

@login_required
def news(request):
    context = {
        'show_navbar': False,
        'show_footer': True,
    }
    return render(request, 'core/news.html', context)

@login_required
def personal(request):
    context = {
        'show_navbar': False,
        'show_footer': False,
    }
    return render(request, 'core/personal.html', context)

@login_required
def calculator(request):
    context = {
        'show_navbar': True,
        'show_footer': True,
    }
    return render(request, 'core/calculator.html', context)

@login_required
def watchlist(request):
    context = {
        'show_navbar': True,
        'show_footer': True,
    }
    return render(request, 'core/watchlist.html', context)

@login_required
def academy(request):
    context = {
        'show_navbar': True,
        'show_footer': False,
    }
    return render(request, 'core/academy.html', context)

@login_required
def streamlit_view(request):
    context = {
        'show_navbar': True,
        'show_footer': True,
    }    
    return render(request, 'core/forecast.html')

def authView(request):
 if request.method == "POST":
  form = UserCreationForm(request.POST or None)
  if form.is_valid():
   form.save()
   return redirect("stockpulseapp:login")
 else:
  form = UserCreationForm()
 return render(request, "registration/signup.html", {"form": form})
    
def logout_view(request):
    logout(request)
    # Redirect to a success page or any other page after logout
    return redirect('stockpulseapp:index')

