# Enhanced Stock Price Prediction App

## Project Description:
This project is a Streamlit-based web application that predicts stock prices using LSTM (Long Short-Term Memory) neural networks. The app also incorporates sentiment analysis of financial news and technical indicators like moving averages (MA) and RSI (Relative Strength Index) for enhanced stock price prediction.

## Features:
- **LSTM Model**: Predicts future stock prices based on historical data.
- **Sentiment Analysis**: Uses news headlines and articles to gauge market sentiment on the stock.
- **Technical Indicators**: Includes Moving Averages (MA10, MA30) and RSI to provide additional market insights.
- **Interactive Visualizations**: Includes enhanced interactive plots for stock prices, technical indicators, and future price predictions.
- **Trading Volume**: Displays trading volume over time as an easy-to-read bar chart.

## Technologies Used:
- **Streamlit**: For building the web app interface.
- **TensorFlow/Keras**: For creating and training the LSTM model.
- **yfinance**: For fetching stock price data.
- **NewsAPI**: For retrieving financial news articles.
- **NLTK (VADER Sentiment)**: For sentiment analysis of news articles.
- **ta**: For calculating technical indicators like RSI.
  
## Screenshots:

## Setup Instructions:
To set up and run the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Aaidt/Stockpulse_v1.git
    ```
2. Navigate to the project directory:
    ```bash
    cd stockpulseapp
    ```
3. Create a virtual environment and activate it:
    ```bash
    python -m venv env
    # For Windows: .\env\Scripts\activate
    ```
4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
5. Add your NewsAPI Key:
    ```bash
   newsapi = NewsApiClient(api_key='YOUR_API_KEY_HERE')
   ```
7. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
8. Run the StockPulse website:
    ```bash
    python manage.py runserver
    ```
    
## Usage:
1. Open the web app in your browser after running it locally.
2. Enter a stock symbol (e.g., AAPL for Apple, TSLA for Tesla).
3. Select the number of days to predict.
4. View the stock's price history, sentiment analysis, technical indicators, and future price predictions.

## How the Sentiment Analysis Works
Sentiment is derived from the most recent 30 days of news articles related to the stock. The VADER sentiment model classifies the news as positive, neutral, or negative, and this sentiment is used as an input feature for the LSTM model.
  
## Future Improvements:
- **User Authentication**: Add login functionality to personalize predictions.
- **More Indicators**: Incorporate additional technical indicators such as MACD.
- **Improved Model**: Use other machine learning techniques for better accuracy.
- **Real-time Predictions**: Implement real-time stock predictions with live data.

## License:
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements:
- [Streamlit](https://streamlit.io/): For building the web app interface.
- [TensorFlow/Kerashttps](https://www.tensorflow.org/): For creating and training the LSTM model.
- [yfinance](https://pypi.org/project/yfinance/): For fetching stock price data.
- [NewsAPI](https://newsapi.org/): For retrieving financial news articles.
- [NLTK (VADER Sentiment)](https://www.nltk.org/): For sentiment analysis of news articles.
