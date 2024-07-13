import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, period='1y', interval='1d'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    return data

from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    data = data[['Open', 'High', 'Low', 'Close']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_model(data):
    X = data[:, :-1]  # Features: Open, High, Low
    y = data[:, -1]   # Target: Close
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def predict(model, data):
    predictions = model.predict(data)
    return predictions

import streamlit as st

def main():
    st.title("Stock Price Prediction App")
    
    # User input for ticker symbol
    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    
    # Fetch and display data
    data = fetch_stock_data(ticker)
    st.write("Recent Data")
    st.write(data.tail())
    
    # Preprocess data and train model
    scaled_data, scaler = preprocess_data(data)
    model, X_test, y_test = train_model(scaled_data)
    
    # Predict and display results
    predictions = predict(model, X_test)
    predictions = scaler.inverse_transform([*X_test.T, predictions]).T  # Inverse scaling
    
    st.write("Predicted vs Actual Prices")
    st.write(pd.DataFrame({'Actual': y_test, 'Predicted': predictions[:, -1]}))

if __name__ == "__main__":
    main()
