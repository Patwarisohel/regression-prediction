import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st

def fetch_stock_data(ticker, period='1y', interval='1d'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    return data

def preprocess_data(data):
    data = data[['Open', 'High', 'Low', 'Close']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def train_model(data):
    X = data[:, :-1]  # Features: Open, High, Low
    y = data[:, -1]   # Target: Close
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test, y

def predict(model, data):
    predictions = model.predict(data)
    return predictions

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
    model, X_test, y_test, y_all = train_model(scaled_data)
    
    # Predict and display results
    predictions = predict(model, X_test)
    
    # Inverse scale the predictions and actual values
    X_test_full = np.hstack((X_test, predictions.reshape(-1, 1)))
    all_data = scaler.inverse_transform(scaled_data)
    scaled_pred_data = np.hstack((X_test, predictions.reshape(-1, 1)))
    inv_predictions = scaler.inverse_transform(scaled_pred_data)[:, -1]
    
    inv_y_test = scaler.inverse_transform(np.hstack((X_test, y_test.reshape(-1, 1))))[:, -1]
    
    result_df = pd.DataFrame({'Actual': inv_y_test, 'Predicted': inv_predictions})
    st.write("Predicted vs Actual Prices")
    st.write(result_df)

if __name__ == "__main__":
    main()
