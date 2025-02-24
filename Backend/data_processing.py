import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import ta

def download_stock_data(ticker, start_date="2010-01-01", end_date="2020-12-31"):
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    df.reset_index(inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    print("Original columns after flattening:", df.columns)
    print("Close column shape before fix:", df["Close"].shape)
    df["Close"] = df["Close"].squeeze()
    print("Close column shape after fix:", df["Close"].shape)
    
    return df

def add_technical_indicators(df):
    # Using the 'ta' library to add technical indicators
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["SMA_200"] = ta.trend.sma_indicator(df["Close"], window=200)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd_diff(df["Close"])
    df = df.dropna()
    return df

def scale_data(df, feature_columns):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df[feature_columns].values)
    return data_scaled, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])  # Predict the 'Close' price (first feature)
    return np.array(X), np.array(y)
