import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from .attention_layer import Attention
from backend.utils.data_processing import download_stock_data, add_technical_indicators, scale_data, create_sequences

def build_advanced_model(input_shape):
    model = Sequential()
    
    # Convolutional layer to extract local features
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    # Bidirectional LSTM to capture temporal dependencies
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3))
    
    # Additional LSTM layer
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    
    # Custom attention mechanism to weight important time steps
    model.add(Attention())
    
    # Fully connected layers for prediction
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_supervised_model(ticker="AAPL", epochs=100, batch_size=32):
    # Download and process stock data
    df = download_stock_data(ticker)
    df = add_technical_indicators(df)
    feature_columns = ['Close', 'SMA_50', 'SMA_200', 'RSI', 'MACD']
    data_scaled, scaler = scale_data(df, feature_columns)
    seq_length = 60
    X, y = create_sequences(data_scaled, seq_length)
    
    # Split data into training and validation sets (80/20 split)
    split_index = int(len(X) * 0.8)
    X_train, y_train = X[:split_index], y[:split_index]
    X_val, y_val = X[split_index:], y[split_index:]
    
    # Build the advanced model
    model = build_advanced_model((X_train.shape[1], X_train.shape[2]))
    
    # Early stopping callback to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val),
                        epochs=epochs, 
                        batch_size=batch_size, 
                        callbacks=[early_stop],
                        verbose=1)
    return model, scaler, seq_length

def predict_stock_price(model, scaler, seq_length, ticker="AAPL"):
    df = download_stock_data(ticker)
    df = add_technical_indicators(df)
    feature_columns = ['Close', 'SMA_50', 'SMA_200', 'RSI', 'MACD']
    data_scaled = scaler.transform(df[feature_columns].values)
    
    # Create sequences
    X, _ = create_sequences(data_scaled, seq_length)
    
    # Get predictions
    predictions = model.predict(X)

    # Ensure predictions are 1D
    predictions = predictions.flatten()  # Fix: Convert (N,1) to (N,)

    # Inverse transform the predicted close price
    def inverse_transform(scaled, scaler, feature_index=0, total_features=len(feature_columns)):
        dummy = np.zeros((scaled.shape[0], total_features))
        dummy[:, feature_index] = scaled
        inv = scaler.inverse_transform(dummy)[:, feature_index]
        return inv

    predicted_prices = inverse_transform(predictions, scaler)
    
    return float(predicted_prices[-1])  # Return the latest predicted price as a float

