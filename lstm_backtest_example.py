# LSTM and Neural Network Backtesting Example
# This script demonstrates how to use the LSTM and Neural Network models for backtesting

from lstm_prediction import (
    load_stock_data,
    run_lstm_prediction,
    run_nn_prediction,
    compare_models_with_buy_hold
)
import os
import pandas as pd
import matplotlib.pyplot as plt
from stock_data import collect_and_save_backtest_data

def main():
    print("\n===== LSTM and Neural Network Backtesting Example =====\n")
    
    # Step 1: Collect recent data for testing
    print("\n----- Step 1: Collecting Recent Data for Testing -----")
    # Collect data from 2023 to present for testing the model on recent data
    test_data_path = collect_and_save_backtest_data(
        'QQQ', 
        interval='1d', 
        output_format='csv', 
        start_date='2023-01-01'
    )
    print(f"Recent test data saved to: {test_data_path}")
    
    # Step 2: Collect historical data for training and backtesting
    print("\n----- Step 2: Collecting Historical Data for Training -----")
    # Collect data from 2018 to 2022 for training and backtesting
    train_data_path = collect_and_save_backtest_data(
        'QQQ', 
        interval='1d', 
        output_format='csv', 
        start_date='2018-01-01', 
        end_date='2022-12-31'
    )
    print(f"Historical training data saved to: {train_data_path}")
    
    # Step 3: Run LSTM prediction on historical data
    print("\n----- Step 3: Running LSTM Model on Historical Data -----")
    # Define feature columns to use
    feature_cols = [
        'Close', 
        'Volume', 
        'Oscillators_RSI', 
        'Oscillators_MACD.macd', 
        'Oscillators_MACD.signal',
        'Moving Averages_EMA20', 
        'Moving Averages_SMA50'
    ]
    
    # Run LSTM prediction
    lstm_model, lstm_backtest = run_lstm_prediction(
        train_data_path,
        feature_cols=feature_cols,
        sequence_length=30,  # Look back 30 days
        train_split=0.8,     # Use 80% for training, 20% for validation
        lstm_units=64,       # Number of LSTM units
        dropout_rate=0.2,    # Dropout rate for regularization
        epochs=50,           # Maximum number of epochs
        batch_size=32        # Batch size
    )
    
    # Step 4: Run Neural Network prediction on historical data
    print("\n----- Step 4: Running Neural Network Model on Historical Data -----")
    # Run Neural Network prediction
    nn_model, nn_backtest = run_nn_prediction(
        train_data_path,
        feature_cols=feature_cols,
        sequence_length=30,   # Look back 30 days
        train_split=0.8,      # Use 80% for training, 20% for validation
        layers=[128, 64, 32], # Hidden layer sizes
        dropout_rate=0.2,     # Dropout rate for regularization
        epochs=50,            # Maximum number of epochs
        batch_size=32         # Batch size
    )
    
    # Step 5: Compare models with Buy & Hold strategy
    print("\n----- Step 5: Comparing Models with Buy & Hold Strategy -----")
    if lstm_backtest is not None and nn_backtest is not None:
        compare_models_with_buy_hold(lstm_backtest, nn_backtest)
    
    # Step 6: Test models on recent data (out-of-sample testing)
    print("\n----- Step 6: Testing Models on Recent Data (Out-of-Sample) -----")
    # Load recent data
    recent_data = load_stock_data(test_data_path)
    
    if recent_data is not None and lstm_model is not None and nn_model is not None:
        print("\nRecent data loaded successfully. Ready for out-of-sample testing.")
        print("This would involve using the trained models to make predictions on the recent data.")
        print("For a complete implementation, you would need to create prediction functions")
        print("that can use pre-trained models on new data.")
        
        # Note: The actual implementation of out-of-sample testing would require
        # additional functions to prepare the recent data in the same way as the training data
        # and to use the trained models to make predictions on this data.
        # This is left as an extension for the user.

if __name__ == "__main__":
    main()