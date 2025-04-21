# PyTorch LSTM and Neural Network Backtesting Example
# This script demonstrates how to use the PyTorch-based LSTM and Neural Network models for backtesting

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_prediction import (
    load_stock_data,
    run_lstm_prediction,
    run_nn_prediction,
    compare_models_with_buy_hold,
    get_backtest_data
)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run PyTorch-based stock prediction backtesting')
    parser.add_argument('--ticker', type=str, default='QQQ', help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, default='2018-01-01', help='Start date for training data (YYYY-MM-DD)')
    parser.add_argument('--split_date', type=str, default='2022-01-01', help='Split date between training and testing (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='End date for testing data (YYYY-MM-DD)')
    args = parser.parse_args()
    
    ticker = args.ticker
    
    print(f"\n===== PyTorch LSTM and Neural Network Backtesting for {ticker} =====\n")
    
    # Step 1: Get backtest data (will check if it exists first)
    print("\n----- Step 1: Getting Backtest Data -----")
    
    # Get backtest data or collect it if missing
    data_path, split_date = get_backtest_data(
        ticker,
        use_backtest=True,
        date_split=args.split_date
    )
    
    if data_path is None:
        print(f"Error: Failed to collect data for {ticker}")
        return
    
    print(f"Data loaded from: {data_path}")
    
    # Step 2: Run LSTM prediction on historical data
    print("\n----- Step 2: Running LSTM Model -----")
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
        data_file=data_path,
        feature_cols=feature_cols,
        sequence_length=30,  # Look back 30 days
        train_split=0.8,     # Use 80% for training, 20% for validation
        lstm_units=64,       # Number of LSTM units
        dropout_rate=0.2,    # Dropout rate for regularization
        epochs=50,           # Maximum number of epochs
        batch_size=32,       # Batch size
        date_split=split_date  # Date to split training and testing data
    )
    
    # Step 3: Run Neural Network prediction on historical data
    print("\n----- Step 3: Running Neural Network Model -----")
    # Run Neural Network prediction
    nn_model, nn_backtest = run_nn_prediction(
        data_file=data_path,
        feature_cols=feature_cols,
        sequence_length=30,   # Look back 30 days
        train_split=0.8,      # Use 80% for training, 20% for validation
        layers=[128, 64, 32], # Hidden layer sizes
        dropout_rate=0.2,     # Dropout rate for regularization
        epochs=50,            # Maximum number of epochs
        batch_size=32,        # Batch size
        date_split=split_date  # Date to split training and testing data
    )
    
    # Step 4: Compare models with Buy & Hold strategy
    print("\n----- Step 4: Comparing Models with Buy & Hold Strategy -----")
    if lstm_backtest is not None and nn_backtest is not None:
        compare_models_with_buy_hold(lstm_backtest, nn_backtest)
    
    print(f"\nBacktesting for {ticker} completed successfully!")

if __name__ == "__main__":
    main()