# Data Collection Script for Stock Prediction
# This script collects and saves stock data without running the LSTM models
# Use this as an alternative when TensorFlow is not available

import os
import pandas as pd
import matplotlib.pyplot as plt
from stock_data import collect_and_save_backtest_data

def main():
    print("\n===== Stock Data Collection for Prediction Models =====\n")
    
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
    
    # Define feature columns that would be used for the models
    feature_cols = [
        'Close', 
        'Volume', 
        'Oscillators_RSI', 
        'Oscillators_MACD.macd', 
        'Oscillators_MACD.signal',
        'Moving Averages_EMA20', 
        'Moving Averages_SMA50'
    ]
    
    print("\n----- Data Collection Complete -----")
    print("\nNote: The LSTM and Neural Network models require TensorFlow,")
    print("which is not compatible with Python 3.13 at this time.")
    print("\nOptions to proceed:")
    print("1. Use a lower version of Python (3.10 or 3.11) where TensorFlow is available")
    print("2. Use alternative models that don't require TensorFlow")
    print("3. Wait for TensorFlow to support Python 3.13")
    
    # Load and display some basic statistics about the collected data
    try:
        # First check the CSV file structure to determine the correct datetime column
        train_data = pd.read_csv(train_data_path)
        
        # Check if 'Date' or 'Datetime' column exists
        datetime_col = None
        if 'Datetime' in train_data.columns:
            datetime_col = 'Datetime'
        elif 'Date' in train_data.columns:
            datetime_col = 'Date'
        
        # If datetime column found, set it as index
        if datetime_col:
            train_data = pd.read_csv(train_data_path, index_col=datetime_col, parse_dates=True)
        
        print("\nTraining Data Statistics:")
        print(train_data.describe())
        
        # Plot the closing prices
        plt.figure(figsize=(12, 6))
        plt.plot(train_data.index, train_data['Close'])
        plt.title('QQQ Closing Prices (Training Data)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        
        # Save the plot
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        plt.savefig(os.path.join(plots_dir, "QQQ_Closing_Prices.png"))
        print(f"\nPlot saved to: {os.path.join(plots_dir, 'QQQ_Closing_Prices.png')}")
        plt.show()
        
    except Exception as e:
        print(f"Error analyzing data: {e}")

if __name__ == "__main__":
    main()