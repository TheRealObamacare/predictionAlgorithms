# PyTorch LSTM and Neural Network Backtesting Example
# This script demonstrates how to use the PyTorch-based LSTM and Neural Network models for backtesting
# with ticker-based data fetching and date-based train/test splitting

from pytorch_prediction import (
    run_lstm_prediction,
    run_nn_prediction,
    compare_models_with_buy_hold
)
import datetime
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run PyTorch-based stock prediction backtesting')
    parser.add_argument('--ticker', type=str, default='QQQ', help='Stock ticker symbol (e.g., AAPL, MSFT, QQQ)')
    parser.add_argument('--train_years', type=int, default=4, help='Number of years of data to use for training')
    parser.add_argument('--test_years', type=int, default=1, help='Number of years of data to use for testing')
    args = parser.parse_args()
    
    # Get ticker from arguments
    ticker = args.ticker.upper()
    
    print(f"\n===== PyTorch LSTM and Neural Network Backtesting for {ticker} =====\n")
    
    # Calculate dates for train/test split
    today = datetime.datetime.now()
    test_start_date = (today - datetime.timedelta(days=365 * args.test_years)).strftime('%Y-%m-%d')
    train_start_date = (today - datetime.timedelta(days=365 * (args.train_years + args.test_years))).strftime('%Y-%m-%d')
    
    print(f"Training period: {train_start_date} to {test_start_date}")
    print(f"Testing period: {test_start_date} to present")
    
    # Define feature columns to use
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Oscillators_RSI', 
        'Oscillators_MACD.macd', 
        'Oscillators_MACD.signal',
        'Moving Averages_EMA20', 
        'Moving Averages_SMA50'
    ]
    
    # Step 1: Run LSTM prediction with ticker-based data fetching and date-based split
    print("\n----- Step 1: Running LSTM Model with Date-Based Split -----")
    lstm_model, lstm_backtest = run_lstm_prediction(
        ticker=ticker,
        feature_cols=feature_cols,
        sequence_length=30,       # Look back 30 days
        date_split=test_start_date,  # Use date-based split
        lstm_units=64,           # Number of LSTM units
        dropout_rate=0.2,        # Dropout rate for regularization
        epochs=50,               # Maximum number of epochs
        batch_size=32            # Batch size
    )
    
    # Step 2: Run Neural Network prediction with ticker-based data fetching and date-based split
    print("\n----- Step 2: Running Neural Network Model with Date-Based Split -----")
    nn_model, nn_backtest = run_nn_prediction(
        ticker=ticker,
        feature_cols=feature_cols,
        sequence_length=30,        # Look back 30 days
        date_split=test_start_date,   # Use date-based split
        layers=[128, 64, 32],     # Hidden layer sizes
        dropout_rate=0.2,         # Dropout rate for regularization
        epochs=50,                # Maximum number of epochs
        batch_size=32             # Batch size
    )
    
    # Step 3: Compare models with Buy & Hold strategy
    print("\n----- Step 3: Comparing Models with Buy & Hold Strategy -----")
    if lstm_backtest is not None and nn_backtest is not None:
        compare_models_with_buy_hold(lstm_backtest, nn_backtest)
    
    print(f"\nBacktesting for {ticker} completed successfully!")

if __name__ == "__main__":
    main()