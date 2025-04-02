# Calculate Missing Technical Indicators
# This script calculates technical indicators for data points that show N/A values
# in the backtestQQQ_data.csv file

import pandas as pd
import numpy as np
import os

def calculate_rsi(prices, window=14):
    """Calculate RSI with alternative method for initial periods"""
    # Traditional RSI calculation
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    
    if down == 0:  # Handle division by zero
        rs = float('inf')
    else:
        rs = up/down
    
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1. + rs)  # Use the initial RS for first window periods
    
    # Calculate RSI for the rest of the periods using traditional method
    for i in range(window, len(prices)):
        delta = deltas[i-1]  # Current price change
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
            
        up = (up * (window - 1) + upval) / window
        down = (down * (window - 1) + downval) / window
        
        if down == 0:  # Handle division by zero
            rs = float('inf')
        else:
            rs = up/down
        
        rsi[i] = 100. - 100./(1. + rs)
    
    return rsi

def calculate_stochastic(close_prices, high_prices, low_prices, k_window=14, d_window=3):
    """Calculate Stochastic Oscillator with alternative method for initial periods"""
    # Initialize arrays
    k_percent = np.zeros_like(close_prices)
    d_percent = np.zeros_like(close_prices)
    
    # For the initial periods, use available data
    for i in range(len(close_prices)):
        if i < k_window:
            # Use available data for initial periods
            low_val = np.min(low_prices[:i+1])
            high_val = np.max(high_prices[:i+1])
        else:
            # Traditional calculation for later periods
            low_val = np.min(low_prices[i-k_window+1:i+1])
            high_val = np.max(high_prices[i-k_window+1:i+1])
        
        if high_val - low_val == 0:  # Handle division by zero
            k_percent[i] = 50  # Neutral value
        else:
            k_percent[i] = 100 * (close_prices[i] - low_val) / (high_val - low_val)
    
    # Calculate %D (moving average of %K)
    for i in range(len(close_prices)):
        if i < d_window:
            # Use available data for initial periods
            d_percent[i] = np.mean(k_percent[:i+1])
        else:
            # Traditional calculation for later periods
            d_percent[i] = np.mean(k_percent[i-d_window+1:i+1])
    
    return k_percent, d_percent

def calculate_cci(high_prices, low_prices, close_prices, window=20):
    """Calculate CCI with alternative method for initial periods"""
    # Calculate typical price
    typical_price = (high_prices + low_prices + close_prices) / 3
    
    # Initialize CCI array
    cci = np.zeros_like(close_prices)
    
    # For each period
    for i in range(len(close_prices)):
        if i < window:
            # Use available data for initial periods
            sma = np.mean(typical_price[:i+1])
            mean_deviation = np.mean(np.abs(typical_price[:i+1] - sma))
        else:
            # Traditional calculation for later periods
            sma = np.mean(typical_price[i-window+1:i+1])
            mean_deviation = np.mean(np.abs(typical_price[i-window+1:i+1] - sma))
        
        if mean_deviation == 0:  # Handle division by zero
            cci[i] = 0  # Neutral value
        else:
            cci[i] = (typical_price[i] - sma) / (0.015 * mean_deviation)
    
    return cci

def calculate_macd(prices, fast_window=12, slow_window=26, signal_window=9):
    """Calculate MACD with alternative method for initial periods"""
    # Initialize arrays
    macd = np.zeros_like(prices)
    signal = np.zeros_like(prices)
    
    # Calculate EMA with alternative method for initial periods
    ema_fast = np.zeros_like(prices)
    ema_slow = np.zeros_like(prices)
    
    # Initialize EMAs with SMA for first periods
    for i in range(len(prices)):
        if i < fast_window:
            ema_fast[i] = np.mean(prices[:i+1])
        else:
            # Traditional EMA calculation
            alpha_fast = 2 / (fast_window + 1)
            ema_fast[i] = prices[i] * alpha_fast + ema_fast[i-1] * (1 - alpha_fast)
        
        if i < slow_window:
            ema_slow[i] = np.mean(prices[:i+1])
        else:
            # Traditional EMA calculation
            alpha_slow = 2 / (slow_window + 1)
            ema_slow[i] = prices[i] * alpha_slow + ema_slow[i-1] * (1 - alpha_slow)
        
        # Calculate MACD
        macd[i] = ema_fast[i] - ema_slow[i]
    
    # Calculate signal line (EMA of MACD)
    for i in range(len(prices)):
        if i < signal_window:
            signal[i] = np.mean(macd[:i+1])
        else:
            # Traditional EMA calculation
            alpha_signal = 2 / (signal_window + 1)
            signal[i] = macd[i] * alpha_signal + signal[i-1] * (1 - alpha_signal)
    
    return macd, signal

def calculate_moving_averages(prices):
    """Calculate various moving averages with alternative method for initial periods"""
    # Initialize arrays for EMAs
    ema5 = np.zeros_like(prices)
    ema10 = np.zeros_like(prices)
    ema20 = np.zeros_like(prices)
    ema50 = np.zeros_like(prices)
    ema100 = np.zeros_like(prices)
    ema200 = np.zeros_like(prices)
    
    # Initialize arrays for SMAs
    sma5 = np.zeros_like(prices)
    sma10 = np.zeros_like(prices)
    sma20 = np.zeros_like(prices)
    sma50 = np.zeros_like(prices)
    sma100 = np.zeros_like(prices)
    sma200 = np.zeros_like(prices)
    
    # Calculate EMAs
    windows = [5, 10, 20, 50, 100, 200]
    ema_arrays = [ema5, ema10, ema20, ema50, ema100, ema200]
    
    for i, window in enumerate(windows):
        for j in range(len(prices)):
            if j < window:
                # Use SMA for initial periods
                ema_arrays[i][j] = np.mean(prices[:j+1])
            else:
                # Traditional EMA calculation
                alpha = 2 / (window + 1)
                ema_arrays[i][j] = prices[j] * alpha + ema_arrays[i][j-1] * (1 - alpha)
    
    # Calculate SMAs
    windows = [5, 10, 20, 50, 100, 200]
    sma_arrays = [sma5, sma10, sma20, sma50, sma100, sma200]
    
    for i, window in enumerate(windows):
        for j in range(len(prices)):
            if j < window:
                # Use available data for initial periods
                sma_arrays[i][j] = np.mean(prices[:j+1])
            else:
                # Traditional SMA calculation
                sma_arrays[i][j] = np.mean(prices[j-window+1:j+1])
    
    return {
        'EMA5': ema5, 'EMA10': ema10, 'EMA20': ema20, 'EMA50': ema50, 'EMA100': ema100, 'EMA200': ema200,
        'SMA5': sma5, 'SMA10': sma10, 'SMA20': sma20, 'SMA50': sma50, 'SMA100': sma100, 'SMA200': sma200
    }

def process_backtest_data(file_path):
    """Process backtest data to calculate missing indicators"""
    # Read the CSV file
    df = pd.read_csv(file_path, parse_dates=['Date'])
    
    # Convert string 'N/A' to actual NaN values for calculations
    for col in df.columns:
        df[col] = df[col].replace('N/A', np.nan)
    
    # Extract price data
    close_prices = df['Close'].values
    high_prices = df['High'].values
    low_prices = df['Low'].values
    
    # Calculate RSI
    rsi_values = calculate_rsi(close_prices)
    df['Oscillators_RSI'] = rsi_values
    
    # Calculate RSI[1] (previous period RSI)
    df['Oscillators_RSI[1]'] = df['Oscillators_RSI'].shift(1)
    
    # Calculate Stochastic Oscillator
    k_percent, d_percent = calculate_stochastic(close_prices, high_prices, low_prices)
    df['Oscillators_STOCH.K'] = k_percent
    df['Oscillators_STOCH.D'] = d_percent
    
    # Calculate CCI
    cci_values = calculate_cci(high_prices, low_prices, close_prices)
    df['Oscillators_CCI'] = cci_values
    
    # Calculate MACD
    macd_values, signal_values = calculate_macd(close_prices)
    df['Oscillators_MACD.macd'] = macd_values
    df['Oscillators_MACD.signal'] = signal_values
    
    # Calculate Moving Averages
    ma_values = calculate_moving_averages(close_prices)
    
    # Update EMA values
    df['Moving Averages_EMA5'] = ma_values['EMA5']
    df['Moving Averages_EMA10'] = ma_values['EMA10']
    df['Moving Averages_EMA20'] = ma_values['EMA20']
    df['Moving Averages_EMA50'] = ma_values['EMA50']
    df['Moving Averages_EMA100'] = ma_values['EMA100']
    df['Moving Averages_EMA200'] = ma_values['EMA200']
    
    # Update SMA values
    df['Moving Averages_SMA5'] = ma_values['SMA5']
    df['Moving Averages_SMA10'] = ma_values['SMA10']
    df['Moving Averages_SMA20'] = ma_values['SMA20']
    df['Moving Averages_SMA50'] = ma_values['SMA50']
    df['Moving Averages_SMA100'] = ma_values['SMA100']
    df['Moving Averages_SMA200'] = ma_values['SMA200']
    
    # Round numerical values to 3 decimal places for readability
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].round(3)
    
    # Save the updated data back to the CSV file
    df.to_csv(file_path, index=False)
    print(f"Updated data saved to {file_path}")
    
    return df

if __name__ == "__main__":
    # Path to the backtest data file
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stockData', 'backtestQQQ_data.csv')
    
    # Process the data
    updated_df = process_backtest_data(file_path)
    print("Technical indicators have been calculated for all data points.")