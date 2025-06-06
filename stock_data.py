# Stock Data Retrieval Module
# This module provides functions to fetch stock data from Yahoo Finance and TradingView
import yfinance as yf
import pandas as pd
import datetime
import requests
import json
import time
import os

# Yahoo Finance Functions
def get_yahoo_historical_data(ticker, period="1y", interval="1d"):
    """
    Fetch historical stock data from Yahoo Finance
    
    Parameters:
    ticker (str): Stock symbol (e.g., 'AAPL', 'MSFT')
    period (str): Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    interval (str): Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    
    Returns:
    pandas.DataFrame: Historical stock data
    """
    try:
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=period, interval=interval)
        return hist_data
    except Exception as e:
        print(f"Error fetching Yahoo Finance historical data: {e}")
        return None
        
def get_yahoo_dividend_data(ticker, start_date=None, end_date=None):
    """
    Fetch dividend data for a stock from Yahoo Finance
    
    Parameters:
    ticker (str): Stock symbol (e.g., 'AAPL', 'MSFT')
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
    pandas.DataFrame: Dividend data with dates and amounts
    """
    try:
        stock = yf.Ticker(ticker)
        
        # If dates are provided, use them; otherwise, get all available dividend data
        if start_date and end_date:
            dividends = stock.dividends.loc[start_date:end_date]
        else:
            dividends = stock.dividends
            
        if len(dividends) > 0:
            # Convert Series to DataFrame with proper column name
            dividends_df = dividends.to_frame().reset_index()
            dividends_df.columns = ['Date', 'Dividend']
            dividends_df['Date'] = pd.to_datetime(dividends_df['Date'])
            dividends_df = dividends_df.set_index('Date')
            return dividends_df
        else:
            print(f"No dividend data found for {ticker}")
            return pd.DataFrame(columns=['Dividend'])
    except Exception as e:
        print(f"Error fetching Yahoo Finance dividend data: {e}")
        return pd.DataFrame(columns=['Dividend'])

def get_yahoo_stock_info(ticker):
    """
    Fetch general information about a stock from Yahoo Finance
    
    Parameters:
    ticker (str): Stock symbol (e.g., 'AAPL', 'MSFT')
    
    Returns:
    dict: Stock information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception as e:
        print(f"Error fetching Yahoo Finance stock info: {e}")
        return None

def get_yahoo_financials(ticker):
    """
    Fetch financial data for a stock from Yahoo Finance
    
    Parameters:
    ticker (str): Stock symbol (e.g., 'AAPL', 'MSFT')
    
    Returns:
    dict: Financial data
    """
    try:
        stock = yf.Ticker(ticker)
        financials = {
            'balance_sheet': stock.balance_sheet,
            'income_statement': stock.income_stmt,
            'cash_flow': stock.cashflow
        }
        return financials
    except Exception as e:
        print(f"Error fetching Yahoo Finance financials: {e}")
        return None

# TradingView Data Functions
# Note: TradingView doesn't have an official API, so this uses a workaround
# that might need adjustments if TradingView changes their interface

def get_tradingview_technical_indicators(ticker, exchange="NASDAQ", screener="america", interval="1d"):
    """
    Fetch technical indicators from TradingView using tradingview-ta library
    
    Parameters:
    ticker (str): Stock symbol (e.g., 'AAPL', 'MSFT')
    exchange (str): Exchange name (e.g., 'NASDAQ', 'NYSE')
    screener (str): Screener to use (e.g., 'america', 'japan')
    interval (str): Time interval (e.g., '1m', '5m', '15m', '1h', '4h', '1d', '1W', '1M')
    
    Returns:
    dict: Technical indicators data including oscillators and moving averages
    """
    try:
        # Import the tradingview-ta library
        from tradingview_ta import TA_Handler, Interval
        
        interval_map = {
            "1m": Interval.INTERVAL_1_MINUTE,
            "5m": Interval.INTERVAL_5_MINUTES,
            "15m": Interval.INTERVAL_15_MINUTES,
            "30m": Interval.INTERVAL_30_MINUTES,
            "1h": Interval.INTERVAL_1_HOUR,
            "2h": Interval.INTERVAL_2_HOURS,
            "4h": Interval.INTERVAL_4_HOURS,
            "1d": Interval.INTERVAL_1_DAY,
            "1W": Interval.INTERVAL_1_WEEK,
            "1M": Interval.INTERVAL_1_MONTH
        }
        
        # Use default interval if provided interval is not valid
        tv_interval = interval_map.get(interval, Interval.INTERVAL_1_DAY)
        
        # Initialize TA_Handler to get analysis from TradingView
        handler = TA_Handler(
            symbol=ticker,
            exchange=exchange,
            screener=screener,
            interval=tv_interval,
            timeout=10
        )
        
        # Get the analysis
        analysis = handler.get_analysis()
        
        # Extract and organize the data
        result = {
            "Summary": {
                "RECOMMENDATION": analysis.summary["RECOMMENDATION"],
                "BUY": analysis.summary["BUY"],
                "SELL": analysis.summary["SELL"],
                "NEUTRAL": analysis.summary["NEUTRAL"]
            },
            "Oscillators": {
                "RECOMMENDATION": analysis.oscillators["RECOMMENDATION"],
                "BUY": analysis.oscillators["BUY"],
                "SELL": analysis.oscillators["SELL"],
                "NEUTRAL": analysis.oscillators["NEUTRAL"],
                "RSI": analysis.indicators.get("RSI", "N/A"),
                "RSI[1]": analysis.indicators.get("RSI[1]", "N/A"),
                "STOCH.K": analysis.indicators.get("Stoch.K", "N/A"),
                "STOCH.D": analysis.indicators.get("Stoch.D", "N/A"),
                "CCI": analysis.indicators.get("CCI", "N/A"),
                "MACD.macd": analysis.indicators.get("MACD.macd", "N/A"),
                "MACD.signal": analysis.indicators.get("MACD.signal", "N/A")
            },
            "Moving Averages": {
                "RECOMMENDATION": analysis.moving_averages["RECOMMENDATION"],
                "BUY": analysis.moving_averages["BUY"],
                "SELL": analysis.moving_averages["SELL"],
                "NEUTRAL": analysis.moving_averages["NEUTRAL"],
                "EMA5": analysis.indicators.get("EMA5", "N/A"),
                "SMA5": analysis.indicators.get("SMA5", "N/A"),
                "EMA10": analysis.indicators.get("EMA10", "N/A"),
                "SMA10": analysis.indicators.get("SMA10", "N/A"),
                "EMA20": analysis.indicators.get("EMA20", "N/A"),
                "SMA20": analysis.indicators.get("SMA20", "N/A"),
                "EMA50": analysis.indicators.get("EMA50", "N/A"),
                "SMA50": analysis.indicators.get("SMA50", "N/A"),
                "EMA100": analysis.indicators.get("EMA100", "N/A"),
                "SMA100": analysis.indicators.get("SMA100", "N/A"),
                "EMA200": analysis.indicators.get("EMA200", "N/A"),
                "SMA200": analysis.indicators.get("SMA200", "N/A")
            },
            "Indicators": {
                "Close": analysis.indicators.get("close", "N/A"),
                "Open": analysis.indicators.get("open", "N/A"),
                "High": analysis.indicators.get("high", "N/A"),
                "Low": analysis.indicators.get("low", "N/A"),
                "Volume": analysis.indicators.get("volume", "N/A"),
                "Volatility": analysis.indicators.get("Volatility", "N/A"),
                "Relative Volume": analysis.indicators.get("Relative Volume", "N/A")
            }
        }
        
        return result
    except ImportError:
        print("Error: tradingview-ta package is not installed.")
        print("Install with: pip install tradingview-ta")
        return None
    except Exception as e:
        print(f"Error accessing TradingView data: {e}")
        return None

# Data Collection and File Creation Functions
def collect_and_save_stock_data(ticker, period="1y", interval="1d", exchange="NASDAQ", screener="america", output_format="csv", start_date=None, end_date=None):
    """
    Collect stock data from Yahoo Finance and calculate technical indicators directly on historical data,
    then save it to a file named after the ticker in the 'stockData' folder.
    
    Parameters:
    ticker (str): Stock symbol (e.g., 'AAPL', 'MSFT')
    period (str): Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                  Note: This parameter is ignored if start_date is provided
    interval (str): Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    exchange (str): Exchange name (e.g., 'NASDAQ', 'NYSE')
    screener (str): Screener to use (e.g., 'america', 'japan')
    output_format (str): Format to save data ('csv' or 'json')
    start_date (str): Start date for historical data in 'YYYY-MM-DD' format (e.g., '2020-01-01')
                      If provided, period parameter is ignored
    end_date (str): End date for historical data in 'YYYY-MM-DD' format (e.g., '2023-12-31')
                    If not provided and start_date is provided, defaults to current date
    
    Returns:
    str: Path to the created file
    """
    try:
        # Get historical data from Yahoo Finance with extra days for indicator calculation
        stock = yf.Ticker(ticker)
        
        # Determine how many extra data points we need to fetch for accurate indicator calculation
        # We need at least 200 data points for the longest indicator (SMA200)
        # Adjust the number of extra days based on the interval
        base_extra_points = 200
        
        # Calculate the extra days needed based on the interval
        if interval.endswith('m'):  # minutes
            minutes = int(interval[:-1])
            # For minute intervals, we need more days to get enough data points
            # Assuming market is open ~6.5 hours per day (390 minutes)
            extra_days = (base_extra_points * minutes) // 390 + 30  # Add buffer days
        elif interval.endswith('h'):  # hours
            hours = int(interval[:-1])
            # For hourly intervals, calculate based on ~6.5 trading hours per day
            extra_days = (base_extra_points * hours) // 6.5 + 15  # Add buffer days
        elif interval == '1d':  # daily
            extra_days = base_extra_points
        elif interval == '1wk':  # weekly
            extra_days = base_extra_points * 7
        elif interval == '1mo':  # monthly
            extra_days = base_extra_points * 30
        else:  # default
            extra_days = base_extra_points * 2
        
        if start_date is not None:
            # If start_date is provided, adjust it to get extra days for calculation
            original_start_date = start_date
            adjusted_start_date = (datetime.datetime.strptime(start_date, '%Y-%m-%d') - 
                                  datetime.timedelta(days=extra_days)).strftime('%Y-%m-%d')
            
            # Set end_date to current date if not provided
            if end_date is None:
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
                
            # Get historical data using adjusted date range
            hist_data = stock.history(start=adjusted_start_date, end=end_date, interval=interval)
            if hist_data.empty:
                print(f"Error: No data found for {ticker} between {adjusted_start_date} and {end_date}")
                return None
        else:
            # Use period parameter if start_date is not provided
            # For period-based requests, we'll fetch a longer period and trim later
            if period.endswith('d'):
                days = int(period[:-1]) + extra_days
                adjusted_period = f"{days}d"
            elif period.endswith('mo'):
                months = int(period[:-2]) + (extra_days // 30) + 1  # Convert extra days to months
                adjusted_period = f"{int(months)}mo"
            elif period.endswith('y'):
                years = int(period[:-1]) + (extra_days // 365) + 1  # Convert extra days to years
                adjusted_period = f"{int(years)}y"
            else:
                # For other periods like 'max', 'ytd', just add extra time
                adjusted_period = period
                
            hist_data = get_yahoo_historical_data(ticker, adjusted_period, interval)
            
        if hist_data is None:
            print(f"Error: Could not retrieve historical data for {ticker}")
            return None
            
        # Store the original date range for later trimming
        original_start_date = start_date
        
        # Get stock info from Yahoo Finance
        stock_info = get_yahoo_stock_info(ticker)
        if stock_info is None:
            print(f"Error: Could not retrieve stock info for {ticker}")
            return None
        
        # Calculate technical indicators directly on the historical data
        # This ensures each historical data point has its own indicator values
        # rather than applying current indicator values to all historical data
        
        # Create copies of the dataframe for technical indicator calculations
        df = hist_data.copy()
        
        # Calculate RSI (14-period)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['Oscillators_RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate RSI[1] (previous period RSI)
        df['Oscillators_RSI[1]'] = df['Oscillators_RSI'].shift(1)
        
        # Calculate Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Oscillators_STOCH.K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Oscillators_STOCH.D'] = df['Oscillators_STOCH.K'].rolling(window=3).mean()
        
        # Calculate CCI (Commodity Channel Index)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        moving_avg_20 = typical_price.rolling(window=20).mean()
        mean_deviation = abs(typical_price - moving_avg_20).rolling(window=20).mean()
        df['Oscillators_CCI'] = (typical_price - moving_avg_20) / (0.015 * mean_deviation)
        
        # Calculate MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['Oscillators_MACD.macd'] = ema_12 - ema_26
        df['Oscillators_MACD.signal'] = df['Oscillators_MACD.macd'].ewm(span=9, adjust=False).mean()
        
        # Calculate Moving Averages
        # EMA
        df['Moving Averages_EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['Moving Averages_EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['Moving Averages_EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['Moving Averages_EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['Moving Averages_EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
        df['Moving Averages_EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # SMA
        df['Moving Averages_SMA5'] = df['Close'].rolling(window=5).mean()
        df['Moving Averages_SMA10'] = df['Close'].rolling(window=10).mean()
        df['Moving Averages_SMA20'] = df['Close'].rolling(window=20).mean()
        df['Moving Averages_SMA50'] = df['Close'].rolling(window=50).mean()
        df['Moving Averages_SMA100'] = df['Close'].rolling(window=100).mean()
        df['Moving Averages_SMA200'] = df['Close'].rolling(window=200).mean()
        
        # Update the historical data with calculated indicators
        hist_data = df
        
        # Trim the data back to the original requested date range
        if original_start_date is not None:
            # If we had a specific start date, trim to that date
            hist_data = hist_data[hist_data.index >= original_start_date]
        elif period != adjusted_period:
            # If we used an adjusted period, trim the extra data points
            # Calculate the number of data points to keep based on the interval
            if period.endswith('d'):
                # For daily data, each data point is one day
                data_points = int(period[:-1])
            elif period.endswith('mo'):
                # For monthly data, approximate based on interval
                months = int(period[:-2])
                if interval.endswith('m'):  # minutes
                    minutes = int(interval[:-1])
                    # Approximate trading minutes in a month (21 trading days * 6.5 hours * 60 minutes)
                    data_points = int(months * 21 * 390 // minutes)
                elif interval.endswith('h'):  # hours
                    hours = int(interval[:-1])
                    # Approximate trading hours in a month (21 trading days * 6.5 hours)
                    data_points = int(months * 21 * 6.5 // hours)
                elif interval == '1d':
                    # Approximate trading days in a month
                    data_points = int(months * 21)
                elif interval == '1wk':
                    # Approximate weeks in a month
                    data_points = int(months * 4)
                else:  # Default to daily
                    data_points = int(months * 21)
            elif period.endswith('y'):
                # For yearly data, approximate based on interval
                years = int(period[:-1])
                if interval.endswith('m'):  # minutes
                    minutes = int(interval[:-1])
                    # Approximate trading minutes in a year (252 trading days * 6.5 hours * 60 minutes)
                    data_points = int(years * 252 * 390 // minutes)
                elif interval.endswith('h'):  # hours
                    hours = int(interval[:-1])
                    # Approximate trading hours in a year (252 trading days * 6.5 hours)
                    data_points = int(years * 252 * 6.5 // hours)
                elif interval == '1d':
                    # Trading days in a year
                    data_points = int(years * 252)
                elif interval == '1wk':
                    # Weeks in a year
                    data_points = int(years * 52)
                elif interval == '1mo':
                    # Months in a year
                    data_points = int(years * 12)
                else:  # Default to daily
                    data_points = int(years * 252)
            else:
                # Default to keeping all data
                data_points = len(hist_data)
                
            # Ensure we don't try to keep more data points than we have
            data_points = min(int(data_points), len(hist_data))
            
            # Keep only the requested number of data points
            if data_points > 0:
                hist_data = hist_data.iloc[-data_points:]
        
        # Remove rows with NaN values in indicator columns
        hist_data = hist_data.dropna()
        
        # Replace any remaining NaN values with 'N/A' string
        hist_data = hist_data.fillna('N/A')
        
        # Add some key stock info as columns
        key_info_fields = [
            'sector', 'industry', 'marketCap', 'trailingPE', 'forwardPE',
            'dividendYield', 'beta', 'fiftyDayAverage', 'twoHundredDayAverage'
        ]
        
        for field in key_info_fields:
            if field in stock_info:
                hist_data[f"Info_{field}"] = stock_info[field]
        
        # Round numerical values to 3 decimal places for readability
        for col in hist_data.select_dtypes(include=['float64']).columns:
            hist_data[col] = hist_data[col].round(3)
        
        # Create stockData directory if it doesn't exist
        stock_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stockData')
        if not os.path.exists(stock_data_dir):
            os.makedirs(stock_data_dir)
            print(f"Created directory: {stock_data_dir}")
        
        # Create filename based on ticker
        filename = f"{ticker.upper()}_data"
        
        # Save to file in the specified format
        if output_format.lower() == 'csv':
            file_path = os.path.join(stock_data_dir, f"{filename}.csv")
            hist_data.to_csv(file_path)
        else:  # json format
            file_path = os.path.join(stock_data_dir, f"{filename}.json")
            # Convert DataFrame to JSON
            hist_data_json = hist_data.reset_index().to_json(orient='records', date_format='iso')
            # Save to file
            with open(file_path, 'w') as f:
                f.write(hist_data_json)
        
        print(f"Data for {ticker} saved to {file_path}")
        return file_path
    
    except Exception as e:
        print(f"Error collecting and saving data for {ticker}: {e}")
        return None

def collect_and_save_backtest_data(ticker, period="5y", interval="1d", exchange="NASDAQ", screener="america", output_format="csv", start_date=None, end_date=None):
    """
    Collect historical stock data for backtesting purposes from Yahoo Finance and TradingView, 
    then save it to a file with 'backtest' prefix in the 'stockData' folder.
    
    Parameters:
    ticker (str): Stock symbol (e.g., 'AAPL', 'MSFT')
    period (str): Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                  Default is 5y to provide more historical data for backtesting
                  Note: This parameter is ignored if start_date is provided
    interval (str): Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    exchange (str): Exchange name (e.g., 'NASDAQ', 'NYSE')
    screener (str): Screener to use (e.g., 'america', 'japan')
    output_format (str): Format to save data ('csv' or 'json')
    start_date (str): Start date for historical data in 'YYYY-MM-DD' format (e.g., '2020-01-01')
                      If provided, period parameter is ignored
    end_date (str): End date for historical data in 'YYYY-MM-DD' format (e.g., '2023-12-31')
                    If not provided and start_date is provided, defaults to current date
    
    Returns:
    str: Path to the created file
    """
    try:
        # Get historical data from Yahoo Finance with extra days for indicator calculation
        stock = yf.Ticker(ticker)
        
        # Determine how many extra data points we need to fetch for accurate indicator calculation
        # We need at least 200 data points for the longest indicator (SMA200)
        # Adjust the number of extra days based on the interval
        base_extra_points = 200
        
        # Calculate the extra days needed based on the interval
        if interval.endswith('m'):  # minutes
            minutes = int(interval[:-1])
            # For minute intervals, we need more days to get enough data points
            # Assuming market is open ~6.5 hours per day (390 minutes)
            extra_days = (base_extra_points * minutes) // 390 + 30  # Add buffer days
        elif interval.endswith('h'):  # hours
            hours = int(interval[:-1])
            # For hourly intervals, calculate based on ~6.5 trading hours per day
            extra_days = (base_extra_points * hours) // 6.5 + 15  # Add buffer days
        elif interval == '1d':  # daily
            extra_days = base_extra_points
        elif interval == '1wk':  # weekly
            extra_days = base_extra_points * 7
        elif interval == '1mo':  # monthly
            extra_days = base_extra_points * 30
        else:  # default
            extra_days = base_extra_points * 2
        
        if start_date is not None:
            # If start_date is provided, adjust it to get extra days for calculation
            original_start_date = start_date
            adjusted_start_date = (datetime.datetime.strptime(start_date, '%Y-%m-%d') - 
                                  datetime.timedelta(days=extra_days)).strftime('%Y-%m-%d')
            
            # Set end_date to current date if not provided
            if end_date is None:
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
                
            # Get historical data using adjusted date range
            hist_data = stock.history(start=adjusted_start_date, end=end_date, interval=interval)
            if hist_data.empty:
                print(f"Error: No data found for {ticker} between {adjusted_start_date} and {end_date}")
                return None
        else:
            # Use period parameter if start_date is not provided
            # For period-based requests, we'll fetch a longer period and trim later
            if period.endswith('d'):
                days = int(period[:-1]) + extra_days
                adjusted_period = f"{days}d"
            elif period.endswith('mo'):
                months = int(period[:-2]) + (extra_days // 30) + 1  # Convert extra days to months
                adjusted_period = f"{int(months)}mo"
            elif period.endswith('y'):
                years = int(period[:-1]) + (extra_days // 365) + 1  # Convert extra days to years
                adjusted_period = f"{int(years)}y"
            else:
                # For other periods like 'max', 'ytd', just add extra time
                adjusted_period = period
                
            hist_data = get_yahoo_historical_data(ticker, adjusted_period, interval)
            
        if hist_data is None:
            print(f"Error: Could not retrieve historical data for {ticker}")
            return None
            
        # Store the original date range for later trimming
        original_start_date = start_date
        
        # Get stock info from Yahoo Finance
        stock_info = get_yahoo_stock_info(ticker)
        if stock_info is None:
            print(f"Error: Could not retrieve stock info for {ticker}")
            return None
        
        # Calculate technical indicators directly on the historical data
        # This ensures each historical data point has its own indicator values
        # rather than applying current indicator values to all historical data
        
        # Create copies of the dataframe for technical indicator calculations
        df = hist_data.copy()
        
        # Calculate RSI (14-period)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['Oscillators_RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate RSI[1] (previous period RSI)
        df['Oscillators_RSI[1]'] = df['Oscillators_RSI'].shift(1)
        
        # Calculate Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Oscillators_STOCH.K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Oscillators_STOCH.D'] = df['Oscillators_STOCH.K'].rolling(window=3).mean()
        
        # Calculate CCI (Commodity Channel Index)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        moving_avg_20 = typical_price.rolling(window=20).mean()
        mean_deviation = abs(typical_price - moving_avg_20).rolling(window=20).mean()
        df['Oscillators_CCI'] = (typical_price - moving_avg_20) / (0.015 * mean_deviation)
        
        # Calculate MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['Oscillators_MACD.macd'] = ema_12 - ema_26
        df['Oscillators_MACD.signal'] = df['Oscillators_MACD.macd'].ewm(span=9, adjust=False).mean()
        
        # Calculate Moving Averages
        # EMA
        df['Moving Averages_EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['Moving Averages_EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['Moving Averages_EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['Moving Averages_EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['Moving Averages_EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
        df['Moving Averages_EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # SMA
        df['Moving Averages_SMA5'] = df['Close'].rolling(window=5).mean()
        df['Moving Averages_SMA10'] = df['Close'].rolling(window=10).mean()
        df['Moving Averages_SMA20'] = df['Close'].rolling(window=20).mean()
        df['Moving Averages_SMA50'] = df['Close'].rolling(window=50).mean()
        df['Moving Averages_SMA100'] = df['Close'].rolling(window=100).mean()
        df['Moving Averages_SMA200'] = df['Close'].rolling(window=200).mean()
        
        # Update the historical data with calculated indicators
        hist_data = df
        
        # Trim the data back to the original requested date range
        if original_start_date is not None:
            # If we had a specific start date, trim to that date
            hist_data = hist_data[hist_data.index >= original_start_date]
        elif period != adjusted_period:
            # If we used an adjusted period, trim the extra data points
            # Calculate the number of data points to keep based on the interval
            if period.endswith('d'):
                # For daily data, each data point is one day
                data_points = int(period[:-1])
            elif period.endswith('mo'):
                # For monthly data, approximate based on interval
                months = int(period[:-2])
                if interval.endswith('m'):  # minutes
                    minutes = int(interval[:-1])
                    # Approximate trading minutes in a month (21 trading days * 6.5 hours * 60 minutes)
                    data_points = int(months * 21 * 390 // minutes)
                elif interval.endswith('h'):  # hours
                    hours = int(interval[:-1])
                    # Approximate trading hours in a month (21 trading days * 6.5 hours)
                    data_points = int(months * 21 * 6.5 // hours)
                elif interval == '1d':
                    # Approximate trading days in a month
                    data_points = int(months * 21)
                elif interval == '1wk':
                    # Approximate weeks in a month
                    data_points = int(months * 4)
                else:  # Default to daily
                    data_points = int(months * 21)
            elif period.endswith('y'):
                # For yearly data, approximate based on interval
                years = int(period[:-1])
                if interval.endswith('m'):  # minutes
                    minutes = int(interval[:-1])
                    # Approximate trading minutes in a year (252 trading days * 6.5 hours * 60 minutes)
                    data_points = int(years * 252 * 390 // minutes)
                elif interval.endswith('h'):  # hours
                    hours = int(interval[:-1])
                    # Approximate trading hours in a year (252 trading days * 6.5 hours)
                    data_points = int(years * 252 * 6.5 // hours)
                elif interval == '1d':
                    # Trading days in a year
                    data_points = int(years * 252)
                elif interval == '1wk':
                    # Weeks in a year
                    data_points = int(years * 52)
                elif interval == '1mo':
                    # Months in a year
                    data_points = int(years * 12)
                else:  # Default to daily
                    data_points = int(years * 252)
            else:
                # Default to keeping all data
                data_points = len(hist_data)
                
            # Ensure we don't try to keep more data points than we have
            data_points = min(int(data_points), len(hist_data))
            
            # Keep only the requested number of data points
            if data_points > 0:
                hist_data = hist_data.iloc[-data_points:]
        
        # Remove rows with NaN values in indicator columns
        hist_data = hist_data.dropna()
        
        # Replace any remaining NaN values with 'N/A' string
        hist_data = hist_data.fillna('N/A')
        
        # Add some key stock info as columns
        key_info_fields = [
            'sector', 'industry', 'marketCap', 'trailingPE', 'forwardPE',
            'dividendYield', 'beta', 'fiftyDayAverage', 'twoHundredDayAverage'
        ]
        
        for field in key_info_fields:
            if field in stock_info:
                hist_data[f"Info_{field}"] = stock_info[field]
        
        # Round numerical values to 3 decimal places for readability
        for col in hist_data.select_dtypes(include=['float64']).columns:
            hist_data[col] = hist_data[col].round(3)
        
        # Create stockData directory if it doesn't exist
        stock_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stockData')
        if not os.path.exists(stock_data_dir):
            os.makedirs(stock_data_dir)
            print(f"Created directory: {stock_data_dir}")
        
        # Create filename based on ticker with 'backtest' prefix
        filename = f"backtest{ticker.upper()}_data"
        
        # Save to file in the specified format
        if output_format.lower() == 'csv':
            file_path = os.path.join(stock_data_dir, f"{filename}.csv")
            hist_data.to_csv(file_path)
        else:  # json format
            file_path = os.path.join(stock_data_dir, f"{filename}.json")
            # Convert DataFrame to JSON
            hist_data_json = hist_data.reset_index().to_json(orient='records', date_format='iso')
            # Save to file
            with open(file_path, 'w') as f:
                f.write(hist_data_json)
        
        print(f"Backtest data for {ticker} saved to {file_path}")
        return file_path
    
    except Exception as e:
        print(f"Error collecting and saving backtest data for {ticker}: {e}")
        return None

# Run if script is executed directly
if __name__ == "__main__":
    print("Stock Data Retrieval and Processing Module")
    print("To use this module, first install required packages:")
    print("pip install yfinance pandas")
    print("For TradingView functionality: pip install tradingview-ta")
    
    # Example usage:
    # collect_and_save_stock_data('AAPL', period='1y', interval='1d', output_format='csv')
    # collect_and_save_backtest_data('AAPL', period='5y', interval='1d', output_format='csv')