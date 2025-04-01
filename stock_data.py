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
def collect_and_save_stock_data(ticker, period="1y", interval="1d", exchange="NASDAQ", screener="america", output_format="csv"):
    """
    Collect stock data from Yahoo Finance and TradingView, then save it to a file named after the ticker
    in the 'stockData' folder.
    
    Parameters:
    ticker (str): Stock symbol (e.g., 'AAPL', 'MSFT')
    period (str): Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    interval (str): Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    exchange (str): Exchange name (e.g., 'NASDAQ', 'NYSE')
    screener (str): Screener to use (e.g., 'america', 'japan')
    output_format (str): Format to save data ('csv' or 'json')
    
    Returns:
    str: Path to the created file
    """
    try:
        # Get historical data from Yahoo Finance
        hist_data = get_yahoo_historical_data(ticker, period, interval)
        if hist_data is None:
            print(f"Error: Could not retrieve historical data for {ticker}")
            return None
        
        # Get stock info from Yahoo Finance
        stock_info = get_yahoo_stock_info(ticker)
        if stock_info is None:
            print(f"Error: Could not retrieve stock info for {ticker}")
            return None
        
        # Get technical indicators from TradingView
        try:
            tech_indicators = get_tradingview_technical_indicators(ticker, exchange, screener, interval)
        except Exception as e:
            print(f"Warning: Could not retrieve TradingView data for {ticker}: {e}")
            tech_indicators = None
        
        # Format the data for prediction algorithms
        # Add technical indicators as columns to the historical data
        if tech_indicators is not None:
            # Extract key technical indicators
            for category in ['Oscillators', 'Moving Averages']:
                for key, value in tech_indicators[category].items():
                    if key not in ['RECOMMENDATION', 'BUY', 'SELL', 'NEUTRAL']:
                        try:
                            # Convert to float if possible
                            hist_data[f"{category}_{key}"] = float(value)
                        except (ValueError, TypeError):
                            # Keep as string if not convertible to float
                            hist_data[f"{category}_{key}"] = value
        
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

# Run if script is executed directly
if __name__ == "__main__":
    print("Stock Data Retrieval and Processing Module")
    print("To use this module, first install required packages:")
    print("pip install yfinance pandas")
    print("For TradingView functionality: pip install tradingview-ta")
    
    # Example usage:
    # collect_and_save_stock_data('AAPL', period='1y', interval='1d', output_format='csv')