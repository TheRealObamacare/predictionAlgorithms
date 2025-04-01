# Example script for stock data retrieval
# This script demonstrates how to use the stock_data module

from stock_data import (
    get_yahoo_historical_data,
    get_yahoo_stock_info,
    get_yahoo_financials,
    get_tradingview_technical_indicators
)
import pandas as pd
import json

def main():
    print("\n===== Stock Data Retrieval Example =====\n")
    
    # Define stock symbols to analyze
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Example 1: Get historical data for multiple stocks
    print("\n----- Example 1: Historical Data -----")
    for symbol in symbols:
        print(f"\nFetching historical data for {symbol}...")
        hist_data = get_yahoo_historical_data(symbol, period="1mo", interval="1d")
        if hist_data is not None:
            print(f"Recent price data for {symbol}:")
            # Format the DataFrame to display numerical values with 3 decimal places
            pd.set_option('display.float_format', '{:.3f}'.format)
            print(hist_data.tail(3))
            # Reset display format to default
            pd.reset_option('display.float_format')
    
    # Example 2: Get current stock information
    print("\n\n----- Example 2: Stock Information -----")
    for symbol in symbols[:1]:  # Just use the first symbol for brevity
        print(f"\nFetching stock information for {symbol}...")
        info = get_yahoo_stock_info(symbol)
        if info is not None:
            # Extract and display key information
            key_info = {
                'Name': info.get('shortName', 'N/A'),
                'Sector': info.get('sector', 'N/A'),
                'Industry': info.get('industry', 'N/A'),
                'Market Cap': info.get('marketCap', 'N/A'),
                'Current Price': info.get('currentPrice', 'N/A'),
                'P/E Ratio': info.get('trailingPE', 'N/A'),
                '52-Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
                '52-Week Low': info.get('fiftyTwoWeekLow', 'N/A')
            }
            for k, v in key_info.items():
                # Format numerical values to 3 decimal places
                if isinstance(v, (int, float)) and k not in ['Market Cap']:
                    print(f"{k}: {v:.3f}")
                elif k == 'Market Cap' and isinstance(v, (int, float)):
                    # Format market cap in billions for readability
                    print(f"{k}: {v/1000000000:.3f}B")
                else:
                    print(f"{k}: {v}")
    
    # Example 3: Get financial data
    print("\n\n----- Example 3: Financial Data -----")
    symbol = symbols[0]  # Use the first symbol
    print(f"\nFetching financial data for {symbol}...")
    financials = get_yahoo_financials(symbol)
    if financials is not None:
        # Display recent income statement data
        print(f"\nRecent Income Statement for {symbol}:")
        income_stmt = financials['income_statement']
        if not income_stmt.empty:
            # Display the most recent quarter with 3 decimal places
            pd.set_option('display.float_format', '{:.3f}'.format)
            print(income_stmt.iloc[:, 0].head(5))
            # Reset display format to default
            pd.reset_option('display.float_format')
    
    # Example 4: TradingView Technical Indicators
    print("\n\n----- Example 4: TradingView Technical Indicators -----")
    symbol = symbols[0]  # Use the first symbol
    print(f"\nFetching technical indicators for {symbol}...")
    tv_data = get_tradingview_technical_indicators(symbol, exchange="NASDAQ", screener="america", interval="1d")
    if tv_data is not None:
        print("\nTradingView Analysis Summary:")
        print(f"Overall Recommendation: {tv_data['Summary']['RECOMMENDATION']}")
        
        # Format numerical values to 3 decimal places
        osc = tv_data['Oscillators']
        print(f"Oscillators: {osc['RECOMMENDATION']} (Buy: {osc['BUY']:.3f}, Sell: {osc['SELL']:.3f}, Neutral: {osc['NEUTRAL']:.3f})")
        
        ma = tv_data['Moving Averages']
        print(f"Moving Averages: {ma['RECOMMENDATION']} (Buy: {ma['BUY']:.3f}, Sell: {ma['SELL']:.3f}, Neutral: {ma['NEUTRAL']:.3f})")
        
        # Format numerical values in the detailed indicators
        for category in tv_data:
            if isinstance(tv_data[category], dict):
                for key, value in tv_data[category].items():
                    if isinstance(value, (int, float)) and key not in ['BUY', 'SELL', 'NEUTRAL']:
                        tv_data[category][key] = f"{value:.3f}"
        
        print("\nDetailed Technical Indicators:")
        print(json.dumps(tv_data, indent=2))

if __name__ == "__main__":
    print("This example demonstrates how to retrieve stock data using Yahoo Finance and TradingView.")
    print("Make sure you have installed the required packages:")
    print("pip install -r requirements.txt")
    
    try:
        main()
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease make sure you have installed all required packages:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"\nAn error occurred: {e}")