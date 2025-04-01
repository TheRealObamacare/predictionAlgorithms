# Stock Data Prediction Algorithms

This repository contains tools for retrieving and analyzing stock data from Yahoo Finance and TradingView.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For TradingView technical indicators functionality, uncomment the last line in requirements.txt before installing.

## Usage

### Yahoo Finance Data Retrieval

```python
from stock_data import get_yahoo_historical_data, get_yahoo_stock_info, get_yahoo_financials

# Get historical stock data
aapl_data = get_yahoo_historical_data('AAPL', period='1y', interval='1d')
print(aapl_data.head())

# Get stock information
msft_info = get_yahoo_stock_info('MSFT')
print(msft_info['shortName'], msft_info['currentPrice'])

# Get financial statements
aapl_financials = get_yahoo_financials('AAPL')
print(aapl_financials['income_statement'].head())
```

### TradingView Technical Indicators

For TradingView functionality, it's recommended to install the tradingview-ta package:

```bash
pip install tradingview-ta
```

Then you can use:

```python
from stock_data import get_tradingview_technical_indicators

# Get technical indicators
tv_data = get_tradingview_technical_indicators('AAPL', exchange='NASDAQ')
print(tv_data)
```

## Example Script

The module includes an example usage function that can be run directly:

```python
from stock_data import example_usage

example_usage()
```

Or run the module directly:

```bash
python stock_data.py
```

## Notes

- Yahoo Finance data is retrieved using the yfinance package
- TradingView doesn't have an official API, so the implementation uses a simplified approach
- For production use with TradingView data, consider using the tradingview-ta library