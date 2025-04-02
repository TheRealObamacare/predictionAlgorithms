from stock_data import collect_and_save_backtest_data

# Example of collecting historical data for backtesting
# This will create a file named 'backtestQQQ_data.csv' in the stockData folder
# with 5 years of historical data at daily intervals
# collect_and_save_backtest_data('QQQ', period='5y', interval='1d', output_format='csv')

# NEW FEATURE: You can now specify exact date ranges for backtesting
# This will collect data from January 1, 2020 to December 31, 2022
collect_and_save_backtest_data('QQQ', interval='1d', output_format='csv', start_date='2020-01-01', end_date='2022-12-31')

# You can also specify just the start date, and it will collect data up to the current date
# collect_and_save_backtest_data('QQQ', interval='1d', output_format='csv', start_date='2021-01-01')