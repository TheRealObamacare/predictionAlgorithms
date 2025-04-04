# LSTM and Neural Network Stock Prediction Module using PyTorch
# This module provides functions to create, train, and test LSTM and Neural Network models for stock prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import datetime

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set device with priority for GPU
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda:0")
    print(f"GPU is available! Using: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    # Print GPU memory information if available
    try:
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"GPU Memory Available: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB used")
    except:
        pass
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU instead.")

print(f"PyTorch device: {device}")

# Force CUDA tensor operations to synchronize for accurate timing
torch.cuda.synchronize() if torch.cuda.is_available() else None

def load_stock_data(file_path):
    """
    Load stock data from a CSV file
    
    Parameters:
    file_path (str): Path to the CSV file containing stock data
    
    Returns:
    pandas.DataFrame: Stock data
    """
    try:
        data = pd.read_csv(file_path, index_col='Datetime', parse_dates=True)
        print(f"Loaded data with shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading stock data: {e}")
        return None

def prepare_data_for_lstm(data, target_col='Close', feature_cols=None, sequence_length=60, train_split=0.8, date_split=None):
    """
    Prepare data for LSTM model training and testing
    
    Parameters:
    data (pandas.DataFrame): Stock data
    target_col (str): Column name for the target variable (default: 'Close')
    feature_cols (list): List of column names for features (default: None, uses all numeric columns)
    sequence_length (int): Number of time steps to look back (default: 60)
    train_split (float): Proportion of data to use for training (default: 0.8)
    date_split (str): Date to split training and testing data (format: 'YYYY-MM-DD')
                     If provided, train_split is ignored
    
    Returns:
    tuple: (X_train, y_train, X_test, y_test, scaler, cols, target_idx)
    """
    try:
        # Select features
        if feature_cols is None:
            # Use all numeric columns except the target as features
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            feature_cols = [col for col in numeric_cols if col != target_col]
        
        # Add target column to features for scaling
        cols = feature_cols + [target_col]
        data_subset = data[cols].copy()
        
        # Handle missing values
        data_subset = data_subset.dropna()
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_subset)
        
        # Create sequences
        X = []
        y = []
        dates = []  # Store corresponding dates for each sequence
        target_idx = cols.index(target_col)
        
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, target_idx])
            dates.append(data_subset.index[i])  # Store the date for this sequence
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into training and testing sets
        if date_split is not None:
            # Convert date_split to datetime
            split_date = pd.to_datetime(date_split)
            
            # Create masks for train and test sets based on date
            train_mask = [date < split_date for date in dates]
            test_mask = [date >= split_date for date in dates]
            
            # Split data based on date
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]
            
            print(f"Date-based split at {date_split}: {sum(train_mask)} training samples, {sum(test_mask)} testing samples")
        else:
            # Use proportion-based split
            train_size = int(len(X) * train_split)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)
        
        print(f"Prepared data with shapes: X_train {X_train.shape}, y_train {y_train.shape}, X_test {X_test.shape}, y_test {y_test.shape}")
        
        return X_train, y_train, X_test, y_test, scaler, cols, target_idx
    
    except Exception as e:
        print(f"Error preparing data for LSTM: {e}")
        return None, None, None, None, None, None, None

class LSTMModel(nn.Module):
    """
    LSTM model for stock prediction
    """
    def __init__(self, input_dim, hidden_dim=50, num_layers=2, dropout=0.2, output_dim=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device  # Store device reference
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, 25)
        self.fc2 = nn.Linear(25, output_dim)
        
    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Initialize hidden state with zeros directly on the correct device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=self.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_dim)
        
        # Get the output from the last time step
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Apply fully connected layers
        out = self.fc(out)
        out = self.fc2(out)
        
        return out

class NeuralNetworkModel(nn.Module):
    """
    Neural Network model for stock prediction
    """
    def __init__(self, input_dim, layers=[64, 32], dropout=0.2, output_dim=1):
        super(NeuralNetworkModel, self).__init__()
        self.device = device  # Store device reference
        
        # Create a list to hold all layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Flatten())
        
        # Hidden layers
        current_dim = input_dim
        for layer_dim in layers:
            self.layers.append(nn.Linear(current_dim, layer_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            current_dim = layer_dim
        
        # Output layer
        self.layers.append(nn.Linear(current_dim, output_dim))
        
    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        for layer in self.layers:
            x = layer(x)
        return x

def create_lstm_model(input_shape, units=50, dropout_rate=0.2):
    """
    Create an LSTM model for stock prediction
    
    Parameters:
    input_shape (tuple): Shape of input data (sequence_length, num_features)
    units (int): Number of LSTM units (default: 50)
    dropout_rate (float): Dropout rate for regularization (default: 0.2)
    
    Returns:
    LSTMModel: LSTM model
    """
    input_dim = input_shape[1]  # Number of features
    model = LSTMModel(input_dim=input_dim, hidden_dim=units, dropout=dropout_rate)
    model = model.to(device)
    return model

def create_nn_model(input_shape, layers=[64, 32], dropout_rate=0.2):
    """
    Create a Neural Network model for stock prediction
    
    Parameters:
    input_shape (tuple): Shape of input data (sequence_length, num_features)
    layers (list): List of units in each hidden layer (default: [64, 32])
    dropout_rate (float): Dropout rate for regularization (default: 0.2)
    
    Returns:
    NeuralNetworkModel: Neural Network model
    """
    input_dim = input_shape[0] * input_shape[1]  # Flattened input dimension
    model = NeuralNetworkModel(input_dim=input_dim, layers=layers, dropout=dropout_rate)
    model = model.to(device)
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, patience=10):
    """
    Train a model with early stopping
    
    Parameters:
    model (nn.Module): Model to train
    X_train (torch.Tensor): Training features
    y_train (torch.Tensor): Training targets
    epochs (int): Maximum number of epochs (default: 50)
    batch_size (int): Batch size (default: 32)
    validation_split (float): Proportion of training data to use for validation (default: 0.1)
    patience (int): Number of epochs with no improvement after which training will be stopped (default: 10)
    
    Returns:
    tuple: (model, history)
    """
    # Log which device is being used for training
    print(f"Training model on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    # Split data into training and validation sets
    val_size = int(len(X_train) * validation_split)
    train_size = len(X_train) - val_size
    
    X_train_final, X_val = X_train[:train_size], X_train[train_size:]
    y_train_final, y_val = y_train[:train_size], y_train[train_size:]
    
    # Ensure data is on the correct device
    X_train_final = X_train_final.to(device)
    y_train_final = y_train_final.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_final, y_train_final)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters())
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            # Ensure data is on the correct device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Ensure data is on the correct device
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        # Store losses
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    model.load_state_dict(best_model)
    
    # Synchronize CUDA operations if using GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    return model, history

def evaluate_model(model, X_test, y_test, scaler, cols, target_idx):
    """
    Evaluate model performance on test data
    
    Parameters:
    model (nn.Module): Trained model
    X_test (torch.Tensor): Test features
    y_test (torch.Tensor): Test targets
    scaler (sklearn.preprocessing.MinMaxScaler): Scaler used to normalize data
    cols (list): List of column names
    target_idx (int): Index of target column
    
    Returns:
    tuple: (metrics, y_true, y_pred)
    """
    # Log which device is being used for evaluation
    print(f"Evaluating model on: {device}")
    
    # Ensure model and data are on the correct device
    model = model.to(device)
    X_test = X_test.to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        # Synchronize before prediction if using GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # Make predictions and move to CPU for numpy conversion
        y_pred_scaled = model(X_test).cpu().numpy()
    
    # Convert targets to numpy
    y_test = y_test.cpu().numpy()
    
    # Inverse transform predictions
    # Create a dummy array with the same shape as the original data
    dummy = np.zeros((len(y_pred_scaled), len(cols)))
    dummy[:, target_idx] = y_pred_scaled.flatten()
    y_pred = scaler.inverse_transform(dummy)[:, target_idx]
    
    # Inverse transform actual values
    dummy = np.zeros((len(y_test), len(cols)))
    dummy[:, target_idx] = y_test
    y_true = scaler.inverse_transform(dummy)[:, target_idx]
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics, y_true, y_pred

def plot_predictions(y_true, y_pred, title="Model Predictions vs Actual"):
    """
    Plot model predictions against actual values
    
    Parameters:
    y_true (numpy.ndarray): Actual values
    y_pred (numpy.ndarray): Predicted values
    title (str): Plot title (default: "Model Predictions vs Actual")
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(plots_dir, f"{title.replace(' ', '_')}_{timestamp}.png"))
    plt.show()

def backtest_strategy(data, predictions, initial_investment=10000, commission=0.001, ticker=None):
    """
    Backtest a trading strategy based on model predictions
    
    Parameters:
    data (pandas.DataFrame): Original stock data
    predictions (numpy.ndarray): Model predictions
    initial_investment (float): Initial investment amount (default: 10000)
    commission (float): Commission rate per trade (default: 0.001)
    ticker (str): Stock ticker symbol for fetching dividend data (default: None)
    
    Returns:
    pandas.DataFrame: Backtest results
    """
    # Create a copy of the data for the backtest period
    backtest_data = data.iloc[-len(predictions):].copy()
    
    # Add predictions to the data
    backtest_data['Predicted_Close'] = predictions
    
    # Calculate daily returns
    backtest_data['Actual_Return'] = backtest_data['Close'].pct_change()
    backtest_data['Predicted_Return'] = backtest_data['Predicted_Close'].pct_change()
    
    # Generate trading signals (1 for buy, -1 for sell, 0 for hold)
    backtest_data['Signal'] = 0
    backtest_data.loc[backtest_data['Predicted_Return'] > 0, 'Signal'] = 1  # Buy signal
    backtest_data.loc[backtest_data['Predicted_Return'] < 0, 'Signal'] = -1  # Sell signal
    
    # Calculate strategy returns
    backtest_data['Strategy_Return'] = backtest_data['Signal'].shift(1) * backtest_data['Actual_Return']
    
    # Account for commission costs on trades
    backtest_data['Trade'] = backtest_data['Signal'].diff().abs()
    backtest_data['Commission'] = backtest_data['Trade'] * commission
    backtest_data['Strategy_Return'] = backtest_data['Strategy_Return'] - backtest_data['Commission']
    
    # Get dividend data if ticker is provided
    if ticker:
        try:
            from stock_data import get_yahoo_dividend_data
            
            # Get start and end dates from backtest data
            start_date = backtest_data.index[0].strftime('%Y-%m-%d')
            end_date = backtest_data.index[-1].strftime('%Y-%m-%d')
            
            # Fetch dividend data
            dividend_data = get_yahoo_dividend_data(ticker, start_date, end_date)
            
            if not dividend_data.empty:
                # Add dividend column to backtest data (initialize with zeros)
                backtest_data['Dividend_Yield'] = 0.0
                
                # For each dividend date, calculate the dividend yield and add to that day's return
                for date, row in dividend_data.iterrows():
                    # Find the closest trading day on or before the dividend date
                    dividend_date = date.strftime('%Y-%m-%d')
                    closest_dates = backtest_data.index[backtest_data.index <= dividend_date]
                    
                    if len(closest_dates) > 0:
                        closest_date = closest_dates[-1]
                        # Calculate dividend yield (dividend amount / closing price)
                        dividend_yield = row['Dividend'] / backtest_data.loc[closest_date, 'Close']
                        backtest_data.loc[closest_date, 'Dividend_Yield'] = dividend_yield
                        
                        # Print dividend information
                        print(f"Dividend of ${row['Dividend']:.4f} on {dividend_date} (yield: {dividend_yield:.4%})")
                
                # Add dividend yields to returns
                backtest_data['Actual_Return'] = backtest_data['Actual_Return'] + backtest_data['Dividend_Yield']
                
                # For strategy returns, only add dividends when holding long position (Signal = 1)
                # When Signal is 1, we're long the stock and receive dividends
                # When Signal is -1, we're short the stock and pay dividends
                # When Signal is 0, we're in cash and receive no dividends
                backtest_data['Strategy_Dividend'] = backtest_data['Dividend_Yield'] * backtest_data['Signal'].shift(1)
                backtest_data['Strategy_Return'] = backtest_data['Strategy_Return'] + backtest_data['Strategy_Dividend']
                
                print(f"Incorporated {len(dividend_data)} dividends into return calculations")
        except Exception as e:
            print(f"Error incorporating dividends: {e}")
    
    # Calculate cumulative returns
    backtest_data['Cumulative_Actual_Return'] = (1 + backtest_data['Actual_Return']).cumprod()
    backtest_data['Cumulative_Strategy_Return'] = (1 + backtest_data['Strategy_Return']).cumprod()
    
    # Calculate portfolio values
    backtest_data['Buy_Hold_Value'] = initial_investment * backtest_data['Cumulative_Actual_Return']
    backtest_data['Strategy_Value'] = initial_investment * backtest_data['Cumulative_Strategy_Return']
    
    # Calculate performance metrics
    buy_hold_return = backtest_data['Cumulative_Actual_Return'].iloc[-1] - 1
    strategy_return = backtest_data['Cumulative_Strategy_Return'].iloc[-1] - 1
    
    buy_hold_annual_return = (1 + buy_hold_return) ** (252 / len(backtest_data)) - 1
    strategy_annual_return = (1 + strategy_return) ** (252 / len(backtest_data)) - 1
    
    buy_hold_sharpe = np.sqrt(252) * backtest_data['Actual_Return'].mean() / backtest_data['Actual_Return'].std()
    strategy_sharpe = np.sqrt(252) * backtest_data['Strategy_Return'].mean() / backtest_data['Strategy_Return'].std()
    
    # Print performance summary
    print("\nBacktest Performance Summary:")
    print(f"Period: {backtest_data.index[0]} to {backtest_data.index[-1]}")
    print(f"Initial Investment: ${initial_investment:.2f}")
    print(f"Buy & Hold Final Value: ${backtest_data['Buy_Hold_Value'].iloc[-1]:.2f}")
    print(f"Strategy Final Value: ${backtest_data['Strategy_Value'].iloc[-1]:.2f}")
    print(f"Buy & Hold Total Return: {buy_hold_return:.2%}")
    print(f"Strategy Total Return: {strategy_return:.2%}")
    print(f"Buy & Hold Annualized Return: {buy_hold_annual_return:.2%}")
    print(f"Strategy Annualized Return: {strategy_annual_return:.2%}")
    print(f"Buy & Hold Sharpe Ratio: {buy_hold_sharpe:.4f}")
    print(f"Strategy Sharpe Ratio: {strategy_sharpe:.4f}")
    
    return backtest_data

def plot_backtest_results(backtest_data, title="Trading Strategy Backtest Results"):
    """
    Plot backtest results
    
    Parameters:
    backtest_data (pandas.DataFrame): Backtest results
    title (str): Plot title (default: "Trading Strategy Backtest Results")
    """
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Portfolio Values
    plt.subplot(2, 1, 1)
    plt.plot(backtest_data['Buy_Hold_Value'], label='Buy & Hold')
    plt.plot(backtest_data['Strategy_Value'], label='Strategy')
    plt.title('Portfolio Value Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Cumulative Returns
    plt.subplot(2, 1, 2)
    plt.plot(backtest_data['Cumulative_Actual_Return'], label='Buy & Hold')
    plt.plot(backtest_data['Cumulative_Strategy_Return'], label='Strategy')
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(plots_dir, f"{title.replace(' ', '_')}_{timestamp}.png"))
    plt.show()

def run_lstm_prediction(ticker=None, data_file=None, feature_cols=None, sequence_length=60, train_split=0.8, 
                       lstm_units=50, dropout_rate=0.2, epochs=50, batch_size=32, date_split=None):
    """
    Run the complete LSTM prediction workflow
    
    Parameters:
    ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT'). If provided, data will be fetched for this ticker.
    data_file (str): Path to the CSV file containing stock data. Used if ticker is None.
    feature_cols (list): List of column names for features (default: None)
    sequence_length (int): Number of time steps to look back (default: 60)
    train_split (float): Proportion of data to use for training (default: 0.8)
    lstm_units (int): Number of LSTM units (default: 50)
    dropout_rate (float): Dropout rate for regularization (default: 0.2)
    epochs (int): Maximum number of epochs (default: 50)
    batch_size (int): Batch size (default: 32)
    date_split (str): Date to split training and testing data (format: 'YYYY-MM-DD')
                      If None and ticker is provided, will use 1 year ago as the split date
    
    Returns:
    tuple: (model, backtest_data)
    """
    # Load or fetch data
    if ticker is not None:
        # Import the stock_data module
        from stock_data import collect_and_save_stock_data
        import datetime
        
        print(f"\nFetching data for ticker: {ticker}")
        
        # If date_split is not provided, use 1 year ago as the default split date
        if date_split is None:
            # Calculate date 1 year ago from today
            today = datetime.datetime.now()
            one_year_ago = (today - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
            date_split = one_year_ago
            print(f"Using default date split: {date_split} (1 year ago)")
        
        # Fetch all historical data
        data_file = collect_and_save_stock_data(
            ticker, 
            period="max",  # Get maximum available history
            interval="1d",  # Daily data
            output_format="csv"
        )
        
        if data_file is None:
            print(f"Error: Failed to fetch data for {ticker}")
            return None, None
    
    # Load data from file
    data = load_stock_data(data_file)
    if data is None:
        return None, None
    
    # Prepare data with date-based split if provided
    X_train, y_train, X_test, y_test, scaler, cols, target_idx = prepare_data_for_lstm(
        data, feature_cols=feature_cols, sequence_length=sequence_length, 
        train_split=train_split, date_split=date_split
    )
    
    if X_train is None:
        return None, None
    
    # Create and train LSTM model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape, units=lstm_units, dropout_rate=dropout_rate)
    model, history = train_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    # Evaluate model
    metrics, y_true, y_pred = evaluate_model(model, X_test, y_test, scaler, cols, target_idx)
    
    # Plot predictions
    title = f"{ticker} LSTM Model Predictions vs Actual" if ticker else "LSTM Model Predictions vs Actual"
    plot_predictions(y_true, y_pred, title=title)
    
    # Backtest strategy
    backtest_data = backtest_strategy(data, y_pred)
    
    # Plot backtest results
    title = f"{ticker} LSTM Trading Strategy Backtest Results" if ticker else "LSTM Trading Strategy Backtest Results"
    plot_backtest_results(backtest_data, title=title)
    
    return model, backtest_data

def run_nn_prediction(ticker=None, data_file=None, feature_cols=None, sequence_length=60, train_split=0.8, 
                     layers=[64, 32], dropout_rate=0.2, epochs=50, batch_size=32, date_split=None):
    """
    Run the complete Neural Network prediction workflow
    
    Parameters:
    ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT'). If provided, data will be fetched for this ticker.
    data_file (str): Path to the CSV file containing stock data. Used if ticker is None.
    feature_cols (list): List of column names for features (default: None)
    sequence_length (int): Number of time steps to look back (default: 60)
    train_split (float): Proportion of data to use for training (default: 0.8)
    layers (list): List of units in each hidden layer (default: [64, 32])
    dropout_rate (float): Dropout rate for regularization (default: 0.2)
    epochs (int): Maximum number of epochs (default: 50)
    batch_size (int): Batch size (default: 32)
    date_split (str): Date to split training and testing data (format: 'YYYY-MM-DD')
                       If None and ticker is provided, will use 1 year ago as the split date
    
    Returns:
    tuple: (model, backtest_data)
    """
    # Load or fetch data
    if ticker is not None:
        # Import the stock_data module
        from stock_data import collect_and_save_stock_data
        import datetime
        
        print(f"\nFetching data for ticker: {ticker}")
        
        # If date_split is not provided, use 1 year ago as the default split date
        if date_split is None:
            # Calculate date 1 year ago from today
            today = datetime.datetime.now()
            one_year_ago = (today - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
            date_split = one_year_ago
            print(f"Using default date split: {date_split} (1 year ago)")
        
        # Fetch all historical data
        data_file = collect_and_save_stock_data(
            ticker, 
            period="max",  # Get maximum available history
            interval="1d",  # Daily data
            output_format="csv"
        )
        
        if data_file is None:
            print(f"Error: Failed to fetch data for {ticker}")
            return None, None
    
    # Load data from file
    data = load_stock_data(data_file)
    if data is None:
        return None, None
    
    # Prepare data with date-based split if provided
    X_train, y_train, X_test, y_test, scaler, cols, target_idx = prepare_data_for_lstm(
        data, feature_cols=feature_cols, sequence_length=sequence_length, 
        train_split=train_split, date_split=date_split
    )
    
    if X_train is None:
        return None, None
    
    # Create and train Neural Network model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_nn_model(input_shape, layers=layers, dropout_rate=dropout_rate)
    model, history = train_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    # Evaluate model
    metrics, y_true, y_pred = evaluate_model(model, X_test, y_test, scaler, cols, target_idx)
    
    # Plot predictions
    title = f"{ticker} Neural Network Model Predictions vs Actual" if ticker else "Neural Network Model Predictions vs Actual"
    plot_predictions(y_true, y_pred, title=title)
    
    # Backtest strategy
    backtest_data = backtest_strategy(data, y_pred)
    
    # Plot backtest results
    title = f"{ticker} Neural Network Trading Strategy Backtest Results" if ticker else "Neural Network Trading Strategy Backtest Results"
    plot_backtest_results(backtest_data, title=title)
    
    return model, backtest_data

def compare_models_with_buy_hold(lstm_backtest, nn_backtest):
    """
    Compare LSTM and Neural Network models with Buy & Hold strategy
    
    Parameters:
    lstm_backtest (pandas.DataFrame): LSTM backtest results
    nn_backtest (pandas.DataFrame): Neural Network backtest results
    """
    if lstm_backtest is None or nn_backtest is None:
        print("Cannot compare models: backtest data is missing")
        return
    
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    
    # Plot cumulative returns
    plt.plot(lstm_backtest['Cumulative_Actual_Return'], label='Buy & Hold')
    plt.plot(lstm_backtest['Cumulative_Strategy_Return'], label='LSTM Strategy')
    plt.plot(nn_backtest['Cumulative_Strategy_Return'], label='NN Strategy')
    
    plt.title('Strategy Comparison: LSTM vs Neural Network vs Buy & Hold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(plots_dir, f"Model_Comparison_{timestamp}.png"))
    plt.show()
    
    # Print comparison summary
    print("\nModel Comparison Summary:")
    print(f"Period: {lstm_backtest.index[0]} to {lstm_backtest.index[-1]}")
    
    # Calculate final returns
    buy_hold_return = lstm_backtest['Cumulative_Actual_Return'].iloc[-1] - 1
    lstm_return = lstm_backtest['Cumulative_Strategy_Return'].iloc[-1] - 1
    nn_return = nn_backtest['Cumulative_Strategy_Return'].iloc[-1] - 1
    
    # Calculate annualized returns
    days = len(lstm_backtest)
    buy_hold_annual = (1 + buy_hold_return) ** (252 / days) - 1
    lstm_annual = (1 + lstm_return) ** (252 / days) - 1
    nn_annual = (1 + nn_return) ** (252 / days) - 1
    
    # Calculate Sharpe ratios
    buy_hold_sharpe = np.sqrt(252) * lstm_backtest['Actual_Return'].mean() / lstm_backtest['Actual_Return'].std()
    lstm_sharpe = np.sqrt(252) * lstm_backtest['Strategy_Return'].mean() / lstm_backtest['Strategy_Return'].std()
    nn_sharpe = np.sqrt(252) * nn_backtest['Strategy_Return'].mean() / nn_backtest['Strategy_Return'].std()
    
    # Print metrics
    print(f"Buy & Hold Total Return: {buy_hold_return:.2%}")
    print(f"LSTM Strategy Total Return: {lstm_return:.2%}")
    print(f"Neural Network Strategy Total Return: {nn_return:.2%}")
    print(f"Buy & Hold Annualized Return: {buy_hold_annual:.2%}")
    print(f"LSTM Strategy Annualized Return: {lstm_annual:.2%}")
    print(f"Neural Network Strategy Annualized Return: {nn_annual:.2%}")
    print(f"Buy & Hold Sharpe Ratio: {buy_hold_sharpe:.4f}")
    print(f"LSTM Strategy Sharpe Ratio: {lstm_sharpe:.4f}")
    print(f"Neural Network Strategy Sharpe Ratio: {nn_sharpe:.4f}")

# Example usage
if __name__ == "__main__":
    # Print detailed GPU information if available
    if torch.cuda.is_available():
        print("\n=== GPU Information ===\n")
        print(f"GPU Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("\n=== No GPU Available - Using CPU ===\n")
    
    # Get ticker from user input
    import sys
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = input("Enter ticker symbol (e.g., AAPL, MSFT, QQQ): ").upper()
    
    # Calculate date 1 year ago for train/test split
    import datetime
    today = datetime.datetime.now()
    one_year_ago = (today - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"\n=== Running Prediction for {ticker} ===")
    print(f"Training on data before {one_year_ago}")
    print(f"Testing on data from {one_year_ago} to present")
    
    # Run LSTM prediction with ticker-based data fetching and date-based split
    print("\n=== Running LSTM Prediction ===\n")
    lstm_model, lstm_backtest = run_lstm_prediction(
        ticker=ticker,
        feature_cols=['Open', 'High', 'Low', 'Close', 'Volume'],
        sequence_length=60,
        date_split=one_year_ago,  # Use date-based split
        lstm_units=50,
        dropout_rate=0.2,
        epochs=50,
        batch_size=32
    )
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory after LSTM: {torch.cuda.memory_allocated() / 1e9:.2f} GB used")
    
    # Run Neural Network prediction with ticker-based data fetching and date-based split
    print("\n=== Running Neural Network Prediction ===\n")
    nn_model, nn_backtest = run_nn_prediction(
        ticker=ticker,
        feature_cols=['Open', 'High', 'Low', 'Close', 'Volume'],
        sequence_length=60,
        date_split=one_year_ago,  # Use date-based split
        layers=[64, 32],
        dropout_rate=0.2,
        epochs=50,
        batch_size=32
    )
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory after NN: {torch.cuda.memory_allocated() / 1e9:.2f} GB used")
    
    # Compare models
    print("\n=== Model Comparison ===\n")
    compare_models_with_buy_hold(lstm_backtest, nn_backtest)
    
    # Final GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nFinal GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB used")
        print("GPU resources released.")
    print(f"\nPrediction for {ticker} completed successfully!")