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

# Setup CUDA synchronization with proper error handling
if torch.cuda.is_available():
    try:
        # Try to synchronize CUDA operations
        torch.cuda.synchronize()
        print("CUDA synchronization successful")
    except RuntimeError as e:
        # Handle CUDA synchronization errors
        print(f"CUDA synchronization error: {e}")
        print("Falling back to CPU for this session")
        device = torch.device("cpu")
        print(f"Device switched to: {device}")
        
        # Provide troubleshooting advice
        print("\nTroubleshooting tips:")
        print("1. Close other GPU-intensive applications")
        print("2. Restart your Python environment")
        print("3. Try setting environment variable: CUDA_LAUNCH_BLOCKING=1")

def load_stock_data(file_path):
    """
    Load stock data from a CSV file
    
    Parameters:
    file_path (str): Path to the CSV file containing stock data
    
    Returns:
    pandas.DataFrame: Stock data
    """
    try:
        # First check the CSV file structure to determine the correct datetime column
        df_peek = pd.read_csv(file_path, nrows=1)
        
        # Check if 'Date' or 'Datetime' column exists
        datetime_col = None
        if 'Datetime' in df_peek.columns:
            datetime_col = 'Datetime'
        elif 'Date' in df_peek.columns:
            datetime_col = 'Date'
        
        # If datetime column found, set it as index
        if datetime_col:
            data = pd.read_csv(file_path, index_col=datetime_col, parse_dates=True)
            print(f"Loaded data with shape: {data.shape}")
            return data
        else:
            print(f"Error: No datetime column found in {file_path}")
            return None
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
            
            # Normalize dates for comparison
            normalized_dates = []
            for date in dates:
                # Convert timezone-aware timestamps to timezone-naive
                if hasattr(date, 'tzinfo') and date.tzinfo is not None:
                    normalized_dates.append(date.replace(tzinfo=None))
                else:
                    normalized_dates.append(date)
            
            # Ensure split_date is timezone-naive
            if hasattr(split_date, 'tzinfo') and split_date.tzinfo is not None:
                split_date = split_date.replace(tzinfo=None)
            
            # Create masks for train and test sets based on normalized dates
            train_mask = [date < split_date for date in normalized_dates]
            test_mask = [date >= split_date for date in normalized_dates]
            
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
    
    # Synchronize CUDA operations if using GPU with error handling
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f"CUDA synchronization error during training: {e}")
            print("Training completed, but CUDA synchronization failed")
    
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
        # Synchronize before prediction if using GPU with error handling
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError as e:
                print(f"CUDA synchronization error during evaluation: {e}")
                print("Continuing with evaluation without synchronization")
            
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

def backtest_strategy(data, predictions, initial_investment=10000, commission=0.001, ticker=None, 
                     use_adaptive_risk=True, max_position_size=1.0, min_position_size=0.1, 
                     base_stop_loss=0.02, base_take_profit=0.04, volatility_window=20, 
                     min_risk_reward_ratio=1.5, max_portfolio_heat=0.8):
    """
    Backtest a trading strategy based on model predictions with adaptive risk management
    
    Parameters:
    data (pandas.DataFrame): Original stock data
    predictions (numpy.ndarray): Model predictions
    initial_investment (float): Initial investment amount (default: 10000)
    commission (float): Commission rate per trade (default: 0.001)
    ticker (str): Stock ticker symbol for fetching dividend data (default: None)
    use_adaptive_risk (bool): Whether to use adaptive risk management (default: True)
    max_position_size (float): Maximum position size as a fraction of portfolio (default: 1.0)
    min_position_size (float): Minimum position size as a fraction of portfolio (default: 0.1)
    base_stop_loss (float): Base stop-loss percentage (default: 0.02 or 2%)
    base_take_profit (float): Base take-profit percentage (default: 0.04 or 4%)
    volatility_window (int): Window size for calculating volatility (default: 20)
    min_risk_reward_ratio (float): Minimum risk-reward ratio for taking a trade (default: 1.5)
    max_portfolio_heat (float): Maximum portfolio heat/exposure allowed (default: 0.8 or 80%)
    
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
    
    if use_adaptive_risk:
        # Calculate prediction confidence (magnitude of predicted return)
        backtest_data['Prediction_Confidence'] = backtest_data['Predicted_Return'].abs()
        
        # Calculate historical volatility using rolling standard deviation
        backtest_data['Volatility'] = backtest_data['Actual_Return'].rolling(window=volatility_window).std()
        
        # Fill NaN values with the mean to handle the initial window period
        backtest_data['Volatility'] = backtest_data['Volatility'].fillna(backtest_data['Volatility'].mean())
        
        # Normalize confidence to range [0, 1] using min-max scaling
        if len(backtest_data) > 1:
            min_conf = backtest_data['Prediction_Confidence'].min()
            max_conf = backtest_data['Prediction_Confidence'].max()
            if max_conf > min_conf:  # Avoid division by zero
                backtest_data['Normalized_Confidence'] = (backtest_data['Prediction_Confidence'] - min_conf) / (max_conf - min_conf)
            else:
                backtest_data['Normalized_Confidence'] = 0.5  # Default if all values are the same
        else:
            backtest_data['Normalized_Confidence'] = 0.5  # Default for single data point
        
        # Calculate adaptive position size based on prediction confidence
        backtest_data['Position_Size'] = min_position_size + (max_position_size - min_position_size) * backtest_data['Normalized_Confidence']
        
        # Calculate dynamic stop-loss and take-profit levels based on volatility
        backtest_data['Stop_Loss'] = base_stop_loss * (1 + backtest_data['Volatility'] / backtest_data['Volatility'].mean())
        backtest_data['Take_Profit'] = base_take_profit * (1 + backtest_data['Volatility'] / backtest_data['Volatility'].mean())
        
        # Calculate risk-reward ratio
        backtest_data['Risk_Reward_Ratio'] = backtest_data['Take_Profit'] / backtest_data['Stop_Loss']
        
        # Generate trading signals with risk management
        backtest_data['Signal'] = 0
        
        # Only take trades with favorable risk-reward ratio
        favorable_risk_reward = backtest_data['Risk_Reward_Ratio'] >= min_risk_reward_ratio
        
        # Buy signal when predicted return is positive and risk-reward is favorable
        backtest_data.loc[(backtest_data['Predicted_Return'] > 0) & favorable_risk_reward, 'Signal'] = 1
        
        # Sell signal when predicted return is negative and risk-reward is favorable
        backtest_data.loc[(backtest_data['Predicted_Return'] < 0) & favorable_risk_reward, 'Signal'] = -1
        
        # Calculate portfolio heat (exposure)
        backtest_data['Portfolio_Heat'] = backtest_data['Position_Size'] * backtest_data['Signal'].abs()
        
        # Apply portfolio heat limit
        backtest_data.loc[backtest_data['Portfolio_Heat'] > max_portfolio_heat, 'Position_Size'] = (
            max_portfolio_heat / backtest_data.loc[backtest_data['Portfolio_Heat'] > max_portfolio_heat, 'Signal'].abs()
        )
        
        # Recalculate portfolio heat after adjustment
        backtest_data['Portfolio_Heat'] = backtest_data['Position_Size'] * backtest_data['Signal'].abs()
        
        # Calculate strategy returns with position sizing
        backtest_data['Strategy_Return'] = backtest_data['Signal'].shift(1) * backtest_data['Position_Size'].shift(1) * backtest_data['Actual_Return']
    else:
        # Traditional approach without adaptive risk management
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
    # Check if adaptive risk management was used
    has_adaptive_risk = 'Position_Size' in backtest_data.columns
    
    if has_adaptive_risk:
        # Create a figure with 4 subplots for adaptive risk management
        plt.figure(figsize=(15, 15))
        
        # Plot 1: Portfolio Values
        plt.subplot(3, 1, 1)
        plt.plot(backtest_data['Buy_Hold_Value'], label='Buy & Hold')
        plt.plot(backtest_data['Strategy_Value'], label='Adaptive Strategy')
        plt.title('Portfolio Value Comparison')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Cumulative Returns
        plt.subplot(3, 1, 2)
        plt.plot(backtest_data['Cumulative_Actual_Return'], label='Buy & Hold')
        plt.plot(backtest_data['Cumulative_Strategy_Return'], label='Adaptive Strategy')
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Risk Management Metrics
        plt.subplot(3, 1, 3)
        
        # Create a twin axis for position size
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot volatility and position size
        ax1.plot(backtest_data['Volatility'], 'r-', label='Volatility', alpha=0.7)
        ax1.set_ylabel('Volatility', color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        
        ax2.plot(backtest_data['Position_Size'], 'b-', label='Position Size', alpha=0.7)
        ax2.set_ylabel('Position Size', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        # Add portfolio heat as a filled area
        ax2.fill_between(backtest_data.index, 0, backtest_data['Portfolio_Heat'], 
                         color='g', alpha=0.3, label='Portfolio Heat')
        
        plt.title('Adaptive Risk Management Metrics')
        plt.xlabel('Date')
        
        # Create a combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.grid(True)
    else:
        # Original 2-subplot figure for traditional strategy
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

def validate_ticker_symbol(ticker):
    """
    Validate a ticker symbol and provide suggestions for common mistakes
    
    Parameters:
    ticker (str): Stock ticker symbol to validate
    
    Returns:
    tuple: (is_valid, suggestion)
    """
    import yfinance as yf
    
    # Common ticker symbol mistakes and their corrections
    common_mistakes = {
        'TSMC': 'TSM',  # Taiwan Semiconductor Manufacturing Company
        'GOOGLE': 'GOOGL',  # Alphabet Inc. (Google)
        'ALPHABET': 'GOOGL',  # Alphabet Inc.
        'FACEBOOK': 'META',  # Meta Platforms (formerly Facebook)
        'FB': 'META',  # Meta Platforms (formerly Facebook)
        'TESLA': 'TSLA',  # Tesla Inc.
        'AMAZON': 'AMZN',  # Amazon.com Inc.
        'MICROSOFT': 'MSFT',  # Microsoft Corporation
        'APPLE': 'AAPL',  # Apple Inc.
        'NETFLIX': 'NFLX',  # Netflix Inc.
        'BERKSHIRE': 'BRK-B',  # Berkshire Hathaway Inc.
        'COCA COLA': 'KO',  # The Coca-Cola Company
        'COCA-COLA': 'KO',  # The Coca-Cola Company
        'MCDONALDS': 'MCD',  # McDonald's Corporation
        'MCDONALD': 'MCD',  # McDonald's Corporation
        'DISNEY': 'DIS',  # The Walt Disney Company
        'NVIDIA': 'NVDA',  # NVIDIA Corporation
        'INTEL': 'INTC',  # Intel Corporation
        'AMD': 'AMD',  # Advanced Micro Devices, Inc.
        'IBM': 'IBM',  # International Business Machines Corporation
        'ORACLE': 'ORCL',  # Oracle Corporation
        'CISCO': 'CSCO',  # Cisco Systems, Inc.
        'SALESFORCE': 'CRM',  # Salesforce, Inc.
        'NIKE': 'NKE',  # NIKE, Inc.
        'ADIDAS': 'ADDYY',  # Adidas AG
        'VOLKSWAGEN': 'VWAGY',  # Volkswagen AG
        'TOYOTA': 'TM',  # Toyota Motor Corporation
        'HONDA': 'HMC',  # Honda Motor Co., Ltd.
        'FORD': 'F',  # Ford Motor Company
        'GM': 'GM',  # General Motors Company
        'GENERAL MOTORS': 'GM',  # General Motors Company
        'EXXON': 'XOM',  # Exxon Mobil Corporation
        'EXXONMOBIL': 'XOM',  # Exxon Mobil Corporation
        'CHEVRON': 'CVX',  # Chevron Corporation
        'BP': 'BP',  # BP p.l.c.
        'SHELL': 'SHEL',  # Shell plc
        'ROYAL DUTCH SHELL': 'SHEL',  # Shell plc
        'PFIZER': 'PFE',  # Pfizer Inc.
        'JOHNSON': 'JNJ',  # Johnson & Johnson
        'JOHNSON & JOHNSON': 'JNJ',  # Johnson & Johnson
        'MERCK': 'MRK',  # Merck & Co., Inc.
        'NOVARTIS': 'NVS',  # Novartis AG
        'ROCHE': 'RHHBY',  # Roche Holding AG
        'ASTRAZENECA': 'AZN',  # AstraZeneca PLC
        'MODERNA': 'MRNA',  # Moderna, Inc.
        'BIONTECH': 'BNTX',  # BioNTech SE
        'BANK OF AMERICA': 'BAC',  # Bank of America Corporation
        'JPMORGAN': 'JPM',  # JPMorgan Chase & Co.
        'JP MORGAN': 'JPM',  # JPMorgan Chase & Co.
        'GOLDMAN SACHS': 'GS',  # The Goldman Sachs Group, Inc.
        'MORGAN STANLEY': 'MS',  # Morgan Stanley
        'WELLS FARGO': 'WFC',  # Wells Fargo & Company
        'CITIGROUP': 'C',  # Citigroup Inc.
        'CITI': 'C',  # Citigroup Inc.
        'HSBC': 'HSBC',  # HSBC Holdings plc
        'BARCLAYS': 'BCS',  # Barclays PLC
        'UBS': 'UBS',  # UBS Group AG
        'CREDIT SUISSE': 'CS',  # Credit Suisse Group AG
        'DEUTSCHE BANK': 'DB',  # Deutsche Bank Aktiengesellschaft
        'PAYPAL': 'PYPL',  # PayPal Holdings, Inc.
        'VISA': 'V',  # Visa Inc.
        'MASTERCARD': 'MA',  # Mastercard Incorporated
        'AMERICAN EXPRESS': 'AXP',  # American Express Company
        'AMEX': 'AXP',  # American Express Company
        'SQUARE': 'SQ',  # Block, Inc. (formerly Square)
        'BLOCK': 'SQ',  # Block, Inc.
        'ROBINHOOD': 'HOOD',  # Robinhood Markets, Inc.
        'COINBASE': 'COIN',  # Coinbase Global, Inc.
        'BITCOIN': 'BTC-USD',  # Bitcoin USD
        'ETHEREUM': 'ETH-USD',  # Ethereum USD
        'DOW JONES': '^DJI',  # Dow Jones Industrial Average
        'DOW': '^DJI',  # Dow Jones Industrial Average
        'S&P 500': '^GSPC',  # S&P 500
        'S&P': '^GSPC',  # S&P 500
        'NASDAQ': '^IXIC',  # NASDAQ Composite
        'RUSSELL 2000': '^RUT',  # Russell 2000
        'VIX': '^VIX',  # CBOE Volatility Index
        'NIKKEI': '^N225',  # Nikkei 225
        'FTSE': '^FTSE',  # FTSE 100
        'DAX': '^GDAXI',  # DAX PERFORMANCE-INDEX
        'CAC 40': '^FCHI',  # CAC 40
        'HANG SENG': '^HSI',  # HANG SENG INDEX
        'SSE': '^SSEC',  # SSE Composite Index
        'SENSEX': '^BSESN',  # S&P BSE SENSEX
        'NIFTY 50': '^NSEI',  # NIFTY 50
        'ASX 200': '^AXJO',  # S&P/ASX 200
    }
    
    # Check if the ticker is in the common mistakes dictionary
    if ticker.upper() in common_mistakes:
        suggestion = common_mistakes[ticker.upper()]
        return False, suggestion
    
    # Try to get info for the ticker to validate it
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Check if we got valid info back
        if 'symbol' in info and info['symbol'] == ticker.upper():
            return True, None
        else:
            # Try to get a suggestion
            for correct_ticker, info in common_mistakes.items():
                if ticker.upper() in correct_ticker or correct_ticker in ticker.upper():
                    return False, info
            return False, None
    except Exception as e:
        # If there was an error, the ticker might not be valid
        # Try to get a suggestion
        for correct_ticker, info in common_mistakes.items():
            if ticker.upper() in correct_ticker or correct_ticker in ticker.upper():
                return False, info
        return False, None

def get_backtest_data(ticker, use_backtest=True, date_split=None, force_refresh=True):
    """
    Get backtest data for a ticker, checking for existing backtest data first.
    If backtest data is missing or force_refresh is True, it will use collect_and_save_stock_data.
    
    Parameters:
    ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    use_backtest (bool): Whether to try using backtest data first (default: True)
    date_split (str): Date to split training and testing data (format: 'YYYY-MM-DD')
                      If None, will use 1 year ago as the split date
    force_refresh (bool): Whether to force refresh data even if it exists (default: True)
    
    Returns:
    tuple: (data_file, date_split)
    """
    import os
    import datetime
    from stock_data import collect_and_save_backtest_data, collect_and_save_stock_data
    
    # Validate the ticker symbol
    is_valid, suggestion = validate_ticker_symbol(ticker)
    if not is_valid and suggestion is not None:
        print(f"Warning: '{ticker}' may not be a valid ticker symbol.")
        print(f"Did you mean '{suggestion}'? Using '{suggestion}' instead.")
        ticker = suggestion
    elif not is_valid:
        print(f"Warning: '{ticker}' may not be a valid ticker symbol. Proceeding anyway, but data retrieval may fail.")
    
    # If date_split is not provided, use 1 year ago as the default split date
    if date_split is None:
        # Calculate date 1 year ago from today
        today = datetime.datetime.now()
        one_year_ago = (today - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
        date_split = one_year_ago
        print(f"Using default date split: {date_split} (1 year ago)")
    
    # Check if backtest data exists
    stock_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stockData')
    backtest_file = os.path.join(stock_data_dir, f"backtest{ticker.upper()}_data.csv")
    
    # Use existing data only if force_refresh is False and the file exists
    if use_backtest and os.path.exists(backtest_file) and not force_refresh:
        print(f"Using existing backtest data for {ticker} from {backtest_file}")
        return backtest_file, date_split
        
    # If force_refresh is True or file doesn't exist, collect fresh data
    if force_refresh and os.path.exists(backtest_file):
        print(f"Force refresh enabled: Collecting fresh data for {ticker}")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Data refresh timestamp: {timestamp}")
    
    # If backtest data doesn't exist or we're not using it, try to get it
    if use_backtest:
        print(f"Backtest data not found for {ticker}, collecting it now...")
        # Get training data from backtest function (historical data)
        backtest_file = collect_and_save_backtest_data(
            ticker,
            interval="1d",
            output_format="csv",
            end_date=date_split  # Use date_split as the end date for training data
        )
        
        if backtest_file is not None:
            print(f"Successfully collected backtest data for {ticker}")
            return backtest_file, date_split
    
    # If we couldn't get backtest data or we're not using it, use regular stock data
    print(f"Using regular stock data collection for {ticker}")
    data_file = collect_and_save_stock_data(
        ticker,
        period="max",  # Get maximum available history
        interval="1d",  # Daily data
        output_format="csv"
    )
    
    if data_file is None:
        print(f"Error: Failed to fetch data for {ticker}")
        return None, date_split
    
    return data_file, date_split

def run_lstm_prediction(ticker=None, data_file=None, feature_cols=None, sequence_length=60, train_split=0.8, 
                       lstm_units=50, dropout_rate=0.2, epochs=50, batch_size=32, date_split=None, force_refresh=True,
                       use_adaptive_risk=True, max_position_size=1.0, min_position_size=0.1, 
                       base_stop_loss=0.02, base_take_profit=0.04, volatility_window=20, 
                       min_risk_reward_ratio=1.5, max_portfolio_heat=0.8):
    """
    Run the complete LSTM prediction workflow with adaptive risk management
    
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
    force_refresh (bool): Whether to force refresh data even if it exists (default: True)
    use_adaptive_risk (bool): Whether to use adaptive risk management (default: True)
    max_position_size (float): Maximum position size as a fraction of portfolio (default: 1.0)
    min_position_size (float): Minimum position size as a fraction of portfolio (default: 0.1)
    base_stop_loss (float): Base stop-loss percentage (default: 0.02 or 2%)
    base_take_profit (float): Base take-profit percentage (default: 0.04 or 4%)
    volatility_window (int): Window size for calculating volatility (default: 20)
    min_risk_reward_ratio (float): Minimum risk-reward ratio for taking a trade (default: 1.5)
    max_portfolio_heat (float): Maximum portfolio heat/exposure allowed (default: 0.8 or 80%)
    
    Returns:
    tuple: (model, backtest_data)
    """
    # Load or fetch data
    if ticker is not None:
        # Get backtest data or regular stock data if backtest is missing
        data_file, date_split = get_backtest_data(ticker, use_backtest=True, date_split=date_split, force_refresh=force_refresh)
        
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
    
    # Backtest strategy with adaptive risk management
    backtest_data = backtest_strategy(
        data, y_pred, 
        use_adaptive_risk=use_adaptive_risk,
        max_position_size=max_position_size,
        min_position_size=min_position_size,
        base_stop_loss=base_stop_loss,
        base_take_profit=base_take_profit,
        volatility_window=volatility_window,
        min_risk_reward_ratio=min_risk_reward_ratio,
        max_portfolio_heat=max_portfolio_heat
    )
    
    # Plot backtest results
    title = f"{ticker} LSTM Trading Strategy Backtest Results" if ticker else "LSTM Trading Strategy Backtest Results"
    plot_backtest_results(backtest_data, title=title)
    
    return model, backtest_data

def run_nn_prediction(ticker=None, data_file=None, feature_cols=None, sequence_length=60, train_split=0.8, 
                     layers=[64, 32], dropout_rate=0.2, epochs=50, batch_size=32, date_split=None, force_refresh=True,
                     use_adaptive_risk=True, max_position_size=1.0, min_position_size=0.1, 
                     base_stop_loss=0.02, base_take_profit=0.04, volatility_window=20, 
                     min_risk_reward_ratio=1.5, max_portfolio_heat=0.8):
    """
    Run the complete Neural Network prediction workflow with adaptive risk management
    
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
    force_refresh (bool): Whether to force refresh data even if it exists (default: True)
    use_adaptive_risk (bool): Whether to use adaptive risk management (default: True)
    max_position_size (float): Maximum position size as a fraction of portfolio (default: 1.0)
    min_position_size (float): Minimum position size as a fraction of portfolio (default: 0.1)
    base_stop_loss (float): Base stop-loss percentage (default: 0.02 or 2%)
    base_take_profit (float): Base take-profit percentage (default: 0.04 or 4%)
    volatility_window (int): Window size for calculating volatility (default: 20)
    min_risk_reward_ratio (float): Minimum risk-reward ratio for taking a trade (default: 1.5)
    max_portfolio_heat (float): Maximum portfolio heat/exposure allowed (default: 0.8 or 80%)
    
    Returns:
    tuple: (model, backtest_data)
    """
    # Load or fetch data
    if ticker is not None:
        # Get backtest data or regular stock data if backtest is missing
        data_file, date_split = get_backtest_data(ticker, use_backtest=True, date_split=date_split, force_refresh=force_refresh)
        
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
    
    # Backtest strategy with adaptive risk management
    backtest_data = backtest_strategy(
        data, y_pred, 
        use_adaptive_risk=use_adaptive_risk,
        max_position_size=max_position_size,
        min_position_size=min_position_size,
        base_stop_loss=base_stop_loss,
        base_take_profit=base_take_profit,
        volatility_window=volatility_window,
        min_risk_reward_ratio=min_risk_reward_ratio,
        max_portfolio_heat=max_portfolio_heat
    )
    
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
    print("\n=== Running LSTM Prediction with Adaptive Risk Management ===\n")
    lstm_model, lstm_backtest = run_lstm_prediction(
        ticker=ticker,
        feature_cols=['Open', 'High', 'Low', 'Close', 'Volume'],
        sequence_length=60,
        date_split=one_year_ago,  # Use date-based split
        lstm_units=50,
        dropout_rate=0.2,
        epochs=50,
        batch_size=32,
        force_refresh=True,  # Always get fresh data
        use_adaptive_risk=True,  # Enable adaptive risk management
        max_position_size=1.0,
        min_position_size=0.1,
        base_stop_loss=0.02,
        base_take_profit=0.04,
        volatility_window=20,
        min_risk_reward_ratio=1.5,
        max_portfolio_heat=0.8
    )
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory after LSTM: {torch.cuda.memory_allocated() / 1e9:.2f} GB used")
    
    # Run Neural Network prediction with ticker-based data fetching and date-based split
    print("\n=== Running Neural Network Prediction with Adaptive Risk Management ===\n")
    nn_model, nn_backtest = run_nn_prediction(
        ticker=ticker,
        feature_cols=['Open', 'High', 'Low', 'Close', 'Volume'],
        sequence_length=60,
        date_split=one_year_ago,  # Use date-based split
        layers=[64, 32],
        dropout_rate=0.2,
        epochs=50,
        batch_size=32,
        force_refresh=True,  # Always get fresh data
        use_adaptive_risk=True,  # Enable adaptive risk management
        max_position_size=1.0,
        min_position_size=0.1,
        base_stop_loss=0.02,
        base_take_profit=0.04,
        volatility_window=20,
        min_risk_reward_ratio=1.5,
        max_portfolio_heat=0.8
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