#!/usr/bin/env python
"""
ML Backtest and Paper Trading Tool for Thor Trading System

This script trains multiple ML models (LSTM, GARCH, Markov) on historical
energy futures data (CL and HO), runs a backtest to evaluate performance,
and can also run in paper trading mode to simulate live trading.

Usage:
    ./ml_backtest.py [options]

Options:
    --mode=MODE         Operation mode: backtest or paper_trade (default: backtest)
    --start=DATE        Start date for backtest period (YYYY-MM-DD)
    --end=DATE          End date for backtest period (YYYY-MM-DD)
    --symbols=SYMBOLS   Comma-separated list of symbols to backtest (default: CL,HO)
    --capital=AMOUNT    Initial capital for backtest (default: 100000)
    --models=MODELS     Comma-separated list of models to use (default: lstm,garch,markov)
    --paper-days=DAYS   Number of days to run paper trading simulation (default: 30)
    --output-dir=DIR    Directory for output files (default: current directory)
    --verbose           Print detailed progress information
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add thor_trading to the path
thor_trading_path = '/mnt/e/Projects/thor_trading'
sys.path.append(thor_trading_path)

# ML libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from arch import arch_model
    from hmmlearn import hmm
except ImportError as e:
    print(f"Error importing ML libraries: {str(e)}")
    print("Installing required packages...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow", "scikit-learn", "arch", "hmmlearn"])
    # Try importing again
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from arch import arch_model
    from hmmlearn import hmm

try:
    from connectors.postgres_connector import PostgresConnector
except ImportError as e:
    print(f"Error importing database connector: {str(e)}")
    print("Creating mock database connector...")
    
    class PostgresConnector:
        """Mock database connector for testing."""
        
        def __init__(self):
            pass
            
        def test_connection(self):
            return True
            
        def query(self, query, params=None):
            """Mock query that returns sample data."""
            # Generate synthetic data for testing
            start_date = datetime.now() - timedelta(days=1000)
            dates = [start_date + timedelta(days=i) for i in range(1000)]
            
            data = []
            prev_price = 60.0  # Starting price for CL
            
            for date in dates:
                # Create a random walk with mean reversion
                change = np.random.normal(0, 1) * 0.5
                # Add mean reversion
                if prev_price > 80:
                    change -= 0.2
                elif prev_price < 40:
                    change += 0.2
                
                # Calculate new price
                price = prev_price + change
                prev_price = price
                
                # Create OHLC data with some random spread
                high = price + abs(np.random.normal(0, 1)) * 0.3
                low = price - abs(np.random.normal(0, 1)) * 0.3
                open_price = price - change * np.random.random()
                
                # Add volume with some randomness
                volume = int(np.random.normal(1000000, 200000))
                if volume < 100000:
                    volume = 100000
                
                data.append({
                    'timestamp': date,
                    'symbol': 'CL',
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume
                })
            
            return data

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Thor Trading System ML Backtest and Paper Trading Tool')
    parser.add_argument('--mode', type=str, default='backtest', choices=['backtest', 'paper_trade'],
                       help='Operation mode: backtest or paper_trade')
    parser.add_argument('--start', type=str, default=None,
                       help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                       help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str, default='CL,HO',
                       help='Comma-separated list of symbols to backtest')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital for backtest')
    parser.add_argument('--models', type=str, default='lstm,garch,markov',
                       help='Comma-separated list of models to use')
    parser.add_argument('--paper-days', type=int, default=30,
                       help='Number of days to run paper trading simulation')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Directory for output files')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress information')
    
    return parser.parse_args()

def load_data(db, symbol, start_date=None, end_date=None):
    """Load market data from database."""
    query = f"""
    SELECT timestamp, symbol, open, high, low, close, volume
    FROM market_data
    WHERE symbol = '{symbol}'
    """
    
    if start_date:
        query += f" AND timestamp >= '{start_date}'"
    if end_date:
        query += f" AND timestamp <= '{end_date}'"
    
    query += " ORDER BY timestamp"
    
    try:
        data = db.query(query)
        df = pd.DataFrame(data)
        if df.empty:
            print(f"No data found for {symbol}")
            return None
            
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Create returns column
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Drop first row with NaN returns
        df = df.dropna()
        
        return df
    except Exception as e:
        print(f"Error loading data for {symbol}: {str(e)}")
        return None

def create_advanced_features(df):
    """Create advanced financial indicators and features."""
    # Copy original dataframe to avoid modifying it
    data = df.copy()
    
    # Technical indicators
    # Moving averages
    data['sma5'] = data['close'].rolling(window=5).mean()
    data['sma10'] = data['close'].rolling(window=10).mean()
    data['sma20'] = data['close'].rolling(window=20).mean()
    data['sma50'] = data['close'].rolling(window=50).mean()
    data['sma200'] = data['close'].rolling(window=200).mean()
    
    # Volatility indicators
    data['std20'] = data['returns'].rolling(window=20).std()
    data['std50'] = data['returns'].rolling(window=50).std()
    
    # Relative moving averages
    data['close_sma10_ratio'] = data['close'] / data['sma10']
    data['sma10_sma50_ratio'] = data['sma10'] / data['sma50']
    
    # Momentum indicators
    data['roc5'] = data['close'].pct_change(periods=5)
    data['roc10'] = data['close'].pct_change(periods=10)
    data['roc20'] = data['close'].pct_change(periods=20)
    
    # Exponential moving averages
    data['ema12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['ema26'] = data['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    data['macd'] = data['ema12'] - data['ema26']
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    
    # RSI (14-period)
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['bb_middle'] = data['close'].rolling(window=20).mean()
    data['bb_std'] = data['close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * 2)
    data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * 2)
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    data['bb_pct_b'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    
    # Volume indicators
    if 'volume' in df.columns:
        data['volume_sma20'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma20']
        
    # Seasonality features (day of week, month)
    if isinstance(data.index, pd.DatetimeIndex):
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
    
    # Drop NaN values from feature calculations
    data = data.dropna()
    
    return data

def preprocess_data(df, sequence_length=20):
    """Preprocess data with advanced features for LSTM model."""
    # Create advanced features
    print("Creating advanced technical indicators...")
    data = create_advanced_features(df)
    
    # Select features for the model
    feature_columns = [
        'close', 'returns', 'log_returns', 
        'sma5', 'sma10', 'sma20', 'sma50',
        'close_sma10_ratio', 'sma10_sma50_ratio',
        'std20', 'roc5', 'roc10', 'roc20',
        'macd', 'macd_signal', 'macd_hist',
        'rsi', 'bb_width', 'bb_pct_b'
    ]
    
    # Add volume features if available
    if 'volume' in data.columns and 'volume_ratio' in data.columns:
        feature_columns.extend(['volume', 'volume_ratio'])
    
    # Filter to selected features
    features = data[feature_columns].copy()
    
    print(f"Using {len(feature_columns)} features for LSTM model")
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        # Predict next day's close price (first column)
        y.append(scaled_data[i+sequence_length, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Split into training and testing sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'data': features,
        'feature_columns': feature_columns
    }

def build_lstm_model(input_shape):
    """Build and compile an enhanced LSTM model."""
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape, 
             recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.3),
        LSTM(80, return_sequences=True),
        Dropout(0.3),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(30, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)
    ])
    
    # Use a more sophisticated optimizer with learning rate scheduling
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(optimizer=optimizer, loss='huber')  # Huber loss is more robust to outliers
    return model

def train_lstm_model(data, verbose=False):
    """Train LSTM model and make predictions."""
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    # Build model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1 if verbose else 0
    )
    
    # Make predictions
    predicted = model.predict(X_test)
    
    # For inverse transform, create an array the same shape as our training data
    num_features = data['X_train'].shape[2]
    
    # Create placeholder arrays with zeros in all columns except the first (close price)
    pred_copies = np.zeros((len(predicted), num_features))
    pred_copies[:, 0] = predicted.flatten()
    
    orig_copies = np.zeros((len(y_test), num_features))
    orig_copies[:, 0] = y_test
    
    # Inverse transform
    predicted_price = data['scaler'].inverse_transform(pred_copies)[:, 0]
    original_price = data['scaler'].inverse_transform(orig_copies)[:, 0]
    
    # Calculate metrics
    mse = mean_squared_error(original_price, predicted_price)
    mae = mean_absolute_error(original_price, predicted_price)
    rmse = np.sqrt(mse)
    
    if verbose:
        print(f"LSTM Model Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
    
    return {
        'model': model,
        'predicted': predicted_price,
        'original': original_price,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'history': history
    }

def train_garch_model(df, verbose=False):
    """Train GARCH model and make predictions."""
    # Use log returns for GARCH
    returns = df['log_returns'].values
    
    # Set up GARCH model - GARCH(1,1)
    model = arch_model(returns, vol='GARCH', p=1, q=1, mean='AR', lags=1)
    
    # Fit model
    if verbose:
        model_fit = model.fit(disp='final')
    else:
        model_fit = model.fit(disp='off')
    
    # Make predictions - forecast volatility for test period
    forecasts = model_fit.forecast(horizon=20)
    
    # Extract conditional volatility
    vol = model_fit.conditional_volatility
    
    # Predict direction
    predicted_direction = np.sign(model_fit.resid[1:])
    actual_direction = np.sign(returns[1:])
    direction_accuracy = np.mean(predicted_direction == actual_direction)
    
    if verbose:
        print(f"GARCH Model Summary:")
        print(model_fit.summary())
        print(f"Direction Accuracy: {direction_accuracy:.4f}")
    
    return {
        'model': model_fit,
        'volatility': vol,
        'forecasts': forecasts,
        'direction_accuracy': direction_accuracy
    }

def train_markov_model(df, n_states=5, verbose=False):
    """Train advanced Markov model to predict market regimes using more features."""
    print("Training enhanced Markov model for regime detection...")
    
    # Create advanced features
    data = create_advanced_features(df)
    
    # Select features for regime detection
    features = [
        'returns',                # Daily returns
        'log_returns',            # Log returns
        'roc5',                   # 5-day rate of change
        'roc20',                  # 20-day rate of change
        'std20',                  # 20-day volatility
        'close_sma10_ratio',      # Price relative to 10-day MA
        'sma10_sma50_ratio',      # Short-term trend vs medium-term trend
        'macd_hist',              # MACD histogram
        'rsi'                     # RSI
    ]
    
    # Add volume feature if available
    if 'volume_ratio' in data.columns:
        features.append('volume_ratio')
    
    # Prepare feature matrix
    X = data[features].values
    
    # Remove NaN values
    X = X[~np.isnan(X).any(axis=1)]
    
    print(f"Using {len(features)} features for market regime detection")
    
    # Scale the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try multiple Markov models with different parameters and select the best
    best_model = None
    best_bic = np.inf
    best_n_states = n_states
    
    # Try different covariance types and state counts if verbose mode
    cov_types = ["full", "diag", "tied"] if verbose else ["full"]
    state_counts = [3, 4, 5, 6] if verbose else [n_states]
    
    for cov_type in cov_types:
        for n_comp in state_counts:
            try:
                print(f"  Testing HMM with {n_comp} states, {cov_type} covariance...")
                # Train HMM
                hmm_model = hmm.GaussianHMM(
                    n_components=n_comp,
                    covariance_type=cov_type,
                    n_iter=2000,  # More iterations for convergence
                    tol=1e-5,     # Tighter convergence tolerance
                    random_state=42
                )
                
                hmm_model.fit(X_scaled)
                
                # Calculate BIC
                # BIC = -2 * log-likelihood + k * log(n)
                # where k = number of parameters, n = number of observations
                n_features = X_scaled.shape[1]
                n_samples = X_scaled.shape[0]
                
                # Number of parameters depends on covariance type
                if cov_type == 'full':
                    # For each state: mean vector (n_features) + covariance matrix (n_features^2)
                    # + transition probabilities (n_states)
                    n_params = n_comp * (n_features + n_features**2) + n_comp**2
                elif cov_type == 'diag':
                    # For each state: mean vector (n_features) + diagonal covariance (n_features)
                    # + transition probabilities (n_states)
                    n_params = n_comp * (2 * n_features) + n_comp**2
                elif cov_type == 'tied':
                    # mean vectors for each state (n_states * n_features) + one shared
                    # covariance matrix (n_features^2) + transition probabilities (n_states^2)
                    n_params = n_comp * n_features + n_features**2 + n_comp**2
                
                # Calculate BIC
                bic = -2 * hmm_model.score(X_scaled) * n_samples + n_params * np.log(n_samples)
                
                print(f"    BIC: {bic:.2f}")
                
                # Update best model if this one has better BIC
                if bic < best_bic:
                    best_bic = bic
                    best_model = hmm_model
                    best_n_states = n_comp
                    
            except Exception as e:
                print(f"    Error training HMM with {n_comp} states: {str(e)}")
    
    # Use the best model
    model = best_model
    n_states = best_n_states
    print(f"Selected HMM with {n_states} states")
    
    # Predict hidden states
    hidden_states = model.predict(X_scaled)
    
    # Calculate comprehensive state statistics
    state_stats = {}
    for state in range(n_states):
        mask = hidden_states == state
        
        # Get sequences of days in this state
        state_sequences = []
        current_seq = []
        
        for i in range(len(mask)):
            if mask[i]:
                current_seq.append(i)
            elif current_seq:
                state_sequences.append(current_seq)
                current_seq = []
                
        if current_seq:  # Add the last sequence if it exists
            state_sequences.append(current_seq)
        
        # Calculate statistics for this state
        returns_in_state = X[mask, 0]  # Returns for this state
        
        # Calculate mean return for each feature
        feature_means = {}
        for i, feature in enumerate(features):
            feature_means[feature] = X[mask, i].mean()
        
        # Calculate how long this state typically lasts
        sequence_lengths = [len(seq) for seq in state_sequences]
        avg_duration = np.mean(sequence_lengths) if sequence_lengths else 0
        
        # Calculate transition frequency (how often we enter this state)
        transitions_to_state = 0
        for i in range(1, len(hidden_states)):
            if hidden_states[i] == state and hidden_states[i-1] != state:
                transitions_to_state += 1
        
        # Store statistics
        state_stats[state] = {
            'mean': returns_in_state.mean(),
            'std': returns_in_state.std(),
            'count': len(returns_in_state),
            'pct': len(returns_in_state) / len(hidden_states) * 100,
            'feature_means': feature_means,
            'avg_duration': avg_duration,
            'transitions_to_state': transitions_to_state,
            'sequences': state_sequences
        }
    
    # Sort states by mean return
    sorted_states = sorted(state_stats.items(), key=lambda x: x[1]['mean'])
    
    # For a 5-state model, label as strong bear, weak bear, neutral, weak bull, strong bull
    if n_states == 5:
        labels = {
            sorted_states[0][0]: 'Strong Bear',
            sorted_states[1][0]: 'Weak Bear',
            sorted_states[2][0]: 'Neutral',
            sorted_states[3][0]: 'Weak Bull',
            sorted_states[4][0]: 'Strong Bull'
        }
    # For a 3-state model, label as bear, neutral, bull
    elif n_states == 3:
        labels = {
            sorted_states[0][0]: 'Bear',
            sorted_states[1][0]: 'Neutral',
            sorted_states[2][0]: 'Bull'
        }
    # For 4 states
    elif n_states == 4:
        labels = {
            sorted_states[0][0]: 'Strong Bear',
            sorted_states[1][0]: 'Weak Bear',
            sorted_states[2][0]: 'Weak Bull',
            sorted_states[3][0]: 'Strong Bull'
        }
    # For other numbers of states
    else:
        labels = {state: f"State_{state}" for state in range(n_states)}
    
    if verbose:
        print("\nMarkov Model State Statistics:")
        for state, stats in sorted(state_stats.items()):
            state_label = labels.get(state, f"State_{state}")
            print(f"{state_label} (State {state}):")
            print(f"  Mean Return: {stats['mean']:.6f}")
            print(f"  Std Dev: {stats['std']:.6f}")
            print(f"  Count: {stats['count']} ({stats['pct']:.2f}%)")
            print(f"  Avg Duration: {stats['avg_duration']:.2f} days")
            print(f"  Feature Characteristics:")
            for feature, value in stats['feature_means'].items():
                print(f"    {feature}: {value:.6f}")
            print()
    
    return {
        'model': model,
        'hidden_states': hidden_states,
        'state_stats': state_stats,
        'labels': labels,
        'scaler': scaler,
        'features': features
    }

def backtest_lstm_strategy(symbol, lstm_results, original_data, initial_capital=100000, verbose=False):
    """Backtest a trading strategy based on LSTM predictions with improved risk management."""
    # Create a copy of the original data for the test period
    test_size = len(lstm_results['original'])
    backtest_data = original_data.iloc[-test_size:].copy()
    
    # Add predictions to the data
    backtest_data['predicted_price'] = lstm_results['predicted']
    backtest_data['predicted_return'] = backtest_data['predicted_price'].pct_change()
    
    # Calculate prediction accuracy metrics (not used for trading yet)
    backtest_data['actual_direction'] = np.sign(backtest_data['returns'])
    backtest_data['predicted_direction'] = np.sign(backtest_data['predicted_return'])
    backtest_data['prediction_correct'] = (backtest_data['actual_direction'] == backtest_data['predicted_direction']).astype(int)
    
    # Calculate 20-day rolling accuracy to determine confidence
    backtest_data['rolling_accuracy'] = backtest_data['prediction_correct'].rolling(20).mean().fillna(0.5)
    
    # Only take trades when:
    # 1. Predicted return exceeds a minimum threshold (stronger signal)
    # 2. Rolling accuracy is good enough (model has been accurate lately)
    # 3. Volatility is not too high (avoid high-risk periods)
    
    # Calculate volatility
    backtest_data['volatility'] = backtest_data['returns'].rolling(20).std().fillna(0)
    backtest_data['vol_percentile'] = backtest_data['volatility'].rolling(60).rank(pct=True).fillna(0.5)
    
    # Create signals with stricter conditions
    min_return_threshold = 0.002  # 0.2% minimum predicted return
    min_accuracy = 0.55  # Require at least 55% recent accuracy
    max_volatility = 0.7  # Avoid highest 30% volatility periods
    
    # Long signal when:
    # - Predicted return is positive and exceeds threshold
    # - Recent prediction accuracy is good
    # - Volatility is not too high
    long_condition = (
        (backtest_data['predicted_return'] > min_return_threshold) & 
        (backtest_data['rolling_accuracy'] > min_accuracy) &
        (backtest_data['vol_percentile'] < max_volatility)
    )
    
    # Short signal when:
    # - Predicted return is negative and exceeds threshold (in abs value)
    # - Recent prediction accuracy is good
    # - Volatility is not too high
    short_condition = (
        (backtest_data['predicted_return'] < -min_return_threshold) & 
        (backtest_data['rolling_accuracy'] > min_accuracy) &
        (backtest_data['vol_percentile'] < max_volatility)
    )
    
    # Generate signals: 1 for long, -1 for short, 0 for no position
    backtest_data['signal'] = 0
    backtest_data.loc[long_condition, 'signal'] = 1
    backtest_data.loc[short_condition, 'signal'] = -1
    
    # Add position sizing based on confidence and volatility
    backtest_data['position_size'] = 1.0
    # Reduce position size during higher volatility
    backtest_data['position_size'] = backtest_data['position_size'] * (1 - backtest_data['vol_percentile'])
    # Scale position size by prediction confidence
    backtest_data['position_size'] = backtest_data['position_size'] * backtest_data['rolling_accuracy']
    
    # Implement trailing stop loss (exit if drawdown exceeds threshold)
    max_drawdown_allowed = 0.05  # 5% maximum drawdown per trade
    
    # Initialize portfolio tracking
    backtest_data['position'] = 0
    backtest_data['equity'] = initial_capital
    backtest_data['cash'] = initial_capital
    backtest_data['invested'] = 0
    backtest_data['trade_active'] = False
    backtest_data['entry_price'] = 0
    backtest_data['trade_return'] = 0
    
    # Track drawdown per trade
    equity_peak = initial_capital
    trade_peak = 0
    current_position = 0
    entry_price = 0
    
    # Implement the strategy with risk management
    for i in range(1, len(backtest_data)):
        prev_equity = backtest_data.iloc[i-1]['equity']
        signal = backtest_data.iloc[i]['signal']
        position_size = backtest_data.iloc[i]['position_size']
        price = backtest_data.iloc[i]['close']
        
        # Update equity peak
        if prev_equity > equity_peak:
            equity_peak = prev_equity
        
        # Calculate current drawdown
        current_drawdown = 1 - (prev_equity / equity_peak) if equity_peak > 0 else 0
                
        # If we have an active trade, check for stop loss
        if current_position != 0:
            # Calculate trade return
            if entry_price > 0:
                trade_return = (price / entry_price - 1) * current_position
                
                # Update trade peak if this is the best we've done
                if trade_return > trade_peak:
                    trade_peak = trade_return
                
                # Calculate drawdown from trade peak
                trade_drawdown = trade_peak - trade_return
                
                # If drawdown exceeds threshold, exit the trade
                if trade_drawdown >= max_drawdown_allowed:
                    # Close position
                    current_position = 0
                    entry_price = 0
                    trade_peak = 0
                    # Record that we're applying a stop loss
                    if verbose and i % 100 == 0:
                        print(f"Stop loss triggered at {backtest_data.index[i]} with {trade_drawdown:.2%} drawdown")
        
        # If there's a new signal and no current position, enter a new trade
        if current_position == 0 and signal != 0:
            # Size position based on confidence and volatility
            current_position = signal * position_size
            entry_price = price
            trade_peak = 0
        
        # Otherwise, if signal changes direction, exit and reverse
        elif current_position * signal < 0:
            # Exit current and enter new position
            current_position = signal * position_size
            entry_price = price
            trade_peak = 0
        
        # Store current position
        backtest_data.iloc[i, backtest_data.columns.get_loc('position')] = current_position
        backtest_data.iloc[i, backtest_data.columns.get_loc('entry_price')] = entry_price
        backtest_data.iloc[i, backtest_data.columns.get_loc('trade_active')] = (current_position != 0)
    
    # Now calculate returns based on positions
    backtest_data['strategy_return'] = backtest_data['position'] * backtest_data['returns']
    
    # Calculate cumulative returns
    backtest_data['cumulative_market_return'] = (1 + backtest_data['returns']).cumprod()
    backtest_data['cumulative_strategy_return'] = (1 + backtest_data['strategy_return']).cumprod()
    
    # Calculate portfolio value
    backtest_data['market_portfolio'] = initial_capital * backtest_data['cumulative_market_return']
    backtest_data['strategy_portfolio'] = initial_capital * backtest_data['cumulative_strategy_return']
    
    # Calculate drawdowns
    backtest_data['market_peak'] = backtest_data['market_portfolio'].cummax()
    backtest_data['strategy_peak'] = backtest_data['strategy_portfolio'].cummax()
    backtest_data['market_drawdown'] = (backtest_data['market_portfolio'] - backtest_data['market_peak']) / backtest_data['market_peak']
    backtest_data['strategy_drawdown'] = (backtest_data['strategy_portfolio'] - backtest_data['strategy_peak']) / backtest_data['strategy_peak']
    
    # Calculate cumulative returns
    backtest_data['cumulative_market_return'] = (1 + backtest_data['returns']).cumprod()
    backtest_data['cumulative_strategy_return'] = (1 + backtest_data['strategy_return']).cumprod()
    
    # Calculate portfolio value
    backtest_data['market_portfolio'] = initial_capital * backtest_data['cumulative_market_return']
    backtest_data['strategy_portfolio'] = initial_capital * backtest_data['cumulative_strategy_return']
    
    # Calculate drawdowns
    backtest_data['market_peak'] = backtest_data['market_portfolio'].cummax()
    backtest_data['strategy_peak'] = backtest_data['strategy_portfolio'].cummax()
    backtest_data['market_drawdown'] = (backtest_data['market_portfolio'] - backtest_data['market_peak']) / backtest_data['market_peak']
    backtest_data['strategy_drawdown'] = (backtest_data['strategy_portfolio'] - backtest_data['strategy_peak']) / backtest_data['strategy_peak']
    
    # Calculate trading metrics
    total_trades = (backtest_data['position'].diff() != 0).sum()
    winning_trades = (backtest_data['strategy_return'] > 0).sum()
    losing_trades = (backtest_data['strategy_return'] < 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculate risk metrics
    strategy_return = backtest_data['strategy_portfolio'].iloc[-1] / initial_capital - 1
    market_return = backtest_data['market_portfolio'].iloc[-1] / initial_capital - 1
    sharpe_ratio = (backtest_data['strategy_return'].mean() / backtest_data['strategy_return'].std() 
                   * np.sqrt(252) if backtest_data['strategy_return'].std() > 0 else 0)
    max_drawdown = backtest_data['strategy_drawdown'].min()
    
    results = {
        'symbol': symbol,
        'model': 'LSTM',
        'initial_capital': initial_capital,
        'final_value': backtest_data['strategy_portfolio'].iloc[-1],
        'total_return': strategy_return * 100,
        'market_return': market_return * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown * 100,
        'total_trades': total_trades,
        'win_rate': win_rate * 100,
        'backtest_data': backtest_data
    }
    
    if verbose:
        print(f"\nLSTM Strategy Backtest Results ({symbol}):")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Market Return: {results['market_return']:.2f}%")
        print(f"Outperformance: {results['total_return'] - results['market_return']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
    
    return results

def backtest_garch_strategy(symbol, garch_results, original_data, initial_capital=100000, verbose=False):
    """Backtest a trading strategy based on GARCH volatility predictions with improved risk management."""
    # Create a copy of the original data
    backtest_data = original_data.copy()
    
    # Add volatility to the data
    backtest_data['volatility'] = garch_results['volatility']
    
    # Handle NaN values in volatility
    backtest_data['volatility'] = backtest_data['volatility'].fillna(
        backtest_data['returns'].rolling(20).std()  # Use simple std as fallback
    )
    
    # Calculate volatility Z-score (rolling 20-day window)
    backtest_data['vol_zscore'] = (backtest_data['volatility'] - 
                                 backtest_data['volatility'].rolling(20).mean()) / backtest_data['volatility'].rolling(20).std()
    backtest_data['vol_zscore'] = backtest_data['vol_zscore'].fillna(0)
    
    # Calculate volatility percentile (0-1 range)
    backtest_data['vol_percentile'] = backtest_data['volatility'].rolling(60).rank(pct=True)
    backtest_data['vol_percentile'] = backtest_data['vol_percentile'].fillna(0.5)  # Default to middle
    
    # Only trade when volatility is in a reasonable range (avoid extremely high volatility)
    max_vol_percentile = 0.7  # Avoid top 30% highest volatility periods
    
    # Calculate multiple momentum signals with different lookback periods
    backtest_data['mom_1d'] = np.sign(backtest_data['returns'])
    backtest_data['mom_5d'] = np.sign(backtest_data['returns'].rolling(5).mean())
    backtest_data['mom_10d'] = np.sign(backtest_data['returns'].rolling(10).mean())
    backtest_data['mom_20d'] = np.sign(backtest_data['returns'].rolling(20).mean())
    
    # Fill NaN values with 0 (no signal)
    for col in ['mom_1d', 'mom_5d', 'mom_10d', 'mom_20d']:
        backtest_data[col] = backtest_data[col].fillna(0)
    
    # Create consensus momentum signal (majority vote)
    backtest_data['mom_consensus'] = (
        backtest_data['mom_1d'] + 
        backtest_data['mom_5d'] + 
        backtest_data['mom_10d'] + 
        backtest_data['mom_20d']
    )
    
    # Only take strong signals (at least 2 of 4 agree)
    backtest_data['mom_signal'] = np.where(backtest_data['mom_consensus'] >= 2, 1, 
                                       np.where(backtest_data['mom_consensus'] <= -2, -1, 0))
    
    # Risk adjustment factor - reduce position size in high volatility
    # Completely avoid trading in extremely high volatility
    backtest_data['risk_factor'] = np.where(
        backtest_data['vol_percentile'] > max_vol_percentile, 
        0.0,  # No trading in extremely high volatility
        np.maximum(0.1, 1.0 - backtest_data['vol_percentile'])  # Gradually reduce size, but at least 0.1
    )
    
    # Limit position size to reasonable levels (even more conservative)
    backtest_data['position_size'] = backtest_data['risk_factor'] * 0.5  # Maximum 50% of capital at risk
    
    # Initialize portfolio tracking
    backtest_data['position'] = 0
    backtest_data['equity'] = initial_capital
    backtest_data['cash'] = initial_capital
    backtest_data['invested'] = 0
    backtest_data['trade_active'] = False
    backtest_data['entry_price'] = 0
    
    # Apply stop loss
    max_drawdown_allowed = 0.05  # 5% maximum drawdown per trade
    
    # Track trade info
    current_position = 0
    entry_price = 0
    trade_peak = 0
    
    # Implement the strategy with risk management
    for i in range(1, len(backtest_data)):
        price = backtest_data.iloc[i]['close']
        signal = backtest_data.iloc[i]['mom_signal']
        position_size = backtest_data.iloc[i]['position_size']
        
        # If we have an active trade, check for stop loss
        if current_position != 0:
            # Calculate trade return
            if entry_price > 0:
                trade_return = (price / entry_price - 1) * np.sign(current_position)
                
                # Update trade peak if this is the best we've done
                if trade_return > trade_peak:
                    trade_peak = trade_return
                
                # Calculate drawdown from trade peak
                trade_drawdown = max(0, trade_peak - trade_return)
                
                # If drawdown exceeds threshold, exit the trade
                if trade_drawdown >= max_drawdown_allowed:
                    # Close position
                    current_position = 0
                    entry_price = 0
                    trade_peak = 0
                    # Record that we're applying a stop loss
                    if verbose and i % 100 == 0:
                        print(f"Stop loss triggered at {backtest_data.index[i]} with {trade_drawdown:.2%} drawdown")
        
        # If there's a new signal and no current position, enter a new trade
        if current_position == 0 and signal != 0:
            # Size position based on confidence and volatility
            current_position = signal * position_size
            entry_price = price
            trade_peak = 0
        
        # Otherwise, if signal changes direction, exit and reverse
        elif current_position * signal < 0:
            # Exit current and enter new position
            current_position = signal * position_size
            entry_price = price
            trade_peak = 0
        
        # Store current position
        backtest_data.iloc[i, backtest_data.columns.get_loc('position')] = current_position
        backtest_data.iloc[i, backtest_data.columns.get_loc('entry_price')] = entry_price
        backtest_data.iloc[i, backtest_data.columns.get_loc('trade_active')] = (current_position != 0)
    
    # Calculate returns based on positions
    backtest_data['strategy_return'] = backtest_data['position'] * backtest_data['returns']
    
    # Calculate cumulative returns
    backtest_data['cumulative_market_return'] = (1 + backtest_data['returns']).cumprod()
    backtest_data['cumulative_strategy_return'] = (1 + backtest_data['strategy_return']).cumprod()
    
    # Calculate portfolio value
    backtest_data['market_portfolio'] = initial_capital * backtest_data['cumulative_market_return']
    backtest_data['strategy_portfolio'] = initial_capital * backtest_data['cumulative_strategy_return']
    
    # Calculate drawdowns
    backtest_data['market_peak'] = backtest_data['market_portfolio'].cummax()
    backtest_data['strategy_peak'] = backtest_data['strategy_portfolio'].cummax()
    backtest_data['market_drawdown'] = (backtest_data['market_portfolio'] - backtest_data['market_peak']) / backtest_data['market_peak']
    backtest_data['strategy_drawdown'] = (backtest_data['strategy_portfolio'] - backtest_data['strategy_peak']) / backtest_data['strategy_peak']
    
    # Calculate trading metrics
    position_changes = backtest_data['position'].diff().abs()
    total_trades = (position_changes > 0.01).sum()  # Count significant position changes
    
    # Calculate win rate based on daily returns
    winning_days = (backtest_data['strategy_return'] > 0).sum()
    total_days = (backtest_data['strategy_return'] != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    # Calculate risk metrics
    strategy_return = backtest_data['strategy_portfolio'].iloc[-1] / initial_capital - 1
    market_return = backtest_data['market_portfolio'].iloc[-1] / initial_capital - 1
    sharpe_ratio = (backtest_data['strategy_return'].mean() / backtest_data['strategy_return'].std() 
                  * np.sqrt(252) if backtest_data['strategy_return'].std() > 0 else 0)
    max_drawdown = backtest_data['strategy_drawdown'].min()
    
    results = {
        'symbol': symbol,
        'model': 'GARCH',
        'initial_capital': initial_capital,
        'final_value': backtest_data['strategy_portfolio'].iloc[-1],
        'total_return': strategy_return * 100,
        'market_return': market_return * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown * 100,
        'total_trades': total_trades,
        'win_rate': win_rate * 100,
        'backtest_data': backtest_data
    }
    
    if verbose:
        print(f"\nGARCH Strategy Backtest Results ({symbol}):")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Market Return: {results['market_return']:.2f}%")
        print(f"Outperformance: {results['total_return'] - results['market_return']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
    
    return results
    
    # Calculate returns based on positions
    backtest_data['strategy_return'] = backtest_data['position'] * backtest_data['returns']
    
    # Calculate cumulative returns
    backtest_data['cumulative_market_return'] = (1 + backtest_data['returns']).cumprod()
    backtest_data['cumulative_strategy_return'] = (1 + backtest_data['strategy_return']).cumprod()
    
    # Calculate portfolio value
    backtest_data['market_portfolio'] = initial_capital * backtest_data['cumulative_market_return']
    backtest_data['strategy_portfolio'] = initial_capital * backtest_data['cumulative_strategy_return']
    
    # Calculate drawdowns
    backtest_data['market_peak'] = backtest_data['market_portfolio'].cummax()
    backtest_data['strategy_peak'] = backtest_data['strategy_portfolio'].cummax()
    backtest_data['market_drawdown'] = (backtest_data['market_portfolio'] - backtest_data['market_peak']) / backtest_data['market_peak']
    backtest_data['strategy_drawdown'] = (backtest_data['strategy_portfolio'] - backtest_data['strategy_peak']) / backtest_data['strategy_peak']
    
    # Calculate trading metrics
    position_changes = backtest_data['position'].diff().abs()
    total_trades = (position_changes > 0.1).sum()  # Count significant position changes
    
    # Calculate win rate based on daily returns
    winning_days = (backtest_data['strategy_return'] > 0).sum()
    total_days = (backtest_data['strategy_return'] != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    # Calculate risk metrics
    strategy_return = backtest_data['strategy_portfolio'].iloc[-1] / initial_capital - 1
    market_return = backtest_data['market_portfolio'].iloc[-1] / initial_capital - 1
    sharpe_ratio = (backtest_data['strategy_return'].mean() / backtest_data['strategy_return'].std() 
                  * np.sqrt(252) if backtest_data['strategy_return'].std() > 0 else 0)
    max_drawdown = backtest_data['strategy_drawdown'].min()
    
    results = {
        'symbol': symbol,
        'model': 'GARCH',
        'initial_capital': initial_capital,
        'final_value': backtest_data['strategy_portfolio'].iloc[-1],
        'total_return': strategy_return * 100,
        'market_return': market_return * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown * 100,
        'total_trades': total_trades,
        'win_rate': win_rate * 100,
        'backtest_data': backtest_data
    }
    
    if verbose:
        print(f"\nGARCH Strategy Backtest Results ({symbol}):")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Market Return: {results['market_return']:.2f}%")
        print(f"Outperformance: {results['total_return'] - results['market_return']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
    
    return results

def backtest_markov_strategy(symbol, markov_results, original_data, initial_capital=100000, verbose=False):
    """Backtest a trading strategy based on Markov hidden states."""
    # Create a copy of the original data
    backtest_data = original_data.copy()
    
    # Add hidden states
    # Create a smaller dataset with the same transformations as the training data
    # to ensure consistent state assignment
    features = markov_results.get('features', [
        'returns', 'log_returns', 'roc5', 'roc20', 'std20',
        'close_sma10_ratio', 'sma10_sma50_ratio', 'macd_hist', 'rsi'
    ])
    
    # First create advanced features if needed
    if 'roc5' not in backtest_data.columns:
        backtest_data = create_advanced_features(backtest_data)
    
    # Initialize hidden state column with NaNs
    backtest_data['hidden_state'] = np.nan
    
    # Select only the necessary features for prediction and remove NaNs
    features_data = backtest_data[features].copy()
    valid_rows = ~np.isnan(features_data).any(axis=1)
    features_data = features_data[valid_rows]
    
    if len(features_data) > 0:
        # Scale the data
        X = features_data.values
        X_scaled = markov_results['scaler'].transform(X)
        
        # Predict states for test data
        predicted_states = markov_results['model'].predict(X_scaled)
        
        # Put states back into the backtest_data where we have valid data
        valid_indices = np.where(valid_rows)[0]
        backtest_data.loc[backtest_data.index[valid_indices], 'hidden_state'] = predicted_states
        
        # Forward fill to handle any remaining NaNs
        backtest_data['hidden_state'] = backtest_data['hidden_state'].ffill().bfill()
    
    # Forward fill to handle NaNs
    backtest_data['hidden_state'] = backtest_data['hidden_state'].ffill()
    
    # Map state labels
    backtest_data['state_label'] = backtest_data['hidden_state'].map(markov_results['labels'])
    
    # Create trading signals based on states with more conservative approach
    state_signals = {}
    for state, stats in markov_results['state_stats'].items():
        # More conservative approach: only take positions in states with strong statistical evidence
        # and reasonable sample size
        
        # Calculate signal strength based on:
        # 1. Mean return (magnitude and sign)
        # 2. Statistical significance (mean / std)
        # 3. Sample size (more samples = more confidence)
        
        mean_return = stats['mean']
        std_dev = stats['std']
        sample_count = stats['count']
        total_samples = sum(s['count'] for s in markov_results['state_stats'].values())
        
        # Calculate t-statistic for mean return
        t_stat = mean_return / (std_dev / np.sqrt(sample_count)) if std_dev > 0 else 0
        
        # Calculate sample weight (more samples = more confidence)
        sample_weight = min(1.0, sample_count / 100)  # Cap at 100 samples
        
        # Only take positions if:
        # 1. State has significant historical returns (|t-stat| > 1.5)
        # 2. Enough samples for statistical confidence
        # 3. Mean return exceeds minimum threshold
        
        min_t_stat = 1.5
        min_samples_pct = 0.05  # At least 5% of all samples
        min_abs_return = 0.0005  # Minimum daily return magnitude (about 0.05%)
        
        if (abs(t_stat) > min_t_stat and
            sample_count / total_samples > min_samples_pct and
            abs(mean_return) > min_abs_return):
            
            # For strong positive returns
            if mean_return > 0.001 and t_stat > 2.0:
                # Strong position with weight based on statistical confidence
                state_signals[state] = 0.8 * sample_weight  # Long, max 80% position
            # For weak positive returns
            elif mean_return > 0:
                state_signals[state] = 0.4 * sample_weight  # Reduced long, max 40% position
            # For strong negative returns
            elif mean_return < -0.001 and t_stat < -2.0:
                state_signals[state] = -0.8 * sample_weight  # Short, max 80% position
            # For weak negative returns
            elif mean_return < 0:
                state_signals[state] = -0.4 * sample_weight  # Reduced short, max 40% position
        else:
            # No position for states without statistical confidence
            state_signals[state] = 0.0
    
    if verbose:
        print("\nMarkov State Signals (after statistical filtering):")
        for state, signal in state_signals.items():
            state_label = markov_results['labels'].get(state, f"State_{state}")
            print(f"  {state_label} (State {state}): {signal:+.2f}")
    
    # Initialize portfolio tracking
    backtest_data['signal'] = backtest_data['hidden_state'].map(state_signals).fillna(0)
    
    # Cap position size to prevent extreme leverage
    max_position_size = 0.5  # Maximum 50% of capital
    backtest_data['signal'] = np.clip(backtest_data['signal'], -max_position_size, max_position_size)
    
    backtest_data['position'] = 0
    backtest_data['entry_price'] = 0
    backtest_data['trade_active'] = False
    
    # Apply stop loss
    max_drawdown_allowed = 0.05  # 5% maximum drawdown per trade
    
    # Track trade info
    current_position = 0
    entry_price = 0
    trade_peak = 0
    
    # Implement the strategy with risk management
    for i in range(1, len(backtest_data)):
        price = backtest_data.iloc[i]['close']
        signal = backtest_data.iloc[i]['signal']
        
        # If we have an active trade, check for stop loss
        if current_position != 0:
            # Calculate trade return
            if entry_price > 0:
                trade_return = (price / entry_price - 1) * np.sign(current_position)
                
                # Update trade peak if this is the best we've done
                if trade_return > trade_peak:
                    trade_peak = trade_return
                
                # Calculate drawdown from trade peak (make sure it's positive)
                trade_drawdown = max(0, trade_peak - trade_return)
                
                # If drawdown exceeds threshold, exit the trade
                if trade_drawdown >= max_drawdown_allowed:
                    # Close position
                    current_position = 0
                    entry_price = 0
                    trade_peak = 0
                    # Record that we're applying a stop loss
                    if verbose and i % 100 == 0:
                        print(f"Stop loss triggered at {backtest_data.index[i]} with {trade_drawdown:.2%} drawdown")
        
        # Signal change - only trade if there's a strong enough signal
        if abs(signal) > 0.1:  # Minimum threshold for trading
            # If no position and new signal, enter position
            if current_position == 0:
                current_position = signal
                entry_price = price
                trade_peak = 0
            # If position exists but signal changed significantly, adjust position
            elif np.sign(current_position) != np.sign(signal) or abs(current_position - signal) > 0.3:
                current_position = signal
                entry_price = price
                trade_peak = 0
        
        # Store current position
        backtest_data.iloc[i, backtest_data.columns.get_loc('position')] = current_position
        backtest_data.iloc[i, backtest_data.columns.get_loc('entry_price')] = entry_price
        backtest_data.iloc[i, backtest_data.columns.get_loc('trade_active')] = (current_position != 0)
    
    # Calculate returns based on positions
    backtest_data['strategy_return'] = backtest_data['position'] * backtest_data['returns']
    
    # Calculate cumulative returns
    backtest_data['cumulative_market_return'] = (1 + backtest_data['returns']).cumprod()
    backtest_data['cumulative_strategy_return'] = (1 + backtest_data['strategy_return']).cumprod()
    
    # Calculate portfolio value
    backtest_data['market_portfolio'] = initial_capital * backtest_data['cumulative_market_return']
    backtest_data['strategy_portfolio'] = initial_capital * backtest_data['cumulative_strategy_return']
    
    # Calculate drawdowns
    backtest_data['market_peak'] = backtest_data['market_portfolio'].cummax()
    backtest_data['strategy_peak'] = backtest_data['strategy_portfolio'].cummax()
    backtest_data['market_drawdown'] = (backtest_data['market_portfolio'] - backtest_data['market_peak']) / backtest_data['market_peak']
    backtest_data['strategy_drawdown'] = (backtest_data['strategy_portfolio'] - backtest_data['strategy_peak']) / backtest_data['strategy_peak']
    
    # Calculate cumulative returns
    backtest_data['cumulative_market_return'] = (1 + backtest_data['returns']).cumprod()
    backtest_data['cumulative_strategy_return'] = (1 + backtest_data['strategy_return']).cumprod()
    
    # Calculate portfolio value
    backtest_data['market_portfolio'] = initial_capital * backtest_data['cumulative_market_return']
    backtest_data['strategy_portfolio'] = initial_capital * backtest_data['cumulative_strategy_return']
    
    # Calculate drawdowns
    backtest_data['market_peak'] = backtest_data['market_portfolio'].cummax()
    backtest_data['strategy_peak'] = backtest_data['strategy_portfolio'].cummax()
    backtest_data['market_drawdown'] = (backtest_data['market_portfolio'] - backtest_data['market_peak']) / backtest_data['market_peak']
    backtest_data['strategy_drawdown'] = (backtest_data['strategy_portfolio'] - backtest_data['strategy_peak']) / backtest_data['strategy_peak']
    
    # Calculate trading metrics
    state_changes = backtest_data['hidden_state'].diff() != 0
    total_trades = state_changes.sum()
    
    # Calculate win rate based on state periods
    regime_returns = []
    current_state = None
    regime_start = 0
    
    for i, row in backtest_data.iterrows():
        if row['hidden_state'] != current_state or i == backtest_data.index[-1]:
            if current_state is not None:
                # Calculate return for this regime
                end_idx = backtest_data.index.get_loc(i) if i != backtest_data.index[-1] else len(backtest_data)
                start_value = backtest_data['strategy_portfolio'].iloc[regime_start]
                end_value = backtest_data['strategy_portfolio'].iloc[end_idx-1]
                regime_return = (end_value / start_value) - 1
                regime_returns.append(regime_return)
            
            current_state = row['hidden_state']
            regime_start = backtest_data.index.get_loc(i)
    
    winning_regimes = sum(1 for r in regime_returns if r > 0)
    win_rate = winning_regimes / len(regime_returns) if regime_returns else 0
    
    # Calculate risk metrics
    strategy_return = backtest_data['strategy_portfolio'].iloc[-1] / initial_capital - 1
    market_return = backtest_data['market_portfolio'].iloc[-1] / initial_capital - 1
    sharpe_ratio = (backtest_data['strategy_return'].mean() / backtest_data['strategy_return'].std() 
                  * np.sqrt(252) if backtest_data['strategy_return'].std() > 0 else 0)
    max_drawdown = backtest_data['strategy_drawdown'].min()
    
    results = {
        'symbol': symbol,
        'model': 'Markov',
        'initial_capital': initial_capital,
        'final_value': backtest_data['strategy_portfolio'].iloc[-1],
        'total_return': strategy_return * 100,
        'market_return': market_return * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown * 100,
        'total_trades': total_trades,
        'win_rate': win_rate * 100,
        'backtest_data': backtest_data,
        'state_signals': state_signals,
        'regime_returns': regime_returns
    }
    
    if verbose:
        print(f"\nMarkov Strategy Backtest Results ({symbol}):")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Market Return: {results['market_return']:.2f}%")
        print(f"Outperformance: {results['total_return'] - results['market_return']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {results['max_drawdown']:.2f}%")
        print(f"State Transition Trades: {results['total_trades']}")
        print(f"Regime Win Rate: {results['win_rate']:.2f}%")
        print(f"\nState Signals:")
        for state, signal in state_signals.items():
            state_label = markov_results['labels'].get(state, f"State_{state}")
            print(f"  {state_label} (State {state}): {signal:+.1f}")
    
    return results

def plot_performance(result, symbol, model_name):
    """Plot backtest performance."""
    data = result['backtest_data']
    
    plt.figure(figsize=(12, 8))
    
    # Plot portfolio values
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['market_portfolio'], label='Buy & Hold')
    plt.plot(data.index, data['strategy_portfolio'], label=f'{model_name} Strategy')
    plt.title(f'{symbol} - {model_name} Strategy vs Buy & Hold')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()
    
    # Plot drawdowns
    plt.subplot(2, 1, 2)
    plt.plot(data.index, data['market_drawdown'] * 100, label='Buy & Hold')
    plt.plot(data.index, data['strategy_drawdown'] * 100, label=f'{model_name} Strategy')
    plt.title('Drawdowns')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = f'{symbol}_{model_name}_backtest.png'
    plt.savefig(output_path)
    print(f"Performance chart saved to {output_path}")
    
    return output_path

def plot_combined_performance(results, symbol, initial_capital):
    """Plot combined performance of all models."""
    plt.figure(figsize=(12, 10))
    
    # Get baseline buy & hold data from the first result
    market_data = results[0]['backtest_data']['market_portfolio']
    
    # Plot portfolio values
    plt.subplot(2, 1, 1)
    plt.plot(market_data.index, market_data, label='Buy & Hold')
    
    for result in results:
        model_name = result['model']
        strategy_data = result['backtest_data']['strategy_portfolio']
        plt.plot(strategy_data.index, strategy_data, label=f'{model_name} Strategy')
    
    plt.title(f'{symbol} - All Strategies vs Buy & Hold')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()
    
    # Plot drawdowns
    plt.subplot(2, 1, 2)
    plt.plot(market_data.index, results[0]['backtest_data']['market_drawdown'] * 100, label='Buy & Hold')
    
    for result in results:
        model_name = result['model']
        drawdown_data = result['backtest_data']['strategy_drawdown'] * 100
        plt.plot(drawdown_data.index, drawdown_data, label=f'{model_name} Strategy')
    
    plt.title('Drawdowns')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = f'{symbol}_combined_backtest.png'
    plt.savefig(output_path)
    print(f"\nCombined performance chart saved to {output_path}")
    
    # Also create a bar chart of returns
    plt.figure(figsize=(10, 6))
    
    # Extract returns and names
    model_names = [r['model'] for r in results]
    returns = [r['total_return'] for r in results]
    market_return = results[0]['market_return']
    
    # Add buy & hold to the comparison
    model_names.append('Buy & Hold')
    returns.append(market_return)
    
    # Create bar chart
    colors = ['green' if r > market_return else 'red' for r in returns]
    colors[-1] = 'blue'  # Buy & hold is blue
    
    plt.bar(model_names, returns, color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title(f'{symbol} - Strategy Returns Comparison')
    plt.ylabel('Total Return (%)')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add return values on top of bars
    for i, v in enumerate(returns):
        plt.text(i, v + (5 if v > 0 else -10), f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    
    # Save figure
    output_path = f'{symbol}_returns_comparison.png'
    plt.savefig(output_path)
    print(f"Returns comparison chart saved to {output_path}")
    
    return output_path

def save_results_summary(all_results, initial_capital):
    """Save a summary of all backtest results."""
    # Create a DataFrame from results
    summary_data = []
    
    for symbol_results in all_results.values():
        for result in symbol_results:
            summary_data.append({
                'Symbol': result['symbol'],
                'Model': result['model'],
                'Initial Capital': result['initial_capital'],
                'Final Value': result['final_value'],
                'Total Return (%)': result['total_return'],
                'Market Return (%)': result['market_return'],
                'Outperformance (%)': result['total_return'] - result['market_return'],
                'Sharpe Ratio': result['sharpe_ratio'],
                'Max Drawdown (%)': result['max_drawdown'],
                'Total Trades': result['total_trades'],
                'Win Rate (%)': result['win_rate']
            })
    
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    output_path = 'ml_backtest_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\nResults summary saved to {output_path}")
    
    # Print overall summary
    print("\n=== OVERALL BACKTEST RESULTS ===")
    print(f"Initial capital per strategy: ${initial_capital:,.2f}")
    print(f"Best performing strategy: {df.loc[df['Total Return (%)'].idxmax(), 'Symbol']} - {df.loc[df['Total Return (%)'].idxmax(), 'Model']}")
    print(f"Best return: {df['Total Return (%)'].max():.2f}%")
    print(f"Worst performing strategy: {df.loc[df['Total Return (%)'].idxmin(), 'Symbol']} - {df.loc[df['Total Return (%)'].idxmin(), 'Model']}")
    print(f"Worst return: {df['Total Return (%)'].min():.2f}%")
    print(f"Average return across all strategies: {df['Total Return (%)'].mean():.2f}%")
    print(f"Average outperformance: {df['Outperformance (%)'].mean():.2f}%")
    
    # Calculate portfolio performance if equal amounts were invested in all strategies
    total_final_value = df['Final Value'].sum()
    total_initial = df['Initial Capital'].sum()
    total_return = (total_final_value / total_initial - 1) * 100
    
    print(f"\nCombined portfolio performance:")
    print(f"Total initial capital: ${total_initial:,.2f}")
    print(f"Total final value: ${total_final_value:,.2f}")
    print(f"Total return: {total_return:.2f}%")
    
    return df

def main():
    """Main execution function."""
    args = parse_args()
    
    if args.mode == 'backtest':
        print("\n==========================================")
        print("THOR TRADING SYSTEM - ML BACKTEST TOOL")
        print("==========================================\n")
    else:
        print("\n=================================================")
        print("THOR TRADING SYSTEM - ML PAPER TRADING SIMULATOR")
        print("=================================================\n")
    
    # Configure operation mode
    mode = args.mode
    print(f"Operation mode: {mode}")
    
    # Create output directory if specified
    output_dir = args.output_dir
    if output_dir != '.':
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # Validate dates for backtest mode
    if mode == 'backtest':
        if args.start and args.end:
            start_date = args.start
            end_date = args.end
            print(f"Backtest period: {start_date} to {end_date}")
        else:
            start_date = None
            end_date = None
            print("Using all available data for backtest")
    else:
        # For paper trading, we'll determine dates differently
        start_date = None
        end_date = None
        print(f"Paper trading simulation days: {args.paper_days}")
    
    # Get list of symbols
    symbols = args.symbols.split(',')
    print(f"Symbols: {', '.join(symbols)}")
    
    # Get list of models to use
    models = args.models.split(',')
    print(f"Models: {', '.join(models)}")
    
    # Initialize database connection
    try:
        db = PostgresConnector()
        if not db.test_connection():
            print("Error: Could not connect to database")
            sys.exit(1)
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        print("Continuing with mock data for demonstration")
        db = PostgresConnector()
    
    # Different flows for backtest vs paper trading
    if mode == 'backtest':
        # Store all results
        all_results = {}
        
        # Process each symbol for backtesting
        for symbol in symbols:
            print(f"\n=== Processing {symbol} ===")
            
            # Load data
            data = load_data(db, symbol, start_date, end_date)
            if data is None:
                print(f"Skipping {symbol} due to missing data")
                continue
            
            print(f"Loaded {len(data)} days of data for {symbol}")
            
            # Store results for this symbol
            symbol_results = []
            
            # Process LSTM model
            if 'lstm' in models:
                print("\nTraining LSTM model...")
                lstm_data = preprocess_data(data)
                lstm_results = train_lstm_model(lstm_data, args.verbose)
                
                print("Backtesting LSTM strategy...")
                lstm_backtest = backtest_lstm_strategy(symbol, lstm_results, data, args.capital, args.verbose)
                symbol_results.append(lstm_backtest)
                
                # Save model performance plot
                if args.verbose:
                    plot_performance(lstm_backtest, symbol, 'LSTM')
            
            # Process GARCH model
            if 'garch' in models:
                print("\nTraining GARCH model...")
                garch_results = train_garch_model(data, args.verbose)
                
                print("Backtesting GARCH strategy...")
                garch_backtest = backtest_garch_strategy(symbol, garch_results, data, args.capital, args.verbose)
                symbol_results.append(garch_backtest)
                
                # Save model performance plot
                if args.verbose:
                    plot_performance(garch_backtest, symbol, 'GARCH')
            
            # Process Markov model
            if 'markov' in models:
                print("\nTraining Markov model...")
                markov_results = train_markov_model(data, n_states=3, verbose=args.verbose)
                
                print("Backtesting Markov strategy...")
                # Import the fixed Markov backtest function to avoid weird bugs
                from fixed_markov_backtest import backtest_markov_strategy as fixed_backtest_markov
                markov_backtest = fixed_backtest_markov(symbol, markov_results, data, args.capital, args.verbose)
                symbol_results.append(markov_backtest)
                
                # Save model performance plot
                if args.verbose:
                    plot_performance(markov_backtest, symbol, 'Markov')
            
            # Plot combined performance for this symbol
            if len(symbol_results) > 1:
                plot_combined_performance(symbol_results, symbol, args.capital)
            
            # Store results for this symbol
            all_results[symbol] = symbol_results
        
        # Save and print overall results summary
        if all_results:
            summary_df = save_results_summary(all_results, args.capital)
        
        print("\nBacktest process complete!")
        
    else:  # Paper trading mode
        # Import paper trading functions
        from ml_backtest_paper_trade import run_paper_trading, train_models_for_paper_trading
        
        # First load all data for training
        data_dict = {}
        for symbol in symbols:
            # Load all available historical data for training (back to 1992 or the earliest available)
            training_end_date = datetime.now()
            training_start_date = datetime(1992, 1, 1)  # Use data all the way back to 1992
            
            print(f"\n=== Loading training data for {symbol} ===")
            print(f"Using full historical data from {training_start_date.strftime('%Y-%m-%d')} to {training_end_date.strftime('%Y-%m-%d')}")
            data = load_data(db, symbol, training_start_date.strftime('%Y-%m-%d'), training_end_date.strftime('%Y-%m-%d'))
            
            if data is None or len(data) < 100:  # Need sufficient data for training
                print(f"Insufficient data for {symbol}, skipping")
                continue
                
            print(f"Loaded {len(data)} days of training data")
            data_dict[symbol] = data
        
        if not data_dict:
            print("No usable data found for any symbols. Exiting.")
            return
            
        # Train all models for paper trading
        print("\n=== Training models for paper trading ===")
        trained_models = train_models_for_paper_trading(data_dict, models, args.output_dir, args.verbose)
        
        # Run paper trading simulation
        paper_results = run_paper_trading(
            db, trained_models, list(data_dict.keys()), 
            args.capital, args.paper_days, args.output_dir, args.verbose
        )
    
    print("\nProcess complete!")

if __name__ == "__main__":
    main()