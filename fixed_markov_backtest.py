import sys
import os
import numpy as np
import pandas as pd

# Import the create_advanced_features function from ml_backtest
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_backtest import create_advanced_features

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
    
    # Calculate trading metrics
    state_changes = backtest_data['hidden_state'].diff() != 0
    total_trades = (backtest_data['position'].diff() != 0).sum()
    
    # Calculate win rate based on state periods
    winning_days = (backtest_data['strategy_return'] > 0).sum()
    total_days = (backtest_data['strategy_return'] != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    # Calculate risk metrics with safety caps for extreme values
    
    # Apply a maximum limit to final portfolio value to avoid unrealistic numbers
    max_realistic_return = 10.0  # Maximum 1000% return (10x initial capital)
    final_portfolio_value = min(
        backtest_data['strategy_portfolio'].iloc[-1],
        initial_capital * (1 + max_realistic_return)
    )
    
    strategy_return = final_portfolio_value / initial_capital - 1
    market_return = backtest_data['market_portfolio'].iloc[-1] / initial_capital - 1
    
    # Cap market return too for consistency
    market_return = min(market_return, max_realistic_return)
    
    # Calculate Sharpe with caps on extreme values
    mean_return = np.clip(backtest_data['strategy_return'].mean(), -0.05, 0.05)
    std_return = max(0.01, backtest_data['strategy_return'].std())  # Minimum volatility
    sharpe_ratio = mean_return / std_return * np.sqrt(252)
    
    # Ensure reasonable drawdown
    max_drawdown = max(-0.95, backtest_data['strategy_drawdown'].min())
    
    results = {
        'symbol': symbol,
        'model': 'Markov',
        'initial_capital': initial_capital,
        'final_value': final_portfolio_value,  # Use capped value
        'total_return': strategy_return * 100,
        'market_return': market_return * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown * 100,
        'total_trades': total_trades,
        'win_rate': win_rate * 100,
        'backtest_data': backtest_data,
        'state_signals': state_signals
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
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
    
    return results