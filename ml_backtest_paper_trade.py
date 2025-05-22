import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import necessary functions from ml_backtest
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from arch import arch_model
from hmmlearn import hmm

# These functions need to be imported from ml_backtest
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_backtest import preprocess_data, train_lstm_model, train_garch_model, train_markov_model, load_data

def run_paper_trading(db, models_dict, symbols, initial_capital, days_to_simulate=30, output_dir='.', verbose=False):
    """Run paper trading simulation using trained models."""
    print("\n=== Running Paper Trading Simulation ===")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Models: {', '.join(models_dict.keys())}")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print(f"Simulation days: {days_to_simulate}")
    
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define paper trading simulation period
    # For simulation, we'll use the most recent data as "live" data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_simulate)
    
    print(f"Simulation period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Store trades for each model and symbol
    all_trades = {}
    all_portfolio_values = {}
    
    # Initialize trading logs
    trade_log_path = os.path.join(output_dir, 'paper_trading_log.csv')
    trade_log_file = open(trade_log_path, 'w')
    trade_log_file.write('timestamp,symbol,model,action,price,quantity,position_value,portfolio_value\n')
    
    # Run simulation for each symbol
    for symbol in symbols:
        print(f"\n=== Paper Trading {symbol} ===")
        
        # Load data for this symbol
        data = load_data(db, symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if data is None or len(data) < 5:  # Need at least 5 days for basic indicators
            print(f"Insufficient data for {symbol}, skipping")
            continue
            
        print(f"Loaded {len(data)} days of data for paper trading simulation")
        
        # For each model, run paper trading
        for model_name, model in models_dict.items():
            print(f"Running paper trading for {model_name} model...")
            
            # Initialize portfolio tracker
            portfolio = {
                'cash': initial_capital,
                'position': 0,
                'position_value': 0,
                'trades': [],
                'daily_values': []
            }
            
            # Process each day in sequence
            for i in range(1, len(data)):  # Start from second day to have previous data
                current_date = data.index[i]
                current_row = data.iloc[i]
                prev_row = data.iloc[i-1]
                
                # Generate signal based on model type
                signal = 0
                if model_name == 'LSTM':
                    # For LSTM, predict price movement
                    prediction = model['predict_func'](data.iloc[:i])
                    if prediction > current_row['close']:
                        signal = 1  # Buy signal
                    else:
                        signal = -1  # Sell signal
                        
                elif model_name == 'GARCH':
                    # For GARCH, use volatility forecast
                    vol_forecast = model['predict_func'](data.iloc[:i])
                    momentum = data['returns'].iloc[max(0, i-5):i].mean()
                    signal = np.sign(momentum)
                    # Adjust position size based on volatility
                    if vol_forecast > data['returns'].iloc[max(0, i-20):i].std() * 1.5:
                        signal *= 0.5  # Reduce position in high volatility
                        
                elif model_name == 'Markov':
                    # For Markov, predict regime
                    regime = model['predict_func'](data.iloc[:i])
                    # Map regime to signal based on historical performance
                    if regime in model['regime_signals']:
                        signal = model['regime_signals'][regime]
                
                # Current market price (we'll use close for simplicity)
                price = current_row['close']
                
                # Determine action based on signal vs current position
                action = 'HOLD'
                quantity = 0
                
                # Simple position management
                target_position = int(signal * (portfolio['cash'] / price))
                
                if target_position > portfolio['position']:
                    # Buy
                    quantity = target_position - portfolio['position']
                    action = 'BUY'
                elif target_position < portfolio['position']:
                    # Sell
                    quantity = portfolio['position'] - target_position
                    action = 'SELL'
                
                # Execute trade if needed
                if quantity > 0:
                    if action == 'BUY':
                        cost = quantity * price
                        if cost <= portfolio['cash']:
                            portfolio['cash'] -= cost
                            portfolio['position'] += quantity
                        else:
                            # Adjust quantity based on available cash
                            quantity = int(portfolio['cash'] / price)
                            cost = quantity * price
                            portfolio['cash'] -= cost
                            portfolio['position'] += quantity
                    else:  # SELL
                        proceed = quantity * price
                        portfolio['cash'] += proceed
                        portfolio['position'] -= quantity
                    
                    # Log the trade
                    trade = {
                        'timestamp': current_date,
                        'action': action,
                        'price': price,
                        'quantity': quantity,
                        'value': quantity * price
                    }
                    portfolio['trades'].append(trade)
                    
                    # Write to trade log
                    trade_log_file.write(f"{current_date},{symbol},{model_name},{action},{price},{quantity},"\
                                       f"{portfolio['position'] * price},{portfolio['cash'] + portfolio['position'] * price}\n")
                    
                    if verbose:
                        print(f"{current_date}: {action} {quantity} units of {symbol} at ${price:.2f}")
                
                # Update position value and total portfolio value
                portfolio['position_value'] = portfolio['position'] * price
                total_value = portfolio['cash'] + portfolio['position_value']
                
                # Record daily portfolio value
                portfolio['daily_values'].append({
                    'date': current_date,
                    'cash': portfolio['cash'],
                    'position': portfolio['position'],
                    'position_value': portfolio['position_value'],
                    'total_value': total_value
                })
            
            # Calculate final metrics
            initial_value = initial_capital
            final_value = portfolio['cash'] + portfolio['position_value']
            total_return = (final_value / initial_value - 1) * 100
            
            # Convert daily values to DataFrame for analysis
            daily_df = pd.DataFrame(portfolio['daily_values'])
            daily_df.set_index('date', inplace=True)
            
            # Calculate additional metrics
            if len(daily_df) > 1:
                daily_returns = daily_df['total_value'].pct_change().dropna()
                sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
                max_drawdown = ((daily_df['total_value'] / daily_df['total_value'].cummax()) - 1).min() * 100
            else:
                sharpe_ratio = 0
                max_drawdown = 0
            
            # Print summary
            print(f"\n{model_name} Paper Trading Results for {symbol}:")
            print(f"Initial Capital: ${initial_value:,.2f}")
            print(f"Final Value: ${final_value:,.2f}")
            print(f"Total Return: {total_return:.2f}%")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2f}%")
            print(f"Total Trades: {len(portfolio['trades'])}")
            
            # Plot equity curve
            plt.figure(figsize=(10, 6))
            plt.plot(daily_df.index, daily_df['total_value'])
            plt.title(f"{symbol} - {model_name} Paper Trading Equity Curve")
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            
            # Save figure
            output_path = os.path.join(output_dir, f"{symbol}_{model_name}_paper_trading.png")
            plt.savefig(output_path)
            plt.close()
            
            # Store results
            if symbol not in all_trades:
                all_trades[symbol] = {}
            all_trades[symbol][model_name] = portfolio['trades']
            
            if symbol not in all_portfolio_values:
                all_portfolio_values[symbol] = {}
            all_portfolio_values[symbol][model_name] = daily_df
    
    # Close trade log
    trade_log_file.close()
    
    # Save combined CSV of all portfolio values
    combined_values = []
    for symbol in all_portfolio_values:
        for model_name in all_portfolio_values[symbol]:
            df = all_portfolio_values[symbol][model_name].copy()
            df['symbol'] = symbol
            df['model'] = model_name
            combined_values.append(df.reset_index())
    
    if combined_values:
        combined_df = pd.concat(combined_values)
        output_path = os.path.join(output_dir, "paper_trading_results.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"\nPaper trading results saved to {output_path}")
    
    print(f"\nPaper trading log saved to {trade_log_path}")
    print("\nPaper trading simulation complete!")
    
    return all_trades, all_portfolio_values

def train_models_for_paper_trading(data_dict, models_to_train, output_dir='.', verbose=False):
    """Train models and return prediction functions for paper trading.
    Also saves trained models to disk for future use."""
    trained_models = {}
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    print(f"Models will be saved to {os.path.join(output_dir, 'models')}")
    
    for symbol, data in data_dict.items():
        trained_models[symbol] = {}
        
        # Preprocess data for LSTM
        lstm_data = None
        if 'lstm' in models_to_train:
            lstm_data = preprocess_data(data)
        
        # Train LSTM model
        if 'lstm' in models_to_train:
            print(f"Training LSTM model for {symbol}...")
            lstm_results = train_lstm_model(lstm_data, verbose)
            
            # Create prediction function
            def lstm_predict_func(new_data, lstm_model=lstm_results['model'], scaler=lstm_data['scaler']):
                # Prepare input data
                sequence_length = 10  # Must match training
                if len(new_data) <= sequence_length:
                    return new_data['close'].iloc[-1]  # Not enough data, return last price
                
                # Extract features
                X = new_data[['close', 'returns', 'log_returns']].iloc[-sequence_length:].copy()
                X_scaled = scaler.transform(X)
                X_reshaped = X_scaled.reshape(1, sequence_length, 3)
                
                # Make prediction
                pred_scaled = lstm_model.predict(X_reshaped)[0][0]
                
                # Inverse transform - create array with right number of features
                num_features = X_scaled.shape[1]
                pred_copy = np.zeros((1, num_features))
                pred_copy[0, 0] = pred_scaled  # Place prediction in first column (close price)
                pred_value = scaler.inverse_transform(pred_copy)[0][0]
                
                return pred_value
            
            # Save LSTM model
            model_path = os.path.join(output_dir, 'models', f'{symbol}_lstm_model')
            try:
                lstm_results['model'].save(model_path)
                print(f"Saved LSTM model to {model_path}")
            except Exception as e:
                print(f"Error saving LSTM model: {str(e)}")
            
            trained_models[symbol]['LSTM'] = {
                'model': lstm_results['model'],
                'predict_func': lstm_predict_func,
                'model_path': model_path
            }
        
        # Train GARCH model
        if 'garch' in models_to_train:
            print(f"Training GARCH model for {symbol}...")
            garch_results = train_garch_model(data, verbose)
            
            # Create prediction function
            def garch_predict_func(new_data, garch_model=garch_results['model']):
                # Need at least 30 observations for reliable GARCH
                if len(new_data) < 30:
                    return new_data['returns'].std()
                
                # Extract log returns
                returns = new_data['log_returns'].values
                
                # Update model with new data
                updated_model = arch_model(returns, vol='GARCH', p=1, q=1, mean='AR', lags=1)
                try:
                    updated_fit = updated_model.fit(disp='off', show_warning=False)
                    forecast = updated_fit.forecast(horizon=1)
                    # Return forecasted volatility
                    return forecast.variance.values[-1][0] ** 0.5
                except:
                    # If fitting fails, return recent volatility
                    return returns[-20:].std()
            
            # Save GARCH model
            model_path = os.path.join(output_dir, 'models', f'{symbol}_garch_model.pkl')
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(garch_results['model'], f)
                print(f"Saved GARCH model to {model_path}")
            except Exception as e:
                print(f"Error saving GARCH model: {str(e)}")
            
            trained_models[symbol]['GARCH'] = {
                'model': garch_results['model'],
                'predict_func': garch_predict_func,
                'model_path': model_path
            }
        
        # Train Markov model
        if 'markov' in models_to_train:
            print(f"Training Markov model for {symbol}...")
            markov_results = train_markov_model(data, n_states=3, verbose=verbose)
            
            # Map states to signals based on historical returns
            regime_signals = {}
            for state, stats in markov_results['state_stats'].items():
                if stats['mean'] > 0.001:
                    regime_signals[state] = 1.0  # Long
                elif stats['mean'] < -0.001:
                    regime_signals[state] = -1.0  # Short
                elif stats['mean'] > 0:
                    regime_signals[state] = 0.5  # Reduced long
                elif stats['mean'] < 0:
                    regime_signals[state] = -0.5  # Reduced short
                else:
                    regime_signals[state] = 0  # Neutral
            
            # Create prediction function
            def markov_predict_func(new_data, hmm_model=markov_results['model'], scaler=markov_results['scaler']):
                if len(new_data) < 20:  # Need at least 20 days of data
                    return 0  # Neutral state
                
                # Create features for prediction
                try:
                    X = np.column_stack([
                        new_data['returns'].values,
                        new_data['log_returns'].values,
                        new_data['close'].pct_change(5).values,
                        new_data['close'].pct_change(20).values
                    ])
                    
                    # Remove NaN rows
                    X = X[~np.isnan(X).any(axis=1)]
                    
                    if len(X) == 0:
                        return 0  # Not enough data
                    
                    # Scale data
                    X_scaled = scaler.transform(X[-1:])  # Just use the last observation
                    
                    # Predict state
                    state = hmm_model.predict(X_scaled)[0]
                    return state
                except:
                    return 0  # Default to neutral on error
            
            # Save Markov model
            model_path = os.path.join(output_dir, 'models', f'{symbol}_markov_model.pkl')
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(markov_results['model'], f)
                print(f"Saved Markov model to {model_path}")
                
                # Also save regime signals
                signals_path = os.path.join(output_dir, 'models', f'{symbol}_markov_signals.pkl')
                with open(signals_path, 'wb') as f:
                    pickle.dump(regime_signals, f)
                print(f"Saved Markov regime signals to {signals_path}")
            except Exception as e:
                print(f"Error saving Markov model: {str(e)}")
            
            trained_models[symbol]['Markov'] = {
                'model': markov_results['model'],
                'predict_func': markov_predict_func,
                'regime_signals': regime_signals,
                'model_path': model_path,
                'signals_path': signals_path
            }
    
    return trained_models