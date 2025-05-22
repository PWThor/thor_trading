import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
import pickle
import json
import logging
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connectors.postgres_connector import PostgresConnector
from features.feature_generator_wrapper import create_feature_generator
from models.training.model_trainer import ModelTrainer

# Set up logging
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Engine for backtesting trading strategies on historical data.
    """
    
    def __init__(
        self,
        db_connector: PostgresConnector,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
        position_size_pct: float = 0.02,
        max_positions: int = 3,
        commission_per_contract: float = 2.50,
        slippage_pct: float = 0.001,
        output_dir: str = None
    ):
        """
        Initialize the backtest engine.
        
        Args:
            db_connector: Database connector
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Initial capital for backtest
            position_size_pct: Position size as percentage of capital
            max_positions: Maximum number of concurrent positions
            commission_per_contract: Commission per contract
            slippage_pct: Slippage as percentage of price
            output_dir: Directory for backtest output
        """
        self.db = db_connector
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.commission_per_contract = commission_per_contract
        self.slippage_pct = slippage_pct
        
        # Initialize the feature generator
        self.feature_generator = create_feature_generator()
        
        # Set up output directory
        if output_dir is None:
            self.output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'results',
                f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data structures
        self.market_data = None
        self.feature_data = None
        self.trades = []
        self.daily_portfolio = []
        self.current_positions = []
        self.signals = []
    
    def load_data(self, symbols=None):
        """
        Load historical data from database.
        
        Args:
            symbols: List of symbols to backtest. If None, will try to load all available symbols
        
        Returns:
            Success flag
        """
        logger.info(f"Loading market data from {self.start_date} to {self.end_date}")
        
        try:
            # If no symbols provided, get all available symbols from the database
            if symbols is None:
                available_symbols = self.db.get_available_symbols()
                symbols = available_symbols
                logger.info(f"Found {len(symbols)} available symbols: {symbols}")
            
            all_market_data = {}
            
            # Load market data for each symbol
            for symbol in symbols:
                symbol_data = self.db.get_market_data(symbol, self.start_date, self.end_date)
                
                if symbol_data is not None and not symbol_data.empty:
                    all_market_data[symbol] = symbol_data
                    logger.info(f"Loaded {len(symbol_data)} rows for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
            
            if not all_market_data:
                logger.error("No market data found for any symbols")
                return False
            
            # Determine which symbol to use as the primary one for backtesting
            # Default to the first symbol if only one is provided
            if len(all_market_data) == 1:
                primary_symbol = next(iter(all_market_data.keys()))
            else:
                # Select the symbol with the most data points
                primary_symbol = max(all_market_data.keys(), key=lambda s: len(all_market_data[s]))
            
            logger.info(f"Using {primary_symbol} as primary symbol for backtesting")
            self.market_data = all_market_data[primary_symbol]
            
            # Store all market data for potential cross-asset features
            self.all_market_data = all_market_data
            
            # Get additional data sources - make these more generic
            try:
                # Try to get alternative data if available
                weather_data = self.db.get_weather_data_range(self.start_date, self.end_date) if hasattr(self.db, 'get_weather_data_range') else None
                fundamental_data = self.db.get_fundamental_data_range(self.start_date, self.end_date) if hasattr(self.db, 'get_fundamental_data_range') else None
                alternative_data = self.db.get_alternative_data_range(self.start_date, self.end_date) if hasattr(self.db, 'get_alternative_data_range') else None
                
                # More specific data sources for commodities if available
                eia_data = self.db.get_eia_data_range(self.start_date, self.end_date) if hasattr(self.db, 'get_eia_data_range') else None
                cot_data = self.db.get_cot_data_range(self.start_date, self.end_date) if hasattr(self.db, 'get_cot_data_range') else None
            except Exception as e:
                logger.warning(f"Error fetching alternative data: {str(e)}. Continuing with market data only.")
                weather_data = fundamental_data = alternative_data = eia_data = cot_data = None
            
            # Merge all data sources
            merged_data = self.market_data.copy()
            
            # Add supplementary data sources if they exist
            data_sources = {
                'weather': weather_data,
                'fundamental': fundamental_data,
                'alternative': alternative_data,
                'eia': eia_data,
                'cot': cot_data
            }
            
            for source_name, source_data in data_sources.items():
                if source_data is not None and not source_data.empty:
                    try:
                        merged_data = pd.merge(
                            merged_data, 
                            source_data, 
                            left_index=True, 
                            right_index=True, 
                            how='left'
                        )
                        logger.info(f"Added {source_name} data with {len(source_data.columns)} features")
                    except Exception as e:
                        logger.warning(f"Could not merge {source_name} data: {str(e)}")
            
            # Generate features
            self.feature_data = self.feature_generator.generate_features(merged_data)
            
            if self.feature_data is None or self.feature_data.empty:
                logger.error("Failed to generate features")
                return False
                
            logger.info(f"Generated features with {len(self.feature_data.columns)} columns")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def train_model(self, train_start_date, train_end_date, prediction_horizon=1):
        """
        Train ML model on historical data.
        
        Args:
            train_start_date: Start date for training
            train_end_date: End date for training
            prediction_horizon: Days ahead to predict
            
        Returns:
            Dictionary with model and metadata
        """
        logger.info(f"Training model from {train_start_date} to {train_end_date}")
        
        try:
            # Initialize model trainer
            trainer = ModelTrainer(
                db_connector=self.db,
                feature_generator=self.feature_generator,
                prediction_horizon=prediction_horizon
            )
            
            # Prepare data
            _, feature_data = trainer.prepare_data(train_start_date, train_end_date)
            
            # Train regression model
            regression_results = trainer.train_regression_model(feature_data)
            
            # Train classification model
            classification_results = trainer.train_classification_model(feature_data)
            
            # Return models
            return {
                'regression_model': regression_results['model'],
                'classification_model': classification_results['model'],
                'regression_rmse': regression_results['rmse'],
                'classification_accuracy': classification_results['accuracy'],
                'training_date': datetime.now(),
                'features': feature_data.columns.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None
    
    def generate_signals(self, model_data, test_data, confidence_threshold=0.6):
        """
        Generate trading signals using trained model.
        
        Args:
            model_data: Dictionary with model and metadata
            test_data: DataFrame with test data
            confidence_threshold: Threshold for actionable signals
            
        Returns:
            DataFrame with signals
        """
        logger.info(f"Generating signals for {len(test_data)} data points")
        
        try:
            # Extract models
            regression_model = model_data['regression_model']
            classification_model = model_data['classification_model']
            
            # Create feature set for prediction
            features_to_exclude = ['open', 'high', 'low', 'close', 'volume', 
                                 'price_change_1d', 'price_change_5d', 'direction_1d']
            
            X = test_data.drop(columns=features_to_exclude, errors='ignore')
            
            # Generate price change predictions
            price_change_pred = regression_model.predict(X)
            
            # Generate direction probabilities
            direction_prob = classification_model.predict_proba(X)[:, 1]
            
            # Generate direction predictions
            direction_pred = classification_model.predict(X)
            
            # Calculate confidence scores
            confidence = np.where(direction_pred == 1, direction_prob, 1 - direction_prob)
            
            # Determine if predictions are actionable
            is_actionable = confidence > confidence_threshold
            
            # Create signals DataFrame
            signals = pd.DataFrame({
                'date': test_data.index,
                'close': test_data['close'],
                'predicted_change': price_change_pred,
                'predicted_direction': direction_pred,
                'direction_probability': direction_prob,
                'confidence': confidence,
                'is_actionable': is_actionable
            })
            
            # Set date as index
            signals.set_index('date', inplace=True)
            
            # Generate buy/sell signals
            signals['signal'] = 0
            signals.loc[(signals['is_actionable']) & (signals['predicted_direction'] == 1), 'signal'] = 1  # Buy
            signals.loc[(signals['is_actionable']) & (signals['predicted_direction'] == 0), 'signal'] = -1  # Sell
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return None
    
    def execute_signal(self, date, row, capital, positions):
        """
        Execute a trading signal.
        
        Args:
            date: Date of signal
            row: Row of signal data
            capital: Current capital
            positions: Current positions
            
        Returns:
            Updated capital, positions, and trades
        """
        trades = []
        
        # Check if we have an actionable signal
        if not row['is_actionable']:
            return capital, positions, trades
            
        # Count active positions
        active_positions = [p for p in positions if p['active']]
        
        # Check if we've reached max positions
        if len(active_positions) >= self.max_positions:
            return capital, positions, trades
            
        # Check if we have a buy signal
        if row['signal'] == 1:
            # Calculate position size
            position_value = capital * self.position_size_pct
            position_size = position_value / row['close']
            
            # Add slippage
            entry_price = row['close'] * (1 + self.slippage_pct)
            
            # Add commission
            commission = position_size * self.commission_per_contract
            
            # Create position
            position = {
                'id': len(positions) + 1,
                'date': date,
                'entry_price': entry_price,
                'size': position_size,
                'direction': 'LONG',
                'stop_loss': entry_price * 0.98,  # 2% stop loss
                'take_profit': entry_price * 1.05,  # 5% take profit
                'active': True,
                'exit_date': None,
                'exit_price': 0,
                'pnl': 0,
                'exit_reason': ''
            }
            
            # Add position to list
            positions.append(position)
            
            # Update capital
            capital -= (position_size * entry_price + commission)
            
            # Add trade
            trade = {
                'date': date,
                'type': 'ENTRY',
                'direction': 'LONG',
                'price': entry_price,
                'size': position_size,
                'commission': commission,
                'position_id': position['id']
            }
            
            trades.append(trade)
            
        # Check if we have a sell signal
        elif row['signal'] == -1:
            # Calculate position size
            position_value = capital * self.position_size_pct
            position_size = position_value / row['close']
            
            # Add slippage
            entry_price = row['close'] * (1 - self.slippage_pct)
            
            # Add commission
            commission = position_size * self.commission_per_contract
            
            # Create position
            position = {
                'id': len(positions) + 1,
                'date': date,
                'entry_price': entry_price,
                'size': position_size,
                'direction': 'SHORT',
                'stop_loss': entry_price * 1.02,  # 2% stop loss
                'take_profit': entry_price * 0.95,  # 5% take profit
                'active': True,
                'exit_date': None,
                'exit_price': 0,
                'pnl': 0,
                'exit_reason': ''
            }
            
            # Add position to list
            positions.append(position)
            
            # Update capital
            capital -= commission  # For shorts, we don't remove capital until exit
            
            # Add trade
            trade = {
                'date': date,
                'type': 'ENTRY',
                'direction': 'SHORT',
                'price': entry_price,
                'size': position_size,
                'commission': commission,
                'position_id': position['id']
            }
            
            trades.append(trade)
        
        return capital, positions, trades
    
    def update_positions(self, date, row, capital, positions):
        """
        Update positions based on current price.
        
        Args:
            date: Current date
            row: Current price data
            capital: Current capital
            positions: Current positions
            
        Returns:
            Updated capital, positions, and trades
        """
        trades = []
        
        # Get current price
        current_price = row['close']
        
        # Loop through active positions
        for position in positions:
            if not position['active']:
                continue
                
            # Check if we've hit stop loss or take profit
            if position['direction'] == 'LONG':
                # Check stop loss
                if current_price <= position['stop_loss']:
                    # Exit position
                    position['active'] = False
                    position['exit_date'] = date
                    position['exit_price'] = current_price
                    position['pnl'] = (position['exit_price'] - position['entry_price']) * position['size']
                    position['exit_reason'] = 'STOP_LOSS'
                    
                    # Add commission
                    commission = position['size'] * self.commission_per_contract
                    
                    # Update capital
                    capital += (position['size'] * position['exit_price'] - commission)
                    
                    # Add trade
                    trade = {
                        'date': date,
                        'type': 'EXIT',
                        'direction': 'LONG',
                        'price': position['exit_price'],
                        'size': position['size'],
                        'commission': commission,
                        'position_id': position['id'],
                        'pnl': position['pnl'],
                        'exit_reason': position['exit_reason']
                    }
                    
                    trades.append(trade)
                
                # Check take profit
                elif current_price >= position['take_profit']:
                    # Exit position
                    position['active'] = False
                    position['exit_date'] = date
                    position['exit_price'] = current_price
                    position['pnl'] = (position['exit_price'] - position['entry_price']) * position['size']
                    position['exit_reason'] = 'TAKE_PROFIT'
                    
                    # Add commission
                    commission = position['size'] * self.commission_per_contract
                    
                    # Update capital
                    capital += (position['size'] * position['exit_price'] - commission)
                    
                    # Add trade
                    trade = {
                        'date': date,
                        'type': 'EXIT',
                        'direction': 'LONG',
                        'price': position['exit_price'],
                        'size': position['size'],
                        'commission': commission,
                        'position_id': position['id'],
                        'pnl': position['pnl'],
                        'exit_reason': position['exit_reason']
                    }
                    
                    trades.append(trade)
            
            elif position['direction'] == 'SHORT':
                # Check stop loss
                if current_price >= position['stop_loss']:
                    # Exit position
                    position['active'] = False
                    position['exit_date'] = date
                    position['exit_price'] = current_price
                    position['pnl'] = (position['entry_price'] - position['exit_price']) * position['size']
                    position['exit_reason'] = 'STOP_LOSS'
                    
                    # Add commission
                    commission = position['size'] * self.commission_per_contract
                    
                    # Update capital
                    capital += (position['size'] * position['entry_price'] - position['size'] * position['exit_price'] - commission)
                    
                    # Add trade
                    trade = {
                        'date': date,
                        'type': 'EXIT',
                        'direction': 'SHORT',
                        'price': position['exit_price'],
                        'size': position['size'],
                        'commission': commission,
                        'position_id': position['id'],
                        'pnl': position['pnl'],
                        'exit_reason': position['exit_reason']
                    }
                    
                    trades.append(trade)
                
                # Check take profit
                elif current_price <= position['take_profit']:
                    # Exit position
                    position['active'] = False
                    position['exit_date'] = date
                    position['exit_price'] = current_price
                    position['pnl'] = (position['entry_price'] - position['exit_price']) * position['size']
                    position['exit_reason'] = 'TAKE_PROFIT'
                    
                    # Add commission
                    commission = position['size'] * self.commission_per_contract
                    
                    # Update capital
                    capital += (position['size'] * position['entry_price'] - position['size'] * position['exit_price'] - commission)
                    
                    # Add trade
                    trade = {
                        'date': date,
                        'type': 'EXIT',
                        'direction': 'SHORT',
                        'price': position['exit_price'],
                        'size': position['size'],
                        'commission': commission,
                        'position_id': position['id'],
                        'pnl': position['pnl'],
                        'exit_reason': position['exit_reason']
                    }
                    
                    trades.append(trade)
        
        return capital, positions, trades
    
    def calculate_portfolio_value(self, date, capital, positions, current_price):
        """
        Calculate portfolio value.
        
        Args:
            date: Current date
            capital: Current capital
            positions: Current positions
            current_price: Current price
            
        Returns:
            Portfolio value
        """
        # Start with cash
        portfolio_value = capital
        
        # Add value of positions
        for position in positions:
            if not position['active']:
                continue
                
            if position['direction'] == 'LONG':
                # For long positions, add current value
                position_value = position['size'] * current_price
                portfolio_value += position_value
            elif position['direction'] == 'SHORT':
                # For short positions, add profit/loss
                position_value = position['size'] * (position['entry_price'] - current_price)
                portfolio_value += position_value
        
        return portfolio_value
    
    def run_walk_forward_backtest(self, symbols=None, train_days=365, test_days=30, confidence_threshold=0.6, retrain=True):
        """
        Run walk-forward backtest.
        
        Args:
            symbols: List of symbols to backtest. If None, will use all available symbols
            train_days: Number of days to use for training
            test_days: Number of days to use for testing
            confidence_threshold: Threshold for actionable signals
            retrain: Whether to retrain the model for each window
            
        Returns:
            Success flag
        """
        logger.info(f"Running walk-forward backtest with {train_days} days training, {test_days} days testing")
        
        try:
            # Load data
            if not self.load_data(symbols=symbols):
                return False
                
            # Get date range
            dates = self.feature_data.index.unique()
            dates = sorted(dates)
            
            # Initialize backtest variables
            capital = self.initial_capital
            positions = []
            all_trades = []
            portfolio_values = []
            all_signals = []
            
            # Loop through test windows
            start_idx = train_days
            while start_idx < len(dates):
                # Get train and test dates
                if start_idx + test_days <= len(dates):
                    test_end_idx = start_idx + test_days
                else:
                    test_end_idx = len(dates)
                    
                train_start_date = dates[start_idx - train_days]
                train_end_date = dates[start_idx - 1]
                test_start_date = dates[start_idx]
                test_end_date = dates[test_end_idx - 1]
                
                logger.info(f"Train: {train_start_date} to {train_end_date}, Test: {test_start_date} to {test_end_date}")
                
                # Get train and test data
                train_data = self.feature_data.loc[train_start_date:train_end_date]
                test_data = self.feature_data.loc[test_start_date:test_end_date]
                
                # Train model
                if retrain:
                    model_data = self.train_model(train_start_date, train_end_date)
                    
                    if model_data is None:
                        logger.error("Failed to train model")
                        return False
                        
                # Generate signals
                signals = self.generate_signals(model_data, test_data, confidence_threshold)
                
                if signals is None:
                    logger.error("Failed to generate signals")
                    return False
                    
                # Add to all signals
                all_signals.append(signals)
                
                # Loop through test days
                for date, row in signals.iterrows():
                    # Execute signal
                    capital, positions, new_trades = self.execute_signal(date, row, capital, positions)
                    
                    if new_trades:
                        all_trades.extend(new_trades)
                    
                    # Update positions
                    capital, positions, new_trades = self.update_positions(date, row, capital, positions)
                    
                    if new_trades:
                        all_trades.extend(new_trades)
                    
                    # Calculate portfolio value
                    portfolio_value = self.calculate_portfolio_value(date, capital, positions, row['close'])
                    
                    # Add to portfolio values
                    portfolio_values.append({
                        'date': date,
                        'cash': capital,
                        'portfolio_value': portfolio_value,
                        'close': row['close']
                    })
                
                # Move to next window
                start_idx += test_days
            
            # Convert lists to DataFrames
            self.signals = pd.concat(all_signals)
            self.trades = pd.DataFrame(all_trades)
            self.daily_portfolio = pd.DataFrame(portfolio_values)
            self.daily_portfolio.set_index('date', inplace=True)
            self.current_positions = positions
            
            # Calculate performance metrics
            self.calculate_performance_metrics()
            
            # Save results
            self.save_results()
            
            return True
            
        except Exception as e:
            logger.error(f"Error running walk-forward backtest: {str(e)}")
            return False
    
    def calculate_performance_metrics(self):
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        logger.info("Calculating performance metrics")
        
        try:
            # Calculate returns
            self.daily_portfolio['daily_return'] = self.daily_portfolio['portfolio_value'].pct_change()
            
            # Calculate cumulative returns
            self.daily_portfolio['cumulative_return'] = (1 + self.daily_portfolio['daily_return']).cumprod() - 1
            
            # Calculate drawdowns
            self.daily_portfolio['rolling_max'] = self.daily_portfolio['portfolio_value'].cummax()
            self.daily_portfolio['drawdown'] = (self.daily_portfolio['portfolio_value'] / self.daily_portfolio['rolling_max']) - 1
            
            # Calculate trade metrics
            if not self.trades.empty and 'pnl' in self.trades.columns:
                winning_trades = self.trades[self.trades['pnl'] > 0]
                losing_trades = self.trades[self.trades['pnl'] <= 0]
                
                total_trades = len(self.trades) // 2  # Divide by 2 because each round trip counts as 2 trades
                winning_trades_count = len(winning_trades) // 2
                win_rate = winning_trades_count / total_trades if total_trades > 0 else 0
                
                avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
                avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
                
                profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if losing_trades['pnl'].sum() != 0 else 0
            else:
                total_trades = 0
                winning_trades_count = 0
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
            
            # Calculate portfolio metrics
            total_return = self.daily_portfolio['portfolio_value'].iloc[-1] / self.initial_capital - 1
            annualized_return = (1 + total_return) ** (252 / len(self.daily_portfolio)) - 1
            max_drawdown = self.daily_portfolio['drawdown'].min()
            sharpe_ratio = np.sqrt(252) * self.daily_portfolio['daily_return'].mean() / self.daily_portfolio['daily_return'].std() if self.daily_portfolio['daily_return'].std() != 0 else 0
            sortino_ratio = np.sqrt(252) * self.daily_portfolio['daily_return'].mean() / self.daily_portfolio['daily_return'][self.daily_portfolio['daily_return'] < 0].std() if self.daily_portfolio['daily_return'][self.daily_portfolio['daily_return'] < 0].std() != 0 else 0
            
            # Store metrics
            self.metrics = {
                'start_date': self.daily_portfolio.index[0].strftime('%Y-%m-%d'),
                'end_date': self.daily_portfolio.index[-1].strftime('%Y-%m-%d'),
                'days': len(self.daily_portfolio),
                'initial_capital': self.initial_capital,
                'final_portfolio_value': self.daily_portfolio['portfolio_value'].iloc[-1],
                'total_return': total_return,
                'annualized_return': annualized_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'total_trades': total_trades,
                'winning_trades': winning_trades_count,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor
            }
            
            logger.info(f"Performance metrics: {json.dumps(self.metrics, indent=2)}")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return None
    
    def save_results(self):
        """
        Save backtest results.
        
        Returns:
            Success flag
        """
        logger.info(f"Saving backtest results to {self.output_dir}")
        
        try:
            # Save signals
            if not self.signals.empty:
                self.signals.to_csv(os.path.join(self.output_dir, 'signals.csv'))
            
            # Save trades
            if not self.trades.empty:
                self.trades.to_csv(os.path.join(self.output_dir, 'trades.csv'))
            
            # Save portfolio values
            if not self.daily_portfolio.empty:
                self.daily_portfolio.to_csv(os.path.join(self.output_dir, 'portfolio.csv'))
            
            # Save metrics
            if hasattr(self, 'metrics'):
                with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
                    json.dump(self.metrics, f, indent=2)
            
            # Generate equity curve plot
            if not self.daily_portfolio.empty:
                plt.figure(figsize=(12, 6))
                plt.plot(self.daily_portfolio.index, self.daily_portfolio['portfolio_value'], label='Portfolio Value')
                plt.plot(self.daily_portfolio.index, self.daily_portfolio['close'] * (self.initial_capital / self.daily_portfolio['close'].iloc[0]), label='Buy & Hold')
                plt.title('Equity Curve')
                plt.xlabel('Date')
                plt.ylabel('Portfolio Value')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'equity_curve.png'))
                
                # Generate drawdown plot
                plt.figure(figsize=(12, 6))
                plt.plot(self.daily_portfolio.index, self.daily_portfolio['drawdown'] * 100)
                plt.title('Drawdowns')
                plt.xlabel('Date')
                plt.ylabel('Drawdown (%)')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'drawdowns.png'))
            
            logger.info(f"Results saved to {self.output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False
    
    def visualize_results(self):
        """
        Generate visualizations of backtest results.
        
        Returns:
            Dictionary with paths to visualization files
        """
        logger.info("Generating visualizations")
        
        try:
            # Generate monthly returns heatmap
            if not self.daily_portfolio.empty:
                # Resample to monthly returns
                monthly_returns = self.daily_portfolio['daily_return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
                
                # Convert to DataFrame with year and month
                monthly_returns = pd.DataFrame({
                    'year': monthly_returns.index.year,
                    'month': monthly_returns.index.month,
                    'return': monthly_returns.values
                })
                
                # Pivot to create heatmap
                heatmap_data = monthly_returns.pivot(index='year', columns='month', values='return')
                
                # Plot heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(heatmap_data * 100, annot=True, fmt='.2f', cmap='RdYlGn', linewidths=0.5,
                           vmin=-10, vmax=10, center=0, cbar_kws={'label': 'Monthly Return (%)'})
                plt.title('Monthly Returns (%)')
                plt.xlabel('Month')
                plt.ylabel('Year')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'monthly_returns.png'))
            
            # Generate trade distribution plot
            if not self.trades.empty and 'pnl' in self.trades.columns:
                plt.figure(figsize=(12, 6))
                plt.hist(self.trades['pnl'], bins=50, alpha=0.75, color='blue')
                plt.axvline(0, color='red', linestyle='dashed', linewidth=1)
                plt.title('Trade P&L Distribution')
                plt.xlabel('P&L')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'trade_distribution.png'))
            
            # Generate trade P&L over time
            if not self.trades.empty and 'pnl' in self.trades.columns and 'date' in self.trades.columns:
                plt.figure(figsize=(12, 6))
                self.trades.set_index('date')['pnl'].cumsum().plot()
                plt.title('Cumulative P&L')
                plt.xlabel('Date')
                plt.ylabel('Cumulative P&L')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'cumulative_pnl.png'))
            
            logger.info("Visualizations generated")
            
            return {
                'equity_curve': os.path.join(self.output_dir, 'equity_curve.png'),
                'drawdowns': os.path.join(self.output_dir, 'drawdowns.png'),
                'monthly_returns': os.path.join(self.output_dir, 'monthly_returns.png'),
                'trade_distribution': os.path.join(self.output_dir, 'trade_distribution.png'),
                'cumulative_pnl': os.path.join(self.output_dir, 'cumulative_pnl.png')
            }
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            return None