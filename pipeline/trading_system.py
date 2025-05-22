import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import traceback
from typing import Dict, List, Tuple, Union, Optional, Callable
import schedule
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connectors.postgres_connector import PostgresConnector
from connectors.ibkr_connector import IBKRConnector
from models.prediction_engine import PredictionEngine
from features.feature_generator_wrapper import create_feature_generator
from features.feature_generator import FeatureGenerator
from trading.execution_engine import ExecutionEngine
from models.training.model_trainer import ModelTrainer


class TradingSystem:
    """
    Automated trading system that integrates data collection, model training, 
    prediction generation, and trade execution.
    """
    
    def __init__(
        self,
        db_connector: PostgresConnector,
        ibkr_connector: IBKRConnector,
        feature_generator: FeatureGenerator,
        execution_engine: ExecutionEngine,
        log_file: str = None,
        model_dir: str = None,
        retrain_days: int = 7,
        confidence_threshold: float = 0.7,
        risk_per_trade: float = 0.02,
        max_positions: int = 3
    ):
        """
        Initialize the TradingSystem.
        
        Args:
            db_connector: Database connector
            ibkr_connector: Interactive Brokers connector
            feature_generator: Feature generator
            execution_engine: Execution engine for placing trades
            log_file: Path to log file
            model_dir: Directory for trained models
            retrain_days: Number of days between model retraining
            confidence_threshold: Confidence threshold for actionable predictions
            risk_per_trade: Risk per trade as a fraction of account equity
            max_positions: Maximum number of concurrent positions
        """
        self.db = db_connector
        self.ibkr = ibkr_connector
        self.feature_generator = feature_generator
        self.execution_engine = execution_engine
        
        # Set up logging
        self.log_file = log_file if log_file else os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'logs',
            f'trading_system_{datetime.now().strftime("%Y%m%d")}.log'
        )
        
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.logger = logging.getLogger('TradingSystem')
        
        # Add console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        
        # Model directory
        if model_dir is None:
            self.model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'models',
                'saved_models'
            )
        else:
            self.model_dir = model_dir
            
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Trading parameters
        self.retrain_days = retrain_days
        self.confidence_threshold = confidence_threshold
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        
        # Initialize prediction engine
        self.prediction_engine = PredictionEngine(
            db_connector=self.db,
            feature_generator=self.feature_generator,
            model_dir=self.model_dir,
            confidence_threshold=self.confidence_threshold
        )
        
        # Initialize scheduler
        self.job_registry = []
        
        # Track last retraining date
        self.last_retrain_date = self._get_last_model_date()
        
        # System state
        self.is_running = False
        self.trading_enabled = False
    
    def _get_last_model_date(self) -> datetime:
        """
        Get the date of the latest model file.
        
        Returns:
            Datetime of the latest model or None if no models exist
        """
        model_files = []
        for root, _, files in os.walk(self.model_dir):
            for file in files:
                if file.endswith('.pkl') and ('price_prediction' in file or 'direction_prediction' in file):
                    model_files.append(os.path.join(root, file))
        
        if not model_files:
            return datetime.now() - timedelta(days=self.retrain_days * 2)  # Force retraining
        
        latest_model = max(model_files, key=os.path.getctime)
        last_modified = datetime.fromtimestamp(os.path.getctime(latest_model))
        
        return last_modified
    
    def train_models(self) -> bool:
        """
        Train new models if needed.
        
        Returns:
            Success flag
        """
        try:
            # Check if retraining is needed
            days_since_retrain = (datetime.now() - self.last_retrain_date).days
            
            if days_since_retrain < self.retrain_days:
                self.logger.info(f"Models are recent ({days_since_retrain} days old). Skipping retraining.")
                return True
            
            self.logger.info(f"Starting model retraining (models are {days_since_retrain} days old)")
            
            # Initialize model trainer
            trainer = ModelTrainer(
                db_connector=self.db,
                feature_generator=self.feature_generator,
                model_output_dir=self.model_dir
            )
            
            # Train models using 2 years of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=2*365)
            
            # Train and save models
            model_paths = trainer.train_and_save_models(start_date, end_date)
            
            self.logger.info(f"Models successfully trained and saved: {model_paths}")
            
            # Update last retrain date
            self.last_retrain_date = datetime.now()
            
            # Load the new models into prediction engine
            success, message = self.prediction_engine.load_latest_models()
            
            if not success:
                self.logger.error(f"Failed to load new models: {message}")
                return False
                
            self.logger.info("New models loaded into prediction engine")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def generate_trading_signals(self) -> Optional[Dict]:
        """
        Generate trading signals based on model predictions.
        
        Returns:
            Dictionary with trading signals or None if no actionable signals
        """
        try:
            self.logger.info("Generating trading signals")
            
            # Generate predictions
            prediction = self.prediction_engine.generate_predictions()
            
            # Save prediction to database
            self.prediction_engine.save_prediction(prediction)
            
            # Log prediction
            self.logger.info(f"Prediction for {prediction['target_date']}:")
            self.logger.info(f"Current price: {prediction['current_price']:.4f}")
            self.logger.info(f"Predicted price: {prediction['predicted_price']:.4f} (change: {prediction['predicted_change']:.4f})")
            self.logger.info(f"Direction: {'UP' if prediction['predicted_direction'] == 1 else 'DOWN'} with {prediction['confidence']:.2f} confidence")
            self.logger.info(f"Actionable: {prediction['is_actionable']}")
            
            # Check if prediction is actionable
            if not prediction['is_actionable']:
                self.logger.info("Prediction is not actionable, no trading signal generated")
                return None
            
            # Current positions
            current_positions = self.execution_engine.get_positions()
            
            # Count open positions
            open_position_count = len([p for p in current_positions if p['active']])
            
            # Check if we can take more positions
            if open_position_count >= self.max_positions:
                self.logger.info(f"Maximum positions reached ({open_position_count}/{self.max_positions}), no new trades")
                return None
            
            # Calculate position size based on risk
            account_value = self.ibkr.get_account_value()
            risk_amount = account_value * self.risk_per_trade
            
            # Set stop loss percentage (this could be more sophisticated)
            stop_loss_pct = 0.01  # 1%
            
            # Set take profit based on predicted move
            take_profit_pct = abs(prediction['predicted_change'] / prediction['current_price'])
            
            # Ensure reasonable take profit (for weird predictions)
            take_profit_pct = min(max(take_profit_pct, 0.005), 0.03)  # Between 0.5% and 3%
            
            # Calculate position size based on stop loss
            position_size = risk_amount / (prediction['current_price'] * stop_loss_pct)
            
            # Generate trading signal
            trading_signal = {
                'symbol': 'CL-HO-SPREAD',
                'direction': 'BUY' if prediction['predicted_direction'] == 1 else 'SELL',
                'entry_price': prediction['current_price'],
                'position_size': position_size,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'confidence': prediction['confidence'],
                'prediction_id': prediction.get('id', None),
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"Generated trading signal: {trading_signal}")
            return trading_signal
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def execute_trades(self, trading_signal: Dict) -> bool:
        """
        Execute trades based on trading signals.
        
        Args:
            trading_signal: Trading signal dictionary
            
        Returns:
            Success flag
        """
        if not self.trading_enabled:
            self.logger.info("Trading is disabled, skipping execution")
            return False
            
        try:
            self.logger.info(f"Executing trade for signal: {trading_signal}")
            
            # Calculate stop loss and take profit prices
            entry_price = trading_signal['entry_price']
            direction = trading_signal['direction']
            stop_loss_pct = trading_signal['stop_loss_pct']
            take_profit_pct = trading_signal['take_profit_pct']
            
            if direction == 'BUY':
                stop_loss_price = entry_price * (1 - stop_loss_pct)
                take_profit_price = entry_price * (1 + take_profit_pct)
            else:  # SELL
                stop_loss_price = entry_price * (1 + stop_loss_pct)
                take_profit_price = entry_price * (1 - take_profit_pct)
            
            # Execute the trade
            order_id = self.execution_engine.place_order(
                symbol=trading_signal['symbol'],
                order_type='MARKET',
                direction=direction,
                quantity=trading_signal['position_size'],
                stop_loss=stop_loss_price,
                take_profit=take_profit_price
            )
            
            if order_id:
                self.logger.info(f"Successfully placed order with ID: {order_id}")
                
                # Store the trading signal with the order ID
                trading_signal['order_id'] = order_id
                
                # Save to database if method exists
                if hasattr(self.db, 'save_trading_signal'):
                    self.db.save_trading_signal(trading_signal)
                
                return True
            else:
                self.logger.error("Failed to place order")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing trades: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def update_positions(self) -> bool:
        """
        Update position status and adjust stops for open positions.
        
        Returns:
            Success flag
        """
        try:
            self.logger.info("Updating positions")
            
            # Get current positions
            positions = self.execution_engine.get_positions()
            
            for position in positions:
                if position['active']:
                    # Get current price
                    current_price = self.ibkr.get_current_price(position['symbol'])
                    
                    # Calculate unrealized P&L
                    if position['direction'] == 'BUY':
                        unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
                    else:  # SELL
                        unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']
                    
                    # Log position status
                    self.logger.info(f"Position {position['id']}: {position['symbol']} {position['direction']} {position['quantity']} @ {position['entry_price']:.4f}")
                    self.logger.info(f"Current price: {current_price:.4f}, Unrealized P&L: {unrealized_pnl:.2f}")
                    
                    # Check if we should update trailing stop
                    # This is a simplified example - in reality, you'd want a more sophisticated trailing stop
                    
                    # For longs, if price has moved up by 1%, move stop loss to break even
                    if position['direction'] == 'BUY' and current_price > position['entry_price'] * 1.01:
                        new_stop = max(position['entry_price'], position['stop_loss'])
                        
                        if new_stop > position['stop_loss']:
                            self.logger.info(f"Updating stop loss for position {position['id']} from {position['stop_loss']:.4f} to {new_stop:.4f}")
                            self.execution_engine.update_stop_loss(position['id'], new_stop)
                    
                    # For shorts, if price has moved down by 1%, move stop loss to break even
                    elif position['direction'] == 'SELL' and current_price < position['entry_price'] * 0.99:
                        new_stop = min(position['entry_price'], position['stop_loss'] if position['stop_loss'] > 0 else float('inf'))
                        
                        if position['stop_loss'] == 0 or new_stop < position['stop_loss']:
                            self.logger.info(f"Updating stop loss for position {position['id']} from {position['stop_loss']:.4f} to {new_stop:.4f}")
                            self.execution_engine.update_stop_loss(position['id'], new_stop)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def run_trading_cycle(self) -> bool:
        """
        Run a complete trading cycle: check for retraining, generate signals, execute trades.
        
        Returns:
            Success flag
        """
        try:
            self.logger.info("Starting trading cycle")
            
            # Check if the market is open
            if not self.ibkr.is_market_open('CL-HO-SPREAD'):
                self.logger.info("Market is closed, skipping trading cycle")
                return True
            
            # Train models if needed
            self.train_models()
            
            # Generate trading signals
            trading_signal = self.generate_trading_signals()
            
            # Execute trades if there's an actionable signal
            if trading_signal:
                self.execute_trades(trading_signal)
            
            # Update positions
            self.update_positions()
            
            self.logger.info("Trading cycle completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def schedule_trading_cycle(self, time_str: str = "10:00") -> None:
        """
        Schedule daily trading cycle.
        
        Args:
            time_str: Time to run the job in 24-hour format (HH:MM)
        """
        job = schedule.every().day.at(time_str).do(self.run_trading_cycle)
        self.job_registry.append(job)
        self.logger.info(f"Scheduled daily trading cycle at {time_str}")
    
    def schedule_position_updates(self, interval_minutes: int = 30) -> None:
        """
        Schedule position updates.
        
        Args:
            interval_minutes: Interval in minutes
        """
        job = schedule.every(interval_minutes).minutes.do(self.update_positions)
        self.job_registry.append(job)
        self.logger.info(f"Scheduled position updates every {interval_minutes} minutes")
    
    def enable_trading(self) -> None:
        """
        Enable automated trading.
        """
        self.trading_enabled = True
        self.logger.info("Automated trading enabled")
    
    def disable_trading(self) -> None:
        """
        Disable automated trading.
        """
        self.trading_enabled = False
        self.logger.info("Automated trading disabled")
    
    def start(self) -> None:
        """
        Start the trading system.
        """
        if self.is_running:
            self.logger.warning("Trading system is already running")
            return
            
        self.is_running = True
        self.logger.info("Starting trading system")
        
        try:
            # Run initial trading cycle
            self.run_trading_cycle()
            
            # Start scheduler
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Trading system stopped by user")
            self.is_running = False
        except Exception as e:
            self.logger.error(f"Trading system error: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.is_running = False
    
    def stop(self) -> None:
        """
        Stop the trading system.
        """
        self.is_running = False
        self.logger.info("Stopping trading system")


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize connectors
        db = PostgresConnector()
        ibkr = IBKRConnector()
        
        # Initialize components
        feature_gen = FeatureGenerator()
        execution_engine = ExecutionEngine(ibkr)
        
        # Initialize trading system
        trading_system = TradingSystem(
            db_connector=db,
            ibkr_connector=ibkr,
            feature_generator=feature_gen,
            execution_engine=execution_engine,
            confidence_threshold=0.7,
            risk_per_trade=0.02,
            max_positions=3
        )
        
        # Schedule trading cycles
        trading_system.schedule_trading_cycle("10:00")  # Morning cycle
        trading_system.schedule_trading_cycle("14:00")  # Afternoon cycle
        
        # Schedule position updates
        trading_system.schedule_position_updates(30)  # Every 30 minutes
        
        # Enable trading (comment out this line if you only want to monitor without trading)
        # trading_system.enable_trading()
        
        # Start the system
        trading_system.start()
        
    except KeyboardInterrupt:
        print("Exiting by user request.")
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()