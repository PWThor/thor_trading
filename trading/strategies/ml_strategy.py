# trading/strategies/ml_strategy.py
"""
Machine Learning Based Trading Strategy

This module implements a trading strategy that uses machine learning predictions
to generate trading signals.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MLTradingStrategy:
    """
    Machine learning based trading strategy that uses model predictions
    to generate trading signals.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ML trading strategy.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        
        # Strategy parameters
        self.entry_threshold = config.get('entry_threshold', 0.5)
        self.exit_threshold = config.get('exit_threshold', 0.2)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.05)
        self.take_profit_pct = config.get('take_profit_pct', 0.10)
        self.max_holding_period = config.get('max_holding_period', 20)
        self.model = None
        
        logger.info(f"Initialized ML trading strategy with entry threshold: {self.entry_threshold}")
    
    def set_model(self, model):
        """
        Set the ML model to use for predictions.
        
        Args:
            model: Trained ML model
        """
        self.model = model
        logger.info("ML model set for trading strategy")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on ML predictions.
        
        Args:
            data: DataFrame with features for prediction
            
        Returns:
            pd.DataFrame: DataFrame with added signals
        """
        if self.model is None:
            logger.error("No ML model available for generating signals")
            return data
        
        logger.info(f"Generating signals for {len(data)} data points")
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Generate predictions if not already present
        if 'prediction' not in df.columns:
            try:
                predictions = self.model.predict(df)
                df['prediction'] = predictions
                logger.info(f"Generated {len(predictions)} predictions")
            except Exception as e:
                logger.error(f"Error generating predictions: {e}")
                return df
        
        # Initialize position and signal columns
        df['signal'] = 0
        df['position'] = 0
        df['entry_price'] = np.nan
        df['exit_price'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        df['days_in_position'] = 0
        
        # Process each day
        active_position = 0
        entry_price = np.nan
        entry_date = None
        position_days = 0
        
        for i in range(1, len(df)):
            curr_idx = df.index[i]
            prev_idx = df.index[i-1]
            
            # Update days in position
            if active_position != 0:
                position_days += 1
                df.loc[curr_idx, 'days_in_position'] = position_days
            
                # Check for exit conditions
                
                # 1. Stop loss hit
                if active_position == 1 and df.loc[curr_idx, 'crack_spread'] <= df.loc[prev_idx, 'stop_loss']:
                    df.loc[curr_idx, 'position'] = 0
                    df.loc[curr_idx, 'exit_price'] = df.loc[prev_idx, 'stop_loss']
                    active_position = 0
                    position_days = 0
                
                elif active_position == -1 and df.loc[curr_idx, 'crack_spread'] >= df.loc[prev_idx, 'stop_loss']:
                    df.loc[curr_idx, 'position'] = 0
                    df.loc[curr_idx, 'exit_price'] = df.loc[prev_idx, 'stop_loss']
                    active_position = 0
                    position_days = 0
                
                # 2. Take profit hit
                elif active_position == 1 and df.loc[curr_idx, 'crack_spread'] >= df.loc[prev_idx, 'take_profit']:
                    df.loc[curr_idx, 'position'] = 0
                    df.loc[curr_idx, 'exit_price'] = df.loc[prev_idx, 'take_profit']
                    active_position = 0
                    position_days = 0
                
                elif active_position == -1 and df.loc[curr_idx, 'crack_spread'] <= df.loc[prev_idx, 'take_profit']:
                    df.loc[curr_idx, 'position'] = 0
                    df.loc[curr_idx, 'exit_price'] = df.loc[prev_idx, 'take_profit']
                    active_position = 0
                    position_days = 0
                
                # 3. Max holding period reached
                elif position_days >= self.max_holding_period:
                    df.loc[curr_idx, 'position'] = 0
                    df.loc[curr_idx, 'exit_price'] = df.loc[curr_idx, 'crack_spread']
                    active_position = 0
                    position_days = 0
                
                # 4. Prediction reversal
                elif (active_position == 1 and df.loc[curr_idx, 'prediction'] < -self.exit_threshold) or \
                     (active_position == -1 and df.loc[curr_idx, 'prediction'] > self.exit_threshold):
                    df.loc[curr_idx, 'position'] = 0
                    df.loc[curr_idx, 'exit_price'] = df.loc[curr_idx, 'crack_spread']
                    active_position = 0
                    position_days = 0
                
                # Maintain position
                else:
                    df.loc[curr_idx, 'position'] = active_position
                    df.loc[curr_idx, 'entry_price'] = entry_price
                    
                    # Update stop loss and take profit (trailing)
                    if self.config.get('trailing_stop', False):
                        if active_position == 1 and df.loc[curr_idx, 'crack_spread'] > entry_price:
                            # Update stop loss for longs if price has moved up
                            new_stop = df.loc[curr_idx, 'crack_spread'] * (1 - self.stop_loss_pct)
                            if new_stop > df.loc[prev_idx, 'stop_loss']:
                                df.loc[curr_idx, 'stop_loss'] = new_stop
                            else:
                                df.loc[curr_idx, 'stop_loss'] = df.loc[prev_idx, 'stop_loss']
                        elif active_position == -1 and df.loc[curr_idx, 'crack_spread'] < entry_price:
                            # Update stop loss for shorts if price has moved down
                            new_stop = df.loc[curr_idx, 'crack_spread'] * (1 + self.stop_loss_pct)
                            if new_stop < df.loc[prev_idx, 'stop_loss']:
                                df.loc[curr_idx, 'stop_loss'] = new_stop
                            else:
                                df.loc[curr_idx, 'stop_loss'] = df.loc[prev_idx, 'stop_loss']
                        else:
                            df.loc[curr_idx, 'stop_loss'] = df.loc[prev_idx, 'stop_loss']
                            
                        df.loc[curr_idx, 'take_profit'] = df.loc[prev_idx, 'take_profit']
                    else:
                        df.loc[curr_idx, 'stop_loss'] = df.loc[prev_idx, 'stop_loss']
                        df.loc[curr_idx, 'take_profit'] = df.loc[prev_idx, 'take_profit']
            
            # Check for entry conditions if not in a position
            elif active_position == 0:
                # Long signal
                if df.loc[curr_idx, 'prediction'] > self.entry_threshold:
                    df.loc[curr_idx, 'position'] = 1
                    df.loc[curr_idx, 'signal'] = 1
                    df.loc[curr_idx, 'entry_price'] = df.loc[curr_idx, 'crack_spread']
                    
                    # Set stop loss and take profit
                    df.loc[curr_idx, 'stop_loss'] = df.loc[curr_idx, 'crack_spread'] * (1 - self.stop_loss_pct)
                    df.loc[curr_idx, 'take_profit'] = df.loc[curr_idx, 'crack_spread'] * (1 + self.take_profit_pct)
                    
                    active_position = 1
                    entry_price = df.loc[curr_idx, 'crack_spread']
                    position_days = 0
                
                # Short signal
                elif df.loc[curr_idx, 'prediction'] < -self.entry_threshold:
                    df.loc[curr_idx, 'position'] = -1
                    df.loc[curr_idx, 'signal'] = -1
                    df.loc[curr_idx, 'entry_price'] = df.loc[curr_idx, 'crack_spread']
                    
                    # Set stop loss and take profit
                    df.loc[curr_idx, 'stop_loss'] = df.loc[curr_idx, 'crack_spread'] * (1 + self.stop_loss_pct)
                    df.loc[curr_idx, 'take_profit'] = df.loc[curr_idx, 'crack_spread'] * (1 - self.take_profit_pct)
                    
                    active_position = -1
                    entry_price = df.loc[curr_idx, 'crack_spread']
                    position_days = 0
        
        # Count trades
        entries = df[df['position'] != 0].iloc[0::].reset_index(drop=True)
        exits = df[df['exit_price'].notna()].reset_index(drop=True)
        
        logger.info(f"Generated {len(entries)} entry signals and {len(exits)} exit signals")
        
        return df

def create_ml_strategy(config):
    """
    Create a machine learning based trading strategy.
    
    Args:
        config: Strategy configuration
        
    Returns:
        MLTradingStrategy: Initialized strategy
    """
    return MLTradingStrategy(config)