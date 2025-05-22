# trading/execution_engine.py

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
from typing import Dict, List, Tuple, Union, Optional

logger = logging.getLogger(__name__)

class ExecutionEngine:
    """
    Engine for executing trades through Interactive Brokers.
    """
    
    def __init__(self, ibkr_connector, risk_per_trade=0.02):
        """
        Initialize the execution engine.
        
        Args:
            ibkr_connector: Interactive Brokers connector
            risk_per_trade: Risk per trade as a fraction of account equity
        """
        self.ibkr = ibkr_connector
        self.risk_per_trade = risk_per_trade
        self.position_id_counter = 1000
        
        # Mock positions for testing
        self._mock_positions = []
        
    def place_order(self, symbol, order_type, direction, quantity, stop_loss=None, take_profit=None):
        """
        Place an order through Interactive Brokers.
        
        Args:
            symbol: Symbol to trade
            order_type: Type of order (MARKET, LIMIT)
            direction: Direction of trade (BUY, SELL)
            quantity: Number of contracts
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Order ID
        """
        logger.info(f"Placing {direction} order for {quantity} {symbol} ({order_type})")
        
        try:
            # Get current price
            current_price = self.ibkr.get_current_price(symbol)
            
            # Generate a random order ID (mock)
            order_id = random.randint(10000, 99999)
            
            # Create a position entry 
            position = {
                'id': self.position_id_counter,
                'symbol': symbol,
                'direction': direction,
                'quantity': quantity,
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'stop_loss': stop_loss if stop_loss else 0,
                'take_profit': take_profit if take_profit else 0,
                'active': True,
                'order_id': order_id
            }
            
            # Add to mock positions
            self._mock_positions.append(position)
            self.position_id_counter += 1
            
            logger.info(f"Order placed successfully: {order_id}, {direction} {quantity} {symbol} @ {current_price}")
            
            return order_id
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def update_stop_loss(self, position_id, new_stop_loss):
        """
        Update stop loss for a position.
        
        Args:
            position_id: Position ID
            new_stop_loss: New stop loss price
            
        Returns:
            Success flag
        """
        logger.info(f"Updating stop loss for position {position_id} to {new_stop_loss}")
        
        try:
            # Find the position
            for position in self._mock_positions:
                if position['id'] == position_id:
                    position['stop_loss'] = new_stop_loss
                    logger.info(f"Updated stop loss for position {position_id} to {new_stop_loss}")
                    return True
            
            logger.warning(f"Position {position_id} not found")
            return False
        except Exception as e:
            logger.error(f"Error updating stop loss: {e}")
            return False
    
    def update_take_profit(self, position_id, new_take_profit):
        """
        Update take profit for a position.
        
        Args:
            position_id: Position ID
            new_take_profit: New take profit price
            
        Returns:
            Success flag
        """
        logger.info(f"Updating take profit for position {position_id} to {new_take_profit}")
        
        try:
            # Find the position
            for position in self._mock_positions:
                if position['id'] == position_id:
                    position['take_profit'] = new_take_profit
                    logger.info(f"Updated take profit for position {position_id} to {new_take_profit}")
                    return True
            
            logger.warning(f"Position {position_id} not found")
            return False
        except Exception as e:
            logger.error(f"Error updating take profit: {e}")
            return False
    
    def close_position(self, position_id, reason="Manual"):
        """
        Close a position.
        
        Args:
            position_id: Position ID
            reason: Reason for closing
            
        Returns:
            Success flag
        """
        logger.info(f"Closing position {position_id} ({reason})")
        
        try:
            # Find the position
            for position in self._mock_positions:
                if position['id'] == position_id and position['active']:
                    # Get current price
                    current_price = self.ibkr.get_current_price(position['symbol'])
                    
                    # Calculate P&L
                    if position['direction'] == 'BUY':
                        pnl = (current_price - position['entry_price']) * position['quantity']
                    else:  # SELL
                        pnl = (position['entry_price'] - current_price) * position['quantity']
                    
                    # Update position
                    position['active'] = False
                    position['exit_price'] = current_price
                    position['exit_time'] = datetime.now()
                    position['pnl'] = pnl
                    position['exit_reason'] = reason
                    
                    logger.info(f"Closed position {position_id} with P&L: {pnl:.2f}")
                    return True
            
            logger.warning(f"Position {position_id} not found or already closed")
            return False
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def get_positions(self):
        """
        Get list of current positions.
        
        Returns:
            List of positions
        """
        return self._mock_positions
    
    def get_active_positions(self):
        """
        Get list of active positions.
        
        Returns:
            List of active positions
        """
        return [p for p in self._mock_positions if p['active']]
    
    def calculate_position_size(self, symbol, entry_price, stop_price):
        """
        Calculate position size based on risk.
        
        Args:
            symbol: Symbol to trade
            entry_price: Entry price
            stop_price: Stop loss price
            
        Returns:
            Position size
        """
        # Get account equity
        account_equity = self.ibkr.get_account_value()
        
        # Calculate dollar risk
        risk_amount = account_equity * self.risk_per_trade
        
        # Calculate risk per contract
        risk_per_contract = abs(entry_price - stop_price)
        
        # Calculate position size
        if risk_per_contract > 0:
            position_size = risk_amount / risk_per_contract
        else:
            logger.warning("Zero risk per contract, using default position size")
            position_size = 1
        
        # Round to nearest integer
        position_size = round(position_size)
        
        # Ensure at least 1 contract
        position_size = max(1, position_size)
        
        logger.info(f"Calculated position size: {position_size} contracts for {symbol}")
        
        return position_size