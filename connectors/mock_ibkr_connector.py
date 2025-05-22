# connectors/mock_ibkr_connector.py

import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import random

logger = logging.getLogger(__name__)

class IBKRConnector:
    """Mock connector for Interactive Brokers API (for testing)"""
    
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        """Initialize IBKR connection parameters"""
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connected = False
        self.timeout = 60  # Default timeout in seconds
        logger.info("Initialized MOCK IBKR connector (for testing)")
    
    def connect(self):
        """Mock connect to IB Gateway or TWS"""
        self.connected = True
        logger.info(f"[MOCK] Connected to IBKR on {self.host}:{self.port}")
        return True
        
    def disconnect(self):
        """Mock disconnect from IB Gateway or TWS"""
        self.connected = False
        logger.info("[MOCK] Disconnected from IBKR")
        return True
    
    def ensure_connected(self):
        """Ensure connection is established"""
        if not self.connected:
            self.connect()
        return self.connected
    
    def get_account_value(self):
        """Mock get account value"""
        return 100000.0  # $100,000 account value
    
    def get_historical_data(self, symbol, duration='1 day', bar_size='1 day', what_to_show='TRADES'):
        """Mock get historical data"""
        logger.info(f"[MOCK] Getting historical data for {symbol} with duration {duration}")
        
        # Create mock data
        end_date = datetime.now()
        
        # Determine number of days based on duration
        if duration == '1 day':
            days = 1
        elif duration == '1 week':
            days = 7
        elif duration == '1 month':
            days = 30
        else:
            days = 90  # Default to 3 months
            
        # Generate dates
        dates = [end_date - timedelta(days=i) for i in range(days)]
        dates.reverse()  # Sort in ascending order
        
        # Generate OHLC data
        if symbol == 'CL':  # Crude oil
            base_price = 75.0
        elif symbol == 'HO':  # Heating oil
            base_price = 2.5
        elif symbol == 'CL-HO-SPREAD':  # Crack spread
            base_price = 20.0
        else:
            base_price = 100.0
            
        # Generate random prices with a slight upward trend
        np.random.seed(42)  # For reproducibility
        
        # Generate mock data with realistic variations
        data = []
        current_price = base_price
        for date in dates:
            # Daily volatility as percentage of price
            daily_volatility = current_price * 0.015  # 1.5% daily volatility
            
            # Generate OHLC values
            open_price = current_price
            high_price = open_price + abs(np.random.normal(0, daily_volatility))
            low_price = open_price - abs(np.random.normal(0, daily_volatility))
            close_price = np.random.normal(open_price, daily_volatility)
            
            # Ensure high is highest and low is lowest
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Random volume between 50,000 and 150,000
            volume = int(np.random.uniform(50000, 150000))
            
            data.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
            
            # Update current price for next day
            current_price = close_price
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        return df
    
    def get_current_price(self, symbol):
        """Mock get current market price"""
        if symbol == 'CL':  # Crude oil
            return 75.25
        elif symbol == 'HO':  # Heating oil
            return 2.55
        elif symbol == 'CL-HO-SPREAD':  # Crack spread
            return 21.35
        else:
            return 100.0
    
    def is_market_open(self, symbol):
        """Mock check if market is open"""
        # Mock market hours: open Monday-Friday, 9:30am-4:00pm ET
        now = datetime.now()
        weekday = now.weekday()
        hour = now.hour
        
        # Mock market is open Monday (0) through Friday (4), 9am-5pm
        is_open = (0 <= weekday <= 4) and (9 <= hour < 17)
        
        return is_open
    
    def place_order(self, contract, order_type, quantity, action='BUY', price=None, stop_price=None):
        """Mock place an order"""
        logger.info(f"[MOCK] Placing {action} order for {quantity} {contract} at price {price}")
        
        # Generate a random order ID
        order_id = random.randint(10000, 99999)
        
        return order_id
        
    def cancel_order(self, order_id):
        """Mock cancel an order"""
        logger.info(f"[MOCK] Cancelling order {order_id}")
        return True
        
    def get_orders(self):
        """Mock get list of active orders"""
        return []
        
    def get_positions(self):
        """Mock get list of current positions"""
        return []