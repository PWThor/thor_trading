# connectors/ibkr_connector.py

import logging
from datetime import datetime, timedelta
import pandas as pd
import pytz
from ib_insync import IB, Future, Contract, MarketOrder, LimitOrder, StopOrder, BarData

logger = logging.getLogger(__name__)

class IBKRConnector:
    """Connector for Interactive Brokers API"""
    
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        """Initialize IBKR connection parameters"""
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self.connected = False
        self.timeout = 60  # Default timeout in seconds
    
    def connect(self):
        """Connect to IB Gateway or TWS"""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            logger.info(f"Connected to IBKR on {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from IB Gateway or TWS"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")
    
    def check_connection(self):
        """Check if connection is active and reconnect if needed"""
        if not self.connected or not self.ib.isConnected():
            logger.warning("Not connected to IBKR, attempting to reconnect...")
            return self.connect()
        return True
    
    def get_active_future_contract(self, symbol):
        """Get the active future contract for a symbol"""
        if not self.check_connection():
            return None
            
        try:
            # For crude oil (CL) and heating oil (HO)
            exchange = 'NYMEX'
            
            # Get current date
            today = datetime.now()
            
            # Calculate appropriate contract month (typically the next month or two)
            # For simplicity, if we're past the 15th, use next month's contract
            if today.day > 15:
                expiry_month = (today + timedelta(days=40)).strftime('%Y%m')
            else:
                expiry_month = (today + timedelta(days=20)).strftime('%Y%m')
            
            # Create Future contract
            contract = Future(symbol=symbol, exchange=exchange, lastTradeDateOrContractMonth=expiry_month)
            
            # Qualify the contract to get full details
            qualified_contracts = self.ib.qualifyContracts(contract)
            
            if qualified_contracts:
                logger.info(f"Found active contract for {symbol}: {qualified_contracts[0].localSymbol}")
                return qualified_contracts[0]
            else:
                logger.error(f"Failed to qualify contract for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting active future contract for {symbol}: {e}")
            return None
    
    def get_market_data(self, contract, duration='1 D', bar_size='1 hour', what_to_show='TRADES'):
        """Get historical market data for a contract"""
        if not self.check_connection() or contract is None:
            return None
            
        try:
            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract=contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,
                formatDate=1
            )
            
            if bars:
                # Convert to DataFrame
                df = pd.DataFrame([
                    {
                        'timestamp': bar.date,
                        'symbol': contract.symbol,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume
                    } for bar in bars
                ])
                
                logger.info(f"Retrieved {len(df)} bars for {contract.symbol}")
                return df
            else:
                logger.warning(f"No historical data received for {contract.symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    def place_market_order(self, contract, action, quantity):
        """Place a market order"""
        if not self.check_connection() or contract is None:
            return None
            
        try:
            # Create order object
            order = MarketOrder(action, quantity)
            
            # Submit order
            trade = self.ib.placeOrder(contract, order)
            
            logger.info(f"Placed market order: {action} {quantity} {contract.symbol}")
            
            # Return the trade object
            return trade
            
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None
    
    def place_limit_order(self, contract, action, quantity, limit_price):
        """Place a limit order"""
        if not self.check_connection() or contract is None:
            return None
            
        try:
            # Create order object
            order = LimitOrder(action, quantity, limit_price)
            
            # Submit order
            trade = self.ib.placeOrder(contract, order)
            
            logger.info(f"Placed limit order: {action} {quantity} {contract.symbol} @ {limit_price}")
            
            # Return the trade object
            return trade
            
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return None
    
    def place_stop_order(self, contract, action, quantity, stop_price):
        """Place a stop order"""
        if not self.check_connection() or contract is None:
            return None
            
        try:
            # Create order object
            order = StopOrder(action, quantity, stop_price)
            
            # Submit order
            trade = self.ib.placeOrder(contract, order)
            
            logger.info(f"Placed stop order: {action} {quantity} {contract.symbol} @ {stop_price}")
            
            # Return the trade object
            return trade
            
        except Exception as e:
            logger.error(f"Error placing stop order: {e}")
            return None
    
    def get_positions(self):
        """Get current positions"""
        if not self.check_connection():
            return []
            
        try:
            # Request positions
            positions = self.ib.positions()
            
            # Convert to a more usable format
            position_data = [
                {
                    'symbol': p.contract.symbol,
                    'exchange': p.contract.exchange,
                    'quantity': p.position,
                    'average_price': p.avgCost
                }
                for p in positions
            ]
            
            logger.info(f"Retrieved {len(position_data)} positions")
            return position_data
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_account_summary(self):
        """Get account summary information"""
        if not self.check_connection():
            return None
            
        try:
            # Request account summary
            summary = self.ib.accountSummary()
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'tag': s.tag,
                    'value': s.value,
                    'currency': s.currency,
                    'account': s.account
                }
                for s in summary
            ])
            
            logger.info("Retrieved account summary")
            return df
            
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return None
    
    def cancel_order(self, order_id):
        """Cancel an order by ID"""
        if not self.check_connection():
            return False
            
        try:
            # Find the order
            for trade in self.ib.trades():
                if trade.order.orderId == order_id:
                    # Cancel the order
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"Cancelled order {order_id}")
                    return True
            
            logger.warning(f"Order {order_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
            
    def is_market_open(self, symbol):
        """Check if the market for a symbol is open"""
        if not self.check_connection():
            return False
            
        try:
            # For real implementation, we should check actual market hours from IB
            # This is a simplification based on common futures market hours
            
            # Get current time in US Eastern timezone (where most futures trade)
            eastern = pytz.timezone('US/Eastern')
            now = datetime.now(eastern)
            weekday = now.weekday()
            
            # Most futures markets are open Sunday 6pm - Friday 5pm ET with breaks
            # For a basic implementation, check if it's a weekday during day hours
            is_open = (0 <= weekday <= 4) and (9 <= now.hour < 17)
            
            logger.info(f"Market for {symbol} is {'open' if is_open else 'closed'}")
            return is_open
            
        except Exception as e:
            logger.error(f"Error checking if market is open: {e}")
            return False
            
    def get_account_value(self):
        """Get the total account value"""
        if not self.check_connection():
            return 0.0
            
        try:
            # Request account summary
            summary = self.ib.accountSummary()
            
            # Find NetLiquidation value which represents total account value
            for item in summary:
                if item.tag == 'NetLiquidation' and item.currency == 'USD':
                    return float(item.value)
            
            logger.warning("Could not find NetLiquidation in account summary")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting account value: {e}")
            return 0.0