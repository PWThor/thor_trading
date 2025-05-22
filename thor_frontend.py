#!/usr/bin/env python
"""
Thor Trading System Frontend

A TradingView-like interface for visualizing price data and fundamental signals
from the Thor Trading System. This frontend connects to IB Gateway for
paper trading based on fundamental ML model signals.

Usage:
    python thor_frontend.py --symbols=CL,HO --port=7497
"""

import argparse
import sys
import os
import time
import json
import logging
from datetime import datetime, timedelta
import threading

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('thor_frontend')

# Attempt to import required packages
required_packages = [
    'pandas', 'numpy', 'plotly', 'dash', 'dash_bootstrap_components',
    'ibapi', 'matplotlib'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    logger.error(f"Missing required packages: {', '.join(missing_packages)}")
    logger.error("Please install required packages using: pip install " + " ".join(missing_packages))
    sys.exit(1)

# Now we can safely import the required packages
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order

# Add thor_trading to the path
thor_trading_path = '/mnt/e/Projects/thor_trading'
sys.path.append(thor_trading_path)

# Try to import Thor Trading System components (handle gracefully if not available)
try:
    from connectors.postgres_connector import PostgresConnector
    db_available = True
except ImportError:
    logger.warning("Database connector not available. Using mock data.")
    db_available = False

# Mock Database Connector for testing
if not db_available:
    class PostgresConnector:
        """Mock database connector for testing."""
        def __init__(self):
            pass
            
        def test_connection(self):
            return True
            
        def query(self, query, params=None):
            """Mock query that returns sample data."""
            # Create mock data for testing
            if 'market_data' in query and 'symbol' in query:
                symbol = query.split("symbol = '")[1].split("'")[0]
                return self._generate_mock_market_data(symbol)
            return []
        
        def query_one(self, query, params=None):
            """Mock query_one that returns a single result."""
            if 'COUNT' in query:
                return {'count': 100}
            return None
            
        def _generate_mock_market_data(self, symbol):
            """Generate mock market data for testing."""
            base_price = 75.0 if symbol == 'CL' else 2.5
            data = []
            for i in range(100):
                date = (datetime.now() - timedelta(days=100-i)).strftime('%Y-%m-%d')
                noise = np.random.normal(0, 1)
                trend = i * 0.05
                seasonal = 2 * np.sin(i/10)
                
                price = base_price + trend + seasonal + noise
                
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'open': price - 0.2,
                    'high': price + 0.5,
                    'low': price - 0.5,
                    'close': price,
                    'volume': int(1000 + 500 * np.random.random()),
                    'Commercial_Long': 350000 + 10000 * np.random.random(),
                    'Commercial_Short': 300000 + 10000 * np.random.random(),
                    'Commercial_Net': 50000 + 5000 * np.random.random(),
                    'NonCommercial_Long': 250000 + 10000 * np.random.random(),
                    'NonCommercial_Short': 200000 + 8000 * np.random.random(),
                    'NonCommercial_Net': 50000 + 5000 * np.random.random(),
                    'Inventory_Level': 400000 + 10000 * np.random.random(),
                    'OPEC_Production': 30.0 + np.random.random(),
                    'OPEC_Compliance': 0.9 + 0.1 * np.random.random(),
                    'Temperature': 65 + 15 * np.random.random()
                })
            return data


# IB API Connection Class
class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.portfolio = {}
        self.positions = {}
        self.account_summary = {}
        self.contract_details = {}
        self.market_data = {}
        self.next_order_id = None
        self.connected = False
        
    def nextValidId(self, orderId):
        self.next_order_id = orderId
        self.connected = True
        logger.info(f"Connected to IB Gateway. Next valid order ID: {orderId}")
        
    def error(self, reqId, errorCode, errorString):
        if errorCode == 2104 or errorCode == 2106:  # Market data farm connection is OK
            return
        logger.error(f"IB API Error {errorCode}: {errorString}")
        
    def position(self, account, contract, position, avgCost):
        symbol = contract.symbol
        self.positions[symbol] = {
            'position': position,
            'avgCost': avgCost,
            'contract': contract
        }
        logger.info(f"Position: {symbol}, Quantity: {position}, Avg Cost: {avgCost}")
        
    def accountSummary(self, reqId, account, tag, value, currency):
        self.account_summary[tag] = {'value': value, 'currency': currency}
        
    def tickPrice(self, reqId, tickType, price, attrib):
        if reqId in self.market_data:
            if tickType == 4:  # Last price
                self.market_data[reqId]['last'] = price
            elif tickType == 1:  # Bid
                self.market_data[reqId]['bid'] = price
            elif tickType == 2:  # Ask
                self.market_data[reqId]['ask'] = price
                
    def tickSize(self, reqId, tickType, size):
        if reqId in self.market_data:
            if tickType == 0:  # Bid size
                self.market_data[reqId]['bidSize'] = size
            elif tickType == 3:  # Ask size
                self.market_data[reqId]['askSize'] = size
            elif tickType == 5:  # Last size
                self.market_data[reqId]['lastSize'] = size
                
    def contractDetails(self, reqId, contractDetails):
        self.contract_details[reqId] = contractDetails
        
    def historicalData(self, reqId, bar):
        if reqId not in self.market_data:
            self.market_data[reqId] = {'bars': []}
        self.market_data[reqId]['bars'].append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })
        
    def historicalDataEnd(self, reqId, start, end):
        logger.info(f"Historical data received for reqId {reqId}")


# Thor Trading Frontend Class
class ThorTradingFrontend:
    """
    Frontend interface for Thor Trading System with TradingView-like charts
    and fundamental data visualization.
    """
    
    def __init__(self, symbols=None, ib_port=7497, use_paper=True):
        """
        Initialize the frontend with specified symbols.
        
        Args:
            symbols (list): List of symbols to display and trade
            ib_port (int): Port number for IB Gateway connection
            use_paper (bool): Whether to use IB paper trading account
        """
        self.symbols = symbols if symbols else ['CL', 'HO']
        self.ib_port = ib_port
        self.use_paper = use_paper
        self.db = PostgresConnector()
        self.ib_app = None
        self.app = None
        self.important_fundamentals = self._get_important_fundamentals()
        
        # Initialize dashboard
        self._initialize_dashboard()
        
    def _get_important_fundamentals(self):
        """
        Get the most important fundamental factors based on ML model results.
        Reads feature importance from saved model outputs if available.
        """
        try:
            # Check if we have feature importance files
            importance_data = {}
            for symbol in self.symbols:
                path = f"/home/pwthor/results/{symbol}_feature_importance.txt"
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        content = f.read()
                        
                    # Parse the content to extract feature importance
                    importance_data[symbol] = self._parse_feature_importance(content)
                else:
                    # Use default importance ranking if file isn't available
                    importance_data[symbol] = {
                        'Commercial_Net': 0.15,
                        'NonCommercial_Net': 0.12,
                        'Inventory_Level': 0.10,
                        'OPEC_Production': 0.08,
                        'OPEC_Compliance': 0.07,
                        'Comm_Price_Divergence': 0.06,
                        'Temperature': 0.05
                    }
            return importance_data
        except Exception as e:
            logger.error(f"Error loading feature importance: {e}")
            # Return a default set of important fundamentals
            return {symbol: {
                'Commercial_Net': 0.15,
                'NonCommercial_Net': 0.12,
                'Inventory_Level': 0.10,
                'OPEC_Production': 0.08,
                'OPEC_Compliance': 0.07
            } for symbol in self.symbols}
    
    def _parse_feature_importance(self, content):
        """Parse feature importance from the feature importance file."""
        importance = {}
        in_rf_section = False
        
        for line in content.split('\n'):
            if "Random Forest Feature Importance" in line:
                in_rf_section = True
                continue
            elif "Gradient Boosting Feature Importance" in line:
                in_rf_section = False
                continue
                
            if in_rf_section and ': ' in line:
                parts = line.split(': ')
                if len(parts) == 2:
                    feature, value = parts
                    try:
                        importance[feature] = float(value)
                    except ValueError:
                        pass
                        
        # Sort by importance and take top features
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
        return sorted_importance
        
    def _initialize_dashboard(self):
        """Initialize the Dash dashboard application."""
        try:
            # Initialize Dash app
            self.app = dash.Dash(
                __name__,
                external_stylesheets=[dbc.themes.DARKLY],
                suppress_callback_exceptions=True
            )
            
            # App layout
            self.app.layout = self._create_layout()
            
            # Set up callbacks
            self._setup_callbacks()
            
            logger.info("Dashboard initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing dashboard: {e}")
            raise
            
    def _create_layout(self):
        """Create the Dash app layout."""
        return dbc.Container(
            [
                # Header
                dbc.Row([
                    dbc.Col([
                        html.H1("THOR TRADING SYSTEM", 
                            className="display-4 text-center text-primary my-3")
                    ], width=12)
                ]),
                
                # Symbol selector and connection controls
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Trading Controls", className="bg-primary"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Symbol:"),
                                        dcc.Dropdown(
                                            id='symbol-dropdown',
                                            options=[{'label': s, 'value': s} for s in self.symbols],
                                            value=self.symbols[0],
                                            className="mb-2"
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("IB Connection:"),
                                        dbc.Button("Connect to IB", id="connect-ib-button", 
                                                  color="success", className="me-2"),
                                        dbc.Button("Disconnect", id="disconnect-ib-button", 
                                                  color="danger", disabled=True)
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("Auto Trading:"),
                                        dbc.Switch(
                                            id="auto-trading-switch",
                                            label="Enable Auto Trading",
                                            value=False,
                                            className="mt-2"
                                        )
                                    ], width=4)
                                ]),
                                html.Div(id="connection-status", 
                                         className="mt-2 text-center font-italic")
                            ])
                        ], className="mb-4")
                    ], width=12)
                ]),
                
                # Main content area - Chart and Fundamentals
                dbc.Row([
                    # Price Chart
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Price Chart", className="bg-primary"),
                            dbc.CardBody([
                                dcc.Graph(id="price-chart", style={"height": "60vh"})
                            ])
                        ])
                    ], width=8),
                    
                    # Fundamental Data and Trading Signals
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Key Fundamental Factors", className="bg-primary"),
                            dbc.CardBody([
                                html.Div(id="fundamental-indicators")
                            ])
                        ], className="mb-4"),
                        
                        dbc.Card([
                            dbc.CardHeader("ML Model Trading Signals", className="bg-primary"),
                            dbc.CardBody([
                                html.Div(id="trading-signals")
                            ])
                        ])
                    ], width=4)
                ]),
                
                # Portfolio and Orders
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Portfolio & Orders", className="bg-primary"),
                            dbc.CardBody([
                                dbc.Tabs([
                                    dbc.Tab([
                                        html.Div(id="positions-table", className="mt-3")
                                    ], label="Positions"),
                                    dbc.Tab([
                                        html.Div(id="orders-table", className="mt-3")
                                    ], label="Orders"),
                                    dbc.Tab([
                                        html.Div(id="account-summary", className="mt-3")
                                    ], label="Account")
                                ])
                            ])
                        ])
                    ], width=12)
                ], className="mt-4"),
                
                # Status bar
                dbc.Row([
                    dbc.Col([
                        html.Div(id="status-bar", className="text-center text-muted mt-4")
                    ], width=12)
                ]),
                
                # Hidden divs for storing data
                html.Div(id='market-data-store', style={'display': 'none'}),
                html.Div(id='fundamental-data-store', style={'display': 'none'}),
                html.Div(id='signals-data-store', style={'display': 'none'}),
                html.Div(id='positions-store', style={'display': 'none'}),
                html.Div(id='orders-store', style={'display': 'none'}),
                html.Div(id='connection-store', style={'display': 'none'}),
                
                # Update interval
                dcc.Interval(
                    id='interval-component',
                    interval=5000,  # milliseconds
                    n_intervals=0
                )
            ],
            fluid=True,
            className="bg-dark text-light"
        )
        
    def _setup_callbacks(self):
        """Set up the Dash callbacks."""
        # Connection button callbacks
        @self.app.callback(
            [Output('connection-status', 'children'),
             Output('connect-ib-button', 'disabled'),
             Output('disconnect-ib-button', 'disabled'),
             Output('connection-store', 'children')],
            [Input('connect-ib-button', 'n_clicks'),
             Input('disconnect-ib-button', 'n_clicks')],
            [State('connection-store', 'children')]
        )
        def handle_connection(connect_clicks, disconnect_clicks, connection_status):
            ctx = dash.callback_context
            if not ctx.triggered:
                # Default state
                return "Not connected to IB Gateway", False, True, json.dumps({"connected": False})
                
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'connect-ib-button' and connect_clicks:
                # Connect to IB Gateway
                try:
                    # In a real implementation, this would connect to IB
                    return (
                        "Connected to IB Gateway (Paper Trading)" if self.use_paper else "Connected to IB Gateway (Live Trading)",
                        True,
                        False,
                        json.dumps({"connected": True, "paper": self.use_paper})
                    )
                except Exception as e:
                    return f"Connection error: {str(e)}", False, True, json.dumps({"connected": False})
                    
            elif button_id == 'disconnect-ib-button' and disconnect_clicks:
                # Disconnect from IB Gateway
                return "Disconnected from IB Gateway", False, True, json.dumps({"connected": False})
                
            # Handle case where connection_status is provided
            if connection_status:
                status = json.loads(connection_status)
                if status.get("connected", False):
                    return (
                        "Connected to IB Gateway (Paper Trading)" if status.get("paper", True) else "Connected to IB Gateway (Live Trading)",
                        True,
                        False,
                        connection_status
                    )
                    
            # Default return
            return "Not connected to IB Gateway", False, True, json.dumps({"connected": False})
            
        # Symbol change callback
        @self.app.callback(
            [Output('market-data-store', 'children'),
             Output('fundamental-data-store', 'children')],
            [Input('symbol-dropdown', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_data(symbol, n_intervals):
            # Load data for the selected symbol
            market_data = self._load_market_data(symbol)
            fundamental_data = self._load_fundamental_data(symbol)
            
            return json.dumps(market_data), json.dumps(fundamental_data)
            
        # Chart update callback
        @self.app.callback(
            Output('price-chart', 'figure'),
            [Input('market-data-store', 'children'),
             Input('signals-data-store', 'children'),
             Input('symbol-dropdown', 'value')]
        )
        def update_chart(market_data_json, signals_data_json, symbol):
            market_data = json.loads(market_data_json) if market_data_json else None
            signals_data = json.loads(signals_data_json) if signals_data_json else None
            
            return self._create_price_chart(market_data, signals_data, symbol)
            
        # Fundamental indicators callback
        @self.app.callback(
            Output('fundamental-indicators', 'children'),
            [Input('fundamental-data-store', 'children'),
             Input('symbol-dropdown', 'value')]
        )
        def update_fundamental_indicators(fundamental_data_json, symbol):
            fundamental_data = json.loads(fundamental_data_json) if fundamental_data_json else None
            
            return self._create_fundamental_indicators(fundamental_data, symbol)
            
        # Trading signals callback
        @self.app.callback(
            [Output('trading-signals', 'children'),
             Output('signals-data-store', 'children')],
            [Input('fundamental-data-store', 'children'),
             Input('market-data-store', 'children'),
             Input('symbol-dropdown', 'value')]
        )
        def update_trading_signals(fundamental_data_json, market_data_json, symbol):
            fundamental_data = json.loads(fundamental_data_json) if fundamental_data_json else None
            market_data = json.loads(market_data_json) if market_data_json else None
            
            signals, signals_data = self._generate_trading_signals(market_data, fundamental_data, symbol)
            
            return signals, json.dumps(signals_data)
            
        # Positions table callback
        @self.app.callback(
            [Output('positions-table', 'children'),
             Output('positions-store', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('connection-store', 'children')]
        )
        def update_positions(n_intervals, connection_status):
            if connection_status:
                status = json.loads(connection_status)
                if status.get("connected", False):
                    # In a real implementation, this would get positions from IB API
                    positions = self._get_mock_positions()
                    
                    # Create positions table
                    return self._create_positions_table(positions), json.dumps(positions)
                    
            # Default empty positions
            return "Connect to IB Gateway to view positions", json.dumps([])
            
        # Orders table callback
        @self.app.callback(
            [Output('orders-table', 'children'),
             Output('orders-store', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('connection-store', 'children')]
        )
        def update_orders(n_intervals, connection_status):
            if connection_status:
                status = json.loads(connection_status)
                if status.get("connected", False):
                    # In a real implementation, this would get orders from IB API
                    orders = self._get_mock_orders()
                    
                    # Create orders table
                    return self._create_orders_table(orders), json.dumps(orders)
                    
            # Default empty orders
            return "Connect to IB Gateway to view orders", json.dumps([])
            
        # Account summary callback
        @self.app.callback(
            Output('account-summary', 'children'),
            [Input('interval-component', 'n_intervals'),
             Input('connection-store', 'children')]
        )
        def update_account_summary(n_intervals, connection_status):
            if connection_status:
                status = json.loads(connection_status)
                if status.get("connected", False):
                    # In a real implementation, this would get account info from IB API
                    account_info = self._get_mock_account_info()
                    
                    # Create account summary
                    return self._create_account_summary(account_info)
                    
            # Default empty account info
            return "Connect to IB Gateway to view account information"
            
        # Status bar callback
        @self.app.callback(
            Output('status-bar', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_status_bar(n_intervals):
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return f"Thor Trading System | Last updated: {now} | Data source: {'Live' if db_available else 'Mock'}"
    
    def _load_market_data(self, symbol):
        """Load market data for the specified symbol."""
        query = f"""
        SELECT timestamp, symbol, open, high, low, close, volume
        FROM market_data
        WHERE symbol = '{symbol}'
        ORDER BY timestamp
        """
        
        try:
            data = self.db.query(query)
            
            # Convert to list of dicts with proper date format
            formatted_data = []
            for row in data:
                row_dict = dict(row)
                if isinstance(row_dict['timestamp'], datetime):
                    row_dict['timestamp'] = row_dict['timestamp'].strftime('%Y-%m-%d')
                formatted_data.append(row_dict)
                
            return formatted_data
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return []
    
    def _load_fundamental_data(self, symbol):
        """Load fundamental data for the specified symbol."""
        # First try separate tables
        fundamental_data = {}
        
        # Try to load from separate tables (cot_data, etc.)
        # If that fails, try to load from market_data table
        
        query = f"""
        SELECT 
            timestamp, 
            Commercial_Long, Commercial_Short, Commercial_Net,
            NonCommercial_Long, NonCommercial_Short, NonCommercial_Net,
            Inventory_Level, OPEC_Production, OPEC_Quota, OPEC_Compliance,
            Temperature, Inventory_Change
        FROM market_data
        WHERE symbol = '{symbol}'
        ORDER BY timestamp
        """
        
        try:
            data = self.db.query(query)
            
            # Convert to list of dicts with proper date format
            formatted_data = []
            for row in data:
                row_dict = dict(row)
                if isinstance(row_dict['timestamp'], datetime):
                    row_dict['timestamp'] = row_dict['timestamp'].strftime('%Y-%m-%d')
                formatted_data.append(row_dict)
                
            return formatted_data
        except Exception as e:
            logger.error(f"Error loading fundamental data: {e}")
            return []
    
    def _create_price_chart(self, market_data, signals_data, symbol):
        """Create a TradingView-like price chart."""
        if not market_data:
            # Create empty chart
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=(f"{symbol} Price", "Volume")
            )
            return fig
            
        # Convert to Pandas DataFrame
        df = pd.DataFrame(market_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create subplots: price chart and volume
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{symbol} Price", "Volume")
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'], 
                high=df['high'],
                low=df['low'], 
                close=df['close'],
                name="Price",
                increasing_line_color='#26a69a', 
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # Add volume chart
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['volume'],
                name="Volume",
                marker_color='rgba(100, 100, 255, 0.5)'
            ),
            row=2, col=1
        )
        
        # Add moving averages
        for ma_period in [20, 50, 200]:
            df[f'MA{ma_period}'] = df['close'].rolling(window=ma_period).mean()
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df[f'MA{ma_period}'],
                    line=dict(width=1),
                    name=f"{ma_period}-day MA"
                ),
                row=1, col=1
            )
        
        # Add trading signals if available
        if signals_data:
            signals_df = pd.DataFrame(signals_data)
            if not signals_df.empty:
                # Make sure timestamps are datetime
                signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
                
                # Add buy signals
                buy_signals = signals_df[signals_df['signal'] == 'buy']
                if not buy_signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_signals['timestamp'],
                            y=buy_signals['price'],
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up',
                                size=10,
                                color='green'
                            ),
                            name="Buy Signal"
                        ),
                        row=1, col=1
                    )
                
                # Add sell signals
                sell_signals = signals_df[signals_df['signal'] == 'sell']
                if not sell_signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_signals['timestamp'],
                            y=sell_signals['price'],
                            mode='markers',
                            marker=dict(
                                symbol='triangle-down',
                                size=10,
                                color='red'
                            ),
                            name="Sell Signal"
                        ),
                        row=1, col=1
                    )
        
        # Update layout
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=600,
            margin=dict(l=10, r=10, t=40, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20,20,20,1)',
            font=dict(color='white'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(80,80,80,0.2)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(80,80,80,0.2)'
            ),
            xaxis2=dict(
                showgrid=True,
                gridcolor='rgba(80,80,80,0.2)'
            ),
            yaxis2=dict(
                showgrid=True,
                gridcolor='rgba(80,80,80,0.2)'
            )
        )
        
        return fig
    
    def _create_fundamental_indicators(self, fundamental_data, symbol):
        """Create indicators for the most important fundamental factors."""
        if not fundamental_data:
            return html.Div("No fundamental data available")
            
        # Get the top important factors for this symbol
        important_factors = self.important_fundamentals.get(symbol, {})
        if not important_factors:
            important_factors = {
                'Commercial_Net': 0.15,
                'NonCommercial_Net': 0.12,
                'Inventory_Level': 0.10,
                'OPEC_Production': 0.08,
                'OPEC_Compliance': 0.07
            }
        
        # Convert to DataFrame
        df = pd.DataFrame(fundamental_data)
        if df.empty:
            return html.Div("No fundamental data available")
            
        # Create indicators
        indicators = []
        
        # Sort by importance
        sorted_factors = sorted(important_factors.items(), key=lambda x: x[1], reverse=True)
        
        for factor, importance in sorted_factors[:5]:  # Show top 5
            if factor in df.columns:
                # Get latest value
                latest_value = df[factor].iloc[-1]
                
                # Calculate Z-score (how many std devs from mean)
                mean_value = df[factor].mean()
                std_value = df[factor].std()
                z_score = (latest_value - mean_value) / std_value if std_value > 0 else 0
                
                # Determine color based on z-score (bullish/bearish)
                if factor in ['Commercial_Net', 'OPEC_Compliance']:
                    # For these factors, higher values are bullish
                    color = self._get_color_from_zscore(z_score)
                elif factor in ['Inventory_Level', 'NonCommercial_Net']:
                    # For these factors, lower values are bullish (inverse)
                    color = self._get_color_from_zscore(-z_score)
                else:
                    # Neutral
                    color = self._get_color_from_zscore(z_score)
                
                # Format value based on factor
                if 'OPEC_Compliance' in factor:
                    value_formatted = f"{latest_value:.1%}"
                elif any(x in factor for x in ['Inventory', 'Production', 'Quota']):
                    value_formatted = f"{latest_value:,.0f}"
                elif any(x in factor for x in ['Commercial', 'NonCommercial']):
                    value_formatted = f"{latest_value:,.0f}"
                elif 'Temperature' in factor:
                    value_formatted = f"{latest_value:.1f}°F"
                else:
                    value_formatted = f"{latest_value:.2f}"
                
                # Create indicator card
                factor_name = factor.replace('_', ' ')
                indicators.append(
                    dbc.Card([
                        dbc.CardBody([
                            html.H6(factor_name, className="card-subtitle"),
                            html.H4(value_formatted, className="card-title", style={"color": color}),
                            html.P(f"Z-Score: {z_score:.2f}", className="card-text small text-muted"),
                            dbc.Progress(value=50 + z_score * 10, color=color.replace('text-', ''))
                        ])
                    ], className="mb-2")
                )
                
        return html.Div(indicators)
                
    def _get_color_from_zscore(self, z_score):
        """Get color based on Z-score value."""
        if z_score > 2.0:
            return "text-success"  # Very bullish
        elif z_score > 0.5:
            return "text-info"  # Bullish
        elif z_score < -2.0:
            return "text-danger"  # Very bearish
        elif z_score < -0.5:
            return "text-warning"  # Bearish
        else:
            return "text-secondary"  # Neutral
    
    def _generate_trading_signals(self, market_data, fundamental_data, symbol):
        """Generate trading signals based on ML model results."""
        try:
            # Check if we have model outputs for this symbol
            model_path = f"/home/pwthor/results/{symbol}_signals.csv"
            
            if os.path.exists(model_path):
                # Real signals from model
                signals_df = pd.read_csv(model_path)
                signals_html = self._format_trading_signals_from_model(signals_df, symbol)
                
                # Prepare signals for chart
                chart_signals = []
                for _, row in signals_df.iterrows():
                    if 'Date' in row and 'Direction' in row and 'Close' in row:
                        signal_type = 'buy' if row['Direction'] > 0 else 'sell'
                        chart_signals.append({
                            'timestamp': row['Date'],
                            'signal': signal_type,
                            'price': row['Close'],
                            'confidence': row.get('Confidence', 0.5)
                        })
                
                return signals_html, chart_signals
            else:
                # Generate mock signals
                if not market_data or not fundamental_data:
                    return html.Div("Insufficient data for signals"), []
                    
                # Convert to DataFrames
                market_df = pd.DataFrame(market_data)
                fund_df = pd.DataFrame(fundamental_data)
                
                # Merge data
                merged_df = pd.merge(
                    market_df, fund_df, how='left', on='timestamp'
                )
                
                if merged_df.empty:
                    return html.Div("Insufficient data for signals"), []
                
                # Get latest data point
                latest = merged_df.iloc[-1]
                
                # Generate a trading signal based on fundamentals
                signal_strength = 0
                
                # Commercial_Net - if positive and high, bullish
                if 'Commercial_Net' in latest:
                    comm_net = latest['Commercial_Net']
                    comm_mean = merged_df['Commercial_Net'].mean()
                    comm_std = merged_df['Commercial_Net'].std()
                    comm_zscore = (comm_net - comm_mean) / comm_std if comm_std > 0 else 0
                    signal_strength += comm_zscore * 0.4  # Weight: 40%
                
                # Inventory_Level - if low, bullish
                if 'Inventory_Level' in latest:
                    inv_level = latest['Inventory_Level']
                    inv_mean = merged_df['Inventory_Level'].mean()
                    inv_std = merged_df['Inventory_Level'].std()
                    inv_zscore = (inv_level - inv_mean) / inv_std if inv_std > 0 else 0
                    signal_strength -= inv_zscore * 0.3  # Weight: 30%, inverted
                
                # OPEC_Compliance - if high, bullish
                if 'OPEC_Compliance' in latest:
                    opec_comp = latest['OPEC_Compliance']
                    opec_mean = merged_df['OPEC_Compliance'].mean()
                    opec_std = merged_df['OPEC_Compliance'].std()
                    opec_zscore = (opec_comp - opec_mean) / opec_std if opec_std > 0 else 0
                    signal_strength += opec_zscore * 0.3  # Weight: 30%
                
                # Determine direction and confidence
                direction = 'BUY' if signal_strength > 0 else 'SELL'
                confidence = min(abs(signal_strength) * 0.25, 0.95)  # Cap at 95%
                
                # Format the signal
                signal_color = "success" if direction == 'BUY' else "danger"
                
                signals_html = html.Div([
                    html.H3(direction, className=f"text-{signal_color} text-center"),
                    html.P(f"Confidence: {confidence:.1%}", className="text-center"),
                    html.Hr(),
                    html.P("Signal basis:", className="mb-2"),
                    html.Ul([
                        html.Li(f"Commercial Net Position: {'+' if comm_zscore > 0 else ''}{comm_zscore:.2f} σ"),
                        html.Li(f"Inventory Level: {'+' if -inv_zscore > 0 else ''}{-inv_zscore:.2f} σ"),
                        html.Li(f"OPEC Compliance: {'+' if opec_zscore > 0 else ''}{opec_zscore:.2f} σ")
                    ]),
                    html.Hr(),
                    html.P(f"Recommended Position Size: {min(abs(signal_strength) * 0.2, 1.0):.1%}", 
                           className="font-weight-bold text-center")
                ])
                
                # Mock signals for chart
                mock_signals = []
                
                # Add current signal
                current_date = datetime.now().strftime('%Y-%m-%d')
                current_price = latest['close']
                
                mock_signals.append({
                    'timestamp': current_date,
                    'signal': 'buy' if direction == 'BUY' else 'sell',
                    'price': current_price,
                    'confidence': confidence
                })
                
                # Add some past signals
                for i in range(1, 6):
                    if len(merged_df) > i * 10:
                        past_row = merged_df.iloc[-i*10]
                        past_signal = 'buy' if i % 2 == 0 else 'sell'  # Alternate signals
                        mock_signals.append({
                            'timestamp': past_row['timestamp'],
                            'signal': past_signal,
                            'price': past_row['close'],
                            'confidence': 0.7 + 0.05 * i  # Random confidence level
                        })
                
                return signals_html, mock_signals
                
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return html.Div(f"Error generating signals: {str(e)}"), []
    
    def _format_trading_signals_from_model(self, signals_df, symbol):
        """Format trading signals from the ML model output."""
        try:
            # Get the most recent signal
            latest_signal = signals_df.iloc[-1]
            
            direction = 'BUY' if latest_signal['Direction'] > 0 else 'SELL'
            confidence = latest_signal.get('Confidence', 0.5)
            position_size = latest_signal.get('Position', 0.5)
            
            # Format the signal
            signal_color = "success" if direction == 'BUY' else "danger"
            
            return html.Div([
                html.H3(direction, className=f"text-{signal_color} text-center"),
                html.P(f"Confidence: {confidence:.1%}", className="text-center"),
                html.Hr(),
                html.P("Model Signal Details:", className="mb-2"),
                html.Ul([
                    html.Li(f"Returns Forecast: {latest_signal.get('Returns_Pred', 0):.2%}"),
                    html.Li(f"Position Size: {position_size:.1%}"),
                    html.Li(f"Stop Loss: ${latest_signal.get('Stop_Loss', 0):.2f}"),
                    html.Li(f"Take Profit: ${latest_signal.get('Take_Profit', 0):.2f}")
                ]),
                html.Hr(),
                html.Div([
                    html.P("Trade Controls:", className="mb-2"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Execute Buy", color="success", className="w-100")
                        ], width=6),
                        dbc.Col([
                            dbc.Button("Execute Sell", color="danger", className="w-100")
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Close Position", color="warning", className="w-100 mt-2")
                        ], width=12)
                    ])
                ])
            ])
            
        except Exception as e:
            logger.error(f"Error formatting trading signals: {e}")
            return html.Div(f"Error formatting signals: {str(e)}")
    
    def _get_mock_positions(self):
        """Get mock positions for testing."""
        return [
            {
                'symbol': 'CL',
                'position': 5,
                'avgCost': 75.32,
                'marketPrice': 76.45,
                'pnl': 565.00,
                'pnlPercent': 1.5
            },
            {
                'symbol': 'HO',
                'position': -10,
                'avgCost': 2.43,
                'marketPrice': 2.37,
                'pnl': 600.00,
                'pnlPercent': 2.47
            }
        ]
    
    def _create_positions_table(self, positions):
        """Create a table displaying current positions."""
        header = html.Thead(html.Tr([
            html.Th("Symbol"),
            html.Th("Position"),
            html.Th("Avg Cost"),
            html.Th("Market Price"),
            html.Th("P&L"),
            html.Th("P&L %")
        ]))
        
        rows = []
        for pos in positions:
            # Determine if position is profitable
            is_profitable = pos['pnl'] > 0
            pnl_color = "text-success" if is_profitable else "text-danger"
            
            rows.append(html.Tr([
                html.Td(pos['symbol']),
                html.Td(html.Span(
                    pos['position'],
                    className="text-success" if pos['position'] > 0 else "text-danger"
                )),
                html.Td(f"${pos['avgCost']:.2f}"),
                html.Td(f"${pos['marketPrice']:.2f}"),
                html.Td(html.Span(f"${pos['pnl']:.2f}", className=pnl_color)),
                html.Td(html.Span(f"{pos['pnlPercent']:.2f}%", className=pnl_color))
            ]))
        
        table = dbc.Table([header, html.Tbody(rows)], striped=True, bordered=True, hover=True)
        
        if not rows:
            return html.Div("No open positions")
            
        return table
    
    def _get_mock_orders(self):
        """Get mock orders for testing."""
        return [
            {
                'orderId': 1001,
                'symbol': 'CL',
                'action': 'BUY',
                'quantity': 5,
                'orderType': 'LMT',
                'limitPrice': 75.50,
                'status': 'Filled',
                'time': '2023-05-20 09:32:15'
            },
            {
                'orderId': 1002,
                'symbol': 'HO',
                'action': 'SELL',
                'quantity': 10,
                'orderType': 'STP',
                'limitPrice': 2.35,
                'status': 'Working',
                'time': '2023-05-20 10:45:30'
            }
        ]
    
    def _create_orders_table(self, orders):
        """Create a table displaying current orders."""
        header = html.Thead(html.Tr([
            html.Th("Order ID"),
            html.Th("Symbol"),
            html.Th("Action"),
            html.Th("Quantity"),
            html.Th("Type"),
            html.Th("Price"),
            html.Th("Status"),
            html.Th("Time")
        ]))
        
        rows = []
        for order in orders:
            # Color based on action and status
            action_color = "text-success" if order['action'] == 'BUY' else "text-danger"
            status_color = "text-success" if order['status'] == 'Filled' else "text-warning"
            
            rows.append(html.Tr([
                html.Td(order['orderId']),
                html.Td(order['symbol']),
                html.Td(html.Span(order['action'], className=action_color)),
                html.Td(order['quantity']),
                html.Td(order['orderType']),
                html.Td(f"${order['limitPrice']:.2f}"),
                html.Td(html.Span(order['status'], className=status_color)),
                html.Td(order['time'])
            ]))
        
        table = dbc.Table([header, html.Tbody(rows)], striped=True, bordered=True, hover=True)
        
        if not rows:
            return html.Div("No active orders")
            
        return table
    
    def _get_mock_account_info(self):
        """Get mock account information for testing."""
        return {
            'NetLiquidation': 250000.00,
            'AvailableFunds': 180000.00,
            'ExcessLiquidity': 175000.00,
            'BuyingPower': 500000.00,
            'Leverage': 1.2,
            'Currency': 'USD',
            'DailyPnL': 650.00,
            'UnrealizedPnL': 1200.00
        }
    
    def _create_account_summary(self, account_info):
        """Create an account summary panel."""
        daily_pnl_class = "text-success" if account_info.get('DailyPnL', 0) > 0 else "text-danger"
        unrealized_pnl_class = "text-success" if account_info.get('UnrealizedPnL', 0) > 0 else "text-danger"
        
        card_content = [
            dbc.Row([
                dbc.Col([
                    html.P("Net Liquidation Value:", className="mb-0 font-weight-bold"),
                    html.H3(f"${account_info.get('NetLiquidation', 0):,.2f}")
                ], width=6),
                dbc.Col([
                    html.P("Available Funds:", className="mb-0 font-weight-bold"),
                    html.H3(f"${account_info.get('AvailableFunds', 0):,.2f}")
                ], width=6)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.P("Buying Power:", className="mb-0"),
                    html.H5(f"${account_info.get('BuyingPower', 0):,.2f}")
                ], width=4),
                dbc.Col([
                    html.P("Daily P&L:", className="mb-0"),
                    html.H5(f"${account_info.get('DailyPnL', 0):,.2f}", className=daily_pnl_class)
                ], width=4),
                dbc.Col([
                    html.P("Unrealized P&L:", className="mb-0"),
                    html.H5(f"${account_info.get('UnrealizedPnL', 0):,.2f}", className=unrealized_pnl_class)
                ], width=4)
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.P("Leverage:", className="mb-0"),
                    html.H5(f"{account_info.get('Leverage', 0):.2f}x")
                ], width=4),
                dbc.Col([
                    html.P("Excess Liquidity:", className="mb-0"),
                    html.H5(f"${account_info.get('ExcessLiquidity', 0):,.2f}")
                ], width=4),
                dbc.Col([
                    html.P("Currency:", className="mb-0"),
                    html.H5(f"{account_info.get('Currency', 'USD')}")
                ], width=4)
            ])
        ]
        
        return html.Div(card_content)
    
    def start_app(self, debug=False, port=8050):
        """Start the Dash application."""
        logger.info(f"Starting Thor Trading Frontend on port {port}")
        self.app.run_server(debug=debug, port=port)
        
    def connect_to_ib(self):
        """Connect to Interactive Brokers API."""
        if self.ib_app:
            if self.ib_app.connected:
                logger.info("Already connected to IB")
                return True
        
        try:
            # Create IB App
            self.ib_app = IBapi()
            
            # Connect to IB Gateway
            self.ib_app.connect('127.0.0.1', self.ib_port, 0)
            
            # Start API thread
            api_thread = threading.Thread(target=self.ib_app.run, daemon=True)
            api_thread.start()
            
            # Wait for connection
            timeout = 5  # seconds
            start_time = time.time()
            while not self.ib_app.connected and time.time() - start_time < timeout:
                time.sleep(0.1)
                
            if not self.ib_app.connected:
                logger.error("Failed to connect to IB Gateway")
                return False
                
            logger.info("Connected to IB Gateway")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to IB: {e}")
            return False
    
    def disconnect_from_ib(self):
        """Disconnect from Interactive Brokers API."""
        if self.ib_app and self.ib_app.connected:
            self.ib_app.disconnect()
            logger.info("Disconnected from IB Gateway")
    
    def place_trade(self, symbol, action, quantity):
        """Place a trade via IB API."""
        if not self.ib_app or not self.ib_app.connected:
            logger.error("Not connected to IB Gateway")
            return False
            
        try:
            # Create contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "FUT"  # Futures
            contract.exchange = "NYMEX"
            contract.currency = "USD"
            
            # Add appropriate details based on symbol
            if symbol == "CL":
                contract.localSymbol = "CLM3"  # Example: Crude Oil June 2023
            elif symbol == "HO":
                contract.localSymbol = "HOM3"  # Example: Heating Oil June 2023
                
            # Create order
            order = Order()
            order.action = action
            order.totalQuantity = quantity
            order.orderType = "MKT"
            
            # Place order
            if self.ib_app.next_order_id:
                order_id = self.ib_app.next_order_id
                self.ib_app.next_order_id += 1
                self.ib_app.placeOrder(order_id, contract, order)
                logger.info(f"Placed {action} order for {quantity} {symbol} with order ID {order_id}")
                return True
            else:
                logger.error("No valid order ID available")
                return False
                
        except Exception as e:
            logger.error(f"Error placing trade: {e}")
            return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Thor Trading System Frontend')
    parser.add_argument('--symbols', type=str, default='CL,HO',
                      help='Comma-separated list of symbols to display')
    parser.add_argument('--port', type=int, default=8050,
                      help='Port number for the web application')
    parser.add_argument('--ib-port', type=int, default=7497,
                      help='Port number for IB Gateway (7497 for paper, 7496 for live)')
    parser.add_argument('--live', action='store_true',
                      help='Connect to live trading (default is paper trading)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Parse symbols
    symbols = args.symbols.split(',')
    
    # Create and start frontend
    frontend = ThorTradingFrontend(
        symbols=symbols,
        ib_port=args.ib_port,
        use_paper=not args.live
    )
    
    # Start the app
    frontend.start_app(debug=True, port=args.port)