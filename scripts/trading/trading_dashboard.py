import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout, 
                            QHBoxLayout, QWidget, QPushButton, QLabel, QComboBox,
                            QTableView, QDateEdit, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QTimer
import psycopg2
from psycopg2.extras import RealDictCursor
from ib_insync import IB, Contract, Order, util

class TradingSystemApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize database connection
        self.conn = psycopg2.connect(
            dbname="trading_db",
            user="postgre",
            password="Makingmoney25!",
            host="localhost"
        )
        
        # Initialize IB connection (not connected yet)
        self.ib = IB()
        
        self.setWindowTitle("Energy Trading System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Create tabs
        self.backtesting_tab = QWidget()
        self.realtime_tab = QWidget()
        self.analysis_tab = QWidget()
        self.settings_tab = QWidget()
        
        self.tabs.addTab(self.backtesting_tab, "Backtesting")
        self.tabs.addTab(self.realtime_tab, "Real-time Trading")
        self.tabs.addTab(self.analysis_tab, "Market Analysis")
        self.tabs.addTab(self.settings_tab, "Settings")
        
        # Setup tab contents
        self.setup_backtesting_tab()
        self.setup_realtime_tab()
        self.setup_analysis_tab()
        self.setup_settings_tab()
        
        # Timer for real-time updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_realtime_data)
        self.update_timer.start(5000)  # Update every 5 seconds
    
    def setup_backtesting_tab(self):
        layout = QVBoxLayout()
        
        # Date range selection
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("Start Date:"))
        self.start_date = QDateEdit()
        self.start_date.setDate(pd.Timestamp('1992-01-01').to_pydatetime().date())
        date_layout.addWidget(self.start_date)
        
        date_layout.addWidget(QLabel("End Date:"))
        self.end_date = QDateEdit()
        self.end_date.setDate(pd.Timestamp('2025-05-01').to_pydatetime().date())
        date_layout.addWidget(self.end_date)
        
        date_layout.addWidget(QLabel("Strategy:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Crack Spread Mean Reversion", 
                                     "Seasonal + Weather", 
                                     "COT Positioning", 
                                     "Combined Strategy"])
        date_layout.addWidget(self.strategy_combo)
        
        self.run_backtest_btn = QPushButton("Run Backtest")
        self.run_backtest_btn.clicked.connect(self.run_backtest)
        date_layout.addWidget(self.run_backtest_btn)
        
        layout.addLayout(date_layout)
        
        # Results section
        self.results_layout = QHBoxLayout()
        
        # Charts area
        self.backtest_figure = plt.figure(figsize=(10, 8))
        self.backtest_canvas = FigureCanvasQTAgg(self.backtest_figure)
        self.results_layout.addWidget(self.backtest_canvas, 7)
        
        # Stats area
        stats_layout = QVBoxLayout()
        self.stats_table = QTableView()
        stats_layout.addWidget(QLabel("Performance Metrics"))
        stats_layout.addWidget(self.stats_table)
        self.results_layout.addLayout(stats_layout, 3)
        
        layout.addLayout(self.results_layout)
        
        self.backtesting_tab.setLayout(layout)
    
    def setup_realtime_tab(self):
        layout = QVBoxLayout()
        
        # Connection controls
        conn_layout = QHBoxLayout()
        conn_layout.addWidget(QLabel("IB Gateway:"))
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_to_ib)
        conn_layout.addWidget(self.connect_btn)
        
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_from_ib)
        self.disconnect_btn.setEnabled(False)
        conn_layout.addWidget(self.disconnect_btn)
        
        self.connection_status = QLabel("Disconnected")
        conn_layout.addWidget(self.connection_status)
        conn_layout.addStretch()
        
        layout.addLayout(conn_layout)
        
        # Trading interface
        trading_layout = QHBoxLayout()
        
        # Market data panel
        market_layout = QVBoxLayout()
        market_layout.addWidget(QLabel("Market Data"))
        
        # Current prices
        price_layout = QHBoxLayout()
        price_layout.addWidget(QLabel("Crude Oil:"))
        self.crude_price = QLabel("$0.00")
        price_layout.addWidget(self.crude_price)
        
        price_layout.addWidget(QLabel("Heating Oil:"))
        self.heating_price = QLabel("$0.00")
        price_layout.addWidget(self.heating_price)
        
        price_layout.addWidget(QLabel("Crack Spread:"))
        self.crack_spread = QLabel("$0.00")
        price_layout.addWidget(self.crack_spread)
        market_layout.addLayout(price_layout)
        
        # Current signals
        signal_layout = QHBoxLayout()
        signal_layout.addWidget(QLabel("Signal:"))
        self.current_signal = QLabel("NEUTRAL")
        signal_layout.addWidget(self.current_signal)
        
        signal_layout.addWidget(QLabel("Strength:"))
        self.signal_strength = QLabel("0%")
        signal_layout.addWidget(self.signal_strength)
        market_layout.addLayout(signal_layout)
        
        # Real-time chart
        self.realtime_figure = plt.figure(figsize=(8, 4))
        self.realtime_canvas = FigureCanvasQTAgg(self.realtime_figure)
        market_layout.addWidget(self.realtime_canvas)
        
        trading_layout.addLayout(market_layout, 7)
        
        # Order panel
        order_layout = QVBoxLayout()
        order_layout.addWidget(QLabel("Trading"))
        
        # Contract selection
        order_layout.addWidget(QLabel("Contract:"))
        self.contract_combo = QComboBox()
        self.contract_combo.addItems(["CL (Crude Oil)", "HO (Heating Oil)", "Crack Spread"])
        order_layout.addWidget(self.contract_combo)
        
        # Quantity
        order_layout.addWidget(QLabel("Quantity:"))
        self.quantity_spin = QSpinBox()
        self.quantity_spin.setRange(1, 100)
        self.quantity_spin.setValue(1)
        order_layout.addWidget(self.quantity_spin)
        
        # Order buttons
        btn_layout = QHBoxLayout()
        self.buy_btn = QPushButton("BUY")
        self.buy_btn.clicked.connect(self.place_buy_order)
        btn_layout.addWidget(self.buy_btn)
        
        self.sell_btn = QPushButton("SELL")
        self.sell_btn.clicked.connect(self.place_sell_order)
        btn_layout.addWidget(self.sell_btn)
        order_layout.addLayout(btn_layout)
        
        # Open positions
        order_layout.addWidget(QLabel("Open Positions:"))
        self.positions_table = QTableView()
        order_layout.addWidget(self.positions_table)
        
        trading_layout.addLayout(order_layout, 3)
        
        layout.addLayout(trading_layout)
        
        self.realtime_tab.setLayout(layout)
    
    def setup_analysis_tab(self):
        layout = QVBoxLayout()
        
        # Analysis selectors
        selector_layout = QHBoxLayout()
        
        selector_layout.addWidget(QLabel("Analysis Type:"))
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems(["Seasonal Patterns", 
                                     "COT Positioning", 
                                     "Weather Impact", 
                                     "EIA Inventory Analysis",
                                     "OPEC Production Impact"])
        self.analysis_combo.currentIndexChanged.connect(self.update_analysis)
        selector_layout.addWidget(self.analysis_combo)
        
        selector_layout.addWidget(QLabel("Timeframe:"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["1 Year", "5 Years", "10 Years", "All Data"])
        self.timeframe_combo.currentIndexChanged.connect(self.update_analysis)
        selector_layout.addWidget(self.timeframe_combo)
        
        layout.addLayout(selector_layout)
        
        # Analysis visualization
        self.analysis_figure = plt.figure(figsize=(10, 6))
        self.analysis_canvas = FigureCanvasQTAgg(self.analysis_figure)
        layout.addWidget(self.analysis_canvas)
        
        self.analysis_tab.setLayout(layout)
    
    def setup_settings_tab(self):
        layout = QVBoxLayout()
        
        # Database settings
        db_layout = QVBoxLayout()
        db_layout.addWidget(QLabel("Database Configuration"))
        # Add database configuration fields
        
        # IB Gateway settings
        ib_layout = QVBoxLayout()
        ib_layout.addWidget(QLabel("Interactive Brokers Configuration"))
        # Add IB configuration fields
        
        # Strategy parameters
        strategy_layout = QVBoxLayout()
        strategy_layout.addWidget(QLabel("Strategy Parameters"))
        
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Z-Score Threshold:"))
        self.zscore_spin = QDoubleSpinBox()
        self.zscore_spin.setRange(0.5, 3.0)
        self.zscore_spin.setSingleStep(0.1)
        self.zscore_spin.setValue(2.0)
        param_layout.addWidget(self.zscore_spin)
        
        param_layout.addWidget(QLabel("Position Size:"))
        self.position_spin = QDoubleSpinBox()
        self.position_spin.setRange(0.1, 1.0)
        self.position_spin.setSingleStep(0.1)
        self.position_spin.setValue(0.5)
        param_layout.addWidget(self.position_spin)
        
        strategy_layout.addLayout(param_layout)
        
        layout.addLayout(db_layout)
        layout.addLayout(ib_layout)
        layout.addLayout(strategy_layout)
        
        self.settings_tab.setLayout(layout)
    
    def run_backtest(self):
        """Run backtest with selected parameters"""
        strategy = self.strategy_combo.currentText()
        start_date = self.start_date.date().toString("yyyy-MM-dd")
        end_date = self.end_date.date().toString("yyyy-MM-dd")
        
        # Fetch data from PostgreSQL
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
            SELECT 
              p.date, 
              p.crude_oil_price, 
              p.heating_oil_price,
              (p.heating_oil_price * 42) - p.crude_oil_price AS crack_spread
            FROM price_data p
            WHERE p.date BETWEEN %s AND %s
            ORDER BY p.date
            """
            cursor.execute(query, (start_date, end_date))
            data = cursor.fetchall()
        
        df = pd.DataFrame(data)
        
        # Calculate returns based on strategy
        if strategy == "Crack Spread Mean Reversion":
            # Implement mean reversion logic
            df['sma_90'] = df['crack_spread'].rolling(90).mean()
            df['stdev_90'] = df['crack_spread'].rolling(90).std()
            df['zscore'] = (df['crack_spread'] - df['sma_90']) / df['stdev_90']
            
            # Generate signals
            df['signal'] = 0
            df.loc[df['zscore'] < -2, 'signal'] = 1  # Buy signal
            df.loc[df['zscore'] > 2, 'signal'] = -1  # Sell signal
            
            # Calculate strategy returns
            df['strategy_returns'] = df['signal'].shift(1) * df['crack_spread'].pct_change()
        
        # Plot results
        self.backtest_figure.clear()
        ax1 = self.backtest_figure.add_subplot(211)
        ax1.plot(df['date'], df['crack_spread'], label='Crack Spread')
        if 'sma_90' in df.columns:
            ax1.plot(df['date'], df['sma_90'], label='90-day SMA')
        ax1.set_title('Crack Spread')
        ax1.legend()
        
        ax2 = self.backtest_figure.add_subplot(212)
        if 'strategy_returns' in df.columns:
            cumulative_returns = (1 + df['strategy_returns'].fillna(0)).cumprod()
            ax2.plot(df['date'], cumulative_returns, label='Strategy')
            # Add benchmark
            benchmark_returns = df['crack_spread'].pct_change().fillna(0)
            cumulative_benchmark = (1 + benchmark_returns).cumprod()
            ax2.plot(df['date'], cumulative_benchmark, label='Buy & Hold')
            ax2.set_title('Cumulative Returns')
            ax2.legend()
        
        self.backtest_canvas.draw()
        
        # Calculate and display stats
        if 'strategy_returns' in df.columns:
            # Performance metrics calculations would go here
            annual_return = df['strategy_returns'].mean() * 252
            sharpe_ratio = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252)
            max_drawdown = (df['strategy_returns'].fillna(0).cumsum().cummax() - 
                           df['strategy_returns'].fillna(0).cumsum()).max()
            
            print(f"Annual Return: {annual_return:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            
            # Update stats table
            # This would use a model/view pattern with QTableView
    
    def connect_to_ib(self):
        """Connect to Interactive Brokers Gateway"""
        try:
            # Connect to IB Gateway
            self.ib.connect('127.0.0.1', 7496, clientId=1)
            
            # Update UI
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.connection_status.setText("Connected")
            
            # Subscribe to market data
            self.subscribe_to_market_data()
        except Exception as e:
            self.connection_status.setText(f"Error: {str(e)}")
    
    def disconnect_from_ib(self):
        """Disconnect from Interactive Brokers Gateway"""
        if self.ib.isConnected():
            self.ib.disconnect()
            
        # Update UI
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.connection_status.setText("Disconnected")
    
    def subscribe_to_market_data(self):
        """Subscribe to required market data from IB"""
        if not self.ib.isConnected():
            return
            
        # Create contracts
        cl_contract = Contract()
        cl_contract.symbol = "CL"
        cl_contract.secType = "FUT"
        cl_contract.exchange = "NYMEX"
        cl_contract.currency = "USD"
        cl_contract.lastTradeDateOrContractMonth = "202506"  # Front month
        
        ho_contract = Contract()
        ho_contract.symbol = "HO"
        ho_contract.secType = "FUT"
        ho_contract.exchange = "NYMEX"
        ho_contract.currency = "USD"
        ho_contract.lastTradeDateOrContractMonth = "202506"  # Front month
        
        # Request market data
        self.cl_data = self.ib.reqMktData(cl_contract)
        self.ho_data = self.ib.reqMktData(ho_contract)
    
    def update_realtime_data(self):
        """Update real-time market data display"""
        if not self.ib.isConnected():
            return
            
        # Process any pending IB messages
        self.ib.sleep(0)
        
        # Update price labels
        if hasattr(self, 'cl_data') and self.cl_data.last:
            self.crude_price.setText(f"${self.cl_data.last:.2f}")
            
        if hasattr(self, 'ho_data') and self.ho_data.last:
            self.heating_price.setText(f"${self.ho_data.last:.2f}")
            
        # Calculate crack spread
        if (hasattr(self, 'cl_data') and self.cl_data.last and 
            hasattr(self, 'ho_data') and self.ho_data.last):
            crack = (self.ho_data.last * 42) - self.cl_data.last
            self.crack_spread.setText(f"${crack:.2f}")
            
            # Fetch historical average for signal generation
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = """
                SELECT 
                  AVG((heating_oil_price * 42) - crude_oil_price) AS avg_crack,
                  STDDEV((heating_oil_price * 42) - crude_oil_price) AS std_crack
                FROM price_data
                WHERE date >= CURRENT_DATE - INTERVAL '90 days'
                """
                cursor.execute(query)
                result = cursor.fetchone()
                
                if result and result['avg_crack'] and result['std_crack']:
                    zscore = (crack - result['avg_crack']) / result['std_crack']
                    
                    # Update signal based on Z-score
                    if zscore < -2:
                        self.current_signal.setText("BUY")
                        strength = min(100, int(abs(zscore) * 25))
                        self.signal_strength.setText(f"{strength}%")
                    elif zscore > 2:
                        self.current_signal.setText("SELL")
                        strength = min(100, int(abs(zscore) * 25))
                        self.signal_strength.setText(f"{strength}%")
                    else:
                        self.current_signal.setText("NEUTRAL")
                        self.signal_strength.setText("0%")
        
        # Update real-time chart
        # This would plot recent price movements
    
    def place_buy_order(self):
        """Place a buy order via IB"""
        if not self.ib.isConnected():
            return
            
        # Get selected contract
        contract_type = self.contract_combo.currentText()
        quantity = self.quantity_spin.value()
        
        # Create contract
        if "CL" in contract_type:
            contract = Contract()
            contract.symbol = "CL"
            contract.secType = "FUT"
            contract.exchange = "NYMEX"
            contract.currency = "USD"
            contract.lastTradeDateOrContractMonth = "202506"
        elif "HO" in contract_type:
            contract = Contract()
            contract.symbol = "HO"
            contract.secType = "FUT"
            contract.exchange = "NYMEX"
            contract.currency = "USD"
            contract.lastTradeDateOrContractMonth = "202506"
        elif "Crack Spread" in contract_type:
            # For crack spread, need to create two contracts
            # This is simplified - in reality would need more complex logic
            print("Crack spread trading would require multiple orders")
            return
        
        # Create order
        order = Order()
        order.action = "BUY"
        order.totalQuantity = quantity
        order.orderType = "MKT"
        
        # Submit order
        trade = self.ib.placeOrder(contract, order)
        print(f"Placed order: {trade}")
    
    def place_sell_order(self):
        """Place a sell order via IB"""
        # Similar to buy but with action = "SELL"
        pass
    
    def update_analysis(self):
        """Update analysis tab based on selections"""
        analysis_type = self.analysis_combo.currentText()
        timeframe = self.timeframe_combo.currentText()
        
        # Determine date range based on timeframe
        end_date = pd.Timestamp.now()
        if timeframe == "1 Year":
            start_date = end_date - pd.DateOffset(years=1)
        elif timeframe == "5 Years":
            start_date = end_date - pd.DateOffset(years=5)
        elif timeframe == "10 Years":
            start_date = end_date - pd.DateOffset(years=10)
        else:  # All Data
            start_date = pd.Timestamp('1992-01-01')
        
        # Fetch and display analysis based on type
        if analysis_type == "Seasonal Patterns":
            self.show_seasonal_analysis(start_date, end_date)
        elif analysis_type == "COT Positioning":
            self.show_cot_analysis(start_date, end_date)
        elif analysis_type == "Weather Impact":
            self.show_weather_analysis(start_date, end_date)
        elif analysis_type == "EIA Inventory Analysis":
            self.show_eia_analysis(start_date, end_date)
        elif analysis_type == "OPEC Production Impact":
            self.show_opec_analysis(start_date, end_date)
    
    def show_seasonal_analysis(self, start_date, end_date):
        """Show seasonal patterns analysis"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
            SELECT 
              EXTRACT(MONTH FROM date) AS month,
              AVG((heating_oil_price * 42) - crude_oil_price) AS avg_crack
            FROM price_data
            WHERE date BETWEEN %s AND %s
            GROUP BY EXTRACT(MONTH FROM date)
            ORDER BY EXTRACT(MONTH FROM date)
            """
            cursor.execute(query, (start_date, end_date))
            data = cursor.fetchall()
        
        df = pd.DataFrame(data)
        
        # Plot seasonal patterns
        self.analysis_figure.clear()
        ax = self.analysis_figure.add_subplot(111)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        if not df.empty:
            ax.bar(df['month'], df['avg_crack'])
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(months)
            ax.set_title('Seasonal Crack Spread Patterns')
            ax.set_ylabel('Average Crack Spread ($)')
        
        self.analysis_canvas.draw()
    
    # Additional analysis methods would be implemented similarly
    
    def closeEvent(self, event):
        """Handle application close"""
        if hasattr(self, 'ib') and self.ib.isConnected():
            self.ib.disconnect()
        
        if hasattr(self, 'conn') and not self.conn.closed:
            self.conn.close()
            
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TradingSystemApp()
    window.show()
    sys.exit(app.exec_())