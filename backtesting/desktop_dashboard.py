#!/usr/bin/env python
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QLabel, QPushButton, QComboBox, QDateEdit, QSpinBox, 
    QDoubleSpinBox, QSlider, QGroupBox, QFormLayout, QGridLayout,
    QTableView, QHeaderView, QSplitter, QFrame, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QDate, QAbstractTableModel, QSortFilterProxyModel, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon
import matplotlib
matplotlib.use('Qt5Agg')

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from connectors.postgres_connector import PostgresConnector
from backtesting.engine import BacktestEngine

# Set global matplotlib style
try:
    plt.style.use('seaborn-darkgrid')  # Older style name
except:
    try:
        plt.style.use('seaborn_darkgrid')  # Newer style name
    except:
        plt.style.use('dark_background')  # Fallback style


class BacktestThread(QThread):
    """Thread for running backtests without freezing the UI."""
    finished = pyqtSignal(bool, str)  # Signal for completion with status and results dir
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        
    def run(self):
        try:
            # Initialize database connector
            db = PostgresConnector()
            
            # Create results directory
            results_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'results',
                f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Create BacktestEngine
            engine = BacktestEngine(
                db_connector=db,
                start_date=self.params['start_date'],
                end_date=self.params['end_date'],
                initial_capital=self.params['initial_capital'],
                position_size_pct=self.params['position_size'],
                max_positions=self.params['max_positions'],
                commission_per_contract=self.params['commission'],
                slippage_pct=self.params['slippage'],
                output_dir=results_dir
            )
            
            # Run walk-forward backtest
            success = engine.run_walk_forward_backtest(
                symbols=self.params['symbols'],
                train_days=self.params['train_days'],
                test_days=self.params['test_days'],
                confidence_threshold=self.params['confidence'],
                retrain=self.params['retrain']
            )
            
            # Generate visualizations
            if success:
                engine.visualize_results()
            
            self.finished.emit(success, results_dir)
        
        except Exception as e:
            print(f"Error in backtest thread: {e}")
            self.finished.emit(False, str(e))


class PandasModel(QAbstractTableModel):
    """Model for displaying pandas DataFrames in a QTableView."""
    
    def __init__(self, data=None):
        super(PandasModel, self).__init__()
        self._data = data
        
    def setDataFrame(self, dataframe):
        self.beginResetModel()
        self._data = dataframe
        self.endResetModel()
        
    def rowCount(self, parent=None):
        if self._data is None:
            return 0
        return len(self._data)
        
    def columnCount(self, parent=None):
        if self._data is None:
            return 0
        return len(self._data.columns)
    
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self._data is None:
            return None
            
        value = self._data.iloc[index.row(), index.column()]
        
        if role == Qt.DisplayRole:
            # Format based on value type
            if isinstance(value, float):
                return f"{value:.4f}"
            elif isinstance(value, (datetime, pd.Timestamp)):
                return value.strftime('%Y-%m-%d %H:%M:%S')
            else:
                return str(value)
                
        if role == Qt.BackgroundRole:
            # Highlight positive/negative values
            if isinstance(value, (int, float)):
                if 'pnl' in self._data.columns[index.column()] or 'return' in self._data.columns[index.column()]:
                    if value > 0:
                        return QColor(200, 255, 200)
                    elif value < 0:
                        return QColor(255, 200, 200)
                        
        return None
    
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                if self._data is not None:
                    return str(self._data.columns[section])
            if orientation == Qt.Vertical:
                if self._data is not None:
                    return str(self._data.index[section])
        return None


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots in the PyQt application."""
    
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.fig.tight_layout()


class BacktestDashboard(QMainWindow):
    """Main window for the backtest dashboard."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thor Trading Backtest Dashboard")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize database connector
        self.db_connector = PostgresConnector()
        
        # Store backtest results
        self.current_results = None
        self.results_directory = None
        
        # Create UI components
        self.create_ui()
        
        # Load available backtest results
        self.load_available_backtests()
    
    def create_ui(self):
        """Create the user interface."""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_results_tab()
        self.create_new_backtest_tab()
    
    def create_results_tab(self):
        """Create the tab for viewing backtest results."""
        results_tab = QWidget()
        layout = QVBoxLayout(results_tab)
        
        # Create top controls for selecting results
        controls_layout = QHBoxLayout()
        
        # Backtest selector
        backtest_label = QLabel("Select Backtest:")
        self.backtest_selector = QComboBox()
        self.backtest_selector.setMinimumWidth(300)
        self.backtest_selector.currentIndexChanged.connect(self.load_selected_backtest)
        
        # Refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.load_available_backtests)
        
        controls_layout.addWidget(backtest_label)
        controls_layout.addWidget(self.backtest_selector)
        controls_layout.addWidget(refresh_button)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Create splitter for results
        splitter = QSplitter(Qt.Vertical)
        
        # Create performance metrics panel
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QGridLayout(metrics_group)
        
        self.metric_widgets = {}
        
        metrics = [
            ("Total Return (%)", "total_return", 0, 0),
            ("Annualized Return (%)", "annualized_return", 0, 1),
            ("Sharpe Ratio", "sharpe_ratio", 0, 2),
            ("Sortino Ratio", "sortino_ratio", 0, 3),
            ("Max Drawdown (%)", "max_drawdown", 1, 0),
            ("Win Rate (%)", "win_rate", 1, 1),
            ("Profit Factor", "profit_factor", 1, 2),
            ("Total Trades", "total_trades", 1, 3)
        ]
        
        for label_text, key, row, col in metrics:
            label = QLabel(label_text)
            value = QLabel("--")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value.setStyleSheet("font-weight: bold;")
            
            metrics_layout.addWidget(label, row, col * 2)
            metrics_layout.addWidget(value, row, col * 2 + 1)
            
            self.metric_widgets[key] = value
        
        splitter.addWidget(metrics_group)
        
        # Create charts tabs
        charts_tabs = QTabWidget()
        
        # Create equity curve tab
        equity_tab = QWidget()
        equity_layout = QVBoxLayout(equity_tab)
        self.equity_canvas = MatplotlibCanvas(width=8, height=5)
        equity_layout.addWidget(self.equity_canvas)
        charts_tabs.addTab(equity_tab, "Equity Curve")
        
        # Create trades tab
        trades_tab = QWidget()
        trades_layout = QVBoxLayout(trades_tab)
        
        # Add trade charts
        trades_splitter = QSplitter(Qt.Vertical)
        
        self.trades_canvas = MatplotlibCanvas(width=8, height=3)
        trades_splitter.addWidget(self.trades_canvas)
        
        self.pnl_canvas = MatplotlibCanvas(width=8, height=3)
        trades_splitter.addWidget(self.pnl_canvas)
        
        trades_layout.addWidget(trades_splitter)
        charts_tabs.addTab(trades_tab, "Trades")
        
        # Create returns tab
        returns_tab = QWidget()
        returns_layout = QVBoxLayout(returns_tab)
        
        # Add monthly returns heatmap
        self.monthly_returns_canvas = MatplotlibCanvas(width=8, height=4)
        returns_layout.addWidget(self.monthly_returns_canvas)
        
        charts_tabs.addTab(returns_tab, "Returns")
        
        # Create data tab with tables
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        
        data_tabs = QTabWidget()
        
        # Create portfolio table
        portfolio_tab = QWidget()
        portfolio_layout = QVBoxLayout(portfolio_tab)
        self.portfolio_table = QTableView()
        self.portfolio_model = PandasModel()
        self.portfolio_table.setModel(self.portfolio_model)
        portfolio_layout.addWidget(self.portfolio_table)
        data_tabs.addTab(portfolio_tab, "Portfolio")
        
        # Create trades table
        trades_data_tab = QWidget()
        trades_data_layout = QVBoxLayout(trades_data_tab)
        self.trades_table = QTableView()
        self.trades_model = PandasModel()
        self.trades_table.setModel(self.trades_model)
        trades_data_layout.addWidget(self.trades_table)
        data_tabs.addTab(trades_data_tab, "Trades")
        
        # Create signals table
        signals_tab = QWidget()
        signals_layout = QVBoxLayout(signals_tab)
        self.signals_table = QTableView()
        self.signals_model = PandasModel()
        self.signals_table.setModel(self.signals_model)
        signals_layout.addWidget(self.signals_table)
        data_tabs.addTab(signals_tab, "Signals")
        
        data_layout.addWidget(data_tabs)
        charts_tabs.addTab(data_tab, "Data")
        
        splitter.addWidget(charts_tabs)
        
        # Set splitter sizes
        splitter.setSizes([100, 500])
        
        layout.addWidget(splitter)
        
        self.tabs.addTab(results_tab, "Backtest Results")
    
    def create_new_backtest_tab(self):
        """Create the tab for running new backtests."""
        backtest_tab = QWidget()
        layout = QVBoxLayout(backtest_tab)
        
        # Create parameters form
        params_group = QGroupBox("Backtest Parameters")
        params_layout = QFormLayout(params_group)
        
        # Symbol selection
        self.symbol_selector = QComboBox()
        self.symbol_selector.addItems(["CL", "HO", "CL-HO-SPREAD"])
        self.symbol_selector.setCurrentText("CL-HO-SPREAD")
        params_layout.addRow("Symbol:", self.symbol_selector)
        
        # Date range
        date_layout = QHBoxLayout()
        
        self.start_date = QDateEdit()
        self.start_date.setDisplayFormat("yyyy-MM-dd")
        self.start_date.setDate(QDate.currentDate().addYears(-5))
        self.start_date.setCalendarPopup(True)
        date_layout.addWidget(QLabel("From:"))
        date_layout.addWidget(self.start_date)
        
        self.end_date = QDateEdit()
        self.end_date.setDisplayFormat("yyyy-MM-dd")
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        date_layout.addWidget(QLabel("To:"))
        date_layout.addWidget(self.end_date)
        
        params_layout.addRow("Date Range:", date_layout)
        
        # Capital and position sizing
        self.initial_capital = QSpinBox()
        self.initial_capital.setRange(10000, 10000000)
        self.initial_capital.setSingleStep(10000)
        self.initial_capital.setValue(100000)
        self.initial_capital.setPrefix("$ ")
        params_layout.addRow("Initial Capital:", self.initial_capital)
        
        self.position_size = QDoubleSpinBox()
        self.position_size.setRange(0.01, 0.2)
        self.position_size.setSingleStep(0.01)
        self.position_size.setValue(0.02)
        self.position_size.setDecimals(2)
        self.position_size.setSuffix(" %")
        self.position_size.setSpecialValueText("1%")
        params_layout.addRow("Position Size (% of capital):", self.position_size)
        
        self.max_positions = QSpinBox()
        self.max_positions.setRange(1, 10)
        self.max_positions.setValue(3)
        params_layout.addRow("Max Positions:", self.max_positions)
        
        # Commission and slippage
        self.commission = QDoubleSpinBox()
        self.commission.setRange(0.0, 10.0)
        self.commission.setSingleStep(0.5)
        self.commission.setValue(2.5)
        self.commission.setPrefix("$ ")
        params_layout.addRow("Commission per Contract:", self.commission)
        
        self.slippage = QDoubleSpinBox()
        self.slippage.setRange(0.0, 0.01)
        self.slippage.setSingleStep(0.0001)
        self.slippage.setValue(0.001)
        self.slippage.setDecimals(4)
        self.slippage.setSpecialValueText("0%")
        params_layout.addRow("Slippage:", self.slippage)
        
        # Walk-forward parameters
        self.train_days = QSpinBox()
        self.train_days.setRange(30, 1000)
        self.train_days.setSingleStep(30)
        self.train_days.setValue(365)
        self.train_days.setSuffix(" days")
        params_layout.addRow("Training Window:", self.train_days)
        
        self.test_days = QSpinBox()
        self.test_days.setRange(5, 180)
        self.test_days.setSingleStep(5)
        self.test_days.setValue(30)
        self.test_days.setSuffix(" days")
        params_layout.addRow("Testing Window:", self.test_days)
        
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.5, 1.0)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(0.6)
        self.confidence.setDecimals(2)
        params_layout.addRow("Confidence Threshold:", self.confidence)
        
        # Retraining option
        self.retrain = QComboBox()
        self.retrain.addItems(["Yes", "No"])
        self.retrain.setCurrentText("Yes")
        params_layout.addRow("Retrain Model for Each Window:", self.retrain)
        
        layout.addWidget(params_group)
        
        # Add run button and status area
        button_layout = QHBoxLayout()
        
        self.run_button = QPushButton("Run Backtest")
        self.run_button.setMinimumHeight(40)
        self.run_button.clicked.connect(self.run_backtest)
        button_layout.addWidget(self.run_button)
        
        layout.addLayout(button_layout)
        
        # Status box
        self.status_box = QLabel("Ready to run backtest")
        self.status_box.setAlignment(Qt.AlignCenter)
        self.status_box.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        self.status_box.setMinimumHeight(50)
        layout.addWidget(self.status_box)
        
        self.tabs.addTab(backtest_tab, "New Backtest")
    
    def load_available_backtests(self):
        """Load available backtest results from the results directory."""
        self.backtest_selector.clear()
        
        # Find available backtest results directories
        results_dirs = []
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        
        if os.path.exists(base_dir):
            results_dirs = [
                d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('backtest_')
            ]
            results_dirs.sort(reverse=True)  # Most recent first
        
        # Add to the combobox
        for d in results_dirs:
            timestamp = datetime.strptime(d.split('_')[1], '%Y%m%d%H%M%S')
            display_text = f"{d} ({timestamp.strftime('%Y-%m-%d %H:%M:%S')})"
            self.backtest_selector.addItem(display_text, d)
    
    def load_selected_backtest(self):
        """Load the selected backtest results."""
        if self.backtest_selector.count() == 0:
            return
            
        directory = self.backtest_selector.currentData()
        if not directory:
            return
            
        results_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'results',
            directory
        )
        
        self.load_backtest_results(results_path)
    
    def load_backtest_results(self, results_dir):
        """Load backtest results from a directory."""
        self.results_directory = results_dir
        self.current_results = {}
        
        # Load metrics
        metrics_path = os.path.join(results_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.current_results['metrics'] = json.load(f)
                self.update_metrics_display()
        else:
            print(f"No metrics.json found in {results_dir}")
            self.current_results['metrics'] = {}
        
        # Load portfolio data
        portfolio_path = os.path.join(results_dir, 'portfolio.csv')
        if os.path.exists(portfolio_path):
            self.current_results['portfolio'] = pd.read_csv(portfolio_path)
            
            # Convert date to datetime
            self.current_results['portfolio']['date'] = pd.to_datetime(self.current_results['portfolio']['date'])
            
            # Set date as index
            self.current_results['portfolio'].set_index('date', inplace=True)
            
            # Update portfolio table
            self.portfolio_model.setDataFrame(self.current_results['portfolio'])
            self.portfolio_table.resizeColumnsToContents()
            
            # Update equity curve
            self.plot_equity_curve()
            
            # Update monthly returns
            self.plot_monthly_returns()
        else:
            print(f"No portfolio.csv found in {results_dir}")
            self.current_results['portfolio'] = pd.DataFrame()
        
        # Load trades data
        trades_path = os.path.join(results_dir, 'trades.csv')
        if os.path.exists(trades_path):
            self.current_results['trades'] = pd.read_csv(trades_path)
            
            # Convert date to datetime
            if 'date' in self.current_results['trades'].columns:
                self.current_results['trades']['date'] = pd.to_datetime(self.current_results['trades']['date'])
                
            # Update trades table
            self.trades_model.setDataFrame(self.current_results['trades'])
            self.trades_table.resizeColumnsToContents()
            
            # Update trades plot
            self.plot_trades()
        else:
            print(f"No trades.csv found in {results_dir}")
            self.current_results['trades'] = pd.DataFrame()
        
        # Load signals data
        signals_path = os.path.join(results_dir, 'signals.csv')
        if os.path.exists(signals_path):
            self.current_results['signals'] = pd.read_csv(signals_path)
            
            # Convert date to datetime
            if 'date' in self.current_results['signals'].columns:
                self.current_results['signals']['date'] = pd.to_datetime(self.current_results['signals']['date'])
                self.current_results['signals'].set_index('date', inplace=True)
                
            # Update signals table
            self.signals_model.setDataFrame(self.current_results['signals'])
            self.signals_table.resizeColumnsToContents()
        else:
            print(f"No signals.csv found in {results_dir}")
            self.current_results['signals'] = pd.DataFrame()
    
    def update_metrics_display(self):
        """Update the display of performance metrics."""
        if not self.current_results.get('metrics'):
            return
            
        metrics = self.current_results['metrics']
        
        # Update metric widgets
        for key, widget in self.metric_widgets.items():
            value = metrics.get(key, 0)
            
            # Format based on metric type
            if key in ['total_return', 'annualized_return', 'max_drawdown', 'win_rate']:
                # Convert to percentage
                formatted_value = f"{value * 100:.2f}%"
                
                # Set color based on value
                if key == 'max_drawdown':
                    color = "red" if value < -0.1 else "black"
                else:
                    color = "green" if value > 0 else "red"
                    
                widget.setStyleSheet(f"font-weight: bold; color: {color};")
            elif key in ['sharpe_ratio', 'sortino_ratio', 'profit_factor']:
                formatted_value = f"{value:.2f}"
                
                # Set color based on value
                threshold = 1.0 if key == 'profit_factor' else 0.5
                color = "green" if value > threshold else "red"
                
                widget.setStyleSheet(f"font-weight: bold; color: {color};")
            else:
                # Handle integer metrics
                formatted_value = f"{int(value)}"
                widget.setStyleSheet("font-weight: bold;")
                
            widget.setText(formatted_value)
    
    def plot_equity_curve(self):
        """Plot the equity curve with drawdown."""
        if not self.current_results.get('portfolio') is not None:
            return
            
        portfolio = self.current_results['portfolio']
        
        if 'portfolio_value' not in portfolio.columns:
            return
            
        # Clear the figure
        self.equity_canvas.axes.clear()
        
        # Plot portfolio value
        ax1 = self.equity_canvas.axes
        ax1.plot(portfolio.index, portfolio['portfolio_value'], label='Portfolio Value', color='blue')
        
        # Plot benchmark if available
        if 'close' in portfolio.columns and 'initial_capital' in self.current_results.get('metrics', {}):
            initial_capital = self.current_results['metrics']['initial_capital']
            benchmark = portfolio['close'] * (initial_capital / portfolio['close'].iloc[0])
            ax1.plot(portfolio.index, benchmark, label='Buy & Hold', color='gray', linestyle='--')
        
        # Add drawdown on secondary y-axis if available
        if 'drawdown' in portfolio.columns:
            ax2 = ax1.twinx()
            ax2.fill_between(portfolio.index, portfolio['drawdown'] * 100, 0, 
                            color='red', alpha=0.2, label='Drawdown')
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_ylim(portfolio['drawdown'].min() * 100 * 1.5, 5)  # Leave room at top
        
        # Format the plot
        ax1.set_title('Equity Curve and Drawdown')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Update the canvas
        self.equity_canvas.fig.tight_layout()
        self.equity_canvas.draw()
    
    def plot_trades(self):
        """Plot trades on the price chart and P&L distribution."""
        if self.current_results.get('trades') is None or self.current_results.get('portfolio') is None:
            return
            
        trades = self.current_results['trades']
        portfolio = self.current_results['portfolio']
        
        if trades.empty or portfolio.empty:
            return
            
        # Plot trades on price chart
        self.trades_canvas.axes.clear()
        
        if 'close' in portfolio.columns:
            # Plot price
            self.trades_canvas.axes.plot(portfolio.index, portfolio['close'], label='Price', color='black')
            
            # Extract entry and exit trades
            entries = trades[trades['type'] == 'ENTRY']
            exits = trades[trades['type'] == 'EXIT']
            
            # Plot entries
            long_entries = entries[entries['direction'] == 'LONG']
            if not long_entries.empty:
                self.trades_canvas.axes.scatter(long_entries['date'], long_entries['price'], 
                                         marker='^', color='green', label='Long Entry', s=50)
            
            short_entries = entries[entries['direction'] == 'SHORT']
            if not short_entries.empty:
                self.trades_canvas.axes.scatter(short_entries['date'], short_entries['price'], 
                                         marker='v', color='red', label='Short Entry', s=50)
            
            # Plot exits
            tp_exits = exits[exits['exit_reason'] == 'TAKE_PROFIT']
            if not tp_exits.empty:
                self.trades_canvas.axes.scatter(tp_exits['date'], tp_exits['price'], 
                                         marker='o', color='green', label='Take Profit', s=30)
            
            sl_exits = exits[exits['exit_reason'] == 'STOP_LOSS']
            if not sl_exits.empty:
                self.trades_canvas.axes.scatter(sl_exits['date'], sl_exits['price'], 
                                         marker='o', color='red', label='Stop Loss', s=30)
            
            # Format the plot
            self.trades_canvas.axes.set_title('Trades')
            self.trades_canvas.axes.set_xlabel('Date')
            self.trades_canvas.axes.set_ylabel('Price')
            self.trades_canvas.axes.legend(loc='best')
            self.trades_canvas.axes.grid(True)
            
            # Format x-axis dates
            self.trades_canvas.axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(self.trades_canvas.axes.xaxis.get_majorticklabels(), rotation=45)
            
            # Update the canvas
            self.trades_canvas.fig.tight_layout()
            self.trades_canvas.draw()
        
        # Plot P&L distribution
        self.pnl_canvas.axes.clear()
        
        if 'pnl' in trades.columns:
            # Get exit trades only
            exit_trades = trades[trades['type'] == 'EXIT']
            
            if not exit_trades.empty:
                # Plot histogram by direction
                long_pnl = exit_trades[exit_trades['direction'] == 'LONG']['pnl']
                short_pnl = exit_trades[exit_trades['direction'] == 'SHORT']['pnl']
                
                bins = np.linspace(exit_trades['pnl'].min(), exit_trades['pnl'].max(), 30)
                
                if not long_pnl.empty:
                    self.pnl_canvas.axes.hist(long_pnl, bins=bins, alpha=0.5, color='green', label='Long')
                
                if not short_pnl.empty:
                    self.pnl_canvas.axes.hist(short_pnl, bins=bins, alpha=0.5, color='red', label='Short')
                
                # Add vertical line at 0
                self.pnl_canvas.axes.axvline(0, color='black', linestyle='--')
                
                # Format the plot
                self.pnl_canvas.axes.set_title('P&L Distribution')
                self.pnl_canvas.axes.set_xlabel('P&L ($)')
                self.pnl_canvas.axes.set_ylabel('Frequency')
                self.pnl_canvas.axes.legend(loc='best')
                self.pnl_canvas.axes.grid(True)
                
                # Update the canvas
                self.pnl_canvas.fig.tight_layout()
                self.pnl_canvas.draw()
    
    def plot_monthly_returns(self):
        """Plot monthly returns heatmap."""
        if self.current_results.get('portfolio') is None:
            return
            
        portfolio = self.current_results['portfolio']
        
        if portfolio.empty:
            return
            
        # Calculate daily returns if not already present
        if 'daily_return' not in portfolio.columns:
            portfolio['daily_return'] = portfolio['portfolio_value'].pct_change()
        
        # Calculate monthly returns
        monthly_returns = portfolio['daily_return'].resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        if monthly_returns.empty:
            return
            
        # Create DataFrame with year and month
        years = monthly_returns.index.year.unique()
        months = range(1, 13)
        
        # Create pivot table
        data = np.zeros((len(years), 12))
        data.fill(np.nan)  # Fill with NaN for cells without data
        
        for i, year in enumerate(years):
            for j, month in enumerate(months):
                # Find matching dates
                matches = monthly_returns[
                    (monthly_returns.index.year == year) & 
                    (monthly_returns.index.month == month)
                ]
                
                if not matches.empty:
                    data[i, j] = matches.iloc[0] * 100  # Convert to percentage
        
        # Clear the figure
        self.monthly_returns_canvas.axes.clear()
        
        # Create heatmap
        cmap = plt.cm.RdYlGn  # Red for negative, green for positive
        im = self.monthly_returns_canvas.axes.imshow(data, cmap=cmap, vmin=-10, vmax=10)
        
        # Add colorbar
        cbar = self.monthly_returns_canvas.fig.colorbar(im, ax=self.monthly_returns_canvas.axes)
        cbar.set_label('Monthly Return (%)')
        
        # Set labels
        self.monthly_returns_canvas.axes.set_title('Monthly Returns')
        self.monthly_returns_canvas.axes.set_xlabel('Month')
        self.monthly_returns_canvas.axes.set_ylabel('Year')
        
        # Set ticks
        self.monthly_returns_canvas.axes.set_xticks(np.arange(12))
        self.monthly_returns_canvas.axes.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        self.monthly_returns_canvas.axes.set_yticks(np.arange(len(years)))
        self.monthly_returns_canvas.axes.set_yticklabels(years)
        
        # Rotate the x labels
        plt.setp(self.monthly_returns_canvas.axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add values in cells
        for i in range(len(years)):
            for j in range(12):
                if not np.isnan(data[i, j]):
                    text_color = 'white' if abs(data[i, j]) > 5 else 'black'
                    self.monthly_returns_canvas.axes.text(j, i, f"{data[i, j]:.1f}%", 
                                ha="center", va="center", color=text_color)
        
        # Update the canvas
        self.monthly_returns_canvas.fig.tight_layout()
        self.monthly_returns_canvas.draw()
    
    def run_backtest(self):
        """Run a new backtest with the current parameters."""
        # Get parameters from the UI
        params = {
            'symbols': [self.symbol_selector.currentText()],
            'start_date': self.start_date.date().toPyDate(),
            'end_date': self.end_date.date().toPyDate(),
            'initial_capital': self.initial_capital.value(),
            'position_size': self.position_size.value() / 100.0,  # Convert from percentage
            'max_positions': self.max_positions.value(),
            'commission': self.commission.value(),
            'slippage': self.slippage.value(),
            'train_days': self.train_days.value(),
            'test_days': self.test_days.value(),
            'confidence': self.confidence.value(),
            'retrain': self.retrain.currentText() == "Yes"
        }
        
        # Update status
        self.status_box.setText("Running backtest... This may take a while.")
        self.status_box.setStyleSheet("background-color: #fff3cd; padding: 10px; border-radius: 5px;")
        self.run_button.setEnabled(False)
        QApplication.processEvents()
        
        # Create and start the backtest thread
        self.backtest_thread = BacktestThread(params)
        self.backtest_thread.finished.connect(self.on_backtest_finished)
        self.backtest_thread.start()
    
    def on_backtest_finished(self, success, results_dir):
        """Handle the completion of a backtest."""
        if success:
            self.status_box.setText(f"Backtest completed successfully! Results saved to: {results_dir}")
            self.status_box.setStyleSheet("background-color: #d4edda; padding: 10px; border-radius: 5px;")
            
            # Refresh available backtests and select the new one
            self.load_available_backtests()
            
            # Find and select the new backtest
            for i in range(self.backtest_selector.count()):
                if self.backtest_selector.itemData(i) == os.path.basename(results_dir):
                    self.backtest_selector.setCurrentIndex(i)
                    break
            
            # Switch to results tab
            self.tabs.setCurrentIndex(0)
        else:
            self.status_box.setText(f"Backtest failed: {results_dir}")
            self.status_box.setStyleSheet("background-color: #f8d7da; padding: 10px; border-radius: 5px;")
        
        self.run_button.setEnabled(True)


def main():
    """Main function."""
    app = QApplication(sys.argv)
    
    # Set the application style
    app.setStyle("Fusion")
    
    # Create and show the main window
    window = BacktestDashboard()
    window.show()
    
    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()