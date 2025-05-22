import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import glob

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connectors.postgres_connector import PostgresConnector
from backtesting.engine import BacktestEngine


def load_backtest_results(results_dir):
    """
    Load backtest results from a directory.
    
    Args:
        results_dir: Path to backtest results directory
    
    Returns:
        Dictionary with loaded data
    """
    results = {}
    
    # Load metrics
    metrics_path = os.path.join(results_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            results['metrics'] = json.load(f)
    else:
        st.warning(f"No metrics.json found in {results_dir}")
        results['metrics'] = {}
    
    # Load portfolio data
    portfolio_path = os.path.join(results_dir, 'portfolio.csv')
    if os.path.exists(portfolio_path):
        results['portfolio'] = pd.read_csv(portfolio_path)
        
        # Convert date to datetime
        results['portfolio']['date'] = pd.to_datetime(results['portfolio']['date'])
        
        # Set date as index
        results['portfolio'].set_index('date', inplace=True)
    else:
        st.warning(f"No portfolio.csv found in {results_dir}")
        results['portfolio'] = pd.DataFrame()
    
    # Load trades data
    trades_path = os.path.join(results_dir, 'trades.csv')
    if os.path.exists(trades_path):
        results['trades'] = pd.read_csv(trades_path)
        
        # Convert date to datetime
        if 'date' in results['trades'].columns:
            results['trades']['date'] = pd.to_datetime(results['trades']['date'])
    else:
        st.warning(f"No trades.csv found in {results_dir}")
        results['trades'] = pd.DataFrame()
    
    # Load signals data
    signals_path = os.path.join(results_dir, 'signals.csv')
    if os.path.exists(signals_path):
        results['signals'] = pd.read_csv(signals_path)
        
        # Convert date to datetime
        if 'date' in results['signals'].columns:
            results['signals']['date'] = pd.to_datetime(results['signals']['date'])
            results['signals'].set_index('date', inplace=True)
    else:
        st.warning(f"No signals.csv found in {results_dir}")
        results['signals'] = pd.DataFrame()
    
    return results


def run_backtest(symbols, start_date, end_date, initial_capital, position_size, train_days, test_days, confidence):
    """
    Run a backtest using the BacktestEngine.
    
    Args:
        symbols: List of symbols to backtest
        start_date: Start date for the backtest
        end_date: End date for the backtest
        initial_capital: Initial capital for the backtest
        position_size: Position size as a percentage of capital
        train_days: Number of days for training
        test_days: Number of days for testing
        confidence: Confidence threshold for actionable signals
        
    Returns:
        Tuple of (success, results_dir) where results_dir is the path to the backtest results
    """
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
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        position_size_pct=position_size,
        max_positions=3,
        commission_per_contract=2.5,
        slippage_pct=0.001,
        output_dir=results_dir
    )
    
    # Run walk-forward backtest
    success = engine.run_walk_forward_backtest(
        symbols=symbols,
        train_days=train_days,
        test_days=test_days,
        confidence_threshold=confidence,
        retrain=True
    )
    
    # Generate visualizations
    if success:
        engine.visualize_results()
    
    return success, results_dir


def plot_equity_curve(portfolio_df, initial_capital):
    """Plot equity curve using Plotly."""
    fig = go.Figure()
    
    # Add portfolio equity line
    fig.add_trace(go.Scatter(
        x=portfolio_df.index,
        y=portfolio_df['portfolio_value'],
        name='Portfolio Value',
        line=dict(color='blue', width=2)
    ))
    
    # Add benchmark line (Buy & Hold of the underlying)
    if 'close' in portfolio_df.columns:
        benchmark = portfolio_df['close'] * (initial_capital / portfolio_df['close'].iloc[0])
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=benchmark,
            name='Buy & Hold',
            line=dict(color='gray', width=1, dash='dot')
        ))
    
    # Add drawdown as area chart on secondary y-axis
    if 'drawdown' in portfolio_df.columns:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['portfolio_value'],
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ), secondary_y=False)
        
        if 'close' in portfolio_df.columns:
            benchmark = portfolio_df['close'] * (initial_capital / portfolio_df['close'].iloc[0])
            fig.add_trace(go.Scatter(
                x=portfolio_df.index,
                y=benchmark,
                name='Buy & Hold',
                line=dict(color='gray', width=1, dash='dot')
            ), secondary_y=False)
        
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['drawdown'] * 100,
            name='Drawdown %',
            fill='tozeroy',
            line=dict(color='red', width=1),
            fillcolor='rgba(255, 0, 0, 0.1)'
        ), secondary_y=True)
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Portfolio Value ($)", secondary_y=False)
        fig.update_yaxes(title_text="Drawdown (%)", secondary_y=True)
    
    fig.update_layout(
        title='Equity Curve and Drawdown',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        legend=dict(x=0, y=1, traceorder='normal'),
        height=600
    )
    
    return fig


def plot_trades(trades_df, portfolio_df):
    """Plot trades on top of price chart."""
    if trades_df.empty or portfolio_df.empty:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add price line
    if 'close' in portfolio_df.columns:
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['close'],
            name='Price',
            line=dict(color='black', width=1)
        ))
    
    # Extract entry and exit trades
    entries = trades_df[trades_df['type'] == 'ENTRY']
    exits = trades_df[trades_df['type'] == 'EXIT']
    
    # Add entry points
    for direction in ['LONG', 'SHORT']:
        direction_entries = entries[entries['direction'] == direction]
        if not direction_entries.empty:
            marker_color = 'green' if direction == 'LONG' else 'red'
            marker_symbol = 'triangle-up' if direction == 'LONG' else 'triangle-down'
            
            fig.add_trace(go.Scatter(
                x=direction_entries['date'],
                y=direction_entries['price'],
                mode='markers',
                name=f'{direction} Entry',
                marker=dict(
                    color=marker_color,
                    size=10,
                    symbol=marker_symbol,
                    line=dict(width=1, color='black')
                )
            ))
    
    # Add exit points
    for exit_reason in exits['exit_reason'].unique():
        if pd.notna(exit_reason):
            reason_exits = exits[exits['exit_reason'] == exit_reason]
            
            marker_color = 'purple'
            if exit_reason == 'TAKE_PROFIT':
                marker_color = 'green'
            elif exit_reason == 'STOP_LOSS':
                marker_color = 'red'
            
            fig.add_trace(go.Scatter(
                x=reason_exits['date'],
                y=reason_exits['price'],
                mode='markers',
                name=f'Exit ({exit_reason})',
                marker=dict(
                    color=marker_color,
                    size=8,
                    symbol='x',
                    line=dict(width=1, color='black')
                )
            ))
    
    fig.update_layout(
        title='Trades and Price',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(x=0, y=1, traceorder='normal'),
        height=600
    )
    
    return fig


def plot_monthly_returns(portfolio_df):
    """Plot monthly returns heatmap."""
    if 'daily_return' not in portfolio_df.columns:
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
    
    # Calculate monthly returns
    monthly_returns = portfolio_df['daily_return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Create DataFrame with year and month
    df_monthly = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values * 100  # Convert to percentage
    })
    
    # Pivot data for heatmap
    pivot_table = df_monthly.pivot(index='Year', columns='Month', values='Return')
    
    # Create a Plotly heatmap
    fig = px.imshow(
        pivot_table,
        labels=dict(x="Month", y="Year", color="Return (%)"),
        x=[f"{m}" for m in range(1, 13)],
        y=pivot_table.index,
        color_continuous_scale='RdYlGn',
        zmin=-10,  # Min 10% loss
        zmax=10,   # Max 10% gain
    )
    
    fig.update_layout(
        title='Monthly Returns (%)',
        xaxis_title='Month',
        yaxis_title='Year',
        coloraxis_colorbar=dict(title='Return (%)'),
        height=500,
        width=800
    )
    
    return fig


def plot_trade_distribution(trades_df):
    """Plot trade P&L distribution."""
    if trades_df.empty or 'pnl' not in trades_df.columns:
        return None
    
    # Only use exit trades to avoid double counting
    exit_trades = trades_df[trades_df['type'] == 'EXIT']
    
    # Create histogram
    fig = px.histogram(
        exit_trades,
        x='pnl',
        color='direction',
        nbins=30,
        opacity=0.7,
        barmode='overlay',
        labels={'pnl': 'P&L ($)', 'direction': 'Direction'},
        color_discrete_map={'LONG': 'green', 'SHORT': 'red'}
    )
    
    # Add vertical line at 0
    fig.add_shape(
        type="line",
        x0=0, x1=0,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="black", width=1, dash="dot")
    )
    
    fig.update_layout(
        title='Trade P&L Distribution',
        xaxis_title='P&L ($)',
        yaxis_title='Frequency',
        legend=dict(x=0, y=1, traceorder='normal'),
        height=400
    )
    
    return fig


def plot_rolling_returns(portfolio_df):
    """Plot rolling returns for different time periods."""
    if 'daily_return' not in portfolio_df.columns:
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
    
    # Calculate rolling returns
    windows = [30, 60, 90, 180, 365]
    
    fig = go.Figure()
    
    for window in windows:
        if len(portfolio_df) > window:
            # Calculate rolling annualized return
            rolling_return = (portfolio_df['daily_return'] + 1).rolling(window=window).apply(
                lambda x: (x.prod() ** (252 / window)) - 1, 
                raw=False
            ) * 100  # Convert to percentage
            
            fig.add_trace(go.Scatter(
                x=portfolio_df.index,
                y=rolling_return,
                name=f'{window}-Day Rolling Return',
                line=dict(width=1)
            ))
    
    fig.update_layout(
        title='Rolling Annualized Returns',
        xaxis_title='Date',
        yaxis_title='Annualized Return (%)',
        legend=dict(x=0, y=1, traceorder='normal'),
        height=400
    )
    
    # Add a horizontal line at 0
    fig.add_shape(
        type="line",
        x0=portfolio_df.index.min(),
        x1=portfolio_df.index.max(),
        y0=0, y1=0,
        line=dict(color="black", width=1)
    )
    
    return fig


def plot_signals(signals_df):
    """Plot signal strength and confidence over time."""
    if signals_df.empty:
        return None
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price
    if 'close' in signals_df.columns:
        fig.add_trace(go.Scatter(
            x=signals_df.index,
            y=signals_df['close'],
            name='Price',
            line=dict(color='black', width=1)
        ), secondary_y=False)
    
    # Add confidence
    if 'confidence' in signals_df.columns:
        fig.add_trace(go.Scatter(
            x=signals_df.index,
            y=signals_df['confidence'],
            name='Confidence',
            line=dict(color='blue', width=1)
        ), secondary_y=True)
    
    # Add signal direction
    if 'signal' in signals_df.columns:
        # Create colormap for signal direction
        signal_colors = {
            1: 'green',   # Buy
            0: 'gray',    # Neutral
            -1: 'red'     # Sell
        }
        
        # Only plot actionable signals
        action_signals = signals_df[signals_df['signal'] != 0]
        
        fig.add_trace(go.Scatter(
            x=action_signals.index,
            y=action_signals['close'] if 'close' in action_signals.columns else [0] * len(action_signals),
            mode='markers',
            name='Signals',
            marker=dict(
                color=[signal_colors.get(s, 'gray') for s in action_signals['signal']],
                size=8,
                symbol=['triangle-up' if s == 1 else 'triangle-down' for s in action_signals['signal']]
            )
        ), secondary_y=False)
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Confidence", secondary_y=True)
    
    fig.update_layout(
        title='Trading Signals and Confidence',
        xaxis_title='Date',
        legend=dict(x=0, y=1, traceorder='normal'),
        height=500
    )
    
    return fig


def main():
    """Main function for Streamlit app."""
    st.set_page_config(page_title="Thor Trading Backtest Dashboard", 
                      page_icon="📈", 
                      layout="wide")
    
    st.title("Thor Trading Backtest Dashboard")
    st.sidebar.header("Options")
    
    tab1, tab2 = st.tabs(["Load Existing Backtest", "Run New Backtest"])
    
    with tab1:
        # Find available backtest results directories
        results_dirs = []
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        
        if os.path.exists(base_dir):
            results_dirs = [
                d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('backtest_')
            ]
            results_dirs.sort(reverse=True)  # Most recent first
        
        if not results_dirs:
            st.warning("No existing backtest results found.")
        else:
            # Select backtest results
            selected_dir = st.selectbox(
                "Select Backtest Results",
                options=results_dirs,
                index=0,
                format_func=lambda x: f"{x} ({datetime.strptime(x.split('_')[1], '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S')})"
            )
            
            if selected_dir:
                results_path = os.path.join(base_dir, selected_dir)
                results = load_backtest_results(results_path)
                
                if results['metrics']:
                    # Display performance metrics
                    st.header("Performance Metrics")
                    
                    # Create columns for metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Key performance metrics
                    with col1:
                        st.metric("Total Return", f"{results['metrics'].get('total_return', 0) * 100:.2f}%")
                        st.metric("Annualized Return", f"{results['metrics'].get('annualized_return', 0) * 100:.2f}%")
                    with col2:
                        st.metric("Sharpe Ratio", f"{results['metrics'].get('sharpe_ratio', 0):.2f}")
                        st.metric("Sortino Ratio", f"{results['metrics'].get('sortino_ratio', 0):.2f}")
                    with col3:
                        st.metric("Max Drawdown", f"{results['metrics'].get('max_drawdown', 0) * 100:.2f}%")
                        st.metric("Win Rate", f"{results['metrics'].get('win_rate', 0) * 100:.2f}%")
                    with col4:
                        st.metric("Total Trades", f"{results['metrics'].get('total_trades', 0)}")
                        st.metric("Profit Factor", f"{results['metrics'].get('profit_factor', 0):.2f}")
                    
                    # Additional metrics in expandable section
                    with st.expander("All Metrics"):
                        st.json(results['metrics'])
                
                # Equity curve
                if not results['portfolio'].empty:
                    st.header("Equity Curve")
                    
                    fig = plot_equity_curve(
                        results['portfolio'], 
                        results['metrics'].get('initial_capital', 100000)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Create tabs for different visualizations
                viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["Trades", "Returns Analysis", "Signals", "Raw Data"])
                
                with viz_tab1:
                    # Trades visualization
                    if not results['trades'].empty:
                        st.subheader("Trades on Price Chart")
                        fig = plot_trades(results['trades'], results['portfolio'])
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("Trade P&L Distribution")
                        fig = plot_trade_distribution(results['trades'])
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Trade statistics
                        if 'pnl' in results['trades'].columns:
                            exit_trades = results['trades'][results['trades']['type'] == 'EXIT']
                            
                            st.subheader("Trade Statistics")
                            
                            trade_stats_col1, trade_stats_col2 = st.columns(2)
                            
                            with trade_stats_col1:
                                # Long trade stats
                                long_trades = exit_trades[exit_trades['direction'] == 'LONG']
                                st.subheader("Long Trades")
                                
                                if not long_trades.empty:
                                    long_wins = len(long_trades[long_trades['pnl'] > 0])
                                    long_losses = len(long_trades[long_trades['pnl'] <= 0])
                                    long_win_rate = long_wins / len(long_trades) if len(long_trades) > 0 else 0
                                    
                                    st.write(f"Total: {len(long_trades)}")
                                    st.write(f"Wins: {long_wins} ({long_win_rate:.1%})")
                                    st.write(f"Losses: {long_losses} ({1-long_win_rate:.1%})")
                                    
                                    if long_wins > 0:
                                        avg_win = long_trades[long_trades['pnl'] > 0]['pnl'].mean()
                                        st.write(f"Avg Win: ${avg_win:.2f}")
                                    
                                    if long_losses > 0:
                                        avg_loss = long_trades[long_trades['pnl'] <= 0]['pnl'].mean()
                                        st.write(f"Avg Loss: ${avg_loss:.2f}")
                                else:
                                    st.write("No long trades")
                            
                            with trade_stats_col2:
                                # Short trade stats
                                short_trades = exit_trades[exit_trades['direction'] == 'SHORT']
                                st.subheader("Short Trades")
                                
                                if not short_trades.empty:
                                    short_wins = len(short_trades[short_trades['pnl'] > 0])
                                    short_losses = len(short_trades[short_trades['pnl'] <= 0])
                                    short_win_rate = short_wins / len(short_trades) if len(short_trades) > 0 else 0
                                    
                                    st.write(f"Total: {len(short_trades)}")
                                    st.write(f"Wins: {short_wins} ({short_win_rate:.1%})")
                                    st.write(f"Losses: {short_losses} ({1-short_win_rate:.1%})")
                                    
                                    if short_wins > 0:
                                        avg_win = short_trades[short_trades['pnl'] > 0]['pnl'].mean()
                                        st.write(f"Avg Win: ${avg_win:.2f}")
                                    
                                    if short_losses > 0:
                                        avg_loss = short_trades[short_trades['pnl'] <= 0]['pnl'].mean()
                                        st.write(f"Avg Loss: ${avg_loss:.2f}")
                                else:
                                    st.write("No short trades")
                            
                            # Exit reasons
                            st.subheader("Exit Reasons")
                            if 'exit_reason' in exit_trades.columns:
                                exit_reasons = exit_trades['exit_reason'].value_counts()
                                
                                reason_cols = st.columns(len(exit_reasons))
                                for i, (reason, count) in enumerate(exit_reasons.items()):
                                    with reason_cols[i]:
                                        st.metric(reason, count)
                                
                                # PnL by exit reason
                                reason_pnl = exit_trades.groupby('exit_reason')['pnl'].agg(['mean', 'sum', 'count'])
                                
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=reason_pnl.index,
                                        y=reason_pnl['sum'],
                                        name='Total P&L',
                                        text=reason_pnl['count'],
                                        textposition='auto',
                                        marker_color=['green' if x > 0 else 'red' for x in reason_pnl['sum']]
                                    )
                                ])
                                
                                fig.update_layout(
                                    title='P&L by Exit Reason (with trade count)',
                                    xaxis_title='Exit Reason',
                                    yaxis_title='Total P&L ($)',
                                    height=300
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                
                with viz_tab2:
                    # Returns analysis
                    if not results['portfolio'].empty:
                        st.subheader("Monthly Returns")
                        fig = plot_monthly_returns(results['portfolio'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("Rolling Returns")
                        fig = plot_rolling_returns(results['portfolio'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Returns distribution
                        if 'daily_return' not in results['portfolio'].columns:
                            results['portfolio']['daily_return'] = results['portfolio']['portfolio_value'].pct_change()
                        
                        st.subheader("Returns Distribution")
                        
                        returns_col1, returns_col2 = st.columns(2)
                        
                        with returns_col1:
                            # Daily returns histogram
                            fig = px.histogram(
                                results['portfolio'],
                                x='daily_return',
                                nbins=50,
                                labels={'daily_return': 'Daily Return'},
                                title='Daily Returns Distribution'
                            )
                            
                            fig.add_shape(
                                type="line",
                                x0=0, x1=0,
                                y0=0, y1=1,
                                yref="paper",
                                line=dict(color="black", width=1, dash="dot")
                            )
                            
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with returns_col2:
                            # Return statistics
                            daily_returns = results['portfolio']['daily_return'].dropna()
                            
                            st.subheader("Return Statistics")
                            st.write(f"Mean Daily Return: {daily_returns.mean() * 100:.4f}%")
                            st.write(f"Median Daily Return: {daily_returns.median() * 100:.4f}%")
                            st.write(f"Daily Return Std Dev: {daily_returns.std() * 100:.4f}%")
                            st.write(f"Skewness: {daily_returns.skew():.4f}")
                            st.write(f"Kurtosis: {daily_returns.kurtosis():.4f}")
                            st.write(f"Positive Days: {(daily_returns > 0).sum()} ({(daily_returns > 0).mean() * 100:.1f}%)")
                            st.write(f"Negative Days: {(daily_returns < 0).sum()} ({(daily_returns < 0).mean() * 100:.1f}%)")
                
                with viz_tab3:
                    # Signal analysis
                    if not results['signals'].empty:
                        st.subheader("Trading Signals")
                        fig = plot_signals(results['signals'])
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Signal statistics
                        if 'signal' in results['signals'].columns:
                            signal_counts = results['signals']['signal'].value_counts()
                            
                            st.subheader("Signal Statistics")
                            
                            signal_cols = st.columns(len(signal_counts))
                            signal_labels = {1: "Buy", 0: "Neutral", -1: "Sell"}
                            
                            for i, (signal, count) in enumerate(signal_counts.items()):
                                with signal_cols[i]:
                                    label = signal_labels.get(signal, f"Signal {signal}")
                                    st.metric(label, count)
                
                with viz_tab4:
                    # Raw data tables
                    st.subheader("Portfolio Data")
                    if not results['portfolio'].empty:
                        st.dataframe(results['portfolio'])
                    
                    st.subheader("Trades Data")
                    if not results['trades'].empty:
                        st.dataframe(results['trades'])
                    
                    st.subheader("Signals Data")
                    if not results['signals'].empty:
                        st.dataframe(results['signals'])
    
    with tab2:
        st.header("Run New Backtest")
        
        # Input parameters
        st.subheader("Backtest Parameters")
        
        input_col1, input_col2 = st.columns(2)
        
        with input_col1:
            # Date range
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365 * 5),  # Default to 5 years ago
                max_value=datetime.now()
            )
            
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                min_value=start_date,
                max_value=datetime.now()
            )
            
            # Symbols
            symbols = st.multiselect(
                "Symbols",
                options=["CL", "HO", "CL-HO-SPREAD"],
                default=["CL", "HO", "CL-HO-SPREAD"]
            )
        
        with input_col2:
            # Capital and position sizing
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=10000
            )
            
            position_size = st.slider(
                "Position Size (% of capital)",
                min_value=0.01,
                max_value=0.2,
                value=0.02,
                step=0.01,
                format="%.2f"
            )
            
            # Walk-forward parameters
            train_days = st.slider(
                "Training Window (days)",
                min_value=30,
                max_value=1000,
                value=365,
                step=30
            )
            
            test_days = st.slider(
                "Testing Window (days)",
                min_value=5,
                max_value=180,
                value=30,
                step=5
            )
            
            confidence = st.slider(
                "Confidence Threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.6,
                step=0.05
            )
        
        # Run backtest button
        if st.button("Run Backtest"):
            if not symbols:
                st.error("Please select at least one symbol")
            else:
                with st.spinner("Running backtest..."):
                    # Convert dates to datetime
                    start_datetime = datetime.combine(start_date, datetime.min.time())
                    end_datetime = datetime.combine(end_date, datetime.min.time())
                    
                    # Run backtest
                    success, results_dir = run_backtest(
                        symbols=symbols,
                        start_date=start_datetime,
                        end_date=end_datetime,
                        initial_capital=initial_capital,
                        position_size=position_size,
                        train_days=train_days,
                        test_days=test_days,
                        confidence=confidence
                    )
                    
                    if success:
                        st.success(f"Backtest completed successfully!")
                        st.info(f"Results saved to: {results_dir}")
                        
                        # Add button to switch to the results tab
                        if st.button("View Results"):
                            # This doesn't actually switch tabs - just advise the user
                            st.info("Please switch to the 'Load Existing Backtest' tab to see the results")
                    else:
                        st.error("Backtest failed. Check the logs for more information.")


if __name__ == "__main__":
    main()