#!/usr/bin/env python
# A very simple backtest using a moving average strategy
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from connectors.postgres_connector import PostgresConnector
from features.feature_generator_wrapper import create_feature_generator

def run_backtest():
    """Run a simple backtest using a moving average strategy."""
    # Connect to the database
    db = PostgresConnector()
    
    # Get CL data
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    logger.info(f"Fetching CL data from {start_date} to {end_date}")
    
    query = f"""
        SELECT timestamp as date, symbol, open, high, low, close, volume
        FROM market_data_backtest
        WHERE symbol = 'CL' 
        AND date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
    """
    
    cl_data = db.query(query)
    
    if not cl_data:
        logger.error("No CL data found")
        return
        
    # Convert to DataFrame
    df = pd.DataFrame(cl_data)
    
    # Ensure date is datetime and set as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    logger.info(f"Loaded {len(df)} rows of CL data")
    
    # Calculate simple moving averages (no need for feature generator complexity)
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # Simple strategy: buy when 20-day SMA crosses above 50-day SMA, sell when it crosses below
    df['signal'] = 0
    df.loc[df['sma_20'] > df['sma_50'], 'signal'] = 1  # Buy signal
    
    # Generate trading signals
    df['position'] = df['signal'].diff()
    
    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate strategy returns (position from yesterday affects today's return)
    df['strategy_return'] = df['signal'].shift(1) * df['daily_return']
    
    # Calculate cumulative returns
    df['cumulative_market_return'] = (1 + df['daily_return']).cumprod() - 1
    df['cumulative_strategy_return'] = (1 + df['strategy_return']).cumprod() - 1
    
    # Drop NaN values for performance metrics
    df_clean = df.dropna()
    
    if df_clean.empty:
        logger.error("No clean data after calculating metrics")
        return
    
    # Calculate performance metrics
    market_return = df_clean['cumulative_market_return'].iloc[-1]
    strategy_return = df_clean['cumulative_strategy_return'].iloc[-1]
    
    # Annualized returns
    days = (df_clean.index[-1] - df_clean.index[0]).days
    market_annual_return = ((1 + market_return) ** (365 / days)) - 1
    strategy_annual_return = ((1 + strategy_return) ** (365 / days)) - 1
    
    # Sharpe ratio (assuming 0% risk-free rate)
    strategy_sharpe = np.sqrt(252) * df_clean['strategy_return'].mean() / df_clean['strategy_return'].std()
    
    # Maximum drawdown
    cum_returns = df_clean['cumulative_strategy_return']
    max_returns = cum_returns.cummax()
    drawdowns = (cum_returns - max_returns) / (1 + max_returns)
    max_drawdown = drawdowns.min()
    
    # Print performance metrics
    logger.info("\nPerformance Metrics:")
    logger.info(f"Total Return (Buy & Hold): {market_return*100:.2f}%")
    logger.info(f"Total Return (Strategy): {strategy_return*100:.2f}%")
    logger.info(f"Annualized Return (Buy & Hold): {market_annual_return*100:.2f}%")
    logger.info(f"Annualized Return (Strategy): {strategy_annual_return*100:.2f}%")
    logger.info(f"Sharpe Ratio (Strategy): {strategy_sharpe:.2f}")
    logger.info(f"Maximum Drawdown (Strategy): {max_drawdown*100:.2f}%")
    
    # Count trades
    trades = df_clean[df_clean['position'] != 0]
    buys = trades[trades['position'] > 0]
    sells = trades[trades['position'] < 0]
    logger.info(f"Number of trades: {len(trades)}")
    logger.info(f"Number of buys: {len(buys)}")
    logger.info(f"Number of sells: {len(sells)}")

    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(df_clean.index, df_clean['close'], label='CL Price')
    plt.plot(df_clean.index, df_clean['sma_20'], label='20-day SMA')
    plt.plot(df_clean.index, df_clean['sma_50'], label='50-day SMA')
    plt.scatter(buys.index, df_clean.loc[buys.index]['close'], marker='^', color='green', label='Buy Signal')
    plt.scatter(sells.index, df_clean.loc[sells.index]['close'], marker='v', color='red', label='Sell Signal')
    plt.title('Crude Oil Price and SMA Crossover Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(df_clean.index, df_clean['cumulative_market_return'] * 100, label='Buy & Hold')
    plt.plot(df_clean.index, df_clean['cumulative_strategy_return'] * 100, label='Strategy')
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_results.png')
    plt.savefig(plot_path)
    logger.info(f"Results plot saved to: {plot_path}")
    
    # Create a simple HTML report
    html_report = f"""
    <html>
    <head>
        <title>Simple Backtest Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            .metrics {{ background-color: #f8f8f8; padding: 20px; border-radius: 5px; }}
            .chart {{ margin-top: 30px; }}
        </style>
    </head>
    <body>
        <h1>Simple Moving Average Crossover Strategy Backtest</h1>
        
        <div class="metrics">
            <h2>Performance Metrics</h2>
            <p><strong>Period:</strong> {df_clean.index[0].strftime('%Y-%m-%d')} to {df_clean.index[-1].strftime('%Y-%m-%d')}</p>
            <p><strong>Total Return (Buy & Hold):</strong> {market_return*100:.2f}%</p>
            <p><strong>Total Return (Strategy):</strong> {strategy_return*100:.2f}%</p>
            <p><strong>Annualized Return (Buy & Hold):</strong> {market_annual_return*100:.2f}%</p>
            <p><strong>Annualized Return (Strategy):</strong> {strategy_annual_return*100:.2f}%</p>
            <p><strong>Sharpe Ratio (Strategy):</strong> {strategy_sharpe:.2f}</p>
            <p><strong>Maximum Drawdown (Strategy):</strong> {max_drawdown*100:.2f}%</p>
            <p><strong>Number of Trades:</strong> {len(trades)}</p>
        </div>
        
        <div class="chart">
            <h2>Performance Chart</h2>
            <img src="backtest_results.png" alt="Backtest Results" style="width: 100%;" />
        </div>
    </body>
    </html>
    """
    
    # Save the HTML report
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_report.html')
    with open(report_path, 'w') as f:
        f.write(html_report)
    
    logger.info(f"HTML report saved to: {report_path}")
    logger.info("\nBacktest completed successfully!")

if __name__ == "__main__":
    run_backtest()