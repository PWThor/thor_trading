#!/usr/bin/env python
import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare backtest strategies')
    
    parser.add_argument('--dirs', type=str, nargs='+', required=True, help='Directories with backtest results')
    parser.add_argument('--names', type=str, nargs='+', help='Names for strategies (defaults to directory names)')
    parser.add_argument('--output', type=str, default='comparison_results', help='Output directory for comparison')
    
    return parser.parse_args()

def load_backtest_results(directory: str) -> Dict:
    """
    Load backtest results from a directory.
    
    Args:
        directory: Path to backtest results directory
        
    Returns:
        Dictionary with loaded data
    """
    results = {}
    
    # Load metrics
    metrics_path = os.path.join(directory, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            results['metrics'] = json.load(f)
    else:
        logger.warning(f"No metrics.json found in {directory}")
        results['metrics'] = {}
    
    # Load portfolio data
    portfolio_path = os.path.join(directory, 'portfolio.csv')
    if os.path.exists(portfolio_path):
        results['portfolio'] = pd.read_csv(portfolio_path)
        
        # Convert date to datetime
        results['portfolio']['date'] = pd.to_datetime(results['portfolio']['date'])
        
        # Set date as index
        results['portfolio'].set_index('date', inplace=True)
    else:
        logger.warning(f"No portfolio.csv found in {directory}")
        results['portfolio'] = pd.DataFrame()
    
    # Load trades data
    trades_path = os.path.join(directory, 'trades.csv')
    if os.path.exists(trades_path):
        results['trades'] = pd.read_csv(trades_path)
        
        # Convert date to datetime
        if 'date' in results['trades'].columns:
            results['trades']['date'] = pd.to_datetime(results['trades']['date'])
    else:
        logger.warning(f"No trades.csv found in {directory}")
        results['trades'] = pd.DataFrame()
    
    return results

def compare_metrics(backtest_results: Dict[str, Dict], strategy_names: List[str]) -> pd.DataFrame:
    """
    Compare metrics from multiple backtest results.
    
    Args:
        backtest_results: Dictionary with backtest results by strategy
        strategy_names: Names of strategies
        
    Returns:
        DataFrame with comparison
    """
    metrics = {}
    
    # Get metrics for each strategy
    for strategy, results in backtest_results.items():
        if 'metrics' in results:
            metrics[strategy] = results['metrics']
    
    if not metrics:
        logger.error("No metrics found for comparison")
        return pd.DataFrame()
    
    # Create comparison DataFrame
    comparison = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Rename index to strategy names
    comparison.index = strategy_names
    
    # Format metrics for display
    for col in ['total_return', 'annualized_return', 'max_drawdown', 'win_rate']:
        if col in comparison.columns:
            comparison[col] = comparison[col] * 100
    
    # Select and reorder important columns
    important_cols = [
        'initial_capital', 'final_portfolio_value', 'total_return', 'annualized_return', 
        'max_drawdown', 'sharpe_ratio', 'sortino_ratio', 'total_trades', 
        'win_rate', 'profit_factor', 'avg_win', 'avg_loss'
    ]
    
    # Filter to only columns that exist
    existing_cols = [col for col in important_cols if col in comparison.columns]
    
    comparison = comparison[existing_cols]
    
    return comparison

def compare_equity_curves(backtest_results: Dict[str, Dict], strategy_names: List[str], output_dir: str) -> None:
    """
    Compare equity curves from multiple backtest results.
    
    Args:
        backtest_results: Dictionary with backtest results by strategy
        strategy_names: Names of strategies
        output_dir: Output directory for comparison charts
    """
    # Get portfolio data for each strategy
    portfolios = {}
    
    for strategy, results in backtest_results.items():
        if 'portfolio' in results and not results['portfolio'].empty:
            portfolios[strategy] = results['portfolio']
    
    if not portfolios:
        logger.error("No portfolio data found for comparison")
        return
    
    # Find common date range
    min_dates = []
    max_dates = []
    
    for portfolio in portfolios.values():
        min_dates.append(portfolio.index.min())
        max_dates.append(portfolio.index.max())
    
    common_start = max(min_dates)
    common_end = min(max_dates)
    
    logger.info(f"Common date range: {common_start} to {common_end}")
    
    # Get initial capital for each strategy
    initial_capitals = {}
    
    for strategy, results in backtest_results.items():
        if 'metrics' in results and 'initial_capital' in results['metrics']:
            initial_capitals[strategy] = results['metrics']['initial_capital']
        else:
            # Default to 100000 if not found
            initial_capitals[strategy] = 100000
    
    # Create equity curves DataFrame
    equity_curves = pd.DataFrame(index=pd.date_range(common_start, common_end))
    
    # Resample and normalize equity curves
    for i, (strategy, portfolio) in enumerate(portfolios.items()):
        # Filter to common date range
        portfolio = portfolio.loc[common_start:common_end]
        
        # Resample to daily frequency if needed
        if portfolio.index.freq != 'D':
            portfolio = portfolio.resample('D').ffill()
        
        # Normalize to percentage gain
        initial_capital = initial_capitals[strategy]
        normalized = portfolio['portfolio_value'] / initial_capital
        
        # Add to equity curves
        equity_curves[strategy_names[i]] = normalized
    
    # Fill missing values
    equity_curves = equity_curves.ffill()
    
    # Plot equity curves
    plt.figure(figsize=(12, 6))
    
    for strategy in equity_curves.columns:
        plt.plot(equity_curves.index, equity_curves[strategy], label=strategy)
    
    plt.title('Equity Curves (Normalized)')
    plt.xlabel('Date')
    plt.ylabel('Growth Multiple')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'equity_curves_comparison.png')
    plt.savefig(output_path)
    logger.info(f"Equity curves comparison saved to {output_path}")
    
    # Plot drawdowns
    plt.figure(figsize=(12, 6))
    
    for strategy in equity_curves.columns:
        # Calculate drawdowns
        rolling_max = equity_curves[strategy].cummax()
        drawdown = (equity_curves[strategy] / rolling_max) - 1
        
        plt.plot(equity_curves.index, drawdown * 100, label=strategy)
    
    plt.title('Drawdowns')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'drawdowns_comparison.png')
    plt.savefig(output_path)
    logger.info(f"Drawdowns comparison saved to {output_path}")

def compare_trade_stats(backtest_results: Dict[str, Dict], strategy_names: List[str], output_dir: str) -> None:
    """
    Compare trade statistics from multiple backtest results.
    
    Args:
        backtest_results: Dictionary with backtest results by strategy
        strategy_names: Names of strategies
        output_dir: Output directory for comparison charts
    """
    # Get trades data for each strategy
    all_trades = {}
    
    for strategy, results in backtest_results.items():
        if 'trades' in results and not results['trades'].empty and 'pnl' in results['trades'].columns:
            all_trades[strategy] = results['trades']
    
    if not all_trades:
        logger.error("No trades data found for comparison")
        return
    
    # Create trade distribution comparison
    plt.figure(figsize=(12, 6))
    
    for i, (strategy, trades) in enumerate(all_trades.items()):
        # Get PnL values
        pnl_values = trades['pnl'].dropna()
        
        # Plot histogram
        plt.hist(pnl_values, bins=30, alpha=0.5, label=strategy_names[i])
    
    plt.axvline(0, color='red', linestyle='dashed', linewidth=1)
    plt.title('Trade P&L Distribution')
    plt.xlabel('P&L')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'trade_distribution_comparison.png')
    plt.savefig(output_path)
    logger.info(f"Trade distribution comparison saved to {output_path}")
    
    # Calculate trade statistics
    trade_stats = []
    
    for i, (strategy, trades) in enumerate(all_trades.items()):
        # Get PnL values
        pnl_values = trades['pnl'].dropna()
        
        # Calculate statistics
        total_trades = len(pnl_values)
        winning_trades = len(pnl_values[pnl_values > 0])
        losing_trades = len(pnl_values[pnl_values <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = pnl_values[pnl_values > 0].mean() if winning_trades > 0 else 0
        avg_loss = pnl_values[pnl_values <= 0].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(pnl_values[pnl_values > 0].sum() / pnl_values[pnl_values <= 0].sum()) if losing_trades > 0 and pnl_values[pnl_values <= 0].sum() != 0 else 0
        
        # Add to statistics list
        trade_stats.append({
            'Strategy': strategy_names[i],
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Rate': win_rate * 100,
            'Avg Win': avg_win,
            'Avg Loss': avg_loss,
            'Profit Factor': profit_factor
        })
    
    # Create trade statistics DataFrame
    trade_stats_df = pd.DataFrame(trade_stats)
    trade_stats_df.set_index('Strategy', inplace=True)
    
    # Save trade statistics to CSV
    output_path = os.path.join(output_dir, 'trade_stats_comparison.csv')
    trade_stats_df.to_csv(output_path)
    logger.info(f"Trade statistics comparison saved to {output_path}")
    
    # Create trade statistics plot
    plt.figure(figsize=(12, 6))
    
    # Plot win rate
    ax = trade_stats_df['Win Rate'].plot(kind='bar', color='green', alpha=0.7)
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rate by Strategy')
    ax.grid(True, axis='y')
    
    # Save figure
    output_path = os.path.join(output_dir, 'win_rate_comparison.png')
    plt.savefig(output_path)
    logger.info(f"Win rate comparison saved to {output_path}")
    
    # Create profit factor plot
    plt.figure(figsize=(12, 6))
    
    # Plot profit factor
    ax = trade_stats_df['Profit Factor'].plot(kind='bar', color='blue', alpha=0.7)
    ax.set_ylabel('Profit Factor')
    ax.set_title('Profit Factor by Strategy')
    ax.grid(True, axis='y')
    
    # Save figure
    output_path = os.path.join(output_dir, 'profit_factor_comparison.png')
    plt.savefig(output_path)
    logger.info(f"Profit factor comparison saved to {output_path}")

def compare_monthly_returns(backtest_results: Dict[str, Dict], strategy_names: List[str], output_dir: str) -> None:
    """
    Compare monthly returns from multiple backtest results.
    
    Args:
        backtest_results: Dictionary with backtest results by strategy
        strategy_names: Names of strategies
        output_dir: Output directory for comparison charts
    """
    # Get portfolio data for each strategy
    portfolios = {}
    
    for strategy, results in backtest_results.items():
        if 'portfolio' in results and not results['portfolio'].empty:
            portfolios[strategy] = results['portfolio']
    
    if not portfolios:
        logger.error("No portfolio data found for comparison")
        return
    
    # Calculate monthly returns for each strategy
    monthly_returns = {}
    
    for i, (strategy, portfolio) in enumerate(portfolios.items()):
        # Calculate daily returns
        if 'daily_return' in portfolio.columns:
            daily_returns = portfolio['daily_return']
        else:
            daily_returns = portfolio['portfolio_value'].pct_change()
        
        # Resample to monthly returns
        strategy_monthly = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Store monthly returns
        monthly_returns[strategy_names[i]] = strategy_monthly
    
    # Create monthly returns DataFrame
    monthly_df = pd.DataFrame(monthly_returns)
    
    # Save monthly returns to CSV
    output_path = os.path.join(output_dir, 'monthly_returns_comparison.csv')
    monthly_df.to_csv(output_path)
    logger.info(f"Monthly returns comparison saved to {output_path}")
    
    # Create correlation matrix
    plt.figure(figsize=(10, 8))
    
    corr_matrix = monthly_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, linewidths=0.5)
    
    plt.title('Strategy Return Correlation Matrix')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(output_path)
    logger.info(f"Correlation matrix saved to {output_path}")
    
    # Create annual returns comparison
    annual_returns = {}
    
    for i, (strategy, portfolio) in enumerate(portfolios.items()):
        # Calculate daily returns
        if 'daily_return' in portfolio.columns:
            daily_returns = portfolio['daily_return']
        else:
            daily_returns = portfolio['portfolio_value'].pct_change()
        
        # Resample to annual returns
        strategy_annual = daily_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        
        # Store annual returns
        annual_returns[strategy_names[i]] = strategy_annual
    
    # Create annual returns DataFrame
    annual_df = pd.DataFrame(annual_returns)
    
    # Save annual returns to CSV
    output_path = os.path.join(output_dir, 'annual_returns_comparison.csv')
    annual_df.to_csv(output_path)
    logger.info(f"Annual returns comparison saved to {output_path}")
    
    # Create annual returns bar chart
    plt.figure(figsize=(12, 6))
    
    annual_df.plot(kind='bar', figsize=(12, 6))
    
    plt.title('Annual Returns by Strategy')
    plt.xlabel('Year')
    plt.ylabel('Return (%)')
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'annual_returns_comparison.png')
    plt.savefig(output_path)
    logger.info(f"Annual returns comparison saved to {output_path}")

def analyze_strategy_correlation(backtest_results: Dict[str, Dict], strategy_names: List[str], output_dir: str) -> None:
    """
    Analyze correlation between strategies and market regimes.
    
    Args:
        backtest_results: Dictionary with backtest results by strategy
        strategy_names: Names of strategies
        output_dir: Output directory for comparison charts
    """
    # Get portfolio data for each strategy
    portfolios = {}
    
    for strategy, results in backtest_results.items():
        if 'portfolio' in results and not results['portfolio'].empty:
            portfolios[strategy] = results['portfolio']
    
    if not portfolios:
        logger.error("No portfolio data found for correlation analysis")
        return
    
    # Calculate daily returns for each strategy
    daily_returns = {}
    
    for i, (strategy, portfolio) in enumerate(portfolios.items()):
        # Calculate daily returns
        if 'daily_return' in portfolio.columns:
            returns = portfolio['daily_return']
        else:
            returns = portfolio['portfolio_value'].pct_change()
        
        # Store daily returns
        daily_returns[strategy_names[i]] = returns
    
    # Create daily returns DataFrame
    returns_df = pd.DataFrame(daily_returns)
    
    # Calculate rolling correlations
    rolling_correlations = {}
    
    if len(strategy_names) > 1:
        # Calculate pairwise rolling correlations
        for i in range(len(strategy_names)):
            for j in range(i+1, len(strategy_names)):
                strategy_i = strategy_names[i]
                strategy_j = strategy_names[j]
                
                # Calculate 90-day rolling correlation
                rolling_corr = returns_df[strategy_i].rolling(90).corr(returns_df[strategy_j])
                
                # Store correlation
                rolling_correlations[f"{strategy_i} vs {strategy_j}"] = rolling_corr
    
        # Create rolling correlations DataFrame
        rolling_df = pd.DataFrame(rolling_correlations)
        
        # Save rolling correlations to CSV
        output_path = os.path.join(output_dir, 'rolling_correlations.csv')
        rolling_df.to_csv(output_path)
        logger.info(f"Rolling correlations saved to {output_path}")
        
        # Plot rolling correlations
        plt.figure(figsize=(12, 6))
        
        for column in rolling_df.columns:
            plt.plot(rolling_df.index, rolling_df[column], label=column)
        
        plt.title('90-Day Rolling Correlations Between Strategies')
        plt.xlabel('Date')
        plt.ylabel('Correlation')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, 'rolling_correlations.png')
        plt.savefig(output_path)
        logger.info(f"Rolling correlations plot saved to {output_path}")
    
    # Calculate drawdown periods
    drawdown_periods = {}
    
    for i, (strategy, portfolio) in enumerate(portfolios.items()):
        # Calculate portfolio value
        if 'portfolio_value' in portfolio.columns:
            value = portfolio['portfolio_value']
        else:
            # Skip if no portfolio value
            continue
        
        # Calculate drawdowns
        rolling_max = value.cummax()
        drawdowns = (value / rolling_max) - 1
        
        # Identify significant drawdown periods (> 10%)
        significant_drawdowns = drawdowns < -0.1
        
        # Store drawdown periods
        drawdown_periods[strategy_names[i]] = significant_drawdowns
    
    # Create drawdown periods DataFrame
    drawdown_df = pd.DataFrame(drawdown_periods)
    
    # Save drawdown periods to CSV
    output_path = os.path.join(output_dir, 'drawdown_periods.csv')
    drawdown_df.to_csv(output_path)
    logger.info(f"Drawdown periods saved to {output_path}")
    
    # Analyze strategy performance during different market regimes
    try:
        # Access first portfolio for market data
        first_portfolio = next(iter(portfolios.values()))
        
        if 'close' in first_portfolio.columns:
            # Calculate market trends (bull = up 20%, bear = down 20%)
            market_price = first_portfolio['close']
            rolling_min = market_price.rolling(90).min()
            rolling_max = market_price.rolling(90).max()
            
            bull_market = (market_price / rolling_min) - 1 > 0.2
            bear_market = (market_price / rolling_max) - 1 < -0.2
            
            # Neutral market is neither bull nor bear
            neutral_market = ~(bull_market | bear_market)
            
            # Create market regimes DataFrame
            regimes_df = pd.DataFrame({
                'Bull Market': bull_market,
                'Bear Market': bear_market,
                'Neutral Market': neutral_market
            })
            
            # Analyze strategy returns during different market regimes
            regime_performance = []
            
            for strategy_name, returns in daily_returns.items():
                # Calculate returns during different market regimes
                bull_returns = returns[bull_market].mean() * 252 * 100  # Annualized
                bear_returns = returns[bear_market].mean() * 252 * 100  # Annualized
                neutral_returns = returns[neutral_market].mean() * 252 * 100  # Annualized
                
                # Add to performance list
                regime_performance.append({
                    'Strategy': strategy_name,
                    'Bull Market Return (%)': bull_returns,
                    'Bear Market Return (%)': bear_returns,
                    'Neutral Market Return (%)': neutral_returns
                })
            
            # Create regime performance DataFrame
            regime_df = pd.DataFrame(regime_performance)
            regime_df.set_index('Strategy', inplace=True)
            
            # Save regime performance to CSV
            output_path = os.path.join(output_dir, 'market_regime_performance.csv')
            regime_df.to_csv(output_path)
            logger.info(f"Market regime performance saved to {output_path}")
            
            # Create regime performance bar chart
            plt.figure(figsize=(12, 6))
            
            regime_df.plot(kind='bar', figsize=(12, 6))
            
            plt.title('Strategy Performance by Market Regime')
            plt.xlabel('Strategy')
            plt.ylabel('Annualized Return (%)')
            plt.grid(True, axis='y')
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(output_dir, 'market_regime_performance.png')
            plt.savefig(output_path)
            logger.info(f"Market regime performance plot saved to {output_path}")
    except Exception as e:
        logger.warning(f"Could not analyze market regimes: {str(e)}")

def generate_full_report(comparison_results: Dict, output_dir: str) -> None:
    """
    Generate a full HTML report with comparison results.
    
    Args:
        comparison_results: Dictionary with comparison results
        output_dir: Output directory for report
    """
    # Create HTML report
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Strategy Comparison Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }
            h1, h2, h3 {
                color: #333;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            img {
                max-width: 100%;
                height: auto;
                margin: 10px 0;
            }
            .section {
                margin-bottom: 30px;
            }
        </style>
    </head>
    <body>
        <h1>Strategy Comparison Report</h1>
        <p>Generated on: %s</p>
        
        <div class="section">
            <h2>Performance Metrics</h2>
            %s
        </div>
        
        <div class="section">
            <h2>Equity Curves</h2>
            <img src="equity_curves_comparison.png" alt="Equity Curves Comparison">
            <img src="drawdowns_comparison.png" alt="Drawdowns Comparison">
        </div>
        
        <div class="section">
            <h2>Trade Statistics</h2>
            %s
            <img src="win_rate_comparison.png" alt="Win Rate Comparison">
            <img src="profit_factor_comparison.png" alt="Profit Factor Comparison">
            <img src="trade_distribution_comparison.png" alt="Trade Distribution Comparison">
        </div>
        
        <div class="section">
            <h2>Return Analysis</h2>
            <img src="annual_returns_comparison.png" alt="Annual Returns Comparison">
            <img src="correlation_matrix.png" alt="Correlation Matrix">
        </div>
        
        <div class="section">
            <h2>Market Regime Analysis</h2>
            <p>Performance across different market conditions:</p>
            %s
            <img src="market_regime_performance.png" alt="Market Regime Performance">
            <img src="rolling_correlations.png" alt="Rolling Correlations">
        </div>
    </body>
    </html>
    """ % (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        comparison_results['metrics'].to_html() if 'metrics' in comparison_results else 'No metrics data available',
        comparison_results['trade_stats'].to_html() if 'trade_stats' in comparison_results else 'No trade statistics available',
        comparison_results['regime_stats'].to_html() if 'regime_stats' in comparison_results else 'No market regime data available'
    )
    
    # Save HTML report
    output_path = os.path.join(output_dir, 'comparison_report.html')
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"Full report saved to {output_path}")

def main():
    """Main entry point for strategy comparison."""
    args = parse_args()
    
    # Check if directories exist
    for directory in args.dirs:
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return 1
    
    # Get strategy names
    if args.names and len(args.names) == len(args.dirs):
        strategy_names = args.names
    else:
        strategy_names = [os.path.basename(directory) for directory in args.dirs]
    
    logger.info(f"Comparing strategies: {', '.join(strategy_names)}")
    
    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Load backtest results
    backtest_results = {}
    
    for i, directory in enumerate(args.dirs):
        backtest_results[f"strategy_{i}"] = load_backtest_results(directory)
    
    # Compare metrics
    metrics_comparison = compare_metrics(backtest_results, strategy_names)
    
    if not metrics_comparison.empty:
        # Save metrics comparison to CSV
        output_path = os.path.join(output_dir, 'metrics_comparison.csv')
        metrics_comparison.to_csv(output_path)
        logger.info(f"Metrics comparison saved to {output_path}")
    
    # Compare equity curves
    compare_equity_curves(backtest_results, strategy_names, output_dir)
    
    # Compare trade statistics
    compare_trade_stats(backtest_results, strategy_names, output_dir)
    
    # Compare monthly returns
    compare_monthly_returns(backtest_results, strategy_names, output_dir)
    
    # Analyze correlation and market regimes
    analyze_strategy_correlation(backtest_results, strategy_names, output_dir)
    
    # Generate full report
    comparison_results = {
        'metrics': metrics_comparison,
        'trade_stats': pd.read_csv(os.path.join(output_dir, 'trade_stats_comparison.csv'), index_col=0) 
            if os.path.exists(os.path.join(output_dir, 'trade_stats_comparison.csv')) else None,
        'regime_stats': pd.read_csv(os.path.join(output_dir, 'market_regime_performance.csv'), index_col=0)
            if os.path.exists(os.path.join(output_dir, 'market_regime_performance.csv')) else None
    }
    
    generate_full_report(comparison_results, output_dir)
    
    logger.info(f"Comparison complete. Results saved to {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())