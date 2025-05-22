# Thor Trading Backtesting Framework

This directory contains a comprehensive backtesting framework for the Thor Trading system, designed to evaluate machine learning trading strategies using historical data.

## Overview

The backtesting framework allows you to:

1. Test ML-based trading strategies on historical market data
2. Evaluate trading performance with realistic simulation of slippage, commissions, etc.
3. Generate detailed performance metrics and visualizations
4. Compare multiple strategies and optimize parameters
5. Identify which strategies work best in different market regimes

## Components

### Backtesting Engine (`engine.py`)

The core backtesting engine implements a walk-forward testing methodology:

- **Walk-Forward Testing**: Train on historical data, then test on out-of-sample data, moving forward in time
- **Position Management**: Simulates entries, exits, stop losses, and take profits
- **Risk Management**: Position sizing and maximum position limits
- **Performance Calculation**: Metrics like Sharpe ratio, drawdown, win rate, and profit factor

### Strategy Comparison (`compare_strategies.py`)

A tool for comparing multiple backtest results:

- **Performance Comparison**: Side-by-side metrics comparison
- **Equity Curve Analysis**: Compare growth trajectories 
- **Correlation Analysis**: Identify diversification benefits
- **Market Regime Analysis**: See how strategies perform in bull vs. bear markets
- **HTML Report Generation**: Creates comprehensive visual reports

## Usage

### Running a Backtest

To run a basic backtest:

```bash
python backtesting/run_backtest.py --years 5 --train-days 365 --test-days 30
```

Parameters:
- `--start`: Start date (YYYY-MM-DD)
- `--end`: End date (YYYY-MM-DD)
- `--years`: Number of years to backtest (default: 5)
- `--initial-capital`: Initial capital (default: 100000)
- `--position-size`: Position size as percentage of capital (default: 0.02)
- `--train-days`: Number of days for training window (default: 365)
- `--test-days`: Number of days for testing window (default: 30)
- `--confidence`: Confidence threshold for actionable signals (default: 0.6)

### Comparing Strategies

To compare multiple backtest results:

```bash
python backtesting/compare_strategies.py --dirs backtest_results/test1 backtest_results/test2 --names "Strategy A" "Strategy B"
```

Parameters:
- `--dirs`: Directories with backtest results
- `--names`: Names for strategies (optional)
- `--output`: Output directory for comparison results (default: 'comparison_results')

## Interpreting Results

Backtest results include:

1. **Performance Metrics**:
   - Total and annualized returns
   - Maximum drawdown
   - Sharpe and Sortino ratios
   - Win rate and profit factor

2. **Trade Analysis**:
   - Trade count and profitability
   - Average win/loss sizes
   - Distribution of trade outcomes

3. **Visualizations**:
   - Equity curve
   - Drawdowns over time
   - Monthly returns heatmap
   - Trade P&L distribution

## Best Practices

For accurate backtesting:

1. **Use sufficient historical data**: At least 5 years to capture different market regimes
2. **Implement realistic assumptions**: Slippage, commissions, position sizing
3. **Avoid overfitting**: Use walk-forward validation to prevent curve-fitting
4. **Compare to benchmarks**: Always compare against a baseline like buy-and-hold
5. **Test multiple market regimes**: Evaluate performance in bull, bear, and sideways markets

## Example Workflow

1. Train ML models on historical data using different features/parameters
2. Backtest each model variation using `run_backtest.py`
3. Compare results with `compare_strategies.py`
4. Select the best performing models
5. Deploy to production trading system

## Interactive Dashboard

The Thor Trading Backtesting framework includes an interactive dashboard built with Streamlit for visualizing and analyzing backtest results.

### Running the Dashboard

```bash
python backtesting/run_dashboard.py
```

This will start a local web server, and you can access the dashboard at http://localhost:8501 in your web browser.

### Dashboard Features

- **Performance Metrics**: Key metrics like returns, Sharpe ratio, win rate, etc.
- **Equity Curve**: Interactive plot of portfolio value over time with drawdowns
- **Trade Analysis**: Visualizations of individual trades, P&L distribution, and statistics
- **Returns Analysis**: Monthly returns heatmap, rolling returns, and return distribution
- **Signal Analysis**: Trading signal visualization and performance by signal type
- **Run New Backtests**: Launch new backtests directly from the dashboard

![Dashboard Screenshot](https://via.placeholder.com/800x500?text=Thor+Trading+Dashboard)

## Directory Structure

- `engine.py`: Core backtesting engine
- `run_backtest.py`: Command-line script for running backtests
- `compare_strategies.py`: Tool for comparing multiple strategies
- `dashboard.py`: Interactive web dashboard for visualizing backtest results
- `run_dashboard.py`: Script to launch the dashboard
- `results/`: Default directory for backtest results
- `README.md`: This documentation file