#!/usr/bin/env python
import os
import sys
import argparse
from datetime import datetime, timedelta
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.postgres_connector import PostgresConnector
from backtesting.engine import BacktestEngine

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
    parser = argparse.ArgumentParser(description='Run a backtest')
    
    # Date range
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--years', type=int, default=5, help='Number of years to backtest')
    
    # Instrument selection
    parser.add_argument('--symbols', type=str, nargs='+', 
                       default=['CL', 'HO', 'CL-HO-SPREAD'],
                       help='Symbols to backtest (defaults to CL, HO, and CL-HO-SPREAD)')
    
    # Backtest parameters
    parser.add_argument('--initial-capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--position-size', type=float, default=0.02, help='Position size as percentage of capital')
    parser.add_argument('--max-positions', type=int, default=3, help='Maximum number of concurrent positions')
    parser.add_argument('--commission', type=float, default=2.5, help='Commission per contract')
    parser.add_argument('--slippage', type=float, default=0.001, help='Slippage as percentage of price')
    
    # Walk-forward parameters
    parser.add_argument('--train-days', type=int, default=365, help='Number of days to use for training')
    parser.add_argument('--test-days', type=int, default=30, help='Number of days to use for testing')
    parser.add_argument('--confidence', type=float, default=0.6, help='Confidence threshold for actionable signals')
    parser.add_argument('--no-retrain', action='store_true', help='Disable retraining for each window')
    
    # Output
    parser.add_argument('--output-dir', type=str, help='Directory for backtest output')
    
    return parser.parse_args()

def main():
    """Main entry point for backtest."""
    args = parse_args()
    
    # Get date range
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    if args.end:
        try:
            end_date = datetime.strptime(args.end, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid end date format: {args.end}. Use YYYY-MM-DD.")
            return 1
    
    if args.start:
        try:
            start_date = datetime.strptime(args.start, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid start date format: {args.start}. Use YYYY-MM-DD.")
            return 1
    else:
        start_date = end_date - timedelta(days=365 * args.years)
    
    logger.info(f"Running backtest from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Initialize database connector
    db = PostgresConnector()
    
    # Use the enhanced backtest connector that supports all data sources
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from connectors.backtest_connector import BacktestDBConnector
    
    # Initialize our enhanced connector
    backtest_db = BacktestDBConnector()
    
    # Initialize backtest engine
    backtest = BacktestEngine(
        db_connector=backtest_db,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.initial_capital,
        position_size_pct=args.position_size,
        max_positions=args.max_positions,
        commission_per_contract=args.commission,
        slippage_pct=args.slippage,
        output_dir=args.output_dir
    )
    
    # Use the symbols specified in the command line arguments
    symbols = args.symbols
    logger.info(f"Backtesting with symbols: {symbols}")
    
    # Run walk-forward backtest
    success = backtest.run_walk_forward_backtest(
        symbols=symbols,
        train_days=args.train_days,
        test_days=args.test_days,
        confidence_threshold=args.confidence,
        retrain=not args.no_retrain
    )
    
    if not success:
        logger.error("Backtest failed")
        return 1
        
    # Generate visualizations
    vis_paths = backtest.visualize_results()
    
    if vis_paths:
        logger.info(f"Visualizations generated: {vis_paths}")
    else:
        logger.warning("Failed to generate visualizations")
    
    # Return performance metrics
    if hasattr(backtest, 'metrics'):
        logger.info("Performance metrics:")
        logger.info(f"Total Return: {backtest.metrics['total_return']*100:.2f}%")
        logger.info(f"Annualized Return: {backtest.metrics['annualized_return']*100:.2f}%")
        logger.info(f"Sharpe Ratio: {backtest.metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {backtest.metrics['max_drawdown']*100:.2f}%")
        logger.info(f"Win Rate: {backtest.metrics['win_rate']*100:.2f}%")
        logger.info(f"Profit Factor: {backtest.metrics['profit_factor']:.2f}")
        
    logger.info(f"Backtest results saved to {backtest.output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())