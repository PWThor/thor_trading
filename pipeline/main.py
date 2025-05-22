import os
import sys
import logging
import argparse
import threading
from datetime import datetime
import time
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.postgres_connector import PostgresConnector
try:
    from connectors.ibkr_connector import IBKRConnector
except ImportError:
    # Use mock connector for testing if real connector not available
    from connectors.mock_ibkr_connector import IBKRConnector
from features.feature_generator_wrapper import create_feature_generator
from trading.execution_engine import ExecutionEngine
from pipeline.data_pipeline import DataPipeline
from pipeline.trading_system import TradingSystem
from models.training.model_trainer import ModelTrainer


def setup_logging(log_dir=None):
    """Set up logging for the application."""
    if log_dir is None:
        log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'logs'
        )
    
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'thor_trading_{datetime.now().strftime("%Y%m%d")}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create a logger
    logger = logging.getLogger('ThorTrading')
    
    return logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Thor Trading System')
    parser.add_argument('--mode', type=str, default='monitor', 
                      choices=['trade', 'monitor', 'backtest', 'train'],
                      help='Trading mode: trade (live trading), monitor (generate signals without trading), backtest, or train')
    parser.add_argument('--collect-data', action='store_true', 
                      help='Enable data collection')
    parser.add_argument('--api-keys', type=str, 
                      help='Path to API keys file (JSON format)')
    parser.add_argument('--max-positions', type=int, default=3,
                      help='Maximum number of concurrent positions')
    parser.add_argument('--risk-per-trade', type=float, default=0.02,
                      help='Risk per trade as a fraction of account equity')
    parser.add_argument('--confidence', type=float, default=0.7,
                      help='Minimum confidence threshold for trading')
    
    return parser.parse_args()


def load_api_keys(api_keys_path):
    """Load API keys from a JSON file."""
    import json
    
    try:
        with open(api_keys_path, 'r') as f:
            api_keys = json.load(f)
        return api_keys
    except Exception as e:
        print(f"Error loading API keys: {str(e)}")
        return {}


def data_collection_thread(data_pipeline):
    """Run data collection in a separate thread."""
    try:
        # Run initial data collection
        data_pipeline.run_daily_data_collection()
        
        # Start scheduler
        data_pipeline.run_scheduler()
    except Exception as e:
        print(f"Error in data collection thread: {str(e)}")
        traceback.print_exc()


def main():
    """Main entry point for the Thor Trading System."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info(f"Starting Thor Trading System in {args.mode} mode")
    
    # Load API keys if provided
    api_keys = {}
    if args.api_keys:
        # Convert path to use OS-specific separators
        api_key_path = os.path.normpath(args.api_keys)
        api_keys = load_api_keys(api_key_path)
    
    # Initialize database connector
    logger.info("Initializing database connector")
    db = PostgresConnector()
    
    # Initialize feature generator
    logger.info("Initializing feature generator")
    feature_gen = create_feature_generator()
    
    # Initialize IBKR connector if not in backtest or train mode
    ibkr = None
    if args.mode != 'backtest' and args.mode != 'train':
        logger.info("Initializing IBKR connector")
        ibkr = IBKRConnector()
    
    # Initialize execution engine if in trade or monitor mode
    execution_engine = None
    if args.mode in ['trade', 'monitor']:
        logger.info("Initializing execution engine")
        execution_engine = ExecutionEngine(ibkr)
    
    # Run data collection if enabled
    if args.collect_data:
        logger.info("Setting up data collection")
        data_pipeline = DataPipeline(
            db_connector=db,
            ibkr_connector=ibkr,
            weather_api_key=api_keys.get('openweather', None),
            eia_api_key=api_keys.get('eia', None)
        )
        
        # Schedule daily data collection
        data_pipeline.schedule_daily_collection("18:00")
        
        # Start data collection in a separate thread
        logger.info("Starting data collection thread")
        data_thread = threading.Thread(target=data_collection_thread, args=(data_pipeline,))
        data_thread.daemon = True
        data_thread.start()
    
    # Handle different modes
    if args.mode == 'train':
        logger.info("Training models")
        
        # Initialize model trainer
        trainer = ModelTrainer(
            db_connector=db,
            feature_generator=feature_gen
        )
        
        # Train models using 2 years of data
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2*365)
        
        # Train and save models
        model_paths = trainer.train_and_save_models(start_date, end_date)
        
        logger.info(f"Models trained and saved: {model_paths}")
        
    elif args.mode in ['trade', 'monitor']:
        logger.info(f"Starting trading system in {args.mode} mode")
        
        # Initialize trading system
        trading_system = TradingSystem(
            db_connector=db,
            ibkr_connector=ibkr,
            feature_generator=feature_gen,
            execution_engine=execution_engine,
            confidence_threshold=args.confidence,
            risk_per_trade=args.risk_per_trade,
            max_positions=args.max_positions
        )
        
        # Schedule trading cycles
        trading_system.schedule_trading_cycle("10:00")  # Morning cycle
        trading_system.schedule_trading_cycle("14:00")  # Afternoon cycle
        
        # Schedule position updates
        trading_system.schedule_position_updates(30)  # Every 30 minutes
        
        # Enable trading if in trade mode
        if args.mode == 'trade':
            logger.info("Enabling automated trading")
            trading_system.enable_trading()
        else:
            logger.info("Running in monitor mode (no trading)")
        
        # Start the system
        try:
            trading_system.start()
        except KeyboardInterrupt:
            logger.info("Trading system stopped by user")
        except Exception as e:
            logger.error(f"Trading system error: {str(e)}")
            logger.error(traceback.format_exc())
    
    elif args.mode == 'backtest':
        logger.info("Backtesting mode not implemented in this script")
        logger.info("Please use the backtesting module directly:")
        logger.info("python -m backtesting.engine --strategy ML --period 1y")
    
    logger.info("Thor Trading System shutting down")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting by user request.")
    except Exception as e:
        print(f"Unhandled error: {str(e)}")
        traceback.print_exc()