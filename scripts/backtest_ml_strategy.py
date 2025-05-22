# scripts/backtest_ml_strategy.py
"""
Backtest ML Trading Strategy

This script:
1. Loads historical crack spread data
2. Loads a trained ML model
3. Generates trading signals
4. Runs a backtest
5. Evaluates performance
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project components
from utils.config import load_config
from connectors.postgres_connector import PostgresConnector
from features.feature_generator import FeatureGenerator
from models.regression.xgboost_model import XGBoostRegressor
from trading.strategies.ml_strategy import MLTradingStrategy
from backtesting.engine import BacktestEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("backtest")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Backtest ML trading strategy')
    parser.add_argument('--config', type=str, default='config/default_config.yml',
                        help='Path to configuration file')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                        help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model file')
    parser.add_argument('--output', type=str, default='backtest_results',
                        help='Directory to save results')
    return parser.parse_args()

def load_data(config, start_date, end_date=None):
    """Load data from the database"""
    logger.info("Loading data from database")
    
    # Use current date if end_date not specified
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Connect to database
    db = PostgresConnector(config['database'])
    if not db.connect():
        logger.error("Failed to connect to database")
        return None
    
    try:
        # Get crack spread data
        crack_data = db.get_crack_spread_data(start_date, end_date)
        if crack_data is None or crack_data.empty:
            logger.error("No crack spread data found")
            return None
        
        logger.info(f"Loaded {len(crack_data)} rows of crack spread data")
        
        # Get weather data if enabled
        weather_data = None
        if config['features'].get('weather', {}).get('enabled', False):
            logger.info("Loading weather data")
            
            locations = config['features']['weather'].get('locations', ["New York", "Chicago"])
            weather_data = db.get_weather_data(start_date, end_date, locations)
            
            if weather_data is None or weather_data.empty:
                logger.warning("No weather data found, continuing without weather features")
            else:
                logger.info(f"Loaded {len(weather_data)} rows of weather data")
        
        # Get COT data if enabled
        cot_data = None
        if config['features'].get('cot', {}).get('enabled', False):
            logger.info("Loading COT data")
            
            symbol = config['features']['cot'].get('symbol', 'HO')
            cot_data = db.get_cot_data(start_date, end_date, symbol)
            
            if cot_data is None or cot_data.empty:
                logger.warning("No COT data found, continuing without COT features")
            else:
                logger.info(f"Loaded {len(cot_data)} rows of COT data")
        
        return {
            'crack_data': crack_data,
            'weather_data': weather_data,
            'cot_data': cot_data
        }
    
    finally:
        # Close database connection
        db.disconnect()

def generate_features(config, data, prediction_horizon=1):
    """Generate features from raw data"""
    logger.info("Generating features")
    
    # Initialize feature generator
    feature_generator = FeatureGenerator(config['features'])
    
    # Generate base features
    features_df = feature_generator.generate_features(
        data['crack_data'], 
        target_column='crack_spread',
        prediction_horizon=prediction_horizon
    )
    
    if features_df is None or features_df.empty:
        logger.error("Failed to generate features")
        return None
    
    # Add weather features if available
    if data['weather_data'] is not None and not data['weather_data'].empty:
        features_df = feature_generator.add_weather_features(features_df, data['weather_data'])
    
    # Add COT features if available
    if data['cot_data'] is not None and not data['cot_data'].empty:
        features_df = feature_generator.add_cot_features(features_df, data['cot_data'])
    
    logger.info(f"Generated features dataframe with shape: {features_df.shape}")
    
    return features_df

def load_model(model_path):
    """Load trained ML model"""
    logger.info(f"Loading model from {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    
    model = XGBoostRegressor()
    success = model.load(model_path)
    
    if success:
        logger.info("Model loaded successfully")
        return model
    else:
        logger.error("Failed to load model")
        return None

def run_backtest(features_df, model, config):
    """Run backtest with ML strategy"""
    logger.info("Setting up ML strategy and backtest")
    
    # Create ML strategy
    strategy = MLTradingStrategy(config['strategy'])
    
    # Set model
    strategy.set_model(model)
    
    # Generate signals
    signals_df = strategy.generate_signals(features_df)
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(config['backtest'])
    
    # Load data
    backtest_engine.load_data(signals_df)
    
    # Run backtest
    results = backtest_engine.run_backtest()
    
    # Plot results
    backtest_engine.plot_results(save_path=config['output_dir'])
    
    # Print summary
    summary = backtest_engine.get_summary()
    print(summary)
    
    # Save summary to file
    with open(os.path.join(config['output_dir'], 'backtest_summary.txt'), 'w') as f:
        f.write(summary)
    
    return results

def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_args()
    
    logger.info("Starting backtest")
    logger.info(f"Arguments: {args}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Add command line args to config
    config['start_date'] = args.start_date
    config['end_date'] = args.end_date
    config['model_path'] = args.model or config.get('model', {}).get('latest_model_path')
    config['output_dir'] = args.output
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load data
    data = load_data(config, config['start_date'], config['end_date'])
    if data is None:
        logger.error("Failed to load data, exiting")
        return
    
    # Generate features
    features_df = generate_features(config, data)
    if features_df is None:
        logger.error("Failed to generate features, exiting")
        return
    
    # Load model
    model = load_model(config['model_path'])
    if model is None:
        logger.error("Failed to load model, exiting")
        return
    
    # Run backtest
    results = run_backtest(features_df, model, config)
    
    logger.info("Backtest completed successfully")

if __name__ == "__main__":
    main()