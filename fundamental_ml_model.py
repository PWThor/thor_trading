#!/usr/bin/env python
"""
Fundamental ML Model for Thor Trading System

This script implements a data-driven machine learning model that prioritizes
fundamental data for trading decisions in energy markets.

Usage:
    python fundamental_ml_model.py --symbols=CL,HO --verbose
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
import sys
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
import logging

# Add thor_trading to the path
thor_trading_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(thor_trading_path)
print(f"Using thor_trading path: {thor_trading_path}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fundamental_ml_model')

# Suppress warnings
warnings.filterwarnings('ignore')

# Create necessary directories
for directory in ['connectors', 'models', 'results']:
    dir_path = os.path.join(thor_trading_path, directory)
    if not os.path.exists(dir_path):
        logger.info(f"Creating directory: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)

# Define MockPostgresConnector
class MockPostgresConnector:
    """Mock database connector for testing."""
    
    def __init__(self):
        logger.info("Using mock database connector for demonstration")
        self.data = {}  # Simple in-memory storage
        
    def test_connection(self):
        return True
        
    def query(self, query, params=None):
        """Mock query that returns sample data."""
        logger.debug(f"Mock query: {query}")
        # Return empty list by default, will be handled by synthetic data generation
        return []
    
    def query_one(self, query, params=None):
        logger.debug(f"Mock query_one: {query}")
        return None
        
    def execute(self, query, params=None):
        logger.debug(f"Mock execute: {query}")
        return True

# Try to import database connector
try:
    from connectors.postgres_connector import PostgresConnector
except ImportError as e:
    logger.warning(f"Error importing database connector: {str(e)}")
    logger.warning("Using mock database connector for demonstration.")
    # Create a simple postgres_connector.py file in the connectors directory
    connector_path = os.path.join(thor_trading_path, 'connectors')
    if not os.path.exists(connector_path):
        os.makedirs(connector_path, exist_ok=True)
    
    connector_file = os.path.join(connector_path, 'postgres_connector.py')
    if not os.path.exists(connector_file):
        with open(connector_file, 'w') as f:
            f.write("""
class PostgresConnector:
    \"\"\"Mock database connector for Thor Trading System.\"\"\"
    
    def __init__(self):
        print("Using mock database connector for demonstration")
        self.data = {}  # Simple in-memory storage
        
    def test_connection(self):
        return True
        
    def query(self, query, params=None):
        \"\"\"Mock query that returns sample data.\"\"\"
        print(f"Mock query: {query}")
        return []
    
    def query_one(self, query, params=None):
        print(f"Mock query_one: {query}")
        return None
        
    def execute(self, query, params=None):
        print(f"Mock execute: {query}")
        return True
""")
        logger.info(f"Created mock PostgresConnector in {connector_file}")
    
    # Use our MockPostgresConnector class
    PostgresConnector = MockPostgresConnector

class FundamentalMLModel:
    """
    A machine learning model that incorporates fundamental data for energy market trading.
    """
    
    def __init__(self, symbols, verbose=False):
        """
        Initialize the model with specified commodity symbols.
        
        Args:
            symbols (list): List of commodity symbols to analyze
            verbose (bool): Whether to display detailed output
        """
        self.symbols = symbols
        self.verbose = verbose
        self.models = {}
        self.scalers = {}
        self.performance = {}
        
        # Set up logging level based on verbosity
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Initializing Fundamental ML Model for symbols: {', '.join(symbols)}")
        
        # Initialize database connection
        try:
            # Directly create the connector with hardcoded working credentials
            logger.info("Connecting to PostgreSQL database")
            self.db = PostgresConnector(
                host="localhost",
                port="5432",
                dbname="trading_db",
                user="postgres",
                password="Makingmoney25!"
            )
                
            if not self.db.test_connection():
                logger.error("Could not connect to database")
                logger.warning("Using mock database connector for demonstration")
                self.db = MockPostgresConnector()
            else:
                logger.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            logger.warning("Using mock database connector for demonstration")
            self.db = MockPostgresConnector()
        
        # Create output directories if they don't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
    def load_data(self, symbol):
        """
        Load price and fundamental data for the specified symbol.
        
        Args:
            symbol (str): The commodity symbol to load data for
            
        Returns:
            pandas.DataFrame: Combined price and fundamental data
        """
        logger.info(f"Loading data for {symbol}")
        
        try:
            # First, check which fundamental data tables exist
            tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            """
            all_tables = self.db.query(tables_query)
            all_table_names = [table['table_name'] for table in all_tables]
            
            print(f"\nAll tables in database: {', '.join(all_table_names)}")
            
            # Now check for fundamental data tables
            fund_tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND 
                 table_name IN ('cot_data', 'eia_data', 'weather_data', 'opec_data', 'daily_market_data', 'market_data')
            """
            fund_tables = self.db.query(fund_tables_query)
            fund_table_names = [table['table_name'] for table in fund_tables]
            
            print(f"Fundamental data tables found: {', '.join(fund_table_names)}")
            
            # Check if we have separate tables for fundamental data
            has_cot_table = 'cot_data' in fund_table_names
            has_eia_table = 'eia_data' in fund_table_names
            has_weather_table = 'weather_data' in fund_table_names
            has_opec_table = 'opec_data' in fund_table_names
            
            print(f"COT data table: {'✓ Found' if has_cot_table else '✗ Missing'}")
            print(f"EIA data table: {'✓ Found' if has_eia_table else '✗ Missing'}")
            print(f"Weather data table: {'✓ Found' if has_weather_table else '✗ Missing'}")
            print(f"OPEC data table: {'✓ Found' if has_opec_table else '✗ Missing'}")
            
            # Option 1: Load from separate fundamental data tables if they exist
            if has_cot_table or has_eia_table or has_weather_table or has_opec_table:
                print(f"=== Processing {symbol} ===")
                print(f"Found cot_data table, loading from separate tables...")
                
                # First check if market_data table has the right columns
                market_col_query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'market_data'
                ORDER BY ordinal_position
                """
                market_cols = self.db.query(market_col_query)
                market_columns = [col['column_name'] for col in market_cols]
                print(f"Market data table columns: {', '.join(market_columns[:10])}...")
                
                # Check for records with this symbol
                symbol_check_query = f"""
                SELECT COUNT(*) as count 
                FROM market_data 
                WHERE symbol = '{symbol}'
                """
                count_result = self.db.query_one(symbol_check_query)
                if count_result and 'count' in count_result:
                    print(f"Market data records for {symbol}: {count_result['count']}")
                
                # Load price data
                price_query = f"""
                SELECT timestamp, symbol, open, high, low, close, volume
                FROM market_data
                WHERE symbol = '{symbol}'
                ORDER BY timestamp
                LIMIT 5
                """
                sample_price_data = self.db.query(price_query)
                
                if sample_price_data:
                    print(f"Sample price data for {symbol}:")
                    for row in sample_price_data[:2]:  # Show just 2 rows for brevity
                        print(f"  {row['timestamp']} - Open: {row['open']}, Close: {row['close']}")
                
                # Now get all price data
                price_query = f"""
                SELECT timestamp, symbol, open, high, low, close, volume
                FROM market_data
                WHERE symbol = '{symbol}'
                ORDER BY timestamp
                """
                price_data = self.db.query(price_query)
                
                if not price_data:
                    print(f"No price data found for {symbol}")
                    print(f"Skipping {symbol} due to missing data")
                    return None
                
                print(f"Loaded {len(price_data)} price records for {symbol}")
                
                # Load COT data - adjust symbol mapping if needed
                cot_symbol = symbol
                if symbol == 'CL':
                    # Check if COT data uses different symbol code for crude oil
                    cot_symbol_options = ['CL', 'CRUDE', '067651']
                elif symbol == 'HO':
                    # Check if COT data uses different symbol code for heating oil
                    cot_symbol_options = ['HO', 'HEAT', '022651']
                else:
                    cot_symbol_options = [symbol]
                
                # First get the column names in the cot_data table
                col_query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'cot_data'
                ORDER BY ordinal_position
                """
                cols = self.db.query(col_query)
                cot_columns = [col['column_name'] for col in cols]
                print(f"COT data table columns: {', '.join(cot_columns)}")
                
                # Get a sample of COT data to inspect
                sample_query = """
                SELECT * FROM cot_data LIMIT 1
                """
                sample = self.db.query(sample_query)
                if sample:
                    print("Sample COT data record:")
                    for key, value in sample[0].items():
                        print(f"  {key}: {value}")
                
                # Try each possible symbol to find matching COT data
                cot_data = None
                cot_symbol_used = None
                
                for cot_sym in cot_symbol_options:
                    # Build query based on actual columns in the table
                    print(f"Trying to find COT data for symbol: {cot_sym}")
                    cot_query = f"""
                    SELECT *
                    FROM cot_data
                    WHERE symbol = '{cot_sym}'
                    ORDER BY report_date
                    LIMIT 5
                    """
                    test_data = self.db.query(cot_query)
                    if test_data:
                        print(f"Found {len(test_data)} records for {cot_sym}")
                        cot_data = test_data
                        cot_symbol_used = cot_sym
                        break
                    else:
                        print(f"No COT data found for symbol {cot_sym}")
                
                # Now get all the data with the correct symbol
                if cot_symbol_used:
                    # Use the actual column names from your database (with typos)
                    cot_query = f"""
                    SELECT id, report_date, symbol, 
                           comm_positions_long, 
                           comm_postions_short, 
                           noncomm_postionns_long, 
                           noncomm_postions_short, 
                           net_positions
                    FROM cot_data
                    WHERE symbol = '{cot_symbol_used}'
                    ORDER BY report_date
                    """
                    cot_data = self.db.query(cot_query)
                
                if not cot_data:
                    print(f"No COT data found for {symbol}")
                    print(f"Skipping {symbol} due to missing fundamental data")
                    return None
                
                print(f"Found COT data using symbol: {cot_symbol_used}")
                
                # Convert database results to DataFrames
                df_price = pd.DataFrame(price_data)
                df_price.set_index('timestamp', inplace=True)
                
                df_cot = pd.DataFrame(cot_data)
                
                # In your cot_data table, the date column is named 'report_date' not 'timestamp'
                if 'report_date' in df_cot.columns and 'timestamp' not in df_cot.columns:
                    df_cot = df_cot.rename(columns={'report_date': 'timestamp'})
                
                # Set the timestamp as index
                if 'timestamp' in df_cot.columns:
                    df_cot.set_index('timestamp', inplace=True)
                
                # Use the actual column names from your cot_data table
                # You mentioned: id, report_date, symbol, comm_positions_long, comm_postions_short, 
                # noncomm_postionns_long, noncomm_postions_short, net_positions
                
                # Check if the required columns exist in df_cot using the actual column names
                # This list should match the columns we selected in the SQL query
                required_columns = ['comm_positions_long', 'comm_postions_short', 
                                   'noncomm_postionns_long', 'noncomm_postions_short', 'net_positions']
                
                # Handle typos in your database column names (positions is misspelled as postions)
                missing_columns = []
                for col in required_columns:
                    if col not in df_cot.columns and col.replace('postions', 'positions') not in df_cot.columns:
                        missing_columns.append(col)
                
                if missing_columns:
                    print(f"COT data table is missing required columns: {', '.join(missing_columns)}")
                    print(f"Available columns: {', '.join(df_cot.columns)}")
                    print(f"Skipping {symbol} due to missing fundamental data columns")
                    return None
                
                # Rename COT columns to match our processing code, handling the typos
                column_mapping = {
                    'comm_positions_long': 'Commercial_Long',
                    'comm_postions_long': 'Commercial_Long',  # Handle typo
                    'comm_positions_short': 'Commercial_Short',
                    'comm_postions_short': 'Commercial_Short',  # Handle typo
                    'noncomm_positions_long': 'NonCommercial_Long',
                    'noncomm_postionns_long': 'NonCommercial_Long',  # Handle typo
                    'noncomm_positions_short': 'NonCommercial_Short',
                    'noncomm_postions_short': 'NonCommercial_Short',  # Handle typo
                    'noncomm_postionns_short': 'NonCommercial_Short'  # Handle typo
                }
                
                # Only include keys that actually exist in the dataframe
                column_mapping = {k: v for k, v in column_mapping.items() if k in df_cot.columns}
                df_cot.rename(columns=column_mapping, inplace=True)
                
                # Calculate net positions if they're not already there
                if 'Commercial_Net' not in df_cot.columns:
                    if 'Commercial_Long' in df_cot.columns and 'Commercial_Short' in df_cot.columns:
                        df_cot['Commercial_Net'] = df_cot['Commercial_Long'] - df_cot['Commercial_Short']
                    elif 'net_positions' in df_cot.columns:
                        df_cot['Commercial_Net'] = df_cot['net_positions']
                
                if 'NonCommercial_Net' not in df_cot.columns and 'NonCommercial_Long' in df_cot.columns and 'NonCommercial_Short' in df_cot.columns:
                    df_cot['NonCommercial_Net'] = df_cot['NonCommercial_Long'] - df_cot['NonCommercial_Short']
                
                # Initialize combined fundamental dataframe with COT data
                fundamental_data = df_cot.copy() if not df_cot.empty else None
                
                # Load EIA data if available
                if has_eia_table:
                    try:
                        eia_symbol = symbol
                        if symbol == 'CL':
                            # EIA series IDs for crude oil
                            eia_series_options = ['PET.WCRFPUS2.W', 'PET.WCRSTUS1.W', 'STEO.PASC_NA.M']
                        elif symbol == 'HO':
                            # EIA series IDs for heating oil
                            eia_series_options = ['PET.WDISTUS1.W', 'PET.WDIUPUS2.W', 'STEO.PATC_NA.M']
                        else:
                            eia_series_options = []
                            
                        if eia_series_options:
                            eia_data = None
                            for series_id in eia_series_options:
                                eia_query = f"""
                                SELECT period as timestamp, value, unit
                                FROM eia_data
                                WHERE series_id = '{series_id}'
                                ORDER BY period
                                """
                                test_data = self.db.query(eia_query)
                                if test_data:
                                    print(f"Found EIA data using series ID: {series_id}")
                                    eia_df = pd.DataFrame(test_data)
                                    eia_df.set_index('timestamp', inplace=True)
                                    eia_df.rename(columns={'value': f'EIA_{series_id.replace(".", "_")}'}, inplace=True)
                                    
                                    # Merge with existing fundamental data if exists
                                    if fundamental_data is not None:
                                        fundamental_data = pd.merge(
                                            fundamental_data, eia_df, 
                                            left_index=True, right_index=True, 
                                            how='outer'
                                        )
                                    else:
                                        fundamental_data = eia_df
                    except Exception as e:
                        print(f"Error loading EIA data: {e}")
                
                # Load weather data if available
                if has_weather_table:
                    try:
                        # Default weather locations by symbol
                        if symbol == 'CL':
                            # Use Cushing, OK or NY as proxy
                            weather_locations = ['CUSHING', 'KNYC', 'NY']
                        elif symbol == 'HO':
                            # Use Northeast cities as proxy for heating oil 
                            weather_locations = ['KNYC', 'KBOS', 'NY', 'BOS']
                        else:
                            weather_locations = []
                            
                        if weather_locations:
                            for location in weather_locations:
                                weather_query = f"""
                                SELECT timestamp, avg_temperature, precipitation, snowfall, 
                                       wind_speed, humidity, cloud_coverage
                                FROM weather_data
                                WHERE location = '{location}'
                                ORDER BY timestamp
                                """
                                test_data = self.db.query(weather_query)
                                if test_data:
                                    print(f"Found weather data for location: {location}")
                                    weather_df = pd.DataFrame(test_data)
                                    weather_df.set_index('timestamp', inplace=True)
                                    
                                    # Rename columns to include location
                                    weather_df = weather_df.add_prefix(f'Weather_{location}_')
                                    
                                    # Merge with existing fundamental data if exists
                                    if fundamental_data is not None:
                                        fundamental_data = pd.merge(
                                            fundamental_data, weather_df, 
                                            left_index=True, right_index=True, 
                                            how='outer'
                                        )
                                    else:
                                        fundamental_data = weather_df
                                    break  # Use first available location
                    except Exception as e:
                        print(f"Error loading weather data: {e}")
                
                # Load OPEC data if available
                if has_opec_table and symbol == 'CL':  # OPEC data only relevant for crude oil
                    try:
                        opec_metrics = ['production', 'quota', 'compliance']
                        for metric in opec_metrics:
                            opec_query = f"""
                            SELECT report_date as timestamp, value
                            FROM opec_data
                            WHERE metric = '{metric}'
                            ORDER BY report_date
                            """
                            test_data = self.db.query(opec_query)
                            if test_data:
                                print(f"Found OPEC {metric} data")
                                opec_df = pd.DataFrame(test_data)
                                opec_df.set_index('timestamp', inplace=True)
                                opec_df.rename(columns={'value': f'OPEC_{metric}'}, inplace=True)
                                
                                # Merge with existing fundamental data if exists
                                if fundamental_data is not None:
                                    fundamental_data = pd.merge(
                                        fundamental_data, opec_df, 
                                        left_index=True, right_index=True, 
                                        how='outer'
                                    )
                                else:
                                    fundamental_data = opec_df
                    except Exception as e:
                        print(f"Error loading OPEC data: {e}")
                
                # If we couldn't find any fundamental data, return None
                if fundamental_data is None:
                    print(f"No fundamental data found for {symbol}")
                    return None
                
                # Merge price data with all fundamental data
                data = pd.merge(df_price, fundamental_data, left_index=True, right_index=True, how='left')
                
                # Forward fill fundamental data (since it's published less frequently)
                data = data.fillna(method='ffill')
                
                # Create return and direction columns
                data['Returns'] = data['close'].pct_change()
                data['Direction'] = np.where(data['Returns'] > 0, 1, 0)
                
                # Drop any remaining NaN values
                data = data.dropna()
                
                if len(data) < 50:
                    print(f"Insufficient data after merging price and fundamental data")
                    print(f"Skipping {symbol}")
                    return None
                
                print(f"Successfully loaded {len(data)} rows with fundamental data")
                
                # Print the first few rows to show what data was loaded
                print("\nSample of combined data:")
                print(data.head(3))
                
                return data
                
            # Option 2: Check if fundamental data exists in market_data table
            check_query = f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'market_data' AND column_name LIKE '%commercial%'
            """
            cot_columns = self.db.query(check_query)
            
            if not cot_columns:
                print(f"=== Processing {symbol} ===")
                print(f"Loading data with fundamental indicators...")
                print(f"No COT data columns found in market_data table")
                print(f"Skipping {symbol} due to missing fundamental data")
                return None
                
            # Check if data exists for this symbol
            columns_list = [col['column_name'] for col in cot_columns]
            check_column = columns_list[0] if columns_list else 'cot_commercial_long'
            
            check_query = f"""
            SELECT COUNT(*) as count
            FROM market_data 
            WHERE symbol = '{symbol}' AND {check_column} IS NOT NULL
            """
            data_check = self.db.query_one(check_query)
            
            if not data_check or data_check['count'] == 0:
                print(f"=== Processing {symbol} ===")
                print(f"Loading data with fundamental indicators...")
                print(f"No fundamental data found for {symbol}")
                print(f"Skipping {symbol} due to missing data")
                return None
                
            # Load price and fundamental data from database
            query = f"""
            SELECT 
                timestamp, symbol, open, high, low, close, volume,
                cot_commercial_long, cot_commercial_short, cot_commercial_net,
                cot_noncommercial_long, cot_noncommercial_short, cot_noncommercial_net,
                eia_value, temperature, humidity, precipitation,
                opec_production, opec_quota, opec_compliance
            FROM market_data
            WHERE symbol = '{symbol}'
            ORDER BY timestamp
            """
            
            data = self.db.query(query)
            
            if not data:
                logger.error(f"No data found for {symbol}")
                print(f"=== Processing {symbol} ===")
                print(f"Loading data with fundamental indicators...")
                print(f"No data found for {symbol}")
                print(f"Skipping {symbol} due to missing data")
                return None
            else:
                # Convert database result to DataFrame
                df = pd.DataFrame(data)
                
                # Set timestamp as index
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
                
                # Rename columns to match our processing code
                column_mapping = {
                    'cot_commercial_long': 'Commercial_Long',
                    'cot_commercial_short': 'Commercial_Short',
                    'cot_commercial_net': 'Commercial_Net',
                    'cot_noncommercial_long': 'NonCommercial_Long',
                    'cot_noncommercial_short': 'NonCommercial_Short',
                    'cot_noncommercial_net': 'NonCommercial_Net',
                    'eia_value': 'Inventory_Level',
                    'opec_production': 'OPEC_Production',
                    'opec_quota': 'OPEC_Quota',
                    'opec_compliance': 'OPEC_Compliance'
                }
                
                df.rename(columns=column_mapping, inplace=True)
                
                # Calculate inventory change (weekly)
                if 'Inventory_Level' in df.columns:
                    df['Inventory_Change'] = df['Inventory_Level'].diff(7)  # Weekly change
                
                data = df
            
            # Forward fill missing values (fundamentals are reported less frequently)
            data = data.fillna(method='ffill')
            
            # Create return and direction columns
            data['Returns'] = data['close'].pct_change()
            data['Direction'] = np.where(data['Returns'] > 0, 1, 0)
            
            # Drop remaining NaN values
            data = data.dropna()
            
            logger.debug(f"Loaded {len(data)} rows of data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            raise
    
    # All synthetic data generation removed to ensure we only use real data
    
    def engineer_features(self, data):
        """
        Engineer features from the raw data.
        
        Args:
            data (pandas.DataFrame): Raw price and fundamental data
            
        Returns:
            pandas.DataFrame: Feature-engineered data
        """
        logger.info("Engineering features from raw data")
        
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Price-based features
        logger.debug("Creating price-based features")
        for window in [5, 10, 20, 50]:
            # Moving averages
            df[f'MA_{window}'] = df['close'].rolling(window=window).mean()
            
            # Price relative to moving average
            df[f'Close_to_MA_{window}'] = df['close'] / df[f'MA_{window}'] - 1
            
            # Volatility (standard deviation of returns)
            df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Additional price-based momentum indicators
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = df['close'].pct_change(periods=period) * 100
            
        # Bollinger Bands
        for window in [20]:
            rolling_mean = df['close'].rolling(window=window).mean()
            rolling_std = df['close'].rolling(window=window).std()
            df[f'BB_Upper_{window}'] = rolling_mean + (rolling_std * 2)
            df[f'BB_Lower_{window}'] = rolling_mean - (rolling_std * 2)
            df[f'BB_Width_{window}'] = (df[f'BB_Upper_{window}'] - df[f'BB_Lower_{window}']) / rolling_mean
            df[f'BB_Position_{window}'] = (df['close'] - df[f'BB_Lower_{window}']) / (df[f'BB_Upper_{window}'] - df[f'BB_Lower_{window}'])
        
        # COT-based features (if available)
        if 'Commercial_Net' in df.columns:
            logger.debug("Creating COT-based features")
            
            # Normalize net positions
            for col in ['Commercial_Net', 'NonCommercial_Net']:
                # Z-score over the past year
                df[f'{col}_Z'] = (df[col] - df[col].rolling(window=52).mean()) / df[col].rolling(window=52).std()
                
                # Percentile rank over the past year (0-100)
                df[f'{col}_Rank'] = df[col].rolling(window=52).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
                )
                
                # Rate of change
                df[f'{col}_RoC'] = df[col].pct_change(periods=4)  # 4-week rate of change
            
            # Commercial vs speculator positioning
            df['Comm_NonComm_Ratio'] = df['Commercial_Net'] / df['NonCommercial_Net'].replace(0, np.nan)
            
            # Extreme positioning indicators
            df['Commercial_Extreme_Long'] = (df['Commercial_Net_Rank'] > 80).astype(int)
            df['Commercial_Extreme_Short'] = (df['Commercial_Net_Rank'] < 20).astype(int)
            df['NonCommercial_Extreme_Long'] = (df['NonCommercial_Net_Rank'] > 80).astype(int)
            df['NonCommercial_Extreme_Short'] = (df['NonCommercial_Net_Rank'] < 20).astype(int)
            
            # Commercials vs price divergence (when commercials are buying but price is falling or vice versa)
            df['Comm_Price_Divergence'] = ((df['Commercial_Net'].diff(4) > 0) & (df['close'].diff(20) < 0)) | \
                                          ((df['Commercial_Net'].diff(4) < 0) & (df['close'].diff(20) > 0))
            df['Comm_Price_Divergence'] = df['Comm_Price_Divergence'].astype(int)
        
        # Inventory-based features (if available)
        if 'Inventory_Level' in df.columns:
            logger.debug("Creating inventory-based features")
            
            # Z-score of inventory versus seasonal norm
            if 'Inventory_vs_Seasonal' in df.columns:
                df['Inventory_vs_Seasonal_Z'] = (df['Inventory_vs_Seasonal'] - 
                                               df['Inventory_vs_Seasonal'].rolling(window=52).mean()) / \
                                               df['Inventory_vs_Seasonal'].rolling(window=52).std()
                
                # Percentile rank of inventory versus seasonal norm
                df['Inventory_vs_Seasonal_Rank'] = df['Inventory_vs_Seasonal'].rolling(window=52).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
                )
            
            # Inventory change momentum
            df['Inventory_Change_4W'] = df['Inventory_Level'].diff(4)
            
            # Extreme inventory indicators
            if 'Inventory_vs_Seasonal_Rank' in df.columns:
                df['Low_Inventory'] = (df['Inventory_vs_Seasonal_Rank'] < 20).astype(int)
                df['High_Inventory'] = (df['Inventory_vs_Seasonal_Rank'] > 80).astype(int)
            
            # Inventory vs price divergence
            if 'Inventory_Change' in df.columns:
                df['Inventory_Price_Divergence'] = ((df['Inventory_Change'].diff(4) > 0) & (df['close'].diff(20) > 0)) | \
                                                  ((df['Inventory_Change'].diff(4) < 0) & (df['close'].diff(20) < 0))
                df['Inventory_Price_Divergence'] = df['Inventory_Price_Divergence'].astype(int)
        
        # Weather-based features (if available)
        if 'Temperature' in df.columns:
            logger.debug("Creating weather-based features")
            
            # Moving averages for temperature
            for window in [7, 14, 30]:
                df[f'Temp_MA_{window}'] = df['Temperature'].rolling(window=window).mean()
                if 'HDDs' in df.columns:
                    df[f'HDDs_Sum_{window}'] = df['HDDs'].rolling(window=window).sum()
                if 'CDDs' in df.columns:
                    df[f'CDDs_Sum_{window}'] = df['CDDs'].rolling(window=window).sum()
            
            # Temperature anomaly (deviation from 30-day average)
            if 'Temp_MA_30' in df.columns:
                df['Temp_Anomaly'] = df['Temperature'] - df['Temp_MA_30']
            
            # Weather vs price relationship
            if 'HDDs_Sum_14' in df.columns:  # Relevant for heating oil
                # Heating oil prices typically rise with increased heating demand
                df['Weather_Price_Alignment'] = ((df['HDDs_Sum_14'].diff() > 0) & (df['close'].diff(5) > 0)) | \
                                             ((df['HDDs_Sum_14'].diff() < 0) & (df['close'].diff(5) < 0))
                df['Weather_Price_Alignment'] = df['Weather_Price_Alignment'].astype(int)
        
        # OPEC-based features (if available)
        if 'OPEC_Production' in df.columns:
            logger.debug("Creating OPEC-based features")
            
            # Calculate base features if we have the data
            if 'OPEC_Quota' in df.columns:
                # Calculate overproduction/underproduction
                df['OPEC_Over_Under'] = df['OPEC_Production'] - df['OPEC_Quota']
                df['OPEC_Over_Under_Pct'] = df['OPEC_Over_Under'] / df['OPEC_Quota'] * 100  # As percentage of quota
                
                # Z-score of over/under production
                df['OPEC_Over_Under_Z'] = (df['OPEC_Over_Under'] - 
                                         df['OPEC_Over_Under'].rolling(window=12).mean()) / \
                                         df['OPEC_Over_Under'].rolling(window=12).std()
                
                # Moving averages of over/under production
                for window in [3, 6, 12]:
                    df[f'OPEC_Over_Under_MA_{window}'] = df['OPEC_Over_Under'].rolling(window=window).mean()
                
                # Rate of change in compliance
                if 'OPEC_Compliance' in df.columns:
                    df['OPEC_Compliance_Change'] = df['OPEC_Compliance'].pct_change(periods=3)  # 3-month change
                    
                    # Compliance trend - positive when increasing compliance
                    df['OPEC_Compliance_Trend'] = df['OPEC_Compliance'].diff(3).rolling(window=3).mean()
                    
                    # Extreme compliance indicators
                    df['OPEC_High_Compliance'] = (df['OPEC_Compliance'] > 0.95).astype(int)  # >95% compliance
                    df['OPEC_Low_Compliance'] = (df['OPEC_Compliance'] < 0.8).astype(int)   # <80% compliance
                    
                    # Compliance regimes (categorical):
                    # 0: Compliance decreasing, negative trend
                    # 1: Compliance decreasing, but trend improving
                    # 2: Compliance increasing, positive trend
                    # 3: Compliance high and stable
                    df['OPEC_Compliance_Regime'] = 0  # Default
                    df.loc[(df['OPEC_Compliance'].diff() < 0) & (df['OPEC_Compliance_Trend'] > 0), 'OPEC_Compliance_Regime'] = 1
                    df.loc[df['OPEC_Compliance'].diff() > 0, 'OPEC_Compliance_Regime'] = 2
                    df.loc[df['OPEC_High_Compliance'] & (abs(df['OPEC_Compliance'].diff()) < 0.02), 'OPEC_Compliance_Regime'] = 3
            
            # Production momentum
            df['OPEC_Production_Mom'] = df['OPEC_Production'].pct_change(periods=3)  # 3-month momentum
            
            # Longer-term production trends
            df['OPEC_Production_6M_Change'] = df['OPEC_Production'].pct_change(periods=6) * 100
            df['OPEC_Production_12M_Change'] = df['OPEC_Production'].pct_change(periods=12) * 100
            
            # OPEC vs price relationship
            df['OPEC_Price_Alignment'] = ((df['OPEC_Production'].diff(3) < 0) & (df['close'].diff(20) > 0)) | \
                                       ((df['OPEC_Production'].diff(3) > 0) & (df['close'].diff(20) < 0))
            df['OPEC_Price_Alignment'] = df['OPEC_Price_Alignment'].astype(int)
            
            # OPEC cuts - binary indicator of production cuts
            df['OPEC_Cut_3M'] = (df['OPEC_Production'].diff(3) < -0.5).astype(int)  # Production cut of >0.5m bpd
            df['OPEC_Large_Cut'] = (df['OPEC_Production'].diff(3) < -1.0).astype(int)  # Large cut of >1m bpd
            
        # Combined fundamental vs price features
        if 'Commercial_Net' in df.columns and 'Inventory_Level' in df.columns:
            logger.debug("Creating combined fundamental vs price features")
            
            # Create a composite fundamental indicator if we have z-scores
            if 'Commercial_Net_Z' in df.columns and 'Inventory_vs_Seasonal_Z' in df.columns:
                # For crude oil, bullish = commercial longs + low inventory
                df['Composite_Fundamental'] = df['Commercial_Net_Z'] - df['Inventory_vs_Seasonal_Z']
                
                # Add OPEC component if available
                if 'OPEC_Over_Under_Z' in df.columns:
                    # Subtract OPEC overproduction (negative is bullish, positive is bearish)
                    df['Composite_Fundamental'] = df['Composite_Fundamental'] - df['OPEC_Over_Under_Z']
                    
                    # Enhanced composite with OPEC compliance
                    if 'OPEC_Compliance' in df.columns:
                        # Add bonus for high compliance
                        compliance_factor = (df['OPEC_Compliance'] - 0.8) * 2  # Scale 0.8-1.0 to 0.0-0.4
                        df['Composite_Fundamental_Enhanced'] = df['Composite_Fundamental'] + compliance_factor
                
                # Fundamental vs price divergence
                df['Fund_Price_Divergence'] = ((df['Composite_Fundamental'].diff(4) > 0) & (df['close'].diff(20) < 0)) | \
                                             ((df['Composite_Fundamental'].diff(4) < 0) & (df['close'].diff(20) > 0))
                df['Fund_Price_Divergence'] = df['Fund_Price_Divergence'].astype(int)
                
                # Fundamental extremes - strong buy/sell signals
                df['Fund_Strong_Buy'] = (df['Composite_Fundamental'] > 2.0).astype(int)
                df['Fund_Strong_Sell'] = (df['Composite_Fundamental'] < -2.0).astype(int)
                
                # Create a categorical market regime indicator based on fundamentals
                # 0: Neutral
                # 1: Bullish fundamentals
                # 2: Very bullish fundamentals
                # 3: Bearish fundamentals
                # 4: Very bearish fundamentals
                df['Fundamental_Regime'] = 0  # Default neutral
                df.loc[df['Composite_Fundamental'] > 1.0, 'Fundamental_Regime'] = 1  # Bullish
                df.loc[df['Composite_Fundamental'] > 2.0, 'Fundamental_Regime'] = 2  # Very bullish
                df.loc[df['Composite_Fundamental'] < -1.0, 'Fundamental_Regime'] = 3  # Bearish
                df.loc[df['Composite_Fundamental'] < -2.0, 'Fundamental_Regime'] = 4  # Very bearish
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    def prepare_training_data(self, data):
        """
        Prepare the feature-engineered data for training.
        
        Args:
            data (pandas.DataFrame): Feature-engineered data
            
        Returns:
            tuple: X_price, X_fund, y_direction, y_returns
        """
        logger.info("Preparing training data")
        
        # Make a copy
        df = data.copy()
        
        # Target variables
        y_direction = df['Direction']
        y_returns = df['Returns']
        
        # Drop variables that shouldn't be used as features
        cols_to_drop = ['open', 'high', 'low', 'close', 'volume', 'Returns', 'Direction']
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # Separate price-based and fundamental-based features
        price_features = [col for col in df.columns if any(x in col for x in ['MA_', 'Volatility_', 'RSI', 'Close_to_MA_'])]
        fund_features = [col for col in df.columns if col not in price_features]
        
        logger.debug(f"Price features: {len(price_features)}")
        logger.debug(f"Fundamental features: {len(fund_features)}")
        
        # Create separate datasets
        X_price = df[price_features]
        X_fund = df[fund_features] if fund_features else None
        
        return X_price, X_fund, y_direction, y_returns
    
    def train_models(self, symbol, X_price, X_fund, y_direction, y_returns):
        """
        Train the machine learning models.
        
        Args:
            symbol (str): The commodity symbol
            X_price (pandas.DataFrame): Price-based features
            X_fund (pandas.DataFrame): Fundamental-based features
            y_direction (pandas.Series): Direction target (0/1)
            y_returns (pandas.Series): Returns target (float)
            
        Returns:
            dict: Trained models and scalers
        """
        logger.info(f"Training models for {symbol}")
        
        # Initialize dictionary to store models
        models = {}
        scalers = {}
        
        # Create time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        split = list(tscv.split(X_price))
        train_idx, test_idx = split[-1]  # Use the last split
        
        # Split data
        X_price_train, X_price_test = X_price.iloc[train_idx], X_price.iloc[test_idx]
        
        if X_fund is not None:
            X_fund_train, X_fund_test = X_fund.iloc[train_idx], X_fund.iloc[test_idx]
        else:
            X_fund_train, X_fund_test = None, None
            
        y_direction_train, y_direction_test = y_direction.iloc[train_idx], y_direction.iloc[test_idx]
        y_returns_train, y_returns_test = y_returns.iloc[train_idx], y_returns.iloc[test_idx]
        
        # Scale the features
        scaler_price = StandardScaler()
        X_price_train_scaled = scaler_price.fit_transform(X_price_train)
        X_price_test_scaled = scaler_price.transform(X_price_test)
        
        scalers['price'] = scaler_price
        
        if X_fund is not None:
            scaler_fund = StandardScaler()
            X_fund_train_scaled = scaler_fund.fit_transform(X_fund_train)
            X_fund_test_scaled = scaler_fund.transform(X_fund_test)
            scalers['fund'] = scaler_fund
        
        # 1. Random Forest for direction prediction
        logger.debug("Training Random Forest model for direction prediction")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        if X_fund is not None:
            # Combine price and fundamental features
            X_combined_train = np.hstack((X_price_train_scaled, X_fund_train_scaled))
            X_combined_test = np.hstack((X_price_test_scaled, X_fund_test_scaled))
            
            rf_model.fit(X_combined_train, y_direction_train)
            rf_pred = rf_model.predict(X_combined_test)
            rf_prob = rf_model.predict_proba(X_combined_test)[:, 1]
            
            # Feature importance
            feature_names = list(X_price.columns) + list(X_fund.columns)
            rf_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
        else:
            rf_model.fit(X_price_train_scaled, y_direction_train)
            rf_pred = rf_model.predict(X_price_test_scaled)
            rf_prob = rf_model.predict_proba(X_price_test_scaled)[:, 1]
            
            # Feature importance
            feature_names = list(X_price.columns)
            rf_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        logger.debug(f"Random Forest accuracy: {accuracy_score(y_direction_test, rf_pred):.4f}")
        logger.debug("\nTop 10 most important features:")
        logger.debug(rf_importance.head(10))
        
        models['random_forest'] = rf_model
        models['rf_importance'] = rf_importance
        
        # 2. Gradient Boosting for return magnitude
        logger.debug("Training Gradient Boosting model for return prediction")
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        if X_fund is not None:
            gb_model.fit(X_combined_train, y_returns_train)
            gb_pred = gb_model.predict(X_combined_test)
            
            # Feature importance
            gb_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': gb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
        else:
            gb_model.fit(X_price_train_scaled, y_returns_train)
            gb_pred = gb_model.predict(X_price_test_scaled)
            
            # Feature importance
            gb_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': gb_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        mse = mean_squared_error(y_returns_test, gb_pred)
        rmse = np.sqrt(mse)
        logger.debug(f"Gradient Boosting RMSE: {rmse:.6f}")
        logger.debug("\nTop 10 most important features for return prediction:")
        logger.debug(gb_importance.head(10))
        
        models['gradient_boosting'] = gb_model
        models['gb_importance'] = gb_importance
        
        # 3. Neural Network with separate inputs for price vs fundamentals
        if X_fund is not None:
            logger.debug("Training Neural Network with separate inputs")
            
            # Build a model with two input branches
            price_input = Input(shape=(X_price_train_scaled.shape[1],), name='price_input')
            price_dense = Dense(32, activation='relu')(price_input)
            price_dense = Dropout(0.2)(price_dense)
            
            fund_input = Input(shape=(X_fund_train_scaled.shape[1],), name='fund_input')
            fund_dense = Dense(32, activation='relu')(fund_input)
            fund_dense = Dropout(0.2)(fund_dense)
            
            merged = Concatenate()([price_dense, fund_dense])
            merged = Dense(32, activation='relu')(merged)
            merged = Dropout(0.2)(merged)
            
            # Output layer for direction classification
            direction_output = Dense(1, activation='sigmoid', name='direction_output')(merged)
            
            # Output layer for return regression
            returns_output = Dense(1, activation='linear', name='returns_output')(merged)
            
            # Create and compile model
            nn_model = Model(
                inputs=[price_input, fund_input],
                outputs=[direction_output, returns_output]
            )
            
            nn_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss={
                    'direction_output': 'binary_crossentropy',
                    'returns_output': 'mse'
                },
                metrics={
                    'direction_output': 'accuracy',
                    'returns_output': 'mse'
                }
            )
            
            # Train the model
            history = nn_model.fit(
                [X_price_train_scaled, X_fund_train_scaled],
                {
                    'direction_output': y_direction_train,
                    'returns_output': y_returns_train
                },
                epochs=50,
                batch_size=32,
                verbose=0,
                validation_data=(
                    [X_price_test_scaled, X_fund_test_scaled],
                    {
                        'direction_output': y_direction_test,
                        'returns_output': y_returns_test
                    }
                )
            )
            
            # Evaluate
            nn_eval = nn_model.evaluate(
                [X_price_test_scaled, X_fund_test_scaled],
                {
                    'direction_output': y_direction_test,
                    'returns_output': y_returns_test
                },
                verbose=0
            )
            
            logger.debug(f"Neural Network combined loss: {nn_eval[0]:.4f}")
            logger.debug(f"Direction prediction accuracy: {nn_eval[3]:.4f}")
            logger.debug(f"Returns prediction MSE: {nn_eval[4]:.6f}")
            
            models['neural_network'] = nn_model
            models['nn_history'] = history.history
        
        # Calculate combined prediction using majority voting for direction
        if X_fund is not None and 'neural_network' in models:
            # RF model prediction
            direction_pred_rf = rf_pred
            
            # NN model prediction
            direction_pred_nn = (nn_model.predict([X_price_test_scaled, X_fund_test_scaled])[0] > 0.5).astype(int).flatten()
            
            # Simple ensemble (majority voting)
            ensemble_pred = ((direction_pred_rf + direction_pred_nn) >= 1).astype(int)
            
            logger.debug(f"Ensemble model accuracy: {accuracy_score(y_direction_test, ensemble_pred):.4f}")
            logger.debug(classification_report(y_direction_test, ensemble_pred))
            
        # Store test predictions for performance evaluation
        performance = {
            'y_direction_test': y_direction_test,
            'y_returns_test': y_returns_test,
            'rf_pred': rf_pred,
            'rf_prob': rf_prob,
            'gb_pred': gb_pred
        }
        
        if X_fund is not None and 'neural_network' in models:
            performance['nn_direction_pred'] = direction_pred_nn.flatten()
            performance['nn_returns_pred'] = nn_model.predict([X_price_test_scaled, X_fund_test_scaled])[1].flatten()
            performance['ensemble_pred'] = ensemble_pred
        
        return models, scalers, performance
    
    def evaluate_performance(self, symbol, performance):
        """
        Evaluate model performance and calculate trading metrics.
        
        Args:
            symbol (str): The commodity symbol
            performance (dict): Performance data
            
        Returns:
            dict: Performance metrics
        """
        logger.info(f"Evaluating performance for {symbol}")
        
        # Get test predictions
        y_direction_test = performance['y_direction_test']
        y_returns_test = performance['y_returns_test']
        rf_pred = performance['rf_pred']
        rf_prob = performance['rf_prob']
        gb_pred = performance['gb_pred']
        
        # Calculate metrics
        metrics = {}
        
        # Random Forest metrics
        metrics['rf_accuracy'] = accuracy_score(y_direction_test, rf_pred)
        metrics['rf_report'] = classification_report(y_direction_test, rf_pred, output_dict=True)
        
        # Gradient Boosting metrics
        metrics['gb_mse'] = mean_squared_error(y_returns_test, gb_pred)
        metrics['gb_rmse'] = np.sqrt(metrics['gb_mse'])
        
        # Neural Network metrics (if available)
        if 'nn_direction_pred' in performance:
            nn_direction_pred = performance['nn_direction_pred']
            nn_returns_pred = performance['nn_returns_pred']
            
            metrics['nn_accuracy'] = accuracy_score(y_direction_test, (nn_direction_pred > 0.5).astype(int))
            metrics['nn_mse'] = mean_squared_error(y_returns_test, nn_returns_pred)
            metrics['nn_rmse'] = np.sqrt(metrics['nn_mse'])
        
        # Ensemble metrics (if available)
        if 'ensemble_pred' in performance:
            ensemble_pred = performance['ensemble_pred']
            metrics['ensemble_accuracy'] = accuracy_score(y_direction_test, ensemble_pred)
            metrics['ensemble_report'] = classification_report(y_direction_test, ensemble_pred, output_dict=True)
        
        # Calculate trading performance
        # We'll simulate a simple trading strategy using the ensemble predictions or RF if ensemble not available
        if 'ensemble_pred' in performance:
            position = performance['ensemble_pred']
        else:
            position = rf_pred
        
        # Calculate returns
        strategy_returns = position * y_returns_test
        
        # Position sizing based on prediction confidence (if available)
        if 'nn_direction_pred' in performance and 'ensemble_pred' not in performance:
            # Scale position size by confidence
            confidence = np.abs(rf_prob - 0.5) * 2  # Scale 0.5-1.0 to 0-1.0
            sized_position = np.sign(rf_prob - 0.5) * confidence
            sized_strategy_returns = sized_position * y_returns_test
            
            metrics['sized_cumulative_return'] = np.cumprod(1 + sized_strategy_returns) - 1
            metrics['sized_total_return'] = metrics['sized_cumulative_return'].iloc[-1]
            metrics['sized_sharpe'] = sized_strategy_returns.mean() / sized_strategy_returns.std() * np.sqrt(252)
        
        # Calculate performance metrics
        metrics['strategy_returns'] = strategy_returns
        metrics['cumulative_return'] = np.cumprod(1 + strategy_returns) - 1
        metrics['total_return'] = metrics['cumulative_return'].iloc[-1]
        metrics['sharpe'] = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        metrics['max_drawdown'] = (metrics['cumulative_return'] - metrics['cumulative_return'].cummax()).min()
        
        # Calculate benchmark metrics (buy and hold)
        benchmark_returns = y_returns_test
        metrics['benchmark_cumulative_return'] = np.cumprod(1 + benchmark_returns) - 1
        metrics['benchmark_total_return'] = metrics['benchmark_cumulative_return'].iloc[-1]
        metrics['benchmark_sharpe'] = benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)
        metrics['benchmark_max_drawdown'] = (metrics['benchmark_cumulative_return'] - metrics['benchmark_cumulative_return'].cummax()).min()
        
        # Log performance summary
        logger.info(f"Performance summary for {symbol}:")
        logger.info(f"Strategy total return: {metrics['total_return']:.2%}")
        logger.info(f"Benchmark total return: {metrics['benchmark_total_return']:.2%}")
        logger.info(f"Strategy Sharpe ratio: {metrics['sharpe']:.2f}")
        logger.info(f"Strategy max drawdown: {metrics['max_drawdown']:.2%}")
        
        return metrics
    
    def plot_performance(self, symbol, performance_data, metrics):
        """
        Plot performance charts.
        
        Args:
            symbol (str): The commodity symbol
            performance_data (dict): Performance data
            metrics (dict): Performance metrics
            
        Returns:
            None
        """
        logger.info(f"Plotting performance charts for {symbol}")
        
        # Set up plots
        plt.figure(figsize=(15, 15))
        
        # 1. Cumulative Returns
        plt.subplot(3, 1, 1)
        plt.plot(metrics['cumulative_return'], label='Strategy')
        plt.plot(metrics['benchmark_cumulative_return'], label='Buy & Hold')
        
        if 'sized_cumulative_return' in metrics:
            plt.plot(metrics['sized_cumulative_return'], label='Sized Strategy')
            
        plt.title(f'Cumulative Returns for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        # 2. Feature Importance
        plt.subplot(3, 1, 2)
        
        if 'rf_importance' in self.models[symbol]:
            rf_importance = self.models[symbol]['rf_importance']
            top_features = rf_importance.head(10)
            
            plt.barh(top_features['feature'], top_features['importance'])
            plt.title(f'Top 10 Features for Direction Prediction ({symbol})')
            plt.xlabel('Importance')
            plt.gca().invert_yaxis()
            plt.grid(True, axis='x')
        
        # 3. Drawdowns
        plt.subplot(3, 1, 3)
        strategy_drawdown = metrics['cumulative_return'] - metrics['cumulative_return'].cummax()
        benchmark_drawdown = metrics['benchmark_cumulative_return'] - metrics['benchmark_cumulative_return'].cummax()
        
        plt.plot(strategy_drawdown, label='Strategy')
        plt.plot(benchmark_drawdown, label='Buy & Hold')
        plt.title(f'Drawdowns for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results/{symbol}_performance.png')
        plt.close()
        
        # Additional plots if Neural Network was used
        if 'nn_history' in self.models[symbol]:
            history = self.models[symbol]['nn_history']
            
            plt.figure(figsize=(15, 10))
            
            # Plot training & validation loss values
            plt.subplot(2, 1, 1)
            plt.plot(history['direction_output_loss'], label='Direction Loss (Train)')
            plt.plot(history['val_direction_output_loss'], label='Direction Loss (Val)')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            
            # Plot training & validation accuracy values
            plt.subplot(2, 1, 2)
            plt.plot(history['direction_output_accuracy'], label='Direction Accuracy (Train)')
            plt.plot(history['val_direction_output_accuracy'], label='Direction Accuracy (Val)')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'results/{symbol}_nn_training.png')
            plt.close()
        
    def generate_trading_signals(self, symbol, data):
        """
        Generate trading signals using the trained models.
        
        Args:
            symbol (str): The commodity symbol
            data (pandas.DataFrame): Recent feature-engineered data
            
        Returns:
            pandas.DataFrame: Trading signals
        """
        logger.info(f"Generating trading signals for {symbol}")
        
        # Make a copy
        df = data.copy()
        
        # Split features
        price_features = [col for col in df.columns if any(x in col for x in ['MA_', 'Volatility_', 'RSI', 'Close_to_MA_'])]
        other_features = [col for col in df.columns if col not in price_features and col not in ['open', 'high', 'low', 'close', 'volume', 'Returns', 'Direction']]
        
        X_price = df[price_features]
        X_fund = df[other_features] if other_features else None
        
        # Scale features
        X_price_scaled = self.scalers[symbol]['price'].transform(X_price)
        
        if X_fund is not None and 'fund' in self.scalers[symbol]:
            X_fund_scaled = self.scalers[symbol]['fund'].transform(X_fund)
        
        # Generate predictions
        # 1. Random Forest for direction
        if X_fund is not None and 'fund' in self.scalers[symbol]:
            X_combined = np.hstack((X_price_scaled, X_fund_scaled))
            rf_pred = self.models[symbol]['random_forest'].predict(X_combined)
            rf_prob = self.models[symbol]['random_forest'].predict_proba(X_combined)[:, 1]
            
            # 2. Neural Network (if available)
            if 'neural_network' in self.models[symbol]:
                nn_output = self.models[symbol]['neural_network'].predict([X_price_scaled, X_fund_scaled])
                nn_direction = (nn_output[0] > 0.5).astype(int).flatten()
                nn_returns = nn_output[1].flatten()
                
                # Ensemble prediction (majority voting)
                ensemble_pred = ((rf_pred + nn_direction) >= 1).astype(int)
                
                # Use ensemble for final direction prediction
                direction_pred = ensemble_pred
                
                # Use neural network for return prediction
                returns_pred = nn_returns
            else:
                direction_pred = rf_pred
                # Use gradient boosting for returns prediction
                returns_pred = self.models[symbol]['gradient_boosting'].predict(X_combined)
        else:
            rf_pred = self.models[symbol]['random_forest'].predict(X_price_scaled)
            rf_prob = self.models[symbol]['random_forest'].predict_proba(X_price_scaled)[:, 1]
            direction_pred = rf_pred
            returns_pred = self.models[symbol]['gradient_boosting'].predict(X_price_scaled)
        
        # Calculate confidence
        confidence = np.abs(rf_prob - 0.5) * 2  # Scale 0.5-1.0 to 0-1.0
        
        # Create signals DataFrame
        signals = pd.DataFrame({
            'Date': df.index,
            'Symbol': symbol,
            'Close': df['close'],
            'Direction': direction_pred,
            'Returns_Pred': returns_pred,
            'Confidence': confidence,
            'Position': np.sign(direction_pred - 0.5) * confidence  # Position size based on confidence
        })
        
        # Calculate risk-adjusted position size
        # Scale position by volatility (inverse of recent volatility)
        if 'Volatility_20' in df.columns:
            volatility = df['Volatility_20'].iloc[-1]
            volatility_factor = 0.01 / max(volatility, 0.001)  # Target 1% daily volatility
            signals['Position'] = signals['Position'] * min(volatility_factor, 2)  # Cap at 2x leverage
        
        # Add stop loss level (2% from entry)
        signals['Stop_Loss'] = np.where(signals['Direction'] > 0, 
                                       signals['Close'] * 0.98,  # Long stop
                                       signals['Close'] * 1.02)  # Short stop
        
        # Add take profit level (estimated return or 2x stop distance)
        signals['Take_Profit'] = np.where(signals['Direction'] > 0,
                                        signals['Close'] * (1 + max(signals['Returns_Pred'] * 2, 0.04)),  # Long take profit
                                        signals['Close'] * (1 - max(signals['Returns_Pred'] * 2, 0.04)))  # Short take profit
        
        return signals
    
    def run(self):
        """
        Run the full model pipeline.
        """
        logger.info("Starting fundamental ML model run")
        
        for symbol in self.symbols:
            try:
                # 1. Load data
                data = self.load_data(symbol)
                
                if data is None or len(data) < 50:
                    logger.error(f"Insufficient data for {symbol}. Skipping.")
                    continue
                
                # 2. Engineer features
                featured_data = self.engineer_features(data)
                
                # 3. Prepare training data
                X_price, X_fund, y_direction, y_returns = self.prepare_training_data(featured_data)
                
                # 4. Train models
                models, scalers, performance_data = self.train_models(
                    symbol, X_price, X_fund, y_direction, y_returns
                )
                
                # 5. Save models and scalers
                self.models[symbol] = models
                self.scalers[symbol] = scalers
                
                # 6. Evaluate performance
                metrics = self.evaluate_performance(symbol, performance_data)
                self.performance[symbol] = metrics
                
                # 7. Plot performance
                self.plot_performance(symbol, performance_data, metrics)
                
                # 8. Generate current trading signals
                recent_data = featured_data.iloc[-100:]
                signals = self.generate_trading_signals(symbol, recent_data)
                
                # Display the latest signal
                latest_signal = signals.iloc[-1]
                logger.info(f"Latest trading signal for {symbol}:")
                logger.info(f"Direction: {'LONG' if latest_signal['Direction'] > 0 else 'SHORT'}")
                logger.info(f"Confidence: {latest_signal['Confidence']:.2f}")
                logger.info(f"Position Size: {latest_signal['Position']:.2f}")
                logger.info(f"Stop Loss: {latest_signal['Stop_Loss']:.2f}")
                logger.info(f"Take Profit: {latest_signal['Take_Profit']:.2f}")
                
                # Save models
                joblib.dump(models, f"models/{symbol}_fundamental_models.pkl")
                joblib.dump(scalers, f"models/{symbol}_fundamental_scalers.pkl")
                
                # Save signals
                signals.to_csv(f"results/{symbol}_signals.csv")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    def save_feature_importance_report(self):
        """
        Generate a detailed report on feature importance across models.
        """
        logger.info("Generating feature importance report")
        
        for symbol in self.symbols:
            if symbol not in self.models:
                continue
                
            with open(f"results/{symbol}_feature_importance.txt", "w") as f:
                f.write(f"FEATURE IMPORTANCE REPORT FOR {symbol}\n")
                f.write("=" * 80 + "\n\n")
                
                if 'rf_importance' in self.models[symbol]:
                    f.write("Random Forest Feature Importance for Direction Prediction:\n")
                    f.write("-" * 80 + "\n")
                    
                    rf_importance = self.models[symbol]['rf_importance']
                    for idx, row in rf_importance.iterrows():
                        f.write(f"{row['feature']}: {row['importance']:.4f}\n")
                    
                    f.write("\n\n")
                
                if 'gb_importance' in self.models[symbol]:
                    f.write("Gradient Boosting Feature Importance for Return Prediction:\n")
                    f.write("-" * 80 + "\n")
                    
                    gb_importance = self.models[symbol]['gb_importance']
                    for idx, row in gb_importance.iterrows():
                        f.write(f"{row['feature']}: {row['importance']:.4f}\n")
                    
                    f.write("\n\n")
                
                # Group features by type and show importance
                if 'rf_importance' in self.models[symbol]:
                    f.write("Feature Importance by Category:\n")
                    f.write("-" * 80 + "\n")
                    
                    rf_importance = self.models[symbol]['rf_importance']
                    
                    # Define feature categories
                    categories = {
                        'Price': ['MA_', 'Close_to_MA_', 'Volatility_', 'RSI'],
                        'COT': ['Commercial_', 'NonCommercial_', 'Comm_NonComm_'],
                        'Inventory': ['Inventory_', 'Low_Inventory', 'High_Inventory'],
                        'Weather': ['Temp_', 'HDDs_', 'CDDs_', 'Extreme_Cold', 'Extreme_Heat'],
                        'OPEC': ['OPEC_', 'Composite_Fundamental'],
                        'Divergence': ['_Divergence', '_Alignment']
                    }
                    
                    for category, patterns in categories.items():
                        category_features = rf_importance[
                            rf_importance['feature'].apply(
                                lambda x: any(pattern in x for pattern in patterns)
                            )
                        ]
                        
                        if not category_features.empty:
                            total_importance = category_features['importance'].sum()
                            f.write(f"{category} Features Total Importance: {total_importance:.4f}\n")
                            
                            for idx, row in category_features.iterrows():
                                f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
                            
                            f.write("\n")
                
                # Performance metrics
                if symbol in self.performance:
                    metrics = self.performance[symbol]
                    
                    f.write("Performance Metrics:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Strategy Total Return: {metrics['total_return']:.2%}\n")
                    f.write(f"Benchmark Total Return: {metrics['benchmark_total_return']:.2%}\n")
                    f.write(f"Strategy Sharpe Ratio: {metrics['sharpe']:.2f}\n")
                    f.write(f"Strategy Maximum Drawdown: {metrics['max_drawdown']:.2%}\n")
                    f.write(f"Random Forest Accuracy: {metrics['rf_accuracy']:.4f}\n")
                    
                    if 'ensemble_accuracy' in metrics:
                        f.write(f"Ensemble Model Accuracy: {metrics['ensemble_accuracy']:.4f}\n")
                        
                    f.write("\n")
            
            logger.info(f"Feature importance report saved for {symbol}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fundamental ML Model for Trading')
    parser.add_argument('--symbols', type=str, default='CL',
                        help='Comma-separated list of symbols to analyze')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    return parser.parse_args()


def check_fundamental_data_availability():
    """
    Check if fundamental data is available in the database and print instructions if it's not.
    """
    try:
        # Use direct connection with hardcoded credentials that are known to work
        db = PostgresConnector(
            host="localhost",
            port="5432",
            dbname="trading_db",
            user="postgres",
            password="Makingmoney25!"
        )
        
        print("\nChecking database structure...")
        print("Database connection successful!")
        
        # First check if there are separate tables for fundamental data
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
        """
        tables = db.query(tables_query)
        
        if not tables:
            print("Error: No tables found in database")
            return False
        
        table_names = [table['table_name'] for table in tables]
        print(f"\nTables in database: {', '.join(table_names)}")
        
        # Look for fundamental data tables
        fundamental_tables_exist = False
        has_cot_table = 'cot_data' in table_names
        has_eia_table = 'eia_data' in table_names
        has_weather_table = 'weather_data' in table_names or 'weather' in table_names
        has_opec_table = 'opec_data' in table_names
        
        if has_cot_table or has_eia_table or has_weather_table or has_opec_table:
            fundamental_tables_exist = True
            print("\nFound separate tables for fundamental data:")
            if has_cot_table:
                print("- cot_data: COT (Commitments of Traders) data")
                # Check if COT data table has the expected columns
                cot_cols_query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'cot_data'
                """
                cot_cols = db.query(cot_cols_query)
                if cot_cols:
                    col_names = [col['column_name'] for col in cot_cols]
                    print(f"  Columns: {', '.join(col_names)}")
                    
                    # Check for typos in column names
                    typos = []
                    if 'comm_postions_long' in col_names:
                        typos.append("'comm_postions_long' (should be 'comm_positions_long')")
                    if 'comm_postions_short' in col_names:
                        typos.append("'comm_postions_short' (should be 'comm_positions_short')")
                    if 'noncomm_postionns_long' in col_names:
                        typos.append("'noncomm_postionns_long' (should be 'noncomm_positions_long')")
                    if 'noncomm_postionns_short' in col_names:
                        typos.append("'noncomm_postionns_short' (should be 'noncomm_positions_short')")
                    
                    if typos:
                        print(f"  Note: Found typos in column names: {', '.join(typos)}")
                        print("  The model will handle these typos automatically")
                    
                    # Check if there's data in the COT table
                    cot_count_query = "SELECT COUNT(*) as count FROM cot_data"
                    cot_count = db.query_one(cot_count_query)
                    if cot_count and cot_count['count'] > 0:
                        print(f"  Records: {cot_count['count']} rows")
                        
                        # Get a small sample of the data to show
                        sample_query = "SELECT * FROM cot_data LIMIT 1"
                        sample = db.query_one(sample_query)
                        if sample:
                            print("\n  Data sample:")
                            for col, val in sample.items():
                                if col != 'id':
                                    print(f"    {col}: {val}")
                    else:
                        print("  Table exists but contains no data")
            
            if has_eia_table:
                print("- eia_data: Energy Information Administration data")
            
            if has_weather_table:
                weather_table = 'weather_data' if 'weather_data' in table_names else 'weather'
                print(f"- {weather_table}: Weather data")
                
            if has_opec_table:
                print("- opec_data: OPEC production, quota, and compliance data")
                # Check if OPEC data table has the expected columns
                opec_cols_query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'opec_data'
                """
                opec_cols = db.query(opec_cols_query)
                if opec_cols:
                    col_names = [col['column_name'] for col in opec_cols]
                    print(f"  Columns: {', '.join(col_names)}")
                    
                    # Check if there's data in the OPEC table
                    opec_count_query = "SELECT COUNT(*) as count FROM opec_data"
                    opec_count = db.query_one(opec_count_query)
                    if opec_count and opec_count['count'] > 0:
                        print(f"  Records: {opec_count['count']} rows")
                        
                        # Get a small sample of the data to show
                        sample_query = "SELECT * FROM opec_data LIMIT 3"
                        samples = db.query(sample_query)
                        if samples:
                            print("\n  Data samples:")
                            for sample in samples:
                                print(f"    {sample['report_date'] if 'report_date' in sample else 'Date'}: {sample['metric'] if 'metric' in sample else 'Metric'} = {sample['value'] if 'value' in sample else 'Value'}")
                    else:
                        print("  Table exists but contains no data")
        
        # If separate tables exist, we can use those instead of expecting columns in market_data
        if fundamental_tables_exist:
            # Check if OPEC data is missing but should be included
            if has_cot_table and not has_opec_table and 'CL' in [r['symbol'] for r in db.query("SELECT DISTINCT symbol FROM market_data WHERE symbol = 'CL'")]:
                print("\n⚠️ OPEC data table is missing, but would be valuable for crude oil analysis.")
                print("To add OPEC data, run the following command:")
                print("   python backfill_missing_data.py --type=opec --start=2018-01-01 --end=2023-12-31")
            
            print("\n✅ Fundamental data tables exist and will be used by the model")
            return True
            
        # If no separate tables, check if market_data has fundamental columns
        print("\nChecking market_data table for fundamental data columns...")
        columns_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'market_data'
        ORDER BY column_name
        """
        columns = db.query(columns_query)
        
        if not columns:
            print("Error: market_data table not found in database")
            return False
            
        # Print all columns to see what's available
        column_names = [col['column_name'] for col in columns]
        print(f"\nColumns in market_data table: {', '.join(column_names)}")
        
        # Check for fundamental data columns
        cot_columns = [col for col in column_names if 'commercial' in col.lower() or 'noncommercial' in col.lower()]
        eia_columns = [col for col in column_names if 'eia' in col.lower()]
        weather_columns = [col for col in column_names if col.lower() in ['temperature', 'humidity', 'precipitation']]
        opec_columns = [col for col in column_names if 'opec' in col.lower()]
        
        # Report on what we found
        if cot_columns:
            print(f"\nFound COT data columns in market_data: {', '.join(cot_columns)}")
        if eia_columns:
            print(f"Found EIA data columns in market_data: {', '.join(eia_columns)}")
        if weather_columns:
            print(f"Found weather data columns in market_data: {', '.join(weather_columns)}")
        if opec_columns:
            print(f"Found OPEC data columns in market_data: {', '.join(opec_columns)}")
        
        # Check if any fundamental data exists
        if not any([cot_columns, eia_columns, weather_columns, opec_columns]):
            print("\n===== FUNDAMENTAL DATA COLUMNS MISSING =====")
            print("No fundamental data columns found in market_data table or separate tables.")
            print("To use this model, you need to either:")
            print("1. Add fundamental data columns to your market_data table, or")
            print("2. Create separate tables for fundamental data (cot_data, eia_data, etc.)")
            print("\nThen backfill fundamental data using the backfill_missing_data.py script:")
            print("\n   python backfill_missing_data.py --type=all --start=2018-01-01 --end=2023-12-31")
            print("\nAfter backfilling the data, run this script again.")
            print("===================================\n")
            return False
            
        # If we have columns, check if they contain data
        if cot_columns:
            check_column = cot_columns[0]
            data_check_query = f"""
            SELECT COUNT(*) as count
            FROM market_data 
            WHERE {check_column} IS NOT NULL
            """
            data_check = db.query_one(data_check_query)
            
            if not data_check or data_check['count'] == 0:
                print("\n===== FUNDAMENTAL DATA MISSING =====")
                print("The database has the required columns, but no data is stored in them.")
                print("You need to backfill fundamental data using the backfill_missing_data.py script.")
                print("\nRun the following command to backfill all fundamental data:")
                print("   python backfill_missing_data.py --type=all --start=2018-01-01 --end=2023-12-31")
                print("\nAfter backfilling the data, run this script again.")
                print("===================================\n")
                return False
        
        # If we reach here, the columns exist and have data
        print("\n✅ Fundamental data is available in the database")
        data_sample_query = f"""
        SELECT 
            timestamp, symbol, 
            cot_commercial_long, cot_commercial_short, cot_commercial_net,
            eia_value, temperature
        FROM market_data
        WHERE symbol = 'CL' AND cot_commercial_long IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 3
        """
        data_sample = db.query(data_sample_query)
        
        if data_sample:
            print("\nSample of fundamental data:")
            for row in data_sample:
                print(f"Date: {row['timestamp'].strftime('%Y-%m-%d')}, Symbol: {row['symbol']}")
                print(f"  COT Commercial Long: {row['cot_commercial_long']}")
                print(f"  COT Commercial Short: {row['cot_commercial_short']}")
                print(f"  COT Commercial Net: {row['cot_commercial_net']}")
                print(f"  EIA Value: {row['eia_value']}")
                print(f"  Temperature: {row['temperature']}")
                print("")
        
        print("====================================")
        return True
        
    except Exception as e:
        print(f"Error checking database: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == '__main__':
    args = parse_arguments()
    symbols = args.symbols.split(',')
    
    print("\n==========================================")
    print("THOR TRADING SYSTEM - FUNDAMENTAL ML BACKTEST")
    print("==========================================\n")
    
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Prediction horizon: 5 days")
    print(f"Initial capital: $100,000.00")
    
    # Print details about model operation
    print("\nMODEL SETTINGS:")
    print("- Data source: Local PostgreSQL database")
    print("- Prediction horizon: 5 days")
    print("- Using fundamental data: COT, EIA, Weather, OPEC")
    
    # First check if fundamental data is available
    if check_fundamental_data_availability():
        start_time = datetime.now()
        model = FundamentalMLModel(symbols=symbols, verbose=args.verbose)
        model.run()
        model.save_feature_importance_report()
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        # Print performance summary
        print("\n==========================================")
        print("FUNDAMENTAL ML MODEL PERFORMANCE SUMMARY")
        print("==========================================")
        num_symbols = len(symbols)
        
        # Print summary information
        print(f"\nRuntime: {runtime:.2f} seconds")
        print(f"Symbols analyzed: {num_symbols}")
        
        # Add performance metrics summary if available
        if hasattr(model, 'performance') and model.performance:
            print("\nModel Performance:")
            for symbol, metrics in model.performance.items():
                if 'total_return' in metrics and 'sharpe' in metrics:
                    print(f"- {symbol}: Return: {metrics['total_return']:.2%}, Sharpe: {metrics['sharpe']:.2f}")
                
                # Print most important features
                if 'rf_importance' in model.models.get(symbol, {}):
                    rf_importance = model.models[symbol]['rf_importance']
                    if not rf_importance.empty:
                        print(f"\nTop 5 features for {symbol}:")
                        for i, (_, row) in enumerate(rf_importance.head(5).iterrows()):
                            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        else:
            print("\nNo performance metrics generated - model may not have processed data correctly")
        
        print("\nOutputs Generated:")
        print("- Trading signals: See results/*.csv")
        print("- Performance charts: See results/*.png")
        print("- Feature importance reports: See results/*.txt")
        print("- Trained models: See models/*.pkl")
        
        print("\nFundamental ML backtest complete!")
    else:
        print("\nFundamental ML backtest aborted: Required fundamental data is missing.")