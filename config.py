"""
Thor Trading System Configuration

This file contains configuration settings for the Thor Trading System.
"""

# Database configuration
DB_CONFIG = {
    'host': 'localhost',     # Database host
    'port': '5432',          # Database port
    'dbname': 'trading_db',  # Your actual database name
    'user': 'postgres',      # Database username
    'password': 'Makingmoney25!'  # Your actual password
}

# Trading system configuration
TRADING_CONFIG = {
    'default_symbols': ['CL', 'HO'],  # Default symbols to trade
    'risk_per_trade': 0.02,           # Default risk per trade (2%)
    'max_positions': 3,               # Maximum number of concurrent positions
    'confidence_threshold': 0.7       # Minimum confidence threshold for trading
}

# File paths
PATHS = {
    'models_dir': 'models',           # Directory to store trained models
    'results_dir': 'results',         # Directory to store results and signals
    'logs_dir': 'logs'                # Directory to store logs
}

# API Keys (placeholder - do not store actual keys here, use a separate .env file)
API_KEYS = {
    'eia': 'YOUR_EIA_API_KEY',        # Energy Information Administration API key
    'weather': 'YOUR_WEATHER_API_KEY'  # Weather API key
}