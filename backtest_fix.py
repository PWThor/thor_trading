#!/usr/bin/env python
# Script to adapt the database for backtesting
import sys
import pandas as pd
from datetime import datetime, timedelta
sys.path.append('.')
from connectors.postgres_connector import PostgresConnector

print("Creating backtest compatibility view...")

# Connect to the database
db = PostgresConnector()

# Check if CL and HO data exist
symbols = db.query("SELECT DISTINCT symbol FROM market_data")
symbols_list = [s['symbol'] for s in symbols]

print(f"Symbols found in database: {symbols_list}")

if 'CL' not in symbols_list or 'HO' not in symbols_list:
    print("Error: Missing required symbols. Need both CL and HO data")
    sys.exit(1)

# Create a view that transforms the market_data table for backtesting
# This:
# 1. Renames 'timestamp' to 'date' for compatibility
# 2. Adds CL-HO-SPREAD calculation
try:
    print("Creating market_data_backtest view...")
    
    db.execute("""
    DROP VIEW IF EXISTS market_data_backtest;
    
    CREATE VIEW market_data_backtest AS
    SELECT 
        md.timestamp as date,
        md.symbol,
        md.open,
        md.high,
        md.low,
        md.close,
        md.volume
    FROM market_data md
    
    UNION ALL
    
    -- Add calculated CL-HO-SPREAD rows
    SELECT 
        cl.timestamp as date,
        'CL-HO-SPREAD' as symbol,
        cl.open - (ho.open * 42) as open,  -- Convert HO to barrel equivalent
        cl.high - (ho.high * 42) as high,
        cl.low - (ho.low * 42) as low,
        cl.close - (ho.close * 42) as close,
        (cl.volume + ho.volume) / 2 as volume  -- Average volume
    FROM market_data cl
    JOIN market_data ho ON cl.timestamp = ho.timestamp AND ho.symbol = 'HO'
    WHERE cl.symbol = 'CL'
    """)
    
    print("View created successfully!")
    
    # Verify the view worked
    count = db.query_one("SELECT COUNT(*) as count FROM market_data_backtest WHERE symbol = 'CL-HO-SPREAD'")
    print(f"CL-HO-SPREAD records available: {count['count']}")
    
    date_range = db.query_one("""
        SELECT MIN(date) as min_date, MAX(date) as max_date 
        FROM market_data_backtest 
        WHERE symbol = 'CL-HO-SPREAD'
    """)
    
    if date_range and date_range['min_date'] and date_range['max_date']:
        print(f"Date range: {date_range['min_date']} to {date_range['max_date']}")
    
    print("\nSetup complete! You can now run backtests with:")
    print("python backtesting/run_backtest.py --symbols CL-HO-SPREAD --years 1")
    
except Exception as e:
    print(f"Error creating view: {str(e)}")
    sys.exit(1)