#!/usr/bin/env python
# Script to fix data issues for backtesting
import sys
import pandas as pd
from datetime import datetime, timedelta
import logging
sys.path.append('.')
from connectors.postgres_connector import PostgresConnector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("Fixing data issues for backtesting...")

# Connect to the database
db = PostgresConnector()

# 1. First, create consistent timestamp/date columns across all tables
tables_to_fix = ['eia_data', 'opec_data', 'cot_data', 'weather_data']

for table in tables_to_fix:
    # Check if table exists
    table_exists = db.query_one(
        f"SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = '{table}')"
    )
    
    if not table_exists or not table_exists.get('exists', False):
        print(f"Table {table} does not exist, skipping")
        continue
        
    # Get column names
    columns = db.query(
        f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}'"
    )
    column_names = [col['column_name'] for col in columns]
    column_types = {col['column_name']: col['data_type'] for col in columns}
    
    print(f"\nTable: {table}")
    print(f"Columns: {', '.join(column_names[:5])}...")
    
    # Check if timestamp column exists
    has_timestamp = 'timestamp' in column_names
    has_date = 'date' in column_names
    
    if has_timestamp and has_date:
        print(f"Table {table} already has both timestamp and date columns")
    elif has_timestamp and not has_date:
        print(f"Adding date column to {table} based on timestamp")
        try:
            db.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS date DATE")
            db.execute(f"UPDATE {table} SET date = timestamp::date")
            print(f"Added date column to {table}")
        except Exception as e:
            print(f"Error adding date column to {table}: {str(e)}")
    elif not has_timestamp and has_date:
        print(f"Adding timestamp column to {table} based on date")
        try:
            db.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP")
            db.execute(f"UPDATE {table} SET timestamp = date")
            print(f"Added timestamp column to {table}")
        except Exception as e:
            print(f"Error adding timestamp column to {table}: {str(e)}")
    else:
        print(f"Table {table} has neither timestamp nor date column, cannot fix")

# 2. Update or drop market_data_backtest view to handle date/timestamp issue
print("\nUpdating market_data_backtest view...")
try:
    db.execute("""
    DROP VIEW IF EXISTS market_data_backtest;
    
    CREATE VIEW market_data_backtest AS
    SELECT 
        md.timestamp,
        md.timestamp as date, -- Add aliased date column
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
        cl.timestamp,
        cl.timestamp as date, -- Add aliased date column
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
    
    print("Updated market_data_backtest view to include both timestamp and date columns")
    
    # Verify the view worked
    count = db.query_one("SELECT COUNT(*) as count FROM market_data_backtest WHERE symbol = 'CL-HO-SPREAD'")
    print(f"CL-HO-SPREAD records available: {count['count']}")
    
except Exception as e:
    print(f"Error updating market_data_backtest view: {str(e)}")

# 3. Create a function to handle missing data in the backtest
print("\nCreating a database function to handle missing backtest data...")

try:
    db.execute("""
    CREATE OR REPLACE FUNCTION get_backtest_data(
        symbol_name VARCHAR,
        start_date TIMESTAMP,
        end_date TIMESTAMP
    )
    RETURNS TABLE (
        date TIMESTAMP,
        symbol VARCHAR,
        open FLOAT,
        high FLOAT,
        low FLOAT, 
        close FLOAT,
        volume BIGINT,
        -- Additional data columns
        eia_inventory FLOAT,
        opec_production FLOAT,
        cot_commercial_net FLOAT,
        avg_temperature_nyc FLOAT,
        avg_temperature_chicago FLOAT
    )
    AS $$
    BEGIN
        RETURN QUERY
        SELECT 
            md.date,
            md.symbol,
            md.open,
            md.high,
            md.low,
            md.close,
            md.volume,
            -- Join with EIA data
            e.inventory as eia_inventory,
            -- Join with OPEC data
            o.production as opec_production,
            -- Join with COT data
            c.commercial_net as cot_commercial_net,
            -- Join with weather data
            w1.avg_temperature as avg_temperature_nyc,
            w2.avg_temperature as avg_temperature_chicago
        FROM 
            market_data_backtest md
        -- Left join with EIA data, matching on date
        LEFT JOIN (
            SELECT date, AVG(inventory) as inventory
            FROM eia_data 
            GROUP BY date
        ) e ON md.date::date = e.date::date
        -- Left join with OPEC data
        LEFT JOIN (
            SELECT date, AVG(production) as production
            FROM opec_data
            GROUP BY date
        ) o ON md.date::date = o.date::date
        -- Left join with COT data
        LEFT JOIN (
            SELECT date, AVG(commercial_net) as commercial_net
            FROM cot_data
            GROUP BY date
        ) c ON md.date::date = c.date::date
        -- Left join with weather data for NYC
        LEFT JOIN (
            SELECT date, AVG(avg_temperature) as avg_temperature
            FROM weather_data
            WHERE location = 'New York' OR location LIKE '%nyc%'
            GROUP BY date
        ) w1 ON md.date::date = w1.date::date
        -- Left join with weather data for Chicago
        LEFT JOIN (
            SELECT date, AVG(avg_temperature) as avg_temperature
            FROM weather_data
            WHERE location = 'Chicago' OR location LIKE '%chicago%'
            GROUP BY date
        ) w2 ON md.date::date = w2.date::date
        WHERE 
            md.symbol = symbol_name
            AND md.date BETWEEN start_date AND end_date
        ORDER BY 
            md.date;
    END;
    $$ LANGUAGE plpgsql;
    """)
    
    print("Created database function for integrated backtest data")
    
    # Test the function
    test_data = db.query("""
        SELECT * FROM get_backtest_data('CL-HO-SPREAD', 
            NOW() - INTERVAL '30 days', 
            NOW()
        ) LIMIT 5
    """)
    
    if test_data:
        print("\nTest query successful. Sample data:")
        columns = list(test_data[0].keys())
        print(f"Columns: {', '.join(columns[:10])}")
        
        # Print sample rows
        for i, row in enumerate(test_data[:2]):
            print(f"\nRow {i+1}:")
            for col in columns[:10]:
                print(f"  {col}: {row[col]}")
    else:
        print("No data returned from test query")
    
except Exception as e:
    print(f"Error creating database function: {str(e)}")

# 4. Update the BacktestDBConnector to use the new function
print("\nNext steps:")
print("1. The database has been updated to make date/timestamp consistent")
print("2. A new get_backtest_data function has been created to join all your data sources")
print("\nRun the backtest again with:")
print("python backtesting\\run_backtest.py --years 1 --symbols CL-HO-SPREAD --train-days 180 --test-days 30")