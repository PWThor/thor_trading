#!/usr/bin/env python
import sys
import os
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta

# Database connection parameters
DB_PARAMS = {
    'host': 'localhost',
    'port': '5432',
    'dbname': 'trading_db',
    'user': 'postgres',
    'password': 'Makingmoney25!'
}

def connect_to_db():
    """Connect to the PostgreSQL database and return connection."""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        conn.autocommit = True
        print(f"Connected to database: {DB_PARAMS['dbname']}")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

def check_table_data(conn, table_name, symbol=None, limit=10):
    """Check data in a specific table for a symbol."""
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    try:
        if symbol:
            cursor.execute(f"SELECT * FROM {table_name} WHERE symbol = %s ORDER BY RANDOM() LIMIT %s", (symbol, limit))
        else:
            cursor.execute(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT %s", (limit,))
        
        records = cursor.fetchall()
        return records
    except Exception as e:
        print(f"Error querying {table_name}: {e}")
        return []

def get_symbol_data_range(conn, symbol):
    """Get date range for a specific symbol in market_data."""
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT MIN(timestamp), MAX(timestamp), COUNT(*) 
            FROM market_data 
            WHERE symbol = %s
        """, (symbol,))
        
        result = cursor.fetchone()
        if result and result[2] > 0:
            return {
                'min_date': result[0],
                'max_date': result[1],
                'count': result[2]
            }
        return None
    except Exception as e:
        print(f"Error getting data range for {symbol}: {e}")
        return None

def main():
    """Main function to check database tables and data."""
    print("Thor Trading System - Database Inspector")
    print("========================================")
    
    conn = connect_to_db()
    
    # 1. List all tables and their sizes
    cursor = conn.cursor()
    cursor.execute("""
        SELECT table_name, 
               (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as column_count,
               (SELECT COUNT(*) FROM ONLY pg_tables JOIN information_schema.tables ON table_name = tablename WHERE table_name = t.table_name) as is_table,
               (SELECT COUNT(*) FROM ONLY pg_views JOIN information_schema.tables ON table_name = viewname WHERE table_name = t.table_name) as is_view
        FROM information_schema.tables t
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    
    tables = cursor.fetchall()
    
    print("\n1. Database Tables and Views:")
    print("-----------------------------")
    for table in tables:
        table_name = table[0]
        column_count = table[1]
        is_table = table[2] > 0
        is_view = table[3] > 0
        
        # Get row count
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            table_type = "TABLE" if is_table else "VIEW" if is_view else "UNKNOWN"
            print(f"- {table_name}: {row_count} rows, {column_count} columns ({table_type})")
        except Exception as e:
            print(f"- {table_name}: Error getting row count - {e}")
    
    # 2. Check market symbols and their data ranges
    print("\n2. Market Data Symbols:")
    print("----------------------")
    
    cursor.execute("SELECT DISTINCT symbol FROM market_data")
    symbols = [row[0] for row in cursor.fetchall()]
    
    for symbol in symbols:
        data_range = get_symbol_data_range(conn, symbol)
        if data_range:
            print(f"- {symbol}: {data_range['count']} records from {data_range['min_date']} to {data_range['max_date']}")
            
            # Sample records
            records = check_table_data(conn, "market_data", symbol, 3)
            if records:
                print("  Sample records:")
                for record in records:
                    print(f"  {record['timestamp']}: Open={record['open']}, Close={record['close']}")
    
    # 3. Check COT data
    print("\n3. COT Data Check:")
    print("----------------")
    cot_records = check_table_data(conn, "cot_data", None, 5)
    
    if cot_records:
        # Get distinct symbols in COT data
        cursor.execute("SELECT DISTINCT symbol FROM cot_data")
        cot_symbols = [row[0] for row in cursor.fetchall()]
        print(f"COT data symbols: {', '.join(cot_symbols)}")
        
        print("Sample COT record:")
        record = cot_records[0]
        for key, value in record.items():
            print(f"  {key}: {value}")
        
        # Check column names to identify any typos
        cursor.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'cot_data'
            ORDER BY ordinal_position
        """)
        columns = [row[0] for row in cursor.fetchall()]
        print(f"COT data columns: {', '.join(columns)}")
        
        # Check for potential typos in column names
        typo_checks = [
            ('positions', 'postions'),
            ('positions', 'postionns'),
        ]
        
        typos_found = []
        for correct, typo in typo_checks:
            typo_columns = [col for col in columns if typo in col]
            if typo_columns:
                typos_found.extend(typo_columns)
        
        if typos_found:
            print("Potential typos found in column names:")
            for col in typos_found:
                print(f"  - {col}")
    else:
        print("No COT data found or error accessing table")
    
    # 4. Check OPEC data
    print("\n4. OPEC Data Check:")
    print("-----------------")
    opec_records = check_table_data(conn, "opec_data", None, 5)
    
    if opec_records:
        # Get distinct metrics in OPEC data
        cursor.execute("SELECT DISTINCT metric FROM opec_data")
        opec_metrics = [row[0] for row in cursor.fetchall()]
        print(f"OPEC metrics: {', '.join(opec_metrics)}")
        
        # Get date range
        cursor.execute("SELECT MIN(report_date), MAX(report_date), COUNT(*) FROM opec_data")
        result = cursor.fetchone()
        if result and result[2] > 0:
            print(f"OPEC data range: {result[0]} to {result[1]} ({result[2]} records)")
        
        print("Sample OPEC records:")
        for record in opec_records[:3]:
            print(f"  {record['report_date']} - {record['metric']}: {record['value']}")
    else:
        print("No OPEC data found or error accessing table")
    
    # 5. Integration test - check if we can join market_data with fundamental tables
    print("\n5. Data Integration Test:")
    print("-----------------------")
    
    # Select a sample date with market data
    cursor.execute("SELECT timestamp::date FROM market_data WHERE symbol = 'CL' ORDER BY RANDOM() LIMIT 1")
    sample_date = cursor.fetchone()
    
    if sample_date:
        sample_date = sample_date[0]
        print(f"Testing data integration for date: {sample_date}")
        
        # Check for COT data on this date
        cursor.execute("SELECT COUNT(*) FROM cot_data WHERE report_date = %s", (sample_date,))
        cot_count = cursor.fetchone()[0]
        print(f"- COT data records for this date: {cot_count}")
        
        # Check for OPEC data on this date
        cursor.execute("SELECT COUNT(*) FROM opec_data WHERE report_date = %s", (sample_date,))
        opec_count = cursor.fetchone()[0]
        print(f"- OPEC data records for this date: {opec_count}")
        
        # Check for weather data on this date
        cursor.execute("SELECT COUNT(*) FROM weather_data WHERE date::date = %s", (sample_date,))
        weather_count = cursor.fetchone()[0]
        print(f"- Weather data records for this date: {weather_count}")
        
        # Check for EIA data on this date
        cursor.execute("SELECT COUNT(*) FROM eia_data WHERE period::date = %s", (sample_date,))
        eia_count = cursor.fetchone()[0]
        print(f"- EIA data records for this date: {eia_count}")
    
    # 6. Suggest possible commands based on data availability
    print("\n6. Suggested Commands:")
    print("--------------------")
    
    # Get the widest date range with data
    cursor.execute("""
        SELECT 
            MIN(dates.min_date) as overall_min,
            MAX(dates.max_date) as overall_max
        FROM (
            SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM market_data
            UNION ALL
            SELECT MIN(report_date), MAX(report_date) FROM cot_data
            UNION ALL
            SELECT MIN(report_date), MAX(report_date) FROM opec_data
            UNION ALL
            SELECT MIN(period), MAX(period) FROM eia_data
        ) as dates
    """)
    
    date_range = cursor.fetchone()
    if date_range and date_range[0] and date_range[1]:
        min_date = date_range[0]
        max_date = date_range[1]
        
        # Suggest dates within the available range, defaulting to recent 1 year of data if available
        suggested_end = max_date
        suggested_start = max(min_date, suggested_end - timedelta(days=365))
        
        print(f"Available data range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        print("\nTo run the fundamental ML model with your actual data:")
        print(f"python fundamental_ml_model.py --symbols=CL,HO --verbose")
        
        print("\nTo backtest using your data:")
        print(f"python ml_backtest.py --mode=backtest --symbols=CL,HO --start={suggested_start.strftime('%Y-%m-%d')} --end={suggested_end.strftime('%Y-%m-%d')}")
        
        print("\nTo simulate paper trading:")
        print(f"python ml_backtest.py --mode=paper_trade --symbols=CL,HO --days=30")
    
    conn.close()
    print("\nDatabase inspection complete.")

if __name__ == "__main__":
    main()