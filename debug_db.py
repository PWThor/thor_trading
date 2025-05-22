#!/usr/bin/env python
"""
Database Debugger for Thor Trading System
"""

import sys
import psycopg2
import psycopg2.extras
import pandas as pd
from datetime import datetime

# Database connection parameters
host = "localhost"
port = "5432"
dbname = "trading_db"
user = "postgres"
password = "Makingmoney25!"

print("Thor Trading System - Database Debugger")
print("======================================")
print(f"Connecting to PostgreSQL database: {dbname}")
print(f"Host: {host}, Port: {port}, User: {user}")
print()

try:
    # Connect to the database
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )
    conn.autocommit = True
    print("✅ Database connection successful!")
    
    # Create a cursor
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # 1. List all tables
    print("\n1. Database Tables:")
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    tables = cursor.fetchall()
    
    if tables:
        for table in tables:
            table_name = table[0]
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"- {table_name}: {count} rows")
    else:
        print("No tables found")
    
    # 2. Check for specific symbols in market_data
    print("\n2. Market Data Symbol Check:")
    symbols_to_check = ['CL', 'HO', 'NG', 'QM', 'RB']
    
    for symbol in symbols_to_check:
        cursor.execute(f"""
            SELECT COUNT(*) FROM market_data WHERE symbol = %s
        """, (symbol,))
        count = cursor.fetchone()[0]
        if count > 0:
            # Get date range
            cursor.execute(f"""
                SELECT MIN(timestamp), MAX(timestamp) FROM market_data WHERE symbol = %s
            """, (symbol,))
            date_range = cursor.fetchone()
            print(f"- {symbol}: {count} rows ({date_range[0]} to {date_range[1]})")
            
            # Get sample data
            cursor.execute(f"""
                SELECT timestamp, open, high, low, close, volume 
                FROM market_data 
                WHERE symbol = %s 
                ORDER BY timestamp DESC 
                LIMIT 2
            """, (symbol,))
            samples = cursor.fetchall()
            print("  Recent samples:")
            for sample in samples:
                print(f"  {sample['timestamp']} | Open: {sample['open']}, Close: {sample['close']}")
        else:
            print(f"- {symbol}: No data found")
    
    # 3. Check COT data
    print("\n3. COT Data Check:")
    cursor.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'cot_data'
        ORDER BY ordinal_position
    """)
    columns = cursor.fetchall()
    print(f"COT data columns: {', '.join([col[0] for col in columns])}")
    
    cursor.execute("""
        SELECT DISTINCT symbol FROM cot_data
    """)
    cot_symbols = cursor.fetchall()
    print(f"COT symbols available: {', '.join([sym[0] for sym in cot_symbols])}")
    
    for symbol in ['CL', 'HO', 'CRUDE', '067651', '022651']:
        cursor.execute("""
            SELECT COUNT(*) FROM cot_data WHERE symbol = %s
        """, (symbol,))
        count = cursor.fetchone()[0]
        if count > 0:
            print(f"- {symbol}: {count} records")
            # Get sample
            cursor.execute(f"""
                SELECT * FROM cot_data WHERE symbol = %s LIMIT 1
            """, (symbol,))
            sample = cursor.fetchone()
            print("  Sample record:")
            for key, value in sample.items():
                print(f"  {key}: {value}")
        else:
            print(f"- {symbol}: No data found")
    
    # 4. Check mapping between market_data and fundamental data
    print("\n4. Data Integration Test:")
    # Try to join market_data with cot_data
    try:
        cursor.execute("""
            SELECT m.timestamp, m.symbol, m.close, c.report_date, c.symbol as cot_symbol
            FROM market_data m
            LEFT JOIN cot_data c ON DATE(m.timestamp) = c.report_date AND (m.symbol = c.symbol OR 
                (m.symbol = 'CL' AND c.symbol IN ('CRUDE', '067651')) OR
                (m.symbol = 'HO' AND c.symbol IN ('HEAT', '022651')))
            WHERE m.symbol = 'CL'
            ORDER BY m.timestamp DESC
            LIMIT 5
        """)
        joined_data = cursor.fetchall()
        print(f"CL joined with COT data - {len(joined_data)} rows returned")
        for row in joined_data:
            print(f"  Market: {row['timestamp']} ({row['symbol']}) - COT: {row['report_date']} ({row['cot_symbol'] or 'None'})")
    except Exception as e:
        print(f"Error joining data: {str(e)}")
    
    # 5. Check OPEC data
    print("\n5. OPEC Data Check:")
    cursor.execute("""
        SELECT COUNT(*) FROM opec_data
    """)
    opec_count = cursor.fetchone()[0]
    print(f"OPEC data records: {opec_count}")
    
    if opec_count > 0:
        cursor.execute("""
            SELECT DISTINCT metric FROM opec_data
        """)
        metrics = cursor.fetchall()
        print(f"OPEC metrics available: {', '.join([m[0] for m in metrics])}")
        
        cursor.execute("""
            SELECT * FROM opec_data ORDER BY report_date DESC LIMIT 3
        """)
        samples = cursor.fetchall()
        print("Recent OPEC data:")
        for sample in samples:
            print(f"  {sample['report_date']} - {sample['metric']}: {sample['value']}")
    
    # Close the cursor and connection
    cursor.close()
    conn.close()
    print("\nDatabase connection closed.")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)

print("\nDebugging complete. Use this information to fix the fundamental ML model.")