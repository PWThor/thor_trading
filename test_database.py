import psycopg2
import pandas as pd

print("Thor Trading - Table Comparison")
print("-" * 35)

# Database connection
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "trading_db"
DB_USER = "postgres"  # Replace with actual username
DB_PASSWORD = r"Makingmoney25!"

try:
    connection = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD
    )
    cursor = connection.cursor()
    
    print("✅ Connected to database!")
    
    # Check all columns in market_data
    print("\n📊 MARKET_DATA COLUMNS:")
    print("-" * 25)
    cursor.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'market_data'
        ORDER BY ordinal_position;
    """)
    market_columns = cursor.fetchall()
    for col, dtype in market_columns:
        print(f"  • {col} ({dtype})")
    
    # Check all columns in daily_market_data
    print("\n📊 DAILY_MARKET_DATA COLUMNS:")
    print("-" * 30)
    cursor.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'daily_market_data'
        ORDER BY ordinal_position;
    """)
    daily_columns = cursor.fetchall()
    for col, dtype in daily_columns:
        print(f"  • {col} ({dtype})")
    
    # Count records in each
    cursor.execute("SELECT COUNT(*) FROM market_data;")
    market_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM daily_market_data;")
    daily_count = cursor.fetchone()[0]
    
    print(f"\n📈 RECORD COUNTS:")
    print(f"  • market_data:       {market_count:,} rows")
    print(f"  • daily_market_data: {daily_count:,} rows")
    
    # Sample from market_data to see what it contains
    print(f"\n👀 SAMPLE FROM market_data:")
    print("-" * 30)
    cursor.execute("SELECT * FROM market_data ORDER BY timestamp DESC LIMIT 3;")
    market_sample = cursor.fetchall()
    
    # Get column names for display
    cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'market_data'
        ORDER BY ordinal_position;
    """)
    market_col_names = [row[0] for row in cursor.fetchall()]
    
    for row in market_sample:
        print("Row:")
        for col_name, value in zip(market_col_names, row):
            print(f"  {col_name}: {value}")
        print()
    
    # Check date ranges
    print("📅 DATE RANGES:")
    print("-" * 15)
    
    # Market data range
    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM market_data;")
    market_range = cursor.fetchone()
    print(f"market_data:       {market_range[0]} to {market_range[1]}")
    
    # Daily market data range  
    cursor.execute("SELECT MIN(trading_day), MAX(trading_day) FROM daily_market_data;")
    daily_range = cursor.fetchone()
    print(f"daily_market_data: {daily_range[0]} to {daily_range[1]}")
    
    # Check if market_data has multiple symbols
    cursor.execute("SELECT DISTINCT symbol FROM market_data LIMIT 10;")
    symbols = cursor.fetchall()
    if symbols:
        print(f"\n📊 SYMBOLS in market_data:")
        for symbol in symbols:
            print(f"  • {symbol[0]}")
    
    connection.close()
    print("\n✅ Table comparison complete!")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\nThis will tell us if market_data has everything! 🎯")