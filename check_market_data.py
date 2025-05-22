# Script to check market data structure and content
import sys
sys.path.append('.')
from connectors.postgres_connector import PostgresConnector

# Connect to the database
db = PostgresConnector()

# Get column information for market_data table
print("\nChecking structure of market_data table:")
try:
    columns = db.query("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'market_data'")
    print("Columns in market_data table:")
    for col in columns:
        print(f"- {col['column_name']} ({col['data_type']})")
except Exception as e:
    print(f"Error checking market_data structure: {str(e)}")

# Get sample data from market_data
print("\nSample data from market_data table:")
try:
    sample = db.query("SELECT * FROM market_data LIMIT 5")
    if sample:
        # Print column names
        columns = list(sample[0].keys())
        print(f"Columns: {', '.join(columns)}")
        
        # Print sample rows
        for i, row in enumerate(sample):
            print(f"\nRow {i+1}:")
            for col in columns:
                print(f"  {col}: {row[col]}")
    else:
        print("No data found in market_data table")
except Exception as e:
    print(f"Error getting sample data: {str(e)}")

# Check for unique symbols in market_data
print("\nSymbols in market_data table:")
try:
    symbols = db.query("SELECT DISTINCT symbol FROM market_data")
    if symbols:
        for sym in symbols:
            # Count rows for this symbol
            count = db.query_one(f"SELECT COUNT(*) as count FROM market_data WHERE symbol = '{sym['symbol']}'")
            print(f"- {sym['symbol']}: {count['count']} rows")
    else:
        print("No symbols found in market_data table")
except Exception as e:
    print(f"Error checking symbols: {str(e)}")

print("\nMarket data inspection complete!")