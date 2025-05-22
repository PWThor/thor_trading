# Create a temporary Python script to inspect database
import sys
sys.path.append('.')
from connectors.postgres_connector import PostgresConnector

# Connect to the database
db = PostgresConnector()

# Get list of tables
tables = db.query("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
print("\nTables in database:")
for table in tables:
    print(f"- {table['table_name']}")
    
    # Get row count for each table
    count = db.query_one(f"SELECT COUNT(*) as count FROM {table['table_name']}")
    if count:
        print(f"  Rows: {count['count']}")

# Check specifically for market data tables and their content
print("\nChecking for market data:")

# Check for crude oil data
try:
    cl_data = db.query("SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as count FROM market_data WHERE symbol = 'CL' LIMIT 1")
    if cl_data and cl_data[0]['count'] > 0:
        print(f"CL data: {cl_data[0]['count']} rows from {cl_data[0]['min_date']} to {cl_data[0]['max_date']}")
    else:
        print("No CL data found")
except Exception as e:
    print(f"Error checking CL data: {str(e)}")

# Check for heating oil data
try:
    ho_data = db.query("SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as count FROM market_data WHERE symbol = 'HO' LIMIT 1")
    if ho_data and ho_data[0]['count'] > 0:
        print(f"HO data: {ho_data[0]['count']} rows from {ho_data[0]['min_date']} to {ho_data[0]['max_date']}")
    else:
        print("No HO data found")
except Exception as e:
    print(f"Error checking HO data: {str(e)}")

# Check for spread data
try:
    spread_data = db.query("SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as count FROM market_data WHERE symbol = 'CL-HO-SPREAD' LIMIT 1")
    if spread_data and spread_data[0]['count'] > 0:
        print(f"CL-HO-SPREAD data: {spread_data[0]['count']} rows from {spread_data[0]['min_date']} to {spread_data[0]['max_date']}")
    else:
        print("No CL-HO-SPREAD data found")
except Exception as e:
    print(f"Error checking spread data: {str(e)}")

print("\nDatabase inspection complete!")