import psycopg2
import sys

# Database connection parameters
host = "localhost"
port = "5432"
dbname = "trading_db"
user = "postgres"
password = "Makingmoney25!"

print(f"Testing connection to PostgreSQL database:")
print(f"Host: {host}")
print(f"Port: {port}")
print(f"Database: {dbname}")
print(f"User: {user}")
print(f"Password: {'*' * len(password)}")
print()

try:
    # Attempt to establish a connection
    print("Connecting to the database...")
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )
    
    # Create a cursor
    cur = conn.cursor()
    
    # Execute a simple query
    print("Executing test query...")
    cur.execute("SELECT version();")
    
    # Fetch and display the results
    db_version = cur.fetchone()
    print(f"PostgreSQL database version: {db_version[0]}")
    
    # List all tables in the database
    print("\nListing all tables in the database:")
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    tables = cur.fetchall()
    
    if tables:
        print("Tables found:")
        for table in tables:
            print(f"- {table[0]}")
            
            # Get row count for each table
            cur.execute(f"SELECT COUNT(*) FROM {table[0]}")
            count = cur.fetchone()[0]
            print(f"  Rows: {count}")
            
            # Get column info for each table
            cur.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table[0]}'
                ORDER BY ordinal_position
            """)
            columns = cur.fetchall()
            if columns:
                print(f"  Columns:")
                for col in columns[:5]:  # Show first 5 columns only
                    print(f"    - {col[0]} ({col[1]})")
                if len(columns) > 5:
                    print(f"    - ... and {len(columns) - 5} more columns")
            print()
    else:
        print("No tables found in the database.")
    
    # Close the cursor and connection
    cur.close()
    conn.close()
    print("Connection closed successfully.")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

print("\nDatabase connection test completed successfully!")