# inspect_trading_db.py
import psycopg2

# Database connection parameters
db_params = {
    'host': 'localhost',
    'port': '5432',
    'database': 'trading_db',
    'user': 'postgres',
    'password': 'Makingmoney25!'  # Replace with your password
}

try:
    with psycopg2.connect(**db_params) as conn:
        cursor = conn.cursor()
        
        # List all tables
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        tables = cursor.fetchall()
        print("Tables in trading_db:", [table[0] for table in tables])

        # Get schema for each table
        for table in tables:
            table_name = table[0]
            print(f"\nSchema for {table_name}:")
            cursor.execute(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s
            """, (table_name,))
            columns = cursor.fetchall()
            for col in columns:
                print(f"  {col[0]}: {col[1]}")

        # Sample data from each table
        for table in tables:
            table_name = table[0]
            print(f"\nSample data from {table_name} (first 5 rows):")
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            rows = cursor.fetchall()
            cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = %s", (table_name,))
            col_names = [col[0] for col in cursor.fetchall()]
            print("Columns:", col_names)
            for row in rows:
                print(row)

except Exception as e:
    print(f"Error: {e}")