import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
import pandas as pd

def connect_to_postgres():
    """Establish connection to PostgreSQL database"""
    try:
        # You'll need to update these parameters with your actual credentials
        conn = psycopg2.connect(
            host="localhost",  # Since it's on your C: drive, likely localhost
            database="trading_db",  # Replace with your database name
            user="postgres",           # Replace with your username
            password="Makingmoney25!"        # Replace with your password
        )
        print("Successfully connected to PostgreSQL!")
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def list_all_tables(conn):
    """List all tables in the database"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema='public'
        ORDER BY table_name;
    """)
    tables = cursor.fetchall()
    cursor.close()
    
    print("Tables in database:")
    for table in tables:
        print(f"- {table[0]}")
    
    return [table[0] for table in tables]

def describe_table(conn, table_name):
    """Describe the structure of a specific table"""
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        SELECT column_name, data_type, is_nullable 
        FROM information_schema.columns 
        WHERE table_schema='public' AND table_name=%s
        ORDER BY ordinal_position;
    """), [table_name])
    columns = cursor.fetchall()
    cursor.close()
    
    print(f"\nStructure of table '{table_name}':")
    for column in columns:
        print(f"- {column[0]}: {column[1]} (Nullable: {column[2]})")
    
    return columns

def sample_data(conn, table_name, limit=5):
    """Show sample data from a specific table"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(sql.SQL("""
        SELECT * FROM {} LIMIT %s;
    """).format(sql.Identifier(table_name)), [limit])
    data = cursor.fetchall()
    cursor.close()
    
    print(f"\nSample data from '{table_name}' (limit {limit}):")
    for row in data:
        print(row)
    
    return data

# Main execution
if __name__ == "__main__":
    conn = connect_to_postgres()
    if conn:
        try:
            tables = list_all_tables(conn)
            
            # If tables exist, describe and show sample data for each
            if tables:
                for table in tables:
                    describe_table(conn, table)
                    sample_data(conn, table)
            else:
                print("No tables found in the database.")
                
        finally:
            conn.close()
            print("Connection closed.")