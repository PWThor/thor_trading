import pandas as pd
import psycopg2
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename="E:/Projects/thor_trading/outputs/logs/load_prophetx_data.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def connect_db():
    return psycopg2.connect(
        dbname="trading_db",
        user="postgres",
        password="Makingmoney25!",
        host="localhost",
        port="5432"
    )

def load_prophetx_data(csv_file):
    # Connect to the database
    conn = connect_db()
    cur = conn.cursor()

    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Ensure correct data types and format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Reshape the data: melt CL and HO columns into a single DataFrame
    # First, create separate DataFrames for CL and HO
    cl_df = df[['Date', 'Close_CL', 'Open_CL', 'High_CL', 'Low_CL', 'Volume_CL']].copy()
    cl_df['Symbol'] = 'CL'
    cl_df = cl_df.rename(columns={
        'Close_CL': 'Close',
        'Open_CL': 'Open',
        'High_CL': 'High',
        'Low_CL': 'Low',
        'Volume_CL': 'Volume'
    })
    
    ho_df = df[['Date', 'Close_HO', 'Open_HO', 'High_HO', 'Low_HO', 'Volume_HO']].copy()
    ho_df['Symbol'] = 'HO'
    ho_df = ho_df.rename(columns={
        'Close_HO': 'Close',
        'Open_HO': 'Open',
        'High_HO': 'High',
        'Low_HO': 'Low',
        'Volume_HO': 'Volume'
    })
    
    # Concatenate the DataFrames
    reshaped_df = pd.concat([cl_df, ho_df], ignore_index=True)
    
    # Filter out rows where all OHLCV values are zero
    reshaped_df = reshaped_df[
        ~((reshaped_df['Open'] == 0) &
          (reshaped_df['High'] == 0) &
          (reshaped_df['Low'] == 0) &
          (reshaped_df['Close'] == 0) &
          (reshaped_df['Volume'] == 0))
    ]
    
    # Ensure correct data types
    reshaped_df['Symbol'] = reshaped_df['Symbol'].astype(str)
    reshaped_df['Open'] = reshaped_df['Open'].astype(float)
    reshaped_df['High'] = reshaped_df['High'].astype(float)
    reshaped_df['Low'] = reshaped_df['Low'].astype(float)
    reshaped_df['Close'] = reshaped_df['Close'].astype(float)
    reshaped_df['Volume'] = reshaped_df['Volume'].astype(int)

    # Insert data into market_data table
    for _, row in reshaped_df.iterrows():
        timestamp = row['Date']
        symbol = row['Symbol']
        open_price = row['Open']
        high_price = row['High']
        low_price = row['Low']
        close_price = row['Close']
        volume = row['Volume']

        try:
            cur.execute("""
                INSERT INTO market_data (timestamp, symbol, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (timestamp, symbol) DO UPDATE
                SET open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """, (
                timestamp, symbol, open_price, high_price, low_price, close_price, volume
            ))
        except Exception as e:
            logging.error(f"Error inserting data for {symbol} at {timestamp}: {e}")
            print(f"Error inserting data for {symbol} at {timestamp}: {e}")

    conn.commit()
    cur.close()
    conn.close()
    logging.info("Loaded ProphetX data into market_data table")
    print("Loaded ProphetX data into market_data table")

if __name__ == "__main__":
    csv_file = "E:/Projects/thor_trading/data/raw/prophetx_data.csv"
    load_prophetx_data(csv_file)