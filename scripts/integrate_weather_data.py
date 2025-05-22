import pandas as pd
import psycopg2
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename="E:/Projects/thor_trading/outputs/logs/load_historical_cot_weather.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def connect_db():
    try:
        conn = psycopg2.connect(
            dbname="trading_db",
            user="postgres",
            password="Makingmoney25!",
            host="localhost",
            port="5432"
        )
        print("Database connection successful")
        logging.info("Database connection successful")
        return conn
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        logging.error(f"Failed to connect to database: {e}")
        raise

def load_cot_data(year, conn):
    file_path = f"E:/Projects/thor_trading/data/raw/cot/fut_disagg_txt_{year}.txt"
    if not os.path.exists(file_path):
        print(f"COT file for {year} not found at {file_path}")
        logging.warning(f"COT file for {year} not found at {file_path}")
        return pd.DataFrame()
    
    print(f"Loading COT data for {year} from {file_path}")
    logging.info(f"Loading COT data for {year} from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        df = df[df['Market_and_Exchange_Names'].str.contains('CRUDE OIL|HEATING OIL', case=False)]
        df['report_date'] = pd.to_datetime(df['Report_Date_as_MM_DD_YYYY'])
        df['net_positions'] = df['NonComm_Positions_Long_All'] - df['NonComm_Positions_Short_All']
        df['symbol'] = df['Market_and_Exchange_Names'].apply(lambda x: 'CL' if 'CRUDE OIL' in x.upper() else 'HO')
        df = df[['report_date', 'symbol', 'net_positions']]
        
        print(f"Loaded {len(df)} rows for COT data in {year}")
        logging.info(f"Loaded {len(df)} rows for COT data in {year}")
        return df
    except Exception as e:
        print(f"Error loading COT data for {year}: {e}")
        logging.error(f"Error loading COT data for {year}: {e}")
        return pd.DataFrame()

def load_weather_data(conn):
    file_path = "E:/Projects/thor_trading/data/raw/weather/weather_data.csv"
    if not os.path.exists(file_path):
        print(f"Weather file not found at {file_path}")
        logging.warning(f"Weather file not found at {file_path}")
        return pd.DataFrame()
    
    print(f"Loading weather data from {file_path}")
    logging.info(f"Loading weather data from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Rename columns to match the weather_data table schema
        df = df.rename(columns={
            'Date': 'timestamp',
            'Region': 'city',
            'AvgTemperature': 'avg_temperature',
            'Precipitation': 'precipitation',
            'WindSpeed': 'wind_speed'
        })
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Standardize city names to match live fetching format
        df['city'] = df['city'].replace({
            'Cushing': 'Cushing,us-OK',
            'Columbus': 'Columbus,us-OH'
        })
        
        # Assign symbols based on city
        df['symbol'] = df['city'].apply(lambda x: 'CL' if x in ['Houston', 'Cushing,us-OK'] else 'HO')
        
        # Keep only the required columns
        required_columns = ['timestamp', 'symbol', 'city', 'avg_temperature', 'precipitation', 'wind_speed']
        df = df[required_columns]
        
        print(f"Loaded {len(df)} rows for weather data")
        logging.info(f"Loaded {len(df)} rows for weather data")
        return df
    except Exception as e:
        print(f"Error loading weather data: {e}")
        logging.error(f"Error loading weather data: {e}")
        return pd.DataFrame()

def store_cot_data(df, conn):
    if df.empty:
        return
    
    cur = conn.cursor()
    inserted_rows = 0
    
    for _, row in df.iterrows():
        report_date = row['report_date'].strftime('%Y-%m-%d')
        symbol = row['symbol']
        net_positions = row['net_positions']
        
        print(f"Inserting COT data for {symbol} on {report_date}: Net Positions={net_positions}")
        logging.info(f"Inserting COT data for {symbol} on {report_date}: Net Positions={net_positions}")
        
        try:
            cur.execute("""
                INSERT INTO cot_data (report_date, symbol, net_positions)
                VALUES (%s, %s, %s)
                ON CONFLICT (report_date, symbol)
                DO UPDATE SET net_positions = EXCLUDED.net_positions
            """, (
                report_date,
                symbol,
                net_positions
            ))
            conn.commit()
            inserted_rows += 1
        except Exception as e:
            print(f"Error inserting COT data for {symbol} on {report_date}: {e}")
            logging.error(f"Error inserting COT data for {symbol} on {report_date}: {e}")
            conn.rollback()
    
    print(f"Inserted {inserted_rows} rows into cot_data")
    logging.info(f"Inserted {inserted_rows} rows into cot_data")

def store_weather_data(df, conn):
    if df.empty:
        return
    
    cur = conn.cursor()
    inserted_rows = 0
    
    for _, row in df.iterrows():
        timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        symbol = row['symbol']
        city = row['city']
        avg_temperature = row['avg_temperature']
        precipitation = row['precipitation']
        wind_speed = row['wind_speed']
        
        print(f"Inserting weather data for {city} at {timestamp}")
        logging.info(f"Inserting weather data for {city} at {timestamp}")
        
        try:
            cur.execute("""
                INSERT INTO weather_data (timestamp, symbol, city, avg_temperature, precipitation, wind_speed)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (timestamp, symbol, city)
                DO UPDATE SET avg_temperature = EXCLUDED.avg_temperature,
                              precipitation = EXCLUDED.precipitation,
                              wind_speed = EXCLUDED.wind_speed
            """, (
                timestamp,
                symbol,
                city,
                avg_temperature,
                precipitation,
                wind_speed
            ))
            conn.commit()
            inserted_rows += 1
        except Exception as e:
            print(f"Error inserting weather data for {city} at {timestamp}: {e}")
            logging.error(f"Error inserting weather data for {city} at {timestamp}: {e}")
            conn.rollback()
    
    print(f"Inserted {inserted_rows} rows into weather_data")
    logging.info(f"Inserted {inserted_rows} rows into weather_data")

def main():
    # Connect to database
    conn = connect_db()
    
    # Load historical COT data (1992 to 2025)
    for year in range(1992, 2026):
        cot_df = load_cot_data(year, conn)
        if not cot_df.empty:
            store_cot_data(cot_df, conn)
    
    # Load historical weather data from the single weather_data.csv file
    weather_df = load_weather_data(conn)
    if not weather_df.empty:
        store_weather_data(weather_df, conn)
    
    conn.close()
    print("Historical COT and weather data loading completed")
    logging.info("Historical COT and weather data loading completed")

if __name__ == "__main__":
    main()