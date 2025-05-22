import pandas as pd
import psycopg2
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename="E:/Projects/thor_trading/outputs/logs/load_historical_cot.log",
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
    file_path = f"E:/Projects/thor_trading/data/raw/cot/cot_{year}.csv"
    if not os.path.exists(file_path):
        print(f"COT file for {year} not found at {file_path}")
        logging.warning(f"COT file for {year} not found at {file_path}")
        return pd.DataFrame()
    
    print(f"Loading COT data for {year} from {file_path}")
    logging.info(f"Loading COT data for {year} from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        # Filter for Crude Oil and Heating Oil
        df = df[df['Market_and_Exchange_Names'].str.contains('CRUDE OIL|HEATING OIL', case=False, na=False)]
        # Convert report date
        df['report_date'] = pd.to_datetime(df['Report_Date_as_MM_DD_YYYY'])
        # Map market names to symbols
        df['symbol'] = df['Market_and_Exchange_Names'].apply(
            lambda x: 'CL' if 'CRUDE OIL' in str(x).upper() else 'HO' if 'HEATING OIL' in str(x).upper() else None
        )
        # Drop rows where symbol couldn't be mapped
        df = df.dropna(subset=['symbol'])
        # Select the required columns
        required_columns = [
            'report_date',
            'symbol',
            'Comm_Positions_Long_All',
            'Comm_Positions_Short_All',
            'NonComm_Positions_Long_All',
            'NonComm_Positions_Short_All'
        ]
        # Add managed money columns if they exist
        optional_columns = ['M_Money_Positions_Long_All', 'M_Money_Positions_Short_All']
        columns_to_select = required_columns + [col for col in optional_columns if col in df.columns]
        df = df[columns_to_select]
        # Rename columns to match the database schema
        df = df.rename(columns={
            'Comm_Positions_Long_All': 'comm_positions_long',
            'Comm_Positions_Short_All': 'comm_positions_short',
            'NonComm_Positions_Long_All': 'noncomm_positions_long',
            'NonComm_Positions_Short_All': 'noncomm_positions_short',
            'M_Money_Positions_Long_All': 'm_money_positions_long',
            'M_Money_Positions_Short_All': 'm_money_positions_short'
        })
        # Add managed money columns with default 0 if they don't exist
        if 'm_money_positions_long' not in df.columns:
            df['m_money_positions_long'] = 0
        if 'm_money_positions_short' not in df.columns:
            df['m_money_positions_short'] = 0
        # Compute net positions
        df['noncomm_net_positions'] = df['noncomm_positions_long'] - df['noncomm_positions_short']
        df['comm_net_positions'] = df['comm_positions_long'] - df['comm_positions_short']
        df['mm_net_positions'] = df['m_money_positions_long'] - df['m_money_positions_short']
        
        print(f"Loaded {len(df)} rows for COT data in {year}")
        logging.info(f"Loaded {len(df)} rows for COT data in {year}")
        return df
    except Exception as e:
        print(f"Error loading COT data for {year}: {e}")
        logging.error(f"Error loading COT data for {year}: {e}")
        return pd.DataFrame()

def store_cot_data(df, conn):
    if df.empty:
        return
    
    cur = conn.cursor()
    inserted_rows = 0
    
    for _, row in df.iterrows():
        report_date = row['report_date'].strftime('%Y-%m-%d')
        symbol = row['symbol']
        comm_positions_long = row['comm_positions_long']
        comm_positions_short = row['comm_positions_short']
        noncomm_positions_long = row['noncomm_positions_long']
        noncomm_positions_short = row['noncomm_positions_short']
        m_money_positions_long = row['m_money_positions_long']
        m_money_positions_short = row['m_money_positions_short']
        noncomm_net_positions = row['noncomm_net_positions']
        comm_net_positions = row['comm_net_positions']
        mm_net_positions = row['mm_net_positions']
        
        print(f"Inserting COT data for {symbol} on {report_date}: Comm Long={comm_positions_long}, Comm Short={comm_positions_short}, NonComm Long={noncomm_positions_long}, NonComm Short={noncomm_positions_short}, MM Long={m_money_positions_long}, MM Short={m_money_positions_short}, NonComm Net={noncomm_net_positions}, Comm Net={comm_net_positions}, MM Net={mm_net_positions}")
        logging.info(f"Inserting COT data for {symbol} on {report_date}: Comm Long={comm_positions_long}, Comm Short={comm_positions_short}, NonComm Long={noncomm_positions_long}, NonComm Short={noncomm_positions_short}, MM Long={m_money_positions_long}, MM Short={m_money_positions_short}, NonComm Net={noncomm_net_positions}, Comm Net={comm_net_positions}, MM Net={mm_net_positions}")
        
        try:
            cur.execute("""
                INSERT INTO cot_data (
                    report_date,
                    symbol,
                    comm_positions_long,
                    comm_positions_short,
                    noncomm_positions_long,
                    noncomm_positions_short,
                    m_money_positions_long,
                    m_money_positions_short,
                    noncomm_net_positions,
                    comm_net_positions,
                    mm_net_positions
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (report_date, symbol)
                DO UPDATE SET
                    comm_positions_long = EXCLUDED.comm_positions_long,
                    comm_positions_short = EXCLUDED.comm_positions_short,
                    noncomm_positions_long = EXCLUDED.noncomm_positions_long,
                    noncomm_positions_short = EXCLUDED.noncomm_positions_short,
                    m_money_positions_long = EXCLUDED.m_money_positions_long,
                    m_money_positions_short = EXCLUDED.m_money_positions_short,
                    noncomm_net_positions = EXCLUDED.noncomm_net_positions,
                    comm_net_positions = EXCLUDED.comm_net_positions,
                    mm_net_positions = EXCLUDED.mm_net_positions
            """, (
                report_date,
                symbol,
                comm_positions_long,
                comm_positions_short,
                noncomm_positions_long,
                noncomm_positions_short,
                m_money_positions_long,
                m_money_positions_short,
                noncomm_net_positions,
                comm_net_positions,
                mm_net_positions
            ))
            conn.commit()
            inserted_rows += 1
        except Exception as e:
            print(f"Error inserting COT data for {symbol} on {report_date}: {e}")
            logging.error(f"Error inserting COT data for {symbol} on {report_date}: {e}")
            conn.rollback()
    
    print(f"Inserted {inserted_rows} rows into cot_data")
    logging.info(f"Inserted {inserted_rows} rows into cot_data")

def main():
    # Connect to database
    conn = connect_db()
    
    # Load historical COT data (1992 to 2025)
    for year in range(1992, 2026):
        cot_df = load_cot_data(year, conn)
        if not cot_df.empty:
            store_cot_data(cot_df, conn)
    
    conn.close()
    print("Historical COT data loading completed")
    logging.info("Historical COT data loading completed")

if __name__ == "__main__":
    main()