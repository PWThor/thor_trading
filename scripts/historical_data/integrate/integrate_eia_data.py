import pandas as pd
import psycopg2
import requests
import logging
from datetime import datetime
import time

# Set up logging
logging.basicConfig(
    filename="E:/Projects/thor_trading/outputs/logs/integrate_eia_data.log",
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

def fetch_eia_data(api_key, dataset, series_id, start_date, end_date):
    if dataset == "steo":
        base_url = "https://api.eia.gov/v2/steo/data/"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    url = (
        f"{base_url}"
        f"?api_key={api_key}"
        "&frequency=monthly"
        "&data[0]=value"
        f"&facets[seriesId][]={series_id}"
        f"&start={start_date.strftime('%Y-%m')}"
        f"&end={end_date.strftime('%Y-%m')}"
        "&sort[0][column]=period"
        "&sort[0][direction]=desc"
        "&offset=0"
        "&length=5000"
    )
    
    print(f"Fetching EIA data for series {series_id} from {url}")
    logging.info(f"Fetching EIA data for series {series_id} from {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['response']['data'])
        print(f"Fetched {len(df)} rows for series {series_id}")
        logging.info(f"Fetched {len(df)} rows for series {series_id}")
        
        # Log the response if no data is returned
        if df.empty:
            print(f"No data returned for series {series_id}. API response: {data}")
            logging.warning(f"No data returned for series {series_id}. API response: {data}")
            return df

        # Convert period to datetime (YYYY-MM-DD format, first of the month)
        df['period'] = pd.to_datetime(df['period']).dt.strftime('%Y-%m-01')
        # Rename columns, handle both 'unit' and 'units' cases
        df = df.rename(columns={
            'period': 'Period',
            'value': 'Value',
            'seriesId': 'SeriesId',
            'unit': 'Unit',
            'units': 'Unit'
        })
        return df
    except Exception as e:
        print(f"Error fetching EIA data for series {series_id}: {e}")
        logging.error(f"Error fetching EIA data for series {series_id}: {e}")
        return pd.DataFrame()

def integrate_eia_data(api_key, start_date, end_date):
    # Connect to the database
    conn = connect_db()
    cur = conn.cursor()

    # Series IDs to fetch (all from STEO, covering both historical and forecast data)
    series_configs = [
        ('CrudeOilPrice', 'WTIPUUS', 'steo'),           # STEO for price (historical + forecast)
        ('CrudeOilProduction', 'COPRPUS', 'steo'),       # STEO for production
        ('CrudeOilRefineryInput', 'CORIPUS', 'steo'),     # STEO for refinery input
        ('CrudeOilStocks', 'COSXPUS', 'steo'),           # STEO for crude oil stocks
        ('HydropowerGenFL', 'HVEPGEN_FL', 'steo'),       # STEO for hydropower generation in Florida
        ('HeatingOilPrice', 'DSWHUUS', 'steo'),          # STEO for heating oil price
        ('HeatingOilConsumption', 'DATCPUS', 'steo'),    # STEO for heating oil consumption
        ('HeatingOilStocks', 'DAPSPUS', 'steo'),         # STEO for heating oil stocks
        ('DistillateInventoryAlt', 'DFPSPUS', 'steo'),   # STEO for distillate inventory alternative
        ('HeatingOilNetImports', 'DFNIPUS', 'steo'),     # STEO for heating oil net imports
        ('HeatingOilSupply', 'DASUPPLY', 'steo')         # STEO for heating oil supply
    ]

    # Fetch data for each series and insert into eia_data table
    for metric, series_id, dataset in series_configs:
        df = fetch_eia_data(api_key, dataset, series_id, start_date, end_date)
        if df.empty:
            print(f"No data returned for series {series_id} ({dataset})")
            logging.warning(f"No data returned for series {series_id} ({dataset})")
            continue

        inserted_rows = 0
        for _, row in df.iterrows():
            period = row['Period']
            value = float(row['Value']) if pd.notna(row['Value']) else None
            # Handle missing 'Unit' column gracefully
            unit = row.get('Unit', None) if 'Unit' in row else None

            if value is None:
                continue

            print(f"Inserting EIA data for metric {metric} (series {series_id}, {dataset}) at {period}: Value={value}, Unit={unit}")
            logging.info(f"Inserting EIA data for metric {metric} (series {series_id}, {dataset}) at {period}: Value={value}, Unit={unit}")

            try:
                cur.execute("""
                    INSERT INTO eia_data (period, series_id, value, unit)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (period, series_id)
                    DO UPDATE SET value = EXCLUDED.value, unit = EXCLUDED.unit
                """, (
                    period,
                    metric,  # Use metric name as series_id in eia_data for consistency
                    value,
                    unit
                ))
                conn.commit()
                inserted_rows += 1
            except Exception as e:
                print(f"Error inserting EIA data for metric {metric} at {period}: {e}")
                logging.error(f"Error inserting EIA data for metric {metric} at {period}: {e}")
                conn.rollback()

        print(f"Inserted {inserted_rows} rows for metric {metric} (series {series_id}, {dataset}) into eia_data")
        logging.info(f"Inserted {inserted_rows} rows for metric {metric} (series {series_id}, {dataset}) into eia_data")
        time.sleep(2)  # Delay to avoid rate limiting

    cur.close()
    conn.close()
    print("Integrated EIA data into eia_data table")
    logging.info("Integrated EIA data into eia_data table")

if __name__ == "__main__":
    api_key = "2GxRjQeUCEzf5sezwOJOWdP8cVXsKvVWOerU29cY"  # Your EIA API key
    start_date = datetime(1992, 1, 1)  # Full range from 1992 to present
    end_date = datetime(2025, 4, 1)
    integrate_eia_data(api_key, start_date, end_date)