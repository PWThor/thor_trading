import pandas as pd
import psycopg2
import requests
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    filename="E:/Projects/thor_trading/outputs/logs/integrate_eia_data.log",
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

def get_last_fetched_date(conn, source):
    cur = conn.cursor()
    cur.execute("SELECT last_fetched_date FROM data_fetch_log WHERE source = %s", (source,))
    result = cur.fetchone()
    cur.close()
    return result[0] if result else None

def update_last_fetched_date(conn, source, date):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO data_fetch_log (source, last_fetched_date)
        VALUES (%s, %s)
        ON CONFLICT (source) DO UPDATE
        SET last_fetched_date = EXCLUDED.last_fetched_date
    """, (source, date))
    conn.commit()
    cur.close()

def fetch_eia_data(api_key, endpoint, series_id, start_date, end_date):
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    url = f"https://api.eia.gov/v2{endpoint}?api_key={api_key}&frequency=weekly&data[0]=value&facets[series_id][]={series_id}&start={start_str}&end={end_str}&sort[0][column]=period&sort[0][direction]=asc"

    print(f"Fetching EIA data for {series_id} from {start_str} to {end_str}")
    logging.info(f"Fetching EIA data for {series_id} from {start_str} to {end_str}")
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch EIA data for {series_id}: HTTP {response.status_code}")
        logging.error(f"Failed to fetch EIA data for {series_id}: HTTP {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    print(f"EIA API response for {series_id}: {data}")
    logging.info(f"EIA API response for {series_id}: {data}")

    eia_data = []
    response_data = data.get('response', {}).get('data', [])
    if not response_data:
        print(f"No data returned for {series_id}")
        logging.warning(f"No data returned for {series_id}")
        return pd.DataFrame()

    for item in response_data:
        eia_data.append({
            'Date': pd.to_datetime(item['period']),
            'Value': item['value']
        })

    df = pd.DataFrame(eia_data)
    print(f"Fetched {len(df)} rows for {series_id}")
    logging.info(f"Fetched {len(df)} rows for {series_id}")
    
    if df.empty:
        return pd.DataFrame()
    
    df.sort_values('Date', inplace=True)
    return df

def integrate_eia_data(symbol, api_key, start_date, end_date, live=False):
    # Connect to the database
    conn = connect_db()
    cur = conn.cursor()

    # Determine the date range for fetching
    if live:
        last_fetched = get_last_fetched_date(conn, f"eia_{symbol}")
        if last_fetched:
            start_date = last_fetched + timedelta(days=1)
        else:
            start_date = start_date  # Fetch all data if no previous fetch
        end_date = datetime.now()

    print(f"Processing {symbol} from {start_date} to {end_date} in {'live' if live else 'historical'} mode")
    logging.info(f"Processing {symbol} from {start_date} to {end_date} in {'live' if live else 'historical'} mode")

    # Fetch market data dates for the symbol within the date range
    cur.execute("""
        SELECT timestamp
        FROM market_data
        WHERE symbol = %s AND timestamp::date BETWEEN %s AND %s
        ORDER BY timestamp
    """, (symbol, start_date, end_date))
    market_dates = [row[0] for row in cur.fetchall()]
    print(f"Found {len(market_dates)} market dates for {symbol} between {start_date} and {end_date}")
    logging.info(f"Found {len(market_dates)} market dates for {symbol} between {start_date} and {end_date}")

    # Define EIA series IDs and their APIv2 endpoints
    series_mapping = {
        'eia_inventory': {'endpoint': '/petroleum/stoc/wstk/data/', 'series_id': 'WCESTUS1'},
        'spr_stocks': {'endpoint': '/petroleum/stoc/wstk/data/', 'series_id': 'WCSSTUS1'},
        'eia_production': {'endpoint': '/petroleum/crd/prd/data/', 'series_id': 'WCRFPUS2'},
        'crude_imports': {'endpoint': '/petroleum/crd/imp/data/', 'series_id': 'WCRIMUS2'},
        'crude_exports': {'endpoint': '/petroleum/crd/exp/data/', 'series_id': 'WCREXUS2'},
        'petroleum_demand': {'endpoint': '/petroleum/pet/psup/data/', 'series_id': 'WPPSTUS1'},
        'distillate_demand': {'endpoint': '/petroleum/pet/distill/data/', 'series_id': 'WDISTUS1'}
    }

    # Fetch data for each series
    eia_data = {}
    for field, info in series_mapping.items():
        df = fetch_eia_data(api_key, info['endpoint'], info['series_id'], start_date, end_date)
        if not df.empty:
            # Create a daily time series by interpolating weekly data
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            daily_df = pd.DataFrame({'Date': date_range})
            daily_df = daily_df.merge(df, on='Date', how='left')
            daily_df['Value'] = daily_df['Value'].ffill()  # Forward fill weekly data to daily
            eia_data[field] = daily_df.set_index('Date')['Value']
        else:
            eia_data[field] = pd.Series(index=pd.date_range(start=start_date, end=end_date, freq='D'), dtype=float)

    # Update market_data with EIA data
    for date in market_dates:
        date_only = date.date()
        values = {}
        for field in series_mapping.keys():
            try:
                values[field] = eia_data[field].loc[date_only] if date_only in eia_data[field].index else None
            except KeyError:
                values[field] = None

        print(f"Updating {symbol} at {date}: {values}")
        logging.info(f"Updating {symbol} at {date}: {values}")

        try:
            cur.execute("""
                UPDATE market_data
                SET eia_inventory = %s,
                    spr_stocks = %s,
                    eia_production = %s,
                    crude_imports = %s,
                    crude_exports = %s,
                    petroleum_demand = %s,
                    distillate_demand = %s
                WHERE timestamp = %s AND symbol = %s
            """, (
                values['eia_inventory'],
                values['spr_stocks'],
                values['eia_production'],
                values['crude_imports'],
                values['crude_exports'],
                values['petroleum_demand'],
                values['distillate_demand'],
                date,
                symbol
            ))
        except Exception as e:
            logging.error(f"Error updating EIA data for {symbol} at {date}: {e}")
            print(f"Error updating EIA data for {symbol} at {date}: {e}")

    # Update the last fetched date if live mode
    if live and market_dates:
        latest_date = max([d.date() for d in market_dates])
        update_last_fetched_date(conn, f"eia_{symbol}", latest_date)

    conn.commit()
    cur.close()
    conn.close()
    logging.info(f"Integrated {'live' if live else 'historical'} EIA data for {symbol}")
    print(f"Integrated {'live' if live else 'historical'} EIA data for {symbol}")

if __name__ == "__main__":
    api_key = "2GxRjQeUCEzf5sezwOJOWdP8cVXsKvVWOerU29cY"  # Your EIA API key
    start_date = datetime(2023, 1, 1)  # Reduced date range for testing
    end_date = datetime(2025, 4, 8)
    # For historical data, set live=False; for live data, set live=True
    for symbol in ['CL', 'HO']:
        integrate_eia_data(symbol, api_key, start_date, end_date, live=False)