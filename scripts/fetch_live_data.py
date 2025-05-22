import pandas as pd
import psycopg2
import requests
import yfinance as yf
import logging
from datetime import datetime, timedelta
import time

# Set up logging
logging.basicConfig(
    filename="E:/Projects/thor_trading/outputs/logs/fetch_live_data.log",
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

def fetch_eia_data(api_key, series_id, start_date, end_date):
    url = (
        "https://api.eia.gov/v2/steo/data/"
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

def fetch_weather_data(api_key, city, symbol):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    print(f"Fetching weather data for {city} from {url}")
    logging.info(f"Fetching weather data for {city} from {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant weather data
        timestamp = datetime.utcfromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        avg_temperature = data['main']['temp']  # Celsius
        wind_speed = data['wind']['speed']  # meters per second
        
        # Precipitation (if available, OpenWeatherMap may not always provide this directly)
        precipitation = data.get('rain', {}).get('1h', 0)  # mm, last 1 hour, default to 0 if not available
        
        print(f"Fetched weather data for {city}: temp={avg_temperature}C, precipitation={precipitation}mm, wind_speed={wind_speed}m/s")
        logging.info(f"Fetched weather data for {city}: temp={avg_temperature}C, precipitation={precipitation}mm, wind_speed={wind_speed}m/s")
        
        return {
            'timestamp': timestamp,
            'symbol': symbol,
            'city': city,
            'avg_temperature': avg_temperature,
            'precipitation': precipitation,
            'wind_speed': wind_speed
        }
    except Exception as e:
        print(f"Error fetching weather data for {city}: {e}")
        logging.error(f"Error fetching weather data for {city}: {e}")
        return None

def fetch_market_data(symbol):
    ticker = yf.Ticker(symbol)
    print(f"Fetching market data for {symbol}")
    logging.info(f"Fetching market data for {symbol}")
    
    try:
        # Fetch the latest data (last 1 day)
        data = ticker.history(period="1d")
        if data.empty:
            print(f"No market data returned for {symbol}")
            logging.warning(f"No market data returned for {symbol}")
            return None
        
        # Get the latest row
        latest = data.iloc[-1]
        timestamp = latest.name.strftime('%Y-%m-%d %H:%M:%S')
        close = latest['Close']
        volume = latest['Volume']
        
        print(f"Fetched market data for {symbol}: close={close}, volume={volume}")
        logging.info(f"Fetched market data for {symbol}: close={close}, volume={volume}")
        
        return {
            'timestamp': timestamp,
            'symbol': symbol,
            'close': close,
            'volume': volume
        }
    except Exception as e:
        print(f"Error fetching market data for {symbol}: {e}")
        logging.error(f"Error fetching market data for {symbol}: {e}")
        return None

def store_eia_data(df, metric, conn):
    cur = conn.cursor()
    inserted_rows = 0
    
    for _, row in df.iterrows():
        period = row['Period']
        value = float(row['Value']) if pd.notna(row['Value']) else None
        unit = row.get('Unit', None) if 'Unit' in row else None

        if value is None:
            continue

        print(f"Inserting EIA data for metric {metric} at {period}: Value={value}, Unit={unit}")
        logging.info(f"Inserting EIA data for metric {metric} at {period}: Value={value}, Unit={unit}")

        try:
            cur.execute("""
                INSERT INTO eia_data (period, series_id, value, unit)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (period, series_id)
                DO UPDATE SET value = EXCLUDED.value, unit = EXCLUDED.unit
            """, (
                period,
                metric,
                value,
                unit
            ))
            conn.commit()
            inserted_rows += 1
        except Exception as e:
            print(f"Error inserting EIA data for metric {metric} at {period}: {e}")
            logging.error(f"Error inserting EIA data for metric {metric} at {period}: {e}")
            conn.rollback()

    print(f"Inserted {inserted_rows} rows for metric {metric} into eia_data")
    logging.info(f"Inserted {inserted_rows} rows for metric {metric} into eia_data")

def store_weather_data(data, conn):
    if not data:
        return
    
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO weather_data (timestamp, symbol, city, avg_temperature, precipitation, wind_speed)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (timestamp, symbol, city)
            DO UPDATE SET avg_temperature = EXCLUDED.avg_temperature,
                          precipitation = EXCLUDED.precipitation,
                          wind_speed = EXCLUDED.wind_speed
        """, (
            data['timestamp'],
            data['symbol'],
            data['city'],
            data['avg_temperature'],
            data['precipitation'],
            data['wind_speed']
        ))
        conn.commit()
        print(f"Inserted weather data for {data['city']} at {data['timestamp']}")
        logging.info(f"Inserted weather data for {data['city']} at {data['timestamp']}")
    except Exception as e:
        print(f"Error inserting weather data for {data['city']}: {e}")
        logging.error(f"Error inserting weather data for {data['city']}: {e}")
        conn.rollback()

def store_market_data(data, conn):
    if not data:
        return
    
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO market_data (timestamp, symbol, close, volume)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (timestamp, symbol)
            DO UPDATE SET close = EXCLUDED.close, volume = EXCLUDED.volume
        """, (
            data['timestamp'],
            data['symbol'],
            data['close'],
            data['volume']
        ))
        conn.commit()
        print(f"Inserted market data for {data['symbol']} at {data['timestamp']}")
        logging.info(f"Inserted market data for {data['symbol']} at {data['timestamp']}")
    except Exception as e:
        print(f"Error inserting market data for {data['symbol']}: {e}")
        logging.error(f"Error inserting market data for {data['symbol']}: {e}")
        conn.rollback()

def main():
    # API keys
    eia_api_key = "2GxRjQeUCEzf5sezwOJOWdP8cVXsKvVWOerU29cY"
    weather_api_key = "your_openweathermap_api_key"  # Replace with your OpenWeatherMap API key
    
    # Date range for EIA data (last month to current month)
    end_date = datetime.now()
    start_date = (end_date - timedelta(days=30)).replace(day=1)  # Last month
    
    # Connect to database
    conn = connect_db()
    
    # EIA data series and metrics
    series_configs = [
        ('CrudeOilPrice', 'WTIPUUS'),
        ('CrudeOilProduction', 'COPRPUS'),
        ('CrudeOilRefineryInput', 'CORIPUS'),
        ('CrudeOilStocks', 'COSXPUS'),
        ('HydropowerGenFL', 'HVEPGEN_FL'),
        ('HeatingOilPrice', 'DSWHUUS'),
        ('HeatingOilConsumption', 'DATCPUS'),
        ('HeatingOilStocks', 'DAPSPUS'),
        ('DistillateInventoryAlt', 'DFPSPUS'),
        ('HeatingOilNetImports', 'DFNIPUS'),
        ('HeatingOilSupply', 'DASUPPLY')
    ]
    
    # Fetch and store EIA data
    for metric, series_id in series_configs:
        df = fetch_eia_data(eia_api_key, series_id, start_date, end_date)
        if not df.empty:
            store_eia_data(df, metric, conn)
        time.sleep(2)  # Avoid rate limiting
    
    # Weather data cities and symbols
    weather_configs = [
        ('Houston', 'CL'),
        ('Columbus,us-OH', 'HO')  # Specify state to avoid ambiguity
    ]
    
    # Fetch and store weather data
    for city, symbol in weather_configs:
        weather_data = fetch_weather_data(weather_api_key, city, symbol)
        if weather_data:
            store_weather_data(weather_data, conn)
        time.sleep(1)  # Avoid rate limiting
    
    # Market data symbols
    market_symbols = ['CL=F', 'HO=F']
    
    # Fetch and store market data
    for symbol in market_symbols:
        market_data = fetch_market_data(symbol)
        if market_data:
            store_market_data(market_data, conn)
        time.sleep(1)  # Avoid rate limiting
    
    conn.close()
    print("Live data fetching and storage completed")
    logging.info("Live data fetching and storage completed")

if __name__ == "__main__":
    main()