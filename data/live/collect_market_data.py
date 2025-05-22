# scripts/data_collection/collect_market_data.py

import sys
import os
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
import requests
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from connectors.postgres_connector import PostgresConnector
from utils.config_loader import ConfigLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/market_data_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_eia_data(api_key, series_id):
    """
    Fetch data from EIA API
    
    Args:
        api_key (str): EIA API key
        series_id (str): EIA series ID
        
    Returns:
        pd.DataFrame: DataFrame with EIA data
    """
    url = f"https://api.eia.gov/series/?api_key={api_key}&series_id={series_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        if 'series' not in data or len(data['series']) == 0:
            logger.error(f"No data returned for series ID {series_id}")
            return pd.DataFrame()
        
        # Extract data points
        series_data = data['series'][0]['data']
        
        # Create DataFrame
        df = pd.DataFrame(series_data, columns=['date', 'value'])
        
        # Convert date string to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Add series ID as column
        df['series_id'] = series_id
        
        return df
    except Exception as e:
        logger.error(f"Error fetching EIA data: {e}")
        return pd.DataFrame()

def fetch_weather_data(api_key, city, start_date, end_date):
    """
    Fetch historical weather data
    
    Args:
        api_key (str): OpenWeather API key
        city (str): City name
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        
    Returns:
        pd.DataFrame: DataFrame with weather data
    """
    # Convert dates to Unix timestamps
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
    
    # Get coordinates for the city
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"
    
    try:
        geo_response = requests.get(geo_url)
        geo_response.raise_for_status()
        
        geo_data = geo_response.json()
        
        if not geo_data:
            logger.error(f"No geo data found for city {city}")
            return pd.DataFrame()
        
        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
        
        # Fetch historical data
        weather_url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={lon}&dt={start_ts}&appid={api_key}&units=imperial"
        
        weather_response = requests.get(weather_url)
        weather_response.raise_for_status()
        
        weather_data = weather_response.json()
        
        if 'data' not in weather_data:
            logger.error(f"No weather data found for {city}")
            return pd.DataFrame()
        
        # Extract data
        records = []
        
        for day_data in weather_data['data']:
            record = {
                'date': datetime.fromtimestamp(day_data['dt']),
                'location': city,
                'avg_temperature': day_data['temp'],
                'humidity': day_data['humidity'],
                'precipitation': day_data.get('rain', {}).get('1h', 0) if 'rain' in day_data else 0,
                'wind_speed': day_data['wind_speed']
            }
            
            records.append(record)
        
        return pd.DataFrame(records)
    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")
        return pd.DataFrame()

def store_eia_data(db, df):
    """
    Store EIA data in database
    
    Args:
        db: Database connector
        df (pd.DataFrame): DataFrame with EIA data
        
    Returns:
        bool: Success status
    """
    if df.empty:
        logger.warning("No EIA data to store")
        return False
    
    try:
        # Convert DataFrame to list of tuples
        data_tuples = [
            (row['date'], row['series_id'], row['value'], 'EIA')
            for _, row in df.iterrows()
        ]
        
        # Use execute_many for bulk insert
        return db.execute_many(
            """
            INSERT INTO eia_data (period, series_id, value, unit)
            VALUES %s
            ON CONFLICT (period, series_id) 
            DO UPDATE SET
                value = EXCLUDED.value
            """,
            data_tuples
        )
    except Exception as e:
        logger.error(f"Error storing EIA data: {e}")
        return False

def store_weather_data(db, df):
    """
    Store weather data in database
    
    Args:
        db: Database connector
        df (pd.DataFrame): DataFrame with weather data
        
    Returns:
        bool: Success status
    """
    if df.empty:
        logger.warning("No weather data to store")
        return False
    
    try:
        # Convert DataFrame to list of tuples
        data_tuples = [
            (row['date'], row['location'], row['avg_temperature'], 
             row['precipitation'], 0, row['wind_speed'], 
             row['humidity'], 0)
            for _, row in df.iterrows()
        ]
        
        # Use execute_many for bulk insert
        return db.execute_many(
            """
            INSERT INTO weather_data (date, location, avg_temperature, precipitation, 
                                      snowfall, wind_speed, humidity, cloud_coverage)
            VALUES %s
            ON CONFLICT (date, location) 
            DO UPDATE SET
                avg_temperature = EXCLUDED.avg_temperature,
                precipitation = EXCLUDED.precipitation,
                wind_speed = EXCLUDED.wind_speed,
                humidity = EXCLUDED.humidity
            """,
            data_tuples
        )
    except Exception as e:
        logger.error(f"Error storing weather data: {e}")
        return False

def main():
    # Load configuration
    config = ConfigLoader().load_config()
    
    # Connect to database
    db_config = config.get('database', {})
    db = PostgresConnector(
        host=db_config.get('host', 'localhost'),
        port=db_config.get('port', 5432),
        database=db_config.get('database', 'trading_db'),
        user=db_config.get('user', 'postgres'),
        password=db_config.get('password', '')
    )
    
    # Test connection
    if not db.test_connection():
        logger.error("Database connection failed")
        return
    
    # Get API keys
    api_keys = config.get('api_keys', {})
    eia_api_key = api_keys.get('eia')
    openweather_api_key = api_keys.get('openweather')
    
    # Get date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    # Fetch and store EIA data
    if eia_api_key:
        logger.info("Fetching EIA data...")
        
        # Crude oil inventory
        crude_inventory_id = 'PET.WCRSTUS1.W'
        crude_inventory_df = fetch_eia_data(eia_api_key, crude_inventory_id)
        
        if not crude_inventory_df.empty:
            logger.info(f"Fetched {len(crude_inventory_df)} EIA crude inventory records")
            store_eia_data(db, crude_inventory_df)
        
        # Crude oil production
        crude_production_id = 'PET.WCRFPUS2.W'
        crude_production_df = fetch_eia_data(eia_api_key, crude_production_id)
        
        if not crude_production_df.empty:
            logger.info(f"Fetched {len(crude_production_df)} EIA crude production records")
            store_eia_data(db, crude_production_df)
    else:
        logger.warning("EIA API key not found, skipping EIA data collection")
    
    # Fetch and store weather data
    if openweather_api_key:
        logger.info("Fetching weather data...")
        
        # Locations relevant to oil markets
        locations = ['New York', 'Chicago', 'Houston', 'Cushing,us-OK']
        
        for location in locations:
            weather_df = fetch_weather_data(openweather_api_key, location, start_date, end_date)
            
            if not weather_df.empty:
                logger.info(f"Fetched {len(weather_df)} weather records for {location}")
                store_weather_data(db, weather_df)
    else:
        logger.warning("OpenWeather API key not found, skipping weather data collection")
    
    logger.info("Data collection completed")

if __name__ == "__main__":
    main()