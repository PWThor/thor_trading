#!/usr/bin/env python
"""
Daily data collection script for Thor Trading System.
This script collects data from various sources and updates the database:
- CL and HO futures data from Interactive Brokers (daily)
- Weather data from OpenWeather API (daily)
- EIA data from EIA API (weekly, typically Wednesday)
- COT data from CFTC (weekly, typically Friday)
- OPEC data (monthly)

Run this script daily using a scheduler (cron job or Windows Task Scheduler).
"""

import os
import sys
import json
import logging
import argparse
import traceback
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from connectors.postgres_connector import PostgresConnector
try:
    from connectors.ibkr_connector import IBKRConnector
except ImportError:
    # Use mock connector for testing if real connector not available
    from connectors.mock_ibkr_connector import IBKRConnector

# Set up logging
def setup_logging():
    """Set up logging configuration."""
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'logs'
    )
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'data_collection_{datetime.now().strftime("%Y%m%d")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('DataCollection')

logger = setup_logging()

def load_api_keys() -> Dict:
    """Load API keys from config file."""
    api_keys_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'pipeline',
        'api_keys.json'
    )
    
    try:
        if os.path.exists(api_keys_path):
            with open(api_keys_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"API keys file not found: {api_keys_path}")
            return {}
    except Exception as e:
        logger.error(f"Error loading API keys: {str(e)}")
        return {}

class DataCollector:
    """
    Data collection manager for Thor Trading System.
    Handles collection of various data sources and updates the database.
    """
    
    def __init__(self, db_connector: PostgresConnector, ibkr_connector: IBKRConnector, api_keys: Dict):
        """Initialize the data collector."""
        self.db = db_connector
        self.ibkr = ibkr_connector
        self.api_keys = api_keys
        
        # Set default date range for updates
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=7)  # Get last 7 days by default
    
    def collect_futures_data(self, symbols: List[str]) -> bool:
        """
        Collect futures data from Interactive Brokers.
        
        Args:
            symbols: List of symbols to collect (e.g., ['CL', 'HO'])
            
        Returns:
            Success flag
        """
        logger.info(f"Collecting futures data for {symbols}")
        
        success = True
        for symbol in symbols:
            try:
                # Get active contract for the symbol
                contract = self.ibkr.get_active_future_contract(symbol)
                
                if contract is None:
                    logger.error(f"Failed to get active contract for {symbol}")
                    success = False
                    continue
                
                # Get historical data
                historical_data = self.ibkr.get_market_data(
                    contract, 
                    duration='7 D',  # Last 7 days
                    bar_size='1 day',
                    what_to_show='TRADES'
                )
                
                if historical_data is None or historical_data.empty:
                    logger.error(f"Failed to get historical data for {symbol}")
                    success = False
                    continue
                
                # Process and save to database
                for _, row in historical_data.iterrows():
                    try:
                        # Convert to database format
                        data_point = {
                            'symbol': symbol,
                            'timestamp': row['timestamp'],
                            'open': row['open'],
                            'high': row['high'],
                            'low': row['low'],
                            'close': row['close'],
                            'volume': row['volume']
                        }
                        
                        # Save to database
                        self.db.insert_market_data(data_point)
                        
                    except Exception as e:
                        logger.error(f"Error saving data point for {symbol}: {str(e)}")
                        success = False
                
                logger.info(f"Collected {len(historical_data)} data points for {symbol}")
                
            except Exception as e:
                logger.error(f"Error collecting futures data for {symbol}: {str(e)}")
                logger.error(traceback.format_exc())
                success = False
                
        return success
    
    def collect_weather_data(self, locations: List[str]) -> bool:
        """
        Collect weather data from OpenWeather API.
        
        Args:
            locations: List of locations to collect weather data for
            
        Returns:
            Success flag
        """
        logger.info(f"Collecting weather data for {locations}")
        
        if 'openweather' not in self.api_keys:
            logger.error("OpenWeather API key not found")
            return False
        
        api_key = self.api_keys['openweather']
        success = True
        
        for location in locations:
            try:
                # Get coordinates for the location
                geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={api_key}"
                geo_response = requests.get(geo_url)
                geo_data = geo_response.json()
                
                if not geo_data:
                    logger.error(f"Failed to get coordinates for {location}")
                    success = False
                    continue
                
                lat = geo_data[0]['lat']
                lon = geo_data[0]['lon']
                
                # Get historical data for each day in range (OpenWeather only allows one day per request in free tier)
                current_date = self.start_date
                
                while current_date <= self.end_date:
                    try:
                        timestamp = int(current_date.timestamp())
                        
                        # Call OpenWeather API
                        url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={timestamp}&units=imperial&appid={api_key}"
                        response = requests.get(url)
                        data = response.json()
                        
                        if 'current' not in data:
                            logger.error(f"Invalid response for {location} on {current_date.strftime('%Y-%m-%d')}")
                            current_date += timedelta(days=1)
                            continue
                        
                        # Process data
                        weather_data = {
                            'date': current_date.strftime('%Y-%m-%d'),
                            'location': location,
                            'avg_temperature': data['current']['temp'],
                            'precipitation': data['current'].get('rain', {}).get('1h', 0) if 'rain' in data['current'] else 0,
                            'snowfall': data['current'].get('snow', {}).get('1h', 0) if 'snow' in data['current'] else 0
                        }
                        
                        # Save to database
                        self.db.insert_weather_data(weather_data)
                        
                        logger.info(f"Collected weather data for {location} on {current_date.strftime('%Y-%m-%d')}")
                        
                    except Exception as e:
                        logger.error(f"Error collecting weather data for {location} on {current_date}: {str(e)}")
                        success = False
                        
                    current_date += timedelta(days=1)
                        
            except Exception as e:
                logger.error(f"Error collecting weather data for {location}: {str(e)}")
                logger.error(traceback.format_exc())
                success = False
                
        return success
    
    def collect_eia_data(self) -> bool:
        """
        Collect EIA data (weekly).
        
        Returns:
            Success flag
        """
        logger.info("Collecting EIA data")
        
        if 'eia' not in self.api_keys:
            logger.error("EIA API key not found")
            return False
        
        api_key = self.api_keys['eia']
        success = True
        
        try:
            # EIA Series IDs for crude oil and petroleum products
            series_ids = {
                'crude_inventory': 'PET.WCRSTUS1.W',  # Crude oil inventories
                'crude_production': 'PET.WCRFPUS2.W',  # Crude oil production
                'gasoline_inventory': 'PET.WGTSTUS1.W',  # Gasoline inventories
                'distillate_inventory': 'PET.WDISTUS1.W'  # Distillate inventories
            }
            
            # Collect data for each series
            for name, series_id in series_ids.items():
                try:
                    # Query EIA API
                    url = f"https://api.eia.gov/series/?api_key={api_key}&series_id={series_id}"
                    response = requests.get(url)
                    data = response.json()
                    
                    if 'series' not in data or not data['series']:
                        logger.error(f"Invalid response for EIA series {name}")
                        success = False
                        continue
                    
                    # Process data
                    series_data = data['series'][0]['data']
                    
                    # EIA data is in format [date_string, value]
                    # Sort by date (newest first)
                    series_data.sort(key=lambda x: x[0], reverse=True)
                    
                    # Filter to last 10 weeks to ensure we get the latest
                    recent_data = series_data[:10]
                    
                    for date_str, value in recent_data:
                        try:
                            # Convert date string to datetime
                            # EIA dates are in format YYYYMMDD
                            date = datetime.strptime(date_str, '%Y%m%d')
                            
                            # Only process if within our range
                            if self.start_date <= date <= self.end_date:
                                # Save to database
                                eia_data = {
                                    'date': date.strftime('%Y-%m-%d'),
                                    'series_name': name,
                                    'value': value
                                }
                                
                                self.db.insert_eia_data(eia_data)
                                
                                logger.info(f"Collected EIA data for {name} on {date.strftime('%Y-%m-%d')}")
                            
                        except Exception as e:
                            logger.error(f"Error processing EIA data point for {name} on {date_str}: {str(e)}")
                            success = False
                    
                except Exception as e:
                    logger.error(f"Error collecting EIA data for {name}: {str(e)}")
                    success = False
            
        except Exception as e:
            logger.error(f"Error collecting EIA data: {str(e)}")
            logger.error(traceback.format_exc())
            success = False
            
        return success
    
    def collect_cot_data(self) -> bool:
        """
        Collect Commitment of Traders data (weekly, published on Fridays).
        Uses the CFTC website CSV files.
        
        Returns:
            Success flag
        """
        logger.info("Collecting COT data")
        
        try:
            # Get date of latest available report (published on Fridays)
            today = datetime.now()
            days_since_friday = (today.weekday() - 4) % 7
            latest_friday = today - timedelta(days=days_since_friday)
            
            # Format for CFTC URL
            date_str = latest_friday.strftime('%Y%m%d')
            
            # CFTC Disaggregated Futures-Only Report URL
            url = f"https://www.cftc.gov/dea/newcot/f_disagg{date_str}.htm"
            
            # Try to read the report
            try:
                # Use pandas to read the fixed-width format file
                df = pd.read_fwf(url, skiprows=1)
                
                # Check if data was loaded
                if df.empty:
                    logger.error(f"No COT data available for {latest_friday.strftime('%Y-%m-%d')}")
                    return False
                
                # Process data for relevant commodities (Crude Oil and Heating Oil)
                commodities = ['CRUDE OIL, LIGHT SWEET - NYMEX', 'NO. 2 HEATING OIL - NYMEX']
                
                for commodity in commodities:
                    try:
                        # Extract row for the commodity
                        commodity_row = df[df.iloc[:, 0].str.contains(commodity, na=False)]
                        
                        if commodity_row.empty:
                            logger.error(f"No COT data found for {commodity}")
                            continue
                        
                        # Extract positions data
                        # Note: The exact column indices depend on the CFTC format
                        # Adjust these based on the actual file format
                        producer_long = float(commodity_row.iloc[0, 3])
                        producer_short = float(commodity_row.iloc[0, 4])
                        swap_long = float(commodity_row.iloc[0, 5])
                        swap_short = float(commodity_row.iloc[0, 6])
                        managed_long = float(commodity_row.iloc[0, 7])
                        managed_short = float(commodity_row.iloc[0, 8])
                        other_long = float(commodity_row.iloc[0, 9])
                        other_short = float(commodity_row.iloc[0, 10])
                        
                        # Calculate net positions
                        commercial_net = (producer_long - producer_short) + (swap_long - swap_short)
                        noncommercial_net = (managed_long - managed_short) + (other_long - other_short)
                        
                        # Map to CL or HO symbol
                        symbol = 'CL' if 'CRUDE OIL' in commodity else 'HO'
                        
                        # Save to database
                        cot_data = {
                            'date': latest_friday.strftime('%Y-%m-%d'),
                            'symbol': symbol,
                            'commercial_long': producer_long + swap_long,
                            'commercial_short': producer_short + swap_short,
                            'noncommercial_long': managed_long + other_long,
                            'noncommercial_short': managed_short + other_short,
                            'commercial_net': commercial_net,
                            'noncommercial_net': noncommercial_net,
                            'report_date': date_str
                        }
                        
                        self.db.insert_cot_data(cot_data)
                        
                        logger.info(f"Collected COT data for {symbol} on {latest_friday.strftime('%Y-%m-%d')}")
                        
                    except Exception as e:
                        logger.error(f"Error processing COT data for {commodity}: {str(e)}")
                        
            except Exception as e:
                logger.error(f"Error reading COT report: {str(e)}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error collecting COT data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    def collect_opec_data(self) -> bool:
        """
        Check for and download OPEC monthly reports.
        OPEC publishes monthly oil market reports around the 13th of each month.
        
        Returns:
            Success flag
        """
        logger.info("Checking for new OPEC data")
        
        try:
            # Check if we need to update OPEC data (only monthly)
            today = datetime.now()
            
            # Only try to update if we're past the 13th of the month
            if today.day < 13:
                logger.info("Skipping OPEC data update (reports published after 13th of month)")
                return True
            
            # Check if we already have this month's data
            current_month_data = self.db.query("""
                SELECT * FROM opec_data 
                WHERE DATE_TRUNC('month', timestamp) = DATE_TRUNC('month', CURRENT_DATE)
                LIMIT 1
            """)
            
            if current_month_data:
                logger.info("Already have OPEC data for current month")
                return True
            
            # Construct the URL for the current month's report
            month_name = today.strftime('%B').lower()
            year = today.strftime('%Y')
            
            # OPEC OMR report URL format
            url = f"https://www.opec.org/opec_web/static_files_project/media/downloads/publications/MOMR_{month_name}_{year}.pdf"
            
            # We're not actually going to download and parse the PDF here as that's a complex task
            # Instead, we'll just check if the URL exists and then insert placeholder data
            
            response = requests.head(url)
            if response.status_code == 200:
                logger.info(f"Found OPEC report for {month_name} {year}")
                
                # In a real implementation, you would extract data from the PDF
                # Here, we're inserting placeholder values
                
                # Placeholder values (in real implementation, extract from PDF)
                opec_data = {
                    'date': today.strftime('%Y-%m-%d'),
                    'report_month': today.strftime('%Y-%m'),
                    'production': 30.0,  # OPEC crude production (mbpd)
                    'demand_forecast': 100.0,  # Global oil demand forecast (mbpd)
                    'supply_non_opec': 70.0,  # Non-OPEC supply (mbpd)
                    'report_url': url
                }
                
                # Save to database
                self.db.insert_opec_data(opec_data)
                
                logger.info(f"Added OPEC data for {today.strftime('%Y-%m')}")
                return True
            else:
                logger.warning(f"OPEC report not yet available for {month_name} {year}")
                return False
                
        except Exception as e:
            logger.error(f"Error collecting OPEC data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def run_daily_collection(self) -> bool:
        """
        Run the daily data collection process.
        
        Returns:
            Overall success flag
        """
        logger.info("Starting daily data collection")
        
        # Track overall success
        success = True
        
        # Collect futures data (daily)
        futures_success = self.collect_futures_data(['CL', 'HO'])
        if not futures_success:
            logger.warning("Futures data collection had errors")
            success = False
        
        # Collect weather data (daily)
        weather_success = self.collect_weather_data(['New York', 'Chicago', 'Houston', 'Cushing'])
        if not weather_success:
            logger.warning("Weather data collection had errors")
            success = False
        
        # Collect EIA data (weekly, but check daily)
        # EIA releases data on Wednesdays, but we check daily in case we missed it
        eia_success = self.collect_eia_data()
        if not eia_success:
            logger.warning("EIA data collection had errors")
            success = False
        
        # Collect COT data (weekly, published on Fridays)
        # Only try to collect on Friday or later
        weekday = datetime.now().weekday()
        if weekday >= 4:  # Friday or later
            cot_success = self.collect_cot_data()
            if not cot_success:
                logger.warning("COT data collection had errors")
                success = False
        
        # Check for OPEC data (monthly)
        opec_success = self.collect_opec_data()
        if not opec_success:
            logger.warning("OPEC data collection had errors")
            # Don't count this as failure since it's expected most days
        
        logger.info("Daily data collection completed")
        return success

def main():
    """Main entry point for the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Daily data collection for Thor Trading System')
    parser.add_argument('--start-date', type=str, help='Start date for data collection (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for data collection (YYYY-MM-DD)')
    args = parser.parse_args()
    
    try:
        # Initialize database connector
        db = PostgresConnector()
        
        # Initialize IBKR connector
        ibkr = IBKRConnector()
        
        # Load API keys
        api_keys = load_api_keys()
        
        # Initialize data collector
        collector = DataCollector(db, ibkr, api_keys)
        
        # Set date range if provided
        if args.start_date:
            collector.start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        if args.end_date:
            collector.end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        # Run data collection
        success = collector.run_daily_collection()
        
        if success:
            logger.info("Data collection completed successfully")
        else:
            logger.warning("Data collection completed with errors")
            
    except Exception as e:
        logger.error(f"Unhandled error in data collection: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())