import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import traceback
from typing import Dict, List, Tuple, Union, Optional, Callable
import schedule
import requests

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connectors.postgres_connector import PostgresConnector
from connectors.ibkr_connector import IBKRConnector

# Handle imports for testing
try:
    from features.feature_generator_wrapper import create_feature_generator
except ImportError:
    print("Warning: Feature generator import failed in data pipeline")


class DataPipeline:
    """
    Automated data pipeline for collecting and storing market data, fundamental data, 
    and alternative data for energy trading.
    """
    
    def __init__(
        self,
        db_connector: PostgresConnector,
        ibkr_connector: Optional[IBKRConnector] = None,
        log_file: str = None,
        weather_api_key: str = None,
        eia_api_key: str = None
    ):
        """
        Initialize the DataPipeline.
        
        Args:
            db_connector: Database connector for storing data
            ibkr_connector: Interactive Brokers connector for market data
            log_file: Path to log file
            weather_api_key: OpenWeatherMap API key
            eia_api_key: EIA API key
        """
        self.db = db_connector
        self.ibkr = ibkr_connector
        
        # Set up logging
        self.log_file = log_file if log_file else os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'logs',
            f'data_pipeline_{datetime.now().strftime("%Y%m%d")}.log'
        )
        
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.logger = logging.getLogger('DataPipeline')
        
        # Add console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        
        # API keys
        self.weather_api_key = weather_api_key
        self.eia_api_key = eia_api_key
        
        # Weather locations important for energy markets
        self.weather_locations = [
            {'city': 'New York', 'country': 'US', 'lat': 40.7128, 'lon': -74.0060},
            {'city': 'Chicago', 'country': 'US', 'lat': 41.8781, 'lon': -87.6298},
            {'city': 'Houston', 'country': 'US', 'lat': 29.7604, 'lon': -95.3698},
            {'city': 'Cushing', 'state': 'OK', 'country': 'US', 'lat': 35.9846, 'lon': -96.7593},
            {'city': 'Boston', 'country': 'US', 'lat': 42.3601, 'lon': -71.0589}
        ]
        
        # EIA data series for energy fundamentals
        self.eia_series = [
            # Crude oil inventories
            'PET.WCESTUS1.W',  # U.S. Ending Stocks of Crude Oil
            'PET.WCRSTUS1.W',  # U.S. Ending Stocks of Crude Oil, excluding SPR
            'PET.MCRFPUS1.M',  # U.S. Field Production of Crude Oil
            
            # Refining
            'PET.WGIRIUS2.W',  # U.S. Percent Utilization of Refinery Operable Capacity
            
            # Heating oil / distillates
            'PET.WDISTUS1.W',  # U.S. Ending Stocks of Distillate Fuel Oil
            'PET.MDIRIPUS3.M',  # U.S. Product Supplied of Distillate Fuel Oil
            
            # Imports/Exports
            'PET.MCRIMUS1.M',  # U.S. Imports of Crude Oil
            'PET.MTTIMUSUS1.M',  # U.S. Imports of Total Petroleum Products
            'PET.MCREXUS1.M',  # U.S. Exports of Crude Oil
            'PET.MTTEXUS1.M'   # U.S. Exports of Total Petroleum Products
        ]
        
        # Schedule for data collection
        self.job_registry = []
    
    def collect_market_data(self) -> bool:
        """
        Collect market data from Interactive Brokers.
        
        Returns:
            Success flag
        """
        try:
            self.logger.info("Collecting market data from Interactive Brokers")
            
            if self.ibkr is None:
                self.logger.error("IBKR connector not initialized")
                return False
            
            # Get CL (Crude Oil) data
            cl_data = self.ibkr.get_historical_data('CL', '1 day')
            
            # Get HO (Heating Oil) data
            ho_data = self.ibkr.get_historical_data('HO', '1 day')
            
            # Calculate the crack spread
            if cl_data is not None and ho_data is not None:
                # Ensure the indices are aligned
                common_dates = cl_data.index.intersection(ho_data.index)
                cl_aligned = cl_data.loc[common_dates]
                ho_aligned = ho_data.loc[common_dates]
                
                # Calculate the crack spread (HO - CL)
                crack_spread = pd.DataFrame(index=common_dates)
                crack_spread['open'] = ho_aligned['open'] - cl_aligned['open']
                crack_spread['high'] = ho_aligned['high'] - cl_aligned['high']
                crack_spread['low'] = ho_aligned['low'] - cl_aligned['low']
                crack_spread['close'] = ho_aligned['close'] - cl_aligned['close']
                crack_spread['volume'] = (ho_aligned['volume'] + cl_aligned['volume']) / 2
                
                # Store data in database
                self.db.store_market_data(cl_data, 'CL')
                self.db.store_market_data(ho_data, 'HO')
                self.db.store_market_data(crack_spread, 'CL-HO-SPREAD')
                
                self.logger.info(f"Successfully collected and stored market data for CL, HO, and crack spread")
                return True
            else:
                self.logger.error("Failed to collect market data from IBKR")
                return False
                
        except Exception as e:
            self.logger.error(f"Error collecting market data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def collect_weather_data(self) -> bool:
        """
        Collect weather data from OpenWeatherMap API.
        
        Returns:
            Success flag
        """
        if not self.weather_api_key:
            self.logger.error("Weather API key not provided")
            return False
            
        try:
            self.logger.info("Collecting weather data from OpenWeatherMap")
            
            for location in self.weather_locations:
                # Construct the API URL
                url = f"https://api.openweathermap.org/data/2.5/weather"
                params = {
                    'lat': location['lat'],
                    'lon': location['lon'],
                    'appid': self.weather_api_key,
                    'units': 'imperial'  # Use imperial units (Fahrenheit)
                }
                
                # Make the API request
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract relevant weather data
                    weather_data = {
                        'location': location['city'],
                        'timestamp': datetime.now(),
                        'temperature': data['main']['temp'],
                        'feels_like': data['main']['feels_like'],
                        'pressure': data['main']['pressure'],
                        'humidity': data['main']['humidity'],
                        'wind_speed': data['wind']['speed'],
                        'weather_main': data['weather'][0]['main'],
                        'weather_description': data['weather'][0]['description']
                    }
                    
                    # Calculate heating/cooling degree days
                    base_temp = 65  # Standard base temperature for HDD/CDD
                    weather_data['hdd'] = max(0, base_temp - data['main']['temp'])
                    weather_data['cdd'] = max(0, data['main']['temp'] - base_temp)
                    
                    # Store in database
                    if hasattr(self.db, 'store_weather_data'):
                        self.db.store_weather_data(weather_data)
                    
                    self.logger.info(f"Collected weather data for {location['city']}")
                else:
                    self.logger.error(f"Failed to collect weather data for {location['city']}: {response.status_code}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error collecting weather data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def collect_eia_data(self) -> bool:
        """
        Collect energy fundamentals data from EIA API.
        
        Returns:
            Success flag
        """
        if not self.eia_api_key:
            self.logger.error("EIA API key not provided")
            return False
            
        try:
            self.logger.info("Collecting EIA data")
            
            for series_id in self.eia_series:
                # Construct the API URL
                url = f"https://api.eia.gov/series/"
                params = {
                    'api_key': self.eia_api_key,
                    'series_id': series_id
                }
                
                # Make the API request
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'series' in data and len(data['series']) > 0:
                        series_data = data['series'][0]
                        
                        # Extract the data and convert to DataFrame
                        if 'data' in series_data and len(series_data['data']) > 0:
                            # EIA data format is [date, value]
                            df = pd.DataFrame(series_data['data'], columns=['date', 'value'])
                            
                            # Convert date string to datetime (format depends on frequency)
                            if 'W' in series_id:  # Weekly data
                                df['date'] = pd.to_datetime(df['date'])
                            elif 'M' in series_id:  # Monthly data
                                df['date'] = pd.to_datetime(df['date'], format='%Y%m')
                            
                            # Set date as index
                            df.set_index('date', inplace=True)
                            
                            # Only keep the last 30 entries (most recent data)
                            df = df.head(30)
                            
                            # Store in database
                            if hasattr(self.db, 'store_eia_data'):
                                self.db.store_eia_data(df, series_id)
                            
                            self.logger.info(f"Collected EIA data for {series_id}")
                        else:
                            self.logger.warning(f"No data available for EIA series {series_id}")
                    else:
                        self.logger.warning(f"No series found for EIA series {series_id}")
                else:
                    self.logger.error(f"Failed to collect EIA data for {series_id}: {response.status_code}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error collecting EIA data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def collect_cot_data(self) -> bool:
        """
        Collect Commitment of Traders (COT) data from CFTC website.
        
        Returns:
            Success flag
        """
        try:
            self.logger.info("Collecting COT data")
            
            # Get current year
            current_year = datetime.now().year
            
            # URLs for COT reports
            url = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{current_year}.zip"
            
            # Create a temporary directory for downloads
            import tempfile
            import zipfile
            
            with tempfile.TemporaryDirectory() as tmpdir:
                # Download the ZIP file
                zip_path = os.path.join(tmpdir, f"cot_{current_year}.zip")
                r = requests.get(url)
                
                if r.status_code == 200:
                    with open(zip_path, 'wb') as f:
                        f.write(r.content)
                    
                    # Extract the ZIP file
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(tmpdir)
                    
                    # Find the extracted text file
                    txt_files = [f for f in os.listdir(tmpdir) if f.endswith('.txt')]
                    
                    if txt_files:
                        txt_path = os.path.join(tmpdir, txt_files[0])
                        
                        # Read and process the COT data
                        # The format is fixed-width, so we need to define the column positions
                        # For simplicity, we'll use pandas read_csv with delimiters
                        # In a production system, this would need to be more robust
                        
                        # This is a simplified version, as the actual COT report format is complex
                        df = pd.read_csv(txt_path, sep=',', header=0)
                        
                        # Filter for crude oil and heating oil
                        df_filtered = df[df['Market_and_Exchange_Names'].str.contains('CRUDE OIL|HEATING OIL', case=False, na=False)]
                        
                        if not df_filtered.empty:
                            # Process and store the data
                            # Convert date column
                            df_filtered['Report_Date_as_MM_DD_YYYY'] = pd.to_datetime(df_filtered['Report_Date_as_MM_DD_YYYY'])
                            df_filtered.set_index('Report_Date_as_MM_DD_YYYY', inplace=True)
                            
                            # Store in database
                            if hasattr(self.db, 'store_cot_data'):
                                self.db.store_cot_data(df_filtered)
                            
                            self.logger.info(f"Successfully collected and processed COT data")
                        else:
                            self.logger.warning("No relevant commodity data found in COT report")
                    else:
                        self.logger.error("No text files found in the extracted COT data")
                else:
                    self.logger.error(f"Failed to download COT data: {r.status_code}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error collecting COT data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def run_daily_data_collection(self) -> bool:
        """
        Run all daily data collection tasks.
        
        Returns:
            Success flag
        """
        self.logger.info("Starting daily data collection")
        
        success = True
        
        # Collect market data
        market_success = self.collect_market_data()
        if not market_success:
            self.logger.warning("Market data collection failed")
            success = False
        
        # Collect weather data
        weather_success = self.collect_weather_data()
        if not weather_success:
            self.logger.warning("Weather data collection failed")
            success = False
        
        # Collect EIA data
        eia_success = self.collect_eia_data()
        if not eia_success:
            self.logger.warning("EIA data collection failed")
            success = False
        
        # Collect COT data (not daily, but we'll check if it's Friday to match CFTC schedule)
        if datetime.now().weekday() == 4:  # Friday
            cot_success = self.collect_cot_data()
            if not cot_success:
                self.logger.warning("COT data collection failed")
                success = False
        
        self.logger.info("Daily data collection completed")
        return success
    
    def schedule_daily_collection(self, time_str: str = "18:00") -> None:
        """
        Schedule daily data collection task.
        
        Args:
            time_str: Time to run the job in 24-hour format (HH:MM)
        """
        job = schedule.every().day.at(time_str).do(self.run_daily_data_collection)
        self.job_registry.append(job)
        self.logger.info(f"Scheduled daily data collection at {time_str}")
    
    def schedule_job(self, job_func: Callable, schedule_str: str) -> None:
        """
        Schedule a custom job.
        
        Args:
            job_func: Function to run
            schedule_str: Schedule string (e.g., 'day.at("18:00")', 'hour', 'minutes.at(":30")')
        """
        # This is a simplified approach. In a production system, you would want more flexibility.
        if "day.at" in schedule_str:
            time_str = schedule_str.split('"')[1]
            job = schedule.every().day.at(time_str).do(job_func)
        elif "hour" in schedule_str:
            job = schedule.every().hour.do(job_func)
        elif "minute" in schedule_str:
            job = schedule.every().minute.do(job_func)
        else:
            raise ValueError(f"Unsupported schedule string: {schedule_str}")
            
        self.job_registry.append(job)
        self.logger.info(f"Scheduled custom job with schedule: {schedule_str}")
    
    def run_scheduler(self) -> None:
        """
        Run the scheduler in an infinite loop.
        """
        self.logger.info("Starting scheduler")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"Scheduler error: {str(e)}")
            self.logger.error(traceback.format_exc())


if __name__ == "__main__":
    # Example usage
    db = PostgresConnector()
    
    try:
        from connectors.ibkr_connector import IBKRConnector
        ibkr = IBKRConnector()
    except ImportError:
        print("IBKR connector not available. Market data collection will be disabled.")
        ibkr = None
    
    # Initialize pipeline
    pipeline = DataPipeline(
        db_connector=db,
        ibkr_connector=ibkr,
        weather_api_key="YOUR_OPENWEATHER_API_KEY",
        eia_api_key="YOUR_EIA_API_KEY"
    )
    
    # Schedule daily data collection
    pipeline.schedule_daily_collection("18:00")
    
    # Run the scheduler
    pipeline.run_scheduler()