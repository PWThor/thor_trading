#!/usr/bin/env python
# Enhanced database connector for backtesting with all data sources
from datetime import datetime
import pandas as pd
from connectors.postgres_connector import PostgresConnector
import logging

logger = logging.getLogger(__name__)

class BacktestDBConnector(PostgresConnector):
    """
    Enhanced database connector for backtesting that integrates all data sources.
    This connector provides methods to access all data tables and presents them
    in the format expected by the feature generator.
    """
    
    def __init__(self):
        """Initialize with parent PostgresConnector"""
        super().__init__()
        
    def get_market_data(self, symbol, start_date, end_date):
        """Get market data from the market_data_backtest view"""
        try:
            # Always use the basic view directly
            query = """
                SELECT * FROM market_data_backtest
                WHERE symbol = %s AND date BETWEEN %s AND %s
                ORDER BY date
            """
            logger.info(f"Using market_data_backtest view for {symbol}")
            
            data = self.query(query, [symbol, start_date, end_date])
            
            if not data:
                logger.warning(f"No market data found for {symbol} between {start_date} and {end_date}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Set date as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return pd.DataFrame()
            
    def get_available_symbols(self):
        """Get all available symbols from the market_data_backtest view"""
        try:
            query = "SELECT DISTINCT symbol FROM market_data_backtest"
            symbols = self.query(query)
            return [row['symbol'] for row in symbols]
        except Exception as e:
            logger.error(f"Error getting available symbols: {str(e)}")
            return []
            
    def get_market_data_range(self, symbol, start_date, end_date):
        """Alias for get_market_data for compatibility with engine.py"""
        return self.get_market_data(symbol, start_date, end_date)
            
    def get_weather_data_range(self, start_date, end_date):
        """Get weather data in the format expected by the feature generator"""
        try:
            # Check if weather_data table exists
            table_exists = self.query_one(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'weather_data')"
            )
            
            if not table_exists or not table_exists.get('exists', False):
                logger.warning("weather_data table does not exist")
                return None
                
            # Get the column names
            columns = self.query(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'weather_data'"
            )
            column_names = [col['column_name'] for col in columns]
            
            # Adjust query based on available columns
            date_col = 'timestamp' if 'timestamp' in column_names else 'date'
            
            # Check for location column
            if 'location' in column_names:
                query = f"""
                    SELECT {date_col} as date, location, 
                           avg_temperature, precipitation, snowfall
                    FROM weather_data
                    WHERE {date_col} BETWEEN %s AND %s
                """
            else:
                # If no location column, use city-specific columns
                temp_cols = [col for col in column_names if 'temperature' in col]
                precip_cols = [col for col in column_names if 'precipitation' in col]
                snow_cols = [col for col in column_names if 'snowfall' in col]
                
                if not temp_cols:
                    logger.warning("No temperature columns found in weather_data")
                    return None
                    
                # Create a query that unpivots city-specific columns
                # This is a simplification - adjust based on your actual schema
                select_parts = [f"{date_col} as date"]
                for city in ['nyc', 'chicago', 'houston', 'cushing']:
                    temp_col = next((col for col in temp_cols if city in col.lower()), None)
                    precip_col = next((col for col in precip_cols if city in col.lower()), None)
                    snow_col = next((col for col in snow_cols if city in col.lower()), None)
                    
                    if temp_col:
                        select_parts.append(f"'{city.capitalize()}' as location")
                        select_parts.append(f"{temp_col} as avg_temperature")
                        select_parts.append(f"{precip_col or 'NULL'} as precipitation")
                        select_parts.append(f"{snow_col or 'NULL'} as snowfall")
                
                query = f"""
                    SELECT {date_col} as date, 
                           {', '.join(select_parts)}
                    FROM weather_data
                    WHERE {date_col} BETWEEN %s AND %s
                """
            
            weather_data = self.query(query, [start_date, end_date])
            
            if not weather_data:
                logger.warning(f"No weather data found between {start_date} and {end_date}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(weather_data)
            
            # Ensure date is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting weather data: {str(e)}")
            return None
    
    def get_eia_data_range(self, start_date, end_date):
        """Get EIA data for the date range"""
        try:
            # Check if eia_data table exists
            table_exists = self.query_one(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'eia_data')"
            )
            
            if not table_exists or not table_exists.get('exists', False):
                logger.warning("eia_data table does not exist")
                return None
                
            # Get the column names to determine schema
            columns = self.query(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'eia_data'"
            )
            column_names = [col['column_name'] for col in columns]
            
            # Adjust query based on available columns
            date_col = 'timestamp' if 'timestamp' in column_names else 'date'
            
            query = f"""
                SELECT * FROM eia_data
                WHERE {date_col} BETWEEN %s AND %s
            """
            
            eia_data = self.query(query, [start_date, end_date])
            
            if not eia_data:
                logger.warning(f"No EIA data found between {start_date} and {end_date}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(eia_data)
            
            # Ensure date is datetime and set as index
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting EIA data: {str(e)}")
            return None
    
    def get_opec_data_range(self, start_date, end_date):
        """Get OPEC data for the date range"""
        try:
            # Check if opec_data table exists
            table_exists = self.query_one(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'opec_data')"
            )
            
            if not table_exists or not table_exists.get('exists', False):
                logger.warning("opec_data table does not exist")
                return None
                
            # Get the column names to determine schema
            columns = self.query(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'opec_data'"
            )
            column_names = [col['column_name'] for col in columns]
            
            # Adjust query based on available columns
            date_col = 'timestamp' if 'timestamp' in column_names else 'date'
            
            query = f"""
                SELECT * FROM opec_data
                WHERE {date_col} BETWEEN %s AND %s
            """
            
            opec_data = self.query(query, [start_date, end_date])
            
            if not opec_data:
                logger.warning(f"No OPEC data found between {start_date} and {end_date}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(opec_data)
            
            # Ensure date is datetime and set as index
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting OPEC data: {str(e)}")
            return None
    
    def get_cot_data_range(self, start_date, end_date):
        """Get Commitment of Traders data for the date range"""
        try:
            # Check if cot_data table exists
            table_exists = self.query_one(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'cot_data')"
            )
            
            if not table_exists or not table_exists.get('exists', False):
                logger.warning("cot_data table does not exist")
                return None
                
            # Get the column names to determine schema
            columns = self.query(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'cot_data'"
            )
            column_names = [col['column_name'] for col in columns]
            
            # Adjust query based on available columns
            date_col = 'timestamp' if 'timestamp' in column_names else 'date'
            
            query = f"""
                SELECT * FROM cot_data
                WHERE {date_col} BETWEEN %s AND %s
            """
            
            cot_data = self.query(query, [start_date, end_date])
            
            if not cot_data:
                logger.warning(f"No COT data found between {start_date} and {end_date}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(cot_data)
            
            # Ensure date is datetime and set as index
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting COT data: {str(e)}")
            return None
            
    # Alias method for backward compatibility
    def get_fundamental_data_range(self, start_date, end_date):
        """Alias that combines EIA and OPEC data as 'fundamental' data"""
        try:
            # Get EIA data
            eia_data = self.get_eia_data_range(start_date, end_date)
            
            # Get OPEC data
            opec_data = self.get_opec_data_range(start_date, end_date)
            
            # Combine if both exist
            if eia_data is not None and opec_data is not None:
                # Merge on index (date)
                combined = pd.merge(
                    eia_data, 
                    opec_data,
                    left_index=True,
                    right_index=True,
                    how='outer'
                )
                return combined
                
            # Return whichever one exists
            return eia_data if eia_data is not None else opec_data
            
        except Exception as e:
            logger.error(f"Error getting fundamental data: {str(e)}")
            return None
            
    # For now, alias alternative_data to COT data
    def get_alternative_data_range(self, start_date, end_date):
        """Alias for get_cot_data_range"""
        return self.get_cot_data_range(start_date, end_date)