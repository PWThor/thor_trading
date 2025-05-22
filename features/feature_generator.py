# features/feature_generator.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureGenerator:
    """Feature generator for creating features from raw data"""
    
    def __init__(self, config):
        """Initialize feature generator with configuration"""
        self.config = config
    
    def generate_features(self, market_data):
        """
        Generate features for energy futures prediction, specialized for crude oil and heating oil.
        
        Args:
            market_data: Market data DataFrame with OHLCV data
            
        Returns:
            DataFrame with generated features
        """
        logger.info(f"Generating features from {len(market_data)} rows of market data")
        
        # Ensure we have data
        if market_data is None or len(market_data) == 0:
            logger.error("No market data provided")
            return None
        
        try:
            # Create copy to avoid modifying original dataframe
            df = market_data.copy()
            
            # Ensure we have the basic OHLCV columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}. Features may be limited.")
            
            # Technical indicator features (work with any market data)
            logger.info("Generating technical features")
            
            # Price-based features (use close price by default)
            price_col = 'close'
            
            # Moving averages
            for window in [5, 10, 20, 30, 50, 90, 200]:
                if window < len(df):
                    df[f'sma_{window}'] = df[price_col].rolling(window=window).mean()
            
            # Exponential moving averages
            for window in [5, 10, 20, 50, 200]:
                if window < len(df):
                    df[f'ema_{window}'] = df[price_col].ewm(span=window, adjust=False).mean()
            
            # Standard deviations / volatility
            for window in [10, 20, 30, 60]:
                if window < len(df):
                    df[f'std_{window}'] = df[price_col].rolling(window=window).std()
                    # Normalize volatility by price level
                    if f'sma_{window}' in df.columns:
                        df[f'volatility_{window}'] = df[f'std_{window}'] / df[f'sma_{window}']
            
            # Z-score (normalized price)
            if 'std_20' in df.columns and 'sma_20' in df.columns:
                df['z_score'] = (df[price_col] - df['sma_20']) / df['std_20']
            
            # Momentum/Rate of Change features
            for period in [1, 5, 10, 20]:
                if period < len(df):
                    df[f'roc_{period}'] = df[price_col].pct_change(period) * 100  # Percentage change
            
            # Trend indicators
            # Price distance from moving averages
            for ma in [20, 50, 200]:
                if f'sma_{ma}' in df.columns:
                    df[f'dist_sma_{ma}'] = (df[price_col] / df[f'sma_{ma}'] - 1) * 100  # Percentage distance
            
            # Moving average crossovers (binary signals)
            if 'sma_10' in df.columns and 'sma_20' in df.columns:
                df['sma_10_20_cross'] = np.where(df['sma_10'] > df['sma_20'], 1, -1)
            if 'sma_50' in df.columns and 'sma_200' in df.columns:
                df['sma_50_200_cross'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)
            
            # OHLCV-specific features (if all data available)
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # Range features
                df['daily_range'] = df['high'] - df['low']
                df['daily_range_pct'] = (df['high'] - df['low']) / df['close'] * 100
                
                # Gap features
                df['gap'] = df['open'] - df['close'].shift(1)
                df['gap_pct'] = (df['open'] / df['close'].shift(1) - 1) * 100
                
                # Candlestick features
                df['body_size'] = abs(df['close'] - df['open'])
                df['body_size_pct'] = df['body_size'] / df['close'] * 100
                df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
                df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
                
                # True range and Average True Range (ATR)
                df['tr'] = np.maximum.reduce([
                    df['high'] - df['low'],
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                ])
                
                for window in [7, 14, 21]:
                    if window < len(df):
                        df[f'atr_{window}'] = df['tr'].rolling(window=window).mean()
            
            # Volume-based features (if volume data available)
            if 'volume' in df.columns:
                # Volume moving averages
                for window in [5, 10, 20]:
                    if window < len(df):
                        df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
                
                # Volume relative to moving average
                if 'volume_sma_20' in df.columns:
                    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
                
                # On-balance volume (OBV)
                df['obv_signal'] = np.where(df['close'] > df['close'].shift(1), 1, 
                                           np.where(df['close'] < df['close'].shift(1), -1, 0))
                df['obv'] = (df['volume'] * df['obv_signal']).cumsum()
            
            # Oscillator indicators
            # Relative Strength Index (RSI)
            for window in [7, 14, 21]:
                if window < len(df):
                    # Calculate daily changes
                    delta = df[price_col].diff()
                    
                    # Separate gains and losses
                    gains = delta.where(delta > 0, 0)
                    losses = -delta.where(delta < 0, 0)
                    
                    # Calculate average gains and losses
                    avg_gain = gains.rolling(window=window).mean()
                    avg_loss = losses.rolling(window=window).mean()
                    
                    # Calculate RS and RSI
                    rs = avg_gain / avg_loss
                    df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
            # MACD (Moving Average Convergence Divergence)
            if len(df) >= 26:  # Need at least 26 periods for MACD
                ema_12 = df[price_col].ewm(span=12, adjust=False).mean()
                ema_26 = df[price_col].ewm(span=26, adjust=False).mean()
                df['macd'] = ema_12 - ema_26
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Calendar/Seasonal features (work with any time series)
            logger.info("Generating calendar features")
            
            if pd.api.types.is_datetime64_any_dtype(df.index):
                # Extract date components
                df['year'] = df.index.year
                df['month'] = df.index.month
                df['day'] = df.index.day
                df['day_of_week'] = df.index.dayofweek
                df['day_of_year'] = df.index.dayofyear
                df['quarter'] = df.index.quarter
                df['week_of_year'] = df.index.isocalendar().week
                
                # Create seasonal indicators
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
                df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
                df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
                
                # Season encoding
                # Winter (Dec-Feb), Spring (Mar-May), Summer (Jun-Aug), Fall (Sep-Nov)
                df['winter'] = df['month'].isin([12, 1, 2]).astype(int)
                df['spring'] = df['month'].isin([3, 4, 5]).astype(int)
                df['summer'] = df['month'].isin([6, 7, 8]).astype(int)
                df['fall'] = df['month'].isin([9, 10, 11]).astype(int)
            
            # Energy market specific features
            if 'symbol' in df.columns:
                # Add features specific to energy markets
                if any(s in str(df['symbol'].iloc[0]) for s in ['CL', 'HO', 'NG']):
                    logger.info("Adding energy market specific features")
                    
                    # Add heating/cooling degree days features if weather data present
                    if any(col.startswith('temperature') for col in df.columns):
                        # Calculate HDD/CDD based on standard 65 deg F base
                        temp_cols = [col for col in df.columns if col.startswith('temperature')]
                        for temp_col in temp_cols:
                            df[f'hdd_{temp_col}'] = df[temp_col].apply(lambda x: max(0, 65 - x) if pd.notnull(x) else np.nan)
                            df[f'cdd_{temp_col}'] = df[temp_col].apply(lambda x: max(0, x - 65) if pd.notnull(x) else np.nan)
                    
                    # Add seasonality features specific to energy
                    # Heating season flag (October-March)
                    if 'month' in df.columns:
                        df['heating_season'] = df['month'].apply(lambda m: 1 if m in [10, 11, 12, 1, 2, 3] else 0)
                    
                    # Handle crack spread specific features
                    if 'CL-HO-SPREAD' in str(df['symbol'].iloc[0]):
                        logger.info("Adding crack spread specific features")
                        
                        # Historical spread relationships
                        if price_col in df.columns:
                            # Spread mean-reversion features
                            for window in [30, 60, 90, 180]:
                                if window < len(df):
                                    # Calculate historical percentiles
                                    df[f'spread_percentile_{window}d'] = df[price_col].rolling(window=window).apply(
                                        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                                    )
                            
                            # Spread momentum features
                            for period in [5, 10, 20]:
                                if period < len(df) and f'roc_{period}' in df.columns:
                                    # Spread acceleration (change in momentum)
                                    df[f'spread_accel_{period}d'] = df[f'roc_{period}'].diff()
            
            # Create target variables for different prediction horizons
            for horizon in [1, 3, 5, 10]:
                if horizon < len(df):
                    # Price change
                    df[f'price_change_{horizon}d'] = df[price_col].shift(-horizon) - df[price_col]
                    
                    # Percentage change
                    df[f'return_{horizon}d'] = df[price_col].pct_change(-horizon) * 100
                    
                    # Direction (up/down)
                    # Handle NaN or infinite values before converting to int
                    price_change = df[f'price_change_{horizon}d'].copy()
                    price_change = price_change.fillna(0)  # Replace NaN with 0
                    
                    # Safe conversion to direction
                    df[f'direction_{horizon}d'] = np.sign(price_change).fillna(0).astype(int)
                    
                    # Replace -1 with 0 (binary classification instead of ternary with 0)
                    df[f'direction_{horizon}d'] = df[f'direction_{horizon}d'].replace(-1, 0)
            
            # Clean up any duplicate columns that might have been created
            df = df.loc[:, ~df.columns.duplicated()]
            
            # Ensure date and symbol columns are properly handled
            # XGBoost doesn't like datetime or object types
            if 'timestamp' in df.columns:
                df = df.drop(columns=['timestamp'])
                
            if 'date' in df.columns and pd.api.types.is_datetime64_dtype(df['date']):
                # Convert to integer for ML (days since epoch)
                df['days_since_epoch'] = (df['date'] - pd.Timestamp('1970-01-01')).dt.days
                df = df.drop(columns=['date'])
                
            if 'symbol' in df.columns and pd.api.types.is_object_dtype(df['symbol']):
                # One-hot encode symbols if there are multiple
                symbols = df['symbol'].unique()
                if len(symbols) > 1:
                    for sym in symbols:
                        df[f'symbol_{sym}'] = (df['symbol'] == sym).astype(int)
                df = df.drop(columns=['symbol'])
            
            # Handle missing values to avoid conversion errors
            logger.info("Handling missing values in generated features")
            
            # Replace any infinities
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill some NAs for time series continuity
            df = df.ffill()
            
            # Fill remaining NAs with appropriate values by column type
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            # For critical columns (OHLCV), missing data is more serious
            critical_cols = ['open', 'high', 'low', 'close', 'volume']
            critical_missing = [col for col in critical_cols if col in df.columns and df[col].isna().any()]
            
            if critical_missing:
                logger.warning(f"Missing data in critical columns: {critical_missing}")
                
                # Forward fill critical cols
                for col in critical_missing:
                    df[col] = df[col].interpolate(method='linear').ffill().bfill()
            
            # Fill other numeric columns with 0
            for col in numeric_cols:
                if col not in critical_cols:
                    na_count = df[col].isna().sum()
                    if na_count > 0:
                        logger.debug(f"Filling {na_count} NaN values in {col} with 0")
                        df[col] = df[col].fillna(0)
            
            # Fill categorical columns with 'unknown'
            for col in categorical_cols:
                na_count = df[col].isna().sum()
                if na_count > 0:
                    logger.debug(f"Filling {na_count} NaN values in {col} with 'unknown'")
                    df[col] = df[col].fillna('unknown')
            
            # If any NaNs remain, warn but don't drop rows
            na_count = df.isna().sum().sum()
            if na_count > 0:
                logger.warning(f"Found {na_count} NaN values after attempted cleaning")
                # Get columns with NaNs
                na_cols = [col for col in df.columns if df[col].isna().any()]
                logger.warning(f"Columns with NaNs: {na_cols}")
                
                # If there are still missing values in critical columns, drop those rows
                critical_na = [col for col in critical_cols if col in df.columns and df[col].isna().any()]
                if critical_na:
                    logger.warning(f"Dropping rows with missing values in critical columns: {critical_na}")
                    df_clean = df.dropna(subset=critical_cols)
                else:
                    # Otherwise, fill remaining NaNs with 0
                    logger.warning("Filling remaining NaN values with 0 to preserve rows")
                    df_clean = df.fillna(0)
            else:
                df_clean = df
            
            logger.info(f"Generated {len(df.columns)} features, {len(df_clean)} clean rows of {len(df)} original rows")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error generating features: {e}")
            return None
    
    def add_weather_features(self, df, weather_data):
        """Add weather features to the dataset"""
        logger.info("Adding weather features")
        
        # Merge weather data with price data
        if weather_data is not None and not weather_data.empty:
            # Ensure date columns are in same format
            weather_data['date'] = pd.to_datetime(weather_data['date']).dt.date
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Group weather data by date and location
            nyc_weather = weather_data[weather_data['location'] == 'New York']
            chicago_weather = weather_data[weather_data['location'] == 'Chicago']
            
            # Create separate dataframes for NYC and Chicago
            nyc_weather = nyc_weather.groupby('date').agg({
                'avg_temperature': 'mean',
                'precipitation': 'sum',
                'snowfall': 'sum'
            }).reset_index()
            
            chicago_weather = chicago_weather.groupby('date').agg({
                'avg_temperature': 'mean',
                'precipitation': 'sum',
                'snowfall': 'sum'
            }).reset_index()
            
            # Rename columns
            nyc_weather = nyc_weather.rename(columns={
                'avg_temperature': 'nyc_temp',
                'precipitation': 'nyc_precip',
                'snowfall': 'nyc_snow'
            })
            
            chicago_weather = chicago_weather.rename(columns={
                'avg_temperature': 'chi_temp',
                'precipitation': 'chi_precip',
                'snowfall': 'chi_snow'
            })
            
            # Merge with price data
            df = pd.merge(df, nyc_weather, on='date', how='left')
            df = pd.merge(df, chicago_weather, on='date', how='left')
            
            # Calculate heating degree days (HDD)
            base_temp = 65  # Standard base temperature for HDD calculation
            df['nyc_hdd'] = df['nyc_temp'].apply(lambda x: max(0, base_temp - x) if pd.notnull(x) else np.nan)
            df['chi_hdd'] = df['chi_temp'].apply(lambda x: max(0, base_temp - x) if pd.notnull(x) else np.nan)
            df['avg_hdd'] = (df['nyc_hdd'] + df['chi_hdd']) / 2
            
            logger.info(f"Added {len(df.columns) - len(chicago_weather.columns)} weather features")
        
        return df