import os
import datetime
import logging
import requests
import pandas as pd
import psycopg2
import numpy as np
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData
import time

# Set up logging
logging.basicConfig(
    filename=r'E:\Projects\thor_trading\outputs\logs\update_trading_db_daily.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
TEMP_DIR = r'E:\Projects\thor_trading\temp'
EIA_API_KEY = "2GxRjQeUCEzf5sezwOJOWdP8cVXsKvVWOerU29cY"
WEATHER_API_KEY = "d321b2a2f6d312616cba1d4711e31994"  # Replace with your OpenWeatherMap API key
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

# Database connection parameters
DB_PARAMS = {
    'dbname': 'trading_db',
    'user': 'postgres',
    'password': 'Makingmoney25!',
    'host': 'localhost',
    'port': '5432'
}

class IBClient(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []
        self.positions = []
        self.portfolio = []
        self.sync_done = False

    def historicalData(self, reqId: int, bar: BarData):
        self.data.append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        super().historicalDataEnd(reqId, start, end)
        logging.info(f"Historical data fetch completed for reqId {reqId}")

    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        self.positions.append({
            'account': account,
            'contract': contract,
            'position': position,
            'avgCost': avgCost
        })

    def updatePortfolio(self, contract: Contract, position: float, marketPrice: float, marketValue: float,
                        averageCost: float, unrealizedPNL: float, realizedPNL: float, accountName: str):
        self.portfolio.append({
            'contract': contract,
            'position': position,
            'marketPrice': marketPrice,
            'marketValue': marketValue,
            'averageCost': averageCost,
            'unrealizedPNL': unrealizedPNL,
            'realizedPNL': realizedPNL,
            'account': accountName
        })

    def positionEnd(self):
        super().positionEnd()
        self.sync_done = True

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        logging.error(f"Error {errorCode}, reqId {reqId}: {errorString}")
        if advancedOrderRejectJson:
            logging.error(f"Advanced order reject: {advancedOrderRejectJson}")

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.start()

    def start(self):
        self.reqPositions()
        for i, symbol in enumerate(['CL', 'HO']):
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "FUT"
            contract.exchange = "NYMEX"
            contract.currency = "USD"
            contract.lastTradeDateOrContractMonth = "202506"  # Specify expiration (June 2025)
            self.reqHistoricalData(i + 1, contract, datetime.datetime.now().strftime("%Y%m%d %H:%M:%S"),
                                   "1 D", "1 day", "TRADES", 1, 1, False, [])

    def stop(self):
        self.done = True
        self.disconnect()

def connect_to_db():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        logging.info("Database connection successful")
        return conn
    except Exception as e:
        logging.error(f"Error connecting to database: {str(e)}")
        raise

def insert_data(conn, table, data):
    cursor = conn.cursor()
    if table == "cot_data":
        query = """
        INSERT INTO cot_data (report_date, symbol, comm_positions_long, comm_positions_short, noncomm_positions_long, noncomm_positions_short, net_positions)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (report_date, symbol) DO UPDATE
        SET comm_positions_long = EXCLUDED.comm_positions_long,
            comm_positions_short = EXCLUDED.comm_positions_short,
            noncomm_positions_long = EXCLUDED.noncomm_positions_long,
            noncomm_positions_short = EXCLUDED.noncomm_positions_short,
            net_positions = EXCLUDED.net_positions
        """
        cursor.execute(query, data)
        logging.info(f"Inserted COT data for {data[1]} on {data[0]}")
    elif table == "market_data":
        query = """
        INSERT INTO market_data (timestamp, symbol, open, high, low, close, volume, eia_inventory, eia_production, spr_stocks, opec_production, opec_demand_forecast, cot_commercial_net, cot_noncommercial_net, weather_data)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (timestamp, symbol) DO UPDATE
        SET open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            eia_inventory = EXCLUDED.eia_inventory,
            eia_production = EXCLUDED.eia_production,
            spr_stocks = EXCLUDED.spr_stocks,
            opec_production = EXCLUDED.opec_production,
            opec_demand_forecast = EXCLUDED.opec_demand_forecast,
            cot_commercial_net = EXCLUDED.cot_commercial_net,
            cot_noncommercial_net = EXCLUDED.cot_noncommercial_net,
            weather_data = EXCLUDED.weather_data
        """
        cursor.execute(query, data)
        logging.info(f"Inserted market data for {data[1]} on {data[0]}")
    elif table == "ml_features":
        query = """
        INSERT INTO ml_features (timestamp, symbol, macd, rsi, sma10, sma50)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (timestamp, symbol) DO UPDATE
        SET macd = EXCLUDED.macd,
            rsi = EXCLUDED.rsi,
            sma10 = EXCLUDED.sma10,
            sma50 = EXCLUDED.sma50
        """
        cursor.execute(query, (data[0], data[1], float(data[2]), float(data[3]), float(data[4]), float(data[5])))
        logging.info(f"Inserted ML features for {data[1]} on {data[0]}")
    conn.commit()
    cursor.close()

# Fetch latest EIA data with fallback (using STEO endpoint and series IDs)
def fetch_eia_data(conn):
    try:
        series_mapping = {
            "COSXPUS": ("CrudeOilStocks", "million barrels, end-of-period"),  # STEO series for crude oil stocks
            "COPRPUS": ("CrudeOilProduction", "million barrels per day"),     # STEO series for crude oil production
            # SPR Stocks not directly available in STEO; will use None for now
        }
        
        eia_data = {}
        for series_id, (metric, unit) in series_mapping.items():
            # Use the STEO endpoint
            url = (
                f"https://api.eia.gov/v2/steo/data/"
                f"?api_key={EIA_API_KEY}"
                "&frequency=monthly"
                "&data[0]=value"
                f"&facets[seriesId][]={series_id}"
                "&sort[0][column]=period"
                "&sort[0][direction]=desc"
                "&offset=0"
                "&length=1"  # Fetch only the latest data point
            )
            logging.info(f"Fetching EIA data for series {series_id} from {url}")
            
            response = requests.get(url, timeout=10, headers=HEADERS)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data['response']['data'])
            if df.empty:
                logging.warning(f"No data returned for series {series_id}. API response: {data}")
                eia_data[metric] = None
                continue
            
            # Extract the latest value
            latest_data = df.iloc[0]
            report_date = datetime.datetime.strptime(latest_data['period'], '%Y-%m').date()
            value = float(latest_data['value'])
            
            # Insert into eia_data table
            insert_data(conn, "eia_data", (report_date, metric, value, unit))
            eia_data[metric] = value
        
        # SPR Stocks not available in STEO; set to None
        eia_data["SPR_Stocks"] = None
        return eia_data
    except Exception as e:
        logging.error(f"Error fetching EIA data: {str(e)}")
        eia_data = {}
        cursor = conn.cursor()
        for metric in ["CrudeOilStocks", "CrudeOilProduction", "SPR_Stocks"]:
            cursor.execute("""
                SELECT value 
                FROM eia_data 
                WHERE series_id = %s 
                ORDER BY period DESC 
                LIMIT 1
            """, (metric,))
            result = cursor.fetchone()
            if result:
                eia_data[metric] = result[0]
            else:
                eia_data[metric] = None
        cursor.close()
        return eia_data

# Fetch latest COT data (short format text file, not ZIP) with fallback
def fetch_cot_data(conn):
    try:
        # Calculate the most recent report date (reports are released on Fridays for the previous Tuesday)
        today = datetime.datetime.now()
        days_since_friday = (today.weekday() - 4) % 7  # Friday is 4 in weekday (0=Mon, 4=Fri)
        most_recent_friday = today - datetime.timedelta(days=days_since_friday)
        report_date = most_recent_friday - datetime.timedelta(days=3)  # Report reflects Tuesday's data
        year = report_date.strftime("%Y")  # e.g., 2025
        report_date_str = report_date.strftime("%m%d%y")  # Format as MMDDYY (e.g., 050625 for May 6, 2025)
        
        # Construct the URL for the Petroleum and Products Short Format report (text file, not ZIP)
        cot_url = f"https://www.cftc.gov/sites/default/files/files/dea/cotarchives/{year}/futures/petroleumproducts_sf_{report_date_str}.txt"
        logging.info(f"Fetching COT data from URL: {cot_url}")
        
        # Download the text file
        response = requests.get(cot_url, timeout=10, headers=HEADERS)
        response.raise_for_status()
        
        # Save the text file temporarily
        txt_path = os.path.join(TEMP_DIR, "petroleumproducts_sf.txt")
        with open(txt_path, 'wb') as f:
            f.write(response.content)
        
        # Read and parse the text file
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        cot_records = []
        for line in lines:
            line = line.strip()
            
            # Look for CL and HO markets
            if "CRUDE OIL, LIGHT SWEET-WTI - ICE FUTURES EUROPE" in line:
                symbol = 'CL'
            elif "NY HARBOR ULSD - NEW YORK MERCANTILE EXCHANGE" in line:
                symbol = 'HO'
            else:
                continue
            
            # Parse the positions line (next line after the market name)
            next_line_index = lines.index(line) + 2  # Skip the "Positions:" line
            if next_line_index >= len(lines):
                continue
            positions_line = lines[next_line_index].strip()
            
            # Split the positions line into fields (space-separated)
            fields = positions_line.split()
            if len(fields) < 11:  # Expect at least 11 fields (Producer Long/Short, Swap Long/Short/Spreading, etc.)
                logging.warning(f"Invalid positions line for {symbol}: {positions_line}")
                continue
            
            # Extract positions
            comm_positions_long = int(fields[0])  # Producer/Merchant Long
            comm_positions_short = int(fields[1])  # Producer/Merchant Short
            noncomm_positions_long = int(fields[6]) + int(fields[9])  # Managed Money Long + Other Reportables Long
            noncomm_positions_short = int(fields[7]) + int(fields[10])  # Managed Money Short + Other Reportables Short
            net_positions = (noncomm_positions_long - noncomm_positions_short) + (comm_positions_long - comm_positions_short)
            
            # Insert into cot_data table
            insert_data(conn, "cot_data", (report_date.date(), symbol, comm_positions_long, comm_positions_short, noncomm_positions_long, noncomm_positions_short, net_positions))
            
            # Store for market_data
            cot_records.append((symbol, comm_positions_long - comm_positions_short, noncomm_positions_long - noncomm_positions_short))
        
        # Clean up temporary file
        if os.path.exists(txt_path):
            os.remove(txt_path)
        
        if not cot_records:
            logging.warning("No CL or HO data found in the COT report")
            raise Exception("No CL or HO data found in the COT report")
        
        return cot_records
    except Exception as e:
        logging.error(f"Error fetching COT data: {str(e)}")
        # Fallback: fetch the most recent COT data from the database
        cot_records = []
        cursor = conn.cursor()
        for symbol in ['CL', 'HO']:
            cursor.execute("""
                SELECT comm_positions_long, comm_positions_short, noncomm_positions_long, noncomm_positions_short, net_positions
                FROM cot_data
                WHERE symbol = %s
                ORDER BY report_date DESC
                LIMIT 1
            """, (symbol,))
            result = cursor.fetchone()
            if result:
                comm_positions_long, comm_positions_short, noncomm_positions_long, noncomm_positions_short, net_positions = result
                cot_records.append((symbol, comm_positions_long - comm_positions_short, noncomm_positions_long - noncomm_positions_short))
            else:
                cot_records.append((symbol, None, None))
        cursor.close()
        return cot_records

# Fetch latest OPEC MOMR data with fallback
def fetch_opec_data(conn):
    try:
        url = "https://www.opec.org/opec_web/en/publications/202.htm"
        response = requests.get(url, timeout=10, headers=HEADERS)
        response.raise_for_status()
        # Placeholder for OPEC MOMR parsing logic (simplified)
        data = {'production': 27.0, 'demand_forecast': 104.5}  # Example values
        report_date = datetime.date.today()
        
        for metric, value in data.items():
            insert_data(conn, "opec_data", (report_date, metric, value, "million barrels per day"))
        
        return data
    except Exception as e:
        logging.error(f"Error fetching and processing OPEC MOMR: {str(e)}")
        opec_data = {}
        cursor = conn.cursor()
        for metric in ["production", "demand_forecast"]:
            cursor.execute("""
                SELECT value 
                FROM opec_data 
                WHERE metric = %s 
                ORDER BY report_date DESC 
                LIMIT 1
            """, (metric,))
            result = cursor.fetchone()
            if result:
                opec_data[metric] = result[0]
            else:
                opec_data[metric] = None
        cursor.close()
        return opec_data

# Fetch weather data using OpenWeatherMap API
def fetch_weather_data():
    locations = {
        'Chicago': (41.8781, -87.6298),
        'Cushing': (35.9851, -96.7670),
        'Houston': (29.7604, -95.3698),
        'NYC': (40.7128, -74.0060)
    }
    weather_data = {}
    
    for city, (lat, lon) in locations.items():
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=imperial"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            weather_data[city] = {
                'temperature': data['main']['temp'],
                'precipitation': data.get('rain', {}).get('1h', 0) + data.get('snow', {}).get('1h', 0),
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed'],
                'snowfall': data.get('snow', {}).get('1h', 0),
                'cloud_coverage': data['clouds']['all'] / 100.0,
                'hurricane_event': 'hurricane' in data.get('weather', [{}])[0].get('main', '').lower()
            }
        except Exception as e:
            logging.error(f"Error fetching weather data for {city}: {str(e)}")
            # Fallback to hardcoded values if API fails
            weather_data[city] = {
                'temperature': 68.72 if city == 'Chicago' else 86.54 if city == 'Cushing' else 90.39 if city == 'Houston' else 64.13,
                'precipitation': 0,
                'humidity': 77 if city == 'Chicago' else 48 if city == 'Cushing' else 52 if city == 'Houston' else 86,
                'wind_speed': 7 if city == 'Chicago' else 14.38 if city == 'Cushing' else 8.99 if city == 'Houston' else 14,
                'snowfall': 0,
                'cloud_coverage': 0.93 if city == 'Chicago' else 0.05 if city == 'Cushing' else 0.02 if city == 'Houston' else 1.0,
                'hurricane_event': False
            }
    
    return weather_data

# Calculate technical indicators
def calculate_technical_indicators(prices):
    if len(prices) < 50:
        return 0, 0, 0, 0
    prices_series = pd.Series(prices)
    macd = prices_series.ewm(span=12, adjust=False).mean() - prices_series.ewm(span=26, adjust=False).mean()
    macd = macd.iloc[-1]
    delta = prices_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs.iloc[-1]))
    sma10 = prices_series.rolling(window=10).mean().iloc[-1]
    sma50 = prices_series.rolling(window=50).mean().iloc[-1]
    return macd, rsi, sma10, sma50

# Main script execution
def main():
    logging.info("Starting the daily trading database update script...")
    
    # Connect to database
    conn = connect_to_db()
    
    # Fetch EIA data
    eia_data = fetch_eia_data(conn)
    
    # Connect to IB Gateway with retry mechanism
    max_retries = 3
    retry_delay = 5  # seconds
    app = IBClient()
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempting to connect to IB Gateway (attempt {attempt + 1}/{max_retries})...")
            app.connect("127.0.0.1", 4002, clientId=1)
            app.run()
            
            while not app.sync_done:
                time.sleep(1)
            
            app.stop()
            break  # Connection successful, exit retry loop
        except Exception as e:
            logging.error(f"Failed to connect to IB Gateway: {str(e)}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error("Max retries reached. Unable to connect to IB Gateway. Using fallback data.")
                # Fallback: Set default values for bar data
                app.data = [
                    {'date': datetime.datetime.now().strftime("%Y%m%d"), 'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 0}
                ]
    
    # Fetch COT data
    cot_data = fetch_cot_data(conn)
    cot_dict = {symbol: (comm_net, noncomm_net) for symbol, comm_net, noncomm_net in cot_data}
    
    # Fetch OPEC data
    opec_data = fetch_opec_data(conn)
    
    # Fetch weather data
    weather_data = fetch_weather_data()
    
    # Process IB data and insert into database
    timestamp = datetime.datetime.now()
    for symbol in ['CL', 'HO']:
        # Get the latest bar data
        bar_data = [d for d in app.data if d['date'] == timestamp.strftime("%Y%m%d")]
        if not bar_data:
            logging.warning(f"No bar data available for {symbol}. Using default values.")
            bar_data = [{'date': timestamp.strftime("%Y%m%d"), 'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 0}]
        latest_bar = bar_data[-1]
        
        # Get COT data for the symbol
        cot_comm_net, cot_noncomm_net = cot_dict.get(symbol, (None, None))
        
        # Insert market data
        market_data = (
            timestamp,
            symbol,
            latest_bar['open'],
            latest_bar['high'],
            latest_bar['low'],
            latest_bar['close'],
            latest_bar['volume'],
            eia_data.get("CrudeOilStocks"),
            eia_data.get("CrudeOilProduction"),
            eia_data.get("SPR_Stocks"),
            opec_data.get("production"),
            opec_data.get("demand_forecast"),
            cot_comm_net,
            cot_noncomm_net,
            weather_data
        )
        insert_data(conn, "market_data", market_data)
        
        # Calculate and insert ML features
        prices = [d['close'] for d in app.data if d['date'] <= timestamp.strftime("%Y%m%d")]
        macd, rsi, sma10, sma50 = calculate_technical_indicators(prices)
        ml_features = (timestamp, symbol, macd, rsi, sma10, sma50)
        insert_data(conn, "ml_features", ml_features)
    
    # Close database connection
    conn.close()
    logging.info("Script completed successfully.")

if __name__ == "__main__":
    main()