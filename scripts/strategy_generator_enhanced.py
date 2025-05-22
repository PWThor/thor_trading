import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from sklearn.preprocessing import MinMaxScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import sqlite3
import xgboost as xgb
import joblib
import cProfile
import pstats
from multiprocessing import Pool
from tqdm import tqdm
import psutil
import time

# Try importing talib, fall back to pandas_ta if unavailable
try:
    import talib
except ImportError:
    import pandas_ta as ta

# Set up logging to save to E:\Projects\thor_trading\outputs\logs
log_dir = "E:/Projects/thor_trading/outputs/logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "trading_log.txt"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
handler = logger.handlers[0]  # Get the file handler to flush logs

# Initialize NLTK VADER for sentiment analysis
sia = SentimentIntensityAnalyzer()

def log_and_flush(message):
    """
    Log a message and flush the log handler to ensure immediate write.
    """
    logging.info(message)
    handler.flush()

def fetch_news_sentiment(symbol, days=7):
    """
    Fetch recent news for a symbol and calculate sentiment score (mock implementation with variation).
    Args:
        symbol (str): 'CL' or 'HO'
        days (int): Number of days to look back
    Returns: float, sentiment score (-1 to 1)
    """
    try:
        search_term = "Crude Oil" if symbol == "CL" else "Heating Oil"
        base_headlines = [
            f"{search_term} prices surge due to supply concerns",
            f"Geopolitical tensions impact {search_term} market",
            f"{search_term} demand expected to rise in winter",
            f"OPEC cuts production, affecting {search_term} supply",
            f"{search_term} market faces volatility amid economic slowdown"
        ]
        if symbol == "CL":
            base_headlines.append(f"Middle East unrest boosts {search_term} prices")
        else:
            base_headlines.append(f"Refinery outage impacts {search_term} supply")

        sentiment_scores = []
        for headline in base_headlines:
            scores = sia.polarity_scores(headline)
            sentiment_scores.append(scores['compound'])
        
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        log_and_flush(f"Sentiment score for {symbol}: {avg_sentiment}")
        return avg_sentiment
    except Exception as e:
        log_and_flush(f"Error fetching news sentiment for {symbol}: {e}")
        raise

def validate_data(prophetx_data):
    """
    Validate the input data to ensure required columns are present and data is not empty.
    Args:
        prophetx_data (DataFrame): Raw data from prophetx_data.csv
    Returns: bool, True if valid, raises exception if not
    """
    required_columns = ['Date', 'OHLCV_Open', 'OHLCV_High', 'OHLCV_Low', 'OHLCV_Close', 'OHLCV_Volume',
                        'OHLCV_Open.1', 'OHLCV_High.1', 'OHLCV_Low.1', 'OHLCV_Close.1', 'OHLCV_Volume.1']
    missing_columns = [col for col in required_columns if col not in prophetx_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in prophetx_data.csv: {missing_columns}")
    
    if prophetx_data.empty:
        raise ValueError("prophetx_data.csv is empty")
    
    cot_columns = [col for col in prophetx_data.columns if col.startswith('COT_')]
    if not cot_columns:
        logging.warning("No COT columns found in prophetx_data.csv. Fundamental analysis will be limited.")
    
    return True

def load_data():
    """
    Load and preprocess data from prophetx_data.csv, sampling the last 5000 rows.
    Returns: cl_data (CL DataFrame), ho_data (HO DataFrame), prophetx_data (raw DataFrame)
    """
    start_time = time.time()
    try:
        prophetx_data = pd.read_csv('E:/Projects/thor_trading/data/raw/prophetx_data.csv')
        # Sample the last 5000 rows to reduce runtime
        prophetx_data = prophetx_data.tail(5000).reset_index(drop=True)
        elapsed = time.time() - start_time
        log_and_flush(f"Loaded and sampled prophetx_data.csv with shape: {prophetx_data.shape} in {elapsed:.2f} seconds")

        validate_data(prophetx_data)

        cl_data = prophetx_data.rename(columns={
            'OHLCV_Open': 'CL_Open',
            'OHLCV_High': 'CL_High',
            'OHLCV_Low': 'CL_Low',
            'OHLCV_Close': 'CL_Close',
            'OHLCV_Volume': 'CL_Volume'
        })[['Date', 'CL_Open', 'CL_High', 'CL_Low', 'CL_Close', 'CL_Volume']].copy()

        ho_data = prophetx_data.rename(columns={
            'OHLCV_Open.1': 'HO_Open',
            'OHLCV_High.1': 'HO_High',
            'OHLCV_Low.1': 'HO_Low',
            'OHLCV_Close.1': 'HO_Close',
            'OHLCV_Volume.1': 'HO_Volume'
        })[['Date', 'HO_Open', 'HO_High', 'HO_Low', 'HO_Close', 'HO_Volume']].copy()

        cot_columns = [col for col in prophetx_data.columns if col.startswith('COT_')]
        if cot_columns:
            cot_data = prophetx_data[cot_columns]
            cl_data = pd.concat([cl_data, cot_data], axis=1)
            ho_data = pd.concat([ho_data, cot_data], axis=1)

        cl_data['Date'] = pd.to_datetime(cl_data['Date'])
        ho_data['Date'] = pd.to_datetime(ho_data['Date'])
        prophetx_data['Date'] = pd.to_datetime(prophetx_data['Date'])

        return cl_data, ho_data, prophetx_data
    except Exception as e:
        log_and_flush(f"Error loading data: {e}")
        raise

def preprocess_data(data, symbol):
    """
    Preprocess data for ML and rule-based strategy: add technical indicators, fundamental features, and sentiment.
    Args:
        data (DataFrame): OHLCV data with COT columns
        symbol (str): 'CL' or 'HO'
    Returns: processed_data (DataFrame)
    """
    start_time = time.time()
    try:
        processed_data = data.copy()

        # Progress bar for preprocessing steps
        steps = 5  # RSI, MA50, MACD, ATR, COT
        with tqdm(total=steps, desc=f"Preprocessing {symbol} data") as pbar:
            try:
                processed_data[f'{symbol}_RSI'] = talib.RSI(processed_data[f'{symbol}_Close'], timeperiod=14)
            except:
                processed_data[f'{symbol}_RSI'] = ta.rsi(processed_data[f'{symbol}_Close'], length=14)
            pbar.update(1)

            try:
                processed_data[f'{symbol}_MA50'] = talib.SMA(processed_data[f'{symbol}_Close'], timeperiod=50)
            except:
                processed_data[f'{symbol}_MA50'] = ta.sma(processed_data[f'{symbol}_Close'], length=50)
            pbar.update(1)

            try:
                macd, signal, _ = talib.MACD(processed_data[f'{symbol}_Close'])
                processed_data[f'{symbol}_MACD'] = macd
                processed_data[f'{symbol}_MACD_Signal'] = signal
            except:
                macd = ta.macd(processed_data[f'{symbol}_Close'])
                processed_data[f'{symbol}_MACD'] = macd[f'MACD_12_26_9']
                processed_data[f'{symbol}_MACD_Signal'] = macd[f'MACDs_12_26_9']
            pbar.update(1)

            try:
                processed_data[f'{symbol}_ATR'] = talib.ATR(
                    processed_data[f'{symbol}_High'],
                    processed_data[f'{symbol}_Low'],
                    processed_data[f'{symbol}_Close'],
                    timeperiod=14
                )
            except:
                processed_data[f'{symbol}_ATR'] = ta.atr(
                    processed_data[f'{symbol}_High'],
                    processed_data[f'{symbol}_Low'],
                    processed_data[f'{symbol}_Close'],
                    length=14
                )
            pbar.update(1)

            cot_columns = [col for col in processed_data.columns if col.startswith('COT_')]
            if 'COT_Close' in cot_columns and 'COT_Close.1' in cot_columns:
                processed_data[f'{symbol}_Net_COT'] = processed_data['COT_Close'] - processed_data['COT_Close.1']
                processed_data[f'{symbol}_COT_Trend'] = processed_data[f'{symbol}_Net_COT'].rolling(window=5).mean().diff()
                scaler = MinMaxScaler()
                processed_data[[f'{symbol}_Net_COT', f'{symbol}_COT_Trend']] = scaler.fit_transform(
                    processed_data[[f'{symbol}_Net_COT', f'{symbol}_COT_Trend']]
                )
            else:
                processed_data[f'{symbol}_Net_COT'] = 0.0
                processed_data[f'{symbol}_COT_Trend'] = 0.0
            pbar.update(1)

        sentiment_score = fetch_news_sentiment(symbol)
        processed_data[f'{symbol}_Sentiment'] = sentiment_score

        processed_data = processed_data.dropna()
        elapsed = time.time() - start_time
        log_and_flush(f"Preprocessed {symbol} data with shape: {processed_data.shape} in {elapsed:.2f} seconds")
        print(f"Preprocessed {symbol} data in {elapsed:.2f} seconds, CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%")
        return processed_data
    except Exception as e:
        log_and_flush(f"Error preprocessing data for {symbol}: {e}")
        raise

def train_xgboost_model(data, symbol, train_ratio=0.8):
    """
    Train an XGBoost model to predict trading signals (long/short/neutral).
    Args:
        data (DataFrame): Preprocessed data with features
        symbol (str): 'CL' or 'HO'
        train_ratio (float): Ratio of data to use for training
    Returns: Trained XGBoost model
    """
    start_time = time.time()
    try:
        # Define features for XGBoost
        features = [
            f'{symbol}_RSI', f'{symbol}_MACD', f'{symbol}_MACD_Signal', f'{symbol}_ATR',
            f'{symbol}_Net_COT', f'{symbol}_COT_Trend', f'{symbol}_Sentiment'
        ]
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")

        # Define target: 1 (long) if next day's return > 0, -1 (short) if < 0, 0 (neutral) otherwise
        data['Next_Return'] = data[f'{symbol}_Close'].shift(-1).pct_change(fill_method=None)
        data['Target'] = np.where(data['Next_Return'] > 0, 1, np.where(data['Next_Return'] < 0, -1, 0))

        # Remap target labels: [-1, 0, 1] -> [0, 1, 2]
        # -1 (short) -> 0, 0 (neutral) -> 1, 1 (long) -> 2
        data['Target_Remapped'] = data['Target'].map({-1: 0, 0: 1, 1: 2})

        # Drop NaN values introduced by shifting
        model_data = data.dropna()

        # Split into train and test
        train_size = int(len(model_data) * train_ratio)
        train_data = model_data.iloc[:train_size]
        test_data = model_data.iloc[train_size:]

        X_train = train_data[features]
        y_train = train_data['Target_Remapped']
        X_test = test_data[features]
        y_test = test_data['Target_Remapped']

        # Train XGBoost model using XGBClassifier with simplified parameters
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,  # Long (2), Neutral (1), Short (0)
            max_depth=4,  # Reduced depth for faster training
            eta=0.3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            n_estimators=50,  # Reduced number of trees for faster training
            early_stopping_rounds=10  # Set in constructor to avoid deprecation warning
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Save the model using joblib
        model_path = f"E:/Projects/thor_trading/models/xgb_{symbol.lower()}_model.joblib"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        elapsed = time.time() - start_time
        log_and_flush(f"Trained and saved XGBoost model for {symbol} at {model_path} in {elapsed:.2f} seconds")
        print(f"Trained {symbol} model in {elapsed:.2f} seconds, CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%")
        return model
    except Exception as e:
        log_and_flush(f"Error training XGBoost model for {symbol}: {e}")
        raise

def load_xgboost_model(symbol):
    """
    Load a pre-trained XGBoost model.
    Args:
        symbol (str): 'CL' or 'HO'
    Returns: Loaded XGBoost model
    """
    try:
        model_path = f"E:/Projects/thor_trading/models/xgb_{symbol.lower()}_model.joblib"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XGBoost model not found at {model_path}. Please train the model first.")
        
        model = joblib.load(model_path)
        log_and_flush(f"Loaded XGBoost model for {symbol} from {model_path}")
        return model
    except Exception as e:
        log_and_flush(f"Error loading XGBoost model for {symbol}: {e}")
        raise

def generate_strategy_params():
    """
    Generate random parameters for rule-based components of the strategy.
    Returns: dict of parameters
    """
    return {
        'stop_loss_atr_multiplier': np.random.uniform(1.0, 3.0),
        'take_profit_atr_multiplier': np.random.uniform(2.0, 5.0),
        'cot_threshold': np.random.uniform(0.2, 0.8),
        'cot_trend_threshold': np.random.uniform(-0.5, 0.5),
        'sentiment_threshold': np.random.uniform(-0.5, 0.5)
    }

def backtest_strategy(args):
    """
    Helper function to backtest a single strategy (for parallel processing).
    Args:
        args: Tuple of (data, symbol, params, xgb_model)
    Returns: Tuple of (cl_metrics, ho_metrics)
    """
    data, symbol, params, xgb_model = args
    try:
        position = 0  # 0: no position, 1: long, -1: short
        trades = []
        equity = 100000  # Starting capital
        shares = 0
        trade_durations = []
        entry_time = None
        entry_price = None
        equity_curve = [equity]

        transaction_cost_per_trade = 0.001
        slippage_per_trade = 0.0005

        # Features for XGBoost prediction
        features = [
            f'{symbol}_RSI', f'{symbol}_MACD', f'{symbol}_MACD_Signal', f'{symbol}_ATR',
            f'{symbol}_Net_COT', f'{symbol}_COT_Trend', f'{symbol}_Sentiment'
        ]

        # Batch predictions for efficiency
        feature_batch = data[features].copy()
        signals = xgb_model.predict(feature_batch)  # 0 (short), 1 (neutral), 2 (long)
        # Remap predictions back to [-1, 0, 1]
        signal_map = {0: -1, 1: 0, 2: 1}
        signals = np.array([signal_map[s] for s in signals])

        for i in range(1, len(data)):
            row = data.iloc[i]
            signal = signals[i]  # Use precomputed signal

            # Entry logic: XGBoost signal + rule-based fundamental filter
            if position == 0:
                fundamental_condition = (
                    row[f'{symbol}_Net_COT'] > params['cot_threshold'] and
                    row[f'{symbol}_COT_Trend'] > params['cot_trend_threshold'] and
                    row[f'{symbol}_Sentiment'] > params['sentiment_threshold']
                )

                if signal == 1 and fundamental_condition:
                    position = 1  # Long
                elif signal == -1 and fundamental_condition:
                    position = -1  # Short

                if position != 0:
                    entry_price = row[f'{symbol}_Close']
                    entry_price *= (1 + slippage_per_trade if position == 1 else 1 - slippage_per_trade)
                    max_risk = equity * 0.1
                    shares = min(max_risk // entry_price, equity // entry_price)
                    if shares == 0:
                        position = 0
                        continue
                    equity_change = shares * entry_price
                    equity -= equity_change if position == 1 else -equity_change
                    equity -= equity_change * transaction_cost_per_trade
                    entry_time = row['Date']
                    trades.append({
                        'entry_date': entry_time,
                        'entry_price': entry_price,
                        'position': 'long' if position == 1 else 'short',
                        'shares': shares
                    })

            # Exit logic: Rule-based using ATR
            elif position != 0:
                atr = row[f'{symbol}_ATR']
                stop_loss = entry_price * (1 - params['stop_loss_atr_multiplier'] * atr / entry_price if position == 1 else 1 + params['stop_loss_atr_multiplier'] * atr / entry_price)
                take_profit = entry_price * (1 + params['take_profit_atr_multiplier'] * atr / entry_price if position == 1 else 1 - params['take_profit_atr_multiplier'] * atr / entry_price)
                should_exit = (position == 1 and (row[f'{symbol}_Close'] <= stop_loss or row[f'{symbol}_Close'] >= take_profit)) or \
                              (position == -1 and (row[f'{symbol}_Close'] >= stop_loss or row[f'{symbol}_Close'] <= take_profit))

                if should_exit:
                    exit_price = row[f'{symbol}_Close']
                    exit_price *= (1 - slippage_per_trade if position == 1 else 1 + slippage_per_trade)
                    equity_change = shares * exit_price
                    if position == 1:
                        equity += equity_change
                    else:
                        equity -= (exit_price - entry_price) * shares
                    equity -= equity_change * transaction_cost_per_trade
                    trades.append({
                        'exit_date': row['Date'],
                        'exit_price': exit_price,
                        'position': 'long' if position == 1 else 'short'
                    })
                    
                    trade_duration = (row['Date'] - entry_time).days
                    trade_durations.append(trade_duration)
                    
                    position = 0
                    shares = 0
                    entry_time = None
                    entry_price = None

            equity_curve.append(equity)

        # Calculate metrics
        returns = (equity - 100000) / 100000 if equity > 0 else -1
        sharpe = returns / (data[f'{symbol}_Close'].pct_change(fill_method=None).std() * np.sqrt(252)) if trades else 0
        valid_trades = [t for t in trades if 'entry_price' in t and 'exit_price' in t]
        drawdown = min(0, min([t['exit_price'] / t['entry_price'] - 1 if t['position'] == 'long' else 1 - t['exit_price'] / t['entry_price'] for t in valid_trades] + [0])) if valid_trades else 0
        win_rate = sum(1 for t in valid_trades if ((t['position'] == 'long' and t['exit_price'] > t['entry_price']) or (t['position'] == 'short' and t['exit_price'] < t['entry_price']))) / len(valid_trades) if valid_trades else 0
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0

        return {
            'total_return': returns,
            'sharpe_ratio': sharpe,
            'max_drawdown': drawdown,
            'win_rate': win_rate,
            'avg_trade_duration': avg_trade_duration,
            'trades': trades,
            'equity_curve': equity_curve,
            'params': params
        }
    except Exception as e:
        logging.error(f"Error backtesting strategy for {symbol}: {e}")
        raise

def save_to_sqlite(results_cl, results_ho):
    """
    Save strategy results to SQLite database with a timestamp.
    Args:
        results_cl (list): CL strategy results
        results_ho (list): HO strategy results
    """
    start_time = time.time()
    try:
        db_path = "E:/Projects/thor_trading/outputs/strategies.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cl_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stop_loss_atr_multiplier FLOAT,
                take_profit_atr_multiplier FLOAT,
                cot_threshold FLOAT,
                cot_trend_threshold FLOAT,
                sentiment_threshold FLOAT,
                total_return FLOAT,
                sharpe_ratio FLOAT,
                max_drawdown FLOAT,
                win_rate FLOAT,
                avg_trade_duration FLOAT,
                score FLOAT,
                run_timestamp TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ho_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stop_loss_atr_multiplier FLOAT,
                take_profit_atr_multiplier FLOAT,
                cot_threshold FLOAT,
                cot_trend_threshold FLOAT,
                sentiment_threshold FLOAT,
                total_return FLOAT,
                sharpe_ratio FLOAT,
                max_drawdown FLOAT,
                win_rate FLOAT,
                avg_trade_duration FLOAT,
                score FLOAT,
                run_timestamp TEXT
            )
        ''')

        run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Insert CL strategies with progress bar
        for result in tqdm(results_cl, desc="Saving CL strategies to SQLite"):
            params = result['params']
            cursor.execute('''
                INSERT INTO cl_strategies (
                    stop_loss_atr_multiplier, take_profit_atr_multiplier, cot_threshold,
                    cot_trend_threshold, sentiment_threshold, total_return, sharpe_ratio,
                    max_drawdown, win_rate, avg_trade_duration, score, run_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                params['stop_loss_atr_multiplier'], params['take_profit_atr_multiplier'],
                params['cot_threshold'], params['cot_trend_threshold'],
                params['sentiment_threshold'], result['total_return'],
                result['sharpe_ratio'], result['max_drawdown'],
                result['win_rate'], result['avg_trade_duration'],
                result['score'], run_timestamp
            ))

        # Insert HO strategies with progress bar
        for result in tqdm(results_ho, desc="Saving HO strategies to SQLite"):
            params = result['params']
            cursor.execute('''
                INSERT INTO ho_strategies (
                    stop_loss_atr_multiplier, take_profit_atr_multiplier, cot_threshold,
                    cot_trend_threshold, sentiment_threshold, total_return, sharpe_ratio,
                    max_drawdown, win_rate, avg_trade_duration, score, run_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                params['stop_loss_atr_multiplier'], params['take_profit_atr_multiplier'],
                params['cot_threshold'], params['cot_trend_threshold'],
                params['sentiment_threshold'], result['total_return'],
                result['sharpe_ratio'], result['max_drawdown'],
                result['win_rate'], result['avg_trade_duration'],
                result['score'], run_timestamp
            ))

        conn.commit()
        conn.close()
        elapsed = time.time() - start_time
        log_and_flush(f"Saved strategy results to {db_path} at {run_timestamp} in {elapsed:.2f} seconds")
        print(f"Saved results to SQLite in {elapsed:.2f} seconds, CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%")
    except Exception as e:
        log_and_flush(f"Error saving to SQLite: {e}")
        raise

def generate_summary_report(results_cl, results_ho):
    """
    Generate a summary report of the backtesting run.
    Args:
        results_cl (list): CL strategy results
        results_ho (list): HO strategy results
    """
    start_time = time.time()
    try:
        report_dir = "E:/Projects/thor_trading/outputs/reports"
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        cl_equity = np.array(results_cl[0]['equity_curve']) if results_cl else np.array([100000])
        ho_equity = np.array(results_ho[0]['equity_curve']) if results_ho else np.array([100000])
        portfolio_equity = (cl_equity + ho_equity) / 2
        portfolio_returns = np.diff(portfolio_equity) / portfolio_equity[:-1]
        portfolio_sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252) if len(portfolio_returns) > 1 else 0

        cl_returns = np.diff(cl_equity) / cl_equity[:-1] if len(cl_equity) > 1 else np.array([0])
        ho_returns = np.diff(ho_equity) / ho_equity[:-1] if len(ho_equity) > 1 else np.array([0])
        if (len(cl_returns) == len(ho_returns) and len(cl_returns) > 1 and
            not np.any(np.isnan(cl_returns)) and not np.any(np.isnan(ho_returns)) and
            np.std(cl_returns) != 0 and np.std(ho_returns) != 0):
            correlation = np.corrcoef(cl_returns, ho_returns)[0, 1]
        else:
            correlation = 0

        with open(report_path, 'w') as f:
            f.write("Backtesting Summary Report\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Top CL Strategy:\n")
            if results_cl:
                top_cl = results_cl[0]
                f.write(f"Score: {top_cl['score']:.2f}\n")
                f.write(f"Total Return: {top_cl['total_return']:.2f}\n")
                f.write(f"Sharpe Ratio: {top_cl['sharpe_ratio']:.2f}\n")
                f.write(f"Max Drawdown: {top_cl['max_drawdown']:.2f}\n")
                f.write(f"Win Rate: {top_cl['win_rate']:.2f}\n")
                f.write(f"Avg Trade Duration: {top_cl['avg_trade_duration']:.2f} days\n\n")
            else:
                f.write("No CL strategies generated.\n\n")

            f.write("Top HO Strategy:\n")
            if results_ho:
                top_ho = results_ho[0]
                f.write(f"Score: {top_ho['score']:.2f}\n")
                f.write(f"Total Return: {top_ho['total_return']:.2f}\n")
                f.write(f"Sharpe Ratio: {top_ho['sharpe_ratio']:.2f}\n")
                f.write(f"Max Drawdown: {top_ho['max_drawdown']:.2f}\n")
                f.write(f"Win Rate: {top_ho['win_rate']:.2f}\n")
                f.write(f"Avg Trade Duration: {top_ho['avg_trade_duration']:.2f} days\n\n")
            else:
                f.write("No HO strategies generated.\n\n")

            f.write("Portfolio Metrics:\n")
            f.write(f"Portfolio Sharpe Ratio: {portfolio_sharpe:.2f}\n")
            f.write(f"Correlation between CL and HO Returns: {correlation:.2f}\n")

        elapsed = time.time() - start_time
        log_and_flush(f"Generated summary report at {report_path} in {elapsed:.2f} seconds")
        print(f"Generated summary report in {elapsed:.2f} seconds, CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%")
    except Exception as e:
        log_and_flush(f"Error generating summary report: {e}")
        raise

def generate_and_backtest_strategies():
    """
    Generate and backtest 500 strategies for CL and HO using XGBoost predictions.
    Returns: results_cl (list), results_ho (list)
    """
    global_start_time = time.time()
    try:
        cl_data, ho_data, prophetx_data = load_data()

        # Preprocess data
        cl_data = preprocess_data(cl_data, 'CL')
        ho_data = preprocess_data(ho_data, 'HO')

        # Delete existing models to force retraining with correct features
        for symbol in ['CL', 'HO']:
            model_path = f"E:/Projects/thor_trading/models/xgb_{symbol.lower()}_model.joblib"
            if os.path.exists(model_path):
                os.remove(model_path)
                log_and_flush(f"Deleted existing model at {model_path} to force retraining")

        # Train XGBoost models (since we deleted the old ones, this will always run)
        cl_xgb_model = train_xgboost_model(cl_data, 'CL')
        ho_xgb_model = train_xgboost_model(ho_data, 'HO')

        # Generate and backtest strategies
        num_strategies = 500  # Reduced to 500 strategies for faster runtime
        params_list = [generate_strategy_params() for _ in tqdm(range(num_strategies), desc="Generating strategy parameters")]

        # Prepare arguments for parallel processing
        cl_args = [(cl_data, 'CL', params, cl_xgb_model) for params in params_list]
        ho_args = [(ho_data, 'HO', params, ho_xgb_model) for params in params_list]

        # Parallel backtesting with progress bar
        print(f"Starting backtesting for {num_strategies} strategies...")
        with Pool() as pool:
            results_cl = list(tqdm(pool.imap(backtest_strategy, cl_args), total=len(cl_args), desc="Backtesting CL strategies"))
            results_ho = list(tqdm(pool.imap(backtest_strategy, ho_args), total=len(ho_args), desc="Backtesting HO strategies"))

        # Sort by composite score
        for res in [results_cl, results_ho]:
            for r in res:
                r['score'] = (r['total_return'] * 0.5 + r['sharpe_ratio'] * 0.3 - r['max_drawdown'] * 0.2)
            res.sort(key=lambda x: x['score'], reverse=True)

        # Plot performance of top strategy
        plot_dir = "E:/Projects/thor_trading/outputs/plots"
        os.makedirs(plot_dir, exist_ok=True)
        
        for symbol, results, data in [('CL', results_cl, cl_data), ('HO', results_ho, ho_data)]:
            if results:
                top_trades = results[0]['trades']
                plt.figure(figsize=(10, 6))
                plt.plot(data['Date'], data[f'{symbol}_Close'], label=f'{symbol} Close Price', color='blue')
                for trade in top_trades:
                    if 'entry_date' in trade and 'entry_price' in trade:
                        plt.scatter(trade['entry_date'], trade['entry_price'], color='green', marker='^', label='Buy' if trade is top_trades[0] else "")
                    if 'exit_date' in trade and 'exit_price' in trade:
                        plt.scatter(trade['exit_date'], trade['exit_price'], color='red', marker='v', label='Sell' if trade is top_trades[0] else "")
                plt.title(f'Top {symbol} Strategy Performance')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(plot_dir, f'{symbol}_top_strategy.png'))
                plt.close()

        total_elapsed = time.time() - global_start_time
        log_and_flush(f"Completed strategy generation and backtesting in {total_elapsed:.2f} seconds")
        print(f"Completed backtesting in {total_elapsed:.2f} seconds, CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%")
        return results_cl, results_ho
    except Exception as e:
        log_and_flush(f"Error in generate_and_backtest_strategies: {e}")
        raise

if __name__ == "__main__":
    try:
        # Profile the script execution
        profiler = cProfile.Profile()
        profiler.enable()

        # Generate and backtest strategies
        log_and_flush("Starting strategy generation and backtesting")
        print("Starting strategy generation and backtesting...")
        results_cl, results_ho = generate_and_backtest_strategies()
        log_and_flush(f"Top CL Strategy: {results_cl[0] if results_cl else 'No strategies generated'}")
        log_and_flush(f"Top HO Strategy: {results_ho[0] if results_ho else 'No strategies generated'}")

        # Save results to SQLite
        save_to_sqlite(results_cl, results_ho)

        # Generate summary report
        generate_summary_report(results_cl, results_ho)

        # Stop profiling and print results
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(10)  # Print top 10 time-consuming functions
    except Exception as e:
        log_and_flush(f"Main execution failed: {e}")
        raise