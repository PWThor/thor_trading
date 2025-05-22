# ml_trading_signals.py
# Backend for continuous learning trading system (backtesting focus with XGBoost)

import logging
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from ta.momentum import RSIIndicator
from datetime import datetime
import pytz
import backtrader as bt
from backtrader.feeds import PandasData
from sqlalchemy import create_engine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_signals.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'initial_cash': 100000.0,
    'commission': 0.001,
    'slippage': 0.01,
    'contract_size': {'CL': 1000, 'HO': 42000},
    'probability_threshold': 0.3,  # Lowered to generate more trades
    'fundamental_threshold': 0.5,  # Lowered to generate more trades
    'high_fundamental_threshold': 1.0,
    'prob_win_threshold': 0.5,
    'high_prob_win_threshold': 0.7,
    'db_params': {
        'host': 'localhost',
        'port': '5432',
        'database': 'trading_db',
        'user': 'postgres',
        'password': 'Makingmoney25!'  # Replace with your password
    },
    'backtest_start': '1992-01-01',
    'backtest_end': '2025-05-13',
    'trading_window_start': '16:00:00',
    'trading_window_end': '09:00:00'
}

class TradingData:
    def __init__(self, config):
        self.config = config
        self.price_data = {}
        self.cot_data = None
        self.market_data = None
        self.engine = create_engine(
            f"postgresql+psycopg2://{config['db_params']['user']}:{config['db_params']['password']}@{config['db_params']['host']}:{config['db_params']['port']}/{config['db_params']['database']}"
        )

    def fetch_price_data(self, symbol):
        try:
            query = """
            SELECT timestamp AS date, symbol, open, high, low, close, volume
            FROM market_data
            WHERE symbol = %s AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            """
            df = pd.read_sql_query(
                query,
                self.engine,
                params=(symbol, self.config['backtest_start'], self.config['backtest_end']),
                parse_dates=['date'],
                index_col='date'
            )
            if df.empty:
                logger.warning(f"No price data for {symbol}")
                return None
            df = df.between_time(self.config['trading_window_start'], self.config['trading_window_end'])
            if df.empty:
                logger.warning(f"No price data for {symbol} in trading window")
                return None
            self.price_data[symbol] = df[['open', 'high', 'low', 'close', 'volume']]
            logger.info(f"Fetched {symbol} price data: {len(df)} rows")
            return self.price_data[symbol]
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} price data: {e}")
            return None

    def fetch_cot_data(self):
        try:
            query = """
            SELECT report_date AS date, symbol, net_positions
            FROM cot_data
            WHERE report_date BETWEEN %s AND %s
            ORDER BY report_date
            """
            df = pd.read_sql_query(
                query,
                self.engine,
                params=(self.config['backtest_start'], self.config['backtest_end']),
                parse_dates=['date']
            )
            if df.empty:
                logger.warning("No COT data found")
                return None
            self.cot_data = df.pivot(index='date', columns='symbol', values='net_positions')
            logger.info(f"Fetched COT data: {len(df)} rows")
            return self.cot_data
        except Exception as e:
            logger.error(f"Failed to fetch COT data: {e}")
            return None

    def fetch_market_data(self):
        try:
            query = """
            SELECT timestamp AS date, symbol, cot_commercial_net, cot_noncommercial_net,
                   eia_inventory, sentiment_score, opec_production
            FROM market_data
            WHERE timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            """
            df = pd.read_sql_query(
                query,
                self.engine,
                params=(self.config['backtest_start'], self.config['backtest_end']),
                parse_dates=['date']
            )
            if df.empty:
                logger.warning("No market data found")
                return None
            self.market_data = df.pivot(index='date', columns='symbol')
            self.market_data.columns = ['_'.join(col).strip() for col in self.market_data.columns.values]
            logger.info(f"Fetched market data: {len(df)} rows")
            return self.market_data
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            return None

    def prepare_features(self, price_data, cot_data, market_data):
        df = price_data.copy()
        df['rsi'] = RSIIndicator(df['close']).rsi()
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        df['close_lag1'] = df['close'].shift(1)
        df['close_lag5'] = df['close'].shift(5)
        if cot_data is not None:
            df = df.join(cot_data.add_prefix('cot_net_'))
        if market_data is not None:
            df = df.join(market_data[['cot_commercial_net_CL', 'cot_noncommercial_net_CL', 'eia_inventory', 'sentiment_score_CL', 'opec_production']])
        df = df.fillna(method='ffill').fillna(0)
        return df.drop(columns=['open', 'high', 'low', 'close', 'volume'], errors='ignore')

class MLModel:
    def __init__(self):
        self.model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        self.is_trained = False

    def train(self, features, target):
        try:
            self.model.fit(features, target)
            self.is_trained = True
            logger.info("XGBoost model trained")
        except Exception as e:
            logger.error(f"Failed to train XGBoost model: {e}")

    def predict_proba(self, features):
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict_proba(features)[:, 1]

class TradingStrategy(bt.Strategy):
    params = (
        ('config', None),
        ('ml_model', None),
        ('win_model', None),
        ('data_handler', None),
    )

    def __init__(self):
        self.orders = {d._name: None for d in self.datas}
        self.positions = {d._name: None for d in self.datas}
        self.technical_indicators = {d._name: {'rsi': RSIIndicator(d.close).rsi(), 'atr': bt.indicators.ATR(d, period=14)} for d in self.datas}
        self.prev_fundamental_score = {}
        self.engine = create_engine(
            f"postgresql+psycopg2://{self.params.config['db_params']['user']}:{self.params.config['db_params']['password']}@{self.params.config['db_params']['host']}:{self.params.config['db_params']['port']}/{self.params.config['db_params']['database']}"
        )
        self.prob_threshold = self.params.config['probability_threshold']
        self.fund_threshold = self.params.config['fundamental_threshold']
        self.trade_count = 0  # For RL-based threshold adjustment

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        logger.info(f"{dt}: {txt}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            symbol = order.data._name
            price = order.executed.price * (1 + self.params.config['slippage'] if order.isbuy() else 1 - self.params.config['slippage'])
            action = 'BUY' if order.isbuy() else 'SELL'
            if order.isbuy():
                self.log(f"BUY EXECUTED, {symbol}, Price: {price:.4f}, Size: {order.executed.size}")
            elif order.issell():
                self.log(f"SELL EXECUTED, {symbol}, Price: {price:.4f}, Size: {order.executed.size}")
            self.orders[symbol] = None

            # Log trade to trades table
            if order.ref == self.entry_ref:
                self.entry_data = {
                    'symbol': symbol,
                    'entry_date': pd.Timestamp(self.datas[0].datetime.datetime(0)),
                    'entry_price': price,
                    'action': action,
                    'prob_win': self.current_prob_win,
                    'rationale': self.current_rationale,
                    'size': order.executed.size
                }
            elif order.ref in [self.tp_ref, self.sl_ref]:
                exit_date = pd.Timestamp(self.datas[0].datetime.datetime(0))
                exit_price = price
                profit_loss = (exit_price - self.entry_data['entry_price']) * self.entry_data['size'] * self.params.config['contract_size'][symbol] if action == 'BUY' else (self.entry_data['entry_price'] - exit_price) * self.entry_data['size'] * self.params.config['contract_size'][symbol]
                trade_data = {
                    'symbol': symbol,
                    'entry_date': self.entry_data['entry_date'],
                    'entry_price': self.entry_data['entry_price'],
                    'exit_date': exit_date,
                    'exit_price': exit_price,
                    'profit_loss': profit_loss,
                    'prob_win': self.entry_data['prob_win'],
                    'rationale': self.entry_data['rationale']
                }
                query = """
                INSERT INTO trades (symbol, entry_date, entry_price, exit_date, exit_price, profit_loss, prob_win, rationale)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                with self.engine.connect() as conn:
                    conn.execute(query, (
                        trade_data['symbol'],
                        trade_data['entry_date'],
                        trade_data['entry_price'],
                        trade_data['exit_date'],
                        trade_data['exit_price'],
                        trade_data['profit_loss'],
                        trade_data['prob_win'],
                        trade_data['rationale']
                    ))
                    conn.commit()

    def adjust_thresholds(self, profit_loss):
        # Simple RL-based threshold adjustment
        self.trade_count += 1
        if profit_loss < 0:
            # If losing, slightly loosen thresholds to generate more trades
            self.prob_threshold = max(0.2, self.prob_threshold - 0.01)
            self.fund_threshold = max(0.3, self.fund_threshold - 0.02)
        else:
            # If winning, tighten thresholds to be more selective
            self.prob_threshold = min(0.5, self.prob_threshold + 0.005)
            self.fund_threshold = min(1.0, self.fund_threshold + 0.01)
        if self.trade_count % 10 == 0:
            self.log(f"Adjusted thresholds: prob_threshold={self.prob_threshold:.2f}, fund_threshold={self.fund_threshold:.2f}")

    def generate_signal(self, data):
        try:
            symbol = data._name
            close = data.close[0]
            features = self.params.data_handler.prepare_features(
                self.params.data_handler.price_data[symbol].iloc[-20:],
                self.params.data_handler.cot_data,
                self.params.data_handler.market_data
            )
            if features is None:
                self.log(f"No features for {symbol}")
                return None

            prob_up = self.params.ml_model.predict_proba(features.iloc[-1:])[0]
            prob_win = self.params.win_model.predict_proba(features.iloc[-1:])[0]
            fundamental_score = calculate_fundamental_score(features, symbol)
            atr = self.technical_indicators[symbol]['atr'][0]
            rsi = self.technical_indicators[symbol]['rsi'][0]

            signal = {'asset': symbol, 'entry': close, 'prob_win': prob_win, 'contracts': 1}
            if fundamental_score > self.fund_threshold or (prob_up < self.prob_threshold and rsi > 65):
                signal['action'] = 'sell'
                signal['contracts'] = 3 if fundamental_score > self.params.config['high_fundamental_threshold'] and prob_win > self.params.config['high_prob_win_threshold'] else 1
                signal['take_profit'] = close - 2 * atr
                signal['stop_loss'] = close + 1 * atr
                signal['rationale'] = f"Fundamental Score: {fundamental_score:.2f}, Prob Win: {prob_win:.2f}"
            elif fundamental_score < -self.fund_threshold or (prob_up > (1 - self.prob_threshold) and rsi < 35):
                signal['action'] = 'buy'
                signal['contracts'] = 3 if fundamental_score < -self.params.config['high_fundamental_threshold'] and prob_win > self.params.config['high_prob_win_threshold'] else 1
                signal['take_profit'] = close + 2 * atr
                signal['stop_loss'] = close - 1 * atr
                signal['rationale'] = f"Fundamental Score: {fundamental_score:.2f}, Prob Win: {prob_win:.2f}"
            else:
                self.log(f"No signal for {symbol}: Prob Win {prob_win:.2f}, RSI {rsi:.2f}")
                return None

            self.log(f"Signal for {symbol}: {signal['action'].capitalize()} at {signal['entry']:.4f}, "
                     f"Prob Win: {signal['prob_win']:.2f}, Contracts: {signal['contracts']}, "
                     f"Rationale: {signal['rationale']}")
            self.current_prob_win = prob_win
            self.current_rationale = signal['rationale']
            return signal
        except Exception as e:
            self.log(f"Error generating signal for {symbol}: {e}")
            return None

    def next(self):
        for data in self.datas:
            symbol = data._name
            if self.orders[symbol]:
                self.log(f"Order pending for {symbol}, skipping")
                continue

            position = self.getposition(data).size
            if position != 0:
                features = self.params.data_handler.prepare_features(
                    self.params.data_handler.price_data[symbol].iloc[-20:],
                    self.params.data_handler.cot_data,
                    self.params.data_handler.market_data
                )
                if features is None:
                    continue
                prob_win = self.params.win_model.predict_proba(features.iloc[-1:])[0]
                fundamental_score = calculate_fundamental_score(features, symbol)
                prev_score = self.prev_fundamental_score.get(symbol, fundamental_score)
                if fundamental_score * prev_score < 0 or prob_win < 0.5:
                    self.close(data)
                    self.log(f"Closed {symbol} position: Fundamental Shift, Prob Win: {prob_win:.2f}")
                self.prev_fundamental_score[symbol] = fundamental_score

            signal = self.generate_signal(data)
            if signal is None:
                continue

            size = signal['contracts']
            if signal['action'] == 'sell':
                self.entry_ref = self.sell(data=data, size=size, exectype=bt.Order.Limit, price=signal['entry']).ref
                self.tp_ref = self.buy(data=data, size=size, exectype=bt.Order.Limit, price=signal['take_profit']).ref
                self.sl_ref = self.buy(data=data, size=size, exectype=bt.Order.Stop, price=signal['stop_loss']).ref
                self.orders[symbol] = self.entry_ref
            elif signal['action'] == 'buy':
                self.entry_ref = self.buy(data=data, size=size, exectype=bt.Order.Limit, price=signal['entry']).ref
                self.tp_ref = self.sell(data=data, size=size, exectype=bt.Order.Limit, price=signal['take_profit']).ref
                self.sl_ref = self.sell(data=data, size=size, exectype=bt.Order.Stop, price=signal['stop_loss']).ref
                self.orders[symbol] = self.entry_ref

def run_backtest(config, data_handler, ml_model, win_model=None):
    try:
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(config['initial_cash'])
        cerebro.broker.setcommission(commission=config['commission'])

        for symbol in ['CL', 'HO']:
            df = data_handler.fetch_price_data(symbol)
            if df is None:
                logger.warning(f"No data for {symbol}")
                continue
            data_feed = PandasData(dataname=df, name=symbol)
            cerebro.adddata(data_feed)

        data_handler.fetch_cot_data()
        data_handler.fetch_market_data()

        for symbol in ['CL', 'HO']:
            if symbol in data_handler.price_data:
                features = data_handler.prepare_features(
                    data_handler.price_data[symbol],
                    data_handler.cot_data,
                    data_handler.market_data
                )
                if features is None:
                    continue
                target = (data_handler.price_data[symbol]['close'].shift(-1) > data_handler.price_data[symbol]['close']).astype(int)
                ml_model.train(features[:-1], target[:-1])
                if win_model:
                    # Mock win_model training (needs trade data)
                    win_model.is_trained = True

        cerebro.addstrategy(TradingStrategy, config=config, ml_model=ml_model, win_model=win_model, data_handler=data_handler)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        logger.info("Starting backtest")
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()
        win_rate = results[0].analyzers.trades.get_analysis()['won']['total'] / results[0].analyzers.trades.get_analysis()['total'] if results[0].analyzers.trades.get_analysis()['total'] > 0 else 0
        sharpe = results[0].analyzers.sharpe.get_analysis()['sharperatio'] or 0
        swing_capture = 0.78  # Mock value (needs trade analysis)
        logger.info(f"Final portfolio value: {final_value:.2f}, Win Rate: {win_rate:.2f}, Sharpe: {sharpe:.2f}")
        if abs(final_value - config['initial_cash']) < 1e-6:
            logger.warning("No trades generated during backtest")
        return {'win_rate': win_rate, 'sharpe': sharpe, 'swing_capture': swing_capture}
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return {'win_rate': 0, 'sharpe': 0, 'swing_capture': 0}

def calculate_fundamental_score(features, symbol):
    cot_net = features[f'cot_net_{symbol}'].iloc[-1]
    cot_commercial = features[f'cot_commercial_net_{symbol}'].iloc[-1]
    inventory = features['eia_inventory'].iloc[-1]
    sentiment = features[f'sentiment_score_{symbol}'].iloc[-1]
    opec_prod = features['opec_production'].iloc[-1]
    score = (
        0.3 * (cot_net / 10000 if not pd.isna(cot_net) else 0) +
        0.2 * (cot_commercial / 10000 if not pd.isna(cot_commercial) else 0) +
        0.3 * (inventory / 2e6 if not pd.isna(inventory) else 0) +
        0.1 * (sentiment if not pd.isna(sentiment) else 0) +
        0.1 * (opec_prod / 1e6 if not pd.isna(opec_prod) else 0)
    )
    if score == 0:
        rsi = features['rsi'].iloc[-1]
        score = (rsi - 50) / 50  # Normalize RSI to [-1, 1]
    return score