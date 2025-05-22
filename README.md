# Thor Trading System

An automated ML-based trading system for energy futures, focusing on Crude Oil (CL) and Heating Oil (HO) contracts on NYMEX. The system uses machine learning to identify profitable trading opportunities in these markets.

## Overview

This system integrates data collection, ML model training, prediction generation, and automated trade execution to build a fully autonomous trading platform for energy futures markets.

### Key Components

1. **Data Collection Pipeline**
   - Collects market data from Interactive Brokers
   - Gathers fundamental data from EIA API
   - Collects weather data from OpenWeatherMap
   - Retrieves COT data from CFTC
   - Stores all data in PostgreSQL database

2. **ML Model Pipeline**
   - Automatic feature generation
   - Time-series based model training with XGBoost
   - Model evaluation and versioning
   - Continuous learning with regular retraining

3. **Trading System**
   - Signal generation from model predictions
   - Risk management and position sizing
   - Execution via Interactive Brokers
   - Monitoring and trailing stop management

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL database
- Interactive Brokers account and API access
- OpenWeatherMap API key
- EIA API key

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/thor_trading.git
cd thor_trading
```

2. Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Set up API keys:
```
cp pipeline/api_keys.json.example pipeline/api_keys.json
```
Then edit the `api_keys.json` file with your actual API keys.

5. Configure database connection in `connectors/postgres_connector.py`

### Usage

The system can be run in several modes:

#### Data Collection Only

```
python pipeline/main.py --collect-data
```

#### Training Models

```
python pipeline/main.py --mode train
```

#### Monitor Mode (generate signals without trading)

```
python pipeline/main.py --mode monitor --collect-data
```

#### Live Trading

```
python pipeline/main.py --mode trade --collect-data
```

### Configuration Options

- `--mode`: Operating mode (`trade`, `monitor`, `backtest`, or `train`)
- `--collect-data`: Enable data collection
- `--api-keys`: Path to API keys file (JSON format)
- `--max-positions`: Maximum number of concurrent positions
- `--risk-per-trade`: Risk per trade as a fraction of account equity
- `--confidence`: Minimum confidence threshold for trading

## Project Structure

- `/backtesting`: Backtesting framework
- `/connectors`: Database and broker connectors
- `/data`: Data storage and collection scripts
- `/features`: Feature generation
- `/models`: ML model training and prediction
- `/pipeline`: System orchestration
- `/trading`: Trading strategy and execution
- `/scripts`: Utility scripts

## License

This project is licensed under the MIT License - see the LICENSE file for details.