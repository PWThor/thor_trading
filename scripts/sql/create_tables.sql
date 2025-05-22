-- Connect to the database
\c trading_db

-- Create table for market data (OHLCV + fundamentals)
CREATE TABLE market_data (
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume BIGINT,
    cot_commercial_net FLOAT,       -- COT: Net positions of commercial hedgers
    cot_noncommercial_net FLOAT,    -- COT: Net positions of non-commercial speculators
    eia_inventory FLOAT,            -- EIA: Crude oil or distillate inventories (barrels)
    eia_production FLOAT,           -- EIA: Crude oil production (barrels per day)
    crude_imports FLOAT,            -- EIA: Weekly U.S. crude oil imports (barrels)
    crude_exports FLOAT,            -- EIA: Weekly U.S. crude oil exports (barrels)
    spr_stocks FLOAT,               -- EIA: Strategic Petroleum Reserve crude oil stocks (barrels)
    petroleum_demand FLOAT,         -- EIA: Total U.S. product supplied (barrels per day)
    distillate_demand FLOAT,        -- EIA: Distillate fuel oil supplied (barrels per day)
    opec_production FLOAT,          -- OPEC: Actual production (barrels per day)
    opec_demand_forecast FLOAT,     -- OPEC: Global oil demand forecast (barrels per day)
    api_gravity FLOAT,              -- API gravity of crude oil
    sulfur_content FLOAT,           -- Sulfur content of crude oil (%)
    rig_count_us FLOAT,             -- Baker Hughes: U.S. oil rig count
    rig_count_global FLOAT,         -- Baker Hughes: Global oil rig count
    refinery_utilization FLOAT,     -- EIA: U.S. refinery utilization rate (%)
    gdp_growth FLOAT,               -- Economic: U.S. GDP growth rate (%)
    unemployment_rate FLOAT,        -- Economic: U.S. unemployment rate (%)
    cpi FLOAT,                      -- Economic: Consumer Price Index
    avg_temperature FLOAT,          -- Weather: Average temperature in key regions (°F)
    precipitation FLOAT,            -- Weather: Precipitation amount (mm)
    wind_speed FLOAT,               -- Weather: Wind speed (m/s)
    hurricane_event BOOLEAN,        -- Weather: Hurricane occurrence (true/false)
    sentiment_score FLOAT,          -- Sentiment: Derived from GROWMARK reports (-1 to 1)
    PRIMARY KEY (timestamp, symbol)
);

-- Create indexes for fast queries
CREATE INDEX idx_market_data_symbol_timestamp ON market_data (symbol, timestamp);

-- Create table for orders
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    quantity INT,
    price FLOAT,
    status VARCHAR(20),
    timestamp TIMESTAMP
);

CREATE INDEX idx_orders_symbol_timestamp ON orders (symbol, timestamp);

-- Create table for positions
CREATE TABLE positions (
    symbol VARCHAR(10) PRIMARY KEY,
    quantity INT,
    average_price FLOAT,
    pnl FLOAT
);

-- Create table for ML features
CREATE TABLE ml_features (
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    sma10 FLOAT,
    sma50 FLOAT,
    rsi FLOAT,
    macd FLOAT,
    PRIMARY KEY (timestamp, symbol)
);

-- Create table for ML predictions
CREATE TABLE ml_predictions (
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    prediction FLOAT,
    confidence FLOAT,
    PRIMARY KEY (timestamp, symbol)
);