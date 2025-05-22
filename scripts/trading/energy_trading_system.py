# energy_backtesting.py
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def connect_to_postgres():
    """Connect to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="trading_db",
            user="postgres",
            password="Makingmoney25!"
        )
        print("Successfully connected to PostgreSQL")
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def calculate_crack_spread(conn, start_date='1992-01-01', end_date=None):
    """Calculate the crack spread between heating oil and crude oil"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Calculating crack spread from {start_date} to {end_date}...")
    
    # Query to get both crude oil and heating oil prices
    query = """
    WITH crude AS (
        SELECT 
            timestamp::date AS date,
            symbol,
            close AS price
        FROM market_data
        WHERE symbol = 'CL'
        AND timestamp::date BETWEEN %s AND %s
    ),
    heating AS (
        SELECT 
            timestamp::date AS date,
            symbol,
            close AS price
        FROM market_data
        WHERE symbol = 'HO'
        AND timestamp::date BETWEEN %s AND %s
    )
    SELECT 
        c.date,
        c.price AS crude_price,
        h.price AS heating_price,
        (h.price * 42) - c.price AS crack_spread
    FROM crude c
    JOIN heating h ON c.date = h.date
    ORDER BY c.date;
    """
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, (start_date, end_date, start_date, end_date))
        data = cursor.fetchall()
        cursor.close()
        
        # Convert to pandas DataFrame for analysis
        df = pd.DataFrame(data)
        
        if not df.empty:
            print(f"Retrieved {len(df)} days of crack spread data")
            # Calculate rolling statistics for z-score
            df['sma_90'] = df['crack_spread'].rolling(window=90).mean()
            df['std_90'] = df['crack_spread'].rolling(window=90).std()
            df['z_score'] = (df['crack_spread'] - df['sma_90']) / df['std_90']
            
            # Generate trading signals based on z-score
            df['signal'] = np.where(df['z_score'] < -2, 'BUY', 
                        np.where(df['z_score'] > 2, 'SELL', 'NEUTRAL'))
            
            return df
        else:
            print("No data found for the specified date range")
            return pd.DataFrame()
    
    except Exception as e:
        print(f"Error calculating crack spread: {e}")
        return pd.DataFrame()

def analyze_seasonal_pattern(conn):
    """Analyze seasonal patterns in crack spread"""
    print("Analyzing seasonal patterns...")
    
    query = """
    WITH crack_data AS (
        SELECT 
            md_cl.timestamp::date AS date,
            md_cl.close AS crude_price,
            md_ho.close AS heating_price,
            (md_ho.close * 42) - md_cl.close AS crack_spread
        FROM market_data md_cl
        JOIN market_data md_ho ON md_cl.timestamp::date = md_ho.timestamp::date
        WHERE md_cl.symbol = 'CL' AND md_ho.symbol = 'HO'
    )
    SELECT 
        EXTRACT(MONTH FROM date) AS month,
        AVG(crack_spread) AS avg_crack_spread,
        STDDEV(crack_spread) AS std_crack_spread,
        MIN(crack_spread) AS min_crack_spread,
        MAX(crack_spread) AS max_crack_spread,
        COUNT(*) AS sample_count
    FROM crack_data
    GROUP BY EXTRACT(MONTH FROM date)
    ORDER BY EXTRACT(MONTH FROM date);
    """
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()
        
        df = pd.DataFrame(data)
        
        if not df.empty:
            print(f"Retrieved seasonal analysis for {len(df)} months")
            # Plot seasonal pattern
            plt.figure(figsize=(12, 6))
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            plt.bar(months, df['avg_crack_spread'])
            plt.errorbar(months, df['avg_crack_spread'], 
                        yerr=df['std_crack_spread'], fmt='o', color='red')
            
            plt.title('Seasonal Pattern in Heating Oil - Crude Oil Crack Spread')
            plt.ylabel('Average Crack Spread ($ per barrel)')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.savefig('seasonal_pattern.png')
            plt.show()
            
            print("Seasonal analysis complete, chart saved as 'seasonal_pattern.png'")
        
        return df
    
    except Exception as e:
        print(f"Error analyzing seasonal pattern: {e}")
        return pd.DataFrame()

def backtest_crack_spread_strategy(conn, start_date, end_date=None, z_threshold=2.0, seasonal_weight=0.0, weather_weight=0.0, cot_weight=0.0):
    """Backtest crack spread trading strategy with multiple factors"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    print(f"Backtesting crack spread strategy from {start_date} to {end_date} with z-threshold {z_threshold}...")
    
    try:
        # Get crack spread data
        crack_data = calculate_crack_spread(conn, start_date, end_date)
        
        if crack_data.empty:
            print("No data available for backtesting")
            return None
        
        # Initialize columns for backtesting
        crack_data['position'] = 0  # 1 for long, -1 for short, 0 for neutral
        crack_data['entry_price'] = np.nan
        crack_data['exit_price'] = np.nan
        crack_data['trade_pnl'] = 0.0
        
        # Add seasonal factor if weight > 0
        if seasonal_weight > 0:
            # Convert date to month
            crack_data['month'] = pd.DatetimeIndex(crack_data['date']).month
            
            # Define seasonal factor (1-5 scale, higher in winter)
            season_map = {
                1: 5,  # January - strong winter
                2: 5,  # February - strong winter
                3: 4,  # March - late winter
                4: 3,  # April - spring
                5: 2,  # May - late spring
                6: 1,  # June - summer
                7: 1,  # July - summer
                8: 1,  # August - summer
                9: 2,  # September - early fall
                10: 3, # October - fall
                11: 4, # November - early winter
                12: 5  # December - strong winter
            }
            
            crack_data['seasonal_factor'] = crack_data['month'].map(season_map)
            
            # Convert to -1 to 1 scale
            crack_data['seasonal_signal'] = (crack_data['seasonal_factor'] - 3) / 2
        else:
            crack_data['seasonal_signal'] = 0
        
        # Add weather factor if needed (simplified for backtest)
        if weather_weight > 0:
            # For simplicity, we'll use a proxy based on month
            # In a full implementation, we would join with actual weather data
            # This is just for demonstration
            crack_data['weather_signal'] = np.where(
                crack_data['month'].isin([12, 1, 2]), 
                1.0,  # Cold weather - bullish for heating oil
                np.where(
                    crack_data['month'].isin([6, 7, 8]),
                    -1.0,  # Warm weather - bearish for heating oil
                    0.0    # Moderate weather - neutral
                )
            )
        else:
            crack_data['weather_signal'] = 0
        
        # Add COT factor if needed (simplified)
        if cot_weight > 0:
            # In a full implementation, we would join with actual COT data
            # For simplicity, we'll use a random signal
            # This is just for demonstration
            np.random.seed(42)  # For reproducibility
            crack_data['cot_signal'] = np.random.uniform(-1, 1, size=len(crack_data))
        else:
            crack_data['cot_signal'] = 0
        
        # Calculate composite signal
        z_score_weight = 1.0 - seasonal_weight - weather_weight - cot_weight
        
        crack_data['composite_signal'] = (
            (-1 * crack_data['z_score'] * z_score_weight) +  # Negative because we want to buy low z-scores
            (crack_data['seasonal_signal'] * seasonal_weight) +
            (crack_data['weather_signal'] * weather_weight) +
            (crack_data['cot_signal'] * cot_weight)
        )
        
        # Generate positions based on composite signal
        for i in range(1, len(crack_data)):
            prev_signal = crack_data.iloc[i-1]['composite_signal']
            curr_signal = crack_data.iloc[i]['composite_signal']
            
            # Entry signals
            if pd.notnull(prev_signal) and pd.notnull(curr_signal):
                # Buy signal
                if prev_signal <= 0.5 and curr_signal > 0.5:
                    crack_data.loc[crack_data.index[i], 'position'] = 1
                    crack_data.loc[crack_data.index[i], 'entry_price'] = crack_data.iloc[i]['crack_spread']
                
                # Sell signal
                elif prev_signal >= -0.5 and curr_signal < -0.5:
                    crack_data.loc[crack_data.index[i], 'position'] = -1
                    crack_data.loc[crack_data.index[i], 'entry_price'] = crack_data.iloc[i]['crack_spread']
                
                # Exit long position
                elif crack_data.iloc[i-1]['position'] == 1 and curr_signal < 0:
                    crack_data.loc[crack_data.index[i], 'position'] = 0
                    crack_data.loc[crack_data.index[i], 'exit_price'] = crack_data.iloc[i]['crack_spread']
                    crack_data.loc[crack_data.index[i], 'trade_pnl'] = (
                        crack_data.iloc[i]['crack_spread'] - crack_data.iloc[i-1]['entry_price']
                    )
                
                # Exit short position
                elif crack_data.iloc[i-1]['position'] == -1 and curr_signal > 0:
                    crack_data.loc[crack_data.index[i], 'position'] = 0
                    crack_data.loc[crack_data.index[i], 'exit_price'] = crack_data.iloc[i]['crack_spread']
                    crack_data.loc[crack_data.index[i], 'trade_pnl'] = (
                        crack_data.iloc[i-1]['entry_price'] - crack_data.iloc[i]['crack_spread']
                    )
                
                # Maintain current position
                else:
                    crack_data.loc[crack_data.index[i], 'position'] = crack_data.iloc[i-1]['position']
                    if crack_data.iloc[i-1]['position'] != 0:
                        crack_data.loc[crack_data.index[i], 'entry_price'] = crack_data.iloc[i-1]['entry_price']
        
        # Calculate cumulative P&L
        crack_data['cumulative_pnl'] = crack_data['trade_pnl'].cumsum()
        
        # Calculate performance metrics
        total_trades = len(crack_data[crack_data['trade_pnl'] != 0])
        winning_trades = len(crack_data[crack_data['trade_pnl'] > 0])
        losing_trades = len(crack_data[crack_data['trade_pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = crack_data[crack_data['trade_pnl'] > 0]['trade_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = crack_data[crack_data['trade_pnl'] < 0]['trade_pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades) / abs(avg_loss * losing_trades) if losing_trades > 0 else float('inf')
        
        # Calculate annualized return and Sharpe ratio
        if not crack_data.empty:
            days = (pd.to_datetime(crack_data['date'].iloc[-1]) - pd.to_datetime(crack_data['date'].iloc[0])).days
            years = days / 365.0
            
            if crack_data['cumulative_pnl'].iloc[-1] > 0:
                annualized_return = ((1 + crack_data['cumulative_pnl'].iloc[-1]/100) ** (1/years) - 1) * 100
            else:
                annualized_return = 0
                
            # Daily returns
            crack_data['daily_return'] = crack_data['trade_pnl'].fillna(0)
            
            # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
            daily_sharpe = crack_data['daily_return'].mean() / crack_data['daily_return'].std() if crack_data['daily_return'].std() > 0 else 0
            sharpe_ratio = daily_sharpe * np.sqrt(252)  # Annualized
        else:
            annualized_return = 0
            sharpe_ratio = 0
        
        # Print backtest results
        print(f"\nBacktest Results ({start_date} to {end_date}):")
        print(f"Strategy Parameters:")
        print(f"- Z-Score Threshold: {z_threshold}")
        print(f"- Z-Score Weight: {z_score_weight:.2f}")
        print(f"- Seasonal Weight: {seasonal_weight:.2f}")
        print(f"- Weather Weight: {weather_weight:.2f}")
        print(f"- COT Weight: {cot_weight:.2f}")
        print(f"\nPerformance Metrics:")
        print(f"- Total Trades: {total_trades}")
        print(f"- Win Rate: {win_rate:.2%}")
        print(f"- Average Win: ${avg_win:.2f}")
        print(f"- Average Loss: ${avg_loss:.2f}")
        print(f"- Profit Factor: {profit_factor:.2f}")
        print(f"- Net P&L: ${crack_data['cumulative_pnl'].iloc[-1]:.2f}")
        print(f"- Annualized Return: {annualized_return:.2f}%")
        print(f"- Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Plot backtest results
        plt.figure(figsize=(12, 10))
        
        # Plot crack spread with signals
        plt.subplot(2, 1, 1)
        plt.plot(crack_data['date'], crack_data['crack_spread'], label='Crack Spread')
        plt.plot(crack_data['date'], crack_data['sma_90'], label='90-day SMA', linestyle='--')
        
        # Add bands at z-score thresholds
        plt.fill_between(
            crack_data['date'],
            crack_data['sma_90'] - z_threshold * crack_data['std_90'],
            crack_data['sma_90'] + z_threshold * crack_data['std_90'],
            alpha=0.2, color='gray'
        )
        
        # Mark buy signals
        buy_signals = crack_data[crack_data['position'] == 1]
        if not buy_signals.empty:
            plt.scatter(buy_signals['date'], buy_signals['crack_spread'], 
                      marker='^', color='g', s=100, label='Buy Signal')
        
        # Mark sell signals
        sell_signals = crack_data[crack_data['position'] == -1]
        if not sell_signals.empty:
            plt.scatter(sell_signals['date'], sell_signals['crack_spread'], 
                      marker='v', color='r', s=100, label='Sell Signal')
        
        plt.title('Crack Spread Trading Signals')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Plot strategy equity curve
        plt.subplot(2, 1, 2)
        plt.plot(crack_data['date'], crack_data['cumulative_pnl'], label='Strategy P&L')
        plt.title('Strategy Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Cumulative P&L ($)')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        plt.show()
        
        print("Backtest complete, chart saved as 'backtest_results.png'")
        
        return crack_data
    
    except Exception as e:
        print(f"Error in backtesting: {e}")
        return None

def optimize_strategy_parameters(conn, start_date, end_date=None):
    """Optimize strategy parameters through grid search"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Optimizing strategy parameters for period {start_date} to {end_date}...")
    
    # Define parameter grid
    z_thresholds = [1.5, 2.0, 2.5, 3.0]
    seasonal_weights = [0.0, 0.1, 0.2, 0.3]
    weather_weights = [0.0, 0.1, 0.2]
    cot_weights = [0.0, 0.1]
    
    # Results storage
    results = []
    
    # Grid search
    for z in z_thresholds:
        for sw in seasonal_weights:
            for ww in weather_weights:
                for cw in cot_weights:
                    # Skip invalid weight combinations
                    if sw + ww + cw >= 1.0:
                        continue
                    
                    print(f"Testing z={z}, seasonal={sw}, weather={ww}, cot={cw}")
                    
                    # Run backtest
                    result = backtest_crack_spread_strategy(
                        conn, 
                        start_date, 
                        end_date, 
                        z_threshold=z,
                        seasonal_weight=sw,
                        weather_weight=ww,
                        cot_weight=cw
                    )
                    
                    if result is not None:
                        # Calculate performance metrics
                        total_trades = len(result[result['trade_pnl'] != 0])
                        win_rate = len(result[result['trade_pnl'] > 0]) / total_trades if total_trades > 0 else 0
                        profit = result['cumulative_pnl'].iloc[-1]
                        
                        # Calculate Sharpe ratio
                        result['daily_return'] = result['trade_pnl'].fillna(0)
                        daily_sharpe = result['daily_return'].mean() / result['daily_return'].std() if result['daily_return'].std() > 0 else 0
                        sharpe = daily_sharpe * np.sqrt(252)  # Annualized
                        
                        # Store results
                        results.append({
                            'z_threshold': z,
                            'seasonal_weight': sw,
                            'weather_weight': ww,
                            'cot_weight': cw,
                            'total_trades': total_trades,
                            'win_rate': win_rate,
                            'profit': profit,
                            'sharpe_ratio': sharpe
                        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        # Sort by Sharpe ratio
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)
        
        print("\nTop 5 Parameter Combinations by Sharpe Ratio:")
        print(results_df.head(5))
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.scatter(results_df['z_threshold'], results_df['sharpe_ratio'])
        plt.title('Z-Threshold vs Sharpe Ratio')
        plt.xlabel('Z-Threshold')
        plt.ylabel('Sharpe Ratio')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.scatter(results_df['seasonal_weight'], results_df['sharpe_ratio'])
        plt.title('Seasonal Weight vs Sharpe Ratio')
        plt.xlabel('Seasonal Weight')
        plt.ylabel('Sharpe Ratio')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.scatter(results_df['total_trades'], results_df['sharpe_ratio'])
        plt.title('Total Trades vs Sharpe Ratio')
        plt.xlabel('Total Trades')
        plt.ylabel('Sharpe Ratio')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.scatter(results_df['win_rate'], results_df['profit'])
        plt.title('Win Rate vs Profit')
        plt.xlabel('Win Rate')
        plt.ylabel('Profit ($)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('optimization_results.png')
        plt.show()
        
        print("Optimization complete, chart saved as 'optimization_results.png'")
        
        # Get best parameters
        best_params = results_df.iloc[0]
        print("\nBest Parameters:")
        print(f"Z-Threshold: {best_params['z_threshold']}")
        print(f"Seasonal Weight: {best_params['seasonal_weight']}")
        print(f"Weather Weight: {best_params['weather_weight']}")
        print(f"COT Weight: {best_params['cot_weight']}")
        print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
        print(f"Total Profit: ${best_params['profit']:.2f}")
        
        return results_df
    else:
        print("No valid parameter combinations found")
        return None

def run_interactive_backtesting():
    """Run interactive backtesting"""
    conn = connect_to_postgres()
    if not conn:
        return
    
    try:
        print("\n===== ENERGY TRADING SYSTEM BACKTESTING =====")
        
        while True:
            print("\nOptions:")
            print("1. Run single backtest")
            print("2. Run optimization")
            print("3. Analyze seasonal patterns")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ")
            
            if choice == '1':
                start_date = input("Enter start date (YYYY-MM-DD): ")
                end_date = input("Enter end date (YYYY-MM-DD) or press Enter for today: ")
                if not end_date.strip():
                    end_date = datetime.now().strftime('%Y-%m-%d')
                
                z_threshold = float(input("Enter Z-score threshold (default 2.0): ") or 2.0)
                seasonal_weight = float(input("Enter seasonal weight 0-0.5 (default 0.0): ") or 0.0)
                weather_weight = float(input("Enter weather weight 0-0.5 (default 0.0): ") or 0.0)
                cot_weight = float(input("Enter COT weight 0-0.5 (default 0.0): ") or 0.0)
                
                # Validate weights sum to less than 1
                if seasonal_weight + weather_weight + cot_weight >= 1.0:
                    print("Error: Weights must sum to less than 1.0")
                    continue
                
                backtest_crack_spread_strategy(
                    conn, 
                    start_date, 
                    end_date, 
                    z_threshold=z_threshold,
                    seasonal_weight=seasonal_weight,
                    weather_weight=weather_weight,
                    cot_weight=cot_weight
                )
                
            elif choice == '2':
                start_date = input("Enter start date (YYYY-MM-DD): ")
                end_date = input("Enter end date (YYYY-MM-DD) or press Enter for today: ")
                if not end_date.strip():
                    end_date = datetime.now().strftime('%Y-%m-%d')
                
                optimize_strategy_parameters(conn, start_date, end_date)
                
            elif choice == '3':
                analyze_seasonal_pattern(conn)
                
            elif choice == '4':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please enter a number between 1 and 4.")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    
    finally:
        conn.close()
        print("Database connection closed")

if __name__ == "__main__":
    run_interactive_backtesting()