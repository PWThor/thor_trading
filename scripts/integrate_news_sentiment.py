import pandas as pd
import psycopg2
import requests
import logging
from datetime import datetime, timedelta
import time

# Set up logging
logging.basicConfig(
    filename="E:/Projects/thor_trading/outputs/logs/integrate_news_sentiment.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def connect_db():
    return psycopg2.connect(
        dbname="trading_db",
        user="postgres",
        password="secure_password",  # Replace with your password
        host="localhost",
        port="5432"
    )

def fetch_news_sentiment(api_key, start_date, end_date):
    # Convert dates to Alpha Vantage format (YYYYMMDDTHHMM)
    start_str = start_date.strftime("%Y%m%dT%H%M")
    end_str = end_date.strftime("%Y%m%dT%H%M")

    # Parameters for the API call
    params = {
        'function': 'NEWS_SENTIMENT',
        'topics': 'energy_transportation',
        'tickers': 'XOM',  # Proxy ticker for energy markets (ExxonMobil)
        'time_from': start_str,
        'time_to': end_str,
        'sort': 'LATEST',
        'limit': 1000,
        'apikey': api_key
    }

    # Fetch news articles (handle pagination for large date ranges)
    all_articles = []
    while True:
        url = "https://www.alphavantage.co/query"
        response = requests.get(url, params=params)
        data = response.json()

        articles = data.get('feed', [])
        all_articles.extend(articles)

        # Check if there are more articles to fetch
        if len(articles) < 1000:
            break

        # Update time_from to the timestamp of the last article
        last_article_time = datetime.strptime(articles[-1]['time_published'], "%Y%m%dT%H%M%S")
        params['time_from'] = last_article_time.strftime("%Y%m%dT%H%M")

        # Respect rate limits (5 calls per minute for free tier)
        time.sleep(12)  # Wait 12 seconds between calls

    # Process articles into a DataFrame
    sentiment_data = []
    for article in all_articles:
        sentiment_data.append({
            'Date': datetime.strptime(article['time_published'], "%Y%m%dT%H%M%S").date(),
            'SentimentScore': article.get('overall_sentiment_score', 0)  # Range: -1 to 1
        })

    sentiment_df = pd.DataFrame(sentiment_data)
    if sentiment_df.empty:
        return pd.DataFrame(columns=['Date', 'SentimentScore'])

    # Aggregate sentiment scores by day
    daily_sentiment = sentiment_df.groupby('Date').agg({
        'SentimentScore': 'mean'
    }).reset_index()
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])

    return daily_sentiment

def integrate_news_sentiment(symbol, api_key, start_date, end_date):
    # Connect to the database
    conn = connect_db()
    cur = conn.cursor()

    # Fetch market data dates for the symbol
    cur.execute("""
        SELECT timestamp
        FROM market_data
        WHERE symbol = %s
        ORDER BY timestamp
    """, (symbol,))
    market_dates = [row[0] for row in cur.fetchall()]

    # Fetch news and sentiment data
    sentiment_df = fetch_news_sentiment(api_key, start_date, end_date)

    # Update market_data with sentiment data
    for date in market_dates:
        date_only = date.date()
        sentiment_row = sentiment_df[sentiment_df['Date'].dt.date == date_only]
        if sentiment_row.empty:
            continue

        sentiment_score = sentiment_row['SentimentScore'].iloc[0]

        try:
            cur.execute("""
                UPDATE market_data
                SET sentiment_score = %s
                WHERE timestamp = %s AND symbol = %s
            """, (
                sentiment_score,
                date,
                symbol
            ))
        except Exception as e:
            logging.error(f"Error updating sentiment data for {symbol} at {date}: {e}")
            print(f"Error updating sentiment data for {symbol} at {date}: {e}")

    conn.commit()
    cur.close()
    conn.close()
    logging.info(f"Integrated news sentiment data for {symbol}")
    print(f"Integrated news sentiment data for {symbol}")

if __name__ == "__main__":
    api_key = "3TAOMQ4MX1XH6U1N"  # Your Alpha Vantage API key
    start_date = datetime(1992, 1, 1)
    end_date = datetime(2025, 4, 8)
    for symbol in ['CL', 'HO']:
        integrate_news_sentiment(symbol, api_key, start_date, end_date)