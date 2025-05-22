import sys
  import os
  sys.path.append(os.path.dirname(os.path.abspath(__file__)))
  from connectors.postgres_connector import PostgresConnector

  # Connect to the database
  db = PostgresConnector()

  # Check what symbols we have
  symbols = db.query("SELECT DISTINCT symbol FROM market_data")
  print("Available symbols:", [s['symbol'] for s in symbols])

  # Check date ranges for CL and HO
  for symbol in ['CL', 'HO']:
      date_range = db.query_one("""
          SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date
          FROM market_data
          WHERE symbol = %s
      """, [symbol])
      print(f"{symbol} date range: {date_range['min_date']} to {date_range['max_date']}")

  # Check date counts
  for symbol in ['CL', 'HO']:
      count = db.query_one("""
          SELECT COUNT(*) as count
          FROM market_data
          WHERE symbol = %s
      """, [symbol])
      print(f"{symbol} data points: {count['count']}")

  print("\nNow let's check if market_data_backtest view works:")
  cl_ho_spread = db.query("""
      SELECT * FROM market_data_backtest WHERE symbol = 'CL-HO-SPREAD' LIMIT 5
  """)
  print(f"CL-HO-SPREAD records found: {len(cl_ho_spread)}")
  if cl_ho_spread:
      print("Sample record:", cl_ho_spread[0])