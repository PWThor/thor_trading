#!/usr/bin/env python
"""
PostgreSQL Connector for Thor Trading System

This module provides connection to the PostgreSQL database for the Thor Trading System.
"""

import psycopg2
import psycopg2.extras
import logging

logger = logging.getLogger(__name__)

class PostgresConnector:
    """Connector for PostgreSQL database operations."""
    
    def __init__(self, host="localhost", port="5432", dbname="trading_db", 
                 user="postgres", password="Makingmoney25!"):
        """
        Initialize the PostgreSQL connector.
        
        Args:
            host (str): Database host
            port (str): Database port
            dbname (str): Database name
            user (str): Database user
            password (str): Database password
        """
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.conn = None
        
        # Try to connect to the database
        try:
            self.connect()
            logger.info(f"Connected to PostgreSQL database: {dbname} at {host}:{port}")
        except Exception as e:
            logger.warning(f"Could not connect to PostgreSQL: {str(e)}")
    
    def connect(self):
        """Establish connection to the database."""
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password
            )
            # Set autocommit mode
            self.conn.autocommit = True
    
    def test_connection(self):
        """Test if the database connection is working."""
        try:
            self.connect()
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result[0] == 1
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False
    
    def query(self, query, params=None):
        """
        Execute a query and return all results.
        
        Args:
            query (str): SQL query to execute
            params (tuple/dict): Parameters for the query
            
        Returns:
            list: List of result records as dictionaries
        """
        try:
            self.connect()
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                # Convert to list of dictionaries
                records = [dict(row) for row in results]
                return records
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            return []
    
    def query_one(self, query, params=None):
        """
        Execute a query and return the first result.
        
        Args:
            query (str): SQL query to execute
            params (tuple/dict): Parameters for the query
            
        Returns:
            dict: First result record as dictionary, or None
        """
        try:
            self.connect()
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query, params)
                result = cursor.fetchone()
                if result:
                    return dict(result)
                return None
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            return None
    
    def execute(self, query, params=None):
        """
        Execute a query without returning results (for INSERT, UPDATE, DELETE).
        
        Args:
            query (str): SQL query to execute
            params (tuple/dict): Parameters for the query
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.connect()
            with self.conn.cursor() as cursor:
                cursor.execute(query, params)
                return True
        except Exception as e:
            logger.error(f"Execute error: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            return False
    
    def store_market_data(self, symbol, timestamp, open_price, high_price, low_price, 
                         close_price, volume, **additional_fields):
        """
        Store market data in the database.
        
        Args:
            symbol (str): Market symbol
            timestamp (datetime): Timestamp of the data
            open_price (float): Opening price
            high_price (float): High price
            low_price (float): Low price
            close_price (float): Close price
            volume (float): Trading volume
            **additional_fields: Any additional fields to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.connect()
            
            # First check if record exists
            check_query = """
            SELECT id FROM market_data 
            WHERE symbol = %s AND timestamp = %s
            """
            
            with self.conn.cursor() as cursor:
                cursor.execute(check_query, (symbol, timestamp))
                exists = cursor.fetchone()
                
                if exists:
                    # Update existing record
                    columns = ['open', 'high', 'low', 'close', 'volume'] + list(additional_fields.keys())
                    values = [open_price, high_price, low_price, close_price, volume] + list(additional_fields.values())
                    
                    update_parts = [f"{col} = %s" for col in columns]
                    update_query = f"""
                    UPDATE market_data 
                    SET {', '.join(update_parts)}
                    WHERE symbol = %s AND timestamp = %s
                    """
                    
                    cursor.execute(update_query, values + [symbol, timestamp])
                else:
                    # Insert new record
                    columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume'] + list(additional_fields.keys())
                    values = [symbol, timestamp, open_price, high_price, low_price, close_price, volume] + list(additional_fields.values())
                    
                    placeholders = ', '.join(['%s'] * len(columns))
                    insert_query = f"""
                    INSERT INTO market_data ({', '.join(columns)})
                    VALUES ({placeholders})
                    """
                    
                    cursor.execute(insert_query, values)
                
                return True
                
        except Exception as e:
            logger.error(f"Error storing market data: {str(e)}")
            return False
    
    def close(self):
        """Close the database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Database connection closed")