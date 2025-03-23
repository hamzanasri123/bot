"""
Data module for fetching and processing market data from Yahoo Finance API.
"""

import sys
import os
import pandas as pd
import numpy as np
import sqlite3
import json
import datetime
import time
from pathlib import Path

# Add path for data API access
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEFAULT_SYMBOLS, TIMEFRAMES, DEFAULT_TIMEFRAME, 
    HISTORICAL_DATA_DAYS, DATABASE
)

class DataManager:
    """
    Handles all data operations including fetching, processing, and storage.
    """
    
    def __init__(self, db_path=None):
        """
        Initialize the DataManager.
        
        Args:
            db_path (str, optional): Path to the SQLite database. Defaults to None.
        """
        self.api_client = ApiClient()
        self.db_path = db_path or DATABASE['path']
        self._ensure_db_path()
        self.conn = self._create_connection()
        self._create_tables()
        
    def _ensure_db_path(self):
        """Ensure the database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
    def _create_connection(self):
        """Create a database connection to the SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            return None
        
    def _create_tables(self):
        """Create necessary database tables if they don't exist."""
        try:
            cursor = self.conn.cursor()
            
            # Create symbols table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS symbols (
                symbol_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT UNIQUE NOT NULL,
                company_name TEXT,
                sector TEXT,
                industry TEXT,
                is_active INTEGER DEFAULT 1,
                added_date TEXT DEFAULT CURRENT_TIMESTAMP,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create OHLCV data table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                data_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                adjusted_close REAL,
                FOREIGN KEY (symbol_id) REFERENCES symbols (symbol_id),
                UNIQUE (symbol_id, timestamp, timeframe)
            )
            ''')
            
            # Create technical indicators table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicators (
                indicator_id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_id INTEGER NOT NULL,
                indicator_type TEXT NOT NULL,
                parameters TEXT NOT NULL,
                value REAL NOT NULL,
                calculated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (data_id) REFERENCES ohlcv_data (data_id)
            )
            ''')
            
            # Create index for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv_data (symbol_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_indicators_data ON technical_indicators (data_id)')
            
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database table creation error: {e}")
    
    def add_symbols(self, symbols):
        """
        Add symbols to the database.
        
        Args:
            symbols (list): List of symbol strings to add.
        """
        try:
            cursor = self.conn.cursor()
            for symbol in symbols:
                cursor.execute(
                    "INSERT OR IGNORE INTO symbols (ticker) VALUES (?)",
                    (symbol,)
                )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error adding symbols: {e}")
    
    def get_symbol_id(self, ticker):
        """
        Get the symbol_id for a given ticker.
        
        Args:
            ticker (str): The ticker symbol.
            
        Returns:
            int: The symbol_id or None if not found.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT symbol_id FROM symbols WHERE ticker = ?",
                (ticker,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            print(f"Error getting symbol_id: {e}")
            return None
    
    def fetch_historical_data(self, symbol, interval=DEFAULT_TIMEFRAME, range_days=HISTORICAL_DATA_DAYS):
        """
        Fetch historical data for a symbol from Yahoo Finance API.
        
        Args:
            symbol (str): The ticker symbol.
            interval (str): Data interval (e.g., '1d', '1h').
            range_days (int): Number of days of historical data.
            
        Returns:
            pandas.DataFrame: DataFrame with historical data or None if error.
        """
        try:
            # Convert days to appropriate range parameter
            if range_days <= 7:
                range_param = f"{range_days}d"
            elif range_days <= 30:
                range_param = "1mo"
            elif range_days <= 90:
                range_param = "3mo"
            elif range_days <= 180:
                range_param = "6mo"
            elif range_days <= 365:
                range_param = "1y"
            elif range_days <= 730:
                range_param = "2y"
            elif range_days <= 1825:
                range_param = "5y"
            else:
                range_param = "max"
                
            # Map interval to Yahoo Finance format
            interval_map = {
                "1m": "1m",
                "5m": "5m",
                "15m": "15m",
                "1h": "60m",
                "1d": "1d"
            }
            yf_interval = interval_map.get(interval, "1d")
            
            # Call Yahoo Finance API
            data = self.api_client.call_api(
                'YahooFinance/get_stock_chart', 
                query={
                    'symbol': symbol,
                    'interval': yf_interval,
                    'range': range_param,
                    'includeAdjustedClose': True
                }
            )
            
            # Process the response
            if data and 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]
                
                # Extract timestamp and indicators
                timestamps = result.get('timestamp', [])
                indicators = result.get('indicators', {})
                
                # Extract OHLCV data
                quotes = indicators.get('quote', [{}])[0]
                opens = quotes.get('open', [])
                highs = quotes.get('high', [])
                lows = quotes.get('low', [])
                closes = quotes.get('close', [])
                volumes = quotes.get('volume', [])
                
                # Extract adjusted close if available
                adj_closes = []
                if 'adjclose' in indicators and indicators['adjclose']:
                    adj_closes = indicators['adjclose'][0].get('adjclose', [])
                
                # Create DataFrame
                data_dict = {
                    'timestamp': [datetime.datetime.fromtimestamp(ts) for ts in timestamps],
                    'open': opens,
                    'high': highs,
                    'low': lows,
                    'close': closes,
                    'volume': volumes
                }
                
                if adj_closes:
                    data_dict['adjusted_close'] = adj_closes
                
                df = pd.DataFrame(data_dict)
                
                # Remove rows with NaN values
                df = df.dropna()
                
                return df
            
            return None
        
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def store_historical_data(self, symbol, df, timeframe=DEFAULT_TIMEFRAME):
        """
        Store historical data in the database.
        
        Args:
            symbol (str): The ticker symbol.
            df (pandas.DataFrame): DataFrame with historical data.
            timeframe (str): Data timeframe.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if df is None or df.empty:
            print(f"No data to store for {symbol}")
            return False
        
        try:
            # Get or create symbol_id
            symbol_id = self.get_symbol_id(symbol)
            if symbol_id is None:
                self.add_symbols([symbol])
                symbol_id = self.get_symbol_id(symbol)
            
            cursor = self.conn.cursor()
            
            # Prepare data for insertion
            for _, row in df.iterrows():
                timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                
                # Check if adjusted_close exists in the row
                adj_close = row.get('adjusted_close', row['close'])
                
                # Insert data
                cursor.execute('''
                INSERT OR REPLACE INTO ohlcv_data 
                (symbol_id, timestamp, timeframe, open, high, low, close, volume, adjusted_close)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol_id, timestamp, timeframe, 
                    row['open'], row['high'], row['low'], row['close'], 
                    row['volume'], adj_close
                ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"Error storing historical data for {symbol}: {e}")
            self.conn.rollback()
            return False
    
    def get_historical_data(self, symbol, timeframe=DEFAULT_TIMEFRAME, start_date=None, end_date=None):
        """
        Retrieve historical data from the database.
        
        Args:
            symbol (str): The ticker symbol.
            timeframe (str): Data timeframe.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
            
        Returns:
            pandas.DataFrame: DataFrame with historical data.
        """
        try:
            symbol_id = self.get_symbol_id(symbol)
            if symbol_id is None:
                print(f"Symbol {symbol} not found in database")
                return None
            
            query = '''
            SELECT timestamp, open, high, low, close, volume, adjusted_close
            FROM ohlcv_data
            WHERE symbol_id = ? AND timeframe = ?
            '''
            
            params = [symbol_id, timeframe]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp"
            
            df = pd.read_sql_query(query, self.conn, params=params)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            print(f"Error retrieving historical data for {symbol}: {e}")
            return None
    
    def update_symbol_data(self, symbol, timeframe=DEFAULT_TIMEFRAME):
        """
        Update data for a symbol by fetching the latest data.
        
        Args:
            symbol (str): The ticker symbol.
            timeframe (str): Data timeframe.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Get the latest data timestamp
            symbol_id = self.get_symbol_id(symbol)
            if symbol_id is None:
                # Symbol not in database, fetch all historical data
                df = self.fetch_historical_data(symbol, timeframe)
                return self.store_historical_data(symbol, df, timeframe)
            
            cursor = self.conn.cursor()
            cursor.execute('''
            SELECT MAX(timestamp) FROM ohlcv_data
            WHERE symbol_id = ? AND timeframe = ?
            ''', (symbol_id, timeframe))
            
            result = cursor.fetchone()
            last_timestamp = result[0] if result and result[0] else None
            
            if last_timestamp:
                # Calculate days since last update
                last_date = datetime.datetime.strptime(last_timestamp, '%Y-%m-%d %H:%M:%S')
                days_diff = (datetime.datetime.now() - last_date).days
                
                # Fetch data since last update
                if days_diff > 0:
                    df = self.fetch_historical_data(symbol, timeframe, range_days=max(7, days_diff + 1))
                    if df is not None and not df.empty:
                        # Filter for new data only
                        df = df[df['timestamp'] > last_date]
                        if not df.empty:
                            return self.store_historical_data(symbol, df, timeframe)
                return True  # No update needed
            else:
                # No existing data, fetch all historical data
                df = self.fetch_historical_data(symbol, timeframe)
                return self.store_historical_data(symbol, df, timeframe)
                
        except Exception as e:
            print(f"Error updating data for {symbol}: {e}")
            return False
    
    def update_all_symbols(self, symbols=None, timeframe=DEFAULT_TIMEFRAME):
        """
        Update data for all symbols or a specified list.
        
        Args:
            symbols (list, optional): List of symbols to update. Defaults to None (all symbols).
            timeframe (str): Data timeframe.
            
        Returns:
            dict: Dictionary with update status for each symbol.
        """
        if symbols is None:
            # Get all active symbols from database
            cursor = self.conn.cursor()
            cursor.execute("SELECT ticker FROM symbols WHERE is_active = 1")
            symbols = [row[0] for row in cursor.fetchall()]
            
            # If no symbols in database, use default symbols
            if not symbols:
                symbols = DEFAULT_SYMBOLS
                self.add_symbols(symbols)
        
        results = {}
        for symbol in symbols:
            results[symbol] = self.update_symbol_data(symbol, timeframe)
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        return results
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()


# Example usage
if __name__ == "__main__":
    # Create data directory
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Initialize DataManager
