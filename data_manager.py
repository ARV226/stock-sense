import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta
import pytz
import json

class DataManager:
    def __init__(self):
        # Create directory for storing historical data
        os.makedirs('stock_data', exist_ok=True)
        
        # Initialize data storage
        self.stock_data = {}
        
        # Load any existing data
        self._load_data()
    
    def _load_data(self):
        """Load saved stock data from disk"""
        for filename in os.listdir('stock_data'):
            if filename.endswith('.csv'):
                symbol = filename.split('.')[0]
                file_path = os.path.join('stock_data', filename)
                try:
                    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    self.stock_data[symbol] = data
                except Exception as e:
                    print(f"Error loading data for {symbol}: {e}")
    
    def _save_data(self, symbol):
        """Save stock data to disk"""
        if symbol in self.stock_data:
            file_path = os.path.join('stock_data', f"{symbol}.csv")
            try:
                self.stock_data[symbol].to_csv(file_path)
            except Exception as e:
                print(f"Error saving data for {symbol}: {e}")
    
    def get_stock_data(self, symbol):
        """
        Get historical stock data for a symbol
        
        Args:
            symbol: The stock symbol
            
        Returns:
            pandas.DataFrame: Historical stock data
        """
        if symbol in self.stock_data:
            # Check if data needs to be updated (if last entry is older than a day)
            last_date = self.stock_data[symbol].index[-1]
            
            # Make sure last_date is timezone-aware (in UTC)
            if not hasattr(last_date, 'tzinfo') or last_date.tzinfo is None:
                last_date = pd.Timestamp(last_date).tz_localize(pytz.UTC)
            elif last_date.tzinfo != pytz.UTC:
                last_date = last_date.astimezone(pytz.UTC)
                
            # Get current time in UTC
            now = datetime.now(pytz.UTC)
            
            # Compare timezone-aware datetimes
            if now - last_date > timedelta(days=1):
                # Data needs updating
                self._update_stock_data(symbol)
                
            return self.stock_data[symbol]
        else:
            # Data not found, fetch it
            return self._fetch_stock_data(symbol)
    
    def _fetch_stock_data(self, symbol):
        """
        Fetch historical stock data for a symbol
        
        Args:
            symbol: The stock symbol
            
        Returns:
            pandas.DataFrame: Historical stock data
        """
        try:
            # Get data for past year (using timezone-aware UTC dates)
            end_date = datetime.now(pytz.UTC)
            start_date = end_date - timedelta(days=365)
            
            # Fetch data using yfinance
            data = yf.download(symbol, start=start_date, end=end_date)
            
            if isinstance(data, pd.DataFrame) and not data.empty:
                # Ensure data index has UTC timezone
                # First check if it's already timezone-aware
                if hasattr(data.index[0], 'tzinfo') and data.index[0].tzinfo is not None:
                    # If it's already timezone-aware but not UTC, convert to UTC
                    if data.index[0].tzinfo != pytz.UTC:
                        data.index = data.index.tz_convert(pytz.UTC)
                else:
                    # If it's timezone-naive, localize to UTC
                    data.index = data.index.tz_localize(pytz.UTC)
                
                # Store data
                self.stock_data[symbol] = data
                self._save_data(symbol)
                return data
            else:
                print(f"No data found for symbol: {symbol}")
                return None
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _update_stock_data(self, symbol):
        """
        Update existing stock data with latest data
        
        Args:
            symbol: The stock symbol
        """
        try:
            # Get latest data
            last_date = self.stock_data[symbol].index[-1]
            
            # Make sure last_date is timezone-aware (in UTC)
            if not hasattr(last_date, 'tzinfo') or last_date.tzinfo is None:
                last_date = pd.Timestamp(last_date).tz_localize(pytz.UTC)
            elif last_date.tzinfo != pytz.UTC:
                last_date = last_date.astimezone(pytz.UTC)
            
            # Get current time in UTC
            now = datetime.now(pytz.UTC)    
            start_date = last_date + timedelta(days=1)
            
            # Only fetch if we need new data
            if start_date < now:
                # Fetch new data
                new_data = yf.download(symbol, start=start_date, end=now)
                
                if isinstance(new_data, pd.DataFrame) and not new_data.empty:
                    # Ensure new data has UTC timezone
                    if hasattr(new_data.index[0], 'tzinfo') and new_data.index[0].tzinfo is not None:
                        if new_data.index[0].tzinfo != pytz.UTC:
                            new_data.index = new_data.index.tz_convert(pytz.UTC)
                    else:
                        new_data.index = new_data.index.tz_localize(pytz.UTC)
                    
                    # Ensure existing data has UTC timezone
                    existing_data = self.stock_data[symbol].copy()
                    if not hasattr(existing_data.index[0], 'tzinfo') or existing_data.index[0].tzinfo is None:
                        existing_data.index = existing_data.index.tz_localize(pytz.UTC)
                    elif existing_data.index[0].tzinfo != pytz.UTC:
                        existing_data.index = existing_data.index.tz_convert(pytz.UTC)
                            
                    # Combine with existing data
                    self.stock_data[symbol] = pd.concat([existing_data, new_data])
                    self._save_data(symbol)
        except Exception as e:
            print(f"Error updating data for {symbol}: {e}")
    
    def update_stock_data(self, symbol, new_data):
        """
        Update stock data with new data (typically called after getting historical data)
        
        Args:
            symbol: The stock symbol
            new_data: pandas.DataFrame with new data
        """
        if not isinstance(new_data, pd.DataFrame) or new_data.empty:
            print(f"Cannot update stock data for {symbol}: Invalid data")
            return
            
        # Ensure data has UTC timezone
        updated_data = new_data.copy()
        
        # Ensure new data has UTC timezone
        if hasattr(updated_data.index[0], 'tzinfo') and updated_data.index[0].tzinfo is not None:
            if updated_data.index[0].tzinfo != pytz.UTC:
                updated_data.index = updated_data.index.tz_convert(pytz.UTC)
        else:
            updated_data.index = updated_data.index.tz_localize(pytz.UTC)
            
        if symbol in self.stock_data:
            # Ensure existing data has UTC timezone
            existing_data = self.stock_data[symbol].copy()
            if hasattr(existing_data.index[0], 'tzinfo') and existing_data.index[0].tzinfo is not None:
                if existing_data.index[0].tzinfo != pytz.UTC:
                    existing_data.index = existing_data.index.tz_convert(pytz.UTC)
            else:
                existing_data.index = existing_data.index.tz_localize(pytz.UTC)
                
            # Check if new data is different from stored data
            # We need to reset the timezone info for comparison only (otherwise equals may fail due to timezone object differences)
            existing_compare = existing_data.copy()
            existing_compare.index = existing_compare.index.map(lambda dt: dt.replace(tzinfo=None))
            
            updated_compare = updated_data.copy()
            updated_compare.index = updated_compare.index.map(lambda dt: dt.replace(tzinfo=None))
            
            if not existing_compare.equals(updated_compare):
                self.stock_data[symbol] = updated_data
                self._save_data(symbol)
        else:
            # New data
            self.stock_data[symbol] = updated_data
            self._save_data(symbol)
