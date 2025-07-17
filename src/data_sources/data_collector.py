"""
Data Collection Module for Indian Stock Market Analysis
Handles fetching data from multiple sources with fallback mechanisms
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import concurrent.futures
from loguru import logger
import requests
from nsepy import get_history
from nsepy.symbols import get_symbol_list

from ..config.config import Config
from ..utils.helpers import setup_logging, handle_rate_limit

setup_logging()


class DataCollector:
    """Handles data collection from multiple sources for stocks and indices"""
    
    def __init__(self):
        self.config = Config()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def fetch_stock_data(self, symbol: str, period: str = "1y", 
                        interval: str = "1d", source: str = "yahoo") -> Optional[pd.DataFrame]:
        """
        Fetch stock data for a single symbol
        
        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            source: Data source ('yahoo' or 'nse')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if source == "yahoo":
                return self._fetch_yahoo_data(symbol, period, interval)
            elif source == "nse":
                return self._fetch_nse_data(symbol, period)
            else:
                logger.warning(f"Unknown data source: {source}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def _fetch_yahoo_data(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {symbol} on Yahoo Finance")
                return None
                
            # Standardize column names
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            data.index.name = 'Date'
            data['Symbol'] = symbol.replace('.NS', '')
            
            return data
            
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {str(e)}")
            return None
    
    def _fetch_nse_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch data from NSE (fallback source)"""
        try:
            # Convert period to start and end dates
            end_date = datetime.now().date()
            
            if period == "1y":
                start_date = end_date - timedelta(days=365)
            elif period == "6mo":
                start_date = end_date - timedelta(days=180)
            elif period == "3mo":
                start_date = end_date - timedelta(days=90)
            elif period == "1mo":
                start_date = end_date - timedelta(days=30)
            else:
                start_date = end_date - timedelta(days=365)  # Default to 1 year
            
            # Remove .NS suffix if present
            clean_symbol = symbol.replace('.NS', '')
            
            data = get_history(symbol=clean_symbol, start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data found for {symbol} on NSE")
                return None
            
            # Standardize column names
            data = data.rename(columns={
                'Symbol': 'Symbol',
                'Open': 'Open',
                'High': 'High', 
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            data['Symbol'] = clean_symbol
            return data
            
        except Exception as e:
            logger.error(f"NSE error for {symbol}: {str(e)}")
            return None
    
    def fetch_multiple_stocks(self, symbols: List[str], period: str = "1y", 
                             max_workers: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks concurrently
        
        Args:
            symbols: List of stock symbols
            period: Time period
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Dictionary mapping symbols to their data DataFrames
        """
        results = {}
        
        def fetch_single(symbol):
            data = self.fetch_stock_data(symbol, period)
            if data is not None:
                return symbol, data
            return symbol, None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(fetch_single, symbol): symbol 
                              for symbol in symbols}
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol, data = future.result()
                if data is not None:
                    results[symbol] = data
                    logger.info(f"Successfully fetched data for {symbol}")
                else:
                    logger.warning(f"Failed to fetch data for {symbol}")
                
                # Rate limiting
                time.sleep(0.1)
        
        return results
    
    def fetch_index_data(self, index_symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Fetch index data
        
        Args:
            index_symbol: Index symbol (e.g., '^NSEI' for Nifty 50)
            period: Time period
            
        Returns:
            DataFrame with index OHLCV data
        """
        try:
            ticker = yf.Ticker(index_symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No data found for index {index_symbol}")
                return None
            
            # Standardize column names
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            data.index.name = 'Date'
            data['Symbol'] = index_symbol
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching index data for {index_symbol}: {str(e)}")
            return None
    
    def fetch_all_indices(self, period: str = "1y") -> Dict[str, pd.DataFrame]:
        """Fetch data for all major indices"""
        indices_data = {}
        
        for index_name, index_symbol in self.config.INDICES.items():
            data = self.fetch_index_data(index_symbol, period)
            if data is not None:
                indices_data[index_name] = data
                logger.info(f"Successfully fetched data for {index_name}")
            else:
                logger.warning(f"Failed to fetch data for {index_name}")
            
            time.sleep(0.1)  # Rate limiting
        
        return indices_data
    
    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get latest prices for multiple symbols
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dictionary mapping symbols to their latest prices
        """
        prices = {}
        
        try:
            # Batch fetch for efficiency
            tickers = yf.Tickers(' '.join(symbols))
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.fast_info
                    prices[symbol] = info.last_price
                except:
                    # Fallback to individual fetch
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        prices[symbol] = hist['Close'].iloc[-1]
                    
        except Exception as e:
            logger.error(f"Error fetching latest prices: {str(e)}")
        
        return prices
    
    def get_market_status(self) -> Dict[str, any]:
        """Get current market status"""
        try:
            # Use a representative stock to check market status
            ticker = yf.Ticker("RELIANCE.NS")
            hist = ticker.history(period="2d")
            
            if len(hist) >= 2:
                last_trading_day = hist.index[-1].date()
                today = datetime.now().date()
                
                market_open = last_trading_day == today
            else:
                market_open = False
            
            return {
                'market_open': market_open,
                'last_trading_day': last_trading_day if len(hist) > 0 else None,
                'current_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            return {
                'market_open': False,
                'last_trading_day': None,
                'current_time': datetime.now()
            }
    
    def validate_data_quality(self, data: pd.DataFrame, symbol: str) -> Tuple[bool, List[str]]:
        """
        Validate data quality and identify issues
        
        Args:
            data: DataFrame to validate
            symbol: Symbol name for logging
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if data.empty:
            issues.append("Empty dataset")
            return False, issues
        
        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for null values
        null_counts = data[required_cols].isnull().sum()
        if null_counts.any():
            issues.append(f"Null values found: {null_counts.to_dict()}")
        
        # Check for unrealistic values
        if (data['High'] < data['Low']).any():
            issues.append("High prices lower than Low prices")
        
        if (data['Open'] > data['High']).any() or (data['Open'] < data['Low']).any():
            issues.append("Open prices outside High-Low range")
        
        if (data['Close'] > data['High']).any() or (data['Close'] < data['Low']).any():
            issues.append("Close prices outside High-Low range")
        
        if (data['Volume'] < 0).any():
            issues.append("Negative volume values")
        
        # Check for extreme price movements (>50% in a day)
        daily_returns = data['Close'].pct_change().abs()
        extreme_moves = daily_returns > 0.5
        if extreme_moves.any():
            issues.append(f"Extreme price movements detected: {extreme_moves.sum()} instances")
        
        is_valid = len(issues) == 0
        
        if issues:
            logger.warning(f"Data quality issues for {symbol}: {'; '.join(issues)}")
        
        return is_valid, issues