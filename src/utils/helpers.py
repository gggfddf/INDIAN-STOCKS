"""
Utility helper functions for the stock market analysis system
"""

import time
import functools
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable
import os


def setup_logging():
    """Setup logging configuration"""
    logger.remove()  # Remove default handler
    
    # Add file handler
    logger.add(
        "logs/stock_analysis_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module} | {message}",
        rotation="1 day",
        retention="30 days",
        level="INFO"
    )
    
    # Add console handler for development
    logger.add(
        lambda msg: print(msg),
        format="{time:HH:mm:ss} | {level} | {message}",
        level="INFO"
    )
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)


def handle_rate_limit(calls_per_minute: int = 100):
    """
    Decorator to handle rate limiting for API calls
    
    Args:
        calls_per_minute: Maximum number of calls per minute
    """
    min_interval = 60.0 / calls_per_minute
    last_call_time = [0.0]
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            time_since_last_call = current_time - last_call_time[0]
            
            if time_since_last_call < min_interval:
                sleep_time = min_interval - time_since_last_call
                time.sleep(sleep_time)
            
            last_call_time[0] = time.time()
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def calculate_percentage_change(current: float, previous: float) -> float:
    """Calculate percentage change between two values"""
    if previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100


def normalize_data(data: pd.Series, method: str = 'minmax') -> pd.Series:
    """
    Normalize data using different methods
    
    Args:
        data: Data to normalize
        method: Normalization method ('minmax', 'zscore', 'robust')
    
    Returns:
        Normalized data
    """
    if method == 'minmax':
        return (data - data.min()) / (data.max() - data.min())
    elif method == 'zscore':
        return (data - data.mean()) / data.std()
    elif method == 'robust':
        median = data.median()
        mad = np.median(np.abs(data - median))
        return (data - median) / mad
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in data
    
    Args:
        data: Data to analyze
        method: Detection method ('iqr', 'zscore', 'modified_zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    elif method == 'modified_zscore':
        median = data.median()
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def calculate_correlation_matrix(data: Dict[str, pd.DataFrame], 
                               column: str = 'Close') -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple stocks
    
    Args:
        data: Dictionary of stock data
        column: Column to use for correlation calculation
    
    Returns:
        Correlation matrix
    """
    # Align all data by date and extract the specified column
    aligned_data = {}
    
    for symbol, df in data.items():
        if column in df.columns:
            aligned_data[symbol] = df[column]
    
    if not aligned_data:
        return pd.DataFrame()
    
    # Create DataFrame with all stocks
    combined_df = pd.DataFrame(aligned_data)
    
    # Calculate correlation matrix
    return combined_df.corr()


def identify_trading_sessions():
    """
    Identify Indian stock market trading sessions
    
    Returns:
        Dictionary with session timings
    """
    return {
        'pre_market': {'start': '09:00', 'end': '09:15'},
        'normal': {'start': '09:15', 'end': '15:30'},
        'post_market': {'start': '15:40', 'end': '16:00'},
        'timezone': 'Asia/Kolkata'
    }


def is_market_open(current_time: datetime = None) -> bool:
    """
    Check if market is currently open
    
    Args:
        current_time: Time to check (defaults to current time)
    
    Returns:
        Boolean indicating if market is open
    """
    if current_time is None:
        current_time = datetime.now()
    
    # Check if it's a weekday (Monday = 0, Sunday = 6)
    if current_time.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check time (simplified - doesn't account for holidays)
    market_start = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_start <= current_time <= market_end


def format_currency(amount: float, currency: str = 'INR') -> str:
    """
    Format currency amounts
    
    Args:
        amount: Amount to format
        currency: Currency code
    
    Returns:
        Formatted currency string
    """
    if currency == 'INR':
        if amount >= 10000000:  # 1 crore
            return f"₹{amount/10000000:.2f} Cr"
        elif amount >= 100000:  # 1 lakh
            return f"₹{amount/100000:.2f} L"
        elif amount >= 1000:  # 1 thousand
            return f"₹{amount/1000:.2f} K"
        else:
            return f"₹{amount:.2f}"
    else:
        return f"{currency} {amount:,.2f}"


def calculate_moving_average(data: pd.Series, window: int, method: str = 'simple') -> pd.Series:
    """
    Calculate moving averages
    
    Args:
        data: Data series
        window: Window size
        method: MA method ('simple', 'exponential', 'weighted')
    
    Returns:
        Moving average series
    """
    if method == 'simple':
        return data.rolling(window=window).mean()
    elif method == 'exponential':
        return data.ewm(span=window).mean()
    elif method == 'weighted':
        weights = np.arange(1, window + 1)
        return data.rolling(window).apply(lambda x: np.average(x, weights=weights))
    else:
        raise ValueError(f"Unknown moving average method: {method}")


def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling volatility
    
    Args:
        returns: Return series
        window: Rolling window size
    
    Returns:
        Volatility series
    """
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized


def detect_support_resistance(prices: pd.Series, window: int = 20, 
                             threshold: float = 0.02) -> Dict[str, List[float]]:
    """
    Detect support and resistance levels
    
    Args:
        prices: Price series
        window: Window for local minima/maxima detection
        threshold: Minimum price difference for level significance
    
    Returns:
        Dictionary with support and resistance levels
    """
    # Find local minima (support) and maxima (resistance)
    local_min = prices[(prices.shift(1) > prices) & (prices.shift(-1) > prices)]
    local_max = prices[(prices.shift(1) < prices) & (prices.shift(-1) < prices)]
    
    # Group similar levels
    support_levels = []
    resistance_levels = []
    
    for level in local_min:
        if not any(abs(level - existing) / existing < threshold for existing in support_levels):
            support_levels.append(level)
    
    for level in local_max:
        if not any(abs(level - existing) / existing < threshold for existing in resistance_levels):
            resistance_levels.append(level)
    
    return {
        'support': sorted(support_levels),
        'resistance': sorted(resistance_levels, reverse=True)
    }


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annual)
    
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate if DataFrame has required structure
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        Boolean indicating if DataFrame is valid
    """
    if df.empty:
        return False
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.warning(f"Missing required columns: {missing_columns}")
        return False
    
    return True


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    return numerator / denominator if denominator != 0 else default


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]