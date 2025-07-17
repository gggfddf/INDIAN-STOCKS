"""
Configuration file for Indian Stock Market Analysis System
"""

import os
from typing import List, Dict, Any

class Config:
    """Main configuration class for the stock market analysis system"""
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///stock_market_analysis.db')
    
    # Data Sources
    DATA_SOURCES = {
        'yahoo': 'yfinance',
        'nse': 'nsepy',
        'backup': 'manual'
    }
    
    # Top 100 F&O Stocks (NSE symbols)
    TOP_100_FO_STOCKS = [
        # Banking & Financial Services
        'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK',
        'BANKBARODA', 'PNB', 'FEDERALBNK', 'IDFCFIRSTB', 'RBLBANK', 'AUBANK',
        'BANDHANBNK', 'CANBK', 'HDFCAMC', 'BAJFINANCE', 'BAJAJFINSV', 'SBILIFE',
        'HDFCLIFE', 'ICICIPRULI', 'LICHSGFIN', 'M&MFIN', 'CHOLAFIN', 'PEL',
        
        # Information Technology
        'TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTI', 'MINDTREE', 'MPHASIS',
        'LTTS', 'COFORGE', 'PERSISTENT', 'OFSS', 'TATAELXSI', 'RAMPGREEN',
        
        # Oil & Gas
        'RELIANCE', 'ONGC', 'IOC', 'BPCL', 'HINDPETRO', 'GAIL', 'OIL', 'PETRONET',
        
        # Automobiles
        'MARUTI', 'HYUNDAI', 'M&M', 'TATAMOTORS', 'BAJAJ-AUTO', 'HEROMOTOCO',
        'TVSMOTORS', 'EICHERMOT', 'ASHOKLEY', 'TVSMOTOR', 'BAJAJHLDNG',
        
        # Pharmaceuticals
        'SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON', 'LUPIN', 'CADILAHC',
        'TORNTPHARM', 'AUROPHARMA', 'GLENMARK', 'ALKEM', 'ABBOTINDIA', 'ZYDUSLIFE',
        
        # FMCG
        'HINDUNILVR', 'ITC', 'NESTLE', 'BRITANNIA', 'DABUR', 'MARICO', 'COLPAL',
        'GODREJCP', 'UBL', 'TATACONSUM', 'VBL', 'EMAMILTD',
        
        # Metals & Mining
        'TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'COALINDIA', 'VEDL', 'NMDC',
        'SAIL', 'JINDALSTEL', 'NALCO', 'MOIL', 'WELCORP',
        
        # Cement
        'ULTRACEMCO', 'GRASIM', 'ACC', 'AMBUJACEMENT', 'SHREECEM', 'RAMCOCEM',
        'HEIDELBERG', 'JKCEMENT', 'INDIACEM',
        
        # Power & Utilities
        'NTPC', 'POWERGRID', 'ADANIGREEN', 'TATAPOWER', 'ADANIPOWER', 'NHPC',
        'SJVN', 'THERMAX',
        
        # Telecom
        'BHARTIARTL', 'IDEA', 'INDUSIND',
        
        # Consumer Durables
        'WHIRLPOOL', 'VOLTAS', 'BLUESTARCO', 'CROMPTON', 'HAVELLS', 'DIXON',
        
        # Infrastructure & Construction
        'LT', 'ADANIPORTS', 'GMR', 'IRB', 'PFC', 'RECLTD', 'IPCALAB'
    ]
    
    # Major Indices
    INDICES = {
        'NIFTY50': '^NSEI',
        'BANKNIFTY': '^NSEBANK',
        'FINNIFTY': '^CNXFIN'
    }
    
    # Technical Analysis Parameters
    TECHNICAL_PARAMS = {
        'sma_periods': [5, 10, 20, 50, 100, 200],
        'ema_periods': [9, 12, 21, 26, 50],
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bollinger_period': 20,
        'bollinger_std': 2,
        'atr_period': 14,
        'stoch_k': 14,
        'stoch_d': 3,
        'adx_period': 14,
        'cci_period': 20,
        'williams_r_period': 14,
        'obv_period': 10,
        'volume_sma': [10, 20, 50]
    }
    
    # Pattern Detection Parameters
    PATTERN_PARAMS = {
        'lookback_days': 252,  # 1 year
        'min_pattern_length': 5,
        'max_pattern_length': 50,
        'similarity_threshold': 0.85,
        'volume_spike_threshold': 2.0,  # 2x average volume
        'breakout_threshold': 0.02,  # 2% price move
        'consolidation_threshold': 0.01,  # 1% daily range
        'trend_strength_threshold': 0.7
    }
    
    # Machine Learning Parameters
    ML_PARAMS = {
        'train_test_split': 0.8,
        'validation_split': 0.2,
        'random_state': 42,
        'cv_folds': 5,
        'lstm_lookback': 60,
        'prediction_horizon': [1, 3, 5, 10],  # days
        'ensemble_models': ['lstm', 'xgboost', 'lightgbm', 'transformer'],
        'model_update_frequency': 'daily'
    }
    
    # Trading Signal Parameters
    SIGNAL_PARAMS = {
        'confidence_threshold': 0.7,
        'risk_reward_ratio': 2.0,
        'max_position_size': 0.05,  # 5% of portfolio
        'stop_loss_pct': 0.02,  # 2%
        'take_profit_pct': 0.04,  # 4%
        'correlation_threshold': 0.8,  # for pair trading
        'divergence_threshold': 0.3
    }
    
    # News & Sentiment Parameters
    NEWS_PARAMS = {
        'sources': ['moneycontrol', 'economictimes', 'livemint', 'business-standard'],
        'sentiment_models': ['vader', 'transformers'],
        'news_lookback_hours': 24,
        'sentiment_threshold': 0.6
    }
    
    # API Configuration
    API_CONFIG = {
        'rate_limit': 100,  # requests per minute
        'timeout': 30,  # seconds
        'retry_attempts': 3,
        'cache_duration': 300  # 5 minutes
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {module} | {message}',
        'rotation': '1 day',
        'retention': '30 days'
    }

    @classmethod
    def get_stock_symbols_with_suffix(cls) -> List[str]:
        """Get stock symbols with .NS suffix for Yahoo Finance"""
        return [f"{symbol}.NS" for symbol in cls.TOP_100_FO_STOCKS]
    
    @classmethod
    def get_all_symbols(cls) -> Dict[str, List[str]]:
        """Get all symbols organized by category"""
        return {
            'stocks': cls.TOP_100_FO_STOCKS,
            'indices': list(cls.INDICES.keys()),
            'yahoo_stocks': cls.get_stock_symbols_with_suffix(),
            'yahoo_indices': list(cls.INDICES.values())
        }