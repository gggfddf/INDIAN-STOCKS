"""
Technical Indicators Module for Stock Market Analysis
Calculates comprehensive technical analysis indicators
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Tuple
from loguru import logger

from ..config.config import Config


class TechnicalIndicators:
    """Calculate technical indicators for stock market analysis"""
    
    def __init__(self):
        self.config = Config()
        self.tech_params = self.config.TECHNICAL_PARAMS
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a given DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicators added
        """
        if df.empty or len(df) < 50:
            logger.warning("Insufficient data for technical indicator calculation")
            return df
        
        try:
            # Make a copy to avoid modifying original
            result_df = df.copy()
            
            # Price-based indicators
            result_df = self._add_price_indicators(result_df)
            
            # Volume-based indicators
            result_df = self._add_volume_indicators(result_df)
            
            # Volatility indicators
            result_df = self._add_volatility_indicators(result_df)
            
            # Momentum indicators
            result_df = self._add_momentum_indicators(result_df)
            
            # Trend indicators
            result_df = self._add_trend_indicators(result_df)
            
            # Support/Resistance levels
            result_df = self._add_support_resistance(result_df)
            
            # Pattern recognition features
            result_df = self._add_pattern_features(result_df)
            
            # Market microstructure indicators
            result_df = self._add_microstructure_indicators(result_df)
            
            logger.info(f"Successfully calculated {len(result_df.columns) - len(df.columns)} technical indicators")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return df
    
    def _add_price_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based indicators"""
        
        # Simple Moving Averages
        for period in self.tech_params['sma_periods']:
            df[f'SMA_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
        
        # Exponential Moving Averages
        for period in self.tech_params['ema_periods']:
            df[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
        
        # Weighted Moving Average
        df['WMA_20'] = df['Close'].rolling(window=20).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x) + 1))
        )
        
        # Hull Moving Average
        df['HMA_20'] = self._calculate_hull_ma(df['Close'], 20)
        
        # Price relative to moving averages
        df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['Price_vs_EMA20'] = (df['Close'] - df['EMA_20']) / df['EMA_20']
        
        # Moving average convergence/divergence
        df['SMA_5_20_diff'] = df['SMA_5'] - df['SMA_20']
        df['EMA_12_26_diff'] = df['EMA_12'] - df['EMA_26']
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        
        # Volume moving averages
        for period in self.tech_params['volume_sma']:
            df[f'Vol_SMA_{period}'] = df['Volume'].rolling(window=period).mean()
        
        # Volume ratio
        df['Volume_Ratio'] = df['Volume'] / df['Vol_SMA_20']
        
        # On-Balance Volume
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['OBV_SMA'] = df['OBV'].rolling(window=self.tech_params['obv_period']).mean()
        
        # Volume-Price Trend
        df['VPT'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])
        
        # Accumulation/Distribution Line
        df['ADL'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Chaikin Money Flow
        df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Money Flow Index
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Volume-weighted average price
        df['VWAP'] = self._calculate_vwap(df)
        df['Price_vs_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']
        
        # Volume spikes
        df['Volume_Spike'] = df['Volume_Ratio'] > self.config.PATTERN_PARAMS['volume_spike_threshold']
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        
        # Average True Range
        df['ATR'] = ta.volatility.average_true_range(
            df['High'], df['Low'], df['Close'], 
            window=self.tech_params['atr_period']
        )
        
        # Bollinger Bands
        bb_period = self.tech_params['bollinger_period']
        bb_std = self.tech_params['bollinger_std']
        
        bb = ta.volatility.BollingerBands(df['Close'], window=bb_period, window_dev=bb_std)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Keltner Channels
        kc = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
        df['KC_Upper'] = kc.keltner_channel_hband()
        df['KC_Middle'] = kc.keltner_channel_mband()
        df['KC_Lower'] = kc.keltner_channel_lband()
        
        # Historical Volatility
        returns = df['Close'].pct_change()
        df['HV_20'] = returns.rolling(window=20).std() * np.sqrt(252)
        df['HV_50'] = returns.rolling(window=50).std() * np.sqrt(252)
        
        # Volatility ratio
        df['Vol_Ratio'] = df['HV_20'] / df['HV_50']
        
        # True Range
        df['TR'] = ta.volatility.true_range(df['High'], df['Low'], df['Close'])
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=self.tech_params['rsi_period'])
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            df['High'], df['Low'], df['Close'],
            window=self.tech_params['stoch_k'],
            smooth_window=self.tech_params['stoch_d']
        )
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # MACD
        macd = ta.trend.MACD(
            df['Close'],
            window_fast=self.tech_params['macd_fast'],
            window_slow=self.tech_params['macd_slow'],
            window_sign=self.tech_params['macd_signal']
        )
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # Williams %R
        df['Williams_R'] = ta.momentum.williams_r(
            df['High'], df['Low'], df['Close'],
            lbp=self.tech_params['williams_r_period']
        )
        
        # Rate of Change
        df['ROC_10'] = ta.momentum.roc(df['Close'], window=10)
        df['ROC_20'] = ta.momentum.roc(df['Close'], window=20)
        
        # Commodity Channel Index
        df['CCI'] = ta.trend.cci(
            df['High'], df['Low'], df['Close'],
            window=self.tech_params['cci_period']
        )
        
        # Price momentum
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators"""
        
        # Average Directional Index
        adx = ta.trend.ADXIndicator(
            df['High'], df['Low'], df['Close'],
            window=self.tech_params['adx_period']
        )
        df['ADX'] = adx.adx()
        df['ADX_Pos'] = adx.adx_pos()
        df['ADX_Neg'] = adx.adx_neg()
        
        # Parabolic SAR
        df['PSAR'] = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close']).psar()
        
        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
        df['Ichimoku_A'] = ichimoku.ichimoku_a()
        df['Ichimoku_B'] = ichimoku.ichimoku_b()
        df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        df['Ichimoku_Conv'] = ichimoku.ichimoku_conversion_line()
        
        # Aroon
        aroon = ta.trend.AroonIndicator(df['High'], df['Low'])
        df['Aroon_Up'] = aroon.aroon_up()
        df['Aroon_Down'] = aroon.aroon_down()
        df['Aroon_Oscillator'] = df['Aroon_Up'] - df['Aroon_Down']
        
        # Trend strength
        df['Trend_Strength'] = self._calculate_trend_strength(df)
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add support and resistance levels"""
        
        # Pivot Points
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low']
        df['S1'] = 2 * df['Pivot'] - df['High']
        df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
        df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
        
        # Distance from key levels
        df['Dist_from_Pivot'] = (df['Close'] - df['Pivot']) / df['Pivot']
        df['Dist_from_R1'] = (df['Close'] - df['R1']) / df['R1']
        df['Dist_from_S1'] = (df['Close'] - df['S1']) / df['S1']
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features"""
        
        # Candlestick patterns (simplified)
        df['Doji'] = self._is_doji(df)
        df['Hammer'] = self._is_hammer(df)
        df['Shooting_Star'] = self._is_shooting_star(df)
        df['Engulfing'] = self._is_engulfing(df)
        
        # Price patterns
        df['Higher_High'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
        df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
        df['Inside_Bar'] = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
        df['Outside_Bar'] = (df['High'] > df['High'].shift(1)) & (df['Low'] < df['Low'].shift(1))
        
        # Gap analysis
        df['Gap_Up'] = df['Low'] > df['High'].shift(1)
        df['Gap_Down'] = df['High'] < df['Low'].shift(1)
        df['Gap_Size'] = np.where(
            df['Gap_Up'], 
            (df['Low'] - df['High'].shift(1)) / df['Close'].shift(1),
            np.where(
                df['Gap_Down'],
                (df['High'] - df['Low'].shift(1)) / df['Close'].shift(1),
                0
            )
        )
        
        return df
    
    def _add_microstructure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure indicators"""
        
        # Bid-Ask spread proxy (using high-low)
        df['Spread_Proxy'] = (df['High'] - df['Low']) / df['Close']
        
        # Intraday returns
        df['Intraday_Return'] = (df['Close'] - df['Open']) / df['Open']
        df['Overnight_Return'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Price efficiency
        df['Price_Efficiency'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'])
        
        # Volume distribution
        df['Volume_at_Close'] = df['Volume'] * (1 - abs(df['Close'] - (df['High'] + df['Low'])/2) / (df['High'] - df['Low']))
        
        return df
    
    def _calculate_hull_ma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Hull Moving Average"""
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        wma_half = series.rolling(window=half_period).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x) + 1))
        )
        wma_full = series.rolling(window=period).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x) + 1))
        )
        
        raw_hma = 2 * wma_half - wma_full
        hma = raw_hma.rolling(window=sqrt_period).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x) + 1))
        )
        
        return hma
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength based on multiple indicators"""
        # Simple trend strength based on ADX and price vs moving averages
        trend_score = 0
        
        # ADX component
        adx_score = np.where(df['ADX'] > 25, 1, 0)
        
        # Price vs MA component
        ma_score = np.where(df['Close'] > df['SMA_20'], 1, -1)
        
        # Combine scores
        trend_strength = (adx_score + ma_score) / 2
        
        return pd.Series(trend_strength, index=df.index)
    
    def _is_doji(self, df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        """Detect Doji candlestick pattern"""
        body_size = abs(df['Close'] - df['Open']) / df['Close']
        return body_size < threshold / 100
    
    def _is_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Detect Hammer candlestick pattern"""
        body = abs(df['Close'] - df['Open'])
        lower_shadow = df['Open'].combine(df['Close'], min) - df['Low']
        upper_shadow = df['High'] - df['Open'].combine(df['Close'], max)
        
        return (lower_shadow > 2 * body) & (upper_shadow < body)
    
    def _is_shooting_star(self, df: pd.DataFrame) -> pd.Series:
        """Detect Shooting Star candlestick pattern"""
        body = abs(df['Close'] - df['Open'])
        lower_shadow = df['Open'].combine(df['Close'], min) - df['Low']
        upper_shadow = df['High'] - df['Open'].combine(df['Close'], max)
        
        return (upper_shadow > 2 * body) & (lower_shadow < body)
    
    def _is_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """Detect Engulfing candlestick pattern"""
        prev_body = abs(df['Close'].shift(1) - df['Open'].shift(1))
        curr_body = abs(df['Close'] - df['Open'])
        
        bullish_engulfing = (
            (df['Close'].shift(1) < df['Open'].shift(1)) &  # Previous red
            (df['Close'] > df['Open']) &  # Current green
            (df['Open'] < df['Close'].shift(1)) &  # Opens below prev close
            (df['Close'] > df['Open'].shift(1)) &  # Closes above prev open
            (curr_body > prev_body)  # Larger body
        )
        
        bearish_engulfing = (
            (df['Close'].shift(1) > df['Open'].shift(1)) &  # Previous green
            (df['Close'] < df['Open']) &  # Current red
            (df['Open'] > df['Close'].shift(1)) &  # Opens above prev close
            (df['Close'] < df['Open'].shift(1)) &  # Closes below prev open
            (curr_body > prev_body)  # Larger body
        )
        
        return bullish_engulfing | bearish_engulfing