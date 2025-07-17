"""
Pattern Detection Module for Stock Market Analysis
Detects recurring technical patterns, volume-price anomalies, and breakout behaviors
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from loguru import logger

from ..config.config import Config
from ..utils.helpers import normalize_data, detect_outliers


class PatternDetector:
    """Detects patterns and anomalies in stock market data"""
    
    def __init__(self):
        self.config = Config()
        self.pattern_params = self.config.PATTERN_PARAMS
        self.scaler = StandardScaler()
        
    def detect_all_patterns(self, stock_data: Dict[str, pd.DataFrame], 
                           indices_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Detect all patterns across stocks and indices
        
        Args:
            stock_data: Dictionary of stock DataFrames
            indices_data: Dictionary of index DataFrames
            
        Returns:
            Dictionary containing all detected patterns
        """
        patterns = {
            'technical_patterns': {},
            'volume_anomalies': {},
            'breakout_signals': {},
            'correlation_patterns': {},
            'sector_patterns': {},
            'index_divergences': {},
            'multi_stock_patterns': {}
        }
        
        try:
            # 1. Technical Pattern Detection
            logger.info("Detecting technical patterns...")
            patterns['technical_patterns'] = self._detect_technical_patterns(stock_data)
            
            # 2. Volume-Price Anomaly Detection
            logger.info("Detecting volume-price anomalies...")
            patterns['volume_anomalies'] = self._detect_volume_anomalies(stock_data)
            
            # 3. Breakout Signal Detection
            logger.info("Detecting breakout signals...")
            patterns['breakout_signals'] = self._detect_breakout_signals(stock_data)
            
            # 4. Correlation Pattern Analysis
            logger.info("Analyzing correlation patterns...")
            patterns['correlation_patterns'] = self._analyze_correlation_patterns(stock_data)
            
            # 5. Sector-based Pattern Detection
            logger.info("Detecting sector patterns...")
            patterns['sector_patterns'] = self._detect_sector_patterns(stock_data)
            
            # 6. Index Divergence Analysis
            logger.info("Analyzing index divergences...")
            patterns['index_divergences'] = self._analyze_index_divergences(stock_data, indices_data)
            
            # 7. Multi-stock Pattern Detection
            logger.info("Detecting multi-stock patterns...")
            patterns['multi_stock_patterns'] = self._detect_multi_stock_patterns(stock_data)
            
            logger.info("Pattern detection completed successfully")
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {str(e)}")
            return patterns
    
    def _detect_technical_patterns(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict]]:
        """Detect recurring technical patterns"""
        technical_patterns = {}
        
        for symbol, df in stock_data.items():
            try:
                patterns = []
                
                if len(df) < self.pattern_params['min_pattern_length']:
                    continue
                
                # 1. Chart Patterns
                patterns.extend(self._detect_chart_patterns(df, symbol))
                
                # 2. Candlestick Patterns
                patterns.extend(self._detect_candlestick_patterns(df, symbol))
                
                # 3. Support/Resistance Patterns
                patterns.extend(self._detect_support_resistance_patterns(df, symbol))
                
                # 4. Trend Patterns
                patterns.extend(self._detect_trend_patterns(df, symbol))
                
                if patterns:
                    technical_patterns[symbol] = patterns
                    
            except Exception as e:
                logger.warning(f"Error detecting technical patterns for {symbol}: {str(e)}")
                continue
        
        return technical_patterns
    
    def _detect_chart_patterns(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Detect chart patterns like triangles, head and shoulders, etc."""
        patterns = []
        
        # Double Top/Bottom Pattern
        peaks, troughs = self._find_peaks_troughs(df['Close'])
        
        # Double Top
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                peak1_idx, peak1_val = peaks[i]
                peak2_idx, peak2_val = peaks[i + 1]
                
                # Check if peaks are similar in height (within 2%)
                if abs(peak1_val - peak2_val) / peak1_val < 0.02:
                    # Check if there's a trough between them
                    troughs_between = [t for t in troughs if peak1_idx < t[0] < peak2_idx]
                    
                    if troughs_between:
                        patterns.append({
                            'type': 'double_top',
                            'symbol': symbol,
                            'start_date': df.index[peak1_idx],
                            'end_date': df.index[peak2_idx],
                            'confidence': 0.8,
                            'target_price': min([t[1] for t in troughs_between]),
                            'pattern_data': {
                                'peak1': {'date': df.index[peak1_idx], 'price': peak1_val},
                                'peak2': {'date': df.index[peak2_idx], 'price': peak2_val},
                                'trough': {'date': df.index[troughs_between[0][0]], 'price': troughs_between[0][1]}
                            }
                        })
        
        # Triangle Patterns
        triangle_pattern = self._detect_triangle_pattern(df, symbol)
        if triangle_pattern:
            patterns.append(triangle_pattern)
        
        # Flag and Pennant Patterns
        flag_pattern = self._detect_flag_pattern(df, symbol)
        if flag_pattern:
            patterns.append(flag_pattern)
        
        return patterns
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Detect candlestick patterns"""
        patterns = []
        
        # Doji patterns
        doji_indices = df[df['Doji']].index
        for idx in doji_indices[-5:]:  # Last 5 doji patterns
            patterns.append({
                'type': 'doji',
                'symbol': symbol,
                'date': idx,
                'confidence': 0.6,
                'signal': 'reversal_potential'
            })
        
        # Hammer patterns
        hammer_indices = df[df['Hammer']].index
        for idx in hammer_indices[-3:]:  # Last 3 hammer patterns
            patterns.append({
                'type': 'hammer',
                'symbol': symbol,
                'date': idx,
                'confidence': 0.7,
                'signal': 'bullish_reversal'
            })
        
        # Engulfing patterns
        engulfing_indices = df[df['Engulfing']].index
        for idx in engulfing_indices[-3:]:  # Last 3 engulfing patterns
            patterns.append({
                'type': 'engulfing',
                'symbol': symbol,
                'date': idx,
                'confidence': 0.8,
                'signal': 'strong_reversal'
            })
        
        return patterns
    
    def _detect_support_resistance_patterns(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Detect support and resistance level patterns"""
        patterns = []
        
        # Test current price against key levels
        current_price = df['Close'].iloc[-1]
        
        # Bollinger Bands squeeze
        bb_width = df['BB_Width'].iloc[-20:]  # Last 20 days
        if bb_width.min() < bb_width.quantile(0.1):  # Very narrow bands
            patterns.append({
                'type': 'bollinger_squeeze',
                'symbol': symbol,
                'date': df.index[-1],
                'confidence': 0.8,
                'signal': 'breakout_imminent',
                'current_price': current_price,
                'bb_width': bb_width.iloc[-1]
            })
        
        # Support/Resistance bounces
        support_level = df['S1'].iloc[-1]
        resistance_level = df['R1'].iloc[-1]
        
        # Check if price is near support
        if abs(current_price - support_level) / support_level < 0.01:
            patterns.append({
                'type': 'support_test',
                'symbol': symbol,
                'date': df.index[-1],
                'confidence': 0.7,
                'signal': 'potential_bounce',
                'support_level': support_level,
                'current_price': current_price
            })
        
        # Check if price is near resistance
        if abs(current_price - resistance_level) / resistance_level < 0.01:
            patterns.append({
                'type': 'resistance_test',
                'symbol': symbol,
                'date': df.index[-1],
                'confidence': 0.7,
                'signal': 'potential_rejection',
                'resistance_level': resistance_level,
                'current_price': current_price
            })
        
        return patterns
    
    def _detect_trend_patterns(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Detect trend-based patterns"""
        patterns = []
        
        # Trend strength analysis
        trend_strength = df['Trend_Strength'].iloc[-20:].mean()
        
        if trend_strength > self.pattern_params['trend_strength_threshold']:
            patterns.append({
                'type': 'strong_trend',
                'symbol': symbol,
                'date': df.index[-1],
                'confidence': min(trend_strength, 0.95),
                'signal': 'trend_continuation',
                'trend_strength': trend_strength,
                'direction': 'bullish' if df['Close'].iloc[-1] > df['SMA_20'].iloc[-1] else 'bearish'
            })
        
        # Moving average convergence
        sma_5 = df['SMA_5'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        
        if sma_5 > sma_20 > sma_50:  # Bullish alignment
            patterns.append({
                'type': 'ma_alignment',
                'symbol': symbol,
                'date': df.index[-1],
                'confidence': 0.8,
                'signal': 'bullish_momentum',
                'direction': 'bullish'
            })
        elif sma_5 < sma_20 < sma_50:  # Bearish alignment
            patterns.append({
                'type': 'ma_alignment',
                'symbol': symbol,
                'date': df.index[-1],
                'confidence': 0.8,
                'signal': 'bearish_momentum',
                'direction': 'bearish'
            })
        
        return patterns
    
    def _detect_volume_anomalies(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict]]:
        """Detect volume-price anomalies indicating pre-breakout behavior"""
        volume_anomalies = {}
        
        for symbol, df in stock_data.items():
            try:
                anomalies = []
                
                if len(df) < 20:
                    continue
                
                # 1. Volume Spikes with Price Compression
                volume_spikes = df[df['Volume_Spike']].index
                for spike_date in volume_spikes[-5:]:  # Last 5 volume spikes
                    spike_idx = df.index.get_loc(spike_date)
                    
                    # Check price compression around volume spike
                    price_range = df['High'].iloc[spike_idx-5:spike_idx+5] - df['Low'].iloc[spike_idx-5:spike_idx+5]
                    avg_range = price_range.mean()
                    
                    if avg_range < df['ATR'].iloc[spike_idx] * 0.5:  # Compressed range
                        anomalies.append({
                            'type': 'volume_spike_compression',
                            'symbol': symbol,
                            'date': spike_date,
                            'confidence': 0.8,
                            'signal': 'breakout_setup',
                            'volume_ratio': df['Volume_Ratio'].loc[spike_date],
                            'price_compression': avg_range / df['ATR'].iloc[spike_idx]
                        })
                
                # 2. Low Volume Squeeze
                recent_volume = df['Volume_Ratio'].iloc[-10:]
                if recent_volume.max() < 0.7:  # Consistently low volume
                    recent_atr = df['ATR'].iloc[-10:]
                    if recent_atr.iloc[-1] < recent_atr.mean() * 0.8:  # Decreasing volatility
                        anomalies.append({
                            'type': 'low_volume_squeeze',
                            'symbol': symbol,
                            'date': df.index[-1],
                            'confidence': 0.7,
                            'signal': 'coiling_for_move',
                            'avg_volume_ratio': recent_volume.mean(),
                            'volatility_compression': recent_atr.iloc[-1] / recent_atr.mean()
                        })
                
                # 3. Volume Climax
                volume_ma = df['Vol_SMA_20'].iloc[-1]
                current_volume = df['Volume'].iloc[-1]
                price_change = abs(df['Close'].pct_change().iloc[-1])
                
                if current_volume > volume_ma * 3 and price_change > 0.05:  # High volume with big move
                    anomalies.append({
                        'type': 'volume_climax',
                        'symbol': symbol,
                        'date': df.index[-1],
                        'confidence': 0.9,
                        'signal': 'exhaustion_move',
                        'volume_multiple': current_volume / volume_ma,
                        'price_change': price_change
                    })
                
                # 4. Divergence between Price and Volume
                price_momentum = df['ROC_10'].iloc[-5:].mean()
                volume_trend = df['Volume_Ratio'].iloc[-5:].mean() / df['Volume_Ratio'].iloc[-10:-5].mean()
                
                if price_momentum > 0.02 and volume_trend < 0.8:  # Price up, volume down
                    anomalies.append({
                        'type': 'bearish_divergence',
                        'symbol': symbol,
                        'date': df.index[-1],
                        'confidence': 0.7,
                        'signal': 'weakness_warning',
                        'price_momentum': price_momentum,
                        'volume_trend': volume_trend
                    })
                elif price_momentum < -0.02 and volume_trend > 1.2:  # Price down, volume up
                    anomalies.append({
                        'type': 'bullish_divergence',
                        'symbol': symbol,
                        'date': df.index[-1],
                        'confidence': 0.7,
                        'signal': 'potential_bottom',
                        'price_momentum': price_momentum,
                        'volume_trend': volume_trend
                    })
                
                if anomalies:
                    volume_anomalies[symbol] = anomalies
                    
            except Exception as e:
                logger.warning(f"Error detecting volume anomalies for {symbol}: {str(e)}")
                continue
        
        return volume_anomalies
    
    def _detect_breakout_signals(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict]]:
        """Detect potential breakout signals"""
        breakout_signals = {}
        
        for symbol, df in stock_data.items():
            try:
                signals = []
                
                if len(df) < 20:
                    continue
                
                current_price = df['Close'].iloc[-1]
                
                # 1. Bollinger Band Breakouts
                bb_upper = df['BB_Upper'].iloc[-1]
                bb_lower = df['BB_Lower'].iloc[-1]
                
                if current_price > bb_upper:
                    signals.append({
                        'type': 'bollinger_breakout',
                        'symbol': symbol,
                        'date': df.index[-1],
                        'confidence': 0.8,
                        'direction': 'bullish',
                        'current_price': current_price,
                        'breakout_level': bb_upper,
                        'target_price': current_price * 1.05
                    })
                elif current_price < bb_lower:
                    signals.append({
                        'type': 'bollinger_breakout',
                        'symbol': symbol,
                        'date': df.index[-1],
                        'confidence': 0.8,
                        'direction': 'bearish',
                        'current_price': current_price,
                        'breakout_level': bb_lower,
                        'target_price': current_price * 0.95
                    })
                
                # 2. Moving Average Breakouts
                sma_20 = df['SMA_20'].iloc[-1]
                prev_close = df['Close'].iloc[-2]
                
                if prev_close <= sma_20 and current_price > sma_20:  # Golden cross
                    signals.append({
                        'type': 'ma_breakout',
                        'symbol': symbol,
                        'date': df.index[-1],
                        'confidence': 0.7,
                        'direction': 'bullish',
                        'breakout_level': sma_20,
                        'volume_confirmation': df['Volume_Ratio'].iloc[-1] > 1.5
                    })
                elif prev_close >= sma_20 and current_price < sma_20:  # Death cross
                    signals.append({
                        'type': 'ma_breakout',
                        'symbol': symbol,
                        'date': df.index[-1],
                        'confidence': 0.7,
                        'direction': 'bearish',
                        'breakout_level': sma_20,
                        'volume_confirmation': df['Volume_Ratio'].iloc[-1] > 1.5
                    })
                
                # 3. Range Breakouts
                recent_high = df['High'].iloc[-20:].max()
                recent_low = df['Low'].iloc[-20:].min()
                
                if current_price > recent_high * 1.01:  # 1% above recent high
                    signals.append({
                        'type': 'range_breakout',
                        'symbol': symbol,
                        'date': df.index[-1],
                        'confidence': 0.8,
                        'direction': 'bullish',
                        'breakout_level': recent_high,
                        'range_size': (recent_high - recent_low) / recent_low
                    })
                elif current_price < recent_low * 0.99:  # 1% below recent low
                    signals.append({
                        'type': 'range_breakout',
                        'symbol': symbol,
                        'date': df.index[-1],
                        'confidence': 0.8,
                        'direction': 'bearish',
                        'breakout_level': recent_low,
                        'range_size': (recent_high - recent_low) / recent_low
                    })
                
                if signals:
                    breakout_signals[symbol] = signals
                    
            except Exception as e:
                logger.warning(f"Error detecting breakout signals for {symbol}: {str(e)}")
                continue
        
        return breakout_signals
    
    def _analyze_correlation_patterns(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze correlation patterns and shifts"""
        correlation_patterns = {
            'correlation_matrix': {},
            'correlation_clusters': {},
            'correlation_shifts': {},
            'pair_opportunities': []
        }
        
        try:
            # Create correlation matrix
            price_data = {}
            for symbol, df in stock_data.items():
                if len(df) > 50:
                    price_data[symbol] = df['Close'].pct_change().dropna()
            
            if len(price_data) < 2:
                return correlation_patterns
            
            # Align data
            aligned_data = pd.DataFrame(price_data).dropna()
            
            # Calculate correlation matrix
            corr_matrix = aligned_data.corr()
            correlation_patterns['correlation_matrix'] = corr_matrix.to_dict()
            
            # Find correlation clusters using DBSCAN
            correlation_patterns['correlation_clusters'] = self._find_correlation_clusters(corr_matrix)
            
            # Detect correlation shifts
            correlation_patterns['correlation_shifts'] = self._detect_correlation_shifts(aligned_data)
            
            # Identify pair trading opportunities
            correlation_patterns['pair_opportunities'] = self._identify_pair_opportunities(aligned_data, corr_matrix)
            
        except Exception as e:
            logger.error(f"Error analyzing correlation patterns: {str(e)}")
        
        return correlation_patterns
    
    def _detect_sector_patterns(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Detect patterns within and across sectors"""
        # Simplified sector mapping (you would expand this)
        sector_mapping = {
            'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK'],
            'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM'],
            'Oil_Gas': ['RELIANCE', 'ONGC', 'IOC', 'BPCL'],
            'Auto': ['MARUTI', 'M&M', 'TATAMOTORS', 'BAJAJ-AUTO'],
            'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB']
        }
        
        sector_patterns = {}
        
        for sector, symbols in sector_mapping.items():
            sector_data = {sym: df for sym, df in stock_data.items() if sym in symbols}
            
            if len(sector_data) >= 2:
                sector_patterns[sector] = self._analyze_sector_movements(sector_data, sector)
        
        return sector_patterns
    
    def _analyze_index_divergences(self, stock_data: Dict[str, pd.DataFrame], 
                                  indices_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze divergences between stocks and indices"""
        divergences = {}
        
        for index_name, index_df in indices_data.items():
            if len(index_df) < 20:
                continue
                
            index_returns = index_df['Close'].pct_change().dropna()
            divergences[index_name] = []
            
            for symbol, stock_df in stock_data.items():
                if len(stock_df) < 20:
                    continue
                    
                stock_returns = stock_df['Close'].pct_change().dropna()
                
                # Align data
                aligned_data = pd.DataFrame({
                    'index': index_returns,
                    'stock': stock_returns
                }).dropna()
                
                if len(aligned_data) < 10:
                    continue
                
                # Calculate rolling correlation
                rolling_corr = aligned_data['index'].rolling(20).corr(aligned_data['stock'])
                recent_corr = rolling_corr.iloc[-5:].mean()
                
                # Detect significant divergence
                if recent_corr < 0.3:  # Low correlation indicates divergence
                    divergences[index_name].append({
                        'symbol': symbol,
                        'correlation': recent_corr,
                        'divergence_strength': 1 - recent_corr,
                        'signal': 'mean_reversion_opportunity'
                    })
        
        return divergences
    
    def _detect_multi_stock_patterns(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Detect patterns across multiple stocks"""
        multi_patterns = {
            'synchronized_moves': [],
            'leader_follower': [],
            'rotation_patterns': []
        }
        
        # Synchronized movements
        synchronized = self._find_synchronized_movements(stock_data)
        multi_patterns['synchronized_moves'] = synchronized
        
        # Leader-follower relationships
        leader_follower = self._find_leader_follower_patterns(stock_data)
        multi_patterns['leader_follower'] = leader_follower
        
        return multi_patterns
    
    def _find_peaks_troughs(self, prices: pd.Series, window: int = 5) -> Tuple[List, List]:
        """Find peaks and troughs in price series"""
        peaks = []
        troughs = []
        
        for i in range(window, len(prices) - window):
            # Check for peak
            if all(prices.iloc[i] >= prices.iloc[i-j] for j in range(1, window+1)) and \
               all(prices.iloc[i] >= prices.iloc[i+j] for j in range(1, window+1)):
                peaks.append((i, prices.iloc[i]))
            
            # Check for trough
            if all(prices.iloc[i] <= prices.iloc[i-j] for j in range(1, window+1)) and \
               all(prices.iloc[i] <= prices.iloc[i+j] for j in range(1, window+1)):
                troughs.append((i, prices.iloc[i]))
        
        return peaks, troughs
    
    def _detect_triangle_pattern(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Detect triangle patterns"""
        # Simplified triangle detection
        if len(df) < 30:
            return None
        
        recent_highs = df['High'].iloc[-30:].rolling(5).max()
        recent_lows = df['Low'].iloc[-30:].rolling(5).min()
        
        # Check if range is contracting
        early_range = recent_highs.iloc[:10].mean() - recent_lows.iloc[:10].mean()
        late_range = recent_highs.iloc[-10:].mean() - recent_lows.iloc[-10:].mean()
        
        if late_range < early_range * 0.7:  # Range contracted by 30%
            return {
                'type': 'triangle',
                'symbol': symbol,
                'start_date': df.index[-30],
                'end_date': df.index[-1],
                'confidence': 0.7,
                'signal': 'breakout_pending',
                'range_contraction': (early_range - late_range) / early_range
            }
        
        return None
    
    def _detect_flag_pattern(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Detect flag and pennant patterns"""
        # Simplified flag detection
        if len(df) < 20:
            return None
        
        # Look for strong move followed by consolidation
        strong_move = abs(df['Close'].iloc[-20] - df['Close'].iloc[-15]) / df['Close'].iloc[-20]
        
        if strong_move > 0.05:  # 5% move
            # Check for consolidation
            consolidation_range = df['High'].iloc[-10:].max() - df['Low'].iloc[-10:].min()
            avg_price = df['Close'].iloc[-10:].mean()
            
            if consolidation_range / avg_price < 0.03:  # Tight consolidation
                return {
                    'type': 'flag',
                    'symbol': symbol,
                    'pattern_start': df.index[-20],
                    'consolidation_start': df.index[-10],
                    'confidence': 0.8,
                    'signal': 'continuation_pattern',
                    'initial_move': strong_move,
                    'consolidation_tightness': consolidation_range / avg_price
                }
        
        return None
    
    def _find_correlation_clusters(self, corr_matrix: pd.DataFrame) -> Dict[str, List[str]]:
        """Find clusters of highly correlated stocks"""
        # Use DBSCAN clustering on correlation matrix
        distance_matrix = 1 - corr_matrix.abs()
        
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Not noise
                if f'cluster_{label}' not in clusters:
                    clusters[f'cluster_{label}'] = []
                clusters[f'cluster_{label}'].append(corr_matrix.index[i])
        
        return clusters
    
    def _detect_correlation_shifts(self, aligned_data: pd.DataFrame) -> List[Dict]:
        """Detect shifts in correlation patterns"""
        shifts = []
        
        # Calculate rolling correlations
        window = 30
        for col1 in aligned_data.columns:
            for col2 in aligned_data.columns:
                if col1 >= col2:
                    continue
                
                rolling_corr = aligned_data[col1].rolling(window).corr(aligned_data[col2])
                
                # Check for significant changes
                recent_corr = rolling_corr.iloc[-10:].mean()
                historical_corr = rolling_corr.iloc[-60:-30].mean()
                
                if abs(recent_corr - historical_corr) > 0.3:
                    shifts.append({
                        'pair': (col1, col2),
                        'historical_correlation': historical_corr,
                        'recent_correlation': recent_corr,
                        'shift_magnitude': abs(recent_corr - historical_corr),
                        'signal': 'correlation_breakdown' if recent_corr < historical_corr else 'correlation_strengthening'
                    })
        
        return shifts
    
    def _identify_pair_opportunities(self, aligned_data: pd.DataFrame, 
                                   corr_matrix: pd.DataFrame) -> List[Dict]:
        """Identify pair trading opportunities"""
        opportunities = []
        
        # Find highly correlated pairs
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                stock1 = corr_matrix.columns[i]
                stock2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                
                if correlation > self.config.SIGNAL_PARAMS['correlation_threshold']:
                    # Calculate spread
                    spread = aligned_data[stock1].cumsum() - aligned_data[stock2].cumsum()
                    
                    # Check for mean reversion opportunity
                    spread_zscore = (spread.iloc[-1] - spread.mean()) / spread.std()
                    
                    if abs(spread_zscore) > 2:  # Significant divergence
                        opportunities.append({
                            'pair': (stock1, stock2),
                            'correlation': correlation,
                            'spread_zscore': spread_zscore,
                            'signal': 'long_short' if spread_zscore > 2 else 'short_long',
                            'confidence': min(correlation, 0.9)
                        })
        
        return opportunities
    
    def _analyze_sector_movements(self, sector_data: Dict[str, pd.DataFrame], 
                                 sector_name: str) -> Dict[str, Any]:
        """Analyze movements within a sector"""
        if len(sector_data) < 2:
            return {}
        
        # Calculate sector average
        sector_returns = {}
        for symbol, df in sector_data.items():
            if len(df) > 20:
                sector_returns[symbol] = df['Close'].pct_change().dropna()
        
        if len(sector_returns) < 2:
            return {}
        
        aligned_returns = pd.DataFrame(sector_returns).dropna()
        sector_avg = aligned_returns.mean(axis=1)
        
        analysis = {
            'sector_momentum': sector_avg.iloc[-5:].mean(),
            'sector_volatility': sector_avg.std(),
            'leaders': [],
            'laggards': [],
            'rotation_signal': None
        }
        
        # Identify leaders and laggards
        recent_performance = {}
        for symbol in aligned_returns.columns:
            recent_performance[symbol] = aligned_returns[symbol].iloc[-5:].mean()
        
        sorted_performance = sorted(recent_performance.items(), key=lambda x: x[1], reverse=True)
        
        analysis['leaders'] = sorted_performance[:2]
        analysis['laggards'] = sorted_performance[-2:]
        
        # Detect rotation signals
        if len(sorted_performance) >= 3:
            top_perf = sorted_performance[0][1]
            bottom_perf = sorted_performance[-1][1]
            
            if top_perf - bottom_perf > 0.02:  # 2% spread
                analysis['rotation_signal'] = 'sector_divergence'
        
        return analysis
    
    def _find_synchronized_movements(self, stock_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Find stocks moving in synchronization"""
        synchronized = []
        
        # Get recent returns for all stocks
        recent_returns = {}
        for symbol, df in stock_data.items():
            if len(df) > 5:
                recent_returns[symbol] = df['Close'].pct_change().iloc[-5:].mean()
        
        if len(recent_returns) < 3:
            return synchronized
        
        # Find groups of stocks with similar movements
        threshold = 0.01  # 1% similarity threshold
        
        for symbol1, return1 in recent_returns.items():
            similar_stocks = []
            for symbol2, return2 in recent_returns.items():
                if symbol1 != symbol2 and abs(return1 - return2) < threshold:
                    similar_stocks.append(symbol2)
            
            if len(similar_stocks) >= 2:
                synchronized.append({
                    'leader': symbol1,
                    'followers': similar_stocks,
                    'movement': return1,
                    'synchronization_strength': len(similar_stocks)
                })
        
        return synchronized
    
    def _find_leader_follower_patterns(self, stock_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Find leader-follower relationships"""
        patterns = []
        
        # Simplified implementation
        symbols = list(stock_data.keys())
        
        for i, leader in enumerate(symbols[:10]):  # Check first 10 stocks as potential leaders
            if len(stock_data[leader]) < 20:
                continue
                
            leader_returns = stock_data[leader]['Close'].pct_change().dropna()
            
            followers = []
            for j, follower in enumerate(symbols):
                if i == j or len(stock_data[follower]) < 20:
                    continue
                
                follower_returns = stock_data[follower]['Close'].pct_change().dropna()
                
                # Align data
                aligned = pd.DataFrame({
                    'leader': leader_returns,
                    'follower': follower_returns
                }).dropna()
                
                if len(aligned) < 10:
                    continue
                
                # Check if follower lags leader
                leader_lagged_corr = aligned['leader'].shift(1).corr(aligned['follower'])
                
                if leader_lagged_corr > 0.6:  # Strong lagged correlation
                    followers.append({
                        'symbol': follower,
                        'lagged_correlation': leader_lagged_corr
                    })
            
            if followers:
                patterns.append({
                    'leader': leader,
                    'followers': followers,
                    'pattern_strength': len(followers)
                })
        
        return patterns