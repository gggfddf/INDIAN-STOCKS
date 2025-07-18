"""
🔮 MARKET SECRET REVEALER 🔮
DEEP LEARNING SYSTEM TO UNCOVER HIDDEN MARKET PATTERNS & ANOMALIES
A SECRET-REVEALING BOOK FOR INDIAN STOCK MARKET (100+ F&O STOCKS)

Features:
- Pattern Detection: Chart patterns, candlestick formations, fractals
- Anomaly Detection: Volume spikes, price anomalies, unusual behavior
- Correlation Analysis: Hidden relationships, pair trading opportunities
- Deep Learning: Neural networks, clustering, dimensionality reduction
- Market Secrets: Sector rotations, lead-lag relationships, hidden signals
"""

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neural_network import MLPRegressor, MLPClassifier
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks, argrelextrema
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import itertools
import os

class MarketSecretRevealer:
    """🔮 Deep Learning System to Reveal Market Secrets 🔮"""
    
    def __init__(self):
        # ALL 100+ F&O STOCKS (Complete List)
        self.all_fo_stocks = [
            # Banking & Financial Services (25 stocks)
            'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK',
            'BANKBARODA', 'PNB', 'FEDERALBNK', 'IDFCFIRSTB', 'RBLBANK', 'AUBANK',
            'BANDHANBNK', 'CANBK', 'BAJFINANCE', 'BAJAJFINSV', 'SBILIFE',
            'HDFCLIFE', 'ICICIPRULI', 'LICHSGFIN', 'MFIN', 'CHOLAFIN', 'PEL',
            'HDFCAMC', 'CDSL',
            
            # IT & Technology (15 stocks)  
            'TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTI', 'MINDTREE', 'MPHASIS',
            'LTTS', 'COFORGE', 'PERSISTENT', 'OFSS', 'TATAELXSI', 'INTELLECT', 'RAMPGREEN',
            
            # Oil & Gas (10 stocks)
            'RELIANCE', 'ONGC', 'IOC', 'BPCL', 'HINDPETRO', 'GAIL', 'OIL', 'PETRONET',
            'MGL', 'IGL',
            
            # Automobiles (10 stocks)
            'MARUTI', 'M&M', 'TATAMOTORS', 'BAJAJ-AUTO', 'HEROMOTOCO', 'TVSMOTORS', 
            'EICHERMOT', 'ASHOKLEY', 'BAJAJHLDNG', 'ESCORTS',
            
            # Pharmaceuticals (12 stocks)
            'SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON', 'LUPIN', 'CADILAHC',
            'TORNTPHARM', 'AUROPHARMA', 'GLENMARK', 'ALKEM', 'ZYDUSLIFE',
            
            # FMCG (12 stocks)
            'HINDUNILVR', 'ITC', 'NESTLE', 'BRITANNIA', 'DABUR', 'MARICO', 'COLPAL',
            'GODREJCP', 'UBL', 'TATACONSUM', 'VBL', 'EMAMILTD',
            
            # Metals & Mining (12 stocks)
            'TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'COALINDIA', 'VEDL', 'NMDC',
            'SAIL', 'JINDALSTEL', 'NALCO', 'MOIL', 'WELCORP', 'APL',
            
            # Cement (8 stocks)
            'ULTRACEMCO', 'GRASIM', 'ACC', 'AMBUJACEMENT', 'SHREECEM', 'RAMCOCEM',
            'JKCEMENT', 'INDIACEM',
            
            # Power & Energy (10 stocks)
            'NTPC', 'POWERGRID', 'ADANIGREEN', 'TATAPOWER', 'ADANIPOWER', 'NHPC',
            'SJVN', 'THERMAX', 'GMRINFRA', 'TORNTPOWER',
            
            # Telecom (3 stocks)
            'BHARTIARTL', 'IDEA', 'INDIGO',
            
            # Infrastructure & Others (15 stocks)
            'LT', 'ADANIPORTS', 'GMR', 'IRB', 'PFC', 'RECLTD', 'SIEMENS',
            'ABB', 'HAVELLS', 'VOLTAS', 'CUMMINSIND', 'BHEL', 'BEL', 'SAIL', 'CONCOR'
        ]
        
        # Major Indices
        self.indices = {
            'NIFTY50': '^NSEI',
            'BANKNIFTY': '^NSEBANK', 
            'FINNIFTY': '^CNXFIN'
        }
        
        # Sector Mapping
        self.sector_mapping = {
            'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK', 'BANKBARODA', 'PNB'],
            'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTI', 'MINDTREE'],
            'Energy': ['RELIANCE', 'ONGC', 'IOC', 'BPCL', 'HINDPETRO', 'GAIL'],
            'Auto': ['MARUTI', 'M&M', 'TATAMOTORS', 'BAJAJ-AUTO', 'HEROMOTOCO'],
            'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON'],
            'FMCG': ['HINDUNILVR', 'ITC', 'NESTLE', 'BRITANNIA', 'DABUR'],
            'Metals': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'COALINDIA', 'VEDL'],
            'Cement': ['ULTRACEMCO', 'GRASIM', 'ACC', 'AMBUJACEMENT']
        }
        
        # Initialize storage
        self.market_data = {}
        self.feature_matrix = None
        self.secret_patterns = {}
        self.hidden_correlations = {}
        self.market_anomalies = {}
        self.pair_opportunities = {}
        self.sector_secrets = {}
        self.deep_insights = {}
        
        print("🔮 MARKET SECRET REVEALER INITIALIZED 🔮")
        print(f"🎯 Target: {len(self.all_fo_stocks)} F&O Stocks + {len(self.indices)} Indices")
        print("🧠 Deep Learning: Neural Networks, Clustering, Anomaly Detection")
        print("🔍 Mission: Uncover Hidden Market Patterns & Secrets")
        print("📊 Analysis: Patterns, Correlations, Anomalies, Fractals")
    
    def collect_comprehensive_data(self, period="1y", interval="1d"):
        """Collect comprehensive market data for all stocks and indices"""
        print(f"\n🚀 COLLECTING DATA FOR ALL {len(self.all_fo_stocks)} STOCKS...")
        
        all_symbols = self.all_fo_stocks + [f"{symbol}.NS" for symbol in self.all_fo_stocks]
        index_symbols = list(self.indices.values())
        
        successful_downloads = []
        failed_downloads = []
        
        for i, symbol in enumerate(self.all_fo_stocks):
            try:
                # Try with .NS suffix first
                data = yf.download(f"{symbol}.NS", period=period, interval=interval, progress=False)
                if data.empty:
                    # Try without suffix
                    data = yf.download(symbol, period=period, interval=interval, progress=False)
                
                if not data.empty and len(data) > 50:  # Minimum data requirement
                    self.market_data[symbol] = data
                    successful_downloads.append(symbol)
                    if (i + 1) % 10 == 0:
                        print(f"✅ Downloaded {i+1}/{len(self.all_fo_stocks)} stocks")
                else:
                    failed_downloads.append(symbol)
                    
            except Exception as e:
                failed_downloads.append(symbol)
                continue
            
            # Rate limiting
            time.sleep(0.1)
        
        # Download indices
        for name, symbol in self.indices.items():
            try:
                data = yf.download(symbol, period=period, interval=interval, progress=False)
                if not data.empty:
                    self.market_data[name] = data
                    successful_downloads.append(name)
            except:
                failed_downloads.append(name)
        
        print(f"✅ Successfully downloaded: {len(successful_downloads)} securities")
        print(f"❌ Failed downloads: {len(failed_downloads)} securities")
        
        if failed_downloads:
            print(f"Failed: {failed_downloads[:10]}...")  # Show first 10 failures
        
        return len(successful_downloads)
    
    def engineer_advanced_features(self):
        """Engineer advanced features for deep learning"""
        print("🔧 ENGINEERING ADVANCED FEATURES...")
        
        feature_data = {}
        
        for symbol, data in self.market_data.items():
            try:
                # Basic features
                features = pd.DataFrame(index=data.index)
                
                # Price features
                features['close'] = data['Close']
                features['high'] = data['High'] 
                features['low'] = data['Low']
                features['open'] = data['Open']
                features['volume'] = data['Volume']
                
                # Returns
                features['returns'] = data['Close'].pct_change()
                features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
                
                # Volatility
                features['volatility'] = features['returns'].rolling(20).std()
                features['high_low_ratio'] = data['High'] / data['Low']
                
                # Technical Indicators
                features['sma_20'] = ta.trend.sma_indicator(data['Close'], window=20)
                features['sma_50'] = ta.trend.sma_indicator(data['Close'], window=50)
                features['ema_12'] = ta.trend.ema_indicator(data['Close'], window=12)
                features['ema_26'] = ta.trend.ema_indicator(data['Close'], window=26)
                
                # Momentum
                features['rsi'] = ta.momentum.rsi(data['Close'], window=14)
                features['macd'] = ta.trend.macd_diff(data['Close'])
                features['stoch'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
                
                # Volume indicators
                features['volume_sma'] = data['Volume'].rolling(20).mean()
                features['volume_ratio'] = data['Volume'] / features['volume_sma']
                features['obv'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
                
                # Bollinger Bands
                bb = ta.volatility.BollingerBands(data['Close'])
                features['bb_upper'] = bb.bollinger_hband()
                features['bb_lower'] = bb.bollinger_lband()
                features['bb_position'] = (data['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
                
                # Advanced patterns
                features['doji'] = self.detect_doji_pattern(data)
                features['hammer'] = self.detect_hammer_pattern(data)
                features['engulfing'] = self.detect_engulfing_pattern(data)
                
                # Fractal dimensions
                features['fractal_dim'] = self.calculate_fractal_dimension(data['Close'])
                
                # Gap detection
                features['gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
                features['gap_up'] = (features['gap'] > 0.02).astype(int)
                features['gap_down'] = (features['gap'] < -0.02).astype(int)
                
                # Support/Resistance levels
                features['support_level'] = self.find_support_resistance(data['Low'], mode='support')
                features['resistance_level'] = self.find_support_resistance(data['High'], mode='resistance')
                
                # Clean and store
                features = features.dropna()
                if len(features) > 30:  # Minimum features requirement
                    feature_data[symbol] = features
                    
            except Exception as e:
                print(f"⚠️ Error processing {symbol}: {str(e)}")
                continue
        
        self.feature_data = feature_data
        print(f"✅ Engineered features for {len(feature_data)} securities")
        return feature_data
    
    def detect_doji_pattern(self, data):
        """Detect Doji candlestick patterns"""
        body_size = abs(data['Close'] - data['Open'])
        candle_range = data['High'] - data['Low']
        return (body_size / candle_range < 0.1).astype(int)
    
    def detect_hammer_pattern(self, data):
        """Detect Hammer candlestick patterns"""
        body_size = abs(data['Close'] - data['Open'])
        lower_shadow = np.minimum(data['Open'], data['Close']) - data['Low']
        upper_shadow = data['High'] - np.maximum(data['Open'], data['Close'])
        return ((lower_shadow > 2 * body_size) & (upper_shadow < body_size)).astype(int)
    
    def detect_engulfing_pattern(self, data):
        """Detect Engulfing patterns"""
        prev_body = abs(data['Close'].shift(1) - data['Open'].shift(1))
        curr_body = abs(data['Close'] - data['Open'])
        bullish_engulfing = ((data['Close'] > data['Open']) & 
                           (data['Close'].shift(1) < data['Open'].shift(1)) &
                           (curr_body > prev_body)).astype(int)
        return bullish_engulfing
    
    def calculate_fractal_dimension(self, price_series, window=50):
        """Calculate fractal dimension using box counting method"""
        fractal_dims = []
        for i in range(window, len(price_series)):
            series = price_series.iloc[i-window:i].values
            # Simplified fractal dimension calculation
            try:
                # Calculate relative ranges at different scales
                scales = [2, 4, 8, 16]
                ranges = []
                for scale in scales:
                    scaled_series = series[::scale]
                    if len(scaled_series) > 1:
                        ranges.append(np.max(scaled_series) - np.min(scaled_series))
                
                if len(ranges) >= 2:
                    # Linear regression to find slope (fractal dimension)
                    log_scales = np.log(scales[:len(ranges)])
                    log_ranges = np.log(ranges)
                    fractal_dim = np.polyfit(log_scales, log_ranges, 1)[0]
                    fractal_dims.append(fractal_dim)
                else:
                    fractal_dims.append(np.nan)
            except:
                fractal_dims.append(np.nan)
        
        # Pad the beginning with NaN
        result = [np.nan] * window + fractal_dims
        return pd.Series(result, index=price_series.index)
    
    def find_support_resistance(self, price_series, window=20, mode='support'):
        """Find support and resistance levels"""
        if mode == 'support':
            extrema = argrelextrema(price_series.values, np.less, order=window)[0]
        else:
            extrema = argrelextrema(price_series.values, np.greater, order=window)[0]
        
        levels = []
        for i in range(len(price_series)):
            if i in extrema:
                levels.append(price_series.iloc[i])
            else:
                # Find nearest level
                if len(levels) > 0:
                    levels.append(levels[-1])
                else:
                    levels.append(np.nan)
        
        return pd.Series(levels, index=price_series.index)
    
    def discover_hidden_patterns(self):
        """Discover hidden market patterns using deep learning"""
        print("🔍 DISCOVERING HIDDEN MARKET PATTERNS...")
        
        if not hasattr(self, 'feature_data') or not self.feature_data:
            print("❌ No feature data available. Run engineer_advanced_features() first.")
            return
        
        # Combine all features into a master matrix
        all_features = []
        symbol_mapping = []
        
        for symbol, features in self.feature_data.items():
            # Select recent data (last 100 days)
            recent_features = features.tail(100)
            if len(recent_features) >= 30:
                # Select key features for pattern analysis
                pattern_features = recent_features[[
                    'returns', 'volatility', 'rsi', 'macd', 'volume_ratio',
                    'bb_position', 'doji', 'hammer', 'engulfing', 'gap'
                ]].fillna(0)
                
                all_features.append(pattern_features.values)
                symbol_mapping.extend([symbol] * len(pattern_features))
        
        if not all_features:
            print("❌ No suitable features for pattern analysis")
            return
        
        # Create master feature matrix
        feature_matrix = np.vstack(all_features)
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # 1. Clustering Analysis - Find Hidden Groups
        print("🔍 Clustering Analysis - Finding Hidden Market Groups...")
        
        # DBSCAN for anomaly detection
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(feature_matrix_scaled)
        
        # K-Means for group identification
        kmeans = KMeans(n_clusters=8, random_state=42)
        kmeans_labels = kmeans.fit_predict(feature_matrix_scaled)
        
        # 2. Dimensionality Reduction - Uncover Hidden Dimensions
        print("🔍 Dimensionality Reduction - Uncovering Hidden Dimensions...")
        
        # PCA for variance explanation
        pca = PCA(n_components=5)
        pca_features = pca.fit_transform(feature_matrix_scaled)
        
        # t-SNE for pattern visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_features = tsne.fit_transform(feature_matrix_scaled)
        
        # 3. Anomaly Detection - Find Market Anomalies
        print("🔍 Anomaly Detection - Finding Market Anomalies...")
        
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = isolation_forest.fit_predict(feature_matrix_scaled)
        
        # 4. Neural Network Pattern Recognition
        print("🔍 Neural Network Pattern Recognition...")
        
        # Prepare data for neural network
        X = feature_matrix_scaled[:-1]  # Features
        y = feature_matrix_scaled[1:, 0]  # Next period returns
        
        # Multi-layer Perceptron
        mlp = MLPRegressor(hidden_layer_sizes=(100, 50, 25), 
                          max_iter=500, random_state=42)
        mlp.fit(X, y)
        
        # Store patterns
        self.secret_patterns = {
            'cluster_groups': {
                'dbscan_labels': cluster_labels,
                'kmeans_labels': kmeans_labels,
                'symbols': symbol_mapping
            },
            'dimensionality': {
                'pca_components': pca_features,
                'pca_variance_ratio': pca.explained_variance_ratio_,
                'tsne_embedding': tsne_features
            },
            'anomalies': {
                'scores': anomaly_scores,
                'symbols': symbol_mapping
            },
            'neural_patterns': {
                'model': mlp,
                'feature_importance': np.abs(mlp.coefs_[0]).mean(axis=1)
            }
        }
        
        print("✅ Hidden patterns discovered successfully!")
        return self.secret_patterns
    
    def analyze_hidden_correlations(self):
        """Analyze hidden correlations and pair trading opportunities"""
        print("🔍 ANALYZING HIDDEN CORRELATIONS...")
        
        if not self.feature_data:
            print("❌ No feature data available")
            return
        
        # Create correlation matrix
        price_data = {}
        for symbol, features in self.feature_data.items():
            price_data[symbol] = features['returns'].dropna()
        
        # Align all series to same time period
        price_df = pd.DataFrame(price_data)
        price_df = price_df.dropna()
        
        if price_df.empty:
            print("❌ No aligned price data for correlation analysis")
            return
        
        # Calculate correlations
        correlation_matrix = price_df.corr()
        
        # Find high correlations (potential pairs)
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.7:  # High correlation threshold
                    high_correlations.append({
                        'stock1': correlation_matrix.columns[i],
                        'stock2': correlation_matrix.columns[j],
                        'correlation': corr,
                        'type': 'positive' if corr > 0 else 'negative'
                    })
        
        # Sector-wise correlation analysis
        sector_correlations = {}
        for sector, stocks in self.sector_mapping.items():
            sector_stocks = [s for s in stocks if s in price_df.columns]
            if len(sector_stocks) >= 2:
                sector_corr = price_df[sector_stocks].corr()
                sector_correlations[sector] = {
                    'avg_correlation': sector_corr.values[np.triu_indices_from(sector_corr.values, k=1)].mean(),
                    'correlation_matrix': sector_corr
                }
        
        # Dynamic correlation analysis (rolling correlations)
        rolling_correlations = {}
        window = 30
        for pair in high_correlations[:10]:  # Top 10 pairs
            stock1, stock2 = pair['stock1'], pair['stock2']
            if stock1 in price_df.columns and stock2 in price_df.columns:
                rolling_corr = price_df[stock1].rolling(window).corr(price_df[stock2])
                rolling_correlations[f"{stock1}_{stock2}"] = rolling_corr
        
        self.hidden_correlations = {
            'correlation_matrix': correlation_matrix,
            'high_correlations': high_correlations,
            'sector_correlations': sector_correlations,
            'rolling_correlations': rolling_correlations
        }
        
        print(f"✅ Found {len(high_correlations)} high correlation pairs")
        return self.hidden_correlations
    
    def detect_market_anomalies(self):
        """Detect market anomalies and unusual patterns"""
        print("🚨 DETECTING MARKET ANOMALIES...")
        
        anomalies = {}
        
        for symbol, features in self.feature_data.items():
            try:
                stock_anomalies = []
                
                # Volume anomalies
                volume_zscore = stats.zscore(features['volume_ratio'].dropna())
                volume_anomalies = np.where(np.abs(volume_zscore) > 3)[0]
                
                # Price anomalies
                return_zscore = stats.zscore(features['returns'].dropna())
                price_anomalies = np.where(np.abs(return_zscore) > 3)[0]
                
                # Gap anomalies
                gap_anomalies = np.where(np.abs(features['gap']) > 0.05)[0]  # 5% gaps
                
                # RSI extremes
                rsi_extremes = np.where((features['rsi'] < 10) | (features['rsi'] > 90))[0]
                
                # Combine all anomalies
                all_anomalies = {
                    'volume_anomalies': len(volume_anomalies),
                    'price_anomalies': len(price_anomalies),
                    'gap_anomalies': len(gap_anomalies),
                    'rsi_extremes': len(rsi_extremes),
                    'total_anomalies': len(volume_anomalies) + len(price_anomalies) + len(gap_anomalies) + len(rsi_extremes)
                }
                
                if all_anomalies['total_anomalies'] > 0:
                    anomalies[symbol] = all_anomalies
                    
            except Exception as e:
                continue
        
        # Sort by total anomalies
        sorted_anomalies = dict(sorted(anomalies.items(), 
                                     key=lambda x: x[1]['total_anomalies'], 
                                     reverse=True))
        
        self.market_anomalies = sorted_anomalies
        print(f"✅ Detected anomalies in {len(sorted_anomalies)} stocks")
        return sorted_anomalies
    
    def discover_pair_trading_opportunities(self):
        """Discover pair trading and arbitrage opportunities"""
        print("💰 DISCOVERING PAIR TRADING OPPORTUNITIES...")
        
        if not hasattr(self, 'hidden_correlations') or not self.hidden_correlations:
            print("❌ Run analyze_hidden_correlations() first")
            return
        
        pair_opportunities = []
        
        # Analyze high correlation pairs
        for pair in self.hidden_correlations['high_correlations']:
            stock1, stock2 = pair['stock1'], pair['stock2']
            correlation = pair['correlation']
            
            if stock1 in self.feature_data and stock2 in self.feature_data:
                # Get recent price data
                data1 = self.feature_data[stock1]['close'].tail(50)
                data2 = self.feature_data[stock2]['close'].tail(50)
                
                # Calculate spread
                spread = (data1 / data1.iloc[0]) - (data2 / data2.iloc[0])
                spread_mean = spread.mean()
                spread_std = spread.std()
                current_spread = spread.iloc[-1]
                
                # Z-score of current spread
                spread_zscore = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
                
                # Trading signal
                signal = None
                if spread_zscore > 2:
                    signal = f"SELL {stock1} / BUY {stock2}"
                elif spread_zscore < -2:
                    signal = f"BUY {stock1} / SELL {stock2}"
                
                if signal:
                    pair_opportunities.append({
                        'pair': f"{stock1}_{stock2}",
                        'correlation': correlation,
                        'spread_zscore': spread_zscore,
                        'signal': signal,
                        'confidence': min(abs(spread_zscore) / 2, 1.0)
                    })
        
        # Sort by confidence
        pair_opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        
        self.pair_opportunities = pair_opportunities
        print(f"✅ Found {len(pair_opportunities)} pair trading opportunities")
        return pair_opportunities
    
    def reveal_sector_secrets(self):
        """Reveal hidden sector rotation and momentum secrets"""
        print("🏭 REVEALING SECTOR SECRETS...")
        
        sector_analysis = {}
        
        for sector, stocks in self.sector_mapping.items():
            sector_stocks = [s for s in stocks if s in self.feature_data]
            
            if len(sector_stocks) >= 3:
                # Calculate sector momentum
                sector_returns = []
                sector_volumes = []
                
                for stock in sector_stocks:
                    features = self.feature_data[stock]
                    recent_return = features['returns'].tail(20).mean()
                    recent_volume = features['volume_ratio'].tail(20).mean()
                    
                    sector_returns.append(recent_return)
                    sector_volumes.append(recent_volume)
                
                # Sector metrics
                avg_return = np.mean(sector_returns)
                avg_volume = np.mean(sector_volumes)
                momentum_score = avg_return * avg_volume
                
                # Sector leadership (best performing stock)
                best_stock_idx = np.argmax(sector_returns)
                worst_stock_idx = np.argmin(sector_returns)
                
                sector_analysis[sector] = {
                    'avg_return': avg_return,
                    'avg_volume': avg_volume,
                    'momentum_score': momentum_score,
                    'leader': sector_stocks[best_stock_idx],
                    'laggard': sector_stocks[worst_stock_idx],
                    'stocks_analyzed': len(sector_stocks)
                }
        
        # Sort sectors by momentum
        sorted_sectors = dict(sorted(sector_analysis.items(), 
                                   key=lambda x: x[1]['momentum_score'], 
                                   reverse=True))
        
        self.sector_secrets = sorted_sectors
        print(f"✅ Analyzed {len(sorted_sectors)} sectors")
        return sorted_sectors
    
    def generate_market_intelligence_report(self):
        """Generate comprehensive market intelligence report"""
        print("📊 GENERATING MARKET INTELLIGENCE REPORT...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Market Intelligence Report
        report = {
            'timestamp': timestamp,
            'analysis_summary': {
                'total_stocks_analyzed': len(self.feature_data) if hasattr(self, 'feature_data') else 0,
                'patterns_discovered': len(self.secret_patterns) if self.secret_patterns else 0,
                'correlations_found': len(self.hidden_correlations.get('high_correlations', [])) if self.hidden_correlations else 0,
                'anomalies_detected': len(self.market_anomalies) if self.market_anomalies else 0,
                'pair_opportunities': len(self.pair_opportunities) if self.pair_opportunities else 0,
                'sectors_analyzed': len(self.sector_secrets) if self.sector_secrets else 0
            },
            'top_discoveries': {
                'hot_sectors': list(self.sector_secrets.keys())[:3] if self.sector_secrets else [],
                'top_anomalies': list(self.market_anomalies.keys())[:5] if self.market_anomalies else [],
                'best_pairs': [p['pair'] for p in self.pair_opportunities[:3]] if self.pair_opportunities else [],
                'pattern_insights': f"Discovered {len(self.secret_patterns)} hidden pattern types" if self.secret_patterns else "No patterns discovered"
            },
            'detailed_analysis': {
                'secret_patterns': self.secret_patterns,
                'hidden_correlations': self.hidden_correlations,
                'market_anomalies': self.market_anomalies,
                'pair_opportunities': self.pair_opportunities,
                'sector_secrets': self.sector_secrets
            }
        }
        
        # Save report
        os.makedirs('market_intelligence', exist_ok=True)
        report_file = f'market_intelligence/market_secrets_report_{timestamp}.json'
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_report = convert_numpy(report)
        
        with open(report_file, 'w') as f:
            json.dump(serializable_report, f, indent=2, default=str)
        
        print(f"✅ Market Intelligence Report saved: {report_file}")
        
        # Print Executive Summary
        self.print_executive_summary(report)
        
        return report
    
    def print_executive_summary(self, report):
        """Print executive summary of discoveries"""
        print("\n" + "="*60)
        print("🔮 MARKET SECRET REVEALER - EXECUTIVE SUMMARY 🔮")
        print("="*60)
        
        summary = report['analysis_summary']
        discoveries = report['top_discoveries']
        
        print(f"📊 ANALYSIS SCOPE:")
        print(f"   • Stocks Analyzed: {summary['total_stocks_analyzed']}")
        print(f"   • Patterns Discovered: {summary['patterns_discovered']}")
        print(f"   • Correlations Found: {summary['correlations_found']}")
        print(f"   • Anomalies Detected: {summary['anomalies_detected']}")
        print(f"   • Pair Opportunities: {summary['pair_opportunities']}")
        print(f"   • Sectors Analyzed: {summary['sectors_analyzed']}")
        
        print(f"\n🏆 TOP DISCOVERIES:")
        print(f"   🔥 Hot Sectors: {', '.join(discoveries['hot_sectors'])}")
        print(f"   ⚠️ Top Anomalies: {', '.join(discoveries['top_anomalies'])}")
        print(f"   💰 Best Pairs: {', '.join(discoveries['best_pairs'])}")
        print(f"   🧠 Pattern Insights: {discoveries['pattern_insights']}")
        
        if self.sector_secrets:
            print(f"\n🏭 SECTOR MOMENTUM RANKING:")
            for i, (sector, data) in enumerate(list(self.sector_secrets.items())[:5], 1):
                print(f"   {i}. {sector}: {data['momentum_score']:.4f} (Leader: {data['leader']})")
        
        if self.pair_opportunities:
            print(f"\n💰 TOP PAIR TRADING OPPORTUNITIES:")
            for i, pair in enumerate(self.pair_opportunities[:3], 1):
                print(f"   {i}. {pair['pair']}: {pair['signal']} (Confidence: {pair['confidence']:.2f})")
        
        if self.market_anomalies:
            print(f"\n⚠️ MARKET ANOMALIES DETECTED:")
            for i, (stock, anomaly) in enumerate(list(self.market_anomalies.items())[:5], 1):
                print(f"   {i}. {stock}: {anomaly['total_anomalies']} anomalies detected")
        
        print("\n" + "="*60)
        print("🔮 MARKET SECRETS REVEALED - ANALYSIS COMPLETE 🔮")
        print("="*60)

def main():
    """Main execution function"""
    print("🚀 STARTING MARKET SECRET REVEALER...")
    
    # Initialize the system
    revealer = MarketSecretRevealer()
    
    # Execute full analysis pipeline
    try:
        # 1. Collect comprehensive data
        print(f"\n{'='*60}")
        print("PHASE 1: DATA COLLECTION")
        print(f"{'='*60}")
        stocks_downloaded = revealer.collect_comprehensive_data(period="1y")
        
        if stocks_downloaded < 10:
            print("❌ Insufficient data downloaded. Exiting...")
            return
        
        # 2. Engineer advanced features
        print(f"\n{'='*60}")
        print("PHASE 2: FEATURE ENGINEERING")
        print(f"{'='*60}")
        revealer.engineer_advanced_features()
        
        # 3. Discover hidden patterns
        print(f"\n{'='*60}")
        print("PHASE 3: PATTERN DISCOVERY")
        print(f"{'='*60}")
        revealer.discover_hidden_patterns()
        
        # 4. Analyze correlations
        print(f"\n{'='*60}")
        print("PHASE 4: CORRELATION ANALYSIS")
        print(f"{'='*60}")
        revealer.analyze_hidden_correlations()
        
        # 5. Detect anomalies
        print(f"\n{'='*60}")
        print("PHASE 5: ANOMALY DETECTION")
        print(f"{'='*60}")
        revealer.detect_market_anomalies()
        
        # 6. Find pair trading opportunities
        print(f"\n{'='*60}")
        print("PHASE 6: PAIR TRADING DISCOVERY")
        print(f"{'='*60}")
        revealer.discover_pair_trading_opportunities()
        
        # 7. Reveal sector secrets
        print(f"\n{'='*60}")
        print("PHASE 7: SECTOR ANALYSIS")
        print(f"{'='*60}")
        revealer.reveal_sector_secrets()
        
        # 8. Generate comprehensive report
        print(f"\n{'='*60}")
        print("PHASE 8: INTELLIGENCE REPORT GENERATION")
        print(f"{'='*60}")
        revealer.generate_market_intelligence_report()
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()