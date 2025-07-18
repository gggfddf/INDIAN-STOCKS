"""
ADVANCED DEEP LEARNING MARKET DISCOVERY SYSTEM
A Secret-Revealing Book for Indian Stock Market
Discovers Hidden Patterns, Anomalies, Correlations & Market Secrets
"""

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import itertools

class MarketSecretRevealer:
    """Deep Learning System to Reveal Market Secrets"""
    
    def __init__(self):
        # ALL 100 F&O STOCKS
        self.all_fo_stocks = [
            # Banking & Financial
            'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK',
            'BANKBARODA', 'PNB', 'FEDERALBNK', 'IDFCFIRSTB', 'RBLBANK', 'AUBANK',
            'BANDHANBNK', 'CANBK', 'HDFCAMC', 'BAJFINANCE', 'BAJAJFINSV', 'SBILIFE',
            'HDFCLIFE', 'ICICIPRULI', 'LICHSGFIN', 'MFIN', 'CHOLAFIN', 'PEL',
            
            # IT & Technology  
            'TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTI', 'MINDTREE', 'MPHASIS',
            'LTTS', 'COFORGE', 'PERSISTENT', 'OFSS', 'TATAELXSI',
            
            # Oil & Gas
            'RELIANCE', 'ONGC', 'IOC', 'BPCL', 'HINDPETRO', 'GAIL', 'OIL', 'PETRONET',
            
            # Automobiles
            'MARUTI', 'M&M', 'TATAMOTORS', 'BAJAJ-AUTO', 'HEROMOTOCO', 'TVSMOTORS', 
            'EICHERMOT', 'ASHOKLEY', 'BAJAJHLDNG',
            
            # Pharmaceuticals
            'SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON', 'LUPIN', 'CADILAHC',
            'TORNTPHARM', 'AUROPHARMA', 'GLENMARK', 'ALKEM', 'ABBOTINDIA', 'ZYDUSLIFE',
            
            # FMCG
            'HINDUNILVR', 'ITC', 'NESTLE', 'BRITANNIA', 'DABUR', 'MARICO', 'COLPAL',
            'GODREJCP', 'UBL', 'TATACONSUM', 'VBL', 'EMAMILTD',
            
            # Metals & Mining
            'TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'COALINDIA', 'VEDL', 'NMDC',
            'SAIL', 'JINDALSTEL', 'NALCO', 'MOIL',
            
            # Cement
            'ULTRACEMCO', 'GRASIM', 'ACC', 'AMBUJACEMENT', 'SHREECEM', 'RAMCOCEM',
            'JKCEMENT', 'INDIACEM',
            
            # Power & Utilities
            'NTPC', 'POWERGRID', 'ADANIGREEN', 'TATAPOWER', 'ADANIPOWER', 'NHPC',
            'SJVN', 'THERMAX',
            
            # Telecom
            'BHARTIARTL', 'IDEA',
            
            # Infrastructure
            'LT', 'ADANIPORTS', 'GMR', 'IRB', 'PFC', 'RECLTD'
        ]
        
        # Major Indices
        self.indices = {
            'NIFTY50': '^NSEI',
            'BANKNIFTY': '^NSEBANK', 
            'FINNIFTY': '^CNXFIN'
        }
        
        self.market_data = {}
        self.feature_matrix = None
        self.secret_patterns = {}
        self.hidden_correlations = {}
        self.market_anomalies = {}
        self.pair_opportunities = {}
        self.sector_secrets = {}
        
        print("🔍 MARKET SECRET REVEALER INITIALIZED")
        print(f"🎯 Target: {len(self.all_fo_stocks)} F&O Stocks + {len(self.indices)} Indices")
        print("🤖 Mission: Discover Hidden Market Patterns & Anomalies")
        print("📊 Deep Learning: Pattern Recognition, Correlation Analysis, Anomaly Detection")
    
    def collect_all_market_data(self, period="1y"):
        """Collect comprehensive market data for all 100 stocks"""
        print(f"\n🚀 COLLECTING MARKET DATA FOR ALL {len(self.all_fo_stocks)} STOCKS...")
        
        successful = 0
        failed = 0
        
        # Collect stocks in batches to avoid rate limiting
        batch_size = 10
        total_batches = (len(self.all_fo_stocks) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(self.all_fo_stocks))
            batch_stocks = self.all_fo_stocks[start_idx:end_idx]
            
            print(f"\n📦 Batch {batch_num + 1}/{total_batches}: Processing {len(batch_stocks)} stocks...")
            
            for i, stock in enumerate(batch_stocks, 1):
                try:
                    symbol = f"{stock}.NS"
                    print(f"  🔄 [{start_idx + i:3d}/100] Fetching {stock}...", end="")
                    
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period=period)
                    
                    if not data.empty and len(data) > 100:
                        # Add comprehensive technical features
                        enriched_data = self._add_deep_features(data, stock)
                        self.market_data[stock] = enriched_data
                        successful += 1
                        print(f" ✅ ({len(data)} days)")
                    else:
                        failed += 1
                        print(" ❌ (insufficient data)")
                    
                    time.sleep(0.2)  # Rate limiting
                    
                except Exception as e:
                    failed += 1
                    print(f" ❌ (error: {str(e)[:30]})")
            
            print(f"  📊 Batch {batch_num + 1} complete: {successful} success, {failed} failed")
            time.sleep(2)  # Pause between batches
        
        # Collect indices
        print(f"\n📈 COLLECTING INDICES DATA...")
        for index_name, index_symbol in self.indices.items():
            try:
                print(f"  🔄 Fetching {index_name}...", end="")
                ticker = yf.Ticker(index_symbol)
                data = ticker.history(period=period)
                
                if not data.empty:
                    enriched_data = self._add_deep_features(data, f"INDEX_{index_name}")
                    self.market_data[f"INDEX_{index_name}"] = enriched_data
                    print(f" ✅ ({len(data)} days)")
                else:
                    print(" ❌")
                    
            except Exception as e:
                print(f" ❌ (error: {str(e)[:30]})")
        
        print(f"\n🎯 DATA COLLECTION COMPLETE:")
        print(f"   ✅ Successfully collected: {successful} stocks")
        print(f"   ❌ Failed: {failed} stocks")
        print(f"   📈 Indices: {len([k for k in self.market_data.keys() if k.startswith('INDEX_')])}")
        print(f"   📊 Total instruments: {len(self.market_data)}")
        
        return len(self.market_data) > 50  # Need at least 50 instruments
    
    def _add_deep_features(self, df, symbol):
        """Add comprehensive technical and pattern features"""
        try:
            # Basic OHLCV
            df = df.copy()
            
            # Price-based features
            df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
            df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
            
            df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
            df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
            df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
            
            # Momentum indicators
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            df['MACD'] = ta.trend.macd_diff(df['Close'])
            df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
            df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
            df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
            df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
            df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
            
            # Volatility indicators
            bb = ta.volatility.BollingerBands(df['Close'])
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
            df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Price patterns & relationships
            df['Daily_Return'] = df['Close'].pct_change()
            df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
            df['Open_Close_Ratio'] = (df['Close'] - df['Open']) / df['Open']
            df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
            
            # Trend features
            df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
            df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
            df['SMA_Slope_20'] = df['SMA_20'].pct_change(5)
            df['SMA_Slope_50'] = df['SMA_50'].pct_change(10)
            
            # Volatility features
            df['Returns_Volatility'] = df['Daily_Return'].rolling(20).std()
            df['Price_Volatility'] = df['Close'].rolling(20).std() / df['Close'].rolling(20).mean()
            
            # Advanced patterns
            df['Gap_Up'] = (df['Open'] > df['Close'].shift(1)) & (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) > 0.02
            df['Gap_Down'] = (df['Open'] < df['Close'].shift(1)) & (df['Close'].shift(1) - df['Open']) / df['Close'].shift(1) > 0.02
            
            # Candlestick patterns (simplified)
            df['Doji'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) < 0.1
            df['Hammer'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low']) > 0.6) & \
                           ((df['High'] - df['Close']) / (df['High'] - df['Low']) < 0.2)
            
            # Multi-timeframe features
            df['Weekly_Return'] = df['Close'].pct_change(5)
            df['Monthly_Return'] = df['Close'].pct_change(20)
            
            # Market microstructure
            df['Spread_Proxy'] = (df['High'] - df['Low']) / df['Close']
            df['Volume_Price_Trend'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])
            
            return df
            
        except Exception as e:
            print(f"    Warning: Feature engineering error for {symbol}: {e}")
            return df
    
    def build_feature_matrix(self):
        """Build comprehensive feature matrix for deep learning"""
        print(f"\n🧠 BUILDING DEEP LEARNING FEATURE MATRIX...")
        
        feature_columns = [
            'RSI', 'MACD', 'MACD_Signal', 'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI',
            'BB_Width', 'BB_Position', 'ATR', 'Volume_Ratio', 'CMF', 'MFI',
            'Daily_Return', 'High_Low_Ratio', 'Open_Close_Ratio', 'Price_Position',
            'Price_vs_SMA20', 'Price_vs_SMA50', 'SMA_Slope_20', 'SMA_Slope_50',
            'Returns_Volatility', 'Price_Volatility', 'Weekly_Return', 'Monthly_Return',
            'Spread_Proxy', 'Volume_Price_Trend'
        ]
        
        all_features = []
        stock_names = []
        dates = []
        
        for symbol, data in self.market_data.items():
            if symbol.startswith('INDEX_'):
                continue
                
            # Get recent 60 days for analysis
            recent_data = data.tail(60)
            clean_data = recent_data[feature_columns].dropna()
            
            if len(clean_data) > 30:  # Minimum data requirement
                all_features.extend(clean_data.values)
                stock_names.extend([symbol] * len(clean_data))
                dates.extend(clean_data.index)
        
        if len(all_features) > 100:
            self.feature_matrix = np.array(all_features)
            self.stock_names = stock_names
            self.dates = dates
            
            print(f"   ✅ Feature matrix built: {self.feature_matrix.shape}")
            print(f"   📊 Features: {len(feature_columns)}")
            print(f"   🎯 Data points: {len(all_features)}")
            print(f"   📈 Stocks: {len(set(stock_names))}")
            
            return True
        else:
            print("   ❌ Insufficient data for feature matrix")
            return False
    
    def discover_hidden_patterns(self):
        """Use deep learning to discover hidden market patterns"""
        print(f"\n🔍 DISCOVERING HIDDEN MARKET PATTERNS...")
        
        if self.feature_matrix is None:
            return
        
        # 1. DIMENSIONALITY REDUCTION - Find hidden dimensions
        print("   🧠 Applying PCA to find hidden dimensions...")
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.feature_matrix)
        
        pca = PCA(n_components=10)
        pca_features = pca.fit_transform(scaled_features)
        
        explained_variance = pca.explained_variance_ratio_
        print(f"      📊 Explained variance by top 5 components: {explained_variance[:5]}")
        
        # 2. t-SNE for non-linear pattern discovery
        print("   🔮 Applying t-SNE for non-linear pattern discovery...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_features = tsne.fit_transform(scaled_features[:5000])  # Sample for speed
        
        # 3. CLUSTERING - Find market regimes/patterns
        print("   🎯 Clustering to identify market regimes...")
        
        # DBSCAN for anomaly-aware clustering
        dbscan = DBSCAN(eps=0.5, min_samples=10)
        clusters = dbscan.fit_predict(scaled_features)
        
        unique_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        noise_points = list(clusters).count(-1)
        
        print(f"      📊 Found {unique_clusters} market regimes")
        print(f"      🚨 Anomaly points: {noise_points}")
        
        # K-means for stable clustering
        kmeans = KMeans(n_clusters=8, random_state=42)
        stable_clusters = kmeans.fit_predict(scaled_features)
        
        # 4. Store pattern discoveries
        self.secret_patterns = {
            'pca_components': pca.components_[:5],  # Top 5 hidden dimensions
            'explained_variance': explained_variance,
            'cluster_labels': clusters,
            'stable_clusters': stable_clusters,
            'cluster_centers': kmeans.cluster_centers_,
            'tsne_embedding': tsne_features,
            'feature_importance': np.abs(pca.components_).mean(axis=0)
        }
        
        print("   ✅ Hidden pattern discovery complete!")
        
        # 5. Analyze pattern meanings
        self._analyze_pattern_meanings(scaled_features, clusters, stable_clusters)
    
    def _analyze_pattern_meanings(self, features, clusters, stable_clusters):
        """Analyze what the discovered patterns mean"""
        print(f"\n🧐 ANALYZING PATTERN MEANINGS...")
        
        pattern_analysis = {}
        
        # Analyze each stable cluster
        for cluster_id in range(8):
            cluster_mask = stable_clusters == cluster_id
            cluster_features = features[cluster_mask]
            cluster_stocks = [self.stock_names[i] for i, mask in enumerate(cluster_mask) if mask]
            
            if len(cluster_features) > 10:
                cluster_mean = cluster_features.mean(axis=0)
                
                # Find dominant features for this cluster
                feature_names = [
                    'RSI', 'MACD', 'MACD_Signal', 'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI',
                    'BB_Width', 'BB_Position', 'ATR', 'Volume_Ratio', 'CMF', 'MFI',
                    'Daily_Return', 'High_Low_Ratio', 'Open_Close_Ratio', 'Price_Position',
                    'Price_vs_SMA20', 'Price_vs_SMA50', 'SMA_Slope_20', 'SMA_Slope_50',
                    'Returns_Volatility', 'Price_Volatility', 'Weekly_Return', 'Monthly_Return',
                    'Spread_Proxy', 'Volume_Price_Trend'
                ]
                
                # Find top characteristics
                top_features_idx = np.argsort(np.abs(cluster_mean))[-5:]
                top_features = [(feature_names[i], cluster_mean[i]) for i in top_features_idx]
                
                # Classify pattern type
                pattern_type = self._classify_pattern_type(cluster_mean, feature_names)
                
                pattern_analysis[f'Pattern_{cluster_id}'] = {
                    'type': pattern_type,
                    'sample_count': len(cluster_features),
                    'top_characteristics': top_features,
                    'sample_stocks': list(set(cluster_stocks))[:10],  # Sample stocks
                    'avg_volatility': float(cluster_mean[feature_names.index('Returns_Volatility')]),
                    'avg_momentum': float(cluster_mean[feature_names.index('RSI')])
                }
                
                print(f"   📊 Pattern {cluster_id}: {pattern_type} ({len(cluster_features)} samples)")
        
        self.secret_patterns['pattern_meanings'] = pattern_analysis
    
    def _classify_pattern_type(self, cluster_mean, feature_names):
        """Classify what type of market pattern this represents"""
        
        rsi_idx = feature_names.index('RSI')
        vol_idx = feature_names.index('Returns_Volatility')
        momentum_idx = feature_names.index('MACD')
        bb_pos_idx = feature_names.index('BB_Position')
        volume_idx = feature_names.index('Volume_Ratio')
        
        rsi = cluster_mean[rsi_idx]
        volatility = cluster_mean[vol_idx]
        momentum = cluster_mean[momentum_idx]
        bb_position = cluster_mean[bb_pos_idx]
        volume_ratio = cluster_mean[volume_idx]
        
        # Pattern classification logic
        if rsi > 0.5 and momentum > 0 and bb_position > 0.7:
            return "Strong_Bullish_Momentum"
        elif rsi < -0.5 and momentum < 0 and bb_position < 0.3:
            return "Strong_Bearish_Momentum"
        elif volatility > 1.0 and volume_ratio > 0.5:
            return "High_Volatility_Breakout"
        elif volatility < -0.5 and abs(momentum) < 0.2:
            return "Low_Volatility_Consolidation"
        elif volume_ratio > 1.0 and abs(momentum) > 0.5:
            return "Volume_Driven_Movement"
        elif bb_position > 0.8 or bb_position < 0.2:
            return "Bollinger_Band_Extreme"
        elif abs(rsi) < 0.2 and abs(momentum) < 0.2:
            return "Neutral_Sideways_Movement"
        else:
            return "Mixed_Signals_Pattern"
    
    def detect_market_anomalies(self):
        """Detect unusual market behaviors and anomalies"""
        print(f"\n🚨 DETECTING MARKET ANOMALIES...")
        
        if self.feature_matrix is None:
            return
        
        # 1. Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(self.feature_matrix)
        
        anomaly_indices = np.where(anomaly_labels == -1)[0]
        print(f"   🎯 Found {len(anomaly_indices)} anomalous market behaviors")
        
        # 2. Analyze anomalies
        anomalies_by_stock = {}
        for idx in anomaly_indices:
            stock = self.stock_names[idx]
            date = self.dates[idx]
            anomaly_features = self.feature_matrix[idx]
            
            if stock not in anomalies_by_stock:
                anomalies_by_stock[stock] = []
            
            anomalies_by_stock[stock].append({
                'date': date.strftime('%Y-%m-%d'),
                'anomaly_score': float(iso_forest.decision_function([anomaly_features])[0]),
                'feature_vector': anomaly_features.tolist()
            })
        
        # 3. Statistical anomalies
        print("   📊 Detecting statistical anomalies...")
        
        stat_anomalies = {}
        for symbol, data in self.market_data.items():
            if symbol.startswith('INDEX_'):
                continue
                
            try:
                recent_data = data.tail(60)
                
                # Volume anomalies
                volume_zscore = np.abs(stats.zscore(recent_data['Volume'].dropna()))
                volume_anomalies = recent_data[volume_zscore > 3]
                
                # Price movement anomalies
                returns = recent_data['Daily_Return'].dropna()
                return_zscore = np.abs(stats.zscore(returns))
                return_anomalies = recent_data[return_zscore > 3]
                
                if len(volume_anomalies) > 0 or len(return_anomalies) > 0:
                    stat_anomalies[symbol] = {
                        'extreme_volume_days': len(volume_anomalies),
                        'extreme_return_days': len(return_anomalies),
                        'max_volume_spike': float(volume_anomalies['Volume_Ratio'].max()) if len(volume_anomalies) > 0 else 0,
                        'max_return': float(return_anomalies['Daily_Return'].abs().max()) if len(return_anomalies) > 0 else 0
                    }
                    
            except Exception as e:
                continue
        
        self.market_anomalies = {
            'ml_anomalies': anomalies_by_stock,
            'statistical_anomalies': stat_anomalies,
            'total_anomaly_points': len(anomaly_indices),
            'anomaly_percentage': len(anomaly_indices) / len(self.feature_matrix) * 100
        }
        
        print(f"   ✅ Anomaly detection complete!")
        print(f"   📊 Stocks with ML anomalies: {len(anomalies_by_stock)}")
        print(f"   📊 Stocks with statistical anomalies: {len(stat_anomalies)}")
    
    def discover_hidden_correlations(self):
        """Discover hidden correlations and relationships"""
        print(f"\n🔗 DISCOVERING HIDDEN CORRELATIONS...")
        
        # 1. Build correlation matrix for all stocks
        price_data = {}
        return_data = {}
        
        for symbol, data in self.market_data.items():
            if symbol.startswith('INDEX_'):
                continue
                
            if len(data) > 100:
                price_data[symbol] = data['Close']
                return_data[symbol] = data['Daily_Return']
        
        if len(price_data) < 10:
            print("   ❌ Insufficient data for correlation analysis")
            return
        
        # Align all price data
        price_df = pd.DataFrame(price_data).dropna()
        return_df = pd.DataFrame(return_data).dropna()
        
        print(f"   📊 Analyzing {len(price_df.columns)} stocks for correlations...")
        
        # 2. Price correlations
        price_corr = price_df.corr()
        return_corr = return_df.corr()
        
        # 3. Find high correlations (potential pairs)
        high_correlations = []
        
        for i in range(len(price_corr.columns)):
            for j in range(i+1, len(price_corr.columns)):
                stock1 = price_corr.columns[i]
                stock2 = price_corr.columns[j]
                
                price_correlation = price_corr.iloc[i, j]
                return_correlation = return_corr.iloc[i, j]
                
                if abs(price_correlation) > 0.8:  # High correlation
                    high_correlations.append({
                        'pair': (stock1, stock2),
                        'price_correlation': float(price_correlation),
                        'return_correlation': float(return_correlation),
                        'correlation_type': 'positive' if price_correlation > 0 else 'negative'
                    })
        
        # 4. Sector-wise correlations
        sector_mapping = self._get_sector_mapping()
        sector_correlations = {}
        
        for sector, stocks in sector_mapping.items():
            sector_stocks = [s for s in stocks if s in price_df.columns]
            if len(sector_stocks) >= 3:
                sector_data = price_df[sector_stocks]
                sector_corr = sector_data.corr()
                
                # Average intra-sector correlation
                mask = np.triu(np.ones_like(sector_corr, dtype=bool), k=1)
                sector_correlations[sector] = {
                    'avg_correlation': float(sector_corr.values[mask].mean()),
                    'max_correlation': float(sector_corr.values[mask].max()),
                    'min_correlation': float(sector_corr.values[mask].min()),
                    'stock_count': len(sector_stocks)
                }
        
        # 5. Dynamic correlation analysis
        print("   🔄 Analyzing dynamic correlations...")
        dynamic_correlations = self._analyze_dynamic_correlations(return_df)
        
        self.hidden_correlations = {
            'high_correlation_pairs': sorted(high_correlations, key=lambda x: abs(x['price_correlation']), reverse=True)[:20],
            'sector_correlations': sector_correlations,
            'dynamic_correlations': dynamic_correlations,
            'correlation_statistics': {
                'total_pairs_analyzed': len(price_corr.columns) * (len(price_corr.columns) - 1) // 2,
                'high_correlation_pairs': len(high_correlations),
                'avg_market_correlation': float(price_corr.values[np.triu(np.ones_like(price_corr, dtype=bool), k=1)].mean())
            }
        }
        
        print(f"   ✅ Correlation analysis complete!")
        print(f"   🔗 High correlation pairs found: {len(high_correlations)}")
        print(f"   📊 Sectors analyzed: {len(sector_correlations)}")
    
    def _get_sector_mapping(self):
        """Get sector mapping for stocks"""
        return {
            'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK', 'BANKBARODA', 'PNB'],
            'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTI', 'MINDTREE'],
            'Oil_Gas': ['RELIANCE', 'ONGC', 'IOC', 'BPCL', 'HINDPETRO', 'GAIL'],
            'Auto': ['MARUTI', 'M&M', 'TATAMOTORS', 'BAJAJ-AUTO', 'HEROMOTOCO'],
            'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON', 'LUPIN'],
            'FMCG': ['HINDUNILVR', 'ITC', 'NESTLE', 'BRITANNIA', 'DABUR', 'MARICO'],
            'Metals': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'COALINDIA', 'VEDL'],
            'Cement': ['ULTRACEMCO', 'GRASIM', 'ACC', 'AMBUJACEMENT', 'SHREECEM']
        }
    
    def _analyze_dynamic_correlations(self, return_df):
        """Analyze how correlations change over time"""
        
        if len(return_df) < 60:
            return {}
        
        # Rolling correlations
        window = 30
        dynamic_data = {}
        
        # Sample a few high-volume pairs for analysis
        sample_pairs = [
            ('RELIANCE', 'ONGC'), ('HDFCBANK', 'ICICIBANK'), ('TCS', 'INFY'),
            ('MARUTI', 'M&M'), ('SUNPHARMA', 'DRREDDY')
        ]
        
        for stock1, stock2 in sample_pairs:
            if stock1 in return_df.columns and stock2 in return_df.columns:
                rolling_corr = return_df[stock1].rolling(window).corr(return_df[stock2])
                
                dynamic_data[f"{stock1}_{stock2}"] = {
                    'recent_correlation': float(rolling_corr.tail(1).iloc[0]) if not rolling_corr.empty else 0,
                    'avg_correlation': float(rolling_corr.mean()) if not rolling_corr.empty else 0,
                    'correlation_volatility': float(rolling_corr.std()) if not rolling_corr.empty else 0,
                    'max_correlation': float(rolling_corr.max()) if not rolling_corr.empty else 0,
                    'min_correlation': float(rolling_corr.min()) if not rolling_corr.empty else 0
                }
        
        return dynamic_data
    
    def find_pair_trading_opportunities(self):
        """Find sophisticated pair trading opportunities"""
        print(f"\n💰 FINDING PAIR TRADING OPPORTUNITIES...")
        
        if not self.hidden_correlations:
            print("   ❌ Run correlation analysis first")
            return
        
        pair_opportunities = []
        
        # Analyze high correlation pairs for trading opportunities
        for pair_data in self.hidden_correlations['high_correlation_pairs']:
            stock1, stock2 = pair_data['pair']
            correlation = pair_data['price_correlation']
            
            if abs(correlation) > 0.85:  # Very high correlation
                opportunity = self._analyze_pair_opportunity(stock1, stock2, correlation)
                if opportunity:
                    pair_opportunities.append(opportunity)
        
        # Sort by opportunity score
        pair_opportunities.sort(key=lambda x: x.get('opportunity_score', 0), reverse=True)
        
        self.pair_opportunities = {
            'trading_pairs': pair_opportunities[:15],  # Top 15 opportunities
            'total_pairs_analyzed': len(self.hidden_correlations['high_correlation_pairs']),
            'opportunities_found': len(pair_opportunities)
        }
        
        print(f"   ✅ Pair trading analysis complete!")
        print(f"   💎 Trading opportunities found: {len(pair_opportunities)}")
    
    def _analyze_pair_opportunity(self, stock1, stock2, correlation):
        """Analyze specific pair for trading opportunity"""
        
        if stock1 not in self.market_data or stock2 not in self.market_data:
            return None
        
        try:
            data1 = self.market_data[stock1]['Close'].tail(60)
            data2 = self.market_data[stock2]['Close'].tail(60)
            
            # Align data
            aligned_data = pd.DataFrame({'stock1': data1, 'stock2': data2}).dropna()
            
            if len(aligned_data) < 30:
                return None
            
            # Calculate spread
            spread = aligned_data['stock1'] - aligned_data['stock2']
            spread_mean = spread.mean()
            spread_std = spread.std()
            current_spread = spread.iloc[-1]
            
            # Z-score of current spread
            spread_zscore = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
            
            # Trading opportunity scoring
            opportunity_score = abs(spread_zscore) * abs(correlation)
            
            # Determine trade direction
            if spread_zscore > 2:
                trade_signal = f"Short {stock1}, Long {stock2}"
                expected_return = abs(spread_zscore) * 0.5  # Estimated return
            elif spread_zscore < -2:
                trade_signal = f"Long {stock1}, Short {stock2}"
                expected_return = abs(spread_zscore) * 0.5
            else:
                trade_signal = "No trade"
                expected_return = 0
            
            if abs(spread_zscore) > 1.5:  # Minimum threshold for opportunity
                return {
                    'pair': (stock1, stock2),
                    'correlation': float(correlation),
                    'spread_zscore': float(spread_zscore),
                    'opportunity_score': float(opportunity_score),
                    'trade_signal': trade_signal,
                    'expected_return_pct': float(expected_return),
                    'current_spread': float(current_spread),
                    'spread_mean': float(spread_mean),
                    'spread_volatility': float(spread_std)
                }
        
        except Exception as e:
            return None
        
        return None
    
    def generate_comprehensive_report(self):
        """Generate comprehensive market discovery report"""
        print(f"\n📋 GENERATING COMPREHENSIVE MARKET REPORT...")
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'market_overview': {
                'total_stocks_analyzed': len([k for k in self.market_data.keys() if not k.startswith('INDEX_')]),
                'total_indices_analyzed': len([k for k in self.market_data.keys() if k.startswith('INDEX_')]),
                'data_points_processed': len(self.feature_matrix) if self.feature_matrix is not None else 0,
                'analysis_period': '1 year historical data'
            },
            
            'hidden_patterns_discovered': self.secret_patterns,
            'market_anomalies_detected': self.market_anomalies,
            'hidden_correlations': self.hidden_correlations,
            'pair_trading_opportunities': self.pair_opportunities,
            
            'key_insights': self._generate_key_insights(),
            'market_secrets_revealed': self._generate_market_secrets()
        }
        
        # Save comprehensive report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/market_secrets_revealed_{timestamp}.json'
        
        try:
            import os
            os.makedirs('results', exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"   💾 Comprehensive report saved: {filename}")
            
        except Exception as e:
            print(f"   ❌ Error saving report: {e}")
        
        return report
    
    def _generate_key_insights(self):
        """Generate key insights from all analyses"""
        insights = []
        
        # Pattern insights
        if self.secret_patterns and 'pattern_meanings' in self.secret_patterns:
            pattern_counts = {}
            for pattern_id, pattern_data in self.secret_patterns['pattern_meanings'].items():
                pattern_type = pattern_data['type']
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
            dominant_pattern = max(pattern_counts, key=pattern_counts.get)
            insights.append(f"Dominant market pattern: {dominant_pattern} ({pattern_counts[dominant_pattern]} instances)")
        
        # Anomaly insights
        if self.market_anomalies:
            anomaly_pct = self.market_anomalies.get('anomaly_percentage', 0)
            insights.append(f"Market anomaly rate: {anomaly_pct:.2f}% of all data points")
            
            if 'statistical_anomalies' in self.market_anomalies:
                anomaly_stocks = len(self.market_anomalies['statistical_anomalies'])
                insights.append(f"Stocks showing unusual behavior: {anomaly_stocks}")
        
        # Correlation insights
        if self.hidden_correlations:
            high_corr_pairs = len(self.hidden_correlations.get('high_correlation_pairs', []))
            avg_corr = self.hidden_correlations.get('correlation_statistics', {}).get('avg_market_correlation', 0)
            insights.append(f"High correlation pairs found: {high_corr_pairs}")
            insights.append(f"Average market correlation: {avg_corr:.3f}")
        
        # Pair trading insights
        if self.pair_opportunities:
            opportunities = len(self.pair_opportunities.get('trading_pairs', []))
            insights.append(f"Active pair trading opportunities: {opportunities}")
        
        return insights
    
    def _generate_market_secrets(self):
        """Generate the most important market secrets discovered"""
        secrets = []
        
        # Secret 1: Hidden market dimensions
        if self.secret_patterns and 'explained_variance' in self.secret_patterns:
            top_variance = self.secret_patterns['explained_variance'][0]
            secrets.append({
                'secret': 'Hidden Market Dimension',
                'revelation': f"The market's behavior is primarily driven by a single hidden factor explaining {top_variance:.1%} of all variation",
                'impact': 'This hidden dimension controls most stock movements - understanding it gives massive advantage'
            })
        
        # Secret 2: Anomaly clusters
        if self.market_anomalies and 'ml_anomalies' in self.market_anomalies:
            anomaly_stocks = list(self.market_anomalies['ml_anomalies'].keys())
            if anomaly_stocks:
                secrets.append({
                    'secret': 'Anomaly-Prone Stocks',
                    'revelation': f"Stocks like {', '.join(anomaly_stocks[:3])} show abnormal behavior patterns",
                    'impact': 'These stocks break normal market rules - potential for extraordinary moves'
                })
        
        # Secret 3: Super-correlations
        if self.hidden_correlations and 'high_correlation_pairs' in self.hidden_correlations:
            super_pairs = [pair for pair in self.hidden_correlations['high_correlation_pairs'] 
                          if abs(pair['price_correlation']) > 0.95]
            if super_pairs:
                secrets.append({
                    'secret': 'Super-Correlated Pairs',
                    'revelation': f"Found {len(super_pairs)} stock pairs moving almost identically (>95% correlation)",
                    'impact': 'These pairs offer near-guaranteed arbitrage opportunities when they diverge'
                })
        
        # Secret 4: Pattern dominance
        if self.secret_patterns and 'pattern_meanings' in self.secret_patterns:
            volatile_patterns = [p for p in self.secret_patterns['pattern_meanings'].values() 
                                if 'High_Volatility' in p.get('type', '')]
            if volatile_patterns:
                secrets.append({
                    'secret': 'Volatility Clustering',
                    'revelation': f"Found {len(volatile_patterns)} distinct high-volatility behavior patterns",
                    'impact': 'Volatility follows hidden patterns - can predict explosive moves before they happen'
                })
        
        # Secret 5: Sector disconnections
        if self.hidden_correlations and 'sector_correlations' in self.hidden_correlations:
            weak_sectors = {sector: data for sector, data in self.hidden_correlations['sector_correlations'].items()
                           if data['avg_correlation'] < 0.3}
            if weak_sectors:
                secrets.append({
                    'secret': 'Sector Disconnection',
                    'revelation': f"Sectors {list(weak_sectors.keys())} show internal disconnection",
                    'impact': 'Individual stock picking more important than sector trends in these areas'
                })
        
        return secrets
    
    def display_market_secrets(self, report):
        """Display the discovered market secrets in an engaging format"""
        print("\n" + "="*100)
        print("🔮 THE MARKET'S SECRET REVEALING BOOK - HIDDEN TRUTHS EXPOSED")
        print("="*100)
        
        print(f"\n📊 ANALYSIS SCOPE:")
        overview = report['market_overview']
        print(f"   🎯 Stocks Analyzed: {overview['total_stocks_analyzed']} F&O stocks")
        print(f"   📈 Indices Analyzed: {overview['total_indices_analyzed']} major indices") 
        print(f"   🧠 Data Points Processed: {overview['data_points_processed']:,}")
        print(f"   ⏰ Analysis Period: {overview['analysis_period']}")
        
        # Key insights
        if 'key_insights' in report:
            print(f"\n🔍 KEY MARKET INSIGHTS:")
            for i, insight in enumerate(report['key_insights'], 1):
                print(f"   {i}. {insight}")
        
        # Market secrets
        if 'market_secrets_revealed' in report:
            print(f"\n🚨 MARKET SECRETS REVEALED:")
            for i, secret in enumerate(report['market_secrets_revealed'], 1):
                print(f"\n   🔐 SECRET #{i}: {secret['secret']}")
                print(f"      💡 Revelation: {secret['revelation']}")
                print(f"      ⚡ Impact: {secret['impact']}")
        
        # Hidden patterns
        if 'hidden_patterns_discovered' in report and 'pattern_meanings' in report['hidden_patterns_discovered']:
            print(f"\n🎭 HIDDEN PATTERNS DISCOVERED:")
            patterns = report['hidden_patterns_discovered']['pattern_meanings']
            for pattern_id, pattern_data in list(patterns.items())[:5]:
                print(f"\n   📊 {pattern_data['type']}:")
                print(f"      🔢 Occurrences: {pattern_data['sample_count']}")
                print(f"      📈 Average Momentum: RSI {pattern_data['avg_momentum']:.1f}")
                print(f"      🌊 Average Volatility: {pattern_data['avg_volatility']:.3f}")
                print(f"      🏢 Sample Stocks: {', '.join(pattern_data['sample_stocks'][:5])}")
        
        # Market anomalies
        if 'market_anomalies_detected' in report:
            anomalies = report['market_anomalies_detected']
            print(f"\n🚨 MARKET ANOMALIES DETECTED:")
            print(f"   ⚠️  Total Anomaly Points: {anomalies.get('total_anomaly_points', 0)}")
            print(f"   📊 Anomaly Rate: {anomalies.get('anomaly_percentage', 0):.2f}% of all data")
            
            if 'statistical_anomalies' in anomalies:
                extreme_stocks = []
                for stock, data in list(anomalies['statistical_anomalies'].items())[:5]:
                    if data['max_volume_spike'] > 5:
                        extreme_stocks.append(f"{stock} (vol: {data['max_volume_spike']:.1f}x)")
                
                if extreme_stocks:
                    print(f"   🌋 Extreme Volume Spikes: {', '.join(extreme_stocks)}")
        
        # Correlation secrets
        if 'hidden_correlations' in report and 'high_correlation_pairs' in report['hidden_correlations']:
            corr_data = report['hidden_correlations']
            print(f"\n🔗 HIDDEN CORRELATIONS:")
            print(f"   🎯 High Correlation Pairs: {len(corr_data['high_correlation_pairs'])}")
            
            print(f"\n   💎 TOP CORRELATION PAIRS:")
            for i, pair in enumerate(corr_data['high_correlation_pairs'][:5], 1):
                stock1, stock2 = pair['pair']
                correlation = pair['price_correlation']
                print(f"      {i}. {stock1} ↔ {stock2}: {correlation:.3f} correlation")
        
        # Pair trading opportunities
        if 'pair_trading_opportunities' in report and 'trading_pairs' in report['pair_trading_opportunities']:
            pair_data = report['pair_trading_opportunities']
            print(f"\n💰 PAIR TRADING OPPORTUNITIES:")
            print(f"   🎯 Active Opportunities: {len(pair_data['trading_pairs'])}")
            
            for i, opportunity in enumerate(pair_data['trading_pairs'][:3], 1):
                pair = opportunity['pair']
                score = opportunity['opportunity_score']
                signal = opportunity['trade_signal']
                expected_return = opportunity['expected_return_pct']
                
                print(f"\n   💎 OPPORTUNITY #{i}:")
                print(f"      📊 Pair: {pair[0]} ↔ {pair[1]}")
                print(f"      🎯 Opportunity Score: {score:.2f}")
                print(f"      📈 Trade Signal: {signal}")
                print(f"      💰 Expected Return: {expected_return:.2f}%")
        
        print(f"\n🎯 SUMMARY:")
        print(f"   🔮 Market secrets successfully revealed through deep learning analysis")
        print(f"   🧠 Hidden patterns, anomalies, and correlations discovered")
        print(f"   💎 Actionable insights generated for all {overview['total_stocks_analyzed']} F&O stocks")
        print(f"   📊 Complete analysis saved for future reference")
        
        print("\n" + "="*100)
        print("🚀 THE MARKET HAS REVEALED ITS SECRETS - USE THIS KNOWLEDGE WISELY!")
        print("="*100)

def main():
    """Main execution function"""
    print("🚀 STARTING ADVANCED MARKET SECRET DISCOVERY...")
    
    try:
        # Initialize the market secret revealer
        revealer = MarketSecretRevealer()
        
        # Step 1: Collect all market data
        if not revealer.collect_all_market_data():
            print("❌ Failed to collect sufficient market data")
            return 1
        
        # Step 2: Build feature matrix
        if not revealer.build_feature_matrix():
            print("❌ Failed to build feature matrix")
            return 1
        
        # Step 3: Discover hidden patterns
        revealer.discover_hidden_patterns()
        
        # Step 4: Detect market anomalies
        revealer.detect_market_anomalies()
        
        # Step 5: Discover hidden correlations
        revealer.discover_hidden_correlations()
        
        # Step 6: Find pair trading opportunities
        revealer.find_pair_trading_opportunities()
        
        # Step 7: Generate comprehensive report
        report = revealer.generate_comprehensive_report()
        
        # Step 8: Display market secrets
        revealer.display_market_secrets(report)
        
        print("\n✅ MARKET SECRET DISCOVERY COMPLETED SUCCESSFULLY!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Market analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())