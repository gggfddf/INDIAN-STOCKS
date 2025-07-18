"""
🔮 ROBUST MARKET ANALYZER 🔮
DEEP LEARNING SYSTEM FOR INDIAN STOCK MARKET PATTERN DISCOVERY
Analyzes 100+ F&O Stocks for Hidden Patterns, Anomalies & Market Secrets
"""

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

class RobustMarketAnalyzer:
    """🔮 Robust Deep Learning Market Analyzer 🔮"""
    
    def __init__(self):
        # Complete F&O Stock List (100+ stocks)
        self.fo_stocks = [
            # Banking & Financial (20 stocks)
            'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK',
            'BANKBARODA', 'PNB', 'FEDERALBNK', 'IDFCFIRSTB', 'BAJFINANCE', 'BAJAJFINSV',
            'SBILIFE', 'HDFCLIFE', 'ICICIPRULI', 'LICHSGFIN', 'CHOLAFIN', 'PEL', 'HDFCAMC', 'CDSL',
            
            # IT & Technology (12 stocks)
            'TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'MPHASIS', 'LTTS', 
            'COFORGE', 'PERSISTENT', 'OFSS', 'TATAELXSI', 'INTELLECT',
            
            # Oil & Gas (8 stocks)
            'RELIANCE', 'ONGC', 'IOC', 'BPCL', 'HINDPETRO', 'GAIL', 'OIL', 'PETRONET',
            
            # Automobiles (8 stocks)
            'MARUTI', 'M&M', 'TATAMOTORS', 'BAJAJ-AUTO', 'HEROMOTOCO', 'EICHERMOT', 'ASHOKLEY', 'ESCORTS',
            
            # Pharmaceuticals (10 stocks)
            'SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON', 'LUPIN', 
            'TORNTPHARM', 'AUROPHARMA', 'GLENMARK', 'ALKEM',
            
            # FMCG (10 stocks)
            'HINDUNILVR', 'ITC', 'BRITANNIA', 'DABUR', 'MARICO', 'COLPAL',
            'GODREJCP', 'UBL', 'TATACONSUM', 'VBL',
            
            # Metals & Mining (8 stocks)
            'TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'COALINDIA', 'VEDL', 'NMDC', 'SAIL', 'JINDALSTEL',
            
            # Cement (6 stocks)
            'ULTRACEMCO', 'GRASIM', 'ACC', 'SHREECEM', 'RAMCOCEM', 'JKCEMENT',
            
            # Power & Energy (8 stocks)
            'NTPC', 'POWERGRID', 'ADANIGREEN', 'TATAPOWER', 'ADANIPOWER', 'NHPC', 'SJVN', 'THERMAX',
            
            # Telecom & Others (15 stocks)
            'BHARTIARTL', 'IDEA', 'INDIGO', 'LT', 'ADANIPORTS', 'IRB', 'PFC', 'RECLTD',
            'SIEMENS', 'ABB', 'HAVELLS', 'VOLTAS', 'CUMMINSIND', 'BHEL', 'BEL'
        ]
        
        # Indices
        self.indices = {
            'NIFTY50': '^NSEI',
            'BANKNIFTY': '^NSEBANK', 
            'FINNIFTY': '^CNXFIN'
        }
        
        # Sector Classification
        self.sectors = {
            'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK', 'BANKBARODA', 'PNB'],
            'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'MPHASIS', 'LTTS'],
            'Energy': ['RELIANCE', 'ONGC', 'IOC', 'BPCL', 'HINDPETRO', 'GAIL'],
            'Auto': ['MARUTI', 'M&M', 'TATAMOTORS', 'BAJAJ-AUTO', 'HEROMOTOCO'],
            'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON'],
            'FMCG': ['HINDUNILVR', 'ITC', 'BRITANNIA', 'DABUR', 'MARICO'],
            'Metals': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'COALINDIA', 'VEDL'],
            'Cement': ['ULTRACEMCO', 'GRASIM', 'ACC', 'SHREECEM']
        }
        
        # Data storage
        self.raw_data = {}
        self.processed_data = {}
        self.patterns = {}
        self.correlations = {}
        self.anomalies = {}
        self.insights = {}
        
        print("🔮 ROBUST MARKET ANALYZER INITIALIZED 🔮")
        print(f"🎯 Analyzing {len(self.fo_stocks)} F&O Stocks + {len(self.indices)} Indices")
        print(f"🏭 Covering {len(self.sectors)} Major Sectors")
        print("🧠 Deep Learning: Pattern Recognition, Anomaly Detection, Correlation Analysis")
    
    def fetch_market_data(self, period="1y", interval="1d"):
        """Fetch market data for all stocks and indices"""
        print(f"\n🚀 FETCHING DATA FOR {len(self.fo_stocks)} STOCKS...")
        
        successful = []
        failed = []
        
        # Fetch stock data
        for i, symbol in enumerate(self.fo_stocks):
            try:
                # Try .NS suffix first
                ticker = f"{symbol}.NS"
                data = yf.download(ticker, period=period, interval=interval, progress=False)
                
                if data.empty:
                    # Try without suffix
                    data = yf.download(symbol, period=period, interval=interval, progress=False)
                
                if not data.empty and len(data) > 100:
                    # Clean column names - remove multi-level indexing
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.droplevel(1)
                    
                    self.raw_data[symbol] = data
                    successful.append(symbol)
                    
                    if (i + 1) % 20 == 0:
                        print(f"✅ Fetched {i+1}/{len(self.fo_stocks)} stocks")
                else:
                    failed.append(symbol)
                    
            except Exception as e:
                failed.append(symbol)
                continue
            
            time.sleep(0.05)  # Rate limiting
        
        # Fetch index data
        for name, ticker in self.indices.items():
            try:
                data = yf.download(ticker, period=period, interval=interval, progress=False)
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.droplevel(1)
                    self.raw_data[name] = data
                    successful.append(name)
            except:
                failed.append(name)
        
        print(f"✅ Successfully fetched: {len(successful)} securities")
        print(f"❌ Failed: {len(failed)} securities")
        if failed:
            print(f"Failed stocks: {failed[:10]}...")
        
        return len(successful)
    
    def engineer_features(self):
        """Engineer comprehensive features for each stock"""
        print("\n🔧 ENGINEERING FEATURES FOR ALL STOCKS...")
        
        for symbol, data in self.raw_data.items():
            try:
                features = pd.DataFrame(index=data.index)
                
                # Price features
                features['close'] = data['Close']
                features['high'] = data['High']
                features['low'] = data['Low']
                features['open'] = data['Open']
                features['volume'] = data['Volume']
                
                # Basic returns and volatility
                features['returns'] = data['Close'].pct_change()
                features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
                features['volatility'] = features['returns'].rolling(20).std()
                
                # Price ratios
                features['high_low_ratio'] = data['High'] / data['Low']
                features['close_open_ratio'] = data['Close'] / data['Open']
                
                # Moving averages (properly handle Series data)
                close_series = data['Close'].squeeze()  # Ensure Series format
                features['sma_5'] = close_series.rolling(5).mean()
                features['sma_10'] = close_series.rolling(10).mean()
                features['sma_20'] = close_series.rolling(20).mean()
                features['sma_50'] = close_series.rolling(50).mean()
                
                # Price position relative to moving averages
                features['price_sma5_ratio'] = close_series / features['sma_5']
                features['price_sma20_ratio'] = close_series / features['sma_20']
                
                # Volume features
                volume_series = data['Volume'].squeeze()
                features['volume_sma'] = volume_series.rolling(20).mean()
                features['volume_ratio'] = volume_series / features['volume_sma']
                
                # Technical indicators using ta library (with proper Series input)
                try:
                    high_series = data['High'].squeeze()
                    low_series = data['Low'].squeeze()
                    
                    features['rsi'] = ta.momentum.rsi(close_series, window=14)
                    features['macd'] = ta.trend.macd_diff(close_series)
                    features['bb_upper'] = ta.volatility.bollinger_hband(close_series)
                    features['bb_lower'] = ta.volatility.bollinger_lband(close_series)
                    features['bb_position'] = (close_series - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
                    
                    # Stochastic
                    features['stoch_k'] = ta.momentum.stoch(high_series, low_series, close_series)
                    
                except Exception as e:
                    # Fallback calculations if ta library fails
                    print(f"⚠️ TA library failed for {symbol}, using fallback calculations")
                    
                    # Simple RSI calculation
                    delta = close_series.diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    features['rsi'] = 100 - (100 / (1 + rs))
                
                # Pattern detection
                features['doji'] = self.detect_doji(data)
                features['hammer'] = self.detect_hammer(data)
                features['gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
                features['gap_up'] = (features['gap'] > 0.02).astype(int)
                features['gap_down'] = (features['gap'] < -0.02).astype(int)
                
                # Support/Resistance detection
                features['support'] = self.find_support_resistance(close_series, mode='support')
                features['resistance'] = self.find_support_resistance(close_series, mode='resistance')
                
                # Clean and store
                features = features.dropna()
                if len(features) > 50:  # Minimum data requirement
                    self.processed_data[symbol] = features
                    
            except Exception as e:
                print(f"⚠️ Feature engineering failed for {symbol}: {str(e)}")
                continue
        
        print(f"✅ Features engineered for {len(self.processed_data)} securities")
        return len(self.processed_data)
    
    def detect_doji(self, data):
        """Detect Doji candlestick patterns"""
        body_size = abs(data['Close'] - data['Open'])
        candle_range = data['High'] - data['Low']
        return (body_size / candle_range < 0.1).astype(int)
    
    def detect_hammer(self, data):
        """Detect Hammer patterns"""
        body_size = abs(data['Close'] - data['Open'])
        lower_shadow = np.minimum(data['Open'], data['Close']) - data['Low']
        upper_shadow = data['High'] - np.maximum(data['Open'], data['Close'])
        return ((lower_shadow > 2 * body_size) & (upper_shadow < body_size)).astype(int)
    
    def find_support_resistance(self, price_series, window=10, mode='support'):
        """Find support and resistance levels"""
        if mode == 'support':
            extrema_idx = argrelextrema(price_series.values, np.less, order=window)[0]
        else:
            extrema_idx = argrelextrema(price_series.values, np.greater, order=window)[0]
        
        levels = pd.Series(index=price_series.index, dtype=float)
        
        for i, idx in enumerate(extrema_idx):
            if idx < len(price_series):
                levels.iloc[idx] = price_series.iloc[idx]
        
        # Forward fill the levels
        levels = levels.fillna(method='ffill')
        return levels
    
    def discover_patterns(self):
        """Discover hidden patterns using ML techniques"""
        print("\n🔍 DISCOVERING HIDDEN PATTERNS...")
        
        if not self.processed_data:
            print("❌ No processed data available")
            return
        
        # Combine features from all stocks
        all_features = []
        stock_mapping = []
        
        feature_columns = ['returns', 'volatility', 'rsi', 'volume_ratio', 'bb_position', 
                          'price_sma20_ratio', 'doji', 'hammer', 'gap']
        
        for symbol, features in self.processed_data.items():
            # Select available columns
            available_cols = [col for col in feature_columns if col in features.columns]
            if len(available_cols) >= 5:  # Minimum features required
                stock_features = features[available_cols].fillna(0).tail(100)  # Last 100 days
                all_features.append(stock_features.values)
                stock_mapping.extend([symbol] * len(stock_features))
        
        if not all_features:
            print("❌ No suitable features for pattern analysis")
            return
        
        # Create feature matrix
        feature_matrix = np.vstack(all_features)
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # 1. Clustering - Find market regimes
        print("🔍 Clustering Analysis - Finding Market Regimes...")
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(feature_matrix_scaled)
        
        # 2. Anomaly Detection
        print("🔍 Anomaly Detection - Finding Unusual Patterns...")
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = isolation_forest.fit_predict(feature_matrix_scaled)
        
        # 3. Dimensionality Reduction
        print("🔍 Dimensionality Reduction - Finding Hidden Dimensions...")
        pca = PCA(n_components=3)
        pca_components = pca.fit_transform(feature_matrix_scaled)
        
        # 4. Neural Network Pattern Learning
        print("🔍 Neural Network - Learning Complex Patterns...")
        X = feature_matrix_scaled[:-1]
        y = feature_matrix_scaled[1:, 0]  # Predict next period returns
        
        mlp = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=300, random_state=42)
        mlp.fit(X, y)
        
        # Store results
        self.patterns = {
            'clusters': clusters,
            'anomalies': anomalies,
            'pca_components': pca_components,
            'pca_variance_ratio': pca.explained_variance_ratio_,
            'neural_model': mlp,
            'stock_mapping': stock_mapping,
            'feature_importance': np.mean(np.abs(mlp.coefs_[0]), axis=1)
        }
        
        print("✅ Pattern discovery completed!")
        return self.patterns
    
    def analyze_correlations(self):
        """Analyze correlations and find pair trading opportunities"""
        print("\n🔍 ANALYZING CORRELATIONS & PAIR OPPORTUNITIES...")
        
        if not self.processed_data:
            print("❌ No processed data available")
            return
        
        # Create returns matrix
        returns_data = {}
        for symbol, features in self.processed_data.items():
            if 'returns' in features.columns:
                returns_data[symbol] = features['returns'].dropna()
        
        if len(returns_data) < 2:
            print("❌ Insufficient data for correlation analysis")
            return
        
        # Align all return series
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Find high correlation pairs
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.6:  # High correlation threshold
                    high_correlations.append({
                        'stock1': correlation_matrix.columns[i],
                        'stock2': correlation_matrix.columns[j],
                        'correlation': corr,
                        'type': 'positive' if corr > 0 else 'negative'
                    })
        
        # Sector analysis
        sector_performance = {}
        for sector, stocks in self.sectors.items():
            sector_stocks = [s for s in stocks if s in returns_df.columns]
            if len(sector_stocks) >= 2:
                sector_returns = returns_df[sector_stocks].mean(axis=1)
                sector_performance[sector] = {
                    'avg_return': sector_returns.mean(),
                    'volatility': sector_returns.std(),
                    'sharpe_ratio': sector_returns.mean() / sector_returns.std() if sector_returns.std() > 0 else 0,
                    'stocks_count': len(sector_stocks)
                }
        
        # Store results
        self.correlations = {
            'correlation_matrix': correlation_matrix,
            'high_correlations': high_correlations,
            'sector_performance': sector_performance,
            'returns_df': returns_df
        }
        
        print(f"✅ Found {len(high_correlations)} high correlation pairs")
        print(f"✅ Analyzed {len(sector_performance)} sectors")
        return self.correlations
    
    def detect_anomalies(self):
        """Detect market anomalies and unusual behaviors"""
        print("\n🚨 DETECTING MARKET ANOMALIES...")
        
        stock_anomalies = {}
        
        for symbol, features in self.processed_data.items():
            try:
                anomaly_count = 0
                anomaly_details = {}
                
                # Volume anomalies
                if 'volume_ratio' in features.columns:
                    volume_z = np.abs(stats.zscore(features['volume_ratio'].dropna()))
                    volume_anomalies = (volume_z > 3).sum()
                    anomaly_count += volume_anomalies
                    anomaly_details['volume_anomalies'] = int(volume_anomalies)
                
                # Price anomalies
                if 'returns' in features.columns:
                    return_z = np.abs(stats.zscore(features['returns'].dropna()))
                    price_anomalies = (return_z > 3).sum()
                    anomaly_count += price_anomalies
                    anomaly_details['price_anomalies'] = int(price_anomalies)
                
                # Gap anomalies
                if 'gap' in features.columns:
                    gap_anomalies = (np.abs(features['gap']) > 0.05).sum()
                    anomaly_count += gap_anomalies
                    anomaly_details['gap_anomalies'] = int(gap_anomalies)
                
                # RSI extremes
                if 'rsi' in features.columns:
                    rsi_extremes = ((features['rsi'] < 20) | (features['rsi'] > 80)).sum()
                    anomaly_count += rsi_extremes
                    anomaly_details['rsi_extremes'] = int(rsi_extremes)
                
                if anomaly_count > 0:
                    anomaly_details['total_anomalies'] = int(anomaly_count)
                    stock_anomalies[symbol] = anomaly_details
                    
            except Exception as e:
                continue
        
        # Sort by total anomalies
        sorted_anomalies = dict(sorted(stock_anomalies.items(), 
                                     key=lambda x: x[1]['total_anomalies'], 
                                     reverse=True))
        
        self.anomalies = sorted_anomalies
        print(f"✅ Detected anomalies in {len(sorted_anomalies)} stocks")
        return sorted_anomalies
    
    def generate_comprehensive_report(self):
        """Generate comprehensive market analysis report"""
        print("\n📊 GENERATING COMPREHENSIVE REPORT...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare insights
        top_anomalies = list(self.anomalies.keys())[:10] if self.anomalies else []
        top_correlations = self.correlations.get('high_correlations', [])[:10] if self.correlations else []
        sector_ranking = []
        
        if self.correlations and 'sector_performance' in self.correlations:
            sector_ranking = sorted(self.correlations['sector_performance'].items(), 
                                  key=lambda x: x[1]['sharpe_ratio'], reverse=True)
        
        # Create comprehensive report
        report = {
            'timestamp': timestamp,
            'analysis_summary': {
                'total_stocks_analyzed': len(self.processed_data),
                'data_period': '1 year',
                'features_engineered': len(self.processed_data),
                'patterns_discovered': len(self.patterns) if self.patterns else 0,
                'correlations_found': len(top_correlations),
                'anomalies_detected': len(self.anomalies),
                'sectors_analyzed': len(sector_ranking)
            },
            'key_discoveries': {
                'market_regimes': f"{len(set(self.patterns['clusters']))} distinct market regimes identified" if self.patterns else "No patterns analyzed",
                'anomaly_stocks': top_anomalies,
                'high_correlation_pairs': [f"{pair['stock1']}-{pair['stock2']} ({pair['correlation']:.3f})" for pair in top_correlations],
                'sector_leaders': [sector[0] for sector in sector_ranking[:3]],
                'pattern_insights': "Deep learning patterns discovered" if self.patterns else "No patterns discovered"
            },
            'detailed_analysis': {
                'patterns': self.patterns,
                'correlations': self.correlations,
                'anomalies': self.anomalies,
                'sector_performance': dict(sector_ranking) if sector_ranking else {}
            }
        }
        
        # Save report
        os.makedirs('market_analysis', exist_ok=True)
        report_file = f'market_analysis/comprehensive_report_{timestamp}.json'
        
        # Convert numpy objects for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return str(obj)
        
        serializable_report = convert_for_json(report)
        
        with open(report_file, 'w') as f:
            json.dump(serializable_report, f, indent=2, default=str)
        
        print(f"✅ Comprehensive report saved: {report_file}")
        
        # Print executive summary
        self.print_executive_summary(report)
        
        return report
    
    def print_executive_summary(self, report):
        """Print executive summary"""
        print("\n" + "="*70)
        print("🔮 ROBUST MARKET ANALYZER - EXECUTIVE SUMMARY 🔮")
        print("="*70)
        
        summary = report['analysis_summary']
        discoveries = report['key_discoveries']
        
        print(f"📊 ANALYSIS SCOPE:")
        print(f"   • Total Stocks Analyzed: {summary['total_stocks_analyzed']}")
        print(f"   • Features Engineered: {summary['features_engineered']}")
        print(f"   • Patterns Discovered: {summary['patterns_discovered']}")
        print(f"   • Correlations Found: {summary['correlations_found']}")
        print(f"   • Anomalies Detected: {summary['anomalies_detected']}")
        print(f"   • Sectors Analyzed: {summary['sectors_analyzed']}")
        
        print(f"\n🏆 KEY DISCOVERIES:")
        print(f"   🧠 Market Regimes: {discoveries['market_regimes']}")
        print(f"   ⚠️ Top Anomaly Stocks: {', '.join(discoveries['anomaly_stocks'][:5])}")
        print(f"   🔗 High Correlation Pairs: {len(discoveries['high_correlation_pairs'])} pairs found")
        print(f"   🏭 Sector Leaders: {', '.join(discoveries['sector_leaders'])}")
        print(f"   🔍 Pattern Insights: {discoveries['pattern_insights']}")
        
        if self.anomalies:
            print(f"\n⚠️ TOP ANOMALY STOCKS:")
            for i, (stock, data) in enumerate(list(self.anomalies.items())[:5], 1):
                print(f"   {i}. {stock}: {data['total_anomalies']} anomalies")
        
        if self.correlations and 'high_correlations' in self.correlations:
            print(f"\n🔗 HIGH CORRELATION PAIRS:")
            for i, pair in enumerate(self.correlations['high_correlations'][:5], 1):
                print(f"   {i}. {pair['stock1']} - {pair['stock2']}: {pair['correlation']:.3f}")
        
        if self.correlations and 'sector_performance' in self.correlations:
            print(f"\n🏭 SECTOR PERFORMANCE RANKING:")
            sector_ranking = sorted(self.correlations['sector_performance'].items(), 
                                  key=lambda x: x[1]['sharpe_ratio'], reverse=True)
            for i, (sector, perf) in enumerate(sector_ranking[:5], 1):
                print(f"   {i}. {sector}: Sharpe {perf['sharpe_ratio']:.3f} ({perf['stocks_count']} stocks)")
        
        print("\n" + "="*70)
        print("🔮 MARKET ANALYSIS COMPLETE - SECRETS REVEALED! 🔮")
        print("="*70)

def main():
    """Main execution function"""
    print("🚀 STARTING ROBUST MARKET ANALYZER...")
    
    analyzer = RobustMarketAnalyzer()
    
    try:
        # Phase 1: Data Collection
        print("\n" + "="*50)
        print("PHASE 1: DATA COLLECTION")
        print("="*50)
        stocks_fetched = analyzer.fetch_market_data(period="1y")
        
        if stocks_fetched < 20:
            print("❌ Insufficient data fetched. Exiting...")
            return
        
        # Phase 2: Feature Engineering
        print("\n" + "="*50)
        print("PHASE 2: FEATURE ENGINEERING")
        print("="*50)
        features_created = analyzer.engineer_features()
        
        if features_created < 10:
            print("❌ Insufficient features created. Exiting...")
            return
        
        # Phase 3: Pattern Discovery
        print("\n" + "="*50)
        print("PHASE 3: PATTERN DISCOVERY")
        print("="*50)
        analyzer.discover_patterns()
        
        # Phase 4: Correlation Analysis
        print("\n" + "="*50)
        print("PHASE 4: CORRELATION ANALYSIS")
        print("="*50)
        analyzer.analyze_correlations()
        
        # Phase 5: Anomaly Detection
        print("\n" + "="*50)
        print("PHASE 5: ANOMALY DETECTION")
        print("="*50)
        analyzer.detect_anomalies()
        
        # Phase 6: Report Generation
        print("\n" + "="*50)
        print("PHASE 6: REPORT GENERATION")
        print("="*50)
        analyzer.generate_comprehensive_report()
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()