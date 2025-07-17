"""
🎯 ROBUST STOCK PREDICTOR & CSV GENERATOR 🎯
Generates Comprehensive Stock Predictions with ML Analysis
Uses Existing Market Data + Handles Missing Values Properly
"""

import numpy as np
import pandas as pd
import yfinance as yf
import ta
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class RobustStockPredictor:
    
    def __init__(self):
        """Initialize with proven F&O stocks list"""
        
        # Core F&O stocks that work reliably
        self.core_stocks = [
            # Banking
            'HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN', 'INDUSINDBK',
            'BANDHANBNK', 'FEDERALBNK', 'IDFCFIRSTB', 'PNB', 'BANKBARODA',
            
            # IT
            'TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTI', 'MINDTREE',
            'COFORGE', 'PERSISTENT', 'LTTS', 'TATAELXSI', 'INTELLECT',
            
            # Oil & Gas
            'RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL',
            
            # Automobiles
            'MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO', 'TVSMOTORS',
            'ESCORTS', 'EICHERMOT', 'MOTHERSUMI', 'BOSCHLTD',
            
            # Pharmaceuticals
            'SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON', 'LUPIN',
            'AUROPHARMA', 'TORNTPHARM', 'GLENMARK',
            
            # FMCG
            'HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'GODREJCP',
            'MARICO', 'COLPAL', 'TATACONSUM',
            
            # Metals
            'TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'SAIL', 'NMDC',
            'COALINDIA', 'JINDALSTEL',
            
            # Infrastructure
            'LT', 'ULTRACEMCO', 'SHREECEM', 'ACC', 'AMBUJACEMNT', 'DLF',
            
            # Power
            'NTPC', 'POWERGRID', 'ADANIPOWER', 'TATAPOWER', 'ADANIGREEN',
            
            # Telecom
            'BHARTIARTL', 'IDEA',
            
            # Financial Services
            'BAJFINANCE', 'BAJAJFINSV', 'SBILIFE', 'HDFCLIFE',
            
            # Others
            'TITAN', 'APOLLOHOSP', 'ASIANPAINT', 'ADANIPORTS'
        ]
        
        self.predictions = []
        self.pair_opportunities = []
        self.market_data = {}
        
    def fetch_stock_data(self, period="6mo"):
        """Fetch reliable stock data"""
        print("📡 Fetching current market data...")
        
        success_count = 0
        
        for symbol in self.core_stocks:
            try:
                ticker = f"{symbol}.NS"
                stock = yf.Ticker(ticker)
                data = stock.history(period=period)
                
                if len(data) >= 30:  # Minimum data requirement
                    self.market_data[symbol] = data
                    success_count += 1
                    print(f"✅ {symbol}: {len(data)} days")
                else:
                    print(f"⚠️ {symbol}: Insufficient data")
                    
            except Exception as e:
                print(f"❌ {symbol}: Failed to fetch")
                
        print(f"\n📊 Successfully loaded {success_count} stocks")
        return success_count
    
    def calculate_safe_features(self, data):
        """Calculate technical features with proper NaN handling"""
        try:
            features = {}
            
            # Basic price features
            features['current_price'] = data['Close'].iloc[-1]
            features['price_1d_change'] = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) >= 2 else 0
            features['price_5d_change'] = ((data['Close'].iloc[-1] - data['Close'].iloc[-6]) / data['Close'].iloc[-6] * 100) if len(data) >= 6 else 0
            features['price_20d_change'] = ((data['Close'].iloc[-1] - data['Close'].iloc[-21]) / data['Close'].iloc[-21] * 100) if len(data) >= 21 else 0
            
            # Moving averages with fallbacks
            try:
                features['sma_5'] = data['Close'].rolling(5).mean().iloc[-1]
                features['sma_10'] = data['Close'].rolling(10).mean().iloc[-1]
                features['sma_20'] = data['Close'].rolling(20).mean().iloc[-1]
            except:
                features['sma_5'] = features['current_price']
                features['sma_10'] = features['current_price']
                features['sma_20'] = features['current_price']
            
            # Price vs moving averages
            features['price_vs_sma5'] = (features['current_price'] - features['sma_5']) / features['sma_5'] * 100
            features['price_vs_sma20'] = (features['current_price'] - features['sma_20']) / features['sma_20'] * 100
            
            # Safe momentum indicators
            try:
                features['rsi'] = ta.momentum.rsi(data['Close'], window=14).iloc[-1]
                if np.isnan(features['rsi']):
                    features['rsi'] = 50  # Neutral default
            except:
                features['rsi'] = 50
                
            try:
                features['macd'] = ta.trend.macd_diff(data['Close']).iloc[-1]
                if np.isnan(features['macd']):
                    features['macd'] = 0
            except:
                features['macd'] = 0
                
            # Volume analysis
            try:
                vol_mean = data['Volume'].rolling(20).mean().iloc[-1]
                features['volume_ratio'] = data['Volume'].iloc[-1] / vol_mean if vol_mean > 0 else 1
                if np.isnan(features['volume_ratio']):
                    features['volume_ratio'] = 1
            except:
                features['volume_ratio'] = 1
                
            # Volatility
            try:
                features['volatility'] = data['Close'].pct_change().rolling(20).std().iloc[-1] * 100
                if np.isnan(features['volatility']):
                    features['volatility'] = 2  # Average volatility
            except:
                features['volatility'] = 2
                
            # Support/Resistance
            try:
                high_20 = data['High'].rolling(20).max().iloc[-1]
                low_20 = data['Low'].rolling(20).min().iloc[-1]
                features['resistance_dist'] = (high_20 - features['current_price']) / features['current_price'] * 100
                features['support_dist'] = (features['current_price'] - low_20) / features['current_price'] * 100
            except:
                features['resistance_dist'] = 5
                features['support_dist'] = 5
                
            return features
            
        except Exception as e:
            print(f"Feature calculation error: {e}")
            return None
    
    def generate_ml_prediction(self, symbol, data):
        """Generate ML-based prediction with proper error handling"""
        try:
            # Prepare training data
            X_data = []
            y_data = []
            
            # Use sliding window approach
            window_size = 10
            for i in range(window_size, len(data) - 5):
                # Get features for window
                window_features = self.calculate_safe_features(data.iloc[i-window_size:i])
                if window_features:
                    feature_values = [v for k, v in window_features.items() if k != 'current_price']
                    # Remove any remaining NaN values
                    feature_values = [0 if np.isnan(x) or np.isinf(x) else x for x in feature_values]
                    
                    if len(feature_values) >= 5:  # Minimum features
                        X_data.append(feature_values)
                        
                        # Target: 5-day forward return
                        future_price = data['Close'].iloc[i + 5]
                        current_price = data['Close'].iloc[i]
                        target_return = (future_price - current_price) / current_price * 100
                        y_data.append(target_return)
            
            if len(X_data) < 10:  # Minimum samples
                return None
                
            X = np.array(X_data)
            y = np.array(y_data)
            
            # Handle any remaining NaN values
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
            
            # Simple train-test split
            split_idx = int(0.8 * len(X))
            X_train = X[:split_idx]
            y_train = y[:split_idx]
            
            # Use only Random Forest (most robust)
            model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            model.fit(X_train, y_train)
            
            # Get current features for prediction
            current_features = self.calculate_safe_features(data)
            if current_features:
                current_values = [v for k, v in current_features.items() if k != 'current_price']
                current_values = [0 if np.isnan(x) or np.isinf(x) else x for x in current_values]
                
                if len(current_values) == X.shape[1]:
                    current_array = np.array([current_values])
                    current_array = imputer.transform(current_array)
                    
                    prediction = model.predict(current_array)[0]
                    return prediction, current_features
                    
            return None
            
        except Exception as e:
            print(f"ML prediction error for {symbol}: {e}")
            return None
    
    def create_prediction_entry(self, symbol, predicted_return, features):
        """Create a comprehensive prediction entry"""
        
        current_price = features['current_price']
        predicted_price = current_price * (1 + predicted_return / 100)
        
        # Determine timeframe
        if abs(predicted_return) > 8:
            timeframe = "1-2 weeks"
        elif abs(predicted_return) > 4:
            timeframe = "2-3 weeks"
        else:
            timeframe = "3-4 weeks"
        
        # Calculate confidence
        confidence = 50  # Base confidence
        if features['rsi'] < 30 or features['rsi'] > 70:
            confidence += 15
        if abs(features['price_vs_sma20']) > 3:
            confidence += 10
        if features['volume_ratio'] > 1.5:
            confidence += 15
        confidence = min(confidence, 95)
        
        # Generate reasoning
        reasoning_parts = []
        
        if features['price_vs_sma20'] > 5:
            reasoning_parts.append("Strong uptrend above SMA20")
        elif features['price_vs_sma20'] < -5:
            reasoning_parts.append("Strong downtrend below SMA20")
        else:
            reasoning_parts.append("Consolidating near SMA20")
            
        if features['rsi'] > 70:
            reasoning_parts.append(f"Overbought RSI ({features['rsi']:.1f})")
        elif features['rsi'] < 30:
            reasoning_parts.append(f"Oversold RSI ({features['rsi']:.1f})")
        else:
            reasoning_parts.append(f"Neutral RSI ({features['rsi']:.1f})")
            
        if features['volume_ratio'] > 1.5:
            reasoning_parts.append("Above average volume")
        elif features['volume_ratio'] < 0.8:
            reasoning_parts.append("Below average volume")
            
        if features['macd'] > 0:
            reasoning_parts.append("Bullish MACD")
        else:
            reasoning_parts.append("Bearish MACD")
            
        # Market classification
        def get_sector(symbol):
            sectors = {
                'Banking': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN', 'INDUSINDBK'],
                'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTI', 'MINDTREE'],
                'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON', 'LUPIN'],
                'Auto': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO'],
                'Metals': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'SAIL'],
                'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR']
            }
            
            for sector, stocks in sectors.items():
                if symbol in stocks:
                    return sector
            return 'Others'
        
        def get_market_cap(symbol):
            large_caps = ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'ITC', 'KOTAKBANK']
            if symbol in large_caps:
                return 'Large Cap'
            elif symbol in self.core_stocks[:40]:
                return 'Mid Cap'
            else:
                return 'Small Cap'
        
        return {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'percentage_change': round(predicted_return, 2),
            'timeframe': timeframe,
            'confidence': confidence,
            'reasoning': "; ".join(reasoning_parts),
            'sector': get_sector(symbol),
            'market_cap': get_market_cap(symbol),
            'rsi': round(features['rsi'], 1),
            'macd': round(features['macd'], 4),
            'volume_ratio': round(features['volume_ratio'], 2),
            'price_vs_sma20': round(features['price_vs_sma20'], 2),
            'volatility': round(features['volatility'], 2),
            'support_distance': round(features['support_dist'], 2),
            'resistance_distance': round(features['resistance_dist'], 2)
        }
    
    def run_analysis(self):
        """Run complete analysis and generate predictions"""
        print("🚀 STARTING ROBUST STOCK PREDICTION ANALYSIS 🚀")
        print("="*60)
        
        # Fetch data
        stocks_loaded = self.fetch_stock_data()
        if stocks_loaded < 5:
            print("❌ Insufficient data. Cannot proceed.")
            return
            
        print(f"\n🤖 Generating ML predictions for {stocks_loaded} stocks...")
        
        successful_predictions = 0
        
        for symbol in self.market_data.keys():
            try:
                data = self.market_data[symbol]
                result = self.generate_ml_prediction(symbol, data)
                
                if result and len(result) == 2:
                    predicted_return, features = result
                    
                    if features and not np.isnan(predicted_return):
                        prediction = self.create_prediction_entry(symbol, predicted_return, features)
                        self.predictions.append(prediction)
                        successful_predictions += 1
                        print(f"✅ {symbol}: {predicted_return:.2f}% in {prediction['timeframe']}")
                    else:
                        print(f"❌ {symbol}: Invalid prediction data")
                else:
                    print(f"❌ {symbol}: Prediction failed")
                    
            except Exception as e:
                print(f"❌ {symbol}: {str(e)}")
                
        print(f"\n📈 Successfully generated {successful_predictions} predictions!")
        
        # Find pair trading opportunities
        self.find_pair_opportunities()
        
        return successful_predictions
    
    def find_pair_opportunities(self):
        """Find simple pair trading opportunities"""
        print("\n🔄 Analyzing pair trading opportunities...")
        
        # Simple correlation-based approach
        symbols = list(self.market_data.keys())
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                try:
                    symbol1, symbol2 = symbols[i], symbols[j]
                    
                    # Get recent price changes
                    data1 = self.market_data[symbol1]
                    data2 = self.market_data[symbol2]
                    
                    returns1 = data1['Close'].pct_change().dropna()
                    returns2 = data2['Close'].pct_change().dropna()
                    
                    min_len = min(len(returns1), len(returns2))
                    if min_len > 30:
                        corr = np.corrcoef(returns1.iloc[-min_len:], returns2.iloc[-min_len:])[0, 1]
                        
                        if abs(corr) > 0.7:  # High correlation
                            pred1 = next((p for p in self.predictions if p['symbol'] == symbol1), None)
                            pred2 = next((p for p in self.predictions if p['symbol'] == symbol2), None)
                            
                            if pred1 and pred2:
                                spread = abs(pred1['percentage_change'] - pred2['percentage_change'])
                                
                                if spread > 3:  # Significant divergence
                                    opportunity = {
                                        'pair': f"{symbol1}-{symbol2}",
                                        'correlation': round(corr, 3),
                                        'long_stock': symbol1 if pred1['percentage_change'] > pred2['percentage_change'] else symbol2,
                                        'short_stock': symbol2 if pred1['percentage_change'] > pred2['percentage_change'] else symbol1,
                                        'spread_opportunity': round(spread, 2),
                                        'risk_level': 'Medium' if abs(corr) > 0.8 else 'High'
                                    }
                                    self.pair_opportunities.append(opportunity)
                                    
                except Exception as e:
                    continue
                    
        print(f"🔍 Found {len(self.pair_opportunities)} pair trading opportunities")
    
    def save_to_csv(self, filename="stock_predictions_comprehensive.csv"):
        """Save all results to CSV files"""
        print(f"\n💾 Saving comprehensive analysis to CSV files...")
        
        if not self.predictions:
            print("❌ No predictions to save")
            return
            
        # Create main predictions DataFrame
        df = pd.DataFrame(self.predictions)
        
        # Sort by predicted return (highest first)
        df = df.sort_values('percentage_change', ascending=False)
        
        # Save main predictions
        df.to_csv(filename, index=False)
        print(f"✅ Saved {len(df)} predictions to {filename}")
        
        # Save pair trading opportunities
        if self.pair_opportunities:
            pair_df = pd.DataFrame(self.pair_opportunities)
            pair_filename = filename.replace('.csv', '_pair_trading.csv')
            pair_df.to_csv(pair_filename, index=False)
            print(f"✅ Saved {len(pair_df)} pair opportunities to {pair_filename}")
        
        # Create summary report
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_predictions': len(df),
            'bullish_predictions': len(df[df['percentage_change'] > 0]),
            'bearish_predictions': len(df[df['percentage_change'] < 0]),
            'high_confidence_predictions': len(df[df['confidence'] > 80]),
            'average_predicted_return': df['percentage_change'].mean(),
            'max_predicted_return': df['percentage_change'].max(),
            'min_predicted_return': df['percentage_change'].min(),
            'sectors_analyzed': df['sector'].nunique(),
            'pair_trading_opportunities': len(self.pair_opportunities)
        }
        
        summary_file = filename.replace('.csv', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Saved summary to {summary_file}")
        
        # Print summary
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"📈 {summary['bullish_predictions']} bullish predictions")
        print(f"📉 {summary['bearish_predictions']} bearish predictions")
        print(f"🎯 {summary['high_confidence_predictions']} high-confidence predictions")
        print(f"📊 Average predicted return: {summary['average_predicted_return']:.2f}%")
        print(f"🔄 {summary['pair_trading_opportunities']} pair trading opportunities")
        
        return filename

def main():
    """Main execution function"""
    predictor = RobustStockPredictor()
    
    # Run analysis
    predictions_count = predictor.run_analysis()
    
    if predictions_count > 0:
        # Save results
        filename = predictor.save_to_csv()
        print(f"\n🎉 ANALYSIS COMPLETE! Results saved to {filename}")
    else:
        print("❌ No predictions generated")

if __name__ == "__main__":
    main()