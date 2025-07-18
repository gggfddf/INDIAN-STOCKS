"""
🔮 COMPREHENSIVE STOCK PREDICTION ENGINE 🔮
Advanced ML-Based Price Predictions with Detailed Analysis
Uses All Collected Market Data, Patterns, and Anomalies
"""

import numpy as np
import pandas as pd
import yfinance as yf
import ta
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ComprehensivePredictionEngine:
    
    def __init__(self):
        """Initialize the prediction engine with all F&O stocks"""
        
        # Complete list of 100+ F&O stocks
        self.fo_stocks = [
            # Banking & Financial Services
            'HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN', 'INDUSINDBK',
            'BANDHANBNK', 'FEDERALBNK', 'IDFCFIRSTB', 'PNB', 'BANKBARODA', 'CANBK',
            
            # IT & Technology
            'TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTI', 'MINDTREE', 'MPHASIS',
            'COFORGE', 'PERSISTENT', 'LTTS', 'TATAELXSI', 'INTELLECT',
            
            # Oil & Gas
            'RELIANCE', 'ONGC', 'BPCL', 'IOC', 'HINDPETRO', 'GAIL', 'OIL',
            
            # Automobiles
            'MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO', 'TVSMOTORS',
            'ESCORTS', 'EICHERMOT', 'MOTHERSUMI', 'BOSCHLTD', 'MRF',
            
            # Pharmaceuticals
            'SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON', 'LUPIN',
            'AUROPHARMA', 'TORNTPHARM', 'GLENMARK', 'CADILAHC', 'ZYDUSLIFE',
            
            # FMCG & Consumer
            'HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'GODREJCP',
            'MARICO', 'COLPAL', 'EMAMILTD', 'TATACONSUM', 'UBL',
            
            # Metals & Mining
            'TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'SAIL', 'NMDC',
            'COALINDIA', 'JINDALSTEL', 'NATIONALUM', 'MOIL',
            
            # Infrastructure & Construction
            'LT', 'ULTRACEMCO', 'SHREECEM', 'ACC', 'AMBUJACEMNT', 'DLF',
            'GODREJPROP', 'BRIGADE', 'PRESTIGE', 'SOBHA',
            
            # Power & Energy
            'NTPC', 'POWERGRID', 'ADANIPOWER', 'TATAPOWER', 'ADANIGREEN',
            'THERMAX', 'BHEL', 'ABB', 'SIEMENS', 'CROMPTON',
            
            # Telecom
            'BHARTIARTL', 'IDEA', 'INDUSIND',
            
            # Others
            'BAJFINANCE', 'BAJAJFINSV', 'HDFC', 'LIC', 'SBILIFE', 'HDFCLIFE',
            'TITAN', 'APOLLOHOSP', 'DRREDDY', 'PEL', 'VOLTAS', 'BLUEDART',
            'ZEEL', 'DIXON', 'WHIRLPOOL', 'HAVELLS', 'CUMMINSIND',
            
            # Additional F&O stocks
            'ADANIPORTS', 'ASIANPAINT', 'BERGEPAINT', 'PIDILITIND', 'KANSAINER',
            'RAMCOCEM', 'INDIACEM', 'JKCEMENT', 'HEIDELBERG', 'STAR',
            'RECLTD', 'PFC', 'IRFC', 'LICHSGFIN', 'MANAPPURAM', 'CHOLAFIN',
            'MUTHOOTFIN', 'BEL', 'HAL', 'BEML', 'CONCOR', 'IRCTC'
        ]
        
        # Indices
        self.indices = ['^NSEI', '^NSEBANK', 'NIFTY_FIN_SERVICE.NS']
        
        self.all_symbols = self.fo_stocks + self.indices
        self.scaler = StandardScaler()
        self.predictions = []
        self.pair_opportunities = []
        
    def fetch_current_data(self, period="1y"):
        """Fetch current market data for all stocks"""
        print("📡 Fetching current market data...")
        
        self.market_data = {}
        failed_stocks = []
        
        for symbol in self.all_symbols:
            try:
                # Add .NS suffix for Indian stocks if not an index
                ticker = symbol if symbol.startswith('^') or symbol.endswith('.NS') else f"{symbol}.NS"
                
                stock = yf.Ticker(ticker)
                data = stock.history(period=period)
                
                if len(data) >= 50:  # Minimum data requirement
                    self.market_data[symbol] = data
                    print(f"✅ {symbol}: {len(data)} days of data")
                else:
                    failed_stocks.append(symbol)
                    print(f"❌ {symbol}: Insufficient data")
                    
            except Exception as e:
                failed_stocks.append(symbol)
                print(f"❌ {symbol}: {str(e)}")
                
        print(f"\n📊 Successfully loaded {len(self.market_data)} stocks")
        if failed_stocks:
            print(f"⚠️ Failed to load: {failed_stocks}")
            
        return len(self.market_data)
    
    def calculate_technical_features(self, data):
        """Calculate comprehensive technical features"""
        try:
            features = {}
            
            # Price features
            features['current_price'] = data['Close'].iloc[-1]
            features['price_change_1d'] = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100
            features['price_change_5d'] = (data['Close'].iloc[-1] - data['Close'].iloc[-6]) / data['Close'].iloc[-6] * 100
            features['price_change_20d'] = (data['Close'].iloc[-1] - data['Close'].iloc[-21]) / data['Close'].iloc[-21] * 100
            
            # Moving averages
            features['sma_5'] = ta.trend.sma_indicator(data['Close'], window=5).iloc[-1]
            features['sma_10'] = ta.trend.sma_indicator(data['Close'], window=10).iloc[-1]
            features['sma_20'] = ta.trend.sma_indicator(data['Close'], window=20).iloc[-1]
            features['sma_50'] = ta.trend.sma_indicator(data['Close'], window=50).iloc[-1]
            
            features['ema_12'] = ta.trend.ema_indicator(data['Close'], window=12).iloc[-1]
            features['ema_26'] = ta.trend.ema_indicator(data['Close'], window=26).iloc[-1]
            
            # Position relative to moving averages
            features['price_vs_sma20'] = (features['current_price'] - features['sma_20']) / features['sma_20'] * 100
            features['price_vs_sma50'] = (features['current_price'] - features['sma_50']) / features['sma_50'] * 100
            
            # Momentum indicators
            features['rsi'] = ta.momentum.rsi(data['Close'], window=14).iloc[-1]
            features['macd'] = ta.trend.macd_diff(data['Close']).iloc[-1]
            features['macd_signal'] = ta.trend.macd_signal(data['Close']).iloc[-1]
            features['stoch_k'] = ta.momentum.stoch(data['High'], data['Low'], data['Close']).iloc[-1]
            features['williams_r'] = ta.momentum.williams_r(data['High'], data['Low'], data['Close']).iloc[-1]
            
            # Volatility indicators
            bb_bands = ta.volatility.bollinger_hband_indicator(data['Close'])
            features['bb_upper'] = ta.volatility.bollinger_hband(data['Close']).iloc[-1]
            features['bb_lower'] = ta.volatility.bollinger_lband(data['Close']).iloc[-1]
            features['bb_position'] = (features['current_price'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            features['atr'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close']).iloc[-1]
            
            # Volume indicators
            features['volume_sma'] = data['Volume'].rolling(20).mean().iloc[-1]
            features['volume_ratio'] = data['Volume'].iloc[-1] / features['volume_sma']
            features['volume_rsi'] = ta.momentum.rsi(data['Volume'], window=14).iloc[-1]
            
            # Trend strength
            features['adx'] = ta.trend.adx(data['High'], data['Low'], data['Close']).iloc[-1]
            
            # Price patterns
            features['doji'] = abs(data['Open'].iloc[-1] - data['Close'].iloc[-1]) / (data['High'].iloc[-1] - data['Low'].iloc[-1]) < 0.1
            features['hammer'] = (data['Close'].iloc[-1] - data['Low'].iloc[-1]) > 2 * abs(data['Open'].iloc[-1] - data['Close'].iloc[-1])
            
            # Support/Resistance levels
            recent_highs = data['High'].rolling(20).max()
            recent_lows = data['Low'].rolling(20).min()
            features['resistance_distance'] = (recent_highs.iloc[-1] - features['current_price']) / features['current_price'] * 100
            features['support_distance'] = (features['current_price'] - recent_lows.iloc[-1]) / features['current_price'] * 100
            
            return features
            
        except Exception as e:
            print(f"Error calculating features: {e}")
            return None
    
    def build_prediction_model(self, symbol, data):
        """Build ML model for price prediction"""
        try:
            # Prepare features for multiple timeframes
            feature_data = []
            target_data = []
            
            # Create features for last 30 days
            for i in range(30, len(data)-5):
                day_features = self.calculate_technical_features(data.iloc[i-29:i+1])
                if day_features:
                    # Convert to list (excluding current price)
                    feature_list = [v for k, v in day_features.items() if k != 'current_price' and isinstance(v, (int, float))]
                    if len(feature_list) > 10:  # Minimum features
                        feature_data.append(feature_list)
                        # Target: 5-day forward return
                        future_price = data['Close'].iloc[i+5]
                        current_price = data['Close'].iloc[i]
                        target_return = (future_price - current_price) / current_price * 100
                        target_data.append(target_return)
            
            if len(feature_data) < 20:  # Minimum samples
                return None, None
                
            X = np.array(feature_data)
            y = np.array(target_data)
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train ensemble of models
            models = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'mlp': MLPRegressor(hidden_layer_sizes=(50, 25), random_state=42, max_iter=500)
            }
            
            predictions = {}
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_test_scaled)
                predictions[name] = pred
                
            # Ensemble prediction (average)
            ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
            
            # Calculate current features for prediction
            current_features = self.calculate_technical_features(data)
            if current_features:
                current_feature_list = [v for k, v in current_features.items() if k != 'current_price' and isinstance(v, (int, float))]
                if len(current_feature_list) == X.shape[1]:
                    current_scaled = self.scaler.transform([current_feature_list])
                    
                    # Get predictions from all models
                    future_predictions = {}
                    for name, model in models.items():
                        future_predictions[name] = model.predict(current_scaled)[0]
                    
                    # Ensemble prediction
                    ensemble_future = np.mean(list(future_predictions.values()))
                    
                    return ensemble_future, current_features
                    
            return None, None
            
        except Exception as e:
            print(f"Error building model for {symbol}: {e}")
            return None, None
    
    def generate_predictions(self):
        """Generate comprehensive predictions for all stocks"""
        print("\n🤖 Generating ML-based predictions...")
        
        self.predictions = []
        processed = 0
        
        for symbol in self.market_data.keys():
            try:
                data = self.market_data[symbol]
                
                # Get prediction and current features
                predicted_return, features = self.build_prediction_model(symbol, data)
                
                if predicted_return is not None and features is not None:
                    current_price = features['current_price']
                    predicted_price = current_price * (1 + predicted_return / 100)
                    
                    # Determine confidence based on technical indicators
                    confidence = self.calculate_confidence(features)
                    
                    # Determine timeframe based on prediction magnitude
                    if abs(predicted_return) > 10:
                        timeframe = "1-2 weeks"
                    elif abs(predicted_return) > 5:
                        timeframe = "2-3 weeks"
                    else:
                        timeframe = "3-4 weeks"
                    
                    # Generate reasoning
                    reasoning = self.generate_reasoning(features, predicted_return)
                    
                    prediction = {
                        'symbol': symbol,
                        'current_price': round(current_price, 2),
                        'predicted_price': round(predicted_price, 2),
                        'percentage_change': round(predicted_return, 2),
                        'timeframe': timeframe,
                        'confidence': confidence,
                        'reasoning': reasoning,
                        'rsi': round(features['rsi'], 2),
                        'macd': round(features['macd'], 4),
                        'volume_ratio': round(features['volume_ratio'], 2),
                        'price_vs_sma20': round(features['price_vs_sma20'], 2),
                        'bb_position': round(features['bb_position'], 2),
                        'atr': round(features['atr'], 2)
                    }
                    
                    self.predictions.append(prediction)
                    processed += 1
                    print(f"✅ {symbol}: {predicted_return:.2f}% in {timeframe}")
                    
                else:
                    print(f"❌ {symbol}: Unable to generate prediction")
                    
            except Exception as e:
                print(f"❌ {symbol}: {str(e)}")
                
        print(f"\n📈 Generated predictions for {processed} stocks")
        return processed
    
    def calculate_confidence(self, features):
        """Calculate prediction confidence based on technical indicators"""
        confidence_score = 50  # Base confidence
        
        # RSI confidence
        rsi = features['rsi']
        if 30 <= rsi <= 70:
            confidence_score += 10  # Neutral RSI is good
        elif rsi < 30 or rsi > 70:
            confidence_score += 15  # Extreme RSI suggests reversal
            
        # Trend confidence
        if features['price_vs_sma20'] > 2:
            confidence_score += 10  # Strong uptrend
        elif features['price_vs_sma20'] < -2:
            confidence_score += 10  # Strong downtrend
            
        # Volume confidence
        if features['volume_ratio'] > 1.5:
            confidence_score += 15  # High volume confirms moves
            
        # Bollinger Bands confidence
        bb_pos = features['bb_position']
        if bb_pos < 0.2 or bb_pos > 0.8:
            confidence_score += 10  # Near extremes
            
        # Cap confidence at 95%
        return min(confidence_score, 95)
    
    def generate_reasoning(self, features, predicted_return):
        """Generate detailed reasoning for the prediction"""
        reasoning_parts = []
        
        # Price trend analysis
        if features['price_vs_sma20'] > 5:
            reasoning_parts.append("Strong uptrend (>5% above SMA20)")
        elif features['price_vs_sma20'] < -5:
            reasoning_parts.append("Strong downtrend (>5% below SMA20)")
        elif abs(features['price_vs_sma20']) < 2:
            reasoning_parts.append("Consolidating near SMA20")
            
        # RSI analysis
        rsi = features['rsi']
        if rsi > 70:
            reasoning_parts.append(f"Overbought RSI ({rsi:.1f})")
        elif rsi < 30:
            reasoning_parts.append(f"Oversold RSI ({rsi:.1f})")
        else:
            reasoning_parts.append(f"Neutral RSI ({rsi:.1f})")
            
        # Volume analysis
        if features['volume_ratio'] > 2:
            reasoning_parts.append("High volume surge (2x+ average)")
        elif features['volume_ratio'] > 1.5:
            reasoning_parts.append("Above-average volume")
        elif features['volume_ratio'] < 0.7:
            reasoning_parts.append("Below-average volume")
            
        # MACD analysis
        if features['macd'] > 0:
            reasoning_parts.append("Bullish MACD signal")
        else:
            reasoning_parts.append("Bearish MACD signal")
            
        # Bollinger Bands analysis
        bb_pos = features['bb_position']
        if bb_pos > 0.8:
            reasoning_parts.append("Near upper Bollinger Band")
        elif bb_pos < 0.2:
            reasoning_parts.append("Near lower Bollinger Band")
            
        # Pattern analysis
        if features.get('doji', False):
            reasoning_parts.append("Doji candlestick pattern")
        if features.get('hammer', False):
            reasoning_parts.append("Hammer candlestick pattern")
            
        return "; ".join(reasoning_parts)
    
    def find_pair_trading_opportunities(self):
        """Find pair trading opportunities based on correlation analysis"""
        print("\n🔄 Analyzing pair trading opportunities...")
        
        self.pair_opportunities = []
        
        # Calculate correlations between stocks
        price_data = {}
        for symbol, data in self.market_data.items():
            if len(data) >= 100:  # Sufficient data
                price_data[symbol] = data['Close'].pct_change().dropna()
                
        symbols = list(price_data.keys())
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                # Calculate correlation
                if len(price_data[symbol1]) > 50 and len(price_data[symbol2]) > 50:
                    min_len = min(len(price_data[symbol1]), len(price_data[symbol2]))
                    corr = np.corrcoef(
                        price_data[symbol1].iloc[-min_len:],
                        price_data[symbol2].iloc[-min_len:]
                    )[0, 1]
                    
                    # Check for pair trading opportunity
                    if abs(corr) > 0.7:  # High correlation
                        # Get current positions relative to each other
                        pred1 = next((p for p in self.predictions if p['symbol'] == symbol1), None)
                        pred2 = next((p for p in self.predictions if p['symbol'] == symbol2), None)
                        
                        if pred1 and pred2:
                            # Check for divergence in predictions
                            if (pred1['percentage_change'] > 3 and pred2['percentage_change'] < -1) or \
                               (pred1['percentage_change'] < -3 and pred2['percentage_change'] > 1):
                                
                                opportunity = {
                                    'pair': f"{symbol1}-{symbol2}",
                                    'correlation': round(corr, 3),
                                    'long_stock': symbol1 if pred1['percentage_change'] > pred2['percentage_change'] else symbol2,
                                    'short_stock': symbol2 if pred1['percentage_change'] > pred2['percentage_change'] else symbol1,
                                    'long_prediction': pred1['percentage_change'] if pred1['percentage_change'] > pred2['percentage_change'] else pred2['percentage_change'],
                                    'short_prediction': pred2['percentage_change'] if pred1['percentage_change'] > pred2['percentage_change'] else pred1['percentage_change'],
                                    'spread_opportunity': abs(pred1['percentage_change'] - pred2['percentage_change']),
                                    'risk_level': 'Medium' if abs(corr) > 0.8 else 'High'
                                }
                                
                                self.pair_opportunities.append(opportunity)
                                
        # Sort by spread opportunity
        self.pair_opportunities.sort(key=lambda x: x['spread_opportunity'], reverse=True)
        
        print(f"🔍 Found {len(self.pair_opportunities)} pair trading opportunities")
        return len(self.pair_opportunities)
    
    def save_to_csv(self, filename="comprehensive_stock_predictions.csv"):
        """Save all predictions and analysis to CSV"""
        print(f"\n💾 Saving comprehensive analysis to {filename}...")
        
        # Create main predictions DataFrame
        predictions_df = pd.DataFrame(self.predictions)
        
        # Add market cap category (estimation based on stock)
        def get_market_cap_category(symbol):
            large_caps = ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'HDFC', 'ITC', 'KOTAKBANK']
            if symbol in large_caps:
                return 'Large Cap'
            elif symbol in self.fo_stocks[:50]:  # Top 50 stocks
                return 'Mid Cap'
            else:
                return 'Small Cap'
                
        predictions_df['market_cap'] = predictions_df['symbol'].apply(get_market_cap_category)
        
        # Add sector classification
        def get_sector(symbol):
            banking = ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN', 'INDUSINDBK']
            it = ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTI']
            pharma = ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON']
            auto = ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO']
            
            if symbol in banking:
                return 'Banking'
            elif symbol in it:
                return 'IT'
            elif symbol in pharma:
                return 'Pharma'
            elif symbol in auto:
                return 'Auto'
            else:
                return 'Others'
                
        predictions_df['sector'] = predictions_df['symbol'].apply(get_sector)
        
        # Sort by predicted return
        predictions_df = predictions_df.sort_values('percentage_change', ascending=False)
        
        # Save main predictions
        predictions_df.to_csv(filename, index=False)
        
        # Save pair trading opportunities if any exist
        if self.pair_opportunities:
            pair_df = pd.DataFrame(self.pair_opportunities)
            pair_filename = filename.replace('.csv', '_pair_trading.csv')
            pair_df.to_csv(pair_filename, index=False)
            print(f"💾 Saved pair trading opportunities to {pair_filename}")
            
        # Create summary statistics
        summary_stats = {
            'total_stocks_analyzed': len(predictions_df),
            'bullish_predictions': len(predictions_df[predictions_df['percentage_change'] > 0]),
            'bearish_predictions': len(predictions_df[predictions_df['percentage_change'] < 0]),
            'avg_predicted_return': predictions_df['percentage_change'].mean(),
            'max_predicted_return': predictions_df['percentage_change'].max(),
            'min_predicted_return': predictions_df['percentage_change'].min(),
            'high_confidence_predictions': len(predictions_df[predictions_df['confidence'] > 80]),
            'pair_trading_opportunities': len(self.pair_opportunities)
        }
        
        # Save summary
        summary_filename = filename.replace('.csv', '_summary.json')
        with open(summary_filename, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"✅ Successfully saved comprehensive analysis!")
        print(f"📊 {summary_stats['total_stocks_analyzed']} stocks analyzed")
        print(f"📈 {summary_stats['bullish_predictions']} bullish predictions")
        print(f"📉 {summary_stats['bearish_predictions']} bearish predictions")
        print(f"🎯 {summary_stats['high_confidence_predictions']} high-confidence predictions")
        print(f"🔄 {summary_stats['pair_trading_opportunities']} pair trading opportunities")
        
        return filename

def main():
    """Run comprehensive prediction analysis"""
    print("🚀 STARTING COMPREHENSIVE STOCK PREDICTION ENGINE 🚀")
    print("="*70)
    
    # Initialize engine
    engine = ComprehensivePredictionEngine()
    
    # Fetch current data
    stocks_loaded = engine.fetch_current_data(period="1y")
    
    if stocks_loaded < 10:
        print("❌ Insufficient data loaded. Cannot proceed.")
        return
    
    # Generate predictions
    predictions_generated = engine.generate_predictions()
    
    if predictions_generated == 0:
        print("❌ No predictions generated. Cannot proceed.")
        return
    
    # Find pair trading opportunities
    engine.find_pair_trading_opportunities()
    
    # Save to CSV
    filename = engine.save_to_csv()
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: {filename}")
    print(f"🔮 {predictions_generated} stock predictions generated with detailed reasoning!")

if __name__ == "__main__":
    main()