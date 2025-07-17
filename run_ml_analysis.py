"""
Simplified Indian Stock Market ML Analysis
Focuses on core ML functionality with lightweight dependencies
"""

import sys
import os
import json
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

# Basic imports for ML analysis
try:
    import yfinance as yf
    import ta
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from loguru import logger
    HAS_IMPORTS = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    HAS_IMPORTS = False

class SimplifiedStockAnalysis:
    """Simplified stock market analysis with core ML functionality"""
    
    def __init__(self):
        # Top 20 F&O stocks for demo
        self.stocks = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 
            'KOTAKBANK', 'SBIN', 'BHARTIARTL', 'ITC', 'LT',
            'AXISBANK', 'MARUTI', 'SUNPHARMA', 'ULTRACEMCO', 'WIPRO',
            'TATAMOTORS', 'M&M', 'BAJFINANCE', 'POWERGRID', 'NTPC'
        ]
        
        # Major indices
        self.indices = {
            'NIFTY50': '^NSEI',
            'BANKNIFTY': '^NSEBANK'
        }
        
        self.stock_data = {}
        self.predictions = {}
        self.signals = []
        
        print("🚀 Simplified Indian Stock Market ML Analysis System Initialized")
    
    def collect_data(self):
        """Collect stock data"""
        print("\n📊 Collecting Market Data...")
        
        successful_fetches = 0
        
        # Collect stock data (with .NS suffix for Yahoo Finance)
        for i, stock in enumerate(self.stocks[:10], 1):  # Limit to 10 stocks for demo
            try:
                symbol = f"{stock}.NS"
                print(f"  Fetching {stock} ({i}/10)...")
                
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="6mo")  # 6 months data
                
                if not data.empty and len(data) > 50:
                    # Add technical indicators
                    data = self._add_technical_indicators(data)
                    self.stock_data[stock] = data
                    successful_fetches += 1
                    print(f"    ✅ Success: {len(data)} days of data")
                else:
                    print(f"    ❌ Failed: Insufficient data")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
        
        # Collect index data
        for index_name, index_symbol in self.indices.items():
            try:
                print(f"  Fetching {index_name}...")
                ticker = yf.Ticker(index_symbol)
                data = ticker.history(period="6mo")
                
                if not data.empty:
                    data = self._add_technical_indicators(data)
                    self.stock_data[f"INDEX_{index_name}"] = data
                    print(f"    ✅ Success: {len(data)} days of data")
                
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
        
        print(f"\n📈 Data Collection Complete: {successful_fetches} stocks + {len(self.indices)} indices")
        return successful_fetches > 0
    
    def _add_technical_indicators(self, df):
        """Add basic technical indicators"""
        try:
            # Moving averages
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
            df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
            
            # Momentum indicators
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            df['MACD'] = ta.trend.macd_diff(df['Close'])
            
            # Volatility indicators
            bb = ta.volatility.BollingerBands(df['Close'])
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Price features
            df['Daily_Return'] = df['Close'].pct_change()
            df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
            df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
            
            return df
        except Exception as e:
            print(f"    Warning: Error adding indicators: {e}")
            return df
    
    def detect_patterns(self):
        """Detect basic trading patterns"""
        print("\n🔍 Detecting Trading Patterns...")
        
        patterns_found = {}
        
        for stock, df in self.stock_data.items():
            if stock.startswith('INDEX_'):
                continue
                
            stock_patterns = []
            
            try:
                current_price = df['Close'].iloc[-1]
                sma_20 = df['SMA_20'].iloc[-1]
                sma_50 = df['SMA_50'].iloc[-1]
                rsi = df['RSI'].iloc[-1]
                volume_ratio = df['Volume_Ratio'].iloc[-1]
                
                # Pattern 1: Golden Cross
                if (df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] and 
                    df['SMA_20'].iloc[-2] <= df['SMA_50'].iloc[-2]):
                    stock_patterns.append({
                        'type': 'golden_cross',
                        'signal': 'bullish',
                        'confidence': 0.8,
                        'description': 'SMA 20 crossed above SMA 50'
                    })
                
                # Pattern 2: RSI Oversold Bounce
                if rsi < 30 and df['RSI'].iloc[-2] < rsi:
                    stock_patterns.append({
                        'type': 'rsi_oversold_bounce',
                        'signal': 'bullish',
                        'confidence': 0.7,
                        'description': f'RSI oversold at {rsi:.1f} and rising'
                    })
                
                # Pattern 3: Volume Breakout
                if volume_ratio > 2.0 and current_price > sma_20:
                    stock_patterns.append({
                        'type': 'volume_breakout',
                        'signal': 'bullish',
                        'confidence': 0.75,
                        'description': f'High volume ({volume_ratio:.1f}x) with price above SMA20'
                    })
                
                # Pattern 4: Bollinger Band Squeeze
                if df['BB_Width'].iloc[-1] < df['BB_Width'].rolling(20).mean().iloc[-1] * 0.5:
                    stock_patterns.append({
                        'type': 'bb_squeeze',
                        'signal': 'breakout_pending',
                        'confidence': 0.6,
                        'description': 'Bollinger Bands contracting - breakout expected'
                    })
                
                if stock_patterns:
                    patterns_found[stock] = stock_patterns
                    print(f"  📊 {stock}: {len(stock_patterns)} patterns detected")
            
            except Exception as e:
                print(f"  ❌ Error analyzing {stock}: {e}")
        
        print(f"\n🎯 Pattern Detection Complete: {len(patterns_found)} stocks with patterns")
        return patterns_found
    
    def run_ml_predictions(self):
        """Run machine learning predictions"""
        print("\n🤖 Running ML Predictions...")
        
        successful_predictions = 0
        
        for stock, df in self.stock_data.items():
            if stock.startswith('INDEX_') or len(df) < 60:
                continue
            
            try:
                print(f"  🔮 Training model for {stock}...")
                
                # Prepare features
                feature_columns = [
                    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI', 'MACD',
                    'BB_Width', 'Volume_Ratio', 'High_Low_Ratio', 'Price_vs_SMA20'
                ]
                
                # Clean data
                clean_df = df[feature_columns + ['Close']].dropna()
                
                if len(clean_df) < 30:
                    print(f"    ❌ Insufficient clean data")
                    continue
                
                # Prepare features and target
                X = clean_df[feature_columns].values
                y = clean_df['Close'].values
                
                # Create future target (next day price)
                X = X[:-1]  # All but last
                y = y[1:]   # All but first (shifted by 1)
                
                if len(X) < 20:
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train Random Forest model
                model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42
                )
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Predict next price
                latest_features = X[-1:] if len(X) > 0 else X_train[-1:]
                latest_scaled = scaler.transform(latest_features.reshape(1, -1))
                next_price_pred = model.predict(latest_scaled)[0]
                
                current_price = clean_df['Close'].iloc[-1]
                price_change_pct = (next_price_pred - current_price) / current_price * 100
                
                # Calculate confidence based on model performance
                confidence = max(0.5, min(0.9, 1 - (mae / current_price)))
                
                self.predictions[stock] = {
                    'current_price': float(current_price),
                    'predicted_price': float(next_price_pred),
                    'price_change_pct': float(price_change_pct),
                    'confidence': float(confidence),
                    'model_mae': float(mae),
                    'model_mse': float(mse)
                }
                
                successful_predictions += 1
                print(f"    ✅ Prediction: {price_change_pct:+.2f}% (confidence: {confidence:.1%})")
                
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
        
        print(f"\n🎯 ML Predictions Complete: {successful_predictions} stocks analyzed")
        return successful_predictions > 0
    
    def generate_signals(self, patterns):
        """Generate trading signals"""
        print("\n📈 Generating Trading Signals...")
        
        signals = []
        
        # Combine patterns and ML predictions
        for stock in self.stock_data.keys():
            if stock.startswith('INDEX_'):
                continue
            
            try:
                current_price = self.stock_data[stock]['Close'].iloc[-1]
                stock_patterns = patterns.get(stock, [])
                ml_prediction = self.predictions.get(stock, {})
                
                # Generate signals based on patterns
                for pattern in stock_patterns:
                    if pattern['signal'] in ['bullish']:
                        signal = {
                            'symbol': stock,
                            'signal_type': 'BUY',
                            'strength': 'MODERATE',
                            'confidence': pattern['confidence'],
                            'entry_price': current_price,
                            'target_price': current_price * 1.05,  # 5% target
                            'stop_loss': current_price * 0.97,     # 3% stop loss
                            'reasoning': [pattern['description']],
                            'source': 'pattern_analysis',
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Enhance with ML prediction if available
                        if ml_prediction and ml_prediction.get('price_change_pct', 0) > 2:
                            signal['confidence'] = min(0.9, signal['confidence'] + 0.1)
                            signal['reasoning'].append(f"ML predicts +{ml_prediction['price_change_pct']:.1f}%")
                            signal['target_price'] = ml_prediction['predicted_price']
                        
                        signals.append(signal)
                
                # Generate ML-based signals
                if ml_prediction and abs(ml_prediction.get('price_change_pct', 0)) > 3:
                    price_change = ml_prediction['price_change_pct']
                    
                    signal = {
                        'symbol': stock,
                        'signal_type': 'BUY' if price_change > 0 else 'SELL',
                        'strength': 'STRONG' if abs(price_change) > 5 else 'MODERATE',
                        'confidence': ml_prediction['confidence'],
                        'entry_price': current_price,
                        'target_price': ml_prediction['predicted_price'],
                        'stop_loss': current_price * (0.97 if price_change > 0 else 1.03),
                        'reasoning': [f"ML model predicts {price_change:+.1f}% move"],
                        'source': 'ml_prediction',
                        'timestamp': datetime.now().isoformat()
                    }
                    signals.append(signal)
            
            except Exception as e:
                print(f"  ❌ Error generating signals for {stock}: {e}")
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        self.signals = signals[:10]  # Top 10 signals
        print(f"\n🎯 Generated {len(self.signals)} high-confidence trading signals")
        
        return signals
    
    def display_results(self, patterns):
        """Display comprehensive results"""
        print("\n" + "="*80)
        print("🚀 INDIAN STOCK MARKET ML ANALYSIS RESULTS")
        print("="*80)
        
        print(f"📊 Analysis Summary:")
        print(f"   • Stocks Analyzed: {len([k for k in self.stock_data.keys() if not k.startswith('INDEX_')])}")
        print(f"   • Patterns Detected: {sum(len(p) for p in patterns.values())}")
        print(f"   • ML Predictions: {len(self.predictions)}")
        print(f"   • Trading Signals: {len(self.signals)}")
        print(f"   • Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.predictions:
            print(f"\n🤖 TOP ML PREDICTIONS:")
            print("-" * 50)
            sorted_predictions = sorted(
                self.predictions.items(), 
                key=lambda x: abs(x[1]['price_change_pct']), 
                reverse=True
            )
            
            for i, (stock, pred) in enumerate(sorted_predictions[:5], 1):
                direction = "📈" if pred['price_change_pct'] > 0 else "📉"
                print(f"{i}. {stock} {direction}")
                print(f"   Current: ₹{pred['current_price']:.2f}")
                print(f"   Predicted: ₹{pred['predicted_price']:.2f}")
                print(f"   Change: {pred['price_change_pct']:+.2f}%")
                print(f"   Confidence: {pred['confidence']:.1%}")
                print()
        
        if self.signals:
            print("📈 TOP TRADING SIGNALS:")
            print("-" * 50)
            for i, signal in enumerate(self.signals[:5], 1):
                action = "🟢 BUY" if signal['signal_type'] == 'BUY' else "🔴 SELL"
                print(f"{i}. {signal['symbol']} - {action}")
                print(f"   Entry: ₹{signal['entry_price']:.2f}")
                print(f"   Target: ₹{signal['target_price']:.2f}")
                print(f"   Stop Loss: ₹{signal['stop_loss']:.2f}")
                print(f"   Confidence: {signal['confidence']:.1%}")
                print(f"   Reason: {signal['reasoning'][0]}")
                print()
        
        if patterns:
            print("🔍 PATTERN HIGHLIGHTS:")
            print("-" * 30)
            pattern_types = {}
            for stock_patterns in patterns.values():
                for pattern in stock_patterns:
                    ptype = pattern['type']
                    pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
            
            for ptype, count in pattern_types.items():
                print(f"   • {ptype.replace('_', ' ').title()}: {count}")
        
        print("\n" + "="*80)
        
        # Save results
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': {
                'stocks_analyzed': len([k for k in self.stock_data.keys() if not k.startswith('INDEX_')]),
                'patterns_detected': sum(len(p) for p in patterns.values()),
                'ml_predictions': len(self.predictions),
                'trading_signals': len(self.signals)
            },
            'predictions': self.predictions,
            'signals': self.signals,
            'patterns': patterns
        }
        
        # Save to file
        try:
            os.makedirs('results', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            with open(f'results/ml_analysis_{timestamp}.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"💾 Results saved to: results/ml_analysis_{timestamp}.json")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
        
        return results

def main():
    """Main execution function"""
    if not HAS_IMPORTS:
        print("❌ Missing required dependencies. Please install:")
        print("   pip install pandas numpy yfinance ta scikit-learn loguru")
        return 1
    
    try:
        print("🚀 Starting Indian Stock Market ML Analysis...")
        
        # Initialize system
        analyzer = SimplifiedStockAnalysis()
        
        # Step 1: Collect data
        if not analyzer.collect_data():
            print("❌ Failed to collect sufficient data")
            return 1
        
        # Step 2: Detect patterns
        patterns = analyzer.detect_patterns()
        
        # Step 3: Run ML predictions
        if not analyzer.run_ml_predictions():
            print("❌ Failed to generate ML predictions")
            return 1
        
        # Step 4: Generate signals
        analyzer.generate_signals(patterns)
        
        # Step 5: Display results
        results = analyzer.display_results(patterns)
        
        print("\n✅ Analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())