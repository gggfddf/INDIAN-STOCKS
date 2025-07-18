"""
🚀 NIFTY 50 - 15 MINUTE ADVANCED ANALYZER 🚀
Dual Approach: Technical Indicators ML + Pure Price Action Analysis
Advanced Candlestick Pattern Recognition & Volume Analysis
"""

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class Nifty50AdvancedAnalyzer:
    
    def __init__(self):
        """Initialize with NIFTY 50 stocks"""
        
        # NIFTY 50 stocks (as of 2024)
        self.nifty50_stocks = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'BHARTIARTL', 'ICICIBANK', 'SBIN', 'LICI',
            'INFY', 'ITC', 'HINDUNILVR', 'LT', 'KOTAKBANK', 'AXISBANK', 'ASIANPAINT',
            'MARUTI', 'SUNPHARMA', 'TITAN', 'ULTRACEMCO', 'ONGC', 'TATAMOTORS',
            'NESTLEIND', 'NTPC', 'WIPRO', 'BAJFINANCE', 'POWERGRID', 'M&M', 'COALINDIA',
            'HCLTECH', 'BAJAJFINSV', 'TATASTEEL', 'GRASIM', 'ADANIPORTS', 'JSWSTEEL',
            'HINDALCO', 'INDUSINDBK', 'APOLLOHOSP', 'HEROMOTOCO', 'CIPLA', 'DIVISLAB',
            'TECHM', 'EICHERMOT', 'DRREDDY', 'BRITANNIA', 'BAJAJ-AUTO', 'SHRIRAMFIN',
            'TRENT', 'ADANIENT', 'SBILIFE', 'BEL', 'HDFCLIFE'
        ]
        
        self.market_data = {}
        self.technical_predictions = []
        self.price_action_signals = []
        self.combined_results = []
        
    def fetch_15min_data(self, period="5d"):
        """Fetch 15-minute data for all NIFTY 50 stocks"""
        print("📡 Fetching 15-minute data for NIFTY 50 stocks...")
        
        success_count = 0
        
        for symbol in self.nifty50_stocks:
            try:
                ticker = f"{symbol}.NS"
                stock = yf.Ticker(ticker)
                
                # Get 15-minute data
                data = stock.history(period=period, interval="15m")
                
                if len(data) >= 50:  # Minimum data points for analysis
                    self.market_data[symbol] = data
                    success_count += 1
                    print(f"✅ {symbol}: {len(data)} 15-min candles")
                else:
                    print(f"⚠️ {symbol}: Insufficient data")
                    
            except Exception as e:
                print(f"❌ {symbol}: Failed to fetch")
                
        print(f"\n📊 Successfully loaded {success_count} stocks with 15-min data")
        return success_count

    def calculate_advanced_technical_features(self, data):
        """Calculate comprehensive technical indicators for ML"""
        try:
            features = {}
            
            # Price-based features
            features['current_price'] = data['Close'].iloc[-1]
            features['price_change_1h'] = (data['Close'].iloc[-1] - data['Close'].iloc[-4]) / data['Close'].iloc[-4] * 100 if len(data) >= 4 else 0
            features['price_change_4h'] = (data['Close'].iloc[-1] - data['Close'].iloc[-16]) / data['Close'].iloc[-16] * 100 if len(data) >= 16 else 0
            features['price_change_1d'] = (data['Close'].iloc[-1] - data['Close'].iloc[-26]) / data['Close'].iloc[-26] * 100 if len(data) >= 26 else 0
            
            # Moving Averages (Multiple timeframes)
            try:
                features['sma_5'] = data['Close'].rolling(5).mean().iloc[-1]  # 1h 15min
                features['sma_20'] = data['Close'].rolling(20).mean().iloc[-1]  # 5h
                features['sma_50'] = data['Close'].rolling(50).mean().iloc[-1]  # 12.5h
                features['ema_9'] = data['Close'].ewm(span=9).mean().iloc[-1]
                features['ema_21'] = data['Close'].ewm(span=21).mean().iloc[-1]
                
                # Price vs MAs
                features['price_vs_sma5'] = (features['current_price'] - features['sma_5']) / features['sma_5'] * 100
                features['price_vs_sma20'] = (features['current_price'] - features['sma_20']) / features['sma_20'] * 100
                features['price_vs_ema9'] = (features['current_price'] - features['ema_9']) / features['ema_9'] * 100
                
            except:
                for key in ['sma_5', 'sma_20', 'sma_50', 'ema_9', 'ema_21', 'price_vs_sma5', 'price_vs_sma20', 'price_vs_ema9']:
                    features[key] = 0
            
            # Momentum Indicators
            try:
                features['rsi_14'] = ta.momentum.rsi(data['Close'], window=14).iloc[-1]
                features['rsi_9'] = ta.momentum.rsi(data['Close'], window=9).iloc[-1]
                features['stoch_k'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=14).iloc[-1]
                features['stoch_d'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'], window=14).iloc[-1]
                features['williams_r'] = ta.momentum.williams_r(data['High'], data['Low'], data['Close'], window=14).iloc[-1]
                
                # RSI divergence
                rsi_series = ta.momentum.rsi(data['Close'], window=14)
                features['rsi_slope'] = (rsi_series.iloc[-1] - rsi_series.iloc[-5]) if len(rsi_series) >= 5 else 0
                
            except:
                for key in ['rsi_14', 'rsi_9', 'stoch_k', 'stoch_d', 'williams_r', 'rsi_slope']:
                    features[key] = 50 if 'rsi' in key else 0
            
            # MACD Analysis
            try:
                features['macd'] = ta.trend.macd_diff(data['Close']).iloc[-1]
                features['macd_signal'] = ta.trend.macd_signal(data['Close']).iloc[-1]
                features['macd_histogram'] = ta.trend.macd_diff(data['Close']).iloc[-1] - ta.trend.macd_signal(data['Close']).iloc[-1]
                
                # MACD momentum
                macd_series = ta.trend.macd_diff(data['Close'])
                features['macd_momentum'] = (macd_series.iloc[-1] - macd_series.iloc[-3]) if len(macd_series) >= 3 else 0
                
            except:
                features['macd'] = features['macd_signal'] = features['macd_histogram'] = features['macd_momentum'] = 0
            
            # Bollinger Bands
            try:
                bb_upper = ta.volatility.bollinger_hband(data['Close'], window=20)
                bb_lower = ta.volatility.bollinger_lband(data['Close'], window=20)
                bb_middle = ta.volatility.bollinger_mavg(data['Close'], window=20)
                
                features['bb_upper'] = bb_upper.iloc[-1]
                features['bb_lower'] = bb_lower.iloc[-1]
                features['bb_position'] = (features['current_price'] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                features['bb_squeeze'] = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
                
            except:
                features['bb_upper'] = features['bb_lower'] = features['current_price']
                features['bb_position'] = features['bb_squeeze'] = 0.5
            
            # Volume Analysis
            try:
                features['volume_current'] = data['Volume'].iloc[-1]
                features['volume_sma'] = data['Volume'].rolling(20).mean().iloc[-1]
                features['volume_ratio'] = features['volume_current'] / features['volume_sma'] if features['volume_sma'] > 0 else 1
                features['volume_rsi'] = ta.momentum.rsi(data['Volume'], window=14).iloc[-1]
                
                # Volume trend
                vol_series = data['Volume'].rolling(5).mean()
                features['volume_trend'] = (vol_series.iloc[-1] - vol_series.iloc[-5]) / vol_series.iloc[-5] * 100 if len(vol_series) >= 5 else 0
                
            except:
                features['volume_current'] = features['volume_sma'] = 1000000
                features['volume_ratio'] = features['volume_rsi'] = features['volume_trend'] = 1
            
            # Volatility Indicators
            try:
                features['atr'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14).iloc[-1]
                features['volatility'] = data['Close'].pct_change().rolling(20).std().iloc[-1] * 100
                
                # Volatility percentile
                vol_series = data['Close'].pct_change().rolling(20).std()
                features['vol_percentile'] = (vol_series.iloc[-1] - vol_series.min()) / (vol_series.max() - vol_series.min()) if len(vol_series) > 20 else 0.5
                
            except:
                features['atr'] = features['volatility'] = features['vol_percentile'] = 1
            
            # Trend Strength
            try:
                features['adx'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=14).iloc[-1]
                features['pdi'] = ta.trend.adx_pos(data['High'], data['Low'], data['Close'], window=14).iloc[-1]
                features['mdi'] = ta.trend.adx_neg(data['High'], data['Low'], data['Close'], window=14).iloc[-1]
                
            except:
                features['adx'] = features['pdi'] = features['mdi'] = 25
            
            # Support/Resistance Analysis
            try:
                high_20 = data['High'].rolling(20).max().iloc[-1]
                low_20 = data['Low'].rolling(20).min().iloc[-1]
                features['resistance_distance'] = (high_20 - features['current_price']) / features['current_price'] * 100
                features['support_distance'] = (features['current_price'] - low_20) / features['current_price'] * 100
                
                # Pivot points
                high_prev = data['High'].iloc[-2]
                low_prev = data['Low'].iloc[-2]
                close_prev = data['Close'].iloc[-2]
                pivot = (high_prev + low_prev + close_prev) / 3
                features['pivot_distance'] = (features['current_price'] - pivot) / pivot * 100
                
            except:
                features['resistance_distance'] = features['support_distance'] = features['pivot_distance'] = 2
            
            # Market Microstructure
            try:
                # Bid-Ask spread proxy using OHLC
                features['spread_proxy'] = (data['High'].iloc[-1] - data['Low'].iloc[-1]) / data['Close'].iloc[-1] * 100
                
                # Price momentum
                returns = data['Close'].pct_change()
                features['momentum_3'] = returns.rolling(3).sum().iloc[-1] * 100
                features['momentum_9'] = returns.rolling(9).sum().iloc[-1] * 100
                
                # Acceleration
                features['acceleration'] = (returns.iloc[-1] - returns.iloc[-2]) * 10000 if len(returns) >= 2 else 0
                
            except:
                features['spread_proxy'] = features['momentum_3'] = features['momentum_9'] = features['acceleration'] = 0
            
            return features
            
        except Exception as e:
            print(f"Advanced technical feature error: {e}")
            return None

    def detect_candlestick_patterns(self, data):
        """Detect comprehensive candlestick patterns"""
        try:
            if len(data) < 5:
                return {}
                
            patterns = {}
            
            # Get OHLC for last 5 candles
            opens = data['Open'].iloc[-5:].values
            highs = data['High'].iloc[-5:].values
            lows = data['Low'].iloc[-5:].values
            closes = data['Close'].iloc[-5:].values
            volumes = data['Volume'].iloc[-5:].values
            
            # Current candle
            o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
            body = abs(c - o)
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            total_range = h - l
            
            # Avoid division by zero
            if total_range == 0:
                return {}
            
            # Single Candle Patterns
            patterns['doji'] = body / total_range < 0.1
            patterns['spinning_top'] = body / total_range < 0.3 and upper_shadow > body and lower_shadow > body
            patterns['hammer'] = (lower_shadow > 2 * body) and (upper_shadow < body) and (c > o)
            patterns['inverted_hammer'] = (upper_shadow > 2 * body) and (lower_shadow < body)
            patterns['shooting_star'] = (upper_shadow > 2 * body) and (lower_shadow < body) and (o > c)
            patterns['hanging_man'] = (lower_shadow > 2 * body) and (upper_shadow < body) and (o > c)
            patterns['marubozu_bullish'] = (body / total_range > 0.9) and (c > o)
            patterns['marubozu_bearish'] = (body / total_range > 0.9) and (o > c)
            
            # Two Candle Patterns
            if len(closes) >= 2:
                prev_o, prev_h, prev_l, prev_c = opens[-2], highs[-2], lows[-2], closes[-2]
                prev_body = abs(prev_c - prev_o)
                
                patterns['bullish_engulfing'] = (prev_c < prev_o) and (c > o) and (o < prev_c) and (c > prev_o)
                patterns['bearish_engulfing'] = (prev_c > prev_o) and (o > c) and (o > prev_c) and (c < prev_o)
                patterns['piercing_line'] = (prev_c < prev_o) and (c > o) and (o < prev_l) and (c > (prev_o + prev_c) / 2) and (c < prev_o)
                patterns['dark_cloud_cover'] = (prev_c > prev_o) and (o > c) and (o > prev_h) and (c < (prev_o + prev_c) / 2) and (c > prev_o)
                patterns['harami_bullish'] = (prev_c < prev_o) and (c > o) and (o > prev_c) and (c < prev_o)
                patterns['harami_bearish'] = (prev_c > prev_o) and (o > c) and (o < prev_c) and (c > prev_o)
            
            # Three Candle Patterns
            if len(closes) >= 3:
                patterns['morning_star'] = self._is_morning_star(opens[-3:], highs[-3:], lows[-3:], closes[-3:])
                patterns['evening_star'] = self._is_evening_star(opens[-3:], highs[-3:], lows[-3:], closes[-3:])
                patterns['three_white_soldiers'] = self._is_three_white_soldiers(opens[-3:], closes[-3:])
                patterns['three_black_crows'] = self._is_three_black_crows(opens[-3:], closes[-3:])
            
            # Volume-based patterns
            if len(volumes) >= 2:
                vol_ratio = volumes[-1] / volumes[-2] if volumes[-2] > 0 else 1
                patterns['volume_surge'] = vol_ratio > 2.0
                patterns['volume_dry_up'] = vol_ratio < 0.5
                patterns['volume_confirmation'] = vol_ratio > 1.5 and ((c > o and patterns.get('bullish_engulfing', False)) or 
                                                                        (o > c and patterns.get('bearish_engulfing', False)))
            
            return patterns
            
        except Exception as e:
            print(f"Candlestick pattern error: {e}")
            return {}

    def _is_morning_star(self, opens, highs, lows, closes):
        """Detect morning star pattern"""
        try:
            if len(closes) < 3:
                return False
            
            # First candle: bearish
            first_bearish = closes[0] < opens[0]
            # Second candle: small body (star)
            second_small = abs(closes[1] - opens[1]) < abs(closes[0] - opens[0]) * 0.3
            # Third candle: bullish and closes above midpoint of first
            third_bullish = closes[2] > opens[2]
            third_recovery = closes[2] > (opens[0] + closes[0]) / 2
            
            return first_bearish and second_small and third_bullish and third_recovery
        except:
            return False

    def _is_evening_star(self, opens, highs, lows, closes):
        """Detect evening star pattern"""
        try:
            if len(closes) < 3:
                return False
            
            # First candle: bullish
            first_bullish = closes[0] > opens[0]
            # Second candle: small body (star)
            second_small = abs(closes[1] - opens[1]) < abs(closes[0] - opens[0]) * 0.3
            # Third candle: bearish and closes below midpoint of first
            third_bearish = closes[2] < opens[2]
            third_decline = closes[2] < (opens[0] + closes[0]) / 2
            
            return first_bullish and second_small and third_bearish and third_decline
        except:
            return False

    def _is_three_white_soldiers(self, opens, closes):
        """Detect three white soldiers pattern"""
        try:
            if len(closes) < 3:
                return False
            
            all_bullish = all(closes[i] > opens[i] for i in range(3))
            ascending = closes[0] < closes[1] < closes[2]
            opens_within_body = all(opens[i] > closes[i-1] * 0.95 and opens[i] < closes[i-1] * 1.05 for i in range(1, 3))
            
            return all_bullish and ascending and opens_within_body
        except:
            return False

    def _is_three_black_crows(self, opens, closes):
        """Detect three black crows pattern"""
        try:
            if len(closes) < 3:
                return False
            
            all_bearish = all(closes[i] < opens[i] for i in range(3))
            descending = closes[0] > closes[1] > closes[2]
            opens_within_body = all(opens[i] < closes[i-1] * 1.05 and opens[i] > closes[i-1] * 0.95 for i in range(1, 3))
            
            return all_bearish and descending and opens_within_body
        except:
            return False

    def analyze_price_action(self, symbol, data):
        """Comprehensive price action analysis"""
        try:
            if len(data) < 20:
                return None
                
            # Get candlestick patterns
            patterns = self.detect_candlestick_patterns(data)
            
            # Price action features
            price_action = {}
            
            # Recent price movement
            closes = data['Close'].iloc[-10:].values
            highs = data['High'].iloc[-10:].values
            lows = data['Low'].iloc[-10:].values
            volumes = data['Volume'].iloc[-10:].values
            
            # Trend analysis
            price_action['short_trend'] = 'UP' if closes[-1] > closes[-5] else 'DOWN'
            price_action['medium_trend'] = 'UP' if closes[-1] > closes[-10] else 'DOWN'
            
            # Swing analysis
            price_action['higher_highs'] = all(highs[i] >= highs[i-1] for i in range(-3, 0))
            price_action['higher_lows'] = all(lows[i] >= lows[i-1] for i in range(-3, 0))
            price_action['lower_highs'] = all(highs[i] <= highs[i-1] for i in range(-3, 0))
            price_action['lower_lows'] = all(lows[i] <= lows[i-1] for i in range(-3, 0))
            
            # Support/Resistance levels
            recent_high = max(highs)
            recent_low = min(lows)
            current_price = closes[-1]
            
            price_action['near_resistance'] = (recent_high - current_price) / current_price < 0.01
            price_action['near_support'] = (current_price - recent_low) / current_price < 0.01
            
            # Volume analysis
            avg_volume = np.mean(volumes[:-1])
            price_action['volume_spike'] = volumes[-1] > avg_volume * 1.5
            price_action['volume_climax'] = volumes[-1] > avg_volume * 2.0
            
            # Combine patterns and price action
            signal_strength = 0
            signal_direction = 'NEUTRAL'
            reasons = []
            
            # Bullish signals
            if patterns.get('hammer', False) or patterns.get('bullish_engulfing', False) or patterns.get('morning_star', False):
                signal_strength += 3
                signal_direction = 'BULLISH'
                reasons.append("Strong bullish candlestick pattern")
            
            if patterns.get('three_white_soldiers', False):
                signal_strength += 4
                signal_direction = 'BULLISH'
                reasons.append("Three white soldiers pattern")
            
            if price_action['higher_highs'] and price_action['higher_lows']:
                signal_strength += 2
                if signal_direction != 'BEARISH':
                    signal_direction = 'BULLISH'
                reasons.append("Higher highs and higher lows")
            
            # Bearish signals
            if patterns.get('shooting_star', False) or patterns.get('bearish_engulfing', False) or patterns.get('evening_star', False):
                signal_strength -= 3
                signal_direction = 'BEARISH'
                reasons.append("Strong bearish candlestick pattern")
            
            if patterns.get('three_black_crows', False):
                signal_strength -= 4
                signal_direction = 'BEARISH'
                reasons.append("Three black crows pattern")
            
            if price_action['lower_highs'] and price_action['lower_lows']:
                signal_strength -= 2
                if signal_direction != 'BULLISH':
                    signal_direction = 'BEARISH'
                reasons.append("Lower highs and lower lows")
            
            # Volume confirmation
            if price_action['volume_spike'] and signal_strength != 0:
                signal_strength *= 1.5
                reasons.append("Volume confirmation")
            
            # Support/Resistance
            if price_action['near_resistance'] and signal_direction == 'BULLISH':
                signal_strength *= 0.7  # Reduce strength near resistance
                reasons.append("Near resistance level")
            
            if price_action['near_support'] and signal_direction == 'BEARISH':
                signal_strength *= 0.7  # Reduce strength near support
                reasons.append("Near support level")
            
            return {
                'symbol': symbol,
                'signal_direction': signal_direction,
                'signal_strength': round(abs(signal_strength), 2),
                'confidence': min(abs(signal_strength) * 10, 95),
                'patterns_detected': [k for k, v in patterns.items() if v],
                'price_action': price_action,
                'reasons': reasons,
                'current_price': current_price,
                'timeframe': '15min'
            }
            
        except Exception as e:
            print(f"Price action analysis error for {symbol}: {e}")
            return None

    def build_advanced_ml_model(self, symbol, data):
        """Build advanced ML model with ensemble approach"""
        try:
            if len(data) < 100:
                return None
                
            # Prepare training data
            X_data = []
            y_data = []
            
            # Use larger window for 15-min data
            window_size = 20
            forecast_horizon = 4  # Predict 1 hour ahead (4 x 15min)
            
            for i in range(window_size, len(data) - forecast_horizon):
                features = self.calculate_advanced_technical_features(data.iloc[i-window_size:i+1])
                if features:
                    feature_values = [v for k, v in features.items() if k != 'current_price' and isinstance(v, (int, float))]
                    feature_values = [0 if np.isnan(x) or np.isinf(x) else x for x in feature_values]
                    
                    if len(feature_values) >= 20:
                        X_data.append(feature_values)
                        
                        # Multi-target: predict direction and magnitude
                        future_price = data['Close'].iloc[i + forecast_horizon]
                        current_price = data['Close'].iloc[i]
                        
                        # Direction (classification target converted to regression)
                        direction = 1 if future_price > current_price else -1
                        magnitude = abs(future_price - current_price) / current_price * 100
                        
                        # Combined target: direction * magnitude
                        target = direction * magnitude
                        y_data.append(target)
            
            if len(X_data) < 30:
                return None
                
            X = np.array(X_data)
            y = np.array(y_data)
            
            # Handle NaN values
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Ensemble of different models
            models = {
                'rf': RandomForestRegressor(
                    n_estimators=200, 
                    max_depth=15, 
                    min_samples_split=5,
                    random_state=42
                ),
                'gbm': GradientBoostingRegressor(
                    n_estimators=150,
                    max_depth=10,
                    learning_rate=0.1,
                    random_state=42
                ),
                'mlp': MLPRegressor(
                    hidden_layer_sizes=(100, 50, 25),
                    activation='relu',
                    solver='adam',
                    max_iter=500,
                    random_state=42
                )
            }
            
            model_scores = {}
            model_predictions = {}
            
            # Train and validate each model
            for name, model in models.items():
                scores = []
                predictions = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Scale features
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    model.fit(X_train_scaled, y_train)
                    pred = model.predict(X_val_scaled)
                    
                    score = mean_absolute_error(y_val, pred)
                    scores.append(score)
                    predictions.extend(pred)
                
                model_scores[name] = np.mean(scores)
                model_predictions[name] = predictions
            
            # Select best model or create ensemble
            best_model_name = min(model_scores, key=model_scores.get)
            best_model = models[best_model_name]
            
            # Final training on all data
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            best_model.fit(X_scaled, y)
            
            # Make prediction for current data
            current_features = self.calculate_advanced_technical_features(data)
            if current_features:
                current_values = [v for k, v in current_features.items() if k != 'current_price' and isinstance(v, (int, float))]
                current_values = [0 if np.isnan(x) or np.isinf(x) else x for x in current_values]
                
                if len(current_values) == X.shape[1]:
                    current_array = np.array([current_values])
                    current_array = imputer.transform(current_array)
                    current_scaled = scaler.transform(current_array)
                    
                    prediction = best_model.predict(current_scaled)[0]
                    
                    # Calculate confidence based on model performance
                    confidence = max(20, min(95, 100 - model_scores[best_model_name] * 10))
                    
                    return {
                        'symbol': symbol,
                        'prediction': prediction,
                        'direction': 'BULLISH' if prediction > 0 else 'BEARISH',
                        'magnitude': abs(prediction),
                        'confidence': confidence,
                        'model_used': best_model_name,
                        'model_score': model_scores[best_model_name],
                        'current_price': current_features['current_price'],
                        'timeframe': '1hour_ahead'
                    }
            
            return None
            
        except Exception as e:
            print(f"ML model error for {symbol}: {e}")
            return None

    def run_comprehensive_analysis(self):
        """Run complete dual analysis"""
        print("🚀 STARTING COMPREHENSIVE NIFTY 50 - 15MIN ANALYSIS 🚀")
        print("="*70)
        
        # Fetch data
        stocks_loaded = self.fetch_15min_data()
        if stocks_loaded < 10:
            print("❌ Insufficient data. Cannot proceed.")
            return
            
        print(f"\n🤖 Running dual analysis on {stocks_loaded} stocks...")
        
        technical_success = 0
        price_action_success = 0
        
        for symbol in self.market_data.keys():
            print(f"\n📊 Analyzing {symbol}...")
            
            data = self.market_data[symbol]
            
            # Technical Indicators ML Analysis
            try:
                tech_result = self.build_advanced_ml_model(symbol, data)
                if tech_result:
                    self.technical_predictions.append(tech_result)
                    technical_success += 1
                    print(f"  ✅ Technical ML: {tech_result['direction']} {tech_result['magnitude']:.2f}% (Conf: {tech_result['confidence']:.0f}%)")
            except Exception as e:
                print(f"  ❌ Technical analysis failed: {e}")
            
            # Price Action Analysis
            try:
                pa_result = self.analyze_price_action(symbol, data)
                if pa_result:
                    self.price_action_signals.append(pa_result)
                    price_action_success += 1
                    print(f"  ✅ Price Action: {pa_result['signal_direction']} (Strength: {pa_result['signal_strength']:.1f})")
                    if pa_result['patterns_detected']:
                        print(f"    🕯️ Patterns: {', '.join(pa_result['patterns_detected'][:3])}")
            except Exception as e:
                print(f"  ❌ Price action analysis failed: {e}")
        
        print(f"\n📈 Technical ML Analysis: {technical_success} successful")
        print(f"🕯️ Price Action Analysis: {price_action_success} successful")
        
        # Combine results
        self.combine_analyses()
        
        return technical_success, price_action_success

    def combine_analyses(self):
        """Combine both analyses for final recommendations"""
        print(f"\n🔄 Combining technical and price action analyses...")
        
        # Create combined results
        for tech in self.technical_predictions:
            symbol = tech['symbol']
            
            # Find corresponding price action result
            pa_result = next((pa for pa in self.price_action_signals if pa['symbol'] == symbol), None)
            
            if pa_result:
                # Combine signals
                combined = {
                    'symbol': symbol,
                    'current_price': tech['current_price'],
                    
                    # Technical ML
                    'ml_direction': tech['direction'],
                    'ml_magnitude': tech['magnitude'],
                    'ml_confidence': tech['confidence'],
                    'ml_model': tech['model_used'],
                    
                    # Price Action
                    'pa_direction': pa_result['signal_direction'],
                    'pa_strength': pa_result['signal_strength'],
                    'pa_patterns': pa_result['patterns_detected'],
                    'pa_reasons': pa_result['reasons'],
                    
                    # Combined Analysis
                    'agreement': tech['direction'] == pa_result['signal_direction'],
                    'combined_confidence': 0,
                    'final_recommendation': 'NEUTRAL',
                    'risk_level': 'MEDIUM'
                }
                
                # Calculate combined confidence
                if combined['agreement']:
                    # Both agree - high confidence
                    combined['combined_confidence'] = min(95, (tech['confidence'] + pa_result['confidence']) / 2 * 1.2)
                    combined['final_recommendation'] = tech['direction']
                    combined['risk_level'] = 'LOW' if combined['combined_confidence'] > 80 else 'MEDIUM'
                else:
                    # Disagreement - lower confidence
                    combined['combined_confidence'] = max(tech['confidence'], pa_result['confidence']) * 0.6
                    
                    # Choose stronger signal
                    if tech['confidence'] > pa_result['confidence'] * 2:
                        combined['final_recommendation'] = tech['direction']
                    elif pa_result['confidence'] > tech['confidence'] * 2:
                        combined['final_recommendation'] = pa_result['signal_direction']
                    else:
                        combined['final_recommendation'] = 'NEUTRAL'
                    
                    combined['risk_level'] = 'HIGH'
                
                self.combined_results.append(combined)
        
        # Sort by combined confidence
        self.combined_results.sort(key=lambda x: x['combined_confidence'], reverse=True)
        
        print(f"✅ Combined {len(self.combined_results)} analyses")

    def generate_comprehensive_report(self):
        """Generate detailed analysis report"""
        print(f"\n📋 GENERATING COMPREHENSIVE ANALYSIS REPORT...")
        
        report = []
        report.append("🚀 NIFTY 50 - 15 MINUTE ADVANCED ANALYSIS REPORT 🚀")
        report.append("="*70)
        report.append(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Timeframe: 15-minute candles")
        report.append(f"Stocks Analyzed: {len(self.combined_results)}")
        report.append("")
        
        # Summary Statistics
        bullish_count = len([r for r in self.combined_results if r['final_recommendation'] == 'BULLISH'])
        bearish_count = len([r for r in self.combined_results if r['final_recommendation'] == 'BEARISH'])
        neutral_count = len([r for r in self.combined_results if r['final_recommendation'] == 'NEUTRAL'])
        agreement_count = len([r for r in self.combined_results if r['agreement']])
        
        report.append("📊 MARKET SENTIMENT OVERVIEW:")
        report.append(f"  📈 Bullish Signals: {bullish_count} ({bullish_count/len(self.combined_results)*100:.1f}%)")
        report.append(f"  📉 Bearish Signals: {bearish_count} ({bearish_count/len(self.combined_results)*100:.1f}%)")
        report.append(f"  ⚪ Neutral Signals: {neutral_count} ({neutral_count/len(self.combined_results)*100:.1f}%)")
        report.append(f"  🤝 ML-PA Agreement: {agreement_count}/{len(self.combined_results)} ({agreement_count/len(self.combined_results)*100:.1f}%)")
        report.append("")
        
        # Top Recommendations
        report.append("🏆 TOP 10 HIGHEST CONFIDENCE SIGNALS:")
        report.append("-" * 70)
        
        for i, result in enumerate(self.combined_results[:10], 1):
            symbol = result['symbol']
            direction = result['final_recommendation']
            confidence = result['combined_confidence']
            price = result['current_price']
            ml_mag = result['ml_magnitude']
            patterns = ', '.join(result['pa_patterns'][:2]) if result['pa_patterns'] else 'None'
            
            direction_emoji = "📈" if direction == "BULLISH" else "📉" if direction == "BEARISH" else "⚪"
            risk_emoji = "🟢" if result['risk_level'] == 'LOW' else "🟡" if result['risk_level'] == 'MEDIUM' else "🔴"
            
            report.append(f"{i:2}. {direction_emoji} {symbol:<12} | {direction:<8} | Conf: {confidence:.0f}% {risk_emoji}")
            report.append(f"    Price: ₹{price:.2f} | ML Magnitude: {ml_mag:.2f}% | Patterns: {patterns}")
            report.append("")
        
        # Detailed Analysis by Category
        report.append("\n🔍 DETAILED ANALYSIS BY SIGNAL TYPE:")
        report.append("-" * 70)
        
        # Strong Bullish Signals
        strong_bullish = [r for r in self.combined_results if r['final_recommendation'] == 'BULLISH' and r['combined_confidence'] > 70]
        if strong_bullish:
            report.append(f"\n📈 STRONG BULLISH SIGNALS ({len(strong_bullish)} stocks):")
            for result in strong_bullish:
                patterns_str = ', '.join(result['pa_patterns']) if result['pa_patterns'] else 'Technical only'
                report.append(f"  • {result['symbol']}: {result['combined_confidence']:.0f}% confidence")
                report.append(f"    ML: {result['ml_direction']} {result['ml_magnitude']:.2f}% | PA: {result['pa_direction']}")
                report.append(f"    Patterns: {patterns_str}")
                report.append("")
        
        # Strong Bearish Signals
        strong_bearish = [r for r in self.combined_results if r['final_recommendation'] == 'BEARISH' and r['combined_confidence'] > 70]
        if strong_bearish:
            report.append(f"\n📉 STRONG BEARISH SIGNALS ({len(strong_bearish)} stocks):")
            for result in strong_bearish:
                patterns_str = ', '.join(result['pa_patterns']) if result['pa_patterns'] else 'Technical only'
                report.append(f"  • {result['symbol']}: {result['combined_confidence']:.0f}% confidence")
                report.append(f"    ML: {result['ml_direction']} {result['ml_magnitude']:.2f}% | PA: {result['pa_direction']}")
                report.append(f"    Patterns: {patterns_str}")
                report.append("")
        
        # Conflicting Signals (High Risk)
        conflicts = [r for r in self.combined_results if not r['agreement'] and r['combined_confidence'] > 40]
        if conflicts:
            report.append(f"\n⚠️  CONFLICTING SIGNALS - HIGH RISK ({len(conflicts)} stocks):")
            for result in conflicts:
                report.append(f"  • {result['symbol']}: ML={result['ml_direction']} vs PA={result['pa_direction']}")
                report.append(f"    Confidence: {result['combined_confidence']:.0f}% | Risk: {result['risk_level']}")
                report.append("")
        
        # Pattern Analysis Summary
        all_patterns = []
        for result in self.combined_results:
            all_patterns.extend(result['pa_patterns'])
        
        if all_patterns:
            from collections import Counter
            pattern_counts = Counter(all_patterns)
            
            report.append(f"\n🕯️  MOST FREQUENT CANDLESTICK PATTERNS:")
            for pattern, count in pattern_counts.most_common(10):
                report.append(f"  • {pattern.replace('_', ' ').title()}: {count} occurrences")
            report.append("")
        
        # Trading Recommendations
        report.append("\n💡 ACTIONABLE TRADING RECOMMENDATIONS:")
        report.append("-" * 70)
        
        # Immediate action stocks
        immediate_bullish = [r for r in self.combined_results[:5] if r['final_recommendation'] == 'BULLISH']
        immediate_bearish = [r for r in self.combined_results[:5] if r['final_recommendation'] == 'BEARISH']
        
        if immediate_bullish:
            report.append("\n🚀 IMMEDIATE BUY CANDIDATES (Next 1 hour):")
            for result in immediate_bullish:
                report.append(f"  • {result['symbol']}: Entry around ₹{result['current_price']:.2f}")
                report.append(f"    Target: +{result['ml_magnitude']:.2f}% | Risk: {result['risk_level']}")
                report.append("")
        
        if immediate_bearish:
            report.append("\n🎯 IMMEDIATE SHORT CANDIDATES (Next 1 hour):")
            for result in immediate_bearish:
                report.append(f"  • {result['symbol']}: Entry around ₹{result['current_price']:.2f}")
                report.append(f"    Target: -{result['ml_magnitude']:.2f}% | Risk: {result['risk_level']}")
                report.append("")
        
        # Risk Management
        report.append("\n🛡️  RISK MANAGEMENT GUIDELINES:")
        report.append("  • High Confidence (>80%): Risk 2-3% of capital per trade")
        report.append("  • Medium Confidence (60-80%): Risk 1-2% of capital per trade")
        report.append("  • Low Confidence (<60%): Risk 0.5-1% of capital per trade")
        report.append("  • Conflicting Signals: Avoid or use very tight stops")
        report.append("  • Always use stop losses: 1-2% for 15-min trades")
        report.append("")
        
        report.append("⚠️  DISCLAIMER: This analysis is for educational purposes. Always do your own research!")
        
        # Save report
        report_text = '\n'.join(report)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"nifty50_15min_analysis_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(report_text)
        
        print(f"✅ Comprehensive report saved to: {filename}")
        print(f"📄 Report contains {len(report)} lines of detailed analysis")
        
        # Print summary to console
        print("\n" + report_text[:2000] + "...")  # Print first part
        
        return filename

def main():
    """Main execution function"""
    analyzer = Nifty50AdvancedAnalyzer()
    
    # Run comprehensive analysis
    tech_count, pa_count = analyzer.run_comprehensive_analysis()
    
    if tech_count > 0 or pa_count > 0:
        # Generate detailed report
        report_file = analyzer.generate_comprehensive_report()
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"📊 Technical Analysis: {tech_count} stocks")
        print(f"🕯️ Price Action Analysis: {pa_count} stocks")
        print(f"📋 Detailed report: {report_file}")
    else:
        print("❌ No analysis completed")

if __name__ == "__main__":
    main()