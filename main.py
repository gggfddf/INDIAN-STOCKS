"""
Main Orchestrator for Indian Stock Market Deep Learning Analysis System
Combines data collection, pattern detection, ML models, and signal generation
"""

import sys
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.config import Config
from src.data_sources.data_collector import DataCollector
from src.feature_engineering.technical_indicators import TechnicalIndicators
from src.pattern_detection.pattern_detector import PatternDetector
from src.models.deep_learning_models import EnsemblePredictor, AnomalyDetector, MarketRegimeDetector
from src.trading_signals.signal_generator import SignalGenerator
from src.utils.helpers import setup_logging, format_currency


class StockMarketAnalysisSystem:
    """Main system orchestrator for Indian stock market analysis"""
    
    def __init__(self):
        self.config = Config()
        self.data_collector = DataCollector()
        self.technical_indicators = TechnicalIndicators()
        self.pattern_detector = PatternDetector()
        self.ensemble_predictor = EnsemblePredictor(self.config)
        self.anomaly_detector = AnomalyDetector()
        self.regime_detector = MarketRegimeDetector()
        self.signal_generator = SignalGenerator()
        
        # Data storage
        self.stock_data = {}
        self.indices_data = {}
        self.enriched_data = {}
        self.detected_patterns = {}
        self.ml_predictions = {}
        self.trading_signals = []
        
        setup_logging()
        logger.info("Stock Market Analysis System initialized")
    
    def run_complete_analysis(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Run complete stock market analysis pipeline
        
        Args:
            save_results: Whether to save results to files
            
        Returns:
            Dictionary containing all analysis results
        """
        start_time = time.time()
        logger.info("Starting complete stock market analysis...")
        
        try:
            # Step 1: Data Collection
            logger.info("Step 1: Collecting market data...")
            self._collect_market_data()
            
            # Step 2: Feature Engineering
            logger.info("Step 2: Engineering technical features...")
            self._engineer_features()
            
            # Step 3: Pattern Detection
            logger.info("Step 3: Detecting market patterns...")
            self._detect_patterns()
            
            # Step 4: ML Predictions
            logger.info("Step 4: Generating ML predictions...")
            self._generate_ml_predictions()
            
            # Step 5: Anomaly Detection
            logger.info("Step 5: Detecting market anomalies...")
            anomalies = self._detect_anomalies()
            
            # Step 6: Market Regime Detection
            logger.info("Step 6: Detecting market regimes...")
            market_regimes = self._detect_market_regimes()
            
            # Step 7: Generate Trading Signals
            logger.info("Step 7: Generating trading signals...")
            self._generate_trading_signals()
            
            # Step 8: Compile Results
            logger.info("Step 8: Compiling analysis results...")
            results = self._compile_results(anomalies, market_regimes)
            
            # Step 9: Save Results
            if save_results:
                logger.info("Step 9: Saving results...")
                self._save_results(results)
            
            execution_time = time.time() - start_time
            logger.info(f"Complete analysis finished in {execution_time:.2f} seconds")
            
            # Print summary
            self._print_analysis_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {str(e)}")
            raise
    
    def _collect_market_data(self):
        """Collect data for all stocks and indices"""
        try:
            # Collect stock data
            logger.info(f"Collecting data for {len(self.config.TOP_100_FO_STOCKS)} stocks...")
            yahoo_symbols = self.config.get_stock_symbols_with_suffix()
            
            # Fetch in batches to avoid rate limiting
            batch_size = 20
            for i in range(0, len(yahoo_symbols), batch_size):
                batch = yahoo_symbols[i:i+batch_size]
                batch_data = self.data_collector.fetch_multiple_stocks(batch, period="1y")
                
                # Convert back to original symbols (remove .NS suffix)
                for yahoo_symbol, data in batch_data.items():
                    original_symbol = yahoo_symbol.replace('.NS', '')
                    self.stock_data[original_symbol] = data
                
                logger.info(f"Collected data for batch {i//batch_size + 1}")
                time.sleep(1)  # Rate limiting
            
            # Collect indices data
            logger.info("Collecting indices data...")
            self.indices_data = self.data_collector.fetch_all_indices(period="1y")
            
            # Data quality check
            valid_stocks = 0
            for symbol, data in self.stock_data.items():
                is_valid, issues = self.data_collector.validate_data_quality(data, symbol)
                if is_valid:
                    valid_stocks += 1
                elif issues:
                    logger.warning(f"Data quality issues for {symbol}: {'; '.join(issues)}")
            
            logger.info(f"Successfully collected data for {valid_stocks} stocks and {len(self.indices_data)} indices")
            
        except Exception as e:
            logger.error(f"Error collecting market data: {str(e)}")
            raise
    
    def _engineer_features(self):
        """Engineer technical features for all collected data"""
        try:
            # Process stock data
            for symbol, data in self.stock_data.items():
                if len(data) > 50:  # Minimum data requirement
                    enriched = self.technical_indicators.calculate_all_indicators(data)
                    self.enriched_data[symbol] = enriched
                else:
                    logger.warning(f"Insufficient data for {symbol}: {len(data)} rows")
            
            # Process indices data
            for index_name, data in self.indices_data.items():
                if len(data) > 50:
                    enriched = self.technical_indicators.calculate_all_indicators(data)
                    self.enriched_data[f"INDEX_{index_name}"] = enriched
            
            logger.info(f"Feature engineering completed for {len(self.enriched_data)} instruments")
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def _detect_patterns(self):
        """Detect patterns across all instruments"""
        try:
            # Separate stock and index data
            stock_enriched = {k: v for k, v in self.enriched_data.items() 
                            if not k.startswith('INDEX_')}
            index_enriched = {k.replace('INDEX_', ''): v for k, v in self.enriched_data.items() 
                            if k.startswith('INDEX_')}
            
            self.detected_patterns = self.pattern_detector.detect_all_patterns(
                stock_enriched, index_enriched
            )
            
            # Count detected patterns
            pattern_counts = {}
            for pattern_type, patterns in self.detected_patterns.items():
                if isinstance(patterns, dict):
                    count = sum(len(p) if isinstance(p, list) else 1 for p in patterns.values())
                elif isinstance(patterns, list):
                    count = len(patterns)
                else:
                    count = 1 if patterns else 0
                pattern_counts[pattern_type] = count
            
            logger.info(f"Pattern detection completed: {pattern_counts}")
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {str(e)}")
            raise
    
    def _generate_ml_predictions(self):
        """Generate ML predictions for stocks"""
        try:
            predictions = {}
            
            # Select top 20 stocks with good data quality for ML training
            valid_stocks = [(symbol, data) for symbol, data in self.enriched_data.items() 
                          if not symbol.startswith('INDEX_') and len(data) > 100]
            
            # Sort by data quality and take top 20
            valid_stocks = sorted(valid_stocks, key=lambda x: len(x[1]), reverse=True)[:20]
            
            for symbol, data in valid_stocks:
                try:
                    # Prepare features and target
                    feature_cols = [col for col in data.columns 
                                  if col not in ['Symbol', 'Close'] and 
                                     not data[col].isnull().all()]
                    
                    if len(feature_cols) < 10:  # Minimum feature requirement
                        continue
                    
                    # Clean data
                    clean_data = data[feature_cols + ['Close']].dropna()
                    
                    if len(clean_data) < 60:  # Minimum data for training
                        continue
                    
                    # Split data
                    train_size = int(len(clean_data) * 0.8)
                    X_train = clean_data[feature_cols].iloc[:train_size]
                    y_train = clean_data['Close'].iloc[:train_size]
                    X_val = clean_data[feature_cols].iloc[train_size:]
                    y_val = clean_data['Close'].iloc[train_size:]
                    
                    # Create ensemble predictor
                    ensemble = EnsemblePredictor(self.config)
                    training_results = ensemble.train_ensemble(X_train, y_train, X_val, y_val)
                    
                    # Make prediction for next period
                    latest_features = clean_data[feature_cols].iloc[-1:].copy()
                    prediction = ensemble.predict(latest_features, 'Close')
                    
                    if len(prediction) > 0:
                        current_price = clean_data['Close'].iloc[-1]
                        predicted_price = prediction[0]
                        
                        # Calculate confidence based on validation performance
                        confidence = 0.7  # Base confidence
                        
                        predictions[symbol] = {
                            'predicted_price': float(predicted_price),
                            'current_price': float(current_price),
                            'price_change_pct': float((predicted_price - current_price) / current_price),
                            'confidence': confidence,
                            'horizon': 5,  # 5 days
                            'model_info': {
                                'models': list(ensemble.models.keys()),
                                'training_results': training_results
                            }
                        }
                        
                        logger.info(f"Generated prediction for {symbol}: "
                                  f"{(predicted_price - current_price) / current_price:.2%}")
                    
                except Exception as e:
                    logger.warning(f"Error generating ML prediction for {symbol}: {str(e)}")
                    continue
            
            self.ml_predictions = predictions
            logger.info(f"ML predictions generated for {len(predictions)} stocks")
            
        except Exception as e:
            logger.error(f"Error in ML predictions: {str(e)}")
    
    def _detect_anomalies(self) -> Dict[str, Any]:
        """Detect market anomalies"""
        try:
            anomalies = {}
            
            # Combine all stock data for anomaly detection
            all_features = []
            stock_names = []
            
            for symbol, data in self.enriched_data.items():
                if symbol.startswith('INDEX_'):
                    continue
                    
                # Select numerical features
                feature_cols = [col for col in data.columns 
                              if data[col].dtype in ['float64', 'int64'] and 
                                 not data[col].isnull().all()]
                
                if len(feature_cols) > 10:
                    clean_data = data[feature_cols].dropna()
                    if len(clean_data) > 20:
                        # Take last 20 days
                        recent_data = clean_data.iloc[-20:]
                        all_features.extend(recent_data.values)
                        stock_names.extend([symbol] * len(recent_data))
            
            if len(all_features) > 100:  # Minimum samples for training
                # Train anomaly detector
                feature_array = np.array(all_features)
                self.anomaly_detector.train(feature_array)
                
                # Detect anomalies in recent data
                anomaly_flags, anomaly_scores = self.anomaly_detector.detect_anomalies(feature_array)
                
                # Group by stock
                stock_anomalies = {}
                for i, (is_anomaly, score, stock) in enumerate(zip(anomaly_flags, anomaly_scores, stock_names)):
                    if is_anomaly:
                        if stock not in stock_anomalies:
                            stock_anomalies[stock] = []
                        stock_anomalies[stock].append({
                            'anomaly_score': float(score),
                            'severity': 'high' if score > np.percentile(anomaly_scores, 99) else 'medium'
                        })
                
                anomalies['stock_anomalies'] = stock_anomalies
                anomalies['total_anomalies'] = int(np.sum(anomaly_flags))
                
                logger.info(f"Detected {anomalies['total_anomalies']} anomalies across {len(stock_anomalies)} stocks")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return {}
    
    def _detect_market_regimes(self) -> Dict[str, Any]:
        """Detect current market regime"""
        try:
            regimes = {}
            
            # Analyze major indices
            for index_name, data in self.indices_data.items():
                if len(data) > 50:
                    features = self.regime_detector.prepare_features(data)
                    
                    if len(features) > 20:
                        regime_labels, regime_probs = self.regime_detector.detect_regimes(features)
                        
                        # Get current regime
                        current_regime, regime_name, confidence = self.regime_detector.get_current_regime(features[-1])
                        
                        regimes[index_name] = {
                            'current_regime': int(current_regime),
                            'regime_name': regime_name,
                            'confidence': float(confidence),
                            'regime_history': regime_labels[-10:].tolist()  # Last 10 periods
                        }
                        
                        logger.info(f"{index_name} current regime: {regime_name} (confidence: {confidence:.2%})")
            
            return regimes
            
        except Exception as e:
            logger.error(f"Error in regime detection: {str(e)}")
            return {}
    
    def _generate_trading_signals(self):
        """Generate trading signals"""
        try:
            self.trading_signals = self.signal_generator.generate_all_signals(
                stock_data=self.enriched_data,
                patterns=self.detected_patterns,
                ml_predictions=self.ml_predictions
            )
            
            # Categorize signals by type
            signal_types = {}
            for signal in self.trading_signals:
                signal_type = signal.signal_type.value
                if signal_type not in signal_types:
                    signal_types[signal_type] = 0
                signal_types[signal_type] += 1
            
            logger.info(f"Generated {len(self.trading_signals)} trading signals: {signal_types}")
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
    
    def _compile_results(self, anomalies: Dict, market_regimes: Dict) -> Dict[str, Any]:
        """Compile all analysis results"""
        
        # Market overview
        market_overview = {
            'analysis_timestamp': datetime.now().isoformat(),
            'stocks_analyzed': len([k for k in self.enriched_data.keys() if not k.startswith('INDEX_')]),
            'indices_analyzed': len([k for k in self.enriched_data.keys() if k.startswith('INDEX_')]),
            'patterns_detected': sum(len(p) if isinstance(p, list) else (
                sum(len(v) if isinstance(v, list) else 1 for v in p.values()) if isinstance(p, dict) else 1
            ) for p in self.detected_patterns.values()),
            'ml_predictions_generated': len(self.ml_predictions),
            'trading_signals_generated': len(self.trading_signals),
            'anomalies_detected': anomalies.get('total_anomalies', 0)
        }
        
        # Top opportunities
        top_opportunities = []
        for signal in self.trading_signals[:10]:  # Top 10 signals
            top_opportunities.append({
                'symbol': signal.symbol,
                'signal_type': signal.signal_type.value,
                'strength': signal.strength.value,
                'confidence': signal.confidence,
                'expected_return': ((signal.target_price - signal.entry_price) / signal.entry_price) * 100,
                'risk_reward_ratio': signal.risk_reward_ratio,
                'time_horizon': signal.time_horizon,
                'primary_reason': signal.reasoning[0] if signal.reasoning else 'Pattern detected'
            })
        
        # Pattern summary
        pattern_summary = {}
        for pattern_type, patterns in self.detected_patterns.items():
            if isinstance(patterns, dict):
                pattern_summary[pattern_type] = {
                    'count': len(patterns),
                    'stocks_affected': list(patterns.keys())[:5]  # First 5 stocks
                }
            elif isinstance(patterns, list):
                pattern_summary[pattern_type] = {
                    'count': len(patterns),
                    'details': patterns[:3]  # First 3 patterns
                }
        
        # Sector analysis
        sector_analysis = self.detected_patterns.get('sector_patterns', {})
        
        return {
            'market_overview': market_overview,
            'top_opportunities': top_opportunities,
            'pattern_summary': pattern_summary,
            'sector_analysis': sector_analysis,
            'market_regimes': market_regimes,
            'anomalies': anomalies,
            'all_signals': [signal.to_dict() for signal in self.trading_signals],
            'ml_predictions': self.ml_predictions,
            'full_patterns': self.detected_patterns
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save analysis results to files"""
        try:
            # Create results directory
            os.makedirs('results', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save complete results
            with open(f'results/complete_analysis_{timestamp}.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save trading signals separately
            signals_df = pd.DataFrame([signal.to_dict() for signal in self.trading_signals])
            signals_df.to_csv(f'results/trading_signals_{timestamp}.csv', index=False)
            
            # Save top opportunities
            top_opportunities_df = pd.DataFrame(results['top_opportunities'])
            top_opportunities_df.to_csv(f'results/top_opportunities_{timestamp}.csv', index=False)
            
            # Save market overview
            with open(f'results/market_overview_{timestamp}.json', 'w') as f:
                json.dump(results['market_overview'], f, indent=2, default=str)
            
            logger.info(f"Results saved to results/ directory with timestamp {timestamp}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def _print_analysis_summary(self, results: Dict[str, Any]):
        """Print analysis summary to console"""
        print("\n" + "="*80)
        print("INDIAN STOCK MARKET ANALYSIS SUMMARY")
        print("="*80)
        
        overview = results['market_overview']
        print(f"Analysis Time: {overview['analysis_timestamp']}")
        print(f"Stocks Analyzed: {overview['stocks_analyzed']}")
        print(f"Patterns Detected: {overview['patterns_detected']}")
        print(f"Trading Signals: {overview['trading_signals_generated']}")
        print(f"Anomalies Detected: {overview['anomalies_detected']}")
        
        print("\nTOP TRADING OPPORTUNITIES:")
        print("-" * 50)
        for i, opp in enumerate(results['top_opportunities'][:5], 1):
            print(f"{i}. {opp['symbol']} - {opp['signal_type']}")
            print(f"   Strength: {opp['strength']} | Confidence: {opp['confidence']:.1%}")
            print(f"   Expected Return: {opp['expected_return']:.1f}% | R/R: {opp['risk_reward_ratio']:.1f}")
            print(f"   Reason: {opp['primary_reason']}")
            print()
        
        # Market regimes
        if results['market_regimes']:
            print("MARKET REGIMES:")
            print("-" * 30)
            for index, regime in results['market_regimes'].items():
                print(f"{index}: {regime['regime_name']} ({regime['confidence']:.1%} confidence)")
        
        # Pattern highlights
        print("\nPATTERN HIGHLIGHTS:")
        print("-" * 30)
        for pattern_type, summary in results['pattern_summary'].items():
            if summary.get('count', 0) > 0:
                print(f"{pattern_type.replace('_', ' ').title()}: {summary['count']} detected")
        
        print("\n" + "="*80)
    
    def get_real_time_signals(self, symbol: str = None) -> List[Dict]:
        """Get real-time trading signals for a specific symbol or all symbols"""
        try:
            if symbol:
                return [signal.to_dict() for signal in self.trading_signals 
                       if signal.symbol == symbol]
            else:
                return [signal.to_dict() for signal in self.trading_signals]
        except Exception as e:
            logger.error(f"Error getting real-time signals: {str(e)}")
            return []
    
    def get_pattern_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get detailed pattern analysis for a specific symbol"""
        try:
            symbol_patterns = {}
            
            for pattern_type, patterns in self.detected_patterns.items():
                if isinstance(patterns, dict) and symbol in patterns:
                    symbol_patterns[pattern_type] = patterns[symbol]
            
            return {
                'symbol': symbol,
                'patterns': symbol_patterns,
                'ml_prediction': self.ml_predictions.get(symbol, {}),
                'signals': [s.to_dict() for s in self.trading_signals if s.symbol == symbol]
            }
            
        except Exception as e:
            logger.error(f"Error getting pattern analysis for {symbol}: {str(e)}")
            return {}


def main():
    """Main execution function"""
    print("Starting Indian Stock Market Deep Learning Analysis System...")
    
    try:
        # Initialize system
        system = StockMarketAnalysisSystem()
        
        # Run complete analysis
        results = system.run_complete_analysis(save_results=True)
        
        print("\nAnalysis completed successfully!")
        print(f"Check the 'results/' directory for detailed output files.")
        
        # Example: Get specific analysis
        if results['top_opportunities']:
            top_stock = results['top_opportunities'][0]['symbol']
            pattern_analysis = system.get_pattern_analysis(top_stock)
            print(f"\nDetailed analysis for top opportunity {top_stock}:")
            print(f"Patterns detected: {len(pattern_analysis.get('patterns', {}))}")
            print(f"ML prediction available: {'Yes' if pattern_analysis.get('ml_prediction') else 'No'}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"Analysis failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())