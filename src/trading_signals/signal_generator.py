"""
Trading Signal Generator for Stock Market Analysis
Combines pattern detection, ML predictions, and risk management to generate actionable signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from ..config.config import Config
from ..utils.helpers import calculate_percentage_change, safe_divide


class SignalType(Enum):
    """Enumeration of signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    PAIR_LONG_SHORT = "PAIR_LONG_SHORT"
    PAIR_SHORT_LONG = "PAIR_SHORT_LONG"


class SignalStrength(Enum):
    """Enumeration of signal strengths"""
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"


@dataclass
class TradingSignal:
    """Data class for trading signals"""
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    position_size: float
    timestamp: datetime
    reasoning: List[str]
    supporting_indicators: Dict[str, Any]
    time_horizon: str  # 'intraday', 'swing', 'positional'
    sector: str = None
    correlation_pairs: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'risk_reward_ratio': self.risk_reward_ratio,
            'position_size': self.position_size,
            'timestamp': self.timestamp.isoformat(),
            'reasoning': self.reasoning,
            'supporting_indicators': self.supporting_indicators,
            'time_horizon': self.time_horizon,
            'sector': self.sector,
            'correlation_pairs': self.correlation_pairs
        }


class SignalGenerator:
    """Generates trading signals based on patterns, ML predictions, and market analysis"""
    
    def __init__(self):
        self.config = Config()
        self.signal_params = self.config.SIGNAL_PARAMS
        
    def generate_all_signals(self, 
                            stock_data: Dict[str, pd.DataFrame],
                            patterns: Dict[str, Any],
                            ml_predictions: Dict[str, Any] = None,
                            market_sentiment: Dict[str, Any] = None) -> List[TradingSignal]:
        """
        Generate all trading signals combining multiple analysis methods
        
        Args:
            stock_data: Dictionary of stock DataFrames with technical indicators
            patterns: Detected patterns from pattern detection module
            ml_predictions: ML model predictions
            market_sentiment: Market sentiment analysis results
            
        Returns:
            List of trading signals
        """
        all_signals = []
        
        try:
            # 1. Technical Pattern Signals
            logger.info("Generating technical pattern signals...")
            pattern_signals = self._generate_pattern_signals(stock_data, patterns)
            all_signals.extend(pattern_signals)
            
            # 2. Breakout Signals
            logger.info("Generating breakout signals...")
            breakout_signals = self._generate_breakout_signals(stock_data, patterns)
            all_signals.extend(breakout_signals)
            
            # 3. Volume Anomaly Signals
            logger.info("Generating volume anomaly signals...")
            volume_signals = self._generate_volume_signals(stock_data, patterns)
            all_signals.extend(volume_signals)
            
            # 4. Pair Trading Signals
            logger.info("Generating pair trading signals...")
            pair_signals = self._generate_pair_signals(stock_data, patterns)
            all_signals.extend(pair_signals)
            
            # 5. ML-based Signals
            if ml_predictions:
                logger.info("Generating ML-based signals...")
                ml_signals = self._generate_ml_signals(stock_data, ml_predictions)
                all_signals.extend(ml_signals)
            
            # 6. Sector Rotation Signals
            logger.info("Generating sector rotation signals...")
            sector_signals = self._generate_sector_signals(stock_data, patterns)
            all_signals.extend(sector_signals)
            
            # 7. Multi-timeframe Signals
            logger.info("Generating multi-timeframe signals...")
            mtf_signals = self._generate_multi_timeframe_signals(stock_data)
            all_signals.extend(mtf_signals)
            
            # 8. Filter and rank signals
            filtered_signals = self._filter_and_rank_signals(all_signals, stock_data)
            
            logger.info(f"Generated {len(filtered_signals)} trading signals")
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            return []
    
    def _generate_pattern_signals(self, stock_data: Dict[str, pd.DataFrame], 
                                 patterns: Dict[str, Any]) -> List[TradingSignal]:
        """Generate signals based on technical patterns"""
        signals = []
        
        technical_patterns = patterns.get('technical_patterns', {})
        
        for symbol, symbol_patterns in technical_patterns.items():
            if symbol not in stock_data:
                continue
                
            df = stock_data[symbol]
            current_price = df['Close'].iloc[-1]
            
            for pattern in symbol_patterns:
                try:
                    signal = self._create_pattern_signal(symbol, pattern, current_price, df)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.warning(f"Error creating pattern signal for {symbol}: {str(e)}")
                    continue
        
        return signals
    
    def _create_pattern_signal(self, symbol: str, pattern: Dict, 
                              current_price: float, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Create a trading signal from a detected pattern"""
        
        pattern_type = pattern.get('type')
        confidence = pattern.get('confidence', 0.5)
        
        # Skip low confidence patterns
        if confidence < self.signal_params['confidence_threshold']:
            return None
        
        reasoning = [f"Detected {pattern_type} pattern"]
        supporting_indicators = {'pattern': pattern}
        
        # Determine signal direction and targets based on pattern type
        if pattern_type in ['double_bottom', 'hammer', 'bullish_divergence', 'ma_alignment']:
            if pattern.get('direction') == 'bearish':
                signal_type = SignalType.SELL
                target_price = current_price * 0.95
                stop_loss = current_price * 1.02
            else:
                signal_type = SignalType.BUY
                target_price = current_price * 1.05
                stop_loss = current_price * 0.98
                
        elif pattern_type in ['double_top', 'shooting_star', 'bearish_divergence']:
            signal_type = SignalType.SELL
            target_price = current_price * 0.95
            stop_loss = current_price * 1.02
            
        elif pattern_type in ['triangle', 'flag', 'bollinger_squeeze']:
            # Breakout patterns - determine direction from additional context
            if df['Close'].iloc[-1] > df['SMA_20'].iloc[-1]:
                signal_type = SignalType.BUY
                target_price = current_price * 1.05
                stop_loss = current_price * 0.98
            else:
                signal_type = SignalType.SELL
                target_price = current_price * 0.95
                stop_loss = current_price * 1.02
        else:
            return None
        
        # Calculate risk-reward ratio
        risk = abs(current_price - stop_loss)
        reward = abs(target_price - current_price)
        risk_reward_ratio = safe_divide(reward, risk, 0)
        
        # Skip signals with poor risk-reward ratio
        if risk_reward_ratio < self.signal_params['risk_reward_ratio']:
            return None
        
        # Determine signal strength
        strength = self._calculate_signal_strength(confidence, risk_reward_ratio, pattern)
        
        # Calculate position size
        position_size = self._calculate_position_size(risk, current_price)
        
        # Add technical indicator support
        self._add_technical_support(df, supporting_indicators, reasoning)
        
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward_ratio,
            position_size=position_size,
            timestamp=datetime.now(),
            reasoning=reasoning,
            supporting_indicators=supporting_indicators,
            time_horizon=self._determine_time_horizon(pattern_type)
        )
    
    def _generate_breakout_signals(self, stock_data: Dict[str, pd.DataFrame], 
                                  patterns: Dict[str, Any]) -> List[TradingSignal]:
        """Generate signals based on breakout patterns"""
        signals = []
        
        breakout_signals = patterns.get('breakout_signals', {})
        
        for symbol, symbol_breakouts in breakout_signals.items():
            if symbol not in stock_data:
                continue
                
            df = stock_data[symbol]
            current_price = df['Close'].iloc[-1]
            
            for breakout in symbol_breakouts:
                try:
                    # Volume confirmation
                    volume_confirmation = breakout.get('volume_confirmation', False)
                    if not volume_confirmation and df['Volume_Ratio'].iloc[-1] < 1.2:
                        continue  # Skip breakouts without volume confirmation
                    
                    signal_type = SignalType.BUY if breakout['direction'] == 'bullish' else SignalType.SELL
                    
                    # Calculate targets based on breakout type
                    if breakout['type'] == 'range_breakout':
                        range_size = breakout.get('range_size', 0.05)
                        if signal_type == SignalType.BUY:
                            target_price = current_price * (1 + range_size)
                            stop_loss = breakout['breakout_level'] * 0.99
                        else:
                            target_price = current_price * (1 - range_size)
                            stop_loss = breakout['breakout_level'] * 1.01
                    else:
                        if signal_type == SignalType.BUY:
                            target_price = current_price * 1.05
                            stop_loss = current_price * 0.98
                        else:
                            target_price = current_price * 0.95
                            stop_loss = current_price * 1.02
                    
                    risk = abs(current_price - stop_loss)
                    reward = abs(target_price - current_price)
                    risk_reward_ratio = safe_divide(reward, risk, 0)
                    
                    if risk_reward_ratio < self.signal_params['risk_reward_ratio']:
                        continue
                    
                    confidence = breakout.get('confidence', 0.7)
                    if volume_confirmation:
                        confidence += 0.1  # Boost confidence for volume confirmation
                    
                    reasoning = [
                        f"Breakout detected: {breakout['type']}",
                        f"Direction: {breakout['direction']}",
                        f"Breakout level: {breakout['breakout_level']:.2f}"
                    ]
                    
                    if volume_confirmation:
                        reasoning.append("Volume confirmation present")
                    
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=signal_type,
                        strength=self._calculate_signal_strength(confidence, risk_reward_ratio, breakout),
                        confidence=min(confidence, 0.95),
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        risk_reward_ratio=risk_reward_ratio,
                        position_size=self._calculate_position_size(risk, current_price),
                        timestamp=datetime.now(),
                        reasoning=reasoning,
                        supporting_indicators={'breakout': breakout},
                        time_horizon='swing'
                    ))
                    
                except Exception as e:
                    logger.warning(f"Error creating breakout signal for {symbol}: {str(e)}")
                    continue
        
        return signals
    
    def _generate_volume_signals(self, stock_data: Dict[str, pd.DataFrame], 
                                patterns: Dict[str, Any]) -> List[TradingSignal]:
        """Generate signals based on volume anomalies"""
        signals = []
        
        volume_anomalies = patterns.get('volume_anomalies', {})
        
        for symbol, anomalies in volume_anomalies.items():
            if symbol not in stock_data:
                continue
                
            df = stock_data[symbol]
            current_price = df['Close'].iloc[-1]
            
            for anomaly in anomalies:
                try:
                    anomaly_type = anomaly.get('type')
                    signal_type = anomaly.get('signal')
                    
                    if anomaly_type == 'volume_spike_compression':
                        # High probability breakout setup
                        direction = 'bullish' if df['Close'].iloc[-1] > df['SMA_20'].iloc[-1] else 'bearish'
                        
                        if direction == 'bullish':
                            target_price = current_price * 1.06
                            stop_loss = current_price * 0.97
                            signal_type_enum = SignalType.BUY
                        else:
                            target_price = current_price * 0.94
                            stop_loss = current_price * 1.03
                            signal_type_enum = SignalType.SELL
                        
                        reasoning = [
                            "Volume spike with price compression detected",
                            "High probability breakout setup",
                            f"Expected direction: {direction}"
                        ]
                        
                    elif anomaly_type == 'low_volume_squeeze':
                        # Coiling pattern - wait for direction
                        continue  # Skip for now, need direction confirmation
                        
                    elif anomaly_type == 'volume_climax':
                        # Exhaustion move - reversal signal
                        if anomaly.get('price_change', 0) > 0:
                            signal_type_enum = SignalType.SELL
                            target_price = current_price * 0.96
                            stop_loss = current_price * 1.02
                        else:
                            signal_type_enum = SignalType.BUY
                            target_price = current_price * 1.04
                            stop_loss = current_price * 0.98
                        
                        reasoning = [
                            "Volume climax detected",
                            "Potential exhaustion move",
                            "Reversal opportunity"
                        ]
                        
                    elif anomaly_type in ['bullish_divergence', 'bearish_divergence']:
                        if anomaly_type == 'bullish_divergence':
                            signal_type_enum = SignalType.BUY
                            target_price = current_price * 1.04
                            stop_loss = current_price * 0.98
                        else:
                            signal_type_enum = SignalType.SELL
                            target_price = current_price * 0.96
                            stop_loss = current_price * 1.02
                        
                        reasoning = [
                            f"{anomaly_type.replace('_', ' ').title()} detected",
                            "Price-volume divergence",
                            "Momentum shift expected"
                        ]
                    else:
                        continue
                    
                    risk = abs(current_price - stop_loss)
                    reward = abs(target_price - current_price)
                    risk_reward_ratio = safe_divide(reward, risk, 0)
                    
                    if risk_reward_ratio < 1.5:  # Lower threshold for volume signals
                        continue
                    
                    confidence = anomaly.get('confidence', 0.7)
                    
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=signal_type_enum,
                        strength=self._calculate_signal_strength(confidence, risk_reward_ratio, anomaly),
                        confidence=confidence,
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        risk_reward_ratio=risk_reward_ratio,
                        position_size=self._calculate_position_size(risk, current_price),
                        timestamp=datetime.now(),
                        reasoning=reasoning,
                        supporting_indicators={'volume_anomaly': anomaly},
                        time_horizon='swing'
                    ))
                    
                except Exception as e:
                    logger.warning(f"Error creating volume signal for {symbol}: {str(e)}")
                    continue
        
        return signals
    
    def _generate_pair_signals(self, stock_data: Dict[str, pd.DataFrame], 
                              patterns: Dict[str, Any]) -> List[TradingSignal]:
        """Generate pair trading signals"""
        signals = []
        
        correlation_patterns = patterns.get('correlation_patterns', {})
        pair_opportunities = correlation_patterns.get('pair_opportunities', [])
        
        for opportunity in pair_opportunities:
            try:
                stock1, stock2 = opportunity['pair']
                
                if stock1 not in stock_data or stock2 not in stock_data:
                    continue
                
                correlation = opportunity['correlation']
                spread_zscore = opportunity['spread_zscore']
                signal_direction = opportunity['signal']
                confidence = opportunity['confidence']
                
                current_price1 = stock_data[stock1]['Close'].iloc[-1]
                current_price2 = stock_data[stock2]['Close'].iloc[-1]
                
                # Determine position sizes (equal dollar amounts)
                total_position = 0.02  # 2% of portfolio
                position1 = total_position / 2
                position2 = total_position / 2
                
                if signal_direction == 'long_short':
                    # Long stock1, short stock2
                    target_spread_zscore = 0  # Expect mean reversion
                    
                    reasoning = [
                        f"Pair trade opportunity: Long {stock1}, Short {stock2}",
                        f"Current spread Z-score: {spread_zscore:.2f}",
                        f"Correlation: {correlation:.3f}",
                        "Expected mean reversion"
                    ]
                    
                    # Create two signals for the pair
                    signals.extend([
                        TradingSignal(
                            symbol=stock1,
                            signal_type=SignalType.PAIR_LONG_SHORT,
                            strength=self._calculate_signal_strength(confidence, 2.0, opportunity),
                            confidence=confidence,
                            entry_price=current_price1,
                            target_price=current_price1 * 1.03,
                            stop_loss=current_price1 * 0.95,
                            risk_reward_ratio=2.0,
                            position_size=position1,
                            timestamp=datetime.now(),
                            reasoning=reasoning,
                            supporting_indicators={'pair_trade': opportunity},
                            time_horizon='swing',
                            correlation_pairs=[stock2]
                        ),
                        TradingSignal(
                            symbol=stock2,
                            signal_type=SignalType.PAIR_SHORT_LONG,
                            strength=self._calculate_signal_strength(confidence, 2.0, opportunity),
                            confidence=confidence,
                            entry_price=current_price2,
                            target_price=current_price2 * 0.97,
                            stop_loss=current_price2 * 1.05,
                            risk_reward_ratio=2.0,
                            position_size=position2,
                            timestamp=datetime.now(),
                            reasoning=reasoning,
                            supporting_indicators={'pair_trade': opportunity},
                            time_horizon='swing',
                            correlation_pairs=[stock1]
                        )
                    ])
                
                elif signal_direction == 'short_long':
                    # Short stock1, long stock2
                    reasoning = [
                        f"Pair trade opportunity: Short {stock1}, Long {stock2}",
                        f"Current spread Z-score: {spread_zscore:.2f}",
                        f"Correlation: {correlation:.3f}",
                        "Expected mean reversion"
                    ]
                    
                    signals.extend([
                        TradingSignal(
                            symbol=stock1,
                            signal_type=SignalType.PAIR_SHORT_LONG,
                            strength=self._calculate_signal_strength(confidence, 2.0, opportunity),
                            confidence=confidence,
                            entry_price=current_price1,
                            target_price=current_price1 * 0.97,
                            stop_loss=current_price1 * 1.05,
                            risk_reward_ratio=2.0,
                            position_size=position1,
                            timestamp=datetime.now(),
                            reasoning=reasoning,
                            supporting_indicators={'pair_trade': opportunity},
                            time_horizon='swing',
                            correlation_pairs=[stock2]
                        ),
                        TradingSignal(
                            symbol=stock2,
                            signal_type=SignalType.PAIR_LONG_SHORT,
                            strength=self._calculate_signal_strength(confidence, 2.0, opportunity),
                            confidence=confidence,
                            entry_price=current_price2,
                            target_price=current_price2 * 1.03,
                            stop_loss=current_price2 * 0.95,
                            risk_reward_ratio=2.0,
                            position_size=position2,
                            timestamp=datetime.now(),
                            reasoning=reasoning,
                            supporting_indicators={'pair_trade': opportunity},
                            time_horizon='swing',
                            correlation_pairs=[stock1]
                        )
                    ])
                
            except Exception as e:
                logger.warning(f"Error creating pair signal: {str(e)}")
                continue
        
        return signals
    
    def _generate_ml_signals(self, stock_data: Dict[str, pd.DataFrame], 
                            ml_predictions: Dict[str, Any]) -> List[TradingSignal]:
        """Generate signals based on ML model predictions"""
        signals = []
        
        for symbol, predictions in ml_predictions.items():
            if symbol not in stock_data:
                continue
                
            try:
                df = stock_data[symbol]
                current_price = df['Close'].iloc[-1]
                
                # Extract prediction information
                predicted_price = predictions.get('predicted_price')
                confidence = predictions.get('confidence', 0.5)
                prediction_horizon = predictions.get('horizon', 5)  # days
                
                if predicted_price is None or confidence < self.signal_params['confidence_threshold']:
                    continue
                
                price_change_pct = (predicted_price - current_price) / current_price
                
                # Generate signal based on predicted direction
                if abs(price_change_pct) > 0.02:  # At least 2% move predicted
                    if price_change_pct > 0:
                        signal_type = SignalType.BUY
                        target_price = predicted_price
                        stop_loss = current_price * 0.97
                    else:
                        signal_type = SignalType.SELL
                        target_price = predicted_price
                        stop_loss = current_price * 1.03
                    
                    risk = abs(current_price - stop_loss)
                    reward = abs(target_price - current_price)
                    risk_reward_ratio = safe_divide(reward, risk, 0)
                    
                    if risk_reward_ratio >= 1.5:  # Lower threshold for ML signals
                        reasoning = [
                            f"ML model prediction: {price_change_pct:.2%} move expected",
                            f"Prediction horizon: {prediction_horizon} days",
                            f"Model confidence: {confidence:.2%}"
                        ]
                        
                        # Add model details
                        model_info = predictions.get('model_info', {})
                        if model_info:
                            reasoning.append(f"Models used: {', '.join(model_info.get('models', []))}")
                        
                        signals.append(TradingSignal(
                            symbol=symbol,
                            signal_type=signal_type,
                            strength=self._calculate_signal_strength(confidence, risk_reward_ratio, predictions),
                            confidence=confidence,
                            entry_price=current_price,
                            target_price=target_price,
                            stop_loss=stop_loss,
                            risk_reward_ratio=risk_reward_ratio,
                            position_size=self._calculate_position_size(risk, current_price),
                            timestamp=datetime.now(),
                            reasoning=reasoning,
                            supporting_indicators={'ml_prediction': predictions},
                            time_horizon='positional' if prediction_horizon > 10 else 'swing'
                        ))
                
            except Exception as e:
                logger.warning(f"Error creating ML signal for {symbol}: {str(e)}")
                continue
        
        return signals
    
    def _generate_sector_signals(self, stock_data: Dict[str, pd.DataFrame], 
                                patterns: Dict[str, Any]) -> List[TradingSignal]:
        """Generate signals based on sector rotation patterns"""
        signals = []
        
        sector_patterns = patterns.get('sector_patterns', {})
        
        for sector, sector_analysis in sector_patterns.items():
            if not sector_analysis:
                continue
                
            try:
                leaders = sector_analysis.get('leaders', [])
                laggards = sector_analysis.get('laggards', [])
                sector_momentum = sector_analysis.get('sector_momentum', 0)
                rotation_signal = sector_analysis.get('rotation_signal')
                
                # Generate signals for sector leaders if sector has positive momentum
                if sector_momentum > 0.01 and leaders:  # 1% positive momentum
                    for leader_symbol, performance in leaders:
                        if leader_symbol in stock_data:
                            df = stock_data[leader_symbol]
                            current_price = df['Close'].iloc[-1]
                            
                            target_price = current_price * 1.05
                            stop_loss = current_price * 0.97
                            
                            risk = current_price - stop_loss
                            reward = target_price - current_price
                            risk_reward_ratio = safe_divide(reward, risk, 0)
                            
                            if risk_reward_ratio >= 2.0:
                                reasoning = [
                                    f"Sector leader in {sector}",
                                    f"Sector momentum: {sector_momentum:.2%}",
                                    f"Relative performance: {performance:.2%}",
                                    "Sector rotation opportunity"
                                ]
                                
                                if rotation_signal:
                                    reasoning.append(f"Rotation signal: {rotation_signal}")
                                
                                signals.append(TradingSignal(
                                    symbol=leader_symbol,
                                    signal_type=SignalType.BUY,
                                    strength=SignalStrength.MODERATE,
                                    confidence=0.7,
                                    entry_price=current_price,
                                    target_price=target_price,
                                    stop_loss=stop_loss,
                                    risk_reward_ratio=risk_reward_ratio,
                                    position_size=self._calculate_position_size(risk, current_price),
                                    timestamp=datetime.now(),
                                    reasoning=reasoning,
                                    supporting_indicators={'sector_analysis': sector_analysis},
                                    time_horizon='swing',
                                    sector=sector
                                ))
                
            except Exception as e:
                logger.warning(f"Error creating sector signal for {sector}: {str(e)}")
                continue
        
        return signals
    
    def _generate_multi_timeframe_signals(self, stock_data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Generate signals based on multi-timeframe analysis"""
        signals = []
        
        # This would typically involve analyzing multiple timeframes
        # For now, we'll implement a simple version based on trend alignment
        
        for symbol, df in stock_data.items():
            try:
                if len(df) < 50:
                    continue
                
                current_price = df['Close'].iloc[-1]
                
                # Check trend alignment across different periods
                sma_5 = df['SMA_5'].iloc[-1]
                sma_20 = df['SMA_20'].iloc[-1]
                sma_50 = df['SMA_50'].iloc[-1]
                
                # Strong bullish alignment
                if sma_5 > sma_20 > sma_50 and current_price > sma_5:
                    # Check if RSI is not overbought
                    rsi = df['RSI'].iloc[-1]
                    if rsi < 70:
                        target_price = current_price * 1.06
                        stop_loss = sma_20 * 0.98
                        
                        risk = current_price - stop_loss
                        reward = target_price - current_price
                        risk_reward_ratio = safe_divide(reward, risk, 0)
                        
                        if risk_reward_ratio >= 2.0:
                            reasoning = [
                                "Multi-timeframe bullish alignment",
                                f"Price above all major MAs",
                                f"RSI not overbought: {rsi:.1f}",
                                "Trend continuation expected"
                            ]
                            
                            signals.append(TradingSignal(
                                symbol=symbol,
                                signal_type=SignalType.BUY,
                                strength=SignalStrength.STRONG,
                                confidence=0.8,
                                entry_price=current_price,
                                target_price=target_price,
                                stop_loss=stop_loss,
                                risk_reward_ratio=risk_reward_ratio,
                                position_size=self._calculate_position_size(risk, current_price),
                                timestamp=datetime.now(),
                                reasoning=reasoning,
                                supporting_indicators={'mtf_analysis': {
                                    'sma_5': sma_5, 'sma_20': sma_20, 'sma_50': sma_50,
                                    'rsi': rsi
                                }},
                                time_horizon='positional'
                            ))
                
                # Strong bearish alignment
                elif sma_5 < sma_20 < sma_50 and current_price < sma_5:
                    rsi = df['RSI'].iloc[-1]
                    if rsi > 30:
                        target_price = current_price * 0.94
                        stop_loss = sma_20 * 1.02
                        
                        risk = stop_loss - current_price
                        reward = current_price - target_price
                        risk_reward_ratio = safe_divide(reward, risk, 0)
                        
                        if risk_reward_ratio >= 2.0:
                            reasoning = [
                                "Multi-timeframe bearish alignment",
                                f"Price below all major MAs",
                                f"RSI not oversold: {rsi:.1f}",
                                "Trend continuation expected"
                            ]
                            
                            signals.append(TradingSignal(
                                symbol=symbol,
                                signal_type=SignalType.SELL,
                                strength=SignalStrength.STRONG,
                                confidence=0.8,
                                entry_price=current_price,
                                target_price=target_price,
                                stop_loss=stop_loss,
                                risk_reward_ratio=risk_reward_ratio,
                                position_size=self._calculate_position_size(risk, current_price),
                                timestamp=datetime.now(),
                                reasoning=reasoning,
                                supporting_indicators={'mtf_analysis': {
                                    'sma_5': sma_5, 'sma_20': sma_20, 'sma_50': sma_50,
                                    'rsi': rsi
                                }},
                                time_horizon='positional'
                            ))
                
            except Exception as e:
                logger.warning(f"Error creating MTF signal for {symbol}: {str(e)}")
                continue
        
        return signals
    
    def _calculate_signal_strength(self, confidence: float, risk_reward_ratio: float, 
                                  context: Dict) -> SignalStrength:
        """Calculate signal strength based on multiple factors"""
        
        # Base score from confidence
        score = confidence
        
        # Boost for good risk-reward ratio
        if risk_reward_ratio >= 3.0:
            score += 0.2
        elif risk_reward_ratio >= 2.0:
            score += 0.1
        
        # Context-specific boosts
        if isinstance(context, dict):
            # Volume confirmation
            if context.get('volume_confirmation', False):
                score += 0.1
                
            # Pattern-specific boosts
            pattern_type = context.get('type', '')
            if pattern_type in ['double_bottom', 'double_top', 'triangle']:
                score += 0.1
            
            # ML prediction boost
            if 'ml_prediction' in str(context):
                score += 0.05
        
        # Convert to enum
        if score >= 0.9:
            return SignalStrength.VERY_STRONG
        elif score >= 0.8:
            return SignalStrength.STRONG
        elif score >= 0.7:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _calculate_position_size(self, risk_amount: float, entry_price: float) -> float:
        """Calculate position size based on risk management"""
        
        # Risk per trade as percentage of portfolio
        risk_per_trade = 0.02  # 2%
        
        # Assume portfolio value (this would come from portfolio manager)
        portfolio_value = 1000000  # 10 lakhs
        
        # Maximum risk amount
        max_risk = portfolio_value * risk_per_trade
        
        # Position size
        position_size = min(max_risk / risk_amount, self.signal_params['max_position_size'])
        
        return round(position_size, 4)
    
    def _determine_time_horizon(self, pattern_type: str) -> str:
        """Determine time horizon based on pattern type"""
        
        short_term_patterns = ['doji', 'hammer', 'engulfing', 'gap']
        medium_term_patterns = ['triangle', 'flag', 'breakout', 'squeeze']
        long_term_patterns = ['double_top', 'double_bottom', 'head_shoulders', 'trend']
        
        if pattern_type in short_term_patterns:
            return 'intraday'
        elif pattern_type in medium_term_patterns:
            return 'swing'
        else:
            return 'positional'
    
    def _add_technical_support(self, df: pd.DataFrame, supporting_indicators: Dict, 
                              reasoning: List[str]):
        """Add technical indicator support to signal"""
        
        try:
            # RSI support
            rsi = df['RSI'].iloc[-1]
            if rsi < 30:
                reasoning.append(f"RSI oversold: {rsi:.1f}")
                supporting_indicators['rsi_oversold'] = True
            elif rsi > 70:
                reasoning.append(f"RSI overbought: {rsi:.1f}")
                supporting_indicators['rsi_overbought'] = True
            
            # MACD support
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_Signal'].iloc[-1]
            if macd > macd_signal:
                reasoning.append("MACD bullish crossover")
                supporting_indicators['macd_bullish'] = True
            elif macd < macd_signal:
                reasoning.append("MACD bearish crossover")
                supporting_indicators['macd_bearish'] = True
            
            # Volume support
            volume_ratio = df['Volume_Ratio'].iloc[-1]
            if volume_ratio > 1.5:
                reasoning.append(f"Above average volume: {volume_ratio:.1f}x")
                supporting_indicators['high_volume'] = True
            
            # Bollinger Bands support
            bb_position = df['BB_Position'].iloc[-1]
            if bb_position > 0.8:
                reasoning.append("Near upper Bollinger Band")
                supporting_indicators['bb_upper'] = True
            elif bb_position < 0.2:
                reasoning.append("Near lower Bollinger Band")
                supporting_indicators['bb_lower'] = True
            
        except Exception as e:
            logger.warning(f"Error adding technical support: {str(e)}")
    
    def _filter_and_rank_signals(self, signals: List[TradingSignal], 
                                 stock_data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Filter and rank signals by quality"""
        
        # Filter out weak signals
        filtered_signals = [s for s in signals if s.strength != SignalStrength.WEAK]
        
        # Remove duplicate signals for the same symbol (keep the best one)
        symbol_signals = {}
        for signal in filtered_signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = signal
            else:
                # Keep the signal with higher confidence
                if signal.confidence > symbol_signals[signal.symbol].confidence:
                    symbol_signals[signal.symbol] = signal
        
        # Convert back to list
        unique_signals = list(symbol_signals.values())
        
        # Sort by combination of strength and confidence
        strength_values = {
            SignalStrength.VERY_STRONG: 4,
            SignalStrength.STRONG: 3,
            SignalStrength.MODERATE: 2,
            SignalStrength.WEAK: 1
        }
        
        unique_signals.sort(
            key=lambda s: (strength_values[s.strength], s.confidence, s.risk_reward_ratio),
            reverse=True
        )
        
        # Return top signals (limit to prevent overtrading)
        max_signals = 20
        return unique_signals[:max_signals]