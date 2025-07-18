"""
🚀 REAL NIFTY 50 ANALYSIS - JULY 18, 2025 🚀
Using ACTUAL MARKET DATA provided by user
5-Minute Intervals with Technical Indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime

def analyze_real_nifty_data():
    """Analyze the real NIFTY 50 data provided by user"""
    
    print("📊 ANALYZING REAL NIFTY 50 DATA")
    print("=" * 60)
    
    # Real data from user (latest entries)
    real_data = {
        'current_price': 24938.90,  # Latest close
        'current_time': "12:55 PM July 18, 2025",
        'day_open': 25108.55,  # July 18 opening
        'day_high': 25144.60,  # July 18 high so far
        'day_low': 24918.65,   # July 18 low so far
        'prev_close': 25109.50, # July 17 close
        'williams_r': -54.90,
        'macd': -13.45,
        'macd_signal': -17.21,
        'macd_hist': 3.77
    }
    
    # Calculate real metrics
    day_change = real_data['current_price'] - real_data['prev_close']
    day_change_pct = (day_change / real_data['prev_close']) * 100
    day_range = real_data['day_high'] - real_data['day_low']
    gap = real_data['day_open'] - real_data['prev_close']
    gap_pct = (gap / real_data['prev_close']) * 100
    
    print("📈 CURRENT MARKET STATUS (REAL DATA)")
    print("-" * 45)
    print(f"Current NIFTY: {real_data['current_price']:.2f}")
    print(f"Time: {real_data['current_time']}")
    print(f"Day Open: {real_data['day_open']:.2f}")
    print(f"Day High: {real_data['day_high']:.2f}")
    print(f"Day Low: {real_data['day_low']:.2f}")
    print(f"Previous Close: {real_data['prev_close']:.2f}")
    print(f"Day Change: {day_change:+.2f} ({day_change_pct:+.2f}%)")
    print(f"Day Range: {day_range:.2f} points")
    print(f"Gap: {gap:+.2f} ({gap_pct:+.2f}%)")
    
    # Technical Analysis
    print(f"\n🔧 TECHNICAL INDICATORS (REAL)")
    print("-" * 35)
    print(f"Williams %R: {real_data['williams_r']:.2f}")
    print(f"MACD: {real_data['macd']:.2f}")
    print(f"MACD Signal: {real_data['macd_signal']:.2f}")
    print(f"MACD Histogram: {real_data['macd_hist']:.2f}")
    
    # Market Analysis
    print(f"\n🎯 MARKET ANALYSIS")
    print("-" * 25)
    
    # Gap Analysis
    if gap_pct < -0.5:
        gap_status = f"GAP DOWN: {gap:+.2f} ({gap_pct:+.2f}%)"
    elif gap_pct > 0.5:
        gap_status = f"GAP UP: {gap:+.2f} ({gap_pct:+.2f}%)"
    else:
        gap_status = f"NO GAP: {gap:+.2f} ({gap_pct:+.2f}%)"
    
    print(f"Gap Status: {gap_status}")
    
    # Intraday Performance
    intraday_move = real_data['current_price'] - real_data['day_open']
    intraday_pct = (intraday_move / real_data['day_open']) * 100
    print(f"From Open: {intraday_move:+.2f} ({intraday_pct:+.2f}%)")
    
    # Position in Range
    position_in_range = (real_data['current_price'] - real_data['day_low']) / day_range
    if position_in_range > 0.8:
        range_pos = "NEAR DAY HIGH"
    elif position_in_range > 0.6:
        range_pos = "UPPER RANGE"
    elif position_in_range > 0.4:
        range_pos = "MIDDLE RANGE"
    elif position_in_range > 0.2:
        range_pos = "LOWER RANGE"
    else:
        range_pos = "NEAR DAY LOW"
    
    print(f"Range Position: {range_pos} ({position_in_range:.1%})")
    
    # Technical Signal Analysis
    williams_signal = "OVERSOLD" if real_data['williams_r'] < -80 else "OVERBOUGHT" if real_data['williams_r'] > -20 else "NEUTRAL"
    macd_signal = "BULLISH" if real_data['macd'] > real_data['macd_signal'] else "BEARISH"
    macd_momentum = "IMPROVING" if real_data['macd_hist'] > 0 else "WEAKENING"
    
    print(f"Williams %R: {williams_signal}")
    print(f"MACD: {macd_signal} & {macd_momentum}")
    
    # Calculate Key Levels
    current_price = real_data['current_price']
    
    # Support/Resistance
    immediate_support = real_data['day_low']
    immediate_resistance = real_data['day_high']
    
    # Options strikes
    atm_strike = round(current_price / 50) * 50
    
    # Psychological levels
    psychological_levels = [24800, 24850, 24900, 24950, 25000, 25050, 25100, 25150, 25200]
    
    print(f"\n📊 KEY TRADING LEVELS")
    print("-" * 25)
    print(f"Immediate Support: {immediate_support:.2f}")
    print(f"Immediate Resistance: {immediate_resistance:.2f}")
    print(f"ATM Strike: {atm_strike}")
    print(f"Day's Range: {immediate_support:.0f} - {immediate_resistance:.0f}")
    
    # Options levels
    call_strikes = [atm_strike + i*50 for i in range(1, 4)]
    put_strikes = [atm_strike - i*50 for i in range(1, 4)]
    
    print(f"\n📈 OPTIONS LEVELS")
    print("-" * 20)
    print(f"ATM: {atm_strike}")
    print(f"Calls: {call_strikes}")
    print(f"Puts: {put_strikes}")
    
    # Market Bias Assessment
    bias_factors = []
    
    # Day performance
    if day_change_pct < -0.5:
        bias_factors.append(("Day Weak", -1))
    elif day_change_pct > 0.5:
        bias_factors.append(("Day Strong", +1))
    
    # Technical indicators
    if williams_signal == "OVERSOLD":
        bias_factors.append(("Williams Oversold", +1))
    elif williams_signal == "OVERBOUGHT":
        bias_factors.append(("Williams Overbought", -1))
    
    if macd_signal == "BULLISH":
        bias_factors.append(("MACD Bullish", +1))
    else:
        bias_factors.append(("MACD Bearish", -1))
    
    if macd_momentum == "IMPROVING":
        bias_factors.append(("MACD Improving", +1))
    else:
        bias_factors.append(("MACD Weakening", -1))
    
    # Range position
    if position_in_range < 0.3:
        bias_factors.append(("Near Low", +1))
    elif position_in_range > 0.7:
        bias_factors.append(("Near High", -1))
    
    total_bias = sum([factor[1] for factor in bias_factors])
    
    print(f"\n🎯 BIAS ANALYSIS")
    print("-" * 20)
    for factor, weight in bias_factors:
        direction = "BULLISH" if weight > 0 else "BEARISH" if weight < 0 else "NEUTRAL"
        print(f"• {factor}: {direction}")
    
    print(f"\nOverall Bias Score: {total_bias}")
    if total_bias >= 2:
        overall_bias = "BULLISH"
    elif total_bias <= -2:
        overall_bias = "BEARISH"
    else:
        overall_bias = "NEUTRAL"
    
    print(f"Overall Bias: {overall_bias}")
    
    # Predictions for remaining session
    remaining_hours = 2.5  # Approx time left in session
    
    # Expected volatility based on current range
    expected_move = min(100, day_range * 0.3)  # Conservative estimate
    
    upside_target = current_price + expected_move
    downside_target = current_price - expected_move
    
    print(f"\n🎯 REMAINING SESSION PREDICTIONS")
    print("-" * 40)
    print(f"Time Remaining: ~{remaining_hours} hours")
    print(f"Expected Move: ±{expected_move:.0f} points")
    print(f"Upside Target: {upside_target:.0f}")
    print(f"Downside Target: {downside_target:.0f}")
    print(f"Bias: {overall_bias}")
    
    # Trading Strategy
    print(f"\n💡 TRADING STRATEGY")
    print("-" * 25)
    
    if overall_bias == "BULLISH":
        print("🔼 BULLISH STRATEGY:")
        print(f"• Long Entry: Near {current_price - 20:.0f}")
        print(f"• Target: {upside_target:.0f}")
        print(f"• Stop Loss: {immediate_support:.0f}")
        print(f"• Options: Buy {atm_strike + 50} CALL")
    elif overall_bias == "BEARISH":
        print("🔽 BEARISH STRATEGY:")
        print(f"• Short Entry: Near {current_price + 20:.0f}")
        print(f"• Target: {downside_target:.0f}")
        print(f"• Stop Loss: {immediate_resistance:.0f}")
        print(f"• Options: Buy {atm_strike - 50} PUT")
    else:
        print("⚖️ NEUTRAL STRATEGY:")
        print(f"• Range Trade: {immediate_support:.0f} - {immediate_resistance:.0f}")
        print(f"• Buy: Near {immediate_support + 20:.0f}")
        print(f"• Sell: Near {immediate_resistance - 20:.0f}")
        print("• Options: Iron Condor or Straddle")
    
    # Key levels to watch
    print(f"\n🚨 KEY LEVELS TO WATCH")
    print("-" * 30)
    print(f"• {immediate_resistance:.0f} - Day High Breakout")
    print(f"• 25000 - Psychological Resistance")
    print(f"• {atm_strike} - ATM Strike")
    print(f"• {immediate_support:.0f} - Day Low Support")
    print(f"• 24900 - Psychological Support")
    
    # Risk Assessment
    volatility_level = "HIGH" if day_range > 150 else "NORMAL" if day_range > 80 else "LOW"
    
    print(f"\n⚠️ RISK ASSESSMENT")
    print("-" * 20)
    print(f"Volatility: {volatility_level} ({day_range:.0f} points)")
    print(f"Current Risk: MEDIUM")
    print(f"Time Decay: HIGH (EOD approaching)")
    
    print(f"\n" + "="*60)
    print("📊 REAL DATA ANALYSIS COMPLETE")
    print("⚠️ Based on actual market data - Trade responsibly")
    print("="*60)

if __name__ == "__main__":
    analyze_real_nifty_data()