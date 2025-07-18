"""
🚀 MANUAL NIFTY 50 ANALYZER 🚀
User Provides REAL Market Data - No Fake Numbers
Accurate Analysis Based on Your Live Data Input
"""

class ManualNiftyAnalyzer:
    
    def __init__(self):
        """Initialize manual analyzer - requires user input of real data"""
        print("📊 MANUAL NIFTY 50 ANALYZER")
        print("⚠️  I CANNOT fetch live data - You provide the real numbers")
        print("✅ I will analyze based on YOUR live market data input")
    
    def analyze_with_user_data(self, market_data):
        """
        Analyze NIFTY 50 with user-provided REAL market data
        
        market_data should contain:
        {
            'current_price': float,     # Current NIFTY price
            'day_open': float,          # Today's opening price  
            'day_high': float,          # Today's high so far
            'day_low': float,           # Today's low so far
            'prev_close': float,        # Yesterday's closing price
            'current_time': str,        # Current market time
            'volume': int,              # Current volume (optional)
        }
        """
        
        print("\n🔍 ANALYZING WITH YOUR REAL MARKET DATA:")
        print("=" * 50)
        
        # Extract real data
        current_price = market_data['current_price']
        day_open = market_data['day_open'] 
        day_high = market_data['day_high']
        day_low = market_data['day_low']
        prev_close = market_data['prev_close']
        current_time = market_data.get('current_time', 'Not provided')
        
        # Calculate real metrics
        day_change = current_price - prev_close
        day_change_pct = (day_change / prev_close) * 100
        day_range = day_high - day_low
        
        # Position in day's range
        if day_range > 0:
            position_in_range = (current_price - day_low) / day_range
        else:
            position_in_range = 0.5
        
        print(f"📈 Current NIFTY: {current_price}")
        print(f"📊 Day Open: {day_open}")
        print(f"🔼 Day High: {day_high}")
        print(f"🔽 Day Low: {day_low}")
        print(f"📉 Prev Close: {prev_close}")
        print(f"📊 Day Change: {day_change:+.2f} ({day_change_pct:+.2f}%)")
        print(f"📏 Day Range: {day_range:.2f} points")
        print(f"⏰ Time: {current_time}")
        
        # Technical Analysis
        analysis = self.perform_technical_analysis(market_data)
        
        # Generate levels
        levels = self.calculate_real_levels(market_data)
        
        # Generate predictions
        predictions = self.generate_real_predictions(market_data, analysis)
        
        # Generate report
        self.generate_real_report(market_data, analysis, levels, predictions)
        
        return analysis, levels, predictions
    
    def perform_technical_analysis(self, data):
        """Perform technical analysis on real data"""
        current_price = data['current_price']
        day_open = data['day_open']
        day_high = data['day_high'] 
        day_low = data['day_low']
        prev_close = data['prev_close']
        
        analysis = {}
        
        # Gap analysis
        gap = day_open - prev_close
        gap_pct = (gap / prev_close) * 100
        
        if gap_pct > 0.5:
            analysis['gap'] = f"GAP UP: {gap:+.2f} ({gap_pct:+.2f}%)"
        elif gap_pct < -0.5:
            analysis['gap'] = f"GAP DOWN: {gap:+.2f} ({gap_pct:+.2f}%)"
        else:
            analysis['gap'] = f"NO SIGNIFICANT GAP: {gap:+.2f} ({gap_pct:+.2f}%)"
        
        # Intraday performance
        intraday_change = current_price - day_open
        intraday_pct = (intraday_change / day_open) * 100
        
        analysis['intraday_performance'] = f"{intraday_change:+.2f} ({intraday_pct:+.2f}%) from open"
        
        # Position in range
        day_range = day_high - day_low
        if day_range > 0:
            position = (current_price - day_low) / day_range
            if position > 0.8:
                analysis['range_position'] = "NEAR DAY HIGH (Strong)"
            elif position > 0.6:
                analysis['range_position'] = "UPPER RANGE (Bullish)"
            elif position > 0.4:
                analysis['range_position'] = "MIDDLE RANGE (Neutral)"
            elif position > 0.2:
                analysis['range_position'] = "LOWER RANGE (Bearish)"
            else:
                analysis['range_position'] = "NEAR DAY LOW (Weak)"
        
        # Trend analysis
        if current_price > day_open > prev_close:
            analysis['trend'] = "STRONG BULLISH (Higher highs)"
        elif current_price > max(day_open, prev_close):
            analysis['trend'] = "BULLISH (Above key levels)"
        elif current_price < day_open < prev_close:
            analysis['trend'] = "STRONG BEARISH (Lower lows)"
        elif current_price < min(day_open, prev_close):
            analysis['trend'] = "BEARISH (Below key levels)"
        else:
            analysis['trend'] = "SIDEWAYS/MIXED"
        
        return analysis
    
    def calculate_real_levels(self, data):
        """Calculate real support/resistance levels"""
        current_price = data['current_price']
        day_high = data['day_high']
        day_low = data['day_low']
        prev_close = data['prev_close']
        
        levels = {}
        
        # Immediate levels
        levels['immediate_resistance'] = day_high
        levels['immediate_support'] = day_low
        
        # Pivot points
        pivot = (day_high + day_low + prev_close) / 3
        levels['pivot'] = round(pivot, 2)
        
        # Options strikes (multiples of 50)
        atm_strike = round(current_price / 50) * 50
        levels['atm_strike'] = atm_strike
        levels['call_strikes'] = [atm_strike + i*50 for i in range(1, 6)]
        levels['put_strikes'] = [atm_strike - i*50 for i in range(1, 6)]
        
        # Psychological levels
        base_level = int(current_price / 100) * 100
        levels['psychological'] = [base_level + i*50 for i in range(-5, 6)]
        
        # Key levels based on current price
        levels['resistance_1'] = current_price + 50
        levels['resistance_2'] = current_price + 100
        levels['support_1'] = current_price - 50
        levels['support_2'] = current_price - 100
        
        return levels
    
    def generate_real_predictions(self, data, analysis):
        """Generate predictions based on real market data"""
        current_price = data['current_price']
        day_range = data['day_high'] - data['day_low']
        
        predictions = {}
        
        # Expected volatility based on current range
        if day_range < 50:
            volatility = "LOW (< 50 points)"
            expected_range = 60
        elif day_range < 100:
            volatility = "NORMAL (50-100 points)"
            expected_range = 100
        elif day_range < 150:
            volatility = "HIGH (100-150 points)"
            expected_range = 150
        else:
            volatility = "VERY HIGH (> 150 points)"
            expected_range = 200
        
        predictions['volatility'] = volatility
        predictions['expected_remaining_range'] = expected_range
        
        # Targets based on current position
        predictions['upside_target'] = current_price + (expected_range * 0.618)
        predictions['downside_target'] = current_price - (expected_range * 0.618)
        
        # Probability based on analysis
        trend = analysis.get('trend', 'SIDEWAYS')
        range_pos = analysis.get('range_position', 'MIDDLE RANGE')
        
        if 'BULLISH' in trend and 'UPPER' in range_pos:
            predictions['bias'] = "BULLISH (70%)"
        elif 'BEARISH' in trend and 'LOWER' in range_pos:
            predictions['bias'] = "BEARISH (70%)"
        elif 'STRONG' in trend:
            predictions['bias'] = f"TRENDING ({trend})"
        else:
            predictions['bias'] = "NEUTRAL (50/50)"
        
        return predictions
    
    def generate_real_report(self, data, analysis, levels, predictions):
        """Generate analysis report with real data"""
        
        print("\n" + "="*60)
        print("📊 REAL MARKET ANALYSIS REPORT")
        print("="*60)
        
        print("\n🎯 TECHNICAL ANALYSIS:")
        for key, value in analysis.items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n📊 KEY LEVELS:")
        print(f"  • ATM Strike: {levels['atm_strike']}")
        print(f"  • Immediate Resistance: {levels['immediate_resistance']}")
        print(f"  • Immediate Support: {levels['immediate_support']}")
        print(f"  • Pivot Point: {levels['pivot']}")
        
        print(f"\n🎯 PREDICTIONS:")
        for key, value in predictions.items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n📈 OPTIONS LEVELS:")
        print(f"  • Call Strikes: {levels['call_strikes'][:3]}")
        print(f"  • Put Strikes: {levels['put_strikes'][:3]}")
        
        print("\n💡 TRADING STRATEGY:")
        bias = predictions['bias']
        if 'BULLISH' in bias:
            print("  🔼 BULLISH BIAS - Look for buying opportunities")
            print(f"  • Entry: Near {levels['support_1']:.0f}")
            print(f"  • Target: {predictions['upside_target']:.0f}")
            print(f"  • Stop: {levels['support_2']:.0f}")
        elif 'BEARISH' in bias:
            print("  🔽 BEARISH BIAS - Look for selling opportunities") 
            print(f"  • Entry: Near {levels['resistance_1']:.0f}")
            print(f"  • Target: {predictions['downside_target']:.0f}")
            print(f"  • Stop: {levels['resistance_2']:.0f}")
        else:
            print("  ⚖️ NEUTRAL - Range trading")
            print(f"  • Buy: Near {levels['immediate_support']}")
            print(f"  • Sell: Near {levels['immediate_resistance']}")


# Example usage with YOUR real data
def analyze_nifty_with_real_data():
    """
    YOU provide the real market data here
    """
    analyzer = ManualNiftyAnalyzer()
    
    # 🔥 REPLACE THESE WITH YOUR REAL LIVE DATA 🔥
    real_market_data = {
        'current_price': 25140.0,  # ← YOUR CURRENT NIFTY PRICE
        'day_open': 25140.0,       # ← TODAY'S OPENING (you mentioned ~25140)
        'day_high': 25180.0,       # ← TODAY'S HIGH SO FAR  
        'day_low': 25100.0,        # ← TODAY'S LOW SO FAR
        'prev_close': 25120.0,     # ← YESTERDAY'S CLOSE
        'current_time': '2:30 PM', # ← CURRENT MARKET TIME
        'volume': 50000000,        # ← VOLUME (optional)
    }
    
    print("🚨 REPLACE THE DATA ABOVE WITH YOUR REAL LIVE NUMBERS! 🚨")
    print("📊 Then run the analysis with accurate data")
    
    # Run analysis
    analysis, levels, predictions = analyzer.analyze_with_user_data(real_market_data)
    
    return analysis, levels, predictions


if __name__ == "__main__":
    print("🚀 MANUAL NIFTY 50 ANALYZER")
    print("⚠️  I cannot fetch live data - YOU provide real numbers")
    print("✅ Update the data in the code with your live market feed")
    print("\n" + "="*50)
    
    analyze_nifty_with_real_data()