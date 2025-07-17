"""
📊 MARKET ANALYSIS SUMMARY REPORT 📊
Summary of Hidden Patterns, Anomalies & Market Secrets Discovered
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from robust_market_analyzer import RobustMarketAnalyzer

def create_summary_report():
    """Create a comprehensive summary of market analysis findings"""
    
    print("🔮 CREATING MARKET ANALYSIS SUMMARY REPORT 🔮")
    print("="*70)
    
    # Initialize analyzer and run analysis
    analyzer = RobustMarketAnalyzer()
    
    # Run analysis phases
    try:
        print("\n🚀 PHASE 1: COLLECTING DATA...")
        stocks_fetched = analyzer.fetch_market_data(period="1y")
        print(f"✅ Successfully fetched data for {stocks_fetched} securities")
        
        print("\n🔧 PHASE 2: ENGINEERING FEATURES...")
        features_created = analyzer.engineer_features()
        print(f"✅ Features engineered for {features_created} securities")
        
        print("\n🔍 PHASE 3: DISCOVERING PATTERNS...")
        patterns = analyzer.discover_patterns()
        
        print("\n🔍 PHASE 4: ANALYZING CORRELATIONS...")
        correlations = analyzer.analyze_correlations()
        
        print("\n🚨 PHASE 5: DETECTING ANOMALIES...")
        anomalies = analyzer.detect_anomalies()
        
        # Generate findings summary
        print("\n📊 GENERATING SUMMARY REPORT...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Market Analysis Summary
        print("\n" + "="*70)
        print("🔮 COMPREHENSIVE MARKET ANALYSIS SUMMARY 🔮")
        print("="*70)
        print(f"📅 Analysis Date: {timestamp}")
        print(f"📊 Total Securities Analyzed: {features_created}")
        print(f"⏰ Data Period: 1 Year")
        print(f"🎯 F&O Stocks Covered: {len(analyzer.fo_stocks)}")
        print(f"🏭 Sectors Analyzed: {len(analyzer.sectors)}")
        
        # PATTERN DISCOVERIES
        print("\n🧠 PATTERN DISCOVERY RESULTS:")
        print("-" * 40)
        if patterns:
            unique_clusters = len(set(patterns['clusters']))
            anomaly_points = (patterns['anomalies'] == -1).sum()
            print(f"   🔍 Market Regimes Identified: {unique_clusters}")
            print(f"   ⚠️ Anomalous Pattern Points: {anomaly_points}")
            print(f"   📈 PCA Variance Explained: {patterns['pca_variance_ratio'][:3].sum():.2%}")
            print(f"   🧠 Neural Network Trained: ✅ Complex Pattern Learning")
        
        # CORRELATION ANALYSIS
        print("\n🔗 CORRELATION ANALYSIS RESULTS:")
        print("-" * 40)
        if correlations:
            high_corr_count = len(correlations['high_correlations'])
            sector_count = len(correlations['sector_performance'])
            print(f"   📊 High Correlation Pairs Found: {high_corr_count}")
            print(f"   🏭 Sectors Analyzed: {sector_count}")
            
            # Top correlation pairs
            print(f"\n   🏆 TOP 5 CORRELATION PAIRS:")
            for i, pair in enumerate(correlations['high_correlations'][:5], 1):
                print(f"      {i}. {pair['stock1']} ↔ {pair['stock2']}: {pair['correlation']:.3f}")
        
        # ANOMALY DETECTION
        print("\n⚠️ ANOMALY DETECTION RESULTS:")
        print("-" * 40)
        if anomalies:
            total_anomaly_stocks = len(anomalies)
            print(f"   🚨 Stocks with Anomalies: {total_anomaly_stocks}")
            
            # Top anomaly stocks
            print(f"\n   🔥 TOP 10 ANOMALY STOCKS:")
            for i, (stock, data) in enumerate(list(anomalies.items())[:10], 1):
                anomaly_breakdown = []
                if data.get('volume_anomalies', 0) > 0:
                    anomaly_breakdown.append(f"V:{data['volume_anomalies']}")
                if data.get('price_anomalies', 0) > 0:
                    anomaly_breakdown.append(f"P:{data['price_anomalies']}")
                if data.get('gap_anomalies', 0) > 0:
                    anomaly_breakdown.append(f"G:{data['gap_anomalies']}")
                if data.get('rsi_extremes', 0) > 0:
                    anomaly_breakdown.append(f"R:{data['rsi_extremes']}")
                
                breakdown_str = " | ".join(anomaly_breakdown)
                print(f"      {i:2d}. {stock:12s}: {data['total_anomalies']:3d} total ({breakdown_str})")
        
        # SECTOR PERFORMANCE
        print("\n🏭 SECTOR PERFORMANCE RANKING:")
        print("-" * 40)
        if correlations and 'sector_performance' in correlations:
            sector_ranking = sorted(correlations['sector_performance'].items(), 
                                  key=lambda x: x[1]['sharpe_ratio'], reverse=True)
            
            for i, (sector, perf) in enumerate(sector_ranking, 1):
                print(f"   {i}. {sector:10s}: Sharpe {perf['sharpe_ratio']:6.3f} | "
                      f"Return {perf['avg_return']:7.4f} | Vol {perf['volatility']:.4f} | "
                      f"Stocks: {perf['stocks_count']}")
        
        # MARKET INSIGHTS
        print("\n💡 KEY MARKET INSIGHTS:")
        print("-" * 40)
        
        # Volume anomaly insights
        volume_anomaly_stocks = []
        price_anomaly_stocks = []
        gap_anomaly_stocks = []
        
        for stock, data in anomalies.items():
            if data.get('volume_anomalies', 0) > 5:
                volume_anomaly_stocks.append(stock)
            if data.get('price_anomalies', 0) > 3:
                price_anomaly_stocks.append(stock)
            if data.get('gap_anomalies', 0) > 2:
                gap_anomaly_stocks.append(stock)
        
        print(f"   📊 High Volume Anomaly Stocks: {len(volume_anomaly_stocks)}")
        if volume_anomaly_stocks:
            print(f"      {', '.join(volume_anomaly_stocks[:10])}")
        
        print(f"   📈 High Price Anomaly Stocks: {len(price_anomaly_stocks)}")
        if price_anomaly_stocks:
            print(f"      {', '.join(price_anomaly_stocks[:10])}")
        
        print(f"   🔀 High Gap Anomaly Stocks: {len(gap_anomaly_stocks)}")
        if gap_anomaly_stocks:
            print(f"      {', '.join(gap_anomaly_stocks[:10])}")
        
        # Correlation insights
        if correlations:
            positive_pairs = [p for p in correlations['high_correlations'] if p['correlation'] > 0]
            negative_pairs = [p for p in correlations['high_correlations'] if p['correlation'] < 0]
            
            print(f"\n   🔗 Positive Correlation Pairs: {len(positive_pairs)}")
            print(f"   🔀 Negative Correlation Pairs: {len(negative_pairs)}")
            
            # Strongest correlations
            strongest_positive = max(positive_pairs, key=lambda x: x['correlation']) if positive_pairs else None
            strongest_negative = min(negative_pairs, key=lambda x: x['correlation']) if negative_pairs else None
            
            if strongest_positive:
                print(f"   💪 Strongest Positive: {strongest_positive['stock1']} ↔ {strongest_positive['stock2']} ({strongest_positive['correlation']:.3f})")
            
            if strongest_negative:
                print(f"   🔻 Strongest Negative: {strongest_negative['stock1']} ↔ {strongest_negative['stock2']} ({strongest_negative['correlation']:.3f})")
        
        # TRADING OPPORTUNITIES
        print("\n💰 POTENTIAL TRADING OPPORTUNITIES:")
        print("-" * 40)
        
        # High anomaly stocks for mean reversion
        high_anomaly_stocks = [(stock, data) for stock, data in anomalies.items() 
                              if data['total_anomalies'] > 10]
        
        print(f"   📊 Mean Reversion Candidates: {len(high_anomaly_stocks)}")
        for stock, data in high_anomaly_stocks[:5]:
            print(f"      • {stock}: {data['total_anomalies']} anomalies detected")
        
        # High correlation pairs for pair trading
        strong_correlations = [p for p in correlations['high_correlations'] 
                             if abs(p['correlation']) > 0.8] if correlations else []
        
        print(f"\n   🔗 Pair Trading Opportunities: {len(strong_correlations)}")
        for pair in strong_correlations[:5]:
            print(f"      • {pair['stock1']} ↔ {pair['stock2']}: {pair['correlation']:.3f}")
        
        print("\n" + "="*70)
        print("🔮 MARKET SECRETS SUCCESSFULLY REVEALED! 🔮")
        print("="*70)
        
        # Save simplified report
        simple_report = {
            'timestamp': timestamp,
            'summary': {
                'stocks_analyzed': features_created,
                'anomaly_stocks': len(anomalies),
                'correlation_pairs': len(correlations['high_correlations']) if correlations else 0,
                'sectors_analyzed': len(correlations['sector_performance']) if correlations and 'sector_performance' in correlations else 0
            },
            'top_anomalies': list(anomalies.keys())[:20] if anomalies else [],
            'top_correlations': [f"{p['stock1']}-{p['stock2']}" for p in correlations['high_correlations'][:20]] if correlations else [],
            'insights': {
                'volume_anomaly_stocks': volume_anomaly_stocks[:10],
                'price_anomaly_stocks': price_anomaly_stocks[:10],
                'gap_anomaly_stocks': gap_anomaly_stocks[:10],
                'strong_correlations': len(strong_correlations),
                'mean_reversion_candidates': len(high_anomaly_stocks)
            }
        }
        
        # Save report
        with open('market_analysis_summary.json', 'w') as f:
            json.dump(simple_report, f, indent=2)
        
        print(f"\n💾 Summary report saved: market_analysis_summary.json")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_summary_report()