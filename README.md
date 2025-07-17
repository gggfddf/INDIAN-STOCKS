# Indian Stock Market Deep Learning Analysis System

A comprehensive, modular, and adaptive deep learning system for analyzing the Indian stock market. This system analyzes the **Top 100 F&O-traded stocks** and **three major indices (Nifty 50, Bank Nifty, and FINNIFTY)** to detect patterns, anomalies, and generate actionable trading signals.

## 🚀 Features

### 📊 Market Coverage
- **100+ F&O Stocks**: Complete coverage of top F&O traded stocks
- **Major Indices**: Nifty 50, Bank Nifty, and FINNIFTY analysis
- **Real-time Data**: Multi-source data collection with fallback mechanisms

### 🔍 Pattern Detection
- **Technical Patterns**: Triangles, double tops/bottoms, head & shoulders, flags
- **Volume-Price Anomalies**: Volume spikes, compression patterns, divergences
- **Breakout Signals**: Range breakouts, moving average breakouts, Bollinger Band breakouts
- **Candlestick Patterns**: Doji, hammer, shooting star, engulfing patterns

### 🤖 Machine Learning Models
- **LSTM Networks**: For sequential pattern recognition and price prediction
- **Transformer Models**: Advanced attention-based pattern detection
- **Ensemble Methods**: Combining XGBoost, LightGBM, and deep learning models
- **Anomaly Detection**: Autoencoder-based unusual pattern identification
- **Regime Detection**: Market regime classification (bull/bear/sideways)

### 📈 Trading Signals
- **Multi-strategy Signals**: Pattern-based, ML-based, and hybrid signals
- **Pair Trading**: Correlation-based pair trading opportunities
- **Sector Rotation**: Sector momentum and leadership analysis
- **Risk Management**: Automated position sizing and risk-reward calculations

### 🎯 Key Capabilities
1. **Continuous Scanning**: Monitors all 100+ stocks simultaneously
2. **Pattern Recognition**: Identifies recurring technical patterns
3. **Anomaly Detection**: Spots unusual volume-price behaviors
4. **Cross-asset Analysis**: Correlation patterns and multi-stock opportunities
5. **Explainable AI**: Clear reasoning for every signal generated
6. **Adaptive Learning**: Models update based on market conditions

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │ Feature Engine  │    │ Pattern Detection│
│                 │────│                 │────│                 │
│ • Yahoo Finance │    │ • Technical     │    │ • Chart Patterns│
│ • NSE API       │    │   Indicators    │    │ • Volume Anomaly│
│ • Backup APIs   │    │ • Market Data   │    │ • Breakouts     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ML Models     │    │ Signal Generator│    │   Output/API    │
│                 │────│                 │────│                 │
│ • LSTM/Trans.   │    │ • Trading Sigs  │    │ • JSON Reports  │
│ • Ensemble      │    │ • Risk Mgmt     │    │ • CSV Exports   │
│ • Anomaly Det.  │    │ • Pair Trading  │    │ • Real-time API │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Requirements

### System Requirements
- Python 3.8+
- 8GB+ RAM (16GB recommended)
- Internet connection for data fetching
- 2GB+ free disk space

### Python Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- `tensorflow>=2.13.0` - Deep learning models
- `pandas>=2.0.3` - Data manipulation
- `numpy>=1.24.3` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning
- `yfinance>=0.2.18` - Market data
- `ta>=0.10.2` - Technical analysis
- `loguru>=0.7.0` - Logging

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd indian-stock-market-analysis

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python main.py
```

### 2. Basic Usage
```python
from main import StockMarketAnalysisSystem

# Initialize system
system = StockMarketAnalysisSystem()

# Run complete analysis
results = system.run_complete_analysis()

# Get trading signals
signals = system.get_real_time_signals()

# Get specific stock analysis
analysis = system.get_pattern_analysis('RELIANCE')
```

### 3. Configuration
Edit `src/config/config.py` to customize:
- Stock universe
- Technical parameters
- ML model settings
- Risk management rules

## 📊 Output Examples

### Trading Signal Output
```json
{
  "symbol": "RELIANCE",
  "signal_type": "BUY",
  "strength": "STRONG",
  "confidence": 0.85,
  "entry_price": 2450.50,
  "target_price": 2573.00,
  "stop_loss": 2401.00,
  "risk_reward_ratio": 2.5,
  "reasoning": [
    "Bollinger Band breakout detected",
    "Volume confirmation present",
    "RSI not overbought: 65.2"
  ],
  "time_horizon": "swing"
}
```

### Pattern Detection Output
```json
{
  "pattern_type": "triangle",
  "symbol": "HDFCBANK",
  "confidence": 0.8,
  "signal": "breakout_pending",
  "range_contraction": 0.35,
  "volume_confirmation": true
}
```

## 📁 Project Structure

```
indian-stock-market-analysis/
├── main.py                     # Main orchestrator
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── src/
│   ├── config/
│   │   └── config.py          # System configuration
│   ├── data_sources/
│   │   └── data_collector.py  # Data collection
│   ├── feature_engineering/
│   │   └── technical_indicators.py # Technical analysis
│   ├── pattern_detection/
│   │   └── pattern_detector.py # Pattern recognition
│   ├── models/
│   │   └── deep_learning_models.py # ML models
│   ├── trading_signals/
│   │   └── signal_generator.py # Signal generation
│   └── utils/
│       └── helpers.py         # Utility functions
├── results/                   # Analysis outputs
├── logs/                     # System logs
└── tests/                    # Unit tests
```

## 🔧 Customization

### Adding New Patterns
```python
# In pattern_detector.py
def _detect_custom_pattern(self, df, symbol):
    # Your pattern logic here
    if pattern_detected:
        return {
            'type': 'custom_pattern',
            'confidence': confidence_score,
            'signal': 'bullish/bearish'
        }
```

### Adding New Indicators
```python
# In technical_indicators.py
def _add_custom_indicators(self, df):
    # Calculate your custom indicator
    df['Custom_Indicator'] = your_calculation(df)
    return df
```

### Custom Signal Rules
```python
# In signal_generator.py
def _generate_custom_signals(self, stock_data, patterns):
    # Your signal generation logic
    return signals
```

## 📈 Performance Optimization

### For Large-Scale Analysis
1. **Parallel Processing**: Increase `max_workers` in data collection
2. **Memory Management**: Process stocks in batches
3. **Model Optimization**: Use quantized models for faster inference
4. **Caching**: Enable result caching for repeated analysis

### Configuration Example
```python
# In config.py
ML_PARAMS = {
    'batch_processing': True,
    'max_parallel_workers': 10,
    'model_quantization': True,
    'cache_results': True
}
```

## 🔍 API Usage

### Real-time Signal Monitoring
```python
# Get all signals
all_signals = system.get_real_time_signals()

# Get signals for specific stock
reliance_signals = system.get_real_time_signals('RELIANCE')

# Get pattern analysis
pattern_data = system.get_pattern_analysis('HDFCBANK')
```

### Batch Analysis
```python
# Analyze specific stocks
stocks_to_analyze = ['RELIANCE', 'TCS', 'HDFCBANK']
results = system.analyze_specific_stocks(stocks_to_analyze)
```

## 📊 Key Metrics & Signals

### Signal Strength Classification
- **VERY_STRONG**: Confidence > 90%, multiple confirmations
- **STRONG**: Confidence > 80%, good risk-reward ratio
- **MODERATE**: Confidence > 70%, basic confirmations
- **WEAK**: Confidence < 70%, filtered out

### Pattern Confidence Scoring
- Volume confirmation: +10%
- Multiple timeframe alignment: +15%
- Technical indicator support: +10%
- ML model agreement: +5%

## 🛠️ Troubleshooting

### Common Issues

1. **Data Collection Failures**
   ```python
   # Check data sources
   system.data_collector.get_market_status()
   ```

2. **Memory Issues**
   ```python
   # Reduce batch size
   Config.BATCH_SIZE = 10
   ```

3. **Model Training Errors**
   ```python
   # Check data quality
   is_valid, issues = system.validate_data_quality(data)
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Write tests
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This system is for educational and research purposes only. It is not financial advice. Always conduct your own research and consult with financial advisors before making investment decisions. Past performance does not guarantee future results.

## 🔗 Links

- [Documentation](docs/)
- [API Reference](docs/api.md)
- [Examples](examples/)
- [Contributing Guidelines](CONTRIBUTING.md)

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the examples

---

**Built with ❤️ for the Indian Stock Market Analysis Community**