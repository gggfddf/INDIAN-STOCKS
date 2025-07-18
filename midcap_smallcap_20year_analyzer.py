"""
🚀 INDIAN MID-CAP & SMALL-CAP 20-YEAR WEEKLY ANALYZER 🚀
Advanced Accumulation Zone Detection & Sector Rotation Analysis
Multi-Dimensional Pattern Recognition System
"""

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import scipy.stats as stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns

class ComprehensiveStockAnalyzer:
    
    def __init__(self):
        """Initialize the comprehensive analyzer with Indian stock universe"""
        self.setup_indian_stock_universe()
        self.scaler = StandardScaler()
        self.accumulation_detector = None
        self.sector_rotation_model = None
        
    def setup_indian_stock_universe(self):
        """Setup comprehensive Indian stock universe - Mid & Small cap"""
        print("🔍 Setting up Indian Mid-Cap & Small-Cap Universe...")
        
        # NIFTY MidCap 150 stocks
        self.midcap_stocks = [
            "ADANIPORTS.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANITRANS.NS",
            "AARTIIND.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", "AFFLE.NS",
            "AIAENG.NS", "AJANTPHARM.NS", "ALKEM.NS", "AMBUJACEM.NS", "ANGELONE.NS",
            "APARINDS.NS", "APOLLOHOSP.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS",
            "ASTRAL.NS", "ATUL.NS", "AUBANK.NS", "AUROPHARMA.NS", "BALKRISIND.NS",
            "BALRAMCHIN.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "BATAINDIA.NS",
            "BEL.NS", "BERGEPAINT.NS", "BHARATFORG.NS", "BHARTIARTL.NS",
            "BHEL.NS", "BIOCON.NS", "BOSCHLTD.NS", "BPCL.NS", "BRITANNIA.NS",
            "BSOFT.NS", "CANBK.NS", "CANFINHOME.NS", "CHAMBLFERT.NS",
            "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS",
            "COLPAL.NS", "CONCOR.NS", "COROMANDEL.NS", "CROMPTON.NS",
            "CUB.NS", "CUMMINSIND.NS", "DABUR.NS", "DALBHARAT.NS",
            "DEEPAKNTR.NS", "DELTACORP.NS", "DIVISLAB.NS", "DIXON.NS",
            "DLF.NS", "DRREDDY.NS", "EICHERMOT.NS", "ESCORTS.NS",
            "EXIDEIND.NS", "FEDERALBNK.NS", "FORTIS.NS", "GAIL.NS",
            "GLENMARK.NS", "GMRINFRA.NS", "GNFC.NS", "GODREJCP.NS",
            "GODREJPROP.NS", "GRANULES.NS", "GRASIM.NS", "GSPL.NS",
            "GUJGASLTD.NS", "HAL.NS", "HAVELLS.NS", "HCLTECH.NS",
            "HDFCAMC.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
            "HINDCOPPER.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "HONAUT.NS",
            "ICICIBANK.NS", "ICICIPRULI.NS", "IDEA.NS", "IDFC.NS",
            "IDFCFIRSTB.NS", "IEX.NS", "IGL.NS", "INDHOTEL.NS",
            "INDIACEM.NS", "INDIAMART.NS", "INDIGO.NS", "INDUSINDBK.NS",
            "INDUSTOWER.NS", "INFY.NS", "IOC.NS", "IPCALAB.NS",
            "IRB.NS", "IRCTC.NS", "ITC.NS", "JINDALSTEL.NS",
            "JKCEMENT.NS", "JSWSTEEL.NS", "JUBLFOOD.NS", "KOTAKBANK.NS",
            "LALPATHLAB.NS", "LAURUSLABS.NS", "LICHSGFIN.NS", "LT.NS",
            "LTF.NS", "LTTS.NS", "LUPIN.NS", "M&M.NS",
            "M&MFIN.NS", "MANAPPURAM.NS", "MARICO.NS", "MARUTI.NS",
            "MCDOWELL-N.NS", "MCX.NS", "METROPOLIS.NS", "MFSL.NS",
            "MGL.NS", "MINDTREE.NS", "MOTHERSON.NS", "MPHASIS.NS",
            "MRF.NS", "MUTHOOTFIN.NS", "NATIONALUM.NS", "NAUKRI.NS",
            "NAVINFLUOR.NS", "NESTLEIND.NS", "NMDC.NS", "NTPC.NS",
            "OBEROIRLTY.NS", "OFSS.NS", "ONGC.NS", "PAGEIND.NS",
            "PEL.NS", "PERSISTENT.NS", "PETRONET.NS", "PFC.NS",
            "PIDILITIND.NS", "PIIND.NS", "PNB.NS", "POLYCAB.NS",
            "POWERGRID.NS", "PVRINOX.NS", "RAMCOCEM.NS", "RBLBANK.NS",
            "RECLTD.NS", "RELIANCE.NS", "SAIL.NS", "SBICARD.NS",
            "SBILIFE.NS", "SBIN.NS", "SHREECEM.NS", "SIEMENS.NS",
            "SRF.NS", "SRTRANSFIN.NS", "SUNPHARMA.NS", "SUNTV.NS",
            "SYNGENE.NS", "TATACHEM.NS", "TATACOMM.NS", "TATACONSUM.NS",
            "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS",
            "TECHM.NS", "TITAN.NS", "TORNTPHARM.NS", "TORNTPOWER.NS",
            "TRENT.NS", "TVSMOTOR.NS", "UBL.NS", "ULTRACEMCO.NS",
            "UPL.NS", "VEDL.NS", "VOLTAS.NS", "WIPRO.NS", "ZEEL.NS"
        ]
        
        # NIFTY SmallCap 250 (representative selection)
        self.smallcap_stocks = [
            "360ONE.NS", "3MINDIA.NS", "AAVAS.NS", "ABFRL.NS", "ADVENZYMES.NS",
            "AEGISCHEM.NS", "AFFLE.NS", "AGARIND.NS", "AGRITECH.NS", "AHLEAST.NS",
            "AIROLAM.NS", "AJANTPHARM.NS", "AKZOINDIA.NS", "ALEMBICLTD.NS",
            "ALKYLAMINE.NS", "ALLCARGO.NS", "AMBER.NS", "AMBUJACEM.NS",
            "ANANTRAJ.NS", "ANGELONE.NS", "ANILAGR.NS", "ANTGRAPHIC.NS",
            "APCOTEXIND.NS", "APLLTD.NS", "ARVINDFASN.NS", "ASAHIINDIA.NS",
            "ASHOKA.NS", "ASIANPAINT.NS", "ASTERDM.NS", "ASTRAL.NS",
            "ASTRAZEN.NS", "ATUL.NS", "AUROPHARMA.NS", "AVANTI.NS",
            "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS",
            "BALKRISIND.NS", "BALMLAWRIE.NS", "BALRAMCHIN.NS", "BANCOINDIA.NS",
            "BANDHANBNK.NS", "BANKBARODA.NS", "BASF.NS", "BATAINDIA.NS",
            "BAYERCROP.NS", "BBLTD.NS", "BDL.NS", "BEL.NS",
            "BEML.NS", "BERGEPAINT.NS", "BHARATFORG.NS", "BHARTIARTL.NS",
            "BHEL.NS", "BIOCON.NS", "BIRLACORPN.NS", "BLISSGVS.NS",
            "BLUEDART.NS", "BLUESTARCO.NS", "BOMDYEING.NS", "BOSCHLTD.NS",
            "BPCL.NS", "BRIGADE.NS", "BRITANNIA.NS", "BSOFT.NS",
            "CADILAHC.NS", "CAMS.NS", "CANBK.NS", "CANFINHOME.NS",
            "CAPLIPOINT.NS", "CARBORUNIV.NS", "CARERATING.NS", "CASTROLIND.NS",
            "CCL.NS", "CDSL.NS", "CENTURYTEX.NS", "CERA.NS",
            "CHALET.NS", "CHAMBLFERT.NS", "CHENNPETRO.NS", "CHOLAFIN.NS",
            "CIPLA.NS", "CLEAN.NS", "COALINDIA.NS", "COCHINSHIP.NS",
            "COFORGE.NS", "COLPAL.NS", "CONCOR.NS", "COROMANDEL.NS",
            "CREDITACC.NS", "CRISIL.NS", "CROMPTON.NS", "CUB.NS",
            "CUMMINSIND.NS", "CYIENT.NS", "DABUR.NS", "DALBHARAT.NS",
            "DBL.NS", "DCBBANK.NS", "DCMSHRIRAM.NS", "DEEPAKNTR.NS",
            "DELTACORP.NS", "DHANI.NS", "DISHTV.NS", "DIVISLAB.NS",
            "DIXON.NS", "DLF.NS", "DRREDDY.NS", "DSSL.NS",
            "DTIL.NS", "EASEMYTRIP.NS", "EICHERMOT.NS", "EIDPARRY.NS",
            "EIHOTEL.NS", "ELGIEQUIP.NS", "EMAMILTD.NS", "ENDURANCE.NS",
            "ENGINERSIN.NS", "EQUITAS.NS", "ESCORTS.NS", "ESSELPACK.NS",
            "EXIDEIND.NS", "FEDERALBNK.NS", "FINEORG.NS", "FINCABLES.NS",
            "FORCEMOT.NS", "FORTIS.NS", "FSL.NS", "GAIL.NS",
            "GALAXYSURF.NS", "GARFIBRES.NS", "GESHIP.NS", "GET&D.NS",
            "GHCL.NS", "GICRE.NS", "GILLETTE.NS", "GLAND.NS",
            "GLAXO.NS", "GLENMARK.NS", "GLOBUSSPR.NS", "GMMPFAUDLR.NS",
            "GMRINFRA.NS", "GNFC.NS", "GODFRYPHLP.NS", "GODREJCP.NS",
            "GODREJIND.NS", "GODREJPROP.NS", "GRANULES.NS", "GRAPHITE.NS",
            "GRASIM.NS", "GREAVESCOT.NS", "GRINDWELL.NS", "GSFC.NS",
            "GSPL.NS", "GUJALKALI.NS", "GUJGASLTD.NS", "GULFOILLUB.NS",
            "HAL.NS", "HAPPSTMNDS.NS", "HATHWAY.NS", "HAVELLS.NS",
            "HCC.NS", "HCLTECH.NS", "HDFC.NS", "HDFCAMC.NS",
            "HDFCBANK.NS", "HDFCLIFE.NS", "HEG.NS", "HEIDELBERG.NS",
            "HEROMOTOCO.NS", "HFCL.NS", "HIMATSEIDE.NS", "HINDALCO.NS",
            "HINDCOPPER.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "HINDZINC.NS",
            "HOMEFIRST.NS", "HONAUT.NS", "HSCL.NS", "HUDCO.NS",
            "IBULHSGFIN.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS",
            "IDBI.NS", "IDEA.NS", "IDFC.NS", "IDFCFIRSTB.NS",
            "IEX.NS", "IFBIND.NS", "IGL.NS", "IIFL.NS",
            "INDHOTEL.NS", "INDIACEM.NS", "INDIAGLYCO.NS", "INDIAMART.NS",
            "INDIANB.NS", "INDIGO.NS", "INDOCO.NS", "INDOSTAR.NS",
            "INDUSINDBK.NS", "INDUSTOWER.NS", "INFIBEAM.NS", "INFY.NS",
            "INOXLEISUR.NS", "INTELLECT.NS", "IOB.NS", "IOC.NS",
            "IPCALAB.NS", "IRB.NS", "IRCON.NS", "IRCTC.NS",
            "ISEC.NS", "ITC.NS", "ITI.NS", "J&KBANK.NS",
            "JAGRAN.NS", "JAICORPLTD.NS", "JAMNAAUTO.NS", "JBCHEPHARM.NS",
            "JCHAC.NS", "JETAIRWAYS.NS", "JHS.NS", "JINDALSTEL.NS",
            "JISLJALEQS.NS", "JKCEMENT.NS", "JKLAKSHMI.NS", "JKPAPER.NS",
            "JKTYRE.NS", "JMFINANCIL.NS", "JSL.NS", "JSWENERGY.NS",
            "JSWSTEEL.NS", "JUBLFOOD.NS", "JUBLINDS.NS", "JUSTDIAL.NS",
            "JYOTHYLAB.NS", "KAJARIACER.NS", "KANSAINER.NS", "KEI.NS"
        ]
        
        # Combined universe
        self.all_stocks = list(set(self.midcap_stocks + self.smallcap_stocks))
        
        # Sector mapping for rotation analysis
        self.sector_mapping = {
            # Banking & Finance
            "BANKING": ["ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "HDFCBANK.NS", 
                       "INDUSINDBK.NS", "FEDERALBNK.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "PNB.NS"],
            
            # Pharmaceuticals
            "PHARMA": ["DRREDDY.NS", "CIPLA.NS", "LUPIN.NS", "BIOCON.NS", "DIVISLAB.NS", 
                      "AUROPHARMA.NS", "GLENMARK.NS", "TORNTPHARM.NS", "AJANTPHARM.NS"],
            
            # Technology
            "TECH": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTTS.NS", 
                    "MINDTREE.NS", "COFORGE.NS", "MPHASIS.NS"],
            
            # Auto & Auto Components
            "AUTO": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", 
                    "EICHERMOT.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "ESCORTS.NS"],
            
            # Metals & Mining
            "METALS": ["TATASTEEL.NS", "JSWSTEEL.NS", "SAIL.NS", "HINDALCO.NS", "VEDL.NS", 
                      "JINDALSTEL.NS", "NMDC.NS", "COALINDIA.NS", "NATIONALUM.NS"],
            
            # Energy & Oil
            "ENERGY": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "HINDPETRO.NS", 
                      "GAIL.NS", "NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS"],
            
            # FMCG
            "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", 
                    "MARICO.NS", "GODREJCP.NS", "COLPAL.NS", "TATACONSUM.NS"],
            
            # Real Estate & Infrastructure
            "REALTY": ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "BRIGADE.NS", 
                      "IRB.NS", "GMRINFRA.NS", "L&T.NS", "HAL.NS"],
            
            # Chemicals
            "CHEMICALS": ["UPL.NS", "SRF.NS", "PIDILITIND.NS", "DEEPAKNTR.NS", "BALKRISIND.NS", 
                         "GNFC.NS", "GSFC.NS", "TATACHEM.NS", "AARTI.NS"],
            
            # IT Services
            "ITSERVICES": ["PERSISTENT.NS", "CYIENT.NS", "INTELLECT.NS", "HAPPSTMNDS.NS", 
                          "COFORGE.NS", "OFSS.NS"]
        }
        
        print(f"✅ Stock Universe Setup Complete!")
        print(f"📊 Total Stocks: {len(self.all_stocks)}")
        print(f"🏢 Mid-Cap Stocks: {len(self.midcap_stocks)}")
        print(f"🏪 Small-Cap Stocks: {len(self.smallcap_stocks)}")
        print(f"🎯 Sectors: {len(self.sector_mapping)}")

    def fetch_20_year_weekly_data(self, symbol):
        """Fetch 20 years of weekly data for a stock"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*20)  # 20 years
            
            # Fetch weekly data
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date, interval="1wk")
            
            if data.empty or len(data) < 100:  # Need sufficient data
                return None
                
            return data
            
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {str(e)}")
            return None

    def advanced_feature_engineering(self, df):
        """Create comprehensive features for accumulation zone detection"""
        if df is None or len(df) < 50:
            return None
            
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Close']
        df['Upper_Shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
        df['Lower_Shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
        
        # Volume features
        df['Volume_MA20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['Volume_Price_Trend'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])
        df['Accumulation_Distribution'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])
        df['Chaikin_MF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
        df['Force_Index'] = ta.volume.force_index(df['Close'], df['Volume'])
        
        # Technical indicators
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
        df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'])
        df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'])
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
        df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Moving averages
        for period in [10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        # Price position relative to MAs
        df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        df['Price_vs_SMA200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']
        
        # Volatility features
        df['Volatility_20'] = df['Returns'].rolling(20).std()
        df['Volatility_50'] = df['Returns'].rolling(50).std()
        
        # Support/Resistance levels
        df['Support_Level'] = df['Low'].rolling(20).min()
        df['Resistance_Level'] = df['High'].rolling(20).max()
        df['Price_Position'] = (df['Close'] - df['Support_Level']) / (df['Resistance_Level'] - df['Support_Level'])
        
        return df

    def detect_accumulation_signals(self, df):
        """Advanced accumulation zone detection with multiple signals"""
        if df is None:
            return {}
            
        signals = {}
        
        # 1. Volume Accumulation Signals
        volume_surge = (df['Volume_Ratio'] > 1.5).sum() / len(df)
        obv_trend = (df['OBV'].diff().tail(20) > 0).sum() / 20
        chaikin_positive = (df['Chaikin_MF'] > 0).sum() / len(df)
        
        signals['Volume_Accumulation_Score'] = (volume_surge + obv_trend + chaikin_positive) / 3
        
        # 2. Price Action Signals
        consolidation_periods = 0
        current_consolidation = 0
        
        for i in range(20, len(df)):
            recent_range = df['Price_Range'].iloc[i-20:i].mean()
            if recent_range < 0.05:  # Less than 5% weekly range
                current_consolidation += 1
            else:
                if current_consolidation > 8:  # 8+ weeks of consolidation
                    consolidation_periods += 1
                current_consolidation = 0
        
        signals['Consolidation_Periods'] = consolidation_periods
        
        # 3. Technical Indicator Signals
        rsi_oversold_recovery = ((df['RSI'] < 30) & (df['RSI'].shift(-1) > df['RSI'])).sum()
        macd_bullish_divergence = ((df['MACD'] > df['MACD_Signal']) & 
                                  (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))).sum()
        
        signals['Technical_Signals'] = rsi_oversold_recovery + macd_bullish_divergence
        
        # 4. Breakout Signals
        recent_breakouts = 0
        for i in range(50, len(df)):
            resistance = df['High'].iloc[i-50:i-1].max()
            if df['Close'].iloc[i] > resistance * 1.02:  # 2% breakout
                recent_breakouts += 1
        
        signals['Breakout_Signals'] = recent_breakouts
        
        # 5. Current Status
        latest = df.iloc[-1]
        signals['Current_RSI'] = latest['RSI']
        signals['Current_Volume_Ratio'] = latest['Volume_Ratio']
        signals['Price_vs_200MA'] = latest['Price_vs_SMA200'] if not pd.isna(latest['Price_vs_SMA200']) else 0
        signals['Current_Price'] = latest['Close']
        
        return signals

    def detect_candlestick_patterns(self, df):
        """Advanced candlestick pattern detection"""
        if df is None or len(df) < 10:
            return {}
            
        patterns = {}
        
        # Calculate candlestick components
        body = abs(df['Close'] - df['Open'])
        upper_shadow = df['High'] - np.maximum(df['Open'], df['Close'])
        lower_shadow = np.minimum(df['Open'], df['Close']) - df['Low']
        total_range = df['High'] - df['Low']
        
        # Pattern detection
        bullish_engulfing = 0
        bearish_engulfing = 0
        hammer = 0
        doji = 0
        spinning_top = 0
        
        for i in range(1, len(df)):
            # Bullish Engulfing
            if (df['Close'].iloc[i-1] < df['Open'].iloc[i-1] and  # Previous red
                df['Close'].iloc[i] > df['Open'].iloc[i] and      # Current green
                df['Open'].iloc[i] < df['Close'].iloc[i-1] and    # Opens below prev close
                df['Close'].iloc[i] > df['Open'].iloc[i-1]):      # Closes above prev open
                bullish_engulfing += 1
            
            # Bearish Engulfing
            if (df['Close'].iloc[i-1] > df['Open'].iloc[i-1] and  # Previous green
                df['Close'].iloc[i] < df['Open'].iloc[i] and      # Current red
                df['Open'].iloc[i] > df['Close'].iloc[i-1] and    # Opens above prev close
                df['Close'].iloc[i] < df['Open'].iloc[i-1]):      # Closes below prev open
                bearish_engulfing += 1
            
            # Hammer (bullish reversal)
            if (lower_shadow.iloc[i] > 2 * body.iloc[i] and
                upper_shadow.iloc[i] < 0.1 * total_range.iloc[i] and
                body.iloc[i] > 0.1 * total_range.iloc[i]):
                hammer += 1
            
            # Doji (indecision)
            if body.iloc[i] < 0.1 * total_range.iloc[i]:
                doji += 1
            
            # Spinning Top (indecision)
            if (body.iloc[i] < 0.3 * total_range.iloc[i] and
                upper_shadow.iloc[i] > 0.2 * total_range.iloc[i] and
                lower_shadow.iloc[i] > 0.2 * total_range.iloc[i]):
                spinning_top += 1
        
        patterns['Bullish_Engulfing'] = bullish_engulfing
        patterns['Bearish_Engulfing'] = bearish_engulfing
        patterns['Hammer'] = hammer
        patterns['Doji'] = doji
        patterns['Spinning_Top'] = spinning_top
        
        return patterns

    def sector_rotation_analysis(self):
        """Analyze sector rotation patterns"""
        print("\n🔄 Analyzing Sector Rotation Patterns...")
        
        sector_performance = {}
        
        for sector, stocks in self.sector_mapping.items():
            sector_scores = []
            
            for stock in stocks:
                if stock in self.all_stocks:
                    try:
                        df = self.fetch_20_year_weekly_data(stock)
                        if df is not None and len(df) > 50:
                            # Calculate recent performance metrics
                            recent_return = df['Close'].iloc[-1] / df['Close'].iloc[-13] - 1  # 3-month return
                            momentum = df['Close'].iloc[-1] / df['Close'].iloc[-26] - 1  # 6-month return
                            
                            # Volume analysis
                            recent_volume = df['Volume'].iloc[-4:].mean()
                            avg_volume = df['Volume'].iloc[-52:].mean()
                            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                            
                            stock_score = recent_return * 0.4 + momentum * 0.4 + (volume_ratio - 1) * 0.2
                            sector_scores.append(stock_score)
                            
                    except Exception as e:
                        continue
            
            if sector_scores:
                sector_performance[sector] = {
                    'avg_score': np.mean(sector_scores),
                    'median_score': np.median(sector_scores),
                    'positive_stocks': sum(1 for score in sector_scores if score > 0),
                    'total_stocks': len(sector_scores)
                }
        
        # Rank sectors
        ranked_sectors = sorted(sector_performance.items(), 
                              key=lambda x: x[1]['avg_score'], reverse=True)
        
        return ranked_sectors

    def comprehensive_stock_analysis(self, symbol):
        """Perform comprehensive analysis on a single stock"""
        print(f"📊 Analyzing {symbol}...")
        
        # Fetch data
        df = self.fetch_20_year_weekly_data(symbol)
        if df is None:
            return None
        
        # Feature engineering
        df = self.advanced_feature_engineering(df)
        if df is None:
            return None
        
        # Accumulation signals
        accumulation_signals = self.detect_accumulation_signals(df)
        
        # Candlestick patterns
        candlestick_patterns = self.detect_candlestick_patterns(df)
        
        # Find sector
        stock_sector = "UNKNOWN"
        for sector, stocks in self.sector_mapping.items():
            if symbol in stocks:
                stock_sector = sector
                break
        
        # Overall scoring
        accumulation_score = (
            accumulation_signals.get('Volume_Accumulation_Score', 0) * 0.3 +
            min(accumulation_signals.get('Consolidation_Periods', 0) / 5, 1) * 0.2 +
            min(accumulation_signals.get('Technical_Signals', 0) / 10, 1) * 0.2 +
            min(accumulation_signals.get('Breakout_Signals', 0) / 5, 1) * 0.3
        )
        
        # Bullish trend probability
        trend_signals = 0
        if accumulation_signals.get('Current_RSI', 50) < 70:  # Not overbought
            trend_signals += 1
        if accumulation_signals.get('Price_vs_200MA', -1) > 0:  # Above 200 MA
            trend_signals += 1
        if accumulation_signals.get('Current_Volume_Ratio', 0) > 1.2:  # High volume
            trend_signals += 1
        if candlestick_patterns.get('Bullish_Engulfing', 0) > candlestick_patterns.get('Bearish_Engulfing', 0):
            trend_signals += 1
        
        bullish_probability = trend_signals / 4
        
        result = {
            'Symbol': symbol,
            'Sector': stock_sector,
            'Current_Price': accumulation_signals.get('Current_Price', 0),
            'Accumulation_Score': round(accumulation_score, 3),
            'Bullish_Probability': round(bullish_probability, 3),
            'Volume_Accumulation': round(accumulation_signals.get('Volume_Accumulation_Score', 0), 3),
            'Consolidation_Periods': accumulation_signals.get('Consolidation_Periods', 0),
            'Technical_Signals': accumulation_signals.get('Technical_Signals', 0),
            'Breakout_Signals': accumulation_signals.get('Breakout_Signals', 0),
            'Current_RSI': round(accumulation_signals.get('Current_RSI', 50), 2),
            'Price_vs_200MA': round(accumulation_signals.get('Price_vs_200MA', 0) * 100, 2),
            'Volume_Ratio': round(accumulation_signals.get('Current_Volume_Ratio', 1), 2),
            'Bullish_Engulfing': candlestick_patterns.get('Bullish_Engulfing', 0),
            'Bearish_Engulfing': candlestick_patterns.get('Bearish_Engulfing', 0),
            'Hammer_Pattern': candlestick_patterns.get('Hammer', 0),
            'Doji_Pattern': candlestick_patterns.get('Doji', 0),
            'Data_Years': round(len(df) / 52, 1)  # Approximate years
        }
        
        return result

    def run_comprehensive_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting Comprehensive 20-Year Analysis...")
        print("=" * 60)
        
        # Store results
        all_results = []
        failed_stocks = []
        
        # Analyze each stock
        total_stocks = len(self.all_stocks)
        for i, symbol in enumerate(self.all_stocks, 1):
            print(f"\n📈 Progress: {i}/{total_stocks} ({i/total_stocks*100:.1f}%)")
            
            try:
                result = self.comprehensive_stock_analysis(symbol)
                if result:
                    all_results.append(result)
                    print(f"✅ {symbol}: Score={result['Accumulation_Score']:.3f}, Bullish={result['Bullish_Probability']:.3f}")
                else:
                    failed_stocks.append(symbol)
                    print(f"❌ {symbol}: No data available")
                    
            except Exception as e:
                failed_stocks.append(symbol)
                print(f"❌ {symbol}: Error - {str(e)}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Sector rotation analysis
        sector_rotation = self.sector_rotation_analysis()
        
        # Generate comprehensive report
        self.generate_comprehensive_report(results_df, sector_rotation, failed_stocks)
        
        return results_df, sector_rotation

    def generate_comprehensive_report(self, results_df, sector_rotation, failed_stocks):
        """Generate comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"midcap_smallcap_20year_analysis_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("🚀 COMPREHENSIVE INDIAN MID-CAP & SMALL-CAP 20-YEAR ANALYSIS 🚀\n")
            f.write("=" * 80 + "\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Timeframe: Weekly (20 Years)\n")
            f.write(f"Universe: Mid-Cap & Small-Cap Stocks\n\n")
            
            # Summary Statistics
            f.write("📊 ANALYSIS SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Stocks Analyzed: {len(results_df)}\n")
            f.write(f"Failed Analysis: {len(failed_stocks)}\n")
            f.write(f"Success Rate: {len(results_df)/(len(results_df)+len(failed_stocks))*100:.1f}%\n\n")
            
            if not results_df.empty:
                # Top Accumulation Opportunities
                f.write("🎯 TOP ACCUMULATION OPPORTUNITIES\n")
                f.write("-" * 40 + "\n")
                top_accumulation = results_df.nlargest(20, 'Accumulation_Score')
                for idx, row in top_accumulation.iterrows():
                    f.write(f"{row['Symbol']:15} | Sector: {row['Sector']:12} | "
                           f"Score: {row['Accumulation_Score']:.3f} | "
                           f"Bullish: {row['Bullish_Probability']:.3f} | "
                           f"Price: ₹{row['Current_Price']:.2f}\n")
                
                # Top Bullish Trend Candidates
                f.write(f"\n📈 TOP BULLISH TREND CANDIDATES\n")
                f.write("-" * 40 + "\n")
                top_bullish = results_df.nlargest(20, 'Bullish_Probability')
                for idx, row in top_bullish.iterrows():
                    f.write(f"{row['Symbol']:15} | Sector: {row['Sector']:12} | "
                           f"Bullish: {row['Bullish_Probability']:.3f} | "
                           f"RSI: {row['Current_RSI']:6.2f} | "
                           f"Volume: {row['Volume_Ratio']:.2f}x\n")
                
                # Sector-wise Analysis
                f.write(f"\n🏭 SECTOR-WISE PERFORMANCE\n")
                f.write("-" * 35 + "\n")
                sector_summary = results_df.groupby('Sector').agg({
                    'Accumulation_Score': ['mean', 'count'],
                    'Bullish_Probability': 'mean',
                    'Current_RSI': 'mean'
                }).round(3)
                
                for sector in sector_summary.index:
                    f.write(f"{sector:12} | Stocks: {sector_summary.loc[sector, ('Accumulation_Score', 'count')]:3.0f} | "
                           f"Avg Score: {sector_summary.loc[sector, ('Accumulation_Score', 'mean')]:.3f} | "
                           f"Bullish: {sector_summary.loc[sector, ('Bullish_Probability', 'mean')]:.3f}\n")
                
                # Sector Rotation Rankings
                f.write(f"\n🔄 SECTOR ROTATION RANKINGS\n")
                f.write("-" * 35 + "\n")
                for i, (sector, data) in enumerate(sector_rotation, 1):
                    f.write(f"{i:2}. {sector:12} | Score: {data['avg_score']:8.3f} | "
                           f"Positive: {data['positive_stocks']}/{data['total_stocks']} stocks\n")
                
                # Volume & Breakout Analysis
                f.write(f"\n🔊 HIGH VOLUME ACCUMULATION STOCKS\n")
                f.write("-" * 40 + "\n")
                high_volume = results_df[results_df['Volume_Ratio'] > 1.5].nlargest(15, 'Volume_Accumulation')
                for idx, row in high_volume.iterrows():
                    f.write(f"{row['Symbol']:15} | Volume: {row['Volume_Ratio']:5.2f}x | "
                           f"Accumulation: {row['Volume_Accumulation']:.3f} | "
                           f"Breakouts: {row['Breakout_Signals']}\n")
                
                # Pattern Analysis
                f.write(f"\n🕯️ CANDLESTICK PATTERN ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Stocks with Bullish Engulfing: {(results_df['Bullish_Engulfing'] > 0).sum()}\n")
                f.write(f"Stocks with Hammer Patterns: {(results_df['Hammer_Pattern'] > 0).sum()}\n")
                f.write(f"Stocks with Doji Patterns: {(results_df['Doji_Pattern'] > 0).sum()}\n")
                
                # Strong Technical Setup
                f.write(f"\n⚡ STRONG TECHNICAL SETUPS\n")
                f.write("-" * 35 + "\n")
                strong_setups = results_df[
                    (results_df['Accumulation_Score'] > 0.6) &
                    (results_df['Bullish_Probability'] > 0.7) &
                    (results_df['Current_RSI'] < 70)
                ].sort_values('Accumulation_Score', ascending=False)
                
                for idx, row in strong_setups.iterrows():
                    f.write(f"{row['Symbol']:15} | {row['Sector']:12} | "
                           f"Setup Score: {row['Accumulation_Score']:.3f} | "
                           f"RSI: {row['Current_RSI']:6.2f} | "
                           f"200MA: {row['Price_vs_200MA']:+6.2f}%\n")
                
                # Risk Analysis
                f.write(f"\n⚠️ RISK CONSIDERATIONS\n")
                f.write("-" * 25 + "\n")
                high_risk = results_df[
                    (results_df['Current_RSI'] > 70) |
                    (results_df['Price_vs_200MA'] < -20)
                ]
                f.write(f"Overbought Stocks (RSI > 70): {(results_df['Current_RSI'] > 70).sum()}\n")
                f.write(f"Weak vs 200MA (< -20%): {(results_df['Price_vs_200MA'] < -20).sum()}\n")
                
            # Failed Stocks
            if failed_stocks:
                f.write(f"\n❌ STOCKS WITH INSUFFICIENT DATA\n")
                f.write("-" * 40 + "\n")
                for stock in failed_stocks:
                    f.write(f"  • {stock}\n")
            
            f.write(f"\n" + "=" * 80 + "\n")
            f.write("📝 ANALYSIS COMPLETE - Data saved for further processing\n")
            f.write("=" * 80 + "\n")
        
        # Save detailed CSV
        csv_filename = f"midcap_smallcap_analysis_{timestamp}.csv"
        results_df.to_csv(csv_filename, index=False)
        
        print(f"\n✅ Analysis Complete!")
        print(f"📄 Report saved: {filename}")
        print(f"📊 Data saved: {csv_filename}")
        print(f"🎯 Total stocks analyzed: {len(results_df)}")
        print(f"⭐ Top accumulation opportunities found!")


# Main execution
if __name__ == "__main__":
    print("🚀 Initializing Comprehensive Stock Analyzer...")
    analyzer = ComprehensiveStockAnalyzer()
    
    # Run complete analysis
    results_df, sector_rotation = analyzer.run_comprehensive_analysis()
    
    print("\n" + "="*60)
    print("🎉 COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY! 🎉")
    print("="*60)