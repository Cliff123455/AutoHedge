import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

@dataclass
class TechnicalSignal:
    """Represents a technical analysis signal"""
    indicator: str
    signal: str  # 'BUY', 'SELL', 'NEUTRAL'
    strength: float  # 0-1 confidence score
    value: float
    threshold: Optional[float] = None
    timestamp: str = None

class TechnicalAnalyzer:
    """
    Advanced technical analysis engine for trading algorithms
    
    Implements multiple technical indicators and generates trading signals
    Uses pandas and numpy for calculations instead of TA-Lib for better compatibility
    """
    
    def __init__(self):
        self.signals_history: List[Dict] = []
        
    def analyze_stock(self, df: pd.DataFrame, symbol: str) -> Dict[str, TechnicalSignal]:
        """
        Comprehensive technical analysis of a stock
        
        Args:
            df: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
            symbol: Stock symbol
            
        Returns:
            Dictionary of technical signals
        """
        signals = {}
        
        # Ensure we have enough data
        if len(df) < 50:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} bars")
            return signals
            
        try:
            # Price action indicators
            signals.update(self._analyze_moving_averages(df))
            signals.update(self._analyze_rsi(df))
            signals.update(self._analyze_macd(df))
            signals.update(self._analyze_bollinger_bands(df))
            signals.update(self._analyze_stochastic(df))
            signals.update(self._analyze_atr(df))
            
            # Volume indicators
            signals.update(self._analyze_volume(df))
            
            # Support/Resistance
            signals.update(self._analyze_support_resistance(df))
            
            # Trend analysis
            signals.update(self._analyze_trend_strength(df))
            
            logger.info(f"Generated {len(signals)} technical signals for {symbol}")
            
        except Exception as e:
            logger.error(f"Error in technical analysis for {symbol}: {e}")
            
        return signals
    
    def _analyze_moving_averages(self, df: pd.DataFrame) -> Dict[str, TechnicalSignal]:
        """Moving average crossover and trend analysis"""
        signals = {}
        
        # Calculate moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        current_price = df['Close'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        
        # SMA Crossover Signal
        if current_price > sma_20 > sma_50:
            signal = 'BUY'
            strength = min(1.0, (current_price - sma_20) / sma_20 * 10)
        elif current_price < sma_20 < sma_50:
            signal = 'SELL'
            strength = min(1.0, (sma_20 - current_price) / current_price * 10)
        else:
            signal = 'NEUTRAL'
            strength = 0.3
            
        signals['sma_crossover'] = TechnicalSignal(
            indicator='SMA_Crossover',
            signal=signal,
            strength=strength,
            value=current_price / sma_20,
            threshold=1.0
        )
        
        return signals
    
    def _analyze_rsi(self, df: pd.DataFrame) -> Dict[str, TechnicalSignal]:
        """Relative Strength Index analysis"""
        signals = {}
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        if current_rsi > 70:
            signal = 'SELL'
            strength = min(1.0, (current_rsi - 70) / 30)
        elif current_rsi < 30:
            signal = 'BUY'
            strength = min(1.0, (30 - current_rsi) / 30)
        else:
            signal = 'NEUTRAL'
            strength = 0.2
            
        signals['rsi'] = TechnicalSignal(
            indicator='RSI',
            signal=signal,
            strength=strength,
            value=current_rsi,
            threshold=70 if signal == 'SELL' else 30
        )
        
        return signals
    
    def _analyze_macd(self, df: pd.DataFrame) -> Dict[str, TechnicalSignal]:
        """MACD trend and momentum analysis"""
        signals = {}
        
        # Calculate MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        macd_hist = macd - macd_signal
        
        current_macd = macd.iloc[-1]
        current_signal = macd_signal.iloc[-1]
        current_hist = macd_hist.iloc[-1]
        prev_hist = macd_hist.iloc[-2] if len(macd_hist) > 1 else 0
        
        # MACD crossover
        if current_hist > 0 and prev_hist <= 0:
            signal = 'BUY'
            strength = min(1.0, abs(current_hist) * 100)
        elif current_hist < 0 and prev_hist >= 0:
            signal = 'SELL'
            strength = min(1.0, abs(current_hist) * 100)
        else:
            signal = 'NEUTRAL'
            strength = 0.3
            
        signals['macd'] = TechnicalSignal(
            indicator='MACD',
            signal=signal,
            strength=strength,
            value=current_macd,
            threshold=current_signal
        )
        
        return signals
    
    def _analyze_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, TechnicalSignal]:
        """Bollinger Bands mean reversion analysis"""
        signals = {}
        
        # Calculate Bollinger Bands
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        upper_band = sma_20 + (std_20 * 2)
        lower_band = sma_20 - (std_20 * 2)
        
        current_price = df['Close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_middle = sma_20.iloc[-1]
        
        # Calculate position within bands
        band_position = (current_price - current_lower) / (current_upper - current_lower)
        
        if band_position > 0.8:  # Near upper band
            signal = 'SELL'
            strength = min(1.0, (band_position - 0.8) / 0.2)
        elif band_position < 0.2:  # Near lower band
            signal = 'BUY'
            strength = min(1.0, (0.2 - band_position) / 0.2)
        else:
            signal = 'NEUTRAL'
            strength = 0.2
            
        signals['bollinger'] = TechnicalSignal(
            indicator='Bollinger_Bands',
            signal=signal,
            strength=strength,
            value=band_position,
            threshold=0.8 if signal == 'SELL' else 0.2
        )
        
        return signals
    
    def _analyze_stochastic(self, df: pd.DataFrame) -> Dict[str, TechnicalSignal]:
        """Stochastic oscillator analysis"""
        signals = {}
        
        # Calculate Stochastic
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        
        k_percent = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        d_percent = k_percent.rolling(window=3).mean()
        
        current_k = k_percent.iloc[-1]
        current_d = d_percent.iloc[-1]
        
        if current_k > 80 and current_d > 80:
            signal = 'SELL'
            strength = min(1.0, (current_k - 80) / 20)
        elif current_k < 20 and current_d < 20:
            signal = 'BUY'
            strength = min(1.0, (20 - current_k) / 20)
        else:
            signal = 'NEUTRAL'
            strength = 0.2
            
        signals['stochastic'] = TechnicalSignal(
            indicator='Stochastic',
            signal=signal,
            strength=strength,
            value=current_k,
            threshold=80 if signal == 'SELL' else 20
        )
        
        return signals
    
    def _analyze_atr(self, df: pd.DataFrame) -> Dict[str, TechnicalSignal]:
        """Average True Range volatility analysis"""
        signals = {}
        
        # Calculate ATR
        high_low = df['High'] - df['Low']
        high_close_prev = np.abs(df['High'] - df['Close'].shift())
        low_close_prev = np.abs(df['Low'] - df['Close'].shift())
        
        tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = tr.rolling(window=14).mean()
        
        current_atr = atr.iloc[-1]
        atr_ma = atr.rolling(20).mean().iloc[-1]
        
        volatility_ratio = current_atr / atr_ma if atr_ma > 0 else 1.0
        
        # High volatility might indicate trend change
        if volatility_ratio > 1.5:
            signal = 'NEUTRAL'  # High volatility = uncertainty
            strength = 0.7
        else:
            signal = 'NEUTRAL'
            strength = 0.3
            
        signals['volatility'] = TechnicalSignal(
            indicator='ATR_Volatility',
            signal=signal,
            strength=strength,
            value=volatility_ratio,
            threshold=1.5
        )
        
        return signals
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, TechnicalSignal]:
        """Volume analysis"""
        signals = {}
        
        if 'Volume' not in df.columns:
            return signals
        
        volume_ma = df['Volume'].rolling(20).mean()
        current_volume = df['Volume'].iloc[-1]
        avg_volume = volume_ma.iloc[-1]
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume confirmation
        if volume_ratio > 1.5:  # High volume
            signal = 'BUY'  # High volume often confirms moves
            strength = min(1.0, (volume_ratio - 1.5) / 2)
        else:
            signal = 'NEUTRAL'
            strength = 0.3
            
        signals['volume'] = TechnicalSignal(
            indicator='Volume_Confirmation',
            signal=signal,
            strength=strength,
            value=volume_ratio,
            threshold=1.5
        )
        
        return signals
    
    def _analyze_support_resistance(self, df: pd.DataFrame) -> Dict[str, TechnicalSignal]:
        """Support and resistance level analysis"""
        signals = {}
        
        # Simple support/resistance using rolling min/max
        high_20 = df['High'].rolling(20).max()
        low_20 = df['Low'].rolling(20).min()
        
        current_price = df['Close'].iloc[-1]
        resistance = high_20.iloc[-1]
        support = low_20.iloc[-1]
        
        # Distance from support/resistance
        support_dist = (current_price - support) / support if support > 0 else 0
        resistance_dist = (resistance - current_price) / current_price if current_price > 0 else 0
        
        if resistance_dist < 0.02:  # Within 2% of resistance
            signal = 'SELL'
            strength = 0.6
        elif support_dist < 0.02:  # Within 2% of support
            signal = 'BUY'
            strength = 0.6
        else:
            signal = 'NEUTRAL'
            strength = 0.2
            
        signals['support_resistance'] = TechnicalSignal(
            indicator='Support_Resistance',
            signal=signal,
            strength=strength,
            value=current_price,
            threshold=resistance if signal == 'SELL' else support
        )
        
        return signals
    
    def _analyze_trend_strength(self, df: pd.DataFrame) -> Dict[str, TechnicalSignal]:
        """Trend strength analysis using custom ADX calculation"""
        signals = {}
        
        try:
            # Calculate directional movement
            high_diff = df['High'].diff()
            low_diff = -df['Low'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            # Calculate ATR for ADX
            high_low = df['High'] - df['Low']
            high_close_prev = np.abs(df['High'] - df['Close'].shift())
            low_close_prev = np.abs(df['Low'] - df['Close'].shift())
            tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            
            # Smooth the values
            atr_14 = pd.Series(tr).rolling(window=14).mean()
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=14).mean() / atr_14)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=14).mean() / atr_14)
            
            # Calculate ADX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=14).mean()
            
            current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25
            current_plus_di = plus_di.iloc[-1] if not pd.isna(plus_di.iloc[-1]) else 20
            current_minus_di = minus_di.iloc[-1] if not pd.isna(minus_di.iloc[-1]) else 20
            
            if current_adx > 25:  # Strong trend
                if current_plus_di > current_minus_di:
                    signal = 'BUY'
                else:
                    signal = 'SELL'
                strength = min(1.0, current_adx / 50)
            else:
                signal = 'NEUTRAL'
                strength = 0.3
                
            signals['trend_strength'] = TechnicalSignal(
                indicator='ADX_Trend',
                signal=signal,
                strength=strength,
                value=current_adx,
                threshold=25
            )
            
        except Exception as e:
            logger.warning(f"Error calculating trend strength: {e}")
            signals['trend_strength'] = TechnicalSignal(
                indicator='ADX_Trend',
                signal='NEUTRAL',
                strength=0.3,
                value=25,
                threshold=25
            )
        
        return signals
    
    def get_consensus_signal(self, signals: Dict[str, TechnicalSignal]) -> TechnicalSignal:
        """
        Generate consensus signal from all technical indicators
        
        Args:
            signals: Dictionary of individual technical signals
            
        Returns:
            Consensus technical signal
        """
        if not signals:
            return TechnicalSignal('Consensus', 'NEUTRAL', 0.0, 0.0)
        
        buy_signals = []
        sell_signals = []
        neutral_signals = []
        
        for signal in signals.values():
            if signal.signal == 'BUY':
                buy_signals.append(signal.strength)
            elif signal.signal == 'SELL':
                sell_signals.append(signal.strength)
            else:
                neutral_signals.append(signal.strength)
        
        # Calculate weighted consensus
        buy_score = sum(buy_signals) / len(signals)
        sell_score = sum(sell_signals) / len(signals)
        
        if buy_score > sell_score and buy_score > 0.3:
            consensus_signal = 'BUY'
            consensus_strength = buy_score
        elif sell_score > buy_score and sell_score > 0.3:
            consensus_signal = 'SELL'
            consensus_strength = sell_score
        else:
            consensus_signal = 'NEUTRAL'
            consensus_strength = 0.3
        
        return TechnicalSignal(
            indicator='Technical_Consensus',
            signal=consensus_signal,
            strength=consensus_strength,
            value=buy_score - sell_score,
            threshold=0.3
        ) 