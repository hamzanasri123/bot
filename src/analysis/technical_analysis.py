"""
Technical analysis module for calculating indicators and generating signals.
"""

import sys
import os
import pandas as pd
import numpy as np
import talib
from enum import Enum

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INDICATORS

class SignalType(Enum):
    """Enum for signal types."""
    BUY = 1
    SELL = 2
    NEUTRAL = 0

class TechnicalAnalysis:
    """
    Handles calculation of technical indicators and generation of trading signals.
    """
    
    def __init__(self, config=None):
        """
        Initialize the TechnicalAnalysis module.
        
        Args:
            config (dict, optional): Configuration dictionary. Defaults to None.
        """
        self.config = config or INDICATORS
    
    def add_indicators(self, df):
        """
        Add technical indicators to the dataframe.
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data.
            
        Returns:
            pandas.DataFrame: DataFrame with added indicators.
        """
        if df is None or df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in result.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        # Add Simple Moving Averages
        if 'SMA' in self.config:
            for period_name, period in self.config['SMA'].items():
                result[f'sma_{period}'] = talib.SMA(result['close'].values, timeperiod=period)
        
        # Add Exponential Moving Averages
        if 'EMA' in self.config:
            for period_name, period in self.config['EMA'].items():
                result[f'ema_{period}'] = talib.EMA(result['close'].values, timeperiod=period)
        
        # Add Relative Strength Index
        if 'RSI' in self.config:
            period = self.config['RSI']['period']
            result[f'rsi_{period}'] = talib.RSI(result['close'].values, timeperiod=period)
        
        # Add MACD
        if 'MACD' in self.config:
            fast_period = self.config['MACD']['fast_period']
            slow_period = self.config['MACD']['slow_period']
            signal_period = self.config['MACD']['signal_period']
            
            macd, macd_signal, macd_hist = talib.MACD(
                result['close'].values,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period
            )
            
            result['macd'] = macd
            result['macd_signal'] = macd_signal
            result['macd_hist'] = macd_hist
        
        # Add Bollinger Bands
        if 'BOLLINGER' in self.config:
            period = self.config['BOLLINGER']['period']
            std_dev = self.config['BOLLINGER']['std_dev']
            
            upper, middle, lower = talib.BBANDS(
                result['close'].values,
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev,
                matype=0
            )
            
            result['bb_upper'] = upper
            result['bb_middle'] = middle
            result['bb_lower'] = lower
            
            # Calculate Bollinger Band width
            result['bb_width'] = (upper - lower) / middle
        
        # Add Average Directional Index
        if 'ADX' in self.config:
            period = self.config['ADX']['period']
            result[f'adx_{period}'] = talib.ADX(
                result['high'].values,
                result['low'].values,
                result['close'].values,
                timeperiod=period
            )
            
            # Add Directional Movement Index
            result['plus_di'] = talib.PLUS_DI(
                result['high'].values,
                result['low'].values,
                result['close'].values,
                timeperiod=period
            )
            
            result['minus_di'] = talib.MINUS_DI(
                result['high'].values,
                result['low'].values,
                result['close'].values,
                timeperiod=period
            )
        
        # Add Average True Range for volatility measurement
        result['atr_14'] = talib.ATR(
            result['high'].values,
            result['low'].values,
            result['close'].values,
            timeperiod=14
        )
        
        # Add Stochastic Oscillator
        slowk, slowd = talib.STOCH(
            result['high'].values,
            result['low'].values,
            result['close'].values,
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
        
        result['stoch_k'] = slowk
        result['stoch_d'] = slowd
        
        # Add On-Balance Volume
        result['obv'] = talib.OBV(result['close'].values, result['volume'].values)
        
        # Drop rows with NaN values that result from indicator calculations
        result = result.dropna()
        
        return result
    
    def generate_trend_following_signals(self, df):
        """
        Generate signals for trend following strategy.
        
        Args:
            df (pandas.DataFrame): DataFrame with technical indicators.
            
        Returns:
            pandas.DataFrame: DataFrame with added signal column.
        """
        if df is None or df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Initialize signal column
        result['trend_signal'] = SignalType.NEUTRAL.value
        
        # Check for required indicators
        required_indicators = ['sma_20', 'sma_50', 'sma_200', 'macd', 'macd_signal', 'adx_14']
        if not all(indicator in result.columns for indicator in required_indicators):
            raise ValueError(f"DataFrame must contain indicators: {required_indicators}")
        
        # SMA alignment check (short > medium > long for uptrend)
        result['sma_aligned_bullish'] = (
            (result['sma_20'] > result['sma_50']) & 
            (result['sma_50'] > result['sma_200'])
        )
        
        # SMA alignment check (short < medium < long for downtrend)
        result['sma_aligned_bearish'] = (
            (result['sma_20'] < result['sma_50']) & 
            (result['sma_50'] < result['sma_200'])
        )
        
        # MACD signal (MACD line crosses above signal line)
        result['macd_bullish_cross'] = (
            (result['macd'] > result['macd_signal']) & 
            (result['macd'].shift(1) <= result['macd_signal'].shift(1))
        )
        
        # MACD signal (MACD line crosses below signal line)
        result['macd_bearish_cross'] = (
            (result['macd'] < result['macd_signal']) & 
            (result['macd'].shift(1) >= result['macd_signal'].shift(1))
        )
        
        # ADX filter (strong trend)
        adx_threshold = self.config['ADX']['threshold']
        result['strong_trend'] = result[f'adx_{self.config["ADX"]["period"]}'] > adx_threshold
        
        # Generate buy signals
        result.loc[
            result['sma_aligned_bullish'] & 
            result['macd_bullish_cross'] & 
            result['strong_trend'],
            'trend_signal'
        ] = SignalType.BUY.value
        
        # Generate sell signals
        result.loc[
            (result['sma_aligned_bearish'] | 
             result['macd_bearish_cross']) & 
            result['strong_trend'],
            'trend_signal'
        ] = SignalType.SELL.value
        
        return result
    
    def generate_mean_reversion_signals(self, df):
        """
        Generate signals for mean reversion strategy.
        
        Args:
            df (pandas.DataFrame): DataFrame with technical indicators.
            
        Returns:
            pandas.DataFrame: DataFrame with added signal column.
        """
        if df is None or df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Initialize signal column
        result['reversion_signal'] = SignalType.NEUTRAL.value
        
        # Check for required indicators
        required_indicators = ['rsi_14', 'bb_upper', 'bb_middle', 'bb_lower']
        if not all(indicator in result.columns for indicator in required_indicators):
            raise ValueError(f"DataFrame must contain indicators: {required_indicators}")
        
        # RSI oversold condition
        rsi_oversold = self.config['RSI']['oversold']
        result['rsi_oversold'] = result[f'rsi_{self.config["RSI"]["period"]}'] < rsi_oversold
        
        # RSI overbought condition
        rsi_overbought = self.config['RSI']['overbought']
        result['rsi_overbought'] = result[f'rsi_{self.config["RSI"]["period"]}'] > rsi_overbought
        
        # Price touches lower Bollinger Band
        result['price_at_lower_band'] = result['close'] <= result['bb_lower']
        
        # Price touches upper Bollinger Band
        result['price_at_upper_band'] = result['close'] >= result['bb_upper']
        
        # Price reaches middle Bollinger Band (for exit)
        result['price_at_middle_band'] = (
            (result['close'] >= result['bb_middle'] * 0.99) & 
            (result['close'] <= result['bb_middle'] * 1.01)
        )
        
        # Generate buy signals
        result.loc[
            result['rsi_oversold'] & 
            result['price_at_lower_band'],
            'reversion_signal'
        ] = SignalType.BUY.value
        
        # Generate sell signals
        result.loc[
            (result['rsi_overbought'] & result['price_at_upper_band']) | 
            (result['reversion_signal'].shift(1) == SignalType.BUY.value & result['price_at_middle_band']),
            'reversion_signal'
        ] = SignalType.SELL.value
        
        return result
    
    def combine_signals(self, df):
        """
        Combine signals from different strategies.
        
        Args:
            df (pandas.DataFrame): DataFrame with strategy signals.
            
        Returns:
            pandas.DataFrame: DataFrame with combined signal column.
        """
        if df is None or df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Initialize combined signal column
        result['combined_signal'] = SignalType.NEUTRAL.value
        
        # Check if we have the strategy signals
        has_trend = 'trend_signal' in result.columns
        has_reversion = 'reversion_signal' in result.columns
        
        if not (has_trend or has_reversion):
            return result
        
        # Combine signals with priority to trend following in strong trends
        if has_trend and has_reversion:
            # Strong trend - use trend following signals
            strong_trend_mask = result['strong_trend'] if 'strong_trend' in result.columns else False
            
            # Use trend signals in strong trends
            result.loc[strong_trend_mask, 'combined_signal'] = result.loc[strong_trend_mask, 'trend_signal']
            
            # Use mean reversion signals in weak trends
            result.loc[~strong_trend_mask, 'combined_signal'] = result.loc[~strong_trend_mask, 'reversion_signal']
            
            # If both strategies agree, strengthen the signal
            agreement_mask = (
                (result['trend_signal'] == result['reversion_signal']) & 
                (result['trend_signal'] != SignalType.NEUTRAL.value)
            )
            result.loc[agreement_mask, 'combined_signal'] = result.loc[agreement_mask, 'trend_signal']
            
        elif has_trend:
            result['combined_signal'] = result['trend_signal']
        elif has_reversion:
            result['combined_signal'] = result['reversion_signal']
        
        return result
    
    def analyze(self, df):
        """
        Perform full technical analysis on the dataframe.
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data.
            
        Returns:
            pandas.DataFrame: DataFrame with indicators and signals.
        """
        if df is None or df.empty:
            return df
        
        # Add technical indicators
        result = self.add_indicators(df)
        
        # Generate strategy signals
        result = self.generate_trend_following_signals(result)
        result = self.generate_mean_reversion_signals(result)
        
        # Combine signals
        result = self.combine_signals(result)
        
        return result


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    # Import data manager
    from data.data_manager import DataManager
    
    # Initialize DataManager
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    data_manager = DataManager(str(data_dir / "tradingbot.db"))
    
    # Get historical data for a symbol
    df = data_manager.get_historical_data("AAPL", timeframe="1d")
    
    if df is not None and not df.empty:
        # Initialize TechnicalAnalysis
        ta = TechnicalAnalysis()
        
        # Analyze data
        result = ta.analyze(df)
        
        # Print results
        print(f"Analyzed {len(result)} records for AAPL")
        print(result[['timestamp', 'close', 'sma_20', 'sma_50', 'rsi_14', 'macd', 'trend_signal', 'reversion_signal', 'combined_signal']].tail(10))
        
        # Count signals
        buy_signals = (result['combined_signal'] == SignalType.BUY.value).sum()
        sell_signals = (result['combined_signal'] == SignalType.SELL.value).sum()
        print(f"Buy signals: {buy_signals}, Sell signals: {sell_signals}")
    
    # Close connection
    data_manager.close()
