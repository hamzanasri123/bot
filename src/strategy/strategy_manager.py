"""
Strategy module for implementing trading strategies.
"""

import sys
import os
import pandas as pd
import numpy as np
from enum import Enum
import datetime

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STRATEGIES
from analysis.technical_analysis import SignalType

class PositionType(Enum):
    """Enum for position types."""
    LONG = 1
    SHORT = 2
    FLAT = 0

class TradeAction(Enum):
    """Enum for trade actions."""
    BUY = 1
    SELL = 2
    HOLD = 0

class TradeRecommendation:
    """Class to represent a trade recommendation."""
    
    def __init__(self, symbol, action, position_type, timestamp, price, 
                 confidence=1.0, strategy_name="", signal_source="", 
                 stop_loss=None, take_profit=None):
        """
        Initialize a trade recommendation.
        
        Args:
            symbol (str): The ticker symbol.
            action (TradeAction): The recommended action.
            position_type (PositionType): The position type.
            timestamp (datetime): The timestamp of the recommendation.
            price (float): The current price.
            confidence (float, optional): Confidence level (0.0-1.0). Defaults to 1.0.
            strategy_name (str, optional): Name of the strategy. Defaults to "".
            signal_source (str, optional): Source of the signal. Defaults to "".
            stop_loss (float, optional): Recommended stop loss price. Defaults to None.
            take_profit (float, optional): Recommended take profit price. Defaults to None.
        """
        self.symbol = symbol
        self.action = action
        self.position_type = position_type
        self.timestamp = timestamp
        self.price = price
        self.confidence = confidence
        self.strategy_name = strategy_name
        self.signal_source = signal_source
        self.stop_loss = stop_loss
        self.take_profit = take_profit
    
    def __str__(self):
        """String representation of the trade recommendation."""
        return (f"TradeRecommendation: {self.symbol} - {self.action.name} "
                f"({self.position_type.name}) at {self.price} - "
                f"Confidence: {self.confidence:.2f} - Strategy: {self.strategy_name}")

class Strategy:
    """Base class for trading strategies."""
    
    def __init__(self, name, config=None):
        """
        Initialize the strategy.
        
        Args:
            name (str): Strategy name.
            config (dict, optional): Strategy configuration. Defaults to None.
        """
        self.name = name
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
    
    def generate_signals(self, df):
        """
        Generate trading signals based on the strategy.
        
        Args:
            df (pandas.DataFrame): DataFrame with technical indicators.
            
        Returns:
            pandas.DataFrame: DataFrame with added signal column.
        """
        raise NotImplementedError("Subclasses must implement generate_signals method")
    
    def generate_recommendations(self, df, symbol):
        """
        Generate trade recommendations based on signals.
        
        Args:
            df (pandas.DataFrame): DataFrame with signals.
            symbol (str): The ticker symbol.
            
        Returns:
            list: List of TradeRecommendation objects.
        """
        raise NotImplementedError("Subclasses must implement generate_recommendations method")

class TrendFollowingStrategy(Strategy):
    """Trend following strategy implementation."""
    
    def __init__(self, config=None):
        """
        Initialize the trend following strategy.
        
        Args:
            config (dict, optional): Strategy configuration. Defaults to None.
        """
        super().__init__("trend_following", config or STRATEGIES.get('trend_following', {}))
    
    def generate_signals(self, df):
        """
        Generate trading signals based on trend following strategy.
        
        Args:
            df (pandas.DataFrame): DataFrame with technical indicators.
            
        Returns:
            pandas.DataFrame: DataFrame with added signal column.
        """
        if df is None or df.empty or not self.enabled:
            return df
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Check if we already have trend signals
        if 'trend_signal' in result.columns:
            return result
        
        # Check for required indicators
        required_indicators = ['sma_20', 'sma_50', 'macd', 'macd_signal']
        if not all(indicator in result.columns for indicator in required_indicators):
            raise ValueError(f"DataFrame must contain indicators: {required_indicators}")
        
        # Initialize signal column
        result['trend_signal'] = SignalType.NEUTRAL.value
        
        # Entry conditions
        entry_conditions = self.config.get('entry_conditions', {})
        
        # SMA alignment check (short > medium > long for uptrend)
        if entry_conditions.get('sma_alignment', False):
            result['sma_aligned_bullish'] = (
                (result['sma_20'] > result['sma_50']) & 
                (result['sma_50'] > result['sma_200'] if 'sma_200' in result.columns else True)
            )
        else:
            result['sma_aligned_bullish'] = True
        
        # MACD signal (MACD line crosses above signal line)
        if entry_conditions.get('macd_signal', False):
            result['macd_bullish_cross'] = (
                (result['macd'] > result['macd_signal']) & 
                (result['macd'].shift(1) <= result['macd_signal'].shift(1))
            )
        else:
            result['macd_bullish_cross'] = True
        
        # ADX filter (strong trend)
        if entry_conditions.get('adx_filter', False) and 'adx_14' in result.columns:
            adx_threshold = 25  # Default threshold
            result['strong_trend'] = result['adx_14'] > adx_threshold
        else:
            result['strong_trend'] = True
        
        # Exit conditions
        exit_conditions = self.config.get('exit_conditions', {})
        
        # SMA reversal (short SMA crosses below medium SMA)
        if exit_conditions.get('sma_reversal', False):
            result['sma_bearish_cross'] = (
                (result['sma_20'] < result['sma_50']) & 
                (result['sma_20'].shift(1) >= result['sma_50'].shift(1))
            )
        else:
            result['sma_bearish_cross'] = False
        
        # MACD reversal (MACD line crosses below signal line)
        if exit_conditions.get('macd_reversal', False):
            result['macd_bearish_cross'] = (
                (result['macd'] < result['macd_signal']) & 
                (result['macd'].shift(1) >= result['macd_signal'].shift(1))
            )
        else:
            result['macd_bearish_cross'] = False
        
        # Generate buy signals
        result.loc[
            result['sma_aligned_bullish'] & 
            result['macd_bullish_cross'] & 
            result['strong_trend'],
            'trend_signal'
        ] = SignalType.BUY.value
        
        # Generate sell signals
        result.loc[
            result['sma_bearish_cross'] | result['macd_bearish_cross'],
            'trend_signal'
        ] = SignalType.SELL.value
        
        return result
    
    def generate_recommendations(self, df, symbol):
        """
        Generate trade recommendations based on trend following signals.
        
        Args:
            df (pandas.DataFrame): DataFrame with signals.
            symbol (str): The ticker symbol.
            
        Returns:
            list: List of TradeRecommendation objects.
        """
        if df is None or df.empty or not self.enabled:
            return []
        
        # Check if we have trend signals
        if 'trend_signal' not in df.columns:
            df = self.generate_signals(df)
        
        recommendations = []
        
        # Process each row with a signal
        for idx, row in df.iterrows():
            if row['trend_signal'] == SignalType.BUY.value:
                # Calculate stop loss and take profit
                stop_loss = row['low'] * 0.98  # 2% below current low
                take_profit = row['close'] * 1.06  # 6% above current close
                
                # Create buy recommendation
                rec = TradeRecommendation(
                    symbol=symbol,
                    action=TradeAction.BUY,
                    position_type=PositionType.LONG,
                    timestamp=row['timestamp'],
                    price=row['close'],
                    confidence=0.8,
                    strategy_name=self.name,
                    signal_source="Trend Following",
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                recommendations.append(rec)
                
            elif row['trend_signal'] == SignalType.SELL.value:
                # Create sell recommendation
                rec = TradeRecommendation(
                    symbol=symbol,
                    action=TradeAction.SELL,
                    position_type=PositionType.FLAT,
                    timestamp=row['timestamp'],
                    price=row['close'],
                    confidence=0.7,
                    strategy_name=self.name,
                    signal_source="Trend Following"
                )
                recommendations.append(rec)
        
        return recommendations

class MeanReversionStrategy(Strategy):
    """Mean reversion strategy implementation."""
    
    def __init__(self, config=None):
        """
        Initialize the mean reversion strategy.
        
        Args:
            config (dict, optional): Strategy configuration. Defaults to None.
        """
        super().__init__("mean_reversion", config or STRATEGIES.get('mean_reversion', {}))
    
    def generate_signals(self, df):
        """
        Generate trading signals based on mean reversion strategy.
        
        Args:
            df (pandas.DataFrame): DataFrame with technical indicators.
            
        Returns:
            pandas.DataFrame: DataFrame with added signal column.
        """
        if df is None or df.empty or not self.enabled:
            return df
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Check if we already have reversion signals
        if 'reversion_signal' in result.columns:
            return result
        
        # Check for required indicators
        required_indicators = ['rsi_14', 'bb_lower', 'bb_middle', 'bb_upper']
        if not all(indicator in result.columns for indicator in required_indicators):
            raise ValueError(f"DataFrame must contain indicators: {required_indicators}")
        
        # Initialize signal column
        result['reversion_signal'] = SignalType.NEUTRAL.value
        
        # Entry conditions
        entry_conditions = self.config.get('entry_conditions', {})
        
        # RSI oversold condition
        if entry_conditions.get('rsi_oversold', False):
            rsi_oversold = 30  # Default threshold
            result['rsi_oversold'] = result['rsi_14'] < rsi_oversold
        else:
            result['rsi_oversold'] = True
        
        # Price touches lower Bollinger Band
        if entry_conditions.get('bollinger_lower', False):
            result['price_at_lower_band'] = result['close'] <= result['bb_lower']
        else:
            result['price_at_lower_band'] = True
        
        # Exit conditions
        exit_conditions = self.config.get('exit_conditions', {})
        
        # RSI overbought condition
        if exit_conditions.get('rsi_overbought', False):
            rsi_overbought = 70  # Default threshold
            result['rsi_overbought'] = result['rsi_14'] > rsi_overbought
        else:
            result['rsi_overbought'] = False
        
        # Price reaches middle Bollinger Band
        if exit_conditions.get('bollinger_middle', False):
            result['price_at_middle_band'] = (
                (result['close'] >= result['bb_middle'] * 0.99) & 
                (result['close'] <= result['bb_middle'] * 1.01)
            )
        else:
            result['price_at_middle_band'] = False
        
        # Generate buy signals
        result.loc[
            result['rsi_oversold'] & 
            result['price_at_lower_band'],
            'reversion_signal'
        ] = SignalType.BUY.value
        
        # Generate sell signals
        result.loc[
            result['rsi_overbought'] | 
            result['price_at_middle_band'],
            'reversion_signal'
        ] = SignalType.SELL.value
        
        return result
    
    def generate_recommendations(self, df, symbol):
        """
        Generate trade recommendations based on mean reversion signals.
        
        Args:
            df (pandas.DataFrame): DataFrame with signals.
            symbol (str): The ticker symbol.
            
        Returns:
            list: List of TradeRecommendation objects.
        """
        if df is None or df.empty or not self.enabled:
            return []
        
        # Check if we have reversion signals
        if 'reversion_signal' not in df.columns:
            df = self.generate_signals(df)
        
        recommendations = []
        
        # Process each row with a signal
        for idx, row in df.iterrows():
            if row['reversion_signal'] == SignalType.BUY.value:
                # Calculate stop loss and take profit
                stop_loss = row['low'] * 0.97  # 3% below current low
                take_profit = row['bb_middle'] if 'bb_middle' in row else row['close'] * 1.03
                
                # Create buy recommendation
                rec = TradeRecommendation(
                    symbol=symbol,
                    action=TradeAction.BUY,
                    position_type=PositionType.LONG,
                    timestamp=row['timestamp'],
                    price=row['close'],
                    confidence=0.7,
                    strategy_name=self.name,
                    signal_source="Mean Reversion",
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                recommendations.append(rec)
                
            elif row['reversion_signal'] == SignalType.SELL.value:
                # Create sell recommendation
                rec = TradeRecommendation(
                    symbol=symbol,
                    action=TradeAction.SELL,
                    position_type=PositionType.FLAT,
                    timestamp=row['timestamp'],
                    price=row['close'],
                    confidence=0.7,
                    strategy_name=self.name,
                    signal_source="Mean Reversion"
                )
                recommendations.append(rec)
        
        return recommendations

class StrategyManager:
    """Manages multiple trading strategies."""
    
    def __init__(self, config=None):
        """
        Initialize the strategy manager.
        
        Args:
            config (dict, optional): Strategy configuration. Defaults to None.
        """
        self.config = config or STRATEGIES
        self.strategies = {}
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize all enabled strategies."""
        # Add trend following strategy if enabled
        if self.config.get('trend_following', {}).get('enabled', True):
            self.strategies['trend_following'] = TrendFollowingStrategy(
                self.config.get('trend_following')
            )
        
        # Add mean reversion<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>