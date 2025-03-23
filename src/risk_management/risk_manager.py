"""
Risk management module for controlling trading risk.
"""

import sys
import os
import pandas as pd
import numpy as np
from enum import Enum
import datetime

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RISK_MANAGEMENT
from strategy.strategy_manager import TradeRecommendation, TradeAction, PositionType

class RiskAdjustedRecommendation:
    """Class to represent a risk-adjusted trade recommendation."""
    
    def __init__(self, original_recommendation, position_size, adjusted_stop_loss=None, 
                 adjusted_take_profit=None, risk_amount=0.0, expected_return=0.0):
        """
        Initialize a risk-adjusted trade recommendation.
        
        Args:
            original_recommendation (TradeRecommendation): The original trade recommendation.
            position_size (float): The recommended position size.
            adjusted_stop_loss (float, optional): Adjusted stop loss price. Defaults to None.
            adjusted_take_profit (float, optional): Adjusted take profit price. Defaults to None.
            risk_amount (float, optional): Amount at risk in currency. Defaults to 0.0.
            expected_return (float, optional): Expected return in currency. Defaults to 0.0.
        """
        self.original = original_recommendation
        self.position_size = position_size
        self.stop_loss = adjusted_stop_loss or original_recommendation.stop_loss
        self.take_profit = adjusted_take_profit or original_recommendation.take_profit
        self.risk_amount = risk_amount
        self.expected_return = expected_return
    
    def __str__(self):
        """String representation of the risk-adjusted recommendation."""
        return (f"RiskAdjustedRecommendation: {self.original.symbol} - {self.original.action.name} "
                f"Size: {self.position_size:.2f} - Risk: ${self.risk_amount:.2f} - "
                f"Stop: {self.stop_loss:.2f} - Target: {self.take_profit:.2f}")

class Portfolio:
    """Class to represent a trading portfolio."""
    
    def __init__(self, initial_capital=100000.0):
        """
        Initialize a portfolio.
        
        Args:
            initial_capital (float, optional): Initial capital. Defaults to 100000.0.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # symbol -> {'size': size, 'entry_price': price, 'current_price': price}
        self.closed_trades = []
        self.trade_history = []
    
    @property
    def equity(self):
        """Calculate the current portfolio equity."""
        position_value = sum(pos['size'] * pos['current_price'] for pos in self.positions.values())
        return self.cash + position_value
    
    @property
    def open_positions_count(self):
        """Get the number of open positions."""
        return len(self.positions)
    
    def update_position_price(self, symbol, current_price):
        """
        Update the current price of a position.
        
        Args:
            symbol (str): The ticker symbol.
            current_price (float): The current price.
        """
        if symbol in self.positions:
            self.positions[symbol]['current_price'] = current_price
    
    def open_position(self, symbol, size, entry_price, stop_loss=None, take_profit=None, timestamp=None):
        """
        Open a new position.
        
        Args:
            symbol (str): The ticker symbol.
            size (float): Position size in shares/contracts.
            entry_price (float): Entry price per share/contract.
            stop_loss (float, optional): Stop loss price. Defaults to None.
            take_profit (float, optional): Take profit price. Defaults to None.
            timestamp (datetime, optional): Trade timestamp. Defaults to None.
            
        Returns:
            bool: True if position opened successfully, False otherwise.
        """
        cost = size * entry_price
        
        if cost > self.cash:
            return False
        
        self.cash -= cost
        
        self.positions[symbol] = {
            'size': size,
            'entry_price': entry_price,
            'current_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': timestamp or datetime.datetime.now()
        }
        
        self.trade_history.append({
            'symbol': symbol,
            'action': 'BUY',
            'size': size,
            'price': entry_price,
            'timestamp': timestamp or datetime.datetime.now(),
            'value': cost
        })
        
        return True
    
    def close_position(self, symbol, exit_price, timestamp=None):
        """
        Close an existing position.
        
        Args:
            symbol (str): The ticker symbol.
            exit_price (float): Exit price per share/contract.
            timestamp (datetime, optional): Trade timestamp. Defaults to None.
            
        Returns:
            dict: Closed trade details or None if position not found.
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        proceeds = position['size'] * exit_price
        self.cash += proceeds
        
        entry_value = position['size'] * position['entry_price']
        exit_value = position['size'] * exit_price
        profit_loss = exit_value - entry_value
        profit_loss_pct = (exit_price / position['entry_price'] - 1) * 100
        
        closed_trade = {
            'symbol': symbol,
            'entry_time': position['entry_time'],
            'entry_price': position['entry_price'],
            'exit_time': timestamp or datetime.datetime.now(),
            'exit_price': exit_price,
            'size': position['size'],
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct
        }
        
        self.closed_trades.append(closed_trade)
        
        self.trade_history.append({
            'symbol': symbol,
            'action': 'SELL',
            'size': position['size'],
            'price': exit_price,
            'timestamp': timestamp or datetime.datetime.now(),
            'value': proceeds
        })
        
        del self.positions[symbol]
        
        return closed_trade
    
    def get_position(self, symbol):
        """
        Get position details for a symbol.
        
        Args:
            symbol (str): The ticker symbol.
            
        Returns:
            dict: Position details or None if not found.
        """
        return self.positions.get(symbol)
    
    def has_position(self, symbol):
        """
        Check if a position exists for a symbol.
        
        Args:
            symbol (str): The ticker symbol.
            
        Returns:
            bool: True if position exists, False otherwise.
        """
        return symbol in self.positions
    
    def get_portfolio_summary(self):
        """
        Get a summary of the portfolio.
        
        Returns:
            dict: Portfolio summary.
        """
        open_positions_value = sum(pos['size'] * pos['current_price'] for pos in self.positions.values())
        total_profit_loss = sum(trade['profit_loss'] for trade in self.closed_trades)
        
        return {
            'initial_capital': self.initial_capital,
            'current_equity': self.equity,
            'cash': self.cash,
            'open_positions_value': open_positions_value,
            'open_positions_count': self.open_positions_count,
            'total_profit_loss': total_profit_loss,
            'return_pct': (self.equity / self.initial_capital - 1) * 100
        }

class RiskManager:
    """Manages trading risk and position sizing."""
    
    def __init__(self, portfolio, config=None):
        """
        Initialize the risk manager.
        
        Args:
            portfolio (Portfolio): The trading portfolio.
            config (dict, optional): Risk management configuration. Defaults to None.
        """
        self.portfolio = portfolio
        self.config = config or RISK_MANAGEMENT
        self.daily_loss = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = portfolio.equity
    
    def calculate_position_size(self, recommendation):
        """
        Calculate the appropriate position size for a trade recommendation.
        
        Args:
            recommendation (TradeRecommendation): The trade recommendation.
            
        Returns:
            float: Recommended position size in shares/contracts.
        """
        if recommendation.action != TradeAction.BUY:
            return 0.0
        
        # Get risk parameters
        max_position_risk = self.config['max_position_size']
        
        # Calculate risk per share
        if recommendation.stop_loss is None:
            # Use default stop loss percentage if not provided
            stop_loss_pct = self.config['stop_loss_pct']
            stop_loss = recommendation.price * (1 - stop_loss_pct)
        else:
            stop_loss = recommendation.stop_loss
        
        risk_per_share = recommendation.price - stop_loss
        
        # Calculate position size based on risk
        portfolio_value = self.portfolio.equity
        max_risk_amount = portfolio_value * max_position_risk
        
        # Adjust for drawdown if necessary
        if self.max_drawdown >= self.config['drawdown_control']['max_drawdown']:
            max_risk_amount *= self.config['drawdown_control']['position_size_reduction']
        
        # Check if we've hit daily loss limit
        if self.daily_loss >= portfolio_value * self.config['drawdown_control']['daily_loss_limit']:
            return 0.0  # No trading if daily loss limit reached
        
        # Calculate shares based on risk
        if risk_per_share > 0:
            shares = max_risk_amount / risk_per_share
        else:
            shares = 0
        
        # Calculate maximum shares based on portfolio allocation
        max_position_value = portfolio_value * max_position_risk
        max_shares_by_value = max_position_value / recommendation.price
        
        # Take the minimum of the two calculations
        shares = min(shares, max_shares_by_value)
        
        # Check if we have enough cash
        if shares * recommendation.price > self.portfolio.cash:
            shares = self.portfolio.cash / recommendation.price
        
        # Round down to whole shares
        shares = int(shares)
        
        return shares
    
    def adjust_recommendation(self, recommendation):
        """
        Apply risk management to a trade recommendation.
        
        Args:
            recommendation (TradeRecommendation): The trade recommendation.
            
        Returns:
            RiskAdjustedRecommendation: Risk-adjusted recommendation.
        """
        if recommendation.action == TradeAction.BUY:
            # Calculate position size
            position_size = self.calculate_position_size(recommendation)
            
            # Calculate risk amount
            if recommendation.stop_loss is not None:
                risk_per_share = recommendation.price - recommendation.stop_loss
                risk_amount = position_size * risk_per_share
            else:
                # Use default stop loss percentage
                stop_loss_pct = self.config['stop_loss_pct']
                risk_amount = position_size * recommendation.price * stop_loss_pct
            
            # Calculate expected return
            if recommendation.take_profit is not None:
                return_per_share = recommendation.take_profit - recommendation.price
                expected_return = position_size * return_per_share
            else:
                # Use a default risk-reward ratio of 2:1
                expected_return = risk_amount * 2
            
            # Create adjusted recommendation
            return RiskAdjustedRecommendation(
                original_recommendation=recommendation,
                position_size=position_size,
                risk_amount=risk_amount,
                expected_return=expected_return
            )
            
        elif recommendation.action == TradeAction.SELL:
            # For sell recommendations, just pass through
            return RiskAdjustedRecommendation(
                original_recommendation=recommendation,
                position_size=0  # Sell entire position
            )
        
        else:  # HOLD
            return RiskAdjustedRecommendation(
                original_recommendation=recommendation,
                position_size=0
            )
    
    def update_portfolio_metrics(self):
        """Update portfolio metrics for risk management."""
        current_equity = self.portfolio.equity
        
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate drawdown
        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def process_recommendations(self, recommendations):
        """
        Process a list of trade recommendations.
        
        Args:
            recommendations (list): List of TradeRecommendation objects.
            
        Returns:
            list: List of RiskAdjustedRecommendation objects.
        """
        # Update portfolio metrics
        self.update_portfolio_metrics()
        
        # Check if we've hit maximum positions
        if (self.portfolio.open_positions_count >= self.config['max_positions'] and 
            not any(r.action == TradeAction.SELL for r in recommendations)):
            return []
        
        # Process each recommendation
        adjusted_recommendations = []
        for rec in recommendations:
            # Skip if we already have a position for this symbol and it's a buy
            if (rec.action == TradeAction.BUY and 
                self.portfolio.has_position(rec.symbol)):
                continue
                
            # Skip if we don't have a position for this symbol and it's a sell
            if (rec.action == TradeAction.SELL and 
                not self.portfolio.has_position(rec.symbol)):
                continue
            
            # Adjust recommendation
            adjusted_rec = self.adjust_recommendation(rec)
            
            # Add to list if valid
            if (adjusted_rec.position_size > 0 or 
                adjusted_rec.original.action == TradeAction.SELL):
                adjusted_recommendations.append(adjusted_rec)
        
        return adjusted_recommendations
    
    def apply_trailing_stops(self, current_prices):
        """
        Apply trailing stops to open positions.
        
        Args:
            current_prices (dict): Dictionary of current prices by symbol.
            
        Returns:
            list: List of symbols to close due to trailing stop.
        """
        symbols_to_close = []
        
        for symbol, position in self.portfolio.positions.items():
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            entry_price = position['entry_price']
            
            # Update current price in position
            self.portfolio.update_position_price(symbol, current_price)
            
            # Check if we should activate trailing stop
            profit_pct = (current_price / entry_price - 1)
            
            if profit_pct >= self.config['trailing_stop_activation']:
                # Calculate trailing stop level
                trailing_stop = current_price * (1 - self.config['trailing_stop_distance'])
   <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>