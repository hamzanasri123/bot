"""
Execution module for handling trade execution.
"""

import sys
import os
import pandas as pd
import numpy as np
import datetime
import time
import json
import logging
from pathlib import Path

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RISK_MANAGEMENT
from strategy.strategy_manager import TradeAction, PositionType
from risk_management.risk_manager import RiskAdjustedRecommendation

class TradeExecutor:
    """Handles trade execution and order management."""
    
    def __init__(self, portfolio, risk_manager):
        """
        Initialize the trade executor.
        
        Args:
            portfolio: The trading portfolio.
            risk_manager: The risk manager.
        """
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.pending_orders = []
        self.executed_orders = []
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Set up logging for trade execution."""
        logger = logging.getLogger('trade_executor')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        logs_dir = Path(__file__).parent.parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Create file handler
        fh = logging.FileHandler(logs_dir / "trades.log")
        fh.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(fh)
        
        return logger
    
    def execute_recommendation(self, recommendation, current_price=None):
        """
        Execute a risk-adjusted trade recommendation.
        
        Args:
            recommendation (RiskAdjustedRecommendation): The risk-adjusted recommendation.
            current_price (float, optional): Current price to use for execution. Defaults to None.
            
        Returns:
            dict: Execution result.
        """
        symbol = recommendation.original.symbol
        action = recommendation.original.action
        timestamp = recommendation.original.timestamp or datetime.datetime.now()
        
        # Use provided price or recommendation price
        price = current_price or recommendation.original.price
        
        result = {
            'symbol': symbol,
            'action': action.name,
            'timestamp': timestamp,
            'price': price,
            'status': 'FAILED',
            'message': ''
        }
        
        try:
            if action == TradeAction.BUY:
                # Check if we already have a position
                if self.portfolio.has_position(symbol):
                    result['message'] = f"Position already exists for {symbol}"
                    return result
                
                # Execute buy order
                success = self.portfolio.open_position(
                    symbol=symbol,
                    size=recommendation.position_size,
                    entry_price=price,
                    stop_loss=recommendation.stop_loss,
                    take_profit=recommendation.take_profit,
                    timestamp=timestamp
                )
                
                if success:
                    result['status'] = 'EXECUTED'
                    result['message'] = f"Bought {recommendation.position_size} shares of {symbol} at ${price:.2f}"
                    result['position_size'] = recommendation.position_size
                    result['stop_loss'] = recommendation.stop_loss
                    result['take_profit'] = recommendation.take_profit
                    
                    # Log the trade
                    self.logger.info(f"BUY: {recommendation.position_size} {symbol} @ ${price:.2f}, "
                                    f"Stop: ${recommendation.stop_loss:.2f}, Target: ${recommendation.take_profit:.2f}")
                else:
                    result['message'] = "Insufficient funds for trade"
            
            elif action == TradeAction.SELL:
                # Check if we have a position to sell
                if not self.portfolio.has_position(symbol):
                    result['message'] = f"No position exists for {symbol}"
                    return result
                
                # Get position details
                position = self.portfolio.get_position(symbol)
                
                # Execute sell order
                closed_trade = self.portfolio.close_position(
                    symbol=symbol,
                    exit_price=price,
                    timestamp=timestamp
                )
                
                if closed_trade:
                    result['status'] = 'EXECUTED'
                    result['message'] = f"Sold {closed_trade['size']} shares of {symbol} at ${price:.2f}"
                    result['position_size'] = closed_trade['size']
                    result['profit_loss'] = closed_trade['profit_loss']
                    result['profit_loss_pct'] = closed_trade['profit_loss_pct']
                    
                    # Log the trade
                    self.logger.info(f"SELL: {closed_trade['size']} {symbol} @ ${price:.2f}, "
                                    f"P/L: ${closed_trade['profit_loss']:.2f} ({closed_trade['profit_loss_pct']:.2f}%)")
                else:
                    result['message'] = f"Failed to close position for {symbol}"
            
            else:  # HOLD
                result['status'] = 'IGNORED'
                result['message'] = f"Hold signal for {symbol} - no action taken"
        
        except Exception as e:
            result['message'] = f"Error executing trade: {str(e)}"
            self.logger.error(f"Trade execution error: {str(e)}")
        
        # Add to executed orders
        self.executed_orders.append(result)
        
        return result
    
    def execute_recommendations(self, recommendations, current_prices=None):
        """
        Execute a list of risk-adjusted trade recommendations.
        
        Args:
            recommendations (list): List of RiskAdjustedRecommendation objects.
            current_prices (dict, optional): Dictionary of current prices by symbol. Defaults to None.
            
        Returns:
            list: List of execution results.
        """
        results = []
        
        for rec in recommendations:
            # Get current price if available
            current_price = None
            if current_prices and rec.original.symbol in current_prices:
                current_price = current_prices[rec.original.symbol]
            
            # Execute recommendation
            result = self.execute_recommendation(rec, current_price)
            results.append(result)
        
        return results
    
    def check_stop_losses(self, current_prices):
        """
        Check stop losses and take profits for all open positions.
        
        Args:
            current_prices (dict): Dictionary of current prices by symbol.
            
        Returns:
            list: List of execution results.
        """
        results = []
        
        # Apply trailing stops
        symbols_to_close = self.risk_manager.apply_trailing_stops(current_prices)
        
        # Close positions that hit stop loss or take profit
        for symbol in symbols_to_close:
            if symbol in current_prices:
                # Create a sell recommendation
                from strategy.strategy_manager import TradeRecommendation
                
                rec = TradeRecommendation(
                    symbol=symbol,
                    action=TradeAction.SELL,
                    position_type=PositionType.FLAT,
                    timestamp=datetime.datetime.now(),
                    price=current_prices[symbol],
                    confidence=1.0,
                    strategy_name="Stop Loss/Take Profit",
                    signal_source="Risk Management"
                )
                
                # Adjust recommendation
                adjusted_rec = self.risk_manager.adjust_recommendation(rec)
                
                # Execute recommendation
                result = self.execute_recommendation(adjusted_rec, current_prices[symbol])
                results.append(result)
                
                # Log the stop loss/take profit
                position = self.portfolio.get_position(symbol)
                if position:
                    self.logger.info(f"Stop/Target triggered for {symbol} @ ${current_prices[symbol]:.2f}")
        
        return results
    
    def apply_partial_profits(self, current_prices):
        """
        Apply partial profit taking for all open positions.
        
        Args:
            current_prices (dict): Dictionary of current prices by symbol.
            
        Returns:
            list: List of execution results.
        """
        results = []
        
        for symbol, position in list(self.portfolio.positions.items()):
            if symbol not in current_prices:
                continue
            
            # Check if we should take partial profits
            should_reduce, shares_to_sell = self.risk_manager.apply_profit_targets(symbol, current_prices[symbol])
            
            if should_reduce and shares_to_sell > 0:
                # Calculate what percentage of the position to sell
                sell_percentage = shares_to_sell / position['size']
                
                # Create a partial sell recommendation
                from strategy.strategy_manager import TradeRecommendation
                
                rec = TradeRecommendation(
                    symbol=symbol,
                    action=TradeAction.SELL,
                    position_type=PositionType.FLAT,
                    timestamp=datetime.datetime.now(),
                    price=current_prices[symbol],
                    confidence=1.0,
                    strategy_name="Partial Profit",
                    signal_source="Risk Management"
                )
                
                # Adjust recommendation
                adjusted_rec = self.risk_manager.adjust_recommendation(rec)
                
                # Execute recommendation
                result = self.execute_recommendation(adjusted_rec, current_prices[symbol])
                results.append(result)
                
                # Log the partial profit
                self.logger.info(f"Partial profit taken for {symbol}: {shares_to_sell} shares @ ${current_prices[symbol]:.2f}")
        
        return results
    
    def get_execution_summary(self):
        """
        Get a summary of all executed trades.
        
        Returns:
            dict: Execution summary.
        """
        if not self.executed_orders:
            return {
                'total_trades': 0,
                'successful_trades': 0,
                'failed_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0
            }
        
        successful_trades = sum(1 for order in self.executed_orders if order['status'] == 'EXECUTED')
        failed_trades = sum(1 for order in self.executed_orders if order['status'] == 'FAILED')
        buy_trades = sum(1 for order in self.executed_orders if order['action'] == 'BUY' and order['status'] == 'EXECUTED')
        sell_trades = sum(1 for order in self.executed_orders if order['action'] == 'SELL' and order['status'] == 'EXECUTED')
        
        return {
            'total_trades': len(self.executed_orders),
            'successful_trades': successful_trades,
            'failed_trades': failed_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades
        }


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    # Import required modules
    from data.data_manager import DataManager
    from analysis.technical_analysis import TechnicalAnalysis
    from strategy.strategy_manager import StrategyManager
    from risk_management.risk_manager import Portfolio, RiskManager
    
    # Initialize components
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    data_manager = DataManager(str(data_dir / "tradingbot.db"))
    ta = TechnicalAnalysis()
    strategy_manager = StrategyManager()
    portfolio = Portfolio(initial_capital=100000.0)
    risk_manager = RiskManager(portfolio)
    trade_executor = TradeExecutor(portfolio, risk_manager)
    
    # Get historical data for a symbol
    df = data_manager.get_historical_data("AAPL", timeframe="1d")
    
    if df is not None and not df.empty:
        # Add indicators
        df_with_indicators = ta.add_indicators(df)
        
        # Apply strategies
        result_df, recommendations = strategy_manager.apply_strategies(df_with_indicators, "AAPL")
        
        # Process recommendations
        adjusted_recommendations = risk_manager.process_recommendations(recommendations[-5:])  # Last 5 recommendations
        
        # Execute recommendations
        execution_results = trade_executor.execute_recommendations(adjusted_recommendations)
        
        # Print execution results
        print("Execution Results:")
        for result in execution_results:
            print(f"{result['action']} {result['symbol']} - {result['status']} - {result['message']}")
        
        # Print portfolio summary
        summary = portfolio.get_portfolio_summary()
        print(f"\nPortfolio Summary:")
        print(f"Equity: ${summary['current_equity']:.2f}")
        print(f"Cash: ${summary['cash']:.2f}")
        print(f"Open Positions: {summary['open_positions_count']}")
        print(f"Open Positions Value: ${summary['open_positions_value']:.2f}")
        
        # Print execution summary
        exec_summary = trade_executor.get_execution_summary()
        print(f"\nExecution Summary:")
        print(f"Total Trades: {exec_summary['total_trades']}")
        print(f"Successful Trades: {exec_summary['successful_trades']}")
        print(f"Failed Trades: {exec_summary['failed_trades']}")
        print(f"Buy Trades: {exec_summary['buy_trades']}")
        print(f"Sell Trades: {exec_summary['sell_trades']}")
    
    # Close connection
    data_manager.close()
