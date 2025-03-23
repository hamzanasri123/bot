"""
Backtesting module for testing trading strategies with historical data.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.data.data_manager import DataManager
from src.analysis.technical_analysis import TechnicalAnalysis, SignalType
from src.strategy.strategy_manager import StrategyManager, TradeAction, PositionType
from src.risk_management.risk_manager import Portfolio, RiskManager
from src.config import BACKTEST, DEFAULT_SYMBOLS, DEFAULT_TIMEFRAME

class BacktestResult:
    """Class to store and analyze backtest results."""
    
    def __init__(self, initial_capital, trades, equity_curve, benchmark_data=None):
        """
        Initialize backtest results.
        
        Args:
            initial_capital (float): Initial capital.
            trades (list): List of executed trades.
            equity_curve (pd.DataFrame): DataFrame with equity curve.
            benchmark_data (pd.DataFrame, optional): DataFrame with benchmark data. Defaults to None.
        """
        self.initial_capital = initial_capital
        self.trades = trades
        self.equity_curve = equity_curve
        self.benchmark_data = benchmark_data
        self.metrics = self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate performance metrics."""
        if not self.trades:
            return {
                'total_return': 0,
                'total_return_pct': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_profit_per_trade': 0,
                'avg_profit_per_winning_trade': 0,
                'avg_loss_per_losing_trade': 0,
                'risk_reward_ratio': 0
            }
        
        # Calculate basic metrics
        final_equity = self.equity_curve['equity'].iloc[-1]
        total_return = final_equity - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Calculate trading days
        start_date = self.equity_curve.index[0]
        end_date = self.equity_curve.index[-1]
        trading_days = len(self.equity_curve)
        years = trading_days / 252  # Approximate trading days in a year
        
        # Calculate annualized return
        annualized_return = ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        daily_returns = self.equity_curve['equity'].pct_change().dropna()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 0 and daily_returns.std() > 0 else 0
        
        # Calculate maximum drawdown
        equity_series = self.equity_curve['equity']
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min()) * 100
        
        # Calculate trade metrics
        winning_trades = [t for t in self.trades if t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('profit_loss', 0) < 0]
        
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t.get('profit_loss', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('profit_loss', 0) for t in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_profit_per_trade = sum(t.get('profit_loss', 0) for t in self.trades) / total_trades if total_trades > 0 else 0
        avg_profit_per_winning_trade = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss_per_losing_trade = total_loss / len(losing_trades) if losing_trades else 0
        
        risk_reward_ratio = abs(avg_profit_per_winning_trade / avg_loss_per_losing_trade) if avg_loss_per_losing_trade != 0 else float('inf')
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_profit_per_trade': avg_profit_per_trade,
            'avg_profit_per_winning_trade': avg_profit_per_winning_trade,
            'avg_loss_per_losing_trade': avg_loss_per_losing_trade,
            'risk_reward_ratio': risk_reward_ratio
        }
    
    def print_summary(self):
        """Print backtest summary."""
        print("\n" + "="*50)
        print("BACKTEST RESULTS SUMMARY")
        print("="*50)
        
        print(f"\nInitial Capital: ${self.initial_capital:.2f}")
        print(f"Final Equity: ${self.equity_curve['equity'].iloc[-1]:.2f}")
        print(f"Total Return: ${self.metrics['total_return']:.2f} ({self.metrics['total_return_pct']:.2f}%)")
        print(f"Annualized Return: {self.metrics['annualized_return']:.2f}%")
        print(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {self.metrics['max_drawdown']:.2f}%")
        
        print("\nTRADE STATISTICS:")
        print(f"Total Trades: {self.metrics['total_trades']}")
        print(f"Winning Trades: {self.metrics['winning_trades']} ({self.metrics['win_rate']:.2f}%)")
        print(f"Losing Trades: {self.metrics['losing_trades']}")
        print(f"Profit Factor: {self.metrics['profit_factor']:.2f}")
        print(f"Average Profit per Trade: ${self.metrics['avg_profit_per_trade']:.2f}")
        print(f"Average Profit per Winning Trade: ${self.metrics['avg_profit_per_winning_trade']:.2f}")
        print(f"Average Loss per Losing Trade: ${self.metrics['avg_loss_per_losing_trade']:.2f}")
        print(f"Risk-Reward Ratio: {self.metrics['risk_reward_ratio']:.2f}")
        
        print("="*50)
    
    def plot_equity_curve(self, show_benchmark=True, save_path=None):
        """
        Plot equity curve.
        
        Args:
            show_benchmark (bool, optional): Whether to show benchmark. Defaults to True.
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        plt.figure(figsize=(12, 6))
        
        # Plot equity curve
        plt.plot(self.equity_curve.index, self.equity_curve['equity'], label='Strategy', linewidth=2)
        
        # Plot benchmark if available
        if show_benchmark and self.benchmark_data is not None:
            # Normalize benchmark to same starting capital
            benchmark_normalized = self.benchmark_data / self.benchmark_data.iloc[0] * self.initial_capital
            plt.plot(self.benchmark_data.index, benchmark_normalized, label='Benchmark', linewidth=2, alpha=0.7)
        
        # Add labels and title
        plt.title('Equity Curve', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Equity ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_drawdown(self, save_path=None):
        """
        Plot drawdown.
        
        Args:
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        plt.figure(figsize=(12, 6))
        
        # Calculate drawdown
        equity_series = self.equity_curve['equity']
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        
        # Plot drawdown
        plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        plt.plot(drawdown.index, drawdown, color='red', linewidth=1)
        
        # Add labels and title
        plt.title('Drawdown (%)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_monthly_returns(self, save_path=None):
        """
        Plot monthly returns heatmap.
        
        Args:
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        # Calculate daily returns
        daily_returns = self.equity_curve['equity'].pct_change().dropna()
        
        # Convert to monthly returns
        monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create a pivot table for the heatmap
        monthly_returns.index = monthly_returns.index.to_period('M')
        monthly_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first()
        monthly_pivot = monthly_pivot.unstack() * 100  # Convert to percentage
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(monthly_pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0, linewidths=1, cbar_kws={'label': 'Monthly Return (%)'})
        
        # Add labels and title
        plt.title('Monthly Returns (%)', fontsize=16)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Year', fontsize=12)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def save_results(self, file_path):
        """
        Save backtest results to file.
        
        Args:
            file_path (str): Path to save results.
        """
        # Create results dictionary
        results = {
            'metrics': self.metrics,
            'trades': self.trades,
            'equity_curve': self.equity_curve.to_dict(),
            'initial_capital': self.initial_capital
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)

class Backtester:
    """Backtester for testing trading strategies with historical data."""
    
    def __init__(self, config=None):
        """
        Initialize the backtester.
        
        Args:
            config (dict, optional): Backtesting configuration. Defaults to None.
        """
        self.config = config or BACKTEST
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Set up logging for backtester."""
        logger = logging.getLogger('backtester')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        logs_dir = Path(__file__).parent.parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Create file handler
        fh = logging.FileHandler(logs_dir / "backtest.log")
        fh.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(fh)
        
        return logger
    
    def run_backtest(self, symbols=None, timeframe=DEFAULT_TIMEFRAME, start_date=None, end_date=None, initial_capital=None):
        """
        Run backtest for specified symbols and timeframe.
        
        Args:
            symbols (list, optional): List of symbols to backtest. Defaults to None.
            timeframe (str, optional): Timeframe to backtest. Defaults to DEFAULT_TIMEFRAME.
            start_date (str, optional): Start date for backtest. Defaults to None.
            end_date (str, optional): End date for backtest. Defaults to None.
            initial_capital (float, optional): Initial capital. Defaults to None.
            
        Returns:
            BacktestResult: Backtest results.
        """
        # Use configuration or provided parameters
        symbols = symbols or DEFAULT_SYMBOLS
        start_date = start_date or self.config['start_date']
        end_date = end_date or self.config['end_date']
        initial_capital = initial_capital or self.config['initial_capital']
        
        self.logger.info(f"Starting backtest for {len(symbols)} symbols from {start_date} to {end_date}")
        self.logger.info(f"Initial capital: ${initial_capital}")
        
        # Initialize components
        data_manager = DataManager()
        ta = TechnicalAnalysis()
        strategy_manager = StrategyManager()
        portfolio = Portfolio(initial_capital=initial_capital)
        risk_manager = RiskManager(portfolio)
        
        # Add symbols to database
        data_manager.add_symbols(symbols)
        
        # Initialize variables for tracking
        all_trades = []
        equity_curve = []
        benchmark_data = None
        
        # Get benchmark data if specified
        benchmark_symbol = self.config['benchmark']
        if benchmark_symbol in symbols:
            benchmark_df = data_manager.get_historical_data(
                benchmark_symbol, 
                timeframe=timeframe, 
                start_date=start_date, 
                end_date=end_date
            )
            if benchmark_df is not None and not benchmark_df.empty:
                benchmark_data = benchmark_df.set_index('timestamp')['close']
        
        # Process each symbol
        for symbol in symbols:
            self.logger.info(f"Processing {symbol}")
            
            try:
                # Get historical data
                df = data_manager.get_historical_data(
                    symbol, 
                    timeframe=timeframe, 
                    start_date=start_date, 
                    end_date=end_date
                )
                
                if df is None or df.empty:
                    self.logger.warning(f"No data available for {symbol}")
                    continue
                
                # Add technical indicators
                df_with_indicators = ta.add_indicators(df)
                
                # Apply strategies
                result_df, _ = strategy_manager.apply_strategies(df_with_indicators, symbol)
                
                # Store results for backtesting
                if 'combined_signal' in result_df.columns:
                    result_df['symbol'] = symbol
                    
                    # Process each day sequentially
                    for i, row in result_df.iterrows():
                        # Skip if not enough data
                        if i < 20:  # Skip first few rows to allow indicators to stabilize
                            continue
                        
                        # Get current date and price
                        current_date = row['timestamp']
                        current_price = row['close']
                        
                        # Check for buy signal
                        if row['combined_signal'] == SignalType.BUY.value and not portfolio.has_position(symbol):
                            # Calculate position size
                            risk_per_share = current_price * 0.02  # 2% stop loss
                            max_risk = portfolio.equity * 0.01  # Risk 1% of portfolio
                            shares = int(max_risk / risk_per_share)
                
            except:
                pass  
                        