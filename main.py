"""
Main application module for running the trading bot.
"""

import sys
import os
import pandas as pd
import numpy as np
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.data.data_manager import DataManager
from src.analysis.technical_analysis import TechnicalAnalysis
from src.strategy.strategy_manager import StrategyManager
from src.risk_management.risk_manager import Portfolio, RiskManager
from src.execution.trade_executor import TradeExecutor
from src.ui.dashboard import TradingDashboard
from src.config import DEFAULT_SYMBOLS, DEFAULT_TIMEFRAME, BOT_NAME, VERSION

def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / "trading_bot.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('trading_bot')

def initialize_components():
    """Initialize all trading bot components."""
    logger = setup_logging()
    logger.info(f"Initializing {BOT_NAME} v{VERSION}")
    
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Initialize components
    logger.info("Initializing DataManager")
    data_manager = DataManager(str(data_dir / "tradingbot.db"))
    
    logger.info("Initializing TechnicalAnalysis")
    ta = TechnicalAnalysis()
    
    logger.info("Initializing StrategyManager")
    strategy_manager = StrategyManager()
    
    logger.info("Initializing Portfolio")
    portfolio = Portfolio(initial_capital=100000.0)
    
    logger.info("Initializing RiskManager")
    risk_manager = RiskManager(portfolio)
    
    logger.info("Initializing TradeExecutor")
    trade_executor = TradeExecutor(portfolio, risk_manager)
    
    return {
        'logger': logger,
        'data_manager': data_manager,
        'ta': ta,
        'strategy_manager': strategy_manager,
        'portfolio': portfolio,
        'risk_manager': risk_manager,
        'trade_executor': trade_executor
    }

def update_market_data(components, symbols=None, timeframe=DEFAULT_TIMEFRAME):
    """Update market data for all symbols."""
    logger = components['logger']
    data_manager = components['data_manager']
    
    symbols = symbols or DEFAULT_SYMBOLS
    
    logger.info(f"Updating market data for {len(symbols)} symbols")
    
    # Add symbols to database
    data_manager.add_symbols(symbols)
    
    # Update data for all symbols
    results = data_manager.update_all_symbols(symbols, timeframe)
    
    success_count = sum(1 for result in results.values() if result)
    logger.info(f"Updated data for {success_count}/{len(symbols)} symbols")
    
    return success_count

def run_trading_cycle(components, symbols=None, timeframe=DEFAULT_TIMEFRAME):
    """Run a complete trading cycle."""
    logger = components['logger']
    data_manager = components['data_manager']
    ta = components['ta']
    strategy_manager = components['strategy_manager']
    risk_manager = components['risk_manager']
    trade_executor = components['trade_executor']
    
    symbols = symbols or DEFAULT_SYMBOLS
    
    logger.info(f"Starting trading cycle for {len(symbols)} symbols")
    
    # Dictionary to store current prices
    current_prices = {}
    all_recommendations = []
    
    # Process each symbol
    for symbol in symbols:
        logger.info(f"Processing {symbol}")
        
        try:
            # Get historical data
            df = data_manager.get_historical_data(symbol, timeframe)
            
            if df is None or df.empty:
                logger.warning(f"No data available for {symbol}")
                continue
            
            # Get current price
            current_price = df['close'].iloc[-1]
            current_prices[symbol] = current_price
            
            # Add technical indicators
            df_with_indicators = ta.add_indicators(df)
            
            # Apply strategies
            result_df, recommendations = strategy_manager.apply_strategies(df_with_indicators, symbol)
            
            # Store recommendations
            if recommendations:
                all_recommendations.extend(recommendations)
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    # Check stop losses and take profits
    logger.info("Checking stop losses and take profits")
    stop_loss_results = trade_executor.check_stop_losses(current_prices)
    
    # Apply partial profits
    logger.info("Applying partial profits")
    partial_profit_results = trade_executor.apply_partial_profits(current_prices)
    
    # Process new recommendations
    if all_recommendations:
        logger.info(f"Processing {len(all_recommendations)} new recommendations")
        
        # Get only the latest recommendation for each symbol
        symbol_to_rec = {}
        for rec in all_recommendations:
            if rec.symbol not in symbol_to_rec or rec.timestamp > symbol_to_rec[rec.symbol].timestamp:
                symbol_to_rec[rec.symbol] = rec
        
        latest_recommendations = list(symbol_to_rec.values())
        
        # Process recommendations through risk manager
        adjusted_recommendations = risk_manager.process_recommendations(latest_recommendations)
        
        if adjusted_recommendations:
            logger.info(f"Executing {len(adjusted_recommendations)} risk-adjusted recommendations")
            execution_results = trade_executor.execute_recommendations(adjusted_recommendations, current_prices)
            
            # Log execution results
            for result in execution_results:
                if result['status'] == 'EXECUTED':
                    logger.info(f"Executed: {result['message']}")
                elif result['status'] == 'FAILED':
                    logger.warning(f"Failed: {result['message']}")
    
    # Update portfolio metrics
    risk_manager.update_portfolio_metrics()
    
    # Get portfolio summary
    portfolio_summary = components['portfolio'].get_portfolio_summary()
    logger.info(f"Portfolio equity: ${portfolio_summary['current_equity']:.2f}")
    logger.info(f"Cash: ${portfolio_summary['cash']:.2f}")
    logger.info(f"Open positions: {portfolio_summary['open_positions_count']}")
    
    return portfolio_summary

def run_dashboard(components, port=8050):
    """Run the trading dashboard."""
    logger = components['logger']
    
    logger.info(f"Starting dashboard on port {port}")
    
    # Initialize dashboard
    dashboard = TradingDashboard(
        components['data_manager'],
        components['portfolio'],
        components['risk_manager'],
        components['trade_executor']
    )
    
    # Run dashboard server
    dashboard.run_server(debug=True, port=port)

def main():
    """Main entry point for the trading bot."""
    parser = argparse.ArgumentParser(description=f'{BOT_NAME} v{VERSION} - Automated Trading Bot')
    parser.add_argument('--mode', choices=['backtest', 'paper', 'live', 'dashboard'], default='dashboard',
                        help='Trading mode (default: dashboard)')
    parser.add_argument('--symbols', nargs='+', help='Symbols to trade')
    parser.add_argument('--timeframe', default=DEFAULT_TIMEFRAME, help=f'Trading timeframe (default: {DEFAULT_TIMEFRAME})')
    parser.add_argument('--port', type=int, default=8050, help='Dashboard port (default: 8050)')
    
    args = parser.parse_args()
    
    # Initialize components
    components = initialize_components()
    logger = components['logger']
    
    # Use provided symbols or default
    symbols = args.symbols or DEFAULT_SYMBOLS
    
    try:
        # Update market data
        update_market_data(components, symbols, args.timeframe)
        
        if args.mode == 'dashboard':
            # Run dashboard
            run_dashboard(components, args.port)
        
        elif args.mode == 'backtest':
            logger.info("Backtest mode not fully implemented yet")
            # TODO: Implement backtesting
        
        elif args.mode == 'paper':
            logger.info("Starting paper trading mode")
            
            # Run continuous trading cycle
            while True:
                run_trading_cycle(components, symbols, args.timeframe)
                
                # Wait for next cycle (5 minutes)
                logger.info("Waiting for next trading cycle...")
                time.sleep(300)
        
        elif args.mode == 'live':
            logger.info("Live trading mode not fully implemented yet")
            # TODO: Implement live trading
    
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    
    except Exception as e:
        logger.error(f"Error running trading bot: {e}", exc_info=True)
    
    finally:
        # Close database connection
        components['data_manager'].close()
        logger.info("Trading bot shutdown complete")

if __name__ == "__main__":
    main()
