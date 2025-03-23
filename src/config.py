"""
Main configuration file for the trading bot.
Contains all configurable parameters and settings.
"""

# General Configuration
BOT_NAME = "PowerTrader"
VERSION = "1.0.0"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Market Configuration
MARKET = "US_EQUITIES"  # US_EQUITIES, FOREX, CRYPTO
TRADING_HOURS = {
    "start": "09:30",  # Eastern Time
    "end": "16:00"     # Eastern Time
}
TRADING_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Data Configuration
DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
    "TSLA", "NVDA", "JPM", "V", "JNJ",
    "SPY", "QQQ", "IWM", "DIA", "XLK"
]
TIMEFRAMES = {
    "1m": 60,        # 1 minute in seconds
    "5m": 300,       # 5 minutes in seconds
    "15m": 900,      # 15 minutes in seconds
    "1h": 3600,      # 1 hour in seconds
    "1d": 86400      # 1 day in seconds
}
DEFAULT_TIMEFRAME = "1h"
HISTORICAL_DATA_DAYS = 365  # Days of historical data to fetch

# Database Configuration
DATABASE = {
    "type": "sqlite",
    "path": "data/tradingbot.db"
}

# Technical Indicators Configuration
INDICATORS = {
    "SMA": {
        "short_period": 20,
        "medium_period": 50,
        "long_period": 200
    },
    "EMA": {
        "short_period": 12,
        "medium_period": 26,
        "long_period": 50
    },
    "RSI": {
        "period": 14,
        "overbought": 70,
        "oversold": 30
    },
    "MACD": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
    },
    "BOLLINGER": {
        "period": 20,
        "std_dev": 2.0
    },
    "ADX": {
        "period": 14,
        "threshold": 25
    }
}

# Strategy Configuration
STRATEGIES = {
    "trend_following": {
        "enabled": True,
        "timeframes": ["1h", "1d"],
        "indicators": ["SMA", "MACD", "ADX"],
        "entry_conditions": {
            "sma_alignment": True,  # Short SMA > Medium SMA > Long SMA
            "macd_signal": True,    # MACD line crosses above Signal line
            "adx_filter": True,     # ADX > threshold (trending market)
        },
        "exit_conditions": {
            "sma_reversal": True,   # Short SMA crosses below Medium SMA
            "macd_reversal": True,  # MACD line crosses below Signal line
            "trailing_stop": True   # Price falls below trailing stop
        }
    },
    "mean_reversion": {
        "enabled": True,
        "timeframes": ["1h"],
        "indicators": ["RSI", "BOLLINGER"],
        "entry_conditions": {
            "rsi_oversold": True,   # RSI < oversold threshold
            "bollinger_lower": True # Price touches lower Bollinger Band
        },
        "exit_conditions": {
            "rsi_overbought": True, # RSI > overbought threshold
            "bollinger_middle": True, # Price reaches middle Bollinger Band
            "max_holding_time": 5   # Maximum holding time in days
        }
    }
}

# Risk Management Configuration
RISK_MANAGEMENT = {
    "max_portfolio_risk": 0.02,     # Maximum portfolio risk per day (2%)
    "max_position_size": 0.02,      # Maximum position size as % of portfolio (2%)
    "max_positions": 10,            # Maximum number of open positions
    "stop_loss_pct": 0.02,          # Default stop loss percentage (2%)
    "trailing_stop_activation": 0.01, # Activate trailing stop after 1% profit
    "trailing_stop_distance": 0.015,  # Trailing stop 1.5% behind price
    "profit_targets": [
        {"target": 0.01, "size": 0.3},  # Take 30% profit at 1% gain
        {"target": 0.02, "size": 0.3},  # Take 30% profit at 2% gain
        {"target": 0.03, "size": 0.4}   # Take 40% profit at 3% gain
    ],
    "drawdown_control": {
        "daily_loss_limit": 0.05,   # Stop trading after 5% daily loss
        "max_drawdown": 0.15,       # Reduce position size after 15% drawdown
        "position_size_reduction": 0.5  # Reduce to 50% of normal size
    }
}

# Backtesting Configuration
BACKTEST = {
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 100000,
    "commission": 0.001,            # 0.1% commission per trade
    "slippage": 0.001,              # 0.1% slippage per trade
    "benchmark": "SPY"              # Benchmark for performance comparison
}

# Logging Configuration
LOGGING = {
    "log_file": "logs/tradingbot.log",
    "max_log_size": 10485760,       # 10 MB
    "backup_count": 5,              # Keep 5 backup logs
    "log_trades": True,
    "log_signals": True,
    "log_portfolio": True
}
