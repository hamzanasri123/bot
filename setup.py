#!/usr/bin/env python3
"""
Setup script for installing the trading bot package.
"""

from setuptools import setup, find_packages

setup(
    name="powertrader",
    version="1.0.0",
    description="Automated trading bot with advanced strategies and risk management",
    author="Manus AI",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "dash>=2.0.0",
        "plotly>=5.0.0",
        "requests>=2.25.0",
        "websocket-client>=1.2.0",
        "apscheduler>=3.8.0",
        "SQLAlchemy>=1.4.0",
        "python-dateutil>=2.8.0",
    ],
    extras_require={
        "talib": ["TA-Lib>=0.4.0"],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "powertrader=src.main:main",
            "powertrader-backtest=src.run_backtest:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
