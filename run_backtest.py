"""
Script to run backtests and generate performance reports.
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
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.backtest.backtester import Backtester
from src.config import DEFAULT_SYMBOLS, DEFAULT_TIMEFRAME, BACKTEST

def run_backtest_suite():
    """Run a suite of backtests with different configurations."""
    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize backtester
    backtester = Backtester()
    
    # Define test configurations
    test_configs = [
        {
            "name": "default_strategy",
            "symbols": ["AAPL", "MSFT", "AMZN", "GOOGL", "META"],
            "timeframe": "1d",
            "start_date": "2022-01-01",
            "end_date": "2022-12-31",
            "initial_capital": 100000
        },
        {
            "name": "trend_following_only",
            "symbols": ["AAPL", "MSFT", "AMZN", "GOOGL", "META"],
            "timeframe": "1d",
            "start_date": "2022-01-01",
            "end_date": "2022-12-31",
            "initial_capital": 100000,
            "strategy_config": {"mean_reversion": {"enabled": False}}
        },
        {
            "name": "mean_reversion_only",
            "symbols": ["AAPL", "MSFT", "AMZN", "GOOGL", "META"],
            "timeframe": "1d",
            "start_date": "2022-01-01",
            "end_date": "2022-12-31",
            "initial_capital": 100000,
            "strategy_config": {"trend_following": {"enabled": False}}
        },
        {
            "name": "tech_sector",
            "symbols": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"],
            "timeframe": "1d",
            "start_date": "2022-01-01",
            "end_date": "2022-12-31",
            "initial_capital": 100000
        },
        {
            "name": "etf_test",
            "symbols": ["SPY", "QQQ", "IWM", "DIA", "XLK"],
            "timeframe": "1d",
            "start_date": "2022-01-01",
            "end_date": "2022-12-31",
            "initial_capital": 100000
        }
    ]
    
    # Run each test configuration
    results = []
    
    for config in test_configs:
        print(f"\nRunning backtest: {config['name']}")
        
        # Run backtest
        result = backtester.run_backtest(
            symbols=config["symbols"],
            timeframe=config["timeframe"],
            start_date=config["start_date"],
            end_date=config["end_date"],
            initial_capital=config["initial_capital"]
        )
        
        # Print results
        result.print_summary()
        
        # Plot results
        config_dir = output_dir / config["name"]
        config_dir.mkdir(exist_ok=True)
        
        result.plot_equity_curve(save_path=config_dir / "equity_curve.png")
        result.plot_drawdown(save_path=config_dir / "drawdown.png")
        result.plot_monthly_returns(save_path=config_dir / "monthly_returns.png")
        
        # Save results
        result.save_results(config_dir / "backtest_results.json")
        
        # Store summary metrics
        results.append({
            "name": config["name"],
            "total_return_pct": result.metrics["total_return_pct"],
            "annualized_return": result.metrics["annualized_return"],
            "sharpe_ratio": result.metrics["sharpe_ratio"],
            "max_drawdown": result.metrics["max_drawdown"],
            "win_rate": result.metrics["win_rate"],
            "profit_factor": result.metrics["profit_factor"],
            "total_trades": result.metrics["total_trades"]
        })
    
    # Create comparison report
    create_comparison_report(results, output_dir)
    
    return results

def create_comparison_report(results, output_dir):
    """Create a comparison report of backtest results."""
    if not results:
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    df.set_index("name", inplace=True)
    
    # Save comparison to CSV
    df.to_csv(output_dir / "backtest_comparison.csv")
    
    # Create comparison charts
    plt.figure(figsize=(12, 8))
    
    # Total return comparison
    plt.subplot(2, 2, 1)
    df["total_return_pct"].plot(kind="bar", color="skyblue")
    plt.title("Total Return (%)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    
    # Sharpe ratio comparison
    plt.subplot(2, 2, 2)
    df["sharpe_ratio"].plot(kind="bar", color="green")
    plt.title("Sharpe Ratio")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    
    # Max drawdown comparison
    plt.subplot(2, 2, 3)
    df["max_drawdown"].plot(kind="bar", color="red")
    plt.title("Maximum Drawdown (%)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    
    # Win rate comparison
    plt.subplot(2, 2, 4)
    df["win_rate"].plot(kind="bar", color="orange")
    plt.title("Win Rate (%)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "backtest_comparison.png", dpi=300, bbox_inches="tight")
    
    # Create HTML report
    html_report = f"""
    <html>
    <head>
        <title>Trading Bot Backtest Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333366; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Trading Bot Backtest Comparison</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Performance Metrics</h2>
        <table>
            <tr>
                <th>Strategy</th>
                <th>Total Return (%)</th>
                <th>Annualized Return (%)</th>
                <th>Sharpe Ratio</th>
                <th>Max Drawdown (%)</th>
                <th>Win Rate (%)</th>
                <th>Profit Factor</th>
                <th>Total Trades</th>
            </tr>
    """
    
    for idx, row in df.iterrows():
        html_report += f"""
            <tr>
                <td>{idx}</td>
                <td class="{'positive' if row['total_return_pct'] >= 0 else 'negative'}">{row['total_return_pct']:.2f}</td>
                <td class="{'positive' if row['annualized_return'] >= 0 else 'negative'}">{row['annualized_return']:.2f}</td>
                <td class="{'positive' if row['sharpe_ratio'] >= 0 else 'negative'}">{row['sharpe_ratio']:.2f}</td>
                <td class="negative">{row['max_drawdown']:.2f}</td>
                <td>{row['win_rate']:.2f}</td>
                <td>{row['profit_factor']:.2f}</td>
                <td>{row['total_trades']}</td>
            </tr>
        """
    
    html_report += """
        </table>
        
        <h2>Comparison Charts</h2>
        <img src="backtest_comparison.png" alt="Backtest Comparison">
        
        <h2>Individual Strategy Results</h2>
    """
    
    for idx in df.index:
        html_report += f"""
        <h3>{idx}</h3>
        <p>
            <a href="{idx}/backtest_results.json">Detailed Results (JSON)</a><br>
            <img src="{idx}/equity_curve.png" alt="Equity Curve" style="max-width: 600px;"><br>
            <img src="{idx}/drawdown.png" alt="Drawdown" style="max-width: 600px;"><br>
            <img src="{idx}/monthly_returns.png" alt="Monthly Returns" style="max-width: 600px;">
        </p>
        """
    
    html_report += """
    </body>
    </html>
    """
    
    with open(output_dir / "backtest_report.html", "w") as f:
        f.write(html_report)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run backtest suite')
    parser.add_argument('--run_all', action='store_true', help='Run all backtest configurations')
    
    args = parser.parse_args()
    
    if args.run_all:
        run_backtest_suite()
    else:
        # Run a single default backtest
        backtester = Backtester()
        result = backtester.run_backtest()
        result.print_summary()
        
        # Create output directory
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        
        # Plot and save results
        result.plot_equity_curve(save_path=output_dir / "equity_curve.png")
        result.plot_drawdown(save_path=output_dir / "drawdown.png")
        result.plot_monthly_returns(save_path=output_dir / "monthly_returns.png")
        result.save_results(output_dir / "backtest_results.json")
