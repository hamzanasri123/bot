"""
Dashboard module for monitoring trading bot performance and status.
"""

import sys
import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BOT_NAME, VERSION

class TradingDashboard:
    """Dashboard for monitoring trading bot performance and status."""
    
    def __init__(self, data_manager, portfolio, risk_manager, trade_executor):
        """
        Initialize the dashboard.
        
        Args:
            data_manager: The data manager instance.
            portfolio: The portfolio instance.
            risk_manager: The risk manager instance.
            trade_executor: The trade executor instance.
        """
        self.data_manager = data_manager
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.trade_executor = trade_executor
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1(f"{BOT_NAME} v{VERSION} - Trading Dashboard"),
                html.Div([
                    html.Button('Refresh Data', id='refresh-button', n_clicks=0),
                    html.Div(id='last-update-time', style={'margin-left': '20px'})
                ], style={'display': 'flex', 'align-items': 'center'}),
            ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 
                      'padding': '10px', 'background-color': '#f8f9fa', 'border-bottom': '1px solid #ddd'}),
            
            # Main content
            html.Div([
                # Left column - Portfolio and Performance
                html.Div([
                    # Portfolio Summary Card
                    html.Div([
                        html.H3("Portfolio Summary"),
                        html.Div(id='portfolio-summary-content')
                    ], className='card'),
                    
                    # Equity Chart Card
                    html.Div([
                        html.H3("Equity Curve"),
                        dcc.Graph(id='equity-chart')
                    ], className='card'),
                    
                    # Performance Metrics Card
                    html.Div([
                        html.H3("Performance Metrics"),
                        html.Div(id='performance-metrics-content')
                    ], className='card')
                ], style={'width': '60%', 'padding': '10px'}),
                
                # Right column - Positions and Trades
                html.Div([
                    # Open Positions Card
                    html.Div([
                        html.H3("Open Positions"),
                        html.Div(id='open-positions-content')
                    ], className='card'),
                    
                    # Recent Trades Card
                    html.Div([
                        html.H3("Recent Trades"),
                        html.Div(id='recent-trades-content')
                    ], className='card'),
                    
                    # Trading Signals Card
                    html.Div([
                        html.H3("Latest Trading Signals"),
                        dcc.Dropdown(
                            id='symbol-dropdown',
                            options=[],
                            value=None,
                            placeholder="Select a symbol"
                        ),
                        html.Div(id='trading-signals-content')
                    ], className='card')
                ], style={'width': '40%', 'padding': '10px'})
            ], style={'display': 'flex', 'flex-wrap': 'wrap'}),
            
            # Hidden div for storing data
            html.Div(id='portfolio-data', style={'display': 'none'}),
            html.Div(id='trades-data', style={'display': 'none'}),
            html.Div(id='symbols-data', style={'display': 'none'}),
            
            # CSS
            html.Style('''
                .card {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 20px;
                    background-color: white;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }
                .metric-box {
                    border: 1px solid #eee;
                    border-radius: 5px;
                    padding: 10px;
                    margin: 5px;
                    text-align: center;
                    background-color: #f8f9fa;
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                }
                .metric-label {
                    font-size: 14px;
                    color: #666;
                }
                .positive {
                    color: green;
                }
                .negative {
                    color: red;
                }
            ''')
        ])
    
    def setup_callbacks(self):
        """Set up the dashboard callbacks."""
        
        @self.app.callback(
            [Output('portfolio-data', 'children'),
             Output('trades-data', 'children'),
             Output('symbols-data', 'children'),
             Output('last-update-time', 'children')],
            [Input('refresh-button', 'n_clicks')]
        )
        def update_data(n_clicks):
            """Update all data sources."""
            # Get portfolio summary
            portfolio_summary = self.portfolio.get_portfolio_summary()
            
            # Get trade history
            trade_history = self.trade_executor.executed_orders
            
            # Get available symbols
            symbols = self.get_available_symbols()
            
            # Update time
            update_time = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return (json.dumps(portfolio_summary), 
                    json.dumps(trade_history), 
                    json.dumps(symbols),
                    update_time)
        
        @self.app.callback(
            Output('portfolio-summary-content', 'children'),
            [Input('portfolio-data', 'children')]
        )
        def update_portfolio_summary(portfolio_data):
            """Update portfolio summary card."""
            if not portfolio_data:
                return html.Div("No data available")
            
            portfolio = json.loads(portfolio_data)
            
            return html.Div([
                html.Div([
                    html.Div([
                        html.Div(f"${portfolio['current_equity']:.2f}", className='metric-value'),
                        html.Div("Current Equity", className='metric-label')
                    ], className='metric-box'),
                    html.Div([
                        html.Div(f"${portfolio['cash']:.2f}", className='metric-value'),
                        html.Div("Available Cash", className='metric-label')
                    ], className='metric-box'),
                    html.Div([
                        html.Div(f"${portfolio['open_positions_value']:.2f}", className='metric-value'),
                        html.Div("Positions Value", className='metric-label')
                    ], className='metric-box')
                ], style={'display': 'flex', 'justify-content': 'space-between'}),
                html.Div([
                    html.Div([
                        html.Div([
                            html.Span(f"{portfolio['return_pct']:.2f}%", 
                                     className='positive' if portfolio['return_pct'] >= 0 else 'negative'),
                        ], className='metric-value'),
                        html.Div("Total Return", className='metric-label')
                    ], className='metric-box'),
                    html.Div([
                        html.Div(f"${portfolio['total_profit_loss']:.2f}", 
                                className='metric-value positive' if portfolio['total_profit_loss'] >= 0 else 'metric-value negative'),
                        html.Div("Total P/L", className='metric-label')
                    ], className='metric-box'),
                    html.Div([
                        html.Div(f"{portfolio['open_positions_count']}", className='metric-value'),
                        html.Div("Open Positions", className='metric-label')
                    ], className='metric-box')
                ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-top': '10px'})
            ])
        
        @self.app.callback(
            Output('equity-chart', 'figure'),
            [Input('portfolio-data', 'children')]
        )
        def update_equity_chart(portfolio_data):
            """Update equity chart."""
            # For now, create a dummy equity curve
            # In a real implementation, this would use historical equity data
            
            dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
            
            if portfolio_data:
                portfolio = json.loads(portfolio_data)
                current_equity = portfolio['current_equity']
                initial_equity = portfolio['initial_capital']
                
                # Generate a somewhat realistic equity curve
                np.random.seed(42)  # For reproducibility
                daily_returns = np.random.normal(0.001, 0.01, len(dates))
                equity_values = [initial_equity]
                
                for ret in daily_returns:
                    equity_values.append(equity_values[-1] * (1 + ret))
                
                # Make sure the last value matches current equity
                equity_values = equity_values[1:]  # Remove the initial value
                scaling_factor = current_equity / equity_values[-1]
                equity_values = [val * scaling_factor for val in equity_values]
            else:
                equity_values = [100000 for _ in range(len(dates))]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=equity_values,
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title='Portfolio Equity Over Time',
                xaxis_title='Date',
                yaxis_title='Equity ($)',
                template='plotly_white',
                height=400,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            return fig
        
        @self.app.callback(
            Output('performance-metrics-content', 'children'),
            [Input('portfolio-data', 'children'),
             Input('trades-data', 'children')]
        )
        def update_performance_metrics(portfolio_data, trades_data):
            """Update performance metrics card."""
            if not portfolio_data or not trades_data:
                return html.Div("No data available")
            
            portfolio = json.loads(portfolio_data)
            trades = json.loads(trades_data)
            
            # Calculate metrics
            executed_trades = [t for t in trades if t['status'] == 'EXECUTED']
            win_trades = [t for t in executed_trades if t.get('profit_loss', 0) > 0]
            loss_trades = [t for t in executed_trades if t.get('profit_loss', 0) < 0]
            
            total_trades = len(executed_trades)
            win_rate = len(win_trades) / total_trades if total_trades > 0 else 0
            
            # Calculate Sharpe ratio (simplified)
            # In a real implementation, this would use actual daily returns
            sharpe_ratio = 1.2  # Placeholder
            
            # Calculate max drawdown (simplified)
            # In a real implementation, this would use historical equity data
            max_drawdown = self.risk_manager.max_drawdown * 100
            
            return html.Div([
                html.Div([
                    html.Div([
                        html.Div(f"{total_trades}", className='metric-value'),
                        html.Div("Total Trades", className='metric-label')
                    ], className='metric-box'),
                    html.Div([
                        html.Div(f"{win_rate:.2%}", className='metric-value'),
                        html.Div("Win Rate", className='metric-label')
                    ], className='metric-box'),
                    html.Div([
                        html.Div(f"{sharpe_ratio:.2f}", className='metric-value'),
                        html.Div("Sharpe Ratio", className='metric-label')
                    ], className='metric-box')
                ], style={'display': 'flex', 'justify-content': 'space-between'}),
                html.Div([
                    html.Div([
                        html.Div(f"{max_drawdown:.2f}%", className='metric-value negative'),
                        html.Div("Max Drawdown", className='metric-label')
                    ], className='metric-box'),
                    html.Div([
                        html.Div(f"{len(win_trades)}", className='metric-value positive'),
                        html.Div("Winning Trades", className='metric-label')
                    ], className='metric-box'),
                    html.Div([
                        html.Div(f"{len(loss_trades)}", className='metric-value negative'),
                        html.Div("Losing Trades", className='metric-label')
                    ], className='metric-box')
                ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-top': '10px'})
            ])
        
        @self.app.callback(
            Output('open-positions-content', 'children'),
            [Input('portfolio-data', 'children')]
        )
        def update_open_positions(portfolio_data):
            """Update open positions card."""
            if not portfolio_data:
                return html.Div("No open positions")
            
            positions = self.portfolio.positions
            
            if not positions:
                return html.Div("No open positions")
            
            # Create table data
            table_data = []
            for symbol, pos in positions.items():
                current_value = pos['size'] * pos['current_price']
                entry_value = pos['size'] * pos['entry_price']
                profit_loss = current_value - entry_value
                profit_loss_pct = (pos['current_price'] / pos['entry_price'] - 1) * 100
                
                table_data.append({
                    'Symbol': symbol,
                    'Size': f"{pos['size']}",
                    'Entry Price': f"${pos['entry_price']:.2f}",
                    'Current Price': f"${pos['current_price']:.2f}",
                    'P/L': f"${profit_loss:.2f} ({profit_loss_pct:.2f}%)",
                    'Stop Loss': f"${pos['stop_loss']:.2f}" if pos['stop_loss'] else "N/A",
                    'Take Profit': f"${pos['take_profit']:.2f}" if pos['take_profit'] else "N/A"
                })
            
            return dash_table.DataTable(
                id='positions-table',
                columns=[{'name': col, 'id': col} for col in table_data[0].keys()],
                data=table_data,
                style_table={'overflowX': 'auto'},)