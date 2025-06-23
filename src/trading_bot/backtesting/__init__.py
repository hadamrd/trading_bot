# src/trading_bot/backtesting/__init__.py
"""
Backtesting package.
"""

from .engine import BacktestEngine
from .analyzer import (
    TradeAnalyzer,
    PortfolioAnalyzer,
    ReportGenerator,
    analyze_backtest_results,
    quick_analysis
)

__all__ = [
    "BacktestEngine",
    "TradeAnalyzer",
    "PortfolioAnalyzer", 
    "ReportGenerator",
    "analyze_backtest_results",
    "quick_analysis"
]