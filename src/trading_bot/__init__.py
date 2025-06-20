"""
Trading Bot Package

A simple, focused trading strategy framework with MongoDB storage.
"""

__version__ = "0.1.0"
__author__ = "Trading Bot Team"

# Core imports for easy access
from .core.models import Position, MarketData, TradingConfig, BacktestConfig
from .core.enums import TimeFrame, OrderSide, TradeStatus
from .data.market_data import MarketDataManager

__all__ = [
    "Position",
    "MarketData", 
    "TradingConfig",
    "BacktestConfig",
    "TimeFrame",
    "OrderSide",
    "TradeStatus",
    "MarketDataManager",
]