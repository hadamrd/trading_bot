"""
Trading Bot Package

A simple, focused trading strategy framework with MongoDB storage.
"""

__version__ = "0.1.0"
__author__ = "Trading Bot Team"

from .core.enums import OrderSide, TimeFrame, TradeStatus

# Core imports for easy access
from .core.models import BacktestConfig, MarketData, Position, TradingConfig
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
