#!/usr/bin/env python3
"""
Strategy Module - Base strategy class and simple implementation
"""

from abc import ABC, abstractmethod
from typing import Optional
from trading_bot.order_book_trading.models import MarketSituation
from trading_bot.order_book_trading.virtual_trader import VirtualTrader


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.trader: Optional[VirtualTrader] = None
    
    def set_trader(self, trader: VirtualTrader):
        """Set the trader instance"""
        self.trader = trader
    
    @abstractmethod
    def on_market_update(self, situation: MarketSituation):
        """Called when new market data arrives"""
        pass
