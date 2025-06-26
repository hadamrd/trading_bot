
import asyncio
import json
import websockets
from dataclasses import dataclass
from datetime import datetime
from typing import List, Callable, Optional

from trading_bot.order_book_trading.models import MarketSituation, OrderLevel


class MarketAnalyzer:
    """Analyzes order book data into MarketSituation"""
    
    def __init__(self, wall_threshold: float = 50000):
        self.wall_threshold = wall_threshold
    
    def analyze(self, symbol: str, bids: List[List], asks: List[List]) -> MarketSituation:
        """Convert raw order book into MarketSituation"""
        
        # Convert to OrderLevel objects
        bid_levels = [OrderLevel(float(b[0]), float(b[1])) for b in bids[:10]]
        ask_levels = [OrderLevel(float(a[0]), float(a[1])) for a in asks[:10]]
        
        if not bid_levels or not ask_levels:
            return self._empty_situation(symbol)
        
        # Current price (mid price)
        current_price = (bid_levels[0].price + ask_levels[0].price) / 2
        
        # Calculate spread
        spread_pct = ((ask_levels[0].price - bid_levels[0].price) / bid_levels[0].price) * 100
        
        # Find large walls
        large_bid_wall = self._find_large_wall(bid_levels)
        large_ask_wall = self._find_large_wall(ask_levels)
        
        # Calculate bid pressure (simple version)
        bid_total = sum(level.value for level in bid_levels[:5])
        ask_total = sum(level.value for level in ask_levels[:5])
        total_value = bid_total + ask_total
        bid_pressure = bid_total / total_value if total_value > 0 else 0.5
        
        return MarketSituation(
            symbol=symbol,
            price=current_price,
            timestamp=datetime.now(),
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            spread_pct=spread_pct,
            large_bid_wall=large_bid_wall,
            large_ask_wall=large_ask_wall,
            bid_pressure=bid_pressure
        )
    
    def _find_large_wall(self, levels: List[OrderLevel]) -> Optional[OrderLevel]:
        """Find largest wall above threshold"""
        large_levels = [level for level in levels if level.value >= self.wall_threshold]
        return max(large_levels, key=lambda x: x.value) if large_levels else None
    
    def _empty_situation(self, symbol: str) -> MarketSituation:
        """Return empty situation when no data"""
        return MarketSituation(
            symbol=symbol,
            price=0.0,
            timestamp=datetime.now(),
            bid_levels=[],
            ask_levels=[]
        )
