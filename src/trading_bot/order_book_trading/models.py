from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class OrderLevel:
    """Order book level - reuse from existing code"""
    price: float
    size: float
    
    @property
    def value(self) -> float:
        return self.price * self.size


@dataclass
class MarketSituation:
    """Current market analysis - simple but flexible"""
    symbol: str
    price: float
    timestamp: datetime
    
    # Order book data
    bid_levels: List[OrderLevel]
    ask_levels: List[OrderLevel]
    
    # Basic analysis
    spread_pct: float = 0.0
    large_bid_wall: Optional[OrderLevel] = None
    large_ask_wall: Optional[OrderLevel] = None
    bid_pressure: float = 0.0  # 0-1, higher = more buying pressure
    
    @property
    def has_buy_opportunity(self) -> bool:
        """Simple logic for buy opportunity"""
        return (self.large_bid_wall is not None or 
                self.bid_pressure > 0.7)
    
    @property
    def has_sell_opportunity(self) -> bool:
        """Simple logic for sell opportunity"""
        return (self.large_ask_wall is not None or 
                self.bid_pressure < 0.3)
