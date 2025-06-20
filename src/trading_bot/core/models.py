"""
Core models for the trading bot.
Consolidated from TradePosition.py, models.py, and scattered model definitions.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from .enums import TradeStatus, OrderSide, TimeFrame


class Position(BaseModel):
    """Trading position model - cleaned up from TradePosition.py"""
    
    symbol: str
    open_time: datetime
    open_price: Decimal
    amount_invested: Decimal
    amount_bought: Decimal
    highest_since_purchase: Decimal
    buy_reason: str
    fee_rate: Decimal
    stop_loss: Decimal
    status: TradeStatus = TradeStatus.OPEN
    close_time: Optional[datetime] = None
    close_price: Optional[Decimal] = None
    sell_reason: Optional[str] = None
    
    @property
    def is_open(self) -> bool:
        return self.status == TradeStatus.OPEN
    
    @property
    def duration_hours(self) -> float:
        end_time = self.close_time or datetime.now()
        return (end_time - self.open_time).total_seconds() / 3600
    
    @property
    def current_price(self) -> Decimal:
        return self.close_price if self.close_price is not None else self.highest_since_purchase
    
    @property
    def sell_return(self) -> Decimal:
        """Calculate return accounting for fees"""
        return self._calculate_sell_return(self.open_price, self.current_price, self.fee_rate)
    
    @property
    def profit(self) -> Decimal:
        return self.sell_return * self.amount_invested
    
    @property
    def liquidation_value(self) -> Decimal:
        return (self.sell_return + 1) * self.amount_invested
    
    @staticmethod
    def _calculate_sell_return(start_price: Decimal, end_price: Decimal, fee_rate: Decimal) -> Decimal:
        """Calculate return accounting for buy and sell fees"""
        return ((end_price / start_price) * ((1 - fee_rate) / (1 + fee_rate))) - 1
    
    def update(self, current_price: Decimal) -> None:
        """Update position with new price data"""
        self.close_price = current_price
        if current_price > self.highest_since_purchase:
            self.highest_since_purchase = current_price
    
    def close_position(self, close_time: datetime, close_price: Decimal, reason: str) -> Decimal:
        """Close the position and return liquidation value"""
        self.close_time = close_time
        self.close_price = close_price
        self.sell_reason = reason
        self.status = TradeStatus.CLOSED
        return self.liquidation_value


class MarketData(BaseModel):
    """Market data candle"""
    
    symbol: str
    timeframe: TimeFrame
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    
    # Technical indicators (optional)
    indicators: Dict[str, float] = Field(default_factory=dict)
    
    def to_mongo_doc(self) -> Dict[str, Any]:
        """Convert to MongoDB document"""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
            "indicators": self.indicators
        }
    
    @classmethod
    def from_mongo_doc(cls, doc: Dict[str, Any]) -> "MarketData":
        """Create from MongoDB document"""
        return cls(
            symbol=doc["symbol"],
            timeframe=doc["timeframe"],
            timestamp=doc["timestamp"],
            open=Decimal(str(doc["open"])),
            high=Decimal(str(doc["high"])),
            low=Decimal(str(doc["low"])),
            close=Decimal(str(doc["close"])),
            volume=Decimal(str(doc["volume"])),
            indicators=doc.get("indicators", {})
        )


class TradingConfig(BaseModel):
    """Trading configuration"""
    
    symbol: str
    timeframe: TimeFrame
    initial_balance: Decimal
    fee_rate: Decimal = Decimal("0.001")
    price_usdt_rate: Decimal = Decimal("1.0")


class BacktestConfig(BaseModel):
    """Backtesting configuration"""
    
    symbols: List[str]
    timeframe: TimeFrame
    since_date: datetime
    test_start_date: datetime
    test_end_date: Optional[datetime] = None
    initial_balance: Decimal
    fee_rate: Decimal = Decimal("0.001")


class BacktestResult(BaseModel):
    """Backtest results - cleaned up from existing results_analyzer.py"""
    
    symbol: str
    strategy_name: str
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_profit: Decimal
    win_rate: float
    
    # Performance metrics
    average_profit: Decimal
    average_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    average_holding_time: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    profit_factor: float
    
    # All positions
    trades: List[Position]


class StrategySignal(BaseModel):
    """Signal generated by a strategy"""
    
    timestamp: datetime
    symbol: str
    action: OrderSide
    reason: str
    confidence: float = Field(ge=0, le=1, default=1.0)
    stop_loss: Optional[Decimal] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)