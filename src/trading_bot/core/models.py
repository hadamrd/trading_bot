"""
Core models for the trading bot.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from .enums import OrderSide, TimeFrame, TradeStatus


class Position(BaseModel):
    """Trading position model"""

    symbol: str
    open_time: datetime
    open_price: float
    amount_invested: float
    amount_bought: float
    highest_since_purchase: float
    buy_reason: str
    fee_rate: float
    stop_loss: float
    status: TradeStatus = TradeStatus.OPEN
    close_time: datetime | None = None
    close_price: float | None = None
    sell_reason: str | None = None

    @property
    def is_open(self) -> bool:
        return self.status == TradeStatus.OPEN

    @property
    def duration_hours(self) -> float:
        end_time = self.close_time or datetime.now()
        return (end_time - self.open_time).total_seconds() / 3600

    @property
    def current_price(self) -> float:
        return self.close_price if self.close_price is not None else self.highest_since_purchase

    @property
    def sell_return(self) -> float:
        """Calculate return accounting for fees - EXACT original formula"""
        return self.calculate_sell_return(self.open_price, self.current_price, self.fee_rate)

    @property
    def profit(self) -> float:
        return self.sell_return * self.amount_invested

    @property
    def liquidation_value(self) -> float:
        return (self.sell_return + 1) * self.amount_invested

    @property
    def return_percentage(self) -> float:
        """Return as percentage"""
        return self.sell_return * 100

    @staticmethod
    def calculate_sell_return(start_price: float, end_price: float, fee_rate: float) -> float:
        """EXACT original fee calculation formula"""
        return ((end_price / start_price) * ((1 - fee_rate) / (1 + fee_rate))) - 1

    def update(self, new_row):
        """Update position with new market data - match original interface"""
        current_price = new_row["close"]
        self.close_price = current_price
        if current_price > self.highest_since_purchase:
            self.highest_since_purchase = current_price

    def close_position(self, close_time: datetime, close_price: float, reason: str) -> float:
        """Close the position and return liquidation value - match original interface"""
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
    indicators: dict[str, float] = Field(default_factory=dict)

    def to_mongo_doc(self) -> dict[str, Any]:
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
    def from_mongo_doc(cls, doc: dict[str, Any]) -> "MarketData":
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

    symbols: list[str]
    timeframe: TimeFrame
    since_date: datetime
    test_start_date: datetime
    test_end_date: datetime | None = None
    initial_balance: Decimal
    fee_rate: Decimal = Decimal("0.001")


class BacktestResult(BaseModel):
    """Backtest results - cleaned up from existing results_analyzer.py"""

    symbol: str
    strategy_name: str
    config: BacktestConfig

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_profit: Decimal  # Total profit in currency
    win_rate: float

    # Performance metrics
    average_profit: Decimal
    average_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    average_holding_time: float  # In hours

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    profit_factor: float

    # Portfolio metrics
    initial_balance: Decimal
    final_balance: Decimal
    total_return: Decimal
    total_return_pct: float

    # All trades
    trades: list[Position]  # Renamed from positions to trades

    # Execution metadata
    start_time: datetime
    end_time: datetime
    execution_time_seconds: float


class StrategySignal(BaseModel):
    """Signal generated by a strategy"""

    timestamp: datetime
    symbol: str
    action: OrderSide
    reason: str
    confidence: float = Field(ge=0, le=1, default=1.0)
    stop_loss: Decimal | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
