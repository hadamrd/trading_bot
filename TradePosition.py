from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from enum import Enum

class TradeStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"

class TradePosition(BaseModel):
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
    close_time: Optional[datetime] = None
    close_price: Optional[float] = None
    sell_reason: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True

    @property
    def is_open(self) -> bool:
        return self.status == TradeStatus.OPEN

    @property
    def duration(self) -> float:
        end_time = self.close_time or datetime.now()
        return (end_time - self.open_time).total_seconds() / 3600  # in hours

    @property
    def liquidation_value(self) -> float:
        return (self.sell_return + 1) * self.amount_invested
    
    @property
    def current_price(self) -> float:
        return self.close_price if self.close_price is not None else self.highest_since_purchase

    @property
    def sell_return(self) -> float:
        return self.calculate_sell_return(self.open_price, self.current_price, self.fee_rate)

    @property
    def profit(self) -> float:
        return self.sell_return * self.amount_invested

    @property
    def price_pnl(self) -> float:
        return (self.current_price / self.open_price) - 1

    @property
    def roi(self) -> float:
        return self.profit / self.amount_invested

    @property
    def break_even_price(self) -> float:
        return self.open_price * (1 + self.fee_rate) / (1 - self.fee_rate)

    @property
    def is_winning(self) -> bool:
        return self.profit > 0

    @staticmethod
    def calculate_sell_return(start_price: float, end_price: float, fee_rate: float) -> float:
        return ((end_price / start_price) * ((1 - fee_rate) / (1 + fee_rate))) - 1

    def update(self, new_row):
        current_price = new_row["close"]
        self.close_price = current_price
        if current_price > self.highest_since_purchase:
            self.highest_since_purchase = current_price

    def close(self, close_time: datetime, close_price: float, sell_reason: str):
        self.close_time = close_time
        self.close_price = close_price
        self.sell_reason = sell_reason
        self.status = TradeStatus.CLOSED
        return self.liquidation_value

    