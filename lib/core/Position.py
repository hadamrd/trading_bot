import lib.core.utils as utils


class PositionType:
    LONG = 0
    SHORT = 1


class PositionStatus:
    OPEN = 0
    CLOSED = 1


class Position:
    def __init__(
        self,
        type,
        open_timestamp,
        open_price,
        open_balance,
        close_timestamp=None,
        close_price=None,
        profit=None,
        time_held=None,
        closed_reason=None,
        status="open",
    ):
        self.type = type
        self.open_timestamp = open_timestamp
        self.open_balance = open_balance
        self.close_timestamp = close_timestamp
        self.open_price = open_price
        self.close_price = close_price
        self.profit = profit
        self.time_held_hours = time_held
        self.min_profit_reached = float("inf")
        self.closed_reason = closed_reason
        self.status = status
        self.max_profit_reached = - float("inf")

    def close(self, price, timestamp, close_balance, reason="n/a"):
        self.close_price = price
        self.close_timestamp = timestamp
        self.status = PositionStatus.CLOSED
        self.profit = close_balance - self.open_balance
        self.time_held_hours = utils.time_diff_hours(self.open_timestamp, timestamp)
        self.closed_reason = reason

    def is_long(self):
        return self.type == PositionType.LONG

    def is_short(self):
        return self.type == PositionType.SHORT

    def to_json(self):
        return {
            "type": self.type,
            "open_timestamp": self.open_timestamp,
            "open_price": self.open_price,
            "open_balance": self.open_balance,
            "close_timestamp": self.close_timestamp,
            "close_price": self.close_price,
            "profit": self.profit,
            "time_held": self.time_held_hours,
            "min_value_reached": self.min_profit_reached,
            "max_value_reached": self.max_profit_reached,
            "closed_reason": self.closed_reason,
            "status": self.status,
        }

    @classmethod
    def from_json(cls, json):
        return cls(**json["type"])
