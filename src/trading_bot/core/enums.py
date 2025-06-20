"""
Core enums for the trading bot.
Consolidated from Action.py and other scattered enum definitions.
"""

from enum import Enum


class TradeStatus(str, Enum):
    """Trade position status"""
    OPEN = "open"
    CLOSED = "closed"


class OrderSide(str, Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class Action(Enum):
    """Trading actions - from old Action.py"""
    BUY = 1
    SELL = -1
    HOLD = 0


class TimeFrame(str, Enum):
    """Trading timeframes"""
    ONE_MINUTE = "1m"
    THREE_MINUTES = "3m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"


class PositionType(Enum):
    """Position types - from old Position.py"""
    LONG = 0
    SHORT = 1


class PositionStatus(Enum):
    """Position status - from old Position.py"""
    OPEN = 0
    CLOSED = 1


# Timeframe to seconds mapping - from old fetch_kline_data.py
TIMEFRAME_TO_SECONDS = {
    '1m': 60,
    '3m': 180,
    '5m': 300,
    '15m': 900,
    '30m': 1800,
    '1h': 3600,
    '6h': 21600,
    '12h': 43200,
    '1d': 86400,
    '3d': 259200,
    '1w': 604800,
    '1M': 2592000
}