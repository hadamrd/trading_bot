from enum import Enum


class Action(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0