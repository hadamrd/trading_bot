from .BaseStrategy import BaseStrategy
from tradingbot2.TradePosition import TradePosition
import pandas as pd
from typing import Tuple, Optional

class EMACrossoverStrategy(BaseStrategy):
    def __init__(self, short_period: int, long_period: int, rsi_upper_bound:int = 40, take_profit_atr:int = 12, stop_loss_atr:int = 4):
        self.short_period = short_period
        self.long_period = long_period
        self.rsi_upper_bound = rsi_upper_bound
        self.take_profit_atr = take_profit_atr
        self.stop_loss_atr = stop_loss_atr

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ema_short"] = calculate_ema(df, self.short_period)
        df["ema_long"] = calculate_ema(df, self.long_period)
        df["rsi"] = calculate_rsi(df)
        df["atr"] = calculate_atr(df, self.short_period)
        df["ema_diff"] = df["ema_short"] - df["ema_long"]
        return df

    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str]]:
        if row['rsi'] < self.rsi_upper_bound and prev_row['ema_diff'] < row['ema_diff'] and row['ema_diff'] + row['atr'] >= 0:
            return True, "EMA golden cross"
        return False, None

    def sell_condition(self, position: TradePosition, current_price: float, row: pd.Series) -> Tuple[bool, Optional[str]]:
        take_profit = self.calculate_take_profit(position, row)
        stop_loss = self.calculate_stop_loss(position, row)
        sell_return = (current_price / position.open_price) - 1

        if sell_return > take_profit:
            return True, "Take Profit"

        if sell_return <= -stop_loss:
            return True, "Stop Loss"

        return False, None

    def calculate_take_profit(self, position: TradePosition, row: pd.Series) -> float:
        return row['atr'] * self.take_profit_atr / position.open_price

    def calculate_stop_loss(self, position: TradePosition, row: pd.Series) -> float:
        return row['atr'] * self.stop_loss_atr / position.open_price
