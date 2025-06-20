from .BaseStrategy import BaseStrategy
from tradingbot2.TradePosition import TradePosition
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import ta
import logging

class FlexibleMultiFactorStrategy(BaseStrategy):
    def __init__(self, 
                 ema_short_period: int = 9,
                 ema_long_period: int = 21,
                 rsi_period: int = 14,
                 rsi_oversold: int = 30,
                 rsi_overbought: int = 70,
                 volume_ma_period: int = 20,
                 volume_factor: float = 1.5,
                 risk_per_trade: float = 0.01,
                 take_profit_factor: float = 1.5,
                 stop_loss_factor: float = 1.0):
        
        super().__init__()
        self.ema_short_period = ema_short_period
        self.ema_long_period = ema_long_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volume_ma_period = volume_ma_period
        self.volume_factor = volume_factor
        self.risk_per_trade = risk_per_trade
        self.take_profit_factor = take_profit_factor
        self.stop_loss_factor = stop_loss_factor
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema_short'] = ta.trend.ema_indicator(df['close'], self.ema_short_period)
        df['ema_long'] = ta.trend.ema_indicator(df['close'], self.ema_long_period)
        df['ema_diff'] = df['ema_short'] - df['ema_long']
        df['rsi'] = ta.momentum.rsi(df['close'], self.rsi_period)
        df['volume_ma'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'], self.volume_ma_period)
        return df

    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        ema_crossover = row['ema_diff'] > 0 and prev_row['ema_diff'] <= 0
        rsi_oversold = row['rsi'] < self.rsi_oversold
        volume_spike = row['volume'] > self.volume_factor * row['volume_ma']
        
        if ema_crossover and rsi_oversold and volume_spike:
            stop_loss = self.calculate_stop_loss(row)
            return True, "Buy signal: EMA crossover, RSI oversold, and volume spike", stop_loss
        return False, None, 0

    def sell_condition(self, position: TradePosition, current_price: float, row: pd.Series) -> Tuple[bool, Optional[str]]:
        take_profit = position.open_price * (1 + self.take_profit_factor * self.risk_per_trade)
        stop_loss = position.open_price * (1 - self.stop_loss_factor * self.risk_per_trade)

        if current_price >= take_profit:
            return True, "Take Profit"
        if current_price <= stop_loss:
            return True, "Stop Loss"
        if row['rsi'] > self.rsi_overbought:
            return True, "RSI overbought"

        return False, None

    def calculate_stop_loss(self, row: pd.Series) -> float:
        return self.stop_loss_factor * self.risk_per_trade

    def calculate_take_profit(self, position: TradePosition, row: pd.Series) -> float:
        return self.take_profit_factor * self.risk_per_trade
    
    def position_size(self, capital: float, current_price: float, row: pd.Series) -> float:
        stop_loss_amount = self.calculate_stop_loss(row) * current_price
        shares = (capital * self.risk_per_trade) / stop_loss_amount
        return shares * current_price