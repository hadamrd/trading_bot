from .BaseStrategy import BaseStrategy
from tradingbot2.TradePosition import TradePosition
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import ta
import logging

class MultiFactorStrategy(BaseStrategy):
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
            stop_loss_factor: float = 1.0,
            atr_period: int = 14,
            short_cross_rsi_threshold: float = 50,
            long_cross_rsi_threshold: float = 60,
            short_bounce_rsi_threshold: float = 40,
            long_bounce_rsi_threshold: float = 50,
            vwap_bounce_rsi_threshold: float = 60,
            ema_golden_cross_rsi_threshold: float = 40,
            short_cross_atr_factor: float = 0.9,
            long_cross_atr_factor: float = 1.1,
            short_bounce_atr_factor: float = 1.0,
            long_bounce_atr_factor: float = 1.0,
            vwap_bounce_atr_factor: float = 1.0,
            ema_golden_cross_atr_factor: float = 1.0,
            ema_short_derivative_factor: float = 0.0001,
            vwap_ema_derivative_factor: float = 0.11,
            use_short_cross: bool = True,
            use_long_cross: bool = True,
            use_short_bounce: bool = True,
            use_long_bounce: bool = True,
            use_vwap_bounce: bool = True,
            use_ema_golden_cross: bool = True
        ):
        
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
        self.atr_period = atr_period
        
        # New parameters
        self.short_cross_rsi_threshold = short_cross_rsi_threshold
        self.long_cross_rsi_threshold = long_cross_rsi_threshold
        self.short_bounce_rsi_threshold = short_bounce_rsi_threshold
        self.long_bounce_rsi_threshold = long_bounce_rsi_threshold
        self.vwap_bounce_rsi_threshold = vwap_bounce_rsi_threshold
        self.ema_golden_cross_rsi_threshold = ema_golden_cross_rsi_threshold
        self.short_cross_atr_factor = short_cross_atr_factor
        self.long_cross_atr_factor = long_cross_atr_factor
        self.short_bounce_atr_factor = short_bounce_atr_factor
        self.long_bounce_atr_factor = long_bounce_atr_factor
        self.vwap_bounce_atr_factor = vwap_bounce_atr_factor
        self.ema_golden_cross_atr_factor = ema_golden_cross_atr_factor
        self.ema_short_derivative_factor = ema_short_derivative_factor
        self.vwap_ema_derivative_factor = vwap_ema_derivative_factor
        
        self.use_short_cross = use_short_cross
        self.use_long_cross = use_long_cross
        self.use_short_bounce = use_short_bounce
        self.use_long_bounce = use_long_bounce
        self.use_vwap_bounce = use_vwap_bounce
        self.use_ema_golden_cross = use_ema_golden_cross
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=self.rsi_period).rsi()
        df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
            df["high"], df["low"], df["close"], df["volume"], window=self.volume_ma_period
        ).volume_weighted_average_price()

        df["emaShort"] = ta.trend.EMAIndicator(df["close"], window=self.ema_short_period).ema_indicator()
        
        df['smoothed_emaShort'] = df['emaShort'].rolling(window=9).mean()
        df['emaShort_first_derivative'] = df['smoothed_emaShort'].diff()
        df['emaShort_second_derivative'] = df['emaShort_first_derivative'].diff()
        
        df["emaLong"] = ta.trend.EMAIndicator(df["close"], window=self.ema_long_period).ema_indicator()
        df["ema_diff"] = df["emaShort"] - df["emaLong"]
        df["shortcross"] = df["close"] - df["emaShort"]
        df['longcross'] = df["close"] - df["emaLong"]
        df['vwapcross'] = df["close"] - df["vwap"]
        
        if len(df['close']) < self.atr_period:
            raise Exception(f"Not enough data to calculate ATR, need at least {self.atr_period} candles but only have {len(df['close'])}")
        df['atr'] = ta.volatility.average_true_range(high=df['high'], low=df['low'], close=df['close'], window=self.atr_period)

        return df

    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        stop_loss = self.calculate_stop_loss(row)
        
        if self.use_short_cross and (row['rsi'] < self.short_cross_rsi_threshold and 
            prev_row["shortcross"] < row["shortcross"] < 0 and 
            row["shortcross"] + self.short_cross_atr_factor * row['atr'] >= 0):
            return True, "Short Cross", stop_loss

        if self.use_long_cross and (row['rsi'] < self.long_cross_rsi_threshold and 
            prev_row["longcross"] < row["longcross"] < 0 and 
            row["longcross"] + self.long_cross_atr_factor * row['atr'] >= 0):
            return True, "Long Cross", stop_loss

        if self.use_short_bounce and (row['rsi'] < self.short_bounce_rsi_threshold and 
            row["shortcross"] <= self.short_bounce_atr_factor * row['atr'] and 
            prev_row["shortcross"] > 0 and 
            row['emaShort_first_derivative'] > self.ema_short_derivative_factor * row['atr']):
            return True, "Short bounce", stop_loss
        
        if self.use_long_bounce and (row['rsi'] < self.long_bounce_rsi_threshold and 
            prev_row["longcross"] > 0 and 
            prev_row["longcross"] < row["longcross"] < 0 and 
            row["longcross"] + self.long_bounce_atr_factor * row['atr'] >= 0):
            return True, "Long bounce", stop_loss
        
        if self.use_vwap_bounce and (row['rsi'] < self.vwap_bounce_rsi_threshold and 
            row["vwapcross"] <= self.vwap_bounce_atr_factor * row['atr'] and 
            prev_row["vwapcross"] >= 0 and 
            row['emaShort_first_derivative'] > self.vwap_ema_derivative_factor * row['atr']):
            return True, "VWAP bounce", stop_loss
        
        if self.use_ema_golden_cross and (row['rsi'] < self.ema_golden_cross_rsi_threshold and 
            prev_row['ema_diff'] < row['ema_diff'] and 
            row['ema_diff'] + self.ema_golden_cross_atr_factor * row['atr'] >= 0):
            return True, "EMA golden cross", stop_loss

        return False, None, 0.0

    def sell_condition(self, position: TradePosition, row: pd.Series) -> Tuple[bool, Optional[str]]:
        current_price = row["close"]
        take_profit = position.open_price * (1 + self.take_profit_factor * self.risk_per_trade)
        stop_loss = position.open_price * (1 - self.stop_loss_factor * self.risk_per_trade)

        if current_price >= take_profit:
            return True, "Take Profit"
        
        if current_price <= stop_loss:
            return True, "Stop Loss"

        return False, None

    def calculate_stop_loss(self, row: pd.Series) -> float:
        return self.stop_loss_factor * self.risk_per_trade

    def calculate_take_profit(self, position: TradePosition, row: pd.Series) -> float:
        return self.take_profit_factor * self.risk_per_trade
    
    def calculate_position_size(self, available_balance: float, current_price: float, row: pd.Series) -> float:
        stop_loss_amount = self.calculate_stop_loss(row) * current_price
        risk_amount = available_balance * self.risk_per_trade
        shares = risk_amount / stop_loss_amount
        return shares * current_price