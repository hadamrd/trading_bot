"""
Multi-Factor Strategy - Combines multiple signal types
Adapted from your old repo to current framework
"""

from typing import Any, Tuple, Optional, Set, Dict
import pandas as pd
import numpy as np
import ta

from ..core.models import Position
from .base import BaseStrategy


class MultiFactorStrategy(BaseStrategy):
    """
    Multi-Factor Strategy combining 6 signal types:
    1. Short Cross (price crossing above short EMA)
    2. Long Cross (price crossing above long EMA)  
    3. Short Bounce (bouncing off short EMA)
    4. Long Bounce (bouncing off long EMA)
    5. VWAP Bounce (bouncing off VWAP)
    6. EMA Golden Cross (short EMA crossing above long EMA)
    """
    
    INDICATOR_PARAMS: Set[str] = {
        'ema_short_period', 'ema_long_period', 'rsi_period', 
        'volume_ma_period', 'atr_period'
    }
    
    def _init_strategy(self,
                      # Core EMA parameters
                      ema_short_period: int = 9,
                      ema_long_period: int = 21,
                      
                      # RSI parameters
                      rsi_period: int = 14,
                      rsi_oversold: int = 30,
                      rsi_overbought: int = 70,
                      
                      # Volume parameters
                      volume_ma_period: int = 20,
                      volume_factor: float = 1.5,
                      
                      # Risk management
                      risk_per_trade: float = 0.01,
                      take_profit_factor: float = 1.5,
                      stop_loss_factor: float = 1.0,
                      atr_period: int = 14,
                      
                      # Signal-specific RSI thresholds
                      short_cross_rsi_threshold: float = 50,
                      long_cross_rsi_threshold: float = 60,
                      short_bounce_rsi_threshold: float = 40,
                      long_bounce_rsi_threshold: float = 50,
                      vwap_bounce_rsi_threshold: float = 60,
                      ema_golden_cross_rsi_threshold: float = 40,
                      
                      # Signal-specific ATR factors
                      short_cross_atr_factor: float = 0.9,
                      long_cross_atr_factor: float = 1.1,
                      short_bounce_atr_factor: float = 1.0,
                      long_bounce_atr_factor: float = 1.0,
                      vwap_bounce_atr_factor: float = 1.0,
                      ema_golden_cross_atr_factor: float = 1.0,
                      
                      # Derivative factors
                      ema_short_derivative_factor: float = 0.0001,
                      vwap_ema_derivative_factor: float = 0.11,
                      
                      # Signal toggles
                      use_short_cross: bool = True,
                      use_long_cross: bool = True,
                      use_short_bounce: bool = True,
                      use_long_bounce: bool = True,
                      use_vwap_bounce: bool = True,
                      use_ema_golden_cross: bool = True):
        
        # Store all parameters
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
        
        # Signal-specific parameters
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
        
        # Signal toggles
        self.use_short_cross = use_short_cross
        self.use_long_cross = use_long_cross
        self.use_short_bounce = use_short_bounce
        self.use_long_bounce = use_long_bounce
        self.use_vwap_bounce = use_vwap_bounce
        self.use_ema_golden_cross = use_ema_golden_cross

    @classmethod
    def calculate_indicators(cls, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate all indicators for multi-factor strategy."""
        df = df.copy()
        
        # Get parameters
        ema_short_period = params.get('ema_short_period', 9)
        ema_long_period = params.get('ema_long_period', 21)
        rsi_period = params.get('rsi_period', 14)
        volume_ma_period = params.get('volume_ma_period', 20)
        atr_period = params.get('atr_period', 14)
        
        if len(df) < max(ema_long_period, volume_ma_period, atr_period):
            return df
        
        try:
            # RSI
            df["rsi"] = ta.momentum.rsi(df["close"], window=rsi_period)
            
            # VWAP
            df["vwap"] = ta.volume.volume_weighted_average_price(
                df["high"], df["low"], df["close"], df["volume"], 
                window=volume_ma_period
            )
            
            # EMAs
            df["emaShort"] = ta.trend.ema_indicator(df["close"], window=ema_short_period)
            df["emaLong"] = ta.trend.ema_indicator(df["close"], window=ema_long_period)
            
            # EMA derivative (momentum)
            df['smoothed_emaShort'] = df['emaShort'].rolling(window=9).mean()
            df['emaShort_first_derivative'] = df['smoothed_emaShort'].diff()
            df['emaShort_second_derivative'] = df['emaShort_first_derivative'].diff()
            
            # Cross signals
            df["ema_diff"] = df["emaShort"] - df["emaLong"]
            df["shortcross"] = df["close"] - df["emaShort"]
            df['longcross'] = df["close"] - df["emaLong"]
            df['vwapcross'] = df["close"] - df["vwap"]
            
            # ATR for position sizing
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=atr_period
            )
            
        except Exception as e:
            print(f"Error calculating multi-factor indicators: {e}")
        
        return df.ffill().fillna(0)

    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        """Check all 6 signal types for buy conditions."""
        
        # Check required indicators
        required = ['rsi', 'shortcross', 'longcross', 'vwapcross', 'ema_diff', 'atr']
        for indicator in required:
            if indicator not in row or pd.isna(row[indicator]):
                return False, None, 0.0
            if indicator not in prev_row or pd.isna(prev_row[indicator]):
                return False, None, 0.0
        
        stop_loss = self.calculate_stop_loss(row)
        
        # Signal 1: Short Cross (price crossing above short EMA)
        if (self.use_short_cross and 
            row['rsi'] < self.short_cross_rsi_threshold and 
            prev_row["shortcross"] < row["shortcross"] < 0 and 
            row["shortcross"] + self.short_cross_atr_factor * row['atr'] >= 0):
            return True, "Short Cross", stop_loss

        # Signal 2: Long Cross (price crossing above long EMA)
        if (self.use_long_cross and 
            row['rsi'] < self.long_cross_rsi_threshold and 
            prev_row["longcross"] < row["longcross"] < 0 and 
            row["longcross"] + self.long_cross_atr_factor * row['atr'] >= 0):
            return True, "Long Cross", stop_loss

        # Signal 3: Short Bounce (bouncing off short EMA)
        if (self.use_short_bounce and 
            row['rsi'] < self.short_bounce_rsi_threshold and 
            row["shortcross"] <= self.short_bounce_atr_factor * row['atr'] and 
            prev_row["shortcross"] > 0 and 
            'emaShort_first_derivative' in row and
            row['emaShort_first_derivative'] > self.ema_short_derivative_factor * row['atr']):
            return True, "Short Bounce", stop_loss
        
        # Signal 4: Long Bounce (bouncing off long EMA)
        if (self.use_long_bounce and 
            row['rsi'] < self.long_bounce_rsi_threshold and 
            prev_row["longcross"] > 0 and 
            prev_row["longcross"] < row["longcross"] < 0 and 
            row["longcross"] + self.long_bounce_atr_factor * row['atr'] >= 0):
            return True, "Long Bounce", stop_loss
        
        # Signal 5: VWAP Bounce (bouncing off VWAP)
        if (self.use_vwap_bounce and 
            row['rsi'] < self.vwap_bounce_rsi_threshold and 
            row["vwapcross"] <= self.vwap_bounce_atr_factor * row['atr'] and 
            prev_row["vwapcross"] >= 0 and 
            'emaShort_first_derivative' in row and
            row['emaShort_first_derivative'] > self.vwap_ema_derivative_factor * row['atr']):
            return True, "VWAP Bounce", stop_loss
        
        # Signal 6: EMA Golden Cross (short EMA crossing above long EMA)
        if (self.use_ema_golden_cross and 
            row['rsi'] < self.ema_golden_cross_rsi_threshold and 
            prev_row['ema_diff'] < row['ema_diff'] and 
            row['ema_diff'] + self.ema_golden_cross_atr_factor * row['atr'] >= 0):
            return True, "EMA Golden Cross", stop_loss

        return False, None, 0.0

    def sell_condition(self, position: Position, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """Sell conditions based on risk management."""
        
        current_price = row["close"]
        take_profit = position.open_price * (1 + self.take_profit_factor * self.risk_per_trade)
        stop_loss = position.open_price * (1 - self.stop_loss_factor * self.risk_per_trade)

        if current_price >= take_profit:
            return True, "Take Profit"
        
        if current_price <= stop_loss:
            return True, "Stop Loss"

        # RSI overbought exit
        if 'rsi' in row and row['rsi'] > self.rsi_overbought:
            return True, "RSI Overbought"

        return False, None

    def calculate_stop_loss(self, row: pd.Series) -> float:
        """Calculate stop loss percentage."""
        return self.stop_loss_factor * self.risk_per_trade

    def calculate_position_size(self, available_balance: float, current_price: float, row: pd.Series) -> float:
        """Calculate position size based on risk per trade."""
        stop_loss_amount = self.calculate_stop_loss(row) * current_price
        risk_amount = available_balance * self.risk_per_trade
        shares = risk_amount / stop_loss_amount
        return shares * current_price

    def __str__(self) -> str:
        active_signals = []
        if self.use_short_cross: active_signals.append("SC")
        if self.use_long_cross: active_signals.append("LC") 
        if self.use_short_bounce: active_signals.append("SB")
        if self.use_long_bounce: active_signals.append("LB")
        if self.use_vwap_bounce: active_signals.append("VB")
        if self.use_ema_golden_cross: active_signals.append("GC")
        
        return f"MultiFactor({'+'.join(active_signals)})"