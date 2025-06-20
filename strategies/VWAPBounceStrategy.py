from .BaseStrategy import BaseStrategy
from tradingbot2.TradePosition import TradePosition
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import ta
import logging

class VWAPBounceStrategy(BaseStrategy):
    # Define which parameters are needed for indicator calculations
    INDICATOR_PARAMS = {'vwap_period', 'rsi_period', 'atr_period'}
    
    @classmethod
    def calculate_indicators(cls, df: pd.DataFrame, indicator_params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate all technical indicators needed for this strategy"""
        df = df.copy()
        
        # Extract indicator parameters
        vwap_period = indicator_params['vwap_period']
        rsi_period = indicator_params['rsi_period']
        atr_period = indicator_params['atr_period']
        
        # Calculate indicators
        df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
            df["high"], df["low"], df["close"], df["volume"], 
            window=vwap_period
        ).volume_weighted_average_price()
        
        df["rsi"] = ta.momentum.RSIIndicator(
            df["close"], 
            window=rsi_period
        ).rsi()
        
        df["atr"] = ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], 
            window=atr_period
        ).average_true_range()
        
        df["vwap_distance"] = (df["close"] - df["vwap"]) / df["vwap"]
        df["volume_ma"] = df["volume"].rolling(window=vwap_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]
        
        return df

    def _init_strategy(self,
                      vwap_period: int = 14,
                      rsi_period: int = 14,
                      rsi_oversold: int = 30,
                      rsi_overbought: int = 70,
                      bounce_threshold: float = 0.001,
                      volume_factor: float = 1.5,
                      take_profit_percentage: float = 0.03,
                      stop_loss_percentage: float = 0.01,
                      atr_period: int = 14):
        """Initialize strategy-specific parameters"""
        self.vwap_period = vwap_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bounce_threshold = bounce_threshold
        self.volume_factor = volume_factor
        self.take_profit_percentage = take_profit_percentage
        self.stop_loss_percentage = stop_loss_percentage
        self.atr_period = atr_period
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        """Determine if we should buy based on strategy conditions"""
        vwap_bounce = (prev_row["vwap_distance"] < -self.bounce_threshold and 
                      row["vwap_distance"] > -self.bounce_threshold and 
                      row["vwap_distance"] < 0)
        
        rsi_oversold = row["rsi"] < self.rsi_oversold
        volume_spike = row["volume_ratio"] > self.volume_factor
        
        if vwap_bounce and rsi_oversold and volume_spike:
            return True, "VWAP Bounce", self.stop_loss_percentage
        
        return False, None, 0.0

    def sell_condition(self, position: TradePosition, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """Determine if we should sell based on strategy conditions"""
        current_price = row["close"]
        take_profit = position.open_price * (1 + self.take_profit_percentage)
        stop_loss = position.open_price * (1 - self.stop_loss_percentage)

        if current_price >= take_profit:
            return True, "Take Profit"
        
        if current_price <= stop_loss:
            return True, "Stop Loss"
        
        if row["rsi"] > self.rsi_overbought:
            return True, "RSI Overbought"

        return False, None

    def calculate_stop_loss(self, position: TradePosition, row: pd.Series) -> float:
        """Calculate the stop loss percentage for a position"""
        return self.stop_loss_percentage

    def calculate_take_profit(self, position: TradePosition, row: pd.Series) -> float:
        """Calculate the take profit percentage for a position"""
        return self.take_profit_percentage

    def calculate_position_size(self, available_balance: float, current_price: float, row: pd.Series) -> float:
        """Calculate the position size for a trade"""
        return available_balance  # Use all available balance for each trade

    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        if not super().validate_parameters():
            return False
            
        # Add VWAP Bounce specific validations
        valid = True
        if self.vwap_period < 1:
            self.logger.error("VWAP period must be positive")
            valid = False
        if self.rsi_period < 1:
            self.logger.error("RSI period must be positive")
            valid = False
        if not 0 <= self.rsi_oversold <= self.rsi_overbought <= 100:
            self.logger.error("Invalid RSI thresholds")
            valid = False
        if self.bounce_threshold <= 0:
            self.logger.error("Bounce threshold must be positive")
            valid = False
        if self.volume_factor <= 0:
            self.logger.error("Volume factor must be positive")
            valid = False
        if self.take_profit_percentage <= 0:
            self.logger.error("Take profit percentage must be positive")
            valid = False
        if self.stop_loss_percentage <= 0:
            self.logger.error("Stop loss percentage must be positive")
            valid = False
        
        return valid
