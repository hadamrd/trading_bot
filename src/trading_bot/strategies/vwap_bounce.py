"""
VWAP Bounce Strategy - Fixed to work with the new system.
"""

import logging
from typing import Any

import pandas as pd
import ta

from ..core.models import Position
from ..strategies.base import BaseStrategy


class VWAPBounceStrategy(BaseStrategy):
    """
    VWAP Bounce Strategy - matches your test parameters.
    """

    # Define which parameters are needed for indicator calculations
    INDICATOR_PARAMS: set[str] = {'vwap_period', 'rsi_period', 'atr_period'}

    @classmethod
    def calculate_indicators(cls, df: pd.DataFrame, indicator_params: dict[str, Any]) -> pd.DataFrame:
        """Calculate all technical indicators needed for this strategy"""
        df = df.copy()

        # Extract indicator parameters
        vwap_period = indicator_params.get('vwap_period', 14)
        rsi_period = indicator_params.get('rsi_period', 14)
        atr_period = indicator_params.get('atr_period', 14)

        # Ensure we have enough data
        min_periods = max(vwap_period, rsi_period, atr_period)
        if len(df) < min_periods:
            print(f"Warning: Not enough data for indicators. Need {min_periods}, have {len(df)}")
            return df

        try:
            # Calculate VWAP
            df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
                df["high"], df["low"], df["close"], df["volume"],
                window=vwap_period
            ).volume_weighted_average_price()

            # Calculate RSI
            df["rsi"] = ta.momentum.RSIIndicator(
                df["close"],
                window=rsi_period
            ).rsi()

            # Calculate ATR
            df["atr"] = ta.volatility.AverageTrueRange(
                df["high"], df["low"], df["close"],
                window=atr_period
            ).average_true_range()

            # Calculate derived indicators
            df["vwap_distance"] = (df["close"] - df["vwap"]) / df["vwap"]
            df["volume_ma"] = df["volume"].rolling(window=vwap_period).mean()
            df["volume_ratio"] = df["volume"] / df["volume_ma"]

            # Fill NaN values with method='ffill' for the first few rows
            df = df.ffill().fillna(0)

        except Exception as e:
            print(f"Error calculating indicators: {e}")
            # Return original dataframe if indicator calculation fails
            return df

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

    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> tuple[bool, str | None, float]:
        """Determine if we should buy based on strategy conditions"""

        # Check if we have required indicators
        required_indicators = ['vwap_distance', 'rsi', 'volume_ratio']
        for indicator in required_indicators:
            if indicator not in row or pd.isna(row[indicator]):
                return False, None, 0.0
            if indicator not in prev_row or pd.isna(prev_row[indicator]):
                return False, None, 0.0

        # VWAP bounce condition
        vwap_bounce = (prev_row["vwap_distance"] < -self.bounce_threshold and
                      row["vwap_distance"] > -self.bounce_threshold and
                      row["vwap_distance"] < 0)

        # RSI oversold condition
        rsi_oversold = row["rsi"] < self.rsi_oversold

        # Volume spike condition
        volume_spike = row["volume_ratio"] > self.volume_factor

        if vwap_bounce and rsi_oversold and volume_spike:
            return True, "VWAP Bounce", self.stop_loss_percentage

        return False, None, 0.0

    def sell_condition(self, position: Position, row: pd.Series) -> tuple[bool, str | None]:
        """Determine if we should sell based on strategy conditions"""
        current_price = row["close"]
        take_profit = position.open_price * (1 + self.take_profit_percentage)
        stop_loss = position.open_price * (1 - self.stop_loss_percentage)

        if current_price >= take_profit:
            return True, "Take Profit"

        if current_price <= stop_loss:
            return True, "Stop Loss"

        # Check RSI overbought (if available)
        if "rsi" in row and not pd.isna(row["rsi"]) and row["rsi"] > self.rsi_overbought:
            return True, "RSI Overbought"

        return False, None

    def calculate_position_size(self, available_balance: float, current_price: float, row: pd.Series) -> float:
        """Calculate the position size for a trade"""
        # Use a reasonable percentage of available balance (e.g., 10%)
        position_size_percentage = 0.1  # 10% of balance per trade
        return available_balance * position_size_percentage

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
