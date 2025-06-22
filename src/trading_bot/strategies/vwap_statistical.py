"""
VWAP Statistical Reversion Strategy with Z-scores and volume confirmation.
Enhanced version of VWAP bounce with statistical rigor.
"""

from typing import Any

import pandas as pd
import ta

from ..core.models import Position
from .base import BaseStrategy


class VWAPStatisticalStrategy(BaseStrategy):
    """
    Statistical VWAP reversion strategy using Z-scores.

    Based on research showing mean reversion works well in crypto,
    especially when using statistical measures of deviation.
    """

    INDICATOR_PARAMS: set[str] = {
        'vwap_period', 'zscore_period', 'volume_period', 'rsi_period', 'atr_period'
    }

    @classmethod
    def calculate_indicators(cls, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Calculate VWAP, Z-scores, and supporting indicators."""
        df = df.copy()

        # Extract parameters
        vwap_period = params.get('vwap_period', 20)
        zscore_period = params.get('zscore_period', 20)
        volume_period = params.get('volume_period', 20)
        rsi_period = params.get('rsi_period', 14)
        atr_period = params.get('atr_period', 14)

        if len(df) < max(vwap_period, zscore_period, volume_period):
            return df

        try:
            # VWAP calculation
            df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
                df['high'], df['low'], df['close'], df['volume'],
                window=vwap_period
            ).volume_weighted_average_price()

            # VWAP deviation and Z-score
            df['vwap_deviation'] = df['close'] - df['vwap']
            df['vwap_deviation_pct'] = df['vwap_deviation'] / df['vwap']

            # Rolling statistics for Z-score
            rolling_mean = df['vwap_deviation_pct'].rolling(window=zscore_period).mean()
            rolling_std = df['vwap_deviation_pct'].rolling(window=zscore_period).std()
            df['vwap_zscore'] = (df['vwap_deviation_pct'] - rolling_mean) / rolling_std

            # Volume analysis
            df['volume_sma'] = df['volume'].rolling(window=volume_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

            # Volume-weighted price change
            df['price_change'] = df['close'].pct_change()
            df['volume_weighted_change'] = df['price_change'] * df['volume_ratio']

            # RSI for overbought/oversold confirmation
            df['rsi'] = ta.momentum.rsi(df['close'], window=rsi_period)

            # ATR for volatility-adjusted stops
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=atr_period
            )

            # VWAP slope (trend of VWAP itself)
            df['vwap_slope'] = df['vwap'].diff(5) / df['vwap'].shift(5)

            # Statistical bands around VWAP
            df['vwap_upper_2std'] = df['vwap'] + 2 * rolling_std * df['vwap']
            df['vwap_lower_2std'] = df['vwap'] - 2 * rolling_std * df['vwap']
            df['vwap_upper_1std'] = df['vwap'] + 1 * rolling_std * df['vwap']
            df['vwap_lower_1std'] = df['vwap'] - 1 * rolling_std * df['vwap']

            # Time-based filters (assuming 15-minute data)
            df['hour'] = pd.to_datetime(df.index).hour if hasattr(df.index, 'hour') else 12
            df['is_active_session'] = (
                ((df['hour'] >= 8) & (df['hour'] <= 11)) |  # European morning
                ((df['hour'] >= 14) & (df['hour'] <= 17)) | # US morning
                ((df['hour'] >= 20) & (df['hour'] <= 23))   # Asian session
            ).astype(int)

            # Momentum divergence
            price_momentum = df['close'].rolling(window=5).apply(
                lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0]
            )
            volume_momentum = df['volume'].rolling(window=5).mean()
            df['momentum_divergence'] = (price_momentum.abs() > 0.01) & (volume_momentum < df['volume_sma'])

        except Exception as e:
            print(f"Error calculating VWAP indicators: {e}")

        return df.ffill().fillna(0)

    def _init_strategy(self,
                      # VWAP parameters
                      vwap_period: int = 20,
                      zscore_period: int = 20,

                      # Entry thresholds
                      zscore_entry_threshold: float = -1.5,  # Buy when Z-score < -1.5
                      zscore_exit_threshold: float = 0.5,    # Exit when Z-score > 0.5

                      # Volume confirmation
                      volume_period: int = 20,
                      min_volume_ratio: float = 1.2,

                      # RSI filters
                      rsi_period: int = 14,
                      rsi_oversold: int = 35,
                      rsi_overbought: int = 65,

                      # Risk management
                      atr_period: int = 14,
                      stop_loss_atr: float = 2.5,
                      take_profit_multiple: float = 2.0,  # Risk-reward ratio

                      # Additional filters
                      require_active_session: bool = True,
                      max_vwap_slope: float = 0.002,  # Avoid strong trends
                      position_size_pct: float = 0.03):
        """Initialize strategy parameters."""

        # VWAP parameters
        self.vwap_period = vwap_period
        self.zscore_period = zscore_period

        # Entry/exit thresholds
        self.zscore_entry_threshold = zscore_entry_threshold
        self.zscore_exit_threshold = zscore_exit_threshold

        # Volume confirmation
        self.volume_period = volume_period
        self.min_volume_ratio = min_volume_ratio

        # RSI filters
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        # Risk management
        self.atr_period = atr_period
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_multiple = take_profit_multiple

        # Additional filters
        self.require_active_session = require_active_session
        self.max_vwap_slope = max_vwap_slope
        self.position_size_pct = position_size_pct

    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> tuple[bool, str | None, float]:
        """Determine buy signal based on VWAP Z-score and confirmations."""

        # Check for required indicators
        required = [
            'vwap_zscore', 'volume_ratio', 'rsi', 'vwap_slope',
            'atr', 'is_active_session'
        ]
        for indicator in required:
            if indicator not in row or pd.isna(row[indicator]):
                return False, None, 0.0

        # Primary signal: VWAP Z-score oversold
        zscore_signal = row['vwap_zscore'] <= self.zscore_entry_threshold

        # Volume confirmation
        volume_confirmation = row['volume_ratio'] >= self.min_volume_ratio

        # RSI confirmation (not too overbought)
        rsi_confirmation = row['rsi'] <= self.rsi_oversold

        # Trend filter: avoid strong uptrends (let mean reversion work)
        trend_filter = abs(row['vwap_slope']) <= self.max_vwap_slope

        # Session filter
        session_filter = (not self.require_active_session) or row['is_active_session']

        # Momentum divergence (price down but volume not confirming)
        momentum_ok = 'momentum_divergence' not in row or not row['momentum_divergence']

        # All conditions must be met
        if (zscore_signal and volume_confirmation and rsi_confirmation and
            trend_filter and session_filter and momentum_ok):

            stop_loss = self.stop_loss_atr * row['atr'] / row['close']

            reason = f"VWAP Z-Score {row['vwap_zscore']:.2f}"
            return True, reason, stop_loss

        return False, None, 0.0

    def sell_condition(self, position: Position, row: pd.Series) -> tuple[bool, str | None]:
        """Determine sell signal based on Z-score reversion and risk management."""

        required = ['vwap_zscore', 'rsi', 'atr']
        for indicator in required:
            if indicator not in row or pd.isna(row[indicator]):
                return False, None

        current_price = row['close']

        # ATR-based stop loss
        atr_stop_loss = position.open_price * (1 - self.stop_loss_atr * row['atr'] / position.open_price)
        if current_price <= atr_stop_loss:
            return True, "ATR Stop Loss"

        # Take profit based on risk-reward ratio
        risk_amount = position.open_price - atr_stop_loss
        take_profit_price = position.open_price + (risk_amount * self.take_profit_multiple)
        if current_price >= take_profit_price:
            return True, "Risk-Reward Target"

        # Z-score reversion exit
        if row['vwap_zscore'] >= self.zscore_exit_threshold:
            return True, f"Z-Score Reversion {row['vwap_zscore']:.2f}"

        # RSI overbought exit
        if row['rsi'] >= self.rsi_overbought:
            return True, "RSI Overbought"

        # Emergency exit if Z-score goes extremely positive (failed reversion)
        if row['vwap_zscore'] >= 2.0:
            return True, "Failed Reversion"

        return False, None

    def calculate_position_size(self, available_balance: float, current_price: float, row: pd.Series) -> float:
        """Calculate position size based on Z-score strength and volatility."""

        base_size = available_balance * self.position_size_pct

        # Adjust size based on Z-score strength
        if 'vwap_zscore' in row and not pd.isna(row['vwap_zscore']):
            zscore = abs(row['vwap_zscore'])

            # Stronger signals get larger positions
            if zscore >= 2.0:
                size_multiplier = 1.5
            elif zscore >= 1.5:
                size_multiplier = 1.2
            else:
                size_multiplier = 1.0

            base_size *= size_multiplier

        # Adjust for volatility (reduce size in high volatility)
        if 'atr' in row and not pd.isna(row['atr']):
            atr_pct = row['atr'] / current_price
            if atr_pct > 0.03:  # High volatility
                base_size *= 0.8
            elif atr_pct < 0.015:  # Low volatility
                base_size *= 1.1

        return min(base_size, available_balance * 0.1)  # Max 10% per trade
