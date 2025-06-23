"""
VWAP Bounce Strategy - Dynamic Support/Resistance Trading
Uses VWAP as institutional support/resistance levels
"""

from typing import Any, Tuple, Optional, Set, Dict
import pandas as pd
import numpy as np
import ta

from ..core.models import Position
from .base import BaseStrategy


class VWAPBounceStrategy(BaseStrategy):
    """
    VWAP Bounce Strategy
    
    Logic:
    - In uptrend (price > VWAP): Buy bounces off VWAP support
    - In downtrend (price < VWAP): Sell rejections at VWAP resistance
    - Uses VWAP as dynamic support/resistance level
    - Goes WITH the trend, not against it
    """
    
    INDICATOR_PARAMS: Set[str] = {
        'vwap_period', 'atr_period', 'volume_period', 'trend_period'
    }
    
    def _init_strategy(self,
                      # VWAP parameters
                      vwap_period: int = 20,
                      
                      # Bounce detection
                      vwap_touch_distance: float = 0.002,    # 0.2% from VWAP = "touching"
                      min_bounce_strength: float = 0.001,    # 0.1% move away from VWAP
                      
                      # Trend filters
                      trend_period: int = 50,                # Longer trend context
                      min_trend_strength: float = 0.01,      # 1% away from longer VWAP
                      
                      # Volume confirmation
                      volume_period: int = 20,
                      min_volume_ratio: float = 1.2,         # Volume spike on bounce
                      
                      # Risk management
                      atr_period: int = 14,
                      stop_loss_atr: float = 1.5,            # Stop beyond VWAP
                      take_profit_atr: float = 3.0,          # 2:1 risk/reward
                      
                      # Position sizing
                      position_size_pct: float = 0.03,
                      
                      # Additional filters
                      require_momentum_confirmation: bool = True,
                      max_time_at_vwap: int = 10):           # Max periods near VWAP
        
        # Store parameters
        self.vwap_period = vwap_period
        self.vwap_touch_distance = vwap_touch_distance
        self.min_bounce_strength = min_bounce_strength
        self.trend_period = trend_period
        self.min_trend_strength = min_trend_strength
        self.volume_period = volume_period
        self.min_volume_ratio = min_volume_ratio
        self.atr_period = atr_period
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.position_size_pct = position_size_pct
        self.require_momentum_confirmation = require_momentum_confirmation
        self.max_time_at_vwap = max_time_at_vwap

    @classmethod
    def calculate_indicators(cls, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate VWAP and supporting indicators."""
        df = df.copy()
        
        # Get parameters
        vwap_period = params.get('vwap_period', 20)
        trend_period = params.get('trend_period', 50)
        volume_period = params.get('volume_period', 20)
        atr_period = params.get('atr_period', 14)
        
        if len(df) < max(vwap_period, trend_period, volume_period):
            return df
        
        try:
            # VWAP calculation
            df['vwap'] = ta.volume.volume_weighted_average_price(
                df['high'], df['low'], df['close'], df['volume'], 
                window=vwap_period
            )
            
            # Longer-term VWAP for trend context
            df['vwap_trend'] = ta.volume.volume_weighted_average_price(
                df['high'], df['low'], df['close'], df['volume'],
                window=trend_period
            )
            
            # Distance from VWAP
            df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
            df['vwap_trend_distance'] = (df['close'] - df['vwap_trend']) / df['vwap_trend']
            
            # Trend identification
            df['uptrend'] = df['close'] > df['vwap']
            df['downtrend'] = df['close'] < df['vwap']
            df['strong_uptrend'] = df['vwap_trend_distance'] > 0.01  # 1% above long VWAP
            df['strong_downtrend'] = df['vwap_trend_distance'] < -0.01  # 1% below long VWAP
            
            # VWAP touch detection
            df['near_vwap'] = abs(df['vwap_distance']) <= 0.002  # Within 0.2% of VWAP
            
            # Bounce detection
            df['vwap_distance_prev'] = df['vwap_distance'].shift(1)
            df['vwap_distance_prev2'] = df['vwap_distance'].shift(2)
            
            # Bullish bounce: was below VWAP, now moving up
            df['bullish_bounce'] = (
                (df['vwap_distance_prev2'] < -0.002) &  # Was below VWAP
                (df['vwap_distance_prev'] < 0) &        # Still below last period
                (df['vwap_distance'] > df['vwap_distance_prev']) &  # Moving toward/above VWAP
                (df['close'] > df['close'].shift(1))    # Price rising
            )
            
            # Bearish rejection: was above VWAP, now moving down
            df['bearish_rejection'] = (
                (df['vwap_distance_prev2'] > 0.002) &   # Was above VWAP
                (df['vwap_distance_prev'] > 0) &        # Still above last period
                (df['vwap_distance'] < df['vwap_distance_prev']) &  # Moving toward/below VWAP
                (df['close'] < df['close'].shift(1))    # Price falling
            )
            
            # Volume confirmation
            df['volume_sma'] = df['volume'].rolling(window=volume_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # ATR for stops
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=atr_period
            )
            
            # Momentum confirmation
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['price_momentum'] = df['close'].pct_change(3)
            
            # Time near VWAP (reduce chop trades)
            df['periods_near_vwap'] = df['near_vwap'].rolling(window=10).sum()
            
        except Exception as e:
            print(f"Error calculating VWAP bounce indicators: {e}")
        
        return df.ffill().fillna(0)

    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        """Buy on bullish VWAP bounce in uptrend."""
        
        # Required indicators check
        required = ['bullish_bounce', 'uptrend', 'volume_ratio', 'atr', 'rsi']
        for indicator in required:
            if indicator not in row or pd.isna(row[indicator]):
                return False, None, 0.0
        
        # 1. Must be in uptrend context
        if not row['uptrend']:
            return False, None, 0.0
        
        # 2. Must have bullish bounce signal
        if not row['bullish_bounce']:
            return False, None, 0.0
        
        # 3. Volume confirmation
        if row['volume_ratio'] < self.min_volume_ratio:
            return False, None, 0.0
        
        # 4. Momentum confirmation (if required)
        if self.require_momentum_confirmation:
            if row['rsi'] < 40:  # Don't buy in strong downward momentum
                return False, None, 0.0
        
        # 5. Don't trade if been chopping around VWAP too long
        if 'periods_near_vwap' in row and row['periods_near_vwap'] > self.max_time_at_vwap:
            return False, None, 0.0
        
        # 6. Trend strength filter
        if 'strong_uptrend' in row and not row['strong_uptrend']:
            # In weak trends, require stronger bounce
            if abs(row.get('vwap_distance', 0)) < 0.003:  # Need 0.3% bounce in weak trends
                return False, None, 0.0
        
        # Calculate stop loss (below VWAP)
        vwap_price = row['close'] / (1 + row['vwap_distance'])  # Back-calculate VWAP price
        atr_stop = self.stop_loss_atr * row['atr']
        stop_distance = max(atr_stop, row['close'] - vwap_price) / row['close']
        
        reason = f"VWAP Bullish Bounce (dist:{row['vwap_distance']*100:.2f}%)"
        return True, reason, stop_distance

    def sell_condition(self, position: Position, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """Sell conditions for VWAP bounce trades."""
        
        current_price = row['close']
        
        # ATR-based stops and targets
        if 'atr' in row and not pd.isna(row['atr']):
            atr_value = row['atr']
            
            # Stop loss
            stop_loss_price = position.open_price * (1 - position.stop_loss)
            if current_price <= stop_loss_price:
                return True, "Stop Loss"
            
            # Take profit
            take_profit_price = position.open_price * (1 + self.take_profit_atr * atr_value / position.open_price)
            if current_price >= take_profit_price:
                return True, "Take Profit"
        
        # VWAP-based exits
        if 'vwap_distance' in row:
            # Exit if price falls back below VWAP decisively
            if row['vwap_distance'] < -0.005:  # 0.5% below VWAP
                return True, "Below VWAP"
            
            # Exit if bearish rejection at higher level
            if 'bearish_rejection' in row and row['bearish_rejection']:
                return True, "VWAP Rejection"
        
        # Momentum exit
        if 'rsi' in row and row['rsi'] > 80:
            return True, "Overbought"
        
        # Time-based exit (don't hold too long)
        if hasattr(position, 'open_time') and hasattr(row, 'name'):
            if hasattr(row.name, 'timestamp'):
                hours_held = (row.name - position.open_time).total_seconds() / 3600
            else:
                hours_held = 0
            
            if hours_held > 24:  # Max 24 hours
                return True, "Time Exit"
        
        return False, None

    def calculate_position_size(self, available_balance: float, current_price: float, row: pd.Series) -> float:
        """Calculate position size based on trend strength and signal quality."""
        
        base_size = available_balance * self.position_size_pct
        
        # Adjust size based on trend strength
        if 'strong_uptrend' in row:
            if row['strong_uptrend']:
                base_size *= 1.3  # Larger positions in strong trends
        
        # Adjust based on volume confirmation
        if 'volume_ratio' in row and not pd.isna(row['volume_ratio']):
            if row['volume_ratio'] > 2.0:  # Very high volume
                base_size *= 1.2
            elif row['volume_ratio'] < 1.5:  # Lower volume
                base_size *= 0.8
        
        # Adjust based on VWAP distance (closer to VWAP = better risk/reward)
        if 'vwap_distance' in row and not pd.isna(row['vwap_distance']):
            distance = abs(row['vwap_distance'])
            if distance < 0.001:  # Very close to VWAP
                base_size *= 1.2
            elif distance > 0.005:  # Far from VWAP
                base_size *= 0.8
        
        return min(base_size, available_balance * 0.08)  # Max 8% per trade

    def __str__(self) -> str:
        return f"VWAPBounce(period={self.vwap_period}, touch_dist={self.vwap_touch_distance*100:.1f}%)"
