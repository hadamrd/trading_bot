"""
Simple EMA Crossover Strategy.
Updated to match original working interface.
"""

from typing import Tuple, Optional, Dict, Any, Set
import pandas as pd
import ta

from .base import BaseStrategy
from ..core.models import Position


class EMACrossoverStrategy(BaseStrategy):
    """
    Simple EMA crossover strategy.
    Restored to match original working interface.
    """
    
    INDICATOR_PARAMS: Set[str] = {'fast_period', 'slow_period'}
    
    def _init_strategy(self, 
                      fast_period: int = 12,
                      slow_period: int = 26,
                      stop_loss_pct: float = 0.02,
                      take_profit_pct: float = 0.04,
                      position_size_pct: float = 0.1):  # Use 10% of balance per trade
        """Initialize strategy parameters."""
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position_size_pct = position_size_pct
    
    @classmethod
    def calculate_indicators(cls, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate EMA indicators."""
        df = df.copy()
        
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        
        # Calculate EMAs
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=fast_period)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=slow_period)
        
        # Calculate crossover signals
        df['ema_diff'] = df['ema_fast'] - df['ema_slow']
        df['ema_diff_prev'] = df['ema_diff'].shift(1)
        
        # Bullish crossover: fast EMA crosses above slow EMA
        df['bullish_cross'] = (
            (df['ema_diff'] > 0) & 
            (df['ema_diff_prev'] <= 0)
        )
        
        # Bearish crossover: fast EMA crosses below slow EMA
        df['bearish_cross'] = (
            (df['ema_diff'] < 0) & 
            (df['ema_diff_prev'] >= 0)
        )
        
        return df
    
    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        """
        Determine if a buy signal is present.
        Match original interface: return (should_buy, reason, stop_loss)
        """
        # Need both EMAs to be calculated
        if pd.isna(row.get('ema_fast')) or pd.isna(row.get('ema_slow')):
            return False, None, 0.0
        
        # Bullish crossover - BUY signal
        if row.get('bullish_cross', False):
            stop_loss = self.stop_loss_pct
            return True, "EMA bullish crossover", stop_loss
        
        return False, None, 0.0
    
    def sell_condition(self, position: Position, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """
        Determine if a sell signal is present.
        Match original interface: return (should_sell, reason)
        """
        current_price = row["close"]
        
        # Check take profit
        take_profit_price = float(position.open_price) * (1 + self.take_profit_pct)
        if current_price >= take_profit_price:
            return True, "Take profit triggered"
        
        # Check stop loss
        stop_loss_price = float(position.open_price) * (1 - self.stop_loss_pct)
        if current_price <= stop_loss_price:
            return True, "Stop loss triggered"
        
        # Check bearish crossover
        if row.get('bearish_cross', False):
            return True, "EMA bearish crossover"
        
        return False, None
    
    def calculate_position_size(self, available_balance: float, current_price: float, row: pd.Series) -> float:
        """
        Calculate position size based on available balance.
        Match original interface: return position size in quote currency.
        """
        return available_balance * self.position_size_pct
    
    def __str__(self) -> str:
        return f"EMACrossover(fast={self.fast_period}, slow={self.slow_period})"