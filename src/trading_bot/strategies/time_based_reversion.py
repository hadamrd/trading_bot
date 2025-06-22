"""
Time-Based Mean Reversion Strategy leveraging intraday patterns.
Exploits documented time-of-day effects in crypto markets.
"""

from typing import Tuple, Optional, Dict, Any, Set
import pandas as pd
import ta
import numpy as np

from ..strategies.base import BaseStrategy
from ..core.models import Position


class TimeBasedReversionStrategy(BaseStrategy):
    """
    Time-based mean reversion strategy exploiting intraday patterns.
    
    Based on research showing crypto has strong intraday seasonality
    and short-term mean reversion effects, especially during specific hours.
    """
    
    INDICATOR_PARAMS: Set[str] = {
        'short_ma_period', 'rsi_period', 'atr_period', 'volume_period'
    }
    
    @classmethod
    def calculate_indicators(cls, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate indicators with focus on short-term mean reversion."""
        df = df.copy()
        
        # Extract parameters
        short_ma_period = params.get('short_ma_period', 10)
        rsi_period = params.get('rsi_period', 9)  # Shorter RSI for scalping
        atr_period = params.get('atr_period', 14)
        volume_period = params.get('volume_period', 20)
        
        if len(df) < max(short_ma_period, rsi_period, atr_period):
            return df
        
        try:
            # Short-term moving average for mean reversion
            df['sma_short'] = ta.trend.sma_indicator(df['close'], window=short_ma_period)
            df['distance_from_ma'] = (df['close'] - df['sma_short']) / df['sma_short']
            
            # Short-period RSI for quick reversals
            df['rsi_short'] = ta.momentum.rsi(df['close'], window=rsi_period)
            
            # ATR for volatility-based position sizing
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=atr_period
            )
            
            # Recent price velocity (momentum)
            df['price_velocity'] = df['close'].pct_change(3)  # 3-period change
            df['velocity_ma'] = df['price_velocity'].rolling(window=5).mean()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=volume_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Time-based features
            if hasattr(df.index, 'hour'):
                df['hour'] = df.index.hour
            else:
                # For numeric index, create mock hours for testing
                df['hour'] = (df.index % 96) // 4  # Assuming 15-min bars, 96 per day
            
            # Define trading sessions (UTC hours)
            df['session'] = 'other'
            df.loc[(df['hour'] >= 7) & (df['hour'] <= 10), 'session'] = 'european_morning'
            df.loc[(df['hour'] >= 13) & (df['hour'] <= 16), 'session'] = 'us_morning'
            df.loc[(df['hour'] >= 19) & (df['hour'] <= 22), 'session'] = 'asian_evening'
            
            # Session-specific volatility
            for session in ['european_morning', 'us_morning', 'asian_evening']:
                session_mask = df['session'] == session
                if session_mask.any():
                    session_vol = df.loc[session_mask, 'atr'].rolling(window=20).mean()
                    df.loc[session_mask, f'{session}_avg_vol'] = session_vol
            
            # Price exhaustion signals (multiple touches of levels)
            df['local_high'] = df['high'].rolling(window=5, center=True).max() == df['high']
            df['local_low'] = df['low'].rolling(window=5, center=True).min() == df['low']
            
            # Cumulative price change over recent periods
            df['cumulative_change_5'] = df['close'].pct_change(5)
            df['cumulative_change_10'] = df['close'].pct_change(10)
            
            # Volume-price divergence
            df['price_change_1'] = df['close'].pct_change()
            df['volume_change_1'] = df['volume'].pct_change()
            df['vp_divergence'] = (
                (df['price_change_1'] > 0) & (df['volume_change_1'] < 0) |
                (df['price_change_1'] < 0) & (df['volume_change_1'] > 0)
            ).astype(int)
            
            # Bollinger Band squeeze detection
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
            df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(window=20).quantile(0.2)
            
        except Exception as e:
            print(f"Error calculating time-based indicators: {e}")
        
        return df.ffill().fillna(0)

    def _init_strategy(self,
                      # Mean reversion parameters
                      short_ma_period: int = 10,
                      max_distance_from_ma: float = 0.008,  # 0.8% from MA
                      
                      # RSI parameters
                      rsi_period: int = 9,
                      rsi_oversold: int = 25,
                      rsi_overbought: int = 75,
                      
                      # Time-based filters
                      preferred_sessions: list = None,
                      avoid_low_volume_hours: bool = True,
                      
                      # Momentum filters
                      max_velocity: float = 0.015,  # Max 1.5% recent move
                      
                      # Volume requirements
                      volume_period: int = 20,
                      min_volume_ratio: float = 0.8,
                      
                      # Risk management
                      atr_period: int = 14,
                      stop_loss_atr: float = 1.5,
                      take_profit_atr: float = 2.5,
                      position_size_pct: float = 0.025):
        """Initialize strategy parameters."""
        
        # Mean reversion
        self.short_ma_period = short_ma_period
        self.max_distance_from_ma = max_distance_from_ma
        
        # RSI
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        # Time-based
        self.preferred_sessions = preferred_sessions or ['european_morning', 'us_morning']
        self.avoid_low_volume_hours = avoid_low_volume_hours
        
        # Momentum
        self.max_velocity = max_velocity
        
        # Volume
        self.volume_period = volume_period
        self.min_volume_ratio = min_volume_ratio
        
        # Risk management
        self.atr_period = atr_period
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.position_size_pct = position_size_pct

    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        """Determine buy signal based on mean reversion and time factors."""
        
        # Check for required indicators
        required = [
            'distance_from_ma', 'rsi_short', 'price_velocity', 
            'volume_ratio', 'session', 'atr'
        ]
        for indicator in required:
            if indicator not in row or pd.isna(row[indicator]):
                return False, None, 0.0
        
        # 1. Mean reversion signal
        price_below_ma = row['distance_from_ma'] <= -self.max_distance_from_ma
        
        # 2. RSI oversold
        rsi_oversold = row['rsi_short'] <= self.rsi_oversold
        
        # 3. Not too much recent momentum (let price settle)
        momentum_ok = abs(row['price_velocity']) <= self.max_velocity
        
        # 4. Session filter
        session_ok = row['session'] in self.preferred_sessions
        
        # 5. Volume filter
        volume_ok = row['volume_ratio'] >= self.min_volume_ratio
        
        # 6. Additional quality filters
        # Avoid buying during strong downtrends
        trend_ok = True
        if 'cumulative_change_10' in row and not pd.isna(row['cumulative_change_10']):
            trend_ok = row['cumulative_change_10'] > -0.03  # Not down more than 3% in 10 periods
        
        # Prefer when price velocity is slowing (momentum exhaustion)
        velocity_slowing = True
        if ('velocity_ma' in row and 'velocity_ma' in prev_row and 
            not pd.isna(row['velocity_ma']) and not pd.isna(prev_row['velocity_ma'])):
            if row['price_velocity'] < 0:  # During downmove
                velocity_slowing = abs(row['velocity_ma']) < abs(prev_row['velocity_ma'])
        
        # All conditions for entry
        if (price_below_ma and rsi_oversold and momentum_ok and 
            session_ok and volume_ok and trend_ok and velocity_slowing):
            
            stop_loss = self.stop_loss_atr * row['atr'] / row['close']
            
            reason = f"Time-based Reversion (RSI:{row['rsi_short']:.1f}, Dist:{row['distance_from_ma']*100:.2f}%)"
            return True, reason, stop_loss
        
        return False, None, 0.0

    def sell_condition(self, position: Position, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """Determine sell signal based on mean reversion completion."""
        
        required = ['distance_from_ma', 'rsi_short', 'atr']
        for indicator in required:
            if indicator not in row or pd.isna(row[indicator]):
                return False, None
        
        current_price = row['close']
        
        # ATR-based stops
        atr_stop_loss = position.open_price * (1 - self.stop_loss_atr * row['atr'] / position.open_price)
        atr_take_profit = position.open_price * (1 + self.take_profit_atr * row['atr'] / position.open_price)
        
        # Hard stops
        if current_price <= atr_stop_loss:
            return True, "ATR Stop Loss"
        
        if current_price >= atr_take_profit:
            return True, "ATR Take Profit"
        
        # Mean reversion completion signals
        
        # 1. Price returns to or above moving average
        if row['distance_from_ma'] >= 0.002:  # 0.2% above MA
            return True, "Returned to MA"
        
        # 2. RSI no longer oversold
        if row['rsi_short'] >= self.rsi_overbought:
            return True, "RSI Overbought"
        
        # 3. Quick exit if RSI reaches neutral (conservative)
        if row['rsi_short'] >= 55:
            return True, "RSI Neutral"
        
        # 4. Time-based exit (don't hold too long)
        hours_held = (row.name - position.open_time).total_seconds() / 3600 if hasattr(row, 'name') else 0
        if hours_held > 6:  # Max holding period
            return True, "Time Exit"
        
        # 5. Emergency exit if going further against us
        if row['distance_from_ma'] <= -0.015:  # 1.5% below MA
            return True, "Failed Reversion"
        
        return False, None

    def calculate_position_size(self, available_balance: float, current_price: float, row: pd.Series) -> float:
        """Calculate position size with time and volatility adjustments."""
        
        base_size = available_balance * self.position_size_pct
        
        # Adjust based on session quality
        if 'session' in row:
            session = row['session']
            if session == 'us_morning':
                base_size *= 1.2  # Higher volume session
            elif session == 'european_morning':
                base_size *= 1.1
            elif session == 'other':
                base_size *= 0.8  # Lower quality session
        
        # Adjust based on signal strength
        if 'rsi_short' in row and not pd.isna(row['rsi_short']):
            if row['rsi_short'] <= 20:  # Very oversold
                base_size *= 1.3
            elif row['rsi_short'] <= 25:  # Moderately oversold
                base_size *= 1.1
        
        # Reduce size during high volatility
        if 'atr' in row and not pd.isna(row['atr']):
            atr_pct = row['atr'] / current_price
            if atr_pct > 0.025:  # High volatility
                base_size *= 0.7
        
        return min(base_size, available_balance * 0.08)  # Max 8% per trade