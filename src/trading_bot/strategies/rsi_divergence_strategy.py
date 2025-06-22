"""
Professional RSI Divergence Strategy
Based on proven institutional trading methods
"""

from typing import Any, Tuple, Optional, Set, Dict
import pandas as pd
import numpy as np
import ta
from scipy.signal import argrelextrema

from ..core.models import Position
from .base import BaseStrategy


class RSIDivergenceStrategy(BaseStrategy):
    """
    Professional RSI Divergence Strategy
    
    Looks for divergences between price and RSI:
    - Bullish: Price makes lower low, RSI makes higher low
    - Bearish: Price makes higher high, RSI makes lower high
    
    This is one of the most reliable trading signals used by professionals.
    Typical win rates: 65-80% when properly implemented.
    """

    INDICATOR_PARAMS: Set[str] = {
        'rsi_period', 'atr_period', 'volume_period', 'divergence_lookback',
        'min_swing_bars', 'ema_trend_period'
    }

    def _init_strategy(self,
                      # RSI settings
                      rsi_period: int = 14,
                      rsi_overbought: float = 70,
                      rsi_oversold: float = 30,
                      
                      # Divergence detection
                      divergence_lookback: int = 20,  # Bars to look back for swings
                      min_swing_bars: int = 5,        # Minimum bars between swings
                      min_divergence_strength: float = 0.5,  # Minimum RSI difference
                      
                      # Confirmations
                      require_volume_confirmation: bool = True,
                      require_trend_confirmation: bool = True,
                      ema_trend_period: int = 50,
                      
                      # Risk management
                      atr_period: int = 14,
                      stop_loss_atr: float = 2.0,
                      take_profit_atr: float = 4.0,  # 2:1 risk reward
                      
                      # Position sizing
                      base_position_size: float = 0.02,
                      max_position_size: float = 0.05,
                      
                      # Filters
                      min_volume_ratio: float = 1.2,
                      volume_period: int = 20):
        
        # RSI parameters
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        
        # Divergence detection
        self.divergence_lookback = divergence_lookback
        self.min_swing_bars = min_swing_bars
        self.min_divergence_strength = min_divergence_strength
        
        # Confirmations
        self.require_volume_confirmation = require_volume_confirmation
        self.require_trend_confirmation = require_trend_confirmation
        self.ema_trend_period = ema_trend_period
        
        # Risk management
        self.atr_period = atr_period
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        
        # Position sizing
        self.base_position_size = base_position_size
        self.max_position_size = max_position_size
        
        # Volume filters
        self.min_volume_ratio = min_volume_ratio
        self.volume_period = volume_period

    @classmethod
    def calculate_indicators(cls, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate all indicators needed for divergence detection."""
        df = df.copy()
        
        # Get parameters
        rsi_period = params.get('rsi_period', 14)
        atr_period = params.get('atr_period', 14)
        volume_period = params.get('volume_period', 20)
        divergence_lookback = params.get('divergence_lookback', 20)
        min_swing_bars = params.get('min_swing_bars', 5)
        ema_trend_period = params.get('ema_trend_period', 50)
        
        if len(df) < max(rsi_period, atr_period, divergence_lookback, ema_trend_period):
            return df
        
        try:
            # Core indicators
            df['rsi'] = ta.momentum.rsi(df['close'], window=rsi_period)
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=atr_period
            )
            
            # Trend filter
            df['ema_trend'] = ta.trend.ema_indicator(df['close'], window=ema_trend_period)
            df['trend_direction'] = np.where(df['close'] > df['ema_trend'], 1, -1)
            
            # Volume analysis
            df['volume_sma'] = df['volume'].rolling(window=volume_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Find swing highs and lows
            df = cls._find_swing_points(df, min_swing_bars)
            
            # Detect divergences
            df = cls._detect_divergences(df, divergence_lookback)
            
            # Additional confirmations
            df['price_momentum'] = df['close'].pct_change(5)
            df['rsi_momentum'] = df['rsi'].diff(5)
            
            # Support/Resistance levels
            df['resistance_level'] = df['high'].rolling(window=20).max()
            df['support_level'] = df['low'].rolling(window=20).min()
            
        except Exception as e:
            print(f"Error calculating RSI divergence indicators: {e}")
        
        return df.ffill().fillna(0)

    @classmethod
    def _find_swing_points(cls, df: pd.DataFrame, min_bars: int) -> pd.DataFrame:
        """Find swing highs and lows in price and RSI."""
        
        # Find price swing points
        high_indices = argrelextrema(df['high'].values, np.greater, order=min_bars)[0]
        low_indices = argrelextrema(df['low'].values, np.less, order=min_bars)[0]
        
        df['price_swing_high'] = False
        df['price_swing_low'] = False
        df.iloc[high_indices, df.columns.get_loc('price_swing_high')] = True
        df.iloc[low_indices, df.columns.get_loc('price_swing_low')] = True
        
        # Find RSI swing points
        rsi_high_indices = argrelextrema(df['rsi'].values, np.greater, order=min_bars)[0]
        rsi_low_indices = argrelextrema(df['rsi'].values, np.less, order=min_bars)[0]
        
        df['rsi_swing_high'] = False
        df['rsi_swing_low'] = False
        df.iloc[rsi_high_indices, df.columns.get_loc('rsi_swing_high')] = True
        df.iloc[rsi_low_indices, df.columns.get_loc('rsi_swing_low')] = True
        
        return df

    @classmethod
    def _detect_divergences(cls, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Detect bullish and bearish divergences."""
        
        df['bullish_divergence'] = False
        df['bearish_divergence'] = False
        df['divergence_strength'] = 0.0
        
        for i in range(lookback, len(df)):
            current_window = df.iloc[i-lookback:i+1]
            
            # Check for bullish divergence (price lower low, RSI higher low)
            price_lows = current_window[current_window['price_swing_low']]
            rsi_lows = current_window[current_window['rsi_swing_low']]
            
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                recent_price_low = price_lows['low'].iloc[-1]
                prev_price_low = price_lows['low'].iloc[-2]
                recent_rsi_low = rsi_lows['rsi'].iloc[-1]
                prev_rsi_low = rsi_lows['rsi'].iloc[-2]
                
                # Bullish divergence: price makes lower low, RSI makes higher low
                if (recent_price_low < prev_price_low and 
                    recent_rsi_low > prev_rsi_low):
                    
                    strength = (recent_rsi_low - prev_rsi_low) / prev_rsi_low
                    df.iloc[i, df.columns.get_loc('bullish_divergence')] = True
                    df.iloc[i, df.columns.get_loc('divergence_strength')] = strength
            
            # Check for bearish divergence (price higher high, RSI lower high)
            price_highs = current_window[current_window['price_swing_high']]
            rsi_highs = current_window[current_window['rsi_swing_high']]
            
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                recent_price_high = price_highs['high'].iloc[-1]
                prev_price_high = price_highs['high'].iloc[-2]
                recent_rsi_high = rsi_highs['rsi'].iloc[-1]
                prev_rsi_high = rsi_highs['rsi'].iloc[-2]
                
                # Bearish divergence: price makes higher high, RSI makes lower high
                if (recent_price_high > prev_price_high and 
                    recent_rsi_high < prev_rsi_high):
                    
                    strength = abs(recent_rsi_high - prev_rsi_high) / prev_rsi_high
                    df.iloc[i, df.columns.get_loc('bearish_divergence')] = True
                    df.iloc[i, df.columns.get_loc('divergence_strength')] = strength
        
        return df

    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        """Detect bullish divergence entry signals."""
        
        # Required indicators check
        required = ['bullish_divergence', 'rsi', 'volume_ratio', 'atr', 'divergence_strength']
        for indicator in required:
            if indicator not in row or pd.isna(row[indicator]):
                return False, None, 0.0
        
        # Primary signal: Bullish divergence
        if not row['bullish_divergence']:
            return False, None, 0.0
        
        # Divergence strength filter
        if row['divergence_strength'] < self.min_divergence_strength:
            return False, None, 0.0
        
        # RSI oversold confirmation
        if row['rsi'] > self.rsi_oversold:
            return False, None, 0.0
        
        # Volume confirmation
        if self.require_volume_confirmation:
            if row['volume_ratio'] < self.min_volume_ratio:
                return False, None, 0.0
        
        # Trend confirmation (don't fight strong downtrends)
        if self.require_trend_confirmation:
            if 'trend_direction' in row and row['trend_direction'] < 0:
                # Only take bullish divergences in downtrends if RSI is very oversold
                if row['rsi'] > 25:
                    return False, None, 0.0
        
        # Calculate stop loss
        stop_loss = self.stop_loss_atr * row['atr'] / row['close']
        
        reason = f"Bullish Divergence (RSI:{row['rsi']:.1f}, Strength:{row['divergence_strength']:.2f})"
        return True, reason, stop_loss

    def sell_condition(self, position: Position, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """Determine exit conditions."""
        
        current_price = row['close']
        
        # ATR-based stops and targets
        if 'atr' in row and not pd.isna(row['atr']):
            atr_value = row['atr']
            take_profit_price = position.open_price * (1 + self.take_profit_atr * atr_value / position.open_price)
            stop_loss_price = position.open_price * (1 - self.stop_loss_atr * atr_value / position.open_price)
            
            # Take profit
            if current_price >= take_profit_price:
                return True, f"Take Profit (ATR {self.take_profit_atr}x)"
            
            # Stop loss
            if current_price <= stop_loss_price:
                return True, f"Stop Loss (ATR {self.stop_loss_atr}x)"
        
        # RSI overbought exit
        if 'rsi' in row and row['rsi'] >= self.rsi_overbought:
            return True, "RSI Overbought Exit"
        
        # Bearish divergence exit (take profits on counter-signal)
        if 'bearish_divergence' in row and row['bearish_divergence']:
            if row.get('divergence_strength', 0) > self.min_divergence_strength:
                return True, "Bearish Divergence Counter-Signal"
        
        # Trend reversal exit
        if 'trend_direction' in row and row['trend_direction'] < 0:
            # Exit if trend turns bearish and we're in profit
            profit = (current_price - position.open_price) / position.open_price
            if profit > 0.02:  # 2% profit
                return True, "Trend Reversal (Profit Secured)"
        
        return False, None

    def calculate_position_size(self, available_balance: float, current_price: float, row: pd.Series) -> float:
        """Calculate position size based on signal strength and volatility."""
        
        base_size = available_balance * self.base_position_size
        
        # Adjust size based on divergence strength
        if 'divergence_strength' in row and not pd.isna(row['divergence_strength']):
            strength_multiplier = min(2.0, 1 + row['divergence_strength'])
            base_size *= strength_multiplier
        
        # Adjust for volatility (reduce size in high volatility)
        if 'atr' in row and not pd.isna(row['atr']):
            volatility = row['atr'] / current_price
            if volatility > 0.04:  # High volatility
                base_size *= 0.7
            elif volatility < 0.02:  # Low volatility
                base_size *= 1.3
        
        # Ensure maximum position size
        max_size = available_balance * self.max_position_size
        return min(base_size, max_size)

    def __str__(self) -> str:
        params = self.get_strategy_params()
        key_params = {k: v for k, v in params.items() 
                     if k in ['rsi_period', 'divergence_lookback', 'min_divergence_strength']}
        return f"RSIDivergence({', '.join(f'{k}={v}' for k, v in key_params.items())})"