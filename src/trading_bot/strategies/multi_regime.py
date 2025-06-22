"""
Multi-Regime Strategy: Switches between mean reversion and momentum based on volatility.
Research-backed approach for crypto markets.
"""

from typing import Tuple, Optional, Dict, Any, Set
import pandas as pd
import ta
import numpy as np

from ..strategies.base import BaseStrategy
from ..core.models import Position


class MultiRegimeStrategy(BaseStrategy):
    """
    Multi-Regime Strategy that adapts to market conditions.
    
    Based on research showing:
    - Low volatility periods favor mean reversion
    - High volatility periods favor momentum
    - Crypto exhibits both effects on different timeframes
    """
    
    INDICATOR_PARAMS: Set[str] = {
        'rsi_period', 'bb_period', 'bb_std', 'atr_period', 
        'volatility_lookback', 'ema_fast', 'ema_slow'
    }
    
    @classmethod
    def calculate_indicators(cls, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate all indicators for regime detection and trading."""
        df = df.copy()
        
        # Extract parameters
        rsi_period = params.get('rsi_period', 14)
        bb_period = params.get('bb_period', 20)
        bb_std = params.get('bb_std', 2.0)
        atr_period = params.get('atr_period', 14)
        volatility_lookback = params.get('volatility_lookback', 50)
        ema_fast = params.get('ema_fast', 9)
        ema_slow = params.get('ema_slow', 21)
        
        if len(df) < max(bb_period, atr_period, volatility_lookback):
            return df
        
        try:
            # Core indicators
            df['rsi'] = ta.momentum.rsi(df['close'], window=rsi_period)
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=atr_period
            )
            
            # Bollinger Bands for mean reversion
            bb = ta.volatility.BollingerBands(df['close'], window=bb_period, window_dev=bb_std)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # EMAs for momentum
            df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=ema_fast)
            df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=ema_slow)
            df['ema_diff'] = df['ema_fast'] - df['ema_slow']
            
            # Volatility regime detection
            df['volatility_percentile'] = df['atr'].rolling(
                window=volatility_lookback
            ).rank(pct=True)
            
            # Price position within BB
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Momentum signals
            df['momentum_signal'] = (
                (df['ema_diff'] > df['ema_diff'].shift(1)) & 
                (df['close'] > df['ema_fast'])
            ).astype(int)
            
            # Mean reversion signals  
            df['mean_reversion_signal'] = (
                ((df['rsi'] < 30) & (df['bb_position'] < 0.2)) |
                ((df['rsi'] > 70) & (df['bb_position'] > 0.8))
            ).astype(int)
            
            # Regime classification
            df['volatility_regime'] = np.where(
                df['volatility_percentile'] > 0.7, 'high_vol',
                np.where(df['volatility_percentile'] < 0.3, 'low_vol', 'medium_vol')
            )
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
        
        return df.fillna(method='ffill').fillna(0)

    def _init_strategy(self,
                      # Regime detection parameters
                      volatility_threshold_high: float = 0.7,
                      volatility_threshold_low: float = 0.3,
                      volatility_lookback: int = 50,
                      
                      # Mean reversion parameters
                      rsi_period: int = 14,
                      rsi_oversold: int = 30,
                      rsi_overbought: int = 70,
                      bb_period: int = 20,
                      bb_std: float = 2.0,
                      
                      # Momentum parameters  
                      ema_fast: int = 9,
                      ema_slow: int = 21,
                      
                      # Risk management
                      atr_period: int = 14,
                      stop_loss_atr: float = 2.0,
                      take_profit_atr: float = 3.0,
                      position_size_pct: float = 0.02):
        """Initialize strategy parameters."""
        
        # Regime detection
        self.volatility_threshold_high = volatility_threshold_high
        self.volatility_threshold_low = volatility_threshold_low
        self.volatility_lookback = volatility_lookback
        
        # Mean reversion
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std
        
        # Momentum
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        
        # Risk management
        self.atr_period = atr_period
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.position_size_pct = position_size_pct

    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        """Determine buy signal based on current volatility regime."""
        
        # Check for required indicators
        required = ['volatility_regime', 'rsi', 'bb_position', 'ema_diff', 'volume_ratio', 'atr']
        for indicator in required:
            if indicator not in row or pd.isna(row[indicator]):
                return False, None, 0.0
        
        regime = row['volatility_regime']
        atr_stop = self.stop_loss_atr * row['atr'] / row['close']
        
        # Low volatility: Mean reversion strategy
        if regime == 'low_vol':
            oversold_condition = (
                row['rsi'] < self.rsi_oversold and
                row['bb_position'] < 0.2 and
                row['volume_ratio'] > 1.2
            )
            
            if oversold_condition:
                return True, "Low Vol Mean Reversion", atr_stop
        
        # High volatility: Momentum strategy
        elif regime == 'high_vol':
            momentum_condition = (
                row['ema_diff'] > 0 and
                row['ema_diff'] > prev_row['ema_diff'] and
                row['rsi'] > 45 and  # Avoid oversold in momentum
                row['volume_ratio'] > 1.5
            )
            
            if momentum_condition:
                return True, "High Vol Momentum", atr_stop
        
        # Medium volatility: Mixed approach
        else:
            # Weaker signals in both directions
            weak_oversold = (
                row['rsi'] < 35 and
                row['bb_position'] < 0.3 and
                row['volume_ratio'] > 1.1
            )
            
            weak_momentum = (
                row['ema_diff'] > 0 and
                prev_row['ema_diff'] < 0 and  # Golden cross
                row['volume_ratio'] > 1.3
            )
            
            if weak_oversold:
                return True, "Medium Vol Reversion", atr_stop
            elif weak_momentum:
                return True, "Medium Vol Momentum", atr_stop
        
        return False, None, 0.0

    def sell_condition(self, position: Position, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """Determine sell signal based on position type and current regime."""
        
        if 'atr' not in row or pd.isna(row['atr']):
            return False, None
        
        current_price = row['close']
        
        # ATR-based stops and targets
        atr_value = row['atr']
        take_profit_price = position.open_price * (1 + self.take_profit_atr * atr_value / position.open_price)
        stop_loss_price = position.open_price * (1 - self.stop_loss_atr * atr_value / position.open_price)
        
        # Take profit
        if current_price >= take_profit_price:
            return True, "Take Profit (ATR)"
        
        # Stop loss
        if current_price <= stop_loss_price:
            return True, "Stop Loss (ATR)"
        
        # Strategy-specific exits
        if 'volatility_regime' in row and not pd.isna(row['volatility_regime']):
            regime = row['volatility_regime']
            
            # Mean reversion exits
            if "Reversion" in position.buy_reason:
                if 'rsi' in row and row['rsi'] > self.rsi_overbought:
                    return True, "RSI Overbought Exit"
                if 'bb_position' in row and row['bb_position'] > 0.8:
                    return True, "BB Upper Exit"
            
            # Momentum exits
            elif "Momentum" in position.buy_reason:
                if 'ema_diff' in row and row['ema_diff'] < 0:  # EMA crossover down
                    return True, "EMA Bearish Cross"
        
        return False, None

    def calculate_position_size(self, available_balance: float, current_price: float, row: pd.Series) -> float:
        """Calculate position size with volatility adjustment."""
        
        base_size = available_balance * self.position_size_pct
        
        # Reduce size in high volatility regimes
        if 'volatility_regime' in row:
            regime = row['volatility_regime']
            if regime == 'high_vol':
                base_size *= 0.7  # Reduce size by 30% in high vol
            elif regime == 'low_vol':
                base_size *= 1.2  # Increase size by 20% in low vol
        
        return min(base_size, available_balance * 0.1)  # Max 10% per trade