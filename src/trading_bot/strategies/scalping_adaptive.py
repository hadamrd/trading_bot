"""
Scalping-Optimized Adaptive Multi-Strategy System
File: src/trading_bot/strategies/adaptive_multi_scalping.py

MUCH more aggressive parameters for 5-minute scalping
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import ta
from datetime import datetime
from enum import Enum

from ..core.models import Position
from .base import BaseStrategy


class MarketRegime(Enum):
    """Market regime types for strategy switching"""
    TRENDING = "trending"       
    RANGING = "ranging"         
    BREAKOUT = "breakout"       


class VolatilityMood(Enum):
    """Volatility-based position sizing moods"""
    CALM = "calm"           
    NORMAL = "normal"       
    NERVOUS = "nervous"     
    PANIC = "panic"         


class ScalpingAdaptiveStrategy(BaseStrategy):
    """
    AGGRESSIVE scalping version - designed to find many trades
    """
    
    INDICATOR_PARAMS = {
        'ema_fast', 'ema_slow', 'bb_period', 'rsi_period', 
        'volume_period', 'atr_period', 'adx_period'
    }
    
    def _init_strategy(self, 
                      # SCALPING PARAMETERS - Much more aggressive!
                      
                      # Regime detection - more sensitive
                      lookback_period: int = 20,  # Shorter lookback
                      volatility_lookback: int = 50,  # Shorter volatility window
                      
                      # Strategy parameters
                      ema_fast: int = 5,   # FASTER EMAs for scalping
                      ema_slow: int = 13,  # FASTER EMAs
                      bb_period: int = 15, # Shorter BB period
                      rsi_period: int = 9, # Faster RSI
                      volume_period: int = 10, # Shorter volume period
                      atr_period: int = 10,    # Faster ATR
                      adx_period: int = 10,    # Faster ADX
                      
                      # Position sizing - smaller for scalping
                      base_position_size: float = 0.01,  # 1% per trade
                      
                      # RELAXED REGIME THRESHOLDS
                      trending_adx_threshold: float = 15,     # Much lower (was 25)
                      trending_strength_threshold: float = 0.003,  # Much lower (was 0.008)
                      breakout_volatility_threshold: float = 0.5,  # Lower (was 0.8)
                      breakout_momentum_threshold: float = 0.008,  # Lower (was 0.015)
                      
                      # RELAXED MEAN REVERSION - trigger more often
                      mr_bb_lower: float = 0.35,        # Higher (was 0.2) = easier to trigger
                      mr_bb_upper: float = 0.65,        # Lower (was 0.8) = easier to trigger  
                      mr_rsi_oversold: float = 45,      # Higher (was 35) = easier to trigger
                      mr_rsi_overbought: float = 55,    # Lower (was 70) = easier to trigger
                      
                      # RELAXED TREND FOLLOWING
                      trend_volume_min: float = 0.8,    # Lower (was 1.2) = easier volume req
                      
                      # RELAXED BREAKOUT 
                      breakout_volume_min: float = 1.2,  # Lower (was 2.0)
                      breakout_momentum_min: float = 0.002, # Much lower (was 0.005)
                      
                      **kwargs):
        
        # Store all parameters (same as before but with aggressive values)
        self.lookback_period = lookback_period
        self.volatility_lookback = volatility_lookback
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.bb_period = bb_period
        self.rsi_period = rsi_period
        self.volume_period = volume_period
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.base_position_size = base_position_size
        
        # Regime thresholds (aggressive)
        self.trending_adx_threshold = trending_adx_threshold
        self.trending_strength_threshold = trending_strength_threshold
        self.breakout_volatility_threshold = breakout_volatility_threshold
        self.breakout_momentum_threshold = breakout_momentum_threshold
        
        # Strategy-specific thresholds (aggressive)
        self.mr_bb_lower = mr_bb_lower
        self.mr_bb_upper = mr_bb_upper
        self.mr_rsi_oversold = mr_rsi_oversold
        self.mr_rsi_overbought = mr_rsi_overbought
        self.trend_volume_min = trend_volume_min
        self.breakout_volume_min = breakout_volume_min
        self.breakout_momentum_min = breakout_momentum_min
        
        # State tracking
        self.current_regime = MarketRegime.RANGING
        self.current_mood = VolatilityMood.NORMAL
        self.regime_history = []
        self.last_regime_update = -1
    
    @classmethod
    def calculate_indicators(cls, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate all technical indicators - same as before but with scalping periods"""
        df = df.copy()
        
        if len(df) < 30:  # Reduced minimum (was 50)
            return df
        
        try:
            # Core price indicators (faster periods)
            df['rsi'] = ta.momentum.rsi(df['close'], window=params.get('rsi_period', 9))
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=params.get('atr_period', 10)
            )
            
            # EMA trend indicators (faster)
            df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=params.get('ema_fast', 5))
            df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=params.get('ema_slow', 13))
            df['ema_diff'] = df['ema_fast'] - df['ema_slow']
            df['ema_diff_prev'] = df['ema_diff'].shift(1)
            
            # EMA crossover signals
            df['bullish_cross'] = (df['ema_diff'] > 0) & (df['ema_diff_prev'] <= 0)
            df['bearish_cross'] = (df['ema_diff'] < 0) & (df['ema_diff_prev'] >= 0)
            
            # Bollinger Bands (shorter period)
            bb = ta.volatility.BollingerBands(df['close'], window=params.get('bb_period', 15))
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume indicators (shorter period)
            df['volume_sma'] = df['volume'].rolling(window=params.get('volume_period', 10)).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Momentum indicators (faster)
            df['price_momentum'] = df['close'].pct_change(2)  # 2-period (was 3)
            df['price_momentum_5'] = df['close'].pct_change(3)  # 3-period (was 5)
            
            # Regime detection indicators (faster)
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=params.get('adx_period', 10))
            df['volatility'] = df['close'].rolling(window=10).std() / df['close']  # Shorter window
            df['volatility_percentile'] = df['volatility'].rolling(
                window=params.get('volatility_lookback', 50)
            ).rank(pct=True)
            
            # Trend strength (normalized by price)
            df['trend_strength'] = abs(df['ema_fast'] - df['ema_slow']) / df['close']
            
        except Exception as e:
            print(f"Error calculating scalping indicators: {e}")
        
        return df.ffill().fillna(0)
    
    def _detect_regime(self, row: pd.Series) -> MarketRegime:
        """Detect regime - more sensitive for scalping"""
        
        adx = row.get('adx', 15)
        volatility_percentile = row.get('volatility_percentile', 0.5)
        trend_strength = row.get('trend_strength', 0)
        price_momentum = abs(row.get('price_momentum_5', 0))
        
        # MUCH MORE LIBERAL regime classification
        
        # 1. High volatility + any momentum = BREAKOUT
        if (volatility_percentile > self.breakout_volatility_threshold and 
            price_momentum > self.breakout_momentum_threshold):
            return MarketRegime.BREAKOUT
        
        # 2. Any significant ADX + any trend = TRENDING  
        if (adx > self.trending_adx_threshold and 
            trend_strength > self.trending_strength_threshold):
            return MarketRegime.TRENDING
        
        # 3. Default to RANGING (mean reversion) - this will fire most often
        return MarketRegime.RANGING
    
    def _detect_volatility_mood(self, row: pd.Series) -> VolatilityMood:
        """Same volatility detection"""
        volatility_percentile = row.get('volatility_percentile', 0.5)
        
        if volatility_percentile > 0.9:
            return VolatilityMood.PANIC
        elif volatility_percentile > 0.7:
            return VolatilityMood.NERVOUS
        elif volatility_percentile < 0.3:
            return VolatilityMood.CALM
        else:
            return VolatilityMood.NORMAL
    
    def _get_volatility_multiplier(self, mood: VolatilityMood) -> float:
        """Same volatility multipliers"""
        multipliers = {
            VolatilityMood.PANIC: 0.3,
            VolatilityMood.NERVOUS: 0.6,
            VolatilityMood.NORMAL: 1.0,
            VolatilityMood.CALM: 1.4
        }
        return multipliers[mood]
    
    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        """AGGRESSIVE buy conditions - should fire much more often"""
        
        # Update regime and mood
        self.current_regime = self._detect_regime(row)
        self.current_mood = self._detect_volatility_mood(row)
        
        # Route to appropriate strategy
        if self.current_regime == MarketRegime.TRENDING:
            return self._trend_buy_condition(row, prev_row)
        elif self.current_regime == MarketRegime.RANGING:
            return self._mean_reversion_buy_condition(row, prev_row)
        elif self.current_regime == MarketRegime.BREAKOUT:
            return self._breakout_buy_condition(row, prev_row)
        
        return False, None, 0.0
    
    def _trend_buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        """AGGRESSIVE trend following - much easier to trigger"""
        
        # Much more relaxed conditions
        if (row.get('bullish_cross', False) and 
            row.get('volume_ratio', 0) > self.trend_volume_min):  # Removed ADX requirement
            
            return True, "[TRENDING] EMA Bullish Cross", 0.008  # Tighter stop for scalping
        
        # ADDITIONAL: Simple trend continuation (new for scalping)
        if (row.get('ema_diff', 0) > 0 and  # Fast EMA above slow
            row.get('close', 0) > row.get('ema_fast', 0) and  # Price above fast EMA
            row.get('rsi', 50) > 45 and  # Not oversold
            row.get('volume_ratio', 0) > 0.8):  # Any volume
            
            return True, "[TRENDING] Trend Continuation", 0.008
        
        return False, None, 0.0
    
    def _mean_reversion_buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        """AGGRESSIVE mean reversion - much easier to trigger"""
        
        # Much more relaxed conditions - this should fire often!
        if (row.get('bb_position', 0.5) < self.mr_bb_lower and 
            row.get('rsi', 50) < self.mr_rsi_oversold):
            
            return True, "[RANGING] Mean Reversion Oversold", 0.006  # Tight scalping stop
        
        # ADDITIONAL: RSI oversold alone (new for scalping)
        if row.get('rsi', 50) < 35:  # Very oversold
            return True, "[RANGING] RSI Oversold", 0.006
        
        # ADDITIONAL: BB lower band touch (new for scalping)
        if row.get('bb_position', 0.5) < 0.25:  # Near lower band
            return True, "[RANGING] BB Support", 0.006
        
        return False, None, 0.0
    
    def _breakout_buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        """AGGRESSIVE breakout - much easier to trigger"""
        
        # Much more relaxed breakout conditions
        if (row.get('volume_ratio', 0) > self.breakout_volume_min and 
            row.get('price_momentum', 0) > self.breakout_momentum_min):
            
            return True, "[BREAKOUT] Volume Momentum", 0.010  # Wider stop for breakouts
        
        # ADDITIONAL: Any significant price move (new for scalping)
        if abs(row.get('price_momentum', 0)) > 0.005:  # 0.5% move
            return True, "[BREAKOUT] Price Move", 0.010
        
        return False, None, 0.0
    
    # Keep all the sell conditions the same - they're fine for scalping
    def sell_condition(self, position: Position, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """Quick scalping exits"""
        
        current_price = row['close']
        
        # TIGHTER stops for scalping
        if self.current_regime == MarketRegime.RANGING:
            stop_loss_pct = 0.004  # 0.4% (was 0.8%)
        elif self.current_regime == MarketRegime.TRENDING:
            stop_loss_pct = 0.006  # 0.6% (was 1.2%)
        else:  # BREAKOUT
            stop_loss_pct = 0.008  # 0.8% (was 1.8%)
        
        if current_price <= position.open_price * (1 - stop_loss_pct):
            return True, f"Stop Loss ({self.current_regime.value})"
        
        # QUICK profit taking for scalping
        if self.current_regime == MarketRegime.TRENDING:
            if current_price >= position.open_price * 1.012:  # 1.2% (was 2.5%)
                return True, "Quick Trend Profit"
        elif self.current_regime == MarketRegime.RANGING:
            if current_price >= position.open_price * 1.008:  # 0.8% profit
                return True, "Quick Mean Reversion"
        elif self.current_regime == MarketRegime.BREAKOUT:
            if current_price >= position.open_price * 1.015:  # 1.5% (was 2%)
                return True, "Quick Breakout Profit"
        
        return False, None
    
    def calculate_position_size(self, available_balance: float, current_price: float, row: pd.Series) -> float:
        """Smaller positions for scalping"""
        
        base_size = available_balance * self.base_position_size  # 1% base
        mood_multiplier = self._get_volatility_multiplier(self.current_mood)
        adjusted_size = base_size * mood_multiplier
        
        return min(adjusted_size, available_balance * 0.03)  # Max 3% per scalping trade
    
    def __str__(self) -> str:
        return f"ScalpingAdaptive(regime={self.current_regime.value}, mood={self.current_mood.value})"