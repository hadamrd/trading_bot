"""
Adaptive Multi-Strategy System
File: src/trading_bot/strategies/adaptive_multi.py

Switches between 3 strategies based on market regime with volatility-based position sizing
Optimized for 5-minute timeframe trading
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
    TRENDING = "trending"       # Strong directional movement - use trend following
    RANGING = "ranging"         # Sideways/choppy market - use mean reversion  
    BREAKOUT = "breakout"       # High volatility breakout - use momentum


class VolatilityMood(Enum):
    """Volatility-based position sizing moods"""
    CALM = "calm"           # Low volatility - larger positions (140%)
    NORMAL = "normal"       # Average volatility - normal positions (100%)
    NERVOUS = "nervous"     # High volatility - smaller positions (60%)
    PANIC = "panic"         # Extreme volatility - minimal positions (30%)


class AdaptiveMultiStrategy(BaseStrategy):
    """
    Adaptive strategy that switches between trend/mean-reversion/breakout
    based on market regime detection with volatility-adjusted position sizing
    """
    
    INDICATOR_PARAMS = {
        'ema_fast', 'ema_slow', 'bb_period', 'rsi_period', 
        'volume_period', 'atr_period', 'adx_period'
    }
    
    def _init_strategy(self, 
                      # Regime detection
                      lookback_period: int = 50,
                      volatility_lookback: int = 200,
                      
                      # Strategy parameters
                      ema_fast: int = 9,
                      ema_slow: int = 21,
                      bb_period: int = 20,
                      rsi_period: int = 14,
                      volume_period: int = 20,
                      atr_period: int = 14,
                      adx_period: int = 14,
                      
                      # Position sizing
                      base_position_size: float = 0.02,
                      
                      # Regime thresholds
                      trending_adx_threshold: float = 25,
                      trending_strength_threshold: float = 0.008,
                      breakout_volatility_threshold: float = 0.8,
                      breakout_momentum_threshold: float = 0.015,
                      
                      # Mean reversion thresholds
                      mr_bb_lower: float = 0.2,
                      mr_bb_upper: float = 0.8,
                      mr_rsi_oversold: float = 35,
                      mr_rsi_overbought: float = 70,
                      
                      # Trend following thresholds
                      trend_volume_min: float = 1.2,
                      
                      # Breakout thresholds
                      breakout_volume_min: float = 2.0,
                      breakout_momentum_min: float = 0.005,
                      
                      **kwargs):
        
        # Store all parameters
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
        
        # Regime thresholds
        self.trending_adx_threshold = trending_adx_threshold
        self.trending_strength_threshold = trending_strength_threshold
        self.breakout_volatility_threshold = breakout_volatility_threshold
        self.breakout_momentum_threshold = breakout_momentum_threshold
        
        # Strategy-specific thresholds
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
        """Calculate all technical indicators needed for regime detection and trading"""
        df = df.copy()
        
        if len(df) < 50:
            return df
        
        try:
            # Core price indicators
            df['rsi'] = ta.momentum.rsi(df['close'], window=params.get('rsi_period', 14))
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=params.get('atr_period', 14)
            )
            
            # EMA trend indicators
            df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=params.get('ema_fast', 9))
            df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=params.get('ema_slow', 21))
            df['ema_diff'] = df['ema_fast'] - df['ema_slow']
            df['ema_diff_prev'] = df['ema_diff'].shift(1)
            
            # EMA crossover signals
            df['bullish_cross'] = (df['ema_diff'] > 0) & (df['ema_diff_prev'] <= 0)
            df['bearish_cross'] = (df['ema_diff'] < 0) & (df['ema_diff_prev'] >= 0)
            
            # Bollinger Bands for mean reversion
            bb = ta.volatility.BollingerBands(df['close'], window=params.get('bb_period', 20))
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=params.get('volume_period', 20)).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Momentum indicators
            df['price_momentum'] = df['close'].pct_change(3)  # 3-period momentum
            df['price_momentum_5'] = df['close'].pct_change(5)  # 5-period momentum
            
            # Regime detection indicators
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=params.get('adx_period', 14))
            df['volatility'] = df['close'].rolling(window=20).std() / df['close']  # Relative volatility
            df['volatility_percentile'] = df['volatility'].rolling(
                window=params.get('volatility_lookback', 200)
            ).rank(pct=True)
            
            # Trend strength (normalized by price)
            df['trend_strength'] = (df['ema_fast'] - df['ema_slow']) / df['close']
            
        except Exception as e:
            print(f"Error calculating adaptive indicators: {e}")
        
        return df.ffill().fillna(0)
    
    def _detect_regime(self, row: pd.Series) -> MarketRegime:
        """Detect current market regime based on technical indicators"""
        
        # Extract regime indicators
        adx = row.get('adx', 25)
        volatility_percentile = row.get('volatility_percentile', 0.5)
        trend_strength = abs(row.get('trend_strength', 0))
        price_momentum = abs(row.get('price_momentum_5', 0))
        
        # Regime classification logic
        
        # 1. High volatility + strong momentum = BREAKOUT
        if (volatility_percentile > self.breakout_volatility_threshold and 
            price_momentum > self.breakout_momentum_threshold):
            return MarketRegime.BREAKOUT
        
        # 2. Strong ADX + significant trend = TRENDING
        if (adx > self.trending_adx_threshold and 
            trend_strength > self.trending_strength_threshold):
            return MarketRegime.TRENDING
        
        # 3. Default to RANGING (mean reversion)
        return MarketRegime.RANGING
    
    def _detect_volatility_mood(self, row: pd.Series) -> VolatilityMood:
        """Detect volatility mood for position sizing"""
        
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
        """Get position size multiplier based on volatility mood"""
        
        multipliers = {
            VolatilityMood.PANIC: 0.3,      # 30% size - extreme volatility
            VolatilityMood.NERVOUS: 0.6,    # 60% size - high volatility
            VolatilityMood.NORMAL: 1.0,     # 100% size - normal volatility
            VolatilityMood.CALM: 1.4        # 140% size - low volatility
        }
        
        return multipliers[mood]
    
    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        """Main buy condition logic that routes to regime-specific strategies"""
        
        # Update regime and mood
        self.current_regime = self._detect_regime(row)
        self.current_mood = self._detect_volatility_mood(row)
        
        # Route to appropriate strategy based on regime
        if self.current_regime == MarketRegime.TRENDING:
            return self._trend_buy_condition(row, prev_row)
        elif self.current_regime == MarketRegime.RANGING:
            return self._mean_reversion_buy_condition(row, prev_row)
        elif self.current_regime == MarketRegime.BREAKOUT:
            return self._breakout_buy_condition(row, prev_row)
        
        return False, None, 0.0
    
    def _trend_buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        """Trend following strategy buy condition"""
        
        # EMA bullish crossover with volume confirmation
        if (row.get('bullish_cross', False) and 
            row.get('volume_ratio', 0) > self.trend_volume_min and
            row.get('adx', 0) > self.trending_adx_threshold):
            
            return True, "[TRENDING] EMA Bullish Cross", 0.015
        
        return False, None, 0.0
    
    def _mean_reversion_buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        """Mean reversion strategy buy condition"""
        
        # Oversold conditions: BB lower band + RSI oversold
        if (row.get('bb_position', 0.5) < self.mr_bb_lower and 
            row.get('rsi', 50) < self.mr_rsi_oversold):
            
            return True, "[RANGING] Mean Reversion Oversold", 0.012
        
        return False, None, 0.0
    
    def _breakout_buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        """Breakout momentum strategy buy condition"""
        
        # High volume breakout with momentum
        if (row.get('volume_ratio', 0) > self.breakout_volume_min and 
            row.get('price_momentum', 0) > self.breakout_momentum_min and
            row.get('volatility_percentile', 0) > 0.7):
            
            return True, "[BREAKOUT] Volume Momentum", 0.018
        
        return False, None, 0.0
    
    def sell_condition(self, position: Position, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """Sell condition logic adapted to current regime"""
        
        current_price = row['close']
        
        # Universal stops (regime-independent)
        
        # Stop loss (tighter for ranging, wider for trending/breakout)
        if self.current_regime == MarketRegime.RANGING:
            stop_loss_pct = 0.008  # 0.8% for mean reversion
        elif self.current_regime == MarketRegime.TRENDING:
            stop_loss_pct = 0.012  # 1.2% for trends
        else:  # BREAKOUT
            stop_loss_pct = 0.018  # 1.8% for breakouts
        
        if current_price <= position.open_price * (1 - stop_loss_pct):
            return True, f"Stop Loss ({self.current_regime.value})"
        
        # Regime-specific exits
        if self.current_regime == MarketRegime.TRENDING:
            return self._trend_sell_condition(position, row)
        elif self.current_regime == MarketRegime.RANGING:
            return self._mean_reversion_sell_condition(position, row)
        elif self.current_regime == MarketRegime.BREAKOUT:
            return self._breakout_sell_condition(position, row)
        
        return False, None
    
    def _trend_sell_condition(self, position: Position, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """Trend following exit conditions"""
        
        current_price = row['close']
        
        # Take profit for trend
        if current_price >= position.open_price * 1.025:  # 2.5%
            return True, "Trend Take Profit"
        
        # Bearish cross exit
        if row.get('bearish_cross', False):
            return True, "Trend Bearish Cross"
        
        return False, None
    
    def _mean_reversion_sell_condition(self, position: Position, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """Mean reversion exit conditions"""
        
        # Exit when reaching upper BB or RSI overbought
        if row.get('bb_position', 0.5) > self.mr_bb_upper:
            return True, "Mean Reversion Target"
        
        if row.get('rsi', 50) > self.mr_rsi_overbought:
            return True, "RSI Overbought"
        
        return False, None
    
    def _breakout_sell_condition(self, position: Position, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """Breakout momentum exit conditions"""
        
        current_price = row['close']
        
        # Quick profit for breakouts
        if current_price >= position.open_price * 1.02:  # 2%
            return True, "Breakout Quick Profit"
        
        # Exit if momentum dies
        if row.get('volume_ratio', 0) < 1.0 and row.get('price_momentum', 0) < 0:
            return True, "Momentum Died"
        
        return False, None
    
    def calculate_position_size(self, available_balance: float, current_price: float, row: pd.Series) -> float:
        """Calculate position size with volatility mood adjustment"""
        
        # Base position size
        base_size = available_balance * self.base_position_size
        
        # Apply volatility mood multiplier
        mood_multiplier = self._get_volatility_multiplier(self.current_mood)
        adjusted_size = base_size * mood_multiplier
        
        # Regime-specific adjustments
        if self.current_regime == MarketRegime.BREAKOUT:
            adjusted_size *= 0.8  # Reduce size for volatile breakouts
        elif self.current_regime == MarketRegime.TRENDING:
            adjusted_size *= 1.1  # Slightly larger for trends
        
        return min(adjusted_size, available_balance * 0.08)  # Max 8% per trade
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get current strategy state for monitoring"""
        return {
            'current_regime': self.current_regime.value,
            'current_mood': self.current_mood.value,
            'volatility_multiplier': self._get_volatility_multiplier(self.current_mood),
            'regime_changes': len(set(h.get('regime', 'unknown') for h in self.regime_history[-20:])),
            'mood_stability': len(set(h.get('mood', 'unknown') for h in self.regime_history[-10:]))
        }
    
    def __str__(self) -> str:
        return f"AdaptiveMulti(regime={self.current_regime.value}, mood={self.current_mood.value})"