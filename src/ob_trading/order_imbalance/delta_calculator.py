#!/usr/bin/env python3
"""
Delta Calculator Module - Volume Delta Analysis for Order Flow
"""

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import statistics


@dataclass
class TradeData:
    """Individual trade data structure"""
    price: float
    quantity: float
    timestamp: int
    is_buyer_maker: bool  # True = sell trade, False = buy trade
    trade_id: str
    
    @property
    def volume_usd(self) -> float:
        return self.price * self.quantity
    
    @property
    def is_buy_trade(self) -> bool:
        return not self.is_buyer_maker
    
    @property
    def is_sell_trade(self) -> bool:
        return self.is_buyer_maker


@dataclass
class DeltaSignal:
    """Delta analysis signal output"""
    timestamp: datetime
    signal_type: str  # "STRONG_BUY_DELTA", "MODERATE_BUY_DELTA", etc.
    current_delta: float
    cumulative_delta: float
    delta_strength: float  # 0-1
    acceleration: float
    confidence: float  # 0-1
    divergence_detected: bool
    reason: str


@dataclass
class DeltaMetrics:
    """Current delta state metrics"""
    cumulative_delta: float
    current_period_delta: float
    buy_volume: float
    sell_volume: float
    total_volume: float
    buy_percentage: float
    sell_percentage: float
    delta_acceleration: float
    recent_delta_trend: str  # "BULLISH", "BEARISH", "NEUTRAL"
    divergence_status: bool
    signal_strength: float


class DeltaCalculator:
    """
    Professional-grade volume delta calculator
    
    Tracks buy/sell volume imbalances to detect institutional flow
    """
    
    def __init__(self,
                 # Data retention
                 trade_history_size: int = 1000,
                 minute_history_size: int = 60,
                 
                 # Signal thresholds
                 strong_delta_threshold: float = 0.7,
                 moderate_delta_threshold: float = 0.4,
                 acceleration_threshold: float = 0.3,
                 divergence_lookback: int = 20,
                 
                 # Signal filtering
                 min_confidence: float = 0.6,
                 min_volume_threshold: float = 1000):  # Minimum USD volume
        
        # Configuration
        self.trade_history_size = trade_history_size
        self.minute_history_size = minute_history_size
        self.strong_delta_threshold = strong_delta_threshold
        self.moderate_delta_threshold = moderate_delta_threshold
        self.acceleration_threshold = acceleration_threshold
        self.divergence_lookback = divergence_lookback
        self.min_confidence = min_confidence
        self.min_volume_threshold = min_volume_threshold
        
        # Core delta tracking
        self.cumulative_delta = 0.0
        self.session_start_time = None
        
        # Trade history
        self.trade_history = deque(maxlen=trade_history_size)
        self.delta_history = deque(maxlen=trade_history_size)
        self.price_history = deque(maxlen=trade_history_size)
        
        # Time-based aggregations
        self.current_minute_delta = 0.0
        self.current_minute_start = None
        self.minute_deltas = deque(maxlen=minute_history_size)
        
        # Volume tracking
        self.session_buy_volume = 0.0
        self.session_sell_volume = 0.0
        self.total_session_volume = 0.0
        
        # Recent period tracking (rolling window)
        self.recent_buy_volume = 0.0
        self.recent_sell_volume = 0.0
        self.recent_period_trades = deque(maxlen=100)  # Last 100 trades
        
        # Signal state
        self.last_signal_time = None
        self.signal_cooldown_seconds = 30
        
        # Performance tracking
        self.total_trades_processed = 0
        self.signals_generated = 0
    
    def process_trade(self, trade: TradeData) -> Tuple[Optional[DeltaSignal], DeltaMetrics]:
        """
        Process individual trade and return signal + current metrics
        
        Returns:
            (DeltaSignal or None, DeltaMetrics)
        """
        
        # Initialize session if first trade
        if self.session_start_time is None:
            self.session_start_time = datetime.fromtimestamp(trade.timestamp / 1000)
        
        # Calculate trade delta
        if trade.is_buy_trade:
            trade_delta = trade.volume_usd
            self.session_buy_volume += trade.volume_usd
            self.recent_buy_volume += trade.volume_usd
        else:
            trade_delta = -trade.volume_usd
            self.session_sell_volume += trade.volume_usd
            self.recent_sell_volume += trade.volume_usd
        
        # Update cumulative delta
        self.cumulative_delta += trade_delta
        self.total_session_volume += trade.volume_usd
        
        # Store in history
        self.trade_history.append(trade)
        self.delta_history.append(trade_delta)
        self.price_history.append(trade.price)
        self.recent_period_trades.append(trade)
        
        # Update time-based aggregations
        self._update_minute_aggregation(trade_delta, trade.timestamp)
        
        # Update recent period (remove old trades)
        self._update_recent_period()
        
        # Calculate current metrics
        metrics = self._calculate_metrics()
        
        # Generate signal if conditions met
        signal = self._generate_signal(trade)
        
        self.total_trades_processed += 1
        
        return signal, metrics
    
    def _update_minute_aggregation(self, trade_delta: float, timestamp: int):
        """Update minute-based delta aggregation"""
        current_minute = timestamp // 60000  # Convert to minute
        
        if self.current_minute_start != current_minute:
            # New minute started
            if self.current_minute_start is not None:
                self.minute_deltas.append(self.current_minute_delta)
            
            self.current_minute_delta = trade_delta
            self.current_minute_start = current_minute
        else:
            # Same minute, accumulate
            self.current_minute_delta += trade_delta
    
    def _update_recent_period(self):
        """Update recent period calculations (rolling window)"""
        # Recalculate recent volumes from actual trades in window
        self.recent_buy_volume = sum(
            trade.volume_usd for trade in self.recent_period_trades 
            if trade.is_buy_trade
        )
        self.recent_sell_volume = sum(
            trade.volume_usd for trade in self.recent_period_trades 
            if trade.is_sell_trade
        )
    
    def _calculate_metrics(self) -> DeltaMetrics:
        """Calculate current delta metrics"""
        
        # Basic volume metrics
        total_volume = self.recent_buy_volume + self.recent_sell_volume
        buy_pct = (self.recent_buy_volume / total_volume * 100) if total_volume > 0 else 50
        sell_pct = 100 - buy_pct
        
        # Delta acceleration (change in delta momentum)
        acceleration = self._calculate_delta_acceleration()
        
        # Recent trend analysis
        trend = self._analyze_recent_trend()
        
        # Divergence detection
        divergence = self._detect_price_delta_divergence()
        
        # Overall signal strength
        signal_strength = self._calculate_current_signal_strength()
        
        return DeltaMetrics(
            cumulative_delta=self.cumulative_delta,
            current_period_delta=self.current_minute_delta,
            buy_volume=self.recent_buy_volume,
            sell_volume=self.recent_sell_volume,
            total_volume=total_volume,
            buy_percentage=buy_pct,
            sell_percentage=sell_pct,
            delta_acceleration=acceleration,
            recent_delta_trend=trend,
            divergence_status=divergence,
            signal_strength=signal_strength
        )
    
    def _calculate_delta_acceleration(self) -> float:
        """Calculate delta acceleration (rate of change)"""
        if len(self.minute_deltas) < 3:
            return 0.0
        
        recent_deltas = list(self.minute_deltas)[-3:]
        
        # Simple acceleration: difference between recent changes
        if len(recent_deltas) >= 3:
            recent_change = recent_deltas[-1] - recent_deltas[-2]
            previous_change = recent_deltas[-2] - recent_deltas[-3]
            return recent_change - previous_change
        
        return 0.0
    
    def _analyze_recent_trend(self) -> str:
        """Analyze recent delta trend direction"""
        if len(self.minute_deltas) < 5:
            return "NEUTRAL"
        
        recent_deltas = list(self.minute_deltas)[-5:]
        
        # Calculate trend strength
        positive_minutes = sum(1 for delta in recent_deltas if delta > 0)
        negative_minutes = sum(1 for delta in recent_deltas if delta < 0)
        
        if positive_minutes >= 4:
            return "BULLISH"
        elif negative_minutes >= 4:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _detect_price_delta_divergence(self) -> bool:
        """Detect price/delta divergence patterns"""
        if len(self.delta_history) < self.divergence_lookback:
            return False
        
        # Get recent data
        recent_deltas = list(self.delta_history)[-self.divergence_lookback:]
        recent_prices = list(self.price_history)[-self.divergence_lookback:]
        
        if not recent_deltas or not recent_prices:
            return False
        
        # Calculate simple trends
        delta_sum_early = sum(recent_deltas[:self.divergence_lookback//2])
        delta_sum_recent = sum(recent_deltas[self.divergence_lookback//2:])
        delta_trend = delta_sum_recent - delta_sum_early
        
        price_early = recent_prices[0]
        price_recent = recent_prices[-1]
        price_trend = price_recent - price_early
        
        # Check for significant divergence
        if abs(delta_trend) > 1000 and abs(price_trend) > 0.01:  # Minimum thresholds
            if (delta_trend > 0 and price_trend < 0) or (delta_trend < 0 and price_trend > 0):
                return True
        
        return False
    
    def _calculate_current_signal_strength(self) -> float:
        """Calculate overall signal strength (0-1)"""
        if not self.recent_period_trades:
            return 0.0
        
        # Base strength on recent delta imbalance
        total_recent = self.recent_buy_volume + self.recent_sell_volume
        if total_recent == 0:
            return 0.0
        
        imbalance = abs(self.recent_buy_volume - self.recent_sell_volume) / total_recent
        
        # Boost strength based on acceleration
        acceleration_boost = min(abs(self._calculate_delta_acceleration()) / 10000, 0.3)
        
        # Boost for trend consistency
        trend_boost = 0.2 if self._analyze_recent_trend() != "NEUTRAL" else 0.0
        
        return min(imbalance + acceleration_boost + trend_boost, 1.0)
    
    def _generate_signal(self, current_trade: TradeData) -> Optional[DeltaSignal]:
        """Generate delta signal based on current conditions"""
        
        # Check cooldown
        if self._is_in_cooldown():
            return None
        
        # Need minimum data
        if len(self.delta_history) < 20:
            return None
        
        # Check minimum volume threshold
        if self.recent_buy_volume + self.recent_sell_volume < self.min_volume_threshold:
            return None
        
        # Calculate signal metrics
        signal_strength = self._calculate_current_signal_strength()
        confidence = self._calculate_signal_confidence(signal_strength)
        
        # Check confidence threshold
        if confidence < self.min_confidence:
            return None
        
        # Determine signal type and direction
        signal_type, reason = self._determine_signal_type(signal_strength)
        
        if signal_type == "NEUTRAL":
            return None
        
        # Create signal
        signal = DeltaSignal(
            timestamp=datetime.fromtimestamp(current_trade.timestamp / 1000),
            signal_type=signal_type,
            current_delta=self.delta_history[-1] if self.delta_history else 0,
            cumulative_delta=self.cumulative_delta,
            delta_strength=signal_strength,
            acceleration=self._calculate_delta_acceleration(),
            confidence=confidence,
            divergence_detected=self._detect_price_delta_divergence(),
            reason=reason
        )
        
        self.last_signal_time = datetime.fromtimestamp(current_trade.timestamp / 1000)
        self.signals_generated += 1
        
        return signal
    
    def _is_in_cooldown(self) -> bool:
        """Check if we're in signal cooldown period"""
        if self.last_signal_time is None:
            return False
        
        time_since = datetime.now() - self.last_signal_time
        return time_since.total_seconds() < self.signal_cooldown_seconds
    
    def _calculate_signal_confidence(self, signal_strength: float) -> float:
        """Calculate confidence in the signal"""
        base_confidence = signal_strength
        
        # Boost confidence for volume
        volume_boost = min((self.recent_buy_volume + self.recent_sell_volume) / 50000, 0.3)
        
        # Boost confidence for divergence
        divergence_boost = 0.2 if self._detect_price_delta_divergence() else 0.0
        
        # Boost confidence for trend consistency
        trend_boost = 0.1 if self._analyze_recent_trend() != "NEUTRAL" else 0.0
        
        return min(base_confidence + volume_boost + divergence_boost + trend_boost, 1.0)
    
    def _determine_signal_type(self, signal_strength: float) -> Tuple[str, str]:
        """Determine signal type and generate reason"""
        
        total_recent = self.recent_buy_volume + self.recent_sell_volume
        buy_ratio = self.recent_buy_volume / total_recent if total_recent > 0 else 0.5
        
        # Strong signals
        if signal_strength >= self.strong_delta_threshold:
            if buy_ratio > 0.65:
                return "STRONG_BUY_DELTA", f"Strong buying pressure: {buy_ratio:.1%} buy volume"
            elif buy_ratio < 0.35:
                return "STRONG_SELL_DELTA", f"Strong selling pressure: {(1-buy_ratio):.1%} sell volume"
        
        # Moderate signals
        elif signal_strength >= self.moderate_delta_threshold:
            if buy_ratio > 0.6:
                return "MODERATE_BUY_DELTA", f"Moderate buying pressure: {buy_ratio:.1%} buy volume"
            elif buy_ratio < 0.4:
                return "MODERATE_SELL_DELTA", f"Moderate selling pressure: {(1-buy_ratio):.1%} sell volume"
        
        return "NEUTRAL", "No significant delta imbalance"
    
    def get_session_statistics(self) -> Dict:
        """Get comprehensive session statistics"""
        return {
            # Volume statistics
            "total_volume": self.total_session_volume,
            "buy_volume": self.session_buy_volume,
            "sell_volume": self.session_sell_volume,
            "buy_percentage": (self.session_buy_volume / self.total_session_volume * 100) if self.total_session_volume > 0 else 50,
            
            # Delta statistics
            "cumulative_delta": self.cumulative_delta,
            "current_minute_delta": self.current_minute_delta,
            "delta_trend": self._analyze_recent_trend(),
            
            # Signal statistics
            "total_trades_processed": self.total_trades_processed,
            "signals_generated": self.signals_generated,
            "signal_rate": (self.signals_generated / self.total_trades_processed * 100) if self.total_trades_processed > 0 else 0,
            
            # Current state
            "signal_strength": self._calculate_current_signal_strength(),
            "divergence_detected": self._detect_price_delta_divergence(),
            "in_cooldown": self._is_in_cooldown(),
            
            # Configuration
            "session_duration": (datetime.now() - self.session_start_time).total_seconds() / 60 if self.session_start_time else 0
        }
    
    def reset_session(self):
        """Reset calculator for new trading session"""
        self.cumulative_delta = 0.0
        self.session_start_time = None
        
        self.trade_history.clear()
        self.delta_history.clear()
        self.price_history.clear()
        
        self.current_minute_delta = 0.0
        self.current_minute_start = None
        self.minute_deltas.clear()
        
        self.session_buy_volume = 0.0
        self.session_sell_volume = 0.0
        self.total_session_volume = 0.0
        
        self.recent_buy_volume = 0.0
        self.recent_sell_volume = 0.0
        self.recent_period_trades.clear()
        
        self.last_signal_time = None
        self.total_trades_processed = 0
        self.signals_generated = 0
    
    def update_config(self, config: Dict):
        """Update calculator configuration"""
        if 'strong_delta_threshold' in config:
            self.strong_delta_threshold = config['strong_delta_threshold']
        if 'moderate_delta_threshold' in config:
            self.moderate_delta_threshold = config['moderate_delta_threshold']
        if 'min_confidence' in config:
            self.min_confidence = config['min_confidence']
        if 'signal_cooldown_seconds' in config:
            self.signal_cooldown_seconds = config['signal_cooldown_seconds']