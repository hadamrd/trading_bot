#!/usr/bin/env python3
"""
Advanced Signal Processor - Order Book + Volume Delta Analysis
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from collections import deque
from dataclasses import dataclass
import statistics

# Import the delta calculator
from ob_trading.order_imbalance.delta_calculator import DeltaCalculator, TradeData, DeltaSignal, DeltaMetrics


@dataclass
class TradingSignal:
    """Professional trading signal with multi-factor confirmation"""
    timestamp: datetime
    signal_type: str
    symbol: str
    price: float
    
    # Signal strength and confidence
    overall_confidence: float
    signal_strength: float
    
    # Component signals
    order_book_signal: Optional[str]
    delta_signal: Optional[DeltaSignal]
    
    # Analysis details
    reason: str
    duration: float
    confirmations: List[str]
    
    # Market context
    bid_ratio: float
    spread_pct: float
    total_volume: float
    
    # Risk metrics
    recommended_position_size: float
    suggested_hold_time: int  # seconds


@dataclass
class OrderBookCondition:
    """Tracks order book imbalance conditions over time"""
    condition_type: str
    first_seen: datetime
    last_seen: datetime
    peak_strength: float
    current_strength: float
    triggered: bool = False


class AdvancedSignalProcessor:
    """
    Professional signal processor combining order book imbalance + volume delta analysis
    
    Features:
    - Multi-factor signal confirmation
    - Delta-confirmed order book analysis
    - Quality scoring and filtering
    - Professional signal timing
    """
    
    def __init__(self, 
                 symbol: str = "SUIUSDT",
                 
                 # Order book thresholds (existing logic)
                 strong_imbalance_threshold: float = 0.80,
                 moderate_imbalance_threshold: float = 0.70,
                 large_wall_threshold: float = 0.25,
                 
                 # Delta integration
                 enable_delta_confirmation: bool = True,
                 delta_weight: float = 0.4,  # Weight of delta vs order book
                 
                 # Quality filters
                 min_persistence_seconds: float = 3.0,
                 signal_cooldown_seconds: float = 30.0,
                 min_overall_confidence: float = 0.7,
                 
                 # Risk management
                 max_position_size: float = 0.02,  # 2% of account
                 base_hold_time: int = 60,  # seconds
                 
                 # History settings
                 history_length: int = 100):
        
        self.symbol = symbol
        
        # Thresholds
        self.strong_imbalance_threshold = strong_imbalance_threshold
        self.moderate_imbalance_threshold = moderate_imbalance_threshold
        self.large_wall_threshold = large_wall_threshold
        
        # Delta integration
        self.enable_delta_confirmation = enable_delta_confirmation
        self.delta_weight = delta_weight
        self.order_book_weight = 1.0 - delta_weight
        
        # Quality filters
        self.min_persistence_seconds = min_persistence_seconds
        self.signal_cooldown_seconds = signal_cooldown_seconds
        self.min_overall_confidence = min_overall_confidence
        
        # Risk management
        self.max_position_size = max_position_size
        self.base_hold_time = base_hold_time
        
        # Initialize delta calculator
        self.delta_calculator = DeltaCalculator(
            min_confidence=0.6,  # Lower threshold for delta component
            trade_history_size=500,
            strong_delta_threshold=0.7,
            moderate_delta_threshold=0.4
        )
        
        # Market data history
        self.market_history = deque(maxlen=history_length)
        self.signal_history = deque(maxlen=50)
        
        # Order book condition tracking (existing logic)
        self.active_conditions = {}
        self.last_signal_times = {}
        self.baseline_bid_ratio = 0.5
        
        # Signal subscribers
        self.signal_subscribers = []
        
        # Statistics
        self.total_updates = 0
        self.total_signals = 0
        self.signals_filtered = 0
        self.delta_confirmations = 0
        self.order_book_only_signals = 0
    
    def subscribe_to_signals(self, callback: Callable[[TradingSignal], None]):
        """Subscribe to professional signals"""
        self.signal_subscribers.append(callback)
    
    def process_market_data(self, market_data: Dict):
        """Process market data and generate professional signals"""
        
        if 'type' in market_data and market_data['type'] == 'status_change':
            return
        
        if 'metrics' not in market_data:
            return
        
        self.total_updates += 1
        metrics = market_data['metrics']
        timestamp = datetime.fromisoformat(market_data['timestamp'])
        
        # Store in history
        self.market_history.append({
            'timestamp': timestamp,
            'metrics': metrics
        })
        
        # Update baseline
        self._update_baseline()
        
        # Track order book conditions
        self._track_order_book_conditions(metrics, timestamp)
        
        # Process trades through delta calculator if available
        delta_signal = None
        delta_metrics = None
        
        if self.enable_delta_confirmation and 'trades' in market_data:
            delta_signal, delta_metrics = self._process_trade_data(market_data['trades'])
        
        # Generate professional signals
        signals = self._generate_signals(metrics, timestamp, delta_signal, delta_metrics)
        
        # Notify subscribers
        for signal in signals:
            self._notify_signal_subscribers(signal)
    
    def _process_trade_data(self, trades_data: List[Dict]) -> tuple[Optional[DeltaSignal], Optional[DeltaMetrics]]:
        """Process trade data through delta calculator"""
        
        if not trades_data:
            return None, None
        
        latest_signal = None
        latest_metrics = None
        
        for trade_dict in trades_data:
            # Convert to TradeData format
            trade = TradeData(
                price=float(trade_dict.get('price', 0)),
                quantity=float(trade_dict.get('quantity', 0)),
                timestamp=int(trade_dict.get('timestamp', 0)),
                is_buyer_maker=bool(trade_dict.get('is_buyer_maker', False)),
                trade_id=str(trade_dict.get('trade_id', ''))
            )
            
            signal, metrics = self.delta_calculator.process_trade(trade)
            
            if signal:
                latest_signal = signal
            latest_metrics = metrics
        
        return latest_signal, latest_metrics
    
    def _update_baseline(self):
        """Update baseline metrics for comparison"""
        if len(self.market_history) >= 50:
            recent_ratios = [data['metrics'].get('bid_ratio', 0.5) 
                           for data in list(self.market_history)[-50:]]
            self.baseline_bid_ratio = statistics.median(recent_ratios)
    
    def _track_order_book_conditions(self, metrics: Dict, timestamp: datetime):
        """Track order book imbalance conditions"""
        
        bid_ratio = metrics.get('bid_ratio', 0.5)
        total_value = metrics.get('total_value', 0)
        largest_bid = metrics.get('largest_bid', {})
        largest_ask = metrics.get('largest_ask', {})
        
        # Track imbalance conditions
        conditions_to_check = [
            ('strong_buy_imbalance', bid_ratio >= self.strong_imbalance_threshold, bid_ratio),
            ('strong_sell_imbalance', bid_ratio <= (1 - self.strong_imbalance_threshold), 1 - bid_ratio),
            ('moderate_buy_imbalance', bid_ratio >= self.moderate_imbalance_threshold, bid_ratio),
            ('moderate_sell_imbalance', bid_ratio <= (1 - self.moderate_imbalance_threshold), 1 - bid_ratio),
        ]
        
        # Track wall conditions
        if total_value > 0:
            wall_threshold_value = total_value * self.large_wall_threshold
            
            if largest_bid.get('value', 0) > wall_threshold_value:
                conditions_to_check.append((
                    'large_bid_wall', 
                    True, 
                    largest_bid.get('value', 0) / total_value
                ))
            
            if largest_ask.get('value', 0) > wall_threshold_value:
                conditions_to_check.append((
                    'large_ask_wall', 
                    True, 
                    largest_ask.get('value', 0) / total_value
                ))
        
        # Update condition tracking
        for condition_type, is_active, strength in conditions_to_check:
            if is_active:
                if condition_type not in self.active_conditions:
                    self.active_conditions[condition_type] = OrderBookCondition(
                        condition_type=condition_type,
                        first_seen=timestamp,
                        last_seen=timestamp,
                        peak_strength=strength,
                        current_strength=strength
                    )
                else:
                    condition = self.active_conditions[condition_type]
                    condition.last_seen = timestamp
                    condition.current_strength = strength
                    condition.peak_strength = max(condition.peak_strength, strength)
            else:
                if condition_type in self.active_conditions:
                    del self.active_conditions[condition_type]
        
        # Clean up old conditions
        self._cleanup_old_conditions(timestamp)
    
    def _cleanup_old_conditions(self, timestamp: datetime):
        """Remove conditions that haven't been seen recently"""
        expired_conditions = []
        
        for condition_type, condition in self.active_conditions.items():
            if (timestamp - condition.last_seen).total_seconds() > 10:
                expired_conditions.append(condition_type)
        
        for condition_type in expired_conditions:
            del self.active_conditions[condition_type]
    
    def _generate_signals(self, 
                         metrics: Dict, 
                         timestamp: datetime,
                         delta_signal: Optional[DeltaSignal],
                         delta_metrics: Optional[DeltaMetrics]) -> List[TradingSignal]:
        """Generate professional signals with multi-factor confirmation"""
        
        signals = []
        
        for condition_type, condition in self.active_conditions.items():
            # Check persistence
            persistence_duration = (timestamp - condition.first_seen).total_seconds()
            
            if (persistence_duration >= self.min_persistence_seconds and 
                not condition.triggered and
                self._check_signal_cooldown(condition_type, timestamp)):
                
                # Calculate order book confidence
                ob_confidence = self._calculate_order_book_confidence(condition, metrics)
                
                # Check for delta confirmation if enabled
                delta_confirmation_score = 0.0
                confirmations = ["ORDER_BOOK"]
                
                if self.enable_delta_confirmation and delta_signal:
                    delta_confirmation_score = self._calculate_delta_confirmation(
                        condition, delta_signal, delta_metrics
                    )
                    if delta_confirmation_score > 0.5:
                        confirmations.append("VOLUME_DELTA")
                        self.delta_confirmations += 1
                
                # Calculate overall confidence
                overall_confidence = self._calculate_overall_confidence(
                    ob_confidence, delta_confirmation_score
                )
                
                if overall_confidence >= self.min_overall_confidence:
                    # Create professional signal
                    signal = self._create_signal(
                        condition, metrics, timestamp, 
                        overall_confidence, delta_signal, 
                        delta_metrics, confirmations
                    )
                    
                    if signal:
                        signals.append(signal)
                        condition.triggered = True
                        self.last_signal_times[condition_type] = timestamp
                        self.total_signals += 1
                        
                        if len(confirmations) == 1:
                            self.order_book_only_signals += 1
                else:
                    self.signals_filtered += 1
        
        return signals
    
    def _calculate_order_book_confidence(self, condition: OrderBookCondition, metrics: Dict) -> float:
        """Calculate confidence from order book analysis"""
        confidence = 0.0
        
        # Base confidence from strength
        confidence += min(condition.peak_strength, 1.0) * 0.4
        
        # Persistence bonus
        persistence = (condition.last_seen - condition.first_seen).total_seconds()
        persistence_score = min(persistence / 10.0, 1.0) * 0.3
        confidence += persistence_score
        
        # Deviation from baseline
        if 'imbalance' in condition.condition_type:
            current_ratio = metrics.get('bid_ratio', 0.5)
            deviation = abs(current_ratio - self.baseline_bid_ratio)
            deviation_score = min(deviation / 0.2, 1.0) * 0.2
            confidence += deviation_score
        
        # Volume context
        total_volume = metrics.get('total_value', 0)
        volume_score = min(total_volume / 100000, 0.1)  # Bonus for high volume
        confidence += volume_score
        
        return min(confidence, 1.0)
    
    def _calculate_delta_confirmation(self, 
                                    condition: OrderBookCondition,
                                    delta_signal: DeltaSignal,
                                    delta_metrics: DeltaMetrics) -> float:
        """Calculate delta confirmation score"""
        
        if not delta_signal or not delta_metrics:
            return 0.0
        
        # Check signal alignment
        alignment_score = 0.0
        
        if 'buy' in condition.condition_type and 'BUY' in delta_signal.signal_type:
            alignment_score = 0.8
        elif 'sell' in condition.condition_type and 'SELL' in delta_signal.signal_type:
            alignment_score = 0.8
        elif 'wall' in condition.condition_type:
            # Walls can support either direction
            alignment_score = 0.6
        
        # Boost for delta strength
        delta_strength_boost = delta_signal.confidence * 0.3
        
        # Boost for delta acceleration in same direction
        if delta_signal.acceleration > 0 and 'buy' in condition.condition_type:
            acceleration_boost = 0.2
        elif delta_signal.acceleration < 0 and 'sell' in condition.condition_type:
            acceleration_boost = 0.2
        else:
            acceleration_boost = 0.0
        
        return min(alignment_score + delta_strength_boost + acceleration_boost, 1.0)
    
    def _calculate_overall_confidence(self, ob_confidence: float, delta_confirmation: float) -> float:
        """Calculate overall signal confidence"""
        
        if not self.enable_delta_confirmation:
            return ob_confidence
        
        # Weighted combination
        weighted_score = (
            ob_confidence * self.order_book_weight + 
            delta_confirmation * self.delta_weight
        )
        
        # Bonus for having both confirmations
        if delta_confirmation > 0.5:
            weighted_score *= 1.1  # 10% bonus for delta confirmation
        
        return min(weighted_score, 1.0)
    
    def _create_signal(self,
                      condition: OrderBookCondition,
                      metrics: Dict,
                      timestamp: datetime,
                      overall_confidence: float,
                      delta_signal: Optional[DeltaSignal],
                      delta_metrics: Optional[DeltaMetrics],
                      confirmations: List[str]) -> Optional[TradingSignal]:
        """Create professional trading signal"""
        
        mid_price = metrics.get('mid_price', 0)
        
        # Map condition types to signal types
        signal_mapping = {
            'strong_buy_imbalance': 'STRONG_BUY',
            'strong_sell_imbalance': 'STRONG_SELL',
            'moderate_buy_imbalance': 'MODERATE_BUY',
            'moderate_sell_imbalance': 'MODERATE_SELL',
            'large_bid_wall': 'BID_WALL_SUPPORT',
            'large_ask_wall': 'ASK_WALL_RESISTANCE'
        }
        
        signal_type = signal_mapping.get(condition.condition_type, 'UNKNOWN')
        
        # Generate reason
        reason = self._generate_signal_reason(condition, metrics, delta_signal, confirmations)
        
        # Calculate position sizing
        position_size = self._calculate_recommended_position_size(overall_confidence)
        
        # Calculate hold time
        hold_time = self._calculate_suggested_hold_time(condition, overall_confidence)
        
        duration = (timestamp - condition.first_seen).total_seconds()
        
        return TradingSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            symbol=self.symbol,
            price=mid_price,
            overall_confidence=overall_confidence,
            signal_strength=condition.peak_strength,
            order_book_signal=condition.condition_type,
            delta_signal=delta_signal,
            reason=reason,
            duration=duration,
            confirmations=confirmations,
            bid_ratio=metrics.get('bid_ratio', 0.5),
            spread_pct=metrics.get('spread_pct', 0),
            total_volume=metrics.get('total_value', 0),
            recommended_position_size=position_size,
            suggested_hold_time=hold_time
        )
    
    def _generate_signal_reason(self, 
                              condition: OrderBookCondition,
                              metrics: Dict,
                              delta_signal: Optional[DeltaSignal],
                              confirmations: List[str]) -> str:
        """Generate detailed signal reason"""
        
        reasons = []
        
        # Order book reason
        if 'imbalance' in condition.condition_type:
            bid_ratio = metrics.get('bid_ratio', 0.5)
            if 'buy' in condition.condition_type:
                reasons.append(f"Buy imbalance: {bid_ratio:.1%} bid ratio")
            else:
                reasons.append(f"Sell imbalance: {(1-bid_ratio):.1%} ask ratio")
        elif 'wall' in condition.condition_type:
            if 'bid' in condition.condition_type:
                wall_value = metrics.get('largest_bid', {}).get('value', 0)
                reasons.append(f"Large bid wall: ${wall_value:,.0f}")
            else:
                wall_value = metrics.get('largest_ask', {}).get('value', 0)
                reasons.append(f"Large ask wall: ${wall_value:,.0f}")
        
        # Delta confirmation
        if delta_signal and "VOLUME_DELTA" in confirmations:
            reasons.append(f"Delta confirms: {delta_signal.reason}")
        
        # Confirmations
        reasons.append(f"Confirmations: {', '.join(confirmations)}")
        
        return " | ".join(reasons)
    
    def _calculate_recommended_position_size(self, confidence: float) -> float:
        """Calculate recommended position size based on confidence"""
        
        # Base size scaled by confidence
        base_size = self.max_position_size * confidence
        
        # Additional scaling factors could be added here
        # (volatility, time of day, etc.)
        
        return min(base_size, self.max_position_size)
    
    def _calculate_suggested_hold_time(self, condition: OrderBookCondition, confidence: float) -> int:
        """Calculate suggested hold time in seconds"""
        
        # Base hold time scaled by signal type and confidence
        if 'strong' in condition.condition_type:
            base_time = self.base_hold_time * 2
        elif 'wall' in condition.condition_type:
            base_time = self.base_hold_time * 1.5
        else:
            base_time = self.base_hold_time
        
        # Scale by confidence
        scaled_time = int(base_time * (0.5 + confidence * 0.5))
        
        return max(scaled_time, 30)  # Minimum 30 seconds
    
    def _check_signal_cooldown(self, signal_type: str, timestamp: datetime) -> bool:
        """Check if enough time has passed since last signal of this type"""
        
        if signal_type not in self.last_signal_times:
            return True
        
        time_since = (timestamp - self.last_signal_times[signal_type]).total_seconds()
        return time_since >= self.signal_cooldown_seconds
    
    def _notify_signal_subscribers(self, signal: TradingSignal):
        """Notify all signal subscribers"""
        self.signal_history.append(signal)
        
        for callback in self.signal_subscribers:
            try:
                callback(signal)
            except Exception as e:
                print(f"⚠️  Error in signal subscriber: {e}")
    
    def get_recent_signals(self, minutes: int = 10) -> List[TradingSignal]:
        """Get signals from the last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [signal for signal in self.signal_history if signal.timestamp > cutoff]
    
    def get_statistics(self) -> Dict:
        """Get comprehensive processor statistics"""
        recent_signals = self.get_recent_signals(5)
        delta_stats = self.delta_calculator.get_session_statistics()
        
        return {
            # Signal statistics
            'total_updates': self.total_updates,
            'total_signals': self.total_signals,
            'signals_filtered': self.signals_filtered,
            'recent_signals_5min': len(recent_signals),
            'active_conditions': len(self.active_conditions),
            
            # Delta integration stats
            'delta_confirmations': self.delta_confirmations,
            'order_book_only_signals': self.order_book_only_signals,
            'delta_confirmation_rate': (self.delta_confirmations / max(self.total_signals, 1)) * 100,
            
            # Quality metrics
            'baseline_bid_ratio': f"{self.baseline_bid_ratio:.1%}",
            'signal_rate': len(recent_signals) / 5 if recent_signals else 0,
            'filter_rate': self.signals_filtered / max(self.total_updates, 1),
            'avg_signal_confidence': statistics.mean([s.overall_confidence for s in recent_signals]) if recent_signals else 0,
            
            # Delta statistics
            'delta_stats': delta_stats
        }
    
    def update_config(self, new_config: Dict):
        """Update processor configuration"""
        
        # Order book config
        if 'strong_imbalance_threshold' in new_config:
            self.strong_imbalance_threshold = new_config['strong_imbalance_threshold']
        if 'moderate_imbalance_threshold' in new_config:
            self.moderate_imbalance_threshold = new_config['moderate_imbalance_threshold']
        if 'large_wall_threshold' in new_config:
            self.large_wall_threshold = new_config['large_wall_threshold']
        if 'min_persistence_seconds' in new_config:
            self.min_persistence_seconds = new_config['min_persistence_seconds']
        if 'signal_cooldown_seconds' in new_config:
            self.signal_cooldown_seconds = new_config['signal_cooldown_seconds']
        if 'min_overall_confidence' in new_config:
            self.min_overall_confidence = new_config['min_overall_confidence']
        
        # Delta integration config
        if 'enable_delta_confirmation' in new_config:
            self.enable_delta_confirmation = new_config['enable_delta_confirmation']
        if 'delta_weight' in new_config:
            self.delta_weight = new_config['delta_weight']
            self.order_book_weight = 1.0 - self.delta_weight
        
        # Update delta calculator config
        delta_config = {}
        delta_mapping = {
            'delta_strong_threshold': 'strong_delta_threshold',
            'delta_moderate_threshold': 'moderate_delta_threshold',
            'delta_min_confidence': 'min_confidence'
        }
        
        for web_key, delta_key in delta_mapping.items():
            if web_key in new_config:
                delta_config[delta_key] = new_config[web_key]
        
        if delta_config:
            self.delta_calculator.update_config(delta_config)
        
        print(f"📋 Advanced signal processor config updated: {new_config}")
    
    def reset_session(self):
        """Reset processor for new trading session"""
        self.active_conditions.clear()
        self.last_signal_times.clear()
        self.market_history.clear()
        self.signal_history.clear()
        
        self.total_updates = 0
        self.total_signals = 0
        self.signals_filtered = 0
        self.delta_confirmations = 0
        self.order_book_only_signals = 0
        
        self.baseline_bid_ratio = 0.5
        
        if self.delta_calculator:
            self.delta_calculator.reset_session()