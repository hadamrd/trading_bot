#!/usr/bin/env python3
"""
Signal Processor Module - Creates high-quality, filtered trading signals
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from collections import deque
from dataclasses import dataclass
import statistics


@dataclass
class TradingSignal:
    """High-quality trading signal"""
    timestamp: datetime
    signal_type: str
    symbol: str
    price: float
    strength: float
    confidence: float
    reason: str
    duration: float  # How long condition persisted
    metadata: Dict


@dataclass
class SignalCondition:
    """Tracks a potential signal condition over time"""
    condition_type: str
    first_seen: datetime
    last_seen: datetime
    peak_strength: float
    current_strength: float
    triggered: bool = False


class SmartSignalProcessor:
    """
    Processes market data and creates high-quality, filtered signals
    
    Key features:
    - Signal deduplication
    - Persistence requirements  
    - Quality scoring
    - Trend analysis
    - Noise filtering
    """
    
    def __init__(self, 
                 symbol: str = "SUIUSDT",
                 # Signal thresholds (more conservative)
                 strong_imbalance_threshold: float = 0.80,
                 moderate_imbalance_threshold: float = 0.70,
                 large_wall_threshold: float = 0.25,
                 
                 # Quality filters
                 min_persistence_seconds: float = 3.0,
                 signal_cooldown_seconds: float = 30.0,
                 min_confidence: float = 0.7,
                 
                 # History settings
                 history_length: int = 100):
        
        self.symbol = symbol
        
        # Thresholds
        self.strong_imbalance_threshold = strong_imbalance_threshold
        self.moderate_imbalance_threshold = moderate_imbalance_threshold
        self.large_wall_threshold = large_wall_threshold
        
        # Quality filters
        self.min_persistence_seconds = min_persistence_seconds
        self.signal_cooldown_seconds = signal_cooldown_seconds
        self.min_confidence = min_confidence
        
        # Data history
        self.market_history = deque(maxlen=history_length)
        self.signal_history = deque(maxlen=50)
        
        # Signal tracking
        self.active_conditions = {}  # condition_type -> SignalCondition
        self.last_signal_times = {}  # signal_type -> timestamp
        self.baseline_bid_ratio = 0.5
        
        # Subscribers
        self.signal_subscribers = []
        
        # Statistics
        self.total_updates = 0
        self.total_signals = 0
        self.signals_filtered = 0
        
    def subscribe_to_signals(self, callback: Callable[[TradingSignal], None]):
        """Subscribe to high-quality signals"""
        self.signal_subscribers.append(callback)
    
    def process_market_data(self, market_data: Dict):
        """Process incoming market data and detect signals"""
        
        if 'type' in market_data and market_data['type'] == 'status_change':
            return  # Skip status updates
        
        if 'metrics' not in market_data:
            return  # Skip if no metrics
        
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
        
        # Track conditions
        self._track_conditions(metrics, timestamp)
        
        # Generate signals
        signals = self._generate_signals(metrics, timestamp)
        
        # Notify subscribers
        for signal in signals:
            self._notify_signal_subscribers(signal)
    
    def _update_baseline(self):
        """Update baseline metrics for comparison"""
        if len(self.market_history) >= 50:
            recent_ratios = [data['metrics'].get('bid_ratio', 0.5) 
                           for data in list(self.market_history)[-50:]]
            self.baseline_bid_ratio = statistics.median(recent_ratios)
    
    def _track_conditions(self, metrics: Dict, timestamp: datetime):
        """Track potential signal conditions over time"""
        
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
                    # New condition
                    self.active_conditions[condition_type] = SignalCondition(
                        condition_type=condition_type,
                        first_seen=timestamp,
                        last_seen=timestamp,
                        peak_strength=strength,
                        current_strength=strength
                    )
                else:
                    # Update existing condition
                    condition = self.active_conditions[condition_type]
                    condition.last_seen = timestamp
                    condition.current_strength = strength
                    condition.peak_strength = max(condition.peak_strength, strength)
            else:
                # Condition no longer active
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
    
    def _generate_signals(self, metrics: Dict, timestamp: datetime) -> List[TradingSignal]:
        """Generate high-quality signals from tracked conditions"""
        
        signals = []
        
        for condition_type, condition in self.active_conditions.items():
            # Check if condition has persisted long enough
            persistence_duration = (timestamp - condition.first_seen).total_seconds()
            
            if (persistence_duration >= self.min_persistence_seconds and 
                not condition.triggered and
                self._check_signal_cooldown(condition_type, timestamp)):
                
                # Calculate confidence
                confidence = self._calculate_confidence(condition, metrics)
                
                if confidence >= self.min_confidence:
                    # Create signal
                    signal = self._create_signal(condition, metrics, timestamp, confidence)
                    
                    if signal:
                        signals.append(signal)
                        condition.triggered = True
                        self.last_signal_times[condition_type] = timestamp
                        self.total_signals += 1
                else:
                    self.signals_filtered += 1
        
        return signals
    
    def _check_signal_cooldown(self, signal_type: str, timestamp: datetime) -> bool:
        """Check if enough time has passed since last signal of this type"""
        
        if signal_type not in self.last_signal_times:
            return True
        
        time_since = (timestamp - self.last_signal_times[signal_type]).total_seconds()
        return time_since >= self.signal_cooldown_seconds
    
    def _calculate_confidence(self, condition: SignalCondition, metrics: Dict) -> float:
        """Calculate confidence score for a signal"""
        
        confidence = 0.0
        
        # Base confidence from strength
        confidence += min(condition.peak_strength, 1.0) * 0.4
        
        # Persistence bonus
        persistence = (condition.last_seen - condition.first_seen).total_seconds()
        persistence_score = min(persistence / 10.0, 1.0) * 0.3  # Max bonus at 10 seconds
        confidence += persistence_score
        
        # Deviation from baseline bonus
        if 'imbalance' in condition.condition_type:
            current_ratio = metrics.get('bid_ratio', 0.5)
            deviation = abs(current_ratio - self.baseline_bid_ratio)
            deviation_score = min(deviation / 0.2, 1.0) * 0.2  # Max bonus at 20% deviation
            confidence += deviation_score
        
        # Trend consistency bonus
        trend_score = self._calculate_trend_consistency(condition.condition_type) * 0.1
        confidence += trend_score
        
        return min(confidence, 1.0)
    
    def _calculate_trend_consistency(self, condition_type: str) -> float:
        """Calculate how consistent the trend has been"""
        
        if len(self.market_history) < 10:
            return 0.0
        
        recent_data = list(self.market_history)[-10:]
        
        if 'buy' in condition_type:
            # Check if bid ratios have been generally increasing
            ratios = [data['metrics'].get('bid_ratio', 0.5) for data in recent_data]
            increases = sum(1 for i in range(1, len(ratios)) if ratios[i] > ratios[i-1])
            return increases / (len(ratios) - 1)
        
        elif 'sell' in condition_type:
            # Check if bid ratios have been generally decreasing
            ratios = [data['metrics'].get('bid_ratio', 0.5) for data in recent_data]
            decreases = sum(1 for i in range(1, len(ratios)) if ratios[i] < ratios[i-1])
            return decreases / (len(ratios) - 1)
        
        return 0.5  # Neutral for wall signals
    
    def _create_signal(self, condition: SignalCondition, metrics: Dict, timestamp: datetime, confidence: float) -> Optional[TradingSignal]:
        """Create a trading signal from a condition"""
        
        mid_price = metrics.get('mid_price', 0)
        
        # Map condition types to signal types and reasons
        signal_mapping = {
            'strong_buy_imbalance': {
                'type': 'STRONG_BUY_PRESSURE',
                'reason': f"Persistent strong buy imbalance ({metrics.get('bid_ratio', 0):.1%} bids)"
            },
            'strong_sell_imbalance': {
                'type': 'STRONG_SELL_PRESSURE', 
                'reason': f"Persistent strong sell imbalance ({1-metrics.get('bid_ratio', 0.5):.1%} asks)"
            },
            'moderate_buy_imbalance': {
                'type': 'MODERATE_BUY_PRESSURE',
                'reason': f"Building buy pressure ({metrics.get('bid_ratio', 0):.1%} bids)"
            },
            'moderate_sell_imbalance': {
                'type': 'MODERATE_SELL_PRESSURE',
                'reason': f"Building sell pressure ({1-metrics.get('bid_ratio', 0.5):.1%} asks)"
            },
            'large_bid_wall': {
                'type': 'LARGE_BID_WALL',
                'reason': f"Large bid wall: ${metrics.get('largest_bid', {}).get('value', 0):,.0f}"
            },
            'large_ask_wall': {
                'type': 'LARGE_ASK_WALL',
                'reason': f"Large ask wall: ${metrics.get('largest_ask', {}).get('value', 0):,.0f}"
            }
        }
        
        if condition.condition_type not in signal_mapping:
            return None
        
        signal_info = signal_mapping[condition.condition_type]
        duration = (timestamp - condition.first_seen).total_seconds()
        
        return TradingSignal(
            timestamp=timestamp,
            signal_type=signal_info['type'],
            symbol=self.symbol,
            price=mid_price,
            strength=condition.peak_strength,
            confidence=confidence,
            reason=signal_info['reason'],
            duration=duration,
            metadata={
                'baseline_deviation': abs(metrics.get('bid_ratio', 0.5) - self.baseline_bid_ratio),
                'total_value': metrics.get('total_value', 0),
                'spread_pct': metrics.get('spread_pct', 0)
            }
        )
    
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
        """Get processor statistics"""
        recent_signals = self.get_recent_signals(5)
        
        return {
            'total_updates': self.total_updates,
            'total_signals': self.total_signals,
            'signals_filtered': self.signals_filtered,
            'recent_signals_5min': len(recent_signals),
            'active_conditions': len(self.active_conditions),
            'baseline_bid_ratio': f"{self.baseline_bid_ratio:.1%}",
            'signal_rate': len(recent_signals) / 5 if recent_signals else 0,  # per minute
            'filter_rate': self.signals_filtered / max(self.total_updates, 1)
        }
    
    def update_config(self, new_config: Dict):
        """Update processor configuration"""
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
        if 'min_confidence' in new_config:
            self.min_confidence = new_config['min_confidence']
        
        print(f"📋 Signal processor config updated: {new_config}")


# Test function
def test_signal_processor():
    """Test the signal processor"""
    
    def on_signal(signal: TradingSignal):
        print(f"🚨 {signal.signal_type} | ${signal.price:.4f} | "
              f"Confidence: {signal.confidence:.1%} | "
              f"Duration: {signal.duration:.1f}s | "
              f"{signal.reason}")
    
    # Create processor with conservative settings
    processor = SmartSignalProcessor(
        strong_imbalance_threshold=0.80,
        min_persistence_seconds=3.0,
        signal_cooldown_seconds=30.0,
        min_confidence=0.7
    )
    
    processor.subscribe_to_signals(on_signal)
    
    # Simulate some market data
    import random
    for i in range(100):
        # Simulate varying market conditions
        bid_ratio = 0.5 + random.uniform(-0.3, 0.3)
        
        fake_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'TEST',
            'metrics': {
                'bid_ratio': bid_ratio,
                'mid_price': 1.0 + random.uniform(-0.1, 0.1),
                'total_value': 100000,
                'largest_bid': {'value': 20000},
                'largest_ask': {'value': 18000}
            }
        }
        
        processor.process_market_data(fake_data)
        time.sleep(0.1)
    
    stats = processor.get_statistics()
    print(f"\n📊 Final stats: {stats}")


if __name__ == "__main__":
    test_signal_processor()