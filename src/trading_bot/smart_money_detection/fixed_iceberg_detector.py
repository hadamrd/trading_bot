#!/usr/bin/env python3
"""
Fixed Iceberg Detection Engine - Much more conservative and realistic
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import numpy as np


@dataclass
class OrderPattern:
    """Tracks repeated orders at same price/size"""
    price: float
    size: float
    side: str  # 'bid' or 'ask'
    first_seen: datetime
    last_seen: datetime
    occurrences: int = 1
    total_volume_traded: float = 0.0
    avg_replenishment_time: float = 0.0  # seconds
    
    @property
    def duration_seconds(self) -> float:
        return (self.last_seen - self.first_seen).total_seconds()
    
    @property
    def frequency(self) -> float:
        """Orders per minute"""
        if self.duration_seconds <= 0:
            return 0
        return (self.occurrences * 60) / self.duration_seconds
    
    @property
    def iceberg_score(self) -> float:
        """Realistic iceberg probability score 0-100"""
        score = 0
        
        # MUCH STRICTER CRITERIA
        
        # Need significant size (>1.0 BTC equivalent)
        if self.size < 1.0:
            return 0  # Ignore tiny orders
        
        # Need many occurrences for high confidence
        if self.occurrences >= 10:
            score += 40
        elif self.occurrences >= 7:
            score += 30
        elif self.occurrences >= 5:
            score += 20
        else:
            return 0  # Not enough occurrences
        
        # Consistent size gets points
        score += 15
        
        # Fast replenishment (but not too fast - avoid HFT noise)
        if 1.0 < self.avg_replenishment_time < 10.0:
            score += 25
        
        # Reasonable frequency (not too high = noise)
        if 5 < self.frequency < 30:  # 5-30 orders/minute
            score += 20
        
        return min(score, 100)


@dataclass
class IcebergDetection:
    """Detected iceberg order result"""
    pattern: OrderPattern
    confidence: float  # 0-1
    estimated_total_size: float
    detected_at: datetime
    side: str  # 'bid' or 'ask'
    
    def __str__(self):
        return (f"Iceberg {self.side.upper()}: ${self.pattern.price:,.2f} "
                f"x{self.pattern.size:.2f} (confidence: {self.confidence:.1%}, "
                f"occurrences: {self.pattern.occurrences})")


class RealisticIcebergDetector:
    """
    Much more conservative iceberg detector for real trading
    """
    
    def __init__(self, 
                 min_occurrences: int = 5,          # Need at least 5 repetitions
                 max_pattern_age: int = 600,        # 10 minutes max
                 min_iceberg_score: float = 80.0,   # Much higher threshold
                 min_size: float = 1.0,             # Minimum 1.0 BTC equivalent
                 size_tolerance: float = 0.02):     # 2% size variation
        
        self.min_occurrences = min_occurrences
        self.max_pattern_age = max_pattern_age  
        self.min_iceberg_score = min_iceberg_score
        self.min_size = min_size
        self.size_tolerance = size_tolerance
        
        # Pattern tracking
        self.bid_patterns: Dict[Tuple[float, float], OrderPattern] = {}
        self.ask_patterns: Dict[Tuple[float, float], OrderPattern] = {}
        
        # Order book history for replenishment detection
        self.order_history: deque = deque(maxlen=200)  # Reduced buffer
        
        # Recent detections (avoid spam)
        self.recent_detections: deque = deque(maxlen=20)
        
        # Statistics
        self.total_patterns_tracked = 0
        self.total_detections = 0
        self.noise_filtered = 0
    
    def update_order_book(self, bids: List[List], asks: List[List], timestamp: Optional[datetime] = None):
        """Process order book with strict filtering"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # PRE-FILTER: Remove noise and tiny orders
        filtered_bids = self._filter_meaningful_orders(bids)
        filtered_asks = self._filter_meaningful_orders(asks)
        
        if not filtered_bids and not filtered_asks:
            return []  # Nothing meaningful to track
        
        # Store in history
        self.order_history.append({
            'timestamp': timestamp,
            'bids': filtered_bids,
            'asks': filtered_asks
        })
        
        # Track patterns only on meaningful orders
        self._track_patterns(filtered_bids, 'bid', timestamp)
        self._track_patterns(filtered_asks, 'ask', timestamp)
        
        # Clean old patterns
        self._cleanup_old_patterns(timestamp)
        
        # Detect icebergs with strict criteria
        detections = self._detect_icebergs(timestamp)
        
        return detections
    
    def _filter_meaningful_orders(self, orders: List[List]) -> List[Tuple[float, float]]:
        """Filter out noise and keep only meaningful orders"""
        
        meaningful_orders = []
        
        for order in orders:
            try:
                price = float(order[0])
                size = float(order[1])
                
                # STRICT FILTERS
                if size < self.min_size:
                    self.noise_filtered += 1
                    continue  # Too small
                
                if size > 1000:  # Probably display/API error
                    continue
                
                if price <= 0:  # Invalid price
                    continue
                
                meaningful_orders.append((price, size))
                
            except (ValueError, IndexError):
                continue  # Invalid data
        
        return meaningful_orders
    
    def _track_patterns(self, levels: List[Tuple[float, float]], side: str, timestamp: datetime):
        """Track patterns only on filtered, meaningful orders"""
        
        patterns = self.bid_patterns if side == 'bid' else self.ask_patterns
        
        for price, size in levels:
            # Create pattern key with tight size tolerance
            size_rounded = self._round_size_for_pattern(size)
            pattern_key = (price, size_rounded)
            
            if pattern_key in patterns:
                # Update existing pattern
                pattern = patterns[pattern_key]
                
                # Check if this is actually a replenishment (time gap)
                time_since_last = (timestamp - pattern.last_seen).total_seconds()
                
                if time_since_last > 0.5:  # At least 500ms gap = real replenishment
                    pattern.last_seen = timestamp
                    pattern.occurrences += 1
                    self._update_replenishment_time(pattern, time_since_last)
                # Ignore if too fast (likely same order still there)
                
            else:
                # New pattern - but only track if significant
                if size >= self.min_size:
                    patterns[pattern_key] = OrderPattern(
                        price=price,
                        size=size_rounded,
                        side=side,
                        first_seen=timestamp,
                        last_seen=timestamp
                    )
                    self.total_patterns_tracked += 1
    
    def _round_size_for_pattern(self, size: float) -> float:
        """Round size with much tighter tolerance"""
        # Use smaller tolerance for better pattern matching
        tolerance_band = max(size * self.size_tolerance, 0.1)  # At least 0.1 BTC
        return round(size / tolerance_band) * tolerance_band
    
    def _update_replenishment_time(self, pattern: OrderPattern, time_gap: float):
        """Update average replenishment time"""
        
        if pattern.avg_replenishment_time == 0:
            pattern.avg_replenishment_time = time_gap
        else:
            # Moving average
            pattern.avg_replenishment_time = (pattern.avg_replenishment_time * 0.7 + time_gap * 0.3)
    
    def _cleanup_old_patterns(self, current_time: datetime):
        """Remove old patterns"""
        
        cutoff_time = current_time - timedelta(seconds=self.max_pattern_age)
        
        # Clean bid patterns
        old_keys = [k for k, p in self.bid_patterns.items() if p.last_seen < cutoff_time]
        for k in old_keys:
            del self.bid_patterns[k]
        
        # Clean ask patterns  
        old_keys = [k for k, p in self.ask_patterns.items() if p.last_seen < cutoff_time]
        for k in old_keys:
            del self.ask_patterns[k]
    
    def _detect_icebergs(self, timestamp: datetime) -> List[IcebergDetection]:
        """Detect icebergs with STRICT criteria"""
        
        detections = []
        
        # Check bid patterns
        for pattern in self.bid_patterns.values():
            if self._is_iceberg_pattern(pattern):
                detection = self._create_detection(pattern, timestamp)
                if detection and not self._is_recent_duplicate(detection):
                    detections.append(detection)
        
        # Check ask patterns
        for pattern in self.ask_patterns.values():
            if self._is_iceberg_pattern(pattern):
                detection = self._create_detection(pattern, timestamp)
                if detection and not self._is_recent_duplicate(detection):
                    detections.append(detection)
        
        # Store detections to avoid duplicates
        self.recent_detections.extend(detections)
        self.total_detections += len(detections)
        
        return detections
    
    def _is_iceberg_pattern(self, pattern: OrderPattern) -> bool:
        """STRICT iceberg pattern validation"""
        
        # Must have minimum occurrences
        if pattern.occurrences < self.min_occurrences:
            return False
        
        # Must have minimum size
        if pattern.size < self.min_size:
            return False
        
        # Must have high iceberg score
        if pattern.iceberg_score < self.min_iceberg_score:
            return False
        
        # Must be recent activity
        if pattern.duration_seconds > self.max_pattern_age:
            return False
        
        # Must have reasonable replenishment timing
        if pattern.avg_replenishment_time > 0:
            if pattern.avg_replenishment_time < 0.5 or pattern.avg_replenishment_time > 30:
                return False  # Too fast (noise) or too slow (not iceberg)
        
        return True
    
    def _create_detection(self, pattern: OrderPattern, timestamp: datetime) -> Optional[IcebergDetection]:
        """Create detection with realistic size estimates"""
        
        confidence = pattern.iceberg_score / 100.0
        
        # Better size estimation based on pattern
        estimated_total = pattern.size * max(pattern.occurrences * 1.5, 10)  # Conservative estimate
        
        return IcebergDetection(
            pattern=pattern,
            confidence=confidence,
            estimated_total_size=estimated_total,
            detected_at=timestamp,
            side=pattern.side
        )
    
    def _is_recent_duplicate(self, detection: IcebergDetection) -> bool:
        """Check for recent duplicates"""
        
        for recent in self.recent_detections:
            if (abs(recent.pattern.price - detection.pattern.price) < 1.0 and  # Within $1
                recent.side == detection.side and
                (detection.detected_at - recent.detected_at).total_seconds() < 300):  # Within 5 minutes
                return True
        
        return False
    
    def get_active_icebergs(self) -> List[OrderPattern]:
        """Get currently active iceberg patterns"""
        
        active = []
        
        for pattern in self.bid_patterns.values():
            if self._is_iceberg_pattern(pattern):
                active.append(pattern)
        
        for pattern in self.ask_patterns.values():
            if self._is_iceberg_pattern(pattern):
                active.append(pattern)
        
        return sorted(active, key=lambda p: p.iceberg_score, reverse=True)
    
    def get_stats(self) -> dict:
        """Get detector statistics"""
        
        return {
            'total_patterns_tracked': self.total_patterns_tracked,
            'active_bid_patterns': len(self.bid_patterns),
            'active_ask_patterns': len(self.ask_patterns),
            'total_detections': self.total_detections,
            'noise_filtered': self.noise_filtered,
            'recent_detections': len(self.recent_detections)
        }


# Test with more realistic parameters
def test_realistic_detector():
    """Test with much stricter parameters"""
    
    detector = RealisticIcebergDetector(
        min_occurrences=5,
        min_iceberg_score=80,
        min_size=1.0,  # At least 1 BTC
        size_tolerance=0.02
    )
    
    print("🧪 Testing REALISTIC Iceberg Detector")
    print("=" * 50)
    print("📋 Parameters:")
    print(f"   Min occurrences: {detector.min_occurrences}")
    print(f"   Min score: {detector.min_iceberg_score}")
    print(f"   Min size: {detector.min_size} BTC")
    print(f"   Size tolerance: {detector.size_tolerance*100}%")
    print()
    
    # Simulate more realistic data
    test_data = []
    
    # Add many noise orders (should be filtered out)
    for i in range(10):
        noise_bids = [[50000 + i, 0.01 + i*0.001] for i in range(20)]  # Tiny sizes
        noise_asks = [[50010 + i, 0.01 + i*0.001] for i in range(20)]
        test_data.append((noise_bids, noise_asks))
    
    # Add a real iceberg pattern: 5 BTC orders at $50,000
    for i in range(8):  # 8 occurrences of same large order
        real_bids = [[50000, 5.0], [49999, 2.1], [49998, 1.8]]  # 5 BTC iceberg
        real_asks = [[50010, 0.5], [50011, 0.3]]  # Small asks
        test_data.append((real_bids, real_asks))
    
    for i, (bids, asks) in enumerate(test_data):
        print(f"📊 Processing update {i+1}")
        
        detections = detector.update_order_book(bids, asks)
        
        if detections:
            for detection in detections:
                print(f"🎯 REAL ICEBERG: {detection}")
        else:
            active = detector.get_active_icebergs()
            if active:
                print(f"   Building patterns... {len(active)} candidates")
            else:
                print("   No significant patterns yet")
    
    # Show final stats
    print(f"\n📈 Final Statistics:")
    stats = detector.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    test_realistic_detector()