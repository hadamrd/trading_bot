#!/usr/bin/env python3
"""
Real-Time Order Book Imbalance Detector
Connects to Binance WebSocket and detects money flow patterns in real-time
"""

import asyncio
import json
import websockets
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
import statistics


@dataclass
class OrderBookLevel:
    """Single order book level"""
    price: float
    size: float
    
    @property
    def value(self) -> float:
        return self.price * self.size


@dataclass
class ImbalanceSignal:
    """Detected imbalance signal"""
    timestamp: datetime
    signal_type: str  # 'STRONG_BUY', 'STRONG_SELL', 'MODERATE_BUY', 'MODERATE_SELL', 'NEUTRAL'
    strength: float  # 0-1, how strong the signal is
    bid_ratio: float  # % of total liquidity on bid side
    total_bid_value: float
    total_ask_value: float
    price: float
    reason: str
    confidence: float


class OrderBookImbalanceDetector:
    """
    Real-time order book imbalance detector
    
    Key signals we detect:
    1. Bid/Ask ratio imbalances (>70% one side = strong signal)
    2. Sudden liquidity changes (walls appearing/disappearing)
    3. Depth concentration (liquidity bunched at specific levels)
    4. Flow acceleration (imbalance getting stronger)
    """
    
    def __init__(self, 
                 symbol: str = "SUIUSDT",
                 levels_to_analyze: int = 20,
                 history_length: int = 50,
                 strong_threshold: float = 0.75,
                 moderate_threshold: float = 0.65):
        
        self.symbol = symbol.upper()
        self.levels_to_analyze = levels_to_analyze
        self.history_length = history_length
        self.strong_threshold = strong_threshold
        self.moderate_threshold = moderate_threshold
        
        # Data storage
        self.current_orderbook = None
        self.imbalance_history = deque(maxlen=history_length)
        self.signal_history = deque(maxlen=100)
        
        # Statistics
        self.update_count = 0
        self.signals_detected = 0
        self.last_signal_time = None
        
        # State tracking
        self.current_trend = "NEUTRAL"
        self.trend_strength = 0.0
        self.consecutive_signals = 0
        
    async def start_monitoring(self):
        """Start real-time monitoring"""
        
        print(f"🚀 Starting Real-Time Order Book Imbalance Detector")
        print(f"📊 Symbol: {self.symbol}")
        print(f"🔍 Analyzing top {self.levels_to_analyze} levels")
        print(f"⚡ Strong signal threshold: {self.strong_threshold:.0%}")
        print("-" * 60)
        
        # WebSocket URL for Binance order book
        ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@depth20@100ms"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                print(f"✅ Connected to Binance WebSocket")
                print(f"🔄 Waiting for order book data...\n")
                
                async for message in websocket:
                    await self._process_orderbook_update(message)
                    
        except Exception as e:
            print(f"❌ Connection error: {e}")
            print("🔄 Retrying in 5 seconds...")
            await asyncio.sleep(5)
            await self.start_monitoring()  # Retry
    
    async def _process_orderbook_update(self, message: str):
        """Process incoming order book update"""
        
        try:
            data = json.loads(message)
            
            if 'bids' in data and 'asks' in data:
                
                # Parse order book
                bids = [OrderBookLevel(float(b[0]), float(b[1])) for b in data['bids'][:self.levels_to_analyze]]
                asks = [OrderBookLevel(float(a[0]), float(a[1])) for a in data['asks'][:self.levels_to_analyze]]
                
                # Calculate imbalance
                imbalance_data = self._calculate_imbalance(bids, asks)
                
                # Detect signals
                signals = self._detect_signals(imbalance_data)
                
                # Store history
                self.imbalance_history.append(imbalance_data)
                self.update_count += 1
                
                # Display updates
                if self.update_count % 10 == 0:  # Every 1 second (100ms * 10)
                    self._display_status(imbalance_data)
                
                # Handle signals
                for signal in signals:
                    self._handle_signal(signal)
                    
        except json.JSONDecodeError:
            pass  # Ignore invalid JSON
        except Exception as e:
            print(f"⚠️  Error processing update: {e}")
    
    def _calculate_imbalance(self, bids: List[OrderBookLevel], asks: List[OrderBookLevel]) -> Dict:
        """Calculate comprehensive imbalance metrics"""
        
        if not bids or not asks:
            return {}
        
        # Basic calculations
        total_bid_value = sum(level.value for level in bids)
        total_ask_value = sum(level.value for level in asks)
        total_value = total_bid_value + total_ask_value
        
        if total_value == 0:
            return {}
        
        bid_ratio = total_bid_value / total_value
        ask_ratio = total_ask_value / total_value
        
        # Top-of-book analysis
        best_bid = bids[0]
        best_ask = asks[0]
        spread = best_ask.price - best_bid.price
        spread_pct = (spread / best_bid.price) * 100
        mid_price = (best_bid.price + best_ask.price) / 2
        
        # Depth analysis
        bid_depth_5 = sum(level.value for level in bids[:5])
        ask_depth_5 = sum(level.value for level in asks[:5])
        
        # Concentration analysis (how much liquidity is in top 25% of levels)
        top_quarter = max(1, self.levels_to_analyze // 4)
        bid_concentration = sum(level.value for level in bids[:top_quarter]) / total_bid_value
        ask_concentration = sum(level.value for level in asks[:top_quarter]) / total_ask_value
        
        return {
            'timestamp': datetime.now(),
            'mid_price': mid_price,
            'spread_pct': spread_pct,
            'total_bid_value': total_bid_value,
            'total_ask_value': total_ask_value,
            'bid_ratio': bid_ratio,
            'ask_ratio': ask_ratio,
            'bid_depth_5': bid_depth_5,
            'ask_depth_5': ask_depth_5,
            'bid_concentration': bid_concentration,
            'ask_concentration': ask_concentration,
            'largest_bid': max(bids, key=lambda x: x.value),
            'largest_ask': max(asks, key=lambda x: x.value),
        }
    
    def _detect_signals(self, imbalance_data: Dict) -> List[ImbalanceSignal]:
        """Detect trading signals from imbalance data"""
        
        if not imbalance_data:
            return []
        
        signals = []
        bid_ratio = imbalance_data['bid_ratio']
        
        # Signal 1: Strong Imbalance
        if bid_ratio >= self.strong_threshold:
            signals.append(ImbalanceSignal(
                timestamp=imbalance_data['timestamp'],
                signal_type='STRONG_BUY',
                strength=bid_ratio,
                bid_ratio=bid_ratio,
                total_bid_value=imbalance_data['total_bid_value'],
                total_ask_value=imbalance_data['total_ask_value'],
                price=imbalance_data['mid_price'],
                reason=f"Strong bid dominance: {bid_ratio:.1%}",
                confidence=min((bid_ratio - 0.5) * 2, 1.0)
            ))
        
        elif bid_ratio <= (1 - self.strong_threshold):
            signals.append(ImbalanceSignal(
                timestamp=imbalance_data['timestamp'],
                signal_type='STRONG_SELL',
                strength=1 - bid_ratio,
                bid_ratio=bid_ratio,
                total_bid_value=imbalance_data['total_bid_value'],
                total_ask_value=imbalance_data['total_ask_value'],
                price=imbalance_data['mid_price'],
                reason=f"Strong ask dominance: {1-bid_ratio:.1%}",
                confidence=min(((1-bid_ratio) - 0.5) * 2, 1.0)
            ))
        
        # Signal 2: Moderate Imbalance (only if we have recent trend)
        elif bid_ratio >= self.moderate_threshold and self._is_trend_building('BUY'):
            signals.append(ImbalanceSignal(
                timestamp=imbalance_data['timestamp'],
                signal_type='MODERATE_BUY',
                strength=bid_ratio,
                bid_ratio=bid_ratio,
                total_bid_value=imbalance_data['total_bid_value'],
                total_ask_value=imbalance_data['total_ask_value'],
                price=imbalance_data['mid_price'],
                reason=f"Building buy pressure: {bid_ratio:.1%}",
                confidence=(bid_ratio - 0.5) * 2
            ))
        
        elif bid_ratio <= (1 - self.moderate_threshold) and self._is_trend_building('SELL'):
            signals.append(ImbalanceSignal(
                timestamp=imbalance_data['timestamp'],
                signal_type='MODERATE_SELL',
                strength=1 - bid_ratio,
                bid_ratio=bid_ratio,
                total_bid_value=imbalance_data['total_bid_value'],
                total_ask_value=imbalance_data['total_ask_value'],
                price=imbalance_data['mid_price'],
                reason=f"Building sell pressure: {1-bid_ratio:.1%}",
                confidence=((1-bid_ratio) - 0.5) * 2
            ))
        
        # Signal 3: Large Order Detection
        large_order_signal = self._detect_large_orders(imbalance_data)
        if large_order_signal:
            signals.append(large_order_signal)
        
        # Signal 4: Sudden Imbalance Change
        momentum_signal = self._detect_momentum_change(imbalance_data)
        if momentum_signal:
            signals.append(momentum_signal)
        
        return signals
    
    def _detect_large_orders(self, imbalance_data: Dict) -> Optional[ImbalanceSignal]:
        """Detect unusually large orders"""
        
        largest_bid = imbalance_data['largest_bid']
        largest_ask = imbalance_data['largest_ask']
        total_value = imbalance_data['total_bid_value'] + imbalance_data['total_ask_value']
        
        # If a single order is >20% of total book value, it's significant
        large_threshold = total_value * 0.2
        
        if largest_bid.value > large_threshold:
            return ImbalanceSignal(
                timestamp=imbalance_data['timestamp'],
                signal_type='LARGE_BID_WALL',
                strength=largest_bid.value / total_value,
                bid_ratio=imbalance_data['bid_ratio'],
                total_bid_value=imbalance_data['total_bid_value'],
                total_ask_value=imbalance_data['total_ask_value'],
                price=largest_bid.price,
                reason=f"Large bid wall: ${largest_bid.value:,.0f} @ ${largest_bid.price:.4f}",
                confidence=0.8
            )
        
        if largest_ask.value > large_threshold:
            return ImbalanceSignal(
                timestamp=imbalance_data['timestamp'],
                signal_type='LARGE_ASK_WALL',
                strength=largest_ask.value / total_value,
                bid_ratio=imbalance_data['bid_ratio'],
                total_bid_value=imbalance_data['total_bid_value'],
                total_ask_value=imbalance_data['total_ask_value'],
                price=largest_ask.price,
                reason=f"Large ask wall: ${largest_ask.value:,.0f} @ ${largest_ask.price:.4f}",
                confidence=0.8
            )
        
        return None
    
    def _detect_momentum_change(self, imbalance_data: Dict) -> Optional[ImbalanceSignal]:
        """Detect sudden changes in imbalance momentum"""
        
        if len(self.imbalance_history) < 5:
            return None
        
        # Get recent imbalance ratios
        recent_ratios = [data['bid_ratio'] for data in list(self.imbalance_history)[-5:]]
        recent_ratios.append(imbalance_data['bid_ratio'])
        
        # Calculate momentum (rate of change)
        momentum = recent_ratios[-1] - recent_ratios[0]
        
        # Significant momentum change (>15% in 5 updates)
        if abs(momentum) > 0.15:
            direction = "BULLISH" if momentum > 0 else "BEARISH"
            
            return ImbalanceSignal(
                timestamp=imbalance_data['timestamp'],
                signal_type=f'MOMENTUM_{direction}',
                strength=abs(momentum),
                bid_ratio=imbalance_data['bid_ratio'],
                total_bid_value=imbalance_data['total_bid_value'],
                total_ask_value=imbalance_data['total_ask_value'],
                price=imbalance_data['mid_price'],
                reason=f"Rapid momentum shift: {momentum:+.1%}",
                confidence=min(abs(momentum) * 3, 0.9)
            )
        
        return None
    
    def _is_trend_building(self, direction: str) -> bool:
        """Check if a trend is building in the specified direction"""
        
        if len(self.imbalance_history) < 3:
            return False
        
        recent_ratios = [data['bid_ratio'] for data in list(self.imbalance_history)[-3:]]
        
        if direction == 'BUY':
            # Check if bid ratios are increasing
            return all(recent_ratios[i] < recent_ratios[i+1] for i in range(len(recent_ratios)-1))
        else:
            # Check if bid ratios are decreasing (ask dominance increasing)
            return all(recent_ratios[i] > recent_ratios[i+1] for i in range(len(recent_ratios)-1))
    
    def _handle_signal(self, signal: ImbalanceSignal):
        """Handle detected signal"""
        
        self.signals_detected += 1
        self.signal_history.append(signal)
        self.last_signal_time = signal.timestamp
        
        # Color coding for different signal types
        colors = {
            'STRONG_BUY': '🟢',
            'STRONG_SELL': '🔴', 
            'MODERATE_BUY': '🟡',
            'MODERATE_SELL': '🟠',
            'LARGE_BID_WALL': '🔵',
            'LARGE_ASK_WALL': '🟣',
            'MOMENTUM_BULLISH': '⚡',
            'MOMENTUM_BEARISH': '⚡'
        }
        
        color = colors.get(signal.signal_type, '⚪')
        
        print(f"\n{color} {signal.signal_type} SIGNAL DETECTED")
        print(f"   Price: ${signal.price:.4f}")
        print(f"   Strength: {signal.strength:.1%}")
        print(f"   Confidence: {signal.confidence:.1%}")
        print(f"   Reason: {signal.reason}")
        print(f"   Bid/Ask Ratio: {signal.bid_ratio:.1%} / {1-signal.bid_ratio:.1%}")
        print(f"   Time: {signal.timestamp.strftime('%H:%M:%S')}")
        print("-" * 60)
    
    def _display_status(self, imbalance_data: Dict):
        """Display current status"""
        
        bid_ratio = imbalance_data['bid_ratio']
        price = imbalance_data['mid_price']
        spread = imbalance_data['spread_pct']
        
        # Status indicator
        if bid_ratio >= self.strong_threshold:
            status = "🟢 STRONG BUY PRESSURE"
        elif bid_ratio <= (1 - self.strong_threshold):
            status = "🔴 STRONG SELL PRESSURE"
        elif bid_ratio >= self.moderate_threshold:
            status = "🟡 MODERATE BUY"
        elif bid_ratio <= (1 - self.moderate_threshold):
            status = "🟠 MODERATE SELL"
        else:
            status = "⚪ BALANCED"
        
        print(f"📊 {self.symbol} | ${price:.4f} | {status} | Ratio: {bid_ratio:.1%} | Spread: {spread:.3f}% | Signals: {self.signals_detected}")
    
    def get_statistics(self) -> Dict:
        """Get detector statistics"""
        
        recent_signals = [s for s in self.signal_history if s.timestamp > datetime.now() - timedelta(minutes=5)]
        
        return {
            'total_updates': self.update_count,
            'total_signals': self.signals_detected,
            'signals_per_minute': len(recent_signals) / 5 if recent_signals else 0,
            'last_signal': self.last_signal_time.strftime('%H:%M:%S') if self.last_signal_time else None,
            'current_trend': self.current_trend,
            'avg_bid_ratio': statistics.mean([data['bid_ratio'] for data in self.imbalance_history]) if self.imbalance_history else 0.5
        }


# Main execution
async def main():
    """Run the imbalance detector"""
    
    # You can customize these parameters
    detector = OrderBookImbalanceDetector(
        symbol="SUIUSDT",           # Change to any Binance symbol
        levels_to_analyze=15,        # How deep to analyze the book
        strong_threshold=0.75,       # 75%+ imbalance = strong signal
        moderate_threshold=0.65      # 65%+ imbalance = moderate signal
    )
    
    try:
        await detector.start_monitoring()
    except KeyboardInterrupt:
        print("\n🛑 Detector stopped by user")
        stats = detector.get_statistics()
        print(f"\n📈 Final Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")


if __name__ == "__main__":
    print("🚀 Real-Time Order Book Imbalance Detector")
    print("Press Ctrl+C to stop\n")
    
    asyncio.run(main())