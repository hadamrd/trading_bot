#!/usr/bin/env python3
"""
Simple Order Book Trader
Finds trading opportunities from live order book data
"""

import asyncio
import json
import websockets
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class OrderLevel:
    price: float
    size: float
    value: float
    
    @property
    def is_significant(self) -> bool:
        """Is this a significant order (>$50k)?"""
        return self.value > 50000


@dataclass
class TradingSignal:
    signal_type: str  # 'buy_wall', 'sell_wall', 'imbalance', 'breakout'
    price: float
    confidence: float  # 0-1
    reasoning: str
    timestamp: datetime


class OrderBookAnalyzer:
    """Analyzes order book for trading opportunities"""
    
    def __init__(self):
        self.previous_bids = []
        self.previous_asks = []
        
    def analyze(self, bids: List[OrderLevel], asks: List[OrderLevel]) -> List[TradingSignal]:
        """Analyze order book and find trading signals"""
        
        signals = []
        
        # 1. Check for significant buy/sell walls
        signals.extend(self._detect_walls(bids, asks))
        
        # 2. Check for order book imbalance  
        signals.extend(self._detect_imbalance(bids, asks))
        
        # 3. Check for wall removal (breakout signal)
        signals.extend(self._detect_wall_removal(bids, asks))
        
        # Store for next comparison
        self.previous_bids = bids.copy()
        self.previous_asks = asks.copy()
        
        return signals
    
    def _detect_walls(self, bids: List[OrderLevel], asks: List[OrderLevel]) -> List[TradingSignal]:
        """Detect significant buy/sell walls"""
        signals = []
        
        # Find largest buy orders (potential support)
        large_bids = [bid for bid in bids if bid.is_significant]
        if large_bids:
            strongest_bid = max(large_bids, key=lambda x: x.value)
            
            # Generate buy signal near the wall
            signals.append(TradingSignal(
                signal_type='buy_wall',
                price=strongest_bid.price * 1.0001,  # Slightly above the wall
                confidence=min(strongest_bid.value / 100000, 1.0),  # Higher value = higher confidence
                reasoning=f"Large buy wall: ${strongest_bid.value:,.0f} at ${strongest_bid.price:.4f}",
                timestamp=datetime.now()
            ))
        
        # Find largest sell orders (potential resistance)
        large_asks = [ask for ask in asks if ask.is_significant]
        if large_asks:
            strongest_ask = max(large_asks, key=lambda x: x.value)
            
            # Generate sell signal near the wall (or avoid buying)
            signals.append(TradingSignal(
                signal_type='sell_wall',
                price=strongest_ask.price * 0.9999,  # Slightly below the wall
                confidence=min(strongest_ask.value / 100000, 1.0),
                reasoning=f"Large sell wall: ${strongest_ask.value:,.0f} at ${strongest_ask.price:.4f}",
                timestamp=datetime.now()
            ))
        
        return signals
    
    def _detect_imbalance(self, bids: List[OrderLevel], asks: List[OrderLevel]) -> List[TradingSignal]:
        """Detect order book imbalance (more buying vs selling pressure)"""
        signals = []
        
        # Calculate total value on each side (top 5 levels)
        bid_total = sum(bid.value for bid in bids[:5])
        ask_total = sum(ask.value for ask in asks[:5])
        
        total_value = bid_total + ask_total
        if total_value == 0:
            return signals
        
        # Calculate imbalance ratio
        bid_ratio = bid_total / total_value
        ask_ratio = ask_total / total_value
        
        # Strong imbalance signals
        if bid_ratio > 0.7:  # 70%+ buying pressure
            mid_price = (bids[0].price + asks[0].price) / 2
            signals.append(TradingSignal(
                signal_type='imbalance',
                price=mid_price,
                confidence=bid_ratio,
                reasoning=f"Strong buy pressure: {bid_ratio:.1%} vs {ask_ratio:.1%}",
                timestamp=datetime.now()
            ))
        
        elif ask_ratio > 0.7:  # 70%+ selling pressure
            mid_price = (bids[0].price + asks[0].price) / 2
            signals.append(TradingSignal(
                signal_type='imbalance',
                price=mid_price * 0.998,  # Slightly lower entry for sell pressure
                confidence=ask_ratio,
                reasoning=f"Strong sell pressure: {ask_ratio:.1%} vs {bid_ratio:.1%}",
                timestamp=datetime.now()
            ))
        
        return signals
    
    def _detect_wall_removal(self, bids: List[OrderLevel], asks: List[OrderLevel]) -> List[TradingSignal]:
        """Detect when large walls disappear (potential breakout)"""
        signals = []
        
        if not self.previous_bids or not self.previous_asks:
            return signals
        
        # Find significant walls that disappeared
        prev_large_bids = [b for b in self.previous_bids if b.is_significant]
        current_prices = {b.price for b in bids}
        
        for prev_bid in prev_large_bids:
            if prev_bid.price not in current_prices:
                # Large bid wall disappeared - potential breakdown
                signals.append(TradingSignal(
                    signal_type='breakout',
                    price=prev_bid.price * 0.9995,  # Below broken support
                    confidence=0.8,
                    reasoning=f"Buy wall removed: ${prev_bid.value:,.0f} @ ${prev_bid.price:.4f}",
                    timestamp=datetime.now()
                ))
        
        # Same for ask walls
        prev_large_asks = [a for a in self.previous_asks if a.is_significant]
        current_ask_prices = {a.price for a in asks}
        
        for prev_ask in prev_large_asks:
            if prev_ask.price not in current_ask_prices:
                # Large ask wall disappeared - potential breakout up
                signals.append(TradingSignal(
                    signal_type='breakout',
                    price=prev_ask.price * 1.0005,  # Above broken resistance
                    confidence=0.8,
                    reasoning=f"Sell wall removed: ${prev_ask.value:,.0f} @ ${prev_ask.price:.4f}",
                    timestamp=datetime.now()
                ))
        
        return signals


class VirtualTrader:
    """Executes virtual trades based on signals"""
    
    def __init__(self, initial_balance: float = 1000):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = []
        self.open_orders = []
        
    def place_order(self, signal: TradingSignal, current_price: float) -> Optional[str]:
        """Place a virtual limit order based on signal"""
        
        # Only trade high confidence signals
        if signal.confidence < 0.6:
            return None
        
        # Calculate position size (risk 2% of balance)
        risk_amount = self.balance * 0.02
        
        if signal.signal_type in ['buy_wall', 'imbalance']:
            # Calculate stop loss (1% below entry)
            stop_loss = signal.price * 0.99
            risk_per_share = signal.price - stop_loss
            
            if risk_per_share > 0:
                shares = risk_amount / risk_per_share
                total_cost = shares * signal.price
                
                if total_cost <= self.balance * 0.1:  # Max 10% per trade
                    order_id = f"BUY_{len(self.open_orders)}"
                    
                    order = {
                        'id': order_id,
                        'type': 'buy',
                        'price': signal.price,
                        'shares': shares,
                        'total': total_cost,
                        'stop_loss': stop_loss,
                        'signal': signal,
                        'timestamp': datetime.now()
                    }
                    
                    self.open_orders.append(order)
                    return order_id
        
        return None
    
    def check_fills(self, current_bid: float, current_ask: float) -> List[str]:
        """Check if any orders should be filled"""
        filled_orders = []
        
        for order in self.open_orders.copy():
            filled = False
            
            if order['type'] == 'buy' and current_ask <= order['price']:
                # Buy order filled
                self.balance -= order['total']
                
                position = {
                    'shares': order['shares'],
                    'entry_price': order['price'],
                    'stop_loss': order['stop_loss'],
                    'entry_time': order['timestamp'],
                    'signal': order['signal']
                }
                
                self.positions.append(position)
                filled = True
                
                print(f"✅ BUY FILLED: {order['shares']:.3f} @ ${order['price']:.4f}")
                print(f"   Reason: {order['signal'].reasoning}")
            
            if filled:
                self.open_orders.remove(order)
                filled_orders.append(order['id'])
        
        return filled_orders
    
    def check_exits(self, current_bid: float) -> List[dict]:
        """Check if any positions should be closed"""
        closed_positions = []
        
        for position in self.positions.copy():
            # Check stop loss
            if current_bid <= position['stop_loss']:
                # Stop loss hit
                proceeds = position['shares'] * current_bid
                self.balance += proceeds
                
                profit = proceeds - (position['shares'] * position['entry_price'])
                
                print(f"🛑 STOP LOSS: {position['shares']:.3f} @ ${current_bid:.4f}")
                print(f"   P&L: ${profit:+.2f}")
                
                closed_positions.append(position)
                self.positions.remove(position)
            
            # Check take profit (simple 2% target)
            elif current_bid >= position['entry_price'] * 1.02:
                # Take profit
                proceeds = position['shares'] * current_bid
                self.balance += proceeds
                
                profit = proceeds - (position['shares'] * position['entry_price'])
                
                print(f"💰 TAKE PROFIT: {position['shares']:.3f} @ ${current_bid:.4f}")
                print(f"   P&L: ${profit:+.2f}")
                
                closed_positions.append(position)
                self.positions.remove(position)
        
        return closed_positions
    
    def get_stats(self) -> dict:
        """Get current trading statistics"""
        total_value = self.balance
        
        # Add value of open positions (at current market price would be needed)
        return {
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'profit_loss': self.balance - self.initial_balance,
            'profit_pct': ((self.balance / self.initial_balance) - 1) * 100,
            'open_positions': len(self.positions),
            'open_orders': len(self.open_orders)
        }


class OrderBookTrader:
    """Main trader that combines analysis and execution"""
    
    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol.lower()
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@depth20"
        
        self.analyzer = OrderBookAnalyzer()
        self.trader = VirtualTrader()
        
        self.update_count = 0
        
    async def start_trading(self):
        """Start live trading based on order book"""
        
        print(f"🤖 Starting Order Book Trader for {self.symbol.upper()}")
        print(f"💰 Initial Balance: ${self.trader.initial_balance:,.2f}")
        print(f"🎯 Strategy: Follow walls, detect imbalances")
        print("-" * 60)
        
        async with websockets.connect(self.ws_url) as websocket:
            async for message in websocket:
                await self._process_update(message)
    
    async def _process_update(self, message):
        """Process each order book update"""
        
        try:
            data = json.loads(message)
            
            if 'bids' in data and 'asks' in data:
                # Convert to our format
                bids = self._convert_levels(data['bids'])
                asks = self._convert_levels(data['asks'])
                
                # Analyze for signals
                signals = self.analyzer.analyze(bids, asks)
                
                # Get current market prices
                current_bid = bids[0].price if bids else 0
                current_ask = asks[0].price if asks else 0
                
                # Check for order fills
                self.trader.check_fills(current_bid, current_ask)
                
                # Check for position exits
                self.trader.check_exits(current_bid)
                
                # Process new signals
                for signal in signals:
                    if signal.confidence >= 0.7:  # High confidence signals only
                        print(f"\n🎯 SIGNAL: {signal.signal_type.upper()}")
                        print(f"   Price: ${signal.price:.4f}")
                        print(f"   Confidence: {signal.confidence:.1%}")
                        print(f"   Reason: {signal.reasoning}")
                        
                        order_id = self.trader.place_order(signal, (current_bid + current_ask) / 2)
                        if order_id:
                            print(f"   📋 Placed order: {order_id}")
                
                # Show stats every 50 updates
                self.update_count += 1
                if self.update_count % 50 == 0:
                    self._show_stats(current_bid, current_ask)
                
        except json.JSONDecodeError:
            pass
    
    def _convert_levels(self, levels) -> List[OrderLevel]:
        """Convert Binance levels to our format"""
        result = []
        
        for level in levels:
            price = float(level[0])
            size = float(level[1])
            value = price * size
            
            if size > 0:  # Only non-zero levels
                result.append(OrderLevel(price, size, value))
        
        return result
    
    def _show_stats(self, current_bid: float, current_ask: float):
        """Show current trading statistics"""
        stats = self.trader.get_stats()
        spread = ((current_ask - current_bid) / current_bid) * 100
        
        print(f"\n📊 TRADING STATS (Update #{self.update_count})")
        print(f"   Price: ${current_bid:.4f} - ${current_ask:.4f} ({spread:.3f}% spread)")
        print(f"   Balance: ${stats['balance']:.2f}")
        print(f"   P&L: ${stats['profit_loss']:+.2f} ({stats['profit_pct']:+.2f}%)")
        print(f"   Positions: {stats['open_positions']} | Orders: {stats['open_orders']}")
        print("-" * 60)


async def main():
    """Main function"""
    
    print("🤖 Simple Order Book Trader")
    print("Finds opportunities in live order book data")
    print()
    
    symbol = input("Enter symbol (default BTCUSDT): ").strip().upper()
    if not symbol:
        symbol = "BTCUSDT"
    
    print(f"\n🚀 Starting trader for {symbol}...")
    print("Press Ctrl+C to stop and see final results")
    print()
    
    trader = OrderBookTrader(symbol)
    
    try:
        await trader.start_trading()
    except KeyboardInterrupt:
        print("\n🛑 Trading stopped")
        
        # Show final results
        stats = trader.trader.get_stats()
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Initial Balance: ${stats['initial_balance']:,.2f}")
        print(f"   Final Balance: ${stats['balance']:,.2f}")
        print(f"   Total P&L: ${stats['profit_loss']:+,.2f}")
        print(f"   Return: {stats['profit_pct']:+.2f}%")
        
        if stats['profit_loss'] > 0:
            print("✅ Profitable strategy!")
        else:
            print("📉 Strategy needs improvement")


if __name__ == "__main__":
    asyncio.run(main())