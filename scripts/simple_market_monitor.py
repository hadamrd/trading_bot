#!/usr/bin/env python3
"""
Simple Buy Levels Monitor
Shows live price levels where people want to buy (bid side)
"""

import asyncio
import json
import websockets
from decimal import Decimal


class SimpleBuyMonitor:
    """Simple monitor for buy interest levels"""
    
    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol.lower()
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@depth20"
        
    async def start_monitoring(self):
        """Start monitoring buy levels"""
        
        print(f"🔌 Connecting to {self.symbol.upper()} order book...")
        print(f"📊 Watching for buy interest levels...")
        print("-" * 60)
        
        async with websockets.connect(self.ws_url) as websocket:
            async for message in websocket:
                await self._process_message(message)
    
    async def _process_message(self, message):
        """Process incoming order book data"""
        
        try:
            data = json.loads(message)
            
            if 'bids' in data:
                self._analyze_buy_levels(data['bids'])
                
        except json.JSONDecodeError:
            print("❌ Error parsing message")
    
    def _analyze_buy_levels(self, bids):
        """Analyze bid levels to find where people want to buy"""
        
        print(f"\n🟢 BUY INTEREST LEVELS ({self.symbol.upper()}):")
        print(f"{'Price':<12} {'Size':<12} {'Value':<15} {'Interest'}")
        print("-" * 60)
        
        # Convert bids to our format and sort by interest
        buy_levels = []
        
        for bid in bids[:10]:  # Top 10 levels
            price = Decimal(bid[0])
            size = Decimal(bid[1])
            value = price * size
            
            buy_levels.append({
                'price': float(price),
                'size': float(size), 
                'value': float(value)
            })
        
        # Sort by value (price * size) to see biggest interest
        buy_levels.sort(key=lambda x: x['value'], reverse=True)
        
        # Show top 5 levels with most buying interest
        for i, level in enumerate(buy_levels[:5], 1):
            price = level['price']
            size = level['size']
            value = level['value']
            
            # Simple interest indicator
            if value > 100000:  # $100k+
                interest = "🔥🔥🔥 HIGH"
            elif value > 50000:  # $50k+
                interest = "🔥🔥 MEDIUM"
            elif value > 20000:  # $20k+
                interest = "🔥 LOW"
            else:
                interest = "• normal"
            
            # Show more decimal places for lower-priced tokens
            if price < 1.0:
                price_str = f"${price:.4f}"  # 4 decimals for sub-$1 tokens
            elif price < 10.0:
                price_str = f"${price:.3f}"  # 3 decimals for $1-$10
            else:
                price_str = f"${price:,.2f}"  # 2 decimals for $10+
            
            print(f"{price_str:<12} {size:<11.3f} ${value:<14,.0f} {interest}")
        
        print("-" * 60)


async def main():
    """Main function"""
    
    print("🎯 Simple Buy Levels Monitor")
    print("Shows where people want to buy right now")
    print()
    
    # Ask user for symbol
    symbol = input("Enter symbol (default BTCUSDT): ").strip().upper()
    if not symbol:
        symbol = "BTCUSDT"
    
    print(f"\n📊 Starting monitor for {symbol}...")
    print("Press Ctrl+C to stop")
    print()
    
    monitor = SimpleBuyMonitor(symbol)
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")


if __name__ == "__main__":
    asyncio.run(main())