#!/usr/bin/env python3
"""
Broker Module - Provides market data and basic analysis
"""

import asyncio
import json
import websockets
from typing import List, Callable

from trading_bot.order_book_trading.market_analyzer import MarketAnalyzer
from trading_bot.order_book_trading.models import MarketSituation


class BinanceBroker:
    """Simple Binance broker - focuses on getting market data"""
    
    def __init__(self):
        self.analyzer = MarketAnalyzer()
        self.callbacks: List[Callable[[MarketSituation], None]] = []
        self.running = False
    
    def add_callback(self, callback: Callable[[MarketSituation], None]):
        """Add callback for market situation updates"""
        self.callbacks.append(callback)
    
    async def start_stream(self, symbol: str):
        """Start streaming market data for symbol"""
        
        symbol_lower = symbol.lower()
        ws_url = f"wss://stream.binance.com:9443/ws/{symbol_lower}@depth20"
        
        print(f"📡 Connecting to {symbol} order book...")
        self.running = True
        
        try:
            async with websockets.connect(ws_url) as websocket:
                print(f"✅ Connected to {symbol} stream")
                
                async for message in websocket:
                    if not self.running:
                        break
                    
                    await self._process_message(symbol, message)
                    
        except Exception as e:
            print(f"❌ Stream error: {e}")
        finally:
            self.running = False
            print(f"🔌 Disconnected from {symbol}")
    
    async def _process_message(self, symbol: str, message: str):
        """Process incoming order book message"""
        
        try:
            data = json.loads(message)
            
            if 'bids' in data and 'asks' in data:
                # Analyze the market situation
                situation = self.analyzer.analyze(symbol, data['bids'], data['asks'])
                
                # Send to all callbacks
                for callback in self.callbacks:
                    try:
                        callback(situation)
                    except Exception as e:
                        print(f"❌ Callback error: {e}")
                        
        except json.JSONDecodeError:
            pass  # Ignore invalid JSON
        except Exception as e:
            print(f"❌ Message processing error: {e}")
    
    def stop(self):
        """Stop the stream"""
        self.running = False


# Example usage
async def test_broker():
    """Test the broker module"""
    
    def on_market_update(situation: MarketSituation):
        """Handle market situation updates"""
        
        print(f"📊 {situation.symbol}: ${situation.price:.2f} "
              f"(spread: {situation.spread_pct:.3f}%)")
        
        if situation.large_bid_wall:
            print(f"   🟢 Large bid wall: ${situation.large_bid_wall.value:,.0f} @ ${situation.large_bid_wall.price:.2f}")
        
        if situation.large_ask_wall:
            print(f"   🔴 Large ask wall: ${situation.large_ask_wall.value:,.0f} @ ${situation.large_ask_wall.price:.2f}")
        
        print(f"   📈 Bid pressure: {situation.bid_pressure:.1%}")
        
        if situation.has_buy_opportunity:
            print(f"   🎯 BUY OPPORTUNITY detected")
        elif situation.has_sell_opportunity:
            print(f"   🎯 SELL OPPORTUNITY detected")
    
    # Create broker and add callback
    broker = BinanceBroker()
    broker.add_callback(on_market_update)
    
    # Start streaming
    try:
        await broker.start_stream("BTCUSDT")
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        broker.stop()


if __name__ == "__main__":
    asyncio.run(test_broker())