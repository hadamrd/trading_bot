#!/usr/bin/env python3
"""
Producer Module - Handles Binance WebSocket connection and raw data processing
"""

import asyncio
import json
import websockets
from datetime import datetime
from typing import Dict, List, Callable, Optional
import threading


class BinanceProducer:
    """
    Handles connection to Binance WebSocket and produces clean market data
    """
    
    def __init__(self, symbol: str = "SUIUSDT"):
        self.symbol = symbol.upper()
        self.symbol_lower = symbol.lower()
        
        # Connection state
        self.running = False
        self.connection_status = 'disconnected'
        
        # Data subscribers
        self.subscribers = []
        
        # Raw data storage
        self.latest_orderbook = None
        self.update_count = 0
        
        # Background thread for WebSocket
        self.thread = None
        
    def subscribe(self, callback: Callable[[Dict], None]):
        """Subscribe to market data updates"""
        self.subscribers.append(callback)
        print(f"📡 Added subscriber: {callback.__name__ if hasattr(callback, '__name__') else 'callback'}")
    
    def unsubscribe(self, callback: Callable[[Dict], None]):
        """Unsubscribe from market data updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def start(self):
        """Start the WebSocket connection in background thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_websocket_loop, daemon=True)
            self.thread.start()
            print(f"📡 Started Binance producer for {self.symbol}")
    
    def stop(self):
        """Stop the WebSocket connection"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print(f"📡 Stopped Binance producer for {self.symbol}")
    
    def _run_websocket_loop(self):
        """Run asyncio loop in separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._websocket_loop())
        except Exception as e:
            print(f"❌ WebSocket loop error: {e}")
        finally:
            loop.close()
    
    async def _websocket_loop(self):
        """Main WebSocket connection loop with auto-reconnect"""
        ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol_lower}@depth20@100ms"
        
        while self.running:
            try:
                print(f"📡 Connecting to Binance: {self.symbol}")
                self.connection_status = 'connecting'
                self._notify_status_change()
                
                async with websockets.connect(ws_url) as websocket:
                    print(f"✅ Connected to {self.symbol} WebSocket")
                    self.connection_status = 'connected'
                    self._notify_status_change()
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        await self._process_message(message)
                        
            except Exception as e:
                print(f"❌ Connection error: {e}")
                self.connection_status = 'error'
                self._notify_status_change()
                
                if self.running:
                    await asyncio.sleep(5)  # Wait before reconnecting
    
    async def _process_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            raw_data = json.loads(message)
            
            if 'bids' in raw_data and 'asks' in raw_data:
                # Clean and structure the data
                market_data = self._structure_market_data(raw_data)
                
                # Store latest
                self.latest_orderbook = market_data
                self.update_count += 1
                
                # Notify all subscribers
                self._notify_subscribers(market_data)
                
        except json.JSONDecodeError:
            pass  # Ignore invalid JSON
        except Exception as e:
            print(f"⚠️  Error processing message: {e}")
    
    def _structure_market_data(self, raw_data: Dict) -> Dict:
        """Convert raw Binance data into clean, structured format"""
        timestamp = datetime.now()
        
        # Parse order book levels
        bids = [[float(b[0]), float(b[1])] for b in raw_data['bids'][:20]]
        asks = [[float(a[0]), float(a[1])] for a in raw_data['asks'][:20]]
        
        # Calculate basic metrics
        metrics = self._calculate_basic_metrics(bids, asks, timestamp)
        
        return {
            'timestamp': timestamp.isoformat(),
            'symbol': self.symbol,
            'connection_status': self.connection_status,
            'update_count': self.update_count,
            'orderbook': {
                'bids': bids,
                'asks': asks
            },
            'metrics': metrics,
            'raw_data': raw_data  # Keep for advanced processing if needed
        }
    
    def _calculate_basic_metrics(self, bids: List[List], asks: List[List], timestamp: datetime) -> Dict:
        """Calculate basic order book metrics"""
        
        if not bids or not asks:
            return {}
        
        # Basic calculations
        total_bid_value = sum(price * size for price, size in bids)
        total_ask_value = sum(price * size for price, size in asks)
        total_value = total_bid_value + total_ask_value
        
        if total_value == 0:
            return {}
        
        # Core metrics
        bid_ratio = total_bid_value / total_value
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid) * 100
        
        # Depth analysis
        bid_depth_5 = sum(price * size for price, size in bids[:5])
        ask_depth_5 = sum(price * size for price, size in asks[:5])
        
        # Largest orders
        largest_bid = max(bids, key=lambda x: x[0] * x[1])
        largest_ask = max(asks, key=lambda x: x[0] * x[1])
        
        return {
            'timestamp': timestamp.isoformat(),
            'mid_price': mid_price,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'spread_pct': spread_pct,
            'bid_ratio': bid_ratio,
            'ask_ratio': 1 - bid_ratio,
            'total_bid_value': total_bid_value,
            'total_ask_value': total_ask_value,
            'total_value': total_value,
            'bid_depth_5': bid_depth_5,
            'ask_depth_5': ask_depth_5,
            'largest_bid': {
                'price': largest_bid[0], 
                'size': largest_bid[1], 
                'value': largest_bid[0] * largest_bid[1]
            },
            'largest_ask': {
                'price': largest_ask[0], 
                'size': largest_ask[1], 
                'value': largest_ask[0] * largest_ask[1]
            },
            # Concentration metrics
            'bid_concentration_top5': bid_depth_5 / total_bid_value if total_bid_value > 0 else 0,
            'ask_concentration_top5': ask_depth_5 / total_ask_value if total_ask_value > 0 else 0,
        }
    
    def _notify_subscribers(self, market_data: Dict):
        """Notify all subscribers of new market data"""
        for callback in self.subscribers:
            try:
                callback(market_data)
            except Exception as e:
                print(f"⚠️  Error in subscriber callback: {e}")
    
    def _notify_status_change(self):
        """Notify subscribers of connection status change"""
        status_data = {
            'type': 'status_change',
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'connection_status': self.connection_status,
            'update_count': self.update_count
        }
        
        for callback in self.subscribers:
            try:
                callback(status_data)
            except Exception as e:
                pass  # Don't spam errors for status updates
    
    def get_latest_data(self) -> Optional[Dict]:
        """Get the latest market data"""
        return self.latest_orderbook
    
    def get_status(self) -> Dict:
        """Get current producer status"""
        return {
            'symbol': self.symbol,
            'connection_status': self.connection_status,
            'update_count': self.update_count,
            'subscribers': len(self.subscribers),
            'running': self.running
        }


# Test function
async def test_producer():
    """Test the producer standalone"""
    def on_data(data):
        if 'metrics' in data:
            metrics = data['metrics']
            print(f"📊 {data['symbol']} | ${metrics.get('mid_price', 0):.4f} | "
                  f"Bid: {metrics.get('bid_ratio', 0):.1%} | "
                  f"Updates: {data.get('update_count', 0)}")
    
    producer = BinanceProducer("SUIUSDT")
    producer.subscribe(on_data)
    producer.start()
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping producer test...")
        producer.stop()


if __name__ == "__main__":
    print("🧪 Testing Binance Producer")
    asyncio.run(test_producer())