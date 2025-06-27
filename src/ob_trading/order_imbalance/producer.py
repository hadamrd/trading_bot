#!/usr/bin/env python3
"""
Enhanced Producer - Order Book + Individual Trades for Delta Analysis
"""

import asyncio
import json
import websockets
from datetime import datetime
from typing import Dict, List, Callable, Optional
import threading
from collections import deque


class BinanceProducer:
    """
    Enhanced producer that captures both order book snapshots AND individual trades
    """
    
    def __init__(self, symbol: str = "SUIUSDT"):
        self.symbol = symbol.upper()
        self.symbol_lower = symbol.lower()
        
        # Connection state
        self.running = False
        self.connection_status = 'disconnected'
        
        # Data subscribers
        self.subscribers = []
        
        # Order book data storage
        self.latest_orderbook = None
        self.update_count = 0
        
        # NEW: Individual trades storage
        self.recent_trades = deque(maxlen=1000)  # Store last 1000 trades
        self.trade_count = 0
        
        # Background threads
        self.orderbook_thread = None
        self.trades_thread = None
        
    def subscribe(self, callback: Callable[[Dict], None]):
        """Subscribe to market data updates"""
        self.subscribers.append(callback)
        print(f"📡 Added subscriber: {callback.__name__ if hasattr(callback, '__name__') else 'callback'}")
    
    def unsubscribe(self, callback: Callable[[Dict], None]):
        """Unsubscribe from market data updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def start(self):
        """Start both order book and trade WebSocket connections"""
        if not self.running:
            self.running = True
            
            # Start order book stream
            self.orderbook_thread = threading.Thread(
                target=self._run_orderbook_websocket, 
                daemon=True
            )
            self.orderbook_thread.start()
            
            # Start trades stream  
            self.trades_thread = threading.Thread(
                target=self._run_trades_websocket,
                daemon=True
            )
            self.trades_thread.start()
            
            print(f"📡 Started enhanced producer for {self.symbol} (Order Book + Trades)")
    
    def stop(self):
        """Stop both WebSocket connections"""
        self.running = False
        if self.orderbook_thread:
            self.orderbook_thread.join(timeout=5)
        if self.trades_thread:
            self.trades_thread.join(timeout=5)
        print(f"📡 Stopped enhanced producer for {self.symbol}")
    
    def _run_orderbook_websocket(self):
        """Run order book WebSocket in separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._orderbook_websocket_loop())
        except Exception as e:
            print(f"❌ Order book WebSocket error: {e}")
        finally:
            loop.close()
    
    def _run_trades_websocket(self):
        """Run trades WebSocket in separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._trades_websocket_loop())
        except Exception as e:
            print(f"❌ Trades WebSocket error: {e}")
        finally:
            loop.close()
    
    async def _orderbook_websocket_loop(self):
        """Order book WebSocket connection loop"""
        ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol_lower}@depth20@100ms"
        
        while self.running:
            try:
                print(f"📊 Connecting to {self.symbol} order book...")
                self.connection_status = 'connecting'
                self._notify_status_change()
                
                async with websockets.connect(ws_url) as websocket:
                    print(f"✅ Connected to {self.symbol} order book")
                    self.connection_status = 'connected'
                    self._notify_status_change()
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        await self._process_orderbook_message(message)
                        
            except Exception as e:
                print(f"❌ Order book connection error: {e}")
                self.connection_status = 'error'
                self._notify_status_change()
                
                if self.running:
                    await asyncio.sleep(5)
    
    async def _trades_websocket_loop(self):
        """Individual trades WebSocket connection loop"""
        ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol_lower}@trade"
        
        while self.running:
            try:
                print(f"💱 Connecting to {self.symbol} trades...")
                
                async with websockets.connect(ws_url) as websocket:
                    print(f"✅ Connected to {self.symbol} trades")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        await self._process_trade_message(message)
                        
            except Exception as e:
                print(f"❌ Trades connection error: {e}")
                
                if self.running:
                    await asyncio.sleep(5)
    
    async def _process_orderbook_message(self, message: str):
        """Process order book WebSocket message"""
        try:
            raw_data = json.loads(message)
            
            if 'bids' in raw_data and 'asks' in raw_data:
                # Create market data with recent trades included
                market_data = self._structure_market_data(raw_data)
                
                # Add recent trades for delta analysis
                market_data['trades'] = list(self.recent_trades)[-100:]  # Last 100 trades
                
                # Store latest
                self.latest_orderbook = market_data
                self.update_count += 1
                
                # Notify subscribers
                self._notify_subscribers(market_data)
                
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"⚠️  Error processing order book: {e}")
    
    async def _process_trade_message(self, message: str):
        """Process individual trade WebSocket message"""
        try:
            raw_trade = json.loads(message)
            
            # Structure trade data for delta calculator
            trade_data = {
                'price': float(raw_trade['p']),
                'quantity': float(raw_trade['q']),
                'timestamp': int(raw_trade['T']),
                'is_buyer_maker': raw_trade['m'],  # True if buyer is maker (sell trade)
                'trade_id': raw_trade['t']
            }
            
            # Store in recent trades
            self.recent_trades.append(trade_data)
            self.trade_count += 1
            
            # Optional: Send trade-only updates for real-time delta
            # (You can enable this for even more responsive delta updates)
            # self._notify_subscribers({
            #     'type': 'trade_update',
            #     'trade': trade_data,
            #     'trade_count': self.trade_count
            # })
                
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"⚠️  Error processing trade: {e}")
    
    def _structure_market_data(self, raw_data: Dict) -> Dict:
        """Convert raw order book data into structured format (existing logic)"""
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
            'trade_count': self.trade_count,  # NEW: Add trade count
            'orderbook': {
                'bids': bids,
                'asks': asks
            },
            'metrics': metrics,
            'raw_data': raw_data
        }
    
    def _calculate_basic_metrics(self, bids: List[List], asks: List[List], timestamp: datetime) -> Dict:
        """Calculate basic order book metrics (existing logic)"""
        
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
            'update_count': self.update_count,
            'trade_count': self.trade_count
        }
        
        for callback in self.subscribers:
            try:
                callback(status_data)
            except Exception as e:
                pass
    
    def get_latest_data(self) -> Optional[Dict]:
        """Get the latest market data"""
        return self.latest_orderbook
    
    def get_recent_trades(self, count: int = 100) -> List[Dict]:
        """Get recent trades for delta analysis"""
        return list(self.recent_trades)[-count:]
    
    def get_status(self) -> Dict:
        """Get current producer status"""
        return {
            'symbol': self.symbol,
            'connection_status': self.connection_status,
            'update_count': self.update_count,
            'trade_count': self.trade_count,  # NEW
            'recent_trades_count': len(self.recent_trades),  # NEW
            'subscribers': len(self.subscribers),
            'running': self.running
        }