#!/usr/bin/env python3
"""
WebSocket Server Bridge - Connects Python detection to React dashboard
"""

import asyncio
import json
import websockets
from datetime import datetime
import sys
from typing import Set

from trading_bot.smart_money_detection.fixed_iceberg_detector import IcebergDetection, RealisticIcebergDetector


class TradingDashboardServer:
    """WebSocket server that bridges Python detection to React dashboard"""
    
    def __init__(self, symbol: str = "wifusdt", port: int = 8765):
        self.symbol = symbol.lower()
        self.port = port
        
        # Iceberg detector
        self.detector = RealisticIcebergDetector(
            min_occurrences=4,        # Lower for demo
            min_iceberg_score=70,     # Lower for demo
            min_size=1000,           # 1000 WIF tokens (adjust per symbol)
            max_pattern_age=600,     # 10 minutes
            size_tolerance=0.03      # 3% tolerance
        )
        
        # Connected React clients
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Statistics
        self.total_updates = 0
        self.total_detections = 0
        self.last_order_book = None
        
    async def start_server(self):
        """Start the dashboard server"""
        
        print(f"🚀 Starting Trading Dashboard Server")
        print(f"   Symbol: {self.symbol.upper()}")
        print(f"   Port: {self.port}")
        print(f"   React dashboard will connect here")
        print("-" * 50)
        
        # Start WebSocket server for React clients
        server_task = websockets.serve(
            self.handle_client,
            "localhost",
            self.port
        )
        
        # Start Binance data collection
        binance_task = self.collect_binance_data()
        
        # Run both concurrently
        await asyncio.gather(
            server_task,
            binance_task
        )
    
    async def handle_client(self, websocket, path):
        """Handle React client connections"""
        
        print(f"📱 React client connected: {websocket.remote_address}")
        self.clients.add(websocket)
        
        try:
            # Send initial data
            if self.last_order_book:
                await self.send_to_client(websocket, {
                    'type': 'orderbook',
                    'data': self.last_order_book
                })
            
            # Keep connection alive
            async for message in websocket:
                # Handle any messages from React (if needed)
                pass
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            print(f"📱 React client disconnected")
    
    async def collect_binance_data(self):
        """Collect data from Binance and run detection"""
        
        url = f"wss://stream.binance.com:9443/ws/{self.symbol}@depth20@500ms"
        
        print(f"📡 Connecting to Binance: {url}")
        
        try:
            async with websockets.connect(url) as websocket:
                print(f"✅ Connected to Binance WebSocket")
                
                async for message in websocket:
                    await self.process_binance_message(message)
                    
        except Exception as e:
            print(f"❌ Binance connection error: {e}")
    
    async def process_binance_message(self, message: str):
        """Process Binance message and run detection"""
        
        try:
            data = json.loads(message)
            
            if 'bids' in data and 'asks' in data:
                self.total_updates += 1
                timestamp = datetime.now()
                
                # Run iceberg detection
                detections = self.detector.update_order_book(
                    data['bids'], 
                    data['asks'],
                    timestamp
                )
                
                # Prepare data for React dashboard
                dashboard_data = {
                    'type': 'update',
                    'timestamp': timestamp.isoformat(),
                    'update_count': self.total_updates,
                    'orderbook': {
                        'bids': data['bids'][:10],  # Top 10 levels
                        'asks': data['asks'][:10]
                    },
                    'detections': [self.detection_to_dict(d) for d in detections],
                    'active_patterns': [self.pattern_to_dict(p) for p in self.detector.get_active_icebergs()],
                    'stats': self.detector.get_stats()
                }
                
                self.last_order_book = dashboard_data
                
                # Send to all connected React clients
                await self.broadcast_to_clients(dashboard_data)
                
                # Log detections
                for detection in detections:
                    self.total_detections += 1
                    print(f"🎯 ICEBERG: {detection}")
        
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"⚠️  Error processing message: {e}")
    
    def detection_to_dict(self, detection: IcebergDetection) -> dict:
        """Convert detection to dict for JSON"""
        return {
            'id': f"{detection.side}_{detection.pattern.price}_{detection.detected_at.timestamp()}",
            'side': detection.side,
            'price': detection.pattern.price,
            'size': detection.pattern.size,
            'confidence': detection.confidence,
            'estimated_total': detection.estimated_total_size,
            'occurrences': detection.pattern.occurrences,
            'detected_at': detection.detected_at.isoformat()
        }
    
    def pattern_to_dict(self, pattern) -> dict:
        """Convert pattern to dict for JSON"""
        return {
            'side': pattern.side,
            'price': pattern.price,
            'size': pattern.size,
            'occurrences': pattern.occurrences,
            'score': pattern.iceberg_score,
            'frequency': pattern.frequency,
            'duration': pattern.duration_seconds,
            'replenishment_time': pattern.avg_replenishment_time
        }
    
    async def send_to_client(self, websocket, data):
        """Send data to specific client"""
        try:
            await websocket.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            pass
    
    async def broadcast_to_clients(self, data):
        """Broadcast data to all connected clients"""
        if self.clients:
            # Send to all clients concurrently
            await asyncio.gather(
                *[self.send_to_client(client, data) for client in self.clients],
                return_exceptions=True
            )


async def run_dashboard_server(symbol: str = "wifusdt"):
    """Run the dashboard server"""
    
    server = TradingDashboardServer(symbol)
    await server.start_server()


if __name__ == "__main__":
    import sys
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else "wifusdt"
    
    print(f"🖥️  Trading Dashboard Server")
    print(f"Symbol: {symbol.upper()}")
    print(f"WebSocket server will run on ws://localhost:8765")
    print(f"Start the React app to see the dashboard")
    print()
    
    try:
        asyncio.run(run_dashboard_server(symbol))
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")