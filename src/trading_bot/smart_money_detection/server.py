#!/usr/bin/env python3
"""
Flask Trading Dashboard with Server-Sent Events
Much better than WebSocket approach!
"""

import asyncio
import json
import os
import websockets
from datetime import datetime
import sys
from pathlib import Path
import threading
import queue
import time

from flask import Flask, render_template_string, Response, jsonify, request, render_template
from flask_cors import CORS

from trading_bot.smart_money_detection.fixed_iceberg_detector import IcebergDetection, RealisticIcebergDetector

app = Flask(__name__)
CORS(app)  # Enable CORS for development

dir_path = os.path.dirname(os.path.realpath(__file__))

class TradingDataStream:
    """Manages real-time trading data stream"""
    
    def __init__(self, symbol: str = "wifusdt"):
        self.symbol = symbol.lower()
        
        # Iceberg detector with adjustable parameters
        self.detector = RealisticIcebergDetector(
            min_occurrences=4,
            min_iceberg_score=70,
            min_size=1000,  # Adjust based on symbol
            max_pattern_age=600,
            size_tolerance=0.03
        )
        
        # Data storage
        self.latest_data = {
            'orderbook': {'bids': [], 'asks': []},
            'active_patterns': [],
            'detections': [],
            'stats': {},
            'update_count': 0,
            'timestamp': None,
            'connection_status': 'disconnected'
        }
        
        # SSE clients
        self.sse_clients = []
        
        # Data queue for thread communication
        self.data_queue = queue.Queue()
        
        # Start background data collection
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the data collection thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_binance_loop, daemon=True)
            self.thread.start()
            print(f"📡 Started data collection for {self.symbol.upper()}")
    
    def stop(self):
        """Stop data collection"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _run_binance_loop(self):
        """Run the asyncio loop in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._collect_binance_data())
        finally:
            loop.close()
    
    async def _collect_binance_data(self):
        """Collect data from Binance WebSocket"""
        url = f"wss://stream.binance.com:9443/ws/{self.symbol}@depth20@500ms"
        
        while self.running:
            try:
                print(f"📡 Connecting to Binance: {url}")
                self.latest_data['connection_status'] = 'connecting'
                self._notify_clients()
                
                async with websockets.connect(url) as websocket:
                    print(f"✅ Connected to Binance WebSocket")
                    self.latest_data['connection_status'] = 'connected'
                    self._notify_clients()
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        await self._process_message(message)
                        
            except Exception as e:
                print(f"❌ Binance connection error: {e}")
                self.latest_data['connection_status'] = 'error'
                self._notify_clients()
                await asyncio.sleep(5)  # Wait before reconnecting
    
    async def _process_message(self, message: str):
        """Process Binance WebSocket message"""
        try:
            data = json.loads(message)
            
            if 'bids' in data and 'asks' in data:
                timestamp = datetime.now()
                
                # Run iceberg detection
                detections = self.detector.update_order_book(
                    data['bids'], 
                    data['asks'],
                    timestamp
                )
                
                # Update latest data
                self.latest_data.update({
                    'orderbook': {
                        'bids': data['bids'][:10],
                        'asks': data['asks'][:10]
                    },
                    'active_patterns': [self._pattern_to_dict(p) for p in self.detector.get_active_icebergs()],
                    'detections': [self._detection_to_dict(d) for d in detections],
                    'stats': self.detector.get_stats(),
                    'update_count': self.latest_data['update_count'] + 1,
                    'timestamp': timestamp.isoformat(),
                    'connection_status': 'connected'
                })
                
                # Notify SSE clients
                self._notify_clients()
                
                # Log detections
                for detection in detections:
                    print(f"🎯 ICEBERG DETECTED: {detection}")
        
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"⚠️  Error processing message: {e}")
    
    def _detection_to_dict(self, detection: IcebergDetection) -> dict:
        """Convert detection to dict"""
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
    
    def _pattern_to_dict(self, pattern) -> dict:
        """Convert pattern to dict"""
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
    
    def _notify_clients(self):
        """Notify all SSE clients with new data"""
        data_json = json.dumps(self.latest_data)
        for client_queue in self.sse_clients:
            try:
                client_queue.put(data_json, timeout=0.1)
            except queue.Full:
                # Remove slow clients
                self.sse_clients.remove(client_queue)
    
    def add_sse_client(self) -> queue.Queue:
        """Add new SSE client"""
        client_queue = queue.Queue(maxsize=10)
        self.sse_clients.append(client_queue)
        
        # Send current data immediately
        client_queue.put(json.dumps(self.latest_data))
        return client_queue
    
    def remove_sse_client(self, client_queue: queue.Queue):
        """Remove SSE client"""
        if client_queue in self.sse_clients:
            self.sse_clients.remove(client_queue)
    
    def update_parameters(self, params: dict):
        """Update detector parameters"""
        if 'min_size' in params:
            self.detector.min_size = float(params['min_size'])
        if 'min_occurrences' in params:
            self.detector.min_occurrences = int(params['min_occurrences'])
        if 'min_iceberg_score' in params:
            self.detector.min_iceberg_score = float(params['min_iceberg_score'])
        if 'size_tolerance' in params:
            self.detector.size_tolerance = float(params['size_tolerance'])
        
        print(f"📋 Updated parameters: {params}")


# Global data stream instance
data_stream = None

@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    htm_path = os.path.join(dir_path, "dashboard.html")
    return render_template(htm_path)

@app.route('/stream')
def stream():
    """Server-Sent Events endpoint"""
    def event_stream():
        client_queue = data_stream.add_sse_client()
        try:
            while True:
                try:
                    # Get data from queue with timeout
                    data = client_queue.get(timeout=30)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    # Send heartbeat
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        except GeneratorExit:
            data_stream.remove_sse_client(client_queue)
    
    return Response(event_stream(), content_type='text/event-stream')

@app.route('/api/status')
def status():
    """Get current status"""
    return jsonify(data_stream.latest_data)

@app.route('/api/parameters', methods=['GET', 'POST'])
def parameters():
    """Get or update detector parameters"""
    if request.method == 'POST':
        params = request.json
        data_stream.update_parameters(params)
        return jsonify({'status': 'updated', 'params': params})
    else:
        return jsonify({
            'min_size': data_stream.detector.min_size,
            'min_occurrences': data_stream.detector.min_occurrences,
            'min_iceberg_score': data_stream.detector.min_iceberg_score,
            'size_tolerance': data_stream.detector.size_tolerance,
            'max_pattern_age': data_stream.detector.max_pattern_age
        })

@app.route('/api/stats')
def stats():
    """Get detailed statistics"""
    return jsonify({
        'detector_stats': data_stream.detector.get_stats(),
        'active_patterns': len(data_stream.detector.get_active_icebergs()),
        'symbol': data_stream.symbol.upper(),
        'uptime': time.time() - app.start_time if hasattr(app, 'start_time') else 0
    })

def run_flask_app(symbol: str = "wifusdt", port: int = 5000):
    """Run the Flask dashboard application"""
    global data_stream
    
    print(f"🚀 Starting Flask Trading Dashboard")
    print(f"   Symbol: {symbol.upper()}")
    print(f"   URL: http://localhost:{port}")
    print("-" * 50)
    
    # Initialize data stream
    data_stream = TradingDataStream(symbol)
    app.start_time = time.time()
    
    # Start data collection
    data_stream.start()
    
    try:
        # Run Flask app
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        if data_stream:
            data_stream.stop()

if __name__ == "__main__":
    import sys
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else "wifusdt"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    
    run_flask_app(symbol, port)