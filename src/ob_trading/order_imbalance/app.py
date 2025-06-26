#!/usr/bin/env python3
"""
Order Book Dashboard - Modular Architecture
Orchestrates producer, signal processor, and web interface
"""

import json
import queue
import os

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS

from ob_trading.order_imbalance.producer import BinanceProducer
from ob_trading.order_imbalance.signal_processor import SmartSignalProcessor, TradingSignal


app = Flask(__name__)
CORS(app)


class DashboardOrchestrator:
    """
    Main orchestrator that coordinates all components
    """
    
    def __init__(self, symbol: str = "SUIUSDT"):
        self.symbol = symbol.upper()
        
        # Initialize components
        self.producer = BinanceProducer(symbol)
        self.signal_processor = SmartSignalProcessor(symbol)
        
        # Dashboard state
        self.dashboard_data = {
            'symbol': self.symbol,
            'connection_status': 'disconnected',
            'latest_metrics': {},
            'latest_orderbook': {'bids': [], 'asks': []},
            'signals': [],
            'update_count': 0,
            'last_update': None,
            'statistics': {}
        }
        
        # SSE clients
        self.sse_clients = []
        
        # Wire everything together
        self._setup_connections()
        
    def _setup_connections(self):
        """Connect all the components"""
        
        # Producer sends data to signal processor
        self.producer.subscribe(self.signal_processor.process_market_data)
        
        # Producer also sends data to dashboard for display
        self.producer.subscribe(self._update_dashboard_data)
        
        # Signal processor sends signals to dashboard
        self.signal_processor.subscribe_to_signals(self._handle_new_signal)
        
        print("🔗 All components connected")
    
    def start(self):
        """Start all components"""
        print(f"🚀 Starting dashboard orchestrator for {self.symbol}")
        
        # Start the producer (which feeds everyone else)
        self.producer.start()
        
        print("✅ All components started")
    
    def stop(self):
        """Stop all components"""
        print("🛑 Stopping all components...")
        self.producer.stop()
        print("✅ All components stopped")
    
    def _update_dashboard_data(self, market_data: dict):
        """Update dashboard with latest market data"""
        
        if 'type' in market_data and market_data['type'] == 'status_change':
            # Handle status updates
            self.dashboard_data['connection_status'] = market_data['connection_status']
            self._notify_sse_clients()
            return
        
        if 'metrics' not in market_data:
            return
        
        # Update dashboard data
        self.dashboard_data.update({
            'connection_status': market_data.get('connection_status', 'connected'),
            'latest_metrics': market_data['metrics'],
            'latest_orderbook': market_data['orderbook'],
            'update_count': market_data.get('update_count', 0),
            'last_update': market_data.get('timestamp'),
            'statistics': self.signal_processor.get_statistics()
        })
        
        # Notify SSE clients
        self._notify_sse_clients()
    
    def _handle_new_signal(self, signal: TradingSignal):
        """Handle new high-quality signal from processor"""
        
        # Convert signal to dict for JSON serialization
        signal_dict = {
            'timestamp': signal.timestamp.isoformat(),
            'type': signal.signal_type,
            'symbol': signal.symbol,
            'price': signal.price,
            'strength': signal.strength,
            'confidence': signal.confidence,
            'reason': signal.reason,
            'duration': signal.duration,
            'metadata': signal.metadata
        }
        
        # Add to signals list (keep last 50)
        self.dashboard_data['signals'].append(signal_dict)
        self.dashboard_data['signals'] = self.dashboard_data['signals'][-50:]
        
        # Log the signal
        print(f"🚨 {signal.signal_type} | ${signal.price:.4f} | "
              f"Confidence: {signal.confidence:.1%} | "
              f"Duration: {signal.duration:.1f}s")
        
        # Notify SSE clients immediately
        self._notify_sse_clients()
    
    def add_sse_client(self) -> queue.Queue:
        """Add new SSE client"""
        client_queue = queue.Queue(maxsize=10)
        self.sse_clients.append(client_queue)
        
        # Send current data immediately
        try:
            client_queue.put(json.dumps(self.dashboard_data), timeout=1)
        except queue.Full:
            pass
        
        return client_queue
    
    def remove_sse_client(self, client_queue: queue.Queue):
        """Remove SSE client"""
        if client_queue in self.sse_clients:
            self.sse_clients.remove(client_queue)
    
    def _notify_sse_clients(self):
        """Notify all SSE clients with updated data"""
        data_json = json.dumps(self.dashboard_data)
        clients_to_remove = []
        
        for client_queue in self.sse_clients:
            try:
                client_queue.put(data_json, timeout=0.1)
            except queue.Full:
                clients_to_remove.append(client_queue)
        
        # Remove slow clients
        for client in clients_to_remove:
            self.sse_clients.remove(client)
    
    def update_config(self, new_config: dict):
        """Update configuration for all components"""
        
        # Update signal processor config
        processor_config = {}
        config_mapping = {
            'strong_threshold': 'strong_imbalance_threshold',
            'moderate_threshold': 'moderate_imbalance_threshold', 
            'wall_threshold': 'large_wall_threshold',
            'persistence_seconds': 'min_persistence_seconds',
            'cooldown_seconds': 'signal_cooldown_seconds',
            'min_confidence': 'min_confidence'
        }
        
        for web_key, processor_key in config_mapping.items():
            if web_key in new_config:
                processor_config[processor_key] = new_config[web_key]
        
        if processor_config:
            self.signal_processor.update_config(processor_config)
        
        print(f"📋 Config updated: {new_config}")
    
    def get_status(self) -> dict:
        """Get comprehensive system status"""
        return {
            'symbol': self.symbol,
            'producer_status': self.producer.get_status(),
            'processor_stats': self.signal_processor.get_statistics(),
            'dashboard_stats': {
                'sse_clients': len(self.sse_clients),
                'signals_count': len(self.dashboard_data['signals']),
                'last_update': self.dashboard_data.get('last_update')
            }
        }


# Global orchestrator
orchestrator = None

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/stream')
def stream():
    """Server-Sent Events endpoint"""
    def event_stream():
        client_queue = orchestrator.add_sse_client()
        try:
            while True:
                try:
                    data = client_queue.get(timeout=30)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        except GeneratorExit:
            orchestrator.remove_sse_client(client_queue)
    
    return Response(event_stream(), content_type='text/event-stream')

@app.route('/api/status')
def status():
    """Get comprehensive system status"""
    return jsonify(orchestrator.get_status())

@app.route('/api/data')
def data():
    """Get current dashboard data"""
    return jsonify(orchestrator.dashboard_data)

@app.route('/api/config', methods=['GET', 'POST'])
def config():
    """Get or update configuration"""
    if request.method == 'POST':
        new_config = request.json
        orchestrator.update_config(new_config)
        return jsonify({'status': 'updated', 'config': new_config})
    else:
        # Return current signal processor config
        return jsonify({
            'strong_threshold': orchestrator.signal_processor.strong_imbalance_threshold,
            'moderate_threshold': orchestrator.signal_processor.moderate_imbalance_threshold,
            'wall_threshold': orchestrator.signal_processor.large_wall_threshold,
            'persistence_seconds': orchestrator.signal_processor.min_persistence_seconds,
            'cooldown_seconds': orchestrator.signal_processor.signal_cooldown_seconds,
            'min_confidence': orchestrator.signal_processor.min_confidence
        })

@app.route('/api/signals/recent')
def recent_signals():
    """Get recent signals"""
    minutes = request.args.get('minutes', 10, type=int)
    signals = orchestrator.signal_processor.get_recent_signals(minutes)
    
    return jsonify([{
        'timestamp': signal.timestamp.isoformat(),
        'type': signal.signal_type,
        'price': signal.price,
        'strength': signal.strength,
        'confidence': signal.confidence,
        'reason': signal.reason,
        'duration': signal.duration
    } for signal in signals])

def run_dashboard(symbol: str = "SUIUSDT", port: int = 5000):
    """Run the modular dashboard"""
    global orchestrator
    
    print(f"🚀 Starting Modular Order Book Dashboard")
    print(f"📊 Symbol: {symbol}")
    print(f"🌐 URL: http://localhost:{port}")
    print(f"🧩 Architecture:")
    print(f"   📡 Producer: Binance WebSocket connection")
    print(f"   🧠 Processor: Smart signal detection") 
    print(f"   🌐 Dashboard: Web interface")
    print("-" * 60)
    
    # Check if template exists
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'dashboard.html')
    if not os.path.exists(template_path):
        print(f"⚠️  Template not found: {template_path}")
        print(f"   Please create the templates directory and add dashboard.html")
        return
    
    # Initialize orchestrator
    orchestrator = DashboardOrchestrator(symbol)
    orchestrator.start()
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        if orchestrator:
            orchestrator.stop()

if __name__ == "__main__":
    import sys
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SUIUSDT"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    
    run_dashboard(symbol, port)