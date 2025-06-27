from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from pybit.unified_trading import HTTP
import websocket
import json
import threading
import time
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class BybitWebSocketFixed:
    def __init__(self):
        # Correct Bybit WebSocket endpoints
        # self.ws_url = "wss://stream-testnet.bybit.com/v5/public/linear"  # For testnet
        self.ws_url = "wss://stream.bybit.com/v5/public/linear"  # For mainnet
        
        self.ws = None
        self.subscribed_symbols = set()
        self.price_data = {}
        self.connected = False
        
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            # print(f"DEBUG - Received message: {json.dumps(data, indent=2)}")
            
            # Handle pong responses
            if data.get('op') == 'pong':
                print("Received pong")
                return
                
            # Handle subscription responses
            if data.get('success') == True:
                print(f"Subscription successful: {data}")
                return
                
            # Handle subscription errors
            if data.get('success') == False:
                print(f"Subscription failed: {data}")
                return
            
            # Handle ticker data
            if 'topic' in data and 'data' in data:
                topic = data.get('topic', '')
                
                if 'tickers' in topic:
                    ticker_info = data['data']
                    
                    # Check if data is a dict (single ticker) or list
                    if isinstance(ticker_info, dict):
                        self.process_ticker_data(ticker_info)
                    elif isinstance(ticker_info, list):
                        for ticker in ticker_info:
                            self.process_ticker_data(ticker)
                            
        except Exception as e:
            print(f"WebSocket message error: {e}")
            print(f"Raw message was: {message}")
    
    def process_ticker_data(self, ticker_data):
        try:
            symbol = ticker_data.get('symbol')
            last_price = ticker_data.get('lastPrice')
            price_24h_pcnt = ticker_data.get('price24hPcnt', '0')
            volume_24h = ticker_data.get('volume24h', '0')
            
            if symbol and last_price:
                price = float(last_price)
                change_24h = float(price_24h_pcnt) * 100
                
                self.price_data[symbol] = {
                    'price': price,
                    'change_24h': change_24h,
                    'volume': float(volume_24h)
                }
                
                print(f"Price update: {symbol} = ${price:.4f} ({change_24h:+.2f}%)")
                
                # Emit to frontend
                socketio.emit('price_update', {
                    'symbol': symbol,
                    'price': price,
                    'change_24h': round(change_24h, 2)
                })
                
        except Exception as e:
            print(f"Error processing ticker data: {e}")
            print(f"Ticker data was: {ticker_data}")
    
    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.connected = False
        # Reconnect after 5 seconds
        print("Reconnecting in 5 seconds...")
        threading.Timer(5.0, self.connect).start()
    
    def on_open(self, ws):
        print("WebSocket connection opened!")
        self.connected = True
        
        # Send ping to keep connection alive
        def send_ping():
            while self.connected:
                try:
                    if self.ws:
                        ping_msg = {"op": "ping"}
                        self.ws.send(json.dumps(ping_msg))
                        print("Sent ping")
                    time.sleep(20)  # Ping every 20 seconds
                except:
                    break
        
        threading.Thread(target=send_ping, daemon=True).start()
        
        # Subscribe to default symbols after a short delay
        def delayed_subscribe():
            time.sleep(1)  # Wait 1 second before subscribing
            default_symbols = ['APTUSDT']
            for symbol in default_symbols:
                self.subscribe_ticker(symbol)
        
        threading.Thread(target=delayed_subscribe, daemon=True).start()
    
    def connect(self):
        try:
            print("Connecting to Bybit WebSocket...")
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            # Run forever in separate thread
            def run_ws():
                self.ws.run_forever()
            
            threading.Thread(target=run_ws, daemon=True).start()
            
        except Exception as e:
            print(f"Failed to connect: {e}")
    
    def subscribe_ticker(self, symbol):
        if not self.connected or not self.ws:
            print(f"Cannot subscribe to {symbol} - not connected")
            return False
            
        if symbol in self.subscribed_symbols:
            print(f"Already subscribed to {symbol}")
            return True
        
        try:
            subscribe_msg = {
                "op": "subscribe",
                "args": [f"tickers.{symbol}"]
            }
            
            print(f"Subscribing to {symbol}...")
            self.ws.send(json.dumps(subscribe_msg))
            self.subscribed_symbols.add(symbol)
            return True
            
        except Exception as e:
            print(f"Failed to subscribe to {symbol}: {e}")
            return False
    
    def get_current_price(self, symbol):
        price_info = self.price_data.get(symbol)
        return price_info.get('price') if price_info else None

# Initialize WebSocket
bybit_ws = BybitWebSocketFixed()

class BybitRiskManager:
    def __init__(self):
        self.api_key = os.getenv('BYBIT_API_KEY', 'your_api_key_here')
        self.api_secret = os.getenv('BYBIT_SECRET', 'your_secret_here')
        self.testnet = True
        
        self.session = HTTP(
            testnet=self.testnet,
            api_key=self.api_key,
            api_secret=self.api_secret,
        )
    
    def get_current_price(self, symbol):
        # Try WebSocket first
        ws_price = bybit_ws.get_current_price(symbol)
        if ws_price:
            return ws_price
        
        # Fallback to REST API
        try:
            ticker = self.session.get_tickers(category="linear", symbol=symbol)
            return float(ticker['result']['list'][0]['lastPrice'])
        except Exception as e:
            print(f"REST API error: {e}")
            return None
    
    def get_balance(self):
        try:
            balance = self.session.get_wallet_balance(accountType="UNIFIED")
            for coin in balance['result']['list'][0]['coin']:
                if coin['coin'] == 'USDT':
                    return float(coin['walletBalance'])
            return 0
        except Exception as e:
            print(f"Balance error: {e}")
            return 0

trader = BybitRiskManager()

@app.route('/')
def index():
    balance = trader.get_balance()
    return render_template('margin_trader.html', balance=balance)

@socketio.on('connect')
def handle_connect():
    print('Frontend client connected')
    emit('connected', {'data': 'Connected to price feed'})

@socketio.on('subscribe_ticker')
def handle_subscribe(data):
    symbol = data['symbol'].upper()
    success = bybit_ws.subscribe_ticker(symbol)
    emit('subscribed', {'symbol': symbol, 'success': success})

@app.route('/get_price/<symbol>')
def get_price(symbol):
    price = trader.get_current_price(symbol)
    return jsonify({"price": price})

@app.route('/place_trade', methods=['POST'])
def place_trade():
    # Your existing trade logic here
    return jsonify({"success": True, "message": "Trade simulation"})

if __name__ == '__main__':
    print("Starting Bybit Margin Trader...")
    
    # Start WebSocket connection
    bybit_ws.connect()
    
    # Give WebSocket time to connect
    time.sleep(2)
    
    # Run Flask app
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)