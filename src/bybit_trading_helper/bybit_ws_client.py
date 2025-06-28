from flask_socketio import SocketIO
import websocket
import json
import threading
import time

class BybitWS:
    def __init__(self, socketio: SocketIO):
        self.ws_url = "wss://stream.bybit.com/v5/public/linear"  # For mainnet
        self.ws = None
        self.subscribed_symbols = set()
        self.price_data = {}
        self.connected = False
        self.socketio = socketio
        
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            
            # Handle pong responses (remove debug print)
            if data.get('op') == 'pong':
                return
                
            # Handle subscription responses (remove debug print)
            if data.get('success') == True:
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
    
    def process_ticker_data(self, ticker_data):
        try:
            symbol = ticker_data.get('symbol')
            last_price = ticker_data.get('lastPrice')
            price_24h_pcnt = ticker_data.get('price24hPcnt', '0')
            volume_24h = ticker_data.get('volume24h', '0')
            
            if symbol and last_price:
                price = float(last_price)
                change_24h = float(price_24h_pcnt) * 100
                
                # Only update if price actually changed
                old_price = self.price_data.get(symbol, {}).get('price', 0)
                if abs(price - old_price) > 0.0001:  # Prevent spam
                    self.price_data[symbol] = {
                        'price': price,
                        'change_24h': change_24h,
                        'volume': float(volume_24h)
                    }
                    
                    print(f"Price update: {symbol} = ${price:.4f} ({change_24h:+.2f}%)")
                    
                    # Emit to frontend
                    self.socketio.emit('price_update', {
                        'symbol': symbol,
                        'price': price,
                        'change_24h': round(change_24h, 2)
                    })
                
        except Exception as e:
            print(f"Error processing ticker data: {e}")
    
    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.connected = False
        print("Reconnecting in 5 seconds...")
        threading.Timer(5.0, self.connect).start()
    
    def on_open(self, ws):
        print("WebSocket connection opened!")
        self.connected = True
        
        # Send ping to keep connection alive (no debug print)
        def send_ping():
            while self.connected:
                try:
                    if self.ws:
                        ping_msg = {"op": "ping"}
                        self.ws.send(json.dumps(ping_msg))
                    time.sleep(20)
                except:
                    break
        
        threading.Thread(target=send_ping, daemon=True).start()
        
        # Subscribe to default symbols
        def delayed_subscribe():
            time.sleep(1)
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
