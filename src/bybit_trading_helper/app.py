from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import time

from bybit_trading_helper.bybit_ws_client import BybitWS
from bybit_trading_helper.risk_manager import BybitRiskManager

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


# Initialize WebSocket
bybit_ws = BybitWS(socketio)
trader = BybitRiskManager(bybit_ws)

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
    data = request.json
    
    result = trader.place_order_with_risk_management(
        symbol=data['symbol'],
        side=data['side'],
        leverage=data['leverage'],
        sl_percent=data['sl_percent'],
        tp_percent=data['tp_percent'],
        risk_percent=data['risk_percent']
    )
    
    return jsonify(result)

if __name__ == '__main__':
    print("Starting Bybit Margin Trader...")
    
    # Start WebSocket connection
    bybit_ws.connect()
    
    # Give WebSocket time to connect
    time.sleep(2)
    
    # Run Flask app
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
