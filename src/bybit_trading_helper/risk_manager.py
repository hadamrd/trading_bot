from pybit.unified_trading import HTTP
import os

from bybit_trading_helper.bybit_ws_client import BybitWS

class BybitRiskManager:
    def __init__(self, bybit_ws: BybitWS):
        self.api_key = os.getenv('BYBIT_API_KEY', 'your_api_key_here')
        self.api_secret = os.getenv('BYBIT_SECRET', 'your_secret_here')
        self.testnet = True
        self.bybit_ws = bybit_ws
        
        self.session = HTTP(
            testnet=self.testnet,
            api_key=self.api_key,
            api_secret=self.api_secret,
        )
    
    def get_current_price(self, symbol):
        # Try WebSocket first
        ws_price = self.bybit_ws.get_current_price(symbol)
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
    
    def place_order_with_risk_management(self, symbol, side, leverage, sl_percent, tp_percent, risk_percent):
        """
        Place order with proper risk management
        """
        try:
            # Get current price
            current_price = self.get_current_price(symbol)
            if not current_price:
                return {"success": False, "error": "Cannot get current price"}
            
            # Get balance
            balance = self.get_balance()
            if balance <= 0:
                return {"success": False, "error": "Insufficient balance"}
            
            # Calculate risk amount
            risk_amount = balance * (risk_percent / 100)
            
            # Calculate SL and TP prices
            if side == 'Buy':
                sl_price = current_price * (1 - sl_percent / 100)
                tp_price = current_price * (1 + tp_percent / 100)
            else:
                sl_price = current_price * (1 + sl_percent / 100)
                tp_price = current_price * (1 - tp_percent / 100)
            
            # Calculate position size based on risk
            price_change = abs(current_price - sl_price) / current_price
            position_value = risk_amount / price_change
            position_size = position_value / current_price
            
            # Round position size to appropriate decimals
            position_size = round(position_size, 6)
            
            # Set leverage first
            try:
                self.session.set_leverage(
                    category="linear",
                    symbol=symbol,
                    buyLeverage=str(leverage),
                    sellLeverage=str(leverage)
                )
            except Exception as e:
                print(f"Leverage setting warning: {e}")
            
            # Place main order
            order_result = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(position_size),
                timeInForce="IOC"
            )
            
            if order_result.get('retCode') != 0:
                return {"success": False, "error": f"Order failed: {order_result.get('retMsg')}"}
            
            order_id = order_result['result']['orderId']
            
            # Place stop loss
            sl_result = self.session.place_order(
                category="linear",
                symbol=symbol,
                side="Sell" if side == "Buy" else "Buy",
                orderType="Market",
                qty=str(position_size),
                stopLoss=str(round(sl_price, 4)),
                timeInForce="GTC"
            )
            
            # Place take profit
            tp_result = self.session.place_order(
                category="linear",
                symbol=symbol,
                side="Sell" if side == "Buy" else "Buy",
                orderType="Market", 
                qty=str(position_size),
                takeProfit=str(round(tp_price, 4)),
                timeInForce="GTC"
            )
            
            # Calculate results
            rr_ratio = abs(tp_price - current_price) / abs(current_price - sl_price)
            
            return {
                "success": True,
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "leverage": leverage,
                "entry_price": current_price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "position_size": position_size,
                "risk_amount": risk_amount,
                "risk_reward_ratio": round(rr_ratio, 2)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
