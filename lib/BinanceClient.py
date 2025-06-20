import os
from binance.spot import Spot
class BinanceClient:
    
    def __init__(self) -> None:
        self.api_key = os.environ.get("BINANCE_API_KEY")
        self.api_secret = os.environ.get("BINANCE_API_SECRET")
        self._spot = Spot(self.api_key, self.api_secret)
    
    def fetch_klines(self, symbol, timeframe, since, until, limit):
        return self._spot.klines(
            symbol=symbol,
            interval=timeframe,
            startTime=since,
            endTime=until,
            limit=limit,
        )
    
    def fetch_transaction_fee(self, symbol):
        trade_fee_info = self._spot.trade_fee(symbol=symbol)
        fees = trade_fee_info[0]
        maker_fee = fees['makerCommission']
        taker_fee = fees['takerCommission']
        return maker_fee, taker_fee
    
    def get_exchange_info(self, symbol):
        exchange_info = self._spot.exchange_info(symbol)
        symbol_info = next((item for item in exchange_info['symbols'] if item['symbol'] == symbol), None)
        if symbol_info:
            min_trade_amount = next((f['minQty'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            min_price_movement = next((f['minPrice'] for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
            return min_trade_amount, min_price_movement
        else:
            raise Exception(f"Symbol {symbol} not found.")
    
    def get_margin_info(self, asset):
        interest_rate_history = self._spot.margin_interest_rate_history(asset=asset, limit=1)
        latest_interest_rate = interest_rate_history[0]['dailyInterestRate']
        return latest_interest_rate

    def fetch_trade_config(self, symbol, asset):
        maker_fee, taker_fee = self.fetch_transaction_fee(symbol)
        min_trade_amount, min_price_movement = self.get_exchange_info(symbol)
        daily_interest_rate = self.get_margin_info(asset)
        hourly_borrow_rate = float(daily_interest_rate) / 24
        return {
            "tr_fee_rate": float(maker_fee),
            "hourly_borrow_rate": float(hourly_borrow_rate),
            "min_trade_amount": float(min_trade_amount),
            "min_price_movement": float(min_price_movement),
        }
        