import json
import queue
import threading
import time

import pandas as pd
import ta
from websocket import create_connection

from lib.core.DataManager import DataManager
from lib.core.fetch_kline_data import timeframe_to_seconds


class DataProvider(threading.Thread):
    
    def __init__(self, symbol, timeframe):
        super().__init__()
        self.symbol = symbol
        self.interval = timeframe
        self.timeframe = timeframe
        self.terminated = threading.Event()
        self.df_lock = threading.Lock()
        self.exception_queue = queue.Queue()
        self.row_queue = queue.Queue()
        self.data_init = threading.Event()

    def stop(self):
        self.terminated.set()
    
    @property
    def df(self):
        with self.df_lock:
            while not self.row_queue.empty():
                newest_row = self.row_queue.get()
                if newest_row["timestamp"] != self._df.iloc[-1]["timestamp"]:
                    self._df = self._df.shift(-1, axis=0)
                self._df.iloc[-1] = pd.Series(newest_row, index=self._df.columns)
            return self._df

    def is_fading(self, token):
        df = self.df
        if len(df) < 2:
            return False
        last_price = df.iloc[-1]['close']
        last_candle_price = df.iloc[-2]['close']
        if last_price < last_candle_price:
            return True

    def is_pumping(self, token):
        df = self.df
        if len(df) < 2:
            return False
        last_price = df.iloc[-1]['close']
        last_candle_price = df.iloc[-2]['close']
        if last_price > last_candle_price * 1.03:
            return True
    
    def is_dumping(self, token):
        df = self.df
        if len(df) < 2:
            return False
        last_price = df.iloc[-1]['close']
        last_candle_price = df.iloc[-2]['close']
        if last_price < last_candle_price * (1 - 0.03):
            return True
        
    def is_undervalued(self, token):
        df = self.df
        if len(df) < 200:
            token['rsi'] = float('inf')
            return False
        df['ema50'] = ta.trend.ema_indicator(close=df['close'], window=50)
        df['ema200'] = ta.trend.ema_indicator(close=df['close'], window=200)
        df['vwap'] = ta.volume.volume_weighted_average_price(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=50)
        df['rsi'] = ta.momentum.rsi(close=df['close'], window=14)
        df['atr'] = ta.volatility.average_true_range(high=df['high'], low=df['low'], close=df['close'], window=14)
        lr = df.iloc[-1]
        token['rsi'] = lr['rsi']
        token['atr'] = lr['atr']
        if lr['rsi'] < 40 and lr['close'] < lr['ema50'] < lr['ema200'] and lr['close'] < lr['vwap']:
            return True

    def get_atr(self):
        df = self.df
        if len(df) < 14:
            return -1
        df['atr'] = ta.volatility.average_true_range(high=df['high'], low=df['low'], close=df['close'], window=14)
        return df.iloc[-1]['atr'] / df.iloc[-1]['close']
    
    def run(self):
        since = int((time.time() - timeframe_to_seconds[self.timeframe] * 250) * 1000)
        self._df = DataManager.fetch_klines(
            self.symbol, 
            self.timeframe, 
            since_timestamp=since
        )
        self.data_init.set()
        try:
            url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.interval}"
            ws = create_connection(url)
            while not self.terminated.is_set():
                response = json.loads(ws.recv())
                kline = response.get('k', {})
                newest_row = {
                    'timestamp': kline.get('t', 0),
                    'open': float(kline.get('o', 0)),
                    'high': float(kline.get('h', 0)),
                    'low': float(kline.get('l', 0)),
                    'close': float(kline.get('c', 0)),
                    'volume': float(kline.get('v', 0))
                }
                self.row_queue.put(newest_row)
        except Exception as e:
            self.exception_queue.put(e)
            return
