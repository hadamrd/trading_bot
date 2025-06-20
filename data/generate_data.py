import os
import ta
from lib.core.DataManager import DataManager
import pandas as pd
import requests

def convert_to_numeric(df, column):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def get_btc_spot_trading_pairs():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)
    data = response.json()

    btc_pairs = [pair['symbol'] for pair in data['symbols'] if pair['symbol'].endswith('BTC') and pair['status'] == 'TRADING' and pair['isSpotTradingAllowed']]
    return btc_pairs

if __name__ == "__main__":
    # symbols = get_btc_spot_trading_pairs()
    symbols = ["SUPERBTC"]
    tf = "15m"
    since = "2023-01-01"
    for symbol in symbols:
        try:
            df = DataManager.get_frame_data_csv(symbol=symbol, timeframe=tf, since_date_str=since, with_tech_indicators=True)
        except FileNotFoundError:
            df = DataManager.fetch_klines(symbol, tf, since)

        # Convert columns to numeric
        for col in ['high', 'low', 'close', 'volume']:
            df = convert_to_numeric(df, col)

        # Calculate the indicators
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()

        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()

        # VWAP
        df['vvwap20'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=20).volume_weighted_average_price()
        df['vvwap200'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=200).volume_weighted_average_price()
        df['vvwap9'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=9).volume_weighted_average_price()
        df['vvwap50'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=50).volume_weighted_average_price()
        DataManager.save_frame_data_csv(symbol, tf, since, True, df)
