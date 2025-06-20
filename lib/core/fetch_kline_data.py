import os
from datetime import datetime

import pandas as pd
from tqdm.autonotebook import tqdm

from lib.BinanceClient import BinanceClient


def iso8601_to_ms_timestamp(iso8601_date):
    dt = datetime.strptime(iso8601_date, "%Y-%m-%d")
    return int(dt.timestamp()) * 1000

timeframe_to_seconds = {
    '1m': 60,         # 1 minute
    '3m': 180,        # 3 minutes
    '5m': 300,        # 5 minutes
    '15m': 900,       # 15 minutes
    '30m': 1800,      # 30 minutes
    '1h': 3600,       # 1 hour
    '6h': 21600,      # 6 hours
    '12h': 43200,     # 12 hours
    '1d': 86400,      # 1 day
    '3d': 259200,     # 3 days
    '1w': 604800,     # 1 week
    '1M': 2592000     # 1 month (approximate)
}

def convert_to_numeric(df, column):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def fetch_historical_data(
    symbol, timeframe, since_date_str=None, since_timestamp=None, until=None, limit=1000
) -> pd.DataFrame:
    if since_timestamp is not None:
        start_date = since_timestamp
    elif since_date_str is not None:
        start_date = iso8601_to_ms_timestamp(since_date_str)
    else:
        raise ValueError("Either since_date_str or since_timestamp must be provided")
    if until is None:
        end_date = int(datetime.now().timestamp() * 1000)
    else:
        end_date = iso8601_to_ms_timestamp(until)
    step_interval = limit * timeframe_to_seconds[timeframe] * 1000
    historical_data = []
    number_steps = (end_date - start_date) // step_interval
    client = BinanceClient()
    seen_timestamps = set()
    for step in tqdm(range(number_steps + 1), desc=f"Fetching historical data of {symbol} {timeframe} since {since_date_str}"):
        since = start_date + step * step_interval
        until = since + step_interval
        data = client.fetch_klines(symbol, timeframe, since, until, limit)
        filtered = [d[:6] for d in data if d[0] not in seen_timestamps]
        historical_data.extend(filtered)
        seen_timestamps.update([d[0] for d in data])
    if timeframe == "1M":
        timeframe = "1Month"
    df = pd.DataFrame(
        historical_data,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df = convert_to_numeric(df, col)
    return df
