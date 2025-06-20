
from datetime import datetime

import numpy as np
import pytz
import requests
from retrying import retry
import pandas as pd
import requests
from ta.momentum import RSIIndicator
from concurrent.futures import ThreadPoolExecutor

def paris_datetime(timestamp):
    utc_time = datetime.utcfromtimestamp(timestamp / 1000)  # Assuming the timestamp is in milliseconds
    utc_time = pytz.utc.localize(utc_time)  # Localize the UTC time
    paris_time = utc_time.astimezone(pytz.timezone('Europe/Paris'))  # Convert to Paris time zone
    # Format the Paris time as a string in the desired format
    formatted_time = paris_time.strftime("%Y-%m-%d %H:%M")
    return formatted_time

def sample_slippage(t, next_t, degrees_of_freedom=3, scale=0.001):
    t_sample = np.random.standard_t(degrees_of_freedom)
    return t + (next_t - t) * t_sample

def time_diff_hours(start_timestamp, end_timestamp):
    return (end_timestamp - start_timestamp) / (1000 * 60 * 60)

def format_time_difference(timestamp1, timestamp2):
    # Convert timestamps from milliseconds to datetime objects
    datetime1 = datetime.utcfromtimestamp(timestamp1 / 1000.0)
    datetime2 = datetime.utcfromtimestamp(timestamp2 / 1000.0)

    # Calculate the difference
    time_diff = datetime2 - datetime1

    # Extract days, hours, and minutes
    days = time_diff.days
    hours, remainder = divmod(time_diff.seconds, 3600)
    minutes, _ = divmod(remainder, 60)

    return f"{days} days, {hours} hours, {minutes} minutes"

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def get_top_tokens():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    response = requests.get(url)
    data = response.json()
    usdt_tokens = [token for token in data if token['symbol'].endswith('USDT') and token['symbol'] != 'TOMOUSDT']
    return usdt_tokens
