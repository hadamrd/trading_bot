"""
Clean Binance client for fetching market data.
Consolidated from lib/BinanceClient.py and lib/core/fetch_kline_data.py
"""

import os
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd
import requests
from binance.spot import Spot
from tqdm import tqdm

from ..core.enums import TIMEFRAME_TO_SECONDS, TimeFrame
from ..core.models import MarketData


class BinanceClient:
    """Clean Binance API client"""

    def __init__(self, api_key: str | None = None, api_secret: str | None = None):
        # Use provided keys or fallback to environment variables
        self.api_key = api_key
        self.api_secret = api_secret

        # If no keys provided, try environment variables
        if not self.api_key:
            self.api_key = os.environ.get("BINANCE_API_KEY")
        if not self.api_secret:
            self.api_secret = os.environ.get("BINANCE_API_SECRET")

        # Initialize client (can work without keys for public data)
        if self.api_key and self.api_secret:
            self.client = Spot(self.api_key, self.api_secret)
        else:
            self.client = Spot()  # Public endpoints only

    def fetch_klines(self,
                    symbol: str,
                    timeframe: TimeFrame,
                    start_time: datetime | None = None,
                    end_time: datetime | None = None,
                    limit: int = 1000) -> list[MarketData]:
        """Fetch klines from Binance API"""

        # Convert datetime to milliseconds
        start_ms = int(start_time.timestamp() * 1000) if start_time else None
        end_ms = int(end_time.timestamp() * 1000) if end_time else None

        try:
            raw_klines = self.client.klines(
                symbol=symbol,
                interval=timeframe,
                startTime=start_ms,
                endTime=end_ms,
                limit=limit
            )

            return self._convert_klines_to_market_data(raw_klines, symbol, timeframe)

        except Exception as e:
            print(f"Error fetching klines: {e}")
            return []

    def fetch_historical_data(self,
                            symbol: str,
                            timeframe: TimeFrame,
                            days: int = 30,
                            end_date: datetime | None = None) -> list[MarketData]:
        """Fetch historical data for specified number of days"""

        if end_date is None:
            end_date = datetime.now()

        start_date = end_date - timedelta(days=days)

        # Calculate how many requests we need
        interval_seconds = TIMEFRAME_TO_SECONDS[timeframe]
        total_intervals = int((end_date - start_date).total_seconds() / interval_seconds)
        limit = 1000  # Binance limit

        all_candles = []
        current_start = start_date

        with tqdm(total=total_intervals, desc=f"Fetching {symbol} {timeframe}") as pbar:
            while current_start < end_date:
                # Calculate end time for this batch
                batch_end = min(
                    current_start + timedelta(seconds=limit * interval_seconds),
                    end_date
                )

                candles = self.fetch_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=current_start,
                    end_time=batch_end,
                    limit=limit
                )

                if not candles:
                    break

                all_candles.extend(candles)
                pbar.update(len(candles))

                # Move to next batch
                current_start = candles[-1].timestamp + timedelta(seconds=interval_seconds)

                # Rate limiting
                time.sleep(0.1)

        return all_candles

    def get_exchange_info(self, symbol: str) -> dict[str, Any]:
        """Get exchange information for a symbol"""
        try:
            info = self.client.exchange_info(symbol=symbol)
            symbol_info = info['symbols'][0] if info['symbols'] else {}

            # Extract useful info
            filters = {f['filterType']: f for f in symbol_info.get('filters', [])}

            return {
                "symbol": symbol,
                "status": symbol_info.get('status'),
                "base_asset": symbol_info.get('baseAsset'),
                "quote_asset": symbol_info.get('quoteAsset'),
                "min_qty": float(filters.get('LOT_SIZE', {}).get('minQty', 0)),
                "min_price": float(filters.get('PRICE_FILTER', {}).get('minPrice', 0)),
                "tick_size": float(filters.get('PRICE_FILTER', {}).get('tickSize', 0)),
            }
        except Exception as e:
            print(f"Error getting exchange info: {e}")
            return {}

    def get_trading_fees(self, symbol: str) -> dict[str, float]:
        """Get trading fees for a symbol"""
        if not self.api_key:
            return {"maker": 0.001, "taker": 0.001}  # Default fees

        try:
            fee_info = self.client.trade_fee(symbol=symbol)
            if fee_info:
                fees = fee_info[0]
                return {
                    "maker": float(fees.get('makerCommission', 0.001)),
                    "taker": float(fees.get('takerCommission', 0.001))
                }
        except Exception as e:
            print(f"Error getting trading fees: {e}")

        return {"maker": 0.001, "taker": 0.001}

    def get_24hr_ticker(self, symbol: str) -> dict[str, Any]:
        """Get 24hr ticker statistics"""
        try:
            ticker = self.client.ticker_24hr(symbol=symbol)
            return {
                "symbol": ticker["symbol"],
                "price_change": float(ticker["priceChange"]),
                "price_change_percent": float(ticker["priceChangePercent"]),
                "open_price": float(ticker["openPrice"]),
                "high_price": float(ticker["highPrice"]),
                "low_price": float(ticker["lowPrice"]),
                "last_price": float(ticker["lastPrice"]),
                "volume": float(ticker["volume"]),
                "quote_volume": float(ticker["quoteVolume"])
            }
        except Exception as e:
            print(f"Error getting 24hr ticker: {e}")
            return {}

    def get_usdt_pairs(self) -> list[str]:
        """Get all USDT trading pairs"""
        try:
            url = "https://api.binance.com/api/v3/exchangeInfo"
            response = requests.get(url)
            data = response.json()

            usdt_pairs = [
                symbol['symbol']
                for symbol in data['symbols']
                if (symbol['symbol'].endswith('USDT') and
                    symbol['status'] == 'TRADING' and
                    symbol['isSpotTradingAllowed'])
            ]

            return sorted(usdt_pairs)

        except Exception as e:
            print(f"Error getting USDT pairs: {e}")
            return []

    def _convert_klines_to_market_data(self,
                                     raw_klines: list[list],
                                     symbol: str,
                                     timeframe: TimeFrame) -> list[MarketData]:
        """Convert raw Binance klines to MarketData objects"""
        market_data = []

        for kline in raw_klines:
            try:
                market_data.append(MarketData(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.fromtimestamp(kline[0] / 1000),
                    open=Decimal(str(kline[1])),
                    high=Decimal(str(kline[2])),
                    low=Decimal(str(kline[3])),
                    close=Decimal(str(kline[4])),
                    volume=Decimal(str(kline[5]))
                ))
            except (ValueError, IndexError) as e:
                print(f"Error converting kline: {e}")
                continue

        return market_data


def iso8601_to_datetime(date_str: str) -> datetime:
    """Convert ISO8601 date string to datetime"""
    return datetime.strptime(date_str, "%Y-%m-%d")


def convert_to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert DataFrame columns to numeric - from old code"""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
