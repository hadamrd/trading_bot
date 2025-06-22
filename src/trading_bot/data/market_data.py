"""
Market data manager that combines MongoDB storage with Binance client.
Using proper settings configuration.
"""

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import ta

from ..core.enums import TimeFrame
from ..core.settings import get_settings
from .binance_client import BinanceClient
from .storage import MongoStorage


class MarketDataManager:
    """
    Manages market data fetching, storage, and retrieval.
    Clean replacement for the old DataManager chaos.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()

        self.settings = settings
        self.storage = MongoStorage(settings)
        self.binance = BinanceClient(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret
        )

    def download_and_store(self,
                          symbol: str,
                          timeframe: TimeFrame,
                          days: int = 30,
                          force_update: bool = False) -> int:
        """
        Download data from Binance and store in MongoDB.
        Returns number of candles stored.
        """

        # Check what we already have
        existing_range = self.storage.get_date_range(symbol, timeframe)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        if existing_range and not force_update:
            # Only fetch missing data
            existing_start, existing_end = existing_range

            if existing_start <= start_date and existing_end >= end_date:
                print(f"Data for {symbol} {timeframe} already up to date")
                return 0

            # Fetch missing data at the end
            if existing_end < end_date:
                start_date = existing_end + timedelta(minutes=1)

        print(f"Downloading {symbol} {timeframe} from {start_date} to {end_date}")

        # Fetch from Binance
        candles = self.binance.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            days=(end_date - start_date).days,
            end_date=end_date
        )

        if not candles:
            print(f"No data fetched for {symbol}")
            return 0

        # Store in MongoDB
        stored_count = self.storage.store_candles(candles)
        print(f"Stored {stored_count}/{len(candles)} candles for {symbol}")

        return stored_count

    def get_data_for_backtest(self,
                             symbol: str,
                             timeframe: TimeFrame,
                             start_date: datetime,
                             end_date: datetime | None = None,
                             with_indicators: bool = True) -> pd.DataFrame:
        """
        Get data formatted for backtesting.
        """
        df = self.storage.get_candles_df(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_date,
            end_time=end_date
        )

        if df.empty:
            print(f"No data found for {symbol} {timeframe}")
            return df

        if with_indicators:
            df = self.calculate_basic_indicators(df)

        return df

    def calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic technical indicators.
        Clean version of the scattered indicator calculations.
        """
        if len(df) < 50:  # Need minimum data for indicators
            return df

        # Make a copy to avoid modifying original
        df = df.copy()

        try:
            # Moving averages
            df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
            df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
            df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)

            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)

            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()

            # ATR
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=14
            )

            # VWAP
            df['vwap'] = ta.volume.volume_weighted_average_price(
                df['high'], df['low'], df['close'], df['volume'], window=20
            )

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()

            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

        except Exception as e:
            print(f"Error calculating indicators: {e}")

        return df

    def update_indicators(self,
                         symbol: str,
                         timeframe: TimeFrame,
                         start_date: datetime | None = None) -> int:
        """
        Update indicators for stored candles.
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)

        # Get raw data
        df = self.storage.get_candles_df(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_date
        )

        if df.empty:
            return 0

        # Calculate indicators
        df = self.calculate_basic_indicators(df)

        # Store indicators back to MongoDB
        updated_count = 0
        for timestamp, row in df.iterrows():
            indicators = {}
            for col in df.columns:
                if col not in ['open', 'high', 'low', 'close', 'volume']:
                    if pd.notna(row[col]):
                        indicators[col] = float(row[col])

            if indicators:
                success = self.storage.store_indicators(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=timestamp,
                    indicators=indicators
                )
                if success:
                    updated_count += 1

        print(f"Updated indicators for {updated_count} candles")
        return updated_count

    def get_latest_price(self, symbol: str, timeframe: TimeFrame = TimeFrame.ONE_MINUTE) -> float | None:
        """Get latest price for a symbol"""
        latest = self.storage.get_latest_candle(symbol, timeframe)
        return float(latest.close) if latest else None

    def get_available_symbols(self) -> list[str]:
        """Get all symbols with data"""
        return self.storage.get_symbols()

    def get_data_info(self, symbol: str, timeframe: TimeFrame) -> dict[str, Any]:
        """Get information about available data"""
        date_range = self.storage.get_date_range(symbol, timeframe)
        count = self.storage.count_candles(symbol, timeframe)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candle_count": count,
            "date_range": date_range,
            "latest_price": self.get_latest_price(symbol, timeframe)
        }

    def cleanup_old_data(self,
                        symbol: str,
                        timeframe: TimeFrame,
                        keep_days: int = 365) -> int:
        """Remove old data to save space"""
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        return self.storage.delete_candles(
            symbol=symbol,
            timeframe=timeframe,
            end_time=cutoff_date
        )

    def get_database_stats(self) -> dict[str, Any]:
        """Get database usage statistics"""
        return self.storage.get_database_stats()

    def bulk_download(self,
                     symbols: list[str],
                     timeframe: TimeFrame = TimeFrame.FIFTEEN_MINUTES,
                     days: int = 30) -> dict[str, int]:
        """Download data for multiple symbols"""
        results = {}

        for symbol in symbols:
            try:
                count = self.download_and_store(symbol, timeframe, days)
                results[symbol] = count
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
                results[symbol] = 0

        return results

    def close(self):
        """Close connections"""
        self.storage.close()
