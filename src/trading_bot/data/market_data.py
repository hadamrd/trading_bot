"""
Market data manager that supports both MongoDB and ClickHouse.
Automatically selects the best storage backend based on configuration.
"""

from datetime import datetime, timedelta
from typing import Any
import logging

import pandas as pd
import ta

from ..core.enums import TimeFrame
from ..core.settings import get_settings
from .binance_client import BinanceClient

logger = logging.getLogger(__name__)

class MarketDataManager:
    """
    Manages market data with support for both MongoDB and ClickHouse.
    ClickHouse is preferred for performance.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()

        self.settings = settings
        
        # Initialize ClickHouse storage
        if settings.database_type.lower() != "clickhouse":
            raise ValueError(f"Only ClickHouse is supported. Set DATABASE_TYPE=clickhouse in .env")

        try:
            # Check if ClickHouse dependencies are available
            import clickhouse_connect
            from .clickhouse_storage import ClickHouseStorage
            
            # Initialize ClickHouse storage
            self.storage = ClickHouseStorage(settings)
            
        except ImportError:
            raise ImportError(
                "ClickHouse dependencies not installed.\n"
                "Install with: poetry add clickhouse-connect"
            )
        except Exception as e:
            raise RuntimeError(
                f"ClickHouse connection failed: {str(e)}\n"
                "Please check your ClickHouse setup and .env configuration."
            )

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
        Download data from Binance and store in database.
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
                logger.info(f"✅ {symbol} data already covers required period ({existing_start.date()} to {existing_end.date()})")
                return 0

            # Fetch missing data at the end
            if existing_end < end_date:
                start_date = existing_end + timedelta(minutes=1)

        logger.info(f"📥 Downloading {symbol} {timeframe} from {start_date.date()} to {end_date.date()}")

        # Fetch from Binance
        candles = self.binance.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            days=(end_date - start_date).days,
            end_date=end_date
        )

        if not candles:
            logger.info(f"❌ No data fetched for {symbol}")
            return 0

        logger.info(f"📦 Fetched {len(candles):,} candles from Binance")
        logger.info(f"   Date range: {candles[0].timestamp} to {candles[-1].timestamp}")

        # Store in database (ClickHouse batch insert is much faster than MongoDB)
        stored_count = self.storage.store_candles(candles)
        
        if stored_count > 0:
            logger.info(f"✅ Stored {stored_count:,} candles successfully")
            
            # Verify storage immediately
            verification_count = self.storage.count_candles(symbol, timeframe)
            logger.info(f"🔍 Verification: {verification_count:,} total candles for {symbol} {timeframe}")
        else:
            logger.info(f"❌ Failed to store candles for {symbol}")

        return stored_count

    def get_data_for_backtest(self,
                             symbol: str,
                             timeframe: TimeFrame,
                             start_date: datetime,
                             end_date: datetime | None = None,
                             with_indicators: bool = True) -> pd.DataFrame:
        """
        Get data formatted for backtesting.
        ClickHouse returns DataFrames much faster than MongoDB.
        """
        logger.info(f"📊 Getting backtest data for {symbol} {timeframe}")
        logger.info(f"   Requested period: {start_date.date()} to {(end_date or datetime.now()).date()}")
        
        df = self.storage.get_candles_df(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_date,
            end_time=end_date
        )

        if df.empty:
            logger.info(f"❌ No data found for {symbol} {timeframe}")
            
            # Debug information
            total_count = self.storage.count_candles(symbol, timeframe)
            logger.info(f"🔍 Debug: Total candles for {symbol} {timeframe}: {total_count:,}")
            
            if total_count > 0:
                date_range = self.storage.get_date_range(symbol, timeframe)
                if date_range:
                    logger.info(f"🔍 Debug: Available date range: {date_range[0]} to {date_range[1]}")
                    logger.info(f"🔍 Debug: Requested range: {start_date} to {end_date}")
            
            return df

        if with_indicators and len(df) >= 50:
            # Only calculate if we don't have indicators or if requested
            if 'rsi' not in df.columns or df['rsi'].isna().all():
                logger.info(f"📊 Calculating indicators for {symbol}...")
                df = self.calculate_basic_indicators(df)

        logger.info(f"📈 Retrieved {len(df)} candles for {symbol} (from {df.index[0].date()} to {df.index[-1].date()})")
        return df

    def calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic technical indicators.
        ClickHouse can store these as separate columns for faster queries.
        """
        if len(df) < 50:  # Need minimum data for indicators
            return df

        # Make a copy to avoid modifying original
        df = df.copy()

        try:
            logger.info(f"🔢 Calculating indicators for {len(df)} candles...")
            
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

            # Fix deprecated fillna warning
            df = df.ffill().fillna(0)

            logger.info(f"✅ Calculated {len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])} indicators")

        except Exception as e:
            logger.info(f"❌ Error calculating indicators: {e}")

        return df

    def update_indicators(self,
                         symbol: str,
                         timeframe: TimeFrame,
                         start_date: datetime | None = None) -> int:
        """
        Update indicators for stored candles - OPTIMIZED for ClickHouse.
        Instead of individual UPDATEs, use bulk operations.
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)

        logger.info(f"🔢 Updating indicators for {symbol} {timeframe.value}...")

        # Get raw data
        df = self.storage.get_candles_df(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_date
        )

        if df.empty:
            logger.info(f"⚠️  No data found for {symbol} to update indicators")
            return 0

        # Calculate indicators
        original_length = len(df)
        df = self.calculate_basic_indicators(df)

        if len(df) == 0:
            logger.info(f"⚠️  No indicators calculated")
            return 0

        logger.info(f"📊 Calculated indicators for {len(df):,} candles")

        # Instead of individual UPDATEs, use ClickHouse's efficient bulk update method
        return self._bulk_update_indicators(symbol, timeframe, df)

    def _bulk_update_indicators(self, symbol: str, timeframe: TimeFrame, df: pd.DataFrame) -> int:
        """
        Efficiently update indicators using ClickHouse's bulk operations.
        This replaces 24k individual UPDATEs with a few bulk operations.
        """
        try:
            # Convert timeframe to string
            timeframe_str = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
            
            logger.info(f"🚀 Using bulk update strategy for {len(df):,} candles...")
            
            # Strategy 1: Delete and re-insert (fastest for large datasets)
            # This is more efficient than 24k individual UPDATEs
            
            # Get min/max dates for the range we're updating
            min_date = df.index.min()
            max_date = df.index.max()
            
            logger.info(f"🔄 Bulk updating period: {min_date} to {max_date}")
            
            # Step 1: Prepare new data with indicators
            new_data = []
            for timestamp, row in df.iterrows():
                new_data.append([
                    symbol,
                    timeframe_str,
                    timestamp,
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume']),
                    # Indicators
                    float(row.get('ema_9', 0)) if pd.notna(row.get('ema_9')) else None,
                    float(row.get('ema_21', 0)) if pd.notna(row.get('ema_21')) else None,
                    float(row.get('ema_50', 0)) if pd.notna(row.get('ema_50')) else None,
                    float(row.get('rsi', 0)) if pd.notna(row.get('rsi')) else None,
                    float(row.get('macd', 0)) if pd.notna(row.get('macd')) else None,
                    float(row.get('macd_signal', 0)) if pd.notna(row.get('macd_signal')) else None,
                    float(row.get('macd_diff', 0)) if pd.notna(row.get('macd_diff')) else None,
                    float(row.get('atr', 0)) if pd.notna(row.get('atr')) else None,
                    float(row.get('vwap', 0)) if pd.notna(row.get('vwap')) else None,
                    float(row.get('bb_upper', 0)) if pd.notna(row.get('bb_upper')) else None,
                    float(row.get('bb_middle', 0)) if pd.notna(row.get('bb_middle')) else None,
                    float(row.get('bb_lower', 0)) if pd.notna(row.get('bb_lower')) else None,
                    float(row.get('volume_sma', 0)) if pd.notna(row.get('volume_sma')) else None,
                    float(row.get('volume_ratio', 0)) if pd.notna(row.get('volume_ratio')) else None,
                ])

            # Step 2: Delete existing data in this range
            delete_query = f"""
            ALTER TABLE market_data DELETE 
            WHERE symbol = '{symbol}' 
              AND timeframe = '{timeframe_str}' 
              AND timestamp >= '{min_date}' 
              AND timestamp <= '{max_date}'
            """
            
            logger.info(f"🗑️  Deleting existing data in range...")
            self.storage.client.command(delete_query)
            
            # Step 3: Insert new data with indicators
            logger.info(f"📦 Inserting updated data with indicators...")
            self.storage.client.insert(
                'market_data',
                new_data,
                column_names=[
                    'symbol', 'timeframe', 'timestamp', 'open', 'high', 'low',
                    'close', 'volume', 'ema_9', 'ema_21', 'ema_50', 'rsi',
                    'macd', 'macd_signal', 'macd_diff', 'atr', 'vwap',
                    'bb_upper', 'bb_middle', 'bb_lower', 'volume_sma', 'volume_ratio'
                ]
            )

            logger.info(f"✅ Bulk update completed for {len(df):,} candles")
            return len(df)

        except Exception as e:
            logger.info(f"❌ Bulk update failed: {e}")
            
            # Fallback: Skip indicator updates and continue
            logger.info(f"⚠️  Skipping indicator updates - proceeding with basic OHLCV data")
            return 0

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
        stats = self.storage.get_database_stats()
        stats["database_type"] = self.settings.database_type
        return stats

    def bulk_download(self,
                     symbols: list[str],
                     timeframe: TimeFrame = TimeFrame.FIFTEEN_MINUTES,
                     days: int = 30) -> dict[str, int]:
        """Download data for multiple symbols"""
        results = {}

        logger.info(f"📥 Bulk downloading {len(symbols)} symbols...")
        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
                count = self.download_and_store(symbol, timeframe, days)
                results[symbol] = count
                
                # Update indicators for successful downloads
                if count > 0:
                    self.update_indicators(symbol, timeframe)
                    
            except Exception as e:
                logger.info(f"❌ Error downloading {symbol}: {e}")
                results[symbol] = 0

        successful = sum(1 for count in results.values() if count > 0)
        total_candles = sum(results.values())
        logger.info(f"\n✅ Bulk download complete: {successful}/{len(symbols)} symbols, {total_candles:,} total candles")

        return results

    def benchmark_performance(self, symbol: str = "BTCUSDT", timeframe: TimeFrame = TimeFrame.FIFTEEN_MINUTES):
        """Benchmark database performance for comparison"""
        import time
        
        logger.info(f"🏃‍♂️ Benchmarking {self.settings.database_type} performance...")
        
        # Test 1: Count query
        start = time.time()
        count = self.storage.count_candles(symbol, timeframe)
        count_time = time.time() - start
        
        # Test 2: Date range query
        start = time.time()
        date_range = self.storage.get_date_range(symbol, timeframe)
        range_time = time.time() - start
        
        # Test 3: Large DataFrame query
        if date_range:
            start = time.time()
            df = self.storage.get_candles_df(symbol, timeframe)
            df_time = time.time() - start
            df_size = len(df)
        else:
            df_time = 0
            df_size = 0
        
        logger.info(f"📊 Performance Results ({self.settings.database_type}):")
        logger.info(f"   Count query: {count_time:.3f}s ({count:,} candles)")
        logger.info(f"   Date range query: {range_time:.3f}s")
        logger.info(f"   DataFrame query: {df_time:.3f}s ({df_size:,} rows)")
        if df_size > 0:
            logger.info(f"   Throughput: {df_size/df_time:.0f} rows/second")

    def close(self):
        """Close connections"""
        self.storage.close()