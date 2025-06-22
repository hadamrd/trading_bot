"""
ClickHouse storage for market data - optimized for time-series analytics.
Much better than MongoDB for financial data analysis.
"""

from datetime import datetime
from typing import Any

import pandas as pd
from clickhouse_connect import get_client
from clickhouse_connect.driver import Client

from ..core.enums import TimeFrame
from ..core.models import MarketData
from ..core.settings import get_settings


class ClickHouseStorage:
    """ClickHouse storage optimized for time-series market data"""

    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()

        self.settings = settings
        
        try:
            print(f"🔌 Connecting to ClickHouse at {settings.clickhouse_host}:{settings.clickhouse_port}")
            print(f"   Username: {settings.clickhouse_username}")
            print(f"   Database: {settings.clickhouse_database}")
            
            # Connect with proper authentication
            self.client: Client = get_client(
                host=settings.clickhouse_host,
                port=settings.clickhouse_port,
                username=settings.clickhouse_username,
                password=settings.clickhouse_password,
                database=settings.clickhouse_database
            )
            
            # Test connection
            self.client.command('SELECT 1')
            print("✅ ClickHouse connected successfully")
            
            # Create tables
            self._create_tables()
            print("✅ Tables verified/created")
            
        except Exception as e:
            error_msg = str(e)
            if "Authentication failed" in error_msg or "password is incorrect" in error_msg:
                raise ConnectionError(
                    f"ClickHouse authentication failed!\n"
                    f"Please check your credentials in .env file:\n"
                    f"  CLICKHOUSE_USERNAME={settings.clickhouse_username}\n"
                    f"  CLICKHOUSE_PASSWORD=***\n"
                    f"Or restart ClickHouse: docker restart trading-clickhouse"
                ) from e
            elif "Connection refused" in error_msg:
                raise ConnectionError(
                    f"ClickHouse server not running on {settings.clickhouse_host}:{settings.clickhouse_port}\n"
                    f"Start it with: docker run -d --name trading-clickhouse -p 8123:8123 ..."
                ) from e
            elif "Unknown database" in error_msg:
                raise ConnectionError(
                    f"Database '{settings.clickhouse_database}' doesn't exist.\n"
                    f"Run setup script to create it properly."
                ) from e
            else:
                raise ConnectionError(f"ClickHouse connection failed: {error_msg}") from e

    def _create_tables(self):
        """Create ClickHouse tables with optimal schema for time-series data"""
        
        # Main market data table - partitioned by date for optimal performance
        create_market_data = """
        CREATE TABLE IF NOT EXISTS market_data (
            symbol String,
            timeframe String,
            timestamp DateTime64(3),
            open Float64,
            high Float64,
            low Float64,
            close Float64,
            volume Float64,
            -- Technical indicators as separate columns (much faster than JSON)
            ema_9 Nullable(Float64),
            ema_21 Nullable(Float64),
            ema_50 Nullable(Float64),
            rsi Nullable(Float64),
            macd Nullable(Float64),
            macd_signal Nullable(Float64),
            macd_diff Nullable(Float64),
            atr Nullable(Float64),
            vwap Nullable(Float64),
            bb_upper Nullable(Float64),
            bb_middle Nullable(Float64),
            bb_lower Nullable(Float64),
            volume_sma Nullable(Float64),
            volume_ratio Nullable(Float64),
            -- Add more indicators as needed
            created_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (symbol, timeframe, timestamp)
        SETTINGS index_granularity = 8192
        """

        self.client.command(create_market_data)

        # Create indexes for fast queries
        # ClickHouse automatically creates primary key index, but we can add more
        try:
            # Skip indexes for ClickHouse (they're different from SQL indexes)
            pass
        except Exception:
            pass  # Indexes might already exist

    def store_candle(self, candle: MarketData) -> bool:
        """Store a single candle"""
        try:
            # Use timeframe.value for proper string conversion
            timeframe_str = candle.timeframe.value if hasattr(candle.timeframe, 'value') else str(candle.timeframe)
            
            # Use INSERT with explicit NULL values for indicators
            data = [[
                candle.symbol,
                timeframe_str,  # Store as '15m' not 'TimeFrame.FIFTEEN_MINUTES'
                candle.timestamp,
                float(candle.open),
                float(candle.high),
                float(candle.low),
                float(candle.close),
                float(candle.volume),
                # All indicators NULL initially (will be updated later)
                None, None, None, None, None, None, None, None, None, None, None, None, None, None
            ]]

            self.client.insert(
                'market_data',
                data,
                column_names=[
                    'symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 
                    'close', 'volume', 'ema_9', 'ema_21', 'ema_50', 'rsi',
                    'macd', 'macd_signal', 'macd_diff', 'atr', 'vwap',
                    'bb_upper', 'bb_middle', 'bb_lower', 'volume_sma', 'volume_ratio'
                ]
            )
            return True
        except Exception as e:
            print(f"Error storing single candle: {e}")
            print(f"  Candle: {candle.symbol} {candle.timestamp}")
            return False

    def store_candles(self, candles: list[MarketData]) -> int:
        """Store multiple candles efficiently using batch insert"""
        if not candles:
            return 0

        try:
            print(f"📦 Storing {len(candles):,} candles to ClickHouse...")
            
            # Prepare batch data - only store OHLCV data initially (indicators come later)
            data = []
            for candle in candles:
                # Use timeframe.value for proper string conversion
                timeframe_str = candle.timeframe.value if hasattr(candle.timeframe, 'value') else str(candle.timeframe)
                
                data.append([
                    candle.symbol,
                    timeframe_str,  # Store as '15m' not 'TimeFrame.FIFTEEN_MINUTES'
                    candle.timestamp,
                    float(candle.open),
                    float(candle.high),
                    float(candle.low),
                    float(candle.close),
                    float(candle.volume),
                    # Set all indicators to NULL initially
                    None, None, None, None, None, None, None, None, None, None, None, None, None, None
                ])

            # Batch insert with explicit NULL handling
            self.client.insert(
                'market_data', 
                data,
                column_names=[
                    'symbol', 'timeframe', 'timestamp', 'open', 'high', 'low',
                    'close', 'volume', 'ema_9', 'ema_21', 'ema_50', 'rsi',
                    'macd', 'macd_signal', 'macd_diff', 'atr', 'vwap',
                    'bb_upper', 'bb_middle', 'bb_lower', 'volume_sma', 'volume_ratio'
                ]
            )
            
            # Verify the insert worked
            symbol = candles[0].symbol
            timeframe = candles[0].timeframe
            count = self.count_candles(symbol, timeframe)
            
            print(f"✅ Successfully stored {len(candles):,} candles")
            print(f"📊 Total candles for {symbol} {timeframe.value}: {count:,}")
            
            return len(candles)
            
        except Exception as e:
            print(f"❌ Error storing candles: {e}")
            print(f"   Symbol: {candles[0].symbol if candles else 'N/A'}")
            print(f"   Count: {len(candles):,}")
            
            # Try to store one by one to identify the problematic candle
            print("🔍 Trying individual inserts to debug...")
            stored_count = 0
            for i, candle in enumerate(candles[:10]):  # Test first 10
                try:
                    if self.store_candle(candle):
                        stored_count += 1
                except Exception as single_error:
                    print(f"❌ Failed on candle {i}: {single_error}")
                    break
            
            return stored_count

    def get_candles(self,
                   symbol: str,
                   timeframe: TimeFrame,
                   start_time: datetime | None = None,
                   end_time: datetime | None = None,
                   limit: int | None = None) -> list[MarketData]:
        """Get candles with optimal ClickHouse query"""

        # Use timeframe.value for proper string conversion
        timeframe_str = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
        
        # Build efficient time-series query
        where_clauses = [
            f"symbol = '{symbol}'",
            f"timeframe = '{timeframe_str}'"
        ]

        if start_time:
            where_clauses.append(f"timestamp >= '{start_time}'")
        if end_time:
            where_clauses.append(f"timestamp <= '{end_time}'")

        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        SELECT symbol, timeframe, timestamp, open, high, low, close, volume,
               ema_9, ema_21, ema_50, rsi, macd, macd_signal, macd_diff, 
               atr, vwap, bb_upper, bb_middle, bb_lower, volume_sma, volume_ratio
        FROM market_data 
        WHERE {where_clause}
        ORDER BY timestamp ASC
        """

        if limit:
            query += f" LIMIT {limit}"

        try:
            result = self.client.query(query)
            candles = []
            
            for row in result.result_rows:
                # Reconstruct indicators dict from columns
                indicators = {}
                indicator_names = [
                    'ema_9', 'ema_21', 'ema_50', 'rsi', 'macd', 'macd_signal', 
                    'macd_diff', 'atr', 'vwap', 'bb_upper', 'bb_middle', 
                    'bb_lower', 'volume_sma', 'volume_ratio'
                ]
                
                for i, name in enumerate(indicator_names, start=8):  # Start after OHLCV
                    if row[i] is not None:
                        indicators[name] = row[i]

                candle = MarketData(
                    symbol=row[0],
                    timeframe=row[1],
                    timestamp=row[2],
                    open=row[3],
                    high=row[4],
                    low=row[5],
                    close=row[6],
                    volume=row[7],
                    indicators=indicators
                )
                candles.append(candle)

            return candles
        except Exception as e:
            print(f"Error getting candles: {e}")
            return []

    def get_candles_df(self,
                      symbol: str,
                      timeframe: TimeFrame,
                      start_time: datetime | None = None,
                      end_time: datetime | None = None,
                      limit: int | None = None) -> pd.DataFrame:
        """Get candles as DataFrame - optimized for ClickHouse"""

        # CRITICAL FIX: Use timeframe.value not str(timeframe)
        timeframe_str = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
        
        where_clauses = [
            f"symbol = '{symbol}'",
            f"timeframe = '{timeframe_str}'"  # Use .value to get '15m' not 'TimeFrame.FIFTEEN_MINUTES'
        ]

        if start_time:
            where_clauses.append(f"timestamp >= '{start_time}'")
        if end_time:
            where_clauses.append(f"timestamp <= '{end_time}'")

        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        SELECT timestamp, open, high, low, close, volume,
               ema_9, ema_21, ema_50, rsi, macd, macd_signal, macd_diff, 
               atr, vwap, bb_upper, bb_middle, bb_lower, volume_sma, volume_ratio
        FROM market_data 
        WHERE {where_clause}
        ORDER BY timestamp ASC
        """

        if limit:
            query += f" LIMIT {limit}"

        try:
            print(f"🔍 Querying ClickHouse for {symbol} {timeframe_str}...")
            
            # ClickHouse can return DataFrame directly - much faster!
            df = self.client.query_df(query)
            
            print(f"📊 Retrieved {len(df):,} rows from ClickHouse")
            
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                # Remove columns with all nulls
                df = df.dropna(axis=1, how='all')
                print(f"✅ DataFrame ready: {len(df):,} rows, {len(df.columns)} columns")
            else:
                print(f"⚠️  No data returned for {symbol} {timeframe_str}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error getting DataFrame: {e}")
            print(f"   Query: {query}")
            return pd.DataFrame()

    def get_latest_candle(self, symbol: str, timeframe: TimeFrame) -> MarketData | None:
        """Get latest candle - very fast with ClickHouse ordering"""
        candles = self.get_candles(symbol, timeframe, limit=1)
        return candles[0] if candles else None

    def store_indicators(self,
                        symbol: str,
                        timeframe: TimeFrame,
                        timestamp: datetime,
                        indicators: dict[str, float]) -> bool:
        """Update indicators for existing candle"""
        try:
            # Use timeframe.value for proper string conversion
            timeframe_str = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
            
            # Build UPDATE query for indicators
            set_clauses = []
            for key, value in indicators.items():
                if key in ['ema_9', 'ema_21', 'ema_50', 'rsi', 'macd', 'macd_signal', 
                          'macd_diff', 'atr', 'vwap', 'bb_upper', 'bb_middle', 
                          'bb_lower', 'volume_sma', 'volume_ratio']:
                    set_clauses.append(f"{key} = {value}")

            if not set_clauses:
                return True

            query = f"""
            ALTER TABLE market_data UPDATE {', '.join(set_clauses)}
            WHERE symbol = '{symbol}' 
              AND timeframe = '{timeframe_str}' 
              AND timestamp = '{timestamp}'
            """

            self.client.command(query)
            return True
        except Exception as e:
            print(f"Error updating indicators: {e}")
            return False

    def get_symbols(self) -> list[str]:
        """Get all available symbols - fast with ClickHouse aggregation"""
        try:
            result = self.client.query("SELECT DISTINCT symbol FROM market_data ORDER BY symbol")
            return [row[0] for row in result.result_rows]
        except Exception as e:
            print(f"Error getting symbols: {e}")
            return []

    def get_timeframes(self, symbol: str) -> list[str]:
        """Get available timeframes for symbol"""
        try:
            query = f"SELECT DISTINCT timeframe FROM market_data WHERE symbol = '{symbol}'"
            result = self.client.query(query)
            return [row[0] for row in result.result_rows]
        except Exception as e:
            print(f"Error getting timeframes: {e}")
            return []

    def get_date_range(self, symbol: str, timeframe: TimeFrame) -> tuple[datetime, datetime] | None:
        """Get date range - ClickHouse aggregation is very fast"""
        try:
            # Use timeframe.value for proper string conversion
            timeframe_str = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
            
            query = f"""
            SELECT min(timestamp) as min_date, max(timestamp) as max_date
            FROM market_data 
            WHERE symbol = '{symbol}' AND timeframe = '{timeframe_str}'
            """
            result = self.client.query(query)
            
            if result.result_rows and result.result_rows[0][0]:
                return result.result_rows[0][0], result.result_rows[0][1]
            return None
        except Exception as e:
            print(f"Error getting date range: {e}")
            return None

    def count_candles(self, symbol: str, timeframe: TimeFrame) -> int:
        """Count candles - very fast with ClickHouse"""
        try:
            # Use timeframe.value for proper string conversion
            timeframe_str = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
            
            query = f"""
            SELECT count(*) FROM market_data 
            WHERE symbol = '{symbol}' AND timeframe = '{timeframe_str}'
            """
            result = self.client.query(query)
            return result.result_rows[0][0] if result.result_rows else 0
        except Exception as e:
            print(f"Error counting candles: {e}")
            return 0

    def delete_candles(self,
                      symbol: str,
                      timeframe: TimeFrame,
                      start_time: datetime | None = None,
                      end_time: datetime | None = None) -> int:
        """Delete candles"""
        try:
            # Use timeframe.value for proper string conversion
            timeframe_str = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
            
            where_clauses = [
                f"symbol = '{symbol}'",
                f"timeframe = '{timeframe_str}'"
            ]

            if start_time:
                where_clauses.append(f"timestamp >= '{start_time}'")
            if end_time:
                where_clauses.append(f"timestamp <= '{end_time}'")

            where_clause = " AND ".join(where_clauses)
            
            # Count first
            count_query = f"SELECT count(*) FROM market_data WHERE {where_clause}"
            count_result = self.client.query(count_query)
            count = count_result.result_rows[0][0] if count_result.result_rows else 0

            # Delete
            delete_query = f"ALTER TABLE market_data DELETE WHERE {where_clause}"
            self.client.command(delete_query)
            
            return count
        except Exception as e:
            print(f"Error deleting candles: {e}")
            return 0

    def get_database_stats(self) -> dict[str, Any]:
        """Get database statistics - ClickHouse system tables are very informative"""
        try:
            # Total candles
            total_result = self.client.query("SELECT count(*) FROM market_data")
            total_candles = total_result.result_rows[0][0] if total_result.result_rows else 0

            # Unique symbols
            symbols_result = self.client.query("SELECT count(DISTINCT symbol) FROM market_data")
            symbols_count = symbols_result.result_rows[0][0] if symbols_result.result_rows else 0

            # Table size
            size_query = """
            SELECT formatReadableSize(sum(data_compressed_bytes)) as compressed_size,
                   formatReadableSize(sum(data_uncompressed_bytes)) as uncompressed_size
            FROM system.parts 
            WHERE table = 'market_data'
            """
            size_result = self.client.query(size_query)
            
            stats = {
                "total_candles": total_candles,
                "symbols": symbols_count,
                "compressed_size": size_result.result_rows[0][0] if size_result.result_rows else "0 B",
                "uncompressed_size": size_result.result_rows[0][1] if size_result.result_rows else "0 B"
            }

            # Per-symbol stats
            symbol_stats_query = """
            SELECT symbol, timeframe, count(*) as candles,
                   min(timestamp) as min_date, max(timestamp) as max_date
            FROM market_data 
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
            """
            symbol_result = self.client.query(symbol_stats_query)
            
            symbol_stats = {}
            for row in symbol_result.result_rows:
                symbol = row[0]
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {}
                
                symbol_stats[symbol][row[1]] = {
                    "candles": row[2],
                    "date_range": (row[3], row[4])
                }

            stats["symbols_detail"] = symbol_stats
            return stats

        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {"total_candles": 0, "symbols": 0}

    def close(self):
        """Close ClickHouse connection"""
        try:
            self.client.close()
        except Exception:
            pass