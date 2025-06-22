"""
MongoDB storage for market data.
Using proper settings configuration instead of YAML chaos.
"""

from datetime import datetime
from typing import Any

import pandas as pd
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from ..core.enums import TimeFrame
from ..core.models import MarketData
from ..core.settings import get_settings


class MongoStorage:
    """MongoDB storage for market data"""

    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()

        self.settings = settings

        # Use MongoDB URL if provided, otherwise build from components
        if settings.mongodb_url and "mongodb://" in settings.mongodb_url:
            self.client = MongoClient(settings.mongodb_url)
        else:
            self.client = MongoClient(
                host=settings.mongodb_host,
                port=settings.mongodb_port,
                username=settings.mongodb_username,
                password=settings.mongodb_password,
                authSource=settings.mongodb_auth_source
            )

        self.db: Database = self.client[settings.mongodb_database]
        self.market_data: Collection = self.db.market_data

        # Create indexes for efficient queries
        self._create_indexes()

    def _create_indexes(self):
        """Create database indexes for efficient queries"""
        # Compound index for symbol, timeframe, timestamp
        self.market_data.create_index([
            ("symbol", ASCENDING),
            ("timeframe", ASCENDING),
            ("timestamp", ASCENDING)
        ], unique=True)

        # Index for timestamp range queries
        self.market_data.create_index([("timestamp", ASCENDING)])

    def store_candle(self, candle: MarketData) -> bool:
        """Store a single candle"""
        try:
            doc = candle.to_mongo_doc()
            self.market_data.replace_one(
                {
                    "symbol": doc["symbol"],
                    "timeframe": doc["timeframe"],
                    "timestamp": doc["timestamp"]
                },
                doc,
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error storing candle: {e}")
            return False

    def store_candles(self, candles: list[MarketData]) -> int:
        """Store multiple candles, returns count of stored candles"""
        stored_count = 0
        for candle in candles:
            if self.store_candle(candle):
                stored_count += 1
        return stored_count

    def get_candles(self,
                   symbol: str,
                   timeframe: TimeFrame,
                   start_time: datetime | None = None,
                   end_time: datetime | None = None,
                   limit: int | None = None) -> list[MarketData]:
        """Get candles from MongoDB"""

        query = {
            "symbol": symbol,
            "timeframe": timeframe
        }

        # Add time range filter
        if start_time or end_time:
            time_filter = {}
            if start_time:
                time_filter["$gte"] = start_time
            if end_time:
                time_filter["$lte"] = end_time
            query["timestamp"] = time_filter

        cursor = self.market_data.find(query).sort("timestamp", ASCENDING)

        if limit:
            cursor = cursor.limit(limit)

        return [MarketData.from_mongo_doc(doc) for doc in cursor]

    def get_latest_candle(self, symbol: str, timeframe: TimeFrame) -> MarketData | None:
        """Get the latest candle for a symbol/timeframe"""
        doc = self.market_data.find_one(
            {"symbol": symbol, "timeframe": timeframe},
            sort=[("timestamp", -1)]
        )
        return MarketData.from_mongo_doc(doc) if doc else None

    def get_candles_df(self,
                      symbol: str,
                      timeframe: TimeFrame,
                      start_time: datetime | None = None,
                      end_time: datetime | None = None,
                      limit: int | None = None) -> pd.DataFrame:
        """Get candles as pandas DataFrame for backtesting"""

        candles = self.get_candles(symbol, timeframe, start_time, end_time, limit)

        if not candles:
            return pd.DataFrame()

        # Convert to DataFrame
        data = []
        for candle in candles:
            row = {
                "timestamp": candle.timestamp,
                "open": float(candle.open),
                "high": float(candle.high),
                "low": float(candle.low),
                "close": float(candle.close),
                "volume": float(candle.volume)
            }
            # Add indicators
            row.update(candle.indicators)
            data.append(row)

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def store_indicators(self,
                        symbol: str,
                        timeframe: TimeFrame,
                        timestamp: datetime,
                        indicators: dict[str, float]) -> bool:
        """Store indicators for a specific candle"""
        try:
            self.market_data.update_one(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": timestamp
                },
                {"$set": {"indicators": indicators}}
            )
            return True
        except Exception as e:
            print(f"Error storing indicators: {e}")
            return False

    def get_symbols(self) -> list[str]:
        """Get all available symbols"""
        return self.market_data.distinct("symbol")

    def get_timeframes(self, symbol: str) -> list[str]:
        """Get available timeframes for a symbol"""
        return self.market_data.distinct("timeframe", {"symbol": symbol})

    def get_date_range(self, symbol: str, timeframe: TimeFrame) -> tuple[datetime, datetime] | None:
        """Get date range for symbol/timeframe"""
        pipeline = [
            {"$match": {"symbol": symbol, "timeframe": timeframe}},
            {"$group": {
                "_id": None,
                "min_date": {"$min": "$timestamp"},
                "max_date": {"$max": "$timestamp"}
            }}
        ]

        result = list(self.market_data.aggregate(pipeline))
        if result:
            return result[0]["min_date"], result[0]["max_date"]
        return None

    def count_candles(self, symbol: str, timeframe: TimeFrame) -> int:
        """Count candles for symbol/timeframe"""
        return self.market_data.count_documents({
            "symbol": symbol,
            "timeframe": timeframe
        })

    def delete_candles(self,
                      symbol: str,
                      timeframe: TimeFrame,
                      start_time: datetime | None = None,
                      end_time: datetime | None = None) -> int:
        """Delete candles, returns count deleted"""
        query = {
            "symbol": symbol,
            "timeframe": timeframe
        }

        if start_time or end_time:
            time_filter = {}
            if start_time:
                time_filter["$gte"] = start_time
            if end_time:
                time_filter["$lte"] = end_time
            query["timestamp"] = time_filter

        result = self.market_data.delete_many(query)
        return result.deleted_count

    def get_database_stats(self) -> dict[str, Any]:
        """Get database statistics"""
        stats = {
            "total_candles": self.market_data.count_documents({}),
            "symbols": len(self.get_symbols()),
            "database_size_mb": self.db.command("dbstats")["dataSize"] / 1024 / 1024
        }

        # Get per-symbol stats
        pipeline = [
            {"$group": {
                "_id": {"symbol": "$symbol", "timeframe": "$timeframe"},
                "count": {"$sum": 1},
                "min_date": {"$min": "$timestamp"},
                "max_date": {"$max": "$timestamp"}
            }}
        ]

        symbol_stats = {}
        for doc in self.market_data.aggregate(pipeline):
            symbol = doc["_id"]["symbol"]
            timeframe = doc["_id"]["timeframe"]

            if symbol not in symbol_stats:
                symbol_stats[symbol] = {}

            symbol_stats[symbol][timeframe] = {
                "candles": doc["count"],
                "date_range": (doc["min_date"], doc["max_date"])
            }

        stats["symbols_detail"] = symbol_stats
        return stats

    def close(self):
        """Close MongoDB connection"""
        self.client.close()
