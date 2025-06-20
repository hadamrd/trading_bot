import pandas as pd
from datetime import datetime
from typing import Optional, Tuple

from tradingbot2.strategies.BaseStrategy import BaseStrategy
from .src.trading_bot.core.models import BacktestConfig
from lib.core.DataManager import DataManager as CoreDataManager

class DataManager:
    @staticmethod
    def fetch_data(symbol: str, timeframe: str, since_date: datetime, end_date: Optional[datetime] = None) -> pd.DataFrame:
        since_date_str = since_date.strftime("%Y-%m-%d")
        try:
            df = CoreDataManager.get_frame_data_csv(
                symbol=symbol,
                timeframe=timeframe,
                since_date_str=since_date_str,
                with_tech_indicators=True,
            )
        except FileNotFoundError:
            df = CoreDataManager.fetch_klines(
                symbol, timeframe, since_date_str=since_date_str
            )
            CoreDataManager.save_frame_data_csv(symbol, timeframe, since_date_str, True, df)
        
        if end_date is None:
            current_utc_timestamp = datetime.utcnow().timestamp()
            last_timestamp = df.iloc[-1]["timestamp"]
            if (current_utc_timestamp * 1000 - last_timestamp) / (1000 * 60) >= 60 * 10:
                remaining_df = CoreDataManager.fetch_klines(
                    symbol,
                    timeframe,
                    since_timestamp=int(last_timestamp) + 1,
                )
                df = pd.concat([df, remaining_df])
        
        return df

    @staticmethod
    def prepare_data(df: pd.DataFrame, config: BacktestConfig, strategy: BaseStrategy) -> pd.DataFrame:
        df = strategy.calculate_indicators(df)
        df = DataManager.filter_dates(df, config.test_start_date, config.test_end_date)
        return df

    @staticmethod
    def filter_dates(df: pd.DataFrame, start_date: datetime, end_date: Optional[datetime]) -> pd.DataFrame:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        start_date = pd.to_datetime(start_date).tz_localize("UTC")
        df = df[df["datetime"] >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date).tz_localize("UTC")
            df = df[df["datetime"] <= end_date]
        return df

    @staticmethod
    def get_symbol_data(config: BacktestConfig, symbol: str, strategy: BaseStrategy) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = DataManager.fetch_data(symbol, config.timeframe, config.since_date, config.test_end_date)
        full_df = strategy.calculate_indicators(df)
        test_df = DataManager.filter_dates(full_df, config.test_start_date, config.test_end_date)
        return full_df, test_df
