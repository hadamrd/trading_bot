import json
import os
from time import perf_counter

import numpy as np
import pandas as pd

from lib.core.fetch_kline_data import (fetch_historical_data,
                                       timeframe_to_seconds)
from lib.core.TradeConfig import TradeConfig
from lib.ModelConfig import ModelConfig


class DataManager:
    data_DIR = "data"
    MARKET_DATA_DIR = os.path.join(data_DIR, "market_data")
    TRAIN_DATA_DIR = os.path.join(data_DIR, "training_data")
    TRADE_CONFIG_DIR = "trade_configs"
    MODELS_DIR = "models"

    @classmethod
    def get_model_dir(cls, model_name):
        model_dir = os.path.join(cls.MODELS_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)
        return model_dir
    
    @classmethod
    def add_model(cls, model_name, model_config: ModelConfig):
        model_dir = os.path.join(cls.MODELS_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)
        model_config_file = os.path.join(model_dir, f"config.json")
        model_config.to_json_file(model_config_file)

    @classmethod
    def get_model_version_file(cls, model_name, model_version_file):
        model_dir = os.path.join(cls.MODELS_DIR, model_name)
        return os.path.join(model_dir, model_version_file)

    @classmethod
    def save_frame_data_csv(
        cls,
        symbol,
        timeframe,
        since_date_str,
        with_tech_indicators,
        data
    ):
        if with_tech_indicators:
            results_folder = os.path.join(cls.MARKET_DATA_DIR, symbol, "with_indicators")
        else:
            results_folder = os.path.join(cls.MARKET_DATA_DIR, symbol, "raw")
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        since = since_date_str.replace('-', '_')
        if timeframe == "1M":
            timeframe = "1Month"
        if with_tech_indicators:
            result_filename = f"{symbol}_{timeframe}_since_{since}_indicators.csv"
        else:
            result_filename = f"{symbol}_{timeframe}_since_{since}.csv"
        result_path = os.path.join(results_folder, result_filename)
        data.to_csv(result_path, index=False)
        
    @classmethod
    def get_frame_data_csv(
        cls,
        symbol="BTCUSDT",
        timeframe="1m",
        since_date_str="2013_01_01",
        with_tech_indicators=False,
    ) -> pd.DataFrame:
        if timeframe == "1M":
            timeframe = "1Month"
        symbol_dir = os.path.join(cls.MARKET_DATA_DIR, symbol)
        file_name = f"{symbol}_{timeframe}_since_{since_date_str.replace('-', '_')}"
        if with_tech_indicators:
            file_path = os.path.join(
                symbol_dir,
                "with_indicators",
                f"{file_name}_indicators.csv",
            )
        else:
            file_path = os.path.join(
                symbol_dir,
                "raw",
                f"{file_name}.csv",
            )
        df = pd.read_csv(file_path)
        return df

    @classmethod
    def get_frame_data(cls, symbol, timeframe):
        symbol_dir = os.path.join(cls.TRAIN_DATA_DIR, symbol)
        if timeframe == "1M":
            timeframe = "1Month"
        filename = f"{symbol}_{timeframe}_X.npy"
        file_path = os.path.join(symbol_dir, "X", filename)
        return np.load(file_path, mmap_mode="r")

    @classmethod
    def get_model_predictions(cls, model_name, model_version_file):
        model_dir = cls.get_model_dir(model_name)
        predictions_file = os.path.join(model_dir, f"pred_labels_{model_version_file}.npy")
        return np.load(predictions_file, mmap_mode="r")
    
    @classmethod
    def get_timestamp_data(cls, symbol, timeframe, ctx=None, lf_minutes=None):
        symbol_dir = os.path.join(cls.TRAIN_DATA_DIR, symbol)
        if timeframe == "1M":
            timeframe = "1Month"
        filename = f"{symbol}_{timeframe}_timestamps.npy"
        file_path = os.path.join(symbol_dir, "timestamps", filename)
        ts = np.load(file_path, mmap_mode="r")
        if ctx is not None:
            ts = cls.get_traintest_slice(ts, ctx, lf_minutes)
        return ts
    
    @classmethod
    def get_traintest_slice(cls, data, ctx, lf=None):
        if lf:
            data = data[:-lf]
        if ctx == "test":
            data = data[int(len(data) * 0.8) + 1:]
        elif ctx == "train":
            data = data[:int(len(data) * 0.8)]
        return data
    
    @classmethod
    def get_price_data(cls, symbol, timeframe, ctx=None, lf_minutes=None):
        symbol_dir = os.path.join(cls.TRAIN_DATA_DIR, symbol)
        if timeframe == "1M":
            timeframe = "1Month"
        filename = f"{symbol}_{timeframe}_price.npy"
        file_path = os.path.join(symbol_dir, "price", filename)
        p = np.load(file_path, mmap_mode="r")
        if ctx is not None:
            p = cls.get_traintest_slice(p, ctx, lf_minutes)
        return p
    
    @classmethod
    def get_trade_config(cls, symbol) -> TradeConfig:
        symbol_dir = os.path.join(cls.MARKET_DATA_DIR, symbol)
        symbol_trade_config_file = os.path.join(
            symbol_dir, f"{symbol}_trade_config.json"
        )
        return TradeConfig.from_json_file(symbol_trade_config_file)

    @classmethod
    def get_model_config(cls, model_name) -> ModelConfig:
        model_dir = os.path.join(cls.MODELS_DIR, model_name)
        model_config_file = os.path.join(model_dir, f"config.json")
        return ModelConfig.from_json_file(model_config_file)

    @classmethod
    def fetch_klines(
        cls, symbol, timeframe, since_date_str=None, since_timestamp=None, until=None
    ) -> pd.DataFrame:
        output_folder = os.path.join(cls.MARKET_DATA_DIR, symbol)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        return fetch_historical_data(
            symbol, 
            timeframe, 
            since_date_str=since_date_str,
            since_timestamp=since_timestamp,
            until=until
        )

    @classmethod
    def get_frames_train_data(self, symbol):
        timeframe_inputs = {}
        timeframe_timestamps = {}
        for fr in timeframe_to_seconds:
            # s = perf_counter()
            timeframe_inputs[fr] = DataManager.get_frame_data(symbol=symbol, timeframe=fr)
            timeframe_timestamps[fr] = DataManager.get_timestamp_data(symbol=symbol, timeframe=fr)
            # print(f"Loaded {fr} data with shape {timeframe_inputs[fr].shape} in {perf_counter() - s:.2f} seconds")
        y = DataManager.get_labels(symbol)
        return timeframe_inputs, timeframe_timestamps, y

    @classmethod
    def get_labels(cls, symbol, ctx=None):
        y = np.load(os.path.join(DataManager.TRAIN_DATA_DIR, symbol, 'y.npy'), mmap_mode="r")
        if ctx == "train":
            y = y[:int(len(y) * 0.8)]
        elif ctx == "test":
            y = y[int(len(y) * 0.8) + 1:]
        return y
    
    @classmethod
    def save_model_version(cls, model_name, model_version, model_version_filename):
        import torch
        model_dir = os.path.join(cls.MODELS_DIR, model_name)
        model_version_file = os.path.join(model_dir, model_version_filename)
        torch.save(
            model_version.state_dict(),
            model_version_file,
        )
    
    @classmethod
    def get_symbol_traindata_dir(cls, symbol):
        return os.path.join(cls.TRAIN_DATA_DIR, symbol)
    
    @classmethod
    def get_symbol_datadir(cls, symbol):
        folder = os.path.join(cls.MARKET_DATA_DIR, symbol)
        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder
    
    @classmethod
    def get_symbol_train_data_dir(cls, symbol):
        folder = os.path.join(cls.TRAIN_DATA_DIR, symbol)
        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder
    
    @classmethod
    def save_trade_config(cls, symbol, data):
        data_dir = DataManager.get_symbol_datadir(symbol)
        outfile = os.path.join(data_dir, f"{symbol}_trade_config.json")
        with open(outfile, "w") as f:
            json.dump(data, f, indent=4)