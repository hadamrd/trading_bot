import numpy as np
import pandas as pd
import pywt

from rl_lab.trading_env.FeaturesEnum import FeatureEnum


class Market:
    def __init__(self, data_file, max_price, min_asset_buy, tr_fee_rate, ocr):
        self.tf = tr_fee_rate
        self.ocr = self.annual_to_minute_rate(ocr)
        self.min_asset_buy = min_asset_buy
        with open(data_file, "r") as f:
            self.df = pd.read_csv(f)
        self.max_price = max_price
        self.feature_bounds = {
            FeatureEnum.YEAR: (2000, 2100),
            FeatureEnum.MONTH: (1, 12),
            FeatureEnum.DAY: (1, 31),
            FeatureEnum.HOUR: (0, 23),
            FeatureEnum.MINUTE: (0, 59),
            FeatureEnum.OPEN: (0, max_price),
            FeatureEnum.HIGH: (0, max_price),
            FeatureEnum.LOW: (0, max_price),
            FeatureEnum.CLOSE: (0, max_price),
            FeatureEnum.WVDENOISED_CLOSE: (0, max_price),
            FeatureEnum.VOLUME: (0, 1e9),
            FeatureEnum.VWAP: (0, 1000000),
            FeatureEnum.OBV: (-1e9, 1e9),
            FeatureEnum.BBM: (0, 1000000),
            FeatureEnum.BBH: (0, 1000000),
            FeatureEnum.BBL: (0, 1000000),
            FeatureEnum.ATR: (0, 10000),
            FeatureEnum.MACD: (-5000, 5000),
            FeatureEnum.MACD_SIGNAL: (-5000, 5000),
            FeatureEnum.EMA_FAST: (0, 40000),
            FeatureEnum.EMA_SLOW: (0, 40000),
            FeatureEnum.RSI: (0, 100),
            FeatureEnum.STOCH: (0, 100),
            FeatureEnum.WR: (-100, 0),
            FeatureEnum.KST: (-1000, 1000),
            FeatureEnum.KST_SIG: (-1000, 1000),
            FeatureEnum.BBP: (-1, 1),
            FeatureEnum.ATR: (0, 10000),
        }
        self.selected_features = list(self.feature_bounds.keys())  # Default to all features
        self.calculate_feature_bounds()
        self.t = 0
    
    def dropna(self):
        self.df = self.df.dropna()
        
    def to_tuples(self):
        self.data = list(self.df.itertuples(index=False, name=None))
        
    def get_price(self, t):
        return self.df[FeatureEnum.CLOSE][t]
    
    def annual_to_minute_rate(self, r_annual):
        r_minute = (1 + r_annual) ** (1 / (365 * 24 * 60)) - 1
        return r_minute
    
    def get_window_flat(self, t, window_size):
        if t < window_size:
            raise ValueError("t must be greater or equal to the window_size")
        market_data = self.df.loc[t - window_size:t - 1, self.selected_features].values  # Select rows and specified columns
        return market_data.flatten()  # Flatten the 2D array to 1D
    
    def wavelet_denoise_price(self, wavelet='haar', level=1):
        original_length = len(self.df[FeatureEnum.CLOSE])
        coeffs = pywt.wavedec(self.df[FeatureEnum.CLOSE], wavelet, level=level)
        coeffs[-1] = np.zeros_like(coeffs[-1])
        denoised = pywt.waverec(coeffs, wavelet)
        if len(denoised) < original_length:
            denoised = np.pad(denoised, (np.NaN, original_length - len(denoised)))  # Pad to original length
        self.df[FeatureEnum.WVDENOISED_CLOSE] = denoised[:original_length]  # Safety slice in case it's too long

    def calculate_price_ema(self, span=20):
        return self.df[FeatureEnum.CLOSE].ewm(span=span, adjust=False).mean()

    def set_selected_features(self, selected_features):
        self.selected_features = selected_features
        self.calculate_feature_bounds()
        
    def calculate_feature_bounds(self):
        self.lower_bounds = [
            self.feature_bounds[feature][0]
            for feature in self.selected_features
        ]
        self.upper_bounds = [
            self.feature_bounds[feature][1]
            for feature in self.selected_features
        ]

    def get_date(self, t):
        row = self.df.iloc[t]
        year = int(row[FeatureEnum.YEAR])
        day = int(row[FeatureEnum.DAY])
        month = int(row[FeatureEnum.MONTH])
        hour = int(row[FeatureEnum.HOUR])
        minute = int(row[FeatureEnum.MINUTE])
        return f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"
    
    def get_direction(self, t):
        row = self.df.iloc[t]
        curr_price = row[FeatureEnum.CLOSE]
        if t == len(self.df) - 1:
            return 1
        next_price = self.df.iloc[t + 1][FeatureEnum.CLOSE]
        return 1 if next_price >= curr_price else -1
    
    @property
    def curr_price(self):
        return self.get_price(self.t)
    
    def tick(self):
        self.t += 1