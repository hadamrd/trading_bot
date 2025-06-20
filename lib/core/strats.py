from lib.core.Action import Action
from lib.core.FeaturesEnum import FeatureEnum
from lib.core.Strat import Strat


class MACDRSI(Strat):
    
    def __init__(self, oversold_threshold=30, overbought_threshold=70, bullish_threshold=0, bearish_threshold=0, cut_loss=0.02):
        super().__init__(cut_loss)
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
    
    def get_action(self, row):
        macd = row[FeatureEnum.MACD]
        macdsignal = row[FeatureEnum.MACD_SIGNAL]
        rsi = row[FeatureEnum.RSI]
        bullish_signal = (macd - macdsignal) > self.bullish_threshold
        bearich_signal = (macd - macdsignal) < self.bearish_threshold
        over_bought_signal = rsi > self.overbought_threshold
        over_sold_signal = rsi < self.oversold_threshold
        if over_sold_signal and bullish_signal:
            return Action.BUY
        elif over_bought_signal and bearich_signal:
            return Action.SELL
        else:
            return Action.HOLD

class CustomStrat(Strat):
    
    def __init__(self, oversold_threshold=30, overbought_threshold=70, bullish_threshold=0, bearish_threshold=0, cut_loss=0.02, bbp_threshold=0.2):
        super().__init__(cut_loss)
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.bbp_threshold = bbp_threshold
    
    def get_action(self, row):
        vwap = row['volume_vwap']
        rsi = row['momentum_rsi']
        tenkan_sen = row['trend_ichimoku_conv']
        kijun_sen = row['trend_ichimoku_base']
        close_price = row['close']
        
        macd = row['trend_macd']
        macd_signal = row['trend_macd_signal']
        
        # MACD crossover logic
        macd_above_signal = macd > macd_signal
        macd_below_signal = macd < macd_signal

        # Ichimoku Cloud logic
        bullish_ichimoku = tenkan_sen > kijun_sen
        bearish_ichimoku = tenkan_sen < kijun_sen
        
        # VWAP logic
        above_vwap = close_price > vwap
        below_vwap = close_price < vwap
        
        # RSI logic
        over_sold = rsi < 30
        over_bought = rsi > 70
        
        if macd_above_signal and bullish_ichimoku and above_vwap and over_sold:
            return Action.BUY
        elif macd_below_signal or bearish_ichimoku or below_vwap or over_bought:
            return Action.SELL
        else:
            return Action.HOLD

