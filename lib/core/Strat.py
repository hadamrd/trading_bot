from functools import lru_cache

from lib.core.TradeConfig import TradeConfig


class Strat:
    
    def __init__(self, config: TradeConfig, cut_loss, profit_margin) -> None:
        self.config = config
        self.cut_loss = cut_loss
        self.profit_margin = profit_margin
        self.trigger_stop_loss_time = None
        self.long_min_rpv = self.compute_long_min_rpv(self.config.tr_fee_rate, self.profit_margin)
        self.long_loss_rpv = self.compute_long_min_rpv(self.config.tr_fee_rate, - self.cut_loss)
    
    @lru_cache(maxsize=None)
    def compute_long_min_rpv(self, tr_fee_rate, profit_margin):
        return (2 * tr_fee_rate + profit_margin * (1 + tr_fee_rate)) / (1 - tr_fee_rate)
    
    def get_action(self, row):
        raise NotImplementedError()