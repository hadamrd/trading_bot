from dataclasses import dataclass, field
import json

@dataclass
class TradeConfig:
    tr_fee_rate: float
    hourly_borrow_rate: float
    min_trade_amount: float
    min_price_movement: float
    margin_level: float = 1.1
    minute_borrow_rate: float = field(init=False)

    def __post_init__(self):
        self.minute_borrow_rate = self.hourly_borrow_rate / 60

    def to_json_file(self, file_name):
        config_dict = {
            'tr_fee_rate': self.tr_fee_rate,
            'hourly_borrow_rate': self.hourly_borrow_rate,
            'margin_level': self.margin_level,
            'min_trade_amount': self.min_trade_amount,
            'min_price_movement': self.min_price_movement,
        }
        with open(file_name, "w") as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def from_json(cls, json_obj):
        return cls(**json_obj)

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, 'r') as f:
            json_obj = json.load(f)
        return cls.from_json(json_obj)