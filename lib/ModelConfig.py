import json
from dataclasses import dataclass, field
from math import pow
from typing import Dict, List, Tuple
from typing import Optional


@dataclass
class TrainingConfig:
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
@dataclass
class TimeframeConfig:
    kernel_specs: List[List[int]]
    seq_length: int

@dataclass
class ModelConfig:
    look_forward_days: int
    cut_loss: float
    target_monthly_return: float
    selected_features: List[str]
    dropout_prob: float
    class_weights: List[float]
    fc_layer_sizes: List[int]
    timeframes: Dict[str, TimeframeConfig]
    training: TrainingConfig
    target_daily_return: Optional[float] = None
    name: Optional[float] = None
    num_features: Optional[int] = None
    look_forward_minutes: Optional[int] = None
    target_lookforward_return : Optional[float] = None
    num_classes: Optional[int] = None

    def __post_init__(self):
        if self.target_daily_return is None and self.target_monthly_return is not None:
            self.target_daily_return = pow(self.target_monthly_return, 1 / 30) - 1
        self.target_lookforward_return = pow(1 + self.target_daily_return, self.look_forward_days) - 1
        self.num_features = len(self.selected_features)
        self.look_forward_minutes = int(self.look_forward_days * 60 * 24)
        self.num_classes = len(self.class_weights)

    @staticmethod
    def from_json_file(model_config_file):
        with open(model_config_file, "r") as f:
            model_config = json.load(f)
        model_config['timeframes'] = {k: TimeframeConfig(**v) for k, v in model_config['timeframes'].items()}
        model_config['training'] = TrainingConfig(**model_config['training'])
        return ModelConfig(**model_config)

    def to_json_file(self, file_name):
        config_dict = self.__dict__.copy()
        config_dict['timeframes'] = {k: v.__dict__ for k, v in self.timeframes.items()}
        config_dict['training'] = self.training.__dict__
        with open(file_name, "w") as f:
            json.dump(config_dict, f, indent=4)