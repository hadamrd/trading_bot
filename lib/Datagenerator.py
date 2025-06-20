import random
import numpy as np
from lib.ModelConfig import ModelConfig


class Datagenerator:
    
    def __init__(self, model_config: ModelConfig, timeframes_inputs, timeframes_timestamps, y, context, shuffle=True, train_ratio=0.8):
        self.y = y
        self.inputs = timeframes_inputs
        self.timestamps = timeframes_timestamps
        self.config = model_config
        max_idx = len(self.timestamps['1m']) - self.config.look_forward_minutes
        split_idx = int(max_idx * train_ratio)
        if context == "train":
            start_idx = self.config.timeframes['1m'].seq_length
            end_idx = split_idx
        elif context == "test":
            start_idx = split_idx
            end_idx = max_idx
        else:
            raise ValueError("context must be either 'train' or 'test'")
        # print pourcentage of each class
        labels = self.y[start_idx:end_idx]
        for i in range(self.config.num_classes):
            print(f"{context} Class {i} : {100 * (labels == i).sum() / len(labels)}%")
        self.total_samples = end_idx - start_idx + 1
        self.batch_size = self.config.training.batch_size
        self.indices = list(range(start_idx, end_idx))
        self.cache = {}
        if shuffle:
            random.shuffle(self.indices)
        
    def __len__(self):
        return self.total_samples // self.batch_size

    def __iter__(self):
        for i in range(0, len(self.indices), self.batch_size):
            batch_data = {}
            batch_indices= self.indices[i: min(i + self.batch_size, len(self.indices))]
            for tfk in self.config.timeframes:
                timeframe_batch_data = np.array([self.select_and_pad_rows(tfk, idx) for idx in batch_indices])
                timeframe_batch_data = np.transpose(timeframe_batch_data, (0, 2, 1))
                batch_data[tfk] = timeframe_batch_data
            if self.y is not None:
                batch_labels = self.y[batch_indices]
                yield batch_data, batch_labels
            else:
                yield batch_data
    
    def select_and_pad_rows(self, tfk, idx):
        t = self.timestamps['1m'][idx]
        n = self.config.timeframes[tfk].seq_length
        if (tfk, t) in self.cache:
            l_idx = self.cache[(tfk, t)]
        else:
            l_idx = np.searchsorted(self.timestamps[tfk], t, side='left')
            self.cache[(tfk, t)] = l_idx
        start_idx = max(0, l_idx - n)
        selected_rows = self.inputs[tfk][start_idx:l_idx, :]
        if selected_rows.shape[0] < n:
            pad_count = n - selected_rows.shape[0]
            padding = np.zeros((pad_count, self.inputs[tfk].shape[1]))
            selected_rows = np.vstack((padding, selected_rows))
        return selected_rows
