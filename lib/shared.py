import glob
import os
import random
import traceback
from collections import Counter, deque
from multiprocessing import Manager, Pool

import numpy as np
import torch
from tqdm import tqdm


def check_gpu():
    # Check if GPU is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        # Get the name of the current GPU
        current_gpu = torch.cuda.current_device()
        print(
            f"Current GPU: {current_gpu}, Name: {torch.cuda.get_device_name(current_gpu)}"
        )
    else:
        print("GPU not available. Using CPU.")


def get_next_available_k(folder_path, name):
    # Find all existing GRU_* folders in the specified directory
    existing_folders = glob.glob(os.path.join(folder_path, f"{name}_*"))
    # Extract the values of k from the existing folder names
    existing_k_values = [int(folder.split("_")[-1]) for folder in existing_folders]
    # Determine the next available k
    if existing_k_values:
        next_k = max(existing_k_values) + 1
    else:
        next_k = 1
    return next_k


class SequenceDataLoader:
    def __init__(self, X, y, batch_size, seq_len, look_up, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.look_up = look_up
        self.end_idx = len(X) - seq_len - look_up
        self.indices = list(range(0, self.end_idx))
        if shuffle:
            random.shuffle(self.indices)

    def __len__(self):
        # This returns the number of batches this dataloader will produce
        return len(self.indices) // self.batch_size

    def __iter__(self):
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i : i + self.batch_size]
            batch_data = self.X[np.r_[np.array(batch_indices)[:, None] + np.arange(self.seq_len)]]
            batch_data = np.transpose(batch_data, (0, 2, 1))
            if self.y is not None:
                batch_labels = self.y[batch_indices + self.seq_len]
                yield batch_data, batch_labels
            else:
                yield batch_data


class SmoothMetrics:
    def __init__(self, window_size):
        self.values = deque(maxlen=window_size)

    def add(self, value):
        self.values.append(value)
        return sum(self.values) / len(self.values)


def parallel_chunk_processing(func, total, chunk_size, args_list, nprocesses=-1):
    """
    Apply a function in parallel on chunks of data.

    Parameters:
    - data: The data to be chunked and processed.
    - func: The function to apply on each chunk.
    - chunk_size: Size of each chunk.
    - nprocesses: Number of processes. If -1, it uses 80% of available CPU cores.
    - *args, **kwargs: Additional arguments and keyword arguments to pass to the func.

    Returns:
    A list of results obtained by applying the function on each chunk.
    """
    if nprocesses == -1:
        nprocesses = int(os.cpu_count() * 0.8)
    results_list = []
    # Using multiprocessing to process data in parallel
    with Manager() as manager:
        results = manager.list()
        # Create a shared progress bar
        with tqdm(total=total) as pbar:
            with Pool(nprocesses) as pool:
                def error_callback(e):
                    traceback.print_exception(type(e), e, e.__traceback__)
                    pool.terminate()
                for args in args_list:
                    pool.apply_async(
                        func,
                        args=(results, *args),
                        callback=lambda _: pbar.update(chunk_size),
                        error_callback=lambda e: error_callback(e)
                    )
                pool.close()
                pool.join()
        results_list.extend(results)
    return results_list

def get_classes_comp(y):
    label_counts = Counter(y)
    total = len(y)
    return {label: (count / total) * 100 for label, count in label_counts.items()}
