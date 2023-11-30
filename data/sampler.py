import torch
import random
import numpy as np
from itertools import islice


class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool = True, seed: int = 0) -> None:
        if isinstance(next(iter(data_source)), dict):
            first_key = next(iter(next(iter(data_source)).keys()))
            self.lengths = [len(d[first_key]) for d in data_source]
        else:
            self.lengths = [len(d) for d in data_source]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        ids = list(range(len(self.lengths)))
        if self.drop_last: ids = ids[:len(ids) // self.batch_size * self.batch_size]
    
        if self.shuffle: random.Random(self.seed).shuffle(ids)
        batches = [ids[i:i+self.batch_size] for i in range(0, len(ids), self.batch_size)]   

        for b in batches: yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)


class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0) -> None:
        self.batch_sampler = LengthBasedBatchSampler(
            data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle, seed=seed
        )
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        max_length = len(
            self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)

    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas
