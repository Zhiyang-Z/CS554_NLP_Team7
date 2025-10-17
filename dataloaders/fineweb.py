from torch.utils.data import IterableDataset, get_worker_info
from collections import deque
import numpy as np
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import os

class PretrainDataset(IterableDataset):
    def __init__(self, dataset_name, subname, tokenizer, pretrain_len=1024):
        # NOTICE: dataset is a streaming dataset, make sure to split it by node and worker
        self.dataset_name, self.subname = dataset_name, subname
        self.dataset = None
        self.tokenizer = tokenizer
        self.eot_id = tokenizer.eos_token_id
        self.pretrain_len = pretrain_len
        self.buffer = deque() # use deque for performance, No other thread in one worker, it's safe.
        self.it = None # iterator for dataset, None represents need to start over.

        self.set_and_shuffle_dataset(42)

    def set_and_shuffle_dataset(self, seed):
        # dataset
        dataset = load_dataset(self.dataset_name, name=self.subname, split="train", streaming=True)
        dataset = dataset.shuffle(seed=seed) # shuffle it, just a routine, we will handle shuffle for each iteration in training loop.
        # split by node
        dataset = split_dataset_by_node(dataset, rank=int(os.environ['RANK']), world_size=int(os.environ['WORLD_SIZE']))
        self.dataset = dataset

    def __iter__(self):
        cur_idx = -1 # current iteration id for split data for workers.
        worker_info = get_worker_info()
        worker_id, num_workers = 0, 1
        if worker_info is not None:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
        while True:
            if len(self.buffer) >= self.pretrain_len:
                input_ids = [self.buffer.popleft() for _ in range(self.pretrain_len)]
                yield np.array(input_ids)
            else:
                self.it = iter(self.dataset) if self.it is None else self.it
                try:
                    while len(self.buffer) < self.pretrain_len:
                        item = next(self.it)
                        # For IterableDataset, it's useless to launch multiple workers???
                        # cur_idx += 1
                        # assert cur_idx >= 0
                        # if cur_idx % num_workers != worker_id:
                        #     continue # skip data not assigned to this worker
                        tokens = self.tokenizer(item['text'], truncation=False, max_length=None)['input_ids']
                        assert tokens[0] != self.eot_id
                        tokens.insert(0, self.eot_id) # add eot at the beginning
                        self.buffer.extend(tokens)
                except StopIteration:
                    assert len(self.buffer) < self.pretrain_len
                    self.it, cur_idx = None, -1 # reset iterator to None, so next time we start over.
                    self.buffer.clear() # drop the remaining tokens.
                    return # Don't use raise StopIteration
                # if stand here, buffer is replenished.
                assert len(self.buffer) >= self.pretrain_len
                input_ids = [self.buffer.popleft() for _ in range(self.pretrain_len)]
                yield np.array(input_ids)
