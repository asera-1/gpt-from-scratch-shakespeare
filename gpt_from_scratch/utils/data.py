
# utils/data.py

import torch
import random

class TokenDataset:
    def __init__(self, token_ids, block_size):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

def get_batch(dataset, batch_size):
    idxs = random.sample(range(len(dataset)), batch_size)
    x_batch = []
    y_batch = []
    for idx in idxs:
        x, y = dataset[idx]
        x_batch.append(x)
        y_batch.append(y)
    x_batch = torch.stack(x_batch)
    y_batch = torch.stack(y_batch)
    return x_batch, y_batch
