import numpy as np
import torch

class DataLoader:
    def __init__(self, config, train = True):
        if train:
            filename = config.train_path
        else:
            filename = config.valid_path
        self.data = np.memmap(filename=filename, dtype=config.token_dtype, mode="r")
        self.device = config.device
        self.causal = config.is_causal
        self.eot_id = config.eot_id
        self.pad_id = config.pad_id

    def get_batch(self, B: int, L: int):
        # (B)
        idx = np.random.randint(0, len(self.data) - 1 - L, size=B)

        x = np.stack([self.data[i:i+L] for i in idx])        
        if self.causal:
            y = np.stack([self.data[i+1:i+L+1] for i in idx])
        else:
            eot_seq = x == self.eot_id
            eot_seq_cum = np.cumsum(eot_seq, -1, np.int64) - eot_seq
            x = np.where(eot_seq_cum == 0, x, self.pad_id)
            y = x.copy()
            

        x = torch.from_numpy(x.astype(np.int64)).to(self.device)
        y = torch.from_numpy(y.astype(np.int64)).to(self.device)

        return x, y
