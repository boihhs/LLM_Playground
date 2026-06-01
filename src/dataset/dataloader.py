import numpy as np
import torch

class DataLoader:
    def __init__(self, path: str, dtype=np.uint16, device="cuda"):
        self.data = np.memmap(filename=path, dtype=dtype, mode="r")
        self.device = device

    def get_batch(self, B: int, L: int):
        # (B)
        idx = np.random.randint(0, len(self.data) - 1 - L, size=B)

        # (B, L)
        x = np.stack([self.data[i:i+L] for i in idx])
        y = np.stack([self.data[i+1:i+L+1] for i in idx])

        x = torch.from_numpy(x.astype(np.int64)).to(self.device)
        y = torch.from_numpy(y.astype(np.int64)).to(self.device)

        return x, y
