from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class CanonLayer(nn.Module):
    """
    Args:
        E (int): Size of embedding dim for query, key, and value
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
        is_causal (bool, optional): Whether to apply causal mask. Default: False
        canon (bool, optional): Whether to include canon (1D depthwise convolution) layers. Default: False
        canon_length (int, optinal): One side of the canon lenght not including the current token. Default: 3
        - If canon=True, canon_length=3,is_causal=True -> kernal_size=3+1=4
        - If canon=True, canon_length=3,is_causal=False -> kernal_size=2*3+1=7
    """
    def __init__(self, 
                 E: int,
                 dropout: float = 0,
                 canon_length: int = 3,
                 is_causal=False,
                 bias=True,
                 device=None,
                 dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.dropout = dropout
        self.canon_length = canon_length
        self.is_causal = is_causal
        self.dropout_layer = nn.Dropout(dropout)

        if self.is_causal:
            self.canon = nn.Conv1d(E, E, kernel_size=self.canon_length+1, padding=0, groups=E, bias=bias, **factory_kwargs)
        else:
            self.canon = nn.Conv1d(E, E, kernel_size=self.canon_length*2+1, padding=self.canon_length, groups=E, bias=bias, **factory_kwargs)

    def forward(
            self,
            x: torch.Tensor,
            kv_cache: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply Canon (1D depthwise convolution); runs the following process:
            1. Change dimensions to match convolution
            2. If causal, pad
            3. Conv and untranspose

        Args:
            x (torch.Tensor): input of shape (``B``, ``L``, ``E``) (L = 1 if using kv_cache)
            canon_layer (Bn.Conv1d)
            kv_cache (torch.Tensor): input of shape (``B``, ``canon_length (not in the beginning)``, ``E``)
        """
        # (B, L, E) -> (B, E, L)
        x = x.transpose(1, 2)

        if self.is_causal:
            padding = (self.canon_length, 0)

            if kv_cache is not None:
                _, L, _ = kv_cache.shape
                # (B, L, E) -> (B, E, L)
                kv_cache = kv_cache.transpose(1, 2)

                # (B, E, canon_length) + (B, E, 1) -> (B, E, canon_length + 1)
                x = torch.cat((kv_cache, x), dim=2)
                if L == self.canon_length:
                    pad = 0
                else:
                    pad = self.canon_length - L
                padding = (pad, 0)

            # (B, E, L) -> (B, E, L+canon_length)
            x = F.pad(x, padding, mode="constant", value=0.0)

        # (B, E, L+canon_length-1) -> (B, E, L) -> (B, L, E) or (B, E, L) -> (B, E, L) -> (B, L, E)
        out = self.dropout_layer(self.canon(x)).transpose(1, 2)

        # (B, L, E)
        return out