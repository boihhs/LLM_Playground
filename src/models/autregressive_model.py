from __future__ import annotations
import torch
import torch.nn as nn
from src.models.model import BaseModel

class Autoreg_Model(BaseModel):
    """
    Autoregressive Model
    """
    def __init__(self, config):
        
        super().__init__(config)
        
    def forward(
        self,
        x: torch.Tensor,
        ground_truth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input of shape (``B``, ``L``)
            ground_truth (torch.Tensor): input of shape (``B``, ``L``) where contains ground truth class indices

        Returns:
            attn_output (torch.Tensor): output of shape (B, L, E)
        """
        # (B, L)
        valid_tokens = x != self.pad_id

        # (B, L) -> (B, L, E)
        x = self.encode(x)
        
        # (B, L, E) -> (B, L, E)
        x, _ = self.run_layers(x, valid_tokens)

        # (B, L, E) -> (B, L, N)
        x = self.decode(x)
        
        loss = None
        if ground_truth is not None:
            # (B, L, N) -> scalar
            loss = self.autoreg_loss(x, ground_truth)
        
        # (B, L, N)
        return x, loss # need to softmax later on
    
    def prefill(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): input of shape (``B``, ``L``)

        Returns:
            cache (list)
        """

        cache = [
            {"K": None, 
             "V": None,
             "c_1": None,
             "c_2": None,
             "c_3": None,}
            for _ in range(self.nlayers)
        ]

        if x.shape[1] > 1:
            z = x[:, :-1]
            # (B, L) -> (B, L, E)
            z = self.encode(z)
                
            # (B, L, E) -> (B, L, E)
            z, cache = self.run_layers(z, cache=cache, prefill=True)
        else:
            B, L = x.shape
            E_head = self.E // self.nheads

            cache = [
            {"K": torch.empty((B, self.nheads, 0, E_head), **self.factory_kwargs), 
             "V": torch.empty((B, self.nheads, 0, E_head), **self.factory_kwargs),
             "c_1": torch.empty((B, 0, self.E), **self.factory_kwargs),
             "c_2": torch.empty((B, 0, self.E), **self.factory_kwargs),
             "c_3": torch.empty((B, 0, self.E), **self.factory_kwargs),}
            for _ in range(self.nlayers)
        ]

        return cache

    
    def generate(self, x: torch.Tensor, max_gen: int):
        """
        Args:
            x (torch.Tensor): input of shape (``B``, ``L``)
            max_gen (int)

        Returns:
            x (torch.Tensor): output of shape (``B``, ``L + max_gen``) (token ids)
        """

        self.eval()

        B, L = x.shape
        cache = self.prefill(x)
    
        i = 0
        # (B, L) -> (B, L) -> (B) -> (B) -> scalar
        cleared = ((x == self.eot_id).sum(-1) > 0).sum(-1)
        while (cleared != B) and (i < max_gen):
            y = x[:, -1:]
            # (B, 1) -> (B, 1, E)
            y = self.encode(y)
            
            # (B, 1, E) -> (B, 1, E)
            y, cache = self.run_layers(y, cache=cache)

            # (B, 1, E) -> (B, 1, N)
            y = self.decode(y)
            # (B, 1, N) -> (B, 1)
            y = torch.argmax(y, dim=-1)

            # (B, L + 1)
            x = torch.cat((x, y), dim=1)
            
            i = i + 1
            cleared = ((x == self.eot_id).sum(-1) > 0).sum(-1)

        return x





        
        
        