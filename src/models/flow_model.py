from __future__ import annotations
import torch
import torch.nn as nn
from model import BaseModel

class Flow_Model(BaseModel):
    """
    ELF Model
    """
    def __init__(self, config):
        
        super().__init__(config)
        self.mse = nn.MSELoss(reduction="mean")
    
    def Flow_Loss(self,
                  x: torch.Tensor,
                  x_clean: torch.Tensor,
                  e: torch.Tensor,
                  z: torch.Tensor,
                  z_pred: torch.Tensor,
                  t: torch.Tensor,
                  ground_truth: torch.Tensor,
                  weight: float = 1.0,
        ):
        """
        ELF Loss: (https://arxiv.org/pdf/2605.10938)

        Args:
            x (torch.Tensor): input of shape (``B``, ``L``, ``N``) (Contains presoftmax decoder projection)
            x_clean (torch.Tensor): input of shape (``B``, ``L``, ``E``) (Contains decoded x)
            e (torch.Tensor): input of shape (``B``, ``L``, ``E``) (Contains noise)
            z (torch.Tensor): input of shape (``B``, ``L``, ``E``) (Contains z_t = t * x + (1-t) * e)
            z_pred (torch.Tensor): input of shape (``B``, ``L``, ``E``) (Contains predicted encode(x))
            t (torch.Tensor): input of shape (``B``) where contains the time for the flow matching
            ground_truth (torch.Tensor): input of shape (``B``, ``L``) where contains ground truth class indices
            weight (float): weight on flow loss

        Returns:
            loss (scalar)
        """

        loss_decode = self.autoreg_loss(x=x, ground_truth=ground_truth)

        v = x_clean - e
        v_pred = (z_pred - z) / (1 - t)
        loss_flow = self.mse(v_pred, v)
      
        return loss_decode + loss_flow * weight
    
    def forward(self,
            x: torch.Tensor,
            t: torch.Tensor, 
            ground_truth: torch.Tensor | None = None,
        ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input of shape (``B``, ``L``)
            t (torch.Tensor): input of shape (``B``) where contains the time for the flow matching
            ground_truth (torch.Tensor): input of shape (``B``, ``L``) where contains ground truth class indices

        Returns:
            attn_output (torch.Tensor): output of shape (B, L, E)
        """

        # (B) -> (B, 1, 1)
        t = t[:, None, None]
        x_clean = self.encode(x).detach()
        
        # (B, L, E)
        e = torch.randn_like(x_clean, **self.factory_kwargs)
        z = t * x_clean + (1 - t) * e
        
        # (B, L, E) -> (B, L, E)
        z_pred, _ = self.run_layers(z)

        # (B, L, E) -> (B, L, N)
        x = self.decode(z_pred)
         
        # (B, L, N) -> scalar
        loss = self.Flow_Loss(x=x,
                            x_clean=x_clean, 
                            e=e, 
                            z=z, 
                            z_pred=z_pred, 
                            t=t, 
                            ground_truth=ground_truth
                )
        
        return x, loss
    
    def generate(self,
            prompt: torch.Tensor,
            L_r: int,
            ts: torch.Tensor, 
        ) -> torch.Tensor:
        """
        Args:
            prompt (torch.Tensor): input of shape (``B``, ``L``) (pad tokens at the end)
            L_r (int): Length of new sequence
            ts (torch.Tensor): input of shape (``B``, ``T``) where contains the time for the flow matching

        Returns:
            x (torch.Tensor): output of shape (B, L) (contains token ids)
        """
        # (B, L_p)
        B, L_p = prompt.shape
        # (B, L_p)
        prompt_no_pad = prompt != self.pad_id
        # (B)
        prompt_lengths = torch.sum(prompt_no_pad, -1)
        # (L) -> (1, L) < (B, 1) -> (B, L) -> (B, L, 1)
        prompt_mask = (torch.arange(L_p + L_r, **self.factory_kwargs)[None, :] < prompt_lengths[:, None])[:, :, None]

        # (B, L_p, E)
        z_prompt = self.encode(prompt)
        # (B, L, E)
        z_rand = torch.randn((B, L_p+L_r, self.E), **self.factory_kwargs) * (~prompt_mask)
        z = torch.zeros_like(z_rand, **self.factory_kwargs)
        z[:, :L_p, :] = z_prompt
        # (B, L, E)
        z = z * prompt_mask + z_rand

        _, T = ts.shape
        for i in range(T - 1):
            # (B, T) -> (B)
            t = ts[:, i]
            # (B, T) -> (B)
            dt = ts[:, i + 1] - ts[i]
            # (B) -> (B, 1, 1)
            t = t[:, None, None]
            
            
            # (B, L, E) -> (B, L, E)
            z_pred, _ = self.run_layers(z_pred)

            # (B, L, E)
            v = (z_pred - z) / (t - 1)
            # (B, L, E)
            z = z + dt * v * (~prompt_mask)

        # (B, L, E) -> (B, L, N)
        x = self.decode(z_pred)

        # (B, L)
        x = torch.argmax(x, dim=-1)
        
        return x


        
        
        