from __future__ import annotations
import torch
import torch.nn as nn
from src.models.model import BaseModel

class Flow_Model(BaseModel):
    """
    ELF Model
    """
    def __init__(self, config):
        
        super().__init__(config)
        self.t_emb = nn.Sequential(
            nn.Linear(1, self.E, bias=self.bias, **self.factory_kwargs),
            nn.GELU(),
            nn.Linear(self.E, self.E, bias=self.bias, **self.factory_kwargs),
        )
        self.mode_emb = nn.Embedding(2, self.E, **self.factory_kwargs)
        self.mse = nn.MSELoss(reduction="mean")
        self.i = 0
    
    def Flow_Loss(self,
                  x_clean: torch.Tensor,
                  e: torch.Tensor,
                  z: torch.Tensor,
                  z_pred: torch.Tensor,
                  t: torch.Tensor,
                  prompt_mask: torch.Tensor,
        ):
        """
        ELF Loss: (https://arxiv.org/pdf/2605.10938)

        Args:
            x_clean (torch.Tensor): input of shape (``B``, ``L``, ``E``) (Contains decoded x)
            e (torch.Tensor): input of shape (``B``, ``L``, ``E``) (Contains noise)
            z (torch.Tensor): input of shape (``B``, ``L``, ``E``) (Contains z_t = t * x + (1-t) * e)
            z_pred (torch.Tensor): input of shape (``B``, ``L``, ``E``) (Contains predicted encode(x))
            t (torch.Tensor): input of shape (``B``, ``1``, ``1``) where contains the time for the flow matching
            prompt_mask (torch.Tensor): input of shape (``B``, ``L``, ``1``) where contains True if part of the mask

        Returns:
            loss (scalar)
        """

        # (B, L, E)
        v = (x_clean - e) * (~prompt_mask)
        v_pred = ((z_pred - z) / (1 - t)) * (~prompt_mask)
        loss_flow = self.mse(v_pred, v)
        # loss_flow = self.mse(x_clean, z_pred)
      
        return loss_flow
    
    def forward(self,
            x: torch.Tensor,
            ground_truth: torch.Tensor | None = None,
        ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input of shape (``B``, ``L``)
            ground_truth (torch.Tensor): input of shape (``B``, ``L``) where contains ground truth class indices (x = ground_truth)

        Returns:
            x (torch.Tensor): output of shape (B, L, N)
        """
        eps = 1e-2
        B, L = x.shape

        logits = loss = loss_flow = loss_decode = None

        progress = (self.i / self.num_steps)
        percent = .8 * (self.i / self.num_steps)**2

        # (B)
        prompt_len = torch.randint(0, int(L*.30), (B,), device=self.device)
        # (L) -> (1, L)
        seq = torch.arange(1, L + 1, 1, device=self.device, dtype=torch.int)[None, :]
        # (1, L) + [(B) -> (B, 1)] ->(B, L) -> (B, L, 1)
        prompt_mask = (seq <= prompt_len[:, None])[:, :, None]

        if torch.rand(()) < percent:
            # (B)
            t = torch.nn.functional.sigmoid(torch.randn((B), **self.factory_kwargs) * .8 - 1.5)
            t = t.clamp(0, 1 - eps)

            # (B) -> (B, 1) -> (B, E) -> (B, 1, E)
            t_emb = self.t_emb(t[:, None])[:, None, :]

            # (B) -> (B, E) -> (B, 1, E)
            mode_emb = self.mode_emb(torch.zeros(B, dtype=torch.long, device=self.device))[:, None, :]
            
            # (B) -> (B, 1, 1)
            t = t[:, None, None]

            # (B, L, E)
            x_clean = self.encode(x)
            x_clean_det = x_clean.detach()
            # (B, L, E)
            e = torch.randn_like(x_clean_det, **self.factory_kwargs) * 2

            # (B, L, E)
            z = t * x_clean_det + (1 - t) * e
            z = (prompt_mask) * x_clean_det + (~prompt_mask) * z
            # (B, L, E)
            z_in = z + t_emb + mode_emb
            # (B, L, E) -> (B, L, E)
            z_pred, _ = self.run_layers(z_in)

            if ground_truth is not None:
                # (B, L, N) -> scalar
                loss = self.Flow_Loss(x_clean=x_clean_det, 
                                    e=e, 
                                    z=z, 
                                    z_pred=z_pred, 
                                    t=t, 
                                    prompt_mask=prompt_mask,
                        )
                loss_flow = loss

        else:
            # (B, L, E)
            x_clean = self.encode(x)

            # (B, L, 1)
            p = torch.nn.functional.sigmoid(torch.randn((B, L, 1), **self.factory_kwargs) * .8 + .8)

            # (B, 1) -> (B, E) -> (B, 1, E)
            t_emb_dec = self.t_emb(torch.ones((B, 1), **self.factory_kwargs))[:, None, :]
            # (B) -> (B, E) -> (B, 1, E)
            mode_emb = self.mode_emb(torch.ones(B, dtype=torch.long, device=self.device))[:, None, :]
            # (B, L, E)
            e_dec = torch.randn_like(x_clean, **self.factory_kwargs) * 2
            # (B, L, E)
            z_dec = p * x_clean + (1 - p) * e_dec
            z_dec = (prompt_mask) * x_clean + (~prompt_mask) * z_dec
            # (B, L, E)
            z_dec = z_dec + t_emb_dec + mode_emb

            # (B, L, E)
            z_pred_dec, _ = self.run_layers(z_dec)

            # (B, L, E) -> (B, L, N)
            logits = self.decode(z_pred_dec)
            if ground_truth is not None:
                # (B, L, N) -> scalar
                loss = self.autoreg_loss(x=logits, ground_truth=ground_truth)
                loss_decode = loss
        
        
        
        self.i = self.i + 1
        return logits, loss, {"loss_flow": loss_flow, "loss_decode": loss_decode, "percent": torch.tensor([percent])}
    
    @torch.no_grad()
    def generate(self,
            prompt: torch.Tensor,
            L_r: int,
        ) -> torch.Tensor:
        """
        Args:
            prompt (torch.Tensor): input of shape (``B``, ``L``) (pad tokens at the end)
            L_r (int): Length of new sequence
            ts (torch.Tensor): input of shape (``B``, ``T``) where contains the time for the flow matching

        Returns:
            x (torch.Tensor): output of shape (B, L) (contains token ids)
        """
        self.eval()

        # (1, T)
        ts = torch.linspace(0.01, 1.0, 100, **self.factory_kwargs).unsqueeze(0)

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
        z_rand = torch.randn((B, L_p+L_r, self.E), **self.factory_kwargs) * (~prompt_mask) * 2
        z = torch.zeros_like(z_rand, **self.factory_kwargs)
        z[:, :L_p, :] = z_prompt
        # (B, L, E)
        z = z * prompt_mask + z_rand

        _, T = ts.shape
        for i in range(0, T - 1):
            # (B, T) -> (B)
            t = ts[:, i]
            
            # (B, T) -> (B)
            dt = ts[:, i + 1] - ts[:, i]
            # (B) -> (B, 1) -> (B, E) -> (B, 1, E)
            t_emb = self.t_emb(t[:, None])[:, None, :]
            # (B) -> (B, E) -> (B, 1, E)
            mode_emb = self.mode_emb(torch.zeros(B, dtype=torch.long, device=self.device))[:, None, :]
            # (B) -> (B, 1, 1)
            t = t[:, None, None]
            
            # (B, L, E)
            z_in = z + t_emb + mode_emb
            
            # (B, L, E) -> (B, L, E)
            z_pred, _ = self.run_layers(z_in)

            # (B, L, E)
            v = (z_pred - z) / (1 - t).clamp_min(1e-2)
            # (B, L, E)
            z = z + dt * v * (~prompt_mask)
        
        # (B) -> (B, 1) -> (B, E) -> (B, 1, E)
        t_emb_dec = self.t_emb(torch.ones(B, 1, **self.factory_kwargs))[:, None, :]
        # (B) -> (B, E) -> (B, 1, E)
        mode_emb = self.mode_emb(torch.ones(B, dtype=torch.long, device=self.device))[:, None, :]
    
        # (B, L, E)
        z_in_dec = z + t_emb_dec + mode_emb
        # (B, L, E) -> (B, L, E)
        z_pred_dec, _ = self.run_layers(z_in_dec)
           
        z = z * (prompt_mask) + z_pred_dec * (~prompt_mask)

        # (B, L, E) -> (B, L, N)
        x = self.decode(z)

        # (B, L)
        x = torch.argmax(x, dim=-1)
        
        return x


        
        
        