from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.CanonLayer import CanonLayer


class MultiHeadAttentionLayer(nn.Module):
    """
    MultiHeadAttention from https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
    Simplifed so all embedding dim are the same, E=E_out, and assumes self attention
    Computes multi-head attention.
    Canon, kv_cache, etc are added. 

    Args:
        E (int): Size of embedding dim for query, key, and value
        nheads (int): Number of heads. Each head has dim E_total // nheads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
        is_causal (bool, optional): Whether to apply causal mask. Default: False
        canon (bool, optional): Whether to include canon (1D depthwise convolution) layers. Default: False
        canon_length (int, optinal): One side of the canon lenght not including the current token. Default: 3
        - If canon=True, canon_length=3,is_causal=True -> kernal_size=3+1=4
        - If canon=True, canon_length=3,is_causal=False -> kernal_size=2*3+1=7
    """

    def __init__(
        self,
        E: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        is_causal=False,
        canon=False,
        canon_length: int = 3,
        eot_id: int = 1,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        
        self.packed_proj = nn.Linear(E, E * 3, bias=bias, **factory_kwargs)
        
        self.out_proj = nn.Linear(E, E, bias=bias, **factory_kwargs)
        assert E % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E // nheads
        self.bias = bias
        self.is_causal = is_causal

        self.up_proj = nn.Linear(E, E*4, bias=bias, **factory_kwargs)
        self.down_proj = nn.Linear(E*4, E, bias=bias, **factory_kwargs)
        self.non_linear = nn.GELU()

        self.layer_norm_1 = nn.LayerNorm(E, **factory_kwargs)
        self.layer_norm_2 = nn.LayerNorm(E, **factory_kwargs)

        self.dropout_layer = nn.Dropout(dropout)

        self.canon = canon
        if canon:
            self.canon_length = canon_length

        if canon:
            self.canon_layer_1 = CanonLayer(E, canon_length=self.canon_length, dropout=dropout, is_causal=self.is_causal, bias=bias, **factory_kwargs)
            self.canon_layer_2 = CanonLayer(E, canon_length=self.canon_length, dropout=dropout, is_causal=self.is_causal, bias=bias, **factory_kwargs)
            self.canon_layer_3 = CanonLayer(4*E, canon_length=self.canon_length, dropout=dropout, is_causal=self.is_causal, bias=bias, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        kv_cache = None,
        prefill = False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Compute attention mask
            3. Run SDPA
            4. Apply output projection

        Args:
            x (torch.Tensor): input of shape (``B``, ``L``, ``E``) (L = 1 if using kv_cache)
            attn_mask (torch.Tensor): input of shape (``B``, ``1``, ``L``, ``L``)
            kv_cache (dic[str -> torch.Tensor]): Contains: "K", "V" (Optional: "c_1", "c_2", "c_3")
            prefill (bool): Are we prefilling the cache
            
        Returns:
            attn_output (torch.Tensor): output of shape (B, L, E) (L = 1 if using kv_cache)
        """

        B, L, E = x.shape
        # Step 1. Apply input projection
        # (B, L, E)
        x_norm = self.layer_norm_1(x)
        if self.canon:
            if (kv_cache is not None) and (not prefill):
                # (B, C-1, E) + (B, 1, E) -> (B, C, E)
                cache = torch.cat((kv_cache["c_1"][:, 1:, :], x_norm), dim=1)
                x_norm = x_norm + self.canon_layer_1(x_norm, kv_cache["c_1"])
                kv_cache["c_1"] = cache
            else:
                x_norm = x_norm + self.canon_layer_1(x_norm)
                if prefill:
                    lenght = self.canon_length
                    if L < self.canon_length:
                        lenght = L
                    # (B, C, E)
                    kv_cache["c_1"] = x_norm[:, -lenght:, ]

        result = self.packed_proj(x_norm)
        query, key, value = torch.chunk(result, 3, dim=-1)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (B, L, E) -> (B, L, nheads, E_head) -> (B, nheads, L, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (B, L, E) -> (B, L, nheads, E_head) -> (B, nheads, L, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (B, L, E) -> (B, L, nheads, E_head) -> (B, nheads, L, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        if (kv_cache is not None) and (not prefill):
            # (B, nheads, L-1, E_head) + (B, nheads, 1, E_head) -> (B, nheads, L, E_head)
            key = torch.cat((kv_cache["K"], key), dim=2)
            value = torch.cat((kv_cache["V"], value), dim=2)
            kv_cache["K"] = key
            kv_cache["V"] = value

        if prefill:
            kv_cache["K"] = key
            kv_cache["V"] = value

        # Step 3. Run SDPA
        # (B, nheads, L, E_head)
        dropout_p = self.dropout if self.training else 0.0

        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p, is_causal=False, attn_mask=attn_mask
        )
        # (B, nheads, L, E_head) -> (B, L, nheads, E_head) -> (B, L, E)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (B, L, E) -> (B, L, E)
        x = self.dropout_layer(self.out_proj(attn_output)) + x

        x_norm = self.layer_norm_2(x)

        if self.canon:
            if (kv_cache is not None) and (not prefill):
                cache = torch.cat((kv_cache["c_2"][:, 1:, :], x_norm), dim=1)
                x_norm = x_norm + self.canon_layer_2(x_norm, kv_cache["c_2"])
                kv_cache["c_2"] = cache
            else:
                x_norm = x_norm + self.canon_layer_2(x_norm)
                if prefill:
                    # (B, C, E)
                    kv_cache["c_2"] = x_norm[:, -lenght:, ]

        # (B, L, E) -> (B, L, E*3) -> (B, L, E*3) -> (B, L, E)
        hidden = self.up_proj(x_norm)
        if self.canon:
            if (kv_cache is not None) and (not prefill):
                cache = torch.cat((kv_cache["c_3"][:, 1:, :], hidden), dim=1)
                hidden = hidden + self.canon_layer_3(hidden, kv_cache["c_3"])
                kv_cache["c_3"] = cache
            else:
                hidden = hidden + self.canon_layer_3(hidden)
                if prefill:
                    # (B, C, E)
                    kv_cache["c_3"] = hidden[:, -lenght:, ]

        output = x + self.dropout_layer(self.down_proj(self.non_linear(hidden)))

        return output, kv_cache