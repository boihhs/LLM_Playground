from __future__ import annotations
import torch
import torch.nn as nn
from ..modules.MultiHeadAttentionLayer import MultiHeadAttentionLayer
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """
    Args:
        E (int): Size of embedding dim for query, key, and value
        nheads (int): Number of heads. Each head has dim E_total // nheads
        vocab_size (int): size of vocab (total number of tokens)
        pad_id (int): token id of pad token
        eot_id (int): token id of the end of text token
        dropout (float, optional): Dropout probability. Default: 0.0
        is_causal (bool, optional): Whether to apply causal mask. Default: True
        canon (bool, optional): Whether to include canon (1D depthwise convolution) layers. Default: False
        canon_length (int, optinal): One side of the canon lenght not including the current token. Default: 3
        - If canon=True, canon_length=3,is_causal=True -> kernal_size=3+1=4
        - If canon=True, canon_length=3,is_causal=False -> kernal_size=2*3+1=7
        nlayers (int): number of transformer layers
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """
    def __init__(self, config):
        factory_kwargs = {"device": config.device, "dtype": config.dtype}
        super().__init__()
        self.factory_kwargs = factory_kwargs
        self.pad_id = config.pad_id
        self.eot_id = config.eot_id
        self.is_causal = config.is_causal
        self.E = config.E
        self.nheads = config.nheads
        self.nlayers = config.nlayers
        self.shared_encoder_decoder = config.shared_encoder_decoder

        padding_idx = self.pad_id
        if not self.is_causal:
            padding_idx = None
        self.encoder = nn.Embedding(config.vocab_size, config.E, padding_idx=padding_idx, **factory_kwargs)

        self.layers = nn.ModuleList(
            [MultiHeadAttentionLayer(
                E=config.E,
                nheads=config.nheads,
                dropout=config.dropout,
                bias=config.bias,
                is_causal=config.is_causal,
                canon=config.canon,
                canon_length=config.canon_length,
                **factory_kwargs,
            )
            for i in range(config.nlayers)]
        )

        self.layer_norm = nn.LayerNorm(config.E, bias=config.bias, **factory_kwargs)

        if not self.shared_encoder_decoder:
            self.decoder = nn.Linear(config.E, config.vocab_size, bias=config.bias, **factory_kwargs)
        else:
            self.decoder_bias = nn.Parameter(torch.zeros(config.vocab_size, **factory_kwargs))

        if not self.is_causal:
            padding_idx = -100
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=padding_idx, reduction="mean")

    def autoreg_loss(self,
                  x: torch.Tensor,
                  ground_truth: torch.Tensor,
        ):
        """
        Cross Entropy Loss

        Args:
            x (torch.Tensor): input of shape (``B``, ``L``, ``E``) (Contains presoftmax decoder projection)
            ground_truth (torch.Tensor): input of shape (``B``, ``L``) where contains ground truth class indices

        Returns:
            loss (scalar)
        """
        B, L, N = x.shape
        loss = self.cross_entropy(x.reshape((B*L, N)), ground_truth.reshape((B*L)))
        return loss
    
    def encode(self, x: torch.Tensor):
        """
        Encodes Token Ids into latent dimension

        Args:
            x (torch.Tensor): input of shape (``B``, ``L``)
        Returns:
            x (torch.Tensor): output of shape (``B``, ``L``, ``E``)
        """
        # (B, L) -> (B, L, E)
        return self.encoder(x)
    
    def run_layers(self, x: torch.Tensor, valid_tokens: torch.tensor | None = None, cache = None, prefill = False):
        """
        Runs latent through all layers followed by layer norm

        Args:
            x (torch.Tensor): input of shape (``B``, ``L``, ``E``)
            valid_tokens (torch.Tensor): input of shape (``B``, ``L``) (True or False. True means take apart in attention. used for padding, not causal)
            kv_cache (dic[str -> torch.Tensor]): Contains: "K", "V" (Optional: "c_1", "c_2", "c_3")
            prefill (bool): Are we prefilling the cache
        Returns:
            x (torch.Tensor): output of shape (``B``, ``L``, ``E``)
            kv_cache (dic[str -> torch.Tensor]): Contains: "K", "V" (Optional: "c_1", "c_2", "c_3")
        """
        kv_cache = None
        for (i, layer) in enumerate(self.layers):
            if cache is not None:
                kv_cache = cache[i]

            # (B, L, E) -> (B, L, E)
            x, kv_cache_new = layer(x, valid_tokens, kv_cache, prefill)

            if cache is not None:
                cache[i] = kv_cache_new
        
        return x, cache
    
    def decode(self, x: torch.Tensor):
        """
        Decodes latent dimension into token space

        Args:
            x (torch.Tensor): input of shape (``B``, ``L``, ``E``)
        Returns:
            x (torch.Tensor): output of shape (``B``, ``L``, ``N``)
        """
        # (B, L, E) -> (B, L, E)
        x = self.layer_norm(x)

        # (B, L, E) -> (B, L, N)
        if not self.shared_encoder_decoder:
            x = self.decoder(x)
        else:
            x = nn.functional.linear(x, self.encoder.weight, self.decoder_bias)
            
        return x
    
    @abstractmethod
    def generate(self, *args, **kwargs):
        """
        Generates sequence based in prompt
        """
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass for training
        """
        pass

        
        
        