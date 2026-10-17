"""Autoregressive Attention Decoder for the Peek-Into-Future AED model.

Implements a 6-layer Pre-LayerNorm Transformer decoder with:
  - Causal self-attention (masked)
  - Cross-attention over encoder output H_e
  - Position-wise FFN with GELU

Paper §4: 6 decoder layers, hidden=512, ff_expansion=4, n_heads=8, pre-layer-norm.

Key design: The final LM projection W_o (Linear d_model -> V) is defined in AEDModel,
not here, so it can be shared with MTP heads. AttentionDecoder returns h_d (hidden states)
before projection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from peek_future.models.encoder import PositionalEncoding


class TransformerDecoderLayerPreLN(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        ff_expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn_ln = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn_ln = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-LN Self-Attention
        normed_tgt = self.self_attn_ln(tgt)
        attn_out, _ = self.self_attn(
            normed_tgt, normed_tgt, normed_tgt, attn_mask=tgt_mask
        )
        tgt = tgt + self.dropout1(attn_out)

        # Pre-LN Cross-Attention
        normed_tgt = self.cross_attn_ln(tgt)
        cross_out, _ = self.cross_attn(
            normed_tgt, memory, memory, key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(cross_out)

        # Pre-LN FFN
        normed_tgt = self.ffn_ln(tgt)
        tgt = tgt + self.ffn(normed_tgt)
        return tgt


class AttentionDecoder(nn.Module):
    """6-layer Autoregressive Transformer Decoder with Pre-LN."""

    def __init__(
        self,
        vocab_size: int = 1024,
        d_model: int = 512,
        num_layers: int = 6,
        n_heads: int = 8,
        ff_expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayerPreLN(d_model, n_heads, ff_expansion, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model)

    def generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)
        return mask

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        memory_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # tgt_tokens: (B, S)
        # memory: (B, T, D)
        seq_len = tgt_tokens.size(1)
        device = tgt_tokens.device

        x = self.embedding(tgt_tokens) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        tgt_mask = self.generate_causal_mask(seq_len, device)

        mem_key_padding_mask = None
        if memory_lengths is not None:
            max_mem_len = memory.size(1)
            mem_key_padding_mask = (
                torch.arange(max_mem_len, device=device).unsqueeze(0)
                >= memory_lengths.unsqueeze(1)
            )

        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=mem_key_padding_mask,
            )

        h_d = self.final_ln(x)
        return h_d
