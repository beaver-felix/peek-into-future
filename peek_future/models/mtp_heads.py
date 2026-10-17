"""Multi-Token Prediction (MTP) heads.

Paper §3.1 — Multi-Token Prediction:
  Instead of predicting only the next token, the decoder simultaneously predicts K
  future tokens using K independent heads, all operating on the same decoder hidden
  state h^d_s.

  Each head k applies a specialized transformation g_k (FFN or Transformer block)
  to h^d_s to produce r^k_s \u2208 R^d, then applies the SHARED LM projection W_o to
  produce logits l^k_s \u2208 R^V.

  This gives the model "lookahead" capability for entity scoring.

Ablation rows in Table 2:
  A1: FFNMTPHead   (Linear FFN)      ← paper default
  A2: TransformerMTPHead             ← worse than A1, included for reproducibility

MTPModule usage:
  mtp_logits = mtp_module(h_d, model.lm_head)  # (B, S, K, V)
  The lm_head argument is passed from the parent AEDModel to enforce weight sharing.
"""

import torch
import torch.nn as nn
from typing import Literal


class FFNMTPHead(nn.Module):
    """FFN-based MTP Head (Proposed approach, Table 2 Row A1)."""

    def __init__(self, d_model: int = 512, ff_expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, h_d: torch.Tensor) -> torch.Tensor:
        # Residual connection
        return h_d + self.net(h_d)


class TransformerMTPHead(nn.Module):
    """Transformer-based MTP Head (Ablation study, Table 2 Row A2)."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        ff_expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, h_d: torch.Tensor) -> torch.Tensor:
        # Self-attention over decoder sequence
        normed = self.ln1(h_d)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = h_d + self.dropout1(attn_out)
        normed2 = self.ln2(x)
        return x + self.ffn(normed2)


class MTPModule(nn.Module):
    """
    Multi-Token Prediction Module containing K prediction heads.
    Produces logits l_s^k for k in 1..K using shared LM projection layer Wo.
    """

    def __init__(
        self,
        k_heads: int = 4,
        d_model: int = 512,
        head_type: Literal["linear", "transformer"] = "linear",
        n_heads: int = 8,
        ff_expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k_heads = k_heads
        self.head_type = head_type

        heads = []
        for _ in range(k_heads):
            if head_type == "transformer":
                head = TransformerMTPHead(d_model, n_heads, ff_expansion, dropout)
            else:
                head = FFNMTPHead(d_model, ff_expansion, dropout)
            heads.append(head)
        self.heads = nn.ModuleList(heads)

    def forward(
        self, h_d: torch.Tensor, lm_head: nn.Linear
    ) -> torch.Tensor:
        """
        Args:
            h_d: Decoder hidden states of shape (B, S, d_model)
            lm_head: Shared language model projection layer W_o (d_model -> V)
        Returns:
            all_logits: Multi-token logits of shape (B, S, K, V)
        """
        head_outputs = []
        for head in self.heads:
            r_k = head(h_d)          # (B, S, d_model)
            l_k = lm_head(r_k)       # (B, S, V)
            head_outputs.append(l_k)

        # Stack along K dimension -> (B, S, K, V)
        return torch.stack(head_outputs, dim=2)
