"""CLAS Baseline: Contextual Listen Attend and Spell (Pundak et al., 2018).

Paper Table 1, row 'CLAS' — a stronger bias baseline than plain AED:
  - Encodes each entity in the bias list with a 2-layer LSTM BiasEncoder.
  - Computes dot-product attention from decoder hidden state h_d to bias entity vectors.
  - Concatenates attended context with h_d, projects back to d_model, adds residually.
  - Same base encoder/decoder architecture as AEDModel.

Key differences from BiasingModel:
  - No MTP: does NOT predict future tokens. Bias representations are explicit encodings.
  - Requires fixed-length subword token sequences for entities at both train and eval time.
  - Tends to over-bias at large N (see paper results).

BiasEncoder input:
  bias_tokens: (B, N, L) where L is max entity subword length (character-level for CLAS).
  Returns context vector via LSTM last hidden state, not via MTP logits.

Note: bias_tokens for CLAS should have length L (full entity subword sequence),
not K (MTP heads). This is different from BiasingModel's bias_tokens which has shape (B, N, K).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from peek_future.models.aed import AEDModel


class BiasEncoder(nn.Module):
    """LSTM Bias Encoder for CLAS."""

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)

    def forward(self, entity_tokens: torch.Tensor) -> torch.Tensor:
        # entity_tokens: (B, N, L)
        B, N, L = entity_tokens.size()
        flat_tokens = entity_tokens.view(B * N, L)
        emb = self.embedding(flat_tokens)
        _, (h_n, _) = self.lstm(emb)
        # Take last layer hidden state: (B*N, hidden_dim)
        c_n = h_n[-1].view(B, N, -1)
        return c_n


class CLASModel(AEDModel):
    """
    Contextual Listen, Attend and Spell (CLAS, Pundak et al., 2018).
    Uses external bias encoder and cross-attention over bias list representations.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        feat_dim: int = 80,
        d_model: int = 512,
        bias_hidden_dim: int = 256,
        enc_layers: int = 12,
        dec_layers: int = 6,
        n_heads: int = 8,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        pad_id: int = 0,
        bos_id: int = 2,
        eos_id: int = 3,
    ):
        super().__init__(
            vocab_size=vocab_size,
            feat_dim=feat_dim,
            d_model=d_model,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
        )
        self.bias_encoder = BiasEncoder(vocab_size, embed_dim=128, hidden_dim=bias_hidden_dim)
        self.bias_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            kdim=bias_hidden_dim,
            vdim=bias_hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.combine_proj = nn.Linear(2 * d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        bias_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with contextual biasing attention.
        Args:
            bias_tokens: (B, N, L) token sequences of entities in bias list
        """
        memory, memory_lengths = self.encode(x, x_lengths)
        dec_in = y[:, :-1]
        h_d = self.decoder(dec_in, memory, memory_lengths)  # (B, S, d_model)

        if bias_tokens is not None:
            # bias_tokens: (B, N, L) -> (B, N, bias_hidden_dim)
            c_bias = self.bias_encoder(bias_tokens)
            # Attend from decoder state h_d to bias representations c_bias
            context, _ = self.bias_attn(h_d, c_bias, c_bias)
            combined = torch.cat([h_d, context], dim=-1)
            h_d = h_d + self.combine_proj(combined)

        logits = self.lm_head(h_d)
        targets = y[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=self.pad_id,
        )
        return logits, loss
