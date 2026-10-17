"""Baseline Attention-Based Encoder-Decoder (AED) ASR model.

This is Ablation Row A0 in Table 2 of the paper:
  AED = ConformerEncoder + AttentionDecoder + shared W_o (lm_head).

No multi-token prediction, no entity scoring. Used as baseline for comparison.
All other models (BiasingModel, CLASModel) inherit from or extend this class.

Shared W_o invariant:
  self.lm_head is the ONLY Linear(d_model -> V) in the system.
  MTPModule.forward() receives this layer as an argument so all K heads reuse it.
  Never create a separate output projection anywhere else.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from peek_future.models.encoder import ConformerEncoder
from peek_future.models.decoder import AttentionDecoder


class AEDModel(nn.Module):
    """Attention-Based Encoder Decoder Baseline (Row A0)."""

    def __init__(
        self,
        vocab_size: int = 1024,
        feat_dim: int = 80,
        d_model: int = 512,
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
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.d_model = d_model

        self.encoder = ConformerEncoder(
            feat_dim=feat_dim,
            d_model=d_model,
            num_layers=enc_layers,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        self.decoder = AttentionDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=dec_layers,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            dropout=dropout,
        )
        # Shared LM projection layer Wo: d_model -> V
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def encode(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x, lengths)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: (B, T, D) audio features
            x_lengths: (B,)
            y: (B, S) target token sequence including bos at index 0
        Returns:
            logits: (B, S, V)
            loss: scalar cross-entropy loss against y[:, 1:]
        """
        memory, memory_lengths = self.encode(x, x_lengths)
        # decoder input is y[:, :-1]
        dec_in = y[:, :-1]
        h_d = self.decoder(dec_in, memory, memory_lengths)
        logits = self.lm_head(h_d)

        # target is y[:, 1:]
        targets = y[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=self.pad_id,
        )
        return logits, loss

    @torch.no_grad()
    def greedy_decode(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        max_len: int = 200,
    ) -> list[list[int]]:
        """Standard greedy autoregressive decoding."""
        self.eval()
        memory, memory_lengths = self.encode(x, x_lengths)
        batch_size = x.size(0)
        device = x.device

        # Initialize with BOS
        generated = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            h_d = self.decoder(generated, memory, memory_lengths)
            next_logits = self.lm_head(h_d[:, -1, :])  # (B, V)
            next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)  # (B, 1)

            # Mask finished sequences
            next_tokens = torch.where(finished.unsqueeze(1), torch.tensor(self.pad_id, device=device), next_tokens)
            generated = torch.cat([generated, next_tokens], dim=1)

            finished = finished | (next_tokens.squeeze(1) == self.eos_id)
            if finished.all():
                break

        results = []
        for seq in generated:
            toks = []
            for t in seq[1:].tolist():
                if t == self.eos_id or t == self.pad_id:
                    break
                toks.append(t)
            results.append(toks)
        return results
