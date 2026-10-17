import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from peek_future.models.aed import AEDModel
from peek_future.models.mtp_heads import MTPModule
from peek_future.models.entity_scorer import EntityScorer


class BiasingModel(AEDModel):
    """
    Proposed Model: Conformer AED + Multi-Token Prediction + Entity Scoring.
    Implements joint training (L_MTP + L_entity) and unified search space decoding.
    """

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
        k_heads: int = 4,
        mtp_head_type: Literal["linear", "transformer"] = "linear",
        scorer_type: Literal["learned", "heuristic"] = "learned",
        scorer_hidden_dim: int = 64,
        loss_weights: list[float] | None = None,
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
        self.k_heads = k_heads
        if loss_weights is None:
            # Default weights from paper for K=4: [1.0, 0.2, 0.1, 0.05]
            default_weights = [1.0, 0.2, 0.1, 0.05]
            self.alpha = default_weights[:k_heads]
        else:
            self.alpha = loss_weights[:k_heads]

        self.mtp_module = MTPModule(
            k_heads=k_heads,
            d_model=d_model,
            head_type=mtp_head_type,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            dropout=dropout,
        )
        self.entity_scorer = EntityScorer(
            k_heads=k_heads,
            scorer_type=scorer_type,
            hidden_dim=scorer_hidden_dim,
            dropout=dropout,
        )

    def compute_mtp_loss(
        self, mtp_logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted multi-token prediction loss L_MTP.
        Args:
            mtp_logits: (B, S, K, V) predictions
            targets: (B, S_full) full target sequence (y[:, 1:])
        """
        B, S, K, V = mtp_logits.size()
        device = mtp_logits.device
        total_loss = torch.tensor(0.0, device=device)

        for k_idx in range(K):
            k = k_idx + 1
            # For head k at step s (0-indexed), target is token at position s + (k - 1)
            # targets is y[:, 1:], so target for head 1 at step s is targets[:, s]
            # target for head k at step s is targets[:, s + k - 1]
            if S + k - 1 <= targets.size(1):
                y_k = targets[:, (k - 1) : S + (k - 1)]
            else:
                # Pad targets if shifted beyond sequence length
                shift = k - 1
                available = targets[:, shift:]
                pad_len = S - available.size(1)
                pad = torch.full((B, pad_len), self.pad_id, dtype=torch.long, device=device)
                y_k = torch.cat([available, pad], dim=1)

            logits_k = mtp_logits[:, :, k_idx, :]  # (B, S, V)
            head_loss = F.cross_entropy(
                logits_k.reshape(-1, V),
                y_k.reshape(-1),
                ignore_index=self.pad_id,
            )
            total_loss = total_loss + self.alpha[k_idx] * head_loss

        return total_loss

    def compute_entity_loss(
        self, z_scores: torch.Tensor, tau_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute entity classification loss L_entity.
        Args:
            z_scores: (B, S, N) entity logit scores
            tau_labels: (B, S) ground truth entity indices (0 = null entity ∅)
        """
        B, S, N = z_scores.size()
        return F.cross_entropy(z_scores.reshape(-1, N), tau_labels.reshape(-1))

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        bias_tokens: torch.Tensor | None = None,
        tau_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Joint training forward pass.
        Args:
            x: (B, T, D) audio features
            x_lengths: (B,)
            y: (B, S_total) target tokens starting with BOS
            bias_tokens: (B, N, K) bias list entity tokens (optional)
            tau_labels: (B, S) step entity labels (optional)
        """
        memory, memory_lengths = self.encode(x, x_lengths)
        dec_in = y[:, :-1]
        h_d = self.decoder(dec_in, memory, memory_lengths)  # (B, S, d_model)

        # Multi-token logits: (B, S, K, V)
        mtp_logits = self.mtp_module(h_d, self.lm_head)
        targets = y[:, 1:]
        l_mtp = self.compute_mtp_loss(mtp_logits, targets)

        outputs = {
            "mtp_logits": mtp_logits,
            "head1_logits": mtp_logits[:, :, 0, :],
            "loss_mtp": l_mtp,
            "loss_total": l_mtp,
        }

        if bias_tokens is not None and tau_labels is not None:
            z_scores, P_e = self.entity_scorer(mtp_logits, bias_tokens)
            l_entity = self.compute_entity_loss(z_scores, tau_labels)
            outputs["z_scores"] = z_scores
            outputs["P_e"] = P_e
            outputs["loss_entity"] = l_entity
            outputs["loss_total"] = l_mtp + l_entity

        return outputs

    @torch.no_grad()
    def decode_unified(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        bias_tokens: torch.Tensor,
        entity_token_lists: list[list[int]],
        lambda_param: float = 1.0,
        gamma_param: float = 0.3,
        max_len: int = 200,
    ) -> list[int]:
        """
        Unified search space greedy decoding for a single utterance.
        Args:
            x: (1, T, D) audio
            x_lengths: (1,)
            bias_tokens: (1, N, K)
            entity_token_lists: list of full subword token lists for each entity (0 = [pad])
            lambda_param: biasing weight lambda
            gamma_param: confidence threshold gamma
            max_len: maximum decoding steps
        """
        self.eval()
        memory, memory_lengths = self.encode(x, x_lengths)
        device = x.device
        N = bias_tokens.size(1)

        generated = [self.bos_id]

        while len(generated) < max_len:
            curr_tokens = torch.tensor([generated], dtype=torch.long, device=device)
            h_d = self.decoder(curr_tokens, memory, memory_lengths)  # (1, S, d_model)
            h_last = h_d[:, -1:, :]                                  # (1, 1, d_model)

            # Predict future logits at current step: (1, 1, K, V)
            mtp_logits = self.mtp_module(h_last, self.lm_head)
            l1 = mtp_logits[0, 0, 0, :]                             # (V,)
            P1 = F.softmax(l1, dim=-1)                              # (V,)

            # Entity scores at current step:
            z_scores, P_e = self.entity_scorer(mtp_logits, bias_tokens)
            P_e = P_e[0, 0, :]                                      # (N,)

            # Apply confidence threshold gamma
            if N > 1:
                max_entity_prob = torch.max(P_e[1:]).item()
                if max_entity_prob < gamma_param:
                    # Prune entities: default to static vocab
                    pe_null = 1.0
                    pe_entities = torch.zeros(N - 1, device=device)
                else:
                    pe_null = P_e[0].item()
                    pe_entities = P_e[1:]
            else:
                pe_null = 1.0
                pe_entities = torch.zeros(0, device=device)

            # Unified search space Q:
            # Q_static: V entries -> pe_null * P1(i)
            # Q_entity: N-1 entries -> lambda * pe_entities(n)
            q_static = pe_null * P1                                 # (V,)
            q_entity = lambda_param * pe_entities                   # (N-1,)

            max_static_val, max_static_id = torch.max(q_static, dim=0)
            if q_entity.numel() > 0:
                max_entity_val, max_entity_id = torch.max(q_entity, dim=0)
            else:
                max_entity_val = -1.0
                max_entity_id = -1

            if max_entity_val > max_static_val:
                # Entity selected! Emit all tokens of that entity
                selected_entity_idx = max_entity_id.item() + 1  # offset because null is 0
                emitted_tokens = entity_token_lists[selected_entity_idx]
                generated.extend(emitted_tokens)
            else:
                # Static token selected
                next_tok = max_static_id.item()
                if next_tok == self.eos_id:
                    break
                generated.append(next_tok)

        # Return sequence excluding BOS
        return generated[1:]
