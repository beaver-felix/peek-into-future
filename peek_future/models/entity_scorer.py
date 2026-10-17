import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class LearnedEntityScorer(nn.Module):
    """Learned Entity Scoring function f_theta: R^K -> R via FFN (Proposed approach B0)."""

    def __init__(self, k_heads: int = 4, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(k_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, p_vectors: torch.Tensor) -> torch.Tensor:
        # p_vectors: (..., K) -> (..., 1) -> (...)
        return self.ffn(p_vectors).squeeze(-1)


class HeuristicEntityScorer(nn.Module):
    """Heuristic Entity Scoring function f: weighted sum of logits (Table 2 Row B1)."""

    def __init__(self, k_heads: int = 4):
        super().__init__()
        # Initialize weights uniformly or decreasing
        self.weights = nn.Parameter(torch.ones(k_heads) / k_heads)

    def forward(self, p_vectors: torch.Tensor) -> torch.Tensor:
        # p_vectors: (..., K)
        return torch.matmul(p_vectors, self.weights)


class EntityScorer(nn.Module):
    """
    Extracts p_n logit vectors from MTP predictions and computes entity probabilities P_e.
    """

    def __init__(
        self,
        k_heads: int = 4,
        scorer_type: Literal["learned", "heuristic"] = "learned",
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k_heads = k_heads
        self.scorer_type = scorer_type

        if scorer_type == "heuristic":
            self.scorer = HeuristicEntityScorer(k_heads)
        else:
            self.scorer = LearnedEntityScorer(k_heads, hidden_dim, dropout)

    def extract_entity_logits(
        self, mtp_logits: torch.Tensor, bias_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract p_n = [l_s^1[e_n^1], ..., l_s^K[e_n^K]] for all entities in bias list.
        Args:
            mtp_logits: (B, S, K, V) logits across all K heads
            bias_tokens: (B, N, K) token IDs for all N entities in bias list
        Returns:
            p_vectors: (B, S, N, K)
        """
        B, S, K, V = mtp_logits.size()
        N = bias_tokens.size(1)

        # Expand mtp_logits: (B, S, 1, K, V) -> broadcast with N
        # Expand bias_tokens: (B, 1, N, K, 1) -> token index to gather along V
        mtp_exp = mtp_logits.unsqueeze(2).expand(B, S, N, K, V)
        tokens_exp = bias_tokens.unsqueeze(1).unsqueeze(-1).expand(B, S, N, K, 1)

        # Gather: along last dim V -> (B, S, N, K, 1)
        p_gathered = torch.gather(mtp_exp, dim=-1, index=tokens_exp)
        # Squeeze last dim -> (B, S, N, K)
        return p_gathered.squeeze(-1)

    def forward(
        self, mtp_logits: torch.Tensor, bias_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mtp_logits: (B, S, K, V)
            bias_tokens: (B, N, K) where index 0 is null entity ∅
        Returns:
            z_scores: (B, S, N) raw entity scores
            P_e: (B, S, N) entity posterior probabilities (Softmax over N)
        """
        p_vectors = self.extract_entity_logits(mtp_logits, bias_tokens)  # (B, S, N, K)
        z_scores = self.scorer(p_vectors)                                # (B, S, N)
        P_e = F.softmax(z_scores, dim=-1)                                # (B, S, N)
        return z_scores, P_e
