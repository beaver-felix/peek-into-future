import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dSubsampling4(nn.Module):
    """Convolutional 2D subsampling (4x reduction)."""

    def __init__(self, in_channels: int = 1, out_channels: int = 512, feat_dim: int = 80):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # feat_dim (80) -> 40 -> 20
        subsampled_feat_dim = 64 * (feat_dim // 4)
        self.out_proj = nn.Linear(subsampled_feat_dim, out_channels)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D) -> (B, 1, T, D)
        x = x.unsqueeze(1)
        x = self.conv(x)  # (B, 64, T//4, D//4)
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        x = self.out_proj(x)
        out_lengths = torch.clamp((lengths + 1) // 4, min=1)
        return x, out_lengths


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.w_1 = nn.Linear(d_model, d_model * expansion_factor)
        self.act = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.w_2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x


class ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        x = self.ln(x)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)  # (B, D, T)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return residual + x


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.self_attn_ln = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout)
        self.conv_module = ConformerConvModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.ff1(x)
        residual = x
        normed = self.self_attn_ln(x)
        attn_out, _ = self.self_attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = residual + self.dropout_attn(attn_out)
        x = self.conv_module(x)
        x = self.ff2(x)
        x = self.final_ln(x)
        return x


class ConformerEncoder(nn.Module):
    """12-layer Conformer Audio Encoder."""

    def __init__(
        self,
        feat_dim: int = 80,
        d_model: int = 512,
        num_layers: int = 12,
        n_heads: int = 8,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.subsampling = Conv2dSubsampling4(1, d_model, feat_dim)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(d_model, n_heads, ff_expansion, conv_kernel_size, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, feat_dim)
        x, out_lengths = self.subsampling(x, lengths)
        x = self.pos_enc(x)

        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= out_lengths.unsqueeze(1)

        for layer in self.layers:
            x = layer(x, key_padding_mask=mask)

        return x, out_lengths
