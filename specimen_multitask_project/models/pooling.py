
from typing import Optional, Tuple

import torch
import torch.nn as nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 0:
        return torch.zeros((x.shape[0], x.shape[-1]), dtype=x.dtype, device=x.device)
    weights = mask.float().unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (x * weights).sum(dim=1) / denom


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 0:
        return torch.zeros((x.shape[0], x.shape[-1]), dtype=x.dtype, device=x.device)
    masked_x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    values = masked_x.max(dim=1).values
    values[torch.isinf(values)] = 0.0
    return values


def masked_softmax(scores, mask, dim=-1, eps=1e-8):
    """
    scores: [B, N] or compatible
    mask:   same shape, True for valid positions, False for padded/missing positions
    """
    mask = mask.to(device=scores.device, dtype=torch.bool)

    # dtype-safe fill value for fp16 / bf16 / fp32
    fill_value = torch.finfo(scores.dtype).min

    masked_scores = scores.masked_fill(~mask, fill_value)
    attn = torch.softmax(masked_scores, dim=dim)

    # zero out invalid positions after softmax
    attn = attn * mask.to(attn.dtype)

    # avoid NaN when an entire row is masked
    denom = attn.sum(dim=dim, keepdim=True).clamp_min(eps)
    attn = attn / denom
    return attn


class MaskedMeanPooling(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return masked_mean(x, mask), None


class MaskedMeanMaxPooling(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pooled = torch.cat([masked_mean(x, mask), masked_max(x, mask)], dim=-1)
        return pooled, None


class AttentionMILPooling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] == 0:
            return (
                torch.zeros((x.shape[0], x.shape[-1]), dtype=x.dtype, device=x.device),
                torch.zeros((x.shape[0], 0), dtype=x.dtype, device=x.device),
            )
        scores = self.attn(x).squeeze(-1)
        attn = masked_softmax(scores, mask)
        pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)
        return pooled, attn


class GatedAttentionMILPooling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.v = nn.Linear(input_dim, hidden_dim)
        self.u = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] == 0:
            return (
                torch.zeros((x.shape[0], x.shape[-1]), dtype=x.dtype, device=x.device),
                torch.zeros((x.shape[0], 0), dtype=x.dtype, device=x.device),
            )
        scores = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))).squeeze(-1)
        attn = masked_softmax(scores, mask)
        pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)
        return pooled, attn


class TransformerMILPooling(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = x.shape[0]
        if x.shape[1] == 0:
            return (
                torch.zeros((batch_size, self.output_dim), dtype=x.dtype, device=x.device),
                None,
            )
        x = self.input_proj(x)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        key_padding_mask = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device),
                ~mask,
            ],
            dim=1,
        )
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        pooled = encoded[:, 0]
        return pooled, None
