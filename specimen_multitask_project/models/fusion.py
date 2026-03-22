from typing import Optional, Tuple

import torch
import torch.nn as nn


def _make_mlp(input_dim: int, hidden_dims, dropout: float) -> nn.Sequential:
    dims = [input_dim] + list(hidden_dims)
    layers = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.extend(
            [
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
    return nn.Sequential(*layers)


def masked_softmax(
    scores: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    dtype-safe masked softmax for fp16 / bf16 / fp32.

    Args:
        scores: [..., N]
        mask:   same shape as scores, True for valid entries
        dim:    softmax dimension
        eps:    numerical stability for all-masked rows

    Returns:
        weights with invalid positions zeroed out and rows safely normalized.
    """
    mask = mask.to(device=scores.device, dtype=torch.bool)

    # Use dtype-safe minimum instead of a hard-coded -1e9.
    fill_value = torch.finfo(scores.dtype).min
    masked_scores = scores.masked_fill(~mask, fill_value)

    weights = torch.softmax(masked_scores, dim=dim)
    weights = weights * mask.to(weights.dtype)

    denom = weights.sum(dim=dim, keepdim=True).clamp_min(eps)
    weights = weights / denom
    return weights


class ConcatFusion(nn.Module):
    def __init__(
        self,
        macro_dim: int,
        micro_dim: int,
        tab_dim: int,
        hidden_dims,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        input_dim = macro_dim + micro_dim + tab_dim + 3
        self.mlp = _make_mlp(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.output_dim = int(hidden_dims[-1]) if hidden_dims else input_dim

    def forward(
        self,
        macro_feat: torch.Tensor,
        micro_feat: torch.Tensor,
        tab_feat: torch.Tensor,
        branch_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        fused = torch.cat([macro_feat, micro_feat, tab_feat, branch_mask.float()], dim=-1)
        if len(self.mlp) > 0:
            fused = self.mlp(fused)
        return fused, None


class GatedFusion(nn.Module):
    def __init__(
        self,
        macro_dim: int,
        micro_dim: int,
        tab_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.macro_proj = nn.Linear(macro_dim, hidden_dim)
        self.micro_proj = nn.Linear(micro_dim, hidden_dim)
        self.tab_proj = nn.Linear(tab_dim, hidden_dim)

        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.post = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim

    def forward(
        self,
        macro_feat: torch.Tensor,
        micro_feat: torch.Tensor,
        tab_feat: torch.Tensor,
        branch_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        branch_mask shape: [B, 3]
        Order: [macro_available, micro_available, tab_available]
        """
        macro_h = self.macro_proj(macro_feat)
        micro_h = self.micro_proj(micro_feat)
        tab_h = self.tab_proj(tab_feat)

        feats = torch.stack([macro_h, micro_h, tab_h], dim=1)  # [B, 3, H]

        mask = branch_mask.to(device=feats.device, dtype=torch.bool)
        scores = self.score(feats).squeeze(-1)  # [B, 3]

        weights = masked_softmax(scores, mask, dim=1)  # [B, 3]

        fused = torch.sum(weights.unsqueeze(-1) * feats, dim=1)  # [B, H]
        fused = self.post(fused)
        return fused, weights
