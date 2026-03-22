
from typing import Sequence, Tuple

import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float = 0.2) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = list(hidden_dims)
        self.output_dim = int(self.hidden_dims[-1]) if len(self.hidden_dims) > 0 else int(input_dim)
        if self.output_dim <= 0:
            self.output_dim = 128
        self.is_dummy = self.input_dim <= 0

        if self.is_dummy:
            self.net = None
        else:
            dims = [self.input_dim] + self.hidden_dims
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
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        if self.is_dummy:
            feat = torch.zeros((batch_size, self.output_dim), dtype=x.dtype, device=x.device)
            mask = torch.zeros((batch_size,), dtype=torch.bool, device=x.device)
            return feat, mask

        feat = self.net(x) if self.net is not None else x
        mask = torch.ones((batch_size,), dtype=torch.bool, device=x.device)
        return feat, mask
