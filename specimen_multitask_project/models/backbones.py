
from typing import Any, Dict

import torch
import torch.nn as nn
import timm


class TimmBackbone(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=in_chans,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.out_dim = int(getattr(self.model, "num_features"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def masked_backbone_forward(
    backbone: TimmBackbone,
    images: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    images: [B, N, C, H, W]
    mask: [B, N] bool
    return: [B, N, D]
    """
    device = images.device
    batch_size = images.shape[0]
    bag_size = images.shape[1]
    feat_dim = backbone.out_dim

    if bag_size == 0:
        return torch.zeros((batch_size, 0, feat_dim), dtype=images.dtype, device=device)

    if mask is None:
        mask = torch.ones((batch_size, bag_size), dtype=torch.bool, device=device)

    if int(mask.sum().item()) == 0:
        return torch.zeros((batch_size, bag_size, feat_dim), dtype=images.dtype, device=device)

    valid_images = images[mask]
    valid_features = backbone(valid_images)
    if valid_features.ndim > 2:
        valid_features = valid_features.flatten(start_dim=1)

    out = torch.zeros(
        (batch_size, bag_size, feat_dim),
        dtype=valid_features.dtype,
        device=device,
    )
    out[mask] = valid_features
    return out
