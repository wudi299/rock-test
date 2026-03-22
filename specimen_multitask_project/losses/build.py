
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .focal import FocalLoss


def build_main_criterion(
    cfg: Dict[str, Any],
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    main_cfg = cfg["loss"]["main"]
    loss_type = str(main_cfg.get("type", "ce")).lower()
    label_smoothing = float(main_cfg.get("label_smoothing", 0.0))

    if class_weights is not None:
        class_weights = class_weights.float()

    if loss_type == "focal":
        return FocalLoss(
            gamma=float(main_cfg.get("focal_gamma", 2.0)),
            weight=class_weights,
            ignore_index=-100,
        )

    return nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )


def build_aux_criterion(cfg: Dict[str, Any]) -> nn.Module:
    return nn.CrossEntropyLoss(ignore_index=-100)
