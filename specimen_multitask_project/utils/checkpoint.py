
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .common import ensure_dir


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"], strict=strict)

    if optimizer is not None and "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception:
            pass
    if scaler is not None and "scaler_state" in ckpt and ckpt["scaler_state"] is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt
