
import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch
import yaml


def ensure_dir(path: Any) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json(path: Any) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Mapping[str, Any], path: Any, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def save_yaml(data: Mapping[str, Any], path: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            dict(data),
            f,
            allow_unicode=True,
            sort_keys=False,
        )


def is_nan_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(np.isnan(value))
    except Exception:
        return False


def safe_str(value: Any, default: str = "") -> str:
    if is_nan_like(value):
        return default
    return str(value)


def safe_float(value: Any, default: float = float("nan")) -> float:
    if is_nan_like(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def deep_copy_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
    return deepcopy(dict(data))


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)


class StringLabelEncoder:
    def __init__(
        self,
        classes: Iterable[str],
        display_names: Optional[Mapping[str, str]] = None,
        name: str = "label",
    ) -> None:
        self.classes = [str(x) for x in classes]
        self.class_to_index = {label: idx for idx, label in enumerate(self.classes)}
        self.display_names = {str(k): str(v) for k, v in (display_names or {}).items()}
        self.name = name

    def __len__(self) -> int:
        return len(self.classes)

    def encode(
        self,
        label: Any,
        unknown_value: int = -100,
        raise_on_unknown: bool = False,
    ) -> int:
        if is_nan_like(label):
            return unknown_value
        label = str(label)
        if label not in self.class_to_index:
            if raise_on_unknown:
                raise KeyError(f"Unknown {self.name}: {label}")
            return unknown_value
        return self.class_to_index[label]

    def decode(self, index: int, default: str = "") -> str:
        if index < 0 or index >= len(self.classes):
            return default
        return self.classes[int(index)]

    def decode_display(self, index: int, default: str = "") -> str:
        raw = self.decode(index, default=default)
        return self.display_names.get(raw, raw)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classes": self.classes,
            "display_names": self.display_names,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StringLabelEncoder":
        return cls(
            classes=data.get("classes", []),
            display_names=data.get("display_names", {}),
            name=str(data.get("name", "label")),
        )


def save_label_encoders(
    main_encoder: StringLabelEncoder,
    aux_encoder: StringLabelEncoder,
    path: Any,
) -> None:
    write_json(
        {
            "main": main_encoder.to_dict(),
            "aux": aux_encoder.to_dict(),
        },
        path,
    )


def load_label_encoders(path: Any) -> Dict[str, StringLabelEncoder]:
    payload = read_json(path)
    return {
        "main": StringLabelEncoder.from_dict(payload["main"]),
        "aux": StringLabelEncoder.from_dict(payload["aux"]),
    }


def recursive_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: recursive_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(recursive_to_device(v, device) for v in obj)
    return obj


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def format_metrics_for_log(metrics: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for key, value in metrics.items():
        if isinstance(value, (float, int, np.floating, np.integer)):
            parts.append(f"{key}={float(value):.4f}")
    return " | ".join(parts)
