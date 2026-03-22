
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"YAML is empty: {path}")
    return cfg


def save_config(cfg: Mapping[str, Any], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, allow_unicode=True, sort_keys=False)


def deep_update(base: Dict[str, Any], updates: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if updates is None:
        return deepcopy(base)
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result
