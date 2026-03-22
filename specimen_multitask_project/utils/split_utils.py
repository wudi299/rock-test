
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .common import ensure_dir


def _safe_non_null_str(value: Any, default: str = "__MISSING__") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value).strip()
    return text if text else default


def _choose_eval_bucket(
    current_counts: Mapping[str, int],
    target_counts: Mapping[str, int],
) -> str:
    val_ratio = current_counts.get("val", 0) / max(target_counts.get("val", 1), 1)
    test_ratio = current_counts.get("test", 0) / max(target_counts.get("test", 1), 1)
    return "val" if val_ratio <= test_ratio else "test"


def _initial_class_allocation(
    n: int,
    ratios: Mapping[str, float],
    current_counts: Mapping[str, int],
    target_counts: Mapping[str, int],
) -> Dict[str, int]:
    alloc = {"train": 0, "val": 0, "test": 0}
    if n <= 0:
        return alloc
    if n == 1:
        alloc["train"] = 1
        return alloc
    if n == 2:
        alloc["train"] = 1
        alloc[_choose_eval_bucket(current_counts, target_counts)] = 1
        return alloc
    if n == 3:
        alloc["train"] = 2
        alloc[_choose_eval_bucket(current_counts, target_counts)] = 1
        return alloc

    val_n = max(1, int(round(n * float(ratios["val"]))))
    test_n = max(1, int(round(n * float(ratios["test"]))))
    train_n = n - val_n - test_n

    if train_n < 1:
        deficit = 1 - train_n
        while deficit > 0:
            if val_n >= test_n and val_n > 1:
                val_n -= 1
            elif test_n > 1:
                test_n -= 1
            else:
                break
            deficit -= 1
        train_n = n - val_n - test_n

    alloc.update({"train": train_n, "val": val_n, "test": test_n})
    return alloc


def generate_specimen_split(
    specimens_df: pd.DataFrame,
    specimen_id_col: str,
    label_col: str,
    aux_label_col: Optional[str],
    ratios: Mapping[str, float],
    seed: int = 42,
) -> pd.DataFrame:
    rng = random.Random(seed)

    split_target = {
        "train": int(round(len(specimens_df) * float(ratios["train"]))),
        "val": int(round(len(specimens_df) * float(ratios["val"]))),
    }
    split_target["test"] = len(specimens_df) - split_target["train"] - split_target["val"]

    grouped: Dict[str, List[str]] = defaultdict(list)
    specimen_to_aux: Dict[str, Any] = {}
    specimen_to_label: Dict[str, str] = {}
    for _, row in specimens_df.iterrows():
        sid = _safe_non_null_str(row[specimen_id_col], default="")
        label = _safe_non_null_str(row[label_col])
        grouped[label].append(sid)
        specimen_to_label[sid] = label
        if aux_label_col and aux_label_col in row:
            specimen_to_aux[sid] = row.get(aux_label_col)

    assignments: Dict[str, str] = {}
    current_counts = Counter({"train": 0, "val": 0, "test": 0})

    labels_sorted = sorted(grouped.keys(), key=lambda k: (len(grouped[k]), k))
    for label in labels_sorted:
        ids = list(grouped[label])
        rng.shuffle(ids)
        alloc = _initial_class_allocation(
            n=len(ids),
            ratios=ratios,
            current_counts=current_counts,
            target_counts=split_target,
        )

        cursor = 0
        for split_name in ["train", "val", "test"]:
            take = alloc[split_name]
            for sid in ids[cursor: cursor + take]:
                assignments[sid] = split_name
                current_counts[split_name] += 1
            cursor += take

        for sid in ids[cursor:]:
            chosen = min(
                ["train", "val", "test"],
                key=lambda s: current_counts[s] / max(split_target[s], 1),
            )
            assignments[sid] = chosen
            current_counts[chosen] += 1

    for desired_split in ["val", "test"]:
        while current_counts[desired_split] < split_target[desired_split]:
            candidates = []
            train_members = [sid for sid, sp in assignments.items() if sp == "train"]
            train_label_counts = Counter(specimen_to_label[sid] for sid in train_members)
            for sid in train_members:
                label = specimen_to_label[sid]
                if train_label_counts[label] > 1:
                    candidates.append(sid)
            if not candidates:
                break
            sid = rng.choice(candidates)
            assignments[sid] = desired_split
            current_counts["train"] -= 1
            current_counts[desired_split] += 1

    split_df = specimens_df[[specimen_id_col]].copy()
    split_df["split"] = split_df[specimen_id_col].map(assignments)
    split_df[label_col] = split_df[specimen_id_col].map(specimen_to_label)
    if aux_label_col is not None:
        split_df[aux_label_col] = split_df[specimen_id_col].map(specimen_to_aux)
    return split_df


def summarize_split(
    split_df: pd.DataFrame,
    label_col: str,
    split_col: str = "split",
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "split_counts": split_df[split_col].value_counts(dropna=False).to_dict(),
        "label_by_split": {},
    }
    crosstab = pd.crosstab(split_df[label_col], split_df[split_col], dropna=False)
    summary["label_by_split"] = {
        str(idx): {str(col): int(crosstab.loc[idx, col]) for col in crosstab.columns}
        for idx in crosstab.index
    }
    return summary


def load_or_create_split(
    specimens_df: pd.DataFrame,
    cfg: Mapping[str, Any],
    run_dir: Optional[str] = None,
    logger: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Optional[Path], Dict[str, Any]]:
    split_cfg = cfg["data"]["split_strategy"]
    cols = cfg["data"]["columns"]

    specimen_id_col = cols["specimen_id"]
    label_col = split_cfg.get("stratify_by", cols["main_label"])
    aux_col = cols.get("aux_label")
    split_col = cols["split"]

    if split_cfg.get("split_csv"):
        split_path = Path(split_cfg["split_csv"])
        split_df = pd.read_csv(split_path)
        summary = summarize_split(split_df, label_col=label_col, split_col=split_col)
        return split_df, split_path, summary

    if split_col in specimens_df.columns:
        existing = specimens_df[[specimen_id_col, split_col, label_col]].copy()
        if aux_col in specimens_df.columns:
            existing[aux_col] = specimens_df[aux_col]
        existing[split_col] = existing[split_col].fillna("").astype(str).str.strip().str.lower()
        valid = existing[existing[split_col].isin(["train", "val", "test"])].copy()
        if len(valid) > 0:
            summary = summarize_split(valid, label_col=label_col, split_col=split_col)
            if logger is not None:
                logger.info("Using split already present in specimens.csv")
            return valid, None, summary

    split_df = generate_specimen_split(
        specimens_df=specimens_df,
        specimen_id_col=specimen_id_col,
        label_col=label_col,
        aux_label_col=aux_col,
        ratios=split_cfg["ratios"],
        seed=int(split_cfg.get("seed", cfg.get("seed", 42))),
    )

    summary = summarize_split(split_df, label_col=label_col, split_col="split")

    saved_path = None
    if run_dir is not None:
        export_dir = ensure_dir(Path(run_dir) / split_cfg.get("export_subdir", "metadata"))
        saved_path = export_dir / split_cfg.get("export_name", "specimen_split.csv")
        split_df.to_csv(saved_path, index=False, encoding="utf-8-sig")
        with open(export_dir / "split_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        if logger is not None:
            logger.info("Generated split CSV saved to %s", saved_path)

    return split_df, saved_path, summary
