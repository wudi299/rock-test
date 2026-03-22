
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .common import ensure_dir


def manual_topk_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int = 3) -> float:
    if len(y_true) == 0:
        return 0.0
    k = min(k, y_prob.shape[1])
    topk = np.argsort(-y_prob, axis=1)[:, :k]
    hits = [int(y_true[i] in topk[i]) for i in range(len(y_true))]
    return float(np.mean(hits))


def compute_classification_metrics(
    y_true: Sequence[int],
    y_prob: np.ndarray,
    class_labels: Sequence[str],
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Any]]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = y_prob.argmax(axis=1) if len(y_prob) > 0 else np.asarray([], dtype=np.int64)

    metrics: Dict[str, Any] = {}
    labels = list(range(len(class_labels)))

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    metrics["macro_f1"] = float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    ) if len(y_true) else 0.0
    metrics["top3_accuracy"] = manual_topk_accuracy(y_true, y_prob, k=3)

    per_class = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    ) if len(y_true) else np.zeros(len(class_labels), dtype=np.float32)

    metrics["per_class_f1"] = {
        str(class_labels[idx]): float(score)
        for idx, score in enumerate(per_class.tolist())
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels) if len(y_true) else np.zeros(
        (len(class_labels), len(class_labels)),
        dtype=np.int64,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[str(x) for x in class_labels],
        output_dict=True,
        zero_division=0,
    ) if len(y_true) else {}
    return metrics, cm, report


def save_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[str],
    path_png: str,
    path_csv: Optional[str] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    path_png = Path(path_png)
    ensure_dir(path_png.parent)

    cm_to_show = cm.astype(np.float32)
    if normalize:
        row_sums = cm_to_show.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_to_show = cm_to_show / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_to_show, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm_to_show.shape[0]):
        for j in range(cm_to_show.shape[1]):
            value = cm_to_show[i, j]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if path_csv is not None:
        df = pd.DataFrame(cm, index=labels, columns=labels)
        df.to_csv(path_csv, encoding="utf-8-sig")


def save_classification_report(
    report: Mapping[str, Any],
    path_csv: str,
    path_json: Optional[str] = None,
) -> None:
    ensure_dir(Path(path_csv).parent)
    df = pd.DataFrame(report).T
    df.to_csv(path_csv, encoding="utf-8-sig")
    if path_json is not None:
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
