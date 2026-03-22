
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .common import AverageMeter, ensure_dir, recursive_to_device
from .metrics import (
    compute_classification_metrics,
    save_classification_report,
    save_confusion_matrix,
)


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: Mapping[str, Any],
    main_encoder: Any,
    aux_encoder: Any,
    main_criterion: Optional[torch.nn.Module] = None,
    aux_criterion: Optional[torch.nn.Module] = None,
    save_attention: bool = False,
    desc: str = "Eval",
) -> Dict[str, Any]:
    model.eval()

    loss_meter = AverageMeter()
    lambda_aux = float(cfg["loss"].get("lambda_aux", 0.3))
    top_k = int(cfg.get("infer", {}).get("top_k", 3))

    pred_rows = []
    attention_rows = []
    fusion_rows = []

    all_main_targets = []
    all_main_probs = []
    all_aux_targets = []
    all_aux_probs = []

    for batch in tqdm(loader, desc=desc, leave=False):
        batch = recursive_to_device(batch, device)
        outputs = model(batch)

        main_logits = outputs["main_logits"]
        aux_logits = outputs["aux_logits"]

        main_probs = torch.softmax(main_logits, dim=1).detach().cpu().numpy()
        aux_probs = torch.softmax(aux_logits, dim=1).detach().cpu().numpy()

        main_targets = batch["main_target"].detach().cpu().numpy()
        aux_targets = batch["aux_target"].detach().cpu().numpy()

        valid_main = main_targets >= 0
        valid_aux = aux_targets >= 0

        if main_criterion is not None and valid_main.any():
            main_loss = main_criterion(main_logits, batch["main_target"])
        else:
            main_loss = torch.tensor(0.0, device=device)

        if aux_criterion is not None and valid_aux.any():
            aux_loss = aux_criterion(aux_logits, batch["aux_target"])
        else:
            aux_loss = torch.tensor(0.0, device=device)

        total_loss = main_loss + lambda_aux * aux_loss
        loss_meter.update(float(total_loss.item()), n=int(main_logits.shape[0]))

        specimen_ids = batch["specimen_id"]
        split_names = batch["split"]
        macro_counts = batch["macro_image_count"].detach().cpu().numpy().tolist()
        micro_counts = batch["micro_image_count"].detach().cpu().numpy().tolist()
        has_macro = batch["has_macro"].detach().cpu().numpy().tolist()
        has_micro = batch["has_micro"].detach().cpu().numpy().tolist()

        if valid_main.any():
            all_main_targets.extend(main_targets[valid_main].tolist())
            all_main_probs.append(main_probs[valid_main])

        if valid_aux.any():
            all_aux_targets.extend(aux_targets[valid_aux].tolist())
            all_aux_probs.append(aux_probs[valid_aux])

        main_pred_idx = main_probs.argmax(axis=1)
        aux_pred_idx = aux_probs.argmax(axis=1)

        for i, specimen_id in enumerate(specimen_ids):
            topk = min(top_k, main_probs.shape[1])
            top_idx = np.argsort(-main_probs[i])[:topk]
            row = {
                "specimen_id": specimen_id,
                "split": split_names[i],
                "true_class_13_id": main_encoder.decode(int(main_targets[i]), default="") if main_targets[i] >= 0 else "",
                "pred_class_13_id": main_encoder.decode(int(main_pred_idx[i]), default=""),
                "top1_prob": float(main_probs[i][main_pred_idx[i]]),
                "top3_classes": "|".join(main_encoder.decode(int(j), default="") for j in top_idx[:3]),
                "top3_probs": "|".join(f"{float(main_probs[i][j]):.6f}" for j in top_idx[:3]),
                "true_weathering_grade_specimen": aux_encoder.decode(int(aux_targets[i]), default="") if aux_targets[i] >= 0 else "",
                "pred_weathering_grade_specimen": aux_encoder.decode(int(aux_pred_idx[i]), default=""),
                "macro_image_count": int(macro_counts[i]),
                "micro_image_count": int(micro_counts[i]),
                "has_macro": int(has_macro[i]),
                "has_micro": int(has_micro[i]),
            }
            pred_rows.append(row)

        if save_attention and outputs.get("macro_attention") is not None:
            macro_attn = outputs["macro_attention"].detach().cpu().numpy()
            macro_mask = batch["macro_mask"].detach().cpu().numpy().astype(bool)
            macro_image_ids = batch["macro_image_ids"]
            macro_paths = batch["macro_paths"]
            for i, specimen_id in enumerate(specimen_ids):
                for j in range(macro_mask.shape[1]):
                    if not macro_mask[i, j]:
                        continue
                    attention_rows.append(
                        {
                            "specimen_id": specimen_id,
                            "branch": "macro",
                            "image_id": macro_image_ids[i][j],
                            "image_path": macro_paths[i][j],
                            "attention_weight": float(macro_attn[i, j]),
                        }
                    )

        if save_attention and outputs.get("micro_attention") is not None:
            micro_attn = outputs["micro_attention"].detach().cpu().numpy()
            micro_mask = batch["micro_mask"].detach().cpu().numpy().astype(bool)
            micro_image_ids = batch["micro_image_ids"]
            micro_paths = batch["micro_paths"]
            for i, specimen_id in enumerate(specimen_ids):
                for j in range(micro_mask.shape[1]):
                    if not micro_mask[i, j]:
                        continue
                    attention_rows.append(
                        {
                            "specimen_id": specimen_id,
                            "branch": "micro",
                            "image_id": micro_image_ids[i][j],
                            "image_path": micro_paths[i][j],
                            "attention_weight": float(micro_attn[i, j]),
                        }
                    )

        if save_attention and outputs.get("fusion_weights") is not None:
            fusion_weights = outputs["fusion_weights"].detach().cpu().numpy()
            for i, specimen_id in enumerate(specimen_ids):
                fusion_rows.append(
                    {
                        "specimen_id": specimen_id,
                        "macro_weight": float(fusion_weights[i, 0]),
                        "micro_weight": float(fusion_weights[i, 1]),
                        "tabular_weight": float(fusion_weights[i, 2]),
                    }
                )

    result: Dict[str, Any] = {
        "loss": float(loss_meter.avg),
        "predictions_df": pd.DataFrame(pred_rows),
        "attention_df": pd.DataFrame(attention_rows),
        "fusion_df": pd.DataFrame(fusion_rows),
        "main_metrics": {},
        "aux_metrics": {},
        "main_confusion": None,
        "aux_confusion": None,
        "main_report": {},
        "aux_report": {},
    }

    if all_main_probs:
        main_probs_full = np.concatenate(all_main_probs, axis=0)
        main_metrics, main_cm, main_report = compute_classification_metrics(
            y_true=all_main_targets,
            y_prob=main_probs_full,
            class_labels=main_encoder.classes,
        )
        result["main_metrics"] = main_metrics
        result["main_confusion"] = main_cm
        result["main_report"] = main_report

    if all_aux_probs:
        aux_probs_full = np.concatenate(all_aux_probs, axis=0)
        aux_metrics, aux_cm, aux_report = compute_classification_metrics(
            y_true=all_aux_targets,
            y_prob=aux_probs_full,
            class_labels=aux_encoder.classes,
        )
        result["aux_metrics"] = aux_metrics
        result["aux_confusion"] = aux_cm
        result["aux_report"] = aux_report

    return result


def save_evaluation_outputs(
    result: Mapping[str, Any],
    output_dir: str,
    main_encoder: Any,
    aux_encoder: Any,
    prefix: str = "",
) -> None:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    name_prefix = f"{prefix}_" if prefix else ""

    predictions_df = result["predictions_df"]
    predictions_df.to_csv(output_dir / f"{name_prefix}predictions.csv", index=False, encoding="utf-8-sig")

    attention_df = result.get("attention_df")
    if attention_df is not None and len(attention_df) > 0:
        attention_df.to_csv(output_dir / f"{name_prefix}attention_weights.csv", index=False, encoding="utf-8-sig")

    fusion_df = result.get("fusion_df")
    if fusion_df is not None and len(fusion_df) > 0:
        fusion_df.to_csv(output_dir / f"{name_prefix}fusion_weights.csv", index=False, encoding="utf-8-sig")

    metrics_payload = {
        "loss": float(result.get("loss", 0.0)),
        "main_metrics": result.get("main_metrics", {}),
        "aux_metrics": result.get("aux_metrics", {}),
    }
    with open(output_dir / f"{name_prefix}metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    if result.get("main_confusion") is not None:
        save_confusion_matrix(
            cm=result["main_confusion"],
            labels=main_encoder.classes,
            path_png=str(output_dir / f"{name_prefix}main_confusion_matrix.png"),
            path_csv=str(output_dir / f"{name_prefix}main_confusion_matrix.csv"),
            normalize=False,
            title="Main Task Confusion Matrix",
        )
        save_confusion_matrix(
            cm=result["main_confusion"],
            labels=main_encoder.classes,
            path_png=str(output_dir / f"{name_prefix}main_confusion_matrix_normalized.png"),
            path_csv=None,
            normalize=True,
            title="Main Task Confusion Matrix (Normalized)",
        )
        save_classification_report(
            report=result["main_report"],
            path_csv=str(output_dir / f"{name_prefix}main_classification_report.csv"),
            path_json=str(output_dir / f"{name_prefix}main_classification_report.json"),
        )

    if result.get("aux_confusion") is not None:
        save_confusion_matrix(
            cm=result["aux_confusion"],
            labels=aux_encoder.classes,
            path_png=str(output_dir / f"{name_prefix}aux_confusion_matrix.png"),
            path_csv=str(output_dir / f"{name_prefix}aux_confusion_matrix.csv"),
            normalize=False,
            title="Aux Task Confusion Matrix",
        )
        save_classification_report(
            report=result["aux_report"],
            path_csv=str(output_dir / f"{name_prefix}aux_classification_report.csv"),
            path_json=str(output_dir / f"{name_prefix}aux_classification_report.json"),
        )
