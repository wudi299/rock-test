
import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, MultiStepLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.metadata import (
    build_label_encoders,
    build_specimen_records,
    load_metadata_tables,
)
from datasets.specimen_dataset import SpecimenDataset, specimen_collate_fn
from datasets.tabular import TabularPreprocessor
from losses.build import build_aux_criterion, build_main_criterion
from models.build import build_model
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.common import (
    count_parameters,
    ensure_dir,
    format_metrics_for_log,
    recursive_to_device,
    save_label_encoders,
    save_yaml,
    timestamp_string,
)
from utils.config import load_config
from utils.evaluation import run_evaluation, save_evaluation_outputs
from utils.logger import CSVLogger, create_summary_writer, setup_logger
from utils.seed import set_seed, seed_worker
from utils.split_utils import load_or_create_split


def parse_args():
    parser = argparse.ArgumentParser(description="Train specimen-level dual-branch multitask model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda or cpu")
    parser.add_argument("--run-dir", type=str, default=None, help="Optional fixed run dir")
    return parser.parse_args()


def build_run_dir(cfg: Dict[str, Any], override: str = None) -> Path:
    if override:
        run_dir = Path(override)
    else:
        work_dir = Path(cfg["experiment"]["work_dir"])
        exp_name = cfg["experiment"]["name"]
        if bool(cfg["experiment"].get("use_timestamp", True)):
            exp_name = f"{exp_name}_{timestamp_string()}"
        run_dir = work_dir / exp_name
    ensure_dir(run_dir)
    return run_dir


def create_dataloader(dataset, cfg: Dict[str, Any], is_train: bool):
    batch_size = int(cfg["train"]["batch_size"] if is_train else cfg["eval"]["batch_size"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        collate_fn=specimen_collate_fn,
        worker_init_fn=seed_worker if int(cfg.get("num_workers", 0)) > 0 else None,
        drop_last=False,
    )


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    base_lr = float(cfg["optimizer"]["lr"])
    weight_decay = float(cfg["optimizer"].get("weight_decay", 1e-4))
    backbone_mult = float(cfg["optimizer"].get("backbone_lr_mult", 1.0))

    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "macro_backbone" in name or "micro_backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": base_lr * backbone_mult},
        {"params": other_params, "lr": base_lr},
    ]
    return AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer):
    sched_cfg = cfg.get("scheduler", {})
    name = str(sched_cfg.get("name", "cosine")).lower()
    epochs = int(cfg["train"]["epochs"])

    if name == "multistep":
        milestones = sched_cfg.get("milestones", [max(1, epochs // 2), max(2, int(epochs * 0.8))])
        gamma = float(sched_cfg.get("gamma", 0.1))
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    if name == "cosine":
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 0))
        min_lr = float(sched_cfg.get("min_lr", 1e-6))
        if warmup_epochs > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=float(sched_cfg.get("warmup_start_factor", 0.2)),
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - warmup_epochs),
                eta_min=min_lr,
            )
            return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=min_lr)

    return None


def compute_class_weights(records: List[Dict[str, Any]], main_encoder) -> torch.Tensor:
    label_indices = []
    for record in records:
        idx = main_encoder.encode(record.get("main_label"), unknown_value=-100, raise_on_unknown=False)
        if idx >= 0:
            label_indices.append(idx)
    counts = np.bincount(label_indices, minlength=len(main_encoder)).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    cfg: Dict[str, Any],
    main_criterion: torch.nn.Module,
    aux_criterion: torch.nn.Module,
    epoch: int,
    logger,
) -> Dict[str, float]:
    model.train()

    freeze_epochs = int(cfg["train"].get("freeze_backbone_epochs", 0))
    backbone_trainable = epoch >= freeze_epochs
    if hasattr(model, "set_backbone_trainable"):
        model.set_backbone_trainable(backbone_trainable)

    lambda_aux = float(cfg["loss"].get("lambda_aux", 0.3))
    grad_clip_norm = cfg["train"].get("grad_clip_norm", None)
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"

    total_loss = 0.0
    total_main = 0.0
    total_aux = 0.0
    total_count = 0

    progress = tqdm(loader, desc=f"Train {epoch}", leave=False)
    for batch in progress:
        batch = recursive_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            outputs = model(batch)
            main_loss = main_criterion(outputs["main_logits"], batch["main_target"])
            aux_loss = aux_criterion(outputs["aux_logits"], batch["aux_target"])
            loss = main_loss + lambda_aux * aux_loss

        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))

        scaler.step(optimizer)
        scaler.update()

        batch_size = int(outputs["main_logits"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_main += float(main_loss.item()) * batch_size
        total_aux += float(aux_loss.item()) * batch_size
        total_count += batch_size

        progress.set_postfix(
            loss=f"{total_loss / max(total_count, 1):.4f}",
            main=f"{total_main / max(total_count, 1):.4f}",
            aux=f"{total_aux / max(total_count, 1):.4f}",
        )

    metrics = {
        "train_loss": total_loss / max(total_count, 1),
        "train_main_loss": total_main / max(total_count, 1),
        "train_aux_loss": total_aux / max(total_count, 1),
    }
    logger.info("Epoch %d train | %s", epoch, format_metrics_for_log(metrics))
    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device

    run_dir = build_run_dir(cfg, override=args.run_dir)
    logger = setup_logger(log_dir=str(run_dir), name="train")
    save_yaml(cfg, run_dir / "config.yaml")
    csv_logger = CSVLogger(run_dir / "train_log.csv")
    tb_writer = create_summary_writer(run_dir, enabled=bool(cfg["train"].get("tensorboard", True)))

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", False)))

    dfs = load_metadata_tables(cfg, allow_missing_labels=False)
    split_df, split_path, split_summary = load_or_create_split(
        specimens_df=dfs["specimens"],
        cfg=cfg,
        run_dir=str(run_dir),
        logger=logger,
    )
    logger.info("Split summary: %s", split_summary["split_counts"])

    encoders = build_label_encoders(dfs=dfs, cfg=cfg)
    save_label_encoders(encoders["main"], encoders["aux"], run_dir / "artifacts" / "label_encoders.json")

    specimen_id_col = cfg["data"]["columns"]["specimen_id"]
    train_specimen_ids = split_df[split_df["split"] == "train"][specimen_id_col].astype(str).tolist()

    tabular_preprocessor = TabularPreprocessor(cfg)
    tabular_preprocessor.fit(dfs=dfs, train_specimen_ids=train_specimen_ids)
    tabular_preprocessor.save(run_dir / "artifacts" / "tabular_preprocessor.json")
    tabular_df = tabular_preprocessor.transform(dfs=dfs)

    records = build_specimen_records(dfs=dfs, cfg=cfg, split_df=split_df, allow_missing_labels=False)
    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"]
    test_records = [r for r in records if r["split"] == "test"]

    if len(train_records) == 0:
        raise RuntimeError("Train split is empty. Please check split generation.")

    logger.info("Records | train=%d | val=%d | test=%d", len(train_records), len(val_records), len(test_records))

    model_name = str(cfg["model"]["name"]).lower()
    train_ds = SpecimenDataset(
        records=train_records,
        tabular_df=tabular_df,
        cfg=cfg,
        main_encoder=encoders["main"],
        aux_encoder=encoders["aux"],
        model_name=model_name,
        is_train=True,
    )
    val_ds = SpecimenDataset(
        records=val_records,
        tabular_df=tabular_df,
        cfg=cfg,
        main_encoder=encoders["main"],
        aux_encoder=encoders["aux"],
        model_name=model_name,
        is_train=False,
    ) if len(val_records) > 0 else None
    test_ds = SpecimenDataset(
        records=test_records,
        tabular_df=tabular_df,
        cfg=cfg,
        main_encoder=encoders["main"],
        aux_encoder=encoders["aux"],
        model_name=model_name,
        is_train=False,
    ) if len(test_records) > 0 else None

    train_loader = create_dataloader(train_ds, cfg, is_train=True)
    val_loader = create_dataloader(val_ds, cfg, is_train=False) if val_ds is not None else None
    test_loader = create_dataloader(test_ds, cfg, is_train=False) if test_ds is not None else None

    model = build_model(
        cfg=cfg,
        main_num_classes=len(encoders["main"]),
        aux_num_classes=len(encoders["aux"]),
        tabular_input_dim=tabular_preprocessor.output_dim,
    ).to(device)
    logger.info("Model params: %s", count_parameters(model))

    loss_type = str(cfg["loss"]["main"].get("type", "ce")).lower()
    class_weights = compute_class_weights(train_records, encoders["main"]) if loss_type in {"weighted_ce", "focal"} else None
    main_criterion = build_main_criterion(cfg=cfg, class_weights=class_weights).to(device)
    aux_criterion = build_aux_criterion(cfg=cfg).to(device)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    scaler = GradScaler(enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")

    start_epoch = 0
    best_score = -float("inf")
    if args.resume:
        ckpt = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=str(device),
            strict=True,
        )
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_score = float(ckpt.get("best_metric", -float("inf")))
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    best_ckpt_path = run_dir / "checkpoints" / "best.pth"
    last_ckpt_path = run_dir / "checkpoints" / "last.pth"
    ensure_dir(best_ckpt_path.parent)

    patience = int(cfg["train"].get("early_stopping_patience", 1000))
    no_improve_epochs = 0

    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cfg=cfg,
            main_criterion=main_criterion,
            aux_criterion=aux_criterion,
            epoch=epoch,
            logger=logger,
        )

        if scheduler is not None:
            scheduler.step()

        val_result = None
        val_score = -float("inf")
        if val_loader is not None:
            val_result = run_evaluation(
                model=model,
                loader=val_loader,
                device=device,
                cfg=cfg,
                main_encoder=encoders["main"],
                aux_encoder=encoders["aux"],
                main_criterion=main_criterion,
                aux_criterion=aux_criterion,
                save_attention=bool(cfg["train"].get("save_attention_in_eval", False)),
                desc=f"Val {epoch}",
            )
            val_score = float(val_result.get("main_metrics", {}).get("macro_f1", -float("inf")))
            logger.info(
                "Epoch %d val | loss=%.4f | main_macro_f1=%.4f | aux_macro_f1=%.4f",
                epoch,
                float(val_result.get("loss", 0.0)),
                float(val_result.get("main_metrics", {}).get("macro_f1", 0.0)),
                float(val_result.get("aux_metrics", {}).get("macro_f1", 0.0)),
            )

        log_row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **train_metrics,
        }
        if val_result is not None:
            log_row.update(
                {
                    "val_loss": float(val_result.get("loss", 0.0)),
                    "val_main_macro_f1": float(val_result.get("main_metrics", {}).get("macro_f1", 0.0)),
                    "val_main_accuracy": float(val_result.get("main_metrics", {}).get("accuracy", 0.0)),
                    "val_main_top3": float(val_result.get("main_metrics", {}).get("top3_accuracy", 0.0)),
                    "val_aux_macro_f1": float(val_result.get("aux_metrics", {}).get("macro_f1", 0.0)),
                    "val_aux_accuracy": float(val_result.get("aux_metrics", {}).get("accuracy", 0.0)),
                }
            )
        csv_logger.log(log_row)
        if tb_writer is not None:
            for key, value in log_row.items():
                if isinstance(value, (float, int)):
                    tb_writer.add_scalar(key, value, epoch)

        save_checkpoint(
            {
                "epoch": epoch,
                "best_metric": best_score,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict(),
                "config": cfg,
            },
            last_ckpt_path,
        )

        improved = val_score > best_score if val_loader is not None else True
        if improved:
            best_score = val_score
            no_improve_epochs = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "best_metric": best_score,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "scaler_state": scaler.state_dict(),
                    "config": cfg,
                },
                best_ckpt_path,
            )
            if val_result is not None:
                save_evaluation_outputs(
                    result=val_result,
                    output_dir=run_dir / "best_val",
                    main_encoder=encoders["main"],
                    aux_encoder=encoders["aux"],
                    prefix="val",
                )
            logger.info("New best model saved with val macro F1 = %.4f", best_score)
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    if best_ckpt_path.exists():
        load_checkpoint(
            checkpoint_path=str(best_ckpt_path),
            model=model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            map_location=str(device),
            strict=True,
        )
    else:
        logger.warning("Best checkpoint not found, using current model for final evaluation.")

    if val_loader is not None:
        val_result = run_evaluation(
            model=model,
            loader=val_loader,
            device=device,
            cfg=cfg,
            main_encoder=encoders["main"],
            aux_encoder=encoders["aux"],
            main_criterion=main_criterion,
            aux_criterion=aux_criterion,
            save_attention=bool(cfg["eval"].get("save_attention_weights", True)),
            desc="Final Val",
        )
        save_evaluation_outputs(
            result=val_result,
            output_dir=run_dir / "final_eval" / "val",
            main_encoder=encoders["main"],
            aux_encoder=encoders["aux"],
            prefix="val",
        )

    if test_loader is not None:
        test_result = run_evaluation(
            model=model,
            loader=test_loader,
            device=device,
            cfg=cfg,
            main_encoder=encoders["main"],
            aux_encoder=encoders["aux"],
            main_criterion=main_criterion,
            aux_criterion=aux_criterion,
            save_attention=bool(cfg["eval"].get("save_attention_weights", True)),
            desc="Final Test",
        )
        save_evaluation_outputs(
            result=test_result,
            output_dir=run_dir / "final_eval" / "test",
            main_encoder=encoders["main"],
            aux_encoder=encoders["aux"],
            prefix="test",
        )
        logger.info("Final test metrics | %s", format_metrics_for_log(test_result.get("main_metrics", {})))

    if tb_writer is not None:
        tb_writer.close()

    logger.info("Training finished. Run dir: %s", run_dir)


if __name__ == "__main__":
    main()
