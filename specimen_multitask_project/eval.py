
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.metadata import build_label_encoders, build_specimen_records, load_metadata_tables
from datasets.specimen_dataset import SpecimenDataset, specimen_collate_fn
from datasets.tabular import TabularPreprocessor
from losses.build import build_aux_criterion, build_main_criterion
from models.build import build_model
from utils.checkpoint import load_checkpoint
from utils.common import load_label_encoders, recursive_to_device
from utils.config import load_config
from utils.evaluation import run_evaluation, save_evaluation_outputs
from utils.logger import setup_logger
from utils.split_utils import load_or_create_split


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best/last checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Optional config path override")
    parser.add_argument("--run-dir", type=str, default=None, help="Optional run dir override")
    parser.add_argument("--split", type=str, default="test", help="Which split to evaluate: val/test/train")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output dir")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else ckpt_path.parents[1]
    config_path = Path(args.config) if args.config else run_dir / "config.yaml"

    cfg = load_config(str(config_path))
    logger = setup_logger(log_dir=str(run_dir), name=f"eval_{args.split}")

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dfs = load_metadata_tables(cfg, allow_missing_labels=False)
    split_df, _, _ = load_or_create_split(dfs["specimens"], cfg, run_dir=str(run_dir), logger=logger)

    encoders = load_label_encoders(run_dir / "artifacts" / "label_encoders.json")
    tabular = TabularPreprocessor.load(run_dir / "artifacts" / "tabular_preprocessor.json", cfg=cfg)
    tabular_df = tabular.transform(dfs=dfs)

    records = build_specimen_records(dfs=dfs, cfg=cfg, split_df=split_df, allow_missing_labels=False)
    eval_records = [r for r in records if r["split"] == args.split]
    if len(eval_records) == 0:
        raise RuntimeError(f"No records found for split={args.split}")

    dataset = SpecimenDataset(
        records=eval_records,
        tabular_df=tabular_df,
        cfg=cfg,
        main_encoder=encoders["main"],
        aux_encoder=encoders["aux"],
        model_name=str(cfg["model"]["name"]).lower(),
        is_train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["eval"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        collate_fn=specimen_collate_fn,
    )

    model = build_model(
        cfg=cfg,
        main_num_classes=len(encoders["main"]),
        aux_num_classes=len(encoders["aux"]),
        tabular_input_dim=tabular.output_dim,
    ).to(device)
    load_checkpoint(str(ckpt_path), model, map_location=str(device), strict=True)

    main_criterion = build_main_criterion(cfg=cfg, class_weights=None).to(device)
    aux_criterion = build_aux_criterion(cfg=cfg).to(device)

    result = run_evaluation(
        model=model,
        loader=loader,
        device=device,
        cfg=cfg,
        main_encoder=encoders["main"],
        aux_encoder=encoders["aux"],
        main_criterion=main_criterion,
        aux_criterion=aux_criterion,
        save_attention=bool(cfg["eval"].get("save_attention_weights", True)),
        desc=f"Eval {args.split}",
    )

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "eval" / f"{args.split}_{ckpt_path.stem}"
    save_evaluation_outputs(
        result=result,
        output_dir=output_dir,
        main_encoder=encoders["main"],
        aux_encoder=encoders["aux"],
        prefix=args.split,
    )
    logger.info("Evaluation finished. Outputs saved to %s", output_dir)
    logger.info("Main metrics: %s", result.get("main_metrics", {}))
    logger.info("Aux metrics: %s", result.get("aux_metrics", {}))


if __name__ == "__main__":
    main()
