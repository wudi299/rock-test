
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.metadata import build_specimen_records, load_metadata_tables
from datasets.specimen_dataset import SpecimenDataset, specimen_collate_fn
from datasets.tabular import TabularPreprocessor
from models.build import build_model
from utils.checkpoint import load_checkpoint
from utils.common import load_label_encoders
from utils.config import load_config
from utils.evaluation import run_evaluation, save_evaluation_outputs
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for new specimens")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Optional config path override")
    parser.add_argument("--run-dir", type=str, default=None, help="Optional run dir override")
    parser.add_argument("--specimens-csv", type=str, default=None, help="Override specimens.csv path")
    parser.add_argument("--images-csv", type=str, default=None, help="Override images.csv path")
    parser.add_argument("--macro-annotations-csv", type=str, default=None, help="Override macro_annotations.csv path")
    parser.add_argument("--micro-annotations-csv", type=str, default=None, help="Override micro_annotations.csv path")
    parser.add_argument("--root-dir", type=str, default=None, help="Override image root dir")
    parser.add_argument("--specimen-ids", type=str, default=None, help="Comma separated specimen ids to infer")
    parser.add_argument("--split", type=str, default=None, help="Optional split filter")
    parser.add_argument("--output-dir", type=str, default=None, help="Output dir")
    return parser.parse_args()


def apply_overrides(cfg, args):
    if args.specimens_csv:
        cfg["data"]["specimens_csv"] = args.specimens_csv
    if args.images_csv:
        cfg["data"]["images_csv"] = args.images_csv
    if args.macro_annotations_csv:
        cfg["data"]["macro_annotations_csv"] = args.macro_annotations_csv
    if args.micro_annotations_csv:
        cfg["data"]["micro_annotations_csv"] = args.micro_annotations_csv
    if args.root_dir:
        cfg["data"]["root_dir"] = args.root_dir
    return cfg


def maybe_get_split_df(dfs, run_dir: Path, cfg):
    cols = cfg["data"]["columns"]
    specimen_id_col = cols["specimen_id"]
    split_col = cols["split"]

    saved_split = run_dir / "metadata" / "specimen_split.csv"
    if saved_split.exists():
        try:
            split_df = pd.read_csv(saved_split)
            if specimen_id_col in split_df.columns and split_col in split_df.columns:
                return split_df
        except Exception:
            pass

    specimens = dfs["specimens"].copy()
    if split_col in specimens.columns:
        valid = specimens[[specimen_id_col, split_col]].copy()
        valid[split_col] = valid[split_col].fillna("").astype(str).str.strip().str.lower()
        valid = valid[valid[split_col].isin(["train", "val", "test"])]
        if len(valid) > 0:
            return valid
    return None


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else ckpt_path.parents[1]
    config_path = Path(args.config) if args.config else run_dir / "config.yaml"

    cfg = load_config(str(config_path))
    cfg = apply_overrides(cfg, args)

    logger = setup_logger(log_dir=str(run_dir), name="infer")
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    dfs = load_metadata_tables(cfg, allow_missing_labels=True)
    split_df = maybe_get_split_df(dfs, run_dir, cfg)

    encoders = load_label_encoders(run_dir / "artifacts" / "label_encoders.json")
    tabular = TabularPreprocessor.load(run_dir / "artifacts" / "tabular_preprocessor.json", cfg=cfg)
    tabular_df = tabular.transform(dfs=dfs)

    records = build_specimen_records(dfs=dfs, cfg=cfg, split_df=split_df, allow_missing_labels=True)

    if args.split:
        records = [r for r in records if r["split"] == args.split]

    if args.specimen_ids:
        wanted = {token.strip() for token in args.specimen_ids.split(",") if token.strip()}
        records = [r for r in records if r["specimen_id"] in wanted]

    if len(records) == 0:
        raise RuntimeError("No records matched the provided inference filter.")

    dataset = SpecimenDataset(
        records=records,
        tabular_df=tabular_df,
        cfg=cfg,
        main_encoder=encoders["main"],
        aux_encoder=encoders["aux"],
        model_name=str(cfg["model"]["name"]).lower(),
        is_train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["infer"]["batch_size"]),
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

    result = run_evaluation(
        model=model,
        loader=loader,
        device=device,
        cfg=cfg,
        main_encoder=encoders["main"],
        aux_encoder=encoders["aux"],
        main_criterion=None,
        aux_criterion=None,
        save_attention=bool(cfg["eval"].get("save_attention_weights", True)),
        desc="Infer",
    )

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "infer"
    save_evaluation_outputs(
        result=result,
        output_dir=output_dir,
        main_encoder=encoders["main"],
        aux_encoder=encoders["aux"],
        prefix="infer",
    )
    logger.info("Inference finished. Saved to %s", output_dir)


if __name__ == "__main__":
    main()
