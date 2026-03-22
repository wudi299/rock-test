
import argparse
from pathlib import Path

from datasets.metadata import load_metadata_tables
from utils.config import load_config
from utils.logger import setup_logger
from utils.split_utils import load_or_create_split


def parse_args():
    parser = argparse.ArgumentParser(description="Create specimen-level train/val/test split")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save split CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = args.output_dir or str(Path(cfg["experiment"]["work_dir"]) / "split_only")
    logger = setup_logger(log_dir=output_dir, name="split")

    dfs = load_metadata_tables(cfg, allow_missing_labels=False)
    split_df, saved_path, summary = load_or_create_split(
        specimens_df=dfs["specimens"],
        cfg=cfg,
        run_dir=output_dir,
        logger=logger,
    )

    logger.info("Split created. Summary: %s", summary["split_counts"])
    logger.info("Saved to: %s", saved_path)
    print(split_df.head().to_string())


if __name__ == "__main__":
    main()
