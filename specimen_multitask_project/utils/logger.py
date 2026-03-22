
import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .common import ensure_dir

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


def setup_logger(log_dir: str, name: str = "train") -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger(f"{name}_{Path(log_dir).as_posix()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class CSVLogger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.fieldnames = None

    def log(self, row: Dict[str, Any]) -> None:
        row = {k: v for k, v in row.items()}
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
            write_header = True
        else:
            write_header = not self.path.exists()

        with open(self.path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def create_summary_writer(log_dir: str, enabled: bool = True) -> Optional["SummaryWriter"]:
    if not enabled or SummaryWriter is None:
        return None
    return SummaryWriter(log_dir=str(Path(log_dir) / "tensorboard"))
