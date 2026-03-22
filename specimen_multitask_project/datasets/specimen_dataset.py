
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.common import safe_str
from utils.image_utils import build_transforms, pil_loader


class SpecimenDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        tabular_df,
        cfg: Mapping[str, Any],
        main_encoder,
        aux_encoder,
        model_name: str,
        is_train: bool,
    ) -> None:
        self.records = [dict(x) for x in records]
        self.tabular_df = tabular_df
        self.cfg = cfg
        self.main_encoder = main_encoder
        self.aux_encoder = aux_encoder
        self.model_name = model_name
        self.is_train = is_train
        self.transform = build_transforms(cfg, is_train=is_train)
        self.image_size = int(cfg["image"]["size"])

    def __len__(self) -> int:
        return len(self.records)

    def _get_sampling_cfg(self) -> Mapping[str, Any]:
        split_name = "train" if self.is_train else "eval"
        return self.cfg["sampling"][self.model_name][split_name]

    def _uniform_take(self, items: List[Dict[str, Any]], max_k: int) -> List[Dict[str, Any]]:
        if len(items) <= max_k:
            return items
        indices = np.linspace(0, len(items) - 1, num=max_k, dtype=int).tolist()
        return [items[idx] for idx in indices]

    def _sample_branch(self, items: List[Dict[str, Any]], branch: str) -> List[Dict[str, Any]]:
        items = list(items)
        cfg = self._get_sampling_cfg()
        if len(items) == 0:
            return []

        if self.model_name == "baseline":
            if self.is_train:
                num_key = f"num_{branch}_samples"
                k = int(cfg.get(num_key, 8))
                with_replacement = bool(cfg.get("with_replacement", True))
                if len(items) >= k:
                    return random.sample(items, k)
                if with_replacement:
                    return items + random.choices(items, k=k - len(items))
                return items
            mode = str(cfg.get("mode", "all")).lower()
            if mode == "max_k":
                max_k = int(cfg.get(f"max_{branch}_images", len(items)))
                return self._uniform_take(items, max_k=max_k)
            return items

        # advanced
        if self.is_train:
            mode = str(cfg.get("mode", "max_k")).lower()
            if mode == "all":
                return items
            max_k = int(cfg.get(f"max_{branch}_images", len(items)))
            if len(items) <= max_k:
                return items
            return random.sample(items, max_k)
        mode = str(cfg.get("mode", "all")).lower()
        if mode == "max_k":
            max_k = int(cfg.get(f"max_{branch}_images", len(items)))
            return self._uniform_take(items, max_k=max_k)
        return items

    def _load_images(self, items: List[Dict[str, Any]]) -> torch.Tensor:
        images: List[torch.Tensor] = []
        for item in items:
            img = pil_loader(item["resolved_path"])
            images.append(self.transform(img))
        if images:
            return torch.stack(images, dim=0)
        return torch.zeros((0, 3, self.image_size, self.image_size), dtype=torch.float32)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        specimen_id = record["specimen_id"]

        macro_items = self._sample_branch(record["macro_items"], "macro")
        micro_items = self._sample_branch(record["micro_items"], "micro")

        if len(macro_items) == 0 and len(micro_items) == 0:
            raise RuntimeError(
                f"specimen_id={specimen_id} has neither macro nor micro images after sampling."
            )

        macro_images = self._load_images(macro_items)
        micro_images = self._load_images(micro_items)

        if self.tabular_df is not None and specimen_id in self.tabular_df.index:
            tabular_vec = self.tabular_df.loc[specimen_id].values.astype(np.float32)
        else:
            dim = 0 if self.tabular_df is None else int(self.tabular_df.shape[1])
            tabular_vec = np.zeros((dim,), dtype=np.float32)

        main_target = self.main_encoder.encode(record.get("main_label"), unknown_value=-100, raise_on_unknown=False)
        aux_target = self.aux_encoder.encode(record.get("aux_label"), unknown_value=-100, raise_on_unknown=False)

        return {
            "specimen_id": specimen_id,
            "split": safe_str(record.get("split"), default=""),
            "main_target": torch.tensor(main_target, dtype=torch.long),
            "aux_target": torch.tensor(aux_target, dtype=torch.long),
            "main_label_str": safe_str(record.get("main_label"), default=""),
            "aux_label_str": safe_str(record.get("aux_label"), default=""),
            "macro_images": macro_images,
            "micro_images": micro_images,
            "macro_image_ids": [item["image_id"] for item in macro_items],
            "micro_image_ids": [item["image_id"] for item in micro_items],
            "macro_paths": [item["resolved_path"] for item in macro_items],
            "micro_paths": [item["resolved_path"] for item in micro_items],
            "macro_image_count": torch.tensor(int(record["macro_image_count"]), dtype=torch.long),
            "micro_image_count": torch.tensor(int(record["micro_image_count"]), dtype=torch.long),
            "has_macro": torch.tensor(int(record["has_macro"]), dtype=torch.bool),
            "has_micro": torch.tensor(int(record["has_micro"]), dtype=torch.bool),
            "tabular": torch.tensor(tabular_vec, dtype=torch.float32),
        }


def specimen_collate_fn(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(batch) == 0:
        raise ValueError("Empty batch is not allowed")

    def pad_branch(key: str):
        bags = [sample[key] for sample in batch]
        max_n = max(bag.shape[0] for bag in bags)
        c, h, w = bags[0].shape[1], bags[0].shape[2], bags[0].shape[3]
        padded = torch.zeros((len(batch), max_n, c, h, w), dtype=torch.float32)
        mask = torch.zeros((len(batch), max_n), dtype=torch.bool)
        for i, bag in enumerate(bags):
            if bag.shape[0] > 0:
                padded[i, : bag.shape[0]] = bag
                mask[i, : bag.shape[0]] = True
        return padded, mask

    macro_images, macro_mask = pad_branch("macro_images")
    micro_images, micro_mask = pad_branch("micro_images")

    tabular = torch.stack([sample["tabular"] for sample in batch], dim=0)
    main_targets = torch.stack([sample["main_target"] for sample in batch], dim=0)
    aux_targets = torch.stack([sample["aux_target"] for sample in batch], dim=0)
    macro_counts = torch.stack([sample["macro_image_count"] for sample in batch], dim=0)
    micro_counts = torch.stack([sample["micro_image_count"] for sample in batch], dim=0)
    has_macro = torch.stack([sample["has_macro"] for sample in batch], dim=0)
    has_micro = torch.stack([sample["has_micro"] for sample in batch], dim=0)

    return {
        "specimen_id": [sample["specimen_id"] for sample in batch],
        "split": [sample["split"] for sample in batch],
        "main_label_str": [sample["main_label_str"] for sample in batch],
        "aux_label_str": [sample["aux_label_str"] for sample in batch],
        "main_target": main_targets,
        "aux_target": aux_targets,
        "macro_images": macro_images,
        "micro_images": micro_images,
        "macro_mask": macro_mask,
        "micro_mask": micro_mask,
        "macro_image_ids": [sample["macro_image_ids"] for sample in batch],
        "micro_image_ids": [sample["micro_image_ids"] for sample in batch],
        "macro_paths": [sample["macro_paths"] for sample in batch],
        "micro_paths": [sample["micro_paths"] for sample in batch],
        "macro_image_count": macro_counts,
        "micro_image_count": micro_counts,
        "has_macro": has_macro,
        "has_micro": has_micro,
        "tabular": tabular,
    }
