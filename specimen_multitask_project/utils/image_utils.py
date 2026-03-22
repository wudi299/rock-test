
import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFile, ImageOps
import torch

from .common import safe_str

ImageFile.LOAD_TRUNCATED_IMAGES = True


def sanitize_feature_name(value: Any) -> str:
    text = safe_str(value, default="missing").strip()
    if text == "":
        text = "missing"
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "missing"


def normalize_path_like(path_str: Any) -> str:
    text = safe_str(path_str, default="").strip()
    text = text.replace("\\", "/")
    text = re.sub(r"/+", "/", text)
    return text


def strip_known_prefix(path_str: str, prefixes: List[str]) -> Optional[str]:
    if not path_str:
        return None
    norm_path = normalize_path_like(path_str)
    norm_lower = norm_path.lower()
    for prefix in prefixes:
        norm_prefix = normalize_path_like(prefix).rstrip("/")
        if not norm_prefix:
            continue
        lower_prefix = norm_prefix.lower()
        if norm_lower == lower_prefix:
            return ""
        if norm_lower.startswith(lower_prefix + "/"):
            return norm_path[len(norm_prefix):].lstrip("/")
    return None


def pil_loader(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, img: Image.Image) -> torch.Tensor:
        out = img
        for transform in self.transforms:
            out = transform(out)
        return out


class Resize:
    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.resize((self.size, self.size), resample=Image.BILINEAR)


class RandomHorizontalFlip:
    def __init__(self, p: float):
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.mirror(img) if random.random() < self.p else img


class RandomVerticalFlip:
    def __init__(self, p: float):
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.flip(img) if random.random() < self.p else img


class RandomResizedCrop:
    def __init__(self, size: int, scale=(0.8, 1.0)):
        self.size = int(size)
        self.scale = scale

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        area = width * height
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(0.75, 1.3333)

        crop_w = int(round((target_area * aspect_ratio) ** 0.5))
        crop_h = int(round((target_area / aspect_ratio) ** 0.5))
        crop_w = min(max(crop_w, 1), width)
        crop_h = min(max(crop_h, 1), height)

        if width == crop_w:
            left = 0
        else:
            left = random.randint(0, width - crop_w)
        if height == crop_h:
            top = 0
        else:
            top = random.randint(0, height - crop_h)

        img = img.crop((left, top, left + crop_w, top + crop_h))
        return img.resize((self.size, self.size), resample=Image.BILINEAR)


class ColorJitter:
    def __init__(self, strength: float):
        self.strength = float(strength)

    def _factor(self):
        return random.uniform(max(0.0, 1.0 - self.strength), 1.0 + self.strength)

    def __call__(self, img: Image.Image) -> Image.Image:
        img = ImageEnhance.Brightness(img).enhance(self._factor())
        img = ImageEnhance.Contrast(img).enhance(self._factor())
        img = ImageEnhance.Color(img).enhance(self._factor())
        return img


class ToTensorNormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return (tensor - self.mean) / self.std


def build_transforms(cfg: Dict[str, Any], is_train: bool = False) -> Compose:
    image_cfg = cfg["image"]
    size = int(image_cfg["size"])
    mean = image_cfg.get("mean", [0.485, 0.456, 0.406])
    std = image_cfg.get("std", [0.229, 0.224, 0.225])
    aug_cfg = image_cfg.get("train_augment", {})

    transforms: List[Callable] = []
    if is_train and bool(aug_cfg.get("random_resized_crop", False)):
        scale = aug_cfg.get("rrc_scale", [0.8, 1.0])
        transforms.append(RandomResizedCrop(size=size, scale=tuple(scale)))
    else:
        transforms.append(Resize(size=size))

    if is_train:
        if float(aug_cfg.get("horizontal_flip", 0.0)) > 0:
            transforms.append(RandomHorizontalFlip(p=float(aug_cfg["horizontal_flip"])))
        if float(aug_cfg.get("vertical_flip", 0.0)) > 0:
            transforms.append(RandomVerticalFlip(p=float(aug_cfg["vertical_flip"])))
        cj = float(aug_cfg.get("color_jitter", 0.0))
        if cj > 0:
            transforms.append(ColorJitter(cj))

    transforms.append(ToTensorNormalize(mean=mean, std=std))
    return Compose(transforms)


def resolve_image_path(
    row: Dict[str, Any],
    cfg: Dict[str, Any],
    raise_if_not_found: bool = True,
) -> Optional[str]:
    data_cfg = cfg["data"]
    cols = data_cfg["columns"]
    path_cfg = data_cfg.get("path_resolution", {})

    root_dir = safe_str(data_cfg.get("root_dir", ""), default="").strip()
    image_path_mode = str(path_cfg.get("image_path_mode", "auto")).lower()
    relative_col = cols.get("relative_path")
    img_path_col = cols.get("img_path")
    file_name_col = cols.get("file_name")
    prefixes = path_cfg.get("path_strip_prefixes", []) or []
    tail_parts = int(path_cfg.get("tail_parts_from_img_path", 2))
    validate_exists = bool(path_cfg.get("validate_image_exists", True))

    candidates: List[str] = []

    def add_candidate(candidate: Any) -> None:
        text = safe_str(candidate, default="").strip()
        if not text:
            return
        candidates.append(str(Path(text)))

    if image_path_mode in {"auto", "relative_path_col"} and relative_col and relative_col in row:
        rel = safe_str(row.get(relative_col), default="").strip()
        if rel:
            if root_dir:
                add_candidate(Path(root_dir) / Path(normalize_path_like(rel)))
            else:
                add_candidate(Path(normalize_path_like(rel)))

    raw_img_path = ""
    if img_path_col and img_path_col in row:
        raw_img_path = safe_str(row.get(img_path_col), default="").strip()

    if raw_img_path:
        stripped = strip_known_prefix(raw_img_path, prefixes)
        if stripped is not None and root_dir:
            add_candidate(Path(root_dir) / Path(normalize_path_like(stripped)))
        if image_path_mode in {"auto", "img_path_raw"}:
            add_candidate(normalize_path_like(raw_img_path))

        norm = normalize_path_like(raw_img_path)
        parts = [p for p in norm.split("/") if p]
        if root_dir and parts:
            add_candidate(Path(root_dir) / Path(parts[-1]))
            if len(parts) >= tail_parts:
                add_candidate(Path(root_dir) / Path(*parts[-tail_parts:]))

    if image_path_mode in {"auto", "basename"} and file_name_col and file_name_col in row:
        file_name = safe_str(row.get(file_name_col), default="").strip()
        if file_name:
            if root_dir:
                add_candidate(Path(root_dir) / file_name)
            else:
                add_candidate(file_name)

    unique_candidates: List[str] = []
    seen = set()
    for candidate in candidates:
        key = normalize_path_like(candidate).lower()
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)

    if not validate_exists:
        return unique_candidates[0] if unique_candidates else None

    for candidate in unique_candidates:
        if Path(candidate).exists():
            return str(Path(candidate))

    if raise_if_not_found:
        specimen_id = row.get(cols["specimen_id"], "unknown_specimen")
        image_id = row.get(cols["image_id"], "unknown_image")
        msg = (
            f"Unable to resolve image path for image_id={image_id}, specimen_id={specimen_id}. "
            f"Tried candidates: {unique_candidates}. "
            f"Please check data.root_dir, data.path_resolution.path_strip_prefixes, "
            f"or provide a relative path column in config."
        )
        raise FileNotFoundError(msg)
    return None
