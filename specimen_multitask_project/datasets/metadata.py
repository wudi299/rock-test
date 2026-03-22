
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from utils.common import StringLabelEncoder, safe_str
from utils.image_utils import resolve_image_path


def _read_csv(path: Optional[str]) -> pd.DataFrame:
    if path is None or str(path).strip() == "":
        return pd.DataFrame()
    path_obj = Path(path)
    if not path_obj.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path_obj)
    except UnicodeDecodeError:
        return pd.read_csv(path_obj, encoding="utf-8-sig")


def _ensure_optional_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if col_name not in df.columns:
        df[col_name] = pd.NA
    return df


def load_metadata_tables(
    cfg: Mapping[str, Any],
    allow_missing_labels: bool = False,
) -> Dict[str, pd.DataFrame]:
    data_cfg = cfg["data"]
    cols = data_cfg["columns"]

    specimens = _read_csv(data_cfg.get("specimens_csv"))
    images = _read_csv(data_cfg.get("images_csv"))
    macro_ann = _read_csv(data_cfg.get("macro_annotations_csv"))
    micro_ann = _read_csv(data_cfg.get("micro_annotations_csv"))
    raw_label_mapping = _read_csv(data_cfg.get("raw_label_mapping_csv"))
    class13_mapping = _read_csv(data_cfg.get("class13_mapping_csv"))

    if specimens.empty:
        raise FileNotFoundError(f"specimens.csv not found: {data_cfg.get('specimens_csv')}")
    if images.empty:
        raise FileNotFoundError(f"images.csv not found: {data_cfg.get('images_csv')}")

    required_specimen_cols = [cols["specimen_id"]]
    if not allow_missing_labels:
        required_specimen_cols.append(cols["main_label"])
        required_specimen_cols.append(cols["aux_label"])

    for col_name in required_specimen_cols:
        if col_name not in specimens.columns:
            raise KeyError(f"Missing required column in specimens.csv: {col_name}")

    # allow split to be absent
    _ensure_optional_column(specimens, cols["split"])

    if allow_missing_labels:
        for col_name in [cols["main_label"], cols["aux_label"], cols.get("class_13_name", "")]:
            if col_name:
                _ensure_optional_column(specimens, col_name)

    required_image_cols = [
        cols["image_id"],
        cols["specimen_id"],
        cols["scale_type"],
    ]
    for col_name in required_image_cols:
        if col_name not in images.columns:
            raise KeyError(f"Missing required column in images.csv: {col_name}")

    has_any_path_source = False
    for candidate in [cols.get("relative_path"), cols.get("img_path"), cols.get("file_name")]:
        if candidate and candidate in images.columns:
            has_any_path_source = True
            break
    if not has_any_path_source:
        raise KeyError(
            "images.csv must contain at least one image path source column: "
            "relative_path, img_path, or file_name. Please update config.data.columns."
        )

    keep_flag_col = cols.get("keep_flag")
    filter_cfg = data_cfg.get("filters", {})
    if (
        bool(filter_cfg.get("keep_only_keep_flag", False))
        and keep_flag_col
        and keep_flag_col in images.columns
    ):
        keep_value = filter_cfg.get("keep_flag_value", 1)
        images = images[images[keep_flag_col] == keep_value].copy()

    if cols["image_id"] in images.columns:
        images = images.drop_duplicates(subset=[cols["image_id"]], keep="first").reset_index(drop=True)

    macro_image_id_col = cols["image_id"]
    micro_image_id_col = cols["image_id"]
    if not macro_ann.empty and macro_image_id_col in macro_ann.columns:
        macro_ann = macro_ann.drop_duplicates(subset=[macro_image_id_col], keep="first").reset_index(drop=True)
    if not micro_ann.empty and micro_image_id_col in micro_ann.columns:
        micro_ann = micro_ann.drop_duplicates(subset=[micro_image_id_col], keep="first").reset_index(drop=True)

    return {
        "specimens": specimens.reset_index(drop=True),
        "images": images.reset_index(drop=True),
        "macro_annotations": macro_ann.reset_index(drop=True),
        "micro_annotations": micro_ann.reset_index(drop=True),
        "raw_label_mapping": raw_label_mapping.reset_index(drop=True),
        "class13_mapping": class13_mapping.reset_index(drop=True),
    }


def merge_split_into_specimens(
    specimens_df: pd.DataFrame,
    split_df: Optional[pd.DataFrame],
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    cols = cfg["data"]["columns"]
    specimen_id_col = cols["specimen_id"]
    split_col = cols["split"]

    if split_df is None or split_df.empty:
        return specimens_df.copy()

    merged = specimens_df.drop(columns=[split_col], errors="ignore").merge(
        split_df[[specimen_id_col, split_col]],
        on=specimen_id_col,
        how="left",
    )
    return merged


def build_label_encoders(
    dfs: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
) -> Dict[str, StringLabelEncoder]:
    cols = cfg["data"]["columns"]
    main_label_col = cols["main_label"]
    aux_label_col = cols["aux_label"]
    main_name_col = cols.get("class_13_name")

    class13_mapping = dfs.get("class13_mapping", pd.DataFrame())
    specimens = dfs["specimens"]

    if not class13_mapping.empty and main_label_col in class13_mapping.columns:
        main_classes = class13_mapping[main_label_col].dropna().astype(str).tolist()
        display_names = {}
        if main_name_col and main_name_col in class13_mapping.columns:
            for _, row in class13_mapping.iterrows():
                label_id = safe_str(row.get(main_label_col), default="")
                if label_id:
                    display_names[label_id] = safe_str(row.get(main_name_col), default=label_id)
    else:
        main_classes = sorted(specimens[main_label_col].dropna().astype(str).unique().tolist())
        display_names = {}

    aux_classes = sorted(specimens[aux_label_col].dropna().astype(str).unique().tolist())
    aux_display = {label: label for label in aux_classes}

    return {
        "main": StringLabelEncoder(classes=main_classes, display_names=display_names, name="class_13_id"),
        "aux": StringLabelEncoder(classes=aux_classes, display_names=aux_display, name="weathering_grade_specimen"),
    }


def build_specimen_records(
    dfs: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
    split_df: Optional[pd.DataFrame] = None,
    allow_missing_labels: bool = False,
) -> List[Dict[str, Any]]:
    data_cfg = cfg["data"]
    cols = data_cfg["columns"]
    scale_values = data_cfg.get("scale_values", {})
    macro_value = str(scale_values.get("macro", "macro")).lower()
    micro_value = str(scale_values.get("micro", "micro")).lower()
    error_on_both_missing = bool(data_cfg.get("error_on_both_missing_branches", True))
    skip_missing_images = bool(data_cfg.get("path_resolution", {}).get("skip_missing_images", False))

    specimens = merge_split_into_specimens(dfs["specimens"], split_df, cfg)
    images = dfs["images"]
    macro_ann = dfs.get("macro_annotations", pd.DataFrame())
    micro_ann = dfs.get("micro_annotations", pd.DataFrame())

    specimen_id_col = cols["specimen_id"]
    main_label_col = cols["main_label"]
    aux_label_col = cols["aux_label"]
    split_col = cols["split"]
    image_id_col = cols["image_id"]
    scale_col = cols["scale_type"]

    image_groups = {sid: frame.copy() for sid, frame in images.groupby(specimen_id_col)}
    macro_lookup = (
        macro_ann.set_index(image_id_col).to_dict(orient="index")
        if not macro_ann.empty and image_id_col in macro_ann.columns
        else {}
    )
    micro_lookup = (
        micro_ann.set_index(image_id_col).to_dict(orient="index")
        if not micro_ann.empty and image_id_col in micro_ann.columns
        else {}
    )

    records: List[Dict[str, Any]] = []
    for _, specimen_row in specimens.iterrows():
        specimen_id = safe_str(specimen_row[specimen_id_col], default="")
        image_rows = image_groups.get(specimen_id, pd.DataFrame())

        macro_items: List[Dict[str, Any]] = []
        micro_items: List[Dict[str, Any]] = []

        if not image_rows.empty:
            for _, image_row in image_rows.iterrows():
                row_dict = image_row.to_dict()
                try:
                    resolved_path = resolve_image_path(
                        row=row_dict,
                        cfg=cfg,
                        raise_if_not_found=not skip_missing_images,
                    )
                except FileNotFoundError:
                    if skip_missing_images:
                        resolved_path = None
                    else:
                        raise

                if resolved_path is None and skip_missing_images:
                    continue

                image_id = safe_str(image_row[image_id_col], default="")
                scale_value = safe_str(image_row[scale_col], default="").strip().lower()

                item = {
                    "image_id": image_id,
                    "specimen_id": specimen_id,
                    "resolved_path": resolved_path,
                    "raw_row": row_dict,
                    "annotation": {},
                }

                if scale_value == macro_value:
                    item["annotation"] = macro_lookup.get(image_id, {})
                    macro_items.append(item)
                elif scale_value == micro_value:
                    item["annotation"] = micro_lookup.get(image_id, {})
                    micro_items.append(item)

        has_macro = len(macro_items) > 0
        has_micro = len(micro_items) > 0

        if not has_macro and not has_micro:
            message = (
                f"specimen_id={specimen_id} has neither macro nor micro images "
                "after filtering/path resolution."
            )
            if error_on_both_missing:
                raise ValueError(message)
            continue

        record = {
            "specimen_id": specimen_id,
            "split": safe_str(specimen_row.get(split_col), default=""),
            "main_label": safe_str(specimen_row.get(main_label_col), default="") if main_label_col in specimen_row else "",
            "aux_label": safe_str(specimen_row.get(aux_label_col), default="") if aux_label_col in specimen_row else "",
            "macro_items": macro_items,
            "micro_items": micro_items,
            "macro_image_count": len(macro_items),
            "micro_image_count": len(micro_items),
            "has_macro": has_macro,
            "has_micro": has_micro,
            "specimen_row": specimen_row.to_dict(),
        }
        records.append(record)

    return records


def split_records_by_name(records: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": [], "": []}
    for record in records:
        split_name = safe_str(record.get("split", ""), default="")
        if split_name not in buckets:
            buckets[split_name] = []
        buckets[split_name].append(dict(record))
    return buckets
