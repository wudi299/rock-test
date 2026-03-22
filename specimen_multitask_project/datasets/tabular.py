
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.common import ensure_dir, safe_float, safe_str
from utils.image_utils import sanitize_feature_name


class TabularPreprocessor:
    """
    Builds leakage-safe specimen-level tabular features using:
    - specimen metadata (configurable safe fields)
    - image count/statistics
    - annotation aggregate statistics
    - one-hot encoding for selected safe specimen-level categorical fields
    """

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.state: Dict[str, Any] = {
            "fitted": False,
            "feature_names": [],
            "continuous_feature_names": [],
            "specimen_numeric_cols": [],
            "specimen_categorical_cols": [],
            "specimen_categorical_vocabs": {},
            "image_numeric_cols": [],
            "image_categorical_vocabs": {},
            "macro_ann_numeric_cols": [],
            "macro_ann_categorical_vocabs": {},
            "macro_ann_multivalue_cols": {},
            "micro_ann_numeric_cols": [],
            "micro_ann_categorical_vocabs": {},
            "micro_ann_multivalue_cols": {},
            "fill_values": {},
            "means": {},
            "stds": {},
        }

    @property
    def output_dim(self) -> int:
        return len(self.state.get("feature_names", []))

    def _specimen_cfg(self) -> Mapping[str, Any]:
        return self.cfg["data"]["tabular"]

    def _columns(self) -> Mapping[str, str]:
        return self.cfg["data"]["columns"]

    def _normalise_cat_value(self, value: Any) -> str:
        text = safe_str(value, default="__MISSING__").strip()
        return text if text else "__MISSING__"

    def _split_multivalue(self, value: Any, separator: str) -> List[str]:
        text = safe_str(value, default="").strip()
        if text == "":
            return []
        return [token.strip() for token in text.split(separator) if token.strip()]

    def _collect_vocab(
        self,
        df: pd.DataFrame,
        column: str,
        specimen_ids: Sequence[str],
        multivalue_separator: Optional[str] = None,
    ) -> List[str]:
        specimen_col = self._columns()["specimen_id"]
        if df.empty or column not in df.columns:
            return []
        subset = df[df[specimen_col].astype(str).isin({str(x) for x in specimen_ids})]
        vocab = set()
        for value in subset[column].tolist():
            if multivalue_separator:
                vocab.update(self._split_multivalue(value, multivalue_separator))
            else:
                vocab.add(self._normalise_cat_value(value))
        return sorted(vocab)

    def fit(
        self,
        dfs: Mapping[str, pd.DataFrame],
        train_specimen_ids: Sequence[str],
    ) -> "TabularPreprocessor":
        tab_cfg = self._specimen_cfg()
        cols = self._columns()

        specimens = dfs["specimens"]
        images = dfs["images"]
        macro_ann = dfs.get("macro_annotations", pd.DataFrame())
        micro_ann = dfs.get("micro_annotations", pd.DataFrame())

        train_specimen_ids = [str(x) for x in train_specimen_ids]
        train_set = set(train_specimen_ids)

        excluded_cols = set(tab_cfg.get("exclude_specimen_cols", []))

        specimen_numeric_cols = [
            col
            for col in tab_cfg.get("specimen_numeric_cols", [])
            if col in specimens.columns and col not in excluded_cols
        ]
        specimen_categorical_cols = [
            col
            for col in tab_cfg.get("specimen_categorical_cols", [])
            if col in specimens.columns and col not in excluded_cols
        ]
        image_numeric_cols = [
            col for col in tab_cfg.get("image_numeric_stat_cols", [])
            if col in images.columns
        ]
        macro_ann_numeric_cols = [
            col for col in tab_cfg.get("macro_annotation_numeric_cols", [])
            if not macro_ann.empty and col in macro_ann.columns
        ]
        micro_ann_numeric_cols = [
            col for col in tab_cfg.get("micro_annotation_numeric_cols", [])
            if not micro_ann.empty and col in micro_ann.columns
        ]

        self.state["specimen_numeric_cols"] = specimen_numeric_cols
        self.state["specimen_categorical_cols"] = specimen_categorical_cols
        self.state["image_numeric_cols"] = image_numeric_cols
        self.state["macro_ann_numeric_cols"] = macro_ann_numeric_cols
        self.state["micro_ann_numeric_cols"] = micro_ann_numeric_cols

        specimen_cat_vocabs = {}
        train_specimens_df = specimens[specimens[cols["specimen_id"]].astype(str).isin(train_set)]
        for col in specimen_categorical_cols:
            specimen_cat_vocabs[col] = sorted(
                {
                    self._normalise_cat_value(v)
                    for v in train_specimens_df[col].tolist()
                }
            )
        self.state["specimen_categorical_vocabs"] = specimen_cat_vocabs

        image_cat_vocabs = {}
        for col in tab_cfg.get("image_categorical_cols", []):
            if col in images.columns:
                image_cat_vocabs[col] = self._collect_vocab(images, col, train_specimen_ids)
        self.state["image_categorical_vocabs"] = image_cat_vocabs

        macro_multi = dict(tab_cfg.get("macro_annotation_multivalue_cols", {}))
        micro_multi = dict(tab_cfg.get("micro_annotation_multivalue_cols", {}))

        macro_cat_vocabs = {}
        for col in tab_cfg.get("macro_annotation_categorical_cols", []):
            if not macro_ann.empty and col in macro_ann.columns:
                macro_cat_vocabs[col] = self._collect_vocab(
                    macro_ann, col, train_specimen_ids, multivalue_separator=macro_multi.get(col)
                )
        self.state["macro_ann_categorical_vocabs"] = macro_cat_vocabs
        self.state["macro_ann_multivalue_cols"] = macro_multi

        micro_cat_vocabs = {}
        for col in tab_cfg.get("micro_annotation_categorical_cols", []):
            if not micro_ann.empty and col in micro_ann.columns:
                micro_cat_vocabs[col] = self._collect_vocab(
                    micro_ann, col, train_specimen_ids, multivalue_separator=micro_multi.get(col)
                )
        self.state["micro_ann_categorical_vocabs"] = micro_cat_vocabs
        self.state["micro_ann_multivalue_cols"] = micro_multi

        raw_train_df, continuous_names = self._build_raw_feature_df(
            dfs=dfs,
            specimen_ids=train_specimen_ids,
        )

        fill_values = {}
        means = {}
        stds = {}
        impute_strategy = str(tab_cfg.get("impute_strategy", "median")).lower()
        for feature_name in continuous_names:
            series = pd.to_numeric(raw_train_df[feature_name], errors="coerce")
            if impute_strategy == "mean":
                fill = float(series.mean()) if series.notna().any() else 0.0
            else:
                fill = float(series.median()) if series.notna().any() else 0.0
            series = series.fillna(fill)
            mean = float(series.mean()) if len(series) else 0.0
            std = float(series.std(ddof=0)) if len(series) else 1.0
            if abs(std) < 1e-8:
                std = 1.0
            fill_values[feature_name] = fill
            means[feature_name] = mean
            stds[feature_name] = std

        feature_names = [col for col in raw_train_df.columns if col != cols["specimen_id"]]
        self.state["feature_names"] = feature_names
        self.state["continuous_feature_names"] = sorted(set(continuous_names))
        self.state["fill_values"] = fill_values
        self.state["means"] = means
        self.state["stds"] = stds
        self.state["fitted"] = True
        return self

    def _add_stats(
        self,
        record: Dict[str, Any],
        prefix: str,
        series: pd.Series,
        continuous_names: List[str],
    ) -> None:
        numeric = pd.to_numeric(series, errors="coerce")
        stats = {
            f"{prefix}_mean": float(numeric.mean()) if numeric.notna().any() else np.nan,
            f"{prefix}_std": float(numeric.std(ddof=0)) if numeric.notna().any() else np.nan,
            f"{prefix}_min": float(numeric.min()) if numeric.notna().any() else np.nan,
            f"{prefix}_max": float(numeric.max()) if numeric.notna().any() else np.nan,
        }
        for k, v in stats.items():
            record[k] = v
            continuous_names.append(k)

    def _count_tokens(
        self,
        values: Iterable[Any],
        vocab: Sequence[str],
        separator: Optional[str] = None,
    ) -> Dict[str, int]:
        counts = defaultdict(int)
        for value in values:
            if separator:
                tokens = self._split_multivalue(value, separator)
            else:
                tokens = [self._normalise_cat_value(value)]
            for token in tokens:
                if token in vocab:
                    counts[token] += 1
        return counts

    def _add_vocab_counts(
        self,
        record: Dict[str, Any],
        prefix: str,
        values: Iterable[Any],
        vocab: Sequence[str],
        continuous_names: List[str],
        separator: Optional[str] = None,
    ) -> None:
        values = list(values)
        counts = self._count_tokens(values, vocab=vocab, separator=separator)
        total = max(sum(counts.values()), 1)
        for token in vocab:
            safe_token = sanitize_feature_name(token)
            count_name = f"{prefix}_{safe_token}_count"
            ratio_name = f"{prefix}_{safe_token}_ratio"
            value = float(counts.get(token, 0))
            record[count_name] = value
            record[ratio_name] = value / float(total)
            continuous_names.extend([count_name, ratio_name])

    def _build_raw_feature_df(
        self,
        dfs: Mapping[str, pd.DataFrame],
        specimen_ids: Sequence[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        if not self.state["specimen_numeric_cols"] and not self.state["specimen_categorical_vocabs"] and not self.state["image_numeric_cols"] and not self.state["image_categorical_vocabs"] and not self.state["macro_ann_numeric_cols"] and not self.state["macro_ann_categorical_vocabs"] and not self.state["micro_ann_numeric_cols"] and not self.state["micro_ann_categorical_vocabs"]:
            cols = self._columns()
            return pd.DataFrame({cols["specimen_id"]: [str(x) for x in specimen_ids]}), []

        cols = self._columns()
        data_cfg = self.cfg["data"]
        tab_cfg = data_cfg["tabular"]

        specimens = dfs["specimens"].copy()
        images = dfs["images"].copy()
        macro_ann = dfs.get("macro_annotations", pd.DataFrame()).copy()
        micro_ann = dfs.get("micro_annotations", pd.DataFrame()).copy()

        specimen_id_col = cols["specimen_id"]
        scale_col = cols["scale_type"]
        macro_value = str(data_cfg.get("scale_values", {}).get("macro", "macro")).lower()
        micro_value = str(data_cfg.get("scale_values", {}).get("micro", "micro")).lower()

        specimens = specimens[specimens[specimen_id_col].astype(str).isin({str(x) for x in specimen_ids})].copy()
        specimen_lookup = {
            str(row[specimen_id_col]): row.to_dict() for _, row in specimens.iterrows()
        }
        image_groups = {
            sid: frame.copy()
            for sid, frame in images.groupby(specimen_id_col)
        }
        macro_groups = {
            sid: frame.copy()
            for sid, frame in macro_ann.groupby(specimen_id_col)
        } if not macro_ann.empty and specimen_id_col in macro_ann.columns else {}
        micro_groups = {
            sid: frame.copy()
            for sid, frame in micro_ann.groupby(specimen_id_col)
        } if not micro_ann.empty and specimen_id_col in micro_ann.columns else {}

        records: List[Dict[str, Any]] = []
        continuous_names: List[str] = []

        for specimen_id in specimen_ids:
            specimen_id = str(specimen_id)
            record: Dict[str, Any] = {specimen_id_col: specimen_id}
            specimen_row = specimen_lookup.get(specimen_id, {})

            # specimen direct numeric
            for col in self.state["specimen_numeric_cols"]:
                feat_name = f"spec_{col}"
                record[feat_name] = safe_float(specimen_row.get(col), default=np.nan)
                continuous_names.append(feat_name)

            # specimen direct categorical one-hot
            for col, vocab in self.state["specimen_categorical_vocabs"].items():
                value = self._normalise_cat_value(specimen_row.get(col))
                for token in vocab:
                    feat_name = f"spec_{col}_{sanitize_feature_name(token)}"
                    record[feat_name] = 1.0 if value == token else 0.0

            image_df = image_groups.get(specimen_id, pd.DataFrame())
            if not image_df.empty and scale_col in image_df.columns:
                image_df = image_df.copy()
                image_df[scale_col] = image_df[scale_col].astype(str).str.lower()

            scope_to_df = {
                "all": image_df,
                "macro": image_df[image_df[scale_col] == macro_value].copy() if not image_df.empty else pd.DataFrame(),
                "micro": image_df[image_df[scale_col] == micro_value].copy() if not image_df.empty else pd.DataFrame(),
            }

            for scope_name, scope_df in scope_to_df.items():
                count_name = f"img_{scope_name}_count"
                record[count_name] = float(len(scope_df))
                continuous_names.append(count_name)

                for num_col in self.state["image_numeric_cols"]:
                    if num_col in scope_df.columns:
                        self._add_stats(
                            record,
                            prefix=f"img_{scope_name}_{num_col}",
                            series=scope_df[num_col],
                            continuous_names=continuous_names,
                        )

                for cat_col, vocab in self.state["image_categorical_vocabs"].items():
                    if cat_col in scope_df.columns:
                        self._add_vocab_counts(
                            record,
                            prefix=f"img_{scope_name}_{cat_col}",
                            values=scope_df[cat_col].tolist(),
                            vocab=vocab,
                            continuous_names=continuous_names,
                        )

            macro_df = macro_groups.get(specimen_id, pd.DataFrame())
            micro_df = micro_groups.get(specimen_id, pd.DataFrame())

            for num_col in self.state["macro_ann_numeric_cols"]:
                if not macro_df.empty and num_col in macro_df.columns:
                    self._add_stats(
                        record,
                        prefix=f"macro_ann_{num_col}",
                        series=macro_df[num_col],
                        continuous_names=continuous_names,
                    )

            for cat_col, vocab in self.state["macro_ann_categorical_vocabs"].items():
                if not macro_df.empty and cat_col in macro_df.columns:
                    self._add_vocab_counts(
                        record,
                        prefix=f"macro_ann_{cat_col}",
                        values=macro_df[cat_col].tolist(),
                        vocab=vocab,
                        continuous_names=continuous_names,
                        separator=self.state["macro_ann_multivalue_cols"].get(cat_col),
                    )

            for num_col in self.state["micro_ann_numeric_cols"]:
                if not micro_df.empty and num_col in micro_df.columns:
                    self._add_stats(
                        record,
                        prefix=f"micro_ann_{num_col}",
                        series=micro_df[num_col],
                        continuous_names=continuous_names,
                    )

            for cat_col, vocab in self.state["micro_ann_categorical_vocabs"].items():
                if not micro_df.empty and cat_col in micro_df.columns:
                    self._add_vocab_counts(
                        record,
                        prefix=f"micro_ann_{cat_col}",
                        values=micro_df[cat_col].tolist(),
                        vocab=vocab,
                        continuous_names=continuous_names,
                        separator=self.state["micro_ann_multivalue_cols"].get(cat_col),
                    )

            records.append(record)

        feature_df = pd.DataFrame(records)
        if specimen_id_col not in feature_df.columns:
            feature_df[specimen_id_col] = [str(x) for x in specimen_ids]
        return feature_df, continuous_names

    def transform(
        self,
        dfs: Mapping[str, pd.DataFrame],
        specimen_ids: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if not self.state.get("fitted", False):
            raise RuntimeError("TabularPreprocessor is not fitted yet.")

        cols = self._columns()
        specimen_id_col = cols["specimen_id"]
        if specimen_ids is None:
            specimen_ids = dfs["specimens"][specimen_id_col].astype(str).tolist()
        specimen_ids = [str(x) for x in specimen_ids]

        raw_df, _ = self._build_raw_feature_df(dfs=dfs, specimen_ids=specimen_ids)
        if raw_df.empty:
            raw_df = pd.DataFrame({specimen_id_col: specimen_ids})

        feature_names = list(self.state["feature_names"])
        continuous = set(self.state["continuous_feature_names"])

        for feature_name in feature_names:
            if feature_name not in raw_df.columns:
                raw_df[feature_name] = 0.0

        raw_df = raw_df[[specimen_id_col] + feature_names].copy()

        for feature_name in feature_names:
            if feature_name in continuous:
                fill = float(self.state["fill_values"].get(feature_name, 0.0))
                mean = float(self.state["means"].get(feature_name, 0.0))
                std = float(self.state["stds"].get(feature_name, 1.0))
                series = pd.to_numeric(raw_df[feature_name], errors="coerce").fillna(fill)
                if bool(self._specimen_cfg().get("standardize_continuous", True)):
                    series = (series - mean) / max(std, 1e-8)
                raw_df[feature_name] = series.astype(np.float32)
            else:
                raw_df[feature_name] = pd.to_numeric(raw_df[feature_name], errors="coerce").fillna(0.0).astype(np.float32)

        raw_df = raw_df.set_index(specimen_id_col)
        return raw_df

    def save(self, path: str) -> None:
        ensure_dir(Path(path).parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, cfg: Mapping[str, Any]) -> "TabularPreprocessor":
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        obj = cls(cfg=cfg)
        obj.state = state
        return obj
