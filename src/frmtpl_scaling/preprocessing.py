"""Shared freMTPL2 preprocessing and Keras input conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from frmtpl_scaling.config import (
    DROP_COLS,
    EXPOSURE_COL,
    NUMERIC_BIN_COUNT,
    SAMPLE_COL,
    TARGET_COL,
)


@dataclass
class FeaturePreprocessor:
    """Simple categorical + quantile-bin encoder for Keras embedding inputs."""

    feature_names: list[str]
    categorical_maps: dict[str, dict[str, int]]
    numeric_bins: dict[str, np.ndarray]
    cardinalities: dict[str, int]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.feature_names:
            if col in self.categorical_maps:
                values = df[col].fillna("__MISSING__").astype(str)
                mapped = values.map(self.categorical_maps[col]).fillna(0).astype("int32")
                out[col] = mapped
            else:
                values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")
                bins = self.numeric_bins[col]
                encoded = np.zeros(len(values), dtype="int32")
                valid = np.isfinite(values)
                encoded[valid] = np.searchsorted(bins, values[valid], side="right") + 1
                out[col] = encoded

        out[TARGET_COL] = df[TARGET_COL].to_numpy(dtype="float32")
        out[EXPOSURE_COL] = df[EXPOSURE_COL].to_numpy(dtype="float32")
        if SAMPLE_COL in df.columns:
            out[SAMPLE_COL] = df[SAMPLE_COL].to_numpy(dtype="float32")
        return out.reset_index(drop=True)


def fit_preprocessor(train_df: pd.DataFrame, n_bins: int = NUMERIC_BIN_COUNT) -> FeaturePreprocessor:
    """Fit category maps and quantile bins on the training split only."""
    feature_names = [col for col in train_df.columns if col not in DROP_COLS]
    categorical_maps: dict[str, dict[str, int]] = {}
    numeric_bins: dict[str, np.ndarray] = {}
    cardinalities: dict[str, int] = {}

    for col in feature_names:
        series = train_df[col]
        is_categorical = (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)
        )
        if is_categorical:
            levels = sorted(series.fillna("__MISSING__").astype(str).unique().tolist())
            mapping = {level: i + 1 for i, level in enumerate(levels)}
            categorical_maps[col] = mapping
            cardinalities[col] = len(mapping) + 1
        else:
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
            values = values[np.isfinite(values)]
            if len(values) == 0:
                cuts = np.array([], dtype="float64")
            else:
                probs = np.linspace(0, 1, n_bins + 1)[1:-1]
                cuts = np.unique(np.quantile(values, probs))
            numeric_bins[col] = cuts
            cardinalities[col] = len(cuts) + 2

    return FeaturePreprocessor(
        feature_names=feature_names,
        categorical_maps=categorical_maps,
        numeric_bins=numeric_bins,
        cardinalities=cardinalities,
    )


def make_keras_data(encoded_df: pd.DataFrame, feature_names: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Convert an encoded frame to the dictionary inputs used by the Keras models."""
    x = {
        col: encoded_df[col].to_numpy(dtype="int32").reshape(-1, 1)
        for col in feature_names
    }
    x["Exposure"] = encoded_df[EXPOSURE_COL].to_numpy(dtype="float32").reshape(-1, 1)
    y = encoded_df[TARGET_COL].to_numpy(dtype="float32").reshape(-1, 1)
    return x, y


def base_rate(encoded_df: pd.DataFrame) -> float:
    """Claim frequency per exposure-year for output-bias initialization."""
    claims = float(encoded_df[TARGET_COL].sum())
    exposure = float(encoded_df[EXPOSURE_COL].sum())
    return claims / max(exposure, 1e-12)
