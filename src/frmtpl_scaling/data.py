"""Data loading helpers for the corrected Wuthrich-Merz freMTPL2 data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from frmtpl_scaling.config import (
    DEFAULT_DATA_PATH,
    EXPOSURE_COL,
    SAMPLE_COL,
    SPLIT_COL,
    TARGET_COL,
    TEST_LABEL,
    TRAIN_LABEL,
)

EXPECTED_WUTHRICH_SPLIT_COUNTS = {
    TRAIN_LABEL: 610_206,
    TEST_LABEL: 67_801,
}


def _should_validate_default_split(data_path: Path) -> bool:
    try:
        return data_path.resolve() == DEFAULT_DATA_PATH.resolve()
    except FileNotFoundError:
        return False


def validate_wuthrich_split(df: pd.DataFrame) -> None:
    """Validate the Wuthrich-Merz Listing 5.2 learning/test split counts."""
    counts = df[SPLIT_COL].value_counts().to_dict()
    observed = {key: int(counts.get(key, 0)) for key in EXPECTED_WUTHRICH_SPLIT_COUNTS}
    if observed != EXPECTED_WUTHRICH_SPLIT_COUNTS:
        raise ValueError(
            "The bundled freMTPL2 split does not match the Wuthrich-Merz Listing 5.2 "
            f"counts. Expected {EXPECTED_WUTHRICH_SPLIT_COUNTS}, observed {observed}."
        )


def load_frmtpl_csv(
    path: str | Path = DEFAULT_DATA_PATH,
    sample_seed: int = 42,
    require_split: bool = True,
    validate_default_split: bool = True,
) -> pd.DataFrame:
    """Load corrected freMTPL2 data and preserve the supplied learning/test split."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find freMTPL2 CSV at {data_path}. "
            "From open_source_frmtpl_scaling, run "
            "`Rscript scripts/prepare_wuthrich_data.R` to regenerate it from "
            "Mario Wuthrich's corrected RDA source."
        )

    df = pd.read_csv(data_path)
    required = {TARGET_COL, EXPOSURE_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required freMTPL2 columns: {sorted(missing)}")

    if SPLIT_COL not in df.columns and require_split:
        raise ValueError(
            "Expected a `set` column with the Wuthrich-Merz learning/test split. "
            "Regenerate the CSV with scripts/prepare_wuthrich_data.R."
        )

    if SPLIT_COL not in df.columns:
        rng = np.random.default_rng(sample_seed)
        is_train = rng.uniform(size=len(df)) < 0.90
        df[SPLIT_COL] = np.where(is_train, TRAIN_LABEL, TEST_LABEL)
    else:
        df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.lower()

    if SAMPLE_COL not in df.columns:
        rng = np.random.default_rng(sample_seed)
        df[SAMPLE_COL] = 1.0
        is_train = df[SPLIT_COL] == TRAIN_LABEL
        df.loc[is_train, SAMPLE_COL] = rng.uniform(size=int(is_train.sum()))

    df = df[df[EXPOSURE_COL] > 0].copy()
    df[TARGET_COL] = df[TARGET_COL].clip(lower=0)
    df[EXPOSURE_COL] = df[EXPOSURE_COL].astype("float32")
    if validate_default_split and _should_validate_default_split(data_path):
        validate_wuthrich_split(df)
    return df


def train_test_split_from_set(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split using the `set` column shipped with the public CSV."""
    train = df[df[SPLIT_COL].astype(str).str.lower() == TRAIN_LABEL].copy()
    test = df[df[SPLIT_COL].astype(str).str.lower() == TEST_LABEL].copy()
    if train.empty or test.empty:
        raise ValueError("Expected non-empty train and test splits in the `set` column.")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def portfolio_summary(df: pd.DataFrame) -> dict[str, float]:
    """Return simple actuarial portfolio totals used in the README/notebook."""
    exposure = float(df[EXPOSURE_COL].sum())
    claims = float(df[TARGET_COL].sum())
    return {
        "rows": float(len(df)),
        "exposure": exposure,
        "claims": claims,
        "frequency": claims / max(exposure, 1e-12),
        "claim_row_share": float((df[TARGET_COL] > 0).mean()),
    }
