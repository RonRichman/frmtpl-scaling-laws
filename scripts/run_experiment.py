#!/usr/bin/env python
"""Run the public freMTPL2 scaling-law experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from frmtpl_scaling.config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    DEFAULT_EPOCHS,
    DEFAULT_REPS,
    DEFAULT_RESULTS_DIR,
    DEFAULT_THRESHOLDS,
    SMOKE_EPOCHS,
    SMOKE_THRESHOLDS,
)
from frmtpl_scaling.train import run_experiment  # noqa: E402


def _parse_thresholds(value: str | None):
    if not value:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _print_scaling_summary(scaling_df) -> None:
    print(scaling_df.to_string(index=False))
    if "alpha" not in scaling_df or scaling_df["alpha"].notna().all():
        return

    missing = ", ".join(scaling_df.loc[scaling_df["alpha"].isna(), "config_name"].astype(str))
    print(
        "Note: scaling-law alpha is undefined for "
        f"{missing} because fewer than two valid training fractions were available. "
        "Run without --smoke, or pass at least two --thresholds values, for fitted scaling exponents."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--models", default="all")
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    thresholds = _parse_thresholds(args.thresholds) or list(DEFAULT_THRESHOLDS)
    reps = args.reps
    epochs = args.epochs
    models = args.models
    if args.smoke:
        thresholds = SMOKE_THRESHOLDS
        reps = 1
        epochs = SMOKE_EPOCHS
        models = "glm,ffn_small"

    run_df, ensemble_df, scaling_df = run_experiment(
        data_path=args.data_path,
        results_dir=args.results_dir,
        models=models,
        thresholds=thresholds,
        reps=reps,
        epochs=epochs,
    )
    print(f"Wrote {len(run_df)} seed-level rows.")
    print(f"Wrote {len(ensemble_df)} ensemble rows.")
    _print_scaling_summary(scaling_df)


if __name__ == "__main__":
    main()
