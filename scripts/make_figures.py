#!/usr/bin/env python
"""Create default figures from `results/ensemble_scores.csv`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from frmtpl_scaling.config import DEFAULT_FIGURES_DIR, DEFAULT_RESULTS_DIR  # noqa: E402
from frmtpl_scaling.plots import make_default_figures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-scores", default=str(DEFAULT_RESULTS_DIR / "ensemble_scores.csv"))
    parser.add_argument("--run-scores", default=str(DEFAULT_RESULTS_DIR / "run_scores.csv"))
    parser.add_argument("--scaling-fits", default=str(DEFAULT_RESULTS_DIR / "scaling_fits.csv"))
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    args = parser.parse_args()
    make_default_figures(args.ensemble_scores, args.figures_dir, args.run_scores, args.scaling_fits)
    print(f"Wrote figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
