#!/usr/bin/env python
"""Create business-facing GLM vs TabM-mini outcome diagnostics.

The scaling-law sweep reports likelihood metrics. This script adds actuarial
model-output views for non-technical review: full-data predictions, driver-age
frequency summaries, and an interaction heatmap.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from frmtpl_scaling.config import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_EPOCHS,
    DEFAULT_RESULTS_DIR,
    EXPOSURE_COL,
    SAMPLE_COL,
    TARGET_COL,
    get_default_model_configs,
)
from frmtpl_scaling.data import load_frmtpl_csv, train_test_split_from_set  # noqa: E402
from frmtpl_scaling.plots import FAMILY_COLORS, set_paper_style  # noqa: E402


MODEL_NAMES = ["glm", "tabm_mini_small"]
DISPLAY_NAMES = {
    "glm": "GLM",
    "tabm_mini_small": "TabM-mini",
}
ROW_PREDICTIONS_FILE = "full_data_test_predictions_glm_tabm.csv"


def _age_band(values: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 24, 29, 39, 49, 59, 69, np.inf],
        labels=["18-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+"],
        right=True,
    )


def _bonus_malus_band(values: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 50, 75, 100, np.inf],
        labels=["<=50", "51-75", "76-100", "101+"],
        right=True,
    )


def _veh_age_band(values: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 0, 2, 5, 10, np.inf],
        labels=["0", "1-2", "3-5", "6-10", "11+"],
        right=True,
    )


def _summarize(predictions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(group_cols, observed=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        exposure = float(group[EXPOSURE_COL].sum())
        claims = float(group[TARGET_COL].sum())
        glm_claims = float(group["glm_pred_count"].sum())
        tabm_claims = float(group["tabm_mini_pred_count"].sum())
        row = {col: key for col, key in zip(group_cols, keys, strict=True)}
        row.update(
            {
                "rows": int(len(group)),
                "exposure": exposure,
                "observed_claims": claims,
                "glm_pred_claims": glm_claims,
                "tabm_mini_pred_claims": tabm_claims,
                "observed_frequency": claims / max(exposure, 1e-12),
                "glm_frequency": glm_claims / max(exposure, 1e-12),
                "tabm_mini_frequency": tabm_claims / max(exposure, 1e-12),
                "tabm_minus_glm_frequency": (tabm_claims - glm_claims) / max(exposure, 1e-12),
                "tabm_to_glm_ratio": tabm_claims / max(glm_claims, 1e-12),
                "tabm_minus_glm_claims_per_1000_exposure": 1000.0
                * (tabm_claims - glm_claims)
                / max(exposure, 1e-12),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _train_full_data_predictions(
    data_path: Path,
    reps: int,
    epochs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import keras

    from frmtpl_scaling.losses import poisson_deviance
    from frmtpl_scaling.models import build_model, set_global_seed
    from frmtpl_scaling.preprocessing import base_rate, fit_preprocessor, make_keras_data
    from frmtpl_scaling.train import train_evaluate_model

    raw = load_frmtpl_csv(data_path)
    train_raw, test_raw = train_test_split_from_set(raw)
    preprocessor = fit_preprocessor(train_raw)
    train_encoded_full = preprocessor.transform(train_raw)
    test_encoded = preprocessor.transform(test_raw)

    train_encoded = train_encoded_full[train_encoded_full[SAMPLE_COL] <= 1.0].reset_index(drop=True)
    train_x, train_y = make_keras_data(train_encoded, preprocessor.feature_names)
    test_x, test_y = make_keras_data(test_encoded, preprocessor.feature_names)
    threshold_base_rate = base_rate(train_encoded)
    configs = get_default_model_configs()

    predictions = test_raw.reset_index(drop=True).copy()
    scores = []
    for config_name in MODEL_NAMES:
        config = configs[config_name]
        test_preds = []
        train_preds = []
        for rep_idx, seed in enumerate(range(1, reps + 1), start=1):
            set_global_seed(seed)
            print(
                f"Training diagnostic {config_name} rep={rep_idx}/{reps} "
                f"n={len(train_encoded)}",
                flush=True,
            )
            model = build_model(
                config_name,
                config,
                preprocessor.feature_names,
                preprocessor.cardinalities,
                threshold_base_rate,
            )
            seed_scores, train_pred, test_pred, _ = train_evaluate_model(
                model,
                train_x,
                train_y,
                test_x,
                test_y,
                batch_size=int(config.get("batch_size", DEFAULT_BATCH_SIZE)),
                epochs=epochs,
                validation_seed=1000 + seed,
            )
            train_preds.append(train_pred)
            test_preds.append(test_pred)
            scores.append(
                {
                    "config_name": config_name,
                    "model": DISPLAY_NAMES[config_name],
                    "rep": rep_idx,
                    "seed": seed,
                    "params": int(model.count_params()),
                    **seed_scores,
                }
            )
            print(
                f"Diagnostic seed {config_name} rep={rep_idx}/{reps} "
                f"train_dev={seed_scores['train_poisson_deviance']:.6f} "
                f"test_dev={seed_scores['test_poisson_deviance']:.6f}",
                flush=True,
            )
            keras.backend.clear_session()

        mean_train_pred = np.mean(np.vstack(train_preds), axis=0)
        mean_test_pred = np.mean(np.vstack(test_preds), axis=0)
        predictions[f"{config_name}_pred_count"] = mean_test_pred
        predictions[f"{config_name}_pred_frequency"] = (
            mean_test_pred / predictions[EXPOSURE_COL].to_numpy(dtype="float64")
        )
        scores.append(
            {
                "config_name": config_name,
                "model": DISPLAY_NAMES[config_name],
                "rep": "ensemble",
                "seed": "ensemble",
                "params": int(scores[-1]["params"]),
                "train_poisson_deviance": poisson_deviance(train_y.reshape(-1), mean_train_pred),
                "test_poisson_deviance": poisson_deviance(test_y.reshape(-1), mean_test_pred),
                "epochs_trained": np.nan,
            }
        )
        print(
            f"Diagnostic ensemble {config_name} "
            f"train_dev={scores[-1]['train_poisson_deviance']:.6f} "
            f"test_dev={scores[-1]['test_poisson_deviance']:.6f}",
            flush=True,
        )

    predictions = predictions.rename(
        columns={
            "glm_pred_count": "glm_pred_count",
            "tabm_mini_small_pred_count": "tabm_mini_pred_count",
            "glm_pred_frequency": "glm_pred_frequency",
            "tabm_mini_small_pred_frequency": "tabm_mini_pred_frequency",
        }
    )
    predictions["driv_age_band"] = _age_band(predictions["DrivAge"]).astype(str)
    predictions["bonus_malus_band"] = _bonus_malus_band(predictions["BonusMalus"]).astype(str)
    predictions["veh_age_band"] = _veh_age_band(predictions["VehAge"]).astype(str)
    predictions["tabm_minus_glm_pred_count"] = (
        predictions["tabm_mini_pred_count"] - predictions["glm_pred_count"]
    )
    predictions["tabm_to_glm_pred_frequency_ratio"] = (
        predictions["tabm_mini_pred_frequency"] / predictions["glm_pred_frequency"].clip(lower=1e-12)
    )

    return predictions, pd.DataFrame(scores)


def _portfolio_summary(predictions: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    exposure = float(predictions[EXPOSURE_COL].sum())
    claims = float(predictions[TARGET_COL].sum())
    ensemble_scores = scores[scores["rep"].astype(str) == "ensemble"].set_index("config_name")
    rows = []
    for config_name, pred_col in [
        ("glm", "glm_pred_count"),
        ("tabm_mini_small", "tabm_mini_pred_count"),
    ]:
        pred_claims = float(predictions[pred_col].sum())
        rows.append(
            {
                "config_name": config_name,
                "model": DISPLAY_NAMES[config_name],
                "rows": int(len(predictions)),
                "exposure": exposure,
                "observed_claims": claims,
                "predicted_claims": pred_claims,
                "observed_frequency": claims / max(exposure, 1e-12),
                "predicted_frequency": pred_claims / max(exposure, 1e-12),
                "test_poisson_deviance": float(
                    ensemble_scores.loc[config_name, "test_poisson_deviance"]
                ),
            }
        )
    out = pd.DataFrame(rows)
    glm_dev = float(out.loc[out["config_name"] == "glm", "test_poisson_deviance"].iloc[0])
    out["deviance_lift_vs_glm"] = glm_dev - out["test_poisson_deviance"]
    out["per_policy_loglik_lift_vs_glm"] = out["deviance_lift_vs_glm"] / 2.0
    return out


def _plot_age_summary(age_summary: pd.DataFrame, figures_dir: Path) -> None:
    set_paper_style()
    df = age_summary.copy()
    df["age_order"] = pd.Categorical(
        df["driv_age_band"],
        categories=["18-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+"],
        ordered=True,
    )
    df = df.sort_values("age_order")
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(
        x,
        100.0 * df["observed_frequency"],
        marker="o",
        linewidth=2.1,
        color="#0072B2",
        markeredgecolor="white",
        markeredgewidth=0.6,
        label="Observed",
    )
    ax.plot(
        x,
        100.0 * df["glm_frequency"],
        marker="s",
        linewidth=2.1,
        color=FAMILY_COLORS["ffn_small"],
        markeredgecolor="white",
        markeredgewidth=0.6,
        label="GLM",
    )
    ax.plot(
        x,
        100.0 * df["tabm_mini_frequency"],
        marker="o",
        linewidth=2.5,
        color=FAMILY_COLORS["tabm_mini_small"],
        markeredgecolor="white",
        markeredgewidth=0.6,
        label="TabM-mini",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["driv_age_band"])
    ax.set_xlabel("Driver age band")
    ax.set_ylabel("Claim frequency per 100 exposure-years")
    ax.set_title("Model-implied frequency by driver age band")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.04, y=0.08)
    fig.tight_layout()
    fig.savefig(
        figures_dir / "age_band_frequency_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_interaction_heatmap(interaction_summary: pd.DataFrame, figures_dir: Path) -> None:
    set_paper_style()
    df = interaction_summary.copy()
    age_order = ["18-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    bm_order = ["<=50", "51-75", "76-100", "101+"]
    df = df[df["exposure"] >= 100.0].copy()
    pivot = (
        df.pivot(
            index="driv_age_band",
            columns="bonus_malus_band",
            values="tabm_minus_glm_claims_per_1000_exposure",
        )
        .reindex(index=age_order, columns=bm_order)
        .astype(float)
    )
    values = pivot.to_numpy(dtype=float)
    max_abs = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    max_abs = max(max_abs, 1e-6)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f2f2f2")
    masked_values = np.ma.masked_invalid(values)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(masked_values, cmap=cmap, vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_xticks(np.arange(len(bm_order)))
    ax.set_xticklabels(bm_order)
    ax.set_yticks(np.arange(len(age_order)))
    ax.set_yticklabels(age_order)
    ax.set_xlabel("Bonus-malus band")
    ax.set_ylabel("Driver age band")
    ax.set_title("TabM-mini minus GLM frequency by age and bonus-malus")
    ax.set_xticks(np.arange(-0.5, len(bm_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(age_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                text_color = "white" if abs(val) > 0.55 * max_abs else "black"
                ax.text(
                    j,
                    i,
                    f"{val:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Predicted claim difference per 1,000 exposure-years")
    fig.tight_layout()
    fig.savefig(
        figures_dir / "age_bonusmalus_interaction_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures"))
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Rebuild diagnostic figures from existing grouped CSV summaries without retraining.",
    )
    parser.add_argument(
        "--no-row-predictions",
        action="store_true",
        help="Skip writing the large row-level prediction CSV; grouped summaries are still written.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    diagnostics_dir = results_dir / "outcome_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        age_summary = pd.read_csv(diagnostics_dir / "driver_age_band_summary.csv")
        interaction_summary = pd.read_csv(
            diagnostics_dir / "driver_age_bonusmalus_interaction_summary.csv"
        )
        _plot_age_summary(age_summary, figures_dir)
        _plot_interaction_heatmap(interaction_summary, figures_dir)
        print(f"Wrote figures to {figures_dir}")
        return

    predictions, scores = _train_full_data_predictions(
        Path(args.data_path),
        reps=args.reps,
        epochs=args.epochs,
    )

    prediction_cols = [
        "IDpol",
        EXPOSURE_COL,
        TARGET_COL,
        "DrivAge",
        "driv_age_band",
        "BonusMalus",
        "bonus_malus_band",
        "VehAge",
        "veh_age_band",
        "Area",
        "Region",
        "VehBrand",
        "VehGas",
        "glm_pred_count",
        "glm_pred_frequency",
        "tabm_mini_pred_count",
        "tabm_mini_pred_frequency",
        "tabm_minus_glm_pred_count",
        "tabm_to_glm_pred_frequency_ratio",
    ]
    if not args.no_row_predictions:
        predictions[prediction_cols].to_csv(
            diagnostics_dir / ROW_PREDICTIONS_FILE,
            index=False,
        )
    scores.to_csv(diagnostics_dir / "full_data_glm_tabm_scores.csv", index=False)

    portfolio = _portfolio_summary(predictions, scores)
    portfolio.to_csv(diagnostics_dir / "portfolio_outcome_summary.csv", index=False)

    age_summary = _summarize(predictions, ["driv_age_band"])
    age_summary.to_csv(diagnostics_dir / "driver_age_band_summary.csv", index=False)

    interaction_summary = _summarize(predictions, ["driv_age_band", "bonus_malus_band"])
    interaction_summary.to_csv(
        diagnostics_dir / "driver_age_bonusmalus_interaction_summary.csv",
        index=False,
    )

    veh_age_interaction = _summarize(predictions, ["driv_age_band", "veh_age_band"])
    veh_age_interaction.to_csv(
        diagnostics_dir / "driver_age_vehicle_age_interaction_summary.csv",
        index=False,
    )

    top_interactions = (
        interaction_summary[interaction_summary["exposure"] >= 100.0]
        .assign(
            abs_claim_delta_per_1000_exposure=lambda x: x[
                "tabm_minus_glm_claims_per_1000_exposure"
            ].abs()
        )
        .sort_values("abs_claim_delta_per_1000_exposure", ascending=False)
        .head(12)
    )
    top_interactions.to_csv(diagnostics_dir / "top_age_bonusmalus_differences.csv", index=False)

    _plot_age_summary(age_summary, figures_dir)
    _plot_interaction_heatmap(interaction_summary, figures_dir)

    print(f"Wrote diagnostics to {diagnostics_dir}")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
