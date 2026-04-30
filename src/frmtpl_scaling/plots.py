"""Matplotlib figures for the public freMTPL2 scaling workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from frmtpl_scaling.analysis import add_glm_lift


DISPLAY_NAMES = {
    "glm": "GLM",
    "ffn_small": "FFN",
    "transformer_multicls_small": "MultiCLS",
    "transformer_multicls_ssl_small": "MultiCLS+SSL",
    "tabm_mini_small": "TabM-mini",
}

FAMILY_COLORS = {
    "glm": "#0072B2",
    "ffn_small": "#E69F00",
    "transformer_multicls_small": "#6A51A3",
    "transformer_multicls_ssl_small": "#CC79A7",
    "tabm_mini_small": "#000000",
}

FAMILY_MARKERS = {
    "glm": "o",
    "ffn_small": "s",
    "transformer_multicls_small": "^",
    "transformer_multicls_ssl_small": "D",
    "tabm_mini_small": "o",
}

MODEL_ORDER = [
    "glm",
    "ffn_small",
    "transformer_multicls_small",
    "transformer_multicls_ssl_small",
    "tabm_mini_small",
]


def set_paper_style() -> None:
    """Apply a consistent publication-oriented Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 9,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "0.85",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def _set_paper_style() -> None:
    set_paper_style()


def _ordered_groups(df: pd.DataFrame):
    seen = set(df["config_name"].astype(str))
    for name in MODEL_ORDER:
        if name in seen:
            yield name, df[df["config_name"] == name].copy()
    for name in sorted(seen - set(MODEL_ORDER)):
        yield name, df[df["config_name"] == name].copy()


def _label(config_name: str) -> str:
    return DISPLAY_NAMES.get(str(config_name), str(config_name).replace("_", " "))


def _color(config_name: str) -> str:
    return FAMILY_COLORS.get(str(config_name), "#333333")


def _marker(config_name: str) -> str:
    return FAMILY_MARKERS.get(str(config_name), "o")


def _savefig(fig, output_path: str | Path) -> None:
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")


def _legend_if_labeled(ax, *args, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(*args, **kwargs)


def plot_data_scaling(ensemble_scores: pd.DataFrame, output_path: str | Path) -> None:
    _set_paper_style()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for name, group in _ordered_groups(ensemble_scores):
        group = group.sort_values("n_train")
        ax.plot(
            group["n_train"],
            group["test_poisson_deviance"],
            marker=_marker(name),
            linewidth=2.2,
            markersize=4.2,
            color=_color(name),
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=_label(name),
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training rows (log scale)")
    ax.set_ylabel("Test Poisson deviance")
    ax.set_title("Public freMTPL2 data scaling by model family")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    _legend_if_labeled(ax, loc="upper right", frameon=True, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.margins(x=0.02, y=0.06)
    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)


def plot_glm_lift(ensemble_scores: pd.DataFrame, output_path: str | Path) -> None:
    _set_paper_style()
    df = add_glm_lift(ensemble_scores)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for name, group in _ordered_groups(df[df["config_name"] != "glm"]):
        group = group.sort_values("threshold")
        ax.plot(
            group["threshold"],
            group["loglik_lift_vs_glm"],
            marker=_marker(name),
            linewidth=2.2,
            markersize=4.2,
            color=_color(name),
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=_label(name),
        )
    ax.axhline(0.0, color="black", linewidth=1.2, alpha=0.75)
    ax.set_xlabel("Training fraction")
    ax.set_ylabel("Per-policy log-likelihood lift vs GLM")
    ax.set_title("Practical lift relative to the GLM baseline")
    _legend_if_labeled(ax, loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.03, y=0.08)
    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)


def _add_params(ensemble_scores: pd.DataFrame, run_scores: pd.DataFrame) -> pd.DataFrame:
    params = (
        run_scores.groupby("config_name", as_index=False)["params"]
        .median()
        .rename(columns={"params": "params_trainable"})
    )
    out = ensemble_scores.merge(params, on="config_name", how="left")
    out["params_trainable"] = pd.to_numeric(out["params_trainable"], errors="coerce")
    return out


def _pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    df = df.dropna(subset=[x_col, y_col]).sort_values([x_col, y_col]).copy()
    best_y = np.inf
    keep = []
    for _, row in df.iterrows():
        y = float(row[y_col])
        if y < best_y - 1e-12:
            keep.append(True)
            best_y = y
        else:
            keep.append(False)
    return df.loc[keep].copy()


def plot_parameter_performance(
    ensemble_scores: pd.DataFrame,
    run_scores: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Main-paper-style parameter/performance view for the public run."""
    _set_paper_style()
    df = _add_params(ensemble_scores, run_scores)
    df = df.dropna(subset=["params_trainable", "test_poisson_deviance"]).copy()
    full_threshold = float(df["threshold"].max())
    full = df[df["threshold"] == full_threshold].copy()
    frontier = _pareto_frontier(full, "params_trainable", "test_poisson_deviance")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    background = df[df["threshold"] != full_threshold]
    ax.scatter(
        background["params_trainable"],
        background["test_poisson_deviance"],
        s=16,
        color="0.35",
        alpha=0.18,
        label="Earlier training fractions",
        zorder=1,
    )
    if len(frontier) >= 2:
        ax.plot(
            frontier["params_trainable"],
            frontier["test_poisson_deviance"],
            color="black",
            linewidth=1.8,
            alpha=0.8,
            label="Full-data frontier",
            zorder=2,
        )

    for name, group in _ordered_groups(full):
        ax.scatter(
            group["params_trainable"],
            group["test_poisson_deviance"],
            s=58,
            marker=_marker(name),
            color=_color(name),
            edgecolor="white",
            linewidth=0.7,
            label=_label(name),
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"Trainable parameters $P$ (log scale)")
    ax.set_ylabel("Test Poisson deviance")
    ax.set_title("Parameter-performance trade-off at full data")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.grid(True, which="both", alpha=0.3)
    _legend_if_labeled(ax, loc="best", frameon=True, ncol=2)
    ax.margins(x=0.04, y=0.07)
    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)


def plot_best_model_lift(ensemble_scores: pd.DataFrame, output_path: str | Path) -> None:
    """Plot the best public model's practical likelihood lift over the GLM."""
    _set_paper_style()
    df = add_glm_lift(ensemble_scores)
    best = (
        df.sort_values(["threshold", "test_poisson_deviance"], ascending=[True, True])
        .groupby("threshold", as_index=False)
        .first()
        .sort_values("n_train")
    )
    best["best_lift_vs_glm"] = (
        best["glm_test_poisson_deviance"] - best["test_poisson_deviance"]
    ) / 2.0

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(
        best["n_train"],
        best["best_lift_vs_glm"],
        color="black",
        linewidth=2.2,
        alpha=0.85,
        zorder=2,
    )
    ordered_best = best.sort_values("n_train").reset_index(drop=True)
    annotate_n = set()
    previous_name = None
    for _, row in ordered_best.iterrows():
        if row["config_name"] != previous_name:
            annotate_n.add(row["n_train"])
        previous_name = row["config_name"]

    for name, group in _ordered_groups(best):
        ax.scatter(
            group["n_train"],
            group["best_lift_vs_glm"],
            s=58,
            marker=_marker(name),
            color=_color(name),
            edgecolor="white",
            linewidth=0.7,
            label=_label(name),
            zorder=4,
        )
        for _, row in group.iterrows():
            if row["n_train"] not in annotate_n:
                continue
            ax.annotate(
                _label(name),
                (row["n_train"], row["best_lift_vs_glm"]),
                textcoords="offset points",
                xytext=(5, 6),
                fontsize=8,
                color=_color(name),
            )

    ax.axhline(0.0, color="black", linewidth=1.1, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Training rows $N$ (log scale)")
    ax.set_ylabel(r"Best per-policy log-likelihood lift vs GLM")
    ax.set_title("Practical significance of the best public model")
    ax.grid(True, which="both", alpha=0.3)
    _legend_if_labeled(ax, loc="upper left", frameon=True, ncol=2)
    ax.margins(x=0.04, y=0.12)
    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)


def plot_reducible_loss_fits(
    ensemble_scores: pd.DataFrame,
    scaling_fits: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot L(N)-L_inf and the fitted power-law line for each model."""
    _set_paper_style()
    fit_by_config = scaling_fits.set_index("config_name")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for name, group in _ordered_groups(ensemble_scores):
        if name not in fit_by_config.index:
            continue
        fit = fit_by_config.loc[name]
        l_inf = float(fit["l_inf"])
        alpha = float(fit["alpha"])
        a = float(fit["a"])

        group = group.sort_values("n_train")
        n = group["n_train"].to_numpy(dtype=float)
        reducible = group["test_poisson_deviance"].to_numpy(dtype=float) - l_inf
        valid = (n > 0) & (reducible > 0)
        if valid.sum() < 2:
            continue

        ax.scatter(
            n[valid],
            reducible[valid],
            s=26,
            marker=_marker(name),
            color=_color(name),
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        x_line = np.geomspace(float(n[valid].min()), float(n[valid].max()), 120)
        y_line = a * (x_line ** (-alpha))
        ax.plot(
            x_line,
            y_line,
            linewidth=2.2,
            color=_color(name),
            label=f"{_label(name)} ($\\alpha={alpha:.2f}$)",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training rows $N$ (log scale)")
    ax.set_ylabel(r"Reducible deviance $L(N)-L_\infty$ (log scale)")
    ax.set_title("Power-law fits on reducible loss")
    _legend_if_labeled(ax, loc="best", frameon=True)
    ax.grid(True, which="both", alpha=0.3)
    ax.margins(x=0.02, y=0.08)
    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)


def plot_stability_diagnostics(
    ensemble_scores: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot seed-averaging gains and train-test gaps, matching the paper diagnostic."""
    _set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2), sharex=True)

    ax = axes[0]
    for name, group in _ordered_groups(ensemble_scores):
        group = group.sort_values("n_train")
        gain = group["mean_seed_test_poisson_deviance"] - group["test_poisson_deviance"]
        ax.plot(
            group["n_train"],
            gain,
            marker=_marker(name),
            linewidth=2.2,
            markersize=4.0,
            color=_color(name),
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=_label(name),
        )
    ax.axhline(0.0, color="black", linewidth=1.1, alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel("Training rows $N$ (log scale)")
    ax.set_ylabel(r"Ensemble gain (mean seed $-$ ensemble)")
    ax.set_title("Seed averaging across data regimes")
    ax.grid(True, which="both", alpha=0.3)
    _legend_if_labeled(ax, loc="best", frameon=True, fontsize=8)

    ax = axes[1]
    for name, group in _ordered_groups(ensemble_scores):
        group = group.sort_values("n_train")
        gap = group["test_poisson_deviance"] - group["train_poisson_deviance"]
        ax.plot(
            group["n_train"],
            gap,
            marker=_marker(name),
            linewidth=2.2,
            markersize=4.0,
            color=_color(name),
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=_label(name),
        )
    ax.axhline(0.0, color="black", linewidth=1.1, alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel("Training rows $N$ (log scale)")
    ax.set_ylabel("Train-test deviance gap")
    ax.set_title("Generalization gap")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Stability diagnostics for the public freMTPL2 sweep", fontsize=12, y=1.03)
    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)


def plot_regime_gaps(ensemble_scores: pd.DataFrame, output_path: str | Path) -> None:
    """Plot each model's gap to the best model at each training fraction."""
    _set_paper_style()
    df = ensemble_scores.copy()
    best = (
        df.groupby("threshold", as_index=False)["test_poisson_deviance"]
        .min()
        .rename(columns={"test_poisson_deviance": "best_test_poisson_deviance"})
    )
    df = df.merge(best, on="threshold", how="left")
    df["gap_to_best"] = df["test_poisson_deviance"] - df["best_test_poisson_deviance"]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for name, group in _ordered_groups(df):
        group = group.sort_values("n_train")
        ax.plot(
            group["n_train"],
            group["gap_to_best"],
            marker=_marker(name),
            linewidth=2.2,
            markersize=4.2,
            color=_color(name),
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=_label(name),
        )
    ax.axhline(0.0, color="black", linewidth=1.2, alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel("Training rows $N$ (log scale)")
    ax.set_ylabel("Test deviance gap to best model")
    ax.set_title("Data-regime crossover gaps")
    ax.grid(True, which="both", alpha=0.3)
    _legend_if_labeled(ax, loc="upper right", frameon=True, ncol=2)
    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)


def make_default_figures(
    ensemble_scores_path: str | Path,
    figures_dir: str | Path,
    run_scores_path: str | Path | None = None,
    scaling_fits_path: str | Path | None = None,
) -> None:
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(ensemble_scores_path)
    plot_data_scaling(scores, figures_path / "data_scaling.png")
    plot_glm_lift(scores, figures_path / "glm_lift.png")
    plot_best_model_lift(scores, figures_path / "best_model_lift.png")

    if run_scores_path is not None and Path(run_scores_path).exists():
        run_scores = pd.read_csv(run_scores_path)
        plot_parameter_performance(scores, run_scores, figures_path / "parameter_performance.png")

    if scaling_fits_path is not None and Path(scaling_fits_path).exists():
        fits = pd.read_csv(scaling_fits_path)
        plot_reducible_loss_fits(scores, fits, figures_path / "reducible_loss_fits.png")
