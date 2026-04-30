"""Scaling-law analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_power_law(n_train: np.ndarray, loss: np.ndarray, l_inf: float | None = None) -> dict:
    """Fit L(N) = L_inf + A * N^-alpha by log-linear regression."""
    n = np.asarray(n_train, dtype="float64")
    y = np.asarray(loss, dtype="float64")
    if len(n) < 2:
        return {"l_inf": np.nan, "alpha": np.nan, "a": np.nan, "r2": np.nan}

    if l_inf is None:
        span = max(float(np.max(y) - np.min(y)), 1e-5)
        l_inf = float(np.min(y) - 0.05 * span)

    reducible = y - l_inf
    valid = (n > 0) & (reducible > 0)
    if valid.sum() < 2:
        return {"l_inf": l_inf, "alpha": np.nan, "a": np.nan, "r2": np.nan}

    x_log = np.log(n[valid])
    y_log = np.log(reducible[valid])
    slope, intercept = np.polyfit(x_log, y_log, deg=1)
    fitted = intercept + slope * x_log
    ss_res = float(np.sum((y_log - fitted) ** 2))
    ss_tot = float(np.sum((y_log - np.mean(y_log)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "l_inf": float(l_inf),
        "alpha": float(-slope),
        "a": float(np.exp(intercept)),
        "r2": float(r2),
    }


def fit_scaling_by_family(ensemble_scores: pd.DataFrame) -> pd.DataFrame:
    """Fit data-scaling curves for each model config in the public sweep."""
    rows = []
    for config_name, group in ensemble_scores.groupby("config_name", sort=False):
        group = group.sort_values("n_train")
        fit = fit_power_law(
            group["n_train"].to_numpy(),
            group["test_poisson_deviance"].to_numpy(),
        )
        rows.append(
            {
                "config_name": config_name,
                "model_type": group["model_type"].iloc[0],
                "n_points": len(group),
                **fit,
            }
        )
    return pd.DataFrame(rows)


def add_glm_lift(ensemble_scores: pd.DataFrame) -> pd.DataFrame:
    """Add deviance and per-policy log-likelihood lift versus GLM by threshold."""
    df = ensemble_scores.copy()
    glm = (
        df[df["config_name"] == "glm"][["threshold", "test_poisson_deviance"]]
        .rename(columns={"test_poisson_deviance": "glm_test_poisson_deviance"})
    )
    df = df.merge(glm, on="threshold", how="left")
    df["deviance_lift_vs_glm"] = df["glm_test_poisson_deviance"] - df["test_poisson_deviance"]
    df["loglik_lift_vs_glm"] = df["deviance_lift_vs_glm"] / 2.0
    return df
