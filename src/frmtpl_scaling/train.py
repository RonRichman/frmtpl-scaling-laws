"""Training loop for the freMTPL2 scaling-law experiments."""

from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras

from frmtpl_scaling.analysis import fit_scaling_by_family
from frmtpl_scaling.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_EPOCHS,
    DEFAULT_REPS,
    DEFAULT_RESULTS_DIR,
    DEFAULT_THRESHOLDS,
    EXPOSURE_COL,
    SAMPLE_COL,
    TARGET_COL,
    select_model_configs,
)
from frmtpl_scaling.data import load_frmtpl_csv, train_test_split_from_set
from frmtpl_scaling.losses import poisson_deviance
from frmtpl_scaling.models import build_model, set_global_seed
from frmtpl_scaling.preprocessing import base_rate, fit_preprocessor, make_keras_data


def _validation_split_indices(n_rows: int, seed: int = 42, validation_split: float = 0.10):
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows)
    rng.shuffle(idx)
    val_count = max(1, int(validation_split * n_rows))
    return idx[val_count:], idx[:val_count]


def _split_input_dict(x: dict[str, np.ndarray], train_idx, val_idx):
    return (
        {key: value[train_idx] for key, value in x.items()},
        {key: value[val_idx] for key, value in x.items()},
    )


def train_evaluate_model(
    model,
    train_x,
    train_y,
    test_x,
    test_y,
    *,
    batch_size: int,
    epochs: int,
    validation_seed: int,
):
    """Train one Keras model and return predictions plus history."""
    train_idx, val_idx = _validation_split_indices(len(train_y), seed=validation_seed)
    train_x_split, val_x_split = _split_input_dict(train_x, train_idx, val_idx)
    train_y_split = train_y[train_idx]
    val_y_split = train_y[val_idx]

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.9,
            patience=15,
            min_lr=1e-5,
            verbose=0,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]

    history = model.fit(
        x=train_x_split,
        y=train_y_split,
        validation_data=(val_x_split, val_y_split),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=True,
        verbose=0,
    )
    train_pred = model.predict(train_x, batch_size=batch_size * 4, verbose=0).reshape(-1)
    test_pred = model.predict(test_x, batch_size=batch_size * 4, verbose=0).reshape(-1)
    scores = {
        "train_poisson_deviance": poisson_deviance(train_y.reshape(-1), train_pred),
        "test_poisson_deviance": poisson_deviance(test_y.reshape(-1), test_pred),
        "epochs_trained": len(history.history.get("loss", [])),
    }
    return scores, train_pred, test_pred, history


def run_experiment(
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    models: str | None = "all",
    thresholds: list[float] | None = None,
    reps: int = DEFAULT_REPS,
    epochs: int = DEFAULT_EPOCHS,
    seeds: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the paper-style freMTPL2 sweep and write result CSVs."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    thresholds = thresholds or list(DEFAULT_THRESHOLDS)
    model_configs: OrderedDict[str, dict] = select_model_configs(models)
    seeds = seeds or list(range(1, reps + 1))

    raw = load_frmtpl_csv(data_path)
    train_raw, test_raw = train_test_split_from_set(raw)
    preprocessor = fit_preprocessor(train_raw)
    train_encoded_full = preprocessor.transform(train_raw)
    test_encoded = preprocessor.transform(test_raw)
    test_x, test_y = make_keras_data(test_encoded, preprocessor.feature_names)

    run_scores: list[dict] = []
    ensemble_scores: list[dict] = []

    for threshold in thresholds:
        train_encoded = train_encoded_full[train_encoded_full[SAMPLE_COL] <= threshold].reset_index(drop=True)
        if train_encoded.empty:
            raise ValueError(f"Threshold {threshold} produced an empty training subset.")

        train_x, train_y = make_keras_data(train_encoded, preprocessor.feature_names)
        threshold_base_rate = base_rate(train_encoded)
        n_train = len(train_encoded)
        exposure_train = float(train_encoded[EXPOSURE_COL].sum())
        claims_train = float(train_encoded[TARGET_COL].sum())

        for config_name, config in model_configs.items():
            test_preds = []
            train_preds = []
            for rep_idx, seed in enumerate(seeds, start=1):
                set_global_seed(seed)
                print(
                    f"Training {config_name} threshold={threshold:.2f} "
                    f"rep={rep_idx}/{len(seeds)} n={n_train}",
                    flush=True,
                )
                model = build_model(
                    config_name,
                    config,
                    preprocessor.feature_names,
                    preprocessor.cardinalities,
                    threshold_base_rate,
                )
                scores, train_pred, test_pred, _ = train_evaluate_model(
                    model,
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                    batch_size=int(config.get("batch_size", 2048)),
                    epochs=epochs,
                    validation_seed=1000 + seed,
                )
                test_preds.append(test_pred)
                train_preds.append(train_pred)
                seed_row = {
                    "config_name": config_name,
                    "model_type": config["type"],
                    "threshold": threshold,
                    "rep": rep_idx,
                    "seed": seed,
                    "n_train": n_train,
                    "exposure_train": exposure_train,
                    "claims_train": claims_train,
                    "base_rate": threshold_base_rate,
                    "params": int(model.count_params()),
                    **scores,
                }
                run_scores.append(seed_row)
                pd.DataFrame(run_scores).to_csv(results_path / "run_scores.csv", index=False)
                print(
                    "Seed score "
                    f"{config_name} threshold={threshold:.2f} rep={rep_idx}/{len(seeds)} "
                    f"epochs={seed_row['epochs_trained']} "
                    f"train_dev={seed_row['train_poisson_deviance']:.6f} "
                    f"test_dev={seed_row['test_poisson_deviance']:.6f}",
                    flush=True,
                )
                keras.backend.clear_session()

            mean_test_pred = np.mean(np.vstack(test_preds), axis=0)
            mean_train_pred = np.mean(np.vstack(train_preds), axis=0)
            ensemble_row = {
                "config_name": config_name,
                "model_type": config["type"],
                "threshold": threshold,
                "n_train": n_train,
                "exposure_train": exposure_train,
                "claims_train": claims_train,
                "base_rate": threshold_base_rate,
                "reps": len(seeds),
                "train_poisson_deviance": poisson_deviance(train_y.reshape(-1), mean_train_pred),
                "test_poisson_deviance": poisson_deviance(test_y.reshape(-1), mean_test_pred),
                "mean_seed_test_poisson_deviance": float(
                    np.mean(
                        [
                            row["test_poisson_deviance"]
                            for row in run_scores
                            if row["config_name"] == config_name and row["threshold"] == threshold
                        ]
                    )
                ),
            }
            ensemble_scores.append(ensemble_row)
            pd.DataFrame(ensemble_scores).to_csv(
                results_path / "ensemble_scores.csv", index=False
            )
            print(
                "Ensemble score "
                f"{config_name} threshold={threshold:.2f} reps={len(seeds)} "
                f"train_dev={ensemble_row['train_poisson_deviance']:.6f} "
                f"test_dev={ensemble_row['test_poisson_deviance']:.6f} "
                f"mean_seed_test_dev={ensemble_row['mean_seed_test_poisson_deviance']:.6f}",
                flush=True,
            )

    run_df = pd.DataFrame(run_scores)
    ensemble_df = pd.DataFrame(ensemble_scores)
    scaling_df = fit_scaling_by_family(ensemble_df)

    run_df.to_csv(results_path / "run_scores.csv", index=False)
    ensemble_df.to_csv(results_path / "ensemble_scores.csv", index=False)
    scaling_df.to_csv(results_path / "scaling_fits.csv", index=False)
    return run_df, ensemble_df, scaling_df
