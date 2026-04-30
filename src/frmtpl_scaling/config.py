"""Configuration for the freMTPL2 open-source scaling-law workflow."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "FRMTPL.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "figures"
WUTHRICH_DATA_URL = "https://people.math.ethz.ch/~wueth/Lecture/freMTPL2freq.rda"

TARGET_COL = "ClaimNb"
EXPOSURE_COL = "Exposure"
SPLIT_COL = "set"
SAMPLE_COL = "sample_unif"
TRAIN_LABEL = "train"
TEST_LABEL = "test"

DROP_COLS = {
    "IDpol",
    "id",
    "ClaimTotal",
    TARGET_COL,
    EXPOSURE_COL,
    SPLIT_COL,
    SAMPLE_COL,
}

DEFAULT_THRESHOLDS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
SMOKE_THRESHOLDS = [0.05]
DEFAULT_REPS = 3
DEFAULT_EPOCHS = 30
SMOKE_EPOCHS = 1
DEFAULT_BATCH_SIZE = 2048

NUMERIC_BIN_COUNT = 12


def get_default_model_configs() -> OrderedDict[str, dict]:
    """Return compact Keras model configs in the style of the main notebooks."""
    return OrderedDict(
        [
            (
                "glm",
                {
                    "type": "glm",
                    "embedding_dim": 1,
                    "learning_rate": 1e-2,
                    "weight_decay": 0.0,
                    "beta_2": 0.999,
                    "batch_size": 4096,
                },
            ),
            (
                "ffn_small",
                {
                    "type": "ffn",
                    "embedding_dim": 5,
                    "dense_layers": [16],
                    "dropout": 0.01,
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-4,
                    "beta_2": 0.999,
                    "batch_size": DEFAULT_BATCH_SIZE,
                },
            ),
            (
                "transformer_multicls_small",
                {
                    "type": "transformer_multicls",
                    "embedding_dim": 24,
                    "n_heads": 2,
                    "ffn_dim": 64,
                    "n_layers": 2,
                    "n_cls": 4,
                    "transformer_dropout_rate": 0.025,
                    "ffn_dropout_rate": 0.015,
                    "cls_layernorm": True,
                    "learning_rate": 1e-3,
                    "weight_decay": 1.5e-3,
                    "beta_2": 0.98,
                    "batch_size": DEFAULT_BATCH_SIZE,
                },
            ),
            (
                "transformer_multicls_ssl_small",
                {
                    "type": "transformer_multicls_ssl",
                    "embedding_dim": 24,
                    "n_heads": 2,
                    "ffn_dim": 64,
                    "n_layers": 2,
                    "n_cls": 4,
                    "transformer_dropout_rate": 0.03,
                    "ffn_dropout_rate": 0.02,
                    "cls_layernorm": True,
                    "swap_alpha": 0.10,
                    "swap_loss_weight": 0.10,
                    "swap_ffn_dim": 32,
                    "learning_rate": 1e-3,
                    "weight_decay": 2e-3,
                    "beta_2": 0.98,
                    "batch_size": DEFAULT_BATCH_SIZE,
                },
            ),
            (
                "tabm_mini_small",
                {
                    "type": "tabm_mini",
                    "k": 8,
                    "embedding_dim": 16,
                    "dense_layers": [64, 32],
                    "dropout": 0.025,
                    "first_adapter_init": "normal_around_one",
                    "first_adapter_stddev": 0.05,
                    "output_kernel_init": "random_normal_small",
                    "output_kernel_stddev": 0.01,
                    "learning_rate": 1e-3,
                    "weight_decay": 3e-4,
                    "beta_2": 0.999,
                    "batch_size": DEFAULT_BATCH_SIZE,
                },
            ),
        ]
    )


def select_model_configs(requested: str | None) -> OrderedDict[str, dict]:
    """Select model configs by comma-separated names or broad group names."""
    all_configs = get_default_model_configs()
    if not requested or requested.strip().lower() == "all":
        return all_configs

    tokens = [token.strip().lower() for token in requested.split(",") if token.strip()]
    group_to_names = {
        "glm": ["glm"],
        "ffn": [name for name, cfg in all_configs.items() if cfg["type"] == "ffn"],
        "transformer": [
            name for name, cfg in all_configs.items() if cfg["type"].startswith("transformer")
        ],
        "ssl": [name for name, cfg in all_configs.items() if cfg["type"].endswith("_ssl")],
        "tabm": [name for name, cfg in all_configs.items() if cfg["type"] == "tabm_mini"],
        "tabm_mini": [name for name, cfg in all_configs.items() if cfg["type"] == "tabm_mini"],
    }

    requested_names: set[str] = set()
    for token in tokens:
        if token in all_configs:
            requested_names.add(token)
        elif token in group_to_names:
            requested_names.update(group_to_names[token])
        else:
            raise ValueError(f"Unknown model or group: {token}")

    return OrderedDict((name, cfg) for name, cfg in all_configs.items() if name in requested_names)
