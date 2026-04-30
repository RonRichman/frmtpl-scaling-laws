"""Poisson losses and metrics used across model families."""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
from keras import ops


def poisson_deviance(obs, pred, epsilon: float = 1e-7) -> float:
    """Mean Poisson deviance for observed and predicted claim counts."""
    y = np.asarray(obs, dtype="float64").reshape(-1)
    mu = np.asarray(pred, dtype="float64").reshape(-1)
    mu = np.maximum(mu, epsilon)
    y_safe = np.maximum(y, 0.0)
    ratio_term = np.zeros_like(y_safe)
    positive = y_safe > 0
    ratio_term[positive] = y_safe[positive] * np.log(y_safe[positive] / mu[positive])
    dev = 2.0 * (ratio_term - (y_safe - mu))
    return float(np.mean(dev))


@keras.saving.register_keras_serializable(package="FrmtplScaling")
class MemberwisePoissonNLL(keras.losses.Loss):
    """Mean memberwise Poisson negative log-likelihood for TabM-mini."""

    def __init__(self, epsilon: float = 1e-7, name: str = "memberwise_poisson_nll"):
        super().__init__(name=name, reduction="sum_over_batch_size")
        self.epsilon = float(epsilon)

    def call(self, y_true, mu_k):
        mu_k = ops.maximum(mu_k, self.epsilon)
        y = ops.expand_dims(ops.maximum(y_true, 0.0), axis=1)
        nll = mu_k - y * ops.log(mu_k)
        ndim = ops.ndim(nll)
        if ndim and ndim > 1:
            nll = ops.mean(nll, axis=tuple(range(1, ndim)))
        return nll

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"epsilon": self.epsilon})
        return cfg
