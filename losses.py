"""Predefined regression losses for TEPIG, selectable by name."""

from __future__ import annotations

import numpy as np

from interfaces import RegressionLoss, RegressionPrediction


def _require(prediction: RegressionPrediction, field: str, loss_name: str):
    value = getattr(prediction, field)
    if value is None:
        raise ValueError(f"Loss {loss_name!r} needs prediction.{field}, but the model did not provide it.")
    return value


def mse(prediction: RegressionPrediction, y_query: np.ndarray) -> np.ndarray:
    """Return squared errors of the predictive mean, shape ``(n_query,)``.

    Equals Gaussian NLL with fixed variance up to scale and an additive
    constant, so TEPIG values rank tasks for a fixed model but are in squared
    target units, not nats.
    """
    mean = np.asarray(_require(prediction, "mean", "mse"), dtype=float)
    return (np.asarray(y_query, dtype=float) - mean) ** 2


def nll(prediction: RegressionPrediction, y_query: np.ndarray) -> np.ndarray:
    """Return negative log predictive densities in nats, shape ``(n_query,)``."""
    log_density = _require(prediction, "log_density", "nll")
    return -np.asarray(log_density(np.asarray(y_query, dtype=float)), dtype=float)


def pinball(prediction: RegressionPrediction, y_query: np.ndarray) -> np.ndarray:
    """Return pinball losses averaged over quantile levels, shape ``(n_query,)``."""
    quantiles = np.asarray(_require(prediction, "quantiles", "pinball"), dtype=float)
    alphas = np.asarray(_require(prediction, "alphas", "pinball"), dtype=float)
    residual = np.asarray(y_query, dtype=float)[:, None] - quantiles
    return np.maximum(alphas * residual, (alphas - 1.0) * residual).mean(axis=1)


LOSSES: dict[str, RegressionLoss] = {
    "mse": mse,
    "nll": nll,
    "pinball": pinball,
}


def resolve_loss(loss: str | RegressionLoss) -> RegressionLoss:
    """Return the loss named ``loss`` from ``LOSSES``, or ``loss`` itself if callable."""
    if isinstance(loss, str):
        if loss not in LOSSES:
            raise ValueError(f"Unknown loss {loss!r}; choose from {sorted(LOSSES)} or pass a callable.")
        return LOSSES[loss]
    if callable(loss):
        return loss
    raise TypeError(f"loss must be a string or a callable, got {type(loss).__name__}.")
