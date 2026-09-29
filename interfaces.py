"""Interfaces between TEPIG estimators, models, and losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np


@dataclass
class RegressionPrediction:
    """Predictive distribution of a regression model at ``n_query`` points.

    A model fills in whatever it can provide and leaves the rest ``None``.
    Each loss states which fields it needs.

    Attributes:
        mean: Predictive means, shape ``(n_query,)``.
        quantiles: Predictive quantiles, shape ``(n_query, n_alphas)``, at
            levels ``alphas``.
        alphas: Quantile levels, shape ``(n_alphas,)``, increasing in
            ``(0, 1)``.
        log_density: Maps ``y_query`` of shape ``(n_query,)`` to
            ``log q(y_query | X_query, context)`` in nats, shape
            ``(n_query,)``. A function of ``y`` so the model never sees the
            query labels.
    """

    mean: np.ndarray | None = None
    quantiles: np.ndarray | None = None
    alphas: np.ndarray | None = None
    log_density: Callable[[np.ndarray], np.ndarray] | None = None


@runtime_checkable
class RegressionModel(Protocol):
    """A regression predictor q that conditions on a context and predicts queries.

    The method is not called ``predict`` so that raw scikit-learn estimators
    fail the ``isinstance`` check instead of passing it with the wrong
    signature.
    """

    def predict_distribution(
        self,
        X_context: np.ndarray,  # shape (n_context, n_features); n_context may be 0
        y_context: np.ndarray,  # shape (n_context,)
        X_query: np.ndarray,    # shape (n_query, n_features)
        random_state: int,
    ) -> RegressionPrediction:
        """Return the predictive distribution at ``X_query`` given the context."""
        ...


class RegressionLoss(Protocol):
    """A per-point loss computed from a prediction and the true query targets."""

    def __call__(
        self,
        prediction: RegressionPrediction,
        y_query: np.ndarray,  # shape (n_query,)
    ) -> np.ndarray:
        """Return per-point losses, shape ``(n_query,)``."""
        ...
