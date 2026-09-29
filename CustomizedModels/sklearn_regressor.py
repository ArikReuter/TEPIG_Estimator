"""Wrap scikit-learn regressors as ``RegressionModel``s."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone

from interfaces import RegressionPrediction


class SklearnRegressor:
    """Mean predictions from a scikit-learn regressor, refitted per context.

    Each call fits a fresh clone of ``estimator``. Its ``random_state`` is set
    to the call's seed if the estimator has one. Only ``mean`` is provided, so
    use with ``loss="mse"``.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def predict_distribution(
        self,
        X_context: np.ndarray,  # shape (n_context, n_features)
        y_context: np.ndarray,  # shape (n_context,)
        X_query: np.ndarray,    # shape (n_query, n_features)
        random_state: int,
    ) -> RegressionPrediction:
        if len(y_context) == 0:
            raise ValueError(
                f"{type(self.estimator).__name__} cannot predict without context; "
                "use min_context_size >= 1."
            )
        est = clone(self.estimator)
        if "random_state" in est.get_params():
            est.set_params(random_state=random_state)
        est.fit(X_context, y_context)
        return RegressionPrediction(mean=np.asarray(est.predict(X_query), dtype=float))
