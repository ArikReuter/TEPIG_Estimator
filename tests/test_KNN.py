"""Tests of the NumPy KNNv2 proxy.

Run with ``pytest tests`` or ``python3 tests/test_KNN.py``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

from CustomizedModels.KNN import KNNv2  # noqa: E402
from interfaces import PrefixRegressionModel  # noqa: E402
from TEPIG import TEPIG_Regression, TEPIG_Regression_KNN, _tepig_knn  # noqa: E402
from toy_tasks import sample_task  # noqa: E402

CONFIGS = [dict(), dict(weights="uniform"), dict(n_neighbors=5), dict(n_neighbors=40, weights="uniform")]


def test_follows_prefix_protocol():
    assert isinstance(KNNv2(), PrefixRegressionModel)


def test_cached_prefix_matches_direct_prediction():
    """Incremental neighbour lists must give the same predictions as a fresh search, for any order of calls."""
    rng = np.random.default_rng(0)
    for d in (1, 4):
        X = rng.normal(size=(90, d))
        y = np.sin(X.sum(axis=1)) + 0.2 * rng.normal(size=90)
        for config in CONFIGS:
            direct, cached = KNNv2(**config), KNNv2(**config)
            for k in list(range(1, 89)) + [60, 2, 89]:
                a = direct.predict_distribution(X[:k], y[:k], X[k:], 0).mean
                b = cached.predict_distribution_prefix(X, y[:k], 0).mean
                assert np.allclose(a, b, rtol=0, atol=1e-12), (d, config, k)


def test_single_context_point_predicts_its_label():
    """The torch original predicts 0 here (all tricube weights vanish); KNNv2 predicts the label."""
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(10, 2)), rng.normal(size=10) + 5.0
    for config in CONFIGS:
        pred = KNNv2(**config).predict_distribution(X[:1], y[:1], X[1:], 0).mean
        assert np.allclose(pred, y[0]), config


def test_tepig_identical_with_and_without_cache():
    X, y = sample_task("sinusoid", 4, 120, rng=2)
    for kwargs in [dict(), dict(group_size=7), dict(assume_predictive_convergence=False), dict(n_orderings=2)]:
        a = TEPIG_Regression(X, y, model=KNNv2(), return_full=True, **kwargs)["tepig_per_dataset"]
        b = TEPIG_Regression(X, y, model=KNNv2(cache_distances=False), return_full=True, **kwargs)["tepig_per_dataset"]
        assert np.array_equal(a, b), kwargs


def test_batched_knn_tepig_matches_tepig_regression():
    """TEPIG_Regression_KNN is TEPIG_Regression with KNNv2, MSE, group size 1 and min context 1, batched."""
    X, y = sample_task("sinusoid", 3, 100, rng=4)
    for kwargs in [dict(), dict(assume_predictive_convergence=True), dict(n_orderings=3), dict(n_train=30, n_test=70)]:
        kwargs = {"n_orderings": 2, **kwargs}
        a = TEPIG_Regression(X, y, model=KNNv2(), assume_predictive_convergence=kwargs.pop("assume_predictive_convergence", False),
                             return_full=True, **kwargs)
        b = TEPIG_Regression_KNN(X, y, assume_predictive_convergence=a["assume_predictive_convergence"],
                                 return_full=True, **kwargs)
        assert np.allclose(a["loss_curve_per_dataset"], b["loss_curve_per_dataset"], rtol=0, atol=1e-12), kwargs
        assert np.isclose(a["tepig"], b["tepig"], rtol=1e-12), kwargs


def test_batched_engine_is_chunk_invariant():
    X, y = sample_task("line", 5, 80, rng=5)
    a = _tepig_knn(X, y, n_orderings=2, return_full=True)["loss_curve_per_dataset"]
    b = _tepig_knn(X, y, n_orderings=2, return_full=True, max_chunk_bytes=1)["loss_curve_per_dataset"]
    assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_knnv2_is_the_default_model():
    X, y = sample_task("line", 3, 60, rng=3)
    assert TEPIG_Regression(X, y) == TEPIG_Regression(X, y, model=KNNv2())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"passed: {name}")
