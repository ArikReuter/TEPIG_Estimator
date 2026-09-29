"""Behavioural tests of TEPIG_Regression on the toy tasks.

Run with ``pytest tests`` or ``python3 tests/test_TEPIG.py``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

from sklearn.neighbors import KNeighborsRegressor  # noqa: E402

from CustomizedModels.sklearn_regressor import SklearnRegressor  # noqa: E402
from TEPIG import TEPIG_Regression  # noqa: E402
from toy_tasks import sample_task  # noqa: E402

KNN = SklearnRegressor(KNeighborsRegressor(n_neighbors=3))

# (n_train, n_test, assume_predictive_convergence, group_size, n_orderings, min_context_size)
ORDERING_CONFIGS = [
    (20, 20, True, 1, 1, 3),
    (20, 40, False, 1, 3, 5),
    (40, 40, False, 1, 1, 3),
    (40, 40, True, 1, 3, 3),
    (40, 10, True, 5, 1, 3),
    (80, 80, True, 5, 1, 5),
]


def _tepig(task, n_train, n_test, r=30, seed=0, **kwargs):
    X, y = sample_task(task, r, n_train + n_test, rng=seed)
    return TEPIG_Regression(KNN, X, y, n_train=n_train, n_test=n_test, return_full=True, **kwargs)


def _clearly_below(a, b, n_se=2.0):
    """Whether ``a["tepig"]`` lies below ``b["tepig"]`` by more than ``n_se`` combined standard errors."""
    return b["tepig"] - a["tepig"] > n_se * np.hypot(a["tepig_se"], b["tepig_se"])


def test_ordering_constant_line_sinusoid():
    """KNN extracts no structure from a constant, some from a line, most from a sinusoid.

    Only for n_train >= 20: with fewer points three periods are not learned,
    and TEPIG_C of the sinusoid legitimately drops below the line.
    """
    for n_train, n_test, conv, g, n_ord, c in ORDERING_CONFIGS:
        kwargs = dict(assume_predictive_convergence=conv, group_size=g, n_orderings=n_ord, min_context_size=c)
        const, line, sin = (_tepig(t, n_train, n_test, seed=s, **kwargs)
                            for s, t in enumerate(["constant", "line", "sinusoid"]))
        config = f"n_train={n_train} n_test={n_test} {kwargs}"
        assert abs(const["tepig"]) < 3 * const["tepig_se"] + 0.05, f"constant not ~0: {const['tepig']:.3f} ({config})"
        assert _clearly_below(const, line), f"constant !< line: {const['tepig']:.3f}, {line['tepig']:.3f} ({config})"
        assert _clearly_below(line, sin), f"line !< sinusoid: {line['tepig']:.3f}, {sin['tepig']:.3f} ({config})"


def test_standard_error_shrinks_like_inverse_sqrt_r():
    """Quadrupling r should roughly halve tepig_se."""
    ses = [_tepig("sinusoid", 20, 20, r=r, seed=10, min_context_size=3)["tepig_se"] for r in (25, 100, 400)]
    for small_r_se, large_r_se in zip(ses, ses[1:]):
        assert 1.4 < small_r_se / large_r_se < 2.8, f"se ratio {small_r_se / large_r_se:.2f} for 4x r; ses={ses}"


def test_convergence_assumption_agrees_only_when_curve_has_converged():
    """TEPIG_C matches exact TEPIG when the loss has plateaued by n_train, and underestimates it otherwise."""
    for task in ("constant", "line"):
        c_res = _tepig(task, 40, 40, r=100, seed=20, min_context_size=3)
        e_res = _tepig(task, 40, 40, r=100, seed=20, min_context_size=3, assume_predictive_convergence=False)
        gap = abs(e_res["tepig"] - c_res["tepig"])
        assert gap < 3 * np.hypot(c_res["tepig_se"], e_res["tepig_se"]), (
            f"{task}: TEPIG_C {c_res['tepig']:.3f} vs TEPIG {e_res['tepig']:.3f}")

    # Ten points cannot resolve three periods, so l_{n_train} is far above l_{k + n_train}.
    c_res = _tepig("sinusoid", 10, 40, r=100, seed=30, min_context_size=3)
    e_res = _tepig("sinusoid", 10, 40, r=100, seed=30, min_context_size=3, assume_predictive_convergence=False)
    assert _clearly_below(c_res, e_res, n_se=3.0), (
        f"sinusoid, n_train=10: TEPIG_C {c_res['tepig']:.3f} should be well below TEPIG {e_res['tepig']:.3f}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"passed: {name}")
