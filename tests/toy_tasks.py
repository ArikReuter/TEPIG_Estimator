"""Three simple 1-D regression tasks for testing TEPIG estimators."""

from __future__ import annotations

import numpy as np

# Signal powers E[f(x)^2] are matched to 1/2 for x ~ Uniform(-1, 1), so the
# tasks differ only in structure, not in scale.
TASKS = {
    "constant": lambda x: np.full_like(x, np.sqrt(0.5)),
    "line": lambda x: np.sqrt(1.5) * x,
    "sinusoid": lambda x: np.sin(3 * np.pi * (x + 1)),  # three periods on [-1, 1]
}


def sample_task(
    task: str,
    r: int,
    n_rows: int,
    noise_std: float = 0.1,
    rng: np.random.Generator | int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``r`` data sets of ``n_rows`` points from one task.

    Return ``set_of_datasets_X`` of shape ``(r, n_rows, 1)`` with
    ``x ~ Uniform(-1, 1)`` and ``set_of_datasets_y`` of shape ``(r, n_rows)``
    with ``y = f(x) + noise_std * eps``, ``eps ~ N(0, 1)``.
    """
    rng = np.random.default_rng(rng)
    X = rng.uniform(-1.0, 1.0, size=(r, n_rows, 1))
    y = TASKS[task](X[..., 0]) + noise_std * rng.standard_normal((r, n_rows))
    return X, y
