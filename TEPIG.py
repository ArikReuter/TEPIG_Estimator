"""Task expected predictive information gain (TEPIG) estimators."""

from __future__ import annotations

import warnings

import numpy as np

from CustomizedModels.KNN import KNNv2, knnv2_loss_curves
from interfaces import RegressionLoss, RegressionModel
from losses import resolve_loss


def TEPIG_Regression(
    set_of_datasets_X: np.ndarray,  # shape (r, n_train + n_test, n_features)
    set_of_datasets_y: np.ndarray,  # shape (r, n_train + n_test)
    model: RegressionModel | None = None,  # default: CustomizedModels.KNN.KNNv2()
    n_train: int | None = None,     # default: set_of_datasets_X.shape[1] // 2
    n_test: int | None = None,      # default: set_of_datasets_X.shape[1] - n_train
    loss: str | RegressionLoss = "mse",
    assume_predictive_convergence: bool = True,
    group_size: int = 1,
    n_orderings: int = 1,
    min_context_size: int = 1,
    random_state: int | np.random.Generator | None = 0,
    return_full: bool = False,
    show_progress_bar: bool = False,
    verbose: bool = False,
) -> float | dict:
    """Estimate TEPIG(q, theta, n_train, n_test) for a set of datasets
    sampled from one regression task theta.

    Let ``l_k`` be the expected loss of ``model`` on a point outside a context
    of size ``k``, and write ``c = min_context_size``. With
    ``assume_predictive_convergence=True``, this returns ``TEPIG_C = sum_{k=c}^{min(n_train, n_test)-1} [l_k - l_{n_train}]``.
    Without, it returns
    ``TEPIG = sum_{k=c}^{n_test-1} [l_k - l_{k+n_train}]``.

    Each ordering shuffles the context rows and the query rows separately and
    concatenates them into one sequence. ``l_k`` is estimated as the mean loss
    on all rows after the first ``k``, given the first ``k`` as context; for
    ``k = n_train`` these are exactly the query rows.

    Args:
        set_of_datasets_X: Inputs of shape ``(r, n_train + n_test, n_features)``:
            ``r`` data sets, all drawn i.i.d. from the same task theta. Rows
            ``[:n_train]`` of each data set form the context and rows
            ``[n_train:n_train + n_test]`` the queries. If
            ``n_train + n_test < set_of_datasets_X.shape[1]``, the remaining
            rows are dropped, with a warning if ``verbose``. If
            ``n_train + n_test > set_of_datasets_X.shape[1]``, a ``ValueError``
            is raised.
        set_of_datasets_y: Targets of shape ``(r, n_train + n_test)``, aligned
            with ``set_of_datasets_X``.
        model: Predictor q following ``interfaces.RegressionModel``: its
            ``predict_distribution(X_context, y_context, X_query, random_state)``
            returns a ``RegressionPrediction`` providing the fields ``loss``
            needs. Defaults to a fresh ``KNNv2()`` (adaptive-K tricube
            nearest neighbours), which only predicts means, so pair it with
            ``loss="mse"``.
        n_train: Context size to consider, defaults to ``set_of_datasets_X.shape[1] // 2``.
        n_test: Number of query points. Defaults to ``set_of_datasets_X.shape[1] - n_train``.
        min_context_size: First context size in the sum; must be below
            ``min(n_train, n_test)``. ``0`` requires a model that predicts
            without context.
        loss: Per-point loss. A name from ``losses.LOSSES`` (``"mse"``,
            ``"nll"``, ``"pinball"``) or a callable following
            ``interfaces.RegressionLoss``. ``"nll"`` gives nats; ``"mse"``
            gives squared target units and only ranks tasks for a fixed model.
        assume_predictive_convergence: Replace ``l_{k+n_train}`` by
            ``l_{n_train}``, so the curve is only evaluated up to context size
            ``min(n_train, n_test) - 1`` plus the plateau at ``n_train``.
        group_size: Evaluate one context size per group of ``group_size``
            consecutive sizes and reuse its loss for the whole group. The
            plateau ``l_{n_train}`` under predictive convergence is always
            evaluated exactly.
        n_orderings: Random context orderings per data set.
        random_state: Seed or generator for orderings and model calls.
        return_full: Return a dict with the estimate, its standard error,
            the loss curve, per-data-set values, and settings instead of the
            estimate alone.
        show_progress_bar: Show a tqdm progress bar.
        verbose: Print warnings and per-data-set diagnostics.

    Returns:
        The TEPIG estimate averaged over the ``r`` data sets, or the full
        result dict if ``return_full``.
    """
    if model is None:
        model = KNNv2()
    if not isinstance(model, RegressionModel):
        raise TypeError(
            f"model must follow interfaces.RegressionModel (a predict_distribution method); "
            f"got {type(model).__name__}. Wrap scikit-learn estimators in SklearnRegressor."
        )
    loss_fn = resolve_loss(loss)
    X, y, n_train, n_test = _check_data(set_of_datasets_X, set_of_datasets_y, n_train, n_test,
                                        min_context_size, group_size, n_orderings, verbose)
    r = X.shape[0]
    c = min_context_size
    context_sizes, groups, position = _context_plan(n_train, n_test, c, group_size,
                                                    assume_predictive_convergence)

    # Prefix-capable models (interfaces.PrefixRegressionModel) reuse work across context sizes.
    predict_prefix = getattr(model, "predict_distribution_prefix", None)

    orders, seeds = _draw_orderings(random_state, r, n_orderings, n_train, n_test, len(groups))
    progress = None
    if show_progress_bar:
        from tqdm.auto import tqdm
        progress = tqdm(total=r * n_orderings * len(groups), desc="TEPIG")

    losses = np.empty((r, n_orderings, len(context_sizes)))
    for j in range(r):
        for o in range(n_orderings):
            X_seq, y_seq = X[j, orders[j, o]], y[j, orders[j, o]]
            for (rep, members), seed in zip(groups, seeds[j, o]):
                seed = int(seed)
                if predict_prefix is not None:
                    prediction = predict_prefix(X_seq, y_seq[:rep], seed)
                else:
                    prediction = model.predict_distribution(X_seq[:rep], y_seq[:rep], X_seq[rep:], seed)
                point_losses = np.asarray(loss_fn(prediction, y_seq[rep:]), dtype=float)
                if point_losses.shape != (len(y_seq) - rep,):
                    raise ValueError(
                        f"loss returned shape {point_losses.shape}; expected ({len(y_seq) - rep},)."
                    )
                value = float(point_losses.mean())
                for k in members:
                    losses[j, o, position[k]] = value
                if progress is not None:
                    progress.update()
    if progress is not None:
        progress.close()

    return _summarize(losses, context_sizes, position, n_train, n_test, c, loss, assume_predictive_convergence,
                      group_size, n_orderings, return_full, verbose)


def TEPIG_Regression_KNN(
    set_of_datasets_X: np.ndarray,  # shape (r, n_train + n_test, n_features)
    set_of_datasets_y: np.ndarray,  # shape (r, n_train + n_test)
    n_train: int | None = None,     # default: set_of_datasets_X.shape[1] // 2
    n_test: int | None = None,      # default: set_of_datasets_X.shape[1] - n_train
    n_orderings: int = 10,
    assume_predictive_convergence: bool = False,
    random_state: int | np.random.Generator | None = 0,
    return_full: bool = False,
    show_progress_bar: bool = False,
    verbose: bool = False,
) -> float | dict:
    """Reference KNN-TEPIG: ``TEPIG_Regression`` with ``KNNv2``, MSE, every context size from 1.

    Equals ``TEPIG_Regression(set_of_datasets_X, set_of_datasets_y, model=KNNv2(), loss="mse",
    group_size=1, min_context_size=1, ...)`` for the same ``random_state``, but evaluates all
    ``r * n_orderings`` sequences in one batch (``CustomizedModels.KNN.knnv2_loss_curves``).
    Arguments and the return value follow ``TEPIG_Regression``.

    Defaults come from a study on 999 TabICL v2 prior tasks (n_train = n_test = 512), scored by
    rank correlation with exact TEPIG over 250 data sets per task. With one data set per task,
    10 orderings raise it from 0.66 to 0.81, and exact TEPIG beats TEPIG_C (0.81 vs 0.76; TEPIG_C
    was ~12% low) at ~1.4x the cost. If more data sets per task can be drawn, pass them instead of
    raising ``n_orderings``: at equal cost, 10 data sets x 1 ordering reached 0.87.
    """
    return _tepig_knn(set_of_datasets_X, set_of_datasets_y, n_train, n_test, n_orderings,
                      assume_predictive_convergence, random_state, return_full, show_progress_bar, verbose)


def _tepig_knn(set_of_datasets_X, set_of_datasets_y, n_train=None, n_test=None, n_orderings=10,
               assume_predictive_convergence=False, random_state=0, return_full=False,
               show_progress_bar=False, verbose=False, group_size=1, min_context_size=1,
               max_chunk_bytes=256 * 2**20):
    """Batched KNNv2 TEPIG with every estimator setting exposed (used for the hyperparameter study)."""
    X, y, n_train, n_test = _check_data(set_of_datasets_X, set_of_datasets_y, n_train, n_test,
                                        min_context_size, group_size, n_orderings, verbose)
    r, n_rows, n_features = X.shape
    c = min_context_size
    if c < 1:
        raise ValueError("KNNv2 cannot predict without context; min_context_size must be >= 1.")
    context_sizes, groups, position = _context_plan(n_train, n_test, c, group_size,
                                                    assume_predictive_convergence)
    orders, _ = _draw_orderings(random_state, r, n_orderings, n_train, n_test, len(groups))

    # All (data set, ordering) sequences, evaluated in chunks that keep the arrays under max_chunk_bytes.
    X_seq = X[np.arange(r)[:, None, None], orders].reshape(r * n_orderings, n_rows, n_features)
    y_seq = y[np.arange(r)[:, None, None], orders].reshape(r * n_orderings, n_rows)
    per_sequence = 8 * n_rows * (n_features + 3 * 31 + 8)  # inputs, neighbour lists, LOO errors, caches
    chunk = max(1, int(max_chunk_bytes // per_sequence))
    reps = [rep for rep, _ in groups]

    progress = None
    if show_progress_bar:
        from tqdm.auto import tqdm
        n_chunks = -(-len(X_seq) // chunk)
        progress = tqdm(total=n_chunks * reps[-1], desc="TEPIG-KNN")
    rep_losses = np.concatenate([
        knnv2_loss_curves(X_seq[i:i + chunk], y_seq[i:i + chunk], reps, progress=progress)
        for i in range(0, len(X_seq), chunk)
    ]).reshape(r, n_orderings, len(reps))
    if progress is not None:
        progress.close()

    losses = np.empty((r, n_orderings, len(context_sizes)))
    for g, (_, members) in enumerate(groups):
        for k in members:
            losses[:, :, position[k]] = rep_losses[:, :, g]
    return _summarize(losses, context_sizes, position, n_train, n_test, c, "mse", assume_predictive_convergence,
                      group_size, n_orderings, return_full, verbose)


def _check_data(set_of_datasets_X, set_of_datasets_y, n_train, n_test, min_context_size, group_size,
                n_orderings, verbose):
    """Validate the inputs; return ``X, y`` truncated to ``n_train + n_test`` rows and the resolved sizes."""
    X = np.asarray(set_of_datasets_X, dtype=float)
    y = np.asarray(set_of_datasets_y, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"set_of_datasets_X must have shape (r, n_rows, n_features); got {X.shape}.")
    if y.shape != X.shape[:2]:
        raise ValueError(f"set_of_datasets_y must have shape {X.shape[:2]}; got {y.shape}.")
    r, n_rows, _ = X.shape

    if n_train is None:
        n_train = n_rows // 2
    if n_test is None:
        n_test = n_rows - n_train
    c = min_context_size
    if r < 1:
        raise ValueError("Need at least one data set.")
    if n_train < 1 or n_test < 1:
        raise ValueError(f"n_train and n_test must be >= 1; got {n_train} and {n_test}.")
    if n_train + n_test > n_rows:
        raise ValueError(f"n_train + n_test = {n_train + n_test} exceeds the {n_rows} rows per data set.")
    if not 0 <= c < min(n_train, n_test):
        raise ValueError(f"min_context_size must be in [0, min(n_train, n_test)) = [0, {min(n_train, n_test)}); got {c}.")
    if group_size < 1:
        raise ValueError("group_size must be >= 1.")
    if n_orderings < 1:
        raise ValueError("n_orderings must be >= 1.")
    if n_train + n_test < n_rows and verbose:
        warnings.warn(f"Using the first {n_train + n_test} of {n_rows} rows per data set; dropping the rest.")
    return X[:, :n_train + n_test], y[:, :n_train + n_test], n_train, n_test


def _context_plan(n_train, n_test, c, group_size, assume_predictive_convergence):
    """Context sizes whose loss enters TEPIG, their evaluation groups, and each size's column."""
    if assume_predictive_convergence:
        curve_sizes = list(range(c, min(n_train, n_test)))
        context_sizes = curve_sizes + [n_train]
        groups = _context_size_groups(curve_sizes, group_size) + [(n_train, [n_train])]
    else:
        context_sizes = sorted(set(range(c, n_test)) | set(range(c + n_train, n_train + n_test)))
        groups = _context_size_groups(context_sizes, group_size)
    return context_sizes, groups, {k: i for i, k in enumerate(context_sizes)}


def _draw_orderings(random_state, r, n_orderings, n_train, n_test, n_groups):
    """Per data set and ordering: a row order (context and query rows shuffled separately) and one
    model seed per group, drawn in a fixed sequence so every estimator sees identical randomness."""
    rng = np.random.default_rng(random_state)
    orders = np.empty((r, n_orderings, n_train + n_test), dtype=np.int64)
    seeds = np.empty((r, n_orderings, n_groups), dtype=np.int64)
    for j in range(r):
        for o in range(n_orderings):
            orders[j, o] = np.concatenate([rng.permutation(n_train), n_train + rng.permutation(n_test)])
            seeds[j, o] = [rng.integers(0, 2**31 - 1) for _ in range(n_groups)]
    return orders, seeds


def _summarize(losses, context_sizes, position, n_train, n_test, c, loss, assume_predictive_convergence,
               group_size, n_orderings, return_full, verbose):
    """Turn losses of shape ``(r, n_orderings, len(context_sizes))`` into TEPIG and the result dict."""
    r = losses.shape[0]
    curve_per_dataset = losses.mean(axis=1)
    if assume_predictive_convergence:
        tepig_per_dataset = (curve_per_dataset[:, :-1] - curve_per_dataset[:, -1:]).sum(axis=1)
    else:
        early = [position[k] for k in range(c, n_test)]
        late = [position[k + n_train] for k in range(c, n_test)]
        tepig_per_dataset = curve_per_dataset[:, early].sum(axis=1) - curve_per_dataset[:, late].sum(axis=1)

    if verbose:
        for j, value in enumerate(tepig_per_dataset):
            print(f"[data set {j + 1}/{r}] TEPIG = {value:.4f}")

    tepig = float(tepig_per_dataset.mean())
    if not return_full:
        return tepig
    return {
        "tepig": tepig,
        "tepig_se": float(tepig_per_dataset.std(ddof=1) / np.sqrt(r)) if r > 1 else float("nan"),
        "tepig_per_dataset": tepig_per_dataset,
        "context_sizes": np.asarray(context_sizes),
        "loss_curve": curve_per_dataset.mean(axis=0),
        "loss_curve_per_dataset": curve_per_dataset,
        "n_datasets": r,
        "n_train": n_train,
        "n_test": n_test,
        "min_context_size": c,
        "loss": loss if isinstance(loss, str) else getattr(loss, "__name__", repr(loss)),
        "assume_predictive_convergence": assume_predictive_convergence,
        "group_size": group_size,
        "n_orderings": n_orderings,
    }


def _context_size_groups(sizes: list[int], group_size: int) -> list[tuple[int, list[int]]]:
    """Split sorted context sizes into groups of at most ``group_size`` consecutive sizes.

    Return ``(representative, members)`` pairs; the representative is the
    middle member, and groups never span a gap in ``sizes``.
    """
    groups = []
    start = 0
    for i in range(1, len(sizes) + 1):
        if i == len(sizes) or sizes[i] != sizes[i - 1] + 1:
            run = sizes[start:i]
            for s in range(0, len(run), group_size):
                members = run[s:s + group_size]
                groups.append((members[len(members) // 2], members))
            start = i
    return groups
