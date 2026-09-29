"""KNNv2: nearest-neighbour proxy with adaptive K and tricube weights, in NumPy.

Port of ``KNNv2Adapter`` from TEPIGMeasuringTFMStructure/experiments/scm/source/adapters.py
(torch). Predictions match it except at context size 1, see ``KNNv2``.
"""

from __future__ import annotations

import numpy as np

from interfaces import RegressionPrediction


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def _pairwise_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Euclidean distances, shape ``(len(A), len(B))``."""
    sq = (A * A).sum(axis=1)[:, None] + (B * B).sum(axis=1)[None, :] - 2.0 * (A @ B.T)
    return np.sqrt(np.maximum(sq, 0.0))


def _nearest(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Column indices and distances of the ``k`` smallest entries per row, sorted by distance."""
    k = min(k, D.shape[1])
    if k < D.shape[1]:
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    else:
        idx = np.broadcast_to(np.arange(D.shape[1]), D.shape).copy()
    dist = np.take_along_axis(D, idx, axis=1)
    order = np.argsort(dist, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(dist, order, axis=1)


class _NeighbourLists:
    """For every row of a sequence, its ``n_keep`` nearest rows within the current prefix.

    A row never lists itself, so for context rows these are leave-one-out
    neighbours and for query rows ordinary neighbours. Growing the prefix by one
    row only touches rows for which the new row beats their current worst entry.
    """

    def __init__(self, X_sequence, n_keep: int):
        self.X_sequence = X_sequence  # identity is the cache key
        X = _as_2d(X_sequence)
        self.D = _pairwise_distances(X, X)
        self.idx = np.zeros((len(X), n_keep), dtype=np.int64)
        self.dist = np.full((len(X), n_keep), np.inf)
        self.size = 0

    def grow_to(self, k: int) -> None:
        cols = np.arange(self.idx.shape[1])
        for p in range(self.size, k):
            d = self.D[:, p]
            rows = np.flatnonzero(d < self.dist[:, -1])
            rows = rows[rows != p]
            if rows.size == 0:
                continue
            d_rows, old_d, old_i = d[rows, None], self.dist[rows], self.idx[rows]
            pos = (old_d <= d_rows).sum(axis=1, keepdims=True)  # after equal distances: lower index first
            shifted_d = np.concatenate([old_d[:, :1], old_d[:, :-1]], axis=1)
            shifted_i = np.concatenate([old_i[:, :1], old_i[:, :-1]], axis=1)
            self.dist[rows] = np.where(cols < pos, old_d, np.where(cols == pos, d_rows, shifted_d))
            self.idx[rows] = np.where(cols < pos, old_i, np.where(cols == pos, p, shifted_i))
        self.size = k


def _weighted_neighbour_mean(dist, nb_y, K, k, weights):
    """KNNv2 predictions from sorted neighbour distances and labels, one K per row (as ``KNNv2._predict``)."""
    n_fetch = min(int(K.max()) + 1, k)
    dist, nb_y = dist[:, :n_fetch], nb_y[:, :n_fetch]
    valid = np.arange(n_fetch) < K[:, None]
    if weights == "uniform":
        return (nb_y * valid).sum(axis=1) / valid.sum(axis=1)
    # Bandwidth: distance to neighbour K+1, or to the farthest one if only K exist.
    bandwidth = np.take_along_axis(dist, np.minimum(K, k - 1)[:, None], axis=1)
    ratio = np.minimum(dist / np.maximum(bandwidth, 1e-12), 1.0)
    w = np.clip(1.0 - ratio**3, 0.0, None) ** 3 * valid
    denom = w.sum(axis=1)
    pred = (w * nb_y).sum(axis=1) / np.maximum(denom, 1e-12)
    all_zero = denom <= 0.0
    if all_zero.any():
        pred[all_zero] = (nb_y[all_zero] * valid[all_zero]).sum(axis=1) / valid[all_zero].sum(axis=1)
    return pred


def knnv2_loss_curves(
    X_sequences: np.ndarray,  # shape (B, n_rows, n_features)
    y_sequences: np.ndarray,  # shape (B, n_rows)
    context_sizes,            # increasing ints in [1, n_rows)
    n_neighbors: int | None = None,
    weights: str = "tricube",
    k_max: int = 30,
    progress=None,
) -> np.ndarray:
    """KNNv2 mean squared error on rows ``k:`` given rows ``:k``, for every sequence and ``k``.

    Returns shape ``(B, len(context_sizes))``. Equals ``KNNv2`` applied to each
    sequence separately (each sequence uses its own K), but processes all
    sequences at once and never repeats work: rows join the context one at a
    time, each row's nearest context rows are updated from the new row's
    distances (no distance matrix is stored), and leave-one-out errors and query
    predictions are recomputed only for rows whose neighbour list changed (or,
    for predictions, whose sequence's K changed). ``progress`` (e.g. a tqdm bar)
    is advanced once per row added.
    """
    if weights not in ("uniform", "tricube"):
        raise ValueError(f"weights must be 'uniform' or 'tricube', got {weights!r}")
    X = np.asarray(X_sequences, dtype=np.float64)
    y = np.asarray(y_sequences, dtype=np.float64)
    B, n_rows, _ = X.shape
    sizes = [int(k) for k in context_sizes]
    if any(b <= a for a, b in zip(sizes, sizes[1:])) or sizes[0] < 1 or sizes[-1] >= n_rows:
        raise ValueError(f"context_sizes must increase within [1, {n_rows}); got {sizes[0]}..{sizes[-1]}.")
    adaptive = n_neighbors is None

    # Neighbour lists: each row's n_keep nearest rows within the current context (never itself).
    n_keep = (k_max if adaptive else max(1, n_neighbors)) + 1
    sq_norms = (X * X).sum(axis=2)
    nb_idx = np.zeros((B, n_rows, n_keep), dtype=np.int64)
    nb_dist = np.full((B, n_rows, n_keep), np.inf)
    flat_idx, flat_dist = nb_idx.reshape(-1, n_keep), nb_dist.reshape(-1, n_keep)
    cols = np.arange(n_keep)

    # Leave-one-out squared error of every context row for K = 1..k_max, and their per-sequence sums.
    loo_err = np.zeros((B, n_rows, k_max if adaptive else 0))
    loo_sum = np.zeros((B, k_max if adaptive else 0))
    loo_stale = np.ones((B, n_rows), dtype=bool)   # list changed since loo_err was computed
    loo_counted = np.zeros((B, n_rows), dtype=bool)
    # Cached query predictions and the K they were made with.
    pred = np.zeros((B, n_rows))
    pred_stale = np.ones((B, n_rows), dtype=bool)
    pred_K = np.full(B, -1)

    out = np.empty((B, len(sizes)))
    p = 0
    for s, k in enumerate(sizes):
        while p < k:  # row p joins every sequence's context
            d = sq_norms + sq_norms[:, p:p + 1] - 2.0 * np.einsum("bnf,bf->bn", X, X[:, p])
            d = np.sqrt(np.maximum(d, 0.0))
            d[:, p] = np.inf
            d = d.ravel()
            rows = np.flatnonzero(d < flat_dist[:, -1])
            if rows.size:
                d_rows, old_d, old_i = d[rows, None], flat_dist[rows], flat_idx[rows]
                pos = (old_d <= d_rows).sum(axis=1, keepdims=True)  # after equal distances: lower index first
                shifted_d = np.concatenate([old_d[:, :1], old_d[:, :-1]], axis=1)
                shifted_i = np.concatenate([old_i[:, :1], old_i[:, :-1]], axis=1)
                flat_dist[rows] = np.where(cols < pos, old_d, np.where(cols == pos, d_rows, shifted_d))
                flat_idx[rows] = np.where(cols < pos, old_i, np.where(cols == pos, p, shifted_i))
                loo_stale.reshape(-1)[rows] = True
                pred_stale.reshape(-1)[rows] = True
            p += 1
            if progress is not None:
                progress.update()

        if not adaptive:
            K = np.full(B, max(1, min(n_neighbors, k)))
        elif k < 2:
            K = np.ones(B, dtype=np.int64)
        else:  # K minimising the summed leave-one-out error, as KNNv2._loo_select_k
            b_i, r_i = np.nonzero(loo_stale[:, :k] | ~loo_counted[:, :k])
            if b_i.size:
                nb_y = y[b_i[:, None], nb_idx[b_i, r_i, :k_max]]
                new = (y[b_i, r_i][:, None] - np.cumsum(nb_y, axis=1) / np.arange(1, k_max + 1)) ** 2
                old = np.where(loo_counted[b_i, r_i][:, None], loo_err[b_i, r_i], 0.0)
                np.add.at(loo_sum, b_i, new - old)
                loo_err[b_i, r_i] = new
                loo_counted[b_i, r_i] = True
                loo_stale[b_i, r_i] = False
            K = np.argmin(loo_sum[:, :min(k_max, k - 1)], axis=1) + 1

        b_q, q = np.nonzero(pred_stale[:, k:] | (K != pred_K)[:, None])
        if b_q.size:
            q = q + k
            pred[b_q, q] = _weighted_neighbour_mean(nb_dist[b_q, q], y[b_q[:, None], nb_idx[b_q, q]],
                                                    K[b_q], k, weights)
            pred_stale[b_q, q] = False
        pred_K = K
        out[:, s] = ((y[:, k:] - pred[:, k:]) ** 2).mean(axis=1)
    return out


class KNNv2:
    """Nearest-neighbour mean predictor following ``interfaces.PrefixRegressionModel``.

    ``n_neighbors=None`` picks K per context by leave-one-out squared error of the
    unweighted K-neighbour mean, ``K in [1, min(k_max, n_context - 1)]`` (K=1 for a
    single context point). Predictions weight the K nearest neighbours by
    ``weights``: ``"tricube"`` with bandwidth the distance to neighbour K+1, or
    ``"uniform"``. Only ``mean`` is provided, so use with ``loss="mse"``.

    With ``cache_distances=True``, ``predict_distribution_prefix`` computes the
    distance matrix of a sequence once and keeps each row's nearest context rows
    up to date as the context grows, instead of searching neighbours per call.

    Difference to the torch original: when all tricube weights are zero, which
    happens at context size 1 (the bandwidth is the single neighbour's own
    distance), the original predicts 0; this predicts the unweighted mean of the
    K neighbours, i.e. the single context label.
    """

    def __init__(self, n_neighbors: int | None = None, weights: str = "tricube", k_max: int = 30,
                 cache_distances: bool = True):
        if weights not in ("uniform", "tricube"):
            raise ValueError(f"weights must be 'uniform' or 'tricube', got {weights!r}")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.k_max = k_max
        self.cache_distances = cache_distances
        self._lists: _NeighbourLists | None = None

    @property
    def _n_keep(self) -> int:
        """Neighbours needed per row: K + 1 for prediction, k_max for leave-one-out selection."""
        return (self.k_max if self.n_neighbors is None else max(1, self.n_neighbors)) + 1

    def predict_distribution(self, X_context, y_context, X_query, random_state: int = 0) -> RegressionPrediction:
        X_context, X_query = _as_2d(X_context), _as_2d(X_query)
        y = np.asarray(y_context, dtype=np.float64)
        self._check_context(y)
        D_context = _pairwise_distances(X_context, X_context)
        np.fill_diagonal(D_context, np.inf)
        context_nb = _nearest(D_context, self._n_keep)
        query_nb = _nearest(_pairwise_distances(X_query, X_context), self._n_keep)
        return RegressionPrediction(mean=self._predict(context_nb, query_nb, y))

    def predict_distribution_prefix(self, X_sequence, y_context, random_state: int = 0) -> RegressionPrediction:
        k = len(y_context)
        if not self.cache_distances:
            return self.predict_distribution(X_sequence[:k], y_context, X_sequence[k:], random_state)
        y = np.asarray(y_context, dtype=np.float64)
        self._check_context(y)
        lists = self._lists
        if lists is None or lists.X_sequence is not X_sequence or lists.size > k:
            lists = self._lists = _NeighbourLists(X_sequence, self._n_keep)
        lists.grow_to(k)
        context_nb = (lists.idx[:k], lists.dist[:k])
        query_nb = (lists.idx[k:], lists.dist[k:])
        return RegressionPrediction(mean=self._predict(context_nb, query_nb, y))

    @staticmethod
    def _check_context(y) -> None:
        if len(y) == 0:
            raise ValueError("KNNv2 cannot predict without context; use min_context_size >= 1.")

    def _predict(self, context_nb, query_nb, y) -> np.ndarray:
        """Predict from sorted neighbour lists (indices, distances); only the first entries that exist are used."""
        k = len(y)
        if self.n_neighbors is None:
            K = 1 if k < 2 else self._loo_select_k(*context_nb, y)
        else:
            K = max(1, min(self.n_neighbors, k))

        n_fetch = min(K + 1, k)
        idx, dist = query_nb[0][:, :n_fetch], query_nb[1][:, :n_fetch]
        if n_fetch <= K:
            # No neighbour K+1: reuse the farthest one as the tricube bandwidth, as the original does.
            dist = np.concatenate([dist, dist[:, -1:]], axis=1)
        main_dist, main_y, bandwidth = dist[:, :K], y[idx[:, :K]], dist[:, K]

        if self.weights == "uniform":
            return main_y.mean(axis=1)
        ratio = np.minimum(main_dist / np.maximum(bandwidth, 1e-12)[:, None], 1.0)
        w = np.clip(1.0 - ratio**3, 0.0, None) ** 3
        denom = w.sum(axis=1)
        pred = (w * main_y).sum(axis=1) / np.maximum(denom, 1e-12)
        all_zero = denom <= 0.0
        pred[all_zero] = main_y[all_zero].mean(axis=1)
        return pred

    def _loo_select_k(self, context_idx, context_dist, y) -> int:
        """K minimising the leave-one-out squared error of the unweighted K-neighbour mean."""
        k_max = min(self.k_max, len(y) - 1)
        loo_pred = np.cumsum(y[context_idx[:, :k_max]], axis=1) / np.arange(1, k_max + 1)
        return int(np.argmin(((y[:, None] - loo_pred) ** 2).mean(axis=0))) + 1
