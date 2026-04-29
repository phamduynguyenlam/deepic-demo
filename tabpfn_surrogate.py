from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


def _as_1d_float(arr: np.ndarray | Sequence[float], *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).reshape(-1)
    if out.ndim != 1 or out.size < 2:
        raise ValueError(f"{name} must be a 1D array with at least 2 elements, got shape={out.shape}.")
    return out


def _validate_bin_edges(bin_edges: np.ndarray) -> None:
    diffs = np.diff(bin_edges)
    if not np.all(np.isfinite(bin_edges)):
        raise ValueError("bin_edges must be finite.")
    if not np.all(diffs > 0):
        raise ValueError("bin_edges must be strictly increasing.")


def discretize_targets_to_bins(y: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Map continuous targets into bin indices in [0, K-1] using predefined edges."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    k = int(bin_edges.size - 1)
    # np.digitize returns indices in [1, K] for right=False
    idx = np.digitize(y, bin_edges[1:-1], right=False).astype(np.int64, copy=False)
    return np.clip(idx, 0, k - 1)


def uniform_bin_edges_from_targets(y: np.ndarray, n_bins: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        raise ValueError("Cannot create bin edges from empty targets.")
    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}.")
    lo = float(np.min(y))
    hi = float(np.max(y))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Targets must be finite to build uniform bins.")
    if hi <= lo:
        # Degenerate case: constant targets. Create a small range around lo.
        hi = lo + 1e-3
    return np.linspace(lo, hi, n_bins + 1, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class TabPFNBins:
    """Discretization bins for TabPFN bar-distribution outputs."""

    edges: np.ndarray  # (K+1,)
    midpoints: np.ndarray  # (K,)

    @classmethod
    def from_edges(cls, edges: np.ndarray | Sequence[float]) -> "TabPFNBins":
        edges_arr = _as_1d_float(edges, name="bin_edges")
        _validate_bin_edges(edges_arr)
        mid = (edges_arr[:-1] + edges_arr[1:]) * 0.5
        return cls(edges=edges_arr, midpoints=mid.astype(np.float32))

    @property
    def k(self) -> int:
        return int(self.midpoints.size)


def tabpfn_probs_to_mean_std(
    probs: np.ndarray,
    bins: TabPFNBins,
    *,
    normalize: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert TabPFN discrete probabilities into mean/std using Eq.(4).

    mean:  y_hat = sum_k p_k * mu_k
    std :  sigma = sqrt( sum_k p_k * (mu_k - y_hat)^2 )
    """
    p = np.asarray(probs, dtype=np.float32)
    if p.ndim != 2:
        raise ValueError(f"probs must have shape (N, K), got shape={p.shape}.")
    if p.shape[1] != bins.k:
        raise ValueError(f"probs K={p.shape[1]} does not match bins K={bins.k}.")

    if normalize:
        denom = np.maximum(p.sum(axis=1, keepdims=True), float(eps))
        p = p / denom

    mu = bins.midpoints.reshape(1, -1)
    mean = np.sum(p * mu, axis=1)
    var = np.sum(p * (mu - mean.reshape(-1, 1)) ** 2, axis=1)
    std = np.sqrt(np.maximum(var, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


class TabPFNObjectiveSurrogate:
    """Single-objective TabPFN surrogate producing (mean, std) via bin probabilities.

    The wrapped `model` is expected to implement:
      - fit(X, y_bin)  (optional; only needed if you call fit)
      - predict_proba(X) -> ndarray (N, K_eff)
    and to expose `classes_` after fitting (like sklearn classifiers), if K_eff < K.
    """

    def __init__(self, model: Any, bin_edges: np.ndarray | Sequence[float]):
        self.model = model
        self.bins = TabPFNBins.from_edges(bin_edges)
        self._fit_classes: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TabPFNObjectiveSurrogate":
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        if x_arr.ndim != 2:
            raise ValueError(f"x must have shape (N, d), got shape={x_arr.shape}.")
        if y_arr.shape[0] != x_arr.shape[0]:
            raise ValueError(f"x and y must have the same number of rows, got {x_arr.shape[0]} and {y_arr.shape[0]}.")

        y_bins = discretize_targets_to_bins(y_arr, self.bins.edges)
        if hasattr(self.model, "fit"):
            self.model.fit(x_arr, y_bins)
        else:
            raise TypeError("Wrapped model does not implement fit().")

        classes = getattr(self.model, "classes_", None)
        self._fit_classes = None if classes is None else np.asarray(classes, dtype=np.int64).reshape(-1)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim != 2:
            raise ValueError(f"x must have shape (N, d), got shape={x_arr.shape}.")
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("Wrapped model does not implement predict_proba().")

        p = np.asarray(self.model.predict_proba(x_arr), dtype=np.float32)
        if p.ndim != 2:
            raise ValueError(f"predict_proba() must return shape (N, K), got shape={p.shape}.")

        # If the classifier was trained on a subset of bins, pad missing bins with zeros.
        if p.shape[1] == self.bins.k:
            return p

        classes = self._fit_classes
        if classes is None:
            raise ValueError(
                f"predict_proba returned K={p.shape[1]} but model has no classes_ to map into K={self.bins.k} bins."
            )

        full = np.zeros((p.shape[0], self.bins.k), dtype=np.float32)
        for col, cls_id in enumerate(classes.tolist()):
            if 0 <= int(cls_id) < self.bins.k:
                full[:, int(cls_id)] = p[:, col]
        return full

    def predict_mean_std(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.predict_proba(x)
        return tabpfn_probs_to_mean_std(p, self.bins, normalize=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_mean_std(x)
        return mean

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        _, std = self.predict_mean_std(x)
        return std


class TabPFNSurrogate:
    """Multi-objective TabPFN surrogate (one classifier per objective)."""

    def __init__(self, objective_models: Sequence[Any], bin_edges: np.ndarray | Sequence[float]):
        if not objective_models:
            raise ValueError("objective_models must be a non-empty sequence.")
        self.objectives = [TabPFNObjectiveSurrogate(model=m, bin_edges=bin_edges) for m in objective_models]

    @property
    def n_objectives(self) -> int:
        return int(len(self.objectives))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TabPFNSurrogate":
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if x_arr.ndim != 2:
            raise ValueError(f"x must have shape (N, d), got shape={x_arr.shape}.")
        if y_arr.ndim != 2:
            raise ValueError(f"y must have shape (N, m), got shape={y_arr.shape}.")
        if y_arr.shape[1] != self.n_objectives:
            raise ValueError(f"y must have {self.n_objectives} objectives, got {y_arr.shape[1]}.")
        if y_arr.shape[0] != x_arr.shape[0]:
            raise ValueError(f"x and y must have the same number of rows, got {x_arr.shape[0]} and {y_arr.shape[0]}.")

        for obj_idx, obj in enumerate(self.objectives):
            obj.fit(x_arr, y_arr[:, obj_idx])
        return self

    def predict_proba(self, x: np.ndarray) -> list[np.ndarray]:
        return [obj.predict_proba(x) for obj in self.objectives]

    def predict_mean_std(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means: list[np.ndarray] = []
        stds: list[np.ndarray] = []
        for obj in self.objectives:
            m, s = obj.predict_mean_std(x)
            means.append(m)
            stds.append(s)
        return np.stack(means, axis=1).astype(np.float32), np.stack(stds, axis=1).astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_mean_std(x)
        return mean

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        _, std = self.predict_mean_std(x)
        return std


def build_tabpfn_surrogate(
    n_objectives: int,
    bin_edges: np.ndarray | Sequence[float],
    *,
    tabpfn_device: str = "cpu",
    use_many_class_extension: bool = False,
    random_state: int | None = 0,
) -> TabPFNSurrogate:
    """Factory helper that constructs TabPFN classifier surrogates (optional dependency).

    This uses TabPFN as a probabilistic classifier over discretized y bins.
    If `use_many_class_extension=True`, it will try to wrap TabPFNClassifier
    with the ManyClass extension when available.
    """
    try:
        from tabpfn import TabPFNClassifier  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("tabpfn is not installed. Install it with `pip install tabpfn`.") from exc

    models: list[Any] = []
    for _ in range(int(n_objectives)):
        base = TabPFNClassifier(device=str(tabpfn_device))
        if use_many_class_extension:
            try:
                from tabpfn_extensions.manyclass_classifier import ManyClassClassifier  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise ImportError(
                    "tabpfn-extensions[many_class] is required for use_many_class_extension=True."
                ) from exc
            base = ManyClassClassifier(estimator=base, random_state=random_state)
        models.append(base)
    return TabPFNSurrogate(objective_models=models, bin_edges=bin_edges)


class TabPFNMinMaxSurrogate:
    """TabPFN surrogate with per-epoch min-max normalization and adaptive bin count K.

    Workflow (per fit):
    - min-max normalize X (per-dimension) from evaluated archive.
    - min-max normalize each objective y to [0, 1] using evaluated archive.
    - choose K = min(20, max(5, int(sqrt(n_samples)))) and uniform bin edges on [0, 1].
    - fit one TabPFN classifier per objective on discretized y bins.

    Predict:
    - returns mean/std in original y scale.
    """

    def __init__(
        self,
        n_objectives: int,
        *,
        tabpfn_device: str = "cpu",
        use_many_class_extension: bool = False,
        random_state: int | None = 0,
    ):
        self.n_objectives = int(n_objectives)
        if self.n_objectives <= 0:
            raise ValueError(f"n_objectives must be positive, got {n_objectives}.")
        self.tabpfn_device = str(tabpfn_device)
        self.use_many_class_extension = bool(use_many_class_extension)
        self.random_state = random_state

        self._x_min: np.ndarray | None = None
        self._x_rng: np.ndarray | None = None
        self._y_min: np.ndarray | None = None
        self._y_rng: np.ndarray | None = None
        self._bins: TabPFNBins | None = None
        self._model: TabPFNSurrogate | None = None

    @staticmethod
    def _minmax_fit(arr: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
        min_v = np.min(arr, axis=0).astype(np.float32)
        max_v = np.max(arr, axis=0).astype(np.float32)
        rng = np.maximum(max_v - min_v, float(eps)).astype(np.float32)
        return min_v, rng

    def _norm_x(self, x: np.ndarray) -> np.ndarray:
        if self._x_min is None or self._x_rng is None:
            raise RuntimeError("TabPFNMinMaxSurrogate is not fit yet (missing x normalization stats).")
        return ((np.asarray(x, dtype=np.float32) - self._x_min) / self._x_rng).astype(np.float32)

    def _norm_y(self, y: np.ndarray) -> np.ndarray:
        if self._y_min is None or self._y_rng is None:
            raise RuntimeError("TabPFNMinMaxSurrogate is not fit yet (missing y normalization stats).")
        return ((np.asarray(y, dtype=np.float32) - self._y_min) / self._y_rng).astype(np.float32)

    def _unnorm_y_mean_std(self, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._y_min is None or self._y_rng is None:
            raise RuntimeError("TabPFNMinMaxSurrogate is not fit yet (missing y normalization stats).")
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        return (self._y_min + mean * self._y_rng).astype(np.float32), (std * self._y_rng).astype(np.float32)

    @staticmethod
    def _choose_k(n_samples: int) -> int:
        n_samples = int(n_samples)
        if n_samples <= 0:
            return 5
        k = int(np.sqrt(float(n_samples)))
        return int(min(20, max(5, k)))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TabPFNMinMaxSurrogate":
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if x_arr.ndim != 2:
            raise ValueError(f"x must have shape (N, d), got shape={x_arr.shape}.")
        if y_arr.ndim != 2:
            raise ValueError(f"y must have shape (N, m), got shape={y_arr.shape}.")
        if y_arr.shape[0] != x_arr.shape[0]:
            raise ValueError(f"x and y must have the same number of rows, got {x_arr.shape[0]} and {y_arr.shape[0]}.")
        if y_arr.shape[1] != self.n_objectives:
            raise ValueError(f"y must have {self.n_objectives} objectives, got {y_arr.shape[1]}.")

        self._x_min, self._x_rng = self._minmax_fit(x_arr)
        self._y_min, self._y_rng = self._minmax_fit(y_arr)

        x_norm = self._norm_x(x_arr)
        y_norm = self._norm_y(y_arr)

        k = self._choose_k(x_norm.shape[0])
        edges = np.linspace(0.0, 1.0, k + 1, dtype=np.float32)
        self._bins = TabPFNBins.from_edges(edges)
        self._model = build_tabpfn_surrogate(
            n_objectives=self.n_objectives,
            bin_edges=self._bins.edges,
            tabpfn_device=self.tabpfn_device,
            use_many_class_extension=self.use_many_class_extension,
            random_state=self.random_state,
        ).fit(x_norm, y_norm)
        return self

    def predict_mean_std(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("TabPFNMinMaxSurrogate is not fit yet.")
        x_norm = self._norm_x(np.asarray(x, dtype=np.float32))
        mean_norm, std_norm = self._model.predict_mean_std(x_norm)
        return self._unnorm_y_mean_std(mean_norm, std_norm)

    def predict(self, x: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_mean_std(x)
        return mean

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        _, std = self.predict_mean_std(x)
        return std
