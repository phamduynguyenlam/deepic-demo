from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(values: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    n = values.shape[0]
    domination_count = np.zeros(n, dtype=int)
    dominated_set = [[] for _ in range(n)]
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(values[p], values[q]):
                dominated_set[p].append(q)
            elif dominates(values[q], values[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    front_id = 0
    while front_id < len(fronts) and fronts[front_id]:
        next_front = []
        for p in fronts[front_id]:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        front_id += 1

    ranks = np.full(n, fill_value=len(fronts), dtype=int)
    for idx, front in enumerate(fronts):
        for individual in front:
            ranks[individual] = idx

    return fronts, ranks


def non_dominated_indices(values: np.ndarray) -> np.ndarray:
    fronts, _ = fast_non_dominated_sort(values)
    if not fronts:
        return np.array([], dtype=np.int64)
    return np.asarray(fronts[0], dtype=np.int64)


def normalize_objectives(values: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    span = np.maximum(maxs - mins, 1e-12)
    return (values - mins) / span


def vector_angle(x: np.ndarray, y: np.ndarray) -> float:
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 1e-12:
        return 0.0
    cos_val = np.clip(np.dot(x, y) / denom, -1.0, 1.0)
    return float(np.arccos(cos_val))


def default_reference_vectors(n_obj: int, n_vectors: int) -> np.ndarray:
    if n_obj == 2:
        weights = np.linspace(0.0, 1.0, n_vectors, dtype=np.float32)
        refs = np.stack([weights, 1.0 - weights], axis=1)
    else:
        rng = np.random.default_rng(0)
        refs = rng.random((n_vectors, n_obj), dtype=np.float32)
        refs = refs / np.maximum(refs.sum(axis=1, keepdims=True), 1e-12)

    norms = np.linalg.norm(refs, axis=1, keepdims=True)
    return refs / np.maximum(norms, 1e-12)


def pbi_value(values: np.ndarray, reference_vector: np.ndarray, theta: float = 5.0) -> np.ndarray:
    ref_norm = np.linalg.norm(reference_vector)
    if ref_norm <= 1e-12:
        raise ValueError("reference_vector must be non-zero")

    d1 = np.abs(values @ reference_vector) / ref_norm
    projection = d1[:, None] * (reference_vector[None, :] / ref_norm)
    d2 = np.linalg.norm(values - projection, axis=1)
    return d1 + theta * d2


def pd_value(values: np.ndarray, reference_vector: np.ndarray, theta: float = 5.0) -> np.ndarray:
    ref_norm = np.linalg.norm(reference_vector)
    if ref_norm <= 1e-12:
        raise ValueError("reference_vector must be non-zero")

    value_norm = np.linalg.norm(values, axis=1)
    cosine = np.clip((values @ reference_vector) / np.maximum(value_norm * ref_norm, 1e-12), -1.0, 1.0)
    sine = np.sqrt(np.maximum(1.0 - cosine ** 2, 0.0))
    return values.mean(axis=1) + theta * value_norm * sine


@dataclass
class ICResult:
    indices: np.ndarray
    solutions: Optional[np.ndarray] = None


class IC(ABC):
    """Base class for infill criteria.

    The main input is a candidate population in surrogate space, and the output
    is the subset selected for real evaluation.
    """

    @abstractmethod
    def select(
        self,
        offspring_x: np.ndarray,
        offspring_pred: np.ndarray,
        archive_pred: np.ndarray,
        offspring_sigma: Optional[np.ndarray] = None,
        n_select: int = 3,
    ) -> ICResult:
        raise NotImplementedError


class EIC(IC):
    """Ensemble Infill Criterion.

    Selection order:
    1. ND-A on the current candidate non-dominated set.
    2. ND-PBI on the remaining candidate non-dominated set.
    3. EPDI on the remaining offspring.
    """

    def __init__(
        self,
        pbi_theta: float = 5.0,
        pd_theta: float = 5.0,
        mc_samples: int = 1000,
        n_reference_vectors: int = 21,
        seed: int = 0,
    ) -> None:
        self.pbi_theta = pbi_theta
        self.pd_theta = pd_theta
        self.mc_samples = mc_samples
        self.n_reference_vectors = n_reference_vectors
        self.rng = np.random.default_rng(seed)

    def select(
        self,
        offspring_x: np.ndarray,
        offspring_pred: np.ndarray,
        archive_pred: np.ndarray,
        offspring_sigma: Optional[np.ndarray] = None,
        n_select: int = 3,
    ) -> ICResult:
        offspring_x = np.asarray(offspring_x)
        offspring_pred = np.asarray(offspring_pred, dtype=np.float32)
        archive_pred = np.asarray(archive_pred, dtype=np.float32)

        if offspring_sigma is None:
            offspring_sigma = np.full_like(offspring_pred, 1e-6, dtype=np.float32)
        else:
            offspring_sigma = np.asarray(offspring_sigma, dtype=np.float32)
            if offspring_sigma.ndim == 1:
                offspring_sigma = np.repeat(offspring_sigma[:, None], offspring_pred.shape[1], axis=1)

        if offspring_pred.shape[0] == 0 or n_select <= 0:
            return ICResult(indices=np.array([], dtype=np.int64), solutions=offspring_x[:0])

        refs = default_reference_vectors(offspring_pred.shape[1], self.n_reference_vectors)

        remaining_idx = np.arange(offspring_pred.shape[0], dtype=np.int64)
        selected_idx: list[int] = []
        archive_aug = archive_pred.copy()

        first = self._select_nd_a(offspring_pred, archive_aug, remaining_idx)
        if first is not None:
            selected_idx.append(first)
            archive_aug = np.vstack([archive_aug, offspring_pred[first][None, :]])
            remaining_idx = remaining_idx[remaining_idx != first]

        if len(selected_idx) < n_select and remaining_idx.size > 0:
            second = self._select_nd_pbi(offspring_pred, archive_aug, remaining_idx, refs)
            if second is not None:
                selected_idx.append(second)
                archive_aug = np.vstack([archive_aug, offspring_pred[second][None, :]])
                remaining_idx = remaining_idx[remaining_idx != second]

        if len(selected_idx) < n_select and remaining_idx.size > 0:
            third = self._select_epdi(offspring_pred, archive_aug, offspring_sigma, remaining_idx, refs)
            if third is not None:
                selected_idx.append(third)
                remaining_idx = remaining_idx[remaining_idx != third]

        if len(selected_idx) < n_select and remaining_idx.size > 0:
            needed = n_select - len(selected_idx)
            fallback = remaining_idx[:needed].tolist()
            selected_idx.extend(fallback)

        selected = np.asarray(selected_idx[:n_select], dtype=np.int64)
        return ICResult(indices=selected, solutions=offspring_x[selected])

    def _select_nd_a(
        self,
        offspring_pred: np.ndarray,
        archive_pred: np.ndarray,
        remaining_idx: np.ndarray,
    ) -> Optional[int]:
        candidate_values = offspring_pred[remaining_idx]
        candidate_nd_local = non_dominated_indices(candidate_values)
        if candidate_nd_local.size == 0:
            return None

        archive_nd = archive_pred[non_dominated_indices(archive_pred)]
        candidate_global = remaining_idx[candidate_nd_local]
        candidate_nd = offspring_pred[candidate_global]

        combined = np.vstack([candidate_nd, archive_nd])
        mins = combined.min(axis=0)
        maxs = combined.max(axis=0)
        candidate_norm = normalize_objectives(candidate_nd, mins, maxs)
        archive_norm = normalize_objectives(archive_nd, mins, maxs)

        best_idx = candidate_global[0]
        best_angle = -np.inf
        for local_i, global_i in enumerate(candidate_global):
            if archive_norm.shape[0] == 0:
                min_angle = np.inf
            else:
                min_angle = min(vector_angle(candidate_norm[local_i], y) for y in archive_norm)
            if min_angle > best_angle:
                best_angle = min_angle
                best_idx = global_i
        return int(best_idx)

    def _select_nd_pbi(
        self,
        offspring_pred: np.ndarray,
        archive_pred: np.ndarray,
        remaining_idx: np.ndarray,
        reference_vectors: np.ndarray,
    ) -> Optional[int]:
        candidate_values = offspring_pred[remaining_idx]
        candidate_nd_local = non_dominated_indices(candidate_values)
        if candidate_nd_local.size == 0:
            return None

        archive_nd = archive_pred[non_dominated_indices(archive_pred)]
        candidate_global = remaining_idx[candidate_nd_local]
        candidate_nd = offspring_pred[candidate_global]

        combined = np.vstack([archive_nd, candidate_nd])
        mins = combined.min(axis=0)
        maxs = combined.max(axis=0)
        archive_norm = normalize_objectives(archive_nd, mins, maxs)
        candidate_norm = normalize_objectives(candidate_nd, mins, maxs)

        archive_assign = self._associate_vectors(archive_norm, reference_vectors)
        candidate_assign = self._associate_vectors(candidate_norm, reference_vectors)

        nonempty_refs = np.unique(archive_assign)
        if nonempty_refs.size == 0:
            nonempty_refs = np.unique(candidate_assign)
        if nonempty_refs.size == 0:
            return int(candidate_global[0])

        pbi_min = {}
        for ref_id in nonempty_refs:
            mask = archive_assign == ref_id
            if np.any(mask):
                pbi_min[int(ref_id)] = float(
                    np.min(pbi_value(archive_norm[mask], reference_vectors[ref_id], theta=self.pbi_theta))
                )

        best_idx = None
        best_improvement = -np.inf
        for local_i, global_i in enumerate(candidate_global):
            ref_id = int(candidate_assign[local_i])
            if ref_id not in pbi_min:
                continue
            cand_pbi = float(
                pbi_value(candidate_norm[local_i][None, :], reference_vectors[ref_id], theta=self.pbi_theta)[0]
            )
            improvement = pbi_min[ref_id] - cand_pbi
            if improvement > best_improvement:
                best_improvement = improvement
                best_idx = int(global_i)

        if best_idx is None:
            best_idx = int(candidate_global[0])
        return best_idx

    def _select_epdi(
        self,
        offspring_pred: np.ndarray,
        archive_pred: np.ndarray,
        offspring_sigma: np.ndarray,
        remaining_idx: np.ndarray,
        reference_vectors: np.ndarray,
    ) -> Optional[int]:
        if remaining_idx.size == 0:
            return None

        candidate = offspring_pred[remaining_idx]
        sigma = offspring_sigma[remaining_idx]

        combined = np.vstack([archive_pred, candidate])
        mins = combined.min(axis=0)
        maxs = combined.max(axis=0)
        archive_norm = normalize_objectives(archive_pred, mins, maxs)
        candidate_norm = normalize_objectives(candidate, mins, maxs)
        sigma_norm = sigma / np.maximum(maxs - mins, 1e-12)

        vr = reference_vectors[self.rng.integers(0, reference_vectors.shape[0])]
        pd_min = float(np.min(pd_value(archive_norm, vr, theta=self.pd_theta)))

        scores = np.zeros(candidate_norm.shape[0], dtype=np.float32)
        for i in range(candidate_norm.shape[0]):
            samples = self.rng.normal(
                loc=candidate_norm[i],
                scale=np.maximum(sigma_norm[i], 1e-6),
                size=(self.mc_samples, candidate_norm.shape[1]),
            ).astype(np.float32)
            samples = np.clip(samples, 0.0, 1.5)
            pd_samples = pd_value(samples, vr, theta=self.pd_theta)
            pdi = np.maximum(pd_min - pd_samples, 0.0)
            scores[i] = float(np.mean(pdi))

        return int(remaining_idx[int(np.argmax(scores))])

    @staticmethod
    def _associate_vectors(values: np.ndarray, reference_vectors: np.ndarray) -> np.ndarray:
        assignments = np.zeros(values.shape[0], dtype=np.int64)
        for i, value in enumerate(values):
            pbi_scores = []
            for ref in reference_vectors:
                ref_norm = np.linalg.norm(ref)
                d1 = np.abs(np.dot(value, ref)) / max(ref_norm, 1e-12)
                projection = d1 * ref / max(ref_norm, 1e-12)
                d2 = np.linalg.norm(value - projection)
                pbi_scores.append(d2)
            assignments[i] = int(np.argmin(np.asarray(pbi_scores)))
        return assignments
