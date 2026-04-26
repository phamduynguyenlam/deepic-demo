from __future__ import annotations

import numpy as np
import torch

import demo


CRITERION_NAMES = (
    "nd_a_convergence",
    "nd_a_diversity",
    "nd_pbi_convergence",
    "nd_pbi_diversity",
    "epdi",
)
LOWER_BETTER_CRITERIA = (False, False, True, True, False)


def normalize_criteria(criteria: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    squeeze_batch = criteria.dim() == 2
    if squeeze_batch:
        criteria = criteria.unsqueeze(0)
    mean = criteria.mean(dim=1, keepdim=True)
    std = criteria.std(dim=1, keepdim=True)
    std = torch.nan_to_num(std, nan=0.0)
    normalized = (criteria - mean) / (std + eps)
    return normalized.squeeze(0) if squeeze_batch else normalized


def _pareto_front(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    fronts, _ = demo.fast_non_dominated_sort(values)
    if not fronts or not fronts[0]:
        return values
    return values[np.asarray(fronts[0], dtype=np.int64)]


def _normalize_objectives(values: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    span = np.maximum(maxs - mins, 1e-12)
    return (values - mins) / span


def _vector_angle(x: np.ndarray, y: np.ndarray) -> float:
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 1e-12:
        return 0.0
    cos_val = np.clip(np.dot(x, y) / denom, -1.0, 1.0)
    return float(np.arccos(cos_val))


def _merge_primary_with_fallback(primary: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    primary = np.asarray(primary, dtype=np.int64).reshape(-1)
    fallback = np.asarray(fallback, dtype=np.int64).reshape(-1)
    if primary.size == 0:
        return fallback
    mask = ~np.isin(fallback, primary)
    return np.concatenate([primary, fallback[mask]])


def _normalize_scalar(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    min_v = float(values.min())
    max_v = float(values.max())
    span = max(max_v - min_v, 1e-12)
    return ((values - min_v) / span).astype(np.float32)


def _simplex_reference_vectors(n_obj: int, n_partitions: int) -> np.ndarray:
    refs: list[np.ndarray] = []
    if n_obj == 2:
        for i in range(n_partitions + 1):
            refs.append(np.array([i, n_partitions - i], dtype=np.float32) / max(n_partitions, 1))
    elif n_obj == 3:
        for i in range(n_partitions + 1):
            for j in range(n_partitions + 1 - i):
                k = n_partitions - i - j
                refs.append(np.array([i, j, k], dtype=np.float32) / max(n_partitions, 1))
    else:
        refs = [np.eye(n_obj, dtype=np.float32)[i] for i in range(n_obj)]

    ref_vectors = np.asarray(refs, dtype=np.float32)
    norms = np.linalg.norm(ref_vectors, axis=1, keepdims=True)
    return ref_vectors / np.maximum(norms, 1e-12)


def _normalize_for_pbi(values: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    all_values = np.vstack([reference_values, values]).astype(np.float32)
    mins = all_values.min(axis=0)
    spans = np.maximum(all_values.max(axis=0) - mins, 1e-12)
    return (values - mins) / spans


def _pbi_stats(normalized_values: np.ndarray, ref_vectors: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d1_all = normalized_values @ ref_vectors.T
    proj = d1_all[..., None] * ref_vectors[None, :, :]
    diff = normalized_values[:, None, :] - proj
    d2_all = np.linalg.norm(diff, axis=2)
    assoc = np.argmin(d2_all, axis=1)
    row_idx = np.arange(normalized_values.shape[0], dtype=np.int64)
    d1 = d1_all[row_idx, assoc]
    d2 = d2_all[row_idx, assoc]
    pbi = d1 + float(theta) * d2
    return assoc.astype(np.int64), d1.astype(np.float32), pbi.astype(np.float32)


def _random_unit_reference_vector(n_obj: int, rng: np.random.Generator) -> np.ndarray:
    vec = rng.random(n_obj, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return np.full(n_obj, 1.0 / np.sqrt(max(n_obj, 1)), dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _pd_value(values: np.ndarray, ref_vector: np.ndarray, theta: float = 5.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    ref_vector = np.asarray(ref_vector, dtype=np.float32)
    ref_norm = np.linalg.norm(ref_vector)
    if ref_norm <= 1e-12:
        ref_vector = np.full(ref_vector.shape[0], 1.0 / np.sqrt(max(ref_vector.shape[0], 1)), dtype=np.float32)
        ref_norm = np.linalg.norm(ref_vector)

    d1 = (values @ ref_vector) / max(ref_norm, 1e-12)
    projection = (d1 / max(ref_norm, 1e-12))[:, None] * ref_vector[None, :]
    d2 = np.linalg.norm(values - projection, axis=1)
    return d1 + float(theta) * d2


def _nd_a_components(candidate_values: np.ndarray, archive_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidate_values = np.asarray(candidate_values, dtype=np.float32)
    archive_front = _pareto_front(archive_values)
    combined = np.vstack([candidate_values, archive_front]).astype(np.float32)
    mins = combined.min(axis=0)
    maxs = combined.max(axis=0)
    candidate_norm = _normalize_objectives(candidate_values, mins, maxs)
    archive_norm = _normalize_objectives(archive_front, mins, maxs)

    angles = np.zeros(candidate_values.shape[0], dtype=np.float32)
    distances = np.zeros(candidate_values.shape[0], dtype=np.float32)
    for idx, candidate in enumerate(candidate_norm):
        if archive_norm.shape[0] == 0:
            angles[idx] = 0.0
            distances[idx] = 0.0
        else:
            angle_values = np.asarray(
                [_vector_angle(candidate, archive_point) for archive_point in archive_norm],
                dtype=np.float32,
            )
            distance_values = np.linalg.norm(archive_norm - candidate[None, :], axis=1).astype(np.float32)
            angles[idx] = float(angle_values.min())
            distances[idx] = float(distance_values.min())
    return angles, distances


def nd_a_scores(
    archive_y: np.ndarray,
    offspring_pred: np.ndarray,
    offspring_sigma: np.ndarray | None = None,
    focus: str = "diversity",
    convergence_lambda: float = 1.0,
    diversity_lambda: float = 1.0,
) -> np.ndarray:
    focus_name = str(focus).lower()
    raw_values = np.asarray(offspring_pred, dtype=np.float32)

    if focus_name == "convergence":
        angles, distances = _nd_a_components(raw_values, archive_y)
        return (-angles - float(convergence_lambda) * distances).astype(np.float32)

    if focus_name != "diversity":
        raise ValueError(f"Unsupported ND-A focus: {focus}")

    penalized_values = raw_values
    uncertainty = np.zeros(raw_values.shape[0], dtype=np.float32)
    if offspring_sigma is not None:
        sigma_arr = np.asarray(offspring_sigma, dtype=np.float32)
        penalized_values = raw_values + sigma_arr
        uncertainty = sigma_arr.mean(axis=1) if sigma_arr.ndim > 1 else sigma_arr.reshape(-1)
    angles, _ = _nd_a_components(penalized_values, archive_y)
    uncertainty = _normalize_scalar(uncertainty)
    return (angles + float(diversity_lambda) * uncertainty).astype(np.float32)


def nd_a_ranking(
    archive_y: np.ndarray,
    offspring_pred: np.ndarray,
    offspring_sigma: np.ndarray | None = None,
    use_penalized: bool | None = None,
    focus: str | None = None,
    convergence_lambda: float = 1.0,
    diversity_lambda: float = 1.0,
) -> np.ndarray:
    if focus is None:
        focus = "diversity" if use_penalized is not False else "convergence"

    primary_scores = nd_a_scores(
        archive_y=archive_y,
        offspring_pred=offspring_pred,
        offspring_sigma=offspring_sigma,
        focus=str(focus),
        convergence_lambda=convergence_lambda,
        diversity_lambda=diversity_lambda,
    )
    primary = np.argsort(-primary_scores).astype(np.int64)

    raw_values = np.asarray(offspring_pred, dtype=np.float32)
    if str(focus).lower() == "diversity" and offspring_sigma is not None:
        fallback_values = raw_values + np.asarray(offspring_sigma, dtype=np.float32)
    else:
        fallback_values = raw_values
    fallback = np.argsort(fallback_values.sum(axis=1)).astype(np.int64)
    return _merge_primary_with_fallback(primary, fallback)


def nd_pbi_values(candidate_values: np.ndarray, archive_values: np.ndarray, theta: float) -> np.ndarray:
    candidate_values = np.asarray(candidate_values, dtype=np.float32)
    archive_front = _pareto_front(archive_values)
    reference_values = np.vstack([archive_front, candidate_values]).astype(np.float32)
    ref_vectors = _simplex_reference_vectors(
        candidate_values.shape[1],
        n_partitions=max(12, candidate_values.shape[1] * 4),
    )
    candidate_norm = _normalize_for_pbi(candidate_values, reference_values)
    _, _, pbi = _pbi_stats(candidate_norm, ref_vectors, theta=float(theta))
    return pbi.astype(np.float32)


def _nd_pbi_branch_params(focus: str) -> tuple[float, float]:
    focus_name = str(focus).lower()
    if focus_name == "convergence":
        return 2.0, 0.0
    if focus_name == "diversity":
        return 8.0, 0.5
    raise ValueError(f"Unsupported ND-PBI focus: {focus}")


def nd_pbi_scores(
    archive_y: np.ndarray,
    offspring_pred: np.ndarray,
    offspring_sigma: np.ndarray,
    focus: str,
) -> np.ndarray:
    theta, empty_bonus = _nd_pbi_branch_params(focus)
    penalized = np.asarray(offspring_pred, dtype=np.float32) + np.asarray(offspring_sigma, dtype=np.float32)
    archive_front = _pareto_front(archive_y)
    reference_values = np.vstack([archive_front, penalized]).astype(np.float32)
    ref_vectors = _simplex_reference_vectors(
        penalized.shape[1],
        n_partitions=max(12, penalized.shape[1] * 4),
    )

    arnd_norm = _normalize_for_pbi(archive_front, reference_values)
    cand_norm = _normalize_for_pbi(penalized, reference_values)
    arnd_assoc, _, arnd_pbi = _pbi_stats(arnd_norm, ref_vectors, theta=theta)
    cand_assoc, cand_d1, cand_pbi = _pbi_stats(cand_norm, ref_vectors, theta=theta)

    nonempty_refs = set(int(idx) for idx in arnd_assoc.tolist())
    focus_name = str(focus).lower()
    scores = np.zeros(penalized.shape[0], dtype=np.float32)
    for idx in range(penalized.shape[0]):
        assoc = int(cand_assoc[idx])
        if assoc in nonempty_refs:
            ref_mask = arnd_assoc == assoc
            pbi_min = float(np.min(arnd_pbi[ref_mask]))
            improvement = pbi_min - float(cand_pbi[idx])
        else:
            improvement = float(empty_bonus) - float(cand_pbi[idx])

        if focus_name == "convergence":
            score = improvement - 0.05 * float(cand_d1[idx])
        else:
            score = improvement + float(empty_bonus)
        scores[idx] = float(score)
    return scores.astype(np.float32)


def nd_pbi_ranking(
    archive_y: np.ndarray,
    offspring_pred: np.ndarray,
    offspring_sigma: np.ndarray,
    focus: str,
) -> np.ndarray:
    penalized = np.asarray(offspring_pred, dtype=np.float32) + np.asarray(offspring_sigma, dtype=np.float32)
    theta, empty_bonus = _nd_pbi_branch_params(focus)
    archive_front = _pareto_front(archive_y)
    chosen: list[int] = []
    remaining = list(range(penalized.shape[0]))
    ref_vectors = _simplex_reference_vectors(
        penalized.shape[1],
        n_partitions=max(12, penalized.shape[1] * 4),
    )
    focus_name = str(focus).lower()

    while remaining:
        selected_values = (
            penalized[np.asarray(chosen, dtype=np.int64)]
            if chosen
            else np.empty((0, penalized.shape[1]), dtype=np.float32)
        )
        arnd = np.vstack([archive_front, selected_values]).astype(np.float32)
        candidate_values = penalized[np.asarray(remaining, dtype=np.int64)]
        reference_values = np.vstack([arnd, candidate_values]).astype(np.float32)

        arnd_norm = _normalize_for_pbi(arnd, reference_values)
        cand_norm = _normalize_for_pbi(candidate_values, reference_values)
        arnd_assoc, _, arnd_pbi = _pbi_stats(arnd_norm, ref_vectors, theta=theta)
        cand_assoc, cand_d1, cand_pbi = _pbi_stats(cand_norm, ref_vectors, theta=theta)

        nonempty_refs = set(int(idx) for idx in arnd_assoc.tolist())
        best_idx = None
        best_score = -np.inf

        for local_idx, global_idx in enumerate(remaining):
            assoc = int(cand_assoc[local_idx])
            if assoc in nonempty_refs:
                ref_mask = arnd_assoc == assoc
                pbi_min = float(np.min(arnd_pbi[ref_mask]))
                improvement = pbi_min - float(cand_pbi[local_idx])
            else:
                improvement = float(empty_bonus) - float(cand_pbi[local_idx])

            if focus_name == "convergence":
                score = improvement - 0.05 * float(cand_d1[local_idx])
            else:
                score = improvement + float(empty_bonus)

            if score > best_score:
                best_score = score
                best_idx = int(global_idx)

        if best_idx is None:
            break

        chosen.append(best_idx)
        remaining.remove(best_idx)

    return np.asarray(chosen, dtype=np.int64)


def epdi_statistics(
    archive_y: np.ndarray,
    offspring_pred: np.ndarray,
    offspring_sigma: np.ndarray,
    mc_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    archive_front = _pareto_front(archive_y)
    combined = np.vstack([archive_front, offspring_pred]).astype(np.float32)
    mins = combined.min(axis=0)
    maxs = combined.max(axis=0)
    archive_norm = _normalize_objectives(archive_front, mins, maxs)
    candidate_norm = _normalize_objectives(np.asarray(offspring_pred, dtype=np.float32), mins, maxs)
    sigma_norm = np.asarray(offspring_sigma, dtype=np.float32) / np.maximum(maxs - mins, 1e-12)

    rng = np.random.default_rng(seed)
    mean_epdi = np.zeros(candidate_norm.shape[0], dtype=np.float32)
    std_epdi = np.zeros(candidate_norm.shape[0], dtype=np.float32)
    n_obj = candidate_norm.shape[1]

    for idx in range(candidate_norm.shape[0]):
        ref_vector = _random_unit_reference_vector(n_obj, rng)
        pd_min = float(np.min(_pd_value(archive_norm, ref_vector))) if archive_norm.size else 0.0
        sigma = np.maximum(sigma_norm[idx], 1e-6)
        samples = rng.normal(
            loc=candidate_norm[idx],
            scale=sigma,
            size=(int(mc_samples), n_obj),
        ).astype(np.float32)
        samples = np.clip(samples, 0.0, 1.5)
        pdi_samples = np.maximum(pd_min - _pd_value(samples, ref_vector), 0.0)
        mean_epdi[idx] = float(np.mean(pdi_samples))
        std_epdi[idx] = float(np.std(pdi_samples))

    return mean_epdi, std_epdi


def epdi_ranking(
    archive_y: np.ndarray,
    offspring_pred: np.ndarray,
    offspring_sigma: np.ndarray,
    mode: str,
    seed: int,
    mc_samples: int = 1000,
) -> np.ndarray:
    epdi_mean, epdi_std = epdi_statistics(
        archive_y=archive_y,
        offspring_pred=offspring_pred,
        offspring_sigma=offspring_sigma,
        mc_samples=mc_samples,
        seed=seed,
    )
    if mode == "exploit":
        score = epdi_mean
    elif mode == "explore":
        score = epdi_mean + epdi_std
    else:
        raise ValueError(f"Unsupported EPDI mode: {mode}")
    return np.argsort(-score).astype(np.int64)


def criterion_matrix(
    archive_y: np.ndarray,
    offspring_pred: np.ndarray,
    offspring_sigma: np.ndarray,
    seed: int,
    epdi_mc_samples: int = 128,
) -> np.ndarray:
    penalized = np.asarray(offspring_pred, dtype=np.float32) + np.asarray(offspring_sigma, dtype=np.float32)
    return np.stack(
        [
            nd_a_scores(
                archive_y=archive_y,
                offspring_pred=offspring_pred,
                offspring_sigma=offspring_sigma,
                focus="convergence",
            ),
            nd_a_scores(
                archive_y=archive_y,
                offspring_pred=offspring_pred,
                offspring_sigma=offspring_sigma,
                focus="diversity",
            ),
            -nd_pbi_scores(
                archive_y=archive_y,
                offspring_pred=offspring_pred,
                offspring_sigma=offspring_sigma,
                focus="convergence",
            ),
            -nd_pbi_scores(
                archive_y=archive_y,
                offspring_pred=offspring_pred,
                offspring_sigma=offspring_sigma,
                focus="diversity",
            ),
            epdi_statistics(
                archive_y=archive_y,
                offspring_pred=offspring_pred,
                offspring_sigma=offspring_sigma,
                mc_samples=epdi_mc_samples,
                seed=seed,
            )[0],
        ],
        axis=1,
    ).astype(np.float32)


def score_candidates_from_action(
    criteria: np.ndarray,
    action: torch.Tensor,
    lower_better_criteria: tuple[bool, ...] = LOWER_BETTER_CRITERIA,
) -> tuple[np.ndarray, np.ndarray]:
    action_batch = action.unsqueeze(0) if action.dim() == 1 else action
    criteria_tensor = torch.as_tensor(criteria, dtype=action_batch.dtype, device=action_batch.device)
    if criteria_tensor.dim() == 2:
        criteria_tensor = criteria_tensor.unsqueeze(0)

    adjusted = criteria_tensor.clone()
    for idx, lower_better in enumerate(lower_better_criteria):
        if lower_better:
            adjusted[:, :, idx] = -adjusted[:, :, idx]

    normalized = normalize_criteria(adjusted)
    scores = torch.einsum("bnc,bc->bn", normalized, action_batch)
    return (
        scores.squeeze(0).detach().cpu().numpy().astype(np.float32),
        normalized.squeeze(0).detach().cpu().numpy().astype(np.float32),
    )


def select_indices_from_action(
    action: torch.Tensor,
    archive_y: np.ndarray,
    offspring_pred: np.ndarray,
    offspring_sigma: np.ndarray,
    k_eval: int,
    seed: int,
    epdi_mc_samples: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    criteria = criterion_matrix(
        archive_y=archive_y,
        offspring_pred=offspring_pred,
        offspring_sigma=offspring_sigma,
        seed=seed,
        epdi_mc_samples=epdi_mc_samples,
    )
    scores, normalized = score_candidates_from_action(criteria, action)
    k_keep = max(1, min(int(k_eval), scores.shape[0]))
    selected_idx = np.argsort(-scores).astype(np.int64)[:k_keep]
    return selected_idx, scores.astype(np.float32), normalized.astype(np.float32)
