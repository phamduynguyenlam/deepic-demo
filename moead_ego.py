from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import demo
import multisource_eva_common as multisource


def _tchebycheff(values: np.ndarray, weights: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Tchebycheff scalarization for minimization."""
    w = np.asarray(weights, dtype=np.float32)
    v = np.asarray(values, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    return np.max(w * np.abs(v - z.reshape(1, -1)), axis=1)


def _ei(mu: np.ndarray, sigma: np.ndarray, best: float, eps: float = 1e-12) -> np.ndarray:
    """Expected Improvement for minimization (improvement = best - f)."""
    mu = np.asarray(mu, dtype=np.float32).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float32).reshape(-1)
    sigma = np.maximum(sigma, float(eps))
    z = (best - mu) / sigma
    # Standard normal pdf/cdf (no scipy dependency)
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    cdf = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
    return (best - mu) * cdf + sigma * pdf


def _random_weight_vectors(n: int, n_obj: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    w = rng.random((int(n), int(n_obj)), dtype=np.float32)
    w = np.maximum(w, 1e-6)
    w = w / w.sum(axis=1, keepdims=True)
    return w.astype(np.float32)


@dataclass(slots=True)
class MOEADEGOConfig:
    candidate_pool: int = 512
    n_weights: int = 64
    additional_fe: int = 40


def run_moead_ego_problem(
    args,
    problem_name: str,
    *,
    initial_archive_x: np.ndarray,
    config: MOEADEGOConfig | None = None,
) -> dict:
    """A lightweight MOEA/D-EGO baseline.

    - Fit independent GP surrogates per objective on the evaluated archive.
    - Generate a candidate pool via Gaussian mutation around the current archive.
    - Score candidates by maximum EI over a set of random weight vectors using
      Tchebycheff scalarization.
    - Evaluate the best candidate, update archive, refit GP; repeat for `additional_fe`.

    Logs HV after each function evaluation in the additional budget.
    """
    cfg = config or MOEADEGOConfig()
    problem = multisource.nda.ZDTProblem(name=problem_name, dim=int(args.dim))
    ref_point = multisource.nsga_eic._reference_point(problem_name, int(args.dim))

    archive_x = np.asarray(initial_archive_x, dtype=np.float32).copy()
    if archive_x.shape != (int(args.archive_size), int(args.dim)):
        raise ValueError("initial_archive_x must have shape (archive_size, dim).")
    archive_y = problem.evaluate(archive_x).astype(np.float32)

    fronts, _ = demo.fast_non_dominated_sort(archive_y)
    front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    hv_history: list[float] = [float(demo.hypervolume_2d(front, ref_point))]

    true_evals = int(archive_x.shape[0])
    print(f"MOEAD-EGO Init | archive={true_evals} | front0={front.shape[0]} | HV={hv_history[-1]:.6f}")

    # Initial fit
    gp = demo.fit_gp_surrogates(archive_x=archive_x, archive_y=archive_y, seed=int(args.seed) + 123)
    n_obj = int(archive_y.shape[1])
    weights = _random_weight_vectors(cfg.n_weights, n_obj=n_obj, seed=int(args.seed) + 456)

    for fe_idx in range(int(cfg.additional_fe)):
        # Candidate pool from current archive
        pool_x = demo.generate_offspring(
            archive_x=archive_x,
            n_offspring=int(cfg.candidate_pool),
            lower=problem.lower,
            upper=problem.upper,
            sigma=float(getattr(args, "mutation_sigma", 0.2)),
        ).astype(np.float32)

        mu = demo.predict_with_gp_mean(gp, pool_x).astype(np.float32)
        std = demo.predict_with_gp_std(gp, pool_x).astype(np.float32)

        # MOEA/D scalar bests based on current evaluated archive (z = ideal point)
        z = np.min(archive_y, axis=0).astype(np.float32)
        bests = []
        for w in weights:
            s = _tchebycheff(archive_y, w, z)
            bests.append(float(np.min(s)))
        bests_arr = np.asarray(bests, dtype=np.float32)

        # Score each candidate by max EI over weight vectors
        scores = np.full(pool_x.shape[0], -np.inf, dtype=np.float32)
        for wi, w in enumerate(weights):
            s_mu = _tchebycheff(mu, w, z)
            # very rough scalar sigma proxy (weighted L2)
            s_sigma = np.sqrt(np.sum((w.reshape(1, -1) * std) ** 2, axis=1)).astype(np.float32)
            ei = _ei(s_mu, s_sigma, best=bests_arr[wi])
            scores = np.maximum(scores, ei.astype(np.float32))

        best_idx = int(np.argmax(scores))
        x_new = pool_x[best_idx : best_idx + 1]
        y_new = problem.evaluate(x_new).astype(np.float32)

        archive_x, archive_y = demo.update_archive(
            archive_x=archive_x,
            archive_y=archive_y,
            new_x=x_new,
            new_y=y_new,
        )
        true_evals += int(x_new.shape[0])

        fronts, _ = demo.fast_non_dominated_sort(archive_y)
        front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
        hv = float(demo.hypervolume_2d(front, ref_point))
        hv_history.append(hv)
        print(f"MOEAD-EGO FE+{fe_idx + 1:02d}/{int(cfg.additional_fe)} | archive={true_evals} | HV={hv:.6f}")

        gp = demo.fit_gp_surrogates(archive_x=archive_x, archive_y=archive_y, seed=int(args.seed) + 123 + fe_idx + 1)

    fronts, _ = demo.fast_non_dominated_sort(archive_y)
    final_front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    true_front = multisource.nsga_eic._true_front(problem_name)

    return {
        "archive_x": archive_x,
        "archive_y": archive_y,
        "final_front": final_front,
        "true_front": true_front,
        "hv_history": hv_history,
        "ref_point": ref_point,
    }

