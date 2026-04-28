from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

import demo


def _load_nda_module():
    nda_path = Path(__file__).resolve().parent / "nsga-nda.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("nsga_nda_module", nda_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


nda = _load_nda_module()


def _reference_point(problem_name: str, n_obj: int) -> np.ndarray:
    # Prefer reference points defined in `demo.py`, fall back to the ones bundled
    # with `nsga-nda.py` (some problems like ZDT7 exist there).
    if problem_name in demo.REFERENCE_POINTS:
        ref = np.asarray(demo.REFERENCE_POINTS[problem_name], dtype=np.float32).reshape(-1)
    elif hasattr(nda, "REFERENCE_POINTS") and problem_name in nda.REFERENCE_POINTS:
        ref = np.asarray(nda.REFERENCE_POINTS[problem_name], dtype=np.float32).reshape(-1)
    else:
        raise ValueError(f"Missing reference point for HV: {problem_name}")
    if ref.shape[0] < n_obj:
        raise ValueError(f"Reference point for {problem_name} has {ref.shape[0]} dims, expected {n_obj}.")
    return ref[:n_obj].astype(np.float32)


def latin_hypercube_sample(lower: float, upper: float, n_samples: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lhs = np.empty((n_samples, dim), dtype=np.float32)
    for j in range(dim):
        perm = rng.permutation(n_samples)
        lhs[:, j] = (perm + rng.random(n_samples)) / n_samples
    return (lower + lhs * (upper - lower)).astype(np.float32)


def generate_offspring(
    archive_x: np.ndarray,
    n_offspring: int,
    lower: float,
    upper: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    parent_idx = rng.integers(0, archive_x.shape[0], size=n_offspring)
    parents = archive_x[parent_idx]
    noise = rng.normal(loc=0.0, scale=sigma, size=parents.shape).astype(np.float32)
    return np.clip(parents + noise, lower, upper).astype(np.float32)


def _nsga2_survival(x: np.ndarray, y: np.ndarray, n_keep: int) -> tuple[np.ndarray, np.ndarray]:
    fronts, _ = demo.fast_non_dominated_sort(y)
    keep_indices: list[int] = []

    for front in fronts:
        if not front:
            continue
        if len(keep_indices) + len(front) <= n_keep:
            keep_indices.extend(front)
            continue

        crowding = demo.crowding_distance(y, front)
        order = np.argsort(-crowding)
        remaining = n_keep - len(keep_indices)
        keep_indices.extend(np.asarray(front, dtype=np.int64)[order[:remaining]].tolist())
        break

    keep = np.asarray(keep_indices, dtype=np.int64)
    return x[keep], y[keep]


def _nsga2_sort_key(values: np.ndarray) -> np.ndarray:
    fronts, ranks = demo.fast_non_dominated_sort(values)
    crowding = np.zeros(values.shape[0], dtype=np.float32)
    for front in fronts:
        if front:
            crowding[np.asarray(front, dtype=np.int64)] = demo.crowding_distance(values, front)
    return np.lexsort((values.sum(axis=1), -crowding, ranks)).astype(np.int64)


def generate_nsga2_pseudo_front(
    archive_x: np.ndarray,
    problem: Any,
    surrogate_models: list[GaussianProcessRegressor],
    n_offspring: int,
    sigma: float,
    surrogate_nsga_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    population_x = generate_offspring(
        archive_x=archive_x,
        n_offspring=n_offspring,
        lower=float(problem.lower),
        upper=float(problem.upper),
        sigma=sigma,
        rng=rng,
    )
    population_y = predict_with_gp(surrogate_models, population_x).astype(np.float32)

    for _ in range(int(surrogate_nsga_steps)):
        offspring_x = generate_offspring(
            archive_x=population_x,
            n_offspring=n_offspring,
            lower=float(problem.lower),
            upper=float(problem.upper),
            sigma=sigma,
            rng=rng,
        )
        offspring_y = predict_with_gp(surrogate_models, offspring_x).astype(np.float32)

        union_x = np.vstack([population_x, offspring_x]).astype(np.float32)
        union_y = np.vstack([population_y, offspring_y]).astype(np.float32)
        population_x, population_y = _nsga2_survival(union_x, union_y, n_keep=n_offspring)

    fronts, _ = demo.fast_non_dominated_sort(population_y)
    front0 = np.asarray(fronts[0], dtype=np.int64)
    pseudo_x = population_x[front0]
    pseudo_y = population_y[front0]
    if pseudo_x.shape[0] < n_offspring:
        order = _nsga2_sort_key(population_y)
        pseudo_x = population_x[order]
        pseudo_y = population_y[order]
    return pseudo_x.astype(np.float32), pseudo_y.astype(np.float32)


def _gp_kernel(dim: int):
    # Conservative default kernel for bounded [0,1] inputs.
    return ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=np.ones(dim, dtype=np.float32),
        length_scale_bounds=(1e-2, 1e2),
    ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-1))


def fit_gp_surrogates(archive_x: np.ndarray, archive_y: np.ndarray, seed: int) -> list[GaussianProcessRegressor]:
    archive_x = np.asarray(archive_x, dtype=np.float32)
    archive_y = np.asarray(archive_y, dtype=np.float32)
    if archive_x.ndim != 2:
        raise ValueError("archive_x must be 2D")
    if archive_y.ndim != 2:
        raise ValueError("archive_y must be 2D")
    if archive_x.shape[0] != archive_y.shape[0]:
        raise ValueError("archive_x and archive_y must have the same number of rows")

    models: list[GaussianProcessRegressor] = []
    for obj_id in range(archive_y.shape[1]):
        gpr = GaussianProcessRegressor(
            kernel=_gp_kernel(archive_x.shape[1]),
            normalize_y=True,
            random_state=seed + obj_id,
            n_restarts_optimizer=0,
        )
        gpr.fit(archive_x, archive_y[:, obj_id])
        models.append(gpr)
    return models


def predict_with_gp(models: list[GaussianProcessRegressor], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    preds = []
    for model in models:
        preds.append(model.predict(x).reshape(-1))
    return np.stack(preds, axis=1).astype(np.float32)


@dataclass(frozen=True)
class RunResult:
    archive_x: np.ndarray
    archive_y: np.ndarray
    hv_history: list[float]
    ref_point: np.ndarray


def run_nsga_random(args) -> RunResult:
    rng = np.random.default_rng(int(args.seed))
    np.random.seed(int(args.seed))

    problem_name = str(args.problem)
    problem = nda.ZDTProblem(name=problem_name, dim=int(args.dim))
    probe_x = rng.uniform(float(problem.lower), float(problem.upper), size=(1, int(args.dim))).astype(np.float32)
    n_obj = int(problem.evaluate(probe_x).shape[1])
    ref_point = _reference_point(problem_name, n_obj)

    archive_x = latin_hypercube_sample(
        lower=float(problem.lower),
        upper=float(problem.upper),
        n_samples=int(args.archive_size),
        dim=int(args.dim),
        seed=int(args.seed),
    )
    archive_y = problem.evaluate(archive_x).astype(np.float32)

    gp_seed_base = int(args.seed) + 17
    gp_models = _fit_gp_surrogates_silent(archive_x, archive_y, seed=gp_seed_base)

    hv_history: list[float] = []
    fronts, _ = demo.fast_non_dominated_sort(archive_y)
    front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    hv_history.append(demo.hypervolume_2d(front, ref_point))
    print(f"Init    | archive={archive_x.shape[0]} | front0={front.shape[0]} | HV={hv_history[-1]:.6f}")

    for step in range(int(args.outer_steps)):
        offspring_x, _ = generate_nsga2_pseudo_front(
            archive_x=archive_x,
            problem=problem,
            surrogate_models=gp_models,
            n_offspring=int(args.offspring_size),
            sigma=float(args.mutation_sigma),
            surrogate_nsga_steps=int(args.surrogate_nsga_steps),
            rng=rng,
        )

        pick = int(rng.integers(0, offspring_x.shape[0]))
        selected_x = offspring_x[pick : pick + 1]
        selected_y = problem.evaluate(selected_x).astype(np.float32)

        archive_x = np.vstack([archive_x, selected_x]).astype(np.float32)
        archive_y = np.vstack([archive_y, selected_y]).astype(np.float32)

        gp_models = _fit_gp_surrogates_silent(archive_x, archive_y, seed=gp_seed_base + step + 1)

        fronts, _ = demo.fast_non_dominated_sort(archive_y)
        front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
        hv = demo.hypervolume_2d(front, ref_point)
        hv_history.append(hv)
        print(
            f"Iter {step + 1:02d} | archive={archive_x.shape[0]} | "
            f"front0={front.shape[0]} | HV={hv:.6f}"
        )

    return RunResult(
        archive_x=archive_x,
        archive_y=archive_y,
        hv_history=hv_history,
        ref_point=ref_point,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Random selection baseline with GP surrogate + surrogate NSGA-II")
    # Backward-compatible: allow either positional problem name or `--problem`.
    parser.add_argument("problem_pos", nargs="?", type=str, help="Problem name (positional, optional)")
    parser.add_argument(
        "--problem",
        type=str,
        default=None,
        help="Problem name (e.g., ZDT1, ZDT2, ZDT3, ZDT7, DTLZ2, DTLZ5, ...)",
    )
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--archive_size", type=int, default=80)
    parser.add_argument("--outer_steps", type=int, default=40)
    parser.add_argument("--offspring_size", type=int, default=40)
    parser.add_argument("--mutation_sigma", type=float, default=0.12)
    parser.add_argument("--surrogate_nsga_steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _fit_gp_surrogates_silent(archive_x: np.ndarray, archive_y: np.ndarray, seed: int) -> list[GaussianProcessRegressor]:
    # scikit-learn GP can emit ConvergenceWarning or similar chatter; suppress it.
    sink = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            return fit_gp_surrogates(archive_x, archive_y, seed=seed)


def main():
    args = parse_args()
    if args.problem is None:
        if args.problem_pos is None:
            raise SystemExit("Missing problem name: pass `--problem <NAME>` (or positional for backward-compat).")
        args.problem = args.problem_pos
    result = run_nsga_random(args)
    plt.figure(figsize=(8, 5))
    plt.title(f"{args.problem} Hypervolume Progress (NSGA-Random, GP surrogate)")
    plt.plot(result.hv_history, marker="o")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
