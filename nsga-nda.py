import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from pymoo.indicators.hv import HV

import demo
from problem.kan import KAN


REWARD_LOG_DIR = Path(__file__).resolve().parent / "reward_logs"


def _save_reward_log(filename: str, payload: dict) -> None:
    REWARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = REWARD_LOG_DIR / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Reward log saved to {path}")


REFERENCE_POINTS = {
    "ZDT1": np.array([0.9994, 6.0576], dtype=np.float32),
    "ZDT2": np.array([0.9994, 6.8960], dtype=np.float32),
    "ZDT3": np.array([0.9994, 6.0571], dtype=np.float32),
    "DTLZ2": np.array([2.8390, 2.9011, 2.8575], dtype=np.float32),
    "DTLZ3": np.array([2421.6427, 1905.2767, 2532.9691], dtype=np.float32),
    "DTLZ4": np.array([3.2675, 2.6443, 2.4263], dtype=np.float32),
    "DTLZ5": np.array([2.6672, 2.8009, 2.8575], dtype=np.float32),
    "DTLZ6": np.array([16.8258, 16.9194, 17.7646], dtype=np.float32),
    "DTLZ7": np.array([0.9984, 0.9961, 22.8114], dtype=np.float32),
    "RE1": np.array([2.76322289e03, 3.68876972e-02], dtype=np.float32),
    "RE2": np.array([528107.18990952, 1279320.81067113], dtype=np.float32),
    "RE3": np.array([7.68527849, 7.28609807, 21.50103909], dtype=np.float32),
    "RE4": np.array([6.79211111, 60.0, 0.4799612], dtype=np.float32),
    "RE5": np.array([0.87449713, 1.05091656, 1.05328528], dtype=np.float32),
    "RE6": np.array([749.92405125, 2229.37483405], dtype=np.float32),
    "RE7": np.array([2.10336300e02, 1.06991599e03, 3.91967702e07], dtype=np.float32),
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def latin_hypercube_sample(lower, upper, n_samples: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower_arr = np.full(dim, lower, dtype=np.float32) if np.isscalar(lower) else np.asarray(lower, dtype=np.float32)
    upper_arr = np.full(dim, upper, dtype=np.float32) if np.isscalar(upper) else np.asarray(upper, dtype=np.float32)

    lhs = np.empty((n_samples, dim), dtype=np.float32)
    for j in range(dim):
        perm = rng.permutation(n_samples)
        lhs[:, j] = (perm + rng.random(n_samples)) / n_samples

    samples = lower_arr + lhs * (upper_arr - lower_arr)
    return samples.astype(np.float32)


class ZDTProblem:
    def __init__(self, name: str, dim: int = 30):
        self.name = name
        self.dim = dim
        self.lower = 0.0
        self.upper = 1.0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        f1 = x[:, 0]
        g = 1.0 + 9.0 / (self.dim - 1.0) * np.sum(x[:, 1:], axis=1)

        if self.name == "ZDT1":
            h = 1.0 - np.sqrt(f1 / g)
        elif self.name == "ZDT2":
            h = 1.0 - (f1 / g) ** 2
        elif self.name == "ZDT3":
            h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)
        elif self.name == "ZDT7":
            g = 1.0 + 10.0 * np.sum(x[:, 1:], axis=1) / (self.dim - 1.0)
            h = 1.0 - (f1 / g) * (1.0 + np.sin(3.0 * np.pi * f1))
        elif self.name == "DTLZ1":
            g = 100.0 * (
                self.dim - 1.0
                + np.sum((x[:, 1:] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[:, 1:] - 0.5)), axis=1)
            )
            f1 = 0.5 * x[:, 0] * (1.0 + g)
            f2 = 0.5 * (1.0 - x[:, 0]) * (1.0 + g)
            return np.stack([f1, f2], axis=1).astype(np.float32)
        elif self.name == "DTLZ2":
            g = np.sum((x[:, 2:] - 0.5) ** 2, axis=1)
            theta1 = 0.5 * np.pi * x[:, 0]
            theta2 = 0.5 * np.pi * x[:, 1]
            f1 = (1.0 + g) * np.cos(theta1) * np.cos(theta2)
            f2 = (1.0 + g) * np.cos(theta1) * np.sin(theta2)
            f3 = (1.0 + g) * np.sin(theta1)
            y = np.stack([f1, f2, f3], axis=1).astype(np.float32)
            return np.maximum(y, 0.0)
        elif self.name == "DTLZ3":
            g = 100.0 * (
                self.dim - 1.0
                + np.sum((x[:, 1:] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[:, 1:] - 0.5)), axis=1)
            )
            f1 = (1.0 + g) * np.cos(0.5 * np.pi * x[:, 0])
            f2 = (1.0 + g) * np.sin(0.5 * np.pi * x[:, 0])
            return np.stack([f1, f2], axis=1).astype(np.float32)
        elif self.name == "DTLZ4":
            alpha = 100.0
            g = np.sum((x[:, 1:] - 0.5) ** 2, axis=1)
            f1 = (1.0 + g) * np.cos(0.5 * np.pi * (x[:, 0] ** alpha))
            f2 = (1.0 + g) * np.sin(0.5 * np.pi * (x[:, 0] ** alpha))
            return np.stack([f1, f2], axis=1).astype(np.float32)
        elif self.name == "DTLZ5":
            g = np.sum((x[:, 2:] - 0.5) ** 2, axis=1)
            theta1 = 0.5 * np.pi * x[:, 0]
            theta2 = (np.pi / (4.0 * (1.0 + g))) * (1.0 + 2.0 * g * x[:, 1])
            f1 = (1.0 + g) * np.cos(theta1) * np.cos(theta2)
            f2 = (1.0 + g) * np.cos(theta1) * np.sin(theta2)
            f3 = (1.0 + g) * np.sin(theta1)
            y = np.stack([f1, f2, f3], axis=1).astype(np.float32)
            return np.maximum(y, 0.0)
        elif self.name == "DTLZ6":
            g = np.sum(x[:, 1:] ** 0.1, axis=1)
            theta = x[:, 0] * np.pi / 2.0
            f1 = (1.0 + g) * np.cos(theta)
            f2 = (1.0 + g) * np.sin(theta)
            return np.stack([f1, f2], axis=1).astype(np.float32)
        elif self.name == "DTLZ7":
            g = 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self.dim - 1.0)
            f1 = x[:, 0]
            h = 2.0 - (f1 / (1.0 + g)) * (1.0 + np.sin(3.0 * np.pi * f1))
            f2 = (1.0 + g) * h
            return np.stack([f1, f2], axis=1).astype(np.float32)
        else:
            raise ValueError(f"Unsupported problem: {self.name}")

        f2 = g * h
        return np.stack([f1, f2], axis=1).astype(np.float32)


def build_dataset(x: np.ndarray, y: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    n = x.shape[0]
    n_train = max(2, int(0.8 * n))
    perm = np.random.permutation(n)
    train_id = perm[:n_train]
    test_id = perm[n_train:] if n_train < n else perm[: min(2, n)]
    return {
        "train_input": torch.tensor(x[train_id], dtype=torch.float32, device=device),
        "train_label": torch.tensor(y[train_id], dtype=torch.float32, device=device),
        "test_input": torch.tensor(x[test_id], dtype=torch.float32, device=device),
        "test_label": torch.tensor(y[test_id], dtype=torch.float32, device=device),
    }


def fit_kan_surrogates(
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    device: str,
    kan_steps: int,
    hidden_width: int,
    grid: int,
    seed: int,
) -> list[KAN]:
    models = []
    for obj_id in range(archive_y.shape[1]):
        dataset = build_dataset(archive_x, archive_y[:, [obj_id]], device)
        model = KAN(
            width=[archive_x.shape[1], hidden_width, 1],
            grid=grid,
            k=3,
            seed=seed + obj_id,
            device=device,
            auto_save=False,
            save_act=False,
        )
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                model.fit(
                    dataset,
                    opt="Adam",
                    steps=kan_steps,
                    lr=1e-2,
                    batch=-1,
                    update_grid=False,
                    lamb=0.0,
                    log=1,
                )
        models.append(model)
    return models


def predict_with_kan(models: Sequence[Any], x: np.ndarray, device: str) -> np.ndarray:
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    preds = []
    for model in models:
        with torch.no_grad():
            pred = model(x_tensor).detach().cpu().numpy().reshape(-1)
        preds.append(pred)
    return np.stack(preds, axis=1).astype(np.float32)


def estimate_uncertainty(
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    archive_pred: np.ndarray,
    offspring_x: np.ndarray,
    n_neighbors: int = 5,
) -> np.ndarray:
    residual = np.abs(archive_pred - archive_y)
    n_neighbors = min(n_neighbors, archive_x.shape[0])
    dist = np.linalg.norm(offspring_x[:, None, :] - archive_x[None, :, :], axis=-1)
    nn_idx = np.argsort(dist, axis=1)[:, :n_neighbors]
    local_residual = residual[nn_idx]
    return local_residual.mean(axis=1) + 1e-6


def generate_offspring(
    archive_x: np.ndarray,
    n_offspring: int,
    lower: float,
    upper: float,
    sigma: float,
) -> np.ndarray:
    parent_idx = np.random.randint(0, archive_x.shape[0], size=n_offspring)
    parents = archive_x[parent_idx]
    noise = np.random.normal(loc=0.0, scale=sigma, size=parents.shape).astype(np.float32)
    return np.clip(parents + noise, lower, upper)


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


def _crowding_distance(values: np.ndarray, front: list[int]) -> np.ndarray:
    if not front:
        return np.array([], dtype=np.float32)

    distance = np.zeros(len(front), dtype=np.float32)
    front_values = values[np.asarray(front, dtype=np.int64)]
    n_obj = values.shape[1]

    for obj_id in range(n_obj):
        order = np.argsort(front_values[:, obj_id])
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        obj_min = front_values[order[0], obj_id]
        obj_max = front_values[order[-1], obj_id]
        denom = max(obj_max - obj_min, 1e-12)

        for idx in range(1, len(front) - 1):
            prev_val = front_values[order[idx - 1], obj_id]
            next_val = front_values[order[idx + 1], obj_id]
            distance[order[idx]] += (next_val - prev_val) / denom

    return distance


def _nsga2_survival(x: np.ndarray, y: np.ndarray, n_keep: int) -> tuple[np.ndarray, np.ndarray]:
    fronts, _ = fast_non_dominated_sort(y)
    keep_indices: list[int] = []

    for front in fronts:
        if not front:
            continue
        if len(keep_indices) + len(front) <= n_keep:
            keep_indices.extend(front)
            continue

        crowding = _crowding_distance(y, front)
        order = np.argsort(-crowding)
        remaining = n_keep - len(keep_indices)
        keep_indices.extend(np.asarray(front, dtype=np.int64)[order[:remaining]].tolist())
        break

    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    return x[keep_indices], y[keep_indices]


def _nsga2_sort_key(values: np.ndarray) -> np.ndarray:
    fronts, ranks = fast_non_dominated_sort(values)
    crowding = np.zeros(values.shape[0], dtype=np.float32)

    for front in fronts:
        if front:
            crowding[np.asarray(front, dtype=np.int64)] = _crowding_distance(values, front)

    return np.lexsort((values.sum(axis=1), -crowding, ranks)).astype(np.int64)


def generate_nsga2_pseudo_front(
    archive_x: np.ndarray,
    problem,
    surrogates,
    device: str,
    n_offspring: int,
    sigma: float,
    surrogate_nsga_steps: int,
    predict_fn,
    generate_fn,
) -> tuple[np.ndarray, np.ndarray]:
    population_x = generate_fn(
        archive_x=archive_x,
        n_offspring=n_offspring,
        lower=problem.lower,
        upper=problem.upper,
        sigma=sigma,
    ).astype(np.float32)
    population_y = predict_fn(surrogates, population_x, device).astype(np.float32)

    for _ in range(surrogate_nsga_steps):
        offspring_x = generate_fn(
            archive_x=population_x,
            n_offspring=n_offspring,
            lower=problem.lower,
            upper=problem.upper,
            sigma=sigma,
        ).astype(np.float32)
        offspring_y = predict_fn(surrogates, offspring_x, device).astype(np.float32)

        union_x = np.vstack([population_x, offspring_x])
        union_y = np.vstack([population_y, offspring_y])
        population_x, population_y = _nsga2_survival(union_x, union_y, n_keep=n_offspring)

    fronts, _ = fast_non_dominated_sort(population_y)
    pseudo_front_idx = np.asarray(fronts[0], dtype=np.int64)
    pseudo_front_x = population_x[pseudo_front_idx]
    pseudo_front_y = population_y[pseudo_front_idx]

    if pseudo_front_x.shape[0] < n_offspring:
        order = _nsga2_sort_key(population_y)
        pseudo_front_x = population_x[order]
        pseudo_front_y = population_y[order]

    return pseudo_front_x.astype(np.float32), pseudo_front_y.astype(np.float32)


def update_archive(
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    new_x: np.ndarray,
    new_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    archive_x = np.asarray(archive_x)
    archive_y = np.asarray(archive_y)
    new_x = np.asarray(new_x)
    new_y = np.asarray(new_y)

    if new_x.size == 0 or new_y.size == 0:
        return archive_x, archive_y

    if archive_x.size == 0:
        return new_x, new_y

    # Keep all true-evaluated individuals (including dominated ones).
    return np.vstack([archive_x, new_x]), np.vstack([archive_y, new_y])


def hypervolume_2d(pareto: np.ndarray, ref: np.ndarray) -> float:
    if pareto.size == 0 or len(pareto) == 0:
        return 0.0
    return float(HV(ref_point=ref)(pareto))


def pre_train_kan_surrogate_for_problem(
    problem: ZDTProblem,
    device: str,
    kan_steps: int,
    hidden_width: int,
    grid: int,
    seed: int,
    n_samples: int = 1000,
) -> tuple[np.ndarray, np.ndarray, list[KAN]]:
    x_data = np.random.uniform(problem.lower, problem.upper, size=(n_samples, problem.dim)).astype(np.float32)
    y_data = problem.evaluate(x_data)
    models = fit_kan_surrogates(
        archive_x=x_data,
        archive_y=y_data,
        device=device,
        kan_steps=kan_steps * 4,
        hidden_width=hidden_width,
        grid=grid,
        seed=seed,
    )
    return x_data, y_data, models


def _normalize_objectives(values: np.ndarray) -> np.ndarray:
    mins = values.min(axis=0)
    spans = np.maximum(values.max(axis=0) - mins, 1e-12)
    scaled = (values - mins) / spans
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    return scaled / np.maximum(norms, 1e-12)


def angle_based_select(values: np.ndarray, k: int) -> np.ndarray:
    if k >= values.shape[0]:
        return np.arange(values.shape[0], dtype=np.int64)
    if k <= 0:
        return np.array([], dtype=np.int64)

    directions = _normalize_objectives(values)
    selected: list[int] = []

    for obj_id in range(values.shape[1]):
        idx = int(np.argmin(values[:, obj_id]))
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= k:
            return np.asarray(selected[:k], dtype=np.int64)

    if not selected:
        selected.append(int(np.argmin(values.sum(axis=1))))

    remaining = [idx for idx in range(values.shape[0]) if idx not in selected]
    while len(selected) < k and remaining:
        best_idx = remaining[0]
        best_score = -np.inf
        for idx in remaining:
            dots = np.clip(directions[selected] @ directions[idx], -1.0, 1.0)
            min_angle = float(np.min(np.arccos(dots)))
            if min_angle > best_score:
                best_score = min_angle
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return np.asarray(selected, dtype=np.int64)


def select_nda(offspring_pred: np.ndarray, offspring_sigma: np.ndarray, k: int) -> np.ndarray:
    penalized = offspring_pred + offspring_sigma
    fronts, _ = fast_non_dominated_sort(penalized)
    chosen: list[int] = []

    for front in fronts:
        if not front:
            continue
        remaining = k - len(chosen)
        if remaining <= 0:
            break
        front_idx = np.asarray(front, dtype=np.int64)
        if len(front_idx) <= remaining:
            chosen.extend(front_idx.tolist())
            continue

        local_pick = angle_based_select(penalized[front_idx], remaining)
        chosen.extend(front_idx[local_pick].tolist())
        break

    return np.asarray(chosen[:k], dtype=np.int64)


def parse_args():
    parser = argparse.ArgumentParser(description="ZDT1 optimization with KAN surrogate + ND-A infill")
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--archive_size", type=int, default=80)
    parser.add_argument("--offspring_size", type=int, default=24)
    parser.add_argument("--k_eval", type=int, default=5)
    parser.add_argument("--max_fe", type=int, default=120)
    parser.add_argument("--mutation_sigma", type=float, default=0.12)
    parser.add_argument("--kan_steps", type=int, default=25)
    parser.add_argument("--kan_hidden", type=int, default=10)
    parser.add_argument("--kan_grid", type=int, default=5)
    parser.add_argument("--deepic_hidden", type=int, default=64)
    parser.add_argument("--deepic_heads", type=int, default=4)
    parser.add_argument("--deepic_ff", type=int, default=128)
    parser.add_argument("--deepic_lr", type=float, default=1e-4)
    parser.add_argument("--deepic_adapt_steps", type=int, default=8)
    parser.add_argument("--surrogate_nsga_steps", type=int, default=40)
    parser.add_argument("--discount", type=float, default=0.99, help="Reward discount/multiplier used during RL updates")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--compare", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-NDA on ZDT1")
    return parser.parse_args()


def run_nsga_nda(args, plot: bool = True, initial_archive_x: np.ndarray | None = None):
    set_seed(args.seed)
    if args.max_fe < args.archive_size:
        raise ValueError("max_fe must be at least as large as archive_size.")
    if (args.max_fe - args.archive_size) % args.k_eval != 0:
        raise ValueError("max_fe - archive_size must be divisible by k_eval.")

    problem = ZDTProblem(name="ZDT1", dim=args.dim)
    ref_point = REFERENCE_POINTS["ZDT1"][:2].astype(np.float32)

    pretrain_x, pretrain_y, surrogates = pre_train_kan_surrogate_for_problem(
        problem=problem,
        device=args.device,
        kan_steps=args.kan_steps,
        hidden_width=args.kan_hidden,
        grid=args.kan_grid,
        seed=args.seed,
    )
    print(f"Pre-trained KAN surrogate on ZDT1 with {pretrain_x.shape[0]} samples.")

    if initial_archive_x is None:
        archive_x = latin_hypercube_sample(
            lower=problem.lower,
            upper=problem.upper,
            n_samples=args.archive_size,
            dim=args.dim,
            seed=args.seed,
        )
    else:
        archive_x = np.asarray(initial_archive_x, dtype=np.float32).copy()
        if archive_x.shape != (args.archive_size, args.dim):
            raise ValueError("initial_archive_x must have shape (archive_size, dim).")
    archive_y = problem.evaluate(archive_x)

    true_evals = args.archive_size
    remaining_budget = args.max_fe - true_evals
    steps_to_run = remaining_budget // args.k_eval
    hv_history: list[float] = []
    reward_history: list[float] = []

    fronts, _ = fast_non_dominated_sort(archive_y)
    front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    initial_hv = hypervolume_2d(front, ref_point)
    hv_history.append(initial_hv)
    print(
        f"Init    | archive={archive_x.shape[0]} | "
        f"front0={front.shape[0]} | HV={initial_hv:.6f}"
    )

    for step in range(steps_to_run):
        offspring_x, offspring_pred = generate_nsga2_pseudo_front(
            archive_x=archive_x,
            problem=problem,
            surrogates=surrogates,
            device=args.device,
            n_offspring=args.offspring_size,
            sigma=args.mutation_sigma,
            surrogate_nsga_steps=args.surrogate_nsga_steps,
            predict_fn=predict_with_kan,
            generate_fn=generate_offspring,
        )
        archive_pred = predict_with_kan(surrogates, archive_x, args.device)
        offspring_sigma = estimate_uncertainty(
            archive_x=archive_x,
            archive_y=archive_y,
            archive_pred=archive_pred,
            offspring_x=offspring_x,
        ).astype(np.float32)

        selected_idx = select_nda(offspring_pred, offspring_sigma, args.k_eval)
        selected_x = offspring_x[selected_idx]
        selected_y = problem.evaluate(selected_x)
        reward_value = float(
            demo.DeepICClass.fpareto_improvement_reward(
                previous_front=archive_y,
                selected_objectives=selected_y,
            )
        )
        reward_history.append(reward_value)

        archive_x, archive_y = update_archive(
            archive_x=archive_x,
            archive_y=archive_y,
            new_x=selected_x,
            new_y=selected_y,
        )

        fronts, _ = fast_non_dominated_sort(archive_y)
        front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
        hv_value = hypervolume_2d(front, ref_point)
        hv_history.append(hv_value)

        print(
            f"Iter {step + 1:02d} | archive={archive_x.shape[0]} | "
            f"front0={front.shape[0]} | HV={hv_value:.6f} | reward={reward_value:.6f}"
        )

        true_evals += args.k_eval
        if true_evals >= args.max_fe:
            break

        combined_x = np.vstack([pretrain_x, archive_x])
        combined_y = np.vstack([pretrain_y, archive_y])
        surrogates = fit_kan_surrogates(
            archive_x=combined_x,
            archive_y=combined_y,
            device=args.device,
            kan_steps=args.kan_steps,
            hidden_width=args.kan_hidden,
            grid=args.kan_grid,
            seed=args.seed + 100 + step,
        )

    fronts, _ = fast_non_dominated_sort(archive_y)
    final_front = archive_y[np.asarray(fronts[0], dtype=np.int64)]

    true_f1 = np.linspace(0.0, 1.0, 200, dtype=np.float32)
    true_f2 = 1.0 - np.sqrt(true_f1)
    true_front = np.stack([true_f1, true_f2], axis=1)

    print("\nObtained Pareto front:")
    print(np.round(final_front, 6))
    print("\nTrue Pareto front:")
    print(np.round(true_front, 6))

    if plot:
        plt.figure(figsize=(7, 5))
        plt.title("ZDT1 Hypervolume Progress (KAN + ND-A)")
        plt.plot(hv_history, marker="o")
        plt.xlabel("Step")
        plt.ylabel("Hypervolume")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(7, 5))
        plt.title("ZDT1 Pareto Front (KAN + ND-A)")
        plt.scatter(final_front[:, 0], final_front[:, 1], s=24, c="blue", alpha=0.8, label="Obtained Front")
        plt.plot(true_front[:, 0], true_front[:, 1], "r-", linewidth=2, label="True Pareto Front")
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.legend()
        plt.grid(True)
        plt.show()

    _save_reward_log(
        "nsga_nda_zdt1_rewards.json",
        {
            "script": "nsga-nda.py",
            "mode": "run_nsga_nda",
            "problem_name": "ZDT1",
            "reward_history": reward_history,
            "hv_history": hv_history,
        },
    )

    return {
        "archive_x": archive_x,
        "archive_y": archive_y,
        "final_front": final_front,
        "true_front": true_front,
        "hv_history": hv_history,
        "reward_history": reward_history,
        "ref_point": ref_point,
    }


def run_comparison(args):
    model_path = "deepic_zdt1.pth"
    if os.path.exists(model_path):
        print(f"Using saved DeepIC model from {model_path}")
        deepic_model = None
    else:
        print("deepic_zdt1.pth not found. Training DeepIC on ZDT1 first...")
        deepic_model = demo.train_deepic_zdt1(args)

    shared_init_x = latin_hypercube_sample(
        lower=0.0,
        upper=1.0,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )
    deepic_result = demo.run_infer_zdt1(args, deepic=deepic_model, plot=False, initial_archive_x=shared_init_x)
    nsga_result = run_nsga_nda(args, plot=False, initial_archive_x=shared_init_x)

    print(f"\nDeepIC-assisted EA final HV: {deepic_result['hv_history'][-1]:.6f}")
    print(f"NSGA-NDA final HV: {nsga_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {deepic_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title("ZDT1 Hypervolume Comparison")
    plt.plot(deepic_result["hv_history"], marker="o", label="Pre-trained DeepIC-assisted EA")
    plt.plot(nsga_result["hv_history"], marker="s", label="NSGA-NDA")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.title("ZDT1 Pareto Front Comparison")
    plt.scatter(
        deepic_result["final_front"][:, 0],
        deepic_result["final_front"][:, 1],
        s=24,
        alpha=0.8,
        label="DeepIC-assisted EA",
    )
    plt.scatter(
        nsga_result["final_front"][:, 0],
        nsga_result["final_front"][:, 1],
        s=24,
        alpha=0.8,
        label="NSGA-NDA",
    )
    plt.plot(
        deepic_result["true_front"][:, 0],
        deepic_result["true_front"][:, 1],
        "k-",
        linewidth=2,
        label="True Pareto Front",
    )
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(True)
    plt.legend()
    plt.show()

    return {"deepic": deepic_result, "nsga_nda": nsga_result}


def main():
    args = parse_args()
    if args.compare:
        run_comparison(args)
    else:
        run_nsga_nda(args, plot=True)


if __name__ == "__main__":
    main()
