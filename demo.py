from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

import multisource_eva_common as multisource
from problem.kan import KAN
from optimizer.Surr_RLDE_Optimizer import SAEA


def load_deepic_class():
    deepic_path = Path(__file__).resolve().parent / "agent" / "deepic_agent.py"
    spec = importlib.util.spec_from_file_location("deepic_agent_local", deepic_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DeepIC


DeepICClass = load_deepic_class()


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


class ToyMultiObjectiveProblem:
    def __init__(self, dim: int, n_obj: int, lower: float = 0.0, upper: float = 1.0):
        self.dim = dim
        self.n_obj = n_obj
        self.lower = lower
        self.upper = upper
        centers = np.linspace(lower + 0.15, upper - 0.15, n_obj)
        self.centers = np.stack([np.full(dim, c, dtype=np.float32) for c in centers], axis=0)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        diff = x[:, None, :] - self.centers[None, :, :]
        sphere = np.mean(diff ** 2, axis=-1)
        ripple = 0.05 * np.sin(np.pi * x[:, None, :]).mean(axis=-1)
        return sphere + ripple


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


def fit_gp_surrogates(
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    seed: int,
) -> list[GaussianProcessRegressor]:
    archive_x = np.asarray(archive_x, dtype=np.float32)
    archive_y = np.asarray(archive_y, dtype=np.float32)
    models: list[GaussianProcessRegressor] = []

    for obj_id in range(archive_y.shape[1]):
        kernel = (
            ConstantKernel(constant_value=1.0, constant_value_bounds="fixed")
            * RBF(length_scale=0.5, length_scale_bounds="fixed")
            + WhiteKernel(noise_level=1e-5, noise_level_bounds="fixed")
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            optimizer=None,
            random_state=seed + obj_id,
        )
        model.fit(archive_x, archive_y[:, obj_id])
        models.append(model)

    return models


def predict_with_kan(models: Sequence[Any], x: np.ndarray, device: str) -> np.ndarray:
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    preds = []
    for model in models:
        with torch.no_grad():
            pred = model(x_tensor).detach().cpu().numpy().reshape(-1)
        preds.append(pred)
    return np.stack(preds, axis=1)


def predict_with_gp(models: Sequence[GaussianProcessRegressor], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float32)
    mean_preds: list[np.ndarray] = []
    std_preds: list[np.ndarray] = []

    for model in models:
        mean, std = model.predict(x_arr, return_std=True)
        mean_preds.append(np.asarray(mean, dtype=np.float32).reshape(-1))
        std_preds.append(np.asarray(std, dtype=np.float32).reshape(-1))

    mean_array = np.stack(mean_preds, axis=1).astype(np.float32)
    std_array = np.stack(std_preds, axis=1).astype(np.float32)
    return mean_array, std_array + 1e-6


def predict_with_gp_mean(models: Sequence[GaussianProcessRegressor], x: np.ndarray, device: str | None = None) -> np.ndarray:
    """Mean-only GP predictions (drop-in for `predict_with_kan` signature)."""
    x_arr = np.asarray(x, dtype=np.float32)
    mean_preds: list[np.ndarray] = []
    for model in models:
        mean = model.predict(x_arr)
        mean_preds.append(np.asarray(mean, dtype=np.float32).reshape(-1))
    return np.stack(mean_preds, axis=1).astype(np.float32)


def predict_with_gp_std(models: Sequence[GaussianProcessRegressor], x: np.ndarray) -> np.ndarray:
    """Std-only GP predictions (useful as DeepIC/ICW sigma input)."""
    x_arr = np.asarray(x, dtype=np.float32)
    std_preds: list[np.ndarray] = []
    for model in models:
        _, std = model.predict(x_arr, return_std=True)
        std_preds.append(np.asarray(std, dtype=np.float32).reshape(-1))
    return np.stack(std_preds, axis=1).astype(np.float32) + 1e-6


def surrogate_model_name(args) -> str:
    return getattr(args, "surrogate_model", getattr(args, "uncertainty_model", "gp"))


def build_uncertainty_models(
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    seed: int,
    surrogate_model: str = "gp",
) -> list[GaussianProcessRegressor] | None:
    if surrogate_model == "gp":
        return fit_gp_surrogates(archive_x=archive_x, archive_y=archive_y, seed=seed)
    if surrogate_model in {"knn", "kan"}:
        return None
    raise ValueError(f"Unsupported surrogate_model: {surrogate_model}")


def predict_offspring_sigma(
    kan_surrogates: Sequence[Any],
    offspring_x: np.ndarray,
    uncertainty_x: np.ndarray,
    uncertainty_y: np.ndarray,
    device: str,
    surrogate_model: str = "gp",
    gp_surrogates: Sequence[GaussianProcessRegressor] | None = None,
) -> np.ndarray:
    if surrogate_model == "gp":
        if gp_surrogates is None:
            raise ValueError("GP uncertainty requested but gp_surrogates is None.")
        _, offspring_sigma = predict_with_gp(gp_surrogates, offspring_x)
        return offspring_sigma.astype(np.float32)

    if surrogate_model not in {"knn", "kan"}:
        raise ValueError(f"Unsupported surrogate_model: {surrogate_model}")

    archive_pred = predict_with_kan(kan_surrogates, uncertainty_x, device).astype(np.float32)
    return estimate_uncertainty(
        archive_x=uncertainty_x,
        archive_y=uncertainty_y,
        archive_pred=archive_pred,
        offspring_x=offspring_x,
    ).astype(np.float32)


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


def init_uncertainty_archive(
    archive_x: np.ndarray,
    archive_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(archive_x, dtype=np.float32).copy(),
        np.asarray(archive_y, dtype=np.float32).copy(),
    )


def update_uncertainty_archive(
    uncertainty_x: np.ndarray,
    uncertainty_y: np.ndarray,
    new_x: np.ndarray,
    new_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    uncertainty_x = np.asarray(uncertainty_x, dtype=np.float32)
    uncertainty_y = np.asarray(uncertainty_y, dtype=np.float32)
    new_x = np.asarray(new_x, dtype=np.float32)
    new_y = np.asarray(new_y, dtype=np.float32)

    if new_x.size == 0 or new_y.size == 0:
        return uncertainty_x, uncertainty_y

    if uncertainty_x.size == 0:
        merged_x = new_x
        merged_y = new_y
    else:
        merged_x = np.vstack([uncertainty_x, new_x])
        merged_y = np.vstack([uncertainty_y, new_y])

    unique_indices = []
    for i in range(merged_x.shape[0]):
        is_duplicate = False
        for j in unique_indices:
            if np.allclose(merged_x[i], merged_x[j]) and np.allclose(merged_y[i], merged_y[j]):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_indices.append(i)

    unique_indices = np.asarray(unique_indices, dtype=np.int64)
    return merged_x[unique_indices], merged_y[unique_indices]


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


def crowding_distance(values: np.ndarray, front: list[int]) -> np.ndarray:
    if not front:
        return np.array([], dtype=np.float32)
    distance = np.zeros(len(front), dtype=np.float32)
    front_values = values[front]
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


def rank_candidates_against_archive(
    archive_y: np.ndarray,
    offspring_y: np.ndarray,
    offspring_sigma: np.ndarray,
) -> np.ndarray:
    penalized_offspring = offspring_y + offspring_sigma
    union = np.vstack([archive_y, penalized_offspring])
    fronts, ranks = fast_non_dominated_sort(union)

    crowding = np.zeros(union.shape[0], dtype=np.float32)
    for front in fronts:
        if len(front) > 0:
            crowding[np.asarray(front)] = crowding_distance(union, front)

    archive_n = archive_y.shape[0]
    offspring_idx = np.arange(archive_n, union.shape[0])
    sort_key = np.lexsort(
        (
            penalized_offspring.sum(axis=1),
            -crowding[offspring_idx],
            ranks[offspring_idx],
        )
    )
    return sort_key.astype(np.int64)


def update_archive(
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    new_x: np.ndarray,
    new_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Append new true-evaluated solutions to the archive.

    In inference/evaluation flows we keep *all* evaluated individuals (including
    dominated ones) instead of maintaining a Pareto-only archive.
    """
    archive_x = np.asarray(archive_x)
    archive_y = np.asarray(archive_y)
    new_x = np.asarray(new_x)
    new_y = np.asarray(new_y)

    if new_x.size == 0 or new_y.size == 0:
        return archive_x, archive_y

    if archive_x.size == 0:
        return new_x, new_y

    return np.vstack([archive_x, new_x]), np.vstack([archive_y, new_y])


def adapt_deepic(
    model: Any,
    optimizer: torch.optim.Optimizer,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    offspring_x: np.ndarray,
    offspring_pred: np.ndarray,
    offspring_sigma: np.ndarray,
    lower: float,
    upper: float,
    progress: float,
    target_ranking: np.ndarray | None,
    reward: float | np.ndarray | None,
    device: str,
    steps: int,
    top_k: int | None = None,
    reward_baseline: float = 0.0,
    reward_discount: float = 0.99,
) -> float:
    if steps <= 0:
        return 0.0

    x_true = torch.tensor(archive_x, dtype=torch.float32, device=device)
    y_true = torch.tensor(archive_y, dtype=torch.float32, device=device)
    x_sur = torch.tensor(offspring_x, dtype=torch.float32, device=device)
    y_sur = torch.tensor(offspring_pred, dtype=torch.float32, device=device)
    sigma_sur = torch.tensor(offspring_sigma, dtype=torch.float32, device=device)
    ranking = None
    if target_ranking is not None:
        ranking = torch.tensor(target_ranking[None, :], dtype=torch.long, device=device)

    reward_tensor = None
    if reward is not None:
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=device).reshape(-1)
        if reward_tensor.numel() == 1:
            reward_tensor = reward_tensor.repeat(1)

    last_loss = 0.0
    model.train()
    for _ in range(steps):
        out = model(
            x_true=x_true,
            y_true=y_true,
            x_sur=x_sur,
            y_sur=y_sur,
            sigma_sur=sigma_sur,
            progress=progress,
            lower_bound=lower,
            upper_bound=upper,
            target_ranking=ranking,
            decode_type="sample" if ranking is None else "greedy",
            max_decode_steps=top_k,
        )

        if reward_tensor is not None and ranking is not None:
            log_prob = model.sequence_log_prob(out["logits"], ranking, top_k=top_k)
            advantage = reward_tensor * float(reward_discount) - float(reward_baseline)
            loss = -(advantage.detach() * log_prob).mean()
        elif ranking is not None:
            loss = model.ranking_loss(out["logits"], ranking)
        else:
            raise ValueError("adapt_deepic requires target_ranking when reward is not provided.")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    return last_loss


def infer_deepic_ranking(
    model: Any,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    offspring_x: np.ndarray,
    offspring_pred: np.ndarray,
    offspring_sigma: np.ndarray,
    lower: float,
    upper: float,
    progress: float,
    device: str,
    top_k: int | None = None,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(
            x_true=torch.tensor(archive_x, dtype=torch.float32, device=device),
            y_true=torch.tensor(archive_y, dtype=torch.float32, device=device),
            x_sur=torch.tensor(offspring_x, dtype=torch.float32, device=device),
            y_sur=torch.tensor(offspring_pred, dtype=torch.float32, device=device),
            sigma_sur=torch.tensor(offspring_sigma, dtype=torch.float32, device=device),
            progress=progress,
            lower_bound=lower,
            upper_bound=upper,
            decode_type="sample",  # Sample from learned probability distribution
            max_decode_steps=top_k,
        )
    return out["ranking"][0].detach().cpu().numpy()


class KanSurrogateWrapper:
    def __init__(self, models, device: str = "cpu"):
        self.models = models
        self.device = device

    def predict(self, x: np.ndarray) -> np.ndarray:
        return predict_with_kan(self.models, x, self.device).astype(np.float32)


class DeepICSAEAAgent:
    """Agent wrapper to connect DeepIC ranking with SAEA selection."""

    def __init__(self, model, optimizer, surrogate_model=None, device: str = "cpu"):
        self.model = model
        self.optimizer = optimizer
        self.surrogate_model = surrogate_model
        self.device = device

    def select(self, offspring_x, offspring_y_pred, k, archive_x, archive_y, lower, upper, progress):
        offspring_sigma = estimate_uncertainty(
            archive_x=archive_x,
            archive_y=archive_y,
            archive_pred=self.predict_archive(archive_x),
            offspring_x=offspring_x,
        ).astype(np.float32)

        ranking = infer_deepic_ranking(
            model=self.model,
            archive_x=archive_x,
            archive_y=archive_y,
            offspring_x=offspring_x,
            offspring_pred=offspring_y_pred,
            offspring_sigma=offspring_sigma,
            lower=lower,
            upper=upper,
            progress=progress,
            device=self.device,
            top_k=k,
        )

        self.last_offspring_sigma = offspring_sigma
        self.last_offspring_pred = offspring_y_pred
        self.last_offspring_x = offspring_x
        self.last_archive_x = archive_x
        self.last_archive_y = archive_y
        self.last_progress = progress

        return ranking[:k]

    def predict_archive(self, archive_x):
        if self.surrogate_model is None:
            raise ValueError('Surrogate model not set for DeepICSAEAAgent')
        return self.surrogate_model.predict(archive_x)

    def adapt(self, steps=8, reward=None, top_k=None):
        if not hasattr(self, 'last_offspring_x'):
            return 0.0

        return adapt_deepic(
            model=self.model,
            optimizer=self.optimizer,
            archive_x=self.last_archive_x,
            archive_y=self.last_archive_y,
            offspring_x=self.last_offspring_x,
            offspring_pred=self.last_offspring_pred,
            offspring_sigma=self.last_offspring_sigma,
            lower=self.last_archive_x.min(axis=0),
            upper=self.last_archive_x.max(axis=0),
            progress=self.last_progress,
            target_ranking=np.arange(len(self.last_offspring_x), dtype=np.int64),
            reward=reward,
            device=self.device,
            steps=steps,
            top_k=top_k,
            reward_discount=0.99,
        )


class ReplayBuffer:
    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def add(self, item):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(item)

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in idx]


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
        assert x.ndim == 2
        f1 = x[:, 0]
        g = 1.0 + 9.0 / (self.dim - 1.0) * np.sum(x[:, 1:], axis=1)

        if self.name == 'ZDT1':
            h = 1.0 - np.sqrt(f1 / g)
        elif self.name == 'ZDT2':
            h = 1.0 - (f1 / g) ** 2
        elif self.name == 'ZDT3':
            h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)
        elif self.name == 'ZDT4':
            g = 1.0 + 10.0 * (self.dim - 1) + np.sum(100.0 * x[:, 1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[:, 1:]), axis=1)
            h = 1.0 - np.sqrt(f1 / g)
        elif self.name == 'ZDT5':
            # approximated with ZDT1-like variant since ZDT5 is discrete binary in original
            h = 1.0 - np.sqrt(f1 / g)
        elif self.name == 'ZDT6':
            f1 = 1.0 - np.exp(-4.0 * f1) * (np.sin(6.0 * np.pi * f1) ** 6)
            g = 1.0 + 9.0 * (np.sum(x[:, 1:], axis=1) / (self.dim - 1.0)) ** 0.25
            h = 1.0 - (f1 / g) ** 2
        elif self.name == 'ZDT7':
            g = 1.0 + 10.0 * np.sum(x[:, 1:], axis=1) / (self.dim - 1.0)
            h = 1.0 - (f1 / g) * (1.0 + np.sin(3.0 * np.pi * f1))
        else:
            raise ValueError(f'Unknown ZDT problem: {self.name}')

        f2 = g * h
        return np.stack([f1, f2], axis=1)


def train_deepic_with_saea(args):
    problem = ToyMultiObjectiveProblem(dim=args.dim, n_obj=args.n_obj, lower=0.0, upper=1.0)
    saea = SAEA(
        config=args,
        surrogate_model=KanSurrogateWrapper([], device=args.device),
        agent=None,
        archive_size=args.archive_size,
        offspring_size=args.offspring_size,
        k_real=args.k_eval,
        generation_strategy='NSGA-III',
    )

    archive_state = saea.init_population(problem)
    archive_x = archive_state['X']
    archive_y = archive_state['y']

    deepic = DeepICClass(
        hidden_dim=args.deepic_hidden,
        n_heads=args.deepic_heads,
        ff_dim=args.deepic_ff,
    ).to(args.device)
    deepic_optimizer = torch.optim.Adam(deepic.parameters(), lr=args.deepic_lr)
    deepic_agent = DeepICSAEAAgent(deepic, deepic_optimizer, device=args.device)

    true_evals_consumed = archive_x.shape[0]
    max_fe = args.max_fe if args.max_fe is not None else args.archive_size + args.iterations * args.k_eval

    for iteration in range(args.iterations):
        # Fit surrogate models on current archive
        models = fit_kan_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            device=args.device,
            kan_steps=args.kan_steps,
            hidden_width=args.kan_hidden,
            grid=args.kan_grid,
            seed=args.seed + iteration * 13,
        )
        surrogate_wrapper = KanSurrogateWrapper(models, device=args.device)
        saea.surrogate_model = surrogate_wrapper

        # helper for deepic agent via saea output
        deepic_agent.surrogate_model = surrogate_wrapper
        saea.agent = deepic_agent

        # Perform one SAEA iteration
        prev_archive_y = archive_y.copy()
        result = saea.update(action=None, problem=problem)
        archive_x = saea.archive_X
        archive_y = saea.archive_y

        # Use Pareto improvement reward based on the new solutions added to the archive.
        added_points = []
        for candidate in archive_y:
            if not np.any(np.all(np.isclose(prev_archive_y, candidate), axis=1)):
                added_points.append(candidate)
        if added_points:
            selected_y = np.asarray(added_points, dtype=np.float32)
            reward = DeepICClass.fpareto_improvement_reward(
                previous_front=prev_archive_y,
                selected_objectives=selected_y,
            )
        else:
            reward = -1.0

        deepic_agent.adapt(steps=args.deepic_adapt_steps, reward=reward, top_k=args.k_eval)

        true_evals_consumed += args.k_eval

        progress_ratio = min(true_evals_consumed / max(max_fe, 1), 1.0)
        fronts, _ = fast_non_dominated_sort(archive_y)
        print(
            f"SAEA-DeepIC Iter {iteration + 1}/{args.iterations} | progress={progress_ratio:.3f} | "
            f"front0={len(fronts[0])} | reward={reward:.6f}"
        )

    return archive_x, archive_y


def hypervolume_2d(pareto: np.ndarray, ref: np.ndarray) -> float:
    """Compute hypervolume using pymoo's HyperVolume indicator with FIXED reference point.
    
    Args:
        pareto: Array of objectives (n_solutions, n_objectives)
        ref: FIXED reference point (must be provided and worse than all solutions)
    
    Returns:
        Hypervolume value
        
    Note:
        HV is computed on the non-dominated front (pass a Pareto-front array),
        so with a fixed reference point it should monotonically increase or stay
        the same during optimization.
    """
    if pareto.size == 0 or len(pareto) == 0:
        return 0.0
    
    pareto = np.asarray(pareto)
    ref = np.asarray(ref, dtype=np.float32)
    hv = HV(ref_point=ref)
    return float(hv(pareto))


def infer_zdt7(args, deepic=None):
    if deepic is None:
        deepic = DeepICClass(
            hidden_dim=args.deepic_hidden,
            n_heads=args.deepic_heads,
            ff_dim=args.deepic_ff,
        ).to(args.device)
        if os.path.exists('deepic_zdt.pth'):
            deepic.load_state_dict(torch.load('deepic_zdt.pth', map_location=args.device))
            print("Loaded DeepIC model from deepic_zdt.pth")
        else:
            print("No saved DeepIC model found, using untrained model")
    
    problem = ZDTProblem(name='ZDT7', dim=args.dim)
    pretrain_x, pretrain_y, surrogates = pre_train_kan_surrogate_for_problem(
        args=args,
        problem_name='ZDT7',
    )
    print(f"Pre-trained KAN surrogate on ZDT7 with {pretrain_x.shape[0]} samples.")

    archive_x = latin_hypercube_sample(
        lower=problem.lower,
        upper=problem.upper,
        n_samples=80,
        dim=args.dim,
        seed=args.seed,
    )
    archive_y = problem.evaluate(archive_x)
    uncertainty_x, uncertainty_y = init_uncertainty_archive(archive_x, archive_y)
    gp_surrogates = build_uncertainty_models(
        archive_x=uncertainty_x,
        archive_y=uncertainty_y,
        seed=args.seed + 31,
        surrogate_model=surrogate_model_name(args),
    )
    
    # Set FIXED reference point for ZDT7 (from literature)
    # ZDT7: f1 ∈ [0,1], f2 depends on dimension and problem structure
    # Standard reference point: [1.1, 11.0]
    # This ensures the reference point is worse than all Pareto-optimal solutions
    ref_point = np.array([1.0, 11.0], dtype=np.float32)
    
    true_evals = 80
    max_fe = 160
    remaining_budget = max_fe - true_evals
    steps_to_run = remaining_budget // args.k_eval

    hv_history = []
    fronts, _ = fast_non_dominated_sort(archive_y)
    front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    initial_hv = hypervolume_2d(front, ref_point)
    hv_history.append(initial_hv)
    print(f"Init    | archive={archive_x.shape[0]} | front0={front.shape[0]} | Hypervolume: {initial_hv:.6f}")
    for step in range(steps_to_run):
        offspring_x = generate_offspring(
            archive_x=archive_x,
            n_offspring=args.offspring_size,
            lower=problem.lower,
            upper=problem.upper,
            sigma=args.mutation_sigma,
        )
        offspring_pred = predict_with_kan(surrogates, offspring_x, args.device).astype(np.float32)
        offspring_sigma = predict_offspring_sigma(
            kan_surrogates=surrogates,
            offspring_x=offspring_x,
            uncertainty_x=uncertainty_x,
            uncertainty_y=uncertainty_y,
            device=args.device,
            surrogate_model=surrogate_model_name(args),
            gp_surrogates=gp_surrogates,
        )

        progress = float(true_evals / max_fe)
        ranking = infer_deepic_ranking(
            model=deepic,
            archive_x=archive_x,
            archive_y=archive_y,
            offspring_x=offspring_x,
            offspring_pred=offspring_pred,
            offspring_sigma=offspring_sigma,
            lower=problem.lower,
            upper=problem.upper,
            progress=progress,
            device=args.device,
            top_k=args.k_eval,
        )

        selected_idx = ranking[: args.k_eval]
        selected_x = offspring_x[selected_idx]
        selected_y = problem.evaluate(selected_x)

        archive_x, archive_y = update_archive(
            archive_x=archive_x,
            archive_y=archive_y,
            new_x=selected_x,
            new_y=selected_y,
        )
        uncertainty_x, uncertainty_y = update_uncertainty_archive(
            uncertainty_x=uncertainty_x,
            uncertainty_y=uncertainty_y,
            new_x=selected_x,
            new_y=selected_y,
        )

        fronts, _ = fast_non_dominated_sort(archive_y)
        front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
        hv_history.append(hypervolume_2d(front, ref_point))

        print(f"Number of individuals in archive: {archive_x.shape[0]}, Hypervolume: {hypervolume_2d(front, ref_point):.6f}")

        true_evals += args.k_eval
        if true_evals >= max_fe:
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
    print("ZDT7 final Pareto front (first 10 points):")
    print(np.round(final_front[:10], 6))

    # Compute true Pareto front for ZDT7
    true_f1 = np.linspace(0, 1, 100)
    true_f2 = 1.0 - true_f1 * (1.0 + np.sin(3.0 * np.pi * true_f1))

    plt.figure(figsize=(7, 5))
    plt.title('ZDT7 Hypervolume Progress')
    plt.plot(hv_history, marker='o')
    plt.xlabel('Step')
    plt.ylabel('Hypervolume')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.title('ZDT7 Pareto Front')
    plt.scatter(final_front[:, 0], final_front[:, 1], s=20, c='blue', alpha=0.8, label='Obtained Front')
    plt.plot(true_f1, true_f2, 'r-', linewidth=2, label='True Pareto Front')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.legend()
    plt.grid(True)
    plt.show()

    return archive_x, archive_y


def run_infer_zdt1(args, deepic=None, plot: bool = True, initial_archive_x: np.ndarray | None = None):
    if deepic is None:
        deepic = DeepICClass(
            hidden_dim=args.deepic_hidden,
            n_heads=args.deepic_heads,
            ff_dim=args.deepic_ff,
        ).to(args.device)
        if os.path.exists('deepic_zdt1.pth'):
            deepic.load_state_dict(torch.load('deepic_zdt1.pth', map_location=args.device))
            print("Loaded DeepIC model from deepic_zdt1.pth")
        else:
            print("No saved DeepIC model found, using untrained model")
    
    problem = ZDTProblem(name='ZDT1', dim=args.dim)
    pretrain_x, pretrain_y, surrogates = pre_train_kan_surrogate_for_problem(
        args=args,
        problem_name='ZDT1',
    )
    print(f"Pre-trained KAN surrogate on ZDT1 with {pretrain_x.shape[0]} samples.")
    if initial_archive_x is None:
        archive_x = latin_hypercube_sample(
            lower=problem.lower,
            upper=problem.upper,
            n_samples=80,
            dim=args.dim,
            seed=args.seed,
        )
    else:
        archive_x = np.asarray(initial_archive_x, dtype=np.float32).copy()
        if archive_x.shape != (80, args.dim):
            raise ValueError("initial_archive_x must have shape (80, dim).")
    archive_y = problem.evaluate(archive_x)
    uncertainty_x, uncertainty_y = init_uncertainty_archive(archive_x, archive_y)
    gp_surrogates = build_uncertainty_models(
        archive_x=uncertainty_x,
        archive_y=uncertainty_y,
        seed=args.seed + 37,
        surrogate_model=surrogate_model_name(args),
    )
    
    ref_point = REFERENCE_POINTS["ZDT1"][:2].astype(np.float32)
    
    true_evals = 80
    max_fe = 160
    remaining_budget = max_fe - true_evals
    steps_to_run = remaining_budget // args.k_eval

    hv_history = []
    fronts, _ = fast_non_dominated_sort(archive_y)
    front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    initial_hv = hypervolume_2d(front, ref_point)
    hv_history.append(initial_hv)
    print(f"Init    | archive={archive_x.shape[0]} | front0={front.shape[0]} | Hypervolume: {initial_hv:.6f}")
    for step in range(steps_to_run):
        surrogates = fit_kan_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            device=args.device,
            kan_steps=args.kan_steps,
            hidden_width=args.kan_hidden,
            grid=args.kan_grid,
            seed=args.seed + step,
        )

        offspring_x = generate_offspring(
            archive_x=archive_x,
            n_offspring=args.offspring_size,
            lower=problem.lower,
            upper=problem.upper,
            sigma=args.mutation_sigma,
        )
        offspring_pred = predict_with_kan(surrogates, offspring_x, args.device).astype(np.float32)
        offspring_sigma = predict_offspring_sigma(
            kan_surrogates=surrogates,
            offspring_x=offspring_x,
            uncertainty_x=uncertainty_x,
            uncertainty_y=uncertainty_y,
            device=args.device,
            surrogate_model=surrogate_model_name(args),
            gp_surrogates=gp_surrogates,
        )

        progress = float(true_evals / max_fe)
        ranking = infer_deepic_ranking(
            model=deepic,
            archive_x=archive_x,
            archive_y=archive_y,
            offspring_x=offspring_x,
            offspring_pred=offspring_pred,
            offspring_sigma=offspring_sigma,
            lower=problem.lower,
            upper=problem.upper,
            progress=progress,
            device=args.device,
            top_k=args.k_eval,
        )

        selected_idx = ranking[: args.k_eval]
        selected_x = offspring_x[selected_idx]
        selected_y = problem.evaluate(selected_x)

        archive_x, archive_y = update_archive(
            archive_x=archive_x,
            archive_y=archive_y,
            new_x=selected_x,
            new_y=selected_y,
        )
        uncertainty_x, uncertainty_y = update_uncertainty_archive(
            uncertainty_x=uncertainty_x,
            uncertainty_y=uncertainty_y,
            new_x=selected_x,
            new_y=selected_y,
        )

        fronts, _ = fast_non_dominated_sort(archive_y)
        front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
        hv_history.append(hypervolume_2d(front, ref_point))

        print(f"Number of individuals in archive: {archive_x.shape[0]}, Hypervolume: {hypervolume_2d(front, ref_point):.6f}")

        true_evals += args.k_eval
        if true_evals >= max_fe:
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
            seed=args.seed + 200 + step,
        )

    fronts, _ = fast_non_dominated_sort(archive_y)
    final_front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    print("ZDT1 final Pareto front (first 10 points):")
    print(np.round(final_front[:10], 6))

    # Compute true Pareto front for ZDT1
    true_f1 = np.linspace(0, 1, 100)
    g = 1 + 9 * np.ones_like(true_f1)  # g=1 on the Pareto front
    true_f2 = 1 - np.sqrt(true_f1)

    if plot:
        plt.figure(figsize=(7, 5))
        plt.title('ZDT1 Hypervolume Progress')
        plt.plot(hv_history, marker='o')
        plt.xlabel('Step')
        plt.ylabel('Hypervolume')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(7, 5))
        plt.title('ZDT1 Pareto Front')
        plt.scatter(final_front[:, 0], final_front[:, 1], s=20, c='blue', alpha=0.8, label='Obtained Front')
        plt.plot(true_f1, true_f2, 'r-', linewidth=2, label='True Pareto Front')
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        "archive_x": archive_x,
        "archive_y": archive_y,
        "final_front": final_front,
        "true_front": np.stack([true_f1, true_f2], axis=1),
        "hv_history": hv_history,
        "ref_point": ref_point,
    }


def infer_zdt1(args, deepic=None):
    result = run_infer_zdt1(args, deepic=deepic, plot=True)
    return result["archive_x"], result["archive_y"]


def pre_train_kan_surrogates(args, problems):
    pre_trained = {}
    for p_name in problems:
        problem = ZDTProblem(name=p_name, dim=args.dim)
        # Generate pre-training dataset
        n_samples = 1000
        x_data = np.random.uniform(problem.lower, problem.upper, size=(n_samples, args.dim)).astype(np.float32)
        y_data = problem.evaluate(x_data)
        
        models = fit_kan_surrogates(
            archive_x=x_data,
            archive_y=y_data,
            device=args.device,
            kan_steps=args.kan_steps * 4,  # More steps for pre-training
            hidden_width=args.kan_hidden,
            grid=args.kan_grid,
            seed=args.seed,
        )
        pre_trained[p_name] = models
    return pre_trained


def pre_train_kan_surrogate_for_problem(
    args,
    problem_name: str,
    n_samples: int = 1000,
    step_multiplier: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[KAN]]:
    problem = ZDTProblem(name=problem_name, dim=args.dim)
    x_data = np.random.uniform(problem.lower, problem.upper, size=(n_samples, args.dim)).astype(np.float32)
    y_data = problem.evaluate(x_data)

    models = fit_kan_surrogates(
        archive_x=x_data,
        archive_y=y_data,
        device=args.device,
        kan_steps=args.kan_steps * step_multiplier,
        hidden_width=args.kan_hidden,
        grid=args.kan_grid,
        seed=args.seed,
    )
    return x_data, y_data, models


def train_deepic_zdt(args):
    problems = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT5", "ZDT6"]
    
    # Pre-train KAN surrogates on ZDT1-6
    print("Pre-training KAN surrogates on ZDT1-6...")
    pre_trained_surrogates = pre_train_kan_surrogates(args, problems)
    print("Pre-training completed.")
    
    replay = ReplayBuffer(capacity=256)

    deepic = DeepICClass(
        hidden_dim=args.deepic_hidden,
        n_heads=args.deepic_heads,
        ff_dim=args.deepic_ff,
    ).to(args.device)
    deepic_optimizer = torch.optim.Adam(deepic.parameters(), lr=1e-3)

    for epoch in range(50):
        print(f"Epoch {epoch+1}/50")

        for p_name in problems:
            problem = ZDTProblem(name=p_name, dim=args.dim)
            archive_x = np.random.uniform(problem.lower, problem.upper, size=(80, args.dim)).astype(np.float32)
            archive_y = problem.evaluate(archive_x)
            uncertainty_x, uncertainty_y = init_uncertainty_archive(archive_x, archive_y)
            gp_surrogates = build_uncertainty_models(
                archive_x=uncertainty_x,
                archive_y=uncertainty_y,
                seed=args.seed + epoch * 1000 + sum(ord(ch) for ch in p_name),
                surrogate_model=surrogate_model_name(args),
            )
            true_evals = 80
            max_fe = 160
            remaining_budget = 80
            steps_to_run = remaining_budget // args.k_eval

            # Use pre-trained surrogates
            surrogates = pre_trained_surrogates[p_name]

            for step in range(steps_to_run):
                offspring_x = generate_offspring(
                    archive_x=archive_x,
                    n_offspring=args.offspring_size,
                    lower=problem.lower,
                    upper=problem.upper,
                    sigma=args.mutation_sigma,
                )
                offspring_pred = predict_with_kan(surrogates, offspring_x, args.device).astype(np.float32)
                offspring_sigma = predict_offspring_sigma(
                    kan_surrogates=surrogates,
                    offspring_x=offspring_x,
                    uncertainty_x=uncertainty_x,
                    uncertainty_y=uncertainty_y,
                    device=args.device,
                    surrogate_model=surrogate_model_name(args),
                    gp_surrogates=gp_surrogates,
                )

                progress = float(true_evals / max_fe)
                ranking = infer_deepic_ranking(
                    model=deepic,
                    archive_x=archive_x,
                    archive_y=archive_y,
                    offspring_x=offspring_x,
                    offspring_pred=offspring_pred,
                    offspring_sigma=offspring_sigma,
                    lower=problem.lower,
                    upper=problem.upper,
                    progress=progress,
                    device=args.device,
                    top_k=args.k_eval,
                )

                selected_idx = ranking[: args.k_eval]
                selected_x = offspring_x[selected_idx]
                selected_y = problem.evaluate(selected_x)

                previous_archive_y = archive_y.copy()
                archive_x, archive_y = update_archive(
                    archive_x=archive_x,
                    archive_y=archive_y,
                    new_x=selected_x,
                    new_y=selected_y,
                )
                uncertainty_x, uncertainty_y = update_uncertainty_archive(
                    uncertainty_x=uncertainty_x,
                    uncertainty_y=uncertainty_y,
                    new_x=selected_x,
                    new_y=selected_y,
                )
                reward = DeepICClass.fpareto_improvement_reward(
                    previous_front=previous_archive_y,
                    selected_objectives=selected_y,
                )

                replay.add(
                    {
                        "archive_x": archive_x,
                        "archive_y": archive_y,
                        "offspring_x": offspring_x,
                        "offspring_pred": offspring_pred,
                        "offspring_sigma": offspring_sigma,
                        "ranking": ranking,
                        "reward": reward,
                        "progress": progress,
                    }
                )

                if len(replay) >= 32:
                    batch = replay.sample(32)
                    for sample in batch:
                        target_ranking = sample["ranking"]  # Use full surrogate ordering here
                        adapt_deepic(
                            model=deepic,
                            optimizer=deepic_optimizer,
                            archive_x=sample["archive_x"],
                            archive_y=sample["archive_y"],
                            offspring_x=sample["offspring_x"],
                            offspring_pred=sample["offspring_pred"],
                            offspring_sigma=sample["offspring_sigma"],
                            lower=problem.lower,
                            upper=problem.upper,
                            progress=sample["progress"],
                            target_ranking=target_ranking,
                            reward=sample["reward"],
                            device=args.device,
                            steps=1,
                            top_k=args.k_eval,
                            reward_discount=args.discount,
                        )

                true_evals += args.k_eval

                if true_evals >= max_fe:
                    break

            print(
                f"{p_name} epoch {epoch+1} done, true_evals={true_evals}, best_obj1={np.min(archive_y[:,0])}\n"
            )
        if (epoch + 1) % 5 == 0:
            multisource.save_colab_model_checkpoint(
                deepic.state_dict(),
                f"deepic_zdt_epoch_{epoch + 1}.pth",
            )

    # Save the trained DeepIC model
    torch.save(deepic.state_dict(), 'deepic_zdt.pth')
    print("DeepIC model saved to deepic_zdt.pth")
    return deepic


def train_deepic_zdt1(args):
    print("Pre-training KAN surrogate on ZDT1...")
    pretrain_x, pretrain_y, surrogates = pre_train_kan_surrogate_for_problem(
        args=args,
        problem_name='ZDT1',
    )
    print("Pre-training completed.")

    replay = ReplayBuffer(capacity=256)

    deepic = DeepICClass(
        hidden_dim=args.deepic_hidden,
        n_heads=args.deepic_heads,
        ff_dim=args.deepic_ff,
    ).to(args.device)
    deepic_optimizer = torch.optim.Adam(deepic.parameters(), lr=1e-3)

    for epoch in range(50):
        print(f"ZDT1 Epoch {epoch + 1}/50")

        problem = ZDTProblem(name='ZDT1', dim=args.dim)
        archive_x = np.random.uniform(problem.lower, problem.upper, size=(80, args.dim)).astype(np.float32)
        archive_y = problem.evaluate(archive_x)
        uncertainty_x, uncertainty_y = init_uncertainty_archive(archive_x, archive_y)
        gp_surrogates = build_uncertainty_models(
            archive_x=uncertainty_x,
            archive_y=uncertainty_y,
            seed=args.seed + epoch * 1000 + 101,
            surrogate_model=surrogate_model_name(args),
        )
        true_evals = 80
        max_fe = 160
        remaining_budget = 80
        steps_to_run = remaining_budget // args.k_eval

        for step in range(steps_to_run):
            offspring_x = generate_offspring(
                archive_x=archive_x,
                n_offspring=args.offspring_size,
                lower=problem.lower,
                upper=problem.upper,
                sigma=args.mutation_sigma,
            )
            offspring_pred = predict_with_kan(surrogates, offspring_x, args.device).astype(np.float32)
            offspring_sigma = predict_offspring_sigma(
                kan_surrogates=surrogates,
                offspring_x=offspring_x,
                uncertainty_x=uncertainty_x,
                uncertainty_y=uncertainty_y,
                device=args.device,
                surrogate_model=surrogate_model_name(args),
                gp_surrogates=gp_surrogates,
            )

            progress = float(true_evals / max_fe)
            ranking = infer_deepic_ranking(
                model=deepic,
                archive_x=archive_x,
                archive_y=archive_y,
                offspring_x=offspring_x,
                offspring_pred=offspring_pred,
                offspring_sigma=offspring_sigma,
                lower=problem.lower,
                upper=problem.upper,
                progress=progress,
                device=args.device,
                top_k=args.k_eval,
            )

            selected_idx = ranking[: args.k_eval]
            selected_x = offspring_x[selected_idx]
            selected_y = problem.evaluate(selected_x)

            previous_archive_y = archive_y.copy()
            archive_x, archive_y = update_archive(
                archive_x=archive_x,
                archive_y=archive_y,
                new_x=selected_x,
                new_y=selected_y,
            )
            uncertainty_x, uncertainty_y = update_uncertainty_archive(
                uncertainty_x=uncertainty_x,
                uncertainty_y=uncertainty_y,
                new_x=selected_x,
                new_y=selected_y,
            )
            reward = DeepICClass.fpareto_improvement_reward(
                previous_front=previous_archive_y,
                selected_objectives=selected_y,
            )

            replay.add(
                {
                    "archive_x": archive_x,
                    "archive_y": archive_y,
                    "offspring_x": offspring_x,
                    "offspring_pred": offspring_pred,
                    "offspring_sigma": offspring_sigma,
                    "ranking": ranking,
                    "reward": reward,
                    "progress": progress,
                }
            )

            if len(replay) >= 32:
                batch = replay.sample(32)
                for sample in batch:
                    adapt_deepic(
                        model=deepic,
                        optimizer=deepic_optimizer,
                        archive_x=sample["archive_x"],
                        archive_y=sample["archive_y"],
                        offspring_x=sample["offspring_x"],
                        offspring_pred=sample["offspring_pred"],
                        offspring_sigma=sample["offspring_sigma"],
                        lower=problem.lower,
                        upper=problem.upper,
                        progress=sample["progress"],
                        target_ranking=sample["ranking"],
                        reward=sample["reward"],
                        device=args.device,
                        steps=1,
                        top_k=args.k_eval,
                        reward_discount=args.discount,
                    )

            true_evals += args.k_eval
            if true_evals >= max_fe:
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
                seed=args.seed + epoch * 100 + step,
            )

        print(
            f"ZDT1 epoch {epoch + 1} done, true_evals={true_evals}, best_obj1={np.min(archive_y[:, 0])}\n"
        )
        if (epoch + 1) % 5 == 0:
            multisource.save_colab_model_checkpoint(
                deepic.state_dict(),
                f"deepic_zdt1_epoch_{epoch + 1}.pth",
            )

    torch.save(deepic.state_dict(), 'deepic_zdt1.pth')
    print("DeepIC model saved to deepic_zdt1.pth")
    return deepic


def parse_args():
    parser = argparse.ArgumentParser(description="Demo of DeepIC + KAN-assisted archive optimization")
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--n_obj", type=int, default=2)
    parser.add_argument("--archive_size", type=int, default=80)
    parser.add_argument("--offspring_size", type=int, default=24)
    parser.add_argument("--k_eval", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--mutation_sigma", type=float, default=0.12)
    parser.add_argument("--kan_steps", type=int, default=25)
    parser.add_argument("--kan_hidden", type=int, default=10)
    parser.add_argument("--kan_grid", type=int, default=5)
    parser.add_argument("--deepic_hidden", type=int, default=64)
    parser.add_argument("--deepic_heads", type=int, default=4)
    parser.add_argument("--deepic_ff", type=int, default=128)
    parser.add_argument("--deepic_lr", type=float, default=1e-4)
    parser.add_argument("--deepic_adapt_steps", type=int, default=8)
    parser.add_argument("--max_fe", type=int, default=120)
    parser.add_argument("--discount", type=float, default=0.99, help="Reward discount/multiplier used during RL updates")
    parser.add_argument(
        "--surrogate_model",
        dest="surrogate_model",
        type=str,
        default="gp",
        choices=["gp", "knn", "kan"],
        help="Surrogate mode: gp=use GP for surrogate predictions/uncertainty, knn=KAN preds + KNN residual uncertainty, kan=same as knn (explicit KAN).",
    )
    parser.add_argument(
        "--uncertainty_model",
        dest="surrogate_model",
        type=str,
        choices=["gp", "knn"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_saea", action='store_true', help="Use SAEA + DeepIC training loop")
    parser.add_argument("--run_zdt", action='store_true', help="Run ZDT1/2/3 DeepIC training regime")
    parser.add_argument("--run_zdt1", action='store_true', help="Train DeepIC on ZDT1 for 50 epochs, then infer on ZDT1")
    parser.add_argument("--train_only", action='store_true', help="Train DeepIC on ZDT1-6 and save model")
    parser.add_argument("--train_zdt1_only", action='store_true', help="Train DeepIC on ZDT1 only and save model")
    parser.add_argument("--infer_only", action='store_true', help="Load trained DeepIC and run inference on specified ZDT problem")
    parser.add_argument("--infer_problem", type=str, default='ZDT7', choices=['ZDT1', 'ZDT7'], help="Which ZDT problem to infer (default: ZDT7)")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    problem = ToyMultiObjectiveProblem(dim=args.dim, n_obj=args.n_obj, lower=0.0, upper=1.0)
    archive_x = np.random.uniform(problem.lower, problem.upper, size=(args.archive_size, args.dim)).astype(np.float32)
    archive_y = problem.evaluate(archive_x).astype(np.float32)
    uncertainty_x, uncertainty_y = init_uncertainty_archive(archive_x, archive_y)
    gp_surrogates = build_uncertainty_models(
        archive_x=uncertainty_x,
        archive_y=uncertainty_y,
        seed=args.seed + 19,
        surrogate_model=surrogate_model_name(args),
    )
    true_evals_consumed = archive_x.shape[0]
    max_fe = args.max_fe if args.max_fe is not None else args.archive_size + args.iterations * args.k_eval

    deepic = DeepICClass(
        hidden_dim=args.deepic_hidden,
        n_heads=args.deepic_heads,
        ff_dim=args.deepic_ff,
    ).to(args.device)
    deepic_optimizer = torch.optim.Adam(deepic.parameters(), lr=args.deepic_lr)

    if args.run_zdt:
        deepic = train_deepic_zdt(args)
        print("\nDeepIC training completed on ZDT1..ZDT6 distribution in 50 epochs.")
        print("Now running ZDT7 inference and plotting results...")
        infer_zdt7(args, deepic)
        return

    if args.run_zdt1:
        deepic = train_deepic_zdt1(args)
        print("\nDeepIC training completed on ZDT1 in 50 epochs.")
        print("Now running ZDT1 inference and plotting results...")
        infer_zdt1(args, deepic)
        return

    if args.train_only:
        train_deepic_zdt(args)
        return

    if args.train_zdt1_only:
        train_deepic_zdt1(args)
        return

    if args.infer_only:
        if args.infer_problem == 'ZDT1':
            infer_zdt1(args)
        elif args.infer_problem == 'ZDT7':
            infer_zdt7(args)
        else:
            print(f"Unknown problem: {args.infer_problem}")
        return

    if args.use_saea:
        archive_x, archive_y = train_deepic_with_saea(args)
        fronts, _ = fast_non_dominated_sort(archive_y)
        final_front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
        print("\nFinal non-dominated archive front from SAEA-DeepIC:")
        print(np.round(final_front, 6))
        return

    print("Starting DeepIC demo")
    print(
        f"Archive={args.archive_size}, Offspring={args.offspring_size}, "
        f"Objectives={args.n_obj}, Selected-per-iteration={args.k_eval}"
    )

    for iteration in range(args.iterations):
        surrogates = fit_kan_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            device=args.device,
            kan_steps=args.kan_steps,
            hidden_width=args.kan_hidden,
            grid=args.kan_grid,
            seed=args.seed + iteration * 13,
        )

        offspring_x = generate_offspring(
            archive_x=archive_x,
            n_offspring=args.offspring_size,
            lower=problem.lower,
            upper=problem.upper,
            sigma=args.mutation_sigma,
        )
        offspring_pred = predict_with_kan(surrogates, offspring_x, args.device).astype(np.float32)
        offspring_sigma = predict_offspring_sigma(
            kan_surrogates=surrogates,
            offspring_x=offspring_x,
            uncertainty_x=uncertainty_x,
            uncertainty_y=uncertainty_y,
            device=args.device,
            surrogate_model=surrogate_model_name(args),
            gp_surrogates=gp_surrogates,
        )
        progress_ratio = min(true_evals_consumed / max(max_fe, 1), 1.0)

        deepic_ranking = infer_deepic_ranking(
            model=deepic,
            archive_x=archive_x,
            archive_y=archive_y,
            offspring_x=offspring_x,
            offspring_pred=offspring_pred,
            offspring_sigma=offspring_sigma,
            lower=problem.lower,
            upper=problem.upper,
            progress=progress_ratio,
            device=args.device,
            top_k=args.k_eval,
        )

        selected_idx = deepic_ranking[: args.k_eval]
        selected_x = offspring_x[selected_idx]
        selected_y = problem.evaluate(selected_x).astype(np.float32)
        true_evals_consumed += selected_x.shape[0]

        archive_x, archive_y = update_archive(
            archive_x=archive_x,
            archive_y=archive_y,
            new_x=selected_x,
            new_y=selected_y,
        )
        uncertainty_x, uncertainty_y = update_uncertainty_archive(
            uncertainty_x=uncertainty_x,
            uncertainty_y=uncertainty_y,
            new_x=selected_x,
            new_y=selected_y,
        )

        fronts, _ = fast_non_dominated_sort(archive_y)
        best_front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
        best_mean = best_front.mean(axis=0)
        print(
            f"Iter {iteration + 1:02d} | "
            f"progress={progress_ratio:.3f} | "
            f"front0={len(fronts[0])} | "
            f"mean(front0 objectives)={np.round(best_mean, 4)}"
        )

    fronts, _ = fast_non_dominated_sort(archive_y)
    final_front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    print("\nFinal non-dominated archive front:")
    print(np.round(final_front, 6))


if __name__ == "__main__":
    # Optional shortcut: run the newer `deepic_demo.py` flow from this entrypoint.
    # We remove the flag from sys.argv to avoid argparse conflicts.
    if "--deepic-demo" in sys.argv:
        sys.argv = [arg for arg in sys.argv if arg != "--deepic-demo"]
        import deepic_demo

        deepic_demo.main()
    else:
        main()
