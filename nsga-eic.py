import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import demo
import multisource_eva_common as multisource
from infil_criterion import EIC


def load_nda_module():
    nda_path = Path(__file__).resolve().parent / "nsga-nda.py"
    spec = importlib.util.spec_from_file_location("nsga_nda_module", nda_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nda = load_nda_module()
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
    fronts, _ = demo.fast_non_dominated_sort(y)
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
    fronts, ranks = demo.fast_non_dominated_sort(values)
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

    fronts, _ = demo.fast_non_dominated_sort(population_y)
    pseudo_front_idx = np.asarray(fronts[0], dtype=np.int64)
    pseudo_front_x = population_x[pseudo_front_idx]
    pseudo_front_y = population_y[pseudo_front_idx]

    if pseudo_front_x.shape[0] < n_offspring:
        order = _nsga2_sort_key(population_y)
        pseudo_front_x = population_x[order]
        pseudo_front_y = population_y[order]

    return pseudo_front_x.astype(np.float32), pseudo_front_y.astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="ZDT1 optimization with KAN surrogate + EIC infill")
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--archive_size", type=int, default=100)
    parser.add_argument("--offspring_size", type=int, default=24)
    parser.add_argument("--k_eval", type=int, default=5)
    parser.add_argument("--max_fe", type=int, default=200)
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
    parser.add_argument("--compare", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-EIC on ZDT1")
    parser.add_argument("--compare_zdt2", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-EIC on ZDT2")
    parser.add_argument("--compare_zdt2_only_model", action="store_true", help="Compare ZDT2-only DeepIC-assisted EA against NSGA-EIC on ZDT2")
    parser.add_argument("--compare_zdt7", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-EIC on ZDT7")
    parser.add_argument("--compare_dtlz1", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-EIC on DTLZ1")
    parser.add_argument("--compare_dtlz2", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-EIC on DTLZ2")
    parser.add_argument("--compare_dtlz3", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-EIC on DTLZ3")
    parser.add_argument("--compare_dtlz4", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-EIC on DTLZ4")
    parser.add_argument("--compare_dtlz5", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-EIC on DTLZ5")
    parser.add_argument("--compare_dtlz6", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-EIC on DTLZ6")
    parser.add_argument("--compare_dtlz7", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-EIC on DTLZ7")
    parser.add_argument("--train_zdt2_only", action="store_true", help="Train DeepIC on ZDT2 only and save model")
    return parser.parse_args()


def _true_front(problem_name: str) -> np.ndarray:
    true_f1 = np.linspace(0.0, 1.0, 200, dtype=np.float32)
    if problem_name == "ZDT1":
        true_f2 = 1.0 - np.sqrt(true_f1)
    elif problem_name == "ZDT2":
        true_f2 = 1.0 - true_f1 ** 2
    elif problem_name == "ZDT7":
        true_f2 = 1.0 - true_f1 * (1.0 + np.sin(3.0 * np.pi * true_f1))
    elif problem_name == "DTLZ1":
        true_f1 = np.linspace(0.0, 0.5, 200, dtype=np.float32)
        true_f2 = 0.5 - true_f1
    elif problem_name == "DTLZ2":
        theta1 = np.linspace(0.0, 0.5 * np.pi, 25, dtype=np.float32)
        theta2 = np.linspace(0.0, 0.5 * np.pi, 25, dtype=np.float32)
        t1, t2 = np.meshgrid(theta1, theta2, indexing="ij")
        f1 = np.cos(t1) * np.cos(t2)
        f2 = np.cos(t1) * np.sin(t2)
        f3 = np.sin(t1)
        return np.stack([f1.ravel(), f2.ravel(), f3.ravel()], axis=1).astype(np.float32)
    elif problem_name == "DTLZ3":
        theta = np.linspace(0.0, 0.5 * np.pi, 200, dtype=np.float32)
        true_f1 = np.cos(theta)
        true_f2 = np.sin(theta)
    elif problem_name == "DTLZ5":
        theta = np.linspace(0.0, 0.5 * np.pi, 200, dtype=np.float32)
        f1 = np.cos(theta) * np.cos(np.pi / 4.0)
        f2 = np.cos(theta) * np.sin(np.pi / 4.0)
        f3 = np.sin(theta)
        return np.stack([f1, f2, f3], axis=1).astype(np.float32)
    elif problem_name in {"DTLZ4", "DTLZ6"}:
        theta = np.linspace(0.0, 0.5 * np.pi, 200, dtype=np.float32)
        true_f1 = np.cos(theta)
        true_f2 = np.sin(theta)
    elif problem_name == "DTLZ7":
        true_f1 = np.linspace(0.0, 1.0, 200, dtype=np.float32)
        true_f2 = 4.0 - true_f1 * (1.0 + np.sin(3.0 * np.pi * true_f1))
    else:
        raise ValueError(f"Unsupported problem: {problem_name}")
    return np.stack([true_f1, true_f2], axis=1)


def _plot_front(title: str, final_front: np.ndarray, true_front: np.ndarray, obtained_label: str):
    if final_front.shape[1] == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title)
        ax.scatter(final_front[:, 0], final_front[:, 1], final_front[:, 2], s=20, alpha=0.8, label=obtained_label)
        ax.scatter(true_front[:, 0], true_front[:, 1], true_front[:, 2], s=8, alpha=0.25, label="True Pareto Front")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
        ax.legend()
        plt.show()
        return

    plt.figure(figsize=(7, 5))
    plt.title(title)
    plt.scatter(final_front[:, 0], final_front[:, 1], s=24, c="blue", alpha=0.8, label=obtained_label)
    plt.plot(true_front[:, 0], true_front[:, 1], "r-", linewidth=2, label="True Pareto Front")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.legend()
    plt.grid(True)
    plt.show()


def _plot_front_comparison(title: str, front_a: np.ndarray, label_a: str, front_b: np.ndarray, label_b: str, true_front: np.ndarray):
    if front_a.shape[1] == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title)
        ax.scatter(front_a[:, 0], front_a[:, 1], front_a[:, 2], s=20, alpha=0.8, label=label_a)
        ax.scatter(front_b[:, 0], front_b[:, 1], front_b[:, 2], s=20, alpha=0.8, label=label_b)
        ax.scatter(true_front[:, 0], true_front[:, 1], true_front[:, 2], s=8, alpha=0.25, label="True Pareto Front")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
        ax.legend()
        plt.show()
        return

    plt.figure(figsize=(8, 5))
    plt.title(title)
    plt.scatter(front_a[:, 0], front_a[:, 1], s=24, alpha=0.8, label=label_a)
    plt.scatter(front_b[:, 0], front_b[:, 1], s=24, alpha=0.8, label=label_b)
    plt.plot(true_front[:, 0], true_front[:, 1], "k-", linewidth=2, label="True Pareto Front")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(True)
    plt.legend()
    plt.show()


def _reference_point(problem_name: str, dim: int) -> np.ndarray:
    key = problem_name.upper()
    if key == "DTLZ1":
        return np.array([1.1, 1.1], dtype=np.float32)
    if key not in REFERENCE_POINTS:
        raise ValueError(f"Unsupported problem: {problem_name}")
    ref = REFERENCE_POINTS[key]
    if key in {"DTLZ2", "DTLZ5"}:
        n_obj = 3
    else:
        n_obj = 2 if key.startswith(("ZDT", "DTLZ")) else ref.shape[0]
    return ref[:n_obj].astype(np.float32)


def _normalization_bounds(problem_name: str, dim: int) -> tuple[np.ndarray, np.ndarray]:
    if problem_name != "DTLZ1":
        raise ValueError(f"Unsupported problem for normalization bounds: {problem_name}")

    rng = np.random.default_rng(0)
    probe_x = rng.uniform(0.0, 1.0, size=(4096, dim)).astype(np.float32)
    probe_y = nda.ZDTProblem(name=problem_name, dim=dim).evaluate(probe_x)
    y_min = np.zeros(probe_y.shape[1], dtype=np.float32)
    y_max = np.max(probe_y, axis=0).astype(np.float32)
    return y_min, y_max


def _normalize_for_hv(problem_name: str, values: np.ndarray, dim: int) -> np.ndarray:
    if problem_name != "DTLZ1":
        return values

    y_min, y_max = _normalization_bounds(problem_name, dim)
    span = np.maximum(y_max - y_min, 1e-12)
    return (values - y_min) / span


def run_nsga_eic_problem(args, problem_name: str, plot: bool = True, initial_archive_x: Optional[np.ndarray] = None):
    nda.set_seed(args.seed)
    if args.max_fe < args.archive_size:
        raise ValueError("max_fe must be at least as large as archive_size.")
    if (args.max_fe - args.archive_size) % args.k_eval != 0:
        raise ValueError("max_fe - archive_size must be divisible by k_eval.")

    problem = nda.ZDTProblem(name=problem_name, dim=args.dim)
    ref_point = _reference_point(problem_name, args.dim)

    pretrain_x, pretrain_y, surrogates = nda.pre_train_kan_surrogate_for_problem(
        problem=problem,
        device=args.device,
        kan_steps=args.kan_steps,
        hidden_width=args.kan_hidden,
        grid=args.kan_grid,
        seed=args.seed,
    )
    print(f"Pre-trained KAN surrogate on {problem_name} with {pretrain_x.shape[0]} samples.")

    if initial_archive_x is None:
        archive_x = multisource.latin_hypercube_sample(
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

    selector = EIC(seed=args.seed)

    true_evals = args.archive_size
    remaining_budget = args.max_fe - true_evals
    steps_to_run = remaining_budget // args.k_eval
    hv_history: list[float] = []
    reward_history: list[float] = []

    fronts, _ = nda.fast_non_dominated_sort(archive_y)
    front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    hv_front = _normalize_for_hv(problem_name, front, args.dim)
    initial_hv = nda.hypervolume_2d(hv_front, ref_point)
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
            predict_fn=nda.predict_with_kan,
            generate_fn=nda.generate_offspring,
        )
        archive_sur_pred = nda.predict_with_kan(surrogates, archive_x, args.device)
        offspring_sigma = nda.estimate_uncertainty(
            archive_x=archive_x,
            archive_y=archive_y,
            archive_pred=archive_sur_pred,
            offspring_x=offspring_x,
        ).astype(np.float32)

        selected = selector.select(
            offspring_x=offspring_x,
            offspring_pred=offspring_pred,
            archive_pred=archive_y,
            offspring_sigma=offspring_sigma,
            n_select=args.k_eval,
        )
        selected_idx = selected.indices
        selected_x = offspring_x[selected_idx]
        selected_y = problem.evaluate(selected_x)
        reward_value = float(
            demo.DeepICClass.fpareto_improvement_reward(
                previous_front=archive_y,
                selected_objectives=selected_y,
            )
        )
        reward_history.append(reward_value)

        archive_x, archive_y = nda.update_archive(
            archive_x=archive_x,
            archive_y=archive_y,
            new_x=selected_x,
            new_y=selected_y,
        )

        fronts, _ = nda.fast_non_dominated_sort(archive_y)
        front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
        hv_front = _normalize_for_hv(problem_name, front, args.dim)
        hv_value = nda.hypervolume_2d(hv_front, ref_point)
        hv_history.append(hv_value)

        print(
            f"Iter {step + 1:02d} | archive={archive_x.shape[0]} | "
            f"front0={front.shape[0]} | HV={hv_value:.6f} | reward={reward_value:.6f} | pseudo_front={offspring_x.shape[0]}"
        )

        true_evals += args.k_eval
        if true_evals >= args.max_fe:
            break

        combined_x = np.vstack([pretrain_x, archive_x])
        combined_y = np.vstack([pretrain_y, archive_y])
        surrogates = nda.fit_kan_surrogates(
            archive_x=combined_x,
            archive_y=combined_y,
            device=args.device,
            kan_steps=args.kan_steps,
            hidden_width=args.kan_hidden,
            grid=args.kan_grid,
            seed=args.seed + 100 + step,
        )

    fronts, _ = nda.fast_non_dominated_sort(archive_y)
    final_front = archive_y[np.asarray(fronts[0], dtype=np.int64)]

    true_front = _true_front(problem_name)

    print("\nObtained Pareto front:")
    print(np.round(final_front, 6))
    print("\nTrue Pareto front:")
    print(np.round(true_front, 6))

    if plot:
        plt.figure(figsize=(7, 5))
        plt.title(f"{problem_name} Hypervolume Progress (KAN + EIC)")
        plt.plot(hv_history, marker="o")
        plt.xlabel("Step")
        plt.ylabel("Hypervolume")
        plt.grid(True)
        plt.show()
        _plot_front(f"{problem_name} Pareto Front (KAN + EIC)", final_front, true_front, "Obtained Front")

    _save_reward_log(
        f"nsga_eic_{problem_name.lower()}_rewards.json",
        {
            "script": "nsga-eic.py",
            "mode": "run_nsga_eic_problem",
            "problem_name": problem_name,
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


def run_nsga_eic(args, plot: bool = True):
    return run_nsga_eic_problem(args, problem_name="ZDT1", plot=plot)


def train_zdt2_only(args):
    print("Pre-training KAN surrogate on ZDT2...")
    pretrain_x, pretrain_y, surrogates = demo.pre_train_kan_surrogate_for_problem(
        args=args,
        problem_name="ZDT2",
    )
    print("Pre-training completed.")

    replay = demo.ReplayBuffer(capacity=256)

    deepic = demo.DeepICClass(
        hidden_dim=args.deepic_hidden,
        n_heads=args.deepic_heads,
        ff_dim=args.deepic_ff,
    ).to(args.device)
    deepic_optimizer = demo.torch.optim.Adam(deepic.parameters(), lr=1e-3)

    for epoch in range(50):
        print(f"ZDT2 Epoch {epoch + 1}/50")

        problem = demo.ZDTProblem(name="ZDT2", dim=args.dim)
        archive_x = np.random.uniform(problem.lower, problem.upper, size=(args.archive_size, args.dim)).astype(np.float32)
        archive_y = problem.evaluate(archive_x)
        true_evals = args.archive_size
        max_fe = args.max_fe
        remaining_budget = max_fe - true_evals
        steps_to_run = remaining_budget // args.k_eval

        for step in range(steps_to_run):
            offspring_x = demo.generate_offspring(
                archive_x=archive_x,
                n_offspring=args.offspring_size,
                lower=problem.lower,
                upper=problem.upper,
                sigma=args.mutation_sigma,
            )
            offspring_pred = demo.predict_with_kan(surrogates, offspring_x, args.device).astype(np.float32)
            archive_pred = demo.predict_with_kan(surrogates, archive_x, args.device).astype(np.float32)
            offspring_sigma = demo.estimate_uncertainty(
                archive_x=archive_x,
                archive_y=archive_y,
                archive_pred=archive_pred,
                offspring_x=offspring_x,
            ).astype(np.float32)

            progress = float(true_evals / max_fe)
            ranking = demo.infer_deepic_ranking(
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

            old_best = np.min(archive_y[:, 0])
            archive_x, archive_y = demo.update_archive(
                archive_x=archive_x,
                archive_y=archive_y,
                new_x=selected_x,
                new_y=selected_y,
            )
            new_best = np.min(archive_y[:, 0])

            reward = float(old_best - new_best)

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
                    demo.adapt_deepic(
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
            surrogates = demo.fit_kan_surrogates(
                archive_x=combined_x,
                archive_y=combined_y,
                device=args.device,
                kan_steps=args.kan_steps,
                hidden_width=args.kan_hidden,
                grid=args.kan_grid,
                seed=args.seed + epoch * 100 + step,
            )

        print(
            f"ZDT2 epoch {epoch + 1} done, true_evals={true_evals}, best_obj1={np.min(archive_y[:, 0])}\n"
        )
        if (epoch + 1) % 5 == 0:
            multisource.save_colab_model_checkpoint(
                deepic.state_dict(),
                f"deepic_zdt2_epoch_{epoch + 1}.pth",
            )

    demo.torch.save(deepic.state_dict(), "deepic_zdt2.pth")
    print("DeepIC model saved to deepic_zdt2.pth")
    return deepic


def run_deepic_problem(
    args,
    problem_name: str,
    plot: bool = True,
    checkpoint_path: Optional[str] = None,
    initial_archive_x: Optional[np.ndarray] = None,
):
    if problem_name not in {"ZDT1", "ZDT2", "ZDT7", "DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"}:
        raise ValueError(f"Unsupported problem: {problem_name}")

    if problem_name in {"DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"}:
        problem = nda.ZDTProblem(name=problem_name, dim=args.dim)
        pretrain_x, pretrain_y, surrogates = nda.pre_train_kan_surrogate_for_problem(
            problem=problem,
            device=args.device,
            kan_steps=args.kan_steps,
            hidden_width=args.kan_hidden,
            grid=args.kan_grid,
            seed=args.seed,
        )
    else:
        problem = demo.ZDTProblem(name=problem_name, dim=args.dim)
        pretrain_x, pretrain_y, surrogates = demo.pre_train_kan_surrogate_for_problem(
            args=args,
            problem_name=problem_name,
        )
    print(f"Pre-trained KAN surrogate on {problem_name} with {pretrain_x.shape[0]} samples.")

    deepic = demo.DeepICClass(
        hidden_dim=args.deepic_hidden,
        n_heads=args.deepic_heads,
        ff_dim=args.deepic_ff,
    ).to(args.device)

    if checkpoint_path is None:
        if problem_name == "ZDT1":
            checkpoint_path = "deepic_zdt1.pth"
        elif problem_name == "ZDT2":
            checkpoint_path = "deepic_zdt2.pth"
        else:
            checkpoint_path = "deepic_zdt.pth"

    if os.path.exists(checkpoint_path):
        deepic.load_state_dict(demo.torch.load(checkpoint_path, map_location=args.device))
        print(f"Loaded DeepIC model from {checkpoint_path}")
    else:
        if checkpoint_path == "deepic_zdt1.pth":
            print("deepic_zdt1.pth not found. Training DeepIC on ZDT1 first...")
            deepic = demo.train_deepic_zdt1(args)
        elif checkpoint_path == "deepic_zdt2.pth":
            print("deepic_zdt2.pth not found. Training DeepIC on ZDT2 first...")
            deepic = train_zdt2_only(args)
        else:
            print(f"{checkpoint_path} not found. Training DeepIC on ZDT1-6 first...")
            deepic = demo.train_deepic_zdt(args)

    if initial_archive_x is None:
        archive_x = multisource.latin_hypercube_sample(
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
    ref_point = _reference_point(problem_name, args.dim)

    true_evals = args.archive_size
    remaining_budget = args.max_fe - true_evals
    steps_to_run = remaining_budget // args.k_eval
    hv_history: list[float] = []
    reward_history: list[float] = []

    fronts, _ = demo.fast_non_dominated_sort(archive_y)
    front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    hv_front = _normalize_for_hv(problem_name, front, args.dim)
    initial_hv = demo.hypervolume_2d(hv_front, ref_point)
    hv_history.append(initial_hv)
    print(
        f"Init    | archive={archive_x.shape[0]} | "
        f"front0={front.shape[0]} | HV={initial_hv:.6f}"
    )

    for step in range(steps_to_run):
        surrogates = demo.fit_kan_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            device=args.device,
            kan_steps=args.kan_steps,
            hidden_width=args.kan_hidden,
            grid=args.kan_grid,
            seed=args.seed + step,
        )

        offspring_x, offspring_pred = generate_nsga2_pseudo_front(
            archive_x=archive_x,
            problem=problem,
            surrogates=surrogates,
            device=args.device,
            n_offspring=args.offspring_size,
            sigma=args.mutation_sigma,
            surrogate_nsga_steps=args.surrogate_nsga_steps,
            predict_fn=demo.predict_with_kan,
            generate_fn=demo.generate_offspring,
        )
        archive_pred = demo.predict_with_kan(surrogates, archive_x, args.device).astype(np.float32)
        offspring_sigma = demo.estimate_uncertainty(
            archive_x=archive_x,
            archive_y=archive_y,
            archive_pred=archive_pred,
            offspring_x=offspring_x,
        ).astype(np.float32)

        progress = float(true_evals / args.max_fe)
        ranking = demo.infer_deepic_ranking(
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
        reward_value = float(
            demo.DeepICClass.fpareto_improvement_reward(
                previous_front=archive_y,
                selected_objectives=selected_y,
            )
        )
        reward_history.append(reward_value)

        archive_x, archive_y = demo.update_archive(
            archive_x=archive_x,
            archive_y=archive_y,
            new_x=selected_x,
            new_y=selected_y,
        )

        fronts, _ = demo.fast_non_dominated_sort(archive_y)
        front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
        hv_front = _normalize_for_hv(problem_name, front, args.dim)
        hv_history.append(demo.hypervolume_2d(hv_front, ref_point))

        print(
            f"Number of individuals in archive: {archive_x.shape[0]}, "
            f"Hypervolume: {hv_history[-1]:.6f}, reward={reward_value:.6f}, pseudo_front={offspring_x.shape[0]}"
        )

        true_evals += args.k_eval
        if true_evals >= args.max_fe:
            break

        combined_x = np.vstack([pretrain_x, archive_x])
        combined_y = np.vstack([pretrain_y, archive_y])
        surrogates = demo.fit_kan_surrogates(
            archive_x=combined_x,
            archive_y=combined_y,
            device=args.device,
            kan_steps=args.kan_steps,
            hidden_width=args.kan_hidden,
            grid=args.kan_grid,
            seed=args.seed + 100 + step,
        )

    fronts, _ = demo.fast_non_dominated_sort(archive_y)
    final_front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    true_front = _true_front(problem_name)

    if plot:
        plt.figure(figsize=(7, 5))
        plt.title(f"{problem_name} Hypervolume Progress (DeepIC)")
        plt.plot(hv_history, marker="o")
        plt.xlabel("Step")
        plt.ylabel("Hypervolume")
        plt.grid(True)
        plt.show()
        _plot_front(f"{problem_name} Pareto Front (DeepIC)", final_front, true_front, "Obtained Front")

    _save_reward_log(
        f"nsga_eic_deepic_{problem_name.lower()}_rewards.json",
        {
            "script": "nsga-eic.py",
            "mode": "run_deepic_problem",
            "problem_name": problem_name,
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
    else:
        print("deepic_zdt1.pth not found. Training DeepIC on ZDT1 first...")
        demo.train_deepic_zdt1(args)

    problem = nda.ZDTProblem(name="ZDT1", dim=args.dim)
    shared_init_x = multisource.latin_hypercube_sample(
        lower=problem.lower,
        upper=problem.upper,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )
    deepic_result = run_deepic_problem(
        args,
        problem_name="ZDT1",
        plot=False,
        checkpoint_path="deepic_zdt1.pth",
        initial_archive_x=shared_init_x,
    )
    eic_result = run_nsga_eic_problem(args, problem_name="ZDT1", plot=False, initial_archive_x=shared_init_x)

    print(f"\nDeepIC-assisted EA final HV: {deepic_result['hv_history'][-1]:.6f}")
    print(f"NSGA-EIC final HV: {eic_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {deepic_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title("ZDT1 Hypervolume Comparison")
    plt.plot(deepic_result["hv_history"], marker="o", label="Pre-trained DeepIC-assisted EA")
    plt.plot(eic_result["hv_history"], marker="s", label="NSGA-EIC")
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
        eic_result["final_front"][:, 0],
        eic_result["final_front"][:, 1],
        s=24,
        alpha=0.8,
        label="NSGA-EIC",
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

    return {"deepic": deepic_result, "nsga_eic": eic_result}


def run_comparison_zdt2(args):
    if os.path.exists("deepic_zdt.pth"):
        print("Using saved DeepIC model from deepic_zdt.pth")
    else:
        print("deepic_zdt.pth not found. Training DeepIC on ZDT1-6 first...")
        demo.train_deepic_zdt(args)

    problem = nda.ZDTProblem(name="ZDT2", dim=args.dim)
    shared_init_x = multisource.latin_hypercube_sample(
        lower=problem.lower,
        upper=problem.upper,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )
    deepic_result = run_deepic_problem(
        args,
        problem_name="ZDT2",
        plot=False,
        checkpoint_path="deepic_zdt.pth",
        initial_archive_x=shared_init_x,
    )
    eic_result = run_nsga_eic_problem(args, problem_name="ZDT2", plot=False, initial_archive_x=shared_init_x)

    print(f"\nDeepIC-assisted EA final HV: {deepic_result['hv_history'][-1]:.6f}")
    print(f"NSGA-EIC final HV: {eic_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {deepic_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title("ZDT2 Hypervolume Comparison")
    plt.plot(deepic_result["hv_history"], marker="o", label="Pre-trained DeepIC-assisted EA")
    plt.plot(eic_result["hv_history"], marker="s", label="NSGA-EIC")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.title("ZDT2 Pareto Front Comparison")
    plt.scatter(
        deepic_result["final_front"][:, 0],
        deepic_result["final_front"][:, 1],
        s=24,
        alpha=0.8,
        label="DeepIC-assisted EA",
    )
    plt.scatter(
        eic_result["final_front"][:, 0],
        eic_result["final_front"][:, 1],
        s=24,
        alpha=0.8,
        label="NSGA-EIC",
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

    return {"deepic": deepic_result, "nsga_eic": eic_result}


def run_comparison_zdt7(args):
    if os.path.exists("deepic_zdt.pth"):
        print("Using saved DeepIC model from deepic_zdt.pth")
    else:
        print("deepic_zdt.pth not found. Training DeepIC on ZDT1-6 first...")
        demo.train_deepic_zdt(args)

    problem = nda.ZDTProblem(name="ZDT7", dim=args.dim)
    shared_init_x = multisource.latin_hypercube_sample(
        lower=problem.lower,
        upper=problem.upper,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )
    deepic_result = run_deepic_problem(
        args,
        problem_name="ZDT7",
        plot=False,
        checkpoint_path="deepic_zdt.pth",
        initial_archive_x=shared_init_x,
    )
    eic_result = run_nsga_eic_problem(args, problem_name="ZDT7", plot=False, initial_archive_x=shared_init_x)

    print(f"\nDeepIC-assisted EA final HV: {deepic_result['hv_history'][-1]:.6f}")
    print(f"NSGA-EIC final HV: {eic_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {deepic_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title("ZDT7 Hypervolume Comparison")
    plt.plot(deepic_result["hv_history"], marker="o", label="Pre-trained DeepIC-assisted EA")
    plt.plot(eic_result["hv_history"], marker="s", label="NSGA-EIC")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.title("ZDT7 Pareto Front Comparison")
    plt.scatter(
        deepic_result["final_front"][:, 0],
        deepic_result["final_front"][:, 1],
        s=24,
        alpha=0.8,
        label="DeepIC-assisted EA",
    )
    plt.scatter(
        eic_result["final_front"][:, 0],
        eic_result["final_front"][:, 1],
        s=24,
        alpha=0.8,
        label="NSGA-EIC",
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

    return {"deepic": deepic_result, "nsga_eic": eic_result}


def run_comparison_dtlz1(args):
    if os.path.exists("deepic_zdt.pth"):
        print("Using saved DeepIC model from deepic_zdt.pth")
    else:
        print("deepic_zdt.pth not found. Training DeepIC on ZDT1-6 first...")
        demo.train_deepic_zdt(args)

    problem = nda.ZDTProblem(name="DTLZ1", dim=args.dim)
    shared_init_x = multisource.latin_hypercube_sample(
        lower=problem.lower,
        upper=problem.upper,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )
    deepic_result = run_deepic_problem(
        args,
        problem_name="DTLZ1",
        plot=False,
        checkpoint_path="deepic_zdt.pth",
        initial_archive_x=shared_init_x,
    )
    eic_result = run_nsga_eic_problem(args, problem_name="DTLZ1", plot=False, initial_archive_x=shared_init_x)

    print(f"\nDeepIC-assisted EA final HV: {deepic_result['hv_history'][-1]:.6f}")
    print(f"NSGA-EIC final HV: {eic_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {deepic_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title("DTLZ1 Hypervolume Comparison")
    plt.plot(deepic_result["hv_history"], marker="o", label="Pre-trained DeepIC-assisted EA")
    plt.plot(eic_result["hv_history"], marker="s", label="NSGA-EIC")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.title("DTLZ1 Pareto Front Comparison")
    plt.scatter(
        deepic_result["final_front"][:, 0],
        deepic_result["final_front"][:, 1],
        s=24,
        alpha=0.8,
        label="DeepIC-assisted EA",
    )
    plt.scatter(
        eic_result["final_front"][:, 0],
        eic_result["final_front"][:, 1],
        s=24,
        alpha=0.8,
        label="NSGA-EIC",
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

    return {"deepic": deepic_result, "nsga_eic": eic_result}


def run_comparison_dtlz2(args):
    if os.path.exists("deepic_zdt.pth"):
        print("Using saved DeepIC model from deepic_zdt.pth")
    else:
        print("deepic_zdt.pth not found. Training DeepIC on ZDT1-6 first...")
        demo.train_deepic_zdt(args)

    problem = nda.ZDTProblem(name="DTLZ2", dim=args.dim)
    shared_init_x = multisource.latin_hypercube_sample(
        lower=problem.lower,
        upper=problem.upper,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )
    deepic_result = run_deepic_problem(
        args,
        problem_name="DTLZ2",
        plot=False,
        checkpoint_path="deepic_zdt.pth",
        initial_archive_x=shared_init_x,
    )
    eic_result = run_nsga_eic_problem(args, problem_name="DTLZ2", plot=False, initial_archive_x=shared_init_x)

    print(f"\nDeepIC-assisted EA final HV: {deepic_result['hv_history'][-1]:.6f}")
    print(f"NSGA-EIC final HV: {eic_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {deepic_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title("DTLZ2 Hypervolume Comparison")
    plt.plot(deepic_result["hv_history"], marker="o", label="Pre-trained DeepIC-assisted EA")
    plt.plot(eic_result["hv_history"], marker="s", label="NSGA-EIC")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend()
    plt.show()

    _plot_front_comparison(
        "DTLZ2 Pareto Front Comparison",
        deepic_result["final_front"],
        "DeepIC-assisted EA",
        eic_result["final_front"],
        "NSGA-EIC",
        deepic_result["true_front"],
    )

    return {"deepic": deepic_result, "nsga_eic": eic_result}


def run_comparison_dtlz3(args):
    if os.path.exists("deepic_zdt.pth"):
        print("Using saved DeepIC model from deepic_zdt.pth")
    else:
        print("deepic_zdt.pth not found. Training DeepIC on ZDT1-6 first...")
        demo.train_deepic_zdt(args)

    problem = nda.ZDTProblem(name="DTLZ3", dim=args.dim)
    shared_init_x = multisource.latin_hypercube_sample(
        lower=problem.lower,
        upper=problem.upper,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )
    deepic_result = run_deepic_problem(
        args,
        problem_name="DTLZ3",
        plot=False,
        checkpoint_path="deepic_zdt.pth",
        initial_archive_x=shared_init_x,
    )
    eic_result = run_nsga_eic_problem(args, problem_name="DTLZ3", plot=False, initial_archive_x=shared_init_x)

    print(f"\nDeepIC-assisted EA final HV: {deepic_result['hv_history'][-1]:.6f}")
    print(f"NSGA-EIC final HV: {eic_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {deepic_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title("DTLZ3 Hypervolume Comparison")
    plt.plot(deepic_result["hv_history"], marker="o", label="Pre-trained DeepIC-assisted EA")
    plt.plot(eic_result["hv_history"], marker="s", label="NSGA-EIC")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.title("DTLZ3 Pareto Front Comparison")
    plt.scatter(
        deepic_result["final_front"][:, 0],
        deepic_result["final_front"][:, 1],
        s=24,
        alpha=0.8,
        label="DeepIC-assisted EA",
    )
    plt.scatter(
        eic_result["final_front"][:, 0],
        eic_result["final_front"][:, 1],
        s=24,
        alpha=0.8,
        label="NSGA-EIC",
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

    return {"deepic": deepic_result, "nsga_eic": eic_result}


def run_comparison_zdt2_only_model(args):
    if os.path.exists("deepic_zdt2.pth"):
        print("Using saved DeepIC model from deepic_zdt2.pth")
    else:
        print("deepic_zdt2.pth not found. Training DeepIC on ZDT2 first...")
        train_zdt2_only(args)

    problem = nda.ZDTProblem(name="ZDT2", dim=args.dim)
    shared_init_x = multisource.latin_hypercube_sample(
        lower=problem.lower,
        upper=problem.upper,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )
    deepic_result = run_deepic_problem(
        args,
        problem_name="ZDT2",
        plot=False,
        checkpoint_path="deepic_zdt2.pth",
        initial_archive_x=shared_init_x,
    )
    eic_result = run_nsga_eic_problem(args, problem_name="ZDT2", plot=False, initial_archive_x=shared_init_x)

    print(f"\nDeepIC-assisted EA final HV: {deepic_result['hv_history'][-1]:.6f}")
    print(f"NSGA-EIC final HV: {eic_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {deepic_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title("ZDT2 Hypervolume Comparison (ZDT2-only DeepIC)")
    plt.plot(deepic_result["hv_history"], marker="o", label="ZDT2-only DeepIC-assisted EA")
    plt.plot(eic_result["hv_history"], marker="s", label="NSGA-EIC")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.title("ZDT2 Pareto Front Comparison (ZDT2-only DeepIC)")
    plt.scatter(
        deepic_result["final_front"][:, 0],
        deepic_result["final_front"][:, 1],
        s=24,
        alpha=0.8,
        label="ZDT2-only DeepIC-assisted EA",
    )
    plt.scatter(
        eic_result["final_front"][:, 0],
        eic_result["final_front"][:, 1],
        s=24,
        alpha=0.8,
        label="NSGA-EIC",
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

    return {"deepic": deepic_result, "nsga_eic": eic_result}


def _run_generic_comparison(args, problem_name: str):
    if os.path.exists("deepic_zdt.pth"):
        print("Using saved DeepIC model from deepic_zdt.pth")
    else:
        print("deepic_zdt.pth not found. Training DeepIC on ZDT1-6 first...")
        demo.train_deepic_zdt(args)

    problem = nda.ZDTProblem(name=problem_name, dim=args.dim)
    shared_init_x = multisource.latin_hypercube_sample(
        lower=problem.lower,
        upper=problem.upper,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )
    deepic_result = run_deepic_problem(
        args,
        problem_name=problem_name,
        plot=False,
        checkpoint_path="deepic_zdt.pth",
        initial_archive_x=shared_init_x,
    )
    eic_result = run_nsga_eic_problem(args, problem_name=problem_name, plot=False, initial_archive_x=shared_init_x)

    print(f"\nDeepIC-assisted EA final HV: {deepic_result['hv_history'][-1]:.6f}")
    print(f"NSGA-EIC final HV: {eic_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {deepic_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title(f"{problem_name} Hypervolume Comparison")
    plt.plot(deepic_result["hv_history"], marker="o", label="Pre-trained DeepIC-assisted EA")
    plt.plot(eic_result["hv_history"], marker="s", label="NSGA-EIC")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.title(f"{problem_name} Pareto Front Comparison")
    plt.scatter(
        deepic_result["final_front"][:, 0],
        deepic_result["final_front"][:, 1],
        s=24,
        alpha=0.8,
        label="DeepIC-assisted EA",
    )
    plt.scatter(
        eic_result["final_front"][:, 0],
        eic_result["final_front"][:, 1],
        s=24,
        alpha=0.8,
        label="NSGA-EIC",
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

    return {"deepic": deepic_result, "nsga_eic": eic_result}


def run_comparison_dtlz4(args):
    return _run_generic_comparison(args, "DTLZ4")


def run_comparison_dtlz5(args):
    return _run_generic_comparison(args, "DTLZ5")


def run_comparison_dtlz6(args):
    return _run_generic_comparison(args, "DTLZ6")


def run_comparison_dtlz7(args):
    return _run_generic_comparison(args, "DTLZ7")


def main():
    args = parse_args()
    if args.train_zdt2_only:
        train_zdt2_only(args)
    elif args.compare_dtlz7:
        run_comparison_dtlz7(args)
    elif args.compare_dtlz6:
        run_comparison_dtlz6(args)
    elif args.compare_dtlz5:
        run_comparison_dtlz5(args)
    elif args.compare_dtlz4:
        run_comparison_dtlz4(args)
    elif args.compare_dtlz3:
        run_comparison_dtlz3(args)
    elif args.compare_dtlz2:
        run_comparison_dtlz2(args)
    elif args.compare_dtlz1:
        run_comparison_dtlz1(args)
    elif args.compare_zdt7:
        run_comparison_zdt7(args)
    elif args.compare_zdt2_only_model:
        run_comparison_zdt2_only_model(args)
    elif args.compare_zdt2:
        run_comparison_zdt2(args)
    elif args.compare:
        run_comparison(args)
    else:
        run_nsga_eic(args, plot=True)


if __name__ == "__main__":
    main()
