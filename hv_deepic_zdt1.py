import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

import demo


def load_module(filename: str, module_name: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_hv_deepic_class():
    path = Path(__file__).resolve().parent / "agent" / "deepic_agent.py"
    spec = importlib.util.spec_from_file_location("deepic_agent_local_hv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HV_DeepIC


nda = load_module("nsga-nda.py", "nsga_nda_module")
nsga_eic = load_module("nsga-eic.py", "nsga_eic_module")
HVDeepICClass = load_hv_deepic_class()

MODEL_PATH = "hv_deepic_zdt1.pth"
PROBLEM_NAME = "ZDT1"
REFERENCE_POINT = np.array([0.9994, 6.0576], dtype=np.float32)


def build_args_namespace(parsed) -> SimpleNamespace:
    return SimpleNamespace(
        dim=parsed.dim,
        archive_size=parsed.archive_size,
        offspring_size=parsed.offspring_size,
        k_eval=parsed.k_eval,
        max_fe=parsed.max_fe,
        mutation_sigma=parsed.mutation_sigma,
        kan_steps=parsed.kan_steps,
        kan_hidden=parsed.kan_hidden,
        kan_grid=parsed.kan_grid,
        deepic_hidden=parsed.deepic_hidden,
        deepic_heads=parsed.deepic_heads,
        deepic_ff=parsed.deepic_ff,
        deepic_lr=parsed.deepic_lr,
        deepic_adapt_steps=parsed.deepic_adapt_steps,
        surrogate_nsga_steps=parsed.surrogate_nsga_steps,
        hv_epsilon=parsed.hv_epsilon,
        seed=parsed.seed,
        device=parsed.device,
    )


def latin_hypercube_sample(lower, upper, n_samples: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower_arr = np.full(dim, lower, dtype=np.float32) if np.isscalar(lower) else np.asarray(lower, dtype=np.float32)
    upper_arr = np.full(dim, upper, dtype=np.float32) if np.isscalar(upper) else np.asarray(upper, dtype=np.float32)

    lhs = np.empty((n_samples, dim), dtype=np.float32)
    for j in range(dim):
        perm = rng.permutation(n_samples)
        lhs[:, j] = (perm + rng.random(n_samples)) / n_samples

    return (lower_arr + lhs * (upper_arr - lower_arr)).astype(np.float32)


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
) -> tuple[np.ndarray, np.ndarray]:
    population_x = demo.generate_offspring(
        archive_x=archive_x,
        n_offspring=n_offspring,
        lower=problem.lower,
        upper=problem.upper,
        sigma=sigma,
    ).astype(np.float32)
    population_y = demo.predict_with_kan(surrogates, population_x, device).astype(np.float32)

    for _ in range(surrogate_nsga_steps):
        offspring_x = demo.generate_offspring(
            archive_x=population_x,
            n_offspring=n_offspring,
            lower=problem.lower,
            upper=problem.upper,
            sigma=sigma,
        ).astype(np.float32)
        offspring_y = demo.predict_with_kan(surrogates, offspring_x, device).astype(np.float32)

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


def train_hv_deepic_zdt1(args):
    demo.set_seed(args.seed)
    problem = nda.ZDTProblem(name=PROBLEM_NAME, dim=args.dim)

    print("Pre-training KAN surrogate on ZDT1...")
    pretrain_x, pretrain_y, surrogates = nda.pre_train_kan_surrogate_for_problem(
        problem=problem,
        device=args.device,
        kan_steps=args.kan_steps,
        hidden_width=args.kan_hidden,
        grid=args.kan_grid,
        seed=args.seed,
    )
    print("Pre-training completed.")

    replay = demo.ReplayBuffer(capacity=256)
    deepic = HVDeepICClass(
        hidden_dim=args.deepic_hidden,
        n_heads=args.deepic_heads,
        ff_dim=args.deepic_ff,
        reward_ref_point=REFERENCE_POINT,
        reward_epsilon=args.hv_epsilon,
    ).to(args.device)
    deepic_optimizer = demo.torch.optim.Adam(deepic.parameters(), lr=args.deepic_lr)

    for epoch in range(50):
        print(f"HV_DeepIC ZDT1 Epoch {epoch + 1}/50")

        archive_x = latin_hypercube_sample(
            lower=problem.lower,
            upper=problem.upper,
            n_samples=args.archive_size,
            dim=args.dim,
            seed=args.seed + epoch,
        )
        archive_y = problem.evaluate(archive_x)
        true_evals = args.archive_size
        steps_to_run = (args.max_fe - true_evals) // args.k_eval

        for step in range(steps_to_run):
            offspring_x, offspring_pred = generate_nsga2_pseudo_front(
                archive_x=archive_x,
                problem=problem,
                surrogates=surrogates,
                device=args.device,
                n_offspring=args.offspring_size,
                sigma=args.mutation_sigma,
                surrogate_nsga_steps=args.surrogate_nsga_steps,
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
            )

            selected_idx = ranking[: args.k_eval]
            selected_x = offspring_x[selected_idx]
            selected_y = problem.evaluate(selected_x)

            reward = deepic.pareto_improvement_reward(
                previous_archive=archive_y,
                selected_objectives=selected_y,
                ref_point=REFERENCE_POINT,
                epsilon=args.hv_epsilon,
            )

            archive_x, archive_y = demo.update_archive(
                archive_x=archive_x,
                archive_y=archive_y,
                new_x=selected_x,
                new_y=selected_y,
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
                    "lower": problem.lower,
                    "upper": problem.upper,
                }
            )

            if len(replay) >= 32:
                for sample in replay.sample(32):
                    demo.adapt_deepic(
                        model=deepic,
                        optimizer=deepic_optimizer,
                        archive_x=sample["archive_x"],
                        archive_y=sample["archive_y"],
                        offspring_x=sample["offspring_x"],
                        offspring_pred=sample["offspring_pred"],
                        offspring_sigma=sample["offspring_sigma"],
                        lower=sample["lower"],
                        upper=sample["upper"],
                        progress=sample["progress"],
                        target_ranking=sample["ranking"],
                        device=args.device,
                        steps=1,
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
                seed=args.seed + epoch * 100 + step,
            )

        fronts, _ = demo.fast_non_dominated_sort(archive_y)
        front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
        hv_value = demo.hypervolume_2d(front, REFERENCE_POINT)
        print(
            f"HV_DeepIC epoch {epoch + 1} done, true_evals={true_evals}, "
            f"front0={front.shape[0]}, hv={hv_value:.6f}, surrogate_nsga_steps={args.surrogate_nsga_steps}"
        )

    demo.torch.save(deepic.state_dict(), MODEL_PATH)
    print(f"HV_DeepIC model saved to {MODEL_PATH}")
    return deepic


def load_or_train_hv_deepic(args):
    deepic = HVDeepICClass(
        hidden_dim=args.deepic_hidden,
        n_heads=args.deepic_heads,
        ff_dim=args.deepic_ff,
        reward_ref_point=REFERENCE_POINT,
        reward_epsilon=args.hv_epsilon,
    ).to(args.device)

    if Path(MODEL_PATH).exists():
        print(f"Using saved HV_DeepIC model from {MODEL_PATH}")
        deepic.load_state_dict(demo.torch.load(MODEL_PATH, map_location=args.device))
        return deepic

    return train_hv_deepic_zdt1(args)


def run_hv_deepic_zdt1(args, deepic, plot: bool = True, initial_archive_x: np.ndarray = None):
    problem = nda.ZDTProblem(name=PROBLEM_NAME, dim=args.dim)

    pretrain_x, pretrain_y, surrogates = nda.pre_train_kan_surrogate_for_problem(
        problem=problem,
        device=args.device,
        kan_steps=args.kan_steps,
        hidden_width=args.kan_hidden,
        grid=args.kan_grid,
        seed=args.seed,
    )
    print(f"Pre-trained KAN surrogate on ZDT1-{args.dim}D with {pretrain_x.shape[0]} samples.")

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
    steps_to_run = (args.max_fe - true_evals) // args.k_eval
    hv_history = []

    for step in range(steps_to_run):
        offspring_x, offspring_pred = generate_nsga2_pseudo_front(
            archive_x=archive_x,
            problem=problem,
            surrogates=surrogates,
            device=args.device,
            n_offspring=args.offspring_size,
            sigma=args.mutation_sigma,
            surrogate_nsga_steps=args.surrogate_nsga_steps,
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
        )

        selected_idx = ranking[: args.k_eval]
        selected_x = offspring_x[selected_idx]
        selected_y = problem.evaluate(selected_x)

        reward = deepic.pareto_improvement_reward(
            previous_archive=archive_y,
            selected_objectives=selected_y,
            ref_point=REFERENCE_POINT,
            epsilon=args.hv_epsilon,
        )

        archive_x, archive_y = demo.update_archive(
            archive_x=archive_x,
            archive_y=archive_y,
            new_x=selected_x,
            new_y=selected_y,
        )

        fronts, _ = demo.fast_non_dominated_sort(archive_y)
        front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
        hv_value = demo.hypervolume_2d(front, REFERENCE_POINT)
        hv_history.append(hv_value)

        print(
            f"Iter {step + 1:02d} | archive={archive_x.shape[0]} | front0={front.shape[0]} | "
            f"HV={hv_value:.6f} | reward={reward:.6f} | pseudo_front={offspring_x.shape[0]}"
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
            seed=args.seed + 200 + step,
        )

    fronts, _ = demo.fast_non_dominated_sort(archive_y)
    final_front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    true_f1 = np.linspace(0.0, 1.0, 200, dtype=np.float32)
    true_f2 = 1.0 - np.sqrt(true_f1)
    true_front = np.stack([true_f1, true_f2], axis=1)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.title(f"{args.dim}D ZDT1 Hypervolume Progress (HV_DeepIC)")
        plt.plot(hv_history, marker="o", label="HV_DeepIC")
        plt.xlabel("Step")
        plt.ylabel("Hypervolume")
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.title(f"{args.dim}D ZDT1 Pareto Front (HV_DeepIC)")
        plt.scatter(final_front[:, 0], final_front[:, 1], s=24, alpha=0.8, label="HV_DeepIC")
        plt.plot(true_front[:, 0], true_front[:, 1], "k-", linewidth=2, label="True Pareto Front")
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.grid(True)
        plt.legend()
        plt.show()

    return {
        "archive_x": archive_x,
        "archive_y": archive_y,
        "final_front": final_front,
        "true_front": true_front,
        "hv_history": hv_history,
        "ref_point": REFERENCE_POINT,
    }


def run_comparison(args):
    deepic = load_or_train_hv_deepic(args)
    shared_init_x = latin_hypercube_sample(
        lower=0.0,
        upper=1.0,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )

    hv_deepic_result = run_hv_deepic_zdt1(
        args,
        deepic=deepic,
        plot=False,
        initial_archive_x=shared_init_x,
    )

    eic_args = build_args_namespace(args)
    eic_result = nsga_eic.run_nsga_eic_problem(
        eic_args,
        problem_name=PROBLEM_NAME,
        plot=False,
        initial_archive_x=shared_init_x,
    )

    print(f"\nHV_DeepIC final HV: {hv_deepic_result['hv_history'][-1]:.6f}")
    print(f"NSGA-EIC final HV: {eic_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {hv_deepic_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title(f"{args.dim}D ZDT1 Hypervolume Comparison")
    plt.plot(hv_deepic_result["hv_history"], marker="o", label="HV_DeepIC")
    plt.plot(eic_result["hv_history"], marker="s", label="NSGA-EIC")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.title(f"{args.dim}D ZDT1 Pareto Front Comparison")
    plt.scatter(hv_deepic_result["final_front"][:, 0], hv_deepic_result["final_front"][:, 1], s=24, alpha=0.8, label="HV_DeepIC")
    plt.scatter(eic_result["final_front"][:, 0], eic_result["final_front"][:, 1], s=24, alpha=0.8, label="NSGA-EIC")
    plt.plot(hv_deepic_result["true_front"][:, 0], hv_deepic_result["true_front"][:, 1], "k-", linewidth=2, label="True Pareto Front")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(True)
    plt.legend()
    plt.show()

    return {"hv_deepic": hv_deepic_result, "nsga_eic": eic_result}


def parse_args():
    parser = argparse.ArgumentParser(description="Train HV_DeepIC on ZDT1 and compare against NSGA-EIC.")
    parser.add_argument("--archive_size", type=int, default=80)
    parser.add_argument("--offspring_size", type=int, default=24)
    parser.add_argument("--k_eval", type=int, default=5)
    parser.add_argument("--max_fe", type=int, default=160)
    parser.add_argument("--mutation_sigma", type=float, default=0.12)
    parser.add_argument("--kan_steps", type=int, default=25)
    parser.add_argument("--kan_hidden", type=int, default=10)
    parser.add_argument("--kan_grid", type=int, default=5)
    parser.add_argument("--deepic_hidden", type=int, default=64)
    parser.add_argument("--deepic_heads", type=int, default=4)
    parser.add_argument("--deepic_ff", type=int, default=128)
    parser.add_argument("--deepic_lr", type=float, default=1e-3)
    parser.add_argument("--deepic_adapt_steps", type=int, default=8)
    parser.add_argument("--surrogate_nsga_steps", type=int, default=40)
    parser.add_argument("--hv_epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--train_only", action="store_true", help="Only train HV_DeepIC on ZDT1.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.train_only:
        train_hv_deepic_zdt1(args)
    else:
        run_comparison(args)


if __name__ == "__main__":
    main()
