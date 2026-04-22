import argparse
import importlib.util
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def load_module(filename: str, module_name: str):
    module_path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nda = load_module("nsga-nda.py", "nsga_nda_module")
eic_base = load_module("nsga-eic.py", "nsga_eic_module")


def parse_args():
    parser = argparse.ArgumentParser(description="Surrogate-only infill baseline with the same setup as NSGA-EIC")
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
    parser.add_argument("--discount", type=float, default=0.99, help="Reward discount/multiplier used during RL updates")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--compare", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-SURR on ZDT1")
    parser.add_argument("--compare_zdt2", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-SURR on ZDT2")
    parser.add_argument("--compare_zdt7", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-SURR on ZDT7")
    parser.add_argument("--compare_dtlz1", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-SURR on DTLZ1")
    parser.add_argument("--compare_dtlz2", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-SURR on DTLZ2")
    parser.add_argument("--compare_dtlz3", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-SURR on DTLZ3")
    parser.add_argument("--compare_dtlz4", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-SURR on DTLZ4")
    parser.add_argument("--compare_dtlz5", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-SURR on DTLZ5")
    parser.add_argument("--compare_dtlz6", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-SURR on DTLZ6")
    parser.add_argument("--compare_dtlz7", action="store_true", help="Compare pre-trained DeepIC-assisted EA against NSGA-SURR on DTLZ7")
    return parser.parse_args()


def select_surr(offspring_pred: np.ndarray, k: int) -> np.ndarray:
    fronts, _ = nda.fast_non_dominated_sort(offspring_pred)
    chosen: list[int] = []

    for front in fronts:
        if not front:
            continue
        remaining = k - len(chosen)
        if remaining <= 0:
            break

        front_idx = np.asarray(front, dtype=np.int64)
        surrogate_score = offspring_pred[front_idx].sum(axis=1)
        order = np.argsort(surrogate_score)
        ordered_front = front_idx[order]
        chosen.extend(ordered_front[:remaining].tolist())

    return np.asarray(chosen[:k], dtype=np.int64)


def run_nsga_surr_problem(
    args,
    problem_name: str,
    plot: bool = True,
    initial_archive_x: Optional[np.ndarray] = None,
):
    nda.set_seed(args.seed)
    if args.max_fe < args.archive_size:
        raise ValueError("max_fe must be at least as large as archive_size.")
    if (args.max_fe - args.archive_size) % args.k_eval != 0:
        raise ValueError("max_fe - archive_size must be divisible by k_eval.")

    problem = nda.ZDTProblem(name=problem_name, dim=args.dim)
    ref_point = eic_base._reference_point(problem_name, args.dim)

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
        archive_x = eic_base.multisource.latin_hypercube_sample(
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
    hv_history: list[float] = []

    fronts, _ = nda.fast_non_dominated_sort(archive_y)
    front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    hv_front = eic_base._normalize_for_hv(problem_name, front, args.dim)
    initial_hv = nda.hypervolume_2d(hv_front, ref_point)
    hv_history.append(initial_hv)
    print(
        f"Init    | archive={archive_x.shape[0]} | "
        f"front0={front.shape[0]} | HV={initial_hv:.6f}"
    )

    for step in range(steps_to_run):
        offspring_x = nda.generate_offspring(
            archive_x=archive_x,
            n_offspring=args.offspring_size,
            lower=problem.lower,
            upper=problem.upper,
            sigma=args.mutation_sigma,
        )
        offspring_pred = nda.predict_with_kan(surrogates, offspring_x, args.device)
        selected_idx = select_surr(offspring_pred=offspring_pred, k=args.k_eval)
        selected_x = offspring_x[selected_idx]
        selected_y = problem.evaluate(selected_x)

        archive_x, archive_y = nda.update_archive(
            archive_x=archive_x,
            archive_y=archive_y,
            new_x=selected_x,
            new_y=selected_y,
        )

        fronts, _ = nda.fast_non_dominated_sort(archive_y)
        front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
        hv_front = eic_base._normalize_for_hv(problem_name, front, args.dim)
        hv_value = nda.hypervolume_2d(hv_front, ref_point)
        hv_history.append(hv_value)

        print(
            f"Iter {step + 1:02d} | archive={archive_x.shape[0]} | "
            f"front0={front.shape[0]} | HV={hv_value:.6f}"
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
    true_front = eic_base._true_front(problem_name)

    print("\nObtained Pareto front:")
    print(np.round(final_front, 6))
    print("\nTrue Pareto front:")
    print(np.round(true_front, 6))

    if plot:
        plt.figure(figsize=(7, 5))
        plt.title(f"{problem_name} Hypervolume Progress (KAN + SURR)")
        plt.plot(hv_history, marker="o")
        plt.xlabel("Step")
        plt.ylabel("Hypervolume")
        plt.grid(True)
        plt.show()
        eic_base._plot_front(f"{problem_name} Pareto Front (KAN + SURR)", final_front, true_front, "Obtained Front")

    return {
        "archive_x": archive_x,
        "archive_y": archive_y,
        "final_front": final_front,
        "true_front": true_front,
        "hv_history": hv_history,
        "ref_point": ref_point,
    }


def run_nsga_surr(args, plot: bool = True):
    return run_nsga_surr_problem(args, problem_name="ZDT1", plot=plot)


def run_comparison_problem(args, problem_name: str, checkpoint_path: Optional[str] = None):
    problem = nda.ZDTProblem(name=problem_name, dim=args.dim)
    shared_init_x = eic_base.multisource.latin_hypercube_sample(
        lower=problem.lower,
        upper=problem.upper,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )
    deepic_result = eic_base.run_deepic_problem(
        args,
        problem_name=problem_name,
        plot=False,
        checkpoint_path=checkpoint_path,
        initial_archive_x=shared_init_x,
    )
    surr_result = run_nsga_surr_problem(args, problem_name=problem_name, plot=False, initial_archive_x=shared_init_x)

    print(f"\nDeepIC-assisted EA final HV: {deepic_result['hv_history'][-1]:.6f}")
    print(f"NSGA-SURR final HV: {surr_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {deepic_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title(f"{problem_name} Hypervolume Comparison")
    plt.plot(deepic_result["hv_history"], marker="o", label="Pre-trained DeepIC-assisted EA")
    plt.plot(surr_result["hv_history"], marker="s", label="NSGA-SURR")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend()
    plt.show()

    eic_base._plot_front_comparison(
        f"{problem_name} Pareto Front Comparison",
        deepic_result["final_front"],
        "DeepIC-assisted EA",
        surr_result["final_front"],
        "NSGA-SURR",
        deepic_result["true_front"],
    )

    return {"deepic": deepic_result, "nsga_surr": surr_result}


def main():
    args = parse_args()

    if args.compare:
        run_comparison_problem(args, problem_name="ZDT1", checkpoint_path="deepic_zdt1.pth")
    elif args.compare_zdt2:
        run_comparison_problem(args, problem_name="ZDT2", checkpoint_path="deepic_zdt.pth")
    elif args.compare_zdt7:
        run_comparison_problem(args, problem_name="ZDT7", checkpoint_path="deepic_zdt.pth")
    elif args.compare_dtlz1:
        run_comparison_problem(args, problem_name="DTLZ1", checkpoint_path="deepic_zdt.pth")
    elif args.compare_dtlz2:
        run_comparison_problem(args, problem_name="DTLZ2", checkpoint_path="deepic_zdt.pth")
    elif args.compare_dtlz3:
        run_comparison_problem(args, problem_name="DTLZ3", checkpoint_path="deepic_zdt.pth")
    elif args.compare_dtlz4:
        run_comparison_problem(args, problem_name="DTLZ4", checkpoint_path="deepic_zdt.pth")
    elif args.compare_dtlz5:
        run_comparison_problem(args, problem_name="DTLZ5", checkpoint_path="deepic_zdt.pth")
    elif args.compare_dtlz6:
        run_comparison_problem(args, problem_name="DTLZ6", checkpoint_path="deepic_zdt.pth")
    elif args.compare_dtlz7:
        run_comparison_problem(args, problem_name="DTLZ7", checkpoint_path="deepic_zdt.pth")
    else:
        run_nsga_surr(args, plot=True)


if __name__ == "__main__":
    main()
