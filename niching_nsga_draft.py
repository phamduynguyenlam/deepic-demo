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


nda = load_module("nsga-nda.py", "nsga_nda_module")
nsga_eic = load_module("nsga-eic.py", "nsga_eic_module")
multisource = load_module("multisource_eva_common.py", "multisource_eva_common_module")
zdt1_eva = load_module("zdt1_eva.py", "zdt1_eva_module")


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


def _nd_pbi_select(values: np.ndarray, archive_values: np.ndarray, k: int, focus: str) -> np.ndarray:
    if k >= values.shape[0]:
        return np.arange(values.shape[0], dtype=np.int64)
    if k <= 0:
        return np.array([], dtype=np.int64)

    theta = 2.0 if focus == "convergence" else 8.0
    empty_bonus = 0.0 if focus == "convergence" else 0.5

    chosen: list[int] = []
    remaining = list(range(values.shape[0]))
    archive_fronts, _ = nda.fast_non_dominated_sort(archive_values)
    archive_front = (
        archive_values[np.asarray(archive_fronts[0], dtype=np.int64)]
        if archive_fronts and archive_fronts[0]
        else archive_values
    )

    while len(chosen) < k and remaining:
        selected_values = (
            values[np.asarray(chosen, dtype=np.int64)]
            if chosen
            else np.empty((0, values.shape[1]), dtype=np.float32)
        )
        arnd = np.vstack([archive_front, selected_values]).astype(np.float32)
        candidate_values = values[np.asarray(remaining, dtype=np.int64)]
        reference_values = np.vstack([arnd, candidate_values]).astype(np.float32)

        ref_vectors = _simplex_reference_vectors(values.shape[1], n_partitions=max(12, values.shape[1] * 4))
        arnd_norm = _normalize_for_pbi(arnd, reference_values)
        cand_norm = _normalize_for_pbi(candidate_values, reference_values)

        arnd_assoc, _, arnd_pbi = _pbi_stats(arnd_norm, ref_vectors, theta=theta)
        cand_assoc, cand_d1, cand_pbi = _pbi_stats(cand_norm, ref_vectors, theta=theta)

        best_idx = None
        best_score = -np.inf
        nonempty_refs = set(int(idx) for idx in arnd_assoc.tolist())

        for local_idx, global_idx in enumerate(remaining):
            assoc = int(cand_assoc[local_idx])
            if assoc in nonempty_refs:
                ref_mask = arnd_assoc == assoc
                pbi_min = float(np.min(arnd_pbi[ref_mask]))
                improvement = pbi_min - float(cand_pbi[local_idx])
            else:
                improvement = empty_bonus - float(cand_pbi[local_idx])

            if focus == "convergence":
                score = improvement - 0.05 * float(cand_d1[local_idx])
            else:
                score = improvement + empty_bonus

            if score > best_score:
                best_score = score
                best_idx = global_idx

        if best_idx is None:
            break
        chosen.append(int(best_idx))
        remaining.remove(int(best_idx))

    return np.asarray(chosen[:k], dtype=np.int64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Niching surrogate-assisted NSGA-II with convergence/diversity ND-PBI selection"
    )
    parser.add_argument("--problem", type=str, default="ZDT1")
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--archive_size", type=int, default=100)
    parser.add_argument("--offspring_size", type=int, default=24)
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib comparison plots")
    return parser.parse_args()


def _validate_budget(args, k_eval_total: int) -> None:
    if args.max_fe < args.archive_size:
        raise ValueError("max_fe must be at least as large as archive_size.")
    if (args.max_fe - args.archive_size) % k_eval_total != 0:
        raise ValueError("max_fe - archive_size must be divisible by the total real evaluations per iteration.")


def _fit_surrogates_with_pretrain(problem, args):
    pretrain_x, pretrain_y, surrogates = nda.pre_train_kan_surrogate_for_problem(
        problem=problem,
        device=args.device,
        kan_steps=args.kan_steps,
        hidden_width=args.kan_hidden,
        grid=args.kan_grid,
        seed=args.seed,
    )
    return pretrain_x, pretrain_y, surrogates


def _run_baseline_nsga_eic(args, initial_archive_x: np.ndarray):
    baseline_args = SimpleNamespace(**vars(args))
    baseline_args.k_eval = 4
    _validate_budget(baseline_args, k_eval_total=baseline_args.k_eval)
    return nsga_eic.run_nsga_eic_problem(
        baseline_args,
        problem_name=args.problem,
        plot=False,
        initial_archive_x=initial_archive_x,
    )


def _run_baseline_saea_deepic_zdt1(args, initial_archive_x: np.ndarray):
    baseline_args = SimpleNamespace(**vars(args))
    baseline_args.k_eval = 4
    baseline_args.eval_epoch = getattr(args, "eval_epoch", None)
    baseline_args.deepic_hidden = getattr(args, "deepic_hidden", 64)
    baseline_args.deepic_heads = getattr(args, "deepic_heads", 4)
    baseline_args.deepic_ff = getattr(args, "deepic_ff", 128)
    baseline_args.deepic_lr = getattr(args, "deepic_lr", 1e-4)
    baseline_args.deepic_adapt_steps = getattr(args, "deepic_adapt_steps", 8)
    _validate_budget(baseline_args, k_eval_total=baseline_args.k_eval)
    deepic = zdt1_eva.load_or_train_deepic(baseline_args)
    return multisource.run_saea_deepic_problem(
        baseline_args,
        target_problem=args.problem,
        deepic=deepic,
        plot=False,
        initial_archive_x=initial_archive_x,
    )


def run_niching_nsga(args, plot: bool = True, initial_archive_x: np.ndarray | None = None):
    k_eval_total = 4
    k_eval_branch = 2
    _validate_budget(args, k_eval_total=k_eval_total)
    nda.set_seed(args.seed)

    problem = nda.ZDTProblem(name=args.problem, dim=args.dim)
    ref_point = nsga_eic._reference_point(args.problem, args.dim)
    pretrain_x, pretrain_y, surrogates = _fit_surrogates_with_pretrain(problem, args)

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

    true_evals = args.archive_size
    steps_to_run = (args.max_fe - true_evals) // k_eval_total
    hv_history: list[float] = []

    fronts, _ = nda.fast_non_dominated_sort(archive_y)
    front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    hv_front = nsga_eic._normalize_for_hv(args.problem, front, args.dim)
    initial_hv = nda.hypervolume_2d(hv_front, ref_point)
    hv_history.append(initial_hv)
    print(
        f"[Niching-NSGA] Init    | archive={archive_x.shape[0]} | "
        f"front0={front.shape[0]} | HV={initial_hv:.6f}"
    )

    for step in range(steps_to_run):
        pop1_x, pop1_pred = nsga_eic.generate_nsga2_pseudo_front(
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
        pop2_x, pop2_pred = nsga_eic.generate_nsga2_pseudo_front(
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

        archive_pred = nda.predict_with_kan(surrogates, archive_x, args.device)
        pop1_sigma = nda.estimate_uncertainty(
            archive_x=archive_x,
            archive_y=archive_y,
            archive_pred=archive_pred,
            offspring_x=pop1_x,
        ).astype(np.float32)
        pop2_sigma = nda.estimate_uncertainty(
            archive_x=archive_x,
            archive_y=archive_y,
            archive_pred=archive_pred,
            offspring_x=pop2_x,
        ).astype(np.float32)

        penalized_pop1 = pop1_pred + pop1_sigma
        penalized_pop2 = pop2_pred + pop2_sigma
        selected_pop1 = _nd_pbi_select(penalized_pop1, archive_y, k_eval_branch, focus="convergence")
        selected_pop2 = _nd_pbi_select(penalized_pop2, archive_y, k_eval_branch, focus="diversity")

        selected_x = np.vstack([pop1_x[selected_pop1], pop2_x[selected_pop2]]).astype(np.float32)
        selected_y = problem.evaluate(selected_x)

        archive_x, archive_y = nda.update_archive(
            archive_x=archive_x,
            archive_y=archive_y,
            new_x=selected_x,
            new_y=selected_y,
        )

        fronts, _ = nda.fast_non_dominated_sort(archive_y)
        front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
        hv_front = nsga_eic._normalize_for_hv(args.problem, front, args.dim)
        hv_history.append(nda.hypervolume_2d(hv_front, ref_point))

        print(
            f"[Niching-NSGA] Iter {step + 1:02d} | archive={archive_x.shape[0]} | "
            f"front0={front.shape[0]} | HV={hv_history[-1]:.6f} | "
            f"P1={len(selected_pop1)} conv, P2={len(selected_pop2)} div"
        )

        true_evals += k_eval_total
        combined_x = np.vstack([pretrain_x, archive_x])
        combined_y = np.vstack([pretrain_y, archive_y])
        surrogates = nda.fit_kan_surrogates(
            archive_x=combined_x,
            archive_y=combined_y,
            device=args.device,
            kan_steps=args.kan_steps,
            hidden_width=args.kan_hidden,
            grid=args.kan_grid,
            seed=args.seed + 200 + step,
        )

    fronts, _ = nda.fast_non_dominated_sort(archive_y)
    final_front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    true_front = nsga_eic._true_front(args.problem)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.title(f"{args.problem} Hypervolume Progress")
        plt.plot(hv_history, marker="o", label="Niching-NSGA")
        plt.xlabel("Step")
        plt.ylabel("Hypervolume")
        plt.grid(True)
        plt.legend()
        plt.show()

        nsga_eic._plot_front(
            f"{args.problem} Pareto Front (Niching-NSGA)",
            final_front,
            true_front,
            "Niching-NSGA",
        )

    return {
        "archive_x": archive_x,
        "archive_y": archive_y,
        "final_front": final_front,
        "true_front": true_front,
        "hv_history": hv_history,
        "ref_point": ref_point,
    }


def run_comparison(args):
    problem = nda.ZDTProblem(name=args.problem, dim=args.dim)
    shared_init_x = multisource.latin_hypercube_sample(
        lower=problem.lower,
        upper=problem.upper,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )

    niching_result = run_niching_nsga(args, plot=False, initial_archive_x=shared_init_x)
    baseline_args = SimpleNamespace(**vars(args))
    baseline_result = _run_baseline_nsga_eic(baseline_args, initial_archive_x=shared_init_x)
    deepic_zdt1_result = _run_baseline_saea_deepic_zdt1(baseline_args, initial_archive_x=shared_init_x)

    print(f"\nNiching-NSGA final HV: {niching_result['hv_history'][-1]:.6f}")
    print(f"NSGA-EIC final HV: {baseline_result['hv_history'][-1]:.6f}")
    print(f"SAEA-DeepIC(ZDT1-trained) final HV: {deepic_zdt1_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {niching_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title(f"{args.problem} Hypervolume Comparison")
    plt.plot(niching_result["hv_history"], marker="o", label="Niching-NSGA")
    plt.plot(baseline_result["hv_history"], marker="s", label="NSGA-EIC")
    plt.plot(deepic_zdt1_result["hv_history"], marker="^", label="SAEA-DeepIC (ZDT1-trained)")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend()
    plt.show()
    true_front = niching_result["true_front"]
    if niching_result["final_front"].shape[1] == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"{args.problem} Pareto Front Comparison")
        ax.scatter(
            niching_result["final_front"][:, 0],
            niching_result["final_front"][:, 1],
            niching_result["final_front"][:, 2],
            s=20,
            alpha=0.8,
            label="Niching-NSGA",
        )
        ax.scatter(
            baseline_result["final_front"][:, 0],
            baseline_result["final_front"][:, 1],
            baseline_result["final_front"][:, 2],
            s=20,
            alpha=0.8,
            label="NSGA-EIC",
        )
        ax.scatter(
            deepic_zdt1_result["final_front"][:, 0],
            deepic_zdt1_result["final_front"][:, 1],
            deepic_zdt1_result["final_front"][:, 2],
            s=20,
            alpha=0.8,
            label="SAEA-DeepIC (ZDT1-trained)",
        )
        ax.scatter(true_front[:, 0], true_front[:, 1], true_front[:, 2], s=8, alpha=0.25, label="True Pareto Front")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
        ax.legend()
        plt.show()
    else:
        plt.figure(figsize=(8, 5))
        plt.title(f"{args.problem} Pareto Front Comparison")
        plt.scatter(niching_result["final_front"][:, 0], niching_result["final_front"][:, 1], s=24, alpha=0.8, label="Niching-NSGA")
        plt.scatter(baseline_result["final_front"][:, 0], baseline_result["final_front"][:, 1], s=24, alpha=0.8, label="NSGA-EIC")
        plt.scatter(
            deepic_zdt1_result["final_front"][:, 0],
            deepic_zdt1_result["final_front"][:, 1],
            s=24,
            alpha=0.8,
            label="SAEA-DeepIC (ZDT1-trained)",
        )
        plt.plot(true_front[:, 0], true_front[:, 1], "k-", linewidth=2, label="True Pareto Front")
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    run_comparison(parse_args())
