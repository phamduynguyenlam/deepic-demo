import matplotlib.pyplot as plt
import numpy as np

import demo
import multisource_eva_common as base


DEFAULT_TARGET_PROBLEM = "ZDT1"
INITIAL_SURROGATE_ARCHIVE_SIZE = 80
SURROGATE_WORKING_SIZE = 40


def _is_duplicate_row(row: np.ndarray, reference: np.ndarray, atol: float = 1e-8) -> bool:
    if reference.size == 0:
        return False
    return bool(np.any(np.all(np.isclose(reference, row, atol=atol), axis=1)))


def _filter_unique_rows(candidates: np.ndarray, existing: np.ndarray | None = None) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.size == 0:
        dim = 0 if candidates.ndim == 1 else candidates.shape[1]
        return np.empty((0, dim), dtype=np.float32)

    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)

    filtered: list[np.ndarray] = []
    existing_arr = (
        np.empty((0, candidates.shape[1]), dtype=np.float32)
        if existing is None
        else np.asarray(existing, dtype=np.float32).reshape(-1, candidates.shape[1])
    )

    for row in candidates:
        if _is_duplicate_row(row, existing_arr):
            continue
        if filtered and _is_duplicate_row(row, np.asarray(filtered, dtype=np.float32)):
            continue
        filtered.append(row.astype(np.float32, copy=False))

    if not filtered:
        return np.empty((0, candidates.shape[1]), dtype=np.float32)
    return np.stack(filtered, axis=0).astype(np.float32)


def _trim_population(x: np.ndarray, y: np.ndarray, n_keep: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.shape[0] <= n_keep:
        return x, y
    return base.nsga_eic._nsga2_survival(x, y, n_keep=n_keep)


def _subsample_archive_for_model(
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    n_keep: int = INITIAL_SURROGATE_ARCHIVE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    archive_x = np.asarray(archive_x, dtype=np.float32)
    archive_y = np.asarray(archive_y, dtype=np.float32)

    if archive_x.shape[0] <= int(n_keep):
        return archive_x, archive_y

    selected_x, selected_y = base.nsga_eic._nsga2_survival(
        archive_x,
        archive_y,
        n_keep=int(n_keep),
    )
    return selected_x.astype(np.float32), selected_y.astype(np.float32)


def _select_surrogate_seed_archive(
    archive_x: np.ndarray,
    archive_y: np.ndarray,
) -> np.ndarray:
    archive_x = np.asarray(archive_x, dtype=np.float32)
    archive_y = np.asarray(archive_y, dtype=np.float32)

    if archive_x.shape[0] >= SURROGATE_WORKING_SIZE:
        selected_x, _ = base.nsga_eic._nsga2_survival(
            archive_x,
            archive_y,
            n_keep=SURROGATE_WORKING_SIZE,
        )
        return selected_x.astype(np.float32)

    # In the SAEA-DeepIC demos we always start from a true-evaluated archive
    # that is at least `SURROGATE_WORKING_SIZE`. If you change archive sizing,
    # you must ensure the true archive has enough points.
    raise ValueError(
        f"Unable to assemble {SURROGATE_WORKING_SIZE} surrogate NSGA seeds "
        f"from true archive only (archive={archive_x.shape[0]})."
    )


def _initialize_surrogate_archive(args, problem, surrogates, archive_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    init_size = INITIAL_SURROGATE_ARCHIVE_SIZE
    surrogate_x, surrogate_y = base.nsga_eic.generate_nsga2_pseudo_front(
        archive_x=archive_x,
        problem=problem,
        surrogates=surrogates,
        device=args.device,
        n_offspring=init_size,
        sigma=args.mutation_sigma,
        surrogate_nsga_steps=args.surrogate_nsga_steps,
        predict_fn=demo.predict_with_kan,
        generate_fn=demo.generate_offspring,
    )
    return _trim_population(surrogate_x, surrogate_y, init_size)


def run_saea_deepic_problem(args, target_problem: str, deepic, plot: bool = True, initial_archive_x: np.ndarray = None):
    problem = base.nda.ZDTProblem(name=target_problem, dim=args.dim)
    ref_point = base.nsga_eic._reference_point(target_problem, args.dim)

    surrogate_mode = demo.surrogate_model_name(args)
    pretrain_x = pretrain_y = None
    kan_surrogates = None
    gp_surrogates = None
    if surrogate_mode != "gp":
        pretrain_entry = base.load_or_prepare_kan_surrogate(target_problem, args.dim, args)
        pretrain_x = pretrain_entry["x"]
        pretrain_y = pretrain_entry["y"]
        kan_surrogates = pretrain_entry["models"]
        print(f"Prepared KAN surrogate on {target_problem}-{args.dim}D with {pretrain_x.shape[0]} samples.")

    if initial_archive_x is None:
        archive_x = base.latin_hypercube_sample(
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
    uncertainty_x = uncertainty_y = None
    if surrogate_mode == "gp":
        gp_surrogates = demo.fit_gp_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            seed=args.seed + base._stable_seed(71, target_problem, args.dim),
        )
    else:
        uncertainty_x, uncertainty_y = demo.init_uncertainty_archive(archive_x, archive_y)
    true_evals = args.archive_size
    steps_to_run = (args.max_fe - true_evals) // args.k_eval
    hv_history = []
    reward_history = []

    fronts, _ = demo.fast_non_dominated_sort(archive_y)
    front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    initial_hv = demo.hypervolume_2d(front, ref_point)
    hv_history.append(initial_hv)
    print(
        f"Init    | archive={archive_x.shape[0]} | "
        f"front0={front.shape[0]} | HV={initial_hv:.6f} | "
        f"seed_archive={min(SURROGATE_WORKING_SIZE, archive_x.shape[0])}"
    )

    for step in range(steps_to_run):
        surrogate_seed_x = _select_surrogate_seed_archive(archive_x=archive_x, archive_y=archive_y)

        if surrogate_mode == "gp":
            if gp_surrogates is None:
                raise ValueError("GP surrogate requested but gp_surrogates is None.")
            offspring_x, offspring_pred = base.nsga_eic.generate_nsga2_pseudo_front(
                archive_x=surrogate_seed_x,
                problem=problem,
                surrogates=gp_surrogates,
                device=args.device,
                n_offspring=SURROGATE_WORKING_SIZE,
                sigma=args.mutation_sigma,
                surrogate_nsga_steps=args.surrogate_nsga_steps,
                predict_fn=demo.predict_with_gp_mean,
                generate_fn=demo.generate_offspring,
            )
        else:
            if kan_surrogates is None:
                raise ValueError("KAN surrogate requested but kan_surrogates is None.")
            offspring_x, offspring_pred = base.nsga_eic.generate_nsga2_pseudo_front(
                archive_x=surrogate_seed_x,
                problem=problem,
                surrogates=kan_surrogates,
                device=args.device,
                n_offspring=SURROGATE_WORKING_SIZE,
                sigma=args.mutation_sigma,
                surrogate_nsga_steps=args.surrogate_nsga_steps,
                predict_fn=demo.predict_with_kan,
                generate_fn=demo.generate_offspring,
            )
        offspring_x, offspring_pred = _trim_population(offspring_x, offspring_pred, SURROGATE_WORKING_SIZE)

        if surrogate_mode == "gp":
            offspring_sigma = demo.predict_with_gp_std(gp_surrogates, offspring_x).astype(np.float32)
        else:
            archive_pred = demo.predict_with_kan(kan_surrogates, uncertainty_x, args.device).astype(np.float32)
            offspring_sigma = demo.estimate_uncertainty(
                archive_x=uncertainty_x,
                archive_y=uncertainty_y,
                archive_pred=archive_pred,
                offspring_x=offspring_x,
            ).astype(np.float32)

        progress = float(true_evals / args.max_fe)
        model_archive_x, model_archive_y = _subsample_archive_for_model(
            archive_x=archive_x,
            archive_y=archive_y,
            n_keep=INITIAL_SURROGATE_ARCHIVE_SIZE,
        )
        ranking = demo.infer_deepic_ranking(
            model=deepic,
            archive_x=model_archive_x,
            archive_y=model_archive_y,
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
        if surrogate_mode != "gp":
            uncertainty_x, uncertainty_y = demo.update_uncertainty_archive(
                uncertainty_x=uncertainty_x,
                uncertainty_y=uncertainty_y,
                new_x=selected_x,
                new_y=selected_y,
            )

        fronts, _ = demo.fast_non_dominated_sort(archive_y)
        front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
        hv_value = demo.hypervolume_2d(front, ref_point)
        hv_history.append(hv_value)

        print(
            f"Iter {step + 1:02d} | archive={archive_x.shape[0]} | "
            f"front0={front.shape[0]} | HV={hv_value:.6f} | reward={reward_value:.6f} | "
            f"seed_archive={surrogate_seed_x.shape[0]}"
        )

        true_evals += args.k_eval
        if true_evals >= args.max_fe:
            break

        if surrogate_mode == "gp":
            gp_surrogates = demo.fit_gp_surrogates(
                archive_x=archive_x,
                archive_y=archive_y,
                seed=args.seed + 200 + step,
            )
        else:
            combined_x = np.vstack([pretrain_x, archive_x])
            combined_y = np.vstack([pretrain_y, archive_y])
            kan_surrogates = demo.fit_kan_surrogates(
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
    true_front = base.nsga_eic._true_front(target_problem)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.title(f"{args.dim}D {target_problem} Hypervolume Comparison")
        plt.plot(hv_history, marker="o", label="SAEA-DeepIC")
        plt.xlabel("Step")
        plt.ylabel("Hypervolume")
        plt.grid(True)
        plt.legend()
        plt.show()

        base.nsga_eic._plot_front(
            f"{args.dim}D {target_problem} Pareto Front",
            final_front,
            true_front,
            "SAEA-DeepIC",
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


def run_comparison(args, target_problem: str, self_train_only: bool = False):
    deepic = base.load_or_train_deepic(args, target_problem, self_train_only=self_train_only)
    problem = base.nda.ZDTProblem(name=target_problem, dim=args.dim)
    shared_init_x = base.latin_hypercube_sample(
        lower=problem.lower,
        upper=problem.upper,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )

    deepic_result = run_saea_deepic_problem(
        args,
        target_problem=target_problem,
        deepic=deepic,
        plot=False,
        initial_archive_x=shared_init_x,
    )

    eic_args = base.build_args_namespace(args)
    eic_result = base.nsga_eic.run_nsga_eic_problem(
        eic_args,
        problem_name=target_problem,
        plot=False,
        initial_archive_x=shared_init_x,
    )

    print(f"\nSAEA-DeepIC final HV: {deepic_result['hv_history'][-1]:.6f}")
    print(f"NSGA-EIC final HV: {eic_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {deepic_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title(f"{args.dim}D {target_problem} Hypervolume Comparison")
    plt.plot(deepic_result["hv_history"], marker="o", label="SAEA-DeepIC")
    plt.plot(eic_result["hv_history"], marker="s", label="NSGA-EIC")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend()
    plt.show()

    base.nsga_eic._plot_front_comparison(
        f"{args.dim}D {target_problem} Pareto Front Comparison",
        deepic_result["final_front"],
        "SAEA-DeepIC",
        eic_result["final_front"],
        "NSGA-EIC",
        deepic_result["true_front"],
    )


def main():
    args = base.parse_args(DEFAULT_TARGET_PROBLEM)
    target_problem = str(getattr(args, "problem", DEFAULT_TARGET_PROBLEM))
    if args.dim != 30:
        print(f"Warning: expected 30D evaluation for {target_problem}, but received dim={args.dim}.")

    if args.archive_size < SURROGATE_WORKING_SIZE:
        raise ValueError(
            f"archive_size must be at least {SURROGATE_WORKING_SIZE} when generating offspring from the true archive "
            f"(got archive_size={args.archive_size})."
        )

    if args.train_only:
        if args.train_algo == "ppo":
            base.train_deepic_multisource_ppo(args, target_problem, self_train_only=True)
        else:
            base.train_deepic_multisource(args, target_problem, self_train_only=True)
    else:
        run_comparison(args, target_problem, self_train_only=True)


if __name__ == "__main__":
    main()
