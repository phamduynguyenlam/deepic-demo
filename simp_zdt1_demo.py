import copy
import sys
from pathlib import Path

import numpy as np

from agent.deepic_agent import SimplifiedDeepIC

import demo
import multisource_eva_common as multisource
import deepic_demo as base_demo


TARGET_PROBLEM = "ZDT1"


def _problem_slug(problem_name: str) -> str:
    return problem_name.lower()


def _epoch_checkpoint_path(problem_name: str, epoch_number: int, self_train_only: bool = False) -> Path:
    root = Path(__file__).resolve().parent
    if self_train_only:
        return root / f"simp_{_problem_slug(problem_name)}_self_model_epoch_{epoch_number}.pth"
    return root / f"simp_{_problem_slug(problem_name)}_model_epoch_{epoch_number}.pth"


def _final_model_path(problem_name: str, self_train_only: bool = False) -> Path:
    root = Path(__file__).resolve().parent
    if self_train_only:
        return root / f"simp_deepic_{_problem_slug(problem_name)}_self_only.pth"
    return root / f"simp_deepic_{_problem_slug(problem_name)}_source_mix.pth"


def _reward_log_path(problem_name: str, self_train_only: bool = False) -> Path:
    return multisource.REWARD_LOG_DIR / f"simp_{_problem_slug(problem_name)}_{'demo' if self_train_only else 'eva'}_train_rewards.json"


def _configure_simplified_deepic() -> None:
    demo.DeepICClass = SimplifiedDeepIC
    multisource.demo.DeepICClass = SimplifiedDeepIC
    base_demo.demo.DeepICClass = SimplifiedDeepIC
    base_demo.base.demo.DeepICClass = SimplifiedDeepIC

    multisource._epoch_checkpoint_path = _epoch_checkpoint_path
    multisource._final_model_path = _final_model_path
    multisource._reward_log_path = _reward_log_path
    multisource._script_variant = lambda self_train_only=False: "simp_demo" if self_train_only else "simp_eva"
    multisource._training_label = lambda self_train_only=False: "simp_self_only" if self_train_only else "simp_source_mix"


def _build_random_simplified_deepic(args):
    return SimplifiedDeepIC(
        hidden_dim=args.deepic_hidden,
        n_heads=args.deepic_heads,
        ff_dim=args.deepic_ff,
    ).to(args.device)


def draft1(
    args=None,
    plot: bool = True,
    self_train_only: bool = True,
    initial_archive_x: np.ndarray | None = None,
) -> dict:
    _configure_simplified_deepic()
    if args is None:
        args = multisource.parse_args(TARGET_PROBLEM)

    deepic = multisource.load_or_train_deepic(args, TARGET_PROBLEM, self_train_only=self_train_only)
    problem = multisource.nda.ZDTProblem(name=TARGET_PROBLEM, dim=args.dim)
    ref_point = multisource.nsga_eic._reference_point(TARGET_PROBLEM, args.dim)

    pretrain_entry = multisource.load_or_prepare_kan_surrogate(TARGET_PROBLEM, args.dim, args)
    pretrain_x = pretrain_entry["x"]
    pretrain_y = pretrain_entry["y"]
    surrogates = pretrain_entry["models"]
    print(f"Prepared KAN surrogate on {TARGET_PROBLEM}-{args.dim}D with {pretrain_x.shape[0]} samples.")

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
    uncertainty_x, uncertainty_y = demo.init_uncertainty_archive(archive_x, archive_y)
    gp_surrogates = None
    if demo.surrogate_model_name(args) == "gp":
        gp_surrogates = demo.fit_gp_surrogates(
            archive_x=uncertainty_x,
            archive_y=uncertainty_y,
            seed=args.seed + multisource._stable_seed(89, TARGET_PROBLEM, args.dim),
        )

    true_evals = args.archive_size
    hv_history: list[float] = []
    reward_history: list[float] = []
    k_eval_history: list[int] = []

    surrogate_archive_x, surrogate_archive_y = base_demo._initialize_surrogate_archive(args, problem, surrogates, archive_x)

    fronts, _ = demo.fast_non_dominated_sort(archive_y)
    front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    initial_hv = demo.hypervolume_2d(front, ref_point)
    hv_history.append(initial_hv)
    print(
        f"Init    | archive={archive_x.shape[0]} | "
        f"front0={front.shape[0]} | HV={initial_hv:.6f} | "
        f"surrogate_archive={surrogate_archive_x.shape[0]}"
    )

    step_idx = 0
    while true_evals < args.max_fe:
        current_k_eval = 10 if step_idx == 0 else 1
        current_k_eval = min(current_k_eval, args.max_fe - true_evals, base_demo.SURROGATE_WORKING_SIZE)
        if current_k_eval <= 0:
            break

        surrogate_seed_x = base_demo._select_surrogate_seed_archive(
            archive_x=archive_x,
            archive_y=archive_y,
            previous_surrogate_x=surrogate_archive_x,
            surrogates=surrogates,
            device=args.device,
        )

        offspring_x, offspring_pred = multisource.nsga_eic.generate_nsga2_pseudo_front(
            archive_x=surrogate_seed_x,
            problem=problem,
            surrogates=surrogates,
            device=args.device,
            n_offspring=base_demo.SURROGATE_WORKING_SIZE,
            sigma=args.mutation_sigma,
            surrogate_nsga_steps=args.surrogate_nsga_steps,
            predict_fn=demo.predict_with_kan,
            generate_fn=demo.generate_offspring,
        )
        offspring_x, offspring_pred = base_demo._trim_population(
            offspring_x,
            offspring_pred,
            base_demo.SURROGATE_WORKING_SIZE,
        )
        surrogate_archive_x = offspring_x.copy()
        surrogate_archive_y = offspring_pred.copy()

        if gp_surrogates is not None:
            _, offspring_sigma = demo.predict_with_gp(gp_surrogates, offspring_x)
            offspring_sigma = offspring_sigma.astype(np.float32)
        else:
            archive_pred = demo.predict_with_kan(surrogates, uncertainty_x, args.device).astype(np.float32)
            offspring_sigma = demo.estimate_uncertainty(
                archive_x=uncertainty_x,
                archive_y=uncertainty_y,
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
            top_k=current_k_eval,
        )

        selected_idx = ranking[:current_k_eval]
        selected_x = offspring_x[selected_idx]
        selected_y = problem.evaluate(selected_x)
        reward_value = float(
            multisource._compute_reward(
                previous_front=archive_y,
                selected_objectives=selected_y,
                reward_scheme=int(getattr(args, "reward_scheme", 1)),
                problem_name=TARGET_PROBLEM,
                dim=args.dim,
            )
        )
        reward_history.append(reward_value)
        k_eval_history.append(current_k_eval)

        archive_x, archive_y = demo.update_archive(
            archive_x=archive_x,
            archive_y=archive_y,
            new_x=selected_x,
            new_y=selected_y,
        )
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
            f"Iter {step_idx + 1:02d} | k_eval={current_k_eval} | "
            f"archive={archive_x.shape[0]} | front0={front.shape[0]} | "
            f"HV={hv_value:.6f} | reward={reward_value:.6f} | "
            f"seed_archive={surrogate_seed_x.shape[0]} | surrogate_archive={surrogate_archive_x.shape[0]}"
        )

        true_evals += current_k_eval
        step_idx += 1
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
            seed=args.seed + 300 + step_idx,
        )

    fronts, _ = demo.fast_non_dominated_sort(archive_y)
    final_front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    true_front = multisource.nsga_eic._true_front(TARGET_PROBLEM)

    if plot:
        multisource.plt.figure(figsize=(8, 5))
        multisource.plt.title(f"{args.dim}D {TARGET_PROBLEM} Draft1 Hypervolume")
        multisource.plt.plot(hv_history, marker="o", label="draft1")
        multisource.plt.xlabel("Step")
        multisource.plt.ylabel("Hypervolume")
        multisource.plt.grid(True)
        multisource.plt.legend()
        multisource.plt.show()

        multisource.nsga_eic._plot_front(
            f"{args.dim}D {TARGET_PROBLEM} Draft1 Pareto Front",
            final_front,
            true_front,
            "draft1",
        )

    return {
        "archive_x": archive_x,
        "archive_y": archive_y,
        "final_front": final_front,
        "true_front": true_front,
        "hv_history": hv_history,
        "reward_history": reward_history,
        "k_eval_history": k_eval_history,
        "ref_point": ref_point,
        "surrogate_archive_x": surrogate_archive_x,
        "surrogate_archive_y": surrogate_archive_y,
    }


def test_infer_mean_reward(
    args=None,
    n_runs: int = 10,
    self_train_only: bool = True,
    random_model: bool = False,
) -> dict:
    _configure_simplified_deepic()
    if args is None:
        args = multisource.parse_args(TARGET_PROBLEM)

    if random_model:
        demo.set_seed(int(args.seed))
        deepic = _build_random_simplified_deepic(args)
    else:
        deepic = multisource.load_or_train_deepic(args, TARGET_PROBLEM, self_train_only=self_train_only)
    run_mean_rewards: list[float] = []
    run_hv_histories: list[list[float]] = []

    for run_idx in range(int(n_runs)):
        run_args = copy.deepcopy(args)
        run_args.seed = int(args.seed) + run_idx
        result = multisource.run_saea_deepic_problem(
            run_args,
            target_problem=TARGET_PROBLEM,
            deepic=deepic,
            plot=False,
        )
        reward_history = result["reward_history"]
        hv_history = [float(x) for x in result["hv_history"]]
        mean_reward = float(np.mean(reward_history)) if reward_history else 0.0
        run_mean_rewards.append(mean_reward)
        run_hv_histories.append(hv_history)
        print(
            f"Infer run {run_idx + 1}/{int(n_runs)} | "
            f"seed={run_args.seed} | mean reward={mean_reward:.6f} | "
            f"final HV={hv_history[-1]:.6f}"
        )

    overall_mean_reward = float(np.mean(run_mean_rewards)) if run_mean_rewards else 0.0
    hv_matrix = np.asarray(run_hv_histories, dtype=np.float32)
    mean_hv_per_step = hv_matrix.mean(axis=0).tolist() if run_hv_histories else []
    print(f"Average mean reward across {int(n_runs)} runs: {overall_mean_reward:.6f}")
    for step_idx, mean_hv in enumerate(mean_hv_per_step):
        label = "Init" if step_idx == 0 else f"Step {step_idx:02d}"
        print(f"Mean HV {label}: {float(mean_hv):.6f}")
    return {
        "run_mean_rewards": run_mean_rewards,
        "overall_mean_reward": overall_mean_reward,
        "run_hv_histories": run_hv_histories,
        "mean_hv_per_step": mean_hv_per_step,
    }


def _extract_test_cli_args(argv: list[str]) -> tuple[list[str], bool, int, bool]:
    filtered_argv: list[str] = []
    run_test = False
    test_runs = 10
    random_model = False
    idx = 0

    while idx < len(argv):
        token = argv[idx]
        if token == "--test_infer_mean_reward":
            run_test = True
            idx += 1
            continue
        if token == "--test_runs":
            if idx + 1 >= len(argv):
                raise ValueError("--test_runs requires an integer value.")
            test_runs = int(argv[idx + 1])
            idx += 2
            continue
        if token == "--test_random_model":
            random_model = True
            idx += 1
            continue
        filtered_argv.append(token)
        idx += 1

    return filtered_argv, run_test, test_runs, random_model


def main():
    _configure_simplified_deepic()
    filtered_argv, run_test, test_runs, random_model = _extract_test_cli_args(sys.argv[1:])
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0], *filtered_argv]
    try:
        args = multisource.parse_args(TARGET_PROBLEM)
    finally:
        sys.argv = original_argv

    if args.dim != 30:
        print(f"Warning: expected 30D evaluation for {TARGET_PROBLEM}, but received dim={args.dim}.")

    if args.archive_size != base_demo.INITIAL_SURROGATE_ARCHIVE_SIZE:
        print(
            f"Warning: this demo initializes a surrogate archive of {base_demo.INITIAL_SURROGATE_ARCHIVE_SIZE} "
            f"individuals while archive_size={args.archive_size}."
        )

    if run_test:
        test_infer_mean_reward(
            args=args,
            n_runs=test_runs,
            self_train_only=True,
            random_model=random_model,
        )
    elif args.train_only:
        if args.train_algo == "ppo":
            multisource.train_deepic_multisource_ppo(args, TARGET_PROBLEM, self_train_only=True)
        else:
            multisource.train_deepic_multisource(args, TARGET_PROBLEM, self_train_only=True)
    else:
        base_demo.run_comparison(args, TARGET_PROBLEM, self_train_only=True)


if __name__ == "__main__":
    main()
