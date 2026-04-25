import copy
import sys
from pathlib import Path

import numpy as np

from agent.deepic_agent import SimplifiedDeepIC

import demo
import multisource_eva_common as multisource
import zdt1_demo as base_demo


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
