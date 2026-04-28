from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

from agent.deepic_agent import SimplifiedDeepIC

import demo
import multisource_eva_common as multisource
import deepic_demo as base_demo


DEFAULT_TARGET_PROBLEM = "ZDT1"


def _consume_target_problem(default: str = DEFAULT_TARGET_PROBLEM) -> str:
    argv = sys.argv
    for flag in ("--problem", "--target_problem"):
        if flag in argv:
            idx = argv.index(flag)
            if idx + 1 < len(argv):
                problem = str(argv[idx + 1])
                del argv[idx : idx + 2]
                return problem

    if len(argv) > 1 and not str(argv[1]).startswith("-"):
        problem = str(argv[1])
        del argv[1]
        return problem

    return str(default)


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
    label = "demo" if self_train_only else "eva"
    return multisource.REWARD_LOG_DIR / f"simp_{_problem_slug(problem_name)}_{label}_train_rewards.json"


def _configure_simplified_deepic() -> None:
    """Configure the codebase to use SimplifiedDeepIC as the DeepIC backbone.

    This mirrors the original simp_zdt1_demo.py approach by patching the
    shared demo/multisource modules so other helpers keep working unchanged.
    """

    demo.DeepICClass = SimplifiedDeepIC
    multisource.demo.DeepICClass = SimplifiedDeepIC
    base_demo.demo.DeepICClass = SimplifiedDeepIC
    base_demo.base.demo.DeepICClass = SimplifiedDeepIC

    multisource._epoch_checkpoint_path = _epoch_checkpoint_path
    multisource._final_model_path = _final_model_path
    multisource._reward_log_path = _reward_log_path
    multisource._script_variant = lambda self_train_only=False: "simp_demo" if self_train_only else "simp_eva"
    multisource._training_label = (
        lambda self_train_only=False: "simp_self_only" if self_train_only else "simp_source_mix"
    )


def _build_random_simplified_deepic(args):
    return SimplifiedDeepIC(
        hidden_dim=args.deepic_hidden,
        n_heads=args.deepic_heads,
        ff_dim=args.deepic_ff,
    ).to(args.device)


def test_infer_mean_reward(
    args=None,
    target_problem: str = DEFAULT_TARGET_PROBLEM,
    n_runs: int = 10,
    self_train_only: bool = True,
    random_model: bool = False,
) -> dict:
    _configure_simplified_deepic()
    if args is None:
        args = multisource.parse_args(target_problem)

    if random_model:
        demo.set_seed(int(args.seed))
        deepic = _build_random_simplified_deepic(args)
    else:
        deepic = multisource.load_or_train_deepic(args, target_problem, self_train_only=self_train_only)
    run_mean_rewards: list[float] = []
    run_hv_histories: list[list[float]] = []

    for run_idx in range(int(n_runs)):
        run_args = copy.deepcopy(args)
        run_args.seed = int(args.seed) + run_idx
        result = multisource.run_saea_deepic_problem(
            run_args,
            target_problem=target_problem,
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


def _parse_args(target_problem: str):
    args = multisource.parse_args(target_problem)
    if "--train_algo" not in sys.argv[1:]:
        args.train_algo = "ppo"
    return args


def main() -> None:
    _configure_simplified_deepic()
    target_problem = _consume_target_problem()
    filtered_argv, run_test, test_runs, random_model = _extract_test_cli_args(sys.argv[1:])
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0], *filtered_argv]
    try:
        args = _parse_args(target_problem)
    finally:
        sys.argv = original_argv

    if args.dim != 30:
        print(f"Warning: expected 30D evaluation for {target_problem}, but received dim={args.dim}.")

    if args.archive_size != base_demo.INITIAL_SURROGATE_ARCHIVE_SIZE:
        print(
            f"Warning: this demo initializes a surrogate archive of {base_demo.INITIAL_SURROGATE_ARCHIVE_SIZE} "
            f"individuals while archive_size={args.archive_size}."
        )

    if run_test:
        test_infer_mean_reward(
            args=args,
            target_problem=target_problem,
            n_runs=test_runs,
            self_train_only=True,
            random_model=random_model,
        )
    elif args.train_only:
        if args.train_algo == "ppo":
            multisource.train_deepic_multisource_ppo(args, target_problem, self_train_only=True)
        else:
            multisource.train_deepic_multisource(args, target_problem, self_train_only=True)
    else:
        base_demo.run_comparison(args, target_problem, self_train_only=True)


if __name__ == "__main__":
    main()

