from pathlib import Path

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


def main():
    _configure_simplified_deepic()
    args = multisource.parse_args(TARGET_PROBLEM)
    if args.dim != 30:
        print(f"Warning: expected 30D evaluation for {TARGET_PROBLEM}, but received dim={args.dim}.")

    if args.archive_size != base_demo.INITIAL_SURROGATE_ARCHIVE_SIZE:
        print(
            f"Warning: this demo initializes a surrogate archive of {base_demo.INITIAL_SURROGATE_ARCHIVE_SIZE} "
            f"individuals while archive_size={args.archive_size}."
        )

    if args.train_only:
        if args.train_algo == "ppo":
            multisource.train_deepic_multisource_ppo(args, TARGET_PROBLEM, self_train_only=True)
        else:
            multisource.train_deepic_multisource(args, TARGET_PROBLEM, self_train_only=True)
    else:
        base_demo.run_comparison(args, TARGET_PROBLEM, self_train_only=True)


if __name__ == "__main__":
    main()
