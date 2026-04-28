from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from agent.icw_agent import ICW
from infill_criteria import CRITERION_NAMES, LOWER_BETTER_CRITERIA, select_indices_from_action

import demo
import multisource_eva_common as multisource
import deepic_demo as base_demo


DEFAULT_TARGET_PROBLEM = "ZDT1"
EPDI_MC_SAMPLES = 128


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


def _pretrain_target_surrogates(args, problem_name: str) -> dict:
    cache: dict = {}
    for dim in multisource.SOURCE_DIMS:
        cache[(problem_name, dim)] = multisource.load_or_prepare_kan_surrogate(problem_name, dim, args)
    return cache


def _problem_slug(problem_name: str) -> str:
    return problem_name.lower()


def _epoch_checkpoint_path(problem_name: str, epoch_number: int, self_train_only: bool = False) -> Path:
    root = Path(__file__).resolve().parent
    if self_train_only:
        return root / f"iwc_{_problem_slug(problem_name)}_self_model_epoch_{epoch_number}.pth"
    return root / f"iwc_{_problem_slug(problem_name)}_model_epoch_{epoch_number}.pth"


def _final_model_path(problem_name: str, self_train_only: bool = False) -> Path:
    root = Path(__file__).resolve().parent
    if self_train_only:
        return root / f"iwc_{_problem_slug(problem_name)}_self_only.pth"
    return root / f"iwc_{_problem_slug(problem_name)}_source_mix.pth"


def _reward_log_path(problem_name: str, self_train_only: bool = False) -> Path:
    label = "demo" if self_train_only else "eva"
    return multisource.REWARD_LOG_DIR / f"iwc_{_problem_slug(problem_name)}_{label}_train_rewards.json"


def _best_reward_model_path(problem_name: str, self_train_only: bool = False) -> Path:
    root = Path(__file__).resolve().parent
    if self_train_only:
        return root / f"iwc_{_problem_slug(problem_name)}_self_only_ppo_best_reward.pth"
    return root / f"iwc_{_problem_slug(problem_name)}_source_mix_ppo_best_reward.pth"


def _build_icw(args) -> ICW:
    return ICW(
        hidden_dim=args.deepic_hidden,
        n_heads=args.deepic_heads,
        ff_dim=args.deepic_ff,
        action_dim=len(CRITERION_NAMES),
    ).to(args.device)


def _criterion_seed(base_seed: int, epoch_idx: int, step_idx: int, problem_name: str, dim: int) -> int:
    return int(base_seed) + epoch_idx * 10000 + step_idx * 137 + multisource._stable_seed(211, problem_name, dim)


def _format_vector(values: np.ndarray) -> str:
    return np.array2string(
        np.asarray(values, dtype=np.float32),
        precision=4,
        separator=", ",
        suppress_small=False,
    )


def _subsample_archive_for_model(
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    n_keep: int,
) -> tuple[np.ndarray, np.ndarray]:
    archive_x = np.asarray(archive_x, dtype=np.float32)
    archive_y = np.asarray(archive_y, dtype=np.float32)
    n_keep = int(n_keep)

    if archive_x.shape[0] != archive_y.shape[0]:
        raise ValueError(
            f"archive_x and archive_y must have the same number of rows, got {archive_x.shape[0]} and {archive_y.shape[0]}."
        )

    if n_keep <= 0:
        raise ValueError(f"n_keep must be positive, got {n_keep}.")

    def _repeat_to_length(arr: np.ndarray, target_rows: int) -> np.ndarray:
        n_rows = int(arr.shape[0])
        if n_rows == target_rows:
            return arr
        if n_rows == 0:
            raise ValueError("Cannot pad an empty archive.")
        idx = np.arange(int(target_rows), dtype=np.int64) % n_rows
        return arr[idx]

    if archive_x.shape[0] < n_keep:
        return _repeat_to_length(archive_x, n_keep), _repeat_to_length(archive_y, n_keep)

    if archive_x.shape[0] == n_keep:
        return archive_x, archive_y

    selected_x, selected_y = multisource.nsga_eic._nsga2_survival(
        archive_x,
        archive_y,
        n_keep=n_keep,
    )
    return selected_x.astype(np.float32), selected_y.astype(np.float32)



# -----------------------------
# Centralized PPO training setup
# -----------------------------
ALL_BBO_PROBLEMS = [
    "ZDT1", "ZDT2", "ZDT3",
    "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7",
]
CENTRALIZED_TRAIN_DIMS = [15, 20]


def _centralized_train_problems(target_problem: str) -> list[str]:
    """Return the 8 training problems used when target_problem is held out."""
    target_upper = str(target_problem).upper()
    problems = [p for p in ALL_BBO_PROBLEMS if p.upper() != target_upper]
    if len(problems) != len(ALL_BBO_PROBLEMS) - 1:
        print(
            f"Warning: target_problem={target_problem!r} is not in ALL_BBO_PROBLEMS; "
            "using all benchmark problems for training."
        )
        problems = list(ALL_BBO_PROBLEMS)
    return problems


def _cpu_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def _shape_group_key(sample: dict) -> tuple:
    """Group samples that can be stacked into one PPO tensor batch."""
    lower = np.asarray(sample["lower"], dtype=np.float32).reshape(-1)
    upper = np.asarray(sample["upper"], dtype=np.float32).reshape(-1)
    return (
        tuple(np.asarray(sample["archive_x"]).shape),
        tuple(np.asarray(sample["archive_y"]).shape),
        tuple(np.asarray(sample["offspring_x"]).shape),
        tuple(np.asarray(sample["offspring_pred"]).shape),
        tuple(np.asarray(sample["offspring_sigma"]).shape),
        tuple(np.asarray(sample["action"]).shape),
        tuple(lower.tolist()),
        tuple(upper.tolist()),
    )


def _split_trajectories_by_shape(trajectories: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for sample in trajectories:
        groups.setdefault(_shape_group_key(sample), []).append(sample)
    return groups


def _merge_loss_stats_weighted(group_stats: list[dict[str, float]]) -> dict[str, float]:
    if not group_stats:
        return {
            "total_loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0,
            "entropy_loss": 0.0, "weight_kl": 0.0, "weight_entropy": 0.0,
            "approx_kl": 0.0, "ratio_mean": 0.0, "ratio_std": 0.0,
            "ratio_min": 0.0, "ratio_max": 0.0, "adv_mean": 0.0,
            "adv_std": 0.0, "adv_min": 0.0, "adv_max": 0.0,
            "logit_reg": 0.0, "updates": 0.0,
        }
    total_updates = sum(float(s.get("updates", 0.0)) for s in group_stats)
    if total_updates <= 0:
        total_updates = float(len(group_stats))
        weights = [1.0 for _ in group_stats]
    else:
        weights = [float(s.get("updates", 0.0)) for s in group_stats]

    keys = [
        "total_loss", "actor_loss", "critic_loss", "entropy_loss", "weight_kl",
        "weight_entropy", "approx_kl", "ratio_mean", "ratio_std", "ratio_min",
        "ratio_max", "adv_mean", "adv_std", "adv_min", "adv_max", "logit_reg",
    ]
    merged = {k: 0.0 for k in keys}
    for s, w in zip(group_stats, weights):
        for k in keys:
            merged[k] += float(s.get(k, 0.0)) * w
    for k in keys:
        merged[k] /= max(total_updates, 1e-12)
    merged["updates"] = total_updates
    return merged


def _update_icw_from_centralized_ppo_groups(
    model: ICW,
    optimizer,
    trajectories: list[dict],
    device: str,
    ppo_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    weight_kl_coef: float,
    grad_clip: float | None,
    target_kl: float | None,
    adv_clip: float = 2.0,
) -> tuple[dict[str, float], dict[str, int]]:
    """Update PPO on all collected trajectories, split only when tensor shapes differ."""
    grouped = _split_trajectories_by_shape(trajectories)
    group_sizes = {str(k): len(v) for k, v in grouped.items()}
    stats_per_group: list[dict[str, float]] = []
    # Largest groups first usually gives the most stable update early.
    for _, group_samples in sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True):
        stats = _update_icw_from_episode_ppo(
            model=model,
            optimizer=optimizer,
            episode_trajectory=group_samples,
            device=device,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            clip_eps=clip_eps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            weight_kl_coef=weight_kl_coef,
            grad_clip=grad_clip,
            target_kl=target_kl,
            adv_clip=adv_clip,
        )
        stats_per_group.append(stats)
    return _merge_loss_stats_weighted(stats_per_group), group_sizes


def _collect_one_icw_env_trajectory(
    args,
    model_state_dict: dict[str, torch.Tensor],
    problem_name: str,
    dim: int,
    epoch: int,
) -> tuple[list[dict], list[dict], dict]:
    """Collect one on-policy trajectory for one (problem, dim) environment.

    This function is intentionally self-contained so it can be called in a CPU
    worker process. It returns raw numpy arrays only; PPO update happens in the
    main process.
    """
    # Avoid CPU oversubscription when several worker processes are used.
    torch.set_num_threads(1)

    worker_args = args
    worker_args.device = "cpu" if str(getattr(args, "device", "cpu")).startswith("cuda") else args.device

    model = _build_icw(worker_args)
    model.load_state_dict(model_state_dict)
    model.eval()

    entry = multisource.load_or_prepare_kan_surrogate(problem_name, dim, worker_args)
    problem = entry["problem"]
    kan_surrogates = entry["models"]

    trajectory: list[dict] = []
    reward_records: list[dict] = []
    surrogate_mode = multisource._surrogate_model_name(worker_args)

    archive_x = multisource.latin_hypercube_sample(
        lower=problem.lower,
        upper=problem.upper,
        n_samples=worker_args.archive_size,
        dim=dim,
        seed=worker_args.seed + epoch * 10000 + multisource._stable_seed(0, problem_name, dim),
    )
    archive_y = problem.evaluate(archive_x)

    uncertainty_x = uncertainty_y = None
    gp_surrogates = None
    if surrogate_mode == "gp":
        gp_surrogates = demo.fit_gp_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            seed=worker_args.seed + epoch * 10000 + multisource._stable_seed(17, problem_name, dim),
        )
    else:
        uncertainty_x, uncertainty_y = demo.init_uncertainty_archive(archive_x, archive_y)

    true_evals = worker_args.archive_size
    remaining_budget = worker_args.max_fe - true_evals
    steps_to_run = remaining_budget // worker_args.k_eval

    for step in range(steps_to_run):
        archive_x_t = archive_x.copy()
        archive_y_t = archive_y.copy()

        if surrogate_mode == "gp":
            if gp_surrogates is None:
                raise ValueError("GP surrogate requested but gp_surrogates is None.")
            offspring_x, offspring_pred = multisource.nsga_eic.generate_nsga2_pseudo_front(
                archive_x=archive_x_t,
                problem=problem,
                surrogates=gp_surrogates,
                device=worker_args.device,
                n_offspring=worker_args.offspring_size,
                sigma=worker_args.mutation_sigma,
                surrogate_nsga_steps=worker_args.surrogate_nsga_steps,
                predict_fn=demo.predict_with_gp_mean,
                generate_fn=demo.generate_offspring,
            )
            offspring_sigma = demo.predict_with_gp_std(gp_surrogates, offspring_x).astype(np.float32)
        else:
            offspring_x, offspring_pred = multisource.nsga_eic.generate_nsga2_pseudo_front(
                archive_x=archive_x_t,
                problem=problem,
                surrogates=kan_surrogates,
                device=worker_args.device,
                n_offspring=worker_args.offspring_size,
                sigma=worker_args.mutation_sigma,
                surrogate_nsga_steps=worker_args.surrogate_nsga_steps,
                predict_fn=demo.predict_with_kan,
                generate_fn=demo.generate_offspring,
            )
            archive_pred = demo.predict_with_kan(kan_surrogates, uncertainty_x, worker_args.device).astype(np.float32)
            offspring_sigma = demo.estimate_uncertainty(
                archive_x=uncertainty_x,
                archive_y=uncertainty_y,
                archive_pred=archive_pred,
                offspring_x=offspring_x,
            ).astype(np.float32)

        progress = float(true_evals / worker_args.max_fe)
        model_archive_x_t, model_archive_y_t = _subsample_archive_for_model(
            archive_x=archive_x_t,
            archive_y=archive_y_t,
            n_keep=int(worker_args.archive_size),
        )

        with torch.no_grad():
            act_out = model.act(
                x_true=torch.as_tensor(model_archive_x_t, dtype=torch.float32, device=worker_args.device),
                y_true=torch.as_tensor(model_archive_y_t, dtype=torch.float32, device=worker_args.device),
                x_sur=torch.as_tensor(offspring_x, dtype=torch.float32, device=worker_args.device),
                y_sur=torch.as_tensor(offspring_pred, dtype=torch.float32, device=worker_args.device),
                sigma_sur=torch.as_tensor(offspring_sigma, dtype=torch.float32, device=worker_args.device),
                progress=progress,
                lower_bound=problem.lower,
                upper_bound=problem.upper,
                deterministic=False,
            )

        current_value = float(act_out["value"][0].detach().cpu())
        if trajectory:
            trajectory[-1]["next_value"] = current_value
            trajectory[-1]["done"] = 0.0

        selected_idx, _, _ = select_indices_from_action(  # type: ignore
            action=act_out["action_weights"][0].detach().cpu(),
            archive_y=archive_y_t,
            offspring_pred=offspring_pred,
            offspring_sigma=offspring_sigma,
            k_eval=worker_args.k_eval,
            seed=_criterion_seed(worker_args.seed, epoch, step, problem_name, dim),
            epdi_mc_samples=EPDI_MC_SAMPLES,
        )
        selected_x = offspring_x[selected_idx]
        selected_y = problem.evaluate(selected_x)

        reward_value = float(
            multisource._compute_reward(
                previous_front=archive_y_t,
                selected_objectives=selected_y,
                reward_scheme=int(getattr(worker_args, "reward_scheme", 1)),
                problem_name=problem_name,
                dim=dim,
            )
        )
        reward_records.append(
            {
                "epoch": epoch + 1,
                "problem": problem_name,
                "dim": int(dim),
                "step": step + 1,
                "progress": float(progress),
                "reward": reward_value,
                "train_algo": "centralized_ppo",
            }
        )

        trajectory.append(
            {
                "problem": problem_name,
                "dim": int(dim),
                "archive_x": model_archive_x_t.astype(np.float32, copy=False),
                "archive_y": model_archive_y_t.astype(np.float32, copy=False),
                "offspring_x": offspring_x.astype(np.float32, copy=False),
                "offspring_pred": offspring_pred.astype(np.float32, copy=False),
                "offspring_sigma": offspring_sigma.astype(np.float32, copy=False),
                "progress": float(progress),
                "lower": np.asarray(problem.lower, dtype=np.float32),
                "upper": np.asarray(problem.upper, dtype=np.float32),
                "action": act_out["action"][0].detach().cpu().numpy().astype(np.float32, copy=False),
                "action_weights": act_out["action_weights"][0].detach().cpu().numpy().astype(np.float32, copy=False),
                "old_logprob": float(act_out["logprob"][0].detach().cpu()),
                "value": current_value,
                "next_value": 0.0,
                "done": 1.0,
                "reward": reward_value,
            }
        )

        archive_x, archive_y = demo.update_archive(
            archive_x=archive_x_t,
            archive_y=archive_y_t,
            new_x=selected_x,
            new_y=selected_y,
        )
        if surrogate_mode == "gp":
            gp_surrogates = demo.fit_gp_surrogates(
                archive_x=archive_x,
                archive_y=archive_y,
                seed=worker_args.seed + epoch * 10000 + multisource._stable_seed(17, problem_name, dim) + step + 1,
            )
        else:
            uncertainty_x, uncertainty_y = demo.update_uncertainty_archive(
                uncertainty_x=uncertainty_x,
                uncertainty_y=uncertainty_y,
                new_x=selected_x,
                new_y=selected_y,
            )

        true_evals += selected_x.shape[0]
        if true_evals >= worker_args.max_fe:
            break

    multisource._attach_ppo_targets(
        trajectory,
        discount=float(worker_args.discount),
        gae_lambda=float(getattr(worker_args, "ppo_gae_lambda", 0.95)),
    )
    summary = {
        "problem": problem_name,
        "dim": int(dim),
        "true_evals": int(true_evals),
        "best_obj1": float(np.min(archive_y[:, 0])),
        "steps": int(len(trajectory)),
        "mean_reward": float(np.mean([s["reward"] for s in trajectory])) if trajectory else 0.0,
    }
    return trajectory, reward_records, summary

def _icw_ppo_loss(
    agent: ICW,
    batch: dict[str, torch.Tensor],
    clip_eps: float = 0.2,
    value_coef: float = 0.1,
    entropy_coef: float = 0.01,
    weight_kl_coef: float = 0.0,
) -> dict[str, torch.Tensor]:
    eval_out = agent.evaluate_actions(
        x_true=batch["x_true"],
        y_true=batch["y_true"],
        x_sur=batch["x_sur"],
        y_sur=batch["y_sur"],
        sigma_sur=batch["sigma_sur"],
        progress=batch["progress"],
        lower_bound=batch["lower_bound"],
        upper_bound=batch["upper_bound"],
        action=batch["action"],
    )

    new_logprob = eval_out["logprob"]
    entropy = eval_out["entropy"]
    value = eval_out["value"]
    old_logprob = batch["old_logprob"]
    advantages = batch["advantages"]
    returns = batch["returns"]
    logit_reg = eval_out.get("logit_reg", torch.zeros((), device=value.device))

    ratio = torch.exp(new_logprob - old_logprob)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate_1 = ratio * advantages
    surrogate_2 = clipped_ratio * advantages
    actor_loss = -torch.mean(torch.minimum(surrogate_1, surrogate_2))
    critic_loss = F.mse_loss(value, returns)
    entropy_loss = -torch.mean(entropy)

    action_weights = eval_out.get("action_weights", None)
    weight_kl = torch.zeros((), device=value.device, dtype=value.dtype)
    weight_entropy = torch.zeros_like(weight_kl)
    if action_weights is not None:
        weights = action_weights.to(device=value.device, dtype=value.dtype).clamp_min(1e-8)
        weight_entropy = -torch.sum(weights * torch.log(weights), dim=-1).mean()
        # KL(weights || uniform) = sum_i w_i log(w_i * action_dim)
        weight_kl = torch.sum(weights * torch.log(weights * float(weights.size(-1))), dim=-1).mean()
    total_loss = (
        actor_loss
        + value_coef * critic_loss
        + entropy_coef * entropy_loss
        + float(weight_kl_coef) * weight_kl
        + agent.logit_reg_coef * logit_reg
    )
    approx_kl = torch.mean(old_logprob - new_logprob)
    
    return {
        "total_loss": total_loss,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "entropy_loss": entropy_loss,
        "weight_kl": weight_kl,
        "weight_entropy": weight_entropy,
        "approx_kl": approx_kl,
        "ratio_mean": torch.mean(ratio),
        "ratio_std": (
            torch.std(ratio, unbiased=False)
            if ratio.numel() > 1
            else torch.zeros((), device=ratio.device, dtype=ratio.dtype)
        ),
        "ratio_min": torch.min(ratio),
        "ratio_max": torch.max(ratio),
        "logit_reg": logit_reg,
    }


def _update_icw_from_episode_ppo(
    model: ICW,
    optimizer,
    episode_trajectory: list[dict],
    device: str,
    ppo_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    weight_kl_coef: float,
    grad_clip: float | None,
    target_kl: float | None,
    adv_clip: float = 2.0,
) -> dict[str, float]:
    if not episode_trajectory:
        return {
            "total_loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy_loss": 0.0,
            "weight_kl": 0.0,
            "weight_entropy": 0.0,
            "approx_kl": 0.0,
            "ratio_mean": 0.0,
            "ratio_std": 0.0,
            "ratio_min": 0.0,
            "ratio_max": 0.0,
            "adv_mean": 0.0,
            "adv_std": 0.0,
            "adv_min": 0.0,
            "adv_max": 0.0,
            "logit_reg": 0.0,
            "updates": 0.0,
        }

    model.train()
    trajectory_size = len(episode_trajectory)
    archive_x = torch.as_tensor(
        np.stack([np.asarray(sample["archive_x"], dtype=np.float32) for sample in episode_trajectory], axis=0),
        dtype=torch.float32,
        device=device,
    )
    archive_y = torch.as_tensor(
        np.stack([np.asarray(sample["archive_y"], dtype=np.float32) for sample in episode_trajectory], axis=0),
        dtype=torch.float32,
        device=device,
    )
    offspring_x = torch.as_tensor(
        np.stack([np.asarray(sample["offspring_x"], dtype=np.float32) for sample in episode_trajectory], axis=0),
        dtype=torch.float32,
        device=device,
    )
    offspring_pred = torch.as_tensor(
        np.stack([np.asarray(sample["offspring_pred"], dtype=np.float32) for sample in episode_trajectory], axis=0),
        dtype=torch.float32,
        device=device,
    )
    offspring_sigma = torch.as_tensor(
        np.stack([np.asarray(sample["offspring_sigma"], dtype=np.float32) for sample in episode_trajectory], axis=0),
        dtype=torch.float32,
        device=device,
    )
    progress = torch.as_tensor(
        [float(sample["progress"]) for sample in episode_trajectory],
        dtype=torch.float32,
        device=device,
    ).reshape(trajectory_size, -1)
    actions = torch.as_tensor(
        np.stack([np.asarray(sample["action"], dtype=np.float32) for sample in episode_trajectory], axis=0),
        dtype=torch.float32,
        device=device,
    )
    old_logprob = torch.as_tensor(
        [float(sample["old_logprob"]) for sample in episode_trajectory],
        dtype=torch.float32,
        device=device,
    )
    advantages = torch.as_tensor(
        [float(sample["advantage"]) for sample in episode_trajectory],
        dtype=torch.float32,
        device=device,
    )
    returns = torch.as_tensor(
        [float(sample["return"]) for sample in episode_trajectory],
        dtype=torch.float32,
        device=device,
    )
    if trajectory_size > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    if adv_clip is not None and float(adv_clip) > 0.0:
        advantages = advantages.clamp(-float(adv_clip), float(adv_clip))

    stats = {
        "total_loss": 0.0,
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy_loss": 0.0,
        "weight_kl": 0.0,
        "weight_entropy": 0.0,
        "approx_kl": 0.0,
        "ratio_mean": 0.0,
        "ratio_std": 0.0,
        "ratio_min": 0.0,
        "ratio_max": 0.0,
        "adv_mean": 0.0,
        "adv_std": 0.0,
        "adv_min": 0.0,
        "adv_max": 0.0,
        "logit_reg": 0.0,
        "updates": 0.0,
    }

    lower_bound = episode_trajectory[0]["lower"]
    upper_bound = episode_trajectory[0]["upper"]

    stop_updates = False
    for _ in range(max(int(ppo_epochs), 1)):
        perm = torch.randperm(trajectory_size, device=device)
        for start in range(0, trajectory_size, max(int(minibatch_size), 1)):
            mb_idx = perm[start : start + max(int(minibatch_size), 1)]
            if mb_idx.numel() == 0:
                continue

            batch = {
                "x_true": archive_x[mb_idx],
                "y_true": archive_y[mb_idx],
                "x_sur": offspring_x[mb_idx],
                "y_sur": offspring_pred[mb_idx],
                "sigma_sur": offspring_sigma[mb_idx],
                "progress": progress[mb_idx],
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "action": actions[mb_idx],
                "old_logprob": old_logprob[mb_idx],
                "advantages": advantages[mb_idx],
                "returns": returns[mb_idx],
            }
            loss_dict = _icw_ppo_loss(
                agent=model,
                batch=batch,
                clip_eps=clip_eps,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                weight_kl_coef=weight_kl_coef,
            )
            optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            if grad_clip is not None and float(grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            optimizer.step()

            adv_slice = advantages[mb_idx]
            stats["total_loss"] += float(loss_dict["total_loss"].detach().cpu())
            stats["actor_loss"] += float(loss_dict["actor_loss"].detach().cpu())
            stats["critic_loss"] += float(loss_dict["critic_loss"].detach().cpu())
            stats["entropy_loss"] += float(loss_dict["entropy_loss"].detach().cpu())
            stats["weight_kl"] += float(loss_dict["weight_kl"].detach().cpu())
            stats["weight_entropy"] += float(loss_dict["weight_entropy"].detach().cpu())
            if "logit_reg" in loss_dict:
                stats["logit_reg"] += float(loss_dict["logit_reg"].detach().cpu())
            stats["approx_kl"] += float(loss_dict["approx_kl"].detach().cpu())
            stats["ratio_mean"] += float(loss_dict["ratio_mean"].detach().cpu())
            stats["ratio_std"] += float(loss_dict["ratio_std"].detach().cpu())
            stats["ratio_min"] += float(loss_dict["ratio_min"].detach().cpu())
            stats["ratio_max"] += float(loss_dict["ratio_max"].detach().cpu())
            stats["adv_mean"] += float(adv_slice.mean().detach().cpu())
            stats["adv_std"] += float(adv_slice.std(unbiased=False).detach().cpu()) if adv_slice.numel() > 1 else 0.0
            stats["adv_min"] += float(adv_slice.min().detach().cpu())
            stats["adv_max"] += float(adv_slice.max().detach().cpu())
            stats["updates"] += 1.0

            if target_kl is not None and float(target_kl) > 0.0:
                approx_kl = float(loss_dict["approx_kl"].detach().cpu())
                if abs(approx_kl) > float(target_kl):
                    stop_updates = True
                    break
        if stop_updates:
            break

    if stats["updates"] > 0:
        for key in [
            "total_loss",
            "actor_loss",
            "critic_loss",
            "entropy_loss",
            "weight_kl",
            "weight_entropy",
            "approx_kl",
            "ratio_mean",
            "ratio_std",
            "ratio_min",
            "ratio_max",
            "adv_mean",
            "adv_std",
            "adv_min",
            "adv_max",
            "logit_reg",
        ]:
            stats[key] /= stats["updates"]

    return stats



def train_icw_multisource_ppo(args, target_problem: str, self_train_only: bool = False) -> ICW:
    """Centralized PPO for ICW.

    Hold out target_problem. Train on the remaining 8 benchmark problems and
    two dimensions [15, 20], i.e. 16 environments per epoch. Rollouts are
    collected with the same frozen policy, then grouped by tensor shape for PPO
    updates. This keeps PPO on-policy while avoiding single-sample groups.
    """
    demo.set_seed(args.seed)
    reward_records: list[dict] = []
    epoch_mean_rewards: list[float] = []
    epoch_mean_total_losses: list[float] = []
    model_path = _final_model_path(target_problem, self_train_only=self_train_only)
    reward_log_path = _reward_log_path(target_problem, self_train_only=self_train_only)
    best_reward_model_path = _best_reward_model_path(target_problem, self_train_only=self_train_only)
    best_epoch_mean_reward = float("-inf")

    # Kaggle free CPU: avoid oversubscription. Increase explicitly if needed.
    torch.set_num_threads(int(getattr(args, "torch_num_threads", 2)))

    model = _build_icw(args)

    ppo_actor_lr = 1e-4
    ppo_critic_lr = 5e-5
    ppo_epochs = 4
    ppo_minibatch_size = int(getattr(args, "ppo_minibatch_size", 32))
    ppo_clip_eps = 0.05
    ppo_value_coef = 0.05
    ppo_entropy_coef = float(getattr(args, "ppo_entropy_coef", 0.01))
    ppo_weight_kl_coef = float(getattr(args, "icw_weight_kl_coef", 0.01))
    ppo_gae_lambda = float(getattr(args, "ppo_gae_lambda", 0.95))
    ppo_grad_clip = float(getattr(args, "ppo_grad_clip", 1.0))
    ppo_target_kl = 0.005
    ppo_adv_clip = float(getattr(args, "ppo_adv_clip", 2.0))

    train_problems = _centralized_train_problems(target_problem)
    configured_train_dims = getattr(args, "centralized_train_dims", None)
    train_dims = list(configured_train_dims) if configured_train_dims else list(CENTRALIZED_TRAIN_DIMS)
    train_envs = [(p, int(d)) for p in train_problems for d in train_dims]
    parallel_workers = int(getattr(args, "ppo_parallel_workers", 1))
    # On Kaggle CPU, 1-2 workers are usually safer than many workers because GP/KAN
    # and NSGA inner loops can oversubscribe BLAS/OpenMP threads.
    parallel_workers = max(1, min(parallel_workers, len(train_envs)))

    print(
        f"Training config (Centralized PPO) | surrogate_nsga_steps={args.surrogate_nsga_steps} | "
        f"discount={args.discount:.4f} | ppo_epochs={ppo_epochs} | ppo_clip_eps={ppo_clip_eps:.3f} | "
        f"actor_lr={ppo_actor_lr:.1e} | critic_lr={ppo_critic_lr:.1e} | vf_coef={ppo_value_coef:.3f} | "
        f"target_kl={ppo_target_kl:.4f} | weight_kl_coef={ppo_weight_kl_coef:.4f} | "
        f"adv_clip={ppo_adv_clip:.2f} | reward_scheme={getattr(args, 'reward_scheme', 1)} | "
        f"surrogate_model={multisource._surrogate_model_name(args)} | heldout={target_problem} | "
        f"train_envs={len(train_envs)} ({len(train_problems)} problems × {len(train_dims)} dims) | "
        f"parallel_workers={parallel_workers}"
    )
    print(f"Training problems: {train_problems}")
    print(f"Training dims: {train_dims}")

    # Warm up / prepare surrogate cache once in the main process. This avoids
    # multiple workers trying to create the same cached file at the same time.
    for p, d in train_envs:
        multisource.load_or_prepare_kan_surrogate(p, d, args)

    optimizer = multisource._build_ppo_optimizer(
        model=model,
        actor_lr=ppo_actor_lr,
        critic_lr=ppo_critic_lr,
    )
    if args.start_epoch > 0:
        checkpoint_path = _epoch_checkpoint_path(target_problem, args.start_epoch, self_train_only=self_train_only)
        if checkpoint_path.exists():
            model.load_state_dict(multisource._torch_load(checkpoint_path, args.device))
            print(f"Loaded model from {checkpoint_path.name}")
        else:
            print(f"Checkpoint {checkpoint_path.name} not found, starting from scratch")

    for epoch in range(args.start_epoch, multisource.TRAIN_EPOCHS):
        print(f"Epoch {epoch + 1}/{multisource.TRAIN_EPOCHS}")
        epoch_trajectories: list[dict] = []
        epoch_reward_records: list[dict] = []
        env_summaries: list[dict] = []

        # Freeze the rollout policy for this epoch. All envs collect on-policy
        # data from the same parameters before any PPO update happens.
        rollout_state = _cpu_state_dict(model.state_dict())

        if parallel_workers == 1:
            for problem_name, dim in train_envs:
                traj, records, summary = _collect_one_icw_env_trajectory(
                    args=args,
                    model_state_dict=rollout_state,
                    problem_name=problem_name,
                    dim=int(dim),
                    epoch=epoch,
                )
                epoch_trajectories.extend(traj)
                epoch_reward_records.extend(records)
                env_summaries.append(summary)
                print(
                    f"{problem_name}-{dim}D epoch {epoch + 1} done, "
                    f"true_evals={summary['true_evals']}, best_obj1={summary['best_obj1']:.6f}, "
                    f"steps={summary['steps']}, mean_reward={summary['mean_reward']:.6f}"
                )
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import copy

            # Use CPU in workers. The main process can still update on args.device.
            worker_args = copy.copy(args)
            worker_args.device = "cpu"
            with ProcessPoolExecutor(max_workers=parallel_workers) as ex:
                futures = [
                    ex.submit(
                        _collect_one_icw_env_trajectory,
                        worker_args,
                        rollout_state,
                        problem_name,
                        int(dim),
                        epoch,
                    )
                    for problem_name, dim in train_envs
                ]
                for fut in as_completed(futures):
                    traj, records, summary = fut.result()
                    epoch_trajectories.extend(traj)
                    epoch_reward_records.extend(records)
                    env_summaries.append(summary)
                    print(
                        f"{summary['problem']}-{summary['dim']}D epoch {epoch + 1} done, "
                        f"true_evals={summary['true_evals']}, best_obj1={summary['best_obj1']:.6f}, "
                        f"steps={summary['steps']}, mean_reward={summary['mean_reward']:.6f}"
                    )

        reward_records.extend(epoch_reward_records)
        epoch_rewards = [float(r["reward"]) for r in epoch_reward_records]

        if epoch_trajectories:
            action_matrix = np.stack([np.asarray(s["action"], dtype=np.float32) for s in epoch_trajectories], axis=0)
            weight_matrix = np.stack([np.asarray(s["action_weights"], dtype=np.float32) for s in epoch_trajectories], axis=0)
            action_mean, action_std = action_matrix.mean(axis=0), action_matrix.std(axis=0)
            weight_mean, weight_std = weight_matrix.mean(axis=0), weight_matrix.std(axis=0)

            loss_stats, group_sizes = _update_icw_from_centralized_ppo_groups(
                model=model,
                optimizer=optimizer,
                trajectories=epoch_trajectories,
                device=args.device,
                ppo_epochs=ppo_epochs,
                minibatch_size=ppo_minibatch_size,
                clip_eps=ppo_clip_eps,
                value_coef=ppo_value_coef,
                entropy_coef=ppo_entropy_coef,
                weight_kl_coef=ppo_weight_kl_coef,
                grad_clip=ppo_grad_clip,
                target_kl=ppo_target_kl,
                adv_clip=ppo_adv_clip,
            )
            epoch_mean_total_losses.append(loss_stats["total_loss"])

            compact_groups = sorted(group_sizes.values(), reverse=True)
            print(
                f"Centralized PPO update | envs={len(train_envs)} | rollout_steps={len(epoch_trajectories)} | "
                f"shape_groups={len(group_sizes)} | group_sizes={compact_groups} | "
                f"total_loss={loss_stats['total_loss']:.6f}, actor={loss_stats['actor_loss']:.6f}, "
                f"critic={loss_stats['critic_loss']:.6f}, entropy={loss_stats['entropy_loss']:.6f}, "
                f"approx_kl={loss_stats['approx_kl']:.6f}, weight_kl={loss_stats.get('weight_kl', 0.0):.6f}, "
                f"weight_ent={loss_stats.get('weight_entropy', 0.0):.6f}, ratio_mean={loss_stats['ratio_mean']:.6f}, "
                f"ratio_std={loss_stats['ratio_std']:.6f}, ratio_min={loss_stats['ratio_min']:.6f}, "
                f"ratio_max={loss_stats['ratio_max']:.6f}, adv_mean={loss_stats['adv_mean']:.6f}, "
                f"adv_std={loss_stats['adv_std']:.6f}, adv_min={loss_stats['adv_min']:.6f}, "
                f"adv_max={loss_stats['adv_max']:.6f}, logit_reg={loss_stats.get('logit_reg', 0.0):.6f}, "
                f"action_mean={_format_vector(action_mean)}, action_std={_format_vector(action_std)} | "
                f"weight_mean={_format_vector(weight_mean)}, weight_std={_format_vector(weight_std)}"
            )
        else:
            epoch_mean_total_losses.append(0.0)

        epoch_mean = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        epoch_mean_loss = float(epoch_mean_total_losses[-1]) if epoch_mean_total_losses else 0.0
        epoch_mean_rewards.append(epoch_mean)
        print(f"Epoch {epoch + 1} mean reward: {epoch_mean:.6f}")
        print(f"Epoch {epoch + 1} mean PPO total loss: {epoch_mean_loss:.6f}")

        if epoch_mean > best_epoch_mean_reward:
            best_epoch_mean_reward = epoch_mean
            torch.save(model.state_dict(), best_reward_model_path)
            print(
                f"New best mean reward at epoch {epoch + 1}: {epoch_mean:.6f} | "
                f"saved to {best_reward_model_path.name}"
            )

        torch.save(
            model.state_dict(),
            _epoch_checkpoint_path(target_problem, epoch + 1, self_train_only=self_train_only),
        )
        if (epoch + 1) % 5 == 0:
            multisource.save_colab_model_checkpoint(
                model.state_dict(),
                f"iwc_{_problem_slug(target_problem)}_{'self_only' if self_train_only else 'source_mix'}_centralized_ppo_epoch_{epoch + 1}.pth",
            )

    torch.save(model.state_dict(), model_path)
    print(f"ICW model saved to {model_path.name}")
    multisource._save_reward_log(
        reward_log_path,
        {
            "script": "icw_eva.py",
            "mode": "train_icw_multisource_centralized_ppo",
            "heldout_target_problem": target_problem,
            "model_path": str(model_path),
            "training_problems": train_problems,
            "train_dims": train_dims,
            "training_label": "iwc_centralized_ppo_holdout",
            "reward_scheme": int(getattr(args, "reward_scheme", 1)),
            "surrogate_model": multisource._surrogate_model_name(args),
            "best_reward_model_path": str(best_reward_model_path),
            "best_epoch_mean_reward": best_epoch_mean_reward,
            "epoch_mean_rewards": epoch_mean_rewards,
            "epoch_mean_total_losses": epoch_mean_total_losses,
            "ppo_epochs": ppo_epochs,
            "ppo_actor_lr": ppo_actor_lr,
            "ppo_critic_lr": ppo_critic_lr,
            "ppo_minibatch_size": ppo_minibatch_size,
            "ppo_clip_eps": ppo_clip_eps,
            "ppo_value_coef": ppo_value_coef,
            "ppo_entropy_coef": ppo_entropy_coef,
            "ppo_weight_kl_coef": ppo_weight_kl_coef,
            "ppo_gae_lambda": ppo_gae_lambda,
            "ppo_grad_clip": ppo_grad_clip,
            "ppo_target_kl": ppo_target_kl,
            "ppo_adv_clip": ppo_adv_clip,
            "ppo_logit_reg_coef": model.logit_reg_coef,
            "criterion_names": list(CRITERION_NAMES),
            "lower_better_criteria": list(LOWER_BETTER_CRITERIA),
            "records": reward_records,
        },
    )
    print(f"Reward log saved to {reward_log_path}")
    return model

def load_or_train_icw(args, target_problem: str, self_train_only: bool = False) -> ICW:
    model_path = _final_model_path(target_problem, self_train_only=self_train_only)
    candidate_paths: list[Path] = []

    if model_path.exists():
        candidate_paths.append(model_path)

    best_reward_path = _best_reward_model_path(target_problem, self_train_only=self_train_only)
    if best_reward_path.exists() and best_reward_path not in candidate_paths:
        candidate_paths.append(best_reward_path)

    for epoch_number in range(multisource.TRAIN_EPOCHS, 0, -1):
        checkpoint_path = _epoch_checkpoint_path(target_problem, epoch_number, self_train_only=self_train_only)
        if checkpoint_path.exists() and checkpoint_path not in candidate_paths:
            candidate_paths.append(checkpoint_path)

    for candidate_path in candidate_paths:
        model = _build_icw(args)
        try:
            model.load_state_dict(multisource._torch_load(candidate_path, args.device))
        except RuntimeError as exc:
            print(f"Skipping incompatible ICW checkpoint {candidate_path.name}: {exc}")
            continue

        print(f"Using saved ICW model from {candidate_path.name}")
        return model

    return train_icw_multisource_ppo(args, target_problem, self_train_only=self_train_only)


def run_saea_icw_problem(
    args,
    target_problem: str,
    model: ICW,
    plot: bool = True,
    initial_archive_x: np.ndarray | None = None,
) -> dict:
    problem = multisource.nda.ZDTProblem(name=target_problem, dim=args.dim)
    ref_point = multisource.nsga_eic._reference_point(target_problem, args.dim)

    surrogate_mode = demo.surrogate_model_name(args)
    pretrain_x = pretrain_y = None
    kan_surrogates = None
    gp_surrogates = None
    if surrogate_mode != "gp":
        pretrain_entry = multisource.load_or_prepare_kan_surrogate(target_problem, args.dim, args)
        pretrain_x = pretrain_entry["x"]
        pretrain_y = pretrain_entry["y"]
        kan_surrogates = pretrain_entry["models"]
        print(f"Prepared KAN surrogate on {target_problem}-{args.dim}D with {pretrain_x.shape[0]} samples.")

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
    uncertainty_x = uncertainty_y = None
    if surrogate_mode == "gp":
        gp_surrogates = demo.fit_gp_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            seed=args.seed + multisource._stable_seed(89, target_problem, args.dim),
        )
    else:
        uncertainty_x, uncertainty_y = demo.init_uncertainty_archive(archive_x, archive_y)

    true_evals = args.archive_size
    hv_history: list[float] = []
    reward_history: list[float] = []

    fronts, _ = demo.fast_non_dominated_sort(archive_y)
    front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    initial_hv = demo.hypervolume_2d(front, ref_point)
    hv_history.append(initial_hv)
    print(
        f"Init    | archive={archive_x.shape[0]} | "
        f"front0={front.shape[0]} | HV={initial_hv:.6f} | "
        f"seed_archive={min(base_demo.SURROGATE_WORKING_SIZE, archive_x.shape[0])}"
    )

    step_idx = 0
    while true_evals < args.max_fe:
        surrogate_seed_x = base_demo._select_surrogate_seed_archive(archive_x=archive_x, archive_y=archive_y)

        if surrogate_mode == "gp":
            if gp_surrogates is None:
                raise ValueError("GP surrogate requested but gp_surrogates is None.")
            offspring_x, offspring_pred = multisource.nsga_eic.generate_nsga2_pseudo_front(
                archive_x=surrogate_seed_x,
                problem=problem,
                surrogates=gp_surrogates,
                device=args.device,
                n_offspring=base_demo.SURROGATE_WORKING_SIZE,
                sigma=args.mutation_sigma,
                surrogate_nsga_steps=args.surrogate_nsga_steps,
                predict_fn=demo.predict_with_gp_mean,
                generate_fn=demo.generate_offspring,
            )
        else:
            if kan_surrogates is None:
                raise ValueError("KAN surrogate requested but kan_surrogates is None.")
            offspring_x, offspring_pred = multisource.nsga_eic.generate_nsga2_pseudo_front(
                archive_x=surrogate_seed_x,
                problem=problem,
                surrogates=kan_surrogates,
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
            n_keep=int(args.archive_size),
        )
        model.eval()
        with torch.no_grad():
            act_out = model.act(
                x_true=torch.as_tensor(model_archive_x, dtype=torch.float32, device=args.device),
                y_true=torch.as_tensor(model_archive_y, dtype=torch.float32, device=args.device),
                x_sur=torch.as_tensor(offspring_x, dtype=torch.float32, device=args.device),
                y_sur=torch.as_tensor(offspring_pred, dtype=torch.float32, device=args.device),
                sigma_sur=torch.as_tensor(offspring_sigma, dtype=torch.float32, device=args.device),
                progress=progress,
                lower_bound=problem.lower,
                upper_bound=problem.upper,
                deterministic=True,
            )

        selected_idx, _, _ = select_indices_from_action(  # type: ignore
            action=act_out["action_weights"][0],
            archive_y=archive_y,
            offspring_pred=offspring_pred,
            offspring_sigma=offspring_sigma,
            k_eval=args.k_eval,
            seed=_criterion_seed(args.seed, 0, step_idx, target_problem, args.dim),
            epdi_mc_samples=EPDI_MC_SAMPLES,
        )
        selected_x = offspring_x[selected_idx]
        selected_y = problem.evaluate(selected_x)

        reward_value = float(
            multisource._compute_reward(
                previous_front=archive_y,
                selected_objectives=selected_y,
                reward_scheme=int(getattr(args, "reward_scheme", 1)),
                problem_name=target_problem,
                dim=args.dim,
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
            f"Iter {step_idx + 1:02d} | archive={archive_x.shape[0]} | "
            f"front0={front.shape[0]} | HV={hv_value:.6f} | reward={reward_value:.6f} | "
            f"seed_archive={surrogate_seed_x.shape[0]}"
        )

        true_evals += selected_x.shape[0]
        step_idx += 1
        if true_evals >= args.max_fe:
            break

        if surrogate_mode == "gp":
            gp_surrogates = demo.fit_gp_surrogates(
                archive_x=archive_x,
                archive_y=archive_y,
                seed=args.seed + 300 + step_idx,
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
                seed=args.seed + 300 + step_idx,
            )

    fronts, _ = demo.fast_non_dominated_sort(archive_y)
    final_front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    true_front = multisource.nsga_eic._true_front(target_problem)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.title(f"{args.dim}D {target_problem} Hypervolume Comparison")
        plt.plot(hv_history, marker="o", label="SAEA-ICW")
        plt.xlabel("Step")
        plt.ylabel("Hypervolume")
        plt.grid(True)
        plt.legend()
        plt.show()

        multisource.nsga_eic._plot_front(
            f"{args.dim}D {target_problem} Pareto Front",
            final_front,
            true_front,
            "SAEA-ICW",
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


def run_comparison(args, target_problem: str, self_train_only: bool = False) -> None:
    model = load_or_train_icw(args, target_problem, self_train_only=self_train_only)
    problem = multisource.nda.ZDTProblem(name=target_problem, dim=args.dim)
    shared_init_x = multisource.latin_hypercube_sample(
        lower=problem.lower,
        upper=problem.upper,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )

    icw_result = run_saea_icw_problem(
        args,
        target_problem=target_problem,
        model=model,
        plot=False,
        initial_archive_x=shared_init_x,
    )

    eic_args = multisource.build_args_namespace(args)
    eic_result = multisource.nsga_eic.run_nsga_eic_problem(
        eic_args,
        problem_name=target_problem,
        plot=False,
        initial_archive_x=shared_init_x,
    )

    print(f"\nSAEA-ICW final HV: {icw_result['hv_history'][-1]:.6f}")
    print(f"NSGA-EIC final HV: {eic_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {icw_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title(f"{args.dim}D {target_problem} Hypervolume Comparison")
    plt.plot(icw_result["hv_history"], marker="o", label="SAEA-ICW")
    plt.plot(eic_result["hv_history"], marker="s", label="NSGA-EIC")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend()
    plt.show()

    multisource.nsga_eic._plot_front_comparison(
        f"{args.dim}D {target_problem} Pareto Front Comparison",
        icw_result["final_front"],
        "SAEA-ICW",
        eic_result["final_front"],
        "NSGA-EIC",
        icw_result["true_front"],
    )


def _parse_args(target_problem: str):
    args = multisource.parse_args(target_problem)
    if "--train_algo" not in sys.argv[1:]:
        args.train_algo = "centralized_ppo"
    if getattr(args, "centralized_train_dims", None) is None:
        args.centralized_train_dims = list(CENTRALIZED_TRAIN_DIMS)
    return args


def main():
    target_problem = _consume_target_problem()
    args = _parse_args(target_problem)
    if args.dim != 30:
        print(f"Warning: expected 30D evaluation for {target_problem}, but received dim={args.dim}.")

    if args.train_only:
        if str(args.train_algo).lower() not in {"ppo", "centralized_ppo"}:
            raise ValueError("icw_eva.py supports centralized PPO training only.")
        train_icw_multisource_ppo(args, target_problem, self_train_only=False)
    else:
        run_comparison(args, target_problem, self_train_only=False)


if __name__ == "__main__":
    main()
