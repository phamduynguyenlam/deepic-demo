from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import demo
import multisource_eva_common as multisource
import deepic_demo as deepic_eval
from agent.moe_deepic_agent import MoE_SimplifiedDeepIC, moe_ppo_loss


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
    return str(problem_name).lower()


def _epoch_checkpoint_path(problem_name: str, epoch_number: int) -> Path:
    root = Path(__file__).resolve().parent
    return root / f"moe_deepic_{_problem_slug(problem_name)}_centralized_ppo_epoch_{epoch_number}.pth"


def _final_model_path(problem_name: str) -> Path:
    root = Path(__file__).resolve().parent
    return root / f"moe_deepic_{_problem_slug(problem_name)}_centralized_ppo.pth"


def _reward_log_path(problem_name: str) -> Path:
    return multisource.REWARD_LOG_DIR / f"moe_deepic_{_problem_slug(problem_name)}_centralized_ppo_train_rewards.json"


def _best_reward_model_path(problem_name: str) -> Path:
    root = Path(__file__).resolve().parent
    return root / f"moe_deepic_{_problem_slug(problem_name)}_centralized_ppo_best_raw_reward.pth"


def _build_deepic(args):
    return MoE_SimplifiedDeepIC(
        hidden_dim=args.deepic_hidden,
        n_heads=args.deepic_heads,
        ff_dim=args.deepic_ff,
        dropout=float(getattr(args, "deepic_dropout", 0.0)),
        n_experts=int(getattr(args, "moe_experts", 4)),
        temperature=float(getattr(args, "moe_temperature", 1.0)),
    ).to(args.device)


def _centralized_train_problems(target_problem: str) -> list[str]:
    """Return the 8 training problems used when target_problem is held out."""
    target_upper = str(target_problem).upper()
    pool = ["ZDT1", "ZDT2", "ZDT3", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]
    problems = [p for p in pool if p.upper() != target_upper]
    if len(problems) != len(pool) - 1:
        print(
            f"Warning: target_problem={target_problem!r} is not in the benchmark pool; using all problems for training."
        )
        return list(pool)
    return problems


def _cpu_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in state_dict.items()}


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

    selected_x, selected_y = multisource.nsga_eic._nsga2_survival(archive_x, archive_y, n_keep=n_keep)
    return selected_x.astype(np.float32), selected_y.astype(np.float32)


def _collect_one_deepic_env_trajectory(
    args,
    model_state_dict: dict[str, torch.Tensor],
    problem_name: str,
    dim: int,
    epoch: int,
) -> tuple[list[dict], list[dict], dict]:
    """Collect one on-policy trajectory for one (problem, dim) environment.

    Runs in a worker process. Returns raw numpy arrays only.
    """
    torch.set_num_threads(1)

    worker_args = args
    worker_args.device = "cpu" if str(getattr(args, "device", "cpu")).startswith("cuda") else args.device

    model = _build_deepic(worker_args)
    model.load_state_dict(model_state_dict)
    model.eval()

    entry = multisource.load_or_prepare_kan_surrogate(problem_name, dim, worker_args)
    problem = entry["problem"]
    kan_surrogates = entry["models"]

    surrogate_mode = multisource._surrogate_model_name(worker_args)
    trajectory: list[dict] = []
    reward_records: list[dict] = []

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

    for step in range(int(steps_to_run)):
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
            encoded = model.encode(
                x_true=torch.as_tensor(model_archive_x_t, dtype=torch.float32, device=worker_args.device),
                y_true=torch.as_tensor(model_archive_y_t, dtype=torch.float32, device=worker_args.device),
                x_sur=torch.as_tensor(offspring_x, dtype=torch.float32, device=worker_args.device),
                y_sur=torch.as_tensor(offspring_pred, dtype=torch.float32, device=worker_args.device),
                sigma_sur=torch.as_tensor(offspring_sigma, dtype=torch.float32, device=worker_args.device),
                progress=progress,
                lower_bound=problem.lower,
                upper_bound=problem.upper,
            )
            act_out = model.act(
                H_surr=encoded["H_surr"],
                H_true=encoded["H_true"],
                progress=encoded["progress"],
                k=int(worker_args.k_eval),
                decode_type="sample",
            )

        current_value = float(act_out["value"][0].detach().cpu())
        if trajectory:
            trajectory[-1]["next_value"] = current_value
            trajectory[-1]["done"] = 0.0

        selected_idx = act_out["actions"][0].detach().cpu().numpy().astype(np.int64, copy=False)
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
                "raw_reward": reward_value,
                "normalized_reward": None,
                "train_algo": "centralized_ppo",
            }
        )

        trajectory.append(
            {
                "problem": problem_name,
                "dim": int(dim),
                "step": int(step + 1),
                "raw_reward": reward_value,
                "archive_x": model_archive_x_t.astype(np.float32, copy=False),
                "archive_y": model_archive_y_t.astype(np.float32, copy=False),
                "offspring_x": offspring_x.astype(np.float32, copy=False),
                "offspring_pred": offspring_pred.astype(np.float32, copy=False),
                "offspring_sigma": offspring_sigma.astype(np.float32, copy=False),
                "progress": float(progress),
                "lower": np.asarray(problem.lower, dtype=np.float32),
                "upper": np.asarray(problem.upper, dtype=np.float32),
                "actions": selected_idx.astype(np.int64, copy=False),
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

        true_evals += int(selected_x.shape[0])
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
        "best_obj1": float(np.min(archive_y[:, 0])) if archive_y.size else 0.0,
        "steps": int(len(trajectory)),
        "mean_reward": float(np.mean([s["reward"] for s in trajectory])) if trajectory else 0.0,
    }
    return trajectory, reward_records, summary


def _normalize_epoch_rewards_by_problem(
    trajectories: list[dict],
    reward_records: list[dict],
    discount: float,
    gae_lambda: float,
    eps: float = 1e-8,
) -> dict[str, dict[str, float]]:
    """Normalize rewards per benchmark problem within the current epoch (z-score).

    After reward replacement, PPO targets are recomputed per (problem, dim) trajectory.
    """
    if not trajectories:
        return {}

    by_problem: dict[str, list[dict]] = {}
    for sample in trajectories:
        by_problem.setdefault(str(sample.get("problem", "unknown")), []).append(sample)

    reward_stats: dict[str, dict[str, float]] = {}
    for problem_name, samples in by_problem.items():
        raw = np.asarray([float(s.get("raw_reward", s["reward"])) for s in samples], dtype=np.float32)
        mean = float(raw.mean()) if raw.size else 0.0
        std = float(raw.std()) if raw.size > 1 else 0.0
        denom = std if std > eps else 1.0
        for sample in samples:
            raw_reward = float(sample.get("raw_reward", sample["reward"]))
            sample["raw_reward"] = raw_reward
            sample["reward"] = float((raw_reward - mean) / denom)
        reward_stats[problem_name] = {"mean": mean, "std": std, "count": float(raw.size)}

    for rec in reward_records:
        pname = str(rec.get("problem", "unknown"))
        stats = reward_stats.get(pname)
        raw_reward = float(rec.get("raw_reward", rec.get("reward", 0.0)))
        rec["raw_reward"] = raw_reward
        if stats is None:
            rec["normalized_reward"] = raw_reward
            rec["reward"] = raw_reward
            continue
        denom = float(stats["std"]) if float(stats["std"]) > eps else 1.0
        norm_reward = float((raw_reward - float(stats["mean"])) / denom)
        rec["normalized_reward"] = norm_reward
        rec["reward"] = norm_reward

    by_env: dict[tuple[str, int], list[dict]] = {}
    for sample in trajectories:
        by_env.setdefault((str(sample.get("problem", "unknown")), int(sample.get("dim", -1))), []).append(sample)
    for env_samples in by_env.values():
        env_samples.sort(key=lambda sample: int(sample.get("step", 0)))
        multisource._attach_ppo_targets(env_samples, discount=float(discount), gae_lambda=float(gae_lambda))

    return reward_stats


def _moe_ppo_loss(
    agent: MoE_SimplifiedDeepIC,
    batch: dict[str, torch.Tensor],
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    balance_coef: float,
) -> dict[str, torch.Tensor]:
    return moe_ppo_loss(
        agent=agent,
        batch=batch,
        clip_eps=float(clip_eps),
        value_coef=float(value_coef),
        entropy_coef=float(entropy_coef),
        balance_coef=float(balance_coef),
    )


def _shape_group_key(sample: dict) -> tuple:
    lower = np.asarray(sample["lower"], dtype=np.float32).reshape(-1)
    upper = np.asarray(sample["upper"], dtype=np.float32).reshape(-1)
    return (
        tuple(np.asarray(sample["archive_x"]).shape),
        tuple(np.asarray(sample["archive_y"]).shape),
        tuple(np.asarray(sample["offspring_x"]).shape),
        tuple(np.asarray(sample["offspring_pred"]).shape),
        tuple(np.asarray(sample["offspring_sigma"]).shape),
        tuple(np.asarray(sample["actions"]).shape),
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
            "total_loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy_loss": 0.0,
            "balance_loss": 0.0,
            "gate_entropy": 0.0,
            "approx_kl": 0.0,
            "ratio_mean": 0.0,
            "ratio_std": 0.0,
            "ratio_min": 0.0,
            "ratio_max": 0.0,
            "adv_mean": 0.0,
            "adv_std": 0.0,
            "adv_min": 0.0,
            "adv_max": 0.0,
            "updates": 0.0,
        }

    total_updates = sum(float(s.get("updates", 0.0)) for s in group_stats)
    if total_updates <= 0:
        total_updates = float(len(group_stats))
        weights = [1.0 for _ in group_stats]
    else:
        weights = [float(s.get("updates", 0.0)) for s in group_stats]

    keys = [
        "total_loss",
        "actor_loss",
        "critic_loss",
        "entropy_loss",
        "balance_loss",
        "gate_entropy",
        "approx_kl",
        "ratio_mean",
        "ratio_std",
        "ratio_min",
        "ratio_max",
        "adv_mean",
        "adv_std",
        "adv_min",
        "adv_max",
    ]
    merged = {k: 0.0 for k in keys}
    for s, w in zip(group_stats, weights):
        for k in keys:
            merged[k] += float(s.get(k, 0.0)) * w
    for k in keys:
        merged[k] /= max(total_updates, 1e-12)
    merged["updates"] = total_updates
    return merged


def _update_deepic_from_episode_ppo(
    model,
    optimizer,
    episode_trajectory: list[dict],
    device: str,
    ppo_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    balance_coef: float,
    grad_clip: float | None,
    target_kl: float | None,
    adv_clip: float | None,
) -> dict[str, float]:
    if not episode_trajectory:
        return {
            "total_loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy_loss": 0.0,
            "balance_loss": 0.0,
            "gate_entropy": 0.0,
            "approx_kl": 0.0,
            "ratio_mean": 0.0,
            "ratio_std": 0.0,
            "ratio_min": 0.0,
            "ratio_max": 0.0,
            "adv_mean": 0.0,
            "adv_std": 0.0,
            "adv_min": 0.0,
            "adv_max": 0.0,
            "updates": 0.0,
        }

    model.train()
    group_size = len(episode_trajectory)

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
    ).reshape(group_size, -1)
    actions = torch.as_tensor(
        np.stack([np.asarray(sample["actions"], dtype=np.int64) for sample in episode_trajectory], axis=0),
        dtype=torch.long,
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
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    if adv_clip is not None and float(adv_clip) > 0.0:
        advantages = advantages.clamp(-float(adv_clip), float(adv_clip))

    stats = {
        "total_loss": 0.0,
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy_loss": 0.0,
        "balance_loss": 0.0,
        "gate_entropy": 0.0,
        "approx_kl": 0.0,
        "ratio_mean": 0.0,
        "ratio_std": 0.0,
        "ratio_min": 0.0,
        "ratio_max": 0.0,
        "adv_mean": 0.0,
        "adv_std": 0.0,
        "adv_min": 0.0,
        "adv_max": 0.0,
        "updates": 0.0,
    }

    lower_bound = episode_trajectory[0]["lower"]
    upper_bound = episode_trajectory[0]["upper"]

    stop_updates = False
    for _ in range(max(int(ppo_epochs), 1)):
        perm = torch.randperm(group_size, device=device)
        for start in range(0, group_size, max(int(minibatch_size), 1)):
            mb_idx = perm[start : start + max(int(minibatch_size), 1)]
            if mb_idx.numel() == 0:
                continue

            encoded = model.encode(
                x_true=archive_x[mb_idx],
                y_true=archive_y[mb_idx],
                x_sur=offspring_x[mb_idx],
                y_sur=offspring_pred[mb_idx],
                sigma_sur=offspring_sigma[mb_idx],
                progress=progress[mb_idx],
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
            batch = {
                "H_surr": encoded["H_surr"],
                "H_true": encoded["H_true"],
                "progress": encoded["progress"],
                "actions": actions[mb_idx],
                "old_logprob": old_logprob[mb_idx],
                "advantages": advantages[mb_idx],
                "returns": returns[mb_idx],
            }
            loss_dict = _moe_ppo_loss(
                agent=model,
                batch=batch,
                clip_eps=clip_eps,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                balance_coef=balance_coef,
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
            if "balance_loss" in loss_dict:
                stats["balance_loss"] += float(loss_dict["balance_loss"].detach().cpu())
            if "gate_entropy" in loss_dict:
                stats["gate_entropy"] += float(loss_dict["gate_entropy"].detach().cpu())
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
                if approx_kl > float(target_kl):
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
            "balance_loss",
            "gate_entropy",
            "approx_kl",
            "ratio_mean",
            "ratio_std",
            "ratio_min",
            "ratio_max",
            "adv_mean",
            "adv_std",
            "adv_min",
            "adv_max",
        ]:
            stats[key] /= stats["updates"]

    return stats


def _update_deepic_from_centralized_ppo_groups(
    model,
    optimizer,
    trajectories: list[dict],
    device: str,
    ppo_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    balance_coef: float,
    grad_clip: float | None,
    target_kl: float | None,
    adv_clip: float | None,
) -> tuple[dict[str, float], dict[str, int]]:
    grouped = _split_trajectories_by_shape(trajectories)
    group_sizes = {str(k): len(v) for k, v in grouped.items()}
    stats_per_group: list[dict[str, float]] = []
    for _, group_samples in sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True):
        stats = _update_deepic_from_episode_ppo(
            model=model,
            optimizer=optimizer,
            episode_trajectory=group_samples,
            device=device,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            clip_eps=clip_eps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            balance_coef=balance_coef,
            grad_clip=grad_clip,
            target_kl=target_kl,
            adv_clip=adv_clip,
        )
        stats_per_group.append(stats)
    return _merge_loss_stats_weighted(stats_per_group), group_sizes


def train_deepic_centralized_ppo(args, target_problem: str) -> object:
    """Centralized PPO for MoE-DeepIC (holdout training).

    Hold out target_problem. Train on the remaining 8 benchmark problems and
    three dimensions [15, 20, 25], i.e. 24 environments per epoch. Rollouts are
    collected with the same frozen policy, normalized per-problem, then PPO
    updates run on grouped (stackable) batches.
    """
    demo.set_seed(args.seed)
    reward_records: list[dict] = []
    epoch_mean_normalized_rewards: list[float] = []
    epoch_mean_raw_rewards: list[float] = []
    epoch_mean_total_losses: list[float] = []

    model_path = _final_model_path(target_problem)
    reward_log_path = _reward_log_path(target_problem)
    best_reward_model_path = _best_reward_model_path(target_problem)
    best_epoch_mean_raw_reward = float("-inf")

    torch.set_num_threads(int(getattr(args, "torch_num_threads", 2)))

    model = _build_deepic(args)

    # Centralized PPO hyperparameters (keep aligned with icw_eva.py).
    args.discount = 1.0
    ppo_epochs = int(getattr(args, "ppo_epochs", 4))
    ppo_clip_eps = float(getattr(args, "ppo_clip_eps", 0.08))
    ppo_actor_lr = float(getattr(args, "ppo_actor_lr", 1e-4))
    ppo_critic_lr = float(getattr(args, "ppo_critic_lr", 1e-4))
    ppo_value_coef = float(getattr(args, "ppo_value_coef", 0.03))
    ppo_entropy_coef = float(getattr(args, "ppo_entropy_coef", 0.01))
    # If not explicitly set, prefer 64 when batch is large; otherwise 32.
    ppo_minibatch_size = int(getattr(args, "ppo_minibatch_size", 0))
    ppo_gae_lambda = float(getattr(args, "ppo_gae_lambda", 0.95))
    ppo_grad_clip = float(getattr(args, "ppo_grad_clip", 1.0))
    ppo_target_kl = float(getattr(args, "ppo_target_kl", 0.01))
    ppo_adv_clip = float(getattr(args, "ppo_adv_clip", 2.0))

    train_problems = _centralized_train_problems(target_problem)
    configured_train_dims = getattr(args, "centralized_train_dims", None)
    train_dims = list(configured_train_dims) if configured_train_dims else [15, 20, 25]
    train_envs = [(p, int(d)) for p in train_problems for d in train_dims]

    parallel_workers = int(getattr(args, "ppo_parallel_workers", 12))
    parallel_workers = max(1, min(parallel_workers, 12, len(train_envs)))

    print(
        f"Training config (MoE-DeepIC Centralized PPO) | surrogate_nsga_steps={args.surrogate_nsga_steps} | "
        f"discount={args.discount:.4f} | ppo_epochs={ppo_epochs} | ppo_clip_eps={ppo_clip_eps:.3f} | "
        f"actor_lr={ppo_actor_lr:.1e} | critic_lr={ppo_critic_lr:.1e} | vf_coef={ppo_value_coef:.3f} | "
        f"target_kl={ppo_target_kl:.4f} | adv_clip={ppo_adv_clip:.2f} | "
        f"reward_scheme={getattr(args, 'reward_scheme', 1)} | "
        f"surrogate_model={multisource._surrogate_model_name(args)} | heldout={target_problem} | "
        f"train_envs={len(train_envs)} ({len(train_problems)} problems × {len(train_dims)} dims) | "
        f"parallel_workers={parallel_workers}"
    )
    print(f"Training problems: {train_problems}")
    print(f"Training dims: {train_dims}")

    for p, d in train_envs:
        multisource.load_or_prepare_kan_surrogate(p, d, args)

    optimizer = multisource._build_ppo_optimizer(model=model, actor_lr=ppo_actor_lr, critic_lr=ppo_critic_lr)

    if int(getattr(args, "start_epoch", 0)) > 0:
        checkpoint_path = _epoch_checkpoint_path(target_problem, int(args.start_epoch))
        if checkpoint_path.exists():
            model.load_state_dict(multisource._torch_load(checkpoint_path, args.device))
            print(f"Loaded model from {checkpoint_path.name}")
        else:
            print(f"Checkpoint {checkpoint_path.name} not found, starting from scratch")

    for epoch in range(int(getattr(args, "start_epoch", 0)), int(getattr(multisource, "TRAIN_EPOCHS", 50))):
        print(f"Epoch {epoch + 1}/{int(getattr(multisource, 'TRAIN_EPOCHS', 50))}")

        epoch_trajectories: list[dict] = []
        epoch_reward_records: list[dict] = []
        env_summaries: list[dict] = []

        rollout_state = _cpu_state_dict(model.state_dict())

        if parallel_workers == 1:
            for problem_name, dim in train_envs:
                traj, records, summary = _collect_one_deepic_env_trajectory(
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
            import copy
            try:
                import ray  # type: ignore
            except ImportError:
                ray = None

            worker_args = copy.copy(args)
            worker_args.device = "cpu"
            if ray is None:
                from concurrent.futures import ProcessPoolExecutor, as_completed

                with ProcessPoolExecutor(max_workers=parallel_workers) as ex:
                    futures = [
                        ex.submit(
                            _collect_one_deepic_env_trajectory,
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
            else:
                if not ray.is_initialized():
                    ray.init(
                        num_cpus=parallel_workers,
                        include_dashboard=False,
                        ignore_reinit_error=True,
                        log_to_driver=False,
                    )
                remote_collect = ray.remote(num_cpus=1)(_collect_one_deepic_env_trajectory)
                worker_args_ref = ray.put(worker_args)
                rollout_state_ref = ray.put(rollout_state)
                pending = [
                    remote_collect.remote(worker_args_ref, rollout_state_ref, problem_name, int(dim), epoch)
                    for problem_name, dim in train_envs
                ]
                while pending:
                    ready, pending = ray.wait(pending, num_returns=1)
                    traj, records, summary = ray.get(ready[0])
                    epoch_trajectories.extend(traj)
                    epoch_reward_records.extend(records)
                    env_summaries.append(summary)
                    print(
                        f"{summary['problem']}-{summary['dim']}D epoch {epoch + 1} done, "
                        f"true_evals={summary['true_evals']}, best_obj1={summary['best_obj1']:.6f}, "
                        f"steps={summary['steps']}, mean_reward={summary['mean_reward']:.6f}"
                    )

        raw_epoch_rewards = [float(r.get("raw_reward", r["reward"])) for r in epoch_reward_records]
        reward_norm_stats = _normalize_epoch_rewards_by_problem(
            trajectories=epoch_trajectories,
            reward_records=epoch_reward_records,
            discount=float(args.discount),
            gae_lambda=ppo_gae_lambda,
        )
        reward_records.extend(epoch_reward_records)
        normalized_epoch_rewards = [float(r["reward"]) for r in epoch_reward_records]
        raw_epoch_mean = float(np.mean(raw_epoch_rewards)) if raw_epoch_rewards else 0.0

        if reward_norm_stats:
            compact_reward_stats = {
                key: (round(val["mean"], 6), round(val["std"], 6), int(val["count"]))
                for key, val in sorted(reward_norm_stats.items())
            }
            print(f"Reward normalization by problem (raw_mean, raw_std, count): {compact_reward_stats}")

        if epoch_trajectories:
            ppo_minibatch = int(ppo_minibatch_size)
            if int(getattr(args, "ppo_minibatch_size", 0)) <= 0:
                ppo_minibatch = 64 if len(epoch_trajectories) > 200 else 32

            loss_stats, group_sizes = _update_deepic_from_centralized_ppo_groups(
                model=model,
                optimizer=optimizer,
                trajectories=epoch_trajectories,
                device=args.device,
                ppo_epochs=ppo_epochs,
                minibatch_size=ppo_minibatch,
                clip_eps=ppo_clip_eps,
                value_coef=ppo_value_coef,
                entropy_coef=ppo_entropy_coef,
                balance_coef=float(getattr(args, "moe_balance_coef", 0.001)),
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
                f"balance={loss_stats.get('balance_loss', 0.0):.6f}, gate_ent={loss_stats.get('gate_entropy', 0.0):.6f}, "
                f"approx_kl={loss_stats['approx_kl']:.6f}, ratio_mean={loss_stats['ratio_mean']:.6f}, "
                f"ratio_std={loss_stats['ratio_std']:.6f}, ratio_min={loss_stats['ratio_min']:.6f}, "
                f"ratio_max={loss_stats['ratio_max']:.6f}, adv_mean={loss_stats['adv_mean']:.6f}, "
                f"adv_std={loss_stats['adv_std']:.6f}, adv_min={loss_stats['adv_min']:.6f}, "
                f"adv_max={loss_stats['adv_max']:.6f}"
            )
        else:
            epoch_mean_total_losses.append(0.0)

        epoch_mean_raw_rewards.append(raw_epoch_mean)
        epoch_mean_normalized_rewards.append(float(np.mean(normalized_epoch_rewards)) if normalized_epoch_rewards else 0.0)
        print(f"Epoch {epoch + 1} mean raw reward: {raw_epoch_mean:.6f}")
        print(f"Epoch {epoch + 1} mean normalized reward: {epoch_mean_normalized_rewards[-1]:.6f}")
        print(f"Epoch {epoch + 1} mean PPO total loss: {epoch_mean_total_losses[-1]:.6f}")

        if raw_epoch_mean > best_epoch_mean_raw_reward:
            best_epoch_mean_raw_reward = raw_epoch_mean
            torch.save(model.state_dict(), best_reward_model_path)
            print(
                f"New best mean raw reward at epoch {epoch + 1}: {raw_epoch_mean:.6f} | "
                f"saved to {best_reward_model_path.name}"
            )

        torch.save(model.state_dict(), _epoch_checkpoint_path(target_problem, epoch + 1))

        if (epoch + 1) % 5 == 0:
            multisource.save_colab_model_checkpoint(
                model.state_dict(),
                f"moe_deepic_{_problem_slug(target_problem)}_centralized_ppo_epoch_{epoch + 1}.pth",
            )

    torch.save(model.state_dict(), model_path)
    print(f"MoE-DeepIC model saved to {model_path.name}")
    multisource._save_reward_log(
        reward_log_path,
        {
            "script": "moe_deepic_eva.py",
            "mode": "train_moe_deepic_centralized_ppo",
            "heldout_target_problem": target_problem,
            "model_path": str(model_path),
            "training_problems": train_problems,
            "train_dims": train_dims,
            "training_label": "moe_deepic_centralized_ppo_holdout",
            "reward_scheme": int(getattr(args, "reward_scheme", 1)),
            "surrogate_model": multisource._surrogate_model_name(args),
            "best_reward_model_path": str(best_reward_model_path),
            "best_epoch_mean_raw_reward": best_epoch_mean_raw_reward,
            "epoch_mean_raw_rewards": epoch_mean_raw_rewards,
            "epoch_mean_normalized_rewards": epoch_mean_normalized_rewards,
            "reward_normalization": "per_problem_zscore_per_epoch",
            "ray_num_cpus": parallel_workers,
            "epoch_mean_total_losses": epoch_mean_total_losses,
            "ppo_epochs": ppo_epochs,
            "ppo_actor_lr": ppo_actor_lr,
            "ppo_critic_lr": ppo_critic_lr,
            "ppo_minibatch_size": ppo_minibatch_size,
            "ppo_clip_eps": ppo_clip_eps,
            "ppo_value_coef": ppo_value_coef,
            "ppo_entropy_coef": ppo_entropy_coef,
            "moe_experts": int(getattr(args, "moe_experts", 4)),
            "moe_temperature": float(getattr(args, "moe_temperature", 1.0)),
            "moe_balance_coef": float(getattr(args, "moe_balance_coef", 0.001)),
            "ppo_gae_lambda": ppo_gae_lambda,
            "ppo_grad_clip": ppo_grad_clip,
            "ppo_target_kl": ppo_target_kl,
            "ppo_adv_clip": ppo_adv_clip,
            "records": reward_records,
        },
    )
    print(f"Reward log saved to {reward_log_path}")
    return model


def load_or_train_deepic(args, target_problem: str):
    model_path = _final_model_path(target_problem)
    candidate_paths: list[Path] = []

    if model_path.exists():
        candidate_paths.append(model_path)

    best_reward_path = _best_reward_model_path(target_problem)
    if best_reward_path.exists() and best_reward_path not in candidate_paths:
        candidate_paths.append(best_reward_path)

    for epoch_number in range(int(getattr(multisource, "TRAIN_EPOCHS", 50)), 0, -1):
        checkpoint_path = _epoch_checkpoint_path(target_problem, epoch_number)
        if checkpoint_path.exists() and checkpoint_path not in candidate_paths:
            candidate_paths.append(checkpoint_path)

    for candidate_path in candidate_paths:
        model = _build_deepic(args)
        try:
            model.load_state_dict(multisource._torch_load(candidate_path, args.device))
        except RuntimeError as exc:
            print(f"Skipping incompatible MoE-DeepIC checkpoint {candidate_path.name}: {exc}")
            continue
        print(f"Using saved MoE-DeepIC model from {candidate_path.name}")
        return model

    return train_deepic_centralized_ppo(args, target_problem)


def run_comparison(args, target_problem: str) -> None:
    deepic = load_or_train_deepic(args, target_problem)
    problem = multisource.nda.ZDTProblem(name=target_problem, dim=args.dim)
    shared_init_x = multisource.latin_hypercube_sample(
        lower=problem.lower,
        upper=problem.upper,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )

    deepic_result = deepic_eval.run_saea_deepic_problem(
        args,
        target_problem=target_problem,
        deepic=deepic,
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

    print(f"\nSAEA-MoE-DeepIC final HV: {deepic_result['hv_history'][-1]:.6f}")
    print(f"NSGA-EIC final HV: {eic_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {deepic_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title(f"{args.dim}D {target_problem} Hypervolume Comparison")
    plt.plot(deepic_result["hv_history"], marker="o", label="SAEA-MoE-DeepIC")
    plt.plot(eic_result["hv_history"], marker="s", label="NSGA-EIC")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend()
    plt.show()

    multisource.nsga_eic._plot_front_comparison(
        f"{args.dim}D {target_problem} Pareto Front Comparison",
        deepic_result["final_front"],
        "SAEA-MoE-DeepIC",
        eic_result["final_front"],
        "NSGA-EIC",
        deepic_result["true_front"],
    )


def _parse_args(target_problem: str):
    args = multisource.parse_args(target_problem)
    if "--train_algo" not in sys.argv[1:]:
        args.train_algo = "centralized_ppo"
    if getattr(args, "centralized_train_dims", None) is None:
        args.centralized_train_dims = [15, 20, 25]
    # Default to 12 rollout workers for centralized PPO unless user overrides.
    if "--ppo_parallel_workers" not in sys.argv[1:]:
        args.ppo_parallel_workers = 12
    # Centralized PPO defaults (aligned with icw_eva.py) unless user overrides.
    if "--ppo_epochs" not in sys.argv[1:]:
        args.ppo_epochs = 4
    if "--ppo_clip_eps" not in sys.argv[1:]:
        args.ppo_clip_eps = 0.08
    if "--ppo_actor_lr" not in sys.argv[1:]:
        args.ppo_actor_lr = 1e-4
    if "--ppo_critic_lr" not in sys.argv[1:]:
        args.ppo_critic_lr = 1e-4
    if "--ppo_value_coef" not in sys.argv[1:]:
        args.ppo_value_coef = 0.03
    if "--ppo_target_kl" not in sys.argv[1:]:
        args.ppo_target_kl = 0.01
    if "--ppo_entropy_coef" not in sys.argv[1:]:
        args.ppo_entropy_coef = 0.01
    # If user does not set minibatch_size, pick based on batch size at runtime.
    if "--ppo_minibatch_size" not in sys.argv[1:]:
        args.ppo_minibatch_size = 0
    if "--ppo_adv_clip" not in sys.argv[1:]:
        args.ppo_adv_clip = 2.0
    if "--discount" not in sys.argv[1:]:
        args.discount = 1.0
    return args


def main():
    target_problem = _consume_target_problem()
    args = _parse_args(target_problem)
    if args.dim != 30:
        print(f"Warning: expected 30D evaluation for {target_problem}, but received dim={args.dim}.")

    if args.train_only:
        train_deepic_centralized_ppo(args, target_problem)
    else:
        run_comparison(args, target_problem)


if __name__ == "__main__":
    main()
