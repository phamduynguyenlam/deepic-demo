from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent.moe_deepic_agent import MoE_SimplifiedDeepIC, moe_ppo_loss

import demo
import multisource_eva_common as multisource
import deepic_demo as base_demo


TARGET_PROBLEM = "ZDT1"


def _problem_slug(problem_name: str) -> str:
    return problem_name.lower()


def _epoch_checkpoint_path(problem_name: str, epoch_number: int, self_train_only: bool = False) -> Path:
    root = Path(__file__).resolve().parent
    if self_train_only:
        return root / f"moe_deepic_{_problem_slug(problem_name)}_self_model_epoch_{epoch_number}.pth"
    return root / f"moe_deepic_{_problem_slug(problem_name)}_model_epoch_{epoch_number}.pth"


def _final_model_path(problem_name: str, self_train_only: bool = False) -> Path:
    root = Path(__file__).resolve().parent
    if self_train_only:
        return root / f"moe_deepic_{_problem_slug(problem_name)}_self_only.pth"
    return root / f"moe_deepic_{_problem_slug(problem_name)}_source_mix.pth"


def _reward_log_path(problem_name: str, self_train_only: bool = False) -> Path:
    label = "demo" if self_train_only else "eva"
    return multisource.REWARD_LOG_DIR / f"moe_deepic_{_problem_slug(problem_name)}_{label}_train_rewards.json"


def _best_reward_model_path(problem_name: str, self_train_only: bool = False) -> Path:
    root = Path(__file__).resolve().parent
    if self_train_only:
        return root / f"moe_deepic_{_problem_slug(problem_name)}_self_only_ppo_best_reward.pth"
    return root / f"moe_deepic_{_problem_slug(problem_name)}_source_mix_ppo_best_reward.pth"


def _build_moe(args) -> MoE_SimplifiedDeepIC:
    return MoE_SimplifiedDeepIC(
        hidden_dim=args.deepic_hidden,
        n_heads=args.deepic_heads,
        ff_dim=args.deepic_ff,
        dropout=float(getattr(args, "deepic_dropout", 0.0)),
        n_experts=int(getattr(args, "moe_experts", 4)),
        temperature=float(getattr(args, "moe_temperature", 1.0)),
    ).to(args.device)


def _format_vector(values: np.ndarray) -> str:
    return np.array2string(
        np.asarray(values, dtype=np.float32),
        precision=4,
        separator=", ",
        suppress_small=False,
    )


def _update_moe_from_episode_ppo(
    model: MoE_SimplifiedDeepIC,
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
) -> dict[str, float]:
    if not episode_trajectory:
        return {
            "total_loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy_loss": 0.0,
            "approx_kl": 0.0,
            "ratio_mean": 0.0,
            "ratio_std": 0.0,
            "ratio_min": 0.0,
            "ratio_max": 0.0,
            "adv_mean": 0.0,
            "adv_std": 0.0,
            "adv_min": 0.0,
            "adv_max": 0.0,
            "balance_loss": 0.0,
            "gate_entropy": 0.0,
            "updates": 0.0,
        }

    model.train()
    grouped_indices: dict[tuple[tuple[int, ...], ...], list[int]] = {}
    for idx, sample in enumerate(episode_trajectory):
        key = (
            tuple(sample["archive_x"].shape),
            tuple(sample["archive_y"].shape),
            tuple(sample["offspring_x"].shape),
            tuple(sample["offspring_pred"].shape),
            tuple(sample["offspring_sigma"].shape),
            tuple(sample["actions"].shape),
            multisource._bound_group_key(sample["lower"]),
            multisource._bound_group_key(sample["upper"]),
        )
        grouped_indices.setdefault(key, []).append(idx)

    stats = {
        "total_loss": 0.0,
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy_loss": 0.0,
        "approx_kl": 0.0,
        "ratio_mean": 0.0,
        "ratio_std": 0.0,
        "ratio_min": 0.0,
        "ratio_max": 0.0,
        "adv_mean": 0.0,
        "adv_std": 0.0,
        "adv_min": 0.0,
        "adv_max": 0.0,
        "balance_loss": 0.0,
        "gate_entropy": 0.0,
        "updates": 0.0,
    }

    for indices in grouped_indices.values():
        group_samples = [episode_trajectory[idx] for idx in indices]
        group_size = len(group_samples)
        if group_size == 0:
            continue

        archive_x = torch.as_tensor(
            np.stack([sample["archive_x"] for sample in group_samples], axis=0),
            dtype=torch.float32,
            device=device,
        )
        archive_y = torch.as_tensor(
            np.stack([sample["archive_y"] for sample in group_samples], axis=0),
            dtype=torch.float32,
            device=device,
        )
        offspring_x = torch.as_tensor(
            np.stack([sample["offspring_x"] for sample in group_samples], axis=0),
            dtype=torch.float32,
            device=device,
        )
        offspring_pred = torch.as_tensor(
            np.stack([sample["offspring_pred"] for sample in group_samples], axis=0),
            dtype=torch.float32,
            device=device,
        )
        offspring_sigma = torch.as_tensor(
            np.stack([sample["offspring_sigma"] for sample in group_samples], axis=0),
            dtype=torch.float32,
            device=device,
        )
        progress = torch.as_tensor(
            [sample["progress"] for sample in group_samples],
            dtype=torch.float32,
            device=device,
        ).reshape(group_size, -1)
        actions = torch.stack(
            [sample["actions"].to(device=device, dtype=torch.long) for sample in group_samples],
            dim=0,
        )
        old_logprob = torch.as_tensor(
            [float(sample["old_logprob"]) for sample in group_samples],
            dtype=torch.float32,
            device=device,
        )
        advantages = torch.as_tensor(
            [sample["advantage"] for sample in group_samples],
            dtype=torch.float32,
            device=device,
        )
        returns = torch.as_tensor(
            [sample["return"] for sample in group_samples],
            dtype=torch.float32,
            device=device,
        )
        if group_size > 1:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

        stop_group_updates = False
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
                    lower_bound=group_samples[0]["lower"],
                    upper_bound=group_samples[0]["upper"],
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

                loss_dict = moe_ppo_loss(
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
                stats["approx_kl"] += float(loss_dict["approx_kl"].detach().cpu())
                stats["ratio_mean"] += float(loss_dict["ratio_mean"].detach().cpu())
                stats["ratio_std"] += float(loss_dict["ratio_std"].detach().cpu())
                stats["ratio_min"] += float(loss_dict["ratio_min"].detach().cpu())
                stats["ratio_max"] += float(loss_dict["ratio_max"].detach().cpu())
                stats["adv_mean"] += float(adv_slice.mean().detach().cpu())
                stats["adv_std"] += (
                    float(adv_slice.std(unbiased=False).detach().cpu()) if adv_slice.numel() > 1 else 0.0
                )
                stats["adv_min"] += float(adv_slice.min().detach().cpu())
                stats["adv_max"] += float(adv_slice.max().detach().cpu())
                if "balance_loss" in loss_dict:
                    stats["balance_loss"] += float(loss_dict["balance_loss"].detach().cpu())
                if "gate_entropy" in loss_dict:
                    stats["gate_entropy"] += float(loss_dict["gate_entropy"].detach().cpu())
                stats["updates"] += 1.0

                if target_kl is not None and float(target_kl) > 0.0:
                    approx_kl = float(loss_dict["approx_kl"].detach().cpu())
                    if abs(approx_kl) > float(target_kl):
                        stop_group_updates = True
                        break
            if stop_group_updates:
                break

    if stats["updates"] > 0:
        for key in [
            "total_loss",
            "actor_loss",
            "critic_loss",
            "entropy_loss",
            "approx_kl",
            "ratio_mean",
            "ratio_std",
            "ratio_min",
            "ratio_max",
            "adv_mean",
            "adv_std",
            "adv_min",
            "adv_max",
            "balance_loss",
            "gate_entropy",
        ]:
            stats[key] /= stats["updates"]

    return stats


def train_moe_deepic_multisource_ppo(args, target_problem: str, self_train_only: bool = False) -> MoE_SimplifiedDeepIC:
    demo.set_seed(args.seed)
    reward_records: list[dict] = []
    epoch_mean_rewards: list[float] = []
    epoch_mean_total_losses: list[float] = []
    model_path = _final_model_path(target_problem, self_train_only=self_train_only)
    reward_log_path = _reward_log_path(target_problem, self_train_only=self_train_only)
    best_reward_model_path = _best_reward_model_path(target_problem, self_train_only=self_train_only)
    best_epoch_mean_reward = float("-inf")

    model = _build_moe(args)

    ppo_actor_lr = 1e-4
    ppo_critic_lr = float(getattr(args, "ppo_critic_lr", 1e-4))
    ppo_epochs = 4
    ppo_minibatch_size = int(getattr(args, "ppo_minibatch_size", 32))
    ppo_clip_eps = 0.1
    ppo_value_coef = float(getattr(args, "ppo_value_coef", 0.1))
    ppo_entropy_coef = float(getattr(args, "ppo_entropy_coef", 0.01))
    ppo_gae_lambda = float(getattr(args, "ppo_gae_lambda", 0.95))
    ppo_grad_clip = float(getattr(args, "ppo_grad_clip", 1.0))
    ppo_target_kl = float(getattr(args, "ppo_target_kl", 0.01))
    ppo_balance_coef = float(getattr(args, "moe_balance_coef", 0.001))

    print(
        f"Training config (MoE PPO) | surrogate_nsga_steps={args.surrogate_nsga_steps} | discount={args.discount:.4f} | "
        f"ppo_epochs={ppo_epochs} | ppo_clip_eps={ppo_clip_eps:.3f} | "
        f"actor_lr={ppo_actor_lr:.1e} | critic_lr={ppo_critic_lr:.1e} | "
        f"vf_coef={ppo_value_coef:.3f} | target_kl={ppo_target_kl:.4f} | "
        f"balance_coef={ppo_balance_coef:.4f} | "
        f"reward_scheme={getattr(args, 'reward_scheme', 1)} | "
        f"surrogate_model={multisource._surrogate_model_name(args)}"
    )

    pretrain_cache = multisource.pretrain_source_surrogates(args, target_problem, self_train_only=self_train_only)
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
        epoch_rewards: list[float] = []
        epoch_total_losses: list[float] = []

        for dim in multisource.SOURCE_DIMS:
            dim_trajectories: list[dict] = []
            dim_problem_count = 0

            for problem_name in multisource.training_problems_for(target_problem, self_train_only=self_train_only):
                entry = pretrain_cache[(problem_name, dim)]
                problem = entry["problem"]
                surrogates = entry["models"]
                episode_trajectory: list[dict] = []

                archive_x = multisource.latin_hypercube_sample(
                    lower=problem.lower,
                    upper=problem.upper,
                    n_samples=args.archive_size,
                    dim=dim,
                    seed=args.seed + epoch * 10000 + multisource._stable_seed(0, problem_name, dim),
                )
                archive_y = problem.evaluate(archive_x)
                uncertainty_x, uncertainty_y = demo.init_uncertainty_archive(archive_x, archive_y)
                gp_surrogates = None
                if multisource._surrogate_model_name(args) == "gp":
                    gp_surrogates = demo.fit_gp_surrogates(
                        archive_x=uncertainty_x,
                        archive_y=uncertainty_y,
                        seed=args.seed + epoch * 10000 + multisource._stable_seed(17, problem_name, dim),
                    )

                true_evals = args.archive_size
                remaining_budget = args.max_fe - true_evals
                steps_to_run = remaining_budget // args.k_eval

                for step in range(steps_to_run):
                    archive_x_t = archive_x.copy()
                    archive_y_t = archive_y.copy()

                    offspring_x, offspring_pred = multisource.nsga_eic.generate_nsga2_pseudo_front(
                        archive_x=archive_x_t,
                        problem=problem,
                        surrogates=surrogates,
                        device=args.device,
                        n_offspring=args.offspring_size,
                        sigma=args.mutation_sigma,
                        surrogate_nsga_steps=args.surrogate_nsga_steps,
                        predict_fn=demo.predict_with_kan,
                        generate_fn=demo.generate_offspring,
                    )
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
                    with torch.no_grad():
                        encoded = model.encode(
                            x_true=torch.as_tensor(archive_x_t, dtype=torch.float32, device=args.device),
                            y_true=torch.as_tensor(archive_y_t, dtype=torch.float32, device=args.device),
                            x_sur=torch.as_tensor(offspring_x, dtype=torch.float32, device=args.device),
                            y_sur=torch.as_tensor(offspring_pred, dtype=torch.float32, device=args.device),
                            sigma_sur=torch.as_tensor(offspring_sigma, dtype=torch.float32, device=args.device),
                            progress=progress,
                            lower_bound=problem.lower,
                            upper_bound=problem.upper,
                        )
                        act_out = model.act(
                            H_surr=encoded["H_surr"],
                            H_true=encoded["H_true"],
                            progress=encoded["progress"],
                            k=args.k_eval,
                            decode_type="sample",
                        )

                    current_value = float(act_out["value"][0].detach().cpu())
                    if episode_trajectory:
                        episode_trajectory[-1]["next_value"] = current_value
                        episode_trajectory[-1]["done"] = 0.0

                    selected_idx = act_out["actions"][0].detach().cpu().numpy()
                    selected_x = offspring_x[selected_idx]
                    selected_y = problem.evaluate(selected_x)

                    reward_value = float(
                        multisource._compute_reward(
                            previous_front=archive_y_t,
                            selected_objectives=selected_y,
                            reward_scheme=int(getattr(args, "reward_scheme", 1)),
                            problem_name=problem_name,
                            dim=dim,
                        )
                    )
                    epoch_rewards.append(reward_value)
                    reward_records.append(
                        {
                            "epoch": epoch + 1,
                            "problem": problem_name,
                            "dim": int(dim),
                            "step": step + 1,
                            "progress": float(progress),
                            "reward": reward_value,
                            "train_algo": "moe_ppo",
                        }
                    )

                    episode_trajectory.append(
                        {
                            "archive_x": archive_x_t,
                            "archive_y": archive_y_t,
                            "offspring_x": offspring_x,
                            "offspring_pred": offspring_pred,
                            "offspring_sigma": offspring_sigma,
                            "progress": progress,
                            "lower": problem.lower,
                            "upper": problem.upper,
                            "actions": act_out["actions"][0].detach().cpu(),
                            "old_logprob": float(act_out["logprob"][0].detach().cpu()),
                            "value": current_value,
                            "next_value": 0.0,
                            "done": 1.0,
                            "reward": reward_value,
                            "gate_weights": act_out["gate_weights"][0].detach().cpu(),
                            "expert_logits": act_out["expert_logits"][0].detach().cpu(),
                        }
                    )

                    archive_x, archive_y = demo.update_archive(
                        archive_x=archive_x_t,
                        archive_y=archive_y_t,
                        new_x=selected_x,
                        new_y=selected_y,
                    )
                    uncertainty_x, uncertainty_y = demo.update_uncertainty_archive(
                        uncertainty_x=uncertainty_x,
                        uncertainty_y=uncertainty_y,
                        new_x=selected_x,
                        new_y=selected_y,
                    )

                    true_evals += selected_x.shape[0]
                    if true_evals >= args.max_fe:
                        break

                multisource._attach_ppo_targets(
                    episode_trajectory,
                    discount=float(args.discount),
                    gae_lambda=ppo_gae_lambda,
                )
                dim_trajectories.extend(episode_trajectory)
                dim_problem_count += 1

                print(
                    f"{problem_name}-{dim}D epoch {epoch + 1} done, "
                    f"true_evals={true_evals}, best_obj1={np.min(archive_y[:, 0]):.6f}"
                )

            if dim_trajectories:
                gate_matrix = np.stack(
                    [
                        sample["gate_weights"].detach().cpu().numpy().astype(np.float32, copy=False)
                        for sample in dim_trajectories
                    ],
                    axis=0,
                )
                gate_mean = gate_matrix.mean(axis=0)
                gate_std = gate_matrix.std(axis=0)
                gate_top1 = gate_matrix.argmax(axis=1)
                top1_gate_usage = np.bincount(gate_top1, minlength=gate_matrix.shape[1]).astype(np.float32)
                top1_gate_usage = top1_gate_usage / max(float(gate_top1.size), 1.0)

                expert_matrix = np.stack(
                    [
                        sample["expert_logits"].detach().cpu().numpy().astype(np.float32, copy=False)
                        for sample in dim_trajectories
                    ],
                    axis=0,
                )  # (T, N, E)
                expert_mean = expert_matrix.mean(axis=(0, 1))
                expert_std = expert_matrix.std(axis=(0, 1))
                expert_flat = expert_matrix.reshape(-1, expert_matrix.shape[-1])
                expert_corr = np.corrcoef(expert_flat, rowvar=False)
                expert_corr = np.nan_to_num(expert_corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

                loss_stats = _update_moe_from_episode_ppo(
                    model=model,
                    optimizer=optimizer,
                    episode_trajectory=dim_trajectories,
                    device=args.device,
                    ppo_epochs=ppo_epochs,
                    minibatch_size=ppo_minibatch_size,
                    clip_eps=ppo_clip_eps,
                    value_coef=ppo_value_coef,
                    entropy_coef=ppo_entropy_coef,
                    balance_coef=ppo_balance_coef,
                    grad_clip=ppo_grad_clip,
                    target_kl=ppo_target_kl,
                )
                epoch_total_losses.append(loss_stats["total_loss"])

                print(
                    f"Updated MoE-DeepIC PPO for {dim}D with {dim_problem_count} problems "
                    f"({len(dim_trajectories)} rollout steps), total_loss={loss_stats['total_loss']:.6f}, "
                    f"actor={loss_stats['actor_loss']:.6f}, critic={loss_stats['critic_loss']:.6f}, "
                    f"entropy={loss_stats['entropy_loss']:.6f}, approx_kl={loss_stats['approx_kl']:.6f}, "
                    f"balance={loss_stats.get('balance_loss', 0.0):.6f}, "
                    f"gate_ent={loss_stats.get('gate_entropy', 0.0):.6f}, "
                    f"ratio_mean={loss_stats['ratio_mean']:.6f}, ratio_std={loss_stats['ratio_std']:.6f}, "
                    f"ratio_min={loss_stats['ratio_min']:.6f}, ratio_max={loss_stats['ratio_max']:.6f}, "
                    f"adv_mean={loss_stats['adv_mean']:.6f}, adv_std={loss_stats['adv_std']:.6f}, "
                    f"adv_min={loss_stats['adv_min']:.6f}, adv_max={loss_stats['adv_max']:.6f} | "
                    f"gate_mean={_format_vector(gate_mean)}, gate_std={_format_vector(gate_std)} | "
                    f"top1_gate_usage={_format_vector(top1_gate_usage)} | "
                    f"expert_mean={_format_vector(expert_mean)}, expert_std={_format_vector(expert_std)} | "
                    f"expert_corr={np.array2string(expert_corr, precision=3, separator=', ', suppress_small=False)}"
                )

        epoch_mean = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        epoch_mean_loss = float(np.mean(epoch_total_losses)) if epoch_total_losses else 0.0
        epoch_mean_rewards.append(epoch_mean)
        epoch_mean_total_losses.append(epoch_mean_loss)
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
                f"moe_deepic_{_problem_slug(target_problem)}_{'self_only' if self_train_only else 'source_mix'}_ppo_epoch_{epoch + 1}.pth",
            )

    torch.save(model.state_dict(), model_path)
    print(f"MoE-DeepIC model saved to {model_path.name}")
    multisource._save_reward_log(
        reward_log_path,
        {
            "script": "moe_deepic_zdt1_demo.py",
            "mode": "train_moe_deepic_multisource_ppo",
            "target_problem": target_problem,
            "model_path": str(model_path),
            "training_problems": multisource.training_problems_for(target_problem, self_train_only=self_train_only),
            "source_dims": multisource.SOURCE_DIMS,
            "training_label": "moe_deepic_self_only" if self_train_only else "moe_deepic_source_mix",
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
            "ppo_gae_lambda": ppo_gae_lambda,
            "ppo_grad_clip": ppo_grad_clip,
            "ppo_target_kl": ppo_target_kl,
            "moe_balance_coef": ppo_balance_coef,
            "moe_experts": int(getattr(args, "moe_experts", 4)),
            "moe_temperature": float(getattr(args, "moe_temperature", 1.0)),
            "records": reward_records,
        },
    )
    print(f"Reward log saved to {reward_log_path}")
    return model


def load_or_train_moe_deepic(args, target_problem: str, self_train_only: bool = False) -> MoE_SimplifiedDeepIC:
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
        model = _build_moe(args)
        try:
            model.load_state_dict(multisource._torch_load(candidate_path, args.device))
        except RuntimeError as exc:
            print(f"Skipping incompatible MoE-DeepIC checkpoint {candidate_path.name}: {exc}")
            continue
        print(f"Using saved MoE-DeepIC model from {candidate_path.name}")
        return model

    return train_moe_deepic_multisource_ppo(args, target_problem, self_train_only=self_train_only)


def run_saea_moe_deepic_problem(
    args,
    target_problem: str,
    model: MoE_SimplifiedDeepIC,
    plot: bool = True,
    initial_archive_x: np.ndarray | None = None,
) -> dict:
    problem = multisource.nda.ZDTProblem(name=target_problem, dim=args.dim)
    ref_point = multisource.nsga_eic._reference_point(target_problem, args.dim)

    pretrain_entry = multisource.load_or_prepare_kan_surrogate(target_problem, args.dim, args)
    surrogates = pretrain_entry["models"]
    print(f"Prepared KAN surrogate on {target_problem}-{args.dim}D with {pretrain_entry['x'].shape[0]} samples.")

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
    if multisource._surrogate_model_name(args) == "gp":
        gp_surrogates = demo.fit_gp_surrogates(
            archive_x=uncertainty_x,
            archive_y=uncertainty_y,
            seed=args.seed + multisource._stable_seed(71, target_problem, args.dim),
        )

    true_evals = args.archive_size
    steps_to_run = (args.max_fe - true_evals) // args.k_eval
    hv_history: list[float] = []

    fronts, _ = demo.fast_non_dominated_sort(archive_y)
    front0 = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    hv_history.append(float(demo.hypervolume_2d(front0, ref_point)))

    for step in range(steps_to_run):
        archive_x_t = archive_x.copy()
        archive_y_t = archive_y.copy()

        offspring_x, offspring_pred = multisource.nsga_eic.generate_nsga2_pseudo_front(
            archive_x=archive_x_t,
            problem=problem,
            surrogates=surrogates,
            device=args.device,
            n_offspring=args.offspring_size,
            sigma=args.mutation_sigma,
            surrogate_nsga_steps=args.surrogate_nsga_steps,
            predict_fn=demo.predict_with_kan,
            generate_fn=demo.generate_offspring,
        )
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
        model.eval()
        with torch.no_grad():
            encoded = model.encode(
                x_true=torch.as_tensor(archive_x_t, dtype=torch.float32, device=args.device),
                y_true=torch.as_tensor(archive_y_t, dtype=torch.float32, device=args.device),
                x_sur=torch.as_tensor(offspring_x, dtype=torch.float32, device=args.device),
                y_sur=torch.as_tensor(offspring_pred, dtype=torch.float32, device=args.device),
                sigma_sur=torch.as_tensor(offspring_sigma, dtype=torch.float32, device=args.device),
                progress=progress,
                lower_bound=problem.lower,
                upper_bound=problem.upper,
            )
            dec_out = model._actor_logits_full(encoded["H_surr"], encoded["progress"])
            logits = dec_out["logits"][0]
            ranking = torch.argsort(logits, descending=True).detach().cpu().numpy()

        selected_idx = ranking[: args.k_eval]
        selected_x = offspring_x[selected_idx]
        selected_y = problem.evaluate(selected_x)

        archive_x, archive_y = demo.update_archive(
            archive_x=archive_x_t,
            archive_y=archive_y_t,
            new_x=selected_x,
            new_y=selected_y,
        )
        uncertainty_x, uncertainty_y = demo.update_uncertainty_archive(
            uncertainty_x=uncertainty_x,
            uncertainty_y=uncertainty_y,
            new_x=selected_x,
            new_y=selected_y,
        )

        true_evals += selected_x.shape[0]
        fronts, _ = demo.fast_non_dominated_sort(archive_y)
        front0 = archive_y[np.asarray(fronts[0], dtype=np.int64)]
        hv_history.append(float(demo.hypervolume_2d(front0, ref_point)))
        print(
            f"Step {step + 1:03d} | true_evals={true_evals:4d} | front0={front0.shape[0]:3d} | "
            f"HV={hv_history[-1]:.6f} | progress={progress:.3f}"
        )

        if true_evals >= args.max_fe:
            break

    result = {
        "hv_history": hv_history,
        "ref_point": ref_point,
        "final_front": demo.DeepICClass.pareto_front(np.asarray(archive_y, dtype=np.float32)),
        "true_front": multisource.load_true_pareto_front(target_problem),
    }

    if plot:
        plt.figure(figsize=(8, 5))
        plt.title(f"{args.dim}D {target_problem} Hypervolume (MoE-DeepIC)")
        plt.plot(hv_history, marker="o")
        plt.xlabel("Step")
        plt.ylabel("Hypervolume")
        plt.grid(True)
        plt.show()

    return result


def run_comparison(args, target_problem: str, self_train_only: bool = False) -> None:
    model = load_or_train_moe_deepic(args, target_problem, self_train_only=self_train_only)
    problem = multisource.nda.ZDTProblem(name=target_problem, dim=args.dim)
    shared_init_x = multisource.latin_hypercube_sample(
        lower=problem.lower,
        upper=problem.upper,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )

    moe_result = run_saea_moe_deepic_problem(
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

    print(f"\nSAEA-MoE-DeepIC final HV: {moe_result['hv_history'][-1]:.6f}")
    print(f"NSGA-EIC final HV: {eic_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {moe_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title(f"{args.dim}D {target_problem} Hypervolume Comparison")
    plt.plot(moe_result["hv_history"], marker="o", label="SAEA-MoE-DeepIC")
    plt.plot(eic_result["hv_history"], marker="s", label="NSGA-EIC")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend()
    plt.show()

    multisource.nsga_eic._plot_front_comparison(
        f"{args.dim}D {target_problem} Pareto Front Comparison",
        moe_result["final_front"],
        "SAEA-MoE-DeepIC",
        eic_result["final_front"],
        "NSGA-EIC",
        moe_result["true_front"],
    )


def _parse_args():
    args = multisource.parse_args(TARGET_PROBLEM)
    if "--train_algo" not in sys.argv[1:]:
        args.train_algo = "ppo"
    return args


def main():
    args = _parse_args()
    if args.dim != 30:
        print(f"Warning: expected 30D evaluation for {TARGET_PROBLEM}, but received dim={args.dim}.")

    if args.archive_size != base_demo.INITIAL_SURROGATE_ARCHIVE_SIZE:
        print(
            f"Warning: this demo initializes a surrogate archive of {base_demo.INITIAL_SURROGATE_ARCHIVE_SIZE} "
            f"individuals while archive_size={args.archive_size}."
        )

    if args.train_only:
        train_moe_deepic_multisource_ppo(args, TARGET_PROBLEM, self_train_only=True)
    else:
        run_comparison(args, TARGET_PROBLEM, self_train_only=True)


if __name__ == "__main__":
    main()
