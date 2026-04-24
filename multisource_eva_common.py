import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

import demo
from agent.deepic_agent import ppo_loss as deepic_ppo_loss
from problem.kan import KAN


def load_module(filename: str, module_name: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nda = load_module("nsga-nda.py", "nsga_nda_module")
nsga_eic = load_module("nsga-eic.py", "nsga_eic_module")


ALL_PROBLEMS = ["ZDT1", "ZDT2", "ZDT3", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]
SOURCE_DIMS = [15, 20, 25]
TRAIN_EPOCHS = 50
REWARD_LOG_DIR = Path(__file__).resolve().parent / "reward_logs"
KAGGLE_MODEL_DIR = Path("/kaggle/working/DeepIC_Models")
COLAB_MODEL_DIR = Path("/content/drive/MyDrive/DeepIC_Models")
LOCAL_MODEL_DIR = Path("./DeepIC_Models")


def _problem_slug(problem_name: str) -> str:
    return problem_name.lower()


def _save_reward_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_colab_model_checkpoint(state_dict, filename: str) -> Path | None:
    try:
        if "KAGGLE_URL_BASE" in os.environ:
            save_dir = KAGGLE_MODEL_DIR
        elif Path("/content").exists():
            save_dir = COLAB_MODEL_DIR
            if not Path("/content/drive").exists():
                try:
                    from google.colab import drive
                except ImportError:
                    save_dir = LOCAL_MODEL_DIR
                else:
                    drive.mount("/content/drive")
        else:
            save_dir = LOCAL_MODEL_DIR

        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
        demo.torch.save(state_dict, save_path)
        print(f"--> [Auto-Save] Saved checkpoint to: {save_path}")
        return save_path
    except Exception as exc:
        print(f"Skipping auto-save checkpoint: {exc}")
        return None


def _torch_load(path: Path | str, map_location: str):
    try:
        return demo.torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return demo.torch.load(path, map_location=map_location)


def _move_models_to_device(models, device: str):
    for model in models:
        if hasattr(model, "to"):
            model.to(device)
        if hasattr(model, "device"):
            model.device = device
    return models


def _serialize_kan_model(model) -> dict:
    return {
        "state_dict": model.state_dict(),
        "config": {
            "width": list(model.width),
            "grid": int(model.grid),
            "k": int(model.k),
            "mult_arity": model.mult_arity,
            "base_fun": model.base_fun_name,
            "symbolic_enabled": bool(model.symbolic_enabled),
            "affine_trainable": bool(model.affine_trainable),
            "grid_eps": float(model.grid_eps),
            "grid_range": list(model.grid_range),
            "sp_trainable": bool(model.sp_trainable),
            "sb_trainable": bool(model.sb_trainable),
            "seed": int(getattr(model, "seed", 1)),
            "save_act": bool(model.save_act),
            "sparse_init": bool(getattr(model, "sparse_init", False)),
        },
    }


def _deserialize_kan_model(payload: dict, device: str):
    config = dict(payload["config"])
    model = KAN(
        width=config["width"],
        grid=config["grid"],
        k=config["k"],
        mult_arity=config["mult_arity"],
        base_fun=config["base_fun"],
        symbolic_enabled=config["symbolic_enabled"],
        affine_trainable=config["affine_trainable"],
        grid_eps=config["grid_eps"],
        grid_range=config["grid_range"],
        sp_trainable=config["sp_trainable"],
        sb_trainable=config["sb_trainable"],
        seed=config["seed"],
        save_act=config["save_act"],
        sparse_init=config.get("sparse_init", False),
        auto_save=False,
        device=device,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    return model


def _stable_seed(base_seed: int, problem_name: str, dim: int) -> int:
    return base_seed + dim * 1000 + sum(ord(ch) for ch in problem_name)


def _kan_checkpoint_path(problem_name: str, dim: int) -> Path:
    return Path(__file__).resolve().parent / f"kan_{_problem_slug(problem_name)}_{dim}d.pth"


def _training_label(self_train_only: bool = False) -> str:
    return "self_only" if self_train_only else "source_mix"


def _script_variant(self_train_only: bool = False) -> str:
    return "demo" if self_train_only else "eva"


def _epoch_checkpoint_path(problem_name: str, epoch_number: int, self_train_only: bool = False) -> Path:
    if self_train_only:
        return Path(__file__).resolve().parent / f"{_problem_slug(problem_name)}_self_model_epoch_{epoch_number}.pth"
    return Path(__file__).resolve().parent / f"{_problem_slug(problem_name)}_model_epoch_{epoch_number}.pth"


def _final_model_path(problem_name: str, self_train_only: bool = False) -> Path:
    if self_train_only:
        return Path(__file__).resolve().parent / f"deepic_{_problem_slug(problem_name)}_self_only.pth"
    return Path(__file__).resolve().parent / f"deepic_{_problem_slug(problem_name)}_source_mix.pth"


def _reward_log_path(problem_name: str, self_train_only: bool = False) -> Path:
    return REWARD_LOG_DIR / f"{_problem_slug(problem_name)}_{_script_variant(self_train_only)}_train_rewards.json"


def build_args_namespace(parsed) -> SimpleNamespace:
    return SimpleNamespace(
        dim=parsed.dim,
        archive_size=parsed.archive_size,
        offspring_size=parsed.offspring_size,
        k_eval=parsed.k_eval,
        max_fe=parsed.max_fe,
        mutation_sigma=parsed.mutation_sigma,
        kan_steps=parsed.kan_steps,
        kan_hidden=parsed.kan_hidden,
        kan_grid=parsed.kan_grid,
        deepic_hidden=parsed.deepic_hidden,
        deepic_heads=parsed.deepic_heads,
        deepic_ff=parsed.deepic_ff,
        deepic_lr=parsed.deepic_lr,
        deepic_adapt_steps=parsed.deepic_adapt_steps,
        surrogate_nsga_steps=parsed.surrogate_nsga_steps,
        discount=parsed.discount,
        seed=parsed.seed,
        device=parsed.device,
    )


def latin_hypercube_sample(lower, upper, n_samples: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower_arr = np.full(dim, lower, dtype=np.float32) if np.isscalar(lower) else np.asarray(lower, dtype=np.float32)
    upper_arr = np.full(dim, upper, dtype=np.float32) if np.isscalar(upper) else np.asarray(upper, dtype=np.float32)

    lhs = np.empty((n_samples, dim), dtype=np.float32)
    for j in range(dim):
        perm = rng.permutation(n_samples)
        lhs[:, j] = (perm + rng.random(n_samples)) / n_samples

    samples = lower_arr + lhs * (upper_arr - lower_arr)
    return samples.astype(np.float32)


def source_problems_for(target_problem: str) -> list[str]:
    return [problem_name for problem_name in ALL_PROBLEMS if problem_name != target_problem]


def training_problems_for(target_problem: str, self_train_only: bool = False) -> list[str]:
    if self_train_only:
        return [target_problem]
    return source_problems_for(target_problem)


def load_or_prepare_kan_surrogate(problem_name: str, dim: int, args) -> dict:
    save_path = _kan_checkpoint_path(problem_name, dim)

    if save_path.exists():
        payload = _torch_load(save_path, args.device)
        if not isinstance(payload, dict) or "models" not in payload or "x_data" not in payload or "y_data" not in payload:
            raise ValueError(f"Invalid KAN checkpoint format: {save_path}")
        print(f"Loaded stored KAN surrogate from {save_path.name}")
        stored_models = payload["models"]
        if stored_models and isinstance(stored_models[0], dict) and "state_dict" in stored_models[0]:
            models = [_deserialize_kan_model(item, args.device) for item in stored_models]
        else:
            models = _move_models_to_device(stored_models, args.device)
        return {
            "problem": nda.ZDTProblem(name=problem_name, dim=dim),
            "x": np.asarray(payload["x_data"], dtype=np.float32),
            "y": np.asarray(payload["y_data"], dtype=np.float32),
            "models": models,
        }

    print(f"Stored KAN surrogate not found for {problem_name}-{dim}D. Pre-training and saving it now...")
    problem = nda.ZDTProblem(name=problem_name, dim=dim)
    x_data, y_data, models = nda.pre_train_kan_surrogate_for_problem(
        problem=problem,
        device=args.device,
        kan_steps=args.kan_steps,
        hidden_width=args.kan_hidden,
        grid=args.kan_grid,
        seed=_stable_seed(args.seed, problem_name, dim),
    )
    models = _move_models_to_device(models, args.device)
    demo.torch.save(
        {
            "models": [_serialize_kan_model(model) for model in models],
            "x_data": x_data,
            "y_data": y_data,
            "problem_name": problem_name,
            "dim": dim,
        },
        save_path,
    )
    print(f"Saved KAN surrogate to {save_path.name}")
    return {
        "problem": problem,
        "x": x_data,
        "y": y_data,
        "models": models,
    }


def pretrain_source_surrogates(args, target_problem: str, self_train_only: bool = False) -> dict:
    cache = {}
    for dim in SOURCE_DIMS:
        for problem_name in training_problems_for(target_problem, self_train_only=self_train_only):
            cache[(problem_name, dim)] = load_or_prepare_kan_surrogate(problem_name, dim, args)
    return cache


def _attach_discounted_returns(episode_trajectory: list[dict], discount: float) -> None:
    if not episode_trajectory:
        return

    returns: list[float] = []
    discounted_return = 0.0
    for sample in reversed(episode_trajectory):
        discounted_return = float(sample["reward"]) + float(discount) * discounted_return
        returns.append(float(discounted_return))
    returns.reverse()

    returns_array = np.asarray(returns, dtype=np.float32)
    if returns_array.size > 1:
        returns_std = float(returns_array.std())
        if returns_std > 1e-8:
            returns_array = (returns_array - float(returns_array.mean())) / returns_std

    for sample, normalized_return in zip(episode_trajectory, returns_array.tolist()):
        sample["return"] = float(normalized_return)


def _bound_group_key(bound) -> tuple[float, ...]:
    bound_arr = np.asarray(bound, dtype=np.float32).reshape(-1)
    return tuple(float(x) for x in bound_arr.tolist())


def _update_deepic_from_episode(
    model,
    optimizer,
    episode_trajectory: list[dict],
    device: str,
    top_k: int,
) -> float:
    if not episode_trajectory:
        return 0.0

    returns = [float(sample["return"]) for sample in episode_trajectory]

    model.train()
    optimizer.zero_grad()
    episode_loss = None
    total_samples = len(episode_trajectory)

    grouped_indices: dict[tuple[tuple[int, ...], ...], list[int]] = {}
    for idx, sample in enumerate(episode_trajectory):
        key = (
            sample["archive_x"].shape,
            sample["archive_y"].shape,
            sample["offspring_x"].shape,
            sample["offspring_pred"].shape,
            sample["offspring_sigma"].shape,
            _bound_group_key(sample["lower"]),
            _bound_group_key(sample["upper"]),
        )
        grouped_indices.setdefault(key, []).append(idx)

    for indices in grouped_indices.values():
        batch_samples = [episode_trajectory[idx] for idx in indices]
        ranking = demo.torch.as_tensor(
            np.stack([sample["ranking"] for sample in batch_samples], axis=0),
            dtype=demo.torch.long,
            device=device,
        )
        out = model(
            x_true=demo.torch.as_tensor(
                np.stack([sample["archive_x"] for sample in batch_samples], axis=0),
                dtype=demo.torch.float32,
                device=device,
            ),
            y_true=demo.torch.as_tensor(
                np.stack([sample["archive_y"] for sample in batch_samples], axis=0),
                dtype=demo.torch.float32,
                device=device,
            ),
            x_sur=demo.torch.as_tensor(
                np.stack([sample["offspring_x"] for sample in batch_samples], axis=0),
                dtype=demo.torch.float32,
                device=device,
            ),
            y_sur=demo.torch.as_tensor(
                np.stack([sample["offspring_pred"] for sample in batch_samples], axis=0),
                dtype=demo.torch.float32,
                device=device,
            ),
            sigma_sur=demo.torch.as_tensor(
                np.stack([sample["offspring_sigma"] for sample in batch_samples], axis=0),
                dtype=demo.torch.float32,
                device=device,
            ),
            progress=demo.torch.as_tensor(
                [sample["progress"] for sample in batch_samples],
                dtype=demo.torch.float32,
                device=device,
            ),
            lower_bound=batch_samples[0]["lower"],
            upper_bound=batch_samples[0]["upper"],
            target_ranking=ranking,
            decode_type="greedy",
        )

        log_prob = model.sequence_log_prob(out["logits"], ranking, top_k=top_k)
        advantage = demo.torch.as_tensor(
            [returns[idx] for idx in indices],
            dtype=demo.torch.float32,
            device=device,
        )
        batch_loss = -(advantage.detach() * log_prob).sum()
        episode_loss = batch_loss if episode_loss is None else episode_loss + batch_loss

    if episode_loss is None:
        return 0.0

    episode_loss = episode_loss / max(total_samples, 1)
    episode_loss.backward()
    optimizer.step()
    return float(episode_loss.detach().cpu())


def _attach_ppo_targets(
    episode_trajectory: list[dict],
    discount: float,
    gae_lambda: float,
) -> None:
    if not episode_trajectory:
        return

    values = np.asarray([float(sample["value"]) for sample in episode_trajectory], dtype=np.float32)
    next_values = np.asarray([float(sample["next_value"]) for sample in episode_trajectory], dtype=np.float32)
    rewards = np.asarray([float(sample["reward"]) for sample in episode_trajectory], dtype=np.float32)
    dones = np.asarray([float(sample["done"]) for sample in episode_trajectory], dtype=np.float32)

    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for idx in range(len(episode_trajectory) - 1, -1, -1):
        not_done = 1.0 - dones[idx]
        delta = rewards[idx] + float(discount) * next_values[idx] * not_done - values[idx]
        gae = delta + float(discount) * float(gae_lambda) * not_done * gae
        advantages[idx] = gae

    returns_array = advantages + values
    advantages_norm = advantages.copy()
    if advantages_norm.size > 1:
        adv_std = float(advantages_norm.std())
        if adv_std > 1e-8:
            advantages_norm = (advantages_norm - float(advantages_norm.mean())) / adv_std

    for sample, ret_value, adv_value in zip(
        episode_trajectory,
        returns_array.tolist(),
        advantages_norm.tolist(),
    ):
        sample["return"] = float(ret_value)
        sample["advantage"] = float(adv_value)


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
    grad_clip: float | None,
) -> dict[str, float]:
    if not episode_trajectory:
        return {
            "total_loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy_loss": 0.0,
            "approx_kl": 0.0,
            "updates": 0.0,
        }

    grouped_indices: dict[tuple[tuple[int, ...], ...], list[int]] = {}
    for idx, sample in enumerate(episode_trajectory):
        key = (
            tuple(sample["archive_x"].shape),
            tuple(sample["archive_y"].shape),
            tuple(sample["offspring_x"].shape),
            tuple(sample["offspring_pred"].shape),
            tuple(sample["offspring_sigma"].shape),
            tuple(sample["actions"].shape),
            _bound_group_key(sample["lower"]),
            _bound_group_key(sample["upper"]),
        )
        grouped_indices.setdefault(key, []).append(idx)

    stats = {
        "total_loss": 0.0,
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy_loss": 0.0,
        "approx_kl": 0.0,
        "updates": 0.0,
    }

    for indices in grouped_indices.values():
        group_samples = [episode_trajectory[idx] for idx in indices]
        group_size = len(group_samples)
        if group_size == 0:
            continue

        archive_x = demo.torch.as_tensor(
            np.stack([sample["archive_x"] for sample in group_samples], axis=0),
            dtype=demo.torch.float32,
            device=device,
        )
        archive_y = demo.torch.as_tensor(
            np.stack([sample["archive_y"] for sample in group_samples], axis=0),
            dtype=demo.torch.float32,
            device=device,
        )
        offspring_x = demo.torch.as_tensor(
            np.stack([sample["offspring_x"] for sample in group_samples], axis=0),
            dtype=demo.torch.float32,
            device=device,
        )
        offspring_pred = demo.torch.as_tensor(
            np.stack([sample["offspring_pred"] for sample in group_samples], axis=0),
            dtype=demo.torch.float32,
            device=device,
        )
        offspring_sigma = demo.torch.as_tensor(
            np.stack([sample["offspring_sigma"] for sample in group_samples], axis=0),
            dtype=demo.torch.float32,
            device=device,
        )
        progress = demo.torch.as_tensor(
            [sample["progress"] for sample in group_samples],
            dtype=demo.torch.float32,
            device=device,
        ).reshape(group_size, -1)
        actions = demo.torch.stack(
            [sample["actions"].to(device=device, dtype=demo.torch.long) for sample in group_samples],
            dim=0,
        )
        old_logprob = demo.torch.stack(
            [sample["old_logprob"].to(device=device, dtype=demo.torch.float32) for sample in group_samples],
            dim=0,
        ).reshape(-1)
        advantages = demo.torch.as_tensor(
            [sample["advantage"] for sample in group_samples],
            dtype=demo.torch.float32,
            device=device,
        )
        returns = demo.torch.as_tensor(
            [sample["return"] for sample in group_samples],
            dtype=demo.torch.float32,
            device=device,
        )

        for _ in range(max(int(ppo_epochs), 1)):
            perm = demo.torch.randperm(group_size, device=device)
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
                loss_dict = deepic_ppo_loss(
                    agent=model,
                    batch=batch,
                    clip_eps=clip_eps,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                )
                optimizer.zero_grad()
                loss_dict["total_loss"].backward()
                if grad_clip is not None and float(grad_clip) > 0.0:
                    demo.torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                optimizer.step()

                stats["total_loss"] += float(loss_dict["total_loss"].detach().cpu())
                stats["actor_loss"] += float(loss_dict["actor_loss"].detach().cpu())
                stats["critic_loss"] += float(loss_dict["critic_loss"].detach().cpu())
                stats["entropy_loss"] += float(loss_dict["entropy_loss"].detach().cpu())
                stats["approx_kl"] += float(loss_dict["approx_kl"].detach().cpu())
                stats["updates"] += 1.0

    if stats["updates"] > 0:
        for key in ["total_loss", "actor_loss", "critic_loss", "entropy_loss", "approx_kl"]:
            stats[key] /= stats["updates"]

    return stats


def train_deepic_multisource(args, target_problem: str, self_train_only: bool = False):
    demo.set_seed(args.seed)
    print(
        f"Training config | deepic_lr={args.deepic_lr:.1e} | "
        f"surrogate_nsga_steps={args.surrogate_nsga_steps} | discount={args.discount:.4f}"
    )
    pretrain_cache = pretrain_source_surrogates(args, target_problem, self_train_only=self_train_only)
    reward_records: list[dict] = []
    epoch_mean_rewards: list[float] = []
    model_path = _final_model_path(target_problem, self_train_only=self_train_only)
    reward_log_path = _reward_log_path(target_problem, self_train_only=self_train_only)

    deepic = demo.DeepICClass(
        hidden_dim=args.deepic_hidden,
        n_heads=args.deepic_heads,
        ff_dim=args.deepic_ff,
    ).to(args.device)
    deepic_optimizer = demo.torch.optim.Adam(deepic.parameters(), lr=args.deepic_lr)

    if args.start_epoch > 0:
        checkpoint_path = _epoch_checkpoint_path(target_problem, args.start_epoch, self_train_only=self_train_only)
        if checkpoint_path.exists():
            deepic.load_state_dict(_torch_load(checkpoint_path, args.device))
            print(f"Loaded model from {checkpoint_path.name}")
        else:
            print(f"Checkpoint {checkpoint_path.name} not found, starting from scratch")

    for epoch in range(args.start_epoch, TRAIN_EPOCHS):
        print(f"Epoch {epoch + 1}/{TRAIN_EPOCHS}")
        epoch_rewards: list[float] = []

        for dim in SOURCE_DIMS:
            dim_trajectories: list[dict] = []
            dim_problem_count = 0
            for problem_name in training_problems_for(target_problem, self_train_only=self_train_only):
                entry = pretrain_cache[(problem_name, dim)]
                problem = entry["problem"]
                surrogates = entry["models"]
                episode_trajectory: list[dict] = []

                archive_x = latin_hypercube_sample(
                    lower=problem.lower,
                    upper=problem.upper,
                    n_samples=args.archive_size,
                    dim=dim,
                    seed=args.seed + epoch * 10000 + _stable_seed(0, problem_name, dim),
                )
                archive_y = problem.evaluate(archive_x)
                true_evals = args.archive_size
                remaining_budget = args.max_fe - true_evals
                steps_to_run = remaining_budget // args.k_eval

                for step in range(steps_to_run):
                    archive_x_t = archive_x.copy()
                    archive_y_t = archive_y.copy()

                    offspring_x, offspring_pred = nsga_eic.generate_nsga2_pseudo_front(
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
                    archive_pred = demo.predict_with_kan(surrogates, archive_x_t, args.device).astype(np.float32)
                    offspring_sigma = demo.estimate_uncertainty(
                        archive_x=archive_x_t,
                        archive_y=archive_y_t,
                        archive_pred=archive_pred,
                        offspring_x=offspring_x,
                    ).astype(np.float32)

                    progress = float(true_evals / args.max_fe)
                    ranking = demo.infer_deepic_ranking(
                        model=deepic,
                        archive_x=archive_x_t,
                        archive_y=archive_y_t,
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

                    reward = demo.DeepICClass.fpareto_improvement_reward(
                        previous_front=archive_y_t,
                        selected_objectives=selected_y,
                    )
                    reward_value = float(reward)
                    epoch_rewards.append(reward_value)
                    reward_records.append(
                        {
                            "epoch": epoch + 1,
                            "problem": problem_name,
                            "dim": int(dim),
                            "step": step + 1,
                            "progress": float(progress),
                            "reward": reward_value,
                        }
                    )

                    episode_trajectory.append(
                        {
                            "archive_x": archive_x_t,
                            "archive_y": archive_y_t,
                            "offspring_x": offspring_x,
                            "offspring_pred": offspring_pred,
                            "offspring_sigma": offspring_sigma,
                            "ranking": ranking,
                            "reward": reward_value,
                            "progress": progress,
                            "lower": problem.lower,
                            "upper": problem.upper,
                        }
                    )

                    archive_x, archive_y = demo.update_archive(
                        archive_x=archive_x_t,
                        archive_y=archive_y_t,
                        new_x=selected_x,
                        new_y=selected_y,
                    )

                    true_evals += args.k_eval
                    if true_evals >= args.max_fe:
                        break

                _attach_discounted_returns(episode_trajectory, float(args.discount))
                dim_trajectories.extend(episode_trajectory)
                dim_problem_count += 1

                print(
                    f"{problem_name}-{dim}D epoch {epoch + 1} done, "
                    f"true_evals={true_evals}, best_obj1={np.min(archive_y[:, 0]):.6f}"
                )

            if dim_trajectories:
                dim_loss = _update_deepic_from_episode(
                    model=deepic,
                    optimizer=deepic_optimizer,
                    episode_trajectory=dim_trajectories,
                    device=args.device,
                    top_k=args.k_eval,
                )
                print(
                    f"Updated DeepIC for {dim}D with {dim_problem_count} problems "
                    f"({len(dim_trajectories)} rollout steps), loss={dim_loss:.6f}"
                )

        epoch_mean = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        epoch_mean_rewards.append(epoch_mean)
        print(f"Epoch {epoch + 1} mean reward: {epoch_mean:.6f}")
        demo.torch.save(
            deepic.state_dict(),
            _epoch_checkpoint_path(target_problem, epoch + 1, self_train_only=self_train_only),
        )
        if (epoch + 1) % 5 == 0:
            save_colab_model_checkpoint(
                deepic.state_dict(),
                f"deepic_{_problem_slug(target_problem)}_{_training_label(self_train_only)}_epoch_{epoch + 1}.pth",
            )

    demo.torch.save(deepic.state_dict(), model_path)
    print(f"DeepIC model saved to {model_path.name}")
    _save_reward_log(
        reward_log_path,
        {
            "script": f"{_problem_slug(target_problem)}_{_script_variant(self_train_only)}.py",
            "mode": "train_deepic_multisource",
            "target_problem": target_problem,
            "model_path": str(model_path),
            "training_problems": training_problems_for(target_problem, self_train_only=self_train_only),
            "source_dims": SOURCE_DIMS,
            "training_label": _training_label(self_train_only),
            "epoch_mean_rewards": epoch_mean_rewards,
            "records": reward_records,
        },
    )
    print(f"Reward log saved to {reward_log_path}")
    return deepic


def train_deepic_multisource_ppo(args, target_problem: str, self_train_only: bool = False):
    demo.set_seed(args.seed)
    print(
        f"Training config (PPO) | deepic_lr={args.deepic_lr:.1e} | "
        f"surrogate_nsga_steps={args.surrogate_nsga_steps} | discount={args.discount:.4f} | "
        f"ppo_epochs={getattr(args, 'ppo_epochs', 4)} | "
        f"ppo_clip_eps={getattr(args, 'ppo_clip_eps', 0.2):.3f}"
    )
    pretrain_cache = pretrain_source_surrogates(args, target_problem, self_train_only=self_train_only)
    reward_records: list[dict] = []
    epoch_mean_rewards: list[float] = []
    epoch_mean_total_losses: list[float] = []
    model_path = _final_model_path(target_problem, self_train_only=self_train_only)
    reward_log_path = _reward_log_path(target_problem, self_train_only=self_train_only)

    deepic = demo.DeepICClass(
        hidden_dim=args.deepic_hidden,
        n_heads=args.deepic_heads,
        ff_dim=args.deepic_ff,
    ).to(args.device)
    if not hasattr(deepic, "act") or not hasattr(deepic, "evaluate_actions"):
        raise TypeError("PPO training requires DeepICClass to implement act() and evaluate_actions().")
    deepic_optimizer = demo.torch.optim.Adam(deepic.parameters(), lr=args.deepic_lr)

    if args.start_epoch > 0:
        checkpoint_path = _epoch_checkpoint_path(target_problem, args.start_epoch, self_train_only=self_train_only)
        if checkpoint_path.exists():
            deepic.load_state_dict(_torch_load(checkpoint_path, args.device))
            print(f"Loaded model from {checkpoint_path.name}")
        else:
            print(f"Checkpoint {checkpoint_path.name} not found, starting from scratch")

    ppo_epochs = int(getattr(args, "ppo_epochs", 4))
    ppo_minibatch_size = int(getattr(args, "ppo_minibatch_size", 32))
    ppo_clip_eps = float(getattr(args, "ppo_clip_eps", 0.2))
    ppo_value_coef = float(getattr(args, "ppo_value_coef", 0.5))
    ppo_entropy_coef = float(getattr(args, "ppo_entropy_coef", 0.01))
    ppo_gae_lambda = float(getattr(args, "ppo_gae_lambda", 0.95))
    ppo_grad_clip = float(getattr(args, "ppo_grad_clip", 1.0))

    for epoch in range(args.start_epoch, TRAIN_EPOCHS):
        print(f"Epoch {epoch + 1}/{TRAIN_EPOCHS}")
        epoch_rewards: list[float] = []
        epoch_total_losses: list[float] = []

        for dim in SOURCE_DIMS:
            dim_trajectories: list[dict] = []
            dim_problem_count = 0
            for problem_name in training_problems_for(target_problem, self_train_only=self_train_only):
                entry = pretrain_cache[(problem_name, dim)]
                problem = entry["problem"]
                surrogates = entry["models"]
                episode_trajectory: list[dict] = []

                archive_x = latin_hypercube_sample(
                    lower=problem.lower,
                    upper=problem.upper,
                    n_samples=args.archive_size,
                    dim=dim,
                    seed=args.seed + epoch * 10000 + _stable_seed(0, problem_name, dim),
                )
                archive_y = problem.evaluate(archive_x)
                true_evals = args.archive_size
                remaining_budget = args.max_fe - true_evals
                steps_to_run = remaining_budget // args.k_eval

                for step in range(steps_to_run):
                    archive_x_t = archive_x.copy()
                    archive_y_t = archive_y.copy()

                    offspring_x, offspring_pred = nsga_eic.generate_nsga2_pseudo_front(
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
                    archive_pred = demo.predict_with_kan(surrogates, archive_x_t, args.device).astype(np.float32)
                    offspring_sigma = demo.estimate_uncertainty(
                        archive_x=archive_x_t,
                        archive_y=archive_y_t,
                        archive_pred=archive_pred,
                        offspring_x=offspring_x,
                    ).astype(np.float32)

                    progress = float(true_evals / args.max_fe)
                    with demo.torch.no_grad():
                        encoded = deepic.encode(
                            x_true=demo.torch.as_tensor(archive_x_t, dtype=demo.torch.float32, device=args.device),
                            y_true=demo.torch.as_tensor(archive_y_t, dtype=demo.torch.float32, device=args.device),
                            x_sur=demo.torch.as_tensor(offspring_x, dtype=demo.torch.float32, device=args.device),
                            y_sur=demo.torch.as_tensor(offspring_pred, dtype=demo.torch.float32, device=args.device),
                            sigma_sur=demo.torch.as_tensor(offspring_sigma, dtype=demo.torch.float32, device=args.device),
                            progress=progress,
                            lower_bound=problem.lower,
                            upper_bound=problem.upper,
                        )
                        act_out = deepic.act(
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
                        demo.DeepICClass.fpareto_improvement_reward(
                            previous_front=archive_y_t,
                            selected_objectives=selected_y,
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
                            "train_algo": "ppo",
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
                            "old_logprob": act_out["logprob"][0].detach().cpu(),
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

                    true_evals += args.k_eval
                    if true_evals >= args.max_fe:
                        break

                _attach_ppo_targets(
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
                loss_stats = _update_deepic_from_episode_ppo(
                    model=deepic,
                    optimizer=deepic_optimizer,
                    episode_trajectory=dim_trajectories,
                    device=args.device,
                    ppo_epochs=ppo_epochs,
                    minibatch_size=ppo_minibatch_size,
                    clip_eps=ppo_clip_eps,
                    value_coef=ppo_value_coef,
                    entropy_coef=ppo_entropy_coef,
                    grad_clip=ppo_grad_clip,
                )
                epoch_total_losses.append(loss_stats["total_loss"])
                print(
                    f"Updated DeepIC PPO for {dim}D with {dim_problem_count} problems "
                    f"({len(dim_trajectories)} rollout steps), total_loss={loss_stats['total_loss']:.6f}, "
                    f"actor={loss_stats['actor_loss']:.6f}, critic={loss_stats['critic_loss']:.6f}, "
                    f"entropy={loss_stats['entropy_loss']:.6f}, approx_kl={loss_stats['approx_kl']:.6f}"
                )

        epoch_mean = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        epoch_mean_loss = float(np.mean(epoch_total_losses)) if epoch_total_losses else 0.0
        epoch_mean_rewards.append(epoch_mean)
        epoch_mean_total_losses.append(epoch_mean_loss)
        print(f"Epoch {epoch + 1} mean reward: {epoch_mean:.6f}")
        print(f"Epoch {epoch + 1} mean PPO total loss: {epoch_mean_loss:.6f}")
        demo.torch.save(
            deepic.state_dict(),
            _epoch_checkpoint_path(target_problem, epoch + 1, self_train_only=self_train_only),
        )
        if (epoch + 1) % 5 == 0:
            save_colab_model_checkpoint(
                deepic.state_dict(),
                f"deepic_{_problem_slug(target_problem)}_{_training_label(self_train_only)}_ppo_epoch_{epoch + 1}.pth",
            )

    demo.torch.save(deepic.state_dict(), model_path)
    print(f"DeepIC model saved to {model_path.name}")
    _save_reward_log(
        reward_log_path,
        {
            "script": f"{_problem_slug(target_problem)}_{_script_variant(self_train_only)}.py",
            "mode": "train_deepic_multisource_ppo",
            "target_problem": target_problem,
            "model_path": str(model_path),
            "training_problems": training_problems_for(target_problem, self_train_only=self_train_only),
            "source_dims": SOURCE_DIMS,
            "training_label": _training_label(self_train_only),
            "epoch_mean_rewards": epoch_mean_rewards,
            "epoch_mean_total_losses": epoch_mean_total_losses,
            "ppo_epochs": ppo_epochs,
            "ppo_minibatch_size": ppo_minibatch_size,
            "ppo_clip_eps": ppo_clip_eps,
            "ppo_value_coef": ppo_value_coef,
            "ppo_entropy_coef": ppo_entropy_coef,
            "ppo_gae_lambda": ppo_gae_lambda,
            "ppo_grad_clip": ppo_grad_clip,
            "records": reward_records,
        },
    )
    print(f"Reward log saved to {reward_log_path}")
    return deepic


def load_or_train_deepic(args, target_problem: str, self_train_only: bool = False):
    model_path = _final_model_path(target_problem, self_train_only=self_train_only)
    if model_path.exists():
        print(f"Using saved DeepIC model from {model_path.name}")
        deepic = demo.DeepICClass(
            hidden_dim=args.deepic_hidden,
            n_heads=args.deepic_heads,
            ff_dim=args.deepic_ff,
        ).to(args.device)
        deepic.load_state_dict(_torch_load(model_path, args.device))
        return deepic
    if getattr(args, "train_algo", "reinforce") == "ppo":
        return train_deepic_multisource_ppo(args, target_problem, self_train_only=self_train_only)
    return train_deepic_multisource(args, target_problem, self_train_only=self_train_only)


def run_saea_deepic_problem(args, target_problem: str, deepic, plot: bool = True, initial_archive_x: np.ndarray = None):
    problem = nda.ZDTProblem(name=target_problem, dim=args.dim)
    ref_point = nsga_eic._reference_point(target_problem, args.dim)

    pretrain_entry = load_or_prepare_kan_surrogate(target_problem, args.dim, args)
    pretrain_x = pretrain_entry["x"]
    pretrain_y = pretrain_entry["y"]
    surrogates = pretrain_entry["models"]
    print(f"Prepared KAN surrogate on {target_problem}-{args.dim}D with {pretrain_x.shape[0]} samples.")

    if initial_archive_x is None:
        archive_x = latin_hypercube_sample(
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
    hv_history = []
    reward_history = []

    fronts, _ = demo.fast_non_dominated_sort(archive_y)
    front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    initial_hv = demo.hypervolume_2d(front, ref_point)
    hv_history.append(initial_hv)
    print(
        f"Init    | archive={archive_x.shape[0]} | "
        f"front0={front.shape[0]} | HV={initial_hv:.6f}"
    )

    for step in range(steps_to_run):
        offspring_x, offspring_pred = nsga_eic.generate_nsga2_pseudo_front(
            archive_x=archive_x,
            problem=problem,
            surrogates=surrogates,
            device=args.device,
            n_offspring=args.offspring_size,
            sigma=args.mutation_sigma,
            surrogate_nsga_steps=args.surrogate_nsga_steps,
            predict_fn=demo.predict_with_kan,
            generate_fn=demo.generate_offspring,
        )
        archive_pred = demo.predict_with_kan(surrogates, archive_x, args.device).astype(np.float32)
        offspring_sigma = demo.estimate_uncertainty(
            archive_x=archive_x,
            archive_y=archive_y,
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

        fronts, _ = demo.fast_non_dominated_sort(archive_y)
        front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
        hv_value = demo.hypervolume_2d(front, ref_point)
        hv_history.append(hv_value)

        print(
            f"Iter {step + 1:02d} | archive={archive_x.shape[0]} | "
            f"front0={front.shape[0]} | HV={hv_value:.6f} | reward={reward_value:.6f} | "
            f"pseudo_front={offspring_x.shape[0]}"
        )

        true_evals += args.k_eval
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
            seed=args.seed + 200 + step,
        )

    fronts, _ = demo.fast_non_dominated_sort(archive_y)
    final_front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
    true_front = nsga_eic._true_front(target_problem)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.title(f"{args.dim}D {target_problem} Hypervolume Comparison")
        plt.plot(hv_history, marker="o", label="SAEA-DeepIC")
        plt.xlabel("Step")
        plt.ylabel("Hypervolume")
        plt.grid(True)
        plt.legend()
        plt.show()

        nsga_eic._plot_front(
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
    deepic = load_or_train_deepic(args, target_problem, self_train_only=self_train_only)
    problem = nda.ZDTProblem(name=target_problem, dim=args.dim)
    shared_init_x = latin_hypercube_sample(
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

    eic_args = build_args_namespace(args)
    eic_result = nsga_eic.run_nsga_eic_problem(
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

    nsga_eic._plot_front_comparison(
        f"{args.dim}D {target_problem} Pareto Front Comparison",
        deepic_result["final_front"],
        "SAEA-DeepIC",
        eic_result["final_front"],
        "NSGA-EIC",
        deepic_result["true_front"],
    )


def parse_args(target_problem: str):
    parser = argparse.ArgumentParser(
        description=f"Train SAEA-DeepIC on 15D/20D/25D source problems and evaluate on 30D {target_problem}"
    )
    parser.add_argument("--archive_size", type=int, default=80)
    parser.add_argument("--offspring_size", type=int, default=24)
    parser.add_argument("--k_eval", type=int, default=1)
    parser.add_argument("--max_fe", type=int, default=160)
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
    parser.add_argument("--discount", type=float, default=0.99, help="Reward discount/multiplier used during RL updates")
    parser.add_argument("--train_algo", type=str, default="reinforce", choices=["reinforce", "ppo"])
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--ppo_minibatch_size", type=int, default=32)
    parser.add_argument("--ppo_clip_eps", type=float, default=0.2)
    parser.add_argument("--ppo_value_coef", type=float, default=0.5)
    parser.add_argument("--ppo_entropy_coef", type=float, default=0.01)
    parser.add_argument("--ppo_gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="Start training from this epoch checkpoint number, if it exists.",
    )
    parser.add_argument("--train_only", action="store_true", help="Only train the mixed-source DeepIC model")
    return parser.parse_args()


def main_for_problem(target_problem: str, self_train_only: bool = False):
    args = parse_args(target_problem)
    if args.dim != 30:
        print(f"Warning: expected 30D evaluation for {target_problem}, but received dim={args.dim}.")

    if args.train_only:
        if args.train_algo == "ppo":
            train_deepic_multisource_ppo(args, target_problem, self_train_only=self_train_only)
        else:
            train_deepic_multisource(args, target_problem, self_train_only=self_train_only)
    else:
        run_comparison(args, target_problem, self_train_only=self_train_only)
