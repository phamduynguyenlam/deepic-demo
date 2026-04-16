import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

import demo
import multisource_eva_common as multisource
from problem.kan import KAN


def load_module(filename: str, module_name: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nda = load_module("nsga-nda.py", "nsga_nda_module")
nsga_eic = load_module("nsga-eic.py", "nsga_eic_module")


SOURCE_PROBLEMS = ["ZDT2", "ZDT3", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]
SOURCE_DIMS = [15, 20, 25]
MODEL_PATH = "deepic_zdt_source_mix.pth"
REWARD_LOG_DIR = Path(__file__).resolve().parent / "reward_logs"
ZDT1_EVA_REWARD_LOG_PATH = REWARD_LOG_DIR / "zdt1_eva_train_rewards.json"


def _save_reward_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def epoch_model_path(epoch: int, suffix: str = ".pth") -> Path:
    return Path(__file__).resolve().parent / f"zdt1_model_epoch_{epoch}{suffix}"


def resolve_epoch_model_path(epoch: int) -> Path:
    preferred = epoch_model_path(epoch, ".pth")
    legacy = epoch_model_path(epoch, ".py")
    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy
    return preferred


def deserialize_kan_model(checkpoint: dict, device: str):
    """
    Deserialize a KAN model from checkpoint.
    Handles multiple formats:
    - Direct KAN object with __call__ method
    - Serialized dict with state_dict and config
    - Older format: dict with direct model objects (return as-is after to(device))
    """
    # If it's already a callable KAN object, ensure it's on the right device
    if hasattr(checkpoint, "__call__") and hasattr(checkpoint, "forward"):
        return checkpoint.to(device) if hasattr(checkpoint, "to") else checkpoint
    
    # If it's a dict with serialized format, reconstruct the model
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and "config" in checkpoint:
            config = checkpoint["config"].copy()
            config["device"] = device
            model = KAN(**config).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            return model
        elif "width" in checkpoint or "grid" in checkpoint:
            # Handle partially serialized dict format
            config = checkpoint.copy()
            config["device"] = device
            model = KAN(**config).to(device)
            return model
    
    # Last resort: assume it's directly a KAN model instance stored in older format
    if hasattr(checkpoint, "forward"):
        return checkpoint.to(device) if hasattr(checkpoint, "to") else checkpoint
    
    raise ValueError(f"Unsupported KAN model checkpoint format: {type(checkpoint)}, keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'N/A'}")


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
        reward=parsed.reward,
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


def generate_hybrid_offspring(archive_x, problem, n_offspring: int, base_sigma: float) -> np.ndarray:
    counts = np.full(3, n_offspring // 3, dtype=int)
    counts[: n_offspring % 3] += 1

    strategy_sigmas = {
        "NSGA-III": base_sigma,
        "CDM-PSL": max(0.2, 1.5 * base_sigma),
        "qNEHVI": min(max(0.05, 0.5 * base_sigma), base_sigma),
    }
    strategy_order = ["NSGA-III", "CDM-PSL", "qNEHVI"]

    offspring_batches = []
    for count, strategy_name in zip(counts, strategy_order):
        if count <= 0:
            continue
        offspring_batches.append(
            demo.generate_offspring(
                archive_x=archive_x,
                n_offspring=int(count),
                lower=problem.lower,
                upper=problem.upper,
                sigma=strategy_sigmas[strategy_name],
            )
        )

    if not offspring_batches:
        return archive_x[:0].copy()
    return np.vstack(offspring_batches).astype(np.float32)


def pretrain_source_surrogates(args):
    cache = {}
    for dim in SOURCE_DIMS:
        for problem_name in SOURCE_PROBLEMS:
            path = Path(__file__).resolve().parent / f"kan_{problem_name.lower()}_{dim}d.pth"
            if path.exists():
                print(f"Loading pre-trained KAN surrogate from {path}")
                checkpoint = demo.torch.load(path, map_location=args.device, weights_only=False)
                cache[(problem_name, dim)] = {
                    "problem": nda.ZDTProblem(name=problem_name, dim=dim),
                    "x": checkpoint['x_data'],
                    "y": checkpoint['y_data'],
                    "models": [deserialize_kan_model(model_checkpoint, args.device) for model_checkpoint in checkpoint['models']],
                }
            else:
                print(f"Pre-training KAN surrogate on {problem_name}-{dim}D...")
                problem = nda.ZDTProblem(name=problem_name, dim=dim)
                x_data, y_data, models = nda.pre_train_kan_surrogate_for_problem(
                    problem=problem,
                    device=args.device,
                    kan_steps=args.kan_steps,
                    hidden_width=args.kan_hidden,
                    grid=args.kan_grid,
                    seed=args.seed + dim,
                )
                cache[(problem_name, dim)] = {
                    "problem": problem,
                    "x": x_data,
                    "y": y_data,
                    "models": models,
                }
    return cache


def train_deepic_multisource(args):
    demo.set_seed(args.seed)
    pretrain_cache = pretrain_source_surrogates(args)
    replay = demo.ReplayBuffer(capacity=256)
    reward_records: list[dict] = []
    epoch_mean_rewards: list[float] = []

    deepic = demo.DeepICClass(
        hidden_dim=args.deepic_hidden,
        n_heads=args.deepic_heads,
        ff_dim=args.deepic_ff,
    ).to(args.device)
    deepic_optimizer = demo.torch.optim.Adam(deepic.parameters(), lr=args.deepic_lr)

    if args.start_epoch > 0:
        model_path = resolve_epoch_model_path(args.start_epoch)
        if Path(model_path).exists():
            deepic.load_state_dict(demo.torch.load(model_path, map_location=args.device))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model {model_path} not found, starting from scratch")

    for epoch in range(args.start_epoch, 50):
        print(f"Epoch {epoch + 1}/50")
        epoch_rewards: list[float] = []
        for dim in SOURCE_DIMS:
            for problem_name in SOURCE_PROBLEMS:
                entry = pretrain_cache[(problem_name, dim)]
                problem = entry["problem"]
                surrogates = entry["models"]

                archive_x = latin_hypercube_sample(
                    lower=problem.lower,
                    upper=problem.upper,
                    n_samples=args.archive_size,
                    dim=dim,
                    seed=args.seed + epoch * 1000 + dim * 100 + sum(ord(ch) for ch in problem_name),
                )
                archive_y = problem.evaluate(archive_x)
                true_evals = args.archive_size
                remaining_budget = args.max_fe - true_evals
                steps_to_run = remaining_budget // args.k_eval

                for step in range(steps_to_run):
                    offspring_x = demo.generate_offspring(
                        archive_x=archive_x,
                        n_offspring=args.offspring_size,
                        lower=problem.lower,
                        upper=problem.upper,
                        sigma=args.mutation_sigma,
                    )
                    offspring_pred = demo.predict_with_kan(surrogates, offspring_x, args.device).astype(np.float32)
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
                    )

                    selected_idx = ranking[: args.k_eval]
                    selected_x = offspring_x[selected_idx]
                    selected_y = problem.evaluate(selected_x)

                    reward = demo.DeepICClass.fpareto_improvement_reward(
                        previous_front=archive_y,
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
                    archive_x, archive_y = demo.update_archive(
                        archive_x=archive_x,
                        archive_y=archive_y,
                        new_x=selected_x,
                        new_y=selected_y,
                    )

                    replay.add(
                        {
                            "archive_x": archive_x,
                            "archive_y": archive_y,
                            "offspring_x": offspring_x,
                            "offspring_pred": offspring_pred,
                            "offspring_sigma": offspring_sigma,
                            "ranking": ranking,
                            "reward": reward,
                            "progress": progress,
                            "lower": problem.lower,
                            "upper": problem.upper,
                        }
                    )

                    if len(replay) >= 32:
                        for sample in replay.sample(32):
                            demo.adapt_deepic(
                                model=deepic,
                                optimizer=deepic_optimizer,
                                archive_x=sample["archive_x"],
                                archive_y=sample["archive_y"],
                                offspring_x=sample["offspring_x"],
                                offspring_pred=sample["offspring_pred"],
                                offspring_sigma=sample["offspring_sigma"],
                                lower=sample["lower"],
                                upper=sample["upper"],
                                progress=sample["progress"],
                                target_ranking=sample["ranking"],
                                reward=sample["reward"],
                                device=args.device,
                                steps=1,
                                top_k=args.k_eval,
                                reward_discount=args.reward,
                            )

                    true_evals += args.k_eval
                    if true_evals >= args.max_fe:
                        break

                print(
                    f"{problem_name}-{dim}D epoch {epoch + 1} done, "
                    f"true_evals={true_evals}, best_obj1={np.min(archive_y[:, 0]):.6f}"
                )
        epoch_mean = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        epoch_mean_rewards.append(epoch_mean)
        print(f"Epoch {epoch + 1} mean reward: {epoch_mean:.6f}")
        demo.torch.save(deepic.state_dict(), epoch_model_path(epoch + 1))
        if (epoch + 1) % 5 == 0:
            multisource.save_colab_model_checkpoint(
                deepic.state_dict(),
                f"zdt1_model_epoch_{epoch + 1}.pth",
            )

    demo.torch.save(deepic.state_dict(), MODEL_PATH)
    print(f"DeepIC model saved to {MODEL_PATH}")
    _save_reward_log(
        ZDT1_EVA_REWARD_LOG_PATH,
        {
            "script": "zdt1_eva.py",
            "mode": "train_deepic_multisource",
            "model_path": MODEL_PATH,
            "source_problems": SOURCE_PROBLEMS,
            "source_dims": SOURCE_DIMS,
            "epoch_mean_rewards": epoch_mean_rewards,
            "records": reward_records,
        },
    )
    print(f"Reward log saved to {ZDT1_EVA_REWARD_LOG_PATH}")
    return deepic


def load_or_train_deepic(args):
    if getattr(args, "eval_epoch", None) is not None:
        checkpoint_path = resolve_epoch_model_path(args.eval_epoch)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Requested epoch checkpoint not found: "
                f"{epoch_model_path(args.eval_epoch, '.pth')} or {epoch_model_path(args.eval_epoch, '.py')}"
            )
        print(f"Using saved DeepIC epoch checkpoint from {checkpoint_path}")
        deepic = demo.DeepICClass(
            hidden_dim=args.deepic_hidden,
            n_heads=args.deepic_heads,
            ff_dim=args.deepic_ff,
        ).to(args.device)
        deepic.load_state_dict(demo.torch.load(checkpoint_path, map_location=args.device))
        return deepic

    if Path(MODEL_PATH).exists():
        print(f"Using saved DeepIC model from {MODEL_PATH}")
        deepic = demo.DeepICClass(
            hidden_dim=args.deepic_hidden,
            n_heads=args.deepic_heads,
            ff_dim=args.deepic_ff,
        ).to(args.device)
        deepic.load_state_dict(demo.torch.load(MODEL_PATH, map_location=args.device))
        return deepic
    return train_deepic_multisource(args)


def run_saea_deepic_zdt1(args, deepic, plot: bool = True, initial_archive_x: np.ndarray = None):
    problem = nda.ZDTProblem(name="ZDT1", dim=args.dim)
    ref_point = np.array([0.9994, 6.0576], dtype=np.float32)

    path = Path(__file__).resolve().parent / f"kan_zdt1_{args.dim}d.pth"
    if path.exists():
        print(f"Loading pre-trained KAN surrogate for ZDT1-{args.dim}D from {path}")
        checkpoint = demo.torch.load(path, map_location=args.device, weights_only=False)
        pretrain_x = checkpoint['x_data']
        pretrain_y = checkpoint['y_data']
        surrogates = [deserialize_kan_model(model_checkpoint, args.device) for model_checkpoint in checkpoint['models']]
    else:
        print(f"Pre-training KAN surrogate for ZDT1-{args.dim}D...")
        pretrain_x, pretrain_y, surrogates = nda.pre_train_kan_surrogate_for_problem(
            problem=problem,
            device=args.device,
            kan_steps=args.kan_steps,
            hidden_width=args.kan_hidden,
            grid=args.kan_grid,
            seed=args.seed,
        )
    print(f"Pre-trained KAN surrogate on ZDT1-{args.dim}D with {pretrain_x.shape[0]} samples.")

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
        )

        selected_idx = ranking[: args.k_eval]
        selected_x = offspring_x[selected_idx]
        selected_y = problem.evaluate(selected_x)

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
            f"front0={front.shape[0]} | HV={hv_value:.6f} | pseudo_front={offspring_x.shape[0]}"
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
    true_f1 = np.linspace(0.0, 1.0, 200, dtype=np.float32)
    true_f2 = 1.0 - np.sqrt(true_f1)
    true_front = np.stack([true_f1, true_f2], axis=1)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.title(f"{args.dim}D ZDT1 Hypervolume Comparison")
        plt.plot(hv_history, marker="o", label="SAEA-DeepIC")
        plt.xlabel("Step")
        plt.ylabel("Hypervolume")
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.title(f"{args.dim}D ZDT1 Pareto Front")
        plt.scatter(final_front[:, 0], final_front[:, 1], s=24, alpha=0.8, label="SAEA-DeepIC")
        plt.plot(true_front[:, 0], true_front[:, 1], "k-", linewidth=2, label="True Pareto Front")
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.grid(True)
        plt.legend()
        plt.show()

    return {
        "archive_x": archive_x,
        "archive_y": archive_y,
        "final_front": final_front,
        "true_front": true_front,
        "hv_history": hv_history,
        "ref_point": ref_point,
    }


def run_comparison(args):
    deepic = load_or_train_deepic(args)
    shared_init_x = latin_hypercube_sample(
        lower=0.0,
        upper=1.0,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )
    deepic_result = run_saea_deepic_zdt1(
        args,
        deepic=deepic,
        plot=False,
        initial_archive_x=shared_init_x,
    )

    eic_args = build_args_namespace(args)
    eic_args.dim = args.dim
    eic_result = nsga_eic.run_nsga_eic_problem(
        eic_args,
        problem_name="ZDT1",
        plot=False,
        initial_archive_x=shared_init_x,
    )

    print(f"\nSAEA-DeepIC final HV: {deepic_result['hv_history'][-1]:.6f}")
    print(f"NSGA-EIC final HV: {eic_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {deepic_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title(f"{args.dim}D ZDT1 Hypervolume Comparison")
    plt.plot(deepic_result["hv_history"], marker="o", label="SAEA-DeepIC")
    plt.plot(eic_result["hv_history"], marker="s", label="NSGA-EIC")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.title(f"{args.dim}D ZDT1 Pareto Front Comparison")
    plt.scatter(deepic_result["final_front"][:, 0], deepic_result["final_front"][:, 1], s=24, alpha=0.8, label="SAEA-DeepIC")
    plt.scatter(eic_result["final_front"][:, 0], eic_result["final_front"][:, 1], s=24, alpha=0.8, label="NSGA-EIC")
    plt.plot(deepic_result["true_front"][:, 0], deepic_result["true_front"][:, 1], "k-", linewidth=2, label="True Pareto Front")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(True)
    plt.legend()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepIC on mixed source problems and evaluate on 30D ZDT1")
    parser.add_argument("--archive_size", type=int, default=80)
    parser.add_argument("--offspring_size", type=int, default=24)
    parser.add_argument("--k_eval", type=int, default=5)
    parser.add_argument("--max_fe", type=int, default=160)
    parser.add_argument("--mutation_sigma", type=float, default=0.12)
    parser.add_argument("--kan_steps", type=int, default=25)
    parser.add_argument("--kan_hidden", type=int, default=10)
    parser.add_argument("--kan_grid", type=int, default=5)
    parser.add_argument("--deepic_hidden", type=int, default=64)
    parser.add_argument("--deepic_heads", type=int, default=4)
    parser.add_argument("--deepic_ff", type=int, default=128)
    parser.add_argument("--deepic_lr", type=float, default=1e-3)
    parser.add_argument("--deepic_adapt_steps", type=int, default=8)
    parser.add_argument("--surrogate_nsga_steps", type=int, default=40)
    parser.add_argument("--reward", type=float, default=0.99, help="Reward discount/multiplier used during RL updates")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--start_epoch", type=int, default=0, help="Start training from this epoch (0-based, load model if exists)")
    parser.add_argument("--train_only", action="store_true", help="Only train the mixed-source DeepIC model")
    parser.add_argument("--eval_epoch", type=int, default=None, help="Load zdt1_model_epoch_<k>.pth for evaluation/comparison")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.train_only:
        train_deepic_multisource(args)
    else:
        run_comparison(args)


if __name__ == "__main__":
    main()
