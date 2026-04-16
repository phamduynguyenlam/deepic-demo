import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import demo
import multisource_eva_common as multisource


def load_hv_deepic2_class():
    path = Path(__file__).resolve().parent / "agent" / "deepic_agent.py"
    spec = importlib.util.spec_from_file_location("deepic_agent_local_hv2_dtlz2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class HV_Deepic2(module.Deepic2):
        def __init__(
            self,
            hidden_dim: int = 128,
            n_heads: int = 8,
            ff_dim: int = 256,
            dropout: float = 0.0,
            reward_ref_point=None,
            reward_epsilon: float = 1e-8,
        ):
            super().__init__(hidden_dim=hidden_dim, n_heads=n_heads, ff_dim=ff_dim, dropout=dropout)
            self.reward_ref_point = (
                None if reward_ref_point is None else np.asarray(reward_ref_point, dtype=np.float32)
            )
            self.reward_epsilon = float(reward_epsilon)

        def set_reward_reference_point(self, ref_point) -> None:
            self.reward_ref_point = np.asarray(ref_point, dtype=np.float32)

        @staticmethod
        def hypervolume(values: np.ndarray, ref_point: np.ndarray) -> float:
            front = module.DeepIC.pareto_front(values)
            if front.size == 0:
                return 0.0
            return float(module.HV(ref_point=np.asarray(ref_point, dtype=np.float32))(front))

        @classmethod
        def hv_improvement_reward(
            cls,
            previous_archive: np.ndarray,
            selected_objectives: np.ndarray,
            ref_point: np.ndarray,
            epsilon: float = 1e-8,
        ) -> float:
            previous_archive = cls._to_numpy(previous_archive).astype(np.float32)
            selected_objectives = cls._to_numpy(selected_objectives).astype(np.float32)
            combined_archive = np.vstack([previous_archive, selected_objectives])

            prev_hv = cls.hypervolume(previous_archive, ref_point)
            next_hv = cls.hypervolume(combined_archive, ref_point)
            return float((next_hv - prev_hv) / (prev_hv + float(epsilon)))

        def pareto_improvement_reward(
            self,
            previous_archive: np.ndarray,
            selected_objectives: np.ndarray,
            ref_point: np.ndarray | None = None,
            epsilon: float | None = None,
        ) -> float:
            if ref_point is None:
                if self.reward_ref_point is None:
                    raise ValueError("HV_Deepic2 requires a hypervolume reference point for reward computation.")
                ref_point = self.reward_ref_point
            if epsilon is None:
                epsilon = self.reward_epsilon
            return self.hv_improvement_reward(
                previous_archive=previous_archive,
                selected_objectives=selected_objectives,
                ref_point=ref_point,
                epsilon=epsilon,
            )

        @classmethod
        def fpareto_improvement_reward(
            cls,
            previous_archive: np.ndarray,
            selected_objectives: np.ndarray,
            ref_point: np.ndarray,
            epsilon: float = 1e-8,
        ) -> float:
            return cls.hv_improvement_reward(
                previous_archive=previous_archive,
                selected_objectives=selected_objectives,
                ref_point=ref_point,
                epsilon=epsilon,
            )

    return HV_Deepic2


HVDeepIC2Class = load_hv_deepic2_class()

PROBLEM_NAME = "DTLZ2"
MODEL_PATH = "hv_deepic2_dtlz2.pth"
REWARD_LOG_DIR = Path(__file__).resolve().parent / "reward_logs"
REWARD_LOG_PATH = REWARD_LOG_DIR / "hv_deepic2_dtlz2_train_rewards.json"


def _reference_point(dim: int) -> np.ndarray:
    return multisource.nsga_eic._reference_point(PROBLEM_NAME, dim)


def _epoch_checkpoint_path(epoch_number: int) -> Path:
    return Path(__file__).resolve().parent / f"hv_deepic2_dtlz2_epoch_{epoch_number}.pth"


def _torch_load(path: Path | str, map_location: str):
    try:
        return demo.torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return demo.torch.load(path, map_location=map_location)


def _save_reward_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_hv_deepic2_dtlz2(args):
    demo.set_seed(args.seed)
    pretrain_cache = multisource.pretrain_source_surrogates(args, PROBLEM_NAME)
    source_problems = multisource.source_problems_for(PROBLEM_NAME)
    source_dims = multisource.SOURCE_DIMS
    print(
        "Prepared mixed-source KAN surrogates for "
        f"{len(source_problems)} problems across dims {source_dims}."
    )

    deepic = HVDeepIC2Class(
        hidden_dim=args.deepic_hidden,
        n_heads=args.deepic_heads,
        ff_dim=args.deepic_ff,
        reward_ref_point=_reference_point(args.dim),
        reward_epsilon=args.hv_epsilon,
    ).to(args.device)
    deepic_optimizer = demo.torch.optim.Adam(deepic.parameters(), lr=args.deepic_lr)
    replay = demo.ReplayBuffer(capacity=256)
    reward_records: list[dict] = []
    epoch_mean_rewards: list[float] = []

    if args.start_epoch > 0:
        checkpoint_path = _epoch_checkpoint_path(args.start_epoch)
        if checkpoint_path.exists():
            deepic.load_state_dict(_torch_load(checkpoint_path, args.device))
            print(f"Loaded model from {checkpoint_path.name}")
        else:
            print(f"Checkpoint {checkpoint_path.name} not found, starting from scratch")

    for epoch in range(args.start_epoch, 50):
        print(f"HV_DeepIC2 source-mix epoch {epoch + 1}/50")
        epoch_rewards: list[float] = []

        for dim in source_dims:
            for problem_name in source_problems:
                entry = pretrain_cache[(problem_name, dim)]
                problem = entry["problem"]
                pretrain_x = entry["x"]
                pretrain_y = entry["y"]
                surrogates = entry["models"]
                ref_point = multisource.nsga_eic._reference_point(problem_name, dim)

                archive_x = multisource.latin_hypercube_sample(
                    lower=problem.lower,
                    upper=problem.upper,
                    n_samples=args.archive_size,
                    dim=dim,
                    seed=args.seed + epoch * 10000 + multisource._stable_seed(0, problem_name, dim),
                )
                archive_y = problem.evaluate(archive_x)
                true_evals = args.archive_size
                steps_to_run = (args.max_fe - true_evals) // args.k_eval

                for step in range(steps_to_run):
                    offspring_x, offspring_pred = multisource.nsga_eic.generate_nsga2_pseudo_front(
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

                    reward = deepic.pareto_improvement_reward(
                        previous_archive=archive_y,
                        selected_objectives=selected_y,
                        ref_point=ref_point,
                        epsilon=args.hv_epsilon,
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

                    combined_x = np.vstack([pretrain_x, archive_x])
                    combined_y = np.vstack([pretrain_y, archive_y])
                    surrogates = demo.fit_kan_surrogates(
                        archive_x=combined_x,
                        archive_y=combined_y,
                        device=args.device,
                        kan_steps=args.kan_steps,
                        hidden_width=args.kan_hidden,
                        grid=args.kan_grid,
                        seed=args.seed + epoch * 10000 + dim * 100 + step,
                    )

                fronts, _ = demo.fast_non_dominated_sort(archive_y)
                front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
                hv_value = deepic.hypervolume(front, ref_point)
                print(
                    f"{problem_name}-{dim}D epoch {epoch + 1} done, true_evals={true_evals}, "
                    f"front0={front.shape[0]}, hv={hv_value:.6f}, surrogate_nsga_steps={args.surrogate_nsga_steps}"
                )

        epoch_mean = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        epoch_mean_rewards.append(epoch_mean)
        print(f"Epoch {epoch + 1} mean reward: {epoch_mean:.6f}")

        demo.torch.save(deepic.state_dict(), _epoch_checkpoint_path(epoch + 1))
        if (epoch + 1) % 5 == 0:
            multisource.save_colab_model_checkpoint(
                deepic.state_dict(),
                f"hv_deepic2_dtlz2_epoch_{epoch + 1}.pth",
            )
    demo.torch.save(deepic.state_dict(), MODEL_PATH)
    print(f"HV_DeepIC2 model saved to {MODEL_PATH}")
    _save_reward_log(
        REWARD_LOG_PATH,
        {
            "script": "hv_deepic2_dtlz2.py",
            "mode": "train_hv_deepic2_dtlz2",
            "target_problem": PROBLEM_NAME,
            "model_path": MODEL_PATH,
            "source_problems": source_problems,
            "source_dims": source_dims,
            "epoch_mean_rewards": epoch_mean_rewards,
            "records": reward_records,
        },
    )
    print(f"Reward log saved to {REWARD_LOG_PATH}")
    return deepic


def load_or_train_hv_deepic2(args):
    deepic = HVDeepIC2Class(
        hidden_dim=args.deepic_hidden,
        n_heads=args.deepic_heads,
        ff_dim=args.deepic_ff,
        reward_ref_point=_reference_point(args.dim),
        reward_epsilon=args.hv_epsilon,
    ).to(args.device)

    if Path(MODEL_PATH).exists():
        print(f"Using saved HV_DeepIC2 model from {MODEL_PATH}")
        deepic.load_state_dict(_torch_load(MODEL_PATH, args.device))
        return deepic

    return train_hv_deepic2_dtlz2(args)


def run_hv_deepic2_dtlz2(args, deepic, plot: bool = True, initial_archive_x: np.ndarray = None):
    problem = multisource.nda.ZDTProblem(name=PROBLEM_NAME, dim=args.dim)
    ref_point = _reference_point(args.dim)

    pretrain_entry = multisource.load_or_prepare_kan_surrogate(PROBLEM_NAME, args.dim, args)
    pretrain_x = pretrain_entry["x"]
    pretrain_y = pretrain_entry["y"]
    surrogates = pretrain_entry["models"]
    print(f"Prepared KAN surrogate on {PROBLEM_NAME}-{args.dim}D with {pretrain_x.shape[0]} samples.")

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
    true_evals = args.archive_size
    steps_to_run = (args.max_fe - true_evals) // args.k_eval
    hv_history = []
    reward_history = []

    for step in range(steps_to_run):
        offspring_x, offspring_pred = multisource.nsga_eic.generate_nsga2_pseudo_front(
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
        reward_value = deepic.pareto_improvement_reward(
            previous_archive=archive_y,
            selected_objectives=selected_y,
            ref_point=ref_point,
            epsilon=args.hv_epsilon,
        )
        reward_history.append(float(reward_value))

        archive_x, archive_y = demo.update_archive(
            archive_x=archive_x,
            archive_y=archive_y,
            new_x=selected_x,
            new_y=selected_y,
        )

        fronts, _ = demo.fast_non_dominated_sort(archive_y)
        front = archive_y[np.asarray(fronts[0], dtype=np.int64)]
        hv_value = deepic.hypervolume(front, ref_point)
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
    true_front = multisource.nsga_eic._true_front(PROBLEM_NAME)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.title(f"{args.dim}D {PROBLEM_NAME} Hypervolume Progress (HV_DeepIC2)")
        plt.plot(hv_history, marker="o", label="HV_DeepIC2")
        plt.xlabel("Step")
        plt.ylabel("Hypervolume")
        plt.grid(True)
        plt.legend()
        plt.show()

        multisource.nsga_eic._plot_front(
            f"{args.dim}D {PROBLEM_NAME} Pareto Front (HV_DeepIC2)",
            final_front,
            true_front,
            "HV_DeepIC2",
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


def run_comparison(args):
    deepic = load_or_train_hv_deepic2(args)
    problem = multisource.nda.ZDTProblem(name=PROBLEM_NAME, dim=args.dim)
    shared_init_x = multisource.latin_hypercube_sample(
        lower=problem.lower,
        upper=problem.upper,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )

    hv_deepic2_result = run_hv_deepic2_dtlz2(
        args,
        deepic=deepic,
        plot=False,
        initial_archive_x=shared_init_x,
    )

    eic_args = multisource.build_args_namespace(args)
    eic_result = multisource.nsga_eic.run_nsga_eic_problem(
        eic_args,
        problem_name=PROBLEM_NAME,
        plot=False,
        initial_archive_x=shared_init_x,
    )

    print(f"\nHV_DeepIC2 final HV: {hv_deepic2_result['hv_history'][-1]:.6f}")
    print(f"NSGA-EIC final HV: {eic_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {hv_deepic2_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title(f"{args.dim}D {PROBLEM_NAME} Hypervolume Comparison")
    plt.plot(hv_deepic2_result["hv_history"], marker="o", label="HV_DeepIC2")
    plt.plot(eic_result["hv_history"], marker="s", label="NSGA-EIC")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend()
    plt.show()

    multisource.nsga_eic._plot_front_comparison(
        f"{args.dim}D {PROBLEM_NAME} Pareto Front Comparison",
        hv_deepic2_result["final_front"],
        "HV_DeepIC2",
        eic_result["final_front"],
        "NSGA-EIC",
        hv_deepic2_result["true_front"],
    )

    return {"hv_deepic2": hv_deepic2_result, "nsga_eic": eic_result}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HV_DeepIC2 on 15D/20D/25D mixed-source problems and evaluate on DTLZ2."
    )
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
    parser.add_argument("--hv_epsilon", type=float, default=1e-8)
    parser.add_argument("--reward", type=float, default=0.99, help="Reward discount/multiplier used during RL updates")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="Start training from this epoch checkpoint number, if it exists.",
    )
    parser.add_argument("--train_only", action="store_true", help="Only train the mixed-source HV_DeepIC2 model.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.train_only:
        train_hv_deepic2_dtlz2(args)
    else:
        run_comparison(args)


if __name__ == "__main__":
    main()
