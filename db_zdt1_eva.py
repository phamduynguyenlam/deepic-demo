import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

import demo
import multisource_eva_common as multisource


def load_module(filename: str, module_name: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nda = load_module("nsga-nda.py", "nsga_nda_module")
nsga_eic = load_module("nsga-eic.py", "nsga_eic_module")
db_saea_agent = load_module("db-saea-agent.py", "db_saea_agent_module")
niching_nsga = load_module("niching_nsga.py", "niching_nsga_module")

DBSAEAMetaPolicy = db_saea_agent.DBSAEAMetaPolicy
GlobalReplayBuffer = db_saea_agent.GlobalReplayBuffer
DistributedLearner = db_saea_agent.DistributedLearner
ray = db_saea_agent.ray
build_global_replay_buffer_actor = db_saea_agent.build_global_replay_buffer_actor
build_db_saea_rollout_worker = db_saea_agent.build_db_saea_rollout_worker


SOURCE_PROBLEMS = ["ZDT2", "ZDT3", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]
SOURCE_DIMS = [15, 20, 25]
ACTION_NAMES = [
    "surrogate_nsga",
    "nd_a",
    "nd_pbi_conv",
    "nd_pbi_div",
    "epdi_exploit",
    "epdi_explore",
]
MODEL_PATH = "db_saea_zdt_source_mix.pth"
REWARD_LOG_DIR = Path(__file__).resolve().parent / "reward_logs"
TRAIN_LOG_PATH = REWARD_LOG_DIR / "db_zdt1_eva_train_rewards.json"
INFER_RESULT_PATH = Path(__file__).resolve().parent / "db_zdt1_infer_results.json"
INFER_NOTEBOOK_PATH = Path(__file__).resolve().parent / "db_zdt1_infer_plots.ipynb"


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def epoch_model_path(epoch: int) -> Path:
    return Path(__file__).resolve().parent / f"db_zdt1_model_epoch_{epoch}.pth"


def _policy_kwargs(args) -> dict:
    return {
        "ela_hidden_dim": args.ela_hidden,
        "n_heads": args.ela_heads,
        "ff_dim": args.ela_ff,
        "dqn_hidden_dim": args.dqn_hidden,
        "num_actions": len(ACTION_NAMES),
        "dropout": args.dropout,
    }


def _build_args_namespace(parsed) -> SimpleNamespace:
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
        surrogate_nsga_steps=parsed.surrogate_nsga_steps,
        seed=parsed.seed,
        device=parsed.device,
    )


def _task_name(problem_name: str, dim: int) -> str:
    return f"{problem_name}|{int(dim)}"


def _parse_task_name(task_name: str) -> tuple[str, int]:
    problem_name, dim = task_name.split("|", maxsplit=1)
    return problem_name, int(dim)


def _all_task_names() -> list[str]:
    return [_task_name(problem_name, dim) for dim in SOURCE_DIMS for problem_name in SOURCE_PROBLEMS]


def _ensure_worker_surrogates_ready(args) -> None:
    loader_args = _build_args_namespace(args)
    task_names = _all_task_names()
    total_tasks = len(task_names)
    print(f"Preparing KAN surrogates for {total_tasks} worker tasks before starting Ray workers...")
    for task_idx, task_name in enumerate(task_names, start=1):
        problem_name, dim = _parse_task_name(task_name)
        checkpoint_path = multisource._kan_checkpoint_path(problem_name, dim)
        status = "loading" if checkpoint_path.exists() else "pre-training"
        print(f"[{task_idx:02d}/{total_tasks:02d}] {status} surrogate for {problem_name}-{dim}D")
        multisource.load_or_prepare_kan_surrogate(problem_name, dim, loader_args)
    print("All worker surrogates are ready.")


def _archive_hv(values: np.ndarray, ref_point: np.ndarray) -> float:
    fronts, _ = demo.fast_non_dominated_sort(values)
    if not fronts or not fronts[0]:
        return 0.0
    front = values[np.asarray(fronts[0], dtype=np.int64)]
    return float(demo.hypervolume_2d(front, ref_point))


def _db_saea_reward(
    previous_archive: np.ndarray,
    selected_y: np.ndarray,
    ref_point: np.ndarray,
    reward_lambda: float,
    epsilon: float,
) -> float:
    previous_archive = np.asarray(previous_archive, dtype=np.float32)
    selected_y = np.asarray(selected_y, dtype=np.float32)
    prev_hv = _archive_hv(previous_archive, ref_point)
    next_hv = _archive_hv(np.vstack([previous_archive, selected_y]), ref_point)
    if next_hv <= prev_hv + float(epsilon):
        return -1.0

    fronts, _ = demo.fast_non_dominated_sort(previous_archive)
    pareto_front = (
        previous_archive[np.asarray(fronts[0], dtype=np.int64)]
        if fronts and fronts[0]
        else previous_archive
    )
    origin = np.zeros(pareto_front.shape[1], dtype=np.float32)
    ratios: list[float] = []
    for candidate in selected_y:
        distances = np.abs(pareto_front - candidate).sum(axis=1)
        nearest_idx = int(np.argmin(distances))
        d = float(distances[nearest_idx])
        d_ref = float(np.abs(pareto_front[nearest_idx] - origin).sum())
        ratios.append(d / max(d_ref, 1e-12))
    mean_ratio = float(np.mean(ratios)) if ratios else 0.0
    return float(1.0 + float(reward_lambda) * mean_ratio)


def _build_state(
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    offspring_x: np.ndarray,
    offspring_pred: np.ndarray,
    offspring_sigma: np.ndarray,
    problem,
    progress: float,
) -> dict:
    return {
        "x_true": np.asarray(archive_x, dtype=np.float32),
        "y_true": np.asarray(archive_y, dtype=np.float32),
        "x_sur": np.asarray(offspring_x, dtype=np.float32),
        "y_sur": np.asarray(offspring_pred, dtype=np.float32),
        "sigma_sur": np.asarray(offspring_sigma, dtype=np.float32),
        "lower_bound": np.asarray(problem.lower, dtype=np.float32),
        "upper_bound": np.asarray(problem.upper, dtype=np.float32),
        "progress": float(progress),
    }


def _policy_inputs_from_state(state: dict, device: str) -> dict:
    policy_state = {}
    for key in ["x_true", "y_true", "x_sur", "y_sur", "sigma_sur", "lower_bound", "upper_bound"]:
        policy_state[key] = torch.as_tensor(state[key], dtype=torch.float32, device=device)
    policy_state["progress"] = torch.as_tensor(state["progress"], dtype=torch.float32, device=device)
    return policy_state


def _select_action_with_a1_cap(policy, policy_state: dict, epsilon: float, a1_count: int, max_a1_actions: int) -> int:
    if int(a1_count) < int(max_a1_actions):
        return int(
            policy.select_action(
                **policy_state,
                epsilon=float(epsilon),
            )
        )

    with torch.no_grad():
        q_values = policy.forward(**policy_state)[0].detach().cpu().numpy()

    non_a1_actions = np.arange(1, len(ACTION_NAMES), dtype=np.int64)
    if np.random.rand() < float(epsilon):
        return int(np.random.choice(non_a1_actions))
    best_local = int(np.argmax(q_values[1:]))
    return int(non_a1_actions[best_local])


def _generate_surrogate_population(problem, surrogates, archive_x: np.ndarray, archive_y: np.ndarray, args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return offspring_x.astype(np.float32), offspring_pred.astype(np.float32), offspring_sigma


def _nda_seed_ranking(offspring_pred: np.ndarray, offspring_sigma: np.ndarray) -> np.ndarray:
    top = nda.select_nda(offspring_pred, offspring_sigma, k=1)
    lcb = np.argsort((offspring_pred + offspring_sigma).sum(axis=1)).astype(np.int64)
    if top.size == 0:
        return lcb
    rest = lcb[lcb != int(top[0])]
    return np.concatenate([top.astype(np.int64), rest.astype(np.int64)])


def _objective_diversity_ranking(archive_y: np.ndarray, offspring_pred: np.ndarray) -> np.ndarray:
    if archive_y.size == 0:
        return np.arange(offspring_pred.shape[0], dtype=np.int64)
    dist = np.linalg.norm(offspring_pred[:, None, :] - archive_y[None, :, :], axis=2)
    min_dist = dist.min(axis=1)
    return np.argsort(-min_dist).astype(np.int64)


def _merge_primary_with_fallback(primary: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    primary = np.asarray(primary, dtype=np.int64).reshape(-1)
    fallback = np.asarray(fallback, dtype=np.int64).reshape(-1)
    if primary.size == 0:
        return fallback
    mask = ~np.isin(fallback, primary)
    return np.concatenate([primary, fallback[mask]])


def _nd_pbi_ranking(archive_y: np.ndarray, offspring_pred: np.ndarray, offspring_sigma: np.ndarray, focus: str) -> np.ndarray:
    penalized = offspring_pred + offspring_sigma
    top = niching_nsga._nd_pbi_select(penalized, archive_y, k=1, focus=focus)
    fallback = np.argsort(penalized.sum(axis=1)).astype(np.int64)
    return _merge_primary_with_fallback(top, fallback)


def _random_unit_reference_vector(n_obj: int, rng: np.random.Generator) -> np.ndarray:
    vec = rng.random(n_obj, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return np.full(n_obj, 1.0 / np.sqrt(max(n_obj, 1)), dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _pd_value(values: np.ndarray, ref_vector: np.ndarray, theta: float = 5.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    ref_vector = np.asarray(ref_vector, dtype=np.float32)
    ref_norm = np.linalg.norm(ref_vector)
    if ref_norm <= 1e-12:
        ref_vector = np.full(ref_vector.shape[0], 1.0 / np.sqrt(max(ref_vector.shape[0], 1)), dtype=np.float32)
        ref_norm = np.linalg.norm(ref_vector)

    d1 = (values @ ref_vector) / max(ref_norm, 1e-12)
    projection = (d1 / max(ref_norm, 1e-12))[:, None] * ref_vector[None, :]
    d2 = np.linalg.norm(values - projection, axis=1)
    return d1 + float(theta) * d2


def _epdi_statistics(
    archive_y: np.ndarray,
    offspring_pred: np.ndarray,
    offspring_sigma: np.ndarray,
    mc_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    archive_fronts, _ = demo.fast_non_dominated_sort(archive_y)
    archive_front = archive_y[np.asarray(archive_fronts[0], dtype=np.int64)] if archive_fronts and archive_fronts[0] else archive_y
    n_obj = offspring_pred.shape[1]
    mean_epdi = np.zeros(offspring_pred.shape[0], dtype=np.float32)
    std_epdi = np.zeros(offspring_pred.shape[0], dtype=np.float32)

    for idx in range(offspring_pred.shape[0]):
        ref_vector = _random_unit_reference_vector(n_obj, rng)
        pd_min = float(np.min(_pd_value(archive_front, ref_vector))) if archive_front.size else 0.0
        sigma = np.maximum(offspring_sigma[idx], 1e-6)
        samples = rng.normal(
            loc=offspring_pred[idx],
            scale=sigma,
            size=(mc_samples, n_obj),
        ).astype(np.float32)
        pdi_samples = np.maximum(pd_min - _pd_value(samples, ref_vector), 0.0)
        mean_epdi[idx] = float(np.mean(pdi_samples))
        std_epdi[idx] = float(np.std(pdi_samples))

    return mean_epdi, std_epdi


def _epdi_ranking(
    archive_y: np.ndarray,
    offspring_pred: np.ndarray,
    offspring_sigma: np.ndarray,
    mode: str,
    seed: int,
    mc_samples: int = 1000,
) -> np.ndarray:
    epdi_mean, epdi_std = _epdi_statistics(
        archive_y=archive_y,
        offspring_pred=offspring_pred,
        offspring_sigma=offspring_sigma,
        mc_samples=mc_samples,
        seed=seed,
    )
    if mode == "exploit":
        score = epdi_mean
    elif mode == "explore":
        score = epdi_mean + epdi_std
    else:
        raise ValueError(f"Unsupported EPDI mode: {mode}")
    return np.argsort(-score).astype(np.int64)


def _ranking_for_action(
    action: int,
    archive_y: np.ndarray,
    offspring_pred: np.ndarray,
    offspring_sigma: np.ndarray,
    seed: int,
) -> np.ndarray:
    action = int(action)
    if action == 0:
        raise ValueError("Action 0 is regenerate-only and should not request a ranking.")
    if action == 1:
        return _nda_seed_ranking(offspring_pred, offspring_sigma)
    if action == 2:
        return _nd_pbi_ranking(archive_y, offspring_pred, offspring_sigma, focus="convergence")
    if action == 3:
        return _nd_pbi_ranking(archive_y, offspring_pred, offspring_sigma, focus="diversity")
    if action == 4:
        return _epdi_ranking(archive_y, offspring_pred, offspring_sigma, mode="exploit", seed=seed)
    if action == 5:
        return _epdi_ranking(archive_y, offspring_pred, offspring_sigma, mode="explore", seed=seed)
    raise ValueError(f"Unsupported action id: {action}")


def _next_state_after_transition(problem, surrogates, archive_x: np.ndarray, archive_y: np.ndarray, true_evals: int, args):
    progress = float(true_evals / args.max_fe)
    offspring_x, offspring_pred, offspring_sigma = _generate_surrogate_population(problem, surrogates, archive_x, archive_y, args)
    state = _build_state(
        archive_x=archive_x,
        archive_y=archive_y,
        offspring_x=offspring_x,
        offspring_pred=offspring_pred,
        offspring_sigma=offspring_sigma,
        problem=problem,
        progress=progress,
    )
    return state, offspring_x, offspring_pred, offspring_sigma


class DBSAEATaskEnv:
    def __init__(self, task_name: str, worker_id: int, args):
        self.task_name = task_name
        self.worker_id = int(worker_id)
        self.args = args
        self.problem_name, self.dim = _parse_task_name(task_name)
        entry = multisource.load_or_prepare_kan_surrogate(self.problem_name, self.dim, _build_args_namespace(args))
        self.problem = entry["problem"]
        self.base_surrogates = entry["models"]
        self.pretrain_x = np.asarray(entry["x"], dtype=np.float32)
        self.pretrain_y = np.asarray(entry["y"], dtype=np.float32)
        self.ref_point = nsga_eic._reference_point(self.problem_name, self.dim)
        self.episode_index = 0
        self.archive_x = None
        self.archive_y = None
        self.surrogates = None
        self.true_evals = 0
        self.step_count = 0
        self.a1_count = 0
        self.state = None
        self.offspring_x = None
        self.offspring_pred = None
        self.offspring_sigma = None

    def _initial_seed(self) -> int:
        return (
            int(self.args.seed)
            + self.episode_index * 10000
            + self.worker_id * 1000
            + self.dim * 100
            + sum(ord(ch) for ch in self.problem_name)
        )

    def reset(self) -> dict:
        self.episode_index += 1
        self.archive_x = multisource.latin_hypercube_sample(
            lower=self.problem.lower,
            upper=self.problem.upper,
            n_samples=self.args.archive_size,
            dim=self.dim,
            seed=self._initial_seed(),
        )
        self.archive_y = self.problem.evaluate(self.archive_x)
        self.surrogates = self.base_surrogates
        self.true_evals = int(self.args.archive_size)
        self.step_count = 0
        self.a1_count = 0
        self.state, self.offspring_x, self.offspring_pred, self.offspring_sigma = _next_state_after_transition(
            problem=self.problem,
            surrogates=self.surrogates,
            archive_x=self.archive_x,
            archive_y=self.archive_y,
            true_evals=self.true_evals,
            args=self.args,
        )
        return self.state

    def select_action(self, policy, state: dict, epsilon: float) -> int:
        policy_state = _policy_inputs_from_state(state, device="cpu")
        return _select_action_with_a1_cap(
            policy=policy,
            policy_state=policy_state,
            epsilon=float(epsilon),
            a1_count=self.a1_count,
            max_a1_actions=self.args.max_a1_actions,
        )

    def step(self, action: int):
        self.step_count += 1
        action = int(action)
        if action == 0:
            self.a1_count += 1
            reward_value = 0.0
            archive_x_next = self.archive_x
            archive_y_next = self.archive_y
            true_evals_next = self.true_evals
            done = False
            surrogates_next = self.surrogates
            next_state, next_offspring_x, next_offspring_pred, next_offspring_sigma = _next_state_after_transition(
                problem=self.problem,
                surrogates=surrogates_next,
                archive_x=archive_x_next,
                archive_y=archive_y_next,
                true_evals=true_evals_next,
                args=self.args,
            )
        else:
            ranking = _ranking_for_action(
                action=action,
                archive_y=self.archive_y,
                offspring_pred=self.offspring_pred,
                offspring_sigma=self.offspring_sigma,
                seed=int(self.args.seed) + self.episode_index * 100000 + self.dim * 1000 + self.step_count,
            )
            selected_idx = ranking[: self.args.k_eval]
            selected_x = self.offspring_x[selected_idx]
            selected_y = self.problem.evaluate(selected_x)
            reward_value = _db_saea_reward(
                previous_archive=self.archive_y,
                selected_y=selected_y,
                ref_point=self.ref_point,
                reward_lambda=self.args.reward_lambda,
                epsilon=self.args.hv_epsilon,
            )
            archive_x_next, archive_y_next = demo.update_archive(
                archive_x=self.archive_x,
                archive_y=self.archive_y,
                new_x=selected_x,
                new_y=selected_y,
            )
            true_evals_next = self.true_evals + self.args.k_eval
            done = bool(true_evals_next >= self.args.max_fe)
            if not done:
                combined_x = np.vstack([self.pretrain_x, archive_x_next])
                combined_y = np.vstack([self.pretrain_y, archive_y_next])
                surrogates_next = demo.fit_kan_surrogates(
                    archive_x=combined_x,
                    archive_y=combined_y,
                    device=self.args.device,
                    kan_steps=self.args.kan_steps,
                    hidden_width=self.args.kan_hidden,
                    grid=self.args.kan_grid,
                    seed=int(self.args.seed) + self.episode_index * 10000 + self.step_count * 100 + self.dim,
                )
                next_state, next_offspring_x, next_offspring_pred, next_offspring_sigma = _next_state_after_transition(
                    problem=self.problem,
                    surrogates=surrogates_next,
                    archive_x=archive_x_next,
                    archive_y=archive_y_next,
                    true_evals=true_evals_next,
                    args=self.args,
                )
            else:
                surrogates_next = self.surrogates
                next_state = self.state
                next_offspring_x = self.offspring_x
                next_offspring_pred = self.offspring_pred
                next_offspring_sigma = self.offspring_sigma

        self.archive_x = archive_x_next
        self.archive_y = archive_y_next
        self.surrogates = surrogates_next
        self.true_evals = true_evals_next
        self.state = next_state
        self.offspring_x = next_offspring_x
        self.offspring_pred = next_offspring_pred
        self.offspring_sigma = next_offspring_sigma
        info = {
            "executed_action": action,
            "problem_name": self.problem_name,
            "dim": int(self.dim),
            "true_evals": int(self.true_evals),
            "a1_count": int(self.a1_count),
            "best_obj1": float(np.min(self.archive_y[:, 0])),
        }
        return next_state, float(reward_value), bool(done), info

    def episode_stats(self) -> dict:
        return {
            "problem_name": self.problem_name,
            "dim": int(self.dim),
            "true_evals": int(self.true_evals),
            "a1_count": int(self.a1_count),
            "best_obj1": float(np.min(self.archive_y[:, 0])),
            "step_count": int(self.step_count),
        }


def _write_infer_notebook(notebook_path: Path, result_json_path: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# DB-SAEA ZDT1 Inference Plots\n",
                    "\n",
                    "Notebook này đọc file kết quả JSON và vẽ lại đồ thị HV cùng Pareto front.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "from pathlib import Path\n",
                    "import matplotlib.pyplot as plt\n",
                    "import numpy as np\n",
                    f"result_path = Path(r'''{result_json_path}''')\n",
                    "result = json.loads(result_path.read_text(encoding='utf-8'))\n",
                    "result.keys()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(8, 5))\n",
                    "plt.title(result['title_hv'])\n",
                    "plt.plot(result['db_saea']['hv_history'], marker='o', markersize=5, label='DB-SAEA')\n",
                    "plt.plot(result['nsga_eic']['hv_history'], marker='s', markersize=5, label='NSGA-EIC')\n",
                    "plt.xlabel('Step')\n",
                    "plt.ylabel('Hypervolume')\n",
                    "plt.grid(True)\n",
                    "plt.legend()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "db_front = np.asarray(result['db_saea']['final_front'], dtype=np.float32)\n",
                    "eic_front = np.asarray(result['nsga_eic']['final_front'], dtype=np.float32)\n",
                    "true_front = np.asarray(result['db_saea']['true_front'], dtype=np.float32)\n",
                    "plt.figure(figsize=(8, 5))\n",
                    "plt.title(result['title_front'])\n",
                    "plt.scatter(db_front[:, 0], db_front[:, 1], s=20, alpha=0.8, label='DB-SAEA')\n",
                    "plt.scatter(eic_front[:, 0], eic_front[:, 1], s=20, alpha=0.8, label='NSGA-EIC')\n",
                    "plt.plot(true_front[:, 0], true_front[:, 1], 'k-', linewidth=2, label='True Pareto Front')\n",
                    "plt.xlabel('f1')\n",
                    "plt.ylabel('f2')\n",
                    "plt.grid(True)\n",
                    "plt.legend()\n",
                    "plt.show()\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def train_db_multisource(args):
    demo.set_seed(args.seed)
    print(
        f"Training config | dqn_lr={args.dqn_lr:.1e} | "
        f"surrogate_nsga_steps={args.surrogate_nsga_steps} | reward_lambda={args.reward_lambda:.4f} | "
        f"max_fe={args.max_fe} | max_a1_actions={args.max_a1_actions}"
    )
    _ensure_worker_surrogates_ready(args)
    if ray is None:
        raise ImportError("Ray is not installed. Install ray to train DB-SAEA with 24 parallel environments.")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    task_names = _all_task_names()
    num_workers = max(1, min(int(args.num_workers), len(task_names)))
    rollout_steps = int(args.max_a1_actions + max(args.max_fe - args.archive_size, 0))
    print(
        f"Distributed training | num_workers={num_workers} | "
        f"task_count={len(task_names)} | rollout_steps={rollout_steps}"
    )

    replay_actor_cls = build_global_replay_buffer_actor()
    worker_actor_cls = build_db_saea_rollout_worker(policy_kwargs=_policy_kwargs(args))
    replay = replay_actor_cls.remote(capacity=args.replay_capacity)

    def env_factory(task_name: str, worker_id: int):
        return DBSAEATaskEnv(task_name=task_name, worker_id=worker_id, args=args)

    workers = [
        worker_actor_cls.remote(
            worker_id=idx,
            env_factory=env_factory,
            replay_buffer=replay,
        )
        for idx in range(num_workers)
    ]

    learner = DistributedLearner(
        policy=DBSAEAMetaPolicy(**_policy_kwargs(args)),
        device=args.device,
        lr=args.dqn_lr,
        num_actions=len(ACTION_NAMES),
    )
    reward_records: list[dict] = []
    epoch_mean_rewards: list[float] = []
    epoch_mean_losses: list[float] = []
    epsilon = float(args.epsilon_start)
    update_count = 0

    if args.start_epoch > 0:
        checkpoint_path = epoch_model_path(args.start_epoch)
        if checkpoint_path.exists():
            learner.global_policy.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
            learner.load_target_from_online()
            print(f"Loaded model from {checkpoint_path}")
        else:
            print(f"Checkpoint {checkpoint_path} not found, starting from scratch")

    for epoch in range(args.start_epoch, args.train_epochs):
        print(f"Epoch {epoch + 1}/{args.train_epochs}")
        epoch_reward_sum = 0.0
        epoch_reward_count = 0
        epoch_losses: list[float] = []
        worker_reward_sums: list[float] = []
        current_weights = learner.state_dict_cpu()
        total_transitions = 0
        for batch_start in range(0, len(task_names), num_workers):
            task_batch = task_names[batch_start : batch_start + num_workers]
            rollout_futures = [
                workers[slot].collect_trajectories.remote(
                    weights=current_weights,
                    epsilon=epsilon,
                    max_steps=rollout_steps,
                    task_name=task_name,
                )
                for slot, task_name in enumerate(task_batch)
            ]
            rollout_summaries = ray.get(rollout_futures)

            for summary in rollout_summaries:
                transition_count = int(summary.get("transition_count", 0))
                reward_sum = float(summary.get("reward_sum", 0.0))
                stats = summary.get("episode_stats", {})
                total_transitions += transition_count
                epoch_reward_sum += reward_sum
                epoch_reward_count += transition_count
                worker_reward_sums.append(reward_sum)
                reward_records.append(
                    {
                        "epoch": epoch + 1,
                        "task_name": summary.get("task_name"),
                        "problem": stats.get("problem_name"),
                        "dim": stats.get("dim"),
                        "transition_count": transition_count,
                        "reward_sum": reward_sum,
                        "reward_mean": reward_sum / max(transition_count, 1),
                        "true_evals": stats.get("true_evals"),
                        "a1_count": stats.get("a1_count"),
                        "best_obj1": stats.get("best_obj1"),
                        "step_count": stats.get("step_count"),
                        "epsilon": float(epsilon),
                    }
                )
                if stats:
                    print(
                        f"{stats['problem_name']}-{stats['dim']}D epoch {epoch + 1} done, "
                        f"true_evals={stats['true_evals']}, a1_count={stats['a1_count']}, "
                        f"eps={epsilon:.4f}, best_obj1={stats['best_obj1']:.6f}"
                    )

        mean_worker_reward = float(np.mean(worker_reward_sums)) if worker_reward_sums else 0.0
        print(
            f"Epoch {epoch + 1} rollout done | "
            f"workers={len(worker_reward_sums)} | mean_worker_reward={mean_worker_reward:.6f}"
        )

        updates_this_epoch = max(1, total_transitions // max(int(args.batch_size), 1))
        for _ in range(updates_this_epoch):
            batch = ray.get(replay.sample.remote(args.batch_size))
            loss_value = learner.update_model(batch=batch, gamma=args.gamma, grad_clip=args.grad_clip)
            if batch:
                epoch_losses.append(float(loss_value))
                update_count += 1
                if update_count % args.target_update_interval == 0:
                    learner.load_target_from_online()

        epsilon = max(float(args.epsilon_end), float(epsilon) * float(args.epsilon_decay))
        mean_reward = float(epoch_reward_sum / max(epoch_reward_count, 1))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        epoch_mean_rewards.append(mean_reward)
        epoch_mean_losses.append(mean_loss)
        print(f"Epoch {epoch + 1} mean reward: {mean_reward:.6f}")
        print(f"Epoch {epoch + 1} mean loss: {mean_loss:.6f}")
        torch.save(learner.global_policy.state_dict(), epoch_model_path(epoch + 1))
        if (epoch + 1) % 5 == 0:
            multisource.save_colab_model_checkpoint(
                learner.global_policy.state_dict(),
                f"db_zdt1_model_epoch_{epoch + 1}.pth",
            )

    torch.save(learner.global_policy.state_dict(), MODEL_PATH)
    print(f"DB-SAEA model saved to {MODEL_PATH}")
    _save_json(
        TRAIN_LOG_PATH,
        {
            "script": "db_zdt1_eva.py",
            "mode": "train_db_multisource_distributed",
            "model_path": MODEL_PATH,
            "source_problems": SOURCE_PROBLEMS,
            "source_dims": SOURCE_DIMS,
            "task_names": task_names,
            "num_workers": num_workers,
            "action_names": ACTION_NAMES,
            "epoch_mean_rewards": epoch_mean_rewards,
            "epoch_mean_losses": epoch_mean_losses,
            "records": reward_records,
        },
    )
    print(f"Training log saved to {TRAIN_LOG_PATH}")
    learner.global_policy.eval()
    return learner.global_policy


def load_or_train_db_policy(args):
    policy = DBSAEAMetaPolicy(**_policy_kwargs(args)).to(args.device)
    if getattr(args, "random_init_eval", False):
        print("Using randomly initialized DB-SAEA policy for evaluation.")
        policy.eval()
        return policy
    if args.eval_epoch is not None:
        checkpoint_path = epoch_model_path(args.eval_epoch)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Requested checkpoint not found: {checkpoint_path}")
        print(f"Using saved DB-SAEA epoch checkpoint from {checkpoint_path}")
        policy.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
        policy.eval()
        return policy
    if Path(MODEL_PATH).exists():
        print(f"Using saved DB-SAEA model from {MODEL_PATH}")
        policy.load_state_dict(torch.load(MODEL_PATH, map_location=args.device))
        policy.eval()
        return policy
    return train_db_multisource(args)


def run_db_saea_zdt1(args, policy, plot: bool = True, initial_archive_x: np.ndarray = None):
    entry = multisource.load_or_prepare_kan_surrogate("ZDT1", args.dim, _build_args_namespace(args))
    problem = entry["problem"]
    pretrain_x = np.asarray(entry["x"], dtype=np.float32)
    pretrain_y = np.asarray(entry["y"], dtype=np.float32)
    surrogates = entry["models"]
    ref_point = nsga_eic._reference_point("ZDT1", args.dim)
    print(f"Prepared KAN surrogate on ZDT1-{args.dim}D with {pretrain_x.shape[0]} samples.")

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
    hv_history = [_archive_hv(archive_y, ref_point)]
    action_history: list[str] = []
    q_value_history: list[list[float]] = []
    step = 0
    a1_count = 0
    fronts, _ = demo.fast_non_dominated_sort(archive_y)
    print(
        f"Init    | archive={archive_x.shape[0]} | "
        f"front0={len(fronts[0]) if fronts and fronts[0] else 0}"
    )
    print(f"Init HV | {hv_history[-1]:.6f}")

    while true_evals < args.max_fe:
        step += 1
        state, offspring_x, offspring_pred, offspring_sigma = _next_state_after_transition(
            problem=problem,
            surrogates=surrogates,
            archive_x=archive_x,
            archive_y=archive_y,
            true_evals=true_evals,
            args=args,
        )
        policy_state = _policy_inputs_from_state(state, args.device)
        q_bundle = policy.forward_with_state(
            **policy_state,
        )
        q_values = q_bundle["q_values"][0].detach().cpu().numpy()
        if a1_count >= args.max_a1_actions:
            action = int(1 + np.argmax(q_values[1:]))
        else:
            action = int(np.argmax(q_values))
        if action == 0:
            a1_count += 1
            hv_value = hv_history[-1]
        else:
            ranking = _ranking_for_action(
                action=action,
                archive_y=archive_y,
                offspring_pred=offspring_pred,
                offspring_sigma=offspring_sigma,
                seed=args.seed + step,
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
            hv_value = _archive_hv(archive_y, ref_point)
        hv_history.append(hv_value)
        action_history.append(ACTION_NAMES[action])
        q_value_history.append(q_values.tolist())

        print(
            f"Iter {step:02d} | archive={archive_x.shape[0]} | "
            f"HV={hv_value:.6f} | action={ACTION_NAMES[action]} | a1_count={a1_count} | pseudo_front={offspring_x.shape[0]}"
        )

        if action != 0:
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
        plt.plot(hv_history, marker="o", markersize=5, label="DB-SAEA")
        plt.xlabel("Step")
        plt.ylabel("Hypervolume")
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.title(f"{args.dim}D ZDT1 Pareto Front")
        plt.scatter(final_front[:, 0], final_front[:, 1], s=20, alpha=0.8, label="DB-SAEA")
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
        "action_history": action_history,
        "q_value_history": q_value_history,
    }


def run_comparison(args):
    policy = load_or_train_db_policy(args)
    shared_init_x = multisource.latin_hypercube_sample(
        lower=0.0,
        upper=1.0,
        n_samples=args.archive_size,
        dim=args.dim,
        seed=args.seed,
    )
    db_result = run_db_saea_zdt1(args, policy=policy, plot=False, initial_archive_x=shared_init_x)

    eic_args = _build_args_namespace(args)
    eic_result = nsga_eic.run_nsga_eic_problem(
        eic_args,
        problem_name="ZDT1",
        plot=False,
        initial_archive_x=shared_init_x,
    )

    print(f"\nDB-SAEA final HV: {db_result['hv_history'][-1]:.6f}")
    print(f"NSGA-EIC final HV: {eic_result['hv_history'][-1]:.6f}")
    print(f"Reference point: {db_result['ref_point']}")

    plt.figure(figsize=(8, 5))
    plt.title(f"{args.dim}D ZDT1 Hypervolume Comparison")
    plt.plot(db_result["hv_history"], marker="o", markersize=5, label="DB-SAEA")
    plt.plot(eic_result["hv_history"], marker="s", markersize=5, label="NSGA-EIC")
    plt.xlabel("Step")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.title(f"{args.dim}D ZDT1 Pareto Front Comparison")
    plt.scatter(db_result["final_front"][:, 0], db_result["final_front"][:, 1], s=20, alpha=0.8, label="DB-SAEA")
    plt.scatter(eic_result["final_front"][:, 0], eic_result["final_front"][:, 1], s=20, alpha=0.8, label="NSGA-EIC")
    plt.plot(db_result["true_front"][:, 0], db_result["true_front"][:, 1], "k-", linewidth=2, label="True Pareto Front")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(True)
    plt.legend()
    plt.show()

    payload = {
        "title_hv": f"{args.dim}D ZDT1 Hypervolume Comparison",
        "title_front": f"{args.dim}D ZDT1 Pareto Front Comparison",
        "db_saea": {
            "hv_history": db_result["hv_history"],
            "final_front": np.asarray(db_result["final_front"], dtype=np.float32).tolist(),
            "true_front": np.asarray(db_result["true_front"], dtype=np.float32).tolist(),
            "action_history": db_result["action_history"],
            "q_value_history": db_result["q_value_history"],
        },
        "nsga_eic": {
            "hv_history": eic_result["hv_history"],
            "final_front": np.asarray(eic_result["final_front"], dtype=np.float32).tolist(),
        },
    }
    _save_json(INFER_RESULT_PATH, payload)
    _write_infer_notebook(INFER_NOTEBOOK_PATH, INFER_RESULT_PATH)
    print(f"Inference result JSON saved to {INFER_RESULT_PATH}")
    print(f"Inference plot notebook saved to {INFER_NOTEBOOK_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DB-SAEA on mixed source problems and evaluate on 30D ZDT1")
    parser.add_argument("--archive_size", type=int, default=80)
    parser.add_argument("--offspring_size", type=int, default=24)
    parser.add_argument("--k_eval", type=int, default=1)
    parser.add_argument("--max_fe", type=int, default=120)
    parser.add_argument("--max_a1_actions", type=int, default=40)
    parser.add_argument("--mutation_sigma", type=float, default=0.12)
    parser.add_argument("--kan_steps", type=int, default=25)
    parser.add_argument("--kan_hidden", type=int, default=10)
    parser.add_argument("--kan_grid", type=int, default=5)
    parser.add_argument("--surrogate_nsga_steps", type=int, default=40)
    parser.add_argument("--ela_hidden", type=int, default=128)
    parser.add_argument("--ela_heads", type=int, default=8)
    parser.add_argument("--ela_ff", type=int, default=256)
    parser.add_argument("--dqn_hidden", type=int, default=256)
    parser.add_argument("--dqn_lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.1)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--target_update_interval", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--replay_capacity", type=int, default=50000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--reward_lambda", type=float, default=1.0)
    parser.add_argument("--hv_epsilon", type=float, default=1e-8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--eval_epoch", type=int, default=None)
    parser.add_argument("--random_init_eval", action="store_true", help="Run inference with randomly initialized DB-SAEA weights.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.train_only:
        train_db_multisource(args)
    else:
        run_comparison(args)


if __name__ == "__main__":
    main()
