from __future__ import annotations

import copy
import random
from collections import deque
from typing import Any

import torch
import torch.nn as nn

try:
    import ray
except ImportError:
    ray = None


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 2048):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )
        pe = torch.zeros(max_len, hidden_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TsAttnBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        return self.norm2(x + self.dropout(ff_out))


class StageOneLandscapeEncoder(nn.Module):
    """
    Input:  [batch, n_obj, n_dim, n_individual, hidden]
    Output: [batch, n_obj, n_individual, hidden]
    """

    def __init__(self, hidden_dim: int, n_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.cross_individual = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)
        self.cross_dimension = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)
        self.dimension_position = PositionalEncoding(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_obj, n_dim, n_individual, hidden_dim = x.shape

        x = x.reshape(batch_size * n_obj * n_dim, n_individual, hidden_dim)
        x = self.cross_individual(x)
        x = x.reshape(batch_size, n_obj, n_dim, n_individual, hidden_dim)

        x = x.permute(0, 1, 3, 2, 4).reshape(batch_size * n_obj * n_individual, n_dim, hidden_dim)
        x = self.dimension_position(x)
        x = self.cross_dimension(x)
        x = x.reshape(batch_size, n_obj, n_individual, n_dim, hidden_dim)

        return x.mean(dim=3)


class StageTwoSpaceAggregator(nn.Module):
    """
    Input:  [batch, n_obj, n_individual, hidden]
    Output: refined [batch, n_obj, n_individual, hidden], pooled [batch, hidden]
    """

    def __init__(self, hidden_dim: int, n_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.cross_individual = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)
        self.cross_objective = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)
        self.objective_position = PositionalEncoding(hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_obj, n_individual, hidden_dim = x.shape

        x = x.reshape(batch_size * n_obj, n_individual, hidden_dim)
        x = self.cross_individual(x)
        x = x.reshape(batch_size, n_obj, n_individual, hidden_dim)

        x = x.permute(0, 2, 1, 3).reshape(batch_size * n_individual, n_obj, hidden_dim)
        x = self.objective_position(x)
        x = self.cross_objective(x)
        x = x.reshape(batch_size, n_individual, n_obj, hidden_dim).permute(0, 2, 1, 3)

        pooled = x.mean(dim=2).mean(dim=1)
        return x, pooled


class BiSpaceELA(nn.Module):
    """
    Bi-space landscape encoder used by DB-SAEA.

    The module follows the PIE -> Stage One -> Stage Two pipeline:
    - PIE normalizes true/surrogate populations and reorganizes them into
      M_true in R^{m x d x n_true x 2} and M_sur in R^{m x d x n_sur x 3}
    - Stage One extracts objective-wise individual representations
    - Stage Two aggregates each space into a single global vector

    There is no cross-attention between true and surrogate spaces in this module.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 8,
        ff_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.Wtrue_emb = nn.Linear(2, hidden_dim)
        self.Wsur_emb = nn.Linear(3, hidden_dim)
        self.stage1_true = StageOneLandscapeEncoder(hidden_dim, n_heads, ff_dim, dropout)
        self.stage1_sur = StageOneLandscapeEncoder(hidden_dim, n_heads, ff_dim, dropout)
        self.stage2_true = StageTwoSpaceAggregator(hidden_dim, n_heads, ff_dim, dropout)
        self.stage2_sur = StageTwoSpaceAggregator(hidden_dim, n_heads, ff_dim, dropout)

    @staticmethod
    def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == 2 else x

    @staticmethod
    def _to_float_tensor(x, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.to(device=device, dtype=torch.float32)
        return torch.as_tensor(x, device=device, dtype=torch.float32)

    @staticmethod
    def _normalize_by_range(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        denom = (upper - lower).clamp_min(1e-12)
        return ((x - lower) / denom).clamp(0.0, 1.0)

    @staticmethod
    def _normalize_by_extrema(x: torch.Tensor) -> torch.Tensor:
        min_v = x.amin(dim=1, keepdim=True)
        max_v = x.amax(dim=1, keepdim=True)
        denom = (max_v - min_v).clamp_min(1e-12)
        return ((x - min_v) / denom).clamp(0.0, 1.0)

    def _prepare_inputs(
        self,
        x_true: torch.Tensor,
        y_true: torch.Tensor,
        x_sur: torch.Tensor,
        y_sur: torch.Tensor,
        sigma_sur: torch.Tensor,
        lower_bound,
        upper_bound,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(x_true):
            device = x_true.device
        elif torch.is_tensor(x_sur):
            device = x_sur.device
        else:
            device = torch.device("cpu")

        x_true = self._ensure_batch(self._to_float_tensor(x_true, device=device))
        y_true = self._ensure_batch(self._to_float_tensor(y_true, device=device))
        x_sur = self._ensure_batch(self._to_float_tensor(x_sur, device=device))
        y_sur = self._ensure_batch(self._to_float_tensor(y_sur, device=device))
        sigma_sur = self._ensure_batch(self._to_float_tensor(sigma_sur, device=device))

        dtype = x_true.dtype
        lower = torch.as_tensor(lower_bound, device=device, dtype=dtype)
        upper = torch.as_tensor(upper_bound, device=device, dtype=dtype)

        if lower.dim() == 0:
            lower = lower.repeat(x_true.size(-1))
        if upper.dim() == 0:
            upper = upper.repeat(x_true.size(-1))

        lower = lower.view(1, 1, -1)
        upper = upper.view(1, 1, -1)

        x_true = self._normalize_by_range(x_true, lower, upper)
        x_sur = self._normalize_by_range(x_sur, lower, upper)
        y_true = self._normalize_by_extrema(y_true)
        y_sur = self._normalize_by_extrema(y_sur)
        sigma_sur = self._normalize_by_extrema(sigma_sur)

        x_true_expand = x_true.transpose(1, 2).unsqueeze(1).unsqueeze(-1)
        x_true_expand = x_true_expand.expand(-1, y_true.size(-1), -1, -1, -1)
        y_true_expand = y_true.transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        y_true_expand = y_true_expand.expand(-1, -1, x_true.size(-1), -1, -1)
        m_true = torch.cat((x_true_expand, y_true_expand), dim=-1)

        x_sur_expand = x_sur.transpose(1, 2).unsqueeze(1).unsqueeze(-1)
        x_sur_expand = x_sur_expand.expand(-1, y_sur.size(-1), -1, -1, -1)
        y_sur_expand = y_sur.transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        y_sur_expand = y_sur_expand.expand(-1, -1, x_sur.size(-1), -1, -1)
        sigma_expand = sigma_sur.transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        sigma_expand = sigma_expand.expand(-1, -1, x_sur.size(-1), -1, -1)
        m_sur = torch.cat((x_sur_expand, y_sur_expand, sigma_expand), dim=-1)

        return m_true, m_sur

    def forward(
        self,
        x_true: torch.Tensor,
        y_true: torch.Tensor,
        x_sur: torch.Tensor,
        y_sur: torch.Tensor,
        sigma_sur: torch.Tensor,
        lower_bound,
        upper_bound,
    ) -> dict[str, torch.Tensor]:
        m_true, m_sur = self._prepare_inputs(
            x_true=x_true,
            y_true=y_true,
            x_sur=x_sur,
            y_sur=y_sur,
            sigma_sur=sigma_sur,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        e_true = self.Wtrue_emb(m_true)
        e_sur = self.Wsur_emb(m_sur)

        s_true = self.stage1_true(e_true)
        s_sur = self.stage1_sur(e_sur)

        s_true_refined, s_true_prime = self.stage2_true(s_true)
        s_sur_refined, s_sur_prime = self.stage2_sur(s_sur)
        z = torch.cat([s_true_prime, s_sur_prime], dim=-1)

        return {
            "M_true": m_true,
            "M_sur": m_sur,
            "E_true": e_true,
            "E_sur": e_sur,
            "S_true": s_true,
            "S_sur": s_sur,
            "S_true_refined": s_true_refined,
            "S_sur_refined": s_sur_refined,
            "s_true_prime": s_true_prime,
            "s_sur_prime": s_sur_prime,
            "z": z,
        }


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network operating on the global DB-SAEA state vector.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_actions: int = 6):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_features = self.shared_net(x)
        value = self.value_net(shared_features)
        advantage = self.advantage_net(shared_features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class DBSAEAMetaPolicy(nn.Module):
    """
    DB-SAEA meta-policy built from:
    - Bi-space ELA for state encoding
    - Dueling DQN for action-value prediction
    """

    def __init__(
        self,
        ela_hidden_dim: int = 128,
        n_heads: int = 8,
        ff_dim: int = 256,
        dqn_hidden_dim: int = 256,
        num_actions: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.ela = BiSpaceELA(
            hidden_dim=ela_hidden_dim,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.dqn = DuelingDQN(
            input_dim=2 * ela_hidden_dim + 1,
            hidden_dim=dqn_hidden_dim,
            num_actions=num_actions,
        )

    @staticmethod
    def _prepare_progress(
        progress,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        progress = torch.as_tensor(progress, device=device, dtype=dtype)
        if progress.dim() == 0:
            progress = progress.repeat(batch_size)
        progress = progress.reshape(batch_size, -1)
        if progress.size(1) != 1:
            progress = progress[:, :1]
        return progress.clamp(0.0, 1.0)

    def encode_state(
        self,
        x_true: torch.Tensor,
        y_true: torch.Tensor,
        x_sur: torch.Tensor,
        y_sur: torch.Tensor,
        sigma_sur: torch.Tensor,
        lower_bound,
        upper_bound,
        progress,
    ) -> dict[str, torch.Tensor]:
        ela_out = self.ela(
            x_true=x_true,
            y_true=y_true,
            x_sur=x_sur,
            y_sur=y_sur,
            sigma_sur=sigma_sur,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        z = ela_out["z"]
        progress_tensor = self._prepare_progress(
            progress=progress,
            batch_size=z.size(0),
            device=z.device,
            dtype=z.dtype,
        )
        state_vector = torch.cat([z, progress_tensor], dim=-1)
        ela_out["progress"] = progress_tensor
        ela_out["state_vector"] = state_vector
        return ela_out

    def forward(
        self,
        x_true: torch.Tensor,
        y_true: torch.Tensor,
        x_sur: torch.Tensor,
        y_sur: torch.Tensor,
        sigma_sur: torch.Tensor,
        lower_bound,
        upper_bound,
        progress,
    ) -> torch.Tensor:
        state = self.encode_state(
            x_true=x_true,
            y_true=y_true,
            x_sur=x_sur,
            y_sur=y_sur,
            sigma_sur=sigma_sur,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            progress=progress,
        )
        return self.dqn(state["state_vector"])

    def forward_with_state(
        self,
        x_true: torch.Tensor,
        y_true: torch.Tensor,
        x_sur: torch.Tensor,
        y_sur: torch.Tensor,
        sigma_sur: torch.Tensor,
        lower_bound,
        upper_bound,
        progress,
    ) -> dict[str, torch.Tensor]:
        state = self.encode_state(
            x_true=x_true,
            y_true=y_true,
            x_sur=x_sur,
            y_sur=y_sur,
            sigma_sur=sigma_sur,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            progress=progress,
        )
        state["q_values"] = self.dqn(state["state_vector"])
        return state

    def select_action(self, *args, epsilon: float = 0.0, **kwargs) -> int:
        if torch.rand(1).item() < epsilon:
            return int(torch.randint(0, self.num_actions, (1,)).item())

        with torch.no_grad():
            q_values = self.forward(*args, **kwargs)
            return int(torch.argmax(q_values, dim=1).item())


def _state_signature(state: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(torch.as_tensor(state["x_true"]).shape),
        tuple(torch.as_tensor(state["y_true"]).shape),
        tuple(torch.as_tensor(state["x_sur"]).shape),
        tuple(torch.as_tensor(state["y_sur"]).shape),
        tuple(torch.as_tensor(state["sigma_sur"]).shape),
        tuple(torch.as_tensor(state["lower_bound"]).shape),
        tuple(torch.as_tensor(state["upper_bound"]).shape),
    )


def _stack_state_dicts(states: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    stacked: dict[str, Any] = {}
    tensor_keys = ["x_true", "y_true", "x_sur", "y_sur", "sigma_sur"]
    for key in tensor_keys:
        stacked[key] = torch.stack(
            [torch.as_tensor(state[key], dtype=torch.float32, device=device) for state in states],
            dim=0,
        )

    stacked["lower_bound"] = torch.as_tensor(states[0]["lower_bound"], dtype=torch.float32, device=device)
    stacked["upper_bound"] = torch.as_tensor(states[0]["upper_bound"], dtype=torch.float32, device=device)
    stacked["progress"] = torch.as_tensor(
        [float(state["progress"]) for state in states],
        dtype=torch.float32,
        device=device,
    )
    return stacked


def _clone_state_for_buffer(state: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


class GlobalReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transitions: list[dict[str, Any]]) -> None:
        self.buffer.extend(transitions)

    def sample(self, batch_size: int) -> list[dict[str, Any]] | None:
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)

    def size(self) -> int:
        return len(self.buffer)


class DistributedLearner:
    """
    Central learner for distributed off-policy DB-SAEA training.

    Expected transition format:
    {
        "state": {
            "x_true", "y_true", "x_sur", "y_sur", "sigma_sur",
            "lower_bound", "upper_bound", "progress"
        },
        "action": int,
        "reward": float,
        "next_state": same structure as state,
        "done": bool,
    }
    """

    def __init__(
        self,
        policy: DBSAEAMetaPolicy | None = None,
        device: torch.device | str | None = None,
        lr: float = 1e-4,
        num_actions: int = 6,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.global_policy = policy if policy is not None else DBSAEAMetaPolicy(num_actions=num_actions)
        self.global_policy = self.global_policy.to(self.device)
        self.target_policy = copy.deepcopy(self.global_policy).to(self.device)
        self.target_policy.eval()
        self.optimizer = torch.optim.Adam(self.global_policy.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def state_dict_cpu(self) -> dict[str, Any]:
        return {key: value.detach().cpu() for key, value in self.global_policy.state_dict().items()}

    def load_target_from_online(self) -> None:
        self.target_policy.load_state_dict(self.global_policy.state_dict())
        self.target_policy.eval()

    def update_model(
        self,
        batch: list[dict[str, Any]] | None,
        gamma: float = 0.99,
        grad_clip: float = 1.0,
    ) -> float:
        if not batch:
            return 0.0

        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for transition in batch:
            signature = (_state_signature(transition["state"]), _state_signature(transition["next_state"]))
            grouped.setdefault(signature, []).append(transition)

        total_count = sum(len(group) for group in grouped.values())
        if total_count == 0:
            return 0.0

        self.global_policy.train()
        self.optimizer.zero_grad()
        total_loss = None

        for transitions in grouped.values():
            states = _stack_state_dicts([item["state"] for item in transitions], device=self.device)
            next_states = _stack_state_dicts([item["next_state"] for item in transitions], device=self.device)
            actions = torch.as_tensor(
                [int(item["action"]) for item in transitions],
                dtype=torch.long,
                device=self.device,
            )
            rewards = torch.as_tensor(
                [float(item["reward"]) for item in transitions],
                dtype=torch.float32,
                device=self.device,
            )
            dones = torch.as_tensor(
                [float(item["done"]) for item in transitions],
                dtype=torch.float32,
                device=self.device,
            )

            q_values = self.global_policy(**states)
            q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_online_q = self.global_policy(**next_states)
                next_actions = next_online_q.argmax(dim=1, keepdim=True)
                next_target_q = self.target_policy(**next_states)
                next_q = next_target_q.gather(1, next_actions).squeeze(1)
                targets = rewards + gamma * next_q * (1.0 - dones)

            group_loss = self.loss_fn(q_action, targets)
            weight = float(len(transitions)) / float(total_count)
            weighted_loss = group_loss * weight
            total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss

        if total_loss is None:
            return 0.0

        total_loss.backward()
        nn.utils.clip_grad_norm_(self.global_policy.parameters(), max_norm=grad_clip)
        self.optimizer.step()
        return float(total_loss.detach().cpu())


def build_db_saea_rollout_worker(policy_kwargs: dict[str, Any] | None = None):
    if ray is None:
        raise ImportError("Ray is required to build distributed DB-SAEA rollout workers.")

    policy_kwargs = dict(policy_kwargs or {})

    @ray.remote(num_cpus=1)
    class DBSAEARolloutWorker:
        def __init__(self, worker_id: int, task_name: str, env_factory):
            self.worker_id = worker_id
            self.task_name = task_name
            self.env = env_factory(task_name, worker_id)
            self.local_policy = DBSAEAMetaPolicy(**policy_kwargs).cpu()
            self.local_policy.eval()

        def sync_weights(self, weights: dict[str, Any]) -> None:
            self.local_policy.load_state_dict(weights)

        def collect_trajectories(
            self,
            weights: dict[str, Any],
            epsilon: float,
            max_steps: int = 50,
        ) -> list[dict[str, Any]]:
            self.sync_weights(weights)
            transitions: list[dict[str, Any]] = []
            state = self.env.reset()

            for _ in range(max_steps):
                action = self.local_policy.select_action(
                    x_true=state["x_true"],
                    y_true=state["y_true"],
                    x_sur=state["x_sur"],
                    y_sur=state["y_sur"],
                    sigma_sur=state["sigma_sur"],
                    lower_bound=state["lower_bound"],
                    upper_bound=state["upper_bound"],
                    progress=state["progress"],
                    epsilon=epsilon,
                )

                next_state, reward, done, _ = self.env.step(action)
                transitions.append(
                    {
                        "state": _clone_state_for_buffer(state),
                        "action": int(action),
                        "reward": float(reward),
                        "next_state": _clone_state_for_buffer(next_state),
                        "done": bool(done),
                        "task_name": self.task_name,
                    }
                )
                state = next_state
                if done:
                    break

            return transitions

    return DBSAEARolloutWorker


def build_global_replay_buffer_actor():
    if ray is None:
        raise ImportError("Ray is required to build the distributed replay buffer.")
    return ray.remote(GlobalReplayBuffer)


def train_db_saea_distributed(
    task_names: list[str],
    env_factory,
    num_workers: int = 4,
    iterations: int = 1000,
    batch_size: int = 128,
    replay_capacity: int = 100000,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_decay: float = 0.995,
    target_update_interval: int = 10,
    rollout_steps: int = 50,
    learner_lr: float = 1e-4,
    grad_clip: float = 1.0,
    ray_init_kwargs: dict[str, Any] | None = None,
    policy_kwargs: dict[str, Any] | None = None,
) -> tuple[DistributedLearner, list[dict[str, float]]]:
    """
    Distributed off-policy training loop for DB-SAEA.

    The environment created by env_factory(task_name, worker_id) must expose:
    - reset() -> state_dict
    - step(action) -> next_state, reward, done, info

    Each state_dict must contain:
    x_true, y_true, x_sur, y_sur, sigma_sur, lower_bound, upper_bound, progress
    """

    if ray is None:
        raise ImportError("Ray is not installed. Install ray to use distributed DB-SAEA training.")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, **(ray_init_kwargs or {}))

    replay_actor_cls = build_global_replay_buffer_actor()
    worker_actor_cls = build_db_saea_rollout_worker(policy_kwargs=policy_kwargs)
    replay_buffer = replay_actor_cls.remote(capacity=replay_capacity)

    worker_tasks = [task_names[idx % len(task_names)] for idx in range(num_workers)]
    workers = [
        worker_actor_cls.remote(worker_id=idx, task_name=worker_tasks[idx], env_factory=env_factory)
        for idx in range(num_workers)
    ]

    learner = DistributedLearner(
        policy=DBSAEAMetaPolicy(**(policy_kwargs or {})),
        lr=learner_lr,
    )
    learner.load_target_from_online()

    epsilon = float(epsilon_start)
    history: list[dict[str, float]] = []

    for iteration in range(iterations):
        current_weights = learner.state_dict_cpu()
        rollout_futures = [
            worker.collect_trajectories.remote(
                weights=current_weights,
                epsilon=epsilon,
                max_steps=rollout_steps,
            )
            for worker in workers
        ]
        rollout_batches = ray.get(rollout_futures)
        for transitions in rollout_batches:
            if transitions:
                replay_buffer.add.remote(transitions)

        batch = ray.get(replay_buffer.sample.remote(batch_size))
        loss = learner.update_model(batch=batch, gamma=gamma, grad_clip=grad_clip)

        if iteration % target_update_interval == 0:
            learner.load_target_from_online()

        epsilon = max(float(epsilon_end), float(epsilon) * float(epsilon_decay))
        buffer_size = int(ray.get(replay_buffer.size.remote()))
        history.append(
            {
                "iteration": float(iteration),
                "loss": float(loss),
                "epsilon": float(epsilon),
                "buffer_size": float(buffer_size),
            }
        )

    return learner, history
