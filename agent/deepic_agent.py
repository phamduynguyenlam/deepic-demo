from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pymoo.indicators.hv import HV


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


class LandscapeEncoder(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.cross_individual = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)
        self.cross_dimension = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)
        self.position = PositionalEncoding(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_obj, n_dim, n_ind, hidden_dim = x.shape

        x = x.reshape(bsz * n_obj * n_dim, n_ind, hidden_dim)
        x = self.cross_individual(x)
        x = x.reshape(bsz, n_obj, n_dim, n_ind, hidden_dim)

        x = x.permute(0, 1, 3, 2, 4).reshape(bsz * n_obj * n_ind, n_dim, hidden_dim)
        x = self.position(x)
        x = self.cross_dimension(x)
        x = x.reshape(bsz, n_obj, n_ind, n_dim, hidden_dim)

        return x.mean(dim=3)


class CrossSpaceAttention(nn.Module):
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

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(query, key, value, need_weights=False)
        x = self.norm1(query + self.dropout(attn_out))
        ff_out = self.ff(x)
        return self.norm2(x + self.dropout(ff_out))


class DecoderContextAttention(nn.Module):
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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(query + self.dropout(attn_out))
        ff_out = self.ff(x)
        return self.norm2(x + self.dropout(ff_out))


class ScoringHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_surr: torch.Tensor) -> torch.Tensor:
        return self.score_net(h_surr).squeeze(-1)


class _DeepICBase(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 8,
        ff_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_true = nn.Linear(2, hidden_dim)
        self.W_surr = nn.Linear(3, hidden_dim)
        self.encoder_true = LandscapeEncoder(hidden_dim, n_heads, ff_dim, dropout)
        self.encoder_surr = LandscapeEncoder(hidden_dim, n_heads, ff_dim, dropout)
        self.cross_space_attn = CrossSpaceAttention(hidden_dim, n_heads, ff_dim, dropout)

    @staticmethod
    def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == 2 else x

    @staticmethod
    def _normalize_by_range(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        denom = (upper - lower).clamp_min(1e-12)
        return ((x - lower) / denom).clamp(0.0, 1.0)

    @staticmethod
    def _normalize_by_extrema(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        stacked = torch.cat(tensors, dim=1)
        min_v = stacked.amin(dim=1, keepdim=True)
        max_v = stacked.amax(dim=1, keepdim=True)
        denom = (max_v - min_v).clamp_min(1e-12)
        return tuple(((tensor - min_v) / denom).clamp(0.0, 1.0) for tensor in tensors)

    @staticmethod
    def _prepare_progress(progress, device: torch.device, dtype: torch.dtype, batch_size: int) -> torch.Tensor:
        progress = torch.as_tensor(progress, device=device, dtype=dtype)
        if progress.dim() == 0:
            progress = progress.repeat(batch_size)
        progress = progress.reshape(batch_size, -1)
        if progress.size(1) != 1:
            progress = progress[:, :1]
        return progress.clamp(0.0, 1.0)

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
        x_true = self._ensure_batch(x_true).float()
        y_true = self._ensure_batch(y_true).float()
        x_sur = self._ensure_batch(x_sur).float()
        y_sur = self._ensure_batch(y_sur).float()
        sigma_sur = self._ensure_batch(sigma_sur).float()

        device = x_true.device
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
        y_true, y_sur = self._normalize_by_extrema(y_true, y_sur)
        (sigma_sur,) = self._normalize_by_extrema(sigma_sur)

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
        m_surr = torch.cat((x_sur_expand, y_sur_expand, sigma_expand), dim=-1)

        return m_true, m_surr

    def encode(
        self,
        x_true: torch.Tensor,
        y_true: torch.Tensor,
        x_sur: torch.Tensor,
        y_sur: torch.Tensor,
        sigma_sur: torch.Tensor,
        progress,
        lower_bound,
        upper_bound,
    ) -> dict[str, torch.Tensor]:
        m_true, m_surr = self._prepare_inputs(
            x_true, y_true, x_sur, y_sur, sigma_sur, lower_bound, upper_bound
        )

        e_true = self.W_true(m_true)
        e_surr = self.W_surr(m_surr)

        s_true = self.encoder_true(e_true)
        s_surr = self.encoder_surr(e_surr)

        h_true = s_true.mean(dim=1)
        h_surr_raw = s_surr.mean(dim=1)
        h_surr = self.cross_space_attn(h_surr_raw, h_true, h_true)

        progress = self._prepare_progress(
            progress=progress,
            device=h_true.device,
            dtype=h_true.dtype,
            batch_size=h_true.size(0),
        )

        return {
            "H_true": h_true,
            "H_surr": h_surr,
            "progress": progress,
        }

    def decode_ranking(
        self,
        H_surr: torch.Tensor,
        H_true: torch.Tensor,
        progress: torch.Tensor,
        target_ranking: torch.Tensor | None = None,
        decode_type: str = "greedy",
        max_decode_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(
        self,
        x_true: torch.Tensor,
        y_true: torch.Tensor,
        x_sur: torch.Tensor,
        y_sur: torch.Tensor,
        sigma_sur: torch.Tensor,
        progress,
        lower_bound,
        upper_bound,
        target_ranking: torch.Tensor | None = None,
        decode_type: str = "greedy",
        max_decode_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode(
            x_true=x_true,
            y_true=y_true,
            x_sur=x_sur,
            y_sur=y_sur,
            sigma_sur=sigma_sur,
            progress=progress,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        decoded = self.decode_ranking(
            H_surr=encoded["H_surr"],
            H_true=encoded["H_true"],
            progress=encoded["progress"],
            target_ranking=target_ranking,
            decode_type=decode_type,
            max_decode_steps=max_decode_steps,
        )
        encoded.update(decoded)
        return encoded

    @staticmethod
    def ranking_loss(logits: torch.Tensor, target_ranking: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 2:
            ordered_scores = logits.gather(1, target_ranking)
            losses = []
            for i in range(ordered_scores.size(1)):
                losses.append(torch.logsumexp(ordered_scores[:, i:], dim=1) - ordered_scores[:, i])
            return torch.stack(losses, dim=1).sum(dim=1).mean()

        if logits.dim() != 3:
            raise ValueError(f"Unsupported logits shape for ranking loss: {tuple(logits.shape)}")

        losses = []
        for step in range(min(logits.size(1), target_ranking.size(1))):
            losses.append(F.cross_entropy(logits[:, step, :], target_ranking[:, step], reduction="none"))
        return torch.stack(losses, dim=1).sum(dim=1).mean()

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        return np.all(a <= b) and np.any(a < b)

    @classmethod
    def pareto_front(cls, values: np.ndarray) -> np.ndarray:
        values = cls._to_numpy(values).astype(np.float32)
        keep = []
        for i in range(values.shape[0]):
            dominated = False
            for j in range(values.shape[0]):
                if i == j:
                    continue
                if cls._dominates(values[j], values[i]):
                    dominated = True
                    break
            if not dominated:
                keep.append(i)
        return values[np.asarray(keep, dtype=np.int64)]

    @classmethod
    def fpareto_improvement_reward(
        cls,
        previous_front: np.ndarray,
        selected_objectives: np.ndarray,
    ) -> float:
        previous_front = cls.pareto_front(previous_front)
        selected_objectives = cls._to_numpy(selected_objectives).astype(np.float32)

        improved = False
        for candidate in selected_objectives:
            if not any(cls._dominates(prev, candidate) for prev in previous_front):
                improved = True
                break

        if not improved:
            return -1.0

        reward = 1.0
        origin = np.zeros(previous_front.shape[1], dtype=np.float32)
        for candidate in selected_objectives:
            distances = np.abs(previous_front - candidate).sum(axis=1)
            nearest_idx = int(np.argmin(distances))
            d_i = float(distances[nearest_idx])
            d_ref_i = float(np.abs(previous_front[nearest_idx] - origin).sum())
            reward += d_i / max(d_ref_i, 1e-12)
        return reward

    @classmethod
    def pareto_improvement_reward(
        cls,
        previous_front: np.ndarray,
        selected_objectives: np.ndarray,
    ) -> float:
        return cls.fpareto_improvement_reward(previous_front, selected_objectives)

    @staticmethod
    def sequence_log_prob(logits: torch.Tensor, ranking: torch.Tensor, top_k: int | None = None) -> torch.Tensor:
        if logits.dim() == 2:
            ordered_scores = logits.gather(1, ranking)
            if top_k is None:
                top_k = ordered_scores.size(1)
            top_k = min(top_k, ordered_scores.size(1))
            terms = []
            for i in range(top_k):
                terms.append(ordered_scores[:, i] - torch.logsumexp(ordered_scores[:, i:], dim=1))
            return torch.stack(terms, dim=1).sum(dim=1)

        if logits.dim() != 3:
            raise ValueError(f"Unsupported logits shape for sequence log prob: {tuple(logits.shape)}")

        if top_k is None:
            top_k = ranking.size(1)
        top_k = min(top_k, ranking.size(1), logits.size(1))
        log_probs = F.log_softmax(logits[:, :top_k, :], dim=-1)
        chosen = ranking[:, :top_k].unsqueeze(-1)
        return log_probs.gather(2, chosen).squeeze(-1).sum(dim=1)

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self._optimizer = optimizer

    def set_baseline(self, baseline: float = 0.0):
        self._baseline = baseline

    def set_top_k(self, top_k: int = 1):
        self._top_k = top_k

    def __init_training_config(self):
        if not hasattr(self, "_optimizer"):
            self._optimizer = None
        if not hasattr(self, "_baseline"):
            self._baseline = 0.0
        if not hasattr(self, "_top_k"):
            self._top_k = 1

    def train_episode(self, env):
        self.__init_training_config()
        if self._optimizer is None:
            raise ValueError("Optimizer must be set via set_optimizer() before training.")

        super().train(True)
        state = env.reset()
        done = False
        total_return = 0.0
        total_loss = 0.0
        step = 0

        while not done:
            x_true = state["x_true"]
            y_true = state["y_true"]
            x_sur = state["x_sur"]
            y_sur = state["y_sur"]
            sigma_sur = state["sigma_sur"]
            progress = state.get("progress", 0.0)
            lower_bound = state["lower_bound"]
            upper_bound = state["upper_bound"]
            true_eval_fn = state.get("true_eval_fn", None)

            device = next(self.parameters()).device
            x_true_t = torch.as_tensor(x_true, dtype=torch.float32, device=device)
            y_true_t = torch.as_tensor(y_true, dtype=torch.float32, device=device)
            x_sur_t = torch.as_tensor(x_sur, dtype=torch.float32, device=device)
            y_sur_t = torch.as_tensor(y_sur, dtype=torch.float32, device=device)
            sigma_sur_t = torch.as_tensor(sigma_sur, dtype=torch.float32, device=device)

            out = self(
                x_true=x_true_t,
                y_true=y_true_t,
                x_sur=x_sur_t,
                y_sur=y_sur_t,
                sigma_sur=sigma_sur_t,
                progress=progress,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                decode_type="sample",
                max_decode_steps=self._top_k,
            )

            ranking = out["ranking"]
            logits = out["logits"]
            top_k = min(self._top_k, ranking.size(1))
            selected_idx = ranking[:, :top_k]

            x_sur_np = self._to_numpy(x_sur_t)
            y_true_np = self._to_numpy(y_true_t)
            if x_sur_np.ndim == 2:
                x_sur_np = x_sur_np[None, ...]
            if y_true_np.ndim == 2:
                y_true_np = y_true_np[None, ...]

            rewards = []
            for batch_id in range(ranking.size(0)):
                chosen_x = x_sur_np[batch_id][selected_idx[batch_id].detach().cpu().numpy()]
                if true_eval_fn is not None:
                    chosen_y = self._to_numpy(true_eval_fn(chosen_x)).astype(np.float32)
                else:
                    chosen_y = self._to_numpy(y_sur_t)[selected_idx[batch_id].detach().cpu().numpy()]
                rewards.append(self.pareto_improvement_reward(y_true_np[batch_id], chosen_y))

            reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=logits.device)
            log_prob = self.sequence_log_prob(logits, ranking, top_k=top_k)
            advantage = reward_tensor - float(self._baseline)
            loss = -(advantage.detach() * log_prob).mean()

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_return += float(reward_tensor.mean().detach().cpu().numpy())
            step += 1

            action = {
                "selected_idx": selected_idx.detach().cpu().numpy(),
                "ranking": ranking.detach().cpu().numpy(),
            }
            state, _, done = env.step(action)

        return {
            "loss": total_loss / max(step, 1),
            "return": total_return / max(step, 1),
            "steps": step,
            "normalizer": getattr(env.optimizer, "cost", [0])[0] if hasattr(env, "optimizer") else 0,
            "gbest": getattr(env.optimizer, "cost", [0])[-1] if hasattr(env, "optimizer") else 0,
        }

    def rollout_episode(self, env):
        self.__init_training_config()
        super().train(False)
        state = env.reset()
        done = False
        total_return = 0.0

        with torch.no_grad():
            while not done:
                x_true = state["x_true"]
                y_true = state["y_true"]
                x_sur = state["x_sur"]
                y_sur = state["y_sur"]
                sigma_sur = state["sigma_sur"]
                progress = state.get("progress", 0.0)
                lower_bound = state["lower_bound"]
                upper_bound = state["upper_bound"]
                true_eval_fn = state.get("true_eval_fn", None)

                device = next(self.parameters()).device
                x_true_t = torch.as_tensor(x_true, dtype=torch.float32, device=device)
                y_true_t = torch.as_tensor(y_true, dtype=torch.float32, device=device)
                x_sur_t = torch.as_tensor(x_sur, dtype=torch.float32, device=device)
                y_sur_t = torch.as_tensor(y_sur, dtype=torch.float32, device=device)
                sigma_sur_t = torch.as_tensor(sigma_sur, dtype=torch.float32, device=device)

                out = self(
                    x_true=x_true_t,
                    y_true=y_true_t,
                    x_sur=x_sur_t,
                    y_sur=y_sur_t,
                    sigma_sur=sigma_sur_t,
                    progress=progress,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    decode_type="greedy",
                    max_decode_steps=self._top_k,
                )

                ranking = out["ranking"]
                top_k = min(self._top_k, ranking.size(1))
                selected_idx = ranking[:, :top_k]

                x_sur_np = self._to_numpy(x_sur_t)
                y_true_np = self._to_numpy(y_true_t)
                if x_sur_np.ndim == 2:
                    x_sur_np = x_sur_np[None, ...]
                if y_true_np.ndim == 2:
                    y_true_np = y_true_np[None, ...]

                rewards = []
                for batch_id in range(ranking.size(0)):
                    chosen_x = x_sur_np[batch_id][selected_idx[batch_id].detach().cpu().numpy()]
                    if true_eval_fn is not None:
                        chosen_y = self._to_numpy(true_eval_fn(chosen_x)).astype(np.float32)
                    else:
                        chosen_y = self._to_numpy(y_sur_t)[selected_idx[batch_id].detach().cpu().numpy()]
                    rewards.append(self.pareto_improvement_reward(y_true_np[batch_id], chosen_y))

                reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=ranking.device)
                total_return += float(reward_tensor.mean().detach().cpu().numpy())

                action = {
                    "selected_idx": selected_idx.detach().cpu().numpy(),
                    "ranking": ranking.detach().cpu().numpy(),
                }
                state, _, done = env.step(action)

        return {
            "cost": getattr(env.optimizer, "cost", []) if hasattr(env, "optimizer") else [],
            "fes": getattr(env.optimizer, "fes", []) if hasattr(env, "optimizer") else [],
            "return": total_return,
        }


class DeepIC(_DeepICBase):
    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 8,
        ff_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__(hidden_dim=hidden_dim, n_heads=n_heads, ff_dim=ff_dim, dropout=dropout)
        self.start_token = nn.Parameter(torch.zeros(hidden_dim))
        self.mean_linear = nn.Linear(2 * hidden_dim, hidden_dim)
        self.query_projection = nn.Linear(hidden_dim + 1, hidden_dim)
        self.decoder_attn = DecoderContextAttention(hidden_dim, n_heads, ff_dim, dropout)

    def decode_ranking(
        self,
        H_surr: torch.Tensor,
        H_true: torch.Tensor,
        progress: torch.Tensor,
        target_ranking: torch.Tensor | None = None,
        decode_type: str = "greedy",
        max_decode_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, n_sur, hidden_dim = H_surr.shape
        dtype = H_surr.dtype
        context = self.mean_linear(torch.cat([H_surr.mean(dim=1), H_true.mean(dim=1)], dim=-1))
        context = context + self.start_token.unsqueeze(0)

        selected_mask = torch.zeros(bsz, n_sur, dtype=torch.bool, device=H_surr.device)
        logits_steps = []
        ranking_steps = []
        decode_steps = n_sur
        if max_decode_steps is not None:
            decode_steps = min(int(max_decode_steps), n_sur)
        if target_ranking is not None:
            decode_steps = min(decode_steps, target_ranking.size(1))

        for step in range(decode_steps):
            remaining_ratio = (1.0 - progress) * float(n_sur - step) / max(float(n_sur), 1.0)
            query_input = torch.cat([context, remaining_ratio.to(dtype=dtype)], dim=-1)
            query = self.query_projection(query_input).unsqueeze(1)
            step_mask = selected_mask
            refined_context = self.decoder_attn(
                query=query,
                key=H_surr,
                value=H_surr,
                key_padding_mask=step_mask,
            ).squeeze(1)

            logits = torch.einsum("bnh,bh->bn", H_surr, refined_context) / np.sqrt(float(hidden_dim))
            logits = logits.masked_fill(step_mask, -1e9)
            logits_steps.append(logits.unsqueeze(1))

            if target_ranking is not None:
                chosen_idx = target_ranking[:, step]
            elif decode_type == "sample":
                chosen_idx = torch.distributions.Categorical(logits=logits).sample()
            else:
                chosen_idx = torch.argmax(logits, dim=-1)

            ranking_steps.append(chosen_idx.unsqueeze(1))
            selected_mask = selected_mask.scatter(1, chosen_idx.unsqueeze(1), True)
            gather_idx = chosen_idx.view(bsz, 1, 1).expand(-1, 1, hidden_dim)
            context = H_surr.gather(1, gather_idx).squeeze(1)

        return {
            "logits": torch.cat(logits_steps, dim=1),
            "ranking": torch.cat(ranking_steps, dim=1),
            "H_surr": H_surr,
            "H_true": H_true,
        }


class SimplifiedDeepIC(_DeepICBase):
    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 8,
        ff_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__(hidden_dim=hidden_dim, n_heads=n_heads, ff_dim=ff_dim, dropout=dropout)
        self.actor_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim + 1),
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.critic_head = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim + 1),
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _ensure_embedding_batch(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == 2 else x

    def _prepare_progress_for_candidates(
        self,
        H_surr: torch.Tensor,
        progress: torch.Tensor,
    ) -> torch.Tensor:
        H_surr = self._ensure_embedding_batch(H_surr)
        progress_prepared = self._prepare_progress(
            progress=progress,
            device=H_surr.device,
            dtype=H_surr.dtype,
            batch_size=H_surr.size(0),
        )
        # (B, 1) -> (B, N, 1) so each candidate sees the same budget feature.
        return progress_prepared.unsqueeze(1).expand(-1, H_surr.size(1), -1)

    def _actor_logits(
        self,
        H_surr: torch.Tensor,
        progress: torch.Tensor,
    ) -> torch.Tensor:
        H_surr = self._ensure_embedding_batch(H_surr)
        progress_feature = self._prepare_progress_for_candidates(H_surr, progress)
        # Concatenate candidate embedding and normalized budget progress: (B, N, h + 1).
        actor_input = torch.cat([H_surr, progress_feature], dim=-1)
        return self.actor_decoder(actor_input).squeeze(-1)

    def get_value(
        self,
        H_surr: torch.Tensor,
        H_true: torch.Tensor,
        progress: torch.Tensor,
    ) -> torch.Tensor:
        H_surr = self._ensure_embedding_batch(H_surr)
        H_true = self._ensure_embedding_batch(H_true)
        progress_prepared = self._prepare_progress(
            progress=progress,
            device=H_surr.device,
            dtype=H_surr.dtype,
            batch_size=H_surr.size(0),
        )
        surr_pool = H_surr.mean(dim=1)
        true_pool = H_true.mean(dim=1)
        critic_input = torch.cat([surr_pool, true_pool, progress_prepared], dim=-1)
        return self.critic_head(critic_input).squeeze(-1)

    def _sample_actions_without_replacement(
        self,
        logits: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n_sur = logits.shape
        selected_mask = torch.zeros(bsz, n_sur, dtype=torch.bool, device=logits.device)
        actions = []
        logprob = torch.zeros(bsz, dtype=logits.dtype, device=logits.device)
        entropy = torch.zeros_like(logprob)
        decode_steps = min(int(k), n_sur)

        for _ in range(decode_steps):
            masked_logits = logits.masked_fill(selected_mask, -1e9)
            dist = torch.distributions.Categorical(logits=masked_logits)
            chosen_idx = dist.sample()
            actions.append(chosen_idx.unsqueeze(1))
            logprob = logprob + dist.log_prob(chosen_idx)
            entropy = entropy + dist.entropy()
            selected_mask = selected_mask.scatter(1, chosen_idx.unsqueeze(1), True)

        return torch.cat(actions, dim=1), logprob, entropy

    def act(
        self,
        H_surr: torch.Tensor,
        H_true: torch.Tensor,
        progress: torch.Tensor,
        k: int = 1,
        decode_type: str = "sample",
    ) -> dict[str, torch.Tensor]:
        H_surr = self._ensure_embedding_batch(H_surr)
        H_true = self._ensure_embedding_batch(H_true)
        logits = self._actor_logits(H_surr, progress)
        value = self.get_value(H_surr, H_true, progress)
        decode_steps = min(int(k), logits.size(1))

        if decode_type == "greedy":
            actions = torch.topk(logits, k=decode_steps, dim=-1).indices
            eval_out = self.evaluate_actions(H_surr, H_true, progress, actions)
            logprob = eval_out["logprob"]
            entropy = eval_out["entropy"]
        elif decode_type == "sample":
            actions, logprob, entropy = self._sample_actions_without_replacement(logits, decode_steps)
        else:
            raise ValueError(f"Unsupported decode_type: {decode_type}")

        return {
            "actions": actions,
            "logprob": logprob,
            "entropy": entropy,
            "value": value,
            "logits": logits,
        }

    def evaluate_actions(
        self,
        H_surr: torch.Tensor,
        H_true: torch.Tensor,
        progress: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        H_surr = self._ensure_embedding_batch(H_surr)
        H_true = self._ensure_embedding_batch(H_true)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        logits = self._actor_logits(H_surr, progress)
        value = self.get_value(H_surr, H_true, progress)

        bsz, n_sur = logits.shape
        decode_steps = min(actions.size(1), n_sur)
        selected_mask = torch.zeros(bsz, n_sur, dtype=torch.bool, device=logits.device)
        logprob = torch.zeros(bsz, dtype=logits.dtype, device=logits.device)
        entropy = torch.zeros_like(logprob)

        for step in range(decode_steps):
            masked_logits = logits.masked_fill(selected_mask, -1e9)
            dist = torch.distributions.Categorical(logits=masked_logits)
            chosen_idx = actions[:, step]
            logprob = logprob + dist.log_prob(chosen_idx)
            entropy = entropy + dist.entropy()
            selected_mask = selected_mask.scatter(1, chosen_idx.unsqueeze(1), True)

        return {
            "logprob": logprob,
            "entropy": entropy,
            "value": value,
            "logits": logits,
        }

    def decode_ranking(
        self,
        H_surr: torch.Tensor,
        H_true: torch.Tensor,
        progress: torch.Tensor,
        target_ranking: torch.Tensor | None = None,
        decode_type: str = "greedy",
        max_decode_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        H_surr = self._ensure_embedding_batch(H_surr)
        H_true = self._ensure_embedding_batch(H_true)
        logits = self._actor_logits(H_surr, progress)

        if target_ranking is not None:
            ranking = target_ranking
        elif decode_type == "sample":
            decode_steps = logits.size(1) if max_decode_steps is None else min(int(max_decode_steps), logits.size(1))
            ranking, _, _ = self._sample_actions_without_replacement(logits, decode_steps)
        else:
            ranking = torch.argsort(logits, dim=-1, descending=True)
            if max_decode_steps is not None:
                ranking = ranking[:, : min(int(max_decode_steps), ranking.size(1))]

        if target_ranking is not None and max_decode_steps is not None:
            ranking = ranking[:, : min(int(max_decode_steps), ranking.size(1))]

        return {
            "logits": logits,
            "ranking": ranking,
            "H_surr": H_surr,
            "H_true": H_true,
        }


def ppo_loss(
    agent: SimplifiedDeepIC,
    batch: dict[str, torch.Tensor],
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> dict[str, torch.Tensor]:
    eval_out = agent.evaluate_actions(
        H_surr=batch["H_surr"],
        H_true=batch["H_true"],
        progress=batch["progress"],
        actions=batch["actions"],
    )

    new_logprob = eval_out["logprob"]
    entropy = eval_out["entropy"]
    value = eval_out["value"]
    old_logprob = batch["old_logprob"]
    advantages = batch["advantages"]
    returns = batch["returns"]

    ratio = torch.exp(new_logprob - old_logprob)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate_1 = ratio * advantages
    surrogate_2 = clipped_ratio * advantages
    actor_loss = -torch.mean(torch.minimum(surrogate_1, surrogate_2))
    critic_loss = F.mse_loss(value, returns)
    entropy_loss = -torch.mean(entropy)
    total_loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss
    approx_kl = torch.mean(old_logprob - new_logprob)

    return {
        "total_loss": total_loss,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "entropy_loss": entropy_loss,
        "approx_kl": approx_kl,
    }


class HV_DeepIC(DeepIC):
    """DeepIC variant that measures archive progress with hypervolume improvement."""

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
        self.reward_ref_point = None if reward_ref_point is None else np.asarray(reward_ref_point, dtype=np.float32)
        self.reward_epsilon = float(reward_epsilon)

    def set_reward_reference_point(self, ref_point) -> None:
        self.reward_ref_point = np.asarray(ref_point, dtype=np.float32)

    @staticmethod
    def hypervolume(values: np.ndarray, ref_point: np.ndarray) -> float:
        front = DeepIC.pareto_front(values)
        if front.size == 0:
            return 0.0
        return float(HV(ref_point=np.asarray(ref_point, dtype=np.float32))(front))

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
        if next_hv <= prev_hv:
            return -1.0
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
                raise ValueError("HV_DeepIC requires a hypervolume reference point for reward computation.")
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
