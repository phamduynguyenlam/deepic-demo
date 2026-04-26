from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from .deepic_agent import _DeepICBase


class ICW(_DeepICBase):
    """State-based Gaussian policy built on DeepIC-style encoders.

    The encoder keeps the same per-space preprocessing as DeepIC, but it does
    not apply cross-attention between surrogate and true spaces. Instead it
    mean-pools each space into a single embedding in R^h, concatenates those
    two embeddings with normalized progress t / FE_max, and feeds the resulting
    state vector in R^(2h + 1) to a Gaussian policy head.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 8,
        ff_dim: int = 256,
        dropout: float = 0.0,
        action_dim: int = 5,
        action_logit_clip: float | None = 2.0,
        softmax_temperature: float = 2.0,
        logit_reg_coef: float = 1e-3,
    ):
        super().__init__(hidden_dim=hidden_dim, n_heads=n_heads, ff_dim=ff_dim, dropout=dropout)
        self.action_dim = int(action_dim)
        state_dim = 2 * hidden_dim + 1
        trunk_dim = hidden_dim // 2
        self.action_logit_clip = action_logit_clip
        self.softmax_temperature = float(softmax_temperature)
        self.logit_reg_coef = float(logit_reg_coef)

        self.shared = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, trunk_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(trunk_dim, self.action_dim)
        self.logstd_head = nn.Linear(trunk_dim, self.action_dim)
        self.critic_head = nn.Linear(trunk_dim, 1)

    @staticmethod
    def _ensure_state_batch(state: torch.Tensor) -> torch.Tensor:
        return state.unsqueeze(0) if state.dim() == 1 else state

    @staticmethod
    def _ensure_action_batch(action: torch.Tensor) -> torch.Tensor:
        return action.unsqueeze(0) if action.dim() == 1 else action

    def process_action_logits(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_logits = action

        if self.action_logit_clip is not None:
            logits = torch.clamp(
                raw_logits,
                -float(self.action_logit_clip),
                float(self.action_logit_clip),
            )
        else:
            logits = raw_logits

        temperature = max(float(self.softmax_temperature), 1e-6)
        weights = torch.softmax(logits / temperature, dim=-1)

        logit_reg = torch.mean(raw_logits.pow(2))

        return {
            "raw_logits": raw_logits,
            "logits": logits,
            "weights": weights,
            "action_logits": logits,
            "action_weights": weights,
            "logit_reg": logit_reg,
        }

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
            x_true=x_true,
            y_true=y_true,
            x_sur=x_sur,
            y_sur=y_sur,
            sigma_sur=sigma_sur,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        e_true = self.W_true(m_true)
        e_surr = self.W_surr(m_surr)

        s_true = self.encoder_true(e_true)
        s_surr = self.encoder_surr(e_surr)

        h_true = s_true.mean(dim=1)
        h_surr = s_surr.mean(dim=1)
        progress_tensor = self._prepare_progress(
            progress=progress,
            device=h_true.device,
            dtype=h_true.dtype,
            batch_size=h_true.size(0),
        )

        pooled_true = h_true.mean(dim=1)
        pooled_surr = h_surr.mean(dim=1)
        state = torch.cat([pooled_surr, pooled_true, progress_tensor], dim=-1)

        return {
            "H_true": h_true,
            "H_surr": h_surr,
            "progress": progress_tensor,
            "state": state,
        }

    def decode_state(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self._ensure_state_batch(state)
        features = self.shared(state)
        mu = self.mu_head(features)
        logstd = self.logstd_head(features).clamp(-5.0, 1.0)
        std = torch.exp(logstd)
        value = self.critic_head(features).squeeze(-1)
        return {
            "state": state,
            "features": features,
            "mu": mu,
            "logstd": logstd,
            "std": std,
            "value": value,
        }

    def get_action_dist(self, state: torch.Tensor) -> Independent:
        decoded = self.decode_state(state)
        return Independent(Normal(decoded["mu"], decoded["std"]), 1)

    def act_from_state(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        decoded = self.decode_state(state)
        dist = Independent(Normal(decoded["mu"], decoded["std"]), 1)
        action = decoded["mu"] if deterministic else dist.rsample()
        processed = self.process_action_logits(action)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        decoded.update(
            {
                "action": action,
                "action_logits": processed["logits"],
                "action_weights": processed["weights"],
                "logit_reg": processed["logit_reg"],
                "logprob": logprob,
                "entropy": entropy,
            }
        )
        return decoded

    def evaluate_action_from_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        decoded = self.decode_state(state)
        action = self._ensure_action_batch(action).to(device=decoded["mu"].device, dtype=decoded["mu"].dtype)
        processed = self.process_action_logits(action)
        dist = Independent(Normal(decoded["mu"], decoded["std"]), 1)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        decoded.update(
            {
                "action": action,
                "action_logits": processed["logits"],
                "action_weights": processed["weights"],
                "logit_reg": processed["logit_reg"],
                "logprob": logprob,
                "entropy": entropy,
            }
        )
        return decoded

    def get_value_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.decode_state(state)["value"]

    def act(
        self,
        x_true: torch.Tensor,
        y_true: torch.Tensor,
        x_sur: torch.Tensor,
        y_sur: torch.Tensor,
        sigma_sur: torch.Tensor,
        progress,
        lower_bound,
        upper_bound,
        deterministic: bool = False,
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
        acted = self.act_from_state(encoded["state"], deterministic=deterministic)
        encoded.update(acted)
        return encoded

    def evaluate_actions(
        self,
        x_true: torch.Tensor,
        y_true: torch.Tensor,
        x_sur: torch.Tensor,
        y_sur: torch.Tensor,
        sigma_sur: torch.Tensor,
        progress,
        lower_bound,
        upper_bound,
        action: torch.Tensor,
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
        eval_out = self.evaluate_action_from_state(encoded["state"], action)
        encoded.update(eval_out)
        return encoded

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
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self.act(
            x_true=x_true,
            y_true=y_true,
            x_sur=x_sur,
            y_sur=y_sur,
            sigma_sur=sigma_sur,
            progress=progress,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            deterministic=deterministic,
        )
