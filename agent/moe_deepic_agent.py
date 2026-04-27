from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.deepic_agent import SimplifiedDeepIC


class StateWiseMoEDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_experts: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.temperature = temperature

        candidate_input_dim = hidden_dim + 1
        state_input_dim = hidden_dim + 1

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(candidate_input_dim),
                nn.Linear(candidate_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(n_experts)
        ])

        self.gate = nn.Sequential(
            nn.LayerNorm(state_input_dim),
            nn.Linear(state_input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_experts),
        )

    def forward(
        self,
        H_surr: torch.Tensor,
        progress_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # H_surr: (B, N, h)
        # progress_feature: (B, N, 1)

        B, N, _ = H_surr.shape

        candidate_input = torch.cat(
            [H_surr, progress_feature],
            dim=-1,
        )  # (B, N, h+1)

        expert_logits = torch.stack(
            [expert(candidate_input).squeeze(-1) for expert in self.experts],
            dim=-1,
        )  # (B, N, E)

        progress_state = progress_feature[:, 0, :]  # (B, 1)
        surr_state = H_surr.mean(dim=1)             # (B, h)

        gate_input = torch.cat(
            [surr_state, progress_state],
            dim=-1,
        )  # (B, h+1)

        gate_logits = self.gate(gate_input)  # (B, E)
        temperature = max(float(self.temperature), 1e-6)
        gate_weights = torch.softmax(
            gate_logits / temperature,
            dim=-1,
        )  # (B, E)

        logits = torch.einsum(
            "bne,be->bn",
            expert_logits,
            gate_weights,
        )  # (B, N)

        return {
            "logits": logits,
            "expert_logits": expert_logits,
            "gate_logits": gate_logits,
            "gate_weights": gate_weights,
        }


class MoE_SimplifiedDeepIC(SimplifiedDeepIC):
    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 8,
        ff_dim: int = 256,
        dropout: float = 0.0,
        n_experts: int = 4,
        temperature: float = 1.0,
    ):
        super().__init__(hidden_dim=hidden_dim, n_heads=n_heads, ff_dim=ff_dim, dropout=dropout)
        self.actor_decoder = StateWiseMoEDecoder(
            hidden_dim=hidden_dim,
            n_experts=n_experts,
            dropout=dropout,
            temperature=temperature,
        )

    def _actor_logits_full(
        self,
        H_surr: torch.Tensor,
        progress: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        H_surr = self._ensure_embedding_batch(H_surr)
        progress_feature = self._prepare_progress_for_candidates(H_surr, progress)
        return self.actor_decoder(H_surr, progress_feature)

    def _actor_logits(
        self,
        H_surr: torch.Tensor,
        progress: torch.Tensor,
    ) -> torch.Tensor:
        return self._actor_logits_full(H_surr, progress)["logits"]

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
        dec_out = self._actor_logits_full(H_surr, progress)
        logits = dec_out["logits"]
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
            "gate_weights": dec_out["gate_weights"],
            "expert_logits": dec_out["expert_logits"],
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

        dec_out = self._actor_logits_full(H_surr, progress)
        logits = dec_out["logits"]
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
            "gate_weights": dec_out["gate_weights"],
            "expert_logits": dec_out["expert_logits"],
        }


def moe_balance_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    if gate_weights.dim() == 3:
        usage = gate_weights.mean(dim=(0, 1))
    elif gate_weights.dim() == 2:
        usage = gate_weights.mean(dim=0)
    else:
        raise ValueError(f"Unexpected gate_weights shape: {tuple(gate_weights.shape)}")
    target = torch.full_like(usage, 1.0 / gate_weights.size(-1))
    return F.mse_loss(usage, target)


def moe_ppo_loss(
    agent: MoE_SimplifiedDeepIC,
    batch: dict[str, torch.Tensor],
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    balance_coef: float = 0.001,
) -> dict[str, torch.Tensor]:
    # Gọi hàm ppo_loss chuẩn trước để tái sử dụng logic tính toán
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

    gate_weights = eval_out["gate_weights"]
    balance_loss = moe_balance_loss(gate_weights)
    gate_entropy = -(gate_weights * (gate_weights + 1e-8).log()).sum(dim=-1).mean()
    total_loss = total_loss + balance_coef * balance_loss

    return {
        "total_loss": total_loss,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "entropy_loss": entropy_loss,
        "approx_kl": approx_kl,
        "ratio_mean": torch.mean(ratio),
        "ratio_std": (
            torch.std(ratio, unbiased=False)
            if ratio.numel() > 1
            else torch.zeros((), device=ratio.device, dtype=ratio.dtype)
        ),
        "ratio_min": torch.min(ratio),
        "ratio_max": torch.max(ratio),
        "balance_loss": balance_loss,
        "gate_entropy": gate_entropy,
    }
