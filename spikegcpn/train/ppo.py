"""
PPO trainer for SpikeGCPN.

Implements the clipped surrogate objective (GCPN Section 3.5 / Eq. 5):

  L^CLIP(θ) = E_t [ min( r_t(θ) Â_t,  clip(r_t(θ), 1-ε, 1+ε) Â_t ) ]

  r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)

Advantages are estimated with GAE(λ) (generalised advantage estimation).
Setting λ=1 recovers the 1-step TD form from idea.md.

One episode produces one rollout.  For each step in the rollout we store a
graph snapshot (detached PyG Data clone) so that the policy can be
re-evaluated with updated weights during K PPO epochs.

Total loss per step:
  L = L_policy + c_v * L_value - c_e * L_entropy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

if TYPE_CHECKING:
    from spikegcpn.model.encoder import GCNEncoder
    from spikegcpn.model.policy import PolicyHead
    from spikegcpn.model.value import ValueFunction


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """One (s_t, a_t, r_t, V_t, done_t) tuple from a rollout."""
    graph: Data        # detached snapshot of the graph before this action
    action: dict       # {a_first, a_second, a_edge, a_stop}
    log_prob: float    # joint log π_{θ_old}(a_t | s_t)
    reward: float
    value: float       # V_ω(s_t) at collection time
    done: bool
    first_mask: "torch.Tensor | None" = None   # valid a_first nodes
    second_mask: "torch.Tensor | None" = None  # valid a_second nodes


class RolloutBuffer:
    def __init__(self) -> None:
        self.transitions: list[Transition] = []

    def push(self, t: Transition) -> None:
        self.transitions.append(t)

    def clear(self) -> None:
        self.transitions = []

    def __len__(self) -> int:
        return len(self.transitions)


def _clone_graph(data: Data) -> Data:
    """Return a fully detached copy of a PyG Data graph."""
    return Data(
        x=data.x.detach().clone(),
        edge_index=data.edge_index.detach().clone(),
        edge_attr=data.edge_attr.detach().clone() if data.edge_attr is not None else None,
    )


# ---------------------------------------------------------------------------
# PPO trainer
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ppo_epochs: int = 4
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5


class PPOTrainer:
    """
    Trains the encoder, policy, and value function jointly using PPO.

    Usage
    -----
    trainer = PPOTrainer(encoder, policy, value_fn, cfg)
    # ... collect transitions into a RolloutBuffer ...
    stats = trainer.update(buffer)
    buffer.clear()
    """

    def __init__(
        self,
        encoder: "GCNEncoder",
        policy: "PolicyHead",
        value_fn: "ValueFunction",
        cfg: PPOConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.policy = policy
        self.value_fn = value_fn
        self.cfg = cfg or PPOConfig()

        params = (
            list(encoder.parameters())
            + list(policy.parameters())
            + list(value_fn.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=self.cfg.lr)

    # ------------------------------------------------------------------
    # Advantage estimation
    # ------------------------------------------------------------------

    def _compute_gae(
        self, buffer: RolloutBuffer, last_value: float = 0.0
    ) -> tuple[list[float], list[float]]:
        """
        GAE(λ) advantage and discounted-return estimates.

        Â_t = δ_t + (γλ) δ_{t+1} + ...
        δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        """
        cfg = self.cfg
        n = len(buffer)
        advantages: list[float] = [0.0] * n
        returns: list[float] = [0.0] * n

        gae = 0.0
        for i in reversed(range(n)):
            t = buffer.transitions[i]
            next_val = buffer.transitions[i + 1].value if i + 1 < n else last_value
            if t.done:
                next_val = 0.0
            delta = t.reward + cfg.gamma * next_val - t.value
            gae = delta + cfg.gamma * cfg.gae_lambda * (0.0 if t.done else gae)
            advantages[i] = gae
            returns[i] = gae + t.value

        return advantages, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, buffer: RolloutBuffer) -> dict:
        """
        Run K epochs of the clipped PPO objective over the stored rollout.

        Returns a dict of mean losses for logging.
        """
        cfg = self.cfg
        advantages, returns = self._compute_gae(buffer)

        adv = torch.tensor(advantages, dtype=torch.float32)
        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = torch.tensor(returns, dtype=torch.float32)

        total_pl = total_vl = total_ent = 0.0
        count = 0

        for _ in range(cfg.ppo_epochs):
            for i, trans in enumerate(buffer.transitions):
                g = trans.graph
                H = self.encoder(g.x, g.edge_index, g.edge_attr)

                # --- policy loss ---
                new_lp = self.policy.evaluate_log_probs(
                    H, trans.action,
                    first_mask=trans.first_mask,
                    second_mask=trans.second_mask,
                )
                new_lp_total = new_lp.total()
                old_lp_total = torch.tensor(trans.log_prob, dtype=torch.float32)

                ratio = torch.exp(new_lp_total - old_lp_total)
                a_i = adv[i]
                surr1 = ratio * a_i
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * a_i
                policy_loss = -torch.min(surr1, surr2)

                # --- value loss ---
                value_pred = self.value_fn(H)
                value_loss = F.mse_loss(value_pred, ret[i])

                # --- entropy bonus ---
                entropy = self.policy.entropy(H)

                loss = policy_loss + cfg.value_coeff * value_loss - cfg.entropy_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                all_params = (
                    list(self.encoder.parameters())
                    + list(self.policy.parameters())
                    + list(self.value_fn.parameters())
                )
                nn.utils.clip_grad_norm_(all_params, cfg.max_grad_norm)
                self.optimizer.step()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy.item()
                count += 1

        denom = max(count, 1)
        return {
            "policy_loss": total_pl / denom,
            "value_loss": total_vl / denom,
            "entropy": total_ent / denom,
        }

    def save(self, path: str) -> None:
        torch.save({
            "encoder": self.encoder.state_dict(),
            "policy": self.policy.state_dict(),
            "value_fn": self.value_fn.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.encoder.load_state_dict(ckpt["encoder"])
        self.policy.load_state_dict(ckpt["policy"])
        self.value_fn.load_state_dict(ckpt["value_fn"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
