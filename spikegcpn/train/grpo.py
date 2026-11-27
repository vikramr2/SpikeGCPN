"""
GRPO-style critic-free policy optimisation.

Each update uses a *group* of G topologies sampled from the same initial
state. With a single terminal reward per trajectory, a learned per-step value
function (PPO) has almost nothing to regress on; instead every step of
trajectory i shares the group-normalised advantage

    A_i = (r_i - mean(r)) / (std(r) + eps)

and the clipped surrogate objective is applied per step:

    L = - mean_t min( ρ_t A_i, clip(ρ_t, 1-ε, 1+ε) A_i ) - c_H · mean_t H_t
    ρ_t = π_θ(a_t|s_t) / π_old(a_t|s_t)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from spikegcpn.model.encoder import GraphEncoder
from spikegcpn.model.policy import PolicyHead
from spikegcpn.train.rollout import Trajectory, rescore_batch


@dataclass
class GRPOConfig:
    lr: float = 3e-4
    clip_eps: float = 0.2
    epochs: int = 2
    minibatch_trajs: int = 4
    entropy_coeff: float = 0.001
    max_grad_norm: float = 1.0
    adv_eps: float = 1e-6


class GRPOTrainer:
    def __init__(self, encoder: GraphEncoder, policy: PolicyHead, cfg: GRPOConfig, seed: int = 0):
        self.encoder = encoder
        self.policy = policy
        self.cfg = cfg
        self.params = list(encoder.parameters()) + list(policy.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=cfg.lr)
        self.rng = random.Random(seed)

    def update(self, trajs: list[Trajectory], rewards: list[float]) -> dict:
        cfg = self.cfg
        r = torch.tensor(rewards, dtype=torch.float32)
        if len(r) < 2 or r.std() < 1e-8:
            return {"loss": 0.0, "clip_frac": 0.0, "approx_kl": 0.0, "entropy": 0.0, "skipped": 1}
        adv = (r - r.mean()) / (r.std() + cfg.adv_eps)

        tot_loss = tot_clip = tot_kl = tot_ent = 0.0
        n_mb = 0
        idx = list(range(len(trajs)))
        for _ in range(cfg.epochs):
            self.rng.shuffle(idx)
            for s in range(0, len(idx), cfg.minibatch_trajs):
                mb = idx[s:s + cfg.minibatch_trajs]
                terms, steps = [], 0
                clipped = kl = ent_sum = 0.0
                scored = rescore_batch([trajs[i] for i in mb], self.encoder, self.policy)
                for i, (lp, ent) in zip(mb, scored):
                    old = torch.tensor(trajs[i].old_log_probs)
                    ratio = torch.exp(lp - old)
                    a = adv[i]
                    surr = torch.min(ratio * a, ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * a)
                    terms.append(-surr.sum() - cfg.entropy_coeff * ent.sum())
                    steps += len(lp)
                    with torch.no_grad():
                        clipped += float(((ratio - 1).abs() > cfg.clip_eps).sum())
                        kl += float((old - lp).sum())
                        ent_sum += float(ent.sum())
                loss = torch.stack(terms).sum() / steps
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
                self.optimizer.step()

                tot_loss += loss.item()
                tot_clip += clipped / steps
                tot_kl += kl / steps
                tot_ent += ent_sum / steps
                n_mb += 1

        return {
            "loss": tot_loss / n_mb,
            "clip_frac": tot_clip / n_mb,
            "approx_kl": tot_kl / n_mb,
            "entropy": tot_ent / n_mb,
            "skipped": 0,
        }

    def state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.encoder.load_state_dict(sd["encoder"])
        self.policy.load_state_dict(sd["policy"])
        self.optimizer.load_state_dict(sd["optimizer"])
