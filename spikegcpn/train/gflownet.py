"""
GFlowNet with the trajectory-balance objective (Malkin et al., 2022).

For a complete trajectory τ = (s_0 → … → s_T → stop) ending in topology x:

    L(τ) = ( log Z + Σ_t log P_F(a_t|s_t) − log R(x) − Σ_t log P_B(s_t|s_{t+1}) )²

* P_F is the autoregressive policy (masks are exact, so log P_F is the log-prob
  of the actual transition).
* P_B is uniform over a state's (parent, action) pairs (validity.num_parents),
  so the policy is not rewarded for the arbitrary order of construction steps.
* log R = beta · (accuracy − efficiency penalty), see rewards.py.
* log Z is learned (own learning rate) and initialised from the first batch:
  with ~2300 candidate tails per step, log P_F of a 50-step trajectory is in
  the hundreds, so starting log Z at 0 would take thousands of steps to catch up.

Note: states are *labelled* graphs (hidden neurons are numbered by creation
order), so the sampler targets P(labelled x) ∝ R(x); an unlabelled topology
with more distinct labellings is sampled proportionally more often.

TB is off-policy: trajectories can come from a tempered / ε-mixed sampler and
from a replay buffer of high-reward graphs, which lets each expensive
simulator evaluation be learned from many times.
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
class TBConfig:
    lr: float = 3e-4
    logz_lr: float = 0.1
    max_grad_norm: float = 10.0
    temperature: float = 1.0
    explore_eps: float = 0.05
    replay_size: int = 256
    replay_batch: int = 8
    chunk_trajs: int = 8         # trajectories re-scored per batched forward (memory bound)


class ReplayBuffer:
    """Keeps the top-`capacity` unique topologies by their reward at insertion time."""

    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.items: dict[str, tuple[float, Trajectory]] = {}
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.items)

    def add(self, traj: Trajectory, log_r: float) -> None:
        if self.capacity <= 0:
            return
        if traj.key in self.items:
            return
        self.items[traj.key] = (log_r, traj)
        if len(self.items) > self.capacity:
            worst = min(self.items, key=lambda k: self.items[k][0])
            del self.items[worst]

    def sample(self, k: int) -> list[Trajectory]:
        k = min(k, len(self.items))
        return [t for _, t in self.rng.sample(list(self.items.values()), k)]


class TBTrainer:
    def __init__(self, encoder: GraphEncoder, policy: PolicyHead, cfg: TBConfig, seed: int = 0):
        self.encoder = encoder
        self.policy = policy
        self.cfg = cfg
        self.log_z = nn.Parameter(torch.zeros(()))
        self._logz_initialised = False
        self.params = list(encoder.parameters()) + list(policy.parameters())
        self.optimizer = torch.optim.Adam([
            {"params": self.params, "lr": cfg.lr},
            {"params": [self.log_z], "lr": cfg.logz_lr},
        ])
        self.replay = ReplayBuffer(cfg.replay_size, seed)

    def _sum_log_pf(self, trajs: list[Trajectory]) -> torch.Tensor:
        return torch.stack([lp.sum() for lp, _ in rescore_batch(trajs, self.encoder, self.policy)])

    def update(self, trajs: list[Trajectory], log_rewards: list[float]) -> dict:
        log_r = torch.tensor(log_rewards, dtype=torch.float32)
        log_pb = torch.tensor([t.log_pb for t in trajs], dtype=torch.float32)
        n, k = len(trajs), self.cfg.chunk_trajs

        if not self._logz_initialised:
            with torch.no_grad():
                lpf = torch.cat([self._sum_log_pf(trajs[i:i + k]) for i in range(0, n, k)])
                self.log_z.fill_(float((log_r + log_pb - lpf).mean()))
            self._logz_initialised = True

        # The loss is a sum over trajectories, so gradients can be accumulated chunk by chunk.
        self.optimizer.zero_grad()
        total, lpf_all = 0.0, []
        for i in range(0, n, k):
            lpf = self._sum_log_pf(trajs[i:i + k])
            delta = self.log_z + lpf - log_r[i:i + k] - log_pb[i:i + k]
            loss = (delta ** 2).sum() / n
            loss.backward()
            total += loss.item()
            lpf_all.append(lpf.detach())
        grad_norm = nn.utils.clip_grad_norm_(self.params, self.cfg.max_grad_norm)
        self.optimizer.step()
        return {
            "loss": total,
            "log_z": float(self.log_z),
            "mean_log_pf": float(torch.cat(lpf_all).mean()),
            "grad_norm": float(grad_norm),
        }

    def state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "policy": self.policy.state_dict(),
            "log_z": self.log_z.detach().clone(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.encoder.load_state_dict(sd["encoder"])
        self.policy.load_state_dict(sd["policy"])
        with torch.no_grad():
            self.log_z.copy_(sd["log_z"])
        self._logz_initialised = True
        self.optimizer.load_state_dict(sd["optimizer"])
