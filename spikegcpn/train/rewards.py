"""
Terminal reward for a finished topology.

    r = accuracy - efficiency_coeff * (num_hidden + num_edges)

GRPO uses r directly (it is normalised within each group of samples).
The GFlowNet needs a positive reward and uses log R = beta * r, i.e. it samples
topologies with probability ∝ exp(beta * r); larger beta = greedier.

There are no per-step rewards any more: every action is legal by construction
(exact masks), so the old validity bonus / invalid penalty never fired, and
the forced stop at max_edges replaces the truncation penalty.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    efficiency_coeff: float = 0.0005
    beta: float = 100.0


def shaped_reward(accuracy: float, num_hidden: int, num_edges: int, cfg: RewardConfig) -> float:
    return accuracy - cfg.efficiency_coeff * (num_hidden + num_edges)


def log_reward(accuracy: float, num_hidden: int, num_edges: int, cfg: RewardConfig) -> float:
    return cfg.beta * shaped_reward(accuracy, num_hidden, num_edges, cfg)
