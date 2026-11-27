"""
Terminal reward for a finished topology.

    r = accuracy - efficiency_coeff * (num_hidden + num_edges)

GRPO normalises r within each group of samples, so only differences matter.
There are no per-step rewards: every action is legal by construction (exact
masks), and the forced stop at max_edges replaces a truncation penalty.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    efficiency_coeff: float = 0.0005


def shaped_reward(accuracy: float, num_hidden: int, num_edges: int, cfg: RewardConfig) -> float:
    return accuracy - cfg.efficiency_coeff * (num_hidden + num_edges)
