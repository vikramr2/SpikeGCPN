"""
Reward functions for SpikeGCPN.

Reward design follows the GCPN paper (Section 3.3) adapted to SNNs:

  Step reward (intermediate):
    +VALIDITY_STEP  if the action was legal and the graph remains a valid DAG
    -INVALID_STEP   if the action was rejected by the environment

  Terminal reward (on a_stop=1):
    r_accuracy    — N-MNIST accuracy returned by SuperNeuroMAT (in [0, 1])
    r_efficiency  — penalty proportional to hidden neuron count and edge count

  Total terminal reward:
    R = r_accuracy - efficiency_coeff * (num_hidden + num_edges)

The advantage estimate used in PPO is:
    Â_t = r_t + γ V_ω(s_{t+1}) - V_ω(s_t)
which is generalised to GAE(λ) in ppo.py.
"""

from __future__ import annotations

from dataclasses import dataclass

VALIDITY_STEP: float = 0.001
INVALID_STEP: float = -0.01
EFFICIENCY_COEFF: float = 0.0005
TRUNCATION_PENALTY: float = 0.5  # applied once when episode hits max_steps


@dataclass
class RewardConfig:
    validity_step: float = VALIDITY_STEP
    invalid_step: float = INVALID_STEP
    efficiency_coeff: float = EFFICIENCY_COEFF
    truncation_penalty: float = TRUNCATION_PENALTY


def step_reward(valid: bool, cfg: RewardConfig | None = None) -> float:
    """
    Intermediate per-step reward.

    Parameters
    ----------
    valid : True if the action was accepted by the environment,
            False if it was rejected (cycle, out-of-range, max_hidden).
    """
    if cfg is None:
        cfg = RewardConfig()
    return cfg.validity_step if valid else cfg.invalid_step


def terminal_reward(
    accuracy: float,
    num_hidden: int,
    num_edges: int,
    cfg: RewardConfig | None = None,
) -> float:
    """
    Reward given at episode termination (a_stop=1).

    Parameters
    ----------
    accuracy   : classification accuracy in [0, 1] from SuperNeuroMAT
    num_hidden : number of hidden neurons in the final topology
    num_edges  : number of synaptic connections in the final topology
    """
    if cfg is None:
        cfg = RewardConfig()
    r_accuracy = accuracy
    r_efficiency = -cfg.efficiency_coeff * (num_hidden + num_edges)
    return r_accuracy + r_efficiency
