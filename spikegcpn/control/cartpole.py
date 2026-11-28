"""
Vectorised CartPole-v1 (identical dynamics, termination and reset to
gymnasium's implementation; tests/test_control.py checks it step for step), a
full-state expert, and the population-code encoder for the masked observation.

Partial observability: the agent sees only cart position x and pole angle θ;
the velocities ẋ and θ̇ are hidden, so any policy that balances the pole has
to infer them from how x and θ change over time, i.e. it needs memory.
"""

from __future__ import annotations

import numpy as np

GRAVITY, MASSCART, MASSPOLE, LENGTH, FORCE, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
TOTAL_MASS = MASSCART + MASSPOLE
POLEMASS_LENGTH = MASSPOLE * LENGTH
X_LIMIT, THETA_LIMIT = 2.4, 12 * 2 * np.pi / 360
MAX_STEPS = 500
OBS_INDEX = (0, 2)                     # visible state components: x, θ
OBS_RANGE = (X_LIMIT, THETA_LIMIT)


def reset(seeds: np.ndarray) -> np.ndarray:
    """[N, 4] initial states, uniform(−0.05, 0.05) as in gymnasium, one RNG per seed."""
    return np.stack([np.random.default_rng(int(s)).uniform(-0.05, 0.05, 4) for s in seeds])


def step(state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Euler step for [..., 4] states and {0, 1} actions → (next state, terminated)."""
    x, x_dot, th, th_dot = (state[..., i] for i in range(4))
    force = np.where(action == 1, FORCE, -FORCE)
    c, s = np.cos(th), np.sin(th)
    temp = (force + POLEMASS_LENGTH * th_dot ** 2 * s) / TOTAL_MASS
    th_acc = (GRAVITY * s - c * temp) / (LENGTH * (4.0 / 3.0 - MASSPOLE * c ** 2 / TOTAL_MASS))
    x_acc = temp - POLEMASS_LENGTH * th_acc * c / TOTAL_MASS
    nxt = np.stack([x + TAU * x_dot, x_dot + TAU * x_acc, th + TAU * th_dot, th_dot + TAU * th_acc], axis=-1)
    done = (np.abs(nxt[..., 0]) > X_LIMIT) | (np.abs(nxt[..., 2]) > THETA_LIMIT)
    return nxt, done


def expert(state: np.ndarray) -> np.ndarray:
    """Full-state linear controller (sees the velocities): push toward where the pole is falling."""
    x, x_dot, th, th_dot = (state[..., i] for i in range(4))
    return (0.1 * x + 0.5 * x_dot + 10.0 * th + 1.5 * th_dot > 0).astype(np.int64)


def population_centres(bins: int) -> np.ndarray:
    """[bins] centres in (−1, 1), denser near 0 (u·|u| spacing), where balancing happens."""
    u = np.linspace(-1.0, 1.0, bins)
    return u * np.abs(u)


def encode(state: np.ndarray, bins: int) -> np.ndarray:
    """
    [..., 4] states → bool [..., 2·bins] sources (x bins, then θ bins): a source
    is active when its Gaussian tuning curve, with width = local centre spacing,
    exceeds 0.5. Usually one or two adjacent sources per variable are active.
    """
    c = population_centres(bins)
    width = np.gradient(c)
    out = []
    for idx, rng in zip(OBS_INDEX, OBS_RANGE):
        v = np.clip(state[..., idx] / rng, -1.0, 1.0)[..., None]
        out.append(np.exp(-0.5 * ((v - c) / width) ** 2) > 0.5)
    return np.concatenate(out, axis=-1)
