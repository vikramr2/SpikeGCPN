"""CartPole dynamics, the stepwise simulator, and the closed-loop control evaluator."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from spikegcpn.control import cartpole as cp
from spikegcpn.control.evaluator import ControlConfig, ControlEvaluator
from spikegcpn.env.snn_env import GraphSpec, SNNEnv
from spikegcpn.simulator.torch_lif import LIFBatch
from spikegcpn.train.rollout import sample_random_trajectory

SPEC = GraphSpec(input_layout="line1d", line_groups=2, num_source=14, num_sink=3, max_hidden=8, max_edges=40,
                 allow_cycles=True, rf_radii=[-1, 0, 1], weight_levels=[0.5, 1.0, 2.0])
SMALL = ControlConfig(ticks_per_step=2, max_steps=60, bc_episodes=4, eval_episodes=3, test_episodes=3, max_iter=30)


def _graphs(n):
    env, rng = SNNEnv(SPEC), random.Random(0)
    return [sample_random_trajectory(env, rng).final for _ in range(n)]


def test_dynamics_match_gymnasium():
    gym = pytest.importorskip("gymnasium")
    env = gym.make("CartPole-v1").unwrapped
    rng = np.random.default_rng(0)
    for ep in range(3):
        obs, _ = env.reset(seed=ep)
        for _ in range(100):
            a = int(rng.integers(2))
            o, _, term, _, _ = env.step(a)
            s, done = cp.step(np.asarray(obs, dtype=np.float64), np.array(a))
            assert np.allclose(s, o, atol=1e-6) and bool(done) == term
            obs = o
            if term:
                break


def test_expert_balances():
    seeds = np.arange(20)
    s, alive, ret = cp.reset(seeds), np.ones(20, bool), np.zeros(20)
    for _ in range(cp.MAX_STEPS):
        s2, d = cp.step(s, cp.expert(s))
        ret += alive
        alive &= ~d
        s = np.where(alive[:, None], s2, s)
    assert ret.min() == cp.MAX_STEPS


def test_line_groups_fields():
    assert SPEC.num_cells == 7 and SPEC.num_pol_choices == 3
    assert SPEC.rf_sources(3, 2, 0) == [2, 3, 4]
    assert SPEC.rf_sources(3, 2, 1) == [9, 10, 11]
    assert SPEC.rf_sources(0, 2, 2) == [0, 1, 7, 8]


def test_step_reproduces_forward_with_per_graph_inputs():
    graphs = _graphs(4)
    X = torch.from_numpy(np.random.default_rng(0).random((2, 30, SPEC.num_source)) < 0.3)
    net = LIFBatch(graphs, SPEC.num_source, SPEC.num_sink, extra_ticks=0, fixed=True)
    _, win = net(X, windows=30)
    st = net.init_state(2)
    per_graph = X.unsqueeze(2).expand(-1, -1, len(graphs), -1)                    # same input, [B, T, G, S]
    stepped = torch.stack([net.step(st, per_graph[:, t]) for t in range(30)], 2)
    assert torch.equal(win, stepped) and win.sum() > 0


def test_control_reward_is_deterministic_and_batch_independent():
    graphs = _graphs(3)
    ev = ControlEvaluator(SPEC.num_source, SPEC.num_sink, SMALL)
    a, _ = ev.train_and_eval(graphs)
    assert a == ev.train_and_eval(graphs)[0]
    for g, x in zip(graphs, a):
        assert ev.train_and_eval([g])[0][0] == pytest.approx(x, abs=1e-12)
    assert all(0 < x <= 1 for x in a)


def test_closed_loop_matches_superneuromat():
    ev = ControlEvaluator(SPEC.num_source, SPEC.num_sink, ControlConfig(**{**SMALL.__dict__, "readout_neurons": "all"}))
    g = _graphs(1)[0]
    (ro,) = ev.fit([g])
    assert ev.verify_superneuromat(g, ro, episodes=2)["superneuromat_match"]
