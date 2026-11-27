"""
Invariant tests for the macro-action environment, masks, GFlowNet parent
counting, batched policy re-scoring and the simulator wrapper.
Run with:  .venv/bin/python -m pytest -q
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from spikegcpn.data.nmnist import NMNISTSample
from spikegcpn.env.snn_env import CONNECT, NEW, NUM_POL_CHOICES, STOP_ACTION, GraphSpec, SNNEnv, prefix_graph
from spikegcpn.env.validity import is_dag, num_parents
from spikegcpn.model.encoder import GraphEncoder
from spikegcpn.model.policy import PolicyHead
from spikegcpn.train.gflownet import TBConfig, TBTrainer
from spikegcpn.train.grpo import GRPOConfig, GRPOTrainer
from spikegcpn.train.rollout import random_action, rescore_batch, sample_trajectories

# 18 sources = 2 polarities × 3×3 patch grid
SMALL = dict(num_source=18, num_sink=2, max_hidden=4, max_edges=24, rf_radii=[0, 1])


def random_state(spec: GraphSpec, rng: random.Random, steps: int) -> SNNEnv:
    env = SNNEnv(spec)
    env.reset()
    for _ in range(steps):
        a = random_action(env, rng)
        if a["a_stop"]:
            break
        env.step(a)
    return env


def _edge_set(g: Data):
    return sorted(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist(), g.edge_attr[:, 0].tolist()))


def _same_graph(a: Data, b: Data) -> bool:
    return a.num_nodes == b.num_nodes and torch.equal(a.x, b.x) and _edge_set(a) == _edge_set(b)


# ---------------------------------------------------------------------------
# Environment and masks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cycles", [False, True])
@pytest.mark.parametrize("seed", range(15))
def test_random_rollouts_stay_valid(seed, cycles):
    spec = GraphSpec(**SMALL, allow_cycles=cycles)
    rng = random.Random(seed)
    env = SNNEnv(spec)
    env.reset()
    while True:
        g, done = env.step(random_action(env, rng, p_stop=0.05))
        if done:
            break
    S, K = spec.num_source, spec.num_sink
    pairs = list(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
    assert len(pairs) == len(set(pairs)), "duplicate edge"
    assert all(u != v for u, v in pairs), "self-loop"
    assert all(v >= S for _, v in pairs), "edge into a source"
    assert all(not (S <= u < S + K) for u, _ in pairs), "edge out of a sink"
    assert all(v >= S + K for u, v in pairs if u < S), "source edges only feed hidden neurons (receptive fields)"
    assert g.edge_index.shape[1] <= spec.max_edges and env.num_hidden <= spec.max_hidden
    if not cycles:
        assert is_dag(g.edge_index, g.num_nodes)
    for j in range(S + K, g.num_nodes):    # every hidden neuron has a non-empty receptive field
        assert any(u < S for u, v in pairs if v == j)


def test_receptive_fields():
    spec = GraphSpec(num_source=18, num_sink=2, rf_radii=[0, 1, 2])
    assert spec.grid == 3
    assert spec.rf_sources(4, 0, 0) == [4]                      # centre cell, OFF
    assert spec.rf_sources(4, 0, 2) == [4, 13]                  # both polarities
    assert len(spec.rf_sources(4, 1, 1)) == 9                   # full 3×3, ON
    assert len(spec.rf_sources(0, 1, 2)) == 8                   # corner, clipped 2×2 × 2
    # the full-grid OFF field: centre at radius 1, or any of the 9 cells at radius 2
    assert spec.rf_multiplicity(tuple(spec.rf_sources(4, 1, 0))) == 10
    t = spec.rf_edge_table()
    assert t.shape == (9, 3, NUM_POL_CHOICES) and int(t[4, 1, 2]) == 18


def test_budget_masks_receptive_fields():
    spec = GraphSpec(**SMALL)
    env = SNNEnv(spec)
    env.reset()
    env.step(dict(STOP_ACTION, a_stop=0, a_kind=NEW, a_cell=4, a_radius=1, a_pol=2, a_node=0, a_edge=0))  # 18 edges
    assert env.masks().remaining == 6
    with pytest.raises(ValueError):    # a full 3×3 × 2 field (18 edges) no longer fits
        env.step(dict(STOP_ACTION, a_stop=0, a_kind=NEW, a_cell=4, a_radius=1, a_pol=2, a_node=0, a_edge=0))


def test_stop_forced_at_edge_budget_and_min_edges():
    spec = GraphSpec(**SMALL, min_edges=5)
    env = SNNEnv(spec)
    env.reset()
    with pytest.raises(ValueError):
        env.step({"a_stop": 1})
    env = random_state(spec, random.Random(0), steps=1000)
    assert not env.masks().continue_allowed and env.masks().stop_allowed


def test_prefix_graph_matches_intermediate_states():
    spec = GraphSpec(**SMALL, allow_cycles=True)
    rng = random.Random(3)
    env = SNNEnv(spec)
    env.reset()
    snaps = []
    while True:
        snaps.append((env.graph, env.num_hidden, env.num_edges))
        g, done = env.step(random_action(env, rng, p_stop=0.05))
        if done:
            break
    for gt, h, e in snaps:
        p = prefix_graph(g, e, h, spec)
        assert torch.equal(p.x, gt.x) and torch.equal(p.edge_index, gt.edge_index)


# ---------------------------------------------------------------------------
# GFlowNet backward policy: num_parents vs. brute force through env.step
# ---------------------------------------------------------------------------

def _brute_force_parents(spec: GraphSpec, g: Data, num_hidden: int) -> int:
    """Count (parent, action) pairs whose env.step reproduces g exactly."""
    hid0 = spec.hidden_start
    E = g.edge_index.shape[1]
    count = 0
    L = len(spec.weight_levels)

    def bin_of(w):
        b = min(range(L), key=lambda i: abs(spec.weight_levels[i] - abs(w)))
        return b if w > 0 else b + L

    def try_step(parent: Data, h: int, action: dict) -> bool:
        env = SNNEnv(spec)
        env.reset()
        env._graph, env.num_hidden = parent, h
        try:
            child, _ = env.step(action)
        except ValueError:
            return False
        return _same_graph(child, g)

    # (i) the last action added one synapse between existing neurons
    for k in range(E):
        keep = [i for i in range(E) if i != k]
        parent = Data(x=g.x, edge_index=g.edge_index[:, keep], edge_attr=g.edge_attr[keep])
        u, v = int(g.edge_index[0, k]), int(g.edge_index[1, k])
        a = dict(STOP_ACTION, a_stop=0, a_kind=CONNECT, a_first=u, a_second=v, a_edge=bin_of(float(g.edge_attr[k, 0])))
        count += try_step(parent, num_hidden, a)

    # (ii) the last action created the most recent hidden neuron (any NEW choice)
    if num_hidden > 0:
        last = hid0 + num_hidden - 1
        touching = (g.edge_index[0] == last) | (g.edge_index[1] == last)
        keep = (~touching).nonzero(as_tuple=True)[0]
        parent = Data(x=g.x[:last], edge_index=g.edge_index[:, keep], edge_attr=g.edge_attr[keep])
        for c in range(spec.num_cells):
            for r in range(len(spec.rf_radii)):
                for p in range(NUM_POL_CHOICES):
                    for node in range(spec.num_presets):
                        for b in range(spec.num_edge_actions):
                            a = dict(STOP_ACTION, a_stop=0, a_kind=NEW, a_cell=c, a_radius=r, a_pol=p, a_node=node, a_edge=b)
                            count += try_step(parent, num_hidden - 1, a)
    return count


@pytest.mark.parametrize("cycles", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_num_parents_matches_brute_force(seed, cycles):
    spec = GraphSpec(**SMALL, allow_cycles=cycles, weight_levels=[0.5, 1.0],
                     hidden_presets=[(0.5, float("inf")), (1.0, 0.0)])
    rng = random.Random(seed)
    env = random_state(spec, rng, steps=rng.randint(1, 12))
    assert num_parents(env.graph.edge_index, spec, env.num_hidden) == \
        _brute_force_parents(spec, env.graph, env.num_hidden)


# ---------------------------------------------------------------------------
# Policy / trainers
# ---------------------------------------------------------------------------

POLICY_SPEC = dict(num_source=50, num_sink=3, max_hidden=6, max_edges=40, rf_radii=[0, 1, 2], allow_cycles=True)


def _models(spec):
    torch.manual_seed(0)
    return GraphEncoder(spec.num_sink, 16, 2), PolicyHead(spec, 16, 32)


def test_rescore_matches_sampling_log_probs():
    spec = GraphSpec(**POLICY_SPEC)
    enc, pol = _models(spec)
    trajs = sample_trajectories(spec, enc, pol, 6)
    assert any(a["a_kind"] == CONNECT for t in trajs for a in t.actions)
    assert any(a["a_kind"] == NEW for t in trajs for a in t.actions)
    for tr, (lp, _) in zip(trajs, rescore_batch(trajs, enc, pol)):
        assert torch.allclose(lp, torch.tensor(tr.old_log_probs), atol=1e-4)


def test_batched_sampling_produces_legal_episodes():
    spec = GraphSpec(**POLICY_SPEC, min_edges=10)
    enc, pol = _models(spec)
    for tr in sample_trajectories(spec, enc, pol, 8, eps=0.2):
        env = SNNEnv(spec)
        env.reset()
        for a in tr.actions:          # replaying through env.step re-checks every mask
            g, done = env.step(a)
        assert done and _same_graph(g, tr.final) and tr.num_edges >= spec.min_edges


def test_sources_and_sinks_are_distinguishable_at_reset():
    spec = GraphSpec(**POLICY_SPEC)
    enc, _ = _models(spec)
    g = SNNEnv(spec).reset()
    H = enc(g.x, g.edge_index, g.edge_attr)
    S = spec.num_source
    assert torch.unique(H[:S].round(decimals=5), dim=0).shape[0] == S
    assert torch.unique(H[S:].round(decimals=5), dim=0).shape[0] == spec.num_sink


@pytest.mark.parametrize("algo", ["grpo", "gflownet"])
def test_trainer_update_changes_parameters(algo):
    spec = GraphSpec(**POLICY_SPEC)
    enc, pol = _models(spec)
    trajs = sample_trajectories(spec, enc, pol, 4)
    before = [p.detach().clone() for p in pol.parameters()]
    if algo == "grpo":
        stats = GRPOTrainer(enc, pol, GRPOConfig(epochs=1, minibatch_trajs=2)).update(trajs, [0.1, 0.2, 0.3, 0.4])
    else:
        stats = TBTrainer(enc, pol, TBConfig(chunk_trajs=3)).update(trajs, [1.0, 2.0, 3.0, 4.0])
    assert np.isfinite(stats["loss"])
    assert any(not torch.equal(a, b) for a, b in zip(before, pol.parameters()))


# ---------------------------------------------------------------------------
# Simulator wrapper: pruning isolated sources must not change accuracy
# ---------------------------------------------------------------------------

def _naive_accuracy(g: Data, samples, S: int, K: int, extra: int) -> float:
    from superneuromat import SNN
    snn = SNN()
    neurons = [snn.create_neuron(threshold=float(g.x[i, 0]), leak=float(g.x[i, 1]),
                                 reset_state=float(g.x[i, 2]), refractory_period=int(g.x[i, 3]))
               for i in range(g.num_nodes)]
    for (s, d), (w, delay) in zip(g.edge_index.t().tolist(), g.edge_attr.tolist()):
        snn.create_synapse(neurons[s], neurons[d], weight=w, delay=int(delay))
    credit = 0.0
    for smp in samples:
        snn.reset()
        T = smp.spike_train.shape[0]
        for t in range(T):
            for i in np.flatnonzero(smp.spike_train[t]):
                snn.add_spike(t, neurons[i], 1.0)
        snn.simulate(T + extra, use="cpu")
        counts = snn.ispikes[:, [neurons[S + c].idx for c in range(K)]].sum(0)
        winners = counts == counts.max()
        credit += winners[smp.label] / winners.sum()
    return credit / len(samples)


@pytest.mark.parametrize("seed", range(5))
def test_pruned_simulation_matches_naive(seed):
    from spikegcpn.env.validity import longest_path_length
    from spikegcpn.simulator.wrapper import evaluate_samples

    spec = GraphSpec(num_source=18, num_sink=3, max_hidden=4, max_edges=20, rf_radii=[0])
    env = random_state(spec, random.Random(seed), steps=20)
    g = env.graph
    nrng = np.random.default_rng(seed)
    samples = [NMNISTSample(spike_train=nrng.random((20, spec.num_source)) < 0.3, label=int(nrng.integers(3)))
               for _ in range(6)]
    fast = evaluate_samples(g, samples, spec.num_source, spec.num_sink)
    slow = _naive_accuracy(g, samples, spec.num_source, spec.num_sink, longest_path_length(g.edge_index, g.num_nodes))
    assert fast == pytest.approx(slow)
