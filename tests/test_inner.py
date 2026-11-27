"""
Tests for the differentiable simulator, inner-loop weight training, patch
pooling and the min_edges stop mask.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from spikegcpn.data.nmnist import NMNISTSample, grid_size, load_sample, num_source_nodes, source_positions
from spikegcpn.env.snn_env import GraphSpec, SNNEnv
from spikegcpn.model.encoder import GraphEncoder
from spikegcpn.model.policy import PolicyHead
from spikegcpn.simulator.torch_lif import LIFBatch
from spikegcpn.simulator.wrapper import sink_spike_counts
from spikegcpn.train.inner import InnerConfig, InnerTrainer
from spikegcpn.train.rollout import sample_trajectories

from test_core import random_state

SPEC = GraphSpec(num_source=18, num_sink=3, max_hidden=5, max_edges=25, rf_radii=[0, 1], weight_levels=[1.0])


def _graphs(n: int, steps: int = 25, refractory: bool = False):
    out = []
    for seed in range(n):
        g = random_state(SPEC, random.Random(seed), steps=steps).graph
        if refractory:
            g.x[SPEC.num_source:, 3] = torch.randint(0, 3, (g.num_nodes - SPEC.num_source,)).float()
        out.append(g)
    return out


def _inputs(n=8, T=30, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.random((n, T, SPEC.num_source)) < 0.3)


@pytest.mark.parametrize("refractory", [False, True])
def test_torch_forward_matches_superneuromat(refractory):
    graphs = _graphs(6, refractory=refractory)
    net = LIFBatch(graphs, SPEC.num_source, SPEC.num_sink, leak_cap=float("inf"), weight_jitter=0.5, seed=1)
    with torch.no_grad():
        net.thr_raw.add_(0.3 * torch.randn_like(net.thr_raw))
        net.leak_raw.add_(0.3 * torch.randn_like(net.leak_raw))
    X = _inputs()
    with torch.no_grad():
        c_torch = net(X).numpy().astype(np.int64)
    samples = [NMNISTSample(spike_train=X[i].numpy(), label=0) for i in range(len(X))]
    total = 0
    for gi, tg in enumerate(net.export(graphs)):
        c_snm = sink_spike_counts(tg, samples, SPEC.num_source, SPEC.num_sink)
        np.testing.assert_array_equal(c_snm, c_torch[:, gi])
        total += int(c_snm.sum())
    assert total > 0, "test is vacuous if no sink ever spikes"


def _trainer(steps=5):
    rng = np.random.default_rng(0)
    Xtr = torch.from_numpy(rng.random((40, 20, SPEC.num_source)) < 0.3)
    ytr = torch.from_numpy(rng.integers(0, SPEC.num_sink, 40))
    Xva = torch.from_numpy(rng.random((20, 20, SPEC.num_source)) < 0.3)
    yva = torch.from_numpy(rng.integers(0, SPEC.num_sink, 20))
    return InnerTrainer(SPEC.num_source, SPEC.num_sink, (Xtr, ytr), (Xva, yva),
                        InnerConfig(steps=steps, batch_size=8))


def test_batched_training_equals_individual_training():
    graphs = _graphs(3)
    accs_b, trained_b = _trainer().train_and_eval(graphs)
    for g, acc, tb in zip(graphs, accs_b, trained_b):
        (acc_1,), (t1,) = _trainer().train_and_eval([g])
        assert acc == pytest.approx(acc_1, abs=1e-12)
        assert torch.allclose(t1.edge_attr, tb.edge_attr, atol=1e-10)
        assert torch.allclose(t1.x, tb.x, atol=1e-10)


def test_training_changes_parameters_and_keeps_signs():
    (g,) = _graphs(1)
    _, (t,) = _trainer(steps=10).train_and_eval([g])
    assert not torch.allclose(t.edge_attr[:, 0], g.edge_attr[:, 0].double())
    assert torch.equal(torch.sign(t.edge_attr[:, 0]), torch.sign(g.edge_attr[:, 0].double())), "Dale's law violated"
    S = SPEC.num_source
    assert torch.equal(t.x[:S], g.x[:S].double()), "source neurons must be untouched"


def test_edgeless_graph_scores_chance():
    spec_graph = SNNEnv(SPEC).reset()
    (acc,), _ = _trainer().train_and_eval([spec_graph])
    assert acc == pytest.approx(1.0 / SPEC.num_sink)


# ---------------------------------------------------------------------------
# Patches and stop mask
# ---------------------------------------------------------------------------

def test_patch_counts_and_positions():
    assert num_source_nodes(1) == 2312 and num_source_nodes(4) == 162 and grid_size(4) == 9
    pos = source_positions(162, patch=4)
    assert pos.shape == (162, 3) and pos[:, :2].min() == 0 and pos[:, :2].max() == 1
    assert set(pos[:, 2].tolist()) == {0.0, 1.0}


def test_patch_pooling_is_or_of_pixels(tmp_path):
    # Build a synthetic .bin file: events (x, y, p, ts) with 5-byte records.
    rng = np.random.default_rng(0)
    n = 300
    x, y, p = rng.integers(0, 34, n), rng.integers(0, 34, n), rng.integers(0, 2, n)
    ts = np.sort(rng.integers(0, 60000, n))
    raw = np.stack([x, y, p, ts >> 8, ts & 0xFF], axis=1).astype(np.uint8)
    f = tmp_path / "s.bin"
    f.write_bytes(raw.tobytes())
    pix = load_sample(str(f), 10, patch=1)
    pat = load_sample(str(f), 10, patch=4)
    G = grid_size(4)
    for idx in range(num_source_nodes(4)):
        pol, rem = divmod(idx, G * G)
        gy, gx = divmod(rem, G)
        cols = [pol * 34 * 34 + yy * 34 + xx
                for yy in range(gy * 4, min(gy * 4 + 4, 34)) for xx in range(gx * 4, min(gx * 4 + 4, 34))]
        np.testing.assert_array_equal(pat[:, idx], pix[:, cols].any(axis=1))


def test_min_edges_masks_stop():
    spec = GraphSpec(num_source=18, num_sink=3, max_hidden=4, max_edges=12, min_edges=6, weight_levels=[1.0])
    torch.manual_seed(0)
    enc, pol = GraphEncoder(spec.num_sink, 16, 2), PolicyHead(spec, 16, 32)
    with torch.no_grad():                                   # make stopping very attractive
        pol.m_stop[-1].bias.copy_(torch.tensor([0.0, 10.0]))
    for tr in sample_trajectories(spec, enc, pol, 5):
        assert tr.num_edges >= spec.min_edges
    env = SNNEnv(spec)
    env.reset()
    with pytest.raises(ValueError):
        env.step({"a_stop": 1})
