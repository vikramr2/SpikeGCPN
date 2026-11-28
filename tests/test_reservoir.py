"""Recurrent topologies (allow_cycles) and reservoir evaluation."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from spikegcpn.data.nmnist import NMNISTSample
from spikegcpn.env.snn_env import GraphSpec
from spikegcpn.env.validity import is_dag
from spikegcpn.simulator.torch_lif import LIFBatch
from spikegcpn.simulator.wrapper import sink_spike_counts, window_spike_counts
from spikegcpn.train.readout import ReadoutConfig, ReservoirEvaluator, input_readout_baselines

from test_core import random_state

CYC = GraphSpec(num_source=18, num_sink=3, max_hidden=6, max_edges=30, allow_cycles=True,
                rf_radii=[0, 1], weight_levels=[0.5, 1.0, 2.0])
EXTRA = 6


def _cyclic_graphs(n: int, steps: int = 30):
    return [random_state(CYC, random.Random(seed), steps=steps).graph for seed in range(n)]


def test_cycles_actually_occur():
    assert any(not is_dag(g.edge_index, g.num_nodes) for g in _cyclic_graphs(20))


def test_fixed_reservoir_matches_superneuromat():
    graphs = _cyclic_graphs(8)
    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.random((6, 25, CYC.num_source)) < 0.3)
    samples = [NMNISTSample(spike_train=X[i].numpy(), label=0) for i in range(len(X))]
    net = LIFBatch(graphs, CYC.num_source, CYC.num_sink, extra_ticks=EXTRA, fixed=True)
    with torch.no_grad():
        counts, win = net(X, windows=4)
    total = 0
    for gi, (g, sl) in enumerate(zip(graphs, net.ns_slices)):
        np.testing.assert_array_equal(
            sink_spike_counts(g, samples, CYC.num_source, CYC.num_sink, EXTRA), counts[:, gi].numpy().astype(int))
        snm = window_spike_counts(g, samples, CYC.num_source, CYC.num_sink, EXTRA, 4)
        np.testing.assert_array_equal(snm, win[:, sl].numpy().astype(int))
        total += int(snm.sum())
    assert total > 0


def _evaluator():
    rng = np.random.default_rng(1)
    tr = (torch.from_numpy(rng.random((60, 20, CYC.num_source)) < 0.3), torch.from_numpy(rng.integers(0, 3, 60)))
    va = (torch.from_numpy(rng.random((30, 20, CYC.num_source)) < 0.3), torch.from_numpy(rng.integers(0, 3, 30)))
    return ReservoirEvaluator(CYC.num_source, CYC.num_sink, tr, va, ReadoutConfig(windows=3, max_iter=50), EXTRA), tr, va


def test_reservoir_eval_deterministic_and_batch_independent():
    graphs = _cyclic_graphs(4)
    ev, _, _ = _evaluator()
    accs_b, _ = ev.train_and_eval(graphs)
    assert accs_b == ev.train_and_eval(graphs)[0]
    for g, a in zip(graphs, accs_b):
        assert ev.train_and_eval([g])[0][0] == pytest.approx(a, abs=1e-12)


def test_input_baselines_run():
    ev, tr, va = _evaluator()
    out = input_readout_baselines(tr, va, 3, ReadoutConfig(windows=3, max_iter=50), matched_neurons=5)
    assert 0 <= out["input_readout_all"] <= 1 and out["matched_channels"] == 5
    assert 0 <= out["input_readout_top"] <= 1


# ---------------------------------------------------------------------------
# Latent neurons, feedforward-only ablation, outputs-only readout
# ---------------------------------------------------------------------------

LAT = GraphSpec(num_source=18, num_sink=4, max_hidden=6, max_edges=30, allow_cycles=True,
                rf_radii=[-1, 0, 1], weight_levels=[0.5, 1.0])


def test_latent_neuron_has_no_input():
    from spikegcpn.env.snn_env import NEW, STOP_ACTION, SNNEnv
    env = SNNEnv(LAT)
    env.reset()
    g, _ = env.step(dict(STOP_ACTION, a_stop=0, a_kind=NEW, a_cell=4, a_radius=0, a_pol=2, a_node=0, a_edge=0))
    assert env.num_hidden == 1 and g.edge_index.shape[1] == 0
    assert int(LAT.rf_edge_table()[:, 0].max()) == 0


@pytest.mark.parametrize("seed", range(10))
def test_feedforward_only_never_wires_hidden_to_hidden(seed):
    spec = GraphSpec(num_source=18, num_sink=3, max_hidden=6, max_edges=30, allow_cycles=True,
                     hidden_to_hidden=False, rf_radii=[-1, 0, 1], weight_levels=[1.0])
    g = random_state(spec, random.Random(seed), steps=30).graph
    src, dst = g.edge_index
    assert not bool(((src >= spec.hidden_start) & (dst >= spec.hidden_start)).any())
    assert bool((dst[src >= spec.hidden_start] < spec.hidden_start).all())


def test_outputs_readout_sees_only_sinks():
    graphs = [random_state(LAT, random.Random(s), steps=25).graph for s in range(3)]
    rng = np.random.default_rng(2)
    X = torch.from_numpy(rng.random((12, 20, LAT.num_source)) < 0.3)
    y = torch.from_numpy(rng.integers(0, 3, 12))
    cfg_all, cfg_out = ReadoutConfig(windows=3, max_iter=20), ReadoutConfig(windows=3, max_iter=20, readout_neurons="outputs")
    f_all = ReservoirEvaluator(LAT.num_source, LAT.num_sink, (X, y), (X, y), cfg_all, EXTRA, 3).features(graphs, X)
    f_out = ReservoirEvaluator(LAT.num_source, LAT.num_sink, (X, y), (X, y), cfg_out, EXTRA, 3).features(graphs, X)
    for a, o in zip(f_all, f_out):
        assert o.shape == (12, LAT.num_sink * 3)
        assert torch.equal(o, a[:, :LAT.num_sink * 3])       # sinks come first, neuron-major
    snm = window_spike_counts(graphs[0], [NMNISTSample(spike_train=X[i].numpy(), label=0) for i in range(12)],
                              LAT.num_source, LAT.num_sink, EXTRA, 3)
    assert np.array_equal(snm[:, :LAT.num_sink].reshape(12, -1), f_out[0].numpy().astype(int))
    assert sum(float(o.sum()) for o in f_out) > 0, "vacuous if no output neuron ever fires"


def test_f_score_ranks_the_informative_channel_first():
    from spikegcpn.train.readout import channel_f_scores
    rng = np.random.default_rng(0)
    y = torch.from_numpy(rng.integers(0, 3, 90))
    X = torch.from_numpy(rng.random((90, 10, 5)) < 0.2)
    X[:, :, 3] = torch.arange(10).view(1, -1) < (3 * y.view(-1, 1) + 1)   # count grows with the label
    assert int(channel_f_scores(X, y, 3).argmax()) == 3
