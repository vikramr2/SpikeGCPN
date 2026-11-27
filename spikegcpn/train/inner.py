"""
Inner loop: train the weights (and optionally thresholds / leaks) of a batch of
fixed topologies with surrogate-gradient BPTT, then report validation accuracy.

The topology policy is the outer loop; its reward for a topology is the
validation accuracy this trainer reaches. Training is deterministic given the
seed: every graph gets the same init jitter stream and the same minibatch
sequence, and graphs trained together do not interact (see torch_lif.LIFBatch),
so the reward is a function of the topology alone and can be cached.

Validation accuracy is computed by the float64 torch simulator, whose forward
pass matches SuperNeuroMAT exactly (tests/test_inner.py); `verify` re-runs
trained graphs in SuperNeuroMAT itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

from spikegcpn.data.nmnist import NMNISTSample
from spikegcpn.simulator.torch_lif import LIFBatch


@dataclass
class InnerConfig:
    steps: int = 100
    batch_size: int = 64
    lr: float = 0.1
    train_neuron_params: bool = True
    slope: float = 5.0
    weight_jitter: float = 0.1
    leak_cap: float = 1e3          # learned leaks are finite; presets with leak=inf start here
    seed: int = 0
    max_graphs_per_batch: int = 16


def stack_samples(samples: tuple[NMNISTSample, ...] | list[NMNISTSample]) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.from_numpy(np.stack([s.spike_train for s in samples]))      # bool [N, T, S]
    y = torch.tensor([s.label for s in samples], dtype=torch.long)
    return X, y


def split_per_class(samples, n_first: int):
    """Split a class-ordered sample list into (first n_first per class, rest)."""
    first, rest, seen = [], [], {}
    for s in samples:
        k = seen.get(s.label, 0)
        (first if k < n_first else rest).append(s)
        seen[s.label] = k + 1
    return first, rest


def tie_credit(counts: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """counts [N, G, K] → per-graph accuracy [G] with 1/#tied credit."""
    winners = counts == counts.max(dim=-1, keepdim=True).values
    hit = winners.gather(-1, labels.view(-1, 1, 1).expand(-1, counts.shape[1], 1)).squeeze(-1)
    return (hit.double() / winners.sum(-1)).mean(0)


class InnerTrainer:
    def __init__(self, num_source: int, num_sink: int, train_set, val_set, cfg: InnerConfig):
        self.S, self.K = num_source, num_sink
        self.Xtr, self.ytr = train_set
        self.Xva, self.yva = val_set
        self.cfg = cfg
        self.graphs_trained = 0

    def _train(self, graphs: list[Data]) -> LIFBatch:
        cfg = self.cfg
        net = LIFBatch(graphs, self.S, self.K, cfg.train_neuron_params, cfg.leak_cap,
                       weight_jitter=cfg.weight_jitter, seed=cfg.seed, slope=cfg.slope)
        params = [p for p in net.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=cfg.lr)
        gen = torch.Generator().manual_seed(cfg.seed)
        N = len(self.ytr)
        for _ in range(cfg.steps):
            idx = torch.randint(N, (cfg.batch_size,), generator=gen)
            loss = net.loss(net(self.Xtr[idx]), self.ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.graphs_trained += len(graphs)
        return net

    @torch.no_grad()
    def _accuracy(self, net: LIFBatch, X: torch.Tensor, y: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        counts = torch.cat([net(X[i:i + chunk]) for i in range(0, len(y), chunk)])
        return tie_credit(counts, y)

    def train_and_eval(self, graphs: list[Data]) -> tuple[list[float], list[Data]]:
        """Validation accuracy and trained graphs for each topology."""
        accs: list[float] = []
        trained: list[Data] = []
        step = self.cfg.max_graphs_per_batch
        for i in range(0, len(graphs), step):
            chunk = graphs[i:i + step]
            net = self._train(chunk)
            accs += self._accuracy(net, self.Xva, self.yva).tolist()
            trained += net.export(chunk)
        return accs, trained
