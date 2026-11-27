"""
Reservoir evaluation: simulate a topology with its fixed (untrained) weights,
take every sink + hidden neuron's spike counts in W time windows as features,
and fit a linear readout: closed-form ridge regression onto one-hot labels
(default) or multinomial logistic regression (L-BFGS). The reward is the
readout's validation accuracy.

Both readouts are deterministic, so a topology's reward is a function of the
topology alone; the cost is one forward simulation of the train + val sets.

Baselines (`input_readout_baselines`): the same readout trained directly on
input-patch spike counts. A reservoir is only doing useful computation if it
beats the input readout with the same number of features.

The readout is off-chip: the reservoir runs on SuperNeuroMAT unchanged and the
readout is a linear layer over its binned spike counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from spikegcpn.simulator.torch_lif import LIFBatch


@dataclass
class ReadoutConfig:
    windows: int = 5
    readout: str = "logistic"        # logistic | ridge
    ridge_lambda: float = 10.0       # on standardised features
    l2: float = 1e-3                 # logistic only
    max_iter: int = 200              # logistic only
    chunk: int = 250                 # samples per simulation chunk (memory bound)
    max_graphs_per_batch: int = 16


@dataclass
class Readout:
    mu: torch.Tensor
    sd: torch.Tensor
    W: torch.Tensor
    b: torch.Tensor

    def predict(self, feats: torch.Tensor) -> torch.Tensor:
        return (((feats - self.mu) / self.sd) @ self.W + self.b).argmax(1)


def fit_readout(Ftr: torch.Tensor, ytr: torch.Tensor, num_classes: int, cfg: "ReadoutConfig") -> Readout:
    Ftr = Ftr.double()
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Z = (Ftr - mu) / sd
    if cfg.readout == "ridge":
        Y = F.one_hot(ytr, num_classes).double()
        b = Y.mean(0)
        A = Z.T @ Z + cfg.ridge_lambda * torch.eye(Z.shape[1], dtype=torch.float64)
        W = torch.linalg.solve(A, Z.T @ (Y - b))
        return Readout(mu, sd, W, b)
    if cfg.readout != "logistic":
        raise ValueError(f"unknown readout {cfg.readout!r}")
    l2, max_iter = cfg.l2, cfg.max_iter
    W = torch.zeros(Z.shape[1], num_classes, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(num_classes, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Z @ W + b, ytr) + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return Readout(mu, sd, W.detach(), b.detach())


def window_counts(X: torch.Tensor, windows: int, total_ticks: int | None = None) -> torch.Tensor:
    """[N, T, C] spike trains → [N, C*windows] counts per window (window-major)."""
    N, T, C = X.shape
    ticks = total_ticks or T
    widx = torch.arange(T) * windows // ticks
    out = torch.zeros(N, C, windows, dtype=torch.float64)
    out.index_add_(2, widx, X.double().permute(0, 2, 1))
    return out.reshape(N, -1)


class ReservoirEvaluator:
    def __init__(self, num_source: int, num_sink: int, train_set, val_set, cfg: ReadoutConfig,
                 extra_ticks: int):
        self.S, self.K = num_source, num_sink
        self.Xtr, self.ytr = train_set
        self.Xva, self.yva = val_set
        self.cfg = cfg
        self.extra_ticks = extra_ticks

    @torch.no_grad()
    def features(self, graphs: list[Data], X: torch.Tensor) -> list[torch.Tensor]:
        """Per graph: [N, M_g * windows] reservoir features."""
        net = LIFBatch(graphs, self.S, self.K, extra_ticks=self.extra_ticks, fixed=True)
        chunks = []
        for i in range(0, len(X), self.cfg.chunk):
            _, win = net(X[i:i + self.cfg.chunk], windows=self.cfg.windows)
            chunks.append(win)
        win = torch.cat(chunks)                                     # [N, total_ns, W]
        return [win[:, sl].reshape(len(X), -1) for sl in net.ns_slices]

    def train_and_eval(self, graphs: list[Data]) -> tuple[list[float], list[Readout]]:
        accs: list[float] = []
        readouts: list[Readout] = []
        step = self.cfg.max_graphs_per_batch
        for i in range(0, len(graphs), step):
            chunk = graphs[i:i + step]
            ftr = self.features(chunk, self.Xtr)
            fva = self.features(chunk, self.Xva)
            for a, v in zip(ftr, fva):
                r = fit_readout(a, self.ytr, self.K, self.cfg)
                accs.append(float((r.predict(v) == self.yva).double().mean()))
                readouts.append(r)
        return accs, readouts


def input_readout_baselines(
    train_set, val_set, num_classes: int, cfg: ReadoutConfig, matched_neurons: int, seeds: int = 3,
) -> dict:
    """
    Readout on input-patch window counts: all channels, and random subsets of
    `matched_neurons` channels (same feature count as the largest reservoir).
    """
    Xtr, ytr = train_set
    Xva, yva = val_set
    Ftr, Fva = window_counts(Xtr, cfg.windows), window_counts(Xva, cfg.windows)
    full = fit_readout(Ftr, ytr, num_classes, cfg)
    out = {"input_readout_all": float((full.predict(Fva) == yva).double().mean()),
           "input_channels": int(Xtr.shape[2])}
    C, W = Xtr.shape[2], cfg.windows
    accs = []
    for s in range(seeds):
        ch = torch.from_numpy(np.random.default_rng(s).choice(C, size=min(matched_neurons, C), replace=False))
        cols = (ch.view(-1, 1) * W + torch.arange(W)).flatten()
        r = fit_readout(Ftr[:, cols], ytr, num_classes, cfg)
        accs.append(float((r.predict(Fva[:, cols]) == yva).double().mean()))
    out["input_readout_matched"] = float(np.mean(accs))
    out["matched_channels"] = int(min(matched_neurons, C))
    return out
