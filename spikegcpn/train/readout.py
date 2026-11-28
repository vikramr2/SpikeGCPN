"""
Reservoir evaluation: simulate a topology with its fixed (untrained) weights,
take spike counts in W time windows as features, and fit a linear readout:
multinomial logistic regression (L-BFGS, default) or closed-form ridge
regression onto one-hot labels. The reward is the readout's validation accuracy.

`readout_neurons` picks which neurons the readout sees:
  all      every sink + hidden neuron (hidden neurons with receptive fields can
           relay the input straight to the readout)
  outputs  only the sinks, which have no input of their own, so everything the
           readout gets has passed through the searched graph. The number of
           sinks (outputs) is then independent of the number of classes.

Both readouts are deterministic, so a topology's reward is a function of the
topology alone; the cost is one forward simulation of the train + val sets.

Baselines (`input_readout_baselines`): the same readout trained directly on
input-patch spike counts: all channels, `matched` random channels, and the
`matched` channels with the highest ANOVA F-statistic on the training split
(the fair one). A reservoir is only doing useful computation if it beats the
input readout with the same number of features.

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
    readout_neurons: str = "all"     # all | outputs (sinks only)
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
                 extra_ticks: int, num_classes: int | None = None):
        if cfg.readout_neurons not in ("all", "outputs"):
            raise ValueError(f"unknown readout_neurons {cfg.readout_neurons!r}")
        self.S, self.K = num_source, num_sink
        self.num_classes = num_classes or num_sink
        self.Xtr, self.ytr = train_set
        self.Xva, self.yva = val_set
        self.cfg = cfg
        self.extra_ticks = extra_ticks

    @torch.no_grad()
    def features(self, graphs: list[Data], X: torch.Tensor) -> list[torch.Tensor]:
        """Per graph: [N, M_g * windows] reservoir features (M_g = read-out neurons)."""
        net = LIFBatch(graphs, self.S, self.K, extra_ticks=self.extra_ticks, fixed=True)
        chunks = []
        for i in range(0, len(X), self.cfg.chunk):
            _, win = net(X[i:i + self.cfg.chunk], windows=self.cfg.windows)
            chunks.append(win)
        win = torch.cat(chunks)                                     # [N, total_ns, W]
        if self.cfg.readout_neurons == "outputs":          # sinks are the first K non-source neurons
            return [win[:, sl][:, :self.K].reshape(len(X), -1) for sl in net.ns_slices]
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
                r = fit_readout(a, self.ytr, self.num_classes, self.cfg)
                accs.append(float((r.predict(v) == self.yva).double().mean()))
                readouts.append(r)
        return accs, readouts


def input_readout_baselines(
    train_set, val_set, num_classes: int, cfg: ReadoutConfig, matched_neurons: int, seeds: int = 3,
) -> dict:
    """
    Readout on input-patch window counts: all channels; random subsets of
    `matched_neurons` channels (same feature count as the largest reservoir);
    and the `matched_neurons` channels with the highest one-way ANOVA F-statistic
    of their total count on the training split (no validation data used).
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
    top = channel_f_scores(Xtr, ytr, num_classes).argsort(descending=True)[:min(matched_neurons, C)]
    cols = (top.view(-1, 1) * W + torch.arange(W)).flatten()
    r = fit_readout(Ftr[:, cols], ytr, num_classes, cfg)
    out["input_readout_top"] = float((r.predict(Fva[:, cols]) == yva).double().mean())
    return out


def channel_f_scores(X: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-way ANOVA F-statistic of each channel's total spike count across classes. [C]"""
    tot = X.double().sum(1)                                          # [N, C]
    grand = tot.mean(0)
    between = torch.zeros_like(grand)
    within = torch.zeros_like(grand)
    for c in range(num_classes):
        t = tot[y == c]
        if len(t) == 0:
            continue
        between += len(t) * (t.mean(0) - grand) ** 2
        within += ((t - t.mean(0)) ** 2).sum(0)
    k, n = num_classes, len(tot)
    return (between / max(1, k - 1)) / (within / max(1, n - k) + 1e-12)
