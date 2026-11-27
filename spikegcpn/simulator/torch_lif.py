"""
Differentiable LIF simulator that reproduces SuperNeuroMAT's CPU update rule
exactly in the forward pass, with a surrogate gradient for spikes.

SuperNeuroMAT tick (neuromorphicmodel.simulate_cpu), per neuron:
    leak : v ← max(v − leak, reset) if v > reset ; min(v + leak, reset) if v < reset
    input: v ← v + external_input[t] + Σ_j w_ji · s_j(t−1)
    spike: s(t) = [v > threshold], suppressed while the refractory counter > 0
           (the counter decrements; spiking neurons reload it)
    reset: v ← reset for spiking neurons
Every synapse here has delay 1. Sources receive 1.0 input spikes; with
threshold < 1, reset 0 and no refractory period their spike train equals the
input train, so they are not simulated as neurons.

Several topologies can be simulated together as one disconnected network
(edge-list message passing, no edges across graphs). Adam is elementwise and
the summed loss separates per graph, so training a batch of graphs gives the
same parameters as training each alone with the same seed and minibatches.

Trainable per graph (Dale's law: synapse signs are fixed by the topology):
    synapse magnitudes  |w| = softplus(·)
    thresholds, leaks   of sinks and hidden neurons (softplus, optional)
    one readout scale   for the cross-entropy over sink spike counts (loss only)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from spikegcpn.env.validity import longest_path_length


class SpikeFn(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid (SuperSpike) derivative backward."""

    @staticmethod
    def forward(ctx, x: Tensor, slope: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: Tensor):
        (x,) = ctx.saved_tensors
        return grad / (1.0 + ctx.slope * x.abs()) ** 2, None


def inv_softplus(y: Tensor) -> Tensor:
    """softplus⁻¹ for y > 0 (inf ↦ inf)."""
    return y + torch.log(-torch.expm1(-y))


class LIFBatch(nn.Module):
    def __init__(
        self,
        graphs: list[Data],
        num_source: int,
        num_sink: int,
        train_neuron_params: bool = True,
        leak_cap: float = 1e3,
        min_param: float = 1e-3,
        weight_jitter: float = 0.1,
        seed: int = 0,
        slope: float = 5.0,
        extra_ticks: int | None = None,
        fixed: bool = False,
    ):
        """extra_ticks: ticks simulated after the last input bin. None = each graph's
        longest path (DAGs only); recurrent graphs need a fixed value.
        fixed: use the graph's weights / thresholds / leaks verbatim (no softplus
        re-parameterisation, no clamping, nothing trainable) — reservoir mode, and
        bit-identical to SuperNeuroMAT on the same graph."""
        super().__init__()
        S, K = num_source, num_sink
        self.num_source, self.num_sink, self.slope = S, K, slope
        self.num_graphs = len(graphs)

        kept_src, tails, heads, w_raw, signs = [], [], [], [], []
        thr, leak, reset, refrac, sink_pos, extra, edge_slices = [], [], [], [], [], [], []
        w_exact, thr_exact, leak_exact = [], [], []
        src_off = ns_off = e_off = 0
        # Tail indices are first assigned as ("src", local) / ("ns", local) and
        # resolved to positions in cat([all kept sources, all non-sources]) below.
        tail_kind: list[tuple[bool, Tensor]] = []
        for g in graphs:
            x = g.x.double()
            if not (bool((x[:S, 0] < 1.0).all()) and bool((x[:S, 2] == 0).all()) and bool((x[:S, 3] == 0).all())):
                raise NotImplementedError("sources must relay inputs: threshold < 1, reset 0, refractory 0")
            n = g.num_nodes
            M = n - S
            ei = g.edge_index
            t_, h_ = ei[0], ei[1]
            is_src = t_ < S
            used = torch.unique(t_[is_src]) if ei.numel() else torch.zeros(0, dtype=torch.long)
            local = torch.full((S,), -1, dtype=torch.long)
            local[used] = torch.arange(len(used))

            kept_src.append(used)
            tail_kind.append((True, local[t_[is_src]] + src_off))
            tail_kind.append((False, t_[~is_src] - S + ns_off))
            heads.append(torch.cat([h_[is_src], h_[~is_src]]) - S + ns_off)

            w0 = torch.cat([g.edge_attr[is_src, 0], g.edge_attr[~is_src, 0]]).double()
            gen = torch.Generator().manual_seed(seed)
            jit = (1.0 + weight_jitter * torch.randn(len(w0), generator=gen, dtype=torch.float64)).clamp(min=0.05)
            signs.append(torch.sign(w0))
            w_exact.append(w0)
            thr_exact.append(x[S:, 0])
            leak_exact.append(x[S:, 1])
            w_raw.append(inv_softplus((w0.abs() * jit).clamp(min=min_param)))
            # Map stored (src-edges, ns-edges) order back to the graph's edge order.
            order = torch.cat([is_src.nonzero(as_tuple=True)[0], (~is_src).nonzero(as_tuple=True)[0]])
            edge_slices.append((e_off, order))

            thr.append(inv_softplus(x[S:, 0].clamp(min=min_param)))
            leak.append(inv_softplus(x[S:, 1].clamp(min=min_param, max=leak_cap)))
            reset.append(x[S:, 2])
            refrac.append(x[S:, 3].long())
            sink_pos.append(torch.arange(K) + ns_off)
            extra.append(longest_path_length(ei, n) if extra_ticks is None else int(extra_ticks))

            src_off += len(used)
            ns_off += M
            e_off += len(w0)

        self.total_src, self.total_ns = src_off, ns_off
        self.register_buffer("kept_src", torch.cat(kept_src))
        self.register_buffer("tails", torch.cat([
            idx if is_s else idx + src_off for is_s, idx in tail_kind
        ]) if tail_kind else torch.zeros(0, dtype=torch.long))
        self.register_buffer("heads", torch.cat(heads))
        self.register_buffer("signs", torch.cat(signs))
        self.register_buffer("reset", torch.cat(reset))
        self.register_buffer("refrac", torch.cat(refrac))
        self.register_buffer("sink_pos", torch.cat(sink_pos))          # [G*K]
        self.register_buffer("extra", torch.tensor(extra))
        self.edge_slices = edge_slices
        self.ns_slices = []
        off = 0
        for g in graphs:
            self.ns_slices.append(slice(off, off + g.num_nodes - S))
            off += g.num_nodes - S
        self._src_counts = [len(k) for k in kept_src]

        self.fixed = fixed
        if fixed:
            self.register_buffer("w_fixed", torch.cat(w_exact))
            self.register_buffer("thr_fixed", torch.cat(thr_exact))
            self.register_buffer("leak_fixed", torch.cat(leak_exact))
        self.w_raw = nn.Parameter(torch.cat(w_raw), requires_grad=not fixed)
        self.thr_raw = nn.Parameter(torch.cat(thr), requires_grad=train_neuron_params and not fixed)
        self.leak_raw = nn.Parameter(torch.cat(leak), requires_grad=train_neuron_params and not fixed)
        self.log_scale = nn.Parameter(torch.full((self.num_graphs,), math.log(0.05), dtype=torch.float64))

    # ------------------------------------------------------------------
    def weights(self) -> Tensor:
        return self.w_fixed if self.fixed else self.signs * F.softplus(self.w_raw)

    def neuron_params(self) -> tuple[Tensor, Tensor]:
        if self.fixed:
            return self.thr_fixed, self.leak_fixed
        return F.softplus(self.thr_raw), F.softplus(self.leak_raw)

    def forward(self, x: Tensor, windows: int = 0):
        """
        x : [B, T, S] input spike trains (bool/float) over all S source channels.
        Returns sink spike counts [B, G, K] over T + extra(g) ticks per graph.
        With windows=W > 0 also returns per-neuron spike counts of every
        non-source neuron in W equal time windows over the T + max(extra) ticks,
        shape [B, total_ns, W] (graph g's neurons are the slice ns_slices[g]);
        reservoir features. Use a fixed extra_ticks so all graphs share the windows.
        """
        B, T, _ = x.shape
        G, K = self.num_graphs, self.num_sink
        xs = x[:, :, self.kept_src].to(torch.float64)
        w = self.weights()
        thr, leak = self.neuron_params()
        reset = self.reset
        has_refrac = bool((self.refrac > 0).any())

        v = reset.expand(B, -1).clone()
        s_ns = torch.zeros(B, self.total_ns, dtype=torch.float64)
        s_src = torch.zeros(B, self.total_src, dtype=torch.float64)
        refr = torch.zeros(B, self.total_ns, dtype=torch.long)
        counts = torch.zeros(B, G * K, dtype=torch.float64)
        stop_at = (T + self.extra).repeat_interleave(K)                # [G*K]
        total_ticks = T + int(self.extra.max())
        win = torch.zeros(B, self.total_ns, windows, dtype=torch.float64) if windows else None

        for t in range(total_ticks):
            d = v - reset
            v = reset + torch.sign(d) * F.relu(d.abs() - leak)
            if self.tails.numel():
                s_all = torch.cat([s_src, s_ns], dim=1)
                msg = s_all[:, self.tails] * w
                v = v + torch.zeros_like(v).index_add(1, self.heads, msg)
            s = SpikeFn.apply(v - thr, self.slope)
            if has_refrac:
                in_ref = refr > 0
                s = s * (~in_ref)
                refr = torch.where(in_ref, refr - 1, refr)
                refr = torch.where(s > 0, self.refrac.expand(B, -1), refr)
            sd = s.detach()
            v = v * (1.0 - sd) + reset * sd
            counts = counts + s[:, self.sink_pos] * (t < stop_at)
            if windows:
                win[:, :, t * windows // total_ticks] += sd
            s_ns = s
            s_src = xs[:, t] if t < T else torch.zeros_like(s_src)
        if windows:
            return counts.view(B, G, K), win
        return counts.view(B, G, K)

    def loss(self, counts: Tensor, labels: Tensor) -> Tensor:
        """Sum over graphs of the batch-mean cross-entropy on scaled sink counts."""
        logits = counts * self.log_scale.exp().view(1, -1, 1)          # [B, G, K]
        B, G, K = logits.shape
        ce = F.cross_entropy(logits.reshape(B * G, K), labels.repeat_interleave(G), reduction="none")
        return ce.view(B, G).mean(0).sum()

    @torch.no_grad()
    def export(self, graphs: list[Data]) -> list[Data]:
        """Graphs carrying the current (trained) parameters, in float64 for exact re-simulation."""
        S = self.num_source
        w = self.weights()
        thr, leak = self.neuron_params()
        out, ns_off = [], 0
        for g, (e_off, order) in zip(graphs, self.edge_slices):
            M = g.num_nodes - S
            x = g.x.double().clone()
            x[S:, 0] = thr[ns_off:ns_off + M]
            x[S:, 1] = leak[ns_off:ns_off + M]
            ea = g.edge_attr.double().clone()
            ea[order, 0] = w[e_off:e_off + len(order)]
            out.append(Data(x=x, edge_index=g.edge_index.clone(), edge_attr=ea))
            ns_off += M
        return out
