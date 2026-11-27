"""
Relational graph encoder for SNN topologies.

Changes from the original GCPN encoder (You et al. 2018, Eq. 2):

* Bidirectional message passing. Each edge type (excitatory / inhibitory) is
  split into a forward relation (tail → head) and a backward relation
  (head → tail), each with its own weights. With forward-only passing a source
  neuron (in-degree 0) never receives messages, so all sources had identical
  embeddings and a_first was exactly uniform over pixels.
* Node identity features. Sources get Fourier features of their (x, y) pixel
  position plus polarity; sinks get a learned per-class embedding. Without
  these, the 2312 sources and 10 sinks are indistinguishable at reset.
* Messages are scaled by |weight| so synaptic strength is visible.
* Mean aggregation per relation, residual connections and LayerNorm
  (BatchNorm over the nodes of a single graph made embeddings depend on the
  batch composition of that graph).

Input features follow snn_env.py:
  [threshold, leak, reset, refractory, node_type, pos_x, pos_y, polarity, sink_id]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from spikegcpn.env.snn_env import (
    THR, LEAK, RESET, REFRAC, TYPE, PX, PY, POL, SINK_ID, SOURCE, SINK,
)

NUM_RELATIONS = 4  # (exc, inh) × (fwd, bwd)


class _RelLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_lin = nn.Linear(dim, dim)
        self.rel_lins = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(NUM_RELATIONS)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: Tensor, rels: list[tuple[int, Tensor, Tensor, Tensor]], dst_all: Tensor) -> Tensor:
        out = self.self_lin(h)
        msgs = [self.rel_lins[r](h[src]) * coef.unsqueeze(1) for r, src, _, coef in rels]
        if msgs:
            out = out.index_add(0, dst_all, torch.cat(msgs))
        return self.norm(h + F.relu(out))


class GraphEncoder(nn.Module):
    def __init__(
        self,
        num_sink: int,
        embed_dim: int = 64,
        num_layers: int = 3,
        num_fourier: int = 6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer("freqs", (2.0 ** torch.arange(num_fourier)) * math.pi, persistent=False)

        self.cont_proj = nn.Linear(4, embed_dim)            # threshold, leak, reset, refractory
        self.type_emb = nn.Embedding(3, embed_dim)
        self.sink_emb = nn.Embedding(num_sink, embed_dim)
        self.pos_proj = nn.Linear(4 * num_fourier + 1, embed_dim)
        self.layers = nn.ModuleList([_RelLayer(embed_dim) for _ in range(num_layers)])

    def embed_nodes(self, x: Tensor) -> Tensor:
        x = x.float()
        cont = torch.stack([
            x[:, THR],
            1.0 / (1.0 + x[:, LEAK].clamp(min=0.0, max=1e6)),   # leak ∈ [0, inf] → (0, 1]
            x[:, RESET],
            x[:, REFRAC],
        ], dim=1)
        ntype = x[:, TYPE].long()
        h = self.cont_proj(cont) + self.type_emb(ntype)

        is_src = ntype == SOURCE
        if is_src.any():
            xy = x[is_src][:, [PX, PY]]                                      # [s, 2]
            ang = xy.unsqueeze(-1) * self.freqs                              # [s, 2, F]
            four = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(1)      # [s, 4F]
            pos_in = torch.cat([four, x[is_src][:, POL:POL + 1]], dim=1)
            h = h.index_add(0, is_src.nonzero(as_tuple=True)[0], self.pos_proj(pos_in))

        is_sink = ntype == SINK
        if is_sink.any():
            h = h.index_add(0, is_sink.nonzero(as_tuple=True)[0], self.sink_emb(x[is_sink][:, SINK_ID].long()))
        return F.relu(h)

    @staticmethod
    def relations(edge_index: Tensor, edge_attr: Tensor | None, n: int):
        """
        Per non-empty relation r (0 exc-fwd, 1 exc-bwd, 2 inh-fwd, 3 inh-bwd): (r, src, dst, coef) with
        coef = |weight| / (#messages dst receives in that relation), i.e. a
        |w|-weighted mean. Also returns the concatenated dst of non-empty relations.
        """
        if edge_index.numel() == 0:
            return [], edge_index.new_zeros(0)
        w = edge_attr[:, 0].float()
        exc = w > 0
        src, dst, mag = edge_index[0], edge_index[1], w.abs()
        rels = []
        for k, m in enumerate((exc, ~exc)):
            s, d, a = src[m], dst[m], mag[m]
            if s.numel() == 0:
                continue
            for j, (rs, rd) in enumerate(((s, d), (d, s))):   # forward: tail → head ; backward: head → tail
                cnt = torch.zeros(n).index_add_(0, rd, torch.ones_like(a))
                rels.append((2 * k + j, rs, rd, a / cnt[rd]))
        return rels, torch.cat([rd for _, _, rd, _ in rels])

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None) -> Tensor:
        h = self.embed_nodes(x)
        rels, dst_all = self.relations(edge_index, edge_attr, h.size(0))
        for layer in self.layers:
            h = layer(h, rels, dst_all)
        return h

    def encode_batch(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, ptr: Tensor, num_source: int) -> Tensor:
        """
        Same result as forward() on a disjoint union of graph states (collate
        layout: sources are local nodes [0, S) of every graph), but cheaper.

        A source without synapses receives no messages, so every layer acts on it
        node-wise and its embedding depends only on its (fixed) features. Those
        embeddings are computed once from graph 0's S sources and gathered; only
        the remaining nodes go through message passing. Most of the 162 sources
        are unconnected in most states, so this roughly halves encoder work.
        """
        N, S = x.size(0), num_source
        B = ptr.numel() - 1
        local = torch.arange(N) - torch.repeat_interleave(ptr[:-1], ptr[1:] - ptr[:-1])
        deg = torch.zeros(N, dtype=torch.long)
        if edge_index.numel():
            deg.index_add_(0, edge_index.reshape(-1), torch.ones(edge_index.numel(), dtype=torch.long))
        iso_src = (local < S) & (deg == 0)
        if B < 2 or not bool(iso_src.any()):
            return self.forward(x, edge_index, edge_attr)
        base = self.forward(x[ptr[0]:ptr[0] + S], edge_index.new_zeros(2, 0), None)   # [S, d]
        keep = ~iso_src
        remap = torch.cumsum(keep.long(), 0) - 1
        hk = self.forward(x[keep], remap[edge_index], edge_attr)
        h = hk.new_empty(N, hk.size(1))
        h[keep] = hk
        h[iso_src] = base[local[iso_src]]
        return h
