"""
Pure graph helpers shared by the environment, the policy, the GFlowNet
backward policy and the simulator wrapper.

Node index layout (see snn_env.py):
    [0, S)            source neurons  (one per input channel)
    [S, S+K)          sink neurons    (one per class)
    [S+K, S+K+H)      hidden neurons, in creation order

Every function here takes the layout sizes explicitly so it can be applied to
any graph prefix (GRPO/TB re-evaluate stored trajectories step by step).
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch


def _adjacency(edge_index: torch.Tensor, num_nodes: int) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Sparse adjacency (most of the ~2400 nodes are isolated)."""
    out_adj: dict[int, list[int]] = defaultdict(list)
    in_adj: dict[int, list[int]] = defaultdict(list)
    if edge_index.numel() > 0:
        for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            out_adj[s].append(d)
            in_adj[d].append(s)
    return out_adj, in_adj


def is_dag(edge_index: torch.Tensor, num_nodes: int) -> bool:
    """Kahn's algorithm: True iff the directed graph has no cycle."""
    out_adj, in_adj = _adjacency(edge_index, num_nodes)
    indeg = [len(in_adj.get(i, ())) for i in range(num_nodes)]
    stack = [i for i in range(num_nodes) if indeg[i] == 0]
    seen = 0
    while stack:
        u = stack.pop()
        seen += 1
        for v in out_adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    return seen == num_nodes


def would_create_cycle(edge_index: torch.Tensor, num_nodes: int, src: int, dst: int) -> bool:
    """True if adding src → dst closes a cycle (dst already reaches src, or src == dst)."""
    if src == dst:
        return True
    out_adj, _ = _adjacency(edge_index, num_nodes)
    seen = {dst}
    stack = [dst]
    while stack:
        u = stack.pop()
        if u == src:
            return True
        for v in out_adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def ancestors(in_adj: list[list[int]], node: int) -> set[int]:
    """All nodes with a directed path to `node` (excluding `node` itself)."""
    seen: set[int] = set()
    stack = list(in_adj[node])
    while stack:
        u = stack.pop()
        if u not in seen:
            seen.add(u)
            stack.extend(in_adj[u])
    return seen


def all_sinks_reachable(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_source: int,
    num_sink: int,
) -> bool:
    """True iff every sink in [S, S+K) is reachable from some source in [0, S)."""
    out_adj, _ = _adjacency(edge_index, num_nodes)
    seen = set(range(num_source))
    stack = list(range(num_source))
    while stack:
        u = stack.pop()
        for v in out_adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return all(i in seen for i in range(num_source, num_source + num_sink))


def longest_path_length(edge_index: torch.Tensor, num_nodes: int) -> int:
    """Number of edges on the longest path of a DAG (0 for an edgeless graph)."""
    out_adj, in_adj = _adjacency(edge_index, num_nodes)
    indeg = [len(in_adj.get(i, ())) for i in range(num_nodes)]
    depth = [0] * num_nodes
    stack = [i for i in range(num_nodes) if indeg[i] == 0]
    while stack:
        u = stack.pop()
        for v in out_adj[u]:
            depth[v] = max(depth[v], depth[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    return max(depth) if depth else 0


# ---------------------------------------------------------------------------
# Masks for CONNECT actions (tail: hidden neuron; head: hidden or sink)
# ---------------------------------------------------------------------------

def connect_first_mask(
    edge_index: torch.Tensor,
    num_source: int,
    num_sink: int,
    num_hidden: int,
    allow_cycles: bool = False,
) -> torch.Tensor:
    """
    bool [n]: hidden neurons that have at least one legal CONNECT head, i.e. a
    sink or another hidden neuron that is not already a target (and, in DAG
    mode, not an ancestor).
    """
    S, K, H = num_source, num_sink, num_hidden
    n = S + K + H
    hid0 = S + K
    mask = torch.zeros(n, dtype=torch.bool)
    if H == 0:
        return mask
    out_adj, in_adj = _adjacency(edge_index, n)
    for u in range(hid0, n):
        hidden_anc = 0 if allow_cycles else sum(1 for a in ancestors(in_adj, u) if a >= hid0)
        legal = K + (H - 1 - hidden_anc) - len(out_adj[u])
        mask[u] = legal > 0
    return mask


def second_mask(
    edge_index: torch.Tensor,
    num_source: int,
    num_sink: int,
    num_hidden: int,
    a_first: int,
    allow_cycles: bool = False,
) -> torch.Tensor:
    """
    bool [n]: True where a_first → v is a legal CONNECT synapse (v is not a
    source, not a_first, not already targeted, and — in DAG mode — not an
    ancestor of a_first).
    """
    S, K, H = num_source, num_sink, num_hidden
    n = S + K + H
    mask = torch.ones(n, dtype=torch.bool)
    mask[:S] = False
    mask[a_first] = False
    if edge_index.numel() > 0:
        out_adj, in_adj = _adjacency(edge_index, n)
        for v in out_adj[a_first]:
            mask[v] = False
        if a_first >= S and not allow_cycles:
            for a in ancestors(in_adj, a_first):
                mask[a] = False
    return mask


# ---------------------------------------------------------------------------
# GFlowNet backward policy support
# ---------------------------------------------------------------------------

def num_parents(edge_index: torch.Tensor, spec, num_hidden: int) -> int:
    """
    Number of (parent state, action) pairs leading to this state under the
    macro action space; used for a uniform backward policy over them,
    P_B = 1 / num_parents(s').

    Synapses with a hidden tail can only come from CONNECT; synapses with a
    source tail only from the NEW action that created their head. Hence:
      * every hidden-tail synapse can be the last action (removing it leaves a
        reachable state; re-adding it is legal, also in DAG mode);
      * the most recent hidden neuron's NEW can be the last action iff it has no
        out-synapses and no hidden-tail in-synapses. Several (cell, radius, pol)
        choices can clip to the same receptive field at the grid border; each
        is a distinct action, so they are counted separately.
    """
    S, K, H = spec.num_source, spec.num_sink, num_hidden
    if edge_index.numel() == 0:
        return 0
    hid0 = S + K
    src, dst = edge_index[0], edge_index[1]
    count = int((src >= hid0).sum())
    if H > 0:
        last = hid0 + H - 1
        no_out = not bool((src == last).any())
        no_conn_in = not bool(((dst == last) & (src >= hid0)).any())
        if no_out and no_conn_in:
            rf = tuple(sorted(src[dst == last].tolist()))
            count += spec.rf_multiplicity(rf)
    return count


def log_num_parents(edge_index: torch.Tensor, spec, num_hidden: int) -> float:
    p = num_parents(edge_index, spec, num_hidden)
    return math.log(p) if p > 0 else 0.0
