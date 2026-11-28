"""
Batched trajectory sampling and re-scoring used by the GRPO trainer.

All trajectories of a group advance together: at every step the current
states of the still-running episodes are encoded as one disjoint batch and
`PolicyHead.forward_batch` samples all of their actions at once. Re-scoring
encodes every prefix state of several trajectories in one batch.

A Trajectory stores the final graph, the actions, the hidden / edge counts
before each action (states are prefixes, see snn_env.prefix_graph) and the
masks used at sampling time, so re-scoring reproduces the exact distributions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch_geometric.data import Data

from spikegcpn.env.snn_env import CONNECT, NEW, STOP_ACTION, GraphSpec, Masks, SNNEnv, prefix_graph
from spikegcpn.model.encoder import GraphEncoder
from spikegcpn.model.policy import PolicyHead, collate


@dataclass
class Trajectory:
    final: Data
    actions: list[dict]
    hidden_counts: list[int]          # hidden neurons before each action (+ final)
    edge_counts: list[int]            # edges before each action (+ final)
    masks: list[Masks] = field(default_factory=list)
    second_masks: list[Tensor | None] = field(default_factory=list)
    old_log_probs: list[float] = field(default_factory=list)
    key: str = ""

    @property
    def num_edges(self) -> int:
        return int(self.final.edge_index.shape[1])

    @property
    def num_hidden(self) -> int:
        return self.hidden_counts[-1]

    def __len__(self) -> int:
        return len(self.actions)


def graph_key(g: Data, spec: GraphSpec) -> str:
    """Hash of the final topology, invariant to edge insertion order."""
    hid = g.x[spec.hidden_start:, :2].tolist()
    edges = sorted(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist(), g.edge_attr[:, 0].tolist()))
    return hashlib.sha1(repr((hid, edges)).encode()).hexdigest()


@torch.no_grad()
def sample_trajectories(
    spec: GraphSpec,
    encoder: GraphEncoder,
    policy: PolicyHead,
    n: int,
    temperature: float = 1.0,
    eps: float = 0.0,
) -> list[Trajectory]:
    envs = [SNNEnv(spec) for _ in range(n)]
    for e in envs:
        e.reset()
    trajs = [Trajectory(final=None, actions=[], hidden_counts=[], edge_counts=[]) for _ in range(n)]
    active = list(range(n))
    while active:
        graphs = [envs[i].graph for i in active]
        hs = [envs[i].num_hidden for i in active]
        ms = [envs[i].masks() for i in active]
        x, ei, ea, ptr = collate(graphs)
        H = encoder.encode_batch(x, ei, ea, ptr, spec.num_source)
        acts, lp, _, sms = policy.forward_batch(
            H, ptr, graphs, hs, ms,
            second_mask_fn=lambda row, u, act=active: envs[act[row]].second_mask(u),
            temperature=temperature, eps=eps,
        )
        still = []
        for row, i in enumerate(active):
            tr, env = trajs[i], envs[i]
            tr.actions.append(acts[row])
            tr.hidden_counts.append(env.num_hidden)
            tr.edge_counts.append(env.num_edges)
            tr.masks.append(ms[row])
            tr.second_masks.append(sms[row])
            tr.old_log_probs.append(float(lp[row]))
            g, done = env.step(acts[row])
            if done:
                tr.final = g
                tr.hidden_counts.append(env.num_hidden)
                tr.edge_counts.append(env.num_edges)
                tr.key = graph_key(g, spec)
            else:
                still.append(i)
        active = still
    return trajs


def rescore_batch(
    trajs: list[Trajectory],
    encoder: GraphEncoder,
    policy: PolicyHead,
    steps: list[list[int]] | None = None,
) -> list[tuple[Tensor, Tensor]]:
    """
    Per trajectory: (log π_θ(a_t|s_t), entropy_t) under current parameters, with
    grad, for the time steps in steps[i] (default: all T steps, in order).
    """
    spec = policy.spec
    if steps is None:
        steps = [list(range(len(tr))) for tr in trajs]
    graphs, hs, ms, sms, acts = [], [], [], [], []
    for tr, ts in zip(trajs, steps):
        for t in ts:
            graphs.append(prefix_graph(tr.final, tr.edge_counts[t], tr.hidden_counts[t], spec))
            hs.append(tr.hidden_counts[t])
            ms.append(tr.masks[t])
            sms.append(tr.second_masks[t])
            acts.append(tr.actions[t])
    x, ei, ea, ptr = collate(graphs)
    H = encoder.encode_batch(x, ei, ea, ptr, spec.num_source)
    _, lp, ent, _ = policy.forward_batch(H, ptr, graphs, hs, ms, actions=acts, second_masks=sms)
    out, o = [], 0
    for ts in steps:
        T = len(ts)
        out.append((lp[o:o + T], ent[o:o + T]))
        o += T
    return out


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

def random_action(env: SNNEnv, rng, p_stop: float = 0.0) -> dict:
    """Uniformly random legal macro action (stop only when forced, or with p_stop when allowed)."""
    spec = env.spec
    m = env.masks()
    if not m.continue_allowed or (m.stop_allowed and rng.random() < p_stop):
        return dict(STOP_ACTION)
    kinds = [k for k, ok in ((NEW, m.new_allowed), (CONNECT, m.connect_allowed)) if ok]
    a = dict(STOP_ACTION, a_stop=0, a_kind=rng.choice(kinds), a_edge=rng.randrange(spec.num_edge_actions))
    if a["a_kind"] == NEW:
        table = spec.rf_edge_table()
        a["a_cell"] = rng.randrange(spec.num_cells)
        fits = table[a["a_cell"]] <= m.remaining
        a["a_radius"] = rng.choice([r for r in range(len(spec.rf_radii)) if bool(fits[r].any())])
        a["a_pol"] = rng.choice([p for p in range(spec.num_pol_choices) if bool(fits[a["a_radius"], p])])
        a["a_node"] = rng.randrange(spec.num_presets)
    else:
        a["a_first"] = rng.choice(m.first.nonzero(as_tuple=True)[0].tolist())
        a["a_second"] = rng.choice(env.second_mask(a["a_first"]).nonzero(as_tuple=True)[0].tolist())
    return a


def sample_random_trajectory(env: SNNEnv, rng) -> Trajectory:
    """Baseline: uniformly random legal macro actions until the episode is forced to stop."""
    spec = env.spec
    env.reset()
    actions, hs, es = [], [], []
    while True:
        a = random_action(env, rng)
        actions.append(a)
        hs.append(env.num_hidden)
        es.append(env.num_edges)
        g, done = env.step(a)
        if done:
            break
    return Trajectory(final=g, actions=actions, hidden_counts=hs + [env.num_hidden],
                      edge_counts=es + [env.num_edges], key=graph_key(g, spec))
