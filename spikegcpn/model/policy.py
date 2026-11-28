"""
Batched autoregressive policy over macro actions (see env/snn_env.py).

For a batch of B graph states encoded together (one GraphEncoder call on their
disjoint union), every decision stage is evaluated for all rows at once:

  a_stop  ~ m_stop(c)                                   {continue, stop}
  a_kind  ~ m_kind(c)                                   {NEW, CONNECT}
  NEW:      a_cell   ~ m_cell([z_cell ‖ c])             per patch cell; z_cell = mean of
                                                        that cell's two source embeddings
            a_radius ~ m_radius([z_cell ‖ c])           radii whose field fits the budget
            a_pol    ~ m_pol([z_cell ‖ c ‖ onehot(r)])  polarities that fit the budget
            a_node   ~ m_node([z_cell ‖ c])             hidden preset
            a_edge   ~ m_wnew([z_cell ‖ c])             shared weight bin of the field
  CONNECT:  a_first  ~ m_first([z_i ‖ c])               legal hidden tails
            a_second ~ m_second([z_first ‖ z_j ‖ c])    legal heads
            a_edge   ~ m_wconn([z_first ‖ z_second ‖ c])

c = [mean(H), mean(H_non-source), E/max_edges, H/max_hidden] per graph.
Masks are exact, so log π(a|s) is the log-probability of the real transition
(the PPO-style importance ratios in GRPO rely on this). Rows where a stage is not reached
contribute log 1 = 0 and no entropy.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
from torch_geometric.data import Data

from spikegcpn.env.snn_env import CONNECT, NEW, SOURCE, STOP_ACTION, TYPE, GraphSpec, Masks


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


def _categorical(logits: Tensor, mask: Tensor, temperature: float, eps: float) -> Categorical:
    if temperature != 1.0:
        logits = logits / temperature
    logits = logits.masked_fill(~mask, float("-inf"))
    if eps > 0.0:
        valid = mask.float()
        probs = (1 - eps) * torch.softmax(logits, dim=-1) + eps * valid / valid.sum(-1, keepdim=True)
        return Categorical(probs=probs)
    return Categorical(logits=logits)


def collate(graphs: list[Data]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Disjoint union: (x, edge_index, edge_attr, ptr) with ptr[i] = first node of graph i."""
    sizes = torch.tensor([g.num_nodes for g in graphs])
    ptr = torch.zeros(len(graphs) + 1, dtype=torch.long)
    ptr[1:] = sizes.cumsum(0)
    x = torch.cat([g.x for g in graphs])
    ei = torch.cat([g.edge_index + ptr[i] for i, g in enumerate(graphs)], dim=1)
    ea = torch.cat([g.edge_attr for g in graphs])
    return x, ei, ea, ptr


class PolicyHead(nn.Module):
    def __init__(self, spec: GraphSpec, embed_dim: int = 64, mlp_hidden: int = 128):
        super().__init__()
        self.spec = spec
        d, c, R, L2 = embed_dim, 2 * embed_dim + 2, len(spec.rf_radii), spec.num_edge_actions
        self.m_stop = _mlp(c, mlp_hidden, 2)
        self.m_kind = _mlp(c, mlp_hidden, 2)
        self.m_cell = _mlp(d + c, mlp_hidden, 1)
        self.m_radius = _mlp(d + c, mlp_hidden, R)
        self.m_pol = _mlp(d + c + R, mlp_hidden, spec.num_pol_choices)
        self.m_node = _mlp(d + c, mlp_hidden, spec.num_presets)
        self.m_wnew = _mlp(d + c, mlp_hidden, L2)
        self.m_first = _mlp(d + c, mlp_hidden, 1)
        self.m_second = _mlp(2 * d + c, mlp_hidden, 1)
        self.m_wconn = _mlp(2 * d + c, mlp_hidden, L2)

        # Start with a small P(stop) where stopping is allowed, so early episodes
        # are not cut short at P(stop) = 0.5.
        with torch.no_grad():
            self.m_stop[-1].bias.copy_(torch.tensor([0.0, float(torch.logit(torch.tensor(0.03)))]))
            self.m_stop[-1].weight.mul_(0.1)

    def forward_batch(
        self,
        H: Tensor,
        ptr: Tensor,
        graphs: list[Data],
        hidden_counts: list[int],
        masks: list[Masks],
        actions: list[dict] | None = None,
        second_masks: list[Tensor | None] | None = None,
        second_mask_fn: Callable[[int, int], Tensor] | None = None,
        temperature: float = 1.0,
        eps: float = 0.0,
    ) -> tuple[list[dict], Tensor, Tensor, list[Tensor | None]]:
        """
        Sample (actions=None, needs second_mask_fn(row, a_first)) or score given
        actions (needs the stored second_masks). Returns (actions, log_prob [B],
        entropy [B], second masks used per row).
        """
        spec = self.spec
        sample = actions is None
        B = len(graphs)
        N = H.size(0)
        S, G2 = spec.num_source, spec.num_cells
        sizes = ptr[1:] - ptr[:-1]
        gid = torch.repeat_interleave(torch.arange(B), sizes)
        local = torch.arange(N) - ptr[gid]
        rows = torch.arange(B)

        # --- graph context ------------------------------------------------
        mean_all = torch.zeros(B, H.size(1)).index_add(0, gid, H) / sizes.unsqueeze(1)
        ns = torch.cat([g.x[:, TYPE] != SOURCE for g in graphs])
        ns_cnt = torch.zeros(B).index_add(0, gid, ns.float()).clamp(min=1)
        mean_ns = torch.zeros(B, H.size(1)).index_add(0, gid[ns], H[ns]) / ns_cnt.unsqueeze(1)
        E = torch.tensor([float(g.edge_index.shape[1]) for g in graphs])
        Hc = torch.tensor([float(h) for h in hidden_counts])
        c = torch.cat([mean_all, mean_ns, (E / spec.max_edges).unsqueeze(1),
                       (Hc / max(1, spec.max_hidden)).unsqueeze(1)], dim=1)

        logp = torch.zeros(B)
        ent = torch.zeros(B)
        chosen: dict[str, Tensor] = {}

        def stored(key: str, active: Tensor) -> Tensor:
            v = torch.tensor([a[key] for a in actions])
            return torch.where(active, v, torch.zeros_like(v))

        def pick(key: str, logits: Tensor, mask: Tensor, active: Tensor) -> Tensor:
            nonlocal logp, ent
            mask = torch.where(active.unsqueeze(1), mask, mask | dummy_ok(mask))
            dist = _categorical(logits, mask, temperature, eps)
            val = dist.sample() if sample else stored(key, active)
            logp = logp + torch.where(active, dist.log_prob(val), torch.zeros(()))
            ent = ent + torch.where(active, dist.entropy(), torch.zeros(()))
            chosen[key] = torch.where(active, val, torch.full_like(val, -1))
            return val

        def dummy_ok(mask: Tensor) -> Tensor:
            # Inactive rows get any finite-logit option so their (ignored) dists are valid.
            ok = torch.zeros_like(mask)
            ok[:, 0] = True
            return ok

        all_rows = torch.ones(B, dtype=torch.bool)

        # --- stop / kind ----------------------------------------------------
        stop_mask = torch.tensor([[m.continue_allowed, m.stop_allowed] for m in masks])
        a_stop = pick("a_stop", self.m_stop(c), stop_mask, all_rows)
        is_cont = a_stop == 0
        kind_mask = torch.tensor([[m.new_allowed, m.connect_allowed] for m in masks])
        a_kind = pick("a_kind", self.m_kind(c), kind_mask, is_cont)
        is_new = is_cont & (a_kind == NEW)
        is_conn = is_cont & (a_kind == CONNECT)

        # --- NEW: receptive field, preset, weight ---------------------------
        if bool(is_new.any()):
            src_idx = ptr[:-1].unsqueeze(1) + torch.arange(S)
            cell_emb = H[src_idx].view(B, spec.num_polarities, G2, -1).mean(1)                      # [B, G2, d]
            cc = c.unsqueeze(1).expand(-1, G2, -1)
            cell_logits = self.m_cell(torch.cat([cell_emb, cc], dim=2)).squeeze(-1)
            a_cell = pick("a_cell", cell_logits, torch.ones(B, G2, dtype=torch.bool), is_new)
            z_cell = cell_emb[rows, a_cell]
            zc = torch.cat([z_cell, c], dim=1)
            remaining = torch.tensor([m.remaining for m in masks])
            fits = spec.rf_edge_table()[a_cell] <= remaining.view(-1, 1, 1)       # [B, R, 3]
            a_rad = pick("a_radius", self.m_radius(zc), fits.any(-1), is_new)
            r_onehot = F.one_hot(a_rad, len(spec.rf_radii)).float()
            pick("a_pol", self.m_pol(torch.cat([zc, r_onehot], dim=1)), fits[rows, a_rad], is_new)
            pick("a_node", self.m_node(zc), torch.ones(B, spec.num_presets, dtype=torch.bool), is_new)
            w_new = pick("a_edge", self.m_wnew(zc), torch.ones(B, spec.num_edge_actions, dtype=torch.bool), is_new)
        else:
            for k in ("a_cell", "a_radius", "a_pol", "a_node"):
                chosen[k] = torch.full((B,), -1)
            w_new = torch.full((B,), -1)

        # --- CONNECT: tail, head, weight -------------------------------------
        used_second: list[Tensor | None] = [None] * B
        w_conn = torch.full((B,), -1)
        if bool(is_conn.any()):
            # Sources are never legal tails or heads: score only sinks + hidden neurons.
            n_max = int(sizes.max())
            cand = local >= S
            g_c, l_c, H_c = gid[cand], local[cand], H[cand]
            valid = torch.zeros(B, n_max, dtype=torch.bool)
            valid[g_c, l_c] = True

            def padded(per_node: Tensor) -> Tensor:
                out = torch.full((B, n_max), float("-inf"))
                out[g_c, l_c] = per_node
                return out

            first_logits = padded(self.m_first(torch.cat([H_c, c[g_c]], dim=1)).squeeze(-1))
            fmask = torch.zeros(B, n_max, dtype=torch.bool)
            for i, m in enumerate(masks):
                fmask[i, : m.first.numel()] = m.first
            a_first = pick("a_first", first_logits, fmask | (~is_conn.unsqueeze(1) & valid), is_conn)
            z_first = H[ptr[:-1] + a_first]

            second_logits = padded(self.m_second(torch.cat([z_first[g_c], H_c, c[g_c]], dim=1)).squeeze(-1))
            smask = valid.clone()
            for i in range(B):
                if bool(is_conn[i]):
                    sm = second_mask_fn(i, int(a_first[i])) if sample else second_masks[i]
                    used_second[i] = sm
                    smask[i] = False
                    smask[i, : sm.numel()] = sm
            a_second = pick("a_second", second_logits, smask, is_conn)
            z_second = H[ptr[:-1] + a_second]
            w_conn = pick("a_edge", self.m_wconn(torch.cat([z_first, z_second, c], dim=1)),
                          torch.ones(B, spec.num_edge_actions, dtype=torch.bool), is_conn)
        else:
            chosen["a_first"] = chosen["a_second"] = torch.full((B,), -1)

        chosen["a_edge"] = torch.where(is_new, w_new, torch.where(is_conn, w_conn, torch.full((B,), -1)))
        out = []
        for i in range(B):
            if not bool(is_cont[i]):
                out.append(dict(STOP_ACTION))
            else:
                out.append({k: int(chosen[k][i]) for k in STOP_ACTION})
        return out, logp, ent, used_second
