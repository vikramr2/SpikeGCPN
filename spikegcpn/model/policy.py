"""
GCPN policy head (You et al., NeurIPS 2018, Section 3.4 / Eq. 3-4).

Action at time t is a tuple (a_first, a_second, a_edge, a_stop):

  f_first(s_t)  = SOFTMAX( m_f(H) )
      → distribution over n existing nodes

  f_second(s_t) = SOFTMAX( m_s([ H[a_first] ‖ H_cand ]) )
      → distribution over n existing nodes + 1 "new hidden neuron" option

  f_edge(s_t)   = SOFTMAX( m_e([ H[a_first] ‖ H[a_second] ]) )
      → distribution over 2 edge types (excitatory / inhibitory)

  f_stop(s_t)   = SOFTMAX( m_t(AGG(H)) )
      → distribution over {continue, stop}

SNN adaptation:
- No scaffold set C; instead a single learnable "new_node_emb" parameter
  represents the prototype embedding of an uninstantiated hidden neuron.
- When a_second == n (the (n+1)-th option), the environment creates a new
  hidden neuron. The env-facing sentinel is NEW_NODE = -1.
"""

from __future__ import annotations

from typing import NamedTuple
import torch
import torch.nn as nn
from torch import Tensor

EMBED_DIM = 64
MLP_HIDDEN = 64
NUM_EDGE_TYPES = 2
NEW_NODE = -1  # sentinel used by the environment


class LogProbs(NamedTuple):
    first: Tensor
    second: Tensor
    edge: Tensor
    stop: Tensor

    def total(self) -> Tensor:
        """Joint log-prob under the factored policy."""
        return self.first + self.second + self.edge + self.stop


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class PolicyHead(nn.Module):
    """
    Four-headed policy following GCPN Eq. 4.

    Parameters
    ----------
    embed_dim : node embedding dimension (must match GCNEncoder output)
    mlp_hidden : hidden layer width for all four MLPs
    """

    def __init__(self, embed_dim: int = EMBED_DIM, mlp_hidden: int = MLP_HIDDEN):
        super().__init__()
        self.embed_dim = embed_dim

        # m_f : z_i          → scalar score per node
        self.m_f = _mlp(embed_dim, mlp_hidden, 1)

        # m_s : [z_first ‖ z_j] → scalar score per candidate j (existing or new)
        self.m_s = _mlp(2 * embed_dim, mlp_hidden, 1)

        # m_e : [z_first ‖ z_second] → score per edge type
        self.m_e = _mlp(2 * embed_dim, mlp_hidden, NUM_EDGE_TYPES)

        # m_t : graph_embedding → score per {continue, stop}
        self.m_t = _mlp(embed_dim, mlp_hidden, 2)

        # Prototype embedding for an uninstantiated new hidden neuron
        self.new_node_emb = nn.Parameter(torch.randn(embed_dim) * 0.1)

    # ------------------------------------------------------------------
    # Core distributions
    # ------------------------------------------------------------------

    def first_dist(
        self,
        H: Tensor,
        mask: Tensor | None = None,
    ) -> torch.distributions.Categorical:
        """
        Distribution over existing n nodes.

        Parameters
        ----------
        mask : optional bool tensor [n]; True = node is a valid a_first choice.
               Source nodes are always valid; sink nodes are masked out.
        """
        logits = self.m_f(H).squeeze(-1)           # [n]
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return torch.distributions.Categorical(logits=logits)

    def second_dist(
        self,
        H: Tensor,
        a_first: int | Tensor,
        mask: Tensor | None = None,
    ) -> torch.distributions.Categorical:
        """
        Distribution over n existing nodes + 1 new-hidden option.
        Candidates: H (rows) + new_node_emb (appended as row n).

        Parameters
        ----------
        mask : optional bool tensor [n]; True = node is a valid a_second choice.
               Source nodes are masked out; the new-hidden option (index n)
               is always kept valid.
        """
        n = H.size(0)
        z_first = H[int(a_first)]                  # [d]

        cand = torch.cat([H, self.new_node_emb.unsqueeze(0)], dim=0)   # [n+1, d]
        z_first_rep = z_first.unsqueeze(0).expand(n + 1, -1)            # [n+1, d]
        pair = torch.cat([z_first_rep, cand], dim=1)                    # [n+1, 2d]
        logits = self.m_s(pair).squeeze(-1)                             # [n+1]

        if mask is not None:
            # mask covers existing n nodes; append True for the new-hidden option
            full_mask = torch.cat([mask, mask.new_ones(1)], dim=0)     # [n+1]
            logits = logits.masked_fill(~full_mask, float("-inf"))

        return torch.distributions.Categorical(logits=logits)

    def edge_dist(
        self, H: Tensor, a_first: int | Tensor, a_second_raw: int | Tensor
    ) -> torch.distributions.Categorical:
        """
        Distribution over edge types.
        a_second_raw is in [0..n] where n = new-node option.
        """
        n = H.size(0)
        a_second_raw = int(a_second_raw)
        z_first = H[int(a_first)]
        z_second = H[a_second_raw] if a_second_raw < n else self.new_node_emb
        pair = torch.cat([z_first, z_second], dim=0).unsqueeze(0)  # [1, 2d]
        logits = self.m_e(pair).squeeze(0)                          # [2]
        return torch.distributions.Categorical(logits=logits)

    def stop_dist(self, H: Tensor) -> torch.distributions.Categorical:
        """Distribution over {continue=0, stop=1}."""
        graph_emb = H.sum(dim=0)               # [d]  SUM pooling
        logits = self.m_t(graph_emb)           # [2]
        return torch.distributions.Categorical(logits=logits)

    # ------------------------------------------------------------------
    # Sample a complete action
    # ------------------------------------------------------------------

    def sample_actions(
        self,
        H: Tensor,
        first_mask: Tensor | None = None,
        second_mask: Tensor | None = None,
    ) -> tuple[dict, LogProbs]:
        """
        Sample (a_stop, a_first, a_second, a_edge) from the current policy.

        Parameters
        ----------
        first_mask  : bool [n] — True = node may be selected as a_first
        second_mask : bool [n] — True = node may be selected as a_second
                      (does not cover the new-hidden option, which is always valid)

        Returns
        -------
        action : dict with keys a_first, a_second, a_edge, a_stop
                 a_second uses the env sentinel NEW_NODE=-1 when applicable
        log_probs : LogProbs namedtuple
        """
        n = H.size(0)

        # a_stop
        d_stop = self.stop_dist(H)
        a_stop = d_stop.sample()
        lp_stop = d_stop.log_prob(a_stop)

        # a_first
        d_first = self.first_dist(H, mask=first_mask)
        a_first = d_first.sample()
        lp_first = d_first.log_prob(a_first)

        # a_second  (index in [0..n], where n = new-hidden option)
        d_second = self.second_dist(H, a_first, mask=second_mask)
        a_second_raw = d_second.sample()      # in [0..n]
        lp_second = d_second.log_prob(a_second_raw)

        # a_edge
        d_edge = self.edge_dist(H, a_first, a_second_raw)
        a_edge = d_edge.sample()
        lp_edge = d_edge.log_prob(a_edge)

        # Convert a_second to env convention
        a_second_env = int(a_second_raw.item()) if int(a_second_raw.item()) < n else NEW_NODE

        action = {
            "a_first":  int(a_first.item()),
            "a_second": a_second_env,
            "a_edge":   int(a_edge.item()),
            "a_stop":   int(a_stop.item()),
        }
        log_probs = LogProbs(
            first=lp_first,
            second=lp_second,
            edge=lp_edge,
            stop=lp_stop,
        )
        return action, log_probs

    # ------------------------------------------------------------------
    # Re-evaluate log-probs for stored actions (PPO update)
    # ------------------------------------------------------------------

    def evaluate_log_probs(
        self,
        H: Tensor,
        action: dict,
        first_mask: Tensor | None = None,
        second_mask: Tensor | None = None,
    ) -> LogProbs:
        """
        Recompute log π_θ(a|s) under the current parameters for a stored action.
        Used during PPO update to compute importance ratios.

        action["a_second"] uses env convention (NEW_NODE = -1);
        we convert back to tensor index n for the distribution query.
        Masks must match those used during sampling to keep old/new log-probs comparable.
        """
        n = H.size(0)
        a_first = action["a_first"]
        a_second_env = action["a_second"]
        a_second_raw = n if a_second_env == NEW_NODE else a_second_env
        a_edge = action["a_edge"]
        a_stop = action["a_stop"]

        lp_first = self.first_dist(H, mask=first_mask).log_prob(
            torch.tensor(a_first, device=H.device)
        )
        lp_second = self.second_dist(H, a_first, mask=second_mask).log_prob(
            torch.tensor(a_second_raw, device=H.device)
        )
        lp_edge = self.edge_dist(H, a_first, a_second_raw).log_prob(
            torch.tensor(a_edge, device=H.device)
        )
        lp_stop = self.stop_dist(H).log_prob(
            torch.tensor(a_stop, device=H.device)
        )

        return LogProbs(first=lp_first, second=lp_second, edge=lp_edge, stop=lp_stop)

    def entropy(self, H: Tensor) -> Tensor:
        """
        Approximate joint entropy: H(stop) + H(first).
        (Full conditional entropy over all heads is expensive to compute exactly.)
        """
        return self.stop_dist(H).entropy() + self.first_dist(H).entropy()
