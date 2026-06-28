"""
Edge-conditioned GCN encoder from GCPN (You et al., NeurIPS 2018), Section 3.4.

Equation 2:
  H^(l+1) = ReLU( SUM_i( D̃_i^{-1/2}  Ẽ_i  D̃_i^{-1/2}  H^(l)  W_i^(l) ) )

where i indexes edge types (excitatory=0, inhibitory=1),
      Ẽ_i = E_i + I  (self-loops added per edge type),
      D̃_i is the degree matrix of Ẽ_i.

BatchNorm is applied after each ReLU layer, as in the paper's experimental setup.

SNN adaptation:
- Two edge types: excitatory (weight > 0) and inhibitory (weight ≤ 0).
- Node features [threshold, leak, reset_state, refractory_period, node_type] are
  projected to embed_dim before the first GCN layer.
- leak=inf is normalised to 0 (1/(1+leak)) so it is finite for linear layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

NODE_FEAT_DIM = 5
EMBED_DIM = 64
NUM_LAYERS = 3
NUM_EDGE_TYPES = 2  # excitatory, inhibitory


class _GCNLayer(nn.Module):
    """
    One layer of the edge-conditioned GCN (Eq. 2):
      out = SUM_i( sym_norm(E_i + I) @ H @ W_i )
    followed by ReLU + BatchNorm.
    """

    def __init__(self, in_dim: int, out_dim: int, num_edge_types: int):
        super().__init__()
        self.num_edge_types = num_edge_types
        self.weights = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_edge_types)]
        )
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, h: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        n = h.size(0)
        dev = h.device
        agg = torch.zeros(n, self.weights[0].out_features, device=dev)

        for i, lin in enumerate(self.weights):
            # Edges of type i + self-loops
            if edge_index.numel() > 0:
                mask = edge_type == i
                ei = edge_index[:, mask]
            else:
                ei = torch.zeros(2, 0, dtype=torch.long, device=dev)

            # Self-loops: every node connects to itself for this edge type
            loop_idx = torch.arange(n, device=dev)
            loop_ei = torch.stack([loop_idx, loop_idx], dim=0)
            ei = torch.cat([ei, loop_ei], dim=1)

            src, dst = ei[0], ei[1]

            # Symmetric normalisation: D̃^{-1/2}
            deg = torch.zeros(n, device=dev)
            deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=dev))
            inv_sqrt_deg = deg.pow(-0.5)
            inv_sqrt_deg[inv_sqrt_deg.isinf()] = 0.0

            # W H  (transform first, cheaper)
            Wh = lin(h)                             # [n, out_dim]
            msg = inv_sqrt_deg[src].unsqueeze(1) * Wh[src]

            contrib = torch.zeros_like(agg)
            contrib.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
            contrib = inv_sqrt_deg.unsqueeze(1) * contrib

            agg = agg + contrib

        return self.bn(F.relu(agg))


class GCNEncoder(nn.Module):
    """
    3-layer edge-conditioned GCN producing 64-dim node embeddings (GCPN §3.4).

    Input node features (5 columns):
      0: threshold
      1: leak          (normalised: 1/(1+leak), so inf→0)
      2: reset_state
      3: refractory_period
      4: node_type     (0=source, 1=hidden, 2=sink)
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        embed_dim: int = EMBED_DIM,
        num_layers: int = NUM_LAYERS,
        num_edge_types: int = NUM_EDGE_TYPES,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, embed_dim),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList(
            [_GCNLayer(embed_dim, embed_dim, num_edge_types) for _ in range(num_layers)]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x          : [n, node_feat_dim]
        edge_index : [2, E]  (COO directed edges)
        edge_attr  : [E, 2]  columns [weight, delay]; None if no edges

        Returns
        -------
        H : [n, embed_dim]  node embeddings
        """
        x = self._preprocess(x)
        edge_index, edge_type = self._edge_types(edge_index, edge_attr, x.device)

        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_type)
        return h

    @staticmethod
    def graph_embedding(h: Tensor) -> Tensor:
        """SUM pooling: [n, d] → [d]."""
        return h.sum(dim=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(x: Tensor) -> Tensor:
        """Normalise potentially-infinite leak values."""
        x = x.clone().float()
        # Column 1 = leak; map inf → 0 via 1/(1+leak)
        x[:, 1] = 1.0 / (1.0 + x[:, 1].clamp(max=1e6))
        return x

    @staticmethod
    def _edge_types(
        edge_index: Tensor,
        edge_attr: Tensor | None,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Return (edge_index, edge_type) where type 0=excitatory, 1=inhibitory."""
        if edge_index.numel() == 0 or edge_attr is None:
            ei = torch.zeros(2, 0, dtype=torch.long, device=device)
            et = torch.zeros(0, dtype=torch.long, device=device)
        else:
            ei = edge_index.to(device)
            et = (edge_attr[:, 0] <= 0).long().to(device)
        return ei, et
