"""
Value function V_ω(s_t) for PPO advantage estimation.

Following GCPN Section 3.5: V_ω is an MLP that maps the aggregated
(SUM-pooled) graph embedding to a scalar expected return estimate.
The same GCN encoder as the policy network is used to compute H,
and V_ω operates on AGG(H) = SUM_i H_i.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

EMBED_DIM = 64
MLP_HIDDEN = 64


class ValueFunction(nn.Module):
    """
    V_ω : graph_embedding ∈ R^{embed_dim} → scalar.

    Parameters
    ----------
    embed_dim  : must match GCNEncoder.embed_dim
    mlp_hidden : hidden layer width
    """

    def __init__(self, embed_dim: int = EMBED_DIM, mlp_hidden: int = MLP_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, H: Tensor) -> Tensor:
        """
        Parameters
        ----------
        H : [n, embed_dim]  node embeddings from GCNEncoder

        Returns
        -------
        value : scalar tensor
        """
        graph_emb = H.sum(dim=0)          # [embed_dim]  SUM pooling
        return self.net(graph_emb).squeeze(-1)   # scalar
