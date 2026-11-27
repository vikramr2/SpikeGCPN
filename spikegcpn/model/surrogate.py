"""
Surrogate accuracy model: an ensemble of small GNN regressors that predict
simulator accuracy from topology. Trained online on every graph the
simulator has scored. The ensemble spread gives an uncertainty estimate used
to decide which graphs are worth a real simulator call (UCB).

Graphs are pruned to the connected sources + sinks + hidden neurons before
encoding: isolated sources carry no information except their (absent)
connectivity, and dropping them makes the surrogate ~100× cheaper.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from spikegcpn.env.snn_env import GraphSpec, SINK, TYPE
from spikegcpn.model.encoder import GraphEncoder


def prune_isolated_sources(g: Data, num_source: int) -> Data:
    """Drop source nodes with no outgoing edge and re-index the rest."""
    n = g.num_nodes
    keep = torch.ones(n, dtype=torch.bool)
    keep[:num_source] = False
    if g.edge_index.numel() > 0:
        keep[g.edge_index[0][g.edge_index[0] < num_source]] = True
    remap = torch.full((n,), -1, dtype=torch.long)
    remap[keep] = torch.arange(int(keep.sum()))
    return Data(x=g.x[keep], edge_index=remap[g.edge_index], edge_attr=g.edge_attr)


class SurrogateNet(nn.Module):
    def __init__(self, num_sink: int, embed_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.encoder = GraphEncoder(num_sink, embed_dim, num_layers)
        self.head = nn.Sequential(
            nn.Linear(3 * embed_dim + 2, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1),
        )

    def forward(self, g: Data) -> Tensor:
        H = self.encoder(g.x, g.edge_index, g.edge_attr)
        sinks = g.x[:, TYPE].long() == SINK
        hs = H[sinks]
        E = g.edge_index.shape[1]
        n_hidden = int((g.x[:, TYPE].long() > SINK).sum())
        feats = torch.cat([
            hs.mean(0), hs.max(0).values, H.mean(0),
            H.new_tensor([torch.log1p(torch.tensor(float(E))).item(), n_hidden / 10.0]),
        ])
        return self.head(feats).squeeze(-1)   # logit of accuracy


class SurrogateEnsemble:
    def __init__(
        self,
        spec: GraphSpec,
        ensemble_size: int = 4,
        embed_dim: int = 64,
        num_layers: int = 3,
        lr: float = 1e-3,
        seed: int = 0,
    ):
        self.spec = spec
        gen = torch.Generator().manual_seed(seed)
        self.members: list[SurrogateNet] = []
        for _ in range(ensemble_size):
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
            self.members.append(SurrogateNet(spec.num_sink, embed_dim, num_layers))
        self.opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in self.members]
        self.data: list[tuple[Data, float]] = []
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.data)

    def add(self, g: Data, accuracy: float) -> None:
        self.data.append((prune_isolated_sources(g, self.spec.num_source), float(accuracy)))

    def fit(self, steps: int = 50, batch_size: int = 32) -> float:
        """A few bootstrap minibatch steps per member. Returns the final mean MSE."""
        if not self.data:
            return float("nan")
        last = 0.0
        for m, opt in zip(self.members, self.opts):
            m.train()
            for _ in range(steps):
                batch = [self.rng.choice(self.data) for _ in range(min(batch_size, len(self.data)))]
                preds = torch.stack([torch.sigmoid(m(g)) for g, _ in batch])
                target = torch.tensor([a for _, a in batch])
                loss = F.mse_loss(preds, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
            last += loss.item()
        return last / len(self.members)

    @torch.no_grad()
    def predict(self, graphs: list[Data]) -> tuple[Tensor, Tensor]:
        """(mean, std) of predicted accuracy over ensemble members, per graph."""
        pruned = [prune_isolated_sources(g, self.spec.num_source) for g in graphs]
        preds = torch.stack([
            torch.stack([torch.sigmoid(m(g)) for g in pruned]) for m in self.members
        ])  # [members, graphs]
        return preds.mean(0), preds.std(0) if len(self.members) > 1 else torch.zeros(len(graphs))

    def state_dict(self) -> dict:
        return {"members": [m.state_dict() for m in self.members]}
