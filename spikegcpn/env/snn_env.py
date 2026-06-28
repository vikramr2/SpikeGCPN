"""
Gymnasium environment for building SNN topologies as DAGs.

Structural invariants maintained at every step:
  - Graph is always a DAG.
  - Source nodes (indices [0, num_source)) are pre-instantiated at reset
    and never removed.
  - Sink nodes (indices [num_nodes-num_sink, num_nodes)) are pre-instantiated
    at reset and never removed.
  - Hidden nodes are inserted between source and sink blocks; sink indices
    shift upward by 1 each time a hidden node is added.

Action dict keys
----------------
a_first  : int  — index of the edge's tail node
a_second : int  — index of the edge's head node; pass NEW_NODE to add a
                  new hidden neuron first (the new node's index is returned
                  in info['new_node_idx'])
a_edge   : int  — 0 = excitatory (weight +1), 1 = inhibitory (weight -1)
a_stop   : bool — terminate the episode

Observation
-----------
The current PyG Data object (data.x, data.edge_index, data.edge_attr).
A gym spaces.Graph space is declared for compatibility, but agents
typically consume the Data object directly.

Node features  [threshold, leak, reset_state, refractory_period, node_type]
Edge features  [weight, delay]

Node types: SOURCE=0, HIDDEN=1, SINK=2
"""

from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from torch_geometric.data import Data

from spikegcpn.env.validity import would_create_cycle, all_sinks_reachable
from spikegcpn.data.nmnist import NUM_SOURCE_NODES, NUM_CLASSES

# Sentinel value for a_second meaning "add a new hidden node".
NEW_NODE: int = -1

# Node-type constants (stored in node feature column 4).
SOURCE: int = 0
HIDDEN: int = 1
SINK: int = 2

_NODE_FEATURES = 5   # threshold, leak, reset_state, refractory_period, node_type
_EDGE_FEATURES = 2   # weight, delay


class SNNEnv(gym.Env):
    """
    Parameters
    ----------
    num_source        : number of input (source) neurons; defaults to N-MNIST
    num_sink          : number of output (sink) neurons; defaults to 10 classes
    max_hidden        : hard cap on hidden neurons per episode
    default_threshold : LIF threshold applied to all neurons at reset
    default_leak      : LIF leak applied to all neurons at reset
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_source: int = NUM_SOURCE_NODES,
        num_sink: int = NUM_CLASSES,
        max_hidden: int = 200,
        default_threshold: float = 0.5,
        default_leak: float = float("inf"),
    ) -> None:
        super().__init__()
        self.num_source = num_source
        self.num_sink = num_sink
        self.max_hidden = max_hidden
        self.default_threshold = default_threshold
        self.default_leak = default_leak

        max_nodes = num_source + max_hidden + num_sink
        self.observation_space = spaces.Graph(
            node_space=spaces.Box(low=-np.inf, high=np.inf, shape=(_NODE_FEATURES,)),
            edge_space=spaces.Box(low=-np.inf, high=np.inf, shape=(_EDGE_FEATURES,)),
        )
        self.action_space = spaces.Dict({
            "a_first":  spaces.Discrete(max_nodes),
            "a_second": spaces.Discrete(max_nodes + 1, start=-1),  # -1 = NEW_NODE
            "a_edge":   spaces.Discrete(2),   # 0=excitatory, 1=inhibitory
            "a_stop":   spaces.Discrete(2),   # 0=continue,   1=stop
        })

        self._graph: Data | None = None
        self._num_hidden: int = 0

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def graph(self) -> Data:
        return self._graph

    @property
    def num_nodes(self) -> int:
        return self.num_source + self._num_hidden + self.num_sink

    @property
    def sink_start_idx(self) -> int:
        return self.num_source + self._num_hidden

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._num_hidden = 0
        self._graph = self._make_initial_graph()
        return self._obs(), {}

    # ------------------------------------------------------------------
    # Action masks
    # ------------------------------------------------------------------

    def valid_first_mask(self) -> "torch.Tensor":
        """
        Boolean mask over existing nodes for a_first (edge source).
        Sink nodes must have out-degree 0, so they are excluded.
        Valid: source nodes [0, num_source) and hidden nodes [num_source, sink_start).
        """
        n = self.num_nodes
        mask = torch.ones(n, dtype=torch.bool)
        mask[self.sink_start_idx:] = False  # exclude sink nodes
        return mask

    def valid_second_mask(self) -> "torch.Tensor":
        """
        Boolean mask over existing nodes for a_second (edge destination).
        Source nodes must have in-degree 0, so they are excluded.
        Valid: hidden [num_source, sink_start) and sink [sink_start, n).
        The extra NEW_NODE option is always valid and is NOT covered by this mask.
        """
        n = self.num_nodes
        mask = torch.ones(n, dtype=torch.bool)
        mask[:self.num_source] = False  # exclude source nodes
        return mask

    # ------------------------------------------------------------------

    def step(self, action: dict):
        a_stop = bool(action["a_stop"])

        if a_stop:
            terminated = True
            reward = 0.0
            info: dict = {"graph": self._graph}
            return self._obs(), reward, terminated, False, info

        a_first = int(action["a_first"])
        a_second = int(action["a_second"])
        a_edge = int(action["a_edge"])

        # --- Validate node indices ----------------------------------------
        cur_nodes = self.num_nodes
        if not (0 <= a_first < cur_nodes):
            return self._obs(), -0.1, False, False, {"invalid": "a_first out of range"}

        # Enforce structural constraint: cannot draw an edge FROM a sink node
        if a_first >= self.sink_start_idx:
            return self._obs(), -0.1, False, False, {"invalid": "a_first is a sink node"}

        new_node_idx: int | None = None

        # --- Optionally add a new hidden neuron ---------------------------
        if a_second == NEW_NODE:
            if self._num_hidden >= self.max_hidden:
                return self._obs(), -0.1, False, False, {"invalid": "max_hidden reached"}
            new_node_idx = self._add_hidden_node()
            a_second = new_node_idx
            cur_nodes = self.num_nodes  # updated after insertion

        if not (0 <= a_second < cur_nodes):
            return self._obs(), -0.1, False, False, {"invalid": "a_second out of range"}

        # Enforce structural constraint: cannot draw an edge TO a source node
        if a_second < self.num_source:
            return self._obs(), -0.1, False, False, {"invalid": "a_second is a source node"}

        # --- Reject duplicate edges ---------------------------------------
        if self._edge_exists(a_first, a_second):
            return self._obs(), -0.01, False, False, {"invalid": "duplicate edge"}

        # --- Check DAG constraint -----------------------------------------
        if would_create_cycle(self._graph.edge_index, cur_nodes, a_first, a_second):
            return self._obs(), -0.01, False, False, {"invalid": "would create cycle"}

        # --- Add edge -----------------------------------------------------
        weight = 1.0 if a_edge == 0 else -1.0
        self._add_edge(a_first, a_second, weight=weight, delay=1)

        # Small validity reward for each legal edge.
        reward = 0.001
        info = {"graph": self._graph}
        if new_node_idx is not None:
            info["new_node_idx"] = new_node_idx

        return self._obs(), reward, False, False, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs(self):
        return self._graph

    def _make_initial_graph(self) -> Data:
        """Pre-instantiate source and sink nodes with empty edge sets."""
        n = self.num_source + self.num_sink
        x = torch.zeros(n, _NODE_FEATURES)

        # Source nodes
        x[: self.num_source, 0] = self.default_threshold
        x[: self.num_source, 1] = self.default_leak
        x[: self.num_source, 4] = SOURCE

        # Sink nodes
        x[self.num_source :, 0] = self.default_threshold
        x[self.num_source :, 1] = self.default_leak
        x[self.num_source :, 4] = SINK

        return Data(
            x=x,
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, _EDGE_FEATURES),
        )

    def _add_hidden_node(self) -> int:
        """
        Insert a hidden node at position sink_start_idx, shifting all sink
        indices in edge_index up by 1. Returns the new node's index.
        """
        insert_at = self.sink_start_idx  # before current sinks

        new_feat = torch.zeros(1, _NODE_FEATURES)
        new_feat[0, 0] = self.default_threshold
        new_feat[0, 1] = self.default_leak
        new_feat[0, 4] = HIDDEN

        x = torch.cat(
            [self._graph.x[:insert_at], new_feat, self._graph.x[insert_at:]], dim=0
        )

        # Shift any existing edge endpoints that pointed at sink nodes.
        ei = self._graph.edge_index.clone()
        ei[ei >= insert_at] += 1

        self._graph = Data(x=x, edge_index=ei, edge_attr=self._graph.edge_attr)
        self._num_hidden += 1
        return insert_at

    def _edge_exists(self, src: int, dst: int) -> bool:
        """Return True if edge src→dst is already present."""
        ei = self._graph.edge_index
        if ei.numel() == 0:
            return False
        return bool(((ei[0] == src) & (ei[1] == dst)).any().item())

    def _add_edge(self, src: int, dst: int, weight: float, delay: int) -> None:
        new_ei = torch.tensor([[src], [dst]], dtype=torch.long)
        new_ea = torch.tensor([[weight, float(delay)]])
        self._graph = Data(
            x=self._graph.x,
            edge_index=torch.cat([self._graph.edge_index, new_ei], dim=1),
            edge_attr=torch.cat([self._graph.edge_attr, new_ea], dim=0),
        )
