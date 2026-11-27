"""
Environment for building SNN topologies with *macro* actions.

Two kinds of construction step (plus stop):

  NEW      create a hidden neuron with a receptive field: all input sources in a
           (2r+1)×(2r+1) window of patch cells centred on `a_cell`, for one or
           both polarities, connected to the new neuron with one shared weight.
           The neuron's (threshold, leak) comes from a preset.
  CONNECT  add one synapse a_first → a_second between existing neurons
           (tail: hidden; head: hidden or sink). Recurrent loops are legal when
           allow_cycles=True, otherwise the graph stays a DAG.

A topology therefore takes ~20–60 decisions instead of one per synapse.

Node index layout (fixed, never shifts):
    [0, S)          source neurons, S = 2·G·G (polarity-major over a G×G patch grid)
    [S, S+K)        sink neurons, one per class (pre-instantiated)
    [S+K, S+K+H)    hidden neurons, appended in creation order

Nodes and edges are only appended, so the state after t actions is a prefix of
the final graph: x[:S+K+h_t], edge_index[:, :e_t], edge_attr[:e_t]; trajectories
store h_t and e_t per step (`prefix_graph`).

Action dict (unused fields are -1)
----------------------------------
a_stop   : 0/1
a_kind   : 0 = NEW, 1 = CONNECT
a_cell   : NEW — receptive-field centre, gy·G + gx
a_radius : NEW — index into spec.rf_radii
a_pol    : NEW — 0 = OFF only, 1 = ON only, 2 = both polarities
a_node   : NEW — hidden preset index (threshold, leak)
a_first  : CONNECT — tail (hidden node index)
a_second : CONNECT — head (hidden or sink node index)
a_edge   : weight bin in [0, 2L) for the new synapse(s): [0, L) excitatory with
           magnitude weight_levels[b], [L, 2L) inhibitory

Actions must respect the masks (`masks()`, `rf_edge_table`, `second_mask`); an
illegal action raises.

Node features (9 columns)
-------------------------
0 threshold   1 leak   2 reset_state   3 refractory_period   4 node_type
5 pos_x       6 pos_y  7 polarity      8 sink_id (class index, -1 otherwise)

Edge features: [weight, delay]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch_geometric.data import Data

from spikegcpn.data.nmnist import NUM_CLASSES, num_source_nodes, source_positions
from spikegcpn.env.validity import connect_first_mask, second_mask

NEW, CONNECT = 0, 1

SOURCE: int = 0
SINK: int = 1
HIDDEN: int = 2

THR, LEAK, RESET, REFRAC, TYPE, PX, PY, POL, SINK_ID = range(9)
NODE_FEATURES = 9
EDGE_FEATURES = 2
NUM_POL_CHOICES = 3   # OFF, ON, both


@dataclass
class GraphSpec:
    """Everything that defines the search space. Shared by env, policy and rewards."""
    num_source: int | None = None    # default: num_source_nodes(patch_size); must be 2·G·G
    num_sink: int = NUM_CLASSES
    patch_size: int = 1              # N-MNIST pixels pooled per source (1 = raw pixels)
    max_hidden: int = 50
    max_edges: int = 100
    min_edges: int = 0               # stop is masked until this many edges exist (= max_edges: fixed size)
    allow_cycles: bool = False       # recurrent connections between hidden neurons (reservoirs)
    rf_radii: list[int] = field(default_factory=lambda: [0, 1, 2])
    # Synapse magnitudes, each offered as +w and -w. With BPTT inner training use
    # a single level: the policy then only picks the sign and gradients set |w|.
    weight_levels: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0])
    # (threshold, leak) presets for hidden neurons. SuperNeuroMAT leak = linear
    # decay toward reset per step: 0 = perfect integrator, inf = memoryless.
    hidden_presets: list[tuple[float, float]] = field(default_factory=lambda: [
        (0.5, float("inf")),
        (1.0, 0.0),
        (1.0, 0.25),
        (2.0, 0.5),
        (3.0, 0.1),
    ])
    source_threshold: float = 0.5
    source_leak: float = float("inf")
    sink_threshold: float = 1.0
    sink_leak: float = 0.0

    def __post_init__(self) -> None:
        if self.num_source is None:
            self.num_source = num_source_nodes(self.patch_size)
        g = math.isqrt(self.num_source // 2)
        if 2 * g * g != self.num_source:
            raise ValueError(f"num_source={self.num_source} must be 2·G·G (polarity × square patch grid)")
        self.grid = g
        self._rf_table: torch.Tensor | None = None

    @property
    def num_cells(self) -> int:
        return self.grid * self.grid

    @property
    def num_edge_actions(self) -> int:
        return 2 * len(self.weight_levels)

    @property
    def num_presets(self) -> int:
        return len(self.hidden_presets)

    @property
    def hidden_start(self) -> int:
        return self.num_source + self.num_sink

    def weight_of(self, a_edge: int) -> float:
        L = len(self.weight_levels)
        sign = 1.0 if a_edge < L else -1.0
        return sign * float(self.weight_levels[a_edge % L])

    def rf_sources(self, cell: int, radius_idx: int, pol: int) -> list[int]:
        """Sorted source indices of the receptive field."""
        G, r = self.grid, self.rf_radii[radius_idx]
        cy, cx = divmod(cell, G)
        cells = [y * G + x for y in range(max(0, cy - r), min(G, cy + r + 1))
                 for x in range(max(0, cx - r), min(G, cx + r + 1))]
        pols = [pol] if pol < 2 else [0, 1]
        return sorted(p * G * G + c for p in pols for c in cells)

    def rf_edge_table(self) -> torch.Tensor:
        """[cells, radii, 3] number of synapses each NEW choice would add."""
        if self._rf_table is None:
            t = torch.zeros(self.num_cells, len(self.rf_radii), NUM_POL_CHOICES, dtype=torch.long)
            for c in range(self.num_cells):
                for r in range(len(self.rf_radii)):
                    for p in range(NUM_POL_CHOICES):
                        t[c, r, p] = len(self.rf_sources(c, r, p))
            self._rf_table = t
        return self._rf_table

    def rf_multiplicity(self, sources: tuple[int, ...]) -> int:
        """How many (cell, radius, pol) choices produce exactly this receptive field."""
        if not hasattr(self, "_rf_mult"):
            from collections import Counter
            self._rf_mult = Counter(
                tuple(self.rf_sources(c, r, p))
                for c in range(self.num_cells) for r in range(len(self.rf_radii)) for p in range(NUM_POL_CHOICES)
            )
        return self._rf_mult[tuple(sources)]

    @classmethod
    def from_config(cls, env_cfg: dict) -> "GraphSpec":
        kw = {k: v for k, v in env_cfg.items() if k in cls.__dataclass_fields__}
        if "hidden_presets" in kw:
            kw["hidden_presets"] = [
                (float(p["threshold"]), float(p["leak"])) if isinstance(p, dict) else tuple(map(float, p))
                for p in kw["hidden_presets"]
            ]
        return cls(**kw)


@dataclass
class Masks:
    stop_allowed: bool
    continue_allowed: bool
    new_allowed: bool
    connect_allowed: bool
    remaining: int               # edge budget left
    first: torch.Tensor          # bool [n]: legal CONNECT tails


def masks(graph: Data, spec: GraphSpec, num_hidden: int) -> Masks:
    E = int(graph.edge_index.shape[1])
    remaining = spec.max_edges - E
    new_ok = num_hidden < spec.max_hidden and remaining >= 1
    first = connect_first_mask(graph.edge_index, spec.num_source, spec.num_sink, num_hidden, spec.allow_cycles)
    conn_ok = remaining >= 1 and bool(first.any())
    cont = new_ok or conn_ok
    return Masks(
        stop_allowed=(not cont) or E >= spec.min_edges,
        continue_allowed=cont,
        new_allowed=new_ok,
        connect_allowed=conn_ok,
        remaining=remaining,
        first=first,
    )


def second_mask_for(graph: Data, spec: GraphSpec, num_hidden: int, a_first: int) -> torch.Tensor:
    return second_mask(graph.edge_index, spec.num_source, spec.num_sink, num_hidden, a_first, spec.allow_cycles)


def initial_graph(spec: GraphSpec) -> Data:
    S, K = spec.num_source, spec.num_sink
    x = torch.zeros(S + K, NODE_FEATURES)
    x[:, SINK_ID] = -1.0

    x[:S, THR] = spec.source_threshold
    x[:S, LEAK] = spec.source_leak
    x[:S, TYPE] = SOURCE
    x[:S, PX:POL + 1] = torch.from_numpy(source_positions(S, spec.patch_size))

    x[S:, THR] = spec.sink_threshold
    x[S:, LEAK] = spec.sink_leak
    x[S:, TYPE] = SINK
    x[S:, SINK_ID] = torch.arange(K, dtype=torch.float32)

    return Data(
        x=x,
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, EDGE_FEATURES),
    )


def prefix_graph(final: Data, num_edges: int, num_hidden: int, spec: GraphSpec) -> Data:
    """State with `num_hidden` hidden neurons and the first `num_edges` edges of `final`."""
    return Data(
        x=final.x[: spec.hidden_start + num_hidden],
        edge_index=final.edge_index[:, :num_edges],
        edge_attr=final.edge_attr[:num_edges],
    )


STOP_ACTION = {"a_stop": 1, "a_kind": -1, "a_cell": -1, "a_radius": -1, "a_pol": -1,
               "a_node": -1, "a_first": -1, "a_second": -1, "a_edge": -1}


class SNNEnv:
    """Minimal stateful wrapper around the pure helpers above."""

    def __init__(self, spec: GraphSpec) -> None:
        self.spec = spec
        self._graph: Data | None = None
        self.num_hidden = 0

    @property
    def graph(self) -> Data:
        return self._graph

    @property
    def num_nodes(self) -> int:
        return self.spec.hidden_start + self.num_hidden

    @property
    def num_edges(self) -> int:
        return int(self._graph.edge_index.shape[1])

    def reset(self) -> Data:
        self.num_hidden = 0
        self._graph = initial_graph(self.spec)
        return self._graph

    def masks(self) -> Masks:
        return masks(self._graph, self.spec, self.num_hidden)

    def second_mask(self, a_first: int) -> torch.Tensor:
        return second_mask_for(self._graph, self.spec, self.num_hidden, a_first)

    def step(self, action: dict) -> tuple[Data, bool]:
        """Apply a legal action. Returns (graph, done)."""
        m = self.masks()
        if action["a_stop"]:
            if not m.stop_allowed:
                raise ValueError(f"stop not allowed before min_edges={self.spec.min_edges}")
            return self._graph, True
        if not m.continue_allowed:
            raise ValueError("episode must stop")

        spec = self.spec
        w = spec.weight_of(int(action["a_edge"]))
        if action["a_kind"] == NEW:
            if not m.new_allowed:
                raise ValueError("NEW not allowed")
            srcs = spec.rf_sources(int(action["a_cell"]), int(action["a_radius"]), int(action["a_pol"]))
            if len(srcs) > m.remaining:
                raise ValueError("receptive field exceeds the edge budget")
            v = self._add_hidden_node(*spec.hidden_presets[int(action["a_node"])])
            self._add_edges(srcs, [v] * len(srcs), w)
        elif action["a_kind"] == CONNECT:
            u, v = int(action["a_first"]), int(action["a_second"])
            if not m.connect_allowed or not bool(m.first[u]) or not bool(self.second_mask(u)[v]):
                raise ValueError(f"illegal synapse {u}->{v}")
            self._add_edges([u], [v], w)
        else:
            raise ValueError(f"unknown a_kind {action['a_kind']}")
        return self._graph, False

    def _add_hidden_node(self, threshold: float, leak: float) -> int:
        feat = torch.zeros(1, NODE_FEATURES)
        feat[0, THR] = threshold
        feat[0, LEAK] = leak
        feat[0, TYPE] = HIDDEN
        feat[0, SINK_ID] = -1.0
        idx = self.num_nodes
        self._graph = Data(
            x=torch.cat([self._graph.x, feat], dim=0),
            edge_index=self._graph.edge_index,
            edge_attr=self._graph.edge_attr,
        )
        self.num_hidden += 1
        return idx

    def _add_edges(self, src: list[int], dst: list[int], weight: float, delay: int = 1) -> None:
        k = len(src)
        self._graph = Data(
            x=self._graph.x,
            edge_index=torch.cat([self._graph.edge_index, torch.tensor([src, dst], dtype=torch.long)], dim=1),
            edge_attr=torch.cat([self._graph.edge_attr, torch.tensor([[weight, float(delay)]] * k)], dim=0),
        )
