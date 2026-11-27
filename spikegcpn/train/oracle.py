"""
Reward oracle: decides, per batch of finished topologies, which ones get a
real (expensive) simulator evaluation and which are scored by the surrogate.

  * Exact cache: a topology already simulated is never simulated again.
  * Warm-up: until `warmup` graphs have been simulated, everything is simulated.
  * After warm-up: the `sim_per_batch` uncached graphs with the highest
    UCB = mean + kappa * std (ensemble spread) are simulated; the rest receive
    the surrogate mean. The surrogate is refit every `refit_every` new points.
  * surrogate=None ⇒ every graph is simulated (the no-surrogate baseline).

`simulate` takes a list of graphs and returns their accuracies, so a whole
batch can be evaluated together (the inner trainer trains them jointly).

The comparison metric across algorithms is accuracy vs. `sim_calls`, since
simulator calls are the budget that matters at scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch_geometric.data import Data

from spikegcpn.model.surrogate import SurrogateEnsemble


@dataclass
class Score:
    accuracy: float
    source: str   # "sim" | "cache" | "surrogate"


class RewardOracle:
    def __init__(
        self,
        simulate: Callable[[list[Data]], list[float]],
        surrogate: SurrogateEnsemble | None = None,
        warmup: int = 64,
        sim_per_batch: int = 4,
        ucb_kappa: float = 1.0,
        refit_every: int = 16,
        refit_steps: int = 50,
    ):
        self.simulate = simulate
        self.surrogate = surrogate
        self.warmup = warmup
        self.sim_per_batch = sim_per_batch
        self.ucb_kappa = ucb_kappa
        self.refit_every = refit_every
        self.refit_steps = refit_steps

        self.cache: dict[str, float] = {}
        self.graphs: dict[str, Data] = {}
        self.sim_calls = 0
        self._since_refit = 0
        self.last_fit_mse = float("nan")

    def _run_sims(self, keys: list[str], graphs: list[Data]) -> list[float]:
        accs = self.simulate(graphs) if graphs else []
        for k, g, acc in zip(keys, graphs, accs):
            self.sim_calls += 1
            self.cache[k] = acc
            self.graphs[k] = g
            if self.surrogate is not None:
                self.surrogate.add(g, acc)
                self._since_refit += 1
        return accs

    def score(self, graphs: list[Data], keys: list[str], allow_sim: bool = True) -> list[Score]:
        out: list[Score | None] = [None] * len(graphs)
        pending: dict[str, list[int]] = {}
        for i, k in enumerate(keys):
            if k in self.cache:
                out[i] = Score(self.cache[k], "cache")
            else:
                pending.setdefault(k, []).append(i)

        uniq = list(pending)
        sim_keys: set[str] = set()
        if allow_sim:
            if self.surrogate is None or len(self.surrogate) < self.warmup:
                sim_keys = set(uniq)
            elif uniq:
                mean, std = self.surrogate.predict([graphs[pending[k][0]] for k in uniq])
                ucb = mean + self.ucb_kappa * std
                order = ucb.argsort(descending=True).tolist()
                sim_keys = {uniq[j] for j in order[: self.sim_per_batch]}

        surrogate_keys = [k for k in uniq if k not in sim_keys]
        sim_list = [k for k in uniq if k in sim_keys]
        for k, acc in zip(sim_list, self._run_sims(sim_list, [graphs[pending[k][0]] for k in sim_list])):
            for i in pending[k]:
                out[i] = Score(acc, "sim")

        if surrogate_keys:
            if self.surrogate is None or len(self.surrogate) == 0:
                raise RuntimeError("no surrogate available to score unsimulated graphs")
            mean, _ = self.surrogate.predict([graphs[pending[k][0]] for k in surrogate_keys])
            for k, m in zip(surrogate_keys, mean.tolist()):
                for i in pending[k]:
                    out[i] = Score(m, "surrogate")

        if self.surrogate is not None and self._since_refit >= self.refit_every:
            self.last_fit_mse = self.surrogate.fit(self.refit_steps)
            self._since_refit = 0
        return out  # type: ignore[return-value]

    def best(self) -> tuple[str, float] | None:
        if not self.cache:
            return None
        k = max(self.cache, key=self.cache.get)
        return k, self.cache[k]
