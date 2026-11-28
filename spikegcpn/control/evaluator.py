"""
Closed-loop control evaluation of a spiking reservoir (inner.mode: control).

  masked obs (x, θ) ──population code──► sources ──► searched reservoir (state persists
  across env steps) ──► read-out neurons' spike counts during this env step ──► linear
  readout ──► action

Each env step lasts `ticks_per_step` simulator ticks with the current
observation's code held on the sources. Per topology:

1. Behaviour cloning. Fixed-seed episodes are driven by the full-state expert
   with DART-style noise (a random action with probability `bc_noise`, label =
   the expert's action), so the data covers recoveries from small mistakes.
   These rollouts do not depend on the reservoir, so they are precomputed once
   and every reservoir is simulated open-loop over the same input trains.
   A logistic readout is fit from the read-out neurons' per-step counts to
   the expert's actions.
2. Closed loop. The reservoir + readout then controls its own environments
   for `eval_episodes` fixed seeds; the reward is the mean return / max_steps.

Everything is seeded, so the reward is a deterministic function of the
topology (the oracle cache relies on this). Several topologies run in
parallel, each controlling its own copy of every episode (LIFBatch.step with
per-graph inputs).

The expert needs the hidden velocities; a policy on the visible (x, θ) alone
cannot balance the pole (best memoryless linear policy ≈ 47 / 500), so a
reservoir only scores well if it supplies memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

from spikegcpn.control import cartpole as cp
from spikegcpn.data.nmnist import NMNISTSample
from spikegcpn.simulator.torch_lif import LIFBatch
from spikegcpn.train.readout import Readout, ReadoutConfig, fit_readout


@dataclass
class ControlConfig:
    ticks_per_step: int = 4
    max_steps: int = cp.MAX_STEPS
    bc_episodes: int = 16
    bc_noise: float = 0.2
    eval_episodes: int = 10
    test_episodes: int = 100
    readout_neurons: str = "outputs"     # outputs (sinks only) | all
    l2: float = 1e-3
    max_iter: int = 200
    seed: int = 0                        # BC noise; episode seeds are fixed ranges below
    max_graphs_per_batch: int = 16


BC_SEEDS, EVAL_SEEDS, TEST_SEEDS = 1000, 2000, 3000


class ControlEvaluator:
    def __init__(self, num_source: int, num_sink: int, cfg: ControlConfig):
        if num_source % 2:
            raise ValueError("control sources are 2 × bins (x code, then θ code)")
        self.S, self.K, self.cfg = num_source, num_sink, cfg
        self.bins = num_source // 2
        self.rcfg = ReadoutConfig(l2=cfg.l2, max_iter=cfg.max_iter)
        states, labels, alive = self._expert_rollouts()
        self.bc_states, self.bc_labels, self.bc_alive = states, labels, alive
        enc = torch.from_numpy(cp.encode(states, self.bins)).permute(1, 0, 2)      # [E, steps, S]
        self.bc_inputs = enc.repeat_interleave(cfg.ticks_per_step, dim=1)          # [E, steps·k, S]

    # ------------------------------------------------------------------
    def _expert_rollouts(self):
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        E, T = cfg.bc_episodes, cfg.max_steps
        state = cp.reset(BC_SEEDS + np.arange(E))
        alive = np.ones(E, dtype=bool)
        states, labels, alive_t = np.zeros((T, E, 4)), np.zeros((T, E), dtype=np.int64), np.zeros((T, E), dtype=bool)
        for t in range(T):
            states[t], labels[t], alive_t[t] = state, cp.expert(state), alive
            act = np.where(rng.random(E) < cfg.bc_noise, rng.integers(0, 2, E), labels[t])
            nxt, done = cp.step(state, act)
            alive &= ~done
            state = np.where(alive[:, None], nxt, state)
        return states, labels, alive_t

    def _read(self, feats: torch.Tensor, sl: slice) -> torch.Tensor:
        """[B, total_ns, ...] → this graph's read-out neurons."""
        f = feats[:, sl]
        return f[:, :self.K] if self.cfg.readout_neurons == "outputs" else f

    @torch.no_grad()
    def fit(self, graphs: list[Data]) -> list[Readout]:
        """Behaviour-cloned readout per graph (open-loop over the expert rollouts)."""
        T = self.cfg.max_steps
        net = LIFBatch(graphs, self.S, self.K, extra_ticks=0, fixed=True)
        _, win = net(self.bc_inputs, windows=T)                                    # [E, total_ns, T]
        mask = torch.from_numpy(self.bc_alive.T.reshape(-1))                       # [E·T]
        y = torch.from_numpy(self.bc_labels.T.reshape(-1))[mask]
        out = []
        for sl in net.ns_slices:
            f = self._read(win, sl).permute(0, 2, 1).reshape(-1, self._read(win, sl).shape[1])[mask]
            out.append(fit_readout(f, y, 2, self.rcfg))
        return out

    @torch.no_grad()
    def closed_loop(self, graphs: list[Data], readouts: list[Readout], seeds: np.ndarray, record: bool = False):
        """Returns [G, E] episode returns (and, with record, per-tick source inputs and per-step features)."""
        cfg, G, E = self.cfg, len(graphs), len(seeds)
        net = LIFBatch(graphs, self.S, self.K, extra_ticks=0, fixed=True)
        st = net.init_state(E)
        state = np.repeat(cp.reset(seeds)[:, None], G, axis=1)                     # [E, G, 4]
        alive = np.ones((E, G), dtype=bool)
        ret = np.zeros((E, G))
        inputs, feats_log = [], []
        for _ in range(cfg.max_steps):
            x = torch.from_numpy(cp.encode(state, self.bins))                      # [E, G, S]
            acc = torch.zeros(E, net.total_ns, dtype=torch.float64)
            for _ in range(cfg.ticks_per_step):
                acc += net.step(st, x)
                if record:
                    inputs.append(x[:, 0].clone())
            act = np.stack([readouts[g].predict(self._read(acc, sl)).numpy()
                            for g, sl in enumerate(net.ns_slices)], axis=1)       # [E, G]
            if record:
                feats_log.append(self._read(acc, net.ns_slices[0]).clone())
            nxt, done = cp.step(state, act)
            ret += alive
            alive &= ~done
            state = np.where(alive[..., None], nxt, state)
            if not alive.any() and not record:
                break
        if record:
            return ret.T, torch.stack(inputs, 1), torch.stack(feats_log, 2)        # [E, T·k, S], [E, M, T]
        return ret.T

    def train_and_eval(self, graphs: list[Data]) -> tuple[list[float], list[Readout]]:
        """Normalised mean closed-loop return over the eval episodes, per graph."""
        scores, readouts = [], []
        seeds = EVAL_SEEDS + np.arange(self.cfg.eval_episodes)
        step = self.cfg.max_graphs_per_batch
        for i in range(0, len(graphs), step):
            chunk = graphs[i:i + step]
            ro = self.fit(chunk)
            ret = self.closed_loop(chunk, ro, seeds)
            scores += (ret.mean(1) / self.cfg.max_steps).tolist()
            readouts += ro
        return scores, readouts

    def test(self, graph: Data, readout: Readout) -> float:
        """Mean return on the held-out test episodes."""
        return float(self.closed_loop([graph], [readout], TEST_SEEDS + np.arange(self.cfg.test_episodes)).mean())

    # ------------------------------------------------------------------
    def verify_superneuromat(self, graph: Data, readout: Readout, episodes: int = 3) -> dict:
        """
        Re-run closed-loop episodes' input trains through SuperNeuroMAT. If its
        per-step read-out counts equal torch's, the actions (hence returns) are
        identical, i.e. the controller behaves the same on SuperNeuroMAT.
        """
        from spikegcpn.simulator.wrapper import window_spike_counts
        ret, inputs, feats = self.closed_loop([graph], [readout], TEST_SEEDS + np.arange(episodes), record=True)
        T = self.cfg.max_steps
        samples = [NMNISTSample(spike_train=inputs[e].numpy().astype(bool), label=0) for e in range(episodes)]
        snm = window_spike_counts(graph, samples, self.S, self.K, 0, T)           # [E, ns, T]
        snm = snm[:, :self.K] if self.cfg.readout_neurons == "outputs" else snm
        return {"superneuromat_match": bool(np.array_equal(snm, feats.numpy().astype(np.int64))),
                "superneuromat_check_episodes": episodes}

    # ------------------------------------------------------------------
    def baselines(self) -> dict:
        """Expert, random, and input readouts (memoryless, and with one step of memory) on the eval and test seeds."""
        cfg = self.cfg
        out = {"expert_return": 500.0 if cfg.max_steps == 500 else float("nan")}

        def run(policy, seeds):
            state = cp.reset(seeds)
            prev = state.copy()
            alive, ret = np.ones(len(seeds), dtype=bool), np.zeros(len(seeds))
            rng = np.random.default_rng(0)
            for _ in range(cfg.max_steps):
                a = policy(state, prev, rng)
                nxt, done = cp.step(state, a)
                ret += alive
                alive &= ~done
                prev, state = state, np.where(alive[:, None], nxt, state)
                if not alive.any():
                    break
            return float(ret.mean())

        test_seeds = TEST_SEEDS + np.arange(cfg.test_episodes)
        out["expert_return"] = run(lambda s, p, r: cp.expert(s), test_seeds)
        out["random_return"] = run(lambda s, p, r: r.integers(0, 2, len(s)), test_seeds)
        S, A = self.bc_states, self.bc_alive
        prev = np.concatenate([S[:1], S[:-1]])                                     # previous step's state
        y = torch.from_numpy(self.bc_labels[A])
        for name, mem in (("memoryless", False), ("two_step", True)):
            def feats(s, p):
                f = cp.encode(s, self.bins).astype(np.float64)
                return np.concatenate([f, cp.encode(p, self.bins)], -1) if mem else f
            ro = fit_readout(torch.from_numpy(feats(S, prev)[A]), y, 2, self.rcfg)
            policy = lambda s, p, r, ro=ro, feats=feats: ro.predict(torch.from_numpy(feats(s, p))).numpy()
            out[f"input_readout_{name}_return"] = run(policy, test_seeds)
        return out
