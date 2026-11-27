"""
Hand-designed reservoirs that each isolate one design choice, scored with the
same reservoir pipeline as the search (fixed weights, logistic readout on
windowed spike counts), next to input-readout baselines and the searched graphs.

  python scripts/probe_reservoirs.py --tag topo1 [--config configs/reservoir.yaml]

Answers "what makes a good reservoir for N-MNIST?" by ablation rather than by
reading the searched graphs: where the receptive fields go (informative vs
random cells), how polarities are pooled, which neuron dynamics, field size,
and whether random recurrence on top helps. Writes experiments/<tag>/probes.yaml.

Cell informativeness (and therefore every "top cells" design) is ranked on the
validation split, which is also what the search optimises, so only the Test
column is an unbiased comparison.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spikegcpn.data.nmnist import load_samples  # noqa: E402
from spikegcpn.env.snn_env import CONNECT, NEW, STOP_ACTION, GraphSpec, SNNEnv  # noqa: E402
from spikegcpn.train.inner import split_per_class, stack_samples  # noqa: E402
from spikegcpn.train.readout import ReadoutConfig, ReservoirEvaluator, fit_readout, window_counts  # noqa: E402
from topology_evolution import cell_informativeness  # noqa: E402

INTEGRATOR, COINCIDENCE, LEAKY, SLOW = 1, 0, 2, 4      # indices into reservoir.yaml hidden_presets


def build(spec: GraphSpec, fields: list[tuple[int, int, int, int, int]], recurrent: int = 0, seed: int = 0):
    """fields: (cell, radius_idx, pol, preset, weight_bin); then `recurrent` random hidden→hidden synapses."""
    env = SNNEnv(spec)
    env.reset()
    for c, r, p, node, w in fields:
        env.step(dict(STOP_ACTION, a_stop=0, a_kind=NEW, a_cell=c, a_radius=r, a_pol=p, a_node=node, a_edge=w))
    rng = random.Random(seed)
    for _ in range(recurrent):
        m = env.masks()
        if not m.connect_allowed:
            break
        u = rng.choice(m.first.nonzero(as_tuple=True)[0].tolist())
        heads = [v for v in env.second_mask(u).nonzero(as_tuple=True)[0].tolist() if v >= spec.hidden_start]
        if not heads:
            continue
        L = len(spec.weight_levels)
        w = rng.choice([1, 1 + L])                      # ±0.5
        env.step(dict(STOP_ACTION, a_stop=0, a_kind=CONNECT, a_first=u, a_second=rng.choice(heads), a_edge=w))
    return env.graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", default="configs/reservoir.yaml")
    ap.add_argument("--test-per-class", type=int, default=300,
                    help="Test samples per class (the search's summary.yaml uses 50, i.e. SE ≈ ±0.015)")
    args = ap.parse_args()
    torch.set_num_threads(1)

    cfg = yaml.safe_load(open(args.config))
    spec = GraphSpec.from_config(cfg["env"])
    s, i = cfg["simulator"], cfg["inner"]
    root, bins, dseed = s.get("nmnist_root", "data/NMNIST"), s["num_bins"], s.get("data_seed", 0)
    pool = load_samples(root, "Train", bins, i["train_per_class"] + s["val_per_class"], None, True, dseed, spec.patch_size)
    tr, va = split_per_class(pool, i["train_per_class"])
    te = load_samples(root, "Test", bins, args.test_per_class, None, True, dseed, spec.patch_size)
    sets = {"train": stack_samples(tr), "val": stack_samples(va), "test": stack_samples(te)}
    rc = ReadoutConfig(windows=i.get("windows", 5), readout=i.get("readout", "logistic"),
                       l2=i.get("l2", 1e-3), max_iter=i.get("max_iter", 200))
    ev = {k: ReservoirEvaluator(spec.num_source, spec.num_sink, sets["train"], sets[k], rc, i.get("extra_ticks", 10))
          for k in ("val", "test")}

    results: dict[str, dict] = {}

    def score_graph(name: str, g, note: str = "") -> None:
        accs = {k: e.train_and_eval([g])[0][0] for k, e in ev.items()}
        H = g.num_nodes - spec.hidden_start
        results[name] = {"val": round(accs["val"], 4), "test": round(accs["test"], 4),
                         "neurons": int(H), "synapses": int(g.edge_index.shape[1]), "note": note}
        print(f"{name:42s} val {accs['val']:.3f} test {accs['test']:.3f}  H={H:3d} E={g.edge_index.shape[1]:3d}  {note}",
              flush=True)

    # --- input-readout baselines (features = input-channel window counts) ---------------
    W, G2 = rc.windows, spec.num_cells
    F = {k: window_counts(X, W) for k, (X, _) in sets.items()}
    y = {k: v[1] for k, v in sets.items()}

    def score_channels(name: str, ch: list[int], note: str) -> None:
        cols = torch.tensor([c * W + w for c in ch for w in range(W)])
        r = fit_readout(F["train"][:, cols], y["train"], spec.num_sink, rc)
        accs = {k: float((r.predict(F[k][:, cols]) == y[k]).double().mean()) for k in ("val", "test")}
        results[name] = {"val": round(accs["val"], 4), "test": round(accs["test"], 4),
                         "channels": len(ch), "note": note}
        print(f"{name:42s} val {accs['val']:.3f} test {accs['test']:.3f}  C={len(ch):3d}        {note}", flush=True)

    info = cell_informativeness(cfg, spec)                 # single-cell readout accuracy on val
    ranked = [int(c) for c in np.argsort(-info)]
    n_match = spec.max_hidden                              # neurons a searched reservoir can have
    score_channels("input: all 162 channels", list(range(spec.num_source)), "810 features")
    score_channels("input: 60 random channels", sorted(random.Random(0).sample(range(spec.num_source), 60)),
                   "the 'matched' baseline in summary.yaml")
    score_channels(f"input: top-{n_match // 2} cells × 2 pol", ranked[:n_match // 2] + [c + G2 for c in ranked[:n_match // 2]],
                   "informative cells, polarities separate")
    pooled = {}
    for k in sets:   # both polarities of a cell summed into one channel
        X = sets[k][0]
        pooled[k] = window_counts(X[:, :, :G2].int() + X[:, :, G2:].int(), W)
    cols = torch.tensor([c * W + w for c in ranked[:n_match] for w in range(W)])
    r = fit_readout(pooled["train"][:, cols], y["train"], spec.num_sink, rc)
    name = f"input: top-{n_match} cells, pol summed"
    accs = {k: float((r.predict(pooled[k][:, cols]) == y[k]).double().mean()) for k in ("val", "test")}
    results[name] = {"val": round(accs["val"], 4), "test": round(accs["test"], 4),
                     "channels": n_match, "note": "no neuron, just summed counts"}
    print(f"{name:42s} val {accs['val']:.3f} test {accs['test']:.3f}  C={n_match:3d}        no neuron", flush=True)

    # --- hand-designed reservoirs ------------------------------------------------------
    W1 = 2                                    # weight bin: +1.0
    BOTH, OFF, ON = 2, 0, 1
    top, rnd = ranked[:n_match], sorted(random.Random(0).sample(range(G2), n_match))
    score_graph("1×1 both, integrator, top cells", build(spec, [(c, 0, BOTH, INTEGRATOR, W1) for c in top]),
                "baseline design")
    score_graph("1×1 both, integrator, random cells", build(spec, [(c, 0, BOTH, INTEGRATOR, W1) for c in rnd]),
                "placement")
    score_graph("1×1 OFF+ON separate, integrator, top", build(spec, [(c, 0, p, INTEGRATOR, W1)
                                                                    for c in top[:n_match // 2] for p in (OFF, ON)]),
                "polarity pooling")
    score_graph("1×1 both, coincidence, top cells", build(spec, [(c, 0, BOTH, COINCIDENCE, W1) for c in top]),
                "dynamics")
    score_graph("1×1 both, leaky (1, .25), top cells", build(spec, [(c, 0, BOTH, LEAKY, W1) for c in top]),
                "dynamics")
    score_graph("1×1 both, slow (3, .1), top cells", build(spec, [(c, 0, BOTH, SLOW, W1) for c in top]),
                "dynamics")
    score_graph("1×1 both, integrator+coinc., top 25", build(spec, [(c, 0, BOTH, n, W1)
                                                                    for c in top[:n_match // 2]
                                                                    for n in (INTEGRATOR, COINCIDENCE)]),
                "mixed dynamics, same cell twice")
    score_graph("1×1 both, integrator |w|=0.5, top", build(spec, [(c, 0, BOTH, INTEGRATOR, 1) for c in top]),
                "weight magnitude")
    score_graph("1×1 both, integrator |w|=2, top", build(spec, [(c, 0, BOTH, INTEGRATOR, 3) for c in top]),
                "weight magnitude")
    G = spec.grid
    tiles = [y_ * G + x_ for y_ in range(1, G, 3) for x_ in range(1, G, 3)]
    score_graph("3×3 both, integrator, 9 tiles", build(spec, [(c, 1, BOTH, INTEGRATOR, W1) for c in tiles]),
                "field size (budget allows few)")
    score_graph("1×1 both + 200 random recurrent", build(spec, [(c, 0, BOTH, INTEGRATOR, W1) for c in top],
                                                         recurrent=200), "recurrence")
    score_graph("1×1 both + 200 random recurrent (s1)", build(spec, [(c, 0, BOTH, INTEGRATOR, W1) for c in top],
                                                              recurrent=200, seed=1), "recurrence")

    # --- the searched graphs ----------------------------------------------------------
    for run in sorted((Path("experiments") / args.tag).glob("grpo*/best_graph.pt")):
        score_graph(f"searched: {run.parent.name}", torch.load(run, weights_only=False), "GRPO best")

    out = Path("experiments") / args.tag / "probes.yaml"
    out.write_text(yaml.safe_dump({"cell_rank": ranked, "cell_informativeness": info.round(4).tolist(),
                                   "results": results}, sort_keys=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
