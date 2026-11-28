"""
How did the sampled reservoir topologies change while GRPO trained?

  python scripts/topology_evolution.py --tag topo1 [--bin 10]

Reads experiments/<tag>/<run>/samples.pt (every sampled trajectory's actions and
accuracy per iteration, written by scripts/train.py) for every run in the tag,
computes per-graph statistics straight from the macro actions, averages them
over bins of iterations and seeds, and writes to experiments/<tag>/topology/:

  evolution.png   (evolution_<variant>.png when the tag has several variants)
                  accuracy, composition, receptive-field size / polarity,
                  neuron presets, weights, CONNECT usage, group similarity
  cells.png       where the receptive fields are placed (early / mid / late)
                  next to how informative each input cell is on its own
  stats.json      the binned numbers behind both figures
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spikegcpn.data.nmnist import NUM_CLASSES, load_samples  # noqa: E402
from spikegcpn.env.snn_env import CONNECT, NEW, GraphSpec  # noqa: E402
from spikegcpn.env.validity import is_dag  # noqa: E402
from spikegcpn.train.inner import split_per_class, stack_samples  # noqa: E402
from spikegcpn.train.readout import ReadoutConfig, fit_readout, window_counts  # noqa: E402

POL_NAMES = ["OFF", "ON", "both"]


def graph_stats(acts: np.ndarray, keys: list[str], spec: GraphSpec) -> dict:
    """Statistics of one finished topology, from its action sequence [T, len(keys)]."""
    k = {name: i for i, name in enumerate(keys)}
    steps = acts[acts[:, k["a_stop"]] == 0]
    new = steps[steps[:, k["a_kind"]] == NEW]
    conn = steps[steps[:, k["a_kind"]] == CONNECT]
    L = len(spec.weight_levels)
    table = spec.rf_edge_table()
    S, K, h0 = spec.num_source, spec.num_sink, spec.hidden_start

    rf_sizes = [int(table[c, r, p]) for c, r, p in new[:, [k["a_cell"], k["a_radius"], k["a_pol"]]]]
    covered = np.zeros(spec.num_cells, dtype=bool)
    cell_use = np.zeros(spec.num_cells)            # receptive-field mass per cell (each RF spreads 1 unit)
    for c, r, p in new[:, [k["a_cell"], k["a_radius"], k["a_pol"]]]:
        cells = sorted({s % spec.num_cells for s in spec.rf_sources(int(c), int(r), int(p))})
        if not cells:                               # latent neuron: no receptive field
            continue
        covered[cells] = True
        cell_use[cells] += 1.0 / len(cells)

    heads = conn[:, k["a_second"]]
    tails = conn[:, k["a_first"]]
    hh = conn[heads >= h0]
    latent = {h0 + j for j, r in enumerate(new[:, k["a_radius"]]) if spec.rf_radii[int(r)] < 0}
    active_latent = len(latent & set(heads.tolist()) & set(tails.tolist()))
    n_hidden = len(new)
    recurrent = False
    if len(hh):
        ei = torch.tensor(hh[:, [k["a_first"], k["a_second"]]].T.astype(np.int64)) - h0
        recurrent = not is_dag(ei, n_hidden)

    return {
        "hidden": n_hidden,
        "rf_synapses": int(sum(rf_sizes)),
        "conn_synapses": len(conn),
        "rf_size_mean": float(np.mean([x for x in rf_sizes if x > 0])) if any(rf_sizes) else 0.0,
        "radius": np.bincount(new[:, k["a_radius"]], minlength=len(spec.rf_radii)),
        "pol": np.bincount(new[:, k["a_pol"]], minlength=3),
        "preset": np.bincount(new[:, k["a_node"]], minlength=spec.num_presets),
        "new_inhib": float((new[:, k["a_edge"]] >= L).mean()) if len(new) else 0.0,
        "new_wmag": np.bincount(new[:, k["a_edge"]] % L, minlength=L),
        "conn_to_sink": float(((heads >= S) & (heads < S + K)).mean()) if len(conn) else 0.0,
        "conn_inhib": float((conn[:, k["a_edge"]] >= L).mean()) if len(conn) else 0.0,
        "recurrent": recurrent,
        "coverage": float(covered.mean()),
        "cell_use": cell_use,
        "latent": len(latent),
        "latent_wired": active_latent,              # latent neurons with both an in- and an out-synapse
        "outputs_driven": len(set(heads[(heads >= S) & (heads < S + K)].tolist())),
        "conn_hh": len(hh),
        "rf_set": {(int(c), int(r), int(p)) for c, r, p in new[:, [k["a_cell"], k["a_radius"], k["a_pol"]]]
                   if spec.rf_radii[int(r)] >= 0},
    }


def group_similarity(sets: list[set]) -> float:
    """Mean pairwise Jaccard similarity of the graphs' receptive-field sets."""
    sims = [len(a & b) / max(1, len(a | b)) for a, b in itertools.combinations(sets, 2)]
    return float(np.mean(sims)) if sims else 0.0


def cell_informativeness(cfg: dict, spec: GraphSpec) -> np.ndarray:
    """Validation accuracy of a readout on one patch cell (both polarities, all windows) alone."""
    s, i = cfg["simulator"], cfg["inner"]
    pool = load_samples(s.get("nmnist_root", "data/NMNIST"), s.get("reward_split", "Train"), s["num_bins"],
                        i["train_per_class"] + s["val_per_class"], None, True, s.get("data_seed", 0), spec.patch_size)
    tr, va = split_per_class(pool, i["train_per_class"])
    (Xtr, ytr), (Xva, yva) = stack_samples(tr), stack_samples(va)
    rc = ReadoutConfig(windows=i.get("windows", 5), max_iter=100)
    C = NUM_CLASSES
    Ftr, Fva = window_counts(Xtr, rc.windows), window_counts(Xva, rc.windows)
    W, G2 = rc.windows, spec.num_cells
    out = np.zeros(G2)
    for c in range(G2):
        cols = torch.tensor([ch * W + w for ch in (c, c + G2) for w in range(W)])
        r = fit_readout(Ftr[:, cols], ytr, C, rc)
        out[c] = float((r.predict(Fva[:, cols]) == yva).double().mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bin", type=int, default=10, help="iterations per bin")
    args = ap.parse_args()

    base = Path("experiments") / args.tag
    runs = sorted(p for p in base.iterdir() if (p / "samples.pt").exists())
    if not runs:
        raise SystemExit(f"no samples.pt under {base}")
    cfg = yaml.safe_load(open(runs[0] / "config.yaml"))
    spec = GraphSpec.from_config(cfg["env"])
    out = base / "topology"
    out.mkdir(exist_ok=True)

    # per run: list over bins of averaged stats
    per_run = []
    for run in runs:
        log = torch.load(run / "samples.pt", weights_only=False)
        keys, iters = log["action_keys"], log["iters"]
        bins = []
        for b0 in range(0, len(iters), args.bin):
            chunk = iters[b0:b0 + args.bin]
            gs, accs, sims = [], [], []
            for it in chunk:
                st = [graph_stats(a, keys, spec) for a in it["actions"]]
                gs += st
                accs += it["acc"].tolist()
                sims.append(group_similarity([s["rf_set"] for s in st]))

            def frac(name):
                v = np.sum([s[name] for s in gs], axis=0).astype(float)
                return (v / max(1.0, v.sum())).tolist()

            bins.append({
                "iter": chunk[-1]["iter"],
                "acc_mean": float(np.mean(accs)),
                "acc_max": float(np.max(accs)),
                "hidden": float(np.mean([s["hidden"] for s in gs])),
                "rf_synapses": float(np.mean([s["rf_synapses"] for s in gs])),
                "conn_synapses": float(np.mean([s["conn_synapses"] for s in gs])),
                "rf_size_mean": float(np.mean([s["rf_size_mean"] for s in gs])),
                "radius": frac("radius"),
                "pol": frac("pol"),
                "preset": frac("preset"),
                "new_wmag": frac("new_wmag"),
                "new_inhib": float(np.mean([s["new_inhib"] for s in gs])),
                "conn_inhib": float(np.mean([s["conn_inhib"] for s in gs])),
                "conn_to_sink": float(np.mean([s["conn_to_sink"] for s in gs])),
                "recurrent": float(np.mean([s["recurrent"] for s in gs])),
                "coverage": float(np.mean([s["coverage"] for s in gs])),
                "latent": float(np.mean([s["latent"] for s in gs])),
                "latent_wired": float(np.mean([s["latent_wired"] for s in gs])),
                "outputs_driven": float(np.mean([s["outputs_driven"] for s in gs])),
                "conn_hh": float(np.mean([s["conn_hh"] for s in gs])),
                "similarity": float(np.mean(sims)),
                "cell_use": np.mean([s["cell_use"] for s in gs], axis=0).tolist(),
            })
        per_run.append({"run": run.name, "bins": bins})
        print(f"{run.name}: {len(iters)} iterations, {len(bins)} bins", flush=True)

    info = cell_informativeness(cfg, spec)
    groups: dict[str, list] = {}                     # variants: run name without the seed
    for r in per_run:
        groups.setdefault(r["run"].rsplit("_s", 1)[0], []).append(r)
    json.dump({"runs": per_run, "cell_informativeness": info.tolist(),
               "presets": [list(p) for p in spec.hidden_presets], "rf_radii": spec.rf_radii,
               "weight_levels": spec.weight_levels}, open(out / "stats.json", "w"))
    for name, rs in groups.items():
        tag = "" if len(groups) == 1 else f"_{name}"
        plot_evolution(rs, spec, out / f"evolution{tag}.png", name)
        plot_cells(rs, info, spec, out / f"cells{tag}.png")
    print("wrote", out)


def _series(per_run, key):
    x = np.array([b["iter"] for b in per_run[0]["bins"]])
    n = min(len(r["bins"]) for r in per_run)
    y = np.array([[b[key] for b in r["bins"][:n]] for r in per_run])
    return x[:n], y


def plot_evolution(per_run, spec: GraphSpec, path: Path, name: str = "") -> None:
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f"{name}: {len(per_run)} seeds (thin) and their mean (thick)", fontsize=12)
    axs = axs.ravel()

    def lines(ax, key, label=None, color=None):
        x, y = _series(per_run, key)
        for row in y:
            ax.plot(x, row, color=color, alpha=0.25, lw=1)
        ax.plot(x, y.mean(0), color=color, lw=2, label=label)

    def stacked(ax, key, names, title):
        x, _ = _series(per_run, "iter")
        n = len(x)
        y = np.mean([[b[key] for b in r["bins"][:n]] for r in per_run], axis=0).T
        ax.stackplot(x, y, labels=names, alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")

    lines(axs[0], "acc_mean", "batch mean", "C0")
    lines(axs[0], "acc_max", "batch max", "C1")
    axs[0].set_title("validation accuracy of sampled reservoirs")
    axs[0].legend()

    lines(axs[1], "hidden", "hidden neurons", "C0")
    lines(axs[1], "rf_synapses", "receptive-field synapses", "C2")
    lines(axs[1], "conn_synapses", "CONNECT synapses", "C3")
    lines(axs[1], "conn_hh", "  of which hidden→hidden", "C1")
    if any(b["latent"] > 0 for r in per_run for b in r["bins"]):
        lines(axs[1], "latent", "latent neurons (no field)", "C4")
        lines(axs[1], "latent_wired", "  of which wired in and out", "C6")
    lines(axs[1], "outputs_driven", "outputs with ≥1 input", "C7")
    axs[1].set_title("composition per graph")
    axs[1].legend()

    lines(axs[2], "rf_size_mean", "synapses per receptive field", "C4")
    lines(axs[2], "coverage", "fraction of cells covered", "C5")
    axs[2].set_title("receptive fields")
    axs[2].legend()

    stacked(axs[3], "radius", ["none (latent)" if r < 0 else f"r={r} ({2 * r + 1}×{2 * r + 1})" for r in spec.rf_radii],
            "receptive-field radius")
    stacked(axs[4], "pol", POL_NAMES, "receptive-field polarity")
    stacked(axs[5], "preset", [f"thr {t:g}, leak {lk:g}" for t, lk in spec.hidden_presets], "hidden neuron preset")
    stacked(axs[6], "new_wmag", [f"|w|={w:g}" for w in spec.weight_levels], "receptive-field weight magnitude")

    lines(axs[7], "new_inhib", "inhibitory receptive fields", "C0")
    lines(axs[7], "conn_inhib", "inhibitory CONNECT synapses", "C3")
    lines(axs[7], "conn_to_sink", "CONNECT into sinks", "C2")
    lines(axs[7], "recurrent", "graphs with a cycle", "C1")
    axs[7].set_ylim(0, 1)
    axs[7].set_title("signs and CONNECT usage")
    axs[7].legend(fontsize=8)

    lines(axs[8], "similarity", "mean pairwise Jaccard", "C0")
    axs[8].set_ylim(0, 1)
    axs[8].set_title("similarity of graphs within a group\n(receptive-field sets)")

    for ax in axs:
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_cells(per_run, info: np.ndarray, spec: GraphSpec, path: Path) -> None:
    G = spec.grid
    n = min(len(r["bins"]) for r in per_run)
    picks = [0, n // 2, n - 1]
    fig, axs = plt.subplots(len(per_run), len(picks) + 1, figsize=(3.4 * (len(picks) + 1), 3.2 * len(per_run)),
                            squeeze=False)
    for i, r in enumerate(per_run):
        for j, b in enumerate(picks):
            use = np.array(r["bins"][b]["cell_use"]).reshape(G, G)
            im = axs[i, j].imshow(use, cmap="viridis")
            axs[i, j].set_title(f"{r['run']}\niters ≤ {r['bins'][b]['iter']}", fontsize=9)
            fig.colorbar(im, ax=axs[i, j], fraction=0.046)
        im = axs[i, -1].imshow(info.reshape(G, G), cmap="magma")
        axs[i, -1].set_title("single-cell readout accuracy\n(input informativeness)", fontsize=9)
        fig.colorbar(im, ax=axs[i, -1], fraction=0.046)
    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("receptive-field mass per patch cell (neurons per graph)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
