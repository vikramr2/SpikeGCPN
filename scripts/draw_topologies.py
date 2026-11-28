"""
Draw the sampled reservoir topologies as training progresses.

  python scripts/draw_topologies.py --tag topo1 [--every 10] [--snapshots 1 25 50 100 150 200]

For each run in experiments/<tag>, the highest-accuracy graph sampled in each
window of `--every` iterations is rebuilt by replaying its actions through
SNNEnv.step, simulated on the validation set, and drawn over the 9×9 patch grid:

  background     mean input events per patch cell (where digits produce events)
  square         a neuron's receptive field; edge colour = polarity
                 (OFF blue, ON orange, both purple), dashed = inhibitory field
  node           the hidden neuron, placed at its field's centre (latent neurons
                 without a field: columns right of the grid); marker = preset,
                 fill = the digit it fires most for (tab10), opacity = how
                 selective it is; grey hollow = never fires on validation data
  arrows         CONNECT synapses (green excitatory, red inhibitory), between
                 hidden neurons or into the sink / output column on the right
                 (hexagons, coloured like hidden neurons)

Writes to experiments/<tag>/topology/:
  graphs_<run>.png    the chosen snapshot iterations side by side
  graphs_<run>.gif    one frame per window, the whole run
  tuning_<run>.png    the final best graph, one panel per digit: that digit's
                      mean input events and each field neuron's response to it
  outputs_<run>.png   the best graph's output (sink) neurons × digits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from spikegcpn.data.nmnist import NUM_CLASSES, load_samples  # noqa: E402
from spikegcpn.env.snn_env import CONNECT, NEW, GraphSpec, SNNEnv  # noqa: E402
from spikegcpn.simulator.torch_lif import LIFBatch  # noqa: E402
from spikegcpn.train.inner import split_per_class, stack_samples  # noqa: E402

POL_COLOR = ["#1f77b4", "#ff7f0e", "#9467bd"]          # OFF, ON, both
POL_NAME = ["OFF", "ON", "both"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
DIGIT = plt.get_cmap("tab10")


def replay(actions: np.ndarray, keys: list[str], spec: GraphSpec):
    """Rebuild a graph and per-neuron NEW info from a logged action array."""
    k = {n: i for i, n in enumerate(keys)}
    env = SNNEnv(spec)
    g = env.reset()
    neurons, conns = [], []
    for row in actions:
        a = {n: int(row[i]) for n, i in k.items()}
        if a["a_stop"] == 0 and a["a_kind"] == NEW:
            neurons.append(a)
        elif a["a_stop"] == 0 and a["a_kind"] == CONNECT:
            conns.append(a)
        g, done = env.step(a)
        if done:
            break
    return g, neurons, conns


@torch.no_grad()
def class_responses(graphs, spec: GraphSpec, X: torch.Tensor, y: torch.Tensor, extra: int) -> list[np.ndarray]:
    """
    Per graph: [K + H, 10] mean spike count of every sink (first K rows) and
    hidden neuron per class, divided by that class's mean over all of them.
    Digits like 0 and 8 produce more events everywhere; without the
    normalisation most neurons would "prefer" them.
    """
    net = LIFBatch(graphs, spec.num_source, spec.num_sink, extra_ticks=extra, fixed=True)
    _, win = net(X, windows=1)                                # [N, total_ns, 1]
    out = []
    for sl in net.ns_slices:
        cnt = win[:, sl, 0]
        r = np.stack([cnt[y == c].mean(0).numpy() for c in range(NUM_CLASSES)], axis=1)
        out.append(r / (r.mean(0, keepdims=True) + 1e-9))
    return out


def place(spec: GraphSpec, neurons) -> tuple[dict, float]:
    """
    Node positions: a neuron with a receptive field sits at its field's centre
    (spread on a small ring when several share a cell); latent neurons (no
    field) fill columns right of the grid. Returns (pos, x of the sink column).
    """
    G = spec.grid
    pos, per_cell, latent = {}, {}, []
    for j, a in enumerate(neurons):
        if spec.rf_radii[a["a_radius"]] < 0:
            latent.append(j)
        else:
            per_cell.setdefault(a["a_cell"], []).append(j)
    for cell, js in per_cell.items():
        cy, cx = divmod(cell, G)
        for q, j in enumerate(js):
            ang, rad = 2 * np.pi * q / len(js), 0.0 if len(js) == 1 else 0.28
            pos[j] = (cx + rad * np.cos(ang), cy + rad * np.sin(ang))
    per_col = 2 * G
    cols = -(-len(latent) // per_col)
    for q, j in enumerate(latent):
        col, row = divmod(q, per_col)
        pos[j] = (G + 0.4 + 0.7 * col, -0.3 + row * (G - 0.4) / max(1, per_col - 1))
    return pos, G + 0.4 + 0.7 * cols + 0.6


def _neuron_marker(ax, x, y, rv, marker, size):
    if rv.max() <= 0:
        ax.scatter(x, y, s=size, marker=marker, facecolor="none", edgecolor="0.5", lw=0.8, zorder=5)
        return
    sel = float((rv.max() - rv.mean()) / rv.max())             # 0 = fires equally for all digits
    ax.scatter(x, y, s=size, marker=marker, c=[DIGIT(int(rv.argmax()))], alpha=0.35 + 0.65 * min(1.0, 2 * sel),
               edgecolor="k", lw=0.5, zorder=5)


def draw(ax, spec: GraphSpec, neurons, conns, resp: np.ndarray, background: np.ndarray, title: str) -> None:
    G, L, h0, S, K = spec.grid, len(spec.weight_levels), spec.hidden_start, spec.num_source, spec.num_sink
    ax.imshow(background, cmap="Greys", origin="upper", extent=(-0.5, G - 0.5, G - 0.5, -0.5), alpha=0.55)
    pos, sink_x = place(spec, neurons)
    sink_y = np.linspace(-0.3, G - 0.7, K)
    if any(spec.rf_radii[a["a_radius"]] < 0 for a in neurons):
        ax.text(G + 0.4, G - 0.05, "latent", fontsize=7, ha="left", color="0.3")

    for a in neurons:
        r = spec.rf_radii[a["a_radius"]]
        if r < 0:
            continue
        cy, cx = divmod(a["a_cell"], G)
        x0, x1 = max(-0.5, cx - r - 0.5), min(G - 0.5, cx + r + 0.5)
        y0, y1 = max(-0.5, cy - r - 0.5), min(G - 0.5, cy + r + 0.5)
        inh = a["a_edge"] >= L
        ax.add_patch(Rectangle((x0 + 0.04, y0 + 0.04), x1 - x0 - 0.08, y1 - y0 - 0.08, fill=False,
                               ec=POL_COLOR[a["a_pol"]], lw=0.8, ls="--" if inh else "-", alpha=0.55, zorder=2))

    def node_xy(n: int):
        return pos[n - h0] if n >= h0 else (sink_x, sink_y[n - S])

    for a in conns:
        u, v = a["a_first"], a["a_second"]
        inh = a["a_edge"] >= L
        w = spec.weight_levels[a["a_edge"] % L]
        ax.add_patch(FancyArrowPatch(node_xy(u), node_xy(v), connectionstyle="arc3,rad=0.25",
                                     arrowstyle="-|>", mutation_scale=6, lw=0.4 + 0.4 * w,
                                     color="#d62728" if inh else "#2ca02c", alpha=0.18, zorder=3))

    for c in range(K):                                         # output / sink neurons
        _neuron_marker(ax, sink_x, sink_y[c], resp[c], "h", 110)
        ax.text(sink_x + 0.35, sink_y[c], str(c), va="center", fontsize=6)
    for j, a in enumerate(neurons):
        _neuron_marker(ax, *pos[j], resp[K + j], MARKERS[a["a_node"] % len(MARKERS)], 40)

    ax.set_xlim(-0.7, sink_x + 0.8)
    ax.set_ylim(G - 0.3, -0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def legend(fig, spec: GraphSpec) -> None:
    h = [Line2D([], [], color=c, lw=1.5, label=f"field: {n}") for c, n in zip(POL_COLOR, POL_NAME)]
    h += [Line2D([], [], color="0.3", lw=1.2, ls="--", label="inhibitory field")]
    h += [Line2D([], [], ls="", marker=MARKERS[i], mfc="0.7", mec="k", label=f"thr {t:g}, leak {lk:g}")
          for i, (t, lk) in enumerate(spec.hidden_presets)]
    h += [Line2D([], [], ls="", marker="h", mfc="0.7", mec="k", ms=8, label="output / sink"),
          Line2D([], [], ls="", marker="o", mfc="none", mec="0.5", label="silent neuron"),
          Line2D([], [], color="#2ca02c", lw=1.2, label="CONNECT exc."),
          Line2D([], [], color="#d62728", lw=1.2, label="CONNECT inh.")]
    h += [Line2D([], [], ls="", marker="o", mfc=DIGIT(c), mec="k", label=f"prefers {c}") for c in range(10)]
    fig.legend(handles=h, loc="lower center", ncol=8, fontsize=7, frameon=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--snapshots", type=int, nargs="+", default=[1, 25, 50, 100, 150, 200])
    args = ap.parse_args()
    torch.set_num_threads(1)

    base = Path("experiments") / args.tag
    runs = sorted(p for p in base.iterdir() if (p / "samples.pt").exists())
    cfg = yaml.safe_load(open(runs[0] / "config.yaml"))
    spec = GraphSpec.from_config(cfg["env"])
    s, i = cfg["simulator"], cfg["inner"]
    pool = load_samples(s.get("nmnist_root", "data/NMNIST"), "Train", s["num_bins"],
                        i["train_per_class"] + s["val_per_class"], None, True, s.get("data_seed", 0), spec.patch_size)
    _, va = split_per_class(pool, i["train_per_class"])
    Xva, yva = stack_samples(va)
    extra = int(i.get("extra_ticks", 10))
    G2 = spec.num_cells
    ev = Xva.float().sum(1)                                   # [N, S] events per source
    cell_ev = ev[:, :G2] + ev[:, G2:]                          # both polarities
    background = cell_ev.mean(0).reshape(spec.grid, spec.grid).numpy()
    digit_bg = [cell_ev[yva == c].mean(0).reshape(spec.grid, spec.grid).numpy() for c in range(NUM_CLASSES)]
    out = base / "topology"
    out.mkdir(exist_ok=True)

    for run in runs:
        spec = GraphSpec.from_config(yaml.safe_load(open(run / "config.yaml"))["env"])   # variants differ
        log = torch.load(run / "samples.pt", weights_only=False)
        keys, iters = log["action_keys"], log["iters"]
        frames = []                                            # (iter range, acc, neurons, conns, graph)
        for b0 in range(0, len(iters), args.every):
            chunk = iters[b0:b0 + args.every]
            ti, tj = max(((a, j) for a, it in enumerate(chunk) for j in range(len(it["acc"]))),
                         key=lambda p: chunk[p[0]]["acc"][p[1]])
            g, neurons, conns = replay(chunk[ti]["actions"][tj], keys, spec)
            frames.append({"iter": chunk[ti]["iter"], "acc": float(chunk[ti]["acc"][tj]),
                           "neurons": neurons, "conns": conns, "graph": g})
        for snap in args.snapshots:                            # exact iterations for the snapshot figure
            it = next((x for x in iters if x["iter"] == snap), None)
            if it is None:
                continue
            j = int(np.argmax(it["acc"]))
            g, neurons, conns = replay(it["actions"][j], keys, spec)
            frames.append({"iter": snap, "acc": float(it["acc"][j]), "neurons": neurons, "conns": conns,
                           "graph": g, "snapshot": True})
        resps = class_responses([f["graph"] for f in frames], spec, Xva, yva, extra)
        for f, r in zip(frames, resps):
            f["resp"] = r

        def title(f):
            return (f"iter {f['iter']}  val acc {f['acc']:.3f}\n"
                    f"H={len(f['neurons'])}  E={int(f['graph'].edge_index.shape[1])}  CONNECT={len(f['conns'])}")

        snaps = [f for f in frames if f.get("snapshot")]
        cols = 3
        rows = -(-len(snaps) // cols)
        fig, axs = plt.subplots(rows, cols, figsize=(5.6 * cols, 5.6 * rows + 1.4), squeeze=False)
        for ax, f in zip(axs.ravel(), snaps):
            draw(ax, spec, f["neurons"], f["conns"], f["resp"], background, title(f))
        for ax in axs.ravel()[len(snaps):]:
            ax.axis("off")
        fig.suptitle(f"{run.name}: best sampled reservoir at each iteration "
                     "(background = mean input events per patch cell)", fontsize=12)
        legend(fig, spec)
        fig.tight_layout(rect=(0, 0.07, 1, 0.97))
        fig.savefig(out / f"graphs_{run.name}.png", dpi=130)
        plt.close(fig)

        anim_frames = [f for f in frames if not f.get("snapshot")]
        fig, ax = plt.subplots(figsize=(5.2, 5.4))

        def update(k):
            ax.clear()
            f = anim_frames[k]
            draw(ax, spec, f["neurons"], f["conns"], f["resp"], background,
                 f"{run.name}  best of iters ≤ {f['iter']}\n" + title(f).split("\n", 1)[1]
                 + f"  acc {f['acc']:.3f}")

        FuncAnimation(fig, update, frames=len(anim_frames)).save(out / f"graphs_{run.name}.gif",
                                                                 writer=PillowWriter(fps=2), dpi=90)
        plt.close(fig)

        # --- digit tuning of the run's best graph ------------------------------------
        # the highest-accuracy graph ever sampled (= the oracle's best_graph.pt)
        bi, bj = max(((a, j) for a, it in enumerate(iters) for j in range(len(it["acc"]))),
                     key=lambda p: iters[p[0]]["acc"][p[1]])
        best_acc = float(iters[bi]["acc"][bj])
        g, neurons, conns = replay(iters[bi]["actions"][bj], keys, spec)
        (resp,) = class_responses([g], spec, Xva, yva, extra)
        K, G = spec.num_sink, spec.grid
        z = (resp - resp.mean(1, keepdims=True)) / (resp.std(1, keepdims=True) + 1e-9)
        pos, _ = place(spec, neurons)
        fig, axs = plt.subplots(2, 5, figsize=(17, 7.6))
        sc = None
        for c, ax in enumerate(axs.ravel()):
            ax.imshow(digit_bg[c], cmap="Greys", origin="upper", extent=(-0.5, G - 0.5, G - 0.5, -0.5))
            for j, a in enumerate(neurons):
                if spec.rf_radii[a["a_radius"]] < 0:
                    continue
                sc = ax.scatter(*pos[j], s=55, marker=MARKERS[a["a_node"] % len(MARKERS)], c=[z[K + j, c]],
                                cmap="coolwarm", vmin=-2.5, vmax=2.5, edgecolor="k", lw=0.4)
            ax.set_title(f"digit {c}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        if sc is not None:
            fig.colorbar(sc, ax=axs, fraction=0.015, label="relative response to this digit (z-score across digits)")
        fig.suptitle(f"{run.name}: best reservoir (iter {iters[bi]['iter']}, val {best_acc:.3f}), neurons with a "
                     "receptive field; background = that digit's mean input events per cell", fontsize=11)
        fig.savefig(out / f"tuning_{run.name}.png", dpi=110)
        plt.close(fig)

        # output neurons: what the readout sees
        fig, ax = plt.subplots(figsize=(6, 0.35 * K + 1.5))
        im = ax.imshow(z[:K], cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_xlabel("digit")
        ax.set_yticks(range(K))
        ax.set_ylabel("output neuron")
        silent = resp[:K].max(1) <= 0
        for k in np.nonzero(silent)[0]:
            ax.text(4.5, k, "silent", ha="center", va="center", fontsize=7, color="0.3")
        fig.colorbar(im, ax=ax, label="relative response (z across digits)")
        ax.set_title(f"{run.name}: output-neuron tuning of the best reservoir", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"outputs_{run.name}.png", dpi=110)
        plt.close(fig)
        print("drew", run.name, flush=True)


if __name__ == "__main__":
    main()
