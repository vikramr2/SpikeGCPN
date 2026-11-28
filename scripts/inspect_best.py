"""
Structure and ablations of each run's best graph, on a large Test set.

  python scripts/inspect_best.py --tag latent1 [--test-per-class 300]

Per run: neuron / synapse composition of best_graph.pt, its val and Test
accuracy (readout as configured for the run), and the same after removing all
hidden→hidden synapses (what recurrence / depth contributes).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spikegcpn.data.datasets import load_splits  # noqa: E402
from spikegcpn.env.snn_env import GraphSpec  # noqa: E402
from spikegcpn.env.validity import is_dag  # noqa: E402
from spikegcpn.train.inner import stack_samples  # noqa: E402
from spikegcpn.train.readout import ReadoutConfig, ReservoirEvaluator  # noqa: E402


def structure(g: Data, spec: GraphSpec) -> dict:
    S, K, h0 = spec.num_source, spec.num_sink, spec.hidden_start
    src, dst = g.edge_index
    H = g.num_nodes - h0
    hidden = set(range(h0, g.num_nodes))
    fed = set(dst[src < S].tolist())                                   # neurons with a receptive field
    latent = hidden - fed
    to_out = (src >= h0) & (dst >= S) & (dst < h0)
    hh = (src >= h0) & (dst >= h0)
    lat_in = {int(v) for v in dst[hh].tolist()} & latent
    lat_out = {int(u) for u in src[(src >= h0)].tolist()} & latent
    hh_ei = g.edge_index[:, hh] - h0
    return {
        "neurons": H, "input_neurons": len(fed), "latent": len(latent),
        "latent_wired": len(lat_in & lat_out),
        "syn_rf": int((src < S).sum()), "syn_hh": int(hh.sum()), "syn_out": int(to_out.sum()),
        "outputs_driven": len(set(dst[to_out].tolist())),
        "inhib_frac": round(float((g.edge_attr[:, 0] < 0).float().mean()), 3),
        "recurrent": bool(hh.any()) and not is_dag(hh_ei, H),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--test-per-class", type=int, default=300)
    args = ap.parse_args()
    torch.set_num_threads(1)
    runs = sorted(p.parent for p in (Path("experiments") / args.tag).glob("*/best_graph.pt"))
    cfg = yaml.safe_load(open(runs[0] / "config.yaml"))
    tr, va, te, NUM_CLASSES = load_splits(cfg, test_per_class=args.test_per_class)
    sets = {"val": stack_samples(va), "test": stack_samples(te)}
    rows = {}
    for run in runs:
        rcfg = yaml.safe_load(open(run / "config.yaml"))
        spec = GraphSpec.from_config(rcfg["env"])
        rc = ReadoutConfig(**{k: v for k, v in rcfg["inner"].items() if k in ReadoutConfig.__dataclass_fields__})
        ev = {k: ReservoirEvaluator(spec.num_source, spec.num_sink, stack_samples(tr), v, rc,
                                    rcfg["inner"].get("extra_ticks", 10), NUM_CLASSES) for k, v in sets.items()}
        g = torch.load(run / "best_graph.pt", weights_only=False)
        keep = ~((g.edge_index[0] >= spec.hidden_start) & (g.edge_index[1] >= spec.hidden_start))
        g_ff = Data(x=g.x, edge_index=g.edge_index[:, keep], edge_attr=g.edge_attr[keep])
        acc = {k: e.train_and_eval([g, g_ff]) [0] for k, e in ev.items()}
        st = structure(g, spec)
        st.update(val=round(acc["val"][0], 4), test=round(acc["test"][0], 4),
                  test_without_hh=round(acc["test"][1], 4), val_without_hh=round(acc["val"][1], 4))
        rows[run.name] = st
        print(f"{run.name:22s} val {st['val']:.3f} test {st['test']:.3f} | without h→h: val {st['val_without_hh']:.3f} "
              f"test {st['test_without_hh']:.3f} | H={st['neurons']} (in {st['input_neurons']}, latent {st['latent']}, "
              f"wired {st['latent_wired']}) syn rf/hh/out={st['syn_rf']}/{st['syn_hh']}/{st['syn_out']} "
              f"outs={st['outputs_driven']} inh={st['inhib_frac']} rec={st['recurrent']}", flush=True)
    (Path("experiments") / args.tag / "best_graphs.yaml").write_text(yaml.safe_dump(rows, sort_keys=False))


if __name__ == "__main__":
    main()
