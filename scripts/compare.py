"""
Launch and/or summarise an algorithm comparison at a matched simulator budget.

  python scripts/compare.py launch --tag cmp1 --seeds 0 1 [--config configs/default.yaml]
  python scripts/compare.py report --tag cmp1

`launch` starts one single-threaded process per (algo ∈ {gflownet, grpo, random},
surrogate on/off, seed)
in parallel; logs go to experiments/<tag>/<run>.log.
`report` prints, per configuration: best accuracy found on the reward split at
several simulator-call checkpoints, and the held-out Test accuracy of the best
graph (from summary.yaml).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = [("gflownet", True), ("grpo", True), ("random", True),
            ("gflownet", False), ("grpo", False), ("random", False)]


def parse_variants(spec: list[str] | None) -> list[tuple[str, bool]]:
    """['grpo:nosur', 'random:sur', ...] → [(algo, surrogate_on), ...]; None → all six."""
    if not spec:
        return VARIANTS
    out = []
    for v in spec:
        algo, _, sur = v.partition(":")
        out.append((algo, sur != "nosur"))
    return out


def launch(tag: str, seeds: list[int], config: str, variants: list[tuple[str, bool]]) -> None:
    out = ROOT / "experiments" / tag
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(ROOT / config))
    cfg.setdefault("train", {})["out_dir"] = str(out)
    cfg_path = out / "base_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    procs = []
    for algo, sur in variants:
        for seed in seeds:
            name = f"{algo}_{'sur' if sur else 'nosur'}_s{seed}"
            cmd = [sys.executable, "-W", "ignore", str(ROOT / "scripts/train.py"), "--config", str(cfg_path),
                   "--algo", algo, "--seed", str(seed), "--run-name", name]
            if not sur:
                cmd.append("--no-surrogate")
            log = open(out / f"{name}.log", "w")
            procs.append((name, subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)))
            print("started", name)
    for name, p in procs:
        p.wait()
        print(f"finished {name} (exit {p.returncode})")


def report(tag: str, checkpoints: list[int]) -> None:
    base = ROOT / "experiments" / tag
    groups: dict[str, list[dict]] = defaultdict(list)
    for run in sorted(p for p in base.iterdir() if p.is_dir()):
        rows = list(csv.DictReader(open(run / "metrics.csv"))) if (run / "metrics.csv").exists() else []
        summ = yaml.safe_load(open(run / "summary.yaml")) if (run / "summary.yaml").exists() else {}
        best_at = {}
        for c in checkpoints:
            vals = [float(r["best_sim_acc"]) for r in rows if int(r["sim_calls"]) <= c]
            best_at[c] = max(vals) if vals else float("nan")
        last = rows[-1] if rows else {}
        groups[run.name.rsplit("_s", 1)[0]].append({
            "run": run.name, "best_at": best_at, "summary": summ,
            "iters": last.get("iter", "-"), "sims": last.get("sim_calls", "-"),
            "acc_mean_last": float(last["acc_mean"]) if last else float("nan"),
            "unique_last": last.get("unique_frac", "-"),
        })

    hdr = ["config", "run", "iters", "sims"] + [f"best@{c}" for c in checkpoints] + \
          ["batch_acc_last", "uniq_last", "snm_val(best)", "test_acc(best)", "test_in_matched", "test_in_all", "E", "H"]
    print(" | ".join(hdr))
    for cfg, runs in groups.items():
        for r in runs:
            s = r["summary"]
            print(" | ".join([cfg, r["run"], str(r["iters"]), str(r["sims"])] +
                             [f"{r['best_at'][c]:.3f}" for c in checkpoints] +
                             [f"{r['acc_mean_last']:.3f}", str(r["unique_last"]),
                              f"{s.get('best_val_acc_superneuromat', float('nan')):.3f}" if s else "running",
                              f"{s.get('best_test_acc', float('nan')):.3f}" if s else "running",
                              f"{s.get('test_input_readout_matched', float('nan')):.3f}" if s else "-",
                              f"{s.get('test_input_readout_all', float('nan')):.3f}" if s else "-",
                              str(s.get("best_edges", "-")), str(s.get("best_hidden", "-"))]))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("launch")
    a.add_argument("--tag", required=True)
    a.add_argument("--seeds", type=int, nargs="+", default=[0])
    a.add_argument("--config", default="configs/default.yaml")
    a.add_argument("--variants", nargs="+", help="e.g. random:nosur grpo:nosur gflownet:sur (default: all six)")
    r = sub.add_parser("report")
    r.add_argument("--tag", required=True)
    r.add_argument("--checkpoints", type=int, nargs="+", default=[64, 128, 256, 400, 800])
    args = p.parse_args()
    if args.cmd == "launch":
        launch(args.tag, args.seeds, args.config, parse_variants(args.variants))
    else:
        report(args.tag, args.checkpoints)


if __name__ == "__main__":
    main()
