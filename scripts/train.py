"""
SpikeGCPN training entry point.

Usage
-----
  python scripts/train.py --algo grpo                        # configs/default.yaml
  python scripts/train.py --algo grpo --no-surrogate
  python scripts/train.py --algo random                      # random-topology baseline
  python scripts/train.py --config configs/smoke.yaml --algo grpo
  python scripts/train.py --resume experiments/<run>/ckpt.pt

Iteration
---------
1. Sample `group_size` topologies (policy, or uniformly random for `random`).
2. Score them through the RewardOracle. A "simulator call" is one topology
   evaluation, depending on `inner.mode`:
     reservoir : simulate with fixed weights, fit a logistic-regression readout
                 on every neuron's windowed spike counts → validation accuracy
     bptt      : train weights / thresholds / leaks with surrogate-gradient
                 BPTT, read out sink spike counts → validation accuracy
     control   : CartPole with hidden velocities; behaviour-clone a readout from
                 the reservoir's read-out neurons, then run it closed loop →
                 mean return / 500 (spikegcpn/control/evaluator.py)
   With `inner.enabled: false` the topology is scored as-is by sink counts.
   The surrogate model (if enabled) decides which topologies get evaluated.
3. GRPO: group-normalised clipped policy-gradient update.
4. Log a CSV row; stop at max_iters or when sim_budget evaluations are used.

The reward function (data subsets and inner-training seed) is fixed by
`simulator.data_seed` and `inner.seed`, not by the run seed, so every run and
algorithm optimises the same objective.

Outputs go to experiments/<run_name>/: metrics.csv, config.yaml, ckpt.pt,
best_graph.pt (trained), summary.yaml (SuperNeuroMAT-verified val + Test accuracy),
samples.pt (every sampled trajectory's actions + accuracy per iteration; replay
through SNNEnv.step to rebuild the graphs, see scripts/topology_evolution.py).
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spikegcpn.control.evaluator import ControlConfig, ControlEvaluator
from spikegcpn.data.datasets import load_splits, num_sources
from spikegcpn.env.snn_env import STOP_ACTION, GraphSpec, SNNEnv
from spikegcpn.env.validity import is_dag
from spikegcpn.model.encoder import GraphEncoder
from spikegcpn.model.policy import PolicyHead
from spikegcpn.model.surrogate import SurrogateEnsemble
from spikegcpn.simulator.wrapper import evaluate_samples, window_spike_counts
from spikegcpn.train.grpo import GRPOConfig, GRPOTrainer
from spikegcpn.train.inner import InnerConfig, InnerTrainer, stack_samples
from spikegcpn.train.oracle import RewardOracle
from spikegcpn.train.readout import ReadoutConfig, ReservoirEvaluator, input_readout_baselines
from spikegcpn.train.rewards import RewardConfig, shaped_reward
from spikegcpn.train.rollout import sample_random_trajectory, sample_trajectories


def _dc(cls, d: dict | None):
    d = d or {}
    return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def train(cfg: dict, resume: str | None = None) -> dict:
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(cfg.get("threads", 1)))
    rng = random.Random(seed)

    algo = cfg["algo"]
    tcfg = cfg.get("train", {})
    sur_cfg = cfg.get("surrogate", {})
    icfg = cfg.get("inner", {})
    mcfg = cfg.get("model", {})
    reward_cfg = _dc(RewardConfig, cfg.get("rewards"))

    spec = GraphSpec.from_config(cfg.get("env", {}))
    env = SNNEnv(spec)
    encoder = GraphEncoder(spec.num_sink, mcfg.get("embed_dim", 64), mcfg.get("num_layers", 3))
    policy = PolicyHead(spec, mcfg.get("embed_dim", 64), mcfg.get("mlp_hidden", 128))

    # --- reward pipeline (fixed across runs) -----------------------------
    use_inner = icfg.get("enabled", True)
    mode = icfg.get("mode", "bptt") if use_inner else "none"
    if mode != "control":
        if spec.num_source != num_sources(cfg):
            raise ValueError(f"env has {spec.num_source} sources but the data has {num_sources(cfg)} input channels")
        train_samples, val_samples, test_samples, NUM_CLASSES = load_splits(cfg)
        if mode != "reservoir" and spec.num_sink != NUM_CLASSES:
            raise ValueError(f"num_sink != {NUM_CLASSES} needs inner.mode: reservoir (sinks are then read-out neurons)")
    if spec.allow_cycles and mode not in ("reservoir", "control"):
        raise ValueError("recurrent topologies (allow_cycles) need inner.mode: reservoir or control")
    inner = reservoir = controller = None
    baselines: dict = {}
    if mode == "control":
        controller = ControlEvaluator(spec.num_source, spec.num_sink, _dc(ControlConfig, icfg))
        baselines = controller.baselines()
        print("control baselines:", baselines, flush=True)

        def simulate(graphs):
            return controller.train_and_eval(graphs)[0]
    elif mode == "bptt":
        inner = InnerTrainer(spec.num_source, spec.num_sink, stack_samples(train_samples),
                             stack_samples(val_samples), _dc(InnerConfig, icfg))

        def simulate(graphs):
            return inner.train_and_eval(graphs)[0]
    elif mode == "reservoir":
        rcfg = _dc(ReadoutConfig, icfg)
        extra_ticks = int(icfg.get("extra_ticks", 10))
        tr_set, va_set = stack_samples(train_samples), stack_samples(val_samples)
        reservoir = ReservoirEvaluator(spec.num_source, spec.num_sink, tr_set, va_set, rcfg, extra_ticks, NUM_CLASSES)
        read_neurons = spec.num_sink if rcfg.readout_neurons == "outputs" else spec.num_sink + spec.max_hidden
        baselines = input_readout_baselines(tr_set, va_set, NUM_CLASSES, rcfg, matched_neurons=read_neurons)
        print("input-readout baselines:", baselines, flush=True)

        def simulate(graphs):
            return reservoir.train_and_eval(graphs)[0]
    else:
        def simulate(graphs):
            return [evaluate_samples(g, val_samples, spec.num_source, spec.num_sink) for g in graphs]

    surrogate = None
    if sur_cfg.get("enabled", True):
        surrogate = SurrogateEnsemble(
            spec, sur_cfg.get("ensemble_size", 4), sur_cfg.get("embed_dim", 32),
            sur_cfg.get("num_layers", 2), sur_cfg.get("lr", 1e-3), seed,
        )
    oracle = RewardOracle(
        simulate, surrogate,
        warmup=sur_cfg.get("warmup", 64),
        sim_per_batch=sur_cfg.get("sim_per_batch", 4),
        ucb_kappa=sur_cfg.get("ucb_kappa", 1.0),
        refit_every=sur_cfg.get("refit_every", 16),
        refit_steps=sur_cfg.get("refit_steps", 50),
    )

    # --- trainer ---------------------------------------------------------
    trainer = None
    if algo == "grpo":
        trainer = GRPOTrainer(encoder, policy, _dc(GRPOConfig, cfg.get("grpo")), seed)
    elif algo != "random":
        raise ValueError(f"unknown algo {algo!r}")
    if resume and trainer is not None:
        trainer.load_state_dict(torch.load(resume, weights_only=False)["trainer"])

    sur_tag = "" if surrogate is not None else "_nosur"
    run_name = cfg.get("run_name") or f"{algo}{sur_tag}_s{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path(tcfg.get("out_dir", "experiments")) / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    group_size = tcfg.get("group_size", 16)
    max_iters = tcfg.get("max_iters", 500)
    sim_budget = tcfg.get("sim_budget", 10**9)
    ckpt_interval = tcfg.get("checkpoint_interval", 50)

    fields = ["iter", "wall_s", "sim_calls", "acc_mean", "acc_max", "frac_sim", "frac_cache",
              "reward_mean", "best_sim_acc", "edges_mean", "hidden_mean", "unique_frac",
              "loss", "aux", "surrogate_mse"]
    f = open(out / "metrics.csv", "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    print(f"[{run_name}] algo={algo} surrogate={'on' if surrogate is not None else 'off'} "
          f"inner={mode} cycles={spec.allow_cycles} sources={spec.num_source} group={group_size} "
          f"budget={sim_budget} evals  →  {out}", flush=True)
    t0 = time.time()
    it = 0
    action_keys = list(STOP_ACTION)
    samples_log: list[dict] = []
    for it in range(1, max_iters + 1):
        if algo == "random":
            trajs = [sample_random_trajectory(env, rng) for _ in range(group_size)]
        else:
            trajs = sample_trajectories(spec, encoder, policy, group_size)

        scores = oracle.score([t.final for t in trajs], [t.key for t in trajs])
        accs = [s.accuracy for s in scores]
        rewards = [shaped_reward(a, t.num_hidden, t.num_edges, reward_cfg) for a, t in zip(accs, trajs)]

        if algo == "grpo":
            stats = trainer.update(trajs, rewards)
            aux = f"kl={stats['approx_kl']:.4f} clip={stats['clip_frac']:.3f} ent={stats['entropy']:.2f}"
        else:
            stats, aux = {"loss": 0.0}, ""

        samples_log.append({
            "iter": it,
            "acc": np.asarray(accs, dtype=np.float32),
            "actions": [np.array([[a[k] for k in action_keys] for a in t.actions], dtype=np.int16) for t in trajs],
        })

        best = oracle.best()
        row = {
            "iter": it,
            "wall_s": round(time.time() - t0, 1),
            "sim_calls": oracle.sim_calls,
            "acc_mean": round(float(np.mean(accs)), 4),
            "acc_max": round(float(np.max(accs)), 4),
            "frac_sim": round(sum(s.source == "sim" for s in scores) / len(scores), 3),
            "frac_cache": round(sum(s.source == "cache" for s in scores) / len(scores), 3),
            "reward_mean": round(float(np.mean(rewards)), 4),
            "best_sim_acc": round(best[1], 4) if best else 0.0,
            "edges_mean": round(float(np.mean([t.num_edges for t in trajs])), 1),
            "hidden_mean": round(float(np.mean([t.num_hidden for t in trajs])), 1),
            "unique_frac": round(len({t.key for t in trajs}) / len(trajs), 3),
            "loss": round(stats["loss"], 4),
            "aux": aux,
            "surrogate_mse": round(oracle.last_fit_mse, 5) if surrogate is not None else "",
        }
        writer.writerow(row)
        f.flush()
        if it % tcfg.get("log_interval", 5) == 0 or it == 1:
            print(f"it {it:4d} | evals {row['sim_calls']:5d} | acc {row['acc_mean']:.3f} (max {row['acc_max']:.3f}) "
                  f"| best {row['best_sim_acc']:.3f} | E {row['edges_mean']:5.1f} H {row['hidden_mean']:4.1f} "
                  f"| uniq {row['unique_frac']:.2f} | loss {row['loss']:.3g} | {aux} | {row['wall_s']:.0f}s",
                  flush=True)

        if it % ckpt_interval == 0:
            torch.save({"action_keys": action_keys, "iters": samples_log}, out / "samples.pt")
            if trainer is not None:
                torch.save({"trainer": trainer.state_dict(), "iter": it}, out / "ckpt.pt")
        if oracle.sim_calls >= sim_budget:
            print(f"evaluation budget reached ({oracle.sim_calls})")
            break
    f.close()
    torch.save({"action_keys": action_keys, "iters": samples_log}, out / "samples.pt")
    if trainer is not None:
        torch.save({"trainer": trainer.state_dict(), "iter": it}, out / "ckpt.pt")

    # --- verify the best topology in SuperNeuroMAT -------------------------
    summary: dict = {"run": run_name, "algo": algo, "surrogate": surrogate is not None,
                     "inner": mode, "allow_cycles": spec.allow_cycles, "iters": it,
                     "sim_calls": oracle.sim_calls, "wall_s": round(time.time() - t0, 1)}
    summary.update({k: round(v, 4) if isinstance(v, float) else v for k, v in baselines.items()})
    best = oracle.best()
    if best:
        g = oracle.graphs[best[0]]
        summary.update(best_val_acc=round(best[1], 4), best_edges=int(g.edge_index.shape[1]),
                       best_hidden=int(g.num_nodes - spec.hidden_start),
                       best_recurrent=bool(spec.allow_cycles and not is_dag(g.edge_index, g.num_nodes)))
        if controller is not None:
            # Refit the behaviour-cloned readout (deterministic), test on held-out episodes,
            # and check that SuperNeuroMAT reproduces the closed-loop read-out spikes exactly.
            _, (ro,) = controller.train_and_eval([g])
            summary.update(best_test_return=round(controller.test(g, ro), 2), **controller.verify_superneuromat(g, ro))
        elif reservoir is not None:
            # Refit the readout (deterministic), then recompute the reservoir features in
            # SuperNeuroMAT itself and apply the readout → val / Test accuracy.
            _, (ro,) = reservoir.train_and_eval([g])
            W = reservoir.cfg.windows
            accs = {}
            for name, samples in (("val", val_samples), ("test", test_samples)):
                win = window_spike_counts(g, samples, spec.num_source, spec.num_sink, reservoir.extra_ticks, W)
                if reservoir.cfg.readout_neurons == "outputs":
                    win = win[:, :spec.num_sink]
                feats = torch.from_numpy(win).reshape(len(samples), -1)
                labels = torch.tensor([smp.label for smp in samples])
                accs[name] = float((ro.predict(feats.double()) == labels).double().mean())
            summary.update(best_val_acc_superneuromat=round(accs["val"], 4), best_test_acc=round(accs["test"], 4))
            test_in = input_readout_baselines(stack_samples(train_samples), stack_samples(test_samples),
                                              NUM_CLASSES, reservoir.cfg, read_neurons)
            summary.update(test_input_readout_all=round(test_in["input_readout_all"], 4),
                           test_input_readout_matched=round(test_in["input_readout_matched"], 4),
                           test_input_readout_top=round(test_in["input_readout_top"], 4))
        else:
            if inner is not None:
                _, (g,) = inner.train_and_eval([g])   # deterministic: same parameters as during search
            summary.update(
                best_val_acc_superneuromat=round(evaluate_samples(g, val_samples, spec.num_source, spec.num_sink), 4),
                best_test_acc=round(evaluate_samples(g, test_samples, spec.num_source, spec.num_sink), 4))
        torch.save(g, out / "best_graph.pt")
    (out / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    print("summary:", summary, flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Train SpikeGCPN")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--algo", choices=["grpo", "random"], help="overrides config 'algo'")
    p.add_argument("--no-surrogate", action="store_true", help="evaluate every sampled topology")
    p.add_argument("--no-inner", action="store_true", help="score topologies with fixed weights (no training)")
    p.add_argument("--seed", type=int)
    p.add_argument("--run-name")
    p.add_argument("--resume")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config entries, e.g. env.allow_cycles=false rewards.efficiency_coeff=1e-4")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    for kv in args.set:
        key, val = kv.split("=", 1)
        *path, last = key.split(".")
        d = cfg
        for k in path:
            d = d.setdefault(k, {})
        d[last] = yaml.safe_load(val)
    if args.algo:
        cfg["algo"] = args.algo
    if args.no_surrogate:
        cfg.setdefault("surrogate", {})["enabled"] = False
    if args.no_inner:
        cfg.setdefault("inner", {})["enabled"] = False
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.run_name:
        cfg["run_name"] = args.run_name
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
