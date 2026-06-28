"""
SpikeGCPN training entry point.

Usage
-----
  python scripts/train.py                          # uses configs/default.yaml
  python scripts/train.py --config configs/my.yaml
  python scripts/train.py --resume experiments/checkpoints/ep500.pt

Episode loop
------------
1. env.reset()  → initial graph (sources + sinks, no hidden, no edges)
2. For each step:
   a. Encode graph with GCNEncoder → node embeddings H
   b. Sample action from PolicyHead
   c. env.step(action) → next graph, step reward, done
   d. If done (a_stop=1): call simulator to get accuracy → add terminal reward
   e. Store Transition in RolloutBuffer
3. PPOTrainer.update(buffer) → gradient step
4. Every eval_interval episodes: run a larger evaluation and log accuracy
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml
import torch
from torch_geometric.data import Data

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from spikegcpn.env.snn_env import SNNEnv
from spikegcpn.model.encoder import GCNEncoder
from spikegcpn.model.policy import PolicyHead
from spikegcpn.model.value import ValueFunction
from spikegcpn.train.ppo import PPOTrainer, PPOConfig, RolloutBuffer, Transition, _clone_graph
from spikegcpn.train.rewards import RewardConfig, step_reward, terminal_reward
from spikegcpn.simulator.wrapper import evaluate, imprint
from spikegcpn.data.nmnist import NUM_SOURCE_NODES, NUM_CLASSES


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get(cfg: dict, *keys, default=None):
    """Nested dict lookup with a default."""
    d = cfg
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# ---------------------------------------------------------------------------
# Quick accuracy estimate (called after every episode)
# ---------------------------------------------------------------------------

def quick_accuracy(
    graph: Data,
    nmnist_root: str,
    num_bins: int,
    n_samples: int,
    num_source: int,
    num_sink: int,
) -> float:
    """
    Run a small N-MNIST evaluation on the final topology.
    Uses per_class=ceil(n_samples/10) to keep the sample balanced.
    """
    per_cls = max(1, n_samples // NUM_CLASSES)
    try:
        return evaluate(
            data=graph,
            nmnist_root=nmnist_root,
            num_source=num_source,
            num_sink=num_sink,
            num_bins=num_bins,
            per_class=per_cls,
            split="Test",
        )
    except Exception as e:
        # Gracefully handle invalid graphs (e.g. sinks not reachable at episode end)
        print(f"  [warn] evaluate() failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Single episode rollout
# ---------------------------------------------------------------------------

def run_episode(
    env: SNNEnv,
    encoder: GCNEncoder,
    policy: PolicyHead,
    value_fn: ValueFunction,
    buffer: RolloutBuffer,
    reward_cfg: RewardConfig,
    sim_cfg: dict,
    max_steps: int,
) -> dict:
    """
    Run one episode and fill `buffer` with Transitions.

    Returns episode statistics dict.
    """
    obs, _ = env.reset()
    buffer.clear()

    ep_reward = 0.0
    ep_steps = 0
    accuracy = 0.0

    for step_idx in range(max_steps):
        g = obs
        is_last_step = (step_idx == max_steps - 1)
        first_mask = env.valid_first_mask()
        second_mask = env.valid_second_mask()

        with torch.no_grad():
            H = encoder(g.x, g.edge_index, g.edge_attr)
            value = value_fn(H).item()
            action, log_probs = policy.sample_actions(
                H, first_mask=first_mask, second_mask=second_mask
            )

        obs_next, r_step, terminated, truncated, info = env.step(action)
        truncated = truncated or is_last_step  # force truncation flag on last step
        done = terminated or truncated

        # Step reward: tiny positive for valid moves, tiny negative for invalid.
        # Kept very small so accumulated steps never dominate the terminal signal.
        valid = "invalid" not in info
        r = step_reward(valid, reward_cfg)

        # Terminal reward on explicit stop (a_stop=1)
        if terminated:
            accuracy = quick_accuracy(
                graph=g,
                nmnist_root=sim_cfg["nmnist_root"],
                num_bins=sim_cfg["num_bins"],
                n_samples=sim_cfg["train_eval_samples"],
                num_source=env.num_source,
                num_sink=env.num_sink,
            )
            r += terminal_reward(
                accuracy=accuracy,
                num_hidden=env._num_hidden,
                num_edges=int(g.edge_index.shape[1]),
                cfg=reward_cfg,
            )
        elif truncated:
            # Penalty for hitting max_steps without stopping: ensures the policy
            # prefers to stop early even with low accuracy vs. running out the clock.
            r -= reward_cfg.truncation_penalty

        # Store masks alongside the graph snapshot so the PPO update can
        # reconstruct the same distributions
        lp_total = log_probs.total().item()
        buffer.push(Transition(
            graph=_clone_graph(g),
            action=action,
            log_prob=lp_total,
            reward=r,
            value=value,
            done=done,
            first_mask=first_mask,
            second_mask=second_mask,
        ))

        ep_reward += r
        ep_steps += 1

        if done:
            break
        obs = obs_next

    # If the episode ended by hitting max_steps (truncated, never stopped),
    # evaluate accuracy on the final graph so the metric is always visible.
    if accuracy == 0.0 and len(buffer) > 0:
        accuracy = quick_accuracy(
            graph=env.graph,
            nmnist_root=sim_cfg["nmnist_root"],
            num_bins=sim_cfg["num_bins"],
            n_samples=sim_cfg["train_eval_samples"],
            num_source=env.num_source,
            num_sink=env.num_sink,
        )

    return {
        "ep_reward": ep_reward,
        "ep_steps": ep_steps,
        "accuracy": accuracy,
        "num_hidden": env._num_hidden,
        "num_edges": int(env.graph.edge_index.shape[1]),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, resume: str | None = None) -> None:
    env_cfg  = cfg.get("env", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    sim_cfg   = cfg.get("simulator", {})
    rew_cfg   = cfg.get("rewards", {})

    # --- Environment ---
    env = SNNEnv(
        num_source=env_cfg.get("num_source", NUM_SOURCE_NODES),
        num_sink=env_cfg.get("num_sink", NUM_CLASSES),
        max_hidden=env_cfg.get("max_hidden", 50),
        default_threshold=env_cfg.get("default_threshold", 0.5),
        default_leak=env_cfg.get("default_leak", float("inf")),
    )

    # --- Models ---
    encoder = GCNEncoder(
        node_feat_dim=model_cfg.get("node_feat_dim", 5),
        embed_dim=model_cfg.get("embed_dim", 64),
        num_layers=model_cfg.get("num_layers", 3),
        num_edge_types=model_cfg.get("num_edge_types", 2),
    )
    policy = PolicyHead(
        embed_dim=model_cfg.get("embed_dim", 64),
        mlp_hidden=model_cfg.get("mlp_hidden", 64),
    )
    value_fn = ValueFunction(
        embed_dim=model_cfg.get("embed_dim", 64),
        mlp_hidden=model_cfg.get("mlp_hidden", 64),
    )

    # --- PPO ---
    ppo_cfg = PPOConfig(
        lr=train_cfg.get("lr", 1e-3),
        gamma=train_cfg.get("gamma", 0.99),
        gae_lambda=train_cfg.get("gae_lambda", 0.95),
        clip_eps=train_cfg.get("clip_eps", 0.2),
        ppo_epochs=train_cfg.get("ppo_epochs", 4),
        value_coeff=train_cfg.get("value_coeff", 0.5),
        entropy_coeff=train_cfg.get("entropy_coeff", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.5),
    )
    trainer = PPOTrainer(encoder, policy, value_fn, ppo_cfg)

    if resume:
        trainer.load(resume)
        print(f"Resumed from {resume}")

    reward_cfg = RewardConfig(
        validity_step=rew_cfg.get("validity_step", 0.001),
        invalid_step=rew_cfg.get("invalid_step", -0.01),
        efficiency_coeff=rew_cfg.get("efficiency_coeff", 0.0005),
        truncation_penalty=rew_cfg.get("truncation_penalty", 0.5),
    )

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "experiments/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    max_episodes       = train_cfg.get("max_episodes", 10000)
    max_steps          = train_cfg.get("max_steps_per_episode", 100)
    log_interval       = train_cfg.get("log_interval", 10)
    eval_interval      = train_cfg.get("eval_interval", 100)
    ckpt_interval      = train_cfg.get("checkpoint_interval", 500)

    sim_dict = {
        "nmnist_root":       sim_cfg.get("nmnist_root", "data/NMNIST"),
        "num_bins":          sim_cfg.get("num_bins", 100),
        "train_eval_samples": sim_cfg.get("train_eval_samples", 50),
        "full_eval_samples":  sim_cfg.get("full_eval_samples", 500),
        "eval_split":         sim_cfg.get("eval_split", "Test"),
    }

    buffer = RolloutBuffer()

    # Running averages for logging
    recent_rewards: list[float] = []
    recent_acc: list[float] = []

    print("=" * 60)
    print("SpikeGCPN training")
    print(f"  Episodes : {max_episodes}")
    print(f"  Max steps: {max_steps}")
    print(f"  N-MNIST  : {sim_dict['nmnist_root']}")
    print("=" * 60)

    for ep in range(1, max_episodes + 1):
        t0 = time.time()

        stats = run_episode(
            env, encoder, policy, value_fn,
            buffer, reward_cfg, sim_dict, max_steps,
        )

        ppo_stats = trainer.update(buffer)
        buffer.clear()

        recent_rewards.append(stats["ep_reward"])
        recent_acc.append(stats["accuracy"])

        if ep % log_interval == 0:
            mean_r = sum(recent_rewards) / len(recent_rewards)
            mean_a = sum(recent_acc) / len(recent_acc)
            recent_rewards.clear()
            recent_acc.clear()
            elapsed = time.time() - t0
            print(
                f"ep {ep:>6d} | "
                f"reward {mean_r:+.4f} | "
                f"acc {mean_a*100:5.1f}% | "
                f"hidden {stats['num_hidden']:>3d} | "
                f"edges {stats['num_edges']:>4d} | "
                f"pl {ppo_stats['policy_loss']:+.4f} | "
                f"vl {ppo_stats['value_loss']:.4f} | "
                f"ent {ppo_stats['entropy']:.3f} | "
                f"{elapsed:.1f}s"
            )

        if ep % eval_interval == 0:
            print(f"\n--- Full eval at episode {ep} ---")
            obs, _ = env.reset()
            # Run a full episode greedily (no grad) and evaluate
            with torch.no_grad():
                for _ in range(max_steps):
                    H = encoder(obs.x, obs.edge_index, obs.edge_attr)
                    action, _ = policy.sample_actions(H)
                    obs, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
            acc_full = quick_accuracy(
                graph=obs,
                nmnist_root=sim_dict["nmnist_root"],
                num_bins=sim_dict["num_bins"],
                n_samples=sim_dict["full_eval_samples"],
                num_source=env.num_source,
                num_sink=env.num_sink,
            )
            print(f"  Accuracy ({sim_dict['full_eval_samples']} samples): {acc_full*100:.2f}%")
            print()

        if ep % ckpt_interval == 0:
            ckpt_path = ckpt_dir / f"ep{ep:06d}.pt"
            trainer.save(str(ckpt_path))
            print(f"  [ckpt] saved {ckpt_path}")

    # Final checkpoint
    trainer.save(str(ckpt_dir / "final.pt"))
    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train SpikeGCPN")
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to a checkpoint to resume from",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
