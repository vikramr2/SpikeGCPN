"""
One entry point for the reward / test data of either dataset, driven by the
`simulator:` config section (shared by scripts/train.py and the analysis scripts).

  simulator.dataset: nmnist | shd
  nmnist: nmnist_root, num_bins, reward_split (Train), test_split (Test); patch from env.patch_size
  shd   : shd_root, num_bins, bin_ms, channel_pool, reward_split (train), test_split (test)

The reward split provides two disjoint per-class subsets: inner.train_per_class
samples to fit the readout and simulator.val_per_class for the reward.
"""

from __future__ import annotations

from spikegcpn.data import nmnist, shd
from spikegcpn.train.inner import split_per_class


def num_classes(cfg: dict) -> int:
    return shd.NUM_CLASSES if cfg.get("simulator", {}).get("dataset", "nmnist") == "shd" else nmnist.NUM_CLASSES


def num_sources(cfg: dict) -> int:
    s = cfg.get("simulator", {})
    if s.get("dataset", "nmnist") == "shd":
        return shd.num_source_nodes(int(s.get("channel_pool", 5)))
    return nmnist.num_source_nodes(int(cfg.get("env", {}).get("patch_size", 1)))


def load_splits(cfg: dict, test_per_class: int | None = None):
    """(train_samples, val_samples, test_samples, num_classes)."""
    s, i = cfg.get("simulator", {}), cfg.get("inner", {})
    n_train, n_val = int(i.get("train_per_class", 100)), int(s.get("val_per_class", 50))
    n_test = int(test_per_class or s.get("test_per_class", 50))
    bins, seed = int(s.get("num_bins", 50)), int(s.get("data_seed", 0))
    if s.get("dataset", "nmnist") == "shd":
        root, pool, bin_ms = s.get("shd_root", "data/SHD"), int(s.get("channel_pool", 5)), float(s.get("bin_ms", 10.0))
        pool_samples = shd.load_samples(root, s.get("reward_split", "train"), bins, n_train + n_val, seed, pool, bin_ms)
        test = shd.load_samples(root, s.get("test_split", "test"), bins, n_test, seed, pool, bin_ms)
    else:
        root, patch = s.get("nmnist_root", "data/NMNIST"), int(cfg.get("env", {}).get("patch_size", 1))
        pool_samples = nmnist.load_samples(root, s.get("reward_split", "Train"), bins, n_train + n_val, None, True,
                                           seed, patch)
        test = nmnist.load_samples(root, s.get("test_split", "Test"), bins, n_test, None, True, seed, patch)
    train, val = split_per_class(pool_samples, n_train)
    return train, val, test, num_classes(cfg)
