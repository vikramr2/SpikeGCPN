"""
Spiking Heidelberg Digits (Cramer et al., 2020): spoken digits 0–9 in English
and German (20 classes), converted to spikes by a cochlea model with 700
frequency channels. Files: data/SHD/shd_{train,test}.h5 from
https://zenkelab.org/datasets/ (each sample: spike times in seconds + channel ids).

Unlike N-MNIST, the class is carried by *when* each frequency band is active,
so a readout over a memoryless relay of the input should do poorly at fine
time scales; this is the task where recurrent reservoirs ought to matter.

Encoding: adjacent channels are pooled in groups of `pool` (OR within each
group) and time is binned at `bin_ms`; a source spikes in a bin if any of its
channels did. Utterances longer than num_bins · bin_ms are truncated (95% of
SHD is < 0.97 s). The test split has two speakers absent from the train split.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from spikegcpn.data.nmnist import NMNISTSample as SpikeSample

NUM_CLASSES = 20
NUM_CHANNELS = 700


def num_source_nodes(pool: int) -> int:
    return -(-NUM_CHANNELS // pool)


def bin_sample(times: np.ndarray, units: np.ndarray, num_bins: int, pool: int, bin_ms: float) -> np.ndarray:
    """bool [num_bins, ceil(700 / pool)]."""
    b = np.floor(times / (bin_ms / 1000.0)).astype(np.int64)
    keep = b < num_bins
    X = np.zeros((num_bins, num_source_nodes(pool)), dtype=bool)
    X[b[keep], units[keep].astype(np.int64) // pool] = True
    return X


@lru_cache(maxsize=8)
def load_samples(
    root: str,
    split: str = "train",
    num_bins: int = 100,
    per_class: int | None = None,
    seed: int = 0,
    pool: int = 5,
    bin_ms: float = 10.0,
) -> tuple[SpikeSample, ...]:
    """
    Class-ordered samples (all of class 0, then class 1, …). With per_class, a
    seeded random subset of that many per class; the order within a class is
    the seeded shuffle, so split_per_class(…, n) gives disjoint subsets.
    """
    import h5py

    with h5py.File(Path(root) / f"shd_{split}.h5", "r") as f:
        labels = f["labels"][:]
        times, units = f["spikes"]["times"], f["spikes"]["units"]
        rng = np.random.default_rng(seed)
        out = []
        for c in range(NUM_CLASSES):
            idx = np.nonzero(labels == c)[0]
            rng.shuffle(idx)
            if per_class is not None:
                idx = idx[:per_class]
            for i in idx:
                out.append(SpikeSample(spike_train=bin_sample(times[i], units[i], num_bins, pool, bin_ms), label=c))
    return tuple(out)
