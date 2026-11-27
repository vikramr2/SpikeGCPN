import numpy as np
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, Optional

# N-MNIST sensor dimensions and derived constants.
# Each pixel fires ON-events (polarity=1) and OFF-events (polarity=0) separately,
# so we expose 2 * H * W source channels — one per (pixel, polarity) pair.
HEIGHT = 34
WIDTH = 34
NUM_POLARITIES = 2
NUM_SOURCE_NODES = HEIGHT * WIDTH * NUM_POLARITIES  # 2312
NUM_CLASSES = 10


def grid_size(patch: int = 1) -> int:
    """Patches per side when the 34×34 sensor is pooled into patch×patch blocks."""
    return -(-WIDTH // patch)


def num_source_nodes(patch: int = 1) -> int:
    """Source channels for a given patch size: 2312 at patch=1, 162 at patch=4."""
    return NUM_POLARITIES * grid_size(patch) ** 2


@dataclass
class NMNISTSample:
    spike_train: np.ndarray  # (num_bins, num_source_nodes(patch)), dtype bool
    label: int


def _read_bin(path: str):
    """
    Parse a 5-bytes-per-event N-MNIST binary file.

    Format (per event):
      byte 0 : x address  (0–33)
      byte 1 : y address  (0–33)
      byte 2 : polarity   (bit 0: 0=OFF, 1=ON)
      bytes 3-4 : timestamp in µs, big-endian uint16

    Timestamps are only 16 bits and wrap at 65 535 µs (~65 ms), but an
    N-MNIST sample spans ~300 ms with three saccades, so we unwrap by
    detecting drops and adding 65 536 on each wrap.
    """
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    n_events = len(raw) // 5
    raw = raw[: n_events * 5].reshape(n_events, 5)

    x = raw[:, 0].astype(np.int32)
    y = raw[:, 1].astype(np.int32)
    p = (raw[:, 2] & 0x01).astype(np.int32)
    ts_raw = ((raw[:, 3].astype(np.int64) << 8) | raw[:, 4])

    # Unwrap 16-bit timestamp rollovers: every drop adds 65 536 to all later events.
    wraps = np.concatenate([[0], np.cumsum(np.diff(ts_raw) < 0)])
    ts = ts_raw + wraps.astype(np.int64) * 65536

    return x, y, p, ts


def load_sample(path: str, num_bins: int = 100, patch: int = 1) -> np.ndarray:
    """
    Load one N-MNIST .bin file and return a binned spike train.

    Returns
    -------
    spike_train : np.ndarray, shape (num_bins, num_source_nodes(patch)), dtype bool
        spike_train[t, p * G*G + (y//patch)*G + x//patch] is True if any pixel of
        that (patch, polarity) fired in bin t, with G = grid_size(patch).
        patch=1 is the raw pixel layout.
    """
    x, y, p, ts = _read_bin(path)
    G = grid_size(patch)
    spike_train = np.zeros((num_bins, num_source_nodes(patch)), dtype=bool)

    if len(ts) == 0:
        return spike_train

    t_min, t_max = ts[0], ts[-1]
    if t_max == t_min:
        t_max = t_min + 1

    bins = ((ts - t_min) * num_bins // (t_max - t_min + 1)).clip(0, num_bins - 1).astype(np.int32)
    node_idx = p * (G * G) + (y // patch) * G + (x // patch)
    spike_train[bins, node_idx] = True

    return spike_train


def iter_dataset(
    root: str,
    split: str = "Test",
    num_bins: int = 100,
    max_samples: Optional[int] = None,
    per_class: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 0,
    patch: int = 1,
) -> Iterator[NMNISTSample]:
    """
    Yield NMNISTSample objects from a local N-MNIST directory.

    Parameters
    ----------
    root        : path to the NMNIST folder (contains Train/ and Test/)
    split       : "Train" or "Test"
    num_bins    : number of discrete time bins for spike encoding
    max_samples : cap on total samples yielded (applied after per_class)
    per_class   : if set, yield at most this many samples per class
                  (ensures a balanced evaluation set)
    shuffle     : shuffle files within each class before sampling
    seed        : RNG seed used when shuffle=True
    patch       : pool pixels into patch×patch blocks (1 = raw pixels)
    """
    import random as _random
    rng = _random.Random(seed)

    split_dir = Path(root) / split
    count = 0
    for label in range(NUM_CLASSES):
        class_dir = split_dir / str(label)
        if not class_dir.exists():
            continue
        files = sorted(class_dir.glob("*.bin"))
        if shuffle:
            rng.shuffle(files)
        if per_class is not None:
            files = files[:per_class]
        for bin_file in files:
            if max_samples is not None and count >= max_samples:
                return
            yield NMNISTSample(
                spike_train=load_sample(str(bin_file), num_bins, patch),
                label=label,
            )
            count += 1


def source_positions(num_source: int = NUM_SOURCE_NODES, patch: int = 1) -> np.ndarray:
    """
    (num_source, 3) float array of [x, y, polarity] per source node, with x and y
    the patch-centre coordinates scaled to [0, 1]. Matches the index layout of
    load_sample. If num_source does not match num_source_nodes(patch) (e.g. a toy
    env), nodes are laid out along x with y = polarity = 0.
    """
    if num_source != num_source_nodes(patch):
        pos = np.zeros((num_source, 3), dtype=np.float32)
        pos[:, 0] = np.arange(num_source) / max(1, num_source - 1)
        return pos
    G = grid_size(patch)
    idx = np.arange(num_source)
    p = idx // (G * G)
    rem = idx % (G * G)
    gy, gx = rem // G, rem % G
    scale = max(1, G - 1)
    return np.stack([gx / scale, gy / scale, p.astype(np.float64)], axis=1).astype(np.float32)


@lru_cache(maxsize=8)
def load_samples(
    root: str,
    split: str = "Test",
    num_bins: int = 100,
    per_class: Optional[int] = None,
    max_samples: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 0,
    patch: int = 1,
) -> tuple[NMNISTSample, ...]:
    """
    Memoised, fully materialised version of iter_dataset. Reward evaluation
    reuses the same fixed subset every episode, so parsing it once matters.
    """
    return tuple(iter_dataset(
        root, split=split, num_bins=num_bins, max_samples=max_samples,
        per_class=per_class, shuffle=shuffle, seed=seed, patch=patch,
    ))
