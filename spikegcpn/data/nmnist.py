import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional

# N-MNIST sensor dimensions and derived constants.
# Each pixel fires ON-events (polarity=1) and OFF-events (polarity=0) separately,
# so we expose 2 * H * W source channels — one per (pixel, polarity) pair.
HEIGHT = 34
WIDTH = 34
NUM_POLARITIES = 2
NUM_SOURCE_NODES = HEIGHT * WIDTH * NUM_POLARITIES  # 2312
NUM_CLASSES = 10


@dataclass
class NMNISTSample:
    spike_train: np.ndarray  # (num_bins, NUM_SOURCE_NODES), dtype bool
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

    # Unwrap 16-bit timestamp rollovers.
    ts = ts_raw.copy()
    offset = np.int64(0)
    for i in range(1, len(ts_raw)):
        if ts_raw[i] < ts_raw[i - 1]:
            offset += np.int64(65536)
        ts[i] = ts_raw[i] + offset

    return x, y, p, ts


def load_sample(path: str, num_bins: int = 100) -> np.ndarray:
    """
    Load one N-MNIST .bin file and return a binned spike train.

    Returns
    -------
    spike_train : np.ndarray, shape (num_bins, NUM_SOURCE_NODES), dtype bool
        spike_train[t, p * H*W + y*W + x] is True if that (pixel, polarity)
        fired at least once in bin t.
    """
    x, y, p, ts = _read_bin(path)
    spike_train = np.zeros((num_bins, NUM_SOURCE_NODES), dtype=bool)

    if len(ts) == 0:
        return spike_train

    t_min, t_max = ts[0], ts[-1]
    if t_max == t_min:
        t_max = t_min + 1

    bins = ((ts - t_min) * num_bins // (t_max - t_min + 1)).clip(0, num_bins - 1).astype(np.int32)
    node_idx = p * (HEIGHT * WIDTH) + y * WIDTH + x
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
                spike_train=load_sample(str(bin_file), num_bins),
                label=label,
            )
            count += 1
