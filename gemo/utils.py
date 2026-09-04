"""
utils
=====
Small shared helpers used by `train.py` and `inference.py`:
reproducibility, device resolution, checkpointing, label-map I/O, a
running-average meter, and streaming feature-normalization statistics.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

logger = logging.getLogger("utils")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    """Resolve 'auto' to 'cuda' if available, else 'cpu'; pass through anything else."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_label_map(label_map: Dict[str, int], path: str) -> None:
    """
    Persist the tract_name -> class_index mapping alongside checkpoints,
    so inference-time class names always match what the model was
    trained on (see `label_map_to_class_names`).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)


def load_label_map(path: str) -> Dict[str, int]:
    with open(path, "r") as f:
        return json.load(f)


def label_map_to_class_names(label_map: Dict[str, int]) -> List[str]:
    """
    Invert a tract_name -> class_index mapping into an ordered
    class_names list, suitable for `streamline_model.classify_tractogram`.
    """
    names: List[Optional[str]] = [None] * len(label_map)
    for name, idx in label_map.items():
        names[idx] = name
    if any(n is None for n in names):
        raise ValueError(f"label_map indices are not a contiguous 0..N-1 range: {label_map}")
    return names  # type: ignore[return-value]


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    best_val_accuracy: float = 0.0,
) -> None:
    """
    Save a full training checkpoint (model + optimizer + progress) so
    training can resume exactly where it left off (see `train.py`,
    which always writes `last.pt` after every epoch and reloads it on
    restart).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_val_accuracy": best_val_accuracy,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)
    logger.info("Saved checkpoint to %s (epoch %d, best_val_accuracy=%.4f)", path, epoch, best_val_accuracy)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict:
    """
    Load a checkpoint saved by `save_checkpoint`, restoring model (and
    optimizer, if given) state in place. Returns the checkpoint dict so
    the caller can pick up `epoch` / `best_val_accuracy` bookkeeping.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info(
        "Loaded checkpoint from %s (epoch %d, best_val_accuracy=%.4f)",
        path,
        checkpoint.get("epoch", 0),
        checkpoint.get("best_val_accuracy", 0.0),
    )
    return checkpoint


class AverageMeter:
    """Tracks a running average of a scalar value (loss, accuracy, ...)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


def compute_feature_normalization_stats(
    features_h5_path: str,
    feature_names: Sequence[str],
    chunk_size: int = 200_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-feature mean/std over the *entire* `features_h5_path`
    file in one streaming pass (read in `chunk_size`-row chunks, so the
    whole file -- potentially ~100,000,000 rows -- is never loaded into
    memory at once), for `StreamlineClassifier.set_feature_normalization_stats`.

    Parameters
    ----------
    features_h5_path : str
    feature_names : sequence of str
        Must match `streamline_model.DEFAULT_FEATURE_NAMES` order used
        for training.
    chunk_size : int
        Rows read per chunk.

    Returns
    -------
    tuple(mean, std)
        Each a torch.float32 tensor of shape (len(feature_names),).
    """
    with h5py.File(features_h5_path, "r") as h5file:
        n_rows = h5file["streamline_name"].shape[0]
        sums = np.zeros(len(feature_names), dtype=np.float64)
        sums_sq = np.zeros(len(feature_names), dtype=np.float64)
        count = 0

        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            block = np.stack([h5file[name][start:end] for name in feature_names], axis=1).astype(np.float64)
            sums += block.sum(axis=0)
            sums_sq += (block**2).sum(axis=0)
            count += end - start

    if count == 0:
        raise ValueError(f"{features_h5_path} contains no rows to compute normalization statistics from.")

    mean = sums / count
    variance = np.clip(sums_sq / count - mean**2, 1e-12, None)
    std = np.sqrt(variance)

    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


def compute_class_statistics(
    trk_dir: str,
    label_map: Dict[str, int],
    separator: str = "__",
) -> Tuple[int, Dict[str, Dict[str, int]]]:
    """
    Scan `trk_dir` and report, per tract class, how many subjects have
    that tract and its total streamline count across the whole
    dataset -- the dataset-imbalance picture printed at the start of
    `train.py`.

    Uses each .trk file's cheap header count (never loading the actual
    streamline data), so this is fast even across ~100,000,000
    streamlines -- it's a small, fixed number of header reads (one per
    .trk file), not a scan of the data itself. Counts are therefore the
    *raw* per-file streamline count (matching what a person would
    expect from "how many streamlines does this tract have"), which
    may be a handful more than the *valid* count actually usable for
    training (a few streamlines per file are typically dropped as
    invalid during offline feature extraction) -- close enough for this
    reporting/weighting purpose, and far cheaper than scanning
    `features.h5`.

    Parameters
    ----------
    trk_dir : str
    label_map : dict[str, int]
        From `streamline_model.build_label_mapping`; only tracts
        appearing in this map are counted (keeps this in sync with
        whatever classes the model is actually being trained on).
    separator : str

    Returns
    -------
    tuple(n_subjects, class_stats)
        `n_subjects` is the total number of subjects found in `trk_dir`.
        `class_stats` maps ``tract_name -> {"n_subjects": int, "n_streamlines": int}``.
    """
    # Imported here (not at module load time) to avoid a hard import-time
    # dependency on internals of two modules this file otherwise doesn't need.
    from .xyz2RGB import group_trk_files_by_subject
    from .streamline_features import _count_streamlines_fast

    subject_groups = group_trk_files_by_subject(trk_dir, separator=separator)
    n_subjects = len(subject_groups)

    class_stats: Dict[str, Dict[str, int]] = {name: {"n_subjects": 0, "n_streamlines": 0} for name in label_map}

    for files in subject_groups.values():
        for trk_path in files:
            file_stem = os.path.splitext(os.path.basename(trk_path))[0]
            parts = file_stem.split(separator, 1)
            tract_name = parts[1] if len(parts) == 2 else None
            if tract_name is None or tract_name not in class_stats:
                continue

            count = _count_streamlines_fast(trk_path)
            if count is None:
                logger.warning("Could not read a header streamline count for %s; treating it as 0 for stats.", trk_path)
                count = 0

            if count > 0:
                class_stats[tract_name]["n_subjects"] += 1
            class_stats[tract_name]["n_streamlines"] += count

    return n_subjects, class_stats


def log_dataset_summary(
    n_subjects: int,
    label_map: Dict[str, int],
    class_stats: Dict[str, Dict[str, int]],
) -> None:
    """
    Log the dataset composition in class-index order, e.g.::

        Detected 78 subjects in the dataset.
        AF_L: 75 tracts, 12000000 streamlines
        AF_R: 74 tracts, 11023988 streamlines
        ...
        UF_R: 59 tracts, 5923018 streamlines
    """
    logger.info("Detected %d subjects in the dataset.", n_subjects)
    for tract_name in sorted(label_map, key=lambda name: label_map[name]):
        stats = class_stats.get(tract_name, {"n_subjects": 0, "n_streamlines": 0})
        logger.info("  %s: %d tracts, %d streamlines", tract_name, stats["n_subjects"], stats["n_streamlines"])


def compute_class_weights(
    label_map: Dict[str, int],
    class_stats: Dict[str, Dict[str, int]],
    scheme: str = "inverse_sqrt",
) -> torch.Tensor:
    """
    Compute per-class loss weights from `compute_class_statistics`'s
    output, ordered by class index (ready for
    `nn.CrossEntropyLoss(weight=...)`).

    Parameters
    ----------
    label_map : dict[str, int]
    class_stats : dict
        From `compute_class_statistics`.
    scheme : {"inverse_sqrt", "inverse"}
        "inverse_sqrt" (default) weights each class by
        ``1 / sqrt(count)``, then normalizes so weights average to 1.
        This is the usual middle ground for severe imbalance (e.g.
        22,500,000 vs. 3,500,000 streamlines): plain inverse-frequency
        ("inverse") weighting over-corrects at this scale, pushing the
        rarest classes' gradients to dominate training and destabilizing
        it; inverse-sqrt still meaningfully boosts rare classes without
        that blow-up.

    Returns
    -------
    torch.Tensor, shape (num_classes,), float32
    """
    n_classes = len(label_map)
    counts = np.ones(n_classes, dtype=np.float64)  # floor of 1 avoids div-by-zero for an empty class
    for tract_name, class_index in label_map.items():
        counts[class_index] = max(1, class_stats.get(tract_name, {}).get("n_streamlines", 0))

    if scheme == "inverse_sqrt":
        raw_weights = 1.0 / np.sqrt(counts)
    elif scheme == "inverse":
        raw_weights = 1.0 / counts
    else:
        raise ValueError(f"Unknown scheme: {scheme!r}")

    normalized = raw_weights / raw_weights.mean()
    return torch.tensor(normalized, dtype=torch.float32)


__all__ = [
    "set_seed",
    "resolve_device",
    "save_label_map",
    "load_label_map",
    "label_map_to_class_names",
    "save_checkpoint",
    "load_checkpoint",
    "AverageMeter",
    "compute_feature_normalization_stats",
    "compute_class_statistics",
    "log_dataset_summary",
    "compute_class_weights",
]
