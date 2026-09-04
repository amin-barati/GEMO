"""
dataset
=======
Thin, import-friendly wrapper around the training-time dataset defined in
`streamline_model.py` (`StreamlineDataset`, `build_label_mapping`), plus
the train/val subject-level split and dataset-construction logic that's
specific to training.

`StreamlineDataset` itself lives in `streamline_model.py` because it's
tightly coupled to the `xyz2RGB` / `streamline_features` internals already
imported there; this module keeps `train.py` and `config.py` decoupled
from that detail.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

from .config import Config
from .xyz2RGB import group_trk_files_by_subject
from .streamline_model import StreamlineDataset, build_label_mapping

__all__ = ["StreamlineDataset", "build_label_mapping", "train_val_subject_split", "build_datasets"]


def train_val_subject_split(
    trk_dir: str,
    val_fraction: float = 0.2,
    separator: str = "__",
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Split subjects (not individual streamlines) into train/validation
    sets. Splitting at the subject level -- rather than randomly across
    streamlines -- avoids leaking near-duplicate streamlines from the
    same subject/tract across the train/val boundary, which would
    otherwise inflate validation accuracy.

    Parameters
    ----------
    trk_dir : str
    val_fraction : float
        Fraction of subjects assigned to validation (default 0.2).
    separator : str
    seed : int
        RNG seed for the shuffle, so the split is reproducible.

    Returns
    -------
    tuple(train_subject_ids, val_subject_ids)
        Both sorted, for deterministic downstream ordering.
    """
    subject_groups = group_trk_files_by_subject(trk_dir, separator=separator)
    subject_ids = sorted(subject_groups.keys())

    rng = random.Random(seed)
    shuffled = subject_ids[:]
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    return train_ids, val_ids


def build_datasets(cfg: Config, label_map: dict) -> Tuple[StreamlineDataset, StreamlineDataset]:
    """
    Build the train and validation `StreamlineDataset`s from a `Config`
    (see `config.py`), using a subject-level split (`train_val_subject_split`).

    The validation dataset is built with `shuffle_buffer_size=0` and
    `shuffle_subjects=False` for a fully deterministic, reproducible
    validation pass every epoch.

    Returns
    -------
    tuple(train_dataset, val_dataset)
    """
    train_ids, val_ids = train_val_subject_split(
        cfg.data.trk_dir,
        val_fraction=cfg.data.val_fraction,
        separator=cfg.data.separator,
        seed=cfg.data.split_seed,
    )

    common_kwargs = dict(
        trk_dir=cfg.data.trk_dir,
        features_h5_path=cfg.data.features_h5_path,
        bounds_h5_path=cfg.data.bounds_h5_path,
        label_map=label_map,
        separator=cfg.data.separator,
        index_width=cfg.data.index_width,
        grid_size=cfg.data.grid_size,
        scales=np.array(cfg.data.scales),
        feature_names=cfg.data.feature_names,
        truncate=cfg.data.truncate,
    )

    train_dataset = StreamlineDataset(
        subject_ids=train_ids,
        shuffle_buffer_size=cfg.data.shuffle_buffer_size,
        max_streamlines_per_class_per_epoch=cfg.data.max_streamlines_per_class_per_epoch,
        seed=cfg.train.seed,
        **common_kwargs,
    )
    val_dataset = StreamlineDataset(
        subject_ids=val_ids,
        shuffle_buffer_size=0,  # deterministic validation pass
        shuffle_subjects=False,  # ditto -- no subject/file-order shuffling either
        # Same cap as training, applied to a fixed (epoch-invariant)
        # subsample since val_dataset's epoch never advances -- keeps
        # validation fast and its metric comparable across epochs
        # rather than timing out on a 22,000,000-streamline class every
        # single epoch. Pass None here instead if full-coverage
        # validation is worth the cost for your dataset size.
        max_streamlines_per_class_per_epoch=cfg.data.max_streamlines_per_class_per_epoch,
        seed=cfg.train.seed,
        **common_kwargs,
    )
    return train_dataset, val_dataset
