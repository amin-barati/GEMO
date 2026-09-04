"""
==============================================================================
GEMO: A Deep Learning Method for Brain Fiber Classification and Tract
Segmentation Using Geometrical and Morphological Features

DOI:        https://doi.org/10.1016/j.acra.2026.07.024

Email       : Amin_br@yahoo.com
GitHub      : https://github.com/amin-barati/GEMO

==============================================================================

dataset
=======

"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

from config import Config
from xyz2RGB import group_trk_files_by_subject
from streamline_model import StreamlineDataset, build_label_mapping

__all__ = ["StreamlineDataset", "build_label_mapping", "train_val_subject_split", "build_datasets"]


def train_val_subject_split(
    trk_dir: str,
    val_fraction: float = 0.2,
    separator: str = "__",
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    
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
        shuffle_subjects=False,  
        max_streamlines_per_class_per_epoch=cfg.data.max_streamlines_per_class_per_epoch,
        seed=cfg.train.seed,
        **common_kwargs,
    )
    return train_dataset, val_dataset
