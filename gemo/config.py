"""
config
======
Central configuration for training/evaluating the streamline classifier.

Everything that `train.py` / `inference.py` / `dataset.py` need is grouped
into four small dataclasses (`DataConfig`, `ModelConfig`, `TrainConfig`,
`InferenceConfig`), bundled into one `Config`. Command-line arguments in
`train.py`/`inference.py` override individual fields on top of these
defaults, so a fresh `Config()` is always a sane, runnable starting point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class DataConfig:
    trk_dir: str = "trk_dir"
    features_h5_path: str = "features.h5"
    bounds_h5_path: str = "bounds.h5"
    separator: str = "__"
    index_width: int = 7
    grid_size: int = 64
    scales: Tuple[int, ...] = (1, 2, 4, 8, 16)
    feature_names: Tuple[str, ...] = (
        "length",
        "curvature",
        "tortuosity",
        "spectral_entropy",
        "fractal_dimension",
        "lacunarity",
    )
    truncate: bool = True
    # Reservoir shuffle buffer size for the training dataset. Subject
    # order and per-subject tract-file round-robin order are already
    # reshuffled every epoch (StreamlineDataset.set_epoch), so this
    # buffer's job is just to smooth subject-to-subject transitions --
    # it no longer has to carry all the class-mixing on its own.
    shuffle_buffer_size: int = 2000
    # Cap on how many streamlines a single (subject, tract-file)
    # contributes per epoch -- the actual fix for class-quantity
    # imbalance (e.g. 22,500,000 vs. 3,500,000 streamlines). A
    # different random subsample of an oversized file is drawn each
    # epoch, so its full data still gets used over many epochs. None
    # disables capping. See StreamlineDataset's docstring for the
    # memory caveat.
    max_streamlines_per_class_per_epoch: Optional[int] = 100_000
    val_fraction: float = 0.2
    split_seed: int = 42


@dataclass
class ModelConfig:
    num_handcrafted_features: int = 6
    num_classes: int = 30
    cnn_base_channels: int = 32
    cnn_embedding_dim: int = 128
    feature_hidden_dims: Tuple[int, ...] = (32, 64)
    feature_embedding_dim: int = 64
    classifier_hidden_dims: Tuple[int, ...] = (256, 128)
    dropout: float = 0.2
    fusion_mode: str = "concat"


@dataclass
class TrainConfig:
    batch_size: int = 256
    num_workers: int = 4
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_step_size: int = 10
    lr_gamma: float = 0.5
    grad_clip_norm: float = 5.0
    device: str = "auto"  # resolved to "cuda" if available, else "cpu"
    checkpoint_dir: str = "checkpoints"
    log_every_n_steps: int = 50
    val_every_n_epochs: int = 1
    # Class-imbalance correction for the loss: per-class weights are
    # computed once from the dataset's actual streamline counts
    # (utils.compute_class_weights) and passed to
    # nn.CrossEntropyLoss(weight=...). "inverse_sqrt" (default) is a
    # middle ground for severe imbalance; "inverse" over-corrects at
    # counts like 22,500,000 vs. 3,500,000; "none" disables weighting.
    class_weight_scheme: str = "inverse_sqrt"
    seed: int = 0


@dataclass
class InferenceConfig:
    threshold: float = 0.7
    batch_size: int = 256
    device: str = "auto"
    output_dir: str = "classified_output"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


__all__ = ["DataConfig", "ModelConfig", "TrainConfig", "InferenceConfig", "Config"]
