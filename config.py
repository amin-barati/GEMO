
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    trk_dir: str = 
    features_h5_path: str = 
    bounds_h5_path: str = 
    separator: str = "__"
    index_width: int =
    grid_size: int = 
    scales: Tuple[int, ...] = 
    feature_names: Tuple[str, ...] = (
        "length",
        "curvature",
        "tortuosity",
        "spectral_entropy",
        "fractal_dimension",
        "lacunarity",
    )
    truncate: bool = True
    shuffle_buffer_size: int = 
    val_fraction: float = 
    split_seed: int = 


@dataclass
class ModelConfig:
    num_handcrafted_features: int = 
    num_classes: int = 
    cnn_base_channels: int = 
    cnn_embedding_dim: int = 
    feature_hidden_dims: Tuple[int, ...] = 
    feature_embedding_dim: int = 
    classifier_hidden_dims: Tuple[int, ...] = 
    dropout: float = 
    fusion_mode: str = 


@dataclass
class TrainConfig:
    batch_size: int = 
    num_workers: int = 
    num_epochs: int = 
    learning_rate: float =
    weight_decay: float = 
    lr_step_size: int = 
    lr_gamma: float = 
    grad_clip_norm: float = 
    device: str = "auto"  
    checkpoint_dir: str = 
    log_every_n_steps: int =
    val_every_n_epochs: int = 
    seed: int = 


@dataclass
class InferenceConfig:
    threshold: float = 
    batch_size: int = 
    device: str = "auto"
    output_dir: str = 


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


__all__ = ["DataConfig", "ModelConfig", "TrainConfig", "InferenceConfig", "Config"]
