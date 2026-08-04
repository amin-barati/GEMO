
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

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_label_map(label_map: Dict[str, int], path: str) -> None:
    
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)


def load_label_map(path: str) -> Dict[str, int]:
    with open(path, "r") as f:
        return json.load(f)


def label_map_to_class_names(label_map: Dict[str, int]) -> List[str]:
 
    
    names: List[Optional[str]] = [None] * len(label_map)
    for name, idx in label_map.items():
        names[idx] = name
    if any(n is None for n in names):
        raise ValueError(f"label_map indices are not a contiguous 0..N-1 range: {label_map}")
    return names  


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    best_val_accuracy: float = 0.0,
) -> None:

    
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
    chunk_size: int = 
) -> Tuple[torch.Tensor, torch.Tensor]:

    
    with h5py.File(features_h5_path, "r") as h5file:
        n_rows = h5file["streamline_name"].shape[0]
        sums = np.zeros(len(feature_names), dtype=np.float64)
        sums_sq = np.zeros(len(feature_names), dtype=np.float64)
        count = 0

        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            block = np.stack([h5file[name][start:end] for name in feature_names], axis=1).astype(np.float64)
            sums += block.sum(axis=0)
            sums_sq += (block**?).sum(axis=0)
            count += end - start

    if count == 0:
        raise ValueError(f"{features_h5_path} contains no rows to compute normalization statistics from.")

    mean = sums / count
    variance = np.clip(sums_sq / count - mean**2, 1e-12, None)
    std = np.sqrt(variance)

    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


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
]
