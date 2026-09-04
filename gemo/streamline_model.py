"""

==============================================================================
GEMO: A Deep Learning Method for Brain Fiber Classification and Tract
Segmentation Using Geometrical and Morphological Features

DOI:        https://doi.org/10.1016/j.acra.2026.07.024

Email       : Amin_br@yahoo.com
GitHub      : https://github.com/amin-barati/GEMO

==============================================================================

streamline_model
=================

A compact PyTorch classifier for tractography streamlines that fuses
two complementary views of each streamline:

1. its 12x12x3 RGB image (from `xyz2RGB`), processed by a small CNN
   built specifically for that tiny spatial size, and
2. its handcrafted geometric/morphological feature vector (from
   `streamline_features`: length, curvature, tortuosity, spectral
   entropy, fractal dimension, lacunarity, ...), processed by a small
   MLP embedding network.

The two learned embeddings are concatenated and passed through a
fully-connected classification head to produce logits over 30 tract
classes.

Typical usage
-------------
>>> from streamline_model import StreamlineClassifier, classify_tractogram
>>> model = StreamlineClassifier(num_handcrafted_features=6, num_classes=30)
>>> model.load_state_dict(torch.load("streamline_classifier.pt"))
>>> classify_tractogram(
...     trk_path="sub-9999_wholebrain.trk",
...     model=model,
...     output_dir="sub-9999_classified",
... )
"""

from __future__ import annotations

import logging
import os
import random
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info
from tqdm import tqdm

import nibabel as nib
from nibabel.streamlines import Tractogram, TrkFile

from xyz2RGB import IMAGE_SIDE, group_trk_files_by_subject, load_bounds_metadata_h5, streamline_to_rgb
from streamline_features import (
    DEFAULT_GRID_SIZE,
    DEFAULT_SCALES,
    FIELD_DTYPES,
    calculate_polyline_length,
    compute_curvature,
    compute_fractal_dimension,
    compute_lacunarity,
    compute_spectral_entropy,
    compute_tortuosity,
    load_subject_row_index,
    rasterize_streamline,
    _load_subject_cache,
    _reduce_lacunarity,
)

logger = logging.getLogger("streamline_model")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


DEFAULT_FEATURE_NAMES: Tuple[str, ...] = (
    "length",
    "curvature",
    "tortuosity",
    "spectral_entropy",
    "fractal_dimension",
    "lacunarity",
)


# Training-time data loading

def build_label_mapping(input_dir: str, separator: str = "__") -> Dict[str, int]:

    subject_groups = group_trk_files_by_subject(input_dir, separator=separator)
    tract_names = set()
    for files in subject_groups.values():
        for trk_path in files:
            file_stem = os.path.splitext(os.path.basename(trk_path))[0]
            parts = file_stem.split(separator, 1)
            if len(parts) == 2:
                tract_names.add(parts[1])
    return {name: idx for idx, name in enumerate(sorted(tract_names))}


class _SubjectFeatureRows:

    def __init__(self, features_h5_path: str, feature_field_names: Sequence[str]) -> None:
        self.features_h5_path = features_h5_path
        self.feature_field_names = feature_field_names
        self.subject_row_index = load_subject_row_index(features_h5_path)

    def load(self, subject_id: str) -> Dict[str, Dict[str, float]]:

        row_range = self.subject_row_index.get(subject_id)
        if row_range is None:
            return {}
        start, end = row_range
        with h5py.File(self.features_h5_path, "r") as h5file:
            raw_names = h5file["streamline_name"][start:end]
            names = [n.decode() if isinstance(n, bytes) else n for n in raw_names]
            columns = {field: h5file[field][start:end] for field in self.feature_field_names}
        return {
            name: {field: columns[field][i] for field in self.feature_field_names} for i, name in enumerate(names)
        }


class StreamlineDataset(IterableDataset):

    def __init__(
        self,
        trk_dir: str,
        features_h5_path: str,
        bounds_h5_path: str,
        label_map: Dict[str, int],
        subject_ids: Optional[Sequence[str]] = None,
        separator: str = "__",
        index_width: int = 7,
        grid_size: int = DEFAULT_GRID_SIZE,
        scales: np.ndarray = DEFAULT_SCALES,
        feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
        truncate: bool = True,
        shuffle_buffer_size: int = 2000,
        shuffle_subjects: bool = True,
        max_streamlines_per_class_per_epoch: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.trk_dir = trk_dir
        self.features_h5_path = features_h5_path
        self.bounds_by_subject = load_bounds_metadata_h5(bounds_h5_path)
        self.label_map = label_map
        self.separator = separator
        self.index_width = index_width
        self.grid_size = grid_size
        self.scales = scales
        self.feature_names = feature_names
        self.truncate = truncate
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_subjects = shuffle_subjects
        self.max_streamlines_per_class_per_epoch = max_streamlines_per_class_per_epoch
        self.seed = seed
        self._current_epoch = 0

        all_groups = group_trk_files_by_subject(trk_dir, separator=separator)
        self.all_subject_groups = {sid: sorted(files) for sid, files in all_groups.items()}
        self.subject_ids = sorted(subject_ids) if subject_ids is not None else sorted(self.all_subject_groups)
        self.subject_groups = {
            sid: self.all_subject_groups[sid] for sid in self.subject_ids if sid in self.all_subject_groups
        }

        feature_field_names = [k for k in FIELD_DTYPES if k != "streamline_name"]
        self._feature_rows = _SubjectFeatureRows(features_h5_path, feature_field_names)
        if not self._feature_rows.subject_row_index:
            logger.warning(
                "%s has no subject row index; regenerate it with the current "
                "streamline_features.py (see load_subject_row_index) so that "
                "StreamlineDataset can range-load per subject.",
                features_h5_path,
            )

    def set_epoch(self, epoch: int) -> None:
        
        self._current_epoch = epoch

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, int]]:
        worker_info = get_worker_info()
        epoch_rng = random.Random(self.seed + self._current_epoch)

        
        ordered_subjects = list(self.subject_ids)
        if self.shuffle_subjects:
            epoch_rng.shuffle(ordered_subjects)

        if worker_info is None:
            my_subjects = ordered_subjects
        else:
            my_subjects = ordered_subjects[worker_info.id :: worker_info.num_workers]

        sample_rng = random.Random(
            self.seed + self._current_epoch * 1_000_003 + (worker_info.id if worker_info else 0)
        )
        buffer: List[Tuple[torch.Tensor, torch.Tensor, int]] = []

        for subject_id in my_subjects:
            files = self.subject_groups.get(subject_id, [])
            if not files:
                continue

            bounds = self.bounds_by_subject.get(subject_id)
            if bounds is None:
                logger.warning("No precomputed bounds for subject '%s'; skipping it.", subject_id)
                continue

            feature_rows = self._feature_rows.load(subject_id)
            if not feature_rows:
                logger.warning("No feature rows found for subject '%s'; skipping it.", subject_id)
                continue

            subject_cache = _load_subject_cache(files)

        
            file_entries: List[Dict[str, object]] = []
            for trk_path in files:
                file_stem = os.path.splitext(os.path.basename(trk_path))[0]
                parts = file_stem.split(self.separator, 1)
                tract_name = parts[1] if len(parts) == 2 else None
                label = self.label_map.get(tract_name) if tract_name is not None else None
                if label is None:
                    continue

                streamlines = subject_cache[trk_path]
                queue = [
                    (f"{file_stem}_sl_{local_index:0{self.index_width}d}", points)
                    for local_index, points in enumerate(streamlines)
                ]

                cap = self.max_streamlines_per_class_per_epoch
                if cap is not None and len(queue) > cap:
                   
                    queue = sample_rng.sample(queue, cap)
                if queue:
                    file_entries.append({"label": label, "queue": queue, "pos": 0})

            if self.shuffle_subjects:
                epoch_rng.shuffle(file_entries)  

            n_remaining = sum(len(entry["queue"]) for entry in file_entries)
            while n_remaining > 0:
                for entry in file_entries:
                    pos = entry["pos"]
                    queue = entry["queue"]
                    if pos >= len(queue):
                        continue
                    name, points = queue[pos]
                    entry["pos"] = pos + 1
                    n_remaining -= 1

                    row = feature_rows.get(name)
                    if row is None:

                        continue

                    feats = np.array([row[f] for f in self.feature_names], dtype=np.float32)
                    image = streamline_to_rgb(points, bounds, truncate=self.truncate)
                    sample = (image_to_tensor(image), torch.from_numpy(feats), entry["label"])

                    if self.shuffle_buffer_size <= 1:
                        yield sample
                        continue

                    buffer.append(sample)
                    if len(buffer) >= self.shuffle_buffer_size:
                        sample_rng.shuffle(buffer)
                        half = self.shuffle_buffer_size // 2
                        while len(buffer) > half:
                            yield buffer.pop()

            del subject_cache, feature_rows 

        sample_rng.shuffle(buffer)
        while buffer:
            yield buffer.pop()


 
# 1. CNN image branch 

class CNNFeatureExtractor(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        mid_channels = base_channels * 2

        self.conv_block = nn.Sequential(
            # 12x12 -> 12x12
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            # 12x12 -> 12x12
            nn.Conv2d(base_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 12x12 -> 6x6
            nn.MaxPool2d(kernel_size=2),
            # 6x6 -> 6x6
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 6x6 -> 1x1
            nn.AdaptiveAvgPool2d(1),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mid_channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.output_dim = embedding_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
     
        
        x = self.conv_block(image)
        return self.projection(x)


 
# 2. Handcrafted-feature branch
 

class HandcraftedFeatureEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int = len(DEFAULT_FEATURE_NAMES),
        hidden_dims: Sequence[int] = (32, 64),
        embedding_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()


        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim

        layers += [
            nn.Linear(prev_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        ]

        self.net = nn.Sequential(*layers)
        self.output_dim = embedding_dim

    def set_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> None:

        self.feature_mean.copy_(torch.as_tensor(mean, dtype=self.feature_mean.dtype))
        self.feature_std.copy_(torch.as_tensor(std, dtype=self.feature_std.dtype).clamp_min(eps))

    def forward(self, features: torch.Tensor) -> torch.Tensor:

        
        standardized = (features - self.feature_mean) / self.feature_std
        return self.net(standardized)


# 3. Fusion module 

class FeatureFusion(nn.Module):

    def __init__(self, mode: str = "concat") -> None:
        super().__init__()
        if mode not in ("concat",):
            raise ValueError(f"Unsupported fusion mode: {mode!r}")
        self.mode = mode

    def forward(self, image_embedding: torch.Tensor, feature_embedding: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            return torch.cat([image_embedding, feature_embedding], dim=1)
        raise ValueError(f"Unsupported fusion mode: {self.mode!r}")  # pragma: no cover

    @staticmethod
    def output_dim(image_embedding_dim: int, feature_embedding_dim: int, mode: str = "concat") -> int:
        if mode == "concat":
            return image_embedding_dim + feature_embedding_dim
        raise ValueError(f"Unsupported fusion mode: {mode!r}")


# 4. Classification head

class ClassificationHead(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        num_classes: int = 30,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:

        
        return self.net(fused)


# 5. Full model

class StreamlineClassifier(nn.Module):

    def __init__(
        self,
        num_handcrafted_features: int = len(DEFAULT_FEATURE_NAMES),
        num_classes: int = 30,
        cnn_base_channels: int = 32,
        cnn_embedding_dim: int = 128,
        feature_hidden_dims: Sequence[int] = (32, 64),
        feature_embedding_dim: int = 64,
        classifier_hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        fusion_mode: str = "concat",
    ) -> None:
        super().__init__()

        self.cnn = CNNFeatureExtractor(
            in_channels=3,
            base_channels=cnn_base_channels,
            embedding_dim=cnn_embedding_dim,
            dropout=dropout,
        )
        self.feature_encoder = HandcraftedFeatureEncoder(
            input_dim=num_handcrafted_features,
            hidden_dims=feature_hidden_dims,
            embedding_dim=feature_embedding_dim,
            dropout=dropout,
        )
        self.fusion = FeatureFusion(mode=fusion_mode)
        fused_dim = FeatureFusion.output_dim(cnn_embedding_dim, feature_embedding_dim, mode=fusion_mode)
        self.classifier = ClassificationHead(
            input_dim=fused_dim,
            hidden_dims=classifier_hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.num_classes = num_classes
        self.num_handcrafted_features = num_handcrafted_features

    def set_feature_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:

        self.feature_encoder.set_normalization_stats(mean, std)

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:

        
        image_embedding = self.cnn(image)
        feature_embedding = self.feature_encoder(features)
        fused = self.fusion(image_embedding, feature_embedding)
        return self.classifier(fused)

    @torch.no_grad()
    def predict_proba(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:

        logits = self.forward(image, features)
        return F.softmax(logits, dim=1)


def count_trainable_parameters(model: nn.Module) -> int:

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# Input preprocessing helpers shared by training and inference

def image_to_tensor(image: np.ndarray) -> torch.Tensor:

    
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def handcrafted_features_from_points(
    points: np.ndarray,
    grid_size: int = DEFAULT_GRID_SIZE,
    scales: np.ndarray = DEFAULT_SCALES,
    feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
) -> np.ndarray:
    
    
    length = calculate_polyline_length(points)
    curvature = compute_curvature(points)
    tortuosity = compute_tortuosity(points, length=length)
    spectral_entropy = compute_spectral_entropy(points)

    grid = rasterize_streamline(points, grid_size=grid_size)
    fractal_dimension = compute_fractal_dimension(grid, scales=scales)
    lacunarity = _reduce_lacunarity(compute_lacunarity(grid, scales=scales))
    del grid

    available = {
        "length": length,
        "curvature": curvature,
        "tortuosity": tortuosity if np.isfinite(tortuosity) else np.finfo(np.float32).max,
        "spectral_entropy": spectral_entropy,
        "fractal_dimension": fractal_dimension,
        "lacunarity": lacunarity,
    }
    return np.array([available[name] for name in feature_names], dtype=np.float32)



# Inference: classify a whole-brain tractogram into per-class .trk files
# 

def _streaming_file_bounds(trk_path: str) -> Tuple[np.ndarray, np.ndarray]:

    
    mins = np.full(3, np.inf, dtype=np.float64)
    maxs = np.full(3, -np.inf, dtype=np.float64)

    tractogram = nib.streamlines.load(trk_path, lazy_load=True)
    for streamline in tractogram.streamlines:
        points = np.asarray(streamline, dtype=np.float64)
        if points.shape[0] == 0:
            continue
        np.minimum(mins, points.min(axis=0), out=mins)
        np.maximum(maxs, points.max(axis=0), out=maxs)
    del tractogram

    if not np.all(np.isfinite(mins)):
        raise ValueError(f"No valid streamlines found in {trk_path} to compute bounds from.")

    return mins, maxs


def _write_class_trk_files(
    streamlines_by_label: Dict[str, List[np.ndarray]],
    output_dir: str,
    file_stem: str,
) -> Dict[str, str]:
    
    
    os.makedirs(output_dir, exist_ok=True)
    written_paths: Dict[str, str] = {}
    for label, streamlines in streamlines_by_label.items():
        if not streamlines:
            continue
        tractogram = Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        out_path = os.path.join(output_dir, f"{file_stem}_{label}.trk")
        TrkFile(tractogram).save(out_path)
        written_paths[label] = out_path
    return written_paths


@torch.no_grad()
def classify_tractogram(
    trk_path: str,
    model: StreamlineClassifier,
    output_dir: str,
    class_names: Optional[Sequence[str]] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    threshold: float = 0.7,
    batch_size: int = 256,
    device: str = "cpu",
    grid_size: int = DEFAULT_GRID_SIZE,
    scales: np.ndarray = DEFAULT_SCALES,
    feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
    truncate: bool = True,
) -> Dict[str, str]:
    """
    Classify every streamline of a whole-brain .trk file into
    `model.num_classes` tract classes, writing one output .trk file per
    class plus one "Unknown" file for streamlines whose top softmax
    probability is below `threshold`.
        
    """
    if class_names is None:
        class_names = [f"class_{i:02d}" for i in range(model.num_classes)]
    if len(class_names) != model.num_classes:
        raise ValueError(
            f"class_names has {len(class_names)} entries but model.num_classes={model.num_classes}"
        )

    was_training = model.training
    model.eval()
    model.to(device)

    if bounds is None:
        logger.info("No precomputed bounds given; computing bounds for %s in a streaming pass.", trk_path)
        bounds = _streaming_file_bounds(trk_path)

    file_stem = os.path.splitext(os.path.basename(trk_path))[0]
    streamlines_by_label: Dict[str, List[np.ndarray]] = {name: [] for name in class_names}
    streamlines_by_label["Unknown"] = []

    n_total = _count_streamlines_header(trk_path)

    image_batch: List[np.ndarray] = []
    feature_batch: List[np.ndarray] = []
    raw_points_batch: List[np.ndarray] = []

    def _flush_batch() -> None:
        if not image_batch:
            return
        images_tensor = torch.stack([image_to_tensor(img) for img in image_batch]).to(device)
        features_tensor = torch.from_numpy(np.stack(feature_batch)).to(device)

        probs = model.predict_proba(images_tensor, features_tensor).cpu().numpy()
        top_class = probs.argmax(axis=1)
        top_prob = probs.max(axis=1)

        for i in range(len(raw_points_batch)):
            if top_prob[i] < threshold:
                streamlines_by_label["Unknown"].append(raw_points_batch[i])
            else:
                streamlines_by_label[class_names[top_class[i]]].append(raw_points_batch[i])

        image_batch.clear()
        feature_batch.clear()
        raw_points_batch.clear()

    tractogram = nib.streamlines.load(trk_path, lazy_load=True)
    with tqdm(total=n_total, unit="sl", desc=f"Classifying {file_stem}", dynamic_ncols=True) as pbar:
        for streamline in tractogram.streamlines:
            points = np.asarray(streamline, dtype=np.float64)

            if points.shape[0] < 2 or not np.all(np.isfinite(points)):

                
                streamlines_by_label["Unknown"].append(points)
                pbar.update(1)
                continue

            try:
                image = streamline_to_rgb(points, bounds, truncate=truncate)
                features = handcrafted_features_from_points(
                    points, grid_size=grid_size, scales=scales, feature_names=feature_names
                )
            except Exception as exc:
                logger.warning("Could not featurize a streamline in %s (%s); routing to Unknown.", trk_path, exc)
                streamlines_by_label["Unknown"].append(points)
                pbar.update(1)
                continue

            image_batch.append(image)
            feature_batch.append(features)
            raw_points_batch.append(points)

            if len(image_batch) >= batch_size:
                _flush_batch()

            pbar.update(1)

        _flush_batch()  # remainder
    del tractogram

    written_paths = _write_class_trk_files(streamlines_by_label, output_dir, file_stem)
    counts = {label: len(streamlines) for label, streamlines in streamlines_by_label.items() if streamlines}
    logger.info("Classified %s -> %s (written to %s)", trk_path, counts, output_dir)

    model.train(was_training)
    return written_paths


def _count_streamlines_header(trk_path: str) -> Optional[int]:
    """Best-effort cheap streamline count from a .trk file's header (for the progress bar only)."""
    try:
        header = nib.streamlines.load(trk_path, lazy_load=True).header
    except Exception:
        return None
    count = header.get("nb_streamlines") if hasattr(header, "get") else None
    if isinstance(count, (int, np.integer)) and count > 0:
        return int(count)
    return None


__all__ = [
    "CNNFeatureExtractor",
    "HandcraftedFeatureEncoder",
    "FeatureFusion",
    "ClassificationHead",
    "StreamlineClassifier",
    "count_trainable_parameters",
    "image_to_tensor",
    "handcrafted_features_from_points",
    "classify_tractogram",
    "DEFAULT_FEATURE_NAMES",
]
