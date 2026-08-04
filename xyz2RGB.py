"""
==============================================================================
GEMO: A Deep Learning Method for Brain Fiber Classification and Tract
Segmentation Using Geometrical and Morphological Features

DOI:        https://doi.org/10.1016/j.acra.2026.07.024

Email       : Amin_br@yahoo.com
GitHub      : https://github.com/amin-barati/GEMO

==============================================================================
"""


from __future__ import annotations

import os
import warnings
from typing import Iterable, List, Optional, Sequence

import numpy as np

IMAGE_SIDE = 12
POINTS_PER_IMAGE = IMAGE_SIDE * IMAGE_SIDE  


# Loading
def load_streamlines(trk_path: str) -> List[np.ndarray]:
 
    try:
        import nibabel as nib
    except ImportError as exc:  
        raise ImportError(
        ) from exc

    if not os.path.isfile(trk_path):
        raise FileNotFoundError(f"TRK file not found: {trk_path}")

    tractogram = nib.streamlines.load(trk_path)
    streamlines = [np.asarray(s, dtype=np.float64) for s in tractogram.streamlines]
    return streamlines



def _minmax_scale_to_255(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax - vmin < 1e-12:

        return np.full_like(values, 127.0, dtype=np.float64)
    return (values - vmin) / (vmax - vmin) * 255.0


def _global_bounds(streamlines: Sequence[np.ndarray]):
    all_points = np.concatenate(streamlines, axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    return mins, maxs  


def normalize_streamline(
    streamline: np.ndarray,
    method: str = "global",
    bounds: Optional[tuple] = None,
) -> np.ndarray:
    if method == "global":
        if bounds is None:
            raise ValueError("`bounds` must be provided when method='global'.")
        mins, maxs = bounds
    elif method == "per_streamline":
        mins = streamline.min(axis=0)
        maxs = streamline.max(axis=0)
    else:
        raise ValueError("method must be 'global' or 'per_streamline'.")

    scaled = np.empty_like(streamline, dtype=np.float64)
    for axis in range(3):
        scaled[:, axis] = _minmax_scale_to_255(streamline[:, axis], mins[axis], maxs[axis])

    scaled = np.clip(np.round(scaled), 0, 255).astype(np.uint8)
    return scaled


# Streamline -> image

def streamline_to_image(
    scaled_streamline: np.ndarray,
    side: int = IMAGE_SIDE,
    truncate: bool = True,
) -> np.ndarray:
    
    capacity = side * side
    n_points = scaled_streamline.shape[0]

    if n_points > capacity:
        if not truncate:
            raise ValueError(
                f"Streamline has {n_points} points, which exceeds the "
                f"{capacity}-point capacity of a {side}x{side} image."
            )
        warnings.warn(
            f"Streamline has {n_points} points > {capacity} capacity; "
            f"truncating to the first {capacity} points.",
            stacklevel=2,
        )
        used = scaled_streamline[:capacity]
    elif n_points < capacity:
        pad = np.zeros((capacity - n_points, 3), dtype=np.uint8)
        used = np.concatenate([scaled_streamline, pad], axis=0)
    else:
        used = scaled_streamline

    image = used.reshape(side, side, 3)
    return image


def trk_to_images(
    trk_path: str,
    normalization: str = "global",
    output_dir: Optional[str] = None,
    image_format: str = "png",
    truncate: bool = True,
) -> np.ndarray:
    streamlines = load_streamlines(trk_path)
    if len(streamlines) == 0:
        return np.empty((0, IMAGE_SIDE, IMAGE_SIDE, 3), dtype=np.uint8)

    bounds = _global_bounds(streamlines) if normalization == "global" else None

    images = []
    for streamline in streamlines:
        scaled = normalize_streamline(streamline, method=normalization, bounds=bounds)
        image = streamline_to_image(scaled, side=IMAGE_SIDE, truncate=truncate)
        images.append(image)

    images = np.stack(images, axis=0)

    if output_dir is not None:
        _save_images(images, output_dir, image_format)

    return images


def _save_images(
    images: np.ndarray,
    output_dir: str,
    image_format: str,
    filename_prefix: str = "streamline",
    index_width: Optional[int] = None,
) -> None:
    
    try:
        from PIL import Image
    except ImportError as exc:  
        raise ImportError(

        ) from exc

    os.makedirs(output_dir, exist_ok=True)
    if index_width is None:
        index_width = max(len(str(len(images) - 1)), 1)
    for i, img in enumerate(images):
        Image.fromarray(img, mode="RGB").save(
            os.path.join(output_dir, f"{filename_prefix}_{i:0{index_width}d}.{image_format}")
        )



def get_subject_id(trk_path: str, separator: str = "__") -> str:
    
    basename = os.path.splitext(os.path.basename(trk_path))[0]
    return basename.split(separator)[0]


def group_trk_files_by_subject(
    input_dir: str,
    separator: str = "__",
) -> dict:
    
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    groups: dict = {}
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(".trk"):
            continue
        full_path = os.path.join(input_dir, fname)
        subject_id = get_subject_id(full_path, separator=separator)
        groups.setdefault(subject_id, []).append(full_path)

    return groups


def compute_subject_bounds(subject_files: Iterable[str]) -> tuple:
   
    all_streamlines: List[np.ndarray] = []
    for trk_path in subject_files:
        all_streamlines.extend(load_streamlines(trk_path))

    if len(all_streamlines) == 0:
        raise ValueError(
            f"No streamlines found across subject files: {list(subject_files)}"
        )

    return _global_bounds(all_streamlines)


def process_trk_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    separator: str = "__",
    image_format: str = "png",
    truncate: bool = True,
    index_width: int = 7,
) -> dict:
    
    subject_groups = group_trk_files_by_subject(input_dir, separator=separator)

    subject_bounds = {}
    for subject_id, files in subject_groups.items():
        subject_bounds[subject_id] = compute_subject_bounds(files)

    results: dict = {}
    for subject_id, files in subject_groups.items():
        bounds = subject_bounds[subject_id]
        for trk_path in files:
            streamlines = load_streamlines(trk_path)
            file_stem = os.path.splitext(os.path.basename(trk_path))[0]

            if len(streamlines) == 0:
                images = np.empty((0, IMAGE_SIDE, IMAGE_SIDE, 3), dtype=np.uint8)
            else:
                images_list = []
                for streamline in streamlines:
                    scaled = normalize_streamline(streamline, method="global", bounds=bounds)
                    image = streamline_to_image(scaled, side=IMAGE_SIDE, truncate=truncate)
                    images_list.append(image)
                images = np.stack(images_list, axis=0)

            results[file_stem] = images

            if output_dir is not None and len(images) > 0:
                _save_images(
                    images,
                    output_dir,
                    image_format,
                    filename_prefix=f"{file_stem}_sl",
                    index_width=index_width,
                )

    return results


__all__ = [
    "load_streamlines",
    "normalize_streamline",
    "streamline_to_image",
    "trk_to_images",
    "get_subject_id",
    "group_trk_files_by_subject",
    "compute_subject_bounds",
    "process_trk_directory",
    "IMAGE_SIDE",
    "POINTS_PER_IMAGE",
]
