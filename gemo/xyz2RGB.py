"""
xyz2RGB
=======

Convert the streamlines of TrackVis (.trk) tractography files into
12 x 12 x 3 RGB images.

Encoding scheme
----------------
- Each streamline is a sequence of 3D points (x, y, z).
- x -> R channel, y -> G channel, z -> B channel.
- Coordinates are first rescaled to the 0-255 integer range (uint8).
- Each image holds 12 * 12 = 144 points, laid out row-major
  (point 0 -> pixel (0, 0), point 1 -> pixel (0, 1), ..., point 143 -> pixel (11, 11)).
- If a streamline has fewer than 144 points (e.g. 120), the remaining
  pixels (e.g. 24) are filled with black, i.e. (0, 0, 0).
- If a streamline has more than 144 points, it is truncated to the
  first 144 points (a warning is emitted) -- see `truncate` if you
  would rather this raise instead.

A single TRK file with n streamlines therefore yields n images, each of
shape (12, 12, 3), dtype uint8.

Subject-wise directory processing
----------------------------------
A directory can contain multiple .trk files belonging to multiple
subjects, where files of the same subject share a common prefix before
the tract name, e.g.:

    sub-1254__SLF_L.trk
    sub-1254__ILF_R.trk
    sub-1254__IFOF_R.trk
    sub-0098__SLF_L.trk
    ...

`process_trk_directory` groups files by subject (the part of the
filename before the separator, "__" by default), computes ONE global
min/max per axis (x, y, z) using every streamline from every one of
that subject's files, and then normalizes every streamline belonging
to that subject with those shared bounds. Different subjects are
normalized completely independently of one another. Output images are
named `<trk_filename_without_extension>_sl_<index>.<ext>`, where
`<index>` is the streamline's position within its source .trk file,
e.g. `sub-1254__SLF_L_sl_0000000.png`.

Precomputed normalization metadata
------------------------------------
Computing a subject's bounds requires reading every streamline of every
one of its .trk files (`compute_subject_bounds`). For large datasets
this should be done **once** and cached, not repeated on every training
epoch or every inference run. `save_bounds_metadata_h5` /
`load_bounds_metadata_h5` persist a ``{subject_id: (mins, maxs)}``
mapping to a small HDF5 file (one row per subject: `subject_id`, `xmin`,
`xmax`, `ymin`, `ymax`, `zmin`, `zmax`), so downstream code (e.g. the
`streamline_features` offline feature extractor, or a training-time
data loader generating RGB images on the fly) can load bounds once at
startup and reuse them for every streamline without ever re-scanning
the raw .trk files.

Typical usage
-------------
>>> from gemo.xyz2RGB import trk_to_images, process_trk_directory

>>> # single file, normalized using only its own streamlines
>>> images = trk_to_images("sub-1254__SLF_L.trk")
>>> images.shape
(n, 12, 12, 3)

>>> # a whole directory of subjects, each normalized subject-wise
>>> process_trk_directory("trk_input_dir", "rgb_output_dir")

>>> # precompute bounds once, reuse everywhere afterward
>>> from gemo.xyz2RGB import group_trk_files_by_subject, compute_subject_bounds, save_bounds_metadata_h5, load_bounds_metadata_h5
>>> groups = group_trk_files_by_subject("trk_input_dir")
>>> bounds = {sid: compute_subject_bounds(files) for sid, files in groups.items()}
>>> save_bounds_metadata_h5(bounds, "subject_bounds.h5")
>>> bounds_reloaded = load_bounds_metadata_h5("subject_bounds.h5")  # no .trk file ever reopened
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

IMAGE_SIDE = 12
POINTS_PER_IMAGE = IMAGE_SIDE * IMAGE_SIDE  # 144


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_streamlines(trk_path: str) -> List[np.ndarray]:
    """
    Load a .trk file and return its streamlines.

    Parameters
    ----------
    trk_path : str
        Path to the .trk file.

    Returns
    -------
    list of np.ndarray
        Each element is an (N_i, 3) float array of the (x, y, z)
        coordinates of one streamline.
    """
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "nibabel is required to read .trk files. Install it with "
            "`pip install nibabel`."
        ) from exc

    if not os.path.isfile(trk_path):
        raise FileNotFoundError(f"TRK file not found: {trk_path}")

    tractogram = nib.streamlines.load(trk_path)
    streamlines = [np.asarray(s, dtype=np.float64) for s in tractogram.streamlines]
    return streamlines


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def _minmax_scale_to_255(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Linearly map `values` from [vmin, vmax] to [0, 255] (float, not yet rounded)."""
    if vmax - vmin < 1e-12:
        # Degenerate case: every value is identical -> map to mid-gray.
        return np.full_like(values, 127.0, dtype=np.float64)
    return (values - vmin) / (vmax - vmin) * 255.0


def _global_bounds(streamlines: Sequence[np.ndarray]):
    """Compute per-axis (x, y, z) min/max across *all* streamlines in the file."""
    all_points = np.concatenate(streamlines, axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    return mins, maxs  # each shape (3,)


def normalize_streamline(
    streamline: np.ndarray,
    method: str = "global",
    bounds: Optional[tuple] = None,
) -> np.ndarray:
    """
    Rescale the (x, y, z) coordinates of a single streamline to 0-255.

    Parameters
    ----------
    streamline : np.ndarray, shape (N, 3)
    method : {"global", "per_streamline"}
        "global"         -> use `bounds` (min/max computed over the whole
                             TRK file) so that all resulting images share
                             the same coordinate scale.
        "per_streamline" -> rescale using this streamline's own min/max
                             per axis (each image uses its own scale).
    bounds : tuple(mins, maxs), required when method == "global"

    Returns
    -------
    np.ndarray, shape (N, 3), dtype uint8
    """
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


# --------------------------------------------------------------------------- #
# Streamline -> image
# --------------------------------------------------------------------------- #
def streamline_to_image(
    scaled_streamline: np.ndarray,
    side: int = IMAGE_SIDE,
    truncate: bool = True,
) -> np.ndarray:
    """
    Turn an (N, 3) array of already-0-255-scaled coordinates into a
    (side, side, 3) uint8 image, padding with black or truncating as needed.

    Parameters
    ----------
    scaled_streamline : np.ndarray, shape (N, 3), dtype uint8
    side : int
        Image side length (default 12, i.e. 144 points per image).
    truncate : bool
        If True (default) and the streamline has more than `side*side`
        points, only the first `side*side` points are kept (with a
        warning). If False, raises a ValueError instead.

    Returns
    -------
    np.ndarray, shape (side, side, 3), dtype uint8
    """
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


# --------------------------------------------------------------------------- #
# Top-level conversion
# --------------------------------------------------------------------------- #
def trk_to_images(
    trk_path: str,
    normalization: str = "global",
    output_dir: Optional[str] = None,
    image_format: str = "png",
    truncate: bool = True,
) -> np.ndarray:
    """
    Convert every streamline in a .trk file to a 12x12x3 RGB image.

    Parameters
    ----------
    trk_path : str
        Path to the input .trk file.
    normalization : {"global", "per_streamline"}
        How coordinates are rescaled to 0-255 before imaging.
        "global" (default) uses one shared min/max (computed over the
        whole tractogram) for all streamlines, so images stay
        comparable to one another. "per_streamline" rescales each
        streamline independently.
    output_dir : str, optional
        If given, each image is additionally saved to this directory as
        `streamline_<i>.<image_format>` (folder is created if needed).
    image_format : str
        File extension/format used when `output_dir` is given
        (anything Pillow supports, e.g. "png", "jpg").
    truncate : bool
        Passed through to `streamline_to_image`; controls behaviour for
        streamlines with more than 144 points.

    Returns
    -------
    np.ndarray, shape (n_streamlines, 12, 12, 3), dtype uint8
    """
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
    """
    Save a stack of images to disk as `<filename_prefix>_<index>.<image_format>`.

    Parameters
    ----------
    images : np.ndarray, shape (n, side, side, 3)
    output_dir : str
    image_format : str
    filename_prefix : str
        Prefix used before the zero-padded streamline index. Default
        "streamline" (used by the single-file `trk_to_images`).
    index_width : int, optional
        Width of the zero-padded index. If None, it is sized to fit
        `len(images) - 1` (minimum width 1).
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Pillow is required to save images. Install it with `pip install Pillow`."
        ) from exc

    os.makedirs(output_dir, exist_ok=True)
    if index_width is None:
        index_width = max(len(str(len(images) - 1)), 1)
    for i, img in enumerate(images):
        Image.fromarray(img, mode="RGB").save(
            os.path.join(output_dir, f"{filename_prefix}_{i:0{index_width}d}.{image_format}")
        )


# --------------------------------------------------------------------------- #
# Subject-wise directory processing
# --------------------------------------------------------------------------- #
def get_subject_id(trk_path: str, separator: str = "__") -> str:
    """
    Extract the subject identifier from a .trk filename: everything
    before the first occurrence of `separator`.

    e.g. "sub-1254__SLF_L.trk" -> "sub-1254"

    If `separator` is not found in the filename, the whole (extension-
    stripped) filename is treated as the subject id.
    """
    basename = os.path.splitext(os.path.basename(trk_path))[0]
    return basename.split(separator)[0]


def group_trk_files_by_subject(
    input_dir: str,
    separator: str = "__",
) -> dict:
    """
    Scan `input_dir` for .trk files and group their full paths by subject id.

    Parameters
    ----------
    input_dir : str
    separator : str
        Separator between the subject prefix and the tract name in the
        filename (default "__").

    Returns
    -------
    dict[str, list[str]]
        Maps subject_id -> sorted list of full .trk file paths
        belonging to that subject.
    """
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
    """
    Compute one global per-axis (x, y, z) min/max across every
    streamline of every file belonging to a single subject.

    Parameters
    ----------
    subject_files : iterable of str
        Full paths to all .trk files of one subject.

    Returns
    -------
    tuple(mins, maxs)
        Each of shape (3,), covering all of the subject's streamlines.
    """
    all_streamlines: List[np.ndarray] = []
    for trk_path in subject_files:
        all_streamlines.extend(load_streamlines(trk_path))

    if len(all_streamlines) == 0:
        raise ValueError(
            f"No streamlines found across subject files: {list(subject_files)}"
        )

    return _global_bounds(all_streamlines)


# --------------------------------------------------------------------------- #
# Precomputed normalization metadata (avoid rescanning .trk files)
# --------------------------------------------------------------------------- #
_BOUNDS_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
_BOUNDS_FIELDS = ("subject_id", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def save_bounds_metadata_h5(
    bounds_by_subject: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_h5_path: str,
) -> None:
    """
    Persist a ``{subject_id: (mins, maxs)}`` mapping to a small HDF5
    metadata file, one row per subject, with fields ``subject_id``,
    ``xmin``, ``xmax``, ``ymin``, ``ymax``, ``zmin``, ``zmax``.

    This is meant to be computed **once** (e.g. via
    `compute_subject_bounds` over each subject's .trk files) and then
    reloaded via `load_bounds_metadata_h5` by every subsequent
    training epoch, inference run, or offline feature-extraction pass,
    so that no .trk file is ever re-scanned just to recover coordinate
    bounds.

    Parameters
    ----------
    bounds_by_subject : dict[str, tuple(mins, maxs)]
        Each `mins`/`maxs` is a shape-(3,) array (x, y, z).
    output_h5_path : str
        Path to the output HDF5 file (overwritten if it exists).
    """
    subject_ids = sorted(bounds_by_subject.keys())
    n = len(subject_ids)

    xmin = np.empty(n, dtype=np.float64)
    xmax = np.empty(n, dtype=np.float64)
    ymin = np.empty(n, dtype=np.float64)
    ymax = np.empty(n, dtype=np.float64)
    zmin = np.empty(n, dtype=np.float64)
    zmax = np.empty(n, dtype=np.float64)

    for i, subject_id in enumerate(subject_ids):
        mins, maxs = bounds_by_subject[subject_id]
        xmin[i], ymin[i], zmin[i] = mins
        xmax[i], ymax[i], zmax[i] = maxs

    with h5py.File(output_h5_path, "w") as h5file:
        h5file.create_dataset("subject_id", data=np.array(subject_ids, dtype=object), dtype=_BOUNDS_STRING_DTYPE)
        h5file.create_dataset("xmin", data=xmin, dtype=np.float64)
        h5file.create_dataset("xmax", data=xmax, dtype=np.float64)
        h5file.create_dataset("ymin", data=ymin, dtype=np.float64)
        h5file.create_dataset("ymax", data=ymax, dtype=np.float64)
        h5file.create_dataset("zmin", data=zmin, dtype=np.float64)
        h5file.create_dataset("zmax", data=zmax, dtype=np.float64)


def load_bounds_metadata_h5(bounds_h5_path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load a subject-bounds metadata HDF5 file written by
    `save_bounds_metadata_h5` back into a
    ``{subject_id: (mins, maxs)}`` mapping, without touching any .trk
    file.

    Parameters
    ----------
    bounds_h5_path : str

    Returns
    -------
    dict[str, tuple(mins, maxs)]
        Each `mins`/`maxs` is a shape-(3,) float64 array (x, y, z).
    """
    if not os.path.isfile(bounds_h5_path):
        raise FileNotFoundError(f"Bounds metadata file not found: {bounds_h5_path}")

    bounds_by_subject: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    with h5py.File(bounds_h5_path, "r") as h5file:
        missing = [f for f in _BOUNDS_FIELDS if f not in h5file]
        if missing:
            raise ValueError(f"Bounds metadata file {bounds_h5_path} is missing fields: {missing}")

        subject_ids = h5file["subject_id"][:]
        xmin, xmax = h5file["xmin"][:], h5file["xmax"][:]
        ymin, ymax = h5file["ymin"][:], h5file["ymax"][:]
        zmin, zmax = h5file["zmin"][:], h5file["zmax"][:]

        for i, raw_id in enumerate(subject_ids):
            subject_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
            mins = np.array([xmin[i], ymin[i], zmin[i]], dtype=np.float64)
            maxs = np.array([xmax[i], ymax[i], zmax[i]], dtype=np.float64)
            bounds_by_subject[subject_id] = (mins, maxs)

    return bounds_by_subject


def streamline_to_rgb(
    points: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    truncate: bool = True,
) -> np.ndarray:
    """
    Convenience wrapper: convert one streamline's raw (N, 3) coordinate
    array directly into its 12x12x3 RGB image, given precomputed
    subject bounds (e.g. loaded via `load_bounds_metadata_h5`).
    Equivalent to `streamline_to_image(normalize_streamline(points,
    "global", bounds), truncate=truncate)`, provided as a single call
    for callers (training-time data loaders, inference pipelines) that
    only ever need the final image and never touch the intermediate
    scaled array.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
    bounds : tuple(mins, maxs)
        Subject-wide per-axis bounds, as returned by
        `compute_subject_bounds` or `load_bounds_metadata_h5`.
    truncate : bool
        Passed through to `streamline_to_image`.

    Returns
    -------
    np.ndarray, shape (12, 12, 3), dtype uint8
    """
    scaled = normalize_streamline(points, method="global", bounds=bounds)
    return streamline_to_image(scaled, side=IMAGE_SIDE, truncate=truncate)


def process_trk_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    separator: str = "__",
    image_format: str = "png",
    truncate: bool = True,
    index_width: int = 7,
) -> dict:
    """
    Convert every .trk file in `input_dir` into RGB images, using
    subject-wise global normalization.

    Pipeline (two passes):
      1. Group all .trk files in `input_dir` by subject (the filename
         prefix before `separator`), then compute ONE shared per-axis
         (x, y, z) min/max for each subject, using every streamline
         from every one of that subject's files.
      2. For each file, normalize each of its streamlines with its
         subject's bounds, then convert to a 12x12x3 image (padding
         short streamlines with black, truncating long ones as in
         `streamline_to_image`).

    Different subjects are normalized completely independently.

    Output images are named:
        "<trk_filename_without_extension>_sl_<index>.<image_format>"
    where <index> is the streamline's position within its source .trk
    file, zero-padded to `index_width` digits, e.g.:
        sub-1254__SLF_L_sl_0000000.png
        sub-1254__SLF_L_sl_0000001.png

    Parameters
    ----------
    input_dir : str
        Directory containing the .trk files (multiple subjects, each
        possibly with multiple tract files).
    output_dir : str, optional
        If given, images are saved here (created if needed). If None,
        images are only returned in memory, not written to disk.
    separator : str
        Separator between subject prefix and tract name in filenames
        (default "__").
    image_format : str
        Image file format used when saving (default "png").
    truncate : bool
        Passed through to `streamline_to_image` for streamlines with
        more than 144 points.
    index_width : int
        Zero-padding width for the per-file streamline index in output
        filenames (default 7, matching e.g. "_sl_0000000").

    Returns
    -------
    dict[str, np.ndarray]
        Maps each .trk filename (without extension) to its
        (n_streamlines, 12, 12, 3) uint8 image array.
    """
    # --- Pass 1: group by subject and compute subject-wise bounds ---
    subject_groups = group_trk_files_by_subject(input_dir, separator=separator)

    subject_bounds = {}
    for subject_id, files in subject_groups.items():
        subject_bounds[subject_id] = compute_subject_bounds(files)

    # --- Pass 2: normalize + generate images per file ---
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
    "streamline_to_rgb",
    "trk_to_images",
    "get_subject_id",
    "group_trk_files_by_subject",
    "compute_subject_bounds",
    "save_bounds_metadata_h5",
    "load_bounds_metadata_h5",
    "process_trk_directory",
    "IMAGE_SIDE",
    "POINTS_PER_IMAGE",
]
