"""
streamline_features
====================

Streaming, memory-efficient extraction of geometric and morphological
features from large collections of tractography (.trk) files, suitable
for datasets on the order of ~100,000,000 streamlines.

Design summary
--------------
- **Subject-wise file grouping is reused, not reimplemented.** Subject
  grouping (`group_trk_files_by_subject`) is imported directly from
  `xyz2RGB.py`, so the same deterministic subject/file ordering and
  `streamline_name` convention are used here.

- **Each .trk file is read from disk exactly once.** Rather than a
  purely lazy, one-streamline-at-a-time iterator (which under the hood
  can still involve repeated small reads), every file is loaded fully
  in a single call and its streamlines are cached in memory as a plain
  list, then processed sequentially. By default the cache scope is one
  **subject** at a time (every one of that subject's .trk files is
  loaded together, per `cache_scope="subject"`), which is what the
  original per-subject bounds computation already required reading
  anyway; the cache is released (`del`) before moving to the next
  subject, so at most one subject's streamlines are ever resident in
  memory -- the dataset as a whole is still processed in a streaming
  fashion. Pass `cache_scope="file"` to cache one file at a time
  instead, for a lower (but slightly less I/O-efficient) memory
  ceiling.

- **Subject coordinate bounds are precomputed once and reused, never
  rescanned.** Previous versions of this module (and `xyz2RGB.py`)
  recomputed a subject's min/max coordinates by re-reading all of that
  subject's .trk files every time they were needed. `xyz2RGB.py` now
  provides `save_bounds_metadata_h5` / `load_bounds_metadata_h5` to
  persist that computation to a small HDF5 file (`subject_id`, `xmin`,
  `xmax`, `ymin`, `ymax`, `zmin`, `zmax`) once, offline.
  `compute_and_save_bounds_metadata` below is that one-time offline
  step; `extract_features_from_directory` takes an optional
  `bounds_h5_path` and, if given, loads it once at startup with zero
  further .trk scanning. This is also the file a training-time or
  inference-time data loader should load once and reuse to generate
  `xyz2RGB` images (`streamline_to_rgb`) with normalization that is
  guaranteed identical to what this module used, without ever
  re-deriving bounds from raw coordinates itself.

- **Small write buffer, not an in-memory list of all features.** Rows
  are accumulated in a short-lived buffer (`flush_every` rows, default
  2000) and flushed to the HDF5 file, which uses resizable
  (chunked, ``maxshape=(None,)``) datasets. No Python list ever holds
  more than one buffer's worth of rows (plus, at most, one subject's
  cached streamlines -- see above).

- **Deterministic global order + resume.** Subjects are processed in
  sorted order, files within a subject in sorted order, and
  streamlines within a file in on-disk order. This fixed order lets
  the module resume purely by counting how many rows already exist in
  the output HDF5 file: it skips exactly that many streamlines (no
  file needs to be re-parsed for its *content*, only iterated past)
  and continues, so nothing is ever reprocessed or duplicated.

Geometric feature algorithms
-----------------------------
`calculate_polyline_length`, `compute_curvature`, and
`compute_tortuosity` are reference implementations (the exact required
code for these was not included in the original request). Each takes
an (N, 3) array of streamline points and returns a single float.

`compute_spectral_entropy` implements the exact formula supplied by
the user: Shannon entropy of the normalized power spectral density
(sum of squared FFT magnitudes across the x, y, z coordinate signals,
each with its mean removed), keeping only the positive-frequency half
of the spectrum.

Morphological feature algorithms
----------------------------------
`compute_fractal_dimension` and `compute_lacunarity` implement the
exact formulas supplied by the user, operating on a temporary 3D
binary occupancy grid built directly from the streamline's own
coordinates by `rasterize_streamline` (min-max normalized to that
streamline's own bounding box, then voxelized into a `grid_size**3`
grid; default 64). This grid is never written to disk and is
discarded immediately after these two features are computed. This
per-streamline normalization is intentionally independent of the
subject-wide bounds discussed above: fractal dimension and lacunarity
are scale-invariant shape descriptors of the streamline itself, while
the subject-wide bounds exist to give the `xyz2RGB` image its absolute
anatomical position within the subject.

`compute_lacunarity` (as supplied) returns one lacunarity value per
box-counting scale rather than a single scalar. Since the output
schema stores one `lacunarity` value per streamline (matching the
`fractal_dimension`, unchanged from the previous version), the
per-scale values are aggregated by taking their mean -- this
aggregation choice is not specified in the original formula and can be
changed via `_reduce_lacunarity` if a different reduction (e.g. a
single fixed scale) is preferred.

Output
------
A single HDF5 file (not one per subject) with resizable float32
datasets for the six numeric features plus a variable-length UTF-8
string dataset for `streamline_name`, named exactly as in
`xyz2RGB.py`'s file-naming convention, e.g.
``sub-1254__SLF_L_sl_0000000``.

Subject-level multiprocessing
-------------------------------
Feature computation is CPU-bound (FFT, box-counting, gliding-box) and
embarrassingly parallel across subjects, so `extract_features_from_directory`
distributes whole subjects across a `multiprocessing.Pool` (one subject
= one task; a worker process loads that subject's .trk files, computes
every streamline's row, and returns them -- individual streamlines are
never split across processes, avoiding millions of tiny tasks and their
scheduling overhead). The main process is the *only* one that ever
touches the output HDF5 file, and it always writes completed subjects
back in the same sorted-subject order tasks were submitted in (via
`Pool.imap`, which yields results in submission order even though
workers finish out of order), so the file's row order -- and therefore
resumability -- is identical to a single-process run. Pass
`use_multiprocessing=False` (or `n_workers=1`) to fall back to the
original single-process behavior, e.g. for debugging.

Typical usage
-------------
>>> from gemo.xyz2RGB import group_trk_files_by_subject, compute_subject_bounds
>>> from gemo.streamline_features import compute_and_save_bounds_metadata, extract_features_from_directory

>>> # One-time offline step: read every subject's .trk files once and
>>> # cache their coordinate bounds to disk.
>>> compute_and_save_bounds_metadata("trk_input_dir", "subject_bounds.h5")

>>> # Main feature extraction: loads subject_bounds.h5 once, never
>>> # rescans for bounds, and processes subjects in parallel across
>>> # CPU cores (one subject per worker task).
>>> extract_features_from_directory(
...     input_dir="trk_input_dir",
...     output_h5_path="features.h5",
...     bounds_h5_path="subject_bounds.h5",
...     n_workers=8,
... )
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy import fft
from skimage.measure import block_reduce
from tqdm import tqdm

import nibabel as nib

from .xyz2RGB import (
    compute_subject_bounds,
    group_trk_files_by_subject,
    load_bounds_metadata_h5,
    save_bounds_metadata_h5,
)

logger = logging.getLogger("streamline_features")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# Default box-counting / gliding-box scales for the morphological features.
DEFAULT_SCALES = np.array([1, 2, 4, 8, 16])
DEFAULT_GRID_SIZE = 64


# =============================================================================
# Geometric feature algorithms (reference implementations -- see module
# docstring: replace bodies with the exact required algorithms if these
# reference versions differ from what was intended).
# =============================================================================
def calculate_polyline_length(points: np.ndarray) -> float:
    """
    Total arc length of a 3D polyline: the sum of Euclidean distances
    between consecutive points.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)

    Returns
    -------
    float
    """
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return float(np.sum(seg_lengths))


def compute_curvature(points: np.ndarray) -> float:
    """
    Mean discrete (Menger) curvature over all consecutive point
    triplets of a 3D polyline.

    For each triplet (p0, p1, p2), the Menger curvature is
    ``4 * area(p0, p1, p2) / (|p0p1| * |p1p2| * |p2p0|)``, i.e. the
    reciprocal of the radius of the circle through the three points.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)

    Returns
    -------
    float
        Mean curvature over the streamline; 0.0 if fewer than 3 points.
    """
    if points.shape[0] < 3:
        return 0.0

    p0, p1, p2 = points[:-2], points[1:-1], points[2:]
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)

    cross = np.cross(p1 - p0, p2 - p0)
    area = 0.5 * np.linalg.norm(cross, axis=1)

    denom = a * b * c
    valid = denom > 1e-12
    curvature = np.zeros_like(denom)
    curvature[valid] = 4.0 * area[valid] / denom[valid]

    return float(np.mean(curvature)) if curvature.size else 0.0


def compute_tortuosity(points: np.ndarray, length: Optional[float] = None) -> float:
    """
    Arc-chord ratio: streamline arc length divided by the straight-line
    (Euclidean) distance between its two endpoints.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
    length : float, optional
        Precomputed arc length (from `calculate_polyline_length`); if
        not given it is computed here.

    Returns
    -------
    float
        1.0 for a perfectly straight streamline; larger values for
        more convoluted paths. Returns ``inf`` for a streamline whose
        endpoints coincide but which has nonzero arc length.
    """
    if points.shape[0] < 2:
        return 0.0

    chord = float(np.linalg.norm(points[-1] - points[0]))
    if length is None:
        length = calculate_polyline_length(points)

    if chord < 1e-9:
        return float("inf") if length > 1e-9 else 0.0

    return float(length / chord)


def compute_spectral_entropy(points: np.ndarray) -> float:
    """
    Compute spectral entropy from FFT of the 3D coordinate signals.

    Exact formula as supplied: the mean is removed from each of the
    x, y, z coordinate signals, their FFT magnitudes are combined into
    a single power spectral density (PSD), the positive-frequency half
    of the PSD is normalized into a probability distribution, and the
    Shannon entropy of that distribution is returned.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)

    Returns
    -------
    float
    """
    # Remove DC component (mean) from each coordinate
    x = points[:, 0] - np.mean(points[:, 0])
    y = points[:, 1] - np.mean(points[:, 1])
    z = points[:, 2] - np.mean(points[:, 2])

    # Compute FFT magnitudes and power spectral density (PSD)
    fft_x = np.abs(fft.fft(x))
    fft_y = np.abs(fft.fft(y))
    fft_z = np.abs(fft.fft(z))
    psd = fft_x**2 + fft_y**2 + fft_z**2

    # Normalize PSD to get probabilities (ignore negative frequencies)
    psd = psd[:len(psd)//2]  # Keep only positive frequencies
    psd_sum = np.sum(psd)
    if psd_sum <= 1e-12:
        # Numerical-stability guard (not part of the original formula):
        # a degenerate/constant signal has an all-zero PSD, which would
        # otherwise divide by zero.
        return 0.0
    psd_normalized = psd / psd_sum

    # Compute Shannon entropy
    entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-10))  # Avoid log(0)
    return float(entropy)


# =============================================================================
# Morphological feature algorithms (operate on a temporary 3D voxel grid
# rasterized directly from the streamline's own coordinates -- exact
# formulas as supplied by the user).
# =============================================================================
def rasterize_streamline(points: np.ndarray, grid_size: int = DEFAULT_GRID_SIZE) -> np.ndarray:
    """Convert a 3D streamline into a binary 3D grid."""
    # Normalize points to [0, 1]^3
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    range_coords = max_coords - min_coords
    range_coords[range_coords == 0] = 1e-10  # Handle zero-division

    normalized = (points - min_coords) / range_coords
    grid_coords = (normalized * (grid_size - 1)).astype(int)

    # Initialize grid and mark occupied voxels
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    for coord in grid_coords:
        grid[tuple(coord)] = True

    return grid


def compute_fractal_dimension(grid: np.ndarray, scales: np.ndarray = DEFAULT_SCALES) -> float:
    """Box-counting fractal dimension (slope of log(N) vs log(1/s))."""
    counts = []
    for s in scales:
        if s > min(grid.shape):
            continue
        # Downsample grid using max-pooling
        pooled = block_reduce(grid, block_size=(s, s, s), func=np.max)
        counts.append(np.sum(pooled))

    # Fit linear regression to log-log data
    valid = np.array(counts) > 0
    if np.sum(valid) < 2:
        return 0.0

    log_scales = np.log(1 / scales[valid])
    log_counts = np.log(np.array(counts)[valid])
    return float(np.polyfit(log_scales, log_counts, 1)[0])


def compute_lacunarity(grid: np.ndarray, scales: np.ndarray = DEFAULT_SCALES) -> Dict[int, float]:
    """Multi-scale lacunarity (variance/mean\u00b2 of box masses)."""
    lacunarities = {}
    for s in scales:
        if s > min(grid.shape):
            continue
        # Downsample grid using sum-pooling
        pooled = block_reduce(grid.astype(float), block_size=(s, s, s), func=np.sum)
        mass = pooled.flatten()
        mean_mass = np.mean(mass)
        if mean_mass == 0:
            lac = 0.0
        else:
            lac = np.var(mass) / (mean_mass ** 2)
        lacunarities[int(s)] = float(lac)
    return lacunarities


def _reduce_lacunarity(lacunarity_by_scale: Dict[int, float]) -> float:
    """
    Reduce the per-scale lacunarity dict returned by `compute_lacunarity`
    to the single scalar stored in the `lacunarity` HDF5 column.

    The supplied formula returns one value per box-counting scale; the
    output schema (matching the previous version) stores a single
    `lacunarity` value per streamline. This aggregation (the mean
    across all computed scales) is not specified by the original
    formula -- swap this function if a different reduction (e.g. a
    single fixed scale) is preferred.
    """
    if not lacunarity_by_scale:
        return 0.0
    return float(np.mean(list(lacunarity_by_scale.values())))


# =============================================================================
# HDF5 schema and I/O helpers
# =============================================================================
_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")

FIELD_DTYPES: Dict[str, object] = {
    "streamline_name": _STRING_DTYPE,
    "spectral_entropy": np.float32,
    "tortuosity": np.float32,
    "curvature": np.float32,
    "length": np.float32,
    "fractal_dimension": np.float32,
    "lacunarity": np.float32,
}


def _create_hdf5_datasets(h5file: h5py.File, chunk_size: int = 4096) -> None:
    """
    Create empty, resizable datasets for every field in `FIELD_DTYPES`,
    plus a small parallel "subject row index" (`index_subject_id`,
    `index_start_row`, `index_end_row`): one row per *subject*, recording
    the contiguous `[start_row, end_row)` range of that subject's rows
    in the main feature datasets above.

    This index exists so that training-time code (`streamline_model.StreamlineDataset`)
    can range-load exactly one subject's feature rows directly (a single
    contiguous HDF5 slice) instead of requiring a strict sequential
    scan matched against the exact global write order -- which is what
    makes it possible for the training dataset to freely shuffle
    subject order and interleave tract files per subject, rather than
    being locked into this module's fixed write order.
    """
    for name, dtype in FIELD_DTYPES.items():
        h5file.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            dtype=dtype,
            chunks=(chunk_size,),
        )
    h5file.create_dataset("index_subject_id", shape=(0,), maxshape=(None,), dtype=_STRING_DTYPE, chunks=(64,))
    h5file.create_dataset("index_start_row", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(64,))
    h5file.create_dataset("index_end_row", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(64,))


def _append_buffer(h5file: h5py.File, buffer: Dict[str, list]) -> None:
    """Append every row currently in `buffer` to the HDF5 datasets and grow them."""
    n_new = len(buffer["streamline_name"])
    if n_new == 0:
        return
    for name in FIELD_DTYPES:
        dataset = h5file[name]
        old_size = dataset.shape[0]
        dataset.resize((old_size + n_new,))
        dataset[old_size: old_size + n_new] = buffer[name]


def _append_subject_index_entry(h5file: h5py.File, subject_id: str, start_row: int, end_row: int) -> None:
    """Append one `(subject_id, start_row, end_row)` entry to the subject row index."""
    if end_row <= start_row:
        return  # subject contributed 0 rows (e.g. every streamline was invalid); nothing to index
    for name, value in (
        ("index_subject_id", subject_id),
        ("index_start_row", start_row),
        ("index_end_row", end_row),
    ):
        dataset = h5file[name]
        old_size = dataset.shape[0]
        dataset.resize((old_size + 1,))
        dataset[old_size] = value


def load_subject_row_index(features_h5_path: str) -> Dict[str, Tuple[int, int]]:
    """
    Load the subject row index written by `_append_subject_index_entry`
    into a ``{subject_id: (start_row, end_row)}`` mapping, for
    range-loading one subject's feature rows directly.

    Returns an empty dict (with a warning) for a `features.h5` produced
    before this index existed -- in that case, training-time code falls
    back to the older strict-sequential-scan matching strategy.
    """
    with h5py.File(features_h5_path, "r") as h5file:
        if "index_subject_id" not in h5file:
            logger.warning(
                "%s has no subject row index (produced by an older version of "
                "this module); range-based per-subject loading isn't available for it.",
                features_h5_path,
            )
            return {}
        subject_ids = h5file["index_subject_id"][:]
        start_rows = h5file["index_start_row"][:]
        end_rows = h5file["index_end_row"][:]

    index: Dict[str, Tuple[int, int]] = {}
    for raw_id, start, end in zip(subject_ids, start_rows, end_rows):
        subject_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
        index[subject_id] = (int(start), int(end))
    return index


def _get_resume_count(output_h5_path: str) -> int:
    """
    Number of streamlines already recorded in `output_h5_path`, i.e.
    how many streamlines (in the module's deterministic global order)
    to skip on this run. Returns 0 if the file doesn't exist yet or is
    a freshly created, empty file.
    """
    if not os.path.exists(output_h5_path):
        return 0
    try:
        with h5py.File(output_h5_path, "r") as h5file:
            if "streamline_name" not in h5file:
                return 0
            return int(h5file["streamline_name"].shape[0])
    except OSError as exc:
        logger.warning(
            "Could not read existing output file %s (%s); starting a new file.",
            output_h5_path,
            exc,
        )
        return 0


# =============================================================================
# TRK helpers -- each file is read from disk exactly once (no lazy,
# per-streamline re-reads); see module docstring.
# =============================================================================
def _count_streamlines_fast(trk_path: str) -> Optional[int]:
    """
    Best-effort, cheap streamline count for a .trk file using its
    header, without reading the streamline data. Returns None if the
    header doesn't carry a trustworthy count.
    """
    try:
        header = nib.streamlines.load(trk_path, lazy_load=True).header
    except Exception:
        return None

    count = header.get("nb_streamlines") if hasattr(header, "get") else None
    # A legitimately empty file (0 streamlines) is just as trustworthy a
    # count as any positive one -- only a missing/non-integer count is
    # actually unreliable.
    if isinstance(count, (int, np.integer)) and count >= 0:
        return int(count)
    return None


def _load_trk_streamlines_full(trk_path: str) -> List[np.ndarray]:
    """
    Load every streamline of a .trk file into memory with a single,
    eager (non-lazy) read. Used by the subject/file caching strategy
    in `extract_features_from_directory` so that a file is opened and
    read from disk exactly once, regardless of how many streamlines it
    contains.

    Returns
    -------
    list of np.ndarray, each shape (N_i, 3), float64
    """
    tractogram = nib.streamlines.load(trk_path)  # eager load: one disk read
    streamlines = [np.asarray(s, dtype=np.float64) for s in tractogram.streamlines]
    del tractogram
    return streamlines


def _load_subject_cache(files: List[str]) -> Dict[str, List[np.ndarray]]:
    """
    Load every .trk file belonging to one subject into memory at once
    (subject-level cache): ``{file_path: [streamline, ...]}``. Each
    file is still read from disk exactly once. A file that fails to
    load is logged and mapped to an empty list so the rest of the
    subject's files are unaffected.
    """
    cache: Dict[str, List[np.ndarray]] = {}
    for trk_path in files:
        try:
            cache[trk_path] = _load_trk_streamlines_full(trk_path)
        except Exception as exc:
            logger.warning("Skipping corrupted/unreadable file %s: %s", trk_path, exc)
            cache[trk_path] = []
    return cache


# =============================================================================
# Offline, one-time subject bounds computation (avoids ever rescanning
# .trk files for normalization bounds again -- see module docstring).
# =============================================================================
def compute_and_save_bounds_metadata(
    input_dir: str,
    bounds_h5_path: str,
    separator: str = "__",
    overwrite_existing: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    One-time offline step: group `input_dir`'s .trk files by subject
    (reusing `xyz2RGB.group_trk_files_by_subject`), compute each
    subject's coordinate bounds (reusing `xyz2RGB.compute_subject_bounds`,
    which reads that subject's files exactly once), and persist the
    result to `bounds_h5_path` via `xyz2RGB.save_bounds_metadata_h5`.

    After this call, `extract_features_from_directory` (via its
    `bounds_h5_path` argument) and any `xyz2RGB`-based image generator
    can load these bounds with a single small HDF5 read and never need
    to open a .trk file again just to recover normalization bounds.

    Parameters
    ----------
    input_dir : str
        Directory containing the dataset's .trk files.
    bounds_h5_path : str
        Output path for the bounds metadata HDF5 file.
    separator : str
        Subject/tract separator in filenames (default "__").
    overwrite_existing : bool
        If `bounds_h5_path` already exists and this is False (default),
        only subjects *not already present* in it are (re)computed, and
        the merged result (old + newly computed subjects) is written
        back out -- supporting incremental updates as new subjects are
        added to the dataset without recomputing everyone else's
        bounds. If True, every subject is recomputed from scratch.

    Returns
    -------
    dict[str, tuple(mins, maxs)]
        The full set of subject bounds that was written to disk.
    """
    subject_groups = group_trk_files_by_subject(input_dir, separator=separator)

    existing_bounds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if not overwrite_existing and os.path.isfile(bounds_h5_path):
        existing_bounds = load_bounds_metadata_h5(bounds_h5_path)
        logger.info("Found %d subjects already in %s; only computing new ones.", len(existing_bounds), bounds_h5_path)

    bounds_by_subject: Dict[str, Tuple[np.ndarray, np.ndarray]] = dict(existing_bounds)

    subjects_to_compute = [sid for sid in sorted(subject_groups) if sid not in existing_bounds]
    for subject_id in tqdm(subjects_to_compute, desc="Computing subject bounds", unit="subject"):
        try:
            bounds_by_subject[subject_id] = compute_subject_bounds(subject_groups[subject_id])
        except Exception as exc:
            logger.warning("Could not compute bounds for subject '%s': %s", subject_id, exc)

    save_bounds_metadata_h5(bounds_by_subject, bounds_h5_path)
    logger.info("Saved bounds metadata for %d subjects to %s.", len(bounds_by_subject), bounds_h5_path)
    return bounds_by_subject


# =============================================================================
# Per-streamline processing
# =============================================================================
def _process_single_streamline(
    points: np.ndarray,
    streamline_name: str,
    grid_size: int = DEFAULT_GRID_SIZE,
    scales: np.ndarray = DEFAULT_SCALES,
) -> Dict[str, object]:
    """
    Run the full feature-extraction pipeline for one streamline:
    geometric features from the raw coordinates, then a temporary 3D
    voxel grid (built by `rasterize_streamline` from this streamline's
    own coordinates) for the morphological features. The grid is never
    returned or persisted -- it goes out of scope as soon as this
    function returns.

    Raises
    ------
    ValueError
        If the streamline is invalid (too few points, non-finite
        coordinates, or zero arc length) -- callers should catch this
        and skip the streamline with a warning.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected an (N, 3) array, got shape {points.shape}")
    if points.shape[0] < 2:
        raise ValueError(f"too few points ({points.shape[0]})")
    if not np.all(np.isfinite(points)):
        raise ValueError("streamline contains non-finite coordinates")

    length = calculate_polyline_length(points)
    if length <= 1e-9:
        raise ValueError("zero-length streamline")

    curvature = compute_curvature(points)
    tortuosity = compute_tortuosity(points, length=length)
    spectral_entropy = compute_spectral_entropy(points)

    # Morphological features: rasterize into a temporary voxel grid, use it, discard it.
    grid = rasterize_streamline(points, grid_size=grid_size)
    fractal_dimension = compute_fractal_dimension(grid, scales=scales)
    lacunarity_by_scale = compute_lacunarity(grid, scales=scales)
    lacunarity = _reduce_lacunarity(lacunarity_by_scale)
    del grid, lacunarity_by_scale  # explicit discard; nothing from the grid survives this function

    return {
        "streamline_name": streamline_name,
        "spectral_entropy": np.float32(spectral_entropy),
        "tortuosity": np.float32(tortuosity if np.isfinite(tortuosity) else np.finfo(np.float32).max),
        "curvature": np.float32(curvature),
        "length": np.float32(length),
        "fractal_dimension": np.float32(fractal_dimension),
        "lacunarity": np.float32(lacunarity),
    }


# =============================================================================
# Subject-level multiprocessing worker
# =============================================================================
@dataclass
class _SubjectTask:
    """
    One unit of work for a `multiprocessing.Pool` worker: everything
    needed to fully process a single subject's .trk files, with no
    dependency on any state living in the main process (so it's cheaply
    picklable and self-contained).
    """

    subject_id: str
    files: List[str]
    local_skip_count: int  # how many of this subject's leading streamlines to skip (resume)
    grid_size: int
    scales: np.ndarray
    index_width: int
    cache_scope: str


@dataclass
class _SubjectResult:
    """Everything the main process needs to write back one subject's results."""

    subject_id: str
    rows: Dict[str, list] = field(default_factory=dict)
    n_streamlines_seen: int = 0  # total raw streamlines walked (skipped + valid + invalid)
    n_skipped_invalid: int = 0
    error: Optional[str] = None


def _process_subject_task(task: _SubjectTask) -> _SubjectResult:
    """
    Worker-process entry point: fully process one subject's .trk files
    -- load them (once each, per `cache_scope`), compute every valid
    streamline's feature row -- and return the results. No HDF5 writing
    happens here; the main process is the sole writer, which is what
    keeps the output file's row order (and therefore resumability)
    identical to a single-process run regardless of which worker
    finishes first (see `extract_features_from_directory`, which
    consumes `Pool.imap` results in submission order).

    Never raises: any failure is caught and reported via
    `_SubjectResult.error` so one bad subject can't crash the whole
    pool of workers.
    """
    buffer: Dict[str, list] = {name: [] for name in FIELD_DTYPES}
    n_skipped_invalid = 0
    n_seen = 0

    try:
        if task.cache_scope == "subject":
            subject_cache = _load_subject_cache(task.files)

        for trk_path in task.files:
            file_stem = os.path.splitext(os.path.basename(trk_path))[0]

            if task.cache_scope == "subject":
                streamlines = subject_cache[trk_path]
            else:
                try:
                    streamlines = _load_trk_streamlines_full(trk_path)
                except Exception as exc:
                    logger.warning("Skipping corrupted/unreadable file %s: %s", trk_path, exc)
                    continue

            for local_index, points in enumerate(streamlines):
                if n_seen < task.local_skip_count:
                    # Already written to the output file in a previous
                    # (interrupted) run; skip cheaply, same as the
                    # single-process resume logic.
                    n_seen += 1
                    continue

                streamline_name = f"{file_stem}_sl_{local_index:0{task.index_width}d}"
                try:
                    row = _process_single_streamline(
                        points, streamline_name, grid_size=task.grid_size, scales=task.scales
                    )
                except ValueError as exc:
                    logger.warning("Skipping invalid streamline '%s': %s", streamline_name, exc)
                    n_skipped_invalid += 1
                    n_seen += 1
                    continue

                for key, value in row.items():
                    buffer[key].append(value)
                n_seen += 1

            if task.cache_scope == "file":
                del streamlines

        if task.cache_scope == "subject":
            del subject_cache

        return _SubjectResult(
            subject_id=task.subject_id,
            rows=buffer,
            n_streamlines_seen=n_seen,
            n_skipped_invalid=n_skipped_invalid,
        )
    except Exception as exc:  # pragma: no cover -- defensive: never let one subject crash the pool
        return _SubjectResult(subject_id=task.subject_id, rows={name: [] for name in FIELD_DTYPES}, error=str(exc))


# =============================================================================
# Main streaming pipeline
# =============================================================================
def extract_features_from_directory(
    input_dir: str,
    output_h5_path: str,
    separator: str = "__",
    flush_every: int = 2000,
    index_width: int = 7,
    grid_size: int = DEFAULT_GRID_SIZE,
    scales: np.ndarray = DEFAULT_SCALES,
    estimate_total_for_progress: bool = True,
    bounds_h5_path: Optional[str] = None,
    cache_scope: str = "subject",
    n_workers: Optional[int] = None,
    use_multiprocessing: bool = True,
) -> None:
    """
    Process every streamline of every .trk file in `input_dir`,
    grouped subject-wise exactly as in `xyz2RGB.py`, and append one
    feature row per streamline to a single HDF5 file.

    I/O profile: each .trk file is read from disk exactly once. By
    default (`cache_scope="subject"`) every file belonging to a
    subject is loaded into memory together, that subject's streamlines
    are processed sequentially, and the cache is released before the
    next subject -- at most one subject's streamlines are ever
    resident in memory (per worker process; see below), so the dataset
    as a whole is still streamed. Use `cache_scope="file"` to cache
    only one file at a time for a lower memory ceiling.

    Parallelism: whole subjects (not individual streamlines) are
    distributed across a `multiprocessing.Pool` -- see the "Subject-
    level multiprocessing" section of the module docstring for the
    full design. The output file's row order (and therefore
    resumability) is identical regardless of `n_workers`.

    Automatically resumes: if `output_h5_path` already contains N
    rows, the first N streamlines (in the deterministic subject/file/
    streamline order below) are skipped without recomputation, and
    processing continues from streamline N+1. No row is ever written
    twice.

    Parameters
    ----------
    input_dir : str
        Directory containing the dataset's .trk files (multiple
        subjects, each with one or more tract files).
    output_h5_path : str
        Path to the (possibly pre-existing, for resume) output HDF5
        file. Only one file is produced for the whole dataset.
    separator : str
        Subject/tract separator in filenames (default "__"), passed
        through to the reused `xyz2RGB` grouping function.
    flush_every : int
        Minimum number of feature rows to buffer in memory before
        flushing to the HDF5 file (default 2000). Since a whole
        subject's rows arrive from a worker at once, the buffer may
        briefly exceed this by up to one subject's worth of rows
        before the next flush check -- larger values trade a little
        more RAM for fewer, larger disk writes either way.
    index_width : int
        Zero-padding width for the per-file streamline index in
        `streamline_name` (default 7), matching `xyz2RGB.py`.
    grid_size : int
        Side length of the temporary 3D voxel grid used by
        `rasterize_streamline` for the morphological features
        (default 64).
    scales : np.ndarray
        Box-counting / gliding-box scales used by
        `compute_fractal_dimension` and `compute_lacunarity`
        (default ``[1, 2, 4, 8, 16]``).
    estimate_total_for_progress : bool
        If True (default), attempt to read each file's streamline
        count from its header (cheap) to show an accurate progress bar
        total, and to let already-fully-processed subjects be skipped
        without ever being loaded (both for resume and for the
        multiprocessing dispatch below). If any file lacks a
        trustworthy header count, the progress bar falls back to an
        indeterminate/growing display and that subject (and all
        subsequent ones) lose the skip-without-loading optimization
        (they are still processed correctly, just without the
        shortcut). Set to False to skip this entirely for maximum
        startup speed on huge datasets.
    bounds_h5_path : str, optional
        Path to a subject-bounds metadata HDF5 file produced by
        `compute_and_save_bounds_metadata` (or `xyz2RGB.save_bounds_metadata_h5`
        directly). If given, it is loaded exactly once at startup (no
        .trk file is ever opened to compute bounds) and logged for
        traceability; this keeps the run's normalization provenance
        consistent with whatever `xyz2RGB`-based pipeline consumes the
        same file downstream (e.g. a training-time image generator).
        Not required for the geometric/morphological features
        themselves, which do not depend on subject-wide bounds (see
        module docstring).
    cache_scope : {"subject", "file"}
        Granularity at which .trk files are cached in memory by each
        worker while being processed (default "subject", i.e. a whole
        subject's files at once; see I/O profile above).
    n_workers : int, optional
        Number of worker processes (default: `os.cpu_count()`).
        Ignored if `use_multiprocessing=False`.
    use_multiprocessing : bool
        If True (default), dispatch subjects across a
        `multiprocessing.Pool`. If False, process subjects one at a
        time in the calling process (identical results, useful for
        debugging or single-core environments).
    """
    subject_groups = group_trk_files_by_subject(input_dir, separator=separator)
    subject_ids_sorted = sorted(subject_groups.keys())

    if cache_scope not in ("subject", "file"):
        raise ValueError(f"cache_scope must be 'subject' or 'file', got {cache_scope!r}")

    # Load precomputed normalization bounds exactly once (no .trk rescanning).
    subject_bounds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if bounds_h5_path is not None:
        subject_bounds = load_bounds_metadata_h5(bounds_h5_path)
        logger.info(
            "Loaded precomputed normalization bounds for %d subjects from %s (no .trk files rescanned).",
            len(subject_bounds),
            bounds_h5_path,
        )
        missing = [sid for sid in subject_ids_sorted if sid not in subject_bounds]
        if missing:
            logger.warning(
                "%d subject(s) in %s have no entry in %s (e.g. %s); downstream "
                "image generation for them will need bounds computed separately.",
                len(missing),
                input_dir,
                bounds_h5_path,
                missing[:3],
            )

    resume_count = _get_resume_count(output_h5_path)
    if resume_count > 0:
        logger.info("Resuming: %d streamlines already recorded in %s.", resume_count, output_h5_path)
    else:
        logger.info("Starting a fresh run; no prior output found at %s.", output_h5_path)

    file_mode = "a" if os.path.exists(output_h5_path) else "w"

    # Best-effort total for the progress bar. Never required for correctness.
    total_streamlines: Optional[int] = 0 if estimate_total_for_progress else None
    per_file_counts: Dict[str, Optional[int]] = {}
    if estimate_total_for_progress:
        for subject_id in subject_ids_sorted:
            for trk_path in sorted(subject_groups[subject_id]):
                count = _count_streamlines_fast(trk_path)
                per_file_counts[trk_path] = count
                if count is None or total_streamlines is None:
                    total_streamlines = None
                else:
                    total_streamlines += count

    buffer: Dict[str, list] = {name: [] for name in FIELD_DTYPES}
    global_index = 0
    n_skipped_invalid = 0

    # ------------------------------------------------------------------
    # Build the list of subject-level tasks to dispatch, resolving
    # resume correctly first:
    #
    # - A subject whose *header* count is trustworthy and shows it lies
    #   entirely before the resume point is skipped here without ever
    #   being loaded (same as the single-process version).
    # - Any subject that might straddle the resume boundary -- because
    #   part of it genuinely needs skipping, or because its header
    #   count isn't trustworthy enough to know either way -- is
    #   resolved *synchronously*, right here in the main process, using
    #   `_process_subject_task`'s own exact per-streamline skip walk.
    #   This keeps `global_index` exact (driven by the real processed
    #   count, never a header estimate) and self-corrects: if that
    #   subject's real count turns out smaller than the skip still
    #   owed, the loop simply re-enters this branch for the next
    #   subject too, cascading correctly however many subjects it takes
    #   (in practice, essentially always at most one). This costs a
    #   small, bounded amount of lost parallelism -- never correctness.
    # - Once `global_index >= resume_count`, every remaining subject is
    #   guaranteed fully past the resume point, so it's dispatched with
    #   `local_skip_count=0` and is safe to fully parallelize.
    # ------------------------------------------------------------------
    tasks: List[_SubjectTask] = []
    boundary_results: List[_SubjectResult] = []
    for subject_id in subject_ids_sorted:
        files = sorted(subject_groups[subject_id])

        subject_total = None
        if estimate_total_for_progress:
            counts = [per_file_counts.get(f) for f in files]
            if all(c is not None for c in counts):
                subject_total = sum(counts)

        if subject_total is not None and global_index + subject_total <= resume_count:
            global_index += subject_total
            continue  # entirely already processed; never loaded at all

        if resume_count > global_index:
            # May straddle the resume boundary (or its count is
            # unreliable) -- resolve exactly, synchronously.
            local_skip = max(0, resume_count - global_index)
            boundary_task = _SubjectTask(
                subject_id=subject_id,
                files=files,
                local_skip_count=local_skip,
                grid_size=grid_size,
                scales=scales,
                index_width=index_width,
                cache_scope=cache_scope,
            )
            boundary_result = _process_subject_task(boundary_task)
            global_index += boundary_result.n_streamlines_seen
            boundary_results.append(boundary_result)
            continue

        # Guaranteed fully past the resume point -- safe to parallelize.
        tasks.append(
            _SubjectTask(
                subject_id=subject_id,
                files=files,
                local_skip_count=0,
                grid_size=grid_size,
                scales=scales,
                index_width=index_width,
                cache_scope=cache_scope,
            )
        )

    n_workers_actual = max(1, n_workers or os.cpu_count() or 1)
    n_workers_actual = min(n_workers_actual, len(tasks)) if tasks else 1
    logger.info(
        "Dispatching %d subject(s) across %d worker process(es) (use_multiprocessing=%s).",
        len(tasks),
        n_workers_actual,
        use_multiprocessing,
    )

    with h5py.File(output_h5_path, file_mode) as h5file:
        if file_mode == "w":
            _create_hdf5_datasets(h5file)
        elif "index_subject_id" not in h5file:
            # Resuming a features.h5 written before the subject row
            # index existed -- add the (empty-so-far) index datasets so
            # this run can start populating it; older rows written
            # before this point simply have no index entry.
            logger.warning(
                "%s predates the subject row index; adding empty index datasets "
                "(only subjects processed from now on will be indexed).",
                output_h5_path,
            )
            h5file.create_dataset("index_subject_id", shape=(0,), maxshape=(None,), dtype=_STRING_DTYPE, chunks=(64,))
            h5file.create_dataset("index_start_row", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(64,))
            h5file.create_dataset("index_end_row", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(64,))

        start_time = time.time()
        # Running count of rows physically written to the HDF5 file so
        # far (starts at `resume_count`, since that's exactly how many
        # rows already exist); used to compute each subject's exact
        # [start_row, end_row) range for the subject row index at flush
        # time, since a subject's rows may sit in the in-memory buffer
        # for a while (spanning several subjects) before actually being
        # written.
        rows_written_so_far = resume_count
        pending_subject_boundaries: List[Tuple[str, int]] = []  # (subject_id, n_rows_from_this_result)

        def _flush_buffer() -> None:
            nonlocal rows_written_so_far
            if not buffer["streamline_name"]:
                pending_subject_boundaries.clear()
                return
            _append_buffer(h5file, buffer)
            offset = rows_written_so_far
            for subject_id, n_rows in pending_subject_boundaries:
                _append_subject_index_entry(h5file, subject_id, offset, offset + n_rows)
                offset += n_rows
            rows_written_so_far = offset
            h5file.flush()
            for key in buffer:
                buffer[key].clear()
            pending_subject_boundaries.clear()

        with tqdm(
            total=total_streamlines,
            initial=min(resume_count, total_streamlines) if total_streamlines else resume_count,
            unit="sl",
            desc="Streamlines",
            dynamic_ncols=True,
        ) as pbar:

            def _write_result(result: _SubjectResult) -> None:
                # Writes a subject's rows to the output buffer (flushing
                # if needed) and updates progress/logging -- but does
                # NOT touch `global_index`, since the synchronous
                # boundary results below already advanced it during
                # task-building (to make correct skip decisions for
                # subsequent subjects); pool/serial results advance it
                # separately, in `_handle_dispatched_result`.
                nonlocal n_skipped_invalid
                if result.error is not None:
                    logger.warning("Subject '%s' failed entirely: %s", result.subject_id, result.error)
                    return

                n_rows_this_result = len(result.rows.get("streamline_name", []))
                for key in buffer:
                    buffer[key].extend(result.rows.get(key, []))
                if n_rows_this_result > 0:
                    pending_subject_boundaries.append((result.subject_id, n_rows_this_result))
                n_skipped_invalid += result.n_skipped_invalid
                pbar.set_postfix(subject=result.subject_id, refresh=False)
                pbar.update(result.n_streamlines_seen)

                if len(buffer["streamline_name"]) >= flush_every:
                    _flush_buffer()

            def _handle_dispatched_result(result: _SubjectResult) -> None:
                nonlocal global_index
                _write_result(result)
                if result.error is None:
                    global_index += result.n_streamlines_seen

            # Write out the (at most a few) subjects resolved
            # synchronously above -- global_index already accounts for
            # these, so only `_write_result` (not `_handle_dispatched_result`)
            # is used here.
            for boundary_result in boundary_results:
                _write_result(boundary_result)

            if use_multiprocessing and n_workers_actual > 1 and tasks:
                # `imap` (not `imap_unordered`) yields results in the same
                # order tasks were submitted -- i.e. sorted-subject order --
                # even though workers may finish in a different order, so
                # the output file's row order stays deterministic.
                with mp.Pool(processes=n_workers_actual) as pool:
                    for result in pool.imap(_process_subject_task, tasks, chunksize=1):
                        _handle_dispatched_result(result)
            else:
                for task in tasks:
                    _handle_dispatched_result(_process_subject_task(task))

            # Flush any remaining buffered rows.
            _flush_buffer()

        elapsed = time.time() - start_time
        logger.info(
            "Done. Processed %d streamlines (%d skipped as invalid) in %.1fs.",
            global_index - resume_count,
            n_skipped_invalid,
            elapsed,
        )


# =============================================================================
# CLI entry point
# =============================================================================
def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract geometric and morphological streamline features from a directory of .trk files."
    )
    parser.add_argument("input_dir", help="Directory containing the dataset's .trk files.")
    parser.add_argument("output_h5_path", help="Path to the output HDF5 file (created or resumed).")
    parser.add_argument("--separator", default="__", help="Subject/tract filename separator (default '__').")
    parser.add_argument("--flush-every", type=int, default=2000, help="Rows buffered before an HDF5 flush.")
    parser.add_argument(
        "--bounds-h5-path",
        default=None,
        help=(
            "Path to a precomputed subject-bounds metadata HDF5 file "
            "(see compute_and_save_bounds_metadata). If omitted, bounds "
            "are simply not loaded (they are not required by the "
            "geometric/morphological features themselves)."
        ),
    )
    parser.add_argument(
        "--cache-scope",
        choices=["subject", "file"],
        default="subject",
        help="Cache a whole subject's .trk files at once (default) or one file at a time.",
    )
    parser.add_argument(
        "--no-progress-total",
        action="store_true",
        help="Skip pre-scanning file headers for an exact progress-bar total (faster startup).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of worker processes for subject-level parallelism (default: os.cpu_count()).",
    )
    parser.add_argument(
        "--no-multiprocessing",
        action="store_true",
        help="Disable multiprocessing entirely; process subjects one at a time in this process.",
    )
    args = parser.parse_args()

    extract_features_from_directory(
        input_dir=args.input_dir,
        output_h5_path=args.output_h5_path,
        separator=args.separator,
        flush_every=args.flush_every,
        estimate_total_for_progress=not args.no_progress_total,
        bounds_h5_path=args.bounds_h5_path,
        cache_scope=args.cache_scope,
        n_workers=args.n_workers,
        use_multiprocessing=not args.no_multiprocessing,
    )


if __name__ == "__main__":
    _main()


__all__ = [
    "calculate_polyline_length",
    "compute_curvature",
    "compute_tortuosity",
    "compute_spectral_entropy",
    "rasterize_streamline",
    "compute_fractal_dimension",
    "compute_lacunarity",
    "compute_and_save_bounds_metadata",
    "extract_features_from_directory",
    "load_subject_row_index",
    "FIELD_DTYPES",
]
