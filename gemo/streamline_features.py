"""

==============================================================================
GEMO: A Deep Learning Method for Brain Fiber Classification and Tract
Segmentation Using Geometrical and Morphological Features

DOI:        https://doi.org/10.1016/j.acra.2026.07.024

Email       : Amin_br@yahoo.com
GitHub      : https://github.com/amin-barati/GEMO

==============================================================================


streamline_features
====================

Streaming, memory-efficient extraction of geometric and morphological
features from large collections of tractography (.trk) files.


Typical usage
-------------
>>> from xyz2RGB import group_trk_files_by_subject, compute_subject_bounds
>>> from streamline_features import compute_and_save_bounds_metadata, extract_features_from_directory

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


# Geometric feature algorithms 
def calculate_polyline_length(points: np.ndarray) -> float:
    
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return float(np.sum(seg_lengths))


def compute_curvature(points: np.ndarray) -> float:
    
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
    
    if points.shape[0] < 2:
        return 0.0

    chord = float(np.linalg.norm(points[-1] - points[0]))
    if length is None:
        length = calculate_polyline_length(points)

    if chord < 1e-9:
        return float("inf") if length > 1e-9 else 0.0

    return float(length / chord)


def compute_spectral_entropy(points: np.ndarray) -> float:
    
    x = points[:, 0] - np.mean(points[:, 0])
    y = points[:, 1] - np.mean(points[:, 1])
    z = points[:, 2] - np.mean(points[:, 2])

    fft_x = np.abs(fft.fft(x))
    fft_y = np.abs(fft.fft(y))
    fft_z = np.abs(fft.fft(z))
    psd = fft_x**2 + fft_y**2 + fft_z**2

    psd = psd[:len(psd)//2]  
    psd_sum = np.sum(psd)
    if psd_sum <= 1e-12:
        
        return 0.0
    psd_normalized = psd / psd_sum

    # Compute Shannon entropy
    entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-10))  # Avoid log(0)
    return float(entropy)


# Morphological feature algorithms 

def rasterize_streamline(points: np.ndarray, grid_size: int = DEFAULT_GRID_SIZE) -> np.ndarray:
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    range_coords = max_coords - min_coords
    range_coords[range_coords == 0] = 1e-10  

    normalized = (points - min_coords) / range_coords
    grid_coords = (normalized * (grid_size - 1)).astype(int)

    grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    for coord in grid_coords:
        grid[tuple(coord)] = True

    return grid


def compute_fractal_dimension(grid: np.ndarray, scales: np.ndarray = DEFAULT_SCALES) -> float:

    counts = []
    for s in scales:
        if s > min(grid.shape):
            continue
        pooled = block_reduce(grid, block_size=(s, s, s), func=np.max)
        counts.append(np.sum(pooled))

    valid = np.array(counts) > 0
    if np.sum(valid) < 2:
        return 0.0

    log_scales = np.log(1 / scales[valid])
    log_counts = np.log(np.array(counts)[valid])
    return float(np.polyfit(log_scales, log_counts, 1)[0])


def compute_lacunarity(grid: np.ndarray, scales: np.ndarray = DEFAULT_SCALES) -> Dict[int, float]:
    lacunarities = {}
    for s in scales:
        if s > min(grid.shape):
            continue

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

    if not lacunarity_by_scale:
        return 0.0
    return float(np.mean(list(lacunarity_by_scale.values())))


# HDF5 schema
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
    n_new = len(buffer["streamline_name"])
    if n_new == 0:
        return
    for name in FIELD_DTYPES:
        dataset = h5file[name]
        old_size = dataset.shape[0]
        dataset.resize((old_size + n_new,))
        dataset[old_size: old_size + n_new] = buffer[name]


def _append_subject_index_entry(h5file: h5py.File, subject_id: str, start_row: int, end_row: int) -> None:

    
    if end_row <= start_row:
        return  
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


# TRK help


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

    if isinstance(count, (int, np.integer)) and count >= 0:
        return int(count)
    return None


def _load_trk_streamlines_full(trk_path: str) -> List[np.ndarray]:

    tractogram = nib.streamlines.load(trk_path)  # eager load: one disk read
    streamlines = [np.asarray(s, dtype=np.float64) for s in tractogram.streamlines]
    del tractogram
    return streamlines


def _load_subject_cache(files: List[str]) -> Dict[str, List[np.ndarray]]:

    cache: Dict[str, List[np.ndarray]] = {}
    for trk_path in files:
        try:
            cache[trk_path] = _load_trk_streamlines_full(trk_path)
        except Exception as exc:
            logger.warning("Skipping corrupted/unreadable file %s: %s", trk_path, exc)
            cache[trk_path] = []
    return cache


# Offline, one-time subject bounds computation 

def compute_and_save_bounds_metadata(
    input_dir: str,
    bounds_h5_path: str,
    separator: str = "__",
    overwrite_existing: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:

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


# 
# Per-streamline processing

def _process_single_streamline(
    points: np.ndarray,
    streamline_name: str,
    grid_size: int = DEFAULT_GRID_SIZE,
    scales: np.ndarray = DEFAULT_SCALES,
) -> Dict[str, object]:

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

    grid = rasterize_streamline(points, grid_size=grid_size)
    fractal_dimension = compute_fractal_dimension(grid, scales=scales)
    lacunarity_by_scale = compute_lacunarity(grid, scales=scales)
    lacunarity = _reduce_lacunarity(lacunarity_by_scale)
    del grid, lacunarity_by_scale  

    return {
        "streamline_name": streamline_name,
        "spectral_entropy": np.float32(spectral_entropy),
        "tortuosity": np.float32(tortuosity if np.isfinite(tortuosity) else np.finfo(np.float32).max),
        "curvature": np.float32(curvature),
        "length": np.float32(length),
        "fractal_dimension": np.float32(fractal_dimension),
        "lacunarity": np.float32(lacunarity),
    }


# Subject-level multiprocessing worker

@dataclass
class _SubjectTask:

    subject_id: str
    files: List[str]
    local_skip_count: int  
    grid_size: int
    scales: np.ndarray
    index_width: int
    cache_scope: str


@dataclass
class _SubjectResult:
    """Everything the main process needs to write back one subject's results."""

    subject_id: str
    rows: Dict[str, list] = field(default_factory=dict)
    n_streamlines_seen: int = 0 
    n_skipped_invalid: int = 0
    error: Optional[str] = None


def _process_subject_task(task: _SubjectTask) -> _SubjectResult:
    
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


# 
# Main streaming pipeline

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
   
    subject_groups = group_trk_files_by_subject(input_dir, separator=separator)
    subject_ids_sorted = sorted(subject_groups.keys())

    if cache_scope not in ("subject", "file"):
        raise ValueError(f"cache_scope must be 'subject' or 'file', got {cache_scope!r}")


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
            continue  

        if resume_count > global_index:
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

            
            logger.warning(
                "%s predates the subject row index; adding empty index datasets "
                "(only subjects processed from now on will be indexed).",
                output_h5_path,
            )
            h5file.create_dataset("index_subject_id", shape=(0,), maxshape=(None,), dtype=_STRING_DTYPE, chunks=(64,))
            h5file.create_dataset("index_start_row", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(64,))
            h5file.create_dataset("index_end_row", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(64,))

        start_time = time.time()

        
        rows_written_so_far = resume_count
        pending_subject_boundaries: List[Tuple[str, int]] = [] 

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

        
            for boundary_result in boundary_results:
                _write_result(boundary_result)

            if use_multiprocessing and n_workers_actual > 1 and tasks:

                with mp.Pool(processes=n_workers_actual) as pool:
                    for result in pool.imap(_process_subject_task, tasks, chunksize=1):
                        _handle_dispatched_result(result)
            else:
                for task in tasks:
                    _handle_dispatched_result(_process_subject_task(task))

            _flush_buffer()

        elapsed = time.time() - start_time
        logger.info(
            "Done. Processed %d streamlines (%d skipped as invalid) in %.1fs.",
            global_index - resume_count,
            n_skipped_invalid,
            elapsed,
        )


# CLI entry point

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
