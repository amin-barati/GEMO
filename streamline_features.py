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

import logging
import os
import time
from typing import Dict, Iterator, List, Optional, Tuple

import h5py
import numpy as np
from scipy import fft
from skimage.measure import block_reduce
from tqdm import tqdm

import nibabel as nib

from xyz2RGB import group_trk_files_by_subject

logger = logging.getLogger("streamline_features")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

DEFAULT_SCALES = np.array([1, 2, 4, 8, 16])
DEFAULT_GRID_SIZE = 64




# Features

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

    # Remove DC component 
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

    entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-10))  # Avoid log(0)
    return float(entropy)


# Morphological feature 

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
   
    if not lacunarity_by_scale:
        return 0.0
    return float(np.mean(list(lacunarity_by_scale.values())))


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


def _append_buffer(h5file: h5py.File, buffer: Dict[str, list]) -> None:
    n_new = len(buffer["streamline_name"])
    if n_new == 0:
        return
    for name in FIELD_DTYPES:
        dataset = h5file[name]
        old_size = dataset.shape[0]
        dataset.resize((old_size + n_new,))
        dataset[old_size: old_size + n_new] = buffer[name]


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


# TRK 
def _count_streamlines_fast(trk_path: str) -> Optional[int]:
  
    try:
        header = nib.streamlines.load(trk_path, lazy_load=True).header
    except Exception:
        return None

    count = header.get("nb_streamlines") if hasattr(header, "get") else None
    if isinstance(count, (int, np.integer)) and count > 0:
        return int(count)
    return None


def _iter_streamlines(trk_path: str) -> Iterator[np.ndarray]:
    tractogram = nib.streamlines.load(trk_path, lazy_load=True)
    for streamline in tractogram.streamlines:
        yield np.asarray(streamline, dtype=np.float64)
    del tractogram


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



def extract_features_from_directory(
    input_dir: str,
    output_h5_path: str,
    separator: str = "__",
    flush_every: int = 2000,
    index_width: int = 7,
    grid_size: int = DEFAULT_GRID_SIZE,
    scales: np.ndarray = DEFAULT_SCALES,
    estimate_total_for_progress: bool = True,
) -> None:

    subject_groups = group_trk_files_by_subject(input_dir, separator=separator)
    subject_ids_sorted = sorted(subject_groups.keys())

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

    with h5py.File(output_h5_path, file_mode) as h5file:
        if file_mode == "w":
            _create_hdf5_datasets(h5file)

        start_time = time.time()
        with tqdm(
            total=total_streamlines,
            initial=min(resume_count, total_streamlines) if total_streamlines else resume_count,
            unit="sl",
            desc="Streamlines",
            dynamic_ncols=True,
        ) as pbar:
            for subject_id in subject_ids_sorted:
                files = sorted(subject_groups[subject_id])

                
                subject_total = None
                if estimate_total_for_progress:
                    counts = [per_file_counts.get(f) for f in files]
                    if all(c is not None for c in counts):
                        subject_total = sum(counts)
                        if global_index + subject_total <= resume_count:
                            global_index += subject_total
                            continue

                for trk_path in files:
                    file_stem = os.path.splitext(os.path.basename(trk_path))[0]

                    try:
                        streamline_iter = _iter_streamlines(trk_path)
                    except Exception as exc:
                        logger.warning("Skipping corrupted/unreadable file %s: %s", trk_path, exc)
                        continue

                    local_index = 0
                    while True:
                        try:
                            points = next(streamline_iter)
                        except StopIteration:
                            break
                        except Exception as exc:
                            logger.warning(
                                "Stopping early on corrupted data in %s (streamline ~%d): %s",
                                trk_path,
                                local_index,
                                exc,
                            )
                            break

                        if global_index < resume_count:
                            global_index += 1
                            local_index += 1
                            continue

                        streamline_name = f"{file_stem}_sl_{local_index:0{index_width}d}"
                        pbar.set_postfix(subject=subject_id, file=file_stem, refresh=False)

                        try:
                            row = _process_single_streamline(
                                points, streamline_name, grid_size=grid_size, scales=scales
                            )
                        except ValueError as exc:
                            logger.warning("Skipping invalid streamline '%s': %s", streamline_name, exc)
                            n_skipped_invalid += 1
                            global_index += 1
                            local_index += 1
                            pbar.update(1)
                            continue

                        for key, value in row.items():
                            buffer[key].append(value)

                        if len(buffer["streamline_name"]) >= flush_every:
                            _append_buffer(h5file, buffer)
                            h5file.flush()
                            for key in buffer:
                                buffer[key].clear()

                        del points, row
                        global_index += 1
                        local_index += 1
                        pbar.update(1)

                    del streamline_iter

            if buffer["streamline_name"]:
                _append_buffer(h5file, buffer)
                h5file.flush()

        elapsed = time.time() - start_time
        logger.info(
            "Done. Processed %d streamlines (%d skipped as invalid) in %.1fs.",
            global_index - resume_count,
            n_skipped_invalid,
            elapsed,
        )


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
        "--no-progress-total",
        action="store_true",
        help="Skip pre-scanning file headers for an exact progress-bar total (faster startup).",
    )
    args = parser.parse_args()

    extract_features_from_directory(
        input_dir=args.input_dir,
        output_h5_path=args.output_h5_path,
        separator=args.separator,
        flush_every=args.flush_every,
        estimate_total_for_progress=not args.no_progress_total,
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
    "extract_features_from_directory",
    "FIELD_DTYPES",
]
