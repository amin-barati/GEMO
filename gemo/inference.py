"""
==============================================================================
GEMO: A Deep Learning Method for Brain Fiber Classification and Tract
Segmentation Using Geometrical and Morphological Features

DOI:        https://doi.org/10.1016/j.acra.2026.07.024

Email       : Amin_br@yahoo.com
GitHub      : https://github.com/amin-barati/GEMO

==============================================================================


inference
=========
Command-line inference entry point: classify a whole-brain .trk file into
per-tract .trk files (plus one "Unknown" file) using a trained
`StreamlineClassifier` checkpoint.

Usage
-----
    python inference.py --trk-path wholebrain.trk --checkpoint checkpoints/best.pt \\
        --label-map checkpoints/label_map.json --output-dir classified_output
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from .config import Config
from .streamline_model import StreamlineClassifier, classify_tractogram
from .utils import label_map_to_class_names, load_checkpoint, load_label_map, resolve_device



from __future__ import annotations

import argparse
import logging
from importlib import resources
from typing import Optional

import numpy as np

from .config import Config
from .streamline_model import StreamlineClassifier, classify_tractogram
from .utils import label_map_to_class_names, load_checkpoint, load_label_map, resolve_device

logger = logging.getLogger("inference")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def _bundled_checkpoint_path() -> Optional[str]:
    """Path to the checkpoint shipped inside the installed package, if present."""
    path = resources.files("gemo").joinpath("checkpoints", "best.pt")
    return str(path) if path.is_file() else None


def _bundled_label_map_path() -> Optional[str]:
    """Path to the label map shipped inside the installed package, if present."""
    path = resources.files("gemo").joinpath("checkpoints", "label_map.json")
    return str(path) if path.is_file() else None


def run_inference(
    trk_path: str,
    checkpoint_path: str,
    label_map_path: str,
    output_dir: str,
    cfg: Config,
) -> dict:
    """
    Load a trained checkpoint + its label map, then classify every
    streamline of `trk_path` into per-tract .trk files under
    `output_dir` (plus "Unknown" for low-confidence streamlines).

    Returns
    -------
    dict[str, str]
        Maps each class name (+ "Unknown") to the .trk file written for it.
    """
    label_map = load_label_map(label_map_path)
    class_names = label_map_to_class_names(label_map)

    device = resolve_device(cfg.inference.device)
    model = StreamlineClassifier(
        num_handcrafted_features=cfg.model.num_handcrafted_features,
        num_classes=len(class_names),
        cnn_base_channels=cfg.model.cnn_base_channels,
        cnn_embedding_dim=cfg.model.cnn_embedding_dim,
        feature_hidden_dims=cfg.model.feature_hidden_dims,
        feature_embedding_dim=cfg.model.feature_embedding_dim,
        classifier_hidden_dims=cfg.model.classifier_hidden_dims,
        dropout=cfg.model.dropout,
        fusion_mode=cfg.model.fusion_mode,
    ).to(device)
    load_checkpoint(checkpoint_path, model, device=str(device))
    model.eval()

    written = classify_tractogram(
        trk_path=trk_path,
        model=model,
        output_dir=output_dir,
        class_names=class_names,
        threshold=cfg.inference.threshold,
        batch_size=cfg.inference.batch_size,
        device=str(device),
        grid_size=cfg.data.grid_size,
        scales=np.array(cfg.data.scales),
        feature_names=cfg.data.feature_names,
        truncate=cfg.data.truncate,
    )
    logger.info("Wrote %d output .trk files: %s", len(written), written)
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a whole-brain tractogram with a trained StreamlineClassifier."
    )
    parser.add_argument("--trk-path", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a .pt checkpoint. Defaults to the checkpoint bundled with the package.",
    )
    parser.add_argument(
        "--label-map",
        default=None,
        help="Path to a label_map.json. Defaults to the one bundled with the package.",
    )
    parser.add_argument("--output-dir", default="classified_output")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def cli() -> None:
    """Entry point for the `gemo-infer` console script (see pyproject.toml)."""
    args = _parse_args()

    checkpoint_path = args.checkpoint or _bundled_checkpoint_path()
    if checkpoint_path is None:
        raise SystemExit(
            "No --checkpoint given and no bundled checkpoint was found in this install. "
            "Pass --checkpoint /path/to/best.pt explicitly."
        )

    label_map_path = args.label_map or _bundled_label_map_path()
    if label_map_path is None:
        raise SystemExit(
            "No --label-map given and no bundled label_map.json was found in this install. "
            "Pass --label-map /path/to/label_map.json explicitly."
        )

    cfg = Config()
    if args.threshold is not None:
        cfg.inference.threshold = args.threshold
    if args.device is not None:
        cfg.inference.device = args.device

    run_inference(args.trk_path, checkpoint_path, label_map_path, args.output_dir, cfg)


if __name__ == "__main__":
    cli()
