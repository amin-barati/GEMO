"""
GEMO -- Geometric & Morphological tractography streamline classifier.

A dual-branch (RGB image + handcrafted geometric/morphological features)
deep learning model for classifying tractography streamlines into white
matter tract classes, plus the full offline preprocessing pipeline
(subject-wise coordinate normalization, streaming feature extraction) it
depends on.

Command-line tools (installed alongside the package)
------------------------------------------------------
    gemo-train   Train a StreamlineClassifier on a directory of .trk files.
    gemo-infer   Classify a whole-brain .trk file with a trained checkpoint
                 (defaults to the checkpoint bundled with this package).

Python API
----------
>>> from gemo import StreamlineClassifier, classify_tractogram
>>> model = StreamlineClassifier(num_handcrafted_features=6, num_classes=30)
"""

from .streamline_model import (
    StreamlineClassifier,
    CNNFeatureExtractor,
    HandcraftedFeatureEncoder,
    FeatureFusion,
    ClassificationHead,
    StreamlineDataset,
    build_label_mapping,
    classify_tractogram,
    count_trainable_parameters,
)
from .xyz2RGB import (
    streamline_to_rgb,
    load_bounds_metadata_h5,
    save_bounds_metadata_h5,
    group_trk_files_by_subject,
)
from .streamline_features import (
    extract_features_from_directory,
    compute_and_save_bounds_metadata,
)
from .config import Config

__version__ = "0.1.0"

__all__ = [
    "StreamlineClassifier",
    "CNNFeatureExtractor",
    "HandcraftedFeatureEncoder",
    "FeatureFusion",
    "ClassificationHead",
    "StreamlineDataset",
    "build_label_mapping",
    "classify_tractogram",
    "count_trainable_parameters",
    "streamline_to_rgb",
    "load_bounds_metadata_h5",
    "save_bounds_metadata_h5",
    "group_trk_files_by_subject",
    "extract_features_from_directory",
    "compute_and_save_bounds_metadata",
    "Config",
    "__version__",
]
