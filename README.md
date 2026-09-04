# GEMO

**GEMO** (**Ge**ometric & **Mo**rphological tractography classifier) is a
compact dual-branch deep learning model that classifies tractography
streamlines into white matter tract classes, using:

1. a small CNN over each streamline's 12x12x3 RGB image (`xyz2RGB`), and
2. an MLP over handcrafted geometric/morphological features (length,
   curvature, tortuosity, spectral entropy, fractal dimension, lacunarity).

This repository includes the full pipeline: offline feature/bounds
extraction, model training, and whole-brain tractogram inference. A trained
checkpoint is bundled with the package, so `gemo-infer` works immediately
after installation with no extra downloads.

## Installation

Install directly from GitHub (no PyPI account needed):

```bash
pip install git+https://github.com/amin-barati/GEMO.git
```

This requires `git` to be available on your machine (pip uses it to fetch
the repository), but nothing else — no need to clone the repo yourself.

## Quick start: classify a tractogram

```bash
gemo-infer --trk-path wholebrain.trk --output-dir classified_output
```

By default this uses the checkpoint and label map bundled with the package
(`gemo/checkpoints/best.pt`, `gemo/checkpoints/label_map.json`). To use a
different, custom-trained model instead:

```bash
gemo-infer --trk-path wholebrain.trk \
    --checkpoint /path/to/best.pt \
    --label-map /path/to/label_map.json \
    --output-dir classified_output \
    --threshold 0.7
```

This writes one `.trk` file per predicted tract class, plus one
`_Unknown.trk` file for streamlines whose top softmax probability falls
below `--threshold`.

## Training on your own data

If you want to train your own model instead of using the bundled
checkpoint, you first need two offline preprocessing steps:

```bash
python -c "from gemo.streamline_features import compute_and_save_bounds_metadata; \
    compute_and_save_bounds_metadata('TRK', 'bounds.h5')"

python -c "from gemo.streamline_features import extract_features_from_directory; \
    extract_features_from_directory('TRK', 'features.h5', bounds_h5_path='bounds.h5')"
```

Then train:

```bash
gemo-train --trk-dir TRK --features-h5 features.h5 --bounds-h5 bounds.h5 \
    --checkpoint-dir my_run
```

Run `gemo-train --help` for every available option (batch size, epochs,
learning rate, per-class streamline cap, class-weighting scheme, etc.).
Training automatically resumes from `my_run/last.pt` if interrupted and
re-run with the same `--checkpoint-dir`.

## Python API

```python
from gemo import StreamlineClassifier, classify_tractogram

model = StreamlineClassifier(num_handcrafted_features=6, num_classes=30)
```

See the docstrings in `gemo/streamline_model.py`, `gemo/streamline_features.py`,
and `gemo/xyz2RGB.py` for the full API.

## Citation

If you use GEMO in your research, please cite this repository.

## License

MIT — see [LICENSE](LICENSE).
