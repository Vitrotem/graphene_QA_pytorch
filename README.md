# vitrotem_grid_ai

Three-class classifier for TEM grid circle crops (`graphene`, `no_graphene`, `wrinkles`) using transfer-learned ResNet18.

## Setup

Requires [Poetry](https://python-poetry.org/) (Python 3.11+).

```bash
poetry install
```

## Dataset layout

```
data/
├── raw/                  # unprocessed full-grid images
│   ├── graphene/
│   ├── no_graphene/
│   └── wrinkles/         # may be empty
└── processed/            # circle crops written by preprocess
    ├── graphene/00001.jpg
    ├── no_graphene/
    └── wrinkles/
```

Place source images in `data/raw/{class}/`. Supported formats: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`

## Preprocess

```bash
preprocess
```

Reads from `data/raw/`, writes numbered crops to `data/processed/`. Use `--force` to rebuild.

## Train, evaluate, predict

```bash
train
eval
predict path/to/image.jpg
predict path/to/folder/
```

Training and evaluation load crops from `data/processed/` (held-out split is automatic). Predicting a single image saves classified crops under `outputs/predictions/<image_stem>/{class}/`.

Passing a folder quantifies every image in it and writes into that folder: `prediction_stats.csv`, plus per-image `<image_stem>/{class}/` crops and `classified_overlay.jpg`. Use `--csv NAME` to rename the report, or `--no-csv` to skip it.

## How it works

1. **preprocess** — detect circles, crop, mask, normalize → `data/processed/`
2. **train / eval** — load crops, resize, classify (stratified train/test split)
3. **predict** — extract circles from a new image or folder, classify each, report counts (and CSV for folders)
