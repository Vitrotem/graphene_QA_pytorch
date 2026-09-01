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
```

Training and evaluation load crops from `data/processed/` (held-out split is automatic). Prediction preprocesses each input image and saves classified crops under `outputs/predictions/<image_stem>/{class}/`.

## How it works

1. **preprocess** — detect circles, crop, mask, normalize → `data/processed/`
2. **train / eval** — load crops, resize, classify (stratified train/test split)
3. **predict** — extract circles from a new image, classify each, report counts
