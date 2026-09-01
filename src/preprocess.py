"""Batch-preprocess TEM grid images into per-circle crops."""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.constants import CLASS_NAMES, PROCESSED_DIR_NAME, RAW_DIR_NAME
from src.preprocessing import extract_circle_crops

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MANIFEST_NAME = "sources.txt"


def collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_manifest(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    return {
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def save_manifest(manifest_path: Path, sources: set[str]) -> None:
    manifest_path.write_text(
        "\n".join(sorted(sources)) + ("\n" if sources else ""),
        encoding="utf-8",
    )


def next_crop_index(processed_class_dir: Path) -> int:
    indices = [
        int(path.stem)
        for path in processed_class_dir.glob("*.jpg")
        if path.stem.isdigit()
    ]
    return max(indices, default=0) + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and save individual circle crops from TEM grid images"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Dataset root containing raw/ (default: data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess images even if output already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.data_dir / RAW_DIR_NAME
    processed_root = args.data_dir / PROCESSED_DIR_NAME

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw images not found at {raw_dir}")

    total_crops = 0
    class_dirs = [
        raw_dir / name for name in sorted(CLASS_NAMES) if (raw_dir / name).exists()
    ]
    print(f"Processing raw/ ({len(class_dirs)} classes)...")

    with tqdm(total=sum(len(collect_images(d)) for d in class_dirs)) as pbar:
        for class_dir in class_dirs:
            processed_class_dir = processed_root / class_dir.name
            if args.force and processed_class_dir.exists():
                shutil.rmtree(processed_class_dir)
            processed_class_dir.mkdir(parents=True, exist_ok=True)

            manifest_path = processed_class_dir / MANIFEST_NAME
            processed_sources = load_manifest(manifest_path)
            crop_index = next_crop_index(processed_class_dir)

            for image_path in collect_images(class_dir):
                source_key = str(image_path.resolve())
                if source_key in processed_sources and not args.force:
                    pbar.update(1)
                    continue

                crops = extract_circle_crops(
                    np.array(Image.open(image_path).convert("L"))
                )
                for crop in crops:
                    crop.image.save(
                        processed_class_dir / f"{crop_index:05d}.jpg",
                        quality=95,
                    )
                    crop_index += 1
                    total_crops += 1

                processed_sources.add(source_key)
                save_manifest(manifest_path, processed_sources)
                pbar.update(1)

    print(f"Done. Wrote {total_crops} circle crop(s) to {processed_root}/")


if __name__ == "__main__":
    main()
