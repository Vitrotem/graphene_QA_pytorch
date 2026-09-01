"""One-time migration: data/train|test -> data/raw and data/processed."""

import shutil
from pathlib import Path

from src.constants import CLASS_NAMES, PROCESSED_DIR_NAME, RAW_DIR_NAME

DATA = Path("data")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def move_raw_images() -> None:
    raw_root = DATA / RAW_DIR_NAME
    for class_name in CLASS_NAMES:
        dest = raw_root / class_name
        dest.mkdir(parents=True, exist_ok=True)
        for split in ("train", "test"):
            src = DATA / split / class_name
            if not src.exists():
                continue
            for path in sorted(src.iterdir()):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                target = dest / path.name
                if target.exists():
                    continue
                shutil.move(str(path), str(target))
                print(f"raw: {path.name} -> {dest}")


def merge_processed_crops() -> None:
    processed_root = DATA / PROCESSED_DIR_NAME
    for class_name in CLASS_NAMES:
        dest = processed_root / class_name
        dest.mkdir(parents=True, exist_ok=True)
        index = 1
        sources: set[str] = set()

        for split in ("train", "test"):
            src = DATA / split / "processed" / class_name
            if not src.exists():
                continue
            manifest = src / "sources.txt"
            if manifest.exists():
                sources.update(
                    line.strip()
                    for line in manifest.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            for path in sorted(src.glob("*.jpg")):
                if not path.stem.isdigit():
                    continue
                shutil.copy2(path, dest / f"{index:05d}.jpg")
                index += 1

        if sources:
            (dest / "sources.txt").write_text(
                "\n".join(sorted(sources)) + "\n", encoding="utf-8"
            )
        print(f"processed/{class_name}: {index - 1} crops")


if __name__ == "__main__":
    move_raw_images()
    merge_processed_crops()
    print("Migration complete.")
