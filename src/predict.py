"""Run inference on TEM grid images and optionally quantify a folder to CSV."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.constants import CLASS_NAMES
from src.dataset import IMAGE_EXTENSIONS, build_transforms
from src.device import resolve_device
from src.model import create_model
from src.preprocessing import CircleCrop, extract_circle_crops, render_classified_overlay

DEFAULT_CSV_NAME = "prediction_stats.csv"


def coalesce_image_paths(parts: list[str]) -> list[Path]:
    """Join argv fragments that were split by spaces in filenames."""
    paths: list[Path] = []
    i = 0
    while i < len(parts):
        matched = False
        for j in range(i + 1, len(parts) + 1):
            candidate = Path(" ".join(parts[i:j]))
            if candidate.exists():
                paths.append(candidate)
                i = j
                matched = True
                break
        if not matched:
            paths.append(Path(" ".join(parts[i:])))
            break
    return paths


def collect_images(paths: list[Path]) -> list[Path]:
    """Expand files and directories into a sorted list of image paths."""
    images: list[Path] = []
    for path in paths:
        if path.is_dir():
            images.extend(
                sorted(
                    p
                    for p in path.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )
            )
        elif path.is_file():
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image type: {path}")
            images.append(path)
        else:
            raise FileNotFoundError(f"Path not found: {path}")
    # Preserve order but drop duplicates
    seen: set[Path] = set()
    unique: list[Path] = []
    for image in images:
        resolved = image.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(image)
    return unique


def resolve_csv_path(
    input_paths: list[Path],
    output_dir: Path,
    csv_name: str,
) -> Path:
    """Write the report into a quantified folder when that is the sole input."""
    if len(input_paths) == 1 and input_paths[0].is_dir():
        return input_paths[0] / csv_name
    return output_dir / csv_name


def load_predictor(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_to_idx: dict[str, int] = checkpoint["class_to_idx"]
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    image_size = checkpoint.get("image_size", 224)

    model = create_model(num_classes=len(class_to_idx))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    transform = build_transforms(image_size, augment=False)
    return model, transform, idx_to_class


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    transform,
    device: torch.device,
    idx_to_class: dict[int, str],
    image_path: Path,
    output_dir: Path | None = None,
) -> tuple[int, dict[str, int], Path | None]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = np.array(Image.open(image_path).convert("L"))
    crops = extract_circle_crops(gray)
    if not crops:
        raise ValueError(f"No circles detected in {image_path}")

    counts: Counter[str] = Counter()
    classified: list[tuple[CircleCrop, str]] = []

    for crop in crops:
        tensor = transform(crop.image.convert("RGB")).unsqueeze(0).to(device)
        pred_idx = model(tensor).argmax(dim=1).item()
        label = idx_to_class[pred_idx]
        counts[label] += 1
        classified.append((crop, label))

    overlay_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in output_dir.glob("*.jpg"):
            if path.stem.isdigit():
                path.unlink()

        class_counters: Counter[str] = Counter()
        for crop, label in classified:
            class_dir = output_dir / label
            class_dir.mkdir(parents=True, exist_ok=True)
            class_counters[label] += 1
            crop.image.save(
                class_dir / f"{class_counters[label]:05d}.jpg",
                quality=95,
            )
        overlay = render_classified_overlay(gray, classified)
        overlay_path = output_dir / "classified_overlay.jpg"
        overlay.save(overlay_path, quality=95)

    return len(crops), dict(counts), overlay_path


def build_stats_row(
    image_path: Path,
    num_circles: int,
    counts: dict[str, int],
    status: str = "ok",
    error: str = "",
) -> dict[str, object]:
    row: dict[str, object] = {
        "image": image_path.name,
        "path": str(image_path),
        "status": status,
        "circles_detected": num_circles,
    }
    for class_name in sorted(CLASS_NAMES):
        count = counts.get(class_name, 0)
        pct = (count / num_circles) * 100 if num_circles else 0.0
        row[class_name] = count
        row[f"{class_name}_pct"] = round(pct, 2)
    row["error"] = error
    return row


def write_stats_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

        totals = Counter()
        ok_rows = [row for row in rows if row["status"] == "ok"]
        for row in ok_rows:
            totals["circles_detected"] += int(row["circles_detected"])
            for class_name in sorted(CLASS_NAMES):
                totals[class_name] += int(row[class_name])

        total_circles = totals["circles_detected"]
        summary: dict[str, object] = {
            "image": "TOTAL",
            "path": "",
            "status": "summary",
            "circles_detected": total_circles,
            "error": "",
        }
        for class_name in sorted(CLASS_NAMES):
            count = totals[class_name]
            pct = (count / total_circles) * 100 if total_circles else 0.0
            summary[class_name] = count
            summary[f"{class_name}_pct"] = round(pct, 2)
        writer.writerow(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict graphene on TEM images. Pass image files and/or a folder; "
            "folder quantification writes a CSV stats report into that folder."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Image file(s) and/or a folder of images (quote paths that contain spaces)",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/best_model.pt")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/predictions"),
        help="Directory for preprocessed crops and color overlay (default: outputs/predictions)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV_NAME,
        help=f"CSV report filename (default: {DEFAULT_CSV_NAME})",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing the CSV stats report",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "gpu", "cpu"],
        default="auto",
        help="Device to use (default: auto)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    input_paths = coalesce_image_paths(args.paths)
    image_paths = collect_images(input_paths)
    if not image_paths:
        raise FileNotFoundError("No images found in the given path(s)")

    model, transform, idx_to_class = load_predictor(args.checkpoint, device)
    rows: list[dict[str, object]] = []

    for image_path in image_paths:
        image_output_dir = args.output_dir / image_path.stem
        try:
            num_circles, counts, overlay_path = predict_image(
                model,
                transform,
                device,
                idx_to_class,
                image_path,
                output_dir=image_output_dir,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch quantification running
            print(f"\n{image_path}")
            print(f"  ERROR: {exc}")
            rows.append(
                build_stats_row(
                    image_path,
                    num_circles=0,
                    counts={},
                    status="error",
                    error=str(exc),
                )
            )
            continue

        rows.append(build_stats_row(image_path, num_circles, counts))
        print(f"\n{image_path}")
        print(f"  circles detected: {num_circles}")
        for class_name in sorted(CLASS_NAMES):
            count = counts.get(class_name, 0)
            pct = (count / num_circles) * 100 if num_circles else 0.0
            print(f"  {class_name}: {count} ({pct:.1f}%)")
        print(f"  crops saved to: {image_output_dir}/{{class}}/")
        if overlay_path is not None:
            print(f"  color overlay: {overlay_path}")

    if not args.no_csv:
        csv_path = resolve_csv_path(input_paths, args.output_dir, args.csv)
        write_stats_csv(rows, csv_path)
        print(f"\nCSV report: {csv_path}")


if __name__ == "__main__":
    main()
