"""Run inference on one or more TEM grid images."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.constants import CLASS_NAMES
from src.dataset import build_transforms
from src.device import resolve_device
from src.model import create_model
from src.preprocessing import CircleCrop, extract_circle_crops, render_classified_overlay


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict graphene on TEM images")
    parser.add_argument(
        "images",
        nargs="+",
        help="One or more image paths (quote paths that contain spaces)",
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

    model, transform, idx_to_class = load_predictor(args.checkpoint, device)

    for image_path in coalesce_image_paths(args.images):
        image_output_dir = args.output_dir / image_path.stem
        num_circles, counts, overlay_path = predict_image(
            model,
            transform,
            device,
            idx_to_class,
            image_path,
            output_dir=image_output_dir,
        )
        print(f"\n{image_path}")
        print(f"  circles detected: {num_circles}")
        for class_name in sorted(CLASS_NAMES):
            count = counts.get(class_name, 0)
            pct = (count / num_circles) * 100 if num_circles else 0.0
            print(f"  {class_name}: {count} ({pct:.1f}%)")
        print(f"  crops saved to: {image_output_dir}/{{class}}/")
        if overlay_path is not None:
            print(f"  color overlay: {overlay_path}")


if __name__ == "__main__":
    main()
