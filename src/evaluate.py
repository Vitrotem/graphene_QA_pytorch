"""Evaluate a trained graphene classifier on the test set."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.dataset import create_dataloaders
from src.device import resolve_device
from src.model import create_model


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in tqdm(loader, desc="Evaluate"):
        images = images.to(device)
        outputs = model(images)
        all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.tolist())

    return all_labels, all_preds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate graphene classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/best_model.pt")
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
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

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_to_idx: dict[str, int] = checkpoint["class_to_idx"]
    image_size = checkpoint.get("image_size", args.image_size)

    _, test_loader, _ = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
    )

    model = create_model(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    target_names = [idx_to_class[i] for i in range(len(class_to_idx))]

    labels, preds = run_inference(model, test_loader, device)

    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Saved test accuracy: {checkpoint.get('test_accuracy', 'unknown')}\n")

    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=target_names))


if __name__ == "__main__":
    main()
