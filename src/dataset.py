"""Dataset loading for TEM grid circle classification."""

from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from src.constants import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    PROCESSED_DIR_NAME,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class ProcessedCircleDataset(Dataset):
    """Load preprocessed circle crops from data/processed/{class}/."""

    def __init__(self, root: Path, transform):
        self.transform = transform
        self.class_to_idx = CLASS_TO_IDX
        self.samples: list[tuple[Path, int]] = []

        for class_name in sorted(CLASS_NAMES):
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.glob("*.jpg")):
                if path.stem.isdigit():
                    self.samples.append((path, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(
    image_size: int = 224,
    augment: bool = False,
):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    steps: list = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
    ]
    if augment:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
            ]
        )
    steps.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(steps)


def processed_dir(data_dir: Path) -> Path:
    path = data_dir / PROCESSED_DIR_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"Processed crops not found at {path}. Run: poetry run vt-preprocess"
        )
    return path


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 0,
    test_fraction: float = 0.15,
    random_state: int = 42,
):
    full_dataset = ProcessedCircleDataset(
        processed_dir(data_dir),
        transform=None,
    )
    _check_class_mapping(full_dataset.class_to_idx, full_dataset.samples)

    labels = [label for _, label in full_dataset.samples]
    train_idx, test_idx = train_test_split(
        range(len(full_dataset)),
        test_size=test_fraction,
        random_state=random_state,
        stratify=labels,
    )

    train_dataset = ProcessedCircleDataset(
        processed_dir(data_dir),
        transform=build_transforms(image_size, augment=True),
    )
    test_dataset = ProcessedCircleDataset(
        processed_dir(data_dir),
        transform=build_transforms(image_size, augment=False),
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(test_dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, train_dataset.class_to_idx


def _check_class_mapping(
    class_to_idx: dict[str, int], samples: list[tuple[Path, int]]
) -> None:
    expected = set(CLASS_NAMES)
    if set(class_to_idx) != expected:
        raise ValueError(
            f"Expected class folders {expected}, found {set(class_to_idx)}"
        )
    present = {CLASS_NAMES[label] for _, label in samples}
    missing = expected - present
    if missing:
        raise ValueError(f"No processed crops found for classes: {sorted(missing)}")
