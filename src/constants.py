"""Shared dataset class names and data layout paths."""

CLASS_NAMES = ("graphene", "no_graphene", "wrinkles")
CLASS_TO_IDX = {name: idx for idx, name in enumerate(sorted(CLASS_NAMES))}
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "graphene": (0, 200, 0),
    "no_graphene": (220, 50, 50),
    "wrinkles": (255, 180, 0),
}

RAW_DIR_NAME = "raw"
PROCESSED_DIR_NAME = "processed"
