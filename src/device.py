"""Device selection for training and evaluation."""

import torch


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU requested but not available")
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device {device!r}; use auto, cuda, gpu, or cpu")
