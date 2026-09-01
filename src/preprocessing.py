"""Circle masking and intensity preprocessing for TEM grid images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from src.constants import CLASS_COLORS

METADATA_FRACTION = 0.08
DETECT_MAX_DIM = 1024
CONTRAST_ALPHA = 1.15
NORMALIZE_PERCENTILES = (2.0, 98.0)
SIZE_TOLERANCE = 0.20
SPACING_TOLERANCE = 0.12
GRID_MIN_NEIGHBORS = 2
INTERIOR_CONTRAST_MIN = 3.0
HOUGH_PARAM1 = 56
HOUGH_PARAM2 = 30
HOUGH_MIN_DIST_FACTOR = 1.8


@dataclass(frozen=True)
class CircleCrop:
    """A preprocessed circle crop with its position in the source image."""

    image: Image.Image
    x: int
    y: int
    r: int


def crop_metadata(gray: np.ndarray) -> tuple[np.ndarray, int]:
    """Crop the metadata bar; return cropped image and original height."""
    original_h = gray.shape[0]
    crop_h = int(original_h * (1.0 - METADATA_FRACTION))
    return gray[:crop_h, :], original_h


def _filter_by_average_size(
    circles: list[tuple[int, int, int]],
    tolerance: float = SIZE_TOLERANCE,
) -> list[tuple[int, int, int]]:
    """Keep circles whose radius is within *tolerance* of the median size."""
    if len(circles) <= 1:
        return circles

    median_r = float(np.median([r for _, _, r in circles]))
    if median_r <= 0:
        return circles

    lo, hi = median_r * (1.0 - tolerance), median_r * (1.0 + tolerance)
    filtered = [(x, y, r) for x, y, r in circles if lo <= r <= hi]
    return filtered if filtered else circles


def _filter_by_interior_brightness(
    gray: np.ndarray,
    circles: list[tuple[int, int, int]],
    min_contrast: float = INTERIOR_CONTRAST_MIN,
) -> list[tuple[int, int, int]]:
    """Keep circles whose interior is brighter than the surrounding ring."""
    h, w = gray.shape
    filtered: list[tuple[int, int, int]] = []

    for x, y, r in circles:
        if r < 3:
            continue
        inner = np.zeros((h, w), dtype=np.uint8)
        ring = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(inner, (x, y), max(1, int(r * 0.65)), 255, thickness=-1)
        cv2.circle(ring, (x, y), int(r * 1.35), 255, thickness=-1)
        cv2.circle(ring, (x, y), int(r * 1.05), 0, thickness=-1)

        inner_vals = gray[inner > 0]
        ring_vals = gray[ring > 0]
        if inner_vals.size == 0 or ring_vals.size == 0:
            continue
        contrast = float(inner_vals.mean()) - float(ring_vals.mean())
        if contrast >= min_contrast:
            filtered.append((x, y, r))

    return filtered if filtered else circles


def _median_grid_pitch(circles: list[tuple[int, int, int]]) -> float | None:
    if len(circles) <= 1:
        return None
    centers = np.array([(x, y) for x, y, _ in circles], dtype=np.float32)
    nearest = []
    for i in range(len(centers)):
        deltas = centers - centers[i]
        dists = np.sqrt((deltas[:, 0] ** 2) + (deltas[:, 1] ** 2))
        dists[i] = np.inf
        nearest.append(float(dists.min()))
    return float(np.median(nearest))


def _filter_by_regular_spacing(
    circles: list[tuple[int, int, int]],
    tolerance: float = SPACING_TOLERANCE,
    min_neighbors: int = GRID_MIN_NEIGHBORS,
) -> list[tuple[int, int, int]]:
    """Keep circles that sit on the grid (enough neighbors at the median pitch)."""
    if len(circles) <= 2:
        return circles

    centers = np.array([(x, y) for x, y, _ in circles], dtype=np.float32)
    n = len(centers)
    pitch = _median_grid_pitch(circles)
    if pitch is None or pitch <= 0:
        return circles

    lo, hi = pitch * (1.0 - tolerance), pitch * (1.0 + tolerance)

    def keep_with(min_n: int) -> list[tuple[int, int, int]]:
        kept: list[tuple[int, int, int]] = []
        for i in range(n):
            deltas = centers - centers[i]
            dists = np.sqrt((deltas[:, 0] ** 2) + (deltas[:, 1] ** 2))
            dists[i] = np.inf
            neighbors = int(np.sum((dists >= lo) & (dists <= hi)))
            if neighbors >= min_n:
                kept.append(circles[i])
        return kept

    filtered = keep_with(min_neighbors)
    if len(filtered) < max(3, n // 4):
        filtered = keep_with(max(1, min_neighbors - 1))
    return filtered if filtered else circles


def _run_hough_circles(
    small: np.ndarray, min_r: int, max_r: int
) -> list[tuple[float, float, float]]:
    blurred = cv2.GaussianBlur(small, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min_r * HOUGH_MIN_DIST_FACTOR),
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in circles[0]]


def detect_circles(gray: np.ndarray) -> list[tuple[int, int, int]]:
    """Return list of (x, y, radius) in full-resolution coordinates."""
    h, w = gray.shape
    scale = min(1.0, DETECT_MAX_DIM / max(h, w))
    if scale < 1.0:
        small = cv2.resize(
            gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        small = gray
        scale = 1.0

    sh, sw = small.shape
    min_r = int(min(sh, sw) * 0.04)
    max_r = int(min(sh, sw) * 0.12)

    detected = _run_hough_circles(small, min_r, max_r)

    if not detected:
        detected_raw = _detect_circles_contours(small, scale)
    else:
        inv = 1.0 / scale
        detected_raw = [
            (int(round(x * inv)), int(round(y * inv)), int(round(r * inv)))
            for x, y, r in detected
        ]

    sized = _filter_by_average_size(detected_raw)
    bright = _filter_by_interior_brightness(gray, sized)
    return _filter_by_regular_spacing(bright)


def _detect_circles_contours(
    small: np.ndarray, scale: float
) -> list[tuple[int, int, int]]:
    """Fallback: Otsu threshold + circularity filter."""
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sh, sw = small.shape
    min_area = np.pi * (min(sh, sw) * 0.03) ** 2
    max_area = np.pi * (min(sh, sw) * 0.14) ** 2
    inv = 1.0 / scale
    circles: list[tuple[int, int, int]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter**2)
        if circularity < 0.8:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        circles.append((int(round(x * inv)), int(round(y * inv)), int(round(r * inv))))

    return circles


def build_circle_mask(
    shape: tuple[int, int], circles: list[tuple[int, int, int]]
) -> np.ndarray:
    """Build a binary mask (uint8 0/255) covering all detected circles."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for x, y, r in circles:
        cv2.circle(mask, (x, y), r, 255, thickness=-1)
    return mask


def normalize_and_enhance(
    gray: np.ndarray,
    mask: np.ndarray,
    contrast_alpha: float = CONTRAST_ALPHA,
    percentiles: tuple[float, float] = NORMALIZE_PERCENTILES,
) -> np.ndarray:
    """Normalize intensity within the mask and slightly increase contrast."""
    result = np.zeros_like(gray, dtype=np.float32)
    circle_pixels = mask > 0
    if not np.any(circle_pixels):
        return gray.astype(np.uint8)

    values = gray[circle_pixels].astype(np.float32)
    original_mean = float(values.mean())
    lo = float(np.percentile(values, percentiles[0]))
    hi = float(np.percentile(values, percentiles[1]))
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        result[circle_pixels] = gray[circle_pixels]
        return result.astype(np.uint8)

    normalized = (values - lo) / (hi - lo)
    normalized = np.clip(normalized, 0.0, 1.0)
    anchor = float(np.median(normalized))
    enhanced = np.clip((normalized - anchor) * contrast_alpha + anchor, 0.0, 1.0)
    output = enhanced * 255.0
    output = np.clip(output + (original_mean - float(output.mean())), 0.0, 255.0)
    result[circle_pixels] = output
    return result.astype(np.uint8)


def extract_circle_crops(gray: np.ndarray) -> list[CircleCrop]:
    """Detect circles and return one masked, normalized crop per circle."""
    cropped, _ = crop_metadata(gray)
    circles = detect_circles(cropped)
    h, w = cropped.shape
    crops: list[CircleCrop] = []

    for x, y, r in circles:
        x0, x1 = max(0, x - r), min(w, x + r)
        y0, y1 = max(0, y - r), min(h, y + r)
        if x1 <= x0 or y1 <= y0:
            continue

        patch = cropped[y0:y1, x0:x1].copy()
        mask = np.zeros(patch.shape, dtype=np.uint8)
        cv2.circle(mask, (x - x0, y - y0), r, 255, thickness=-1)
        masked = np.zeros_like(patch)
        masked[mask > 0] = patch[mask > 0]
        enhanced = normalize_and_enhance(masked, mask)
        crops.append(
            CircleCrop(
                image=Image.fromarray(enhanced, mode="L"),
                x=x,
                y=y,
                r=r,
            )
        )

    return crops


def render_classified_overlay(
    gray: np.ndarray,
    classified: list[tuple[CircleCrop, str]],
) -> Image.Image:
    """Draw color-coded circle outlines on the grid image by predicted class."""
    cropped, _ = crop_metadata(gray)
    rgb = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)

    for crop, label in classified:
        color = CLASS_COLORS.get(label, (255, 255, 255))
        cv2.circle(rgb, (crop.x, crop.y), crop.r, color, 2)

    return Image.fromarray(rgb, mode="RGB")


def preprocess_array(gray: np.ndarray) -> np.ndarray:
    """Detect circles, mask non-circle regions, normalize and enhance."""
    cropped, original_h = crop_metadata(gray)
    circles = detect_circles(cropped)
    mask = build_circle_mask(cropped.shape, circles)
    processed = normalize_and_enhance(cropped, mask)

    if processed.shape[0] < original_h:
        padded = np.zeros((original_h, processed.shape[1]), dtype=np.uint8)
        padded[: processed.shape[0], :] = processed
        return padded
    return processed


def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess a PIL image; returns a grayscale PIL image."""
    gray = np.array(image.convert("L"))
    processed = preprocess_array(gray)
    return Image.fromarray(processed, mode="L")
