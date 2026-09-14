"""Interactive UI for tuning Hough circle detection parameters."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.preprocessing import (
    CircleDetectParams,
    crop_metadata,
    detect_circles,
    matches_skip_center,
)

PREVIEW_MAX_DIM = 900
DEBOUNCE_MS = 80
KEEP_COLOR = (0, 220, 0)
SKIP_COLOR = (220, 40, 40)


@dataclass(frozen=True)
class CircleTuneResult:
    """Detection params plus circles the user clicked to skip on the preview image."""

    params: CircleDetectParams
    skipped_centers: tuple[tuple[int, int], ...]


def _build_preview(
    cropped: np.ndarray,
    params: CircleDetectParams,
    display_scale: float,
    skipped_centers: list[tuple[int, int]],
) -> tuple[Image.Image, list[tuple[int, int, int]], int, int]:
    circles = detect_circles(cropped, params=params)
    rgb = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
    thickness = max(1, int(round(2 / display_scale))) if display_scale > 0 else 2
    kept = 0
    skipped = 0
    for x, y, r in circles:
        is_skipped = matches_skip_center(x, y, r, skipped_centers)
        color = SKIP_COLOR if is_skipped else KEEP_COLOR
        cv2.circle(rgb, (x, y), r, color, thickness)
        if is_skipped:
            skipped += 1
            # Mark skipped with a small cross at the center
            arm = max(3, r // 5)
            cv2.line(rgb, (x - arm, y - arm), (x + arm, y + arm), SKIP_COLOR, thickness)
            cv2.line(rgb, (x - arm, y + arm), (x + arm, y - arm), SKIP_COLOR, thickness)
        else:
            kept += 1

    if display_scale < 1.0:
        h, w = rgb.shape[:2]
        rgb = cv2.resize(
            rgb,
            (int(w * display_scale), int(h * display_scale)),
            interpolation=cv2.INTER_AREA,
        )
    return Image.fromarray(rgb, mode="RGB"), circles, kept, skipped


def tune_circle_params(image_path: Path) -> CircleTuneResult | None:
    """Show a slider UI for circle detection; return result or None if cancelled."""
    gray = np.array(Image.open(image_path).convert("L"))
    cropped, _ = crop_metadata(gray)
    h, w = cropped.shape
    display_scale = min(1.0, PREVIEW_MAX_DIM / max(h, w))

    defaults = CircleDetectParams()
    result: dict[str, CircleTuneResult | None] = {"value": None}
    skipped_centers: list[tuple[int, int]] = []
    current_circles: list[tuple[int, int, int]] = []

    root = tk.Tk()
    root.title(f"Tune circle detection — {image_path.name}")
    root.resizable(True, True)

    main = ttk.Frame(root, padding=8)
    main.pack(fill=tk.BOTH, expand=True)
    main.columnconfigure(0, weight=1)
    main.columnconfigure(1, weight=0)
    main.rowconfigure(0, weight=1)

    # Left: preview image
    preview_frame = ttk.Frame(main)
    preview_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 8))

    preview_label = ttk.Label(preview_frame)
    preview_label.pack()
    preview_label.configure(cursor="hand2")

    # Right: status, sliders, buttons
    side = ttk.Frame(main)
    side.grid(row=0, column=1, sticky=tk.NS)

    count_var = tk.StringVar(value="Circles: 0 kept, 0 skipped")
    ttk.Label(side, textvariable=count_var).pack(anchor=tk.W, pady=(0, 2))
    ttk.Label(
        side,
        text="Click a circle to skip or include it\n(red = skipped)",
        justify=tk.LEFT,
    ).pack(anchor=tk.W, pady=(0, 8))

    controls = ttk.Frame(side)
    controls.pack(fill=tk.X)

    param1_var = tk.DoubleVar(value=defaults.param1)
    param2_var = tk.DoubleVar(value=defaults.param2)
    min_dist_var = tk.DoubleVar(value=defaults.min_dist_factor)
    min_r_pct_var = tk.DoubleVar(value=defaults.min_radius_frac * 100.0)
    max_r_pct_var = tk.DoubleVar(value=defaults.max_radius_frac * 100.0)

    value_labels: dict[str, tk.StringVar] = {}
    debounce_id: list[str | None] = [None]
    photo_ref: list[ImageTk.PhotoImage | None] = [None]

    def current_params() -> CircleDetectParams:
        min_frac = min_r_pct_var.get() / 100.0
        max_frac = max_r_pct_var.get() / 100.0
        if max_frac <= min_frac:
            max_frac = min_frac + 0.01
        return CircleDetectParams(
            param1=float(param1_var.get()),
            param2=float(param2_var.get()),
            min_dist_factor=float(min_dist_var.get()),
            min_radius_frac=float(min_frac),
            max_radius_frac=float(max_frac),
        )

    def refresh_preview() -> None:
        nonlocal current_circles
        params = current_params()
        preview, circles, kept, skipped = _build_preview(
            cropped, params, display_scale, skipped_centers
        )
        current_circles = circles
        photo = ImageTk.PhotoImage(preview)
        photo_ref[0] = photo
        preview_label.configure(image=photo)
        count_var.set(f"Circles: {kept} kept, {skipped} skipped")
        value_labels["param1"].set(f"{params.param1:.0f}")
        value_labels["param2"].set(f"{params.param2:.0f}")
        value_labels["min_dist"].set(f"{params.min_dist_factor:.2f}")
        value_labels["min_r"].set(f"{params.min_radius_frac * 100:.1f}")
        value_labels["max_r"].set(f"{params.max_radius_frac * 100:.1f}")

    def schedule_refresh(*_args: object) -> None:
        if debounce_id[0] is not None:
            root.after_cancel(debounce_id[0])
        debounce_id[0] = root.after(DEBOUNCE_MS, refresh_preview)

    def on_preview_click(event: tk.Event) -> None:
        if display_scale <= 0 or not current_circles:
            return
        full_x = event.x / display_scale
        full_y = event.y / display_scale

        best: tuple[int, int, int] | None = None
        best_dist = float("inf")
        for x, y, r in current_circles:
            dist = ((full_x - x) ** 2 + (full_y - y) ** 2) ** 0.5
            # Allow clicking slightly outside the rim
            if dist <= r * 1.15 and dist < best_dist:
                best = (x, y, r)
                best_dist = dist

        if best is None:
            return

        bx, by, br = best
        # Toggle: remove matching skip, or add this center
        matched_idx = None
        for i, (sx, sy) in enumerate(skipped_centers):
            if matches_skip_center(bx, by, br, [(sx, sy)]):
                matched_idx = i
                break
        if matched_idx is not None:
            skipped_centers.pop(matched_idx)
        else:
            skipped_centers.append((bx, by))
        refresh_preview()

    def add_slider(
        row: int,
        label: str,
        variable: tk.DoubleVar,
        from_: float,
        to: float,
        key: str,
        resolution: float,
    ) -> None:
        ttk.Label(controls, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        scale = tk.Scale(
            controls,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            resolution=resolution,
            variable=variable,
            length=220,
            showvalue=False,
            command=lambda _v: schedule_refresh(),
        )
        scale.grid(row=row, column=1, sticky=tk.EW, padx=8, pady=2)
        value_var = tk.StringVar()
        value_labels[key] = value_var
        ttk.Label(controls, textvariable=value_var, width=6).grid(
            row=row, column=2, sticky=tk.E, pady=2
        )

    controls.columnconfigure(1, weight=1)
    add_slider(0, "param1", param1_var, 10, 200, "param1", 1)
    add_slider(1, "param2", param2_var, 5, 100, "param2", 1)
    add_slider(2, "min dist factor", min_dist_var, 1.0, 4.0, "min_dist", 0.05)
    add_slider(3, "min radius %", min_r_pct_var, 1.0, 20.0, "min_r", 0.1)
    add_slider(4, "max radius %", max_r_pct_var, 2.0, 30.0, "max_r", 0.1)

    buttons = ttk.Frame(side)
    buttons.pack(fill=tk.X, pady=(16, 0))

    def on_reset() -> None:
        param1_var.set(defaults.param1)
        param2_var.set(defaults.param2)
        min_dist_var.set(defaults.min_dist_factor)
        min_r_pct_var.set(defaults.min_radius_frac * 100.0)
        max_r_pct_var.set(defaults.max_radius_frac * 100.0)
        skipped_centers.clear()
        refresh_preview()

    def on_ok() -> None:
        result["value"] = CircleTuneResult(
            params=current_params(),
            skipped_centers=tuple(skipped_centers),
        )
        root.destroy()

    def on_cancel() -> None:
        result["value"] = None
        root.destroy()

    preview_label.bind("<Button-1>", on_preview_click)

    ttk.Button(buttons, text="Reset", command=on_reset).pack(fill=tk.X, pady=(0, 4))
    ttk.Button(buttons, text="OK", command=on_ok).pack(fill=tk.X, pady=(0, 4))
    ttk.Button(buttons, text="Cancel", command=on_cancel).pack(fill=tk.X)

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    refresh_preview()
    root.mainloop()
    return result["value"]
