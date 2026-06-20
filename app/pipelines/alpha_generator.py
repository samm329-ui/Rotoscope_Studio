"""Stage 8: Alpha matte generation.

Converts the fused / smoothed mask list into a clean ``(H, W)``
uint8 alpha array at the original source resolution. Each alpha is
resized with bilinear interpolation to preserve soft edges.
"""
from __future__ import annotations

import cv2
import numpy as np


def to_alpha_at_size(
    mask: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Resize a uint8 mask to ``(target_h, target_w)`` with linear interp."""
    if mask.shape != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(mask, 0, 255).astype(np.uint8)


def save_alpha_sequence(masks: list, out_dir: str, prefix: str = "frame_") -> list:
    """Persist each mask as a PNG. Returns the list of written paths."""
    import os
    import pathlib
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for fn in os.listdir(d):
        if fn.startswith(prefix) and fn.endswith(".png"):
            try:
                os.unlink(d / fn)
            except OSError:
                pass
    paths = []
    for i, m in enumerate(masks):
        p = d / f"{prefix}{i:06d}.png"
        Image = __import__("PIL.Image", fromlist=["Image"]).fromarray
        Image(m, mode="L").save(str(p), format="PNG", optimize=True)
        paths.append(str(p))
    return paths
