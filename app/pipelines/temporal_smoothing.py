"""Stage 7: Temporal stabilisation.

A 3-tap weighted average across consecutive frames removes the
flicker that a per-frame matting model produces. This is much
cheaper than optical flow and works well for moderate motion.
"""
from __future__ import annotations

import numpy as np
import cv2


def _resize_to(a: np.ndarray, shape) -> np.ndarray:
    if a.shape == shape:
        return a
    return cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def temporal_smooth(
    masks: list,
    *,
    prev_weight: float = 0.2,
    next_weight: float = 0.2,
    current_weight: float = 0.6,
) -> list:
    """Return a new list of masks (uint8) with each frame blended
    against its neighbours. ``masks`` may contain numpy arrays of
    different shapes; each is resized to its target shape on the fly.
    """
    if not masks:
        return masks
    n = len(masks)
    out: list = []
    for i, m in enumerate(masks):
        cur = m.astype(np.float32)
        if i == 0 or i == n - 1:
            out.append(np.clip(cur, 0, 255).astype(np.uint8))
            continue
        prev = masks[i - 1].astype(np.float32)
        nxt = masks[i + 1].astype(np.float32)
        prev = _resize_to(prev, cur.shape)
        nxt = _resize_to(nxt, cur.shape)
        blended = (
            current_weight * cur
            + prev_weight * prev
            + next_weight * nxt
        )
        out.append(np.clip(blended, 0, 255).astype(np.uint8))
    return out
