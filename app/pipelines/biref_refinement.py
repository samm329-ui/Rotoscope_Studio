"""Stage 5: BiRefNet refinement.

BiRefNet is a heavy matting model. We only run it on the
``refinement_queue`` selected by ``frame_selector``. The remaining
frames keep their SAM2 masks (still good enough after temporal
smoothing).
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np
from PIL import Image


def refine_with_birefnet(
    frames_bgr: Sequence[np.ndarray],
    indices: Sequence[int],
    *,
    target_long_side: int = 384,
) -> dict:
    """Return ``{frame_idx: alpha_pil}`` for the requested frames.

    The alpha is produced at the source resolution so the downstream
    stages can use it as a drop-in replacement for the SAM2 mask.
    """
    if not indices:
        return {}

    from app.models.birefnet_model import get_birefnet_session
    session = get_birefnet_session()

    out: dict = {}
    t0 = time.time()
    for i, idx in enumerate(indices):
        frame = frames_bgr[idx]
        rgb = frame[:, :, ::-1]
        pil = Image.fromarray(rgb, mode="RGB")
        alpha = session.matte_pil(pil, target_long_side=target_long_side)
        out[idx] = alpha
    elapsed = time.time() - t0
    print(
        f"[birefnet] {len(indices)} frames refined in {elapsed:.2f}s "
        f"({len(indices) / max(elapsed, 1e-6):.2f} fps, long_side={target_long_side})",
        flush=True,
    )
    return out
