"""Stage 6: Boundary fusion.

For each refined frame we blend the BiRefNet alpha with the SAM2
mask. The weight is confidence-driven: when SAM2 is uncertain we
trust BiRefNet more; when SAM2 is confident we keep its structure
and only borrow edges from BiRefNet.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def fuse_mask(
    sam_mask_u8: np.ndarray,
    biref_alpha_pil: Image.Image,
    *,
    sam_confidence: float,
) -> np.ndarray:
    """Return a fused ``(H, W)`` uint8 mask.

    Parameters
    ----------
    sam_mask_u8 : np.ndarray
        ``(H, W)`` uint8 (0/255) mask from SAM2.
    biref_alpha_pil : PIL.Image.Image
        ``(H, W)`` grayscale alpha from BiRefNet (or fallback).
    sam_confidence : float
        Confidence proxy in [0, 1] from SAM2.
    """
    biref = np.asarray(biref_alpha_pil.convert("L"), dtype=np.float32)
    if biref.shape != sam_mask_u8.shape:
        biref = np.asarray(
            biref_alpha_pil.convert("L").resize(
                (sam_mask_u8.shape[1], sam_mask_u8.shape[0]),
                Image.BILINEAR,
            ),
            dtype=np.float32,
        )
    sam = sam_mask_u8.astype(np.float32)
    # Blend weight: 0.3 + 0.4 * (1 - confidence). When SAM2 is unsure
    # we lean more on BiRefNet, when SAM2 is confident we mostly
    # preserve its structure and only borrow BiRefNet's edges.
    w = 0.3 + 0.4 * (1.0 - float(sam_confidence))
    fused = (1.0 - w) * sam + w * biref
    return np.clip(fused, 0, 255).astype(np.uint8)
