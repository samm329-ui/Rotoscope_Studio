"""Stage 4: Confidence analysis and frame selection.

Decides which frames need BiRefNet refinement based on:
  - low SAM2 confidence
  - large confidence drops vs neighbouring frames (motion / occlusion)
  - edge density (hair-heavy heuristic)
  - spatial disagreement with neighbour (boundary-failure heuristic)

Outputs a ``refinement_queue`` of frame indices.
"""
from __future__ import annotations

from typing import List, Sequence

import cv2
import numpy as np


def _edge_density(mask: np.ndarray) -> float:
    """Approximate boundary complexity using a Laplacian on the mask."""
    if mask.size == 0:
        return 0.0
    f = mask.astype(np.float32) / 255.0
    lap = cv2.Laplacian(f, cv2.CV_32F)
    return float(np.mean(np.abs(lap)))


def _neighbour_disagreement(mask: np.ndarray, neighbours: Sequence[np.ndarray]) -> float:
    if not neighbours:
        return 0.0
    diffs = []
    for n in neighbours:
        a = mask.astype(np.float32) / 255.0
        b = n.astype(np.float32) / 255.0
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
        diffs.append(float(np.mean(np.abs(a - b))))
    return float(np.mean(diffs))


def select_refinement_frames(
    masks: Sequence[np.ndarray],
    confidences: Sequence[float],
    *,
    confidence_threshold: float = 0.55,
    edge_threshold: float = 0.10,
    disagreement_threshold: float = 0.10,
    max_fraction: float = 0.25,
    min_spacing: int = 2,
) -> List[int]:
    """Return the list of frame indices to send to BiRefNet.

    The set is capped at ``max_fraction`` of the clip length and
    de-duplicated to ensure a minimum spacing between refinements.
    """
    n = len(masks)
    if n == 0:
        return []

    # Pre-compute per-frame scores.
    scores = []
    for i, (m, c) in enumerate(zip(masks, confidences)):
        edge = _edge_density(m)
        prev = masks[i - 1] if i > 0 else None
        nxt = masks[i + 1] if i < n - 1 else None
        neigh = [x for x in (prev, nxt) if x is not None]
        dis = _neighbour_disagreement(m, neigh)
        scores.append({
            "low_conf": c < confidence_threshold,
            "edge": edge > edge_threshold,
            "disagreement": dis > disagreement_threshold,
            "confidence": c,
        })

    # Build candidate set: any frame that is low-confidence, edge-heavy,
    # or disagrees with neighbours. We always also include the first and
    # last frame for temporal stability at the boundaries.
    candidates: List[int] = []
    for i, s in enumerate(scores):
        if s["low_conf"] or s["edge"] or s["disagreement"]:
            candidates.append(i)
    if 0 not in candidates:
        candidates.append(0)
    if n - 1 not in candidates:
        candidates.append(n - 1)

    # Apply max-fraction cap.
    cap = max(2, int(round(max_fraction * n)))
    if len(candidates) > cap:
        # Keep the top-``cap`` candidates ranked by lowest confidence.
        ranked = sorted(
            candidates,
            key=lambda i: (
                scores[i]["confidence"],
                -i,
            ),
        )
        candidates = sorted(ranked[:cap])

    # Enforce minimum spacing.
    candidates.sort()
    spaced: List[int] = []
    last = -10 ** 9
    for c in candidates:
        if c - last >= min_spacing:
            spaced.append(c)
            last = c
    return spaced
