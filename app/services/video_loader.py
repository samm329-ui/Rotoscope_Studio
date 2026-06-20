"""Stage 0/1: Video loader.

Reads a video file into a list of ``BGR`` uint8 numpy frames. Keeps
the entire clip in memory; the downstream stages consume the list
in order. For very long clips (>= 600 frames at 720p) we cap the
returned list at ``max_frames`` and inform the caller via the
``truncated`` flag in the returned dict.
"""
from __future__ import annotations

import os
from typing import List, Sequence

import cv2
import numpy as np


def load_video(
    video_path: str,
    *,
    max_frames: int = 600,
    max_long_side: int = 1280,
) -> dict:
    """Load a video into memory as a list of BGR uint8 frames.

    Returns a dict with keys:
        - ``frames`` : list of ``(H, W, 3)`` uint8 BGR frames.
        - ``fps``    : float, source fps.
        - ``size``   : ``(H, W)`` of the (possibly downscaled) frames.
        - ``orig_size`` : ``(H, W)`` of the source frames.
        - ``truncated`` : bool, whether the clip was capped.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"OpenCV cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: List[np.ndarray] = []
    truncated = False
    try:
        while True:
            if len(frames) >= max_frames:
                truncated = True
                break
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        return {"frames": [], "fps": fps, "size": (0, 0), "orig_size": (src_h, src_w), "truncated": truncated}

    # Optional downscale for huge source videos.
    h, w = frames[0].shape[:2]
    if max(h, w) > max_long_side:
        scale = max_long_side / float(max(h, w))
        new_w = max(2, int(round(w * scale)) // 2 * 2)
        new_h = max(2, int(round(h * scale)) // 2 * 2)
        resized = []
        for f in frames:
            r = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA)
            resized.append(r)
        frames = resized
    return {
        "frames": frames,
        "fps": float(fps),
        "size": (frames[0].shape[0], frames[0].shape[1]),
        "orig_size": (src_h, src_w),
        "truncated": truncated,
    }
