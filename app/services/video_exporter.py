"""Stage 9: Video / alpha exporter.

Two outputs are produced by the hybrid pipeline:

1. ``alpha_sequence/<frame_xxxxxx.png>`` - one grayscale alpha PNG
   per frame, at the source resolution. Drop-in compatible with
   the existing preview / export / sprite code.
2. ``rgba.mov`` - a quicktime-compatible video with the alpha
   premultiplied into RGBA, suitable for editors that want a
   single file.
"""
from __future__ import annotations

import os
import subprocess
from typing import Sequence

import numpy as np


def write_alpha_sequence(masks: Sequence[np.ndarray], out_dir: str) -> list:
    """Write each mask as a grayscale PNG. Returns the list of paths."""
    import pathlib
    from PIL import Image
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, m in enumerate(masks):
        p = d / f"frame_{i:06d}.png"
        Image.fromarray(m, mode="L").save(str(p), format="PNG", optimize=True)
        paths.append(str(p))
    return paths


def write_rgba_video(
    frames_bgr: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    out_path: str,
    fps: float = 30.0,
) -> str:
    """Premultiply the alpha onto the frames and write an MP4.

    Returns the path to the written file. Falls back gracefully if
    ffmpeg is not available (in which case the file is not created
    and the function returns an empty string).
    """
    if shutil_which := __import__("shutil").which("ffmpeg"):
        pass
    else:
        return ""
    if len(frames_bgr) != len(masks) or not frames_bgr:
        return ""

    import cv2
    h, w = frames_bgr[0].shape[:2]
    # Build an rgba rawvideo stream for ffmpeg.
    rgba = []
    for frame, mask in zip(frames_bgr, masks):
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
        a = mask.astype(np.float32) / 255.0
        premul = (rgb * a[..., None] * 255.0 + 0.5).astype(np.uint8)
        a8 = mask
        rgba.append(np.concatenate([premul, a8[..., None]], axis=-1))

    raw_path = out_path + ".rgba"
    with open(raw_path, "wb") as fh:
        for frame in rgba:
            fh.write(frame.tobytes())

    cmd = [
        shutil_which,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgba",
        "-r", str(fps),
        "-i", raw_path,
        "-c:v", "qtrle",
        "-pix_fmt", "rgba",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except Exception:
        return ""
    finally:
        try:
            os.unlink(raw_path)
        except OSError:
            pass
    return out_path
