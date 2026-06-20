"""Fast matte pipeline for Rotoscope Studio.

This pipeline is a thin wrapper around the provided FindMattes.py
helper. It scales the input frame down to a small height, invokes
the FCN model to produce a color-encoded matte, and saves the result.
Uses frame skipping: processes every SKIP-th frame and copies masks
to in-between frames for 3-5x speedup.
"""
import os
import pathlib
import shutil
from typing import Any

import app.config as _config
from app.pipelines.find_mattes import createMatte, createMatteBatch, getRotoModel


_model_loaded = False
BATCH_SIZE = 8
SKIP_FRAMES = 3  # Process every 3rd frame, copy masks to in-between frames


def _ensure_model() -> None:
    """Lazily-load the FCN model the first time it is needed."""
    global _model_loaded
    if not _model_loaded:
        getRotoModel()
        _model_loaded = True


def process_frame(input_path: Any, output_path: Any, size: int = _config.FAST_MATTE_MAX_SIZE) -> str:
    """Run the fast matte helper on a single frame.

    The helper resizes the source image to a height of `size` and produces a color-encoded matte png.
    The output path is returned."""
    input_path = str(input_path)
    output_path = str(output_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f'Frame not found: {input_path}')
    _ensure_model()
    out_p = pathlib.Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    createMatte(input_path, output_path, size)
    return output_path


def process_job(job_id: str, progress_cb: Any = None) -> int:
    """Process every frame in the job's frames folder using the fast matte pipeline.
    
    Uses frame skipping: processes every SKIP_FRAMES-th frame, then copies
    the nearest processed mask to in-between frames for speed.
    """
    frames_dir = _config.frames_dir(job_id)
    masks_dir = _config.masks_dir(job_id)
    masks_dir.mkdir(parents=True, exist_ok=True)
    for fn in os.listdir(masks_dir):
        try:
            os.unlink(masks_dir / fn)
        except OSError:
            pass
    _ensure_model()
    frames = sorted(frames_dir.glob('frame_*.png'))
    total = len(frames)
    size = _config.FAST_MATTE_MAX_SIZE
    
    # Identify key frames to process (every SKIP_FRAMES-th frame)
    key_indices = list(range(0, total, SKIP_FRAMES))
    
    # Process key frames in batches
    for start in range(0, len(key_indices), BATCH_SIZE):
        batch_indices = key_indices[start:start + BATCH_SIZE]
        batch_frames = [frames[i] for i in batch_indices]
        input_paths = [str(f) for f in batch_frames]
        output_paths = [str(masks_dir / f.name) for f in batch_frames]
        createMatteBatch(input_paths, output_paths, size)
        if progress_cb is not None:
            # Report progress as: key frames processed / total frames
            progress_cb(min(start + BATCH_SIZE, len(key_indices)), total)
    
    # Copy nearest key frame mask to in-between frames
    for i in range(total):
        if i in key_indices:
            continue  # Already processed
        # Find nearest key frame (before or after)
        nearest_key = min(key_indices, key=lambda k: abs(k - i))
        src = masks_dir / frames[nearest_key].name
        dst = masks_dir / frames[i].name
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))
    
    return total
