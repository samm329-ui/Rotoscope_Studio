"""Fast matte pipeline for Rotoscope Studio.

This pipeline is a thin wrapper around the provided FindMattes.py
helper. It scales the input frame down to a small height, invokes
the FCN model to produce a color-encoded matte, and saves the result.
"""
import os
import pathlib
from typing import Any

import app.config as _config
from app.pipelines.find_mattes import createMatte, getRotoModel


_model_loaded = False


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
    """Process every frame in the job's frames folder using the fast matte pipeline."""
    import os
    import pathlib
    frames_dir = _config.frames_dir(job_id)
    masks_dir = _config.masks_dir(job_id)
    masks_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale masks from previous runs.
    for fn in os.listdir(masks_dir):
        try:
            os.unlink(masks_dir / fn)
        except OSError:
            pass
    frames = sorted(frames_dir.glob('frame_*.png'))
    total = len(frames)
    for i, frame_path in enumerate(frames):
        out_path = masks_dir / frame_path.name
        process_frame(str(frame_path), str(out_path), _config.FAST_MATTE_MAX_SIZE)
        if progress_cb is not None:
            progress_cb(i, total)
    return total
