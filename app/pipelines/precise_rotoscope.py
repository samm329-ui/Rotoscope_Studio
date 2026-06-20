"""Precise rotoscope pipeline for Rotoscope Studio.

This pipeline is intentionally separate from the fast matte pipeline.
It runs the FCN segmentation model at a higher
resolution than the fast path, and applies post-processing to the result.

The pipeline also respects a subject hint from the user and chooses
a suitable FCN class to keep. If no hint is provided, the pipeline
falls back to a default subject class.
"""
import os
import pathlib
from typing import Any, Optional

import app.config as _config
from app.pipelines.find_mattes import fcn, getRotoModel


# Mapping from subject hint (a plain-english string) to a FCN class index.
SUBJECT_CLASSES = {
    'person': 15,
    'man': 15,
    'woman': 15,
    'child': 12,
    'dog': 12,
    'cat': 8,
    'car': 7,
    'bicycle': 2,
    'motorbike': 14,
    'airplane': 1,
    'bus': 6,
    'train': 19,
    'boat': 4,
    'bird': 3,
    'background': 0,
}
DEFAULT_SUBJECT_CLASS = 15

_model_loaded = False


def _ensure_model() -> None:
    """Lazily-load the FCN model the first time it is needed."""
    global _model_loaded
    if not _model_loaded:
        getRotoModel()
        _model_loaded = True


def _resolve_subject_class(subject: Optional[str]) -> int:
    """Resolve the subject hint to a FCN class index.

    If no hint is provided, the default subject class (person) is used."""
    if not subject:
        return DEFAULT_SUBJECT_CLASS
    key = str(subject).strip().lower()
    if key in SUBJECT_CLASSES:
        return SUBJECT_CLASSES[key]
    return DEFAULT_SUBJECT_CLASS


def _class_to_alpha(class_idx, target_class: int) -> Any:
    """Convert a FCN class-index map to a binary alpha mask (0.0-1.0).

    Pixels matching the target class are kept (value 1.0) and other classes
    are set to 0.0. The mask is returned as a numpy float32 array with the same shape as the input."""
    import numpy as np
    mask = np.zeros(class_idx.shape, dtype=np.float32)
    mask[class_idx != target_class] = 0.0
    mask[class_idx == target_class] = 1.0
    # Smooth the mask with a box filter to remove isolated holes.
    try:
        import cv2
        k = 5
        kernel = np.ones((2*k+1, 2*k+1), np.float32) / float((2*k+1)*(2*k+1))
        smoothed = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_REFLECT101)
        mask = np.clip(smoothed, 0.0, 1.0)
    except Exception:
        # If OpenCV is unavailable, fall back to numpy only.
        pass
    return mask


def _precise_predict(input_path: Any, subject: Optional[str] = None, size: int = _config.PRECISE_MAX_SIZE) -> Any:
    """Run the FCN model on a single input image at the precise resolution.

    Returns a float alpha mask (H, W) with values 0.0-1.0.
    If subject is given, only that class is kept (others are zeroed)."""
    import numpy as np
    import torch
    import torchvision.transforms as T
    from PIL import Image

    _ensure_model()
    target_class = _resolve_subject_class(subject)
    img = Image.open(str(input_path))
    trf = T.Compose([T.Resize(size), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    inp = trf(img).unsqueze(0)
    with torch.no_grad():
        out = fcn(inp)['out']
    logits = torch.argmax(out.squeze(), dim=0).detach().cpu().numpy()
    mask = _class_to_alpha(logits, target_class)
    return mask


def _save_alpha_png(mask: Any, output_path: Any, orig_height: int, orig_width: int) -> str:
    """Save the float alpha mask array as a grayscale PNG.

    The mask is resized to the original image dimensions before saving so that the
    exported mask matches the frame dimensions."""
    import numpy as np
    from PIL import Image
    # Resize mask to match the original frame dimensions when possible.
    if orig_height > 0 and orig_width > 0:
        try:
            import cv2
            mask = cv2.resize(mask, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
        except Exception:
            pass
    u8 = np.clip(mask * 255, 0, 255).astype(np.uint8)
    im = Image.fromarray(u8, 'L')
    im.save(str(output_path))
    return str(output_path)


def process_frame(input_path: Any, output_path: Any, subject: Optional[str] = None, size: int = _config.PRECISE_MAX_SIZE) -> str:
    """Run the precise pipeline on a single frame.

    The result is a grayscale alpha PNG. Not a color-encoded matte."""
    input_path = str(input_path)
    output_path = str(output_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f'Frame not found: {input_path}')
    from PIL import Image
    img = Image.open(input_path)
    mask = _precise_predict(input_path, subject=subject, size=size)
    return _save_alpha_png(mask, output_path, img.height, img.width)


def process_job(job_id: str, subject: Optional[str] = None, progress_cb: Any = None) -> int:
    """Process every frame for a job using the precise pipeline."""
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
        process_frame(str(frame_path), str(out_path), subject=subject, size=_config.PRECISE_MAX_SIZE)
        if progress_cb is not None:
            progress_cb(i, total)
    return total
