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
from typing import Any, Optional, List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import app.config as _config
from app.pipelines.FindMattes import getRotoModel as _getRotoModel
import app.pipelines.FindMattes as _FindMattes


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
BATCH_SIZE = 8

_model_loaded = False
_trf_cache = {}


def _ensure_model() -> None:
    global _model_loaded
    if not _model_loaded:
        _getRotoModel()
        _model_loaded = True


def _get_transform(size):
    if size not in _trf_cache:
        _trf_cache[size] = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
    return _trf_cache[size]


def _resolve_subject_class(subject: Optional[str]) -> int:
    if not subject:
        return DEFAULT_SUBJECT_CLASS
    key = str(subject).strip().lower()
    return SUBJECT_CLASSES.get(key, DEFAULT_SUBJECT_CLASS)


def _class_to_alpha(class_idx, target_class: int):
    mask = (class_idx == target_class).astype(np.float32)
    try:
        import cv2
        k = 5
        kernel = np.ones((2*k+1, 2*k+1), np.float32) / float((2*k+1)*(2*k+1))
        smoothed = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_REFLECT101)
        mask = np.clip(smoothed, 0.0, 1.0)
    except Exception:
        pass
    return mask


def _save_alpha_png(mask, output_path, orig_height, orig_width):
    if orig_height > 0 and orig_width > 0:
        try:
            import cv2
            mask = cv2.resize(mask, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
        except Exception:
            pass
    u8 = np.clip(mask * 255, 0, 255).astype(np.uint8)
    Image.fromarray(u8, 'L').save(str(output_path))
    return str(output_path)


def process_frame(input_path, output_path, subject=None, size=_config.PRECISE_MAX_SIZE):
    input_path = str(input_path)
    output_path = str(output_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f'Frame not found: {input_path}')
    _ensure_model()
    target_class = _resolve_subject_class(subject)
    img = Image.open(input_path).convert('RGB')
    trf = _get_transform(size)
    inp = trf(img).unsqueeze(0)
    with torch.no_grad():
        out = _FindMattes.fcn(inp)['out']
    logits = torch.argmax(out.squeeze(), dim=0).cpu().numpy()
    mask = _class_to_alpha(logits, target_class)
    return _save_alpha_png(mask, output_path, img.height, img.width)


def process_frame_batch(input_paths, output_paths, subject=None, size=_config.PRECISE_MAX_SIZE):
    _ensure_model()
    target_class = _resolve_subject_class(subject)
    trf = _get_transform(size)
    batch = []
    originals = []
    for p in input_paths:
        img = Image.open(str(p)).convert('RGB')
        originals.append((img.height, img.width))
        batch.append(trf(img))
    inp = torch.stack(batch, dim=0)
    with torch.no_grad():
        out = _FindMattes.fcn(inp)['out']
    preds = torch.argmax(out, dim=1).cpu().numpy()
    for i in range(len(input_paths)):
        mask = _class_to_alpha(preds[i], target_class)
        _save_alpha_png(mask, output_paths[i], originals[i][0], originals[i][1])


def process_job(job_id, subject=None, progress_cb=None):
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
    size = _config.PRECISE_MAX_SIZE
    for start in range(0, total, BATCH_SIZE):
        batch_frames = frames[start:start + BATCH_SIZE]
        input_paths = [str(f) for f in batch_frames]
        output_paths = [str(masks_dir / f.name) for f in batch_frames]
        process_frame_batch(input_paths, output_paths, subject=subject, size=size)
        if progress_cb is not None:
            progress_cb(min(start + BATCH_SIZE, total), total)
    return total
