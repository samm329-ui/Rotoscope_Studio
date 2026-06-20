"""Preview builder service for Rotoscope Studio.

Builds a side-by-side sprite sheet preview from a few sample
frames. The preview combines the original frame and the mask so that
the user can see what was kept and what was removed.
"""
import os
import pathlib
from typing import Any, Dict, List

import app.config as _config
from app.services import job_store


PREVIEW_SAMPLE_COUNT = 6
PREVIEW_TILE = 'preview_sprite.png'
THUMB_HEIGHT = 120
THUMB_WIDTH = 160


def _sample_frames(frames_dir: pathlib.Path, count: int = PREVIEW_SAMPLE_COUNT) -> List:
    """Pick a small set of frames evenly spaced through the video."""
    frames = sorted(frames_dir.glob('frame_*.png'))
    if not frames:
        return []
    step = max(1, len(frames) // count)
    return frames[::step][:count]


def build_preview(job_id: str, workflow: str = 'fast_matte') -> List:
    """Build a side-by-side sprite sheet preview.

    Returns a list of paths to generated preview assets. If no frames exist
    the function returns a empty list."""
    from PIL import Image

    job = job_store.get_job(job_id)
    frames_dir = _config.frames_dir(job_id)
    masks_dir = _config.masks_dir(job_id)
    previews_dir = _config.previews_dir(job_id)
    previews_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale preview assets from previous runs.
    for fn in os.listdir(previews_dir):
        try:
            os.unlink(previews_dir / fn)
        except OSError:
            pass
    frames = _sample_frames(frames_dir, PREVIEW_SAMPLE_COUNT)
    if not frames:
        return []
    combined_thumbs: List = []
    for i, frame_path in enumerate(frames):
        mask_path = masks_dir / frame_path.name
        thumb_img = Image.open(frame_path).resize((THUMB_WIDTH, THUMB_HEIGHT))
        thumb_img = thumb_img.convert('RGB')
        if mask_path.is_file():
            mask_img = Image.open(mask_path)
            if mask_img.mode != 'L':
                mask_img = mask_img.convert('L')
            mask_img = mask_img.resize((THUMB_WIDTH, THUMB_HEIGHT))
            combo = Image.new('RGB',(THUMB_WIDTH * 2, THUMB_HEIGHT),(220, 220, 220))
            combo.paste(thumb_img, (0, 0))
            combo.paste(mask_img, (THUMB_WIDTH, 0))
            pair_path = previews_dir / f'pair_{i:04d}.png'
            combo.save(str(pair_path))
            combined_thumbs.append(combo)
        else:
            combo = Image.new('RGB',(THUMB_WIDTH, THUMB_HEIGHT),(220, 220, 220))
            combo.paste(thumb_img, (0, 0))
            pair_path = previews_dir / f'pair_{i:04d}.png'
            combo.save(str(pair_path))
            combined_thumbs.append(combo)
    if not combined_thumbs:
        return []
    # Build a sprite sheet with all samples stacked horizontally.
    total_width = sum(t.size[0] for t in combined_thumbs)
    max_height = max((t.size[1] for t in combined_thumbs))
    sprite = Image.new('RGB', (total_width, max_height), (225, 225, 225))
    x_offset = 0

    for t in combined_thumbs:
        sprite.paste(t, (x_offset, 0))
        x_offset += t.size[0]

    sprite_path = previews_dir / PREVIEW_TILE
    sprite.save(str(sprite_path))
    generated = [str(p) for p in combined_thumbs]
    generated.append(str(sprite_path))
    if job is not None:
        job_store.update_job(job_id, current_step='preview_ready')
    return generated


def list_previews(job_id: str) -> List:
    """Return a list of paths to existing preview assets for a job."""
    previews_dir = _config.previews_dir(job_id)
    if not previews_dir.is_dir():
        return []
    return [str(previews_dir / fn) for fn in sorted(os.listdir(previews_dir)) if fn.lower().endswith(('.png', '.jpg', '.jpeg'))]


def preview_summary(job_id: str) -> Dict:
    """Return a short summary of the preview assets for the job."""
    previews = list_previews(job_id)
    return {'count': len(previews), 'assets': previews}
