"""SAM2 model loader.

The pipeline is designed to use the smallest viable SAM2 checkpoint
(``sam2_hiera_tiny``) so that even CPU-only machines can run video
propagation. The model file is downloaded on first use and cached in
``<project>/assets/``.

Public entry points:
    - ``get_sam2_video_predictor()``: returns a configured
      ``SAM2VideoPredictor`` instance.
    - ``get_sam2_image_predictor()``: returns a configured
      ``SAM2ImagePredictor`` (used for the first-frame prompt).
    - ``ensure_sam2_checkpoint()``: downloads the checkpoint if missing.
"""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from typing import Optional

import app.config as _config


SAM2_REPO_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
    "sam2_hiera_tiny.pt"
)
SAM2_CONFIG_NAME = "configs/sam2_hiera_t.yaml"
SAM2_CHECKPOINT_NAME = "sam2_hiera_tiny.pt"


def _checkpoint_path() -> Path:
    assets = _config.ROOT / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    return assets / SAM2_CHECKPOINT_NAME


def ensure_sam2_checkpoint() -> Path:
    """Download the SAM2 tiny checkpoint on first use; return the path."""
    target = _checkpoint_path()
    if target.is_file() and target.stat().st_size > 50_000_000:
        return target
    print(f"[sam2] Downloading {SAM2_CHECKPOINT_NAME} (~156 MB) to {target}", flush=True)
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(SAM2_REPO_URL, str(target))
            if target.is_file() and target.stat().st_size > 50_000_000:
                print("[sam2] Download complete.", flush=True)
                return target
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2 * (attempt + 1))
    raise FileNotFoundError(
        f"Could not download SAM2 checkpoint from {SAM2_REPO_URL}: {last_err!r}. "
        f"You can manually place {SAM2_CHECKPOINT_NAME} inside {target.parent}."
    )


# Lazy singletons.
_video_predictor = None
_image_predictor = None


def get_sam2_video_predictor():
    """Return a configured SAM2VideoPredictor (singleton)."""
    global _video_predictor
    if _video_predictor is not None:
        return _video_predictor
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore
    ckpt = ensure_sam2_checkpoint()
    # The installed sam2 package ships configs/ inside the module dir.
    import sam2 as _sam2_pkg
    config_path = os.path.join(os.path.dirname(_sam2_pkg.__file__), "sam2_hiera_t.yaml")
    # torch.set_num_threads is set globally in the pipeline entry; we don't override here.
    _video_predictor = build_sam2_video_predictor(config_path, str(ckpt), device="cpu")
    return _video_predictor


def get_sam2_image_predictor():
    """Return a configured SAM2ImagePredictor (singleton)."""
    global _image_predictor
    if _image_predictor is not None:
        return _image_predictor
    from sam2.build_sam import build_sam2_image_predictor  # type: ignore
    ckpt = ensure_sam2_checkpoint()
    import sam2 as _sam2_pkg
    config_path = os.path.join(os.path.dirname(_sam2_pkg.__file__), "sam2_hiera_t.yaml")
    _image_predictor = build_sam2_image_predictor(config_path, str(ckpt), device="cpu")
    return _image_predictor
