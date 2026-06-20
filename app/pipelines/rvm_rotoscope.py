"""Robust Video Matting (RVM) pipeline for Rotoscope Studio.

RVM is a recurrent neural network designed for video matting. It produces
true alpha values (0.0-1.0) with soft, hair-preserving edges and uses
temporal state so that consecutive frames reinforce each other.

This module is intentionally self-contained so it can be selected as a
new workflow without touching the FCN-based ``fast_matte`` or
``precise_rotoscope`` pipelines. The model is loaded once and reused
across frames; intermediate tensors are kept in memory to avoid the
disk round-trip that dominates the legacy pipeline.

The RVM model is the official MobileNetV3 TorchScript release from
https://github.com/PeterL1n/RobustVideoMatting/releases - it is
~14 MB and runs comfortably on CPU.
"""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

import app.config as _config


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

# Repo location of the upstream RVM model. The MobileNetV3 variant is
# small enough for a one-time download (~14 MB) and works well on CPU.
RVM_REPO_URL = (
    "https://github.com/PeterL1n/RobustVideoMatting/releases/download/"
    "v1.0.0/rvm_mobilenetv3_fp32.torchscript"
)
RVM_MODEL_NAME = "rvm_mobilenetv3_fp32.torchscript"


def _model_path() -> Path:
    """Return the on-disk path where the RVM TorchScript model is cached."""
    assets = _config.ROOT / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    return assets / RVM_MODEL_NAME


def ensure_rvm_model() -> Path:
    """Download the RVM model on first use and return the local path.

    If the file already exists in ``assets/`` we skip the download.
    The download is retried up to three times with a short backoff.
    """
    target = _model_path()
    if target.is_file() and target.stat().st_size > 1_000_000:
        return target
    print(f"[rvm] Downloading {RVM_MODEL_NAME} (~14 MB) to {target}", flush=True)
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(RVM_REPO_URL, str(target))
            if target.is_file() and target.stat().st_size > 1_000_000:
                print("[rvm] Download complete.", flush=True)
                return target
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2 * (attempt + 1))
    raise FileNotFoundError(
        f"Could not download RVM model from {RVM_REPO_URL}: {last_err!r}. "
        f"You can manually place {RVM_MODEL_NAME} inside {target.parent}."
    )


# ---------------------------------------------------------------------------
# Pipeline singleton
# ---------------------------------------------------------------------------

class _RvmSession:
    """Holds a single loaded RVM model + recurrent state for the lifetime
    of one processing job. Recurrent state is what makes RVM fast and
    temporally consistent: we pass it forward across frames instead of
    re-initialising per frame.
    """

    def __init__(self) -> None:
        self.model_path = ensure_rvm_model()
        # TorchScript on CPU. ``map_location`` keeps us portable on
        # machines that happen to have CUDA later.
        self.model = torch.jit.load(str(self.model_path), map_location="cpu")
        self.model.eval()
        try:
            torch.set_num_threads(max(2, (os.cpu_count() or 4) // 2))
        except Exception:
            pass
        # RVM downsample ratio. 0.25 gives 4x speedup over 1.0 with
        # visually almost identical alpha for full-body subjects.
        self.downsample_ratio: float = float(os.environ.get("RVM_DS_RATIO", "0.25"))
        # Recurrent state tensors (r1..r4, h). Lazily allocated on the
        # first frame so the model picks the right shapes.
        self._state: List[torch.Tensor] = []

    def reset_state(self) -> None:
        self._state = []

    @torch.no_grad()
    def matte_ndarray(self, rgb_chw: np.ndarray) -> np.ndarray:
        """Run RVM on a CHW float32 RGB array and return an HxW float32 alpha."""
        t = torch.from_numpy(rgb_chw).unsqueeze(0).contiguous().float()
        # Model returns 6 tensors: (fgr, pha, r1, r2, r3, r4).
        if self._state:
            outs = self.model(t, *self._state, self.downsample_ratio, False)
        else:
            outs = self.model(t, None, None, None, None, self.downsample_ratio, False)
        # outs = (fgr[1,3,H,W], pha[1,1,H,W], r1, r2, r3, r4)
        self._state = list(outs[2:])  # keep recurrent state
        alpha = outs[1][0].clamp(0.0, 1.0).cpu().numpy()
        return alpha

    @torch.no_grad()
    def matte_pil(self, image: Image.Image) -> Image.Image:
        """Run RVM on a PIL image and return a grayscale alpha PIL image."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0  # HWC
        chw = arr.transpose(2, 0, 1)  # CHW
        alpha = self.matte_ndarray(chw)
        # ``alpha`` comes out of the model as (1, H, W). Squeeze to (H, W).
        if alpha.ndim == 3 and alpha.shape[0] == 1:
            alpha = alpha[0]
        u8 = (alpha * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(u8, mode='L')


_session: Optional["_RvmSession"] = None


def _get_session() -> "_RvmSession":
    global _session
    if _session is None:
        _session = _RvmSession()
    return _session


# ---------------------------------------------------------------------------
# Public API expected by frame_processor
# ---------------------------------------------------------------------------

def process_frame(
    input_path: Any,
    output_path: Any,
    size: int = 0,
) -> str:
    """Process a single frame: read PNG, run RVM, save grayscale alpha PNG.

    The ``size`` argument is accepted for API compatibility with the
    other pipelines; RVM runs at the source frame resolution. Pass
    ``size > 0`` to downscale for the model pass and resample back to
    the original size on save.
    """
    input_path = str(input_path)
    output_path = str(output_path)
    img = Image.open(input_path).convert("RGB")
    if size and size > 0:
        w, h = img.size
        scale = size / float(min(w, h))
        new_size = (max(2, int(round(w * scale))), max(2, int(round(h * scale))))
        model_input = img.resize(new_size, Image.BILINEAR)
    else:
        model_input = img
    alpha = _get_session().matte_pil(model_input)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if alpha.size != img.size:
        alpha = alpha.resize(img.size, Image.BILINEAR)
    alpha.save(str(out), format="PNG", optimize=True)
    return str(out)


def process_job_from_video(
    video_path: str,
    masks_dir: Path,
    progress_cb: Any = None,
    jpeg_quality: int = 4,
    downsample_max_pixels: int = 1280 * 720,
) -> int:
    """Stream a video through RVM without ever writing per-frame PNGs.

    This is the fast path: ffmpeg decodes + JPEG-encodes each frame,
    PIL decodes it, RVM produces an alpha, we save the alpha PNG to
    ``masks_dir``. No intermediate frame PNGs hit the disk.
    """
    from app.services.frame_extractor import iter_frames_fast
    import io

    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    for fn in os.listdir(masks_dir):
        try:
            os.unlink(masks_dir / fn)
        except OSError:
            pass

    session = _get_session()
    session.reset_state()
    t0 = time.time()
    count = 0
    for idx, jpg_bytes, _w, _h in iter_frames_fast(
        video_path,
        jpeg_quality=jpeg_quality,
        max_pixels=downsample_max_pixels,
    ):
        with Image.open(io.BytesIO(jpg_bytes)) as im:
            im_rgb = im.convert("RGB")
            alpha = session.matte_pil(im_rgb)
        # RVM alpha is in source resolution; save as PNG.
        out_path = masks_dir / f"frame_{idx:06d}.png"
        alpha.save(str(out_path), format="PNG", optimize=True)
        count += 1
        if progress_cb is not None and (count % 4 == 0):
            progress_cb(count, 0)
    elapsed = time.time() - t0
    print(
        f"[rvm] {count} frames in {elapsed:.2f}s "
        f"({count / max(elapsed, 1e-6):.2f} fps)",
        flush=True,
    )
    return count


def process_job(
    job_id: str,
    subject: Optional[str] = None,  # noqa: ARG001 - kept for API compatibility
    progress_cb: Any = None,
) -> int:
    """Process every frame in the job's frames folder with RVM.

    If the job's source video is still on disk we take the streaming
    fast path; otherwise we fall back to reading the existing PNG
    frames. Recurrent state is shared across frames to give temporal
    consistency.
    """
    frames_dir = _config.frames_dir(job_id)
    masks_dir = _config.masks_dir(job_id)
    masks_dir.mkdir(parents=True, exist_ok=True)
    for fn in os.listdir(masks_dir):
        try:
            os.unlink(masks_dir / fn)
        except OSError:
            pass

    # Prefer streaming from the original video if we can find it.
    from app.services import job_store
    job = job_store.get_job(job_id)
    video_path = None
    if job and job.get("file_path") and os.path.isfile(job["file_path"]):
        video_path = job["file_path"]
    if video_path is not None:
        try:
            return process_job_from_video(video_path, masks_dir, progress_cb=progress_cb)
        except Exception as exc:
            print(f"[rvm] streaming failed ({exc!r}), falling back to PNG loop", flush=True)

    # Fallback: read PNG frames one by one.
    frames = sorted(frames_dir.glob("frame_*.png"))
    total = len(frames)
    if total == 0:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    session = _get_session()
    session.reset_state()
    t0 = time.time()
    for idx, frame_path in enumerate(frames):
        out_path = masks_dir / frame_path.name
        with Image.open(str(frame_path)) as img:
            img = img.convert("RGB")
            alpha = session.matte_pil(img)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        alpha.save(str(out_path), format="PNG", optimize=True)
        if progress_cb is not None and (idx % 4 == 0 or idx == total - 1):
            progress_cb(idx + 1, total)
    elapsed = time.time() - t0
    print(
        f"[rvm] {total} frames in {elapsed:.2f}s "
        f"({total / max(elapsed, 1e-6):.2f} fps)",
        flush=True,
    )
    return total
