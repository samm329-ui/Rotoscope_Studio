"""Stage 2 + 3: SAM2 video propagation.

The first-frame prompt can be supplied three ways:

  1. ``point_prompt=(x, y)`` - a single positive click from the UI.
     Works for any subject (person, animal, cartoon character, prop).
  2. ``box_prompt=(x0, y0, x1, y1)`` - a bounding box. Useful for
     very small subjects or when the user wants to disambiguate.
  3. None - run a saliency-based auto-pick on the first frame and
     use the largest connected non-edge region as the subject.

SAM2 then propagates the prompt across every frame and we collect a
binary mask + a confidence score (the fraction of pixels marked as
foreground) for each frame.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb, mode="RGB")


def _empty_mask(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _ensure_jpeg_sequence(
    frames_bgr: Sequence[np.ndarray],
    work_dir: str,
    long_side: int = 480,
) -> Tuple[List[str], int, int]:
    """Write frames as JPEGs into ``work_dir`` for the SAM2 video
    predictor. Returns ``(frame_names, H, W)``.
    """
    d_local = __import__("pathlib").Path(work_dir)
    if d_local.exists():
        for fn in os.listdir(d_local):
            try:
                os.unlink(d_local / fn)
            except OSError:
                pass
    d_local.mkdir(parents=True, exist_ok=True)
    first = frames_bgr[0]
    h, w = first.shape[:2]
    if max(h, w) > long_side:
        scale = long_side / float(max(h, w))
        new_w = max(2, int(round(w * scale)) // 2 * 2)
        new_h = max(2, int(round(h * scale)) // 2 * 2)
    else:
        new_w, new_h = w, h
    names: List[str] = []
    for i, frame in enumerate(frames_bgr):
        pil = _bgr_to_pil(frame)
        if (pil.size[0], pil.size[1]) != (new_w, new_h):
            pil = pil.resize((new_w, new_h), Image.BILINEAR)
        name = f"{i:06d}.jpg"
        pil.save(str(d_local / name), format="JPEG", quality=85)
        names.append(name)
    return names, new_h, new_w


def _saliency_auto_box(first_bgr: np.ndarray) -> np.ndarray:
    """Pick a bounding box for the most salient subject in the first frame.

    Strategy:
      1. Convert to grayscale and run spectral residual saliency.
      2. Threshold the saliency map and find the largest contour.
      3. If the largest contour is too small (< 1% of the frame),
         fall back to a centred inner-70% box. This handles very
         flat scenes where saliency gives no signal.
    """
    h, w = first_bgr.shape[:2]
    try:
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, smap = saliency.computeSaliency(first_bgr)
    except Exception:
        ok = False
        smap = None
    box = None
    if ok and smap is not None:
        smap_u8 = (np.clip(smap, 0, 1) * 255).astype(np.uint8)
        # Smooth a little before thresholding.
        smap_blur = cv2.GaussianBlur(smap_u8, (5, 5), 0)
        _, thr = cv2.threshold(smap_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Drop the outer 4% to avoid edge-of-frame noise.
        border = max(2, int(0.04 * min(h, w)))
        thr[:border, :] = 0
        thr[-border:, :] = 0
        thr[:, :border] = 0
        thr[:, -border:] = 0
        # Open with a small kernel to remove specks.
        kernel = np.ones((5, 5), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            best = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(best)
            # Pad the box a little.
            pad = int(0.05 * max(bw, bh))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + bw + pad)
            y1 = min(h, y + bh + pad)
            # Reject tiny detections.
            if (x1 - x0) * (y1 - y0) > 0.01 * h * w:
                box = np.array([x0, y0, x1, y1], dtype=np.float32)
    if box is None:
        # Fall back to a centred inner-70% box.
        x0 = int(w * 0.15)
        y0 = int(h * 0.15)
        x1 = int(w * 0.85)
        y1 = int(h * 0.85)
        box = np.array([x0, y0, x1, y1], dtype=np.float32)
    return box


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def track_with_sam2(
    frames_bgr: Sequence[np.ndarray],
    *,
    work_dir: str,
    point_prompt: Optional[Tuple[float, float]] = None,
    box_prompt: Optional[Tuple[float, float, float, float]] = None,
    subject_hint: Optional[str] = None,
    long_side: int = 480,
) -> Dict[str, Any]:
    """Run SAM2 video propagation and return per-frame masks.

    Parameters
    ----------
    frames_bgr : list of ndarray
        Source frames as BGR uint8, at the source resolution.
    work_dir : str
        Scratch directory for SAM2's JPEG sequence.
    point_prompt : (x, y), optional
        A single positive click in source pixel coordinates. The
        first frame is downscaled for the model; the prompt is
        rescaled accordingly before being passed to SAM2.
    box_prompt : (x0, y0, x1, y1), optional
        A bounding box in source pixel coordinates. Used if no
        point prompt is given.
    subject_hint : str, optional
        Currently logged but unused. Reserved for future filtering
        (e.g. "person" -> only accept VOC person class).
    long_side : int
        The longest side the model sees.

    Returns
    -------
    dict with keys:
        - ``masks`` : list of ``(H, W)`` uint8 arrays (0/255) at the
          model's working resolution.
        - ``confidences`` : list of floats in [0, 1].
        - ``orig_size`` : ``(H, W)`` of the source frames.
        - ``model_size`` : ``(H, W)`` actually used for SAM2.
    """
    if not frames_bgr:
        return {"masks": [], "confidences": [], "orig_size": (0, 0), "model_size": (0, 0)}

    orig_h, orig_w = frames_bgr[0].shape[:2]
    frame_names, mh, mw = _ensure_jpeg_sequence(frames_bgr, work_dir, long_side=long_side)

    # Compute scale between source frames and the SAM2 model input so
    # we can rescale the user prompt correctly.
    scale = mw / float(orig_w)

    t0 = time.time()
    from app.models.sam2_model import get_sam2_video_predictor
    predictor = get_sam2_video_predictor()
    state = predictor.init_state(video_path=work_dir)
    masks: List[np.ndarray] = []
    confs: List[float] = []
    try:
        predictor.reset_state(state)

        if point_prompt is not None:
            # SAM2 wants points normalised to [0, 1] when
            # ``normalize_coords=True`` (the default).
            px, py = point_prompt
            points = np.array([[px * scale, py * scale]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)  # positive click
            _, _, _ = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
            )
            print(
                f"[sam2] using point prompt at source=({px:.0f},{py:.0f}) "
                f"model=({px * scale:.0f},{py * scale:.0f})",
                flush=True,
            )
        elif box_prompt is not None:
            x0, y0, x1, y1 = box_prompt
            box = np.array(
                [x0 * scale, y0 * scale, x1 * scale, y1 * scale],
                dtype=np.float32,
            )
            _, _, _ = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                box=box,
            )
            print(f"[sam2] using box prompt at source={box_prompt}", flush=True)
        else:
            # No user prompt: run saliency on the first source frame
            # and pass the auto-detected bounding box to SAM2.
            box_src = _saliency_auto_box(frames_bgr[0])
            box_model = box_src * scale
            _, _, _ = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                box=box_model.astype(np.float32),
            )
            print(
                f"[sam2] auto saliency box at source=("
                f"{box_src[0]:.0f},{box_src[1]:.0f})-"
                f"({box_src[2]:.0f},{box_src[3]:.0f})",
                flush=True,
            )

        for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(state):
            if video_res_masks is None or len(video_res_masks) == 0:
                masks.append(_empty_mask(mh, mw))
                confs.append(0.0)
                continue
            if hasattr(video_res_masks, "cpu"):
                # [N, H, W] boolean / float tensor across objects.
                m = (video_res_masks > 0.0).any(dim=0).cpu().numpy()
            else:
                m = (video_res_masks[0] > 0.0)
            arr = m.astype(np.uint8) * 255
            masks.append(arr)
            confs.append(float(m.mean()))
    finally:
        try:
            predictor.reset_state(state)
        except Exception:
            pass

    elapsed = time.time() - t0
    print(
        f"[sam2] {len(masks)} frames propagated in {elapsed:.2f}s "
        f"({len(masks) / max(elapsed, 1e-6):.2f} fps, model {mw}x{mh})",
        flush=True,
    )
    return {
        "masks": masks,
        "confidences": confs,
        "orig_size": (orig_h, orig_w),
        "model_size": (mh, mw),
    }
