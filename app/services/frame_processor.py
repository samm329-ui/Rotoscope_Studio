"""Frame processing service for Rotoscope Studio.

It dispatches each frame through the selected pipeline
(fast matte, precise rotoscope, RVM, or hybrid SAM2 + BiRefNet) and
writes masks into the job folder.
"""
import os
import pathlib
from typing import Any, Dict, Optional

import app.config as _config
from app.services import job_store
from app.pipelines import fast_matte as _fast_matte
from app.pipelines import precise_rotoscope as _precise
from app.pipelines import rvm_rotoscope as _rvm


WORKFLOW_FAST = 'fast_matte'
WORKFLOW_PRECISE = 'precise_rotoscope'
WORKFLOW_RVM = 'rvm_rotoscope'
WORKFLOW_HYBRID = 'hybrid_rotoscope'
VALID_WORKFLOWS = {WORKFLOW_FAST, WORKFLOW_PRECISE, WORKFLOW_RVM, WORKFLOW_HYBRID}


def _resolve_pipeline(workflow: str):
    """Resolve a workflow name to its pipeline module."""
    if workflow in (WORKFLOW_FAST, 'fast_matte'):
        return _fast_matte
    if workflow in (WORKFLOW_PRECISE, 'precise_rotoscope'):
        return _precise
    if workflow in (WORKFLOW_RVM, 'rvm_rotoscope'):
        return _rvm
    raise ValueError(f'Unknown workflow: {workflow}')


def process_frames(job_id: str, workflow: str, subject: Optional[str] = None) -> int:
    """Run the selected pipeline on all frames for a given job."""
    job = job_store.get_job(job_id)
    if subject is None and job is not None:
        subject = job.get('subject')
    if workflow == WORKFLOW_HYBRID:
        # The hybrid pipeline consumes the source video directly, so
        # we need the upload path; the orchestrator reads it.
        from app.core.roto_pipeline import process_video

        def _progress_cb(percent: int, step_idx: int, name: str) -> None:
            perc = 25 + int(0.5 * max(0, min(percent, 100)))
            job_store.update_job(
                job_id,
                progress_percent=perc,
                current_step=name or 'processing_frames',
            )

        file_path = job.get('file_path') if job else None
        if not file_path or not os.path.isfile(file_path):
            raise FileNotFoundError(
                f'Job {job_id} has no source video on disk for the hybrid pipeline.'
            )
        # Optional user click on the first frame. The upload form
        # can carry 'click_x' and 'click_y' (pixels in source frame).
        point_prompt = None
        if job is not None:
            cx = job.get('click_x')
            cy = job.get('click_y')
            if cx is not None and cy is not None:
                point_prompt = (float(cx), float(cy))

        result = process_video(
            job_id,
            file_path,
            subject=subject,
            point_prompt=point_prompt,
            progress_cb=_progress_cb,
        )
        return len(result.get('masks') or [])

    pipeline = _resolve_pipeline(workflow)
    frames_dir = _config.frames_dir(job_id)
    if not frames_dir.is_dir() or not any(frames_dir.iterdir()):
        raise FileNotFoundError(f'No frames found for job {job_id}. Extract frames first.')

    def _progress_cb(idx: int, total: int) -> None:
        if total <= 0:
            return
        perc = 25 + (50 * idx / float(total))
        perc = max(25, min(75, perc))
        job_store.update_job(job_id, progress_percent=int(perc), current_step='processing_frames')

    if workflow == WORKFLOW_FAST:
        total_count = _fast_matte.process_job(job_id, progress_cb=_progress_cb)
    elif workflow == WORKFLOW_PRECISE:
        total_count = _precise.process_job(job_id, subject=subject, progress_cb=_progress_cb)
    elif workflow == WORKFLOW_RVM:
        total_count = _rvm.process_job(job_id, subject=subject, progress_cb=_progress_cb)
    else:
        raise ValueError(f'Unknown workflow: {workflow}')
    return total_count


def get_progress(job_id: str) -> Dict:
    """Return a small status dict for the job. Useful for the UI to show a concise summary."""
    job = job_store.get_job(job_id)
    frames_dir = _config.frames_dir(job_id)
    masks_dir = _config.masks_dir(job_id)
    frame_count = 0
    if frames_dir.is_dir():
        frame_count = len(list(frames_dir.glob('frame_*.png')))
    mask_count = 0
    if masks_dir.is_dir():
        mask_count = len(list(masks_dir.glob('frame_*.png')))
    return {
        'frame_count': frame_count,
        'mask_count': mask_count,
        'progress_percent': job.get('progress_percent', 0) if job else 0,
        'current_step': job.get('current_step', '') if job else '',
    }
