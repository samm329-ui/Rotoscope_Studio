"""Frame processing service for Rotoscope Studio.

It dispatches each frame through the selected pipeline
(fast matte or precise rotoscope) and writes masks into the job folder.
"""
import os
import pathlib
from typing import Any, Dict, Optional

import app.config as _config
from app.services import job_store
from app.pipelines import fast_matte as _fast_matte
from app.pipelines import precise_rotoscope as _precise


WORKFLOW_FAST = 'fast_matte'
WORKFLOW_PRECISE = 'precise_rotoscope'
VALID_WORKFLOWS = {WORKFLOW_FAST, WORKFLOW_PRECISE}


def _resolve_pipeline(workflow: str):
    """Resolve a workflow name to its pipeline module."""
    if workflow == WORKFLOW_FAST or workflow == 'fast_matte':
        return _fast_matte
    if workflow == WORKFLOW_PRECISE or workflow == 'precise_rotoscope':
        return _precise
    raise ValueError(f'Unknown workflow: {workflow}')


def process_frames(job_id: str, workflow: str, subject: Optional[str] = None) -> int:
    """Run the selected pipeline on all frames for a given job.

    The job's frames folder must already be populated with PNG files
    by the frame extractor. The subject hint is passed only to the precise
    pipeline."""
    pipeline = _resolve_pipeline(workflow)
    job = job_store.get_job(job_id)
    if subject is None and job is not None:
        subject = job.get('subject')
    frames_dir = _config.frames_dir(job_id)
    if not frames_dir.is_dir() or not any(frames_dir.iterdir()):
        raise FileNotFoundError(f'No frames found for job {job_id}. Extract frames first.')
    def _progress_cb(idx: int, total: int) -> None:
        if total <= 0:
            return
        # Map the pipeline internal index (0..N-1) to a 0..100 percentage and shift to the 25-75 range that the routes expect.
        perc = 25 + (50 * idx / float(total))
        perc = max(25, min(75, perc))
        job_store.update_job(job_id, progress_percent=int(perc), current_step='processing_frames')
    if workflow == WORKFLOW_FAST:
        total_count = _fast_matte.process_job(job_id, progress_cb=_progress_cb)
    elif workflow == WORKFLOW_PRECISE:
        total_count = _precise.process_job(job_id, subject=subject, progress_cb=_progress_cb)
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
    return {'frame_count': frame_count, 'mask_count': mask_count, 'progress_percent': job.get('progress_percent', 0) if job else 0, 'current_step': job.get('current_step', '') if job else ''}
