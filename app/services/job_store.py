# In-memory job store for the Rotoscope Studio backend.
#
# Job records are kept as JSON in a file (job.json) inside the
# job folder. The file is the source of truth for the rest of the
# backend, and it can be re-imported later by the video editor project.
import json
import pathlib
import threading
from typing import Any, Dict, Optional

import app.config as _config


_lock = threading.Lock()
_jobs: Dict[str, Dict] = {}


VALID_STATES = {
    'waiting_for_upload',
    'file_ready',
    'workflow_selected',
    'subject_selected',
    'processing',
    'preview_ready',
    'export_ready',
    'completed',
    'failed',
}


def _job_path(job_id: str) -> pathlib.Path:
    """Return the path to the job.json file."""
    return _config.JOBS_DIR / job_id / 'job.json'


def _save_job(job: Dict[str, Any]) -> None:
    """Persist the job to disk as json for survival across restarts."""
    path = _job_path(job['job_id'])
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(job, f, indent=2)
    except OSError:
        pass


def _load_job(job_id: str) -> Optional[Dict]:
    """Read the job from disk if it exists."""
    path = _job_path(job_id)
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def create_job(job_id: str, file_name: str, file_path: str,
         workflow: str = 'fast_matte', subject: Optional[str] = None) -> Dict:
    """Create a new job record and persist it."""
    job = {
        'job_id': job_id,
        'file_name': file_name,
        'file_path': file_path,
        'workflow': workflow,
        'subject': subject,
        'state': 'waiting_for_upload',
        'current_step': '',
        'progress_percent': 0,
        'error_message': None,
    }
    _save_job(job)
    with _lock:
        _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[Dict]:
    """Return the job record for a given job_id."""
    with _lock:
        return _jobs.get(job_id)


def update_job(job_id: str, **kwargs) -> None:
    """Update fields on the job and persist."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.update(kwargs)
    _save_job(job)


def delete_job(job_id: str) -> None:
    """Remove the job from the in-memory store."""
    with _lock:
        _jobs.pop(job_id, None)


def set_subject(job_id: str, subject: Optional[str]) -> None:
    """Update the subject field for a job used by the precise workflow."""
    update_job(job_id, subject=subject)


def set_workflow(job_id: str, workflow: str) -> None:
    """Update the workflow field for a job."""
    update_job(job_id, workflow=workflow)
