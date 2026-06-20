# Rotoscope Studio - central configuration.
import os
import pathlib


# Root directory of the project (the folder holding the entire app).
ROOT = pathlib.Path(os.path.dirname(__file__)).parent

# Folders for run-time artifacts (extracted frames, masks, previews, exports).
JOBS_DIR = ROOT / 'jobs'
UPLOADS_DIR = ROOT / 'uploads'
FRAMES_DIR = ROOT / 'frames'
MASKS_DIR = ROOT / 'masks'
PREVIEWS_DIR = ROOT / 'previews'
EXPORTS_DIR = ROOT / 'exports'
LOGS_DIR = ROOT / 'logs'
# Sub-directory names inside each job folder.
UPLOADS_SUBDIR = 'uploads'
FRAMES_SUBDIR = 'frames'
MASKS_SUBDIR = 'masks'
PREVIEWS_SUBDIR = 'previews'
EXPORTS_SUBDIR = 'exports'
LOGS_SUBDIR = 'logs'
FRONTEND_DIR = ROOT / 'frontend'


# Server settings.
HOST = os.environ.get('ROTOHOST', '127.0.0.1')
PORT = int(os.environ.get('ROTOPORT', 8000))


# Processing tunables.
FAST_MATTE_MAX_SIZE = 480
PRECISE_MAX_SIZE = 1080


# Supported video extensions for frame extraction.
SUPPORTED_EXT = {'.mp4', '.mov', '.avi', '.webm', '.mkv'}


def ensure_job_dir(job_id: str) -> pathlib.Path:
    """Make sure a job folder exists for the given job_id.

    The job folder is the canonical location for all per-request artifacts: frames, masks, previews,
    exports, and logs."""
    d = JOBS_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    for sub in (UPLOADS_SUBDIR, FRAMES_SUBDIR, MASKS_SUBDIR, PREVIEWS_SUBDIR,
              EXPORTS_SUBDIR, LOGS_SUBDIR):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def job_path(job_id: str) -> pathlib.Path:
    """Return the root folder for a given job_id."""
    return ensure_job_dir(job_id)


def upload_path(job_id: str) -> pathlib.Path:
    """Return the path where uploaded video file is stored."""
    return job_path(job_id) / 'uploads'


def frames_dir(job_id: str) -> pathlib.Path:
    """Return the path for extracted frames."""
    return job_path(job_id) / 'frames'


def masks_dir(job_id: str) -> pathlib.Path:
    """Return the path for generated masks."""
    return job_path(job_id) / 'masks'


def previews_dir(job_id: str) -> pathlib.Path:
    """Return the path for preview assets."""
    return job_path(job_id) / 'previews'


def exports_dir(job_id: str) -> pathlib.Path:
    """Return the path for export outputs."""
    return job_path(job_id) / 'exports'


def log_path(job_id: str) -> pathlib.Path:
    """Return the path to the log file for this job."""
    d = ensure_job_dir(job_id)
    log = d / 'job.log'
    log.touch()
    return log


def frame_path(job_id: str, name: str) -> pathlib.Path:
    """Return the path to a single extracted frame file within the job folder."""
    return frames_dir(job_id) / name


def mask_path(job_id: str, name: str) -> pathlib.Path:
    """Return the path to a single generated mask file."""
    return masks_dir(job_id) / name
