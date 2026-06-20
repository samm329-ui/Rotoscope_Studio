# Frame extraction service for Rotoscope Studio.
#
# Extracts frames from a video file using OpenCV-Python.
# The extracted frames are saved as PNG files
# inside the frames/ subfolder of the job folder.
import os
import pathlib
from typing import Any, Dict

import app.config as _config


def extract_frames(job_id: str, video_path: str) -> int:
    """Extract frames from a video file using OpenCV-Python.

    Frames are saved as PNG files in the frames/ subfolder of the job folder. The number of
    frames extracted is returned. If the video cannot be opened,
    a FileNotFoundError is raised so the caller can react gracefully."""
    import cv2
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f'Video not found: {video_path}')
    frames_dir = _config.frames_dir(job_id)
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Clear any previous frames to avoid stale results.
    for fn in os.listdir(frames_dir):
        try:
            os.unlink(frames_dir / fn)
        except OSError:
            pass
    # Open the video file using OpenCV.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'OpenCV cannot open video: {video_path}')
    try:
        frame_count = 0
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame_path = frames_dir / f'frame_{frame_index:06d}.png'
            cv2.imwrite(str(frame_path), frame)
            frame_index += 1
    finally:
        cap.release()
    return frame_count


def get_video_metadata(video_path: str) -> Dict:
    """Read basic metadata from the video file using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return {'fps': fps, 'width': width, 'height': height, 'frame_count': frame_count}
