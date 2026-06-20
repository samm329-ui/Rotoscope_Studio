# Rotoscope Studio

Rotoscope Studio is a standalone tool that separates a subject from the background of a video. It supports two workflows:

- **Fast Matte** - quick segmentation, lower resolution. Built on top of the provided `FindMattes.py` helper.
- **Precise Rotoscope** - higher resolution, subject-aware, with a separate pipeline.

The project is organized so the backend processing layer can later be reused inside a larger video editor.

## Quickstart

### Prerequisites
- Python 3.9 or higher
- Internet access (only on first run, for installing Python packages)

### Windows
Double-click `start.bat` or run it from a terminal:

```
start.bat
```

### macOS / Linux
From a terminal:

```
bash start.sh
```

The starter script will:
1. Verify that Python and the required packages are present.
2. Install any missing packages from `requirements.txt`.
3. Start the backend at `http://127.0.0.1:8000`.

The browser should open automatically. If it does not, navigate to `http://127.0.0.1:8000`.

## Project Structure

```
Rotoscope_studio/
+- app/
|  +- api/             # FastAPI routes
|  +- pipelines/       # Fast matte and precise rotoscope pipelines
|  +- services/        # Frame extraction, processing, preview, export
|  +- config.py        # Central configuration
|  +- main.py          # FastAPI entry point
+- frontend/
|  +- css/             # Styles
|  +- js/              # Frontend logic
|  +- index.html       # Single-page UI
+- scripts/            # setup_check.py, install_deps.py
+- jobs/               # Per-job artifacts (created at runtime)
+- logs/               # Per-job logs (created at runtime)
+- start.bat           # Windows starter
+- start.sh            # macOS / Linux starter
+- requirements.txt
+- README.md
```

## Workflow

1. **Welcome** - the user lands on a simple welcome screen.
2. **Upload** - drag and drop a video file (MP4, MOV, AVI, WebM, MKV).
3. **Workflow** - choose Fast Matte or Precise Rotoscope.
4. **Subject** (precise only) - pick the person or object to keep.
5. **Process** - watch progress and status updates.
6. **Preview** - inspect a side-by-side sprite of original frames and generated masks.
7. **Export** - download a zip bundle that contains the frames, masks, preview, and metadata.

## API Endpoints

| Endpoint                          | Method | Description                              |
|-----------------------------------|--------|------------------------------------------|
| /api/upload                       | POST   | Upload a video and create a job.         |
| /api/process/{job_id}             | POST   | Start processing the job.                |
| /api/status/{job_id}              | GET    | Poll job status.                         |
| /api/preview/{job_id}             | GET    | List preview assets.                     |
| /api/export/{job_id}              | GET    | Get the export bundle path.              |
| /api/job/{job_id}                 | DELETE | Delete a job and its artifacts.          |
| /api/health                       | GET    | Health check used by the setup helper.   |
| /files/{job_id}/{subdir}/{name}   | GET    | Serve a per-job file (preview, mask).    |

## Pipelines

### Fast Matte
- Located in `app/pipelines/fast_matte.py`.
- Wraps the provided `app/pipelines/FindMattes.py` helper.
- Each frame is resized to a height of 480 px before segmentation.

### Precise Rotoscope
- Located in `app/pipelines/precise_rotoscope.py`.
- A separate pipeline that runs the FCN model at a higher resolution (1080 px).
- Respects a subject hint from the user.
- Outputs grayscale alpha PNGs (one per frame).

`FindMattes.py` is treated as the authoritative source for the fast matte workflow and is not rewritten. The precise pipeline is a separate path with its own logic and output.

## Reusability

The backend code is structured so that it can be imported into a larger video editor:

- `find_mattes.py` is the snake-case alias for the helper.
- The frame extraction, processing, preview, and export services are decoupled from the API layer.
- The job state model is reusable: a job is identified by a short id and persisted as `jobs/<id>/job.json`.

The standalone frontend, setup helper, and demo screens are intended to be replaced in the editor integration.

## Troubleshooting

- **Python not found**: install Python 3.9+ from https://www.python.org/ and make sure it is on your PATH.
- **Package install fails**: check your internet connection and run `start.bat` again.
- **Video cannot be opened**: try converting the file to MP4 with H.264 video and AAC audio.
- **Out of memory**: the precise workflow loads the FCN model and processes each frame at higher resolution. A machine with 8 GB or more of RAM is recommended.