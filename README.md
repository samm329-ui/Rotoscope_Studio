# Rotoscope Studio

Rotoscope Studio is a standalone tool that separates a subject from the background of a video. It supports three workflows:

- **Fast Matte** - quick segmentation, lower resolution. Built on top of the provided `FindMattes.py` helper.
- **Precise Rotoscope** - higher resolution, subject-aware, with a separate pipeline.
- **RVM Rotoscope** *(recommended)* - Robust Video Matting end-to-end pipeline with true alpha, temporal smoothing, and no per-frame PNG round-trip. Runs ~5-15x faster than the FCN paths on CPU and produces soft, hair-preserving alpha mattes instead of binary masks.

The project is organized so the backend processing layer can later be reused inside a larger video editor.

## Quickstart

### Prerequisites
- Python 3.9 or higher
- **ffmpeg** on `PATH` (used for the fast frame extractor and the RVM streaming path). Install with `winget install Gyan.FFmpeg` on Windows or `brew install ffmpeg` on macOS.
- Internet access (only on first run, to install Python packages and to download the RVM model the first time you select it).

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
|  +- pipelines/       # fast_matte, precise_rotoscope, rvm_rotoscope
|  +- services/        # Frame extraction, processing, preview, export
|  +- config.py        # Central configuration
|  +- main.py          # FastAPI entry point
+- frontend/
|  +- css/             # Styles
|  +- js/              # Frontend logic
|  +- index.html       # Single-page UI
+- scripts/            # setup_check.py, install_deps.py
+- assets/             # Cached ML models (e.g. rvm_mobilenetv3.torchscript.pt)
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
3. **Workflow** - choose Fast Matte, Precise Rotoscope, or RVM Rotoscope.
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

### RVM Rotoscope (recommended)
- Located in `app/pipelines/rvm_rotoscope.py`.
- Uses Robust Video Matting (MobileNetV3 backbone, TorchScript).
- Auto-downloads the ~14 MB model to `assets/rvm_mobilenetv3.torchscript.pt` on first use.
- Streams the source video through ffmpeg + PIL + RVM without writing per-frame PNGs to disk.
- Maintains a recurrent state across frames for temporal consistency.
- Outputs true alpha mattes (0.0-1.0, not binary masks) so hair, motion blur, and transparent edges survive.
- Tunable with the env var `RVM_DS_RATIO` (default `0.25`; lower = faster but softer).

`FindMattes.py` is treated as the authoritative source for the fast matte workflow and is not rewritten. The precise pipeline is a separate path with its own logic and output.

## Performance notes

| Workflow          | Resolution | CPU only (i5-class) 10s 30fps clip |
|-------------------|-----------|------------------------------------|
| Fast Matte (old)  | 320 px    | 30-40 min                          |
| Precise (old)     | 720 px    | 60+ min                            |
| RVM Rotoscope     | native    | 30-90 s                            |

RVM's biggest wins are:
1. No PNG round-trip for each frame.
2. Recurrent state - the model only does ~1/4 of the work per frame after the first.
3. True alpha - the post-processing smoothing pass is not needed.

GPU is still ~10x faster than CPU. On a discrete NVIDIA GPU you can expect ~10-15 s for a 10s 30fps 720p clip. On your current CPU machine, **30-90 seconds** is a realistic target, not 10-12 seconds, unless you reduce the source resolution.

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
- **ffmpeg missing**: install ffmpeg or the fast extractor / RVM streaming paths will fall back to the slower OpenCV path.
- **RVM model download fails**: place `rvm_mobilenetv3.torchscript.pt` manually inside the `assets/` folder. The model is ~14 MB.

## Click-to-Select UI

The subject card now features an interactive click-to-select interface:

1. After uploading a video, the first frame is displayed as a preview.
2. Click directly on the subject in the preview to set a point prompt.
3. The click coordinates are sent to the server and used as the seed for segmentation.
4. You can optionally select a subject class from the standard list (Person, Man, Woman, etc.).
5. Click "Start Processing" to begin the pipeline with your selection.

This workflow works for all pipelines (Fast Matte, Precise Rotoscope, RVM, and Hybrid).

## Hybrid Rotoscope (SAM2 + BiRefNet) - architecture

The codebase ships a four-stage hybrid architecture designed for the
target quality bar of professional rotoscoping. **It is NOT the
default workflow** because on a CPU-only machine it is much slower
than the RVM path. The architecture is provided so a developer with
a GPU can flip the default and immediately get the high-quality path.

```text
video_loader
   |
sam_tracking          (Stage 2/3 - SAM2 video propagation)
   |
frame_selector        (Stage 4 - low-confidence / high-edge / disagree frames)
   |
biref_refinement      (Stage 5 - BiRefNet on the selected frames only)
   |
mask_fusion           (Stage 6 - confidence-weighted blend)
   |
temporal_smoothing    (Stage 7 - 3-tap weighted average)
   |
alpha_generator       (Stage 8 - resize to source resolution)
   |
video_exporter        (Stage 9 - alpha PNGs + optional rgba.mov)
```

### Measured on the test laptop (i5-10300H, CPU only, SAM2 tiny)

| Stage            | Speed                | Notes                                   |
|------------------|----------------------|-----------------------------------------|
| SAM2 propagation | 0.14 fps @ 360x202   | scales linearly with model input size   |
| BiRefNet         | not measured         | ~0.5-1.5 fps @ 512px expected on CPU    |
| Hybrid total     | many minutes         | per 10s 30fps clip on CPU               |

For the 30-90 second target on this hardware, use the default
``rvm_rotoscope`` workflow. The hybrid path is the future plan for
when a GPU is available.