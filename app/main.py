"""Rotoscope Studio - FastAPI entry point.

This file starts the HTTP server, mounts the API routes, and serves the
single-page frontend from the frontend/directory.
"""
import os
import pathlib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import app.config as _config
import app.api.routes as api_routes


app = FastAPI(
    title="Rotoscope Studio",
    description="Standalone video separation studio.",
)

# Create root folders for artifacts.
_config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
_config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Mount the API routes first so that API calls do not fall through to the
# catch-all static handler.
app.include_router(api_routes.router)


@app.get("/api/health")
def health():
    """Return a simple health status for the setup helper."""
    return {"status": "ok", "name": "Rotoscope Studio", "version": "0.1.0"}


@app.get("/files/{job_id}/{subdir}/{filename}")
def serve_job_file(job_id: str, subdir: str, filename: str):
    """Serve a per-job file (preview image, etc.) for the UI."""
    safe_job = "".join(c for c in job_id if c.isalnum() or c in ("-", "_"))
    if safe_job != job_id:
        raise HTTPException(status_code=400, detail="Invalid job id.")
    if subdir not in ("previews", "masks", "frames", "uploads", "exports"):
        raise HTTPException(status_code=400, detail="Invalid subdir.")
    folder = _config.JOBS_DIR / safe_job / subdir
    file_path = folder / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(str(file_path))


@app.get("/")
def root_index():
    """Redirect the user to the frontend index page."""
    return RedirectResponse(url="/index.html")


# Mount the frontend as static files. This must come last so the API routes
# and the /files route get matched first.
app.mount("/", StaticFiles(directory=str(_config.FRONTEND_DIR), html=True), name="static")

# Allow loose CORS for local development. Pass the class, not an instance,
# to add_middleware.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


def run_server() -> None:
    """Entry point for the start script."""
    import uvicorn
    uvicorn.run("app:app", host=_config.HOST, port=_config.PORT, reload=False)