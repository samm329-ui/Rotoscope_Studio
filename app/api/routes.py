"""Rotoscope Studio API routes."""
import os, shutil, threading, uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse


import app.config as _config
from app.services import job_store, frame_extractor, frame_processor, preview_builder, export_packager


router = APIRouter(prefix='/api')

WORKFLOW_FAST = 'fast_matte'
WORKFLOW_PRECISE = 'precise_rotoscope'
WORKFLOW_RVM = 'rvm_rotoscope'
WORKFLOW_HYBRID = 'hybrid_rotoscope'
VALID_WORKFLOWS = {WORKFLOW_FAST, WORKFLOW_PRECISE, WORKFLOW_RVM, WORKFLOW_HYBRID}


@router.post('/upload')
async def upload_video(
    file: UploadFile = File(...),
    workflow: str = Form(WORKFLOW_RVM),
    subject: Optional[str] = Form(None),
    click_x: Optional[float] = Form(None),
    click_y: Optional[float] = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail='No filename provided.')
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in _config.SUPPORTED_EXT:
        raise HTTPException(status_code=400, detail=f'Unsupported file type: {ext}')
    if workflow not in VALID_WORKFLOWS:
        raise HTTPException(status_code=400, detail=f'Unknown workflow: {workflow}')
    job_id = uuid.uuid4().hex[:12]
    _config.ensure_job_dir(job_id)
    upload_path = _config.upload_path(job_id) / file.filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    with open(upload_path, 'wb') as out:
        shutil.copyfileobj(file.file, out)
    # Stash the click coordinates on the job record so the
    # processing thread can read them when it starts.
    create_kwargs = dict(file_name=file.filename, file_path=str(upload_path), workflow=workflow, subject=subject)
    if click_x is not None:
        create_kwargs['click_x'] = float(click_x)
    if click_y is not None:
        create_kwargs['click_y'] = float(click_y)
    job = job_store.create_job(job_id=job_id, **create_kwargs)
    return {'job_id': job_id, 'file_name': file.filename, 'workflow': workflow, 'status': job['state']}


@router.post('/job/{job_id}/prompt')
async def set_prompt(job_id: str, click_x: Optional[float] = Form(None), click_y: Optional[float] = Form(None), subject: Optional[str] = Form(None)):
    # Attach a click prompt (and / or subject class) to an existing
    # job. Used by the click-to-select UI between the subject step
    # and the process step.
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found.')
    kwargs = {}
    if click_x is not None:
        kwargs['click_x'] = float(click_x)
    if click_y is not None:
        kwargs['click_y'] = float(click_y)
    if subject:
        kwargs['subject'] = subject
    job_store.update_job(job_id, **kwargs)
    return {'job_id': job_id, 'prompt': kwargs}

@router.post('/process/{job_id}')
async def start_processing(job_id: str):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found.')
    if job['state'] in ('completed', 'processing'):
        return {'job_id': job_id, 'status': job['state'], 'current_step': job.get('current_step', '')}
    t = threading.Thread(target=_run_pipeline, args=(job_id,), daemon=True)
    t.start()
    return {'job_id': job_id, 'status': 'processing', 'current_step': 'starting'}


def _run_pipeline(job_id: str) -> None:
    try:
        job = job_store.get_job(job_id)
        file_path = job['file_path']
        workflow = job['workflow']
        job_store.update_job(job_id, state='processing', current_step='starting')
        if workflow == WORKFLOW_HYBRID:
            # Hybrid pipeline: video -> SAM2 -> selective BiRefNet ->
            # fusion -> temporal smoothing -> alpha. No per-frame PNGs
            # are extracted to disk; masks are produced directly.
            frame_processor.process_frames(job_id, workflow)
        elif workflow == WORKFLOW_RVM:
            if frame_extractor._has_ffmpeg():
                frame_extractor.extract_frames_fast(job_id, file_path)
            else:
                frame_extractor.extract_frames(job_id, file_path)
            job_store.update_job(job_id, current_step='processing_frames', progress_percent=25)
            frame_processor.process_frames(job_id, workflow)
        else:
            if frame_extractor._has_ffmpeg():
                frame_extractor.extract_frames_fast(job_id, file_path)
            else:
                frame_extractor.extract_frames(job_id, file_path)
            job_store.update_job(job_id, current_step='processing_frames', progress_percent=25)
            frame_processor.process_frames(job_id, workflow)
        job_store.update_job(job_id, current_step='building_preview', progress_percent=75)
        preview_builder.build_preview(job_id, workflow)
        job_store.update_job(job_id, current_step='packaging_export', progress_percent=90)
        export_packager.package_export(job_id)
        job_store.update_job(job_id, state='completed', current_step='done', progress_percent=100)
    except Exception as exc:
        job_store.update_job(job_id, state='failed', error_message=str(exc))


@router.get('/first_frame/{job_id}')
async def get_first_frame(job_id: str):
    # Return a JPEG preview of the first frame so the click-to-select
    # UI can display it. Generated on demand from the source video
    # and cached in the job folder so the second hit is instant.
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found.')
    fp = job.get('file_path')
    if not fp or not os.path.isfile(fp):
        raise HTTPException(status_code=404, detail='Source video not on disk.')
    cache = _config.frames_dir(job_id) / 'first_preview.jpg'
    if not cache.is_file():
        import cv2
        cap = cv2.VideoCapture(fp)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail='Cannot open source video.')
        try:
            ret, frame = cap.read()
            if not ret:
                raise HTTPException(status_code=500, detail='No frames in source video.')
            cache.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cache), frame)
        finally:
            cap.release()
    return FileResponse(str(cache), media_type='image/jpeg')

@router.get('/status/{job_id}')
async def get_status(job_id: str):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found.')
    return {'state': job['state'], 'progress_percent': job.get('progress_percent', 0), 'current_step': job.get('current_step', ''), 'error_message': job.get('error_message')}


@router.get('/preview/{job_id}')
async def get_preview(job_id: str):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found.')
    if job['state'] not in ('preview_ready', 'completed'):
        raise HTTPException(status_code=409, detail='Preview not ready yet.')
    return {'preview_assets': preview_builder.list_previews(job_id), 'preview_summary': preview_builder.preview_summary(job_id), 'workflow_used': job['workflow']}


@router.get('/export/{job_id}')
async def get_export(job_id: str):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found.')
    if job['state'] != 'completed':
        raise HTTPException(status_code=409, detail='Export not ready yet.')
    export_path = _config.exports_dir(job_id) / 'rotoscope_export.zip'
    if not export_path.exists():
        raise HTTPException(status_code=500, detail='Export file missing.')
    return {'export_file': str(export_path), 'export_ready': True}


@router.delete('/job/{job_id}')
async def delete_job(job_id: str):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found.')
    job_folder = _config.JOBS_DIR / job_id
    if job_folder.exists():
        shutil.rmtree(job_folder)
    job_store.delete_job(job_id)
    return {'job_id': job_id, 'deleted': True}
