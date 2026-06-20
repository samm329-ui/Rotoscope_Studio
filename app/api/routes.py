"""Rotoscope Studio API routes."""
import os, shutil, threading, uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile


import app.config as _config
from app.services import job_store, frame_extractor, frame_processor, preview_builder, export_packager


router = APIRouter(prefix='/api')

WORKFLOW_FAST = 'fast_matte'
WORKFLOW_PRECISE = 'precise_rotoscope'
VALID_WORKFLOWS = {WORKFLOW_FAST, WORKFLOW_PRECISE}


@router.post('/upload')
async def upload_video(
    file: UploadFile = File(...),
    workflow: str = Form(WORKFLOW_FAST),
    subject: Optional[str] = Form(None),
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
    job = job_store.create_job(job_id=job_id, file_name=file.filename, file_path=str(upload_path), workflow=workflow, subject=subject)
    return {'job_id': job_id, 'file_name': file.filename, 'workflow': workflow, 'status': job['state']}


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
        job_store.update_job(job_id, state='processing', current_step='extracting_frames')
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
