// Rotoscope Studio - frontend logic.
// State machine for the user journey. Talks to the backend over fetch.

const API_BASE = '/api';

const STATES = {
  WELCOME: 'welcome',
  UPLOAD: 'upload',
  WORKFLOW: 'workflow',
  SUBJECT: 'subject',
  PROCESS: 'process',
  PREVIEW: 'preview',
  EXPORT: 'export',
};

const app = {
  state: STATES.WELCOME,
  jobId: null,
  fileName: null,
  workflow: null,
  subject: null,
  clickX: null,
  clickY: null,
  statusPollHandle: null,
};

// Card management.
const CARD_IDS = {
  welcome: 'welcome-card',
  upload: 'upload-card',
  workflow: 'workflow-card',
  subject: 'subject-card',
  process: 'process-card',
  preview: 'preview-card',
  export: 'export-card',
};

function showCard(name) {
  Object.values(CARD_IDS).forEach(function (id) {
    const el = document.getElementById(id);
    if (el) el.classList.add('hidden');
  });
  const target = document.getElementById(CARD_IDS[name]);
  if (target) target.classList.remove('hidden');
  app.state = name;
  updateStepTracker();
}

function updateStepTracker() {
  const order = ['welcome', 'upload', 'workflow', 'subject', 'process', 'preview', 'export'];
  const currentIdx = order.indexOf(app.state);
  const steps = document.querySelectorAll('#step-tracker .step');
  steps.forEach(function (el) {
    el.classList.remove('active', 'done');
    const idx = order.indexOf(el.getAttribute('data-step'));
    if (idx < 0) return;
    if (idx === currentIdx) el.classList.add('active');
    else if (idx < currentIdx) el.classList.add('done');
  });
}

function showToast(message, isError) {
  const t = document.getElementById('toast');
  if (!t) return;
  t.textContent = message;
  t.classList.remove('hidden', 'error');
  if (isError) t.classList.add('error');
  setTimeout(function () {
    t.classList.add('hidden');
  }, 4000);
}

function setStatus(message) {
  const el = document.getElementById('process-status');
  if (el) el.textContent = message;
}

function setProgress(percent) {
  const fill = document.getElementById('progress-fill');
  const label = document.getElementById('process-percent');
  if (fill) fill.style.width = percent + '%';
  if (label) label.textContent = percent + '%';
}

// =============== WELCOME ===============

document.getElementById('start-btn').addEventListener('click', function () {
  showCard(STATES.UPLOAD);
});

document.getElementById('setup-btn').addEventListener('click', async function () {
  const out = document.getElementById('setup-output');
  out.classList.remove('hidden');
  out.textContent = 'Running setup helper ...';
  try {
    const res = await fetch(API_BASE + '/health');
    if (!res.ok) throw new Error('health check failed: ' + res.status);
    const data = await res.json();
    out.textContent =
      'Backend status: ' + (data.status || 'unknown') + '\n' +
      'App: ' + (data.name || 'Rotoscope Studio') + '\n' +
      'Version: ' + (data.version || '0.0.0') + '\n\n' +
      'Open the README for full setup instructions. The one-click start.bat / start.sh scripts will install missing Python packages for you.';
  } catch (e) {
    out.textContent = 'Could not reach the backend. Make sure the server is running.';
  }
});

// =============== UPLOAD ===============

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const fileNameEl = document.getElementById('file-name');
const nextBtn = document.getElementById('upload-next-btn');

function setFile(file) {
  if (!file) return;
  app.fileName = file.name;
  fileNameEl.textContent = file.name;
  fileInfo.classList.remove('hidden');
  nextBtn.disabled = false;
  dropzone.classList.add('hidden');
}

function clearFile() {
  app.fileName = null;
  fileInput.value = '';
  fileInfo.classList.add('hidden');
  nextBtn.disabled = true;
  dropzone.classList.remove('hidden');
}

document.getElementById('browse-btn').addEventListener('click', function () {
  fileInput.click();
});

fileInput.addEventListener('change', function (e) {
  const f = e.target.files && e.target.files[0];
  if (f) setFile(f);
});

['dragenter', 'dragover'].forEach(function (evt) {
  dropzone.addEventListener(evt, function (e) {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach(function (evt) {
  dropzone.addEventListener(evt, function (e) {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove('dragover');
  });
});

dropzone.addEventListener('drop', function (e) {
  const dt = e.dataTransfer;
  if (dt && dt.files && dt.files[0]) setFile(dt.files[0]);
});

document.getElementById('remove-btn').addEventListener('click', clearFile);

nextBtn.addEventListener('click', async function () {
  if (!app.fileName) {
    showToast('Please choose a video file first.', true);
    return;
  }
  nextBtn.disabled = true;
  nextBtn.textContent = 'Uploading ...';
  try {
    const fd = new FormData();
    fd.append('file', fileInput.files[0]);
    fd.append('workflow', app.workflow || 'fast_matte');
    if (app.subject) fd.append('subject', app.subject);
    const res = await fetch(API_BASE + '/upload', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json().catch(function () { return { detail: 'Upload failed' }; });
      throw new Error(err.detail || 'Upload failed');
    }
    const data = await res.json();
    app.jobId = data.job_id;
    showCard(STATES.WORKFLOW);
    showToast('Upload ready.');
  } catch (e) {
    showToast('Upload failed: ' + e.message, true);
  } finally {
    nextBtn.disabled = false;
    nextBtn.textContent = 'Next';
  }
});

// =============== WORKFLOW ===============

document.querySelectorAll('.workflow-card').forEach(function (card) {
  card.addEventListener('click', function () {
    document.querySelectorAll('.workflow-card').forEach(function (c) { c.classList.remove('selected'); });
    card.classList.add('selected');
    app.workflow = card.getAttribute('data-workflow');
    setTimeout(function () {
        // Always go through the subject card so the user can
        // click on the subject for a precise prompt.
        showCard(STATES.SUBJECT);
        loadFirstFramePreview();
      }, 150);
    });
  });

// =============== SUBJECT ===============

// Subject chips: optional class label that is forwarded to the backend
// in addition to (or instead of) a click prompt.
document.querySelectorAll('.subject-chip').forEach(function (chip) {
  chip.addEventListener('click', function () {
    document.querySelectorAll('.subject-chip').forEach(function (c) { c.classList.remove('selected'); });
    chip.classList.add('selected');
    app.subject = chip.getAttribute('data-subject');
    var label = document.getElementById('subject-label');
    if (label) label.textContent = app.subject;
    var info = document.getElementById('subject-info');
    if (info) info.classList.remove('hidden');
    var next = document.getElementById('subject-next-btn');
    if (next) next.disabled = false;
  });
});

function _clearSubjectClass() {
  app.subject = null;
  document.querySelectorAll('.subject-chip').forEach(function (c) { c.classList.remove('selected'); });
  var label = document.getElementById('subject-label');
  if (label) label.textContent = '';
}

var changeBtn = document.getElementById('change-subject-btn');
if (changeBtn) changeBtn.addEventListener('click', _clearSubjectClass);



// Click-to-select: click on the first-frame preview to set a point
// prompt that is forwarded to the backend before processing.
async function startProcessingWithPrompt() {
  if (!app.jobId) {
    showToast('Upload a video first.', true);
    return;
  }
  try {
    var fd = new FormData();
    if (app.clickX !== null && app.clickY !== null) {
      fd.append('click_x', String(app.clickX));
      fd.append('click_y', String(app.clickY));
    }
    if (app.subject) {
      fd.append('subject', app.subject);
    }
    await fetch(API_BASE + '/job/' + app.jobId + '/prompt', {
      method: 'POST',
      body: fd,
    });
  } catch (e) {
    showToast('Could not save click prompt: ' + e.message, true);
    return;
  }
  startProcessing();
}
document.getElementById('subject-next-btn').addEventListener('click', startProcessingWithPrompt);

async function loadFirstFramePreview() {
  var img = document.getElementById('first-frame-img');
  var marker = document.getElementById('click-marker');
  if (!img || !app.jobId) return;
  img.src = API_BASE + '/first_frame/' + app.jobId + '?t=' + Date.now();
  img.onload = function () {
    marker.classList.add('hidden');
    document.getElementById('subject-next-btn').disabled = false;
  };
  img.onerror = function () {
    showToast('Could not load the first-frame preview.', true);
  };
  img.onclick = function (ev) {
    var rect = img.getBoundingClientRect();
    var scaleX = img.naturalWidth / rect.width;
    var scaleY = img.naturalHeight / rect.height;
    app.clickX = Math.round((ev.clientX - rect.left) * scaleX);
    app.clickY = Math.round((ev.clientY - rect.top) * scaleY);
    marker.style.left = (ev.clientX - rect.left) + 'px';
    marker.style.top = (ev.clientY - rect.top) + 'px';
    marker.classList.remove('hidden');
    var label = document.getElementById('click-label');
    if (label) label.textContent = '(' + app.clickX + ', ' + app.clickY + ')';
    document.getElementById('subject-info').classList.remove('hidden');
  };
}

var reloadLink = document.getElementById('reload-preview-link');
if (reloadLink) reloadLink.addEventListener('click', function (e) {
  e.preventDefault();
  loadFirstFramePreview();
});

var clearClick = document.getElementById('clear-click-btn');
if (clearClick) clearClick.addEventListener('click', function () {
  app.clickX = null;
  app.clickY = null;
  var marker = document.getElementById('click-marker');
  if (marker) marker.classList.add('hidden');
  var label = document.getElementById('click-label');
  if (label) label.textContent = '(none)';
});

// =============== PROCESS ===============

async function startProcessing() {
  showCard(STATES.PROCESS);
  setStatus('Starting ...');
  setProgress(0);
  try {
    const res = await fetch(API_BASE + '/process/' + app.jobId, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(function () { return { detail: 'Process failed' }; });
      throw new Error(err.detail || 'Process failed');
    }
    pollStatus();
  } catch (e) {
    showToast('Process failed: ' + e.message, true);
    setStatus('Failed: ' + e.message);
  }
}

function pollStatus() {
  if (app.statusPollHandle) clearInterval(app.statusPollHandle);
  app.statusPollHandle = setInterval(async function () {
    try {
      const res = await fetch(API_BASE + '/status/' + app.jobId);
      if (!res.ok) return;
      const data = await res.json();
      setStatus(humanizeStep(data.current_step));
      setProgress(data.progress_percent || 0);
      if (data.state === 'failed') {
        clearInterval(app.statusPollHandle);
        setStatus('Failed: ' + (data.error_message || 'Unknown error'));
        showToast('Processing failed.', true);
        document.getElementById('retry-btn').classList.remove('hidden');
      } else if (data.state === 'completed' || data.state === 'preview_ready' || data.state === 'export_ready') {
        clearInterval(app.statusPollHandle);
        setStatus('Done');
        setProgress(100);
        showCard(STATES.PREVIEW);
        loadPreview();
      }
    } catch (e) {
      // network blip - keep polling
    }
  }, 1000);
}

function humanizeStep(step) {
  if (!step) return 'Working ...';
  const map = {
    extracting_frames: 'Extracting frames ...',
    processing_frames: 'Processing frames ...',
    building_preview: 'Building preview ...',
    packaging_export: 'Packaging export ...',
    done: 'Done',
  };
  return map[step] || step;
}

document.getElementById('cancel-btn').addEventListener('click', async function () {
  if (app.statusPollHandle) clearInterval(app.statusPollHandle);
  if (app.jobId) {
    try {
      await fetch(API_BASE + '/job/' + app.jobId, { method: 'DELETE' });
    } catch (e) { /* ignore */ }
  }
  resetSession();
});

document.getElementById('retry-btn').addEventListener('click', function () {
  document.getElementById('retry-btn').classList.add('hidden');
  startProcessing();
});

// =============== PREVIEW ===============

async function loadPreview() {
  const container = document.getElementById('preview-container');
  container.innerHTML = '<p class="muted">Loading preview ...</p>';
  try {
    const res = await fetch(API_BASE + '/preview/' + app.jobId);
    if (!res.ok) throw new Error('preview not ready');
    const data = await res.json();
    container.innerHTML = '';
    const summary = data.preview_summary || {};
    const assets = data.preview_assets || [];
    if (summary.count === 0 || assets.length === 0) {
      container.innerHTML = '<p class="muted">No preview assets were generated.</p>';
      return;
    }
    const heading = document.createElement('p');
    heading.className = 'muted small';
    heading.textContent = 'Workflow: ' + (data.workflow_used || 'unknown') + ' - ' + summary.count + ' preview asset(s)';
    container.appendChild(heading);
    assets.forEach(function (path) {
      const parts = path.split(/[/\\]/);
      const filename = parts[parts.length - 1];
      const subdir = parts[parts.length - 2];
      const url = '/files/' + app.jobId + '/' + subdir + '/' + filename;
      const img = document.createElement('img');
      img.src = url;
      img.alt = 'Preview asset';
      container.appendChild(img);
    });
  } catch (e) {
    document.getElementById('preview-container').innerHTML = '<p class="muted">Could not load preview.</p>';
  }
}

document.getElementById('view-preview-btn').addEventListener('click', function () {
  showCard(STATES.PREVIEW);
});

document.getElementById('compare-btn').addEventListener('click', function () {
  showToast('Switch to the other workflow from step 3 to compare results.', false);
  showCard(STATES.WORKFLOW);
});

document.getElementById('to-export-btn').addEventListener('click', function () {
  showCard(STATES.EXPORT);
});

// =============== EXPORT ===============

document.getElementById('download-btn').addEventListener('click', async function () {
  try {
    const res = await fetch(API_BASE + '/export/' + app.jobId);
    if (!res.ok) throw new Error('Export not ready');
    const data = await res.json();
    if (data.export_file) {
      showToast('Export ready: ' + data.export_file);
    }
  } catch (e) {
    showToast('Could not prepare export: ' + e.message, true);
  }
});

document.getElementById('reset-btn').addEventListener('click', resetSession);

async function resetSession() {
  if (app.statusPollHandle) clearInterval(app.statusPollHandle);
  if (app.jobId) {
    try { await fetch(API_BASE + '/job/' + app.jobId, { method: 'DELETE' }); } catch (e) { /* ignore */ }
  }
  app.jobId = null;
  app.fileName = null;
  app.workflow = null;
  app.subject = null;
  app.clickX = null;
  app.clickY = null;
  clearFile();
  document.getElementById('retry-btn').classList.add('hidden');
  document.querySelectorAll('.workflow-card').forEach(function (c) { c.classList.remove('selected'); });
  document.querySelectorAll('.subject-chip').forEach(function (c) { c.classList.remove('selected'); });
  document.getElementById('subject-info').classList.add('hidden');
  document.getElementById('subject-next-btn').disabled = true;
  setStatus('Ready');
  setProgress(0);
  showCard(STATES.WELCOME);
}

// Initial render.
showCard(STATES.WELCOME);