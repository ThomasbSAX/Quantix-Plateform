// Labelling Studio — page Import (minimale)

const API_BASE = '/api/labelling';

function byId(id) {
  return document.getElementById(id);
}

function setStatus(message, { isError = false } = {}) {
  const top = byId('lsStatus');
  if (top) {
    top.textContent = message || '';
    top.style.color = isError ? '#b91c1c' : '#374151';
  }
  const dz = byId('lsDropzoneStatus');
  if (dz) {
    dz.textContent = message || '';
    dz.style.color = isError ? '#b91c1c' : '#6b7280';
  }
}

async function api(path, options = {}) {
  const resp = await fetch(`${API_BASE}${path}`, {
    headers: {
      ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...(options.headers || {}),
    },
    ...options,
  });

  const contentType = (resp.headers.get('content-type') || '').toLowerCase();
  const isJson = contentType.includes('application/json');
  const data = isJson ? await resp.json().catch(() => null) : await resp.text().catch(() => '');
  if (!resp.ok) {
    const msg = (data && data.error) ? data.error : (typeof data === 'string' ? data : `HTTP ${resp.status}`);
    throw new Error(msg);
  }
  return data;
}

async function uploadFile(file) {
  if (!file) return;

  const dropzone = byId('lsDropzone');
  const input = byId('lsDropInput');
  const btn = byId('lsUploadBtn');

  try {
    if (dropzone) dropzone.classList.add('is-dragover');
    if (input) input.disabled = true;
    if (btn) btn.disabled = true;

    setStatus(`Import… (${file.name})`);

    const fd = new FormData();
    fd.append('file', file);
    const res = await api('/upload', { method: 'POST', body: fd });

    if (res && res.doc_id) {
      setStatus('Import terminé');
      window.location.assign(`/labelling/annotate?doc_id=${encodeURIComponent(res.doc_id)}`);
      return;
    }

    setStatus('Import terminé (réponse inattendue)', { isError: true });
  } catch (e) {
    setStatus(e.message || String(e), { isError: true });
  } finally {
    if (dropzone) dropzone.classList.remove('is-dragover');
    if (input) input.disabled = false;
    if (btn) btn.disabled = false;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const dropzone = byId('lsDropzone');
  const input = byId('lsDropInput');
  const btn = byId('lsUploadBtn');

  const preventDefault = (e) => e.preventDefault();
  document.addEventListener('dragover', preventDefault);
  document.addEventListener('drop', preventDefault);

  if (btn) {
    btn.addEventListener('click', async (e) => {
      e.preventDefault();
      e.stopPropagation();
      const file = input && input.files && input.files[0];
      if (!file) {
        setStatus('Aucun fichier sélectionné', { isError: true });
        return;
      }
      await uploadFile(file);
    });
  }

  if (input) {
    input.addEventListener('change', async () => {
      const file = input.files && input.files[0];
      if (!file) return;
      await uploadFile(file);
      input.value = '';
    });

    input.addEventListener('click', (e) => e.stopPropagation());
  }

  if (dropzone) {
    const setDrag = (on) => dropzone.classList.toggle('is-dragover', !!on);

    dropzone.addEventListener('dragenter', (e) => { e.preventDefault(); setDrag(true); });
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); setDrag(true); });
    dropzone.addEventListener('dragleave', (e) => { e.preventDefault(); setDrag(false); });
    dropzone.addEventListener('drop', async (e) => {
      e.preventDefault();
      setDrag(false);
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (!file) return;
      await uploadFile(file);
    });

    dropzone.addEventListener('click', (e) => {
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'BUTTON' || t.closest('input,button,label'))) {
        return;
      }
      if (input) input.click();
    });

    dropzone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        if (input) input.click();
      }
    });
  }

  setStatus('Prêt');
});
