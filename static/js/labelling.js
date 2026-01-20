// Labelling Studio (API-driven)

const API_BASE = '/api/labelling';

const state = {
  docId: null,
  docs: [],
  markers: [],
  annotations: [],
  relations: [],
  history: null,
  text: '',
  selection: { start: null, end: null, text: '' },
};

function $(id) {
  return document.getElementById(id);
}

function setDisabled(id, disabled) {
  const el = $(id);
  if (el) el.disabled = !!disabled;
}

function setText(id, text) {
  const el = $(id);
  if (el) el.textContent = text;
}

function setStatus(message, { isError = false } = {}) {
  const el = $('lsStatus');
  if (!el) return;
  el.textContent = message || '';
  el.style.color = isError ? '#b91c1c' : '#374151';

  const dzStatus = $('lsDropzoneStatus');
  if (dzStatus) {
    dzStatus.textContent = message || '';
    dzStatus.style.color = isError ? '#b91c1c' : '#6b7280';
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

function escapeHtml(s) {
  return (s || '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[c]));
}

function sentenceForSpan(text, start, end) {
  const t = text || '';
  if (start == null || end == null || start < 0 || end > t.length || start >= end) {
    return { sentence: '', index: 0 };
  }

  // Frontières simples: ., !, ?, \n
  let s0 = 0;
  for (let i = start; i >= 0; i--) {
    const ch = t[i];
    if (ch === '\n' || ch === '.' || ch === '!' || ch === '?') {
      s0 = Math.min(t.length, i + 1);
      break;
    }
  }

  let s1 = t.length;
  for (let i = end; i < t.length; i++) {
    const ch = t[i];
    if (ch === '\n' || ch === '.' || ch === '!' || ch === '?') {
      s1 = i + 1;
      break;
    }
  }

  const sentence = t.slice(s0, s1).trim();

  // index_phrase approx: nombre de séparateurs avant s0
  let idx = 0;
  for (let i = 0; i < s0; i++) {
    const ch = t[i];
    if (ch === '\n' || ch === '.' || ch === '!' || ch === '?') idx++;
  }

  return { sentence, index: idx };
}

function computePreviewHtml(text, annotations) {
  const t = text || '';
  const anns = (annotations || [])
    .filter(a => Number.isInteger(a.span_start) && Number.isInteger(a.span_end))
    .map(a => ({
      start: a.span_start,
      end: a.span_end,
      color: a.couleur || '#fff59d',
      label: a.marqueur || 'MARK',
    }))
    .filter(a => a.start >= 0 && a.end > a.start && a.end <= t.length)
    .sort((a, b) => a.start - b.start);

  // évite les overlaps (MVP)
  const cleaned = [];
  let lastEnd = -1;
  for (const a of anns) {
    if (a.start < lastEnd) continue;
    cleaned.push(a);
    lastEnd = a.end;
  }

  let out = '';
  let cursor = 0;
  for (const a of cleaned) {
    out += escapeHtml(t.slice(cursor, a.start));
    const chunk = escapeHtml(t.slice(a.start, a.end));
    out += `<mark style="background:${escapeHtml(a.color)}" title="${escapeHtml(a.label)}">${chunk}</mark>`;
    cursor = a.end;
  }
  out += escapeHtml(t.slice(cursor));
  return out;
}

function enableDocActions(enabled) {
  setDisabled('lsRefreshBtn', !enabled);
  setDisabled('lsAnnotateBtn', !enabled);
  setDisabled('lsEraseBtn', !enabled);
  setDisabled('lsUndoBtn', !enabled);
  setDisabled('lsRedoBtn', !enabled);
  setDisabled('lsExportCsvBtn', !enabled);
  setDisabled('lsExportXlsxBtn', !enabled);
  setDisabled('lsDatasetZipBtn', !enabled);
  setDisabled('lsBackupBtn', !enabled);
  setDisabled('lsRestoreBtn', !enabled);
  setDisabled('lsSuggestionsBtn', !enabled);
}

function renderDocs() {
  const container = $('lsDocs');
  if (!container) return;
  container.innerHTML = '';

  if (!state.docs.length) {
    container.innerHTML = `<div class="ls-muted" style="padding:8px;">Aucun document</div>`;
    return;
  }

  for (const d of state.docs) {
    const item = document.createElement('div');
    item.className = 'ls-item' + (d.doc_id === state.docId ? ' is-active' : '');
    item.onclick = () => loadDocument(d.doc_id);

    const left = document.createElement('div');
    left.className = 'ls-item-left';
    left.innerHTML = `
      <div class="ls-item-title">${escapeHtml(d.name || d.doc_id)}</div>
      <div class="ls-item-sub">${escapeHtml(d.doc_id)} · ${d.annotation_count || 0} annot.</div>
    `;

    const badge = document.createElement('div');
    badge.className = 'ls-badge';
    badge.textContent = String(d.text_length || 0);

    item.appendChild(left);
    item.appendChild(badge);
    container.appendChild(item);
  }
}

function renderMarkers() {
  const select = $('lsMarkerSelect');
  if (!select) return;
  select.innerHTML = '';

  const quickColor = $('lsQuickColor');
  const applyColorBtn = $('lsApplyColorBtn');
  if (applyColorBtn) applyColorBtn.disabled = true;

  for (const m of state.markers) {
    const opt = document.createElement('option');
    opt.value = m.name;
    opt.textContent = m.name;
    select.appendChild(opt);
  }

  if (quickColor && state.markers.length) {
    const selected = state.markers.find(x => x.name === select.value) || state.markers[0];
    if (selected && selected.color) quickColor.value = selected.color;
    if (applyColorBtn) applyColorBtn.disabled = false;
  }
}

function renderAnnotationsList() {
  const container = $('lsAnnotations');
  if (!container) return;
  container.innerHTML = '';

  if (!state.annotations.length) {
    container.innerHTML = `<div class="ls-muted" style="padding:8px;">Aucune annotation</div>`;
    return;
  }

  for (const a of state.annotations.slice().reverse().slice(0, 200)) {
    const label = a.marqueur || '';
    const word = a.mot || '';
    const item = document.createElement('div');
    item.className = 'ls-item';
    item.style.cursor = 'default';
    item.innerHTML = `
      <div class="ls-item-left">
        <div class="ls-item-title">${escapeHtml(word)}</div>
        <div class="ls-item-sub">${escapeHtml(label)} · ${escapeHtml(a.ann_id || '')}</div>
      </div>
      <div class="ls-badge" title="span">${Number.isInteger(a.span_start) ? `${a.span_start}-${a.span_end}` : '—'}</div>
    `;
    container.appendChild(item);
  }
}

function renderSuggestions(list) {
  const container = $('lsSuggestions');
  if (!container) return;
  container.innerHTML = '';
  const suggestions = (list && list.suggestions) ? list.suggestions : [];
  if (!suggestions.length) {
    container.innerHTML = `<div class="ls-muted" style="padding:8px;">Aucune suggestion</div>`;
    return;
  }

  for (const s of suggestions.slice(0, 100)) {
    const item = document.createElement('div');
    item.className = 'ls-item';
    item.style.cursor = 'default';

    item.innerHTML = `
      <div class="ls-item-left">
        <div class="ls-item-title">${escapeHtml(s.mot || '')}</div>
        <div class="ls-item-sub">${escapeHtml(s.marqueur || '')} · score ${typeof s.score === 'number' ? s.score.toFixed(2) : ''}</div>
      </div>
      <button class="ls-btn" type="button">Appliquer</button>
    `;

    item.querySelector('button').onclick = async () => {
      if (!state.docId || !state.text) return;
      const mot = String(s.mot || '').trim();
      if (!mot) return;

      // span approximatif: première occurrence
      const idx = state.text.toLowerCase().indexOf(mot.toLowerCase());
      const spanStart = idx >= 0 ? idx : null;
      const spanEnd = idx >= 0 ? (idx + mot.length) : null;
      const sent = (idx >= 0) ? sentenceForSpan(state.text, spanStart, spanEnd) : { sentence: '', index: 0 };

      const payload = {
        mot,
        marqueur: String(s.marqueur || 'SUGGESTION'),
        couleur: String(s.couleur || '#00FF00'),
        phrase: sent.sentence,
        index_phrase: sent.index,
        span_start: spanStart,
        span_end: spanEnd,
      };

      try {
        await api(`/annotate?doc_id=${encodeURIComponent(state.docId)}&auto_expand=false&dedupe=true`, {
          method: 'POST',
          body: JSON.stringify(payload),
        });
        await loadDocument(state.docId);
        setStatus('Suggestion appliquée');
      } catch (e) {
        setStatus(e.message, { isError: true });
      }
    };

    container.appendChild(item);
  }
}

function updateSelectionInfo() {
  const info = $('lsSelectionInfo');
  if (!state.docId) {
    info.textContent = '';
    return;
  }
  const sel = state.selection;
  if (sel.start == null || sel.end == null || sel.start === sel.end) {
    info.textContent = 'Aucune sélection';
    return;
  }
  const sample = sel.text.length > 64 ? sel.text.slice(0, 64) + '…' : sel.text;
  info.textContent = `Sélection: ${sel.start}-${sel.end} · "${sample}"`;
}

async function refreshDocs() {
  state.docs = (await api('/documents')).documents || [];
  renderDocs();
}

async function refreshMarkers() {
  const data = await api('/markers');
  state.markers = Array.isArray(data) ? data : (data.markers || []);
  renderMarkers();
}

async function applyColorToSelectedMarker() {
  const select = $('lsMarkerSelect');
  const quickColor = $('lsQuickColor');
  if (!select || !quickColor) return;
  const markerName = select.value;
  const color = quickColor.value;
  if (!markerName || !color) return;
  try {
    setStatus('Mise à jour couleur…');
    await api(`/markers/${encodeURIComponent(markerName)}`, { method: 'PUT', body: JSON.stringify({ color }) });
    await refreshMarkers();
    setStatus('Couleur mise à jour');
  } catch (e) {
    setStatus(e.message, { isError: true });
  }
}

function renderHistoryPanel() {
  const list = $('lsHistoryList');
  const meta = $('lsHistoryMeta');
  if (!list || !meta) return;

  const h = state.history;
  const items = (h && Array.isArray(h.history)) ? h.history : [];
  const current = (h && typeof h.current === 'number') ? h.current : null;

  meta.textContent = (current != null)
    ? `Position: ${current + 1} / ${items.length}`
    : (items.length ? `${items.length} entrées` : '');

  list.innerHTML = '';
  if (!items.length) {
    list.innerHTML = '<div class="ls-muted" style="padding:8px;">Aucun historique</div>';
    return;
  }

  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    const row = document.createElement('div');
    row.className = 'ls-history-item' + (current === i ? ' is-current' : '');

    const label = (it && typeof it === 'object')
      ? (it.action || it.type || it.label || `Étape ${i + 1}`)
      : `Étape ${i + 1}`;
    const detail = (it && typeof it === 'object')
      ? (it.detail || it.note || it.meta || '')
      : '';

    row.innerHTML = `
      <div class="ls-item-title">${escapeHtml(String(label))}</div>
      ${detail ? `<div class="ls-item-sub">${escapeHtml(String(detail))}</div>` : ''}
    `;
    list.appendChild(row);
  }
}

async function loadDocument(docId) {
  try {
    setStatus('Chargement…');
    const data = await api(`/documents/${encodeURIComponent(docId)}`);
    state.docId = docId;
    state.text = data.text || '';
    state.annotations = data.annotations || [];
    state.relations = data.relations || [];
    state.history = data.history || null;

    setText('lsDocTitle', (data.info && data.info.name) ? data.info.name : docId);
    setText('lsDocMeta', `${(data.info && data.info.annotation_count) || 0} annotations`);

    const textarea = $('lsText');
    if (textarea) {
      textarea.value = state.text;
      textarea.dataset.lastSaved = state.text;
    }
    setDisabled('lsSaveTextBtn', true);

    const dz = $('lsDropzone');
    if (dz) dz.style.display = 'none';

    const preview = $('lsPreview');
    if (preview) preview.innerHTML = computePreviewHtml(state.text, state.annotations);
    renderAnnotationsList();

    const h = state.history;
    if (h && typeof h.current === 'number') {
      setText('lsHistoryInfo', `Position: ${h.current + 1} / ${(h.history || []).length}`);
    } else {
      setText('lsHistoryInfo', '');
    }

    renderHistoryPanel();

    enableDocActions(true);
    setDisabled('lsAnnotateBtn', true); // activé uniquement sur sélection
    setStatus('Prêt');
    renderDocs();
  } catch (e) {
    setStatus(e.message, { isError: true });
    enableDocActions(false);
  }
}

async function createDocument() {
  const name = ($('lsNewDocName').value || '').trim() || 'document';
  try {
    setStatus('Création…');
    const res = await api('/documents', { method: 'POST', body: JSON.stringify({ name, text: '' }) });
    await refreshDocs();
    if (res && res.doc_id) {
      await loadDocument(res.doc_id);
    }
    setStatus('Document créé');
  } catch (e) {
    setStatus(e.message, { isError: true });
  }
}

async function uploadDocument() {
  const input = $('lsDropInput') || $('lsUploadInput');
  if (!input || !input.files || !input.files.length) {
    setStatus('Aucun fichier sélectionné', { isError: true });
    return;
  }
  return uploadDocumentFromFile(input.files[0]);
}

async function uploadDocumentFromFile(file) {
  if (!file) return;
  const fd = new FormData();
  fd.append('file', file);

  try {
    // Affichage immédiat (UX): pour les .txt on peut prévisualiser sans attendre le serveur.
    const name = (file.name || '').toLowerCase();
    if (name.endsWith('.txt')) {
      try {
        const txt = await file.text();
        state.text = txt;
        const textarea = $('lsText');
        if (textarea) {
          textarea.value = txt;
          textarea.dataset.lastSaved = txt;
        }
        const preview = $('lsPreview');
        if (preview) preview.innerHTML = computePreviewHtml(txt, []);
        const dz0 = $('lsDropzone');
        if (dz0) dz0.style.display = 'none';
        setStatus(`Texte chargé: ${file.name}`);
      } catch {
        // ignore: fallback sur import serveur
      }
    }

    const dz = $('lsDropzone');
    const dropInput = $('lsDropInput');
    const uploadBtn = $('lsUploadBtn');
    if (dz) dz.classList.add('is-dragover');
    if (dropInput) dropInput.disabled = true;
    if (uploadBtn) uploadBtn.disabled = true;

    setStatus(`Import… (${file.name})`);
    const res = await api('/upload', { method: 'POST', body: fd });
    if (res && res.doc_id) {
      // Applique immédiatement le texte (plus robuste que d’attendre un second call)
      state.docId = res.doc_id;
      state.text = (typeof res.text === 'string') ? res.text : '';
      state.annotations = [];
      state.relations = [];
      state.history = null;

      setText('lsDocTitle', file.name || 'Document');
      setText('lsDocMeta', '0 annotations');

      const textarea = $('lsText');
      if (textarea) {
        textarea.value = state.text;
        textarea.dataset.lastSaved = state.text;
      }
      setDisabled('lsSaveTextBtn', true);

      const preview = $('lsPreview');
      if (preview) preview.innerHTML = computePreviewHtml(state.text, []);
      renderAnnotationsList();
      renderHistoryPanel();

      const dz2 = $('lsDropzone');
      if (dz2) dz2.style.display = 'none';

      enableDocActions(true);
      setDisabled('lsAnnotateBtn', true);
      setStatus('Import terminé');

      // Best-effort: récupère l’état complet (history, meta, etc.)
      loadDocument(res.doc_id).catch(() => {});
    } else {
      setStatus('Import terminé (réponse inattendue)', { isError: true });
    }
  } catch (e) {
    setStatus(e.message, { isError: true });
    const dz = $('lsDropzone');
    if (dz) dz.classList.remove('is-dragover');
  }
  finally {
    const dropInput = $('lsDropInput');
    const uploadBtn = $('lsUploadBtn');
    if (dropInput) dropInput.disabled = false;
    if (uploadBtn) uploadBtn.disabled = false;
  }
}

async function addMarker() {
  const name = ($('lsMarkerName').value || '').trim();
  const color = $('lsMarkerColor').value;
  if (!name) {
    setStatus('Nom de marqueur requis', { isError: true });
    return;
  }
  try {
    await api('/markers', { method: 'POST', body: JSON.stringify({ name, color }) });
    $('lsMarkerName').value = '';
    await refreshMarkers();
    setStatus('Marqueur ajouté');
  } catch (e) {
    setStatus(e.message, { isError: true });
  }
}

async function annotateSelection() {
  if (!state.docId) {
    setStatus('Importez un document avant d’annoter', { isError: true });
    return;
  }
  const sel = state.selection;
  if (sel.start == null || sel.end == null || sel.start === sel.end) {
    setStatus('Sélectionnez un passage à annoter', { isError: true });
    return;
  }

  const markerSelect = $('lsMarkerSelect');
  const markerName = markerSelect ? markerSelect.value : '';
  const marker = state.markers.find(m => m.name === markerName);
  if (!marker) {
    setStatus('Ajoutez ou sélectionnez un marqueur', { isError: true });
    return;
  }

  const sent = sentenceForSpan(state.text, sel.start, sel.end);

  const payload = {
    mot: sel.text,
    marqueur: marker.name,
    couleur: marker.color,
    phrase: sent.sentence,
    index_phrase: sent.index,
    span_start: sel.start,
    span_end: sel.end,
  };

  const autoExpand = 'true';
  const dedupe = 'true';

  try {
    setStatus('Annotation…');
    await api(`/annotate?doc_id=${encodeURIComponent(state.docId)}&auto_expand=${autoExpand}&dedupe=${dedupe}`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    await loadDocument(state.docId);
    setStatus('Annotation ajoutée');
  } catch (e) {
    setStatus(e.message, { isError: true });
  }
}

async function eraseSelection() {
  if (!state.docId) {
    setStatus('Importez un document avant d’effacer', { isError: true });
    return;
  }
  const sel = state.selection;
  if (sel.start == null || sel.end == null || sel.start === sel.end) {
    setStatus('Sélectionnez un passage à effacer', { isError: true });
    return;
  }
  try {
    setStatus('Effacement…');
    await api(`/annotations/delete_by_span?doc_id=${encodeURIComponent(state.docId)}`, {
      method: 'POST',
      body: JSON.stringify({ span_start: sel.start, span_end: sel.end }),
    });
    await loadDocument(state.docId);
    setStatus('Effacement terminé');
  } catch (e) {
    setStatus(e.message, { isError: true });
  }
}

async function undo() {
  if (!state.docId) return;
  try {
    await api(`/history/undo?doc_id=${encodeURIComponent(state.docId)}`, { method: 'POST' });
    await loadDocument(state.docId);
    setStatus('Undo');
  } catch (e) {
    setStatus(e.message, { isError: true });
  }
}

async function redo() {
  if (!state.docId) return;
  try {
    await api(`/history/redo?doc_id=${encodeURIComponent(state.docId)}`, { method: 'POST' });
    await loadDocument(state.docId);
    setStatus('Redo');
  } catch (e) {
    setStatus(e.message, { isError: true });
  }
}

async function saveText() {
  if (!state.docId) return;
  const textarea = $('lsText');
  const text = textarea.value || '';
  try {
    if (!confirm('Enregistrer le texte réinitialise les annotations et l’historique pour ce document. Continuer ?')) {
      return;
    }
    setStatus('Enregistrement…');
    await api(`/documents/${encodeURIComponent(state.docId)}/text`, {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
    await loadDocument(state.docId);
    setStatus('Texte enregistré');
  } catch (e) {
    setStatus(e.message, { isError: true });
  }
}

async function fetchSuggestions() {
  if (!state.docId) return;
  try {
    setStatus('Suggestions…');
    const data = await api(`/suggestions?doc_id=${encodeURIComponent(state.docId)}`);
    renderSuggestions(data);
    setStatus('Suggestions reçues');
  } catch (e) {
    setStatus(e.message, { isError: true });
  }
}

function download(url) {
  window.location.href = url;
}

async function restoreBackup() {
  if (!state.docId) return;
  const input = $('lsRestoreInput');
  if (!input.files || !input.files.length) {
    setStatus('Sélectionnez un fichier JSON', { isError: true });
    return;
  }
  const fd = new FormData();
  fd.append('file', input.files[0]);
  try {
    setStatus('Restore…');
    await api(`/restore?doc_id=${encodeURIComponent(state.docId)}`, { method: 'POST', body: fd });
    await loadDocument(state.docId);
    setStatus('Restore terminé');
  } catch (e) {
    setStatus(e.message, { isError: true });
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  enableDocActions(false);

  // Affiche l’overlay tant qu’aucun document n’est chargé
  const dz = $('lsDropzone');
  if (dz) dz.style.display = 'flex';
  const btnCreate = $('lsCreateDocBtn');
  if (btnCreate) btnCreate.onclick = createDocument;

  const btnUpload = $('lsUploadBtn');
  if (btnUpload) btnUpload.onclick = uploadDocument;

  const btnAddMarker = $('lsAddMarkerBtn');
  if (btnAddMarker) btnAddMarker.onclick = addMarker;

  const btnAnnotate = $('lsAnnotateBtn');
  if (btnAnnotate) btnAnnotate.onclick = annotateSelection;

  const btnErase = $('lsEraseBtn');
  if (btnErase) btnErase.onclick = eraseSelection;

  const btnApplyColor = $('lsApplyColorBtn');
  if (btnApplyColor) btnApplyColor.onclick = applyColorToSelectedMarker;

  const btnUndo = $('lsUndoBtn');
  if (btnUndo) btnUndo.onclick = undo;

  const btnRedo = $('lsRedoBtn');
  if (btnRedo) btnRedo.onclick = redo;

  const btnSaveText = $('lsSaveTextBtn');
  if (btnSaveText) btnSaveText.onclick = saveText;

  const btnRefresh = $('lsRefreshBtn');
  if (btnRefresh) btnRefresh.onclick = () => state.docId && loadDocument(state.docId);

  const btnSuggestions = $('lsSuggestionsBtn');
  if (btnSuggestions) btnSuggestions.onclick = fetchSuggestions;

  const btnRestore = $('lsRestoreBtn');
  if (btnRestore) btnRestore.onclick = restoreBackup;

  const btnExportCsv = $('lsExportCsvBtn');
  if (btnExportCsv) btnExportCsv.onclick = () => state.docId && download(`${API_BASE}/export/csv?doc_id=${encodeURIComponent(state.docId)}`);

  const btnExportXlsx = $('lsExportXlsxBtn');
  if (btnExportXlsx) btnExportXlsx.onclick = () => state.docId && download(`${API_BASE}/export/xlsx?doc_id=${encodeURIComponent(state.docId)}`);

  const btnDatasetZip = $('lsDatasetZipBtn');
  if (btnDatasetZip) btnDatasetZip.onclick = () => state.docId && download(`${API_BASE}/dataset.zip?doc_id=${encodeURIComponent(state.docId)}`);

  const btnBackup = $('lsBackupBtn');
  if (btnBackup) btnBackup.onclick = () => state.docId && download(`${API_BASE}/backup?doc_id=${encodeURIComponent(state.docId)}`);

  const textarea = $('lsText');
  if (textarea) {
    textarea.addEventListener('input', () => {
      const last = textarea.dataset.lastSaved || '';
      setDisabled('lsSaveTextBtn', (textarea.value === last) || !state.docId);
    });
  }
  const updateSel = () => {
    if (!state.docId) return;
    if (!textarea) return;
    const s = textarea.selectionStart;
    const e = textarea.selectionEnd;
    const txt = (s != null && e != null && e > s) ? textarea.value.slice(s, e) : '';
    state.selection = { start: s, end: e, text: txt };
    updateSelectionInfo();
    setDisabled('lsAnnotateBtn', !(txt && txt.trim().length));
    setDisabled('lsEraseBtn', !(txt && txt.trim().length));
  };
  if (textarea) {
    textarea.addEventListener('mouseup', updateSel);
    textarea.addEventListener('keyup', updateSel);
  }

  const historyToggle = $('lsHistoryToggleBtn');
  const historyPanel = $('lsHistoryPanel');
  if (historyToggle && historyPanel) {
    historyToggle.onclick = () => {
      const isOpen = historyPanel.style.display !== 'none';
      historyPanel.style.display = isOpen ? 'none' : 'block';
      historyToggle.setAttribute('aria-expanded', String(!isOpen));
      if (!isOpen) renderHistoryPanel();
    };
  }

  const dropzone = $('lsDropzone');
  const dropInput = $('lsDropInput');
  if (dropzone && dropInput) {
    const setDrag = (on) => dropzone.classList.toggle('is-dragover', !!on);

    // Important: empêcher le navigateur d'ouvrir le fichier en navigation
    // et supporter un drop "un peu à côté" de la zone.
    const preventDefault = (e) => {
      e.preventDefault();
    };

    document.addEventListener('dragover', preventDefault);
    document.addEventListener('drop', preventDefault);

    document.addEventListener('dragenter', (e) => {
      // highlight si un fichier est en train d'être dragué
      if (e.dataTransfer && Array.from(e.dataTransfer.types || []).includes('Files')) {
        setDrag(true);
      }
    });

    dropzone.addEventListener('dragenter', (e) => { e.preventDefault(); setDrag(true); });
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); setDrag(true); });
    dropzone.addEventListener('dragleave', (e) => { e.preventDefault(); setDrag(false); });
    dropzone.addEventListener('drop', async (e) => {
      e.preventDefault();
      setDrag(false);
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (!file) return;
      setStatus(`Fichier reçu: ${file.name}`);
      await uploadDocumentFromFile(file);
    });

    // Fallback: drop n'importe où sur la page
    document.addEventListener('drop', async (e) => {
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (!file) return;
      setDrag(false);
      setStatus(`Fichier reçu: ${file.name}`);
      await uploadDocumentFromFile(file);
    });

    // IMPORTANT: ne pas intercepter les clics sur l'input / bouton (sinon l'UX devient incohérente)
    dropzone.addEventListener('click', (e) => {
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'BUTTON' || t.closest('input,button,label'))) {
        return;
      }
      dropInput.click();
    });
    dropzone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        dropInput.click();
      }
    });

    dropInput.addEventListener('change', async () => {
      const file = dropInput.files && dropInput.files[0];
      if (!file) return;
      setStatus(`Fichier sélectionné: ${file.name}`);
      await uploadDocumentFromFile(file);
      dropInput.value = '';
    });

    // Empêche la propagation: cliquer sur le bouton "Importer" ne doit pas ré-ouvrir le sélecteur.
    const uploadBtn = $('lsUploadBtn');
    if (uploadBtn) {
      uploadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
      });
    }
    dropInput.addEventListener('click', (e) => e.stopPropagation());
  }

  try {
    setStatus('Initialisation…');
    await refreshMarkers();

    const select = $('lsMarkerSelect');
    const quickColor = $('lsQuickColor');
    if (select) {
      select.addEventListener('change', () => {
        const m = state.markers.find(x => x.name === select.value);
        if (quickColor && m && m.color) quickColor.value = m.color;
      });
    }
    setStatus('Prêt');
  } catch (e) {
    setStatus(e.message, { isError: true });
  }
});
