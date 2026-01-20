// Labelling Studio — page Annotation (sobre, minimal)

const API_BASE = '/api/labelling';

const state = {
  docId: null,
  markers: [],
  selectedMarker: null,
  text: '',
  annotations: [],
  selection: { start: null, end: null, text: '' },
};

function byId(id) {
  return document.getElementById(id);
}

function setDisabled(id, disabled) {
  const el = byId(id);
  if (el) el.disabled = !!disabled;
}

function setText(id, text) {
  const el = byId(id);
  if (el) el.textContent = text;
}

function setStatus(message, { isError = false } = {}) {
  const el = byId('lsStatus');
  if (!el) return;
  el.textContent = message || '';
  el.style.color = isError ? '#b91c1c' : '#374151';
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

function computeHighlightedHtml(text, annotations) {
  const t = text || '';
  const groups = new Map();
  for (const a of (annotations || [])) {
    if (!Number.isInteger(a.span_start) || !Number.isInteger(a.span_end)) continue;
    const start = a.span_start;
    const end = a.span_end;
    if (!(start >= 0 && end > start && end <= t.length)) continue;

    const key = `${start}:${end}`;
    if (!groups.has(key)) {
      groups.set(key, {
        start,
        end,
        mot: a.mot || t.slice(start, end),
        phrase: a.phrase || '',
        labels: [],
        colors: [],
      });
    }
    const g = groups.get(key);
    const label = a.marqueur || 'MARK';
    const color = a.couleur || '#fff59d';
    if (!g.labels.includes(label)) g.labels.push(label);
    if (!g.colors.includes(color)) g.colors.push(color);
  }

  const anns = Array.from(groups.values())
    .map(g => ({
      start: g.start,
      end: g.end,
      colors: (g.colors || []).slice(0, 3),
      labels: (g.labels || []).slice(0, 3),
    }))
    .sort((a, b) => a.start - b.start);

  // MVP: ignore overlaps
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
    const c1 = (a.colors && a.colors[0]) ? a.colors[0] : '#fff59d';
    const c2 = (a.colors && a.colors[1]) ? a.colors[1] : null;
    const c3 = (a.colors && a.colors[2]) ? a.colors[2] : null;
    const title = (a.labels || []).join(' | ');
    const shadows = [];
    if (c2) shadows.push(`inset 0 -2px 0 ${escapeHtml(c2)}`);
    if (c3) shadows.push(`inset 0 -4px 0 ${escapeHtml(c3)}`);
    const style = `background:${escapeHtml(c1)};${shadows.length ? `box-shadow:${shadows.join(',')};` : ''}`;
    out += `<mark class="ls-mark" style="${style}" title="${escapeHtml(title)}">${chunk}</mark>`;
    cursor = a.end;
  }
  out += escapeHtml(t.slice(cursor));
  return out;
}

function csvEscape(value) {
  const s = String(value == null ? '' : value);
  if (/[\n\r,\"]/g.test(s)) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

function buildCsvModel(text, annotations) {
  const t = text || '';
  const rowsBySpan = new Map();
  for (const a of (annotations || [])) {
    if (!Number.isInteger(a.span_start) || !Number.isInteger(a.span_end)) continue;
    const start = a.span_start;
    const end = a.span_end;
    if (!(start >= 0 && end > start && end <= t.length)) continue;
    const key = `${start}:${end}`;
    if (!rowsBySpan.has(key)) {
      rowsBySpan.set(key, {
        start,
        end,
        element: a.mot || t.slice(start, end),
        phrase: a.phrase || sentenceForSpan(t, start, end).sentence,
        markers: [],
        comments: [],
      });
    }
    const row = rowsBySpan.get(key);
    const m = String(a.marqueur || '').trim();
    if (m && !row.markers.includes(m)) row.markers.push(m);

    const c = String((a.commentaire != null ? a.commentaire : (a.meta && a.meta.commentaire)) || '').trim();
    if (c && !row.comments.includes(c)) row.comments.push(c);
  }

  const rows = Array.from(rowsBySpan.values())
    .sort((a, b) => (a.start - b.start) || (a.end - b.end))
    .map(r => {
    const markers = (r.markers || []).slice(0, 3);
    const comment = (r.comments || []).join(' | ');
    return {
      element: r.element || '',
      phrase: r.phrase || '',
      m1: markers[0] || '',
      m2: markers[1] || '',
      m3: markers[2] || '',
      comment: comment || '',
    };
    });

  const header = ['element_surligne', 'phrase_contexte', 'marqueur1', 'marqueur2', 'marqueur3', 'commentaire'];
  const lines = [header.join(',')];
  for (const r of rows) {
    lines.push([
      csvEscape(r.element),
      csvEscape(r.phrase),
      csvEscape(r.m1),
      csvEscape(r.m2),
      csvEscape(r.m3),
      csvEscape(r.comment),
    ].join(','));
  }

  return { rows, csv: lines.join('\n') };
}

function renderCsv(text, annotations) {
  const wrap = byId('lsCsvTable');
  const ta = byId('lsCsvText');
  const copyBtn = byId('lsCopyCsvBtn');
  const dlBtn = byId('lsDownloadCsvBtn');
  if (!wrap || !ta || !copyBtn) return;

  const model = buildCsvModel(text, annotations);
  ta.value = model.csv;
  copyBtn.disabled = model.rows.length === 0;
  if (dlBtn) dlBtn.disabled = model.rows.length === 0;

  if (!model.rows.length) {
    wrap.innerHTML = `<div class="ls-muted" style="padding:10px;">Aucune annotation</div>`;
    return;
  }

  const thead = `<thead><tr><th>Élément surligné</th><th>Phrase contexte</th><th>Marqueur 1</th><th>Marqueur 2</th><th>Marqueur 3</th><th>Commentaire</th></tr></thead>`;
  const tbody = model.rows.map(r => {
    return `<tr><td>${escapeHtml(r.element)}</td><td>${escapeHtml(r.phrase)}</td><td>${escapeHtml(r.m1)}</td><td>${escapeHtml(r.m2)}</td><td>${escapeHtml(r.m3)}</td><td>${escapeHtml(r.comment)}</td></tr>`;
  }).join('');
  wrap.innerHTML = `<table>${thead}<tbody>${tbody}</tbody></table>`;
}

async function ensureCommentMarker() {
  const targetName = 'COMMENTAIRE';
  const existing = (state.markers || []).find(m => String(m.name || '').toUpperCase() === targetName);
  if (existing) return existing;

  // Crée le marqueur si absent
  try {
    await api('/markers', { method: 'POST', body: JSON.stringify({ doc_id: state.docId, name: targetName, color: '#e5e7eb' }) });
  } catch (e) {
    // Si déjà créé en parallèle, on ignore et on refresh
  }
  await refreshMarkers(state.docId);
  return (state.markers || []).find(m => String(m.name || '').toUpperCase() === targetName) || null;
}

async function addCommentForSelection() {
  if (!state.docId) return;

  const textarea = byId('lsText');
  const isModified = textarea ? ((textarea.value || '') !== (textarea.dataset.lastSaved || '')) : false;
  if (isModified) {
    setStatus('Impossible: enregistrez le texte d’abord', { isError: true });
    return;
  }

  const sel = state.selection;
  if (sel.start == null || sel.end == null || sel.start === sel.end) {
    setStatus('Sélectionnez un passage à commenter', { isError: true });
    return;
  }

  const commentEl = byId('lsCommentText');
  const commentaire = commentEl ? String(commentEl.value || '').trim() : '';
  if (!commentaire) {
    setStatus('Commentaire requis', { isError: true });
    return;
  }

  let marker = await ensureCommentMarker();
  if (!marker) {
    // fallback si création impossible
    marker = { name: 'COMMENTAIRE', color: '#e5e7eb' };
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
    commentaire,
    meta: { kind: 'commentaire' },
  };

  try {
    setStatus('Ajout commentaire…');
    await api(`/annotate?doc_id=${encodeURIComponent(state.docId)}&auto_expand=false&dedupe=false`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    if (commentEl) commentEl.value = '';
    await loadDocument(state.docId);
    setStatus('Commentaire ajouté');
  } catch (e) {
    setStatus(e.message, { isError: true });
  }
}

function renderOverlay() {
  const textarea = byId('lsText');
  const overlay = byId('lsOverlay');
  if (!textarea || !overlay) return;

  const text = textarea.value || '';
  const last = textarea.dataset.lastSaved || '';

  // Si le texte est modifié localement, on affiche du texte brut pour éviter les spans incohérents.
  if (text !== last) {
    overlay.innerHTML = escapeHtml(text);
    setStatus('Texte modifié: enregistrez pour réinitialiser les annotations', { isError: false });

    const wrap = byId('lsCsvTable');
    const ta = byId('lsCsvText');
    const copyBtn = byId('lsCopyCsvBtn');
    const dlBtn = byId('lsDownloadCsvBtn');
    if (wrap && ta && copyBtn) {
      wrap.innerHTML = `<div class="ls-muted" style="padding:10px;">CSV désactivé (texte modifié localement)</div>`;
      ta.value = '';
      copyBtn.disabled = true;
      if (dlBtn) dlBtn.disabled = true;
    }
    return;
  }

  overlay.innerHTML = computeHighlightedHtml(text, state.annotations);
}

function sentenceForSpan(text, start, end) {
  const t = text || '';
  if (start == null || end == null || start < 0 || end > t.length || start >= end) {
    return { sentence: '', index: 0 };
  }

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
  let idx = 0;
  for (let i = 0; i < s0; i++) {
    const ch = t[i];
    if (ch === '\n' || ch === '.' || ch === '!' || ch === '?') idx++;
  }
  return { sentence, index: idx };
}

function updateSelectionInfo() {
  const info = byId('lsSelectionInfo');
  const sel = state.selection;
  if (!info) return;

  if (sel.start == null || sel.end == null || sel.start === sel.end) {
    info.textContent = 'Aucune sélection';
    return;
  }
  const sample = sel.text.length > 64 ? sel.text.slice(0, 64) + '…' : sel.text;
  info.textContent = `Sélection: ${sel.start}-${sel.end} · "${sample}"`;
}

function renderMarkerPills() {
  const wrap = byId('lsMarkerPills');
  if (!wrap) return;
  wrap.innerHTML = '';

  if (!state.markers.length) {
    wrap.innerHTML = `<div class="ls-muted" style="padding:8px;">Aucun marqueur</div>`;
    return;
  }

  for (const m of state.markers) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'ls-marker-pill' + (state.selectedMarker && state.selectedMarker.name === m.name ? ' is-active' : '');
    btn.innerHTML = `<span class="ls-marker-dot" style="background:${escapeHtml(m.color)}"></span>${escapeHtml(m.name)}`;
    btn.onclick = () => {
      state.selectedMarker = m;
      const qc = byId('lsQuickColor');
      if (qc) qc.value = m.color || '#fff59d';
      setDisabled('lsApplyColorBtn', false);
      renderMarkerPills();
      setStatus(`Marqueur sélectionné: ${m.name}`);
    };
    wrap.appendChild(btn);
  }

  if (!state.selectedMarker) {
    state.selectedMarker = state.markers[0];
    const qc = byId('lsQuickColor');
    if (qc) qc.value = state.selectedMarker.color || '#fff59d';
    setDisabled('lsApplyColorBtn', false);
    renderMarkerPills();
  }
}

async function refreshMarkers(docId) {
  const effectiveDocId = docId || state.docId;
  const qs = effectiveDocId ? `?doc_id=${encodeURIComponent(effectiveDocId)}` : '';
  const data = await api(`/markers${qs}`);
  state.markers = Array.isArray(data) ? data : (data.markers || []);
  // garde la sélection si possible
  if (state.selectedMarker) {
    const still = state.markers.find(x => x.name === state.selectedMarker.name);
    state.selectedMarker = still || null;
  }
  renderMarkerPills();
}

async function loadDocument(docId) {
  setStatus('Chargement…');
  const data = await api(`/documents/${encodeURIComponent(docId)}`);
  state.docId = docId;
  state.text = data.text || '';
  state.annotations = data.annotations || [];

  setText('lsDocTitle', (data.info && data.info.name) ? data.info.name : docId);
  setText('lsDocMeta', `${(data.info && data.info.annotation_count) || 0} annotations`);

  const textarea = byId('lsText');
  if (textarea) {
    textarea.value = state.text;
    textarea.dataset.lastSaved = state.text;
  }
  setDisabled('lsSaveTextBtn', true);

  renderOverlay();
  renderCsv(state.text, state.annotations);

  setDisabled('lsRefreshBtn', false);
  setStatus('Prêt');
}

async function saveText() {
  if (!state.docId) return;
  const textarea = byId('lsText');
  const text = textarea ? (textarea.value || '') : '';
  try {
    if (!confirm('Enregistrer le texte réinitialise les annotations pour ce document. Continuer ?')) return;
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

async function addMarker() {
  const name = (byId('lsMarkerName').value || '').trim();
  const color = byId('lsMarkerColor').value;
  if (!name) {
    setStatus('Nom de marqueur requis', { isError: true });
    return;
  }
  try {
    setStatus('Création marqueur…');
    await api('/markers', { method: 'POST', body: JSON.stringify({ doc_id: state.docId, name, color }) });
    byId('lsMarkerName').value = '';
    await refreshMarkers(state.docId);
    // sélectionne le marqueur nouvellement créé
    const created = state.markers.find(x => x.name.toLowerCase() === name.toLowerCase());
    if (created) state.selectedMarker = created;
    renderMarkerPills();
    setStatus('Marqueur créé');
  } catch (e) {
    setStatus(e.message, { isError: true });
  }
}

async function applyColorToSelectedMarker() {
  if (!state.selectedMarker) return;
  const color = byId('lsQuickColor').value;
  try {
    setStatus('Mise à jour couleur…');
    await api(`/markers/${encodeURIComponent(state.selectedMarker.name)}?doc_id=${encodeURIComponent(state.docId)}`, { method: 'PUT', body: JSON.stringify({ color }) });
    await refreshMarkers(state.docId);
    setStatus('Couleur mise à jour');
  } catch (e) {
    setStatus(e.message, { isError: true });
  }
}

async function annotateSelection() {
  if (!state.docId) return;
  if (!state.selectedMarker) {
    setStatus('Sélectionnez un marqueur', { isError: true });
    return;
  }

  const sel = state.selection;
  if (sel.start == null || sel.end == null || sel.start === sel.end) {
    setStatus('Sélectionnez un passage à surligner', { isError: true });
    return;
  }

  const sent = sentenceForSpan(state.text, sel.start, sel.end);
  const payload = {
    mot: sel.text,
    marqueur: state.selectedMarker.name,
    couleur: state.selectedMarker.color,
    phrase: sent.sentence,
    index_phrase: sent.index,
    span_start: sel.start,
    span_end: sel.end,
  };

  try {
    setStatus('Surlignage…');
    await api(`/annotate?doc_id=${encodeURIComponent(state.docId)}&auto_expand=true&dedupe=true`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    await loadDocument(state.docId);
    setStatus('Surligné');
  } catch (e) {
    setStatus(e.message, { isError: true });
  }
}

async function eraseSelection() {
  if (!state.docId) return;
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
    setStatus('Effacé');
  } catch (e) {
    setStatus(e.message, { isError: true });
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  const root = document.querySelector('.ls-shell');
  const docId = (root && root.getAttribute('data-doc-id')) || '';
  if (!docId) {
    window.location.assign('/labelling');
    return;
  }

  setDisabled('lsRefreshBtn', true);
  setDisabled('lsAnnotateBtn', true);
  setDisabled('lsEraseBtn', true);
  setDisabled('lsSaveTextBtn', true);
  setDisabled('lsApplyColorBtn', true);

  byId('lsAddMarkerBtn').onclick = addMarker;
  byId('lsApplyColorBtn').onclick = applyColorToSelectedMarker;
  byId('lsAnnotateBtn').onclick = annotateSelection;
  byId('lsEraseBtn').onclick = eraseSelection;
  const addCommentBtn = byId('lsAddCommentBtn');
  if (addCommentBtn) addCommentBtn.onclick = addCommentForSelection;
  byId('lsSaveTextBtn').onclick = saveText;
  byId('lsRefreshBtn').onclick = () => state.docId && loadDocument(state.docId);
  byId('lsCopyCsvBtn').onclick = async () => {
    const ta = byId('lsCsvText');
    if (!ta) return;
    try {
      await navigator.clipboard.writeText(ta.value || '');
      setStatus('CSV copié');
    } catch (e) {
      setStatus('Copie CSV impossible', { isError: true });
    }
  };

  byId('lsDownloadCsvBtn').onclick = () => {
    (async () => {
      const textarea = byId('lsText');
      if (!textarea) return;
      const isModified = (textarea.value || '') !== (textarea.dataset.lastSaved || '');
      if (isModified) {
        setStatus('Impossible: enregistrez le texte d’abord', { isError: true });
        return;
      }
      if (!state.docId) return;

      try {
        setStatus('Export CSV…');
        const url = `${API_BASE}/export/csv_view?doc_id=${encodeURIComponent(state.docId)}`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

        const blob = await resp.blob();
        const cd = resp.headers.get('content-disposition') || '';
        const match = cd.match(/filename\*=UTF-8''([^;]+)|filename=([^;]+)/i);
        let filename = '';
        if (match) {
          filename = decodeURIComponent((match[1] || match[2] || '').trim().replace(/^"|"$/g, ''));
        }
        if (!filename) {
          const safeId = String(state.docId || 'document').replace(/[^a-zA-Z0-9._-]+/g, '_');
          filename = `${safeId}_annotations_view.csv`;
        }

        const blobUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = blobUrl;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(blobUrl);
        setStatus('CSV exporté');
      } catch (e) {
        setStatus(e.message || 'Export CSV impossible', { isError: true });
      }
    })();
  };

  const textarea = byId('lsText');
  const updateSel = () => {
    if (!textarea) return;
    const s = textarea.selectionStart;
    const e = textarea.selectionEnd;
    const txt = (s != null && e != null && e > s) ? textarea.value.slice(s, e) : '';
    state.selection = { start: s, end: e, text: txt };
    updateSelectionInfo();
    const isModified = (textarea.value || '') !== (textarea.dataset.lastSaved || '');
    const enable = !!(txt && txt.trim().length) && !isModified;
    setDisabled('lsAnnotateBtn', !enable);
    setDisabled('lsEraseBtn', !enable);
    setDisabled('lsAddCommentBtn', !enable);
  };

  if (textarea) {
    textarea.addEventListener('mouseup', updateSel);
    textarea.addEventListener('keyup', updateSel);
    textarea.addEventListener('scroll', () => {
      const overlay = byId('lsOverlay');
      if (!overlay) return;
      overlay.scrollTop = textarea.scrollTop;
      overlay.scrollLeft = textarea.scrollLeft;
    });
    textarea.addEventListener('input', () => {
      const last = textarea.dataset.lastSaved || '';
      setDisabled('lsSaveTextBtn', (textarea.value === last) || !state.docId);
      renderOverlay();
    });
  }

  try {
    setStatus('Initialisation…');
    state.docId = docId;
    await refreshMarkers(docId);
    await loadDocument(docId);
    setStatus('Prêt');
  } catch (e) {
    setStatus(e.message, { isError: true });
  }
});
