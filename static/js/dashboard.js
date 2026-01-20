/**
 * Quantix+ Dashboard Avancé — version condensée
 * Interface interactive type Palantir/Excel pour l’analyse et le nettoyage d’un dataset unique.
 */

document.addEventListener('DOMContentLoaded', () => {
    let currentFile = null, currentData = null;
    let rightFile = null, rightData = null;
    const selectedOps = new Set();

    const isHttpContext = () => {
        try {
            const proto = String(window.location && window.location.protocol || '').toLowerCase();
            return proto === 'http:' || proto === 'https:';
        } catch {
            return false;
        }
    };

    const baseOrigin = (() => {
        try {
            const o = String(window.location && window.location.origin || '');
            return (o && o !== 'null') ? o : '';
        } catch {
            return '';
        }
    })();

    const apiUrl = (path) => {
        // path attendu: "/..."
        if (!path) return path;
        if (/^https?:\/\//i.test(path)) return path;
        if (baseOrigin) return baseOrigin + path;
        return path;
    };

    // État table (filtres / recherche)
    const tableState = {
        q: '',
        filters: {}, // { col: { op, value } }
        limit: 50,
    };

    const $ = id => document.getElementById(id);
    const els = {
        fileInput: $('fileInput'),
        uploadArea: $('uploadArea'),
        fileInfo: $('fileInfo'),
        fileName: $('fileName'),
        currentFileName: $('currentFileName'),

        fileInputRight: $('fileInputRight'),
        uploadAreaRight: $('uploadAreaRight'),
        fileInfoRight: $('fileInfoRight'),
        fileNameRight: $('fileNameRight'),
        mergePanel: $('mergePanel'),
        mergeHow: $('mergeHow'),
        mergeLeftKey: $('mergeLeftKey'),
        mergeRightKey: $('mergeRightKey'),
        mergeBtn: $('mergeBtn'),

        stats: { rows: $('rowCount'), cols: $('colCount'), issues: $('issueCount') },
        preview: $('previewContent'),
        queue: $('operationsQueue'),
        queueContent: $('queueContent'),
        execBtn: $('executeBtn'),
        resetBtn: $('resetBtn'),
        overlay: $('loadingOverlay'),
        msg: $('loadingMessage'),
        results: $('resultsModal'),
        resultsContent: $('resultsContent'),
        download: $('downloadResult'),

        tableSearch: $('tableSearch'),
        filterPopover: $('columnFilterPopover'),
        excelModal: $('excelModal'),
        excelModalTitle: $('excelModalTitle'),
        excelModalBody: $('excelModalBody'),
        excelModalSubmit: $('excelModalSubmit'),

        descriptiveBtn: $('btnDescriptiveStats'),
        resultsTitle: $('resultsTitle'),
        applyFilteredRows: $('applyFilteredRows'),
        targetColumns: $('targetColumns'),
    };

        function parseColumnsInput(raw) {
            const s = String(raw || '').trim();
            if (!s) return [];
            return s.split(',').map(x => String(x).trim()).filter(Boolean);
        }

    const escapeAttr = (v) => {
        if (typeof CSS !== 'undefined' && CSS.escape) return CSS.escape(v);
        return String(v).replace(/\\/g, '\\\\').replace(/"/g, '\\"');
    };

    const escapeHtmlAttr = (v) => String(v)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');

    // ========= Notifications et chargement =========
    function notify(msg, type = 'info') {
        const div = document.createElement('div');
        const color = { info: '#3b82f6', success: '#10b981', error: '#ef4444' }[type];
        div.textContent = msg;
        Object.assign(div.style, {
            position: 'fixed', top: '20px', right: '20px', padding: '12px 20px',
            background: color, color: '#fff', borderRadius: '8px',
            fontFamily: 'Inter,sans-serif', zIndex: 10000, transition: 'transform .3s',
            transform: 'translateX(100%)'
        });
        document.body.appendChild(div);
        setTimeout(() => div.style.transform = 'translateX(0)', 10);
        setTimeout(() => div.remove(), 3000);
    }

    if (!isHttpContext()) {
        notify("Cette page doit être ouverte via le serveur (http://127.0.0.1:5002/dashboard) — pas en ouvrant le fichier HTML directement.", 'error');
        // On ne bind pas les handlers upload/fetch en contexte non HTTP(S).
        return;
    }
    const loading = (msg = null) => { els.overlay.style.display = msg ? 'flex' : 'none'; if (msg) els.msg.textContent = msg; };
    const api = (url, data, msg, cb) => {
        loading(msg);
        return fetch(apiUrl(url), { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) })
            .then(r => r.json())
            .then(j => j.success ? cb(j) : notify(j.error, 'error'))
            .catch(e => {
                const m = String(e && e.message || e);
                if (m.toLowerCase().includes('expected pattern')) {
                    notify("Import impossible: l’URL n’est pas valide. Ouvre le Dashboard via l’adresse du serveur (ex: http://127.0.0.1:5002/dashboard).", 'error');
                    return;
                }
                notify('Erreur : ' + m, 'error');
            })
            .finally(() => loading());
    };

    // ========= Helpers table =========
    const isMissingValue = (v) => {
        if (v === null || typeof v === 'undefined') return true;
        if (typeof v === 'number' && Number.isNaN(v)) return true;
        const s = String(v);
        if (!s) return true;
        const t = s.trim();
        if (t === '') return true;
        const low = t.toLowerCase();
        return (low === 'null' || low === 'none' || low === 'nan' || low === 'na' || low === 'n/a');
    };

    const escapeHtml = (s) => String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');

    function renderCellValue(v) {
        if (isMissingValue(v)) {
            return '<span class="cell-missing" title="Valeur manquante">∅</span>';
        }
        const s = String(v);
        const clipped = s.length > 60 ? s.slice(0, 60) + '…' : s;
        return escapeHtml(clipped);
    }

    // ========= Gestion fichier =========
    function handleFile(f, slot = 'primary') {
        const allowed = ['csv', 'xlsx', 'xls', 'json', 'txt'];
        const ext = f.name.split('.').pop().toLowerCase();
        if (!allowed.includes(ext)) return notify('Format non supporté', 'error');

        if (slot === 'right') {
            rightFile = f;
            els.fileNameRight.textContent = f.name;
            els.uploadAreaRight.style.display = 'none';
            els.fileInfoRight.style.display = 'flex';
        } else {
            currentFile = f;
            els.fileName.textContent = f.name;
            els.currentFileName.textContent = f.name;
            els.uploadArea.style.display = 'none';
            els.fileInfo.style.display = 'flex';
        }

        const fd = new FormData();
        fd.append('file', f);
        fd.append('slot', slot);
        loading('Upload + analyse...');
        fetch(apiUrl('/upload-file'), { method: 'POST', credentials: 'same-origin', body: fd })
            .then(r => r.json())
            .then(j => {
                if (!j.success) {
                    notify(j.error || 'Upload impossible', 'error');
                    return;
                }
                return analyzeSlot(slot);
            })
            .catch(e => {
                const m = String(e && e.message || e);
                if (m.toLowerCase().includes('expected pattern')) {
                    notify("Import impossible: l’URL n’est pas valide. Ouvre le Dashboard via l’adresse du serveur (ex: http://127.0.0.1:5002/dashboard).", 'error');
                    return;
                }
                notify('Erreur : ' + m, 'error');
            })
            .finally(() => loading());
    }

    // Empêche le navigateur d'ouvrir le fichier lors d'un drop en dehors des zones.
    ['dragover', 'drop'].forEach((ev) => {
        document.addEventListener(ev, (e) => {
            if (e.dataTransfer && e.dataTransfer.types && Array.from(e.dataTransfer.types).includes('Files')) {
                e.preventDefault();
            }
        });
    });

    els.uploadArea.addEventListener('click', () => els.fileInput.click());
    els.uploadArea.addEventListener('dragover', e => { e.preventDefault(); els.uploadArea.classList.add('active'); });
    els.uploadArea.addEventListener('dragleave', e => els.uploadArea.classList.remove('active'));
    els.uploadArea.addEventListener('drop', e => { e.preventDefault(); els.uploadArea.classList.remove('active'); e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0], 'primary'); });
    els.fileInput.addEventListener('change', e => e.target.files[0] && handleFile(e.target.files[0], 'primary'));

    els.uploadAreaRight?.addEventListener('click', () => els.fileInputRight?.click());
    els.uploadAreaRight?.addEventListener('dragover', e => { e.preventDefault(); els.uploadAreaRight.classList.add('active'); });
    els.uploadAreaRight?.addEventListener('dragleave', e => els.uploadAreaRight.classList.remove('active'));
    els.uploadAreaRight?.addEventListener('drop', e => { e.preventDefault(); els.uploadAreaRight.classList.remove('active'); e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0], 'right'); });
    els.fileInputRight?.addEventListener('change', e => e.target.files[0] && handleFile(e.target.files[0], 'right'));

    function analyzeSlot(slot = 'primary') {
        return fetch(apiUrl(`/api/dashboard/analyze?slot=${encodeURIComponent(slot)}`), { credentials: 'same-origin' })
            .then(r => r.json())
            .then(j => {
                if (!j.success) {
                    notify(j.error || 'Analyse impossible', 'error');
                    return;
                }
                renderAnalysis(j.analysis, slot);
                return j;
            })
            .catch(e => {
                const m = String(e && e.message || e);
                if (m.toLowerCase().includes('expected pattern')) {
                    notify("Import impossible: l’URL n’est pas valide. Ouvre le Dashboard via l’adresse du serveur (ex: http://127.0.0.1:5002/dashboard).", 'error');
                    return;
                }
                notify('Erreur : ' + m, 'error');
            });
    }

    // ========= Rendu analyse =========
    function renderAnalysis(a, slot = 'primary') {
        if (slot === 'right') {
            rightData = a;
            els.mergePanel.style.display = 'block';
            hydrateMergeSelectors();
            notify('Dataset secondaire prêt', 'success');
            return;
        }

        currentData = a;
        els.stats.rows.textContent = a.shape[0];
        els.stats.cols.textContent = a.shape[1];
        const issues = Object.values(a.missing_values || {}).reduce((s, v) => s + v, 0) + (a.duplicates || 0);
        els.stats.issues.textContent = issues;
        renderTable(a.preview);
        highlightOps(a.suggestions);
        els.execBtn.disabled = !selectedOps.size || !currentFile;

        if (els.descriptiveBtn) {
            els.descriptiveBtn.disabled = false;
        }
    }

        // ========= Stats descriptives =========
        function setModalTitle(html) {
                if (els.resultsTitle) {
                        els.resultsTitle.innerHTML = html;
                }
        }

        function fmtNum(v) {
                if (v === null || typeof v === 'undefined') return '—';
                const n = Number(v);
                if (!Number.isFinite(n)) return '—';
                const abs = Math.abs(n);
                if (abs >= 1e6 || (abs > 0 && abs < 1e-4)) return n.toExponential(3);
                // 4 décimales max, sans trailing zeros
                return n.toFixed(4).replace(/\.0+$/, '').replace(/(\.[0-9]*?)0+$/, '$1');
        }

        function renderDescriptiveStatsTable(summary) {
                const cols = Array.isArray(summary?.columns) ? summary.columns : [];
                const numeric = cols.filter(c => c && (typeof c.mean !== 'undefined' || typeof c.median !== 'undefined' || typeof c.p25 !== 'undefined'));

                if (!numeric.length) {
                        return `<p>Aucune colonne numérique détectée pour calculer moyenne/médiane/quantiles.</p>`;
                }

                const rows = numeric.map(c => {
                        const name = escapeHtml(c.column || '');
                        const dtype = escapeHtml(c.dtype || '');
                        const missing = (typeof c.missing === 'number') ? c.missing : 0;
                        return `
                            <tr>
                                <td><b>${name}</b><div style="opacity:.75;font-size:12px;">${dtype}</div></td>
                                <td>${missing}</td>
                                <td>${fmtNum(c.mean)}</td>
                                <td>${fmtNum(c.median)}</td>
                                <td>${fmtNum(c.p25)}</td>
                                <td>${fmtNum(c.p75)}</td>
                                <td>${fmtNum(c.min)}</td>
                                <td>${fmtNum(c.max)}</td>
                                <td>${fmtNum(c.std)}</td>
                            </tr>
                        `;
                }).join('');

                return `
                    <div class="results-summary">
                        <h4>Dataset</h4>
                        <div class="stats-grid">
                            <div><b>Fichier</b><p>${escapeHtml(summary.filename || 'dataset en session')}</p></div>
                            <div><b>Lignes</b><p>${summary.rows ?? '—'}</p></div>
                            <div><b>Colonnes</b><p>${summary.cols ?? '—'}</p></div>
                        </div>
                    </div>
                    <h4>Colonnes numériques</h4>
                    <div style="overflow:auto; max-height: 55vh;">
                        <table>
                            <thead>
                                <tr>
                                    <th>Colonne</th>
                                    <th>Manquants</th>
                                    <th>Moyenne</th>
                                    <th>Médiane</th>
                                    <th>Q1 (25%)</th>
                                    <th>Q3 (75%)</th>
                                    <th>Min</th>
                                    <th>Max</th>
                                    <th>Écart-type</th>
                                </tr>
                            </thead>
                            <tbody>${rows}</tbody>
                        </table>
                    </div>
                `;
        }

        if (els.descriptiveBtn) {
                els.descriptiveBtn.addEventListener('click', () => {
                        if (!currentData) {
                                notify('Chargez un fichier d’abord.', 'error');
                                return;
                        }
                        loading('Calcul des stats descriptives...');
                        fetch(apiUrl('/api/dashboard/summary'), { credentials: 'same-origin' })
                                .then(r => r.json())
                                .then(j => {
                                        if (!j.success) {
                                                notify(j.error || 'Erreur stats descriptives', 'error');
                                                return;
                                        }
                                        setModalTitle('<i class="fas fa-chart-bar"></i> Statistiques descriptives');
                                        els.resultsContent.innerHTML = renderDescriptiveStatsTable(j);

                                        if (els.download) {
                                                els.download.onclick = null;
                                                els.download.style.display = 'none';
                                        }
                                        els.results.style.display = 'flex';
                                })
                                .catch(e => notify('Erreur : ' + String(e && e.message || e), 'error'))
                                .finally(() => loading());
                });
        }

    function toFiltersPayload() {
        const arr = [];
        Object.keys(tableState.filters || {}).forEach((col) => {
            const f = tableState.filters[col];
            if (!f || !f.op) return;
            arr.push({ column: col, op: f.op, value: f.value });
        });
        return arr;
    }

    function refreshPreviewFiltered() {
        if (!currentFile) return;
        loading('Chargement de l\'aperçu...');
        fetch(apiUrl('/api/dashboard/preview'), {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                slot: 'primary',
                limit: tableState.limit,
                q: tableState.q,
                filters: toFiltersPayload(),
            })
        })
            .then(r => r.json())
            .then(j => {
                if (!j.success) {
                    notify(j.error || 'Preview impossible', 'error');
                    return;
                }
                renderTable(j.rows);
            })
            .catch(e => {
                const m = String(e && e.message || e);
                if (m.toLowerCase().includes('expected pattern')) {
                    notify("Import impossible: l’URL n’est pas valide. Ouvre le Dashboard via l’adresse du serveur (ex: http://127.0.0.1:5002/dashboard).", 'error');
                    return;
                }
                notify('Erreur : ' + m, 'error');
            })
            .finally(() => loading());
    }

    function renderTable(rows) {
        if (!rows?.length) return els.preview.innerHTML = '<p>Aucune donnée</p>';
        const allCols = Object.keys(rows[0]);
        const cols = allCols.filter(c => c !== '_row');

        let html = '<table class="data-table"><thead><tr>' + cols.map(c => {
            const active = tableState.filters[c] ? 'active' : '';
            return `<th><div class="th-wrap"><span class="th-label" title="${escapeHtml(c)}">${escapeHtml(c)}</span>` +
                                `<button class="th-filter ${active}" type="button" data-col="${escapeHtmlAttr(c)}" title="Filtrer">
                  <i class="fas fa-filter"></i>
                </button></div></th>`;
        }).join('') + '</tr></thead><tbody>';

        rows.slice(0, 50).forEach(r => {
            const rowIndex = (typeof r._row !== 'undefined' && r._row !== null) ? Number(r._row) : null;
            html += `<tr data-row="${rowIndex !== null && !Number.isNaN(rowIndex) ? rowIndex : ''}">` + cols.map(c => {
                const v = r[c];
                return `<td data-col="${escapeHtmlAttr(c)}">${renderCellValue(v)}</td>`;
            }).join('') + '</tr>';
        });
        els.preview.innerHTML = html + '</tbody></table>';
    }

    const highlightOps = list => {
        document.querySelectorAll('.operation-btn').forEach(b => b.classList.remove('suggested'));
        list?.forEach(s => {
            try {
                const sel = `[data-operation="${escapeAttr(s.recommended_action)}"]`;
                document.querySelector(sel)?.classList.add('suggested');
            } catch (e) {
                console.warn('Skipping invalid selector for recommended_action:', s.recommended_action, e);
            }
        });
    };

    // ========= Opérations =========
    document.querySelectorAll('.operation-btn').forEach(b => b.addEventListener('click', () => toggleOp(b)));
    const toggleOp = b => {
        const op = b.dataset.operation;
        selectedOps.has(op) ? selectedOps.delete(op) : selectedOps.add(op);
        b.classList.toggle('active');
        renderQueue();
    };

    function renderQueue() {
        els.queue.style.display = selectedOps.size ? 'block' : 'none';
        els.queueContent.innerHTML = [...selectedOps].map(op => {
            const btn = (() => {
                try { return document.querySelector(`[data-operation="${escapeAttr(op)}"]`); }
                catch(e) { console.warn('Invalid selector for op', op, e); return null; }
            })();
            const label = btn ? btn.querySelector('span')?.textContent : op;
            return `<div class="queue-item"><span>${label || op}</span>
                <button type="button" onclick="removeOp('${op}')" aria-label="Retirer">&times;</button></div>`;
        }).join('');
        els.execBtn.disabled = !selectedOps.size || !currentFile;
    }

    window.removeOp = op => {
        selectedOps.delete(op);
        try { document.querySelector(`[data-operation="${escapeAttr(op)}"]`)?.classList.remove('active'); } catch(e){ console.warn('removeOp selector error', op, e); }
        renderQueue();
    };

    els.execBtn.addEventListener('click', () => {
        if (!selectedOps.size) return;
        api('/api/dashboard/operations', { operations: [...selectedOps], replace_session_file: true, preview_rows: 20 }, 'Exécution...', (j) => {
            showResults(j);
            analyzeSlot('primary');
        });
    });

    // ========= Recherche (aperçu) =========
    let searchTimer = null;
    els.tableSearch?.addEventListener('input', (e) => {
        tableState.q = String(e.target.value || '').trim();
        if (searchTimer) clearTimeout(searchTimer);
        searchTimer = setTimeout(() => refreshPreviewFiltered(), 250);
    });

    // ========= Filtre colonnes =========
    function closeFilterPopover() {
        if (!els.filterPopover) return;
        els.filterPopover.style.display = 'none';
        els.filterPopover.innerHTML = '';
    }

    function openFilterPopover(buttonEl, col) {
        if (!els.filterPopover) return;
        const rect = buttonEl.getBoundingClientRect();
        const existing = tableState.filters[col] || { op: 'contains', value: '' };

        els.filterPopover.innerHTML = `
            <div class="fp-title">Filtrer: ${escapeHtml(col)}</div>
            <label>Opérateur</label>
            <select id="fp-op">
              <option value="contains">Contient</option>
              <option value="equals">Égal à</option>
              <option value="regex">Regex</option>
              <option value="isnull">Est vide (NA)</option>
              <option value="notnull">N'est pas vide</option>
              <option value=">">&gt;</option>
              <option value=">=">&gt;=</option>
              <option value="<">&lt;</option>
              <option value="<=">&lt;=</option>
            </select>
            <label>Valeur</label>
            <input id="fp-value" type="text" placeholder="ex: ^abc.*" />
            <div class="fp-actions">
              <button type="button" id="fp-clear">Réinitialiser</button>
              <button type="button" class="primary" id="fp-apply">Appliquer</button>
            </div>
        `;
        els.filterPopover.style.left = Math.min(rect.left, window.innerWidth - 340) + 'px';
        els.filterPopover.style.top = (rect.bottom + 8) + 'px';
        els.filterPopover.style.display = 'block';

        const opSel = els.filterPopover.querySelector('#fp-op');
        const valInp = els.filterPopover.querySelector('#fp-value');
        if (opSel) opSel.value = existing.op || 'contains';
        if (valInp) valInp.value = existing.value || '';

        const syncDisabled = () => {
            const op = (opSel?.value || 'contains');
            if (!valInp) return;
            valInp.disabled = (op === 'isnull' || op === 'notnull');
            if (valInp.disabled) valInp.value = '';
        };
        opSel?.addEventListener('change', syncDisabled);
        syncDisabled();

        els.filterPopover.querySelector('#fp-clear')?.addEventListener('click', () => {
            delete tableState.filters[col];
            closeFilterPopover();
            refreshPreviewFiltered();
        });
        els.filterPopover.querySelector('#fp-apply')?.addEventListener('click', () => {
            const op = (opSel?.value || 'contains');
            const value = valInp?.value || '';
            tableState.filters[col] = { op, value };
            closeFilterPopover();
            refreshPreviewFiltered();
        });
    }

    document.addEventListener('click', (e) => {
        const btn = e.target.closest && e.target.closest('.th-filter');
        if (btn && btn.dataset && btn.dataset.col) {
            const col = btn.dataset.col;
            e.preventDefault();
            e.stopPropagation();
            openFilterPopover(btn, col);
            return;
        }
        const inside = els.filterPopover && els.filterPopover.contains(e.target);
        if (!inside) closeFilterPopover();
    });

    // ========= Édition inline (double-clic) =========
    function startCellEdit(td) {
        if (!td || td.classList.contains('editing')) return;
        const tr = td.closest('tr');
        const row = tr ? tr.getAttribute('data-row') : null;
        const col = td.getAttribute('data-col');
        if (!row || !col || row === '') return;

        const currentText = td.textContent === '∅' ? '' : td.textContent;
        td.classList.add('editing');
        td.innerHTML = `<input class="cell-editor" type="text" />`;
        const input = td.querySelector('input');
        input.value = currentText;
        input.focus();
        input.select();

        const cancel = () => {
            td.classList.remove('editing');
            td.innerHTML = renderCellValue(currentText);
        };
        const save = () => {
            const newVal = input.value;
            loading('Sauvegarde...');
            fetch(apiUrl('/api/dashboard/edit-cell'), {
                method: 'POST',
                credentials: 'same-origin',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ row: Number(row), column: col, value: newVal, slot: 'primary' })
            })
                .then(r => r.json())
                .then(j => {
                    if (!j.success) {
                        notify(j.error || 'Édition impossible', 'error');
                        cancel();
                        return;
                    }
                    td.classList.remove('editing');
                    td.innerHTML = renderCellValue(newVal);
                    // rafraîchir stats (et ré-appliquer filtres si nécessaire)
                    analyzeSlot('primary').then(() => {
                        if (tableState.q || Object.keys(tableState.filters).length) refreshPreviewFiltered();
                    });
                })
                .catch(e => { notify('Erreur : ' + e.message, 'error'); cancel(); })
                .finally(() => loading());
        };

        input.addEventListener('keydown', (ev) => {
            if (ev.key === 'Escape') {
                ev.preventDefault();
                cancel();
            }
            if (ev.key === 'Enter') {
                ev.preventDefault();
                save();
            }
        });
        input.addEventListener('blur', () => save());
    }

    els.preview?.addEventListener('dblclick', (e) => {
        const td = e.target.closest && e.target.closest('td[data-col]');
        if (td) startCellEdit(td);
    });

    // ========= Modal Outils Excel =========
    function openExcelModal(title, bodyHtml, onSubmit) {
        if (!els.excelModal) return;
        els.excelModalTitle.textContent = title;
        els.excelModalBody.innerHTML = bodyHtml;
        els.excelModal.style.display = 'flex';
        const btn = els.excelModalSubmit;
        btn.onclick = () => onSubmit && onSubmit();
    }

    window.closeExcelModal = () => {
        if (!els.excelModal) return;
        els.excelModal.style.display = 'none';
        els.excelModalBody.innerHTML = '';
    };

    const getColumns = () => (currentData && currentData.columns) ? currentData.columns : [];
    const optionsForColumns = () => getColumns().map(c => `<option value="${String(c).replace(/"/g, '&quot;')}">${escapeHtml(c)}</option>`).join('');

    function runExcelAction(url, payload, successMsg) {
        api(url, payload, 'Application...', (j) => {
            notify(successMsg || 'Appliqué', 'success');
            window.closeExcelModal();
            analyzeSlot('primary').then(() => {
                if (tableState.q || Object.keys(tableState.filters).length) refreshPreviewFiltered();
            });
        });
    }

    document.querySelectorAll('.excel-action-btn').forEach((b) => {
        b.addEventListener('click', () => {
            if (!currentFile) return notify('Charge un fichier d\'abord', 'error');
            const act = b.dataset.action;
            if (act === 'rename-column') {
                openExcelModal('Renommer une colonne', `
                    <div class="field"><label>Colonne</label><select id="rn-old">${optionsForColumns()}</select></div>
                    <div class="field"><label>Nouveau nom</label><input id="rn-new" type="text" placeholder="ex: score_total"/></div>
                `, () => {
                    const old = document.getElementById('rn-old').value;
                    const neu = document.getElementById('rn-new').value;
                    runExcelAction('/api/dashboard/rename-column', { old, new: neu }, 'Colonne renommée');
                });
            }
            if (act === 'replace-regex') {
                openExcelModal('Remplacer (regex)', `
                    <div class="field"><label>Colonne</label><select id="rr-col">${optionsForColumns()}</select></div>
                    <div class="field"><label>Pattern</label><input id="rr-pattern" type="text" placeholder="ex: \\s+"/></div>
                    <div class="field"><label>Remplacement</label><input id="rr-repl" type="text" placeholder="ex: _"/></div>
                    <div class="field"><label>Mode</label><select id="rr-mode"><option value="regex">Regex</option><option value="text">Texte (littéral)</option></select></div>
                `, () => {
                    const column = document.getElementById('rr-col').value;
                    const pattern = document.getElementById('rr-pattern').value;
                    const repl = document.getElementById('rr-repl').value;
                    const mode = document.getElementById('rr-mode').value;
                    runExcelAction('/api/dashboard/replace-regex', { column, pattern, repl, regex: mode === 'regex' }, 'Remplacement effectué');
                });
            }
            if (act === 'derive-column') {
                openExcelModal('Créer une colonne', `
                    <div class="field"><label>Type</label>
                      <select id="dc-kind">
                        <option value="concat">Concaténer (texte)</option>
                        <option value="numeric_op">Opération (numérique)</option>
                        <option value="regex_extract">Extraction (regex)</option>
                      </select>
                    </div>
                    <div id="dc-fields"></div>
                    <div class="field"><label>Nom de la nouvelle colonne</label><input id="dc-new" type="text" placeholder="ex: score_x2"/></div>
                `, () => {
                    const kind = document.getElementById('dc-kind').value;
                    const new_name = document.getElementById('dc-new').value;
                    const payload = { kind, new_name };
                    if (kind === 'concat') {
                        payload.col_a = document.getElementById('dc-a').value;
                        payload.col_b = document.getElementById('dc-b').value;
                        payload.sep = document.getElementById('dc-sep').value;
                    }
                    if (kind === 'numeric_op') {
                        payload.col_a = document.getElementById('dc-a').value;
                        payload.col_b = document.getElementById('dc-b').value;
                        payload.op = document.getElementById('dc-op').value;
                    }
                    if (kind === 'regex_extract') {
                        payload.source = document.getElementById('dc-src').value;
                        payload.pattern = document.getElementById('dc-pattern').value;
                        payload.group = Number(document.getElementById('dc-group').value || '1');
                    }
                    runExcelAction('/api/dashboard/derive-column', payload, 'Colonne créée');
                });

                const setFields = () => {
                    const kind = document.getElementById('dc-kind').value;
                    const wrap = document.getElementById('dc-fields');
                    if (!wrap) return;
                    if (kind === 'concat') {
                        wrap.innerHTML = `
                          <div class="field"><label>Colonne A</label><select id="dc-a">${optionsForColumns()}</select></div>
                          <div class="field"><label>Colonne B</label><select id="dc-b">${optionsForColumns()}</select></div>
                          <div class="field"><label>Séparateur</label><input id="dc-sep" type="text" placeholder="ex:  -  "/></div>
                        `;
                    } else if (kind === 'numeric_op') {
                        wrap.innerHTML = `
                          <div class="field"><label>Colonne A</label><select id="dc-a">${optionsForColumns()}</select></div>
                          <div class="field"><label>Opération</label><select id="dc-op"><option value="+">+</option><option value="-">-</option><option value="*">*</option><option value="/">/</option></select></div>
                          <div class="field"><label>Colonne B</label><select id="dc-b">${optionsForColumns()}</select></div>
                        `;
                    } else {
                        wrap.innerHTML = `
                          <div class="field"><label>Colonne source</label><select id="dc-src">${optionsForColumns()}</select></div>
                          <div class="field"><label>Regex</label><input id="dc-pattern" type="text" placeholder="ex: (\\d+)"/></div>
                          <div class="field"><label>Groupe</label><input id="dc-group" type="number" min="1" value="1"/></div>
                        `;
                    }
                };
                document.getElementById('dc-kind').addEventListener('change', setFields);
                setFields();
            }
        });
    });

    function hydrateMergeSelectors() {
        if (!currentData?.columns?.length || !rightData?.columns?.length) {
            els.mergeBtn.disabled = true;
            return;
        }
        const mkOptions = (cols) => cols.map(c => `<option value="${String(c).replace(/"/g, '&quot;')}">${c}</option>`).join('');
        els.mergeLeftKey.innerHTML = mkOptions(currentData.columns);
        els.mergeRightKey.innerHTML = mkOptions(rightData.columns);
        els.mergeBtn.disabled = false;
    }

    els.mergeBtn?.addEventListener('click', () => {
        if (!currentData || !rightData) return;
        const how = els.mergeHow.value;
        const left_on = els.mergeLeftKey.value;
        const right_on = els.mergeRightKey.value;
        api('/api/dashboard/merge', { how, left_on, right_on }, 'Fusion en cours...', (j) => {
            notify(j.message || 'Fusion terminée', 'success');
            els.currentFileName.textContent = j.output_file || 'Dataset fusionné';
            analyzeSlot('primary');
        });
    });

    // ========= Résultats =========
    function showResults(r) {
                setModalTitle('<i class="fas fa-check-circle"></i> Opérations terminées');
        let html = `
        <div class="results-summary">
          <h4>Résumé</h4>
          <div class="stats-grid">
            <div><b>Lignes</b><p>${r.stats.lignes_finales}</p></div>
            <div><b>Colonnes</b><p>${r.stats.colonnes_finales}</p></div>
            <div><b>Succès</b><p>${r.stats.operations_executees}</p></div>
          </div></div><h4>Détail</h4>`;
        r.operations_results.forEach(o => {
            const status = o.success ? 'OK' : 'ERREUR';
            const msg = o.success ? o.message : o.error;
            html += `<div><b>${status}</b> ${o.operation} — ${msg}</div>`;
        });
        if (r.preview?.length) {
            html += '<h4>Aperçu</h4>' + renderMiniTable(r.preview);
        }
        els.resultsContent.innerHTML = html;
        els.download.onclick = () => r.download_url && window.open(r.download_url);
        if (els.download) {
            els.download.style.display = '';
        }
        els.results.style.display = 'flex';
        resetOps();
    }

    const renderMiniTable = rows => {
        const cols = Object.keys(rows[0]);
        let h = '<table><thead><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
        rows.slice(0, 5).forEach(r => h += '<tr>' + cols.map(c => `<td>${r[c]}</td>`).join('') + '</tr>');
        return h + '</tbody></table>';
    };

    window.closeResultsModal = () => els.results.style.display = 'none';

    // ========= Reset =========
    els.resetBtn.addEventListener('click', resetAll);
    function resetAll() {
        currentFile = null; currentData = null;
        rightFile = null; rightData = null;
        tableState.q = '';
        tableState.filters = {};
        if (els.tableSearch) els.tableSearch.value = '';
        if (els.filterPopover) { els.filterPopover.style.display = 'none'; els.filterPopover.innerHTML = ''; }
        if (els.excelModal) { els.excelModal.style.display = 'none'; }
        els.uploadArea.style.display = 'block';
        els.fileInfo.style.display = 'none';
        els.fileInput.value = '';
        if (els.uploadAreaRight) els.uploadAreaRight.style.display = 'block';
        if (els.fileInfoRight) els.fileInfoRight.style.display = 'none';
        if (els.fileInputRight) els.fileInputRight.value = '';
        if (els.mergePanel) els.mergePanel.style.display = 'none';
        Object.values(els.stats).forEach(e => e.textContent = '0');
        els.preview.innerHTML = '<p>Aucun fichier chargé</p>';
        if (els.descriptiveBtn) {
            els.descriptiveBtn.disabled = true;
        }
        resetOps();
    }
    function resetOps() {
        selectedOps.clear();
        document.querySelectorAll('.operation-btn.active').forEach(b => b.classList.remove('active'));
        renderQueue();
    }

    // ========= Catégories (toggle) =========
    window.toggleCategory = cat => {
        try {
            const v = (typeof CSS !== 'undefined' && CSS.escape) ? CSS.escape(cat) : String(cat).replace(/\\/g,'\\\\').replace(/"/g,'\\"');
            document.querySelector(`[data-category="${v}"]`)?.classList.toggle('collapsed');
        } catch (e) {
            console.warn('toggleCategory invalid selector', cat, e);
        }
    };

    renderQueue(); // init
});
