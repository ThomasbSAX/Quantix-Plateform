/**
 * Quantix – DatasetManager condensé
 * Upload, analyse, comparaison, fusion de plusieurs jeux de données
 */
class DatasetManager {
    constructor() {
        this.datasets = [];
        this.selected = new Set();
        this.init();
    }

    /** === INITIALISATION === */
    init() {
        this.bindUI();
        this.fetchDatasets();
    }

    bindUI() {
        const $ = id => document.getElementById(id);

        $('#addDatasetBtn')?.addEventListener('click', () => this.promptUpload());
        $('#compareBtn')?.addEventListener('click', () => this.openModal('compare'));
        $('#mergeBtn')?.addEventListener('click', () => this.openModal('merge'));

        $('#additionalFileInput')?.addEventListener('change', e => e.target.files[0] && this.upload(e.target.files[0]));
        $('#runComparisonBtn')?.addEventListener('click', () => this.runComparison());
        $('#runMergeBtn')?.addEventListener('click', () => this.runMerge());
        $('#joinTypeSelect')?.addEventListener('change', e => this.updateJoinKeys());

        const drop = $('#additionalDropZone');
        if (drop) {
            ['dragover', 'drop'].forEach(ev => drop.addEventListener(ev, e => this.handleDrop(e)));
            drop.addEventListener('click', () => $('#additionalFileInput').click());
        }

        document.addEventListener('click', e => {
            if (e.target.classList.contains('modal') || e.target.classList.contains('close'))
                document.querySelectorAll('.modal').forEach(m => (m.style.display = 'none'));
        });
    }

    /** === UPLOAD ET LISTE === */
    async upload(file) {
        try {
            this.loading(`Upload de ${file.name}...`);
            const fd = new FormData();
            fd.append('file', file);
            const r = await fetch('/upload-additional-file', { method: 'POST', body: fd });
            const j = await r.json();
            if (!j.success) throw new Error(j.error);
            this.datasets.push(j.dataset);
            this.renderList();
            this.notify(`"${file.name}" ajouté`, 'success');
        } catch (e) {
            this.notify(e.message, 'error');
        } finally {
            this.loading(false);
        }
    }

    async fetchDatasets() {
        try {
            const r = await fetch('/get-datasets');
            const j = await r.json();
            if (j.success) {
                this.datasets = j.datasets;
                this.renderList();
            }
        } catch {}
    }

    renderList() {
        const c = document.getElementById('datasetsList');
        if (!c) return;
        c.innerHTML = '';
        this.datasets.forEach((d, i) => c.appendChild(this.makeItem(d, i)));
        this.updateButtons();
    }

    makeItem(d, i) {
        const div = document.createElement('div');
        div.className = `dataset-item ${this.selected.has(i) ? 'selected' : ''}`;
        div.innerHTML = `
        <div class="dataset-header">
          <h3>${d.name}</h3><span>${d.status}</span>
          <input type="checkbox" ${this.selected.has(i) ? 'checked' : ''}>
        </div>
        <div class="stats">
          <span>${d.rows || 0} lignes</span> | <span>${d.columns || 0} cols</span> | 
          <span>${this.fmtSize(d.size)}</span>
        </div>
        <div class="dataset-actions">
          <button onclick="datasetManager.analyze(${i})"><i class="fas fa-chart-line"></i></button>
          <button onclick="datasetManager.clean(${i})"><i class="fas fa-broom"></i></button>
          <button onclick="datasetManager.remove(${i})"><i class="fas fa-trash"></i></button>
        </div>`;
        div.querySelector('input').addEventListener('change', e => {
            e.target.checked ? this.selected.add(i) : this.selected.delete(i);
            this.updateButtons();
        });
        return div;
    }

    fmtSize(b) {
        const s = ['B', 'KB', 'MB', 'GB'];
        const i = b ? Math.floor(Math.log(b) / Math.log(1024)) : 0;
        return (b / 1024 ** i).toFixed(2) + ' ' + s[i];
    }

    /** === ACTIONS PRINCIPALES === */
    async analyze(i) {
        await this.api(`/analyze-dataset`, { dataset_index: i }, 'Analyse en cours...', res => {
            this.datasets[i].status = 'PROCESSED';
            this.renderList();
        });
    }

    async clean(i) {
        await this.api(`/clean-dataset`, { dataset_index: i }, 'Nettoyage...', res => {
            this.datasets[i].status = 'CLEANED';
            this.renderList();
        });
    }

    remove(i) {
        if (!confirm('Supprimer ce dataset ?')) return;
        this.datasets.splice(i, 1);
        this.selected.delete(i);
        this.renderList();
    }

    /** === COMPARAISON ET FUSION === */
    openModal(type) {
        const modal = document.getElementById(`${type}Modal`);
        if (!modal) return;
        const sel = modal.querySelectorAll('.dataset-select');
        sel.forEach(s => {
            s.innerHTML = '<option value="">— Sélectionner —</option>';
            this.datasets.forEach((d, i) => (s.innerHTML += `<option value="${i}">${d.name}</option>`));
        });
        modal.style.display = 'flex';
    }

    async runComparison() {
        const a = +document.getElementById('compareDataset1').value;
        const b = +document.getElementById('compareDataset2').value;
        if (isNaN(a) || isNaN(b)) return this.notify('Choisir deux datasets', 'error');
        await this.api('/compare-datasets', { dataset1_index: a, dataset2_index: b }, 'Comparaison...', res => {
            const r = res.comparison;
            const out = document.getElementById('comparisonResults');
            out.innerHTML = `
            <p>${r.common_columns.length} colonnes communes</p>
            <div>${r.common_columns.map(c => `<span>${c}</span>`).join('')}</div>
            <p>Compatibilité : ${r.merge_compatibility}% (${r.merge_recommendation})</p>`;
            out.style.display = 'block';
        });
    }

    async runMerge() {
        const a = +document.getElementById('mergeDataset1').value;
        const b = +document.getElementById('mergeDataset2').value;
        const join = document.getElementById('joinTypeSelect').value;
        const keys = [...document.querySelectorAll('.join-key-checkbox:checked')].map(x => x.value);
        if (!keys.length) return this.notify('Sélectionnez des clés', 'error');
        await this.api('/merge-datasets', {
            dataset1_index: a,
            dataset2_index: b,
            join_type: join,
            join_keys: keys
        }, 'Fusion...', res => {
            this.datasets.push(res.merged_dataset);
            this.renderList();
            this.hide('mergeModal');
        });
    }

    async updateJoinKeys() {
        const a = +document.getElementById('mergeDataset1').value;
        const b = +document.getElementById('mergeDataset2').value;
        if (isNaN(a) || isNaN(b)) return;
        const r = await fetch('/get-common-columns', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset1_index: a, dataset2_index: b })
        });
        const j = await r.json();
        if (!j.success) return;
        const c = document.getElementById('joinKeysList');
        c.innerHTML = j.common_columns
            .map(col => `<label><input type="checkbox" class="join-key-checkbox" value="${col}">${col}</label>`)
            .join('');
    }

    /** === UTILITAIRES === */
    async api(url, body, msg, onSuccess) {
        try {
            this.loading(msg);
            const r = await fetch(url, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
            });
            const j = await r.json();
            if (!j.success) throw new Error(j.error);
            onSuccess(j);
            this.notify('Opération réussie', 'success');
        } catch (e) {
            this.notify(e.message, 'error');
        } finally {
            this.loading(false);
        }
    }

    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        const f = e.dataTransfer?.files?.[0];
        f && this.upload(f);
    }

    loading(msg) {
        let l = document.getElementById('globalLoader');
        if (!msg) return l && (l.style.display = 'none');
        if (l) {
            l.querySelector('.loading-text').textContent = msg;
            l.style.display = 'flex';
        }
    }

    updateButtons() {
        const n = this.datasets.length;
        document.getElementById('compareBtn').disabled = n < 2;
        document.getElementById('mergeBtn').disabled = n < 2;
    }

    hide(id) {
        const m = document.getElementById(id);
        if (m) m.style.display = 'none';
    }

    notify(msg, type = 'info') {
        const n = document.createElement('div');
        n.textContent = msg;
        n.className = `notif ${type}`;
        n.style = `
          position:fixed;top:20px;right:20px;background:${
            { info: '#3b82f6', success: '#16a34a', error: '#dc2626' }[type]
          };color:#fff;padding:10px 16px;border-radius:6px;z-index:9999`;
        document.body.appendChild(n);
        setTimeout(() => n.remove(), 3000);
    }
}

/** === INIT === */
let datasetManager;
document.addEventListener('DOMContentLoaded', () => (datasetManager = new DatasetManager()));
