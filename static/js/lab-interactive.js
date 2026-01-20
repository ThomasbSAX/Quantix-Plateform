/**
 * Quantix Lab — Interactive Workspace (version condensée)
 * Interface d’analyse numérique interactive inspirée de GeoGebra
 */

class InteractiveWorkspace {
    constructor() {
        this.currentAnalyses = new Map();
        this.activeColumn = null;
        this.plots = new Map();
        this.realTime = true;
        this.init();
    }

    /** === INITIALISATION === */
    init() {
        this.bindControls();
        this.initNotifications();
    }

    bindControls() {
        const degree = document.getElementById('degreeInput');
        if (degree)
            degree.addEventListener('input', e => this.updatePolynomialDegree(this.activeColumn, +e.target.value));

        const normalize = document.getElementById('normalizeToggle');
        if (normalize)
            normalize.addEventListener('change', e => this.toggleNormalization(this.activeColumn, e.target.checked));

        const select = document.getElementById('columnSelect');
        if (select)
            select.addEventListener('change', e => {
                this.activeColumn = e.target.value;
                if (e.target.value) this.showColumnPreview(e.target.value);
            });
    }

    /** === APERÇU DE COLONNE === */
    showColumnPreview(col) {
        fetch('/matrix/column_preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ column: col })
        })
            .then(r => r.json())
            .then(d => d.success && this.displayColumnPreview(d.preview))
            .catch(console.error);
    }

    displayColumnPreview(p) {
        let div = document.getElementById('columnPreview');
        if (!div) {
            div = document.createElement('div');
            div.id = 'columnPreview';
            document.getElementById('workspaceArea').prepend(div);
        }

        div.innerHTML = `
        <div class="preview-card">
          <h4>${p.column}</h4>
          <div class="grid">
            <span>Min: ${p.min.toFixed(3)}</span>
            <span>Max: ${p.max.toFixed(3)}</span>
            <span>Moy: ${p.mean.toFixed(3)}</span>
            <span>Obs: ${p.count}</span>
          </div>
          <div id="mini_${p.column}" class="mini-chart"></div>
        </div>`;
        this.drawMiniHistogram(`mini_${p.column}`, p.histogram_data);
    }

    drawMiniHistogram(id, data) {
        const t = { x: data.values, type: 'histogram', marker: { color: 'rgba(99,102,241,0.6)' } };
        Plotly.newPlot(id, [t], { margin: { t: 5, b: 25, l: 30, r: 5 }, height: 120 }, { displayModeBar: false });
    }

    /** === ANALYSES INTERACTIVES === */
    runInteractiveAnalysis(type, params) {
        const col = params.column || this.activeColumn;
        if (!col) return this.notify('Sélectionnez une colonne', 'warning');

        this.activeColumn = col;
        this.loading(`Analyse ${type}...`);
        fetch(`/matrix/analyze_${type}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...params, column: col })
        })
            .then(r => r.json())
            .then(d => {
                if (!d.success) return this.notify(d.error, 'error');
                this.currentAnalyses.set(col, d);
                this.displayResults(type, d);
                this.notify('Analyse terminée', 'success');
            })
            .finally(() => this.loading(false));
    }

    updatePolynomialDegree(col, deg) {
        if (!col) return;
        this.loading('Recalcul...');
        fetch('/matrix/update_polynomial', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ column: col, degree: deg })
        })
            .then(r => r.json())
            .then(d => d.success && this.updatePlot(col, d))
            .finally(() => this.loading(false));
    }

    toggleNormalization(col, norm) {
        if (!col) return;
        this.loading('Normalisation...');
        fetch('/matrix/toggle_normalization', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ column: col, normalize: norm })
        })
            .then(r => r.json())
            .then(d => d.success && this.updatePlot(col, d))
            .finally(() => this.loading(false));
    }

    updatePlot(col, d) {
        const c = document.getElementById('distributionPlot');
        if (!c) return;
        Plotly.react(c, d.plot_data || [], d.layout || {});
        this.plots.set(col, d);
    }

    displayResults(type, d) {
        if (type === 'auto_regression') return this.displayRegression(d);
        if (d.plot_data) this.updatePlot(this.activeColumn, d);
    }

    displayRegression(d) {
        const w = document.getElementById('workspaceArea');
        w.innerHTML = `
        <div class="regression">
          <h3>Régression - ${d.target_variable}</h3>
          <p>R²: ${(d.metrics.r2 * 100).toFixed(2)}%</p>
          <div id="predictionPlot"></div>
        </div>`;
        const plot = JSON.parse(d.interactive_plots.prediction_vs_actual);
        Plotly.newPlot('predictionPlot', plot.data, plot.layout);
    }

    /** === CONTROLES GEO-GEBRA STYLE === */
    setupInteractiveElements(col, el) {
        if (!el) return;
        const c = document.getElementById('interactiveControls') || this.makeControlsContainer();
        Object.entries(el.sliders || {}).forEach(([k, s]) =>
            c.innerHTML += this.makeSliderHTML(col, k, s)
        );
    }

    makeControlsContainer() {
        const div = document.createElement('div');
        div.id = 'interactiveControls';
        div.className = 'controls-panel';
        document.getElementById('workspaceArea').appendChild(div);
        return div;
    }

    makeSliderHTML(col, k, s) {
        return `
        <div class="slider">
          <label>${s.label} (${s.min}-${s.max})</label>
          <input type="range" min="${s.min}" max="${s.max}" step="${s.step}" 
                 value="${s.default}" 
                 oninput="interactiveWorkspace.executeCallback('${s.callback}','${col}',this.value)">
        </div>`;
    }

    executeCallback(cb, col, val) {
        const f = {
            'updatePolynomial': this.updatePolynomialDegree,
            'toggleNormalization': this.toggleNormalization
        }[cb];
        if (f) f.call(this, col, val);
    }

    /** === NOTIFICATIONS & CHARGEMENT === */
    initNotifications() {
        const div = document.createElement('div');
        div.id = 'notif';
        div.style = 'position:fixed;top:20px;right:20px;z-index:9999';
        document.body.appendChild(div);
    }

    notify(msg, type = 'info') {
        const n = document.createElement('div');
        n.className = `n-${type}`;
        n.textContent = msg;
        n.style = `
            background:${{ info: '#3b82f6', success: '#10b981', warning: '#f59e0b', error: '#ef4444' }[type]};
            color:#fff;padding:8px 12px;margin-top:8px;border-radius:4px;`;
        document.getElementById('notif').appendChild(n);
        setTimeout(() => n.remove(), 2500);
    }

    loading(show) {
        let l = document.getElementById('loader');
        if (!show) return l && (l.remove(), null);
        if (!l) {
            l = document.createElement('div');
            l.id = 'loader';
            l.style = `
                position:fixed;top:0;left:0;width:100%;height:100%;
                background:rgba(255,255,255,0.7);display:flex;
                align-items:center;justify-content:center;font-size:1.1rem;z-index:9998;`;
            document.body.appendChild(l);
        }
        l.textContent = typeof show === 'string' ? show : 'Chargement...';
    }

    /** === SAUVEGARDE / SESSION === */
    saveSession(name) {
        localStorage.setItem(`quantix_${name}`, JSON.stringify({
            col: this.activeColumn,
            analyses: [...this.currentAnalyses.entries()]
        }));
        this.notify(`Session ${name} sauvegardée`, 'success');
    }

    loadSession(name) {
        const s = JSON.parse(localStorage.getItem(`quantix_${name}`));
        if (!s) return this.notify('Session introuvable', 'error');
        this.activeColumn = s.col;
        this.currentAnalyses = new Map(s.analyses);
        this.notify(`Session ${name} chargée`, 'success');
    }
}

/** === INITIALISATION === */
let interactiveWorkspace;
document.addEventListener('DOMContentLoaded', () => {
    interactiveWorkspace = new InteractiveWorkspace();
    window.interactiveWorkspace = interactiveWorkspace;
});
