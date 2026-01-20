/* Quantix Graph Studio */

(function () {
  const els = {
    session: document.getElementById('graph-session'),
    reload: document.getElementById('graph-reload'),
    type: document.getElementById('graph-type'),
    form: document.getElementById('graph-form'),
    run: document.getElementById('graph-run'),
    plot: document.getElementById('graph-plot'),
    metrics: document.getElementById('graph-metrics'),
    summary: document.getElementById('graph-summary'),
    error: document.getElementById('graph-error'),
    download: document.getElementById('graph-download'),
    fileInput: document.getElementById('file-input'),
    fileInfo: document.getElementById('file-info'),
    includeLayout: document.getElementById('include-layout'),
    replaceSession: document.getElementById('replace-session'),
  };

  let catalog = null;
  let demoRendered = false;

  const defaultPlotHtml = els.plot ? els.plot.innerHTML : '';
  const defaultMetricsHtml = els.metrics ? els.metrics.innerHTML : '';
  const defaultSummaryHtml = els.summary ? els.summary.innerHTML : '';

  function setPill(type, text) {
    if (!els.session) return;
    els.session.className = 'pill ' + type;
    els.session.textContent = text;
  }

  function showError(msg) {
    els.error.style.display = 'block';
    els.error.textContent = msg;
  }

  function clearError() {
    els.error.style.display = 'none';
    els.error.textContent = '';
  }

  async function checkSession() {
    try {
      const r = await fetch('/api/session/status');
      const d = await r.json();
      if (d.success && d.has_file) {
        setPill(d.is_data_file ? 'pill--ok' : 'pill--warn', 'Fichier: ' + d.filename);
        clearDemo();
        return d;
      }
      setPill('pill--bad', 'Aucun fichier en session');
      renderDemo();
      return d;
    } catch (e) {
      setPill('pill--warn', 'Session: indisponible');
      // En cas d'indisponibilité, on garde l'UI sobre (pas de démo forcée)
      return null;
    }
  }

  function clearDemo() {
    if (!demoRendered) return;
    demoRendered = false;
    if (els.plot) {
      els.plot.innerHTML = defaultPlotHtml;
      try { Plotly && Plotly.purge && Plotly.purge(els.plot); } catch (e) {}
    }
    if (els.metrics) els.metrics.innerHTML = defaultMetricsHtml;
    if (els.summary) {
      els.summary.style.display = 'none';
      els.summary.innerHTML = defaultSummaryHtml;
    }
  }

  function renderDemo() {
    if (!els.plot) return;
    if (demoRendered) return;
    // Si un graphe a déjà été rendu, ne pas écraser.
    if (els.plot.dataset && els.plot.dataset.hasLive === '1') return;

    demoRendered = true;
    els.plot.innerHTML = `
      <div class="demo">
        <div class="demo__header">
          <div class="demo__title">Exemples de visualisations</div>
          <div class="demo__hint">Importez un fichier pour générer vos propres graphes.</div>
        </div>
        <div class="demo__grid">
          <div class="demo__card">
            <div class="demo__cardTitle">Corrélation (heatmap)</div>
            <div id="demo-corr" class="demo__plot"></div>
          </div>
          <div class="demo__card">
            <div class="demo__cardTitle">Co-occurrence (réseau)</div>
            <div id="demo-net" class="demo__plot"></div>
          </div>
          <div class="demo__card">
            <div class="demo__cardTitle">Distribution (exemple)</div>
            <div id="demo-dist" class="demo__plot"></div>
          </div>
        </div>
      </div>
    `;

    // Mets des métriques d'exemple à droite.
    renderMetrics({ num_nodes: 42, num_edges: 88, diameter: 7, clustering: 0.23 });
    if (els.summary) {
      els.summary.style.display = 'block';
      els.summary.innerHTML = 'Mode démo • exemples indicatifs (à venir)';
    }

    // 1) Heatmap (corrélation)
    const corrEl = document.getElementById('demo-corr');
    if (corrEl && window.Plotly) {
      const labs = ['A', 'B', 'C', 'D', 'E', 'F'];
      const z = [
        [1.0, 0.62, -0.12, 0.30, 0.05, -0.41],
        [0.62, 1.0, 0.08, 0.44, -0.19, -0.10],
        [-0.12, 0.08, 1.0, -0.55, 0.22, 0.14],
        [0.30, 0.44, -0.55, 1.0, 0.09, 0.33],
        [0.05, -0.19, 0.22, 0.09, 1.0, 0.51],
        [-0.41, -0.10, 0.14, 0.33, 0.51, 1.0],
      ];
      Plotly.newPlot(
        corrEl,
        [{ type: 'heatmap', z, x: labs, y: labs, colorscale: 'RdBu', zmid: 0, showscale: false }],
        { margin: { l: 30, r: 10, t: 10, b: 25 }, paper_bgcolor: '#fff', plot_bgcolor: '#fff' },
        { responsive: true, displayModeBar: false }
      );
    }

    // 2) Réseau (co-occurrence)
    const netEl = document.getElementById('demo-net');
    if (netEl && window.Plotly) {
      const nodes = ['IA', 'Données', 'Texte', 'Image', 'PDF', 'CSV'];
      const pos = {
        IA: { x: 0.0, y: 0.2 },
        Données: { x: -0.6, y: 0.6 },
        Texte: { x: 0.6, y: 0.6 },
        Image: { x: 0.8, y: -0.2 },
        PDF: { x: -0.2, y: -0.6 },
        CSV: { x: -0.8, y: -0.2 },
      };
      const links = [
        ['IA', 'Texte'],
        ['IA', 'Image'],
        ['IA', 'PDF'],
        ['Données', 'CSV'],
        ['PDF', 'Texte'],
        ['Image', 'CSV'],
        ['Texte', 'CSV'],
      ];

      const edgeX = [];
      const edgeY = [];
      links.forEach(([s, t]) => {
        const ps = pos[s];
        const pt = pos[t];
        edgeX.push(ps.x, pt.x, null);
        edgeY.push(ps.y, pt.y, null);
      });

      const xs = nodes.map(n => pos[n].x);
      const ys = nodes.map(n => pos[n].y);

      Plotly.newPlot(
        netEl,
        [
          { x: edgeX, y: edgeY, mode: 'lines', line: { width: 1, color: 'rgba(120,120,130,0.35)' }, hoverinfo: 'none', type: 'scattergl' },
          { x: xs, y: ys, mode: 'markers+text', text: nodes, textposition: 'top center', hoverinfo: 'text', marker: { size: 10, color: 'rgba(0,102,255,0.85)', line: { width: 1, color: '#fff' } }, type: 'scattergl' },
        ],
        { margin: { l: 10, r: 10, t: 10, b: 10 }, showlegend: false, paper_bgcolor: '#fff', plot_bgcolor: '#fff', xaxis: { visible: false }, yaxis: { visible: false } },
        { responsive: true, displayModeBar: false }
      );
    }

    // 3) Distribution (bar)
    const distEl = document.getElementById('demo-dist');
    if (distEl && window.Plotly) {
      const x = ['A', 'B', 'C', 'D', 'E'];
      const y = [12, 28, 18, 34, 22];
      Plotly.newPlot(
        distEl,
        [{ type: 'bar', x, y, marker: { color: 'rgba(0,102,255,0.6)' } }],
        { margin: { l: 30, r: 10, t: 10, b: 25 }, paper_bgcolor: '#fff', plot_bgcolor: '#fff' },
        { responsive: true, displayModeBar: false }
      );
    }
  }

  async function uploadFile(file) {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch('/upload-file', { method: 'POST', body: fd });
    const d = await r.json();
    if (!d.success) throw new Error(d.error || 'Upload failed');
    return d;
  }

  async function loadCatalog() {
    clearError();
    els.type.innerHTML = '';
    els.form.innerHTML = '';
    try {
      const r = await fetch('/api/graph/catalog');
      const d = await r.json();
      if (!d.success) throw new Error(d.error || 'Catalogue indisponible');
      catalog = d.catalog;

      const types = (catalog && catalog.graph_types) ? catalog.graph_types : [];
      if (!types.length) {
        els.type.innerHTML = '<option value="">Aucun type disponible</option>';
        return;
      }

      types.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t.graph_type;
        opt.textContent = (t.label || t.graph_type);
        opt.dataset.description = t.description || '';
        els.type.appendChild(opt);
      });

      renderFormForCurrentType();
    } catch (e) {
      showError(e.message);
    }
  }

  function getCurrentTypeDef() {
    if (!catalog || !catalog.graph_types) return null;
    const gt = els.type.value;
    return catalog.graph_types.find(x => x.graph_type === gt) || null;
  }

  function renderFormForCurrentType() {
    const def = getCurrentTypeDef();
    els.form.innerHTML = '';
    if (!def) return;

    if (def.description) {
      const p = document.createElement('div');
      p.className = 'muted';
      p.style.fontSize = '12px';
      p.style.marginBottom = '8px';
      p.textContent = def.description;
      els.form.appendChild(p);
    }

    const schema = def.schema || {};
    Object.keys(schema).forEach(key => {
      const f = schema[key];
      const row = document.createElement('div');
      row.className = 'form-row';

      const label = document.createElement('label');
      label.textContent = key;
      row.appendChild(label);

      let input;
      if (f.type === 'bool') {
        input = document.createElement('select');
        input.innerHTML = '<option value="true">true</option><option value="false">false</option>';
        input.value = String(f.default ?? true);
      } else if (f.type === 'enum') {
        input = document.createElement('select');
        (f.choices || []).forEach(c => {
          const o = document.createElement('option');
          o.value = c;
          o.textContent = c;
          input.appendChild(o);
        });
        if (f.default != null) input.value = String(f.default);
      } else if (f.type === 'json' || f.type === 'json_optional') {
        input = document.createElement('textarea');
        input.placeholder = '{ ... }';
        input.value = (f.default != null) ? JSON.stringify(f.default, null, 2) : '';
      } else {
        input = document.createElement('input');
        input.type = (f.type === 'int' || f.type === 'number') ? 'number' : 'text';
        if (f.min != null) input.min = String(f.min);
        if (f.max != null) input.max = String(f.max);
        if (f.default != null) input.value = String(f.default);
        if (f.type && f.type.indexOf('optional') !== -1) input.placeholder = '(optionnel)';
      }

      input.className = 'field';
      input.dataset.key = key;
      input.dataset.kind = f.type;
      row.appendChild(input);
      els.form.appendChild(row);
    });
  }

  function collectOptions() {
    const opts = {};
    const nodes = els.form.querySelectorAll('[data-key]');
    nodes.forEach(n => {
      const key = n.dataset.key;
      const kind = n.dataset.kind;
      let val = n.value;

      if (kind === 'bool') {
        opts[key] = (val === 'true');
      } else if (kind === 'int') {
        if (val === '') return;
        opts[key] = parseInt(val, 10);
      } else if (kind === 'number') {
        if (val === '') return;
        opts[key] = parseFloat(val);
      } else if (kind === 'json' || kind === 'json_optional') {
        if (!val.trim()) {
          if (kind === 'json') throw new Error('Champ JSON requis: ' + key);
          return;
        }
        opts[key] = JSON.parse(val);
      } else if (kind && kind.indexOf('optional') !== -1) {
        if (val === '') return;
        opts[key] = val;
      } else {
        opts[key] = val;
      }
    });
    return opts;
  }

  function renderMetrics(metrics) {
    if (!metrics) {
      els.metrics.innerHTML = '<div class="muted">Aucune métrique.</div>';
      return;
    }
    const rows = [];
    rows.push(['Noeuds', metrics.num_nodes]);
    rows.push(['Arêtes', metrics.num_edges]);
    rows.push(['Diamètre', metrics.diameter == null ? '—' : metrics.diameter]);

    let clustering = metrics.clustering;
    if (clustering && typeof clustering === 'object') {
      // afficher moyenne
      const vals = Object.values(clustering);
      const avg = vals.length ? (vals.reduce((a, b) => a + b, 0) / vals.length) : 0;
      clustering = avg;
    }
    rows.push(['Clustering', clustering == null ? '—' : (typeof clustering === 'number' ? clustering.toFixed(3) : clustering)]);

    els.metrics.innerHTML = rows
      .map(([k, v]) => `<div class="metric"><span>${escapeHtml(String(k))}</span><strong>${escapeHtml(String(v))}</strong></div>`)
      .join('');
  }

  function plotGraph(payload) {
    const graph = payload.graph;
    const layout = payload.layout || {};
    const nodes = graph.nodes || [];
    const links = graph.links || graph.edges || [];

    if (!nodes.length) {
      els.plot.innerHTML = '<div class="plot__placeholder"><div class="plot__placeholder__title">Graphe vide</div><div class="plot__placeholder__hint">Aucun noeud après filtrage.</div></div>';
      return;
    }

    // Node ids (NetworkX node_link uses id)
    const nodeIds = nodes.map(n => String(n.id ?? n.name ?? n));

    const x = [];
    const y = [];
    nodeIds.forEach(id => {
      const p = layout[id] || { x: Math.random(), y: Math.random() };
      x.push(p.x);
      y.push(p.y);
    });

    // Build edge traces as one polyline trace (with null separators)
    const edgeX = [];
    const edgeY = [];

    links.forEach(e => {
      const s = String(e.source);
      const t = String(e.target);
      const ps = layout[s];
      const pt = layout[t];
      if (!ps || !pt) return;
      edgeX.push(ps.x, pt.x, null);
      edgeY.push(ps.y, pt.y, null);
    });

    const edgeTrace = {
      x: edgeX,
      y: edgeY,
      mode: 'lines',
      line: { width: 1, color: 'rgba(120,120,130,0.35)' },
      hoverinfo: 'none',
      type: 'scattergl',
      name: 'edges',
    };

    const nodeTrace = {
      x,
      y,
      mode: 'markers+text',
      text: nodeIds.map(id => (id.length > 20 ? id.slice(0, 20) + '…' : id)),
      textposition: 'top center',
      hovertext: nodeIds,
      hoverinfo: 'text',
      marker: {
        size: 9,
        color: 'rgba(0,102,255,0.85)',
        line: { width: 1, color: '#fff' },
      },
      type: 'scattergl',
      name: 'nodes',
    };

    els.plot.innerHTML = '';
    if (els.plot && els.plot.dataset) els.plot.dataset.hasLive = '1';
    Plotly.newPlot(
      els.plot,
      [edgeTrace, nodeTrace],
      {
        margin: { l: 10, r: 10, t: 10, b: 10 },
        showlegend: false,
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        xaxis: { visible: false },
        yaxis: { visible: false },
      },
      { responsive: true }
    );

    els.summary.style.display = 'block';
    els.summary.innerHTML = `Type: <strong>${escapeHtml(payload.graph_type)}</strong> • Noeuds: <strong>${nodes.length}</strong> • Arêtes: <strong>${links.length}</strong>`;
  }

  async function runGraph() {
    clearError();

    const gt = els.type.value;
    if (!gt) {
      showError('Choisissez un type de graphe.');
      return;
    }

    let options;
    try {
      options = collectOptions();
    } catch (e) {
      showError(e.message);
      return;
    }

    els.run.disabled = true;
    els.run.textContent = 'Génération…';

    // Dès qu'on lance une génération, on considère que la zone plot est "live"
    if (els.plot && els.plot.dataset) els.plot.dataset.hasLive = '1';

    try {
      const r = await fetch('/api/graph/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          graph_type: gt,
          options,
          include_layout: els.includeLayout.checked,
          replace_session_file: els.replaceSession.checked,
        }),
      });
      const d = await r.json();
      if (!d.success) throw new Error(d.error || 'Erreur');

      renderMetrics(d.metrics);

      if (d.download_url) {
        els.download.style.display = 'inline-flex';
        els.download.href = d.download_url;
      } else {
        els.download.style.display = 'none';
      }

      if (d.layout) {
        plotGraph(d);
      } else {
        // no layout: show placeholder
        els.plot.innerHTML = '<div class="plot__placeholder"><div class="plot__placeholder__title">Export prêt</div><div class="plot__placeholder__hint">Layout désactivé ou indisponible.</div></div>';
      }

      await checkSession();
    } catch (e) {
      showError(e.message);
    } finally {
      els.run.disabled = false;
      els.run.textContent = 'Générer le graphe';
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Wiring
  els.reload?.addEventListener('click', loadCatalog);
  els.type?.addEventListener('change', renderFormForCurrentType);
  els.run?.addEventListener('click', runGraph);

  els.fileInput?.addEventListener('change', async function () {
    const file = els.fileInput.files && els.fileInput.files[0];
    if (!file) return;
    clearError();
    els.fileInfo.style.display = 'none';

    try {
      const res = await uploadFile(file);
      els.fileInfo.style.display = 'block';
      els.fileInfo.innerHTML = `<strong>${escapeHtml(res.file_info?.name || file.name)}</strong><div class="muted">Upload OK • prêt pour génération</div>`;
      await checkSession();
    } catch (e) {
      showError(e.message);
    }
  });

  // Init
  (async function init() {
    await checkSession();
    await loadCatalog();
  })();
})();
