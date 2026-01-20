// Quantix Lab JavaScript - Interface utilisateur pour les outils d'analyse
console.log('Quantix Lab JavaScript chargé');

// Vérifier automatiquement le statut de la session au chargement
document.addEventListener('DOMContentLoaded', function() {
    checkSessionStatus();

    // Initialiser l'UI des onglets (Nettoyage / Transformation / Visualisation)
    if (window.quantixLab && typeof window.quantixLab.initLabTabs === 'function') {
        window.quantixLab.initLabTabs();
    }

    // Initialiser Plot Studio (catalogue modules/plots)
    if (window.quantixLab && typeof window.quantixLab.initPlotStudio === 'function') {
        window.quantixLab.initPlotStudio();
    }

    // Initialiser Data Explorer (aperçu tableau + transformations)
    if (window.quantixLab && typeof window.quantixLab.initDataExplorer === 'function') {
        window.quantixLab.initDataExplorer();
    }
});

// Fonction pour vérifier si un fichier est présent en session
function checkSessionStatus() {
    fetch('/api/session/status', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log('Statut de session:', data);
        
        if (data.success && data.has_file) {
            if (data.is_data_file) {
                showSessionInfo('Fichier "' + data.filename + '" disponible pour l\'analyse', 'success');
            } else {
                showSessionInfo('Fichier "' + data.filename + '" non compatible (dataset requis)', 'warning');
            }
        } else {
            // Pas de toast automatique si aucun fichier n'est chargé.
            // Les erreurs restent affichées lorsqu'une action utilisateur nécessite un dataset.
            clearSessionInfo();
        }
    })
    .catch(error => {
        console.error('Erreur lors de la vérification de session:', error);
        // Évite un toast intrusif au chargement (garde l'info dans la console).
        clearSessionInfo();
    });
}

function clearSessionInfo() {
    var statusDiv = document.getElementById('session-status');
    if (statusDiv) statusDiv.remove();
}

// Fonction pour afficher les informations de session
function showSessionInfo(message, type) {
    var statusDiv = document.getElementById('session-status');
    if (!statusDiv) {
        statusDiv = document.createElement('div');
        statusDiv.id = 'session-status';
        statusDiv.style.cssText = 'position: fixed; top: 20px; right: 20px; padding: 12px 16px; border-radius: 8px; color: white; font-weight: 500; font-size: 14px; max-width: 300px; z-index: 1000; box-shadow: 0 4px 12px rgba(0,0,0,0.2);';
        document.body.appendChild(statusDiv);
    }
    
    var colors = {
        success: '#10b981',
        warning: '#f59e0b', 
        error: '#ef4444'
    };
    
    statusDiv.style.backgroundColor = colors[type] || colors.error;
    statusDiv.textContent = message;
    
    // Masquer automatiquement après 5 secondes si succès
    if (type === 'success') {
        setTimeout(function() {
            statusDiv.style.opacity = '0';
            setTimeout(function() { statusDiv.remove(); }, 300);
        }, 5000);
    }
}

// Objet global pour gérer les outils Lab
window.quantixLab = {
    _labTabs: {
        active: 'clean',
        history: []
    },

    initLabTabs: function() {
        var self = this;
        var tabButtons = Array.prototype.slice.call(document.querySelectorAll('.lab-tab'));
        var backBtn = document.getElementById('tab-back');
        if (!tabButtons || tabButtons.length === 0) return;

        function onClick(ev) {
            try {
                var btn = ev.currentTarget;
                var tab = btn && btn.getAttribute('data-tab');
                if (!tab) return;
                self.setActiveTab(tab);
            } catch (e) {
                console.warn(e);
            }
        }

        tabButtons.forEach(function(btn) {
            btn.addEventListener('click', onClick);
        });

        if (backBtn) {
            backBtn.addEventListener('click', function() {
                self.goBack();
            });
        }

        var saved = null;
        try { saved = sessionStorage.getItem('quantixLab.activeTab'); } catch (e) {}
        var initial = (saved === 'clean' || saved === 'transform' || saved === 'viz') ? saved : 'clean';
        self.setActiveTab(initial, { initial: true });
    },

    goBack: function() {
        var h = this._labTabs.history || [];
        if (h.length === 0) {
            this._updateBackButtonState();
            return;
        }
        var prev = h.pop();
        this._labTabs.history = h;
        this.setActiveTab(prev, { fromHistory: true });
    },

    _updateBackButtonState: function() {
        var backBtn = document.getElementById('tab-back');
        if (!backBtn) return;
        var h = this._labTabs.history || [];
        var disabled = (h.length === 0);
        backBtn.disabled = disabled;
        backBtn.style.opacity = disabled ? '0.55' : '1';
        backBtn.style.cursor = disabled ? 'not-allowed' : 'pointer';
    },

    setActiveTab: function(tab, opts) {
        opts = opts || {};
        if (['clean', 'transform', 'viz'].indexOf(tab) === -1) tab = 'clean';

        var prevTab = this._labTabs.active;
        if (!opts.initial && !opts.fromHistory && prevTab && prevTab !== tab) {
            var h = this._labTabs.history || [];
            // éviter les doublons consécutifs
            if (h.length === 0 || h[h.length - 1] !== prevTab) {
                h.push(prevTab);
            }
            // cap simple pour éviter croissance infinie
            if (h.length > 20) h = h.slice(h.length - 20);
            this._labTabs.history = h;
        }

        this._labTabs.active = tab;

        try { sessionStorage.setItem('quantixLab.activeTab', tab); } catch (e) {}

        var panelData = document.getElementById('lab-panel-data');
        var panelCleanSummary = document.getElementById('lab-panel-clean-summary');
        var panelTransform = document.getElementById('lab-panel-transform');
        var panelViz = document.getElementById('lab-panel-viz');

        var cleaningTools = document.getElementById('data-explorer-cleaning-tools');
        var transformTools = document.getElementById('data-explorer-transform-tools');

        // Panneaux
        if (panelData) panelData.style.display = (tab === 'viz') ? 'none' : 'block';
        if (panelCleanSummary) panelCleanSummary.style.display = (tab === 'clean') ? 'block' : 'none';
        if (panelTransform) panelTransform.style.display = (tab === 'transform') ? 'block' : 'none';
        if (panelViz) panelViz.style.display = (tab === 'viz') ? 'block' : 'none';

        // Sous-outils dans le Data Explorer
        if (cleaningTools) cleaningTools.style.display = (tab === 'clean') ? 'flex' : 'none';
        if (transformTools) transformTools.style.display = (tab === 'transform') ? 'flex' : 'none';

        // Style boutons onglets
        var tabButtons = Array.prototype.slice.call(document.querySelectorAll('.lab-tab'));
        tabButtons.forEach(function(btn) {
            var t = btn.getAttribute('data-tab');
            var selected = (t === tab);
            btn.setAttribute('aria-selected', selected ? 'true' : 'false');
            btn.classList.remove('btn-primary');
            if (selected) btn.classList.add('btn-primary');
        });

        this._updateBackButtonState();

        // Focus utile
        if (!opts.initial && tab === 'viz') {
            var search = document.getElementById('plots-search');
            if (search) {
                setTimeout(function() {
                    try { search.focus(); } catch (e) {}
                }, 150);
            }
        }
    },

    startTool: function(toolName) {
        console.log('Démarrage de l\'outil:', toolName);
        
        var self = this;
        
        // Vérifier d'abord si un fichier est disponible
        fetch('/api/session/status')
        .then(function(response) { return response.json(); })
        .then(function(sessionData) {
            if (!sessionData.has_file || !sessionData.is_data_file) {
                showError('Aucun dataset compatible chargé. Importez un fichier CSV/Excel/JSON dans cette page.');
                return Promise.reject('Pas de fichier');
            }
            
            // Charger les informations du dataset
            return fetch('/api/lab/dataset-info');
        })
        .then(function(response) { return response.json(); })
        .then(function(datasetInfo) {
            if (!datasetInfo.success) {
                throw new Error(datasetInfo.error);
            }
            
            // For some tools, show options modal first
            var toolsRequiringOptions = ['histogram','scatter-plot','box-plot','correlation'];
            if (toolsRequiringOptions.indexOf(toolName) !== -1) {
                openLabOptions(toolName, datasetInfo, function(options) {
                    // when confirmed, call the tool and pass options
                    var fn = self.callTool.bind(self, toolName, datasetInfo);
                    // many tool functions accept (datasetInfo, options) pattern
                    try {
                        // try to call specific function with options if exists
                        var toolFunc = self[toolName.replace(/-([a-z])/g, function(m,g){return g.toUpperCase();})] || null;
                    } catch(e) { var toolFunc = null; }
                    // Fallback: call mapped functions (createHistogram etc)
                    switch(toolName) {
                        case 'histogram': return self.createHistogram(datasetInfo, options);
                        case 'scatter-plot': return self.createScatterPlot(datasetInfo, options);
                        case 'box-plot': return self.createBoxplot(datasetInfo, options);
                        case 'correlation': return self.createCorrelationMatrix(datasetInfo, options);
                        default: return self.callTool(toolName, datasetInfo);
                    }
                });
            } else {
                // Appeler l'outil spécifique avec les informations du dataset
                self.callTool(toolName, datasetInfo);
            }
        })
        .catch(function(error) {
            if (error !== 'Pas de fichier') {
                console.error('Erreur:', error);
                showError('Erreur: ' + error.message);
            }
        });
    },
    
    callTool: function(toolName, datasetInfo) {
        var self = this;
        var tools = {
            'scatter-plot': function() { return self.createScatterPlot(datasetInfo); },
            'histogram': function() { return self.createHistogram(datasetInfo); },
            'box-plot': function() { return self.createBoxplot(datasetInfo); },
            'correlation': function() { return self.createCorrelationMatrix(datasetInfo); },
            'pca': function() { return self.runPCA(datasetInfo); },
            'ica': function() { return self.runICA(datasetInfo); },
            'afc': function() { return self.runAFC(datasetInfo); },
            'spectral': function() { return self.runSpectral(datasetInfo); },
            'distribution-compare': function() { return self.runDistributionCompare(datasetInfo); },
            'multivariate': function() { return self.runMultivariate(datasetInfo); },
            'statistical': function() { return self.runStatisticalDescribe(datasetInfo); },
            'clustering': function() { return self.runClustering(datasetInfo); },
            'polynomial-approximation': function() { return self.polynomialApproximation(datasetInfo); },
            'distribution-analysis': function() { return self.distributionAnalysis(datasetInfo); },
            'curve-fitting': function() { return self.curveFitting(datasetInfo); },
            'linear-regression': function() { return self.linearRegression(datasetInfo); },
            'time-series': function() { return self.timeSeries(datasetInfo); },
            'classification': function() { return self.classification(datasetInfo); },
            'descriptive-stats': function() { return self.descriptiveStats(datasetInfo); },
            'hypothesis-test': function() { return self.hypothesisTest(datasetInfo); },
            'outlier-detection': function() { return self.outlierDetection(datasetInfo); }
        };
        
        var toolFunction = tools[toolName];
        if (toolFunction) {
            toolFunction();
        } else {
            showError('Outil non trouvé: ' + toolName);
        }
    }
};

// ============================
// Plot Studio (modules/plots)
// ============================

quantixLab._plotCatalog = null;

// Récupère la requête active (Data Explorer) sous forme de payload API.
// Retourne null si aucun filtre/recherche n'est actif.
quantixLab.getActiveQueryPayload = function() {
    try {
        var q = (this._dataExplorer && this._dataExplorer.query) ? this._dataExplorer.query : null;
        if (!q) return null;
        var qq = (q.q != null) ? String(q.q).trim() : '';
        var ff = Array.isArray(q.filters) ? q.filters : [];
        if (!qq && ff.length === 0) return null;
        return { q: qq, filters: ff };
    } catch (e) {
        return null;
    }
};

quantixLab.initPlotStudio = function() {
    var self = this;
    var container = document.getElementById('plots-tools');
    var searchInput = document.getElementById('plots-search');
    var reloadBtn = document.getElementById('plots-reload');
    if (!container) return;

    var classicItems = [
        { tool: 'histogram', label: 'Histogramme', description: 'Distribution d\'une colonne numérique.' },
        { tool: 'scatter-plot', label: 'Nuage de points', description: 'Relation entre deux colonnes numériques.' },
        { tool: 'box-plot', label: 'Boxplot', description: 'Dispersion / outliers sur une ou plusieurs colonnes.' },
        { tool: 'correlation', label: 'Matrice de corrélation', description: 'Corrélations (Pearson) sur colonnes numériques.' },
    ];

    function renderGroups(examples, filterText) {
        container.innerHTML = '';

        var h1 = document.createElement('div');
        h1.style.cssText = 'margin: 4px 0 10px; font-weight: 700; color:#111827;';
        h1.textContent = 'Classiques (disponibles)';
        container.appendChild(h1);

        classicItems.forEach(function(it) {
            var div = document.createElement('div');
            div.className = 'tool-item';
            div.style.cursor = 'pointer';
            div.onclick = function() { self.startTool(it.tool); };
            div.innerHTML = '<h4>' + escapeHtml(it.label) + '</h4><span>' + escapeHtml(it.description || '') + '</span>';
            container.appendChild(div);
        });

        var sep = document.createElement('div');
        sep.style.cssText = 'height:1px; background:#f3f4f6; margin: 12px 0;';
        container.appendChild(sep);

        var h2 = document.createElement('div');
        h2.style.cssText = 'margin: 0 0 10px; font-weight: 700; color:#111827;';
        h2.textContent = 'Exemples (bientôt disponible)';
        container.appendChild(h2);

        var note = document.createElement('div');
        note.style.cssText = 'margin: 0 0 10px; color:#6b7280; font-size:0.92rem;';
        note.textContent = 'Ces visualisations restent visibles comme exemples, mais les options avancées seront ajoutées plus tard.';
        container.appendChild(note);

        var filtered = (examples || []);
        if (filterText) {
            var q = String(filterText).toLowerCase().trim();
            if (q) {
                filtered = filtered.filter(function(it){
                    return it.label.toLowerCase().indexOf(q) !== -1 || it.plot_type.toLowerCase().indexOf(q) !== -1 || (it.description||'').toLowerCase().indexOf(q) !== -1;
                });
            }
        }

        if (!filtered || filtered.length === 0) {
            var empty = document.createElement('div');
            empty.className = 'tool-item';
            empty.innerHTML = '<h4>Aucun résultat</h4><span>Essayez un autre filtre.</span>';
            container.appendChild(empty);
            return;
        }

        filtered.forEach(function(it) {
            var div = document.createElement('div');
            div.className = 'tool-item';
            div.style.opacity = '0.6';
            div.style.cursor = 'not-allowed';
            div.onclick = function() { showError('Bientôt disponible'); };
            div.innerHTML = '<h4>' + escapeHtml(it.label) + '</h4><span>' + escapeHtml(it.description || '') + '</span>';
            container.appendChild(div);
        });
    }

    function load() {
        container.innerHTML = '<div class="tool-item"><h4>Chargement…</h4><span>Catalogue Plot Studio</span></div>';
        fetch('/api/plots/catalog')
            .then(function(r){ return r.json(); })
            .then(function(data){
                if (!data.success) throw new Error(data.error || 'Impossible de charger le catalogue');
                self._plotCatalog = data.catalog;
                var plotTypes = (data.catalog && data.catalog.plot_types) ? data.catalog.plot_types : {};
                var items = Object.keys(plotTypes).map(function(k){
                    return {
                        plot_type: k,
                        label: (plotTypes[k].label || k),
                        description: plotTypes[k].description || ''
                    };
                });
                items.sort(function(a,b){ return a.label.localeCompare(b.label); });
                renderGroups(items, (searchInput ? searchInput.value : ''));

                if (searchInput) {
                    searchInput.oninput = function() {
                        renderGroups(items, searchInput.value || '');
                    };
                }
            })
            .catch(function(e){
                container.innerHTML = '<div class="tool-item"><h4>Plot Studio indisponible</h4><span>' + escapeHtml(e.message) + '</span></div>';
            });
    }

    if (reloadBtn) reloadBtn.onclick = load;
    load();
};

quantixLab.startPlot = function(plotType) {
    // Conservé uniquement comme placeholder: les options avancées seront ajoutées plus tard.
    showError('Bientôt disponible');
};

quantixLab.renderPlotSpec = function(spec, plotType) {
    showLoading('Rendu Plot Studio…');
    fetch('/api/plots/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(spec)
    })
    .then(function(r){ return r.json(); })
    .then(function(data){
        hideLoading();
        if (data.success) {
            var label = (this._plotCatalog && this._plotCatalog.plot_types && this._plotCatalog.plot_types[plotType] && this._plotCatalog.plot_types[plotType].label) ? this._plotCatalog.plot_types[plotType].label : plotType;
            showResult(data, 'Plot Studio — ' + label);
        } else {
            showError(data.error || 'Erreur Plot Studio');
        }
    }.bind(this))
    .catch(function(e){ hideLoading(); showError(e.message); });
};


// ============================
// Quickstart plots (inline)
// ============================

quantixLab._quickstart = {
    datasetInfo: null,
    lastSpec: null,
};

quantixLab._renderQuickstartSummary = function() {
    var root = document.getElementById('qs-summary');
    if (!root) return;

    root.innerHTML = '<div style="padding:14px; color:#6b7280; font-size:0.95rem;">Calcul des statistiques descriptives…</div>';

    fetch('/api/lab/summary')
        .then(function(r){ return r.json(); })
        .then(function(data){
            if (!data.success) throw new Error(data.error || 'summary failed');
            var cols = data.columns || [];

            var html = '';
            html += '<div style="padding:12px 12px 0; color:#111827; font-weight:600;">' + escapeHtml(data.filename || '') + '</div>';
            html += '<div style="padding:0 12px 12px; color:#6b7280; font-size:0.9rem;">' + (data.rows || 0) + ' lignes • ' + (data.cols || 0) + ' colonnes</div>';
            html += '<table style="width:100%; border-collapse:collapse; min-width: 980px;">';
            html += '<thead><tr>';
            ['Colonne','Type','Manquants','%','Uniques','%','Stats / Top valeurs'].forEach(function(h){
                html += '<th style="text-align:left; padding:10px; border-bottom:1px solid #e5e5e5; background:#fff; position:sticky; top:0; font-size:0.85rem; color:#111827;">' + h + '</th>';
            });
            html += '</tr></thead><tbody>';

            cols.forEach(function(c){
                var missingPct = (typeof c.missing_pct === 'number') ? (c.missing_pct * 100).toFixed(1) + '%' : '';
                var uniquePct = (typeof c.unique_pct === 'number') ? (c.unique_pct * 100).toFixed(1) + '%' : '';

                var details = '';
                if (c.mean !== undefined) {
                    // numeric
                    var parts = [];
                    if (c.mean !== null && c.mean !== undefined) parts.push('mean=' + Number(c.mean).toFixed(3));
                    if (c.std !== null && c.std !== undefined) parts.push('std=' + Number(c.std).toFixed(3));
                    if (c.min !== null && c.min !== undefined) parts.push('min=' + Number(c.min).toFixed(3));
                    if (c.median !== null && c.median !== undefined) parts.push('median=' + Number(c.median).toFixed(3));
                    if (c.max !== null && c.max !== undefined) parts.push('max=' + Number(c.max).toFixed(3));
                    details = parts.join(' • ');
                } else if (c.top_values && c.top_values.length) {
                    details = c.top_values.map(function(tv){
                        return escapeHtml(String(tv.value)) + ' (' + tv.count + ')';
                    }).join(' • ');
                } else if (c.examples && c.examples.length) {
                    details = c.examples.map(function(x){ return escapeHtml(String(x)); }).join(' • ');
                }

                html += '<tr>';
                html += '<td style="padding:10px; border-bottom:1px solid #f3f4f6; font-size:0.9rem; color:#111827; white-space:nowrap;">' + escapeHtml(String(c.column)) + '</td>';
                html += '<td style="padding:10px; border-bottom:1px solid #f3f4f6; font-size:0.9rem; color:#374151; white-space:nowrap;">' + escapeHtml(String(c.dtype || '')) + '</td>';
                html += '<td style="padding:10px; border-bottom:1px solid #f3f4f6; font-size:0.9rem; color:#111827;">' + (c.missing || 0) + '</td>';
                html += '<td style="padding:10px; border-bottom:1px solid #f3f4f6; font-size:0.9rem; color:#6b7280;">' + missingPct + '</td>';
                html += '<td style="padding:10px; border-bottom:1px solid #f3f4f6; font-size:0.9rem; color:#111827;">' + (c.unique || 0) + '</td>';
                html += '<td style="padding:10px; border-bottom:1px solid #f3f4f6; font-size:0.9rem; color:#6b7280;">' + uniquePct + '</td>';
                html += '<td style="padding:10px; border-bottom:1px solid #f3f4f6; font-size:0.9rem; color:#111827;">' + (details || '') + '</td>';
                html += '</tr>';
            });

            html += '</tbody></table>';
            root.innerHTML = html;
        })
        .catch(function(e){
            root.innerHTML = '<div style="padding:14px; color:#b91c1c; font-size:0.95rem;">' + escapeHtml(e.message || String(e)) + '</div>';
        });
};

quantixLab.renderInlinePlotSpec = function(spec, containerId, title) {
    var container = document.getElementById(containerId);
    if (!container) return;

    // Petite UX: message de chargement dans le container
    container.innerHTML = '<div style="padding:14px; color:#6b7280; font-size:0.95rem;">Rendu du graphique…</div>';

    // Applique automatiquement le filtre actif du Data Explorer (si présent)
    // sans muter l'objet spec original.
    var finalSpec = Object.assign({}, spec || {});
    if (!finalSpec.query && window.quantixLab && typeof window.quantixLab.getActiveQueryPayload === 'function') {
        var activeQuery = window.quantixLab.getActiveQueryPayload();
        if (activeQuery) finalSpec.query = activeQuery;
    }

    fetch('/api/plots/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(finalSpec)
    })
    .then(function(r){ return r.json(); })
    .then(function(data){
        if (!data.success) throw new Error(data.error || 'Erreur Plot Studio');
        if (!data.plotly_json) throw new Error('plotly_json manquant');

        var plotData = JSON.parse(data.plotly_json);
        // Respecter la taille du container; le template/height est géré côté backend
        Plotly.react(containerId, plotData.data, plotData.layout, { responsive: true });

        var meta = document.getElementById('quickstart-meta');
        if (meta && title) {
            meta.textContent = title;
        }
    })
    .catch(function(e){
        container.innerHTML = '<div style="padding:14px; color:#b91c1c; font-size:0.95rem;">' + escapeHtml(e.message || String(e)) + '</div>';
    });
};

quantixLab._bindQuickstartButtons = function(datasetInfo) {
    var btnSummary = document.getElementById('qs-btn-summary');
    var btnMissing = document.getElementById('qs-btn-missing');
    var btnHist = document.getElementById('qs-btn-hist');
    var btnCount = document.getElementById('qs-btn-count');

    var numeric = (datasetInfo && datasetInfo.numeric_columns) ? datasetInfo.numeric_columns : [];
    var histSpec = (numeric.length >= 1)
        ? { plot_type: 'histogram', x: numeric[0] }
        : null;

    if (btnSummary) {
        btnSummary.onclick = function() {
            quantixLab._renderQuickstartSummary();
        };
    }
    if (btnMissing) {
        btnMissing.onclick = function() {
            var spec = { plot_type: 'missing_values' };
            quantixLab._quickstart.lastSpec = spec;
            quantixLab.renderInlinePlotSpec(spec, 'qs-plot', 'Valeurs manquantes par colonne');
        };
    }
    if (btnHist) {
        btnHist.onclick = function() {
            if (!histSpec) {
                return showError('Aucune colonne numérique disponible pour un histogramme.');
            }
            quantixLab._quickstart.lastSpec = histSpec;
            quantixLab.renderInlinePlotSpec(histSpec, 'qs-plot', 'Histogramme: ' + histSpec.x);
        };
    }

    if (btnCount) {
        btnCount.onclick = function() {
            var categorical = (datasetInfo && datasetInfo.categorical_columns) ? datasetInfo.categorical_columns : [];
            if (!categorical.length) {
                return showError('Aucune colonne catégorielle/texte disponible pour un comptage.');
            }
            var spec = { plot_type: 'count', x: categorical[0] };
            quantixLab._quickstart.lastSpec = spec;
            quantixLab.renderInlinePlotSpec(spec, 'qs-plot', 'Comptage: ' + spec.x);
        };
    }
};

quantixLab.quickStart = function() {
    var section = document.getElementById('quickstart-summary');
    var container = document.getElementById('qs-plot');
    if (!section || !container) return;

    // Vérifier session puis récupérer dataset-info
    fetch('/api/session/status')
        .then(function(r){ return r.json(); })
        .then(function(s){
            if (!s.success || !s.has_file || !s.is_data_file) {
                section.style.display = 'none';
                return Promise.reject('Pas de dataset');
            }
            return fetch('/api/lab/dataset-info');
        })
        .then(function(r){ return r.json(); })
        .then(function(info){
            if (!info.success) throw new Error(info.error || 'dataset-info failed');
            quantixLab._quickstart.datasetInfo = info;

            section.style.display = '';

            // Bind buttons once (idempotent-ish)
            quantixLab._bindQuickstartButtons(info);

            // Default view: summary (pas de corrélations)
            quantixLab._renderQuickstartSummary();
        })
        .catch(function(e){
            // Pas de dataset: ne rien afficher
            if (String(e) === 'Pas de dataset' || String(e) === 'Pas de fichier') return;
            console.error(e);
        });
};


// ============================
// Data Explorer (preview table)
// ============================

quantixLab._dataExplorer = {
    preview: null,
    sort: { column: null, dir: 'asc' },
    formatters: {}, // col -> {mode:'auto'|'scientific'|'fixed', decimals:number}
    query: { q: '', filters: [] },
};

quantixLab.initDataExplorer = function() {
    var self = this;
    var root = document.getElementById('data-explorer');
    if (!root) return;

    var btnRefresh = document.getElementById('data-refresh');
    var meta = document.getElementById('data-explorer-meta');
    var thead = document.getElementById('data-thead');
    var tbody = document.getElementById('data-tbody');
    var table = document.getElementById('data-table');

    // Query UI
    var queryQ = document.getElementById('query-q');
    var queryCol = document.getElementById('query-column');
    var queryOp = document.getElementById('query-op');
    var queryValue = document.getElementById('query-value');
    var queryApply = document.getElementById('query-apply');
    var queryClear = document.getElementById('query-clear');
    var queryExport = document.getElementById('query-export');
    var queryHint = document.getElementById('query-hint');

    var displayCol = document.getElementById('display-column');
    var displayMode = document.getElementById('display-mode');
    var displayDecimals = document.getElementById('display-decimals');
    var displayApply = document.getElementById('display-apply');

    // Column deletion (cleaning)
    var dropColumns = document.getElementById('drop-columns');
    var dropApply = document.getElementById('drop-apply');

    var transformCol = document.getElementById('transform-column');
    var transformAction = document.getElementById('transform-action');
    var transformValue = document.getElementById('transform-value');
    var transformFind = document.getElementById('transform-find');
    var transformReplace = document.getElementById('transform-replace');
    var transformMode = document.getElementById('transform-mode');
    var transformNewCol = document.getElementById('transform-newcol');
    var transformApply = document.getElementById('transform-apply');
    var transformHint = document.getElementById('transform-hint');

    function setHint() {
        if (!transformHint) return;
        var a = (transformAction && transformAction.value) ? transformAction.value : '';
        var hints = {
            linear: 'Crée x = a·y + b. Paramètres: "a,b" (ex: 1.2, 3).',
            log10: 'Requiert des valeurs > 0. Idéal pour SALAIRE, MONTANT, etc.',
            ln: 'Requiert des valeurs > 0.',
            sqrt: 'Requiert des valeurs >= 0.',
            minmax: 'Normalise dans [0,1] : (x-min)/(max-min).',
            zscore: 'Standardise : (x-mean)/std.',
            multiply: 'Paramètre: k (ex: 1.2)',
            add: 'Paramètre: k (ex: 1000)',
            power: 'Paramètre: p (ex: 2 ou 0.5)',
            round: 'Paramètre: d (nombre de décimales, ex: 2)',
            replace: 'Paramètres: find / replace (remplacement littéral)',
        };
        transformHint.textContent = hints[a] || '';

        var showFind = (a === 'replace');
        if (transformFind) {
            transformFind.style.display = showFind ? '' : 'none';
            transformFind.placeholder = 'find';
        }
        var showReplace = (a === 'replace');
        if (transformReplace) transformReplace.style.display = showReplace ? '' : 'none';

        var needsValue = ['linear', 'multiply', 'add', 'power', 'round'].indexOf(a) !== -1;
        if (transformValue) {
            transformValue.style.display = needsValue ? '' : 'none';
            if (a === 'linear') transformValue.placeholder = 'a,b (ex: 1.2,3)';
            else if (a === 'round') transformValue.placeholder = 'd (ex: 2)';
            else if (a === 'power') transformValue.placeholder = 'p (ex: 2)';
            else transformValue.placeholder = 'k (ex: 1.2)';
        }

        var needsNew = (transformMode && transformMode.value === 'new');
        if (transformNewCol) transformNewCol.style.display = needsNew ? '' : 'none';
    }

    function isNumericColumn(col) {
        var p = self._dataExplorer.preview;
        if (!p) return false;
        var dt = (p.dtypes && p.dtypes[col]) ? String(p.dtypes[col]).toLowerCase() : '';
        if (dt.indexOf('int') !== -1 || dt.indexOf('float') !== -1 || dt.indexOf('double') !== -1 || dt.indexOf('number') !== -1) {
            return true;
        }
        return false;
    }

    function formatCell(col, value) {
        if (value === null || value === undefined) return '';
        var fmt = self._dataExplorer.formatters[col] || { mode: 'auto' };
        if (fmt.mode === 'auto') return String(value);

        var n = Number(value);
        if (isNaN(n)) return String(value);

        if (fmt.mode === 'scientific') {
            var d = (typeof fmt.decimals === 'number') ? fmt.decimals : 3;
            return n.toExponential(d);
        }
        if (fmt.mode === 'fixed') {
            var dd = (typeof fmt.decimals === 'number') ? fmt.decimals : 2;
            return n.toFixed(dd);
        }
        return String(value);
    }

    function compareValues(a, b, col, dir) {
        var av = a[col];
        var bv = b[col];
        var asc = dir === 'asc';

        // Nulls last
        var an = (av === null || av === undefined || av === '');
        var bn = (bv === null || bv === undefined || bv === '');
        if (an && bn) return 0;
        if (an) return 1;
        if (bn) return -1;

        // Numeric if possible
        var numeric = isNumericColumn(col);
        if (numeric) {
            var na = Number(av);
            var nb = Number(bv);
            if (!isNaN(na) && !isNaN(nb)) {
                if (na === nb) return 0;
                return (na < nb ? -1 : 1) * (asc ? 1 : -1);
            }
        }

        // String fallback
        var sa = String(av).toLowerCase();
        var sb = String(bv).toLowerCase();
        if (sa === sb) return 0;
        return (sa < sb ? -1 : 1) * (asc ? 1 : -1);
    }

    function renderTable() {
        var p = self._dataExplorer.preview;
        if (!p || !thead || !tbody) return;

        var cols = p.columns || [];
        var rows = (p.rows || []).slice();

        if (self._dataExplorer.sort.column) {
            var c = self._dataExplorer.sort.column;
            var d = self._dataExplorer.sort.dir;
            rows.sort(function(r1, r2) { return compareValues(r1, r2, c, d); });
        }

        // Header
        var trh = document.createElement('tr');
        cols.forEach(function(col) {
            var th = document.createElement('th');
            th.textContent = col;
            th.style.cssText = 'text-align:left; padding:10px 10px; border-bottom:1px solid #e5e5e5; font-size:0.9rem; color:#111827; position:sticky; top:0; background:#fff; cursor:pointer; user-select:none;';

            var marker = '';
            if (self._dataExplorer.sort.column === col) {
                marker = self._dataExplorer.sort.dir === 'asc' ? '  ▲' : '  ▼';
            }
            th.textContent = col + marker;

            th.onclick = function() {
                if (self._dataExplorer.sort.column === col) {
                    self._dataExplorer.sort.dir = (self._dataExplorer.sort.dir === 'asc') ? 'desc' : 'asc';
                } else {
                    self._dataExplorer.sort.column = col;
                    self._dataExplorer.sort.dir = 'asc';
                }
                renderTable();
            };
            trh.appendChild(th);
        });
        thead.innerHTML = '';
        thead.appendChild(trh);

        // Body
        tbody.innerHTML = '';
        rows.forEach(function(r) {
            var tr = document.createElement('tr');
            cols.forEach(function(col) {
                var td = document.createElement('td');
                td.textContent = formatCell(col, r[col]);
                td.style.cssText = 'padding:9px 10px; border-bottom:1px solid #f3f4f6; font-size:0.9rem; color:#111827; white-space:nowrap;';

                // Excel-like edit: double click
                if (col !== '_row') {
                    td.ondblclick = function() {
                        var rowPos = r['_row'];
                        if (rowPos === null || rowPos === undefined) return;

                        var oldVal = (r[col] === null || r[col] === undefined) ? '' : String(r[col]);
                        td.innerHTML = '';
                        var inp = document.createElement('input');
                        inp.type = 'text';
                        inp.value = oldVal;
                        inp.style.cssText = 'width:100%; min-width:120px; padding:6px 8px; border:1px solid #c7d2fe; border-radius:6px; outline:none;';
                        td.appendChild(inp);
                        inp.focus();
                        inp.select();

                        function save() {
                            var newVal = inp.value;
                            // optimistic update only after success
                            showLoading('Sauvegarde…');
                            fetch('/api/lab/edit-cell', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ row: rowPos, column: col, value: newVal })
                            })
                            .then(function(resp){
                                return resp.json().then(function(j){ return { ok: resp.ok, json: j }; });
                            })
                            .then(function(resp){
                                hideLoading();
                                if (!resp.ok || !resp.json || !resp.json.success) {
                                    throw new Error((resp.json && resp.json.error) ? resp.json.error : 'Erreur sauvegarde');
                                }
                                // update in-memory
                                r[col] = (newVal === '' ? null : newVal);
                                renderTable();
                            })
                            .catch(function(e){
                                hideLoading();
                                showError(e.message || String(e));
                                renderTable();
                            });
                        }

                        inp.onkeydown = function(ev) {
                            if (ev.key === 'Enter') {
                                ev.preventDefault();
                                save();
                            }
                            if (ev.key === 'Escape') {
                                ev.preventDefault();
                                renderTable();
                            }
                        };
                        inp.onblur = function() {
                            save();
                        };
                    };
                }
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    }

    function fillSelect(selectEl, cols) {
        if (!selectEl) return;
        selectEl.innerHTML = '';
        cols.forEach(function(c) {
            var opt = document.createElement('option');
            opt.value = c;
            opt.textContent = c;
            selectEl.appendChild(opt);
        });
    }

    function getMultiSelectValues(selectEl) {
        if (!selectEl) return [];
        var out = [];
        for (var i = 0; i < selectEl.options.length; i++) {
            var opt = selectEl.options[i];
            if (opt && opt.selected) out.push(opt.value);
        }
        return out;
    }

    function setQueryHint(text) {
        if (!queryHint) return;
        queryHint.textContent = text || '';
    }

    function buildQueryPayload() {
        var q = queryQ ? String(queryQ.value || '').trim() : '';
        var col = queryCol ? String(queryCol.value || '').trim() : '';
        var op = queryOp ? String(queryOp.value || '').trim() : '';
        var val = queryValue ? String(queryValue.value || '').trim() : '';

        var filters = [];
        if (col && op) {
            if (op === 'isnull' || op === 'notnull') {
                filters.push({ column: col, op: op });
            } else if (val) {
                filters.push({ column: col, op: op, value: val });
            }
        }
        return { q: q, filters: filters, limit: 200 };
    }

    function hasActiveQuery(payload) {
        if (!payload) return false;
        var qq = payload.q && String(payload.q).trim();
        var ff = payload.filters && payload.filters.length;
        return Boolean(qq || ff);
    }

    function loadPreviewWithQuery(payload) {
        return fetch('/api/lab/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        }).then(function(r){ return r.json(); });
    }

    function loadBasePreview() {
        return fetch('/api/lab/preview?limit=200').then(function(r){ return r.json(); });
    }

    self.refreshDataExplorer = function() {
        fetch('/api/session/status')
            .then(function(r){ return r.json(); })
            .then(function(s){
                if (!s.has_file || !s.is_data_file) {
                    root.style.display = 'none';
                    return Promise.reject('Pas de fichier');
                }
                var payload = self._dataExplorer.query || { q: '', filters: [] };
                if (hasActiveQuery(payload)) {
                    return loadPreviewWithQuery(payload);
                }
                return loadBasePreview();
            })
            .then(function(p){
                if (!p.success) throw new Error(p.error || 'preview failed');
                self._dataExplorer.preview = p;
                root.style.display = 'block';
                if (meta) {
                    if (typeof p.filtered_rows === 'number') {
                        meta.textContent = '— ' + (p.filtered_rows || 0) + ' / ' + (p.total_rows || 0) + ' lignes';
                    } else {
                        meta.textContent = '— ' + (p.total_rows || 0) + ' lignes';
                    }
                }
                var cols = p.columns || [];
                fillSelect(displayCol, cols);
                fillSelect(transformCol, cols);
                fillSelect(queryCol, cols.filter(function(c){ return c !== '_row'; }));
                fillSelect(dropColumns, cols.filter(function(c){ return c !== '_row'; }));
                renderTable();
                setHint();
            })
            .catch(function(e){
                if (e !== 'Pas de fichier') {
                    console.warn(e);
                }
            });
    };

    if (btnRefresh) btnRefresh.onclick = self.refreshDataExplorer;

    // Query actions
    if (queryApply) {
        queryApply.onclick = function() {
            var payload = buildQueryPayload();
            self._dataExplorer.query = payload;
            setQueryHint(hasActiveQuery(payload) ? 'Filtre actif (aperçu filtré).' : '');
            self.refreshDataExplorer();
        };
    }

    if (queryClear) {
        queryClear.onclick = function() {
            if (queryQ) queryQ.value = '';
            if (queryValue) queryValue.value = '';
            self._dataExplorer.query = { q: '', filters: [] };
            setQueryHint('');
            self.refreshDataExplorer();
        };
    }

    if (queryExport) {
        queryExport.onclick = function() {
            var payload = self._dataExplorer.query || buildQueryPayload();
            if (!hasActiveQuery(payload)) {
                return showError('Aucun filtre/recherche actif à exporter.');
            }
            showLoading('Export CSV filtré…');
            fetch('/api/lab/export-query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(function(r){ return r.json().then(function(j){ return { ok: r.ok, json: j }; }); })
            .then(function(resp){
                hideLoading();
                if (!resp.ok || !resp.json || !resp.json.success) {
                    throw new Error((resp.json && resp.json.error) ? resp.json.error : 'Erreur export');
                }
                // déclencher téléchargement
                var a = document.createElement('a');
                a.href = resp.json.download_url;
                a.download = resp.json.filename || 'filtered.csv';
                document.body.appendChild(a);
                a.click();
                a.remove();
            })
            .catch(function(e){ hideLoading(); showError(e.message || String(e)); });
        };
    }

    // Enter key triggers apply
    if (queryQ) {
        queryQ.onkeydown = function(ev){ if (ev.key === 'Enter') { ev.preventDefault(); if (queryApply) queryApply.click(); } };
    }
    if (queryValue) {
        queryValue.onkeydown = function(ev){ if (ev.key === 'Enter') { ev.preventDefault(); if (queryApply) queryApply.click(); } };
    }

    if (displayApply) {
        displayApply.onclick = function() {
            var col = displayCol ? displayCol.value : '';
            if (!col) return;
            var mode = displayMode ? displayMode.value : 'auto';
            var dec = displayDecimals ? Number(displayDecimals.value) : 2;
            if (isNaN(dec)) dec = 2;
            self._dataExplorer.formatters[col] = { mode: mode, decimals: dec };
            renderTable();
        };
    }

    if (dropApply) {
        dropApply.onclick = function() {
            var cols = getMultiSelectValues(dropColumns);
            if (!cols || cols.length === 0) {
                return showError('Sélectionnez au moins une colonne à supprimer.');
            }
            var ok = confirm('Supprimer définitivement ' + cols.length + ' colonne(s) ?\n\n' + cols.join(', '));
            if (!ok) return;

            showLoading('Suppression de colonne(s)…');
            fetch('/api/lab/transform', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'drop_column', columns: cols })
            })
            .then(function(r){ return r.json().then(function(j){ return { ok: r.ok, json: j }; }); })
            .then(function(resp){
                hideLoading();
                if (!resp.ok || !resp.json || !resp.json.success) {
                    throw new Error((resp.json && resp.json.error) ? resp.json.error : 'Erreur suppression');
                }
                // Refresh preview + (optionnel) résumé
                self.refreshDataExplorer();
                if (window.quantixLab && typeof window.quantixLab.quickStart === 'function') {
                    try { window.quantixLab.quickStart(); } catch(e) {}
                }
            })
            .catch(function(e){ hideLoading(); showError(e.message || String(e)); });
        };
    }

    function buildTransformPayload() {
        var col = transformCol ? transformCol.value : '';
        var action = transformAction ? transformAction.value : '';
        var mode = transformMode ? transformMode.value : 'new';
        var payload = { column: col, action: action, mode: mode };

        if (mode === 'new' && transformNewCol && transformNewCol.value) {
            payload.new_column = transformNewCol.value;
        }
        if (action === 'replace') {
            payload.find = transformFind ? transformFind.value : '';
            payload.replace = transformReplace ? transformReplace.value : '';
        }

        if (action === 'linear') {
            // "a,b" (compat backend: peut parser value aussi)
            payload.value = transformValue ? String(transformValue.value || '').trim() : '';
        }

        if (['multiply', 'add', 'power'].indexOf(action) !== -1) {
            payload.value = transformValue ? transformValue.value : '';
        }
        if (action === 'round') {
            payload.decimals = transformValue ? transformValue.value : '';
        }
        return payload;
    }

    if (transformApply) {
        transformApply.onclick = function() {
            var payload = buildTransformPayload();
            if (!payload.column || !payload.action) return;

            showLoading('Transformation en cours…');
            fetch('/api/lab/transform', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(function(r){ return r.json().then(function(j){ return { ok: r.ok, json: j }; }); })
            .then(function(resp){
                hideLoading();
                if (!resp.ok || !resp.json || !resp.json.success) {
                    throw new Error((resp.json && resp.json.error) ? resp.json.error : 'Erreur transformation');
                }
                // Recharger l'aperçu pour refléter les changements
                self.refreshDataExplorer();
            })
            .catch(function(e){ hideLoading(); showError(e.message || String(e)); });
        };
    }

    if (transformAction) transformAction.onchange = setHint;
    if (transformMode) transformMode.onchange = setHint;
    setHint();

    // Initial load if session already has a file
    self.refreshDataExplorer();
};

function escapeHtml(s) {
    return String(s || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function openPlotOptions(plotType, catalog, datasetInfo, onConfirm) {
    var modal = document.getElementById('lab-options-modal');
    var title = document.getElementById('lab-options-title');
    var body = document.getElementById('lab-options-body');
    var btnCancel = document.getElementById('lab-options-cancel');
    var btnConfirm = document.getElementById('lab-options-confirm');

    var plotDef = (catalog && catalog.plot_types) ? catalog.plot_types[plotType] : null;
    if (!plotDef) {
        showError('Plot type introuvable: ' + plotType);
        return;
    }

    title.textContent = 'Options — ' + (plotDef.label || plotType);
    body.innerHTML = '';

    var allCols = datasetInfo.columns || [];
    var numeric = datasetInfo.numeric_columns || [];
    var categorical = datasetInfo.categorical_columns || [];
    var textCols = datasetInfo.text_columns || [];
    var datetimeCols = datasetInfo.datetime_columns || [];
    var numericOrDatetime = numeric.concat(datetimeCols.filter(function(c){ return numeric.indexOf(c) === -1; }));

    function addField(labelText, innerHTML) {
        var wrapper = document.createElement('div');
        wrapper.style.marginBottom = '10px';
        var label = document.createElement('label');
        label.style.display = 'block';
        label.style.marginBottom = '6px';
        label.textContent = labelText;
        wrapper.appendChild(label);
        var container = document.createElement('div');
        container.innerHTML = innerHTML;
        wrapper.appendChild(container);
        body.appendChild(wrapper);
        return container;
    }

    function optionsHtml(list) {
        return (list || []).map(function(c){ return '<option value="'+escapeHtml(c)+'">'+escapeHtml(c)+'</option>'; }).join('');
    }

    var required = plotDef.required || [];
    var optional = plotDef.optional || [];
    var params = required.concat(optional);

    // Core style controls
    addField('Titre (optionnel)', '<input id="plot-opt-title" type="text" placeholder="Titre" style="width:100%; padding:6px;"/>');
    addField('Couleur (optionnel)', '<input id="plot-opt-color" type="text" value="' + escapeHtml((catalog.defaults && catalog.defaults.color) ? catalog.defaults.color : '#1f77b4') + '" style="width:220px; padding:6px;"/>');
    addField('Opacité', '<input id="plot-opt-opacity" type="number" min="0" max="1" step="0.05" value="' + escapeHtml((catalog.defaults && catalog.defaults.opacity) ? catalog.defaults.opacity : 0.85) + '" style="width:120px; padding:6px;"/>');

    // Option: appliquer le filtre actif du Data Explorer (si présent)
    var activeQuery = null;
    try {
        if (window.quantixLab && typeof window.quantixLab.getActiveQueryPayload === 'function') {
            activeQuery = window.quantixLab.getActiveQueryPayload();
        }
    } catch (e) {
        activeQuery = null;
    }
    if (activeQuery) {
        var hint = [];
        if (activeQuery.q) hint.push('q="' + String(activeQuery.q) + '"');
        if (activeQuery.filters && activeQuery.filters.length) hint.push(String(activeQuery.filters.length) + ' filtre(s)');
        addField('Sous-ensemble (filtre actif)',
            '<label style="display:flex; gap:8px; align-items:center;">' +
                '<input id="plot-opt-use-active-query" type="checkbox" checked/> ' +
                '<span>Appliquer le filtre actif du Data Explorer</span>' +
            '</label>' +
            '<div style="margin-top:6px; color:#6b7280; font-size:0.85rem;">' + escapeHtml(hint.join(' • ')) + '</div>'
        );
    }

    params.forEach(function(p) {
        var id = 'plot-opt-' + p;

        // Column selectors
        if (p === 'x') {
            addField('X', '<select id="'+id+'" style="width:100%; padding:6px;">' + optionsHtml(numericOrDatetime.length ? numericOrDatetime : allCols) + '</select>');
            return;
        }
        if (p === 'y') {
            addField('Y', '<select id="'+id+'" style="width:100%; padding:6px;">' + optionsHtml(numeric.length ? numeric : allCols) + '</select>');
            return;
        }
        if (p === 'color_by') {
            addField('Colorer par (optionnel)', '<select id="'+id+'" style="width:100%; padding:6px;"><option value="">(aucun)</option>' + optionsHtml(allCols) + '</select>');
            return;
        }
        if (p === 'group_by') {
            addField('Grouper par', '<select id="'+id+'" style="width:100%; padding:6px;">' + optionsHtml(categorical.length ? categorical : allCols) + '</select>');
            return;
        }
        if (p === 'text') {
            addField('Colonne texte', '<select id="'+id+'" style="width:100%; padding:6px;">' + optionsHtml(textCols.length ? textCols : allCols) + '</select>');
            return;
        }

        // Multi-selects
        if (p === 'dimensions' || p === 'metrics') {
            addField(p === 'dimensions' ? 'Dimensions (Ctrl/Cmd)' : 'Métriques (Ctrl/Cmd)', '<select id="'+id+'" multiple style="width:100%; padding:6px; height:140px;">' + optionsHtml(numeric.length ? numeric : allCols) + '</select>');
            return;
        }
        if (p === 'stats_lines') {
            addField('Lignes stats (Ctrl/Cmd)', '<select id="'+id+'" multiple style="width:100%; padding:6px; height:120px;">' + optionsHtml(catalog.stats_lines || []) + '</select>');
            return;
        }
        if (p === 'compare_distributions') {
            addField('Comparer distributions (Ctrl/Cmd)', '<select id="'+id+'" multiple style="width:100%; padding:6px; height:120px;">' + optionsHtml(catalog.known_distributions || []) + '</select>');
            return;
        }

        // Enums
        if (p === 'normalize') {
            // Ambiguïté volontaire: histogramme -> histnorm (str), radar -> normalisation min-max (bool)
            if (plotType === 'radar') {
                addField('Normaliser (min-max)', '<label style="display:flex; gap:8px; align-items:center;"><input id="'+id+'" type="checkbox" checked/> <span>Activer</span></label>');
                return;
            }

            addField('Normalisation', '<select id="'+id+'" style="width:100%; padding:6px;">' +
                '<option value="">(aucune)</option>' +
                '<option value="probability">probability</option>' +
                '<option value="percent">percent</option>' +
                '<option value="density">density</option>' +
                '<option value="probability density">probability density</option>' +
            '</select>');
            return;
        }
        if (p === 'agg') {
            addField('Agrégation', '<select id="'+id+'" style="width:220px; padding:6px;">' +
                '<option value="mean">mean</option>' +
                '<option value="median">median</option>' +
                '<option value="min">min</option>' +
                '<option value="max">max</option>' +
            '</select>');
            return;
        }
        if (p === 'method') {
            addField('Méthode', '<select id="'+id+'" style="width:220px; padding:6px;">' +
                '<option value="pearson">pearson</option>' +
                '<option value="spearman">spearman</option>' +
                '<option value="kendall">kendall</option>' +
            '</select>');
            return;
        }
        if (p === 'regression') {
            addField('Régression', '<select id="'+id+'" style="width:100%; padding:6px;"><option value="">(aucune)</option>' + optionsHtml(catalog.regression || []) + '</select>');
            return;
        }
        if (p === 'gradient_mode') {
            addField('Mode gradient', '<select id="'+id+'" style="width:100%; padding:6px;">' + optionsHtml(catalog.gradient_mode || []) + '</select>');
            return;
        }
        if (p === 'orientation') {
            addField('Orientation', '<select id="'+id+'" style="width:180px; padding:6px;"><option value="v">v</option><option value="h">h</option></select>');
            return;
        }
        if (p === 'points') {
            addField('Points (box/violin)', '<select id="'+id+'" style="width:220px; padding:6px;">' +
                '<option value="outliers">outliers</option>' +
                '<option value="all">all</option>' +
                '<option value="suspectedoutliers">suspectedoutliers</option>' +
                '<option value="">(none)</option>' +
            '</select>');
            return;
        }
        if (p === 'trendline') {
            addField('Trendline', '<select id="'+id+'" style="width:220px; padding:6px;">' +
                '<option value="">(aucune)</option>' +
                '<option value="linear">linear</option>' +
                '<option value="ols">ols</option>' +
            '</select>');
            return;
        }

        // Booleans
        var boolParams = {
            log_x: true, log_y: true, rug: true, fit_line: true, markers: true,
            diagonal_visible: true, fill: true,
            show_stationary_points: true, show_inflection_points: true, show_fixed_points: true,
            show_gradient: true, highlight_extrema: true,
            compare_distributions_auto: false,
            include_zero: false
        };
        if (boolParams.hasOwnProperty(p)) {
            addField(p.replace(/_/g, ' '), '<label style="display:flex; gap:8px; align-items:center;"><input id="'+id+'" type="checkbox"/> <span>Activer</span></label>');
            return;
        }

        // Special: tangent_at = liste de x (a,b,c)
        if (p === 'tangent_at') {
            addField('Tangente en x (a,b,c)', '<input id="'+id+'" type="text" placeholder="ex: 12.5, 42, 80" style="width:100%; padding:6px;"/>');
            return;
        }

        // Numbers
        var numberParams = {
            nbins: 30, nbinsx: 40, nbinsy: 40, top_n: 30, ngram: 1, max_words: 150,
            min_token_len: 2, bandwidth: '',
            window: 10, lag: 1, nlags: 40,
            highlight_top_residuals: 0
        };
        if (numberParams.hasOwnProperty(p)) {
            var val = numberParams[p];
            addField(p.replace(/_/g, ' '), '<input id="'+id+'" type="number" value="'+escapeHtml(val)+'" style="width:160px; padding:6px;"/>');
            return;
        }

        // Fallback: string
        addField(p.replace(/_/g, ' '), '<input id="'+id+'" type="text" style="width:100%; padding:6px;"/>');
    });

    modal.style.display = 'flex';

    btnCancel.onclick = function() { modal.style.display = 'none'; };
    btnConfirm.onclick = function() {
        var spec = { plot_type: plotType };
        var t = document.getElementById('plot-opt-title');
        var c = document.getElementById('plot-opt-color');
        var o = document.getElementById('plot-opt-opacity');
        if (t && t.value) spec.title = t.value;
        if (c && c.value) spec.color = c.value;
        if (o && o.value !== '') spec.opacity = parseFloat(o.value);

        params.forEach(function(p) {
            var el = document.getElementById('plot-opt-' + p);
            if (!el) return;

            if (p === 'normalize' && plotType === 'radar' && el.type === 'checkbox') {
                spec[p] = !!el.checked;
                return;
            }

            if (el.tagName === 'SELECT' && el.multiple) {
                var values = Array.from(el.selectedOptions).map(function(opt){ return opt.value; }).filter(function(v){ return v !== ''; });
                if (values.length) spec[p] = values;
                return;
            }

            if (el.type === 'checkbox') {
                if (el.checked) spec[p] = true;
                return;
            }

            var v = el.value;
            if (v === '' || v == null) return;

            if (p === 'tangent_at') {
                // Accepte "a,b,c" -> [a,b,c]
                var xs = String(v)
                    .split(',')
                    .map(function(s){ return s.trim(); })
                    .filter(function(s){ return s.length > 0; })
                    .map(function(s){ return Number(s); })
                    .filter(function(n){ return !isNaN(n); });
                if (xs.length) spec[p] = xs;
                return;
            }

            // numeric conversion for known keys
            if (['nbins','nbinsx','nbinsy','top_n','ngram','max_words','min_token_len','bandwidth','window','lag','nlags','highlight_top_residuals'].indexOf(p) !== -1) {
                var nv = Number(v);
                if (!isNaN(nv)) spec[p] = nv;
                return;
            }

            spec[p] = v;
        });

        // Brancher la requête active si demandé
        var useActive = document.getElementById('plot-opt-use-active-query');
        if (useActive && useActive.checked && activeQuery) {
            spec.query = activeQuery;
        }

        modal.style.display = 'none';
        if (typeof onConfirm === 'function') onConfirm(spec);
    };
}

// Helper to open the options modal and populate fields based on tool
function openLabOptions(toolName, datasetInfo, onConfirm) {
    var modal = document.getElementById('lab-options-modal');
    var title = document.getElementById('lab-options-title');
    var body = document.getElementById('lab-options-body');
    var btnCancel = document.getElementById('lab-options-cancel');
    var btnConfirm = document.getElementById('lab-options-confirm');

    var toolTitles = {
        'histogram': 'Histogramme',
        'scatter-plot': 'Nuage de points',
        'box-plot': 'Boxplot',
        'correlation': 'Matrice de corrélation'
    };
    title.textContent = 'Options — ' + (toolTitles[toolName] || toolName);
    body.innerHTML = '';

    var numeric = datasetInfo.numeric_columns || [];

    // On ne propose que des options stables (classiques).
    var allowed = ['histogram', 'scatter-plot', 'box-plot', 'correlation'];
    if (allowed.indexOf(toolName) === -1) {
        showError('Bientôt disponible');
        return;
    }

    // Common fields
    function addField(labelText, innerHTML) {
        var wrapper = document.createElement('div');
        wrapper.style.marginBottom = '10px';
        var label = document.createElement('label');
        label.style.display = 'block';
        label.style.marginBottom = '6px';
        label.textContent = labelText;
        wrapper.appendChild(label);
        var container = document.createElement('div');
        container.innerHTML = innerHTML;
        wrapper.appendChild(container);
        body.appendChild(wrapper);
        return container;
    }

    // Build fields depending on tool
    if (['histogram'].indexOf(toolName) !== -1) {
        // select column
        var opts = numeric.map(function(c){ return '<option value="'+c+'">'+c+'</option>'; }).join('');
        var sel = addField('Colonne numérique', '<select id="lab-opt-column" style="width:100%; padding:6px;">'+opts+'</select>');
        // bins
        if (toolName === 'histogram') addField('Bins', '<input id="lab-opt-bins" type="number" value="30" style="width:120px; padding:6px;"/>');
    }

    if (toolName === 'scatter-plot') {
        var opts = numeric.map(function(c){ return '<option value="'+c+'">'+c+'</option>'; }).join('');
        addField('X (axe horizontal)', '<select id="lab-opt-x" style="width:100%; padding:6px;">'+opts+'</select>');
        addField('Y (axe vertical)', '<select id="lab-opt-y" style="width:100%; padding:6px;">'+opts+'</select>');
    }

    if (toolName === 'box-plot') {
        // allow selecting up to 6 columns
        var opts = numeric.map(function(c){ return '<option value="'+c+'">'+c+'</option>'; }).join('');
        addField('Colonnes (sélection multiple, Ctrl/Cmd)', '<select id="lab-opt-columns" multiple style="width:100%; padding:6px; height:120px;">'+opts+'</select>');
    }

    if (toolName === 'correlation') {
        var opts = numeric.map(function(c){ return '<option value="'+c+'">'+c+'</option>'; }).join('');
        addField('Colonnes (sélection multiple, Ctrl/Cmd)', '<select id="lab-opt-columns" multiple style="width:100%; padding:6px; height:140px;">'+opts+'</select>');
        addField('Méthode', '<select id="lab-opt-method" style="width:100%; padding:6px;"><option value="pearson">pearson</option><option value="spearman">spearman</option><option value="kendall">kendall</option></select>');
    }

    // default selections
    if (numeric.length > 0) {
        var el = document.getElementById('lab-opt-column') || document.getElementById('lab-opt-x') || document.getElementById('lab-opt-columns');
        if (el && el.options && el.options.length>0) el.selectedIndex = 0;
        var elY = document.getElementById('lab-opt-y'); if (elY && elY.options && elY.options.length>1) elY.selectedIndex = Math.min(1, elY.options.length-1);
    }

    // show modal
    modal.style.display = 'flex';

    btnCancel.onclick = function() { modal.style.display = 'none'; };
    btnConfirm.onclick = function() {
        // collect options
        var opts = {};
        if (document.getElementById('lab-opt-column')) opts.column = document.getElementById('lab-opt-column').value;
        if (document.getElementById('lab-opt-bins')) opts.bins = parseInt(document.getElementById('lab-opt-bins').value) || 30;
        if (document.getElementById('lab-opt-x')) opts.x_column = document.getElementById('lab-opt-x').value;
        if (document.getElementById('lab-opt-y')) opts.y_column = document.getElementById('lab-opt-y').value;
        if (document.getElementById('lab-opt-method')) opts.method = document.getElementById('lab-opt-method').value;
        if (document.getElementById('lab-opt-columns')) {
            var sel = document.getElementById('lab-opt-columns');
            opts.columns = Array.from(sel.selectedOptions).map(function(o){return o.value;});
        }

        modal.style.display = 'none';
        if (typeof onConfirm === 'function') onConfirm(opts);
    };
}

// Fonctions utilitaires
function showLoading(message) {
    var loader = document.getElementById('lab-loader');
    if (!loader) {
        loader = document.createElement('div');
        loader.id = 'lab-loader';
        loader.style.cssText = 'position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(0,0,0,0.8); color: white; padding: 20px 30px; border-radius: 10px; z-index: 9999; text-align: center;';
        document.body.appendChild(loader);
    }
    
    loader.innerHTML = '<div style="display: inline-block; width: 20px; height: 20px; border: 3px solid #ffffff30; border-radius: 50%; border-top: 3px solid #ffffff; animation: spin 1s linear infinite; margin-right: 10px;"></div>' + message;
    
    if (!document.getElementById('spinner-styles')) {
        var style = document.createElement('style');
        style.id = 'spinner-styles';
        style.textContent = '@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }';
        document.head.appendChild(style);
    }
    
    loader.style.display = 'block';
}

function hideLoading() {
    var loader = document.getElementById('lab-loader');
    if (loader) {
        loader.style.display = 'none';
    }
}

function showResult(data, title) {
    var resultsArea = document.getElementById('lab-results');
    if (!resultsArea) {
        resultsArea = document.createElement('div');
        resultsArea.id = 'lab-results';
        resultsArea.style.cssText = 'position: fixed; top: 10%; left: 10%; width: 80%; height: 80%; background: white; border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); z-index: 9998; overflow: auto; padding: 20px;';
        document.body.appendChild(resultsArea);
    }
    
    var content = '<div style="border-bottom: 1px solid #eee; margin-bottom: 20px; padding-bottom: 10px;"><h2 style="margin: 0; color: #333;">' + title + '</h2><button onclick="document.getElementById(\'lab-results\').style.display=\'none\'" style="float: right; background: #f44336; color: white; border: none; padding: 5px 10px; border-radius: 5px; cursor: pointer;">Fermer</button></div>';

    // Mettre en avant la formule si elle existe (polynôme / distributions)
    if (data && data.formula) {
        content += '<div style="margin: 10px 0 16px; padding: 12px 14px; background:#f8fafc; border:1px solid #e5e7eb; border-radius:8px;">';
        content += '<div style="font-weight:600; color:#111827; margin-bottom:6px;">Formule</div>';
        content += '<div style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; white-space: pre-wrap; color:#111827;">' + String(data.formula) + '</div>';
        content += '<div style="margin-top:6px; font-size:0.85rem; color:#6b7280;">(Format LaTeX — affiché en texte si MathJax/KaTeX n\'est pas chargé)</div>';
        content += '</div>';
    }
    
    if (data.plotly_json) {
        var containerId = 'plotly-container-' + Date.now();
        content += '<div id="' + containerId + '" style="width: 100%; height: 400px;"></div>';
        resultsArea.innerHTML = content;
        
        var plotData = JSON.parse(data.plotly_json);
        Plotly.newPlot(containerId, plotData.data, plotData.layout);
    } else {
        // If backend returned an image URL, show the image first
        if (data.plot_url) {
            // add export buttons: download PNG and export to PDF
            var pngName = data.plot_url.split('/').pop();
            var pngDownload = data.plot_url + (data.plot_url.indexOf('?') === -1 ? '?' + Date.now() : '&' + Date.now());
            content += '<div style="text-align:center; margin-bottom:12px;">';
            content += '<img src="' + data.plot_url + '" style="max-width:100%; height:auto; border:1px solid #eee; border-radius:6px;"/>';
            content += '<div style="margin-top:8px; display:flex; gap:8px; justify-content:center;">';
            content += '<a class="btn" href="' + data.plot_url + '" download="' + pngName + '">Télécharger PNG</a>';
            content += '<button class="btn" id="export-pdf-btn">Exporter en PDF</button>';
            content += '</div></div>';
        }

        // If backend returned a dendrogram image, show it too
        if (data.dendrogram_url) {
            var dendroName = data.dendrogram_url.split('/').pop();
            content += '<div style="text-align:center; margin:16px 0 12px;">';
            content += '<h3 style="margin: 0 0 10px; color:#333; font-size:16px;">Dendrogramme</h3>';
            content += '<img src="' + data.dendrogram_url + '" style="max-width:100%; height:auto; border:1px solid #eee; border-radius:6px;"/>';
            content += '<div style="margin-top:8px; display:flex; gap:8px; justify-content:center;">';
            content += '<a class="btn" href="' + data.dendrogram_url + '" download="' + dendroName + '">Télécharger PNG</a>';
            content += '</div></div>';
        }
        content += '<pre style="background: #f5f5f5; padding: 15px; border-radius: 5px; overflow: auto;">' + JSON.stringify(data, null, 2) + '</pre>';
        resultsArea.innerHTML = content;
    }
    
    resultsArea.style.display = 'block';

    // bind export to PDF button if present
    var exportBtn = document.getElementById('export-pdf-btn');
    if (exportBtn) {
        exportBtn.onclick = function() {
            if (!data.plot_url) return showError('URL du graphique introuvable');
            showLoading('Export en PDF en cours...');
            fetch('/api/export/plot-to-pdf', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ plot_url: data.plot_url })
            })
            .then(r => r.json())
            .then(function(resp) {
                hideLoading();
                if (resp.success && resp.pdf_url) {
                    // trigger download
                    var a = document.createElement('a');
                    a.href = resp.pdf_url;
                    a.download = resp.pdf_url.split('/').pop();
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                } else {
                    showError(resp.error || 'Erreur lors de la conversion en PDF');
                }
            }).catch(function(e){ hideLoading(); showError(e.message);});
        };
    }
}


// New high-level tools mapped to modules/sciences endpoints
quantixLab.runPCA = function(datasetInfo) {
    showLoading('ACP (PCA) en cours...');
    fetch('/api/sciences/pca', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_components: 2 })
    })
    .then(r => r.json())
    .then(function(data) {
        hideLoading();
        if (data.success) {
            showResult(data, 'ACP (PCA)');
        } else {
            showError(data.error);
        }
    }).catch(function(e){ hideLoading(); showError(e.message);});
};

quantixLab.runICA = function(datasetInfo) {
    showLoading('ICA en cours...');
    fetch('/api/sciences/ica', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_components: 2 })
    })
    .then(r => r.json())
    .then(function(data) {
        hideLoading();
        if (data.success) showResult(data, 'ICA'); else showError(data.error);
    }).catch(function(e){ hideLoading(); showError(e.message);});
};

quantixLab.runAFC = function(datasetInfo) {
    showLoading('AFC en cours...');
    fetch('/api/sciences/afc', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) })
    .then(r => r.json()).then(function(data){ hideLoading(); if (data.success) showResult(data, 'AFC'); else showError(data.error); }).catch(function(e){ hideLoading(); showError(e.message);});
};

quantixLab.runSpectral = function(datasetInfo) {
    showLoading('Analyse spectrale / corrélations...');
    fetch('/api/sciences/spectral-summary', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) })
    .then(r => r.json()).then(function(data){ hideLoading(); if (data.success) showResult(data, 'Analyse spectrale'); else showError(data.error); }).catch(function(e){ hideLoading(); showError(e.message);});
};

quantixLab.runDistributionCompare = function(datasetInfo, options) {
    showLoading('Comparaison de distribution...');
    var column = (options && options.column) ? options.column : (datasetInfo.numeric_columns[0] || null);
    var compare_to = (options && options.compare_to) ? options.compare_to : null;
    var payload = { column: column };
    if (compare_to) payload.compare_to = compare_to;
    fetch('/api/lab/distribution-compare', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
    .then(r => r.json()).then(function(data){ hideLoading(); if (data.success) showResult(data, 'Comparaison de distribution'); else showError(data.error); }).catch(function(e){ hideLoading(); showError(e.message);});
};

quantixLab.runMultivariate = function(datasetInfo) {
    showLoading('Analyse multivariée (rapport)...');
    fetch('/api/sciences/multivariate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'report' }) })
    .then(r => r.json()).then(function(data){ hideLoading(); if (data.success) showResult(data, 'Analyse multivariée'); else showError(data.error); }).catch(function(e){ hideLoading(); showError(e.message);});
};

quantixLab.runStatisticalDescribe = function(datasetInfo) {
    showLoading('Statistiques descriptives et tests...');
    fetch('/api/sciences/statistical-describe', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) })
    .then(r => r.json()).then(function(data){ hideLoading(); if (data.success) showResult(data, 'Analyse statistique'); else showError(data.error); }).catch(function(e){ hideLoading(); showError(e.message);});
};

quantixLab.runClustering = function(datasetInfo) {
    showLoading('Clustering hiérarchique...');
    fetch('/api/sciences/spectral-summary', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) })
    .then(r => r.json()).then(function(data){ hideLoading(); if (data.success) showResult(data, 'Clustering (dendrogramme)'); else showError(data.error); }).catch(function(e){ hideLoading(); showError(e.message);});
};

quantixLab.runMultipleCorrelation = function(datasetInfo) {
    showLoading('Corrélation multiple...');
    var target = datasetInfo.numeric_columns[0] || null;
    fetch('/api/sciences/multiple-correlation', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ target: target }) })
    .then(r => r.json()).then(function(data){ hideLoading(); if (data.success) showResult(data, 'Corrélation multiple'); else showError(data.error); }).catch(function(e){ hideLoading(); showError(e.message);});
};

function showError(message) {
    var errorDiv = document.getElementById('lab-error');
    if (!errorDiv) {
        errorDiv = document.createElement('div');
        errorDiv.id = 'lab-error';
        errorDiv.style.cssText = 'position: fixed; top: 20px; left: 50%; transform: translateX(-50%); background: #f44336; color: white; padding: 15px 20px; border-radius: 5px; z-index: 9999; max-width: 500px;';
        document.body.appendChild(errorDiv);
    }
    
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    
    setTimeout(function() {
        errorDiv.style.display = 'none';
    }, 5000);
}

// ===== MÉTHODES D'OUTILS SPÉCIFIQUES =====

// Ajouter les méthodes aux outils du quantixLab
quantixLab.createScatterPlot = function(datasetInfo, options) {
    var numericColumns = datasetInfo.numeric_columns;
    if (numericColumns.length < 2) {
        showError('Au moins 2 colonnes numériques sont nécessaires pour un nuage de points');
        return;
    }
    
    showLoading('Génération du nuage de points...');
    var xcol = (options && options.x_column) ? options.x_column : numericColumns[0];
    var ycol = (options && options.y_column) ? options.y_column : numericColumns[1];

    var title = 'Nuage de points: ' + xcol + ' vs ' + ycol;
    var payload = { x_column: xcol, y_column: ycol };
    fetch('/api/lab/scatter-plot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(function(resp){ return resp.json(); })
    .then(function(data){
        hideLoading();
        if (data.success) showResult(data, title); else showError(data.error);
    })
    .catch(function(e){ hideLoading(); showError(e.message); });
};

quantixLab.createHistogram = function(datasetInfo, options) {
    var numericColumns = datasetInfo.numeric_columns;
    if (numericColumns.length === 0) {
        showError('Aucune colonne numérique trouvée pour l\'histogramme');
        return;
    }
    
    showLoading('Génération de l\'histogramme...');
    var column = (options && options.column) ? options.column : numericColumns[0];
    var bins = (options && options.bins) ? options.bins : 30;

    var title = 'Histogramme: ' + column;
    var payload = { column: column, bins: bins };
    fetch('/api/lab/histogram', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(function(resp){ return resp.json(); })
    .then(function(data){
        hideLoading();
        if (data.success) showResult(data, title); else showError(data.error);
    })
    .catch(function(e){ hideLoading(); showError(e.message); });
};

quantixLab.createBoxplot = function(datasetInfo, options) {
    var numericColumns = datasetInfo.numeric_columns;
    if (numericColumns.length === 0) {
        showError('Aucune colonne numérique trouvée pour les boîtes à moustaches');
        return;
    }
    
    showLoading('Génération des boîtes à moustaches...');
    var cols = (options && options.columns && options.columns.length>0) ? options.columns : numericColumns.slice(0,5);

    var payload = { columns: cols };
    fetch('/api/lab/boxplot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(function(resp){ return resp.json(); })
    .then(function(data){
        hideLoading();
        if (data.success) showResult(data, 'Boîtes à moustaches'); else showError(data.error);
    })
    .catch(function(e){ hideLoading(); showError(e.message); });
};

quantixLab.createCorrelationMatrix = function(datasetInfo, options) {
    var numericColumns = datasetInfo.numeric_columns;
    if (numericColumns.length < 2) {
        showError('Au moins 2 colonnes numériques sont nécessaires pour la matrice de corrélation');
        return;
    }

    var cols = (options && options.columns && options.columns.length >= 2) ? options.columns : numericColumns;
    var method = (options && options.method) ? options.method : 'pearson';

    showLoading('Calcul de la matrice de corrélation...');
    fetch('/api/lab/correlation-matrix', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ columns: cols, method: method })
    })
    .then(function(resp){ return resp.json(); })
    .then(function(data){
        hideLoading();
        if (data.success) showResult(data, 'Matrice de corrélation'); else showError(data.error);
    })
    .catch(function(e){ hideLoading(); showError(e.message); });
};

quantixLab.polynomialApproximation = function(datasetInfo, options) {
    showLoading('Approximation polynomiale en cours...');
    var column = (options && options.column) ? options.column : (datasetInfo.numeric_columns[0] || 'auto');
    var degree = (options && options.degree) ? options.degree : 6;
    var payload = { column: column, degree: degree };
    // color option removed — server will use default color

    fetch('/api/lab/polynomial-approximation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
        hideLoading();
        if (data.success) {
            showResult(data, 'Approximation polynomiale');
        } else {
            showError(data.error);
        }
    })
    .catch(function(error) {
        hideLoading();
        showError('Erreur: ' + error.message);
    });
};

quantixLab.distributionAnalysis = function(datasetInfo, options) {
    showLoading('Analyse de distribution en cours...');
    var column = (options && options.column) ? options.column : (datasetInfo.numeric_columns[0] || 'auto');
    var payload = { column: column };
    // color option removed — server will use default color
    fetch('/api/lab/distribution-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
        hideLoading();
        if (data.success) {
            showResult(data, 'Analyse de distribution');
        } else {
            showError(data.error);
        }
    })
    .catch(function(error) {
        hideLoading();
        showError('Erreur: ' + error.message);
    });
};

quantixLab.curveFitting = function(datasetInfo, options) {
    showLoading('Ajustement de courbe en cours...');
    var x = (options && options.x_column) ? options.x_column : (datasetInfo.numeric_columns[0] || 'auto');
    var y = (options && options.y_column) ? options.y_column : (datasetInfo.numeric_columns[1] || 'auto');
    var method = (options && options.method) ? options.method : 'polynomial';
    var payload = { x_column: x, y_column: y, method: method };
    // color option removed — server will use default color
    fetch('/api/lab/curve-fitting', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
        hideLoading();
        if (data.success) {
            showResult(data, 'Ajustement de courbe');
        } else {
            showError(data.error);
        }
    })
    .catch(function(error) {
        hideLoading();
        showError('Erreur: ' + error.message);
    });
};

quantixLab.linearRegression = function(datasetInfo, options) {
    showLoading('Régression linéaire en cours...');
    var x = (options && options.x_column) ? options.x_column : (datasetInfo.numeric_columns[0] || 'auto');
    var y = (options && options.y_column) ? options.y_column : (datasetInfo.numeric_columns[1] || 'auto');
    var payload = { x_column: x, y_column: y };
    // color option removed — server will use default color
    fetch('/api/lab/linear-regression', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
        hideLoading();
        if (data.success) {
            showResult(data, 'Régression linéaire');
        } else {
            showError(data.error);
        }
    })
    .catch(function(error) {
        hideLoading();
        showError('Erreur: ' + error.message);
    });
};

quantixLab.timeSeries = function(datasetInfo) {
    showLoading('Analyse de série temporelle...');
    
    fetch('/api/lab/time-series', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            auto_detect: true
        })
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
        hideLoading();
        if (data.success) {
            showResult(data, 'Analyse de série temporelle');
        } else {
            showError(data.error);
        }
    })
    .catch(function(error) {
        hideLoading();
        showError('Erreur: ' + error.message);
    });
};

quantixLab.classification = function(datasetInfo) {
    showLoading('Classification automatique...');
    
    fetch('/api/lab/classification', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            auto_mode: true
        })
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
        hideLoading();
        if (data.success) {
            showResult(data, 'Classification automatique');
        } else {
            showError(data.error);
        }
    })
    .catch(function(error) {
        hideLoading();
        showError('Erreur: ' + error.message);
    });
};

quantixLab.descriptiveStats = function(datasetInfo) {
    showLoading('Calcul des statistiques descriptives...');
    
    fetch('/api/lab/descriptive-stats', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
        hideLoading();
        if (data.success) {
            showResult(data, 'Statistiques descriptives');
        } else {
            showError(data.error);
        }
    })
    .catch(function(error) {
        hideLoading();
        showError('Erreur: ' + error.message);
    });
};

quantixLab.hypothesisTest = function(datasetInfo) {
    showLoading('Test d\'hypothèse en cours...');
    
    fetch('/api/lab/hypothesis-test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            test_type: 'correlation',
            var1: datasetInfo.numeric_columns[0] || 'auto',
            var2: datasetInfo.numeric_columns[1] || 'auto'
        })
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
        hideLoading();
        if (data.success) {
            showResult(data, 'Test d\'hypothèse');
        } else {
            showError(data.error);
        }
    })
    .catch(function(error) {
        hideLoading();
        showError('Erreur: ' + error.message);
    });
};

quantixLab.outlierDetection = function(datasetInfo) {
    showLoading('Détection des valeurs aberrantes...');
    
    fetch('/api/lab/outlier-detection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            method: 'isolation_forest'
        })
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
        hideLoading();
        if (data.success) {
            showResult(data, 'Détection des valeurs aberrantes');
        } else {
            showError(data.error);
        }
    })
    .catch(function(error) {
        hideLoading();
        showError('Erreur: ' + error.message);
    });
};

console.log('Tous les outils Lab sont initialisés et prêts');
