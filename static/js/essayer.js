/**
 * Quantix+ Essayer.js — version condensée et unifiée
 * Gestion simplifiée de l’upload et analyse visuelle initiale
 */

document.addEventListener('DOMContentLoaded', () => {
    const $ = id => document.getElementById(id);
    const els = {
        drop: $('dropZone'), input: $('fileInput'), fileSec: $('fileSection'),
        actions: $('fileActionsSection'), results: $('resultsArea')
    };
    let fileData = null;

    // Variables pour stocker les résultats de traitement
    let processedFiles = {
        cleaned: null,
        converted: null,
        report: null,
        transcription: null,
        translation: null
    };

    // === Utilitaires ===
    const notify = (msg, type='info') => {
        const div = document.createElement('div');
        const color = {info:'#3b82f6',success:'#16a34a',error:'#dc2626'}[type];
        Object.assign(div.style, {
            position:'fixed',top:'20px',right:'20px',background:color,color:'#fff',
            padding:'12px 20px',borderRadius:'8px',zIndex:9999,transition:'transform .3s',
            transform:'translateX(100%)'
        });
        div.textContent = msg; document.body.appendChild(div);
        setTimeout(()=>div.style.transform='translateX(0)',10);
        setTimeout(()=>div.remove(),3000);
    };
    const show = e => e && (e.style.display='block');
    const hide = e => e && (e.style.display='none');

    const api = (url, data, msg, cb) => {
        notify(msg,'info');
        return fetch(url,{
            method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify(data)
        }).then(r=>r.json()).then(j=>j.success?cb(j):notify(j.error,'error'))
        .catch(e=>notify(e.message,'error'));
    };

    // === Upload avec redirection intelligente ===
    function handleFile(file){
        const fd=new FormData();fd.append('file',file);
        fetch('/upload-file',{method:'POST',body:fd})
        .then(r=>r.json())
        .then(j=>{
            if(!j.success) return notify(j.error,'error');
            fileData = j; // Stocker toute la réponse, pas seulement file_info
            // Sauvegarder en sessionStorage pour réutilisation dans les pages 'essayer_pages'
            try{
                const savedName = j.file_info?.name || j.file_info?.nom || null;
                if(savedName){
                    sessionStorage.setItem('uploadedFileName', savedName);
                }
                sessionStorage.setItem('uploadedFile', JSON.stringify(j.file_info || j));
            }catch(e){
                console.warn('Impossible de sauvegarder en sessionStorage', e);
            }
            console.log('Fichier uploadé:', fileData);
            renderFileInfo(file,j);
            show(els.fileSec);
            
            // === REDIRECTION INTELLIGENTE ===
            if(j.should_redirect && j.platform_recommendation){
                const recommendation = j.platform_recommendation;
                
                // Afficher la recommandation intégrée dans la page
                if(recommendation.platform !== 'essayer'){
                    showPlatformRecommendation(recommendation, j.redirect_url);
                } else {
                    notify('Fichier chargé, vous pouvez utiliser les outils de nettoyage ci-dessous', 'success');
                }
            } else {
                notify('Fichier analysé, prêt pour le traitement', 'success');
            }
            // Mettre à jour le bouton Lab selon les stats serveur
            try{ updateLabButton(); }catch(e){console.warn('updateLabButton error',e)}
        })
        .catch(e=>notify(e.message,'error'));
    }
    
    // === Interface de recommandation de plateforme intégrée ===
    function showPlatformRecommendation(recommendation, redirectUrl){
        const platformSection = document.getElementById('platformRecommendation');
        const contentDiv = document.getElementById('recommendationContent');
        
        if(!platformSection || !contentDiv) return;
        
        // Stocker l'URL pour les boutons
        window.currentRedirectUrl = redirectUrl;
        
        // Contenu de la recommandation
        contentDiv.innerHTML = `
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 12px;">
                <div>
                    <strong style="color: white;">Plateforme suggérée:</strong><br>
                    <span style="font-size: 1.1em;">${getPlatformName(recommendation.platform)}</span>
                </div>
                <div>
                    <strong style="color: white;">Confiance:</strong><br>
                    <span style="font-size: 1.1em;">${Math.round(recommendation.confidence * 100)}%</span>
                </div>
            </div>
            <div style="margin-bottom: 12px;">
                <strong style="color: white;">Raison:</strong><br>
                <span>${recommendation.reason}</span>
            </div>
            <div>
                <strong style="color: white;">Action suggérée:</strong><br>
                <span>${getActionDescription(recommendation.suggested_action)}</span>
            </div>
        `;
        
        // Afficher la section avec animation
        platformSection.style.display = 'block';
        
        // Auto-redirection si très haute confiance
        if(recommendation.confidence > 0.90){
            let countdown = 8;
            const acceptBtn = document.getElementById('acceptRedirectBtn');
            const originalText = acceptBtn.textContent;
            
            const timer = setInterval(() => {
                acceptBtn.textContent = `Redirection dans ${countdown}s...`;
                countdown--;
                if(countdown < 0){
                    clearInterval(timer);
                    if(window.currentRedirectUrl) {
                        window.location.href = window.currentRedirectUrl;
                    }
                }
            }, 1000);
            
            // Annuler la redirection automatique si l'utilisateur interagit
            document.getElementById('stayHereBtn').addEventListener('click', () => {
                clearInterval(timer);
                acceptBtn.textContent = originalText;
            }, { once: true });
        }
    }
    
    // === Helpers pour la redirection ===
    function getPlatformName(platform){
        const names = {
            'lab': 'Quantix Lab (Analyse Statistique)',
            'logos': 'Quantix Logos (Analyse Textuelle)', 
            'sonar': 'Quantix Sonar (Transcription Audio)',
            'saphir': 'Quantix Saphir (IA Avancée)',
            'essayer': 'Outils de Nettoyage'
        };
        return names[platform] || platform;
    }
    
    function getActionDescription(action){
        const descriptions = {
            'analyse_statistique': 'Graphiques, corrélations, machine learning',
            'analyse_textuelle': 'Analyse de sentiment, extraction d\'entités',
            'retranscription': 'Conversion audio vers texte',
            'nettoyage_conversion': 'Nettoyage et conversion de format'
        };
        return descriptions[action] || action;
    }
    
    function redirectToPlatform(url){
        if(url) {
            notify('Redirection en cours...', 'info');
            window.location.href = url;
        }
    }

    // === Drag & Drop ===
    els.drop?.addEventListener('click',()=>els.input.click());
    ['dragover','dragleave','drop'].forEach(ev=>{
        els.drop?.addEventListener(ev,e=>{
            e.preventDefault();
            if(ev==='dragover') els.drop.classList.add('dragover');
            if(ev==='dragleave') els.drop.classList.remove('dragover');
            if(ev==='drop'){ els.drop.classList.remove('dragover');
                const f=e.dataTransfer.files[0];f&&handleFile(f);}
        });
    });
    els.input?.addEventListener('change',e=>{
        const f=e.target.files[0];f&&handleFile(f);
    });

    // Si un fichier est déjà uploadé (sessionStorage), vérifier les stats côté serveur
    if(sessionStorage.getItem('uploadedFileName')){
        try{ updateLabButton(); }catch(e){console.warn('updateLabButton init error',e)}
    }

    // === Rendu infos fichier CORRIGÉ ===
    function renderFileInfo(file,data){
        const info=data.file_info,stats=data.analyse_donnees;
        
        // Informations de base du fichier
        const filename = info.nom || info.name || 'file';
        $('fileName').textContent=filename;
        $('fileSize').textContent=info.taille || info.size || '-';

        // Déterminer l'icône selon l'extension (mapping vers les fichiers dans static/image)
        const ext = (filename.includes('.') ? filename.split('.').pop().toLowerCase() : 'file');
        const iconMap = {
            'csv': 'csv.png', 'xls': 'excel.png', 'xlsx': 'excel.png',
            'json': 'csv.png', 'txt': 'txt.png', 'md': 'markdown.png', 'markdown': 'markdown.png',
            'pdf': 'pdf.png', 'doc': 'docx.png', 'docx': 'docx.png',
            'png': 'png.png', 'jpg': 'jpeg.png', 'jpeg': 'jpeg.png', 'gif': 'png.png',
            'mp3': 'mp3.png', 'wav': 'mp3.png', 'm4a': 'mp3.png',
            'mp4': 'mp4.png', 'mov': 'mp4.png',
            'ppt': 'powerpoint.png', 'pptx': 'powerpoint.png',
        };
        const iconFile = iconMap[ext] || 'txt.png';
        $('fileIcon').src = `/static/image/${iconFile}`;
        
        // Informations techniques
        $('fileType').textContent = info.extension || 'Inconnu';
        $('fileEncoding').textContent = info.encodage || 'UTF-8';
        $('fileModified').textContent = info.modifie_le || 'Inconnu';
        
        show(els.fileSec);

        if(data.is_data_file && stats){
            // Afficher les statistiques dans les bons éléments
            $('rowCount').textContent = stats.lignes || '0';
            $('colCount').textContent = stats.colonnes || '0';
            $('missingCount').textContent = stats.valeurs_manquantes || '0';
            $('duplicateCount').textContent = stats.doublons || '0';
            
            // Créer le tableau d'aperçu
            if(data.apercu_donnees && data.apercu_donnees.length > 0) {
                renderPreviewTable(data.apercu_donnees);
            }
            
            // Rendre visible la section des statistiques
            show($('dataStats'));
            show($('technicalInfo'));
            show($('dataPreview'));
            
        } else if(!data.is_data_file){
            // Pour les fichiers non-data, masquer les stats et afficher un message
            hide($('dataStats'));
            $('technicalInfo').innerHTML = `
                <h4 style="font-size: 0.9rem; color: #222; margin-bottom: 8px;">Fichier multimédia</h4>
                <div class="alert" style="background: #fef3c7; border: 1px solid #f59e0b; color: #92400e; padding: 12px; border-radius: 6px;">
                    <p>📁 ${info.nom} (${info.extension}, ${info.taille})</p>
                    <p><small>Utilisez les plateformes spécialisées pour traiter ce type de fichier.</small></p>
                </div>
            `;
            hide($('dataPreview'));
        }
    }
    
    // === Rendu du tableau d'aperçu ===
    function renderPreviewTable(previewData) {
        const table = $('previewTable');
        if (!table || !previewData.length) return;
        
        const columns = Object.keys(previewData[0]);
        
        // Header
        let html = '<thead><tr>';
        columns.forEach(col => {
            html += `<th style="padding: 8px; border-bottom: 1px solid #e5e7eb; background: #f9fafb; font-weight: 600; text-align: left;">${col}</th>`;
        });
        html += '</tr></thead><tbody>';
        
        // Rows (max 5)
        previewData.slice(0, 5).forEach(row => {
            html += '<tr>';
            columns.forEach(col => {
                const value = row[col] || '';
                const displayValue = String(value).length > 30 ? String(value).substring(0, 30) + '...' : value;
                html += `<td style="padding: 8px; border-bottom: 1px solid #f3f4f6;">${displayValue}</td>`;
            });
            html += '</tr>';
        });
        
        html += '</tbody>';
        table.innerHTML = html;
    }

    // === Lab button update ===
    async function updateLabButton(){
        const btn = document.getElementById('openLabBtn');
        const hint = document.getElementById('openLabHint');
        if(!btn || !hint) return;
        try{
            const r = await fetch('/api/file-stats');
            const j = await r.json();
            if(j.success && j.is_data_file && j.has_numeric_column){
                const col = (j.numeric_columns && j.numeric_columns.length)? encodeURIComponent(j.numeric_columns[0]) : null;
                btn.style.display = 'inline-flex';
                hint.style.display = 'inline';
                if(col){
                    btn.onclick = ()=> window.location.href = `/lab?has_numeric=1&numeric_col=${col}`;
                } else {
                    btn.onclick = ()=> window.location.href = '/lab?has_numeric=1';
                }
            } else {
                btn.style.display = 'none';
                hint.style.display = 'none';
            }
        }catch(e){
            console.warn('updateLabButton failed', e);
            btn.style.display = 'none';
            hint.style.display = 'none';
        }
    }

    // === Bloc d’analyse visuelle type Palantir ===
    function renderPalantirAnalysis(d){
        const s=d.analyse_donnees;
        let html=`
        <div class="palantir-analysis-container">
          <div class="analysis-header">
            <h2>${d.file_info.nom}</h2><div class="status-badge">Ready</div>
          </div>
          <div class="metrics-grid">
            <div class="metric-card"><span>Records</span><p>${s.lignes}</p></div>
            <div class="metric-card"><span>Features</span><p>${s.colonnes}</p></div>
            <div class="metric-card"><span>Missing</span><p>${s.valeurs_manquantes}</p></div>
            <div class="metric-card"><span>Memory</span><p>${s.memoire_mb} MB</p></div>
          </div>`;
        if(d.colonnes_detaillees?.length){
            html+=`<div class="schema-table"><h3>Schema</h3>`;
            d.colonnes_detaillees.forEach(c=>{
                html+=`<div class="schema-row">
                  <span>${c.nom}</span><span>${c.type}</span>
                  <span>${c.valeurs_uniques||'-'} uniques</span></div>`;
            });
            html+='</div>';
        }
        if(d.apercu_donnees?.length){
            const cols=Object.keys(d.apercu_donnees[0]);
            html+='<table class="palantir-table"><thead><tr>'+
            cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead><tbody>'+
            d.apercu_donnees.slice(0,10).map((r,i)=>
                `<tr class="${i%2?'odd':'even'}">`+
                cols.map(c=>`<td>${r[c]??'<span class="null">null</span>'}</td>`).join('')+
                '</tr>').join('')+'</tbody></table>';
        }
        html+='</div>'; els.results.innerHTML=html;

        if(d.is_purely_numeric){
            notify('Dataset numérique détecté → Lab','success');
            setTimeout(()=>window.location.href='/lab',2500);
        }
    }

    // === Actions principales avec délégation d'événements ===
    
    // Variables pour stocker les résultats de traitement (déjà déclarée plus haut)
    
    // Délégation d'événements pour tous les boutons
    document.addEventListener('click', (e) => {
        console.log('Click détecté sur:', e.target.id, e.target.className);
        
        // Bouton d'acceptation de redirection
        if(e.target.id === 'acceptRedirectBtn') {
            console.log('Redirection acceptée');
            if(window.currentRedirectUrl) {
                notify('Redirection vers la plateforme recommandée...', 'info');
                window.location.href = window.currentRedirectUrl;
            }
            return;
        }
        
        // Bouton pour rester sur la page
        if(e.target.id === 'stayHereBtn') {
            console.log('Utilisateur reste sur la page');
            document.getElementById('platformRecommendation').style.display = 'none';
            notify('Vous pouvez utiliser les outils de nettoyage ci-dessous', 'info');
            return;
        }
        
        // Bouton de nettoyage
        if(e.target.id === 'cleanBtn') {
            console.log('Bouton nettoyage cliqué');
            handleCleanData();
        }
        
        // Bouton de conversion
        if(e.target.id === 'convertBtn') {
            console.log('Bouton conversion cliqué');
            handleConvertData();
        }

        // Bouton de transcription
        if(e.target.id === 'transcribeBtn') {
            console.log('Bouton transcrire cliqué');
            handleTranscription();
        }

        // Bouton de traduction
        if(e.target.id === 'translateBtn' || e.target.id === 'translate-btn') {
            console.log('Bouton traduire cliqué');
            handleTranslation();
        }
        
        // Boutons de téléchargement
        if(e.target.id === 'downloadCleanedBtn' && processedFiles.cleaned) {
            downloadFile(processedFiles.cleaned, 'fichier_nettoye');
        }
        if(e.target.id === 'downloadConvertedBtn' && processedFiles.converted) {
            downloadFile(processedFiles.converted, 'fichier_converti');
        }
        if(e.target.id === 'downloadReportBtn' && processedFiles.report) {
            downloadFile(processedFiles.report, 'rapport_analyse');
        }
    });
    
    // Fonction de nettoyage des données
    function handleCleanData() {
        if(!fileData) {
            notify('Aucun fichier uploadé', 'error');
            return;
        }
        
        // Masquer la recommandation de plateforme
        const platformRec = document.getElementById('platformRecommendation');
        if(platformRec) platformRec.style.display = 'none';
        
        notify('Nettoyage en cours...', 'info');
        api('/nettoyer-mvp', {}, 'Nettoyage des données', (res) => {
            if(res.success) {
                processedFiles.cleaned = res.download_url;
                processedFiles.report = res.report_url;
                showDownloadOptions('cleaned');
                notify('Nettoyage terminé avec succès!', 'success');
            }
        });
    }

    // === Transcription ===
    async function handleTranscription(){
        if(!fileData) return notify('Aucun fichier uploadé', 'error');
        notify('Retranscription en cours...', 'info');
        const transBox = document.getElementById('transResultBox');
        const container = document.getElementById('transTranslateSection');
        transBox.style.display = 'none';
        container.style.display = 'block';

        try{
            const r = await fetch('/transcribe', {
                method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({})
            });
            const j = await r.json();
            if(j.success){
                processedFiles.transcription = j.transcription || '';
                let downloadHtml = '';
                if(j.download_url){
                    downloadHtml = `<a href="${j.download_url}" target="_blank" class="btn btn-secondary">Télécharger .docx</a>`;
                }
                transBox.innerHTML = `<h4>Transcription</h4><pre style="white-space:pre-wrap;">${escapeHtml(processedFiles.transcription)}</pre><p>${downloadHtml}</p>`;
                transBox.style.display = 'block';
                // activer bouton Traduire
                const transBtn = document.getElementById('translateBtn');
                if(transBtn) transBtn.disabled = false;
                notify('Transcription terminée', 'success');
            } else {
                transBox.innerHTML = `<div class=\"error-box\">Erreur: ${j.error}</div>`;
                transBox.style.display = 'block';
            }
        }catch(err){
            transBox.innerHTML = `<div class=\"error-box\">Erreur réseau: ${err.message}</div>`;
            transBox.style.display = 'block';
            notify('Erreur lors de la transcription','error');
        }
    }

    // === Traduction ===
    async function handleTranslation(){
        if(!processedFiles.transcription) return notify('Aucune transcription disponible', 'error');
        const target = prompt('Langue cible (code, ex: fr, en, es):', 'fr');
        if(!target) return;
        notify('Traduction en cours... Cela peut prendre quelques secondes à plusieurs minutes selon la taille du document.', 'info');
        const transBox = document.getElementById('translateResultBox');
        transBox.style.display = 'none';
        // Récupérer le nom du fichier uploadé
        const uploadedFileName = sessionStorage.getItem('uploadedFileName');
        if(!uploadedFileName){
            notify('Aucun fichier DOCX uploadé','error');
            return;
        }
        let fileToSend = null;
        // Essayer de récupérer le blob depuis fileData
        if(fileData && fileData.file_blob){
            fileToSend = new File([fileData.file_blob], uploadedFileName);
        } else {
            // Sinon, récupérer depuis l'input file
            const input = document.getElementById('fileInput');
            if(input && input.files && input.files.length > 0){
                fileToSend = input.files[0];
            } else {
                notify('Impossible de retrouver le fichier à traduire. Merci de le recharger.', 'error');
                return;
            }
        }
        try{
            const fd = new FormData();
            fd.append('file', fileToSend);
            fd.append('target_lang', target);
            fd.append('preserve_foreign_quotes', 'true');
            // Appel API traduction
            const r = await fetch('/api/translate', {method:'POST', body:fd});
            const j = await r.json();
            if(j.success && j.task_id){
                transBox.innerHTML = `<h4>Traduction (${target})</h4><p>La traduction est en cours. Vous pourrez la télécharger une fois terminée.</p>`;
                transBox.style.display = 'block';
                notify('Traduction lancée', 'success');
                // Vérifier régulièrement le statut de la traduction
                let intervalId = setInterval(async () => {
                    const statusRes = await fetch(`/api/translate/progress/${j.task_id}`);
                    const statusJson = await statusRes.json();
                    if(statusJson.status === 'completed' && statusJson.output_path){
                        clearInterval(intervalId);
                        const downloadUrl = `/download/${statusJson.output_path}`;
                        transBox.innerHTML = `<h4>Traduction (${target}) terminée</h4><a href="${downloadUrl}" target="_blank" class="btn btn-success">Télécharger le fichier traduit</a>`;
                        transBox.style.display = 'block';
                        notify('Traduction terminée, prêt à télécharger','success');
                    } else if(statusJson.status === 'error'){
                        clearInterval(intervalId);
                        transBox.innerHTML = `<div class=\"error-box\">Erreur: ${statusJson.error}</div>`;
                        transBox.style.display = 'block';
                        notify('Erreur lors de la traduction','error');
                    }
                }, 3000);
            } else {
                transBox.innerHTML = `<div class=\"error-box\">Erreur: ${j.error}</div>`;
                transBox.style.display = 'block';
            }
        }catch(err){
            transBox.innerHTML = `<div class=\"error-box\">Erreur réseau: ${err.message}</div>`;
            transBox.style.display = 'block';
            notify('Erreur lors de la traduction','error');
        }
    }

    function escapeHtml(str){
        if(!str) return '';
        return String(str).replace(/[&<>"']/g, function(s){
            return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"}[s];
        });
    }
    
    // Fonction de conversion des données
    function handleConvertData() {
        if(!fileData) {
            notify('Aucun fichier uploadé', 'error');
            return;
        }
        
        // Masquer la recommandation de plateforme
        const platformRec = document.getElementById('platformRecommendation');
        if(platformRec) platformRec.style.display = 'none';
        
        // Modal pour choisir le format
        const formats = ['csv', 'xlsx', 'json', 'tsv'];
        const formatChoice = prompt(
            'Format de conversion souhaité:\n' + 
            formats.map((f,i) => `${i+1}. ${f.toUpperCase()}`).join('\n') +
            '\n\nEntrez le nom du format:', 'csv'
        );
        
        if(!formatChoice || !formats.includes(formatChoice.toLowerCase())) {
            notify('Format non valide', 'error');
            return;
        }
        
        notify('Conversion en cours...', 'info');
        api('/convert', {target_format: formatChoice.toLowerCase()}, 'Conversion de format', (res) => {
            if(res.success) {
                processedFiles.converted = res.download_url;
                showDownloadOptions('converted');
                notify(`Conversion vers ${formatChoice.toUpperCase()} terminée!`, 'success');
            }
        });
    }
    
    // Fonction pour afficher les options de téléchargement
    function showDownloadOptions(type) {
        const resultsSection = document.getElementById('resultsSection');
        const cleanedBtn = document.getElementById('downloadCleanedBtn');
        const convertedBtn = document.getElementById('downloadConvertedBtn');
        const reportBtn = document.getElementById('downloadReportBtn');
        
        if(resultsSection) {
            resultsSection.style.display = 'block';
        }
        
        if(type === 'cleaned' && cleanedBtn) {
            cleanedBtn.style.display = 'flex';
        }
        if(type === 'converted' && convertedBtn) {
            convertedBtn.style.display = 'flex';
        }
        if(processedFiles.report && reportBtn) {
            reportBtn.style.display = 'flex';
        }
    }
    
    // Fonction de téléchargement
    function downloadFile(url, defaultName) {
        try {
            const link = document.createElement('a');
            link.href = url;
            link.download = defaultName;
            link.target = '_blank';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            notify('Téléchargement initié', 'success');
        } catch(error) {
            notify('Erreur de téléchargement', 'error');
        }
    }

    // Fonction helper pour les appels API simples (conservée pour compatibilité)
    function runSimple(endpoint, okMsg, body={}){
        if(!fileData) return notify('Aucun fichier', 'error');
        api(endpoint, body, okMsg, res => renderResult(res));
    }

    function renderResult(r){
        if(!els.results) return;
        let html='';
        if(r.success){
            html=`<div class="success-box"><h3>${r.message||'Succès'}</h3>`;
            if(r.stats) html+='<ul>'+Object.entries(r.stats).map(([k,v])=>`<li>${k}: ${v}</li>`).join('')+'</ul>';
            if(r.download_url) html+=`<a href="${r.download_url}" target="_blank">Télécharger</a>`;
            if(r.graph_url) html+=`<img src="${r.graph_url}" style="max-width:100%;">`;
            html+='</div>';
        }else html=`<div class="error-box"><h3>Erreur</h3><p>${r.error}</p></div>`;
        els.results.innerHTML=html;
    }
});
