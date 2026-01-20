/**
 * Quantix Error Handler - Fonctions JavaScript pour la gestion des erreurs
 * et le tracking des opérations côté client
 */

// ===== GESTION DES ERREURS CÔTÉ CLIENT =====

function displayQuantixError(error, containerId = 'error-container') {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error('Container d\'erreur introuvable:', containerId);
        return;
    }
    
    const errorHtml = `
        <div class="quantix-error alert alert-danger" role="alert">
            <div class="error-icon">
                <i class="fas fa-exclamation-triangle"></i>
            </div>
            <div class="error-content">
                <h5 class="error-title">Erreur Quantix</h5>
                <div class="error-message">${error}</div>
            </div>
            <button type="button" class="btn-close" onclick="clearQuantixError('${containerId}')">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    container.innerHTML = errorHtml;
    container.style.display = 'block';
    
    // Auto-hide après 10 secondes
    setTimeout(() => {
        clearQuantixError(containerId);
    }, 10000);
}

function clearQuantixError(containerId = 'error-container') {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = '';
        container.style.display = 'none';
    }
}

function handleApiError(response, operation = 'opération') {
    if (response.error) {
        displayQuantixError(response.error);
    } else {
        displayQuantixError(`Erreur lors de ${operation}. Veuillez réessayer.`);
    }
}

// ===== RAPPORTS ET STATISTIQUES =====

async function getSessionReport() {
    try {
        const response = await fetch('/get_session_report', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            displaySessionReport(data);
        } else {
            handleApiError(data, 'la récupération du rapport de session');
        }
    } catch (error) {
        displayQuantixError('Erreur de connexion lors de la récupération du rapport.');
    }
}

function displaySessionReport(data) {
    const { session_summary, operations_report } = data;
    
    const reportHtml = `
        <div class="session-report">
            <h3>Rapport de Session Quantix</h3>
            
            <div class="session-stats">
                <div class="stat-item">
                    <span class="stat-label">Fichiers traités:</span>
                    <span class="stat-value">${session_summary.files_processed}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Opérations totales:</span>
                    <span class="stat-value">${session_summary.total_operations}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Durée:</span>
                    <span class="stat-value">${session_summary.duration_seconds.toFixed(1)}s</span>
                </div>
            </div>
            
            <div class="operations-breakdown">
                <h4>Répartition des opérations:</h4>
                <ul>
                    ${Object.entries(session_summary.operation_types).map(([type, count]) => 
                        `<li>${type}: ${count}</li>`
                    ).join('')}
                </ul>
            </div>
            
            <div class="detailed-report">
                <h4>Rapport détaillé:</h4>
                <pre>${operations_report}</pre>
            </div>
        </div>
    `;
    
    // Afficher dans une modale ou un conteneur dédié
    showModal('Rapport de Session', reportHtml);
}

async function exportFullReport() {
    try {
        showLoading('Génération du rapport complet...');
        
        const response = await fetch('/export_full_report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        hideLoading();
        
        if (data.success) {
            showSuccessMessage(`Rapport généré avec succès: ${data.report_filename}`);
            
            // Proposer le téléchargement
            const downloadLink = document.createElement('a');
            downloadLink.href = data.download_url;
            downloadLink.download = data.report_filename;
            downloadLink.click();
        } else {
            handleApiError(data, 'l\'export du rapport');
        }
    } catch (error) {
        hideLoading();
        displayQuantixError('Erreur lors de l\'export du rapport.');
    }
}

async function getErrorStatistics() {
    try {
        const response = await fetch('/get_error_statistics', {
            method: 'GET'
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayErrorStatistics(data.error_statistics);
        } else {
            handleApiError(data, 'la récupération des statistiques d\'erreur');
        }
    } catch (error) {
        displayQuantixError('Erreur lors de la récupération des statistiques.');
    }
}

function displayErrorStatistics(stats) {
    if (stats.total === 0) {
        showSuccessMessage('Aucune erreur enregistrée dans cette session ! ');
        return;
    }
    
    const statsHtml = `
        <div class="error-statistics">
            <h3>Statistiques des Erreurs</h3>
            
            <div class="total-errors">
                <strong>Total: ${stats.total} erreur(s)</strong>
            </div>
            
            <div class="error-breakdown">
                <h4>Par type:</h4>
                <ul>
                    ${Object.entries(stats.by_type).map(([type, count]) => 
                        `<li>${type}: ${count}</li>`
                    ).join('')}
                </ul>
            </div>
            
            <div class="severity-breakdown">
                <h4>Par gravité:</h4>
                <ul>
                    ${Object.entries(stats.by_severity).map(([severity, count]) => 
                        `<li class="severity-${severity}">${severity}: ${count}</li>`
                    ).join('')}
                </ul>
            </div>
            
            ${stats.most_frequent ? `
                <div class="most-frequent">
                    <strong>Erreur la plus fréquente:</strong> ${stats.most_frequent}
                </div>
            ` : ''}
        </div>
    `;
    
    showModal('Statistiques des Erreurs', statsHtml);
}

async function clearSession() {
    if (!confirm('Êtes-vous sûr de vouloir effacer toute la session ? Toutes les données et opérations seront perdues.')) {
        return;
    }
    
    try {
        const response = await fetch('/clear_session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            showSuccessMessage('Session effacée avec succès !');
            
            // Recharger la page après un délai
            setTimeout(() => {
                window.location.reload();
            }, 2000);
        } else {
            handleApiError(data, 'l\'effacement de la session');
        }
    } catch (error) {
        displayQuantixError('Erreur lors de l\'effacement de la session.');
    }
}

// ===== FONCTIONS UTILITAIRES =====

function showLoading(message = 'Chargement...') {
    const loadingHtml = `
        <div id="quantix-loading" class="quantix-loading">
            <div class="loading-content">
                <div class="spinner"></div>
                <div class="loading-message">${message}</div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', loadingHtml);
}

function hideLoading() {
    const loading = document.getElementById('quantix-loading');
    if (loading) {
        loading.remove();
    }
}

function showSuccessMessage(message) {
    const successHtml = `
        <div class="quantix-success alert alert-success" role="alert">
            <div class="success-icon">
                <i class="fas fa-check-circle"></i>
            </div>
            <div class="success-message">${message}</div>
        </div>
    `;
    
    const container = document.getElementById('success-container') || document.getElementById('error-container');
    if (container) {
        container.innerHTML = successHtml;
        container.style.display = 'block';
        
        setTimeout(() => {
            container.innerHTML = '';
            container.style.display = 'none';
        }, 5000);
    }
}

function showModal(title, content) {
    const modalHtml = `
        <div id="quantix-modal" class="quantix-modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>${title}</h3>
                    <button type="button" class="modal-close" onclick="closeModal()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" onclick="closeModal()">Fermer</button>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHtml);
}

function closeModal() {
    const modal = document.getElementById('quantix-modal');
    if (modal) {
        modal.remove();
    }
}

// ===== INTÉGRATION AVEC L'API EXISTANTE =====

// Wrapper pour les appels API existants avec gestion d'erreurs améliorée
function enhancedApiCall(originalFunction) {
    return async function(...args) {
        try {
            const result = await originalFunction.apply(this, args);
            return result;
        } catch (error) {
            if (error.response && error.response.error) {
                displayQuantixError(error.response.error);
            } else {
                displayQuantixError('Une erreur inattendue s\'est produite.');
            }
            throw error;
        }
    };
}

// ===== CSS POUR LES NOUVEAUX ÉLÉMENTS =====
const quantixErrorStyles = `
<style>
.quantix-error {
    display: flex;
    align-items: flex-start;
    padding: 1rem;
    margin: 1rem 0;
    border-left: 4px solid #dc3545;
    background: #f8d7da;
    border-radius: 0.375rem;
    color: #721c24;
}

.quantix-error .error-icon {
    margin-right: 0.75rem;
    color: #dc3545;
}

.quantix-error .error-content {
    flex-grow: 1;
}

.quantix-error .error-title {
    margin: 0 0 0.5rem 0;
    font-weight: 600;
}

.quantix-error .error-message {
    margin: 0;
    white-space: pre-line;
}

.quantix-success {
    display: flex;
    align-items: center;
    padding: 1rem;
    margin: 1rem 0;
    border-left: 4px solid #28a745;
    background: #d4edda;
    border-radius: 0.375rem;
    color: #155724;
}

.quantix-success .success-icon {
    margin-right: 0.75rem;
    color: #28a745;
}

.quantix-loading {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
}

.quantix-loading .loading-content {
    background: white;
    padding: 2rem;
    border-radius: 0.5rem;
    text-align: center;
}

.quantix-loading .spinner {
    width: 40px;
    height: 40px;
    border: 4px solid #f3f3f3;
    border-top: 4px solid #007bff;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 1rem;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.quantix-modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 10000;
}

.quantix-modal .modal-content {
    background: white;
    max-width: 90%;
    max-height: 90%;
    overflow-y: auto;
    border-radius: 0.5rem;
}

.quantix-modal .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    border-bottom: 1px solid #dee2e6;
}

.quantix-modal .modal-body {
    padding: 1rem;
}

.quantix-modal .modal-footer {
    padding: 1rem;
    border-top: 1px solid #dee2e6;
    text-align: right;
}

.session-report .session-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.session-report .stat-item {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem;
    background: #f8f9fa;
    border-radius: 0.25rem;
}

.session-report .stat-label {
    font-weight: 600;
}

.session-report .detailed-report pre {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 0.25rem;
    overflow-x: auto;
    font-size: 0.875rem;
}

.error-statistics .severity-critical { color: #dc3545; }
.error-statistics .severity-high { color: #fd7e14; }
.error-statistics .severity-medium { color: #ffc107; }
.error-statistics .severity-low { color: #28a745; }
</style>
`;

// Injecter les styles
document.head.insertAdjacentHTML('beforeend', quantixErrorStyles);