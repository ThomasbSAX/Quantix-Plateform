/**
 * Quantix Advanced Converter - Interface JavaScript
 * Gestion avancée des conversions de fichiers
 */

// ===== INTERFACE DE CONVERSION AVANCÉE =====

class QuantixConverter {
    constructor() {
        this.supportedFormats = {
            data: ['csv', 'xlsx', 'json', 'parquet', 'tsv', 'xml', 'html'],
            documents: ['txt', 'md', 'markdown', 'rtf', 'docx', 'html'],
            pdf: ['pdf'],
            images: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']
        };
        
        this.conversionQueue = [];
        this.isConverting = false;
    }

    async getConversionInfo(filename) {
        try {
            const response = await fetch(`/get_conversion_info/${encodeURIComponent(filename)}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const data = await response.json();
            
            if (data.success) {
                return data;
            } else {
                throw new Error(data.error || 'Erreur lors de la récupération des informations');
            }
        } catch (error) {
            console.error('Erreur getConversionInfo:', error);
            throw error;
        }
    }

    async convertFile(filename, targetFormat, options = {}) {
        try {
            showLoading(`Conversion vers ${targetFormat.toUpperCase()}...`);

            const response = await fetch('/convert_file', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    filename: filename,
                    format: targetFormat,
                    options: options
                })
            });

            const data = await response.json();
            hideLoading();

            if (data.success) {
                this.displayConversionSuccess(data);
                return data;
            } else {
                throw new Error(data.error || 'Erreur lors de la conversion');
            }
        } catch (error) {
            hideLoading();
            displayQuantixError(`Erreur de conversion: ${error.message}`);
            throw error;
        }
    }

    async convertBatch(filenames, targetFormat, options = {}) {
        try {
            showLoading(`Conversion en lot vers ${targetFormat.toUpperCase()}...`);

            const response = await fetch('/convert_batch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    filenames: filenames,
                    format: targetFormat,
                    options: options
                })
            });

            const data = await response.json();
            hideLoading();

            if (data.success) {
                this.displayBatchConversionResults(data);
                return data;
            } else {
                throw new Error(data.error || 'Erreur lors de la conversion en lot');
            }
        } catch (error) {
            hideLoading();
            displayQuantixError(`Erreur de conversion en lot: ${error.message}`);
            throw error;
        }
    }

    displayConversionSuccess(result) {
        const successHtml = `
            <div class="conversion-success">
                <h4>Conversion réussie</h4>
                
                <div class="conversion-details">
                    <div class="detail-item">
                        <strong>Fichier source:</strong> ${result.original_file}
                    </div>
                    <div class="detail-item">
                        <strong>Fichier converti:</strong> ${result.converted_file}
                    </div>
                    <div class="detail-item">
                        <strong>Format source:</strong> ${result.source_format}
                    </div>
                    <div class="detail-item">
                        <strong>Format cible:</strong> ${result.target_format}
                    </div>
                    <div class="detail-item">
                        <strong>Type de contenu:</strong> ${result.source_type}
                    </div>
                    <div class="detail-item">
                        <strong>Taille:</strong> ${result.file_size_mb} MB
                    </div>
                </div>

                <div class="conversion-actions">
                    <a href="${result.download_url}" class="btn btn-primary" download="${result.converted_file}">
                        <i class="fas fa-download"></i> Télécharger
                    </a>
                    <button class="btn btn-secondary" onclick="showConversionMetadata('${JSON.stringify(result.metadata).replace(/'/g, "\\'")}')">
                        <i class="fas fa-info-circle"></i> Détails
                    </button>
                </div>
            </div>
        `;

        showModal('Conversion Réussie', successHtml);
    }

    displayBatchConversionResults(results) {
        const { converted_files, errors, total_files, successful_conversions, failed_conversions } = results;

        let resultsHtml = `
            <div class="batch-conversion-results">
                <h4>📦 Conversion en lot terminée</h4>
                
                <div class="batch-summary">
                    <div class="summary-stat">
                        <span class="stat-value">${total_files}</span>
                        <span class="stat-label">Fichiers total</span>
                    </div>
                    <div class="summary-stat success">
                        <span class="stat-value">${successful_conversions}</span>
                        <span class="stat-label">Réussies</span>
                    </div>
                    <div class="summary-stat error">
                        <span class="stat-value">${failed_conversions}</span>
                        <span class="stat-label">Échecs</span>
                    </div>
                </div>

                ${successful_conversions > 0 ? `
                    <div class="successful-conversions">
                        <h5>Conversions réussies :</h5>
                        <div class="conversions-list">
                            ${converted_files.map(file => `
                                <div class="conversion-item">
                                    <div class="file-info">
                                        <span class="original-file">${file.original}</span>
                                        <i class="fas fa-arrow-right"></i>
                                        <span class="converted-file">${file.converted}</span>
                                    </div>
                                    <a href="${file.download_url}" class="btn btn-sm btn-primary" download="${file.converted}">
                                        <i class="fas fa-download"></i>
                                    </a>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}

                ${failed_conversions > 0 ? `
                    <div class="failed-conversions">
                        <h5>Échecs de conversion :</h5>
                        <ul class="error-list">
                            ${errors.map(error => `<li>${error}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}

                <div class="batch-actions">
                    ${successful_conversions > 0 ? `
                        <button class="btn btn-success" onclick="downloadAllConversions('${JSON.stringify(converted_files).replace(/'/g, "\\'")}')">
                            <i class="fas fa-download"></i> Tout télécharger
                        </button>
                    ` : ''}
                    <button class="btn btn-secondary" onclick="closeModal()">
                        Fermer
                    </button>
                </div>
            </div>
        `;

        showModal('Résultats de Conversion', resultsHtml);
    }

    async showConversionModal(filename) {
        try {
            const info = await this.getConversionInfo(filename);
            const { file_info, conversion_categories, recommended_formats } = info;

            const modalHtml = `
                <div class="conversion-modal">
                    <div class="file-summary">
                        <h4>📄 ${file_info.metadata.name}</h4>
                        <div class="file-details">
                            <span class="file-type">${file_info.source_type}</span>
                            <span class="file-format">${file_info.source_format}</span>
                            <span class="file-size">${file_info.metadata.size_mb} MB</span>
                        </div>
                    </div>

                    <div class="conversion-options">
                        <div class="format-categories">
                            ${Object.entries(conversion_categories).map(([category, formats]) => {
                                if (formats.length === 0) return '';
                                
                                return `
                                    <div class="format-category">
                                        <h5>${this.getCategoryIcon(category)} ${this.getCategoryName(category)}</h5>
                                        <div class="format-buttons">
                                            ${formats.map(format => `
                                                <button class="btn format-btn ${this.isRecommended(format, recommended_formats) ? 'btn-primary' : 'btn-outline-secondary'}" 
                                                        onclick="quantixConverter.convertFile('${filename}', '${format.replace('.', '')}')">
                                                    ${format.replace('.', '').toUpperCase()}
                                                    ${this.isRecommended(format, recommended_formats) ? ' ⭐' : ''}
                                                </button>
                                            `).join('')}
                                        </div>
                                    </div>
                                `;
                            }).join('')}
                        </div>

                        <div class="advanced-options">
                            <h5>⚙️ Options avancées</h5>
                            <div class="options-form" id="conversionOptionsForm">
                                ${this.generateAdvancedOptions(file_info)}
                            </div>
                        </div>
                    </div>
                </div>
            `;

            showModal(`Convertir ${filename}`, modalHtml);
        } catch (error) {
            displayQuantixError(`Erreur: ${error.message}`);
        }
    }

    getCategoryIcon(category) {
        const icons = {
            data: '📊',
            documents: '📝',
            pdf: '📕',
            images: '🖼️'
        };
        return icons[category] || '📄';
    }

    getCategoryName(category) {
        const names = {
            data: 'Données',
            documents: 'Documents',
            pdf: 'PDF',
            images: 'Images'
        };
        return names[category] || category;
    }

    isRecommended(format, recommendedFormats) {
        return Object.values(recommendedFormats).some(category => 
            category.includes(format.replace('.', ''))
        );
    }

    generateAdvancedOptions(fileInfo) {
        let optionsHtml = '';

        // Options pour les images
        if (fileInfo.has_image) {
            optionsHtml += `
                <div class="option-group">
                    <label>Qualité (JPEG):</label>
                    <input type="range" id="imageQuality" min="10" max="100" value="90" class="form-range">
                    <span id="qualityValue">90</span>%
                </div>
                <div class="option-group">
                    <label>Redimensionner:</label>
                    <div class="size-inputs">
                        <input type="number" id="resizeWidth" placeholder="Largeur" min="1" class="form-control">
                        <span>×</span>
                        <input type="number" id="resizeHeight" placeholder="Hauteur" min="1" class="form-control">
                    </div>
                </div>
            `;
        }

        // Options pour les données
        if (fileInfo.has_dataframe) {
            optionsHtml += `
                <div class="option-group">
                    <label>Orientation JSON:</label>
                    <select id="jsonOrient" class="form-select">
                        <option value="records">Records</option>
                        <option value="index">Index</option>
                        <option value="values">Values</option>
                        <option value="table">Table</option>
                    </select>
                </div>
            `;
        }

        // Options générales
        optionsHtml += `
            <div class="option-group">
                <div class="form-check">
                    <input type="checkbox" id="optimizeOutput" class="form-check-input" checked>
                    <label for="optimizeOutput" class="form-check-label">Optimiser la sortie</label>
                </div>
            </div>
        `;

        return optionsHtml;
    }
}

// Instance globale
const quantixConverter = new QuantixConverter();

// ===== FONCTIONS UTILITAIRES =====

function showConversionMetadata(metadataJson) {
    const metadata = JSON.parse(metadataJson);
    
    const metadataHtml = `
        <div class="conversion-metadata">
            <h4>📋 Métadonnées de Conversion</h4>
            
            <div class="metadata-grid">
                <div class="metadata-item">
                    <strong>Taille originale:</strong>
                    <span>${formatBytes(metadata.original_size)}</span>
                </div>
                <div class="metadata-item">
                    <strong>Heure de conversion:</strong>
                    <span>${new Date(metadata.conversion_time).toLocaleString()}</span>
                </div>
                <div class="metadata-item">
                    <strong>Contient des données:</strong>
                    <span>${metadata.has_dataframe ? 'OK' : 'Non'}</span>
                </div>
                <div class="metadata-item">
                    <strong>Contient du texte:</strong>
                    <span>${metadata.has_text ? 'OK' : 'Non'}</span>
                </div>
                <div class="metadata-item">
                    <strong>Contient des images:</strong>
                    <span>${metadata.has_image ? 'OK' : 'Non'}</span>
                </div>
            </div>
        </div>
    `;
    
    showModal('Métadonnées de Conversion', metadataHtml);
}

function downloadAllConversions(conversionsJson) {
    const conversions = JSON.parse(conversionsJson);
    
    conversions.forEach((conversion, index) => {
        setTimeout(() => {
            const link = document.createElement('a');
            link.href = conversion.download_url;
            link.download = conversion.converted;
            link.click();
        }, index * 500); // Délai de 500ms entre chaque téléchargement
    });
    
    showSuccessMessage(`Téléchargement de ${conversions.length} fichier(s) lancé`);
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// ===== INTÉGRATION AVEC L'INTERFACE EXISTANTE =====

// Ajouter un bouton de conversion dans les interfaces existantes
function addConversionButton(filename, container) {
    const button = document.createElement('button');
    button.className = 'btn btn-info btn-sm';
    button.innerHTML = '<i class="fas fa-exchange-alt"></i> Convertir';
    button.onclick = () => quantixConverter.showConversionModal(filename);
    
    if (container) {
        container.appendChild(button);
    }
    
    return button;
}

// Mise à jour du slider de qualité en temps réel
document.addEventListener('DOMContentLoaded', function() {
    document.addEventListener('input', function(e) {
        if (e.target.id === 'imageQuality') {
            const valueSpan = document.getElementById('qualityValue');
            if (valueSpan) {
                valueSpan.textContent = e.target.value;
            }
        }
    });
});

// CSS pour les nouvelles interfaces
const conversionStyles = `
<style>
.conversion-modal .file-summary {
    text-align: center;
    margin-bottom: 1.5rem;
    padding: 1rem;
    background: #f8f9fa;
    border-radius: 0.375rem;
}

.conversion-modal .file-details {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 0.5rem;
}

.conversion-modal .file-details span {
    padding: 0.25rem 0.5rem;
    background: #e9ecef;
    border-radius: 0.25rem;
    font-size: 0.875rem;
}

.format-category {
    margin-bottom: 1.5rem;
}

.format-category h5 {
    margin-bottom: 0.75rem;
    color: #495057;
}

.format-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.format-btn {
    min-width: 80px;
    font-weight: 600;
}

.advanced-options {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #dee2e6;
}

.option-group {
    margin-bottom: 1rem;
}

.option-group label {
    display: block;
    margin-bottom: 0.25rem;
    font-weight: 500;
}

.size-inputs {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.size-inputs input {
    width: 100px;
}

.batch-conversion-results .batch-summary {
    display: flex;
    justify-content: space-around;
    margin: 1rem 0;
    text-align: center;
}

.batch-conversion-results .summary-stat {
    padding: 0.75rem;
    border-radius: 0.375rem;
    background: #f8f9fa;
}

.batch-conversion-results .summary-stat.success {
    background: #d1edff;
    color: #0f5132;
}

.batch-conversion-results .summary-stat.error {
    background: #f8d7da;
    color: #842029;
}

.batch-conversion-results .stat-value {
    display: block;
    font-size: 1.5rem;
    font-weight: bold;
}

.batch-conversion-results .stat-label {
    display: block;
    font-size: 0.875rem;
}

.conversions-list .conversion-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem;
    margin: 0.5rem 0;
    background: #f8f9fa;
    border-radius: 0.25rem;
}

.conversions-list .file-info {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.conversions-list .original-file {
    font-weight: 500;
}

.conversions-list .converted-file {
    color: #28a745;
    font-weight: 500;
}

.conversion-success .conversion-details {
    margin: 1rem 0;
}

.conversion-success .detail-item {
    display: flex;
    justify-content: space-between;
    padding: 0.25rem 0;
    border-bottom: 1px solid #eee;
}

.conversion-actions {
    margin-top: 1.5rem;
    text-align: center;
}

.conversion-actions .btn {
    margin: 0 0.5rem;
}

.metadata-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin: 1rem 0;
}

.metadata-item {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem;
    background: #f8f9fa;
    border-radius: 0.25rem;
}
</style>
`;

// Injecter les styles
document.head.insertAdjacentHTML('beforeend', conversionStyles);