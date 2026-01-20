import os
import csv
import time
import re
from pathlib import Path
from docling.document_converter import DocumentConverter
import fitz  # PyMuPDF pour extraction OCR des formules


def extract_formulas_from_pdf(pdf_path: Path, markdown_content: str) -> dict:
    """
    Extrait les formules du PDF original pour remplacer les <!-- formula-not-decoded -->.
    
    Stratégie:
    1. Identifier les positions des formules non décodées dans le Markdown
    2. Extraire le texte OCR autour de ces positions depuis le PDF
    3. Tenter de reconstruire la formule en LaTeX
    
    Args:
        pdf_path: Chemin du PDF source
        markdown_content: Contenu Markdown avec <!-- formula-not-decoded -->
        
    Returns:
        Dict {line_number: formula_latex}
    """
    
    formulas = {}
    
    # Trouver toutes les positions de <!-- formula-not-decoded -->
    lines = markdown_content.split('\n')
    formula_positions = []
    
    for i, line in enumerate(lines):
        if '<!-- formula-not-decoded -->' in line:
            # Extraire le contexte avant/après
            context_before = ' '.join(lines[max(0, i-3):i]).strip()
            context_after = ' '.join(lines[i+1:min(len(lines), i+4)]).strip()
            
            formula_positions.append({
                'line': i,
                'context_before': context_before,
                'context_after': context_after
            })
    
    if not formula_positions:
        return formulas
    
    # Ouvrir le PDF pour extraction
    try:
        doc = fitz.open(pdf_path)
        
        for pos in formula_positions:
            # Chercher le contexte dans le PDF pour localiser la page
            context_search = pos['context_before'][-50:] if len(pos['context_before']) > 50 else pos['context_before']
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if context_search.lower() in text.lower():
                    # Page trouvée, extraire le texte de la zone probable de la formule
                    # Chercher les patterns mathématiques communs
                    math_patterns = [
                        r'[∑∏∫∂√∞≤≥≠±×÷]',  # Symboles mathématiques
                        r'[α-ωΑ-Ω]',  # Lettres grecques
                        r'\b[a-z]\s*[=≈]\s*[0-9]',  # Équations simples
                        r'[xyz]\^[0-9]',  # Puissances
                        r'[xyz]_[0-9]',  # Indices
                    ]
                    
                    # Extraire les blocs contenant des symboles mathématiques
                    blocks = page.get_text("blocks")
                    for block in blocks:
                        if len(block) >= 5:
                            block_text = block[4]
                            
                            # Vérifier si le bloc contient des symboles mathématiques
                            for pattern in math_patterns:
                                if re.search(pattern, block_text):
                                    # Tentative de conversion en LaTeX basique
                                    latex = convert_to_latex_basic(block_text)
                                    if latex:
                                        formulas[pos['line']] = latex
                                    break
                    break
        
        doc.close()
        
    except Exception as e:
        print(f"  ⚠ Extraction formules: {e}")
    
    return formulas


def convert_to_latex_basic(text: str) -> str:
    """
    Convertit un texte mathématique basique en LaTeX.
    
    Args:
        text: Texte contenant des symboles mathématiques
        
    Returns:
        Formule LaTeX ou None
    """
    
    # Nettoyage
    text = text.strip()
    
    if not text or len(text) > 200:  # Pas une formule si trop long
        return None
    
    # Remplacements de symboles communs
    replacements = {
        '≤': r'\leq',
        '≥': r'\geq',
        '≠': r'\neq',
        '±': r'\pm',
        '×': r'\times',
        '÷': r'\div',
        '∑': r'\sum',
        '∏': r'\prod',
        '∫': r'\int',
        '∂': r'\partial',
        '√': r'\sqrt',
        '∞': r'\infty',
        'α': r'\alpha',
        'β': r'\beta',
        'γ': r'\gamma',
        'δ': r'\delta',
        'ε': r'\epsilon',
        'θ': r'\theta',
        'λ': r'\lambda',
        'μ': r'\mu',
        'π': r'\pi',
        'σ': r'\sigma',
        'φ': r'\phi',
        'ω': r'\omega',
    }
    
    latex = text
    for symbol, latex_cmd in replacements.items():
        latex = latex.replace(symbol, latex_cmd)
    
    # Détecter et convertir les puissances (x^2 -> x^{2})
    latex = re.sub(r'(\w)\^(\d+)', r'\1^{\2}', latex)
    
    # Détecter et convertir les indices (x_i -> x_{i})
    latex = re.sub(r'(\w)_(\w+)', r'\1_{\2}', latex)
    
    # Envelopper dans des délimiteurs LaTeX si pas déjà fait
    if not latex.startswith('$'):
        latex = f"${latex}$"
    
    return latex


def fix_formulas_in_markdown(markdown_content: str, pdf_path: Path, aggressive: bool = True) -> str:
    """
    Post-traitement du Markdown pour corriger les formules non décodées.
    
    Stratégies:
    1. Extraction depuis le PDF original (OCR)
    2. Remplacement par placeholder informatif si extraction impossible
    3. Mode agressif: tente de reconstruire depuis le contexte
    
    Args:
        markdown_content: Contenu Markdown brut
        pdf_path: Chemin du PDF source
        aggressive: Si True, tente reconstruction agressive
        
    Returns:
        Markdown avec formules corrigées
    """
    
    # Compter les formules non décodées
    formula_count = markdown_content.count('<!-- formula-not-decoded -->')
    
    if formula_count == 0:
        return markdown_content
    
    print(f"  ⚠ {formula_count} formules non décodées détectées")
    print(f"  🔧 Tentative de récupération...")
    
    # Stratégie 1: Extraction depuis PDF
    formulas = extract_formulas_from_pdf(pdf_path, markdown_content)
    
    if formulas:
        print(f"  ✓ {len(formulas)} formules récupérées depuis le PDF")
        
        # Remplacer dans le Markdown
        lines = markdown_content.split('\n')
        for line_num, latex in formulas.items():
            if line_num < len(lines):
                lines[line_num] = lines[line_num].replace(
                    '<!-- formula-not-decoded -->',
                    latex
                )
        
        markdown_content = '\n'.join(lines)
    
    # Stratégie 2: Remplacement par placeholder informatif
    remaining_count = markdown_content.count('<!-- formula-not-decoded -->')
    
    if remaining_count > 0:
        if aggressive:
            # Mode agressif: marquer pour révision manuelle
            markdown_content = markdown_content.replace(
                '<!-- formula-not-decoded -->',
                '`[FORMULE_À_RÉVISER]`'
            )
            print(f"  ℹ {remaining_count} formules marquées pour révision manuelle")
        else:
            print(f"  ⚠ {remaining_count} formules restent non décodées")
    
    return markdown_content


def granite(
    input_path: str,
    output_format: str,
    preserve_formulas: bool = False,
    fix_formulas: bool = True,
):
    """
    Convertit un document (PDF, DOCX, etc.) en Markdown/Text/JSON avec Granite (Docling).

    Note: conversion best-effort. Actuellement, la préservation des formules et des images
    n'est pas garantie (fonctionnalité en développement). Les options associées sont donc
    désactivées par défaut.
    
    Args:
        input_path: Chemin du fichier source
        output_format: 'markdown' | 'text' | 'json'
        preserve_formulas: Active l'enrichissement des formules dans Docling (expérimental)
        fix_formulas: Post-traitement pour corriger les <!-- formula-not-decoded -->
        
    Returns:
        Chemin du fichier de sortie
    """

    t0 = time.time()

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_format = output_format.lower()

    suffix_map = {
        "markdown": ".md",
        "text": ".txt",
        "json": ".json",
    }
    if output_format not in suffix_map:
        raise ValueError(f"Unsupported output_format: {output_format}")

    # Organiser les sorties dans result/ avec sous-dossiers par type
    result_base = input_path.parent / "result"
    
    if output_format == "markdown":
        output_dir = result_base / "markdown"
    elif output_format == "text":
        output_dir = result_base / "texte"
    else:  # json
        output_dir = result_base / "json"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (input_path.stem + suffix_map[output_format])
    
    # Créer le dossier csv pour les logs
    csv_dir = result_base / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    devlog_csv_path = csv_dir / "granite_devlog.csv"

    # Configuration pour préserver les formules mathématiques
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import PdfFormatOption
    
    pipeline_options = PdfPipelineOptions()
    # OCR activé par défaut pour améliorer la robustesse d'extraction sur PDF.
    pipeline_options.do_ocr = True
    # Important pour PDF→table (ex: PDF avec tableaux) : structure des tableaux.
    # Certaines versions de Docling n'exposent pas forcément ce flag; on le fait en best-effort.
    if hasattr(pipeline_options, "do_table_structure"):
        pipeline_options.do_table_structure = True
    pipeline_options.generate_page_images = False
    # NOTE: extraction d'images non garantie pour le moment (désactivée par défaut)
    pipeline_options.generate_picture_images = False

    if preserve_formulas:
        # Expérimental: enrichissement des formules
        pipeline_options.do_formula_enrichment = True
    
    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
    
    converter = DocumentConverter(format_options=format_options)
    result = converter.convert(str(input_path))

    if output_format == "markdown":
        content = result.document.export_to_markdown()
    elif output_format == "text":
        content = result.document.export_to_text()
    else:  # json
        content = result.document.model_dump_json(indent=2)

    # POST-TRAITEMENT: Corriger les formules non décodées (uniquement pour Markdown)
    if output_format == "markdown" and preserve_formulas and fix_formulas:
        original_count = content.count('<!-- formula-not-decoded -->')
        if original_count > 0:
            content = fix_formulas_in_markdown(content, input_path, aggressive=True)
            fixed_count = original_count - content.count('<!-- formula-not-decoded -->')
            if fixed_count > 0:
                print(f"  ✓ {fixed_count}/{original_count} formules récupérées")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    duration = round(time.time() - t0, 3)

    write_header = not devlog_csv_path.exists()
    with open(devlog_csv_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "timestamp",
                "input_file",
                "input_extension",
                "output_file",
                "output_format",
                "file_size_bytes",
                "processing_time_seconds",
            ])
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            str(input_path),
            input_path.suffix,
            str(output_path),
            output_format,
            input_path.stat().st_size,
            duration,
        ])

    return output_path
