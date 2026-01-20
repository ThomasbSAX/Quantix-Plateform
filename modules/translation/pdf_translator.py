"""Traducteur pour les fichiers PDF.

Note: ce module est importé par Flask; on évite donc les dépendances lourdes
au niveau module. Les imports (PyMuPDF/langdetect) se font à l'appel.
"""

import time

from .text_translation import translate_text
from .math_detection import is_mathematical_content


def translate_pdf(input_path, output_path, translator, pages_to_translate=None, preserve_foreign_quotes=True, progress_callback=None):
    """
    Traduit un PDF en préservant TOUT : gras, couleurs, structure, graphiques, formules.
    Approche algorithmique étape par étape pour éviter les chevauchements.
    
    Args:
        input_path: Chemin du PDF source
        output_path: Chemin du PDF traduit
        translator: Objet traducteur
        pages_to_translate: Liste des numéros de pages à traduire (commence à 0) ou None pour tout
        preserve_foreign_quotes: Si True, ne traduit pas les courtes citations en langue étrangère
        progress_callback: Fonction appelée avec (page_actuelle, total_pages) pour suivre la progression
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("Installation de PyMuPDF...")
        import subprocess
        subprocess.run(["pip", "install", "PyMuPDF"], check=True)
        import fitz
    
    doc = fitz.open(input_path)
    
    # ÉTAPE PRÉLIMINAIRE: Détection de la langue dominante du document
    document_lang = None
    if preserve_foreign_quotes:
        # Import lazy pour éviter de dépendre de langdetect à l'import Flask
        try:
            from .language_detection import detect_document_language
        except Exception:
            detect_document_language = None

        all_text = ""
        # Échantillonne quelques pages pour détecter la langue
        sample_pages = [0, len(doc) // 2, len(doc) - 1] if len(doc) > 2 else range(len(doc))
        for sample_page in sample_pages:
            if sample_page < len(doc):
                all_text += doc[sample_page].get_text()
        if detect_document_language:
            document_lang = detect_document_language(all_text)
        print(f"Langue détectée du document: {document_lang}")
    
    # Affichage du nombre de pages
    total_pages = len(doc)
    print(f"Document: {total_pages} page{'s' if total_pages > 1 else ''}")
    
    # Si aucune page spécifiée, traduire toutes les pages
    if pages_to_translate is None:
        pages_to_translate = list(range(total_pages))
    
    nb_pages = len(pages_to_translate)
    print(f"Pages à traduire: {nb_pages} page{'s' if nb_pages > 1 else ''}")
    
    # Timing / estimation
    start_time = time.time()
    pages_done = 0
    cumulative_seconds = 0.0

    for page_num in range(len(doc)):
        # Vérifier si cette page doit être traduite
        if page_num not in pages_to_translate:
            print(f"Page {page_num + 1}/{total_pages} - ignorée")
            continue

        # Calculer estimation actuelle
        elapsed_seconds = time.time() - start_time
        avg_per_page = (cumulative_seconds / pages_done) if pages_done > 0 else None

        # Appel du callback de progression avant traitement (permet l'arrêt)
        should_continue = True
        if progress_callback:
            try:
                should_continue = progress_callback(page_num + 1, total_pages, elapsed_seconds, avg_per_page)
                if should_continue is None:
                    should_continue = True
            except Exception as e:
                print(f"Erreur callback progression: {e}")

        if not should_continue:
            print(f"Arrêt demandé avant page {page_num + 1}. Sauvegarde du fichier partiel...")
            doc.save(output_path, garbage=4, deflate=True, clean=True)
            doc.close()
            return

        print(f"Traduction page {page_num + 1}/{total_pages}...")
        page = doc[page_num]
        
        # ÉTAPE 1: Sauvegarde de TOUTES les annotations (surlignements, notes, formes)
        annotations = []
        for annot in page.annots():
            annotations.append({
                'type': annot.type[0],
                'rect': annot.rect,
                'colors': annot.colors,
                'opacity': annot.opacity if hasattr(annot, 'opacity') else 1.0,
            })
        
        # ÉTAPE 2: Sauvegarde de TOUS les graphiques et images
        images = []
        for img_index, img in enumerate(page.get_images(full=True)):
            images.append({
                'xref': img[0],
                'info': img
            })
        
        # ÉTAPE 3: Extraction du texte avec positions exactes et styles
        text_instances = page.get_text("dict")["blocks"]
        texts_to_replace = []
        
        DELIM = '<<LINE_BREAK_@@>>'
        for block in text_instances:
            if block["type"] == 0:  # Bloc de texte
                # Collecter toutes les lignes du bloc
                block_lines = []
                for line in block["lines"]:
                    line_text = ""
                    line_spans = []
                    for span in line["spans"]:
                        line_text += span["text"]
                        line_spans.append(span)
                    block_lines.append({
                        'text': line_text,
                        'spans': line_spans
                    })

                # Préparer les segments à traduire en batch pour réduire les appels réseau
                to_translate = []
                translate_indices = []
                for idx, ln in enumerate(block_lines):
                    if not ln['text'].strip():
                        to_translate.append(None)
                        continue
                    if is_mathematical_content(ln['text']):
                        to_translate.append(None)
                    else:
                        to_translate.append(ln['text'])
                        translate_indices.append(idx)

                # Traduction en batch par groupe contigu
                translated_cache = {}
                i = 0
                while i < len(to_translate):
                    if to_translate[i] is None:
                        i += 1
                        continue
                    # regrouper contiguous non-None
                    j = i
                    parts = []
                    idxs = []
                    while j < len(to_translate) and to_translate[j] is not None:
                        parts.append(to_translate[j])
                        idxs.append(j)
                        j += 1
                    # Batch join with delimiter
                    joined = DELIM.join(parts)
                    try:
                        translated_joined = translate_text(
                            joined,
                            translator,
                            document_lang=document_lang,
                            preserve_foreign_quotes=preserve_foreign_quotes
                        )
                        translated_parts = translated_joined.split(DELIM)
                    except Exception as e:
                        # Fallback: translate one by one
                        translated_parts = []
                        for p in parts:
                            try:
                                translated_parts.append(translate_text(p, translator, document_lang=document_lang, preserve_foreign_quotes=preserve_foreign_quotes))
                            except:
                                translated_parts.append(p)

                    # Store translations
                    for k, idx in enumerate(idxs):
                        translated_cache[idx] = translated_parts[k] if k < len(translated_parts) else parts[k]

                    i = j

                # Construire texts_to_replace et appliquer redactions
                for idx, ln in enumerate(block_lines):
                    line_text = ln['text']
                    line_spans = ln['spans']
                    if not line_text.strip():
                        # nothing to do
                        for span in line_spans:
                            page.add_redact_annot(fitz.Rect(span["bbox"]), fill=(1, 1, 1))
                        continue

                    if idx in translated_cache:
                        translated_text = translated_cache[idx]
                    else:
                        # mathematical or left untranslated
                        translated_text = line_text

                    for span in line_spans:
                        font_size = span["size"]
                        font_flags = span.get("flags", 0)

                        is_bold = font_flags & 16
                        is_italic = font_flags & 2

                        text_color = span.get("color", 0)
                        color_rgb = (
                            (text_color & 0xFF) / 255.0,
                            ((text_color >> 8) & 0xFF) / 255.0,
                            ((text_color >> 16) & 0xFF) / 255.0
                        )

                        if is_bold and is_italic:
                            font_to_use = "helv-boldoblique"
                        elif is_bold:
                            font_to_use = "helv-bold"
                        elif is_italic:
                            font_to_use = "helv-oblique"
                        else:
                            font_to_use = "helv"

                        texts_to_replace.append({
                            'bbox': span["bbox"],
                            'text': span["text"],
                            'translated': translated_text,
                            'point': fitz.Point(span["bbox"][0], span["bbox"][3]),
                            'fontsize': font_size,
                            'fontname': font_to_use,
                            'color': color_rgb,
                            'is_first': span == line_spans[0]
                        })

                        page.add_redact_annot(fitz.Rect(span["bbox"]), fill=(1, 1, 1))
        
        # ÉTAPE 7: Application de toutes les suppressions
        page.apply_redactions()
        
        # ÉTAPE 8: Réinsertion du texte traduit avec positions exactes
        for text_info in texts_to_replace:
            if text_info['is_first']:  # Insérer le texte complet seulement pour le premier span
                try:
                    page.insert_text(
                        text_info['point'],
                        text_info['translated'],
                        fontsize=text_info['fontsize'],
                        fontname=text_info['fontname'],
                        color=text_info['color']
                    )
                except Exception as e:
                    # Fallback sans fontname spécifique
                    try:
                        page.insert_text(
                            text_info['point'],
                            text_info['translated'],
                            fontsize=text_info['fontsize'],
                            color=text_info['color']
                        )
                    except:
                        pass  # Si ça échoue, on passe
        
        # ÉTAPE 9: Restauration de toutes les annotations
        for annot_info in annotations:
            try:
                if annot_info['type'] == 8:  # Highlight
                    highlight = page.add_highlight_annot(annot_info['rect'])
                    if annot_info['colors']:
                        highlight.set_colors(stroke=annot_info['colors']['stroke'])
                    highlight.set_opacity(annot_info['opacity'])
                    highlight.update()
            except:
                pass  # Ignore si erreur de restauration

        # Mesurer le temps passé sur cette page
        page_elapsed = time.time() - (start_time + cumulative_seconds)
        cumulative_seconds += page_elapsed
        pages_done += 1

        # Appel du callback pour mettre à jour la moyenne/elapsed
        if progress_callback:
            try:
                elapsed_seconds = time.time() - start_time
                avg_per_page = cumulative_seconds / pages_done if pages_done > 0 else None
                progress_callback(page_num + 1, total_pages, elapsed_seconds, avg_per_page)
            except Exception:
                pass
    
    # ÉTAPE 10: Sauvegarde avec compression
    doc.save(output_path, garbage=4, deflate=True, clean=True)
    doc.close()
