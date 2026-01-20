"""
Traducteur amélioré pour images utilisant Docling Granite.
"""

from pathlib import Path
import tempfile
from .text_translation import translate_text
from .docling_formula_detector import (
    detect_formulas_with_docling,
    protect_formulas_in_markdown,
    restore_formulas_in_markdown
)


def translate_image_with_docling(input_path, output_path, translator):
    """
    Traduit une image en utilisant Docling pour détecter automatiquement les formules.
    Cette approche est plus robuste que l'OCR + regex.
    
    Workflow:
    1. Docling convertit l'image en Markdown avec détection des formules
    2. Les formules sont marquées avec <!-- formula-not-decoded -->
    3. On protège ces formules pendant la traduction
    4. On réinjecte les formules dans le texte traduit
    5. On reconvertit en image si nécessaire
    
    Args:
        input_path: Chemin de l'image source
        output_path: Chemin de l'image traduite
        translator: Objet traducteur
    """
    print(f"Traduction de l'image avec Docling Granite...")
    
    # ÉTAPE 1: Détection avec Docling
    docling_result = detect_formulas_with_docling(input_path)
    
    if docling_result is None:
        print("Docling non disponible pour ce format, utilisation de la méthode traditionnelle...")
        # Fallback vers l'ancienne méthode
        from .image_translator import translate_image
        return translate_image(input_path, output_path, translator)
    
    markdown = docling_result['markdown']
    has_formulas = docling_result['has_formulas']
    
    print(f"Docling: {len(docling_result['formulas'])} formule(s) détectée(s)")
    
    # ÉTAPE 2: Protection des formules
    protected_text, formula_map = protect_formulas_in_markdown(markdown)
    
    print(f"Protection: {len(formula_map)} formule(s) protégée(s)")
    
    # ÉTAPE 3: Traduction du texte (sans les formules)
    translated_text = translate_text(protected_text, translator)
    
    # ÉTAPE 4: Restauration des formules
    final_markdown = restore_formulas_in_markdown(translated_text, formula_map)
    
    # ÉTAPE 5: Sauvegarde du résultat
    # Pour l'instant, on sauvegarde en Markdown
    # TODO: Reconvertir en image si nécessaire
    output_md = Path(output_path).with_suffix('.md')
    output_md.write_text(final_markdown, encoding='utf-8')
    
    print(f"Traduction terminée: {output_md}")
    print(f"Note: Le résultat est au format Markdown. Les formules ont été préservées.")
    
    return str(output_md)


def translate_image_smart(input_path, output_path, translator, use_docling=True):
    """
    Traduction intelligente d'image avec choix automatique de la meilleure méthode.
    
    Args:
        input_path: Chemin de l'image source
        output_path: Chemin de sortie
        translator: Objet traducteur
        use_docling: Si True, tente d'utiliser Docling en premier
    """
    if use_docling:
        try:
            return translate_image_with_docling(input_path, output_path, translator)
        except Exception as e:
            print(f"Erreur Docling: {e}")
            print("Utilisation de la méthode traditionnelle...")
    
    # Fallback vers l'ancienne méthode
    from .image_translator import translate_image
    return translate_image(input_path, output_path, translator)
