"""
Gestionnaires de traduction pour les formats simples (TXT, CSV, JSON).
"""

import csv
import json
import re
from .text_translation import translate_text
from .math_detection import is_mathematical_content


def translate_txt(input_path, output_path, translator):
    """
    Traduit un fichier TXT en préservant les sauts de ligne et les formules mathématiques.
    Protège les formules LaTeX (inline et display) et les blocs de code.
    """
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Protection des éléments non-traduisibles
    protected_elements = []
    temp_content = content
    
    # Patterns à protéger (formules math et code)
    patterns = [
        # FORMULES MATHÉMATIQUES
        (r"\$\$[\s\S]*?\$\$", "MATH_DISPLAY"),  # Formules display ($$...$$)
        (r"\\\[[\s\S]*?\\\]", "MATH_DISPLAY_BRACKET"),  # Formules display (\[...\])
        (r"\$[^\$\n]+?\$", "MATH_INLINE"),  # Formules inline ($...$)
        (r"\\\([^)]*?\\\)", "MATH_INLINE_PAREN"),  # Formules inline (\(...\))
        
        # ENVIRONNEMENTS LATEX
        (r"\\begin\{equation\}[\s\S]*?\\end\{equation\}", "LATEX_EQUATION"),
        (r"\\begin\{align\}[\s\S]*?\\end\{align\}", "LATEX_ALIGN"),
        (r"\\begin\{gather\}[\s\S]*?\\end\{gather\}", "LATEX_GATHER"),
        (r"\\begin\{array\}[\s\S]*?\\end\{array\}", "LATEX_ARRAY"),
        (r"\\begin\{matrix\}[\s\S]*?\\end\{matrix\}", "LATEX_MATRIX"),
        (r"\\begin\{cases\}[\s\S]*?\\end\{cases\}", "LATEX_CASES"),
        
        # BLOCS DE CODE
        (r"```[\s\S]*?```", "CODE_BLOCK"),
        (r"`[^`]+`", "INLINE_CODE"),
    ]
    
    # Protection algorithmique
    for pattern, element_type in patterns:
        matches = list(re.finditer(pattern, temp_content, re.MULTILINE))
        for match in reversed(matches):
            original = match.group(0)
            placeholder = f"___PROTECTED_{element_type}_{len(protected_elements)}___"
            protected_elements.append({
                'original': original,
                'placeholder': placeholder
            })
            temp_content = temp_content[:match.start()] + placeholder + temp_content[match.end():]
    
    # Traduction du contenu
    translated = translate_text(temp_content, translator)
    
    # Restauration des éléments protégés
    for i in reversed(range(len(protected_elements))):
        element = protected_elements[i]
        if element['placeholder'] in translated:
            translated = translated.replace(element['placeholder'], element['original'], 1)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated)


def translate_csv(input_path, output_path, translator):
    """Traduit un fichier CSV en préservant la structure exacte et les formules mathématiques."""
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Traduit chaque cellule individuellement en protégeant les formules
    translated_rows = []
    for row in rows:
        translated_row = []
        for cell in row:
            if cell.strip():
                # Ne traduit pas si c'est une formule mathématique
                if is_mathematical_content(cell):
                    translated_row.append(cell)
                else:
                    translated_row.append(translate_text(cell, translator))
            else:
                translated_row.append(cell)
        translated_rows.append(translated_row)
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(translated_rows)


def translate_json(input_path, output_path, translator):
    """Traduit un fichier JSON en préservant la structure et les formules mathématiques."""
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    
    def translate_recursive(obj):
        if isinstance(obj, str):
            # Ne traduit pas si c'est une formule mathématique
            if is_mathematical_content(obj):
                return obj
            return translate_text(obj, translator)
        elif isinstance(obj, dict):
            return {k: translate_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [translate_recursive(item) for item in obj]
        else:
            return obj
    
    translated_data = translate_recursive(data)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
