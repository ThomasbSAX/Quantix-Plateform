"""Helpers Docling (optionnel).

Ce projet référence Docling Granite pour détecter/protéger des formules dans des PDFs/images.
Comme Docling peut ne pas être installé dans l'environnement (ex: serveur Flask minimal),
ce module est conçu pour:

- ne jamais casser l'import du package
- permettre un fallback propre (retourner None si Docling indisponible)
- fournir une protection/restauration robuste des formules dans un Markdown
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


def detect_formulas_with_docling(input_path: str):
    """Tente d'extraire un Markdown avec formules via Docling.

    Retour attendu (si dispo):
    {
        'markdown': str,
        'has_formulas': bool,
        'formulas': list[str]
    }

    Si Docling n'est pas disponible, retourne None.
    """
    # Docling n'est pas packagé ici; on le considère optionnel.
    try:
        # Placeholder: à adapter si vous ajoutez Docling plus tard.
        # from docling import ...
        raise ImportError("Docling non installé")
    except Exception:
        return None


_FORMULA_PATTERNS: List[Tuple[str, str]] = [
    (r"\$\$[\s\S]*?\$\$", "LATEX_DISPLAY"),
    (r"\\\[[\s\S]*?\\\]", "LATEX_DISPLAY_BRACKET"),
    (r"\\\([\s\S]*?\\\)", "LATEX_INLINE_PAREN"),
    # Inline $...$ (évite de capturer $$...$$ déjà protégé)
    (r"(?<!\$)\$[^\n\$]+?\$(?!\$)", "LATEX_INLINE"),
    (r"\\begin\{[a-zA-Z*]+\}[\s\S]*?\\end\{[a-zA-Z*]+\}", "LATEX_ENV"),
    # Marqueur Docling évoqué dans les docs du projet
    (r"<!--\s*formula-not-decoded\s*-->[\s\S]*?(?=\n\n|$)", "DOCLING_MARKER"),
]


def protect_formulas_in_markdown(markdown: str) -> Tuple[str, List[Dict[str, str]]]:
    """Remplace les formules par des placeholders pour éviter leur traduction.

    Retourne (texte_protégé, formula_map) où formula_map est une liste de dicts:
    {'placeholder': '___DOCLING_FORMULA_0___', 'original': '...'}
    """
    if not markdown:
        return markdown, []

    protected: List[Dict[str, str]] = []
    temp = markdown

    for pattern, kind in _FORMULA_PATTERNS:
        matches = list(re.finditer(pattern, temp, flags=re.MULTILINE))
        for match in reversed(matches):
            original = match.group(0)
            placeholder = f"___DOCLING_FORMULA_{len(protected)}___"
            protected.append({
                "placeholder": placeholder,
                "original": original,
                "type": kind,
            })
            temp = temp[:match.start()] + placeholder + temp[match.end():]

    return temp, protected


def restore_formulas_in_markdown(translated_text: str, formula_map: List[Dict[str, str]]) -> str:
    """Ré-injecte les formules protégées dans le texte traduit."""
    if not translated_text or not formula_map:
        return translated_text

    restored = translated_text
    # On restaure en ordre inverse pour éviter les collisions de remplacements
    for item in reversed(formula_map):
        placeholder = item.get("placeholder", "")
        original = item.get("original", "")
        if placeholder and placeholder in restored:
            restored = restored.replace(placeholder, original, 1)

    return restored
