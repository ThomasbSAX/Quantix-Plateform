"""
Traducteur pour les fichiers Markdown.
"""

import re
from .text_translation import translate_text


def translate_md(input_path, output_path, translator):
    """
    Traduit un fichier Markdown en préservant TOUTE la syntaxe et structure.
    Gère les formules mathématiques LaTeX (inline et display), les graphiques,
    et évite tout chevauchement de texte.
    """
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Étape 1: Protection des éléments non-traduisibles
    # On utilise un mapping pour éviter les chevauchements
    protected_elements = []
    temp_content = content
    
    # Patterns dans l'ordre de priorité (du plus spécifique au plus général)
    patterns = [
        # FORMULES MATHÉMATIQUES - PRIORITÉ MAXIMALE
        (r"\$\$[\s\S]*?\$\$", "MATH_DISPLAY"),  # Formules display ($$...$$)
        (r"\\\[[\s\S]*?\\\]", "MATH_DISPLAY_BRACKET"),  # Formules display (\[...\])
        (r"\$[^\$]+?\$", "MATH_INLINE"),  # Formules inline ($...$) - Élargi pour multi-lignes
        (r"\\\([^)]*?\\\)", "MATH_INLINE_PAREN"),  # Formules inline (\(...\))
        
        # ENVIRONNEMENTS LATEX
        (r"\\begin\{equation\}[\s\S]*?\\end\{equation\}", "LATEX_EQUATION"),
        (r"\\begin\{align\}[\s\S]*?\\end\{align\}", "LATEX_ALIGN"),
        (r"\\begin\{gather\}[\s\S]*?\\end\{gather\}", "LATEX_GATHER"),
        (r"\\begin\{array\}[\s\S]*?\\end\{array\}", "LATEX_ARRAY"),
        (r"\\begin\{matrix\}[\s\S]*?\\end\{matrix\}", "LATEX_MATRIX"),
        (r"\\begin\{cases\}[\s\S]*?\\end\{cases\}", "LATEX_CASES"),
        
        # BLOCS DE CODE - PRIORITÉ HAUTE
        (r"```[\s\S]*?```", "CODE_BLOCK"),  # Blocs de code avec backticks
        (r"~~~[\s\S]*?~~~", "CODE_BLOCK_TILDE"),  # Blocs de code avec tildes
        (r"`[^`]+`", "INLINE_CODE"),  # Code inline - Élargi
        
        # IMAGES ET GRAPHIQUES
        (r"!\[([^\]]*)\]\(([^\)]+)\)", "IMAGE"),  # Images
        
        # LIENS (on garde la structure, on pourra traduire le texte du lien)
        (r"\[([^\]]+)\]\(([^\)]+)\)", "LINK"),
        
        # TABLEAUX
        (r"\|[^\n]+\|", "TABLE_ROW"),  # Lignes de tableaux
        
        # LIGNES HORIZONTALES
        (r"^[ \t]*[-*_]{3,}[ \t]*$", "HORIZONTAL_RULE"),
        
        # MARQUEURS DE STRUCTURE (à protéger en début de ligne)
        (r"^[ \t]*#{1,6}[ \t]+", "HEADING_MARKER"),  # Titres
        (r"^[ \t]*[-*+][ \t]+", "LIST_MARKER"),  # Listes non ordonnées
        (r"^[ \t]*\d+\.[ \t]+", "ORDERED_LIST_MARKER"),  # Listes ordonnées
        (r"^[ \t]*>[ \t]+", "QUOTE_MARKER"),  # Citations
        
        # HTML EMBARQUÉ
        (r"<[^>]+>", "HTML_TAG"),  # Tags HTML
    ]
    
    # Étape 2: Protection algorithmique - un par un pour éviter chevauchements
    for pattern, element_type in patterns:
        matches = list(re.finditer(pattern, temp_content, re.MULTILINE))
        # Traitement en ordre inverse pour ne pas décaler les positions
        for match in reversed(matches):
            original = match.group(0)
            placeholder = f"___PROTECTED_{element_type}_{len(protected_elements)}___"
            protected_elements.append({
                'original': original,
                'type': element_type,
                'placeholder': placeholder,
                'start': match.start(),
                'end': match.end()
            })
            temp_content = temp_content[:match.start()] + placeholder + temp_content[match.end():]
    
    # Étape 3: Traduction du contenu restant (texte pur uniquement)
    translated = translate_text(temp_content, translator)
    
    # Étape 4: Restauration précise - en ordre inverse pour éviter les chevauchements
    for i in reversed(range(len(protected_elements))):
        element = protected_elements[i]
        placeholder = element['placeholder']
        original = element['original']
        
        # Remplacement unique et précis
        if placeholder in translated:
            translated = translated.replace(placeholder, original, 1)
    
    # Étape 5: Écriture du résultat
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated)
