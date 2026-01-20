"""
Module de détection de contenu mathématique.
"""

import re


def is_mathematical_content(text):
    """
    Détecte si un texte contient des formules mathématiques, du code ou du LaTeX.
    Retourne True si le texte doit être protégé.
    """
    if not text or len(text.strip()) < 2:
        return False
    
    # PATTERNS PRIORITAIRES - Si présent, c'est forcément du contenu mathématique/code
    priority_patterns = [
        r'\$\$',  # Formules display LaTeX
        r'\$[^$]+\$',  # Formules inline LaTeX
        r'\\\[',  # Début formule display
        r'\\\]',  # Fin formule display
        r'\\\(',  # Début formule inline
        r'\\\)',  # Fin formule inline
        r'\\begin\{',  # Environnements LaTeX
        r'\\end\{',
        r'\\frac',  # Fractions
        r'\\sum',  # Sommes
        r'\\int',  # Intégrales
        r'\\prod',  # Produits
        r'\\lim',  # Limites
        r'\\sqrt',  # Racines
        r'\\alpha|\\beta|\\gamma|\\delta|\\theta',  # Lettres grecques LaTeX
        r'```',  # Blocs de code
        r'~~~',  # Blocs de code alternatifs
    ]
    
    for pattern in priority_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    # SYMBOLES MATHÉMATIQUES UNICODE
    unicode_math_symbols = [
        r'[∫∑∏√∂∇∆∞≈≠≤≥±∓×÷⊕⊗⊂⊃∈∉∀∃]',  # Opérateurs mathématiques
        r'[α-ωΑ-Ω]',  # Alphabet grec
        r'[ℝℂℕℤℚ]',  # Ensembles de nombres
        r'[→⇒⇔←↔⟹]',  # Flèches logiques
    ]
    
    for pattern in unicode_math_symbols:
        if re.search(pattern, text):
            return True
    
    # PATTERNS DE CODE
    code_patterns = [
        r'def\s+\w+\s*\(',  # Fonction Python
        r'function\s+\w+\s*\(',  # Fonction JavaScript
        r'class\s+\w+',  # Classe
        r'import\s+\w+',  # Import Python
        r'from\s+\w+\s+import',  # Import Python from
        r'#include\s*<',  # Include C/C++
        r'=>',  # Arrow function
        r'\w+\s*:\s*\w+\s*=',  # Type hints
    ]
    
    for pattern in code_patterns:
        if re.search(pattern, text):
            return True
    
    # PATTERNS MATHÉMATIQUES STANDARDS
    math_patterns = [
        r'\^[0-9{]',  # Exposants (x^2 ou x^{10})
        r'_[0-9{]',  # Indices (x_i ou x_{10})
        r'\d+\s*[+\-*/=<>]\s*\d+',  # Opérations (2+2, x=5)
        r'\b[a-zA-Z]\s*=\s*\d+\b',  # Équations simples (x = 5, y = 10)
        r'\\[a-zA-Z]+\{',  # Commandes LaTeX avec accolades
        r'[a-zA-Z]_\{[^}]+\}',  # Indices complexes
        r'[a-zA-Z]\^\{[^}]+\}',  # Exposants complexes
        r'\bsin\b|\bcos\b|\btan\b|\blog\b|\bexp\b|\bln\b',  # Fonctions math
    ]
    
    for pattern in math_patterns:
        if re.search(pattern, text):
            return True
    
    return False


def is_likely_math_notation(text):
    """
    Détecte si un texte court est probablement une notation mathématique.
    Critères plus stricts pour les images (notation en italique, indices, etc.)
    """
    if not text or len(text) > 10:  # Les notations math sont généralement courtes
        return False
    
    # Patterns spécifiques aux notations mathématiques courantes
    math_indicators = [
        r'^[a-zA-Z]_[a-zA-Z0-9]+$',  # x_t, g_t, s_t (indices)
        r'^[a-zA-Z]\^[a-zA-Z0-9]+$',  # x^2, x^n (exposants)
        r'^[a-zA-Z]_\{[^}]+\}$',  # x_{10}
        r'^[a-zA-Z]\^\{[^}]+\}$',  # x^{10}
        r'^[a-zA-Z][0-9]$',  # x1, x2
        r'^[α-ωΑ-Ω]',  # Lettres grecques
        r'^[∫∑∏√∂∇]',  # Symboles mathématiques
    ]
    
    for pattern in math_indicators:
        if re.search(pattern, text):
            return True
    
    # Détection de lettres uniques (souvent des variables)
    if len(text) == 1 and text.isalpha():
        return True
    
    return False
