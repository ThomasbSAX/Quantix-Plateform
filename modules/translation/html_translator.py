"""
Traducteur pour les fichiers HTML.
"""

from .text_translation import translate_text
from .math_detection import is_mathematical_content


def translate_html(input_path, output_path, translator):
    """
    Traduit un fichier HTML en préservant EXACTEMENT la structure et tous les attributs.
    Gère les formules MathML et LaTeX dans le HTML.
    """
    from bs4 import BeautifulSoup, NavigableString
    
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        original_html = f.read()
    
    soup = BeautifulSoup(original_html, "html.parser")
    
    # ÉTAPE 1: Protection des balises mathématiques (MathML, LaTeX)
    protected_tags = ["script", "style", "meta", "link", "code", "pre", 
                      "math", "mrow", "mi", "mo", "mn", "msup", "msub"]
    
    # ÉTAPE 2: Traduction sélective du contenu texte
    for element in soup.descendants:
        if isinstance(element, NavigableString):
            parent = element.parent
            
            # ÉTAPE 3: Vérifications de protection
            if parent and parent.name not in protected_tags:
                text = str(element).strip()
                
                # ÉTAPE 4: Protection des formules inline
                if text and not text.startswith("<") and not is_mathematical_content(text):
                    # ÉTAPE 5: Traduction et remplacement précis
                    translated = translate_text(text, translator)
                    element.replace_with(NavigableString(translated))
    
    # ÉTAPE 6: Écriture du résultat
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(soup))
