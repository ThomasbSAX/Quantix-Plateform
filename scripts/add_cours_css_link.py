#!/usr/bin/env python3
"""
Script pour ajouter le lien vers cours.css dans tous les fichiers de cours HTML.
"""

import os
import re
from pathlib import Path

COURS_DIR = Path("/Users/thomasbenyazza/Downloads/Project website/Code site web/templates/cours_html")

CSS_LINK = '  <link rel="stylesheet" href="{{ url_for(\'static\', filename=\'css/cours.css\') }}">\n'

def add_cours_css(file_path):
    """
    Ajoute le lien vers cours.css avant la fermeture de </head>.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Vérifier si le lien est déjà présent
        if 'css/cours.css' in content:
            return False
        
        # Ajouter le lien juste avant </head>
        pattern = r'(</head>)'
        replacement = CSS_LINK + r'\1'
        
        new_content = re.sub(pattern, replacement, content, count=1)
        
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Erreur avec {file_path}: {e}")
        return False

def main():
    """
    Parcourt tous les fichiers HTML et ajoute le lien CSS.
    """
    if not COURS_DIR.exists():
        print(f"❌ Le dossier {COURS_DIR} n'existe pas!")
        return
    
    html_files = list(COURS_DIR.glob("*.html"))
    print(f"📁 Trouvé {len(html_files)} fichiers HTML")
    
    modified_count = 0
    
    for html_file in html_files:
        if add_cours_css(html_file):
            modified_count += 1
            if modified_count % 100 == 0:
                print(f"✓ {modified_count} fichiers modifiés...")
    
    print(f"\n✅ Terminé! {modified_count}/{len(html_files)} fichiers modifiés")

if __name__ == "__main__":
    main()
