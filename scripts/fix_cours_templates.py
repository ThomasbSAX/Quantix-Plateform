#!/usr/bin/env python3
"""
Script pour remplacer les headers et footers complets par des includes Jinja2
dans tous les fichiers de cours HTML.
"""

import os
import re
from pathlib import Path

# Chemin vers le dossier des cours HTML
COURS_DIR = Path("/Users/thomasbenyazza/Downloads/Project website/Code site web/templates/cours_html")

def fix_template_includes(file_path):
    """
    Remplace les headers et footers complets par des includes Jinja2.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern pour matcher le header complet (de <!-- ===== HEADER ===== --> jusqu'avant <!DOCTYPE html>)
        header_pattern = r'<!-- ===== HEADER ===== -->.*?</script>\s*(?=<!DOCTYPE html>)'
        
        # Pattern pour matcher le footer complet (de <!-- ===== FOOTER ===== --> jusqu'à la fin)
        footer_pattern = r'<!-- ===== FOOTER ===== -->.*?<script src="{{ url_for\(\'static\', filename=\'js/navigation\.js\'\) }}"></script>'
        
        # Remplacer le header par l'include
        content = re.sub(header_pattern, '{% include "header.html" %}\n', content, flags=re.DOTALL)
        
        # Remplacer le footer par l'include
        content = re.sub(footer_pattern, '{% include "footer.html" %}', content, flags=re.DOTALL)
        
        # Si le contenu a changé, écrire le fichier
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
        
    except Exception as e:
        print(f"Erreur avec {file_path}: {e}")
        return False

def main():
    """
    Parcourt tous les fichiers HTML et remplace les headers/footers.
    """
    if not COURS_DIR.exists():
        print(f"❌ Le dossier {COURS_DIR} n'existe pas!")
        return
    
    html_files = list(COURS_DIR.glob("*.html"))
    print(f"📁 Trouvé {len(html_files)} fichiers HTML dans {COURS_DIR}")
    
    modified_count = 0
    
    for html_file in html_files:
        if fix_template_includes(html_file):
            modified_count += 1
            if modified_count % 100 == 0:
                print(f"✓ {modified_count} fichiers modifiés...")
    
    print(f"\n✅ Terminé! {modified_count}/{len(html_files)} fichiers modifiés")

if __name__ == "__main__":
    main()
