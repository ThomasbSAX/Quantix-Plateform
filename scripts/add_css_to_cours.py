#!/usr/bin/env python3
"""
Script pour ajouter les liens CSS nécessaires dans le <head> de chaque cours HTML.
"""

import os
import re
from pathlib import Path

COURS_DIR = Path("/Users/thomasbenyazza/Downloads/Project website/Code site web/templates/cours_html")

CSS_LINKS = """  <!-- Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

  <!-- Font Awesome -->
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">

  <!-- CSS du site -->
  <link rel="stylesheet" href="{{ url_for('static', filename='css/base.css') }}">
  <link rel="stylesheet" href="{{ url_for('static', filename='css/layout.css') }}">
  <link rel="stylesheet" href="{{ url_for('static', filename='css/navbar.css') }}">
  <link rel="stylesheet" href="{{ url_for('static', filename='css/footer.css') }}">
  <link rel="stylesheet" href="{{ url_for('static', filename='css/components.css') }}">
"""

def add_css_to_head(file_path):
    """
    Ajoute les liens CSS dans le <head> de chaque fichier de cours.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Vérifier si les liens CSS sont déjà présents
        if 'css/navbar.css' in content:
            return False
        
        # Trouver la balise <head> et insérer les liens CSS après
        pattern = r'(<head>\s*\n)'
        replacement = r'\1' + CSS_LINKS + '\n'
        
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
    Parcourt tous les fichiers HTML et ajoute les liens CSS.
    """
    if not COURS_DIR.exists():
        print(f"❌ Le dossier {COURS_DIR} n'existe pas!")
        return
    
    html_files = list(COURS_DIR.glob("*.html"))
    print(f"📁 Trouvé {len(html_files)} fichiers HTML")
    
    modified_count = 0
    
    for html_file in html_files:
        if add_css_to_head(html_file):
            modified_count += 1
            if modified_count % 100 == 0:
                print(f"✓ {modified_count} fichiers modifiés...")
    
    print(f"\n✅ Terminé! {modified_count}/{len(html_files)} fichiers modifiés")

if __name__ == "__main__":
    main()
