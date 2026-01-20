import os
import json
import re

BASE = "/Users/thomasbenyazza/Downloads/Project website/Code site web"
COURS_DIR = os.path.join(BASE, "templates/cours_html")
OUTPUT_JSON = os.path.join(BASE, "static/cours_list.json")

def categorize_course(title):
    """Catégoriser un cours selon son titre"""
    title_lower = title.lower()
    
    if any(word in title_lower for word in ['trigonométrique', 'trigonometr', 'sinus', 'cosinus', 'tangent', 'arcsin', 'arccos', 'arctan']):
        return 'Trigonométrie'
    elif any(word in title_lower for word in ['série', 'series', 'somme', 'convergence', 'harmonique']):
        return 'Séries'
    elif any(word in title_lower for word in ['intégrale', 'integral', 'integration']):
        return 'Intégrales'
    elif any(word in title_lower for word in ['inégalité', 'inegalite', 'inequality']):
        return 'Inégalités'
    elif any(word in title_lower for word in ['complexité', 'complexite']):
        return 'Complexité algorithmique'
    elif any(word in title_lower for word in ['distance', 'corrélation', 'correlation']):
        return 'Distances et Corrélations'
    elif any(word in title_lower for word in ['loss', 'learning', 'neural', 'kernel', 'embedding', 'lstm', 'bert', 'transformer']):
        return 'Machine Learning'
    elif any(word in title_lower for word in ['développement', 'developpement', 'taylor', 'maclaurin', 'laurent']):
        return 'Développements limités'
    elif any(word in title_lower for word in ['matrice', 'vecteur', 'déterminant', 'determinant', 'espace vectoriel', 'algebre']):
        return 'Algèbre linéaire'
    elif any(word in title_lower for word in ['théorème', 'theoreme', 'lemme', 'proposition']):
        return 'Théorèmes'
    elif any(word in title_lower for word in ['variance', 'covariance', 'bayésien', 'bayesian', 'probabilité']):
        return 'Statistiques et Probabilités'
    elif any(word in title_lower for word in ['équation', 'equation', 'différentiel']):
        return 'Équations différentielles'
    elif any(word in title_lower for word in ['topologie', 'métrique', 'espace de']):
        return 'Topologie et Analyse fonctionnelle'
    else:
        return 'Autres'

def clean_title(filename):
    """Nettoyer le nom de fichier pour obtenir un titre lisible"""
    title = filename.replace('.html', '').replace('_', ' ')
    # Décoder les caractères spéciaux
    title = title.replace('eacute', 'é').replace('egrave', 'è')
    return title

print("=== GÉNÉRATION DE LA LISTE DES COURS ===\n")

cours_list = []
html_files = [f for f in os.listdir(COURS_DIR) if f.endswith('.html')]

print(f"Traitement de {len(html_files)} fichiers HTML...")

for html_file in html_files:
    title = clean_title(html_file)
    category = categorize_course(title)
    
    cours_list.append({
        "file": html_file,
        "title": title,
        "category": category
    })

# Trier par catégorie puis par titre
cours_list.sort(key=lambda x: (x['category'], x['title']))

# Sauvegarder en JSON
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(cours_list, f, ensure_ascii=False, indent=2)

print(f"✓ {len(cours_list)} cours enregistrés dans {OUTPUT_JSON}")

# Statistiques par catégorie
from collections import Counter
cat_count = Counter(c['category'] for c in cours_list)
print("\nRépartition par catégorie:")
for cat, count in sorted(cat_count.items(), key=lambda x: -x[1]):
    print(f"  - {cat}: {count}")
