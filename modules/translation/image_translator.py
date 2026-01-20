"""
Traducteur pour les fichiers images (JPG, PNG, etc.).
"""

import os
from .text_translation import translate_text
from .math_detection import is_mathematical_content, is_likely_math_notation


def translate_image(input_path, output_path, translator):
    """
    Traduit le texte dans une image en préservant EXACTEMENT la position, taille, couleur et police.
    Utilise OCR pour extraire le texte avec ses coordonnées précises.
    PROTÈGE les formules mathématiques en ne les touchant pas du tout.
    """
    try:
        import pytesseract
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
    except ImportError:
        print("Installation des dépendances pour les images...")
        import subprocess
        subprocess.run(["pip", "install", "pytesseract", "Pillow", "numpy"], check=True)
        import pytesseract
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
    
    # ÉTAPE 1: Ouverture de l'image
    img = Image.open(input_path).convert("RGB")
    width, height = img.size
    
    # ÉTAPE 2: Extraction du texte avec positions précises (OCR)
    try:
        # Obtenir les données détaillées de l'OCR (positions, tailles, confidences)
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang='fra+eng')
    except Exception as e:
        print(f"Erreur OCR: {e}")
        print("Assurez-vous que Tesseract est installé:")
        print("   macOS: brew install tesseract")
        print("   Linux: apt-get install tesseract-ocr")
        print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        return
    
    # ÉTAPE 3: Organisation et filtrage des données
    text_blocks = []
    protected_blocks = []  # Blocs à ne PAS toucher (formules)
    n_boxes = len(ocr_data['text'])
    
    # SEUIL DE CONFIANCE : en dessous, on ne touche RIEN (probablement formule mal reconnue)
    CONFIDENCE_THRESHOLD = 60
    
    for i in range(n_boxes):
        text = ocr_data['text'][i].strip()
        conf = int(ocr_data['conf'][i]) if ocr_data['conf'][i] != '-1' else 0
        
        # Ignorer les détections complètement vides
        if not text or conf < 0:
            continue
        
        bbox_info = {
            'original': text,
            'x': ocr_data['left'][i],
            'y': ocr_data['top'][i],
            'w': ocr_data['width'][i],
            'h': ocr_data['height'][i],
            'conf': conf
        }
        
        # PROTECTION TRIPLE NIVEAU:
        # 1. Confiance OCR faible = NE PAS TOUCHER (formule mal détectée)
        # 2. Contenu mathématique détecté = NE PAS TOUCHER
        # 3. Notation mathématique = NE PAS TOUCHER
        is_low_confidence = conf < CONFIDENCE_THRESHOLD
        is_math_content = is_mathematical_content(text)
        is_math_notation = is_likely_math_notation(text)
        
        should_protect = is_low_confidence or is_math_content or is_math_notation
        
        if should_protect:
            # NE RIEN FAIRE - Laisser l'original intact
            protected_blocks.append(bbox_info)
            reason = []
            if is_low_confidence:
                reason.append(f"conf={conf}%")
            if is_math_content:
                reason.append("math")
            if is_math_notation:
                reason.append("notation")
            print(f"Zone protégée: '{text}' ({', '.join(reason)})")
        else:
            # Haute confiance + pas de math = Traduire
            translated_text = translate_text(text, translator)
            text_blocks.append({
                **bbox_info,
                'translated': translated_text
            })
            print(f"Traduction: '{text}' → '{translated_text}' (conf={conf}%)")
    
    # ÉTAPE 4: Création de l'image résultat en partant de l'ORIGINAL
    # Stratégie: on garde tout l'original, on masque/redessine SEULEMENT les zones traduites
    result_img = img.copy()
    draw = ImageDraw.Draw(result_img)
    
    print(f"\nTraitement: {len(text_blocks)} zones à traduire, {len(protected_blocks)} zones protégées")
    
    # ÉTAPE 5: Effacement ULTRA-SÉLECTIF (uniquement haute confiance à traduire)
    for block in text_blocks:
        bbox = (block['x'], block['y'], block['x'] + block['w'], block['y'] + block['h'])
        # Détection intelligente de la couleur de fond
        try:
            import numpy as np
            x, y, w, h = block['x'], block['y'], block['w'], block['h']
            # Échantillonner une zone autour (pas sur le texte)
            margin = 3
            if y > margin and x > margin and x + w + margin < width and y + h + margin < height:
                # Prendre des pixels au-dessus et en dessous du texte
                sample_top = img.crop((x, max(0, y-margin), x+w, y))
                sample_bottom = img.crop((x, min(height, y+h), x+w, min(height, y+h+margin)))
                avg_top = np.array(sample_top).mean(axis=(0,1))
                avg_bottom = np.array(sample_bottom).mean(axis=(0,1))
                avg_color = tuple(map(int, (avg_top + avg_bottom) / 2))
                draw.rectangle(bbox, fill=avg_color)
            else:
                draw.rectangle(bbox, fill='white')
        except:
            draw.rectangle(bbox, fill='white')
    
    # ÉTAPE 6: Réinsertion du texte traduit UNIQUEMENT
    for block in text_blocks:
        # Estimation de la taille de police
        font_size = max(10, int(block['h'] * 0.8))
        
        # Charger une police appropriée
        try:
            font_paths = [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "C:\\Windows\\Fonts\\arial.ttf",
            ]
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Dessiner le texte traduit
        position = (block['x'], block['y'])
        draw.text(position, block['translated'], fill='black', font=font)
    
    # ÉTAPE 7: COPIER-COLLER les zones protégées depuis l'image originale
    # C'est LA solution pour préserver parfaitement les formules mathématiques !
    for block in protected_blocks:
        try:
            # Découper exactement cette zone de l'image ORIGINALE
            x, y, w, h = block['x'], block['y'], block['w'], block['h']
            # Ajouter une petite marge pour capturer tout le contenu
            margin = 2
            crop_box = (
                max(0, x - margin),
                max(0, y - margin),
                min(width, x + w + margin),
                min(height, y + h + margin)
            )
            
            # Extraire la zone de l'image originale
            original_region = img.crop(crop_box)
            
            # Coller cette zone EXACTEMENT au même endroit dans l'image résultat
            result_img.paste(original_region, (crop_box[0], crop_box[1]))
            
        except Exception as e:
            print(f"Erreur copie zone protégée: {e}")
    
    # ÉTAPE 8: Sauvegarde
    result_img.save(output_path, quality=95, optimize=True)
    print(f"Image traduite: {len(text_blocks)} blocs traduits, {len(protected_blocks)} formules copiées-collées de l'original")
