import re
from typing import Optional, Dict


def normalize_space(s: Optional[str]) -> Optional[str]:
    """
    Normalise espaces et ponctuation sans supprimer l’information :
    - supprime espaces de début/fin
    - remplace espaces multiples / tabs / retours ligne par un seul espace
    - supprime espaces avant , . ; : ! ? ) ] } et après ( [ {
    - garde une espace après , ; : sauf si fin de chaîne
    - recolle les séparateurs de milliers dans les nombres :
        "1 000" -> "1000", "1.000" -> "1000", "1 000" -> "1000"
      (uniquement quand c’est clairement un groupement par 3 chiffres)
    - normalise les séparateurs décimaux collés : "3 , 14" -> "3,14", "3 . 14" -> "3.14"
    """
    if s is None:
        return None
    s = str(s)

    # espaces Unicode (NBSP, narrow NBSP, etc.) -> espace standard
    s = s.replace("\u00A0", " ").replace("\u202F", " ").replace("\u2009", " ")

    # normalise tous les blancs en espace
    s = re.sub(r"\s+", " ", s).strip()

    # supprime espaces après ouvrants
    s = re.sub(r"([\(\[\{])\s+", r"\1", s)

    # supprime espaces avant fermants et ponctuation forte
    s = re.sub(r"\s+([,\.;:!\?\)\]\}])", r"\1", s)

    # impose espace après , ; : (si suivi d’un non-espace et pas fin)
    s = re.sub(r"([,;:])(?=\S)", r"\1 ", s)

    # retire espace après . si c’est un décimal collé (ex: "3. 14")
    s = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", s)
    s = re.sub(r"(\d),\s+(\d)", r"\1,\2", s)

    # recolle groupements de milliers "1 000" / "1.000" / "12 345 678" / "12.345.678"
    # (séparateur = espace ou point, groupes de 3 chiffres)
    s = re.sub(r"(?<=\d)[ .](?=\d{3}\b)", "", s)
    # répéter pour couvrir plusieurs groupes (12 345 678 -> 12345678)
    s = re.sub(r"(?<=\d)[ .](?=\d{3}\b)", "", s)

    # nettoyage final des espaces multiples potentiellement réintroduits
    s = re.sub(r"\s+", " ", s).strip()
    return s


def advanced_normalize_space(
    text: Optional[str], 
    *,
    preserve_structure: bool = False,
    normalize_unicode: bool = True,
    fix_punctuation: bool = True,
    handle_numbers: bool = True,
    custom_rules: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """
    Normalisation avancée des espaces avec options configurables.
    
    Args:
        text: Texte à normaliser
        preserve_structure: Préserver retours à la ligne et indentations
        normalize_unicode: Normaliser les caractères Unicode
        fix_punctuation: Corriger la ponctuation française
        handle_numbers: Gérer les séparateurs de milliers et décimaux
        custom_rules: Règles personnalisées {pattern: replacement}
    """
    if text is None:
        return None
        
    text = str(text)
    
    # Normalisation Unicode
    if normalize_unicode:
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        # Conversion des espaces Unicode courants
        unicode_spaces = {
            '\u00A0': ' ',  # Non-breaking space
            '\u202F': ' ',  # Narrow no-break space
            '\u2009': ' ',  # Thin space
            '\u2007': ' ',  # Figure space
            '\u2006': ' ',  # Six-per-em space
            '\u2005': ' ',  # Four-per-em space
            '\u2004': ' ',  # Three-per-em space  
            '\u2003': ' ',  # Em space
            '\u2002': ' ',  # En space
            '\u2000': ' ',  # En quad
            '\u2001': ' ',  # Em quad
        }
        for unicode_char, replacement in unicode_spaces.items():
            text = text.replace(unicode_char, replacement)
    
    if preserve_structure:
        # Préserver la structure mais nettoyer les espaces sur chaque ligne
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            # Préserver l'indentation
            indent = re.match(r'^(\s*)', line).group(1)
            content = line.lstrip()
            if content:
                content = re.sub(r'\s+', ' ', content).strip()
                normalized_lines.append(indent.replace('\t', '    ') + content)
            else:
                normalized_lines.append('')
        text = '\n'.join(normalized_lines)
    else:
        # Normalisation standard - tout en une ligne
        text = re.sub(r'\s+', ' ', text).strip()
    
    # Gestion de la ponctuation française
    if fix_punctuation:
        text = fix_french_punctuation(text)
    
    # Gestion des nombres
    if handle_numbers:
        text = normalize_numeric_formatting(text)
    
    # Règles personnalisées
    if custom_rules:
        for pattern, replacement in custom_rules.items():
            text = re.sub(pattern, replacement, text)
    
    return text


def fix_french_punctuation(text: str) -> str:
    """Corrige la ponctuation selon les règles françaises."""
    # Supprime espaces avant , . ) ] }
    text = re.sub(r'\s+([,.)\]}])', r'\1', text)
    
    # Ajoute espace avant : ; ! ? (règles françaises)
    text = re.sub(r'(\S)\s*([;:!?])', r'\1 \2', text)
    
    # Supprime espaces après ( [ {
    text = re.sub(r'([\(\[{])\s+', r'\1', text)
    
    # Ajoute espace après , ; (si suivi de caractère)
    text = re.sub(r'([,;])(?=\S)', r'\1 ', text)
    
    # Gestion des guillemets français
    text = re.sub(r'"\s*([^"]*?)\s*"', r'« \1 »', text)
    
    return text


def normalize_numeric_formatting(text: str) -> str:
    """Normalise le formatage des nombres."""
    # Séparateurs de milliers - groupes de 3 chiffres
    text = re.sub(r'(\d)[\s.]\s*(?=\d{3}(?:\D|$))', r'\1', text)
    
    # Décimaux - espaces autour des virgules/points décimaux
    text = re.sub(r'(\d)\s*[,.]\s*(\d)', r'\1,\2', text)  # Virgule française
    
    # Pourcentages
    text = re.sub(r'(\d)\s*%', r'\1%', text)
    
    # Unités courantes
    units = ['km', 'cm', 'mm', 'm', 'kg', 'g', 'L', 'ml', '€', '$']
    for unit in units:
        text = re.sub(rf'(\d)\s*{re.escape(unit)}', rf'\1{unit}', text)
    
    return text


import pandas as pd
from typing import Union, Dict, List


def normalize_dataframe_spaces(
    df: pd.DataFrame,
    *,
    columns: Optional[List[str]] = None,
    method: str = "standard",
    preserve_structure: bool = False,
    return_changes: bool = False
) -> Union[pd.DataFrame, tuple]:
    """
    Normalise les espaces dans un DataFrame.
    
    Args:
        df: DataFrame à traiter
        columns: Liste des colonnes (None = toutes les colonnes texte)
        method: "standard", "advanced", "french" 
        preserve_structure: Préserver structure pour le texte long
        return_changes: Retourner info sur les modifications
        
    Returns:
        DataFrame normalisé + optionnellement dict des changements
    """
    df_result = df.copy()
    changes_info = {
        'columns_processed': [],
        'changes_made': {},
        'total_changes': 0
    }
    
    # Sélection des colonnes à traiter
    if columns is None:
        columns = [col for col in df.columns if df[col].dtype == 'object']
    
    for col in columns:
        if col not in df.columns:
            continue
            
        original_series = df[col].copy()
        
        if method == "standard":
            df_result[col] = df[col].astype(str).apply(normalize_space)
        elif method == "advanced":
            df_result[col] = df[col].astype(str).apply(
                lambda x: advanced_normalize_space(
                    x, 
                    preserve_structure=preserve_structure,
                    normalize_unicode=True,
                    fix_punctuation=True,
                    handle_numbers=True
                )
            )
        elif method == "french":
            df_result[col] = df[col].astype(str).apply(
                lambda x: advanced_normalize_space(
                    x,
                    preserve_structure=preserve_structure,
                    normalize_unicode=True, 
                    fix_punctuation=True,
                    handle_numbers=False  # Éviter de casser les nombres français
                )
            )
        
        # Comptage des changements
        changes_made = (original_series != df_result[col]).sum()
        if changes_made > 0:
            changes_info['columns_processed'].append(col)
            changes_info['changes_made'][col] = int(changes_made)
            changes_info['total_changes'] += changes_made
    
    if return_changes:
        return df_result, changes_info
    return df_result
