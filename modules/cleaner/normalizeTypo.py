import re
import pandas as pd
import unicodedata
from typing import Optional, Dict, List, Union, Tuple, Literal
import warnings


def normalize_typography(
    text: Optional[str], 
    *,
    mode: Literal["basic", "strict", "preserve_intent"] = "basic",
    custom_rules: Optional[Dict[str, str]] = None,
    preserve_formatting: bool = False
) -> Optional[str]:
    """
    Normalisation typographique avancée avec modes flexibles.

    Args:
        text: Texte à normaliser
        mode: 
            - "basic": normalisation standard ASCII-safe
            - "strict": normalisation maximale
            - "preserve_intent": préserve l'intention typographique
        custom_rules: Règles personnalisées {pattern: replacement}
        preserve_formatting: Préserver certains éléments de formatage
        
    Returns:
        Texte normalisé ou None si entrée None
    """
    if text is None:
        return None
    
    text = str(text).strip()
    if not text:
        return text
    
    # Règles de base communes
    normalized = text
    
    if mode == "basic":
        normalized = _apply_basic_normalization(normalized)
    elif mode == "strict":
        normalized = _apply_strict_normalization(normalized)
    elif mode == "preserve_intent":
        normalized = _apply_intent_preserving_normalization(normalized)
    
    # Application des règles personnalisées
    if custom_rules:
        for pattern, replacement in custom_rules.items():
            try:
                normalized = re.sub(pattern, replacement, normalized)
            except re.error as e:
                warnings.warn(f"Erreur dans la règle personnalisée '{pattern}': {e}")
    
    # Post-traitement selon preserve_formatting
    if not preserve_formatting:
        normalized = _clean_whitespace(normalized)
    
    return normalized


def _apply_basic_normalization(text: str) -> str:
    """Normalisation de base ASCII-safe."""
    # Guillemets typographiques → guillemets ASCII
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("«", '"').replace("»", '"')
    text = text.replace("„", '"').replace("‚", "'")
    
    # Apostrophes typographiques → apostrophe ASCII
    text = (text.replace("'", "'")
               .replace("'", "'")
               .replace("‛", "'")
               .replace("ʼ", "'")
               .replace("´", "'")
               .replace("`", "'"))
    
    # Tirets et moins Unicode → trait d'union ASCII
    text = (text.replace("–", "-")   # en-dash
               .replace("—", "-")   # em-dash
               .replace("−", "-")   # minus math
               .replace("‑", "-")   # non-breaking hyphen
               .replace("⎼", "-")   # horizontal line extension
               .replace("⎻", "-")   # horizontal line extension
               .replace("﹣", "-")   # small hyphen-minus
               .replace("－", "-"))  # fullwidth hyphen-minus
    
    # Points de suspension
    text = text.replace("…", "...")
    text = re.sub(r"\.{4,}", "...", text)  # Multiples points
    
    # Espaces insécables et autres espaces
    text = text.replace("\u00A0", " ")  # Non-breaking space
    text = text.replace("\u2007", " ")  # Figure space
    text = text.replace("\u2009", " ")  # Thin space
    text = text.replace("\u200A", " ")  # Hair space
    text = text.replace("\u202F", " ")  # Narrow no-break space
    
    return text


def _apply_strict_normalization(text: str) -> str:
    """Normalisation stricte avec décomposition Unicode."""
    # D'abord la normalisation de base
    text = _apply_basic_normalization(text)
    
    # Décomposition et normalisation Unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Suppression des accents et diacritiques (mode strict)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # Conversion majuscules accentuées
    text = text.replace("À", "A").replace("Á", "A").replace("Â", "A").replace("Ã", "A")
    text = text.replace("È", "E").replace("É", "E").replace("Ê", "E").replace("Ë", "E")
    text = text.replace("Ì", "I").replace("Í", "I").replace("Î", "I").replace("Ï", "I")
    text = text.replace("Ò", "O").replace("Ó", "O").replace("Ô", "O").replace("Õ", "O")
    text = text.replace("Ù", "U").replace("Ú", "U").replace("Û", "U").replace("Ü", "U")
    text = text.replace("Ç", "C").replace("Ñ", "N")
    
    # Ligatures
    text = text.replace("œ", "oe").replace("Œ", "OE")
    text = text.replace("æ", "ae").replace("Æ", "AE")
    text = text.replace("ß", "ss")
    
    # Symboles mathématiques vers ASCII
    text = text.replace("×", "x").replace("÷", "/")
    text = text.replace("±", "+/-").replace("∞", "infini")
    
    # Monnaies
    text = text.replace("€", "EUR").replace("£", "GBP").replace("¥", "JPY")
    text = text.replace("¢", "cents").replace("₽", "RUB")
    
    return text


def _apply_intent_preserving_normalization(text: str) -> str:
    """Normalisation qui préserve l'intention typographique."""
    # Préserver certains guillemets selon le contexte
    # Guillemets doubles en début/fin vs intérieur
    text = re.sub(r'^[""]|[""]$', '"', text)  # Début/fin → ASCII
    text = re.sub(r'\s+[""]|[""]\s+', ' " ', text)  # Avec espaces → ASCII
    
    # Préserver les apostrophes dans les contractions françaises
    contractions_fr = r"\b(l'|d'|n'|s'|t'|m'|j'|qu'|c'|aujourd'hui)"
    text = re.sub(contractions_fr, lambda m: m.group(0).replace("'", "'"), text, flags=re.IGNORECASE)
    
    # Préserver les tirets dans les mots composés
    text = re.sub(r"\b\w+[-–—]\w+\b", lambda m: m.group(0).replace("–", "-").replace("—", "-"), text)
    
    # Points de suspension : préserver l'intention
    text = text.replace("…", "...")
    
    # Espaces : normaliser sans tout supprimer
    text = re.sub(r"[ \u00A0\u2007\u2009\u200A\u202F]+", " ", text)
    
    return text


def _clean_whitespace(text: str) -> str:
    """Nettoyage des espaces et ponctuation."""
    # Espaces autour de la ponctuation
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s*'\s*", "'", text) 
    text = re.sub(r"\s*\"\s*", '"', text)
    text = re.sub(r"\s*\.{3}\s*", "...", text)
    
    # Suppression espaces multiples
    text = re.sub(r"\s+", " ", text)
    
    # Nettoyage début/fin
    text = text.strip()
    
    return text


def normalize_dataframe_typography(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    *,
    mode: Literal["basic", "strict", "preserve_intent"] = "basic",
    inplace: bool = False,
    return_info: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Normalise la typographie de toutes les colonnes textuelles d'un DataFrame.
    
    Args:
        df: DataFrame à traiter
        columns: Colonnes spécifiques (None = toutes les colonnes object)
        mode: Mode de normalisation
        inplace: Modifier en place
        return_info: Retourner informations de traitement
        
    Returns:
        DataFrame normalisé + optionnellement infos
    """
    df_result = df if inplace else df.copy()
    
    if columns is None:
        columns = [col for col in df.columns if df[col].dtype == 'object']
    else:
        columns = [col for col in columns if col in df.columns and df[col].dtype == 'object']
    
    norm_info = {
        "columns_processed": [],
        "mode_used": mode,
        "changes_count": {},
        "sample_changes": {}
    }
    
    for col in columns:
        # Compteur de changements
        changes_count = 0
        sample_changes = []
        
        original_values = df_result[col].dropna().astype(str)
        
        # Application de la normalisation
        normalized_series = original_values.apply(
            lambda x: normalize_typography(x, mode=mode)
        )
        
        # Identification des changements
        mask_changed = original_values != normalized_series
        changes_count = mask_changed.sum()
        
        if changes_count > 0:
            # Échantillon de changements
            changed_indices = original_values[mask_changed].index[:5]
            for idx in changed_indices:
                sample_changes.append({
                    "original": original_values[idx],
                    "normalized": normalized_series[idx]
                })
            
            # Application des changements
            df_result[col] = df_result[col].map(
                lambda x: normalize_typography(x, mode=mode) if pd.notna(x) else x
            )
            
            norm_info["columns_processed"].append(col)
            norm_info["changes_count"][col] = int(changes_count)
            norm_info["sample_changes"][col] = sample_changes
    
    if return_info:
        return df_result, norm_info
    return df_result


def detect_typography_issues(
    text: str,
    *,
    return_positions: bool = False
) -> Union[List[str], Tuple[List[str], Dict[str, List[int]]]]:
    """
    Détecte les problèmes typographiques dans un texte.
    
    Args:
        text: Texte à analyser
        return_positions: Retourner aussi les positions des problèmes
        
    Returns:
        Liste des problèmes + optionnellement positions
    """
    if not text:
        return ([], {}) if return_positions else []
    
    issues = []
    positions = {} if return_positions else None
    
    # Détection des problèmes courants
    
    # Guillemets typographiques
    if re.search(r'[""]', text):
        issues.append("guillemets_typographiques")
        if return_positions:
            positions["guillemets_typographiques"] = [
                m.start() for m in re.finditer(r'[""]', text)
            ]
    
    # Apostrophes typographiques
    if re.search(r"['']", text):
        issues.append("apostrophes_typographiques")
        if return_positions:
            positions["apostrophes_typographiques"] = [
                m.start() for m in re.finditer(r"['']", text)
            ]
    
    # Tirets non-ASCII
    if re.search(r'[–—−]', text):
        issues.append("tirets_non_ascii")
        if return_positions:
            positions["tirets_non_ascii"] = [
                m.start() for m in re.finditer(r'[–—−]', text)
            ]
    
    # Points de suspension Unicode
    if '…' in text:
        issues.append("points_suspension_unicode")
        if return_positions:
            positions["points_suspension_unicode"] = [
                i for i, c in enumerate(text) if c == '…'
            ]
    
    # Espaces insécables
    if '\u00A0' in text:
        issues.append("espaces_insecables")
        if return_positions:
            positions["espaces_insecables"] = [
                i for i, c in enumerate(text) if c == '\u00A0'
            ]
    
    # Espaces multiples
    if re.search(r'  +', text):
        issues.append("espaces_multiples")
        if return_positions:
            positions["espaces_multiples"] = [
                m.start() for m in re.finditer(r'  +', text)
            ]
    
    # Ponctuation mal espacée
    if re.search(r'\w[;:!?]', text):
        issues.append("ponctuation_mal_espacee")
        if return_positions:
            positions["ponctuation_mal_espacee"] = [
                m.start() for m in re.finditer(r'\w[;:!?]', text)
            ]
    
    return (issues, positions) if return_positions else issues


def create_typography_rules(
    language: Literal["fr", "en", "de", "es", "custom"] = "fr"
) -> Dict[str, str]:
    """
    Crée des règles typographiques selon la langue.
    
    Args:
        language: Code langue ou "custom"
        
    Returns:
        Dictionnaire de règles regex {pattern: replacement}
    """
    rules = {}
    
    if language == "fr":
        # Règles françaises
        rules.update({
            # Espacement avant ponctuation haute
            r'(\w)\s*([;:!?])': r'\1 \2',
            # Guillemets français avec espaces insécables
            r'"([^"]*)"': r'« \1 »',
            # Apostrophes dans contractions
            r"\b(l|d|n|s|t|m|j|qu|c)[''](\w)": r"\1'\2",
            # Nombres avec espaces insécables
            r'(\d)\s+(\d{3})': r'\1 \2'
        })
        
    elif language == "en":
        # Règles anglaises
        rules.update({
            # Pas d'espace avant ponctuation
            r'(\w)\s+([;:!?,.])': r'\1\2',
            # Apostrophes dans contractions
            r"\b(don|won|can|shouldn|wouldn|isn|aren)['']t": r"\1't",
            r"\b(I|you|we|they)['']re": r"\1're",
            r"\b(I|you|we|they)['']ve": r"\1've"
        })
        
    elif language == "de":
        # Règles allemandes
        rules.update({
            # Guillemets allemands
            r'"([^"]*)"': r'„\1"',
            # Pas d'apostrophes dans les génitifs
            r"(\w)['']s\b": r'\1s'
        })
        
    elif language == "es":
        # Règles espagnoles
        rules.update({
            # Points d'interrogation et exclamation inversés
            r'\?': '¿?',
            r'!': '¡!',
            # Pas d'apostrophes
            r"['']": ''
        })
    
    return rules


def normalize_with_language_rules(
    text: str,
    language: Literal["fr", "en", "de", "es"] = "fr",
    *,
    additional_rules: Optional[Dict[str, str]] = None
) -> str:
    """
    Normalise selon les règles typographiques d'une langue.
    
    Args:
        text: Texte à normaliser
        language: Langue cible
        additional_rules: Règles supplémentaires
        
    Returns:
        Texte normalisé selon la langue
    """
    if not text:
        return text
    
    # Normalisation de base d'abord
    normalized = normalize_typography(text, mode="preserve_intent")
    
    # Application des règles linguistiques
    lang_rules = create_typography_rules(language)
    if additional_rules:
        lang_rules.update(additional_rules)
    
    for pattern, replacement in lang_rules.items():
        try:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        except re.error as e:
            warnings.warn(f"Erreur dans la règle linguistique '{pattern}': {e}")
    
    return normalized.strip()


def batch_typography_analysis(
    texts: List[str],
    *,
    mode: Literal["basic", "strict", "preserve_intent"] = "basic",
    return_detailed: bool = False
) -> Union[Dict[str, int], Dict[str, any]]:
    """
    Analyse typographique par lot.
    
    Args:
        texts: Liste de textes à analyser
        mode: Mode d'analyse
        return_detailed: Retourner analyse détaillée
        
    Returns:
        Statistiques globales ou analyse détaillée
    """
    if not texts:
        return {}
    
    analysis = {
        "total_texts": len(texts),
        "issues_summary": {},
        "normalization_impact": {},
        "recommendations": []
    }
    
    all_issues = []
    normalization_changes = 0
    
    for i, text in enumerate(texts):
        if not text:
            continue
            
        # Détection des problèmes
        issues = detect_typography_issues(str(text))
        all_issues.extend(issues)
        
        # Test de normalisation
        normalized = normalize_typography(str(text), mode=mode)
        if normalized != str(text):
            normalization_changes += 1
        
        # Analyse détaillée si demandée
        if return_detailed and i < 10:  # Limiter aux 10 premiers pour performance
            analysis[f"text_{i}_sample"] = {
                "original": str(text)[:100] + "..." if len(str(text)) > 100 else str(text),
                "normalized": normalized[:100] + "..." if len(normalized) > 100 else normalized,
                "issues": issues
            }
    
    # Résumé des problèmes
    from collections import Counter
    issue_counts = Counter(all_issues)
    analysis["issues_summary"] = dict(issue_counts)
    
    # Impact de la normalisation
    analysis["normalization_impact"] = {
        "texts_requiring_changes": normalization_changes,
        "change_percentage": round(normalization_changes / len(texts) * 100, 2)
    }
    
    # Recommandations
    recommendations = []
    if issue_counts.get("guillemets_typographiques", 0) > len(texts) * 0.1:
        recommendations.append("Normalisation des guillemets recommandée (>10% des textes)")
    if issue_counts.get("espaces_multiples", 0) > len(texts) * 0.05:
        recommendations.append("Nettoyage des espaces multiples nécessaire")
    if normalization_changes > len(texts) * 0.2:
        recommendations.append("Normalisation typographique fortement recommandée (>20% des textes)")
        
    analysis["recommendations"] = recommendations
    
    return analysis
    """
    Normalisation typographique avancée (machine-safe).

    Opérations :
    - guillemets typographiques → guillemets ASCII
        “ ” « » → "
    - apostrophes typographiques → '
        ’ ʻ ʻ → '
    - tirets et moins Unicode → -
        – — − - → -
    - points de suspension → ...
        … → ...
    - normalisation espaces autour de la ponctuation modifiée
    """
    if s is None:
        return None
    s = str(s)

    # guillemets
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("«", '"').replace("»", '"')

    # apostrophes
    s = (
        s.replace("’", "'")
         .replace("‘", "'")
         .replace("‛", "'")
         .replace("ʼ", "'")
    )

    # tirets / moins
    s = (
        s.replace("–", "-")   # en-dash
         .replace("—", "-")   # em-dash
         .replace("−", "-")   # minus math
         .replace("-", "-")   # non-breaking hyphen
    )

    # points de suspension
    s = s.replace("…", "...")

    # nettoyage espaces autour de la ponctuation transformée
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*'\s*", "'", s)
    s = re.sub(r"\s*\"\s*", '"', s)
    s = re.sub(r"\s*\.{3}\s*", "...", s)

    return s
