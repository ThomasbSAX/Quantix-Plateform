import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Optional, Tuple, Union, Literal
from dateutil.parser import parse as date_parse
from dateutil.parser import ParserError
import warnings
from decimal import Decimal, InvalidOperation


def infer_and_convert_types(
    df: pd.DataFrame, 
    *,
    aggressive: bool = False,
    confidence_threshold: float = 0.8,
    preserve_nulls: bool = True,
    return_info: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Détection et conversion automatique des types avec intelligence avancée.
    
    Args:
        df: DataFrame à analyser
        aggressive: Tentatives de conversion plus poussées
        confidence_threshold: Seuil de confiance pour conversion (0.0-1.0)
        preserve_nulls: Préserver les valeurs nulles originales
        return_info: Retourner les infos de conversion
    
    Returns:
        DataFrame optimisé + optionnellement dict d'infos
    """
    df_result = df.copy()
    conversion_info = {
        "original_dtypes": dict(df.dtypes.astype(str)),
        "conversions_applied": {},
        "memory_optimization": {},
        "warnings": [],
        "confidence_scores": {}
    }
    
    original_memory = df.memory_usage(deep=True).sum()
    
    for col in df.columns:
        col_info = {"original_dtype": str(df[col].dtype)}
        
        # Analyse de base de la colonne
        col_analysis = analyze_column_content(df[col])
        conversion_info["confidence_scores"][col] = col_analysis
        
        # Si déjà numérique, optimisation
        if pd.api.types.is_numeric_dtype(df[col]):
            df_result[col] = optimize_numeric_dtype(df[col])
            col_info["action"] = "numeric_optimization"
            col_info["final_dtype"] = str(df_result[col].dtype)
            
        # Tentative de conversion numérique intelligente
        elif col_analysis["numeric_confidence"] >= confidence_threshold:
            numeric_result = smart_numeric_conversion(df[col], preserve_nulls)
            if numeric_result is not None:
                df_result[col] = numeric_result
                col_info["action"] = "to_numeric"
                col_info["final_dtype"] = str(df_result[col].dtype)
                col_info["confidence"] = col_analysis["numeric_confidence"]
        
        # Tentative de conversion datetime
        elif aggressive and col_analysis["datetime_confidence"] >= confidence_threshold:
            datetime_result = smart_datetime_conversion(df[col], preserve_nulls)
            if datetime_result is not None:
                df_result[col] = datetime_result
                col_info["action"] = "to_datetime"
                col_info["final_dtype"] = str(df_result[col].dtype)
                col_info["confidence"] = col_analysis["datetime_confidence"]
        
        # Conversion booléenne
        elif col_analysis["boolean_confidence"] >= confidence_threshold:
            bool_result = smart_boolean_conversion(df[col], preserve_nulls)
            if bool_result is not None:
                df_result[col] = bool_result
                col_info["action"] = "to_boolean"
                col_info["final_dtype"] = str(df_result[col].dtype)
        
        # Optimisation des strings/categorical
        elif df[col].dtype == 'object':
            optimized_col = optimize_string_dtype(df[col], aggressive)
            df_result[col] = optimized_col
            col_info["action"] = "string_optimization"
            col_info["final_dtype"] = str(df_result[col].dtype)
        
        else:
            col_info["action"] = "no_change"
            col_info["final_dtype"] = str(df[col].dtype)
        
        conversion_info["conversions_applied"][col] = col_info
    
    # Calcul des gains de mémoire
    final_memory = df_result.memory_usage(deep=True).sum()
    conversion_info["memory_optimization"] = {
        "original_mb": round(original_memory / 1024 / 1024, 2),
        "final_mb": round(final_memory / 1024 / 1024, 2),
        "reduction_percent": round((1 - final_memory / original_memory) * 100, 2)
    }
    
    if return_info:
        return df_result, conversion_info
    return df_result


def analyze_column_content(series: pd.Series) -> Dict[str, float]:
    """
    Analyse le contenu d'une colonne pour déterminer les types possibles.
    
    Args:
        series: Colonne à analyser
        
    Returns:
        Dict avec scores de confiance pour chaque type
    """
    non_null_data = series.dropna().astype(str)
    total_count = len(non_null_data)
    
    if total_count == 0:
        return {
            "numeric_confidence": 0.0,
            "datetime_confidence": 0.0,
            "boolean_confidence": 0.0,
            "categorical_confidence": 0.0
        }
    
    # Patterns de détection
    numeric_patterns = [
        r'^[-+]?\d+$',  # Entiers
        r'^[-+]?\d+\.\d*$',  # Décimaux
        r'^[-+]?\d*\.\d+$',  # Décimaux sans zéro initial
        r'^[-+]?\d+[,]\d*$',  # Décimaux avec virgule
        r'^[-+]?\d*[,]\d+$',  # Décimaux avec virgule sans zéro initial
        r'^[-+]?\d+\.?\d*[eE][-+]?\d+$',  # Notation scientifique
    ]
    
    datetime_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # ISO format
        r'\d{2}/\d{2}/\d{4}',  # US format
        r'\d{2}/\d{2}/\d{2}',  # Short format
        r'\d{1,2}/\d{1,2}/\d{4}',  # Variable day/month
        r'\d{4}/\d{2}/\d{2}',  # Alternative ISO
        r'\d{2}-\d{2}-\d{4}',  # European format
    ]
    
    boolean_values = {
        'true', 'false', '1', '0', 'yes', 'no', 'y', 'n', 
        'oui', 'non', 'vrai', 'faux', 'true', 'false'
    }
    
    # Comptage des matches
    numeric_matches = 0
    datetime_matches = 0
    boolean_matches = 0
    
    for value in non_null_data:
        value_lower = str(value).lower().strip()
        
        # Test numérique
        if any(re.match(pattern, value_lower) for pattern in numeric_patterns):
            numeric_matches += 1
        
        # Test datetime
        elif any(re.search(pattern, value_lower) for pattern in datetime_patterns):
            datetime_matches += 1
            # Double vérification avec dateutil
            try:
                date_parse(value_lower)
                # Bonus pour parsing réussi
            except (ParserError, ValueError):
                datetime_matches -= 0.5  # Pénalité partielle
        
        # Test booléen
        elif value_lower in boolean_values:
            boolean_matches += 1
    
    # Calcul des scores de confiance
    unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
    
    return {
        "numeric_confidence": min(numeric_matches / total_count, 1.0),
        "datetime_confidence": min(datetime_matches / total_count, 1.0),
        "boolean_confidence": min(boolean_matches / total_count, 1.0),
        "categorical_confidence": 1.0 - unique_ratio if unique_ratio < 0.5 else 0.0
    }


def smart_numeric_conversion(
    series: pd.Series, 
    preserve_nulls: bool = True
) -> Optional[pd.Series]:
    """
    Conversion numérique intelligente avec nettoyage automatique.
    
    Args:
        series: Série à convertir
        preserve_nulls: Conserver les NaN originaux
        
    Returns:
        Série numérique ou None si échec
    """
    # Copie de travail
    working_series = series.copy()
    
    # Nettoyage préliminaire
    if working_series.dtype == 'object':
        # Suppression des espaces
        working_series = working_series.astype(str).str.strip()
        
        # Remplacement virgule par point
        working_series = working_series.str.replace(',', '.', regex=False)
        
        # Suppression des caractères non-numériques courants
        working_series = working_series.str.replace(r'[€$£¥%]', '', regex=True)
        working_series = working_series.str.replace(r'\s+', '', regex=True)
        
        # Gestion des parenthèses négatives (comptabilité)
        mask_parentheses = working_series.str.contains(r'\(.*\)', na=False)
        if mask_parentheses.any():
            working_series.loc[mask_parentheses] = '-' + working_series.loc[mask_parentheses].str.replace(r'[()]', '', regex=True)
    
    # Tentative de conversion
    try:
        converted = pd.to_numeric(working_series, errors='coerce')
        
        # Vérification de la qualité de conversion
        success_rate = converted.notna().sum() / len(series)
        if success_rate < 0.7:  # Moins de 70% de succès
            return None
            
        # Optimisation du type numérique
        converted = optimize_numeric_dtype(converted)
        
        # Préservation des nulls originaux
        if preserve_nulls:
            original_nulls = series.isna()
            converted.loc[original_nulls] = np.nan
            
        return converted
        
    except Exception:
        return None


def smart_datetime_conversion(
    series: pd.Series, 
    preserve_nulls: bool = True
) -> Optional[pd.Series]:
    """
    Conversion datetime intelligente avec formats multiples.
    
    Args:
        series: Série à convertir
        preserve_nulls: Conserver les NaN originaux
        
    Returns:
        Série datetime ou None si échec
    """
    # Formats datetime courants à tester
    datetime_formats = [
        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y',
        '%Y/%m/%d', '%d/%m/%y', '%m/%d/%y', '%d-%m-%y',
        '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S',
        '%Y-%m-%d %H:%M', '%d/%m/%Y %H:%M',
        '%d/%m/%Y %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S.%f'
    ]
    
    best_conversion = None
    best_success_rate = 0
    
    # Test de chaque format
    for fmt in datetime_formats:
        try:
            converted = pd.to_datetime(series, format=fmt, errors='coerce')
            success_rate = converted.notna().sum() / len(series)
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_conversion = converted
                
        except Exception:
            continue
    
    # Tentative avec inférence automatique si formats échouent
    if best_success_rate < 0.7:
        try:
            converted = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
            success_rate = converted.notna().sum() / len(series)
            if success_rate > best_success_rate:
                best_conversion = converted
                best_success_rate = success_rate
        except Exception:
            pass
    
    # Retour si succès suffisant
    if best_success_rate >= 0.7:
        if preserve_nulls:
            original_nulls = series.isna()
            best_conversion.loc[original_nulls] = pd.NaT
        return best_conversion
    
    return None


def smart_boolean_conversion(
    series: pd.Series, 
    preserve_nulls: bool = True
) -> Optional[pd.Series]:
    """
    Conversion booléenne intelligente avec valeurs multiples.
    
    Args:
        series: Série à convertir
        preserve_nulls: Conserver les NaN originaux
        
    Returns:
        Série booléenne ou None si échec
    """
    # Mapping des valeurs booléennes
    true_values = {'1', 'true', 'yes', 'y', 'oui', 'vrai', 'on', 'active', 'enabled'}
    false_values = {'0', 'false', 'no', 'n', 'non', 'faux', 'off', 'inactive', 'disabled'}
    
    working_series = series.astype(str).str.lower().str.strip()
    
    # Vérification que toutes les valeurs non-null sont mappables
    non_null_values = set(working_series[series.notna()])
    mappable_values = true_values | false_values | {'nan', 'none', ''}
    
    if not non_null_values.issubset(mappable_values):
        return None
    
    # Conversion
    result = pd.Series(index=series.index, dtype='boolean')
    
    for idx, val in working_series.items():
        if series.isna().iloc[idx] or val in ['nan', 'none', '']:
            result.iloc[idx] = pd.NA
        elif val in true_values:
            result.iloc[idx] = True
        elif val in false_values:
            result.iloc[idx] = False
        else:
            return None  # Valeur non mappable trouvée
    
    return result


def optimize_numeric_dtype(series: pd.Series) -> pd.Series:
    """
    Optimise le type numérique pour économiser la mémoire.
    
    Args:
        series: Série numérique
        
    Returns:
        Série optimisée
    """
    if not pd.api.types.is_numeric_dtype(series):
        return series
    
    # Entiers
    if pd.api.types.is_integer_dtype(series):
        # Vérifier si on peut utiliser des types plus petits
        min_val = series.min()
        max_val = series.max()
        
        if pd.isna(min_val) or pd.isna(max_val):
            return series
            
        # Choix du type optimal
        if min_val >= 0:  # Unsigned
            if max_val <= np.iinfo(np.uint8).max:
                return series.astype('uint8')
            elif max_val <= np.iinfo(np.uint16).max:
                return series.astype('uint16')
            elif max_val <= np.iinfo(np.uint32).max:
                return series.astype('uint32')
        else:  # Signed
            if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
                return series.astype('int8')
            elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                return series.astype('int16')
            elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                return series.astype('int32')
    
    # Flottants
    elif pd.api.types.is_float_dtype(series):
        # Test si conversion en int possible (pas de décimales)
        if series.notna().all():
            if (series == series.astype(int)).all():
                return optimize_numeric_dtype(series.astype(int))
        
        # Sinon, optimiser float
        return pd.to_numeric(series, downcast='float')
    
    return series


def optimize_string_dtype(series: pd.Series, aggressive: bool = False) -> pd.Series:
    """
    Optimise les colonnes string/object pour la mémoire.
    
    Args:
        series: Série string
        aggressive: Optimisation plus poussée
        
    Returns:
        Série optimisée
    """
    if series.dtype != 'object':
        return series
    
    n_unique = series.nunique()
    n_total = len(series)
    
    if n_total == 0:
        return series
    
    unique_ratio = n_unique / n_total
    
    # Conversion en categorical si peu de valeurs uniques
    if unique_ratio < 0.5:  # Moins de 50% de valeurs uniques
        try:
            return series.astype('category')
        except Exception:
            pass
    
    # Pour mode agressif : autres optimisations
    if aggressive:
        # String accessors pour optimisation mémoire
        try:
            return series.astype('string')
        except Exception:
            pass
    
    return series


def force_type_conversion(
    df: pd.DataFrame,
    type_mapping: Dict[str, str],
    *,
    errors: Literal['raise', 'coerce', 'ignore'] = 'coerce',
    return_errors: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Force la conversion de types selon un mapping spécifié.
    
    Args:
        df: DataFrame
        type_mapping: Dict {colonne: type_cible}
        errors: Stratégie en cas d'erreur
        return_errors: Retourner les erreurs rencontrées
        
    Returns:
        DataFrame converti + optionnellement dict d'erreurs
    """
    df_result = df.copy()
    conversion_errors = {}
    
    for col, target_type in type_mapping.items():
        if col not in df.columns:
            if errors == 'raise':
                raise KeyError(f"Colonne '{col}' non trouvée")
            elif errors == 'coerce':
                conversion_errors[col] = f"Colonne non trouvée"
            continue
        
        try:
            if target_type in ['int', 'int64']:
                df_result[col] = pd.to_numeric(df[col], errors=errors).astype('int64')
            elif target_type in ['float', 'float64']:
                df_result[col] = pd.to_numeric(df[col], errors=errors)
            elif target_type == 'datetime':
                df_result[col] = pd.to_datetime(df[col], errors=errors)
            elif target_type == 'bool':
                df_result[col] = smart_boolean_conversion(df[col])
                if df_result[col] is None:
                    raise ValueError("Conversion booléenne impossible")
            elif target_type == 'category':
                df_result[col] = df[col].astype('category')
            elif target_type == 'string':
                df_result[col] = df[col].astype('string')
            else:
                df_result[col] = df[col].astype(target_type)
                
        except Exception as e:
            if errors == 'raise':
                raise
            elif errors == 'coerce':
                conversion_errors[col] = str(e)
            # 'ignore' ne fait rien
    
    if return_errors:
        return df_result, conversion_errors
    return df_result


def generate_type_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Génère un rapport détaillé sur les types de données.
    
    Args:
        df: DataFrame à analyser
        
    Returns:
        Rapport complet avec recommandations
    """
    report = {
        "summary": {
            "total_columns": len(df.columns),
            "total_memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "dtypes_distribution": dict(df.dtypes.value_counts())
        },
        "column_analysis": {},
        "optimization_potential": {},
        "recommendations": []
    }
    
    for col in df.columns:
        col_analysis = analyze_column_content(df[col])
        
        memory_current = df[col].memory_usage(deep=True)
        unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
        
        column_info = {
            "current_dtype": str(df[col].dtype),
            "memory_usage_mb": round(memory_current / 1024 / 1024, 3),
            "null_percentage": round(df[col].isna().sum() / len(df) * 100, 2),
            "unique_values": df[col].nunique(),
            "unique_ratio": round(unique_ratio, 3),
            "confidence_scores": col_analysis
        }
        
        # Recommandations d'optimisation
        recommendations = []
        if col_analysis["numeric_confidence"] > 0.8 and df[col].dtype == 'object':
            recommendations.append("Conversion numérique recommandée")
        if col_analysis["datetime_confidence"] > 0.8 and df[col].dtype == 'object':
            recommendations.append("Conversion datetime recommandée")
        if col_analysis["boolean_confidence"] > 0.8:
            recommendations.append("Conversion booléenne recommandée")
        if unique_ratio < 0.5 and df[col].dtype == 'object':
            recommendations.append("Conversion category recommandée")
            
        column_info["recommendations"] = recommendations
        report["column_analysis"][col] = column_info
    
    return report


def force_numeric(
    series: pd.Series, 
    errors: Literal['coerce', 'raise', 'ignore'] = 'coerce',
    *,
    clean_first: bool = True
) -> pd.Series:
    """
    Force la conversion numérique avec nettoyage optionnel.
    
    Args:
        series: Série à convertir
        errors: Stratégie d'erreur
        clean_first: Nettoyer avant conversion
        
    Returns:
        Série numérique
    """
    working_series = series.copy()
    
    if clean_first and working_series.dtype == 'object':
        # Nettoyage automatique
        working_series = working_series.astype(str)
        working_series = working_series.str.replace(',', '.', regex=False)

        # Préserver la notation scientifique (e/E) sinon '1e3' devient '13'.
        # On enlève le reste (unités, texte, etc.), mais on garde e/E *temporairement*.
        working_series = working_series.str.replace(r'[^\d.eE+-]', '', regex=True)

        # Retirer les e/E "parasites" (ex: '4989 euros' -> '4989e' après nettoyage)
        # On ne conserve e/E que si c'est entre des chiffres et suivi d'un exposant.
        working_series = working_series.str.replace(r'(?i)(?<!\d)[eE]', '', regex=True)
        working_series = working_series.str.replace(r'(?i)[eE](?![+-]?\d)', '', regex=True)

        # Garde-fou: éviter les faux positifs / exponents énormes (ex: 123e4567)
        sci_pat = re.compile(r'^[+-]?\d+(?:\.\d+)?[eE]([+-]?\d+)$')

        def _nullify_huge_exponent(x: str) -> str:
            if not x or ('e' not in x.lower()):
                return x
            m = sci_pat.match(x)
            if not m:
                return x
            exp_token = m.group(1)
            if len(exp_token.lstrip('+-')) > 3:
                return ''
            try:
                exp = int(exp_token)
            except Exception:
                return ''
            if exp > 308 or exp < -308:
                return ''
            return x

        working_series = working_series.apply(_nullify_huge_exponent)
    
    return pd.to_numeric(working_series, errors=errors)
