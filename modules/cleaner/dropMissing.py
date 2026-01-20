import pandas as pd
import numpy as np
from typing import Literal, Optional, Dict, Any, Tuple
import warnings


def drop_missing(
    df: pd.DataFrame,
    *,
    threshold: float,
    axis: Literal["x", "y", "both"] = "y",
    strategy: Literal["percentage", "count", "smart"] = "percentage",
    preserve_columns: Optional[list] = None,
    preserve_rows: Optional[list] = None,
    return_info: bool = False
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Supprime lignes ou colonnes selon la proportion de valeurs manquantes.
    Version avancée avec stratégies multiples et préservation sélective.

    Args:
        df: DataFrame à nettoyer
        threshold: Seuil de suppression [0, 1] pour percentage, nombre pour count
        axis: "x" (lignes), "y" (colonnes), "both" (les deux)
        strategy: "percentage", "count", "smart" (analyse automatique)
        preserve_columns: Liste de colonnes à préserver
        preserve_rows: Liste d'indices de lignes à préserver  
        return_info: Si True, retourne aussi les infos de suppression

    Returns:
        DataFrame nettoyé + optionnellement dict d'infos
    """
    if strategy == "percentage" and not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold doit être dans [0, 1] pour strategy='percentage'")
    
    preserve_columns = preserve_columns or []
    preserve_rows = preserve_rows or []
    
    info = {
        "rows_dropped": 0,
        "cols_dropped": 0,
        "dropped_columns": [],
        "dropped_rows": [],
        "original_shape": df.shape,
        "strategy_used": strategy
    }
    
    df_result = df.copy()
    
    # Détection des valeurs manquantes (includes NaN, None, empty strings, "NULL", etc.)
    def is_missing(x):
        if pd.isna(x):
            return True
        if isinstance(x, str):
            return x.strip() in ['', 'NULL', 'null', 'None', 'N/A', 'NA', '#N/A', 'NaN']
        return False
    
    missing_mask = df_result.applymap(is_missing)
    
    # Stratégie smart: analyse automatique des seuils optimaux
    if strategy == "smart":
        # Analyse de la distribution des valeurs manquantes
        col_missing_rates = missing_mask.mean(axis=0)
        row_missing_rates = missing_mask.mean(axis=1)
        
        # Seuils adaptatifs basés sur la distribution
        col_threshold = max(0.5, col_missing_rates.quantile(0.75))
        row_threshold = max(0.3, row_missing_rates.quantile(0.9))
        
        info["adaptive_thresholds"] = {
            "column_threshold": col_threshold,
            "row_threshold": row_threshold
        }
    else:
        col_threshold = row_threshold = threshold
    
    # Suppression des colonnes
    if axis in ["y", "both"]:
        if strategy == "percentage":
            col_missing_rates = missing_mask.mean(axis=0)
            cols_to_drop = col_missing_rates[col_missing_rates >= col_threshold].index
        elif strategy == "count":
            col_missing_counts = missing_mask.sum(axis=0)
            cols_to_drop = col_missing_counts[col_missing_counts >= threshold].index
        else:  # smart
            col_missing_rates = missing_mask.mean(axis=0)
            cols_to_drop = col_missing_rates[col_missing_rates >= col_threshold].index
            
        # Préservation de colonnes critiques
        cols_to_drop = [col for col in cols_to_drop if col not in preserve_columns]
        
        if len(cols_to_drop) > 0:
            df_result = df_result.drop(columns=cols_to_drop)
            info["cols_dropped"] = len(cols_to_drop)
            info["dropped_columns"] = list(cols_to_drop)
            
            # Recalcul du mask après suppression de colonnes
            missing_mask = df_result.applymap(is_missing)
    
    # Suppression des lignes
    if axis in ["x", "both"]:
        if strategy == "percentage":
            row_missing_rates = missing_mask.mean(axis=1)
            rows_to_drop = row_missing_rates[row_missing_rates >= row_threshold].index
        elif strategy == "count":
            row_missing_counts = missing_mask.sum(axis=1)
            rows_to_drop = row_missing_counts[row_missing_counts >= threshold].index
        else:  # smart
            row_missing_rates = missing_mask.mean(axis=1)
            rows_to_drop = row_missing_rates[row_missing_rates >= row_threshold].index
            
        # Préservation de lignes critiques
        rows_to_drop = [idx for idx in rows_to_drop if idx not in preserve_rows]
        
        if len(rows_to_drop) > 0:
            df_result = df_result.drop(index=rows_to_drop)
            info["rows_dropped"] = len(rows_to_drop)
            info["dropped_rows"] = list(rows_to_drop)
    
    info["final_shape"] = df_result.shape
    info["data_retention"] = (df_result.size / df.size) * 100 if df.size > 0 else 0
    
    # Warnings pour pertes importantes de données
    if info["data_retention"] < 50:
        warnings.warn(f"⚠️ Perte importante de données: {info['data_retention']:.1f}% conservé")
    
    if return_info:
        return df_result, info
    return df_result


def drop_empty_rows(df: pd.DataFrame, return_info: bool = False) -> pd.DataFrame | Tuple[pd.DataFrame, Dict]:
    """Supprime les lignes complètement vides."""
    empty_rows = df.isnull().all(axis=1) | (df.astype(str).str.strip() == '').all(axis=1)
    df_clean = df[~empty_rows]
    
    if return_info:
        info = {
            "empty_rows_dropped": empty_rows.sum(),
            "empty_row_indices": list(df.index[empty_rows])
        }
        return df_clean, info
    return df_clean


def drop_empty_columns(df: pd.DataFrame, return_info: bool = False) -> pd.DataFrame | Tuple[pd.DataFrame, Dict]:
    """Supprime les colonnes complètement vides."""
    empty_cols = df.isnull().all(axis=0) | (df.astype(str).str.strip() == '').all(axis=0)
    df_clean = df.loc[:, ~empty_cols]
    
    if return_info:
        info = {
            "empty_cols_dropped": empty_cols.sum(),
            "empty_columns": list(df.columns[empty_cols])
        }
        return df_clean, info
    return df_clean


def smart_missing_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyse intelligente des patterns de valeurs manquantes pour optimiser la stratégie.
    """
    missing_mask = df.isnull() | (df.astype(str).str.strip().isin(['', 'NULL', 'null', 'None', 'N/A', 'NA']))
    
    analysis = {
        "total_missing": missing_mask.sum().sum(),
        "missing_percentage": (missing_mask.sum().sum() / df.size) * 100,
        "columns_analysis": {},
        "rows_analysis": {},
        "patterns": {},
        "recommendations": []
    }
    
    # Analyse par colonne
    col_missing = missing_mask.mean(axis=0)
    analysis["columns_analysis"] = {
        "worst_columns": col_missing.nlargest(5).to_dict(),
        "columns_over_50_percent": list(col_missing[col_missing > 0.5].index),
        "completely_empty_cols": list(col_missing[col_missing == 1.0].index)
    }
    
    # Analyse par ligne
    row_missing = missing_mask.mean(axis=1)
    analysis["rows_analysis"] = {
        "worst_rows": row_missing.nlargest(5).to_dict(),
        "rows_over_50_percent": len(row_missing[row_missing > 0.5]),
        "completely_empty_rows": len(row_missing[row_missing == 1.0])
    }
    
    # Patterns de données manquantes
    analysis["patterns"] = {
        "missing_clusters": detect_missing_clusters(missing_mask),
        "systematic_missing": detect_systematic_missing(df, missing_mask),
        "missing_correlation": missing_mask.corr().abs().max().max()
    }
    
    # Recommandations automatiques
    if analysis["missing_percentage"] < 5:
        analysis["recommendations"].append("Données de bonne qualité - suppression conservative recommandée")
    elif analysis["missing_percentage"] > 30:
        analysis["recommendations"].append("Beaucoup de données manquantes - considérer l'imputation")
    
    if len(analysis["columns_analysis"]["completely_empty_cols"]) > 0:
        analysis["recommendations"].append("Supprimer les colonnes complètement vides")
        
    if analysis["rows_analysis"]["completely_empty_rows"] > 0:
        analysis["recommendations"].append("Supprimer les lignes complètement vides")
    
    return analysis


def detect_missing_clusters(missing_mask: pd.DataFrame) -> Dict[str, Any]:
    """Détecte les clusters de valeurs manquantes."""
    # Cherche des patterns dans les valeurs manquantes
    clusters = {}
    
    # Colonnes qui ont tendance à être manquantes ensemble
    if missing_mask.shape[1] > 1:
        corr_matrix = missing_mask.corr()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append({
                        "col1": corr_matrix.columns[i],
                        "col2": corr_matrix.columns[j], 
                        "correlation": corr_matrix.iloc[i, j]
                    })
        
        clusters["correlated_missing"] = high_corr_pairs
    
    return clusters


def detect_systematic_missing(df: pd.DataFrame, missing_mask: pd.DataFrame) -> Dict[str, Any]:
    """Détecte des patterns systématiques dans les valeurs manquantes."""
    systematic = {}
    
    # Vérifie si certaines valeurs dans une colonne prédisent des valeurs manquantes dans d'autres
    for col in df.columns:
        if df[col].dtype == 'object':
            for target_col in df.columns:
                if col != target_col:
                    grouped_missing = df.groupby(col)[target_col].apply(lambda x: x.isnull().mean())
                    if grouped_missing.max() - grouped_missing.min() > 0.5:  # Grande variation
                        systematic[f"{col}_predicts_{target_col}"] = {
                            "predictor": col,
                            "target": target_col,
                            "missing_rates_by_group": grouped_missing.to_dict()
                        }
    
    return systematic
