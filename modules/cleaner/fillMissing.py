import pandas as pd
import numpy as np
from typing import Literal, Any, Dict, List, Optional, Union, Tuple
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings


def fill_missing(
    df: pd.DataFrame,
    *,
    strategy: Literal["mean", "median", "mode", "forward", "backward", "constant", "knn", "iterative", "smart"] = "smart",
    constant_value: Any = 0,
    columns: Optional[List[str]] = None,
    return_info: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Remplit les valeurs manquantes avec stratégies avancées et intelligence contextuelle.
    
    Args:
        df: DataFrame à traiter
        strategy: Stratégie de remplissage
            - 'mean': moyenne (numérique uniquement)
            - 'median': médiane (numérique uniquement)
            - 'mode': valeur la plus fréquente
            - 'forward': propagation avant (ffill)
            - 'backward': propagation arrière (bfill)
            - 'constant': valeur constante
            - 'knn': K-nearest neighbors imputation
            - 'iterative': MICE (Multiple Imputation by Chained Equations)
            - 'smart': stratégie automatique selon le type de données
        constant_value: Valeur pour strategy='constant'
        columns: Colonnes à traiter (None = toutes avec des valeurs manquantes)
        return_info: Retourner infos sur le remplissage
        
    Returns:
        DataFrame traité + optionnellement dict d'infos
    """
    df_result = df.copy()
    
    # Identification des colonnes à traiter
    if columns is None:
        columns = [col for col in df.columns if df[col].isna().any()]
    else:
        columns = [col for col in columns if col in df.columns and df[col].isna().any()]
    
    if not columns:
        info = {"message": "Aucune valeur manquante détectée", "columns_processed": []}
        return (df_result, info) if return_info else df_result
    
    # Informations de traçabilité
    fill_info = {
        "strategy_used": strategy,
        "columns_processed": {},
        "original_missing_count": df[columns].isna().sum().to_dict(),
        "final_missing_count": {},
        "fill_values_used": {}
    }
    
    if strategy == "smart":
        # Stratégie automatique intelligente
        df_result, smart_info = smart_fill_missing(df_result, columns)
        fill_info.update(smart_info)
        
    elif strategy in ["mean", "median", "mode", "forward", "backward", "constant"]:
        # Stratégies classiques améliorées
        df_result = classic_fill_strategies(df_result, columns, strategy, constant_value, fill_info)
        
    elif strategy == "knn":
        # K-nearest neighbors
        df_result = knn_fill_missing(df_result, columns, fill_info)
        
    elif strategy == "iterative":
        # MICE imputation
        df_result = iterative_fill_missing(df_result, columns, fill_info)
        
    else:
        raise ValueError(f"Stratégie inconnue: {strategy}")
    
    # Calcul final des statistiques
    for col in columns:
        fill_info["final_missing_count"][col] = int(df_result[col].isna().sum())
    
    if return_info:
        return df_result, fill_info
    return df_result


def smart_fill_missing(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Stratégie de remplissage intelligente selon le contexte des données.
    
    Args:
        df: DataFrame
        columns: Colonnes à traiter
        
    Returns:
        DataFrame traité + info sur stratégies utilisées
    """
    df_result = df.copy()
    smart_info = {"smart_strategies": {}, "columns_processed": {}}
    
    for col in columns:
        col_info = {"original_missing": int(df[col].isna().sum())}
        
        # Analyse du type et distribution de la colonne
        if pd.api.types.is_numeric_dtype(df[col]):
            # Colonne numérique
            non_null_data = df[col].dropna()
            
            if len(non_null_data) == 0:
                chosen_strategy = "constant"
                fill_value = 0
            else:
                # Choix basé sur la distribution
                skewness = abs(non_null_data.skew()) if len(non_null_data) > 1 else 0
                
                if skewness > 1:  # Distribution asymétrique
                    chosen_strategy = "median"
                    fill_value = non_null_data.median()
                else:  # Distribution symétrique
                    chosen_strategy = "mean"
                    fill_value = non_null_data.mean()
                
                df_result[col] = df_result[col].fillna(fill_value)
                
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Colonne datetime
            if df[col].notna().any():
                chosen_strategy = "forward_backward"
                df_result[col] = df_result[col].fillna(method='ffill').fillna(method='bfill')
                fill_value = "interpolation temporelle"
            else:
                chosen_strategy = "constant"
                fill_value = pd.NaT
                
        elif df[col].dtype == 'bool':
            # Colonne booléenne
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                chosen_strategy = "mode"
                fill_value = mode_val[0]
                df_result[col] = df_result[col].fillna(fill_value)
            else:
                chosen_strategy = "constant"
                fill_value = False
                df_result[col] = df_result[col].fillna(fill_value)
                
        else:
            # Colonne catégorielle/object
            unique_ratio = df[col].nunique() / len(df[col].dropna()) if df[col].notna().any() else 0
            
            if unique_ratio < 0.1:  # Très catégorielle
                # Mode ou valeur spéciale "Inconnu"
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    chosen_strategy = "mode"
                    fill_value = mode_val[0]
                    df_result[col] = df_result[col].fillna(fill_value)
                else:
                    chosen_strategy = "constant"
                    fill_value = "Inconnu"
                    df_result[col] = df_result[col].fillna(fill_value)
            else:
                # Données plus variées - forward/backward fill ou mode
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    chosen_strategy = "mode"
                    fill_value = mode_val[0]
                    df_result[col] = df_result[col].fillna(fill_value)
                else:
                    chosen_strategy = "constant"
                    fill_value = "Non spécifié"
                    df_result[col] = df_result[col].fillna(fill_value)
        
        col_info.update({
            "strategy_chosen": chosen_strategy,
            "fill_value": str(fill_value)[:50],  # Limiter la longueur pour affichage
            "final_missing": int(df_result[col].isna().sum())
        })
        
        smart_info["columns_processed"][col] = col_info
        smart_info["smart_strategies"][col] = chosen_strategy
    
    return df_result, smart_info


def classic_fill_strategies(
    df: pd.DataFrame, 
    columns: List[str], 
    strategy: str, 
    constant_value: Any,
    fill_info: Dict
) -> pd.DataFrame:
    """
    Application des stratégies classiques améliorées.
    
    Args:
        df: DataFrame
        columns: Colonnes à traiter
        strategy: Stratégie choisie
        constant_value: Valeur constante si applicable
        fill_info: Dict pour stocker les infos
        
    Returns:
        DataFrame traité
    """
    df_result = df.copy()
    
    for col in columns:
        original_missing = df[col].isna().sum()
        col_info = {"strategy": strategy, "original_missing": int(original_missing)}
        
        if strategy == "mean":
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].mean()
                df_result[col] = df_result[col].fillna(fill_value)
                col_info["fill_value"] = float(fill_value) if not pd.isna(fill_value) else None
            else:
                # Tentative de conversion pour colonnes object
                numeric_version = pd.to_numeric(df[col], errors='coerce')
                if numeric_version.notna().sum() > 0:
                    fill_value = numeric_version.mean()
                    # Reconvertir en format original
                    if df[col].dtype == 'object':
                        fill_str = str(int(fill_value)) if fill_value == int(fill_value) else str(fill_value)
                        df_result[col] = df_result[col].fillna(fill_str)
                    col_info["fill_value"] = float(fill_value)
                else:
                    col_info["error"] = "Impossible de calculer la moyenne sur cette colonne"
        
        elif strategy == "median":
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median()
                df_result[col] = df_result[col].fillna(fill_value)
                col_info["fill_value"] = float(fill_value) if not pd.isna(fill_value) else None
            else:
                # Tentative de conversion pour colonnes object
                numeric_version = pd.to_numeric(df[col], errors='coerce')
                if numeric_version.notna().sum() > 0:
                    fill_value = numeric_version.median()
                    if df[col].dtype == 'object':
                        fill_str = str(int(fill_value)) if fill_value == int(fill_value) else str(fill_value)
                        df_result[col] = df_result[col].fillna(fill_str)
                    col_info["fill_value"] = float(fill_value)
                else:
                    col_info["error"] = "Impossible de calculer la médiane sur cette colonne"
        
        elif strategy == "mode":
            mode_values = df[col].mode()
            if len(mode_values) > 0:
                fill_value = mode_values[0]
                df_result[col] = df_result[col].fillna(fill_value)
                col_info["fill_value"] = str(fill_value)
                col_info["mode_count"] = len(mode_values)
            else:
                col_info["error"] = "Aucun mode détecté"
        
        elif strategy == "forward":
            df_result[col] = df_result[col].fillna(method='ffill')
            col_info["method"] = "propagation_avant"
            
        elif strategy == "backward":
            df_result[col] = df_result[col].fillna(method='bfill')
            col_info["method"] = "propagation_arriere"
            
        elif strategy == "constant":
            df_result[col] = df_result[col].fillna(constant_value)
            col_info["fill_value"] = str(constant_value)
        
        col_info["final_missing"] = int(df_result[col].isna().sum())
        fill_info["columns_processed"][col] = col_info
    
    return df_result


def knn_fill_missing(df: pd.DataFrame, columns: List[str], fill_info: Dict, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Remplissage par K-nearest neighbors.
    
    Args:
        df: DataFrame
        columns: Colonnes à traiter
        fill_info: Dict pour infos
        n_neighbors: Nombre de voisins
        
    Returns:
        DataFrame traité
    """
    df_result = df.copy()
    
    # Séparer colonnes numériques et catégorielles
    numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in columns if col not in numeric_cols]
    
    # KNN pour colonnes numériques
    if numeric_cols:
        try:
            # Prendre toutes les colonnes numériques pour le contexte
            all_numeric = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            
            imputer = KNNImputer(n_neighbors=min(n_neighbors, len(df)-1))
            imputed_values = imputer.fit_transform(df[all_numeric])
            
            # Remplacer seulement les colonnes demandées
            for i, col in enumerate(all_numeric):
                if col in numeric_cols:
                    df_result[col] = imputed_values[:, i]
                    fill_info["columns_processed"][col] = {
                        "method": "KNN",
                        "n_neighbors": n_neighbors,
                        "original_missing": int(df[col].isna().sum()),
                        "final_missing": 0
                    }
                    
        except Exception as e:
            # Fallback vers médiane si KNN échoue
            for col in numeric_cols:
                fill_value = df[col].median()
                df_result[col] = df_result[col].fillna(fill_value)
                fill_info["columns_processed"][col] = {
                    "method": "median_fallback",
                    "error": f"KNN failed: {str(e)}",
                    "fill_value": float(fill_value) if not pd.isna(fill_value) else None
                }
    
    # Mode pour colonnes catégorielles
    for col in categorical_cols:
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df_result[col] = df_result[col].fillna(mode_val[0])
            fill_info["columns_processed"][col] = {
                "method": "mode",
                "fill_value": str(mode_val[0]),
                "original_missing": int(df[col].isna().sum()),
                "final_missing": int(df_result[col].isna().sum())
            }
    
    return df_result


def iterative_fill_missing(df: pd.DataFrame, columns: List[str], fill_info: Dict) -> pd.DataFrame:
    """
    Remplissage par imputation itérative (MICE).
    
    Args:
        df: DataFrame
        columns: Colonnes à traiter
        fill_info: Dict pour infos
        
    Returns:
        DataFrame traité
    """
    df_result = df.copy()
    
    # Séparer colonnes numériques et catégorielles
    numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in columns if col not in numeric_cols]
    
    # MICE pour colonnes numériques
    if numeric_cols:
        try:
            all_numeric = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            
            # Vérifier qu'il y a assez de données
            if len(df) > 10 and len(all_numeric) > 1:
                imputer = IterativeImputer(
                    estimator=RandomForestRegressor(n_estimators=10, random_state=42),
                    max_iter=10,
                    random_state=42
                )
                
                imputed_values = imputer.fit_transform(df[all_numeric])
                
                for i, col in enumerate(all_numeric):
                    if col in numeric_cols:
                        df_result[col] = imputed_values[:, i]
                        fill_info["columns_processed"][col] = {
                            "method": "MICE",
                            "estimator": "RandomForest",
                            "original_missing": int(df[col].isna().sum()),
                            "final_missing": 0
                        }
            else:
                # Fallback pour datasets trop petits
                raise ValueError("Dataset trop petit pour MICE")
                
        except Exception as e:
            # Fallback vers moyenne/médiane
            for col in numeric_cols:
                fill_value = df[col].median()
                df_result[col] = df_result[col].fillna(fill_value)
                fill_info["columns_processed"][col] = {
                    "method": "median_fallback",
                    "error": f"MICE failed: {str(e)}",
                    "fill_value": float(fill_value) if not pd.isna(fill_value) else None
                }
    
    # Mode pour colonnes catégorielles
    for col in categorical_cols:
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df_result[col] = df_result[col].fillna(mode_val[0])
            fill_info["columns_processed"][col] = {
                "method": "mode",
                "fill_value": str(mode_val[0]),
                "original_missing": int(df[col].isna().sum()),
                "final_missing": int(df_result[col].isna().sum())
            }
    
    return df_result


def interpolate_missing(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    *,
    method: Literal["linear", "polynomial", "spline", "time"] = "linear",
    order: int = 2,
    return_info: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Remplissage par interpolation pour données temporelles ou ordonnées.
    
    Args:
        df: DataFrame
        columns: Colonnes à interpoler
        method: Méthode d'interpolation
        order: Ordre pour polynomial/spline
        return_info: Retourner infos
        
    Returns:
        DataFrame interpolé + optionnellement infos
    """
    df_result = df.copy()
    
    if columns is None:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and df[col].isna().any()]
    
    interp_info = {"method": method, "columns_processed": {}}
    
    for col in columns:
        if col not in df.columns or not df[col].isna().any():
            continue
            
        original_missing = df[col].isna().sum()
        
        try:
            if method == "linear":
                df_result[col] = df_result[col].interpolate(method='linear')
            elif method == "polynomial":
                df_result[col] = df_result[col].interpolate(method='polynomial', order=order)
            elif method == "spline":
                df_result[col] = df_result[col].interpolate(method='spline', order=order)
            elif method == "time":
                # Nécessite un index temporel
                if pd.api.types.is_datetime64_any_dtype(df.index):
                    df_result[col] = df_result[col].interpolate(method='time')
                else:
                    # Fallback vers linear
                    df_result[col] = df_result[col].interpolate(method='linear')
                    method = "linear_fallback"
            
            final_missing = df_result[col].isna().sum()
            
            interp_info["columns_processed"][col] = {
                "original_missing": int(original_missing),
                "final_missing": int(final_missing),
                "method_used": method,
                "success": True
            }
            
        except Exception as e:
            interp_info["columns_processed"][col] = {
                "original_missing": int(original_missing),
                "error": str(e),
                "success": False
            }
    
    if return_info:
        return df_result, interp_info
    return df_result


def advanced_missing_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyse avancée des patterns de valeurs manquantes.
    
    Args:
        df: DataFrame à analyser
        
    Returns:
        Rapport détaillé sur les valeurs manquantes
    """
    report = {
        "summary": {
            "total_missing": int(df.isna().sum().sum()),
            "missing_percentage": round(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
            "columns_with_missing": int((df.isna().sum() > 0).sum()),
            "rows_with_missing": int(df.isna().any(axis=1).sum())
        },
        "by_column": {},
        "missing_patterns": {},
        "recommendations": []
    }
    
    # Analyse par colonne
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            report["by_column"][col] = {
                "missing_count": int(missing_count),
                "missing_percentage": round(missing_count / len(df) * 100, 2),
                "dtype": str(df[col].dtype),
                "unique_values": int(df[col].nunique()),
                "most_common": str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else None
            }
    
    # Patterns de valeurs manquantes
    missing_patterns = df.isna().value_counts()
    report["missing_patterns"]["top_patterns"] = [
        {
            "pattern": [col for col, is_missing in zip(df.columns, pattern) if is_missing],
            "count": int(count)
        }
        for pattern, count in missing_patterns.head(5).items()
    ]
    
    # Recommandations
    recommendations = []
    for col, info in report["by_column"].items():
        if info["missing_percentage"] > 50:
            recommendations.append(f"Colonne '{col}': {info['missing_percentage']:.1f}% manquant - considérer suppression")
        elif info["missing_percentage"] > 20:
            recommendations.append(f"Colonne '{col}': {info['missing_percentage']:.1f}% manquant - imputation avancée recommandée")
        elif info["dtype"] == "float64" and info["missing_percentage"] > 5:
            recommendations.append(f"Colonne '{col}': colonne numérique avec {info['missing_percentage']:.1f}% manquant - médiane ou KNN recommandé")
    
    report["recommendations"] = recommendations
    
    return report


def create_missing_heatmap(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Crée une heatmap des valeurs manquantes.
    
    Args:
        df: DataFrame
        save_path: Chemin de sauvegarde optionnel
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isna(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Heatmap des Valeurs Manquantes')
        plt.xlabel('Colonnes')
        plt.ylabel('Lignes')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()
        
    except ImportError:
        warnings.warn("matplotlib/seaborn non disponible pour la visualisation")
