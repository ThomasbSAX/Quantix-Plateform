import pandas as pd
import numpy as np
from typing import Literal, Optional, Dict, List, Tuple, Union
import warnings

# Dépendances optionnelles (on veut que le module reste importable sur le site)
try:
    from scipy import stats  # type: ignore
except Exception:
    stats = None

try:
    from sklearn.ensemble import IsolationForest  # type: ignore
    from sklearn.neighbors import LocalOutlierFactor  # type: ignore
    from sklearn.cluster import DBSCAN  # type: ignore
except Exception:
    IsolationForest = None
    LocalOutlierFactor = None
    DBSCAN = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None


def _require(module_obj, feature_name: str) -> None:
    if module_obj is None:
        raise ImportError(
            f"Fonctionnalité '{feature_name}' indisponible: dépendance optionnelle manquante"
        )


def _median_abs_deviation(values: np.ndarray) -> float:
    """MAD robuste sans SciPy."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return mad


def detect_outliers(
    series: pd.Series,
    *,
    method: Literal["iqr", "zscore", "modified_zscore", "isolation_forest", "lof", "dbscan"] = "iqr",
    threshold: float = 1.5,
    return_bounds: bool = False
) -> Union[pd.Series, Tuple[pd.Series, Dict]]:
    """
    Détecte les valeurs aberrantes dans une série numérique avec méthodes avancées.
    
    Args:
        series: Série pandas numérique
        method: Méthode de détection
        threshold: Seuil (varie selon la méthode)
        return_bounds: Retourner aussi les limites calculées
    
    Returns:
        Masque booléen (True = outlier) + optionnellement dict avec infos
    """
    series_clean = series.dropna()
    if len(series_clean) == 0:
        return pd.Series(dtype=bool, index=series.index)
    
    outlier_mask = pd.Series(False, index=series.index)
    bounds_info = {}
    
    if method == "iqr":
        Q1 = series_clean.quantile(0.25)
        Q3 = series_clean.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        outlier_mask = (series < lower) | (series > upper)
        bounds_info = {"lower": lower, "upper": upper, "Q1": Q1, "Q3": Q3, "IQR": IQR}
    
    elif method == "zscore":
        std = float(series_clean.std())
        if std == 0.0 or np.isnan(std):
            z_scores = pd.Series(0.0, index=series_clean.index)
        else:
            z_scores = np.abs((series_clean - series_clean.mean()) / std)
        outlier_indices = series_clean[z_scores > threshold].index
        outlier_mask.loc[outlier_indices] = True
        bounds_info = {
            "mean": series_clean.mean(),
            "std": std,
            "threshold": threshold,
            "z_scores_max": z_scores.max()
        }
    
    elif method == "modified_zscore":
        # Utilise la médiane au lieu de la moyenne (plus robuste)
        median = series_clean.median()
        values = series_clean.to_numpy(dtype=float)
        mad = _median_abs_deviation(values)
        if mad == 0.0 or np.isnan(mad):
            modified_z_scores = pd.Series(0.0, index=series_clean.index)
        else:
            modified_z_scores = 0.6745 * (series_clean - median) / mad
        outlier_indices = series_clean[np.abs(modified_z_scores) > threshold].index
        outlier_mask.loc[outlier_indices] = True
        bounds_info = {
            "median": median,
            "mad": mad,
            "threshold": threshold,
            "modified_z_max": np.abs(modified_z_scores).max()
        }
    
    elif method == "isolation_forest":
        _require(IsolationForest, "isolation_forest")
        iso_forest = IsolationForest(contamination=threshold, random_state=42)
        outlier_scores = iso_forest.fit_predict(series_clean.values.reshape(-1, 1))
        outlier_indices = series_clean[outlier_scores == -1].index
        outlier_mask.loc[outlier_indices] = True
        bounds_info = {"contamination": threshold, "n_outliers": sum(outlier_scores == -1)}
    
    elif method == "lof":
        _require(LocalOutlierFactor, "lof")
        n_neighbors = min(20, len(series_clean) - 1) if len(series_clean) > 1 else 1
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=threshold)
        outlier_scores = lof.fit_predict(series_clean.values.reshape(-1, 1))
        outlier_indices = series_clean[outlier_scores == -1].index
        outlier_mask.loc[outlier_indices] = True
        bounds_info = {"n_neighbors": n_neighbors, "contamination": threshold}
    
    elif method == "dbscan":
        _require(DBSCAN, "dbscan")
        # DBSCAN pour détecter les points isolés
        dbscan = DBSCAN(eps=threshold, min_samples=2)
        labels = dbscan.fit_predict(series_clean.values.reshape(-1, 1))
        outlier_indices = series_clean[labels == -1].index  # -1 = noise/outliers
        outlier_mask.loc[outlier_indices] = True
        bounds_info = {"eps": threshold, "n_clusters": len(set(labels)) - (1 if -1 in labels else 0)}
    
    else:
        raise ValueError(f"Méthode inconnue: {method}")
    
    if return_bounds:
        return outlier_mask, bounds_info
    return outlier_mask


def detect_multivariate_outliers(
    df: pd.DataFrame,
    columns: List[str],
    *,
    method: Literal["mahalanobis", "isolation_forest", "lof"] = "mahalanobis",
    threshold: float = 3.0,
    return_scores: bool = False
) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Détection d'outliers multivariés.
    
    Args:
        df: DataFrame
        columns: Colonnes numériques à considérer
        method: Méthode de détection
        threshold: Seuil de détection
        return_scores: Retourner aussi les scores
        
    Returns:
        Masque booléen + optionnellement scores
    """
    # Sélectionner et nettoyer les données
    data = df[columns].select_dtypes(include=[np.number]).dropna()
    
    if len(data) == 0 or len(data.columns) == 0:
        outlier_mask = pd.Series(False, index=df.index)
        if return_scores:
            return outlier_mask, pd.Series(0.0, index=df.index)
        return outlier_mask
    
    outlier_mask = pd.Series(False, index=df.index)
    scores = pd.Series(0.0, index=df.index)
    
    if method == "mahalanobis":
        # Distance de Mahalanobis
        mean = data.mean()
        cov = data.cov()
        
        try:
            inv_cov = np.linalg.pinv(cov)  # Pseudo-inverse pour robustesse
            diff = data - mean
            mahal_dist = [np.sqrt(diff.iloc[i].T @ inv_cov @ diff.iloc[i]) for i in range(len(data))]
            
            scores.loc[data.index] = mahal_dist
            if stats is not None:
                chi2_threshold = stats.chi2.ppf(1 - (threshold / 100), df=len(data.columns))
                outlier_mask.loc[data.index] = np.array(mahal_dist) > chi2_threshold
            else:
                # Sans SciPy: on interprète `threshold` comme un seuil direct sur la distance
                outlier_mask.loc[data.index] = np.array(mahal_dist) > float(threshold)
            
        except np.linalg.LinAlgError:
            warnings.warn("Matrice de covariance singulière, utilisation d'Isolation Forest")
            return detect_multivariate_outliers(df, columns, method="isolation_forest", 
                                              threshold=threshold/100, return_scores=return_scores)
    
    elif method == "isolation_forest":
        _require(IsolationForest, "isolation_forest")
        iso_forest = IsolationForest(contamination=threshold, random_state=42)
        outlier_predictions = iso_forest.fit_predict(data)
        outlier_scores = iso_forest.decision_function(data)
        
        scores.loc[data.index] = outlier_scores
        outlier_mask.loc[data.index] = outlier_predictions == -1
    
    elif method == "lof":
        _require(LocalOutlierFactor, "lof")
        n_neighbors = min(20, len(data) - 1) if len(data) > 1 else 1
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=threshold)
        outlier_predictions = lof.fit_predict(data)
        outlier_scores = lof.negative_outlier_factor_
        
        scores.loc[data.index] = outlier_scores
        outlier_mask.loc[data.index] = outlier_predictions == -1
    
    if return_scores:
        return outlier_mask, scores
    return outlier_mask


def remove_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    *,
    method: Literal["iqr", "zscore", "modified_zscore", "isolation_forest", "lof"] = "iqr",
    threshold: float = 1.5,
    strategy: Literal["remove", "cap", "transform"] = "remove",
    return_info: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Supprime ou traite les valeurs aberrantes avec stratégies multiples.
    
    Args:
        df: DataFrame
        columns: Colonnes à traiter (None = toutes les colonnes numériques)
        method: Méthode de détection
        threshold: Seuil
        strategy: "remove", "cap" (écrêtage), "transform" (transformation)
        return_info: Retourner infos sur le traitement
        
    Returns:
        DataFrame traité + optionnellement dict d'infos
    """
    df_result = df.copy()
    
    if columns is None:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    info = {
        "columns_processed": [],
        "outliers_by_column": {},
        "treatment_applied": strategy,
        "total_outliers": 0,
        "original_shape": df.shape
    }
    
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # Détection des outliers
        outlier_mask, bounds = detect_outliers(
            df[col], 
            method=method, 
            threshold=threshold,
            return_bounds=True
        )
        
        outlier_count = outlier_mask.sum()
        if outlier_count == 0:
            continue
            
        info["columns_processed"].append(col)
        info["outliers_by_column"][col] = {
            "count": int(outlier_count),
            "percentage": float(outlier_count / len(df) * 100),
            "bounds": bounds,
            "outlier_values": df.loc[outlier_mask, col].tolist()[:10]  # Exemples
        }
        info["total_outliers"] += outlier_count
        
        # Application de la stratégie
        if strategy == "remove":
            # Supprimer les lignes avec outliers
            df_result = df_result[~outlier_mask]
            
        elif strategy == "cap":
            # Écrêtage aux limites
            if method in ["iqr", "zscore", "modified_zscore"] and "lower" in bounds and "upper" in bounds:
                df_result.loc[df_result[col] < bounds["lower"], col] = bounds["lower"]
                df_result.loc[df_result[col] > bounds["upper"], col] = bounds["upper"]
            else:
                # Écrêtage aux percentiles pour autres méthodes
                lower_pct = df[col].quantile(0.05)
                upper_pct = df[col].quantile(0.95)
                df_result.loc[df_result[col] < lower_pct, col] = lower_pct
                df_result.loc[df_result[col] > upper_pct, col] = upper_pct
                
        elif strategy == "transform":
            # Transformation log ou box-cox
            if (df[col] > 0).all():
                # Transformation log si toutes les valeurs sont positives
                df_result[col] = np.log1p(df[col])
                info["outliers_by_column"][col]["transformation"] = "log1p"
            else:
                # Standardisation robuste
                median = df[col].median()
                mad = stats.median_abs_deviation(df[col], scale='normal')
                df_result[col] = (df[col] - median) / mad
                info["outliers_by_column"][col]["transformation"] = "robust_standardization"
    
    info["final_shape"] = df_result.shape
    info["data_retention"] = (len(df_result) / len(df)) * 100 if len(df) > 0 else 0
    
    if return_info:
        return df_result, info
    return df_result


def outlier_analysis_report(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    methods: List[str] = ["iqr", "zscore", "modified_zscore"],
    save_plots: bool = False,
    plot_dir: str = "outlier_plots"
) -> Dict[str, any]:
    """
    Rapport complet d'analyse des outliers avec comparaison de méthodes.
    
    Args:
        df: DataFrame à analyser
        columns: Colonnes à analyser
        methods: Liste des méthodes à comparer
        save_plots: Sauvegarder les graphiques
        plot_dir: Dossier pour les graphiques
        
    Returns:
        Rapport détaillé avec statistiques et recommandations
    """
    if columns is None:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    report = {
        "summary": {},
        "by_column": {},
        "method_comparison": {},
        "recommendations": []
    }
    
    for col in columns:
        if col not in df.columns:
            continue
            
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
            
        col_analysis = {
            "basic_stats": {
                "count": len(col_data),
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "skewness": float(stats.skew(col_data)),
                "kurtosis": float(stats.kurtosis(col_data))
            },
            "outlier_detection": {}
        }
        
        # Test de chaque méthode
        method_results = {}
        for method in methods:
            try:
                if method in ["iqr", "zscore", "modified_zscore"]:
                    outlier_mask, bounds = detect_outliers(
                        col_data, method=method, return_bounds=True
                    )
                    outlier_count = outlier_mask.sum()
                    
                    method_results[method] = {
                        "outlier_count": int(outlier_count),
                        "outlier_percentage": float(outlier_count / len(col_data) * 100),
                        "bounds": bounds,
                        "outlier_indices": list(col_data[outlier_mask].index[:5])
                    }
                    
            except Exception as e:
                method_results[method] = {"error": str(e)}
        
        col_analysis["outlier_detection"] = method_results
        
        # Visualisations si demandées
        if save_plots:
            create_outlier_plots(col_data, col, method_results, plot_dir)
        
        report["by_column"][col] = col_analysis
    
    # Comparaison des méthodes et recommandations
    report["method_comparison"] = compare_outlier_methods(report["by_column"])
    report["recommendations"] = generate_outlier_recommendations(report)
    
    return report


def create_outlier_plots(data: pd.Series, column_name: str, method_results: Dict, plot_dir: str):
    """Crée des visualisations pour l'analyse des outliers."""
    import os
    os.makedirs(plot_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Analyse des Outliers - {column_name}', fontsize=16)
    
    # Boxplot
    axes[0, 0].boxplot(data)
    axes[0, 0].set_title('Boxplot')
    axes[0, 0].set_ylabel('Valeurs')
    
    # Histogramme
    axes[0, 1].hist(data, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Distribution')
    axes[0, 1].set_xlabel('Valeurs')
    axes[0, 1].set_ylabel('Fréquence')
    
    # Q-Q plot
    stats.probplot(data, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normalité)')
    
    # Scatter plot avec outliers marqués
    y_pos = np.arange(len(data))
    colors = ['red' if i in method_results.get('iqr', {}).get('outlier_indices', []) else 'blue' 
              for i in data.index]
    axes[1, 1].scatter(data.values, y_pos, c=colors, alpha=0.6)
    axes[1, 1].set_title('Valeurs avec Outliers (rouge)')
    axes[1, 1].set_xlabel('Valeurs')
    axes[1, 1].set_ylabel('Index')
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/outliers_{column_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def compare_outlier_methods(column_analyses: Dict) -> Dict:
    """Compare l'efficacité des différentes méthodes de détection."""
    comparison = {
        "method_agreement": {},
        "sensitivity_ranking": {},
        "recommended_method": {}
    }
    
    for col, analysis in column_analyses.items():
        detection_results = analysis.get("outlier_detection", {})
        
        if len(detection_results) < 2:
            continue
            
        # Accord entre méthodes
        outlier_counts = {method: result.get("outlier_count", 0) 
                         for method, result in detection_results.items()
                         if "error" not in result}
        
        if outlier_counts:
            comparison["sensitivity_ranking"][col] = dict(sorted(
                outlier_counts.items(), key=lambda x: x[1], reverse=True
            ))
            
            # Recommandation basée sur les stats de la colonne
            basic_stats = analysis["basic_stats"]
            if abs(basic_stats["skewness"]) > 1:  # Distribution asymétrique
                comparison["recommended_method"][col] = "modified_zscore"
            elif basic_stats["kurtosis"] > 3:  # Distribution leptokurtique
                comparison["recommended_method"][col] = "iqr"
            else:
                comparison["recommended_method"][col] = "zscore"
    
    return comparison


def generate_outlier_recommendations(report: Dict) -> List[str]:
    """Génère des recommandations basées sur l'analyse."""
    recommendations = []
    
    for col, analysis in report["by_column"].items():
        basic_stats = analysis["basic_stats"]
        detection_results = analysis["outlier_detection"]
        
        # Recommandations basées sur les statistiques
        if abs(basic_stats["skewness"]) > 2:
            recommendations.append(
                f"Colonne '{col}': Distribution très asymétrique (skew={basic_stats['skewness']:.2f}) "
                f"- considérer transformation log ou méthode robuste"
            )
            
        if basic_stats["kurtosis"] > 5:
            recommendations.append(
                f"Colonne '{col}': Distribution leptokurtique (kurtosis={basic_stats['kurtosis']:.2f}) "
                f"- méthode IQR recommandée"
            )
            
        # Recommandations basées sur les outliers détectés
        max_outliers = 0
        best_method = None
        for method, result in detection_results.items():
            if "error" not in result:
                outlier_pct = result["outlier_percentage"]
                if outlier_pct > max_outliers:
                    max_outliers = outlier_pct
                    best_method = method
        
        if max_outliers > 10:
            recommendations.append(
                f"Colonne '{col}': {max_outliers:.1f}% d'outliers détectés avec {best_method} "
                f"- investigation manuelle recommandée"
            )
        elif max_outliers > 5:
            recommendations.append(
                f"Colonne '{col}': {max_outliers:.1f}% d'outliers - écrêtage ou transformation recommandé"
            )
    
    return recommendations


# Fonctions compatibilité (si nécessaires pour l'ancien code)
def quick_outlier_removal(
    df: pd.DataFrame,
    columns: List[str] = None,
    method: str = 'iqr',
    threshold: float = None
) -> pd.DataFrame:
    """Suppression rapide d'outliers (wrapper)."""
    if threshold is None:
        threshold = 1.5
    return remove_outliers(df, columns=columns, method=method, threshold=float(threshold), strategy="remove")
