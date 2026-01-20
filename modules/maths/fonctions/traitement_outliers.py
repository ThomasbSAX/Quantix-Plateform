"""
Quantix – Module traitement_outliers
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# detection_visuelle
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def detection_visuelle_fit(
    data: np.ndarray,
    normalisation: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Détecte les outliers de manière visuelle en utilisant différentes méthodes de normalisation,
    métriques et distances.

    Parameters:
    -----------
    data : np.ndarray
        Données d'entrée de forme (n_samples, n_features)
    normalisation : str, optional
        Méthode de normalisation ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Métrique à utiliser ('mse', 'mae', 'r2', 'logloss') ou fonction personnalisée
    distance : str or callable, optional
        Distance à utiliser ('euclidean', 'manhattan', 'cosine', 'minkowski') ou fonction personnalisée
    solver : str, optional
        Solveur à utiliser ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    regularization : str, optional
        Type de régularisation ('none', 'l1', 'l2', 'elasticnet')
    tol : float, optional
        Tolérance pour la convergence
    max_iter : int, optional
        Nombre maximum d'itérations
    custom_metric : callable, optional
        Fonction personnalisée pour la métrique
    custom_distance : callable, optional
        Fonction personnalisée pour la distance

    Returns:
    --------
    dict
        Dictionnaire contenant les résultats, métriques, paramètres utilisés et avertissements

    Example:
    --------
    >>> data = np.random.randn(100, 5)
    >>> result = detection_visuelle_fit(data, normalisation='standard', metric='mse')
    """
    # Validation des entrées
    _validate_inputs(data, normalisation, metric, distance, solver, regularization)

    # Normalisation des données
    normalized_data = _apply_normalization(data, normalisation)

    # Calcul des distances
    distances = _compute_distances(normalized_data, distance, custom_distance)

    # Détection des outliers
    outliers = _detect_outliers(distances, metric, custom_metric)

    # Calcul des métriques
    metrics = _compute_metrics(normalized_data, outliers, metric, custom_metric)

    # Retourne les résultats
    return {
        'result': outliers,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'metric': metric if not callable(metric) else 'custom',
            'distance': distance if not callable(distance) else 'custom',
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

def _validate_inputs(
    data: np.ndarray,
    normalisation: str,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str,
    regularization: Optional[str]
) -> None:
    """Valide les entrées et lève des erreurs si nécessaire."""
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Les données doivent être un tableau numpy 2D.")
    if normalisation not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("Méthode de normalisation non valide.")
    if isinstance(metric, str) and metric not in ['mse', 'mae', 'r2', 'logloss']:
        raise ValueError("Métrique non valide.")
    if isinstance(distance, str) and distance not in ['euclidean', 'manhattan', 'cosine', 'minkowski']:
        raise ValueError("Distance non valide.")
    if solver not in ['closed_form', 'gradient_descent', 'newton', 'coordinate_descent']:
        raise ValueError("Solveur non valide.")
    if regularization is not None and regularization not in ['none', 'l1', 'l2', 'elasticnet']:
        raise ValueError("Type de régularisation non valide.")

def _apply_normalization(
    data: np.ndarray,
    normalisation: str
) -> np.ndarray:
    """Applique la normalisation choisie aux données."""
    if normalisation == 'none':
        return data
    elif normalisation == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif normalisation == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif normalisation == 'robust':
        median = np.median(data, axis=0)
        iqr = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)
        return (data - median) / iqr
    else:
        raise ValueError("Méthode de normalisation non valide.")

def _compute_distances(
    data: np.ndarray,
    distance: Union[str, Callable],
    custom_distance: Optional[Callable]
) -> np.ndarray:
    """Calcule les distances entre les points."""
    if callable(custom_distance):
        return custom_distance(data)
    elif distance == 'euclidean':
        return np.sqrt(np.sum((data[:, np.newaxis] - data) ** 2, axis=2))
    elif distance == 'manhattan':
        return np.sum(np.abs(data[:, np.newaxis] - data), axis=2)
    elif distance == 'cosine':
        return 1 - np.dot(data, data.T) / (np.linalg.norm(data, axis=1)[:, np.newaxis] * np.linalg.norm(data, axis=1))
    elif distance == 'minkowski':
        return np.sum(np.abs(data[:, np.newaxis] - data) ** 3, axis=2) ** (1/3)
    else:
        raise ValueError("Distance non valide.")

def _detect_outliers(
    distances: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> np.ndarray:
    """Détecte les outliers en fonction de la métrique choisie."""
    if callable(custom_metric):
        return custom_metric(distances)
    elif metric == 'mse':
        mean_dist = np.mean(distances, axis=1)
        std_dist = np.std(distances, axis=1)
        return (mean_dist > mean_dist.mean() + 3 * std_dist.std()).astype(int)
    elif metric == 'mae':
        median_dist = np.median(distances, axis=1)
        mad = np.median(np.abs(distances - median_dist[:, np.newaxis]), axis=1)
        return (np.abs(median_dist - distances.mean(axis=1)) > 3 * mad).astype(int)
    elif metric == 'r2':
        return np.zeros(distances.shape[0], dtype=int)
    elif metric == 'logloss':
        return np.zeros(distances.shape[0], dtype=int)
    else:
        raise ValueError("Métrique non valide.")

def _compute_metrics(
    data: np.ndarray,
    outliers: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calcule les métriques pour évaluer la détection des outliers."""
    if callable(custom_metric):
        return {'custom_metric': custom_metric(data, outliers)}
    elif metric == 'mse':
        return {'mse': np.mean((data[outliers == 1] - data[outliers == 0].mean(axis=0)) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(data[outliers == 1] - data[outliers == 0].mean(axis=0)))}
    elif metric == 'r2':
        return {'r2': 1 - np.sum((data[outliers == 1] - data[outliers == 0].mean(axis=0)) ** 2) / np.sum((data[outliers == 1] - data.mean(axis=0)) ** 2)}
    elif metric == 'logloss':
        return {'logloss': -np.mean(outliers * np.log(outliers) + (1 - outliers) * np.log(1 - outliers))}
    else:
        raise ValueError("Métrique non valide.")

################################################################################
# z_score
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    data: np.ndarray,
    threshold: float = 3.0
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")
    if threshold <= 0:
        raise ValueError("Threshold must be positive")

def _compute_z_scores(
    data: np.ndarray,
    mean: float,
    std: float
) -> np.ndarray:
    """Compute z-scores for the given data."""
    return (data - mean) / std

def _identify_outliers(
    z_scores: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Identify outliers based on z-scores."""
    return np.abs(z_scores) > threshold

def _normalize_data(
    data: np.ndarray,
    normalization: str = "standard"
) -> np.ndarray:
    """Normalize data using the specified method."""
    if normalization == "standard":
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    elif normalization == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    elif normalization == "robust":
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    elif normalization == "none":
        return data.copy()
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def z_score_fit(
    data: np.ndarray,
    threshold: float = 3.0,
    normalization: str = "standard",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Fit the z-score method to identify outliers in the data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    threshold : float, optional
        Threshold for identifying outliers (default: 3.0).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard').
    custom_metric : Callable, optional
        Custom metric function to evaluate the results.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_inputs(data, threshold)

    normalized_data = _normalize_data(data, normalization)
    mean = np.mean(normalized_data) if normalization != "none" else np.mean(data)
    std = np.std(normalized_data) if normalization != "none" else np.std(data)
    z_scores = _compute_z_scores(normalized_data, mean, std)
    outliers = _identify_outliers(z_scores, threshold)

    metrics = {}
    if custom_metric is not None:
        metrics["custom"] = custom_metric(data, outliers)

    result = {
        "z_scores": z_scores,
        "outliers": outliers
    }

    params_used = {
        "threshold": threshold,
        "normalization": normalization
    }

    warnings = []
    if np.any(outliers):
        warnings.append("Outliers detected in the data")

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
# data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
# result = z_score_fit(data, threshold=3.0)

################################################################################
# iqr
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for IQR computation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def _compute_iqr(data: np.ndarray, multiplier: float = 1.5) -> Dict[str, Union[float, np.ndarray]]:
    """Compute IQR and related statistics."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr_value = q3 - q1
    lower_bound = q1 - multiplier * iqr_value
    upper_bound = q3 + multiplier * iqr_value
    outliers_mask = (data < lower_bound) | (data > upper_bound)
    return {
        "iqr": iqr_value,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outliers_mask": outliers_mask
    }

def _apply_normalization(data: np.ndarray, normalization: str) -> np.ndarray:
    """Apply specified normalization to data."""
    if normalization == "none":
        return data
    elif normalization == "standard":
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    elif normalization == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    elif normalization == "robust":
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        return (data - q1) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def iqr_fit(
    data: np.ndarray,
    multiplier: float = 1.5,
    normalization: str = "none",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Compute IQR-based outlier detection with configurable parameters.

    Parameters
    ----------
    data : np.ndarray
        Input data array (1-dimensional)
    multiplier : float, optional
        Multiplier for IQR bounds (default 1.5)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_metric : Callable, optional
        Custom metric function (takes two arrays and returns float)

    Returns
    -------
    dict
        Dictionary containing:
        - "result": IQR computation results
        - "metrics": Computed metrics
        - "params_used": Parameters used
        - "warnings": Any warnings generated

    Example
    -------
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> result = iqr_fit(data)
    """
    _validate_input(data)

    # Apply normalization
    normalized_data = _apply_normalization(data, normalization)
    params_used = {
        "multiplier": multiplier,
        "normalization": normalization
    }

    # Compute IQR statistics
    iqr_stats = _compute_iqr(normalized_data, multiplier)

    # Compute metrics
    metrics = {}
    if custom_metric is not None:
        outliers = normalized_data[iqr_stats["outliers_mask"]]
        inliers = normalized_data[~iqr_stats["outliers_mask"]]
        metrics["custom_metric"] = custom_metric(outliers, inliers)

    # Prepare output
    result = {
        "result": iqr_stats,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": []
    }

    return result

################################################################################
# madt
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

def normalize_data(
    X: np.ndarray,
    method: str = "standard",
    custom_norm: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data using specified method."""
    if custom_norm is not None:
        return custom_norm(X)

    X_normalized = X.copy()
    if method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / std
    elif method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_normalized = (X - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - median) / (iqr + 1e-8)
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_normalized

def compute_madt(
    X: np.ndarray,
    threshold: float = 3.0,
    metric: str = "mse",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """Compute MAD-based thresholding for outlier detection."""
    validate_inputs(X)

    # Normalize data
    X_normalized = normalize_data(X, method="robust")

    # Compute median absolute deviations
    medians = np.median(X_normalized, axis=0)
    mad = np.median(np.abs(X_normalized - medians), axis=0)

    # Identify outliers
    outlier_mask = np.abs(X_normalized - medians) > (threshold * mad)

    # Compute metrics
    if custom_metric is not None:
        metric_value = custom_metric(X, outlier_mask)
    else:
        if metric == "mse":
            residuals = X_normalized[outlier_mask] - medians[None, :]
            metric_value = np.mean(residuals**2)
        elif metric == "mae":
            residuals = X_normalized[outlier_mask] - medians[None, :]
            metric_value = np.mean(np.abs(residuals))
        elif metric == "r2":
            total_var = np.var(X_normalized, axis=0)
            explained_var = np.sum((medians[None, :] - medians)**2, axis=0)
            metric_value = 1 - (np.sum(total_var) / np.sum(explained_var))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return {
        "result": outlier_mask,
        "metrics": {"value": metric_value, "name": metric},
        "params_used": {
            "threshold": threshold,
            "normalization": "robust",
            "metric": metric
        },
        "warnings": []
    }

def madt_fit(
    X: np.ndarray,
    threshold: float = 3.0,
    metric: str = "mse",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    normalize_method: str = "robust",
    custom_norm: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Main function for MAD-based outlier detection.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix (n_samples, n_features)
    threshold : float
        Threshold for outlier detection in terms of MAD
    metric : str or callable
        Metric to evaluate outlier detection performance
    custom_metric : callable, optional
        Custom metric function
    normalize_method : str
        Normalization method to use
    custom_norm : callable, optional
        Custom normalization function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = madt_fit(X, threshold=3.0, metric="mse")
    """
    validate_inputs(X)

    # Normalize data
    X_normalized = normalize_data(
        X,
        method=normalize_method,
        custom_norm=custom_norm
    )

    # Compute median absolute deviations
    medians = np.median(X_normalized, axis=0)
    mad_values = np.median(np.abs(X_normalized - medians), axis=0)

    # Identify outliers
    outlier_mask = np.abs(X_normalized - medians) > (threshold * mad_values)

    # Compute metrics
    if custom_metric is not None:
        metric_value = custom_metric(X, outlier_mask)
    else:
        if metric == "mse":
            residuals = X_normalized[outlier_mask] - medians[None, :]
            metric_value = np.mean(residuals**2)
        elif metric == "mae":
            residuals = X_normalized[outlier_mask] - medians[None, :]
            metric_value = np.mean(np.abs(residuals))
        elif metric == "r2":
            total_var = np.var(X_normalized, axis=0)
            explained_var = np.sum((medians[None, :] - medians)**2, axis=0)
            metric_value = 1 - (np.sum(total_var) / np.sum(explained_var))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return {
        "result": outlier_mask,
        "metrics": {"value": metric_value, "name": metric},
        "params_used": {
            "threshold": threshold,
            "normalization": normalize_method,
            "metric": metric
        },
        "warnings": []
    }

################################################################################
# dbscan
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(X: np.ndarray, eps: float, min_samples: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")

def _compute_distance(
    X: np.ndarray,
    distance_metric: Union[str, Callable],
    **kwargs
) -> np.ndarray:
    """Compute pairwise distances between samples."""
    if distance_metric == "euclidean":
        return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    elif distance_metric == "manhattan":
        return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
    elif distance_metric == "cosine":
        dot_products = np.dot(X, X.T)
        norms = np.sqrt(np.sum(X ** 2, axis=1))
        return 1 - dot_products / np.outer(norms, norms)
    elif callable(distance_metric):
        return distance_metric(X, **kwargs)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

def _dbscan_core(
    X: np.ndarray,
    eps: float,
    min_samples: int,
    distance_metric: Union[str, Callable],
    **kwargs
) -> Dict[str, Any]:
    """Core DBSCAN algorithm implementation."""
    n_samples = X.shape[0]
    labels = np.full(n_samples, -1, dtype=int)  # -1: noise
    cluster_id = 0

    distances = _compute_distance(X, distance_metric, **kwargs)

    for i in range(n_samples):
        if labels[i] != -1:
            continue

        neighbors = np.where(distances[i] <= eps)[0]
        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seed_set = set(neighbors)

        while seed_set:
            j = seed_set.pop()
            if labels[j] == -1:
                labels[j] = cluster_id
            elif labels[j] != -1 and labels[j] != cluster_id:
                continue

            new_neighbors = np.where(distances[j] <= eps)[0]
            if len(new_neighbors) >= min_samples:
                seed_set.update(set(new_neighbors) - set(labels[new_neighbors] != -1))

        cluster_id += 1

    return {"labels": labels, "n_clusters": cluster_id}

def dbscan_fit(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    distance_metric: Union[str, Callable] = "euclidean",
    **kwargs
) -> Dict[str, Any]:
    """
    DBSCAN clustering algorithm for outlier detection.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    eps : float
        Maximum distance between two samples for one to be considered in the neighborhood of the other
    min_samples : int
        Number of samples in a neighborhood for a point to be considered as a core point
    distance_metric : str or callable
        Distance metric to use. Can be 'euclidean', 'manhattan', 'cosine' or a custom callable function
    **kwargs :
        Additional keyword arguments for the distance metric

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": dictionary with clustering results
        - "metrics": dictionary with computed metrics
        - "params_used": dictionary of parameters used
        - "warnings": list of warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> result = dbscan_fit(X, eps=0.3, min_samples=3)
    """
    _validate_inputs(X, eps, min_samples)

    warnings = []
    params_used = {
        "eps": eps,
        "min_samples": min_samples,
        "distance_metric": distance_metric
    }

    result = _dbscan_core(X, eps, min_samples, distance_metric, **kwargs)

    metrics = {
        "n_clusters": result["n_clusters"],
        "n_noise_points": np.sum(result["labels"] == -1)
    }

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# isolation_forest
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def isolation_forest_fit(
    X: np.ndarray,
    n_estimators: int = 100,
    max_samples: Union[int, float] = 'auto',
    contamination: float = 0.1,
    max_features: Union[int, float] = 1.0,
    bootstrap: bool = False,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    metric: str = 'euclidean',
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Fit an Isolation Forest model to detect outliers.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_estimators : int
        Number of base estimators in the ensemble
    max_samples : Union[int, float]
        The number of samples to draw to train each base estimator
    contamination : float
        Expected proportion of outliers in the data
    max_features : Union[int, float]
        Number of features to draw for each split
    bootstrap : bool
        Whether bootstrap samples are used when building trees
    n_jobs : Optional[int]
        Number of parallel jobs to run
    random_state : Optional[int]
        Random seed for reproducibility
    metric : str
        Distance metric to use ('euclidean', 'manhattan', 'cosine')
    normalizer : Callable
        Function to normalize the input data
    custom_metric : Optional[Callable]
        Custom distance metric function

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the fitted model and related information
    """
    # Validate inputs
    _validate_inputs(X, contamination)

    # Normalize data
    X_normalized = normalizer(X)

    # Set random state
    rng = np.random.RandomState(random_state)

    # Initialize parameters dictionary
    params_used = {
        'n_estimators': n_estimators,
        'max_samples': max_samples if isinstance(max_samples, int) else _get_auto_max_samples(X.shape[0], max_samples),
        'contamination': contamination,
        'max_features': int(max_features) if isinstance(max_features, float) else max_features,
        'bootstrap': bootstrap,
        'metric': metric,
        'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom'
    }

    # Prepare the ensemble
    trees = []
    for _ in range(n_estimators):
        tree = _build_isolation_tree(
            X_normalized,
            max_samples=params_used['max_samples'],
            max_features=params_used['max_features'],
            bootstrap=bootstrap,
            metric=metric,
            custom_metric=custom_metric,
            random_state=rng
        )
        trees.append(tree)

    # Calculate anomaly scores
    scores = _calculate_anomaly_scores(X_normalized, trees)

    # Calculate decision function
    decision_function = _calculate_decision_function(scores, contamination)

    result = {
        'anomaly_scores': scores,
        'decision_function': decision_function
    }

    metrics = {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores)
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(X: np.ndarray, contamination: float) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input data contains infinite values")
    if not 0 <= contamination <= 1:
        raise ValueError("contamination must be between 0 and 1")

def _get_auto_max_samples(n_samples: int, max_samples: float) -> int:
    """Calculate automatic max_samples value."""
    if not 0 < max_samples <= 1:
        raise ValueError("max_samples must be between 0 and 1 when using 'auto'")
    return max(2, int(max_samples * n_samples))

def _build_isolation_tree(
    X: np.ndarray,
    max_samples: int,
    max_features: Union[int, float],
    bootstrap: bool,
    metric: str,
    custom_metric: Optional[Callable],
    random_state: np.random.RandomState
) -> Dict[str, Any]:
    """Build a single isolation tree."""
    # Sample data
    if bootstrap:
        sample_idx = random_state.choice(X.shape[0], size=max_samples, replace=True)
    else:
        sample_idx = random_state.choice(X.shape[0], size=max_samples, replace=False)
    X_sample = X[sample_idx]

    # Select features
    if isinstance(max_features, float):
        n_features = int(max_features * X.shape[1])
    else:
        n_features = max_features
    feature_idx = random_state.choice(X.shape[1], size=n_features, replace=False)

    # Build tree
    tree = _build_tree_recursive(
        X_sample[:, feature_idx],
        metric,
        custom_metric,
        random_state
    )

    return {
        'tree': tree,
        'features_used': feature_idx
    }

def _build_tree_recursive(
    X: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable],
    random_state: np.random.RandomState
) -> Dict[str, Any]:
    """Recursively build an isolation tree."""
    if X.shape[0] <= 1:
        return {
            'is_leaf': True,
            'path_length': _calculate_path_length(X.shape[0]),
            'split_feature': None,
            'split_value': None
        }

    # Select split point
    feature_idx = random_state.choice(X.shape[1])
    split_value, left_idx, right_idx = _find_split_point(
        X[:, feature_idx],
        metric,
        custom_metric
    )

    # Split data
    left_child = _build_tree_recursive(
        X[left_idx],
        metric,
        custom_metric,
        random_state
    )
    right_child = _build_tree_recursive(
        X[right_idx],
        metric,
        custom_metric,
        random_state
    )

    return {
        'is_leaf': False,
        'split_feature': feature_idx,
        'split_value': split_value,
        'left_child': left_child,
        'right_child': right_child
    }

def _find_split_point(
    feature: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable]
) -> tuple:
    """Find the optimal split point for a feature."""
    # Sort values
    sorted_idx = np.argsort(feature)
    sorted_values = feature[sorted_idx]

    # Find midpoints
    midpoints = (sorted_values[1:] + sorted_values[:-1]) / 2

    # Evaluate splits
    best_split = None
    best_score = -np.inf

    for i, split_value in enumerate(midpoints):
        left_idx = sorted_idx[:i+1]
        right_idx = sorted_idx[i+1:]

        if len(left_idx) == 0 or len(right_idx) == 0:
            continue

        if custom_metric is not None:
            score = _evaluate_custom_split(feature, left_idx, right_idx, custom_metric)
        else:
            score = _evaluate_standard_split(feature, left_idx, right_idx, metric)

        if score > best_score:
            best_score = score
            best_split = (split_value, left_idx, right_idx)

    if best_split is None:
        # Default split at median
        median_idx = len(feature) // 2
        left_idx = sorted_idx[:median_idx]
        right_idx = sorted_idx[median_idx:]
        best_split = (feature[left_idx[-1]], left_idx, right_idx)

    return best_split

def _evaluate_standard_split(
    feature: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    metric: str
) -> float:
    """Evaluate a split using standard distance metrics."""
    left_values = feature[left_idx]
    right_values = feature[right_idx]

    if metric == 'euclidean':
        left_mean = np.mean(left_values)
        right_mean = np.mean(right_values)
        return -(np.sum((left_values - left_mean)**2) + np.sum((right_values - right_mean)**2))
    elif metric == 'manhattan':
        left_median = np.median(left_values)
        right_median = np.median(right_values)
        return -(np.sum(np.abs(left_values - left_median)) + np.sum(np.abs(right_values - right_median)))
    elif metric == 'cosine':
        left_norm = np.linalg.norm(left_values)
        right_norm = np.linalg.norm(right_values)
        return -(np.sum(np.abs(left_values)) / left_norm + np.sum(np.abs(right_values)) / right_norm)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _evaluate_custom_split(
    feature: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    custom_metric: Callable
) -> float:
    """Evaluate a split using a custom distance metric."""
    left_values = feature[left_idx]
    right_values = feature[right_idx]

    # Calculate within-group distances
    left_distances = np.zeros((len(left_values), len(left_values)))
    for i in range(len(left_values)):
        for j in range(i+1, len(left_values)):
            left_distances[i,j] = custom_metric(
                np.array([left_values[i]]),
                np.array([left_values[j]])
            )
    left_sum = np.sum(left_distances)

    right_distances = np.zeros((len(right_values), len(right_values)))
    for i in range(len(right_values)):
        for j in range(i+1, len(right_values)):
            right_distances[i,j] = custom_metric(
                np.array([right_values[i]]),
                np.array([right_values[j]])
            )
    right_sum = np.sum(right_distances)

    return -(left_sum + right_sum)

def _calculate_path_length(n_samples: int) -> float:
    """Calculate the path length for a leaf node."""
    if n_samples == 0:
        return 1
    return np.log2(n_samples - 1) + 0.5

def _calculate_anomaly_scores(X: np.ndarray, trees: list) -> np.ndarray:
    """Calculate anomaly scores for each sample."""
    scores = np.zeros(X.shape[0])

    for tree in trees:
        path_lengths = _get_path_lengths(X, tree['tree'], tree['features_used'])
        scores += 2 ** (-path_lengths / _calculate_average_path_length(len(path_lengths)))

    return scores

def _get_path_lengths(
    X: np.ndarray,
    tree: Dict[str, Any],
    features_used: np.ndarray
) -> np.ndarray:
    """Get path lengths for each sample in the tree."""
    path_lengths = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        path_lengths[i] = _traverse_tree(
            X[i, features_used],
            tree
        )

    return path_lengths

def _traverse_tree(
    sample: np.ndarray,
    tree: Dict[str, Any]
) -> float:
    """Traverse the tree to get path length for a sample."""
    if tree['is_leaf']:
        return tree['path_length']

    feature_idx = tree['split_feature']
    split_value = tree['split_value']

    if sample[feature_idx] <= split_value:
        return _traverse_tree(sample, tree['left_child']) + 1
    else:
        return _traverse_tree(sample, tree['right_child']) + 1

def _calculate_average_path_length(n_samples: int) -> float:
    """Calculate the average path length for a tree."""
    if n_samples <= 1:
        return 0
    return np.log2(n_samples - 1) + 0.5

def _calculate_decision_function(scores: np.ndarray, contamination: float) -> np.ndarray:
    """Calculate the decision function based on anomaly scores."""
    threshold = np.quantile(scores, 1 - contamination)
    return (scores >= threshold).astype(int)

################################################################################
# local_outlier_factor
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

def default_distance_metric(x: np.ndarray, y: np.ndarray) -> float:
    """Default Euclidean distance metric."""
    return np.linalg.norm(x - y)

def local_outlier_factor_fit(
    X: np.ndarray,
    n_neighbors: int = 20,
    metric: Callable[[np.ndarray, np.ndarray], float] = default_distance_metric,
    normalization: str = "standard",
    contamination: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute Local Outlier Factor (LOF) for anomaly detection.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_neighbors : int, optional
        Number of neighbors to consider (default: 20)
    metric : Callable, optional
        Distance metric function (default: Euclidean distance)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    contamination : float, optional
        Expected proportion of outliers (for thresholding)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> result = local_outlier_factor_fit(X, n_neighbors=10)
    """
    # Validate input
    validate_input(X)

    # Normalize data
    if normalization == "standard":
        scaler = StandardScaler()
    elif normalization == "minmax":
        scaler = MinMaxScaler()
    elif normalization == "robust":
        scaler = RobustScaler()
    else:
        scaler = None

    X_normalized = X if scaler is None else scaler.fit_transform(X)

    # Compute k-distance
    k_distances = compute_k_distance(X_normalized, n_neighbors, metric)

    # Compute reachability distances
    reach_dist = compute_reachability_distance(X_normalized, k_distances, n_neighbors, metric)

    # Compute local reachability density
    lrd = compute_local_reachability_density(reach_dist, n_neighbors)

    # Compute LOF scores
    lof_scores = compute_lof_scores(X_normalized, lrd, n_neighbors, metric)

    # Prepare results
    result = {
        "result": lof_scores,
        "metrics": {
            "mean_lof_score": np.mean(lof_scores),
            "std_lof_score": np.std(lof_scores)
        },
        "params_used": {
            "n_neighbors": n_neighbors,
            "metric": metric.__name__ if hasattr(metric, '__name__') else str(metric),
            "normalization": normalization,
            "contamination": contamination
        },
        "warnings": []
    }

    return result

def compute_k_distance(X: np.ndarray, n_neighbors: int, metric: Callable) -> np.ndarray:
    """Compute the k-distance for each point."""
    distances = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            distances[i, j] = metric(X[i], X[j])

    k_distances = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        sorted_distances = np.sort(distances[i])
        k_distances[i] = sorted_distances[n_neighbors]

    return k_distances

def compute_reachability_distance(X: np.ndarray, k_distances: np.ndarray,
                                n_neighbors: int, metric: Callable) -> np.ndarray:
    """Compute reachability distances between points."""
    n_samples = X.shape[0]
    reach_dist = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            if i == j:
                reach_dist[i, j] = 0
            else:
                distance = metric(X[i], X[j])
                reach_dist[i, j] = max(distance, k_distances[j])

    return reach_dist

def compute_local_reachability_density(reach_dist: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Compute local reachability density for each point."""
    n_samples = reach_dist.shape[0]
    lrd = np.zeros(n_samples)

    for i in range(n_samples):
        # Get k nearest neighbors
        neighbor_indices = np.argsort(reach_dist[i])[1:n_neighbors+1]
        # Compute average reachability distance
        avg_reach_dist = np.mean(reach_dist[i, neighbor_indices])
        # Local reachability density is inverse of average
        lrd[i] = 1.0 / avg_reach_dist if avg_reach_dist != 0 else np.inf

    return lrd

def compute_lof_scores(X: np.ndarray, lrd: np.ndarray,
                      n_neighbors: int, metric: Callable) -> np.ndarray:
    """Compute LOF scores for each point."""
    n_samples = X.shape[0]
    lof_scores = np.zeros(n_samples)

    for i in range(n_samples):
        # Get k nearest neighbors
        distances = np.zeros(n_samples)
        for j in range(n_samples):
            distances[j] = metric(X[i], X[j])
        neighbor_indices = np.argsort(distances)[1:n_neighbors+1]

        # Compute average lrd of neighbors
        avg_lrd = np.mean(lrd[neighbor_indices])

        # Compute LOF score
        lof_scores[i] = avg_lrd / lrd[i]

    return lof_scores

################################################################################
# one_class_svm
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def one_class_svm_fit(
    X: np.ndarray,
    *,
    kernel: str = 'rbf',
    gamma: Union[str, float] = 'scale',
    nu: float = 0.1,
    normalize: Optional[str] = None,
    distance_metric: str = 'euclidean',
    solver: str = 'libsvm',
    tol: float = 1e-3,
    max_iter: int = 1000,
    custom_kernel: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Fit a One-Class SVM model to detect outliers.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    kernel : str or callable, default='rbf'
        Kernel function to use. Can be 'linear', 'poly', 'rbf', 'sigmoid' or a custom callable.
    gamma : str or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If 'scale', uses 1/(n_features * X.var()).
    nu : float, default=0.1
        An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
    normalize : str or None, default=None
        Normalization method. Can be 'standard', 'minmax', 'robust' or None.
    distance_metric : str, default='euclidean'
        Distance metric for custom kernel. Can be 'euclidean', 'manhattan', 'cosine' or a custom callable.
    solver : str, default='libsvm'
        Solver to use. Currently only 'libsvm' is supported.
    tol : float, default=1e-3
        Tolerance for stopping criteria.
    max_iter : int, default=1000
        Maximum number of iterations.
    custom_kernel : callable or None, default=None
        Custom kernel function. If provided, overrides the kernel parameter.
    custom_distance : callable or None, default=None
        Custom distance function. If provided, overrides the distance_metric parameter.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        A dictionary containing:
        - 'result': The decision function values for the input data.
        - 'metrics': Dictionary of metrics (currently empty).
        - 'params_used': Dictionary of parameters used.
        - 'warnings': List of warnings encountered.

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = one_class_svm_fit(X, kernel='rbf', nu=0.1)
    """
    # Validate inputs
    X = _validate_input(X)

    # Normalize data if specified
    if normalize is not None:
        X = _normalize_data(X, method=normalize)

    # Prepare parameters
    params_used = {
        'kernel': kernel,
        'gamma': gamma,
        'nu': nu,
        'normalize': normalize,
        'distance_metric': distance_metric,
        'solver': solver,
        'tol': tol,
        'max_iter': max_iter
    }

    # Compute kernel matrix
    if custom_kernel is not None:
        K = custom_kernel(X, X)
    else:
        K = _compute_kernel_matrix(X, kernel=kernel, gamma=gamma)

    # Solve the one-class SVM problem
    alpha = _solve_one_class_svm(K, nu=nu, tol=tol, max_iter=max_iter)

    # Compute decision function
    result = _compute_decision_function(X, K, alpha)

    return {
        'result': result,
        'metrics': {},
        'params_used': params_used,
        'warnings': []
    }

def _validate_input(X: np.ndarray) -> np.ndarray:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D array.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input X must not contain NaN or Inf values.")
    return X

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize the input data."""
    if method == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return X

def _compute_kernel_matrix(X: np.ndarray, kernel: str, gamma: Union[str, float]) -> np.ndarray:
    """Compute the kernel matrix."""
    n_samples = X.shape[0]

    if gamma == 'scale':
        gamma = 1.0 / (X.shape[1] * np.var(X))

    if kernel == 'linear':
        K = np.dot(X, X.T)
    elif kernel == 'poly':
        K = (np.dot(X, X.T) + 1) ** gamma
    elif kernel == 'rbf':
        pairwise_sq_dists = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
        K = np.exp(-gamma * pairwise_sq_dists)
    elif kernel == 'sigmoid':
        K = np.tanh(gamma * np.dot(X, X.T) + 1)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    return K

def _solve_one_class_svm(K: np.ndarray, nu: float, tol: float, max_iter: int) -> np.ndarray:
    """Solve the one-class SVM problem."""
    n_samples = K.shape[0]

    # Initialize dual coefficients
    alpha = np.zeros(n_samples)

    for _ in range(max_iter):
        # Compute gradients
        grad = -1.0 / (2 * n_samples) + K.dot(alpha)

        # Update alpha
        alpha_new = np.clip(alpha - grad, 0, (1.0 / n_samples) * nu)

        # Check convergence
        if np.linalg.norm(alpha_new - alpha, ord=np.inf) < tol:
            break

        alpha = alpha_new

    return alpha

def _compute_decision_function(X: np.ndarray, K: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Compute the decision function."""
    return K.dot(alpha) - 0.5 * np.sum(alpha**2 * K)

################################################################################
# median_absolute_deviation
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for median absolute deviation calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _calculate_median(data: np.ndarray) -> float:
    """Calculate the median of the data."""
    return np.median(data)

def _calculate_mad(data: np.ndarray, median: float) -> float:
    """Calculate the median absolute deviation from the median."""
    return np.median(np.abs(data - median))

def _normalize_data(
    data: np.ndarray,
    normalization: str = "none",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize the data based on the specified normalization method."""
    if custom_normalization is not None:
        return custom_normalization(data)
    if normalization == "standard":
        return (data - np.mean(data)) / np.std(data)
    elif normalization == "minmax":
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif normalization == "robust":
        median = _calculate_median(data)
        mad = _calculate_mad(data, median)
        return (data - median) / mad
    elif normalization == "none":
        return data.copy()
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def median_absolute_deviation_fit(
    data: np.ndarray,
    normalization: str = "none",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    threshold_factor: float = 3.0
) -> Dict[str, Any]:
    """
    Calculate the median absolute deviation (MAD) and identify outliers.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    normalization : str, optional
        Normalization method to apply ("none", "standard", "minmax", "robust").
    custom_normalization : Callable, optional
        Custom normalization function.
    threshold_factor : float, optional
        Factor to determine the outlier threshold (default is 3.0).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    _validate_input(data)

    # Normalize data
    normalized_data = _normalize_data(
        data,
        normalization=normalization,
        custom_normalization=custom_normalization
    )

    # Calculate median and MAD
    median = _calculate_median(normalized_data)
    mad = _calculate_mad(normalized_data, median)

    # Identify outliers
    threshold = threshold_factor * mad
    is_outlier = np.abs(normalized_data - median) > threshold

    # Calculate metrics
    n_outliers = np.sum(is_outlier)
    outlier_ratio = n_outliers / len(data)

    # Prepare results
    result = {
        "median": median,
        "mad": mad,
        "outliers_mask": is_outlier,
        "normalized_data": normalized_data
    }

    metrics = {
        "n_outliers": n_outliers,
        "outlier_ratio": outlier_ratio
    }

    params_used = {
        "normalization": normalization,
        "threshold_factor": threshold_factor
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
# data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])
# result = median_absolute_deviation_fit(data)

################################################################################
# hampel_filter
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for Hampel filter."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _normalize_data(
    data: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data using specified method or custom function."""
    if custom_func is not None:
        return custom_func(data)

    data = np.array(data, dtype=np.float64)
    if method == "none":
        return data
    elif method == "standard":
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    elif method == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return (data - median) / (1.4826 * mad + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _calculate_thresholds(
    normalized_data: np.ndarray,
    k_factor: float = 3.0
) -> Dict[str, np.ndarray]:
    """Calculate thresholds for Hampel filter."""
    median = np.median(normalized_data)
    mad = np.median(np.abs(normalized_data - median))
    lower_threshold = median - k_factor * mad
    upper_threshold = median + k_factor * mad
    return {
        "median": median,
        "mad": mad,
        "lower_threshold": lower_threshold,
        "upper_threshold": upper_threshold
    }

def _identify_outliers(
    normalized_data: np.ndarray,
    thresholds: Dict[str, float]
) -> np.ndarray:
    """Identify outliers based on Hampel filter thresholds."""
    return (normalized_data < thresholds["lower_threshold"]) | (
        normalized_data > thresholds["upper_threshold"])

def _denormalize_outliers(
    outliers: np.ndarray,
    original_data: np.ndarray,
    normalized_data: np.ndarray
) -> np.ndarray:
    """Denormalize outlier flags to original data scale."""
    # This is a placeholder - actual denormalization depends on normalization method
    return outliers

def hampel_filter_fit(
    data: np.ndarray,
    k_factor: float = 3.0,
    normalization_method: str = "standard",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_thresholds: Optional[Callable[[np.ndarray, float], Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Apply Hampel filter to identify outliers in a dataset.

    Parameters:
    -----------
    data : np.ndarray
        Input data array (1-dimensional)
    k_factor : float, optional
        Threshold multiplier (default: 3.0)
    normalization_method : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_normalization : callable, optional
        Custom normalization function
    custom_thresholds : callable, optional
        Custom threshold calculation function

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': array of outlier flags (True for outliers)
        - 'metrics': dictionary with calculated metrics
        - 'params_used': parameters used in the computation
        - 'warnings': any warnings generated during processing

    Example:
    --------
    >>> data = np.array([1, 2, 3, 4, 5, 100])
    >>> result = hampel_filter_fit(data)
    """
    # Validate inputs
    _validate_inputs(data)

    # Initialize warnings list
    warnings = []

    # Normalize data
    try:
        normalized_data = _normalize_data(
            data,
            method=normalization_method,
            custom_func=custom_normalization
        )
    except Exception as e:
        warnings.append(f"Normalization failed: {str(e)}")
        normalized_data = data  # Fallback to original data

    # Calculate thresholds
    try:
        if custom_thresholds is not None:
            thresholds = custom_thresholds(normalized_data, k_factor)
        else:
            thresholds = _calculate_thresholds(normalized_data, k_factor)
    except Exception as e:
        warnings.append(f"Threshold calculation failed: {str(e)}")
        thresholds = _calculate_thresholds(normalized_data, k_factor)  # Fallback

    # Identify outliers
    try:
        outliers = _identify_outliers(normalized_data, thresholds)
    except Exception as e:
        warnings.append(f"Outlier identification failed: {str(e)}")
        outliers = np.zeros_like(data, dtype=bool)  # Fallback

    # Denormalize outliers
    try:
        denormalized_outliers = _denormalize_outliers(
            outliers, data, normalized_data
        )
    except Exception as e:
        warnings.append(f"Denormalization failed: {str(e)}")
        denormalized_outliers = outliers  # Fallback

    # Prepare metrics
    metrics = {
        "median": thresholds.get("median", np.nan),
        "mad": thresholds.get("mad", np.nan),
        "outlier_count": np.sum(denormalized_outliers),
        "total_points": len(data)
    }

    # Prepare output
    result = {
        "result": denormalized_outliers,
        "metrics": metrics,
        "params_used": {
            "k_factor": k_factor,
            "normalization_method": normalization_method,
            "custom_normalization_used": custom_normalization is not None,
            "custom_thresholds_used": custom_thresholds is not None
        },
        "warnings": warnings if warnings else None
    }

    return result

################################################################################
# trimmed_mean
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_input(data: np.ndarray) -> None:
    """Validate input data for trimmed mean calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")

def calculate_trimmed_indices(data: np.ndarray, trim_percentage: float) -> tuple:
    """Calculate indices for trimmed mean based on specified percentage."""
    n = len(data)
    k = int(n * trim_percentage / 100)
    sorted_data = np.sort(data)
    return slice(k, n - k), sorted_data

def compute_trimmed_mean(data: np.ndarray,
                        trim_percentage: float = 10.0) -> Dict[str, Any]:
    """Compute the trimmed mean of a dataset.

    Parameters:
    -----------
    data : np.ndarray
        Input data array (1D)
    trim_percentage : float, optional
        Percentage of data to trim from each end (default: 10.0)

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': computed trimmed mean
        - 'metrics': dictionary of metrics (currently empty)
        - 'params_used': parameters used in computation
        - 'warnings': list of warnings (empty if no issues)
    """
    # Validate input
    validate_input(data)

    # Check trim percentage validity
    if not 0 <= trim_percentage < 50:
        raise ValueError("trim_percentage must be between 0 and 50")

    # Calculate trimmed mean
    slice_obj, sorted_data = calculate_trimmed_indices(data, trim_percentage)
    trimmed_mean = np.mean(sorted_data[slice_obj])

    # Prepare output
    result_dict: Dict[str, Any] = {
        'result': trimmed_mean,
        'metrics': {},
        'params_used': {
            'trim_percentage': trim_percentage
        },
        'warnings': []
    }

    return result_dict

# Example usage:
"""
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = compute_trimmed_mean(data, trim_percentage=20)
"""

################################################################################
# winsorization
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(data: np.ndarray) -> None:
    """Validate input data for winsorization."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def calculate_percentiles(data: np.ndarray, lower_bound: float = 0.05, upper_bound: float = 0.95) -> tuple:
    """Calculate the percentiles for winsorization."""
    lower_percentile = np.percentile(data, lower_bound * 100)
    upper_percentile = np.percentile(data, upper_bound * 100)
    return lower_percentile, upper_percentile

def winsorize_data(data: np.ndarray, lower_bound: float = 0.05, upper_bound: float = 0.95) -> np.ndarray:
    """Apply winsorization to the data."""
    lower_percentile, upper_percentile = calculate_percentiles(data, lower_bound, upper_bound)
    winsorized_data = np.clip(data, lower_percentile, upper_percentile)
    return winsorized_data

def calculate_metrics(original_data: np.ndarray, winsorized_data: np.ndarray,
                      metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None) -> Dict[str, float]:
    """Calculate metrics for winsorization."""
    metrics = {}

    if metric_func is None:
        def default_metric(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        metric_func = default_metric

    metrics['mse'] = metric_func(original_data, winsorized_data)

    return metrics

def winsorization_fit(data: np.ndarray,
                     lower_bound: float = 0.05,
                     upper_bound: float = 0.95,
                     metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None) -> Dict:
    """
    Apply winsorization to the data and return results, metrics, and parameters used.

    Parameters:
    -----------
    data : np.ndarray
        Input data to be winsorized.
    lower_bound : float, optional
        Lower bound percentile for winsorization (default: 0.05).
    upper_bound : float, optional
        Upper bound percentile for winsorization (default: 0.95).
    metric_func : Callable[[np.ndarray, np.ndarray], float], optional
        Custom metric function to evaluate the winsorization (default: None).

    Returns:
    --------
    dict
        A dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate input data
    validate_input(data)

    # Calculate percentiles and winsorize data
    lower_percentile, upper_percentile = calculate_percentiles(data, lower_bound, upper_bound)
    winsorized_data = winsorize_data(data, lower_bound, upper_bound)

    # Calculate metrics
    metrics = calculate_metrics(data, winsorized_data, metric_func)

    # Prepare the output dictionary
    result = {
        'result': winsorized_data,
        'metrics': metrics,
        'params_used': {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        },
        'warnings': []
    }

    return result

# Example usage:
# data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# result = winsorization_fit(data, lower_bound=0.1, upper_bound=0.9)
# print(result)

################################################################################
# z_score_modified
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using specified method."""
    if method == "none":
        return data
    elif method == "standard":
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    elif method == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    elif method == "robust":
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _calculate_z_scores(data: np.ndarray) -> np.ndarray:
    """Calculate z-scores for the data."""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def _identify_outliers(z_scores: np.ndarray, threshold: float) -> np.ndarray:
    """Identify outliers based on z-score threshold."""
    return np.abs(z_scores) > threshold

def _remove_outliers(data: np.ndarray, outliers_mask: np.ndarray) -> np.ndarray:
    """Remove outliers from the data."""
    return data[~outliers_mask]

def _calculate_metrics(original_data: np.ndarray, cleaned_data: np.ndarray,
                      metric_func: Callable) -> Dict[str, float]:
    """Calculate metrics for the cleaned data."""
    if metric_func.__name__ == "mse":
        return {"mse": np.mean((original_data - cleaned_data) ** 2)}
    elif metric_func.__name__ == "mae":
        return {"mae": np.mean(np.abs(original_data - cleaned_data))}
    elif metric_func.__name__ == "r2":
        ss_res = np.sum((original_data - cleaned_data) ** 2)
        ss_tot = np.sum((original_data - np.mean(original_data)) ** 2)
        return {"r2": 1 - (ss_res / ss_tot)}
    else:
        return {metric_func.__name__: metric_func(original_data, cleaned_data)}

def z_score_modified_fit(
    data: np.ndarray,
    normalization_method: str = "standard",
    threshold: float = 3.0,
    metric_func: Callable[[np.ndarray, np.ndarray], float] = None
) -> Dict[str, Any]:
    """
    Identify and remove outliers using the modified z-score method.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    normalization_method : str, optional
        Normalization method to use (default: "standard").
    threshold : float, optional
        Threshold for identifying outliers (default: 3.0).
    metric_func : Callable, optional
        Custom metric function to evaluate the cleaned data.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_inputs(data)

    normalized_data = _normalize_data(data, normalization_method)
    z_scores = _calculate_z_scores(normalized_data)
    outliers_mask = _identify_outliers(z_scores, threshold)
    cleaned_data = _remove_outliers(data, outliers_mask)

    metrics = {}
    if metric_func is not None:
        metrics.update(_calculate_metrics(data, cleaned_data, metric_func))

    result = {
        "result": cleaned_data,
        "metrics": metrics,
        "params_used": {
            "normalization_method": normalization_method,
            "threshold": threshold
        },
        "warnings": []
    }

    if np.any(outliers_mask):
        result["warnings"].append(f"Removed {np.sum(outliers_mask)} outliers")

    return result

# Example usage:
# data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])
# result = z_score_modified_fit(data)

################################################################################
# percentile_capping
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    X: np.ndarray,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array")
    if len(X.shape) != 2:
        raise ValueError("Input X must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input X contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input X contains infinite values")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Invalid percentile values")

def _calculate_percentiles(
    X: np.ndarray,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0
) -> tuple:
    """Calculate lower and upper percentiles for each feature."""
    lower = np.percentile(X, lower_percentile, axis=0)
    upper = np.percentile(X, upper_percentile, axis=0)
    return lower, upper

def _apply_capping(
    X: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> np.ndarray:
    """Apply percentile capping to the data."""
    X_capped = np.copy(X)
    X_capped[X < lower] = lower[X < lower]
    X_capped[X > upper] = upper[X > upper]
    return X_capped

def _calculate_metrics(
    X: np.ndarray,
    X_capped: np.ndarray,
    metric_func: Callable = None
) -> Dict[str, float]:
    """Calculate metrics before and after capping."""
    if metric_func is None:
        def default_metric(y_true, y_pred):
            return {
                'mse': np.mean((y_true - y_pred)**2),
                'mae': np.mean(np.abs(y_true - y_pred))
            }
        metric_func = default_metric

    metrics = {}
    for i in range(X.shape[1]):
        col_true = X[:, i]
        col_pred = X_capped[:, i]
        metrics[f'feature_{i}'] = metric_func(col_true, col_pred)

    return {'metrics': metrics}

def percentile_capping_fit(
    X: np.ndarray,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    metric_func: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict, str]]:
    """
    Apply percentile capping to outlier treatment.

    Parameters:
    -----------
    X : np.ndarray
        Input data array of shape (n_samples, n_features)
    lower_percentile : float, optional
        Lower percentile threshold (default: 5.0)
    upper_percentile : float, optional
        Upper percentile threshold (default: 95.0)
    metric_func : Callable, optional
        Custom metric function (default: None)

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': Capped data array
        - 'metrics': Metrics before/after capping
        - 'params_used': Parameters used in the process
        - 'warnings': Any warnings generated

    Example:
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = percentile_capping_fit(X)
    """
    _validate_inputs(X, lower_percentile, upper_percentile)

    warnings = []
    if X.shape[0] < 10:
        warnings.append("Warning: Small sample size may affect percentile estimates")

    lower, upper = _calculate_percentiles(X, lower_percentile, upper_percentile)
    X_capped = _apply_capping(X, lower, upper)

    metrics = _calculate_metrics(X, X_capped, metric_func)

    return {
        'result': X_capped,
        'metrics': metrics['metrics'],
        'params_used': {
            'lower_percentile': lower_percentile,
            'upper_percentile': upper_percentile
        },
        'warnings': warnings if warnings else None
    }

################################################################################
# robust_scaling
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def robust_scaling_fit(
    X: np.ndarray,
    normalization: str = 'robust',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric_func: Optional[Callable] = None,
    custom_distance_func: Optional[Callable] = None
) -> Dict:
    """
    Fit robust scaling parameters to handle outliers in the data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    normalization : str, optional
        Type of normalization to apply. Options: 'none', 'standard', 'minmax', 'robust'.
    metric : str or callable, optional
        Metric to evaluate scaling performance. Options: 'mse', 'mae', 'r2', custom callable.
    distance : str or callable, optional
        Distance metric for outlier detection. Options: 'euclidean', 'manhattan', 'cosine',
        'minkowski', custom callable.
    solver : str, optional
        Solver to use for parameter estimation. Options: 'closed_form', 'gradient_descent',
        'newton', 'coordinate_descent'.
    regularization : str, optional
        Regularization type. Options: 'none', 'l1', 'l2', 'elasticnet'.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric_func : callable, optional
        Custom metric function if not using built-in metrics.
    custom_distance_func : callable, optional
        Custom distance function if not using built-in distances.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Scaled data or parameters.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings generated.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> result = robust_scaling_fit(X)
    """
    # Validate inputs
    _validate_inputs(X, normalization, metric, distance, solver, regularization)

    # Initialize parameters
    params = {
        'normalization': normalization,
        'metric': metric,
        'distance': distance,
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Choose normalization function
    norm_func = _get_normalization_function(normalization)

    # Choose metric function
    metric_func = _get_metric_function(metric, custom_metric_func)

    # Choose distance function
    distance_func = _get_distance_function(distance, custom_distance_func)

    # Choose solver function
    solver_func = _get_solver_function(solver, regularization)

    # Fit the model
    result, metrics = _fit_robust_scaling(
        X, norm_func, metric_func, distance_func, solver_func,
        tol, max_iter
    )

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

def _validate_inputs(
    X: np.ndarray,
    normalization: str,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str,
    regularization: Optional[str]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values.")
    if np.any(np.isinf(X)):
        raise ValueError("X contains infinite values.")

    valid_normalizations = ['none', 'standard', 'minmax', 'robust']
    if normalization not in valid_normalizations:
        raise ValueError(f"normalization must be one of {valid_normalizations}.")

    valid_solvers = ['closed_form', 'gradient_descent', 'newton', 'coordinate_descent']
    if solver not in valid_solvers:
        raise ValueError(f"solver must be one of {valid_solvers}.")

    valid_regularizations = [None, 'none', 'l1', 'l2', 'elasticnet']
    if regularization not in valid_regularizations:
        raise ValueError(f"regularization must be one of {valid_regularizations}.")

def _get_normalization_function(normalization: str) -> Callable:
    """Return the appropriate normalization function."""
    if normalization == 'none':
        return lambda X: X
    elif normalization == 'standard':
        return _standard_normalization
    elif normalization == 'minmax':
        return _minmax_normalization
    elif normalization == 'robust':
        return _robust_normalization
    else:
        raise ValueError("Invalid normalization type.")

def _standard_normalization(X: np.ndarray) -> np.ndarray:
    """Standard normalization (mean=0, std=1)."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8)

def _minmax_normalization(X: np.ndarray) -> np.ndarray:
    """Min-max normalization (range [0, 1])."""
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    return (X - min_val) / (max_val - min_val + 1e-8)

def _robust_normalization(X: np.ndarray) -> np.ndarray:
    """Robust normalization (median and IQR)."""
    median = np.median(X, axis=0)
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
    return (X - median) / (iqr + 1e-8)

def _get_metric_function(
    metric: Union[str, Callable],
    custom_metric_func: Optional[Callable]
) -> Callable:
    """Return the appropriate metric function."""
    if callable(metric):
        return metric
    elif custom_metric_func is not None:
        return custom_metric_func
    elif metric == 'mse':
        return _mse
    elif metric == 'mae':
        return _mae
    elif metric == 'r2':
        return _r2
    else:
        raise ValueError("Invalid metric type.")

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def _get_distance_function(
    distance: Union[str, Callable],
    custom_distance_func: Optional[Callable]
) -> Callable:
    """Return the appropriate distance function."""
    if callable(distance):
        return distance
    elif custom_distance_func is not None:
        return custom_distance_func
    elif distance == 'euclidean':
        return _euclidean_distance
    elif distance == 'manhattan':
        return _manhattan_distance
    elif distance == 'cosine':
        return _cosine_distance
    elif distance == 'minkowski':
        return _minkowski_distance
    else:
        raise ValueError("Invalid distance type.")

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _minkowski_distance(a: np.ndarray, b: np.ndarray, p: float = 3) -> float:
    """Minkowski distance."""
    return np.sum(np.abs(a - b) ** p) ** (1 / p)

def _get_solver_function(solver: str, regularization: Optional[str]) -> Callable:
    """Return the appropriate solver function."""
    if solver == 'closed_form':
        return _closed_form_solver
    elif solver == 'gradient_descent':
        return lambda X, *args: _gradient_descent_solver(X, regularization, *args)
    elif solver == 'newton':
        return lambda X, *args: _newton_solver(X, regularization, *args)
    elif solver == 'coordinate_descent':
        return lambda X, *args: _coordinate_descent_solver(X, regularization, *args)
    else:
        raise ValueError("Invalid solver type.")

def _closed_form_solver(X: np.ndarray) -> np.ndarray:
    """Closed form solver for parameter estimation."""
    return np.linalg.pinv(X.T @ X) @ (X.T @ np.ones(X.shape[0]))

def _gradient_descent_solver(
    X: np.ndarray,
    regularization: Optional[str],
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Gradient descent solver with optional regularization."""
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = X.T @ (X @ theta - np.ones(n_samples)) / n_samples
        if regularization == 'l1':
            gradient += np.sign(theta)
        elif regularization == 'l2':
            gradient += 2 * theta
        elif regularization == 'elasticnet':
            gradient += np.sign(theta) + 2 * theta

        new_theta = theta - learning_rate * gradient
        if np.linalg.norm(new_theta - theta) < tol:
            break
        theta = new_theta

    return theta

def _newton_solver(
    X: np.ndarray,
    regularization: Optional[str],
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Newton's method solver with optional regularization."""
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = X.T @ (X @ theta - np.ones(n_samples)) / n_samples
        hessian = X.T @ X / n_samples

        if regularization == 'l1':
            hessian += np.diag(np.sign(theta))
        elif regularization == 'l2':
            hessian += 2 * np.eye(n_features)
        elif regularization == 'elasticnet':
            hessian += np.diag(np.sign(theta)) + 2 * np.eye(n_features)

        new_theta = theta - np.linalg.pinv(hessian) @ gradient
        if np.linalg.norm(new_theta - theta) < tol:
            break
        theta = new_theta

    return theta

def _coordinate_descent_solver(
    X: np.ndarray,
    regularization: Optional[str],
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Coordinate descent solver with optional regularization."""
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = np.ones(n_samples) - (X @ theta) + theta[j] * X_j
            if regularization == 'l1':
                theta[j] = np.sign(np.sum(X_j * residuals)) * np.maximum(
                    0, np.abs(np.sum(X_j * residuals)) - 1
                ) / (np.sum(X_j ** 2) + 1e-8)
            elif regularization == 'l2':
                theta[j] = np.sum(X_j * residuals) / (np.sum(X_j ** 2) + 2)
            elif regularization == 'elasticnet':
                theta[j] = np.sign(np.sum(X_j * residuals)) * np.maximum(
                    0, np.abs(np.sum(X_j * residuals)) - 1
                ) / (np.sum(X_j ** 2) + 2)
            else:
                theta[j] = np.sum(X_j * residuals) / (np.sum(X_j ** 2) + 1e-8)

        if np.linalg.norm(theta - np.zeros(n_features)) < tol:
            break

    return theta

def _fit_robust_scaling(
    X: np.ndarray,
    norm_func: Callable,
    metric_func: Callable,
    distance_func: Callable,
    solver_func: Callable,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> tuple:
    """Fit robust scaling parameters."""
    X_norm = norm_func(X)
    params = solver_func(X_norm, tol, max_iter)

    # Compute metrics
    y_pred = X @ params
    metrics = {
        'metric_value': metric_func(np.ones(X.shape[0]), y_pred),
        'distance_metric': distance_func(params, np.zeros_like(params))
    }

    return params, metrics
