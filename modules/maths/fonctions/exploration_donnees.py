"""
Quantix – Module exploration_donnees
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# statistiques_descriptives
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values.")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using specified method."""
    if method == "none":
        return data
    elif method == "standard":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_metrics(data: np.ndarray, metrics: Dict[str, Callable]) -> Dict[str, float]:
    """Compute specified metrics."""
    results = {}
    for name, func in metrics.items():
        if callable(func):
            results[name] = func(data)
    return results

def statistiques_descriptives_fit(
    data: np.ndarray,
    normalization_method: str = "none",
    metrics: Optional[Dict[str, Union[str, Callable]]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute descriptive statistics for the given data.

    Parameters:
    - data: Input data as a 2D numpy array.
    - normalization_method: Normalization method ('none', 'standard', 'minmax', 'robust').
    - metrics: Dictionary of metrics to compute. Keys are metric names, values are either
               predefined strings ('mean', 'std', 'min', 'max') or callable functions.
    - **kwargs: Additional keyword arguments for future extensions.

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_input(data)

    # Default metrics if none provided
    default_metrics = {
        "mean": lambda x: np.mean(x, axis=0),
        "std": lambda x: np.std(x, axis=0),
        "min": lambda x: np.min(x, axis=0),
        "max": lambda x: np.max(x, axis=0)
    }

    # Merge default and user-provided metrics
    if metrics is None:
        metrics = default_metrics
    else:
        for name, func in metrics.items():
            if isinstance(func, str):
                if func == "mean":
                    metrics[name] = lambda x: np.mean(x, axis=0)
                elif func == "std":
                    metrics[name] = lambda x: np.std(x, axis=0)
                elif func == "min":
                    metrics[name] = lambda x: np.min(x, axis=0)
                elif func == "max":
                    metrics[name] = lambda x: np.max(x, axis=0)
                else:
                    raise ValueError(f"Unknown metric string: {func}")

    # Normalize data
    normalized_data = _normalize_data(data, normalization_method)

    # Compute metrics
    computed_metrics = _compute_metrics(normalized_data, metrics)

    return {
        "result": normalized_data,
        "metrics": computed_metrics,
        "params_used": {
            "normalization_method": normalization_method
        },
        "warnings": []
    }

################################################################################
# visualisation_donnees
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def visualisation_donnees_fit(
    data: np.ndarray,
    features: Optional[np.ndarray] = None,
    target: Optional[np.ndarray] = None,
    normalization: str = 'standard',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Fonction principale pour la visualisation des données avec options paramétrables.

    Parameters:
    -----------
    data : np.ndarray
        Matrice des données d'entrée.
    features : Optional[np.ndarray], default=None
        Matrice des caractéristiques. Si None, utilise data.
    target : Optional[np.ndarray], default=None
        Vecteur cible. Si None, utilise la dernière colonne de data.
    normalization : str, default='standard'
        Type de normalisation: 'none', 'standard', 'minmax', 'robust'.
    metric : Union[str, Callable], default='mse'
        Métrique d'évaluation: 'mse', 'mae', 'r2', 'logloss' ou une fonction personnalisée.
    distance : Union[str, Callable], default='euclidean'
        Distance utilisée: 'euclidean', 'manhattan', 'cosine', 'minkowski' ou une fonction personnalisée.
    solver : str, default='closed_form'
        Solveur utilisé: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.
    regularization : Optional[str], default=None
        Type de régularisation: 'none', 'l1', 'l2', 'elasticnet'.
    tol : float, default=1e-4
        Tolérance pour la convergence.
    max_iter : int, default=1000
        Nombre maximal d'itérations.
    custom_metric : Optional[Callable], default=None
        Fonction personnalisée pour la métrique.
    custom_distance : Optional[Callable], default=None
        Fonction personnalisée pour la distance.

    Returns:
    --------
    Dict[str, Any]
        Dictionnaire contenant les résultats, métriques, paramètres utilisés et avertissements.
    """
    # Validation des entrées
    _validate_inputs(data, features, target)

    # Normalisation des données
    normalized_data = _apply_normalization(data, normalization)

    # Séparation des features et target
    X, y = _prepare_data(normalized_data, features, target)

    # Choix de la métrique
    metric_func = _get_metric(metric, custom_metric)

    # Choix de la distance
    distance_func = _get_distance(distance, custom_distance)

    # Choix du solveur
    solver_func = _get_solver(solver, X, y, distance_func, regularization, tol, max_iter)

    # Estimation des paramètres
    params = solver_func(X, y)

    # Calcul des métriques
    metrics = _compute_metrics(y, params, X, metric_func)

    # Retourne le dictionnaire structuré
    return {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

def _validate_inputs(data: np.ndarray, features: Optional[np.ndarray], target: Optional[np.ndarray]) -> None:
    """Validation des entrées."""
    if not isinstance(data, np.ndarray):
        raise TypeError("data doit être un tableau NumPy.")
    if features is not None and not isinstance(features, np.ndarray):
        raise TypeError("features doit être un tableau NumPy ou None.")
    if target is not None and not isinstance(target, np.ndarray):
        raise TypeError("target doit être un tableau NumPy ou None.")
    if features is not None and data.shape[0] != features.shape[0]:
        raise ValueError("data et features doivent avoir le même nombre de lignes.")
    if target is not None and data.shape[0] != target.shape[0]:
        raise ValueError("data et target doivent avoir le même nombre de lignes.")

def _apply_normalization(data: np.ndarray, normalization: str) -> np.ndarray:
    """Application de la normalisation choisie."""
    if normalization == 'none':
        return data
    elif normalization == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif normalization == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif normalization == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Normalisation '{normalization}' non reconnue.")

def _prepare_data(data: np.ndarray, features: Optional[np.ndarray], target: Optional[np.ndarray]) -> tuple:
    """Préparation des données (features et target)."""
    if features is None and target is None:
        return data[:, :-1], data[:, -1]
    elif features is not None and target is not None:
        return features, target
    else:
        raise ValueError("Soit les deux (features et target) doivent être fournis, soit aucun.")

def _get_metric(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Récupération de la fonction de métrique."""
    if custom_metric is not None:
        return custom_metric
    elif metric == 'mse':
        return _mean_squared_error
    elif metric == 'mae':
        return _mean_absolute_error
    elif metric == 'r2':
        return _r_squared
    elif metric == 'logloss':
        return _log_loss
    else:
        raise ValueError(f"Métrique '{metric}' non reconnue.")

def _get_distance(distance: Union[str, Callable], custom_distance: Optional[Callable]) -> Callable:
    """Récupération de la fonction de distance."""
    if custom_distance is not None:
        return custom_distance
    elif distance == 'euclidean':
        return _euclidean_distance
    elif distance == 'manhattan':
        return _manhattan_distance
    elif distance == 'cosine':
        return _cosine_distance
    elif distance == 'minkowski':
        return _minkowski_distance
    else:
        raise ValueError(f"Distance '{distance}' non reconnue.")

def _get_solver(solver: str, X: np.ndarray, y: np.ndarray, distance_func: Callable,
                regularization: Optional[str], tol: float, max_iter: int) -> Callable:
    """Récupération de la fonction de solveur."""
    if solver == 'closed_form':
        return lambda X, y: _closed_form_solution(X, y)
    elif solver == 'gradient_descent':
        return lambda X, y: _gradient_descent(X, y, distance_func, regularization, tol, max_iter)
    elif solver == 'newton':
        return lambda X, y: _newton_method(X, y, distance_func, regularization, tol, max_iter)
    elif solver == 'coordinate_descent':
        return lambda X, y: _coordinate_descent(X, y, distance_func, regularization, tol, max_iter)
    else:
        raise ValueError(f"Solveur '{solver}' non reconnu.")

def _compute_metrics(y_true: np.ndarray, params: np.ndarray, X: np.ndarray, metric_func: Callable) -> Dict[str, float]:
    """Calcul des métriques."""
    y_pred = X @ params
    return {"metric": metric_func(y_true, y_pred)}

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcul de l'erreur quadratique moyenne."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcul de l'erreur absolue moyenne."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcul du coefficient de détermination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcul de la log-loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calcul de la distance euclidienne."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calcul de la distance de Manhattan."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calcul de la distance cosinus."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _minkowski_distance(a: np.ndarray, b: np.ndarray, p: int = 3) -> float:
    """Calcul de la distance de Minkowski."""
    return np.sum(np.abs(a - b) ** p) ** (1 / p)

def _closed_form_solution(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Résolution en forme fermée."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _gradient_descent(X: np.ndarray, y: np.ndarray, distance_func: Callable,
                      regularization: Optional[str], tol: float, max_iter: int) -> np.ndarray:
    """Descente de gradient."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = (2 / n_samples) * X.T @ (X @ params - y)
        if regularization == 'l1':
            gradient += np.sign(params)
        elif regularization == 'l2':
            gradient += 2 * params
        params -= learning_rate * gradient

    return params

def _newton_method(X: np.ndarray, y: np.ndarray, distance_func: Callable,
                   regularization: Optional[str], tol: float, max_iter: int) -> np.ndarray:
    """Méthode de Newton."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = (2 / n_samples) * X.T @ (X @ params - y)
        hessian = (2 / n_samples) * X.T @ X
        if regularization == 'l1':
            hessian += np.diag(np.sign(params))
        elif regularization == 'l2':
            hessian += 2 * np.eye(n_features)
        params -= np.linalg.inv(hessian) @ gradient

    return params

def _coordinate_descent(X: np.ndarray, y: np.ndarray, distance_func: Callable,
                        regularization: Optional[str], tol: float, max_iter: int) -> np.ndarray:
    """Descente de coordonnées."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residual = y - np.dot(X, params) + params[j] * X_j
            if regularization == 'l1':
                params[j] = np.sign(np.dot(X_j, residual)) * np.maximum(
                    0, np.abs(np.dot(X_j, residual)) - learning_rate
                )
            elif regularization == 'l2':
                params[j] = np.dot(X_j, residual) / (np.dot(X_j, X_j) + learning_rate)
            else:
                params[j] = np.dot(X_j, residual) / np.dot(X_j, X_j)

    return params

################################################################################
# nettoyage_donnees
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def nettoyage_donnees_fit(
    X: np.ndarray,
    normalisation: str = 'standard',
    imputation_strategie: str = 'mean',
    seuil_nan: float = 0.5,
    custom_normalisation: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_imputation: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Fonction principale pour le nettoyage des données.

    Parameters
    ----------
    X : np.ndarray
        Matrice des données à nettoyer.
    normalisation : str, optional
        Stratégie de normalisation ('none', 'standard', 'minmax', 'robust').
    imputation_strategie : str, optional
        Stratégie d'imputation ('mean', 'median', 'mode').
    seuil_nan : float, optional
        Seuil de tolérance pour les NaN (0-1).
    custom_normalisation : Callable, optional
        Fonction personnalisée de normalisation.
    custom_imputation : Callable, optional
        Fonction personnalisée d'imputation.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        Dictionnaire contenant les résultats du nettoyage.
    """
    # Validation des entrées
    X = _valider_entrees(X)

    # Suppression des colonnes avec trop de NaN
    X = _supprimer_colonnes_nan(X, seuil_nan)

    # Imputation des NaN
    if custom_imputation is not None:
        X = custom_imputation(X)
    else:
        X = _imputer_nan(X, imputation_strategie)

    # Normalisation
    if custom_normalisation is not None:
        X = custom_normalisation(X)
    else:
        X = _normaliser_donnees(X, normalisation)

    # Retour des résultats
    return {
        "result": X,
        "metrics": {"nan_removed": np.isnan(X).sum()},
        "params_used": {
            "normalisation": normalisation,
            "imputation_strategie": imputation_strategie,
            "seuil_nan": seuil_nan
        },
        "warnings": _generer_alertes(X)
    }

def _valider_entrees(X: np.ndarray) -> np.ndarray:
    """Valide les entrées et vérifie la présence de NaN."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X doit être un tableau NumPy.")
    if X.ndim != 2:
        raise ValueError("X doit être une matrice 2D.")
    return X

def _supprimer_colonnes_nan(X: np.ndarray, seuil: float) -> np.ndarray:
    """Supprime les colonnes avec un pourcentage de NaN supérieur au seuil."""
    nan_counts = np.isnan(X).sum(axis=0) / X.shape[0]
    mask = nan_counts <= seuil
    return X[:, mask]

def _imputer_nan(X: np.ndarray, strategie: str) -> np.ndarray:
    """Impute les NaN selon la stratégie spécifiée."""
    X_imputed = X.copy()
    for i in range(X.shape[1]):
        col = X[:, i]
        if np.isnan(col).any():
            if strategie == 'mean':
                val = np.nanmean(col)
            elif strategie == 'median':
                val = np.nanmedian(col)
            elif strategie == 'mode':
                val = np.nanmax(np.unique(col[~np.isnan(col)]))
            else:
                raise ValueError(f"Stratégie d'imputation inconnue: {strategie}")
            X_imputed[:, i] = np.where(np.isnan(col), val, col)
    return X_imputed

def _normaliser_donnees(X: np.ndarray, normalisation: str) -> np.ndarray:
    """Normalise les données selon la méthode spécifiée."""
    X_norm = X.copy()
    if normalisation == 'standard':
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        X_norm = (X - mean) / std
    elif normalisation == 'minmax':
        min_val = np.nanmin(X, axis=0)
        max_val = np.nanmax(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val)
    elif normalisation == 'robust':
        median = np.nanmedian(X, axis=0)
        iqr = np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)
        X_norm = (X - median) / iqr
    elif normalisation != 'none':
        raise ValueError(f"Normalisation inconnue: {normalisation}")
    return X_norm

def _generer_alertes(X: np.ndarray) -> list:
    """Génère des alertes sur les données nettoyées."""
    alerts = []
    if np.isinf(X).any():
        alerts.append("Des valeurs infinies ont été détectées.")
    if np.isnan(X).any():
        alerts.append("Des NaN résiduels ont été détectés.")
    return alerts

################################################################################
# analyse_univariee
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for univariate analysis."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _normalize_data(
    data: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """Normalize data using specified method or custom function."""
    if custom_func is not None:
        return custom_func(data)

    data = data.copy()
    if method == "none":
        return data
    elif method == "standard":
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_metrics(
    data: np.ndarray,
    metrics: Union[str, List[str], Callable],
    params: Optional[Dict] = None
) -> Dict:
    """Compute specified metrics for univariate data."""
    if params is None:
        params = {}

    results = {}
    if callable(metrics):
        results["custom"] = metrics(data, **params)
    else:
        if isinstance(metrics, str):
            metrics = [metrics]

        for metric in metrics:
            if metric == "mean":
                results["mean"] = np.mean(data)
            elif metric == "median":
                results["median"] = np.median(data)
            elif metric == "std":
                results["std"] = np.std(data)
            elif metric == "var":
                results["variance"] = np.var(data)
            elif metric == "min":
                results["minimum"] = np.min(data)
            elif metric == "max":
                results["maximum"] = np.max(data)
            elif metric == "skewness":
                results["skewness"] = _compute_skewness(data)
            elif metric == "kurtosis":
                results["kurtosis"] = _compute_kurtosis(data)
            elif metric == "iqr":
                results["iqr"] = np.percentile(data, 75) - np.percentile(data, 25)
            else:
                raise ValueError(f"Unknown metric: {metric}")

    return results

def _compute_skewness(data: np.ndarray) -> float:
    """Compute skewness of the data."""
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    return (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)

def _compute_kurtosis(data: np.ndarray) -> float:
    """Compute kurtosis of the data."""
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((data - mean) / std) ** 4) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

def analyse_univariee_fit(
    data: np.ndarray,
    normalization: str = "standard",
    custom_normalization: Optional[Callable] = None,
    metrics: Union[str, List[str], Callable] = ["mean", "median", "std"],
    metric_params: Optional[Dict] = None,
    warnings: bool = True
) -> Dict:
    """
    Perform univariate analysis on input data.

    Parameters:
    - data: Input one-dimensional numpy array
    - normalization: Normalization method ("none", "standard", "minmax", "robust")
    - custom_normalization: Custom normalization function
    - metrics: Metrics to compute (list of strings, single string, or callable)
    - metric_params: Parameters for custom metrics function
    - warnings: Whether to include warnings in output

    Returns:
    Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate input
    _validate_input(data)

    # Initialize output dictionary
    result = {
        "result": None,
        "metrics": {},
        "params_used": {
            "normalization": normalization,
            "custom_normalization": custom_normalization is not None,
            "metrics": metrics if isinstance(metrics, (str, list)) else "custom",
        },
        "warnings": []
    }

    # Normalize data
    try:
        normalized_data = _normalize_data(data, normalization, custom_normalization)
    except Exception as e:
        if warnings:
            result["warnings"].append(f"Normalization failed: {str(e)}")
        normalized_data = data.copy()

    # Compute metrics
    try:
        result["metrics"] = _compute_metrics(normalized_data, metrics, metric_params)
    except Exception as e:
        if warnings:
            result["warnings"].append(f"Metrics computation failed: {str(e)}")

    # Store normalized data in result
    result["result"] = {
        "original_data": data,
        "normalized_data": normalized_data
    }

    return result

# Example usage:
# data = np.random.randn(100)
# analysis = analyse_univariee_fit(data, normalization="standard", metrics=["mean", "std"])

################################################################################
# analyse_multivariee
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def analyse_multivariee_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalization: str = 'standard',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Perform multivariate analysis on the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features).
    y : Optional[np.ndarray]
        Target values vector of shape (n_samples,) or None for unsupervised analysis.
    normalization : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : Union[str, Callable]
        Metric to evaluate performance: 'mse', 'mae', 'r2', 'logloss', or custom callable.
    distance : Union[str, Callable]
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'minkowski', or custom callable.
    solver : str
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : Optional[str]
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable]
        Custom metric function if not using built-in metrics.
    custom_distance : Optional[Callable]
        Custom distance function if not using built-in distances.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalization(X, normalization)

    # Initialize parameters
    params_used = {
        'normalization': normalization,
        'metric': metric if isinstance(metric, str) else 'custom',
        'distance': distance if isinstance(distance, str) else 'custom',
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Choose solver and compute results
    if y is not None:
        result = _solve_supervised(X_normalized, y, solver, regularization,
                                  tol, max_iter, metric, custom_metric)
    else:
        result = _solve_unsupervised(X_normalized, solver, tol, max_iter,
                                    distance, custom_distance)

    # Compute metrics
    metrics = _compute_metrics(result, X_normalized, y,
                              metric, custom_metric)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

    return output

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array")
    if y is not None and (not isinstance(y, np.ndarray) or y.ndim != 1):
        raise ValueError("y must be a 1D numpy array or None")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the data."""
    if method == 'none':
        return X
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _solve_supervised(
    X: np.ndarray,
    y: np.ndarray,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict[str, Any]:
    """Solve supervised multivariate problem."""
    if solver == 'closed_form':
        return _solve_closed_form(X, y, regularization)
    elif solver == 'gradient_descent':
        return _solve_gradient_descent(X, y, regularization,
                                      tol, max_iter, metric, custom_metric)
    elif solver == 'newton':
        return _solve_newton(X, y, regularization,
                            tol, max_iter, metric, custom_metric)
    elif solver == 'coordinate_descent':
        return _solve_coordinate_descent(X, y, regularization,
                                        tol, max_iter, metric, custom_metric)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _solve_unsupervised(
    X: np.ndarray,
    solver: str,
    tol: float,
    max_iter: int,
    distance: Union[str, Callable],
    custom_distance: Optional[Callable]
) -> Dict[str, Any]:
    """Solve unsupervised multivariate problem."""
    if solver == 'closed_form':
        return _solve_unsupervised_closed_form(X, distance, custom_distance)
    elif solver == 'gradient_descent':
        return _solve_unsupervised_gradient_descent(X, distance,
                                                   tol, max_iter, custom_distance)
    else:
        raise ValueError(f"Unknown solver for unsupervised problem: {solver}")

def _compute_metrics(
    result: Dict[str, Any],
    X: np.ndarray,
    y: Optional[np.ndarray],
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}
    if y is not None:
        y_pred = result.get('predictions', np.zeros_like(y))
        if isinstance(metric, str):
            if metric == 'mse':
                metrics['mse'] = np.mean((y - y_pred) ** 2)
            elif metric == 'mae':
                metrics['mae'] = np.mean(np.abs(y - y_pred))
            elif metric == 'r2':
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                metrics['r2'] = 1 - (ss_res / ss_tot)
            elif metric == 'logloss':
                metrics['logloss'] = -np.mean(y * np.log(y_pred + 1e-8) +
                                             (1 - y) * np.log(1 - y_pred + 1e-8))
        if custom_metric is not None:
            metrics['custom'] = custom_metric(y, y_pred)
    return metrics

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.random.rand(100)

result = analyse_multivariee_fit(
    X=X,
    y=y,
    normalization='standard',
    metric='mse',
    solver='gradient_descent'
)
"""

################################################################################
# correlation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def correlation_fit(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'pearson',
    normalize_X: Optional[str] = None,
    normalize_y: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Calculate correlation between features and target.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    method : str, optional
        Correlation method ('pearson', 'spearman', 'kendall')
    normalize_X : str, optional
        Normalization for features ('standard', 'minmax', 'robust')
    normalize_y : str, optional
        Normalization for target ('standard', 'minmax', 'robust')
    custom_metric : callable, optional
        Custom correlation function

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': correlation values
        - 'metrics': additional metrics if applicable
        - 'params_used': parameters used
        - 'warnings': any warnings

    Examples
    --------
    >>> X = np.random.rand(10, 5)
    >>> y = np.random.rand(10)
    >>> result = correlation_fit(X, y, method='pearson')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if requested
    X_norm = _normalize_data(X, normalize_X) if normalize_X else X.copy()
    y_norm = _normalize_data(y.reshape(-1, 1), normalize_y).flatten() if normalize_y else y.copy()

    # Calculate correlation
    if custom_metric is not None:
        correlations = np.array([custom_metric(X_norm[:, i], y_norm) for i in range(X.shape[1])])
    else:
        correlations = _calculate_correlation(X_norm, y_norm, method)

    # Prepare output
    return {
        'result': correlations,
        'metrics': {},
        'params_used': {
            'method': method,
            'normalize_X': normalize_X,
            'normalize_y': normalize_y
        },
        'warnings': _check_warnings(X_norm, y_norm)
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays contain infinite values")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _calculate_correlation(X: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    """Calculate correlation between features and target."""
    if method == 'pearson':
        return np.corrcoef(X, y, rowvar=False)[-1, :-1]
    elif method == 'spearman':
        return np.corrcoef(_rank_data(X), _rank_data(y.reshape(-1, 1)), rowvar=False)[-1, :-1].flatten()
    elif method == 'kendall':
        return np.array([_kendall_tau(X[:, i], y) for i in range(X.shape[1])])
    else:
        raise ValueError(f"Unknown correlation method: {method}")

def _rank_data(data: np.ndarray) -> np.ndarray:
    """Convert data to ranks."""
    return np.argsort(np.argsort(data, axis=0), axis=0)

def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Kendall's tau correlation."""
    n = len(x)
    concordant = discordant = 0

    for i in range(n):
        for j in range(i+1, n):
            if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                concordant += 1
            elif (x[i] - x[j]) * (y[i] - y[j]) < 0:
                discordant += 1

    return (concordant - discordant) / ((n * (n-1)) / 2)

def _check_warnings(X: np.ndarray, y: np.ndarray) -> list:
    """Check for potential issues in the data."""
    warnings = []

    if np.any(np.std(X, axis=0) == 0):
        warnings.append("Some features have zero variance")
    if np.std(y) == 0:
        warnings.append("Target has zero variance")

    return warnings

################################################################################
# distribution_donnees
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def distribution_donnees_fit(
    data: np.ndarray,
    *,
    normalization: str = 'none',
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
    Estimate the distribution of data with configurable parameters.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric to evaluate the fit ('mse', 'mae', 'r2', etc.)
    distance : str or callable, optional
        Distance metric for clustering ('euclidean', 'manhattan', etc.)
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', etc.)
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    custom_metric : callable, optional
        Custom metric function
    custom_distance : callable, optional
        Custom distance function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.random.rand(100, 5)
    >>> result = distribution_donnees_fit(data, normalization='standard', metric='mse')
    """
    # Validate inputs
    _validate_inputs(data, normalization)

    # Normalize data
    normalized_data = _apply_normalization(data, normalization)

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

    # Fit distribution based on solver choice
    if solver == 'closed_form':
        result = _fit_closed_form(normalized_data, metric, regularization)
    elif solver == 'gradient_descent':
        result = _fit_gradient_descent(normalized_data, metric, distance,
                                      regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, result, metric,
                                custom_metric=custom_metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

def _validate_inputs(data: np.ndarray, normalization: str) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

    valid_normalizations = ['none', 'standard', 'minmax', 'robust']
    if normalization not in valid_normalizations:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the data."""
    if method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _fit_closed_form(data: np.ndarray, metric: Union[str, Callable], regularization: Optional[str]) -> Dict:
    """Fit distribution using closed-form solution."""
    # Placeholder for actual implementation
    return {
        'parameters': np.zeros(data.shape[1]),
        'converged': True
    }

def _fit_gradient_descent(
    data: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict:
    """Fit distribution using gradient descent."""
    # Placeholder for actual implementation
    return {
        'parameters': np.zeros(data.shape[1]),
        'converged': True,
        'iterations': 0
    }

def _calculate_metrics(
    data: np.ndarray,
    result: Dict,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Calculate evaluation metrics."""
    if custom_metric is not None:
        return {'custom': custom_metric(data, result)}

    if metric == 'mse':
        return {'mse': np.mean((data - result['parameters'])**2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(data - result['parameters']))}
    elif metric == 'r2':
        ss_res = np.sum((data - result['parameters'])**2)
        ss_tot = np.sum((data - np.mean(data, axis=0))**2)
        return {'r2': 1 - ss_res / ss_tot}
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# outliers
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def outliers_fit(
    data: np.ndarray,
    method: str = 'zscore',
    threshold: float = 3.0,
    normalize: Optional[str] = None,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = np.linalg.norm,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Detect outliers in a dataset using various statistical methods.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features).
    method : str, optional
        Method to use for outlier detection ('zscore', 'iqr', 'mahalanobis', 'custom').
    threshold : float, optional
        Threshold value for outlier detection.
    normalize : str or None, optional
        Normalization method ('standard', 'minmax', 'robust').
    distance_metric : Callable, optional
        Distance metric function for custom methods.
    custom_metric : Callable or None, optional
        Custom outlier detection function.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data)

    # Normalize data if specified
    normalized_data = _normalize_data(data, normalize) if normalize else data

    # Detect outliers based on method
    if method == 'zscore':
        result = _detect_zscore_outliers(normalized_data, threshold)
    elif method == 'iqr':
        result = _detect_iqr_outliers(normalized_data, threshold)
    elif method == 'mahalanobis':
        result = _detect_mahalanobis_outliers(normalized_data, threshold)
    elif method == 'custom':
        if custom_metric is None:
            raise ValueError("Custom metric function must be provided for 'custom' method.")
        result = _detect_custom_outliers(normalized_data, custom_metric, threshold)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate metrics
    metrics = _calculate_metrics(data, result['outliers'])

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'method': method,
            'threshold': threshold,
            'normalize': normalize,
            'distance_metric': distance_metric.__name__ if callable(distance_metric) else str(distance_metric),
            'custom_metric': custom_metric.__name__ if callable(custom_metric) else str(custom_metric)
        },
        'warnings': []
    }

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values.")

def _normalize_data(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _detect_zscore_outliers(
    data: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """Detect outliers using Z-score method."""
    z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
    outliers = z_scores > threshold
    return {
        'outliers': outliers,
        'statistics': {'z_scores': z_scores}
    }

def _detect_iqr_outliers(
    data: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """Detect outliers using IQR method."""
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outliers = (data < lower_bound) | (data > upper_bound)
    return {
        'outliers': outliers,
        'statistics': {'lower_bound': lower_bound, 'upper_bound': upper_bound}
    }

def _detect_mahalanobis_outliers(
    data: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """Detect outliers using Mahalanobis distance."""
    cov_matrix = np.cov(data, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mean_data = np.mean(data, axis=0)
    mahalanobis_distances = np.array([np.sqrt((x - mean_data).T @ inv_cov_matrix @ (x - mean_data)) for x in data])
    outliers = mahalanobis_distances > threshold
    return {
        'outliers': outliers,
        'statistics': {'mahalanobis_distances': mahalanobis_distances}
    }

def _detect_custom_outliers(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    threshold: float
) -> Dict[str, Any]:
    """Detect outliers using custom metric function."""
    # Example implementation - adjust as needed
    central_tendency = np.median(data, axis=0)
    distances = np.array([metric_func(x, central_tendency) for x in data])
    outliers = distances > threshold
    return {
        'outliers': outliers,
        'statistics': {'distances': distances}
    }

def _calculate_metrics(
    data: np.ndarray,
    outliers: np.ndarray
) -> Dict[str, float]:
    """Calculate metrics for outlier detection."""
    n_outliers = np.sum(outliers)
    total_samples = data.shape[0]
    outlier_percentage = (n_outliers / total_samples) * 100
    return {
        'n_outliers': int(n_outliers),
        'outlier_percentage': float(outlier_percentage)
    }

################################################################################
# missing_values
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def missing_values_fit(
    data: np.ndarray,
    imputation_method: str = 'mean',
    normalization: Optional[str] = None,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = np.linalg.norm,
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fit a missing values imputation model.

    Parameters:
    -----------
    data : np.ndarray
        Input data array with missing values (represented as NaN).
    imputation_method : str, optional
        Method for imputing missing values ('mean', 'median', 'knn').
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    distance_metric : Callable, optional
        Distance metric function for KNN imputation.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent').
    regularization : str, optional
        Regularization method ('none', 'l1', 'l2', 'elasticnet').
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    custom_metric : Callable, optional
        Custom metric function for evaluation.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data)

    # Normalize data if specified
    normalized_data, norm_params = _apply_normalization(data, normalization)

    # Impute missing values
    imputed_data = _impute_missing_values(normalized_data, imputation_method, distance_metric, **kwargs)

    # Calculate metrics
    metrics = _calculate_metrics(imputed_data, data, custom_metric)

    # Prepare output
    result = {
        'result': imputed_data,
        'metrics': metrics,
        'params_used': {
            'imputation_method': imputation_method,
            'normalization': normalization,
            'distance_metric': distance_metric.__name__ if hasattr(distance_metric, '__name__') else 'custom',
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

    return result

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if np.isnan(data).any():
        warnings.warn("Input data contains NaN values.")

def _apply_normalization(
    data: np.ndarray,
    method: Optional[str]
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Apply normalization to the data."""
    if method is None or method == 'none':
        return data, {}

    norm_params = {}
    if method == 'standard':
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        normalized_data = (data - mean) / std
        norm_params['mean'] = mean
        norm_params['std'] = std
    elif method == 'minmax':
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
        norm_params['min'] = min_val
        norm_params['max'] = max_val
    elif method == 'robust':
        median = np.nanmedian(data, axis=0)
        iqr = np.nanpercentile(data, 75, axis=0) - np.nanpercentile(data, 25, axis=0)
        normalized_data = (data - median) / (iqr + 1e-8)
        norm_params['median'] = median
        norm_params['iqr'] = iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized_data, norm_params

def _impute_missing_values(
    data: np.ndarray,
    method: str,
    distance_metric: Callable[[np.ndarray, np.ndarray], float],
    **kwargs
) -> np.ndarray:
    """Impute missing values using the specified method."""
    imputed_data = data.copy()
    mask = np.isnan(imputed_data)

    if method == 'mean':
        for i in range(imputed_data.shape[1]):
            imputed_data[mask[:, i], i] = np.nanmean(imputed_data[:, i])
    elif method == 'median':
        for i in range(imputed_data.shape[1]):
            imputed_data[mask[:, i], i] = np.nanmedian(imputed_data[:, i])
    elif method == 'knn':
        for i in range(imputed_data.shape[1]):
            if np.any(mask[:, i]):
                non_missing = imputed_data[~mask[:, i], :]
                missing_indices = np.where(mask[:, i])[0]
                for idx in missing_indices:
                    distances = [distance_metric(imputed_data[idx, :], row) for row in non_missing]
                    nearest_neighbor = np.argmin(distances)
                    imputed_data[idx, i] = non_missing[nearest_neighbor, i]
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    return imputed_data

def _calculate_metrics(
    imputed_data: np.ndarray,
    original_data: np.ndarray,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate metrics for the imputed data."""
    mask = np.isnan(original_data)
    original_values = original_data[~mask]
    imputed_values = imputed_data[~mask]

    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(original_values, imputed_values)

    return metrics

################################################################################
# scaling_normalisation
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

def standard_scaling(X: np.ndarray) -> np.ndarray:
    """Standard scaling (z-score normalization)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + 1e-8)

def minmax_scaling(X: np.ndarray, feature_range: tuple = (0, 1)) -> np.ndarray:
    """Min-max scaling."""
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    return (X - min_val) / (max_val - min_val + 1e-8) * (feature_range[1] - feature_range[0]) + feature_range[0]

def robust_scaling(X: np.ndarray) -> np.ndarray:
    """Robust scaling using median and IQR."""
    median = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    iqr = q75 - q25
    return (X - median) / (iqr + 1e-8)

def compute_metrics(X: np.ndarray, X_scaled: np.ndarray, metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute metrics for scaled data."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(X, X_scaled)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def scaling_normalisation_fit(
    X: np.ndarray,
    method: str = 'standard',
    feature_range: tuple = (0, 1),
    metric_funcs: Optional[Dict[str, Callable]] = None,
    custom_scaler: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Fit scaling/normalization to data.

    Parameters:
    - X: Input data (2D numpy array)
    - method: Scaling method ('none', 'standard', 'minmax', 'robust')
    - feature_range: Range for min-max scaling
    - metric_funcs: Dictionary of metric functions to compute
    - custom_scaler: Custom scaling function

    Returns:
    Dictionary containing results, metrics, parameters used, and warnings
    """
    validate_input(X)

    # Initialize default metric functions if not provided
    if metric_funcs is None:
        metric_funcs = {
            'mse': lambda X, X_scaled: np.mean((X - X_scaled) ** 2),
            'mae': lambda X, X_scaled: np.mean(np.abs(X - X_scaled))
        }

    # Apply scaling
    if custom_scaler is not None:
        X_scaled = custom_scaler(X)
    elif method == 'standard':
        X_scaled = standard_scaling(X)
    elif method == 'minmax':
        X_scaled = minmax_scaling(X, feature_range)
    elif method == 'robust':
        X_scaled = robust_scaling(X)
    else:
        X_scaled = X.copy()

    # Compute metrics
    metrics = compute_metrics(X, X_scaled, metric_funcs)

    # Prepare output
    result = {
        'result': X_scaled,
        'metrics': metrics,
        'params_used': {
            'method': method,
            'feature_range': feature_range if method == 'minmax' else None
        },
        'warnings': []
    }

    return result

# Example usage:
"""
X = np.random.rand(100, 5)
result = scaling_normalisation_fit(X, method='standard')
print(result['metrics'])
"""

################################################################################
# encoding_categoriel
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional, Any

def _validate_inputs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    encoding_method: str = "one_hot",
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if encoding_method not in ["one_hot", "ordinal", "target", "frequency"]:
        raise ValueError("encoding_method must be one of: 'one_hot', 'ordinal', 'target', 'frequency'")
    if y is not None and len(X) != len(y):
        raise ValueError("X and y must have the same length")

def _one_hot_encode(X: np.ndarray) -> Dict[str, Any]:
    """Perform one-hot encoding on categorical data."""
    unique_values = [np.unique(col) for col in X.T]
    encoded = np.zeros((X.shape[0], sum(len(u) for u in unique_values)))
    current_col = 0
    result_dict = {}

    for i, col in enumerate(X.T):
        for val in unique_values[i]:
            mask = (col == val)
            encoded[mask, current_col] = 1
            result_dict[f"col_{i}_val_{val}"] = current_col
            current_col += 1

    return {"encoded": encoded, "feature_names": result_dict}

def _ordinal_encode(X: np.ndarray) -> Dict[str, Any]:
    """Perform ordinal encoding on categorical data."""
    encoded = np.zeros_like(X, dtype=float)
    feature_names = {}

    for i, col in enumerate(X.T):
        unique_values = np.unique(col)
        value_to_int = {val: idx for idx, val in enumerate(unique_values)}
        encoded[:, i] = np.vectorize(value_to_int.get)(col)
        feature_names[f"col_{i}"] = value_to_int

    return {"encoded": encoded, "feature_names": feature_names}

def _target_encode(
    X: np.ndarray,
    y: np.ndarray,
    smoothing: float = 1.0
) -> Dict[str, Any]:
    """Perform target encoding on categorical data."""
    encoded = np.zeros_like(X, dtype=float)
    feature_names = {}

    for i, col in enumerate(X.T):
        unique_values = np.unique(col)
        value_to_target_mean = {}
        n_total = len(y)

        for val in unique_values:
            mask = (col == val)
            n_val = np.sum(mask)
            target_mean = np.mean(y[mask])
            smoothed_target_mean = (target_mean * n_val + y.mean() * smoothing) / (n_val + smoothing)
            value_to_target_mean[val] = smoothed_target_mean

        encoded[:, i] = np.vectorize(value_to_target_mean.get)(col)
        feature_names[f"col_{i}"] = value_to_target_mean

    return {"encoded": encoded, "feature_names": feature_names}

def _frequency_encode(X: np.ndarray) -> Dict[str, Any]:
    """Perform frequency encoding on categorical data."""
    encoded = np.zeros_like(X, dtype=float)
    feature_names = {}

    for i, col in enumerate(X.T):
        unique_values, counts = np.unique(col, return_counts=True)
        value_to_freq = {val: cnt / len(col) for val, cnt in zip(unique_values, counts)}
        encoded[:, i] = np.vectorize(value_to_freq.get)(col)
        feature_names[f"col_{i}"] = value_to_freq

    return {"encoded": encoded, "feature_names": feature_names}

def encoding_categoriel_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    encoding_method: str = "one_hot",
    smoothing: float = 1.0,
    custom_encoder: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Fit categorical encoding to data.

    Parameters:
    -----------
    X : np.ndarray
        Input categorical data (2D array)
    y : Optional[np.ndarray]
        Target values for target encoding
    encoding_method : str
        Encoding method to use ('one_hot', 'ordinal', 'target', 'frequency')
    smoothing : float
        Smoothing parameter for target encoding
    custom_encoder : Optional[Callable]
        Custom encoding function

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing encoded data and metadata
    """
    _validate_inputs(X, y, encoding_method)

    if custom_encoder is not None:
        result = custom_encoder(X, y)
    elif encoding_method == "one_hot":
        result = _one_hot_encode(X)
    elif encoding_method == "ordinal":
        result = _ordinal_encode(X)
    elif encoding_method == "target" and y is not None:
        result = _target_encode(X, y, smoothing)
    elif encoding_method == "frequency":
        result = _frequency_encode(X)
    else:
        raise ValueError("Invalid encoding method or missing target for target encoding")

    return {
        "result": result["encoded"],
        "metrics": {},
        "params_used": {
            "encoding_method": encoding_method,
            "smoothing": smoothing
        },
        "warnings": []
    }

################################################################################
# dimension_reduction
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

def normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'none':
        return X
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_distance_matrix(X: np.ndarray, distance_metric: str = 'euclidean') -> np.ndarray:
    """Compute distance matrix using specified metric."""
    if distance_metric == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    elif distance_metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
    elif distance_metric == 'cosine':
        dot_products = np.dot(X, X.T)
        norms = np.sqrt(np.sum(X ** 2, axis=1))
        return 1 - dot_products / (np.outer(norms, norms) + 1e-8)
    elif distance_metric == 'minkowski':
        p = 3
        return np.sum(np.abs(X[:, np.newaxis] - X) ** p, axis=2) ** (1/p)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

def pca(X: np.ndarray, n_components: int) -> Dict[str, Any]:
    """Principal Component Analysis."""
    validate_input(X)
    X_norm = normalize_data(X, 'standard')
    cov_matrix = np.cov(X_norm, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    components = eigenvectors[:, :n_components]
    explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)
    return {
        'components': components,
        'explained_variance': explained_variance
    }

def t_sne(X: np.ndarray, n_components: int = 2, perplexity: float = 30.0,
          learning_rate: float = 200.0, n_iter: int = 1000) -> Dict[str, Any]:
    """t-Distributed Stochastic Neighbor Embedding."""
    validate_input(X)
    X_norm = normalize_data(X, 'standard')
    distance_matrix = compute_distance_matrix(X_norm)
    n_samples = X.shape[0]

    # Early exaggeration
    P = np.exp(-distance_matrix ** 2 / (2 * perplexity ** 2))
    P = P / np.sum(P, axis=1)[:, np.newaxis]

    # Initialize solution randomly
    Y = np.random.randn(n_samples, n_components)

    for i in range(n_iter):
        # Compute pairwise affinities
        sum_Y = np.sum(Y ** 2, axis=1)
        num = 1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)
        Q = 1 / (1 + num)
        Q = (Q + Q.T) / (2 * n_samples)

        # Compute gradient
        PQ = (P - Q) * Q
        for j in range(n_samples):
            grad = np.repeat(Y[j] / num[j], n_samples).reshape(-1, n_components)
            grad -= Y
            grad = np.sum(grad * PQ[:, j], axis=0)
            Y[j] += learning_rate * grad

        # Early exaggeration
        if i == 100:
            P = P / 4

    return {
        'embedding': Y,
        'params_used': {
            'perplexity': perplexity,
            'learning_rate': learning_rate,
            'n_iter': n_iter
        }
    }

def dimension_reduction_fit(X: np.ndarray, method: str = 'pca', n_components: int = 2,
                           **kwargs) -> Dict[str, Any]:
    """
    Perform dimensionality reduction using specified method.

    Parameters:
    - X: Input data (n_samples, n_features)
    - method: Reduction method ('pca', 'tsne')
    - n_components: Number of dimensions in the embedded space
    - **kwargs: Additional method-specific parameters

    Returns:
    Dictionary containing results, metrics, and parameters used
    """
    validate_input(X)

    if method == 'pca':
        result = pca(X, n_components)
        metrics = {
            'explained_variance_ratio': result['explained_variance']
        }
    elif method == 'tsne':
        result = t_sne(X, n_components, **kwargs)
        metrics = {}
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'method': method,
            'n_components': n_components,
            **kwargs
        },
        'warnings': []
    }

# Example usage:
# result = dimension_reduction_fit(X, method='pca', n_components=2)

################################################################################
# clustering
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def clustering_fit(
    data: np.ndarray,
    n_clusters: int = 3,
    distance_metric: Union[str, Callable] = 'euclidean',
    normalization: str = 'standard',
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Perform clustering on the given data.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    n_clusters : int, optional
        Number of clusters to form (default: 3)
    distance_metric : str or callable, optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine', or custom callable)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', or 'robust')
    max_iter : int, optional
        Maximum number of iterations (default: 300)
    tol : float, optional
        Tolerance for stopping criteria (default: 1e-4)
    random_state : int, optional
        Random seed for reproducibility (default: None)
    custom_distance : callable, optional
        Custom distance function if not using built-in metrics

    Returns:
    --------
    dict
        Dictionary containing clustering results, metrics, parameters used, and warnings

    Example:
    --------
    >>> data = np.random.rand(100, 5)
    >>> result = clustering_fit(data, n_clusters=3, distance_metric='euclidean')
    """
    # Validate inputs
    _validate_inputs(data, n_clusters)

    # Normalize data
    normalized_data = _apply_normalization(data, normalization)

    # Initialize centroids
    if random_state is not None:
        np.random.seed(random_state)
    centroids = _initialize_centroids(normalized_data, n_clusters)

    # Main clustering loop
    for _ in range(max_iter):
        # Assign clusters
        labels = _assign_clusters(normalized_data, centroids, distance_metric, custom_distance)

        # Update centroids
        new_centroids = _update_centroids(normalized_data, labels)

        # Check for convergence
        if _check_convergence(centroids, new_centroids, tol):
            break

        centroids = new_centroids

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, labels, centroids, distance_metric)

    return {
        'result': {
            'labels': labels,
            'centroids': centroids
        },
        'metrics': metrics,
        'params_used': {
            'n_clusters': n_clusters,
            'distance_metric': distance_metric,
            'normalization': normalization
        },
        'warnings': []
    }

def _validate_inputs(data: np.ndarray, n_clusters: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if len(data.shape) != 2:
        raise ValueError("Input data must be 2-dimensional")
    if n_clusters <= 0:
        raise ValueError("Number of clusters must be positive")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the data."""
    if method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _initialize_centroids(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """Initialize cluster centroids."""
    indices = np.random.choice(data.shape[0], n_clusters, replace=False)
    return data[indices]

def _assign_clusters(
    data: np.ndarray,
    centroids: np.ndarray,
    distance_metric: Union[str, Callable],
    custom_distance: Optional[Callable] = None
) -> np.ndarray:
    """Assign data points to the nearest cluster."""
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_function(distance_metric)

    distances = np.array([distance_func(data, centroid) for centroid in centroids])
    return np.argmin(distances, axis=0)

def _get_distance_function(metric: str) -> Callable:
    """Get the appropriate distance function."""
    if metric == 'euclidean':
        return _euclidean_distance
    elif metric == 'manhattan':
        return _manhattan_distance
    elif metric == 'cosine':
        return _cosine_distance
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance."""
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Manhattan distance."""
    return np.sum(np.abs(a - b), axis=1)

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Cosine distance."""
    return 1 - np.dot(a, b.T) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b))

def _update_centroids(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Update cluster centroids."""
    centroids = []
    for k in range(np.max(labels) + 1):
        cluster_points = data[labels == k]
        if len(cluster_points) > 0:
            centroids.append(np.mean(cluster_points, axis=0))
        else:
            # If a cluster becomes empty, keep the previous centroid
            centroids.append(np.zeros(data.shape[1]))
    return np.array(centroids)

def _check_convergence(old_centroids: np.ndarray, new_centroids: np.ndarray, tol: float) -> bool:
    """Check if centroids have converged."""
    return np.linalg.norm(new_centroids - old_centroids) < tol

def _calculate_metrics(
    data: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    distance_metric: Union[str, Callable]
) -> Dict[str, float]:
    """Calculate clustering metrics."""
    if distance_metric == 'euclidean':
        distances = np.array([_euclidean_distance(data, centroid) for centroid in centroids])
    else:
        distances = np.array([_get_distance_function(distance_metric)(data, centroid) for centroid in centroids])

    intra_cluster_distances = distances[np.arange(len(labels)), labels]
    total_distance = np.sum(intra_cluster_distances)
    inertia = total_distance / len(labels)

    return {
        'inertia': float(inertia),
        'silhouette_score': _calculate_silhouette(data, labels)
    }

def _calculate_silhouette(data: np.ndarray, labels: np.ndarray) -> float:
    """Calculate silhouette score."""
    n_samples = len(data)
    if n_samples < 2:
        return 0.0

    silhouette_scores = []
    for i in range(n_samples):
        a = np.mean([_euclidean_distance(data[i], data[j]) for j in range(n_samples) if labels[j] == labels[i]])
        b = np.min([np.mean([_euclidean_distance(data[i], data[j]) for j in range(n_samples) if labels[j] == k])
                   for k in np.unique(labels) if k != labels[i]])
        s = (b - a) / max(a, b)
        silhouette_scores.append(s)

    return float(np.mean(silhouette_scores))

################################################################################
# regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict:
    """
    Fit a regression model with configurable parameters.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - normalizer: Function to normalize features
    - metric: Metric for evaluation ('mse', 'mae', 'r2', 'logloss') or custom callable
    - distance: Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable
    - solver: Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    - regularization: Regularization type (None, 'l1', 'l2', 'elasticnet')
    - tol: Tolerance for stopping criteria
    - max_iter: Maximum number of iterations
    - learning_rate: Learning rate for gradient descent
    - alpha: Regularization strength
    - l1_ratio: ElasticNet mixing parameter (0 <= l1_ratio <= 1)
    - custom_metric: Custom metric function
    - custom_distance: Custom distance function

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize features
    X_normalized = normalizer(X)

    # Choose solver and fit model
    if solver == 'closed_form':
        params = closed_form_solver(X_normalized, y, regularization, alpha, l1_ratio)
    elif solver == 'gradient_descent':
        params = gradient_descent_solver(X_normalized, y, metric, distance,
                                        regularization, alpha, l1_ratio,
                                        tol, max_iter, learning_rate)
    elif solver == 'newton':
        params = newton_solver(X_normalized, y, metric, distance,
                              regularization, alpha, l1_ratio,
                              tol, max_iter)
    elif solver == 'coordinate_descent':
        params = coordinate_descent_solver(X_normalized, y, metric, distance,
                                          regularization, alpha, l1_ratio,
                                          tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate predictions and metrics
    y_pred = predict(X_normalized, params)
    metrics = calculate_metrics(y, y_pred, metric, custom_metric)

    # Prepare results
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter,
            'learning_rate': learning_rate,
            'alpha': alpha,
            'l1_ratio': l1_ratio
        },
        'warnings': []
    }

    return result

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def closed_form_solver(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    alpha: float,
    l1_ratio: float
) -> np.ndarray:
    """Closed form solution for linear regression."""
    if regularization is None:
        params = np.linalg.inv(X.T @ X) @ X.T @ y
    elif regularization == 'l2':
        params = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ y
    elif regularization == 'l1':
        # Use coordinate descent for L1
        params = coordinate_descent_solver(X, y, 'mse', 'euclidean',
                                          'l1', alpha, 1.0,
                                          1e-4, 1000)
    elif regularization == 'elasticnet':
        params = coordinate_descent_solver(X, y, 'mse', 'euclidean',
                                          'elasticnet', alpha, l1_ratio,
                                          1e-4, 1000)
    else:
        raise ValueError(f"Unknown regularization: {regularization}")
    return params

def gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    alpha: float,
    l1_ratio: float,
    tol: float,
    max_iter: int,
    learning_rate: float
) -> np.ndarray:
    """Gradient descent solver for regression."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        gradients = compute_gradients(X, y, params, metric, distance,
                                    regularization, alpha, l1_ratio)
        params -= learning_rate * gradients

        current_loss = compute_loss(y, predict(X, params), metric)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return params

def newton_solver(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    alpha: float,
    l1_ratio: float,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Newton's method solver for regression."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        gradients = compute_gradients(X, y, params, metric, distance,
                                    regularization, alpha, l1_ratio)
        hessian = compute_hessian(X, metric)

        if regularization == 'l2':
            hessian += alpha * np.eye(n_features)

        params -= np.linalg.inv(hessian) @ gradients

        current_loss = compute_loss(y, predict(X, params), metric)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return params

def coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    alpha: float,
    l1_ratio: float,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Coordinate descent solver for regression."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residual = y - (X @ params) + params[j] * X_j

            if regularization == 'l1':
                params[j] = soft_threshold(residual @ X_j, alpha)
            elif regularization == 'elasticnet':
                params[j] = elasticnet_threshold(residual @ X_j, alpha, l1_ratio)
            else:
                params[j] = (residual @ X_j) / (X_j @ X_j)

        current_loss = compute_loss(y, predict(X, params), metric)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return params

def compute_gradients(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    alpha: float,
    l1_ratio: float
) -> np.ndarray:
    """Compute gradients for the given metric."""
    y_pred = predict(X, params)
    if callable(metric):
        gradients = metric_gradient(y, y_pred, X, metric)
    elif metric == 'mse':
        gradients = 2 * (y_pred - y) @ X / len(y)
    elif metric == 'mae':
        gradients = np.sign(y_pred - y) @ X / len(y)
    elif metric == 'r2':
        gradients = 2 * (y_pred - y) @ X / len(y)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if regularization == 'l2':
        gradients += 2 * alpha * params
    elif regularization == 'l1':
        gradients += alpha * np.sign(params)
    elif regularization == 'elasticnet':
        gradients += alpha * (l1_ratio * np.sign(params) + (1 - l1_ratio) * 2 * params)

    return gradients

def compute_hessian(
    X: np.ndarray,
    metric: Union[str, Callable]
) -> np.ndarray:
    """Compute Hessian matrix for the given metric."""
    if callable(metric):
        raise NotImplementedError("Custom Hessian computation not implemented")
    elif metric == 'mse':
        return 2 * (X.T @ X) / len(X)
    elif metric == 'mae':
        raise NotImplementedError("Hessian for MAE not defined")
    elif metric == 'r2':
        return 2 * (X.T @ X) / len(X)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def compute_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Compute loss for the given metric."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot
    elif metric == 'logloss':
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate multiple metrics."""
    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        metrics['r2'] = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    elif metric == 'logloss':
        metrics['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)

    return metrics

def predict(
    X: np.ndarray,
    params: np.ndarray
) -> np.ndarray:
    """Make predictions using the model parameters."""
    return X @ params

def soft_threshold(
    x: float,
    alpha: float
) -> float:
    """Soft thresholding operator for L1 regularization."""
    if x > alpha:
        return x - alpha
    elif x < -alpha:
        return x + alpha
    else:
        return 0

def elasticnet_threshold(
    x: float,
    alpha: float,
    l1_ratio: float
) -> float:
    """Thresholding operator for ElasticNet regularization."""
    if l1_ratio == 1:
        return soft_threshold(x, alpha)
    elif l1_ratio == 0:
        if x > alpha:
            return max(0, x - alpha)
        elif x < -alpha:
            return min(0, x + alpha)
        else:
            return 0
    else:
        # Combined L1 and L2 thresholding
        if x > alpha * l1_ratio:
            return max(0, (x - alpha * (1 - l1_ratio)) / (l1_ratio + 2 * alpha * (1 - l1_ratio)))
        elif x < -alpha * l1_ratio:
            return min(0, (x + alpha * (1 - l1_ratio)) / (l1_ratio + 2 * alpha * (1 - l1_ratio)))
        else:
            return 0

# Example usage
if __name__ == "__main__":
    X = np.random.rand(100, 5)
    y = np.random.rand(100)

    result = regression_fit(
        X, y,
        normalizer=lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0),
        metric='mse',
        solver='gradient_descent',
        regularization='l2',
        alpha=0.1,
        learning_rate=0.01
    )

################################################################################
# classification
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def classification_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "accuracy",
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Fit a classification model to the given data.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the features. Default is None.
    distance_metric : str, optional
        Distance metric to use. Options: "euclidean", "manhattan", "cosine", "minkowski". Default is "euclidean".
    solver : str, optional
        Solver to use. Options: "closed_form", "gradient_descent", "newton", "coordinate_descent". Default is "closed_form".
    regularization : Optional[str], optional
        Regularization type. Options: None, "l1", "l2", "elasticnet". Default is None.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to evaluate the model. Options: "accuracy", "f1", "precision", "recall", or a custom callable. Default is "accuracy".
    tol : float, optional
        Tolerance for stopping criteria. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function. Default is None.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function. Default is None.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Choose distance metric
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_metric(distance_metric)

    # Choose solver
    if solver == "closed_form":
        params = _closed_form_solver(X_normalized, y)
    elif solver == "gradient_descent":
        params = _gradient_descent_solver(X_normalized, y, tol=tol, max_iter=max_iter)
    elif solver == "newton":
        params = _newton_solver(X_normalized, y, tol=tol, max_iter=max_iter)
    elif solver == "coordinate_descent":
        params = _coordinate_descent_solver(X_normalized, y, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization is not None:
        params = _apply_regularization(params, X_normalized, y, regularization)

    # Choose metric
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric(metric)

    # Calculate metrics
    y_pred = _predict(X_normalized, params)
    metrics = {"accuracy": metric_func(y, y_pred)}

    # Prepare output
    result = {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "distance_metric": distance_metric,
            "solver": solver,
            "regularization": regularization,
            "metric": metric if isinstance(metric, str) else None
        },
        "warnings": []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate the input data."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or inf values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or inf values.")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Apply normalization to the feature matrix."""
    if normalizer is not None:
        return normalizer(X)
    return X

def _get_distance_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the distance metric function."""
    metrics = {
        "euclidean": lambda a, b: np.linalg.norm(a - b),
        "manhattan": lambda a, b: np.sum(np.abs(a - b)),
        "cosine": lambda a, b: 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
        "minkowski": lambda a, b: np.sum(np.abs(a - b) ** 3) ** (1/3)
    }
    if metric not in metrics:
        raise ValueError(f"Unknown distance metric: {metric}")
    return metrics[metric]

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form solver for classification."""
    # Placeholder for actual implementation
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _gradient_descent_solver(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Gradient descent solver for classification."""
    # Placeholder for actual implementation
    params = np.zeros(X.shape[1])
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y)
        params -= tol * gradient
    return params

def _newton_solver(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Newton solver for classification."""
    # Placeholder for actual implementation
    params = np.zeros(X.shape[1])
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y)
        hessian = 2 * X.T @ X
        params -= np.linalg.pinv(hessian) @ gradient
    return params

def _coordinate_descent_solver(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Coordinate descent solver for classification."""
    # Placeholder for actual implementation
    params = np.zeros(X.shape[1])
    for _ in range(max_iter):
        for i in range(X.shape[1]):
            X_i = X[:, i]
            params[i] -= tol * (2 * X_i.T @ (X @ params - y))
    return params

def _apply_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray, regularization: str) -> np.ndarray:
    """Apply regularization to the parameters."""
    if regularization == "l1":
        params -= 0.1 * np.sign(params)
    elif regularization == "l2":
        params -= 0.1 * params
    elif regularization == "elasticnet":
        params -= 0.1 * (np.sign(params) + params)
    return params

def _get_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the metric function."""
    metrics = {
        "accuracy": lambda y_true, y_pred: np.mean(y_true == y_pred),
        "f1": lambda y_true, y_pred: 2 * (np.sum(y_true == y_pred)) / (np.sum(y_true) + np.sum(y_pred)),
        "precision": lambda y_true, y_pred: np.sum(y_true == y_pred) / np.sum(y_pred),
        "recall": lambda y_true, y_pred: np.sum(y_true == y_pred) / np.sum(y_true)
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _predict(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Predict the target values."""
    return (X @ params > 0.5).astype(int)

################################################################################
# feature_selection
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def feature_selection_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric_weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Perform feature selection on the given dataset.

    Parameters:
    -----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    normalizer : Optional[Callable]
        Function to normalize the features. Default is None.
    metric : Union[str, Callable]
        Metric to evaluate feature selection. Can be "mse", "mae", "r2", or a custom callable.
    distance : Union[str, Callable]
        Distance metric for feature selection. Can be "euclidean", "manhattan", "cosine", or a custom callable.
    solver : str
        Solver to use for feature selection. Can be "closed_form", "gradient_descent", etc.
    regularization : Optional[str]
        Regularization type. Can be "l1", "l2", or None.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric_weights : Optional[np.ndarray]
        Weights for custom metrics.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_normalized = _apply_normalization(X, normalizer)

    # Select solver and compute feature selection
    if solver == "closed_form":
        result = _closed_form_solver(X_normalized, y, regularization)
    elif solver == "gradient_descent":
        result = _gradient_descent_solver(X_normalized, y, metric, distance, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(y, result["predictions"], metric, custom_metric_weights)

    # Prepare output
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return output

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to the features."""
    if normalizer is None:
        return X
    return normalizer(X)

def _closed_form_solver(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str]
) -> Dict[str, Any]:
    """Closed form solution for feature selection."""
    if regularization is None:
        # Simple linear regression
        XTX = np.dot(X.T, X)
        if not np.allclose(XTX, XTX.T):
            raise ValueError("X^T X is not symmetric positive definite")
        coefficients = np.linalg.solve(XTX, np.dot(X.T, y))
    elif regularization == "l2":
        # Ridge regression
        lambda_ = 1.0  # Default regularization strength
        coefficients = np.linalg.solve(XTX + lambda_ * np.eye(X.shape[1]), np.dot(X.T, y))
    else:
        raise ValueError(f"Unknown regularization: {regularization}")

    predictions = np.dot(X, coefficients)
    return {"coefficients": coefficients, "predictions": predictions}

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Gradient descent solver for feature selection."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    learning_rate = 0.01
    prev_loss = float('inf')

    for _ in range(max_iter):
        gradients = _compute_gradients(X, y, coefficients, distance)
        coefficients -= learning_rate * gradients
        current_loss = _compute_metric(y, np.dot(X, coefficients), metric)

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    predictions = np.dot(X, coefficients)
    return {"coefficients": coefficients, "predictions": predictions}

def _compute_gradients(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    distance: Union[str, Callable]
) -> np.ndarray:
    """Compute gradients for gradient descent."""
    predictions = np.dot(X, coefficients)
    if distance == "euclidean":
        residuals = y - predictions
        gradients = -2 * np.dot(X.T, residuals) / X.shape[0]
    elif callable(distance):
        gradients = distance(X, y - predictions)
    else:
        raise ValueError(f"Unknown distance: {distance}")
    return gradients

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Compute the specified metric."""
    if metric == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    elif callable(metric):
        return metric(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric_weights: Optional[np.ndarray]
) -> Dict[str, float]:
    """Compute all relevant metrics."""
    metrics = {}
    if isinstance(metric, str):
        metrics[metric] = _compute_metric(y_true, y_pred, metric)
    else:
        metrics["custom"] = _compute_metric(y_true, y_pred, metric)

    if custom_metric_weights is not None:
        weighted_metrics = {}
        for i, weight in enumerate(custom_metric_weights):
            weighted_metrics[f"custom_{i}"] = _compute_metric(y_true, y_pred, metric) * weight
        metrics.update(weighted_metrics)

    return metrics

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.random.rand(100)

result = feature_selection_fit(
    X=X,
    y=y,
    normalizer=None,
    metric="mse",
    distance="euclidean",
    solver="closed_form",
    regularization=None
)
"""

################################################################################
# exploration_temporelle
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    time_series: np.ndarray,
    window_size: int,
    step_size: int
) -> None:
    """Validate input data and parameters."""
    if not isinstance(time_series, np.ndarray):
        raise TypeError("time_series must be a numpy array")
    if time_series.ndim != 1:
        raise ValueError("time_series must be 1-dimensional")
    if np.any(np.isnan(time_series)) or np.any(np.isinf(time_series)):
        raise ValueError("time_series contains NaN or infinite values")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")

def _normalize_data(
    data: np.ndarray,
    method: str = 'standard'
) -> np.ndarray:
    """Normalize the time series data."""
    if method == 'none':
        return data
    elif method == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    metrics: Union[str, Callable]
) -> Dict[str, float]:
    """Compute the specified metrics."""
    result = {}
    if isinstance(metrics, str):
        if metrics == 'mse':
            result['mse'] = np.mean((actual - predicted) ** 2)
        elif metrics == 'mae':
            result['mae'] = np.mean(np.abs(actual - predicted))
        elif metrics == 'r2':
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            result['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
        elif metrics == 'logloss':
            result['logloss'] = -np.mean(actual * np.log(predicted + 1e-8) +
                                        (1 - actual) * np.log(1 - predicted + 1e-8))
    else:
        result['custom'] = metrics(actual, predicted)
    return result

def _compute_statistics(
    window: np.ndarray,
    statistic_func: Callable
) -> float:
    """Compute the specified statistic for a window."""
    return statistic_func(window)

def exploration_temporelle_fit(
    time_series: np.ndarray,
    window_size: int = 10,
    step_size: int = 1,
    normalize_method: str = 'standard',
    statistic_func: Callable = np.mean,
    metrics: Union[str, Callable] = 'mse',
    **kwargs
) -> Dict[str, Any]:
    """
    Perform temporal exploration on a time series.

    Parameters:
    -----------
    time_series : np.ndarray
        Input time series data.
    window_size : int, optional
        Size of the sliding window (default: 10).
    step_size : int, optional
        Step size for the sliding window (default: 1).
    normalize_method : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard').
    statistic_func : Callable, optional
        Function to compute statistics on each window (default: np.mean).
    metrics : Union[str, Callable], optional
        Metrics to compute ('mse', 'mae', 'r2', 'logloss') or custom callable (default: 'mse').

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_inputs(time_series, window_size, step_size)

    normalized_data = _normalize_data(time_series, normalize_method)
    n_windows = (len(normalized_data) - window_size) // step_size + 1

    results = []
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        window = normalized_data[start:end]
        stat = _compute_statistics(window, statistic_func)
        results.append(stat)

    params_used = {
        'window_size': window_size,
        'step_size': step_size,
        'normalize_method': normalize_method,
        'statistic_func': statistic_func.__name__ if callable(statistic_func) else statistic_func,
        'metrics': metrics.__name__ if callable(metrics) else metrics
    }

    warnings = []
    if len(results) == 0:
        warnings.append("No windows could be computed with the given parameters.")

    return {
        'result': np.array(results),
        'metrics': {},
        'params_used': params_used,
        'warnings': warnings
    }
