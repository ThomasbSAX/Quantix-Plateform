"""
Quantix – Module distance_correlation
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# definition
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    Y: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    """Validate input arrays and distance metric."""
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Inputs must be 2-dimensional arrays.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
        raise ValueError("Y contains NaN or infinite values.")

    # Test distance metric with sample data
    test_X = np.array([[1, 2], [3, 4]])
    test_Y = np.array([[5, 6], [7, 8]])
    try:
        distance_metric(test_X, test_Y)
    except Exception as e:
        raise ValueError(f"Distance metric failed with error: {str(e)}")

def _compute_distance_matrix(
    data: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Compute pairwise distance matrix for given data."""
    n = data.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = distance_metric(data[i], data[j])
    return dist_matrix

def _center_distance_matrix(
    dist_matrix: np.ndarray,
) -> np.ndarray:
    """Center the distance matrix by subtracting row and column means."""
    n = dist_matrix.shape[0]
    row_means = np.mean(dist_matrix, axis=1)
    col_means = np.mean(dist_matrix, axis=0)
    double_centered = dist_matrix - row_means[:, np.newaxis] - col_means + np.mean(row_means)
    return double_centered

def _compute_distance_covariance(
    A: np.ndarray,
    B: np.ndarray,
) -> float:
    """Compute the distance covariance between two centered distance matrices."""
    n = A.shape[0]
    numerator = np.sum(A * B) / (n ** 2)
    denominator_A = np.sqrt(np.sum(A ** 2) / (n ** 3))
    denominator_B = np.sqrt(np.sum(B ** 2) / (n ** 3))
    return numerator / (denominator_A * denominator_B)

def definition_compute(
    X: np.ndarray,
    Y: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.linalg.norm,
    normalization: Optional[str] = None,
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Compute the distance correlation between two datasets.

    Parameters:
    -----------
    X : np.ndarray
        First dataset of shape (n_samples, n_features)
    Y : np.ndarray
        Second dataset of shape (n_samples, n_features)
    distance_metric : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Distance metric to use for computing pairwise distances
    normalization : str, optional
        Normalization method (not implemented yet)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(10, 5)
    >>> Y = np.random.rand(10, 3)
    >>> result = definition_compute(X, Y)
    """
    # Validate inputs
    _validate_inputs(X, Y, distance_metric)

    # Compute distance matrices
    A = _compute_distance_matrix(X, distance_metric)
    B = _compute_distance_matrix(Y, distance_metric)

    # Center the distance matrices
    A_centered = _center_distance_matrix(A)
    B_centered = _center_distance_matrix(B)

    # Compute distance covariance
    dcov = _compute_distance_covariance(A_centered, B_centered)

    # Compute distance variances
    dvar_X = _compute_distance_covariance(A_centered, A_centered)
    dvar_Y = _compute_distance_covariance(B_centered, B_centered)

    # Compute distance correlation
    if dvar_X == 0 or dvar_Y == 0:
        dcor = 0.0
    else:
        dcor = dcov / np.sqrt(dvar_X * dvar_Y)

    # Prepare output
    result = {
        "result": {"distance_correlation": float(dcor)},
        "metrics": {
            "distance_covariance": float(dcov),
            "distance_variance_X": float(dvar_X),
            "distance_variance_Y": float(dvar_Y)
        },
        "params_used": {
            "distance_metric": str(distance_metric.__name__),
            "normalization": normalization
        },
        "warnings": []
    }

    return result

# Default distance metrics
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)

def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan distance between two vectors."""
    return np.sum(np.abs(a - b))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors (1 - cosine similarity)."""
    return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

################################################################################
# properties
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def properties_fit(
    X: np.ndarray,
    y: np.ndarray,
    distance_metric: Union[str, Callable] = "euclidean",
    normalization: str = "none",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute properties of distance correlation.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    distance_metric : str or callable
        Distance metric to use. Can be 'euclidean', 'manhattan', 'cosine',
        'minkowski', or a custom callable.
    normalization : str
        Normalization method. Can be 'none', 'standard', 'minmax', or 'robust'.
    solver : str
        Solver to use. Can be 'closed_form', 'gradient_descent', 'newton',
        or 'coordinate_descent'.
    regularization : str, optional
        Regularization method. Can be 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function.

    Returns:
    --------
    Dict containing:
        - result: Computed properties.
        - metrics: Performance metrics.
        - params_used: Parameters used in the computation.
        - warnings: Any warnings encountered.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_norm = _apply_normalization(X, normalization)
    y_norm = _apply_normalization(y.reshape(-1, 1), normalization).flatten()

    # Compute distance matrices
    distance_func = _get_distance_function(distance_metric)
    A = distance_func(X_norm)
    B = distance_func(y_norm.reshape(-1, 1))

    # Compute distance correlation properties
    result = _compute_properties(A, B, solver, regularization, tol, max_iter)

    # Compute metrics
    metrics = _compute_metrics(X_norm, y_norm, result, custom_metric)

    # Prepare output
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "distance_metric": distance_metric,
            "normalization": normalization,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return output

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or Inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or Inf values")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to data."""
    if method == "none":
        return data
    elif method == "standard":
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == "minmax":
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == "robust":
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _get_distance_function(metric: Union[str, Callable]) -> Callable:
    """Get distance function based on metric."""
    if callable(metric):
        return metric
    elif metric == "euclidean":
        return lambda x: np.sqrt(np.sum((x[:, np.newaxis] - x) ** 2, axis=2))
    elif metric == "manhattan":
        return lambda x: np.sum(np.abs(x[:, np.newaxis] - x), axis=2)
    elif metric == "cosine":
        return lambda x: 1 - np.dot(x, x.T) / (np.linalg.norm(x, axis=1)[:, np.newaxis] * np.linalg.norm(x, axis=1))
    elif metric == "minkowski":
        return lambda x: np.sum(np.abs(x[:, np.newaxis] - x) ** 3, axis=2) ** (1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _compute_properties(A: np.ndarray, B: np.ndarray, solver: str,
                        regularization: Optional[str], tol: float, max_iter: int) -> Dict:
    """Compute distance correlation properties."""
    # Placeholder for actual computation
    return {
        "distance_correlation": 0.5,
        "statistic_value": 1.234,
        "p_value": 0.05
    }

def _compute_metrics(X: np.ndarray, y: np.ndarray, result: Dict,
                     custom_metric: Optional[Callable]) -> Dict:
    """Compute performance metrics."""
    metrics = {
        "mse": np.mean((X @ result["coefficients"] - y) ** 2),
        "mae": np.mean(np.abs(X @ result["coefficients"] - y)),
        "r2": 1 - np.sum((y - X @ result["coefficients"]) ** 2) / np.sum((y - np.mean(y)) ** 2)
    }
    if custom_metric is not None:
        metrics["custom"] = custom_metric(X, y)
    return metrics

# Example usage:
"""
X_example = np.random.rand(100, 5)
y_example = np.random.rand(100)

result = properties_fit(
    X=X_example,
    y=y_example,
    distance_metric="euclidean",
    normalization="standard",
    solver="closed_form"
)
"""

################################################################################
# calculation_method
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("Inputs must have the same shape.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Inputs must not contain NaN values.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Inputs must not contain infinite values.")

def _compute_distance_matrix(a: np.ndarray, metric: Callable) -> np.ndarray:
    """Compute distance matrix using the specified metric."""
    n = a.shape[0]
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dm[i, j] = metric(a[i], a[j])
    return dm

def _center_distance_matrix(dm: np.ndarray) -> np.ndarray:
    """Center the distance matrix."""
    n = dm.shape[0]
    a = np.mean(dm, axis=1)
    b = np.mean(dm, axis=0)
    return dm - a[:, np.newaxis] - b[np.newaxis, :] + np.mean(a)

def _compute_distance_covariance(dm_x: np.ndarray, dm_y: np.ndarray) -> float:
    """Compute the distance covariance."""
    n = dm_x.shape[0]
    dcov = np.sqrt(np.sum(dm_x * dm_y) / (n ** 2))
    return dcov

def _compute_distance_correlation(dcov: float, dvar_x: float, dvar_y: float) -> float:
    """Compute the distance correlation."""
    if dvar_x == 0 or dvar_y == 0:
        return 0.0
    return dcov / np.sqrt(dvar_x * dvar_y)

def calculation_method_fit(
    x: np.ndarray,
    y: np.ndarray,
    metric_x: Callable = np.linalg.norm,
    metric_y: Callable = np.linalg.norm,
    normalization: str = "none",
) -> Dict[str, Any]:
    """
    Compute the distance correlation between two arrays.

    Parameters:
    - x: Input array 1.
    - y: Input array 2.
    - metric_x: Distance metric for x (default: Euclidean).
    - metric_y: Distance metric for y (default: Euclidean).
    - normalization: Normalization method ("none", "standard", "minmax", "robust").

    Returns:
    - Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, y)

    # Normalization
    if normalization == "standard":
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    elif normalization == "minmax":
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == "robust":
        x = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))

    # Compute distance matrices
    dm_x = _compute_distance_matrix(x, metric_x)
    dm_y = _compute_distance_matrix(y, metric_y)

    # Center distance matrices
    dm_x_centered = _center_distance_matrix(dm_x)
    dm_y_centered = _center_distance_matrix(dm_y)

    # Compute distance covariance and variances
    dcov = _compute_distance_covariance(dm_x_centered, dm_y_centered)
    dvar_x = np.sqrt(np.sum(dm_x_centered ** 2) / (x.shape[0] ** 2))
    dvar_y = np.sqrt(np.sum(dm_y_centered ** 2) / (y.shape[0] ** 2))

    # Compute distance correlation
    dcor = _compute_distance_correlation(dcov, dvar_x, dvar_y)

    return {
        "result": dcor,
        "metrics": {"dcov": dcov, "dvar_x": dvar_x, "dvar_y": dvar_y},
        "params_used": {
            "metric_x": metric_x.__name__,
            "metric_y": metric_y.__name__,
            "normalization": normalization,
        },
        "warnings": [],
    }

################################################################################
# range_values
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def range_values_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalization: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute range values for distance correlation analysis.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    distance_metric : str or callable, optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine', etc.)
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

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = range_values_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = _apply_normalization(X, normalization)
    y_norm = _apply_normalization(y.reshape(-1, 1), normalization).flatten()

    # Choose distance metric
    if isinstance(distance_metric, str):
        distance_func = _get_distance_function(distance_metric)
    else:
        distance_func = distance_metric

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(X_norm, y_norm, distance_func)
    else:
        params = _solve_iterative(
            X_norm, y_norm, distance_func,
            solver=solver,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter
        )

    # Calculate metrics
    metrics = _calculate_metrics(X_norm, y_norm, params, distance_func, custom_metric)

    return {
        'result': _compute_range_values(X_norm, y_norm, params),
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'distance_metric': distance_metric,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(X_norm, y_norm)
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to data."""
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

def _get_distance_function(metric_name: str) -> Callable:
    """Get distance function based on name."""
    if metric_name == 'euclidean':
        return lambda a, b: np.linalg.norm(a - b)
    elif metric_name == 'manhattan':
        return lambda a, b: np.sum(np.abs(a - b))
    elif metric_name == 'cosine':
        return lambda a, b: 1 - np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    elif metric_name == 'minkowski':
        return lambda a, b, p=3: np.sum(np.abs(a - b)**p)**(1/p)
    else:
        raise ValueError(f"Unknown distance metric: {metric_name}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray, distance_func: Callable) -> Dict:
    """Solve using closed form solution."""
    # Placeholder for actual implementation
    return {'coefficients': np.linalg.pinv(X) @ y}

def _solve_iterative(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict:
    """Solve using iterative methods."""
    # Placeholder for actual implementation
    return {'coefficients': np.zeros(X.shape[1])}

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict,
    distance_func: Callable,
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate various metrics."""
    metrics = {}
    y_pred = X @ params['coefficients']

    # Default metrics
    metrics['mse'] = np.mean((y - y_pred)**2)
    metrics['mae'] = np.mean(np.abs(y - y_pred))
    metrics['r2'] = 1 - metrics['mse']/np.var(y)

    # Custom metric if provided
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y, y_pred)

    return metrics

def _compute_range_values(X: np.ndarray, y: np.ndarray, params: Dict) -> float:
    """Compute range values for distance correlation."""
    # Placeholder for actual implementation
    return np.max(np.abs(X @ params['coefficients'] - y))

def _check_warnings(X: np.ndarray, y: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if X.shape[1] > X.shape[0]:
        warnings.append("Warning: Number of features exceeds number of samples")
    if np.any(np.std(X, axis=0) == 0):
        warnings.append("Warning: Zero-variance features detected")
    return warnings

################################################################################
# independence_test
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def independence_test_fit(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    distance_metric: Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = 'euclidean',
    normalization: str = 'none',
    test_statistic: Callable[[np.ndarray, np.ndarray], float] = None,
    n_bootstraps: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Test d'indépendance basé sur la distance de corrélation.

    Parameters
    ----------
    X : np.ndarray
        Premier ensemble de données (n_samples, n_features).
    Y : np.ndarray
        Deuxième ensemble de données (n_samples, n_features).
    distance_metric : str or callable
        Métrique de distance à utiliser ('euclidean', 'manhattan', 'cosine') ou une fonction personnalisée.
    normalization : str
        Type de normalisation ('none', 'standard', 'minmax', 'robust').
    test_statistic : callable
        Fonction pour calculer la statistique de test personnalisée.
    n_bootstraps : int
        Nombre d'itérations pour le bootstrap.
    random_state : int, optional
        Graine aléatoire pour la reproductibilité.

    Returns
    -------
    Dict[str, Any]
        Dictionnaire contenant les résultats du test.

    Examples
    --------
    >>> X = np.random.rand(100, 2)
    >>> Y = np.random.rand(100, 2)
    >>> result = independence_test_fit(X, Y)
    """
    # Validation des entrées
    X, Y = _validate_inputs(X, Y)

    # Normalisation
    if normalization != 'none':
        X = _apply_normalization(X, normalization)
        Y = _apply_normalization(Y, normalization)

    # Calcul des matrices de distance
    A = _compute_distance_matrix(X, X, distance_metric)
    B = _compute_distance_matrix(Y, Y, distance_metric)

    # Calcul de la statistique de test
    if test_statistic is None:
        statistic = _compute_distance_correlation_statistic(A, B)
    else:
        statistic = test_statistic(A, B)

    # Bootstrap pour le p-value
    p_value = _compute_p_value(statistic, A, B, n_bootstraps, random_state)

    return {
        'result': {
            'statistic': statistic,
            'p_value': p_value
        },
        'params_used': {
            'distance_metric': distance_metric,
            'normalization': normalization,
            'n_bootstraps': n_bootstraps
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validation des entrées."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X et Y doivent avoir le même nombre d'échantillons.")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contient des NaN ou des inf.")
    if np.isnan(Y).any() or np.isinf(Y).any():
        raise ValueError("Y contient des NaN ou des inf.")
    return X, Y

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Application de la normalisation."""
    if method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    return data

def _compute_distance_matrix(X: np.ndarray, Y: np.ndarray, metric: Union[str, Callable]) -> np.ndarray:
    """Calcul de la matrice de distance."""
    if callable(metric):
        return metric(X, Y)
    elif metric == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1))
    elif metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis, :] - Y[np.newaxis, :, :]), axis=-1)
    elif metric == 'cosine':
        return 1 - np.dot(X, Y.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(Y, axis=1)[np.newaxis, :])
    else:
        raise ValueError("Métrique de distance non reconnue.")

def _compute_distance_correlation_statistic(A: np.ndarray, B: np.ndarray) -> float:
    """Calcul de la statistique de distance de corrélation."""
    n = A.shape[0]
    a_sq = np.sum(A ** 2) / (n ** 2)
    b_sq = np.sum(B ** 2) / (n ** 2)
    ab = np.sum(A * B) / (n ** 2)
    dcov = ab - a_sq - b_sq
    dvar_a = np.sqrt(np.sum((A - a_sq) ** 2) / (n ** 2))
    dvar_b = np.sqrt(np.sum((B - b_sq) ** 2) / (n ** 2))
    return dcov / np.sqrt(dvar_a * dvar_b)

def _compute_p_value(statistic: float, A: np.ndarray, B: np.ndarray, n_bootstraps: int, random_state: Optional[int]) -> float:
    """Calcul du p-value par bootstrap."""
    rng = np.random.RandomState(random_state)
    boot_stats = []
    n = A.shape[0]
    for _ in range(n_bootstraps):
        idx = rng.choice(n, n, replace=True)
        A_boot = A[idx, :][:, idx]
        B_boot = B[idx, :][:, idx]
        boot_stats.append(_compute_distance_correlation_statistic(A_boot, B_boot))
    return np.mean(np.abs(boot_stats) >= np.abs(statistic))

################################################################################
# advantages
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def advantages_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Compute the advantages using distance correlation methodology.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric to compute ('mse', 'mae', 'r2', 'logloss') or custom callable
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics
    custom_distance : callable, optional
        Custom distance function if not using built-in distances

    Returns:
    --------
    Dict containing:
        - 'result': computed advantages
        - 'metrics': dictionary of metrics
        - 'params_used': parameters actually used
        - 'warnings': any warnings encountered

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = advantages_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = _apply_normalization(X, normalization)
    y_norm = _apply_normalization(y.reshape(-1, 1), normalization).flatten()

    # Prepare parameters
    params_used = {
        'normalization': normalization,
        'metric': metric if isinstance(metric, str) else 'custom',
        'distance': distance,
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Select metric function
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        if custom_metric is None:
            raise ValueError("custom_metric must be provided when metric is a callable")
        metric_func = custom_metric

    # Select distance function
    if isinstance(distance, str):
        distance_func = _get_distance_function(distance)
    else:
        if custom_distance is None:
            raise ValueError("custom_distance must be provided when distance is a callable")
        distance_func = custom_distance

    # Compute advantages based on solver
    if solver == 'closed_form':
        result = _compute_advantages_closed_form(X_norm, y_norm, distance_func)
    elif solver == 'gradient_descent':
        result = _compute_advantages_gradient_descent(
            X_norm, y_norm, distance_func, metric_func,
            regularization=regularization, tol=tol, max_iter=max_iter
        )
    else:
        raise ValueError(f"Unsupported solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_norm, y_norm, result, metric_func)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to data."""
    if method == 'none':
        return data
    elif method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _get_metric_function(metric_name: str) -> Callable:
    """Return the appropriate metric function."""
    metrics = {
        'mse': _mse,
        'mae': _mae,
        'r2': _r2_score,
        'logloss': _log_loss
    }
    if metric_name not in metrics:
        raise ValueError(f"Unknown metric: {metric_name}")
    return metrics[metric_name]

def _get_distance_function(distance_name: str) -> Callable:
    """Return the appropriate distance function."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': lambda x, y: np.linalg.norm(x - y, ord=3)
    }
    if distance_name not in distances:
        raise ValueError(f"Unknown distance: {distance_name}")
    return distances[distance_name]

def _compute_advantages_closed_form(X: np.ndarray, y: np.ndarray, distance_func: Callable) -> np.ndarray:
    """Compute advantages using closed-form solution."""
    # This is a placeholder for the actual implementation
    return np.random.rand(X.shape[1])

def _compute_advantages_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable,
    metric_func: Callable,
    *,
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Compute advantages using gradient descent."""
    # This is a placeholder for the actual implementation
    return np.random.rand(X.shape[1])

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    advantages: np.ndarray,
    metric_func: Callable
) -> Dict:
    """Compute various metrics for the advantages."""
    predictions = X @ advantages

    return {
        'metric': metric_func(y, predictions),
        'r2_score': _r2_score(y, predictions),
        'mae': _mae(y, predictions)
    }

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Euclidean distance."""
    return np.linalg.norm(x - y)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Manhattan distance."""
    return np.sum(np.abs(x - y))

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine distance."""
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

################################################################################
# limitations
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def limitations_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizations: Dict[str, str] = {"X": "none", "y": "none"},
    distance_metric: Union[str, Callable] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute limitations based on distance correlation.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    normalizations : Dict[str, str]
        Dictionary specifying normalization for X and y.
        Options: "none", "standard", "minmax", "robust".
    distance_metric : Union[str, Callable]
        Distance metric to use. Options: "euclidean", "manhattan",
        "cosine", "minkowski", or custom callable.
    solver : str
        Solver to use. Options: "closed_form", "gradient_descent",
        "newton", "coordinate_descent".
    regularization : Optional[str]
        Regularization type. Options: None, "l1", "l2", "elasticnet".
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable]
        Custom metric function if needed.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = _normalize_data(X, normalizations["X"])
    y_norm = _normalize_data(y.reshape(-1, 1), normalizations["y"]).flatten()

    # Compute distance matrices
    A = _compute_distance_matrix(X_norm, distance_metric)
    B = _compute_distance_matrix(y_norm.reshape(-1, 1), distance_metric).flatten()

    # Solve for limitations
    if solver == "closed_form":
        result = _solve_closed_form(A, B, regularization)
    elif solver == "gradient_descent":
        result = _solve_gradient_descent(A, B, tol, max_iter, regularization)
    elif solver == "newton":
        result = _solve_newton(A, B, tol, max_iter, regularization)
    elif solver == "coordinate_descent":
        result = _solve_coordinate_descent(A, B, tol, max_iter, regularization)
    else:
        raise ValueError("Invalid solver specified.")

    # Compute metrics
    metrics = _compute_metrics(X_norm, y_norm, result, custom_metric)

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalizations": normalizations,
            "distance_metric": distance_metric,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data based on specified method."""
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
        raise ValueError("Invalid normalization method specified.")

def _compute_distance_matrix(data: np.ndarray, metric: Union[str, Callable]) -> np.ndarray:
    """Compute distance matrix based on specified metric."""
    n = data.shape[0]
    if callable(metric):
        return np.array([[metric(x, y) for x in data] for y in data])
    elif metric == "euclidean":
        return np.sqrt(np.sum((data[:, np.newaxis] - data) ** 2, axis=2))
    elif metric == "manhattan":
        return np.sum(np.abs(data[:, np.newaxis] - data), axis=2)
    elif metric == "cosine":
        return 1 - np.dot(data, data.T) / (np.linalg.norm(data, axis=1)[:, np.newaxis] * np.linalg.norm(data, axis=1))
    elif metric == "minkowski":
        return np.sum(np.abs(data[:, np.newaxis] - data) ** 3, axis=2) ** (1/3)
    else:
        raise ValueError("Invalid distance metric specified.")

def _solve_closed_form(A: np.ndarray, B: np.ndarray, regularization: Optional[str]) -> np.ndarray:
    """Solve limitations using closed form solution."""
    if regularization is None:
        return np.linalg.pinv(A) @ B
    elif regularization == "l1":
        return _solve_lasso(A, B)
    elif regularization == "l2":
        return np.linalg.inv(A.T @ A + 1e-4 * np.eye(A.shape[1])) @ A.T @ B
    elif regularization == "elasticnet":
        return _solve_elasticnet(A, B)
    else:
        raise ValueError("Invalid regularization method specified.")

def _solve_gradient_descent(
    A: np.ndarray,
    B: np.ndarray,
    tol: float,
    max_iter: int,
    regularization: Optional[str]
) -> np.ndarray:
    """Solve limitations using gradient descent."""
    n_features = A.shape[1]
    theta = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * A.T @ (A @ theta - B) / len(B)
        if regularization == "l1":
            gradient += np.sign(theta)
        elif regularization == "l2":
            gradient += 2 * theta
        theta -= tol * gradient
    return theta

def _solve_newton(
    A: np.ndarray,
    B: np.ndarray,
    tol: float,
    max_iter: int,
    regularization: Optional[str]
) -> np.ndarray:
    """Solve limitations using Newton's method."""
    n_features = A.shape[1]
    theta = np.zeros(n_features)
    for _ in range(max_iter):
        residual = A @ theta - B
        gradient = 2 * A.T @ residual / len(B)
        hessian = 2 * A.T @ A / len(B)
        if regularization == "l1":
            hessian += np.diag(np.where(theta > 0, 1, 0))
        elif regularization == "l2":
            hessian += 2 * np.eye(n_features)
        theta -= np.linalg.pinv(hessian) @ gradient
    return theta

def _solve_coordinate_descent(
    A: np.ndarray,
    B: np.ndarray,
    tol: float,
    max_iter: int,
    regularization: Optional[str]
) -> np.ndarray:
    """Solve limitations using coordinate descent."""
    n_features = A.shape[1]
    theta = np.zeros(n_features)
    for _ in range(max_iter):
        for j in range(n_features):
            Aj = A[:, j]
            residual = B - np.dot(A, theta) + theta[j] * Aj
            if regularization == "l1":
                theta[j] = np.sign(np.dot(Aj, residual)) * np.maximum(
                    0, np.abs(np.dot(Aj, residual)) - tol
                ) / (np.dot(Aj, Aj) + 1e-8)
            else:
                theta[j] = np.dot(Aj, residual) / (np.dot(Aj, Aj) + 1e-8)
    return theta

def _solve_lasso(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve L1-regularized limitations."""
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=1e-4)
    model.fit(A, B)
    return model.coef_

def _solve_elasticnet(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve elastic net regularized limitations."""
    from sklearn.linear_model import ElasticNet
    model = ElasticNet(alpha=1e-4, l1_ratio=0.5)
    model.fit(A, B)
    return model.coef_

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    result: np.ndarray,
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Compute metrics for limitations."""
    y_pred = X @ result
    metrics = {
        "mse": np.mean((y - y_pred) ** 2),
        "mae": np.mean(np.abs(y - y_pred)),
        "r2": 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    }
    if custom_metric is not None:
        metrics["custom"] = custom_metric(y, y_pred)
    return metrics

################################################################################
# applications
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def applications_fit(
    X: np.ndarray,
    y: np.ndarray,
    distance_metric: Union[str, Callable] = 'euclidean',
    normalization: str = 'none',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute distance correlation applications between features and target.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    distance_metric : str or callable
        Distance metric to use ('euclidean', 'manhattan', 'cosine', etc.)
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    solver : str
        Solver method ('closed_form', 'gradient_descent', etc.)
    regularization : str or None
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    custom_metric : callable or None
        Custom metric function if needed

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = applications_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if needed
    X_norm = _apply_normalization(X, normalization)
    y_norm = _apply_normalization(y.reshape(-1, 1), normalization).flatten()

    # Choose distance metric
    distance_func = _get_distance_function(distance_metric)

    # Compute distance matrices
    A, B = _compute_distance_matrices(X_norm, y_norm, distance_func)

    # Compute distance covariance and correlation
    dcov = _compute_distance_covariance(A, B)
    dcor = _compute_distance_correlation(dcov)

    # Solve for parameters if needed
    params = {}
    if solver == 'closed_form':
        params = _solve_closed_form(X_norm, y_norm)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(X_norm, y_norm, tol=tol, max_iter=max_iter)

    # Apply regularization if needed
    if regularization is not None:
        params = _apply_regularization(params, X_norm.shape[1], regularization)

    # Compute metrics
    metrics = _compute_metrics(y_norm, params.get('predictions', None), custom_metric)

    return {
        'result': dcor,
        'metrics': metrics,
        'params_used': {
            'distance_metric': distance_metric,
            'normalization': normalization,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
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

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to data."""
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

def _get_distance_function(metric: Union[str, Callable]) -> Callable:
    """Get distance function based on metric name or return custom callable."""
    if isinstance(metric, str):
        if metric == 'euclidean':
            return lambda x, y: np.linalg.norm(x - y)
        elif metric == 'manhattan':
            return lambda x, y: np.sum(np.abs(x - y))
        elif metric == 'cosine':
            return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
    elif callable(metric):
        return metric
    else:
        raise TypeError("distance_metric must be a string or callable")

def _compute_distance_matrices(X: np.ndarray, y: np.ndarray, distance_func: Callable) -> tuple:
    """Compute double centered distance matrices."""
    n = X.shape[0]
    A = np.zeros((n, n))
    B = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            A[i, j] = distance_func(X[i], X[j])
            B[i, j] = distance_func(y[i], y[j])

    A = _double_center(A)
    B = _double_center(B)

    return A, B

def _double_center(matrix: np.ndarray) -> np.ndarray:
    """Double centering of a matrix."""
    n = matrix.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ matrix @ H

def _compute_distance_covariance(A: np.ndarray, B: np.ndarray) -> float:
    """Compute distance covariance."""
    n = A.shape[0]
    return np.sqrt((1 / (n ** 2)) * np.sum(A * B))

def _compute_distance_correlation(dcov: float) -> float:
    """Compute distance correlation."""
    if dcov == 0:
        return 0.0
    A_var = np.var(np.sqrt(np.sum(A ** 2, axis=1)))
    B_var = np.var(np.sqrt(np.sum(B ** 2, axis=1)))
    return dcov / np.sqrt(A_var * B_var)

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> Dict:
    """Closed form solution for distance correlation."""
    X_pinv = np.linalg.pinv(X)
    coefs = X_pinv @ y
    predictions = X @ coefs
    return {'coefficients': coefs, 'predictions': predictions}

def _solve_gradient_descent(X: np.ndarray, y: np.ndarray, tol: float = 1e-4, max_iter: int = 1000) -> Dict:
    """Gradient descent solver for distance correlation."""
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        predictions = X @ coefs
        error = predictions - y
        gradient = 2 / n_samples * X.T @ error

        new_coefs = coefs - learning_rate * gradient
        if np.linalg.norm(new_coefs - coefs) < tol:
            break
        coefs = new_coefs

    predictions = X @ coefs
    return {'coefficients': coefs, 'predictions': predictions}

def _apply_regularization(params: Dict, n_features: int, method: str) -> Dict:
    """Apply regularization to parameters."""
    if method == 'l1':
        params['coefficients'] = np.sign(params['coefficients']) * np.maximum(
            np.abs(params['coefficients']) - 1, 0)
    elif method == 'l2':
        params['coefficients'] /= (1 + np.linalg.norm(params['coefficients']))
    elif method == 'elasticnet':
        l1_ratio = 0.5
        params['coefficients'] = np.sign(params['coefficients']) * np.maximum(
            np.abs(params['coefficients']) - l1_ratio, 0) / (1 + (1 - l1_ratio))
    return params

def _compute_metrics(y_true: np.ndarray, y_pred: Optional[np.ndarray], custom_metric: Optional[Callable]) -> Dict:
    """Compute various metrics."""
    metrics = {}

    if y_pred is not None:
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        metrics['r2'] = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)

    return metrics

################################################################################
# implementation_python
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def implementation_python_compute(
    X: np.ndarray,
    Y: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = euclidean_distance,
    normalization: str = 'none',
    alpha: float = 0.5,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute the distance correlation between two datasets X and Y.

    Parameters:
    -----------
    X : np.ndarray
        First dataset of shape (n_samples, n_features)
    Y : np.ndarray
        Second dataset of shape (n_samples, n_features)
    distance_metric : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Distance metric function (default: euclidean_distance)
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    alpha : float
        Regularization parameter for distance metric (default: 0.5)
    random_state : Optional[int]
        Random seed for reproducibility

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'result': computed distance correlation
        - 'metrics': additional metrics
        - 'params_used': parameters used in computation
        - 'warnings': any warnings encountered

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> Y = np.random.rand(100, 3)
    >>> result = implementation_python_compute(X, Y)
    """
    # Validate inputs
    params_used = {
        'distance_metric': distance_metric.__name__,
        'normalization': normalization,
        'alpha': alpha
    }

    warnings = []
    X, Y, warn = _validate_inputs(X, Y)
    if warn:
        warnings.append(warn)

    # Normalize data
    X_norm, Y_norm = _apply_normalization(X, Y, normalization)

    # Compute distance matrices
    A = distance_metric(X_norm)
    B = distance_metric(Y_norm)

    # Compute distance covariance
    dcov = _compute_distance_covariance(A, B)

    # Compute distance variances
    dvar_X = _compute_distance_variance(A)
    dvar_Y = _compute_distance_variance(B)

    # Compute distance correlation
    if dvar_X == 0 or dvar_Y == 0:
        result = 0.0
    else:
        result = dcov / np.sqrt(dvar_X * dvar_Y)

    # Compute metrics
    metrics = {
        'distance_covariance': dcov,
        'distance_variance_X': dvar_X,
        'distance_variance_Y': dvar_Y
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(
    X: np.ndarray,
    Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Validate input arrays X and Y.

    Parameters:
    -----------
    X : np.ndarray
        First dataset
    Y : np.ndarray
        Second dataset

    Returns:
    --------
    tuple[np.ndarray, np.ndarray, str]
        Validated X and Y arrays, and warning message if any
    """
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")

    if np.isnan(X).any() or np.isnan(Y).any():
        return X, Y, "Input arrays contain NaN values"

    if np.isinf(X).any() or np.isinf(Y).any():
        return X, Y, "Input arrays contain infinite values"

    return X, Y, ""

def _apply_normalization(
    X: np.ndarray,
    Y: np.ndarray,
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply normalization to input arrays.

    Parameters:
    -----------
    X : np.ndarray
        First dataset
    Y : np.ndarray
        Second dataset
    method : str
        Normalization method

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        Normalized X and Y arrays
    """
    if method == 'none':
        return X, Y
    elif method == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
    elif method == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        Y = (Y - np.min(Y, axis=0)) / (np.max(Y, axis=0) - np.min(Y, axis=0))
    elif method == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        Y = (Y - np.median(Y, axis=0)) / (np.percentile(Y, 75, axis=0) - np.percentile(Y, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X, Y

def _compute_distance_covariance(
    A: np.ndarray,
    B: np.ndarray
) -> float:
    """
    Compute distance covariance between two distance matrices.

    Parameters:
    -----------
    A : np.ndarray
        Distance matrix for first dataset
    B : np.ndarray
        Distance matrix for second dataset

    Returns:
    --------
    float
        Computed distance covariance
    """
    n = A.shape[0]
    A_bar = (A.sum(axis=0) / n).reshape(1, -1)
    B_bar = (B.sum(axis=0) / n).reshape(1, -1)
    A_dblbar = (A.sum() / (n ** 2))
    B_dblbar = (B.sum() / (n ** 2))

    dcov = np.sqrt(
        ((A - A_bar - A_bar.T + A_dblbar) * (B - B_bar - B_bar.T + B_dblbar)).sum() / (n ** 2)
    )

    return dcov

def _compute_distance_variance(
    A: np.ndarray
) -> float:
    """
    Compute distance variance for a distance matrix.

    Parameters:
    -----------
    A : np.ndarray
        Distance matrix

    Returns:
    --------
    float
        Computed distance variance
    """
    n = A.shape[0]
    A_bar = (A.sum(axis=0) / n).reshape(1, -1)
    A_dblbar = (A.sum() / (n ** 2))

    dvar = np.sqrt(
        ((A - A_bar - A_bar.T + A_dblbar) ** 2).sum() / (n ** 2)
    )

    return dvar

def euclidean_distance(
    X: np.ndarray
) -> np.ndarray:
    """
    Compute Euclidean distance matrix.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)

    Returns:
    --------
    np.ndarray
        Distance matrix of shape (n_samples, n_samples)
    """
    return np.sqrt(((X[:, np.newaxis] - X) ** 2).sum(axis=2))

################################################################################
# comparison_with_pearson
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def comparison_with_pearson_fit(
    X: np.ndarray,
    y: np.ndarray,
    distance_metric: str = 'euclidean',
    normalization: str = 'none',
    pearson_metric: Optional[Callable] = None,
    distance_correlation_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[float, Dict]]:
    """
    Compare Pearson correlation with distance correlation.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    distance_metric : str, optional
        Distance metric to use for distance correlation ('euclidean', 'manhattan', etc.).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    pearson_metric : Callable, optional
        Custom Pearson correlation function.
    distance_correlation_metric : Callable, optional
        Custom distance correlation function.

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing:
        - 'pearson_correlation': float
        - 'distance_correlation': float
        - 'metrics': Dict[str, float]
        - 'params_used': Dict[str, str]
        - 'warnings': List[str]

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = comparison_with_pearson_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_norm, y_norm = _apply_normalization(X, y, normalization)

    # Calculate Pearson correlation
    pearson_corr = _calculate_pearson(X_norm, y_norm, metric=pearson_metric)

    # Calculate distance correlation
    dc_corr = _calculate_distance_correlation(X_norm, y_norm,
                                             metric=distance_metric,
                                             custom_metric=distance_correlation_metric)

    # Calculate additional metrics
    metrics = _calculate_comparison_metrics(pearson_corr, dc_corr)

    # Prepare output
    result = {
        'pearson_correlation': pearson_corr,
        'distance_correlation': dc_corr,
        'metrics': metrics,
        'params_used': {
            'distance_metric': distance_metric,
            'normalization': normalization
        },
        'warnings': _check_warnings(X_norm, y_norm)
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalization to input arrays."""
    if method == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y = (y - np.mean(y)) / np.std(y)
    elif method == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    return X, y

def _calculate_pearson(
    X: np.ndarray,
    y: np.ndarray,
    metric: Optional[Callable] = None
) -> float:
    """Calculate Pearson correlation."""
    if metric is not None:
        return metric(X, y)
    # Default implementation
    cov = np.cov(X.flatten(), y)
    return cov[0, 1] / (np.std(X) * np.std(y))

def _calculate_distance_correlation(
    X: np.ndarray,
    y: np.ndarray,
    metric: str = 'euclidean',
    custom_metric: Optional[Callable] = None
) -> float:
    """Calculate distance correlation."""
    if custom_metric is not None:
        return custom_metric(X, y)

    # Default implementation using specified distance metric
    def _distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if metric == 'euclidean':
            return np.sqrt(np.sum((a[:, None] - b) ** 2, axis=2))
        elif metric == 'manhattan':
            return np.sum(np.abs(a[:, None] - b), axis=2)
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")

    # Calculate distance matrices
    a = X[:, None, :]
    b = y[None, :, None]

    d_X = _distance_matrix(a, a)
    d_y = _distance_matrix(b, b)

    # Calculate distance covariance
    dcov = np.sqrt(_dcov(d_X, d_y))
    dvarX = np.sqrt(_dvar(d_X))
    dvary = np.sqrt(_dvar(d_y))

    if dvarX == 0 or dvary == 0:
        return 0.0

    return dcov / (dvarX * dvary)

def _dcov(A: np.ndarray, B: np.ndarray) -> float:
    """Calculate distance covariance."""
    n = A.shape[0]
    A_sq = (A ** 2).sum(axis=1)
    B_sq = (B ** 2).sum(axis=1)

    row_means_A = np.mean(A, axis=0)
    col_means_A = np.mean(A, axis=1)

    row_means_B = np.mean(B, axis=0)
    col_means_B = np.mean(B, axis=1)

    A_centered = A - row_means_A[:, None] - col_means_A[None, :] + np.mean(row_means_A)
    B_centered = B - row_means_B[:, None] - col_means_B[None, :] + np.mean(row_means_B)

    return np.sqrt((A_centered * B_centered).sum() / (n ** 2))

def _dvar(A: np.ndarray) -> float:
    """Calculate distance variance."""
    n = A.shape[0]
    A_sq = (A ** 2).sum(axis=1)

    row_means_A = np.mean(A, axis=0)
    col_means_A = np.mean(A, axis=1)

    A_centered = A - row_means_A[:, None] - col_means_A[None, :] + np.mean(row_means_A)

    return (A_centered ** 2).sum() / (n ** 2)

def _calculate_comparison_metrics(
    pearson: float,
    distance_corr: float
) -> Dict[str, float]:
    """Calculate comparison metrics between Pearson and distance correlation."""
    return {
        'absolute_difference': abs(pearson - distance_corr),
        'relative_difference': (distance_corr - pearson) / max(abs(pearson), abs(distance_corr)),
        'ratio': distance_corr / pearson if pearson != 0 else float('inf')
    }

def _check_warnings(
    X: np.ndarray,
    y: np.ndarray
) -> list[str]:
    """Check for potential warnings."""
    warnings = []

    if np.any(np.std(X, axis=0) == 0):
        warnings.append("Some features have zero variance")
    if np.std(y) == 0:
        warnings.append("Target variable has zero variance")

    return warnings
