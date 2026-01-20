"""
Quantix – Module selection_variables
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# variable_categorielle
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def variable_categorielle_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalisation: str = 'none',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularisation: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Fonction principale pour la sélection de variables catégorielles.

    Parameters
    ----------
    X : np.ndarray
        Matrice des variables catégorielles (n_samples, n_features).
    y : np.ndarray
        Vecteur cible (n_samples,).
    normalisation : str, optional
        Type de normalisation à appliquer ('none', 'standard', 'minmax', 'robust').
    distance_metric : str or callable, optional
        Métrique de distance à utiliser ('euclidean', 'manhattan', 'cosine', 'minkowski') ou fonction personnalisée.
    solver : str, optional
        Solveur à utiliser ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularisation : str, optional
        Type de régularisation ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolérance pour la convergence.
    max_iter : int, optional
        Nombre maximal d'itérations.
    custom_metric : callable, optional
        Fonction de métrique personnalisée.

    Returns
    -------
    Dict
        Dictionnaire contenant les résultats, métriques, paramètres utilisés et avertissements.
    """
    # Validation des entrées
    _validate_inputs(X, y)

    # Normalisation
    X_normalised = _apply_normalisation(X, normalisation)

    # Calcul du critère
    criterion_value = _compute_criterion(X_normalised, y, distance_metric)

    # Estimation des paramètres
    params = _estimate_parameters(X_normalised, y, solver, regularisation, tol, max_iter)

    # Calcul des métriques
    metrics = _compute_metrics(X_normalised, y, params, custom_metric)

    # Retourne le dictionnaire structuré
    return {
        "result": criterion_value,
        "metrics": metrics,
        "params_used": params,
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validation des entrées."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X doit être une matrice 2D et y un vecteur 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X et y doivent avoir le même nombre d'échantillons.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contient des valeurs NaN ou inf.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contient des valeurs NaN ou inf.")

def _apply_normalisation(X: np.ndarray, normalisation: str) -> np.ndarray:
    """Application de la normalisation."""
    if normalisation == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalisation == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalisation == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X

def _compute_criterion(X: np.ndarray, y: np.ndarray, distance_metric: Union[str, Callable]) -> float:
    """Calcul du critère de sélection."""
    if isinstance(distance_metric, str):
        if distance_metric == 'euclidean':
            distances = np.linalg.norm(X - y[:, np.newaxis], axis=1)
        elif distance_metric == 'manhattan':
            distances = np.sum(np.abs(X - y[:, np.newaxis]), axis=1)
        elif distance_metric == 'cosine':
            distances = 1 - np.dot(X, y) / (np.linalg.norm(X, axis=1) * np.linalg.norm(y))
        elif distance_metric == 'minkowski':
            distances = np.sum(np.abs(X - y[:, np.newaxis])**2, axis=1)**(1/2)
        else:
            raise ValueError("Métrique de distance non reconnue.")
    else:
        distances = distance_metric(X, y)
    return np.mean(distances)

def _estimate_parameters(
    X: np.ndarray,
    y: np.ndarray,
    solver: str,
    regularisation: Optional[str],
    tol: float,
    max_iter: int
) -> Dict:
    """Estimation des paramètres."""
    params = {}
    if solver == 'closed_form':
        params['coefficients'] = np.linalg.pinv(X) @ y
    elif solver == 'gradient_descent':
        params['coefficients'] = _gradient_descent(X, y, tol, max_iter)
    elif solver == 'newton':
        params['coefficients'] = _newton_method(X, y, tol, max_iter)
    elif solver == 'coordinate_descent':
        params['coefficients'] = _coordinate_descent(X, y, tol, max_iter)
    else:
        raise ValueError("Solveur non reconnu.")
    return params

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict,
    custom_metric: Optional[Callable]
) -> Dict:
    """Calcul des métriques."""
    metrics = {}
    y_pred = X @ params['coefficients']
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y, y_pred)
    else:
        metrics['mse'] = np.mean((y - y_pred)**2)
        metrics['mae'] = np.mean(np.abs(y - y_pred))
        metrics['r2'] = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    return metrics

def _gradient_descent(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Solveur par descente de gradient."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    learning_rate = 0.01
    for _ in range(max_iter):
        gradient = X.T @ (X @ coefficients - y) / len(y)
        new_coefficients = coefficients - learning_rate * gradient
        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients
    return coefficients

def _newton_method(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Solveur par méthode de Newton."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = X.T @ (X @ coefficients - y) / len(y)
        hessian = X.T @ X / len(y)
        new_coefficients = coefficients - np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients
    return coefficients

def _coordinate_descent(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Solveur par descente de coordonnées."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - X @ coefficients + coefficients[j] * X_j
            coefficients[j] = np.sum(X_j * residuals) / np.sum(X_j**2)
        if np.linalg.norm(coefficients - coefficients) < tol:
            break
    return coefficients

################################################################################
# variable_continue
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def variable_continue_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Select continuous variables using various statistical methods.

    Parameters:
    -----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    normalizer : Optional[Callable]
        Function to normalize features. Default is None.
    metric : Union[str, Callable]
        Metric to evaluate performance. Can be 'mse', 'mae', 'r2', or custom callable.
    distance : Union[str, Callable]
        Distance metric for feature selection. Can be 'euclidean', 'manhattan', etc.
    solver : str
        Solver to use. Options: 'closed_form', 'gradient_descent', etc.
    regularization : Optional[str]
        Regularization type. Options: 'l1', 'l2', 'elasticnet'.
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

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Select solver and compute
    if solver == 'closed_form':
        result = _closed_form_solver(X_normalized, y, regularization)
    elif solver == 'gradient_descent':
        result = _gradient_descent_solver(X_normalized, y, metric, tol, max_iter, regularization)
    else:
        raise ValueError(f"Solver {solver} not supported.")

    # Compute metrics
    metrics = _compute_metrics(y, result['predictions'], metric, custom_metric)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return output

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    return normalizer(X)

def _closed_form_solver(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str] = None
) -> Dict:
    """Closed form solution for feature selection."""
    if regularization is None:
        beta = np.linalg.pinv(X) @ y
    elif regularization == 'l2':
        beta = np.linalg.inv(X.T @ X + 1e-6 * np.eye(X.shape[1])) @ X.T @ y
    else:
        raise ValueError(f"Regularization {regularization} not supported in closed form.")

    predictions = X @ beta
    return {'coefficients': beta, 'predictions': predictions}

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    tol: float,
    max_iter: int,
    regularization: Optional[str] = None
) -> Dict:
    """Gradient descent solver for feature selection."""
    n_features = X.shape[1]
    beta = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, beta, metric, regularization)
        beta -= 0.01 * gradient
        current_loss = _compute_loss(y, X @ beta, metric)

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    predictions = X @ beta
    return {'coefficients': beta, 'predictions': predictions}

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    metric: Union[str, Callable],
    regularization: Optional[str] = None
) -> np.ndarray:
    """Compute gradient for gradient descent."""
    predictions = X @ beta
    if metric == 'mse':
        gradient = -2 * X.T @ (y - predictions)
    elif metric == 'mae':
        gradient = -X.T @ np.sign(y - predictions)
    else:
        raise ValueError(f"Metric {metric} not supported for gradient computation.")

    if regularization == 'l2':
        gradient += 2 * 1e-4 * beta
    elif regularization == 'l1':
        gradient += np.sign(beta) * 1e-4

    return gradient

def _compute_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Compute loss based on specified metric."""
    if metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    else:
        raise ValueError(f"Metric {metric} not supported for loss computation.")

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute evaluation metrics."""
    metrics_dict = {}

    if metric == 'mse':
        metrics_dict['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics_dict['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        metrics_dict['r2'] = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    elif callable(metric):
        metrics_dict['custom'] = metric(y_true, y_pred)

    if custom_metric is not None:
        metrics_dict['custom'] = custom_metric(y_true, y_pred)

    return metrics_dict

################################################################################
# variable_binaire
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def variable_binaire_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a binary variable selection model.

    Parameters:
    -----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target binary vector of shape (n_samples,).
    normalizer : Optional[Callable]
        Function to normalize the input features.
    metric : Union[str, Callable]
        Metric to evaluate the model. Can be "mse", "mae", "r2", or a custom callable.
    distance : str
        Distance metric for the solver. Can be "euclidean", "manhattan", "cosine", or "minkowski".
    solver : str
        Solver to use. Can be "closed_form", "gradient_descent", "newton", or "coordinate_descent".
    regularization : Optional[str]
        Regularization type. Can be "l1", "l2", or "elasticnet".
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
    Dict
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if a normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Choose metric
    if isinstance(metric, str):
        metric_func = _get_metric(metric)
    else:
        metric_func = metric

    # Choose distance
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance(distance)

    # Choose solver and fit the model
    if solver == "closed_form":
        params = _closed_form_solver(X_normalized, y)
    elif solver == "gradient_descent":
        params = _gradient_descent_solver(X_normalized, y, distance_func, tol, max_iter)
    elif solver == "newton":
        params = _newton_solver(X_normalized, y, distance_func, tol, max_iter)
    elif solver == "coordinate_descent":
        params = _coordinate_descent_solver(X_normalized, y, distance_func, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization is not None:
        params = _apply_regularization(params, X_normalized, y, regularization)

    # Calculate metrics
    metrics = _calculate_metrics(y, np.dot(X_normalized, params), metric_func)

    # Prepare the result dictionary
    result = {
        "result": params,
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
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to the input data."""
    if normalizer is None:
        return X
    return normalizer(X)

def _get_metric(metric: str) -> Callable:
    """Get the metric function based on the input string."""
    metrics = {
        "mse": _mean_squared_error,
        "mae": _mean_absolute_error,
        "r2": _r_squared
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance(distance: str) -> Callable:
    """Get the distance function based on the input string."""
    distances = {
        "euclidean": _euclidean_distance,
        "manhattan": _manhattan_distance,
        "cosine": _cosine_distance,
        "minkowski": _minkowski_distance
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form solution for binary variable selection."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent solver for binary variable selection."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y)
        params -= tol * gradient
    return params

def _newton_solver(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Newton's method solver for binary variable selection."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y)
        hessian = 2 * X.T @ X
        params -= np.linalg.inv(hessian) @ gradient
    return params

def _coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Coordinate descent solver for binary variable selection."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            params[j] = (y - np.dot(X, params) + params[j] * X_j).T @ X_j / (X_j.T @ X_j)
    return params

def _apply_regularization(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    regularization: str
) -> np.ndarray:
    """Apply regularization to the parameters."""
    if regularization == "l1":
        return _apply_l1_regularization(params, X, y)
    elif regularization == "l2":
        return _apply_l2_regularization(params, X, y)
    elif regularization == "elasticnet":
        return _apply_elasticnet_regularization(params, X, y)
    else:
        raise ValueError(f"Unknown regularization: {regularization}")

def _apply_l1_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Apply L1 regularization."""
    return params - 0.1 * np.sign(params)

def _apply_l2_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Apply L2 regularization."""
    return params - 0.1 * params

def _apply_elasticnet_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Apply elastic net regularization."""
    return params - 0.1 * (np.sign(params) + params)

def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric_func: Callable) -> Dict:
    """Calculate the metrics for the model."""
    return {"metric": metric_func(y_true, y_pred)}

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the R-squared metric."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _minkowski_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the Minkowski distance."""
    return np.sum(np.abs(a - b) ** 3) ** (1/3)

################################################################################
# variable_ordinale
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def variable_ordinale_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalisation: str = 'standard',
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
    Fit an ordinal variable selection model.

    Parameters:
    -----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    normalisation : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : str or callable, optional
        Metric to optimize: 'mse', 'mae', 'r2', 'logloss', or custom callable.
    distance : str, optional
        Distance metric: 'euclidean', 'manhattan', 'cosine', or 'minkowski'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : str, optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float, optional
        Tolerance for stopping criteria.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    custom_distance : callable, optional
        Custom distance function if not using built-in distances.

    Returns:
    --------
    Dict containing:
        - 'result': Fitted model parameters.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _normalize_data(X, normalisation)

    # Select solver and fit model
    if solver == 'closed_form':
        params = _closed_form_solver(X_normalized, y, distance, regularization)
    elif solver == 'gradient_descent':
        params = _gradient_descent_solver(X_normalized, y, metric, distance,
                                         regularization, tol, max_iter)
    elif solver == 'newton':
        params = _newton_solver(X_normalized, y, metric, distance,
                               regularization, tol, max_iter)
    elif solver == 'coordinate_descent':
        params = _coordinate_descent_solver(X_normalized, y, metric, distance,
                                           regularization, tol, max_iter)
    else:
        raise ValueError("Unsupported solver method.")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, params, metric, custom_metric)

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
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
        raise ValueError("Unsupported normalization method.")

def _closed_form_solver(X: np.ndarray, y: np.ndarray,
                        distance: str, regularization: Optional[str]) -> np.ndarray:
    """Closed form solution for ordinal variable selection."""
    # Implement closed form solution based on distance and regularization
    pass

def _gradient_descent_solver(X: np.ndarray, y: np.ndarray,
                             metric: Union[str, Callable],
                             distance: str,
                             regularization: Optional[str],
                             tol: float, max_iter: int) -> np.ndarray:
    """Gradient descent solver for ordinal variable selection."""
    # Implement gradient descent based on metric, distance, and regularization
    pass

def _newton_solver(X: np.ndarray, y: np.ndarray,
                   metric: Union[str, Callable],
                   distance: str,
                   regularization: Optional[str],
                   tol: float, max_iter: int) -> np.ndarray:
    """Newton's method solver for ordinal variable selection."""
    # Implement Newton's method based on metric, distance, and regularization
    pass

def _coordinate_descent_solver(X: np.ndarray, y: np.ndarray,
                               metric: Union[str, Callable],
                               distance: str,
                               regularization: Optional[str],
                               tol: float, max_iter: int) -> np.ndarray:
    """Coordinate descent solver for ordinal variable selection."""
    # Implement coordinate descent based on metric, distance, and regularization
    pass

def _compute_metrics(X: np.ndarray, y: np.ndarray,
                     params: np.ndarray,
                     metric: Union[str, Callable],
                     custom_metric: Optional[Callable]) -> Dict:
    """Compute metrics for the fitted model."""
    if custom_metric is not None:
        return {'custom': custom_metric(X, y, params)}

    metrics_dict = {}
    if metric == 'mse':
        metrics_dict['mse'] = np.mean((X @ params - y) ** 2)
    elif metric == 'mae':
        metrics_dict['mae'] = np.mean(np.abs(X @ params - y))
    elif metric == 'r2':
        ss_res = np.sum((X @ params - y) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics_dict['r2'] = 1 - (ss_res / ss_tot)
    elif metric == 'logloss':
        # Implement log loss calculation
        pass
    else:
        raise ValueError("Unsupported metric.")

    return metrics_dict

# Example usage
"""
X = np.random.rand(100, 5)
y = np.random.rand(100)

result = variable_ordinale_fit(
    X, y,
    normalisation='standard',
    metric='mse',
    distance='euclidean',
    solver='gradient_descent'
)
"""

################################################################################
# variable_numerique
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def variable_numerique_fit(
    X: np.ndarray,
    y: np.ndarray,
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
    Select numerical variables based on given criteria and parameters.

    Parameters:
    -----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    normalisation : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to optimize ('mse', 'mae', 'r2', 'logloss') or custom callable.
    distance : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    custom_distance : callable, optional
        Custom distance function if not using built-in distances.

    Returns:
    --------
    Dict containing:
        - 'result': Selected variables or model coefficients.
        - 'metrics': Computed metrics dictionary.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings encountered.

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = variable_numerique_fit(X, y, normalisation='standard', metric='r2')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalization(X, normalisation)

    # Prepare metric and distance functions
    metric_func = _get_metric_function(metric, custom_metric)
    distance_func = _get_distance_function(distance, custom_distance)

    # Solve for variables
    result, params_used = _solve_variables(
        X_normalized, y,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Compute metrics
    metrics = _compute_metrics(y, result['predictions'], metric_func)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
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

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the input data."""
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

def _get_metric_function(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Get the metric function based on input parameters."""
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
        raise ValueError(f"Unknown metric: {metric}")

def _get_distance_function(distance: Union[str, Callable], custom_distance: Optional[Callable]) -> Callable:
    """Get the distance function based on input parameters."""
    if custom_distance is not None:
        return custom_distance
    elif distance == 'euclidean':
        return _euclidean_distance
    elif distance == 'manhattan':
        return _manhattan_distance
    elif distance == 'cosine':
        return _cosine_distance
    elif distance == 'minkowski':
        return lambda x, y: _minkowski_distance(x, y, p=3)
    else:
        raise ValueError(f"Unknown distance: {distance}")

def _solve_variables(
    X: np.ndarray,
    y: np.ndarray,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple:
    """Solve for variables using specified solver and regularization."""
    if solver == 'closed_form':
        return _solve_closed_form(X, y, regularization)
    elif solver == 'gradient_descent':
        return _solve_gradient_descent(X, y, regularization, tol, max_iter)
    elif solver == 'newton':
        return _solve_newton(X, y, regularization, tol, max_iter)
    elif solver == 'coordinate_descent':
        return _solve_coordinate_descent(X, y, regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric_func: Callable) -> Dict:
    """Compute metrics for the given predictions."""
    return {
        'metric_value': metric_func(y_true, y_pred),
        'r2_score': _r_squared(y_true, y_pred)
    }

# Example of metric functions
def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example of distance functions
def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Euclidean distance."""
    return np.linalg.norm(x - y)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Manhattan distance."""
    return np.sum(np.abs(x - y))

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine distance."""
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def _minkowski_distance(x: np.ndarray, y: np.ndarray, p: int) -> float:
    """Compute Minkowski distance."""
    return np.sum(np.abs(x - y) ** p) ** (1 / p)

# Example of solver functions
def _solve_closed_form(X: np.ndarray, y: np.ndarray, regularization: Optional[str]) -> tuple:
    """Solve using closed form solution."""
    if regularization == 'none':
        coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
    elif regularization == 'l2':
        lambda_ = 1.0  # Default value, should be parameterized
        coefficients = np.linalg.inv(X.T @ X + lambda_ * np.eye(X.shape[1])) @ X.T @ y
    else:
        raise ValueError(f"Unsupported regularization for closed form: {regularization}")

    predictions = X @ coefficients
    return {
        'coefficients': coefficients,
        'predictions': predictions
    }, {
        'solver': 'closed_form',
        'regularization': regularization
    }

def _solve_gradient_descent(X: np.ndarray, y: np.ndarray, regularization: Optional[str], tol: float, max_iter: int) -> tuple:
    """Solve using gradient descent."""
    # Implementation of gradient descent
    coefficients = np.zeros(X.shape[1])
    learning_rate = 0.01  # Default value, should be parameterized
    for _ in range(max_iter):
        gradient = X.T @ (X @ coefficients - y)
        if regularization == 'l2':
            gradient += 1.0 * coefficients  # Default lambda, should be parameterized
        coefficients -= learning_rate * gradient
    predictions = X @ coefficients
    return {
        'coefficients': coefficients,
        'predictions': predictions
    }, {
        'solver': 'gradient_descent',
        'regularization': regularization,
        'learning_rate': learning_rate
    }

def _solve_newton(X: np.ndarray, y: np.ndarray, regularization: Optional[str], tol: float, max_iter: int) -> tuple:
    """Solve using Newton's method."""
    # Implementation of Newton's method
    coefficients = np.zeros(X.shape[1])
    for _ in range(max_iter):
        gradient = X.T @ (X @ coefficients - y)
        hessian = X.T @ X
        if regularization == 'l2':
            hessian += 1.0 * np.eye(X.shape[1])  # Default lambda, should be parameterized
        coefficients -= np.linalg.inv(hessian) @ gradient
    predictions = X @ coefficients
    return {
        'coefficients': coefficients,
        'predictions': predictions
    }, {
        'solver': 'newton',
        'regularization': regularization
    }

def _solve_coordinate_descent(X: np.ndarray, y: np.ndarray, regularization: Optional[str], tol: float, max_iter: int) -> tuple:
    """Solve using coordinate descent."""
    # Implementation of coordinate descent
    coefficients = np.zeros(X.shape[1])
    for _ in range(max_iter):
        for i in range(X.shape[1]):
            X_i = X[:, i]
            residuals = y - (X @ coefficients) + coefficients[i] * X_i
            if regularization == 'l1':
                coefficients[i] = _soft_threshold(residuals @ X_i, 1.0) / (X_i @ X_i)
            else:
                coefficients[i] = (residuals @ X_i) / (X_i @ X_i)
    predictions = X @ coefficients
    return {
        'coefficients': coefficients,
        'predictions': predictions
    }, {
        'solver': 'coordinate_descent',
        'regularization': regularization
    }

def _soft_threshold(rho: float, lambda_: float) -> float:
    """Soft thresholding function."""
    if rho < -lambda_:
        return rho + lambda_
    elif rho > lambda_:
        return rho - lambda_
    else:
        return 0.0

################################################################################
# variable_temporelle
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def variable_temporelle_fit(
    X: np.ndarray,
    y: np.ndarray,
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
    Select temporal variables using various statistical methods.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features, n_timesteps)
    y : np.ndarray
        Target values of shape (n_samples,)
    normalisation : str, optional
        Normalization method: 'none', 'standard', 'minmax', 'robust'
    metric : str or callable, optional
        Metric to optimize: 'mse', 'mae', 'r2', 'logloss' or custom callable
    distance : str or callable, optional
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'minkowski' or custom callable
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'
    regularization : str, optional
        Regularization type: 'none', 'l1', 'l2', 'elasticnet'
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
    >>> X = np.random.rand(100, 5, 10)
    >>> y = np.random.rand(100)
    >>> result = variable_temporelle_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalization(X, normalisation)

    # Prepare metric and distance functions
    metric_func = _get_metric_function(metric, custom_metric)
    distance_func = _get_distance_function(distance, custom_distance)

    # Solve the problem
    if solver == 'closed_form':
        params = _solve_closed_form(X_normalized, y, regularization)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(X_normalized, y, metric_func,
                                        distance_func, regularization,
                                        tol, max_iter)
    elif solver == 'newton':
        params = _solve_newton(X_normalized, y, metric_func,
                              distance_func, regularization,
                              tol, max_iter)
    elif solver == 'coordinate_descent':
        params = _solve_coordinate_descent(X_normalized, y, metric_func,
                                          distance_func, regularization,
                                          tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(X_normalized, y, params, metric_func)

    # Prepare results
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'metric': metric if isinstance(metric, str) else 'custom',
            'distance': distance if isinstance(distance, str) else 'custom',
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 3:
        raise ValueError("X must be a 3D array (n_samples, n_features, n_timesteps)")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the input data."""
    if method == 'none':
        return X
    elif method == 'standard':
        mean = np.mean(X, axis=(0, 2), keepdims=True)
        std = np.std(X, axis=(0, 2), keepdims=True)
        return (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(X, axis=(0, 2), keepdims=True)
        max_val = np.max(X, axis=(0, 2), keepdims=True)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=(0, 2), keepdims=True)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=(0, 2), keepdims=True))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _get_metric_function(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Get the metric function based on input."""
    if custom_metric is not None:
        return custom_metric
    if metric == 'mse':
        return _mean_squared_error
    elif metric == 'mae':
        return _mean_absolute_error
    elif metric == 'r2':
        return _r_squared
    elif metric == 'logloss':
        return _log_loss
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _get_distance_function(distance: Union[str, Callable], custom_distance: Optional[Callable]) -> Callable:
    """Get the distance function based on input."""
    if custom_distance is not None:
        return custom_distance
    if distance == 'euclidean':
        return _euclidean_distance
    elif distance == 'manhattan':
        return _manhattan_distance
    elif distance == 'cosine':
        return _cosine_distance
    elif distance == 'minkowski':
        return lambda x, y: _minkowski_distance(x, y, p=3)
    else:
        raise ValueError(f"Unknown distance: {distance}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray, regularization: Optional[str]) -> np.ndarray:
    """Solve the problem using closed form solution."""
    # Reshape X to 2D array
    n_samples, n_features, n_timesteps = X.shape
    X_reshaped = X.reshape(n_samples, -1)

    if regularization == 'none':
        params = np.linalg.pinv(X_reshaped) @ y
    elif regularization == 'l1':
        params = _solve_lasso(X_reshaped, y)
    elif regularization == 'l2':
        params = _solve_ridge(X_reshaped, y)
    elif regularization == 'elasticnet':
        params = _solve_elasticnet(X_reshaped, y)
    else:
        raise ValueError(f"Unknown regularization: {regularization}")

    return params.reshape(n_features, n_timesteps)

def _solve_gradient_descent(X: np.ndarray, y: np.ndarray, metric_func: Callable,
                           distance_func: Callable, regularization: Optional[str],
                           tol: float, max_iter: int) -> np.ndarray:
    """Solve the problem using gradient descent."""
    # Reshape X to 2D array
    n_samples, n_features, n_timesteps = X.shape
    X_reshaped = X.reshape(n_samples, -1)
    n_params = X_reshaped.shape[1]

    # Initialize parameters
    params = np.zeros(n_params)

    for _ in range(max_iter):
        gradient = _compute_gradient(X_reshaped, y, params, metric_func,
                                   distance_func, regularization)
        params -= gradient * tol

    return params.reshape(n_features, n_timesteps)

def _solve_newton(X: np.ndarray, y: np.ndarray, metric_func: Callable,
                  distance_func: Callable, regularization: Optional[str],
                  tol: float, max_iter: int) -> np.ndarray:
    """Solve the problem using Newton's method."""
    # Reshape X to 2D array
    n_samples, n_features, n_timesteps = X.shape
    X_reshaped = X.reshape(n_samples, -1)
    n_params = X_reshaped.shape[1]

    # Initialize parameters
    params = np.zeros(n_params)

    for _ in range(max_iter):
        gradient = _compute_gradient(X_reshaped, y, params, metric_func,
                                   distance_func, regularization)
        hessian = _compute_hessian(X_reshaped, y, params, metric_func,
                                  distance_func, regularization)
        params -= np.linalg.pinv(hessian) @ gradient

    return params.reshape(n_features, n_timesteps)

def _solve_coordinate_descent(X: np.ndarray, y: np.ndarray, metric_func: Callable,
                             distance_func: Callable, regularization: Optional[str],
                             tol: float, max_iter: int) -> np.ndarray:
    """Solve the problem using coordinate descent."""
    # Reshape X to 2D array
    n_samples, n_features, n_timesteps = X.shape
    X_reshaped = X.reshape(n_samples, -1)
    n_params = X_reshaped.shape[1]

    # Initialize parameters
    params = np.zeros(n_params)

    for _ in range(max_iter):
        for i in range(n_params):
            # Create a copy of params with zero at position i
            params_copy = params.copy()
            params_copy[i] = 0

            # Compute gradient for parameter i
            gradient_i = _compute_gradient_component(X_reshaped, y, params_copy,
                                                   i, metric_func,
                                                   distance_func, regularization)

            # Update parameter
            params[i] -= gradient_i * tol

    return params.reshape(n_features, n_timesteps)

def _calculate_metrics(X: np.ndarray, y: np.ndarray, params: np.ndarray,
                      metric_func: Callable) -> Dict:
    """Calculate various metrics for the solution."""
    # Reshape X to 2D array
    n_samples, n_features, n_timesteps = X.shape
    X_reshaped = X.reshape(n_samples, -1)

    # Calculate predictions
    y_pred = X_reshaped @ params.flatten()

    metrics = {
        'metric': metric_func(y, y_pred),
        'mse': _mean_squared_error(y, y_pred),
        'mae': _mean_absolute_error(y, y_pred),
        'r2': _r_squared(y, y_pred)
    }

    return metrics

# Metric functions
def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Distance functions
def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.linalg.norm(x - y)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(x - y))

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate cosine distance."""
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def _minkowski_distance(x: np.ndarray, y: np.ndarray, p: float) -> float:
    """Calculate Minkowski distance."""
    return np.sum(np.abs(x - y) ** p) ** (1/p)

# Regularization solvers
def _solve_lasso(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve Lasso regression."""
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=1.0)
    model.fit(X, y)
    return model.coef_

def _solve_ridge(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve Ridge regression."""
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model.coef_

def _solve_elasticnet(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve ElasticNet regression."""
    from sklearn.linear_model import ElasticNet
    model = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model.fit(X, y)
    return model.coef_

# Gradient and Hessian computations
def _compute_gradient(X: np.ndarray, y: np.ndarray, params: np.ndarray,
                     metric_func: Callable, distance_func: Callable,
                     regularization: Optional[str]) -> np.ndarray:
    """Compute gradient of the objective function."""
    epsilon = 1e-8
    n_params = params.shape[0]
    gradient = np.zeros(n_params)

    for i in range(n_params):
        params_plus = params.copy()
        params_plus[i] += epsilon
        y_pred_plus = X @ params_plus

        params_minus = params.copy()
        params_minus[i] -= epsilon
        y_pred_minus = X @ params_minus

        gradient[i] = (metric_func(y, y_pred_plus) - metric_func(y, y_pred_minus)) / (2 * epsilon)

        if regularization == 'l1':
            gradient[i] += np.sign(params[i])
        elif regularization == 'l2':
            gradient[i] += 2 * params[i]
        elif regularization == 'elasticnet':
            gradient[i] += np.sign(params[i]) + 2 * params[i]

    return gradient

def _compute_hessian(X: np.ndarray, y: np.ndarray, params: np.ndarray,
                    metric_func: Callable, distance_func: Callable,
                    regularization: Optional[str]) -> np.ndarray:
    """Compute Hessian matrix of the objective function."""
    epsilon = 1e-8
    n_params = params.shape[0]
    hessian = np.zeros((n_params, n_params))

    for i in range(n_params):
        params_plus = params.copy()
        params_plus[i] += epsilon
        y_pred_plus = X @ params_plus

        params_minus = params.copy()
        params_minus[i] -= epsilon
        y_pred_minus = X @ params_minus

        for j in range(n_params):
            if i == j:
                hessian[i, j] = (metric_func(y, y_pred_plus) - 2 * metric_func(y, X @ params) +
                                metric_func(y, y_pred_minus)) / (epsilon ** 2)
            else:
                params_ij_plus = params.copy()
                params_ij_plus[i] += epsilon
                params_ij_plus[j] += epsilon
                y_pred_ij_plus = X @ params_ij_plus

                params_ij_minus = params.copy()
                params_ij_minus[i] += epsilon
                params_ij_minus[j] -= epsilon
                y_pred_ij_minus = X @ params_ij_minus

                hessian[i, j] = (metric_func(y, y_pred_ij_plus) - metric_func(y, y_pred_ij_minus)) / (2 * epsilon)

            if regularization == 'l1':
                hessian[i, j] += np.eye(n_params)[i, j]
            elif regularization == 'l2':
                hessian[i, j] += 2 * np.eye(n_params)[i, j]
            elif regularization == 'elasticnet':
                hessian[i, j] += np.eye(n_params)[i, j] + 2 * np.eye(n_params)[i, j]

    return hessian

def _compute_gradient_component(X: np.ndarray, y: np.ndarray, params: np.ndarray,
                              component_idx: int, metric_func: Callable,
                              distance_func: Callable, regularization: Optional[str]) -> float:
    """Compute gradient for a single component."""
    epsilon = 1e-8
    params_plus = params.copy()
    params_plus[component_idx] += epsilon
    y_pred

################################################################################
# variable_spatiale
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """
    Validate input arrays for spatial variable selection.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target vector of shape (n_samples,) or None

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array")
    if y is not None and (not isinstance(y, np.ndarray) or y.ndim != 1):
        raise ValueError("y must be a 1D numpy array or None")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or (y is not None and np.any(np.isnan(y))):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or (y is not None and np.any(np.isinf(y))):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize feature matrix using specified method.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features)
    method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')

    Returns
    -------
    np.ndarray
        Normalized feature matrix
    """
    if method == 'none':
        return X.copy()
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

def _compute_distance_matrix(X: np.ndarray, distance_metric: Union[str, Callable]) -> np.ndarray:
    """
    Compute pairwise distance matrix using specified metric.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features)
    distance_metric : Union[str, Callable]
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable

    Returns
    -------
    np.ndarray
        Pairwise distance matrix of shape (n_samples, n_samples)
    """
    n = X.shape[0]
    dist_matrix = np.zeros((n, n))

    if isinstance(distance_metric, str):
        if distance_metric == 'euclidean':
            for i in range(n):
                dist_matrix[i] = np.linalg.norm(X - X[i], axis=1)
        elif distance_metric == 'manhattan':
            for i in range(n):
                dist_matrix[i] = np.sum(np.abs(X - X[i]), axis=1)
        elif distance_metric == 'cosine':
            for i in range(n):
                dist_matrix[i] = 1 - np.dot(X, X[i]) / (np.linalg.norm(X, axis=1) * np.linalg.norm(X[i]))
        elif distance_metric == 'minkowski':
            for i in range(n):
                dist_matrix[i] = np.sum(np.abs(X - X[i])**3, axis=1)**(1/3)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
    elif callable(distance_metric):
        for i in range(n):
            dist_matrix[i] = distance_metric(X, X[i])
    else:
        raise ValueError("distance_metric must be a string or callable")

    return dist_matrix

def _compute_spatial_weights(dist_matrix: np.ndarray, method: str = 'inverse') -> np.ndarray:
    """
    Compute spatial weights from distance matrix.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Pairwise distance matrix of shape (n_samples, n_samples)
    method : str
        Weighting method ('inverse', 'gaussian')

    Returns
    -------
    np.ndarray
        Spatial weights matrix of shape (n_samples, n_samples)
    """
    if method == 'inverse':
        weights = 1 / (dist_matrix + 1e-8)
    elif method == 'gaussian':
        sigma = np.median(dist_matrix[dist_matrix > 0]) / 2
        weights = np.exp(-dist_matrix**2 / (2 * sigma**2))
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Set diagonal to zero
    np.fill_diagonal(weights, 0)
    return weights

def _select_variables(X: np.ndarray,
                     y: Optional[np.ndarray] = None,
                     distance_metric: Union[str, Callable] = 'euclidean',
                     normalization: str = 'standard',
                     weighting_method: str = 'inverse',
                     metric: Union[str, Callable] = 'mse',
                     solver: str = 'closed_form') -> Dict:
    """
    Select spatial variables based on specified criteria.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target vector of shape (n_samples,) or None
    distance_metric : Union[str, Callable]
        Distance metric for spatial weighting
    normalization : str
        Normalization method for features
    weighting_method : str
        Method for computing spatial weights
    metric : Union[str, Callable]
        Performance metric to optimize
    solver : str
        Solver method for variable selection

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = _normalize_data(X, normalization)

    # Compute distance matrix and weights
    dist_matrix = _compute_distance_matrix(X_norm, distance_metric)
    weights = _compute_spatial_weights(dist_matrix, weighting_method)

    # Variable selection logic based on solver
    if solver == 'closed_form':
        # Closed form solution for spatial variable selection
        pass  # Implementation would go here

    elif solver == 'gradient_descent':
        # Gradient descent based solution
        pass  # Implementation would go here

    elif solver == 'coordinate_descent':
        # Coordinate descent based solution
        pass  # Implementation would go here

    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    if callable(metric):
        computed_metric = metric(X, y)
    elif metric == 'mse':
        if y is None:
            raise ValueError("Target vector y required for MSE metric")
        computed_metric = np.mean((y - X @ coefficients)**2)
    elif metric == 'mae':
        if y is None:
            raise ValueError("Target vector y required for MAE metric")
        computed_metric = np.mean(np.abs(y - X @ coefficients))
    elif metric == 'r2':
        if y is None:
            raise ValueError("Target vector y required for R2 metric")
        ss_total = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - X @ coefficients)**2)
        computed_metric = 1 - ss_res / (ss_total + 1e-8)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Prepare results
    result = {
        'selected_variables': selected_vars,  # Would be computed by solver
        'coefficients': coefficients,         # Would be computed by solver
    }

    metrics = {
        'performance_metric': computed_metric,
    }

    params_used = {
        'distance_metric': distance_metric,
        'normalization': normalization,
        'weighting_method': weighting_method,
        'metric': metric,
        'solver': solver,
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def variable_spatiale_fit(X: np.ndarray,
                         y: Optional[np.ndarray] = None,
                         distance_metric: Union[str, Callable] = 'euclidean',
                         normalization: str = 'standard',
                         weighting_method: str = 'inverse',
                         metric: Union[str, Callable] = 'mse',
                         solver: str = 'closed_form') -> Dict:
    """
    Fit spatial variable selection model.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target vector of shape (n_samples,) or None
    distance_metric : Union[str, Callable]
        Distance metric for spatial weighting ('euclidean', 'manhattan',
        'cosine', 'minkowski') or custom callable
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    weighting_method : str
        Method for computing spatial weights ('inverse', 'gaussian')
    metric : Union[str, Callable]
        Performance metric to optimize ('mse', 'mae', 'r2') or custom callable
    solver : str
        Solver method ('closed_form', 'gradient_descent', 'coordinate_descent')

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used and warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = variable_spatiale_fit(X, y,
    ...                              distance_metric='euclidean',
    ...                              normalization='standard')
    """
    return _select_variables(X, y, distance_metric, normalization,
                           weighting_method, metric, solver)

################################################################################
# variable_texte
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def variable_texte_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Select variables for text data using various statistical methods.

    Parameters:
    -----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    normalizer : callable, optional
        Function to normalize the input features. Default is identity function.
    metric : str or callable, optional
        Metric to evaluate the model. Can be 'mse', 'mae', 'r2', or a custom callable.
    distance : str or callable, optional
        Distance metric for feature selection. Can be 'euclidean', 'manhattan',
        'cosine', or a custom callable.
    solver : str, optional
        Solver to use for optimization. Options: 'closed_form', 'gradient_descent',
        'newton', 'coordinate_descent'.
    regularization : str, optional
        Regularization type. Options: None, 'l1', 'l2', 'elasticnet'.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_normalized = normalizer(X)

    # Select solver and compute parameters
    if solver == 'closed_form':
        params = _solve_closed_form(X_normalized, y, regularization)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(X_normalized, y, regularization,
                                        tol=tol, max_iter=max_iter)
    elif solver == 'newton':
        params = _solve_newton(X_normalized, y, regularization,
                              tol=tol, max_iter=max_iter)
    elif solver == 'coordinate_descent':
        params = _solve_coordinate_descent(X_normalized, y, regularization,
                                          tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, params,
                              metric=metric, custom_metric=custom_metric)

    # Compute feature importance
    feature_importance = _compute_feature_importance(X_normalized, y, params,
                                                    distance=distance)

    return {
        'result': feature_importance,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': _check_warnings(X_normalized, y)
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
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

def _solve_closed_form(X: np.ndarray, y: np.ndarray,
                      regularization: Optional[str] = None) -> np.ndarray:
    """Solve using closed form solution."""
    if regularization is None:
        return np.linalg.pinv(X.T @ X) @ X.T @ y
    elif regularization == 'l2':
        return (X.T @ X + 1e-4 * np.eye(X.shape[1])).T @ (X.T @ X + 1e-4 * np.eye(X.shape[1])) @ X.T @ y
    else:
        raise ValueError(f"Regularization {regularization} not supported for closed form")

def _solve_gradient_descent(X: np.ndarray, y: np.ndarray,
                           regularization: Optional[str] = None,
                           tol: float = 1e-4, max_iter: int = 1000) -> np.ndarray:
    """Solve using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, params, regularization)
        params -= learning_rate * gradient
        if np.linalg.norm(gradient) < tol:
            break

    return params

def _compute_gradient(X: np.ndarray, y: np.ndarray,
                     params: np.ndarray, regularization: Optional[str] = None) -> np.ndarray:
    """Compute gradient for optimization."""
    residuals = X @ params - y
    gradient = (X.T @ residuals) / len(y)

    if regularization == 'l1':
        gradient += np.sign(params)
    elif regularization == 'l2':
        gradient += 2 * params

    return gradient

def _solve_newton(X: np.ndarray, y: np.ndarray,
                 regularization: Optional[str] = None,
                 tol: float = 1e-4, max_iter: int = 1000) -> np.ndarray:
    """Solve using Newton's method."""
    n_features = X.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, params, regularization)
        hessian = X.T @ X / len(y)

        if regularization == 'l2':
            hessian += 2 * np.eye(n_features)

        params -= np.linalg.pinv(hessian) @ gradient

        if np.linalg.norm(gradient) < tol:
            break

    return params

def _solve_coordinate_descent(X: np.ndarray, y: np.ndarray,
                             regularization: Optional[str] = None,
                             tol: float = 1e-4, max_iter: int = 1000) -> np.ndarray:
    """Solve using coordinate descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - (X @ params - X_j * params[j])

            if regularization == 'l1':
                if np.sum(X_j ** 2) != 0:
                    params[j] = np.sign(np.sum(X_j * residuals)) * \
                               max(0, abs(np.sum(X_j * residuals)) - 1) / np.sum(X_j ** 2)
            else:
                params[j] = (X_j.T @ residuals) / np.sum(X_j ** 2)

        if np.linalg.norm(_compute_gradient(X, y, params, regularization)) < tol:
            break

    return params

def _compute_metrics(X: np.ndarray, y: np.ndarray,
                    params: np.ndarray,
                    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
                    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None) -> Dict[str, float]:
    """Compute evaluation metrics."""
    y_pred = X @ params
    metrics_dict = {}

    if custom_metric is not None:
        metrics_dict['custom'] = custom_metric(y, y_pred)

    if metric == 'mse':
        metrics_dict['mse'] = np.mean((y - y_pred) ** 2)
    elif metric == 'mae':
        metrics_dict['mae'] = np.mean(np.abs(y - y_pred))
    elif metric == 'r2':
        metrics_dict['r2'] = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    elif metric == 'logloss':
        metrics_dict['logloss'] = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics_dict

def _compute_feature_importance(X: np.ndarray, y: np.ndarray,
                               params: np.ndarray,
                               distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean') -> Dict[str, Any]:
    """Compute feature importance based on distance metrics."""
    y_pred = X @ params
    residuals = y - y_pred

    if callable(distance):
        feature_importance = {f'feature_{i}': distance(X[:, i], residuals) for i in range(X.shape[1])}
    elif distance == 'euclidean':
        feature_importance = {f'feature_{i}': np.linalg.norm(X[:, i] - residuals) for i in range(X.shape[1])}
    elif distance == 'manhattan':
        feature_importance = {f'feature_{i}': np.sum(np.abs(X[:, i] - residuals)) for i in range(X.shape[1])}
    elif distance == 'cosine':
        feature_importance = {f'feature_{i}': 1 - np.dot(X[:, i], residuals) / (np.linalg.norm(X[:, i]) * np.linalg.norm(residuals)) for i in range(X.shape[1])}
    else:
        raise ValueError(f"Unknown distance metric: {distance}")

    return feature_importance

def _check_warnings(X: np.ndarray, y: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []

    if X.shape[1] > 100:
        warnings.append("Warning: High dimensional data may lead to overfitting")

    if np.any(np.abs(X) > 1e6):
        warnings.append("Warning: Features have very large values, consider normalization")

    if len(y) < 10:
        warnings.append("Warning: Small sample size may affect results")

    return warnings

################################################################################
# variable_dichotomique
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def variable_dichotomique_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalisation: str = 'none',
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
    Fit a dichotomous variable selection model.

    Parameters:
    -----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,) or (n_samples, 1).
    normalisation : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : str or callable
        Metric to optimize: 'mse', 'mae', 'r2', 'logloss', or custom callable.
    distance : str or callable
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'minkowski', or custom callable.
    solver : str
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : str, optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    custom_distance : callable, optional
        Custom distance function if not using built-in distances.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_normalized = _apply_normalization(X, normalisation)

    # Prepare metric and distance functions
    metric_func = _get_metric(metric, custom_metric)
    distance_func = _get_distance(distance, custom_distance)

    # Fit model based on solver choice
    if solver == 'closed_form':
        params = _fit_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        params = _fit_gradient_descent(X_normalized, y, metric_func,
                                      regularization, tol, max_iter)
    elif solver == 'newton':
        params = _fit_newton(X_normalized, y, metric_func,
                            regularization, tol, max_iter)
    elif solver == 'coordinate_descent':
        params = _fit_coordinate_descent(X_normalized, y, metric_func,
                                        regularization, tol, max_iter)
    else:
        raise ValueError("Invalid solver specified.")

    # Calculate metrics
    metrics = _calculate_metrics(X_normalized, y, params, metric_func)

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if X.ndim != 2 or y.ndim not in (1, 2):
        raise ValueError("X must be 2D and y must be 1D or 2D with one column.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the input data."""
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
        raise ValueError("Invalid normalization method specified.")

def _get_metric(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Get metric function based on input."""
    if callable(metric):
        return metric
    elif custom_metric is not None:
        return custom_metric
    elif metric == 'mse':
        return _mse
    elif metric == 'mae':
        return _mae
    elif metric == 'r2':
        return _r2
    elif metric == 'logloss':
        return _logloss
    else:
        raise ValueError("Invalid metric specified.")

def _get_distance(distance: Union[str, Callable], custom_distance: Optional[Callable]) -> Callable:
    """Get distance function based on input."""
    if callable(distance):
        return distance
    elif custom_distance is not None:
        return custom_distance
    elif distance == 'euclidean':
        return _euclidean_distance
    elif distance == 'manhattan':
        return _manhattan_distance
    elif distance == 'cosine':
        return _cosine_distance
    elif distance == 'minkowski':
        return lambda x, y: np.sum(np.abs(x - y)**2, axis=1)**(1/2)
    else:
        raise ValueError("Invalid distance specified.")

def _fit_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit model using closed form solution."""
    X_tX = X.T @ X
    if np.linalg.det(X_tX) == 0:
        raise ValueError("Matrix is singular, cannot use closed form solution.")
    X_tX_inv = np.linalg.inv(X_tX)
    return X_tX_inv @ X.T @ y

def _fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Fit model using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01
    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, params, metric_func)
        if regularization == 'l1':
            gradient += np.sign(params)  # L1 regularization
        elif regularization == 'l2':
            gradient += 2 * params  # L2 regularization
        params_new = params - learning_rate * gradient
        if np.linalg.norm(params_new - params) < tol:
            break
        params = params_new
    return params

def _fit_newton(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Fit model using Newton's method."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, params, metric_func)
        hessian = _compute_hessian(X, y, params, metric_func)
        if regularization == 'l1':
            hessian += np.diag(np.sign(params))
        elif regularization == 'l2':
            hessian += 2 * np.eye(n_features)
        params_new = params - np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(params_new - params) < tol:
            break
        params = params_new
    return params

def _fit_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Fit model using coordinate descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        for i in range(n_features):
            X_i = X[:, i]
            residual = y - np.dot(X, params) + params[i] * X_i
            if regularization == 'l1':
                params[i] = _soft_threshold(np.dot(X_i, residual), 1)
            elif regularization == 'l2':
                params[i] = np.dot(X_i, residual) / (np.dot(X_i, X_i) + 2)
            else:
                params[i] = np.dot(X_i, residual) / np.dot(X_i, X_i)
        if np.linalg.norm(params - params) < tol:
            break
    return params

def _compute_gradient(X: np.ndarray, y: np.ndarray, params: np.ndarray, metric_func: Callable) -> np.ndarray:
    """Compute gradient of the loss function."""
    predictions = X @ params
    if metric_func == _mse:
        return -2 * (X.T @ (y - predictions)) / len(y)
    elif metric_func == _mae:
        return -np.sign(X.T @ (y - predictions)) / len(y)
    elif metric_func == _r2:
        return -2 * (X.T @ (y - predictions)) / np.sum((y - np.mean(y))**2)
    elif metric_func == _logloss:
        exp_pred = np.exp(predictions)
        return X.T @ (exp_pred / (1 + exp_pred) - y) / len(y)
    else:
        return metric_func(X, y, params, gradient=True)

def _compute_hessian(X: np.ndarray, y: np.ndarray, params: np.ndarray, metric_func: Callable) -> np.ndarray:
    """Compute Hessian matrix of the loss function."""
    predictions = X @ params
    if metric_func == _mse:
        return 2 * (X.T @ X) / len(y)
    elif metric_func == _logloss:
        exp_pred = np.exp(predictions)
        return X.T @ (exp_pred / ((1 + exp_pred)**2) * X) / len(y)
    else:
        raise ValueError("Hessian computation not implemented for this metric.")

def _soft_threshold(value: float, threshold: float) -> float:
    """Soft-thresholding function for L1 regularization."""
    if value > threshold:
        return value - threshold
    elif value < -threshold:
        return value + threshold
    else:
        return 0

def _calculate_metrics(X: np.ndarray, y: np.ndarray, params: np.ndarray, metric_func: Callable) -> Dict:
    """Calculate various metrics for the model."""
    predictions = X @ params
    return {
        'mse': _mse(y, predictions),
        'mae': _mae(y, predictions),
        'r2': _r2(y, predictions)
    }

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred)**2)

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / (ss_tot + 1e-8)

def _logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Log Loss."""
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Euclidean distance."""
    return np.linalg.norm(x - y)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Manhattan distance."""
    return np.sum(np.abs(x - y))

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine distance."""
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

################################################################################
# variable_multiclasse
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def variable_multiclasse_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Select variables for multiclass classification problems.

    Parameters:
    -----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,) with class labels
    normalizer : Callable, optional
        Function to normalize features (default: None)
    metric : str or callable, optional
        Metric to evaluate model performance (default: "mse")
    distance : str or callable, optional
        Distance metric for feature selection (default: "euclidean")
    solver : str, optional
        Solver algorithm (default: "closed_form")
    regularization : str, optional
        Regularization type (none, l1, l2, elasticnet)
    tol : float, optional
        Tolerance for convergence (default: 1e-4)
    max_iter : int, optional
        Maximum iterations (default: 1000)
    custom_metric : callable, optional
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 3, size=100)
    >>> result = variable_multiclasse_fit(X, y, normalizer="standard", metric="r2")
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Apply normalization if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Get metric function
    metric_func = _get_metric_function(metric, custom_metric)

    # Get distance function
    distance_func = _get_distance_function(distance)

    # Solve for variable selection
    if solver == "closed_form":
        result = _solve_closed_form(X_normalized, y, distance_func)
    elif solver == "gradient_descent":
        result = _solve_gradient_descent(X_normalized, y, distance_func,
                                       regularization=regularization,
                                       tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(X_normalized, y, result["coefficients"],
                               metric_func)

    return {
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

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
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
    """Apply normalization to features."""
    if normalizer is None:
        return X
    return normalizer(X)

def _get_metric_function(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Get metric function based on input."""
    if custom_metric is not None:
        return custom_metric

    metrics = {
        "mse": _mean_squared_error,
        "mae": _mean_absolute_error,
        "r2": _r_squared,
        "logloss": _log_loss
    }

    if isinstance(metric, str):
        return metrics.get(metric.lower(), _mean_squared_error)
    elif callable(metric):
        return metric
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _get_distance_function(distance: Union[str, Callable]) -> Callable:
    """Get distance function based on input."""
    distances = {
        "euclidean": _euclidean_distance,
        "manhattan": _manhattan_distance,
        "cosine": _cosine_distance,
        "minkowski": _minkowski_distance
    }

    if isinstance(distance, str):
        return distances.get(distance.lower(), _euclidean_distance)
    elif callable(distance):
        return distance
    else:
        raise ValueError(f"Unknown distance: {distance}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray, distance_func: Callable) -> Dict[str, Any]:
    """Solve variable selection using closed form solution."""
    # Implement closed form solution logic
    coefficients = np.linalg.pinv(X) @ y
    return {
        "coefficients": coefficients,
        "method": "closed_form"
    }

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable,
    *,
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """Solve variable selection using gradient descent."""
    # Implement gradient descent logic
    coefficients = np.zeros(X.shape[1])
    for _ in range(max_iter):
        # Update coefficients
        pass  # Gradient descent implementation

    return {
        "coefficients": coefficients,
        "method": "gradient_descent",
        "iterations": max_iter
    }

def _calculate_metrics(X: np.ndarray, y: np.ndarray, coefficients: np.ndarray,
                      metric_func: Callable) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    predictions = X @ coefficients
    return {
        "metric_value": metric_func(y, predictions),
        "predictions": predictions
    }

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate log loss."""
    # Implement multiclass log loss
    pass

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine distance."""
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _minkowski_distance(a: np.ndarray, b: np.ndarray, p: float = 3) -> float:
    """Calculate Minkowski distance."""
    return np.sum(np.abs(a - b) ** p) ** (1/p)
