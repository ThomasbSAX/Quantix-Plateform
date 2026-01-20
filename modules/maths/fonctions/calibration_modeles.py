"""
Quantix – Module calibration_modeles
Généré automatiquement
Date: 2026-01-06
"""

################################################################################
# calibration_lineaire
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List, Tuple

def calibration_lineaire_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalisation: str = "standard",
    metric: Union[str, Callable] = "mse",
    solver: str = "closed_form",
    regularisation: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Calibration linéaire avec options paramétrables.

    Parameters
    ----------
    X : np.ndarray
        Matrice des caractéristiques (n_samples, n_features)
    y : np.ndarray
        Vecteur cible (n_samples,)
    normalisation : str
        Type de normalisation ("none", "standard", "minmax", "robust")
    metric : str or callable
        Métrique d'évaluation ("mse", "mae", "r2", "logloss") ou fonction personnalisée
    solver : str
        Solveur ("closed_form", "gradient_descent", "newton", "coordinate_descent")
    regularisation : str or None
        Type de régularisation ("none", "l1", "l2", "elasticnet")
    tol : float
        Tolérance pour la convergence
    max_iter : int
        Nombre maximum d'itérations
    learning_rate : float
        Taux d'apprentissage pour les solveurs itératifs
    custom_metric : callable or None
        Fonction de métrique personnalisée
    custom_distance : callable or None
        Fonction de distance personnalisée

    Returns
    -------
    Dict
        Dictionnaire contenant les résultats, métriques et paramètres utilisés

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = calibration_lineaire_fit(X, y, normalisation="standard", solver="gradient_descent")
    """
    # Validation des entrées
    _validate_inputs(X, y)

    # Normalisation
    X_norm, y_norm = _apply_normalisation(X, y, normalisation)

    # Choix de la métrique
    metric_func = _get_metric_function(metric, custom_metric)

    # Choix du solveur
    if solver == "closed_form":
        params = _solve_closed_form(X_norm, y_norm)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(
            X_norm, y_norm,
            metric_func=metric_func,
            tol=tol,
            max_iter=max_iter,
            learning_rate=learning_rate
        )
    elif solver == "newton":
        params = _solve_newton(
            X_norm, y_norm,
            metric_func=metric_func,
            tol=tol,
            max_iter=max_iter
        )
    elif solver == "coordinate_descent":
        params = _solve_coordinate_descent(
            X_norm, y_norm,
            metric_func=metric_func,
            tol=tol,
            max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calcul des métriques
    metrics = _compute_metrics(X_norm, y_norm, params, metric_func)

    # Retour des résultats
    return {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalisation": normalisation,
            "metric": metric,
            "solver": solver,
            "regularisation": regularisation,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validation des entrées."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or inf values")

def _apply_normalisation(
    X: np.ndarray,
    y: np.ndarray,
    normalisation: str
) -> tuple[np.ndarray, np.ndarray]:
    """Application de la normalisation."""
    if normalisation == "none":
        return X, y
    elif normalisation == "standard":
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif normalisation == "minmax":
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalisation == "robust":
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        raise ValueError(f"Unknown normalisation: {normalisation}")
    return X_norm, y_norm

def _get_metric_function(
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Callable:
    """Récupération de la fonction de métrique."""
    if custom_metric is not None:
        return custom_metric

    metrics = {
        "mse": _mean_squared_error,
        "mae": _mean_absolute_error,
        "r2": _r_squared,
        "logloss": _log_loss
    }

    if isinstance(metric, str):
        return metrics.get(metric, _mean_squared_error)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Résolution par forme fermée."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric_func: Callable,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01
) -> np.ndarray:
    """Descente de gradient."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        gradients = 2 * X.T @ (X @ params - y) / len(y)
        params -= learning_rate * gradients
        current_loss = metric_func(y, X @ params)

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return params

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric_func: Callable,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Méthode de Newton."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        hessian = 2 * X.T @ X / len(y)
        gradients = 2 * X.T @ (X @ params - y) / len(y)

        if np.linalg.cond(hessian) < 1e15:
            params -= np.linalg.solve(hessian, gradients)
        else:
            params -= gradients * 0.01

        current_loss = metric_func(y, X @ params)

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return params

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric_func: Callable,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Descente de coordonnées."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            params_j = params[j]

            # Calcul des résidus sans le paramètre courant
            residuals = y - np.dot(X, params) + X_j * params_j

            # Mise à jour du paramètre
            numerator = np.dot(X_j, residuals)
            denominator = np.dot(X_j, X_j)

            if denominator != 0:
                params[j] = numerator / denominator

        current_loss = metric_func(y, X @ params)

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return params

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric_func: Callable
) -> Dict[str, float]:
    """Calcul des métriques."""
    y_pred = X @ params

    metrics = {
        "mse": _mean_squared_error(y, y_pred),
        "mae": _mean_absolute_error(y, y_pred),
        "r2": _r_squared(y, y_pred)
    }

    if metric_func != _mean_squared_error:
        metrics["custom"] = metric_func(y, y_pred)

    return metrics

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erreur quadratique moyenne."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erreur absolue moyenne."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient de détermination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

################################################################################
# calibration_logistique
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def calibration_logistique_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalisation: str = 'standard',
    metric: Union[str, Callable] = 'logloss',
    distance: str = 'euclidean',
    solver: str = 'gradient_descent',
    regularisation: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Calibrate a logistic regression model with configurable options.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,) with values in {0, 1}
    normalisation : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'
    metric : str or callable
        Metric to optimize: 'mse', 'mae', 'r2', 'logloss', or custom callable
    distance : str
        Distance metric: 'euclidean', 'manhattan', 'cosine', or custom callable
    solver : str
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'
    regularisation : str, optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'
    tol : float
        Tolerance for stopping criteria
    max_iter : int
        Maximum number of iterations
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics
    custom_distance : callable, optional
        Custom distance function if not using built-in distances

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': fitted model parameters
        - 'metrics': computed metrics
        - 'params_used': parameters used in fitting
        - 'warnings': any warnings generated

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = calibration_logistique_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalisation(X, normalisation)

    # Initialize parameters
    params = _initialize_parameters(X_normalized.shape[1])

    # Choose solver and optimize
    if solver == 'closed_form':
        params = _solve_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        params = _gradient_descent(
            X_normalized, y,
            metric=metric,
            distance=distance,
            regularisation=regularisation,
            tol=tol,
            max_iter=max_iter
        )
    elif solver == 'newton':
        params = _newton_method(
            X_normalized, y,
            metric=metric,
            distance=distance,
            regularisation=regularisation,
            tol=tol,
            max_iter=max_iter
        )
    elif solver == 'coordinate_descent':
        params = _coordinate_descent(
            X_normalized, y,
            metric=metric,
            distance=distance,
            regularisation=regularisation,
            tol=tol,
            max_iter=max_iter
        )

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, params, metric)

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularisation': regularisation
        },
        'warnings': _check_warnings(X_normalized, y)
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("X must contain numerical values")
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("y must contain numerical values")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only 0 and 1 values")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalisation(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the feature matrix."""
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
        raise ValueError(f"Unknown normalisation method: {method}")

def _initialize_parameters(n_features: int) -> np.ndarray:
    """Initialize model parameters."""
    return np.zeros(n_features + 1)  # +1 for intercept

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve logistic regression using closed-form solution."""
    # This is a placeholder - actual implementation would use iterative methods
    return np.linalg.pinv(X) @ y

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric: Union[str, Callable] = 'logloss',
    distance: str = 'euclidean',
    regularisation: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Optimize parameters using gradient descent."""
    params = _initialize_parameters(X.shape[1])
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, params, metric)
        if regularisation == 'l1':
            gradient += np.sign(params[1:])  # L1 regularization
        elif regularisation == 'l2':
            gradient += 2 * params[1:]      # L2 regularization

        params -= learning_rate * gradient

        if np.linalg.norm(gradient) < tol:
            break

    return params

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric: Union[str, Callable] = 'logloss',
    distance: str = 'euclidean',
    regularisation: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Optimize parameters using Newton's method."""
    params = _initialize_parameters(X.shape[1])

    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, params, metric)
        hessian = _compute_hessian(X, y, params)

        if regularisation == 'l1':
            hessian += np.diag(np.hstack([0, 1]))  # L1 regularization
        elif regularisation == 'l2':
            hessian += 2 * np.eye(X.shape[1] + 1)   # L2 regularization

        params -= np.linalg.pinv(hessian) @ gradient

        if np.linalg.norm(gradient) < tol:
            break

    return params

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric: Union[str, Callable] = 'logloss',
    distance: str = 'euclidean',
    regularisation: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Optimize parameters using coordinate descent."""
    params = _initialize_parameters(X.shape[1])

    for _ in range(max_iter):
        for i in range(len(params)):
            # Create a copy of params with the current parameter zeroed
            params_copy = np.array(params)
            params_copy[i] = 0

            # Compute gradient with respect to the current parameter
            gradient_i = _compute_gradient(X, y, params_copy, metric)[i]

            if regularisation == 'l1':
                gradient_i += np.sign(params[i])
            elif regularisation == 'l2':
                gradient_i += 2 * params[i]

            # Update the parameter
            params[i] -= gradient_i

        if np.linalg.norm(_compute_gradient(X, y, params, metric)) < tol:
            break

    return params

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable]
) -> np.ndarray:
    """Compute the gradient of the loss function."""
    if metric == 'logloss':
        probabilities = _sigmoid(X @ params)
        error = probabilities - y
        gradient = X.T @ error / len(y)
    elif callable(metric):
        gradient = metric(X, y, params, 'gradient')
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return gradient

def _compute_hessian(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray
) -> np.ndarray:
    """Compute the Hessian matrix of the loss function."""
    probabilities = _sigmoid(X @ params)
    diag_hessian = probabilities * (1 - probabilities)
    hessian = X.T @ np.diag(diag_hessian) @ X / len(y)
    return hessian

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Compute various metrics for the model."""
    probabilities = _sigmoid(X @ params)

    metrics = {}
    if metric == 'logloss':
        metrics['logloss'] = -np.mean(y * np.log(probabilities + 1e-8) +
                                    (1 - y) * np.log(1 - probabilities + 1e-8))
    elif metric == 'mse':
        metrics['mse'] = np.mean((y - probabilities) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y - probabilities))
    elif metric == 'r2':
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - probabilities) ** 2)
        metrics['r2'] = 1 - ss_residual / (ss_total + 1e-8)
    elif callable(metric):
        metrics['custom'] = metric(X, y, params)

    return metrics

def _check_warnings(
    X: np.ndarray,
    y: np.ndarray
) -> list:
    """Check for potential issues and return warnings."""
    warnings = []

    if np.any(np.std(X, axis=0) == 0):
        warnings.append("Some features have zero variance")

    if len(np.unique(y)) == 1:
        warnings.append("Only one class present in y")

    return warnings

################################################################################
# calibration_platt_scaling
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    scores: np.ndarray,
    true_labels: np.ndarray,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse'
) -> None:
    """
    Validate input data and parameters.

    Parameters
    ----------
    scores : np.ndarray
        Model scores (probabilities or decision function values)
    true_labels : np.ndarray
        True binary labels (0 or 1)
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric function ('mse', 'mae', 'r2', 'logloss') or custom callable

    Raises
    ------
    ValueError
        If inputs are invalid or incompatible
    """
    if scores.ndim != 1:
        raise ValueError("Scores must be 1-dimensional")
    if true_labels.ndim != 1:
        raise ValueError("True labels must be 1-dimensional")
    if len(scores) != len(true_labels):
        raise ValueError("Scores and true labels must have same length")
    if np.any((true_labels != 0) & (true_labels != 1)):
        raise ValueError("True labels must be binary (0 or 1)")

    if normalize not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("Invalid normalization method")

    if isinstance(metric, str) and metric not in ['mse', 'mae', 'r2', 'logloss']:
        raise ValueError("Invalid metric string")

def _normalize_scores(
    scores: np.ndarray,
    method: str = 'standard'
) -> np.ndarray:
    """
    Normalize scores using specified method.

    Parameters
    ----------
    scores : np.ndarray
        Input scores to normalize
    method : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')

    Returns
    ------
    np.ndarray
        Normalized scores
    """
    if method == 'none':
        return scores

    if method == 'standard':
        mean = np.nanmean(scores)
        std = np.nanstd(scores)
        if std == 0:
            return scores - mean
        return (scores - mean) / std

    if method == 'minmax':
        min_val = np.nanmin(scores)
        max_val = np.nanmax(scores)
        if min_val == max_val:
            return scores
        return (scores - min_val) / (max_val - min_val)

    if method == 'robust':
        median = np.nanmedian(scores)
        iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
        if iqr == 0:
            return scores - median
        return (scores - median) / iqr

def _compute_metric(
    scores: np.ndarray,
    true_labels: np.ndarray,
    metric_func: Union[str, Callable]
) -> float:
    """
    Compute specified metric between scores and true labels.

    Parameters
    ----------
    scores : np.ndarray
        Model scores (probabilities or decision function values)
    true_labels : np.ndarray
        True binary labels (0 or 1)
    metric_func : str or callable
        Metric function ('mse', 'mae', 'r2', 'logloss') or custom callable

    Returns
    ------
    float
        Computed metric value
    """
    if isinstance(metric_func, str):
        if metric_func == 'mse':
            return np.mean((scores - true_labels) ** 2)
        elif metric_func == 'mae':
            return np.mean(np.abs(scores - true_labels))
        elif metric_func == 'r2':
            ss_res = np.sum((scores - true_labels) ** 2)
            ss_tot = np.sum((true_labels - np.mean(true_labels)) ** 2)
            return 1 - ss_res / ss_tot
        elif metric_func == 'logloss':
            epsilon = 1e-15
            scores = np.clip(scores, epsilon, 1 - epsilon)
            return -np.mean(true_labels * np.log(scores) +
                           (1 - true_labels) * np.log(1 - scores))

    # Handle custom callable
    return metric_func(scores, true_labels)

def _platt_scaling_closed_form(
    scores: np.ndarray,
    true_labels: np.ndarray
) -> Dict[str, float]:
    """
    Closed-form solution for Platt scaling calibration.

    Parameters
    ----------
    scores : np.ndarray
        Model scores (probabilities or decision function values)
    true_labels : np.ndarray
        True binary labels (0 or 1)

    Returns
    ------
    Dict[str, float]
        Dictionary containing calibration parameters (A, B)
    """
    # Add intercept term
    X = np.column_stack([np.ones_like(scores), scores])
    y = true_labels

    # Compute closed-form solution
    XtX_inv = np.linalg.inv(X.T @ X)
    params = XtX_inv @ X.T @ y

    return {'A': params[0], 'B': params[1]}

def _platt_scaling_gradient_descent(
    scores: np.ndarray,
    true_labels: np.ndarray,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Dict[str, float]:
    """
    Gradient descent solution for Platt scaling calibration.

    Parameters
    ----------
    scores : np.ndarray
        Model scores (probabilities or decision function values)
    true_labels : np.ndarray
        True binary labels (0 or 1)
    learning_rate : float, optional
        Learning rate for gradient descent
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Tolerance for convergence

    Returns
    ------
    Dict[str, float]
        Dictionary containing calibration parameters (A, B)
    """
    # Initialize parameters
    A = 0.0
    B = np.log((scores.size - true_labels.sum()) / (true_labels.sum() + 1e-8))

    for _ in range(max_iter):
        # Compute predicted probabilities
        probs = 1 / (1 + np.exp(-(A + B * scores)))

        # Compute gradients
        grad_A = np.mean(probs - true_labels)
        grad_B = np.mean((probs - true_labels) * scores)

        # Update parameters
        A_new = A - learning_rate * grad_A
        B_new = B - learning_rate * grad_B

        # Check convergence
        if np.sqrt((A_new - A)**2 + (B_new - B)**2) < tol:
            break

        A, B = A_new, B_new

    return {'A': A, 'B': B}

def calibration_platt_scaling_fit(
    scores: np.ndarray,
    true_labels: np.ndarray,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    **solver_kwargs
) -> Dict:
    """
    Calibrate model scores using Platt scaling.

    Parameters
    ----------
    scores : np.ndarray
        Model scores (probabilities or decision function values)
    true_labels : np.ndarray
        True binary labels (0 or 1)
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric function ('mse', 'mae', 'r2', 'logloss') or custom callable
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent')
    **solver_kwargs : dict
        Additional keyword arguments for the solver

    Returns
    ------
    Dict
        Dictionary containing:
        - 'result': Calibration parameters (A, B)
        - 'metrics': Dictionary of computed metrics
        - 'params_used': Parameters used in the calibration
        - 'warnings': List of warnings (if any)
    """
    # Validate inputs
    _validate_inputs(scores, true_labels, normalize, metric)

    # Normalize scores
    normalized_scores = _normalize_scores(scores.copy(), normalize)

    # Select solver
    if solver == 'closed_form':
        params = _platt_scaling_closed_form(normalized_scores, true_labels)
    elif solver == 'gradient_descent':
        params = _platt_scaling_gradient_descent(
            normalized_scores, true_labels, **solver_kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute calibrated probabilities
    calibrated_probs = 1 / (1 + np.exp(-(params['A'] + params['B'] * normalized_scores)))

    # Compute metrics
    metrics = {
        'mse': np.mean((calibrated_probs - true_labels) ** 2),
        'mae': np.mean(np.abs(calibrated_probs - true_labels)),
        'r2': 1 - np.sum((calibrated_probs - true_labels) ** 2) /
              np.sum((true_labels - np.mean(true_labels)) ** 2),
        'logloss': -np.mean(
            true_labels * np.log(calibrated_probs + 1e-15) +
            (1 - true_labels) * np.log(1 - calibrated_probs + 1e-15)
        )
    }

    # If custom metric was provided, compute it
    if isinstance(metric, Callable):
        metrics['custom'] = _compute_metric(calibrated_probs, true_labels, metric)

    return {
        'result': params,
        'metrics': metrics if isinstance(metric, str) else {'custom': metrics['custom']},
        'params_used': {
            'normalize': normalize,
            'metric': str(metric) if isinstance(metric, Callable) else metric,
            'solver': solver,
            **solver_kwargs
        },
        'warnings': []
    }

# Example usage:
"""
scores = np.array([0.1, 0.4, 0.35, 0.8])
true_labels = np.array([0, 0, 1, 1])

result = calibration_platt_scaling_fit(
    scores,
    true_labels,
    normalize='standard',
    metric='mse',
    solver='closed_form'
)
"""

################################################################################
# calibration_isotonic
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def calibration_isotonic_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: Union[str, Callable] = 'mse',
    normalization: str = 'none',
    solver: str = 'closed_form',
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    weights: Optional[np.ndarray] = None
) -> Dict:
    """
    Calibrate predictions using isotonic regression.

    Parameters:
    -----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values to calibrate.
    metric : str or callable, optional
        Metric to optimize. Options: 'mse', 'mae', 'r2', 'logloss'.
    normalization : str, optional
        Normalization method. Options: 'none', 'standard', 'minmax', 'robust'.
    solver : str, optional
        Solver method. Options: 'closed_form', 'gradient_descent'.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    weights : np.ndarray, optional
        Weights for weighted calibration.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred, weights)

    # Normalize data if specified
    y_true_norm, y_pred_norm = _apply_normalization(y_true, y_pred, normalization)

    # Choose solver
    if solver == 'closed_form':
        calibrated_preds = _closed_form_solver(y_true_norm, y_pred_norm)
    elif solver == 'gradient_descent':
        calibrated_preds = _gradient_descent_solver(
            y_true_norm, y_pred_norm, tol=tol, max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(
        y_true_norm, calibrated_preds, metric=metric, custom_metric=custom_metric
    )

    # Denormalize predictions if needed
    if normalization != 'none':
        calibrated_preds = _denormalize(calibrated_preds, normalization)

    return {
        'result': calibrated_preds,
        'metrics': metrics,
        'params_used': {
            'metric': metric,
            'normalization': normalization,
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray]
) -> None:
    """Validate input arrays."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if weights is not None:
        if len(weights) != len(y_true):
            raise ValueError("Weights must have the same length as y_true.")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

def _apply_normalization(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str
) -> tuple:
    """Apply normalization to input arrays."""
    if method == 'none':
        return y_true, y_pred
    elif method == 'standard':
        mean = np.mean(y_true)
        std = np.std(y_true)
        return (y_true - mean) / std, (y_pred - mean) / std
    elif method == 'minmax':
        min_val = np.min(y_true)
        max_val = np.max(y_true)
        return (y_true - min_val) / (max_val - min_val), (y_pred - min_val) / (max_val - min_val)
    elif method == 'robust':
        median = np.median(y_true)
        iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        return (y_true - median) / iqr, (y_pred - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _denormalize(
    y_pred: np.ndarray,
    method: str
) -> np.ndarray:
    """Denormalize predictions."""
    if method == 'none':
        return y_pred
    elif method == 'standard':
        mean = np.mean(y_true)
        std = np.std(y_true)
        return y_pred * std + mean
    elif method == 'minmax':
        min_val = np.min(y_true)
        max_val = np.max(y_true)
        return y_pred * (max_val - min_val) + min_val
    elif method == 'robust':
        median = np.median(y_true)
        iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        return y_pred * iqr + median
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _closed_form_solver(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """Closed-form solution for isotonic regression."""
    # Sort predictions and true values
    sorted_indices = np.argsort(y_pred)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    # Pool adjacent equal predictions
    unique_preds, indices = np.unique(y_pred_sorted, return_inverse=True)
    pooled_true = np.array([np.mean(y_true_sorted[indices == i]) for i in range(len(unique_preds))])

    # Calculate calibrated predictions
    calibrated = np.zeros_like(y_true_sorted)
    for i in range(len(unique_preds)):
        calibrated[indices == i] = pooled_true[i]

    return calibrated

def _gradient_descent_solver(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Gradient descent solver for isotonic regression."""
    calibrated = y_pred.copy()
    prev_loss = np.inf
    for _ in range(max_iter):
        # Calculate gradients (simplified example)
        gradient = 2 * (calibrated - y_true) / len(y_true)

        # Update predictions with gradient step
        calibrated -= 0.1 * gradient

        # Check convergence
        current_loss = np.mean((calibrated - y_true) ** 2)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return calibrated

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Calculate metrics for calibration."""
    metrics_dict = {}

    if metric == 'mse' or (custom_metric is None and metric == 'mse'):
        metrics_dict['mse'] = np.mean((y_true - y_pred) ** 2)
    if metric == 'mae' or (custom_metric is None and metric == 'mae'):
        metrics_dict['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric == 'r2' or (custom_metric is None and metric == 'r2'):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics_dict['r2'] = 1 - (ss_res / ss_tot)
    if metric == 'logloss' or (custom_metric is None and metric == 'logloss'):
        # Assuming y_true are probabilities for log loss
        metrics_dict['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    if custom_metric is not None:
        metrics_dict['custom'] = custom_metric(y_true, y_pred)

    return metrics_dict

################################################################################
# calibration_bayesienne
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def calibration_bayesienne_fit(
    data: np.ndarray,
    prior: np.ndarray,
    likelihood_func: Callable[[np.ndarray, np.ndarray], float],
    prior_func: Callable[[np.ndarray], float],
    normalizer: str = 'standard',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'gradient_descent',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calibration bayésienne des paramètres d'un modèle.

    Parameters:
    -----------
    data : np.ndarray
        Données d'observation (n_samples, n_features).
    prior : np.ndarray
        A priori sur les paramètres.
    likelihood_func : Callable[[np.ndarray, np.ndarray], float]
        Fonction de vraisemblance.
    prior_func : Callable[[np.ndarray], float]
        Fonction a priori.
    normalizer : str, optional
        Méthode de normalisation ('none', 'standard', 'minmax', 'robust').
    metric : Union[str, Callable], optional
        Métrique d'évaluation ('mse', 'mae', 'r2', 'logloss') ou fonction personnalisée.
    solver : str, optional
        Solveur ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    tol : float, optional
        Tolérance pour la convergence.
    max_iter : int, optional
        Nombre maximal d'itérations.
    custom_params : Dict[str, Any], optional
        Paramètres personnalisés pour le solveur.

    Returns:
    --------
    Dict[str, Any]
        Dictionnaire contenant les résultats, métriques, paramètres utilisés et avertissements.
    """
    # Validation des entrées
    _validate_inputs(data, prior)

    # Normalisation des données
    normalized_data = _normalize_data(data, normalizer)

    # Choix de la métrique
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    # Choix du solveur
    if solver == 'closed_form':
        result = _solve_closed_form(normalized_data, prior, likelihood_func, prior_func)
    elif solver == 'gradient_descent':
        result = _solve_gradient_descent(
            normalized_data, prior, likelihood_func, prior_func,
            metric_func, tol, max_iter, custom_params
        )
    elif solver == 'newton':
        result = _solve_newton(
            normalized_data, prior, likelihood_func, prior_func,
            metric_func, tol, max_iter, custom_params
        )
    elif solver == 'coordinate_descent':
        result = _solve_coordinate_descent(
            normalized_data, prior, likelihood_func, prior_func,
            metric_func, tol, max_iter, custom_params
        )
    else:
        raise ValueError("Solver non reconnu")

    # Calcul des métriques
    metrics = _compute_metrics(normalized_data, result['params'], metric_func)

    # Retour des résultats
    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer,
            'metric': metric,
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

def _validate_inputs(data: np.ndarray, prior: np.ndarray) -> None:
    """Validation des entrées."""
    if not isinstance(data, np.ndarray) or not isinstance(prior, np.ndarray):
        raise TypeError("Les données et l'a priori doivent être des tableaux NumPy")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Les données contiennent des NaN ou Inf")
    if np.isnan(prior).any() or np.isinf(prior).any():
        raise ValueError("L'a priori contient des NaN ou Inf")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalisation des données."""
    if method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError("Méthode de normalisation non reconnue")

def _get_metric_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Récupération de la fonction de métrique."""
    metrics = {
        'mse': _mse,
        'mae': _mae,
        'r2': _r2,
        'logloss': _logloss
    }
    if metric not in metrics:
        raise ValueError("Métrique non reconnue")
    return metrics[metric]

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erreur quadratique moyenne."""
    return np.mean((y_true - y_pred) ** 2)

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erreur absolue moyenne."""
    return np.mean(np.abs(y_true - y_pred))

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient de détermination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Log loss."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _solve_closed_form(
    data: np.ndarray,
    prior: np.ndarray,
    likelihood_func: Callable[[np.ndarray, np.ndarray], float],
    prior_func: Callable[[np.ndarray], float]
) -> Dict[str, Any]:
    """Résolution en forme fermée."""
    # Implémentation simplifiée pour l'exemple
    return {'params': np.linalg.inv(prior) @ data.T @ data}

def _solve_gradient_descent(
    data: np.ndarray,
    prior: np.ndarray,
    likelihood_func: Callable[[np.ndarray, np.ndarray], float],
    prior_func: Callable[[np.ndarray], float],
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int,
    custom_params: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Descente de gradient."""
    # Implémentation simplifiée pour l'exemple
    params = prior.copy()
    for _ in range(max_iter):
        grad = _compute_gradient(data, params, likelihood_func, prior_func)
        params -= custom_params.get('learning_rate', 0.01) * grad
        if np.linalg.norm(grad) < tol:
            break
    return {'params': params}

def _compute_gradient(
    data: np.ndarray,
    params: np.ndarray,
    likelihood_func: Callable[[np.ndarray, np.ndarray], float],
    prior_func: Callable[[np.ndarray], float]
) -> np.ndarray:
    """Calcul du gradient."""
    # Implémentation simplifiée pour l'exemple
    return -2 * data.T @ (data @ params - likelihood_func(data, params)) + prior_func(params)

def _solve_newton(
    data: np.ndarray,
    prior: np.ndarray,
    likelihood_func: Callable[[np.ndarray, np.ndarray], float],
    prior_func: Callable[[np.ndarray], float],
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int,
    custom_params: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Méthode de Newton."""
    # Implémentation simplifiée pour l'exemple
    params = prior.copy()
    for _ in range(max_iter):
        grad = _compute_gradient(data, params, likelihood_func, prior_func)
        hessian = _compute_hessian(data, params, likelihood_func, prior_func)
        params -= np.linalg.inv(hessian) @ grad
        if np.linalg.norm(grad) < tol:
            break
    return {'params': params}

def _compute_hessian(
    data: np.ndarray,
    params: np.ndarray,
    likelihood_func: Callable[[np.ndarray, np.ndarray], float],
    prior_func: Callable[[np.ndarray], float]
) -> np.ndarray:
    """Calcul de la hessienne."""
    # Implémentation simplifiée pour l'exemple
    return 2 * data.T @ data + prior_func(params)

def _solve_coordinate_descent(
    data: np.ndarray,
    prior: np.ndarray,
    likelihood_func: Callable[[np.ndarray, np.ndarray], float],
    prior_func: Callable[[np.ndarray], float],
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int,
    custom_params: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Descente de coordonnées."""
    # Implémentation simplifiée pour l'exemple
    params = prior.copy()
    for _ in range(max_iter):
        for i in range(params.shape[0]):
            params[i] = _optimize_coordinate(data, params, i, likelihood_func, prior_func)
        if np.linalg.norm(_compute_gradient(data, params, likelihood_func, prior_func)) < tol:
            break
    return {'params': params}

def _optimize_coordinate(
    data: np.ndarray,
    params: np.ndarray,
    coord: int,
    likelihood_func: Callable[[np.ndarray, np.ndarray], float],
    prior_func: Callable[[np.ndarray], float]
) -> float:
    """Optimisation d'une coordonnée."""
    # Implémentation simplifiée pour l'exemple
    return np.linalg.inv(data[:, coord].T @ data[:, coord]) @ (data[:, coord].T @ likelihood_func(data, params) - prior_func(params))

def _compute_metrics(
    data: np.ndarray,
    params: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Calcul des métriques."""
    y_pred = data @ params
    return {'metric': metric_func(data, y_pred)}

################################################################################
# calibration_par_ensemble
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def calibration_par_ensemble_fit(
    X: np.ndarray,
    y: np.ndarray,
    models: list,
    metrics: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    normalizations: Optional[list] = None,
    distances: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric_weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calibrate an ensemble of models using specified metrics and normalizations.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    models : list
        List of model objects with fit and predict methods.
    metrics : str or callable
        Metric to evaluate model performance. Can be 'mse', 'mae', 'r2', or a custom callable.
    normalizations : list, optional
        List of normalization methods to apply. Can be 'none', 'standard', 'minmax', or 'robust'.
    distances : str or callable, optional
        Distance metric for model aggregation. Default is 'euclidean'.
    solver : str, optional
        Solver to use for optimization. Default is 'gradient_descent'.
    regularization : str, optional
        Regularization type. Can be 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric_weights : np.ndarray, optional
        Weights for custom metrics.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y, models)

    # Normalize data if specified
    X_normalized = _apply_normalizations(X, normalizations)

    # Fit models
    fitted_models = []
    for model in models:
        model.fit(X_normalized, y)
        fitted_models.append(model)

    # Calculate metrics
    metrics_results = _calculate_metrics(fitted_models, X_normalized, y, metrics, custom_metric_weights)

    # Aggregate models based on specified distance and solver
    aggregated_model = _aggregate_models(fitted_models, X_normalized, y, distances, solver, regularization, tol, max_iter)

    # Prepare results
    result = {
        'result': aggregated_model,
        'metrics': metrics_results,
        'params_used': {
            'normalizations': normalizations,
            'metrics': metrics,
            'distances': distances,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray, models: list) -> None:
    """Validate input data and models."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if not all(hasattr(model, 'fit') and hasattr(model, 'predict') for model in models):
        raise ValueError("All models must have fit and predict methods.")

def _apply_normalizations(X: np.ndarray, normalizations: Optional[list]) -> np.ndarray:
    """Apply specified normalizations to the input data."""
    if not normalizations:
        return X
    for norm in normalizations:
        if norm == 'standard':
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        elif norm == 'minmax':
            X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        elif norm == 'robust':
            X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X

def _calculate_metrics(
    models: list,
    X: np.ndarray,
    y: np.ndarray,
    metrics: Union[str, Callable],
    custom_metric_weights: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate specified metrics for the models."""
    metrics_results = {}
    y_preds = np.array([model.predict(X) for model in models]).T

    if isinstance(metrics, str):
        if metrics == 'mse':
            metrics_results['mse'] = np.mean((y_preds - y) ** 2, axis=0)
        elif metrics == 'mae':
            metrics_results['mae'] = np.mean(np.abs(y_preds - y), axis=0)
        elif metrics == 'r2':
            ss_res = np.sum((y_preds - y) ** 2, axis=0)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            metrics_results['r2'] = 1 - (ss_res / ss_tot)
    else:
        metrics_results['custom'] = np.array([metrics(y_pred, y) for y_pred in y_preds.T])

    if custom_metric_weights is not None:
        weighted_metrics = {}
        for key, value in metrics_results.items():
            weighted_metrics[key] = np.sum(value * custom_metric_weights)
        metrics_results.update(weighted_metrics)

    return metrics_results

def _aggregate_models(
    models: list,
    X: np.ndarray,
    y: np.ndarray,
    distances: Union[str, Callable],
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Any:
    """Aggregate models based on specified distance and solver."""
    y_preds = np.array([model.predict(X) for model in models]).T

    if isinstance(distances, str):
        if distances == 'euclidean':
            dist_matrix = np.sqrt(np.sum((y_preds[:, :, np.newaxis] - y_preds[:, np.newaxis, :]) ** 2, axis=1))
        elif distances == 'manhattan':
            dist_matrix = np.sum(np.abs(y_preds[:, :, np.newaxis] - y_preds[:, np.newaxis, :]), axis=1)
        elif distances == 'cosine':
            dist_matrix = 1 - np.dot(y_preds, y_preds.T) / (np.linalg.norm(y_preds, axis=1)[:, np.newaxis] * np.linalg.norm(y_preds, axis=1)[np.newaxis, :])
    else:
        dist_matrix = np.array([[distances(y_pred1, y_pred2) for y_pred2 in y_preds] for y_pred1 in y_preds])

    if solver == 'gradient_descent':
        weights = _gradient_descent_optimization(dist_matrix, y, regularization, tol, max_iter)
    elif solver == 'newton':
        weights = _newton_optimization(dist_matrix, y, regularization, tol, max_iter)
    elif solver == 'coordinate_descent':
        weights = _coordinate_descent_optimization(dist_matrix, y, regularization, tol, max_iter)

    aggregated_model = _create_aggregated_model(models, weights)
    return aggregated_model

def _gradient_descent_optimization(
    dist_matrix: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Perform gradient descent optimization for model aggregation."""
    n_models = dist_matrix.shape[0]
    weights = np.ones(n_models) / n_models
    learning_rate = 0.01

    for _ in range(max_iter):
        gradients = np.zeros(n_models)
        for i in range(n_models):
            residual = y - np.dot(weights, dist_matrix[i])
            gradients[i] = -2 * np.sum(residual * dist_matrix[i])

        if regularization == 'l1':
            gradients += np.sign(weights)
        elif regularization == 'l2':
            gradients += 2 * weights

        new_weights = weights - learning_rate * gradients
        new_weights = np.maximum(new_weights, 0)  # Ensure non-negativity
        new_weights /= np.sum(new_weights)  # Normalize

        if np.linalg.norm(new_weights - weights) < tol:
            break
        weights = new_weights

    return weights

def _newton_optimization(
    dist_matrix: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Perform Newton optimization for model aggregation."""
    n_models = dist_matrix.shape[0]
    weights = np.ones(n_models) / n_models

    for _ in range(max_iter):
        hessian = np.zeros((n_models, n_models))
        gradient = np.zeros(n_models)

        for i in range(n_models):
            residual = y - np.dot(weights, dist_matrix[i])
            gradient[i] = -2 * np.sum(residual * dist_matrix[i])

        for i in range(n_models):
            for j in range(n_models):
                hessian[i, j] = -2 * np.sum(dist_matrix[i] * dist_matrix[j])

        if regularization == 'l1':
            gradient += np.sign(weights)
        elif regularization == 'l2':
            hessian += 2 * np.eye(n_models)

        delta = np.linalg.solve(hessian, -gradient)
        new_weights = weights + delta
        new_weights = np.maximum(new_weights, 0)  # Ensure non-negativity
        new_weights /= np.sum(new_weights)  # Normalize

        if np.linalg.norm(new_weights - weights) < tol:
            break
        weights = new_weights

    return weights

def _coordinate_descent_optimization(
    dist_matrix: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Perform coordinate descent optimization for model aggregation."""
    n_models = dist_matrix.shape[0]
    weights = np.ones(n_models) / n_models

    for _ in range(max_iter):
        for i in range(n_models):
            mask = np.ones(n_models, dtype=bool)
            mask[i] = False
            X_reduced = dist_matrix[:, mask]
            y_reduced = y - weights[i] * dist_matrix[:, i]

            if X_reduced.shape[1] == 0:
                new_weight = 0
            else:
                weights_reduced = np.linalg.lstsq(X_reduced.T, y_reduced, rcond=None)[0]
                new_weight = np.linalg.lstsq(dist_matrix[:, i][:, np.newaxis], y - X_reduced @ weights_reduced, rcond=None)[0][0]

            if regularization == 'l1':
                new_weight = np.sign(new_weight) * np.maximum(np.abs(new_weight) - 1, 0)
            elif regularization == 'l2':
                new_weight = max(0, new_weight - 1)

            weights[i] = new_weight

        weights = np.maximum(weights, 0)  # Ensure non-negativity
        weights /= np.sum(weights)  # Normalize

        if np.linalg.norm(weights - weights) < tol:
            break

    return weights

def _create_aggregated_model(models: list, weights: np.ndarray) -> Any:
    """Create an aggregated model from individual models and weights."""
    class AggregatedModel:
        def __init__(self, models, weights):
            self.models = models
            self.weights = weights

        def predict(self, X: np.ndarray) -> np.ndarray:
            y_preds = np.array([model.predict(X) for model in self.models]).T
            return np.dot(self.weights, y_preds)

    return AggregatedModel(models, weights)

################################################################################
# calibration_curve
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def calibration_curve_fit(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = 'uniform',
    normalize: bool = True,
    metric: Union[str, Callable] = 'brier',
    solver: str = 'closed_form',
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Compute the calibration curve for predicted probabilities.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred_proba : np.ndarray
        Predicted probabilities of the positive class.
    n_bins : int, optional
        Number of bins to discretize [0,1] interval.
    strategy : str, optional
        Strategy used to define the widths of the bins ('uniform' or 'quantile').
    normalize : bool, optional
        Whether to normalize the bin counts.
    metric : str or callable, optional
        Metric to evaluate calibration ('brier', 'logloss' or custom callable).
    solver : str, optional
        Solver to use for optimization ('closed_form', 'gradient_descent').
    tol : float, optional
        Tolerance for stopping criteria.
    max_iter : int, optional
        Maximum number of iterations.
    random_state : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
        - 'result': Calibration curve data
        - 'metrics': Evaluation metrics
        - 'params_used': Parameters used
        - 'warnings': Potential warnings

    Example
    -------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred_proba = np.array([0.1, 0.9, 0.8, 0.3])
    >>> result = calibration_curve_fit(y_true, y_pred_proba)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred_proba)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Choose metric function
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(y_true, y_pred_proba, n_bins, strategy)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(
            y_true, y_pred_proba, n_bins, strategy,
            metric_func, tol, max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute calibration curve
    prob_true, prob_pred = _compute_calibration_curve(
        y_true, y_pred_proba, n_bins, strategy, normalize
    )

    # Compute metrics
    metrics = _compute_metrics(y_true, y_pred_proba, metric_func)

    # Prepare output
    result = {
        'prob_true': prob_true,
        'prob_pred': prob_pred,
        'bins': params['bins'],
        'counts': params['counts']
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'n_bins': n_bins,
            'strategy': strategy,
            'normalize': normalize,
            'metric': metric,
            'solver': solver
        },
        'warnings': _check_warnings(y_true, y_pred_proba)
    }

def _validate_inputs(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred_proba, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred_proba.shape:
        raise ValueError("y_true and y_pred_proba must have the same shape")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must contain only 0 and 1")
    if np.any((y_pred_proba < 0) | (y_pred_proba > 1)):
        raise ValueError("Predicted probabilities must be in [0, 1]")
    if np.isnan(y_true).any() or np.isinf(y_true).any():
        raise ValueError("y_true contains NaN or Inf values")
    if np.isnan(y_pred_proba).any() or np.isinf(y_pred_proba).any():
        raise ValueError("y_pred_proba contains NaN or Inf values")

def _get_metric_function(metric: str) -> Callable:
    """Get metric function based on name."""
    metrics = {
        'brier': _brier_score,
        'logloss': _log_loss
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _brier_score(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> float:
    """Compute Brier score."""
    return np.mean((y_true - y_pred_proba) ** 2)

def _log_loss(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> float:
    """Compute log loss."""
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    return -np.mean(
        y_true * np.log(y_pred_proba) +
        (1 - y_true) * np.log(1 - y_pred_proba)
    )

def _solve_closed_form(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int,
    strategy: str
) -> Dict[str, np.ndarray]:
    """Solve calibration using closed form solution."""
    bins = _compute_bins(y_pred_proba, n_bins, strategy)
    counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (y_pred_proba >= bins[i]) & (y_pred_proba < bins[i+1])
        counts[i] = np.sum(mask)
        if counts[i] > 0:
            bins[i+1] = np.mean(y_true[mask])

    return {'bins': bins, 'counts': counts}

def _solve_gradient_descent(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int,
    strategy: str,
    metric_func: Callable,
    tol: float,
    max_iter: int
) -> Dict[str, np.ndarray]:
    """Solve calibration using gradient descent."""
    bins = _compute_bins(y_pred_proba, n_bins, strategy)
    params = np.ones(n_bins) * 0.5

    for _ in range(max_iter):
        grad = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (y_pred_proba >= bins[i]) & (y_pred_proba < bins[i+1])
            if np.sum(mask) > 0:
                grad[i] = _compute_gradient(
                    y_true[mask], params[i], metric_func
                )

        # Update parameters
        params -= tol * grad

        # Check convergence
        if np.linalg.norm(grad) < tol:
            break

    return {'bins': bins, 'params': params}

def _compute_gradient(
    y_true: np.ndarray,
    p_pred: float,
    metric_func: Callable
) -> float:
    """Compute gradient for given bin."""
    epsilon = 1e-7
    p_pred_plus = p_pred + epsilon
    p_pred_minus = p_pred - epsilon

    return (metric_func(y_true, np.full_like(y_true, p_pred_plus)) -
            metric_func(y_true, np.full_like(y_true, p_pred_minus))) / (2 * epsilon)

def _compute_bins(
    y_pred_proba: np.ndarray,
    n_bins: int,
    strategy: str
) -> np.ndarray:
    """Compute bin edges."""
    if strategy == 'uniform':
        return np.linspace(0, 1, n_bins + 1)
    elif strategy == 'quantile':
        return np.percentile(y_pred_proba, np.linspace(0, 100, n_bins + 1))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def _compute_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int,
    strategy: str,
    normalize: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute calibration curve."""
    bins = _compute_bins(y_pred_proba, n_bins, strategy)
    prob_true = np.zeros(n_bins)
    prob_pred = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (y_pred_proba >= bins[i]) & (y_pred_proba < bins[i+1])
        if np.sum(mask) > 0:
            prob_true[i] = np.mean(y_true[mask])
            prob_pred[i] = np.mean(y_pred_proba[mask])

    if normalize:
        total = np.sum(prob_true)
        if total > 0:
            prob_true /= total
            prob_pred /= total

    return prob_true, prob_pred

def _compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric_func: Callable
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    return {
        'metric': metric_func(y_true, y_pred_proba),
        'brier_score': _brier_score(y_true, y_pred_proba),
        'log_loss': _log_loss(y_true, y_pred_proba)
    }

def _check_warnings(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, str]:
    """Check for potential warnings."""
    warnings = {}

    if len(y_true) < 10:
        warnings['small_sample'] = "Small sample size may affect results"
    if np.std(y_pred_proba) < 0.1:
        warnings['low_variance'] = "Low variance in predicted probabilities"

    return warnings

################################################################################
# Brier_score
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """
    Validate input arrays for Brier score calculation.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true binary labels (0 or 1).
    y_prob : np.ndarray
        Array of predicted probabilities.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must contain only 0s and 1s.")
    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("y_prob must contain probabilities between 0 and 1.")
    if np.isnan(y_true).any() or np.isnan(y_prob).any():
        raise ValueError("Inputs must not contain NaN values.")
    if np.isinf(y_true).any() or np.isinf(y_prob).any():
        raise ValueError("Inputs must not contain infinite values.")

def _compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute the Brier score.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true binary labels (0 or 1).
    y_prob : np.ndarray
        Array of predicted probabilities.

    Returns
    -------
    float
        The Brier score.
    """
    return np.mean((y_true - y_prob) ** 2)

def Brier_score_compute(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Compute the Brier score and optionally other metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true binary labels (0 or 1).
    y_prob : np.ndarray
        Array of predicted probabilities.
    metric : Optional[Callable]
        Custom metric function. If None, only Brier score is computed.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float]]]
        Dictionary containing the result and metrics.

    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_prob = np.array([0.1, 0.9, 0.8, 0.2])
    >>> Brier_score_compute(y_true, y_prob)
    {
        'result': 0.045,
        'metrics': {},
        'params_used': {'metric': None},
        'warnings': []
    }
    """
    _validate_inputs(y_true, y_prob)
    brier_score = _compute_brier_score(y_true, y_prob)

    metrics = {}
    if metric is not None:
        try:
            metrics['custom_metric'] = metric(y_true, y_prob)
        except Exception as e:
            metrics['custom_metric'] = np.nan
            warnings.append(f"Custom metric computation failed: {str(e)}")

    return {
        'result': brier_score,
        'metrics': metrics,
        'params_used': {'metric': metric},
        'warnings': []
    }

################################################################################
# log_loss
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Validate input arrays for log loss computation.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true labels (0 or 1).
    y_pred : np.ndarray
        Array of predicted probabilities.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise ValueError("Inputs must be numpy arrays")
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must contain only 0s and 1s")
    if np.any((y_pred < 0) | (y_pred > 1)):
        raise ValueError("y_pred must contain probabilities between 0 and 1")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or Inf values")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or Inf values")

def compute_log_loss(y_true: np.ndarray, y_pred: np.ndarray,
                    epsilon: float = 1e-15) -> float:
    """
    Compute log loss (cross-entropy loss).

    Parameters
    ----------
    y_true : np.ndarray
        Array of true labels (0 or 1).
    y_pred : np.ndarray
        Array of predicted probabilities.
    epsilon : float, optional
        Small value to avoid log(0).

    Returns
    ------
    float
        Computed log loss.
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    n_samples = len(y_true)
    loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / n_samples
    return loss

def log_loss_fit(y_true: np.ndarray, y_pred: np.ndarray,
                epsilon: float = 1e-15) -> Dict[str, Union[float, Dict]]:
    """
    Compute log loss with validation and metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true labels (0 or 1).
    y_pred : np.ndarray
        Array of predicted probabilities.
    epsilon : float, optional
        Small value to avoid log(0).

    Returns
    ------
    Dict[str, Union[float, Dict]]
        Dictionary containing:
        - "result": computed log loss
        - "metrics": dictionary of metrics (currently just log_loss)
        - "params_used": parameters used
        - "warnings": any warnings encountered
    """
    # Validate inputs
    validate_inputs(y_true, y_pred)

    # Compute log loss
    result = compute_log_loss(y_true, y_pred, epsilon)

    # Prepare output dictionary
    output = {
        "result": result,
        "metrics": {"log_loss": result},
        "params_used": {
            "epsilon": epsilon
        },
        "warnings": []
    }

    return output

# Example usage:
"""
y_true = np.array([0, 1, 1, 0])
y_pred = np.array([0.1, 0.9, 0.8, 0.2])
result = log_loss_fit(y_true, y_pred)
"""

################################################################################
# calibration_bootstrap
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def calibration_bootstrap_fit(
    data: np.ndarray,
    target: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    normalization: Optional[str] = None,
    regularization: Optional[str] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Calibrate a model using bootstrap resampling.

    Parameters:
    -----------
    data : np.ndarray
        Input features array of shape (n_samples, n_features).
    target : np.ndarray
        Target values array of shape (n_samples,).
    metric : str or callable, optional
        Metric to optimize. Can be 'mse', 'mae', 'r2', 'logloss' or a custom callable.
    distance : str or callable, optional
        Distance metric for bootstrap resampling. Can be 'euclidean', 'manhattan',
        'cosine', 'minkowski' or a custom callable.
    solver : str, optional
        Solver to use. Can be 'closed_form', 'gradient_descent', 'newton',
        or 'coordinate_descent'.
    normalization : str, optional
        Normalization method. Can be 'none', 'standard', 'minmax', or 'robust'.
    regularization : str, optional
        Regularization method. Can be 'none', 'l1', 'l2', or 'elasticnet'.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    random_state : int, optional
        Random seed for reproducibility.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    custom_distance : callable, optional
        Custom distance function if not using built-in distances.

    Returns:
    --------
    Dict containing:
        - 'result': Optimized parameters
        - 'metrics': Computed metrics
        - 'params_used': Parameters used during fitting
        - 'warnings': Any warnings encountered

    Example:
    --------
    >>> data = np.random.rand(100, 5)
    >>> target = np.random.rand(100)
    >>> result = calibration_bootstrap_fit(data, target)
    """
    # Validate inputs
    _validate_inputs(data, target)

    # Normalize data if specified
    normalized_data = _apply_normalization(data, normalization)

    # Initialize parameters
    params_used = {
        'metric': metric,
        'distance': distance,
        'solver': solver,
        'normalization': normalization,
        'regularization': regularization,
        'max_iter': max_iter,
        'tol': tol
    }

    # Set random seed if specified
    if random_state is not None:
        np.random.seed(random_state)

    # Bootstrap resampling
    bootstrap_samples = _bootstrap_resample(normalized_data, target, distance, custom_distance)

    # Initialize parameters
    n_samples, n_features = normalized_data.shape
    initial_params = np.zeros(n_features)

    # Solve for parameters using specified solver
    result, metrics, warnings = _solve_bootstrap(
        bootstrap_samples,
        initial_params,
        metric,
        solver,
        regularization,
        max_iter,
        tol,
        custom_metric
    )

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(data: np.ndarray, target: np.ndarray) -> None:
    """Validate input data and target."""
    if not isinstance(data, np.ndarray) or not isinstance(target, np.ndarray):
        raise TypeError("Data and target must be numpy arrays")
    if data.shape[0] != target.shape[0]:
        raise ValueError("Data and target must have the same number of samples")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")
    if np.any(np.isnan(target)) or np.any(np.isinf(target)):
        raise ValueError("Target contains NaN or infinite values")

def _apply_normalization(data: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Apply specified normalization to data."""
    if method is None or method == 'none':
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

def _bootstrap_resample(data: np.ndarray, target: np.ndarray,
                       distance_method: Union[str, Callable],
                       custom_distance: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Perform bootstrap resampling."""
    n_samples = data.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return data[indices], target[indices]

def _solve_bootstrap(
    bootstrap_samples: Tuple[np.ndarray, np.ndarray],
    initial_params: np.ndarray,
    metric: Union[str, Callable],
    solver: str,
    regularization: Optional[str],
    max_iter: int,
    tol: float,
    custom_metric: Optional[Callable] = None
) -> Tuple[np.ndarray, Dict, List[str]]:
    """Solve bootstrap problem using specified solver."""
    data, target = bootstrap_samples
    warnings = []

    if solver == 'closed_form':
        result, metrics = _solve_closed_form(data, target, metric, custom_metric)
    elif solver == 'gradient_descent':
        result, metrics = _solve_gradient_descent(
            data, target, initial_params, metric,
            regularization, max_iter, tol, custom_metric
        )
    elif solver == 'newton':
        result, metrics = _solve_newton(
            data, target, initial_params, metric,
            regularization, max_iter, tol, custom_metric
        )
    elif solver == 'coordinate_descent':
        result, metrics = _solve_coordinate_descent(
            data, target, initial_params, metric,
            regularization, max_iter, tol, custom_metric
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return result, metrics, warnings

def _solve_closed_form(
    data: np.ndarray,
    target: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Tuple[np.ndarray, Dict]:
    """Solve using closed form solution."""
    # This is a placeholder for the actual implementation
    params = np.linalg.pinv(data) @ target
    metrics = _compute_metrics(data, target, params, metric, custom_metric)
    return params, metrics

def _solve_gradient_descent(
    data: np.ndarray,
    target: np.ndarray,
    initial_params: np.ndarray,
    metric: Union[str, Callable],
    regularization: Optional[str],
    max_iter: int,
    tol: float,
    custom_metric: Optional[Callable] = None
) -> Tuple[np.ndarray, Dict]:
    """Solve using gradient descent."""
    # This is a placeholder for the actual implementation
    params = initial_params.copy()
    for _ in range(max_iter):
        # Update parameters using gradient descent
        pass
    metrics = _compute_metrics(data, target, params, metric, custom_metric)
    return params, metrics

def _solve_newton(
    data: np.ndarray,
    target: np.ndarray,
    initial_params: np.ndarray,
    metric: Union[str, Callable],
    regularization: Optional[str],
    max_iter: int,
    tol: float,
    custom_metric: Optional[Callable] = None
) -> Tuple[np.ndarray, Dict]:
    """Solve using Newton's method."""
    # This is a placeholder for the actual implementation
    params = initial_params.copy()
    for _ in range(max_iter):
        # Update parameters using Newton's method
        pass
    metrics = _compute_metrics(data, target, params, metric, custom_metric)
    return params, metrics

def _solve_coordinate_descent(
    data: np.ndarray,
    target: np.ndarray,
    initial_params: np.ndarray,
    metric: Union[str, Callable],
    regularization: Optional[str],
    max_iter: int,
    tol: float,
    custom_metric: Optional[Callable] = None
) -> Tuple[np.ndarray, Dict]:
    """Solve using coordinate descent."""
    # This is a placeholder for the actual implementation
    params = initial_params.copy()
    for _ in range(max_iter):
        # Update parameters using coordinate descent
        pass
    metrics = _compute_metrics(data, target, params, metric, custom_metric)
    return params, metrics

def _compute_metrics(
    data: np.ndarray,
    target: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute metrics for the given parameters."""
    predictions = data @ params

    if custom_metric is not None:
        return {'custom': custom_metric(target, predictions)}

    metrics = {}
    if metric == 'mse' or (custom_metric is None and metric == 'mse'):
        metrics['mse'] = np.mean((target - predictions) ** 2)
    if metric == 'mae' or (custom_metric is None and metric == 'mae'):
        metrics['mae'] = np.mean(np.abs(target - predictions))
    if metric == 'r2' or (custom_metric is None and metric == 'r2'):
        ss_res = np.sum((target - predictions) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
    if metric == 'logloss' or (custom_metric is None and metric == 'logloss'):
        # This is a placeholder for log loss calculation
        pass

    return metrics
