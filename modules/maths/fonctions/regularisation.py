"""
Quantix – Module regularisation
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# ridge_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, y: np.ndarray,
                   normalization: str = 'standard') -> tuple:
    """Normalize data according to specified method."""
    if normalization == 'none':
        return X, y
    elif normalization == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / std
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_normalized = (y - y_mean) / y_std
    elif normalization == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_normalized = (X - min_val) / (max_val - min_val + 1e-8)
        y_min = np.min(y)
        y_max = np.max(y)
        y_normalized = (y - y_min) / (y_max - y_min + 1e-8)
    elif normalization == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - median) / (iqr + 1e-8)
        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        y_normalized = (y - y_median) / (y_iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    return X_normalized, y_normalized

def _compute_ridge_penalty(alpha: float,
                          regularization: str = 'l2',
                          weights: Optional[np.ndarray] = None) -> Callable:
    """Compute ridge penalty function."""
    if regularization == 'none':
        return lambda w: 0
    elif regularization == 'l1':
        if weights is None:
            return lambda w: alpha * np.sum(np.abs(w))
        else:
            return lambda w: alpha * np.sum(weights * np.abs(w))
    elif regularization == 'l2':
        if weights is None:
            return lambda w: alpha * np.sum(w**2)
        else:
            return lambda w: alpha * np.sum(weights * (w**2))
    elif regularization == 'elasticnet':
        if weights is None:
            return lambda w: alpha * (np.sum(np.abs(w)) + np.sum(w**2))
        else:
            return lambda w: alpha * (np.sum(weights * np.abs(w)) + np.sum(weights * (w**2)))
    else:
        raise ValueError(f"Unknown regularization method: {regularization}")

def _closed_form_solver(X: np.ndarray, y: np.ndarray,
                       alpha: float = 1.0) -> np.ndarray:
    """Closed form solution for ridge regression."""
    I = np.eye(X.shape[1])
    return np.linalg.inv(X.T @ X + alpha * I) @ (X.T @ y)

def _gradient_descent_solver(X: np.ndarray, y: np.ndarray,
                            alpha: float = 1.0,
                            learning_rate: float = 0.01,
                            max_iter: int = 1000,
                            tol: float = 1e-4) -> np.ndarray:
    """Gradient descent solver for ridge regression."""
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        gradient = 2 * (X.T @ (X @ w - y) + alpha * w)
        w -= learning_rate * gradient
        current_loss = np.sum((X @ w - y)**2) + alpha * np.sum(w**2)

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return w

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric_funcs: Optional[Dict[str, Callable]] = None) -> Dict:
    """Compute metrics for regression."""
    default_metrics = {
        'mse': lambda yt, yp: np.mean((yt - yp)**2),
        'mae': lambda yt, yp: np.mean(np.abs(yt - yp)),
        'r2': lambda yt, yp: 1 - np.sum((yt - yp)**2) / np.sum((yt - np.mean(yt))**2)
    }

    metrics = default_metrics if metric_funcs is None else metric_funcs
    results = {}

    for name, func in metrics.items():
        try:
            results[name] = func(y_true, y_pred)
        except Exception as e:
            results[f"{name}_error"] = str(e)

    return results

def ridge_regression_fit(X: np.ndarray, y: np.ndarray,
                        alpha: float = 1.0,
                        normalization: str = 'standard',
                        regularization: str = 'l2',
                        solver: str = 'closed_form',
                        weights: Optional[np.ndarray] = None,
                        metric_funcs: Optional[Dict[str, Callable]] = None) -> Dict:
    """
    Perform ridge regression with configurable parameters.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    alpha : float, optional
        Regularization strength, by default 1.0
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), by default 'standard'
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet'), by default 'l2'
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent'), by default 'closed_form'
    weights : Optional[np.ndarray], optional
        Weights for regularization, by default None
    metric_funcs : Optional[Dict[str, Callable]], optional
        Custom metrics functions, by default None

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used and warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = ridge_regression_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization)

    # Compute penalty function
    penalty_func = _compute_ridge_penalty(alpha, regularization, weights)

    # Solve for coefficients
    if solver == 'closed_form':
        coef = _closed_form_solver(X_norm, y_norm, alpha)
    elif solver == 'gradient_descent':
        coef = _gradient_descent_solver(X_norm, y_norm,
                                      alpha=alpha,
                                      learning_rate=0.01,
                                      max_iter=1000,
                                      tol=1e-4)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Make predictions
    y_pred = X_norm @ coef

    # Compute metrics
    metrics = _compute_metrics(y_norm, y_pred, metric_funcs)

    # Prepare results
    result = {
        'result': {
            'coefficients': coef,
            'predictions': y_pred
        },
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'normalization': normalization,
            'regularization': regularization,
            'solver': solver
        },
        'warnings': []
    }

    return result

################################################################################
# lasso_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or infinite values")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("y contains NaN or infinite values")

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

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
                  metric: Union[str, Callable]) -> float:
    """Compute specified metric."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def lasso_regression_fit(X: np.ndarray, y: np.ndarray,
                        alpha: float = 1.0,
                        max_iter: int = 1000,
                        tol: float = 1e-4,
                        normalize_method: str = 'standard',
                        metric: Union[str, Callable] = 'mse',
                        solver: str = 'coordinate_descent') -> Dict:
    """Fit Lasso regression model with specified parameters."""
    validate_inputs(X, y)
    X_norm = normalize_data(X, method=normalize_method)

    if solver == 'coordinate_descent':
        coef = coordinate_descent_solver(X_norm, y, alpha, max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    y_pred = X_norm @ coef
    result_metric = compute_metric(y, y_pred, metric)

    return {
        "result": coef,
        "metrics": {"metric": result_metric},
        "params_used": {
            "alpha": alpha,
            "max_iter": max_iter,
            "tol": tol,
            "normalize_method": normalize_method,
            "metric": metric,
            "solver": solver
        },
        "warnings": []
    }

def coordinate_descent_solver(X: np.ndarray, y: np.ndarray,
                             alpha: float, max_iter: int, tol: float) -> np.ndarray:
    """Coordinate descent solver for Lasso regression."""
    n_features = X.shape[1]
    coef = np.zeros(n_features)
    for _ in range(max_iter):
        old_coef = coef.copy()
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - X @ coef + coef[j] * X_j
            corr = np.dot(X_j, residuals)
            if corr < -alpha / 2:
                coef[j] = (corr - alpha / 2) / np.dot(X_j, X_j)
            elif corr > alpha / 2:
                coef[j] = (corr + alpha / 2) / np.dot(X_j, X_j)
            else:
                coef[j] = 0
        if np.linalg.norm(coef - old_coef) < tol:
            break
    return coef

################################################################################
# elastic_net
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def normalize_data(X: np.ndarray, y: Optional[np.ndarray], method: str = 'standard') -> tuple:
    """Normalize data using specified method."""
    if method == 'none':
        return X, y
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / std
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_normalized = (X - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if y is not None:
        y_mean = np.mean(y)
        y_std = np.std(y)
        if method == 'standard':
            y_normalized = (y - y_mean) / y_std
        elif method == 'minmax':
            y_min = np.min(y)
            y_max = np.max(y)
            y_normalized = (y - y_min) / (y_max - y_min + 1e-8)
        else:
            y_normalized = y
    else:
        y_normalized = None

    return X_normalized, y_normalized

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str = 'mse') -> float:
    """Compute specified metric between true and predicted values."""
    if metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def elastic_net_closed_form(X: np.ndarray, y: np.ndarray,
                           alpha_l1: float = 1.0, alpha_l2: float = 1.0) -> np.ndarray:
    """Closed form solution for Elastic Net regression."""
    n_samples, n_features = X.shape
    I = np.eye(n_features)
    penalty = alpha_l1 * np.linalg.norm(X, ord=1, axis=0) + alpha_l2 * np.trace(np.dot(X.T, X))
    if penalty == 0:
        raise ValueError("Penalty term cannot be zero")
    coefficients = np.linalg.solve(np.dot(X.T, X) + alpha_l2 * I,
                                 np.dot(X.T, y) - alpha_l1 * np.sign(np.dot(X.T, X)))
    return coefficients

def elastic_net_gradient_descent(X: np.ndarray, y: np.ndarray,
                               alpha_l1: float = 1.0, alpha_l2: float = 1.0,
                               learning_rate: float = 0.01, n_iter: int = 1000,
                               tol: float = 1e-4) -> np.ndarray:
    """Gradient descent solver for Elastic Net regression."""
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(n_iter):
        gradients = np.dot(X.T, (np.dot(X, coefficients) - y)) / n_samples
        l1_grad = alpha_l1 * np.sign(coefficients)
        l2_grad = 2 * alpha_l2 * coefficients
        gradients += (l1_grad + l2_grad) / n_samples

        coefficients -= learning_rate * gradients
        current_loss = np.mean((np.dot(X, coefficients) - y) ** 2)

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return coefficients

def elastic_net_coordinate_descent(X: np.ndarray, y: np.ndarray,
                                 alpha_l1: float = 1.0, alpha_l2: float = 1.0,
                                 max_iter: int = 1000, tol: float = 1e-4) -> np.ndarray:
    """Coordinate descent solver for Elastic Net regression."""
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    prev_coefficients = coefficients.copy()
    Xy = np.dot(X.T, y)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - np.dot(X, coefficients) + coefficients[j] * X_j
            corr = np.dot(X_j, residuals)
            Rss = np.sum(residuals ** 2)

            a = (np.dot(X_j, X_j) + alpha_l2)
            b = corr - np.sign(coefficients[j]) * alpha_l1

            if a == 0:
                coefficients[j] = 0
            else:
                coefficients[j] = np.sign(b) * np.maximum(np.abs(b) - alpha_l1, 0) / a

        if np.linalg.norm(coefficients - prev_coefficients, ord=np.inf) < tol:
            break
        prev_coefficients = coefficients.copy()

    return coefficients

def elastic_net_fit(X: np.ndarray, y: np.ndarray,
                   alpha_l1: float = 1.0, alpha_l2: float = 1.0,
                   normalize_method: str = 'standard',
                   solver: str = 'closed_form', max_iter: int = 1000,
                   tol: float = 1e-4, learning_rate: float = 0.01,
                   metric: str = 'mse', custom_metric: Optional[Callable] = None) -> Dict:
    """
    Elastic Net regression with configurable parameters.

    Parameters:
    - X: Feature matrix of shape (n_samples, n_features)
    - y: Target vector of shape (n_samples,)
    - alpha_l1: L1 regularization parameter
    - alpha_l2: L2 regularization parameter
    - normalize_method: Normalization method ('none', 'standard', 'minmax', 'robust')
    - solver: Solver method ('closed_form', 'gradient_descent', 'coordinate_descent')
    - max_iter: Maximum number of iterations
    - tol: Tolerance for stopping criteria
    - learning_rate: Learning rate for gradient descent
    - metric: Evaluation metric ('mse', 'mae', 'r2', 'logloss')
    - custom_metric: Custom metric function

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, method=normalize_method)

    # Choose solver
    if solver == 'closed_form':
        coefficients = elastic_net_closed_form(X_norm, y_norm, alpha_l1, alpha_l2)
    elif solver == 'gradient_descent':
        coefficients = elastic_net_gradient_descent(X_norm, y_norm,
                                                  alpha_l1, alpha_l2,
                                                  learning_rate, max_iter, tol)
    elif solver == 'coordinate_descent':
        coefficients = elastic_net_coordinate_descent(X_norm, y_norm,
                                                    alpha_l1, alpha_l2,
                                                    max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Make predictions
    y_pred = np.dot(X_norm, coefficients)

    # Compute metrics
    if custom_metric is not None:
        custom_score = custom_metric(y_norm, y_pred)
    else:
        custom_score = None

    metrics = {
        'mse': compute_metric(y_norm, y_pred, 'mse'),
        'mae': compute_metric(y_norm, y_pred, 'mae'),
        'r2': compute_metric(y_norm, y_pred, 'r2')
    }

    if metric in metrics:
        primary_metric = metrics[metric]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Prepare output
    result = {
        'coefficients': coefficients,
        'intercept': 0,  # Elastic Net doesn't have an intercept term in this implementation
        'y_pred': y_pred
    }

    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'alpha_l1': alpha_l1,
            'alpha_l2': alpha_l2,
            'normalize_method': normalize_method,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol,
            'learning_rate': learning_rate
        },
        'warnings': []
    }

    if custom_score is not None:
        output['metrics']['custom'] = custom_score

    return output

################################################################################
# dropout
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def dropout_fit(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    p_dropout: float = 0.5,
    n_iterations: int = 100,
    random_state: Optional[int] = None,
    metric: Union[str, Callable] = "mse",
    normalize: str = "standard",
    **kwargs
) -> Dict:
    """
    Applique la méthode de dropout pour la regularisation d'un modèle.

    Parameters
    ----------
    X : np.ndarray
        Matrice des features de forme (n_samples, n_features).
    y : np.ndarray
        Vecteur des cibles de forme (n_samples,).
    model : Callable
        Fonction d'entraînement du modèle prenant X et y en entrée.
    p_dropout : float, optional
        Probabilité de dropout (0 <= p_dropout < 1), par défaut 0.5.
    n_iterations : int, optional
        Nombre d'itérations de dropout, par défaut 100.
    random_state : Optional[int], optional
        Graine aléatoire pour la reproductibilité, par défaut None.
    metric : Union[str, Callable], optional
        Métrique d'évaluation ("mse", "mae", "r2" ou callable), par défaut "mse".
    normalize : str, optional
        Type de normalisation ("none", "standard", "minmax", "robust"), par défaut "standard".
    **kwargs
        Arguments supplémentaires pour le modèle.

    Returns
    -------
    Dict
        Dictionnaire contenant les résultats, métriques, paramètres utilisés et avertissements.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> def simple_model(X, y):
    ...     return np.linalg.lstsq(X, y, rcond=None)[0]
    >>> result = dropout_fit(X, y, simple_model)
    """
    # Validation des entrées
    _validate_inputs(X, y)

    # Normalisation
    X_normalized = _normalize_data(X, normalize)

    # Initialisation du générateur aléatoire
    rng = np.random.RandomState(random_state)

    # Initialisation des résultats
    results = []
    metrics_values = []

    for _ in range(n_iterations):
        # Appliquer le dropout
        X_dropout = _apply_dropout(X_normalized, p_dropout, rng)

        # Entraîner le modèle
        model_params = model(X_dropout, y)

        # Prédire sur les données originales
        y_pred = model_params @ X_normalized.T

        # Calculer la métrique
        metric_value = _compute_metric(y, y_pred, metric)

        results.append(model_params)
        metrics_values.append(metric_value)

    # Calculer les statistiques des résultats
    final_params = np.mean(results, axis=0)
    std_params = np.std(results, axis=0)

    # Calculer la métrique finale
    final_metric = _compute_metric(y, final_params @ X_normalized.T, metric)

    return {
        "result": final_params,
        "metrics": {"value": final_metric, "values": metrics_values},
        "params_used": {
            "p_dropout": p_dropout,
            "n_iterations": n_iterations,
            "metric": metric,
            "normalize": normalize
        },
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Valide les entrées X et y."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X et y doivent être des tableaux NumPy.")
    if X.ndim != 2:
        raise ValueError("X doit être une matrice 2D.")
    if y.ndim != 1:
        raise ValueError("y doit être un vecteur 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X et y doivent avoir le même nombre d'échantillons.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X et y ne doivent pas contenir de NaN.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("X et y ne doivent pas contenir d'infini.")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalise les données selon la méthode spécifiée."""
    if method == "none":
        return X
    elif method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Méthode de normalisation inconnue: {method}")

def _apply_dropout(X: np.ndarray, p_dropout: float, rng: np.random.RandomState) -> np.ndarray:
    """Applique le dropout aux données."""
    if p_dropout < 0 or p_dropout >= 1:
        raise ValueError("p_dropout doit être dans [0, 1).")
    mask = rng.binomial(1, 1 - p_dropout, size=X.shape)
    return X * mask

def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: Union[str, Callable]) -> float:
    """Calcule la métrique spécifiée."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    else:
        raise ValueError(f"Métrique inconnue: {metric}")

################################################################################
# early_stopping
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Inputs contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Inputs contain infinite values")

def normalize_data(X: np.ndarray, y: np.ndarray, method: str = 'standard') -> tuple:
    """Normalize input data using specified method."""
    if method == 'none':
        return X, y
    elif method == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # Avoid division by zero
        X_normalized = (X - X_mean) / X_std
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std == 0:
            y_std = 1
        y_normalized = (y - y_mean) / y_std
    elif method == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
        y_min = np.min(y)
        y_max = np.max(y)
        y_normalized = (y - y_min) / (y_max - y_min + 1e-8)
    elif method == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_iqr[X_iqr == 0] = 1
        X_normalized = (X - X_median) / X_iqr
        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        if y_iqr == 0:
            y_iqr = 1
        y_normalized = (y - y_median) / y_iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return X_normalized, y_normalized

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str = 'mse') -> float:
    """Compute specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def early_stopping_fit(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable,
    n_iter: int = 100,
    tol: float = 1e-4,
    patience: int = 5,
    metric: Union[str, Callable] = 'mse',
    normalizer: str = 'standard',
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Early stopping implementation for model training.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - model_func: Callable that implements the model training
    - n_iter: Maximum number of iterations
    - tol: Tolerance for improvement
    - patience: Number of iterations to wait without improvement before stopping
    - metric: Metric to monitor for early stopping
    - normalizer: Normalization method ('none', 'standard', 'minmax', 'robust')
    - random_state: Random seed for reproducibility

    Returns:
    Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalizer)

    # Initialize variables
    best_metric = np.inf if metric in ['mse', 'mae'] else -np.inf
    best_params = None
    best_iteration = 0
    no_improvement_count = 0

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Early stopping loop
    for i in range(n_iter):
        # Train model
        params = model_func(X_norm, y_norm)

        # Make predictions
        y_pred = np.dot(X_norm, params['coef']) + params.get('intercept', 0)

        # Compute metric
        current_metric = compute_metric(y_norm, y_pred, metric)

        # Check for improvement
        if (metric in ['mse', 'mae'] and current_metric < best_metric - tol) or \
           (metric in ['r2', 'logloss'] and current_metric > best_metric + tol):
            best_metric = current_metric
            best_params = params.copy()
            best_iteration = i + 1
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Early stopping condition
        if no_improvement_count >= patience:
            break

    # Prepare results
    result = {
        'result': best_params,
        'metrics': {
            'best_metric': best_metric,
            'best_iteration': best_iteration
        },
        'params_used': {
            'n_iter': n_iter,
            'tol': tol,
            'patience': patience,
            'metric': metric,
            'normalizer': normalizer
        },
        'warnings': []
    }

    if no_improvement_count >= patience:
        result['warnings'].append(f"Early stopping triggered at iteration {best_iteration}")

    return result

# Example usage:
"""
def simple_linear_model(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    # Simple linear regression model for demonstration
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    coef, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
    return {'coef': coef[1:], 'intercept': coef[0]}

X_example = np.random.rand(100, 5)
y_example = np.dot(X_example, np.array([2.3, -1.7, 0.5, 1.2, -0.8])) + np.random.normal(0, 0.1, 100)

result = early_stopping_fit(
    X_example,
    y_example,
    model_func=simple_linear_model,
    n_iter=50,
    tol=1e-4,
    patience=3,
    metric='mse',
    normalizer='standard'
)
"""

################################################################################
# weight_decay
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for weight decay."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

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

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
                   metric: Union[str, Callable]) -> float:
    """Compute specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == 'logloss':
        return -np.mean(y_true * np.log(y_pred + 1e-8) +
                       (1 - y_true) * np.log(1 - y_pred + 1e-8))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def closed_form_solution(X: np.ndarray, y: np.ndarray,
                         alpha: float = 1.0) -> np.ndarray:
    """Compute closed form solution with L2 regularization."""
    I = np.eye(X.shape[1])
    return np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y

def gradient_descent_solver(X: np.ndarray, y: np.ndarray,
                            alpha: float = 1.0, lr: float = 0.01,
                            max_iter: int = 1000, tol: float = 1e-4) -> np.ndarray:
    """Solve using gradient descent with L2 regularization."""
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = X.T @ (X @ weights - y) + 2 * alpha * weights
        new_weights = weights - lr * gradient
        if np.linalg.norm(new_weights - weights) < tol:
            break
        weights = new_weights
    return weights

def weight_decay_fit(X: np.ndarray, y: np.ndarray,
                     normalization: str = 'standard',
                     metric: Union[str, Callable] = 'mse',
                     solver: str = 'closed_form',
                     alpha: float = 1.0,
                     **solver_kwargs) -> Dict:
    """
    Perform weight decay regularization.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable
        Metric to evaluate performance ('mse', 'mae', 'r2', 'logloss')
    solver : str
        Solver to use ('closed_form', 'gradient_descent')
    alpha : float
        Regularization strength
    **solver_kwargs :
        Additional solver-specific parameters

    Returns:
    --------
    dict
        Dictionary containing results, metrics, and other information
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm = normalize_data(X, normalization)

    # Solve for weights
    if solver == 'closed_form':
        weights = closed_form_solution(X_norm, y, alpha)
    elif solver == 'gradient_descent':
        weights = gradient_descent_solver(X_norm, y, alpha, **solver_kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Make predictions
    y_pred = X_norm @ weights

    # Compute metrics
    metrics = {
        'train_metric': compute_metric(y, y_pred, metric),
        'weights_norm': np.linalg.norm(weights)
    }

    # Prepare output
    result = {
        'result': y_pred,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver,
            'alpha': alpha
        },
        'warnings': []
    }

    return result

# Example usage:
"""
X = np.random.rand(100, 5)
y = X @ np.array([1.2, -3.4, 0.5, 2.1, -0.8]) + np.random.normal(0, 0.1, 100)
result = weight_decay_fit(X, y, normalization='standard', metric='r2')
"""

################################################################################
# pruning
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def pruning_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalize: str = "standard",
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
    Perform pruning regularization on a dataset.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalize : str, optional
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : str or callable, optional
        Metric to optimize: "mse", "mae", "r2", "logloss", or custom callable.
    distance : str, optional
        Distance metric: "euclidean", "manhattan", "cosine", or "minkowski".
    solver : str, optional
        Solver method: "closed_form", "gradient_descent", "newton", or "coordinate_descent".
    regularization : str, optional
        Regularization type: "none", "l1", "l2", or "elasticnet".
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
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = pruning_fit(X, y, normalize="standard", metric="mse")
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _normalize_data(X, method=normalize)

    # Prepare metric and distance functions
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    if isinstance(distance, str):
        distance_func = _get_distance_function(distance)
    else:
        distance_func = distance

    # Solve the problem
    if solver == "closed_form":
        params, metrics = _solve_closed_form(X_normalized, y, metric_func, distance_func)
    elif solver == "gradient_descent":
        params, metrics = _solve_gradient_descent(X_normalized, y, metric_func, distance_func,
                                                 regularization=regularization, tol=tol, max_iter=max_iter)
    elif solver == "newton":
        params, metrics = _solve_newton(X_normalized, y, metric_func, distance_func,
                                       regularization=regularization, tol=tol, max_iter=max_iter)
    elif solver == "coordinate_descent":
        params, metrics = _solve_coordinate_descent(X_normalized, y, metric_func, distance_func,
                                                   regularization=regularization, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Prepare output
    result = {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalize": normalize,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": _check_warnings(params, metrics)
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _normalize_data(X: np.ndarray, method: str = "standard") -> np.ndarray:
    """Normalize the input data."""
    if method == "none":
        return X
    elif method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _get_metric_function(metric: str) -> Callable:
    """Get the metric function based on the input string."""
    metrics = {
        "mse": _mse,
        "mae": _mae,
        "r2": _r2,
        "logloss": _logloss
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance_function(distance: str) -> Callable:
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

def _logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Log Loss."""
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Euclidean distance."""
    if Y is None:
        Y = X
    return np.sqrt(np.sum((X[:, np.newaxis] - Y) ** 2, axis=2))

def _manhattan_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Manhattan distance."""
    if Y is None:
        Y = X
    return np.sum(np.abs(X[:, np.newaxis] - Y), axis=2)

def _cosine_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Cosine distance."""
    if Y is None:
        Y = X
    dot_products = np.dot(X, Y.T)
    norms = np.sqrt(np.sum(X ** 2, axis=1))[:, np.newaxis] * np.sqrt(np.sum(Y ** 2, axis=1))
    return 1 - (dot_products / (norms + 1e-8))

def _minkowski_distance(X: np.ndarray, Y: Optional[np.ndarray] = None, p: float = 2) -> np.ndarray:
    """Minkowski distance."""
    if Y is None:
        Y = X
    return np.sum(np.abs(X[:, np.newaxis] - Y) ** p, axis=2) ** (1 / p)

def _solve_closed_form(X: np.ndarray, y: np.ndarray, metric_func: Callable,
                       distance_func: Callable) -> tuple:
    """Solve the problem using closed-form solution."""
    # Placeholder for actual implementation
    params = np.linalg.pinv(X) @ y
    metrics = {"metric": metric_func(y, X @ params)}
    return params, metrics

def _solve_gradient_descent(X: np.ndarray, y: np.ndarray, metric_func: Callable,
                            distance_func: Callable, regularization: Optional[str],
                            tol: float, max_iter: int) -> tuple:
    """Solve the problem using gradient descent."""
    # Placeholder for actual implementation
    params = np.zeros(X.shape[1])
    metrics = {}
    return params, metrics

def _solve_newton(X: np.ndarray, y: np.ndarray, metric_func: Callable,
                  distance_func: Callable, regularization: Optional[str],
                  tol: float, max_iter: int) -> tuple:
    """Solve the problem using Newton's method."""
    # Placeholder for actual implementation
    params = np.zeros(X.shape[1])
    metrics = {}
    return params, metrics

def _solve_coordinate_descent(X: np.ndarray, y: np.ndarray, metric_func: Callable,
                              distance_func: Callable, regularization: Optional[str],
                              tol: float, max_iter: int) -> tuple:
    """Solve the problem using coordinate descent."""
    # Placeholder for actual implementation
    params = np.zeros(X.shape[1])
    metrics = {}
    return params, metrics

def _check_warnings(params: np.ndarray, metrics: Dict) -> list:
    """Check for warnings in the results."""
    warnings = []
    if np.any(np.isnan(params)):
        warnings.append("Parameters contain NaN values")
    if np.any(np.isinf(params)):
        warnings.append("Parameters contain infinite values")
    return warnings

################################################################################
# data_augmentation
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def data_augmentation_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = lambda a, b: np.linalg.norm(a - b),
    solver: str = 'closed_form',
    regularization: str = 'none',
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    metric_func: Callable[[np.ndarray, np.ndarray], float] = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
    custom_params: Optional[Dict] = None
) -> Dict:
    """
    Perform data augmentation with various configurable options.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    y : Optional[np.ndarray]
        Target values if available. Shape (n_samples,).
    normalizer : Callable
        Function to normalize the input data.
    distance_metric : Callable
        Distance metric function for regularization.
    solver : str
        Solver to use ('closed_form', 'gradient_descent', etc.).
    regularization : str
        Type of regularization ('none', 'l1', 'l2', 'elasticnet').
    alpha : float
        Regularization strength.
    max_iter : int
        Maximum number of iterations for iterative solvers.
    tol : float
        Tolerance for convergence.
    random_state : Optional[int]
        Random seed for reproducibility.
    metric_func : Callable
        Function to compute the evaluation metric.
    custom_params : Optional[Dict]
        Additional parameters for custom solvers or metrics.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = normalizer(X)

    # Initialize parameters
    params_used = {
        'normalization': normalizer.__name__,
        'distance_metric': distance_metric.__name__,
        'solver': solver,
        'regularization': regularization,
        'alpha': alpha,
        'max_iter': max_iter,
        'tol': tol
    }

    # Solve the problem based on solver choice
    if solver == 'closed_form':
        result = _solve_closed_form(X_normalized, y, regularization, alpha)
    elif solver == 'gradient_descent':
        result = _solve_gradient_descent(X_normalized, y, regularization, alpha, max_iter, tol, random_state)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(y, result['y_pred'], metric_func)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

    return output

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None.")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or (y is not None and np.any(np.isnan(y))):
        raise ValueError("Input data contains NaN values.")

def _solve_closed_form(X: np.ndarray, y: Optional[np.ndarray], regularization: str, alpha: float) -> Dict:
    """Solve the problem using closed-form solution."""
    if y is None:
        raise ValueError("Target values (y) are required for closed-form solution.")

    n_samples, n_features = X.shape
    XTX = X.T @ X

    if regularization == 'l2':
        penalty = alpha * np.eye(n_features)
        XTX += penalty
    elif regularization == 'l1':
        raise NotImplementedError("L1 regularization not implemented for closed-form solution.")
    elif regularization == 'elasticnet':
        raise NotImplementedError("ElasticNet regularization not implemented for closed-form solution.")

    XTy = X.T @ y
    coef = np.linalg.solve(XTX, XTy)
    y_pred = X @ coef

    return {
        'coef': coef,
        'y_pred': y_pred
    }

def _solve_gradient_descent(
    X: np.ndarray,
    y: Optional[np.ndarray],
    regularization: str,
    alpha: float,
    max_iter: int,
    tol: float,
    random_state: Optional[int]
) -> Dict:
    """Solve the problem using gradient descent."""
    if y is None:
        raise ValueError("Target values (y) are required for gradient descent.")

    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    coef = np.random.randn(n_features)
    prev_coef = None

    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, coef, regularization, alpha)
        coef -= gradient

        if prev_coef is not None and np.linalg.norm(coef - prev_coef) < tol:
            break

        prev_coef = coef.copy()

    y_pred = X @ coef
    return {
        'coef': coef,
        'y_pred': y_pred
    }

def _compute_gradient(X: np.ndarray, y: np.ndarray, coef: np.ndarray, regularization: str, alpha: float) -> np.ndarray:
    """Compute the gradient for gradient descent."""
    n_samples = X.shape[0]
    y_pred = X @ coef
    error = y_pred - y

    gradient = (X.T @ error) / n_samples

    if regularization == 'l2':
        gradient += 2 * alpha * coef
    elif regularization == 'l1':
        gradient += alpha * np.sign(coef)
    elif regularization == 'elasticnet':
        gradient += alpha * (0.5 * coef + np.sign(coef))

    return gradient

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric_func: Callable) -> Dict:
    """Compute evaluation metrics."""
    return {
        'metric': metric_func(y_true, y_pred)
    }

# Example usage:
"""
X = np.random.randn(100, 5)
y = np.random.randn(100)

result = data_augmentation_fit(
    X,
    y,
    normalizer=lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0),
    distance_metric=lambda a, b: np.linalg.norm(a - b),
    solver='gradient_descent',
    regularization='l2',
    alpha=0.1,
    max_iter=1000,
    tol=1e-4,
    random_state=42
)
"""

################################################################################
# batch_normalization
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    X: np.ndarray,
    epsilon: float = 1e-7
) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("X contains infinite values")

def _compute_moments(
    X: np.ndarray,
    axis: int = 0
) -> Dict[str, np.ndarray]:
    """Compute mean and variance along specified axis."""
    mean = np.mean(X, axis=axis)
    var = np.var(X, axis=axis)
    return {"mean": mean, "variance": var}

def _standardize(
    X: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    epsilon: float = 1e-7
) -> np.ndarray:
    """Standardize data using given mean and variance."""
    return (X - mean) / np.sqrt(variance + epsilon)

def _minmax_scale(
    X: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> np.ndarray:
    """Min-max scale data."""
    min_X = np.min(X, axis=0)
    max_X = np.max(X, axis=0)
    return min_val + (X - min_X) * (max_val - min_val) / (max_X - min_X + 1e-7)

def _robust_scale(
    X: np.ndarray
) -> np.ndarray:
    """Robust scale data using median and IQR."""
    med = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    iqr = q75 - q25
    return (X - med) / (iqr + 1e-7)

def _compute_metrics(
    X: np.ndarray,
    X_normalized: np.ndarray,
    metric_func: Callable = None
) -> Dict[str, float]:
    """Compute metrics for normalized data."""
    metrics = {}
    if metric_func is not None:
        metrics["custom"] = metric_func(X, X_normalized)
    return metrics

def batch_normalization_fit(
    X: np.ndarray,
    normalization_type: str = "standard",
    epsilon: float = 1e-7,
    metric_func: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, np.ndarray]]]:
    """
    Fit batch normalization parameters and normalize data.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    normalization_type : str
        Type of normalization ("none", "standard", "minmax", "robust")
    epsilon : float
        Small value to avoid division by zero
    metric_func : Callable, optional
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": normalized data
        - "metrics": computed metrics
        - "params_used": parameters used for normalization
        - "warnings": any warnings generated

    Example:
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = batch_normalization_fit(X, normalization_type="standard")
    """
    _validate_inputs(X)

    warnings = []
    params_used = {}
    metrics = {}

    if normalization_type == "none":
        X_normalized = X
    elif normalization_type == "standard":
        moments = _compute_moments(X)
        params_used.update(moments)
        X_normalized = _standardize(X, moments["mean"], moments["variance"], epsilon)
    elif normalization_type == "minmax":
        X_normalized = _minmax_scale(X)
    elif normalization_type == "robust":
        X_normalized = _robust_scale(X)
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")

    if metric_func is not None:
        metrics = _compute_metrics(X, X_normalized, metric_func)

    return {
        "result": X_normalized,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# l1_regularization
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input matrices and vectors."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize the input data."""
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

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
                   metric: Union[str, Callable]) -> float:
    """Compute the specified metric."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def l1_regularization_fit(X: np.ndarray, y: np.ndarray,
                          alpha: float = 1.0,
                          normalize_method: str = 'standard',
                          metric: Union[str, Callable] = 'mse',
                          solver: str = 'coordinate_descent',
                          max_iter: int = 1000,
                          tol: float = 1e-4) -> Dict:
    """
    Perform L1 regularization (Lasso regression).

    Parameters
    ----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values vector of shape (n_samples,)
    alpha : float, optional
        Regularization strength (default=1.0)
    normalize_method : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default='standard')
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2', 'logloss') or custom callable (default='mse')
    solver : str, optional
        Solver method ('coordinate_descent') (default='coordinate_descent')
    max_iter : int, optional
        Maximum number of iterations (default=1000)
    tol : float, optional
        Tolerance for stopping criteria (default=1e-4)

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': estimated coefficients
        - 'metrics': computed metrics
        - 'params_used': parameters used in the fit
        - 'warnings': any warnings generated

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = l1_regularization_fit(X, y)
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm = normalize_data(X, method=normalize_method)
    y_norm = normalize_data(y.reshape(-1, 1), method='none').flatten()

    # Initialize warnings
    warnings = []

    # Solve using the specified solver
    if solver == 'coordinate_descent':
        coef = coordinate_descent_solver(X_norm, y_norm, alpha, max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute predictions
    y_pred = X_norm @ coef

    # Compute metrics
    metrics = {
        'train_metric': compute_metric(y_norm, y_pred, metric)
    }

    # Return results
    return {
        'result': coef,
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'normalize_method': normalize_method,
            'metric': metric,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': warnings
    }

def coordinate_descent_solver(X: np.ndarray, y: np.ndarray,
                             alpha: float, max_iter: int, tol: float) -> np.ndarray:
    """Coordinate descent solver for L1 regularization."""
    n_features = X.shape[1]
    coef = np.zeros(n_features)
    prev_coef = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            # Compute residuals without feature j
            r = y - X @ coef + coef[j] * X[:, j]

            # Compute correlation
            corr = np.dot(X[:, j], r)

            if corr < -alpha / 2:
                coef[j] = (corr + alpha / 2) / np.dot(X[:, j], X[:, j])
            elif corr > alpha / 2:
                coef[j] = (corr - alpha / 2) / np.dot(X[:, j], X[:, j])
            else:
                coef[j] = 0

        # Check convergence
        if np.linalg.norm(coef - prev_coef) < tol:
            break

        prev_coef = coef.copy()

    return coef

################################################################################
# l2_regularization
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Inputs contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Inputs contain infinite values")

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

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
                  metric: Union[str, Callable]) -> float:
    """Compute specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def closed_form_solution(X: np.ndarray, y: np.ndarray,
                        alpha: float = 1.0) -> np.ndarray:
    """Compute closed form solution for L2 regularization."""
    I = np.eye(X.shape[1])
    return np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y

def gradient_descent_solver(X: np.ndarray, y: np.ndarray,
                           alpha: float = 1.0, lr: float = 0.01,
                           max_iter: int = 1000, tol: float = 1e-4) -> np.ndarray:
    """Solve using gradient descent."""
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * (X @ weights - y) @ X + 2 * alpha * weights
        new_weights = weights - lr * gradient
        if np.linalg.norm(new_weights - weights) < tol:
            break
        weights = new_weights
    return weights

def l2_regularization_fit(X: np.ndarray, y: np.ndarray,
                         alpha: float = 1.0,
                         normalize_method: str = 'standard',
                         solver: str = 'closed_form',
                         metric: Union[str, Callable] = 'mse',
                         **solver_kwargs) -> Dict:
    """
    Perform L2 regularization on input data.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - alpha: Regularization strength
    - normalize_method: Normalization method ('none', 'standard', 'minmax', 'robust')
    - solver: Solver method ('closed_form' or 'gradient_descent')
    - metric: Metric to compute ('mse', 'mae', 'r2', 'logloss') or callable
    - solver_kwargs: Additional arguments for the solver

    Returns:
    Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm = normalize_data(X, method=normalize_method)

    # Choose solver
    if solver == 'closed_form':
        weights = closed_form_solution(X_norm, y, alpha)
    elif solver == 'gradient_descent':
        weights = gradient_descent_solver(X_norm, y, alpha, **solver_kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Make predictions
    y_pred = X_norm @ weights

    # Compute metrics
    metric_value = compute_metric(y, y_pred, metric)

    return {
        'result': weights,
        'metrics': {metric: metric_value},
        'params_used': {
            'alpha': alpha,
            'normalize_method': normalize_method,
            'solver': solver,
            'metric': metric
        },
        'warnings': []
    }

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.random.rand(100)

result = l2_regularization_fit(X, y,
                             alpha=0.1,
                             normalize_method='standard',
                             solver='closed_form',
                             metric='mse')
"""
