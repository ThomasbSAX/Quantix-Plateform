"""
Quantix – Module filtrage
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# filtre_lineaire
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

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

def _normalize_data(X: np.ndarray, y: np.ndarray,
                   normalization: str = 'standard',
                   custom_norm: Optional[Callable] = None) -> tuple:
    """Normalize input data."""
    if custom_norm is not None:
        X_norm = custom_norm(X)
        y_norm = custom_norm(y.reshape(-1, 1)).flatten()
    elif normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / (X_std + 1e-8)
        y_norm = (y - np.mean(y)) / (np.std(y) + 1e-8)
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        y_min = np.min(y)
        y_max = np.max(y)
        y_norm = (y - y_min) / (y_max - y_min + 1e-8)
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_q75 = np.percentile(X, 75, axis=0)
        X_q25 = np.percentile(X, 25, axis=0)
        X_norm = (X - X_median) / (X_q75 - X_q25 + 1e-8)
        y_median = np.median(y)
        y_q75 = np.percentile(y, 75)
        y_q25 = np.percentile(y, 25)
        y_norm = (y - y_median) / (y_q75 - y_q25 + 1e-8)
    else:
        X_norm = X.copy()
        y_norm = y.copy()

    return X_norm, y_norm

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric: str = 'mse',
                    custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)
    if metric == 'mse' or (metric is None and custom_metric is None):
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    if metric == 'mae' or (metric is None and custom_metric is None):
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric == 'r2' or (metric is None and custom_metric is None):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
    if metric == 'logloss' or (metric is None and custom_metric is None):
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        metrics['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return metrics

def _solve_closed_form(X: np.ndarray, y: np.ndarray,
                      regularization: str = 'none',
                      alpha: float = 1.0) -> np.ndarray:
    """Solve linear filter using closed form solution."""
    if regularization == 'none':
        weights = np.linalg.inv(X.T @ X) @ (X.T @ y)
    elif regularization == 'l1':
        # Using coordinate descent for L1
        weights = _solve_lasso(X, y, alpha)
    elif regularization == 'l2':
        # Ridge regression
        identity = np.eye(X.shape[1])
        weights = np.linalg.inv(X.T @ X + alpha * identity) @ (X.T @ y)
    elif regularization == 'elasticnet':
        # ElasticNet regression
        identity = np.eye(X.shape[1])
        XtX = X.T @ X
        weights_prev = np.zeros(X.shape[1])
        tol = 1e-4
        max_iter = 1000

        for _ in range(max_iter):
            weights_new = np.linalg.inv(XtX + alpha * identity) @ (X.T @ y)
            if np.linalg.norm(weights_new - weights_prev, ord=np.inf) < tol:
                break
            weights_prev = weights_new

        weights = weights_new
    else:
        raise ValueError(f"Unknown regularization type: {regularization}")

    return weights

def _solve_lasso(X: np.ndarray, y: np.ndarray,
                alpha: float = 1.0) -> np.ndarray:
    """Solve Lasso regression using coordinate descent."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    tol = 1e-4
    max_iter = 1000

    for _ in range(max_iter):
        weights_old = weights.copy()
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - np.dot(X, weights) + weights[j] * X_j
            corr = np.dot(X_j, residuals)
            if corr < -alpha / 2:
                weights[j] = (corr + alpha / 2) / np.dot(X_j, X_j)
            elif corr > alpha / 2:
                weights[j] = (corr - alpha / 2) / np.dot(X_j, X_j)
            else:
                weights[j] = 0

        if np.linalg.norm(weights - weights_old) < tol:
            break

    return weights

def filtre_lineaire_fit(X: np.ndarray, y: np.ndarray,
                      normalization: str = 'standard',
                      metric: str = 'mse',
                      regularization: str = 'none',
                      alpha: float = 1.0,
                      solver: str = 'closed_form',
                      custom_norm: Optional[Callable] = None,
                      custom_metric: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Fit a linear filter to the data.

    Parameters:
    -----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    normalization : str or callable
        Normalization method ('standard', 'minmax', 'robust', or custom function)
    metric : str
        Evaluation metric ('mse', 'mae', 'r2', 'logloss')
    regularization : str
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    alpha : float
        Regularization strength
    solver : str
        Solver method ('closed_form', 'gradient_descent', etc.)
    custom_norm : callable, optional
        Custom normalization function
    custom_metric : callable, optional
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = filtre_lineaire_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization, custom_norm)

    # Solve for weights
    if solver == 'closed_form':
        weights = _solve_closed_form(X_norm, y_norm, regularization, alpha)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Make predictions
    y_pred = X_norm @ weights

    # Compute metrics
    metrics = _compute_metrics(y_norm, y_pred, metric, custom_metric)

    # Prepare output
    result = {
        'result': y_pred,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization if custom_norm is None else 'custom',
            'metric': metric if custom_metric is None else 'custom',
            'regularization': regularization,
            'alpha': alpha,
            'solver': solver
        },
        'warnings': []
    }

    return result

################################################################################
# filtre_non_lineaire
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Validate input data and normalizer."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or inf values")

    # Test normalizer
    try:
        _ = normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalizer failed: {str(e)}")

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
) -> Dict[str, float]:
    """Compute specified metric(s)."""
    metrics = {}

    if isinstance(metric, str):
        if metric == "mse":
            metrics["mse"] = np.mean((y_true - y_pred) ** 2)
        elif metric == "mae":
            metrics["mae"] = np.mean(np.abs(y_true - y_pred))
        elif metric == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics["r2"] = 1 - (ss_res / ss_tot)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        metrics["custom"] = metric(y_true, y_pred)
    else:
        raise ValueError("Metric must be string or callable")

    return metrics

def _estimate_parameters(
    X: np.ndarray,
    y: np.ndarray,
    solver: str,
    max_iter: int = 1000,
    tol: float = 1e-4,
    learning_rate: float = 0.01,
) -> np.ndarray:
    """Estimate parameters using specified solver."""
    if solver == "gradient_descent":
        n_features = X.shape[1]
        params = np.zeros(n_features)

        for _ in range(max_iter):
            gradients = 2 * X.T @ (X @ params - y) / len(y)
            params -= learning_rate * gradients

            if np.linalg.norm(gradients) < tol:
                break
        return params
    elif solver == "closed_form":
        return np.linalg.pinv(X.T @ X) @ X.T @ y
    else:
        raise ValueError(f"Unknown solver: {solver}")

def filtre_non_lineaire_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    solver: str = "gradient_descent",
    max_iter: int = 1000,
    tol: float = 1e-4,
    learning_rate: float = 0.01,
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """Fit a non-linear filter to the data.

    Args:
        X: Input features (n_samples, n_features)
        y: Target values (n_samples,)
        normalizer: Function to normalize features
        metric: Metric to evaluate performance
        solver: Optimization algorithm
        max_iter: Maximum iterations for iterative solvers
        tol: Tolerance for convergence
        learning_rate: Learning rate for gradient descent

    Returns:
        Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    _validate_inputs(X, y, normalizer)

    # Normalize data
    X_norm = normalizer(X)
    y_norm = (y - np.mean(y)) / np.std(y) if normalizer is not None else y

    # Estimate parameters
    params = _estimate_parameters(X_norm, y_norm, solver, max_iter, tol, learning_rate)

    # Make predictions
    y_pred = X_norm @ params

    # Compute metrics
    metrics = _compute_metric(y_norm, y_pred, metric)

    return {
        "result": y_pred,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if hasattr(normalizer, "__name__") else "custom",
            "metric": metric if isinstance(metric, str) else "custom",
            "solver": solver,
        },
        "warnings": [],
    }

# Example usage:
"""
X = np.random.rand(100, 5)
y = X @ np.array([1.2, -0.8, 0.5, 1.1, -0.3]) + np.random.normal(0, 0.1, 100)

result = filtre_non_lineaire_fit(
    X,
    y,
    normalizer=lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0),
    metric="mse",
    solver="gradient_descent"
)
"""

################################################################################
# filtrage_frequentiel
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    normalize: str = 'none',
    custom_normalize: Optional[Callable] = None
) -> Dict[str, np.ndarray]:
    """Validate and preprocess input data."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    if normalize == 'none':
        X_normalized = X
    elif normalize == 'standard':
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalize == 'minmax':
        X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalize == 'robust':
        X_normalized = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    elif callable(custom_normalize):
        X_normalized = custom_normalize(X)
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")

    return {'X': X_normalized, 'y': y}

def _compute_fourier_basis(
    n_samples: int,
    n_components: int = 10
) -> np.ndarray:
    """Compute Fourier basis functions."""
    t = np.linspace(0, 2 * np.pi, n_samples)
    basis = np.zeros((n_samples, n_components))
    for k in range(n_components):
        basis[:, k] = np.cos(k * t)
    return basis

def _solve_closed_form(
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Solve using closed-form solution."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    n_iter: int = 1000,
    tol: float = 1e-4
) -> np.ndarray:
    """Solve using gradient descent."""
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    for _ in range(n_iter):
        gradient = 2 * X.T @ (X @ weights - y) / len(y)
        new_weights = weights - learning_rate * gradient
        if np.linalg.norm(new_weights - weights) < tol:
            break
        weights = new_weights
    return weights

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - ss_res / ss_tot
    elif callable(custom_metric):
        metrics['custom'] = custom_metric(y_true, y_pred)
    return metrics

def filtrage_frequentiel_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 10,
    normalize: str = 'none',
    custom_normalize: Optional[Callable] = None,
    solver: str = 'closed_form',
    learning_rate: float = 0.01,
    n_iter: int = 1000,
    tol: float = 1e-4,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Perform frequency-based filtering.

    Parameters:
    - X: Input features (2D array)
    - y: Target values (1D array)
    - n_components: Number of Fourier components
    - normalize: Normalization method ('none', 'standard', 'minmax', 'robust')
    - custom_normalize: Custom normalization function
    - solver: Solver method ('closed_form', 'gradient_descent')
    - learning_rate: Learning rate for gradient descent
    - n_iter: Number of iterations for gradient descent
    - tol: Tolerance for convergence
    - metric: Evaluation metric ('mse', 'mae', 'r2')
    - custom_metric: Custom evaluation function

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    validated = _validate_inputs(X, y, normalize, custom_normalize)
    X_norm = validated['X']
    y_norm = validated['y']

    # Compute Fourier basis
    fourier_basis = _compute_fourier_basis(X.shape[0], n_components)

    # Solve for coefficients
    if solver == 'closed_form':
        coeffs = _solve_closed_form(fourier_basis, y_norm)
    elif solver == 'gradient_descent':
        coeffs = _solve_gradient_descent(fourier_basis, y_norm, learning_rate, n_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute predictions
    y_pred = fourier_basis @ coeffs

    # Compute metrics
    metrics = _compute_metrics(y_norm, y_pred, metric, custom_metric)

    return {
        'result': {'coefficients': coeffs, 'predictions': y_pred},
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'normalize': normalize,
            'solver': solver,
            'metric': metric
        },
        'warnings': []
    }

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.sin(np.linspace(0, 2 * np.pi, 100)) + 0.1 * np.random.randn(100)
result = filtrage_frequentiel_fit(X, y, n_components=5, normalize='standard', solver='closed_form')
"""

################################################################################
# filtrage_spatial
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> None:
    """Validate input arrays for spatial filtering."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays contain NaN values")
    if weights is not None:
        if weights.shape[0] != X.shape[0]:
            raise ValueError("Weights must have the same length as samples")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")

def normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'standard',
    custom_norm: Optional[Callable] = None
) -> tuple:
    """Normalize input data using specified method."""
    if custom_norm is not None:
        X_norm = custom_norm(X)
        y_norm = custom_norm(y.reshape(-1, 1)).flatten()
    elif method == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / (X_std + 1e-8)
        y_norm = (y - np.mean(y)) / (np.std(y) + 1e-8)
    elif method == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        y_min = np.min(y)
        y_max = np.max(y)
        y_norm = (y - y_min) / (y_max - y_min + 1e-8)
    elif method == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - X_median) / (X_iqr + 1e-8)
        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        y_norm = (y - y_median) / (y_iqr + 1e-8)
    else:
        X_norm = X.copy()
        y_norm = y.copy()

    return X_norm, y_norm

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Union[str, list] = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Compute specified metrics between true and predicted values."""
    result = {}

    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if metric == 'mse':
            result['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            result['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            result['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
        elif metric == 'logloss':
            y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
            result['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif custom_metric is not None and metric == 'custom':
            result['custom'] = custom_metric(y_true, y_pred)

    return result

def spatial_filter_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    distance_metric: str = 'euclidean',
    custom_distance: Optional[Callable] = None
) -> np.ndarray:
    """Compute spatial filter using closed-form solution."""
    if custom_distance is not None:
        distance_matrix = custom_distance(X)
    elif distance_metric == 'euclidean':
        distance_matrix = np.sqrt(((X[:, np.newaxis] - X) ** 2).sum(axis=2))
    elif distance_metric == 'manhattan':
        distance_matrix = np.abs(X[:, np.newaxis] - X).sum(axis=2)
    elif distance_metric == 'cosine':
        dot_products = np.dot(X, X.T)
        norms = np.sqrt(np.sum(X**2, axis=1))
        distance_matrix = 1 - (dot_products / np.outer(norms, norms))

    # Add small value to diagonal for numerical stability
    distance_matrix[np.diag_indices_from(distance_matrix)] += 1e-8

    # Compute weights using inverse distance
    W = 1 / (distance_matrix + 1e-8)
    D = np.diag(np.sum(W, axis=1))
    L = D - W

    # Solve the system (L + λI)β = y
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    beta = np.dot(eigenvectors, np.dot(eigenvectors.T, y))

    return beta

def filtrage_spatial_fit(
    X: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    normalization: str = 'standard',
    custom_norm: Optional[Callable] = None,
    distance_metric: str = 'euclidean',
    custom_distance: Optional[Callable] = None,
    solver: str = 'closed_form',
    metrics: Union[str, list] = 'mse',
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """Main function for spatial filtering.

    Example:
        >>> X = np.random.rand(100, 5)
        >>> y = np.random.rand(100)
        >>> result = filtrage_spatial_fit(X, y, normalization='standard', metrics=['mse', 'r2'])
    """
    # Validate inputs
    validate_inputs(X, y, weights)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization, custom_norm)

    # Compute spatial filter
    if solver == 'closed_form':
        beta = spatial_filter_closed_form(X_norm, y_norm, distance_metric, custom_distance)
    else:
        raise ValueError(f"Solver {solver} not implemented")

    # Make predictions
    y_pred = np.dot(X_norm, beta)

    # Compute metrics
    metric_results = compute_metrics(y_norm, y_pred, metrics, custom_metric)

    return {
        'result': y_pred,
        'metrics': metric_results,
        'params_used': {
            'normalization': normalization if custom_norm is None else 'custom',
            'distance_metric': distance_metric if custom_distance is None else 'custom',
            'solver': solver,
            'metrics': metrics
        },
        'warnings': []
    }

################################################################################
# filtre_passe_bas
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    signal: np.ndarray,
    cutoff_freq: float,
    sampling_rate: float
) -> None:
    """Validate input parameters for low-pass filter."""
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal must be a numpy array")
    if signal.ndim != 1:
        raise ValueError("Signal must be 1-dimensional")
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        raise ValueError("Signal contains NaN or infinite values")
    if cutoff_freq <= 0:
        raise ValueError("Cutoff frequency must be positive")
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive")

def _normalization(
    signal: np.ndarray,
    method: str = 'none'
) -> np.ndarray:
    """Apply normalization to the input signal."""
    if method == 'none':
        return signal
    elif method == 'standard':
        return (signal - np.mean(signal)) / np.std(signal)
    elif method == 'minmax':
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    elif method == 'robust':
        median = np.median(signal)
        iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
        return (signal - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_filter_coefficients(
    cutoff_freq: float,
    sampling_rate: float
) -> np.ndarray:
    """Compute filter coefficients for low-pass filter."""
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    numtaps = 101  # Default number of taps for FIR filter
    return np.sinc(2 * normalized_cutoff * (np.arange(numtaps) - (numtaps - 1) / 2))

def _apply_filter(
    signal: np.ndarray,
    coefficients: np.ndarray
) -> np.ndarray:
    """Apply the filter to the signal using convolution."""
    return np.convolve(signal, coefficients, mode='same')

def _compute_metrics(
    original: np.ndarray,
    filtered: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Compute metrics between original and filtered signals."""
    if callable(metric):
        return {"custom_metric": metric(original, filtered)}

    metrics = {}
    if metric == 'mse' or 'all' in metric:
        metrics['mse'] = np.mean((original - filtered) ** 2)
    if metric == 'mae' or 'all' in metric:
        metrics['mae'] = np.mean(np.abs(original - filtered))
    if metric == 'r2' or 'all' in metric:
        ss_res = np.sum((original - filtered) ** 2)
        ss_tot = np.sum((original - np.mean(original)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

def filtre_passe_bas_fit(
    signal: np.ndarray,
    cutoff_freq: float,
    sampling_rate: float,
    normalization_method: str = 'none',
    metric: Union[str, Callable] = 'mse',
    **kwargs
) -> Dict:
    """
    Apply a low-pass filter to the input signal.

    Parameters:
    -----------
    signal : np.ndarray
        Input signal to be filtered.
    cutoff_freq : float
        Cutoff frequency of the filter in Hz.
    sampling_rate : float
        Sampling rate of the signal in Hz.
    normalization_method : str, optional
        Normalization method to apply ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to compute between original and filtered signals.

    Returns:
    --------
    dict
        Dictionary containing the filtered signal, metrics, parameters used,
        and any warnings.
    """
    # Validate inputs
    _validate_inputs(signal, cutoff_freq, sampling_rate)

    # Normalize signal if specified
    normalized_signal = _normalization(signal, normalization_method)

    # Compute filter coefficients
    coefficients = _compute_filter_coefficients(cutoff_freq, sampling_rate)

    # Apply filter
    filtered_signal = _apply_filter(normalized_signal, coefficients)

    # Compute metrics
    metrics = _compute_metrics(signal, filtered_signal, metric)

    return {
        "result": filtered_signal,
        "metrics": metrics,
        "params_used": {
            "cutoff_freq": cutoff_freq,
            "sampling_rate": sampling_rate,
            "normalization_method": normalization_method
        },
        "warnings": []
    }

# Example usage:
# filtered = filtre_passe_bas_fit(
#     signal=np.random.randn(1000),
#     cutoff_freq=5.0,
#     sampling_rate=100.0
# )

################################################################################
# filtre_passe_haut
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    signal: np.ndarray,
    cutoff_freq: float,
    sampling_rate: float
) -> None:
    """Validate input parameters for high-pass filter."""
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal must be a numpy array")
    if signal.ndim != 1:
        raise ValueError("Signal must be 1-dimensional")
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        raise ValueError("Signal contains NaN or infinite values")
    if cutoff_freq <= 0:
        raise ValueError("Cutoff frequency must be positive")
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive")

def _normalize_signal(
    signal: np.ndarray,
    method: str = "standard"
) -> np.ndarray:
    """Normalize signal using specified method."""
    if method == "none":
        return signal
    elif method == "standard":
        return (signal - np.mean(signal)) / np.std(signal)
    elif method == "minmax":
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    elif method == "robust":
        median = np.median(signal)
        iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
        return (signal - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_filter_coefficients(
    cutoff_freq: float,
    sampling_rate: float
) -> np.ndarray:
    """Compute filter coefficients for high-pass filter."""
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    numtaps = 101  # Odd number for symmetric filter
    return np.sinc(2 * normalized_cutoff * (np.arange(numtaps) - (numtaps - 1) / 2))

def _apply_filter(
    signal: np.ndarray,
    coefficients: np.ndarray
) -> np.ndarray:
    """Apply filter to signal using convolution."""
    return np.convolve(signal, coefficients, mode='same')

def _compute_metrics(
    original: np.ndarray,
    filtered: np.ndarray,
    metric_funcs: Dict[str, Callable]
) -> Dict[str, float]:
    """Compute specified metrics between original and filtered signals."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(original, filtered)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def filtre_passe_haut_fit(
    signal: np.ndarray,
    cutoff_freq: float,
    sampling_rate: float,
    normalization_method: str = "standard",
    metric_funcs: Optional[Dict[str, Callable]] = None,
    custom_filter_func: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict, str]]:
    """
    Apply high-pass filter to a signal with configurable options.

    Parameters:
    -----------
    signal : np.ndarray
        Input signal to be filtered (1D array)
    cutoff_freq : float
        Cutoff frequency of the filter in Hz
    sampling_rate : float
        Sampling rate of the signal in Hz
    normalization_method : str, optional
        Normalization method to apply ("none", "standard", "minmax", "robust")
    metric_funcs : dict, optional
        Dictionary of metric functions to compute (keys are names)
    custom_filter_func : callable, optional
        Custom filter function to use instead of default

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": filtered signal
        - "metrics": computed metrics
        - "params_used": parameters used
        - "warnings": any warnings generated

    Example:
    --------
    >>> signal = np.random.randn(1000)
    >>> result = filtre_passe_haut_fit(signal, cutoff_freq=10, sampling_rate=100)
    """
    # Initialize output dictionary
    output = {
        "result": None,
        "metrics": {},
        "params_used": {},
        "warnings": []
    }

    # Validate inputs
    _validate_inputs(signal, cutoff_freq, sampling_rate)

    # Normalize signal
    normalized_signal = _normalize_signal(signal, normalization_method)
    output["params_used"]["normalization"] = normalization_method

    # Apply filter
    if custom_filter_func is not None:
        try:
            filtered_signal = custom_filter_func(normalized_signal, cutoff_freq, sampling_rate)
        except Exception as e:
            output["warnings"].append(f"Custom filter function failed: {str(e)}")
            filtered_signal = _apply_filter(normalized_signal, _compute_filter_coefficients(cutoff_freq, sampling_rate))
    else:
        filtered_signal = _apply_filter(normalized_signal, _compute_filter_coefficients(cutoff_freq, sampling_rate))

    # Compute metrics if requested
    if metric_funcs is not None:
        output["metrics"] = _compute_metrics(signal, filtered_signal, metric_funcs)

    # Store results
    output["result"] = filtered_signal
    output["params_used"]["cutoff_freq"] = cutoff_freq
    output["params_used"]["sampling_rate"] = sampling_rate

    return output

################################################################################
# filtre_passe_bande
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float
) -> None:
    """Validate input parameters for bandpass filter."""
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal must be a numpy array")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or infinite values")
    if lowcut <= 0 or highcut <= 0:
        raise ValueError("Cutoff frequencies must be positive")
    if lowcut >= highcut:
        raise ValueError("Low cutoff must be less than high cutoff")
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")

def _normalize_signal(
    signal: np.ndarray,
    method: str = "standard"
) -> np.ndarray:
    """Normalize signal using specified method."""
    if method == "none":
        return signal
    elif method == "standard":
        return (signal - np.mean(signal)) / np.std(signal)
    elif method == "minmax":
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    elif method == "robust":
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        return (signal - median) / mad
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _design_filter(
    lowcut: float,
    highcut: float,
    fs: float,
    method: str = "butterworth"
) -> Dict[str, np.ndarray]:
    """Design bandpass filter coefficients."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    if method == "butterworth":
        # Butterworth filter design
        b, a = _design_butterworth(low, high)
    elif method == "chebyshev":
        # Chebyshev filter design
        b, a = _design_chebyshev(low, high)
    else:
        raise ValueError(f"Unknown filter design method: {method}")

    return {"numerator": b, "denominator": a}

def _design_butterworth(low: float, high: float) -> tuple:
    """Design Butterworth bandpass filter coefficients."""
    # Implementation of Butterworth filter design
    pass

def _design_chebyshev(low: float, high: float) -> tuple:
    """Design Chebyshev bandpass filter coefficients."""
    # Implementation of Chebyshev filter design
    pass

def _apply_filter(
    signal: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray
) -> np.ndarray:
    """Apply filter to signal using specified coefficients."""
    return np.zeros_like(signal)  # Placeholder for actual implementation

def _compute_metrics(
    original: np.ndarray,
    filtered: np.ndarray,
    metric_funcs: Dict[str, Callable]
) -> Dict[str, float]:
    """Compute specified metrics between original and filtered signals."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(original, filtered)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def filtre_passe_bande_fit(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    normalize_method: str = "standard",
    filter_design: str = "butterworth",
    metric_funcs: Optional[Dict[str, Callable]] = None,
    **kwargs
) -> Dict:
    """
    Apply bandpass filter to a signal with configurable parameters.

    Parameters
    ----------
    signal : np.ndarray
        Input signal to filter.
    lowcut : float
        Low cutoff frequency of the bandpass filter (Hz).
    highcut : float
        High cutoff frequency of the bandpass filter (Hz).
    fs : float
        Sampling frequency of the signal (Hz).
    normalize_method : str, optional
        Normalization method for the input signal.
    filter_design : str, optional
        Method to design the bandpass filter.
    metric_funcs : dict of callables, optional
        Dictionary of metric functions to compute between original and filtered signals.
    **kwargs : dict
        Additional keyword arguments for specific filter designs.

    Returns
    -------
    dict
        Dictionary containing:
        - "result": filtered signal
        - "metrics": computed metrics
        - "params_used": parameters used in the filtering process
        - "warnings": any warnings generated during processing

    Example
    -------
    >>> signal = np.random.randn(1000)
    >>> result = filtre_passe_bande_fit(signal, 5.0, 10.0, 100.0)
    """
    # Validate inputs
    _validate_inputs(signal, lowcut, highcut, fs)

    # Initialize output dictionary
    result_dict = {
        "result": None,
        "metrics": {},
        "params_used": {
            "normalize_method": normalize_method,
            "filter_design": filter_design
        },
        "warnings": []
    }

    try:
        # Normalize signal
        normalized_signal = _normalize_signal(signal, normalize_method)

        # Design filter
        filter_coeffs = _design_filter(lowcut, highcut, fs, filter_design)

        # Apply filter
        filtered_signal = _apply_filter(normalized_signal, **filter_coeffs)

        # Compute metrics if provided
        if metric_funcs is not None:
            result_dict["metrics"] = _compute_metrics(signal, filtered_signal, metric_funcs)

        result_dict["result"] = filtered_signal

    except Exception as e:
        result_dict["warnings"].append(str(e))

    return result_dict

################################################################################
# filtre_rejecte_bande
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    normalize: str = "none"
) -> None:
    """Validate input parameters for band-reject filter."""
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal must be a numpy array")
    if signal.ndim != 1:
        raise ValueError("Signal must be 1-dimensional")
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    if lowcut >= highcut:
        raise ValueError("Low cut frequency must be less than high cut frequency")
    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("Invalid normalization method")

def _normalize_signal(
    signal: np.ndarray,
    method: str = "none"
) -> np.ndarray:
    """Normalize the input signal."""
    if method == "standard":
        return (signal - np.mean(signal)) / np.std(signal)
    elif method == "minmax":
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    elif method == "robust":
        median = np.median(signal)
        iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
        return (signal - median) / iqr
    else:
        return signal.copy()

def _design_band_reject_filter(
    fs: float,
    lowcut: float,
    highcut: float
) -> np.ndarray:
    """Design a band-reject filter using a simple FIR approach."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Simple band-reject filter design (example implementation)
    numtaps = 101
    taps = np.ones(numtaps)

    # Create band-reject filter coefficients
    for i in range(numtaps):
        if low <= abs((i - (numtaps-1)/2) / (numtaps-1)) <= high:
            taps[i] = 0

    return taps / np.sum(taps)  # Normalize coefficients

def _apply_filter(
    signal: np.ndarray,
    filter_coeffs: np.ndarray
) -> np.ndarray:
    """Apply the designed filter to the signal."""
    return np.convolve(signal, filter_coeffs, mode='same')

def _calculate_metrics(
    original: np.ndarray,
    filtered: np.ndarray,
    metric_funcs: Dict[str, Callable]
) -> Dict[str, float]:
    """Calculate performance metrics."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(original, filtered)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def filtre_rejecte_bande_fit(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    normalize: str = "none",
    metric_funcs: Optional[Dict[str, Callable]] = None,
    custom_filter_design: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict, str]]:
    """
    Apply a band-reject filter to the input signal.

    Parameters:
    -----------
    signal : np.ndarray
        Input signal to be filtered.
    fs : float
        Sampling frequency of the signal (Hz).
    lowcut : float
        Low cut-off frequency of the band to reject (Hz).
    highcut : float
        High cut-off frequency of the band to reject (Hz).
    normalize : str, optional
        Normalization method for the input signal ("none", "standard", "minmax", "robust").
    metric_funcs : dict, optional
        Dictionary of metric functions to evaluate filter performance.
    custom_filter_design : callable, optional
        Custom function for designing the band-reject filter.

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": Filtered signal
        - "metrics": Performance metrics
        - "params_used": Parameters used in the filtering process
        - "warnings": Any warnings generated during processing

    Example:
    --------
    >>> signal = np.random.randn(1000)
    >>> fs = 1000
    >>> lowcut = 50
    >>> highcut = 60
    >>> result = filtre_rejecte_bande_fit(signal, fs, lowcut, highcut)
    """
    # Validate inputs
    _validate_inputs(signal, fs, lowcut, highcut, normalize)

    # Initialize output dictionary
    output = {
        "result": None,
        "metrics": {},
        "params_used": {
            "normalize": normalize,
            "fs": fs,
            "lowcut": lowcut,
            "highcut": highcut
        },
        "warnings": []
    }

    # Normalize signal if requested
    normalized_signal = _normalize_signal(signal, normalize)

    try:
        # Design filter (use custom if provided)
        if custom_filter_design is not None:
            filter_coeffs = custom_filter_design(fs, lowcut, highcut)
        else:
            filter_coeffs = _design_band_reject_filter(fs, lowcut, highcut)

        # Apply filter
        filtered_signal = _apply_filter(normalized_signal, filter_coeffs)

        # Calculate metrics if provided
        if metric_funcs is not None:
            output["metrics"] = _calculate_metrics(signal, filtered_signal, metric_funcs)

        # Store result
        output["result"] = filtered_signal

    except Exception as e:
        output["warnings"].append(f"Error during filtering: {str(e)}")

    return output

# Example metric functions
def mse(original: np.ndarray, filtered: np.ndarray) -> float:
    """Mean Squared Error metric."""
    return np.mean((original - filtered) ** 2)

def mae(original: np.ndarray, filtered: np.ndarray) -> float:
    """Mean Absolute Error metric."""
    return np.mean(np.abs(original - filtered))

def r2_score(original: np.ndarray, filtered: np.ndarray) -> float:
    """R-squared metric."""
    ss_res = np.sum((original - filtered) ** 2)
    ss_tot = np.sum((original - np.mean(original)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

################################################################################
# filtrage_median
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(
    data: np.ndarray,
    window_size: int,
    axis: Optional[int] = None
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if window_size <= 0:
        raise ValueError("Window size must be positive.")
    if axis is not None and (axis < -data.ndim or axis >= data.ndim):
        raise ValueError("Axis out of bounds.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values.")

def compute_median_filter(
    data: np.ndarray,
    window_size: int,
    axis: Optional[int] = None
) -> np.ndarray:
    """Compute median filter along the specified axis."""
    if axis is None:
        axis = 0
    pad_width = [(0, 0)] * data.ndim
    pad_width[axis] = (window_size // 2, window_size - 1 - window_size // 2)
    padded_data = np.pad(data, pad_width, mode='reflect')
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[axis]):
        slice_obj = [slice(None)] * data.ndim
        start_idx = i if axis == 0 else slice(None, i + window_size // 2)
        end_idx = i + window_size if axis == 0 else slice(i - window_size // 2, None)
        slice_obj[axis] = slice(start_idx, end_idx)
        window = padded_data[tuple(slice_obj)]
        filtered_data[tuple(slice_obj[:-1] + (i,))] = np.median(window, axis=axis)
    return filtered_data

def filtrage_median_fit(
    data: np.ndarray,
    window_size: int = 3,
    axis: Optional[int] = None,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Apply median filter to the input data.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    window_size : int, optional
        Size of the filtering window (default is 3).
    axis : Optional[int], optional
        Axis along which to apply the filter (default is None).
    metric : str, optional
        Metric to evaluate the filtering ('mse', 'mae', etc.) (default is 'mse').
    custom_metric : Optional[Callable], optional
        Custom metric function (default is None).

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing the filtered data, metrics, and parameters used.
    """
    validate_inputs(data, window_size, axis)
    filtered_data = compute_median_filter(data, window_size, axis)

    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(data, filtered_data)
    else:
        if metric == 'mse':
            metrics['mse'] = np.mean((data - filtered_data) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(data - filtered_data))
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    return {
        'result': filtered_data,
        'metrics': metrics,
        'params_used': {
            'window_size': window_size,
            'axis': axis,
            'metric': metric
        },
        'warnings': []
    }

################################################################################
# filtrage_gaussien
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    normalize: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean'
) -> None:
    """
    Validate input data and parameters.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    normalize : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    distance_metric : str or callable
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable

    Raises
    ------
    ValueError
        If inputs are invalid or parameters are not supported
    """
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

    if normalize not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError(f"Unsupported normalization method: {normalize}")

    if isinstance(distance_metric, str) and distance_metric not in ['euclidean', 'manhattan', 'cosine', 'minkowski']:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

def normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'standard'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize input data.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    method : str
        Normalization method ('standard', 'minmax', 'robust')

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Normalized X and y arrays
    """
    if method == 'none':
        return X, y

    X_normalized = X.copy()
    y_normalized = y.copy()

    if method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1.0
        X_normalized = (X - mean) / std

    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        X_normalized = (X - min_val) / range_val

    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        iqr[iqr == 0] = 1.0
        X_normalized = (X - median) / iqr

    return X_normalized, y_normalized

def compute_distance(
    X: np.ndarray,
    metric: Union[str, Callable] = 'euclidean',
    p: float = 2.0
) -> np.ndarray:
    """
    Compute pairwise distances between samples.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    metric : str or callable
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable
    p : float
        Power parameter for Minkowski distance

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_samples, n_samples)
    """
    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    if isinstance(metric, str):
        if metric == 'euclidean':
            distance_matrix = np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
        elif metric == 'manhattan':
            distance_matrix = np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
        elif metric == 'cosine':
            dot_products = np.dot(X, X.T)
            norms = np.linalg.norm(X, axis=1)[:, np.newaxis]
            distance_matrix = 1 - dot_products / (norms * norms.T)
        elif metric == 'minkowski':
            distance_matrix = np.sum(np.abs(X[:, np.newaxis] - X) ** p, axis=2) ** (1/p)
    else:
        for i in range(n_samples):
            for j in range(n_samples):
                distance_matrix[i, j] = metric(X[i], X[j])

    return distance_matrix

def gaussian_filter(
    distance_matrix: np.ndarray,
    bandwidth: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian filter to distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Distance matrix of shape (n_samples, n_samples)
    bandwidth : float
        Bandwidth parameter for Gaussian kernel

    Returns
    -------
    np.ndarray
        Filtered matrix of shape (n_samples, n_samples)
    """
    return np.exp(-distance_matrix**2 / (2 * bandwidth**2))

def compute_weights(
    filtered_matrix: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute weights from filtered matrix.

    Parameters
    ----------
    filtered_matrix : np.ndarray
        Filtered matrix of shape (n_samples, n_samples)
    normalize : bool
        Whether to normalize weights

    Returns
    -------
    np.ndarray
        Weight matrix of shape (n_samples, n_samples)
    """
    weights = filtered_matrix.copy()
    if normalize:
        row_sums = np.sum(weights, axis=1)
        row_sums[row_sums == 0] = 1.0
        weights /= row_sums[:, np.newaxis]
    return weights

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Union[str, list, Callable] = 'mse'
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True target values of shape (n_samples,)
    y_pred : np.ndarray
        Predicted target values of shape (n_samples,)
    metrics : str or list or callable
        Metrics to compute ('mse', 'mae', 'r2') or list of metrics or custom callable

    Returns
    -------
    Dict[str, float]
        Dictionary of computed metrics
    """
    result = {}

    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if callable(metric):
            result['custom'] = metric(y_true, y_pred)
        elif metric == 'mse':
            result['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            result['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            result['r2'] = 1 - (ss_res / ss_tot)

    return result

def filtrage_gaussien_fit(
    X: np.ndarray,
    y: np.ndarray,
    bandwidth: float = 1.0,
    normalize: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean',
    metrics: Union[str, list, Callable] = 'mse',
    p: float = 2.0,
    normalize_weights: bool = True
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, any]]]:
    """
    Perform Gaussian filtering on input data.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    bandwidth : float
        Bandwidth parameter for Gaussian kernel
    normalize : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    distance_metric : str or callable
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable
    metrics : str or list or callable
        Metrics to compute ('mse', 'mae', 'r2') or list of metrics or custom callable
    p : float
        Power parameter for Minkowski distance
    normalize_weights : bool
        Whether to normalize weights

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, any]]]
        Dictionary containing:
        - 'result': Filtered y values
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in the computation
        - 'warnings': Any warnings generated during computation

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = filtrage_gaussien_fit(X, y, bandwidth=0.5, normalize='standard')
    """
    warnings = []

    # Validate inputs
    validate_inputs(X, y, normalize, distance_metric)

    # Normalize data
    X_normalized, y_normalized = normalize_data(X, y, normalize)

    # Compute distance matrix
    distance_matrix = compute_distance(X_normalized, distance_metric, p)

    # Apply Gaussian filter
    filtered_matrix = gaussian_filter(distance_matrix, bandwidth)

    # Compute weights
    weights = compute_weights(filtered_matrix, normalize_weights)

    # Compute filtered y values
    y_filtered = np.dot(weights, y_normalized)

    # Compute metrics
    metric_results = compute_metrics(y_normalized, y_filtered, metrics)

    # Prepare output
    result = {
        'result': y_filtered,
        'metrics': metric_results,
        'params_used': {
            'bandwidth': bandwidth,
            'normalize': normalize,
            'distance_metric': distance_metric if not callable(distance_metric) else 'custom',
            'metrics': metrics,
            'p': p,
            'normalize_weights': normalize_weights
        },
        'warnings': warnings
    }

    return result

################################################################################
# filtrage_bilateral
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(
    signal: np.ndarray,
    kernel: np.ndarray,
    spatial_sigma: float = 1.0,
    range_sigma: float = 1.0
) -> None:
    """Validate inputs for bilateral filtering."""
    if not isinstance(signal, np.ndarray) or not isinstance(kernel, np.ndarray):
        raise TypeError("signal and kernel must be numpy arrays")
    if signal.ndim != 2 or kernel.ndim != 2:
        raise ValueError("signal and kernel must be 2D arrays")
    if signal.shape != kernel.shape:
        raise ValueError("signal and kernel must have the same dimensions")
    if spatial_sigma <= 0 or range_sigma <= 0:
        raise ValueError("sigmas must be positive")
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        raise ValueError("signal contains NaN or Inf values")

def compute_weights(
    signal: np.ndarray,
    kernel: np.ndarray,
    spatial_sigma: float = 1.0,
    range_sigma: float = 1.0
) -> np.ndarray:
    """Compute weights for bilateral filtering."""
    height, width = signal.shape
    center_y, center_x = height // 2, width // 2

    # Spatial weights
    y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
    spatial_weights = np.exp(-(x**2 + y**2) / (2 * spatial_sigma**2))

    # Range weights
    range_weights = np.exp(-((signal - signal[center_y, center_x])**2) / (2 * range_sigma**2))

    # Combine weights
    weights = spatial_weights * range_weights * kernel

    return weights / np.sum(weights)

def apply_filter(
    signal: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """Apply the bilateral filter to the signal."""
    filtered = np.zeros_like(signal)
    height, width = signal.shape
    center_y, center_x = height // 2, width // 2

    for y in range(height):
        for x in range(width):
            filtered[y, x] = np.sum(signal * weights)

    return filtered

def filtrage_bilateral_fit(
    signal: np.ndarray,
    kernel: Optional[np.ndarray] = None,
    spatial_sigma: float = 1.0,
    range_sigma: float = 1.0,
    normalize_kernel: bool = True
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Apply bilateral filtering to a 2D signal.

    Parameters
    ----------
    signal : np.ndarray
        Input 2D signal to be filtered.
    kernel : np.ndarray, optional
        Filtering kernel. If None, a Gaussian kernel is used.
    spatial_sigma : float, optional
        Standard deviation for the spatial Gaussian.
    range_sigma : float, optional
        Standard deviation for the range Gaussian.
    normalize_kernel : bool, optional
        Whether to normalize the kernel.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': filtered signal
        - 'metrics': dictionary of metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings

    Example
    -------
    >>> signal = np.random.rand(10, 10)
    >>> result = filtrage_bilateral_fit(signal)
    """
    # Validate inputs
    validate_inputs(signal, kernel if kernel is not None else np.ones((3, 3)))

    # Default kernel
    if kernel is None:
        size = max(3, int(6 * spatial_sigma + 1))
        if size % 2 == 0:
            size += 1
        y, x = np.ogrid[-size//2:size//2+1, -size//2:size//2+1]
        kernel = np.exp(-(x**2 + y**2) / (2 * spatial_sigma**2))
        if normalize_kernel:
            kernel = kernel / np.sum(kernel)

    # Compute weights
    weights = compute_weights(signal, kernel, spatial_sigma, range_sigma)

    # Apply filter
    filtered = apply_filter(signal, weights)

    # Calculate metrics
    mse = np.mean((signal - filtered)**2)
    mae = np.mean(np.abs(signal - filtered))

    return {
        'result': filtered,
        'metrics': {'mse': mse, 'mae': mae},
        'params_used': {
            'spatial_sigma': spatial_sigma,
            'range_sigma': range_sigma,
            'normalize_kernel': normalize_kernel
        },
        'warnings': []
    }

################################################################################
# filtrage_adaptatif
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> None:
    """
    Validate input data for adaptive filtering.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    weights : Optional[np.ndarray]
        Sample weights array of shape (n_samples,)

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if weights is not None:
        if weights.shape[0] != X.shape[0]:
            raise ValueError("weights must have the same length as samples")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")

def normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'standard',
    weights: Optional[np.ndarray] = None
) -> tuple:
    """
    Normalize input data according to specified method.

    Parameters
    ----------
    X : np.ndarray
        Input features array
    y : np.ndarray
        Target values array
    method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    weights : Optional[np.ndarray]
        Sample weights

    Returns
    ------
    tuple
        (X_normalized, y_normalized, normalization_params)
    """
    if method == 'none':
        return X.copy(), y.copy(), None

    # Implement different normalization methods
    if method == 'standard':
        mean = np.average(X, axis=0, weights=weights)
        std = np.sqrt(np.average((X - mean)**2, axis=0, weights=weights))
        X_norm = (X - mean) / std
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_norm, y.copy(), {'method': method}

def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute specified metric between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    metric : Union[str, Callable]
        Metric name or custom callable function
    weights : Optional[np.ndarray]
        Sample weights

    Returns
    ------
    float
        Computed metric value
    """
    if callable(metric):
        return metric(y_true, y_pred)

    if metric == 'mse':
        return np.average((y_true - y_pred)**2, weights=weights)
    elif metric == 'mae':
        return np.average(np.abs(y_true - y_pred), weights=weights)
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - ss_res / (ss_tot + 1e-8)
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.average(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), weights=weights)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def adaptive_filter(
    X: np.ndarray,
    y: np.ndarray,
    solver: str = 'closed_form',
    metric: Union[str, Callable] = 'mse',
    normalize_method: str = 'standard',
    weights: Optional[np.ndarray] = None,
    **solver_kwargs
) -> Dict:
    """
    Perform adaptive filtering with configurable parameters.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    solver : str
        Solver method ('closed_form', 'gradient_descent', etc.)
    metric : Union[str, Callable]
        Evaluation metric
    normalize_method : str
        Data normalization method
    weights : Optional[np.ndarray]
        Sample weights array of shape (n_samples,)
    **solver_kwargs
        Additional solver-specific parameters

    Returns
    ------
    Dict
        Dictionary containing:
        - result: Filtering results
        - metrics: Computed metrics
        - params_used: Parameters used in the computation
        - warnings: Any warnings generated
    """
    # Validate inputs
    validate_inputs(X, y, weights)

    # Normalize data
    X_norm, y_norm, norm_params = normalize_data(X, y, normalize_method, weights)

    # Initialize warnings
    warnings = []

    # Solve the filtering problem based on chosen solver
    if solver == 'closed_form':
        from scipy.linalg import solve
        try:
            coefs = solve(X_norm.T @ X_norm, X_norm.T @ y_norm)
        except np.linalg.LinAlgError as e:
            warnings.append(f"Closed form solution failed: {str(e)}")
            # Fallback to gradient descent
            solver = 'gradient_descent'
            solver_kwargs.setdefault('max_iter', 1000)
    elif solver == 'gradient_descent':
        from scipy.optimize import minimize
        def loss(coefs):
            return compute_metric(y_norm, X_norm @ coefs, metric)

        def grad(coefs):
            return 2 * X_norm.T @ (X_norm @ coefs - y_norm)

        res = minimize(loss, np.zeros(X.shape[1]), jac=grad, **solver_kwargs)
        coefs = res.x
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute predictions and metrics
    y_pred = X_norm @ coefs
    main_metric = compute_metric(y_norm, y_pred, metric, weights)

    # Prepare output
    result = {
        'coefficients': coefs,
        'predictions': y_pred
    }

    return {
        'result': result,
        'metrics': {metric: main_metric},
        'params_used': {
            'solver': solver,
            'metric': metric,
            'normalize_method': normalize_method,
            **solver_kwargs
        },
        'warnings': warnings
    }

# Example usage:
"""
X = np.random.rand(100, 5)
y = X @ np.array([2.3, -1.7, 0.5, 1.2, -0.8]) + np.random.normal(0, 0.1, 100)
weights = np.ones(100)

result = adaptive_filter(
    X, y,
    solver='gradient_descent',
    metric='mse',
    normalize_method='standard',
    weights=weights,
    max_iter=100
)
"""

################################################################################
# filtrage_mouvant
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def filtrage_mouvant_fit(
    data: np.ndarray,
    window_size: int = 3,
    method: str = 'simple',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    weights: Optional[np.ndarray] = None
) -> Dict:
    """
    Applique un filtrage mouvant sur les données fournies.

    Parameters
    ----------
    data : np.ndarray
        Données d'entrée à filtrer.
    window_size : int, optional
        Taille de la fenêtre de filtrage (par défaut 3).
    method : str, optional
        Méthode de filtrage ('simple', 'exponentiel') (par défaut 'simple').
    normalizer : Callable, optional
        Fonction de normalisation personnalisée (par défaut None).
    metric : Union[str, Callable], optional
        Métrique d'évaluation ('mse', 'mae', 'r2') ou fonction personnalisée (par défaut 'mse').
    solver : str, optional
        Solveur utilisé ('closed_form', 'gradient_descent') (par défaut 'closed_form').
    regularization : str, optional
        Type de régularisation ('l1', 'l2') (par défaut None).
    tol : float, optional
        Tolérance pour la convergence (par défaut 1e-6).
    max_iter : int, optional
        Nombre maximal d'itérations (par défaut 1000).
    weights : np.ndarray, optional
        Poids pour le filtrage (par défaut None).

    Returns
    -------
    Dict
        Dictionnaire contenant les résultats, métriques et paramètres utilisés.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> result = filtrage_mouvant_fit(data)
    """
    # Validation des entrées
    _validate_inputs(data, window_size, weights)

    # Normalisation
    if normalizer is not None:
        data = normalizer(data)

    # Choix de la méthode de filtrage
    if method == 'simple':
        filtered_data = _simple_moving_average(data, window_size, weights)
    elif method == 'exponentiel':
        filtered_data = _exponential_moving_average(data, window_size)
    else:
        raise ValueError("Méthode de filtrage non reconnue.")

    # Calcul des métriques
    metrics = _compute_metrics(data, filtered_data, metric)

    # Retourne les résultats
    return {
        "result": filtered_data,
        "metrics": metrics,
        "params_used": {
            "window_size": window_size,
            "method": method,
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

def _validate_inputs(
    data: np.ndarray,
    window_size: int,
    weights: Optional[np.ndarray]
) -> None:
    """Valide les entrées fournies."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Les données doivent être un tableau NumPy.")
    if window_size <= 0:
        raise ValueError("La taille de la fenêtre doit être positive.")
    if weights is not None:
        if len(weights) != window_size:
            raise ValueError("Les poids doivent avoir la même taille que la fenêtre.")
        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError("La somme des poids doit être égale à 1.")

def _simple_moving_average(
    data: np.ndarray,
    window_size: int,
    weights: Optional[np.ndarray]
) -> np.ndarray:
    """Calcule la moyenne mobile simple."""
    if weights is None:
        weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def _exponential_moving_average(
    data: np.ndarray,
    window_size: int
) -> np.ndarray:
    """Calcule la moyenne mobile exponentielle."""
    alpha = 2 / (window_size + 1)
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

def _compute_metrics(
    data: np.ndarray,
    filtered_data: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Calcule les métriques d'évaluation."""
    metrics = {}
    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((data - filtered_data) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(data - filtered_data))
        elif metric == 'r2':
            ss_res = np.sum((data - filtered_data) ** 2)
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
        else:
            raise ValueError("Métrique non reconnue.")
    elif callable(metric):
        metrics['custom'] = metric(data, filtered_data)
    else:
        raise TypeError("La métrique doit être une chaîne ou un callable.")
    return metrics

################################################################################
# filtrage_par_transformee
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    transform_func: Optional[Callable] = None
) -> None:
    """
    Validate input data and transformation function.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values array of shape (n_samples,) or None
    transform_func : Optional[Callable]
        Transformation function to apply

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise ValueError("y must be a numpy array or None")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if transform_func is not None and not callable(transform_func):
        raise ValueError("transform_func must be a callable or None")

def apply_normalization(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalization: str = 'standard'
) -> tuple:
    """
    Apply specified normalization to input data.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values array of shape (n_samples,) or None
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')

    Returns
    ------
    tuple
        Normalized X and y (if provided)
    """
    if normalization == 'none':
        return X, y

    # Normalize features
    if normalization == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
    elif normalization == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
    elif normalization == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    # Normalize target if provided
    y_norm = None
    if y is not None:
        if normalization == 'standard':
            mean_y = np.mean(y)
            std_y = np.std(y)
            y_norm = (y - mean_y) / (std_y + 1e-8)
        elif normalization == 'minmax':
            min_y = np.min(y)
            max_y = np.max(y)
            y_norm = (y - min_y) / (max_y - min_y + 1e-8)
        elif normalization == 'robust':
            median_y = np.median(y)
            iqr_y = np.subtract(*np.percentile(y, [75, 25]))
            y_norm = (y - median_y) / (iqr_y + 1e-8)

    return X_norm, y_norm

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Union[str, list] = 'mse',
    metric_funcs: Optional[Dict[str, Callable]] = None
) -> Dict[str, float]:
    """
    Compute specified metrics between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values of shape (n_samples,)
    y_pred : np.ndarray
        Predicted values of shape (n_samples,)
    metrics : Union[str, list]
        Metrics to compute ('mse', 'mae', 'r2', or custom)
    metric_funcs : Optional[Dict[str, Callable]]
        Dictionary of custom metric functions

    Returns
    ------
    Dict[str, float]
        Computed metrics
    """
    if metric_funcs is None:
        metric_funcs = {}

    results = {}

    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))

    default_metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if metric in default_metrics:
            results[metric] = default_metrics[metric](y_true, y_pred)
        elif metric in metric_funcs:
            results[metric] = metric_funcs[metric](y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results

def filtrage_par_transformee_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    transform_func: Optional[Callable] = None,
    normalization: str = 'standard',
    metrics: Union[str, list] = 'mse',
    metric_funcs: Optional[Dict[str, Callable]] = None,
    **kwargs
) -> Dict:
    """
    Fit a filtering model using specified transformation.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values array of shape (n_samples,) or None
    transform_func : Optional[Callable]
        Transformation function to apply
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metrics : Union[str, list]
        Metrics to compute ('mse', 'mae', 'r2', or custom)
    metric_funcs : Optional[Dict[str, Callable]]
        Dictionary of custom metric functions
    **kwargs :
        Additional parameters for the transformation function

    Returns
    ------
    Dict
        Dictionary containing:
        - 'result': Filtered data or model results
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in the computation
        - 'warnings': Any warnings generated

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = filtrage_par_transformee_fit(X, y,
    ...     transform_func=lambda x: np.sin(x),
    ...     normalization='standard',
    ...     metrics=['mse', 'r2'])
    """
    # Validate inputs
    validate_inputs(X, y, transform_func)

    warnings = []

    # Apply normalization
    X_norm, y_norm = apply_normalization(X, y, normalization)

    # Apply transformation if provided
    X_transformed = X_norm
    if transform_func is not None:
        try:
            X_transformed = transform_func(X_norm, **kwargs)
        except Exception as e:
            warnings.append(f"Transformation function failed: {str(e)}")
            X_transformed = X_norm

    # Store results
    result = {
        'X_transformed': X_transformed,
        'y_normalized': y_norm
    }

    # Compute metrics if target is provided
    metrics_result = {}
    if y_norm is not None:
        # For demonstration, we'll use a simple linear regression as the filter
        try:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_transformed, y_norm)
            y_pred = model.predict(X_transformed)

            metrics_result = compute_metrics(y_norm, y_pred, metrics, metric_funcs)
        except ImportError:
            warnings.append("scikit-learn not available for metrics computation")
        except Exception as e:
            warnings.append(f"Metrics computation failed: {str(e)}")

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics_result,
        'params_used': {
            'normalization': normalization,
            'transform_func': transform_func.__name__ if transform_func else None,
            'metrics': metrics
        },
        'warnings': warnings
    }

    return output

################################################################################
# filtrage_par_dictionnaire
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for dictionary filtering."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays contain infinite values")

def normalize_data(X: np.ndarray, y: np.ndarray,
                  normalization: str = 'standard',
                  custom_norm: Optional[Callable] = None) -> tuple:
    """Normalize input data according to specified method."""
    if custom_norm is not None:
        X_norm = custom_norm(X)
        y_norm = custom_norm(y.reshape(-1, 1)).flatten()
    elif normalization == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif normalization == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == 'robust':
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        X_norm = X.copy()
        y_norm = y.copy()

    return X_norm, y_norm

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
                  metric: str = 'mse',
                  custom_metric: Optional[Callable] = None) -> float:
    """Compute specified metric between true and predicted values."""
    if custom_metric is not None:
        return custom_metric(y_true, y_pred)

    if metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def compute_distance(X: np.ndarray, dictionary: np.ndarray,
                    distance_metric: str = 'euclidean',
                    custom_distance: Optional[Callable] = None) -> np.ndarray:
    """Compute distance between data points and dictionary atoms."""
    if custom_distance is not None:
        return np.array([custom_distance(x, dictionary) for x in X])

    if distance_metric == 'euclidean':
        return np.linalg.norm(X[:, np.newaxis] - dictionary, axis=2)
    elif distance_metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis] - dictionary), axis=2)
    elif distance_metric == 'cosine':
        return 1 - np.dot(X, dictionary.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(dictionary, axis=1))
    elif distance_metric == 'minkowski':
        return np.sum(np.abs(X[:, np.newaxis] - dictionary) ** 3, axis=2) ** (1/3)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

def solve_optimization(X: np.ndarray, y: np.ndarray,
                      dictionary: np.ndarray,
                      solver: str = 'closed_form',
                      tol: float = 1e-4,
                      max_iter: int = 1000,
                      custom_solver: Optional[Callable] = None) -> np.ndarray:
    """Solve the dictionary filtering optimization problem."""
    if custom_solver is not None:
        return custom_solver(X, y, dictionary)

    if solver == 'closed_form':
        # Closed form solution for least squares
        return np.linalg.pinv(dictionary.T @ X) @ dictionary.T @ y
    elif solver == 'gradient_descent':
        # Gradient descent implementation
        weights = np.zeros(dictionary.shape[1])
        for _ in range(max_iter):
            grad = 2 * (dictionary @ weights - y) @ X.T
            weights -= tol * grad
        return weights
    elif solver == 'newton':
        # Newton's method implementation
        weights = np.zeros(dictionary.shape[1])
        for _ in range(max_iter):
            resid = dictionary @ weights - y
            grad = 2 * resid @ X.T
            hess = 2 * dictionary.T @ dictionary
            weights -= np.linalg.pinv(hess) @ grad
        return weights
    elif solver == 'coordinate_descent':
        # Coordinate descent implementation
        weights = np.zeros(dictionary.shape[1])
        for _ in range(max_iter):
            resid = dictionary @ weights - y
            for i in range(weights.shape[0]):
                grad_i = 2 * (dictionary[:, i].T @ resid)
                weights[i] -= tol * grad_i
        return weights
    else:
        raise ValueError(f"Unknown solver: {solver}")

def filtrage_par_dictionnaire_fit(X: np.ndarray, y: np.ndarray,
                                dictionary: np.ndarray,
                                normalization: str = 'standard',
                                metric: str = 'mse',
                                distance_metric: str = 'euclidean',
                                solver: str = 'closed_form',
                                tol: float = 1e-4,
                                max_iter: int = 1000,
                                custom_norm: Optional[Callable] = None,
                                custom_metric: Optional[Callable] = None,
                                custom_distance: Optional[Callable] = None,
                                custom_solver: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Perform dictionary filtering on input data.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - dictionary: Dictionary atoms to use for filtering
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - metric: Performance metric to compute
    - distance_metric: Distance metric for dictionary atoms
    - solver: Optimization solver to use
    - tol: Tolerance for optimization
    - max_iter: Maximum iterations for iterative solvers
    - custom_norm: Custom normalization function
    - custom_metric: Custom metric function
    - custom_distance: Custom distance function
    - custom_solver: Custom solver function

    Returns:
    Dictionary containing:
    - result: Filtered output
    - metrics: Computed performance metrics
    - params_used: Parameters used in the computation
    - warnings: Any warnings generated during computation

    Example:
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> dictionary = np.random.rand(5, 3)
    >>> result = filtrage_par_dictionnaire_fit(X, y, dictionary)
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization, custom_norm)

    # Compute distances
    distances = compute_distance(X_norm, dictionary, distance_metric, custom_distance)

    # Solve optimization problem
    weights = solve_optimization(X_norm, y_norm, dictionary,
                               solver, tol, max_iter, custom_solver)

    # Compute filtered result
    result = dictionary @ weights

    # Compute metrics
    metrics = {
        'primary': compute_metric(y_norm, result, metric, custom_metric),
        'distance_mean': np.mean(distances),
        'distance_std': np.std(distances)
    }

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization if custom_norm is None else 'custom',
            'metric': metric if custom_metric is None else 'custom',
            'distance_metric': distance_metric if custom_distance is None else 'custom',
            'solver': solver if custom_solver is None else 'custom'
        },
        'warnings': []
    }

    return output

################################################################################
# filtrage_par_apprentissage
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def filtrage_par_apprentissage_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalisation: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fonction principale pour le filtrage par apprentissage.

    Parameters:
    -----------
    X : np.ndarray
        Matrice des caractéristiques (n_samples, n_features)
    y : np.ndarray
        Vecteur cible (n_samples,)
    normalisation : str, optional
        Type de normalisation ('none', 'standard', 'minmax', 'robust')
    distance_metric : str or callable, optional
        Métrique de distance ('euclidean', 'manhattan', 'cosine', 'minkowski') ou fonction personnalisée
    solver : str, optional
        Solveur ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    regularization : str, optional
        Type de régularisation ('none', 'l1', 'l2', 'elasticnet')
    tol : float, optional
        Tolérance pour la convergence
    max_iter : int, optional
        Nombre maximal d'itérations
    custom_metric : callable, optional
        Fonction de métrique personnalisée

    Returns:
    --------
    Dict[str, Any]
        Dictionnaire contenant les résultats, métriques, paramètres utilisés et avertissements
    """
    # Validation des entrées
    _validate_inputs(X, y)

    # Normalisation
    X_normalized = _apply_normalization(X, normalisation)

    # Choix du solveur
    if solver == 'closed_form':
        params = _solve_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(X_normalized, y, tol=tol, max_iter=max_iter)
    elif solver == 'newton':
        params = _solve_newton(X_normalized, y, tol=tol, max_iter=max_iter)
    elif solver == 'coordinate_descent':
        params = _solve_coordinate_descent(X_normalized, y, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Application de la régularisation
    if regularization:
        params = _apply_regularization(params, X_normalized, y, regularization)

    # Calcul des métriques
    metrics = _compute_metrics(X_normalized, y, params, custom_metric)

    # Retour des résultats
    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'distance_metric': distance_metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validation des entrées."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be a 2D array and y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _apply_normalization(X: np.ndarray, normalisation: str) -> np.ndarray:
    """Application de la normalisation."""
    if normalisation == 'standard':
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalisation == 'minmax':
        X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalisation == 'robust':
        X_normalized = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        X_normalized = X
    return X_normalized

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Résolution par forme fermée."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Résolution par descente de gradient."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y) / len(y)
        new_params = params - learning_rate * gradient
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params
    return params

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Résolution par méthode de Newton."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y) / len(y)
        hessian = 2 * X.T @ X / len(y)
        new_params = params - np.linalg.pinv(hessian) @ gradient
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params
    return params

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Résolution par descente de coordonnées."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residual = y - X @ params + params[j] * X_j
            params[j] = np.sum(X_j * residual) / np.sum(X_j ** 2)
        if np.linalg.norm(params - params) < tol:
            break
    return params

def _apply_regularization(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    regularization: str
) -> np.ndarray:
    """Application de la régularisation."""
    if regularization == 'l1':
        params = _solve_lasso(X, y)
    elif regularization == 'l2':
        params = _solve_ridge(X, y)
    elif regularization == 'elasticnet':
        params = _solve_elasticnet(X, y)
    return params

def _solve_lasso(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Résolution Lasso."""
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=1.0)
    model.fit(X, y)
    return model.coef_

def _solve_ridge(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Résolution Ridge."""
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model.coef_

def _solve_elasticnet(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Résolution ElasticNet."""
    from sklearn.linear_model import ElasticNet
    model = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model.fit(X, y)
    return model.coef_

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Calcul des métriques."""
    y_pred = X @ params
    metrics = {
        'mse': np.mean((y - y_pred) ** 2),
        'mae': np.mean(np.abs(y - y_pred)),
        'r2': 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    }
    if custom_metric:
        metrics['custom'] = custom_metric(y, y_pred)
    return metrics
