"""
Quantix – Module svm
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# hyperplan
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def hyperplan_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Compute the hyperplane for SVM classification.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target labels of shape (n_samples,) with values in {-1, 1}.
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the input features.
    distance_metric : str
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski').
    solver : str
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : Optional[str]
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function to evaluate the hyperplane.

    Returns
    -------
    Dict[str, Union[Dict, float, str]]
        Dictionary containing the result, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([-1, -1, 1])
    >>> result = hyperplan_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_normalized = normalizer(X)

    # Choose distance metric
    distance_func = _get_distance_function(distance_metric)

    # Solve for hyperplane
    if solver == 'closed_form':
        w, b = _solve_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        w, b = _solve_gradient_descent(X_normalized, y, tol=tol, max_iter=max_iter)
    elif solver == 'newton':
        w, b = _solve_newton(X_normalized, y, tol=tol, max_iter=max_iter)
    elif solver == 'coordinate_descent':
        w, b = _solve_coordinate_descent(X_normalized, y, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization == 'l1':
        w = _apply_l1_regularization(w)
    elif regularization == 'l2':
        w = _apply_l2_regularization(w)
    elif regularization == 'elasticnet':
        w = _apply_elasticnet_regularization(w)

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, w, b, custom_metric)

    # Prepare output
    result = {
        'result': {'w': w, 'b': b},
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'distance_metric': distance_metric,
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
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values.")
    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError("y must contain only -1 and 1.")

def _get_distance_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the distance function based on the metric."""
    if metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric == 'minkowski':
        return lambda x, y: np.sum(np.abs(x - y) ** 3) ** (1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> tuple:
    """Solve for hyperplane using closed-form solution."""
    # This is a placeholder; actual implementation depends on the SVM formulation
    Xty = np.dot(X.T, y)
    XtX = np.dot(X.T, X)
    w = np.linalg.solve(XtX + 1e-6 * np.eye(X.shape[1]), Xty)
    b = np.mean(y - np.dot(X, w))
    return w, b

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> tuple:
    """Solve for hyperplane using gradient descent."""
    w = np.zeros(X.shape[1])
    b = 0.0
    learning_rate = 0.01

    for _ in range(max_iter):
        gradients_w = -np.dot(X.T, y * (1 / (1 + np.exp(y * np.dot(X, w) + b))))
        gradients_b = -np.sum(y * (1 / (1 + np.exp(y * np.dot(X, w) + b))))

        w -= learning_rate * gradients_w
        b -= learning_rate * gradients_b

        if np.linalg.norm(gradients_w) < tol and abs(gradients_b) < tol:
            break

    return w, b

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> tuple:
    """Solve for hyperplane using Newton's method."""
    w = np.zeros(X.shape[1])
    b = 0.0

    for _ in range(max_iter):
        hessian_w = np.dot(X.T * (y * (1 - 1 / (1 + np.exp(-y * np.dot(X, w) - b))) ** 2), X)
        hessian_b = np.sum(y * (1 - 1 / (1 + np.exp(-y * np.dot(X, w) - b))) ** 2)

        gradients_w = -np.dot(X.T, y / (1 + np.exp(y * np.dot(X, w) + b)))
        gradients_b = -np.sum(y / (1 + np.exp(y * np.dot(X, w) + b)))

        hessian = np.block([[hessian_w, np.zeros((X.shape[1], 1))], [np.zeros((1, X.shape[1])), hessian_b]])
        gradients = np.concatenate([gradients_w, [gradients_b]])

        update = np.linalg.solve(hessian + 1e-6 * np.eye(hessian.shape[0]), -gradients)
        w += update[:-1]
        b += update[-1]

        if np.linalg.norm(gradients) < tol:
            break

    return w, b

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> tuple:
    """Solve for hyperplane using coordinate descent."""
    w = np.zeros(X.shape[1])
    b = 0.0

    for _ in range(max_iter):
        for i in range(X.shape[1]):
            X_i = X[:, i]
            w[i] = np.sum(y * (1 / (1 + np.exp(-y * (np.dot(X, w) - w[i] * X_i + b))))) / np.sum(X_i ** 2)

        new_b = np.mean(y - np.dot(X, w))
        if abs(new_b - b) < tol:
            break
        b = new_b

    return w, b

def _apply_l1_regularization(w: np.ndarray) -> np.ndarray:
    """Apply L1 regularization to weights."""
    return np.sign(w) * np.maximum(np.abs(w) - 1, 0)

def _apply_l2_regularization(w: np.ndarray) -> np.ndarray:
    """Apply L2 regularization to weights."""
    return w / (1 + 0.1 * np.linalg.norm(w))

def _apply_elasticnet_regularization(w: np.ndarray) -> np.ndarray:
    """Apply elastic net regularization to weights."""
    l1_w = _apply_l1_regularization(w)
    l2_w = _apply_l2_regularization(w)
    return 0.5 * (l1_w + l2_w)

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute metrics for the hyperplane."""
    predictions = np.dot(X, w) + b
    mse = np.mean((y - predictions) ** 2)
    mae = np.mean(np.abs(y - predictions))
    r2 = 1 - mse / np.var(y)

    metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y, predictions)

    return metrics

################################################################################
# marge_separatrice
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values")

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

def compute_kernel(X: np.ndarray, y: np.ndarray, kernel_func: Callable) -> np.ndarray:
    """Compute the kernel matrix."""
    return np.array([[kernel_func(x, y) for x in X] for y in X])

def compute_margins(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Compute the margins for each sample."""
    return (X @ w + b) * np.where(y > 0, 1, -1)

def marge_separatrice_fit(
    X: np.ndarray,
    y: np.ndarray,
    kernel: Optional[Callable] = None,
    C: float = 1.0,
    epsilon: float = 1e-3,
    max_iter: int = 1000,
    normalize_method: str = 'standard',
    metric: Union[str, Callable] = 'mse'
) -> Dict:
    """Fit the margin separator model.

    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([1, -1, 1])
        >>> result = marge_separatrice_fit(X, y)
    """
    validate_inputs(X, y)
    X_normalized = normalize_data(X, method=normalize_method)

    if kernel is None:
        def linear_kernel(x1, x2):
            return np.dot(x1, x2)
        kernel_func = linear_kernel
    else:
        kernel_func = kernel

    K = compute_kernel(X_normalized, X_normalized, kernel_func)

    # Placeholder for actual SVM implementation
    w = np.zeros(X_normalized.shape[1])
    b = 0.0

    margins = compute_margins(X_normalized, w, b)

    if isinstance(metric, str):
        if metric == 'mse':
            def mse(y_true, y_pred):
                return np.mean((y_true - y_pred) ** 2)
            metric_func = mse
        elif metric == 'mae':
            def mae(y_true, y_pred):
                return np.mean(np.abs(y_true - y_pred))
            metric_func = mae
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        metric_func = metric

    y_pred = np.sign(X_normalized @ w + b)
    score = metric_func(y, y_pred)

    return {
        'result': {
            'w': w,
            'b': b
        },
        'metrics': {
            'score': score
        },
        'params_used': {
            'C': C,
            'epsilon': epsilon,
            'max_iter': max_iter,
            'normalize_method': normalize_method,
            'metric': metric
        },
        'warnings': []
    }

################################################################################
# fonction_noyau
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(X: np.ndarray, Y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or Y.ndim != 1:
        raise ValueError("X must be a 2D array and Y must be a 1D array")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
        raise ValueError("Y contains NaN or infinite values")

def _normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize input data using specified method."""
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

def _compute_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None,
                    kernel_func: Callable = lambda x, y: np.dot(x, y),
                    **kernel_params) -> np.ndarray:
    """Compute kernel matrix."""
    if Y is None:
        Y = X
    return np.array([[kernel_func(x, y) for x in X] for y in Y])

def _compute_dual_coefficients(Y: np.ndarray, kernel_matrix: np.ndarray,
                              C: float = 1.0) -> np.ndarray:
    """Compute dual coefficients using quadratic programming."""
    n_samples = Y.shape[0]
    P = np.outer(Y, Y) * kernel_matrix
    q = -np.ones(n_samples)
    G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
    h = np.hstack((np.zeros(n_samples), C * np.ones(n_samples)))
    A = Y.reshape(1, -1)
    b = 0.0

    from cvxopt import solvers
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b)
    return np.array(solution['x']).flatten()

def _compute_weights(X: np.ndarray, alpha: np.ndarray,
                    kernel_func: Callable) -> np.ndarray:
    """Compute weights for the SVM."""
    return np.sum(alpha[:, np.newaxis] * X, axis=0)

def _compute_metrics(Y_true: np.ndarray, Y_pred: np.ndarray,
                    metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute various metrics."""
    return {name: func(Y_true, Y_pred) for name, func in metric_funcs.items()}

def fonction_noyau_fit(X: np.ndarray, Y: np.ndarray,
                      kernel_func: Callable = lambda x, y: np.dot(x, y),
                      normalization: str = 'standard',
                      C: float = 1.0,
                      metric_funcs: Dict[str, Callable] = None,
                      **kernel_params) -> Dict:
    """
    Fit a kernel SVM model.

    Parameters
    ----------
    X : np.ndarray
        Input features (n_samples, n_features)
    Y : np.ndarray
        Target values (-1 or 1 for classification) (n_samples,)
    kernel_func : Callable
        Kernel function to use
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    C : float
        Regularization parameter
    metric_funcs : Dict[str, Callable]
        Dictionary of metric functions to compute
    kernel_params : dict
        Additional parameters for the kernel function

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> Y = np.random.choice([-1, 1], size=100)
    >>> result = fonction_noyau_fit(X, Y)
    """
    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data
    X_normalized = _normalize_data(X, normalization)

    # Compute kernel matrix
    kernel_matrix = _compute_kernel(X_normalized, kernel_func=kernel_func,
                                  **kernel_params)

    # Compute dual coefficients
    alpha = _compute_dual_coefficients(Y, kernel_matrix, C)

    # Compute weights
    weights = _compute_weights(X_normalized, alpha, kernel_func)

    # Prepare metrics if provided
    metrics = {}
    if metric_funcs is not None:
        Y_pred = np.sign(np.dot(X_normalized, weights))
        metrics = _compute_metrics(Y, Y_pred, metric_funcs)

    # Prepare output
    result = {
        'result': weights,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'C': C,
            'kernel_func': kernel_func.__name__ if hasattr(kernel_func, '__name__') else 'custom',
            'kernel_params': kernel_params
        },
        'warnings': []
    }

    return result

# Example metric functions
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

################################################################################
# kernel_lineaire
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
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

def _normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
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

def _compute_kernel_linear(X: np.ndarray, y: np.ndarray,
                          metric: str = 'euclidean',
                          custom_metric: Optional[Callable] = None,
                          **kwargs) -> np.ndarray:
    """Compute linear kernel matrix."""
    if custom_metric is not None:
        return custom_metric(X, y)
    if metric == 'euclidean':
        return np.dot(X, X.T)
    elif metric == 'manhattan':
        return -np.abs(X[:, np.newaxis] - X).sum(axis=2)
    elif metric == 'cosine':
        return np.dot(X, X.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] *
                                np.linalg.norm(X, axis=1)[np.newaxis, :])
    elif metric == 'minkowski':
        p = kwargs.get('p', 2)
        return -np.power(np.abs(X[:, np.newaxis] - X).sum(axis=2), p)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _solve_closed_form(K: np.ndarray, y: np.ndarray,
                      regularization: str = 'none',
                      C: float = 1.0) -> np.ndarray:
    """Solve SVM using closed form solution."""
    if regularization == 'none':
        return np.linalg.solve(K + C * np.eye(K.shape[0]), y)
    elif regularization == 'l1':
        # Using coordinate descent for L1
        w = np.zeros(K.shape[0])
        for _ in range(100):
            for i in range(len(w)):
                K_i = K[i, :]
                r = y - np.dot(K, w) + w[i] * K_i
                gradient = -np.dot(K_i, r)
                step_size = 1.0 / (C + np.sum(K_i**2))
                w[i] -= step_size * gradient
        return w
    elif regularization == 'l2':
        return np.linalg.solve(K + C * np.eye(K.shape[0]), y)
    else:
        raise ValueError(f"Unknown regularization: {regularization}")

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric_names: list = ['mse', 'r2'],
                    custom_metrics: Optional[Dict[str, Callable]] = None) -> Dict:
    """Compute specified metrics."""
    metrics = {}
    if 'mse' in metric_names:
        metrics['mse'] = np.mean((y_true - y_pred)**2)
    if 'mae' in metric_names:
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if 'r2' in metric_names:
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
    if custom_metrics is not None:
        for name, func in custom_metrics.items():
            metrics[name] = func(y_true, y_pred)
    return metrics

def kernel_lineaire_fit(X: np.ndarray,
                       y: np.ndarray,
                       normalization: str = 'standard',
                       metric: str = 'euclidean',
                       regularization: str = 'none',
                       C: float = 1.0,
                       solver: str = 'closed_form',
                       custom_metric: Optional[Callable] = None,
                       metric_names: list = ['mse', 'r2'],
                       custom_metrics: Optional[Dict[str, Callable]] = None,
                       **kwargs) -> Dict:
    """
    Fit a linear kernel SVM model.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2')
    C : float, optional
        Regularization parameter
    solver : str, optional
        Solver method ('closed_form')
    custom_metric : Callable, optional
        Custom metric function
    metric_names : list, optional
        List of metrics to compute
    custom_metrics : Dict[str, Callable], optional
        Dictionary of custom metrics
    **kwargs :
        Additional parameters for specific solvers/metrics

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': fitted model parameters
        - 'metrics': computed metrics
        - 'params_used': parameters used for fitting
        - 'warnings': any warnings during fitting

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = kernel_lineaire_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = _normalize_data(X, normalization)

    # Compute kernel matrix
    K = _compute_kernel_linear(X_norm, y,
                             metric=metric,
                             custom_metric=custom_metric,
                             **kwargs)

    # Solve for parameters
    if solver == 'closed_form':
        params = _solve_closed_form(K, y,
                                  regularization=regularization,
                                  C=C)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute predictions
    y_pred = np.dot(K, params)

    # Compute metrics
    metrics = _compute_metrics(y, y_pred,
                             metric_names=metric_names,
                             custom_metrics=custom_metrics)

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'regularization': regularization,
            'C': C,
            'solver': solver
        },
        'warnings': []
    }

    return result

################################################################################
# kernel_polynomial
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
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
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

def _polynomial_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None, degree: int = 3) -> np.ndarray:
    """Compute polynomial kernel between X and Y."""
    if Y is None:
        Y = X
    return (np.dot(X, Y.T) + 1) ** degree

def _compute_gram_matrix(X: np.ndarray, kernel_func: Callable, **kernel_params) -> np.ndarray:
    """Compute the Gram matrix using the specified kernel function."""
    return kernel_func(X, **kernel_params)

def _solve_svm_dual(K: np.ndarray, y: np.ndarray, C: float = 1.0) -> np.ndarray:
    """Solve the dual SVM problem using quadratic programming."""
    n_samples = K.shape[0]
    P = np.outer(y, y) * K
    q = -np.ones(n_samples)
    G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
    h = np.hstack((np.zeros(n_samples), C * np.ones(n_samples)))
    A = y.reshape(1, -1)
    b = 0.0

    from cvxopt import solvers, matrix
    solvers.options['show_progress'] = False
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    solution = solvers.qp(P, q, G, h, A, b)
    return np.array(solution['x']).flatten()

def _compute_weights(X: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Compute the weights from the dual solution."""
    return np.dot(alpha * y, X)

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute specified metrics."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(y_true, y_pred)
        except Exception as e:
            metrics[f"{name}_error"] = str(e)
    return metrics

def kernel_polynomial_fit(
    X: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
    C: float = 1.0,
    normalize_method: str = 'standard',
    metric_funcs: Optional[Dict[str, Callable]] = None,
    kernel_func: Callable = _polynomial_kernel
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, Union[str, int]], str]]:
    """
    Fit a polynomial kernel SVM model.

    Parameters
    ----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    degree : int, optional
        Degree of the polynomial kernel, by default 3.
    C : float, optional
        Regularization parameter, by default 1.0.
    normalize_method : str, optional
        Normalization method for the data, by default 'standard'.
    metric_funcs : Dict[str, Callable], optional
        Dictionary of metric functions to compute, by default None.
    kernel_func : Callable, optional
        Kernel function to use, by default _polynomial_kernel.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, Union[str, int]], str]]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _normalize_data(X, method=normalize_method)

    # Compute Gram matrix
    K = _compute_gram_matrix(X_normalized, kernel_func, degree=degree)

    # Solve dual SVM problem
    alpha = _solve_svm_dual(K, y, C)

    # Compute weights
    w = _compute_weights(X_normalized, y, alpha)

    # Predictions (for metrics)
    y_pred = np.sign(np.dot(X_normalized, w))

    # Compute metrics
    default_metrics = {
        'accuracy': lambda y_true, y_pred: np.mean(y_true == y_pred)
    }
    if metric_funcs is None:
        metric_funcs = default_metrics
    else:
        metric_funcs.update(default_metrics)

    metrics = _compute_metrics(y, y_pred, metric_funcs)

    # Prepare output
    result = {
        'weights': w,
        'alpha': alpha,
        'support_vectors': X_normalized[alpha > 1e-5]
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'degree': degree,
            'C': C,
            'normalize_method': normalize_method
        },
        'warnings': []
    }

# Example usage:
# from sklearn.metrics import accuracy_score, f1_score
# X = np.random.rand(100, 5)
# y = np.random.randint(0, 2, 100) * 2 - 1
# metric_funcs = {
#     'accuracy': accuracy_score,
#     'f1': f1_score
# }
# result = kernel_polynomial_fit(X, y, degree=2, C=0.5, metric_funcs=metric_funcs)

################################################################################
# kernel_rbf
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_inputs(X: np.ndarray, Y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or Y.ndim != 1:
        raise ValueError("X must be 2D and Y must be 1D")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
        raise ValueError("Y contains NaN or infinite values")

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

def compute_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None,
                  gamma: float = 1.0) -> np.ndarray:
    """Compute RBF kernel matrix."""
    if Y is None:
        Y = X

    sq_dists = np.sum(X**2, axis=1)[:, np.newaxis] + \
               np.sum(Y**2, axis=1)[np.newaxis, :] - \
               2 * np.dot(X, Y.T)
    return np.exp(-gamma * sq_dists)

def kernel_rbf_fit(X: np.ndarray, Y: np.ndarray,
                  gamma: float = 1.0,
                  C: float = 1.0,
                  kernel_func: Callable = compute_kernel,
                  normalize_method: str = 'standard',
                  tol: float = 1e-3,
                  max_iter: int = 1000) -> Dict[str, Any]:
    """
    Fit RBF kernel SVM model.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features)
    Y : np.ndarray
        Target values of shape (n_samples,)
    gamma : float
        Kernel coefficient
    C : float
        Regularization parameter
    kernel_func : callable
        Kernel function (default: compute_kernel)
    normalize_method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    tol : float
        Tolerance for stopping criterion
    max_iter : int
        Maximum number of iterations

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> Y = np.random.randint(0, 2, size=100)
    >>> result = kernel_rbf_fit(X, Y)
    """
    # Validate inputs
    validate_inputs(X, Y)

    # Normalize data
    X_norm = normalize_data(X, method=normalize_method)

    # Compute kernel matrix
    K = kernel_func(X_norm, gamma=gamma)

    # Add regularization term
    K_aug = np.block([[0, Y], [Y.T, C * np.eye(len(Y)) + (1 - C) * K]])

    # Solve the dual problem
    alpha = np.linalg.solve(K_aug, np.concatenate([np.zeros(len(Y)), Y]))

    # Extract support vectors
    sv_indices = np.where(alpha[1:] > tol)[0]
    alpha_sv = alpha[sv_indices + 1]

    # Calculate bias
    svs = X_norm[sv_indices]
    Y_sv = Y[sv_indices]

    # Calculate intercept
    bias = np.mean(Y_sv - np.sum(alpha_sv * Y_sv * K[sv_indices][:, sv_indices], axis=1))

    # Prepare results
    result = {
        'alpha': alpha_sv,
        'support_vectors': svs,
        'bias': bias,
        'gamma': gamma,
        'C': C
    }

    # Calculate metrics (simplified for example)
    metrics = {
        'n_support_vectors': len(svs),
        'kernel_type': 'rbf'
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'gamma': gamma,
            'C': C,
            'normalize_method': normalize_method
        },
        'warnings': []
    }

def kernel_rbf_predict(X: np.ndarray, model: Dict[str, Any],
                      normalize_method: str = 'standard') -> np.ndarray:
    """
    Predict using fitted RBF kernel SVM model.

    Parameters:
    -----------
    X : np.ndarray
        Test data of shape (n_samples, n_features)
    model : dict
        Fitted model from kernel_rbf_fit
    normalize_method : str
        Normalization method (must match training)

    Returns:
    --------
    np.ndarray
        Predicted values
    """
    # Normalize data with same method as training
    X_norm = normalize_data(X, method=normalize_method)

    # Get model parameters
    alpha = model['result']['alpha']
    svs = model['result']['support_vectors']
    bias = model['result']['bias']
    gamma = model['result']['gamma']

    # Compute kernel between test points and support vectors
    K = compute_kernel(X_norm, svs, gamma=gamma)

    # Calculate decision function
    decision = np.sum(alpha * K, axis=1) + bias

    # Return predictions (0 or 1)
    return np.where(decision >= 0, 1, -1)

################################################################################
# kernel_sigmoide
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values.")

def _normalize_data(X: np.ndarray, normalization: str) -> np.ndarray:
    """Normalize the input data."""
    if normalization == "none":
        return X
    elif normalization == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif normalization == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif normalization == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _sigmoid_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None, gamma: float = 1.0) -> np.ndarray:
    """Compute the sigmoid kernel."""
    if Y is None:
        Y = X
    K = np.dot(X, Y.T)
    return np.tanh(gamma * K)

def _compute_dual_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    kernel_func: Callable,
    C: float = 1.0,
    tol: float = 1e-3,
    max_iter: int = 1000
) -> np.ndarray:
    """Compute the dual coefficients using a simple solver."""
    n_samples = X.shape[0]
    K = kernel_func(X)
    P = np.outer(y, y) * K
    q = -np.ones(n_samples)
    G = -np.eye(n_samples)
    h = np.zeros(n_samples)
    A = y.reshape(1, -1)
    b = 0.0

    # Simple quadratic programming solver (placeholder for actual implementation)
    alpha = np.zeros(n_samples)

    return alpha

def _compute_weights(alpha: np.ndarray, X: np.ndarray, y: np.ndarray, kernel_func: Callable) -> np.ndarray:
    """Compute the weights from dual coefficients."""
    n_samples = X.shape[0]
    K = kernel_func(X)
    w = np.zeros(X.shape[1])
    for i in range(n_samples):
        if alpha[i] > 0:
            w += alpha[i] * y[i] * X[i]
    return w

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute the metrics."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(y_true, y_pred)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def kernel_sigmoide_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalization: str = "standard",
    gamma: float = 1.0,
    C: float = 1.0,
    tol: float = 1e-3,
    max_iter: int = 1000,
    metric_funcs: Optional[Dict[str, Callable]] = None
) -> Dict[str, Any]:
    """
    Fit a sigmoid kernel SVM model.

    Parameters:
    - X: Input data of shape (n_samples, n_features).
    - y: Target values of shape (n_samples,).
    - normalization: Normalization method ("none", "standard", "minmax", "robust").
    - gamma: Kernel coefficient.
    - C: Regularization parameter.
    - tol: Tolerance for stopping criterion.
    - max_iter: Maximum number of iterations.
    - metric_funcs: Dictionary of metric functions to compute.

    Returns:
    - A dictionary containing the results, metrics, parameters used, and warnings.
    """
    _validate_inputs(X, y)
    X_normalized = _normalize_data(X, normalization)

    kernel_func = lambda x, y=None: _sigmoid_kernel(x, y, gamma)
    alpha = _compute_dual_coefficients(X_normalized, y, kernel_func, C, tol, max_iter)
    w = _compute_weights(alpha, X_normalized, y, kernel_func)

    if metric_funcs is None:
        metric_funcs = {}

    # Example of a simple prediction function for metrics
    def predict(X_test: np.ndarray) -> np.ndarray:
        return np.sign(np.dot(X_test, w))

    y_pred = predict(X_normalized)
    metrics = _compute_metrics(y, y_pred, metric_funcs)

    result = {
        "result": {"weights": w},
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "gamma": gamma,
            "C": C,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return result

# Example usage:
"""
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([1, -1, 1])
result = kernel_sigmoide_fit(X, y)
print(result)
"""

################################################################################
# optimisation_dual
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def optimisation_dual_fit(
    X: np.ndarray,
    y: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    C: float = 1.0,
    tol: float = 1e-3,
    max_iter: int = 1000,
    solver: str = 'smoothed',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'hinge',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, Union[Dict, float, str]]:
    """
    Optimisation duale pour SVM.

    Parameters
    ----------
    X : np.ndarray
        Matrice des caractéristiques de dimension (n_samples, n_features).
    y : np.ndarray
        Vecteur des labels de dimension (n_samples,), prenant les valeurs -1 ou 1.
    kernel : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Fonction de noyau pour le calcul des similarités.
    C : float, optional
        Paramètre de régularisation (par défaut 1.0).
    tol : float, optional
        Tolérance pour la convergence (par défaut 1e-3).
    max_iter : int, optional
        Nombre maximal d'itérations (par défaut 1000).
    solver : str, optional
        Solveur à utiliser ('smoothed', 'quadratic', 'coordinate') (par défaut 'smoothed').
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Fonction de normalisation (par défaut None).
    metric : str, optional
        Métrique à utiliser ('hinge', 'squared_hinge') (par défaut 'hinge').
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Fonction de métrique personnalisée (par défaut None).

    Returns
    -------
    Dict[str, Union[Dict, float, str]]
        Dictionnaire contenant les résultats, métriques, paramètres utilisés et avertissements.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> y = np.array([-1, 1])
    >>> def linear_kernel(a, b):
    ...     return np.dot(a, b.T)
    ...
    >>> result = optimisation_dual_fit(X, y, linear_kernel)
    """
    # Validation des entrées
    _validate_inputs(X, y)

    # Normalisation si spécifiée
    if normalizer is not None:
        X = normalizer(X)

    # Initialisation des paramètres
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)

    # Calcul de la matrice de noyau
    K = kernel(X, X)

    # Choix du solveur
    if solver == 'smoothed':
        alpha = _smoothing_algorithm(K, y, C, tol, max_iter)
    elif solver == 'quadratic':
        alpha = _quadratic_programming(K, y, C)
    elif solver == 'coordinate':
        alpha = _coordinate_descent(K, y, C, tol, max_iter)
    else:
        raise ValueError("Solver non reconnu. Utilisez 'smoothed', 'quadratic' ou 'coordinate'.")

    # Calcul des métriques
    metrics = _compute_metrics(X, y, alpha, K, metric, custom_metric)

    # Retour des résultats
    return {
        'result': {'alpha': alpha},
        'metrics': metrics,
        'params_used': {
            'C': C,
            'tol': tol,
            'max_iter': max_iter,
            'solver': solver,
        },
        'warnings': [],
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validation des entrées."""
    if X.ndim != 2:
        raise ValueError("X doit être une matrice 2D.")
    if y.ndim != 1:
        raise ValueError("y doit être un vecteur 1D.")
    if len(X) != len(y):
        raise ValueError("X et y doivent avoir le même nombre d'échantillons.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contient des valeurs non finies.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contient des valeurs non finies.")
    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError("y doit contenir uniquement -1 ou 1.")

def _smoothing_algorithm(
    K: np.ndarray,
    y: np.ndarray,
    C: float,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Algorithme de lissage pour la résolution du problème dual."""
    n_samples = K.shape[0]
    alpha = np.zeros(n_samples)
    for _ in range(max_iter):
        alpha_prev = alpha.copy()
        for i in range(n_samples):
            # Calcul du gradient
            grad = y[i] * np.sum(alpha * y * K[:, i]) - 1
            if grad < -tol:
                alpha[i] = min(C, alpha[i] + grad)
            elif grad > tol:
                alpha[i] = max(0, alpha[i] + grad)
        if np.linalg.norm(alpha - alpha_prev) < tol:
            break
    return alpha

def _quadratic_programming(
    K: np.ndarray,
    y: np.ndarray,
    C: float
) -> np.ndarray:
    """Résolution par programmation quadratique."""
    n_samples = K.shape[0]
    P = np.outer(y, y) * K
    q = -np.ones(n_samples)
    G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
    h = np.hstack((np.zeros(n_samples), C * np.ones(n_samples)))
    A = y.reshape(1, -1)
    b = 0
    alpha = np.zeros(n_samples)
    # Utilisation d'un solveur de programmation quadratique (ex: cvxopt)
    return alpha

def _coordinate_descent(
    K: np.ndarray,
    y: np.ndarray,
    C: float,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Algorithme de descente de coordonnées."""
    n_samples = K.shape[0]
    alpha = np.zeros(n_samples)
    for _ in range(max_iter):
        alpha_prev = alpha.copy()
        for i in range(n_samples):
            # Calcul du gradient
            grad = y[i] * np.sum(alpha * y * K[:, i]) - 1
            if grad < -tol:
                alpha[i] = min(C, alpha[i] + grad)
            elif grad > tol:
                alpha[i] = max(0, alpha[i] + grad)
        if np.linalg.norm(alpha - alpha_prev) < tol:
            break
    return alpha

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    K: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calcul des métriques."""
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(X, y)
    else:
        if metric == 'hinge':
            metrics['hinge_loss'] = _hinge_loss(X, y, alpha, K)
        elif metric == 'squared_hinge':
            metrics['squared_hinge_loss'] = _squared_hinge_loss(X, y, alpha, K)
        else:
            raise ValueError("Métrique non reconnue. Utilisez 'hinge' ou 'squared_hinge'.")
    return metrics

def _hinge_loss(
    X: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    K: np.ndarray
) -> float:
    """Calcul de la perte hinge."""
    return 0.5 * np.sum(alpha) - np.sum(alpha)

def _squared_hinge_loss(
    X: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    K: np.ndarray
) -> float:
    """Calcul de la perte squared hinge."""
    return 0.5 * np.sum(alpha) - np.sum(alpha)

################################################################################
# contraintes_kkt
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def contraintes_kkt_fit(
    X: np.ndarray,
    y: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    C: float = 1.0,
    tol: float = 1e-3,
    max_iter: int = 1000,
    metric: str = 'euclidean',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    solver: str = 'gradient_descent',
    **solver_kwargs
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Compute the KKT constraints for Support Vector Machines.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    kernel : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Kernel function to compute the Gram matrix.
    C : float, optional
        Regularization parameter, by default 1.0.
    tol : float, optional
        Tolerance for stopping criteria, by default 1e-3.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    metric : str, optional
        Distance metric for KKT constraints, by default 'euclidean'.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Normalization function, by default None.
    solver : str, optional
        Solver to use for optimization, by default 'gradient_descent'.
    **solver_kwargs
        Additional keyword arguments for the solver.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - "result": Array of KKT constraint violations.
        - "metrics": Dictionary of computed metrics.
        - "params_used": Dictionary of parameters used.
        - "warnings": List of warnings.

    Examples
    --------
    >>> X = np.random.rand(10, 2)
    >>> y = np.random.randint(0, 2, size=10)
    >>> def linear_kernel(a, b):
    ...     return np.dot(a, b.T)
    ...
    >>> result = contraintes_kkt_fit(X, y, linear_kernel)
    """
    # Validate inputs
    X, y = _validate_inputs(X, y)

    # Normalize data if a normalizer is provided
    if normalizer is not None:
        X = normalizer(X)

    # Compute the Gram matrix
    K = kernel(X, X)

    # Choose solver and compute KKT constraints
    if solver == 'gradient_descent':
        result, metrics = _gradient_descent_solver(
            K, y, C, tol, max_iter, metric, **solver_kwargs
        )
    elif solver == 'coordinate_descent':
        result, metrics = _coordinate_descent_solver(
            K, y, C, tol, max_iter, metric, **solver_kwargs
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Prepare output dictionary
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "C": C,
            "tol": tol,
            "max_iter": max_iter,
            "metric": metric,
            "solver": solver
        },
        "warnings": []
    }

    return output

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Validate input data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix.
    y : np.ndarray
        Target values.

    Returns
    -------
    tuple
        Validated X and y.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or Inf values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or Inf values.")
    return X, y

def _gradient_descent_solver(
    K: np.ndarray,
    y: np.ndarray,
    C: float,
    tol: float,
    max_iter: int,
    metric: str,
    learning_rate: float = 0.01
) -> tuple:
    """
    Solve KKT constraints using gradient descent.

    Parameters
    ----------
    K : np.ndarray
        Gram matrix.
    y : np.ndarray
        Target values.
    C : float
        Regularization parameter.
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    metric : str
        Distance metric for KKT constraints.
    learning_rate : float, optional
        Learning rate for gradient descent, by default 0.01.

    Returns
    -------
    tuple
        Tuple of (result, metrics).
    """
    n_samples = K.shape[0]
    alpha = np.zeros(n_samples)

    for _ in range(max_iter):
        # Compute gradients
        grad = -y * (K @ alpha) + y

        # Update alpha
        alpha -= learning_rate * grad

        # Project alpha onto feasible region
        alpha = np.clip(alpha, 0, C)

        # Check convergence
        if np.linalg.norm(grad) < tol:
            break

    # Compute KKT constraint violations
    result = _compute_kkt_violations(K, y, alpha, C, metric)

    # Compute metrics
    metrics = {
        "mse": np.mean((y - (K @ alpha))**2),
        "mae": np.mean(np.abs(y - (K @ alpha)))
    }

    return result, metrics

def _coordinate_descent_solver(
    K: np.ndarray,
    y: np.ndarray,
    C: float,
    tol: float,
    max_iter: int,
    metric: str
) -> tuple:
    """
    Solve KKT constraints using coordinate descent.

    Parameters
    ----------
    K : np.ndarray
        Gram matrix.
    y : np.ndarray
        Target values.
    C : float
        Regularization parameter.
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    metric : str
        Distance metric for KKT constraints.

    Returns
    -------
    tuple
        Tuple of (result, metrics).
    """
    n_samples = K.shape[0]
    alpha = np.zeros(n_samples)

    for _ in range(max_iter):
        for i in range(n_samples):
            # Compute gradient for alpha_i
            grad = -y[i] * (K[i, :] @ alpha) + y[i]

            # Update alpha_i
            alpha[i] -= grad

            # Project alpha_i onto feasible region
            alpha[i] = np.clip(alpha[i], 0, C)

        # Check convergence
        if np.linalg.norm(grad) < tol:
            break

    # Compute KKT constraint violations
    result = _compute_kkt_violations(K, y, alpha, C, metric)

    # Compute metrics
    metrics = {
        "mse": np.mean((y - (K @ alpha))**2),
        "mae": np.mean(np.abs(y - (K @ alpha)))
    }

    return result, metrics

def _compute_kkt_violations(
    K: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    C: float,
    metric: str
) -> np.ndarray:
    """
    Compute KKT constraint violations.

    Parameters
    ----------
    K : np.ndarray
        Gram matrix.
    y : np.ndarray
        Target values.
    alpha : np.ndarray
        Dual coefficients.
    C : float
        Regularization parameter.
    metric : str
        Distance metric for KKT constraints.

    Returns
    -------
    np.ndarray
        Array of KKT constraint violations.
    """
    y_pred = (K @ alpha) * y
    violations = np.zeros_like(y)

    if metric == 'euclidean':
        violations = y * (1 - y_pred)
    elif metric == 'manhattan':
        violations = np.abs(y) * (1 - y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return violations

################################################################################
# marge_maximale
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

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

def compute_kernel(X: np.ndarray, kernel_func: Callable) -> np.ndarray:
    """Compute kernel matrix."""
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_func(X[i], X[j])
    return K

def solve_dual_problem(K: np.ndarray, y: np.ndarray,
                       C: float = 1.0,
                       tol: float = 1e-3,
                       max_iter: int = 1000) -> np.ndarray:
    """Solve the dual SVM problem using coordinate descent."""
    n_samples = K.shape[0]
    alpha = np.zeros(n_samples)
    for _ in range(max_iter):
        alpha_prev = alpha.copy()
        for i in range(n_samples):
            sum_alpha_y = np.sum(alpha * y)
            alpha[i] = min(C, max(0, alpha[i] + y[i] * (1 - np.dot(K[i], alpha) * y)))
        if np.linalg.norm(alpha - alpha_prev) < tol:
            break
    return alpha

def compute_margins(alpha: np.ndarray, X: np.ndarray,
                    y: np.ndarray, kernel_func: Callable) -> Dict[str, float]:
    """Compute margins and other SVM metrics."""
    sv_indices = alpha > 1e-5
    support_vectors = X[sv_indices]
    sv_alphas = alpha[sv_indices]
    sv_y = y[sv_indices]

    # Compute decision function
    def decision_function(x: np.ndarray) -> float:
        return np.sum(sv_alphas * sv_y * kernel_func(x, support_vectors))

    # Compute margins
    margins = []
    for i in range(len(X)):
        d = decision_function(X[i])
        margin = y[i] * d
        margins.append(margin)

    return {
        'max_margin': np.max(np.abs(margins)),
        'min_margin': np.min(np.abs(margins)),
        'mean_margin': np.mean(np.abs(margins))
    }

def marge_maximale_fit(X: np.ndarray,
                       y: np.ndarray,
                       kernel_func: Callable = lambda x, y: np.dot(x, y),
                       normalization: str = 'standard',
                       C: float = 1.0,
                       tol: float = 1e-3) -> Dict[str, Union[Dict, np.ndarray]]:
    """
    Compute maximum margin SVM.

    Parameters:
    -----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Target labels (-1 or 1)
    kernel_func : Callable
        Kernel function (default: linear)
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    C : float
        Regularization parameter
    tol : float
        Tolerance for stopping criterion

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([1, -1, 1])
    >>> result = marge_maximale_fit(X, y)
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_normalized = normalize_data(X, normalization)

    # Compute kernel matrix
    K = compute_kernel(X_normalized, kernel_func)

    # Solve dual problem
    alpha = solve_dual_problem(K, y, C, tol)

    # Compute margins
    margins = compute_margins(alpha, X_normalized, y, kernel_func)

    return {
        'result': {
            'alpha': alpha,
            'support_vectors': X[alpha > 1e-5]
        },
        'metrics': margins,
        'params_used': {
            'kernel_func': kernel_func.__name__ if hasattr(kernel_func, '__name__') else 'custom',
            'normalization': normalization,
            'C': C,
            'tol': tol
        },
        'warnings': []
    }

################################################################################
# support_vectors
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def support_vectors_fit(
    X: np.ndarray,
    y: np.ndarray,
    kernel: Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = 'linear',
    C: float = 1.0,
    epsilon: float = 1e-3,
    max_iter: int = 1000,
    tol: float = 1e-3,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    normalize: Optional[str] = None,
    solver: str = 'smoothed',
    custom_kernel: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Fit a Support Vector Machine model and return the support vectors.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, n_outputs).
    kernel : str or callable, optional
        Kernel function to use. Default is 'linear'.
    C : float, optional
        Regularization parameter. Default is 1.0.
    epsilon : float, optional
        Epsilon for the epsilon-insensitive tube in SVR. Default is 1e-3.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    tol : float, optional
        Tolerance for stopping criteria. Default is 1e-3.
    metric : str or callable, optional
        Distance metric to use. Default is 'euclidean'.
    normalize : str or None, optional
        Normalization method to use. Default is None.
    solver : str, optional
        Solver to use. Default is 'smoothed'.
    custom_kernel : callable or None, optional
        Custom kernel function. Default is None.
    custom_metric : callable or None, optional
        Custom metric function. Default is None.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - 'result': np.ndarray of shape (n_support_vectors, n_features)
            The support vectors.
        - 'metrics': dict
            Dictionary of computed metrics.
        - 'params_used': dict
            Dictionary of parameters used during fitting.
        - 'warnings': list
            List of warning messages.

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([1, -1, 1])
    >>> result = support_vectors_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _normalize_data(X, normalize) if normalize else X

    # Prepare kernel and metric functions
    kernel_func = _get_kernel_function(kernel, custom_kernel)
    metric_func = _get_metric_function(metric, custom_metric)

    # Fit the SVM model and get support vectors
    support_vectors = _fit_svm(
        X_normalized, y, kernel_func, C, epsilon, max_iter, tol, solver
    )

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, support_vectors, metric_func)

    # Prepare output
    result = {
        'result': support_vectors,
        'metrics': metrics,
        'params_used': {
            'kernel': kernel if isinstance(kernel, str) else 'custom',
            'C': C,
            'epsilon': epsilon,
            'max_iter': max_iter,
            'tol': tol,
            'metric': metric if isinstance(metric, str) else 'custom',
            'normalize': normalize,
            'solver': solver
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays.")
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

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize the input data."""
    if method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _get_kernel_function(kernel: Union[str, Callable], custom_kernel: Optional[Callable]) -> Callable:
    """Get the kernel function."""
    if custom_kernel is not None:
        return custom_kernel
    elif kernel == 'linear':
        return lambda x, y: np.dot(x, y.T)
    elif kernel == 'rbf':
        return lambda x, y: np.exp(-np.linalg.norm(x[:, np.newaxis] - y, axis=2) ** 2)
    elif kernel == 'poly':
        return lambda x, y: (np.dot(x, y.T) + 1) ** 3
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

def _get_metric_function(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Get the metric function."""
    if custom_metric is not None:
        return custom_metric
    elif metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric == 'minkowski':
        return lambda x, y: np.sum(np.abs(x - y) ** 3) ** (1/3)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _fit_svm(
    X: np.ndarray,
    y: np.ndarray,
    kernel_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    C: float,
    epsilon: float,
    max_iter: int,
    tol: float,
    solver: str
) -> np.ndarray:
    """Fit the SVM model and return support vectors."""
    # This is a placeholder for the actual SVM fitting logic
    # In a real implementation, this would involve solving the dual optimization problem
    # and extracting the support vectors based on the Lagrange multipliers

    # For demonstration purposes, we'll return a subset of the data as "support vectors"
    n_samples = X.shape[0]
    support_indices = np.random.choice(n_samples, size=min(3, n_samples), replace=False)
    return X[support_indices]

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    support_vectors: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Compute metrics for the support vectors."""
    # This is a placeholder for the actual metric computation
    # In a real implementation, this would involve computing relevant metrics like accuracy, precision, recall, etc.

    # For demonstration purposes, we'll compute a simple metric
    distances = np.array([metric_func(x, y) for x in support_vectors for y in X])
    mean_distance = np.mean(distances)
    return {'mean_distance': mean_distance}

################################################################################
# regularisation_c
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regularisation_c_fit(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    kernel: str = 'linear',
    normalize: Optional[str] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'closed_form',
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_kernel: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, any], str]]:
    """
    Fit a SVM model with regularization parameter C.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, n_outputs).
    C : float, optional
        Regularization parameter. Default is 1.0.
    kernel : str, optional
        Kernel type ('linear', 'poly', 'rbf', 'sigmoid'). Default is 'linear'.
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'). Default is None.
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2'). Default is 'mse'.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent'). Default is 'closed_form'.
    tol : float, optional
        Tolerance for stopping criteria. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_kernel : callable, optional
        Custom kernel function. Default is None.
    custom_metric : callable, optional
        Custom metric function. Default is None.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, any], str]]
        Dictionary containing the result, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> y = np.array([0, 1])
    >>> result = regularisation_c_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _normalize_data(X, normalize) if normalize else X

    # Choose kernel
    kernel_func = _get_kernel(kernel, custom_kernel)

    # Compute kernel matrix
    K = kernel_func(X_normalized)

    # Choose solver
    if solver == 'closed_form':
        w, b = _solve_closed_form(K, y, C)
    elif solver == 'gradient_descent':
        w, b = _solve_gradient_descent(K, y, C, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, w, b, metric, custom_metric)

    # Prepare output
    result = {
        'result': {'weights': w, 'bias': b},
        'metrics': metrics,
        'params_used': {
            'C': C,
            'kernel': kernel if custom_kernel is None else 'custom',
            'normalize': normalize,
            'metric': metric if custom_metric is None else 'custom',
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or Inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or Inf values")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data based on the specified method."""
    if method == 'standard':
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        X_normalized = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return X_normalized

def _get_kernel(kernel: str, custom_kernel: Optional[Callable]) -> Callable:
    """Get the kernel function."""
    if custom_kernel is not None:
        return custom_kernel
    if kernel == 'linear':
        return lambda X: np.dot(X, X.T)
    elif kernel == 'poly':
        return lambda X: (np.dot(X, X.T) + 1) ** 2
    elif kernel == 'rbf':
        return lambda X: np.exp(-np.sum((X[:, np.newaxis] - X) ** 2, axis=2) / (2 * 1**2))
    elif kernel == 'sigmoid':
        return lambda X: np.tanh(np.dot(X, X.T))
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

def _solve_closed_form(K: np.ndarray, y: np.ndarray, C: float) -> tuple:
    """Solve the SVM problem using closed-form solution."""
    n_samples = K.shape[0]
    H = np.outer(y, y) * K
    P = cvxopt.matrix(H)
    q = cvxopt.matrix(-np.ones(n_samples))
    G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
    h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C)))
    A = cvxopt.matrix(y, (1, n_samples), 'd')
    b = cvxopt.matrix(0.0)
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.array(solution['x']).flatten()
    w = np.sum(alphas * y[:, np.newaxis] * X, axis=0)
    b = y[alphas > 1e-5][0] - np.dot(w, X[alphas > 1e-5][0])
    return w, b

def _solve_gradient_descent(K: np.ndarray, y: np.ndarray, C: float, tol: float, max_iter: int) -> tuple:
    """Solve the SVM problem using gradient descent."""
    n_samples = K.shape[0]
    w = np.zeros(n_samples)
    b = 0.0
    for _ in range(max_iter):
        grad_w = -y + C * np.maximum(0, 1 - y * (np.dot(K, w) + b))
        grad_b = C * np.sum(np.maximum(0, 1 - y * (np.dot(K, w) + b)))
        w -= tol * grad_w
        b -= tol * grad_b
    return w, b

def _compute_metrics(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float,
                     metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Dict[str, float]:
    """Compute metrics based on the specified method."""
    y_pred = np.dot(X, w) + b
    if custom_metric is not None:
        return {'custom': custom_metric(y, y_pred)}
    if metric == 'mse':
        return {'mse': np.mean((y - y_pred) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(y - y_pred))}
    elif metric == 'r2':
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return {'r2': 1 - ss_res / ss_tot}
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# decision_function
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input arrays for SVM decision function.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, n_outputs)

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("Inputs must be numpy arrays")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if y.ndim not in (1, 2):
        raise ValueError("y must be a 1D or 2D array")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")

    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize input data using specified method.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features)
    method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')

    Returns
    ------
    np.ndarray
        Normalized feature matrix
    """
    if method == 'none':
        return X

    X_norm = X.copy()
    n_samples, n_features = X.shape

    if method == 'standard':
        mean = np.mean(X_norm, axis=0)
        std = np.std(X_norm, axis=0)
        std[std == 0] = 1.0
        X_norm = (X_norm - mean) / std

    elif method == 'minmax':
        min_val = np.min(X_norm, axis=0)
        max_val = np.max(X_norm, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        X_norm = (X_norm - min_val) / range_val

    elif method == 'robust':
        median = np.median(X_norm, axis=0)
        iqr = np.subtract(*np.percentile(X_norm, [75, 25], axis=0))
        iqr[iqr == 0] = 1.0
        X_norm = (X_norm - median) / iqr

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_norm

def compute_kernel(X: np.ndarray, kernel: str = 'linear', gamma: float = 1.0,
                   degree: int = 3, coef0: float = 0.0) -> np.ndarray:
    """
    Compute kernel matrix for SVM.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features)
    kernel : str
        Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
    gamma : float
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
    degree : int
        Degree of polynomial kernel
    coef0 : float
        Independent term in kernel function

    Returns
    ------
    np.ndarray
        Kernel matrix of shape (n_samples, n_samples)
    """
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    if kernel == 'linear':
        K = X @ X.T

    elif kernel == 'poly':
        K = (gamma * X @ X.T + coef0) ** degree

    elif kernel == 'rbf':
        pairwise_sq_dists = np.sum(X**2, axis=1)[:, np.newaxis] + \
                           np.sum(X**2, axis=1)[np.newaxis, :] - \
                           2 * X @ X.T
        K = np.exp(-gamma * pairwise_sq_dists)

    elif kernel == 'sigmoid':
        K = np.tanh(gamma * X @ X.T + coef0)

    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    return K

def solve_svm_dual(K: np.ndarray, y: np.ndarray, C: float = 1.0,
                   solver: str = 'liblinear') -> np.ndarray:
    """
    Solve the dual SVM problem.

    Parameters
    ----------
    K : np.ndarray
        Kernel matrix of shape (n_samples, n_samples)
    y : np.ndarray
        Target values of shape (n_samples,)
    C : float
        Regularization parameter
    solver : str
        Solver type ('liblinear', 'lbfgs')

    Returns
    ------
    np.ndarray
        Dual coefficients alpha of shape (n_samples,)
    """
    n_samples = K.shape[0]

    if solver == 'liblinear':
        # Simplified implementation for demonstration
        from sklearn.svm import LibLinear
        model = LibLinear(C=C)
        model.fit(K, y)
        alpha = np.zeros(n_samples)
        for i in range(n_samples):
            if model.dual_coef_[0, 0] != 0:
                alpha[i] = model.dual_coef_[0, 0]
        return alpha

    elif solver == 'lbfgs':
        # Simplified implementation for demonstration
        from scipy.optimize import minimize

        def objective(alpha):
            return 0.5 * alpha.T @ K @ alpha - np.sum(alpha)

        def jacobian(alpha):
            return K @ alpha - y

        constraints = [{'type': 'eq', 'fun': lambda a: np.dot(a, y)},
                      {'type': 'ineq', 'fun': lambda a: C - a}]
        bounds = [(0, C) for _ in range(n_samples)]

        result = minimize(objective, np.zeros(n_samples),
                          jac=jacobian,
                          bounds=bounds,
                          constraints=constraints)
        return result.x

    else:
        raise ValueError(f"Unknown solver: {solver}")

def compute_decision_function(X: np.ndarray, X_train: np.ndarray,
                             y_train: np.ndarray, alpha: np.ndarray,
                             kernel: str = 'linear', gamma: float = 1.0,
                             degree: int = 3, coef0: float = 0.0) -> np.ndarray:
    """
    Compute decision function values for new samples.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features)
    X_train : np.ndarray
        Training feature matrix of shape (n_train_samples, n_features)
    y_train : np.ndarray
        Training target values of shape (n_train_samples,)
    alpha : np.ndarray
        Dual coefficients from SVM solution
    kernel : str
        Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
    gamma : float
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
    degree : int
        Degree of polynomial kernel
    coef0 : float
        Independent term in kernel function

    Returns
    ------
    np.ndarray
        Decision function values of shape (n_samples,)
    """
    n_samples = X.shape[0]
    decision_values = np.zeros(n_samples)

    for i in range(n_samples):
        k = compute_kernel(X[i:i+1], X_train, kernel, gamma, degree, coef0)
        decision_values[i] = np.sum(alpha * y_train * k.flatten())

    return decision_values

def decision_function_fit(X: np.ndarray, y: np.ndarray,
                         kernel: str = 'linear', gamma: float = 1.0,
                         degree: int = 3, coef0: float = 0.0,
                         C: float = 1.0, solver: str = 'liblinear',
                         normalize_method: str = 'standard') -> Dict:
    """
    Fit SVM model and compute decision function.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, n_outputs)
    kernel : str
        Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
    gamma : float
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
    degree : int
        Degree of polynomial kernel
    coef0 : float
        Independent term in kernel function
    C : float
        Regularization parameter
    solver : str
        Solver type ('liblinear', 'lbfgs')
    normalize_method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')

    Returns
    ------
    Dict
        Dictionary containing:
        - result: Decision function values
        - metrics: Computed metrics
        - params_used: Parameters used in the computation
        - warnings: Any warnings generated
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm = normalize_data(X.copy(), method=normalize_method)
    y_norm = y.copy()

    # Compute kernel matrix
    K_train = compute_kernel(X_norm, kernel, gamma, degree, coef0)

    # Solve SVM dual problem
    alpha = solve_svm_dual(K_train, y_norm, C, solver)

    # Compute decision function
    decision_values = compute_decision_function(X_norm, X_norm,
                                               y_norm, alpha,
                                               kernel, gamma, degree, coef0)

    # Prepare output
    result = {
        'result': decision_values,
        'metrics': {},
        'params_used': {
            'kernel': kernel,
            'gamma': gamma,
            'degree': degree,
            'coef0': coef0,
            'C': C,
            'solver': solver,
            'normalize_method': normalize_method
        },
        'warnings': []
    }

    return result

def decision_function_compute(X: np.ndarray, X_train: np.ndarray,
                            y_train: np.ndarray, alpha: np.ndarray,
                            kernel: str = 'linear', gamma: float = 1.0,
                            degree: int = 3, coef0: float = 0.0,
                            normalize_method: str = 'standard') -> Dict:
    """
    Compute decision function for new samples using pre-trained model.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features)
    X_train : np.ndarray
        Training feature matrix of shape (n_train_samples, n_features)
    y_train : np.ndarray
        Training target values of shape (n_train_samples,)
    alpha : np.ndarray
        Dual coefficients from SVM solution
    kernel : str
        Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
    gamma : float
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
    degree : int
        Degree of polynomial kernel
    coef0 : float
        Independent term in kernel function
    normalize_method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')

    Returns
    ------
    Dict
        Dictionary containing:
        - result: Decision function values
        - metrics: Computed metrics
        - params_used: Parameters used in the computation
        - warnings: Any warnings generated
    """
    # Validate inputs
    validate_inputs(X, y_train)

    # Normalize data (using same method as training)
    X_norm = normalize_data(X.copy(), method=normalize_method)

    # Compute decision function
    decision_values = compute_decision_function(X_norm, X_train,
                                               y_train, alpha,
                                               kernel, gamma, degree, coef0)

    # Prepare output
    result = {
        'result': decision_values,
        'metrics': {},
        'params_used': {
            'kernel': kernel,
            'gamma': gamma,
            'degree': degree,
            'coef0': coef0,
            'normalize_method': normalize_method
        },
        'warnings': []
    }

    return result

################################################################################
# classification_binaire
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def classification_binaire_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalisation: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Fit a binary classification SVM model.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target labels of shape (n_samples,) with values in {-1, 1}.
    normalisation : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    distance_metric : str or callable
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'minkowski', or custom callable.
    solver : str
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : str, optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function for evaluation.

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
    X_normalized = _apply_normalization(X, normalisation)

    # Choose distance metric
    if isinstance(distance_metric, str):
        distance_func = _get_distance_function(distance_metric)
    else:
        distance_func = distance_metric

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(X_normalized, y, tol, max_iter)
    elif solver == 'newton':
        params = _solve_newton(X_normalized, y, tol, max_iter)
    elif solver == 'coordinate_descent':
        params = _solve_coordinate_descent(X_normalized, y, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization:
        params = _apply_regularization(params, X_normalized, y, regularization)

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, params, custom_metric)

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
    """Validate input data dimensions and types."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError("y must contain only -1 and 1")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the data."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _get_distance_function(metric: str) -> Callable:
    """Get distance function based on metric name."""
    if metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric == 'minkowski':
        return lambda x, y: np.sum(np.abs(x - y)**3)**(1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve SVM using closed-form solution."""
    # This is a placeholder for the actual implementation
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve SVM using gradient descent."""
    # This is a placeholder for the actual implementation
    w = np.zeros(X.shape[1])
    for _ in range(max_iter):
        gradient = -2 * X.T @ (y * (1 / (1 + np.exp(y * X @ w))))
        w -= 0.01 * gradient
    return w

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve SVM using Newton's method."""
    # This is a placeholder for the actual implementation
    w = np.zeros(X.shape[1])
    for _ in range(max_iter):
        gradient = -2 * X.T @ (y * (1 / (1 + np.exp(y * X @ w))))
        hessian = 2 * X.T @ (np.diagflat(1 / (1 + np.exp(y * X @ w))**2) @ X)
        w -= np.linalg.pinv(hessian) @ gradient
    return w

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve SVM using coordinate descent."""
    # This is a placeholder for the actual implementation
    w = np.zeros(X.shape[1])
    for _ in range(max_iter):
        for i in range(X.shape[1]):
            X_i = X[:, i]
            w[i] -= (2 * np.sum((y - np.exp(y * X @ w)) * y * X_i) /
                     (2 * np.sum((y - np.exp(y * X @ w))**2 * X_i**2)))
    return w

def _apply_regularization(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply regularization to the parameters."""
    if method == 'l1':
        return np.sign(params) * np.maximum(np.abs(params) - 0.1, 0)
    elif method == 'l2':
        return params / (1 + 0.1 * np.linalg.norm(params))
    elif method == 'elasticnet':
        return (np.sign(params) * np.maximum(np.abs(params) - 0.1, 0) +
                params / (1 + 0.1 * np.linalg.norm(params))) / 2
    else:
        raise ValueError(f"Unknown regularization method: {method}")

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute metrics for the model."""
    predictions = np.sign(X @ params)
    accuracy = np.mean(predictions == y)

    metrics = {
        'accuracy': accuracy,
        'precision': np.sum((predictions == 1) & (y == 1)) / np.sum(predictions == 1),
        'recall': np.sum((predictions == 1) & (y == 1)) / np.sum(y == 1),
        'f1_score': 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    }

    if custom_metric:
        metrics['custom'] = custom_metric(y, predictions)

    return metrics

################################################################################
# classification_multiclasse
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def classification_multiclasse_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Fit a multi-class SVM classifier with configurable parameters.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target labels of shape (n_samples,).
    normalizer : Optional[Callable], optional
        Function to normalize the input features. Default is None.
    distance_metric : str, optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski'). Default is 'euclidean'.
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent'). Default is 'closed_form'.
    regularization : Optional[str], optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet'). Default is None.
    tol : float, optional
        Tolerance for stopping criteria. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric : Optional[Callable], optional
        Custom metric function. Default is None.

    Returns
    -------
    Dict[str, Union[Dict, float, str]]
        Dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 3, size=100)
    >>> result = classification_multiclasse_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if a normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Get the number of classes
    n_classes = len(np.unique(y))

    # Initialize the result dictionary
    result_dict: Dict[str, Union[Dict, float, str]] = {
        "result": {},
        "metrics": {},
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "distance_metric": distance_metric,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    # Train one-vs-one classifiers for each pair of classes
    classifiers = {}
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            X_ij = X_normalized[(y == i) | (y == j)]
            y_ij = y[(y == i) | (y == j)]

            # Train binary SVM for classes i and j
            classifier = _train_binary_svm(
                X_ij, y_ij,
                distance_metric=distance_metric,
                solver=solver,
                regularization=regularization,
                tol=tol,
                max_iter=max_iter
            )
            classifiers[(i, j)] = classifier

    result_dict["result"]["classifiers"] = classifiers

    # Calculate metrics
    y_pred = _predict_multiclass(X_normalized, classifiers)
    result_dict["metrics"] = _calculate_metrics(y, y_pred, custom_metric)

    return result_dict

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
    if normalizer is not None:
        return normalizer(X)
    return X

def _train_binary_svm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict[str, np.ndarray]:
    """Train a binary SVM classifier."""
    # Implement the binary SVM training logic here
    # This is a placeholder for the actual implementation
    return {
        "weights": np.random.rand(X.shape[1]),
        "bias": 0.0
    }

def _predict_multiclass(X: np.ndarray, classifiers: Dict[tuple, Dict[str, np.ndarray]]) -> np.ndarray:
    """Predict multi-class labels using one-vs-one classifiers."""
    n_samples = X.shape[0]
    y_pred = np.zeros(n_samples, dtype=int)

    for i in range(len(classifiers)):
        for j in range(i + 1, len(classifiers)):
            classifier = classifiers[(i, j)]
            # Implement the prediction logic here
            pass

    return y_pred

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Calculate the metrics for the classification."""
    metrics = {}

    if custom_metric is not None:
        metrics["custom"] = custom_metric(y_true, y_pred)

    # Add other default metrics here
    return metrics

################################################################################
# regression_svm
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regression_svm_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda x, y: np.dot(x, y),
    C: float = 1.0,
    epsilon: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-3,
    metric: str = 'mse',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    solver: str = 'smoothed',
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a Support Vector Regression (SVR) model.

    Parameters
    ----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    kernel : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Kernel function. Default is linear kernel.
    C : float
        Regularization parameter. Must be positive.
    epsilon : float
        Epsilon-tube within which no penalty is given to errors.
    max_iter : int
        Maximum number of iterations for the solver.
    tol : float
        Tolerance for stopping criteria.
    metric : str or Callable[[np.ndarray, np.ndarray], float]
        Metric to evaluate the model. Can be 'mse', 'mae', 'r2', or a custom callable.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]]
        Function to normalize the input data. If None, no normalization is applied.
    solver : str
        Solver to use. Can be 'smoothed' or 'dual'.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing the fitted model, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.array([[1], [2], [3]])
    >>> y = np.array([1, 2, 3])
    >>> result = regression_svm_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if a normalizer is provided
    if normalizer is not None:
        X = normalizer(X)

    # Initialize parameters
    n_samples, n_features = X.shape

    # Choose solver
    if solver == 'smoothed':
        alpha, b = _smo_solver(X, y, kernel, C, epsilon, max_iter, tol)
    elif solver == 'dual':
        alpha, b = _dual_solver(X, y, kernel, C, epsilon, max_iter, tol)
    else:
        raise ValueError("Solver must be 'smoothed' or 'dual'")

    # Compute predictions
    y_pred = _predict(X, alpha, b, kernel)

    # Compute metrics
    if isinstance(metric, str):
        if metric == 'mse':
            mse = _mean_squared_error(y, y_pred)
            metrics = {'mse': mse}
        elif metric == 'mae':
            mae = _mean_absolute_error(y, y_pred)
            metrics = {'mae': mae}
        elif metric == 'r2':
            r2 = _r_squared(y, y_pred)
            metrics = {'r2': r2}
        else:
            raise ValueError("Metric must be 'mse', 'mae', or 'r2'")
    else:
        custom_metric = metric(y, y_pred)
        metrics = {'custom': custom_metric}

    # Prepare output
    result = {
        'result': {
            'alpha': alpha,
            'b': b
        },
        'metrics': metrics,
        'params_used': {
            'kernel': kernel.__name__ if hasattr(kernel, '__name__') else 'custom',
            'C': C,
            'epsilon': epsilon,
            'max_iter': max_iter,
            'tol': tol,
            'metric': metric if isinstance(metric, str) else 'custom',
            'normalizer': normalizer.__name__ if normalizer is not None and hasattr(normalizer, '__name__') else 'none',
            'solver': solver
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X must not contain NaN or Inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y must not contain NaN or Inf values")

def _smo_solver(
    X: np.ndarray,
    y: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    C: float,
    epsilon: float,
    max_iter: int,
    tol: float
) -> tuple[np.ndarray, float]:
    """Smoothed Support Vector Regression solver."""
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)
    b = 0.0

    for _ in range(max_iter):
        alpha_prev = np.copy(alpha)

        # Update alpha
        for i in range(n_samples):
            Ei = _compute_error(X, y, alpha, b, kernel, i)

            if (y[i] - Ei < -epsilon and alpha[i] < C) or (y[i] - Ei > epsilon and alpha[i] > 0):
                j = np.random.choice([idx for idx in range(n_samples) if idx != i])
                Ej = _compute_error(X, y, alpha, b, kernel, j)

                alpha_i_old, alpha_j_old = alpha[i], alpha[j]

                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])

                if L == H:
                    continue

                eta = 2 * kernel(X[i], X[j]) - kernel(X[i], X[i]) - kernel(X[j], X[j])

                if eta >= 0:
                    continue

                alpha[j] -= y[j] * (Ei - Ej) / eta
                alpha[j] = np.clip(alpha[j], L, H)

                if abs(alpha[j] - alpha_j_old) < tol:
                    continue

                alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])

                # Update bias
                b1 = b - Ei - y[i] * (alpha[i] - alpha_i_old) * kernel(X[i], X[i]) - y[j] * (alpha[j] - alpha_j_old) * kernel(X[i], X[j])
                b2 = b - Ej - y[i] * (alpha[i] - alpha_i_old) * kernel(X[i], X[j]) - y[j] * (alpha[j] - alpha_j_old) * kernel(X[j], X[j])

                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

        if np.linalg.norm(alpha - alpha_prev) < tol:
            break

    return alpha, b

def _dual_solver(
    X: np.ndarray,
    y: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    C: float,
    epsilon: float,
    max_iter: int,
    tol: float
) -> tuple[np.ndarray, float]:
    """Dual Support Vector Regression solver."""
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)
    b = 0.0

    for _ in range(max_iter):
        alpha_prev = np.copy(alpha)

        # Update alpha
        for i in range(n_samples):
            Ei = _compute_error(X, y, alpha, b, kernel, i)

            if (y[i] - Ei < -epsilon and alpha[i] < C) or (y[i] - Ei > epsilon and alpha[i] > 0):
                j = np.random.choice([idx for idx in range(n_samples) if idx != i])
                Ej = _compute_error(X, y, alpha, b, kernel, j)

                alpha_i_old, alpha_j_old = alpha[i], alpha[j]

                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])

                if L == H:
                    continue

                eta = 2 * kernel(X[i], X[j]) - kernel(X[i], X[i]) - kernel(X[j], X[j])

                if eta >= 0:
                    continue

                alpha[j] -= y[j] * (Ei - Ej) / eta
                alpha[j] = np.clip(alpha[j], L, H)

                if abs(alpha[j] - alpha_j_old) < tol:
                    continue

                alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])

                # Update bias
                b1 = b - Ei - y[i] * (alpha[i] - alpha_i_old) * kernel(X[i], X[i]) - y[j] * (alpha[j] - alpha_j_old) * kernel(X[i], X[j])
                b2 = b - Ej - y[i] * (alpha[i] - alpha_i_old) * kernel(X[i], X[j]) - y[j] * (alpha[j] - alpha_j_old) * kernel(X[j], X[j])

                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

        if np.linalg.norm(alpha - alpha_prev) < tol:
            break

    return alpha, b

def _compute_error(
    X: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    b: float,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    i: int
) -> float:
    """Compute the error for sample i."""
    y_pred = _predict_single(X, alpha, b, kernel, i)
    return y_pred - y[i]

def _predict(
    X: np.ndarray,
    alpha: np.ndarray,
    b: float,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> np.ndarray:
    """Predict using the fitted model."""
    y_pred = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        y_pred[i] = _predict_single(X, alpha, b, kernel, i)
    return y_pred

def _predict_single(
    X: np.ndarray,
    alpha: np.ndarray,
    b: float,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    i: int
) -> float:
    """Predict a single sample."""
    return np.sum(alpha * kernel(X[i], X)) + b

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

################################################################################
# scaling_des_donnees
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.size == 0:
        raise ValueError("Input array must not be empty")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input array must not contain NaN or Inf values")

def _standard_scaling(X: np.ndarray) -> np.ndarray:
    """Standard scaling (z-score normalization)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + 1e-8)

def _minmax_scaling(X: np.ndarray) -> np.ndarray:
    """Min-max scaling."""
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    return (X - min_val) / ((max_val - min_val + 1e-8))

def _robust_scaling(X: np.ndarray) -> np.ndarray:
    """Robust scaling using median and IQR."""
    median = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    iqr = q75 - q25
    return (X - median) / (iqr + 1e-8)

def _custom_scaling(X: np.ndarray, scaling_func: Callable) -> np.ndarray:
    """Apply custom scaling function."""
    return scaling_func(X)

def _compute_metrics(
    X_scaled: np.ndarray,
    metric_funcs: Dict[str, Callable],
    params_used: Dict
) -> Dict:
    """Compute metrics for scaled data."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(X_scaled)
        except Exception as e:
            metrics[f"{name}_error"] = str(e)
    return metrics

def scaling_des_donnees_fit(
    X: np.ndarray,
    scaling_method: str = "standard",
    custom_scaling_func: Optional[Callable] = None,
    metric_funcs: Dict[str, Callable] = None
) -> Dict:
    """
    Scale data using specified method and compute metrics.

    Parameters:
    - X: Input data array
    - scaling_method: "none", "standard", "minmax", or "robust"
    - custom_scaling_func: Optional callable for custom scaling
    - metric_funcs: Dictionary of metric functions to compute

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    _validate_input(X)

    # Initialize default metric functions if none provided
    if metric_funcs is None:
        metric_funcs = {
            "mean": lambda x: np.mean(x),
            "std": lambda x: np.std(x)
        }

    # Apply scaling
    if scaling_method == "none":
        X_scaled = X.copy()
    elif scaling_method == "standard":
        X_scaled = _standard_scaling(X)
    elif scaling_method == "minmax":
        X_scaled = _minmax_scaling(X)
    elif scaling_method == "robust":
        X_scaled = _robust_scaling(X)
    elif custom_scaling_func is not None:
        X_scaled = _custom_scaling(X, custom_scaling_func)
    else:
        raise ValueError(f"Unknown scaling method: {scaling_method}")

    # Compute metrics
    metrics = _compute_metrics(X_scaled, metric_funcs, {
        "scaling_method": scaling_method,
        "custom_scaling_used": custom_scaling_func is not None
    })

    return {
        "result": X_scaled,
        "metrics": metrics,
        "params_used": {
            "scaling_method": scaling_method,
            "custom_scaling_func": custom_scaling_func.__name__ if custom_scaling_func else None
        },
        "warnings": []
    }

# Example usage:
"""
X = np.random.rand(10, 5)
result = scaling_des_donnees_fit(
    X,
    scaling_method="standard",
    metric_funcs={
        "mean": lambda x: np.mean(x),
        "std": lambda x: np.std(x)
    }
)
"""

################################################################################
# selection_du_kernel
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def selection_du_kernel_fit(
    X: np.ndarray,
    y: np.ndarray,
    kernel_functions: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    normalization: str = 'standard',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_weights: Optional[np.ndarray] = None
) -> Dict[str, Union[Dict, np.ndarray]]:
    """
    Main function to fit a kernel selection model.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,) or (n_samples, n_outputs).
    kernel_functions : Dict[str, Callable]
        Dictionary of available kernel functions.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), by default 'standard'.
    metric : Union[str, Callable], optional
        Metric to evaluate performance ('mse', 'mae', 'r2', 'logloss'), by default 'mse'.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent'), by default 'closed_form'.
    regularization : Optional[str], optional
        Regularization method (None, 'l1', 'l2', 'elasticnet'), by default None.
    tol : float, optional
        Tolerance for stopping criteria, by default 1e-4.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    custom_weights : Optional[np.ndarray], optional
        Custom weights for samples, by default None.

    Returns
    -------
    Dict[str, Union[Dict, np.ndarray]]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalization(X, normalization)

    # Initialize results dictionary
    results = {
        'result': {},
        'metrics': {},
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    # Select kernel and fit model
    for kernel_name, kernel_func in kernel_functions.items():
        try:
            # Compute kernel matrix
            K = _compute_kernel_matrix(X_normalized, kernel_func)

            # Fit model based on solver
            if solver == 'closed_form':
                params = _fit_closed_form(K, y)
            elif solver == 'gradient_descent':
                params = _fit_gradient_descent(K, y, tol=tol, max_iter=max_iter)
            elif solver == 'newton':
                params = _fit_newton(K, y, tol=tol, max_iter=max_iter)
            elif solver == 'coordinate_descent':
                params = _fit_coordinate_descent(K, y, tol=tol, max_iter=max_iter)
            else:
                raise ValueError(f"Unknown solver: {solver}")

            # Apply regularization if specified
            if regularization:
                params = _apply_regularization(params, regularization)

            # Compute metrics
            metrics = _compute_metrics(y, params, metric)

            # Store results for this kernel
            results['result'][kernel_name] = params
            results['metrics'][kernel_name] = metrics

        except Exception as e:
            results['warnings'].append(f"Kernel {kernel_name} failed: {str(e)}")

    return results

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1 and y.ndim != 2:
        raise ValueError("y must be a 1D or 2D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to input data."""
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

def _compute_kernel_matrix(X: np.ndarray, kernel_func: Callable) -> np.ndarray:
    """Compute the kernel matrix using the specified kernel function."""
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_func(X[i], X[j])
    return K

def _fit_closed_form(K: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit model using closed-form solution."""
    return np.linalg.solve(K + 1e-8 * np.eye(K.shape[0]), y)

def _fit_gradient_descent(
    K: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Fit model using gradient descent."""
    n_samples = K.shape[0]
    params = np.zeros(n_samples)
    learning_rate = 1.0 / n_samples
    for _ in range(max_iter):
        gradient = np.dot(K, params) - y
        params -= learning_rate * gradient
        if np.linalg.norm(gradient) < tol:
            break
    return params

def _fit_newton(
    K: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Fit model using Newton's method."""
    n_samples = K.shape[0]
    params = np.zeros(n_samples)
    for _ in range(max_iter):
        gradient = np.dot(K, params) - y
        hessian = K + 1e-8 * np.eye(n_samples)
        params -= np.linalg.solve(hessian, gradient)
        if np.linalg.norm(gradient) < tol:
            break
    return params

def _fit_coordinate_descent(
    K: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Fit model using coordinate descent."""
    n_samples = K.shape[0]
    params = np.zeros(n_samples)
    for _ in range(max_iter):
        for i in range(n_samples):
            residual = y - np.dot(K, params)
            update = residual[i] / (K[i, i] + 1e-8)
            params[i] += update
        if np.linalg.norm(residual) < tol:
            break
    return params

def _apply_regularization(params: np.ndarray, method: str) -> np.ndarray:
    """Apply regularization to parameters."""
    if method == 'l1':
        return np.sign(params) * np.maximum(np.abs(params) - 1e-4, 0)
    elif method == 'l2':
        return params / (1 + 1e-4)
    elif method == 'elasticnet':
        return np.sign(params) * np.maximum(np.abs(params) - 1e-4, 0) / (1 + 1e-4)
    else:
        raise ValueError(f"Unknown regularization method: {method}")

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Compute specified metrics."""
    metrics = {}
    if callable(metric):
        metrics['custom'] = metric(y_true, y_pred)
    else:
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - ss_res / (ss_tot + 1e-8)
        elif metric == 'logloss':
            y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
            metrics['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return metrics

################################################################################
# overfitting
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

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
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

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

def fit_svm(X: np.ndarray, y: np.ndarray,
           kernel: str = 'linear',
           C: float = 1.0,
           epsilon: float = 1e-3,
           max_iter: int = 1000) -> Dict:
    """Fit SVM model with specified parameters."""
    # Placeholder for actual SVM implementation
    # In a real implementation, this would use an appropriate solver
    weights = np.random.randn(X.shape[1])
    bias = 0.0

    for _ in range(max_iter):
        # Update weights and bias (simplified)
        pass

    return {'weights': weights, 'bias': bias}

def overfitting_fit(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   normalization: str = 'standard',
                   metric: Union[str, Callable] = 'mse',
                   kernel: str = 'linear',
                   C: float = 1.0,
                   epsilon: float = 1e-3,
                   max_iter: int = 1000) -> Dict:
    """
    Detect overfitting by comparing training and validation performance.

    Parameters:
    -----------
    X_train : np.ndarray
        Training data features
    y_train : np.ndarray
        Training data targets
    X_val : np.ndarray
        Validation data features
    y_val : np.ndarray
        Validation data targets
    normalization : str, optional
        Normalization method (default: 'standard')
    metric : str or callable, optional
        Metric to evaluate performance (default: 'mse')
    kernel : str, optional
        SVM kernel type (default: 'linear')
    C : float, optional
        Regularization parameter (default: 1.0)
    epsilon : float, optional
        Tolerance for stopping criterion (default: 1e-3)
    max_iter : int, optional
        Maximum number of iterations (default: 1000)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    validate_inputs(X_train, y_train)
    validate_inputs(X_val, y_val)

    # Normalize data
    X_train_norm = normalize_data(X_train, normalization)
    X_val_norm = normalize_data(X_val, normalization)

    # Fit SVM model
    train_model = fit_svm(X_train_norm, y_train,
                         kernel=kernel, C=C,
                         epsilon=epsilon, max_iter=max_iter)

    # Make predictions
    y_train_pred = np.dot(X_train_norm, train_model['weights']) + train_model['bias']
    y_val_pred = np.dot(X_val_norm, train_model['weights']) + train_model['bias']

    # Compute metrics
    train_metric = compute_metric(y_train, y_train_pred, metric)
    val_metric = compute_metric(y_val, y_val_pred, metric)

    # Check for overfitting
    overfit = train_metric < val_metric

    return {
        'result': {
            'overfitting_detected': overfit,
            'train_metric': train_metric,
            'val_metric': val_metric
        },
        'metrics': {
            'train': train_metric,
            'validation': val_metric
        },
        'params_used': {
            'normalization': normalization,
            'metric': metric if not callable(metric) else 'custom',
            'kernel': kernel,
            'C': C,
            'epsilon': epsilon,
            'max_iter': max_iter
        },
        'warnings': []
    }

################################################################################
# underfitting
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def underfitting_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    kernel: str = 'linear',
    C: float = 1.0,
    epsilon: float = 1e-3,
    max_iter: int = 1000,
    metric: str = 'mse',
    normalization: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    solver: str = 'smo',
    tol: float = 1e-4
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Fit an SVM model with underfitting detection.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, n_outputs).
    kernel : str, optional
        Kernel type ('linear', 'poly', 'rbf', 'sigmoid').
    C : float, optional
        Regularization parameter.
    epsilon : float, optional
        Tolerance for stopping criterion.
    max_iter : int, optional
        Maximum number of iterations.
    metric : str or callable, optional
        Metric to evaluate underfitting ('mse', 'mae', 'r2').
    normalization : str, optional
        Normalization method ('standard', 'minmax', 'robust').
    custom_metric : callable, optional
        Custom metric function.
    solver : str, optional
        Solver type ('smo', 'gradient_descent').
    tol : float, optional
        Tolerance for convergence.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        Dictionary containing:
        - 'result': Fitted model parameters.
        - 'metrics': Evaluation metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> result = underfitting_fit(X, y, kernel='linear', C=1.0)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalization)

    # Initialize parameters
    params_used = {
        'kernel': kernel,
        'C': C,
        'epsilon': epsilon,
        'max_iter': max_iter,
        'metric': metric,
        'normalization': normalization,
        'solver': solver,
        'tol': tol
    }

    # Choose metric function
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric_function(metric)

    # Choose solver function
    if solver == 'smo':
        solve_func = _solve_smo
    elif solver == 'gradient_descent':
        solve_func = _solve_gradient_descent
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Fit model
    result = solve_func(X_normalized, y, kernel=kernel, C=C,
                        epsilon=epsilon, max_iter=max_iter, tol=tol)

    # Evaluate metrics
    metrics = _compute_metrics(X_normalized, y, result, metric_func)

    # Check for underfitting
    warnings = _check_underfitting(metrics, metric_func)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
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

def _apply_normalization(X: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Apply normalization to input features."""
    if method is None:
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

def _get_metric_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get metric function based on input string."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def _solve_smo(
    X: np.ndarray,
    y: np.ndarray,
    *,
    kernel: str = 'linear',
    C: float = 1.0,
    epsilon: float = 1e-3,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Dict[str, np.ndarray]:
    """Solve SVM using SMO algorithm."""
    # Placeholder for actual SMO implementation
    return {
        'alpha': np.zeros(X.shape[0]),
        'bias': 0.0,
        'support_vectors': np.zeros((0, X.shape[1]))
    }

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    kernel: str = 'linear',
    C: float = 1.0,
    epsilon: float = 1e-3,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Dict[str, np.ndarray]:
    """Solve SVM using gradient descent."""
    # Placeholder for actual gradient descent implementation
    return {
        'weights': np.zeros(X.shape[1]),
        'bias': 0.0
    }

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    model: Dict[str, np.ndarray],
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    # Placeholder for actual metric computation
    y_pred = np.zeros_like(y)
    return {
        'metric': metric_func(y, y_pred),
        'accuracy': 0.5  # Placeholder
    }

def _check_underfitting(
    metrics: Dict[str, float],
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> str:
    """Check for underfitting."""
    if metrics['metric'] > 0.5:  # Placeholder threshold
        return "Warning: Potential underfitting detected"
    return ""
