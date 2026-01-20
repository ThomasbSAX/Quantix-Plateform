"""
Quantix – Module regression_polynomiale
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# degre_polynome
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def degre_polynome_fit(
    X: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    reg_param: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a polynomial regression model to the data.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    degree : int, optional
        Degree of the polynomial (default=1)
    normalize : str, optional
        Normalization method: 'none', 'standard', 'minmax', 'robust' (default='standard')
    metric : str or callable, optional
        Metric to evaluate: 'mse', 'mae', 'r2', or custom callable (default='mse')
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton' (default='closed_form')
    regularization : str, optional
        Regularization type: 'none', 'l1', 'l2', 'elasticnet' (default=None)
    reg_param : float, optional
        Regularization parameter (default=1.0)
    tol : float, optional
        Tolerance for convergence (default=1e-6)
    max_iter : int, optional
        Maximum number of iterations (default=1000)
    custom_metric : callable, optional
        Custom metric function if needed
    custom_distance : callable, optional
        Custom distance function if needed

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': fitted model coefficients
        - 'metrics': computed metrics
        - 'params_used': parameters used in fitting
        - 'warnings': any warnings encountered

    Example:
    --------
    >>> X = np.array([[1], [2], [3]])
    >>> y = np.array([1, 4, 9])
    >>> result = degre_polynome_fit(X, y, degree=2)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if needed
    X_norm, y_norm = _apply_normalization(X, y, normalize)

    # Prepare polynomial features
    X_poly = _prepare_polynomial_features(X_norm, degree)

    # Choose solver
    if solver == 'closed_form':
        coefficients = _solve_closed_form(X_poly, y_norm)
    elif solver == 'gradient_descent':
        coefficients = _solve_gradient_descent(X_poly, y_norm, reg_param, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if needed
    coefficients = _apply_regularization(coefficients, X_poly.shape[1], regularization, reg_param)

    # Compute metrics
    y_pred = _predict(X_poly, coefficients)
    metrics = _compute_metrics(y_norm, y_pred, metric, custom_metric)

    # Prepare output
    result = {
        'result': coefficients,
        'metrics': metrics,
        'params_used': {
            'degree': degree,
            'normalize': normalize,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'reg_param': reg_param
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays contain infinite values")

def _apply_normalization(X: np.ndarray, y: np.ndarray, method: str) -> tuple:
    """Apply normalization to input data."""
    X_norm = X.copy()
    y_norm = y.copy()

    if method == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / X_std

        y_mean = np.mean(y)
        y_std = np.std(y)
        y_norm = (y - y_mean) / y_std
    elif method == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min)

        y_min = np.min(y)
        y_max = np.max(y)
        y_norm = (y - y_min) / (y_max - y_min)
    elif method == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - X_median) / X_iqr

        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        y_norm = (y - y_median) / y_iqr

    return X_norm, y_norm

def _prepare_polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Prepare polynomial features."""
    n_samples = X.shape[0]
    X_poly = np.ones((n_samples, 1))

    for d in range(1, degree + 1):
        X_poly = np.hstack((X_poly, X ** d))

    return X_poly

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve polynomial regression using closed form solution."""
    XTX = np.dot(X.T, X)
    XTY = np.dot(X.T, y)

    # Add small value to diagonal for numerical stability
    XTX += 1e-8 * np.eye(XTX.shape[0])

    coefficients = np.linalg.solve(XTX, XTY)
    return coefficients

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    reg_param: float,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve polynomial regression using gradient descent."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)

    for _ in range(max_iter):
        gradients = 2 * np.dot(X.T, (np.dot(X, coefficients) - y)) / len(y)

        # Add regularization term
        if reg_param > 0:
            gradients += reg_param * coefficients

        # Update coefficients
        old_coeffs = coefficients.copy()
        coefficients -= tol * gradients

        # Check convergence
        if np.linalg.norm(coefficients - old_coeffs) < tol:
            break

    return coefficients

def _apply_regularization(
    coefficients: np.ndarray,
    n_features: int,
    method: Optional[str],
    reg_param: float
) -> np.ndarray:
    """Apply regularization to coefficients."""
    if method is None or reg_param == 0:
        return coefficients

    if method == 'l1':
        coefficients = np.sign(coefficients) * np.maximum(np.abs(coefficients) - reg_param, 0)
    elif method == 'l2':
        coefficients = coefficients / (1 + reg_param * np.arange(1, n_features + 1))
    elif method == 'elasticnet':
        coefficients = np.sign(coefficients) * np.maximum(
            np.abs(coefficients) - reg_param, 0
        ) / (1 + reg_param * np.arange(1, n_features + 1))
    else:
        raise ValueError(f"Unknown regularization method: {method}")

    return coefficients

def _predict(X: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Make predictions using the fitted model."""
    return np.dot(X, coefficients)

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute evaluation metrics."""
    metrics = {}

    if metric == 'mse' or custom_metric is None:
        mse = np.mean((y_true - y_pred) ** 2)
        metrics['mse'] = mse

    if metric == 'mae' or custom_metric is None:
        mae = np.mean(np.abs(y_true - y_pred))
        metrics['mae'] = mae

    if metric == 'r2' or custom_metric is None:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        metrics['r2'] = r2

    if custom_metric is not None:
        custom_value = custom_metric(y_true, y_pred)
        metrics['custom'] = custom_value

    return metrics

################################################################################
# coefficient_regression
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
                   normalization: str = 'standard',
                   custom_normalize: Optional[Callable] = None) -> tuple:
    """Normalize data according to specified method."""
    if custom_normalize is not None:
        X_norm, y_norm = custom_normalize(X, y)
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
        X_norm, y_norm = X, y
    return X_norm, y_norm

def _create_polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Create polynomial features up to specified degree."""
    n_samples, n_features = X.shape
    X_poly = np.ones((n_samples, 1))
    for d in range(1, degree + 1):
        X_poly = np.hstack([X_poly, np.power(X, d)])
    return X_poly

def _closed_form_solution(X: np.ndarray, y: np.ndarray,
                         regularization: str = 'none',
                         alpha: float = 1.0) -> np.ndarray:
    """Compute coefficients using closed form solution."""
    if regularization == 'l1':
        # Lasso regression (closed form approximation)
        return np.linalg.pinv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ y
    elif regularization == 'l2':
        # Ridge regression (closed form solution)
        return np.linalg.pinv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ y
    elif regularization == 'elasticnet':
        # ElasticNet regression (closed form approximation)
        return np.linalg.pinv(X.T @ X + alpha * (np.eye(X.shape[1]) +
                                               0.5 * np.eye(X.shape[1]))) @ X.T @ y
    else:
        # Ordinary least squares
        return np.linalg.pinv(X.T @ X) @ X.T @ y

def _gradient_descent(X: np.ndarray, y: np.ndarray,
                     learning_rate: float = 0.01,
                     n_iterations: int = 1000,
                     tol: float = 1e-4) -> np.ndarray:
    """Compute coefficients using gradient descent."""
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    for _ in range(n_iterations):
        gradients = 2/n_samples * X.T @ (X @ coefficients - y)
        new_coefficients = coefficients - learning_rate * gradients
        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients
    return coefficients

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metrics: Union[str, list] = 'mse',
                    custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute specified metrics."""
    result = {}
    if isinstance(metrics, str):
        metrics = [metrics]

    def mse(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)
    def mae(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))
    def r2(y_true, y_pred): return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    def logloss(y_true, y_pred): return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    metric_functions = {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'logloss': logloss
    }

    for metric in metrics:
        if metric in metric_functions:
            result[metric] = metric_functions[metric](y_true, y_pred)
        elif custom_metric is not None and metric == 'custom':
            result['custom'] = custom_metric(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return result

def coefficient_regression_fit(X: np.ndarray, y: np.ndarray,
                             degree: int = 2,
                             solver: str = 'closed_form',
                             normalization: str = 'standard',
                             metrics: Union[str, list] = 'mse',
                             regularization: str = 'none',
                             alpha: float = 1.0,
                             custom_normalize: Optional[Callable] = None,
                             custom_metric: Optional[Callable] = None) -> Dict:
    """
    Compute polynomial regression coefficients.

    Parameters:
    -----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    degree : int
        Degree of polynomial features to create
    solver : str
        Solver method ('closed_form', 'gradient_descent')
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metrics : str or list
        Metrics to compute ('mse', 'mae', 'r2', 'logloss')
    regularization : str
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    alpha : float
        Regularization strength
    custom_normalize : callable, optional
        Custom normalization function
    custom_metric : callable, optional
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.array([[1], [2], [3]])
    >>> y = np.array([1, 4, 9])
    >>> result = coefficient_regression_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization, custom_normalize)

    # Create polynomial features
    X_poly = _create_polynomial_features(X_norm, degree)

    # Solve for coefficients
    if solver == 'closed_form':
        coefficients = _closed_form_solution(X_poly, y_norm, regularization, alpha)
    elif solver == 'gradient_descent':
        coefficients = _gradient_descent(X_poly, y_norm)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Make predictions
    y_pred = X_poly @ coefficients

    # Compute metrics
    computed_metrics = _compute_metrics(y_norm, y_pred, metrics, custom_metric)

    # Prepare output
    result = {
        "result": coefficients,
        "metrics": computed_metrics,
        "params_used": {
            "degree": degree,
            "solver": solver,
            "normalization": normalization,
            "metrics": metrics,
            "regularization": regularization,
            "alpha": alpha
        },
        "warnings": []
    }

    return result

################################################################################
# fonction_cout
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def fonction_cout_fit(
    X: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, np.ndarray]]:
    """
    Calculate the cost function for polynomial regression.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    degree : int, optional
        Degree of the polynomial features, by default 1.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the features, by default None.
    metric : str, optional
        Metric to evaluate the cost function ('mse', 'mae', 'r2'), by default 'mse'.
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent'), by default 'closed_form'.
    regularization : Optional[str], optional
        Type of regularization ('l1', 'l2'), by default None.
    alpha : float, optional
        Regularization strength, by default 1.0.
    tol : float, optional
        Tolerance for convergence, by default 1e-4.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function, by default None.

    Returns
    -------
    Dict[str, Union[Dict, float, np.ndarray]]
        Dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.array([[1], [2], [3]])
    >>> y = np.array([2, 4, 6])
    >>> result = fonction_cout_fit(X, y, degree=1)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if a normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Create polynomial features
    X_poly = _create_polynomial_features(X_normalized, degree)

    # Solve for coefficients
    if solver == 'closed_form':
        coefs = _closed_form_solution(X_poly, y, regularization, alpha)
    elif solver == 'gradient_descent':
        coefs = _gradient_descent(X_poly, y, tol, max_iter, regularization, alpha)
    else:
        raise ValueError("Invalid solver specified.")

    # Calculate predictions
    y_pred = _predict(X_poly, coefs)

    # Calculate metrics
    metrics = _calculate_metrics(y, y_pred, metric, custom_metric)

    # Prepare the result dictionary
    result = {
        'result': metrics,
        'metrics': metrics,
        'params_used': {
            'degree': degree,
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'alpha': alpha
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate the input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be a 2D array and y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Apply normalization to the features."""
    if normalizer is not None:
        X = normalizer(X)
    return X

def _create_polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Create polynomial features."""
    n_samples = X.shape[0]
    X_poly = np.ones((n_samples, 1))
    for d in range(1, degree + 1):
        X_poly = np.hstack((X_poly, X ** d))
    return X_poly

def _closed_form_solution(X: np.ndarray, y: np.ndarray, regularization: Optional[str], alpha: float) -> np.ndarray:
    """Solve for coefficients using the closed-form solution."""
    if regularization is None:
        coefs = np.linalg.inv(X.T @ X) @ X.T @ y
    elif regularization == 'l2':
        identity = np.eye(X.shape[1])
        identity[0, 0] = 0
        coefs = np.linalg.inv(X.T @ X + alpha * identity) @ X.T @ y
    else:
        raise ValueError("Invalid regularization type specified.")
    return coefs

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    alpha: float
) -> np.ndarray:
    """Solve for coefficients using gradient descent."""
    n_features = X.shape[1]
    coefs = np.zeros(n_features)
    for _ in range(max_iter):
        gradients = _compute_gradients(X, y, coefs, regularization, alpha)
        coefs -= gradients
        if np.linalg.norm(gradients) < tol:
            break
    return coefs

def _compute_gradients(
    X: np.ndarray,
    y: np.ndarray,
    coefs: np.ndarray,
    regularization: Optional[str],
    alpha: float
) -> np.ndarray:
    """Compute the gradients for gradient descent."""
    residuals = y - X @ coefs
    gradients = -(2 / len(y)) * (X.T @ residuals)
    if regularization == 'l1':
        gradients += alpha * np.sign(coefs)
    elif regularization == 'l2':
        gradients += 2 * alpha * coefs
    return gradients

def _predict(X: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """Predict the target values."""
    return X @ coefs

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate the metrics."""
    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)
    return metrics

################################################################################
# methode_optimisation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input arrays for polynomial regression.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)

    Raises
    ------
    ValueError
        If inputs are invalid (NaN, inf, wrong dimensions)
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or inf values")

def normalize_data(X: np.ndarray, y: np.ndarray,
                  normalization: str = 'standard',
                  custom_normalization: Optional[Callable] = None) -> tuple:
    """
    Normalize input data according to specified method.

    Parameters
    ----------
    X : np.ndarray
        Input features array
    y : np.ndarray
        Target values array
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_normalization : callable, optional
        Custom normalization function

    Returns
    -------
    tuple
        (X_normalized, y_normalized)
    """
    if custom_normalization is not None:
        return custom_normalization(X, y)

    X_norm = X.copy()
    y_norm = y.copy()

    if normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / X_std
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - X_median) / (X_iqr + 1e-8)

    return X_norm, y_norm

def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Generate polynomial features.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    degree : int
        Degree of polynomial features

    Returns
    -------
    np.ndarray
        Polynomial features array of shape (n_samples, n_features * degree)
    """
    n_samples = X.shape[0]
    result = np.zeros((n_samples, 1))
    for d in range(degree + 1):
        result = np.hstack((result, X ** d))
    return result[:, 1:]

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   metrics: Union[str, list] = 'mse',
                   custom_metrics: Optional[Dict[str, Callable]] = None) -> Dict:
    """
    Compute regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    metrics : str or list, optional
        Metrics to compute ('mse', 'mae', 'r2')
    custom_metrics : dict, optional
        Dictionary of custom metrics {name: callable}

    Returns
    -------
    dict
        Computed metrics
    """
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
            result['r2'] = 1 - (ss_res / ss_tot)

    if custom_metrics is not None:
        for name, func in custom_metrics.items():
            result[name] = func(y_true, y_pred)

    return result

def closed_form_solution(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute closed form solution for polynomial regression.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (polynomial features)
    y : np.ndarray
        Target values

    Returns
    -------
    np.ndarray
        Estimated coefficients
    """
    return np.linalg.inv(X.T @ X) @ X.T @ y

def gradient_descent(X: np.ndarray, y: np.ndarray,
                    learning_rate: float = 0.01,
                    max_iter: int = 1000,
                    tol: float = 1e-4) -> np.ndarray:
    """
    Gradient descent optimization for polynomial regression.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (polynomial features)
    y : np.ndarray
        Target values
    learning_rate : float, optional
        Learning rate
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Tolerance for stopping criterion

    Returns
    -------
    np.ndarray
        Estimated coefficients
    """
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)

    for _ in range(max_iter):
        gradients = 2/n_samples * X.T @ (X @ coefs - y)
        new_coefs = coefs - learning_rate * gradients

        if np.linalg.norm(new_coefs - coefs) < tol:
            break

        coefs = new_coefs

    return coefs

def methode_optimisation_fit(X: np.ndarray, y: np.ndarray,
                           degree: int = 2,
                           normalization: str = 'standard',
                           solver: str = 'closed_form',
                           metrics: Union[str, list] = 'mse',
                           custom_metrics: Optional[Dict[str, Callable]] = None,
                           **solver_kwargs) -> Dict:
    """
    Fit polynomial regression model with optimization method.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    degree : int, optional
        Degree of polynomial features
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent')
    metrics : str or list, optional
        Metrics to compute ('mse', 'mae', 'r2')
    custom_metrics : dict, optional
        Dictionary of custom metrics {name: callable}
    **solver_kwargs : dict
        Additional solver-specific parameters

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': estimated coefficients
        - 'metrics': computed metrics
        - 'params_used': parameters used in fitting
        - 'warnings': any warnings generated

    Example
    -------
    >>> X = np.random.rand(100, 2)
    >>> y = 3 + 4*X[:,0] + 5*X[:,1]**2
    >>> result = methode_optimisation_fit(X, y, degree=2)
    """
    # Validate inputs
    validate_inputs(X, y)

    warnings = []

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization)

    # Generate polynomial features
    X_poly = polynomial_features(X_norm, degree)

    # Solve for coefficients
    if solver == 'closed_form':
        coefs = closed_form_solution(X_poly, y_norm)
    elif solver == 'gradient_descent':
        coefs = gradient_descent(X_poly, y_norm, **solver_kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute predictions
    y_pred = X_poly @ coefs

    # Compute metrics
    computed_metrics = compute_metrics(y_norm, y_pred, metrics, custom_metrics)

    return {
        'result': coefs,
        'metrics': computed_metrics,
        'params_used': {
            'degree': degree,
            'normalization': normalization,
            'solver': solver,
            'metrics': metrics
        },
        'warnings': warnings
    }

################################################################################
# sous_ajustement
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def sous_ajustement_fit(
    X: np.ndarray,
    y: np.ndarray,
    degree: int = 2,
    normalizer: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_normalizer: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit a polynomial regression model with underfitting control.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    degree : int, optional
        Degree of the polynomial features, by default 2.
    normalizer : str or callable, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') or custom callable, by default 'standard'.
    metric : str or callable, optional
        Metric to evaluate the model ('mse', 'mae', 'r2') or custom callable, by default 'mse'.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent'), by default 'closed_form'.
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2'), by default None.
    tol : float, optional
        Tolerance for convergence, by default 1e-6.
    max_iter : int, optional
        Maximum number of iterations for iterative solvers, by default 1000.
    custom_normalizer : callable, optional
        Custom normalizer function, by default None.
    custom_metric : callable, optional
        Custom metric function, by default None.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Fitted model coefficients.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used for fitting.
        - 'warnings': Any warnings encountered.

    Example
    -------
    >>> X = np.array([[1], [2], [3]])
    >>> y = np.array([1, 4, 9])
    >>> result = sous_ajustement_fit(X, y, degree=2)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, normalizer_used = _apply_normalization(X, y, normalizer, custom_normalizer)

    # Create polynomial features
    X_poly = _create_polynomial_features(X_norm, degree)

    # Choose solver
    if solver == 'closed_form':
        coefficients = _solve_closed_form(X_poly, y)
    elif solver == 'gradient_descent':
        coefficients = _solve_gradient_descent(X_poly, y, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization:
        coefficients = _apply_regularization(coefficients, X_poly, y, regularization)

    # Compute metrics
    metrics = _compute_metrics(X_poly, y, coefficients, metric, custom_metric)

    # Prepare output
    result = {
        'result': coefficients,
        'metrics': metrics,
        'params_used': {
            'degree': degree,
            'normalizer': normalizer_used,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

    return result

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

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: str,
    custom_normalizer: Optional[Callable]
) -> tuple:
    """Apply normalization to input data."""
    if custom_normalizer is not None:
        X_norm = custom_normalizer(X)
        y_norm = custom_normalizer(y.reshape(-1, 1)).flatten()
        return X_norm, 'custom'
    elif normalizer == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / X_std
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_norm = (y - y_mean) / y_std
        return X_norm, 'standard'
    elif normalizer == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min)
        y_min = np.min(y)
        y_max = np.max(y)
        y_norm = (y - y_min) / (y_max - y_min)
        return X_norm, 'minmax'
    elif normalizer == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - X_median) / X_iqr
        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        y_norm = (y - y_median) / y_iqr
        return X_norm, 'robust'
    else:
        return X, 'none'

def _create_polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Create polynomial features."""
    n_samples = X.shape[0]
    X_poly = np.ones((n_samples, 1))
    for d in range(1, degree + 1):
        X_poly = np.hstack([X_poly, X ** d])
    return X_poly

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve using closed-form solution."""
    XTX = np.dot(X.T, X)
    if not np.allclose(np.linalg.det(XTX), 0):
        coefficients = np.linalg.solve(XTX, np.dot(X.T, y))
    else:
        coefficients = np.linalg.pinv(XTX).dot(X.T).dot(y)
    return coefficients

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> np.ndarray:
    """Solve using gradient descent."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    learning_rate = 0.01
    prev_loss = float('inf')

    for _ in range(max_iter):
        predictions = np.dot(X, coefficients)
        errors = predictions - y
        gradient = 2 * np.dot(X.T, errors) / len(y)
        coefficients -= learning_rate * gradient
        current_loss = np.mean(errors ** 2)

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return coefficients

def _apply_regularization(
    coefficients: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    regularization: str
) -> np.ndarray:
    """Apply regularization to coefficients."""
    if regularization == 'l1':
        # Lasso regression
        alpha = 0.1
        XTX = np.dot(X.T, X)
        diag = alpha * np.eye(len(coefficients))
        coefficients = np.linalg.solve(XTX + diag, np.dot(X.T, y))
    elif regularization == 'l2':
        # Ridge regression
        alpha = 0.1
        XTX = np.dot(X.T, X)
        diag = alpha * np.eye(len(coefficients))
        coefficients = np.linalg.solve(XTX + diag, np.dot(X.T, y))
    elif regularization == 'elasticnet':
        # Elastic net regression
        alpha = 0.1
        l1_ratio = 0.5
        XTX = np.dot(X.T, X)
        diag_l1 = l1_ratio * alpha * np.eye(len(coefficients))
        diag_l2 = (1 - l1_ratio) * alpha * np.eye(len(coefficients))
        coefficients = np.linalg.solve(XTX + diag_l2, np.dot(X.T, y))
        coefficients = np.sign(coefficients) * np.maximum(np.abs(coefficients) - diag_l1, 0)
    return coefficients

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute evaluation metrics."""
    predictions = np.dot(X, coefficients)
    metrics = {}

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y, predictions)
    elif metric == 'mse':
        metrics['mse'] = np.mean((y - predictions) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y - predictions))
    elif metric == 'r2':
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

################################################################################
# sur_ajustement
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

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

def _normalize_data(X: np.ndarray, y: np.ndarray,
                   normalization: str = 'standard',
                   custom_norm: Optional[Callable] = None) -> tuple:
    """Normalize data based on user choice."""
    if custom_norm is not None:
        X_norm = custom_norm(X)
        y_norm = custom_norm(y.reshape(-1, 1)).flatten()
    elif normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / X_std
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_norm = (y - y_mean) / y_std
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
        X_iqr = X_q75 - X_q25
        X_norm = (X - X_median) / (X_iqr + 1e-8)
        y_median = np.median(y)
        y_q75 = np.percentile(y, 75)
        y_q25 = np.percentile(y, 25)
        y_iqr = y_q75 - y_q25
        y_norm = (y - y_median) / (y_iqr + 1e-8)
    else:
        X_norm = X.copy()
        y_norm = y.copy()

    return X_norm, y_norm

def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
                   metric: str = 'mse',
                   custom_metric: Optional[Callable] = None) -> float:
    """Compute metric based on user choice."""
    if custom_metric is not None:
        return custom_metric(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Generate polynomial features."""
    n_samples = X.shape[0]
    X_poly = np.ones((n_samples, 1))
    for d in range(1, degree + 1):
        X_poly = np.hstack([X_poly, X ** d])
    return X_poly

def _closed_form_solution(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute closed form solution for polynomial regression."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _gradient_descent(X: np.ndarray, y: np.ndarray,
                     learning_rate: float = 0.01,
                     n_iter: int = 1000,
                     tol: float = 1e-4) -> np.ndarray:
    """Gradient descent solver for polynomial regression."""
    n_features = X.shape[1]
    coefs = np.zeros(n_features)
    for _ in range(n_iter):
        gradient = 2 * X.T @ (X @ coefs - y) / len(y)
        new_coefs = coefs - learning_rate * gradient
        if np.linalg.norm(new_coefs - coefs) < tol:
            break
        coefs = new_coefs
    return coefs

def sur_ajustement_fit(X: np.ndarray, y: np.ndarray,
                      degree: int = 2,
                      normalization: str = 'standard',
                      metric: str = 'mse',
                      solver: str = 'closed_form',
                      learning_rate: float = 0.01,
                      n_iter: int = 1000,
                      tol: float = 1e-4,
                      custom_norm: Optional[Callable] = None,
                      custom_metric: Optional[Callable] = None) -> Dict:
    """
    Fit polynomial regression model and detect overfitting.

    Parameters:
    -----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    degree : int
        Degree of polynomial features
    normalization : str or callable
        Normalization method ('standard', 'minmax', 'robust') or custom function
    metric : str or callable
        Evaluation metric ('mse', 'mae', 'r2') or custom function
    solver : str
        Solver method ('closed_form', 'gradient_descent')
    learning_rate : float
        Learning rate for gradient descent
    n_iter : int
        Number of iterations for gradient descent
    tol : float
        Tolerance for convergence
    custom_norm : callable, optional
        Custom normalization function
    custom_metric : callable, optional
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 2)
    >>> y = 3*X[:,0]**2 + 2*X[:,1] - 1
    >>> result = sur_ajustement_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization, custom_norm)

    # Generate polynomial features
    X_poly = _polynomial_features(X_norm, degree)

    # Solve for coefficients
    if solver == 'closed_form':
        coefs = _closed_form_solution(X_poly, y_norm)
    elif solver == 'gradient_descent':
        coefs = _gradient_descent(X_poly, y_norm, learning_rate, n_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Make predictions
    y_pred = X_poly @ coefs

    # Compute metrics
    metric_value = _compute_metric(y_norm, y_pred, metric, custom_metric)

    # Check for overfitting
    warnings = []
    if degree > 1 and metric_value < -0.5:
        warnings.append("High risk of overfitting detected")

    return {
        'result': y_pred,
        'metrics': {metric: metric_value},
        'params_used': {
            'degree': degree,
            'normalization': normalization if custom_norm is None else 'custom',
            'metric': metric if custom_metric is None else 'custom',
            'solver': solver,
            'learning_rate': learning_rate if solver == 'gradient_descent' else None,
            'n_iter': n_iter if solver == 'gradient_descent' else None
        },
        'warnings': warnings
    }

################################################################################
# validation_croisee
################################################################################

import numpy as np
from typing import Callable, Dict, List, Optional, Union, Any

def validation_croisee_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    normalisation: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Perform cross-validation for polynomial regression.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    n_splits : int, optional
        Number of cross-validation splits.
    normalisation : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : Union[str, Callable], optional
        Metric to evaluate: 'mse', 'mae', 'r2', or custom callable.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : Optional[str], optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float, optional
        Tolerance for solver convergence.
    max_iter : int, optional
        Maximum iterations for iterative solvers.
    random_state : Optional[int], optional
        Random seed for reproducibility.
    custom_metric : Optional[Callable], optional
        Custom metric function if not using built-in metrics.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized, y_normalized = _apply_normalization(X, y, normalisation)

    # Initialize results storage
    results = {
        'result': [],
        'metrics': [],
        'params_used': {
            'n_splits': n_splits,
            'normalisation': normalisation,
            'metric': metric if isinstance(metric, str) else 'custom',
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    # Perform cross-validation
    for fold in range(n_splits):
        X_train, X_test, y_train, y_test = _split_data(X_normalized, y_normalized, n_splits, fold, random_state)

        # Fit model
        model = _fit_polynomial_regression(
            X_train, y_train,
            solver=solver,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter
        )

        # Predict and compute metrics
        y_pred = _predict_polynomial_regression(model, X_test)
        metric_value = _compute_metric(y_test, y_pred, metric, custom_metric)

        results['result'].append(model)
        results['metrics'].append(metric_value)

    return results

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values")

def _apply_normalization(X: np.ndarray, y: np.ndarray, method: str) -> tuple:
    """Apply specified normalization to data."""
    if method == 'none':
        return X, y
    elif method == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / (X_std + 1e-8)
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_normalized = (y - y_mean) / (y_std + 1e-8)
        return X_normalized, y_normalized
    elif method == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
        y_min = np.min(y)
        y_max = np.max(y)
        y_normalized = (y - y_min) / (y_max - y_min + 1e-8)
        return X_normalized, y_normalized
    elif method == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - X_median) / (X_iqr + 1e-8)
        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        y_normalized = (y - y_median) / (y_iqr + 1e-8)
        return X_normalized, y_normalized
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _split_data(X: np.ndarray, y: np.ndarray, n_splits: int, fold: int, random_state: Optional[int]) -> tuple:
    """Split data into training and test sets for cross-validation."""
    np.random.seed(random_state)
    indices = np.random.permutation(X.shape[0])
    fold_size = X.shape[0] // n_splits
    test_indices = indices[fold * fold_size : (fold + 1) * fold_size]
    train_indices = np.setdiff1d(indices, test_indices)
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def _fit_polynomial_regression(
    X: np.ndarray,
    y: np.ndarray,
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """Fit polynomial regression model using specified solver."""
    if solver == 'closed_form':
        return _fit_closed_form(X, y, regularization)
    elif solver == 'gradient_descent':
        return _fit_gradient_descent(X, y, regularization, tol, max_iter)
    elif solver == 'newton':
        return _fit_newton(X, y, regularization, tol, max_iter)
    elif solver == 'coordinate_descent':
        return _fit_coordinate_descent(X, y, regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _fit_closed_form(X: np.ndarray, y: np.ndarray, regularization: Optional[str]) -> Dict[str, Any]:
    """Fit polynomial regression using closed-form solution."""
    # Add polynomial features
    X_poly = _add_polynomial_features(X)

    # Regularization matrix
    if regularization == 'l1':
        penalty = np.eye(X_poly.shape[1])
    elif regularization == 'l2':
        penalty = np.eye(X_poly.shape[1])
    elif regularization == 'elasticnet':
        penalty = np.eye(X_poly.shape[1])
    else:
        penalty = np.zeros((X_poly.shape[1], X_poly.shape[1]))

    # Closed-form solution
    coefficients = np.linalg.inv(X_poly.T @ X_poly + penalty) @ X_poly.T @ y

    return {'coefficients': coefficients, 'solver': 'closed_form'}

def _fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Fit polynomial regression using gradient descent."""
    # Add polynomial features
    X_poly = _add_polynomial_features(X)

    # Initialize coefficients
    coefficients = np.zeros(X_poly.shape[1])

    for _ in range(max_iter):
        # Compute predictions and residuals
        y_pred = X_poly @ coefficients
        residuals = y_pred - y

        # Compute gradient
        gradient = X_poly.T @ residuals / len(y)

        # Add regularization term if specified
        if regularization == 'l1':
            gradient += np.sign(coefficients)
        elif regularization == 'l2':
            gradient += coefficients
        elif regularization == 'elasticnet':
            gradient += np.sign(coefficients) + coefficients

        # Update coefficients
        new_coefficients = coefficients - tol * gradient

        # Check for convergence
        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break

        coefficients = new_coefficients

    return {'coefficients': coefficients, 'solver': 'gradient_descent'}

def _fit_newton(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Fit polynomial regression using Newton's method."""
    # Add polynomial features
    X_poly = _add_polynomial_features(X)

    # Initialize coefficients
    coefficients = np.zeros(X_poly.shape[1])

    for _ in range(max_iter):
        # Compute predictions and residuals
        y_pred = X_poly @ coefficients
        residuals = y_pred - y

        # Compute gradient and Hessian
        gradient = X_poly.T @ residuals / len(y)
        hessian = (X_poly.T @ X_poly) / len(y)

        # Add regularization term if specified
        if regularization == 'l1':
            hessian += np.diag(np.sign(coefficients))
        elif regularization == 'l2':
            hessian += np.eye(X_poly.shape[1])
        elif regularization == 'elasticnet':
            hessian += np.diag(np.sign(coefficients)) + np.eye(X_poly.shape[1])

        # Update coefficients
        new_coefficients = coefficients - np.linalg.inv(hessian) @ gradient

        # Check for convergence
        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break

        coefficients = new_coefficients

    return {'coefficients': coefficients, 'solver': 'newton'}

def _fit_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Fit polynomial regression using coordinate descent."""
    # Add polynomial features
    X_poly = _add_polynomial_features(X)

    # Initialize coefficients
    coefficients = np.zeros(X_poly.shape[1])

    for _ in range(max_iter):
        for i in range(len(coefficients)):
            # Compute residuals without current coefficient
            residuals = y - (X_poly @ coefficients)

            # Compute gradient for current coefficient
            gradient = X_poly[:, i].T @ residuals

            # Add regularization term if specified
            if regularization == 'l1':
                gradient += np.sign(coefficients[i])
            elif regularization == 'l2':
                gradient += coefficients[i]
            elif regularization == 'elasticnet':
                gradient += np.sign(coefficients[i]) + coefficients[i]

            # Update coefficient
            new_coefficient = coefficients[i] - tol * gradient

            # Check for convergence
            if np.abs(new_coefficient - coefficients[i]) < tol:
                continue

            coefficients[i] = new_coefficient

    return {'coefficients': coefficients, 'solver': 'coordinate_descent'}

def _add_polynomial_features(X: np.ndarray) -> np.ndarray:
    """Add polynomial features to the input data."""
    # This is a placeholder - actual implementation would depend on degree
    return X  # Replace with actual polynomial feature expansion

def _predict_polynomial_regression(model: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Predict using fitted polynomial regression model."""
    # Add polynomial features
    X_poly = _add_polynomial_features(X)

    return X_poly @ model['coefficients']

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> float:
    """Compute specified metric between true and predicted values."""
    if isinstance(metric, str):
        if metric == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)
        elif metric == 'logloss':
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        return metric(y_true, y_pred)
    else:
        raise ValueError("Metric must be a string or callable")

################################################################################
# regularisation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regularisation_fit(
    X: np.ndarray,
    y: np.ndarray,
    degree: int = 2,
    reg_type: str = 'l2',
    alpha: float = 1.0,
    solver: str = 'closed_form',
    metric: Union[str, Callable] = 'mse',
    normalizer: Optional[Callable] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict:
    """
    Fit a polynomial regression model with regularization.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    degree : int, optional
        Degree of the polynomial features (default=2)
    reg_type : str, optional
        Type of regularization: 'none', 'l1', 'l2', or 'elasticnet' (default='l2')
    alpha : float, optional
        Regularization strength (default=1.0)
    solver : str, optional
        Solver to use: 'closed_form', 'gradient_descent', or 'coordinate_descent' (default='closed_form')
    metric : str or callable, optional
        Metric to evaluate: 'mse', 'mae', 'r2', or custom callable (default='mse')
    normalizer : callable, optional
        Normalization function or None (default=None)
    tol : float, optional
        Tolerance for convergence (default=1e-4)
    max_iter : int, optional
        Maximum number of iterations (default=1000)
    random_state : int, optional
        Random seed for reproducibility (default=None)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.array([[1], [2], [3]])
    >>> y = np.array([1, 4, 9])
    >>> result = regularisation_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_norm, y_norm = _apply_normalization(X, y, normalizer)

    # Create polynomial features
    X_poly = _create_polynomial_features(X_norm, degree)

    # Choose solver
    if solver == 'closed_form':
        coefficients = _solve_closed_form(X_poly, y_norm, reg_type, alpha)
    elif solver == 'gradient_descent':
        coefficients = _solve_gradient_descent(X_poly, y_norm, reg_type, alpha,
                                             tol, max_iter, random_state)
    elif solver == 'coordinate_descent':
        coefficients = _solve_coordinate_descent(X_poly, y_norm, reg_type, alpha,
                                               tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    y_pred = X_poly @ coefficients
    metrics = _calculate_metrics(y_norm, y_pred, metric)

    return {
        'result': coefficients,
        'metrics': metrics,
        'params_used': {
            'degree': degree,
            'reg_type': reg_type,
            'alpha': alpha,
            'solver': solver,
            'metric': metric.__name__ if callable(metric) else metric
        },
        'warnings': _check_warnings(y_norm, y_pred)
    }

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
    normalizer: Optional[Callable]
) -> tuple:
    """Apply normalization if specified."""
    X_norm = X.copy()
    y_norm = y.copy()

    if normalizer is not None:
        X_norm = normalizer(X)
        y_norm = normalizer(y.reshape(-1, 1)).flatten()

    return X_norm, y_norm

def _create_polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Create polynomial features."""
    n_samples = X.shape[0]
    X_poly = np.ones((n_samples, 1))

    for d in range(1, degree + 1):
        X_poly = np.hstack([X_poly, X ** d])

    return X_poly

def _solve_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    reg_type: str,
    alpha: float
) -> np.ndarray:
    """Solve using closed form solution."""
    n_features = X.shape[1]
    identity = np.eye(n_features)

    if reg_type == 'l2':
        regularizer = alpha * identity
    elif reg_type == 'l1':
        raise NotImplementedError("L1 regularization not implemented for closed form")
    elif reg_type == 'elasticnet':
        raise NotImplementedError("ElasticNet regularization not implemented for closed form")
    elif reg_type == 'none':
        regularizer = np.zeros_like(identity)
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")

    # Add small value to diagonal for numerical stability
    regularizer[0, 0] += 1e-6

    XtX = X.T @ X
    Xty = X.T @ y

    coefficients = np.linalg.solve(XtX + regularizer, Xty)

    return coefficients

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    reg_type: str,
    alpha: float,
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> np.ndarray:
    """Solve using gradient descent."""
    if random_state is not None:
        np.random.seed(random_state)

    n_features = X.shape[1]
    coefficients = np.random.randn(n_features)
    prev_cost = float('inf')

    for _ in range(max_iter):
        gradients = _compute_gradients(X, y, coefficients, reg_type, alpha)
        coefficients -= 0.01 * gradients

        current_cost = _compute_cost(X, y, coefficients, reg_type, alpha)

        if abs(prev_cost - current_cost) < tol:
            break

        prev_cost = current_cost

    return coefficients

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    reg_type: str,
    alpha: float,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve using coordinate descent."""
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    prev_cost = float('inf')

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - (X @ coefficients) + coefficients[j] * X_j

            if reg_type == 'l2':
                numerator = X_j.T @ residuals
                denominator = X_j.T @ X_j + alpha * n_samples
            elif reg_type == 'l1':
                numerator = np.sum(X_j * (residuals > 0)) - alpha / 2
                denominator = np.sum(X_j * (residuals > 0))
            elif reg_type == 'none':
                numerator = X_j.T @ residuals
                denominator = X_j.T @ X_j
            else:
                raise ValueError(f"Unknown regularization type: {reg_type}")

            coefficients[j] = numerator / denominator

        current_cost = _compute_cost(X, y, coefficients, reg_type, alpha)

        if abs(prev_cost - current_cost) < tol:
            break

        prev_cost = current_cost

    return coefficients

def _compute_gradients(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    reg_type: str,
    alpha: float
) -> np.ndarray:
    """Compute gradients for gradient descent."""
    residuals = y - X @ coefficients
    gradients = -(X.T @ residuals) / len(y)

    if reg_type == 'l2':
        gradients += 2 * alpha * coefficients
    elif reg_type == 'l1':
        gradients += alpha * np.sign(coefficients)
    elif reg_type != 'none':
        raise ValueError(f"Unknown regularization type: {reg_type}")

    return gradients

def _compute_cost(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    reg_type: str,
    alpha: float
) -> float:
    """Compute cost function."""
    residuals = y - X @ coefficients
    mse = np.mean(residuals ** 2)

    if reg_type == 'l2':
        penalty = alpha * np.sum(coefficients ** 2)
    elif reg_type == 'l1':
        penalty = alpha * np.sum(np.abs(coefficients))
    elif reg_type == 'elasticnet':
        penalty = alpha * (np.sum(coefficients ** 2) + np.sum(np.abs(coefficients)))
    else:
        penalty = 0

    return mse + penalty

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate evaluation metrics."""
    metrics = {}

    if metric == 'mse' or (callable(metric) and metric.__name__ == 'mse'):
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    if metric == 'mae' or (callable(metric) and metric.__name__ == 'mae'):
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric == 'r2' or (callable(metric) and metric.__name__ == 'r2'):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - ss_res / ss_tot

    if callable(metric):
        metrics['custom'] = metric(y_true, y_pred)

    return metrics

def _check_warnings(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> list:
    """Check for potential warnings."""
    warnings = []

    if np.any(np.isnan(y_pred)):
        warnings.append("Predictions contain NaN values")
    if np.any(np.isinf(y_pred)):
        warnings.append("Predictions contain infinite values")

    return warnings

################################################################################
# erreur_quadratique_moyenne
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

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

def _normalize_data(X: np.ndarray, y: np.ndarray, normalization: str) -> tuple:
    """Normalize data according to specified method."""
    if normalization == 'none':
        return X, y
    elif normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        y_mean = np.mean(y)
        y_std = np.std(y)
        X_normalized = (X - X_mean) / X_std
        y_normalized = (y - y_mean) / y_std
        return X_normalized, y_normalized
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        y_min = np.min(y)
        y_max = np.max(y)
        X_normalized = (X - X_min) / (X_max - X_min)
        y_normalized = (y - y_min) / (y_max - y_min)
        return X_normalized, y_normalized
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_q75, X_q25 = np.percentile(X, [75, 25], axis=0)
        X_iqr = X_q75 - X_q25
        y_median = np.median(y)
        y_q75, y_q25 = np.percentile(y, [75, 25])
        y_iqr = y_q75 - y_q25
        X_normalized = (X - X_median) / X_iqr
        y_normalized = (y - y_median) / y_iqr
        return X_normalized, y_normalized
    else:
        raise ValueError("Unknown normalization method")

def _compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def _compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _compute_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Log Loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _compute_custom_metric(y_true: np.ndarray, y_pred: np.ndarray, metric_func: Callable) -> float:
    """Compute custom metric using provided callable."""
    return metric_func(y_true, y_pred)

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve polynomial regression using closed-form solution."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01,
                           n_iter: int = 1000, tol: float = 1e-4) -> np.ndarray:
    """Solve polynomial regression using gradient descent."""
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    for _ in range(n_iter):
        gradient = 2 * X.T @ (X @ coefficients - y) / n_samples
        new_coefficients = coefficients - learning_rate * gradient
        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients
    return coefficients

def _solve_newton(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve polynomial regression using Newton's method."""
    return _solve_closed_form(X, y)  # Simplified for this example

def _solve_coordinate_descent(X: np.ndarray, y: np.ndarray, n_iter: int = 1000,
                             tol: float = 1e-4) -> np.ndarray:
    """Solve polynomial regression using coordinate descent."""
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    for _ in range(n_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - X @ coefficients + coefficients[j] * X_j
            coefficients[j] = np.sum(X_j * residuals) / np.sum(X_j ** 2)
        if np.linalg.norm(coefficients - np.zeros(n_features)) < tol:
            break
    return coefficients

def erreur_quadratique_moyenne_fit(X: np.ndarray, y: np.ndarray,
                                  normalization: str = 'none',
                                  metric: str = 'mse',
                                  solver: str = 'closed_form',
                                  custom_metric: Optional[Callable] = None,
                                  **solver_kwargs) -> Dict[str, Any]:
    """
    Compute polynomial regression with configurable options.

    Parameters:
    - X: Input features (2D array)
    - y: Target values (1D array)
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - metric: Metric to compute ('mse', 'mae', 'r2', 'logloss')
    - solver: Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    - custom_metric: Custom metric function (callable)
    - solver_kwargs: Additional arguments for the solver

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization)

    # Solve regression
    if solver == 'closed_form':
        coefficients = _solve_closed_form(X_norm, y_norm)
    elif solver == 'gradient_descent':
        coefficients = _solve_gradient_descent(X_norm, y_norm, **solver_kwargs)
    elif solver == 'newton':
        coefficients = _solve_newton(X_norm, y_norm)
    elif solver == 'coordinate_descent':
        coefficients = _solve_coordinate_descent(X_norm, y_norm, **solver_kwargs)
    else:
        raise ValueError("Unknown solver method")

    # Compute predictions
    y_pred = X_norm @ coefficients

    # Compute metrics
    metrics = {}
    if metric == 'mse':
        metrics['mse'] = _compute_mse(y_norm, y_pred)
    elif metric == 'mae':
        metrics['mae'] = _compute_mae(y_norm, y_pred)
    elif metric == 'r2':
        metrics['r2'] = _compute_r2(y_norm, y_pred)
    elif metric == 'logloss':
        metrics['logloss'] = _compute_logloss(y_norm, y_pred)
    elif custom_metric is not None:
        metrics['custom'] = _compute_custom_metric(y_norm, y_pred, custom_metric)
    else:
        raise ValueError("Unknown metric or no custom metric provided")

    # Prepare output
    result = {
        'result': y_pred,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver,
            'coefficients': coefficients
        },
        'warnings': []
    }

    return result

# Example usage:
"""
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 2, 3])
result = erreur_quadratique_moyenne_fit(X, y)
"""

################################################################################
# matrice_design
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def validate_inputs(X: np.ndarray) -> None:
    """Validate input matrix dimensions and types."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")

def normalize_data(
    X: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize the input data using specified method or custom function."""
    if custom_func is not None:
        return custom_func(X)

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
    return X_normalized

def compute_design_matrix(
    X: np.ndarray,
    degree: int = 1,
    include_bias: bool = True
) -> np.ndarray:
    """Compute the polynomial design matrix."""
    n_samples, n_features = X.shape
    design_matrix = np.ones((n_samples, 1)) if include_bias else np.empty((n_samples, 0))

    for d in range(1, degree + 1):
        if d == 1:
            design_matrix = np.hstack([design_matrix, X])
        else:
            poly_features = X.copy()
            for _ in range(d - 1):
                poly_features *= X
            design_matrix = np.hstack([design_matrix, poly_features])

    return design_matrix

def matrice_design_fit(
    X: np.ndarray,
    degree: int = 1,
    include_bias: bool = True,
    normalization_method: str = "standard",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute the polynomial design matrix with configurable options.

    Parameters:
    - X: Input data matrix of shape (n_samples, n_features)
    - degree: Degree of the polynomial features
    - include_bias: Whether to include a bias column (intercept)
    - normalization_method: Normalization method ('none', 'standard', 'minmax', 'robust')
    - custom_normalization: Custom normalization function

    Returns:
    - Dictionary containing the design matrix, metrics, and parameters used
    """
    # Validate inputs
    validate_inputs(X)

    # Normalize data if required
    if normalization_method != "none" or custom_normalization is not None:
        X = normalize_data(X, normalization_method, custom_normalization)

    # Compute design matrix
    design_matrix = compute_design_matrix(X, degree, include_bias)

    # Prepare output
    result = {
        "design_matrix": design_matrix,
        "metrics": {},
        "params_used": {
            "degree": degree,
            "include_bias": include_bias,
            "normalization_method": normalization_method if custom_normalization is None else "custom",
        },
        "warnings": []
    }

    return result

# Example usage:
"""
X = np.array([[1, 2], [3, 4], [5, 6]])
result = matrice_design_fit(X, degree=2, include_bias=True)
print(result['design_matrix'])
"""

################################################################################
# polynome_interpolation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for polynomial interpolation."""
    if X.ndim != 1 or y.ndim != 1:
        raise ValueError("X and y must be 1-dimensional arrays.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def normalize_data(X: np.ndarray, y: np.ndarray, normalization: str) -> tuple:
    """Normalize data based on the specified method."""
    if normalization == "none":
        return X, y
    elif normalization == "standard":
        X_mean = np.mean(X)
        X_std = np.std(X)
        y_mean = np.mean(y)
        y_std = np.std(y)
        X_normalized = (X - X_mean) / X_std
        y_normalized = (y - y_mean) / y_std
        return X_normalized, y_normalized
    elif normalization == "minmax":
        X_min = np.min(X)
        X_max = np.max(X)
        y_min = np.min(y)
        y_max = np.max(y)
        X_normalized = (X - X_min) / (X_max - X_min)
        y_normalized = (y - y_min) / (y_max - y_min)
        return X_normalized, y_normalized
    elif normalization == "robust":
        X_median = np.median(X)
        X_q1 = np.percentile(X, 25)
        X_q3 = np.percentile(X, 75)
        y_median = np.median(y)
        y_q1 = np.percentile(y, 25)
        y_q3 = np.percentile(y, 75)
        X_normalized = (X - X_median) / (X_q3 - X_q1)
        y_normalized = (y - y_median) / (y_q3 - y_q1)
        return X_normalized, y_normalized
    else:
        raise ValueError("Invalid normalization method.")

def build_design_matrix(X: np.ndarray, degree: int) -> np.ndarray:
    """Build the design matrix for polynomial regression."""
    n_samples = len(X)
    design_matrix = np.zeros((n_samples, degree + 1))
    for d in range(degree + 1):
        design_matrix[:, d] = X ** d
    return design_matrix

def closed_form_solution(design_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the closed-form solution for polynomial coefficients."""
    return np.linalg.inv(design_matrix.T @ design_matrix) @ design_matrix.T @ y

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metrics: Dict[str, Callable]) -> Dict[str, float]:
    """Compute specified metrics for the polynomial interpolation."""
    results = {}
    for name, metric_func in metrics.items():
        if name == "mse":
            results[name] = np.mean((y_true - y_pred) ** 2)
        elif name == "mae":
            results[name] = np.mean(np.abs(y_true - y_pred))
        elif name == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            results[name] = 1 - (ss_res / ss_tot)
        else:
            results[name] = metric_func(y_true, y_pred)
    return results

def polynome_interpolation_fit(X: np.ndarray, y: np.ndarray,
                              degree: int = 3,
                              normalization: str = "none",
                              solver: str = "closed_form",
                              metrics: Dict[str, Union[str, Callable]] = None,
                              custom_metric: Optional[Callable] = None) -> Dict:
    """
    Perform polynomial interpolation on the given data.

    Parameters:
    - X: Input feature array.
    - y: Target values array.
    - degree: Degree of the polynomial (default: 3).
    - normalization: Normalization method ("none", "standard", "minmax", "robust").
    - solver: Solver method ("closed_form" for now).
    - metrics: Dictionary of metric names or callables to compute.
    - custom_metric: Optional custom metric function.

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data if specified
    X_norm, y_norm = normalize_data(X, y, normalization)

    # Build design matrix
    design_matrix = build_design_matrix(X_norm, degree)

    # Solve for coefficients
    if solver == "closed_form":
        coefficients = closed_form_solution(design_matrix, y_norm)
    else:
        raise ValueError("Unsupported solver method.")

    # Predict values
    y_pred = design_matrix @ coefficients

    # Compute metrics
    default_metrics = {"mse": "mse", "mae": "mae", "r2": "r2"}
    if metrics is None:
        metrics = default_metrics
    else:
        metrics.update(default_metrics)

    if custom_metric is not None:
        metrics["custom"] = custom_metric

    metric_functions = {
        name: (lambda y_true, y_pred, m=metric: compute_metrics(y_true, y_pred, {m: None})[m]
               if isinstance(metric, str) else metric)
        for name, metric in metrics.items()
    }

    computed_metrics = compute_metrics(y_norm, y_pred, metric_functions)

    # Prepare results
    result = {
        "result": {
            "coefficients": coefficients,
            "predicted_values": y_pred
        },
        "metrics": computed_metrics,
        "params_used": {
            "degree": degree,
            "normalization": normalization,
            "solver": solver
        },
        "warnings": []
    }

    return result

# Example usage:
"""
X = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])

result = polynome_interpolation_fit(X, y, degree=2)
print(result)
"""

################################################################################
# selection_degree
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    degree_range: tuple,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse'
) -> None:
    """Validate input data and parameters."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Input arrays must contain only finite values")
    if degree_range[0] < 1 or degree_range[1] <= degree_range[0]:
        raise ValueError("Invalid degree range")

def _normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    normalize: str = 'standard'
) -> tuple:
    """Normalize data according to specified method."""
    if normalize == 'none':
        return X, y
    elif normalize == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        y_mean = np.mean(y)
        y_std = np.std(y)
        X_norm = (X - X_mean) / X_std
        y_norm = (y - y_mean) / y_std
    elif normalize == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        y_min = np.min(y)
        y_max = np.max(y)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        y_norm = (y - y_min) / (y_max - y_min + 1e-8)
    elif normalize == 'robust':
        X_median = np.median(X, axis=0)
        X_q75 = np.quantile(X, 0.75, axis=0)
        X_q25 = np.quantile(X, 0.25, axis=0)
        y_median = np.median(y)
        y_q75 = np.quantile(y, 0.75)
        y_q25 = np.quantile(y, 0.25)
        X_norm = (X - X_median) / (X_q75 - X_q25 + 1e-8)
        y_norm = (y - y_median) / (y_q75 - y_q25 + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")
    return X_norm, y_norm

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> float:
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
        return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _polynomial_regression(
    X: np.ndarray,
    y: np.ndarray,
    degree: int,
    solver: str = 'closed_form'
) -> tuple:
    """Perform polynomial regression for given degree."""
    X_poly = np.column_stack([X ** d for d in range(1, degree + 1)])
    if solver == 'closed_form':
        XTX = np.dot(X_poly.T, X_poly)
        XTY = np.dot(X_poly.T, y)
        try:
            coeffs = np.linalg.solve(XTX, XTY)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.pinv(XTX) @ XTY
        y_pred = np.dot(X_poly, coeffs)
    else:
        raise ValueError(f"Unknown solver: {solver}")
    return y_pred, coeffs

def selection_degree_fit(
    X: np.ndarray,
    y: np.ndarray,
    degree_range: tuple = (1, 5),
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form'
) -> Dict:
    """Select optimal polynomial degree for regression.

    Example:
        >>> X = np.random.rand(100, 2)
        >>> y = 3 + 2*X[:, 0] - X[:, 1]**2
        >>> result = selection_degree_fit(X, y)
    """
    _validate_inputs(X, y, degree_range, normalize, metric)

    X_norm, y_norm = _normalize_data(X, y, normalize)
    best_degree = None
    best_metric = float('inf')
    results = []

    for degree in range(degree_range[0], degree_range[1] + 1):
        y_pred, coeffs = _polynomial_regression(X_norm, y_norm, degree, solver)
        current_metric = _compute_metric(y_norm, y_pred, metric)

        if current_metric < best_metric:
            best_metric = current_metric
            best_degree = degree

        results.append({
            'degree': degree,
            'metric': current_metric,
            'coefficients': coeffs
        })

    # Recompute with best degree on original scale
    y_pred_best, coeffs_best = _polynomial_regression(X_norm, y_norm, best_degree, solver)
    metric_best = _compute_metric(y, np.dot(np.column_stack([X ** d for d in range(1, best_degree + 1)]), coeffs_best), metric)

    return {
        'result': best_degree,
        'metrics': {d['degree']: d['metric'] for d in results},
        'params_used': {
            'normalize': normalize,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

################################################################################
# splines
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 1 or y.ndim != 1:
        raise ValueError("X and y must be 1-dimensional arrays.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def normalize_data(X: np.ndarray, y: np.ndarray, method: str) -> tuple:
    """Normalize data based on the specified method."""
    if method == "none":
        return X, y
    elif method == "standard":
        X_mean = np.mean(X)
        X_std = np.std(X)
        y_mean = np.mean(y)
        y_std = np.std(y)
        X_normalized = (X - X_mean) / X_std
        y_normalized = (y - y_mean) / y_std
        return X_normalized, y_normalized
    elif method == "minmax":
        X_min = np.min(X)
        X_max = np.max(X)
        y_min = np.min(y)
        y_max = np.max(y)
        X_normalized = (X - X_min) / (X_max - X_min)
        y_normalized = (y - y_min) / (y_max - y_min)
        return X_normalized, y_normalized
    elif method == "robust":
        X_median = np.median(X)
        X_q1 = np.percentile(X, 25)
        X_q3 = np.percentile(X, 75)
        y_median = np.median(y)
        y_q1 = np.percentile(y, 25)
        y_q3 = np.percentile(y, 75)
        X_normalized = (X - X_median) / (X_q3 - X_q1)
        y_normalized = (y - y_median) / (y_q3 - y_q1)
        return X_normalized, y_normalized
    else:
        raise ValueError("Invalid normalization method.")

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Compute the specified metric."""
    if metric == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    elif metric == "logloss":
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError("Invalid metric.")

def splines_fit(X: np.ndarray,
                y: np.ndarray,
                degree: int = 3,
                n_knots: int = 5,
                normalization: str = "none",
                metric: str = "mse",
                solver: str = "closed_form",
                regularization: Optional[str] = None,
                alpha: float = 1.0,
                tol: float = 1e-4,
                max_iter: int = 1000) -> Dict:
    """
    Fit spline regression model to the data.

    Parameters:
    - X: Input features (1D array).
    - y: Target values (1D array).
    - degree: Degree of the spline.
    - n_knots: Number of knots for the spline.
    - normalization: Normalization method ("none", "standard", "minmax", "robust").
    - metric: Metric to evaluate the model ("mse", "mae", "r2", "logloss").
    - solver: Solver method ("closed_form", "gradient_descent", "newton", "coordinate_descent").
    - regularization: Regularization method (None, "l1", "l2", "elasticnet").
    - alpha: Regularization strength.
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations.

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings.
    """
    validate_inputs(X, y)
    X_norm, y_norm = normalize_data(X, y, normalization)

    # Placeholder for spline fitting logic
    coefficients = np.random.rand(degree + 1)  # Dummy coefficients

    y_pred = np.polyval(coefficients, X_norm)
    metric_value = compute_metric(y_norm, y_pred, metric)

    result = {
        "result": coefficients,
        "metrics": {metric: metric_value},
        "params_used": {
            "degree": degree,
            "n_knots": n_knots,
            "normalization": normalization,
            "metric": metric,
            "solver": solver,
            "regularization": regularization,
            "alpha": alpha
        },
        "warnings": []
    }

    return result

# Example usage:
# X = np.array([1, 2, 3, 4, 5])
# y = np.array([2, 3, 5, 7, 11])
# result = splines_fit(X, y)

################################################################################
# outliers
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def outliers_fit(
    X: np.ndarray,
    y: np.ndarray,
    degree: int = 2,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Detect outliers in polynomial regression.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    degree : int, optional
        Degree of the polynomial regression, by default 2.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Normalization function, by default None.
    distance_metric : str, optional
        Distance metric for outlier detection, by default "euclidean".
    solver : str, optional
        Solver for polynomial regression, by default "closed_form".
    regularization : Optional[str], optional
        Regularization type, by default None.
    tol : float, optional
        Tolerance for convergence, by default 1e-4.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function, by default None.
    **kwargs : dict
        Additional solver-specific parameters.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> y = np.array([1, 2, 3, 4, 50])
    >>> result = outliers_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_norm = _apply_normalization(X, normalizer)
    y_norm = _apply_normalization(y.reshape(-1, 1), normalizer).flatten()

    # Prepare polynomial features
    X_poly = _prepare_polynomial_features(X_norm, degree)

    # Fit polynomial regression model
    coefficients = _fit_polynomial_regression(
        X_poly, y_norm, solver=solver,
        regularization=regularization,
        tol=tol, max_iter=max_iter, **kwargs
    )

    # Predict and compute residuals
    y_pred = _predict_polynomial(X_poly, coefficients)
    residuals = y_norm - y_pred

    # Compute distances
    distances = _compute_distances(residuals, metric=distance_metric)

    # Compute metrics
    metrics = _compute_metrics(y_norm, y_pred, custom_metric)

    # Detect outliers
    is_outlier = _detect_outliers(distances, threshold=np.percentile(distances, 95))

    return {
        "result": {
            "coefficients": coefficients,
            "residuals": residuals,
            "distances": distances,
            "is_outlier": is_outlier
        },
        "metrics": metrics,
        "params_used": {
            "degree": degree,
            "normalizer": normalizer.__name__ if normalizer else None,
            "distance_metric": distance_metric,
            "solver": solver,
            "regularization": regularization
        },
        "warnings": _check_warnings(X, y)
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _apply_normalization(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to data."""
    if normalizer is None:
        return data
    return normalizer(data)

def _prepare_polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Prepare polynomial features."""
    n_samples = X.shape[0]
    X_poly = np.ones((n_samples, degree + 1))
    for d in range(1, degree + 1):
        X_poly[:, d] = X[:, 0] ** d
    return X_poly

def _fit_polynomial_regression(
    X: np.ndarray,
    y: np.ndarray,
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    **kwargs
) -> np.ndarray:
    """Fit polynomial regression model."""
    if solver == "closed_form":
        return _closed_form_solution(X, y, regularization)
    elif solver == "gradient_descent":
        return _gradient_descent(X, y, tol=tol, max_iter=max_iter, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _closed_form_solution(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Closed form solution for polynomial regression."""
    XtX = X.T @ X
    if regularization == "l2":
        alpha = kwargs.get("alpha", 1.0)
        XtX += alpha * np.eye(X.shape[1])
    coefficients = np.linalg.pinv(XtX) @ X.T @ y
    return coefficients

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01
) -> np.ndarray:
    """Gradient descent for polynomial regression."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ coefficients - y) / len(y)
        new_coefficients = coefficients - learning_rate * gradient
        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients
    return coefficients

def _predict_polynomial(X: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Predict using polynomial regression."""
    return X @ coefficients

def _compute_distances(residuals: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Compute distances for outlier detection."""
    if metric == "euclidean":
        return np.abs(residuals)
    elif metric == "manhattan":
        return np.sum(np.abs(residuals))
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for polynomial regression."""
    metrics = {}
    if custom_metric is not None:
        metrics["custom"] = custom_metric(y_true, y_pred)
    return metrics

def _detect_outliers(distances: np.ndarray, threshold: float) -> np.ndarray:
    """Detect outliers based on distances."""
    return distances > threshold

def _check_warnings(X: np.ndarray, y: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if X.shape[1] == 1:
        warnings.append("Single feature detected")
    return warnings

################################################################################
# residus
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def residus_fit(
    X: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Calculate residuals for polynomial regression.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    degree : int, optional
        Degree of the polynomial regression, by default 1.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Normalization function, by default None.
    metric : str, optional
        Metric to compute ('mse', 'mae', 'r2'), by default 'mse'.
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent'), by default 'closed_form'.
    regularization : Optional[str], optional
        Regularization type ('none', 'l1', 'l2'), by default None.
    tol : float, optional
        Tolerance for convergence, by default 1e-6.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function, by default None.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]
        Dictionary containing residuals, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.array([[1], [2], [3]])
    >>> y = np.array([1, 4, 9])
    >>> result = residus_fit(X, y, degree=2)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if normalizer is provided
    X_norm = _apply_normalization(X, normalizer)
    y_norm = _apply_normalization(y.reshape(-1, 1), normalizer).flatten()

    # Prepare polynomial features
    X_poly = _prepare_polynomial_features(X_norm, degree)

    # Solve for coefficients
    if solver == 'closed_form':
        coefs = _closed_form_solver(X_poly, y_norm)
    else:
        raise ValueError("Unsupported solver")

    # Apply regularization if specified
    if regularization is not None:
        coefs = _apply_regularization(coefs, X_poly, y_norm, regularization)

    # Calculate residuals
    y_pred = np.dot(X_poly, coefs)
    residuals = y_norm - y_pred

    # Calculate metrics
    metrics = _calculate_metrics(y_norm, y_pred, metric, custom_metric)

    # Prepare output
    result = {
        "result": residuals,
        "metrics": metrics,
        "params_used": {
            "degree": degree,
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "solver": solver,
            "regularization": regularization
        },
        "warnings": []
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

def _apply_normalization(data: np.ndarray, normalizer: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Apply normalization to data if normalizer is provided."""
    if normalizer is not None:
        return normalizer(data)
    return data

def _prepare_polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Prepare polynomial features."""
    n_samples = X.shape[0]
    X_poly = np.ones((n_samples, degree + 1))
    for d in range(degree + 1):
        X_poly[:, d] = X[:, 0] ** d
    return X_poly

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve for coefficients using closed-form solution."""
    X_t = X.T
    coefs = np.linalg.inv(X_t @ X) @ X_t @ y
    return coefs

def _apply_regularization(coefs: np.ndarray, X: np.ndarray, y: np.ndarray, regularization: str) -> np.ndarray:
    """Apply regularization to coefficients."""
    if regularization == 'l2':
        lambda_ = 1.0
        X_t = X.T
        identity = np.eye(X.shape[1])
        coefs = np.linalg.inv(X_t @ X + lambda_ * identity) @ X_t @ y
    return coefs

def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric: str, custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]) -> Dict[str, float]:
    """Calculate metrics based on the specified metric."""
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)
    else:
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
    return metrics

################################################################################
# bias_variance_tradeoff
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def bias_variance_tradeoff_fit(
    X: np.ndarray,
    y: np.ndarray,
    degrees: Union[int, np.ndarray],
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    solver: str = "closed_form",
    n_splits: int = 5,
    random_state: Optional[int] = None
) -> Dict:
    """
    Compute the bias-variance tradeoff for polynomial regression.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    degrees : Union[int, np.ndarray]
        Polynomial degrees to evaluate. If int, evaluates from 1 to degrees.
    normalizer : Optional[Callable]
        Function to normalize features. If None, no normalization is applied.
    metric : Union[str, Callable]
        Metric to evaluate. Can be "mse", "mae", "r2" or a custom callable.
    solver : str
        Solver to use. Options: "closed_form", "gradient_descent".
    n_splits : int
        Number of cross-validation splits.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Dict
        Dictionary containing:
        - "result": List of results for each degree.
        - "metrics": List of metrics for each degree.
        - "params_used": Dictionary of parameters used.
        - "warnings": List of warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 2)
    >>> y = np.random.rand(100)
    >>> result = bias_variance_tradeoff_fit(X, y, degrees=5)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Prepare results dictionary
    result = {
        "result": [],
        "metrics": [],
        "params_used": {
            "degrees": degrees,
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric if isinstance(metric, str) else "custom",
            "solver": solver,
            "n_splits": n_splits
        },
        "warnings": []
    }

    # Determine degrees to evaluate
    if isinstance(degrees, int):
        degrees = np.arange(1, degrees + 1)
    else:
        degrees = np.array(degrees)

    # Normalize data if specified
    X_norm, y_norm = _apply_normalization(X, y, normalizer)

    # Evaluate each degree
    for deg in degrees:
        try:
            # Fit model and compute metrics
            train_mse, test_mse = _evaluate_polynomial_degree(
                X_norm, y_norm, deg, solver, n_splits, random_state
            )

            # Compute bias and variance
            bias = np.mean(train_mse)
            variance = np.var(test_mse)

            # Store results
            result["result"].append({
                "degree": deg,
                "bias": bias,
                "variance": variance
            })

            # Compute and store metric
            if isinstance(metric, str):
                result["metrics"].append(_compute_metric(test_mse, metric))
            else:
                result["metrics"].append(metric(y_norm, _predict_polynomial(X_norm, deg)))

        except Exception as e:
            result["warnings"].append(f"Degree {deg} failed: {str(e)}")

    return result

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

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> tuple:
    """Apply normalization to features and target."""
    X_norm = X.copy()
    y_norm = y.copy()

    if normalizer is not None:
        X_norm = normalizer(X)
        # Normalize target if it's a callable that can handle 1D arrays
        try:
            y_norm = normalizer(y.reshape(-1, 1)).flatten()
        except:
            pass

    return X_norm, y_norm

def _evaluate_polynomial_degree(
    X: np.ndarray,
    y: np.ndarray,
    degree: int,
    solver: str,
    n_splits: int,
    random_state: Optional[int]
) -> tuple:
    """Evaluate polynomial regression for a given degree."""
    # Implement cross-validation and return train/test MSE
    # This is a placeholder - actual implementation would include:
    # 1. Polynomial feature expansion
    # 2. Splitting data into folds
    # 3. Training and evaluating for each fold
    # 4. Returning average train/test MSE

    # For now, return dummy values
    return np.random.rand(n_splits), np.random.rand(n_splits)

def _compute_metric(
    y_true: np.ndarray,
    metric_name: str
) -> float:
    """Compute specified metric."""
    if metric_name == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif metric_name == "mae":
        return np.mean(np.abs(y_true - y_pred))
    elif metric_name == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

def _predict_polynomial(
    X: np.ndarray,
    degree: int
) -> np.ndarray:
    """Predict using polynomial features."""
    # Placeholder for actual implementation
    return np.random.rand(X.shape[0])
