"""
Quantix – Module regression_logistique
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# principe_base
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    normalize: str = "standard",
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, np.ndarray]:
    """
    Validate and preprocess input data.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalize : str or None
        Normalization method: "none", "standard", "minmax", "robust"
    custom_normalize : callable or None
        Custom normalization function

    Returns:
    --------
    dict
        Dictionary containing validated and normalized data
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    if normalize == "none" and custom_normalize is None:
        X_normalized = X
    elif normalize in ["standard", "minmax", "robust"]:
        if normalize == "standard":
            X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        elif normalize == "minmax":
            X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        elif normalize == "robust":
            X_normalized = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    elif custom_normalize is not None:
        X_normalized = custom_normalize(X)
    else:
        raise ValueError("Invalid normalization method")

    return {"X": X_normalized, "y": y}

def _compute_log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute log loss (cross-entropy loss).

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred : np.ndarray
        Predicted probabilities

    Returns:
    --------
    float
        Log loss value
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> np.ndarray:
    """
    Gradient descent solver for logistic regression.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    learning_rate : float
        Learning rate for gradient descent
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for stopping criterion

    Returns:
    --------
    np.ndarray
        Estimated coefficients
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(max_iter):
        linear_model = np.dot(X, weights) + bias
        predictions = 1 / (1 + np.exp(-linear_model))

        gradient_weights = np.dot(X.T, (predictions - y)) / n_samples
        gradient_bias = np.mean(predictions - y)

        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias

        if np.linalg.norm(gradient_weights) < tol and abs(gradient_bias) < tol:
            break

    return np.concatenate([weights, [bias]])

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-4
) -> np.ndarray:
    """
    Newton's method solver for logistic regression.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for stopping criterion

    Returns:
    --------
    np.ndarray
        Estimated coefficients
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features + 1)

    for _ in range(max_iter):
        linear_model = np.dot(X, weights[:-1]) + weights[-1]
        predictions = 1 / (1 + np.exp(-linear_model))

        gradient = np.concatenate([
            np.dot(X.T, (predictions - y)) / n_samples,
            [np.mean(predictions - y)]
        ])

        hessian = np.zeros((n_features + 1, n_features + 1))
        hessian[:-1, :-1] = np.dot(X.T * (predictions * (1 - predictions)), X) / n_samples
        hessian[:-1, -1] = np.mean(predictions * (1 - predictions), axis=0)
        hessian[-1, :-1] = np.mean(predictions * (1 - predictions), axis=0)
        hessian[-1, -1] = np.mean(predictions * (1 - predictions))

        weights -= np.linalg.solve(hessian, gradient)

        if np.linalg.norm(gradient) < tol:
            break

    return weights

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "logloss",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred : np.ndarray
        Predicted probabilities
    metric : str or None
        Metric to compute: "mse", "mae", "r2", "logloss"
    custom_metric : callable or None
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing computed metrics
    """
    metrics = {}

    if metric == "logloss" or (custom_metric is None and metric is None):
        metrics["log_loss"] = _compute_log_loss(y_true, y_pred)

    if custom_metric is not None:
        metrics["custom"] = custom_metric(y_true, y_pred)

    return metrics

def principe_base_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalize: str = "standard",
    solver: str = "gradient_descent",
    metric: str = "logloss",
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, np.ndarray, str]]:
    """
    Fit logistic regression model using specified solver and parameters.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalize : str or None
        Normalization method: "none", "standard", "minmax", "robust"
    solver : str
        Solver to use: "gradient_descent" or "newton"
    metric : str or None
        Metric to compute: "mse", "mae", "r2", "logloss"
    custom_normalize : callable or None
        Custom normalization function
    custom_metric : callable or None
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = principe_base_fit(X, y, normalize="standard", solver="newton")
    """
    # Validate and preprocess inputs
    validated_data = _validate_inputs(X, y, normalize, custom_normalize)
    X_norm = validated_data["X"]
    y_true = validated_data["y"]

    # Choose solver
    if solver == "gradient_descent":
        coefficients = _gradient_descent(X_norm, y_true)
    elif solver == "newton":
        coefficients = _newton_method(X_norm, y_true)
    else:
        raise ValueError("Invalid solver specified")

    # Compute predictions
    weights = coefficients[:-1]
    bias = coefficients[-1]
    linear_model = np.dot(X_norm, weights) + bias
    y_pred = 1 / (1 + np.exp(-linear_model))

    # Compute metrics
    metrics = _compute_metrics(y_true, y_pred, metric, custom_metric)

    # Prepare output
    result = {
        "result": {
            "coefficients": weights,
            "bias": bias
        },
        "metrics": metrics,
        "params_used": {
            "normalize": normalize,
            "solver": solver,
            "metric": metric
        },
        "warnings": []
    }

    return result

################################################################################
# fonction_sigmoide
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

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

def _normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize input data."""
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

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-z))

def _compute_loss(y_true: np.ndarray, y_pred: np.ndarray, metric: str = 'logloss') -> float:
    """Compute loss based on specified metric."""
    if metric == 'logloss':
        return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _gradient_descent(X: np.ndarray, y: np.ndarray,
                      learning_rate: float = 0.01,
                      n_iter: int = 1000,
                      tol: float = 1e-4) -> np.ndarray:
    """Perform gradient descent optimization."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(n_iter):
        z = np.dot(X, weights) + bias
        y_pred = _sigmoid(z)
        gradient_weights = np.dot(X.T, (y_pred - y)) / n_samples
        gradient_bias = np.mean(y_pred - y)

        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias

        if np.linalg.norm(gradient_weights) < tol and abs(gradient_bias) < tol:
            break

    return np.append(weights, bias)

def fonction_sigmoide_fit(X: np.ndarray,
                         y: np.ndarray,
                         solver: str = 'gradient_descent',
                         normalization: str = 'standard',
                         metric: Union[str, Callable] = 'logloss',
                         **kwargs) -> Dict:
    """
    Fit a logistic regression model using the sigmoid function.

    Parameters:
    - X: Input features (2D array)
    - y: Target values (1D array)
    - solver: Optimization method ('gradient_descent', etc.)
    - normalization: Data normalization method
    - metric: Loss function to optimize
    - **kwargs: Additional solver-specific parameters

    Returns:
    - Dictionary containing results, metrics, and parameters used
    """
    _validate_inputs(X, y)
    X_normalized = _normalize_data(X, normalization)

    if solver == 'gradient_descent':
        params = _gradient_descent(X_normalized, y, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    weights = params[:-1]
    bias = params[-1]

    z = np.dot(X_normalized, weights) + bias
    y_pred = _sigmoid(z)

    if isinstance(metric, str):
        loss = _compute_loss(y, y_pred, metric)
    else:
        loss = metric(y, y_pred)

    return {
        'result': {'weights': weights, 'bias': bias},
        'metrics': {'loss': loss},
        'params_used': {
            'solver': solver,
            'normalization': normalization,
            'metric': metric
        },
        'warnings': []
    }

################################################################################
# coefficient_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def coefficient_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    penalty: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000,
    metric: str = 'logloss',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit logistic regression coefficients using specified solver and parameters.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalizer : Optional[Callable]
        Function to normalize features. If None, no normalization.
    solver : str
        Solver method: 'closed_form', 'gradient_descent', 'newton'
    regularization : Optional[str]
        Regularization type: None, 'l1', 'l2', 'elasticnet'
    penalty : float
        Regularization strength
    tol : float
        Tolerance for stopping criteria
    max_iter : int
        Maximum number of iterations
    metric : str
        Primary evaluation metric: 'logloss', 'mse', 'mae'
    custom_metric : Optional[Callable]
        Custom metric function if needed
    verbose : bool
        Whether to print progress information

    Returns
    -------
    Dict containing:
        - 'result': fitted coefficients
        - 'metrics': computed metrics
        - 'params_used': parameters used during fitting
        - 'warnings': any warnings encountered

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = coefficient_regression_fit(X, y, solver='gradient_descent')
    """
    # Input validation
    _validate_inputs(X, y)

    # Normalize features if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize parameters based on solver choice
    params = {
        'solver': solver,
        'regularization': regularization,
        'penalty': penalty,
        'tol': tol,
        'max_iter': max_iter
    }

    # Solve for coefficients
    if solver == 'closed_form':
        coef = _closed_form_solution(X_normalized, y)
    elif solver == 'gradient_descent':
        coef = _gradient_descent(X_normalized, y, penalty=penalty,
                               tol=tol, max_iter=max_iter)
    elif solver == 'newton':
        coef = _newton_method(X_normalized, y, penalty=penalty,
                            tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, coef,
                             metric=metric,
                             custom_metric=custom_metric)

    return {
        'result': coef,
        'metrics': metrics,
        'params_used': params,
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

def _apply_normalization(
    X: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply specified normalization to features."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _closed_form_solution(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute closed-form solution for logistic regression."""
    # This is a placeholder - actual implementation would use iterative methods
    # as closed-form solution doesn't exist for logistic regression
    raise NotImplementedError("Closed form solution not available for logistic regression")

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    penalty: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Gradient descent solver for logistic regression."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features + 1)  # Include intercept

    for _ in range(max_iter):
        # Compute predictions
        linear_model = np.dot(X, coef[1:]) + coef[0]
        predictions = _sigmoid(linear_model)

        # Compute gradient
        gradient = np.zeros_like(coef)
        gradient[0] = np.mean(predictions - y)  # Intercept
        gradient[1:] = (np.dot(X.T, predictions - y) +
                       penalty * coef[1:]) / n_samples

        # Update coefficients
        coef -= gradient * 0.1  # Learning rate of 0.1

        if np.linalg.norm(gradient) < tol:
            break

    return coef

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    *,
    penalty: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Newton's method solver for logistic regression."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features + 1)  # Include intercept

    for _ in range(max_iter):
        # Compute predictions
        linear_model = np.dot(X, coef[1:]) + coef[0]
        predictions = _sigmoid(linear_model)

        # Compute gradient and Hessian
        gradient = np.zeros_like(coef)
        hessian = np.zeros((n_features + 1, n_features + 1))

        gradient[0] = np.mean(predictions - y)
        gradient[1:] = (np.dot(X.T, predictions - y) +
                       penalty * coef[1:]) / n_samples

        hessian[0, 0] = np.mean(predictions * (1 - predictions))
        hessian[0, 1:] = np.mean(X.T * (predictions * (1 - predictions)), axis=1)
        hessian[1:, 0] = np.mean(X * (predictions * (1 - predictions)), axis=0)
        hessian[1:, 1:] = np.dot(X.T * (predictions * (1 - predictions)), X) / n_samples
        hessian[1:, 1:] += penalty * np.eye(n_features)

        # Update coefficients
        coef -= np.linalg.solve(hessian, gradient)

        if np.linalg.norm(gradient) < tol:
            break

    return coef

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    *,
    metric: str = 'logloss',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute specified metrics for logistic regression."""
    linear_model = np.dot(X, coef[1:]) + coef[0]
    predictions = _sigmoid(linear_model)

    metrics = {}

    if metric == 'logloss':
        metrics['logloss'] = _log_loss(y, predictions)
    elif metric == 'mse':
        metrics['mse'] = np.mean((y - predictions) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y - predictions))
    elif metric == 'r2':
        metrics['r2'] = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y, predictions)

    return metrics

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-z))

def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Compute log loss."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

################################################################################
# intercept
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def intercept_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalization: str = "none",
    metric: Union[str, Callable] = "logloss",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a logistic regression model and compute the intercept.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalization : str, optional
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : str or callable, optional
        Metric to evaluate: "mse", "mae", "r2", "logloss", or custom callable.
    solver : str, optional
        Solver method: "closed_form", "gradient_descent", "newton", or "coordinate_descent".
    regularization : str, optional
        Regularization type: "none", "l1", "l2", or "elasticnet".
    tol : float, optional
        Tolerance for stopping criteria.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function.
    custom_distance : callable, optional
        Custom distance function.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - "result": Estimated intercept.
        - "metrics": Computed metrics.
        - "params_used": Parameters used during fitting.
        - "warnings": Any warnings encountered.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> y = np.array([0, 1])
    >>> result = intercept_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_normalized = _normalize_data(X, method=normalization)

    # Choose solver and compute intercept
    if solver == "closed_form":
        intercept = _intercept_closed_form(X_normalized, y)
    elif solver == "gradient_descent":
        intercept = _intercept_gradient_descent(X_normalized, y, tol=tol, max_iter=max_iter)
    elif solver == "newton":
        intercept = _intercept_newton(X_normalized, y, tol=tol, max_iter=max_iter)
    elif solver == "coordinate_descent":
        intercept = _intercept_coordinate_descent(X_normalized, y, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, intercept, metric=metric, custom_metric=custom_metric)

    # Prepare output
    result = {
        "result": intercept,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
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

def _normalize_data(X: np.ndarray, method: str = "none") -> np.ndarray:
    """Normalize the feature matrix."""
    if method == "standard":
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == "minmax":
        X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == "robust":
        X_normalized = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        X_normalized = X.copy()
    return X_normalized

def _intercept_closed_form(X: np.ndarray, y: np.ndarray) -> float:
    """Compute intercept using closed-form solution."""
    n_samples = X.shape[0]
    X_with_intercept = np.hstack([np.ones((n_samples, 1)), X])
    coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
    return coefficients[0]

def _intercept_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> float:
    """Compute intercept using gradient descent."""
    n_samples = X.shape[0]
    intercept = 0.0
    learning_rate = 0.01

    for _ in range(max_iter):
        predictions = 1 / (1 + np.exp(-(intercept + X @ np.zeros(X.shape[1]))))
        gradient = np.mean(y - predictions)
        intercept -= learning_rate * gradient

        if abs(gradient) < tol:
            break

    return intercept

def _intercept_newton(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> float:
    """Compute intercept using Newton's method."""
    n_samples = X.shape[0]
    intercept = 0.0

    for _ in range(max_iter):
        predictions = 1 / (1 + np.exp(-(intercept + X @ np.zeros(X.shape[1]))))
        gradient = np.mean(y - predictions)
        hessian = np.mean(predictions * (1 - predictions))
        intercept -= gradient / hessian

        if abs(gradient) < tol:
            break

    return intercept

def _intercept_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> float:
    """Compute intercept using coordinate descent."""
    n_samples = X.shape[0]
    intercept = 0.0

    for _ in range(max_iter):
        predictions = 1 / (1 + np.exp(-(intercept + X @ np.zeros(X.shape[1]))))
        gradient = np.mean(y - predictions)
        intercept -= learning_rate * gradient

        if abs(gradient) < tol:
            break

    return intercept

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    intercept: float,
    *,
    metric: Union[str, Callable] = "logloss",
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Compute metrics for the logistic regression model."""
    predictions = 1 / (1 + np.exp(-(intercept + X @ np.zeros(X.shape[1]))))

    metrics = {}

    if metric == "logloss" or custom_metric is None:
        log_loss = -np.mean(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))
        metrics["logloss"] = log_loss

    if custom_metric is not None:
        metrics["custom_metric"] = custom_metric(y, predictions)

    return metrics

################################################################################
# log_odds
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def log_odds_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    solver: str = 'gradient_descent',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'logloss',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    penalty: float = 1.0,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit logistic regression model and compute log-odds.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,) with values in {0, 1}.
    normalizer : Optional[Callable]
        Function to normalize features. If None, no normalization.
    solver : str
        Solver type: 'gradient_descent', 'newton', or 'coordinate_descent'.
    metric : Union[str, Callable]
        Metric to evaluate: 'logloss', 'mse', or custom callable.
    regularization : Optional[str]
        Regularization type: 'l1', 'l2', or None.
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    learning_rate : float
        Learning rate for gradient descent.
    penalty : float
        Regularization strength.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Estimated coefficients.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> y = np.array([0, 1])
    >>> result = log_odds_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize parameters
    n_features = X_normalized.shape[1]
    coefficients = np.zeros(n_features)
    intercept = 0.0

    # Choose solver
    if solver == 'gradient_descent':
        coefficients, intercept = _gradient_descent(
            X_normalized, y, coefficients, intercept,
            learning_rate=learning_rate, max_iter=max_iter,
            tol=tol, regularization=regularization,
            penalty=penalty
        )
    elif solver == 'newton':
        coefficients, intercept = _newton_method(
            X_normalized, y, coefficients, intercept,
            max_iter=max_iter, tol=tol
        )
    elif solver == 'coordinate_descent':
        coefficients, intercept = _coordinate_descent(
            X_normalized, y, coefficients, intercept,
            max_iter=max_iter, tol=tol
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(
        X_normalized, y,
        coefficients, intercept,
        metric=metric
    )

    # Prepare output
    result = {
        'result': np.concatenate([[intercept], coefficients]),
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'solver': solver,
            'metric': metric if isinstance(metric, str) else 'custom',
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter,
            'learning_rate': learning_rate,
            'penalty': penalty
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only 0 and 1 values")

def _apply_normalization(
    X: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply feature normalization if specified."""
    if normalizer is None:
        return X
    return normalizer(X)

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    *,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    regularization: Optional[str] = None,
    penalty: float = 1.0
) -> tuple[np.ndarray, float]:
    """Gradient descent solver for logistic regression."""
    n_samples = X.shape[0]
    prev_loss = float('inf')

    for _ in range(max_iter):
        # Compute predictions
        linear_model = X @ coefficients + intercept
        probabilities = _sigmoid(linear_model)

        # Compute gradients
        gradient_coef = (X.T @ (probabilities - y)) / n_samples
        gradient_intercept = np.mean(probabilities - y)

        # Apply regularization if specified
        if regularization == 'l1':
            gradient_coef += penalty * np.sign(coefficients)
        elif regularization == 'l2':
            gradient_coef += 2 * penalty * coefficients

        # Update parameters
        coefficients -= learning_rate * gradient_coef
        intercept -= learning_rate * gradient_intercept

        # Compute current loss
        current_loss = _logloss(y, probabilities)

        # Check convergence
        if abs(prev_loss - current_loss) < tol:
            break

        prev_loss = current_loss

    return coefficients, intercept

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    *,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> tuple[np.ndarray, float]:
    """Newton's method solver for logistic regression."""
    n_samples = X.shape[0]

    for _ in range(max_iter):
        # Compute predictions
        linear_model = X @ coefficients + intercept
        probabilities = _sigmoid(linear_model)

        # Compute Hessian and gradient
        W = np.diag(probabilities * (1 - probabilities))
        hessian = X.T @ W @ X / n_samples
        gradient_coef = (X.T @ (probabilities - y)) / n_samples
        gradient_intercept = np.mean(probabilities - y)

        # Update parameters
        delta_coef = np.linalg.solve(hessian, -gradient_coef)
        coefficients += delta_coef
        intercept -= gradient_intercept / hessian.shape[0]

    return coefficients, intercept

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    *,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> tuple[np.ndarray, float]:
    """Coordinate descent solver for logistic regression."""
    n_samples = X.shape[0]
    prev_loss = float('inf')

    for _ in range(max_iter):
        for j in range(X.shape[1]):
            # Compute predictions without current feature
            linear_model = intercept + np.dot(X, coefficients) - X[:, j] * coefficients[j]
            probabilities = _sigmoid(linear_model)

            # Compute gradient for current feature
            gradient_j = np.dot(X[:, j], probabilities - y) / n_samples

            # Update current coefficient
            coefficients[j] -= gradient_j

        # Compute intercept update
        linear_model = X @ coefficients + intercept
        probabilities = _sigmoid(linear_model)
        intercept -= np.mean(probabilities - y)

        # Compute current loss
        current_loss = _logloss(y, probabilities)

        # Check convergence
        if abs(prev_loss - current_loss) < tol:
            break

        prev_loss = current_loss

    return coefficients, intercept

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-z))

def _logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'logloss'
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    linear_model = X @ coefficients + intercept
    probabilities = _sigmoid(linear_model)

    metrics_dict = {}

    if isinstance(metric, str):
        if metric == 'logloss':
            metrics_dict['logloss'] = _logloss(y, probabilities)
        elif metric == 'mse':
            metrics_dict['mse'] = np.mean((y - probabilities) ** 2)
        elif metric == 'r2':
            ss_res = np.sum((y - probabilities) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            metrics_dict['r2'] = 1 - ss_res / ss_tot
    else:
        metrics_dict['custom_metric'] = metric(y, probabilities)

    return metrics_dict

################################################################################
# probabilite_prediction
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def probabilite_prediction_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalisation: str = 'standard',
    solver: str = 'gradient_descent',
    metric: Union[str, Callable] = 'logloss',
    regularisation: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a logistic regression model and compute prediction probabilities.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,) with values in {0, 1}
    normalisation : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'
    solver : str
        Solver method: 'gradient_descent', 'newton', or 'coordinate_descent'
    metric : str or callable
        Evaluation metric: 'mse', 'mae', 'r2', 'logloss', or custom callable
    regularisation : str, optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'
    tol : float
        Tolerance for stopping criteria
    max_iter : int
        Maximum number of iterations
    learning_rate : float
        Learning rate for gradient descent
    custom_metric : callable, optional
        Custom metric function
    custom_distance : callable, optional
        Custom distance function

    Returns:
    --------
    Dict containing:
        - 'result': Predicted probabilities
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in fitting
        - 'warnings': Any warnings generated

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = probabilite_prediction_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalisation(X, normalisation)

    # Initialize parameters
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0

    # Choose solver
    if solver == 'gradient_descent':
        weights, bias = _gradient_descent(
            X_normalized, y, weights, bias,
            tol=tol, max_iter=max_iter,
            learning_rate=learning_rate,
            regularisation=regularisation
        )
    elif solver == 'newton':
        weights, bias = _newton_method(
            X_normalized, y, weights, bias,
            tol=tol, max_iter=max_iter
        )
    elif solver == 'coordinate_descent':
        weights, bias = _coordinate_descent(
            X_normalized, y, weights, bias,
            tol=tol, max_iter=max_iter
        )

    # Compute predicted probabilities
    probabilities = _sigmoid(np.dot(X_normalized, weights) + bias)

    # Compute metrics
    metrics = _compute_metrics(y, probabilities, metric, custom_metric)

    return {
        'result': probabilities,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'solver': solver,
            'metric': metric,
            'regularisation': regularisation,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': _check_warnings(y, probabilities)
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
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only 0 and 1 values")

def _apply_normalisation(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to features."""
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

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-z))

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    regularisation: Optional[str] = None
) -> tuple:
    """Gradient descent optimization for logistic regression."""
    n_samples, n_features = X.shape
    for _ in range(max_iter):
        # Compute predictions and error
        linear_model = np.dot(X, weights) + bias
        probabilities = _sigmoid(linear_model)
        error = probabilities - y

        # Compute gradients
        grad_weights = np.dot(X.T, error) / n_samples
        grad_bias = np.mean(error)

        # Apply regularization if needed
        if regularisation == 'l1':
            grad_weights += np.sign(weights) * 0.1
        elif regularisation == 'l2':
            grad_weights += weights * 0.1

        # Update parameters
        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias

        # Check convergence
        if np.linalg.norm(grad_weights) < tol and abs(grad_bias) < tol:
            break

    return weights, bias

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    tol: float = 1e-4,
    max_iter: int = 100
) -> tuple:
    """Newton's method optimization for logistic regression."""
    n_samples, n_features = X.shape
    for _ in range(max_iter):
        # Compute predictions and error
        linear_model = np.dot(X, weights) + bias
        probabilities = _sigmoid(linear_model)
        error = probabilities - y

        # Compute Hessian
        hessian_diag = probabilities * (1 - probabilities)
        hessian = np.dot(X.T * hessian_diag, X) / n_samples

        # Compute gradients
        grad_weights = np.dot(X.T, error) / n_samples
        grad_bias = np.mean(error)

        # Update parameters using Newton's method
        delta_weights = np.linalg.solve(hessian, -grad_weights)
        weights += delta_weights
        bias -= grad_bias / n_samples

        # Check convergence
        if np.linalg.norm(delta_weights) < tol and abs(grad_bias) / n_samples < tol:
            break

    return weights, bias

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    tol: float = 1e-4,
    max_iter: int = 100
) -> tuple:
    """Coordinate descent optimization for logistic regression."""
    n_samples, n_features = X.shape
    for _ in range(max_iter):
        converged = True

        # Update each weight one at a time
        for j in range(n_features):
            # Save current weight
            w_old = weights[j]

            # Compute predictions without current feature
            linear_model = np.dot(X[:, :j], weights[:j]) + np.dot(X[:, j+1:], weights[j+1:]) + bias
            probabilities = _sigmoid(linear_model)

            # Compute gradient for current feature
            grad_j = np.dot(X[:, j], probabilities - y) / n_samples

            # Update weight
            weights[j] -= grad_j

            # Check convergence for this feature
            if abs(weights[j] - w_old) > tol:
                converged = False

        # Update bias
        linear_model = np.dot(X, weights) + bias
        probabilities = _sigmoid(linear_model)
        grad_bias = np.mean(probabilities - y)
        bias -= grad_bias

        if converged:
            break

    return weights, bias

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute specified metrics."""
    metrics = {}

    if metric == 'logloss':
        metrics['logloss'] = _log_loss(y_true, y_pred)
    elif metric == 'mse':
        metrics['mse'] = _mean_squared_error(y_true, y_pred)
    elif metric == 'mae':
        metrics['mae'] = _mean_absolute_error(y_true, y_pred)
    elif metric == 'r2':
        metrics['r2'] = _r_squared(y_true, y_pred)
    elif callable(metric):
        metrics['custom'] = metric(y_true, y_pred)

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)

    return metrics

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

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
    return 1 - (ss_res / ss_tot)

def _check_warnings(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []

    if not np.all(np.isfinite(y_pred)):
        warnings.append("Predicted probabilities contain NaN or infinite values")

    if np.any(y_pred < 0) or np.any(y_pred > 1):
        warnings.append("Predicted probabilities outside [0, 1] range")

    return warnings

################################################################################
# fonction_perte_logistique
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def fonction_perte_logistique_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalisation: str = 'standard',
    metric: Union[str, Callable] = 'logloss',
    solver: str = 'gradient_descent',
    regularisation: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict, str]]:
    """
    Calcule la fonction de perte pour une régression logistique avec options paramétrables.

    Parameters
    ----------
    X : np.ndarray
        Matrice des caractéristiques (n_samples, n_features)
    y : np.ndarray
        Vecteur des labels (n_samples,)
    normalisation : str, optional
        Type de normalisation ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Métrique à utiliser ('mse', 'mae', 'r2', 'logloss') ou fonction personnalisée
    solver : str, optional
        Solveur à utiliser ('closed_form', 'gradient_descent', 'newton')
    regularisation : str, optional
        Type de régularisation ('none', 'l1', 'l2', 'elasticnet')
    tol : float, optional
        Tolérance pour la convergence
    max_iter : int, optional
        Nombre maximal d'itérations
    learning_rate : float, optional
        Taux d'apprentissage pour les solveurs itératifs
    custom_metric : callable, optional
        Fonction de métrique personnalisée
    custom_distance : callable, optional
        Fonction de distance personnalisée

    Returns
    -------
    dict
        Dictionnaire contenant les résultats, métriques, paramètres utilisés et avertissements

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> y = np.array([0, 1])
    >>> result = fonction_perte_logistique_fit(X, y)
    """
    # Validation des entrées
    _validate_inputs(X, y)

    # Normalisation
    X_normalized = _apply_normalisation(X, normalisation)

    # Initialisation des paramètres
    params = _initialize_parameters(X_normalized.shape[1])

    # Choix du solveur
    if solver == 'closed_form':
        params = _solve_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(
            X_normalized, y,
            tol=tol,
            max_iter=max_iter,
            learning_rate=learning_rate
        )
    elif solver == 'newton':
        params = _solve_newton(X_normalized, y)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Application de la régularisation
    if regularisation:
        params = _apply_regularisation(params, regularisation)

    # Calcul des métriques
    metrics = _compute_metrics(
        X_normalized, y,
        params,
        metric=metric,
        custom_metric=custom_metric
    )

    # Retour des résultats
    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'metric': metric,
            'solver': solver,
            'regularisation': regularisation,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Valide les entrées pour la régression logistique."""
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

def _apply_normalisation(X: np.ndarray, normalisation: str) -> np.ndarray:
    """Applique la normalisation choisie."""
    if normalisation == 'none':
        return X
    elif normalisation == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalisation == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalisation == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalisation: {normalisation}")

def _initialize_parameters(n_features: int) -> np.ndarray:
    """Initialise les paramètres aléatoirement."""
    return np.random.randn(n_features + 1)  # +1 pour le biais

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Résolution par forme fermée (méthode des moindres carrés)."""
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    return np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01
) -> np.ndarray:
    """Résolution par descente de gradient."""
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    params = _initialize_parameters(X_with_bias.shape[1])

    for _ in range(max_iter):
        gradient = _compute_gradient(X_with_bias, y, params)
        params -= learning_rate * gradient

        if np.linalg.norm(gradient) < tol:
            break

    return params

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Résolution par méthode de Newton."""
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    params = _initialize_parameters(X_with_bias.shape[1])

    for _ in range(100):
        gradient = _compute_gradient(X_with_bias, y, params)
        hessian = _compute_hessian(X_with_bias, y, params)
        params -= np.linalg.inv(hessian) @ gradient

    return params

def _compute_gradient(X: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Calcule le gradient de la fonction de coût."""
    m = X.shape[0]
    predictions = _sigmoid(X @ params)
    error = predictions - y
    return (X.T @ error) / m

def _compute_hessian(X: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Calcule la hessienne de la fonction de coût."""
    m = X.shape[0]
    predictions = _sigmoid(X @ params)
    diag = predictions * (1 - predictions)
    return (X.T * diag) @ X / m

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Fonction sigmoïde."""
    return 1 / (1 + np.exp(-z))

def _apply_regularisation(params: np.ndarray, regularisation: str) -> np.ndarray:
    """Applique la régularisation choisie."""
    if regularisation == 'l1':
        return np.sign(params) * np.maximum(np.abs(params) - 0.1, 0)
    elif regularisation == 'l2':
        return params / (1 + 0.1 * np.abs(params))
    elif regularisation == 'elasticnet':
        return _apply_regularisation(_apply_regularisation(params, 'l1'), 'l2')
    else:
        return params

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    *,
    metric: Union[str, Callable] = 'logloss',
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Calcule les métriques demandées."""
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    predictions = _sigmoid(X_with_bias @ params)

    metrics_dict = {}

    if metric == 'logloss' or custom_metric is None:
        metrics_dict['logloss'] = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

    if metric == 'mse':
        metrics_dict['mse'] = np.mean((y - predictions) ** 2)

    if metric == 'mae':
        metrics_dict['mae'] = np.mean(np.abs(y - predictions))

    if metric == 'r2':
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics_dict['r2'] = 1 - (ss_res / ss_tot)

    if custom_metric is not None:
        metrics_dict['custom'] = custom_metric(y, predictions)

    return metrics_dict

################################################################################
# gradient_descent
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

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

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-z))

def compute_loss(y_true: np.ndarray, y_pred: np.ndarray, metric: str = 'logloss') -> float:
    """Compute loss using specified metric."""
    if metric == 'logloss':
        return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def gradient_descent_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                          learning_rate: float, penalty: str = 'none',
                          l1_ratio: float = 0.5) -> np.ndarray:
    """Perform one gradient descent step."""
    m = X.shape[0]
    y_pred = sigmoid(X @ weights)
    error = y_pred - y

    gradient = X.T @ error / m

    if penalty == 'l1':
        gradient += l1_ratio * np.sign(weights)
    elif penalty == 'l2':
        gradient += 2 * l1_ratio * weights
    elif penalty == 'elasticnet':
        gradient += l1_ratio * (np.sign(weights) + 2 * weights)

    return weights - learning_rate * gradient

def check_convergence(prev_loss: float, current_loss: float,
                      tol: float = 1e-4) -> bool:
    """Check if convergence criteria is met."""
    return np.abs(prev_loss - current_loss) < tol

def gradient_descent_fit(X: np.ndarray, y: np.ndarray,
                         learning_rate: float = 0.01, max_iter: int = 1000,
                         tol: float = 1e-4, penalty: str = 'none',
                         l1_ratio: float = 0.5,
                         normalize_method: str = 'standard',
                         metric: str = 'logloss') -> Dict[str, Any]:
    """
    Perform logistic regression using gradient descent.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - learning_rate: Learning rate for gradient descent
    - max_iter: Maximum number of iterations
    - tol: Tolerance for stopping criteria
    - penalty: Type of regularization ('none', 'l1', 'l2', 'elasticnet')
    - l1_ratio: Regularization strength
    - normalize_method: Data normalization method
    - metric: Loss metric to optimize

    Returns:
    Dictionary containing results, metrics, parameters used and warnings
    """
    validate_inputs(X, y)
    X_normalized = normalize_data(X, method=normalize_method)

    n_features = X_normalized.shape[1]
    weights = np.zeros(n_features)
    prev_loss = float('inf')

    for i in range(max_iter):
        weights = gradient_descent_step(X_normalized, y, weights,
                                      learning_rate, penalty, l1_ratio)
        y_pred = sigmoid(X_normalized @ weights)
        current_loss = compute_loss(y, y_pred, metric)

        if check_convergence(prev_loss, current_loss, tol):
            break

        prev_loss = current_loss

    y_pred_final = sigmoid(X_normalized @ weights)
    final_metric = compute_loss(y, y_pred_final, metric)

    return {
        'result': weights,
        'metrics': {metric: final_metric},
        'params_used': {
            'learning_rate': learning_rate,
            'max_iter': i + 1,
            'tol': tol,
            'penalty': penalty,
            'l1_ratio': l1_ratio,
            'normalize_method': normalize_method
        },
        'warnings': []
    }

# Example usage:
"""
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 1])
result = gradient_descent_fit(X, y)
"""

################################################################################
# regularisation_l1
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regularisation_l1_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalisation: str = 'standard',
    metric: Union[str, Callable] = 'logloss',
    solver: str = 'coordinate_descent',
    penalty: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit a logistic regression model with L1 regularization.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalisation : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2', 'logloss') or custom callable.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    penalty : float, optional
        Regularization strength.
    tol : float, optional
        Tolerance for stopping criteria.
    max_iter : int, optional
        Maximum number of iterations.
    random_state : int or None, optional
        Random seed for reproducibility.
    custom_metric : callable, optional
        Custom metric function.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> result = regularisation_l1_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _normalize_data(X, method=normalisation)

    # Initialize parameters
    n_features = X_normalized.shape[1]
    coef_ = np.zeros(n_features)
    intercept_ = 0.0

    # Choose solver
    if solver == 'coordinate_descent':
        coef_, intercept_ = _coordinate_descent(
            X_normalized, y, penalty=penalty, tol=tol,
            max_iter=max_iter, random_state=random_state
        )
    else:
        raise ValueError(f"Solver {solver} not implemented for L1 regularization.")

    # Compute metrics
    metrics = _compute_metrics(
        X_normalized, y, coef_, intercept_,
        metric=metric, custom_metric=custom_metric
    )

    # Prepare output
    result = {
        'result': {'coef': coef_, 'intercept': intercept_},
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'metric': metric,
            'solver': solver,
            'penalty': penalty,
            'tol': tol,
            'max_iter': max_iter
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
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X and y must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("X and y must not contain infinite values.")

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
        raise ValueError(f"Normalization method {method} not recognized.")

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    penalty: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> tuple:
    """Coordinate descent solver for L1 regularized logistic regression."""
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    coef_ = np.zeros(n_features)
    intercept_ = 0.0

    for _ in range(max_iter):
        old_coef = coef_.copy()

        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - _sigmoid(X @ coef_ + intercept_)
            correlation = X_j.T @ residuals

            if correlation < -penalty / 2:
                coef_[j] = 0
            else:
                coef_[j] = (correlation + penalty / 2) / (X_j.T @ X_j + 1e-8)

        intercept_ = np.mean(y - _sigmoid(X @ coef_))

        if np.linalg.norm(coef_ - old_coef) < tol:
            break

    return coef_, intercept_

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-z))

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    *,
    metric: Union[str, Callable] = 'logloss',
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute metrics for logistic regression."""
    y_pred_proba = _sigmoid(X @ coef + intercept)
    metrics = {}

    if metric == 'logloss':
        metrics['logloss'] = -np.mean(y * np.log(y_pred_proba + 1e-8) +
                                     (1 - y) * np.log(1 - y_pred_proba + 1e-8))
    elif metric == 'mse':
        metrics['mse'] = np.mean((y - y_pred_proba) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y - y_pred_proba))
    elif metric == 'r2':
        ss_res = np.sum((y - y_pred_proba) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics['r2'] = 1 - ss_res / (ss_tot + 1e-8)
    elif callable(metric):
        metrics['custom'] = metric(y, y_pred_proba)
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y, y_pred_proba)

    return metrics

################################################################################
# regularisation_l2
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regularisation_l2_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalisation: str = 'standard',
    metric: Union[str, Callable] = 'logloss',
    solver: str = 'closed_form',
    penalty: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    verbose: bool = False
) -> Dict:
    """
    Fit a logistic regression model with L2 regularization.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalisation : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : str or callable, optional
        Metric to evaluate: 'mse', 'mae', 'r2', 'logloss', or custom callable.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    penalty : float, optional
        Regularization strength.
    tol : float, optional
        Tolerance for stopping criteria.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function.
    verbose : bool, optional
        Whether to print progress.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> result = regularisation_l2_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _normalize_data(X, method=normalisation)

    # Add intercept term
    X_normalized = np.column_stack([np.ones(X_normalized.shape[0]), X_normalized])

    # Initialize parameters
    n_features = X_normalized.shape[1]
    beta = np.zeros(n_features)

    # Choose solver
    if solver == 'closed_form':
        beta = _closed_form_solver(X_normalized, y, penalty)
    elif solver == 'gradient_descent':
        beta = _gradient_descent_solver(X_normalized, y, penalty, tol, max_iter, verbose)
    elif solver == 'newton':
        beta = _newton_solver(X_normalized, y, penalty, tol, max_iter, verbose)
    elif solver == 'coordinate_descent':
        beta = _coordinate_descent_solver(X_normalized, y, penalty, tol, max_iter, verbose)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, beta, metric, custom_metric)

    # Prepare results
    result = {
        'result': beta,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'metric': metric,
            'solver': solver,
            'penalty': penalty,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
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

def _closed_form_solver(X: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    """Solve logistic regression with L2 regularization using closed form."""
    n_samples, n_features = X.shape
    y = y.astype(float)
    y[y == 0] = -1

    # Compute the closed form solution
    XtX = X.T @ X
    Xty = X.T @ y

    # Add L2 penalty
    identity = np.eye(n_features)
    identity[0, 0] = 0  # No penalty on intercept

    beta = np.linalg.inv(XtX + penalty * identity) @ Xty
    return beta

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    penalty: float,
    tol: float,
    max_iter: int,
    verbose: bool
) -> np.ndarray:
    """Solve logistic regression with L2 regularization using gradient descent."""
    n_samples, n_features = X.shape
    y = y.astype(float)
    y[y == 0] = -1

    beta = np.zeros(n_features)
    learning_rate = 1.0 / (penalty + 1e-8)

    for i in range(max_iter):
        gradient = _logistic_gradient(X, y, beta)
        l2_penalty_grad = penalty * np.concatenate([[0], beta[1:]])

        beta_new = beta - learning_rate * (gradient + l2_penalty_grad)

        if np.linalg.norm(beta_new - beta) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break

        beta = beta_new

    return beta

def _logistic_gradient(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute the gradient of the logistic loss."""
    m = X.shape[0]
    y_pred = _sigmoid(X @ beta)
    gradient = (X.T @ (y_pred - y)) / m
    return gradient

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute metrics for the model."""
    y_pred = _sigmoid(X @ beta)
    metrics_dict = {}

    if metric == 'mse':
        metrics_dict['mse'] = np.mean((y - y_pred) ** 2)
    elif metric == 'mae':
        metrics_dict['mae'] = np.mean(np.abs(y - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics_dict['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == 'logloss':
        metrics_dict['logloss'] = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
    elif callable(metric):
        metrics_dict['custom'] = metric(y, y_pred)
    if custom_metric is not None:
        metrics_dict['custom_metric'] = custom_metric(y, y_pred)

    return metrics_dict

def _newton_solver(
    X: np.ndarray,
    y: np.ndarray,
    penalty: float,
    tol: float,
    max_iter: int,
    verbose: bool
) -> np.ndarray:
    """Solve logistic regression with L2 regularization using Newton's method."""
    n_samples, n_features = X.shape
    y = y.astype(float)
    y[y == 0] = -1

    beta = np.zeros(n_features)

    for i in range(max_iter):
        gradient = _logistic_gradient(X, y, beta)
        hessian = _logistic_hessian(X, y_pred)

        # Add L2 penalty
        hessian_diag = np.diag(hessian)
        hessian_diag[1:] += penalty
        hessian = np.diag(hessian_diag)

        beta_new = beta - np.linalg.inv(hessian) @ gradient

        if np.linalg.norm(beta_new - beta) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break

        beta = beta_new

    return beta

def _logistic_hessian(X: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute the Hessian of the logistic loss."""
    m = X.shape[0]
    W = np.diag(y_pred * (1 - y_pred))
    hessian = X.T @ W @ X / m
    return hessian

def _coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    penalty: float,
    tol: float,
    max_iter: int,
    verbose: bool
) -> np.ndarray:
    """Solve logistic regression with L2 regularization using coordinate descent."""
    n_samples, n_features = X.shape
    y = y.astype(float)
    y[y == 0] = -1

    beta = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            beta_j_old = beta[j]

            # Compute residual
            residual = y - _sigmoid(X @ beta)

            # Update beta_j
            numerator = X_j.T @ residual + penalty * beta_j_old
            denominator = X_j.T @ (X_j * _sigmoid(X @ beta) * (1 - _sigmoid(X @ beta))) + penalty

            if denominator != 0:
                beta[j] = numerator / denominator
            else:
                beta[j] = 0

        if np.linalg.norm(beta - np.array([beta_j_old if i == j else beta[i] for i in range(n_features)])) < tol:
            if verbose:
                print(f"Converged at iteration {_}")
            break

    return beta

################################################################################
# matrice_confusion
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or Inf values")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or Inf values")

def _compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                             normalize: Optional[str] = None) -> np.ndarray:
    """Compute confusion matrix with optional normalization."""
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(unique_classes)
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    for i, true_class in enumerate(unique_classes):
        for j, pred_class in enumerate(unique_classes):
            matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))

    if normalize is not None:
        if normalize == 'true':
            matrix = matrix / matrix.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            matrix = matrix / matrix.sum(axis=0, keepdims=True)
        elif normalize == 'all':
            matrix = matrix / matrix.sum()
        else:
            raise ValueError("normalize must be 'true', 'pred', 'all' or None")

    return matrix

def _compute_metrics(matrix: np.ndarray, metrics: Optional[list] = None) -> Dict[str, float]:
    """Compute various metrics from confusion matrix."""
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']

    result = {}
    n_classes = matrix.shape[0]
    tp = np.diag(matrix)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    tn = matrix.sum() - (tp + fp + fn)

    if 'accuracy' in metrics:
        result['accuracy'] = (tp.sum() + tn.sum()) / matrix.sum()

    if 'precision' in metrics:
        result['precision'] = tp / (tp + fp)

    if 'recall' in metrics:
        result['recall'] = tp / (tp + fn)

    if 'f1' in metrics:
        precision = result.get('precision', tp / (tp + fp))
        recall = result.get('recall', tp / (tp + fn))
        result['f1'] = 2 * (precision * recall) / (precision + recall)

    return result

def matrice_confusion_fit(y_true: np.ndarray, y_pred: np.ndarray,
                          normalize: Optional[str] = None,
                          metrics: Optional[list] = None,
                          custom_metric: Optional[Callable] = None) -> Dict:
    """
    Compute confusion matrix and metrics for classification results.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
    normalize : str, optional
        Normalization method for confusion matrix ('true', 'pred', 'all' or None).
    metrics : list, optional
        List of metric names to compute ('accuracy', 'precision', 'recall', 'f1').
    custom_metric : callable, optional
        Custom metric function that takes confusion matrix and returns a dict.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'confusion_matrix': computed confusion matrix
        - 'metrics': dictionary of computed metrics
        - 'params_used': parameters used for computation
        - 'warnings': any warnings generated

    Example
    -------
    >>> y_true = np.array([0, 1, 2, 2])
    >>> y_pred = np.array([0, 1, 1, 2])
    >>> result = matrice_confusion_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Compute confusion matrix
    confusion_matrix = _compute_confusion_matrix(y_true, y_pred, normalize)

    # Compute metrics
    computed_metrics = _compute_metrics(confusion_matrix, metrics)

    # Add custom metric if provided
    if custom_metric is not None:
        computed_metrics.update(custom_metric(confusion_matrix))

    # Prepare output
    result = {
        'confusion_matrix': confusion_matrix,
        'metrics': computed_metrics,
        'params_used': {
            'normalize': normalize,
            'metrics': metrics,
            'custom_metric': custom_metric is not None
        },
        'warnings': []
    }

    return result

################################################################################
# precision
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def precision_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float = 0.5,
    normalize: str = "none",
    metric: Union[str, Callable] = "accuracy",
    solver: str = "closed_form",
    penalty: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Compute precision for logistic regression model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,) with binary values
    threshold : float, default=0.5
        Decision threshold for classification
    normalize : str, default="none"
        Normalization method: "none", "standard", "minmax", or "robust"
    metric : str or callable, default="accuracy"
        Evaluation metric: "accuracy", "precision", "recall", or custom callable
    solver : str, default="closed_form"
        Solver method: "closed_form", "gradient_descent", etc.
    penalty : str or None, default=None
        Regularization type: "none", "l1", "l2", or "elasticnet"
    tol : float, default=1e-4
        Tolerance for stopping criteria
    max_iter : int, default=1000
        Maximum number of iterations
    custom_metric : callable or None, default=None
        Custom metric function if needed

    Returns
    -------
    dict
        Dictionary containing:
        - "result": predicted probabilities or classes
        - "metrics": computed metrics dictionary
        - "params_used": parameters used in computation
        - "warnings": any warnings encountered

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = precision_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_normalized = _normalize_data(X, method=normalize)

    # Choose solver and fit model
    if solver == "closed_form":
        coef_, intercept_ = _closed_form_solver(X_normalized, y)
    else:
        raise ValueError(f"Solver {solver} not implemented")

    # Compute predictions
    probas_ = _sigmoid(X_normalized @ coef_ + intercept_)
    y_pred = (probas_ >= threshold).astype(int)

    # Compute metrics
    metrics = _compute_metrics(y, y_pred, metric=metric)

    return {
        "result": probas_,
        "metrics": metrics,
        "params_used": {
            "threshold": threshold,
            "normalize": normalize,
            "metric": metric,
            "solver": solver,
            "penalty": penalty
        },
        "warnings": []
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
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only binary values (0 or 1)")

def _normalize_data(X: np.ndarray, method: str = "none") -> np.ndarray:
    """Normalize feature matrix."""
    if method == "standard":
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == "minmax":
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == "robust":
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> tuple:
    """Closed form solution for logistic regression."""
    # Add intercept term
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    # Compute coefficients using inverse of Hessian
    coef_ = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
    intercept_ = coef_[0]
    coef_ = coef_[1:]
    return coef_, intercept_

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-z))

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric: Union[str, Callable]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    if callable(metric):
        return {"custom_metric": metric(y_true, y_pred)}

    metrics = {}
    if metric == "accuracy" or metric is None:
        metrics["accuracy"] = np.mean(y_true == y_pred)
    if metric == "precision" or metric is None:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    if metric == "recall" or metric is None:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    return metrics

################################################################################
# rappel
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def rappel_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'logloss',
    solver: str = 'gradient_descent',
    max_iter: int = 1000,
    tol: float = 1e-4,
    learning_rate: float = 0.01,
    penalty: Optional[str] = None,
    alpha: float = 1.0,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a logistic regression model with configurable options.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalizer : Optional[Callable]
        Function to normalize the input features. Default is None.
    metric : str
        Metric to evaluate model performance. Options: 'logloss', 'mse', 'mae', 'r2'.
    solver : str
        Optimization algorithm. Options: 'gradient_descent', 'newton', 'coordinate_descent'.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for stopping criteria.
    learning_rate : float
        Learning rate for gradient descent.
    penalty : Optional[str]
        Regularization type. Options: 'l1', 'l2', 'elasticnet'.
    alpha : float
        Regularization strength.
    custom_metric : Optional[Callable]
        Custom metric function. Signature: (y_true, y_pred) -> float.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Fitted model parameters.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> result = rappel_fit(X, y, solver='gradient_descent', penalty='l2')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if normalizer is provided
    if normalizer is not None:
        X = normalizer(X)

    # Initialize parameters
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0

    # Choose solver
    if solver == 'gradient_descent':
        weights, bias = _gradient_descent(
            X, y, weights, bias,
            max_iter=max_iter,
            tol=tol,
            learning_rate=learning_rate,
            penalty=penalty,
            alpha=alpha,
            verbose=verbose
        )
    elif solver == 'newton':
        weights, bias = _newton_method(
            X, y, weights, bias,
            max_iter=max_iter,
            tol=tol,
            penalty=penalty,
            alpha=alpha,
            verbose=verbose
        )
    elif solver == 'coordinate_descent':
        weights, bias = _coordinate_descent(
            X, y, weights, bias,
            max_iter=max_iter,
            tol=tol,
            penalty=penalty,
            alpha=alpha,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    y_pred = _sigmoid(np.dot(X, weights) + bias)
    metrics = _compute_metrics(y, y_pred, metric=metric, custom_metric=custom_metric)

    # Prepare output
    result = {
        'result': {'weights': weights, 'bias': bias},
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol,
            'learning_rate': learning_rate,
            'penalty': penalty,
            'alpha': alpha
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
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

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-z))

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    max_iter: int = 1000,
    tol: float = 1e-4,
    learning_rate: float = 0.01,
    penalty: Optional[str] = None,
    alpha: float = 1.0,
    verbose: bool = False
) -> tuple[np.ndarray, float]:
    """Gradient descent optimization."""
    n_samples = X.shape[0]
    for i in range(max_iter):
        # Compute predictions
        z = np.dot(X, weights) + bias
        y_pred = _sigmoid(z)

        # Compute gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        # Apply regularization
        if penalty == 'l1':
            dw += alpha * np.sign(weights)
        elif penalty == 'l2':
            dw += 2 * alpha * weights
        elif penalty == 'elasticnet':
            dw += alpha * (np.sign(weights) + 2 * weights)

        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Check convergence
        if np.linalg.norm(dw) < tol and np.abs(db) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break

    return weights, bias

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    max_iter: int = 1000,
    tol: float = 1e-4,
    penalty: Optional[str] = None,
    alpha: float = 1.0,
    verbose: bool = False
) -> tuple[np.ndarray, float]:
    """Newton's method optimization."""
    n_samples = X.shape[0]
    for i in range(max_iter):
        # Compute predictions
        z = np.dot(X, weights) + bias
        y_pred = _sigmoid(z)

        # Compute gradients and Hessian
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        # Apply regularization
        if penalty == 'l1':
            dw += alpha * np.sign(weights)
        elif penalty == 'l2':
            dw += 2 * alpha * weights
        elif penalty == 'elasticnet':
            dw += alpha * (np.sign(weights) + 2 * weights)

        # Hessian matrix
        hessian = (1 / n_samples) * np.dot(X.T, y_pred * (1 - y_pred) * X)
        if penalty == 'l2':
            hessian += 2 * alpha * np.eye(X.shape[1])

        # Update parameters
        delta_w = np.linalg.solve(hessian, -dw)
        weights += delta_w
        bias -= db / (1e-8 + np.sum(y_pred * (1 - y_pred)))

        # Check convergence
        if np.linalg.norm(delta_w) < tol and np.abs(db) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break

    return weights, bias

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    max_iter: int = 1000,
    tol: float = 1e-4,
    penalty: Optional[str] = None,
    alpha: float = 1.0,
    verbose: bool = False
) -> tuple[np.ndarray, float]:
    """Coordinate descent optimization."""
    n_samples, n_features = X.shape
    for i in range(max_iter):
        for j in range(n_features):
            # Compute residuals
            z = np.dot(X, weights) + bias - X[:, j] * weights[j]
            y_pred = _sigmoid(z)

            # Compute gradient for feature j
            r = (y - y_pred) * X[:, j]
            grad_j = np.sum(r)

            # Apply regularization
            if penalty == 'l1':
                grad_j -= alpha * np.sign(weights[j])
            elif penalty == 'l2':
                grad_j -= 2 * alpha * weights[j]
            elif penalty == 'elasticnet':
                grad_j -= alpha * (np.sign(weights[j]) + 2 * weights[j])

            # Update weight j
            weights[j] += grad_j / (np.sum(X[:, j]**2) + 1e-8)

        # Update bias
        z = np.dot(X, weights) + bias
        y_pred = _sigmoid(z)
        db = np.sum(y - y_pred) / n_samples
        bias += db

        # Check convergence
        if np.linalg.norm(db) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break

    return weights, bias

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: str = 'logloss',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}

    if metric == 'logloss':
        metrics['logloss'] = -np.mean(y_true * np.log(y_pred + 1e-8) +
                                     (1 - y_true) * np.log(1 - y_pred + 1e-8))
    elif metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        metrics['r2'] = 1 - (ss_residual / ss_total)

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)

    return metrics

################################################################################
# f1_score
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must contain only 0 and 1")
    if np.any((y_pred < 0) | (y_pred > 1)):
        raise ValueError("y_pred must be between 0 and 1")

def _compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                             threshold: float = 0.5) -> Dict[str, int]:
    """Compute confusion matrix components."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    tn = np.sum((y_true == 0) & (y_pred_binary == 0))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

def f1_score_compute(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute F1 score for binary classification.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth (binary) target values.
    y_pred : np.ndarray
        Estimated probabilities or decision function.
    threshold : float, optional
        Decision threshold for converting probabilities to binary predictions.

    Returns:
    --------
    dict
        Dictionary containing:
        - result: computed F1 score
        - metrics: precision, recall, confusion matrix components
        - params_used: parameters used in computation
        - warnings: any warnings generated

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0.1, 0.9, 0.8, 0.2])
    >>> f1_score_compute(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Compute confusion matrix components
    cm = _compute_confusion_matrix(y_true, y_pred, threshold)

    # Calculate metrics
    precision = cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) > 0 else 0
    recall = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "result": f1,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "confusion_matrix": cm
        },
        "params_used": {
            "threshold": threshold
        },
        "warnings": []
    }

################################################################################
# auc_roc
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def auc_roc_compute(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    *,
    method: str = 'trapezoidal',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the Area Under the ROC Curve (AUC-ROC) for binary classification.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred_proba : np.ndarray
        Predicted probabilities for the positive class.
    method : str, optional
        Integration method for AUC computation ('trapezoidal' or 'custom').
    custom_metric : Callable, optional
        Custom function to compute AUC if method='custom'.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the AUC result and related information.

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred_proba = np.array([0.1, 0.9, 0.8, 0.2])
    >>> result = auc_roc_compute(y_true, y_pred_proba)
    """
    _validate_inputs(y_true, y_pred_proba)

    if method == 'trapezoidal':
        auc_value = _compute_auc_trapezoidal(y_true, y_pred_proba)
    elif method == 'custom' and custom_metric is not None:
        auc_value = _compute_auc_custom(y_true, y_pred_proba, custom_metric)
    else:
        raise ValueError("Invalid method or missing custom metric function.")

    return {
        "result": auc_value,
        "metrics": {"auc": auc_value},
        "params_used": {
            "method": method,
            "custom_metric": custom_metric.__name__ if custom_metric else None
        },
        "warnings": []
    }

def _validate_inputs(y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred_proba, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if y_true.shape != y_pred_proba.shape:
        raise ValueError("y_true and y_pred_proba must have the same shape.")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must contain only 0 and 1.")
    if np.any((y_pred_proba < 0) | (y_pred_proba > 1)):
        raise ValueError("y_pred_proba must be between 0 and 1.")

def _compute_auc_trapezoidal(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Compute AUC using trapezoidal rule."""
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred_proba[sorted_indices]

    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    tpr = tp / np.sum(y_true)
    fpr = fp / np.sum(1 - y_true)

    auc_value = np.trapz(tpr, fpr)
    return auc_value

def _compute_auc_custom(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    custom_metric: Callable[[np.ndarray, np.ndarray], float]
) -> float:
    """Compute AUC using a custom metric function."""
    return custom_metric(y_true, y_pred_proba)

################################################################################
# surcharge
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def surcharge_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = "logloss",
    solver: str = "gradient_descent",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Fit a logistic regression model with configurable options.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    normalizer : Optional[Callable]
        Function to normalize the input features
    metric : Union[str, Callable]
        Metric to evaluate model performance. Can be "logloss", "mse", "mae", "r2" or custom callable
    solver : str
        Solver to use. Options: "gradient_descent", "newton", "coordinate_descent"
    regularization : Optional[str]
        Regularization type. Options: None, "l1", "l2", "elasticnet"
    tol : float
        Tolerance for stopping criteria
    max_iter : int
        Maximum number of iterations
    learning_rate : float
        Learning rate for gradient descent
    custom_metric : Optional[Callable]
        Custom metric function if not using built-in metrics

    Returns:
    --------
    Dict containing:
        - result: fitted model parameters
        - metrics: computed metrics
        - params_used: configuration used
        - warnings: any warnings generated

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> result = surcharge_fit(X, y, normalizer=np.std, metric="logloss")
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize parameters
    n_features = X_normalized.shape[1]
    params = np.zeros(n_features)

    # Choose solver
    if solver == "gradient_descent":
        params = _gradient_descent(
            X_normalized, y, params,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter,
            learning_rate=learning_rate
        )
    elif solver == "newton":
        params = _newton_method(
            X_normalized, y, params,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter
        )
    elif solver == "coordinate_descent":
        params = _coordinate_descent(
            X_normalized, y, params,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(
        X_normalized, y, params,
        metric=metric,
        custom_metric=custom_metric
    )

    # Prepare output
    result = {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": _generate_warnings(X_normalized, y)
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

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to input features."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    learning_rate: float
) -> np.ndarray:
    """Gradient descent solver for logistic regression."""
    n_samples, n_features = X.shape
    converged = False

    for _ in range(max_iter):
        # Compute predictions and gradient
        predictions = _sigmoid(X @ params)
        gradient = X.T @ (predictions - y) / n_samples

        # Apply regularization if specified
        if regularization == "l1":
            gradient += np.sign(params) * (1 / n_samples)
        elif regularization == "l2":
            gradient += 2 * params / n_samples
        elif regularization == "elasticnet":
            gradient += np.sign(params) * (1 / n_samples) + 2 * params / n_samples

        # Update parameters
        params_prev = np.copy(params)
        params -= learning_rate * gradient

        # Check convergence
        if np.linalg.norm(params - params_prev) < tol:
            converged = True
            break

    if not converged:
        warnings.warn("Gradient descent did not converge")

    return params

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Newton's method solver for logistic regression."""
    n_samples, n_features = X.shape
    converged = False

    for _ in range(max_iter):
        # Compute predictions and Hessian
        predictions = _sigmoid(X @ params)
        hessian = X.T @ np.diag(predictions * (1 - predictions)) @ X / n_samples

        # Apply regularization if specified
        if regularization == "l2":
            hessian += 2 * np.eye(n_features) / n_samples

        # Update parameters
        params_prev = np.copy(params)
        gradient = X.T @ (predictions - y) / n_samples
        params -= np.linalg.inv(hessian) @ gradient

        # Check convergence
        if np.linalg.norm(params - params_prev) < tol:
            converged = True
            break

    if not converged:
        warnings.warn("Newton's method did not converge")

    return params

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Coordinate descent solver for logistic regression."""
    n_samples, n_features = X.shape
    converged = False

    for _ in range(max_iter):
        params_prev = np.copy(params)

        for j in range(n_features):
            # Compute residual
            X_j = X[:, j]
            residual = y - _sigmoid(X @ params)

            # Compute update
            if regularization == "l1":
                beta_j = _soft_threshold(
                    X_j.T @ residual / n_samples,
                    1 / n_samples
                )
            elif regularization == "l2":
                beta_j = (X_j.T @ residual / n_samples) / (
                    X_j.T @ X_j / n_samples + 2 / n_samples
                )
            else:
                beta_j = (X_j.T @ residual) / (X_j.T @ X_j)

            params[j] = beta_j

        # Check convergence
        if np.linalg.norm(params - params_prev) < tol:
            converged = True
            break

    if not converged:
        warnings.warn("Coordinate descent did not converge")

    return params

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate evaluation metrics."""
    predictions = _sigmoid(X @ params)

    metrics = {}

    if metric == "logloss":
        metrics["log_loss"] = _log_loss(y, predictions)
    elif metric == "mse":
        metrics["mse"] = np.mean((y - predictions) ** 2)
    elif metric == "mae":
        metrics["mae"] = np.mean(np.abs(y - predictions))
    elif metric == "r2":
        metrics["r2"] = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
    elif callable(metric):
        metrics["custom_metric"] = metric(y, predictions)

    if custom_metric is not None:
        metrics["custom_metric"] = custom_metric(y, predictions)

    return metrics

def _generate_warnings(X: np.ndarray, y: np.ndarray) -> list:
    """Generate warnings about potential issues."""
    warnings = []

    if np.any(np.abs(X) > 1e6):
        warnings.append("Warning: Features contain very large values")

    if np.any(np.abs(y) > 1.5):
        warnings.append("Warning: Target values are not in [0,1] range")

    return warnings

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-z))

def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Compute log loss."""
    return -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))

def _soft_threshold(rho: float, alpha: float) -> float:
    """Soft thresholding operator."""
    if rho < -alpha:
        return rho + alpha
    elif rho > alpha:
        return rho - alpha
    else:
        return 0.0

################################################################################
# sous_charge
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def sous_charge_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'logloss',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    penalty: float = 1.0,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit a logistic regression model with configurable parameters.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalizer : Optional[Callable]
        Function to normalize the input features. Default is None.
    metric : Union[str, Callable]
        Metric to evaluate the model. Can be 'mse', 'mae', 'r2', 'logloss' or a custom callable.
    solver : str
        Solver to use. Options are 'gradient_descent', 'newton', 'coordinate_descent'.
    regularization : Optional[str]
        Regularization type. Options are None, 'l1', 'l2', 'elasticnet'.
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    learning_rate : float
        Learning rate for gradient descent.
    penalty : float
        Regularization penalty strength.
    custom_metric : Optional[Callable]
        Custom metric function.

    Returns:
    --------
    Dict
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if a normalizer is provided
    if normalizer is not None:
        X = normalizer(X)

    # Initialize parameters
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    intercept = 0.0

    # Choose solver
    if solver == 'gradient_descent':
        coefficients, intercept = _gradient_descent(
            X, y, coefficients, intercept,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            regularization=regularization,
            penalty=penalty
        )
    elif solver == 'newton':
        coefficients, intercept = _newton_method(
            X, y, coefficients, intercept,
            max_iter=max_iter,
            tol=tol,
            regularization=regularization,
            penalty=penalty
        )
    elif solver == 'coordinate_descent':
        coefficients, intercept = _coordinate_descent(
            X, y, coefficients, intercept,
            max_iter=max_iter,
            tol=tol,
            regularization=regularization,
            penalty=penalty
        )
    else:
        raise ValueError("Invalid solver specified.")

    # Calculate metrics
    metrics = _calculate_metrics(X, y, coefficients, intercept, metric, custom_metric)

    # Prepare output
    result = {
        'result': {
            'coefficients': coefficients,
            'intercept': intercept
        },
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter,
            'learning_rate': learning_rate,
            'penalty': penalty
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
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

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    learning_rate: float,
    max_iter: int,
    tol: float,
    regularization: Optional[str],
    penalty: float
) -> tuple:
    """Perform gradient descent optimization."""
    n_samples = X.shape[0]
    for _ in range(max_iter):
        # Compute predictions
        linear_model = np.dot(X, coefficients) + intercept
        predictions = _sigmoid(linear_model)

        # Compute gradients
        gradient = np.dot(X.T, (predictions - y)) / n_samples

        # Apply regularization
        if regularization == 'l1':
            gradient += penalty * np.sign(coefficients)
        elif regularization == 'l2':
            gradient += 2 * penalty * coefficients
        elif regularization == 'elasticnet':
            gradient += penalty * (np.sign(coefficients) + 2 * coefficients)

        # Update parameters
        coefficients -= learning_rate * gradient
        intercept -= learning_rate * np.mean(predictions - y)

        # Check convergence
        if np.linalg.norm(gradient) < tol:
            break

    return coefficients, intercept

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    max_iter: int,
    tol: float,
    regularization: Optional[str],
    penalty: float
) -> tuple:
    """Perform Newton's method optimization."""
    n_samples = X.shape[0]
    for _ in range(max_iter):
        # Compute predictions
        linear_model = np.dot(X, coefficients) + intercept
        predictions = _sigmoid(linear_model)

        # Compute gradients and Hessian
        gradient = np.dot(X.T, (predictions - y)) / n_samples
        hessian_diag = np.diag(np.dot(X, (predictions * (1 - predictions)) * X.T) / n_samples)

        # Apply regularization
        if regularization == 'l1':
            gradient += penalty * np.sign(coefficients)
        elif regularization == 'l2':
            hessian_diag += 2 * penalty * np.eye(X.shape[1])
        elif regularization == 'elasticnet':
            gradient += penalty * (np.sign(coefficients) + 2 * coefficients)
            hessian_diag += 2 * penalty * np.eye(X.shape[1])

        # Update parameters
        coefficients -= np.linalg.solve(hessian_diag, gradient)
        intercept -= np.mean(predictions - y)

        # Check convergence
        if np.linalg.norm(gradient) < tol:
            break

    return coefficients, intercept

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    max_iter: int,
    tol: float,
    regularization: Optional[str],
    penalty: float
) -> tuple:
    """Perform coordinate descent optimization."""
    n_samples, n_features = X.shape
    for _ in range(max_iter):
        for j in range(n_features):
            # Compute residuals
            linear_model = np.dot(X, coefficients) + intercept - X[:, j] * coefficients[j]
            predictions = _sigmoid(linear_model)

            # Compute gradient for feature j
            r_j = np.dot(X[:, j], (predictions - y))

            # Update coefficient for feature j
            if regularization == 'l1':
                coefficients[j] = _soft_threshold(r_j / n_samples, penalty)
            elif regularization == 'l2':
                coefficients[j] = r_j / (n_samples + 2 * penalty)
            elif regularization == 'elasticnet':
                coefficients[j] = _soft_threshold(r_j / (n_samples + 2 * penalty), penalty)
            else:
                coefficients[j] = r_j / n_samples

        # Update intercept
        linear_model = np.dot(X, coefficients)
        predictions = _sigmoid(linear_model)
        intercept -= np.mean(predictions - y)

        # Check convergence
        if np.linalg.norm(r_j) < tol:
            break

    return coefficients, intercept

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))

def _soft_threshold(r: float, lambda_: float) -> float:
    """Compute the soft thresholding operator."""
    if r < -lambda_:
        return 0
    elif r > lambda_:
        return r - lambda_
    else:
        return 0

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate the specified metrics."""
    linear_model = np.dot(X, coefficients) + intercept
    predictions = _sigmoid(linear_model)

    metrics_dict = {}

    if metric == 'mse':
        metrics_dict['mse'] = np.mean((predictions - y) ** 2)
    elif metric == 'mae':
        metrics_dict['mae'] = np.mean(np.abs(predictions - y))
    elif metric == 'r2':
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics_dict['r2'] = 1 - (ss_res / ss_tot)
    elif metric == 'logloss':
        metrics_dict['logloss'] = -np.mean(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))
    elif callable(metric):
        metrics_dict['custom_metric'] = metric(y, predictions)

    if custom_metric is not None:
        metrics_dict['custom_metric'] = custom_metric(y, predictions)

    return metrics_dict

################################################################################
# seuil_classification
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using specified method."""
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

def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
                    metric: Union[str, Callable]) -> float:
    """Compute specified metric."""
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
    elif metric == "logloss":
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_threshold(y_scores: np.ndarray, method: str,
                      **kwargs: Dict[str, float]) -> float:
    """Compute classification threshold using specified method."""
    if method == "fixed":
        return kwargs.get("threshold", 0.5)
    elif method == "optimize":
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_scores, kwargs.get("y_true", None))
        if "metric" in kwargs:
            metric = kwargs["metric"]
            if callable(metric):
                scores = [metric(t, f) for t in thresholds]
            else:
                if metric == "f1":
                    from sklearn.metrics import f1_score
                    scores = [f1_score(kwargs["y_true"], y_scores >= t) for t in thresholds]
                else:
                    raise ValueError(f"Unknown optimization metric: {metric}")
            return thresholds[np.argmax(scores)]
        else:
            raise ValueError("Optimization method requires a metric.")
    else:
        raise ValueError(f"Unknown threshold computation method: {method}")

def seuil_classification_fit(X: np.ndarray, y_scores: np.ndarray,
                            method: str = "fixed",
                            **kwargs: Dict[str, Union[float, Callable]]) -> Dict:
    """Fit classification threshold to optimize specified metric.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y_scores : np.ndarray
        Predicted scores or probabilities of shape (n_samples,).
    method : str, optional
        Method to compute threshold ("fixed" or "optimize").
    **kwargs : dict
        Additional parameters for the method.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_inputs(X, y_scores)

    threshold = _compute_threshold(y_scores, method, **kwargs)
    y_pred = (y_scores >= threshold).astype(int)

    metrics = {}
    if "metric" in kwargs:
        metric_func = kwargs["metric"]
        metrics["optimized_metric"] = _compute_metric(y_pred, y_scores, metric_func)

    return {
        "result": {"threshold": threshold},
        "metrics": metrics,
        "params_used": {
            "method": method,
            **kwargs
        },
        "warnings": []
    }

################################################################################
# overfitting
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def overfitting_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'logloss',
    solver: str = 'gradient_descent',
    max_iter: int = 1000,
    tol: float = 1e-4,
    l2_penalty: float = 0.0,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a logistic regression model and evaluate overfitting.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize features. Default is identity.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric to evaluate model performance. Options: 'mse', 'mae', 'r2', 'logloss'.
    solver : str
        Solver to use. Options: 'gradient_descent', 'newton'.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for stopping criteria.
    l2_penalty : float
        L2 regularization strength.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Fitted model parameters.
        - 'metrics': Evaluation metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> result = overfitting_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_normalized = normalizer(X)

    # Initialize parameters
    n_features = X_normalized.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0

    # Choose solver
    if solver == 'gradient_descent':
        weights, bias = _gradient_descent(
            X_normalized, y, weights, bias,
            max_iter=max_iter, tol=tol,
            l2_penalty=l2_penalty,
            random_state=random_state
        )
    elif solver == 'newton':
        weights, bias = _newton_method(
            X_normalized, y, weights, bias,
            max_iter=max_iter, tol=tol,
            l2_penalty=l2_penalty
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    y_pred = _sigmoid(np.dot(X_normalized, weights) + bias)
    metrics = _compute_metrics(y, y_pred, metric)

    # Prepare output
    result = {
        'result': {'weights': weights, 'bias': bias},
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'metric': metric,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol,
            'l2_penalty': l2_penalty
        },
        'warnings': []
    }

    return result

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

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-z))

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    max_iter: int = 1000,
    tol: float = 1e-4,
    l2_penalty: float = 0.0,
    random_state: Optional[int] = None
) -> tuple[np.ndarray, float]:
    """Gradient descent solver for logistic regression."""
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    learning_rate = 0.1

    for _ in range(max_iter):
        # Compute predictions
        linear_model = np.dot(X, weights) + bias
        y_pred = _sigmoid(linear_model)

        # Compute gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + l2_penalty * weights
        db = (1 / n_samples) * np.sum(y_pred - y)

        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Check convergence
        if np.linalg.norm(dw) < tol and abs(db) < tol:
            break

    return weights, bias

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    max_iter: int = 1000,
    tol: float = 1e-4,
    l2_penalty: float = 0.0
) -> tuple[np.ndarray, float]:
    """Newton's method solver for logistic regression."""
    n_samples, n_features = X.shape

    for _ in range(max_iter):
        # Compute predictions
        linear_model = np.dot(X, weights) + bias
        y_pred = _sigmoid(linear_model)

        # Compute Hessian and gradient
        hessian = (1 / n_samples) * np.dot(X.T * (y_pred * (1 - y_pred))[:, None], X)
        hessian += l2_penalty * np.eye(n_features)

        gradient = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        gradient += l2_penalty * weights

        # Update parameters
        delta = np.linalg.solve(hessian, -gradient)
        weights += delta[:-1]
        bias += delta[-1]

        # Check convergence
        if np.linalg.norm(delta) < tol:
            break

    return weights, bias

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
        elif metric == 'logloss':
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            metrics['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        metrics['custom'] = metric(y_true, y_pred)

    return metrics

################################################################################
# underfitting
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def underfitting_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = "logloss",
    solver: str = "gradient_descent",
    max_iter: int = 1000,
    tol: float = 1e-4,
    learning_rate: float = 0.01,
    penalty: Optional[str] = None,
    alpha: float = 1.0,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a logistic regression model with underfitting detection.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize features
    metric : str or callable
        Metric to evaluate model performance ("mse", "mae", "r2", "logloss")
    solver : str
        Optimization algorithm ("closed_form", "gradient_descent", "newton")
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for stopping criterion
    learning_rate : float
        Learning rate for gradient descent
    penalty : str or None
        Regularization type ("l1", "l2", "elasticnet")
    alpha : float
        Regularization strength
    custom_metric : callable or None
        Custom metric function if needed
    verbose : bool
        Whether to print progress

    Returns
    -------
    dict
        Dictionary containing:
        - "result": fitted model parameters
        - "metrics": computed metrics
        - "params_used": actual parameters used
        - "warnings": any warnings generated

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = underfitting_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_normalized = normalizer(X)
    n_samples, n_features = X.shape

    # Initialize parameters
    params = _initialize_parameters(n_features)

    # Choose solver
    if solver == "gradient_descent":
        params = _gradient_descent(
            X_normalized, y, params,
            max_iter=max_iter,
            tol=tol,
            learning_rate=learning_rate,
            penalty=penalty,
            alpha=alpha
        )
    elif solver == "newton":
        params = _newton_method(
            X_normalized, y, params,
            max_iter=max_iter,
            tol=tol
        )
    elif solver == "closed_form":
        params = _closed_form_solution(X_normalized, y)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, params, metric, custom_metric)

    # Check for underfitting
    warnings = _check_underfitting(metrics, metric)

    return {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if hasattr(normalizer, '__name__') else "custom",
            "metric": metric,
            "solver": solver,
            "max_iter": max_iter,
            "tol": tol,
            "learning_rate": learning_rate,
            "penalty": penalty,
            "alpha": alpha
        },
        "warnings": warnings
    }

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

def _initialize_parameters(n_features: int) -> np.ndarray:
    """Initialize model parameters."""
    return np.zeros(n_features + 1)  # +1 for bias term

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-4,
    learning_rate: float = 0.01,
    penalty: Optional[str] = None,
    alpha: float = 1.0
) -> np.ndarray:
    """Gradient descent optimization for logistic regression."""
    n_samples, _ = X.shape
    prev_loss = float('inf')

    for i in range(max_iter):
        # Compute predictions
        linear_model = np.dot(X, params[1:]) + params[0]
        predictions = _sigmoid(linear_model)

        # Compute gradient
        gradient = np.zeros_like(params)
        gradient[0] = np.mean(predictions - y)  # bias term
        gradient[1:] = (np.dot(X.T, predictions - y) / n_samples)

        # Add regularization if needed
        if penalty == "l2":
            gradient[1:] += alpha * params[1:]
        elif penalty == "l1":
            gradient[1:] += alpha * np.sign(params[1:])
        elif penalty == "elasticnet":
            gradient[1:] += alpha * (params[1:] + np.sign(params[1:]))

        # Update parameters
        params -= learning_rate * gradient

        # Check convergence
        current_loss = _logloss(y, predictions)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return params

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> np.ndarray:
    """Newton's method optimization for logistic regression."""
    n_samples, _ = X.shape
    prev_loss = float('inf')

    for i in range(max_iter):
        # Compute predictions
        linear_model = np.dot(X, params[1:]) + params[0]
        predictions = _sigmoid(linear_model)

        # Compute gradient and hessian
        gradient = np.zeros_like(params)
        gradient[0] = np.mean(predictions - y)  # bias term
        gradient[1:] = (np.dot(X.T, predictions - y) / n_samples)

        hessian = np.zeros((len(params), len(params)))
        p = predictions * (1 - predictions)
        hessian[0, 0] = np.mean(p) / n_samples
        hessian[0, 1:] = (np.dot(X.T, p) / n_samples).flatten()
        hessian[1:, 0] = hessian[0, 1:]
        hessian[1:, 1:] = (np.dot(X.T * p[:, np.newaxis], X) / n_samples)

        # Update parameters
        params -= np.linalg.solve(hessian, gradient)

        # Check convergence
        current_loss = _logloss(y, predictions)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return params

def _closed_form_solution(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form solution for logistic regression (approximate)."""
    # This is actually ridge regression as exact closed form doesn't exist
    # for logistic regression, but provided for completeness
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    return np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    linear_model = np.dot(X, params[1:]) + params[0]
    predictions = _sigmoid(linear_model)

    metrics = {}

    if metric == "mse" or custom_metric is None:
        metrics["mse"] = np.mean((y - predictions) ** 2)
    if metric == "mae" or custom_metric is None:
        metrics["mae"] = np.mean(np.abs(y - predictions))
    if metric == "r2" or custom_metric is None:
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics["r2"] = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    if metric == "logloss" or custom_metric is None:
        metrics["logloss"] = _logloss(y, predictions)

    if custom_metric is not None:
        metrics["custom"] = custom_metric(y, predictions)

    return metrics

def _check_underfitting(metrics: Dict[str, float], metric: str) -> list:
    """Check for underfitting conditions."""
    warnings = []

    if metric == "logloss" and metrics.get("logloss", float('inf')) > 0.7:
        warnings.append("High log loss suggests potential underfitting")
    elif metric == "r2" and metrics.get("r2", float('nan')) < 0.1:
        warnings.append("Low R-squared suggests potential underfitting")

    return warnings

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-x))

def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    """Compute log loss."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
