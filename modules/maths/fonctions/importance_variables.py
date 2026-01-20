"""
Quantix – Module importance_variables
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# p_value
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def p_value_fit(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'permutation',
    metric: Union[str, Callable] = 'mse',
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
    **kwargs
) -> Dict:
    """
    Compute p-values for variable importance using different methods.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    method : str
        Method for p-value computation ('permutation', 'bootstrap')
    metric : str or callable
        Metric to evaluate importance ('mse', 'mae', 'r2', custom callable)
    n_permutations : int
        Number of permutations for permutation test
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = p_value_fit(X, y, method='permutation', metric='mse')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Choose metric function
    metric_func = _get_metric_function(metric)

    # Compute base score
    base_score = metric_func(y, _predict(X))

    # Compute p-values based on method
    if method == 'permutation':
        p_values = _compute_permutation_pvalues(X, y, metric_func, base_score,
                                              n_permutations, rng)
    elif method == 'bootstrap':
        p_values = _compute_bootstrap_pvalues(X, y, metric_func, base_score,
                                            n_permutations, rng)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'permutation', 'bootstrap'")

    return {
        "result": p_values,
        "metrics": {"base_score": base_score},
        "params_used": {
            "method": method,
            "metric": metric,
            "n_permutations": n_permutations,
            "random_state": random_state
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
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _get_metric_function(metric: Union[str, Callable]) -> Callable:
    """Get metric function based on input."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }

    if isinstance(metric, str):
        if metric.lower() not in metrics:
            raise ValueError(f"Unknown metric: {metric}. Choose from {list(metrics.keys())}")
        return metrics[metric.lower()]
    elif callable(metric):
        return metric
    else:
        raise TypeError("Metric must be either a string or callable")

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

def _predict(X: np.ndarray) -> np.ndarray:
    """Simple prediction function (mean of each feature)."""
    return np.mean(X, axis=0)

def _compute_permutation_pvalues(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    base_score: float,
    n_permutations: int,
    rng: np.random.RandomState
) -> np.ndarray:
    """Compute p-values using permutation test."""
    n_features = X.shape[1]
    permuted_scores = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Permute target variable
        y_permuted = rng.permutation(y)
        permuted_scores[i] = metric_func(y_permuted, _predict(X))

    # Compute p-values
    p_values = np.mean(permuted_scores >= base_score, axis=0)

    return p_values

def _compute_bootstrap_pvalues(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    base_score: float,
    n_permutations: int,
    rng: np.random.RandomState
) -> np.ndarray:
    """Compute p-values using bootstrap."""
    n_samples = X.shape[0]
    n_features = X.shape[1]
    bootstrap_scores = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Bootstrap sample
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        bootstrap_scores[i] = metric_func(y_boot, _predict(X_boot))

    # Compute p-values
    p_values = np.mean(bootstrap_scores <= base_score, axis=0)

    return p_values

################################################################################
# r_squared
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> None:
    """Validate input arrays."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if sample_weight is not None:
        if len(sample_weight) != len(y_true):
            raise ValueError("sample_weight must have the same length as y_true.")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values.")

def _compute_r_squared(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """Compute R-squared metric."""
    _validate_inputs(y_true, y_pred, sample_weight)

    if sample_weight is None:
        sample_weight = np.ones_like(y_true)

    y_mean = np.average(y_true, weights=sample_weight)
    ss_total = np.sum(sample_weight * (y_true - y_mean) ** 2)
    ss_residual = np.sum(sample_weight * (y_true - y_pred) ** 2)

    if ss_total == 0:
        return 1.0  # Perfect prediction

    r_squared = 1 - (ss_residual / ss_total)
    return np.clip(r_squared, -np.inf, 1.0)

def r_squared_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Compute R-squared metric for variable importance analysis.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    sample_weight : Optional[np.ndarray], default=None
        Individual weights for each sample.

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing:
        - "result": float, the computed R-squared value
        - "metrics": Dict[str, float], additional metrics (currently just r_squared)
        - "params_used": Dict[str, any], parameters used in computation
        - "warnings": List[str], any warnings generated

    Examples
    --------
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> r_squared_fit(y_true, y_pred)
    {
        'result': 0.9486081370449679,
        'metrics': {'r_squared': 0.9486081370449679},
        'params_used': {'sample_weight': None},
        'warnings': []
    }
    """
    warnings = []

    # Compute R-squared
    r_squared_value = _compute_r_squared(y_true, y_pred, sample_weight)

    # Prepare output
    result = {
        "result": r_squared_value,
        "metrics": {"r_squared": r_squared_value},
        "params_used": {
            "sample_weight": sample_weight
        },
        "warnings": warnings
    }

    return result

################################################################################
# rmse
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> None:
    """Validate input arrays."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs contain infinite values.")
    if sample_weight is not None:
        if y_true.shape != sample_weight.shape:
            raise ValueError("y_true and sample_weight must have the same shape.")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight must contain non-negative values.")

def _compute_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """Compute RMSE (Root Mean Squared Error)."""
    residuals = y_true - y_pred
    if sample_weight is not None:
        weighted_residuals = residuals * np.sqrt(sample_weight)
        rmse = np.sqrt(np.mean(weighted_residuals ** 2))
    else:
        rmse = np.sqrt(np.mean(residuals ** 2))
    return float(rmse)

def rmse_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute RMSE (Root Mean Squared Error) between true and predicted values.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    sample_weight : Optional[np.ndarray], default=None
        Individual weights for each sample.

    Returns:
    --------
    Dict containing:
        - result: float, the computed RMSE value
        - metrics: dict with metric name and value
        - params_used: dict of parameters used in computation
        - warnings: list of warning messages

    Example:
    --------
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> result = rmse_fit(y_true, y_pred)
    """
    _validate_inputs(y_true, y_pred, sample_weight)

    result = _compute_rmse(y_true, y_pred, sample_weight)
    metrics = {"rmse": result}
    params_used = {
        "sample_weight": "provided" if sample_weight is not None else "none"
    }
    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# mae
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def mae_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Compute Mean Absolute Error (MAE) based variable importance.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalizer : Optional[Callable]
        Function to normalize features. Default is None.
    distance_metric : str
        Distance metric for feature importance calculation. Options: "euclidean", "manhattan", "cosine".
    solver : str
        Solver method. Options: "closed_form", "gradient_descent".
    regularization : Optional[str]
        Regularization type. Options: None, "l1", "l2".
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable]
        Custom metric function. If provided, overrides default metrics.

    Returns
    -------
    Dict[str, Union[Dict, float, str]]
        Dictionary containing results, metrics, parameters used and warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = mae_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Compute feature importance using MAE
    importance = _compute_mae_importance(
        X_normalized,
        y,
        distance_metric=distance_metric,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Compute metrics
    metrics = _compute_metrics(y, importance, custom_metric)

    return {
        "result": {"importance_scores": importance},
        "metrics": metrics,
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
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _compute_mae_importance(
    X: np.ndarray,
    y: np.ndarray,
    *,
    distance_metric: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Compute feature importance using MAE."""
    if solver == "closed_form":
        return _mae_closed_form(X, y, distance_metric=distance_metric)
    elif solver == "gradient_descent":
        return _mae_gradient_descent(
            X, y,
            distance_metric=distance_metric,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _mae_closed_form(X: np.ndarray, y: np.ndarray, distance_metric: str) -> np.ndarray:
    """Closed form solution for MAE feature importance."""
    if distance_metric == "euclidean":
        # Example implementation - replace with actual MAE calculation
        return np.abs(X.T @ (y - np.mean(y))) / X.shape[0]
    elif distance_metric == "manhattan":
        # Example implementation - replace with actual MAE calculation
        return np.sum(np.abs(X.T @ (y.reshape(-1, 1) - y)), axis=1) / X.shape[0]
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

def _mae_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    distance_metric: str = "euclidean",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Gradient descent solution for MAE feature importance."""
    # Example implementation - replace with actual gradient descent
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    for _ in range(max_iter):
        # Update coefficients using gradient descent
        pass  # Replace with actual implementation
    return np.abs(coefficients)

def _compute_metrics(
    y: np.ndarray,
    importance_scores: np.ndarray,
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}

    if custom_metric is not None:
        try:
            metrics["custom"] = custom_metric(y, importance_scores)
        except Exception as e:
            raise ValueError(f"Custom metric computation failed: {str(e)}")

    # Add default metrics if needed
    return metrics

################################################################################
# mse
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> None:
    """Validate input arrays and normalizer function."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")
    if normalizer is not None and not callable(normalizer):
        raise ValueError("normalizer must be a callable function")

def _normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> tuple:
    """Normalize input data using the specified normalizer."""
    if normalizer is None:
        return X, y
    X_normalized = normalizer(X)
    y_normalized = normalizer(y.reshape(-1, 1)).flatten()
    return X_normalized, y_normalized

def _compute_mse(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """Compute the Mean Squared Error between true and predicted values."""
    return np.mean((y_true - y_pred) ** 2)

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_functions: Dict[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute multiple metrics using provided metric functions."""
    return {name: func(y_true, y_pred) for name, func in metric_functions.items()}

def mse_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric_functions: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = None,
    solver: Optional[Callable] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute the Mean Squared Error (MSE) for variable importance.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalizer : Optional[Callable]
        Function to normalize the data
    metric_functions : Dict[str, Callable]
        Dictionary of additional metric functions to compute
    solver : Optional[Callable]
        Function to solve for variable importance

    Returns:
    --------
    Dict containing:
        - "result": float, the computed MSE
        - "metrics": Dict[str, float], additional metrics
        - "params_used": Dict[str, str], parameters used
        - "warnings": List[str], any warnings generated

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = mse_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y, normalizer)

    # Initialize warnings and metrics
    warnings = []
    if metric_functions is None:
        metric_functions = {}
    else:
        for name, func in metric_functions.items():
            if not callable(func):
                warnings.append(f"Metric function {name} is not callable")
                del metric_functions[name]

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalizer)

    # Use default solver if none provided
    if solver is None:
        def default_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
            """Default solver using least squares."""
            return np.linalg.pinv(X.T @ X) @ X.T @ y
        solver = default_solver

    # Compute variable importance (coefficients)
    try:
        coefficients = solver(X_norm, y_norm)
    except Exception as e:
        raise ValueError(f"Solver failed: {str(e)}")

    # Compute predictions
    y_pred = X_norm @ coefficients

    # Compute MSE
    mse_result = _compute_mse(y_norm, y_pred)

    # Compute additional metrics
    metrics = _compute_metrics(y_norm, y_pred, metric_functions)

    # Prepare output
    return {
        "result": mse_result,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else "none",
            "solver": solver.__name__,
            "metrics": list(metric_functions.keys())
        },
        "warnings": warnings
    }

################################################################################
# accuracy
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def accuracy_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, np.ndarray, list]]:
    """
    Compute variable importance based on accuracy metrics.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Function to normalize features, by default None
    metric : str, optional
        Metric to use ('mse', 'mae', 'r2', 'logloss'), by default 'mse'
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski'), by default 'euclidean'
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent'), by default 'closed_form'
    regularization : str, optional
        Regularization type (None, 'l1', 'l2', 'elasticnet'), by default None
    tol : float, optional
        Tolerance for convergence, by default 1e-4
    max_iter : int, optional
        Maximum iterations, by default 1000
    custom_metric : Callable[[np.ndarray, np.ndarray], float], optional
        Custom metric function, by default None
    custom_distance : Callable[[np.ndarray, np.ndarray], float], optional
        Custom distance function, by default None

    Returns
    -------
    Dict[str, Union[Dict, np.ndarray, list]]
        Dictionary containing:
        - "result": variable importance scores
        - "metrics": computed metrics
        - "params_used": parameters used
        - "warnings": any warnings

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = accuracy_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Select metric function
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric_function(metric)

    # Select distance function
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_function(distance)

    # Select solver function
    solver_func = _get_solver_function(solver, regularization)

    # Fit model and compute importance
    params = solver_func(X_normalized, y, tol=tol, max_iter=max_iter)
    importance_scores = _compute_importance(X_normalized, y, params, metric_func)

    # Compute additional metrics
    metrics = _compute_metrics(X_normalized, y, params, metric_func)

    # Prepare output
    result = {
        "result": importance_scores,
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
        "warnings": _check_warnings(X_normalized, y)
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
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _get_metric_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get metric function based on input string."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared,
        'logloss': _log_loss
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance_function(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get distance function based on input string."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': lambda x, y: np.sum(np.abs(x - y)**3, axis=1)  # Example for Minkowski
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _get_solver_function(solver: str, regularization: Optional[str]) -> Callable:
    """Get solver function based on input string."""
    solvers = {
        'closed_form': _closed_form_solver,
        'gradient_descent': lambda X, y, **kwargs: _gradient_descent(X, y, regularization, **kwargs),
        'newton': lambda X, y, **kwargs: _newton_method(X, y, regularization, **kwargs),
        'coordinate_descent': lambda X, y, **kwargs: _coordinate_descent(X, y, regularization, **kwargs)
    }
    if solver not in solvers:
        raise ValueError(f"Unknown solver: {solver}")
    return solvers[solver]

def _compute_importance(X: np.ndarray, y: np.ndarray, params: Dict, metric_func: Callable) -> np.ndarray:
    """Compute variable importance based on accuracy metrics."""
    # This is a simplified example - actual implementation would be more sophisticated
    baseline_pred = np.mean(y)
    baseline_score = metric_func(np.full_like(y, baseline_pred), y)

    importance_scores = []
    for i in range(X.shape[1]):
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, i])
        pred = _predict(X_permuted, params)
        score = metric_func(pred, y)
        importance_scores.append(baseline_score - score)

    return np.array(importance_scores)

def _compute_metrics(X: np.ndarray, y: np.ndarray, params: Dict, metric_func: Callable) -> Dict:
    """Compute additional metrics."""
    pred = _predict(X, params)
    return {
        'metric_value': metric_func(pred, y),
        'r2_score': _r_squared(pred, y)
    }

def _check_warnings(X: np.ndarray, y: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if X.shape[1] > 100:
        warnings.append("Warning: High dimensional data may impact performance")
    if np.any(np.std(X, axis=0) == 0):
        warnings.append("Warning: Features with zero variance detected")
    return warnings

# Example metric functions
def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred)**2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Simplified log loss for binary classification
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example distance functions
def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    return np.linalg.norm(x - y)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    return np.sum(np.abs(x - y))

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# Example solver functions
def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> Dict:
    return {"coefficients": np.linalg.inv(X.T @ X) @ X.T @ y}

def _gradient_descent(X: np.ndarray, y: np.ndarray, regularization: Optional[str], tol: float = 1e-4, max_iter: int = 1000) -> Dict:
    # Simplified gradient descent
    n_features = X.shape[1]
    coefs = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = X.T @ (X @ coefs - y) / len(y)
        if regularization == 'l2':
            gradient += 2 * coefs
        elif regularization == 'l1':
            gradient += np.sign(coefs)

        new_coefs = coefs - learning_rate * gradient
        if np.linalg.norm(new_coefs - coefs) < tol:
            break
        coefs = new_coefs

    return {"coefficients": coefs}

def _predict(X: np.ndarray, params: Dict) -> np.ndarray:
    """Make predictions using model parameters."""
    return X @ params["coefficients"]

################################################################################
# precision
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def precision_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, np.ndarray, list]]:
    """
    Compute variable importance based on precision metrics.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize features
    metric : str or callable
        Metric to evaluate precision ('mse', 'mae', 'r2', 'logloss')
    distance : str or callable
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    solver : str
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    custom_metric : callable, optional
        Custom metric function
    custom_distance : callable, optional
        Custom distance function

    Returns
    -------
    Dict[str, Union[Dict, np.ndarray, list]]
        Dictionary containing:
        - 'result': computed importance scores
        - 'metrics': evaluation metrics
        - 'params_used': parameters used
        - 'warnings': any warnings generated

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = precision_fit(X, y, metric='mse', solver='gradient_descent')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_norm = normalizer(X)

    # Choose metric function
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric_function(metric)

    # Choose distance function
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_function(distance)

    # Solve for parameters
    params, convergence_info = _solve_precision(
        X_norm, y,
        metric_func=metric_func,
        distance_func=distance_func,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Calculate importance scores
    importance_scores = _calculate_importance(X_norm, params, distance_func)

    # Calculate metrics
    metrics = _calculate_metrics(y, X_norm @ params, metric_func)

    # Prepare output
    result = {
        'result': importance_scores,
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
        'warnings': convergence_info.get('warnings', [])
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

def _get_metric_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the appropriate metric function."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared,
        'logloss': _log_loss
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance_function(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the appropriate distance function."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': lambda x, y: np.sum(np.abs(x - y)**2, axis=1)
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _solve_precision(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    solver: str,
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> tuple[np.ndarray, Dict[str, Union[bool, list]]]:
    """Solve for precision parameters using specified solver."""
    solvers = {
        'closed_form': _solve_closed_form,
        'gradient_descent': lambda X, y: _solve_gradient_descent(X, y, tol, max_iter),
        'newton': lambda X, y: _solve_newton(X, y, tol, max_iter),
        'coordinate_descent': lambda X, y: _solve_coordinate_descent(X, y, tol, max_iter)
    }

    if solver not in solvers:
        raise ValueError(f"Unknown solver: {solver}")

    params, info = solvers[solver](X, y)

    if regularization is not None:
        params = _apply_regularization(params, X, y, regularization)

    return params, info

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, Dict[str, Union[bool, list]]]:
    """Solve using closed form solution."""
    XTX = X.T @ X
    if np.linalg.matrix_rank(XTX) < X.shape[1]:
        params = np.linalg.pinv(XTX) @ X.T @ y
    else:
        params = np.linalg.inv(XTX) @ X.T @ y
    return params, {'converged': True, 'warnings': []}

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> tuple[np.ndarray, Dict[str, Union[bool, list]]]:
    """Solve using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01
    prev_loss = float('inf')
    warnings = []

    for i in range(max_iter):
        gradients = _compute_gradient(X, y, params)
        params -= learning_rate * gradients
        current_loss = _mean_squared_error(y, X @ params)

        if abs(prev_loss - current_loss) < tol:
            return params, {'converged': True, 'warnings': warnings}

        prev_loss = current_loss

    warnings.append("Maximum iterations reached without convergence")
    return params, {'converged': False, 'warnings': warnings}

def _compute_gradient(X: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Compute gradient for gradient descent."""
    residuals = y - X @ params
    return -(2 / len(y)) * (X.T @ residuals)

def _apply_regularization(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    regularization: str
) -> np.ndarray:
    """Apply specified regularization."""
    if regularization == 'l1':
        params = _apply_l1_regularization(params, X, y)
    elif regularization == 'l2':
        params = _apply_l2_regularization(params, X, y)
    elif regularization == 'elasticnet':
        params = _apply_elasticnet(params, X, y)
    return params

def _calculate_importance(
    X: np.ndarray,
    params: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Calculate variable importance scores."""
    # This is a placeholder implementation
    # Actual implementation would depend on the specific precision metric being used
    importance = np.abs(params)
    return importance / np.sum(importance)

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    return {'metric': metric_func(y_true, y_pred)}

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
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.linalg.norm(x - y)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(x - y))

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate cosine distance."""
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

################################################################################
# recall
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def recall_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float = 0.5,
    normalize: str = "none",
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "binary",
    **kwargs
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute recall score for binary classification.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
    threshold : float, default=0.5
        The threshold to use for binary classification.
    normalize : str, default="none"
        Normalization method: "none", "standard", or "minmax".
    metric : str or callable, default="binary"
        Metric to compute. Can be "binary" for binary classification
        or a custom callable function.
    **kwargs : dict
        Additional keyword arguments for the metric function.

    Returns:
    --------
    result : float
        The computed recall score.
    metrics : dict
        Additional metrics if applicable.
    params_used : dict
        Parameters used in the computation.
    warnings : list
        Any warnings generated during computation.

    Examples:
    ---------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    >>> recall_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize predictions if required
    y_pred = _apply_normalization(y_pred, normalize)

    # Apply threshold to get binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Compute recall
    result, metrics = _compute_recall(y_true, y_pred_binary, metric=metric, **kwargs)

    # Prepare output
    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "threshold": threshold,
            "normalize": normalize,
            "metric": metric.__name__ if callable(metric) else metric
        },
        "warnings": []
    }

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs must not contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs must not contain infinite values.")

def _apply_normalization(
    y_pred: np.ndarray,
    normalize: str
) -> np.ndarray:
    """Apply normalization to predictions."""
    if normalize == "standard":
        y_pred = (y_pred - np.mean(y_pred)) / np.std(y_pred)
    elif normalize == "minmax":
        y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))
    return y_pred

def _compute_recall(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    *,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "binary",
    **kwargs
) -> tuple[float, Dict[str, float]]:
    """Compute recall and additional metrics."""
    if metric == "binary":
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        metrics = {
            "true_positives": tp,
            "false_negatives": fn
        }
    elif callable(metric):
        recall = metric(y_true, y_pred_binary, **kwargs)
        metrics = {}
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return recall, metrics

################################################################################
# f1_score
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 1.0
) -> None:
    """Validate input arrays and parameters."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if beta <= 0:
        raise ValueError("beta must be positive")

def _compute_precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute precision and recall metrics."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {"precision": precision, "recall": recall}

def _compute_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 1.0
) -> float:
    """Compute F1 score with given beta parameter."""
    metrics = _compute_precision_recall(y_true, y_pred)
    precision = metrics["precision"]
    recall = metrics["recall"]

    if (precision + recall) == 0:
        return 0.0

    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return f1

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute additional metrics for diagnostics."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "specificity": specificity,
        "true_positive_rate": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0
    }

def f1_score_compute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 1.0,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute F1 score with configurable parameters.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth (binary) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
    beta : float, default=1.0
        Weighting factor for recall in the F-score.
    custom_metric : callable, optional
        Custom metric function that takes (y_true, y_pred) and returns a float.

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": computed F1 score
        - "metrics": additional metrics dictionary
        - "params_used": parameters used in computation
        - "warnings": any warnings encountered

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> result = f1_score_compute(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred, beta)

    # Compute main result
    f1 = _compute_f1_score(y_true, y_pred, beta)

    # Compute additional metrics
    metrics = _compute_metrics(y_true, y_pred)

    # Add custom metric if provided
    if custom_metric is not None:
        try:
            metrics["custom"] = custom_metric(y_true, y_pred)
        except Exception as e:
            warnings.append(f"Custom metric computation failed: {str(e)}")

    # Prepare output
    return {
        "result": f1,
        "metrics": metrics,
        "params_used": {
            "beta": beta
        },
        "warnings": []
    }

################################################################################
# auc_roc
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def auc_roc_fit(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    normalization: str = 'none',
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'roc_auc',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute the Area Under the ROC Curve (AUC-ROC) for variable importance.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_scores : np.ndarray
        Predicted scores or probabilities.
    normalization : str, optional
        Normalization method for the scores ('none', 'standard', 'minmax', 'robust').
    custom_normalization : Callable, optional
        Custom normalization function.
    metric : str, optional
        Metric to compute ('roc_auc').
    custom_metric : Callable, optional
        Custom metric function.

    Returns:
    --------
    Dict[str, Union[float, Dict[str, float], Dict[str, str]]]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(y_true, y_scores)

    # Normalize scores if required
    normalized_scores = _normalize_scores(y_scores, normalization, custom_normalization)

    # Compute the AUC-ROC
    auc_value = _compute_auc_roc(y_true, normalized_scores, metric, custom_metric)

    # Prepare the output
    result = {
        'result': auc_value,
        'metrics': {'auc_roc': auc_value},
        'params_used': {
            'normalization': normalization,
            'metric': metric
        },
        'warnings': []
    }

    return result

def _validate_inputs(y_true: np.ndarray, y_scores: np.ndarray) -> None:
    """
    Validate the input arrays.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels.
    y_scores : np.ndarray
        Predicted scores or probabilities.

    Raises:
    -------
    ValueError
        If inputs are invalid.
    """
    if y_true.ndim != 1 or y_scores.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length.")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must contain only 0s and 1s.")
    if np.any((y_scores < 0) | (y_scores > 1)):
        raise ValueError("y_scores must be between 0 and 1.")

def _normalize_scores(
    scores: np.ndarray,
    normalization: str = 'none',
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """
    Normalize the scores based on the specified method.

    Parameters:
    -----------
    scores : np.ndarray
        Predicted scores or probabilities.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    custom_normalization : Callable, optional
        Custom normalization function.

    Returns:
    --------
    np.ndarray
        Normalized scores.
    """
    if custom_normalization is not None:
        return custom_normalization(scores)

    if normalization == 'none':
        return scores
    elif normalization == 'standard':
        return (scores - np.mean(scores)) / np.std(scores)
    elif normalization == 'minmax':
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    elif normalization == 'robust':
        return (scores - np.median(scores)) / (np.percentile(scores, 75) - np.percentile(scores, 25))
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_auc_roc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = 'roc_auc',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> float:
    """
    Compute the AUC-ROC value.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels.
    y_scores : np.ndarray
        Predicted scores or probabilities.
    metric : str, optional
        Metric to compute ('roc_auc').
    custom_metric : Callable, optional
        Custom metric function.

    Returns:
    --------
    float
        AUC-ROC value.
    """
    if custom_metric is not None:
        return custom_metric(y_true, y_scores)

    if metric != 'roc_auc':
        raise ValueError(f"Only 'roc_auc' metric is supported for AUC-ROC computation.")

    return _compute_roc_auc(y_true, y_scores)

def _compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute the ROC AUC using the trapezoidal rule.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels.
    y_scores : np.ndarray
        Predicted scores or probabilities.

    Returns:
    --------
    float
        AUC-ROC value.
    """
    # Sort the scores and corresponding true labels
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_scores = y_scores[sorted_indices]
    sorted_labels = y_true[sorted_indices]

    # Compute the ROC curve
    n_positives = np.sum(sorted_labels)
    n_negatives = len(sorted_labels) - n_positives

    if n_positives == 0 or n_negatives == 0:
        raise ValueError("AUC-ROC cannot be computed with all positive or all negative labels.")

    tpr = np.zeros(len(sorted_labels) + 1)
    fpr = np.zeros(len(sorted_labels) + 1)

    tpr[0] = 0.0
    fpr[0] = 0.0

    for i, (score, label) in enumerate(zip(sorted_scores, sorted_labels)):
        tpr[i + 1] = tpr[i]
        fpr[i + 1] = fpr[i]

        if label == 1:
            tpr[i + 1] += 1 / n_positives
        else:
            fpr[i + 1] += 1 / n_negatives

    # Compute the AUC using the trapezoidal rule
    auc = 0.0
    for i in range(len(tpr) - 1):
        auc += (fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2

    return auc

################################################################################
# log_loss
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-15
) -> None:
    """Validate input arrays for log loss calculation."""
    if y_true.ndim != 1 or y_pred.ndim != 2:
        raise ValueError("y_true must be 1D and y_pred must be 2D")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples")
    if np.any(y_pred < 0) or np.any(y_pred > 1):
        raise ValueError("Predicted probabilities must be between 0 and 1")
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

def _compute_log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """Compute the log loss (cross-entropy loss)."""
    n_samples = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred)) / n_samples

def log_loss_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-15,
    sample_weight: Optional[np.ndarray] = None,
    normalize: bool = True
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute the log loss (cross-entropy loss) between true labels and predicted probabilities.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true binary labels (0 or 1)
    y_pred : np.ndarray
        Array of predicted probabilities for class 1 (shape: n_samples, n_classes)
    epsilon : float
        Small value to avoid log(0) which is undefined
    sample_weight : Optional[np.ndarray]
        Individual weights for each sample
    normalize : bool
        If True, the log loss is normalized by n_samples

    Returns:
    --------
    dict
        Dictionary containing the log loss result, metrics, parameters used, and warnings

    Example:
    --------
    >>> y_true = np.array([0, 1, 1])
    >>> y_pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])
    >>> result = log_loss_fit(y_true, y_pred)
    """
    _validate_inputs(y_true, y_pred, epsilon)

    if sample_weight is not None:
        if len(sample_weight) != len(y_true):
            raise ValueError("sample_weight must have the same length as y_true")
        weighted_log_loss = -np.sum(sample_weight * (y_true * np.log(y_pred + epsilon))) / np.sum(sample_weight)
    else:
        weighted_log_loss = _compute_log_loss(y_true, y_pred + epsilon)

    if not normalize:
        weighted_log_loss *= len(y_true)

    return {
        "result": weighted_log_loss,
        "metrics": {"log_loss": weighted_log_loss},
        "params_used": {
            "epsilon": epsilon,
            "sample_weight": sample_weight is not None,
            "normalize": normalize
        },
        "warnings": []
    }

################################################################################
# brier_score
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> None:
    """Validate input arrays."""
    if y_true.shape != y_proba.shape:
        raise ValueError("y_true and y_proba must have the same shape")
    if np.any((y_proba < 0) | (y_proba > 1)):
        raise ValueError("Probabilities must be between 0 and 1")
    if sample_weight is not None:
        if y_true.shape != sample_weight.shape:
            raise ValueError("y_true and sample_weight must have the same shape")
        if np.any(sample_weight < 0):
            raise ValueError("Sample weights must be non-negative")

def _compute_brier_score(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """Compute the Brier score."""
    error = (y_proba - y_true) ** 2
    if sample_weight is not None:
        error = error * sample_weight
    return np.mean(error)

def brier_score_fit(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    normalize: str = 'none',
    metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Compute the Brier score for probability predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true binary labels (0 or 1).
    y_proba : np.ndarray
        Array of predicted probabilities.
    sample_weight : Optional[np.ndarray], default=None
        Individual weights for each sample.
    normalize : str, default='none'
        Normalization method ('none', 'standard', 'minmax').
    metric_func : Optional[Callable], default=None
        Custom metric function if needed.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], str]]
        Dictionary containing:
        - 'result': computed Brier score
        - 'metrics': additional metrics if provided
        - 'params_used': parameters used in computation
        - 'warnings': any warnings generated

    Example
    -------
    >>> y_true = np.array([0, 1, 1])
    >>> y_proba = np.array([0.1, 0.6, 0.7])
    >>> result = brier_score_fit(y_true, y_proba)
    """
    _validate_inputs(y_true, y_proba, sample_weight)

    # Compute Brier score
    brier_score = _compute_brier_score(y_true, y_proba, sample_weight)

    # Apply normalization if needed
    if normalize == 'standard':
        brier_score = (brier_score - np.mean(y_true)) / np.std(y_true)
    elif normalize == 'minmax':
        brier_score = (brier_score - np.min(y_true)) / (np.max(y_true) - np.min(y_true))

    # Compute additional metrics if provided
    metrics = {}
    if metric_func is not None:
        metrics['custom_metric'] = metric_func(y_true, y_proba)

    return {
        'result': brier_score,
        'metrics': metrics,
        'params_used': {
            'normalize': normalize,
            'sample_weight': sample_weight is not None
        },
        'warnings': []
    }

################################################################################
# adjusted_r_squared
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int
) -> None:
    """Validate input arrays and parameters."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-dimensional arrays")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if n_features <= 0:
        raise ValueError("n_features must be positive")

def _compute_adjusted_r_squared(
    r_squared: float,
    n_samples: int,
    n_features: int
) -> float:
    """Compute adjusted R-squared value."""
    return 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - n_features - 1)

def _compute_r_squared(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """Compute R-squared value."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

def adjusted_r_squared_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
    r_squared_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute adjusted R-squared value for model evaluation.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    n_features : int
        Number of features in the model.
    r_squared_func : Optional[Callable]
        Custom R-squared computation function. If None, uses default implementation.

    Returns:
    --------
    Dict containing:
        - result: float
            Adjusted R-squared value.
        - metrics: Dict[str, float]
            Dictionary of computed metrics (R-squared and adjusted R-squared).
        - params_used: Dict[str, str]
            Dictionary of parameters used in computation.
        - warnings: Dict[str, str]
            Any warnings generated during computation.

    Example:
    --------
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> result = adjusted_r_squared_fit(y_true, y_pred, n_features=2)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred, n_features)

    warnings: Dict[str, str] = {}

    # Compute R-squared
    if r_squared_func is None:
        r2 = _compute_r_squared(y_true, y_pred)
    else:
        try:
            r2 = r_squared_func(y_true, y_pred)
        except Exception as e:
            raise ValueError(f"Custom R-squared function failed: {str(e)}")

    # Compute adjusted R-squared
    n_samples = len(y_true)
    adj_r2 = _compute_adjusted_r_squared(r2, n_samples, n_features)

    # Check for potential issues
    if np.isnan(adj_r2):
        warnings["nan_value"] = "Adjusted R-squared is NaN, check input data"
    elif adj_r2 > 1.0:
        warnings["invalid_value"] = "Adjusted R-squared > 1, check model specification"

    return {
        "result": adj_r2,
        "metrics": {
            "r_squared": r2,
            "adjusted_r_squared": adj_r2
        },
        "params_used": {
            "n_features": str(n_features),
            "r_squared_func": "default" if r_squared_func is None else "custom"
        },
        "warnings": warnings
    }

################################################################################
# variance_explained
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def variance_explained_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Compute the variance explained by features in a dataset.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalizer : Optional[Callable]
        Function to normalize features. Default is None.
    metric : Union[str, Callable]
        Metric to use for evaluation. Can be 'mse', 'mae', 'r2', or a custom callable.
    solver : str
        Solver to use. Options: 'closed_form', 'gradient_descent'.
    regularization : Optional[str]
        Regularization type. Options: None, 'l1', 'l2'.
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

    # Choose solver and compute coefficients
    if solver == 'closed_form':
        coefs = _closed_form_solver(X_normalized, y, regularization)
    elif solver == 'gradient_descent':
        coefs = _gradient_descent_solver(X_normalized, y, regularization, tol, max_iter)
    else:
        raise ValueError("Unsupported solver. Choose 'closed_form' or 'gradient_descent'.")

    # Compute variance explained
    result = _compute_variance_explained(X_normalized, y, coefs)

    # Compute metrics
    metrics = _compute_metrics(y, result['predictions'], metric, custom_metric)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
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
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or infinite values.")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("y contains NaN or infinite values.")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    return normalizer(X)

def _closed_form_solver(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Closed form solution for linear regression."""
    if regularization is None:
        coefs = np.linalg.inv(X.T @ X) @ X.T @ y
    elif regularization == 'l2':
        identity = np.eye(X.shape[1])
        coefs = np.linalg.inv(X.T @ X + 0.1 * identity) @ X.T @ y
    else:
        raise ValueError("Unsupported regularization type.")
    return coefs

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent solver for linear regression."""
    n_features = X.shape[1]
    coefs = np.zeros(n_features)
    learning_rate = 0.01
    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, coefs, regularization)
        new_coefs = coefs - learning_rate * gradient
        if np.linalg.norm(new_coefs - coefs) < tol:
            break
        coefs = new_coefs
    return coefs

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    coefs: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Compute gradient for gradient descent."""
    residuals = y - X @ coefs
    gradient = -(X.T @ residuals) / len(y)
    if regularization == 'l2':
        gradient += 0.1 * coefs
    return gradient

def _compute_variance_explained(
    X: np.ndarray,
    y: np.ndarray,
    coefs: np.ndarray
) -> Dict:
    """Compute variance explained by features."""
    predictions = X @ coefs
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - predictions) ** 2)
    variance_explained = 1 - (ss_residual / ss_total)
    return {
        'variance_explained': variance_explained,
        'predictions': predictions
    }

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute evaluation metrics."""
    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        metrics['r2'] = 1 - (ss_residual / ss_total)
    elif callable(metric):
        metrics['custom'] = metric(y_true, y_pred)
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)
    return metrics

################################################################################
# information_gain
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list] = None
) -> None:
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
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

    if feature_names is not None and len(feature_names) != X.shape[1]:
        raise ValueError("feature_names length must match number of features")

def _normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    normalization: str = "standard",
    custom_normalization: Optional[Callable] = None
) -> tuple:
    """Normalize input data."""
    if custom_normalization is not None:
        X = custom_normalization(X)
        y = custom_normalization(y.reshape(-1, 1)).flatten()
    elif normalization == "standard":
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y = (y - np.mean(y)) / np.std(y)
    elif normalization == "minmax":
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == "robust":
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    return X, y

def _compute_entropy(y: np.ndarray) -> float:
    """Compute entropy of target variable."""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def _compute_information_gain(
    X: np.ndarray,
    y: np.ndarray,
    metric: str = "entropy",
    custom_metric: Optional[Callable] = None
) -> np.ndarray:
    """Compute information gain for each feature."""
    if custom_metric is not None:
        return np.array([custom_metric(X[:, i], y) for i in range(X.shape[1])])

    if metric == "entropy":
        base_entropy = _compute_entropy(y)
        gains = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            unique_values = np.unique(X[:, i])
            weighted_entropy = 0.0
            for value in unique_values:
                mask = X[:, i] == value
                if np.sum(mask) > 0:
                    weighted_entropy += (np.sum(mask) / X.shape[0]) * _compute_entropy(y[mask])
            gains[i] = base_entropy - weighted_entropy
        return gains

    raise ValueError(f"Unknown metric: {metric}")

def information_gain_fit(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list] = None,
    normalization: str = "standard",
    metric: str = "entropy",
    custom_metric: Optional[Callable] = None,
    custom_normalization: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, dict]]:
    """
    Compute information gain for each feature in X with respect to y.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target variable of shape (n_samples,)
    feature_names : list, optional
        Names of features, by default None
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust"), by default "standard"
    metric : str, optional
        Metric to use ("entropy", custom callable), by default "entropy"
    custom_metric : Callable, optional
        Custom metric function, by default None
    custom_normalization : Callable, optional
        Custom normalization function, by default None

    Returns
    -------
    Dict[str, Union[np.ndarray, dict]]
        Dictionary containing:
        - "result": array of information gains
        - "metrics": dictionary of computed metrics
        - "params_used": dictionary of parameters used
        - "warnings": list of warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> result = information_gain_fit(X, y)
    """
    _validate_inputs(X, y, feature_names)

    warnings = []
    params_used = {
        "normalization": normalization,
        "metric": metric
    }

    X_norm, y_norm = _normalize_data(X, y, normalization, custom_normalization)

    gains = _compute_information_gain(X_norm, y_norm, metric, custom_metric)

    metrics = {
        "information_gain": gains
    }

    if feature_names is not None:
        metrics["feature_importance"] = dict(zip(feature_names, gains))

    return {
        "result": gains,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# mutual_information
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
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

def _compute_joint_entropy(X: np.ndarray, y: np.ndarray,
                          distance_metric: Callable = np.linalg.norm) -> float:
    """Compute joint entropy between X and y."""
    # Implementation of joint entropy calculation
    pass

def _compute_conditional_entropy(X: np.ndarray, y: np.ndarray,
                                distance_metric: Callable = np.linalg.norm) -> float:
    """Compute conditional entropy of y given X."""
    # Implementation of conditional entropy calculation
    pass

def _compute_mutual_information(X: np.ndarray, y: np.ndarray,
                               distance_metric: Callable = np.linalg.norm) -> float:
    """Compute mutual information between X and y."""
    h_x = _compute_joint_entropy(X, y, distance_metric)
    h_y_given_x = _compute_conditional_entropy(X, y, distance_metric)
    return h_x - h_y_given_x

def mutual_information_fit(X: np.ndarray, y: np.ndarray,
                          distance_metric: Callable = np.linalg.norm,
                          normalize: bool = False) -> Dict[str, Union[float, Dict]]:
    """
    Compute mutual information between features and target.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    distance_metric : Callable, optional
        Distance metric function to use (default: np.linalg.norm)
    normalize : bool, optional
        Whether to normalize the mutual information values (default: False)

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": computed mutual information values
        - "metrics": additional metrics if any
        - "params_used": parameters used in computation
        - "warnings": any warnings generated

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = mutual_information_fit(X, y)
    """
    _validate_inputs(X, y)

    # Compute mutual information for each feature
    mi_values = np.array([_compute_mutual_information(X[:, i], y, distance_metric)
                          for i in range(X.shape[1])])

    if normalize:
        mi_values = (mi_values - np.min(mi_values)) / (np.max(mi_values) - np.min(mi_values))

    return {
        "result": mi_values,
        "metrics": {},
        "params_used": {
            "distance_metric": distance_metric.__name__,
            "normalize": normalize
        },
        "warnings": []
    }

################################################################################
# gini_importance
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

def _compute_gini_impurity(y: np.ndarray) -> float:
    """Compute Gini impurity for a set of labels."""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

def _split_criterion(X: np.ndarray, y: np.ndarray,
                    feature_idx: int,
                    threshold: float) -> float:
    """Compute the Gini importance for a given feature and threshold."""
    left_mask = X[:, feature_idx] <= threshold
    right_mask = ~left_mask

    n_left, n_right = np.sum(left_mask), np.sum(right_mask)
    if n_left == 0 or n_right == 0:
        return float('inf')

    y_left, y_right = y[left_mask], y[right_mask]
    gini_left = _compute_gini_impurity(y_left)
    gini_right = _compute_gini_impurity(y_right)

    weighted_gini = (n_left * gini_left + n_right * gini_right) / len(y)
    return weighted_gini

def _find_best_split(X: np.ndarray, y: np.ndarray,
                    feature_idx: int) -> Dict[str, Union[float, np.ndarray]]:
    """Find the best split for a given feature."""
    thresholds = np.unique(X[:, feature_idx])
    best_gini = float('inf')
    best_threshold = None

    for threshold in thresholds:
        current_gini = _split_criterion(X, y, feature_idx, threshold)
        if current_gini < best_gini:
            best_gini = current_gini
            best_threshold = threshold

    return {
        'best_threshold': best_threshold,
        'best_gini': best_gini
    }

def _compute_feature_importance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Gini importance for all features."""
    n_features = X.shape[1]
    importance = np.zeros(n_features)

    for feature_idx in range(n_features):
        best_split = _find_best_split(X, y, feature_idx)
        importance[feature_idx] = best_split['best_gini']

    return importance

def gini_importance_fit(X: np.ndarray, y: np.ndarray,
                       normalize: str = 'none',
                       metric: Optional[Callable] = None) -> Dict:
    """
    Compute Gini importance for features in a dataset.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : Callable, optional
        Custom metric function

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': array of feature importances
        - 'metrics': dictionary of computed metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> result = gini_importance_fit(X, y)
    """
    _validate_inputs(X, y)

    importance = _compute_feature_importance(X, y)

    if normalize == 'standard':
        mean = np.mean(importance)
        std = np.std(importance)
        importance = (importance - mean) / std
    elif normalize == 'minmax':
        min_val = np.min(importance)
        max_val = np.max(importance)
        importance = (importance - min_val) / (max_val - min_val)

    metrics = {}
    if metric is not None:
        metrics['custom_metric'] = metric(importance)

    return {
        'result': importance,
        'metrics': metrics,
        'params_used': {
            'normalize': normalize,
            'metric': metric.__name__ if metric else None
        },
        'warnings': []
    }

################################################################################
# permutation_importance
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def permutation_importance_fit(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    n_repeats: int = 10,
    random_state: Optional[int] = None,
    n_jobs: int = 1
) -> Dict[str, Any]:
    """
    Compute permutation importance for a given model.

    Parameters:
    -----------
    model : Any
        A fitted model with predict method.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    metric : Union[str, Callable]
        Metric to evaluate model performance. Can be 'mse', 'mae', 'r2',
        or a custom callable.
    n_repeats : int
        Number of permutations for each feature.
    random_state : Optional[int]
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel jobs to run.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Get baseline score
    baseline_score = _compute_baseline_score(model, X, y, metric)

    # Compute permutation importance
    importances = _compute_permutation_importance(
        model, X, y, metric, n_repeats, rng, n_jobs
    )

    # Prepare results
    result = {
        'importances': importances,
        'baseline_score': baseline_score,
    }

    metrics = {
        'metric_used': metric if isinstance(metric, str) else 'custom',
    }

    params_used = {
        'n_repeats': n_repeats,
        'random_state': random_state,
        'n_jobs': n_jobs,
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings,
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data."""
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

def _compute_baseline_score(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Compute baseline score for the model."""
    y_pred = model.predict(X)
    return _evaluate_metric(y, y_pred, metric)

def _compute_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    n_repeats: int,
    rng: np.random.RandomState,
    n_jobs: int
) -> np.ndarray:
    """Compute permutation importance for each feature."""
    n_features = X.shape[1]
    importances = np.zeros(n_features)

    for i in range(n_features):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, i] = rng.permutation(X_permuted[:, i])
            y_pred = model.predict(X_permuted)
            score = _evaluate_metric(y, y_pred, metric)
            scores.append(score)

        importances[i] = np.mean(scores) - _compute_baseline_score(model, X, y, metric)

    return importances

def _evaluate_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Evaluate the given metric."""
    if isinstance(metric, str):
        if metric == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        return metric(y_true, y_pred)

################################################################################
# shapley_value
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def shapley_value_fit(
    model: Callable,
    X: np.ndarray,
    feature_names: Optional[list] = None,
    metric: Union[str, Callable] = "mse",
    normalizer: str = "none",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute Shapley values for feature importance.

    Parameters:
    -----------
    model : Callable
        A callable that takes a subset of features and returns predictions.
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    feature_names : Optional[list], default=None
        List of feature names. If None, features are named as integers.
    metric : Union[str, Callable], default="mse"
        Metric to evaluate model performance. Can be "mse", "mae", "r2", or a custom callable.
    normalizer : str, default="none"
        Normalization method. Options: "none", "standard", "minmax", "robust".
    solver : str, default="closed_form"
        Solver method. Options: "closed_form", "gradient_descent", "newton", "coordinate_descent".
    regularization : Optional[str], default=None
        Regularization method. Options: "l1", "l2", "elasticnet".
    tol : float, default=1e-4
        Tolerance for convergence.
    max_iter : int, default=1000
        Maximum number of iterations for iterative solvers.
    random_state : Optional[int], default=None
        Random seed for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, feature_names)

    # Set random state if provided
    np.random.seed(random_state) if random_state is not None else None

    # Normalize data
    X_normalized = _apply_normalization(X, normalizer)

    # Get feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Initialize results dictionary
    results = {
        "result": {},
        "metrics": {},
        "params_used": {
            "metric": metric,
            "normalizer": normalizer,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    # Compute Shapley values for each feature
    shapley_values = _compute_shapley_values(model, X_normalized, metric, solver, regularization, tol, max_iter)

    # Store results
    for i, name in enumerate(feature_names):
        results["result"][name] = shapley_values[i]

    # Compute and store metrics
    results["metrics"] = _compute_metrics(model, X_normalized, metric)

    return results

def _validate_inputs(X: np.ndarray, feature_names: Optional[list]) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if np.isnan(X).any():
        raise ValueError("X contains NaN values.")
    if feature_names is not None and len(feature_names) != X.shape[1]:
        raise ValueError("Length of feature_names must match number of features in X.")

def _apply_normalization(X: np.ndarray, normalizer: str) -> np.ndarray:
    """Apply normalization to the input data."""
    if normalizer == "standard":
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalizer == "minmax":
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalizer == "robust":
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X

def _compute_shapley_values(
    model: Callable,
    X: np.ndarray,
    metric: Union[str, Callable],
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Compute Shapley values for each feature."""
    n_features = X.shape[1]
    shapley_values = np.zeros(n_features)

    for i in range(n_features):
        # Compute Shapley value for feature i
        shapley_values[i] = _compute_single_shapley_value(model, X, i, metric, solver, regularization, tol, max_iter)

    return shapley_values

def _compute_single_shapley_value(
    model: Callable,
    X: np.ndarray,
    feature_idx: int,
    metric: Union[str, Callable],
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> float:
    """Compute Shapley value for a single feature."""
    n_samples = X.shape[0]
    shapley_value = 0.0

    for m in range(n_samples):
        # Compute the marginal contribution of feature_idx
        subset = X[m, :]
        with_feature = np.insert(subset[:-1], feature_idx, subset[feature_idx])
        without_feature = np.delete(subset, feature_idx)

        # Predict with and without the feature
        pred_with = model(with_feature.reshape(1, -1))
        pred_without = model(without_feature.reshape(1, -1))

        # Compute the difference in metric
        if callable(metric):
            diff = metric(pred_with, pred_without)
        else:
            diff = _compute_metric(pred_with, pred_without, metric)

        shapley_value += diff / n_samples

    return shapley_value

def _compute_metric(
    pred_with: np.ndarray,
    pred_without: np.ndarray,
    metric: str
) -> float:
    """Compute the specified metric."""
    if metric == "mse":
        return np.mean((pred_with - pred_without) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(pred_with - pred_without))
    elif metric == "r2":
        ss_res = np.sum((pred_with - pred_without) ** 2)
        ss_tot = np.sum((pred_with - np.mean(pred_with)) ** 2)
        return 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_metrics(
    model: Callable,
    X: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Compute metrics for the model."""
    if callable(metric):
        return {"custom_metric": metric(model(X), X)}

    metrics = {}
    pred = model(X)

    if metric == "mse":
        metrics["mse"] = np.mean((pred - X) ** 2)
    elif metric == "mae":
        metrics["mae"] = np.mean(np.abs(pred - X))
    elif metric == "r2":
        ss_res = np.sum((pred - X) ** 2)
        ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
        metrics["r2"] = 1 - (ss_res / ss_tot)

    return metrics

################################################################################
# feature_importance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def feature_importance_fit(
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
    weights: Optional[np.ndarray] = None
) -> Dict[str, Union[np.ndarray, float, dict]]:
    """
    Compute feature importance using various statistical and machine learning methods.

    Parameters
    ----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    normalizer : Optional[Callable], default=None
        Function to normalize the input features. If None, no normalization is applied.
    metric : Union[str, Callable], default="mse"
        Metric to evaluate feature importance. Can be "mse", "mae", "r2", or a custom callable.
    distance : Union[str, Callable], default="euclidean"
        Distance metric for feature importance calculation. Can be "euclidean", "manhattan",
        "cosine", or a custom callable.
    solver : str, default="closed_form"
        Solver to use for feature importance calculation. Options include "closed_form",
        "gradient_descent", "newton", or "coordinate_descent".
    regularization : Optional[str], default=None
        Regularization method. Options include "l1", "l2", or "elasticnet".
    tol : float, default=1e-4
        Tolerance for convergence.
    max_iter : int, default=1000
        Maximum number of iterations for iterative solvers.
    custom_metric : Optional[Callable], default=None
        Custom metric function if not using built-in metrics.
    weights : Optional[np.ndarray], default=None
        Sample weights for weighted feature importance calculation.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, dict]]
        Dictionary containing:
        - "result": Feature importance scores.
        - "metrics": Computed metrics.
        - "params_used": Parameters used in the computation.
        - "warnings": Any warnings generated during computation.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = feature_importance_fit(X, y, normalizer=None, metric="mse", solver="closed_form")
    """
    # Validate inputs
    _validate_inputs(X, y, weights)

    # Normalize features if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Prepare parameters for the computation
    params_used = {
        "normalizer": normalizer.__name__ if normalizer else None,
        "metric": metric if isinstance(metric, str) else "custom",
        "distance": distance if isinstance(distance, str) else "custom",
        "solver": solver,
        "regularization": regularization,
        "tol": tol,
        "max_iter": max_iter
    }

    # Compute feature importance based on the chosen solver
    if solver == "closed_form":
        result = _closed_form_feature_importance(X_normalized, y, metric, distance)
    elif solver == "gradient_descent":
        result = _gradient_descent_feature_importance(X_normalized, y, metric, distance,
                                                     regularization, tol, max_iter)
    elif solver == "newton":
        result = _newton_feature_importance(X_normalized, y, metric, distance,
                                           regularization, tol, max_iter)
    elif solver == "coordinate_descent":
        result = _coordinate_descent_feature_importance(X_normalized, y, metric, distance,
                                                       regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, result, metric, custom_metric)

    # Prepare the output dictionary
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": []
    }

    return output

def _validate_inputs(X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if weights is not None:
        if weights.shape[0] != X.shape[0]:
            raise ValueError("weights must have the same length as X.")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to the input features."""
    if normalizer is None:
        return X
    return normalizer(X)

def _closed_form_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable]
) -> np.ndarray:
    """Compute feature importance using closed-form solution."""
    # Placeholder for actual implementation
    return np.random.rand(X.shape[1])

def _gradient_descent_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Compute feature importance using gradient descent."""
    # Placeholder for actual implementation
    return np.random.rand(X.shape[1])

def _newton_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Compute feature importance using Newton's method."""
    # Placeholder for actual implementation
    return np.random.rand(X.shape[1])

def _coordinate_descent_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Compute feature importance using coordinate descent."""
    # Placeholder for actual implementation
    return np.random.rand(X.shape[1])

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    feature_importance: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Compute metrics based on the feature importance."""
    # Placeholder for actual implementation
    return {"metric_value": np.random.rand()}

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
    return 1 - (ss_res / ss_tot)

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
