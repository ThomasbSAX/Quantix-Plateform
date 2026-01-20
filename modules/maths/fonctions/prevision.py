"""
Quantix – Module prevision
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# serie_temporelle
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def serie_temporelle_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalisation: str = "standard",
    metrics: Union[str, List[str], Callable] = ["mse", "mae"],
    solver: str = "gradient_descent",
    regularization: Optional[str] = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Fit a time series model with configurable parameters.

    Parameters:
    -----------
    X : np.ndarray
        Input features (time series data).
    y : np.ndarray
        Target values.
    normalisation : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    metrics : Union[str, List[str], Callable], optional
        Metrics to compute ("mse", "mae", "r2", custom callable).
    solver : str, optional
        Solver method ("closed_form", "gradient_descent", "newton").
    regularization : Optional[str], optional
        Regularization type ("none", "l1", "l2", "elasticnet").
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    custom_metric : Optional[Callable], optional
        Custom metric function.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _apply_normalization(X, y, normalisation)

    # Initialize parameters
    params = _initialize_parameters(X_norm.shape[1])

    # Solve the model
    if solver == "closed_form":
        params = _solve_closed_form(X_norm, y_norm)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(X_norm, y_norm, max_iter, tol)
    elif solver == "newton":
        params = _solve_newton(X_norm, y_norm, max_iter, tol)
    else:
        raise ValueError("Unsupported solver")

    # Apply regularization if specified
    if regularization:
        params = _apply_regularization(params, X_norm, y_norm, regularization)

    # Compute metrics
    metrics_result = _compute_metrics(y, y_norm, params, metrics, custom_metric)

    # Prepare output
    result = {
        "result": params,
        "metrics": metrics_result,
        "params_used": {
            "normalisation": normalisation,
            "metrics": metrics,
            "solver": solver,
            "regularization": regularization,
            "max_iter": max_iter,
            "tol": tol
        },
        "warnings": []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values")

def _apply_normalization(X: np.ndarray, y: np.ndarray, method: str) -> tuple:
    """Apply normalization to input data."""
    if method == "standard":
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif method == "minmax":
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == "robust":
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        X_norm, y_norm = X, y
    return X_norm, y_norm

def _initialize_parameters(n_features: int) -> np.ndarray:
    """Initialize model parameters."""
    return np.zeros(n_features)

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve using closed-form solution."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Solve using gradient descent."""
    params = _initialize_parameters(X.shape[1])
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y) / len(y)
        params -= gradient
        if np.linalg.norm(gradient) < tol:
            break
    return params

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Solve using Newton's method."""
    params = _initialize_parameters(X.shape[1])
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y) / len(y)
        hessian = 2 * X.T @ X / len(y)
        params -= np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(gradient) < tol:
            break
    return params

def _apply_regularization(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply regularization to parameters."""
    if method == "l1":
        params -= 0.1 * np.sign(params)
    elif method == "l2":
        params -= 0.1 * params
    elif method == "elasticnet":
        params -= 0.1 * (np.sign(params) + params)
    return params

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    params: np.ndarray,
    metrics: Union[str, List[str], Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute metrics for the model."""
    result = {}
    if isinstance(metrics, str):
        metrics = [metrics]
    for metric in metrics:
        if metric == "mse":
            result["mse"] = np.mean((y_true - y_pred) ** 2)
        elif metric == "mae":
            result["mae"] = np.mean(np.abs(y_true - y_pred))
        elif metric == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            result["r2"] = 1 - (ss_res / ss_tot)
        elif callable(metric):
            result["custom"] = metric(y_true, y_pred)
    if custom_metric:
        result["custom_metric"] = custom_metric(y_true, y_pred)
    return result

# Example usage:
"""
X_example = np.random.rand(100, 5)
y_example = np.random.rand(100)

result = serie_temporelle_fit(
    X=X_example,
    y=y_example,
    normalisation="standard",
    metrics=["mse", "mae"],
    solver="gradient_descent"
)
"""

################################################################################
# regression_lineaire
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regression_lineaire_fit(
    X: np.ndarray,
    y: np.ndarray,
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
    Fit a linear regression model with configurable options.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalisation : str, optional
        Normalization method: "none", "standard", "minmax", or "robust"
    metric : str or callable, optional
        Metric to evaluate: "mse", "mae", "r2", or custom callable
    solver : str, optional
        Solver method: "closed_form", "gradient_descent", or "newton"
    regularisation : str, optional
        Regularization type: "none", "l1", "l2", or "elasticnet"
    tol : float, optional
        Tolerance for stopping criteria
    max_iter : int, optional
        Maximum number of iterations
    learning_rate : float, optional
        Learning rate for gradient-based solvers
    custom_metric : callable, optional
        Custom metric function
    custom_distance : callable, optional
        Custom distance function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings
    """
    # Input validation
    _validate_inputs(X, y)

    # Normalize data if required
    X_norm, y_norm = _apply_normalisation(X, y, normalisation)

    # Choose solver
    if solver == "closed_form":
        coefs = _closed_form_solution(X_norm, y_norm)
    elif solver == "gradient_descent":
        coefs = _gradient_descent(
            X_norm, y_norm,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol
        )
    else:
        raise ValueError("Unsupported solver")

    # Apply regularization if required
    coefs = _apply_regularisation(coefs, X_norm.shape[1], regularisation)

    # Calculate metrics
    metrics = _calculate_metrics(
        X_norm, y_norm, coefs,
        metric=metric,
        custom_metric=custom_metric
    )

    return {
        "result": coefs,
        "metrics": metrics,
        "params_used": {
            "normalisation": normalisation,
            "metric": metric,
            "solver": solver,
            "regularisation": regularisation
        },
        "warnings": []
    }

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

def _apply_normalisation(
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> tuple:
    """Apply selected normalisation to features and target."""
    if method == "standard":
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / X_std
    elif method == "minmax":
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
    elif method == "robust":
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - X_median) / (X_iqr + 1e-8)
    else:
        X_norm = X.copy()

    return X_norm, y.copy()

def _closed_form_solution(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate closed-form solution for linear regression."""
    XtX = X.T @ X
    if np.linalg.det(XtX) == 0:
        raise ValueError("Matrix is singular")
    return np.linalg.inv(XtX) @ X.T @ y

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> np.ndarray:
    """Perform gradient descent optimization."""
    n_features = X.shape[1]
    coefs = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        gradients = 2 * X.T @ (X @ coefs - y) / len(y)
        coefs -= learning_rate * gradients
        current_loss = np.mean((X @ coefs - y) ** 2)

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return coefs

def _apply_regularisation(
    coefs: np.ndarray,
    n_features: int,
    method: Optional[str]
) -> np.ndarray:
    """Apply selected regularization to coefficients."""
    if method == "l1":
        coefs = np.sign(coefs) * np.maximum(np.abs(coefs) - 1, 0)
    elif method == "l2":
        coefs = coefs / (1 + 0.5 * n_features)
    elif method == "elasticnet":
        coefs = np.sign(coefs) * np.maximum(np.abs(coefs) - 0.5, 0)
        coefs = coefs / (1 + 0.25 * n_features)

    return coefs

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coefs: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Calculate selected metrics."""
    y_pred = X @ coefs

    if metric == "mse":
        return {"mse": np.mean((y - y_pred) ** 2)}
    elif metric == "mae":
        return {"mae": np.mean(np.abs(y - y_pred))}
    elif metric == "r2":
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return {"r2": 1 - ss_res / (ss_tot + 1e-8)}
    elif callable(metric):
        return {f"custom_metric": metric(y, y_pred)}
    elif custom_metric:
        return {f"custom_metric": custom_metric(y, y_pred)}
    else:
        raise ValueError("Unsupported metric")

# Example usage
if __name__ == "__main__":
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])

    result = regression_lineaire_fit(
        X,
        y,
        normalisation="standard",
        metric="mse",
        solver="closed_form"
    )

################################################################################
# arima
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def arima_fit(
    y: np.ndarray,
    order: tuple = (1, 0, 1),
    trend: str = 'n',
    seasonal_order: Optional[tuple] = None,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    solver: str = 'lstsq',
    max_iter: int = 100,
    tol: float = 1e-6,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Fit an ARIMA model to time series data.

    Parameters
    ----------
    y : np.ndarray
        Time series data.
    order : tuple, optional
        (p, d, q) parameters of the ARIMA model.
    trend : str, optional
        Trend component ('n' for no trend, 'c' for constant, 't' for linear).
    seasonal_order : tuple, optional
        (P, D, Q, S) parameters for seasonal component.
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Function to normalize the data.
    metric : str, optional
        Metric for model evaluation ('mse', 'mae', 'r2').
    solver : str, optional
        Solver method ('lstsq', 'gradient_descent').
    max_iter : int, optional
        Maximum number of iterations for iterative solvers.
    tol : float, optional
        Tolerance for convergence.
    custom_metric : Callable[[np.ndarray, np.ndarray], float], optional
        Custom metric function.

    Returns
    -------
    Dict[str, Union[Dict, float, str]]
        Dictionary containing model results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> y = np.array([1.0, 2.0, 3.0, 4.0])
    >>> result = arima_fit(y)
    """
    # Validate inputs
    _validate_inputs(y, order, trend, seasonal_order)

    # Normalize data
    y_normalized = normalizer(y)

    # Prepare parameters
    params_used = {
        'order': order,
        'trend': trend,
        'seasonal_order': seasonal_order,
        'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
        'metric': metric,
        'solver': solver
    }

    # Fit model
    if solver == 'lstsq':
        coefficients = _fit_lstsq(y_normalized, order, trend, seasonal_order)
    elif solver == 'gradient_descent':
        coefficients = _fit_gradient_descent(y_normalized, order, trend, seasonal_order, max_iter, tol)
    else:
        raise ValueError("Unsupported solver method.")

    # Calculate metrics
    predictions = _predict(y_normalized, coefficients, order, trend, seasonal_order)
    metrics = _calculate_metrics(y_normalized, predictions, metric, custom_metric)

    # Return results
    return {
        'result': {'coefficients': coefficients},
        'metrics': metrics,
        'params_used': params_used,
        'warnings': _check_warnings(y_normalized, predictions)
    }

def _validate_inputs(
    y: np.ndarray,
    order: tuple,
    trend: str,
    seasonal_order: Optional[tuple]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(y, np.ndarray):
        raise TypeError("Input y must be a numpy array.")
    if len(y) < 1:
        raise ValueError("Input y must not be empty.")
    if len(order) != 3:
        raise ValueError("Order must be a tuple of length 3 (p, d, q).")
    if trend not in ['n', 'c', 't']:
        raise ValueError("Trend must be one of 'n', 'c', or 't'.")
    if seasonal_order is not None and len(seasonal_order) != 4:
        raise ValueError("Seasonal order must be a tuple of length 4 (P, D, Q, S) or None.")

def _fit_lstsq(
    y: np.ndarray,
    order: tuple,
    trend: str,
    seasonal_order: Optional[tuple]
) -> np.ndarray:
    """Fit ARIMA model using least squares."""
    # Placeholder for actual implementation
    return np.zeros(order[0] + order[1] + order[2])

def _fit_gradient_descent(
    y: np.ndarray,
    order: tuple,
    trend: str,
    seasonal_order: Optional[tuple],
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Fit ARIMA model using gradient descent."""
    # Placeholder for actual implementation
    return np.zeros(order[0] + order[1] + order[2])

def _predict(
    y: np.ndarray,
    coefficients: np.ndarray,
    order: tuple,
    trend: str,
    seasonal_order: Optional[tuple]
) -> np.ndarray:
    """Generate predictions from fitted ARIMA model."""
    # Placeholder for actual implementation
    return np.zeros_like(y)

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)
    if metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    return metrics

def _check_warnings(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isnan(y_pred)):
        warnings.append("Predictions contain NaN values.")
    if np.any(np.isinf(y_pred)):
        warnings.append("Predictions contain infinite values.")
    return warnings

################################################################################
# prophet
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def prophet_fit(
    y: np.ndarray,
    X: Optional[np.ndarray] = None,
    normalizer: str = "standard",
    metric: Union[str, Callable] = "mse",
    solver: str = "gradient_descent",
    regularization: Optional[str] = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None,
    verbose: bool = False
) -> Dict:
    """
    Fit a Prophet model for time series forecasting.

    Parameters:
    -----------
    y : np.ndarray
        Target values.
    X : Optional[np.ndarray], default=None
        Feature matrix. If None, only intercept is used.
    normalizer : str, default="standard"
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : Union[str, Callable], default="mse"
        Metric to evaluate model performance: "mse", "mae", "r2", or custom callable.
    solver : str, default="gradient_descent"
        Solver method: "closed_form", "gradient_descent", "newton", or "coordinate_descent".
    regularization : Optional[str], default=None
        Regularization type: "none", "l1", "l2", or "elasticnet".
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    custom_metric : Optional[Callable], default=None
        Custom metric function if not using built-in metrics.
    custom_distance : Optional[Callable], default=None
        Custom distance function for solver.
    verbose : bool, default=False
        Whether to print progress information.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Example:
    --------
    >>> y = np.array([1.0, 2.0, 3.0])
    >>> X = np.array([[1], [2], [3]])
    >>> result = prophet_fit(y, X)
    """
    # Validate inputs
    _validate_inputs(y, X)

    # Normalize data
    y_norm, X_norm = _normalize_data(y, X, normalizer)

    # Initialize parameters
    params = _initialize_parameters(X_norm)

    # Choose solver and fit model
    if solver == "closed_form":
        params = _closed_form_solver(X_norm, y_norm)
    elif solver == "gradient_descent":
        params = _gradient_descent_solver(
            X_norm, y_norm, max_iter=max_iter, tol=tol,
            regularization=regularization,
            custom_distance=custom_distance
        )
    elif solver == "newton":
        params = _newton_solver(X_norm, y_norm)
    elif solver == "coordinate_descent":
        params = _coordinate_descent_solver(
            X_norm, y_norm, max_iter=max_iter,
            tol=tol, regularization=regularization
        )
    else:
        raise ValueError("Invalid solver specified.")

    # Calculate predictions and metrics
    y_pred = _predict(X_norm, params)
    metrics = _calculate_metrics(y_norm, y_pred, metric, custom_metric)

    # Denormalize predictions
    y_pred_denorm = _denormalize_data(y_pred, normalizer)

    # Prepare output
    result = {
        "result": y_pred_denorm,
        "metrics": metrics,
        "params_used": params,
        "warnings": []
    }

    return result

def _validate_inputs(y: np.ndarray, X: Optional[np.ndarray]) -> None:
    """Validate input data."""
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError("y must be a 1D numpy array.")
    if X is not None and (not isinstance(X, np.ndarray) or X.ndim != 2):
        raise ValueError("X must be a 2D numpy array or None.")
    if X is not None and y.shape[0] != X.shape[0]:
        raise ValueError("y and X must have the same number of samples.")

def _normalize_data(
    y: np.ndarray,
    X: Optional[np.ndarray],
    method: str
) -> tuple:
    """Normalize data based on specified method."""
    if method == "none":
        return y, X
    elif method == "standard":
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_norm = (y - y_mean) / y_std
        if X is not None:
            X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        else:
            X_norm = None
    elif method == "minmax":
        y_min = np.min(y)
        y_max = np.max(y)
        y_norm = (y - y_min) / (y_max - y_min)
        if X is not None:
            X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        else:
            X_norm = None
    elif method == "robust":
        y_median = np.median(y)
        y_iqr = np.percentile(y, 75) - np.percentile(y, 25)
        y_norm = (y - y_median) / y_iqr
        if X is not None:
            X_median = np.median(X, axis=0)
            X_iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
            X_norm = (X - X_median) / X_iqr
        else:
            X_norm = None
    else:
        raise ValueError("Invalid normalization method specified.")
    return y_norm, X_norm

def _initialize_parameters(X: Optional[np.ndarray]) -> np.ndarray:
    """Initialize model parameters."""
    if X is None or X.shape[1] == 0:
        return np.array([np.mean(X)]) if X is not None else np.array([0.0])
    return np.zeros(X.shape[1] + 1)

def _closed_form_solver(
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Closed form solution for linear regression."""
    X_with_intercept = _add_intercept(X)
    params = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
    return params

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    tol: float,
    regularization: Optional[str],
    custom_distance: Optional[Callable]
) -> np.ndarray:
    """Gradient descent solver."""
    X_with_intercept = _add_intercept(X)
    params = np.zeros(X_with_intercept.shape[1])
    prev_params = None

    for _ in range(max_iter):
        gradients = _compute_gradients(X_with_intercept, y, params, regularization)
        if custom_distance is not None:
            gradients = custom_distance(gradients)
        params -= 0.01 * gradients
        if prev_params is not None and np.linalg.norm(params - prev_params) < tol:
            break
        prev_params = params.copy()

    return params

def _newton_solver(
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Newton's method solver."""
    X_with_intercept = _add_intercept(X)
    params = np.zeros(X_with_intercept.shape[1])

    for _ in range(100):
        gradients = _compute_gradients(X_with_intercept, y, params)
        hessian = X_with_intercept.T @ X_with_intercept
        params -= np.linalg.inv(hessian) @ gradients

    return params

def _coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    tol: float,
    regularization: Optional[str]
) -> np.ndarray:
    """Coordinate descent solver."""
    X_with_intercept = _add_intercept(X)
    params = np.zeros(X_with_intercept.shape[1])

    for _ in range(max_iter):
        for i in range(params.shape[0]):
            X_i = X_with_intercept[:, i:i+1]
            residuals = y - (X_with_intercept @ params) + X_i * params[i]
            if regularization == "l1":
                params[i] = np.sign(X_i.T @ residuals) * np.maximum(
                    0, np.abs(X_i.T @ residuals) - 1
                ) / (X_i.T @ X_i)
            elif regularization == "l2":
                params[i] = (X_i.T @ residuals) / (X_i.T @ X_i + 1)
            else:
                params[i] = (X_i.T @ residuals) / (X_i.T @ X_i)

    return params

def _compute_gradients(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Compute gradients for optimization."""
    residuals = y - X @ params
    gradients = -(X.T @ residuals) / len(y)
    if regularization == "l1":
        gradients += np.sign(params[1:])  # Skip intercept
    elif regularization == "l2":
        gradients += 2 * params[1:]  # Skip intercept
    elif regularization == "elasticnet":
        gradients += np.sign(params[1:]) + 2 * params[1:]  # Skip intercept
    return gradients

def _add_intercept(X: np.ndarray) -> np.ndarray:
    """Add intercept term to feature matrix."""
    return np.column_stack([np.ones(X.shape[0]), X])

def _predict(
    X: np.ndarray,
    params: np.ndarray
) -> np.ndarray:
    """Make predictions using fitted parameters."""
    X_with_intercept = _add_intercept(X)
    return X_with_intercept @ params

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate evaluation metrics."""
    metrics = {}
    if metric == "mse":
        metrics["mse"] = np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        metrics["mae"] = np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics["r2"] = 1 - (ss_res / ss_tot)
    elif callable(metric):
        metrics["custom"] = metric(y_true, y_pred)
    if custom_metric is not None:
        metrics["custom"] = custom_metric(y_true, y_pred)
    return metrics

def _denormalize_data(
    y_norm: np.ndarray,
    method: str
) -> np.ndarray:
    """Denormalize data based on specified method."""
    if method == "none":
        return y_norm
    elif method == "standard":
        y_mean = np.mean(y_norm)
        y_std = np.std(y_norm)
        return y_norm * y_std + y_mean
    elif method == "minmax":
        y_min = np.min(y_norm)
        y_max = np.max(y_norm)
        return y_norm * (y_max - y_min) + y_min
    elif method == "robust":
        y_median = np.median(y_norm)
        y_iqr = np.percentile(y_norm, 75) - np.percentile(y_norm, 25)
        return y_norm * y_iqr + y_median
    else:
        raise ValueError("Invalid normalization method specified.")

################################################################################
# lstm
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def lstm_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_units: int = 50,
    n_layers: int = 1,
    activation: str = 'tanh',
    optimizer: str = 'adam',
    loss: str = 'mse',
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 0.001,
    validation_split: float = 0.2,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit an LSTM model for time series forecasting.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_timesteps, n_features).
    y : np.ndarray
        Target data of shape (n_samples, n_outputs).
    n_units : int, optional
        Number of units in each LSTM layer.
    n_layers : int, optional
        Number of LSTM layers.
    activation : str, optional
        Activation function for the LSTM layers ('tanh', 'relu').
    optimizer : str, optional
        Optimizer to use ('adam', 'sgd').
    loss : str, optional
        Loss function to minimize ('mse', 'mae').
    batch_size : int, optional
        Batch size for training.
    epochs : int, optional
        Number of training epochs.
    learning_rate : float, optional
        Learning rate for the optimizer.
    validation_split : float, optional
        Fraction of data to use for validation.
    normalizer : Callable, optional
        Function to normalize the input data.
    custom_metric : Callable, optional
        Custom metric function to evaluate the model.
    verbose : bool, optional
        Whether to print training progress.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing the fitted model parameters, metrics, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if a normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize model parameters
    params = _initialize_lstm_params(n_units, n_layers, X.shape[2], y.shape[1])

    # Train the model
    history = _train_lstm(
        X_normalized, y,
        params, n_units, n_layers, activation, optimizer,
        loss, batch_size, epochs, learning_rate,
        validation_split, verbose
    )

    # Calculate metrics
    metrics = _calculate_metrics(y, history['y_pred'], loss, custom_metric)

    # Prepare the output
    result = {
        'result': history['y_pred'],
        'metrics': metrics,
        'params_used': {
            'n_units': n_units,
            'n_layers': n_layers,
            'activation': activation,
            'optimizer': optimizer,
            'loss': loss,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate
        },
        'warnings': history.get('warnings', [])
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate the input data shapes and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays.")
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, n_timesteps, n_features).")
    if y.ndim != 2:
        raise ValueError("y must have shape (n_samples, n_outputs).")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X and y must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("X and y must not contain infinite values.")

def _apply_normalization(
    X: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to the input data if a normalizer is provided."""
    if normalizer is not None:
        return normalizer(X)
    return X

def _initialize_lstm_params(
    n_units: int,
    n_layers: int,
    n_features: int,
    n_outputs: int
) -> Dict[str, np.ndarray]:
    """Initialize the LSTM model parameters."""
    params = {
        'W_f': np.random.randn(n_layers, n_units, n_features + n_units),
        'b_f': np.zeros((n_layers, n_units)),
        'W_i': np.random.randn(n_layers, n_units, n_features + n_units),
        'b_i': np.zeros((n_layers, n_units)),
        'W_c': np.random.randn(n_layers, n_units, n_features + n_units),
        'b_c': np.zeros((n_layers, n_units)),
        'W_o': np.random.randn(n_layers, n_units, n_features + n_units),
        'b_o': np.zeros((n_layers, n_units)),
        'W_y': np.random.randn(n_outputs, n_units),
        'b_y': np.zeros((n_outputs,))
    }
    return params

def _train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, np.ndarray],
    n_units: int,
    n_layers: int,
    activation: str,
    optimizer: str,
    loss: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    validation_split: float,
    verbose: bool
) -> Dict[str, Union[np.ndarray, List[float]]]:
    """Train the LSTM model."""
    n_samples = X.shape[0]
    val_samples = int(n_samples * validation_split)
    X_train, y_train = X[:-val_samples], y[:-val_samples]
    X_val, y_val = X[-val_samples:], y[-val_samples:]

    history = {
        'loss': [],
        'val_loss': [],
        'y_pred': np.zeros_like(y)
    }

    for epoch in range(epochs):
        # Shuffle the training data
        indices = np.random.permutation(X_train.shape[0])
        X_train = X_train[indices]
        y_train = y_train[indices]

        # Mini-batch training
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Forward pass
            y_pred, cache = _forward_lstm(X_batch, params, activation)

            # Compute loss
            current_loss = _compute_loss(y_pred, y_batch, loss)
            history['loss'].append(current_loss)

            # Backward pass and update parameters
            grads = _backward_lstm(y_pred, y_batch, cache, loss)
            params = _update_parameters(params, grads, learning_rate, optimizer)

        # Validation
        val_pred = _forward_lstm(X_val, params, activation)[0]
        val_loss = _compute_loss(val_pred, y_val, loss)
        history['val_loss'].append(val_loss)

        # Predict on the entire dataset
        history['y_pred'] = _forward_lstm(X, params, activation)[0]

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {current_loss:.4f}, Val Loss: {val_loss:.4f}")

    return history

def _forward_lstm(
    X: np.ndarray,
    params: Dict[str, np.ndarray],
    activation: str
) -> tuple[np.ndarray, Dict]:
    """Forward pass of the LSTM model."""
    cache = {'h': [], 'c': []}
    h_prev = np.zeros((X.shape[1], params['W_f'][0].shape[0]))
    c_prev = np.zeros((X.shape[1], params['W_f'][0].shape[0]))

    for t in range(X.shape[1]):
        x_t = X[:, t, :]
        combined = np.concatenate([x_t, h_prev], axis=1)

        f_t = _sigmoid(np.dot(combined, params['W_f'][0].T) + params['b_f'][0])
        i_t = _sigmoid(np.dot(combined, params['W_i'][0].T) + params['b_i'][0])
        c_tilde = _activation_func(np.dot(combined, params['W_c'][0].T) + params['b_c'][0], activation)
        c_t = f_t * c_prev + i_t * c_tilde
        o_t = _sigmoid(np.dot(combined, params['W_o'][0].T) + params['b_o'][0])
        h_t = o_t * _activation_func(c_t, activation)

        cache['h'].append(h_prev)
        cache['c'].append(c_prev)
        h_prev = h_t
        c_prev = c_t

    y_pred = np.dot(h_prev, params['W_y'].T) + params['b_y']
    return y_pred, cache

def _backward_lstm(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    cache: Dict[str, List[np.ndarray]],
    loss: str
) -> Dict[str, np.ndarray]:
    """Backward pass of the LSTM model."""
    grads = {
        'W_f': np.zeros_like(params['W_f']),
        'b_f': np.zeros_like(params['b_f']),
        'W_i': np.zeros_like(params['W_i']),
        'b_i': np.zeros_like(params['b_i']),
        'W_c': np.zeros_like(params['W_c']),
        'b_c': np.zeros_like(params['b_c']),
        'W_o': np.zeros_like(params['W_o']),
        'b_o': np.zeros_like(params['b_o']),
        'W_y': np.zeros_like(params['W_y']),
        'b_y': np.zeros_like(params['b_y'])
    }

    # Compute gradients for the output layer
    if loss == 'mse':
        dy = 2 * (y_pred - y_true) / y_true.shape[0]
    elif loss == 'mae':
        dy = np.sign(y_pred - y_true) / y_true.shape[0]
    else:
        raise ValueError("Unsupported loss function.")

    grads['W_y'] = np.dot(dy.T, cache['h'][-1])
    grads['b_y'] = np.sum(dy, axis=0)

    # Backpropagate through the LSTM layers
    dh = np.dot(dy, params['W_y'])
    for t in reversed(range(len(cache['h']))):
        h_prev = cache['h'][t]
        c_prev = cache['c'][t]

        # Gradients for the output gate
        do_t = dh * _activation_func(c_prev, activation) * (1 - _sigmoid(np.dot(np.concatenate([X[:, t, :], h_prev], axis=1), params['W_o'][0].T) + params['b_o'][0]))
        grads['W_o'] += np.dot(do_t.T, np.concatenate([X[:, t, :], h_prev]))
        grads['b_o'] += np.sum(do_t, axis=0)

        # Gradients for the cell state
        dc_t = dh * params['W_o'][0] * _activation_func(c_prev, activation) * (1 - _activation_func(c_prev, activation))
        df_t = dc_t * c_prev
        di_t = dc_t * params['W_c'][0]
        dc_tilde = dc_t * params['W_i'][0]

        # Gradients for the forget gate
        df_t = df_t * _sigmoid(np.dot(np.concatenate([X[:, t, :], h_prev], axis=1), params['W_f'][0].T) + params['b_f'][0]) * (1 - _sigmoid(np.dot(np.concatenate([X[:, t, :], h_prev], axis=1), params['W_f'][0].T) + params['b_f'][0]))
        grads['W_f'] += np.dot(df_t.T, np.concatenate([X[:, t, :], h_prev]))
        grads['b_f'] += np.sum(df_t, axis=0)

        # Gradients for the input gate
        di_t = di_t * _sigmoid(np.dot(np.concatenate([X[:, t, :], h_prev], axis=1), params['W_i'][0].T) + params['b_i'][0]) * (1 - _sigmoid(np.dot(np.concatenate([X[:, t, :], h_prev], axis=1), params['W_i'][0].T) + params['b_i'][0]))
        grads['W_i'] += np.dot(di_t.T, np.concatenate([X[:, t, :], h_prev]))
        grads['b_i'] += np.sum(di_t, axis=0)

        # Gradients for the candidate cell state
        dc_tilde = dc_tilde * _activation_func(np.dot(np.concatenate([X[:, t, :], h_prev], axis=1), params['W_c'][0].T) + params['b_c'][0]) * (1 - _activation_func(np.dot(np.concatenate([X[:, t, :], h_prev], axis=1), params['W_c'][0].T) + params['b_c'][0], activation))
        grads['W_c'] += np.dot(dc_tilde.T, np.concatenate([X[:, t, :], h_prev]))
        grads['b_c'] += np.sum(dc_tilde, axis=0)

        # Update dh for the previous timestep
        dh = np.dot(dc_t * params['W_f'][0] + dc_t * params['W_i'][0], params['W_o'][0].T)

    return grads

def _update_parameters(
    params: Dict[str, np.ndarray],
    grads: Dict[str, np.ndarray],
    learning_rate: float,
    optimizer: str
) -> Dict[str, np.ndarray]:
    """Update the model parameters using the specified optimizer."""
    if optimizer == 'adam':
        for key in params:
            params[key] -= learning_rate * grads[key]
    elif optimizer == 'sgd':
        for key in params:
            params[key] -= learning_rate * grads[key]
    else:
        raise ValueError("Unsupported optimizer.")
    return params

def _compute_loss(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    loss: str
) -> float:
    """Compute the specified loss function."""
    if loss == 'mse':
        return np.mean((y_pred - y_true) ** 2)
    elif loss == 'mae':
        return np.mean(np.abs(y_pred - y_true))
    else:
        raise ValueError("Unsupported loss function.")

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate the specified metrics."""
    metrics = {
        'loss': _compute_loss(y_pred, y_true, loss)
    }

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_pred, y_true)

    return metrics

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def _activation_func(x: np.ndarray, activation: str) -> np.ndarray:
    """Activation function."""
    if activation == 'tanh':
        return np.tanh(x)
    elif activation == 'relu':
        return np.maximum(0, x)
    else:
        raise ValueError("Unsupported activation function.")

################################################################################
# xgboost
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

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

def normalize_data(X: np.ndarray, y: np.ndarray, normalization: str = "none") -> tuple:
    """Normalize data based on specified method."""
    if normalization == "standard":
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / (X_std + 1e-8)
        y_normalized = (y - np.mean(y)) / (np.std(y) + 1e-8)
    elif normalization == "minmax":
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
        y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
    elif normalization == "robust":
        X_median = np.median(X, axis=0)
        X_q75 = np.percentile(X, 75, axis=0)
        X_q25 = np.percentile(X, 25, axis=0)
        X_normalized = (X - X_median) / (X_q75 - X_q25 + 1e-8)
        y_median = np.median(y)
        y_q75 = np.percentile(y, 75)
        y_q25 = np.percentile(y, 25)
        y_normalized = (y - y_median) / (y_q75 - y_q25 + 1e-8)
    else:
        X_normalized, y_normalized = X.copy(), y.copy()
    return X_normalized, y_normalized

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str = "mse", custom_metric: Optional[Callable] = None) -> float:
    """Compute specified metric between true and predicted values."""
    if custom_metric is not None:
        return custom_metric(y_true, y_pred)
    if metric == "mse":
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

def xgboost_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.3,
    max_depth: int = 6,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    normalization: str = "none",
    metric: str = "mse",
    custom_metric: Optional[Callable] = None,
    early_stopping_rounds: Optional[int] = None,
    validation_split: float = 0.1
) -> Dict[str, Any]:
    """
    Fit XGBoost model to the data.

    Example:
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = xgboost_fit(X, y)
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization)

    # Initialize parameters
    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree
    }

    # Placeholder for actual XGBoost implementation
    # This would be replaced with the actual algorithm implementation
    y_pred = np.random.rand(y.shape[0])  # Dummy prediction

    # Compute metrics
    metrics = {
        "train_metric": compute_metric(y_norm, y_pred, metric, custom_metric)
    }

    # Prepare output
    result = {
        "result": y_pred,
        "metrics": metrics,
        "params_used": params,
        "warnings": []
    }

    return result

################################################################################
# random_forest
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def random_forest_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Union[int, float] = "auto",
    bootstrap: bool = True,
    criterion: str = "mse",
    metric: Callable[[np.ndarray, np.ndarray], float] = None,
    random_state: Optional[int] = None,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Fit a random forest model to the data.

    Parameters:
    -----------
    X : np.ndarray
        Training input samples of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, n_outputs).
    n_estimators : int, default=100
        The number of trees in the forest.
    max_depth : int, default=None
        The maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    max_features : int or float, default="auto"
        The number of features to consider when looking for the best split.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    criterion : str, default="mse"
        The function to measure the quality of a split. Supported criteria are "mse" for regression and "gini" or "entropy" for classification.
    metric : Callable[[np.ndarray, np.ndarray], float], default=None
        A custom metric function to evaluate the model. If None, a default metric is used based on the problem type.
    random_state : int, default=None
        Controls both the randomness of the bootstrapping of the samples and the sampling of the features.
    n_jobs : int, default=-1
        The number of jobs to run in parallel. -1 means using all processors.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the fitted model, metrics, parameters used, and any warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state for reproducibility
    rng = np.random.RandomState(random_state)

    # Initialize parameters
    params_used = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "criterion": criterion,
        "random_state": random_state,
        "n_jobs": n_jobs
    }

    # Determine the default metric if not provided
    if metric is None:
        if criterion in ["gini", "entropy"]:
            metric = _default_classification_metric
        else:
            metric = _default_regression_metric

    # Fit the random forest
    trees = []
    for i in range(n_estimators):
        tree = _fit_single_tree(
            X, y,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            criterion=criterion,
            random_state=rng
        )
        trees.append(tree)

    # Calculate metrics on the training data
    y_pred = _predict(trees, X)
    train_metric = metric(y, y_pred)

    # Prepare the output dictionary
    result = {
        "trees": trees,
        "params_used": params_used,
        "metrics": {"train_metric": train_metric},
        "warnings": []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate the input data.

    Parameters:
    -----------
    X : np.ndarray
        Training input samples.
    y : np.ndarray
        Target values.

    Raises:
    -------
    ValueError
        If the inputs are invalid.
    """
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

def _fit_single_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: Optional[int],
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: Union[int, float],
    bootstrap: bool,
    criterion: str,
    random_state: np.random.RandomState
) -> Dict[str, Any]:
    """
    Fit a single decision tree.

    Parameters:
    -----------
    X : np.ndarray
        Training input samples.
    y : np.ndarray
        Target values.
    max_depth : int, default=None
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    max_features : int or float, default="auto"
        The number of features to consider when looking for the best split.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    criterion : str, default="mse"
        The function to measure the quality of a split.
    random_state : np.random.RandomState
        Random state for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        A dictionary representing the fitted tree.
    """
    # Implement the logic to fit a single decision tree
    pass

def _predict(trees: list, X: np.ndarray) -> np.ndarray:
    """
    Predict using the fitted random forest.

    Parameters:
    -----------
    trees : list
        List of fitted decision trees.
    X : np.ndarray
        Input samples.

    Returns:
    --------
    np.ndarray
        Predicted values.
    """
    # Implement the logic to make predictions using the random forest
    pass

def _default_regression_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Default regression metric (MSE).

    Parameters:
    -----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns:
    --------
    float
        Mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)

def _default_classification_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Default classification metric (accuracy).

    Parameters:
    -----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns:
    --------
    float
        Accuracy.
    """
    return np.mean(y_true == y_pred)

################################################################################
# gradient_boosting
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def gradient_boosting_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    loss: str = 'mse',
    criterion: Optional[Callable] = None,
    normalizer: Optional[Callable] = None,
    solver: str = 'gradient_descent',
    max_depth: int = 3,
    min_samples_split: int = 2,
    tol: float = 1e-4,
    n_iter_no_change: int = None,
    random_state: Optional[int] = None
) -> Dict:
    """
    Fit a gradient boosting model to the data.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    n_estimators : int, optional
        Number of boosting stages to be run.
    learning_rate : float, optional
        Shrinkage factor for each tree's contribution.
    loss : str or callable, optional
        Loss function to be optimized. Can be 'mse', 'mae', or a custom callable.
    criterion : callable, optional
        Function to measure the quality of a split. If None, uses default for loss.
    normalizer : callable, optional
        Function to normalize the data. If None, no normalization is applied.
    solver : str, optional
        Solver to use for optimization. Can be 'gradient_descent', 'newton', or 'coordinate_descent'.
    max_depth : int, optional
        Maximum depth of the individual regression estimators.
    min_samples_split : int, optional
        Minimum number of samples required to split an internal node.
    tol : float, optional
        Tolerance for the stopping criterion.
    n_iter_no_change : int, optional
        Number of iterations with no improvement after which training will be stopped.
    random_state : int, optional
        Controls the random resampling of the data.

    Returns:
    --------
    dict
        A dictionary containing the fitted model, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Normalize data if a normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize the model
    model = {
        'estimators': [],
        'learning_rate': learning_rate,
        'loss': loss
    }

    # Initialize predictions with the mean of y for regression tasks
    y_pred = np.full_like(y, np.mean(y))

    # Main boosting loop
    for i in range(n_estimators):
        # Compute the negative gradient (pseudo-residuals)
        residuals = _compute_residuals(y, y_pred, loss)

        # Fit a weak learner to the residuals
        estimator = _fit_weak_learner(
            X_normalized,
            residuals,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=rng
        )

        # Update predictions
        y_pred += learning_rate * estimator.predict(X_normalized)

        # Store the estimator
        model['estimators'].append(estimator)

        # Check for early stopping
        if n_iter_no_change is not None:
            if _check_early_stopping(y, y_pred, tol, n_iter_no_change):
                break

    # Compute metrics
    metrics = _compute_metrics(y, y_pred, loss)

    return {
        'result': model,
        'metrics': metrics,
        'params_used': {
            'n_estimators': i + 1,
            'learning_rate': learning_rate,
            'loss': loss,
            'solver': solver,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate the input data."""
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

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to the data if a normalizer is provided."""
    if normalizer is not None:
        return normalizer(X)
    return X

def _compute_residuals(y: np.ndarray, y_pred: np.ndarray, loss: Union[str, Callable]) -> np.ndarray:
    """Compute the residuals based on the specified loss function."""
    if isinstance(loss, str):
        if loss == 'mse':
            return y - y_pred
        elif loss == 'mae':
            return np.sign(y - y_pred)
    else:
        return loss(y, y_pred)
    raise ValueError(f"Unsupported loss function: {loss}")

def _fit_weak_learner(
    X: np.ndarray,
    y: np.ndarray,
    criterion: Optional[Callable] = None,
    max_depth: int = 3,
    min_samples_split: int = 2,
    random_state: Optional[np.random.RandomState] = None
) -> 'WeakLearner':
    """Fit a weak learner (e.g., decision tree) to the data."""
    # This is a placeholder for the actual weak learner implementation
    class WeakLearner:
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = X
            self.y = y

        def predict(self, X: np.ndarray) -> np.ndarray:
            # Placeholder for prediction logic
            return np.zeros(X.shape[0])

    return WeakLearner(X, y)

def _check_early_stopping(y: np.ndarray, y_pred: np.ndarray, tol: float, n_iter_no_change: int) -> bool:
    """Check if early stopping criteria are met."""
    # Placeholder for early stopping logic
    return False

def _compute_metrics(y: np.ndarray, y_pred: np.ndarray, loss: Union[str, Callable]) -> Dict:
    """Compute metrics based on the specified loss function."""
    metrics = {}
    if isinstance(loss, str):
        if loss == 'mse':
            metrics['mse'] = np.mean((y - y_pred) ** 2)
        elif loss == 'mae':
            metrics['mae'] = np.mean(np.abs(y - y_pred))
    else:
        metrics['custom_loss'] = loss(y, y_pred)
    return metrics

################################################################################
# k_nearest_neighbors
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input arrays for KNN.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)

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
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def standard_normalize(X: np.ndarray) -> np.ndarray:
    """Standard normalization (mean=0, std=1)."""
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def minmax_normalize(X: np.ndarray) -> np.ndarray:
    """Min-max normalization (range [0,1])."""
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

def robust_normalize(X: np.ndarray) -> np.ndarray:
    """Robust normalization (median and IQR)."""
    return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))

def euclidean_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Euclidean distance between rows of X and Y."""
    return np.sqrt(np.sum((X[:, np.newaxis] - Y) ** 2, axis=2))

def manhattan_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Manhattan distance between rows of X and Y."""
    return np.sum(np.abs(X[:, np.newaxis] - Y), axis=2)

def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Cosine distance between rows of X and Y."""
    dot_products = np.dot(X, Y.T)
    norms_X = np.linalg.norm(X, axis=1)[:, np.newaxis]
    norms_Y = np.linalg.norm(Y, axis=1)
    return 1 - (dot_products / (norms_X * norms_Y))

def minkowski_distance(X: np.ndarray, Y: np.ndarray, p: float = 2) -> np.ndarray:
    """Minkowski distance between rows of X and Y."""
    return np.sum(np.abs(X[:, np.newaxis] - Y) ** p, axis=2) ** (1/p)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   metric_functions: Dict[str, Callable]) -> Dict[str, float]:
    """
    Compute multiple metrics for predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    metric_functions : Dict[str, Callable]
        Dictionary of metric functions

    Returns
    ------
    Dict[str, float]
        Computed metrics
    """
    return {name: func(y_true, y_pred) for name, func in metric_functions.items()}

def k_nearest_neighbors_fit(X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_test: Optional[np.ndarray] = None,
                           k: int = 5,
                           distance_metric: str = 'euclidean',
                           normalization: Optional[str] = None,
                           custom_distance: Optional[Callable] = None,
                           metric_functions: Dict[str, Callable] = None) -> Dict[str, Any]:
    """
    K-Nearest Neighbors regression/classification.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape (n_samples, n_features)
    y_train : np.ndarray
        Training target values of shape (n_samples,)
    X_test : Optional[np.ndarray]
        Test feature matrix of shape (n_samples, n_features)
    k : int
        Number of neighbors to consider
    distance_metric : str
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski')
    normalization : Optional[str]
        Normalization method ('standard', 'minmax', 'robust')
    custom_distance : Optional[Callable]
        Custom distance function if not using built-in metrics
    metric_functions : Dict[str, Callable]
        Dictionary of metric functions to compute

    Returns
    ------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings

    Examples
    --------
    >>> X_train = np.random.rand(100, 5)
    >>> y_train = np.random.rand(100)
    >>> X_test = np.random.rand(10, 5)
    >>> result = k_nearest_neighbors_fit(X_train, y_train, X_test,
    ...                                 k=3, normalization='standard')
    """
    # Validate inputs
    validate_inputs(X_train, y_train)
    if X_test is not None:
        validate_inputs(X_test, np.zeros(len(X_test)))  # Dummy validation for X_test

    # Normalize data if requested
    normalization_funcs = {
        'standard': standard_normalize,
        'minmax': minmax_normalize,
        'robust': robust_normalize
    }

    X_train_norm = normalization_funcs.get(normalization, lambda x: x)(X_train)
    if X_test is not None:
        X_test_norm = normalization_funcs.get(normalization, lambda x: x)(X_test)
    else:
        X_test_norm = None

    # Prepare distance function
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_metrics = {
            'euclidean': euclidean_distance,
            'manhattan': manhattan_distance,
            'cosine': cosine_distance
        }
        if distance_metric == 'minkowski':
            distance_func = lambda X, Y: minkowski_distance(X, Y, p=3)
        else:
            distance_func = distance_metrics[distance_metric]

    # Find k-nearest neighbors for each test point
    distances = distance_func(X_train_norm, X_test_norm if X_test is not None else X_train_norm)
    k_nearest_indices = np.argsort(distances, axis=1)[:, :k]

    # Make predictions (mean for regression, mode for classification)
    if y_train.dtype.kind in 'biu':  # Classification
        y_pred = np.array([np.argmax(np.bincount(y_train[k_nearest_indices[i]].astype(int)))
                          for i in range(len(k_nearest_indices))])
    else:  # Regression
        y_pred = np.array([np.mean(y_train[k_nearest_indices[i]]) for i in range(len(k_nearest_indices))])

    # Compute metrics if requested
    metrics = {}
    if metric_functions is not None:
        metrics = compute_metrics(y_train, y_pred, metric_functions)

    # Prepare output
    result = {
        'result': y_pred,
        'metrics': metrics,
        'params_used': {
            'k': k,
            'distance_metric': distance_metric if custom_distance is None else 'custom',
            'normalization': normalization,
            'n_samples_train': X_train.shape[0],
            'n_features': X_train.shape[1]
        },
        'warnings': []
    }

    return result

################################################################################
# support_vector_machine
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def support_vector_machine_fit(
    X: np.ndarray,
    y: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda x1, x2: np.dot(x1, x2),
    C: float = 1.0,
    epsilon: float = 1e-3,
    max_iter: int = 1000,
    tol: float = 1e-3,
    metric: str = 'mse',
    normalize: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, np.ndarray]]:
    """
    Fit a Support Vector Machine model to the given data.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    kernel : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Kernel function. Default is linear kernel.
    C : float
        Regularization parameter. Default is 1.0.
    epsilon : float
        Epsilon for the stopping criterion. Default is 1e-3.
    max_iter : int
        Maximum number of iterations. Default is 1000.
    tol : float
        Tolerance for stopping criterion. Default is 1e-3.
    metric : str
        Metric to evaluate the model. Default is 'mse'.
    normalize : Optional[str]
        Normalization method. Options: None, 'standard', 'minmax', 'robust'.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function. If provided, overrides the `metric` parameter.

    Returns
    -------
    Dict[str, Union[Dict, float, np.ndarray]]
        A dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _normalize_data(X, normalize) if normalize else X

    # Initialize parameters
    n_samples = X_normalized.shape[0]
    alpha = np.zeros(n_samples)
    b = 0.0

    # Compute kernel matrix
    K = kernel(X_normalized, X_normalized)

    # SMO algorithm to solve the dual problem
    alpha, b = _smo_algorithm(K, y, C, epsilon, max_iter, tol)

    # Calculate support vectors
    sv_indices = alpha > 1e-5
    support_vectors = X_normalized[sv_indices]
    sv_alphas = alpha[sv_indices]
    sv_y = y[sv_indices]

    # Calculate decision function
    def decision_function(x: np.ndarray) -> float:
        return np.sum(sv_alphas * sv_y * kernel(x, support_vectors)) + b

    # Predictions
    y_pred = np.array([decision_function(x) for x in X_normalized])

    # Calculate metrics
    metrics = _calculate_metrics(y, y_pred, metric, custom_metric)

    # Prepare output
    result = {
        'support_vectors': support_vectors,
        'alphas': sv_alphas,
        'bias': b,
        'decision_function': decision_function
    }

    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'kernel': kernel.__name__ if hasattr(kernel, '__name__') else 'custom',
            'C': C,
            'epsilon': epsilon,
            'max_iter': max_iter,
            'tol': tol,
            'metric': metric if custom_metric is None else 'custom',
            'normalize': normalize
        },
        'warnings': []
    }

    return output

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

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize the input data."""
    if method == 'standard':
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

def _smo_algorithm(K: np.ndarray, y: np.ndarray, C: float, epsilon: float, max_iter: int, tol: float) -> tuple:
    """Sequential Minimal Optimization algorithm to solve the dual problem."""
    alpha = np.zeros(y.shape[0])
    b = 0.0
    for _ in range(max_iter):
        num_changed_alphas = 0
        for i in range(y.shape[0]):
            Ei = _decision_function(alpha, y, K, i) - y[i]
            if (y[i] * Ei < -tol and alpha[i] < C) or (y[i] * Ei > tol and alpha[i] > 0):
                j = _select_second_alpha(i, y.shape[0])
                Ej = _decision_function(alpha, y, K, j) - y[j]
                alpha_i_old = alpha[i]
                alpha_j_old = alpha[j]

                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])

                if L == H:
                    continue

                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                alpha[j] -= y[j] * (Ei - Ej) / eta
                alpha[j] = np.clip(alpha[j], L, H)

                if abs(alpha[j] - alpha_j_old) < epsilon:
                    continue

                alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])

                b1 = b - Ei - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                b2 = b - Ej - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[j, j]

                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                num_changed_alphas += 1
        if num_changed_alphas == 0:
            break
    return alpha, b

def _decision_function(alpha: np.ndarray, y: np.ndarray, K: np.ndarray, i: int) -> float:
    """Calculate the decision function for a given sample."""
    return np.sum(alpha * y * K[i, :]) + 0.5

def _select_second_alpha(i: int, n_samples: int) -> int:
    """Select the second alpha to optimize."""
    j = i
    while j == i:
        j = np.random.randint(0, n_samples)
    return j

def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric: str, custom_metric: Optional[Callable]) -> Dict[str, float]:
    """Calculate the specified metrics."""
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
        elif metric == 'logloss':
            metrics['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return metrics

################################################################################
# naive_bayes
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def naive_bayes_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    prior: Optional[np.ndarray] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "logloss",
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    normalize: bool = True,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Fit a Naive Bayes classifier to the given data.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    prior : Optional[np.ndarray], default=None
        Prior probabilities of the classes. If None, computed from data.
    metric : Union[str, Callable], default="logloss"
        Metric to evaluate the model. Can be "logloss", "accuracy", or a custom callable.
    distance : Union[str, Callable], default="euclidean"
        Distance metric for feature computation. Can be "euclidean", "manhattan", or a custom callable.
    normalize : bool, default=True
        Whether to normalize the features.
    normalizer : Optional[Callable], default=None
        Custom normalization function. If None, standard scaling is used.
    custom_metric : Optional[Callable], default=None
        Custom metric function. Overrides the `metric` parameter.
    custom_distance : Optional[Callable], default=None
        Custom distance function. Overrides the `distance` parameter.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the fitted model parameters and evaluation metrics.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if required
    X_normalized = _normalize_features(X, normalize, normalizer)

    # Compute class priors
    classes = np.unique(y)
    n_classes = len(classes)
    if prior is None:
        prior = np.bincount(y) / len(y)

    # Compute class conditional probabilities
    class_means = _compute_class_means(X_normalized, y, classes)
    class_variances = _compute_class_variances(X_normalized, y, classes)

    # Prepare the result dictionary
    result = {
        "class_means": class_means,
        "class_variances": class_variances,
        "prior": prior
    }

    # Compute metrics if validation data is provided
    metrics = {}
    if custom_metric:
        y_pred = _predict(X_normalized, class_means, class_variances, prior)
        metrics["custom_metric"] = custom_metric(y, y_pred)
    else:
        if metric == "logloss":
            y_pred_proba = _predict_proba(X_normalized, class_means, class_variances, prior)
            metrics["logloss"] = _compute_log_loss(y, y_pred_proba)
        elif metric == "accuracy":
            y_pred = _predict(X_normalized, class_means, class_variances, prior)
            metrics["accuracy"] = np.mean(y_pred == y)

    # Return the result
    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "prior": prior is not None,
            "metric": metric if custom_metric is None else "custom",
            "distance": distance if custom_distance is None else "custom",
            "normalize": normalize,
            "normalizer": normalizer is not None
        },
        "warnings": []
    }

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

def _normalize_features(
    X: np.ndarray,
    normalize: bool,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Normalize the features."""
    if not normalize or normalizer is None:
        return X
    if normalizer is not None:
        return normalizer(X)
    # Default to standard scaling
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8)

def _compute_class_means(
    X: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray
) -> Dict[int, np.ndarray]:
    """Compute the mean of each feature for each class."""
    class_means = {}
    for cls in classes:
        X_cls = X[y == cls]
        class_means[cls] = np.mean(X_cls, axis=0)
    return class_means

def _compute_class_variances(
    X: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray
) -> Dict[int, np.ndarray]:
    """Compute the variance of each feature for each class."""
    class_variances = {}
    for cls in classes:
        X_cls = X[y == cls]
        class_variances[cls] = np.var(X_cls, axis=0)
    return class_variances

def _predict(
    X: np.ndarray,
    class_means: Dict[int, np.ndarray],
    class_variances: Dict[int, np.ndarray],
    prior: np.ndarray
) -> np.ndarray:
    """Predict the class labels for the given data."""
    y_pred_proba = _predict_proba(X, class_means, class_variances, prior)
    return np.argmax(y_pred_proba, axis=1)

def _predict_proba(
    X: np.ndarray,
    class_means: Dict[int, np.ndarray],
    class_variances: Dict[int, np.ndarray],
    prior: np.ndarray
) -> np.ndarray:
    """Predict the class probabilities for the given data."""
    n_samples = X.shape[0]
    n_classes = len(class_means)
    y_pred_proba = np.zeros((n_samples, n_classes))

    for i, cls in enumerate(class_means):
        mean = class_means[cls]
        var = class_variances[cls] + 1e-8
        log_prior = np.log(prior[i])
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
        y_pred_proba[:, i] = log_prior + np.sum(log_likelihood, axis=1)

    # Softmax to get probabilities
    y_pred_proba = np.exp(y_pred_proba - np.max(y_pred_proba, axis=1, keepdims=True))
    y_pred_proba /= np.sum(y_pred_proba, axis=1, keepdims=True)

    return y_pred_proba

def _compute_log_loss(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> float:
    """Compute the log loss."""
    epsilon = 1e-8
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred_proba), axis=1))

################################################################################
# decision_tree
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def decision_tree_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    criterion: str = 'mse',
    splitter: str = 'best',
    max_features: Optional[Union[int, float]] = None,
    random_state: Optional[int] = None,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = 'euclidean',
    normalize: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict:
    """
    Fit a decision tree model to the data.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    max_depth : Optional[int], default=None
        Maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    criterion : str, default='mse'
        Function to measure the quality of a split. Supported criteria are "mse" for regression and "gini" or "entropy" for classification.
    splitter : str, default='best'
        Strategy used to choose the split at each node. Supported strategies are "best" to choose the best split and "random" to choose the best random split.
    max_features : Optional[Union[int, float]], default=None
        Number of features to consider when looking for the best split. If int, then consider max_features features at each split. If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
    random_state : Optional[int], default=None
        Controls the randomness of the estimator. Pass an int for reproducible output across multiple function calls.
    distance_metric : Callable[[np.ndarray, np.ndarray], float] or str, default='euclidean'
        Distance metric used for splitting. Can be a callable function or one of the predefined metrics: 'euclidean', 'manhattan', 'cosine'.
    normalize : Optional[str], default=None
        Normalization method applied to features. Supported methods are "standard", "minmax", and "robust".
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], default=None
        Custom metric function to evaluate the quality of a split. If provided, overrides the criterion parameter.

    Returns:
    --------
    Dict
        A dictionary containing the fitted model, metrics, parameters used, and any warnings.
    """
    # Input validation
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _normalize_data(X, normalize) if normalize else X

    # Initialize tree structure
    tree = {
        'node_count': 0,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'criterion': criterion,
        'splitter': splitter,
        'max_features': max_features,
        'random_state': random_state
    }

    # Build the tree
    root = _build_tree(
        X_normalized, y,
        depth=0,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        splitter=splitter,
        max_features=max_features,
        random_state=random_state,
        distance_metric=distance_metric,
        custom_metric=custom_metric
    )

    tree['root'] = root

    # Calculate metrics
    y_pred = _predict(tree, X_normalized)
    metrics = _calculate_metrics(y, y_pred)

    return {
        'result': tree,
        'metrics': metrics,
        'params_used': {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'criterion': criterion,
            'splitter': splitter,
            'max_features': max_features,
            'random_state': random_state
        },
        'warnings': []
    }

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

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize the input data."""
    if method == 'standard':
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        X_normalized = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return X_normalized

def _build_tree(
    X: np.ndarray,
    y: np.ndarray,
    *,
    depth: int,
    max_depth: Optional[int],
    min_samples_split: int,
    min_samples_leaf: int,
    criterion: str,
    splitter: str,
    max_features: Optional[Union[int, float]],
    random_state: Optional[int],
    distance_metric: Callable[[np.ndarray, np.ndarray], float],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict:
    """Recursively build the decision tree."""
    n_samples, n_features = X.shape

    # Check if node should be a leaf
    if (max_depth is not None and depth >= max_depth) or \
       n_samples < min_samples_split or \
       (criterion == 'mse' and np.var(y) == 0):
        return _create_leaf_node(y)

    # Choose features to consider
    if max_features is not None:
        if isinstance(max_features, int):
            features = np.random.choice(n_features, max_features, replace=False)
        else:
            features = np.random.choice(n_features, int(max_features * n_features), replace=False)
    else:
        features = np.arange(n_features)

    # Find the best split
    if custom_metric is not None:
        criterion_func = lambda left_y, right_y: -custom_metric(left_y, right_y)
    else:
        criterion_func = _get_criterion_function(criterion)

    best_split = None
    best_score = -np.inf

    for feature in features:
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_indices, right_indices = _split_data(X[:, feature], threshold)
            if len(left_indices) < min_samples_leaf or len(right_indices) < min_samples_leaf:
                continue

            left_y = y[left_indices]
            right_y = y[right_indices]

            score = criterion_func(left_y, right_y)
            if score > best_score:
                best_score = score
                best_split = {
                    'feature': feature,
                    'threshold': threshold,
                    'left_indices': left_indices,
                    'right_indices': right_indices
                }

    if best_split is None:
        return _create_leaf_node(y)

    # Split the data
    left_indices = best_split['left_indices']
    right_indices = best_split['right_indices']

    # Recursively build left and right subtrees
    left_subtree = _build_tree(
        X[left_indices], y[left_indices],
        depth=depth + 1,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        splitter=splitter,
        max_features=max_features,
        random_state=random_state,
        distance_metric=distance_metric,
        custom_metric=custom_metric
    )

    right_subtree = _build_tree(
        X[right_indices], y[right_indices],
        depth=depth + 1,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        splitter=splitter,
        max_features=max_features,
        random_state=random_state,
        distance_metric=distance_metric,
        custom_metric=custom_metric
    )

    return {
        'is_leaf': False,
        'feature': best_split['feature'],
        'threshold': best_split['threshold'],
        'left': left_subtree,
        'right': right_subtree
    }

def _create_leaf_node(y: np.ndarray) -> Dict:
    """Create a leaf node."""
    return {
        'is_leaf': True,
        'value': np.mean(y) if len(y) > 0 else 0
    }

def _get_criterion_function(criterion: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the criterion function based on the specified criterion."""
    if criterion == 'mse':
        return lambda left_y, right_y: -_calculate_mse(left_y, right_y)
    elif criterion == 'gini':
        return lambda left_y, right_y: -_calculate_gini(left_y, right_y)
    elif criterion == 'entropy':
        return lambda left_y, right_y: -_calculate_entropy(left_y, right_y)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

def _split_data(X_column: np.ndarray, threshold: float) -> tuple:
    """Split data into left and right subsets based on a threshold."""
    left_indices = np.where(X_column <= threshold)[0]
    right_indices = np.where(X_column > threshold)[0]
    return left_indices, right_indices

def _calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def _calculate_gini(y: np.ndarray) -> float:
    """Calculate Gini impurity."""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

def _calculate_entropy(y: np.ndarray) -> float:
    """Calculate entropy."""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def _predict(tree: Dict, X: np.ndarray) -> np.ndarray:
    """Predict using the decision tree."""
    y_pred = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        y_pred[i] = _predict_sample(tree, X[i])
    return y_pred

def _predict_sample(node: Dict, sample: np.ndarray) -> float:
    """Predict a single sample using the decision tree."""
    if node['is_leaf']:
        return node['value']
    else:
        if sample[node['feature']] <= node['threshold']:
            return _predict_sample(node['left'], sample)
        else:
            return _predict_sample(node['right'], sample)

def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate various metrics."""
    mse = _calculate_mse(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

################################################################################
# ensemble_methods
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def ensemble_methods_fit(
    X: np.ndarray,
    y: np.ndarray,
    estimators: list,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'mse',
    weights: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Fit an ensemble of prediction models.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    estimators : list
        List of estimator objects with fit and predict methods
    normalizer : callable, optional
        Function to normalize the input features (default: None)
    metric : str or callable, optional
        Metric to evaluate predictions ('mse', 'mae', 'r2', or custom callable)
    weights : np.ndarray, optional
        Weights for each estimator (default: None)
    random_state : int, optional
        Random seed for reproducibility (default: None)

    Returns:
    --------
    dict
        Dictionary containing fitted ensemble results, metrics and parameters used

    Example:
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> estimators = [LinearRegression() for _ in range(5)]
    >>> result = ensemble_methods_fit(X, y, estimators)
    """
    # Input validation
    _validate_inputs(X, y, weights)

    # Initialize results dictionary
    results = {
        'result': {},
        'metrics': {},
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric if isinstance(metric, str) else 'custom',
            'weights': weights
        },
        'warnings': []
    }

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Fit each estimator and store results
    for i, est in enumerate(estimators):
        est.fit(X_normalized, y)

        # Make predictions
        y_pred = est.predict(X_normalized)

        # Calculate metric
        results['result'][f'estimator_{i}'] = y_pred
        results['metrics'][f'estimator_{i}'] = _calculate_metric(y, y_pred, metric)

    # Calculate ensemble prediction (weighted average)
    if weights is None:
        weights = np.ones(len(estimators)) / len(estimators)

    ensemble_pred = np.average(
        [results['result'][f'estimator_{i}'] for i in range(len(estimators))],
        axis=0,
        weights=weights
    )

    # Calculate ensemble metric
    results['result']['ensemble'] = ensemble_pred
    results['metrics']['ensemble'] = _calculate_metric(y, ensemble_pred, metric)

    return results

def _validate_inputs(X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray]) -> None:
    """Validate input arrays and parameters."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if weights is not None:
        if len(weights) != X.shape[0]:
            raise ValueError("Weights must have the same length as number of samples")
        if not np.allclose(np.sum(weights), 1.0):
            raise ValueError("Weights must sum to 1")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to input features."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _calculate_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Calculate specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)

    metrics_map = {
        'mse': lambda yt, yp: np.mean((yt - yp) ** 2),
        'mae': lambda yt, yp: np.mean(np.abs(yt - yp)),
        'r2': lambda yt, yp: 1 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2)
    }

    if metric not in metrics_map:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics_map[metric](y_true, y_pred)

################################################################################
# cross_validation
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def cross_validation_fit(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable,
    cv_scheme: Callable,
    metric_func: Callable = None,
    normalizer: Optional[Callable] = None,
    solver_params: Dict[str, Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform cross-validation for model fitting and evaluation.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    model_func : Callable
        Function that implements the model fitting and prediction
    cv_scheme : Callable
        Cross-validation scheme generator (e.g., KFold)
    metric_func : Callable, optional
        Metric function to evaluate model performance (default: None)
    normalizer : Callable, optional
        Normalization function to preprocess data (default: None)
    solver_params : Dict[str, Any], optional
        Parameters for the model fitting (default: None)
    **kwargs : dict
        Additional keyword arguments for customization

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Input validation
    _validate_inputs(X, y)

    # Initialize results dictionary
    results = {
        "result": {},
        "metrics": {},
        "params_used": {"solver_params": solver_params or {}},
        "warnings": []
    }

    # Apply normalization if specified
    X_normalized, y_normalized = _apply_normalization(X, y, normalizer)

    # Perform cross-validation
    cv_results = _perform_cross_validation(
        X_normalized, y_normalized,
        model_func, cv_scheme,
        metric_func=metric_func,
        solver_params=solver_params
    )

    # Store results
    results["result"] = cv_results["results"]
    results["metrics"] = cv_results["metrics"]

    return results

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

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable] = None
) -> tuple:
    """Apply normalization to input data."""
    X_normalized, y_normalized = X.copy(), y.copy()

    if normalizer is not None:
        try:
            X_normalized = normalizer(X)
            y_normalized = normalizer(y.reshape(-1, 1)).flatten()
        except Exception as e:
            raise ValueError(f"Normalization failed: {str(e)}")

    return X_normalized, y_normalized

def _perform_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable,
    cv_scheme: Callable,
    metric_func: Optional[Callable] = None,
    solver_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Perform cross-validation using the specified scheme."""
    results = []
    metrics = []

    for train_idx, test_idx in cv_scheme.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit model
        model = model_func(X_train, y_train, **(solver_params or {}))

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics if provided
        metric_value = None
        if metric_func is not None:
            try:
                metric_value = metric_func(y_test, y_pred)
            except Exception as e:
                warnings.warn(f"Metric calculation failed: {str(e)}")

        results.append({
            "train_indices": train_idx,
            "test_indices": test_idx,
            "predictions": y_pred
        })
        metrics.append(metric_value)

    return {
        "results": results,
        "metrics": metrics
    }

# Example usage:
"""
from sklearn.model_selection import KFold

def linear_regression(X, y, learning_rate=0.01):
    # Simple linear regression implementation
    class Model:
        def __init__(self, coef, intercept):
            self.coef = coef
            self.intercept = intercept

        def predict(self, X):
            return X.dot(self.coef) + self.intercept
    # ... implementation details ...
    return Model(coef, intercept)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

X = np.random.rand(100, 5)
y = np.random.rand(100)

cv_scheme = KFold(n_splits=5)
results = cross_validation_fit(
    X, y,
    model_func=linear_regression,
    cv_scheme=cv_scheme,
    metric_func=mse
)
"""

################################################################################
# feature_engineering
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def feature_engineering_fit(
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
    Main function for feature engineering in prediction tasks.

    Parameters:
    -----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    normalizer : callable
        Function to normalize features. Default is identity function.
    metric : str or callable
        Metric to evaluate performance. Can be 'mse', 'mae', 'r2' or custom callable.
    distance : str or callable
        Distance metric for feature transformations. Can be 'euclidean', 'manhattan',
        'cosine' or custom callable.
    solver : str
        Solver algorithm. Options: 'closed_form', 'gradient_descent', etc.
    regularization : str or None
        Regularization type. Options: 'l1', 'l2', 'elasticnet' or None.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : callable or None
        Custom metric function if needed.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_normalized = normalizer(X)

    # Prepare metric function
    metric_func = _get_metric_function(metric, custom_metric)

    # Prepare distance function
    distance_func = _get_distance_function(distance)

    # Solve according to chosen method
    if solver == 'closed_form':
        result = _solve_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        result = _solve_gradient_descent(
            X_normalized, y,
            distance_func=distance_func,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(y, result['predictions'], metric_func)

    return {
        'result': result,
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
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _get_metric_function(
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Callable:
    """Get metric function based on input."""
    if callable(metric):
        return metric
    if custom_metric is not None:
        return custom_metric

    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }

    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance_function(distance: Union[str, Callable]) -> Callable:
    """Get distance function based on input."""
    if callable(distance):
        return distance

    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance
    }

    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Closed form solution for linear regression."""
    XTX = np.dot(X.T, X)
    if not np.allclose(XTX, XTX.T):
        raise ValueError("X^T X is not symmetric positive definite")
    try:
        coef = np.linalg.solve(XTX, np.dot(X.T, y))
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is singular or nearly singular")

    predictions = np.dot(X, coef)
    return {
        'coefficients': coef,
        'predictions': predictions
    }

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Gradient descent solver with optional regularization."""
    n_features = X.shape[1]
    coef = np.zeros(n_features)
    prev_loss = float('inf')

    for i in range(max_iter):
        predictions = np.dot(X, coef)
        error = y - predictions

        # Compute gradient
        grad = -2 * np.dot(X.T, error) / len(y)

        # Add regularization if needed
        if regularization == 'l1':
            grad += np.sign(coef)
        elif regularization == 'l2':
            grad += 2 * coef
        elif regularization == 'elasticnet':
            grad += np.sign(coef) + 2 * coef

        # Update coefficients
        coef -= grad * tol

        # Check convergence
        current_loss = np.mean(error ** 2)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    predictions = np.dot(X, coef)
    return {
        'coefficients': coef,
        'predictions': predictions
    }

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Cosine distance."""
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable
) -> Dict[str, float]:
    """Calculate all requested metrics."""
    return {
        'metric_value': metric_func(y_true, y_pred)
    }

################################################################################
# data_scaling
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def data_scaling_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalization: str = 'standard',
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
    Fit data scaling parameters and compute metrics.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    y : Optional[np.ndarray]
        Target values. If None, only scaling is performed.
    normalization : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : Union[str, Callable]
        Metric to evaluate: 'mse', 'mae', 'r2', 'logloss', or custom callable.
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
        Custom metric function if needed.
    custom_distance : Optional[Callable]
        Custom distance function if needed.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Choose normalization method
    if normalization == 'standard':
        X_scaled, params = _standard_normalization(X)
    elif normalization == 'minmax':
        X_scaled, params = _minmax_normalization(X)
    elif normalization == 'robust':
        X_scaled, params = _robust_normalization(X)
    else:
        X_scaled, params = X, {}

    # Choose solver
    if solver == 'closed_form':
        result = _closed_form_solver(X_scaled, y)
    elif solver == 'gradient_descent':
        result = _gradient_descent_solver(X_scaled, y, tol, max_iter)
    elif solver == 'newton':
        result = _newton_solver(X_scaled, y, tol, max_iter)
    elif solver == 'coordinate_descent':
        result = _coordinate_descent_solver(X_scaled, y, tol, max_iter)
    else:
        raise ValueError("Unsupported solver method.")

    # Compute metrics
    metrics = _compute_metrics(result, X_scaled, y, metric, custom_metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None.")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values.")
    if y is not None and np.any(np.isnan(y)):
        raise ValueError("y contains NaN values.")

def _standard_normalization(X: np.ndarray) -> tuple:
    """Standard normalization (mean=0, std=1)."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled, {'mean': mean, 'std': std}

def _minmax_normalization(X: np.ndarray) -> tuple:
    """Min-max normalization (range [0, 1])."""
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    X_scaled = (X - min_val) / (max_val - min_val + 1e-8)
    return X_scaled, {'min': min_val, 'max': max_val}

def _robust_normalization(X: np.ndarray) -> tuple:
    """Robust normalization (median and IQR)."""
    median = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    iqr = q75 - q25
    X_scaled = (X - median) / (iqr + 1e-8)
    return X_scaled, {'median': median, 'iqr': iqr}

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form solution for linear regression."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent solver."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = (2 / n_samples) * X.T @ (X @ weights - y)
        new_weights = weights - learning_rate * gradient
        if np.linalg.norm(new_weights - weights) < tol:
            break
        weights = new_weights

    return weights

def _newton_solver(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Newton's method solver."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = (2 / n_samples) * X.T @ (X @ weights - y)
        hessian = (2 / n_samples) * X.T @ X
        new_weights = weights - np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(new_weights - weights) < tol:
            break
        weights = new_weights

    return weights

def _coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Coordinate descent solver."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - X @ weights + weights[j] * X_j
            weights[j] = np.sum(X_j * residuals) / np.sum(X_j ** 2)
        if np.linalg.norm(weights - weights) < tol:
            break

    return weights

def _compute_metrics(
    result: np.ndarray,
    X: np.ndarray,
    y: Optional[np.ndarray],
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute metrics based on the chosen method."""
    if y is None:
        return {}

    predictions = X @ result
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
        metrics_dict['logloss'] = -np.mean(y * np.log(predictions + 1e-8) + (1 - y) * np.log(1 - predictions + 1e-8))
    elif custom_metric is not None:
        metrics_dict['custom'] = custom_metric(y, predictions)

    return metrics_dict

################################################################################
# outlier_detection
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def outlier_detection_fit(
    X: np.ndarray,
    method: str = 'zscore',
    threshold: float = 3.0,
    normalize: Optional[str] = None,
    distance_metric: str = 'euclidean',
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    contamination: Optional[float] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Detect outliers in a dataset using various statistical methods.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    method : str, optional
        Method for outlier detection ('zscore', 'iqr', 'dbscan', 'lof')
    threshold : float, optional
        Threshold for outlier detection (depends on method)
    normalize : str or None, optional
        Normalization method ('standard', 'minmax', 'robust')
    distance_metric : str, optional
        Distance metric for methods that require it ('euclidean', 'manhattan', 'cosine')
    custom_distance : callable or None, optional
        Custom distance function if needed
    contamination : float or None, optional
        Expected proportion of outliers

    Returns
    -------
    dict
        Dictionary containing:
        - 'outliers': boolean array indicating outliers
        - 'scores': outlier scores
        - 'params_used': parameters used for the computation
        - 'warnings': any warnings generated

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = outlier_detection_fit(X, method='zscore', threshold=3.0)
    """
    # Validate inputs
    _validate_inputs(X, method, threshold, normalize, distance_metric)

    # Normalize data if required
    X_normalized = _apply_normalization(X, normalize)

    # Get outlier detection method
    if method == 'zscore':
        outliers, scores = _zscore_outliers(X_normalized, threshold)
    elif method == 'iqr':
        outliers, scores = _iqr_outliers(X_normalized, threshold)
    elif method == 'dbscan':
        outliers, scores = _dbscan_outliers(X_normalized, threshold, distance_metric, custom_distance)
    elif method == 'lof':
        outliers, scores = _lof_outliers(X_normalized, threshold, distance_metric, custom_distance)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate metrics
    metrics = _calculate_metrics(X, outliers)

    return {
        'outliers': outliers,
        'scores': scores,
        'metrics': metrics,
        'params_used': {
            'method': method,
            'threshold': threshold,
            'normalize': normalize,
            'distance_metric': distance_metric
        },
        'warnings': []
    }

def _validate_inputs(
    X: np.ndarray,
    method: str,
    threshold: float,
    normalize: Optional[str],
    distance_metric: str
) -> None:
    """Validate input parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values")
    if threshold <= 0:
        raise ValueError("Threshold must be positive")

def _apply_normalization(
    X: np.ndarray,
    method: Optional[str]
) -> np.ndarray:
    """Apply normalization to the data."""
    if method is None:
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        return (X - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _zscore_outliers(
    X: np.ndarray,
    threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Detect outliers using z-score method."""
    scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
    outliers = np.any(scores > threshold, axis=1)
    return outliers, scores

def _iqr_outliers(
    X: np.ndarray,
    threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Detect outliers using IQR method."""
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    scores = np.zeros_like(X)
    for i in range(X.shape[1]):
        scores[:, i] = np.where((X[:, i] < lower_bound[i]) | (X[:, i] > upper_bound[i]), 1, 0)

    outliers = np.any(scores > 0, axis=1)
    return outliers, scores

def _dbscan_outliers(
    X: np.ndarray,
    threshold: float,
    distance_metric: str,
    custom_distance: Optional[Callable]
) -> tuple[np.ndarray, np.ndarray]:
    """Detect outliers using DBSCAN method."""
    # This is a simplified version - in practice you would use sklearn's DBSCAN
    from sklearn.cluster import DBSCAN

    if custom_distance:
        clustering = DBSCAN(eps=threshold, metric=custom_distance).fit(X)
    else:
        clustering = DBSCAN(eps=threshold, metric=distance_metric).fit(X)

    outliers = clustering.labels_ == -1
    scores = np.zeros_like(clustering.labels_, dtype=float)
    scores[outliers] = 1.0
    return outliers, scores

def _lof_outliers(
    X: np.ndarray,
    threshold: float,
    distance_metric: str,
    custom_distance: Optional[Callable]
) -> tuple[np.ndarray, np.ndarray]:
    """Detect outliers using Local Outlier Factor method."""
    # This is a simplified version - in practice you would use sklearn's LOF
    from sklearn.neighbors import LocalOutlierFactor

    if custom_distance:
        lof = LocalOutlierFactor(contamination=threshold, metric=custom_distance)
    else:
        lof = LocalOutlierFactor(contamination=threshold, metric=distance_metric)

    scores = lof.negative_outlier_factor_
    outliers = lof.fit_predict(X) == -1
    return outliers, scores

def _calculate_metrics(
    X: np.ndarray,
    outliers: np.ndarray
) -> Dict[str, float]:
    """Calculate metrics for outlier detection."""
    n_outliers = np.sum(outliers)
    total_samples = X.shape[0]
    return {
        'outlier_ratio': n_outliers / total_samples,
        'n_outliers': n_outliers
    }

################################################################################
# time_series_decomposition
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def time_series_decomposition_fit(
    y: np.ndarray,
    trend_method: str = 'linear',
    seasonal_method: str = 'additive',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Decompose a time series into trend, seasonal and residual components.

    Parameters
    ----------
    y : np.ndarray
        Input time series data.
    trend_method : str, optional
        Method for trend estimation ('linear', 'polynomial', 'exponential').
    seasonal_method : str, optional
        Method for seasonal decomposition ('additive', 'multiplicative').
    normalizer : Callable, optional
        Function to normalize the data (e.g., standard scaling).
    metric : str or Callable, optional
        Metric to evaluate decomposition quality ('mse', 'mae', 'r2').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent').
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing decomposition results, metrics and parameters used.

    Examples
    --------
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> result = time_series_decomposition_fit(y)
    """
    # Validate inputs
    _validate_inputs(y, trend_method, seasonal_method, normalizer, metric)

    # Normalize data if specified
    y_normalized = _apply_normalization(y, normalizer)

    # Decompose time series
    trend, seasonal, residual = _decompose_time_series(
        y_normalized,
        trend_method=trend_method,
        seasonal_method=seasonal_method,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state
    )

    # Calculate metrics
    metrics = _calculate_metrics(y_normalized, trend, seasonal, residual, metric)

    # Prepare output
    result = {
        'result': {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        },
        'metrics': metrics,
        'params_used': {
            'trend_method': trend_method,
            'seasonal_method': seasonal_method,
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(y_normalized, trend, seasonal, residual)
    }

    return result

def _validate_inputs(
    y: np.ndarray,
    trend_method: str,
    seasonal_method: str,
    normalizer: Optional[Callable],
    metric: Union[str, Callable]
) -> None:
    """Validate input parameters."""
    if not isinstance(y, np.ndarray):
        raise TypeError("Input y must be a numpy array")
    if len(y.shape) != 1:
        raise ValueError("Input y must be a 1D array")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input y contains NaN or infinite values")
    if trend_method not in ['linear', 'polynomial', 'exponential']:
        raise ValueError("Invalid trend_method")
    if seasonal_method not in ['additive', 'multiplicative']:
        raise ValueError("Invalid seasonal_method")
    if normalizer is not None and not callable(normalizer):
        raise TypeError("normalizer must be a callable or None")
    if isinstance(metric, str) and metric not in ['mse', 'mae', 'r2']:
        raise ValueError("Invalid metric")
    if not isinstance(metric, (str, Callable)):
        raise TypeError("metric must be a string or callable")

def _apply_normalization(
    y: np.ndarray,
    normalizer: Optional[Callable]
) -> np.ndarray:
    """Apply normalization to the input data."""
    if normalizer is None:
        return y
    return normalizer(y)

def _decompose_time_series(
    y: np.ndarray,
    trend_method: str,
    seasonal_method: str,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> tuple:
    """Decompose time series into trend, seasonal and residual components."""
    if solver == 'closed_form':
        return _decompose_closed_form(y, trend_method, seasonal_method)
    elif solver == 'gradient_descent':
        return _decompose_gradient_descent(
            y, trend_method, seasonal_method,
            regularization, tol, max_iter, random_state
        )
    else:
        raise ValueError("Invalid solver")

def _decompose_closed_form(
    y: np.ndarray,
    trend_method: str,
    seasonal_method: str
) -> tuple:
    """Closed form decomposition of time series."""
    n = len(y)
    if trend_method == 'linear':
        x = np.arange(n).reshape(-1, 1)
        trend_coef = np.linalg.lstsq(x, y, rcond=None)[0]
        trend = x @ trend_coef
    else:
        raise NotImplementedError(f"Trend method {trend_method} not implemented")

    if seasonal_method == 'additive':
        seasonal = _estimate_seasonal_additive(y, trend)
    else:
        seasonal = _estimate_seasonal_multiplicative(y, trend)

    residual = y - trend - seasonal
    return trend, seasonal, residual

def _estimate_seasonal_additive(
    y: np.ndarray,
    trend: np.ndarray
) -> np.ndarray:
    """Estimate additive seasonal component."""
    return y - trend

def _estimate_seasonal_multiplicative(
    y: np.ndarray,
    trend: np.ndarray
) -> np.ndarray:
    """Estimate multiplicative seasonal component."""
    return (y / trend) - 1

def _decompose_gradient_descent(
    y: np.ndarray,
    trend_method: str,
    seasonal_method: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> tuple:
    """Gradient descent decomposition of time series."""
    np.random.seed(random_state)
    trend = np.zeros_like(y)
    seasonal = np.zeros_like(y)

    for _ in range(max_iter):
        # Update trend
        if trend_method == 'linear':
            x = np.arange(len(y)).reshape(-1, 1)
            trend_coef = np.linalg.lstsq(x, y - seasonal + trend, rcond=None)[0]
            new_trend = x @ trend_coef
        else:
            raise NotImplementedError(f"Trend method {trend_method} not implemented")

        # Update seasonal
        if seasonal_method == 'additive':
            new_seasonal = y - new_trend
        else:
            new_seasonal = (y / (new_trend + 1e-8)) - 1

        # Check convergence
        if np.linalg.norm(new_trend - trend) < tol and np.linalg.norm(new_seasonal - seasonal) < tol:
            break

        trend, seasonal = new_trend, new_seasonal

    residual = y - trend - seasonal
    return trend, seasonal, residual

def _calculate_metrics(
    y: np.ndarray,
    trend: np.ndarray,
    seasonal: np.ndarray,
    residual: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Calculate decomposition metrics."""
    if callable(metric):
        return {'custom': metric(y, trend + seasonal)}

    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((y - (trend + seasonal))**2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y - (trend + seasonal)))
    elif metric == 'r2':
        ss_res = np.sum(residual**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

def _check_warnings(
    y: np.ndarray,
    trend: np.ndarray,
    seasonal: np.ndarray,
    residual: np.ndarray
) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isnan(trend)) or np.any(np.isinf(trend)):
        warnings.append("Trend contains NaN or infinite values")
    if np.any(np.isnan(seasonal)) or np.any(np.isinf(seasonal)):
        warnings.append("Seasonal contains NaN or infinite values")
    if np.any(np.isnan(residual)) or np.any(np.isinf(residual)):
        warnings.append("Residual contains NaN or infinite values")
    return warnings

################################################################################
# stationarity_test
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def stationarity_test_fit(
    data: np.ndarray,
    test_type: str = 'adf',
    window_size: Optional[int] = None,
    normalize: bool = True,
    normalization_method: str = 'standard',
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Perform stationarity test on time series data.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.
    test_type : str, optional
        Type of stationarity test ('adf', 'kpss').
    window_size : int, optional
        Size of rolling window for local stationarity tests.
    normalize : bool, optional
        Whether to normalize data before testing.
    normalization_method : str, optional
        Normalization method ('standard', 'minmax', 'robust').
    custom_metric : callable, optional
        Custom metric function for stationarity assessment.
    **kwargs :
        Additional parameters specific to the test type.

    Returns:
    --------
    dict
        Dictionary containing test results, metrics, parameters used and warnings.

    Example:
    --------
    >>> data = np.random.randn(100)
    >>> result = stationarity_test_fit(data, test_type='adf')
    """
    # Validate inputs
    _validate_inputs(data)

    # Normalize data if required
    normalized_data = _normalize_data(data, normalize, normalization_method) if normalize else data

    # Select test method
    if test_type == 'adf':
        result = _augmented_dickey_fuller_test(normalized_data, **kwargs)
    elif test_type == 'kpss':
        result = _kpss_test(normalized_data, **kwargs)
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    # Calculate metrics
    metrics = _calculate_metrics(result, custom_metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'test_type': test_type,
            'window_size': window_size,
            'normalize': normalize,
            'normalization_method': normalization_method
        },
        'warnings': _check_warnings(data, normalized_data)
    }

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if len(data.shape) != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def _normalize_data(
    data: np.ndarray,
    normalize: bool,
    method: str = 'standard'
) -> np.ndarray:
    """Normalize data using specified method."""
    if not normalize:
        return data

    if method == 'standard':
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

def _augmented_dickey_fuller_test(
    data: np.ndarray,
    max_lag: int = 12,
    regression: str = 'c',
    **kwargs
) -> Dict:
    """Perform Augmented Dickey-Fuller test."""
    # Implementation of ADF test
    # This is a placeholder - actual implementation would use statsmodels or similar
    p_value = 0.123456789  # Placeholder value
    statistic = -3.123456789  # Placeholder value
    used_lag = min(max_lag, len(data) - 2)

    return {
        'statistic': statistic,
        'p_value': p_value,
        'used_lag': used_lag,
        'critical_values': {
            '1%': -3.48,
            '5%': -2.88,
            '10%': -2.58
        },
        'regression_type': regression
    }

def _kpss_test(
    data: np.ndarray,
    regression: str = 'c',
    **kwargs
) -> Dict:
    """Perform KPSS test."""
    # Implementation of KPSS test
    # This is a placeholder - actual implementation would use statsmodels or similar
    p_value = 0.987654321  # Placeholder value
    statistic = 0.123456789  # Placeholder value

    return {
        'statistic': statistic,
        'p_value': p_value,
        'critical_values': {
            '1%': 0.739,
            '5%': 0.463,
            '10%': 0.347
        },
        'regression_type': regression
    }

def _calculate_metrics(
    test_result: Dict,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Calculate metrics from test results."""
    metrics = {
        'p_value': test_result['p_value'],
        'statistic': test_result['statistic']
    }

    if custom_metric is not None:
        metrics['custom'] = custom_metric(test_result)

    return metrics

def _check_warnings(
    original_data: np.ndarray,
    processed_data: np.ndarray
) -> list:
    """Check for potential warnings."""
    warnings = []

    if not np.array_equal(original_data, processed_data):
        warnings.append("Data was normalized before testing")

    if len(original_data) < 20:
        warnings.append("Small sample size may affect test reliability")

    return warnings

################################################################################
# autocorrelation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(series: np.ndarray) -> None:
    """Validate input series for autocorrelation computation."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if series.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array")
    if np.isnan(series).any() or np.isinf(series).any():
        raise ValueError("Input contains NaN or infinite values")

def compute_autocorrelation(series: np.ndarray,
                           max_lag: int = 10,
                           normalization: str = 'standard',
                           metric: Union[str, Callable] = 'pearson') -> Dict:
    """
    Compute autocorrelation for a time series.

    Parameters
    ----------
    series : np.ndarray
        Input time series data.
    max_lag : int, optional
        Maximum lag to compute autocorrelation for, by default 10.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), by default 'standard'.
    metric : Union[str, Callable], optional
        Metric to use ('pearson', 'spearman', custom callable), by default 'pearson'.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used and warnings.
    """
    validate_input(series)

    # Normalize the series
    normalized_series = _normalize(series, method=normalization)

    # Compute autocorrelation for each lag
    acf_values = []
    for lag in range(max_lag + 1):
        acf_value = _compute_acf(normalized_series, lag, metric)
        acf_values.append(acf_value)

    # Prepare output
    result = {
        'lags': np.arange(max_lag + 1),
        'autocorrelation_values': np.array(acf_values)
    }

    return {
        'result': result,
        'metrics': {'metric_used': metric},
        'params_used': {
            'max_lag': max_lag,
            'normalization': normalization
        },
        'warnings': []
    }

def _normalize(series: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize the input series."""
    if method == 'none':
        return series
    elif method == 'standard':
        mean = np.mean(series)
        std = np.std(series)
        return (series - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(series)
        max_val = np.max(series)
        return (series - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(series)
        iqr = np.percentile(series, 75) - np.percentile(series, 25)
        return (series - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_acf(series: np.ndarray, lag: int,
                 metric: Union[str, Callable] = 'pearson') -> float:
    """Compute autocorrelation for a given lag."""
    if lag == 0:
        return 1.0

    if callable(metric):
        # Use custom metric
        x = series[:-lag]
        y = series[lag:]
        return metric(x, y)
    elif metric == 'pearson':
        x = series[:-lag]
        y = series[lag:]
        cov = np.cov(x, y)[0, 1]
        var_x = np.var(x)
        return cov / (var_x + 1e-8)
    elif metric == 'spearman':
        x = series[:-lag]
        y = series[lag:]
        return np.corrcoef(np.argsort(x), np.argsort(y))[0, 1]
    else:
        raise ValueError(f"Unknown metric: {metric}")

# Example usage
if __name__ == "__main__":
    import numpy as np

    # Generate a sample time series
    np.random.seed(42)
    series = np.cumsum(np.random.randn(100))

    # Compute autocorrelation
    result = compute_autocorrelation(
        series=series,
        max_lag=10,
        normalization='standard',
        metric='pearson'
    )

    print(result)

################################################################################
# moving_average
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array")
    if X.ndim != 1:
        raise ValueError("Input X must be a 1-dimensional array")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input X contains NaN or infinite values")

def normalize_data(X: np.ndarray, method: str = 'none') -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'standard':
        return (X - np.mean(X)) / np.std(X)
    elif method == 'minmax':
        return (X - np.min(X)) / (np.max(X) - np.min(X))
    elif method == 'robust':
        return (X - np.median(X)) / (np.percentile(X, 75) - np.percentile(X, 25))
    elif method == 'none':
        return X
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
        return 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def moving_average_fit(X: np.ndarray, window_size: int,
                       normalization: str = 'none',
                       metric: Union[str, Callable] = 'mse') -> Dict:
    """
    Compute moving average with configurable parameters.

    Parameters:
    -----------
    X : np.ndarray
        Input time series data.
    window_size : int
        Size of the moving window.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to evaluate the moving average ('mse', 'mae', 'r2' or custom).

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate input
    validate_input(X)

    # Normalize data
    X_norm = normalize_data(X, normalization)

    # Compute moving average
    result = np.convolve(X_norm, np.ones(window_size)/window_size, mode='valid')

    # Compute metrics
    if len(result) < 2:
        metrics = {}
    else:
        metrics = {
            'metric': compute_metric(X_norm[window_size-1:], result, metric)
        }

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'window_size': window_size,
            'normalization': normalization
        },
        'warnings': []
    }

    return output

# Example usage:
# result = moving_average_fit(np.array([1, 2, 3, 4, 5]), window_size=3)

################################################################################
# exponential_smoothing
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def exponential_smoothing_fit(
    y: np.ndarray,
    alpha: float = 0.1,
    initial_value: Optional[float] = None,
    normalize: str = "none",
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    solver: str = "closed_form",
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Fit exponential smoothing model to time series data.

    Parameters:
    -----------
    y : np.ndarray
        Time series data to fit.
    alpha : float, optional
        Smoothing parameter (0 < alpha <= 1).
    initial_value : float, optional
        Initial value for the smoothing. If None, uses first observation.
    normalize : str or callable, optional
        Normalization method: "none", "standard", "minmax", or custom callable.
    metric : str or callable, optional
        Metric to evaluate model: "mse", "mae", "r2", or custom callable.
    solver : str, optional
        Solver method: "closed_form" or other future options.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_normalize : callable, optional
        Custom normalization function.
    custom_metric : callable, optional
        Custom metric function.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(y, alpha, initial_value)

    # Normalize data if required
    y_normalized = _normalize_data(y, normalize, custom_normalize)

    # Initialize parameters
    params = {"alpha": alpha}
    if initial_value is None:
        initial_value = y_normalized[0]
    params["initial_value"] = initial_value

    # Fit model
    if solver == "closed_form":
        smoothed_values, warnings = _fit_closed_form(y_normalized, alpha, initial_value)
    else:
        raise ValueError(f"Solver {solver} not implemented")

    # Calculate metrics
    metrics = _calculate_metrics(y_normalized, smoothed_values, metric, custom_metric)

    return {
        "result": {"smoothed_values": smoothed_values},
        "metrics": metrics,
        "params_used": params,
        "warnings": warnings
    }

def _validate_inputs(
    y: np.ndarray,
    alpha: float,
    initial_value: Optional[float]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array")
    if len(y.shape) != 1:
        raise ValueError("y must be a 1-dimensional array")
    if np.any(np.isnan(y)):
        raise ValueError("y contains NaN values")
    if alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be in (0, 1]")
    if initial_value is not None and np.isnan(initial_value):
        raise ValueError("initial_value cannot be NaN")

def _normalize_data(
    y: np.ndarray,
    normalize: str,
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Normalize data according to specified method."""
    if custom_normalize is not None:
        return custom_normalize(y)

    if normalize == "standard":
        mean = np.mean(y)
        std = np.std(y)
        return (y - mean) / std
    elif normalize == "minmax":
        min_val = np.min(y)
        max_val = np.max(y)
        return (y - min_val) / (max_val - min_val)
    elif normalize == "robust":
        median = np.median(y)
        iqr = np.percentile(y, 75) - np.percentile(y, 25)
        return (y - median) / iqr
    elif normalize == "none":
        return y.copy()
    else:
        raise ValueError(f"Normalization method {normalize} not recognized")

def _fit_closed_form(
    y: np.ndarray,
    alpha: float,
    initial_value: float
) -> tuple[np.ndarray, Dict[str, str]]:
    """Fit exponential smoothing using closed-form solution."""
    warnings = {}
    n = len(y)
    smoothed_values = np.zeros(n)

    # Initialize
    smoothed_values[0] = initial_value

    # Compute smoothed values
    for t in range(1, n):
        smoothed_values[t] = alpha * y[t-1] + (1 - alpha) * smoothed_values[t-1]

    return smoothed_values, warnings

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    if custom_metric is not None:
        return {"custom": custom_metric(y_true, y_pred)}

    metrics = {}
    if metric == "mse":
        metrics["mse"] = np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        metrics["mae"] = np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics["r2"] = 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Metric {metric} not recognized")

    return metrics

# Example usage:
"""
y = np.array([1, 2, 3, 4, 5])
result = exponential_smoothing_fit(y, alpha=0.3)
print(result)
"""

################################################################################
# seasonality_adjustment
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def seasonality_adjustment_fit(
    y: np.ndarray,
    seasonal_periods: int = 12,
    normalizer: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    custom_normalizer: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit a seasonality adjustment model to the input data.

    Parameters
    ----------
    y : np.ndarray
        Input time series data.
    seasonal_periods : int, optional
        Number of periods in the seasonality cycle (default is 12).
    normalizer : str or callable, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') or custom callable (default is 'standard').
    metric : str or callable, optional
        Metric to optimize ('mse', 'mae', 'r2', 'logloss') or custom callable (default is 'mse').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default is 'closed_form').
    regularization : str, optional
        Regularization method ('none', 'l1', 'l2', 'elasticnet') (default is None).
    max_iter : int, optional
        Maximum number of iterations for iterative solvers (default is 1000).
    tol : float, optional
        Tolerance for convergence (default is 1e-6).
    custom_normalizer : callable, optional
        Custom normalizer function (default is None).
    custom_metric : callable, optional
        Custom metric function (default is None).

    Returns
    -------
    Dict
        Dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> result = seasonality_adjustment_fit(y)
    """
    # Validate inputs
    _validate_inputs(y, seasonal_periods)

    # Normalize data
    if custom_normalizer is not None:
        y_normalized = custom_normalizer(y)
    else:
        y_normalized, normalizer_func = _apply_normalization(y, normalizer)

    # Prepare seasonal components
    X_seasonal = _prepare_seasonal_components(y, seasonal_periods)

    # Choose solver
    if solver == 'closed_form':
        params = _closed_form_solver(X_seasonal, y_normalized)
    elif solver == 'gradient_descent':
        params = _gradient_descent_solver(X_seasonal, y_normalized, max_iter, tol)
    elif solver == 'newton':
        params = _newton_solver(X_seasonal, y_normalized, max_iter, tol)
    elif solver == 'coordinate_descent':
        params = _coordinate_descent_solver(X_seasonal, y_normalized, max_iter, tol)
    else:
        raise ValueError("Invalid solver specified.")

    # Apply regularization if needed
    if regularization is not None:
        params = _apply_regularization(params, X_seasonal, y_normalized, regularization)

    # Compute metrics
    if custom_metric is not None:
        metric_value = custom_metric(y_normalized, X_seasonal @ params)
    else:
        metric_value = _compute_metric(y_normalized, X_seasonal @ params, metric)

    # Prepare output
    result = {
        'result': y_normalized - X_seasonal @ params,
        'metrics': {metric: metric_value},
        'params_used': {
            'seasonal_periods': seasonal_periods,
            'normalizer': normalizer,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

    return result

def _validate_inputs(y: np.ndarray, seasonal_periods: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(y, np.ndarray):
        raise TypeError("Input y must be a numpy array.")
    if len(y.shape) != 1:
        raise ValueError("Input y must be a 1-dimensional array.")
    if seasonal_periods <= 0:
        raise ValueError("Seasonal periods must be a positive integer.")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("Input y contains NaN or infinite values.")

def _apply_normalization(y: np.ndarray, method: str) -> tuple:
    """Apply normalization to the input data."""
    if method == 'none':
        return y, lambda x: x
    elif method == 'standard':
        mean = np.mean(y)
        std = np.std(y)
        normalized = (y - mean) / std
        return normalized, lambda x: (x - mean) / std
    elif method == 'minmax':
        min_val = np.min(y)
        max_val = np.max(y)
        normalized = (y - min_val) / (max_val - min_val)
        return normalized, lambda x: (x - min_val) / (max_val - min_val)
    elif method == 'robust':
        median = np.median(y)
        iqr = np.percentile(y, 75) - np.percentile(y, 25)
        normalized = (y - median) / iqr
        return normalized, lambda x: (x - median) / iqr
    else:
        raise ValueError("Invalid normalization method specified.")

def _prepare_seasonal_components(y: np.ndarray, seasonal_periods: int) -> np.ndarray:
    """Prepare the design matrix for seasonal components."""
    n = len(y)
    X = np.zeros((n, seasonal_periods))
    for i in range(seasonal_periods):
        X[:, i] = np.sin(2 * np.pi * (i + 1) * np.arange(n) / seasonal_periods)
    return X

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve using closed-form solution."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _gradient_descent_solver(X: np.ndarray, y: np.ndarray, max_iter: int, tol: float) -> np.ndarray:
    """Solve using gradient descent."""
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

def _newton_solver(X: np.ndarray, y: np.ndarray, max_iter: int, tol: float) -> np.ndarray:
    """Solve using Newton's method."""
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

def _coordinate_descent_solver(X: np.ndarray, y: np.ndarray, max_iter: int, tol: float) -> np.ndarray:
    """Solve using coordinate descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        for i in range(n_features):
            X_i = X[:, i]
            residual = y - X @ params + params[i] * X_i
            params[i] = np.sum(X_i * residual) / np.sum(X_i ** 2)
        if np.linalg.norm(params - params) < tol:
            break
    return params

def _apply_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    """Apply regularization to the parameters."""
    if method == 'l1':
        alpha = 0.1
        params -= alpha * np.sign(params)
    elif method == 'l2':
        alpha = 0.1
        params -= alpha * params
    elif method == 'elasticnet':
        alpha = 0.1
        l1_ratio = 0.5
        params -= alpha * (l1_ratio * np.sign(params) + (1 - l1_ratio) * params)
    else:
        raise ValueError("Invalid regularization method specified.")
    return params

def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Compute the specified metric."""
    if metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError("Invalid metric specified.")

################################################################################
# forecast_accuracy_metrics
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def forecast_accuracy_metrics_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    normalization: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict[str, float], Dict[str, str], list]]:
    """
    Calculate forecast accuracy metrics between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    metrics : str or callable, optional
        Metric(s) to compute. Can be 'mse', 'mae', 'r2', or a custom callable.
    normalization : str, optional
        Normalization method. Can be 'standard', 'minmax', or None.
    custom_metric : callable, optional
        Custom metric function.

    Returns
    -------
    Dict[str, Union[Dict[str, float], Dict[str, str], list]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 2.2, 3.3])
    >>> forecast_accuracy_metrics_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize data if required
    if normalization:
        y_true_norm, y_pred_norm = _normalize_data(y_true, y_pred, normalization)
    else:
        y_true_norm, y_pred_norm = y_true, y_pred

    # Compute metrics
    results = {}
    if isinstance(metrics, str):
        if metrics == 'mse':
            results['mse'] = _compute_mse(y_true_norm, y_pred_norm)
        elif metrics == 'mae':
            results['mae'] = _compute_mae(y_true_norm, y_pred_norm)
        elif metrics == 'r2':
            results['r2'] = _compute_r2(y_true_norm, y_pred_norm)
        else:
            raise ValueError(f"Unknown metric: {metrics}")
    elif callable(metrics):
        results['custom_metric'] = metrics(y_true_norm, y_pred_norm)
    else:
        raise ValueError("metrics must be a string or callable")

    if custom_metric is not None:
        results['custom_metric'] = custom_metric(y_true_norm, y_pred_norm)

    return {
        'result': results,
        'metrics': list(results.keys()),
        'params_used': {
            'normalization': normalization,
            'metrics': metrics if isinstance(metrics, str) else 'custom'
        },
        'warnings': _check_warnings(y_true_norm, y_pred_norm)
    }

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("Inputs must not contain NaN values")
    if np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError("Inputs must not contain infinite values")

def _normalize_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize data using specified method."""
    if method == 'standard':
        mean = np.mean(y_true)
        std = np.std(y_true)
        y_true_norm = (y_true - mean) / std
        y_pred_norm = (y_pred - mean) / std
    elif method == 'minmax':
        min_val = np.min(y_true)
        max_val = np.max(y_true)
        y_true_norm = (y_true - min_val) / (max_val - min_val)
        y_pred_norm = (y_pred - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return y_true_norm, y_pred_norm

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

def _check_warnings(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(y_true == y_pred):
        warnings.append("Some true values are equal to predicted values")
    return warnings
