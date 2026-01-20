"""
Quantix – Module series_temporelles
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# stationnarite
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(series: np.ndarray) -> None:
    """Validate input series for stationarity analysis."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input series must be a numpy array")
    if series.ndim != 1:
        raise ValueError("Input series must be one-dimensional")
    if np.isnan(series).any() or np.isinf(series).any():
        raise ValueError("Input series contains NaN or infinite values")

def _normalize_series(series: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize the time series using specified method."""
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

def _compute_statistic(series: np.ndarray, statistic_func: Callable) -> float:
    """Compute the stationarity statistic using provided function."""
    return statistic_func(series)

def _compute_metrics(original: np.ndarray, transformed: np.ndarray,
                     metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute various metrics between original and transformed series."""
    metrics = {}
    for name, func in metric_funcs.items():
        if callable(func):
            metrics[name] = func(original, transformed)
    return metrics

def stationnarite_fit(series: np.ndarray,
                     statistic_func: Callable = lambda x: np.var(x),
                     normalize_method: str = 'standard',
                     metric_funcs: Optional[Dict[str, Callable]] = None,
                     **kwargs) -> Dict:
    """
    Analyze stationarity of a time series.

    Parameters
    ----------
    series : np.ndarray
        Input time series to analyze.
    statistic_func : Callable, optional
        Function to compute stationarity statistic (default: variance).
    normalize_method : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric_funcs : Dict[str, Callable], optional
        Dictionary of metric functions to compute.
    **kwargs :
        Additional parameters for metric functions.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used and warnings.

    Example
    -------
    >>> series = np.random.randn(100)
    >>> result = stationnarite_fit(series, normalize_method='standard')
    """
    # Validate input
    _validate_input(series)

    # Initialize default metrics if not provided
    if metric_funcs is None:
        metric_funcs = {
            'mse': lambda x, y: np.mean((x - y) ** 2),
            'mae': lambda x, y: np.mean(np.abs(x - y))
        }

    # Normalize series
    normalized_series = _normalize_series(series, normalize_method)

    # Compute stationarity statistic
    statistic_value = _compute_statistic(normalized_series, statistic_func)

    # Compute metrics
    metrics = _compute_metrics(series, normalized_series, metric_funcs)

    # Prepare result dictionary
    result = {
        'result': {
            'statistic_value': statistic_value,
            'normalized_series': normalized_series
        },
        'metrics': metrics,
        'params_used': {
            'normalize_method': normalize_method,
            'statistic_func': statistic_func.__name__ if hasattr(statistic_func, '__name__') else 'custom',
            'metric_functions': list(metric_funcs.keys())
        },
        'warnings': []
    }

    return result

################################################################################
# tendance_seculaire
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def tendance_seculaire_fit(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    normalisation: str = 'none',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularisation: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Estimate the secular trend of a time series.

    Parameters:
    -----------
    y : np.ndarray
        Time series data.
    x : Optional[np.ndarray], default=None
        Design matrix. If None, uses linear time index.
    normalisation : str, default='none'
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : Union[str, Callable], default='mse'
        Metric to evaluate the fit: 'mse', 'mae', 'r2', or custom callable.
    solver : str, default='closed_form'
        Solver method: 'closed_form', 'gradient_descent', or 'newton'.
    regularisation : Optional[str], default=None
        Regularization type: None, 'l1', 'l2', or 'elasticnet'.
    tol : float, default=1e-4
        Tolerance for convergence.
    max_iter : int, default=1000
        Maximum number of iterations for iterative solvers.
    custom_metric : Optional[Callable], default=None
        Custom metric function if not using built-in metrics.
    custom_distance : Optional[Callable], default=None
        Custom distance function for solver.

    Returns:
    --------
    Dict
        Dictionary containing:
        - 'result': Estimated trend parameters.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used in the fit.
        - 'warnings': Any warnings generated during fitting.

    Example:
    --------
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> result = tendance_seculaire_fit(y)
    """
    # Validate inputs
    _validate_inputs(y, x)

    # Normalize data if required
    y_norm, x_norm = _apply_normalisation(y, x, normalisation)

    # Prepare design matrix
    if x is None:
        x_norm = np.vstack([np.ones(len(y_norm)), np.arange(len(y_norm))]).T

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(x_norm, y_norm, regularisation)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(
            x_norm, y_norm, tol, max_iter,
            regularisation, custom_distance
        )
    elif solver == 'newton':
        params = _solve_newton(
            x_norm, y_norm, tol, max_iter,
            regularisation, custom_distance
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(
        y_norm, x_norm @ params,
        metric, custom_metric
    )

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'metric': metric,
            'solver': solver,
            'regularisation': regularisation
        },
        'warnings': []
    }

    return result

def _validate_inputs(y: np.ndarray, x: Optional[np.ndarray]) -> None:
    """Validate input arrays."""
    if not isinstance(y, np.ndarray) or (x is not None and not isinstance(x, np.ndarray)):
        raise TypeError("Inputs must be numpy arrays")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if x is not None and x.ndim != 2:
        raise ValueError("x must be a 2D array")
    if np.any(np.isnan(y)) or (x is not None and np.any(np.isnan(x))):
        raise ValueError("Input arrays must not contain NaN values")
    if x is not None and y.shape[0] != x.shape[0]:
        raise ValueError("y and x must have the same number of samples")

def _apply_normalisation(
    y: np.ndarray,
    x: Optional[np.ndarray],
    method: str
) -> tuple:
    """Apply normalization to input arrays."""
    y_norm = y.copy()
    x_norm = x.copy() if x is not None else None

    if method == 'standard':
        y_norm = (y - np.mean(y)) / np.std(y)
        if x is not None:
            x_norm = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    elif method == 'minmax':
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
        if x is not None:
            x_norm = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
    elif method == 'robust':
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
        if x is not None:
            x_norm = (x - np.median(x, axis=0)) / (
                np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
            )

    return y_norm, x_norm

def _solve_closed_form(
    x: np.ndarray,
    y: np.ndarray,
    regularisation: Optional[str]
) -> np.ndarray:
    """Solve using closed-form solution."""
    if regularisation is None:
        params = np.linalg.pinv(x) @ y
    elif regularisation == 'l2':
        params = _ridge_regression(x, y)
    else:
        raise ValueError(f"Regularization {regularisation} not implemented for closed_form solver")

    return params

def _solve_gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    regularisation: Optional[str],
    distance_func: Optional[Callable]
) -> np.ndarray:
    """Solve using gradient descent."""
    n_features = x.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = _compute_gradient(x, y, params, regularisation)
        if distance_func is not None:
            gradient = distance_func(gradient)

        params -= tol * gradient
        if np.linalg.norm(gradient) < tol:
            break

    return params

def _solve_newton(
    x: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    regularisation: Optional[str],
    distance_func: Optional[Callable]
) -> np.ndarray:
    """Solve using Newton's method."""
    n_features = x.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = _compute_gradient(x, y, params, regularisation)
        hessian = _compute_hessian(x)

        if distance_func is not None:
            gradient = distance_func(gradient)
            hessian = distance_func(hessian)

        params -= np.linalg.pinv(hessian) @ gradient
        if np.linalg.norm(gradient) < tol:
            break

    return params

def _compute_gradient(
    x: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    regularisation: Optional[str]
) -> np.ndarray:
    """Compute gradient for optimization."""
    residuals = y - x @ params
    gradient = -2 * x.T @ residuals

    if regularisation == 'l1':
        gradient += np.sign(params)
    elif regularisation == 'l2':
        gradient += 2 * params

    return gradient

def _compute_hessian(x: np.ndarray) -> np.ndarray:
    """Compute Hessian matrix."""
    return 2 * x.T @ x

def _ridge_regression(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Perform ridge regression."""
    alpha = 1.0
    return np.linalg.inv(x.T @ x + alpha * np.identity(x.shape[1])) @ x.T @ y

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute evaluation metrics."""
    metrics = {}

    if metric == 'mse' or (custom_metric is None and metric == 'mse'):
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae' or (custom_metric is None and metric == 'mae'):
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2' or (custom_metric is None and metric == 'r2'):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)

    return metrics

################################################################################
# saisonnalite
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def saisonnalite_fit(
    series: np.ndarray,
    period: int = 12,
    normalizer: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_normalizer: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Estimate seasonality components in a time series.

    Parameters:
    -----------
    series : np.ndarray
        Input time series data.
    period : int, default=12
        Seasonal period to estimate.
    normalizer : str or callable, default='standard'
        Normalization method: 'none', 'standard', 'minmax', 'robust'.
    metric : str or callable, default='mse'
        Metric to evaluate seasonality: 'mse', 'mae', 'r2'.
    distance : str, default='euclidean'
        Distance metric for seasonality components: 'euclidean', 'manhattan'.
    solver : str, default='closed_form'
        Solver method: 'closed_form', 'gradient_descent'.
    regularization : str or None, default=None
        Regularization type: 'none', 'l1', 'l2'.
    tol : float, default=1e-6
        Tolerance for convergence.
    max_iter : int, default=1000
        Maximum number of iterations for iterative solvers.
    custom_normalizer : callable or None, default=None
        Custom normalization function.
    custom_metric : callable or None, default=None
        Custom metric function.
    custom_distance : callable or None, default=None
        Custom distance function.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(series, period)

    # Normalize data
    normalized_series = _apply_normalization(
        series,
        normalizer=normalizer,
        custom_normalizer=custom_normalizer
    )

    # Prepare seasonality estimation
    seasonal_components = _estimate_seasonal_components(
        normalized_series,
        period=period,
        distance=distance,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter,
        custom_distance=custom_distance
    )

    # Calculate metrics
    metrics = _calculate_metrics(
        series,
        seasonal_components,
        metric=metric,
        custom_metric=custom_metric
    )

    # Prepare output
    result = {
        'result': seasonal_components,
        'metrics': metrics,
        'params_used': {
            'period': period,
            'normalizer': normalizer if custom_normalizer is None else 'custom',
            'metric': metric if custom_metric is None else 'custom',
            'distance': distance if custom_distance is None else 'custom',
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(series, seasonal_components)
    }

    return result

def _validate_inputs(series: np.ndarray, period: int) -> None:
    """Validate input series and parameters."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input series must be a numpy array.")
    if len(series.shape) != 1:
        raise ValueError("Input series must be one-dimensional.")
    if period <= 0 or not isinstance(period, int):
        raise ValueError("Period must be a positive integer.")
    if np.isnan(series).any():
        raise ValueError("Input series contains NaN values.")
    if np.isinf(series).any():
        raise ValueError("Input series contains infinite values.")

def _apply_normalization(
    series: np.ndarray,
    normalizer: str = 'standard',
    custom_normalizer: Optional[Callable] = None
) -> np.ndarray:
    """Apply normalization to the input series."""
    if custom_normalizer is not None:
        return custom_normalizer(series)

    if normalizer == 'none':
        return series
    elif normalizer == 'standard':
        mean = np.mean(series)
        std = np.std(series)
        return (series - mean) / (std + 1e-8)
    elif normalizer == 'minmax':
        min_val = np.min(series)
        max_val = np.max(series)
        return (series - min_val) / (max_val - min_val + 1e-8)
    elif normalizer == 'robust':
        median = np.median(series)
        iqr = np.percentile(series, 75) - np.percentile(series, 25)
        return (series - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalizer: {normalizer}")

def _estimate_seasonal_components(
    series: np.ndarray,
    period: int,
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_distance: Optional[Callable] = None
) -> np.ndarray:
    """Estimate seasonal components using the specified solver."""
    n = len(series)
    if n < period:
        raise ValueError("Series length must be greater than or equal to the period.")

    if solver == 'closed_form':
        return _closed_form_solution(series, period)
    elif solver == 'gradient_descent':
        return _gradient_descent_solution(
            series,
            period,
            distance=distance,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter,
            custom_distance=custom_distance
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _closed_form_solution(series: np.ndarray, period: int) -> np.ndarray:
    """Closed-form solution for seasonal component estimation."""
    n = len(series)
    X = np.zeros((n, period))
    for i in range(n):
        X[i, i % period] = 1
    coefficients = np.linalg.pinv(X) @ series
    seasonal_components = X @ coefficients
    return seasonal_components

def _gradient_descent_solution(
    series: np.ndarray,
    period: int,
    distance: str = 'euclidean',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_distance: Optional[Callable] = None
) -> np.ndarray:
    """Gradient descent solution for seasonal component estimation."""
    n = len(series)
    coefficients = np.zeros(period)

    for _ in range(max_iter):
        gradients = np.zeros(period)
        for i in range(n):
            seasonal_component = coefficients[i % period]
            residual = series[i] - seasonal_component
            if custom_distance is not None:
                gradients[i % period] -= custom_distance(residual)
            elif distance == 'euclidean':
                gradients[i % period] -= residual
            elif distance == 'manhattan':
                gradients[i % period] -= np.sign(residual)

        # Apply regularization
        if regularization == 'l1':
            gradients += np.sign(coefficients)
        elif regularization == 'l2':
            gradients += 2 * coefficients

        # Update coefficients
        old_coefficients = coefficients.copy()
        coefficients -= tol * gradients

        # Check convergence
        if np.linalg.norm(coefficients - old_coefficients) < tol:
            break

    seasonal_components = np.zeros(n)
    for i in range(n):
        seasonal_components[i] = coefficients[i % period]

    return seasonal_components

def _calculate_metrics(
    series: np.ndarray,
    seasonal_components: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Calculate metrics for the seasonal components."""
    residuals = series - seasonal_components

    if custom_metric is not None:
        return {'custom': custom_metric(series, seasonal_components)}

    metrics = {}
    if metric == 'mse' or (isinstance(metric, str) and 'mse' in metric):
        metrics['mse'] = np.mean(residuals ** 2)
    if metric == 'mae' or (isinstance(metric, str) and 'mae' in metric):
        metrics['mae'] = np.mean(np.abs(residuals))
    if metric == 'r2' or (isinstance(metric, str) and 'r2' in metric):
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((series - np.mean(series)) ** 2)
        metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-8))

    return metrics

def _check_warnings(series: np.ndarray, seasonal_components: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isnan(seasonal_components)):
        warnings.append("NaN values detected in seasonal components.")
    if np.any(np.isinf(seasonal_components)):
        warnings.append("Infinite values detected in seasonal components.")
    if len(series) < 2 * seasonal_components[0]:
        warnings.append("Series length might be too short for reliable seasonality estimation.")
    return warnings

################################################################################
# composante_aleatoire
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def composante_aleatoire_fit(
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Estimate the random component of a time series.

    Parameters
    ----------
    y : np.ndarray
        Input time series data.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the data. If None, no normalization is applied.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to evaluate the performance. Can be 'mse', 'mae', 'r2', or a custom callable.
    solver : str, optional
        Solver to use. Options: 'closed_form', 'gradient_descent'.
    regularization : Optional[str], optional
        Regularization type. Options: None, 'l1', 'l2'.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    random_state : Optional[int], optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> y = np.array([1.0, 2.0, 3.0, 4.0])
    >>> result = composante_aleatoire_fit(y)
    """
    # Validate inputs
    _validate_inputs(y)

    # Normalize data if specified
    y_normalized = y.copy()
    if normalizer is not None:
        y_normalized = normalizer(y)

    # Initialize random state
    if random_state is not None:
        np.random.seed(random_state)

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(y_normalized, regularization)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(y_normalized, regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(y, y_normalized, params, metric)

    # Prepare output
    result = {
        'result': y_normalized - params['trend'],
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric if isinstance(metric, str) else 'custom',
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(y: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(y, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if y.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input contains NaN or infinite values.")

def _solve_closed_form(y: np.ndarray, regularization: Optional[str]) -> Dict[str, np.ndarray]:
    """Solve for the random component using closed-form solution."""
    n = len(y)
    X = np.vstack([np.ones(n), np.arange(n)]).T
    if regularization == 'l2':
        # Add L2 regularization
        X = np.hstack([X, np.eye(2)])
    params = np.linalg.pinv(X) @ y
    return {'trend': params[1] * np.arange(n) + params[0]}

def _solve_gradient_descent(
    y: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, np.ndarray]:
    """Solve for the random component using gradient descent."""
    n = len(y)
    params = np.random.rand(2)  # Initial guess
    for _ in range(max_iter):
        gradient = _compute_gradient(y, params, regularization)
        params -= tol * gradient
    return {'trend': params[1] * np.arange(n) + params[0]}

def _compute_gradient(
    y: np.ndarray,
    params: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Compute the gradient for gradient descent."""
    n = len(y)
    trend = params[1] * np.arange(n) + params[0]
    residuals = y - trend
    gradient = np.zeros(2)
    gradient[0] = -2 * np.sum(residuals) / n
    gradient[1] = -2 * np.sum((np.arange(n) - np.mean(np.arange(n))) * residuals) / n
    if regularization == 'l2':
        gradient += 2 * params
    return gradient

def _compute_metrics(
    y: np.ndarray,
    y_normalized: np.ndarray,
    params: Dict[str, np.ndarray],
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for the model."""
    residuals = y_normalized - params['trend']
    if isinstance(metric, str):
        if metric == 'mse':
            return {'mse': np.mean(residuals ** 2)}
        elif metric == 'mae':
            return {'mae': np.mean(np.abs(residuals))}
        elif metric == 'r2':
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return {'r2': 1 - ss_res / ss_tot}
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        return {'custom': metric(y, params['trend'])}

################################################################################
# decomposition_series_temporelles
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def decomposition_series_temporelles_fit(
    time_series: np.ndarray,
    trend_method: str = 'linear',
    seasonal_method: str = 'fft',
    residual_method: str = 'default',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_trend_func: Optional[Callable] = None,
    custom_seasonal_func: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Decompose a time series into trend, seasonal and residual components.

    Parameters
    ----------
    time_series : np.ndarray
        Input time series data.
    trend_method : str, optional
        Method for trend decomposition ('linear', 'polynomial', 'exponential').
    seasonal_method : str, optional
        Method for seasonal decomposition ('fft', 'stl').
    residual_method : str, optional
        Method for residual handling ('default', 'none').
    normalizer : Callable, optional
        Function to normalize the time series.
    metric : str, optional
        Metric for evaluation ('mse', 'mae', 'r2').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent').
    regularization : str, optional
        Regularization method ('none', 'l1', 'l2').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_trend_func : Callable, optional
        Custom function for trend decomposition.
    custom_seasonal_func : Callable, optional
        Custom function for seasonal decomposition.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing decomposition results, metrics, parameters used and warnings.

    Examples
    --------
    >>> time_series = np.array([1, 2, 3, 4, 5])
    >>> result = decomposition_series_temporelles_fit(time_series)
    """
    # Validate inputs
    _validate_inputs(time_series, trend_method, seasonal_method, residual_method)

    # Normalize data if normalizer is provided
    normalized_data = _apply_normalization(time_series, normalizer)

    # Decompose the time series
    trend, seasonal, residual = _decompose_time_series(
        normalized_data,
        trend_method,
        seasonal_method,
        residual_method,
        custom_trend_func,
        custom_seasonal_func
    )

    # Calculate metrics
    metrics = _calculate_metrics(time_series, trend + seasonal + residual, metric)

    # Prepare the result dictionary
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
            'residual_method': residual_method,
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

    return result

def _validate_inputs(
    time_series: np.ndarray,
    trend_method: str,
    seasonal_method: str,
    residual_method: str
) -> None:
    """Validate the input time series and methods."""
    if not isinstance(time_series, np.ndarray):
        raise ValueError("time_series must be a numpy array")
    if len(time_series.shape) != 1:
        raise ValueError("time_series must be a 1-dimensional array")
    if np.any(np.isnan(time_series)):
        raise ValueError("time_series contains NaN values")
    if np.any(np.isinf(time_series)):
        raise ValueError("time_series contains infinite values")

    valid_trend_methods = ['linear', 'polynomial', 'exponential']
    if trend_method not in valid_trend_methods:
        raise ValueError(f"trend_method must be one of {valid_trend_methods}")

    valid_seasonal_methods = ['fft', 'stl']
    if seasonal_method not in valid_seasonal_methods:
        raise ValueError(f"seasonal_method must be one of {valid_seasonal_methods}")

    valid_residual_methods = ['default', 'none']
    if residual_method not in valid_residual_methods:
        raise ValueError(f"residual_method must be one of {valid_residual_methods}")

def _apply_normalization(
    time_series: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to the time series if a normalizer is provided."""
    if normalizer is not None:
        return normalizer(time_series)
    return time_series

def _decompose_time_series(
    time_series: np.ndarray,
    trend_method: str,
    seasonal_method: str,
    residual_method: str,
    custom_trend_func: Optional[Callable] = None,
    custom_seasonal_func: Optional[Callable] = None
) -> tuple:
    """Decompose the time series into trend, seasonal and residual components."""
    if custom_trend_func is not None:
        trend = custom_trend_func(time_series)
    else:
        trend = _compute_trend(time_series, trend_method)

    if custom_seasonal_func is not None:
        seasonal = custom_seasonal_func(time_series)
    else:
        seasonal = _compute_seasonal(time_series, seasonal_method)

    residual = time_series - trend - seasonal

    if residual_method == 'none':
        residual = np.zeros_like(time_series)

    return trend, seasonal, residual

def _compute_trend(
    time_series: np.ndarray,
    method: str
) -> np.ndarray:
    """Compute the trend component of the time series."""
    n = len(time_series)
    t = np.arange(n)

    if method == 'linear':
        trend = _fit_linear_trend(time_series, t)
    elif method == 'polynomial':
        trend = _fit_polynomial_trend(time_series, t)
    elif method == 'exponential':
        trend = _fit_exponential_trend(time_series, t)
    else:
        raise ValueError(f"Unknown trend method: {method}")

    return trend

def _fit_linear_trend(
    time_series: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """Fit a linear trend to the time series."""
    A = np.vstack([t, np.ones(len(t))]).T
    m, c = np.linalg.lstsq(A, time_series, rcond=None)[0]
    return m * t + c

def _fit_polynomial_trend(
    time_series: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """Fit a polynomial trend to the time series."""
    coefficients = np.polyfit(t, time_series, 2)
    return np.polyval(coefficients, t)

def _fit_exponential_trend(
    time_series: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """Fit an exponential trend to the time series."""
    log_series = np.log(time_series)
    coefficients = np.polyfit(t, log_series, 1)
    return np.exp(coefficients[0] * t + coefficients[1])

def _compute_seasonal(
    time_series: np.ndarray,
    method: str
) -> np.ndarray:
    """Compute the seasonal component of the time series."""
    if method == 'fft':
        seasonal = _compute_seasonal_fft(time_series)
    elif method == 'stl':
        seasonal = _compute_seasonal_stl(time_series)
    else:
        raise ValueError(f"Unknown seasonal method: {method}")

    return seasonal

def _compute_seasonal_fft(
    time_series: np.ndarray
) -> np.ndarray:
    """Compute the seasonal component using FFT."""
    fft = np.fft.fft(time_series)
    frequencies = np.fft.fftfreq(len(time_series))
    seasonal_freqs = [f for f in frequencies if 0 < abs(f) <= 0.5]
    seasonal_fft = np.zeros_like(fft)
    for freq in seasonal_freqs:
        idx = np.where(frequencies == freq)[0][0]
        seasonal_fft[idx] = fft[idx]
    return np.fft.ifft(seasonal_fft).real

def _compute_seasonal_stl(
    time_series: np.ndarray
) -> np.ndarray:
    """Compute the seasonal component using STL decomposition."""
    # Placeholder for STL implementation
    return np.zeros_like(time_series)

def _calculate_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    metric: str
) -> Dict[str, float]:
    """Calculate the metrics for the decomposition."""
    if metric == 'mse':
        return {'mse': np.mean((original - reconstructed) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(original - reconstructed))}
    elif metric == 'r2':
        ss_res = np.sum((original - reconstructed) ** 2)
        ss_tot = np.sum((original - np.mean(original)) ** 2)
        return {'r2': 1 - (ss_res / ss_tot)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# lissage_exponentiel
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(y: np.ndarray,
                    alpha: float = 0.5,
                    beta: Optional[float] = None,
                    gamma: Optional[float] = None) -> None:
    """Validate inputs for exponential smoothing."""
    if not isinstance(y, np.ndarray):
        raise TypeError("Input y must be a numpy array")
    if len(y.shape) != 1:
        raise ValueError("Input y must be a 1-dimensional array")
    if np.any(np.isnan(y)):
        raise ValueError("Input y contains NaN values")
    if np.any(np.isinf(y)):
        raise ValueError("Input y contains infinite values")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if beta is not None and (not 0 <= beta <= 1):
        raise ValueError("beta must be between 0 and 1 or None")
    if gamma is not None and (not 0 <= gamma <= 1):
        raise ValueError("gamma must be between 0 and 1 or None")

def _initialize_parameters(y: np.ndarray,
                          alpha: float = 0.5,
                          beta: Optional[float] = None,
                          gamma: Optional[float] = None) -> Dict[str, float]:
    """Initialize parameters for exponential smoothing."""
    params = {
        'level': np.mean(y),
        'trend': 0.0,
        'seasonal': 0.0 if gamma is None else np.mean(y[:len(y)//gamma])
    }
    return params

def _update_parameters(params: Dict[str, float],
                      y_t: float,
                      alpha: float = 0.5,
                      beta: Optional[float] = None,
                      gamma: Optional[float] = None) -> Dict[str, float]:
    """Update parameters for exponential smoothing."""
    params['level'] = alpha * y_t + (1 - alpha) * (params['level'] + params['trend'])
    if beta is not None:
        params['trend'] = beta * (params['level'] - params['level_prev']) + (1 - beta) * params['trend']
    if gamma is not None:
        seasonal_period = int(1 / gamma)
        params['seasonal'] = gamma * (y_t - params['level']) + (1 - gamma) * params['seasonal']
    return params

def _compute_metrics(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     metric: Union[str, Callable] = 'mse') -> Dict[str, float]:
    """Compute metrics for exponential smoothing."""
    if callable(metric):
        return {'custom_metric': metric(y_true, y_pred)}

    metrics = {}
    if metric in ['mse', 'mae', 'r2']:
        residuals = y_true - y_pred
        if metric == 'mse':
            metrics['mse'] = np.mean(residuals**2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(residuals))
        elif metric == 'r2':
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

def lissage_exponentiel_fit(y: np.ndarray,
                           alpha: float = 0.5,
                           beta: Optional[float] = None,
                           gamma: Optional[float] = None,
                           metric: Union[str, Callable] = 'mse',
                           normalize: Optional[Callable] = None) -> Dict:
    """Fit exponential smoothing model to time series data.

    Parameters
    ----------
    y : np.ndarray
        Time series data.
    alpha : float, optional
        Smoothing parameter for level (0 <= alpha <= 1).
    beta : float, optional
        Smoothing parameter for trend (0 <= beta <= 1 or None).
    gamma : float, optional
        Smoothing parameter for seasonality (0 <= gamma <= 1 or None).
    metric : Union[str, Callable], optional
        Metric to evaluate the model ('mse', 'mae', 'r2' or custom callable).
    normalize : Callable, optional
        Normalization function to apply to the data.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used and warnings.

    Examples
    --------
    >>> y = np.array([10, 12, 15, 14, 18])
    >>> result = lissage_exponentiel_fit(y, alpha=0.5)
    """
    _validate_inputs(y, alpha, beta, gamma)

    if normalize is not None:
        y = normalize(y)

    params = _initialize_parameters(y, alpha, beta, gamma)
    predictions = []
    warnings = []

    for t in range(len(y)):
        try:
            params['level_prev'] = params['level']
            params = _update_parameters(params, y[t], alpha, beta, gamma)
            if gamma is not None:
                prediction = params['level'] + params['trend'] + params['seasonal']
            else:
                prediction = params['level'] + params['trend']
            predictions.append(prediction)
        except Exception as e:
            warnings.append(str(e))

    metrics = _compute_metrics(y, np.array(predictions), metric)

    return {
        'result': predictions,
        'metrics': metrics,
        'params_used': {'alpha': alpha, 'beta': beta, 'gamma': gamma},
        'warnings': warnings
    }

################################################################################
# arima
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_arima_inputs(
    y: np.ndarray,
    order: tuple = (1, 0, 1),
    seasonal_order: Optional[tuple] = None,
    normalize: str = 'none',
) -> Dict[str, Union[np.ndarray, tuple]]:
    """
    Validate and preprocess inputs for ARIMA model.

    Parameters
    ----------
    y : np.ndarray
        Time series data.
    order : tuple, optional
        (p, d, q) order of the ARIMA model.
    seasonal_order : tuple, optional
        (P, D, Q, S) order of the seasonal component.
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').

    Returns
    -------
    dict
        Validated and preprocessed inputs.
    """
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError("y must be a 1D numpy array")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("y contains NaN or inf values")
    if len(order) != 3:
        raise ValueError("order must be a tuple of length 3 (p, d, q)")
    if seasonal_order is not None and len(seasonal_order) != 4:
        raise ValueError("seasonal_order must be a tuple of length 4 (P, D, Q, S)")

    # Normalization
    if normalize == 'standard':
        y_norm = (y - np.mean(y)) / np.std(y)
    elif normalize == 'minmax':
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalize == 'robust':
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        y_norm = y.copy()

    return {
        'y': y_norm,
        'order': order,
        'seasonal_order': seasonal_order,
    }

def compute_arima_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable] = 'mse',
) -> Dict[str, float]:
    """
    Compute metrics for ARIMA model.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    metric : str or callable, optional
        Metric to compute ('mse', 'mae', 'r2', or custom callable).

    Returns
    -------
    dict
        Computed metrics.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    elif callable(metric):
        metrics['custom'] = metric(y_true, y_pred)
    else:
        raise ValueError("Invalid metric specified")

    return metrics

def fit_arima_model(
    y: np.ndarray,
    order: tuple = (1, 0, 1),
    seasonal_order: Optional[tuple] = None,
    solver: str = 'gradient_descent',
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Fit ARIMA model to time series data.

    Parameters
    ----------
    y : np.ndarray
        Time series data.
    order : tuple, optional
        (p, d, q) order of the ARIMA model.
    seasonal_order : tuple, optional
        (P, D, Q, S) order of the seasonal component.
    solver : str, optional
        Solver method ('gradient_descent', 'newton').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    dict
        Fitted model parameters and diagnostics.
    """
    # Placeholder for actual ARIMA fitting logic
    params = np.random.rand(order[0] + order[1] + order[2])
    if seasonal_order is not None:
        params = np.concatenate([params, np.random.rand(seasonal_order[0] + seasonal_order[1] + seasonal_order[2])])

    return {
        'params': params,
        'converged': True,
        'iterations': max_iter,
    }

def arima_fit(
    y: np.ndarray,
    order: tuple = (1, 0, 1),
    seasonal_order: Optional[tuple] = None,
    normalize: str = 'none',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'gradient_descent',
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> Dict[str, Union[Dict, float]]:
    """
    Fit ARIMA model to time series data with configurable options.

    Parameters
    ----------
    y : np.ndarray
        Time series data.
    order : tuple, optional
        (p, d, q) order of the ARIMA model.
    seasonal_order : tuple, optional
        (P, D, Q, S) order of the seasonal component.
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to compute ('mse', 'mae', 'r2', or custom callable).
    solver : str, optional
        Solver method ('gradient_descent', 'newton').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    dict
        Structured results including model parameters, metrics, and diagnostics.
    """
    # Validate inputs
    validated_inputs = validate_arima_inputs(y, order, seasonal_order, normalize)
    y_norm = validated_inputs['y']

    # Fit model
    model_result = fit_arima_model(y_norm, order, seasonal_order, solver, tol, max_iter)

    # Compute metrics (placeholder for actual predictions)
    y_pred = np.random.rand(len(y_norm))
    metrics = compute_arima_metrics(y_norm, y_pred, metric)

    return {
        'result': model_result,
        'metrics': metrics,
        'params_used': validated_inputs,
        'warnings': [],
    }

# Example usage
if __name__ == "__main__":
    y = np.random.rand(100)
    result = arima_fit(y, order=(2, 1, 2), normalize='standard', metric='mse')
    print(result)

################################################################################
# sarima
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def sarima_fit(
    y: np.ndarray,
    order: Tuple[int, int, int] = (1, 0, 1),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: Optional[str] = None,
    normalization: str = 'none',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'lstsq',
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict:
    """
    Fit a SARIMA model to time series data.

    Parameters
    ----------
    y : np.ndarray
        Time series data.
    order : tuple of int, optional
        (p, d, q) order of the ARIMA model.
    seasonal_order : tuple of int, optional
        (P, D, Q, S) order of the seasonal component.
    trend : str or None, optional
        Trend type: 'c' (constant), 't' (linear), or None.
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : str or callable, optional
        Metric to evaluate model performance: 'mse', 'mae', 'r2', or custom callable.
    solver : str, optional
        Solver method: 'lstsq', 'gradient_descent', or 'newton'.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    random_state : int or None, optional
        Random seed for reproducibility.
    custom_metric : callable or None, optional
        Custom metric function.

    Returns
    -------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> result = sarima_fit(y)
    """
    # Validate inputs
    _validate_inputs(y, order, seasonal_order)

    # Normalize data if required
    y_normalized = _normalize_data(y, normalization)

    # Fit SARIMA model
    params = _fit_sarima(
        y_normalized,
        order,
        seasonal_order,
        trend,
        solver,
        max_iter,
        tol,
        random_state
    )

    # Calculate metrics
    y_pred = _predict_sarima(y_normalized, params)
    metrics = _calculate_metrics(y_normalized, y_pred, metric, custom_metric)

    # Prepare output
    result = {
        'result': y_pred,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

    return result

def _validate_inputs(
    y: np.ndarray,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array")
    if len(order) != 3:
        raise ValueError("order must be a tuple of length 3 (p, d, q)")
    if len(seasonal_order) != 4:
        raise ValueError("seasonal_order must be a tuple of length 4 (P, D, Q, S)")
    if np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values")

def _normalize_data(
    y: np.ndarray,
    method: str
) -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'none':
        return y
    elif method == 'standard':
        mean = np.mean(y)
        std = np.std(y)
        return (y - mean) / std
    elif method == 'minmax':
        min_val = np.min(y)
        max_val = np.max(y)
        return (y - min_val) / (max_val - min_val)
    elif method == 'robust':
        median = np.median(y)
        iqr = np.percentile(y, 75) - np.percentile(y, 25)
        return (y - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _fit_sarima(
    y: np.ndarray,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    trend: Optional[str],
    solver: str,
    max_iter: int,
    tol: float,
    random_state: Optional[int]
) -> Dict:
    """Fit SARIMA model and return parameters."""
    # Placeholder for actual SARIMA fitting logic
    params = {
        'ar': np.random.rand(order[0]) if order[0] > 0 else None,
        'diff': order[1],
        'ma': np.random.rand(order[2]) if order[2] > 0 else None,
        'seasonal_ar': np.random.rand(seasonal_order[0]) if seasonal_order[0] > 0 else None,
        'seasonal_diff': seasonal_order[1],
        'seasonal_ma': np.random.rand(seasonal_order[2]) if seasonal_order[2] > 0 else None,
        'seasonal_period': seasonal_order[3],
        'trend': trend
    }
    return params

def _predict_sarima(
    y: np.ndarray,
    params: Dict
) -> np.ndarray:
    """Predict using fitted SARIMA model."""
    # Placeholder for actual prediction logic
    return y + np.random.randn(len(y)) * 0.1

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate metrics for model evaluation."""
    metrics = {}

    if metric == 'mse' or (custom_metric is None and metric == 'mse'):
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    if metric == 'mae' or (custom_metric is None and metric == 'mae'):
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric == 'r2' or (custom_metric is None and metric == 'r2'):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)

    return metrics

################################################################################
# prophet
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List
from enum import Enum

class Normalization(Enum):
    NONE = "none"
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"

class Metric(Enum):
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"
    LOGLOSS = "logloss"

class Solver(Enum):
    CLOSED_FORM = "closed_form"
    GRADIENT_DESCENT = "gradient_descent"
    NEWTON = "newton"
    COORDINATE_DESCENT = "coordinate_descent"

class Regularization(Enum):
    NONE = "none"
    L1 = "l1"
    L2 = "l2"
    ELASTICNET = "elasticnet"

def validate_inputs(
    y: np.ndarray,
    X: Optional[np.ndarray] = None,
    normalize: Normalization = Normalization.NONE
) -> Dict[str, np.ndarray]:
    """
    Validate and preprocess input data.

    Parameters:
        y: Target values
        X: Feature matrix (optional)
        normalize: Normalization method

    Returns:
        Dictionary containing validated and processed data
    """
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X is not None and y.shape[0] != X.shape[0]:
        raise ValueError("y and X must have the same number of samples")

    # Handle missing values
    mask = ~np.isnan(y)
    if X is not None:
        mask &= ~np.isnan(X).any(axis=1)

    y_clean = y[mask]
    X_clean = None if X is None else X[mask]

    # Normalization
    y_normalized, X_normalized = apply_normalization(y_clean, X_clean, normalize)

    return {
        "y": y_normalized,
        "X": X_normalized,
        "original_mask": mask
    }

def apply_normalization(
    y: np.ndarray,
    X: Optional[np.ndarray],
    method: Normalization
) -> tuple:
    """
    Apply normalization to data.

    Parameters:
        y: Target values
        X: Feature matrix (optional)
        method: Normalization method

    Returns:
        Tuple of normalized y and X
    """
    if method == Normalization.NONE:
        return y, X

    # Normalize y
    if method == Normalization.STANDARD:
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_normalized = (y - y_mean) / y_std
    elif method == Normalization.MINMAX:
        y_min = np.min(y)
        y_max = np.max(y)
        y_normalized = (y - y_min) / (y_max - y_min)
    elif method == Normalization.ROBUST:
        y_median = np.median(y)
        y_iqr = np.percentile(y, 75) - np.percentile(y, 25)
        y_normalized = (y - y_median) / y_iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Normalize X if provided
    if X is not None:
        if method == Normalization.STANDARD:
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            X_normalized = (X - X_mean) / X_std
        elif method == Normalization.MINMAX:
            X_min = np.min(X, axis=0)
            X_max = np.max(X, axis=0)
            X_normalized = (X - X_min) / (X_max - X_min)
        elif method == Normalization.ROBUST:
            X_median = np.median(X, axis=0)
            X_iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
            X_normalized = (X - X_median) / X_iqr
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    else:
        X_normalized = None

    return y_normalized, X_normalized

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_funcs: Dict[Metric, Callable] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Parameters:
        y_true: True values
        y_pred: Predicted values
        metric_funcs: Dictionary of custom metric functions

    Returns:
        Dictionary of computed metrics
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    metrics = {}

    # Default metrics
    default_metrics = {
        Metric.MSE: lambda yt, yp: np.mean((yt - yp) ** 2),
        Metric.MAE: lambda yt, yp: np.mean(np.abs(yt - yp)),
        Metric.R2: lambda yt, yp: 1 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2),
        Metric.LOGLOSS: lambda yt, yp: -np.mean(yt * np.log(yp) + (1 - yt) * np.log(1 - yp))
    }

    # Use provided metrics or defaults
    metric_funcs = metric_funcs if metric_funcs is not None else default_metrics

    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(y_true, y_pred)
        except Exception as e:
            metrics[f"{name}_error"] = str(e)

    return metrics

def fit_model(
    y: np.ndarray,
    X: Optional[np.ndarray] = None,
    solver: Solver = Solver.CLOSED_FORM,
    regularization: Regularization = Regularization.NONE,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Dict[str, np.ndarray]:
    """
    Fit the model to the data.

    Parameters:
        y: Target values
        X: Feature matrix (optional)
        solver: Solver method
        regularization: Regularization type
        alpha: Regularization strength
        l1_ratio: Elastic net mixing parameter
        max_iter: Maximum iterations
        tol: Tolerance for convergence

    Returns:
        Dictionary containing model parameters and fit information
    """
    if solver == Solver.CLOSED_FORM:
        params = closed_form_solution(y, X)
    elif solver == Solver.GRADIENT_DESCENT:
        params = gradient_descent(y, X, max_iter, tol)
    elif solver == Solver.NEWTON:
        params = newton_method(y, X, max_iter, tol)
    elif solver == Solver.COORDINATE_DESCENT:
        params = coordinate_descent(y, X, max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if needed
    if regularization != Regularization.NONE:
        params = apply_regularization(params, y, X, regularization, alpha, l1_ratio)

    return {
        "params": params,
        "converged": True,  # In real implementation, track convergence
        "iterations": max_iter
    }

def closed_form_solution(
    y: np.ndarray,
    X: Optional[np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Compute closed form solution for linear regression.

    Parameters:
        y: Target values
        X: Feature matrix (optional)

    Returns:
        Dictionary containing model parameters
    """
    if X is None:
        # Simple mean for no features
        return {"intercept": np.mean(y)}

    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    params, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)

    return {
        "intercept": params[0],
        "coefficients": params[1:]
    }

def gradient_descent(
    y: np.ndarray,
    X: Optional[np.ndarray],
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Dict[str, np.ndarray]:
    """
    Perform gradient descent optimization.

    Parameters:
        y: Target values
        X: Feature matrix (optional)
        max_iter: Maximum iterations
        tol: Tolerance for convergence

    Returns:
        Dictionary containing model parameters
    """
    if X is None:
        # Simple mean for no features
        return {"intercept": np.mean(y)}

    n_samples, n_features = X.shape
    theta = np.zeros(n_features + 1)  # +1 for intercept

    learning_rate = 0.01
    prev_loss = float('inf')

    for _ in range(max_iter):
        # Compute predictions
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
        predictions = X_with_intercept @ theta

        # Compute gradients
        error = predictions - y
        gradient = (X_with_intercept.T @ error) / n_samples

        # Update parameters
        theta -= learning_rate * gradient

        # Check convergence
        current_loss = np.mean(error ** 2)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return {
        "intercept": theta[0],
        "coefficients": theta[1:]
    }

def newton_method(
    y: np.ndarray,
    X: Optional[np.ndarray],
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Dict[str, np.ndarray]:
    """
    Perform Newton's method optimization.

    Parameters:
        y: Target values
        X: Feature matrix (optional)
        max_iter: Maximum iterations
        tol: Tolerance for convergence

    Returns:
        Dictionary containing model parameters
    """
    if X is None:
        # Simple mean for no features
        return {"intercept": np.mean(y)}

    n_samples, n_features = X.shape
    theta = np.zeros(n_features + 1)  # +1 for intercept

    prev_loss = float('inf')

    for _ in range(max_iter):
        # Compute predictions
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
        predictions = X_with_intercept @ theta

        # Compute gradients and Hessian
        error = predictions - y
        gradient = (X_with_intercept.T @ error) / n_samples

        hessian = (X_with_intercept.T @ X_with_intercept) / n_samples

        # Update parameters
        theta -= np.linalg.solve(hessian, gradient)

        # Check convergence
        current_loss = np.mean(error ** 2)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return {
        "intercept": theta[0],
        "coefficients": theta[1:]
    }

def coordinate_descent(
    y: np.ndarray,
    X: Optional[np.ndarray],
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Dict[str, np.ndarray]:
    """
    Perform coordinate descent optimization.

    Parameters:
        y: Target values
        X: Feature matrix (optional)
        max_iter: Maximum iterations
        tol: Tolerance for convergence

    Returns:
        Dictionary containing model parameters
    """
    if X is None:
        # Simple mean for no features
        return {"intercept": np.mean(y)}

    n_samples, n_features = X.shape
    theta = np.zeros(n_features + 1)  # +1 for intercept

    prev_loss = float('inf')

    for _ in range(max_iter):
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
        predictions = X_with_intercept @ theta
        error = predictions - y

        for i in range(n_features + 1):
            # Compute residual without current feature
            residual = error - X_with_intercept[:, i] * theta[i]

            # Compute optimal value for current feature
            if np.sum(X_with_intercept[:, i] ** 2) > 0:
                theta[i] = np.sum(X_with_intercept[:, i] * residual) / np.sum(X_with_intercept[:, i] ** 2)

        # Check convergence
        current_loss = np.mean(error ** 2)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return {
        "intercept": theta[0],
        "coefficients": theta[1:]
    }

def apply_regularization(
    params: Dict[str, np.ndarray],
    y: np.ndarray,
    X: Optional[np.ndarray],
    method: Regularization,
    alpha: float = 1.0,
    l1_ratio: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Apply regularization to model parameters.

    Parameters:
        params: Current model parameters
        y: Target values
        X: Feature matrix (optional)
        method: Regularization type
        alpha: Regularization strength
        l1_ratio: Elastic net mixing parameter

    Returns:
        Dictionary containing regularized parameters
    """
    if X is None:
        return params  # No features to regularize

    n_samples = len(y)
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    theta = np.concatenate([[params['intercept']], params['coefficients']])

    if method == Regularization.L1:
        # Lasso regularization
        for i in range(len(theta)):
            if X_with_intercept[:, i].var() > 0:
                theta[i] = np.sign(theta[i]) * max(abs(theta[i]) - alpha, 0)
    elif method == Regularization.L2:
        # Ridge regularization
        pass  # Handled by the solver
    elif method == Regularization.ELASTICNET:
        # Elastic net regularization
        for i in range(len(theta)):
            if X_with_intercept[:, i].var() > 0:
                theta[i] = np.sign(theta[i]) * max(abs(theta[i]) - alpha * l1_ratio, 0)
                theta[i] *= (1 - alpha) / (1 - alpha * l1_ratio)

    return {
        "intercept": theta[0],
        "coefficients": theta[1:]
    }

def prophet_fit(
    y: np.ndarray,
    X: Optional[np.ndarray] = None,
    normalize: Normalization = Normalization.NONE,
    solver: Solver = Solver.CLOSED_FORM,
    regularization: Regularization = Regularization.NONE,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4,
    metric_funcs: Optional[Dict[Metric, Callable]] = None
) -> Dict:
    """
    Main function to fit a Prophet model.

    Parameters:
        y: Target values
        X: Feature matrix (optional)
        normalize: Normalization method
        solver: Solver method
        regularization: Regularization type
        alpha: Regularization strength
        l1_ratio: Elastic net mixing parameter
        max_iter: Maximum iterations
        tol: Tolerance for convergence
        metric_funcs: Custom metric functions

    Returns:
        Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    processed_data = validate_inputs(y, X, normalize)
    y_processed = processed_data["y"]
    X_processed = processed_data["X"]

    # Fit model
    fit_result = fit_model(
        y_processed,
        X_processed,
        solver=solver,
        regularization=regularization,
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=tol
    )

    # Make predictions
    if X_processed is None:
        y_pred = np.full_like(y_processed, fit_result["params"]["intercept"])
    else:
        X_with_intercept = np.column_stack([np.ones(len(X_processed)), X_processed])
        y_pred = X_with_intercept @ np.concatenate([
            [fit_result["params"]["intercept"]],
            fit_result["params"]["coefficients"]
        ])

    # Compute metrics
    metrics = compute_metrics(y_processed, y_pred, metric_funcs)

    # Reconstruct original scale predictions if needed
    if normalize != Normalization.NONE:
        y_pred_original = inverse_normalize(y_pred, X_processed is not None, normalize)
    else:
        y_pred_original = y_pred

    return {
        "result": {
            "params": fit_result["params"],
            "predictions": y_pred_original,
            "converged": fit_result["converged"],
            "iterations": fit_result["iterations"]
        },
        "metrics": metrics,
        "params_used": {
            "normalize": normalize.value,
            "solver": solver.value,
            "regularization": regularization.value,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "max_iter": max_iter,
            "tol": tol
        },
        "warnings": []
    }

def inverse_normalize(
    y: np.ndarray,
    has_features: bool,
    method: Normalization
) -> np.ndarray:
    """
    Inverse normalization of data.

    Parameters:
        y: Normalized values
        has_features: Whether features were present (for intercept)
        method: Normalization method used

    Returns:
        Original scale values
    """
    if method == Normalization.NONE:
        return y

    # This would require storing the original statistics
    # In a real implementation, you'd store these during normalization
    raise NotImplementedError("Inverse normalization requires storing original statistics")

# Example usage:
# y = np.array([1, 2, 3, 4, 5])

################################################################################
# acf_pacf
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(series: np.ndarray) -> None:
    """Validate input series for ACF/PACF computation."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input series must be a numpy array")
    if len(series.shape) != 1:
        raise ValueError("Input series must be one-dimensional")
    if np.any(np.isnan(series)) or np.any(np.isinf(series)):
        raise ValueError("Input series contains NaN or infinite values")

def compute_acf(series: np.ndarray, n_lags: int = 10,
                normalization: str = 'standard') -> np.ndarray:
    """Compute Autocorrelation Function (ACF)."""
    n = len(series)
    mean = np.mean(series)

    def standard_normalization(x: float) -> float:
        return x / (n - np.arange(n_lags + 1))

    def no_normalization(x: float) -> float:
        return x

    normalization_funcs = {
        'standard': standard_normalization,
        'none': no_normalization
    }

    if normalization not in normalization_funcs:
        raise ValueError(f"Unknown normalization: {normalization}")

    acf = np.zeros(n_lags + 1)
    for lag in range(n_lags + 1):
        if lag == 0:
            acf[lag] = normalization_funcs[normalization](np.sum((series - mean) ** 2))
        else:
            cov = np.sum((series[lag:] - mean) * (series[:-lag] - mean))
            acf[lag] = normalization_funcs[normalization](cov)

    return acf

def compute_pacf(series: np.ndarray, n_lags: int = 10,
                 method: str = 'yw') -> np.ndarray:
    """Compute Partial Autocorrelation Function (PACF)."""
    n = len(series)
    pacf = np.zeros(n_lags + 1)

    if method == 'yw':
        for lag in range(1, n_lags + 1):
            residuals = series[lag:]
            for i in range(1, lag):
                residuals -= pacf[i] * series[lag - 1 - i:]
            pacf[lag] = np.corrcoef(series[:n - lag], residuals)[0, 1]
    else:
        raise ValueError(f"Unknown PACF method: {method}")

    return pacf

def acf_pacf_fit(series: np.ndarray,
                 n_lags: int = 10,
                 acf_normalization: str = 'standard',
                 pacf_method: str = 'yw') -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Compute ACF and PACF for a time series.

    Parameters
    ----------
    series : np.ndarray
        Input time series.
    n_lags : int, optional
        Number of lags to compute (default: 10).
    acf_normalization : str, optional
        Normalization for ACF ('standard' or 'none', default: 'standard').
    pacf_method : str, optional
        Method for PACF computation ('yw', default: 'yw').

    Returns
    -------
    dict
        Dictionary containing:
        - "acf": computed ACF values
        - "pacf": computed PACF values
        - "params_used": parameters used in computation
        - "warnings": any warnings generated

    Example
    -------
    >>> series = np.random.randn(100)
    >>> result = acf_pacf_fit(series, n_lags=5)
    """
    warnings = []
    params_used = {
        'n_lags': n_lags,
        'acf_normalization': acf_normalization,
        'pacf_method': pacf_method
    }

    try:
        validate_input(series)
    except Exception as e:
        warnings.append(str(e))

    if len(warnings) > 0:
        return {
            "acf": None,
            "pacf": None,
            "params_used": params_used,
            "warnings": warnings
        }

    acf = compute_acf(series, n_lags, acf_normalization)
    pacf = compute_pacf(series, n_lags, pacf_method)

    return {
        "acf": acf,
        "pacf": pacf,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# test_dickey_fuller
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    series: np.ndarray,
    max_lag: int = 1,
    regression: str = 'c',
    autolag: Optional[str] = None,
    trend: str = 'c'
) -> Dict[str, np.ndarray]:
    """
    Validate and preprocess inputs for the Dickey-Fuller test.

    Parameters
    ----------
    series : np.ndarray
        Input time series data.
    max_lag : int, optional
        Maximum lag to consider for the test, by default 1.
    regression : str, optional
        Type of regression to include ('c', 'ct', 'ctt'), by default 'c'.
    autolag : str, optional
        Method to determine lag ('AIC', 'BIC', None), by default None.
    trend : str, optional
        Trend to include ('c', 'ct', 'ctt'), by default 'c'.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing validated and processed inputs.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(series, np.ndarray):
        raise ValueError("Input series must be a numpy array.")
    if len(series.shape) != 1:
        raise ValueError("Input series must be one-dimensional.")
    if np.any(np.isnan(series)) or np.any(np.isinf(series)):
        raise ValueError("Input series must not contain NaN or inf values.")

    # Additional validation logic can be added here

    return {
        'series': series,
        'max_lag': max_lag,
        'regression': regression,
        'autolag': autolag,
        'trend': trend
    }

def _compute_statistic(
    series: np.ndarray,
    lag: int,
    regression: str = 'c',
    trend: str = 'c'
) -> float:
    """
    Compute the Dickey-Fuller test statistic.

    Parameters
    ----------
    series : np.ndarray
        Input time series data.
    lag : int
        Lag to use for the test.
    regression : str, optional
        Type of regression to include ('c', 'ct', 'ctt'), by default 'c'.
    trend : str, optional
        Trend to include ('c', 'ct', 'ctt'), by default 'c'.

    Returns
    -------
    float
        Computed Dickey-Fuller test statistic.
    """
    # Implementation of the Dickey-Fuller test statistic computation
    # This is a placeholder for the actual implementation

    return 0.0  # Placeholder value

def _estimate_parameters(
    series: np.ndarray,
    lag: int,
    regression: str = 'c',
    trend: str = 'c'
) -> Dict[str, float]:
    """
    Estimate parameters for the Dickey-Fuller test.

    Parameters
    ----------
    series : np.ndarray
        Input time series data.
    lag : int
        Lag to use for the test.
    regression : str, optional
        Type of regression to include ('c', 'c', 'ctt'), by default 'c'.
    trend : str, optional
        Trend to include ('c', 'ct', 'ctt'), by default 'c'.

    Returns
    -------
    Dict[str, float]
        Dictionary containing estimated parameters.
    """
    # Implementation of parameter estimation
    # This is a placeholder for the actual implementation

    return {'param1': 0.0, 'param2': 0.0}  # Placeholder values

def _compute_metrics(
    series: np.ndarray,
    params: Dict[str, float],
    statistic: float
) -> Dict[str, float]:
    """
    Compute metrics for the Dickey-Fuller test.

    Parameters
    ----------
    series : np.ndarray
        Input time series data.
    params : Dict[str, float]
        Estimated parameters from the test.
    statistic : float
        Computed Dickey-Fuller test statistic.

    Returns
    -------
    Dict[str, float]
        Dictionary containing computed metrics.
    """
    # Implementation of metric computation
    # This is a placeholder for the actual implementation

    return {'metric1': 0.0, 'metric2': 0.0}  # Placeholder values

def test_dickey_fuller_fit(
    series: np.ndarray,
    max_lag: int = 1,
    regression: str = 'c',
    autolag: Optional[str] = None,
    trend: str = 'c',
    custom_statistic_func: Optional[Callable] = None,
    custom_metrics_func: Optional[Callable] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Perform the Dickey-Fuller test on a time series.

    Parameters
    ----------
    series : np.ndarray
        Input time series data.
    max_lag : int, optional
        Maximum lag to consider for the test, by default 1.
    regression : str, optional
        Type of regression to include ('c', 'ct', 'ctt'), by default 'c'.
    autolag : str, optional
        Method to determine lag ('AIC', 'BIC', None), by default None.
    trend : str, optional
        Trend to include ('c', 'ct', 'ctt'), by default 'c'.
    custom_statistic_func : Callable, optional
        Custom function to compute the test statistic, by default None.
    custom_metrics_func : Callable, optional
        Custom function to compute metrics, by default None.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], str]]
        Dictionary containing the test results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> series = np.random.randn(100)
    >>> result = test_dickey_fuller_fit(series)
    """
    # Validate inputs
    validated_inputs = _validate_inputs(
        series, max_lag, regression, autolag, trend
    )

    # Determine lag if autolag is specified
    lag = validated_inputs['max_lag']
    if validated_inputs['autolag'] is not None:
        # Placeholder for autolag determination logic
        pass

    # Compute test statistic
    if custom_statistic_func is not None:
        statistic = custom_statistic_func(
            validated_inputs['series'], lag, regression, trend
        )
    else:
        statistic = _compute_statistic(
            validated_inputs['series'], lag, regression, trend
        )

    # Estimate parameters
    params = _estimate_parameters(
        validated_inputs['series'], lag, regression, trend
    )

    # Compute metrics
    if custom_metrics_func is not None:
        metrics = custom_metrics_func(
            validated_inputs['series'], params, statistic
        )
    else:
        metrics = _compute_metrics(
            validated_inputs['series'], params, statistic
        )

    # Prepare output dictionary
    result = {
        'result': {
            'statistic': statistic,
            'p_value': 0.0,  # Placeholder for p-value
            'critical_values': {},  # Placeholder for critical values
            'lag': lag,
            'regression': regression,
            'trend': trend
        },
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

    return result

################################################################################
# kpss_test
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(series: np.ndarray) -> None:
    """Validate input series for KPSS test."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if series.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array")
    if np.isnan(series).any() or np.isinf(series).any():
        raise ValueError("Input contains NaN or infinite values")

def _compute_kpss_statistic(series: np.ndarray, regression: str = 'c') -> float:
    """Compute the KPSS test statistic."""
    n = len(series)
    if regression == 'c':
        # Constant regression
        y = series - np.mean(series)
    elif regression == 'ct':
        # Constant and trend regression
        x = np.vstack([np.ones(n), np.arange(1, n+1)]).T
        coeffs = np.linalg.lstsq(x, series, rcond=None)[0]
        y = series - x @ coeffs
    else:
        raise ValueError("Regression must be either 'c' or 'ct'")

    s = np.cumsum(y)
    numerator = np.sum(s**2) - (np.sum(s)**2)/n
    denominator = np.sum(y**2)
    kpss_stat = (n * numerator) / (denominator + 1e-10)
    return kpss_stat

def _compute_critical_values(regression: str = 'c') -> Dict[str, float]:
    """Compute critical values for KPSS test."""
    if regression == 'c':
        return {'1%': 0.739, '5%': 0.463, '10%': 0.347}
    elif regression == 'ct':
        return {'1%': 0.216, '5%': 0.146, '10%': 0.119}
    else:
        raise ValueError("Regression must be either 'c' or 'ct'")

def kpss_test_fit(
    series: np.ndarray,
    regression: str = 'c',
    custom_metric: Optional[Callable[[np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Perform KPSS test for stationarity.

    Parameters:
    -----------
    series : np.ndarray
        Input time series data.
    regression : str, optional
        Type of regression ('c' for constant, 'ct' for constant and trend).
    custom_metric : callable, optional
        Custom metric function to compute additional statistics.

    Returns:
    --------
    dict
        Dictionary containing test results, metrics, parameters used, and warnings.
    """
    _validate_input(series)

    # Compute KPSS statistic
    kpss_stat = _compute_kpss_statistic(series, regression)

    # Compute critical values
    critical_values = _compute_critical_values(regression)

    # Compute custom metric if provided
    metrics = {}
    if custom_metric is not None:
        try:
            metrics['custom'] = custom_metric(series)
        except Exception as e:
            warnings.append(f"Custom metric computation failed: {str(e)}")

    # Determine result
    p_values = {
        '1%': kpss_stat > critical_values['1%'],
        '5%': kpss_stat > critical_values['5%'],
        '10%': kpss_stat > critical_values['10%']
    }

    result = {
        'statistic': kpss_stat,
        'critical_values': critical_values,
        'p_values': p_values
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {'regression': regression},
        'warnings': []
    }

# Example usage:
"""
series = np.random.randn(100)
result = kpss_test_fit(series, regression='c')
print(result)
"""

################################################################################
# bruit_blanc
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    series: np.ndarray,
    window_size: int = 10,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> None:
    """Validate input series and parameters."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input series must be a numpy array")
    if len(series.shape) != 1:
        raise ValueError("Input series must be 1-dimensional")
    if np.any(np.isnan(series)) or np.any(np.isinf(series)):
        raise ValueError("Input series contains NaN or infinite values")
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    if normalizer is not None and not callable(normalizer):
        raise TypeError("Normalizer must be a callable or None")

def _apply_normalization(
    series: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Apply normalization to the series."""
    if normalizer is not None:
        return normalizer(series)
    return series

def _compute_acf(
    series: np.ndarray,
    window_size: int = 10
) -> np.ndarray:
    """Compute autocorrelation function."""
    n = len(series)
    acf = np.zeros(window_size + 1)

    for lag in range(window_size + 1):
        if lag == 0:
            acf[lag] = np.corrcoef(series, series)[0, 1]
        else:
            acf[lag] = np.corrcoef(series[:-lag], series[lag:])[0, 1]

    return acf

def _compute_metrics(
    acf: np.ndarray,
    metric_funcs: Dict[str, Callable[[np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for the ACF."""
    return {name: func(acf) for name, func in metric_funcs.items()}

def bruit_blanc_fit(
    series: np.ndarray,
    window_size: int = 10,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric_funcs: Dict[str, Callable[[np.ndarray], float]] = None
) -> Dict[str, Union[Dict[str, float], np.ndarray, Dict[str, str]]]:
    """
    Test if a time series is white noise.

    Parameters:
    -----------
    series : np.ndarray
        Input time series.
    window_size : int, optional
        Size of the window for autocorrelation calculation (default: 10).
    normalizer : callable, optional
        Function to normalize the series (default: None).
    metric_funcs : dict, optional
        Dictionary of metric functions to compute (default: None).

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(series, window_size, normalizer)

    # Apply normalization if specified
    normalized_series = _apply_normalization(series, normalizer)

    # Compute ACF
    acf = _compute_acf(normalized_series, window_size)

    # Compute metrics if specified
    metrics = {}
    if metric_funcs is not None:
        metrics = _compute_metrics(acf, metric_funcs)

    # Prepare output
    result = {
        "result": acf,
        "metrics": metrics,
        "params_used": {
            "window_size": window_size,
            "normalizer": normalizer.__name__ if normalizer else None
        },
        "warnings": []
    }

    return result

# Example usage:
"""
import numpy as np
from typing import Dict, Callable

def standard_normalizer(series: np.ndarray) -> np.ndarray:
    return (series - np.mean(series)) / np.std(series)

def mse_metric(acf: np.ndarray) -> float:
    return np.mean((acf[1:] - 0)**2)

series = np.random.normal(0, 1, 100)
metric_funcs = {"mse": mse_metric}

result = bruit_blanc_fit(
    series=series,
    window_size=10,
    normalizer=standard_normalizer,
    metric_funcs=metric_funcs
)
"""

################################################################################
# forecasting
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def forecasting_fit(
    y: np.ndarray,
    X: Optional[np.ndarray] = None,
    normalizer: str = "standard",
    metric: Union[str, Callable] = "mse",
    solver: str = "gradient_descent",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a forecasting model to time series data.

    Parameters:
    -----------
    y : np.ndarray
        Target time series values.
    X : Optional[np.ndarray]
        Feature matrix (if None, only y is used).
    normalizer : str
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : Union[str, Callable]
        Metric to optimize: "mse", "mae", "r2", or custom callable.
    solver : str
        Solver method: "closed_form", "gradient_descent", "newton", or "coordinate_descent".
    regularization : Optional[str]
        Regularization type: "none", "l1", "l2", or "elasticnet".
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable]
        Custom metric function if not using built-in metrics.
    custom_distance : Optional[Callable]
        Custom distance function for solver.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(y, X)

    # Normalize data
    y_norm, X_norm = _apply_normalization(y, X, normalizer)

    # Initialize parameters
    params = _initialize_parameters(X_norm)

    # Choose solver and optimize
    if solver == "closed_form":
        params = _solve_closed_form(y_norm, X_norm)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(
            y_norm, X_norm, metric, regularization,
            tol, max_iter, custom_metric, custom_distance
        )
    # Add other solvers as needed

    # Compute metrics
    metrics = _compute_metrics(y_norm, X_norm @ params if X is not None else y_norm, metric)

    # Prepare output
    result = {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer,
            "metric": metric,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return result

def _validate_inputs(y: np.ndarray, X: Optional[np.ndarray]) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(y, np.ndarray) or (X is not None and not isinstance(X, np.ndarray)):
        raise TypeError("Inputs must be numpy arrays.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X is not None and X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X is not None and y.shape[0] != X.shape[0]:
        raise ValueError("y and X must have the same number of samples.")
    if np.any(np.isnan(y)) or (X is not None and np.any(np.isnan(X))):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(y)) or (X is not None and np.any(np.isinf(X))):
        raise ValueError("Input arrays must not contain infinite values.")

def _apply_normalization(
    y: np.ndarray,
    X: Optional[np.ndarray],
    method: str
) -> tuple:
    """Apply normalization to input data."""
    if method == "none":
        return y, X
    elif method == "standard":
        y_norm = (y - np.mean(y)) / np.std(y)
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0) if X is not None else None
    elif method == "minmax":
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0)) if X is not None else None
    elif method == "robust":
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)) if X is not None else None
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return y_norm, X_norm

def _initialize_parameters(X: Optional[np.ndarray]) -> np.ndarray:
    """Initialize model parameters."""
    if X is None:
        return np.zeros(1)
    n_features = X.shape[1]
    return np.random.randn(n_features)

def _solve_closed_form(y: np.ndarray, X: Optional[np.ndarray]) -> np.ndarray:
    """Solve using closed-form solution (normal equations)."""
    if X is None:
        return np.array([np.mean(y)])
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(
    y: np.ndarray,
    X: Optional[np.ndarray],
    metric: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric: Optional[Callable],
    custom_distance: Optional[Callable]
) -> np.ndarray:
    """Solve using gradient descent."""
    params = _initialize_parameters(X)
    for _ in range(max_iter):
        grad = _compute_gradient(y, X, params, metric, regularization, custom_metric)
        if np.linalg.norm(grad) < tol:
            break
        params -= 0.01 * grad
    return params

def _compute_gradient(
    y: np.ndarray,
    X: Optional[np.ndarray],
    params: np.ndarray,
    metric: Union[str, Callable],
    regularization: Optional[str],
    custom_metric: Optional[Callable]
) -> np.ndarray:
    """Compute gradient for optimization."""
    if X is None:
        raise NotImplementedError("Gradient computation not implemented for univariate case.")
    predictions = X @ params
    if metric == "mse":
        error = predictions - y
        grad = 2 * X.T @ error / len(y)
    elif metric == "mae":
        grad = np.sign(predictions - y) @ X / len(y)
    elif callable(metric):
        grad = custom_metric(predictions, y, X)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if regularization == "l1":
        grad += np.sign(params)
    elif regularization == "l2":
        grad += 2 * params
    # Add other regularizations as needed

    return grad

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Compute evaluation metrics."""
    metrics = {}
    if metric == "mse":
        metrics["mse"] = np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        metrics["mae"] = np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics["r2"] = 1 - ss_res / ss_tot
    elif callable(metric):
        metrics["custom"] = metric(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics

################################################################################
# retropropagation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def retropropagation_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = "mse",
    distance: Union[str, Callable] = "euclidean",
    solver: str = "gradient_descent",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a model using backpropagation for time series data.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalizer : Optional[Callable], default=None
        Function to normalize the input data.
    metric : Union[str, Callable], default="mse"
        Metric to evaluate the model. Can be "mse", "mae", "r2", or a custom callable.
    distance : Union[str, Callable], default="euclidean"
        Distance metric for the loss function. Can be "euclidean", "manhattan", "cosine",
        "minkowski", or a custom callable.
    solver : str, default="gradient_descent"
        Solver to use. Can be "closed_form", "gradient_descent", "newton", or
        "coordinate_descent".
    regularization : Optional[str], default=None
        Regularization type. Can be "l1", "l2", or "elasticnet".
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    max_iter : int, default=1000
        Maximum number of iterations.
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    custom_metric : Optional[Callable], default=None
        Custom metric function if not using built-in metrics.
    custom_distance : Optional[Callable], default=None
        Custom distance function if not using built-in distances.

    Returns
    -------
    Dict
        Dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = retropropagation_fit(X, y)
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
    if solver == "closed_form":
        weights, bias = _closed_form_solver(X, y)
    elif solver == "gradient_descent":
        weights, bias = _gradient_descent(
            X, y, weights, bias, learning_rate, tol, max_iter,
            regularization, distance, metric, custom_metric, custom_distance
        )
    elif solver == "newton":
        weights, bias = _newton_solver(X, y, weights, bias, tol, max_iter)
    elif solver == "coordinate_descent":
        weights, bias = _coordinate_descent(
            X, y, weights, bias, tol, max_iter,
            regularization, distance, metric, custom_metric, custom_distance
        )
    else:
        raise ValueError("Invalid solver specified.")

    # Calculate metrics
    metrics = _calculate_metrics(X, y, weights, bias, metric, custom_metric)

    # Prepare results
    result = {
        "result": {"weights": weights, "bias": bias},
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter,
            "learning_rate": learning_rate
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
        raise ValueError("X contains NaN or Inf values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or Inf values.")

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> tuple:
    """Solve using closed-form solution."""
    X_tx = np.dot(X.T, X)
    if np.linalg.det(X_tx) == 0:
        raise ValueError("Matrix is singular.")
    weights = np.linalg.solve(X_tx, np.dot(X.T, y))
    bias = 0.0
    return weights, bias

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    learning_rate: float,
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    distance: Union[str, Callable],
    metric: Union[str, Callable],
    custom_metric: Optional[Callable],
    custom_distance: Optional[Callable]
) -> tuple:
    """Perform gradient descent optimization."""
    for _ in range(max_iter):
        # Compute predictions
        y_pred = np.dot(X, weights) + bias

        # Compute gradients
        if custom_distance is not None:
            grad_weights = _compute_custom_gradient(X, y, y_pred, custom_distance)
        else:
            grad_weights = _compute_gradient(X, y, y_pred, distance)

        if regularization == "l1":
            grad_weights += np.sign(weights)
        elif regularization == "l2":
            grad_weights += 2 * weights
        elif regularization == "elasticnet":
            grad_weights += np.sign(weights) + 2 * weights

        # Update weights and bias
        weights -= learning_rate * grad_weights
        bias -= learning_rate * np.mean(y_pred - y)

        # Check convergence
        if np.linalg.norm(grad_weights) < tol:
            break

    return weights, bias

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    distance: str
) -> np.ndarray:
    """Compute gradient based on the specified distance metric."""
    if distance == "euclidean":
        grad = np.dot(X.T, (y_pred - y)) / X.shape[0]
    elif distance == "manhattan":
        grad = np.dot(X.T, np.sign(y_pred - y)) / X.shape[0]
    elif distance == "cosine":
        grad = np.dot(X.T, (y_pred - y) / (np.linalg.norm(y_pred) * np.linalg.norm(y)))
    elif distance == "minkowski":
        grad = np.dot(X.T, (y_pred - y) ** 2) / X.shape[0]
    else:
        raise ValueError("Invalid distance metric specified.")
    return grad

def _compute_custom_gradient(
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    custom_distance: Callable
) -> np.ndarray:
    """Compute gradient using a custom distance function."""
    return np.dot(X.T, custom_distance(y_pred, y)) / X.shape[0]

def _newton_solver(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    tol: float,
    max_iter: int
) -> tuple:
    """Perform Newton's method optimization."""
    for _ in range(max_iter):
        # Compute predictions
        y_pred = np.dot(X, weights) + bias

        # Compute Hessian and gradient
        hessian = np.dot(X.T, X) / X.shape[0]
        grad = np.dot(X.T, (y_pred - y)) / X.shape[0]

        # Update weights and bias
        if np.linalg.det(hessian) == 0:
            raise ValueError("Hessian is singular.")
        weights -= np.linalg.solve(hessian, grad)
        bias -= np.mean(y_pred - y)

        # Check convergence
        if np.linalg.norm(grad) < tol:
            break

    return weights, bias

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    distance: Union[str, Callable],
    metric: Union[str, Callable],
    custom_metric: Optional[Callable],
    custom_distance: Optional[Callable]
) -> tuple:
    """Perform coordinate descent optimization."""
    for _ in range(max_iter):
        for i in range(X.shape[1]):
            # Compute predictions
            y_pred = np.dot(X, weights) + bias

            # Compute gradient for the i-th feature
            if custom_distance is not None:
                grad_i = np.dot(X[:, i], (custom_distance(y_pred, y)))
            else:
                grad_i = np.dot(X[:, i], (y_pred - y))

            if regularization == "l1":
                grad_i += np.sign(weights[i])
            elif regularization == "l2":
                grad_i += 2 * weights[i]
            elif regularization == "elasticnet":
                grad_i += np.sign(weights[i]) + 2 * weights[i]

            # Update the i-th weight
            weights[i] -= grad_i / np.linalg.norm(X[:, i]) ** 2

        # Update bias
        bias -= np.mean(y_pred - y)

        # Check convergence
        if np.linalg.norm(grad_i) < tol:
            break

    return weights, bias

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate metrics based on the specified metric."""
    y_pred = np.dot(X, weights) + bias
    metrics = {}

    if custom_metric is not None:
        metrics["custom"] = custom_metric(y, y_pred)
    else:
        if metric == "mse":
            metrics["mse"] = np.mean((y_pred - y) ** 2)
        elif metric == "mae":
            metrics["mae"] = np.mean(np.abs(y_pred - y))
        elif metric == "r2":
            metrics["r2"] = 1 - np.sum((y_pred - y) ** 2) / np.sum((y - np.mean(y)) ** 2)
        elif metric == "logloss":
            metrics["logloss"] = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        else:
            raise ValueError("Invalid metric specified.")

    return metrics

################################################################################
# transformee_fourier
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(x: np.ndarray) -> None:
    """Validate input array for Fourier transform."""
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if x.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array")
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Input array contains NaN or infinite values")

def _normalize_data(x: np.ndarray, normalization: str) -> np.ndarray:
    """Normalize the input data according to specified method."""
    if normalization == "none":
        return x
    elif normalization == "standard":
        mean = np.mean(x)
        std = np.std(x)
        return (x - mean) / std
    elif normalization == "minmax":
        min_val = np.min(x)
        max_val = np.max(x)
        return (x - min_val) / (max_val - min_val)
    elif normalization == "robust":
        median = np.median(x)
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
        return (x - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_fourier_transform(x: np.ndarray) -> np.ndarray:
    """Compute the Fourier transform of the input signal."""
    return np.fft.fft(x)

def _compute_inverse_fourier_transform(coeffs: np.ndarray, n: int) -> np.ndarray:
    """Compute the inverse Fourier transform."""
    return np.fft.ifft(coeffs, n=n)

def _compute_metrics(x: np.ndarray, x_reconstructed: np.ndarray,
                     metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute specified metrics between original and reconstructed signal."""
    metrics = {}
    for name, func in metric_funcs.items():
        if callable(func):
            metrics[name] = func(x, x_reconstructed)
    return metrics

def transformee_fourier_fit(
    x: np.ndarray,
    normalization: str = "none",
    metric_funcs: Optional[Dict[str, Callable]] = None,
    custom_metrics: Optional[Dict[str, Callable]] = None
) -> Dict:
    """
    Compute the Fourier transform of a time series with configurable options.

    Parameters:
    - x: Input time series as numpy array
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - metric_funcs: Dictionary of built-in metrics to compute
    - custom_metrics: Dictionary of custom metric functions

    Returns:
    - Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate input
    _validate_input(x)

    # Initialize default metrics if none provided
    if metric_funcs is None:
        metric_funcs = {
            'mse': lambda x, y: np.mean((x - y) ** 2),
            'mae': lambda x, y: np.mean(np.abs(x - y))
        }

    # Normalize data
    x_normalized = _normalize_data(x, normalization)

    # Compute Fourier transform
    fourier_coeffs = _compute_fourier_transform(x_normalized)

    # Compute inverse transform to reconstruct signal
    x_reconstructed = _compute_inverse_fourier_transform(fourier_coeffs, n=len(x))

    # Compute metrics
    all_metrics = {**metric_funcs}
    if custom_metrics:
        all_metrics.update(custom_metrics)
    metrics = _compute_metrics(x, x_reconstructed, all_metrics)

    # Prepare output
    result = {
        "fourier_coefficients": fourier_coeffs,
        "reconstructed_signal": x_reconstructed
    }

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization
        },
        "warnings": []
    }

################################################################################
# ondelettes
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    signal: np.ndarray,
    wavelet: np.ndarray,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'mse'
) -> None:
    """Validate input data and parameters."""
    if not isinstance(signal, np.ndarray) or not isinstance(wavelet, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if signal.ndim != 1 or wavelet.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional")
    if len(signal) < len(wavelet):
        raise ValueError("Signal must be longer than wavelet")
    if normalization not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("Invalid normalization method")
    if isinstance(metric, str) and metric not in ['mse', 'mae', 'r2']:
        raise ValueError("Invalid metric")

def _normalize_signal(
    signal: np.ndarray,
    method: str = 'none'
) -> np.ndarray:
    """Normalize the input signal."""
    if method == 'standard':
        return (signal - np.mean(signal)) / np.std(signal)
    elif method == 'minmax':
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    elif method == 'robust':
        median = np.median(signal)
        iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
        return (signal - median) / iqr
    else:
        return signal.copy()

def _compute_metric(
    original: np.ndarray,
    reconstructed: np.ndarray,
    metric_func: Union[str, Callable]
) -> float:
    """Compute the specified metric between original and reconstructed signals."""
    if isinstance(metric_func, str):
        if metric_func == 'mse':
            return np.mean((original - reconstructed) ** 2)
        elif metric_func == 'mae':
            return np.mean(np.abs(original - reconstructed))
        elif metric_func == 'r2':
            ss_res = np.sum((original - reconstructed) ** 2)
            ss_tot = np.sum((original - np.mean(original)) ** 2)
            return 1 - (ss_res / ss_tot)
    else:
        return metric_func(original, reconstructed)

def _wavelet_transform(
    signal: np.ndarray,
    wavelet: np.ndarray
) -> Dict[str, np.ndarray]:
    """Perform the wavelet transform."""
    n = len(signal)
    m = len(wavelet)

    # Compute the convolution
    coefficients = np.zeros(n - m + 1)
    for i in range(n - m + 1):
        coefficients[i] = np.sum(signal[i:i+m] * wavelet)

    return {
        'coefficients': coefficients,
        'positions': np.arange(n - m + 1)
    }

def ondelettes_fit(
    signal: np.ndarray,
    wavelet: np.ndarray,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form'
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Perform wavelet transform on a time series signal.

    Parameters
    ----------
    signal : np.ndarray
        Input time series signal.
    wavelet : np.ndarray
        Wavelet function to use for the transform.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to evaluate the reconstruction ('mse', 'mae', 'r2' or custom function).
    solver : str, optional
        Solver method ('closed_form').

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Dictionary with transform results
        - 'metrics': Dictionary of computed metrics
        - 'params_used': Dictionary of parameters used
        - 'warnings': List of warnings

    Example
    -------
    >>> signal = np.random.randn(100)
    >>> wavelet = np.array([1, -1])
    >>> result = ondelettes_fit(signal, wavelet)
    """
    # Validate inputs
    _validate_inputs(signal, wavelet, normalization, metric)

    # Normalize signal
    normalized_signal = _normalize_signal(signal.copy(), normalization)

    # Perform wavelet transform
    transform_result = _wavelet_transform(normalized_signal, wavelet)

    # Reconstruct signal (inverse transform)
    reconstructed = np.zeros_like(signal)
    m = len(wavelet)
    for i, coeff in enumerate(transform_result['coefficients']):
        reconstructed[i:i+m] += coeff * wavelet

    # Compute metrics
    metrics = {
        'metric_value': _compute_metric(signal, reconstructed, metric),
        'normalization_used': normalization
    }

    # Prepare output
    return {
        'result': transform_result,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': str(metric) if callable(metric) else metric,
            'solver': solver
        },
        'warnings': []
    }

################################################################################
# scalogramme
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def scalogramme_fit(
    signal: np.ndarray,
    wavelet: Callable[[int], np.ndarray],
    scales: Union[np.ndarray, int] = 10,
    normalization: str = 'standard',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'gradient_descent',
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_params: Optional[Dict] = None
) -> Dict:
    """
    Compute the scalogram of a time series signal using wavelet transform.

    Parameters
    ----------
    signal : np.ndarray
        Input 1D time series signal.
    wavelet : Callable[[int], np.ndarray]
        Wavelet function that takes the scale as input and returns the wavelet coefficients.
    scales : Union[np.ndarray, int], optional
        Number of scales or array of specific scales to use. Default is 10.
    normalization : str, optional
        Type of normalization: 'none', 'standard', 'minmax', or 'robust'. Default is 'standard'.
    metric : Union[str, Callable], optional
        Metric to evaluate the quality of the transform: 'mse', 'mae', 'r2', or custom callable. Default is 'mse'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'. Default is 'gradient_descent'.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_params : Optional[Dict], optional
        Additional parameters for the solver. Default is None.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': The computed scalogram.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings generated during computation.

    Examples
    --------
    >>> signal = np.random.randn(100)
    >>> def morlet(scale):
    ...     return np.exp(-0.5 * (np.arange(-10, 11) / scale)**2) * np.exp(1j * 5.5 * np.arange(-10, 11) / scale)
    >>> scalogramme = scalogramme_fit(signal, morlet, scales=5)
    """
    # Validate inputs
    _validate_inputs(signal, wavelet, scales)

    # Initialize parameters
    params_used = {
        'scales': _initialize_scales(scales),
        'normalization': normalization,
        'metric': metric,
        'solver': solver,
        'tol': tol,
        'max_iter': max_iter
    }

    # Normalize the signal
    normalized_signal = _normalize(signal, normalization)

    # Compute wavelet coefficients
    coefficients = _compute_wavelet_coefficients(normalized_signal, wavelet, params_used['scales'])

    # Solve for optimal parameters
    if solver == 'closed_form':
        result = _solve_closed_form(coefficients, metric)
    elif solver == 'gradient_descent':
        result = _solve_gradient_descent(coefficients, metric, tol, max_iter, custom_params)
    elif solver == 'newton':
        result = _solve_newton(coefficients, metric, tol, max_iter, custom_params)
    elif solver == 'coordinate_descent':
        result = _solve_coordinate_descent(coefficients, metric, tol, max_iter, custom_params)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(result, coefficients, metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

def _validate_inputs(signal: np.ndarray, wavelet: Callable, scales: Union[np.ndarray, int]) -> None:
    """Validate the inputs for scalogram computation."""
    if not isinstance(signal, np.ndarray) or signal.ndim != 1:
        raise ValueError("Signal must be a 1D numpy array.")
    if not callable(wavelet):
        raise ValueError("Wavelet must be a callable function.")
    if isinstance(scales, int):
        if scales <= 0:
            raise ValueError("Number of scales must be positive.")
    elif isinstance(scales, np.ndarray):
        if len(scales) == 0:
            raise ValueError("Scales array must not be empty.")
    else:
        raise ValueError("Scales must be an integer or a numpy array.")

def _initialize_scales(scales: Union[np.ndarray, int]) -> np.ndarray:
    """Initialize the scales for wavelet transform."""
    if isinstance(scales, int):
        return np.linspace(1, scales, scales)
    return scales

def _normalize(signal: np.ndarray, normalization: str) -> np.ndarray:
    """Normalize the signal according to the specified method."""
    if normalization == 'none':
        return signal
    elif normalization == 'standard':
        mean = np.mean(signal)
        std = np.std(signal)
        return (signal - mean) / (std + 1e-8)
    elif normalization == 'minmax':
        min_val = np.min(signal)
        max_val = np.max(signal)
        return (signal - min_val) / (max_val - min_val + 1e-8)
    elif normalization == 'robust':
        median = np.median(signal)
        iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
        return (signal - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_wavelet_coefficients(signal: np.ndarray, wavelet: Callable, scales: np.ndarray) -> np.ndarray:
    """Compute the wavelet coefficients for each scale."""
    n = len(signal)
    coefficients = np.zeros((len(scales), n), dtype=complex)

    for i, scale in enumerate(scales):
        wavelet_coeffs = wavelet(scale)
        for j in range(n - len(wavelet_coeffs)):
            coefficients[i, j] = np.sum(signal[j:j+len(wavelet_coeffs)] * wavelet_coeffs)

    return coefficients

def _compute_metrics(result: np.ndarray, coefficients: np.ndarray, metric: Union[str, Callable]) -> Dict:
    """Compute the metrics for the scalogram."""
    if callable(metric):
        return {'custom_metric': metric(result, coefficients)}
    elif metric == 'mse':
        return {'mse': np.mean((result - coefficients) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(result - coefficients))}
    elif metric == 'r2':
        ss_res = np.sum((result - coefficients) ** 2)
        ss_tot = np.sum((coefficients - np.mean(coefficients)) ** 2)
        return {'r2': 1 - (ss_res / (ss_tot + 1e-8))}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _solve_closed_form(coefficients: np.ndarray, metric: Union[str, Callable]) -> np.ndarray:
    """Solve for optimal parameters using closed-form solution."""
    return coefficients.mean(axis=1, keepdims=True)

def _solve_gradient_descent(
    coefficients: np.ndarray,
    metric: Union[str, Callable],
    tol: float,
    max_iter: int,
    custom_params: Optional[Dict]
) -> np.ndarray:
    """Solve for optimal parameters using gradient descent."""
    result = coefficients.copy()
    learning_rate = custom_params.get('learning_rate', 0.01) if custom_params else 0.01

    for _ in range(max_iter):
        gradient = _compute_gradient(result, coefficients, metric)
        result -= learning_rate * gradient
        if np.linalg.norm(gradient) < tol:
            break

    return result

def _compute_gradient(result: np.ndarray, coefficients: np.ndarray, metric: Union[str, Callable]) -> np.ndarray:
    """Compute the gradient for gradient descent."""
    if callable(metric):
        return metric(result, coefficients)
    elif metric == 'mse':
        return 2 * (result - coefficients) / len(coefficients)
    elif metric == 'mae':
        return np.sign(result - coefficients) / len(coefficients)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _solve_newton(
    coefficients: np.ndarray,
    metric: Union[str, Callable],
    tol: float,
    max_iter: int,
    custom_params: Optional[Dict]
) -> np.ndarray:
    """Solve for optimal parameters using Newton's method."""
    result = coefficients.copy()

    for _ in range(max_iter):
        gradient = _compute_gradient(result, coefficients, metric)
        hessian = _compute_hessian(result, coefficients, metric)
        result -= np.linalg.solve(hessian, gradient)
        if np.linalg.norm(gradient) < tol:
            break

    return result

def _compute_hessian(result: np.ndarray, coefficients: np.ndarray, metric: Union[str, Callable]) -> np.ndarray:
    """Compute the Hessian for Newton's method."""
    if metric == 'mse':
        return 2 * np.eye(len(result))
    else:
        raise ValueError(f"Hessian computation not supported for metric: {metric}")

def _solve_coordinate_descent(
    coefficients: np.ndarray,
    metric: Union[str, Callable],
    tol: float,
    max_iter: int,
    custom_params: Optional[Dict]
) -> np.ndarray:
    """Solve for optimal parameters using coordinate descent."""
    result = coefficients.copy()

    for _ in range(max_iter):
        for i in range(len(result)):
            result[i] = _optimize_coordinate(result, coefficients, metric, i)
        if np.linalg.norm(_compute_gradient(result, coefficients, metric)) < tol:
            break

    return result

def _optimize_coordinate(result: np.ndarray, coefficients: np.ndarray, metric: Union[str, Callable], i: int) -> float:
    """Optimize a single coordinate in coordinate descent."""
    if metric == 'mse':
        residual = coefficients - result
        residual[i] = 0
        return np.sum(residual) / len(result)
    else:
        raise ValueError(f"Coordinate optimization not supported for metric: {metric}")

################################################################################
# autocorrelation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(series: np.ndarray) -> None:
    """Validate the input time series."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if series.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array.")
    if np.isnan(series).any() or np.isinf(series).any():
        raise ValueError("Input must not contain NaN or infinite values.")

def normalize_series(series: np.ndarray, method: str = 'none') -> np.ndarray:
    """Normalize the time series using the specified method."""
    if method == 'none':
        return series
    elif method == 'standard':
        mean = np.mean(series)
        std = np.std(series)
        return (series - mean) / std
    elif method == 'minmax':
        min_val = np.min(series)
        max_val = np.max(series)
        return (series - min_val) / (max_val - min_val)
    elif method == 'robust':
        median = np.median(series)
        iqr = np.percentile(series, 75) - np.percentile(series, 25)
        return (series - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_autocorrelation(series: np.ndarray, lags: int = 1) -> float:
    """Compute the autocorrelation for a given lag."""
    n = len(series)
    mean = np.mean(series)
    numerator = np.sum((series[:n-lags] - mean) * (series[lags:] - mean))
    denominator = np.sum((series[:n-lags] - mean) ** 2)
    return numerator / denominator if denominator != 0 else 0.0

def autocorrelation_fit(
    series: np.ndarray,
    lags: Optional[Union[int, np.ndarray]] = None,
    normalization: str = 'none',
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[Dict[int, float], Dict[str, float], Dict[str, str], list]]:
    """
    Compute the autocorrelation for a time series.

    Parameters
    ----------
    series : np.ndarray
        Input time series.
    lags : Optional[Union[int, np.ndarray]], default=None
        Lags to compute autocorrelation for. If None, computes for lags 1 to len(series)//2.
    normalization : str, default='none'
        Normalization method for the series. Options: 'none', 'standard', 'minmax', 'robust'.
    metric : Optional[Callable[[np.ndarray, np.ndarray], float]], default=None
        Custom metric function. If None, uses the default autocorrelation computation.

    Returns
    -------
    Dict[str, Union[Dict[int, float], Dict[str, float], Dict[str, str], list]]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate input
    validate_input(series)

    # Normalize series
    normalized_series = normalize_series(series, normalization)

    # Determine lags
    if lags is None:
        max_lag = len(normalized_series) // 2
        lags = np.arange(1, max_lag + 1)
    elif isinstance(lags, int):
        lags = np.array([lags])
    else:
        lags = np.asarray(lags)

    # Compute autocorrelation
    if metric is None:
        results = {lag: compute_autocorrelation(normalized_series, lag) for lag in lags}
    else:
        results = {lag: metric(normalized_series, np.roll(normalized_series, lag)) for lag in lags}

    # Prepare output
    output = {
        'result': results,
        'metrics': {},
        'params_used': {
            'normalization': normalization,
            'lags': lags.tolist(),
            'metric': metric.__name__ if metric else None
        },
        'warnings': []
    }

    return output

# Example usage:
# series = np.array([1, 2, 3, 4, 5])
# result = autocorrelation_fit(series, lags=1)

################################################################################
# cross_correlation
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for cross-correlation."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Inputs must have the same length.")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input x contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input y contains NaN or infinite values.")

def _normalize_data(x: np.ndarray, normalization: str) -> np.ndarray:
    """Normalize data according to specified method."""
    if normalization == "none":
        return x
    elif normalization == "standard":
        return (x - np.mean(x)) / np.std(x)
    elif normalization == "minmax":
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    elif normalization == "robust":
        return (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    lags: int = 10,
    metric: str = "pearson",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """Compute cross-correlation between two time series."""
    if custom_metric is not None:
        return _compute_custom_cross_correlation(x, y, lags, custom_metric)

    results = {}
    for lag in range(-lags, lags + 1):
        if lag > 0:
            x_lagged = np.roll(x, -lag)
        elif lag < 0:
            x_lagged = np.roll(x, abs(lag))
        else:
            x_lagged = x.copy()

        if metric == "pearson":
            correlation = np.corrcoef(x_lagged, y)[0, 1]
        elif metric == "spearman":
            correlation = np.corrcoef(np.argsort(x_lagged), np.argsort(y))[0, 1]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        results[lag] = correlation

    return results

def _compute_custom_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    lags: int = 10,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """Compute cross-correlation using a custom metric."""
    if custom_metric is None:
        raise ValueError("custom_metric doit être fourni pour le calcul personnalisé")
    results = {}
    for lag in range(-lags, lags + 1):
        if lag > 0:
            x_lagged = np.roll(x, -lag)
        elif lag < 0:
            x_lagged = np.roll(x, abs(lag))
        else:
            x_lagged = x.copy()

        results[lag] = custom_metric(x_lagged, y)

    return results

def cross_correlation_fit(
    x: np.ndarray,
    y: np.ndarray,
    lags: int = 10,
    normalization_x: str = "none",
    normalization_y: str = "none",
    metric: str = "pearson",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute cross-correlation between two time series.

    Parameters:
    -----------
    x : np.ndarray
        First time series.
    y : np.ndarray
        Second time series.
    lags : int, optional
        Number of lags to consider (default: 10).
    normalization_x : str, optional
        Normalization method for x ("none", "standard", "minmax", "robust").
    normalization_y : str, optional
        Normalization method for y ("none", "standard", "minmax", "robust").
    metric : str, optional
        Metric to use ("pearson", "spearman").
    custom_metric : Callable, optional
        Custom metric function.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, y)

    x_norm = _normalize_data(x, normalization_x)
    y_norm = _normalize_data(y, normalization_y)

    results = _compute_cross_correlation(
        x_norm,
        y_norm,
        lags,
        metric,
        custom_metric
    )

    return {
        "result": results,
        "metrics": {"metric_used": metric if custom_metric is None else "custom"},
        "params_used": {
            "lags": lags,
            "normalization_x": normalization_x,
            "normalization_y": normalization_y
        },
        "warnings": []
    }

# Example usage:
# result = cross_correlation_fit(
#     x=np.array([1, 2, 3, 4, 5]),
#     y=np.array([5, 4, 3, 2, 1]),
#     lags=2,
#     normalization_x="standard",
#     normalization_y="minmax"
# )
