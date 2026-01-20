"""
Quantix – Module moments_distribution
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# moment_1
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(data: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
    """Validate input data and optional weights."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Data contains infinite values.")

    if weights is not None:
        if not isinstance(weights, np.ndarray):
            raise TypeError("Weights must be a numpy array.")
        if weights.ndim != 1:
            raise ValueError("Weights must be a 1-dimensional array.")
        if len(weights) != len(data):
            raise ValueError("Weights and data must have the same length.")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")
        if np.any(np.isnan(weights)):
            raise ValueError("Weights contain NaN values.")
        if np.any(np.isinf(weights)):
            raise ValueError("Weights contain infinite values.")

def _normalize_data(data: np.ndarray, method: str = 'none') -> np.ndarray:
    """Normalize data based on the specified method."""
    if method == 'none':
        return data
    elif method == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    elif method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    elif method == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_moment(data: np.ndarray, order: int = 1, weights: Optional[np.ndarray] = None) -> float:
    """Compute the moment of specified order."""
    if weights is not None:
        return np.sum(weights * (data ** order)) / np.sum(weights)
    else:
        return np.mean(data ** order)

def _compute_metrics(actual: float, predicted: float, metric: str = 'mse', custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute metrics between actual and predicted moments."""
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(actual, predicted)

    if metric == 'mse':
        metrics['mse'] = (actual - predicted) ** 2
    elif metric == 'mae':
        metrics['mae'] = np.abs(actual - predicted)
    elif metric == 'r2':
        metrics['r2'] = 1 - ((actual - predicted) ** 2) / np.var(actual)
    elif metric == 'logloss':
        metrics['logloss'] = - (actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

def moment_1_fit(
    data: np.ndarray,
    order: int = 1,
    normalization: str = 'none',
    metric: str = 'mse',
    weights: Optional[np.ndarray] = None,
    custom_metric: Optional[Callable[[float, float], float]] = None
) -> Dict[str, Any]:
    """
    Compute the first moment of a distribution with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data.
    order : int, optional
        Order of the moment to compute (default is 1).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str, optional
        Metric to evaluate the moment ('mse', 'mae', 'r2', 'logloss').
    weights : Optional[np.ndarray], optional
        Weights for weighted moment computation.
    custom_metric : Optional[Callable[[float, float], float]], optional
        Custom metric function.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_inputs(data, weights)

    normalized_data = _normalize_data(data, normalization)
    moment_value = _compute_moment(normalized_data, order, weights)

    # For simplicity, we assume the actual moment is the computed one
    metrics = _compute_metrics(moment_value, moment_value, metric, custom_metric)

    return {
        "result": moment_value,
        "metrics": metrics,
        "params_used": {
            "order": order,
            "normalization": normalization,
            "metric": metric,
            "weights": weights is not None
        },
        "warnings": []
    }

# Example usage:
# data = np.array([1, 2, 3, 4, 5])
# result = moment_1_fit(data, order=1, normalization='standard', metric='mse')
# print(result)

################################################################################
# moment_2
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def moment_2_fit(
    data: np.ndarray,
    order: int = 2,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute the second moment of a distribution with various options.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    order : int, optional
        Order of the moment to compute (default is 2).
    normalization : str, optional
        Type of normalization ('none', 'standard', 'minmax', 'robust').
    metric : Union[str, Callable], optional
        Metric to use ('mse', 'mae', 'r2', custom callable).
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'newton').
    regularization : Optional[str], optional
        Type of regularization ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for convergence (default is 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default is 1000).
    custom_metric : Optional[Callable], optional
        Custom metric function.
    **kwargs :
        Additional keyword arguments for the solver.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.normal(0, 1, 1000)
    >>> result = moment_2_fit(data, order=2, normalization='standard')
    """
    # Validate inputs
    _validate_inputs(data, order)

    # Normalize data
    normalized_data = _normalize_data(data, normalization)

    # Compute moment
    result = _compute_moment(normalized_data, order, solver, regularization, tol, max_iter, **kwargs)

    # Compute metrics
    metrics = _compute_metrics(normalized_data, result, metric, custom_metric)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'order': order,
            'normalization': normalization,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return output

def _validate_inputs(data: np.ndarray, order: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if order < 0:
        raise ValueError("Order must be non-negative.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")

def _normalize_data(
    data: np.ndarray,
    normalization: str
) -> np.ndarray:
    """Normalize the input data."""
    if normalization == 'none':
        return data
    elif normalization == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    elif normalization == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

def _compute_moment(
    data: np.ndarray,
    order: int,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    **kwargs
) -> float:
    """Compute the moment of the data."""
    if solver == 'closed_form':
        return _compute_moment_closed_form(data, order)
    elif solver == 'gradient_descent':
        return _compute_moment_gradient_descent(data, order, tol, max_iter, **kwargs)
    elif solver == 'newton':
        return _compute_moment_newton(data, order, tol, max_iter, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _compute_moment_closed_form(
    data: np.ndarray,
    order: int
) -> float:
    """Compute the moment using closed-form solution."""
    return np.mean(data ** order)

def _compute_moment_gradient_descent(
    data: np.ndarray,
    order: int,
    tol: float,
    max_iter: int,
    **kwargs
) -> float:
    """Compute the moment using gradient descent."""
    # Placeholder for gradient descent implementation
    return np.mean(data ** order)

def _compute_moment_newton(
    data: np.ndarray,
    order: int,
    tol: float,
    max_iter: int,
    **kwargs
) -> float:
    """Compute the moment using Newton's method."""
    # Placeholder for Newton's method implementation
    return np.mean(data ** order)

def _compute_metrics(
    data: np.ndarray,
    result: float,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute the metrics for the moment."""
    metrics = {}

    if metric == 'mse':
        metrics['mse'] = np.mean((data ** 2 - result) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(data ** 2 - result))
    elif metric == 'r2':
        ss_total = np.sum((data ** 2 - np.mean(data ** 2)) ** 2)
        ss_res = np.sum((data ** 2 - result) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_total)
    elif callable(metric):
        metrics['custom'] = metric(data, result)
    elif custom_metric is not None:
        metrics['custom'] = custom_metric(data, result)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

################################################################################
# moment_3
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def moment_3_fit(
    data: np.ndarray,
    method: str = 'closed_form',
    normalization: Optional[str] = None,
    metric: Union[str, Callable] = 'mse',
    solver_options: Optional[Dict] = None,
    custom_moment_func: Optional[Callable] = None
) -> Dict:
    """
    Calculate the third moment of a distribution with configurable options.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    method : str, optional
        Method to compute the moment ('closed_form' or 'sample_estimator'), by default 'closed_form'.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), by default None.
    metric : Union[str, Callable], optional
        Metric to evaluate the moment ('mse', 'mae', 'r2'), or a custom callable, by default 'mse'.
    solver_options : Dict, optional
        Options for the solver (e.g., {'tol': 1e-6}), by default None.
    custom_moment_func : Callable, optional
        Custom function to compute the moment, by default None.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.normal(0, 1, 1000)
    >>> result = moment_3_fit(data, method='closed_form', normalization='standard')
    """
    # Validate inputs
    _validate_inputs(data, method, normalization, metric)

    # Normalize data if required
    normalized_data = _apply_normalization(data, normalization)

    # Compute the third moment
    if custom_moment_func is not None:
        result = _compute_custom_moment(normalized_data, custom_moment_func)
    else:
        result = _compute_third_moment(normalized_data, method)

    # Compute metrics
    metrics = _compute_metrics(data, result, metric)

    # Prepare output dictionary
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'method': method,
            'normalization': normalization,
            'metric': metric
        },
        'warnings': []
    }

    return output

def _validate_inputs(
    data: np.ndarray,
    method: str,
    normalization: Optional[str],
    metric: Union[str, Callable]
) -> None:
    """Validate input parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if method not in ['closed_form', 'sample_estimator']:
        raise ValueError("Method must be either 'closed_form' or 'sample_estimator'.")
    if normalization not in [None, 'none', 'standard', 'minmax', 'robust']:
        raise ValueError("Normalization must be one of: none, standard, minmax, robust.")
    if isinstance(metric, str) and metric not in ['mse', 'mae', 'r2']:
        raise ValueError("Metric must be either 'mse', 'mae', 'r2' or a callable.")

def _apply_normalization(
    data: np.ndarray,
    normalization: Optional[str]
) -> np.ndarray:
    """Apply normalization to the data."""
    if normalization is None or normalization == 'none':
        return data
    elif normalization == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    elif normalization == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        raise ValueError("Unknown normalization method.")

def _compute_third_moment(
    data: np.ndarray,
    method: str
) -> float:
    """Compute the third moment of the data."""
    if method == 'closed_form':
        mean = np.mean(data)
        centered_data = data - mean
        return np.mean(centered_data ** 3)
    elif method == 'sample_estimator':
        return np.mean(data ** 3)
    else:
        raise ValueError("Unknown method.")

def _compute_custom_moment(
    data: np.ndarray,
    custom_func: Callable
) -> float:
    """Compute the third moment using a custom function."""
    return custom_func(data)

def _compute_metrics(
    data: np.ndarray,
    result: float,
    metric: Union[str, Callable]
) -> Dict:
    """Compute metrics for the moment estimation."""
    if isinstance(metric, str):
        if metric == 'mse':
            return {'mse': np.mean((data - result) ** 2)}
        elif metric == 'mae':
            return {'mae': np.mean(np.abs(data - result))}
        elif metric == 'r2':
            ss_total = np.sum((data - np.mean(data)) ** 2)
            ss_residual = np.sum((data - result) ** 2)
            return {'r2': 1 - (ss_residual / ss_total)}
    else:
        return {metric.__name__: metric(data, result)}

def _compute_closed_form_moment(
    data: np.ndarray
) -> float:
    """Compute the third moment using closed-form solution."""
    mean = np.mean(data)
    centered_data = data - mean
    return np.mean(centered_data ** 3)

def _compute_sample_estimator_moment(
    data: np.ndarray
) -> float:
    """Compute the third moment using sample estimator."""
    return np.mean(data ** 3)

################################################################################
# moment_4
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def moment_4_fit(
    data: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Estimate the fourth moment of a distribution using various methods.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the data. Default is identity.
    metric : str
        Metric to evaluate the fit quality. Options: 'mse', 'mae', 'r2'.
    solver : str
        Solver to use. Options: 'closed_form', 'gradient_descent'.
    regularization : Optional[str]
        Regularization type. Options: None, 'l1', 'l2'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function.
    **kwargs
        Additional solver-specific parameters.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.normal(0, 1, 1000)
    >>> result = moment_4_fit(data)
    """
    # Validate inputs
    _validate_inputs(data, normalizer)

    # Normalize data
    normalized_data = normalizer(data)

    # Choose solver
    if solver == 'closed_form':
        params, metrics = _solve_closed_form(normalized_data, metric, custom_metric)
    elif solver == 'gradient_descent':
        params, metrics = _solve_gradient_descent(
            normalized_data,
            metric,
            regularization,
            tol,
            max_iter,
            custom_metric,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Prepare output
    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': _check_warnings(params, metrics)
    }

def _validate_inputs(data: np.ndarray, normalizer: Callable[[np.ndarray], np.ndarray]) -> None:
    """Validate input data and normalizer."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1D array.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")
    if not callable(normalizer):
        raise TypeError("Normalizer must be a callable.")

def _solve_closed_form(
    data: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> tuple[np.ndarray, Dict[str, float]]:
    """Solve for the fourth moment using closed-form solution."""
    mean = np.mean(data)
    variance = np.var(data, ddof=1)
    skewness = np.sum((data - mean) ** 3) / (len(data) * variance ** 1.5)
    kurtosis = np.sum((data - mean) ** 4) / (len(data) * variance ** 2)

    params = np.array([mean, variance, skewness, kurtosis])

    if custom_metric is not None:
        metrics = {'custom': custom_metric(data, params)}
    else:
        metrics = _compute_metrics(data, params, metric)

    return params, metrics

def _solve_gradient_descent(
    data: np.ndarray,
    metric: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> tuple[np.ndarray, Dict[str, float]]:
    """Solve for the fourth moment using gradient descent."""
    # Initial guess
    initial_params = np.array([np.mean(data), np.var(data, ddof=1), 0.0, 3.0])

    # Gradient descent parameters
    learning_rate = kwargs.get('learning_rate', 0.01)
    params = initial_params.copy()

    for _ in range(max_iter):
        grad = _compute_gradient(data, params, metric)
        if regularization == 'l1':
            grad += np.sign(params) * kwargs.get('alpha', 0.1)
        elif regularization == 'l2':
            grad += 2 * kwargs.get('alpha', 0.1) * params

        params -= learning_rate * grad

        if np.linalg.norm(grad) < tol:
            break

    if custom_metric is not None:
        metrics = {'custom': custom_metric(data, params)}
    else:
        metrics = _compute_metrics(data, params, metric)

    return params, metrics

def _compute_gradient(
    data: np.ndarray,
    params: np.ndarray,
    metric: str
) -> np.ndarray:
    """Compute the gradient of the loss function."""
    mean, variance, skewness, kurtosis = params
    n = len(data)

    if metric == 'mse':
        # Compute gradient for MSE
        grad_mean = -2 * np.sum(data - mean) / n
        grad_variance = -2 * np.sum((data - mean) ** 2 - variance) / (n * 2 * variance)
        grad_skewness = -2 * np.sum((data - mean) ** 3 - skewness * variance ** 1.5) / (n * variance ** 1.5)
        grad_kurtosis = -2 * np.sum((data - mean) ** 4 - kurtosis * variance ** 2) / (n * variance ** 2)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return np.array([grad_mean, grad_variance, grad_skewness, grad_kurtosis])

def _compute_metrics(
    data: np.ndarray,
    params: np.ndarray,
    metric: str
) -> Dict[str, float]:
    """Compute metrics for the fit."""
    mean, variance, skewness, kurtosis = params
    n = len(data)

    if metric == 'mse':
        mse = np.mean((data - mean) ** 2)
    elif metric == 'mae':
        mae = np.mean(np.abs(data - mean))
    elif metric == 'r2':
        ss_total = np.sum((data - np.mean(data)) ** 2)
        ss_residual = np.sum((data - mean) ** 2)
        r2 = 1 - (ss_residual / ss_total)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return {'mse': mse, 'mae': mae, 'r2': r2} if metric != 'custom' else {}

def _check_warnings(
    params: np.ndarray,
    metrics: Dict[str, float]
) -> list[str]:
    """Check for warnings in the results."""
    warnings = []
    if np.any(np.isnan(params)) or np.any(np.isinf(params)):
        warnings.append("Parameters contain NaN or infinite values.")
    if metrics.get('mse', float('inf')) > 1e6:
        warnings.append("High MSE value, check data and parameters.")
    return warnings

################################################################################
# variance
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def variance_fit(
    data: np.ndarray,
    method: str = 'sample',
    ddof: int = 0,
    normalization: Optional[str] = None,
    metric: Callable[[np.ndarray, np.ndarray], float] = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Compute the variance of a distribution with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    method : str, optional
        Method to compute variance ('sample' or 'population'), default is 'sample'.
    ddof : int, optional
        Delta degrees of freedom for sample variance, default is 0.
    normalization : str or None, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), default is None.
    metric : callable, optional
        Metric function to evaluate the variance computation, default is MSE.
    solver : str, optional
        Solver method ('closed_form'), default is 'closed_form'.
    tol : float, optional
        Tolerance for convergence, default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations for iterative solvers, default is 1000.
    custom_normalization : callable or None, optional
        Custom normalization function, default is None.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data)

    # Normalize data if required
    normalized_data = _apply_normalization(
        data,
        normalization=normalization,
        custom_normalization=custom_normalization
    )

    # Compute variance based on method
    result = _compute_variance(
        normalized_data,
        method=method,
        ddof=ddof
    )

    # Compute metrics
    metrics = _compute_metrics(
        data,
        result['variance'],
        metric=metric
    )

    # Prepare output dictionary
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'method': method,
            'ddof': ddof,
            'normalization': normalization,
            'solver': solver
        },
        'warnings': []
    }

    return output

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values.")

def _apply_normalization(
    data: np.ndarray,
    normalization: Optional[str] = None,
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Apply normalization to the data."""
    if custom_normalization is not None:
        return custom_normalization(data)

    if normalization == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    elif normalization == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        return data.copy()

def _compute_variance(
    data: np.ndarray,
    method: str = 'sample',
    ddof: int = 0
) -> Dict[str, Any]:
    """Compute variance based on the specified method."""
    if method == 'sample':
        variance = np.var(data, ddof=ddof)
    elif method == 'population':
        variance = np.var(data, ddof=0)
    else:
        raise ValueError("Invalid method. Choose 'sample' or 'population'.")

    return {
        'variance': variance,
        'method': method,
        'ddof': ddof
    }

def _compute_metrics(
    data: np.ndarray,
    variance: float,
    metric: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, Any]:
    """Compute metrics for the variance computation."""
    mean = np.mean(data)
    squared_errors = (data - mean) ** 2
    return {
        'metric_value': metric(data, squared_errors),
        'mean_squared_error': np.mean(squared_errors)
    }

# Example usage:
# data = np.array([1, 2, 3, 4, 5])
# result = variance_fit(data, method='sample', normalization='standard')

################################################################################
# skewness
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for skewness calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values.")

def _calculate_moments(data: np.ndarray, normalized: bool = False) -> Dict[str, float]:
    """Calculate the first three moments of the data."""
    mean = np.mean(data)
    std = np.std(data) if normalized else 1.0
    n = len(data)

    moment_2 = np.sum((data - mean) ** 2) / n
    moment_3 = np.sum((data - mean) ** 3) / n

    return {
        'mean': mean,
        'moment_2': moment_2,
        'moment_3': moment_3,
        'std': std
    }

def _skewness_formula(moments: Dict[str, float], normalization: str = 'standard') -> float:
    """Calculate skewness using the formula based on moments."""
    if normalization == 'standard':
        return (moments['moment_3'] / (moments['std'] ** 3))
    elif normalization == 'none':
        return moments['moment_3']
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def skewness_fit(
    data: np.ndarray,
    normalization: str = 'standard',
    custom_metric: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Calculate the skewness of a dataset.

    Parameters:
    -----------
    data : np.ndarray
        Input data for which skewness is to be calculated.
    normalization : str, optional
        Normalization method ('standard' or 'none'). Default is 'standard'.
    custom_metric : Callable, optional
        Custom metric function to evaluate skewness. If provided, overrides default calculation.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_input(data)

    moments = _calculate_moments(data, normalized=(normalization == 'standard'))
    skewness_value = _skewness_formula(moments, normalization)

    metrics = {}
    if custom_metric is not None:
        try:
            metrics['custom_metric'] = custom_metric(data)
        except Exception as e:
            warnings.append(f"Custom metric calculation failed: {str(e)}")

    result = {
        'result': skewness_value,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'custom_metric': custom_metric.__name__ if custom_metric else None
        },
        'warnings': []
    }

    return result

# Example usage:
# data = np.array([1, 2, 3, 4, 5])
# result = skewness_fit(data)

################################################################################
# kurtosis
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for kurtosis calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def _compute_moments(data: np.ndarray, order: int) -> float:
    """Compute the central moment of given order."""
    mean = np.mean(data)
    return np.mean((data - mean) ** order)

def _kurtosis_standard(data: np.ndarray, fisher: bool = True) -> float:
    """Compute standard kurtosis (excess or not)."""
    m4 = _compute_moments(data, 4)
    m2 = _compute_moments(data, 2)
    kurtosis = (m4 / (m2 ** 2)) - 3
    return kurtosis if fisher else kurtosis + 3

def _kurtosis_robust(data: np.ndarray) -> float:
    """Compute robust kurtosis using median and MAD."""
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    if mad == 0:
        return 0.0
    z_scores = (data - med) / mad
    kurtosis = np.mean(z_scores ** 4) - 3 * (np.mean(z_scores ** 2)) ** 2
    return kurtosis

def _kurtosis_custom(data: np.ndarray, normalization_func: Callable) -> float:
    """Compute kurtosis using custom normalization."""
    normalized_data = normalization_func(data)
    return _kurtosis_standard(normalized_data)

def kurtosis_fit(
    data: np.ndarray,
    normalization: str = "standard",
    fisher: bool = True,
    custom_normalization: Optional[Callable] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Compute kurtosis of a distribution with various normalization options.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    normalization : str, optional
        Normalization method ("none", "standard", "robust").
    fisher : bool, optional
        Whether to return Fisher's definition (excess kurtosis).
    custom_normalization : callable, optional
        Custom normalization function.

    Returns
    -------
    dict
        Dictionary containing:
        - "result": computed kurtosis value
        - "metrics": dictionary of additional metrics
        - "params_used": parameters used in computation
        - "warnings": any warnings generated

    Examples
    --------
    >>> data = np.random.normal(0, 1, 1000)
    >>> kurtosis_fit(data)
    """
    _validate_input(data)

    params_used = {
        "normalization": normalization,
        "fisher": fisher
    }

    warnings = []

    if custom_normalization is not None:
        kurtosis_value = _kurtosis_custom(data, custom_normalization)
    elif normalization == "none":
        kurtosis_value = _kurtosis_standard(data, fisher)
    elif normalization == "robust":
        kurtosis_value = _kurtosis_robust(data)
    elif normalization == "standard":
        kurtosis_value = _kurtosis_standard(data, fisher)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    metrics = {
        "mean": np.mean(data),
        "std": np.std(data),
        "skewness": _compute_moments(data, 3) / (_compute_moments(data, 2) ** (3/2))
    }

    return {
        "result": kurtosis_value,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# central_moment
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def central_moment_fit(
    data: np.ndarray,
    order: int = 2,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Compute the central moment of a distribution with configurable options.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    order : int, optional
        Order of the central moment to compute (default is 2).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : Union[str, Callable], optional
        Metric to use ('mse', 'mae', 'r2', or custom callable).
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton').
    regularization : Optional[str], optional
        Regularization method (None, 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for convergence (default is 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default is 1000).
    custom_metric : Optional[Callable], optional
        Custom metric function.
    custom_distance : Optional[Callable], optional
        Custom distance function.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.normal(0, 1, 1000)
    >>> result = central_moment_fit(data, order=2)
    """
    # Validate inputs
    _validate_inputs(data, order)

    # Normalize data if required
    normalized_data = _normalize_data(data, normalization)

    # Compute central moment based on solver choice
    if solver == 'closed_form':
        result = _compute_central_moment_closed_form(normalized_data, order)
    elif solver == 'gradient_descent':
        result = _compute_central_moment_gradient_descent(
            normalized_data, order, tol, max_iter
        )
    elif solver == 'newton':
        result = _compute_central_moment_newton(
            normalized_data, order, tol, max_iter
        )
    else:
        raise ValueError("Unsupported solver method.")

    # Compute metrics
    metrics = _compute_metrics(normalized_data, result, metric, custom_metric)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'order': order,
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(normalized_data, result)
    }

    return output

def _validate_inputs(data: np.ndarray, order: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if order < 0:
        raise ValueError("Order of central moment must be non-negative.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values.")

def _normalize_data(
    data: np.ndarray,
    method: str = 'none'
) -> np.ndarray:
    """Normalize data based on the specified method."""
    if method == 'none':
        return data
    elif method == 'standard':
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
        raise ValueError("Unsupported normalization method.")

def _compute_central_moment_closed_form(
    data: np.ndarray,
    order: int
) -> float:
    """Compute central moment using closed-form solution."""
    mean = np.mean(data)
    return np.mean((data - mean) ** order)

def _compute_central_moment_gradient_descent(
    data: np.ndarray,
    order: int,
    tol: float,
    max_iter: int
) -> float:
    """Compute central moment using gradient descent."""
    mean = np.mean(data)
    current = 0.0
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = np.mean(order * (data - mean) ** (order - 1))
        current -= learning_rate * gradient
        if abs(gradient) < tol:
            break

    return current

def _compute_central_moment_newton(
    data: np.ndarray,
    order: int,
    tol: float,
    max_iter: int
) -> float:
    """Compute central moment using Newton's method."""
    mean = np.mean(data)
    current = 0.0

    for _ in range(max_iter):
        gradient = np.mean(order * (data - mean) ** (order - 1))
        hessian = np.mean(order * (order - 1) * (data - mean) ** (order - 2))
        current -= gradient / hessian
        if abs(gradient) < tol:
            break

    return current

def _compute_metrics(
    data: np.ndarray,
    result: float,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute metrics based on the specified method."""
    mean = np.mean(data)
    residuals = (data - mean) ** 2

    if metric == 'mse':
        return {'mse': np.mean(residuals)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(residuals))}
    elif metric == 'r2':
        ss_total = np.sum((data - mean) ** 2)
        return {'r2': 1 - (np.sum(residuals) / ss_total)}
    elif metric == 'logloss':
        return {'logloss': -np.mean(data * np.log(result + 1e-8))}
    elif custom_metric is not None:
        return {'custom': custom_metric(data, result)}
    else:
        raise ValueError("Unsupported metric method.")

def _check_warnings(
    data: np.ndarray,
    result: float
) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isnan(result)):
        warnings.append("Result contains NaN values.")
    if np.std(data) < 1e-6:
        warnings.append("Data has very low variance.")
    return warnings

################################################################################
# raw_moment
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def raw_moment_fit(
    data: np.ndarray,
    order: int = 1,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Compute raw moments of a distribution with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    order : int, optional
        Order of the moment to compute (default: 1).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    metric : Union[str, Callable], optional
        Metric to evaluate ('mse', 'mae', 'r2', custom callable) (default: 'mse').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton') (default: 'closed_form').
    regularization : Optional[str], optional
        Regularization method (None, 'l1', 'l2', 'elasticnet') (default: None).
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    custom_metric : Optional[Callable], optional
        Custom metric function (default: None).
    custom_distance : Optional[Callable], optional
        Custom distance function (default: None).

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, order)

    # Normalize data if required
    normalized_data = _apply_normalization(data, normalization)

    # Compute raw moment based on solver
    if solver == 'closed_form':
        result = _compute_raw_moment_closed_form(normalized_data, order)
    elif solver == 'gradient_descent':
        result = _compute_raw_moment_gradient_descent(
            normalized_data, order, tol, max_iter, regularization
        )
    elif solver == 'newton':
        result = _compute_raw_moment_newton(
            normalized_data, order, tol, max_iter, regularization
        )
    else:
        raise ValueError("Unsupported solver method")

    # Compute metrics
    metrics = _compute_metrics(data, result, metric, custom_metric)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'order': order,
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(data, result)
    }

    return output

def _validate_inputs(data: np.ndarray, order: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if order < 1:
        raise ValueError("Order must be a positive integer")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to the data."""
    if method == 'none':
        return data
    elif method == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    elif method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    elif method == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        raise ValueError("Unsupported normalization method")

def _compute_raw_moment_closed_form(data: np.ndarray, order: int) -> float:
    """Compute raw moment using closed-form solution."""
    return np.mean(data ** order)

def _compute_raw_moment_gradient_descent(
    data: np.ndarray,
    order: int,
    tol: float,
    max_iter: int,
    regularization: Optional[str]
) -> float:
    """Compute raw moment using gradient descent."""
    # Simplified implementation for demonstration
    current = np.mean(data ** order)
    for _ in range(max_iter):
        gradient = 2 * (np.mean(data ** (order - 1)) - current)
        if abs(gradient) < tol:
            break
        current -= gradient
    return current

def _compute_raw_moment_newton(
    data: np.ndarray,
    order: int,
    tol: float,
    max_iter: int,
    regularization: Optional[str]
) -> float:
    """Compute raw moment using Newton's method."""
    # Simplified implementation for demonstration
    current = np.mean(data ** order)
    for _ in range(max_iter):
        gradient = 2 * (np.mean(data ** (order - 1)) - current)
        hessian = 2 * np.mean(data ** (order - 2))
        if abs(gradient) < tol:
            break
        current -= gradient / hessian
    return current

def _compute_metrics(
    data: np.ndarray,
    result: float,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute metrics based on the specified method."""
    if isinstance(metric, str):
        if metric == 'mse':
            return {'mse': np.mean((data - result) ** 2)}
        elif metric == 'mae':
            return {'mae': np.mean(np.abs(data - result))}
        elif metric == 'r2':
            ss_res = np.sum((data - result) ** 2)
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            return {'r2': 1 - (ss_res / ss_tot)}
        else:
            raise ValueError("Unsupported metric")
    elif callable(metric):
        return {'custom_metric': metric(data, result)}
    else:
        raise TypeError("Metric must be a string or callable")

def _check_warnings(data: np.ndarray, result: float) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isnan(result)):
        warnings.append("Result contains NaN values")
    if np.std(data) == 0:
        warnings.append("Data has zero standard deviation")
    return warnings

# Example usage
if __name__ == "__main__":
    data = np.random.normal(0, 1, 1000)
    result = raw_moment_fit(data, order=2, normalization='standard', metric='mse')
    print(result)
