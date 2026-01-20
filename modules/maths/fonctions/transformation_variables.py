"""
Quantix – Module transformation_variables
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# log_transformation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    normalization: str = "none",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """
    Validate and normalize input data.

    Parameters
    ----------
    X : np.ndarray
        Input data to validate and normalize.
    normalization : str, optional
        Type of normalization to apply. Options: "none", "standard", "minmax", "robust".
    custom_normalization : callable, optional
        Custom normalization function.

    Returns
    -------
    np.ndarray
        Normalized input data.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if X.ndim != 1:
        raise ValueError("Input X must be a 1-dimensional array.")
    if np.any(np.isnan(X)):
        raise ValueError("Input X contains NaN values.")
    if np.any(np.isinf(X)):
        raise ValueError("Input X contains infinite values.")

    if normalization == "none":
        return X
    elif normalization == "standard":
        mean = np.mean(X)
        std = np.std(X)
        return (X - mean) / std
    elif normalization == "minmax":
        min_val = np.min(X)
        max_val = np.max(X)
        return (X - min_val) / (max_val - min_val)
    elif normalization == "robust":
        median = np.median(X)
        iqr = np.percentile(X, 75) - np.percentile(X, 25)
        return (X - median) / iqr
    elif custom_normalization is not None:
        return custom_normalization(X)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _log_transformation_compute(
    X: np.ndarray,
    base: float = np.e,
) -> np.ndarray:
    """
    Compute the log transformation of input data.

    Parameters
    ----------
    X : np.ndarray
        Input data to transform.
    base : float, optional
        Base of the logarithm. Default is natural logarithm (e).

    Returns
    -------
    np.ndarray
        Log-transformed data.
    """
    if base <= 0:
        raise ValueError("Base of the logarithm must be positive.")
    if np.any(X <= 0):
        raise ValueError("Input data must be strictly positive for log transformation.")
    return np.log(X) / np.log(base)

def _compute_metrics(
    X: np.ndarray,
    X_transformed: np.ndarray,
    metric: str = "mse",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, float]:
    """
    Compute metrics between original and transformed data.

    Parameters
    ----------
    X : np.ndarray
        Original input data.
    X_transformed : np.ndarray
        Transformed data.
    metric : str, optional
        Metric to compute. Options: "mse", "mae", "r2".
    custom_metric : callable, optional
        Custom metric function.

    Returns
    -------
    Dict[str, float]
        Dictionary of computed metrics.
    """
    if not isinstance(X, np.ndarray) or not isinstance(X_transformed, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if X.shape != X_transformed.shape:
        raise ValueError("Input arrays must have the same shape.")

    metrics = {}
    if metric == "mse":
        metrics["mse"] = np.mean((X - X_transformed) ** 2)
    elif metric == "mae":
        metrics["mae"] = np.mean(np.abs(X - X_transformed))
    elif metric == "r2":
        ss_res = np.sum((X - X_transformed) ** 2)
        ss_tot = np.sum((X - np.mean(X)) ** 2)
        metrics["r2"] = 1 - (ss_res / ss_tot)
    elif custom_metric is not None:
        metrics["custom"] = custom_metric(X, X_transformed)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

def log_transformation_fit(
    X: np.ndarray,
    base: float = np.e,
    normalization: str = "none",
    metric: str = "mse",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Fit and compute log transformation of input data with optional normalization.

    Parameters
    ----------
    X : np.ndarray
        Input data to transform.
    base : float, optional
        Base of the logarithm. Default is natural logarithm (e).
    normalization : str, optional
        Type of normalization to apply. Options: "none", "standard", "minmax", "robust".
    metric : str, optional
        Metric to compute. Options: "mse", "mae", "r2".
    custom_normalization : callable, optional
        Custom normalization function.
    custom_metric : callable, optional
        Custom metric function.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]
        Dictionary containing:
        - "result": Transformed data.
        - "metrics": Computed metrics.
        - "params_used": Parameters used in the transformation.
        - "warnings": Any warnings generated during processing.
    """
    result = {}
    warnings = []

    # Validate and normalize input data
    X_normalized = _validate_inputs(X, normalization, custom_normalization)

    # Compute log transformation
    try:
        X_transformed = _log_transformation_compute(X_normalized, base)
    except ValueError as e:
        warnings.append(str(e))
        X_transformed = np.full_like(X_normalized, np.nan)

    # Compute metrics
    try:
        metrics = _compute_metrics(X_normalized, X_transformed, metric, custom_metric)
    except ValueError as e:
        warnings.append(str(e))
        metrics = {}

    # Prepare output
    result["result"] = X_transformed
    result["metrics"] = metrics
    result["params_used"] = {
        "base": base,
        "normalization": normalization,
        "metric": metric,
    }
    result["warnings"] = warnings

    return result

################################################################################
# sqrt_transformation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def sqrt_transformation_fit(
    X: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Applies square root transformation to variables with configurable options.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize data before transformation
    metric : str or callable
        Metric to evaluate transformation quality ('mse', 'mae', 'r2')
    custom_metric : Optional[Callable]
        Custom metric function if needed
    solver : str
        Solver to use ('closed_form', 'gradient_descent')
    regularization : Optional[str]
        Regularization type ('l1', 'l2', 'elasticnet')
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    random_state : Optional[int]
        Random seed for reproducibility

    Returns
    -------
    Dict with keys:
        - 'result': transformed data
        - 'metrics': computed metrics
        - 'params_used': parameters used
        - 'warnings': any warnings

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = sqrt_transformation_fit(X)
    """
    # Validate inputs
    _validate_inputs(X, normalizer)

    # Normalize data if needed
    X_norm = normalizer(X) if callable(normalizer) else X

    # Choose solver
    if solver == 'closed_form':
        transformed, params = _closed_form_solver(X_norm)
    elif solver == 'gradient_descent':
        transformed, params = _gradient_descent_solver(
            X_norm,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(
        X,
        transformed,
        metric=metric,
        custom_metric=custom_metric
    )

    return {
        'result': transformed,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, normalizer: Callable) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")
    if not callable(normalizer):
        raise TypeError("normalizer must be a callable")

def _closed_form_solver(X: np.ndarray) -> tuple[np.ndarray, Dict]:
    """Closed form solution for square root transformation."""
    # Simple implementation - in practice would be more sophisticated
    transformed = np.sqrt(X)
    return transformed, {'method': 'closed_form'}

def _gradient_descent_solver(
    X: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> tuple[np.ndarray, Dict]:
    """Gradient descent solver for square root transformation."""
    np.random.seed(random_state)
    # Initialize parameters
    params = {
        'method': 'gradient_descent',
        'tol': tol,
        'max_iter': max_iter
    }
    # Placeholder implementation
    transformed = np.sqrt(X)
    return transformed, params

def _calculate_metrics(
    X: np.ndarray,
    y_pred: np.ndarray,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Calculate transformation metrics."""
    metrics = {}

    if custom_metric is not None:
        metrics['custom'] = custom_metric(X, y_pred)
    else:
        if metric == 'mse':
            metrics['mse'] = np.mean((X - y_pred) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(X - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((X - y_pred) ** 2)
            ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

################################################################################
# boxcox_transformation
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for Box-Cox transformation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values.")
    if np.any(data <= 0):
        raise ValueError("Input data must be strictly positive for Box-Cox transformation.")

def _boxcox_transformation(data: np.ndarray, lambda_: float) -> np.ndarray:
    """Apply Box-Cox transformation to the data."""
    if lambda_ == 0:
        return np.log(data)
    else:
        return (data ** lambda_ - 1) / lambda_

def _compute_optimal_lambda(data: np.ndarray, metric: Callable = None) -> float:
    """Compute the optimal lambda for Box-Cox transformation."""
    if metric is None:
        metric = _default_metric

    lambdas = np.linspace(-2, 3, 100)
    metrics = []
    for l in lambdas:
        try:
            transformed = _boxcox_transformation(data, l)
            metrics.append(metric(transformed))
        except:
            metrics.append(np.inf)

    optimal_idx = np.argmin(metrics)
    return lambdas[optimal_idx]

def _default_metric(data: np.ndarray) -> float:
    """Default metric for Box-Cox transformation (negative log-likelihood)."""
    n = len(data)
    mean = np.mean(data)
    variance = np.var(data, ddof=1)
    return -n * (np.log(variance) + 1 + np.log(n))

def boxcox_transformation_fit(
    data: np.ndarray,
    metric: Optional[Callable] = None,
    lambda_: Optional[float] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit Box-Cox transformation to the data.

    Parameters
    ----------
    data : np.ndarray
        Input data to be transformed.
    metric : Callable, optional
        Metric function to optimize. If None, uses default negative log-likelihood.
    lambda_ : float, optional
        Fixed lambda value. If None, computes optimal lambda.

    Returns
    -------
    dict
        Dictionary containing the transformed data, optimal lambda, metrics, and warnings.
    """
    _validate_inputs(data)

    if lambda_ is None:
        lambda_ = _compute_optimal_lambda(data, metric)

    transformed_data = _boxcox_transformation(data, lambda_)

    result = {
        "result": transformed_data,
        "metrics": {"optimal_lambda": lambda_},
        "params_used": {
            "metric": metric.__name__ if metric else "_default_metric",
            "lambda_": lambda_
        },
        "warnings": []
    }

    return result

def boxcox_transformation_compute(
    data: np.ndarray,
    lambda_: float
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Compute Box-Cox transformation with a given lambda.

    Parameters
    ----------
    data : np.ndarray
        Input data to be transformed.
    lambda_ : float
        Lambda value for the transformation.

    Returns
    -------
    dict
        Dictionary containing the transformed data and warnings.
    """
    _validate_inputs(data)

    transformed_data = _boxcox_transformation(data, lambda_)

    result = {
        "result": transformed_data,
        "metrics": {},
        "params_used": {"lambda_": lambda_},
        "warnings": []
    }

    return result

################################################################################
# yeo_johnson_transformation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def yeo_johnson_transformation_fit(
    X: np.ndarray,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Fit the Yeo-Johnson transformation to data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    normalization : str, optional
        Type of normalization to apply. Options: 'none', 'standard', 'minmax', 'robust'.
    metric : str or callable, optional
        Metric to optimize. Options: 'mse', 'mae', 'r2', or custom callable.
    solver : str, optional
        Solver to use. Options: 'closed_form', 'gradient_descent'.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    **kwargs :
        Additional solver-specific parameters.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Transformed data.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> result = yeo_johnson_transformation_fit(X)
    """
    # Validate inputs
    X, warnings = _validate_inputs(X)

    # Normalize data
    normalized_X, norm_params = _normalize_data(X, normalization)

    # Choose metric function
    metric_func = _get_metric_function(metric, custom_metric)

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(normalized_X)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(
            normalized_X, metric_func, tol, max_iter, **kwargs
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Transform data
    transformed_X = _transform_data(normalized_X, params)

    # Compute metrics
    metrics = _compute_metrics(X, transformed_X, metric_func)

    return {
        'result': transformed_X,
        'metrics': metrics,
        'params_used': params,
        'warnings': warnings
    }

def _validate_inputs(X: np.ndarray) -> tuple:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be 2-dimensional")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        warnings = ["Input contains NaN or inf values"]
    else:
        warnings = []
    return X, warnings

def _normalize_data(X: np.ndarray, normalization: str) -> tuple:
    """Normalize data based on specified method."""
    if normalization == 'none':
        return X, {}
    elif normalization == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        normalized_X = (X - mean) / (std + 1e-8)
    elif normalization == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        normalized_X = (X - min_val) / (max_val - min_val + 1e-8)
    elif normalization == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        normalized_X = (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    return normalized_X, locals()

def _get_metric_function(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Get metric function based on input."""
    if custom_metric is not None:
        return custom_metric
    elif metric == 'mse':
        return _mean_squared_error
    elif metric == 'mae':
        return _mean_absolute_error
    elif metric == 'r2':
        return _r_squared
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _solve_closed_form(X: np.ndarray) -> Dict:
    """Solve for Yeo-Johnson parameters using closed-form solution."""
    # Placeholder for actual implementation
    return {'lambda': np.zeros(X.shape[1])}

def _solve_gradient_descent(
    X: np.ndarray,
    metric_func: Callable,
    tol: float,
    max_iter: int,
    **kwargs
) -> Dict:
    """Solve for Yeo-Johnson parameters using gradient descent."""
    # Placeholder for actual implementation
    return {'lambda': np.zeros(X.shape[1])}

def _transform_data(X: np.ndarray, params: Dict) -> np.ndarray:
    """Apply Yeo-Johnson transformation to data."""
    # Placeholder for actual implementation
    return X

def _compute_metrics(X: np.ndarray, transformed_X: np.ndarray, metric_func: Callable) -> Dict:
    """Compute metrics for the transformation."""
    return {'metric': metric_func(X, transformed_X)}

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

################################################################################
# standard_scaler
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input contains infinite values")

def _standard_normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply standard normalization to data."""
    return (X - mean) / std

def _minmax_normalize(X: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    """Apply min-max normalization to data."""
    return (X - min_vals) / (max_vals - min_vals)

def _robust_normalize(X: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    """Apply robust normalization to data."""
    return (X - median) / iqr

def _compute_metrics(X: np.ndarray, X_transformed: np.ndarray,
                     metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute metrics for transformed data."""
    metrics = {}
    for name, func in metric_funcs.items():
        if callable(func):
            metrics[name] = func(X, X_transformed)
    return metrics

def standard_scaler_fit(
    X: np.ndarray,
    normalization: str = "standard",
    metric_funcs: Optional[Dict[str, Callable]] = None,
    custom_normalize: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Fit a standard scaler and transform data.

    Parameters:
    - X: Input data (2D numpy array)
    - normalization: Type of normalization ("standard", "minmax", "robust")
    - metric_funcs: Dictionary of metric functions to compute
    - custom_normalize: Custom normalization function

    Returns:
    - Dictionary containing transformed data, metrics, parameters used, and warnings
    """
    _validate_inputs(X)

    if metric_funcs is None:
        metric_funcs = {}

    params_used = {
        "normalization": normalization,
        "metric_functions": list(metric_funcs.keys())
    }

    warnings = []

    if custom_normalize is not None:
        X_transformed = custom_normalize(X)
    else:
        if normalization == "standard":
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            X_transformed = _standard_normalize(X, mean, std)
        elif normalization == "minmax":
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)
            X_transformed = _minmax_normalize(X, min_vals, max_vals)
        elif normalization == "robust":
            median = np.median(X, axis=0)
            iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
            X_transformed = _robust_normalize(X, median, iqr)
        else:
            raise ValueError(f"Unknown normalization type: {normalization}")

    metrics = _compute_metrics(X, X_transformed, metric_funcs)

    return {
        "result": X_transformed,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
"""
X = np.array([[1, 2], [3, 4], [5, 6]])
result = standard_scaler_fit(X, normalization="standard")
print(result)
"""

################################################################################
# minmax_scaler
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input contains infinite values")

def _minmax_transform(
    X: np.ndarray,
    feature_range: tuple = (0, 1),
    custom_min: Optional[np.ndarray] = None,
    custom_max: Optional[np.ndarray] = None
) -> np.ndarray:
    """Apply min-max scaling to the input data."""
    if custom_min is not None and custom_max is not None:
        min_vals = custom_min
        max_vals = custom_max
    else:
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)

    # Avoid division by zero
    delta = max_vals - min_vals
    delta[delta == 0] = 1

    X_scaled = (X - min_vals) / delta
    X_scaled *= (feature_range[1] - feature_range[0])
    X_scaled += feature_range[0]

    return X_scaled

def minmax_scaler_fit(
    X: np.ndarray,
    feature_range: tuple = (0, 1),
    custom_min: Optional[np.ndarray] = None,
    custom_max: Optional[np.ndarray] = None
) -> Dict[str, Union[np.ndarray, dict]]:
    """
    Fit min-max scaler and return transformation parameters.

    Parameters
    ----------
    X : np.ndarray
        Input data to be scaled.
    feature_range : tuple, optional
        Desired range of transformed data (default is (0, 1)).
    custom_min : np.ndarray, optional
        Custom minimum values for each feature.
    custom_max : np.ndarray, optional
        Custom maximum values for each feature.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Transformed data
        - 'metrics': Metrics dictionary (empty for this function)
        - 'params_used': Parameters used
        - 'warnings': Warnings dictionary (empty for this function)
    """
    _validate_input(X)

    if custom_min is not None:
        if len(custom_min) != X.shape[1]:
            raise ValueError("custom_min must have same number of features as input")
    if custom_max is not None:
        if len(custom_max) != X.shape[1]:
            raise ValueError("custom_max must have same number of features as input")

    X_scaled = _minmax_transform(X, feature_range, custom_min, custom_max)

    return {
        'result': X_scaled,
        'metrics': {},
        'params_used': {
            'feature_range': feature_range,
            'custom_min': custom_min,
            'custom_max': custom_max
        },
        'warnings': {}
    }

# Example usage:
# X = np.array([[1, 2], [3, 4], [5, 6]])
# result = minmax_scaler_fit(X)

################################################################################
# robust_scaler
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def robust_scaler_fit(
    X: np.ndarray,
    method: str = 'median',
    scale_method: str = 'mad',
    axis: int = 0,
    with_centering: bool = True,
    with_scaling: bool = True,
    custom_center_func: Optional[Callable] = None,
    custom_scale_func: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Fit a robust scaler to the data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    method : str, optional
        Method to compute center ('median' or 'mean')
    scale_method : str, optional
        Method to compute scale ('mad', 'std', or 'range')
    axis : int, optional
        Axis along which to compute statistics (0 for columns, 1 for rows)
    with_centering : bool, optional
        Whether to center the data
    with_scaling : bool, optional
        Whether to scale the data
    custom_center_func : callable, optional
        Custom function to compute center
    custom_scale_func : callable, optional
        Custom function to compute scale

    Returns
    -------
    dict
        Dictionary containing:
        - 'center': computed center values
        - 'scale': computed scale values
        - 'params_used': parameters used for fitting
    """
    # Validate input
    _validate_input(X, axis)

    # Compute center
    if custom_center_func is not None:
        center = custom_center_func(X, axis=axis)
    else:
        if method == 'median':
            center = np.median(X, axis=axis)
        elif method == 'mean':
            center = np.mean(X, axis=axis)
        else:
            raise ValueError("method must be 'median' or 'mean'")

    # Compute scale
    if custom_scale_func is not None:
        scale = custom_scale_func(X, axis=axis)
    else:
        if scale_method == 'mad':
            scale = _compute_mad(X, axis=axis)
        elif scale_method == 'std':
            scale = np.std(X, axis=axis)
        elif scale_method == 'range':
            scale = np.ptp(X, axis=axis)
        else:
            raise ValueError("scale_method must be 'mad', 'std', or 'range'")

    # Handle division by zero
    scale[scale == 0] = 1.0

    return {
        'center': center,
        'scale': scale,
        'params_used': {
            'method': method,
            'scale_method': scale_method,
            'axis': axis,
            'with_centering': with_centering,
            'with_scaling': with_scaling
        }
    }

def robust_scaler_compute(
    X: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    with_centering: bool = True,
    with_scaling: bool = True
) -> np.ndarray:
    """
    Transform data using precomputed center and scale.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    center : np.ndarray
        Precomputed center values of shape (n_features,)
    scale : np.ndarray
        Precomputed scale values of shape (n_features,)
    with_centering : bool, optional
        Whether to center the data
    with_scaling : bool, optional
        Whether to scale the data

    Returns
    -------
    np.ndarray
        Transformed data of shape (n_samples, n_features)
    """
    # Validate input dimensions
    if X.shape[1] != len(center) or X.shape[1] != len(scale):
        raise ValueError("Input dimensions must match center and scale dimensions")

    X_transformed = X.copy()

    if with_centering:
        X_transformed -= center

    if with_scaling:
        X_transformed /= scale

    return X_transformed

def _compute_mad(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute Median Absolute Deviation.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    axis : int, optional
        Axis along which to compute MAD

    Returns
    -------
    np.ndarray
        MAD values of shape (n_features,)
    """
    median = np.median(X, axis=axis)
    mad = np.median(np.abs(X - median), axis=axis)
    return mad

def _validate_input(X: np.ndarray, axis: int) -> None:
    """
    Validate input data.

    Parameters
    ----------
    X : np.ndarray
        Input data to validate
    axis : int
        Axis along which statistics will be computed

    Raises
    ------
    ValueError
        If input is invalid
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a numpy array")

    if X.ndim != 2:
        raise ValueError("Input must be 2-dimensional")

    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")

################################################################################
# normalizer
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

def _standard_normalize(X: np.ndarray) -> np.ndarray:
    """Standard normalization (z-score)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + 1e-8)

def _minmax_normalize(X: np.ndarray) -> np.ndarray:
    """Min-max normalization."""
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    return (X - min_vals) / (max_vals - min_vals + 1e-8)

def _robust_normalize(X: np.ndarray) -> np.ndarray:
    """Robust normalization using median and IQR."""
    med = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    iqr = q75 - q25
    return (X - med) / (iqr + 1e-8)

def _compute_metrics(X: np.ndarray, X_normalized: np.ndarray,
                     metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute metrics for normalized data."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(X, X_normalized)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def normalizer_fit(X: np.ndarray,
                   normalization_type: str = 'standard',
                   metric_funcs: Optional[Dict[str, Callable]] = None,
                   custom_normalize: Optional[Callable] = None) -> Dict:
    """
    Fit a normalizer to data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    normalization_type : str
        Type of normalization ('none', 'standard', 'minmax', 'robust')
    metric_funcs : dict
        Dictionary of metric functions to compute
    custom_normalize : callable
        Custom normalization function

    Returns
    -------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> result = normalizer_fit(X, normalization_type='standard')
    """
    _validate_inputs(X)

    # Initialize output dictionary
    result = {
        'result': None,
        'metrics': {},
        'params_used': {
            'normalization_type': normalization_type
        },
        'warnings': []
    }

    # Apply normalization
    if custom_normalize is not None:
        X_normalized = custom_normalize(X)
        result['params_used']['normalization_type'] = 'custom'
    elif normalization_type == 'none':
        X_normalized = X.copy()
    elif normalization_type == 'standard':
        X_normalized = _standard_normalize(X)
    elif normalization_type == 'minmax':
        X_normalized = _minmax_normalize(X)
    elif normalization_type == 'robust':
        X_normalized = _robust_normalize(X)
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")

    result['result'] = X_normalized

    # Compute metrics if provided
    if metric_funcs is not None:
        result['metrics'] = _compute_metrics(X, X_normalized, metric_funcs)

    return result

################################################################################
# quantile_transformer
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def quantile_transformer_fit(
    X: np.ndarray,
    n_quantiles: int = 100,
    output_distribution: str = 'normal',
    subsample: Optional[int] = None,
    random_state: Optional[int] = None,
    copy: bool = True
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Fit the quantile transformer on an array of features.

    Parameters
    ----------
    X : np.ndarray
        Input data to be transformed. Shape (n_samples, n_features).
    n_quantiles : int, default=100
        Number of quantiles to be computed.
    output_distribution : str, default='normal'
        Desired distribution of the transformed data. Can be 'normal' or 'uniform'.
    subsample : int, optional
        Number of samples to use for quantile estimation. If None, uses all samples.
    random_state : int, optional
        Random seed for reproducibility when subsampling.
    copy : bool, default=True
        Whether to copy the input data.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
        - 'quantiles_': Quantiles computed for each feature.
        - 'sample_quantiles_': Quantile values corresponding to the quantiles.
        - 'params_used': Parameters used for fitting.
        - 'warnings': Any warnings encountered during fitting.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> result = quantile_transformer_fit(X)
    """
    # Input validation
    X, input_shape = _validate_input(X, copy=copy)

    n_samples, n_features = X.shape

    # Subsampling if required
    if subsample is not None:
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.random.choice(n_samples, subsample, replace=False)
        X_subsampled = X[indices]
    else:
        X_subsampled = X

    # Initialize results
    quantiles_ = np.zeros((n_features, n_quantiles))
    sample_quantiles_ = np.linspace(0, 1, n_quantiles)

    warnings = []

    for i in range(n_features):
        try:
            feature_data = X_subsampled[:, i]
            if np.all(np.isnan(feature_data)):
                raise ValueError("Feature contains only NaN values")

            quantiles_[:, i] = np.nanpercentile(feature_data, sample_quantiles_ * 100)
        except Exception as e:
            warnings.append(f"Feature {i} could not be transformed: {str(e)}")
            quantiles_[:, i] = np.linspace(np.nanmin(X_subsampled[:, i]),
                                          np.nanmax(X_subsampled[:, i]), n_quantiles)

    return {
        'result': quantiles_,
        'sample_quantiles': sample_quantiles_,
        'params_used': {
            'n_quantiles': n_quantiles,
            'output_distribution': output_distribution,
            'subsample': subsample,
            'random_state': random_state
        },
        'warnings': warnings
    }

def _validate_input(X: np.ndarray, copy: bool = True) -> tuple:
    """
    Validate input data.

    Parameters
    ----------
    X : np.ndarray
        Input data to validate.
    copy : bool, default=True
        Whether to return a copy of the input data.

    Returns
    -------
    tuple
        Validated input array and its original shape.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")

    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")

    if np.isnan(X).any():
        raise ValueError("Input contains NaN values")

    if copy:
        return X.copy(), X.shape
    else:
        return X, X.shape

def quantile_transformer_compute(
    X: np.ndarray,
    transformer_params: Dict[str, Union[int, str, Optional[int]]],
    copy: bool = True
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Transform data using precomputed quantile transformer parameters.

    Parameters
    ----------
    X : np.ndarray
        Input data to be transformed. Shape (n_samples, n_features).
    transformer_params : Dict[str, Union[int, str, Optional[int]]]
        Parameters from the fitted quantile transformer.
    copy : bool, default=True
        Whether to copy the input data.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
        - 'transformed_data': Transformed data.
        - 'params_used': Parameters used for transformation.
        - 'warnings': Any warnings encountered during transformation.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> fit_result = quantile_transformer_fit(X)
    >>> transform_result = quantile_transformer_compute(X, fit_result['params_used'])
    """
    # Input validation
    X, _ = _validate_input(X, copy=copy)

    quantiles_ = transformer_params['result']
    sample_quantiles_ = transformer_params.get('sample_quantiles', np.linspace(0, 1, len(quantiles_[0])))
    output_distribution = transformer_params.get('output_distribution', 'normal')

    n_samples, n_features = X.shape

    # Initialize transformed data
    transformed_data = np.zeros_like(X, dtype=np.float64)

    warnings = []

    for i in range(n_features):
        try:
            feature_data = X[:, i]
            if np.all(np.isnan(feature_data)):
                raise ValueError("Feature contains only NaN values")

            # Find the quantile for each value
            indices = np.searchsorted(quantiles_[:, i], feature_data)
            transformed_values = sample_quantiles_[indices]

            # Apply inverse CDF of the desired distribution
            if output_distribution == 'normal':
                transformed_data[:, i] = _inverse_normal_cdf(transformed_values)
            else:
                transformed_data[:, i] = transformed_values
        except Exception as e:
            warnings.append(f"Feature {i} could not be transformed: {str(e)}")
            transformed_data[:, i] = X[:, i]

    return {
        'result': transformed_data,
        'params_used': transformer_params,
        'warnings': warnings
    }

def _inverse_normal_cdf(x: np.ndarray) -> np.ndarray:
    """
    Inverse of the standard normal cumulative distribution function.

    Parameters
    ----------
    x : np.ndarray
        Values between 0 and 1 to transform.

    Returns
    -------
    np.ndarray
        Transformed values.
    """
    # Using the approximation from Wikipedia
    a1 = -3.969683028665376e+01
    a2 =  2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 =  1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 =  2.506628277459239e+00
    b1 = -5.447609879822406e+01
    b2 =  1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 =  6.680131188771972e+01
    b5 = -1.328068155288572e+01
    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 =  4.374664141464968e+00
    c6 =  2.938163982698783e+00
    d1 =  7.784695709041462e-03
    d2 =  3.224671290700398e-01
    d3 =  2.445134137142996e+00
    d4 =  3.754408661907416e+00

    # Define breakpoints
    p_low = 0.02425
    p_high = 1 - p_low

    # Rational approximation for lower region
    if x < p_low:
        q = np.sqrt(-2*np.log(x))
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q + c6) / ((((d1*q+d2)*q+d3)*q+d4)*q + 1)

    # Rational approximation for central region
    if p_low <= x <= p_high:
        q = x - 0.5
        r = q*q
        return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r + a6)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r + 1)

    # Rational approximation for upper region
    if x > p_high:
        q = np.sqrt(-2*np.log(1-x))
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q + c6) / ((((d1*q+d2)*q+d3)*q+d4)*q + 1)

    return x

def _inverse_uniform_cdf(x: np.ndarray) -> np.ndarray:
    """
    Inverse of the uniform cumulative distribution function.

    Parameters
    ----------
    x : np.ndarray
        Values between 0 and 1 to transform.

    Returns
    -------
    np.ndarray
        Transformed values.
    """
    return x

################################################################################
# power_transformer
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input contains infinite values")

def _normalize_data(
    X: np.ndarray,
    method: str = "standard",
    **kwargs
) -> np.ndarray:
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

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    **kwargs
) -> float:
    """Compute specified metric between true and predicted values."""
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
        return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _solve_power_transform(
    X: np.ndarray,
    solver: str = "closed_form",
    **kwargs
) -> np.ndarray:
    """Solve for power transformation parameters."""
    if solver == "closed_form":
        # Box-Cox transform (lambda estimation)
        n, p = X.shape
        log_X = np.log(X + 1e-8)
        ones = np.ones((n, p))
        X_centered = log_X - (np.sum(log_X) / n)
        X_augmented = np.column_stack([X_centered, ones])
        cov_matrix = (1 / n) * X_augmented.T @ X_augmented
        eigvals, _ = np.linalg.eigh(cov_matrix)
        return eigvals[-1] / (eigvals.sum() + 1e-8)
    elif solver == "gradient_descent":
        # Gradient descent implementation
        lambda_ = np.ones(p)
        learning_rate = kwargs.get("learning_rate", 0.01)
        max_iter = kwargs.get("max_iter", 1000)
        tol = kwargs.get("tol", 1e-6)

        for _ in range(max_iter):
            grad = -2 * (X ** lambda_) @ (X ** (lambda_ - 1)) / n
            new_lambda = lambda_ - learning_rate * grad
            if np.linalg.norm(new_lambda - lambda_) < tol:
                break
            lambda_ = new_lambda

        return lambda_
    else:
        raise ValueError(f"Unknown solver: {solver}")

def power_transformer_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalization: str = "standard",
    metric: Union[str, Callable] = "mse",
    solver: str = "closed_form",
    **kwargs
) -> Dict:
    """Fit power transformer to data."""
    _validate_input(X)

    # Normalize data
    X_norm = _normalize_data(X, normalization, **kwargs)

    # Solve for power transform parameters
    lambda_ = _solve_power_transform(X_norm, solver, **kwargs)

    # Compute metrics if y is provided
    metrics = {}
    if y is not None:
        X_transformed = X ** lambda_
        metrics["metric"] = _compute_metric(y, X_transformed, metric, **kwargs)

    return {
        "result": X ** lambda_,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "solver": solver
        },
        "warnings": []
    }

################################################################################
# binary_encoding
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    X: np.ndarray,
    n_bits: int = 3,
    normalize: str = 'none',
    custom_metric: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, str]]:
    """
    Validate and preprocess inputs for binary encoding.

    Parameters
    ----------
    X : np.ndarray
        Input data to be encoded.
    n_bits : int, optional
        Number of bits for encoding (default: 3).
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    custom_metric : Callable, optional
        Custom metric function (default: None).

    Returns
    -------
    Dict[str, Union[np.ndarray, str]]
        Validated and processed inputs.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("Input X must be a numpy array.")
    if n_bits < 1:
        raise ValueError("n_bits must be at least 1.")
    if normalize not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("normalize must be one of: 'none', 'standard', 'minmax', 'robust'.")

    if normalize != 'none':
        X = _normalize_data(X, method=normalize)

    return {
        'X': X,
        'n_bits': n_bits,
        'normalize': normalize
    }

def _normalize_data(
    X: np.ndarray,
    method: str = 'standard'
) -> np.ndarray:
    """
    Normalize data using specified method.

    Parameters
    ----------
    X : np.ndarray
        Input data to normalize.
    method : str, optional
        Normalization method (default: 'standard').

    Returns
    -------
    np.ndarray
        Normalized data.
    """
    if method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        return X

def _binary_encode(
    X: np.ndarray,
    n_bits: int = 3
) -> np.ndarray:
    """
    Perform binary encoding on input data.

    Parameters
    ----------
    X : np.ndarray
        Input data to encode.
    n_bits : int, optional
        Number of bits for encoding (default: 3).

    Returns
    -------
    np.ndarray
        Binary encoded data.
    """
    max_val = 2 ** n_bits - 1
    scaled_X = np.clip(X * max_val, 0, max_val)
    encoded_X = np.zeros((X.shape[0], X.shape[1] * n_bits))

    for i in range(n_bits):
        mask = (scaled_X // (2 ** i)) % 2
        encoded_X[:, i * X.shape[1]:(i + 1) * X.shape[1]] = mask

    return encoded_X

def _compute_metrics(
    original: np.ndarray,
    encoded: np.ndarray,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """
    Compute metrics between original and encoded data.

    Parameters
    ----------
    original : np.ndarray
        Original input data.
    encoded : np.ndarray
        Encoded data.
    metric : str, optional
        Metric to compute ('mse', 'mae', 'r2') (default: 'mse').
    custom_metric : Callable, optional
        Custom metric function (default: None).

    Returns
    -------
    Dict[str, float]
        Computed metrics.
    """
    metrics = {}

    if custom_metric is not None:
        metrics['custom'] = custom_metric(original, encoded)
    else:
        if metric == 'mse':
            metrics['mse'] = np.mean((original - encoded) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(original - encoded))
        elif metric == 'r2':
            ss_res = np.sum((original - encoded) ** 2)
            ss_tot = np.sum((original - np.mean(original)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

def binary_encoding_fit(
    X: np.ndarray,
    n_bits: int = 3,
    normalize: str = 'none',
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Perform binary encoding on input data with configurable options.

    Parameters
    ----------
    X : np.ndarray
        Input data to be encoded.
    n_bits : int, optional
        Number of bits for encoding (default: 3).
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    metric : str, optional
        Metric to compute ('mse', 'mae', 'r2') (default: 'mse').
    custom_metric : Callable, optional
        Custom metric function (default: None).

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]
        Dictionary containing:
        - 'result': Encoded data
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in the transformation
        - 'warnings': Any warnings generated during processing

    Examples
    --------
    >>> X = np.array([[1.0], [2.5], [3.7]])
    >>> result = binary_encoding_fit(X, n_bits=2, normalize='standard')
    """
    warnings = []

    # Validate and preprocess inputs
    validated_inputs = _validate_inputs(X, n_bits=n_bits, normalize=normalize, custom_metric=custom_metric)
    X_processed = validated_inputs['X']
    n_bits_used = validated_inputs['n_bits']
    normalize_used = validated_inputs['normalize']

    # Perform binary encoding
    encoded_X = _binary_encode(X_processed, n_bits=n_bits_used)

    # Compute metrics
    metrics = _compute_metrics(X_processed, encoded_X, metric=metric, custom_metric=custom_metric)

    return {
        'result': encoded_X,
        'metrics': metrics,
        'params_used': {
            'n_bits': n_bits_used,
            'normalize': normalize_used
        },
        'warnings': warnings
    }

################################################################################
# one_hot_encoding
################################################################################

import numpy as np
from typing import Dict, Any, Callable, Optional, Union

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if len(X.shape) != 2:
        raise ValueError("Input X must be a 2D array.")
    if np.any(np.isnan(X)):
        raise ValueError("Input X contains NaN values.")
    if np.any(np.isinf(X)):
        raise ValueError("Input X contains infinite values.")

def _one_hot_encode(
    X: np.ndarray,
    categories: Optional[Dict[int, list]] = None
) -> np.ndarray:
    """Perform one-hot encoding on the input data."""
    if categories is None:
        unique_values = [np.unique(X[:, i]) for i in range(X.shape[1])]
    else:
        unique_values = [categories.get(i, np.unique(X[:, i])) for i in range(X.shape[1])]

    encoded = []
    for i, values in enumerate(unique_values):
        if len(values) == 1:
            encoded_col = np.zeros((X.shape[0], 1))
        else:
            encoded_col = np.zeros((X.shape[0], len(values)))
            for j, val in enumerate(values):
                encoded_col[:, j] = (X[:, i] == val).astype(int)
        encoded.append(encoded_col)

    return np.hstack(encoded)

def one_hot_encoding_fit(
    X: np.ndarray,
    categories: Optional[Dict[int, list]] = None,
    handle_unknown: str = 'ignore',
    sparse: bool = False
) -> Dict[str, Any]:
    """
    Fit one-hot encoding transformation.

    Parameters:
    -----------
    X : np.ndarray
        Input data to be transformed.
    categories : Optional[Dict[int, list]]
        Categories for each feature. If None, categories are inferred from the data.
    handle_unknown : str
        Strategy for handling unknown categories. Options: 'ignore', 'error'.
    sparse : bool
        Whether to return a sparse matrix.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the transformation result and metadata.
    """
    _validate_inputs(X)

    if categories is not None:
        for col, cats in categories.items():
            if len(np.unique(X[:, col])) > len(cats):
                raise ValueError(f"Feature {col} has more unique values than provided categories.")

    encoded = _one_hot_encode(X, categories)

    if sparse:
        from scipy.sparse import csr_matrix
        encoded = csr_matrix(encoded)

    return {
        "result": encoded,
        "metrics": {},
        "params_used": {
            "handle_unknown": handle_unknown,
            "sparse": sparse
        },
        "warnings": []
    }

# Example usage:
# X = np.array([[1, 2], [3, 4], [1, 5]])
# result = one_hot_encoding_fit(X)

################################################################################
# ordinal_encoding
################################################################################

import numpy as np
from typing import Dict, Any, Callable, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None
) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

def _compute_ordinal_mapping(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form'
) -> Dict[str, Any]:
    """Compute ordinal mapping based on specified parameters."""
    # This is a placeholder for the actual implementation
    # In a real scenario, this would contain the logic for different solvers and metrics

    if solver == 'closed_form':
        # Closed form solution implementation
        pass
    elif solver == 'gradient_descent':
        # Gradient descent implementation
        pass
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Placeholder for metric calculation
    if isinstance(metric, str):
        if metric == 'mse':
            pass  # MSE calculation
        elif metric == 'mae':
            pass  # MAE calculation
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        pass  # Custom metric
    else:
        raise TypeError("Metric must be a string or callable")

    # Placeholder for ordinal mapping
    ordinal_mapping = {}

    return {
        'ordinal_mapping': ordinal_mapping,
        'metric_value': 0.0,  # Placeholder
        'converged': True     # Placeholder
    }

def ordinal_encoding_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    normalization: Optional[str] = None,
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """
    Fit ordinal encoding to the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values array of shape (n_samples,), optional
    metric : Union[str, Callable]
        Metric to optimize. Can be 'mse', 'mae', or a custom callable.
    solver : str
        Solver to use. Can be 'closed_form', 'gradient_descent', etc.
    normalization : Optional[str]
        Normalization method. Can be None, 'standard', 'minmax', etc.
    regularization : Optional[str]
        Regularization method. Can be None, 'l1', 'l2', etc.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    _validate_inputs(X, y)

    # Placeholder for normalization
    if normalization is not None:
        pass  # Normalization implementation

    result = _compute_ordinal_mapping(X, y, metric, solver)

    return {
        'result': result['ordinal_mapping'],
        'metrics': {'value': result['metric_value']},
        'params_used': {
            'metric': metric,
            'solver': solver,
            'normalization': normalization,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

# Example usage:
# encoding = ordinal_encoding_fit(X_train, y_train, metric='mse', solver='closed_form')

################################################################################
# target_encoding
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray],
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
) -> None:
    """Validate input data and parameters."""
    if X.ndim != 1 or y.ndim != 1:
        raise ValueError("X and y must be 1-dimensional arrays.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X and y must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("X and y must not contain infinite values.")

def _normalize(
    X: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Apply normalization to the input array."""
    return normalizer(X)

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
) -> float:
    """Compute the specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    elif metric == "logloss":
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _fit_target_encoding(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray],
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
) -> Dict:
    """Fit target encoding to the data."""
    _validate_inputs(X, y, normalizer, metric)

    X_normalized = _normalize(X, normalizer)
    unique_values = np.unique(X_normalized)

    encoding_dict = {}
    metrics = {}

    for value in unique_values:
        mask = X_normalized == value
        y_sub = y[mask]
        if len(y_sub) > 0:
            encoding_dict[value] = np.mean(y_sub)
            metrics[f"mean_{value}"] = _compute_metric(y_sub, np.array([np.mean(y_sub)]), metric)

    return {
        "result": encoding_dict,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if hasattr(normalizer, "__name__") else "custom",
            "metric": metric if callable(metric) else metric,
        },
        "warnings": [],
    }

def target_encoding_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
) -> Dict:
    """
    Fit target encoding to the data.

    Parameters
    ----------
    X : np.ndarray
        Input categorical variable.
    y : np.ndarray
        Target variable.
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Normalization function to apply to X. Default is no normalization.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to evaluate the encoding. Default is "mse".

    Returns
    -------
    Dict
        Dictionary containing the encoding result, metrics, parameters used, and warnings.

    Example
    -------
    >>> X = np.array([1, 2, 1, 3, 2])
    >>> y = np.array([0.5, 1.0, 0.7, 1.2, 0.8])
    >>> result = target_encoding_fit(X, y)
    """
    return _fit_target_encoding(X, y, normalizer, metric)

def target_encoding_compute(
    X: np.ndarray,
    encoding_dict: Dict,
) -> np.ndarray:
    """
    Compute target encoding for new data using a pre-fitted encoding dictionary.

    Parameters
    ----------
    X : np.ndarray
        Input categorical variable to encode.
    encoding_dict : Dict
        Pre-fitted encoding dictionary from target_encoding_fit.

    Returns
    -------
    np.ndarray
        Encoded values for the input X.
    """
    encoded_values = []
    for value in X:
        if value in encoding_dict:
            encoded_values.append(encoding_dict[value])
        else:
            raise ValueError(f"Value {value} not found in encoding dictionary.")
    return np.array(encoded_values)

################################################################################
# frequency_encoding
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def frequency_encoding_fit(
    X: np.ndarray,
    normalize: str = 'none',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit frequency encoding transformation.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2', 'logloss') or custom callable.
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : str, optional
        Regularization method (None, 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function.
    custom_distance : callable, optional
        Custom distance function.

    Returns:
    --------
    Dict containing:
        - 'result': fitted transformation
        - 'metrics': computed metrics
        - 'params_used': parameters used
        - 'warnings': potential warnings

    Example:
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = frequency_encoding_fit(X)
    """
    # Validate inputs
    _validate_inputs(X, normalize, metric, distance, solver, regularization)

    # Normalize data
    X_normalized = _normalize_data(X, normalize)

    # Compute frequency encoding
    freq_encoding = _compute_frequency_encoding(X_normalized, distance)

    # Solve for transformation
    transformation = _solve_transformation(
        X_normalized,
        freq_encoding,
        solver,
        regularization,
        tol,
        max_iter
    )

    # Compute metrics
    metrics = _compute_metrics(X_normalized, transformation, metric, custom_metric)

    return {
        'result': transformation,
        'metrics': metrics,
        'params_used': {
            'normalize': normalize,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(X_normalized, transformation)
    }

def _validate_inputs(
    X: np.ndarray,
    normalize: str,
    metric: Union[str, Callable],
    distance: str,
    solver: str,
    regularization: Optional[str]
) -> None:
    """Validate input parameters."""
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if normalize not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("normalize must be one of: none, standard, minmax, robust")
    if isinstance(metric, str) and metric not in ['mse', 'mae', 'r2', 'logloss']:
        raise ValueError("metric must be one of: mse, mae, r2, logloss or a callable")
    if distance not in ['euclidean', 'manhattan', 'cosine', 'minkowski']:
        raise ValueError("distance must be one of: euclidean, manhattan, cosine, minkowski")
    if solver not in ['closed_form', 'gradient_descent', 'newton', 'coordinate_descent']:
        raise ValueError("solver must be one of: closed_form, gradient_descent, newton, coordinate_descent")
    if regularization is not None and regularization not in ['l1', 'l2', 'elasticnet']:
        raise ValueError("regularization must be one of: None, l1, l2, elasticnet")

def _normalize_data(
    X: np.ndarray,
    method: str
) -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError("Unknown normalization method")

def _compute_frequency_encoding(
    X: np.ndarray,
    distance_method: str
) -> np.ndarray:
    """Compute frequency encoding based on specified distance."""
    if distance_method == 'euclidean':
        distances = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
    elif distance_method == 'manhattan':
        distances = np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
    elif distance_method == 'cosine':
        distances = 1 - np.dot(X, X.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(X, axis=1)[np.newaxis, :])
    elif distance_method == 'minkowski':
        distances = np.sum(np.abs(X[:, np.newaxis] - X)**3, axis=2)**(1/3)
    else:
        raise ValueError("Unknown distance method")

    freq_encoding = 1 / (distances + np.eye(len(X)))
    return np.mean(freq_encoding, axis=1)

def _solve_transformation(
    X: np.ndarray,
    freq_encoding: np.ndarray,
    solver_method: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve for transformation using specified solver."""
    if solver_method == 'closed_form':
        return _solve_closed_form(X, freq_encoding, regularization)
    elif solver_method == 'gradient_descent':
        return _solve_gradient_descent(X, freq_encoding, regularization, tol, max_iter)
    elif solver_method == 'newton':
        return _solve_newton(X, freq_encoding, regularization, tol, max_iter)
    elif solver_method == 'coordinate_descent':
        return _solve_coordinate_descent(X, freq_encoding, regularization, tol, max_iter)
    else:
        raise ValueError("Unknown solver method")

def _solve_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Solve transformation using closed form solution."""
    XTX = X.T @ X
    if regularization == 'l1':
        # Lasso solution would require different approach
        pass
    elif regularization == 'l2':
        XTX += np.eye(X.shape[1]) * 0.1
    elif regularization == 'elasticnet':
        XTX += np.eye(X.shape[1]) * 0.1
    return np.linalg.inv(XTX) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve transformation using gradient descent."""
    # Implementation of gradient descent
    pass

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve transformation using Newton's method."""
    # Implementation of Newton's method
    pass

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve transformation using coordinate descent."""
    # Implementation of coordinate descent
    pass

def _compute_metrics(
    X: np.ndarray,
    transformation: np.ndarray,
    metric_method: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute metrics for the transformation."""
    y_pred = X @ transformation
    if isinstance(metric_method, str):
        if metric_method == 'mse':
            return {'mse': np.mean((y_pred - X) ** 2)}
        elif metric_method == 'mae':
            return {'mae': np.mean(np.abs(y_pred - X))}
        elif metric_method == 'r2':
            ss_res = np.sum((y_pred - X) ** 2)
            ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
            return {'r2': 1 - ss_res / ss_tot}
        elif metric_method == 'logloss':
            return {'logloss': -np.mean(X * np.log(y_pred) + (1-X) * np.log(1-y_pred))}
    elif callable(metric_method):
        return {'custom_metric': metric_method(X, y_pred)}
    elif callable(custom_metric):
        return {'custom_metric': custom_metric(X, y_pred)}
    else:
        raise ValueError("Unknown metric method")

def _check_warnings(
    X: np.ndarray,
    transformation: np.ndarray
) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isnan(X)):
        warnings.append("Input data contains NaN values")
    if np.any(np.isinf(transformation)):
        warnings.append("Transformation contains infinite values")
    return warnings

################################################################################
# label_encoding
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    categories: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Validate input data for label encoding.

    Parameters:
    -----------
    X : np.ndarray
        Input features array.
    y : Optional[np.ndarray]
        Target values (if supervised).
    categories : Optional[Dict[str, np.ndarray]]
        Predefined categories for each feature.

    Returns:
    --------
    Dict[str, Any]
        Validation results and warnings.
    """
    validation = {"warnings": []}

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")

    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")

    if categories is not None and not isinstance(categories, dict):
        raise TypeError("categories must be a dictionary or None")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    if np.any(np.isnan(X)):
        validation["warnings"].append("NaN values found in X and will be replaced by 0")

    if y is not None and np.any(np.isnan(y)):
        validation["warnings"].append("NaN values found in y and will be replaced by 0")

    return validation

def _compute_label_encoding(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    categories: Optional[Dict[str, np.ndarray]] = None,
    metric: Union[str, Callable] = "mse",
    solver: str = "closed_form"
) -> Dict[str, Any]:
    """
    Compute label encoding for categorical features.

    Parameters:
    -----------
    X : np.ndarray
        Input features array.
    y : Optional[np.ndarray]
        Target values (if supervised).
    categories : Optional[Dict[str, np.ndarray]]
        Predefined categories for each feature.
    metric : Union[str, Callable]
        Metric to optimize ("mse", "mae", "r2", or custom callable).
    solver : str
        Solver to use ("closed_form", "gradient_descent").

    Returns:
    --------
    Dict[str, Any]
        Encoding results and metrics.
    """
    # Initialize result dictionary
    result = {
        "result": None,
        "metrics": {},
        "params_used": {
            "metric": metric,
            "solver": solver
        },
        "warnings": []
    }

    # Validate inputs
    validation = _validate_inputs(X, y, categories)
    result["warnings"].extend(validation["warnings"])

    # Replace NaN values
    X = np.nan_to_num(X)
    if y is not None:
        y = np.nan_to_num(y)

    # Determine categories
    if categories is None:
        unique_values = [np.unique(X[:, i]) for i in range(X.shape[1])]
    else:
        unique_values = [categories.get(str(i), np.unique(X[:, i])) for i in range(X.shape[1])]

    # Initialize encoding
    encoded_X = np.zeros_like(X, dtype=float)

    # Encode each feature
    for i in range(X.shape[1]):
        # Get unique categories for this feature
        cats = unique_values[i]

        if len(cats) == 1:
            # All values are the same, assign 0
            encoded_X[:, i] = 0.0
        else:
            # Assign numerical labels to categories
            cat_to_label = {cat: idx for idx, cat in enumerate(cats)}
            encoded_X[:, i] = np.array([cat_to_label.get(val, 0) for val in X[:, i]])

    result["result"] = encoded_X

    # Compute metrics if y is provided
    if y is not None:
        result["metrics"] = _compute_metrics(encoded_X, y, metric)

    return result

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """
    Compute metrics for label encoding.

    Parameters:
    -----------
    X : np.ndarray
        Encoded features array.
    y : np.ndarray
        Target values.
    metric : Union[str, Callable]
        Metric to compute ("mse", "mae", "r2", or custom callable).

    Returns:
    --------
    Dict[str, float]
        Computed metrics.
    """
    metrics = {}

    if metric == "mse":
        metrics["mse"] = np.mean((X - y) ** 2)
    elif metric == "mae":
        metrics["mae"] = np.mean(np.abs(X - y))
    elif metric == "r2":
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - X) ** 2)
        metrics["r2"] = 1 - (ss_residual / ss_total)
    elif callable(metric):
        metrics["custom"] = metric(X, y)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

def label_encoding_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    categories: Optional[Dict[str, np.ndarray]] = None,
    metric: Union[str, Callable] = "mse",
    solver: str = "closed_form"
) -> Dict[str, Any]:
    """
    Fit label encoding to data.

    Parameters:
    -----------
    X : np.ndarray
        Input features array.
    y : Optional[np.ndarray]
        Target values (if supervised).
    categories : Optional[Dict[str, np.ndarray]]
        Predefined categories for each feature.
    metric : Union[str, Callable]
        Metric to optimize ("mse", "mae", "r2", or custom callable).
    solver : str
        Solver to use ("closed_form", "gradient_descent").

    Returns:
    --------
    Dict[str, Any]
        Encoding results and metrics.

    Example:
    --------
    >>> X = np.array([['A', 'B'], ['A', 'C'], ['D', 'E']])
    >>> y = np.array([1, 2, 3])
    >>> result = label_encoding_fit(X.astype('object'), y)
    """
    return _compute_label_encoding(X, y, categories, metric, solver)

# Example usage:
# X = np.array([['A', 'B'], ['A', 'C'], ['D', 'E']], dtype='object')
# y = np.array([1, 2, 3])
# result = label_encoding_fit(X, y)

################################################################################
# binning
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def binning_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    n_bins: int = 10,
    strategy: str = 'uniform',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    normalization: Optional[str] = None,
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict:
    """
    Fit binning transformation to data.

    Parameters:
    -----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values (n_samples,) for supervised binning
    n_bins : int
        Number of bins to create
    strategy : str
        Binning strategy ('uniform', 'quantile', 'kmeans')
    metric : Union[str, Callable]
        Metric to optimize ('mse', 'mae', 'r2', custom callable)
    distance : str
        Distance metric for clustering ('euclidean', 'manhattan', etc.)
    solver : str
        Solver method ('closed_form', 'gradient_descent')
    normalization : Optional[str]
        Normalization method (None, 'standard', 'minmax', 'robust')
    regularization : Optional[str]
        Regularization type (None, 'l1', 'l2', 'elasticnet')
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum iterations
    random_state : Optional[int]
        Random seed for reproducibility

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> binning_fit(X_train, y_train)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if requested
    X_normalized = _apply_normalization(X, normalization)

    # Choose binning strategy
    if strategy == 'uniform':
        bins = _uniform_binning(X_normalized, n_bins)
    elif strategy == 'quantile':
        bins = _quantile_binning(X_normalized, n_bins)
    elif strategy == 'kmeans':
        bins = _kmeans_binning(X_normalized, n_bins, distance)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Calculate metrics
    metrics = _calculate_metrics(X_normalized, y, bins, metric)

    # Prepare output
    result = {
        'bins': bins,
        'result': {'bin_edges': _get_bin_edges(bins)},
        'metrics': metrics,
        'params_used': {
            'n_bins': n_bins,
            'strategy': strategy,
            'metric': metric,
            'distance': distance,
            'normalization': normalization
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if y is not None and len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")

def _apply_normalization(
    X: np.ndarray,
    method: Optional[str]
) -> np.ndarray:
    """Apply normalization to data."""
    if method is None or method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _uniform_binning(X: np.ndarray, n_bins: int) -> Dict:
    """Create uniform bins."""
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    bin_edges = [np.linspace(min_val, max_val, n_bins + 1) for min_val, max_val in zip(min_vals, max_vals)]
    return {'bin_edges': bin_edges}

def _quantile_binning(X: np.ndarray, n_bins: int) -> Dict:
    """Create quantile-based bins."""
    bin_edges = [np.percentile(X[:, i], np.linspace(0, 100, n_bins + 1)) for i in range(X.shape[1])]
    return {'bin_edges': bin_edges}

def _kmeans_binning(X: np.ndarray, n_bins: int, distance: str) -> Dict:
    """Create bins using k-means clustering."""
    from sklearn.cluster import KMeans

    # Initialize and fit k-means
    kmeans = KMeans(n_clusters=n_bins, random_state=42)
    labels = kmeans.fit_predict(X)

    # Get bin centers
    bin_centers = kmeans.cluster_centers_

    return {'bin_centers': bin_centers}

def _calculate_metrics(
    X: np.ndarray,
    y: Optional[np.ndarray],
    bins: Dict,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate metrics for binning."""
    if y is None:
        return {'metric': 'unsupervised'}

    if isinstance(metric, str):
        if metric == 'mse':
            return {'mse': _calculate_mse(X, y, bins)}
        elif metric == 'mae':
            return {'mae': _calculate_mae(X, y, bins)}
        elif metric == 'r2':
            return {'r2': _calculate_r2(X, y, bins)}
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        return {'custom': metric(X, y, bins)}

def _calculate_mse(
    X: np.ndarray,
    y: np.ndarray,
    bins: Dict
) -> float:
    """Calculate mean squared error."""
    # Implementation of MSE calculation for binning
    pass

def _calculate_mae(
    X: np.ndarray,
    y: np.ndarray,
    bins: Dict
) -> float:
    """Calculate mean absolute error."""
    # Implementation of MAE calculation for binning
    pass

def _calculate_r2(
    X: np.ndarray,
    y: np.ndarray,
    bins: Dict
) -> float:
    """Calculate R-squared."""
    # Implementation of R2 calculation for binning
    pass

def _get_bin_edges(bins: Dict) -> np.ndarray:
    """Get bin edges from bins dictionary."""
    if 'bin_edges' in bins:
        return np.array(bins['bin_edges'])
    elif 'bin_centers' in bins:
        # For k-means, we need to estimate bin edges from centers
        return _estimate_bin_edges_from_centers(bins['bin_centers'])
    else:
        raise ValueError("Invalid bins dictionary")

################################################################################
# discretization
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def discretization_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    n_bins: int = 10,
    strategy: str = 'uniform',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    normalization: Optional[str] = None,
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict:
    """
    Fit a discretization model to the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values array of shape (n_samples,)
    n_bins : int
        Number of bins for discretization
    strategy : str
        Strategy for binning ('uniform', 'quantile', 'kmeans')
    metric : Union[str, Callable]
        Metric to optimize ('mse', 'mae', 'r2', custom callable)
    distance : str
        Distance metric for clustering ('euclidean', 'manhattan', etc.)
    solver : str
        Solver method ('closed_form', 'gradient_descent', etc.)
    normalization : Optional[str]
        Normalization method (None, 'standard', 'minmax', etc.)
    regularization : Optional[str]
        Regularization method (None, 'l1', 'l2', etc.)
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    random_state : Optional[int]
        Random seed for reproducibility

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_normalized = _apply_normalization(X, normalization)

    # Choose discretization strategy
    if strategy == 'uniform':
        bins = _uniform_binning(X_normalized, n_bins)
    elif strategy == 'quantile':
        bins = _quantile_binning(X_normalized, n_bins)
    elif strategy == 'kmeans':
        bins = _kmeans_binning(X_normalized, n_bins, distance)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Fit model based on solver
    if solver == 'closed_form':
        params = _closed_form_solver(X_normalized, y, bins)
    elif solver == 'gradient_descent':
        params = _gradient_descent_solver(X_normalized, y, bins, tol, max_iter, random_state)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if required
    if regularization:
        params = _apply_regularization(params, regularization)

    # Calculate metrics
    metrics = _calculate_metrics(X_normalized, y, params, metric)

    return {
        'result': bins,
        'metrics': metrics,
        'params_used': {
            'n_bins': n_bins,
            'strategy': strategy,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'normalization': normalization,
            'regularization': regularization
        },
        'warnings': _check_warnings(X_normalized, y)
    }

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

def _apply_normalization(X: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Apply normalization to input data."""
    if method is None:
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _uniform_binning(X: np.ndarray, n_bins: int) -> np.ndarray:
    """Uniform binning strategy."""
    bins = np.linspace(np.min(X), np.max(X), n_bins + 1)
    return bins

def _quantile_binning(X: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile binning strategy."""
    bins = np.percentile(X, np.linspace(0, 100, n_bins + 1))
    return bins

def _kmeans_binning(X: np.ndarray, n_bins: int, distance: str) -> np.ndarray:
    """K-means binning strategy."""
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_bins, random_state=42)
    kmeans.fit(X)
    return kmeans.cluster_centers_

def _closed_form_solver(X: np.ndarray, y: Optional[np.ndarray], bins: np.ndarray) -> Dict:
    """Closed form solution for discretization."""
    if y is None:
        raise ValueError("Target values y are required for closed form solver")
    # Implement closed form solution
    return {}

def _gradient_descent_solver(X: np.ndarray, y: Optional[np.ndarray], bins: np.ndarray,
                            tol: float, max_iter: int, random_state: Optional[int]) -> Dict:
    """Gradient descent solver for discretization."""
    if y is None:
        raise ValueError("Target values y are required for gradient descent solver")
    # Implement gradient descent solution
    return {}

def _apply_regularization(params: Dict, method: str) -> Dict:
    """Apply regularization to parameters."""
    if method == 'l1':
        # Implement L1 regularization
        return params
    elif method == 'l2':
        # Implement L2 regularization
        return params
    else:
        raise ValueError(f"Unknown regularization method: {method}")

def _calculate_metrics(X: np.ndarray, y: Optional[np.ndarray], params: Dict,
                      metric: Union[str, Callable]) -> Dict:
    """Calculate metrics for discretization."""
    if y is None:
        raise ValueError("Target values y are required to calculate metrics")
    # Implement metric calculation
    return {}

def _check_warnings(X: np.ndarray, y: Optional[np.ndarray]) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isnan(X)):
        warnings.append("Input data contains NaN values")
    if y is not None and np.any(np.isnan(y)):
        warnings.append("Target values contain NaN values")
    return warnings

################################################################################
# polynomial_features
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input X contains NaN or infinite values")

def _normalize_data(
    X: np.ndarray,
    normalization: str = "standard",
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize input data."""
    if custom_normalizer is not None:
        return custom_normalizer(X)

    if normalization == "standard":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        return (X - mean) / (std + 1e-8)
    elif normalization == "minmax":
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif normalization == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    elif normalization == "none":
        return X
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_polynomial_features(
    X: np.ndarray,
    degree: int = 2,
    include_bias: bool = True
) -> np.ndarray:
    """Compute polynomial features."""
    n_samples, n_features = X.shape
    X_poly = np.ones((n_samples, 1)) if include_bias else np.zeros((n_samples, 0))

    for d in range(1, degree + 1):
        for i in range(n_features):
            if d == 1:
                X_poly = np.hstack((X_poly, X[:, i:i+1]))
            else:
                for j in range(i, n_features):
                    if i == j:
                        X_poly = np.hstack((X_poly, (X[:, i:i+1] ** d)))
                    else:
                        X_poly = np.hstack((X_poly, (X[:, i:i+1] * X[:, j:j+1])))

    return X_poly

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    if custom_metric is not None:
        return {"custom": custom_metric(y_true, y_pred)}

    metrics_dict = {}
    if isinstance(metrics, str):
        if metrics == "mse":
            metrics_dict["mse"] = np.mean((y_true - y_pred) ** 2)
        elif metrics == "mae":
            metrics_dict["mae"] = np.mean(np.abs(y_true - y_pred))
        elif metrics == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics_dict["r2"] = 1 - (ss_res / (ss_tot + 1e-8))
        elif metrics == "logloss":
            y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
            metrics_dict["logloss"] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            raise ValueError(f"Unknown metric: {metrics}")
    elif callable(metrics):
        metrics_dict["custom"] = metrics(y_true, y_pred)
    else:
        raise TypeError("metrics must be a string or callable")

    return metrics_dict

def polynomial_features_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    degree: int = 2,
    include_bias: bool = True,
    normalization: str = "standard",
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metrics: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Fit polynomial features transformation.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values for metrics calculation
    degree : int, default=2
        Degree of the polynomial features
    include_bias : bool, default=True
        Whether to include bias (intercept) term
    normalization : str or callable, default="standard"
        Normalization method ("none", "standard", "minmax", "robust")
    custom_normalizer : Optional[Callable]
        Custom normalization function
    metrics : str or callable, default="mse"
        Metrics to compute ("mse", "mae", "r2", "logloss")
    custom_metric : Optional[Callable]
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": transformed features
        - "metrics": computed metrics (if y is provided)
        - "params_used": parameters used
        - "warnings": any warnings

    Example:
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = polynomial_features_fit(X, degree=2)
    """
    _validate_inputs(X)

    warnings = []
    params_used = {
        "degree": degree,
        "include_bias": include_bias,
        "normalization": normalization if custom_normalizer is None else "custom",
    }

    X_normalized = _normalize_data(X, normalization, custom_normalizer)
    X_poly = _compute_polynomial_features(X_normalized, degree, include_bias)

    metrics_dict = {}
    if y is not None:
        _validate_inputs(y.reshape(-1, 1))
        if len(y) != X.shape[0]:
            warnings.append("Warning: y length doesn't match X rows")
        else:
            metrics_dict = _compute_metrics(y, np.ones_like(y), metrics, custom_metric)

    return {
        "result": X_poly,
        "metrics": metrics_dict,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# interaction_terms
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def interaction_terms_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    interaction_degree: int = 1,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Compute interaction terms for feature transformation.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : Optional[np.ndarray]
        Target values of shape (n_samples,) if provided.
    interaction_degree : int
        Degree of interaction terms to compute (default: 1).
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : Union[str, Callable]
        Metric to evaluate interaction terms ('mse', 'mae', 'r2', etc.).
    distance : str
        Distance metric for interaction terms ('euclidean', 'manhattan', etc.).
    solver : str
        Solver method for optimization ('closed_form', 'gradient_descent', etc.).
    regularization : Optional[str]
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    tol : float
        Tolerance for convergence (default: 1e-4).
    max_iter : int
        Maximum number of iterations (default: 1000).
    custom_metric : Optional[Callable]
        Custom metric function if not using built-in metrics.
    custom_distance : Optional[Callable]
        Custom distance function if not using built-in distances.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if required
    X_normalized = _normalize_features(X, normalization)

    # Compute interaction terms
    interaction_terms = _compute_interaction_terms(X_normalized, interaction_degree)

    # Prepare output
    result = {
        'result': interaction_terms,
        'metrics': {},
        'params_used': {
            'interaction_degree': interaction_degree,
            'normalization': normalization,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

    # Compute metrics if y is provided
    if y is not None:
        result['metrics'] = _compute_metrics(interaction_terms, y, metric, custom_metric)

    return result

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    """Validate input dimensions and types."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None.")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

def _normalize_features(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize features based on the specified method."""
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

def _compute_interaction_terms(X: np.ndarray, degree: int) -> np.ndarray:
    """Compute interaction terms up to the specified degree."""
    if degree == 0:
        return X.copy()
    elif degree == 1:
        return np.column_stack([X, np.ones(X.shape[0])])
    else:
        terms = [X.copy()]
        for d in range(2, degree + 1):
            new_terms = []
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    if i == j:
                        new_terms.append(X[:, i] ** d)
                    else:
                        new_terms.append(X[:, i] * X[:, j])
            terms.extend(new_terms)
        return np.column_stack(terms)

def _compute_metrics(
    X_transformed: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute metrics for the transformed features."""
    if custom_metric is not None:
        return {'custom': custom_metric(X_transformed, y)}

    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((X_transformed @ np.linalg.pinv(X_transformed) @ y - y) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(X_transformed @ np.linalg.pinv(X_transformed) @ y - y))
    elif metric == 'r2':
        ss_res = np.sum((y - X_transformed @ np.linalg.pinv(X_transformed) @ y) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

################################################################################
# splines
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def splines_fit(
    x: np.ndarray,
    y: np.ndarray,
    knots: Optional[np.ndarray] = None,
    degree: int = 3,
    normalization: str = "standard",
    metric: Union[str, Callable] = "mse",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit spline transformations to the input data.

    Parameters:
    -----------
    x : np.ndarray
        Input features.
    y : np.ndarray
        Target values.
    knots : Optional[np.ndarray]
        Positions of the knots. If None, uses quantiles.
    degree : int
        Degree of the spline (default: 3).
    normalization : str
        Normalization method ("none", "standard", "minmax", "robust").
    metric : Union[str, Callable]
        Metric to optimize ("mse", "mae", "r2", custom callable).
    solver : str
        Solver method ("closed_form", "gradient_descent", "newton").
    regularization : Optional[str]
        Regularization type ("none", "l1", "l2", "elasticnet").
    alpha : float
        Regularization strength (default: 1.0).
    tol : float
        Tolerance for convergence (default: 1e-6).
    max_iter : int
        Maximum iterations (default: 1000).
    custom_metric : Optional[Callable]
        Custom metric function.
    custom_distance : Optional[Callable]
        Custom distance function.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(x, y)

    # Normalize data
    x_norm, y_norm = _normalize_data(x, y, normalization)

    # Set default knots if not provided
    if knots is None:
        knots = _set_default_knots(x, degree)

    # Choose solver
    if solver == "closed_form":
        params = _spline_closed_form(x_norm, y_norm, knots, degree)
    elif solver == "gradient_descent":
        params = _spline_gradient_descent(x_norm, y_norm, knots, degree, tol, max_iter)
    elif solver == "newton":
        params = _spline_newton(x_norm, y_norm, knots, degree, tol, max_iter)
    else:
        raise ValueError("Unsupported solver")

    # Apply regularization if specified
    if regularization is not None:
        params = _apply_regularization(params, x_norm, y_norm, knots, degree, regularization, alpha)

    # Compute metrics
    metrics = _compute_metrics(y_norm, params, x_norm, metric, custom_metric)

    # Prepare output
    result = {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "knots": knots,
            "degree": degree,
            "normalization": normalization,
            "metric": metric,
            "solver": solver,
            "regularization": regularization,
            "alpha": alpha
        },
        "warnings": []
    }

    return result

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-dimensional arrays")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("x and y must not contain NaN values")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("x and y must not contain infinite values")

def _normalize_data(x: np.ndarray, y: np.ndarray, method: str) -> tuple:
    """Normalize input data."""
    if method == "standard":
        x_norm = (x - np.mean(x)) / np.std(x)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif method == "minmax":
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == "robust":
        x_norm = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        x_norm, y_norm = x.copy(), y.copy()
    return x_norm, y_norm

def _set_default_knots(x: np.ndarray, degree: int) -> np.ndarray:
    """Set default knots using quantiles."""
    n_knots = max(2, len(x) // 10)
    quantiles = np.linspace(0, 1, n_knots)[1:-1]
    knots = np.quantile(x, quantiles)
    return np.sort(knots)

def _spline_closed_form(x: np.ndarray, y: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Closed-form solution for spline fitting."""
    # Design matrix construction and solving
    design_matrix = _construct_design_matrix(x, knots, degree)
    params = np.linalg.lstsq(design_matrix, y, rcond=None)[0]
    return params

def _spline_gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    knots: np.ndarray,
    degree: int,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent solver for spline fitting."""
    design_matrix = _construct_design_matrix(x, knots, degree)
    params = np.zeros(design_matrix.shape[1])
    for _ in range(max_iter):
        gradient = 2 * design_matrix.T @ (design_matrix @ params - y)
        params -= tol * gradient
    return params

def _spline_newton(
    x: np.ndarray,
    y: np.ndarray,
    knots: np.ndarray,
    degree: int,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Newton's method solver for spline fitting."""
    design_matrix = _construct_design_matrix(x, knots, degree)
    params = np.zeros(design_matrix.shape[1])
    for _ in range(max_iter):
        residual = design_matrix @ params - y
        gradient = 2 * design_matrix.T @ residual
        hessian = 2 * design_matrix.T @ design_matrix
        params -= np.linalg.solve(hessian, gradient)
    return params

def _construct_design_matrix(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Construct the design matrix for spline basis functions."""
    # This is a simplified version; actual implementation would use B-splines or similar
    n_basis = len(knots) + degree - 1
    design_matrix = np.zeros((len(x), n_basis))
    for i in range(len(x)):
        design_matrix[i, :] = _basis_functions(x[i], knots, degree)
    return design_matrix

def _basis_functions(x: float, knots: np.ndarray, degree: int) -> np.ndarray:
    """Compute basis functions for a given x."""
    # Simplified implementation; actual implementation would use proper spline basis
    return np.array([x**i for i in range(degree + 1)])

def _apply_regularization(
    params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    knots: np.ndarray,
    degree: int,
    method: str,
    alpha: float
) -> np.ndarray:
    """Apply regularization to the parameters."""
    design_matrix = _construct_design_matrix(x, knots, degree)
    if method == "l1":
        params = _l1_regularization(params, design_matrix, y, alpha)
    elif method == "l2":
        params = _l2_regularization(params, design_matrix, y, alpha)
    elif method == "elasticnet":
        params = _elasticnet_regularization(params, design_matrix, y, alpha)
    return params

def _l1_regularization(params: np.ndarray, design_matrix: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """L1 regularization (LASSO)."""
    # Simplified implementation; actual implementation would use coordinate descent or similar
    return params

def _l2_regularization(params: np.ndarray, design_matrix: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """L2 regularization (Ridge)."""
    # Simplified implementation; actual implementation would use closed-form solution
    return params

def _elasticnet_regularization(params: np.ndarray, design_matrix: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Elastic net regularization."""
    # Simplified implementation; actual implementation would combine L1 and L2
    return params

def _compute_metrics(
    y_true: np.ndarray,
    params: np.ndarray,
    x: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute metrics for the spline fit."""
    y_pred = _predict(x, params)
    metrics = {}
    if metric == "mse":
        metrics["mse"] = np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        metrics["mae"] = np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        metrics["r2"] = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    elif callable(metric):
        metrics["custom"] = metric(y_true, y_pred)
    if custom_metric is not None:
        metrics["custom"] = custom_metric(y_true, y_pred)
    return metrics

def _predict(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Predict using the fitted spline."""
    design_matrix = _construct_design_matrix(x, np.array([]), params.size)
    return design_matrix @ params

################################################################################
# fourier_transform
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    x: np.ndarray,
    n: Optional[int] = None,
    norm_type: str = 'ortho',
    axis: int = -1
) -> None:
    """Validate inputs for Fourier transform."""
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if x.ndim == 0:
        raise ValueError("Input must be at least 1-dimensional")
    if n is not None and (not isinstance(n, int) or n <= 0):
        raise ValueError("n must be a positive integer")
    if norm_type not in ['ortho', 'backward', 'forward']:
        raise ValueError("norm_type must be 'ortho', 'backward' or 'forward'")
    if axis not in range(-x.ndim, x.ndim):
        raise ValueError("axis must be a valid dimension")

def _compute_fft(
    x: np.ndarray,
    n: Optional[int] = None,
    norm_type: str = 'ortho',
    axis: int = -1
) -> np.ndarray:
    """Compute the FFT with specified normalization."""
    if n is None:
        n = x.shape[axis]

    fft_result = np.fft.fft(x, n=n, axis=axis)

    if norm_type == 'ortho':
        fft_result /= np.sqrt(n)
    elif norm_type == 'backward':
        fft_result *= 1 / n
    # else: norm_type == 'forward' (no normalization)

    return fft_result

def _compute_ifft(
    x: np.ndarray,
    n: Optional[int] = None,
    norm_type: str = 'ortho',
    axis: int = -1
) -> np.ndarray:
    """Compute the IFFT with specified normalization."""
    if n is None:
        n = x.shape[axis]

    ifft_result = np.fft.ifft(x, n=n, axis=axis)

    if norm_type == 'ortho':
        ifft_result /= np.sqrt(n)
    elif norm_type == 'forward':
        ifft_result *= 1 / n
    # else: norm_type == 'backward' (no normalization)

    return ifft_result

def fourier_transform_fit(
    x: np.ndarray,
    transform_type: str = 'fft',
    n: Optional[int] = None,
    norm_type: str = 'ortho',
    axis: int = -1
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Compute Fourier transform with configurable parameters.

    Parameters
    ----------
    x : np.ndarray
        Input array to transform.
    transform_type : str, optional
        Type of transform ('fft' or 'ifft'), by default 'fft'.
    n : int, optional
        Number of points in the transform, by default None (use x.shape[axis]).
    norm_type : str, optional
        Normalization type ('ortho', 'backward', or 'forward'), by default 'ortho'.
    axis : int, optional
        Axis along which to compute the transform, by default -1.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
        - 'result': The transformed array
        - 'params_used': Parameters used in the computation
        - 'warnings': Any warnings generated

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> result = fourier_transform_fit(x)
    """
    _validate_inputs(x, n, norm_type, axis)

    warnings = {}

    if transform_type == 'fft':
        result = _compute_fft(x, n, norm_type, axis)
    elif transform_type == 'ifft':
        result = _compute_ifft(x, n, norm_type, axis)
    else:
        raise ValueError("transform_type must be 'fft' or 'ifft'")

    return {
        'result': result,
        'params_used': {
            'transform_type': transform_type,
            'n': n,
            'norm_type': norm_type,
            'axis': axis
        },
        'warnings': warnings
    }

################################################################################
# wavelet_transform
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def wavelet_transform_fit(
    data: np.ndarray,
    wavelet_type: str = 'haar',
    level: int = 1,
    mode: str = 'zero',
    normalization: Optional[str] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'closed_form',
    custom_wavelet: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Perform wavelet transform on input data.

    Parameters
    ----------
    data : np.ndarray
        Input signal to be transformed.
    wavelet_type : str, optional
        Type of wavelet to use ('haar', 'db1', etc.).
    level : int, optional
        Decomposition level.
    mode : str, optional
        Signal extension mode ('zero', 'symmetric', etc.).
    normalization : str or None, optional
        Normalization method ('none', 'standard', etc.).
    metric : str or callable, optional
        Metric to evaluate transform quality.
    solver : str, optional
        Solver method ('closed_form', etc.).
    custom_wavelet : callable or None, optional
        Custom wavelet function.
    **kwargs :
        Additional solver-specific parameters.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Transformed data
        - 'metrics': Evaluation metrics
        - 'params_used': Parameters used
        - 'warnings': Any warnings

    Example
    -------
    >>> data = np.random.randn(1024)
    >>> result = wavelet_transform_fit(data, wavelet_type='haar', level=3)
    """
    # Validate inputs
    _validate_inputs(data, wavelet_type, custom_wavelet)

    # Select wavelet function
    wavelet_func = _get_wavelet_function(wavelet_type, custom_wavelet)

    # Perform transform
    transformed_data = _apply_wavelet_transform(
        data, wavelet_func, level, mode
    )

    # Normalize if requested
    if normalization:
        transformed_data = _apply_normalization(
            transformed_data, method=normalization
        )

    # Calculate metrics
    metrics = _calculate_metrics(
        data, transformed_data, metric=metric
    )

    # Prepare output
    result = {
        'result': transformed_data,
        'metrics': metrics,
        'params_used': {
            'wavelet_type': wavelet_type,
            'level': level,
            'mode': mode,
            'normalization': normalization,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    wavelet_type: str,
    custom_wavelet: Optional[Callable]
) -> None:
    """Validate input parameters."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array")
    if custom_wavelet is not None and wavelet_type != 'custom':
        raise ValueError("wavelet_type must be 'custom' when providing custom_wavelet")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")

def _get_wavelet_function(
    wavelet_type: str,
    custom_wavelet: Optional[Callable]
) -> Callable:
    """Get wavelet function based on parameters."""
    if custom_wavelet is not None:
        return custom_wavelet
    # In a real implementation, this would import the appropriate wavelet function
    return _default_wavelet_function

def _apply_wavelet_transform(
    data: np.ndarray,
    wavelet_func: Callable,
    level: int,
    mode: str
) -> np.ndarray:
    """Apply wavelet transform to data."""
    # In a real implementation, this would perform the actual transformation
    transformed = np.zeros_like(data)
    # Placeholder for actual transform logic
    return transformed

def _apply_normalization(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply normalization to data."""
    if method == 'standard':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        return data

def _calculate_metrics(
    original: np.ndarray,
    transformed: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Calculate metrics between original and transformed data."""
    metrics = {}

    if callable(metric):
        metrics['custom'] = metric(original, transformed)
    else:
        if metric == 'mse':
            metrics['mse'] = np.mean((original - transformed) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(original - transformed))
        elif metric == 'r2':
            ss_res = np.sum((original - transformed) ** 2)
            ss_tot = np.sum((original - np.mean(original)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

def _default_wavelet_function(data: np.ndarray) -> np.ndarray:
    """Default wavelet function (placeholder)."""
    return data

################################################################################
# pca_transformation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def pca_transformation_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalization: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Perform Principal Component Analysis (PCA) transformation on input data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int, optional
        Number of principal components to keep (default: 2)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard')
    distance_metric : Union[str, Callable], optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable (default: 'euclidean')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton') (default: 'closed_form')
    tol : float, optional
        Tolerance for convergence (default: 1e-6)
    max_iter : int, optional
        Maximum number of iterations (default: 1000)
    custom_metric : Optional[Callable], optional
        Custom metric function (default: None)

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Transformed data
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in the computation
        - 'warnings': Any warnings generated

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> result = pca_transformation_fit(X, n_components=3)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Normalize data
    X_normalized = _apply_normalization(X, normalization)

    # Choose distance metric
    distance_func = _get_distance_function(distance_metric)

    # Perform PCA based on solver choice
    if solver == 'closed_form':
        components, explained_variance = _pca_closed_form(X_normalized, n_components)
    elif solver == 'gradient_descent':
        components, explained_variance = _pca_gradient_descent(
            X_normalized, n_components, tol, max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Transform data
    X_transformed = _transform_data(X_normalized, components)

    # Calculate metrics
    metrics = _calculate_metrics(
        X_transformed,
        explained_variance,
        custom_metric=custom_metric
    )

    # Prepare output
    result = {
        'result': X_transformed,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'normalization': normalization,
            'distance_metric': distance_metric,
            'solver': solver
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array")
    if len(X.shape) != 2:
        raise ValueError("Input X must be a 2D array")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if n_components > X.shape[1]:
        raise ValueError("n_components cannot be greater than number of features")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the data."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _get_distance_function(metric: Union[str, Callable]) -> Callable:
    """Get distance function based on metric specification."""
    if callable(metric):
        return metric
    elif metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric == 'minkowski':
        return lambda x, y: np.sum(np.abs(x - y)**3)**(1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _pca_closed_form(X: np.ndarray, n_components: int) -> tuple:
    """Perform PCA using closed form solution."""
    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    components = eigenvectors[:, :n_components]
    explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)

    return components, explained_variance

def _pca_gradient_descent(
    X: np.ndarray,
    n_components: int,
    tol: float,
    max_iter: int
) -> tuple:
    """Perform PCA using gradient descent."""
    n_samples, n_features = X.shape
    components = np.random.randn(n_features, n_components)

    for _ in range(max_iter):
        # Project data
        projections = X @ components

        # Update components
        new_components = (X.T @ projections) / (projections.T @ projections)

        # Normalize components
        new_components = new_components / np.linalg.norm(new_components, axis=0)

        # Check convergence
        if np.allclose(components, new_components, atol=tol):
            break

        components = new_components

    # Calculate explained variance
    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    explained_variance = np.sum(eigenvalues) / np.sum(eigenvalues)

    return components, explained_variance

def _transform_data(X: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Transform data using principal components."""
    return X @ components

def _calculate_metrics(
    X_transformed: np.ndarray,
    explained_variance: Union[float, np.ndarray],
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Calculate metrics for PCA results."""
    metrics = {
        'explained_variance': explained_variance,
        'reconstruction_error': _calculate_reconstruction_error(X_transformed)
    }

    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(X_transformed)

    return metrics

def _calculate_reconstruction_error(X_transformed: np.ndarray) -> float:
    """Calculate reconstruction error."""
    # This is a placeholder - actual implementation would depend on the original data
    return np.mean(np.square(X_transformed))

################################################################################
# kernel_pca
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D array.")
    if np.any(np.isnan(X)):
        raise ValueError("Input X contains NaN values.")
    if np.any(np.isinf(X)):
        raise ValueError("Input X contains infinite values.")

def _normalize_data(
    X: np.ndarray,
    normalization: str = "standard",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize the input data."""
    if custom_normalization is not None:
        return custom_normalization(X)

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

def _compute_kernel_matrix(
    X: np.ndarray,
    kernel: str = "rbf",
    gamma: float = 1.0,
    custom_kernel: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> np.ndarray:
    """Compute the kernel matrix."""
    if custom_kernel is not None:
        return np.array([[custom_kernel(xi, xj) for xi in X] for xj in X])

    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    if kernel == "linear":
        K = X @ X.T
    elif kernel == "rbf":
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = np.exp(-gamma * np.linalg.norm(X[i] - X[j])**2)
    elif kernel == "poly":
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = (gamma * X[i].dot(X[j]) + 1)**3
    elif kernel == "sigmoid":
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = np.tanh(gamma * X[i].dot(X[j]) + 1)
    else:
        raise ValueError(f"Unknown kernel method: {kernel}")

    return K

def _center_kernel_matrix(K: np.ndarray) -> np.ndarray:
    """Center the kernel matrix."""
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n

def _eig_decomposition(K: np.ndarray, n_components: int) -> tuple:
    """Perform eigenvalue decomposition on the kernel matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues[:n_components], eigenvectors[:, :n_components]

def kernel_pca_fit(
    X: np.ndarray,
    n_components: int = 2,
    kernel: str = "rbf",
    gamma: float = 1.0,
    normalization: str = "standard",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_kernel: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, Union[str, float]]]]:
    """
    Perform Kernel PCA on the input data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    n_components : int, optional
        Number of principal components to keep, by default 2.
    kernel : str, optional
        Kernel type ("linear", "rbf", "poly", "sigmoid"), by default "rbf".
    gamma : float, optional
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid', by default 1.0.
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust"), by default "standard".
    custom_normalization : Callable[[np.ndarray], np.ndarray], optional
        Custom normalization function, by default None.
    custom_kernel : Callable[[np.ndarray, np.ndarray], float], optional
        Custom kernel function, by default None.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, Union[str, float]]]]
        Dictionary containing the transformed data, metrics, and parameters used.
    """
    _validate_inputs(X)

    # Normalize the data
    X_normalized = _normalize_data(
        X, normalization=normalization, custom_normalization=custom_normalization
    )

    # Compute the kernel matrix
    K = _compute_kernel_matrix(
        X_normalized, kernel=kernel, gamma=gamma, custom_kernel=custom_kernel
    )

    # Center the kernel matrix
    K_centered = _center_kernel_matrix(K)

    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = _eig_decomposition(K_centered, n_components)

    # Project the data onto the eigenvectors
    X_transformed = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    return {
        "result": X_transformed,
        "metrics": {
            "explained_variance_ratio": eigenvalues / np.sum(eigenvalues)
        },
        "params_used": {
            "n_components": n_components,
            "kernel": kernel if custom_kernel is None else "custom",
            "gamma": gamma,
            "normalization": normalization if custom_normalization is None else "custom"
        },
        "warnings": []
    }

# Example usage:
# X = np.random.rand(100, 5)
# result = kernel_pca_fit(X, n_components=2, kernel="rbf", gamma=1.0)

################################################################################
# tsne
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(X: np.ndarray) -> None:
    """Validate input data for t-SNE."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2-dimensional array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

def default_distance_metric(x: np.ndarray, y: np.ndarray) -> float:
    """Default Euclidean distance metric."""
    return np.linalg.norm(x - y)

def compute_pairwise_distances(X: np.ndarray, metric: Callable) -> np.ndarray:
    """Compute pairwise distances between data points."""
    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distances[i, j] = metric(X[i], X[j])
            distances[j, i] = distances[i, j]
    return distances

def compute_pairwise_affinities(distances: np.ndarray, perplexity: float) -> np.ndarray:
    """Compute pairwise affinities using Gaussian kernel."""
    n_samples = distances.shape[0]
    P = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        # Binary search to find sigma
        sigmas = np.ones(n_samples) * 0.1
        target_entropy = np.log(perplexity)
        low = -20.0
        high = 20.0
        for j in range(50):
            # Compute Gaussian kernel
            sum_P = np.sum(np.exp(-distances[i] ** 2 / (2 * sigmas ** 2)))
            if sum_P == 0:
                sum_P = 1e-10
            P[i] = np.exp(-distances[i] ** 2 / (2 * sigmas ** 2)) / sum_P
            P[i, i] = 0.0
            entropy = np.sum(P[i] * np.log2(P[i] + 1e-10))
            if np.abs(entropy - target_entropy) < 1e-5:
                break
            if entropy > target_entropy:
                high = sigmas[i]
                sigmas[i] *= np.exp((target_entropy - entropy) / entropy)
            else:
                low = sigmas[i]
                sigmas[i] /= np.exp((target_entropy - entropy) / (high - low))
    return P

def gradient_descent(Y: np.ndarray, P: np.ndarray, n_iter: int = 1000,
                     learning_rate: float = 200.0, momentum: float = 0.8) -> np.ndarray:
    """Perform gradient descent to optimize t-SNE embedding."""
    n_samples = Y.shape[0]
    Y_history = np.zeros((n_iter, n_samples, 2))
    dY = np.zeros_like(Y)
    iY = np.zeros_like(Y)
    gains = np.ones_like(Y)

    for iter in range(n_iter):
        # Compute pairwise affinities in low-dimensional space
        sum_Y = np.sum(np.square(Y), 1)
        num = -2.0 * np.dot(Y, Y.T)
        num = 1.0 / (1.0 + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n_samples), range(n_samples)] = 0.0
        Q = num / np.sum(num)
        PQ = (P - Q) * num

        # Compute gradient
        for i in range(n_samples):
            grad = np.zeros(2)
            for j in range(n_samples):
                if i != j:
                    grad += PQ[i, j] * (Y[i] - Y[j])
            dY[i] = grad

        # Perform the update
        if iter > 20:
            dY = momentum * iY - learning_rate * dY
        else:
            dY = -learning_rate * dY

        Y += dY
        Y_history[iter] = Y.copy()
        iY = dY

    return Y, Y_history

def tsne_fit(X: np.ndarray,
             n_components: int = 2,
             perplexity: float = 30.0,
             metric: Optional[Callable] = None,
             n_iter: int = 1000,
             learning_rate: float = 200.0,
             momentum: float = 0.8) -> Dict:
    """
    Perform t-SNE transformation on input data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Dimension of the embedded space (default: 2).
    perplexity : float, optional
        Perplexity parameter (default: 30.0).
    metric : Callable, optional
        Distance metric function (default: Euclidean distance).
    n_iter : int, optional
        Number of iterations (default: 1000).
    learning_rate : float, optional
        Learning rate for gradient descent (default: 200.0).
    momentum : float, optional
        Momentum for gradient descent (default: 0.8).

    Returns
    -------
    Dict
        Dictionary containing:
        - "result": Embedded data points of shape (n_samples, n_components)
        - "metrics": Dictionary of computed metrics
        - "params_used": Dictionary of parameters used
        - "warnings": List of warnings encountered

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> result = tsne_fit(X, n_components=2)
    """
    # Validate input
    validate_input(X)

    # Set default metric if not provided
    if metric is None:
        metric = default_distance_metric

    # Compute pairwise distances and affinities
    distances = compute_pairwise_distances(X, metric)
    P = compute_pairwise_affinities(distances, perplexity)

    # Initialize solution randomly
    np.random.seed(0)
    Y = np.random.randn(X.shape[0], n_components)

    # Optimize using gradient descent
    Y, _ = gradient_descent(Y, P, n_iter=n_iter,
                           learning_rate=learning_rate,
                           momentum=momentum)

    # Compute metrics
    metrics = {
        'kl_divergence': np.sum(P * np.log2(P / (P + 1e-10) + 1e-10))
    }

    # Prepare output
    result = {
        'result': Y,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'perplexity': perplexity,
            'metric': metric.__name__ if hasattr(metric, '__name__') else 'custom',
            'n_iter': n_iter,
            'learning_rate': learning_rate,
            'momentum': momentum
        },
        'warnings': []
    }

    return result

################################################################################
# umap
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def umap_fit(
    X: np.ndarray,
    n_components: int = 2,
    metric: Union[str, Callable] = 'euclidean',
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    learning_rate: float = 1.0,
    n_epochs: int = 200,
    random_state: Optional[int] = None,
    normalization: str = 'none',
    distance_function: Optional[Callable] = None,
    solver: str = 'gradient_descent',
    reg_strength: float = 0.1,
    tol: float = 1e-4,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Fit UMAP (Uniform Manifold Approximation and Projection) to the data.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of dimensions for the output data (default: 2).
    metric : str or callable, optional
        Distance metric to use (default: 'euclidean').
    n_neighbors : int, optional
        Number of nearest neighbors to consider (default: 15).
    min_dist : float, optional
        Minimum distance between embedded points (default: 0.1).
    spread : float, optional
        Effective scale of embedded distances (default: 1.0).
    learning_rate : float, optional
        Learning rate for optimization (default: 1.0).
    n_epochs : int, optional
        Number of training epochs (default: 200).
    random_state : int, optional
        Random seed for reproducibility (default: None).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    distance_function : callable, optional
        Custom distance function if not using a built-in metric.
    solver : str, optional
        Optimization solver ('gradient_descent', 'newton') (default: 'gradient_descent').
    reg_strength : float, optional
        Regularization strength (default: 0.1).
    tol : float, optional
        Tolerance for stopping criterion (default: 1e-4).
    verbose : bool, optional
        Whether to print progress (default: False).

    Returns:
    --------
    dict
        Dictionary containing the following keys:
        - 'result': Embedded data of shape (n_samples, n_components).
        - 'metrics': Dictionary of computed metrics.
        - 'params_used': Parameters used in the fitting process.
        - 'warnings': List of warnings encountered during fitting.
    """
    # Validate inputs
    _validate_inputs(X, n_components, metric, distance_function)

    # Normalize data if required
    X_normalized = _apply_normalization(X, normalization)

    # Compute distances
    if distance_function is None:
        distances = _compute_distances(X_normalized, metric)
    else:
        distances = distance_function(X_normalized)

    # Initialize UMAP parameters
    params_used = {
        'n_components': n_components,
        'metric': metric,
        'n_neighbors': n_neighbors,
        'min_dist': min_dist,
        'spread': spread,
        'learning_rate': learning_rate,
        'n_epochs': n_epochs,
        'random_state': random_state,
        'normalization': normalization,
        'solver': solver,
        'reg_strength': reg_strength
    }

    # Initialize embedded data
    if random_state is not None:
        np.random.seed(random_state)
    embedded_data = np.random.rand(X.shape[0], n_components)

    # Optimize embedding
    embedded_data = _optimize_embedding(
        distances, embedded_data, n_neighbors, min_dist,
        spread, learning_rate, n_epochs, solver,
        reg_strength, tol, verbose
    )

    # Compute metrics
    metrics = _compute_metrics(X_normalized, embedded_data, distances)

    # Prepare output
    result = {
        'result': embedded_data,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

    return result

def _validate_inputs(
    X: np.ndarray,
    n_components: int,
    metric: Union[str, Callable],
    distance_function: Optional[Callable]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if n_components <= 0:
        raise ValueError("n_components must be positive.")
    if metric not in ['euclidean', 'manhattan', 'cosine', 'minkowski'] and not callable(distance_function):
        raise ValueError("Invalid metric or distance function.")
    if distance_function is not None and not callable(distance_function):
        raise TypeError("distance_function must be a callable.")

def _apply_normalization(
    X: np.ndarray,
    normalization: str
) -> np.ndarray:
    """Apply specified normalization to the data."""
    if normalization == 'standard':
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalization == 'minmax':
        X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalization == 'robust':
        X_normalized = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        X_normalized = X.copy()
    return X_normalized

def _compute_distances(
    X: np.ndarray,
    metric: str
) -> np.ndarray:
    """Compute pairwise distances between data points."""
    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if metric == 'euclidean':
                dist = np.linalg.norm(X[i] - X[j])
            elif metric == 'manhattan':
                dist = np.sum(np.abs(X[i] - X[j]))
            elif metric == 'cosine':
                dist = 1 - np.dot(X[i], X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]))
            elif metric == 'minkowski':
                dist = np.sum(np.abs(X[i] - X[j]) ** 3) ** (1/3)
            distances[i, j] = dist
            distances[j, i] = dist

    return distances

def _optimize_embedding(
    distances: np.ndarray,
    embedded_data: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    spread: float,
    learning_rate: float,
    n_epochs: int,
    solver: str,
    reg_strength: float,
    tol: float,
    verbose: bool
) -> np.ndarray:
    """Optimize the embedding using specified solver."""
    if solver == 'gradient_descent':
        embedded_data = _gradient_descent(
            distances, embedded_data, n_neighbors,
            min_dist, spread, learning_rate,
            n_epochs, reg_strength, tol, verbose
        )
    elif solver == 'newton':
        embedded_data = _newton_method(
            distances, embedded_data, n_neighbors,
            min_dist, spread, learning_rate,
            n_epochs, reg_strength, tol, verbose
        )
    return embedded_data

def _gradient_descent(
    distances: np.ndarray,
    embedded_data: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    spread: float,
    learning_rate: float,
    n_epochs: int,
    reg_strength: float,
    tol: float,
    verbose: bool
) -> np.ndarray:
    """Perform gradient descent optimization."""
    for epoch in range(n_epochs):
        gradients = _compute_gradients(
            distances, embedded_data, n_neighbors,
            min_dist, spread
        )
        embedded_data -= learning_rate * gradients

        # Regularization
        if reg_strength > 0:
            embedded_data -= reg_strength * np.sign(embedded_data)

        # Check convergence
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {_compute_loss(distances, embedded_data)}")

        if np.linalg.norm(gradients) < tol:
            break
    return embedded_data

def _compute_gradients(
    distances: np.ndarray,
    embedded_data: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    spread: float
) -> np.ndarray:
    """Compute gradients for the embedding."""
    n_samples = embedded_data.shape[0]
    gradients = np.zeros_like(embedded_data)

    for i in range(n_samples):
        # Compute pairwise gradients
        for j in range(n_samples):
            if i != j:
                diff = embedded_data[i] - embedded_data[j]
                dist = np.linalg.norm(diff)
                if dist > 0:
                    grad = (dist - min_dist) / (spread * dist)
                    gradients[i] += grad * diff
    return gradients

def _compute_loss(
    distances: np.ndarray,
    embedded_data: np.ndarray
) -> float:
    """Compute the loss function."""
    n_samples = embedded_data.shape[0]
    total_loss = 0.0

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = np.linalg.norm(embedded_data[i] - embedded_data[j])
            total_loss += (distances[i, j] - dist) ** 2
    return total_loss / (n_samples * (n_samples - 1))

def _compute_metrics(
    X: np.ndarray,
    embedded_data: np.ndarray,
    distances: np.ndarray
) -> Dict[str, float]:
    """Compute metrics for the embedding."""
    # Compute reconstruction error
    reconstructed_distances = np.zeros_like(distances)
    for i in range(embedded_data.shape[0]):
        for j in range(i + 1, embedded_data.shape[0]):
            reconstructed_distances[i, j] = np.linalg.norm(embedded_data[i] - embedded_data[j])
            reconstructed_distances[j, i] = reconstructed_distances[i, j]

    reconstruction_error = np.mean((distances - reconstructed_distances) ** 2)

    # Compute explained variance
    total_variance = np.var(X)
    embedded_variance = np.var(embedded_data)
    explained_variance = embedded_variance / total_variance

    return {
        'reconstruction_error': reconstruction_error,
        'explained_variance': explained_variance
    }

def _newton_method(
    distances: np.ndarray,
    embedded_data: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    spread: float,
    learning_rate: float,
    n_epochs: int,
    reg_strength: float,
    tol: float,
    verbose: bool
) -> np.ndarray:
    """Perform Newton method optimization."""
    for epoch in range(n_epochs):
        gradients = _compute_gradients(
            distances, embedded_data, n_neighbors,
            min_dist, spread
        )
        hessian = _compute_hessian(embedded_data)

        # Update embedded data
        delta = np.linalg.solve(hessian, gradients)
        embedded_data -= learning_rate * delta

        # Regularization
        if reg_strength > 0:
            embedded_data -= reg_strength * np.sign(embedded_data)

        # Check convergence
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {_compute_loss(distances, embedded_data)}")

        if np.linalg.norm(gradients) < tol:
            break
    return embedded_data

def _compute_hessian(
    embedded_data: np.ndarray
) -> np.ndarray:
    """Compute the Hessian matrix."""
    n_samples = embedded_data.shape[0]
    hessian = np.zeros((n_samples * embedded_data.shape[1], n_samples * embedded_data.shape[1]))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            diff = embedded_data[i] - embedded_data[j]
            dist = np.linalg.norm(diff)
            if dist > 0:
                hessian[i, j] = (1 / dist) - ((diff[0] ** 2 + diff[1] ** 2) / (dist ** 3))
    return hessian

################################################################################
# autoencoder_transformation
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def autoencoder_transformation_fit(
    X: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Fit an autoencoder transformation model to the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], default=None
        Function to normalize the data. If None, no normalization is applied.
    distance_metric : str, default='euclidean'
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine', 'minkowski'.
    solver : str, default='gradient_descent'
        Solver to use. Options: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.
    regularization : Optional[str], default=None
        Regularization type. Options: None, 'l1', 'l2', 'elasticnet'.
    learning_rate : float, default=0.01
        Learning rate for gradient-based solvers.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], default=None
        Custom metric function to evaluate the transformation.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X)

    # Normalize data if a normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize parameters
    n_features = X.shape[1]
    encoder_weights = _initialize_weights(n_features, random_state)
    decoder_weights = _initialize_weights(n_features, random_state)

    # Fit the autoencoder
    encoder_weights, decoder_weights = _fit_autoencoder(
        X_normalized,
        encoder_weights,
        decoder_weights,
        distance_metric,
        solver,
        regularization,
        learning_rate,
        max_iter,
        tol
    )

    # Compute metrics
    reconstructed = _reconstruct_data(X_normalized, encoder_weights, decoder_weights)
    metrics = _compute_metrics(reconstructed, X_normalized, custom_metric)

    # Prepare output
    result = {
        'encoder_weights': encoder_weights,
        'decoder_weights': decoder_weights
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'distance_metric': distance_metric,
            'solver': solver,
            'regularization': regularization,
            'learning_rate': learning_rate,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray) -> None:
    """Validate the input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if len(X.shape) != 2:
        raise ValueError("Input X must be a 2D array.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input X must not contain NaN or infinite values.")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Apply normalization to the input data."""
    if normalizer is not None:
        return normalizer(X)
    return X

def _initialize_weights(n_features: int, random_state: Optional[int]) -> np.ndarray:
    """Initialize weights for the autoencoder."""
    if random_state is not None:
        np.random.seed(random_state)
    return np.random.randn(n_features, n_features)

def _fit_autoencoder(
    X: np.ndarray,
    encoder_weights: np.ndarray,
    decoder_weights: np.ndarray,
    distance_metric: str,
    solver: str,
    regularization: Optional[str],
    learning_rate: float,
    max_iter: int,
    tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the autoencoder using the specified solver."""
    for _ in range(max_iter):
        # Update weights based on the chosen solver
        if solver == 'gradient_descent':
            encoder_weights, decoder_weights = _gradient_descent_step(
                X,
                encoder_weights,
                decoder_weights,
                distance_metric,
                regularization,
                learning_rate
            )
        elif solver == 'closed_form':
            encoder_weights, decoder_weights = _closed_form_solution(X)
        # Add other solvers as needed

        # Check for convergence
        if _check_convergence(encoder_weights, decoder_weights, tol):
            break

    return encoder_weights, decoder_weights

def _gradient_descent_step(
    X: np.ndarray,
    encoder_weights: np.ndarray,
    decoder_weights: np.ndarray,
    distance_metric: str,
    regularization: Optional[str],
    learning_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Perform a single gradient descent step."""
    # Compute gradients and update weights
    # Implementation depends on the distance metric and regularization
    return encoder_weights, decoder_weights

def _closed_form_solution(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the closed-form solution for the autoencoder."""
    # Implementation of closed-form solution
    return np.linalg.pinv(X), X @ np.linalg.pinv(X)

def _check_convergence(encoder_weights: np.ndarray, decoder_weights: np.ndarray, tol: float) -> bool:
    """Check if the weights have converged."""
    # Implementation of convergence check
    return False

def _reconstruct_data(
    X: np.ndarray,
    encoder_weights: np.ndarray,
    decoder_weights: np.ndarray
) -> np.ndarray:
    """Reconstruct the data using the autoencoder."""
    return decoder_weights @ encoder_weights @ X

def _compute_metrics(
    reconstructed: np.ndarray,
    original: np.ndarray,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for the autoencoder."""
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(reconstructed, original)
    # Add other default metrics as needed
    return metrics

# Example usage:
# result = autoencoder_transformation_fit(X_train, normalizer=standard_normalize, distance_metric='euclidean')
