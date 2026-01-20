from __future__ import annotations

def _validate_inputs_line(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for line plot."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("x contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")
"""
Quantix – Module visualisation_2d
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# scatter_plot
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def _validate_inputs_scatter(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("x must be 2D and y must be 1D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("x contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def scatter_plot_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    normalization: str = "none",
    distance_metric: Union[str, Callable] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    custom_metric: Optional[Callable] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict:
    """
    Fit a scatter plot with configurable parameters.

    Parameters:
    -----------
    x : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalization : str, optional
        Normalization method: "none", "standard", "minmax", or "robust".
    distance_metric : str or callable, optional
        Distance metric: "euclidean", "manhattan", "cosine", "minkowski", or custom callable.
    solver : str, optional
        Solver method: "closed_form", "gradient_descent", "newton", or "coordinate_descent".
    regularization : str, optional
        Regularization type: None, "l1", "l2", or "elasticnet".
    custom_metric : callable, optional
        Custom metric function.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    random_state : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs_scatter(x, y)

    # Normalize data if specified
    x_normalized = _apply_normalization(x, normalization)

    # Prepare solver parameters
    solver_params = {
        "tol": tol,
        "max_iter": max_iter,
        "random_state": random_state
    }

    # Fit model based on solver choice
    if solver == "closed_form":
        params = _closed_form_solver(x_normalized, y)
    elif solver == "gradient_descent":
        params = _gradient_descent_solver(x_normalized, y, **solver_params)
    elif solver == "newton":
        params = _newton_solver(x_normalized, y, **solver_params)
    elif solver == "coordinate_descent":
        params = _coordinate_descent_solver(x_normalized, y, **solver_params)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization:
        params = _apply_regularization(params, x_normalized, y, regularization)

    # Calculate metrics
    metrics = _calculate_metrics(x_normalized, y, params, custom_metric)

    # Prepare output
    result = {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "distance_metric": distance_metric,
            "solver": solver,
            "regularization": regularization
        },
        "warnings": []
    }

    return result

def _validate_inputs_scatter(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("x must be 2D and y must be 1D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("x contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(x: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to input data."""
    if method == "none":
        return x
    elif method == "standard":
        return (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    elif method == "minmax":
        return (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
    elif method == "robust":
        return (x - np.median(x, axis=0)) / (np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _closed_form_solver(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form solution for scatter plot fitting."""
    return np.linalg.pinv(x.T @ x) @ x.T @ y

def _gradient_descent_solver(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Gradient descent solver for scatter plot fitting."""
    if random_state is not None:
        np.random.seed(random_state)
    n_features = x.shape[1]
    params = np.random.randn(n_features)

    for _ in range(max_iter):
        gradient = 2 * x.T @ (x @ params - y) / len(y)
        params -= gradient
        if np.linalg.norm(gradient) < tol:
            break

    return params

def _newton_solver(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Newton's method solver for scatter plot fitting."""
    if random_state is not None:
        np.random.seed(random_state)
    n_features = x.shape[1]
    params = np.random.randn(n_features)

    for _ in range(max_iter):
        gradient = 2 * x.T @ (x @ params - y) / len(y)
        hessian = 2 * x.T @ x / len(y)
        params -= np.linalg.pinv(hessian) @ gradient
        if np.linalg.norm(gradient) < tol:
            break

    return params

def _coordinate_descent_solver(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Coordinate descent solver for scatter plot fitting."""
    if random_state is not None:
        np.random.seed(random_state)
    n_features = x.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        for i in range(n_features):
            x_i = x[:, i]
            residual = y - np.dot(x, params) + params[i] * x_i
            params[i] = np.sum(x_i * residual) / np.sum(x_i ** 2)

        if np.linalg.norm(np.dot(x, params) - y) < tol:
            break

    return params

def _apply_regularization(
    params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply regularization to parameters."""
    if method == "l1":
        return _l1_regularization(params, x, y)
    elif method == "l2":
        return _l2_regularization(params, x, y)
    elif method == "elasticnet":
        return _elasticnet_regularization(params, x, y)
    else:
        raise ValueError(f"Unknown regularization method: {method}")

def _l1_regularization(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """L1 regularization."""
    alpha = 0.1
    gradient = 2 * x.T @ (x @ params - y) / len(y)
    gradient += alpha * np.sign(params)
    return params - gradient

def _l2_regularization(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """L2 regularization."""
    alpha = 0.1
    gradient = 2 * x.T @ (x @ params - y) / len(y)
    gradient += 2 * alpha * params
    return params - gradient

def _elasticnet_regularization(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Elastic net regularization."""
    alpha = 0.1
    l1_ratio = 0.5
    gradient = 2 * x.T @ (x @ params - y) / len(y)
    gradient += alpha * (l1_ratio * np.sign(params) + (1 - l1_ratio) * 2 * params)
    return params - gradient

def _calculate_metrics(
    x: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Calculate metrics for the scatter plot."""
    y_pred = x @ params
    metrics = {
        "mse": np.mean((y - y_pred) ** 2),
        "mae": np.mean(np.abs(y - y_pred)),
        "r2": 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    }

    if custom_metric is not None:
        metrics["custom"] = custom_metric(y, y_pred)

    return metrics

################################################################################
# line_plot
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for line plot."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("x contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _normalize_data(x: np.ndarray, y: np.ndarray,
                   x_norm: str = 'none',
                   y_norm: str = 'none') -> tuple:
    """Normalize x and y data based on specified normalization methods."""
    norm_funcs = {
        'none': lambda a: a,
        'standard': lambda a: (a - np.mean(a)) / np.std(a),
        'minmax': lambda a: (a - np.min(a)) / (np.max(a) - np.min(a)),
        'robust': lambda a: (a - np.median(a)) / (np.percentile(a, 75) - np.percentile(a, 25))
    }

    x_normalized = norm_funcs[x_norm](x)
    y_normalized = norm_funcs[y_norm](y)

    return x_normalized, y_normalized

def _compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    metrics: Union[str, List[str], Callable]) -> Dict:
    """Compute specified metrics between true and predicted values."""
    metric_funcs = {
        'mse': lambda yt, yp: np.mean((yt - yp) ** 2),
        'mae': lambda yt, yp: np.mean(np.abs(yt - yp)),
        'r2': lambda yt, yp: 1 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2),
        'logloss': lambda yt, yp: -np.mean(yt * np.log(yp) + (1 - yt) * np.log(1 - yp))
    }

    results = {}

    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if callable(metric):
            results[metric.__name__] = metric(y_true, y_pred)
        else:
            if metric not in metric_funcs:
                raise ValueError(f"Unknown metric: {metric}")
            results[metric] = metric_funcs[metric](y_true, y_pred)

    return results

def _fit_line(x: np.ndarray,
             y: np.ndarray,
             method: str = 'closed_form') -> tuple:
    """Fit a line to the data using specified method."""
    if method == 'closed_form':
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return m, c
    else:
        raise ValueError(f"Unsupported fitting method: {method}")

def line_plot_fit(x: np.ndarray,
                 y: np.ndarray,
                 x_norm: str = 'none',
                 y_norm: str = 'none',
                 metrics: Union[str, List[str], Callable] = ['mse', 'mae'],
                 method: str = 'closed_form') -> Dict:


    """
    Fit a line to 2D data and compute metrics.

    Parameters:
    -----------
    x : np.ndarray
        Input array for x-axis.
    y : np.ndarray
        Input array for y-axis.
    x_norm : str, optional
        Normalization method for x ('none', 'standard', 'minmax', 'robust').
    y_norm : str, optional
        Normalization method for y ('none', 'standard', 'minmax', 'robust').
    metrics : str, list of str, or callable, optional
        Metrics to compute ('mse', 'mae', 'r2', 'logloss') or custom callable.
    method : str, optional
        Fitting method ('closed_form').

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs_line(x, y)

    # Normalize data
    x_normed, y_normed = _normalize_data(x, y, x_norm, y_norm)

    # Fit line
    slope, intercept = _fit_line(x_normed, y_normed, method)

    # Compute predictions
    y_pred = slope * x_normed + intercept

    # Compute metrics
    metric_results = _compute_metrics(y_normed, y_pred, metrics)

    # Prepare output
    result = {
        'result': {
            'slope': slope,
            'intercept': intercept
        },
        'metrics': metric_results,
        'params_used': {
            'x_norm': x_norm,
            'y_norm': y_norm,
            'metrics': metrics if isinstance(metrics, list) else [metrics],
            'method': method
        },
        'warnings': []
    }

    return result

# Example usage:
# line_plot_fit(np.array([1, 2, 3]), np.array([2, 4, 6]))

################################################################################
# bar_plot
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for bar plot."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-dimensional arrays")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("x contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(data: np.ndarray, normalization: str) -> np.ndarray:
    """Apply specified normalization to data."""
    if normalization == "none":
        return data
    elif normalization == "standard":
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    elif normalization == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    elif normalization == "robust":
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     metrics: List[str],
                     custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute specified metrics between true and predicted values."""
    result = {}
    if "mse" in metrics:
        result["mse"] = np.mean((y_true - y_pred) ** 2)
    if "mae" in metrics:
        result["mae"] = np.mean(np.abs(y_true - y_pred))
    if "r2" in metrics:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        result["r2"] = 1 - (ss_res / ss_tot)
    if "logloss" in metrics:
        result["logloss"] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    if custom_metric is not None:
        result["custom"] = custom_metric(y_true, y_pred)
    return result

def bar_plot_fit(x: np.ndarray,
                 y: np.ndarray,
                 normalization: str = "none",
                 metrics: List[str] = ["mse"],
                 custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                 **kwargs) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], List[str]]]:
    """
    Create a bar plot with configurable options.

    """
    warnings = []
    # Validate inputs
    _validate_inputs_bar(x, y)
    # Apply normalization
    normalized_y = _apply_normalization(y.copy(), normalization)
    # Compute metrics (using y as both true and predicted for demonstration)
    computed_metrics = _compute_metrics(y, normalized_y, metrics, custom_metric)
    # Prepare output
    result = {
        "result": normalized_y,
        "metrics": computed_metrics,
        "params_used": {
            "normalization": normalization,
            "metrics": metrics
        },
        "warnings": warnings
    }
    return result

    if x.ndim != 1 or y.ndim != 1:
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-dimensional arrays")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("x contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")
        raise ValueError("x and y must be 1-dimensional arrays")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("x contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")


    # Apply normalization
    normalized_y = _apply_normalization(y.copy(), normalization)

    # Compute metrics (using y as both true and predicted for demonstration)
    computed_metrics = _compute_metrics(y, normalized_y, metrics, custom_metric)

    # Prepare output
    result = {
        "result": normalized_y,
        "metrics": computed_metrics,
        "params_used": {
            "normalization": normalization,
            "metrics": metrics
        },
        "warnings": warnings
    }

    return result

################################################################################
# histogram
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(data: np.ndarray) -> None:
    """Validate input data for histogram computation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def compute_bins(data: np.ndarray, bins: Union[int, np.ndarray], range: Optional[tuple] = None) -> tuple:
    """Compute bin edges and indices for histogram."""
    if isinstance(bins, int):
        if range is None:
            bin_edges = np.histogram_bin_edges(data, bins=bins)
        else:
            bin_edges = np.linspace(range[0], range[1], bins + 1)
    else:
        bin_edges = np.asarray(bins)

    if range is not None and (bin_edges[0] < range[0] or bin_edges[-1] > range[1]):
        raise ValueError("Bin edges must be within the specified range")

    bin_indices = np.digitize(data, bin_edges) - 1
    return bin_edges, bin_indices

def normalize_counts(counts: np.ndarray, norm: str = 'none') -> np.ndarray:
    """Normalize histogram counts."""
    if norm == 'none':
        return counts
    elif norm == 'standard':
        return (counts - np.mean(counts)) / np.std(counts)
    elif norm == 'minmax':
        return (counts - np.min(counts)) / (np.max(counts) - np.min(counts))
    elif norm == 'robust':
        return (counts - np.median(counts)) / (np.percentile(counts, 75) - np.percentile(counts, 25))
    else:
        raise ValueError(f"Unknown normalization method: {norm}")

def compute_metrics(counts: np.ndarray, bin_edges: np.ndarray, metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute metrics for histogram."""
    metrics = {}
    bin_widths = np.diff(bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(counts, bin_widths, bin_centers)
        except Exception as e:
            metrics[name] = np.nan
            print(f"Warning: Failed to compute metric {name}: {str(e)}")

    return metrics

def histogram_fit(
    data: np.ndarray,
    bins: Union[int, np.ndarray] = 10,
    range: Optional[tuple] = None,
    norm: str = 'none',
    metric_funcs: Dict[str, Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute a histogram with configurable parameters.

    Parameters:
    -----------
    data : np.ndarray
        Input data (1D array).
    bins : int or np.ndarray
        Number of bins or explicit bin edges.
    range : tuple, optional
        Lower/upper range of the bins.
    norm : str
        Normalization mode.
    metric_funcs : dict[str, Callable], optional
        Custom metric functions.
    """
    # Validate input data
    validate_input_hist(data)
    # Default metric functions if none provided
    if metric_funcs is None:
        metric_funcs = {
            'mean': lambda counts, widths, centers: np.sum(counts * centers) / np.sum(counts),
            'std': lambda counts, widths, centers: np.sqrt(np.sum(counts * (centers - np.mean(centers))**2) / np.sum(counts))
        }
    # Compute bins
    bin_edges, bin_indices = compute_bins(data, bins, range)
    # Compute counts
    counts, _ = np.histogram(data, bins=bin_edges)
    # Normalize counts
    normalized_counts = normalize_counts(counts, norm)
    # Compute metrics
    metrics = compute_metrics(normalized_counts, bin_edges, metric_funcs)
    # Prepare output
    result = {
        'counts': counts,
        'bin_edges': bin_edges,
        'normalized_counts': normalized_counts
    }
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'bins': bins,
            'range': range,
            'norm': norm
        },
        'warnings': []
    }
    return output


    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")
    # Prepare output
    result = {
        'counts': counts,
        'bin_edges': bin_edges,
        'normalized_counts': normalized_counts
    }

    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'bins': bins,
            'range': range,
            'norm': norm
        },
        'warnings': []
    }

    return output

# Example usage:
# data = np.random.normal(0, 1, 1000)
# result = histogram_fit(data, bins=30, norm='standard')

################################################################################
# box_plot
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def box_plot_fit(
    data: np.ndarray,
    *,
    normalization: str = 'none',
    metrics: Union[str, List[str]] = ['median', 'q1', 'q3', 'iqr', 'whiskers', 'outliers'],
    custom_metrics: Optional[Callable[[np.ndarray], Dict[str, float]]] = None,
    whisker_method: str = 'iqr',
    outlier_threshold: float = 1.5,
    return_outliers: bool = False
) -> Dict[str, Union[Dict[str, float], np.ndarray]]:
    """
    Compute box plot statistics for 2D visualization.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metrics : str or list of str, optional
        Metrics to compute ('median', 'q1', 'q3', 'iqr', 'whiskers', 'outliers')
    custom_metrics : callable, optional
        Custom metric function that takes data and returns dict of metrics
    whisker_method : str, optional
        Method for calculating whiskers ('iqr', 'std')
    outlier_threshold : float, optional
        Threshold for outlier detection (multiplier of IQR)
    return_outliers : bool, optional
        Whether to return outlier indices

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': computed metrics for each feature
        - 'metrics': list of requested metrics
        - 'params_used': parameters used in computation
        - 'warnings': any warnings encountered

    Example
    -------
    >>> data = np.random.randn(100, 5)
    >>> result = box_plot_fit(data, normalization='standard', metrics=['median', 'q1'])
    """
    # Validate inputs
    _validate_inputs(data, metrics)

    # Normalize data if requested
    normalized_data = _apply_normalization(data, normalization)

    # Compute box plot statistics for each feature
    results = []
    warnings = []

    for i in range(normalized_data.shape[1]):
        feature_data = normalized_data[:, i]
        metrics_result = _compute_boxplot_metrics(
            feature_data,
            metrics=metrics,
            custom_metrics=custom_metrics,
            whisker_method=whisker_method,
            outlier_threshold=outlier_threshold
        )
        results.append(metrics_result)

    output = {
        'result': results,
        'metrics': metrics if isinstance(metrics, list) else [metrics],
        'params_used': {
            'normalization': normalization,
            'whisker_method': whisker_method,
            'outlier_threshold': outlier_threshold
        },
        'warnings': warnings
    }

    if return_outliers:
        output['outliers'] = _get_all_outliers(normalized_data, metrics_result)

    return output

def _validate_inputs(data: np.ndarray, metrics: Union[str, List[str]]) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input data must be a 2D numpy array")

    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values")

    valid_metrics = {'median', 'q1', 'q3', 'iqr', 'whiskers', 'outliers'}
    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if metric not in valid_metrics and not callable(metric):
            raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics} or a callable")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply requested normalization to data."""
    if method == 'none':
        return data.copy()

    normalized = np.zeros_like(data, dtype=np.float64)

    for i in range(data.shape[1]):
        feature = data[:, i]

        if method == 'standard':
            mean = np.mean(feature)
            std = np.std(feature)
            normalized[:, i] = (feature - mean) / (std + 1e-8)

        elif method == 'minmax':
            min_val = np.min(feature)
            max_val = np.max(feature)
            normalized[:, i] = (feature - min_val) / (max_val - min_val + 1e-8)

        elif method == 'robust':
            q1, q3 = np.percentile(feature, [25, 75])
            iqr = q3 - q1
            normalized[:, i] = (feature - q1) / (iqr + 1e-8)

    return normalized

def _compute_boxplot_metrics(
    data: np.ndarray,
    *,
    metrics: Union[str, List[str]],
    custom_metrics: Optional[Callable[[np.ndarray], Dict[str, float]]] = None,
    whisker_method: str = 'iqr',
    outlier_threshold: float = 1.5
) -> Dict[str, Union[float, np.ndarray]]:
    """Compute box plot metrics for a single feature."""
    result = {}

    if isinstance(metrics, str):
        metrics = [metrics]

    # Compute basic statistics
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1

    if 'median' in metrics:
        result['median'] = median
    if 'q1' in metrics:
        result['q1'] = q1
    if 'q3' in metrics:
        result['q3'] = q3
    if 'iqr' in metrics:
        result['iqr'] = iqr

    # Compute whiskers
    if 'whiskers' in metrics:
        if whisker_method == 'iqr':
            lower_whisker = q1 - outlier_threshold * iqr
            upper_whisker = q3 + outlier_threshold * iqr
        elif whisker_method == 'std':
            std = np.std(data)
            lower_whisker = median - 2 * std
            upper_whisker = median + 2 * std

        result['whiskers'] = np.array([lower_whisker, upper_whisker])

    # Compute outliers
    if 'outliers' in metrics:
        lower_bound = q1 - outlier_threshold * iqr
        upper_bound = q3 + outlier_threshold * iqr
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        result['outliers'] = outliers

    # Compute custom metrics if provided
    if custom_metrics is not None:
        result.update(custom_metrics(data))

    return result

def _get_all_outliers(data: np.ndarray, metrics_result: Dict[str, Union[float, np.ndarray]]) -> List[np.ndarray]:
    """Get outlier indices for all features."""
    outliers = []
    for i in range(data.shape[1]):
        feature_outliers = metrics_result[i].get('outliers')
        if feature_outliers is not None:
            outliers.append(np.where(data[:, i] == feature_outliers)[0])
    return outliers

################################################################################
# area_plot
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for area plot."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("x contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _normalize_data(x: np.ndarray, y: np.ndarray,
                   normalization: str = 'none') -> tuple[np.ndarray, np.ndarray]:
    """Normalize data according to specified method."""
    if normalization == 'none':
        return x, y
    elif normalization == 'standard':
        x_mean = np.mean(x)
        x_std = np.std(x)
        y_mean = np.mean(y)
        y_std = np.std(y)
        return (x - x_mean) / x_std, (y - y_mean) / y_std
    elif normalization == 'minmax':
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)
        return (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)
    elif normalization == 'robust':
        x_q1 = np.percentile(x, 25)
        x_q3 = np.percentile(x, 75)
        y_q1 = np.percentile(y, 25)
        y_q3 = np.percentile(y, 75)
        x_iqr = x_q3 - x_q1
        y_iqr = y_q3 - y_q1
        return (x - x_q1) / x_iqr, (y - y_q1) / y_iqr
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute specified metrics between true and predicted values."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = float(func(y_true, y_pred))
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def _compute_area(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the area under the curve using trapezoidal rule."""
    return np.trapz(y, x)

def area_plot_fit(x: np.ndarray,
                 y: np.ndarray,
                 normalization: str = 'none',
                 metric_funcs: Optional[Dict[str, Callable]] = None,
                 custom_metrics: bool = False) -> Dict:
    """
    Compute and visualize area plot with configurable options.

    Parameters:
    -----------
    x : np.ndarray
        Input values (x-axis)
    y : np.ndarray
        Output values (y-axis)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric_funcs : Dict[str, Callable], optional
        Dictionary of metric functions to compute
    custom_metrics : bool, optional
        Whether to include custom metrics

    Returns:
    --------
    Dict containing:
        - result: computed area
        - metrics: dictionary of computed metrics
        - params_used: parameters used in computation
        - warnings: any warnings generated

    Example:
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([4, 5, 6])
    >>> area_plot_fit(x, y)
    """
    # Initialize output dictionary
    result = {
        'result': None,
        'metrics': {},
        'params_used': {
            'normalization': normalization
        },
        'warnings': []
    }

    # Validate inputs
    _validate_inputs(x, y)

    # Normalize data if specified
    x_norm, y_norm = _normalize_data(x, y, normalization)
    result['params_used']['normalization'] = normalization

    # Compute area
    area = _compute_area(x_norm, y_norm)
    result['result'] = area

    # Compute metrics if specified
    if metric_funcs is not None:
        result['metrics'] = _compute_metrics(y, y_norm, metric_funcs)

    return result

################################################################################
# pie_chart
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def pie_chart_fit(
    values: np.ndarray,
    labels: Optional[np.ndarray] = None,
    normalize: str = 'none',
    start_angle: float = 0.0,
    colors: Optional[np.ndarray] = None,
    explode: Optional[np.ndarray] = None,
    autopct: Union[str, Callable[[float], str]] = None,
    pctdistance: float = 0.6,
    labeldistance: float = 1.1,
    shadow: bool = False,
    radius: float = 1.0
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Compute parameters for a pie chart visualization.

    Parameters
    ----------
    values : np.ndarray
        Array of numerical data to be plotted.
    labels : Optional[np.ndarray], default=None
        Labels for each wedge. If None, no labels are shown.
    normalize : str, default='none'
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    start_angle : float, default=0.0
        Rotation of the start of the pie chart.
    colors : Optional[np.ndarray], default=None
        Array of colors for each wedge. If None, default colors are used.
    explode : Optional[np.ndarray], default=None
        Array of offsets for each wedge. If None, no wedges are exploded.
    autopct : Union[str, Callable[[float], str]], default=None
        Format string or function for displaying percentages.
    pctdistance : float, default=0.6
        Distance between center and edge for displaying percentages.
    labeldistance : float, default=1.1
        Distance between center and edge for displaying labels.
    shadow : bool, default=False
        Whether to draw a shadow beneath the pie.
    radius : float, default=1.0
        Radius of the pie chart.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
        - "result": Computed parameters for the pie chart.
        - "metrics": Metrics related to the pie chart.
        - "params_used": Parameters used in the computation.
        - "warnings": Any warnings generated during computation.

    Examples
    --------
    >>> values = np.array([30, 25, 20, 15])
    >>> labels = np.array(['A', 'B', 'C', 'D'])
    >>> result = pie_chart_fit(values, labels)
    """
    # Validate inputs
    _validate_inputs(values, labels, colors, explode)

    # Normalize values if required
    normalized_values = _normalize_values(values, normalize)

    # Compute angles for each wedge
    total = np.sum(normalized_values)
    angles = (normalized_values / total) * 360

    # Prepare result dictionary
    result = {
        'angles': angles,
        'start_angle': start_angle,
        'radius': radius
    }

    # Prepare metrics dictionary
    metrics = {
        'total': total,
        'normalized_values': normalized_values
    }

    # Prepare params_used dictionary
    params_used = {
        'normalize': normalize,
        'start_angle': start_angle,
        'radius': radius
    }

    # Prepare warnings dictionary
    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(
    values: np.ndarray,
    labels: Optional[np.ndarray],
    colors: Optional[np.ndarray],
    explode: Optional[np.ndarray]
) -> None:
    """Validate input parameters."""
    if not isinstance(values, np.ndarray):
        raise TypeError("values must be a numpy array")
    if len(values) == 0:
        raise ValueError("values must not be empty")
    if labels is not None and len(labels) != len(values):
        raise ValueError("labels must have the same length as values")
    if colors is not None and len(colors) != len(values):
        raise ValueError("colors must have the same length as values")
    if explode is not None and len(explode) != len(values):
        raise ValueError("explode must have the same length as values")

def _normalize_values(
    values: np.ndarray,
    normalize: str
) -> np.ndarray:
    """Normalize the input values based on the specified method."""
    if normalize == 'none':
        return values
    elif normalize == 'standard':
        mean = np.mean(values)
        std = np.std(values)
        return (values - mean) / std
    elif normalize == 'minmax':
        min_val = np.min(values)
        max_val = np.max(values)
        return (values - min_val) / (max_val - min_val)
    elif normalize == 'robust':
        median = np.median(values)
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        return (values - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")

################################################################################
# heatmap
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def heatmap_fit(
    data: np.ndarray,
    x_bins: int = 10,
    y_bins: int = 10,
    normalization: str = "none",
    metric: Union[str, Callable] = "mean",
    distance_metric: str = "euclidean",
    interpolation: Optional[str] = None,
    **kwargs
) -> Dict:
    """
    Compute a 2D heatmap from input data.

    Parameters
    ----------
    data : np.ndarray
        Input 2D array of shape (n_samples, n_features).
    x_bins : int, optional
        Number of bins for the x-axis.
    y_bins : int, optional
        Number of bins for the y-axis.
    normalization : str, optional
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : str or callable, optional
        Aggregation metric: "mean", "sum", "max", "min", or custom callable.
    distance_metric : str, optional
        Distance metric for binning: "euclidean", "manhattan", or "cosine".
    interpolation : str, optional
        Interpolation method for smoothing: "linear", "cubic", or None.
    **kwargs
        Additional keyword arguments for specific methods.

    Returns
    -------
    Dict
        Dictionary containing:
        - "result": 2D heatmap array.
        - "metrics": Computed metrics.
        - "params_used": Parameters used in computation.
        - "warnings": Any warnings generated.

    Examples
    --------
    >>> data = np.random.rand(100, 2)
    >>> heatmap_fit(data, x_bins=5, y_bins=5, normalization="standard")
    """
    # Validate inputs
    _validate_inputs(data, x_bins, y_bins)

    # Normalize data if required
    normalized_data = _apply_normalization(data, normalization)

    # Compute heatmap bins
    x_edges, y_edges = _compute_bins(normalized_data, x_bins, y_bins)

    # Compute heatmap values
    heatmap_values = _compute_heatmap_values(
        normalized_data, x_edges, y_edges, metric
    )

    # Apply interpolation if required
    if interpolation is not None:
        heatmap_values = _apply_interpolation(heatmap_values, interpolation)

    # Prepare output
    result = {
        "result": heatmap_values,
        "metrics": {"x_bins": x_bins, "y_bins": y_bins},
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "distance_metric": distance_metric,
            "interpolation": interpolation,
        },
        "warnings": [],
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    x_bins: int,
    y_bins: int
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")
    if x_bins <= 0 or y_bins <= 0:
        raise ValueError("Number of bins must be positive integers.")

def _apply_normalization(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply normalization to the input data."""
    if method == "none":
        return data
    elif method == "standard":
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == "minmax":
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == "robust":
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_bins(
    data: np.ndarray,
    x_bins: int,
    y_bins: int
) -> tuple:
    """Compute bin edges for x and y axes."""
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
    x_edges = np.linspace(x_min, x_max, x_bins + 1)
    y_edges = np.linspace(y_min, y_max, y_bins + 1)
    return x_edges, y_edges

def _compute_heatmap_values(
    data: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    metric: Union[str, Callable]
) -> np.ndarray:
    """Compute heatmap values for each bin."""
    x_indices = np.digitize(data[:, 0], x_edges) - 1
    y_indices = np.digitize(data[:, 1], y_edges) - 1

    heatmap_values = np.zeros((len(y_edges) - 1, len(x_edges) - 1))

    for i in range(len(y_edges) - 1):
        for j in range(len(x_edges) - 1):
            mask = (x_indices == j) & (y_indices == i)
            bin_data = data[mask]

            if isinstance(metric, str):
                if metric == "mean":
                    heatmap_values[i, j] = np.mean(bin_data)
                elif metric == "sum":
                    heatmap_values[i, j] = np.sum(bin_data)
                elif metric == "max":
                    heatmap_values[i, j] = np.max(bin_data)
                elif metric == "min":
                    heatmap_values[i, j] = np.min(bin_data)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
            else:
                heatmap_values[i, j] = metric(bin_data)

    return heatmap_values

def _apply_interpolation(
    heatmap: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply interpolation to the heatmap."""
    if method == "linear":
        return _interpolate_linear(heatmap)
    elif method == "cubic":
        return _interpolate_cubic(heatmap)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

def _interpolate_linear(
    heatmap: np.ndarray
) -> np.ndarray:
    """Apply linear interpolation to the heatmap."""
    # Placeholder for actual interpolation logic
    return heatmap

def _interpolate_cubic(
    heatmap: np.ndarray
) -> np.ndarray:
    """Apply cubic interpolation to the heatmap."""
    # Placeholder for actual interpolation logic
    return heatmap

################################################################################
# violin_plot
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def violin_plot_fit(
    data: np.ndarray,
    *,
    normalization: str = 'none',
    kernel: str = 'gaussian',
    bandwidth: float = 1.0,
    num_points: int = 100,
    scale_factor: float = 1.0,
    quantiles: Optional[np.ndarray] = None,
    custom_kernel: Optional[Callable[[float], float]] = None,
    custom_bandwidth: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Union[Dict, np.ndarray]]:
    """
    Compute the violin plot representation for 2D data.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    kernel : str, optional
        Kernel type ('gaussian', 'epanechnikov', 'tophat').
    bandwidth : float, optional
        Bandwidth for kernel density estimation.
    num_points : int, optional
        Number of points to evaluate the KDE.
    scale_factor : float, optional
        Scaling factor for the violin plot width.
    quantiles : np.ndarray, optional
        Quantile levels to compute (default: [0.05, 0.95]).
    custom_kernel : Callable[[float], float], optional
        Custom kernel function.
    custom_bandwidth : Callable[[np.ndarray], float], optional
        Custom bandwidth function.

    Returns
    -------
    Dict[str, Union[Dict, np.ndarray]]
        Dictionary containing:
        - 'result': Computed violin plot data
        - 'metrics': Metrics used in computation
        - 'params_used': Parameters actually used
        - 'warnings': Any warnings generated

    Example
    -------
    >>> data = np.random.randn(100, 2)
    >>> result = violin_plot_fit(data, normalization='standard', kernel='gaussian')
    """
    # Validate inputs
    _validate_inputs(data, normalization)

    # Normalize data if required
    normalized_data = _apply_normalization(data, normalization)

    # Set default quantiles if not provided
    if quantiles is None:
        quantiles = np.array([0.05, 0.95])

    # Compute KDE for each feature
    kde_results = []
    for i in range(normalized_data.shape[1]):
        feature_data = normalized_data[:, i]
        kde_result = _compute_kde(
            feature_data,
            kernel=kernel,
            bandwidth=bandwidth,
            num_points=num_points,
            custom_kernel=custom_kernel,
            custom_bandwidth=custom_bandwidth
        )
        kde_results.append(kde_result)

    # Compute quantiles for each feature
    quantile_results = []
    for i in range(normalized_data.shape[1]):
        feature_data = normalized_data[:, i]
        q_result = _compute_quantiles(feature_data, quantiles)
        quantile_results.append(q_result)

    # Prepare output
    result = {
        'kde': kde_results,
        'quantiles': quantile_results
    }

    metrics = {
        'normalization_used': normalization,
        'kernel_type': kernel if custom_kernel is None else 'custom',
        'bandwidth_value': bandwidth,
        'num_points_used': num_points
    }

    params_used = {
        'normalization': normalization,
        'kernel': kernel if custom_kernel is None else 'custom',
        'bandwidth': bandwidth,
        'num_points': num_points,
        'scale_factor': scale_factor
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(data: np.ndarray, normalization: str) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional")
    if normalization not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("Invalid normalization method")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the data."""
    if method == 'none':
        return data
    elif method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)

def _compute_kde(
    data: np.ndarray,
    *,
    kernel: str = 'gaussian',
    bandwidth: float = 1.0,
    num_points: int = 100,
    custom_kernel: Optional[Callable[[float], float]] = None,
    custom_bandwidth: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, np.ndarray]:
    """Compute kernel density estimation for 1D data."""
    if custom_bandwidth is not None:
        bandwidth = custom_bandwidth(data)

    x_min, x_max = np.min(data), np.max(data)
    x_grid = np.linspace(x_min, x_max, num_points)

    if custom_kernel is not None:
        kernel_func = custom_kernel
    else:
        if kernel == 'gaussian':
            kernel_func = lambda u: np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        elif kernel == 'epanechnikov':
            kernel_func = lambda u: 0.75 * (1 - u**2) if np.abs(u) <= 1 else 0
        elif kernel == 'tophat':
            kernel_func = lambda u: 0.5 if np.abs(u) <= 1 else 0
        else:
            raise ValueError("Invalid kernel type")

    kde_values = np.zeros_like(x_grid)
    for i, x in enumerate(x_grid):
        u = (x - data) / bandwidth
        kde_values[i] = np.mean(kernel_func(u))

    return {
        'x_grid': x_grid,
        'kde_values': kde_values
    }

def _compute_quantiles(data: np.ndarray, quantiles: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute specified quantiles for 1D data."""
    sorted_data = np.sort(data)
    n = len(sorted_data)

    q_values = []
    for q in quantiles:
        if q <= 0 or q >= 1:
            raise ValueError("Quantile values must be between 0 and 1")
        idx = int(n * q)
        if idx == n:
            idx -= 1
        q_values.append(sorted_data[idx])

    return {
        'quantiles': quantiles,
        'values': np.array(q_values)
    }

################################################################################
# radar_plot
################################################################################

import numpy as np
from typing import Callable, Dict, List, Optional, Union

def validate_inputs(data: np.ndarray, features: List[str], normalize: str) -> None:
    """Validate input data and parameters for radar plot."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if len(features) != data.shape[1]:
        raise ValueError("Number of features must match number of columns in data")
    if normalize not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("Invalid normalization method")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")

def normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError("Invalid normalization method")

def compute_radar_angles(features: List[str]) -> np.ndarray:
    """Compute angles for radar plot based on number of features."""
    num_features = len(features)
    return np.linspace(0, 2 * np.pi, num_features, endpoint=False)

def radar_plot_fit(
    data: np.ndarray,
    features: List[str],
    normalize: str = 'standard',
    angle_offset: float = 0.0,
    metric: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict, List]]:
    """
    Create a radar plot visualization of multidimensional data.

    Parameters:
    -----------
    data : np.ndarray
        2D array of shape (n_samples, n_features)
    features : List[str]
        List of feature names
    normalize : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    angle_offset : float
        Angle offset in radians for the plot
    metric : Optional[Callable]
        Custom metric function to compute on normalized data

    Returns:
    --------
    Dict containing:
    - result: Normalized data ready for plotting
    - metrics: Computed metrics if provided
    - params_used: Dictionary of parameters used
    - warnings: List of any warnings generated

    Example:
    --------
    data = np.random.rand(5, 4)
    features = ['A', 'B', 'C', 'D']
    result = radar_plot_fit(data, features)
    """
    # Validate inputs
    validate_inputs(data, features, normalize)

    # Normalize data
    normalized_data = normalize_data(data.copy(), normalize)

    # Compute angles for radar plot
    angles = compute_radar_angles(features) + angle_offset

    # Compute metrics if provided
    metrics = {}
    if metric is not None:
        try:
            metrics['custom'] = metric(normalized_data)
        except Exception as e:
            warnings.append(f"Metric computation failed: {str(e)}")

    # Prepare output
    result = {
        'result': normalized_data,
        'metrics': metrics,
        'params_used': {
            'normalize': normalize,
            'angle_offset': angle_offset
        },
        'warnings': []
    }

    return result

def _compute_radar_metrics(data: np.ndarray, metrics: List[str]) -> Dict[str, float]:
    """Compute standard radar plot metrics."""
    result = {}
    if 'mean' in metrics:
        result['mean'] = np.mean(data, axis=0)
    if 'std' in metrics:
        result['std'] = np.std(data, axis=0)
    if 'min' in metrics:
        result['min'] = np.min(data, axis=0)
    if 'max' in metrics:
        result['max'] = np.max(data, axis=0)
    return result

def radar_plot_compute(
    data: np.ndarray,
    features: List[str],
    normalize: str = 'standard',
    metrics: Optional[List[str]] = None,
    angle_offset: float = 0.0
) -> Dict[str, Union[np.ndarray, Dict, List]]:
    """
    Compute radar plot visualization with standard metrics.

    Parameters:
    -----------
    data : np.ndarray
        2D array of shape (n_samples, n_features)
    features : List[str]
        List of feature names
    normalize : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metrics : Optional[List[str]]
        List of standard metrics to compute ('mean', 'std', 'min', 'max')
    angle_offset : float
        Angle offset in radians for the plot

    Returns:
    --------
    Dict containing:
    - result: Normalized data ready for plotting
    - metrics: Computed standard metrics if requested
    - params_used: Dictionary of parameters used
    - warnings: List of any warnings generated

    Example:
    --------
    data = np.random.rand(5, 4)
    features = ['A', 'B', 'C', 'D']
    result = radar_plot_compute(data, features, metrics=['mean', 'std'])
    """
    # Validate inputs
    validate_inputs(data, features, normalize)

    if metrics is None:
        metrics = []

    # Normalize data
    normalized_data = normalize_data(data.copy(), normalize)

    # Compute standard metrics if requested
    computed_metrics = {}
    if metrics:
        try:
            computed_metrics = _compute_radar_metrics(normalized_data, metrics)
        except Exception as e:
            warnings.append(f"Metrics computation failed: {str(e)}")

    # Prepare output
    result = {
        'result': normalized_data,
        'metrics': computed_metrics,
        'params_used': {
            'normalize': normalize,
            'angle_offset': angle_offset
        },
        'warnings': []
    }

    return result

################################################################################
# bubble_plot
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List, Tuple

def validate_inputs(x: np.ndarray,
                   y: np.ndarray,
                   sizes: np.ndarray,
                   normalize_x: bool = True,
                   normalize_y: bool = True,
                   size_normalization: str = 'none') -> None:
    """
    Validate inputs for bubble plot.

    Parameters
    ----------
    x : np.ndarray
        X coordinates of bubbles.
    y : np.ndarray
        Y coordinates of bubbles.
    sizes : np.ndarray
        Sizes of bubbles.
    normalize_x : bool, optional
        Whether to normalize x coordinates.
    normalize_y : bool, optional
        Whether to normalize y coordinates.
    size_normalization : str, optional
        Normalization method for sizes ('none', 'standard', 'minmax').

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if x.shape != y.shape or x.shape != sizes.shape:
        raise ValueError("x, y and sizes must have the same shape")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(np.isnan(sizes)):
        raise ValueError("Inputs must not contain NaN values")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)) or np.any(np.isinf(sizes)):
        raise ValueError("Inputs must not contain infinite values")

def normalize_data(data: np.ndarray,
                   method: str = 'standard') -> np.ndarray:
    """
    Normalize data using specified method.

    Parameters
    ----------
    data : np.ndarray
        Data to normalize.
    method : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').

    Returns
    -------
    np.ndarray
        Normalized data.
    """
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
        raise ValueError(f"Unknown normalization method: {method}")

def compute_metrics(x: np.ndarray,
                    y: np.ndarray,
                    sizes: np.ndarray,
                    metric_funcs: List[Callable] = None) -> Dict[str, float]:
    """
    Compute metrics for bubble plot.

    Parameters
    ----------
    x : np.ndarray
        X coordinates of bubbles.
    y : np.ndarray
        Y coordinates of bubbles.
    sizes : np.ndarray
        Sizes of bubbles.
    metric_funcs : List[Callable], optional
        Custom metric functions.

    Returns
    -------
    Dict[str, float]
        Dictionary of computed metrics.
    """
    metrics = {}
    if metric_funcs is None:
        metric_funcs = [np.mean, np.std]

    for func in metric_funcs:
        try:
            metrics[func.__name__] = func(sizes)
        except Exception as e:
            metrics[f"error_{func.__name__}"] = str(e)

    return metrics

def bubble_plot_fit(x: np.ndarray,
                    y: np.ndarray,
                    sizes: np.ndarray,
                    normalize_x: bool = True,
                    normalize_y: bool = True,
                    size_normalization: str = 'none',
                    metric_funcs: List[Callable] = None) -> Dict:
    """
    Fit a bubble plot with configurable options.

    Parameters
    ----------
    x : np.ndarray
        X coordinates of bubbles.
    y : np.ndarray
        Y coordinates of bubbles.
    sizes : np.ndarray
        Sizes of bubbles.
    normalize_x : bool, optional
        Whether to normalize x coordinates.
    normalize_y : bool, optional
        Whether to normalize y coordinates.
    size_normalization : str, optional
        Normalization method for sizes ('none', 'standard', 'minmax').
    metric_funcs : List[Callable], optional
        Custom metric functions.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(x, y, sizes, normalize_x, normalize_y, size_normalization)

    # Normalize data
    x_norm = normalize_data(x, 'standard' if normalize_x else 'none')
    y_norm = normalize_data(y, 'standard' if normalize_y else 'none')
    sizes_norm = normalize_data(sizes, size_normalization)

    # Compute metrics
    metrics = compute_metrics(x_norm, y_norm, sizes_norm, metric_funcs)

    # Prepare output
    result = {
        'x': x_norm,
        'y': y_norm,
        'sizes': sizes_norm
    }

    params_used = {
        'normalize_x': normalize_x,
        'normalize_y': normalize_y,
        'size_normalization': size_normalization
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

# Example usage:
"""
x = np.random.rand(10)
y = np.random.rand(10)
sizes = np.random.rand(10) * 10

result = bubble_plot_fit(x, y, sizes)
print(result)
"""

################################################################################
# contour_plot
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def contour_plot_fit(
    X: np.ndarray,
    Y: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    model_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    normalization: str = "none",
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    custom_params: Optional[Dict] = None
) -> Dict:
    """
    Compute and return a contour plot from given data and model.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    Y : np.ndarray
        Target values array of shape (n_samples,).
    x_grid : np.ndarray
        Grid points for x-axis.
    y_grid : np.ndarray
        Grid points for y-axis.
    model_func : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Function that computes the model predictions.
    normalization : str
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric to evaluate the model: "mse", "mae", "r2", or custom callable.
    solver : str
        Solver method: "closed_form", "gradient_descent", or "newton".
    regularization : Optional[str]
        Regularization method: None, "l1", "l2", or "elasticnet".
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    learning_rate : float
        Learning rate for gradient descent.
    custom_params : Optional[Dict]
        Additional parameters for the model.

    Returns
    -------
    Dict
        Dictionary containing:
        - "result": Contour plot data.
        - "metrics": Computed metrics.
        - "params_used": Parameters used in the computation.
        - "warnings": Any warnings encountered.

    Example
    -------
    >>> X = np.random.rand(100, 2)
    >>> Y = np.random.rand(100)
    >>> x_grid = np.linspace(0, 1, 100)
    >>> y_grid = np.linspace(0, 1, 100)
    >>> def model_func(X, params):
    ...     return X @ params
    >>> contour_plot_fit(X, Y, x_grid, y_grid, model_func)
    """
    # Validate inputs
    _validate_inputs(X, Y, x_grid, y_grid)

    # Normalize data if required
    X_norm, Y_norm = _apply_normalization(X, Y, normalization)

    # Prepare model parameters
    params = _initialize_params(X_norm.shape[1], custom_params)

    # Fit the model
    fitted_model = _fit_model(
        X_norm, Y_norm, params, solver, regularization,
        tol, max_iter, learning_rate
    )

    # Compute contour plot data
    contour_data = _compute_contour_data(
        x_grid, y_grid, fitted_model, model_func
    )

    # Compute metrics
    metrics = _compute_metrics(Y_norm, fitted_model.predict(X_norm), metric)

    # Prepare output
    result = {
        "result": contour_data,
        "metrics": metrics,
        "params_used": fitted_model.params,
        "warnings": []
    }

    return result

def _validate_inputs(
    X: np.ndarray,
    Y: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray
) -> None:
    """Validate input arrays."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")
    if len(x_grid.shape) != 1 or len(y_grid.shape) != 1:
        raise ValueError("x_grid and y_grid must be 1D arrays.")
    if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(Y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _apply_normalization(
    X: np.ndarray,
    Y: np.ndarray,
    method: str
) -> tuple:
    """Apply normalization to input data."""
    if method == "standard":
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        Y_norm = (Y - np.mean(Y)) / np.std(Y)
    elif method == "minmax":
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        Y_norm = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
    elif method == "robust":
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        Y_norm = (Y - np.median(Y)) / (np.percentile(Y, 75) - np.percentile(Y, 25))
    else:
        X_norm, Y_norm = X.copy(), Y.copy()
    return X_norm, Y_norm

def _initialize_params(
    n_features: int,
    custom_params: Optional[Dict]
) -> np.ndarray:
    """Initialize model parameters."""
    if custom_params is not None and "params" in custom_params:
        return np.array(custom_params["params"])
    return np.zeros(n_features)

def _fit_model(
    X: np.ndarray,
    Y: np.ndarray,
    params: np.ndarray,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    learning_rate: float
) -> object:
    """Fit the model using the specified solver."""
    if solver == "closed_form":
        return _fit_closed_form(X, Y, params, regularization)
    elif solver == "gradient_descent":
        return _fit_gradient_descent(X, Y, params, regularization, tol, max_iter, learning_rate)
    elif solver == "newton":
        return _fit_newton(X, Y, params, regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _fit_closed_form(
    X: np.ndarray,
    Y: np.ndarray,
    params: np.ndarray,
    regularization: Optional[str]
) -> object:
    """Fit model using closed-form solution."""
    if regularization is None:
        params = np.linalg.pinv(X) @ Y
    elif regularization == "l2":
        params = np.linalg.inv(X.T @ X + 1e-4 * np.eye(X.shape[1])) @ X.T @ Y
    # Add other regularization methods as needed
    return _ModelResult(params)

def _fit_gradient_descent(
    X: np.ndarray,
    Y: np.ndarray,
    params: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    learning_rate: float
) -> object:
    """Fit model using gradient descent."""
    for _ in range(max_iter):
        grad = _compute_gradient(X, Y, params, regularization)
        params -= learning_rate * grad
        if np.linalg.norm(grad) < tol:
            break
    return _ModelResult(params)

def _fit_newton(
    X: np.ndarray,
    Y: np.ndarray,
    params: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> object:
    """Fit model using Newton's method."""
    for _ in range(max_iter):
        grad = _compute_gradient(X, Y, params, regularization)
        hess = _compute_hessian(X, regularization)
        params -= np.linalg.pinv(hess) @ grad
        if np.linalg.norm(grad) < tol:
            break
    return _ModelResult(params)

def _compute_gradient(
    X: np.ndarray,
    Y: np.ndarray,
    params: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Compute gradient of the loss function."""
    residuals = Y - X @ params
    grad = -X.T @ residuals / len(Y)
    if regularization == "l2":
        grad += 1e-4 * params
    # Add other regularization methods as needed
    return grad

def _compute_hessian(
    X: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Compute Hessian matrix."""
    hess = X.T @ X / len(X)
    if regularization == "l2":
        hess += 1e-4 * np.eye(X.shape[1])
    return hess

def _compute_contour_data(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    model: object,
    model_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> np.ndarray:
    """Compute contour plot data."""
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
    Z = model_func(grid_points, model.params).reshape(X_grid.shape)
    return Z

def _compute_metrics(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict:
    """Compute evaluation metrics."""
    if callable(metric):
        return {"custom_metric": metric(Y_true, Y_pred)}
    elif metric == "mse":
        return {"mse": np.mean((Y_true - Y_pred) ** 2)}
    elif metric == "mae":
        return {"mae": np.mean(np.abs(Y_true - Y_pred))}
    elif metric == "r2":
        ss_res = np.sum((Y_true - Y_pred) ** 2)
        ss_tot = np.sum((Y_true - np.mean(Y_true)) ** 2)
        return {"r2": 1 - ss_res / ss_tot}
    else:
        raise ValueError(f"Unknown metric: {metric}")

class _ModelResult:
    """Simple model result container."""
    def __init__(self, params: np.ndarray):
        self.params = params

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return X @ self.params

################################################################################
# stem_plot
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for stem plot."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-dimensional arrays")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("x contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _normalize_data(x: np.ndarray, y: np.ndarray,
                   normalization: str = 'none') -> tuple[np.ndarray, np.ndarray]:
    """Normalize data according to specified method."""
    if normalization == 'none':
        return x, y
    elif normalization == 'standard':
        x_mean = np.mean(x)
        x_std = np.std(x)
        y_mean = np.mean(y)
        y_std = np.std(y)
        return (x - x_mean) / x_std, (y - y_mean) / y_std
    elif normalization == 'minmax':
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        return (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)
    elif normalization == 'robust':
        x_q1, x_q3 = np.percentile(x, [25, 75])
        y_q1, y_q3 = np.percentile(y, [25, 75])
        x_iqr = x_q3 - x_q1
        y_iqr = y_q3 - y_q1
        return (x - x_q1) / x_iqr, (y - y_q1) / y_iqr
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _calculate_metrics(x: np.ndarray, y: np.ndarray,
                      metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Calculate specified metrics between x and y."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(x, y)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def stem_plot_fit(x: np.ndarray, y: np.ndarray,
                 normalization: str = 'none',
                 metrics: Optional[Dict[str, Callable]] = None,
                 **kwargs) -> Dict[str, Any]:
    """
    Create a stem plot with configurable options.

    Parameters:
    -----------
    x : np.ndarray
        1D array of x-coordinates
    y : np.ndarray
        1D array of y-coordinates
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metrics : dict, optional
        Dictionary of metric names and callable functions

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([4, 5, 6])
    >>> stem_plot_fit(x, y)
    """
    # Validate inputs
    _validate_inputs(x, y)

    # Set default metrics if none provided
    if metrics is None:
        metrics = {
            'mse': lambda x, y: np.mean((x - y)**2),
            'mae': lambda x, y: np.mean(np.abs(x - y))
        }

    # Normalize data
    x_norm, y_norm = _normalize_data(x, y, normalization)

    # Calculate metrics
    calculated_metrics = _calculate_metrics(x_norm, y_norm, metrics)

    # Prepare result dictionary
    result = {
        'result': {
            'x_normalized': x_norm,
            'y_normalized': y_norm
        },
        'metrics': calculated_metrics,
        'params_used': {
            'normalization': normalization
        },
        'warnings': []
    }

    return result

# Example metric functions that could be provided by the user
def mse_metric(x: np.ndarray, y: np.ndarray) -> float:
    """Mean Squared Error metric."""
    return np.mean((x - y)**2)

def mae_metric(x: np.ndarray, y: np.ndarray) -> float:
    """Mean Absolute Error metric."""
    return np.mean(np.abs(x - y))

def r2_metric(x: np.ndarray, y: np.ndarray) -> float:
    """R-squared metric."""
    ss_res = np.sum((x - y)**2)
    ss_tot = np.sum((x - np.mean(x))**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

################################################################################
# step_plot
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def step_plot_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    normalization: str = "none",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a step plot model to 2D data with configurable parameters.

    Parameters
    ----------
    x : np.ndarray
        Input feature values (1D array).
    y : np.ndarray
        Target values (1D array).
    normalization : str, optional
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : str or callable, optional
        Metric to optimize: "mse", "mae", "r2", or custom callable.
    distance : str, optional
        Distance metric: "euclidean", "manhattan", "cosine", or "minkowski".
    solver : str, optional
        Solver method: "closed_form", "gradient_descent", "newton", or "coordinate_descent".
    regularization : str, optional
        Regularization type: None, "l1", "l2", or "elasticnet".
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    learning_rate : float, optional
        Learning rate for gradient-based solvers.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    custom_distance : callable, optional
        Custom distance function if not using built-in distances.

    Returns
    -------
    Dict
        Dictionary containing:
        - "result": Fitted model parameters.
        - "metrics": Computed metrics.
        - "params_used": Parameters used during fitting.
        - "warnings": Any warnings generated.

    Example
    -------
    >>> x = np.array([1, 2, 3, 4])
    >>> y = np.array([2, 3, 5, 8])
    >>> result = step_plot_fit(x, y)
    """
    # Validate inputs
    _validate_inputs(x, y)

    # Normalize data if required
    x_norm, y_norm = _apply_normalization(x, y, normalization)

    # Prepare solver parameters
    solver_params = {
        "tol": tol,
        "max_iter": max_iter,
        "learning_rate": learning_rate
    }

    # Choose solver and fit model
    if solver == "closed_form":
        params = _solve_closed_form(x_norm, y_norm)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(
            x_norm, y_norm,
            metric=metric,
            distance=distance,
            regularization=regularization,
            **solver_params
        )
    elif solver == "newton":
        params = _solve_newton(
            x_norm, y_norm,
            metric=metric,
            distance=distance,
            regularization=regularization,
            **solver_params
        )
    elif solver == "coordinate_descent":
        params = _solve_coordinate_descent(
            x_norm, y_norm,
            metric=metric,
            distance=distance,
            regularization=regularization,
            **solver_params
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(
        x_norm, y_norm,
        params,
        metric=metric,
        custom_metric=custom_metric
    )

    # Prepare output
    result = {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            **solver_params
        },
        "warnings": []
    }

    return result

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays contain NaN values")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays contain infinite values")

def _apply_normalization(
    x: np.ndarray,
    y: np.ndarray,
    method: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply normalization to input arrays."""
    if method == "none":
        return x.copy(), y.copy()
    elif method == "standard":
        x_mean, x_std = np.mean(x), np.std(x)
        y_mean, y_std = np.mean(y), np.std(y)
        return (x - x_mean) / x_std, (y - y_mean) / y_std
    elif method == "minmax":
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        return (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)
    elif method == "robust":
        x_q1, x_q3 = np.percentile(x, [25, 75])
        y_q1, y_q3 = np.percentile(y, [25, 75])
        x_iqr = x_q3 - x_q1
        y_iqr = y_q3 - y_q1
        return (x - x_q1) / x_iqr, (y - y_q1) / y_iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _solve_closed_form(x: np.ndarray, y: np.ndarray) -> Dict:
    """Solve step plot using closed form solution."""
    # This is a placeholder for the actual implementation
    return {"step_points": np.unique(x), "step_values": np.interp(np.unique(x), x, y)}

def _solve_gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    *,
    metric: str = "mse",
    distance: str = "euclidean",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01
) -> Dict:
    """Solve step plot using gradient descent."""
    # This is a placeholder for the actual implementation
    return {"step_points": np.unique(x), "step_values": np.interp(np.unique(x), x, y)}

def _solve_newton(
    x: np.ndarray,
    y: np.ndarray,
    *,
    metric: str = "mse",
    distance: str = "euclidean",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict:
    """Solve step plot using Newton's method."""
    # This is a placeholder for the actual implementation
    return {"step_points": np.unique(x), "step_values": np.interp(np.unique(x), x, y)}

def _solve_coordinate_descent(
    x: np.ndarray,
    y: np.ndarray,
    *,
    metric: str = "mse",
    distance: str = "euclidean",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict:
    """Solve step plot using coordinate descent."""
    # This is a placeholder for the actual implementation
    return {"step_points": np.unique(x), "step_values": np.interp(np.unique(x), x, y)}

def _compute_metrics(
    x: np.ndarray,
    y: np.ndarray,
    params: Dict,
    *,
    metric: Union[str, Callable] = "mse",
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute metrics for the fitted model."""
    y_pred = np.interp(x, params["step_points"], params["step_values"])

    metrics = {}

    if metric == "mse" or (custom_metric is None and metric == "mse"):
        metrics["mse"] = np.mean((y - y_pred) ** 2)
    elif metric == "mae" or (custom_metric is None and metric == "mae"):
        metrics["mae"] = np.mean(np.abs(y - y_pred))
    elif metric == "r2" or (custom_metric is None and metric == "r2"):
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics["r2"] = 1 - (ss_res / ss_tot)

    if custom_metric is not None:
        metrics["custom"] = custom_metric(y, y_pred)

    return metrics

################################################################################
# polar_plot
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def _validate_inputs(
    theta: np.ndarray,
    r: np.ndarray,
    normalization: str = "none",
    metric: Union[str, Callable] = "euclidean"
) -> None:
    """
    Validate input arrays and parameters for polar plot.

    Parameters
    ----------
    theta : np.ndarray
        Array of angles in radians.
    r : np.ndarray
        Array of radii.
    normalization : str, optional
        Normalization method (none, standard, minmax, robust).
    metric : str or callable, optional
        Distance metric to use (euclidean, manhattan, cosine, minkowski, or custom callable).

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if theta.ndim != 1 or r.ndim != 1:
        raise ValueError("theta and r must be 1-dimensional arrays.")
    if len(theta) != len(r):
        raise ValueError("theta and r must have the same length.")
    if np.any(np.isnan(theta)) or np.any(np.isnan(r)):
        raise ValueError("theta and r must not contain NaN values.")
    if np.any(np.isinf(theta)) or np.any(np.isinf(r)):
        raise ValueError("theta and r must not contain infinite values.")

    if normalization not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("Invalid normalization method.")

    if isinstance(metric, str) and metric not in ["euclidean", "manhattan", "cosine", "minkowski"]:
        raise ValueError("Invalid metric string.")
    elif callable(metric):
        try:
            # Test the callable with dummy data
            metric(np.array([0, 1]), np.array([0, 1]))
        except Exception as e:
            raise ValueError(f"Custom metric callable failed: {str(e)}")

def _apply_normalization(
    r: np.ndarray,
    method: str = "none"
) -> np.ndarray:
    """
    Apply normalization to radii.

    Parameters
    ----------
    r : np.ndarray
        Array of radii.
    method : str, optional
        Normalization method (none, standard, minmax, robust).

    Returns
    ------
    np.ndarray
        Normalized radii.
    """
    if method == "none":
        return r
    elif method == "standard":
        mean = np.mean(r)
        std = np.std(r)
        return (r - mean) / std
    elif method == "minmax":
        min_val = np.min(r)
        max_val = np.max(r)
        return (r - min_val) / (max_val - min_val + 1e-10)
    elif method == "robust":
        median = np.median(r)
        iqr = np.percentile(r, 75) - np.percentile(r, 25)
        return (r - median) / (iqr + 1e-10)

def _compute_metric(
    theta: np.ndarray,
    r: np.ndarray,
    metric: Union[str, Callable] = "euclidean"
) -> float:
    """
    Compute the specified metric for polar plot.

    Parameters
    ----------
    theta : np.ndarray
        Array of angles in radians.
    r : np.ndarray
        Array of radii.
    metric : str or callable, optional
        Distance metric to use (euclidean, manhattan, cosine, minkowski, or custom callable).

    Returns
    ------
    float
        Computed metric value.
    """
    if isinstance(metric, str):
        if metric == "euclidean":
            return np.sqrt(np.mean((r ** 2)))
        elif metric == "manhattan":
            return np.mean(np.abs(r))
        elif metric == "cosine":
            # Convert polar to cartesian and compute cosine similarity
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        elif metric == "minkowski":
            return np.mean(np.abs(r) ** 2) ** (1/2)
    else:
        # Custom metric
        return metric(theta, r)

def polar_plot_fit(
    theta: np.ndarray,
    r: np.ndarray,
    normalization: str = "none",
    metric: Union[str, Callable] = "euclidean",
    **kwargs
) -> Dict:
    """
    Compute and return polar plot results with configurable parameters.

    Parameters
    ----------
    theta : np.ndarray
        Array of angles in radians.
    r : np.ndarray
        Array of radii.
    normalization : str, optional
        Normalization method (none, standard, minmax, robust).
    metric : str or callable, optional
        Distance metric to use (euclidean, manhattan, cosine, minkowski, or custom callable).
    **kwargs
        Additional keyword arguments for future extensions.

    Returns
    ------
    Dict
        Dictionary containing:
        - "result": Computed result (normalized radii)
        - "metrics": Dictionary of computed metrics
        - "params_used": Parameters used in computation
        - "warnings": List of warnings (if any)
    """
    # Validate inputs
    _validate_inputs(theta, r, normalization, metric)

    # Apply normalization
    normalized_r = _apply_normalization(r, normalization)

    # Compute metrics
    metric_value = _compute_metric(theta, normalized_r, metric)
    metrics = {
        "primary": metric_value,
        "mean_radius": np.mean(normalized_r),
        "std_radius": np.std(normalized_r)
    }

    # Prepare output
    result = {
        "result": normalized_r,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric if isinstance(metric, str) else "custom",
            **kwargs
        },
        "warnings": []
    }

    return result

# Example usage:
"""
theta = np.linspace(0, 2 * np.pi, 100)
r = np.random.rand(100) * 5
result = polar_plot_fit(theta, r, normalization="standard", metric="euclidean")
"""

################################################################################
# quiver_plot
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def validate_inputs(x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
    """
    Validate input arrays for quiver plot.

    Parameters
    ----------
    x, y : np.ndarray
        Coordinates arrays.
    u, v : np.ndarray
        Vector components.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not all(isinstance(arr, np.ndarray) for arr in [x, y, u, v]):
        raise ValueError("All inputs must be numpy arrays")
    if x.shape != y.shape or u.shape != v.shape:
        raise ValueError("x and y must have same shape, u and v must have same shape")
    if x.shape != u.shape:
        raise ValueError("x and y must have same shape as u and v")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("x contains NaN or inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or inf values")
    if np.any(np.isnan(u)) or np.any(np.isinf(u)):
        raise ValueError("u contains NaN or inf values")
    if np.any(np.isnan(v)) or np.any(np.isinf(v)):
        raise ValueError("v contains NaN or inf values")

def normalize_vectors(u: np.ndarray, v: np.ndarray,
                     method: str = 'none',
                     custom_func: Optional[Callable] = None) -> tuple:
    """
    Normalize vector components.

    Parameters
    ----------
    u, v : np.ndarray
        Vector components.
    method : str
        Normalization method ('none', 'standard', 'minmax', 'robust').
    custom_func : callable, optional
        Custom normalization function.

    Returns
    -------
    tuple
        Normalized u and v arrays.
    """
    if custom_func is not None:
        return custom_func(u, v)

    if method == 'none':
        return u, v
    elif method == 'standard':
        mean_u, std_u = np.mean(u), np.std(u)
        mean_v, std_v = np.mean(v), np.std(v)
        u_norm = (u - mean_u) / std_u if std_u != 0 else u
        v_norm = (v - mean_v) / std_v if std_v != 0 else v
        return u_norm, v_norm
    elif method == 'minmax':
        min_u, max_u = np.min(u), np.max(u)
        min_v, max_v = np.min(v), np.max(v)
        u_norm = (u - min_u) / (max_u - min_u + 1e-8)
        v_norm = (v - min_v) / (max_v - min_v + 1e-8)
        return u_norm, v_norm
    elif method == 'robust':
        median_u = np.median(u)
        iqr_u = np.percentile(u, 75) - np.percentile(u, 25)
        median_v = np.median(v)
        iqr_v = np.percentile(v, 75) - np.percentile(v, 25)
        u_norm = (u - median_u) / iqr_u if iqr_u != 0 else u
        v_norm = (v - median_v) / iqr_v if iqr_v != 0 else v
        return u_norm, v_norm
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_metrics(u_true: np.ndarray, v_true: np.ndarray,
                   u_pred: np.ndarray, v_pred: np.ndarray,
                   metric_names: list = ['mse'],
                   custom_metrics: Optional[Dict[str, Callable]] = None) -> Dict[str, float]:
    """
    Compute metrics between true and predicted vectors.

    Parameters
    ----------
    u_true, v_true : np.ndarray
        True vector components.
    u_pred, v_pred : np.ndarray
        Predicted vector components.
    metric_names : list
        List of metrics to compute ('mse', 'mae', 'r2').
    custom_metrics : dict, optional
        Dictionary of custom metrics {name: callable}.

    Returns
    -------
    dict
        Computed metrics.
    """
    metrics = {}

    if 'mse' in metric_names:
        mse_u = np.mean((u_true - u_pred) ** 2)
        mse_v = np.mean((v_true - v_pred) ** 2)
        metrics['mse'] = (mse_u + mse_v) / 2

    if 'mae' in metric_names:
        mae_u = np.mean(np.abs(u_true - u_pred))
        mae_v = np.mean(np.abs(v_true - v_pred))
        metrics['mae'] = (mae_u + mae_v) / 2

    if 'r2' in metric_names:
        ss_res_u = np.sum((u_true - u_pred) ** 2)
        ss_tot_u = np.sum((u_true - np.mean(u_true)) ** 2)
        ss_res_v = np.sum((v_true - v_pred) ** 2)
        ss_tot_v = np.sum((v_true - np.mean(v_true)) ** 2)
        r2_u = 1 - (ss_res_u / ss_tot_u) if ss_tot_u != 0 else 0
        r2_v = 1 - (ss_res_v / ss_tot_v) if ss_tot_v != 0 else 0
        metrics['r2'] = (r2_u + r2_v) / 2

    if custom_metrics is not None:
        for name, func in custom_metrics.items():
            metrics[name] = func(u_true, v_true, u_pred, v_pred)

    return metrics

def quiver_plot_fit(x: np.ndarray, y: np.ndarray,
                   u_true: np.ndarray, v_true: np.ndarray,
                   normalize_method: str = 'none',
                   custom_normalize: Optional[Callable] = None,
                   metrics_names: list = ['mse'],
                   custom_metrics: Optional[Dict[str, Callable]] = None,
                   **kwargs) -> Dict[str, Any]:
    """
    Main function for quiver plot visualization.

    Parameters
    ----------
    x, y : np.ndarray
        Coordinates arrays.
    u_true, v_true : np.ndarray
        True vector components.
    normalize_method : str
        Normalization method for vectors.
    custom_normalize : callable, optional
        Custom normalization function.
    metrics_names : list
        List of metrics to compute.
    custom_metrics : dict, optional
        Dictionary of custom metrics {name: callable}.
    **kwargs :
        Additional parameters for visualization.

    Returns
    -------
    dict
        Dictionary containing results, metrics, and parameters used.
    """
    # Validate inputs
    validate_inputs(x, y, u_true, v_true)

    # Normalize vectors
    u_norm, v_norm = normalize_vectors(u_true, v_true,
                                     method=normalize_method,
                                     custom_func=custom_normalize)

    # Compute metrics
    metrics = compute_metrics(u_true, v_true,
                            u_norm, v_norm,
                            metric_names=metrics_names,
                            custom_metrics=custom_metrics)

    # Prepare output
    result = {
        'x': x,
        'y': y,
        'u': u_norm,
        'v': v_norm
    }

    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalize_method': normalize_method,
            'custom_normalize': custom_normalize is not None,
            'metrics_names': metrics_names
        },
        'warnings': []
    }

    return output

# Example usage:
"""
x = np.array([[0, 1], [2, 3]])
y = np.array([[0, 1], [2, 3]])
u_true = np.array([[1, -1], [-1, 1]])
v_true = np.array([[-1, 1], [1, -1]])

result = quiver_plot_fit(x, y, u_true, v_true,
                       normalize_method='standard',
                       metrics_names=['mse', 'r2'])
"""

################################################################################
# scatter_matrix
################################################################################

import numpy as np
from typing import Callable, Dict, List, Optional, Union

def scatter_matrix_fit(
    data: np.ndarray,
    normalize: str = "none",
    distance_metric: Union[str, Callable] = "euclidean",
    diagonal: str = "histogram",
    figsize: tuple = (10, 10),
    alpha: float = 0.5,
    color_map: Optional[str] = None
) -> Dict:
    """
    Compute and return a scatter matrix plot configuration.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    normalize : str, optional
        Normalization method: "none", "standard", "minmax", or "robust"
    distance_metric : str or callable, optional
        Distance metric for pairwise distances: "euclidean", "manhattan",
        "cosine", or custom callable
    diagonal : str, optional
        Diagonal plot type: "histogram" or "kde"
    figsize : tuple, optional
        Figure size (width, height)
    alpha : float, optional
        Transparency of points in scatter plots
    color_map : str or None, optional
        Color map for the scatter plots

    Returns:
    --------
    dict
        Dictionary containing plot configuration and metadata
    """
    # Validate inputs
    _validate_inputs(data, normalize)

    # Normalize data if required
    normalized_data = _apply_normalization(data, normalize)

    # Compute pairwise distances if needed
    if isinstance(distance_metric, str):
        distance_func = _get_distance_function(distance_metric)
    else:
        distance_func = distance_metric

    # Prepare scatter matrix configuration
    config = _prepare_scatter_matrix_config(
        normalized_data,
        distance_func,
        diagonal,
        figsize,
        alpha,
        color_map
    )

    return config

def _validate_inputs(data: np.ndarray, normalize: str) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional")
    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("Invalid normalization method")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the data."""
    if method == "none":
        return data
    elif method == "standard":
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == "minmax":
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == "robust":
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / iqr

def _get_distance_function(metric: str) -> Callable:
    """Return the appropriate distance function."""
    if metric == "euclidean":
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == "manhattan":
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == "cosine":
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        raise ValueError("Unsupported distance metric")

def _prepare_scatter_matrix_config(
    data: np.ndarray,
    distance_func: Callable,
    diagonal: str,
    figsize: tuple,
    alpha: float,
    color_map: Optional[str]
) -> Dict:
    """Prepare the configuration dictionary for scatter matrix."""
    n_features = data.shape[1]
    config = {
        "result": {
            "data": data,
            "n_features": n_features
        },
        "metrics": {
            "pairwise_distances": np.array([
                [distance_func(data[i], data[j]) for j in range(n_features)]
                for i in range(n_features)
            ])
        },
        "params_used": {
            "normalize": _get_normalization_name(data),
            "distance_metric": distance_func.__name__ if hasattr(distance_func, "__name__") else "custom",
            "diagonal": diagonal,
            "figsize": figsize,
            "alpha": alpha,
            "color_map": color_map
        },
        "warnings": []
    }
    return config

def _get_normalization_name(data: np.ndarray) -> str:
    """Determine the normalization applied to data."""
    if np.allclose(data, _apply_normalization(data, "standard")):
        return "standard"
    elif np.allclose(data, _apply_normalization(data, "minmax")):
        return "minmax"
    elif np.allclose(data, _apply_normalization(data, "robust")):
        return "robust"
    else:
        return "none"

# Example usage
if __name__ == "__main__":
    # Generate some random data
    np.random.seed(42)
    data = np.random.randn(100, 3)

    # Compute scatter matrix configuration
    config = scatter_matrix_fit(
        data=data,
        normalize="standard",
        distance_metric="euclidean",
        diagonal="histogram",
        figsize=(8, 8),
        alpha=0.7,
        color_map="viridis"
    )

    # The config dictionary can now be used to generate the actual plot

################################################################################
# parallel_coordinates
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def parallel_coordinates_fit(
    data: np.ndarray,
    *,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_solver: Optional[Callable] = None
) -> Dict:
    """
    Compute parallel coordinates visualization for high-dimensional data.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom function
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    custom_metric : callable, optional
        Custom metric function if needed
    custom_solver : callable, optional
        Custom solver function if needed

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.random.rand(10, 5)
    >>> result = parallel_coordinates_fit(data, normalize='standard', metric='euclidean')
    """
    # Validate inputs
    _validate_inputs(data, normalize)

    # Normalize data
    normalized_data = _apply_normalization(data, normalize)

    # Prepare parameters dictionary
    params_used = {
        'normalize': normalize,
        'metric': metric if isinstance(metric, str) else 'custom',
        'solver': solver,
        'tol': tol,
        'max_iter': max_iter
    }

    # Compute parallel coordinates visualization
    result = _compute_parallel_coordinates(
        normalized_data,
        metric=metric,
        solver=solver,
        tol=tol,
        max_iter=max_iter,
        custom_metric=custom_metric,
        custom_solver=custom_solver
    )

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, result)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

def _validate_inputs(data: np.ndarray, normalize: str) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

    valid_normalizations = ['none', 'standard', 'minmax', 'robust']
    if normalize not in valid_normalizations:
        raise ValueError(f"Normalization must be one of {valid_normalizations}")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the data."""
    if method == 'none':
        return data.copy()

    normalized = np.empty_like(data, dtype=np.float64)

    if method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / (std + 1e-8)

    elif method == 'minmax':
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        normalized = (data - min_vals) / (max_vals - min_vals + 1e-8)

    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        normalized = (data - median) / (iqr + 1e-8)

    return normalized

def _compute_parallel_coordinates(
    data: np.ndarray,
    *,
    metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_solver: Optional[Callable] = None
) -> Dict:
    """Compute the parallel coordinates visualization."""
    # Select metric function
    distance_func = _get_metric_function(metric, custom_metric)

    # Select solver function
    if custom_solver is not None:
        solver_func = custom_solver
    else:
        solver_func = _get_solver_function(solver)

    # Compute parallel coordinates
    result = solver_func(data, distance_func, tol=tol, max_iter=max_iter)

    return result

def _get_metric_function(
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Callable:
    """Return the appropriate metric function."""
    if custom_metric is not None:
        return custom_metric

    metrics = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': lambda x, y: np.sum(np.abs(x - y)**2, axis=1)**(1/2)
    }

    if isinstance(metric, str):
        return metrics.get(metric.lower(), _euclidean_distance)
    else:
        raise ValueError("Invalid metric specified")

def _get_solver_function(solver: str) -> Callable:
    """Return the appropriate solver function."""
    solvers = {
        'closed_form': _closed_form_solver,
        'gradient_descent': _gradient_descent_solver,
        'newton': _newton_solver,
        'coordinate_descent': _coordinate_descent_solver
    }

    return solvers.get(solver.lower(), _closed_form_solver)

def _calculate_metrics(data: np.ndarray, result: Dict) -> Dict:
    """Calculate various metrics for the visualization."""
    # This is a placeholder - actual metrics would depend on the specific implementation
    return {
        'data_range': np.ptp(data, axis=0),
        'mean_distance': _euclidean_distance(data, result.get('projection', data)).mean()
    }

# Distance functions
def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance between corresponding rows of x and y."""
    return np.linalg.norm(x - y, axis=1)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Manhattan distance between corresponding rows of x and y."""
    return np.sum(np.abs(x - y), axis=1)

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute cosine distance between corresponding rows of x and y."""
    return 1 - np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))

# Solver functions
def _closed_form_solver(data: np.ndarray, distance_func: Callable, **kwargs) -> Dict:
    """Closed form solver for parallel coordinates."""
    # Placeholder implementation
    return {'projection': data.copy()}

def _gradient_descent_solver(data: np.ndarray, distance_func: Callable, **kwargs) -> Dict:
    """Gradient descent solver for parallel coordinates."""
    # Placeholder implementation
    return {'projection': data.copy()}

def _newton_solver(data: np.ndarray, distance_func: Callable, **kwargs) -> Dict:
    """Newton's method solver for parallel coordinates."""
    # Placeholder implementation
    return {'projection': data.copy()}

def _coordinate_descent_solver(data: np.ndarray, distance_func: Callable, **kwargs) -> Dict:
    """Coordinate descent solver for parallel coordinates."""
    # Placeholder implementation
    return {'projection': data.copy()}

################################################################################
# treemap
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def treemap_fit(
    data: np.ndarray,
    value_column: int = -1,
    label_columns: Optional[Union[int, list]] = None,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    max_iter: int = 100,
    tol: float = 1e-4,
    custom_normalization: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Compute a treemap visualization from hierarchical data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array where each row represents an observation.
    value_column : int, optional
        Index of the column containing the values to be visualized (default: -1).
    label_columns : int or list, optional
        Index(es) of the column(s) containing hierarchical labels (default: None).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    metric : str or callable, optional
        Metric to optimize ('mse', 'mae', 'r2') or custom callable (default: 'mse').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent') (default: 'closed_form').
    max_iter : int, optional
        Maximum number of iterations for iterative solvers (default: 100).
    tol : float, optional
        Tolerance for convergence (default: 1e-4).
    custom_normalization : callable, optional
        Custom normalization function (default: None).
    custom_metric : callable, optional
        Custom metric function (default: None).

    Returns:
    --------
    dict
        Dictionary containing the treemap result, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, value_column, label_columns)

    # Normalize data if required
    normalized_data = _apply_normalization(data, value_column, normalization, custom_normalization)

    # Prepare hierarchical structure
    hierarchy = _prepare_hierarchy(normalized_data, value_column, label_columns)

    # Compute treemap
    result = _compute_treemap(hierarchy, metric, solver, max_iter, tol, custom_metric)

    # Calculate metrics
    metrics = _calculate_metrics(result['values'], metric, custom_metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

def _validate_inputs(data: np.ndarray, value_column: int, label_columns: Optional[Union[int, list]]) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if value_column >= data.shape[1] or value_column < -data.shape[1]:
        raise ValueError("Value column index is out of bounds.")
    if label_columns is not None:
        if isinstance(label_columns, int):
            if label_columns >= data.shape[1] or label_columns < -data.shape[1]:
                raise ValueError("Label column index is out of bounds.")
        elif isinstance(label_columns, list):
            for col in label_columns:
                if col >= data.shape[1] or col < -data.shape[1]:
                    raise ValueError("Label column index is out of bounds.")

def _apply_normalization(
    data: np.ndarray,
    value_column: int,
    normalization: str,
    custom_normalization: Optional[Callable]
) -> np.ndarray:
    """Apply normalization to the data."""
    values = data[:, value_column].copy()

    if custom_normalization is not None:
        values = custom_normalization(values)
    elif normalization == 'standard':
        values = (values - np.mean(values)) / np.std(values)
    elif normalization == 'minmax':
        values = (values - np.min(values)) / (np.max(values) - np.min(values))
    elif normalization == 'robust':
        values = (values - np.median(values)) / (np.percentile(values, 75) - np.percentile(values, 25))

    data[:, value_column] = values
    return data

def _prepare_hierarchy(
    data: np.ndarray,
    value_column: int,
    label_columns: Optional[Union[int, list]]
) -> Dict[str, Any]:
    """Prepare hierarchical structure from data."""
    if label_columns is None:
        return {'root': {'value': np.sum(data[:, value_column]), 'children': []}}

    if isinstance(label_columns, int):
        label_columns = [label_columns]

    hierarchy = {}
    for labels in np.unique(data[:, tuple(label_columns)], axis=0):
        mask = np.all(data[:, label_columns] == labels, axis=1)
        group_data = data[mask]
        children = _prepare_hierarchy(group_data, value_column, label_columns[:-1]) if len(label_columns) > 1 else []
        hierarchy[str(labels)] = {
            'value': np.sum(group_data[:, value_column]),
            'children': children
        }

    return {'root': hierarchy}

def _compute_treemap(
    hierarchy: Dict[str, Any],
    metric: Union[str, Callable],
    solver: str,
    max_iter: int,
    tol: float,
    custom_metric: Optional[Callable]
) -> Dict[str, Any]:
    """Compute treemap layout."""
    if solver == 'closed_form':
        return _compute_closed_form_treemap(hierarchy, metric, custom_metric)
    elif solver == 'gradient_descent':
        return _compute_gradient_descent_treemap(hierarchy, metric, max_iter, tol, custom_metric)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _compute_closed_form_treemap(
    hierarchy: Dict[str, Any],
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict[str, Any]:
    """Compute treemap using closed-form solution."""
    # Simplified implementation for demonstration
    total_value = hierarchy['root']['value']
    result = {'values': {}, 'coordinates': {}}

    def _recursive_compute(node, x, y, width, height):
        if not node['children']:
            result['values'][node] = node['value']
            result['coordinates'][node] = {'x': x, 'y': y, 'width': width, 'height': height}
            return

        sorted_children = sorted(node['children'].items(), key=lambda item: item[1]['value'], reverse=True)
        current_x, current_y = x, y

        for child_name, child in sorted_children:
            child_value = child['value']
            child_width = (child_value / total_value) * width
            _recursive_compute(child, current_x, current_y, child_width, height)
            current_x += child_width

    _recursive_compute(hierarchy['root'], 0, 0, 1, 1)
    return result

def _compute_gradient_descent_treemap(
    hierarchy: Dict[str, Any],
    metric: Union[str, Callable],
    max_iter: int,
    tol: float,
    custom_metric: Optional[Callable]
) -> Dict[str, Any]:
    """Compute treemap using gradient descent."""
    # Placeholder for gradient descent implementation
    raise NotImplementedError("Gradient descent solver not implemented yet.")

def _calculate_metrics(
    values: Dict[str, float],
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Calculate metrics for the treemap."""
    if custom_metric is not None:
        return {'custom': custom_metric(values)}

    metrics = {}
    if metric == 'mse':
        # Example calculation
        total_value = sum(values.values())
        mse = np.mean([(v - total_value/len(values))**2 for v in values.values()])
        metrics['mse'] = mse
    elif metric == 'mae':
        # Example calculation
        total_value = sum(values.values())
        mae = np.mean([abs(v - total_value/len(values)) for v in values.values()])
        metrics['mae'] = mae
    elif metric == 'r2':
        # Example calculation
        total_value = sum(values.values())
        ss_total = np.sum([(v - total_value/len(values))**2 for v in values.values()])
        ss_res = np.sum([(v - total_value/len(values))**2 for v in values.values()])
        r2 = 1 - (ss_res / ss_total) if ss_total != 0 else 0
        metrics['r2'] = r2

    return metrics

################################################################################
# word_cloud
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def word_cloud_fit(
    words: np.ndarray,
    frequencies: np.ndarray,
    image_shape: Tuple[int, int],
    normalization: str = "none",
    metric: Union[str, Callable] = "euclidean",
    solver: str = "closed_form",
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None,
    custom_solver: Optional[Callable] = None
) -> Dict:
    """
    Compute a word cloud visualization based on given words and frequencies.

    Parameters
    ----------
    words : np.ndarray
        Array of words to be visualized.
    frequencies : np.ndarray
        Array of corresponding word frequencies.
    image_shape : Tuple[int, int]
        Shape of the output image (height, width).
    normalization : str, optional
        Normalization method for frequencies. Options: "none", "standard", "minmax", "robust".
    metric : Union[str, Callable], optional
        Distance metric to use. Options: "euclidean", "manhattan", "cosine", "minkowski".
    solver : str, optional
        Solver method. Options: "closed_form", "gradient_descent", "newton", "coordinate_descent".
    max_iter : int, optional
        Maximum number of iterations for iterative solvers.
    tol : float, optional
        Tolerance for convergence.
    random_state : Optional[int], optional
        Random seed for reproducibility.
    custom_metric : Optional[Callable], optional
        Custom metric function if not using built-in metrics.
    custom_solver : Optional[Callable], optional
        Custom solver function if not using built-in solvers.

    Returns
    -------
    Dict
        Dictionary containing:
        - "result": Computed word cloud image.
        - "metrics": Dictionary of computed metrics.
        - "params_used": Parameters used in the computation.
        - "warnings": List of warnings encountered.

    Examples
    --------
    >>> words = np.array(["hello", "world", "python"])
    >>> frequencies = np.array([10, 5, 8])
    >>> image_shape = (200, 300)
    >>> result = word_cloud_fit(words, frequencies, image_shape)
    """
    # Validate inputs
    _validate_inputs(words, frequencies, image_shape)

    # Normalize frequencies
    normalized_freqs = _normalize_frequencies(frequencies, normalization)

    # Initialize random state
    rng = np.random.RandomState(random_state)

    # Choose metric function
    if isinstance(metric, str):
        distance_func = _get_metric_function(metric)
    else:
        if custom_metric is None:
            raise ValueError("Custom metric function must be provided.")
        distance_func = custom_metric

    # Choose solver
    if isinstance(solver, str):
        solver_func = _get_solver_function(solver)
    else:
        if custom_solver is None:
            raise ValueError("Custom solver function must be provided.")
        solver_func = custom_solver

    # Compute word cloud
    result_image, params_used = solver_func(
        words,
        normalized_freqs,
        image_shape,
        distance_func,
        max_iter=max_iter,
        tol=tol,
        rng=rng
    )

    # Compute metrics
    metrics = _compute_metrics(result_image, normalized_freqs)

    return {
        "result": result_image,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": []
    }

def _validate_inputs(
    words: np.ndarray,
    frequencies: np.ndarray,
    image_shape: Tuple[int, int]
) -> None:
    """Validate input arrays and parameters."""
    if words.ndim != 1 or frequencies.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional.")
    if len(words) != len(frequencies):
        raise ValueError("Words and frequencies must have the same length.")
    if len(image_shape) != 2 or any(s <= 0 for s in image_shape):
        raise ValueError("Image shape must be a tuple of two positive integers.")
    if np.any(np.isnan(words)) or np.any(np.isinf(words)):
        raise ValueError("Words array contains NaN or infinite values.")
    if np.any(np.isnan(frequencies)) or np.any(np.isinf(frequencies)):
        raise ValueError("Frequencies array contains NaN or infinite values.")

def _normalize_frequencies(
    frequencies: np.ndarray,
    method: str
) -> np.ndarray:
    """Normalize frequencies using the specified method."""
    if method == "none":
        return frequencies
    elif method == "standard":
        return (frequencies - np.mean(frequencies)) / np.std(frequencies)
    elif method == "minmax":
        return (frequencies - np.min(frequencies)) / (np.max(frequencies) - np.min(frequencies))
    elif method == "robust":
        return (frequencies - np.median(frequencies)) / (np.percentile(frequencies, 75) - np.percentile(frequencies, 25))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _get_metric_function(metric_name: str) -> Callable:
    """Return the metric function based on the name."""
    metrics = {
        "euclidean": _euclidean_distance,
        "manhattan": _manhattan_distance,
        "cosine": _cosine_distance,
        "minkowski": lambda x, y, p=2: np.sum(np.abs(x - y) ** p) ** (1 / p)
    }
    if metric_name not in metrics:
        raise ValueError(f"Unknown metric: {metric_name}")
    return metrics[metric_name]

def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Euclidean distance between two points."""
    return np.linalg.norm(x - y)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Manhattan distance between two points."""
    return np.sum(np.abs(x - y))

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine distance between two points."""
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def _get_solver_function(solver_name: str) -> Callable:
    """Return the solver function based on the name."""
    solvers = {
        "closed_form": _closed_form_solver,
        "gradient_descent": _gradient_descent_solver,
        "newton": _newton_solver,
        "coordinate_descent": _coordinate_descent_solver
    }
    if solver_name not in solvers:
        raise ValueError(f"Unknown solver: {solver_name}")
    return solvers[solver_name]

def _closed_form_solver(
    words: np.ndarray,
    frequencies: np.ndarray,
    image_shape: Tuple[int, int],
    distance_func: Callable,
    **kwargs
) -> Tuple[np.ndarray, Dict]:
    """Closed form solution for word cloud placement."""
    # Placeholder implementation
    result_image = np.zeros(image_shape)
    params_used = {"solver": "closed_form"}
    return result_image, params_used

def _gradient_descent_solver(
    words: np.ndarray,
    frequencies: np.ndarray,
    image_shape: Tuple[int, int],
    distance_func: Callable,
    max_iter: int = 1000,
    tol: float = 1e-4,
    rng: np.random.RandomState = None
) -> Tuple[np.ndarray, Dict]:
    """Gradient descent solver for word cloud placement."""
    # Placeholder implementation
    result_image = np.zeros(image_shape)
    params_used = {"solver": "gradient_descent", "max_iter": max_iter, "tol": tol}
    return result_image, params_used

def _newton_solver(
    words: np.ndarray,
    frequencies: np.ndarray,
    image_shape: Tuple[int, int],
    distance_func: Callable,
    **kwargs
) -> Tuple[np.ndarray, Dict]:
    """Newton's method solver for word cloud placement."""
    # Placeholder implementation
    result_image = np.zeros(image_shape)
    params_used = {"solver": "newton"}
    return result_image, params_used

def _coordinate_descent_solver(
    words: np.ndarray,
    frequencies: np.ndarray,
    image_shape: Tuple[int, int],
    distance_func: Callable,
    **kwargs
) -> Tuple[np.ndarray, Dict]:
    """Coordinate descent solver for word cloud placement."""
    # Placeholder implementation
    result_image = np.zeros(image_shape)
    params_used = {"solver": "coordinate_descent"}
    return result_image, params_used

def _compute_metrics(
    result_image: np.ndarray,
    frequencies: np.ndarray
) -> Dict:
    """Compute metrics for the word cloud."""
    # Placeholder implementation
    return {
        "coverage": np.sum(result_image) / (result_image.shape[0] * result_image.shape[1]),
        "frequency_preservation": np.corrcoef(frequencies, result_image.ravel())[0, 1]
    }
