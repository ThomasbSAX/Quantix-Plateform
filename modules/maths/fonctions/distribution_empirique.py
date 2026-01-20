"""
Quantix – Module distribution_empirique
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# histogramme
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
    """Compute histogram bins."""
    if isinstance(bins, int):
        if range is None:
            bin_edges = np.histogram_bin_edges(data, bins=bins)
        else:
            bin_edges = np.linspace(range[0], range[1], bins + 1)
    else:
        bin_edges = np.asarray(bins)
        if len(bin_edges) < 2:
            raise ValueError("bins must contain at least two elements")
    return bin_edges

def normalize_counts(counts: np.ndarray, normalization: str = 'none') -> np.ndarray:
    """Normalize histogram counts."""
    if normalization == 'none':
        return counts
    elif normalization == 'standard':
        return (counts - np.mean(counts)) / np.std(counts)
    elif normalization == 'minmax':
        return (counts - np.min(counts)) / (np.max(counts) - np.min(counts))
    elif normalization == 'robust':
        return (counts - np.median(counts)) / (np.percentile(counts, 75) - np.percentile(counts, 25))
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def compute_metrics(counts: np.ndarray, bin_edges: np.ndarray, metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute metrics for histogram."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(counts, bin_edges)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def histogram_fit(
    data: np.ndarray,
    bins: Union[int, np.ndarray] = 10,
    range: Optional[tuple] = None,
    normalization: str = 'none',
    metric_funcs: Dict[str, Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute histogram with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data to compute histogram from.
    bins : int or np.ndarray, optional
        Number of bins or bin edges. Default is 10.
    range : tuple, optional
        Range of the bins. Default is None (automatic).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'). Default is 'none'.
    metric_funcs : Dict[str, Callable], optional
        Dictionary of metric functions to compute. Default is None.

    Returns:
    --------
    dict
        Dictionary containing histogram result, metrics, parameters used and warnings.
    """
    # Validate input data
    validate_input(data)

    # Compute bin edges
    bin_edges = compute_bins(data, bins, range)

    # Compute histogram counts
    counts, _ = np.histogram(data, bins=bin_edges)

    # Normalize counts
    normalized_counts = normalize_counts(counts, normalization)

    # Compute metrics if provided
    metrics = {}
    if metric_funcs is not None:
        metrics = compute_metrics(counts, bin_edges, metric_funcs)

    # Prepare output
    result = {
        'result': {
            'counts': counts,
            'bin_edges': bin_edges,
            'normalized_counts': normalized_counts
        },
        'metrics': metrics,
        'params_used': {
            'bins': bins,
            'range': range,
            'normalization': normalization
        },
        'warnings': []
    }

    return result

# Example usage:
"""
data = np.random.normal(0, 1, 1000)
metrics = {
    'mean': lambda counts, edges: np.mean(counts),
    'std': lambda counts, edges: np.std(counts)
}
result = histogram_fit(data, bins=20, normalization='standard', metric_funcs=metrics)
"""

################################################################################
# density_plot
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def density_plot_fit(
    data: np.ndarray,
    bandwidth: float = 1.0,
    kernel: Callable[[np.ndarray], np.ndarray] = lambda x: (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2),
    normalization: str = 'standard',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_weights: Optional[np.ndarray] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Fit a density plot to empirical data.

    Parameters
    ----------
    data : np.ndarray
        Input data for which the density plot is computed.
    bandwidth : float, optional
        Bandwidth parameter for the kernel density estimation (default is 1.0).
    kernel : Callable[[np.ndarray], np.ndarray], optional
        Kernel function used for density estimation (default is Gaussian kernel).
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust' (default is 'standard').
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'minkowski', or custom callable (default is 'euclidean').
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent' (default is 'closed_form').
    regularization : Optional[str], optional
        Regularization method: None, 'l1', 'l2', or 'elasticnet' (default is None).
    tol : float, optional
        Tolerance for convergence (default is 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default is 1000).
    custom_weights : Optional[np.ndarray], optional
        Custom weights for the data points (default is None).

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, custom_weights)

    # Normalize data if required
    normalized_data = _normalize_data(data, normalization)

    # Compute density plot based on the chosen solver
    if solver == 'closed_form':
        result = _compute_closed_form_density(normalized_data, bandwidth, kernel)
    elif solver == 'gradient_descent':
        result = _compute_gradient_descent_density(normalized_data, bandwidth, kernel, tol, max_iter)
    elif solver == 'newton':
        result = _compute_newton_density(normalized_data, bandwidth, kernel, tol, max_iter)
    elif solver == 'coordinate_descent':
        result = _compute_coordinate_descent_density(normalized_data, bandwidth, kernel, tol, max_iter)
    else:
        raise ValueError("Invalid solver specified.")

    # Compute metrics
    metrics = _compute_metrics(data, result, metric)

    # Prepare output dictionary
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "bandwidth": bandwidth,
            "normalization": normalization,
            "metric": metric if isinstance(metric, str) else "custom",
            "solver": solver,
            "regularization": regularization if regularization is not None else "none",
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return output

def _validate_inputs(data: np.ndarray, custom_weights: Optional[np.ndarray]) -> None:
    """Validate input data and weights."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")
    if custom_weights is not None:
        if not isinstance(custom_weights, np.ndarray):
            raise TypeError("Custom weights must be a numpy array.")
        if len(custom_weights) != len(data):
            raise ValueError("Custom weights must have the same length as data.")
        if np.any(custom_weights < 0):
            raise ValueError("Custom weights must be non-negative.")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data based on the specified method."""
    if method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        raise ValueError("Invalid normalization method specified.")

def _compute_closed_form_density(data: np.ndarray, bandwidth: float, kernel: Callable) -> np.ndarray:
    """Compute density plot using closed-form solution."""
    grid = np.linspace(np.min(data), np.max(data), 100)
    density = np.zeros_like(grid)
    for i, x in enumerate(grid):
        density[i] = np.mean(kernel((data - x) / bandwidth))
    return density

def _compute_gradient_descent_density(data: np.ndarray, bandwidth: float, kernel: Callable,
                                     tol: float, max_iter: int) -> np.ndarray:
    """Compute density plot using gradient descent."""
    grid = np.linspace(np.min(data), np.max(data), 100)
    density = np.ones_like(grid) * 0.5
    for _ in range(max_iter):
        gradient = np.zeros_like(grid)
        for i, x in enumerate(grid):
            gradient[i] = np.mean((data - x) * kernel((data - x) / bandwidth)) / (bandwidth ** 2)
        density -= tol * gradient
        if np.linalg.norm(gradient) < tol:
            break
    return density

def _compute_newton_density(data: np.ndarray, bandwidth: float, kernel: Callable,
                           tol: float, max_iter: int) -> np.ndarray:
    """Compute density plot using Newton's method."""
    grid = np.linspace(np.min(data), np.max(data), 100)
    density = np.ones_like(grid) * 0.5
    for _ in range(max_iter):
        gradient = np.zeros_like(grid)
        hessian = np.zeros((len(grid), len(grid)))
        for i, x in enumerate(grid):
            gradient[i] = np.mean((data - x) * kernel((data - x) / bandwidth)) / (bandwidth ** 2)
            hessian[i, i] = np.mean(kernel((data - x) / bandwidth)) / (bandwidth ** 2)
        update = np.linalg.solve(hessian, gradient)
        density -= update
        if np.linalg.norm(update) < tol:
            break
    return density

def _compute_coordinate_descent_density(data: np.ndarray, bandwidth: float, kernel: Callable,
                                       tol: float, max_iter: int) -> np.ndarray:
    """Compute density plot using coordinate descent."""
    grid = np.linspace(np.min(data), np.max(data), 100)
    density = np.ones_like(grid) * 0.5
    for _ in range(max_iter):
        for i, x in enumerate(grid):
            gradient = np.mean((data - x) * kernel((data - x) / bandwidth)) / (bandwidth ** 2)
            density[i] -= tol * gradient
        if np.linalg.norm(gradient) < tol:
            break
    return density

def _compute_metrics(data: np.ndarray, result: np.ndarray,
                     metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]) -> Dict[str, float]:
    """Compute metrics for the density plot."""
    if isinstance(metric, str):
        if metric == 'euclidean':
            return {'euclidean_distance': np.linalg.norm(data - result)}
        elif metric == 'manhattan':
            return {'manhattan_distance': np.sum(np.abs(data - result))}
        elif metric == 'cosine':
            return {'cosine_similarity': np.dot(data, result) / (np.linalg.norm(data) * np.linalg.norm(result))}
        elif metric == 'minkowski':
            return {'minkowski_distance': np.sum(np.abs(data - result) ** 2) ** (1/2)}
        else:
            raise ValueError("Invalid metric specified.")
    else:
        return {'custom_metric': metric(data, result)}

################################################################################
# ecdf
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for ECDF computation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _compute_ecdf_values(sorted_data: np.ndarray, normalized: bool = True) -> np.ndarray:
    """Compute ECDF values from sorted data."""
    n = len(sorted_data)
    y_values = np.arange(1, n + 1) / n if normalized else np.arange(1, n + 1)
    return y_values

def _apply_normalization(data: np.ndarray, method: str = 'none') -> np.ndarray:
    """Apply normalization to the data."""
    if method == 'standard':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    elif method == 'none':
        return data.copy()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def ecdf_fit(data: np.ndarray,
             normalized: bool = True,
             normalization_method: str = 'none',
             custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None) -> Dict[str, Any]:
    """
    Compute the Empirical Cumulative Distribution Function (ECDF) for given data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    normalized : bool, optional
        Whether to normalize the ECDF values (default is True).
    normalization_method : str, optional
        Normalization method to apply to the data ('none', 'standard', 'minmax', 'robust').
    custom_metric : Callable, optional
        Custom metric function to compute additional metrics.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the ECDF result, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data)

    # Apply normalization if specified
    normalized_data = _apply_normalization(data, normalization_method)

    # Sort the data
    sorted_data = np.sort(normalized_data)

    # Compute ECDF values
    y_values = _compute_ecdf_values(sorted_data, normalized)

    # Prepare metrics
    metrics = {}
    if custom_metric is not None:
        try:
            metrics['custom_metric'] = custom_metric(sorted_data, y_values)
        except Exception as e:
            metrics['custom_metric_error'] = str(e)

    # Prepare output
    result = {
        'x_values': sorted_data,
        'y_values': y_values,
        'result': {
            'x': sorted_data,
            'y': y_values
        },
        'metrics': metrics,
        'params_used': {
            'normalized': normalized,
            'normalization_method': normalization_method
        },
        'warnings': []
    }

    return result

# Example usage:
# ecdf_result = ecdf_fit(np.random.randn(100), normalized=True, normalization_method='standard')

################################################################################
# quantiles
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def quantiles_fit(
    data: np.ndarray,
    probabilities: Union[np.ndarray, float],
    normalization: str = 'none',
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = np.linalg.norm,
    solver: str = 'closed_form',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Compute quantiles from empirical data.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    probabilities : Union[np.ndarray, float]
        Probability or array of probabilities for which to compute quantiles.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    distance_metric : Callable, optional
        Distance metric function.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent').
    custom_metric : Callable, optional
        Custom metric function.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> probabilities = np.array([0.25, 0.5, 0.75])
    >>> result = quantiles_fit(data, probabilities)
    """
    # Validate inputs
    _validate_inputs(data, probabilities)

    # Normalize data if required
    normalized_data = _normalize_data(data, normalization)

    # Compute quantiles based on solver choice
    if solver == 'closed_form':
        quantiles = _compute_quantiles_closed_form(normalized_data, probabilities)
    elif solver == 'gradient_descent':
        quantiles = _compute_quantiles_gradient_descent(
            normalized_data, probabilities, distance_metric, custom_metric, tol, max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(data, quantiles, custom_metric)

    # Prepare output
    result = {
        'result': quantiles,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'distance_metric': distance_metric.__name__ if callable(distance_metric) else str(distance_metric),
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(data: np.ndarray, probabilities: Union[np.ndarray, float]) -> None:
    """Validate input data and probabilities."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if not isinstance(probabilities, (np.ndarray, float)):
        raise TypeError("Probabilities must be a numpy array or float.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Data contains infinite values.")

def _normalize_data(
    data: np.ndarray,
    normalization: str = 'none'
) -> np.ndarray:
    """Normalize data based on specified method."""
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
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_quantiles_closed_form(
    data: np.ndarray,
    probabilities: Union[np.ndarray, float]
) -> np.ndarray:
    """Compute quantiles using closed-form solution."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    if isinstance(probabilities, float):
        probabilities = np.array([probabilities])
    quantiles = np.zeros_like(probabilities)
    for i, p in enumerate(probabilities):
        h = (n - 1) * p
        j = int(h)
        g = h - j
        if j == n - 1:
            quantiles[i] = sorted_data[j]
        else:
            quantiles[i] = (1 - g) * sorted_data[j] + g * sorted_data[j + 1]
    return quantiles

def _compute_quantiles_gradient_descent(
    data: np.ndarray,
    probabilities: Union[np.ndarray, float],
    distance_metric: Callable[[np.ndarray, np.ndarray], float],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Compute quantiles using gradient descent."""
    # Placeholder for gradient descent implementation
    raise NotImplementedError("Gradient descent solver not implemented yet.")

def _compute_metrics(
    data: np.ndarray,
    quantiles: np.ndarray,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for quantile estimation."""
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(data, quantiles)
    return metrics

################################################################################
# boxplot
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def boxplot_fit(
    data: np.ndarray,
    normalize: str = "none",
    metrics: Union[str, Callable] = "default",
    custom_metric: Optional[Callable] = None,
    whisker_method: str = "iqr",
    outlier_threshold: float = 1.5,
    show_median: bool = True,
    show_mean: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Compute boxplot statistics for empirical distribution.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples,)
    normalize : str
        Normalization method: "none", "standard", "minmax", or "robust"
    metrics : str or callable
        Metric to compute: "default" (IQR, median), custom function
    custom_metric : callable, optional
        Custom metric function if metrics="custom"
    whisker_method : str
        Method for whiskers: "iqr" (1.5*IQR), "std" (3*std), or "percentile"
    outlier_threshold : float
        Threshold for outliers (for whisker_method="iqr")
    show_median : bool
        Whether to compute and return median
    show_mean : bool
        Whether to compute and return mean

    Returns
    -------
    dict
        Dictionary containing:
        - "result": boxplot statistics (min, q1, median, q3, max)
        - "metrics": computed metrics
        - "params_used": parameters used in computation
        - "warnings": any warnings generated

    Examples
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = boxplot_fit(data)
    """
    # Validate input
    data = _validate_input(data)

    # Normalize if requested
    normalized_data, norm_params = _apply_normalization(data, normalize)

    # Compute boxplot statistics
    stats = _compute_boxplot_stats(
        normalized_data,
        whisker_method=whisker_method,
        outlier_threshold=outlier_threshold
    )

    # Compute metrics
    metrics_result = _compute_metrics(
        normalized_data,
        stats,
        metrics=metrics,
        custom_metric=custom_metric
    )

    # Prepare output
    result = {
        "result": stats,
        "metrics": metrics_result,
        "params_used": {
            "normalize": normalize,
            "whisker_method": whisker_method,
            "outlier_threshold": outlier_threshold
        },
        "warnings": []
    }

    return result

def _validate_input(data: np.ndarray) -> np.ndarray:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input must be 1-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input contains NaN values")
    return data

def _apply_normalization(
    data: np.ndarray,
    method: str
) -> tuple[np.ndarray, Dict]:
    """Apply normalization to data."""
    norm_params = {}
    if method == "standard":
        mean, std = np.mean(data), np.std(data)
        normalized_data = (data - mean) / std
        norm_params["mean"] = mean
        norm_params["std"] = std
    elif method == "minmax":
        min_val, max_val = np.min(data), np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        norm_params["min"] = min_val
        norm_params["max"] = max_val
    elif method == "robust":
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        normalized_data = (data - q1) / iqr
        norm_params["q1"] = q1
        norm_params["iqr"] = iqr
    else:
        normalized_data = data.copy()
    return normalized_data, norm_params

def _compute_boxplot_stats(
    data: np.ndarray,
    whisker_method: str,
    outlier_threshold: float
) -> Dict[str, Union[float, np.ndarray]]:
    """Compute boxplot statistics."""
    q1 = np.percentile(data, 25)
    median = np.median(data)
    q3 = np.percentile(data, 75)

    if whisker_method == "iqr":
        iqr = q3 - q1
        lower_whisker = q1 - outlier_threshold * iqr
        upper_whisker = q3 + outlier_threshold * iqr
    elif whisker_method == "std":
        std = np.std(data)
        lower_whisker = median - 3 * std
        upper_whisker = median + 3 * std
    else:  # percentile
        lower_whisker = np.percentile(data, 5)
        upper_whisker = np.percentile(data, 95)

    min_val = np.min(data)
    max_val = np.max(data)

    # Handle whiskers beyond data range
    lower_whisker = max(lower_whisker, min_val)
    upper_whisker = min(upper_whisker, max_val)

    # Find outliers
    outliers = data[(data < lower_whisker) | (data > upper_whisker)]

    return {
        "min": min_val,
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": max_val,
        "lower_whisker": lower_whisker,
        "upper_whisker": upper_whisker,
        "outliers": outliers
    }

def _compute_metrics(
    data: np.ndarray,
    stats: Dict,
    metrics: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute metrics for boxplot."""
    metric_results = {}

    if metrics == "default" or custom_metric is None:
        # Default metrics
        metric_results["iqr"] = stats["q3"] - stats["q1"]
        metric_results["median"] = stats["median"]

    if metrics == "custom" and custom_metric is not None:
        try:
            metric_results["custom"] = custom_metric(data)
        except Exception as e:
            raise ValueError(f"Custom metric function failed: {str(e)}")

    return metric_results

################################################################################
# violin_plot
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def violin_plot_fit(
    data: np.ndarray,
    *,
    bandwidth_method: str = 'scott',
    kernel: Callable[[np.ndarray], np.ndarray] = None,
    normalization: str = 'none',
    quantiles: Optional[np.ndarray] = None,
    plot_options: Dict[str, Union[bool, str]] = None
) -> Dict:
    """
    Compute and return the components needed for a violin plot from empirical data.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples,)
    bandwidth_method : str, optional
        Method to calculate the bandwidth ('scott', 'silverman')
    kernel : Callable[[np.ndarray], np.ndarray], optional
        Custom kernel function. If None, uses Gaussian kernel.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax')
    quantiles : np.ndarray, optional
        Quantile values to compute. If None, uses default [0.05, 0.25, 0.5, 0.75, 0.95]
    plot_options : Dict[str, Union[bool, str]], optional
        Dictionary of plot options

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Computed violin plot components
        - 'metrics': Performance metrics
        - 'params_used': Parameters used in computation
        - 'warnings': Any warnings generated

    Examples
    --------
    >>> data = np.random.normal(0, 1, 1000)
    >>> result = violin_plot_fit(data)
    """
    # Validate inputs
    _validate_inputs(data, quantiles)

    # Set default plot options if not provided
    if plot_options is None:
        plot_options = {
            'show_median': True,
            'show_quartiles': True,
            'scale': 'width'
        }

    # Set default quantiles if not provided
    if quantiles is None:
        quantiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])

    # Compute kernel density estimation
    kde = _compute_kde(data, bandwidth_method, kernel)

    # Compute quantiles and statistics
    stats = _compute_violin_stats(data, quantiles)

    # Normalize the data if required
    normalized_data = _normalize_data(data, normalization)

    # Prepare result dictionary
    result = {
        'kde': kde,
        'stats': stats,
        'normalized_data': normalized_data
    }

    # Compute metrics
    metrics = _compute_metrics(data, kde)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'bandwidth_method': bandwidth_method,
            'normalization': normalization,
            'quantiles': quantiles
        },
        'warnings': _check_warnings(data, kde)
    }

def _validate_inputs(
    data: np.ndarray,
    quantiles: Optional[np.ndarray]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Data must be 1-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")

    if quantiles is not None:
        if not isinstance(quantiles, np.ndarray):
            raise TypeError("Quantiles must be a numpy array")
        if not np.all((quantiles >= 0) & (quantiles <= 1)):
            raise ValueError("Quantiles must be between 0 and 1")

def _compute_kde(
    data: np.ndarray,
    bandwidth_method: str,
    kernel: Optional[Callable[[np.ndarray], np.ndarray]]
) -> Dict:
    """Compute kernel density estimation."""
    if kernel is None:
        kernel = _gaussian_kernel

    bandwidth = _compute_bandwidth(data, bandwidth_method)
    grid_points = np.linspace(min(data), max(data), 1000)

    kde_values = np.array([
        _apply_kernel(data, point, bandwidth, kernel)
        for point in grid_points
    ])

    return {
        'grid': grid_points,
        'values': kde_values,
        'bandwidth': bandwidth
    }

def _compute_bandwidth(
    data: np.ndarray,
    method: str
) -> float:
    """Compute bandwidth for KDE."""
    std = np.std(data)
    n = len(data)

    if method == 'scott':
        return std * (4 / (3 * n)) ** (1/5)
    elif method == 'silverman':
        return std * (0.9 / n ** 0.2) ** (1/5)
    else:
        raise ValueError(f"Unknown bandwidth method: {method}")

def _gaussian_kernel(x: np.ndarray) -> np.ndarray:
    """Gaussian kernel function."""
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

def _apply_kernel(
    data: np.ndarray,
    point: float,
    bandwidth: float,
    kernel: Callable[[np.ndarray], np.ndarray]
) -> float:
    """Apply kernel function to data."""
    scaled_data = (data - point) / bandwidth
    return np.mean(kernel(scaled_data)) / bandwidth

def _compute_violin_stats(
    data: np.ndarray,
    quantiles: np.ndarray
) -> Dict:
    """Compute statistics for violin plot."""
    sorted_data = np.sort(data)
    n = len(sorted_data)

    stats = {
        'quantiles': np.percentile(data, quantiles * 100),
        'median': np.median(data),
        'min': min(data),
        'max': max(data)
    }

    return stats

def _normalize_data(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Normalize data according to specified method."""
    if method == 'none':
        return data.copy()

    normalized = data.copy()
    if method == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        normalized = (data - mean) / std
    elif method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        normalized = (data - min_val) / (max_val - min_val)

    return normalized

def _compute_metrics(
    data: np.ndarray,
    kde: Dict
) -> Dict:
    """Compute metrics for the violin plot."""
    grid = kde['grid']
    values = kde['values']

    # Compute integral of KDE (should be close to 1)
    integral = np.trapz(values, grid)

    return {
        'kde_integral': integral,
        'data_range': max(data) - min(data)
    }

def _check_warnings(
    data: np.ndarray,
    kde: Dict
) -> list:
    """Check for potential warnings."""
    warnings = []

    if len(data) < 30:
        warnings.append("Small sample size may affect KDE accuracy")

    if kde['bandwidth'] > (max(data) - min(data)) / 2:
        warnings.append("Large bandwidth may oversmooth the distribution")

    return warnings

################################################################################
# rug_plot
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_input(data: np.ndarray) -> None:
    """Validate input data for rug plot."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array")
    if np.any(np.isnan(data)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input contains infinite values")

def compute_rug_positions(data: np.ndarray, normalization: str = 'none') -> np.ndarray:
    """Compute positions for rug plot based on data normalization."""
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
        raise ValueError(f"Unknown normalization method: {normalization}")

def rug_plot_fit(
    data: np.ndarray,
    normalization: str = 'none',
    height: float = 0.1,
    color: str = 'black',
    linewidth: float = 1.0
) -> Dict[str, Any]:
    """
    Create a rug plot for empirical distribution visualization.

    Parameters:
    -----------
    data : np.ndarray
        Input data to visualize.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    height : float, optional
        Height of the rug lines.
    color : str, optional
        Color of the rug lines.
    linewidth : float, optional
        Line width of the rug lines.

    Returns:
    --------
    dict
        Dictionary containing plot parameters and warnings.
    """
    validate_input(data)
    positions = compute_rug_positions(data, normalization)

    result = {
        'positions': positions,
        'height': height,
        'color': color,
        'linewidth': linewidth
    }

    warnings = []
    if normalization not in ['none', 'standard', 'minmax', 'robust']:
        warnings.append(f"Unknown normalization method: {normalization}")

    return {
        'result': result,
        'metrics': {},
        'params_used': {
            'normalization': normalization,
            'height': height,
            'color': color,
            'linewidth': linewidth
        },
        'warnings': warnings
    }

# Example usage:
# data = np.random.normal(0, 1, 100)
# plot_params = rug_plot_fit(data, normalization='standard')

################################################################################
# kernel_density_estimation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def kernel_density_estimation_fit(
    data: np.ndarray,
    bandwidth: float = 1.0,
    kernel: Callable[[np.ndarray], np.ndarray] = lambda x: (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2),
    normalization: str = 'none',
    grid_points: int = 100,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Estimate the probability density function of a sample using kernel density estimation.

    Parameters
    ----------
    data : np.ndarray
        Input data points.
    bandwidth : float, optional
        Bandwidth of the kernel. Default is 1.0.
    kernel : callable, optional
        Kernel function. Default is Gaussian kernel.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'). Default is 'none'.
    grid_points : int, optional
        Number of points in the evaluation grid. Default is 100.
    range_min : float, optional
        Minimum value of the evaluation grid. If None, use data min.
    range_max : float, optional
        Maximum value of the evaluation grid. If None, use data max.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Estimated density values.
        - 'metrics': Metrics dictionary (empty for now).
        - 'params_used': Parameters used in the estimation.
        - 'warnings': Warnings dictionary (empty for now).

    Examples
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = kernel_density_estimation_fit(data)
    """
    # Validate inputs
    _validate_inputs(data, bandwidth, kernel, normalization)

    # Set evaluation grid
    if range_min is None:
        range_min = np.min(data)
    if range_max is None:
        range_max = np.max(data)
    grid = np.linspace(range_min, range_max, grid_points)

    # Compute density
    density = _compute_density(data, bandwidth, kernel, grid)

    # Normalize if needed
    density = _apply_normalization(density, normalization)

    return {
        'result': density,
        'metrics': {},
        'params_used': {
            'bandwidth': bandwidth,
            'kernel': kernel.__name__ if hasattr(kernel, '__name__') else 'custom',
            'normalization': normalization,
            'grid_points': grid_points
        },
        'warnings': {}
    }

def _validate_inputs(
    data: np.ndarray,
    bandwidth: float,
    kernel: Callable[[np.ndarray], np.ndarray],
    normalization: str
) -> None:
    """Validate input parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if len(data.shape) != 1:
        raise ValueError("Data must be a 1-dimensional array.")
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive.")
    if normalization not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("Normalization must be one of: 'none', 'standard', 'minmax', 'robust'.")

def _compute_density(
    data: np.ndarray,
    bandwidth: float,
    kernel: Callable[[np.ndarray], np.ndarray],
    grid: np.ndarray
) -> np.ndarray:
    """Compute the kernel density estimate."""
    density = np.zeros_like(grid)
    for i, x in enumerate(grid):
        distances = (data - x) / bandwidth
        density[i] = np.sum(kernel(distances)) / (len(data) * bandwidth)
    return density

def _apply_normalization(
    density: np.ndarray,
    normalization: str
) -> np.ndarray:
    """Apply normalization to the density estimate."""
    if normalization == 'standard':
        return (density - np.mean(density)) / np.std(density)
    elif normalization == 'minmax':
        return (density - np.min(density)) / (np.max(density) - np.min(density))
    elif normalization == 'robust':
        median = np.median(density)
        iqr = np.percentile(density, 75) - np.percentile(density, 25)
        return (density - median) / iqr
    else:
        return density

################################################################################
# empirical_cdf
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for empirical CDF calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data must not contain NaN or infinite values")

def _compute_empirical_cdf(data: np.ndarray, normalized: bool = True) -> np.ndarray:
    """Compute the empirical CDF from data."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    cdf_values = np.arange(1, n + 1) / n if normalized else np.arange(n)
    return sorted_data, cdf_values

def _apply_normalization(data: np.ndarray, method: str = 'none') -> np.ndarray:
    """Apply normalization to the data."""
    if method == 'standard':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    elif method == 'none':
        return data
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def empirical_cdf_fit(
    data: np.ndarray,
    normalized: bool = True,
    normalization_method: str = 'none',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the empirical CDF of given data.

    Parameters:
    -----------
    data : np.ndarray
        Input data for which to compute the empirical CDF.
    normalized : bool, optional
        Whether to normalize the CDF values (default is True).
    normalization_method : str, optional
        Method for normalizing the input data ('none', 'standard', 'minmax', 'robust').
    custom_metric : Callable, optional
        Custom metric function to evaluate the CDF.

    Returns:
    --------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_input(data)

    # Apply normalization
    normalized_data = _apply_normalization(data, normalization_method)

    # Compute empirical CDF
    x_values, y_values = _compute_empirical_cdf(normalized_data, normalized)

    # Compute metrics
    metrics = {}
    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(x_values, y_values)

    # Prepare output
    result = {
        'x_values': x_values,
        'y_values': y_values
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalized': normalized,
            'normalization_method': normalization_method
        },
        'warnings': []
    }

# Example usage:
# data = np.random.randn(100)
# result = empirical_cdf_fit(data, normalized=True, normalization_method='standard')

################################################################################
# frequency_table
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def frequency_table_fit(
    data: np.ndarray,
    bins: Union[int, np.ndarray, Callable],
    normalization: str = 'count',
    metric: Optional[Union[str, Callable]] = None,
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_weights: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute the frequency table for empirical distribution.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    bins : Union[int, np.ndarray, Callable]
        Number of bins or bin edges or callable for custom binning.
    normalization : str, optional
        Normalization method ('count', 'density', 'percent').
    metric : Optional[Union[str, Callable]], optional
        Metric for evaluation ('mse', 'mae', 'r2') or custom callable.
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent').
    regularization : Optional[str], optional
        Regularization type ('none', 'l1', 'l2').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_weights : Optional[np.ndarray], optional
        Custom weights for data points.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(data, bins, custom_weights)

    # Compute frequency table
    freq_table = _compute_frequency_table(data, bins, normalization)

    # Compute metrics if specified
    metrics = {}
    if metric is not None:
        metrics = _compute_metrics(freq_table, data, metric, distance)

    # Estimate parameters if needed
    params_used = _estimate_parameters(freq_table, solver, regularization, tol, max_iter)

    # Prepare output
    result = {
        'frequency_table': freq_table,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    bins: Union[int, np.ndarray, Callable],
    custom_weights: Optional[np.ndarray]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if custom_weights is not None and len(data) != len(custom_weights):
        raise ValueError("Data and custom weights must have the same length.")
    if isinstance(bins, int) and bins <= 0:
        raise ValueError("Number of bins must be positive.")
    if isinstance(bins, np.ndarray) and len(bins) < 2:
        raise ValueError("Bin edges must have at least two elements.")

def _compute_frequency_table(
    data: np.ndarray,
    bins: Union[int, np.ndarray, Callable],
    normalization: str
) -> np.ndarray:
    """Compute the frequency table."""
    if callable(bins):
        bin_edges = bins(data)
    elif isinstance(bins, int):
        bin_edges = np.linspace(np.min(data), np.max(data), bins + 1)
    else:
        bin_edges = bins

    counts, _ = np.histogram(data, bins=bin_edges)

    if normalization == 'density':
        counts = counts / (np.sum(counts) * np.diff(bin_edges).mean())
    elif normalization == 'percent':
        counts = counts / np.sum(counts) * 100

    return counts

def _compute_metrics(
    freq_table: np.ndarray,
    data: np.ndarray,
    metric: Union[str, Callable],
    distance: str
) -> Dict:
    """Compute metrics for the frequency table."""
    metrics = {}

    if metric == 'mse':
        # Mean Squared Error
        pass  # Implement MSE calculation
    elif metric == 'mae':
        # Mean Absolute Error
        pass  # Implement MAE calculation
    elif metric == 'r2':
        # R-squared
        pass  # Implement R2 calculation
    elif callable(metric):
        metrics['custom'] = metric(freq_table, data)

    return metrics

def _estimate_parameters(
    freq_table: np.ndarray,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict:
    """Estimate parameters using the specified solver."""
    params_used = {
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    if solver == 'closed_form':
        # Closed form solution
        pass  # Implement closed form estimation
    elif solver == 'gradient_descent':
        # Gradient descent
        pass  # Implement gradient descent

    return params_used

# Example usage:
# data = np.random.randn(100)
# result = frequency_table_fit(data, bins=10, normalization='density', metric='mse')

################################################################################
# cumulative_frequency
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(data: np.ndarray) -> None:
    """Validate input data for cumulative frequency calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def compute_bins(data: np.ndarray, bins: Union[int, np.ndarray]) -> tuple:
    """Compute histogram bins for cumulative frequency."""
    if isinstance(bins, int):
        return np.histogram_bin_edges(data, bins=bins)
    elif isinstance(bins, np.ndarray):
        if len(bins) < 2:
            raise ValueError("Custom bins must have at least two elements")
        return np.sort(bins)
    else:
        raise TypeError("Bins must be either an integer or a numpy array")

def cumulative_frequency_compute(
    data: np.ndarray,
    bins: Union[int, np.ndarray] = 10,
    normed: bool = True,
    density: bool = False,
    cumulative: bool = True
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Compute cumulative frequency distribution.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    bins : Union[int, np.ndarray], optional
        Number of bins or bin edges. Default is 10.
    normed : bool, optional
        Whether to normalize the histogram. Default is True.
    density : bool, optional
        Whether to return probability density instead of counts. Default is False.
    cumulative : bool, optional
        Whether to return cumulative values. Default is True.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': numpy array of cumulative frequencies
        - 'metrics': dictionary with additional metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings

    Example
    -------
    >>> data = np.random.randn(100)
    >>> result = cumulative_frequency_compute(data, bins=5)
    """
    # Validate input
    validate_input(data)

    # Compute histogram
    hist, bin_edges = np.histogram(
        data,
        bins=compute_bins(data, bins),
        normed=normed,
        density=density
    )

    # Compute cumulative frequency
    if cumulative:
        result = np.cumsum(hist)
    else:
        result = hist

    # Prepare output
    output = {
        "result": result,
        "metrics": {
            "bin_edges": bin_edges,
            "total_count": np.sum(hist),
            "mean": np.mean(data)
        },
        "params_used": {
            "bins": bins,
            "normed": normed,
            "density": density,
            "cumulative": cumulative
        },
        "warnings": []
    }

    return output

def _cumulative_frequency_standardize(
    data: np.ndarray,
    method: str = "none"
) -> np.ndarray:
    """Standardize data using specified method."""
    if method == "none":
        return data
    elif method == "standard":
        return (data - np.mean(data)) / np.std(data)
    elif method == "minmax":
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == "robust":
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown standardization method: {method}")

def cumulative_frequency_fit(
    data: np.ndarray,
    bins: Union[int, np.ndarray] = 10,
    normed: bool = True,
    density: bool = False,
    cumulative: bool = True,
    standardization: str = "none"
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Fit cumulative frequency distribution with optional standardization.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    bins : Union[int, np.ndarray], optional
        Number of bins or bin edges. Default is 10.
    normed : bool, optional
        Whether to normalize the histogram. Default is True.
    density : bool, optional
        Whether to return probability density instead of counts. Default is False.
    cumulative : bool, optional
        Whether to return cumulative values. Default is True.
    standardization : str, optional
        Standardization method ("none", "standard", "minmax", "robust"). Default is "none".

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': numpy array of cumulative frequencies
        - 'metrics': dictionary with additional metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings

    Example
    -------
    >>> data = np.random.randn(100)
    >>> result = cumulative_frequency_fit(data, bins=5, standardization="standard")
    """
    # Standardize data if requested
    standardized_data = _cumulative_frequency_standardize(data, standardization)

    # Compute cumulative frequency
    result = cumulative_frequency_compute(
        standardized_data,
        bins=bins,
        normed=normed,
        density=density,
        cumulative=cumulative
    )

    # Update metrics with standardization info if applied
    if standardization != "none":
        result["metrics"]["standardization"] = {
            "method": standardization,
            "mean_before": np.mean(data),
            "std_before": np.std(data),
            "mean_after": np.mean(standardized_data),
            "std_after": np.std(standardized_data)
        }

    # Update params_used
    result["params_used"]["standardization"] = standardization

    return result

################################################################################
# relative_frequency
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(data: np.ndarray) -> None:
    """Validate input data for relative frequency computation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def normalize_data(data: np.ndarray, method: str = 'none') -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_bins(data: np.ndarray, bins: Union[int, np.ndarray]) -> tuple:
    """Compute bin edges and indices for data."""
    if isinstance(bins, int):
        return np.histogram_bin_edges(data, bins=bins), None
    elif isinstance(bins, np.ndarray):
        if not np.all(np.diff(bins) > 0):
            raise ValueError("Bin edges must be strictly increasing")
        return bins, None
    else:
        raise TypeError("Bins must be either an integer or a numpy array")

def relative_frequency_compute(
    data: np.ndarray,
    bins: Union[int, np.ndarray],
    normalization: str = 'none',
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, float, dict]]:
    """
    Compute relative frequency distribution of empirical data.

    Parameters:
    -----------
    data : np.ndarray
        Input data to compute relative frequency for.
    bins : int or np.ndarray
        Number of bins or bin edges.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : Callable, optional
        Custom metric function to evaluate the distribution.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    validate_inputs(data)

    # Normalize data
    normalized_data = normalize_data(data, normalization)
    params_used = {'normalization': normalization}

    # Compute bins
    bin_edges, _ = compute_bins(normalized_data, bins)
    params_used['bins'] = bin_edges

    # Compute histogram
    counts, _ = np.histogram(normalized_data, bins=bin_edges)
    relative_freq = counts / np.sum(counts)

    # Compute metrics
    metrics = {}
    if metric is not None:
        try:
            metrics['custom_metric'] = metric(relative_freq, normalized_data)
        except Exception as e:
            metrics['custom_metric_error'] = str(e)

    # Prepare output
    result = {
        'relative_frequency': relative_freq,
        'bin_edges': bin_edges
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

def relative_frequency_fit(
    data: np.ndarray,
    bins: Union[int, np.ndarray],
    normalization: str = 'none',
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, float, dict]]:
    """
    Fit relative frequency distribution to empirical data.

    Parameters:
    -----------
    data : np.ndarray
        Input data to fit relative frequency distribution.
    bins : int or np.ndarray
        Number of bins or bin edges for the distribution.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : Callable, optional
        Custom metric function to evaluate the fit.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings.
    """
    return relative_frequency_compute(data, bins, normalization, metric, **kwargs)

################################################################################
# percentile
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def percentile_fit(
    data: np.ndarray,
    percentiles: Union[float, list],
    normalization: str = 'none',
    interpolation_method: str = 'linear',
    custom_normalization: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Calculate percentiles from empirical data with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    percentiles : Union[float, list]
        Percentile(s) to compute (0-100).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    interpolation_method : str, optional
        Interpolation method ('linear', 'nearest').
    custom_normalization : Callable, optional
        Custom normalization function.
    **kwargs :
        Additional parameters for specific methods.

    Returns:
    --------
    Dict containing:
        - 'result': computed percentiles
        - 'metrics': normalization metrics if applicable
        - 'params_used': parameters used in computation
        - 'warnings': potential issues encountered

    Example:
    --------
    >>> percentile_fit(np.random.randn(100), 95)
    """
    # Input validation
    data = _validate_input(data)

    # Normalization
    normalized_data, metrics = _apply_normalization(
        data,
        normalization=normalization,
        custom_func=custom_normalization
    )

    # Calculate percentiles
    result = _calculate_percentiles(
        normalized_data,
        percentiles=percentiles,
        method=interpolation_method
    )

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'interpolation_method': interpolation_method
        },
        'warnings': _check_warnings(data, result)
    }

def _validate_input(data: np.ndarray) -> np.ndarray:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input must be 1-dimensional")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input contains NaN or infinite values")
    return data

def _apply_normalization(
    data: np.ndarray,
    normalization: str = 'none',
    custom_func: Optional[Callable] = None
) -> tuple:
    """Apply selected normalization."""
    metrics = {}

    if custom_func is not None:
        normalized_data = custom_func(data)
    elif normalization == 'standard':
        mean, std = np.mean(data), np.std(data)
        normalized_data = (data - mean) / std
        metrics.update({'mean': mean, 'std': std})
    elif normalization == 'minmax':
        min_val, max_val = np.min(data), np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        metrics.update({'min': min_val, 'max': max_val})
    elif normalization == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        normalized_data = (data - median) / iqr
        metrics.update({'median': median, 'iqr': iqr})
    else:
        normalized_data = data.copy()

    return normalized_data, metrics

def _calculate_percentiles(
    data: np.ndarray,
    percentiles: Union[float, list],
    method: str = 'linear'
) -> np.ndarray:
    """Calculate percentiles using specified interpolation."""
    if isinstance(percentiles, (int, float)):
        percentiles = [percentiles]

    return np.percentile(data, percentiles, method=method)

def _check_warnings(
    data: np.ndarray,
    result: np.ndarray
) -> list:
    """Check for potential issues."""
    warnings = []

    if len(data) < 10:
        warnings.append("Small sample size may affect percentile accuracy")

    if np.any(np.isnan(result)):
        warnings.append("NaN values in result - check input data")

    return warnings

################################################################################
# quartiles
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for quartiles computation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values.")

def _compute_quartiles(
    data: np.ndarray,
    method: str = 'linear',
    interpolation: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute quartiles of empirical data.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    method : str, optional
        Method for quartile computation ('linear', 'nearest').
    interpolation : Callable, optional
        Custom interpolation function.

    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        Dictionary containing quartiles (Q1, median, Q3).
    """
    if interpolation is not None:
        data = interpolation(data)

    sorted_data = np.sort(data)
    n = len(sorted_data)
    positions = [0.25, 0.5, 0.75]

    quartiles = {}
    for pos in positions:
        if method == 'linear':
            k = (n - 1) * pos
            f = np.floor(k)
            c = np.ceil(k)
            d0 = sorted_data[int(f)] * (c - k)
            d1 = sorted_data[int(c)] * (k - f)
            quartiles[f'Q{int(pos*100)}'] = d0 + d1
        elif method == 'nearest':
            k = (n - 1) * pos
            f = np.floor(k)
            c = np.ceil(k)
            d0 = sorted_data[int(f)] * (c - k)
            d1 = sorted_data[int(c)] * (k - f)
            quartiles[f'Q{int(pos*100)}'] = d0 + d1 if (k - f) > 0.5 else sorted_data[int(f)]
        else:
            raise ValueError("Invalid method specified.")

    return quartiles

def _compute_metrics(
    data: np.ndarray,
    quartiles: Dict[str, float],
    metric_funcs: Optional[Dict[str, Callable]] = None
) -> Dict[str, float]:
    """
    Compute metrics based on quartiles.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    quartiles : Dict[str, float]
        Dictionary of computed quartiles.
    metric_funcs : Dict[str, Callable], optional
        Custom metric functions.

    Returns
    -------
    Dict[str, float]
        Dictionary of computed metrics.
    """
    if metric_funcs is None:
        metric_funcs = {}

    metrics = {}
    for name, func in metric_funcs.items():
        if name == 'iqr':
            metrics[name] = quartiles['Q75'] - quartiles['Q25']
        else:
            metrics[name] = func(data, quartiles)

    return metrics

def quartiles_fit(
    data: np.ndarray,
    method: str = 'linear',
    interpolation: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric_funcs: Optional[Dict[str, Callable]] = None
) -> Dict[str, Union[Dict[str, float], Dict[str, float], Dict[str, str], list]]:
    """
    Compute quartiles and metrics for empirical data.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    method : str, optional
        Method for quartile computation ('linear', 'nearest').
    interpolation : Callable, optional
        Custom interpolation function.
    metric_funcs : Dict[str, Callable], optional
        Custom metric functions.

    Returns
    -------
    Dict[str, Union[Dict[str, float], Dict[str, float], Dict[str, str], list]]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_input(data)

    quartiles = _compute_quartiles(data, method, interpolation)
    metrics = _compute_metrics(data, quartiles, metric_funcs)

    result = {
        'result': quartiles,
        'metrics': metrics,
        'params_used': {
            'method': method,
            'interpolation': interpolation.__name__ if interpolation else None
        },
        'warnings': []
    }

    return result

# Example usage:
# data = np.random.normal(0, 1, 100)
# result = quartiles_fit(data, method='linear')

################################################################################
# interquartile_range
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for interquartile range calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")

def _calculate_quartiles(data: np.ndarray, method: str = 'linear') -> Dict[str, float]:
    """Calculate quartiles using the specified method."""
    if method not in ['linear', 'lower', 'higher', 'nearest']:
        raise ValueError("Method must be one of: 'linear', 'lower', 'higher', 'nearest'")

    sorted_data = np.sort(data)
    n = len(sorted_data)

    def _get_quartile(pos: float) -> float:
        k = (n - 1) * pos
        f = np.floor(k)
        c = np.ceil(k)

        if method == 'linear':
            d0 = sorted_data[int(f)] * (c - k)
            d1 = sorted_data[int(c)] * (k - f)
            return d0 + d1
        elif method == 'lower':
            return sorted_data[int(f)]
        elif method == 'higher':
            return sorted_data[int(c)]
        else:  # nearest
            if k - f < c - k:
                return sorted_data[int(f)]
            else:
                return sorted_data[int(c)]

    q1 = _get_quartile(0.25)
    q3 = _get_quartile(0.75)

    return {'q1': q1, 'q3': q3}

def interquartile_range_compute(
    data: np.ndarray,
    method: str = 'linear',
    custom_quartile_func: Optional[Callable[[np.ndarray], Dict[str, float]]] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Compute the interquartile range (IQR) of a dataset.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    method : str, optional
        Method for quartile calculation ('linear', 'lower', 'higher', 'nearest').
    custom_quartile_func : Callable, optional
        Custom function to calculate quartiles. Must return a dict with 'q1' and 'q3'.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], str]]
        Dictionary containing:
        - 'result': IQR value
        - 'metrics': Quartile values and method used
        - 'params_used': Parameters used in calculation
        - 'warnings': Any warnings generated

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> result = interquartile_range_compute(data)
    """
    _validate_input(data)

    warnings = []

    if custom_quartile_func is not None:
        try:
            quartiles = custom_quartile_func(data)
        except Exception as e:
            warnings.append(f"Custom quartile function failed: {str(e)}")
            quartiles = _calculate_quartiles(data, method)
    else:
        quartiles = _calculate_quartiles(data, method)

    iqr = quartiles['q3'] - quartiles['q1']

    return {
        'result': iqr,
        'metrics': {
            'quartiles': quartiles,
            'method': method if custom_quartile_func is None else 'custom'
        },
        'params_used': {
            'method': method,
            'custom_quartile_func': custom_quartile_func is not None
        },
        'warnings': warnings if warnings else None
    }

################################################################################
# mean
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for mean computation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _compute_mean(
    data: np.ndarray,
    normalization: Optional[str] = None,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute the mean of empirical data with optional normalization.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    normalization : str, optional
        Type of normalization ('none', 'standard', 'minmax', 'robust')
    weights : np.ndarray, optional
        Weights for weighted mean computation

    Returns
    -------
    float
        Computed mean value
    """
    if weights is not None:
        if len(data) != len(weights):
            raise ValueError("Data and weights must have the same length")
        if normalization == 'standard':
            data = (data - np.mean(data)) / np.std(data)
        elif normalization == 'minmax':
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
        elif normalization == 'robust':
            data = (data - np.median(data)) / (np.percentile(data, 75) - np.percentile(data, 25))
        return np.sum(data * weights) / np.sum(weights)
    else:
        if normalization == 'standard':
            data = (data - np.mean(data)) / np.std(data)
        elif normalization == 'minmax':
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
        elif normalization == 'robust':
            data = (data - np.median(data)) / (np.percentile(data, 75) - np.percentile(data, 25))
        return np.mean(data)

def _compute_metrics(
    data: np.ndarray,
    computed_mean: float,
    metric_funcs: Dict[str, Callable]
) -> Dict[str, float]:
    """
    Compute various metrics for the mean computation.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    computed_mean : float
        Computed mean value
    metric_funcs : Dict[str, Callable]
        Dictionary of metric functions to compute

    Returns
    -------
    Dict[str, float]
        Dictionary of computed metrics
    """
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            if name == 'mse':
                metrics[name] = np.mean((data - computed_mean) ** 2)
            elif name == 'mae':
                metrics[name] = np.mean(np.abs(data - computed_mean))
            elif name == 'r2':
                ss_total = np.sum((data - np.mean(data)) ** 2)
                ss_res = np.sum((data - computed_mean) ** 2)
                metrics[name] = 1 - (ss_res / ss_total)
            else:
                metrics[name] = func(data, computed_mean)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def mean_fit(
    data: np.ndarray,
    normalization: Optional[str] = None,
    weights: Optional[np.ndarray] = None,
    metrics: Optional[Dict[str, Union[str, Callable]]] = None
) -> Dict:
    """
    Compute the mean of empirical data with various options.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    normalization : str, optional
        Type of normalization ('none', 'standard', 'minmax', 'robust')
    weights : np.ndarray, optional
        Weights for weighted mean computation
    metrics : Dict[str, Union[str, Callable]], optional
        Dictionary of metrics to compute

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': computed mean value
        - 'metrics': dictionary of computed metrics
        - 'params_used': parameters used in computation
        - 'warnings': any warnings generated

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> result = mean_fit(data)
    """
    _validate_inputs(data)

    # Prepare metrics
    metric_funcs = {}
    if metrics is not None:
        for name, func in metrics.items():
            if isinstance(func, str):
                if func == 'mse':
                    metric_funcs[name] = lambda x, y: np.mean((x - y) ** 2)
                elif func == 'mae':
                    metric_funcs[name] = lambda x, y: np.mean(np.abs(x - y))
                elif func == 'r2':
                    metric_funcs[name] = lambda x, y: 1 - (np.sum((x - y) ** 2) / np.sum((x - np.mean(x)) ** 2))
            else:
                metric_funcs[name] = func

    # Compute mean
    computed_mean = _compute_mean(data, normalization, weights)

    # Compute metrics
    computed_metrics = _compute_metrics(data, computed_mean, metric_funcs)

    # Prepare output
    result = {
        'result': computed_mean,
        'metrics': computed_metrics,
        'params_used': {
            'normalization': normalization,
            'weights': weights is not None
        },
        'warnings': []
    }

    return result

################################################################################
# median
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def median_fit(
    data: np.ndarray,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Compute the median of an empirical distribution using specified distance metric and solver.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    distance_metric : str or callable
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable
    solver : str
        Solver to use ('closed_form', 'gradient_descent')
    custom_distance : callable, optional
        Custom distance function if not using built-in metrics
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': computed median point
        - 'metrics': dictionary of metrics
        - 'params_used': parameters used in computation
        - 'warnings': any warnings encountered

    Examples
    --------
    >>> data = np.random.rand(100, 2)
    >>> result = median_fit(data, distance_metric='euclidean')
    """
    # Validate inputs
    _validate_inputs(data)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Choose distance metric
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_function(distance_metric)

    # Choose solver
    if solver == 'closed_form':
        median_point = _compute_median_closed_form(data, distance_func)
    elif solver == 'gradient_descent':
        median_point = _compute_median_gradient_descent(
            data, distance_func, tol=tol, max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(data, median_point, distance_func)

    return {
        'result': median_point,
        'metrics': metrics,
        'params_used': {
            'distance_metric': distance_metric if custom_distance is None else 'custom',
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _get_distance_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get distance function based on metric name."""
    metrics = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': lambda x, y: np.sum(np.abs(x - y)**2, axis=1)**(1/2)
    }
    if metric not in metrics:
        raise ValueError(f"Unknown distance metric: {metric}")
    return metrics[metric]

def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Euclidean distance between points."""
    return np.linalg.norm(x - y, axis=1)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Manhattan distance between points."""
    return np.sum(np.abs(x - y), axis=1)

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine distance between points."""
    return 1 - np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))

def _compute_median_closed_form(data: np.ndarray, distance_func: Callable) -> np.ndarray:
    """Compute median using closed-form solution."""
    # For small datasets, we can use the geometric median approximation
    if data.shape[0] <= 100:
        return np.median(data, axis=0)
    else:
        # For larger datasets, use Weiszfeld's algorithm
        return _compute_median_gradient_descent(data, distance_func)

def _compute_median_gradient_descent(
    data: np.ndarray,
    distance_func: Callable,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> np.ndarray:
    """Compute median using gradient descent."""
    n_samples, n_features = data.shape
    median_point = np.mean(data, axis=0)

    for _ in range(max_iter):
        distances = distance_func(data, median_point)
        weights = 1 / (distances + 1e-10)  # Avoid division by zero

        new_median = np.sum(weights[:, np.newaxis] * data, axis=0) / np.sum(weights)

        if np.linalg.norm(new_median - median_point) < tol:
            break

        median_point = new_median

    return median_point

def _compute_metrics(
    data: np.ndarray,
    median_point: np.ndarray,
    distance_func: Callable
) -> Dict[str, float]:
    """Compute metrics for the median."""
    distances = distance_func(data, median_point)
    return {
        'mean_distance': np.mean(distances),
        'median_distance': np.median(distances),
        'max_distance': np.max(distances)
    }

################################################################################
# mode
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def mode_fit(
    data: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = np.linalg.norm,
    normalize: bool = False,
    normalization_method: str = 'standard',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Compute the mode of an empirical distribution.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    distance_metric : Callable, optional
        Distance metric function (default: Euclidean norm).
    normalize : bool, optional
        Whether to normalize the data (default: False).
    normalization_method : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard').
    custom_metric : Callable, optional
        Custom distance metric function.
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Estimated mode.
        - 'metrics': Dictionary of metrics (e.g., convergence status).
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> result = mode_fit(data)
    """
    # Validate inputs
    _validate_inputs(data, distance_metric, custom_metric)

    # Normalize data if required
    normalized_data = _normalize_data(data, normalize, normalization_method)

    # Compute mode using the specified distance metric
    mode_result = _compute_mode(
        normalized_data,
        distance_metric if custom_metric is None else custom_metric,
        tol,
        max_iter
    )

    # Prepare output dictionary
    output = {
        'result': mode_result['mode'],
        'metrics': {
            'converged': mode_result['converged'],
            'iterations': mode_result['iterations']
        },
        'params_used': {
            'distance_metric': distance_metric.__name__ if custom_metric is None else 'custom',
            'normalize': normalize,
            'normalization_method': normalization_method if normalize else 'none',
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': mode_result.get('warnings', [])
    }

    return output

def _validate_inputs(
    data: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], float],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be a 1-dimensional array.")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values.")
    if custom_metric is not None and not callable(custom_metric):
        raise TypeError("Custom metric must be a callable function.")

def _normalize_data(
    data: np.ndarray,
    normalize: bool,
    method: str
) -> np.ndarray:
    """Normalize data based on the specified method."""
    if not normalize:
        return data

    if method == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return data - mean
        return (data - mean) / std
    elif method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        if min_val == max_val:
            return data
        return (data - min_val) / (max_val - min_val)
    elif method == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        if iqr == 0:
            return data - median
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_mode(
    data: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> Dict[str, Union[np.ndarray, bool, int]]:
    """Compute the mode of the data using the specified distance metric."""
    # Initialize with the median as a starting point
    mode = np.median(data)
    prev_mode = None
    iterations = 0

    while iterations < max_iter and (prev_mode is None or distance_metric(mode, prev_mode) > tol):
        prev_mode = mode
        # Compute the weighted average using the distance metric
        weights = np.exp(-distance_metric(data.reshape(-1, 1), mode))
        mode = np.sum(weights * data) / np.sum(weights)
        iterations += 1

    converged = distance_metric(mode, prev_mode) <= tol if prev_mode is not None else False
    warnings = [] if converged or iterations < max_iter else ["Did not converge within the maximum number of iterations."]

    return {
        'mode': mode,
        'converged': converged,
        'iterations': iterations,
        'warnings': warnings
    }

################################################################################
# variance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def variance_fit(
    data: np.ndarray,
    ddof: int = 1,
    normalization: Optional[str] = None,
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Compute the variance of an empirical distribution.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    ddof : int, optional
        Delta degrees of freedom. The divisor used in the calculation is N - ddof,
        where N is the number of elements. By default 1 for sample variance.
    normalization : str, optional
        Type of normalization to apply. Options: 'none', 'standard', 'minmax', 'robust'.
    custom_normalization : callable, optional
        Custom normalization function. If provided, overrides `normalization`.

    Returns
    -------
    dict
        Dictionary containing:
        - "result": computed variance
        - "metrics": dictionary of metrics (currently empty)
        - "params_used": dictionary of parameters used
        - "warnings": list of warnings

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> variance_fit(data)
    {
        'result': 2.0,
        'metrics': {},
        'params_used': {'ddof': 1, 'normalization': None},
        'warnings': []
    }
    """
    # Validate input
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if len(data.shape) != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

    # Apply normalization if specified
    normalized_data = _apply_normalization(data, normalization, custom_normalization)

    # Compute variance
    result = np.var(normalized_data, ddof=ddof)

    # Prepare output
    output = {
        "result": result,
        "metrics": {},
        "params_used": {
            "ddof": ddof,
            "normalization": normalization if custom_normalization is None else "custom"
        },
        "warnings": []
    }

    return output

def _apply_normalization(
    data: np.ndarray,
    normalization: Optional[str],
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """
    Apply normalization to the data.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    normalization : str, optional
        Type of normalization to apply.
    custom_normalization : callable, optional
        Custom normalization function.

    Returns
    -------
    np.ndarray
        Normalized data array.
    """
    if custom_normalization is not None:
        return custom_normalization(data)

    if normalization == "standard":
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
    elif normalization is not None:
        raise ValueError(f"Unknown normalization method: {normalization}")

    return data.copy()

################################################################################
# standard_deviation
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def standard_deviation_fit(
    data: np.ndarray,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'sample',
    ddof: int = 0,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Calculate the standard deviation of an empirical distribution with configurable options.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    normalization : str, optional
        Type of normalization to apply ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to use ('sample', 'population', custom callable).
    ddof : int, optional
        Delta degrees of freedom for sample standard deviation.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing:
        - 'result': calculated standard deviation
        - 'metrics': additional metrics if applicable
        - 'params_used': parameters used in calculation
        - 'warnings': any warnings generated

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> result = standard_deviation_fit(data)
    """
    # Validate inputs
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if len(data.shape) != 1:
        raise ValueError("Input data must be one-dimensional")

    # Handle normalization
    normalized_data = _apply_normalization(data, normalization)

    # Calculate standard deviation based on metric
    if callable(metric):
        std_dev = metric(normalized_data)
    elif metric == 'sample':
        std_dev = np.std(normalized_data, ddof=ddof)
    elif metric == 'population':
        std_dev = np.std(normalized_data, ddof=0)
    else:
        raise ValueError("Invalid metric specified")

    # Calculate additional metrics if needed
    metrics = {}
    if normalization != 'none':
        metrics['normalized_mean'] = np.mean(normalized_data)
        metrics['normalized_std'] = std_dev

    # Prepare output
    result_dict = {
        'result': float(std_dev),
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric if not callable(metric) else 'custom',
            'ddof': ddof
        },
        'warnings': []
    }

    return result_dict

def _apply_normalization(
    data: np.ndarray,
    normalization: str
) -> np.ndarray:
    """
    Apply specified normalization to the data.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    normalization : str
        Type of normalization to apply.

    Returns
    -------
    np.ndarray
        Normalized data array.
    """
    if normalization == 'none':
        return data.copy()

    if normalization == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std

    if normalization == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)

    if normalization == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr

    raise ValueError(f"Unknown normalization method: {normalization}")

def _validate_data(data: np.ndarray) -> None:
    """
    Validate input data array.

    Parameters
    ----------
    data : np.ndarray
        Input data array to validate.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if len(data.shape) != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")

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
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data must not contain NaN or infinite values.")

def _calculate_moment(data: np.ndarray, order: int) -> float:
    """Calculate the nth moment of the data."""
    return np.mean(data ** order)

def _calculate_skewness(
    data: np.ndarray,
    bias_correction: bool = True
) -> float:
    """Calculate the skewness of the data."""
    mean = np.mean(data)
    std = np.std(data, ddof=1 if bias_correction else 0)
    n = len(data)

    if std == 0:
        return 0.0

    skewness = (np.mean((data - mean) ** 3)) / (std ** 3)
    if bias_correction:
        skewness *= np.sqrt(n * (n - 1)) / (n - 2)
    return skewness

def _calculate_metrics(
    data: np.ndarray,
    skewness_value: float
) -> Dict[str, float]:
    """Calculate additional metrics related to skewness."""
    return {
        "skewness": skewness_value,
        "kurtosis": _calculate_kurtosis(data),
    }

def _calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate the kurtosis of the data."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)

    if std == 0:
        return 0.0

    kurtosis = (np.mean((data - mean) ** 4)) / (std ** 4)
    kurtosis = kurtosis - 3.0
    return kurtosis

def skewness_fit(
    data: np.ndarray,
    bias_correction: bool = True,
    custom_metrics: Optional[Callable[[np.ndarray, float], Dict[str, float]]] = None
) -> Dict[str, Any]:
    """
    Calculate the skewness of a dataset.

    Parameters:
    -----------
    data : np.ndarray
        Input data for which to calculate skewness.
    bias_correction : bool, optional
        Whether to apply bias correction (default is True).
    custom_metrics : Callable, optional
        Custom function to calculate additional metrics.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_input(data)

    skewness_value = _calculate_skewness(data, bias_correction)
    metrics = _calculate_metrics(data, skewness_value)

    if custom_metrics is not None:
        metrics.update(custom_metrics(data, skewness_value))

    return {
        "result": skewness_value,
        "metrics": metrics,
        "params_used": {
            "bias_correction": bias_correction,
            "custom_metrics": custom_metrics is not None
        },
        "warnings": []
    }

# Example usage:
# data = np.array([1, 2, 3, 4, 5])
# result = skewness_fit(data)

################################################################################
# kurtosis
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for kurtosis calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values.")

def _compute_kurtosis(data: np.ndarray, fisher: bool = True) -> float:
    """Compute the kurtosis of a dataset."""
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)

    if std_dev == 0:
        raise ValueError("Standard deviation is zero, cannot compute kurtosis.")

    deviations = data - mean
    kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(deviations**4) / (std_dev**4) - 3 * ((n - 1)**2 / ((n - 2) * (n - 3)))

    if fisher:
        kurtosis -= 3

    return kurtosis

def _compute_metrics(data: np.ndarray, kurtosis_value: float) -> Dict[str, float]:
    """Compute additional metrics related to kurtosis."""
    return {
        "kurtosis": kurtosis_value,
        "excess_kurtosis": kurtosis_value if fisher else kurtosis_value + 3
    }

def kurtosis_fit(
    data: np.ndarray,
    fisher: bool = True,
    metric_func: Optional[Callable[[np.ndarray, float], Dict[str, float]]] = None
) -> Dict[str, Any]:
    """
    Compute the kurtosis of a dataset with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data for which to compute kurtosis.
    fisher : bool, optional
        Whether to use Fisher's definition of kurtosis (default: True).
    metric_func : Callable, optional
        Custom function to compute additional metrics (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.

    Example:
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> kurtosis_fit(data)
    {
        'result': {'kurtosis': -1.2},
        'metrics': {'kurtosis': -1.2, 'excess_kurtosis': -1.2},
        'params_used': {'fisher': True, 'metric_func': None},
        'warnings': []
    }
    """
    _validate_input(data)

    kurtosis_value = _compute_kurtosis(data, fisher)
    metrics = _compute_metrics(data, kurtosis_value)

    if metric_func is not None:
        custom_metrics = metric_func(data, kurtosis_value)
        metrics.update(custom_metrics)

    return {
        "result": {"kurtosis": kurtosis_value},
        "metrics": metrics,
        "params_used": {
            "fisher": fisher,
            "metric_func": metric_func.__name__ if metric_func else None
        },
        "warnings": []
    }
