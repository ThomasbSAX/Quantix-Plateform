"""
Quantix – Module correlation_exploratoire
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# matrice_correlation
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input contains infinite values")

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

def compute_correlation(X: np.ndarray, metric: str = 'pearson', custom_metric: Optional[Callable] = None) -> np.ndarray:
    """Compute correlation matrix using specified metric."""
    if custom_metric is not None:
        return custom_metric(X)
    if metric == 'pearson':
        cov = np.cov(X, rowvar=False)
        stddev = np.std(X, axis=0, ddof=1)
        corr = cov / np.outer(stddev, stddev)
    elif metric == 'spearman':
        ranks = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), axis=0, arr=X)
        corr = np.corrcoef(ranks, rowvar=False)
    elif metric == 'kendall':
        corr = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                tau, _ = kendalltau(X[:, i], X[:, j])
                corr[i, j] = tau
                corr[j, i] = tau
    else:
        raise ValueError(f"Unknown correlation metric: {metric}")
    return corr

def kendalltau(x: np.ndarray, y: np.ndarray) -> tuple:
    """Compute Kendall's tau."""
    n = len(x)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i+1, n):
            if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                concordant += 1
            elif (x[i] - x[j]) * (y[i] - y[j]) < 0:
                discordant += 1
    tau = (concordant - discordant) / (0.5 * n * (n - 1))
    return tau, None

def compute_metrics(corr_matrix: np.ndarray) -> Dict[str, float]:
    """Compute metrics from correlation matrix."""
    return {
        'mean_correlation': np.mean(corr_matrix),
        'max_correlation': np.max(corr_matrix),
        'min_correlation': np.min(corr_matrix),
        'median_correlation': np.median(corr_matrix)
    }

def matrice_correlation_fit(X: np.ndarray,
                           metric: str = 'pearson',
                           normalization: str = 'standard',
                           custom_metric: Optional[Callable] = None) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict]]:
    """
    Compute correlation matrix with configurable options.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    metric : str or callable
        Correlation metric to use ('pearson', 'spearman', 'kendall')
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_metric : callable, optional
        Custom correlation metric function

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': correlation matrix
        - 'metrics': computed metrics
        - 'params_used': parameters used
        - 'warnings': any warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> result = matrice_correlation_fit(X, metric='spearman', normalization='standard')
    """
    # Validate input
    validate_input(X)

    # Normalize data
    X_normalized = normalize_data(X, method=normalization)

    # Compute correlation matrix
    corr_matrix = compute_correlation(X_normalized, metric=metric, custom_metric=custom_metric)

    # Compute metrics
    metrics = compute_metrics(corr_matrix)

    return {
        'result': corr_matrix,
        'metrics': metrics,
        'params_used': {
            'metric': metric if custom_metric is None else 'custom',
            'normalization': normalization
        },
        'warnings': []
    }

################################################################################
# coefficient_pearson
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("Inputs must have the same shape.")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("Inputs must not contain NaN values.")
    if np.isinf(x).any() or np.isinf(y).any():
        raise ValueError("Inputs must not contain infinite values.")

def _normalize_data(x: np.ndarray, y: np.ndarray, normalization: str) -> tuple:
    """Normalize input arrays based on the specified method."""
    if normalization == "standard":
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    elif normalization == "minmax":
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == "robust":
        x = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    return x, y

def _compute_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    cov = np.cov(x, y)[0, 1]
    std_x = np.std(x)
    std_y = np.std(y)
    return cov / (std_x * std_y)

def coefficient_pearson_fit(
    x: np.ndarray,
    y: np.ndarray,
    normalization: str = "none",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the Pearson correlation coefficient between two arrays.

    Parameters:
    -----------
    x : np.ndarray
        First input array.
    y : np.ndarray
        Second input array.
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    custom_metric : Callable, optional
        Custom metric function to compute correlation.

    Returns:
    --------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, y)

    x_norm, y_norm = _normalize_data(x, y, normalization)

    if custom_metric is not None:
        result = custom_metric(x_norm, y_norm)
    else:
        result = _compute_pearson(x_norm, y_norm)

    metrics = {"pearson": result}
    params_used = {
        "normalization": normalization,
        "custom_metric": custom_metric is not None
    }
    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([2, 4, 6, 8, 10])
# result = coefficient_pearson_fit(x, y)

################################################################################
# coefficient_spearman
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Spearman's coefficient calculation."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _rank_data(data: np.ndarray) -> np.ndarray:
    """Compute ranks for the input data, handling ties appropriately."""
    sorted_data = np.sort(data)
    ranks = np.zeros_like(data)
    i = 0
    while i < len(sorted_data):
        j = i
        while j < len(sorted_data) and sorted_data[j] == sorted_data[i]:
            j += 1
        ranks[data == sorted_data[i]] = (i + 1 + j - 1) / 2
        i = j
    return ranks

def _compute_spearman_coefficient(x_ranks: np.ndarray, y_ranks: np.ndarray) -> float:
    """Compute Spearman's rank correlation coefficient."""
    n = len(x_ranks)
    covariance = np.cov(x_ranks, y_ranks)[0, 1]
    x_var = np.var(x_ranks, ddof=1)
    y_var = np.var(y_ranks, ddof=1)
    if x_var == 0 or y_var == 0:
        return 0.0
    return covariance / np.sqrt(x_var * y_var)

def coefficient_spearman_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    normalization: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute Spearman's rank correlation coefficient between two arrays.

    Parameters:
    -----------
    x : np.ndarray
        First input array.
    y : np.ndarray
        Second input array.
    normalization : str, optional
        Normalization method to apply (not used in Spearman's coefficient but kept for consistency).
    custom_metric : callable, optional
        Custom metric function (not used in Spearman's coefficient but kept for consistency).

    Returns:
    --------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, y)

    x_ranks = _rank_data(x)
    y_ranks = _rank_data(y)

    spearman_coeff = _compute_spearman_coefficient(x_ranks, y_ranks)

    result = {
        "result": spearman_coeff,
        "metrics": {},
        "params_used": {
            "normalization": normalization,
            "custom_metric": custom_metric.__name__ if custom_metric else None
        },
        "warnings": []
    }

    return result

# Example usage:
# coeff = coefficient_spearman_fit(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1]))

################################################################################
# coefficient_kendall
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Kendall's coefficient calculation."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input x contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input y contains NaN or infinite values.")

def _count_concordant_discordant_pairs(x: np.ndarray, y: np.ndarray) -> Dict[str, int]:
    """Count concordant and discordant pairs for Kendall's coefficient."""
    n = len(x)
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            delta_x = x[i] - x[j]
            delta_y = y[i] - y[j]

            if delta_x * delta_y > 0:
                concordant += 1
            elif delta_x * delta_y < 0:
                discordant += 1

    return {"concordant": concordant, "discordant": discordant}

def _calculate_kendall_tau(pairs: Dict[str, int], n: int) -> float:
    """Calculate Kendall's tau coefficient from concordant and discordant pairs."""
    concordant = pairs["concordant"]
    discordant = pairs["discordant"]

    if concordant + discordant == 0:
        return 0.0

    tau = (concordant - discordant) / np.sqrt((concordant + discordant) * n * (n - 1))
    return tau

def coefficient_kendall_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    normalization: Optional[str] = None,
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Compute Kendall's tau coefficient between two arrays.

    Parameters:
    -----------
    x : np.ndarray
        First input array.
    y : np.ndarray
        Second input array.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    custom_normalization : callable, optional
        Custom normalization function.

    Returns:
    --------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, y)
    n = len(x)

    # Apply normalization if specified
    if normalization == 'standard':
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    elif normalization == 'minmax':
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == 'robust':
        x = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    elif custom_normalization is not None:
        x = custom_normalization(x)
        y = custom_normalization(y)

    # Calculate concordant and discordant pairs
    pairs = _count_concordant_discordant_pairs(x, y)

    # Calculate Kendall's tau
    tau = _calculate_kendall_tau(pairs, n)

    # Prepare output dictionary
    result = {
        "result": tau,
        "metrics": {
            "concordant_pairs": pairs["concordant"],
            "discordant_pairs": pairs["discordant"]
        },
        "params_used": {
            "normalization": normalization,
            "custom_normalization": custom_normalization is not None
        },
        "warnings": []
    }

    return result

# Example usage:
# x = np.array([1, 2, 3, 4])
# y = np.array([4, 3, 2, 1])
# result = coefficient_kendall_fit(x, y)

################################################################################
# heatmap_correlation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input_matrix(matrix: np.ndarray) -> None:
    """Validate the input matrix for correlation analysis."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2-dimensional array")
    if np.any(np.isnan(matrix)):
        raise ValueError("Input matrix contains NaN values")
    if np.any(np.isinf(matrix)):
        raise ValueError("Input matrix contains infinite values")

def normalize_matrix(
    matrix: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """Normalize the input matrix using specified method."""
    if custom_func is not None:
        return custom_func(matrix)

    if method == "none":
        return matrix
    elif method == "standard":
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        return (matrix - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(matrix, axis=0)
        max_val = np.max(matrix, axis=0)
        return (matrix - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(matrix, axis=0)
        iqr = np.subtract(*np.percentile(matrix, [75, 25], axis=0))
        return (matrix - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_correlation_matrix(
    matrix: np.ndarray,
    metric: str = "pearson",
    custom_metric: Optional[Callable] = None
) -> np.ndarray:
    """Compute correlation matrix using specified metric."""
    if custom_metric is not None:
        return custom_metric(matrix)

    n = matrix.shape[0]
    if metric == "pearson":
        mean = np.mean(matrix, axis=0)
        centered = matrix - mean
        cov = np.dot(centered.T, centered) / (n - 1)
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
    elif metric == "spearman":
        ranks = np.argsort(np.argsort(matrix, axis=0), axis=0)
        corr = compute_correlation_matrix(ranks, "pearson")
    else:
        raise ValueError(f"Unknown correlation metric: {metric}")

    # Ensure diagonal is 1
    np.fill_diagonal(corr, 1)
    return corr

def heatmap_correlation_fit(
    matrix: np.ndarray,
    normalization: str = "standard",
    metric: str = "pearson",
    custom_normalization: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Compute correlation heatmap with configurable options.

    Parameters:
    - matrix: Input data matrix (n_samples, n_features)
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - metric: Correlation metric ('pearson', 'spearman')
    - custom_normalization: Custom normalization function
    - custom_metric: Custom correlation metric function

    Returns:
    Dictionary containing:
    - result: Computed correlation matrix
    - metrics: Dictionary of computed metrics
    - params_used: Parameters used in computation
    - warnings: Any warnings generated during computation

    Example:
    >>> data = np.random.randn(100, 5)
    >>> result = heatmap_correlation_fit(data)
    """
    # Validate input
    validate_input_matrix(matrix)

    warnings = []

    # Normalize data
    normalized_matrix = normalize_matrix(
        matrix,
        method=normalization,
        custom_func=custom_normalization
    )

    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(
        normalized_matrix,
        metric=metric,
        custom_metric=custom_metric
    )

    # Calculate some basic metrics
    metrics = {
        "mean_correlation": np.mean(corr_matrix),
        "max_correlation": np.max(corr_matrix),
        "min_correlation": np.min(corr_matrix)
    }

    # Prepare output
    result = {
        "result": corr_matrix,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization if custom_normalization is None else "custom",
            "metric": metric if custom_metric is None else "custom"
        },
        "warnings": warnings
    }

    return result

################################################################################
# scatter_plot
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def validate_inputs(
    x: np.ndarray,
    y: np.ndarray,
    normalize_x: Optional[Callable] = None,
    normalize_y: Optional[Callable] = None
) -> Dict[str, np.ndarray]:
    """
    Validate and normalize input data.

    Parameters:
    -----------
    x : np.ndarray
        Input feature array.
    y : np.ndarray
        Target array.
    normalize_x : Optional[Callable]
        Normalization function for x. If None, no normalization.
    normalize_y : Optional[Callable]
        Normalization function for y. If None, no normalization.

    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing normalized x and y.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples.")

    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("x contains NaN or infinite values.")

    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

    if normalize_x is not None:
        x = normalize_x(x)

    if normalize_y is not None:
        y = normalize_y(y)

    return {"x": x, "y": y}

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_functions: Dict[str, Callable]
) -> Dict[str, float]:
    """
    Compute metrics for the scatter plot.

    Parameters:
    -----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values from the scatter plot fit.
    metric_functions : Dict[str, Callable]
        Dictionary of metric functions to compute.

    Returns:
    --------
    Dict[str, float]
        Dictionary containing computed metrics.
    """
    metrics = {}
    for name, func in metric_functions.items():
        try:
            metrics[name] = func(y_true, y_pred)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def scatter_plot_fit(
    x: np.ndarray,
    y: np.ndarray,
    normalize_x: Optional[Callable] = None,
    normalize_y: Optional[Callable] = None,
    metric_functions: Dict[str, Callable] = None
) -> Dict[str, Any]:
    """
    Fit a scatter plot and compute metrics.

    Parameters:
    -----------
    x : np.ndarray
        Input feature array.
    y : np.ndarray
        Target array.
    normalize_x : Optional[Callable]
        Normalization function for x. If None, no normalization.
    normalize_y : Optional[Callable]
        Normalization function for y. If None, no normalization.
    metric_functions : Dict[str, Callable]
        Dictionary of metric functions to compute.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Default metric functions if none provided
    default_metrics = {
        "mse": lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
        "mae": lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
        "r2": lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    }
    if metric_functions is None:
        metric_functions = default_metrics
    else:
        metric_functions.update(default_metrics)

    # Validate and normalize inputs
    validated_data = validate_inputs(x, y, normalize_x, normalize_y)
    x_norm = validated_data["x"]
    y_norm = validated_data["y"]

    # Compute the scatter plot fit (linear regression for simplicity)
    A = np.vstack([x_norm, np.ones(len(x_norm))]).T
    m, c = np.linalg.lstsq(A, y_norm, rcond=None)[0]
    y_pred = m * x_norm + c

    # Compute metrics
    metrics = compute_metrics(y_norm, y_pred, metric_functions)

    return {
        "result": {"slope": m, "intercept": c},
        "metrics": metrics,
        "params_used": {
            "normalize_x": normalize_x.__name__ if normalize_x else None,
            "normalize_y": normalize_y.__name__ if normalize_y else None
        },
        "warnings": []
    }

# Example usage:
if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])

    def standard_normalize(arr: np.ndarray) -> np.ndarray:
        return (arr - np.mean(arr)) / np.std(arr)

    result = scatter_plot_fit(
        x=x,
        y=y,
        normalize_x=standard_normalize,
        normalize_y=None
    )
    print(result)

################################################################################
# pairplot
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional, List
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
    CUSTOM = "custom"

class Distance(Enum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    MINKOWSKI = "minkowski"
    CUSTOM = "custom"

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or Inf values")

def normalize_data(
    X: np.ndarray,
    normalization: Union[Normalization, str] = Normalization.STANDARD
) -> np.ndarray:
    """Apply normalization to the data."""
    if isinstance(normalization, str):
        normalization = Normalization(normalization)

    X_normalized = X.copy()
    if normalization == Normalization.STANDARD:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / std
    elif normalization == Normalization.MINMAX:
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_normalized = (X - min_val) / (max_val - min_val)
    elif normalization == Normalization.ROBUST:
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - median) / iqr
    return X_normalized

def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[Metric, str] = Metric.MSE,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> float:
    """Compute the specified metric."""
    if isinstance(metric, str):
        metric = Metric(metric)

    if custom_metric is not None:
        return custom_metric(y_true, y_pred)

    if metric == Metric.MSE:
        return np.mean((y_true - y_pred) ** 2)
    elif metric == Metric.MAE:
        return np.mean(np.abs(y_true - y_pred))
    elif metric == Metric.R2:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

def compute_distance(
    x: np.ndarray,
    y: np.ndarray,
    distance: Union[Distance, str] = Distance.EUCLIDEAN,
    p: float = 2.0,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> float:
    """Compute the specified distance between two vectors."""
    if isinstance(distance, str):
        distance = Distance(distance)

    if custom_distance is not None:
        return custom_distance(x, y)

    if distance == Distance.EUCLIDEAN:
        return np.linalg.norm(x - y)
    elif distance == Distance.MANHATTAN:
        return np.sum(np.abs(x - y))
    elif distance == Distance.COSINE:
        return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif distance == Distance.MINKOWSKI:
        return np.sum(np.abs(x - y) ** p) ** (1 / p)

def pairplot_fit(
    X: np.ndarray,
    normalization: Union[Normalization, str] = Normalization.STANDARD,
    metric: Union[Metric, str] = Metric.MSE,
    distance: Union[Distance, str] = Distance.EUCLIDEAN,
    p: float = 2.0,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], List[str]]]:
    """
    Compute pairplot for exploratory correlation analysis.

    Parameters:
    - X: Input data array of shape (n_samples, n_features)
    - normalization: Normalization method to apply
    - metric: Metric for evaluation
    - distance: Distance metric for pairwise comparisons
    - p: Parameter for Minkowski distance
    - custom_metric: Custom metric function
    - custom_distance: Custom distance function

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    validate_input(X)

    X_normalized = normalize_data(X, normalization)
    n_features = X.shape[1]
    correlation_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                distance_ij = compute_distance(
                    X_normalized[:, i],
                    X_normalized[:, j],
                    distance,
                    p,
                    custom_distance
                )
                correlation_matrix[i, j] = 1 / (1 + distance_ij)

    metrics = {
        "metric": str(metric) if not isinstance(metric, str) else metric,
        "distance": str(distance) if not isinstance(distance, str) else distance
    }

    params_used = {
        "normalization": normalization.value if isinstance(normalization, Normalization) else normalization,
        "p": p
    }

    warnings = []

    return {
        "result": correlation_matrix,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
"""
X = np.random.rand(100, 5)
result = pairplot_fit(X, normalization="standard", metric="r2")
print(result["result"])
"""

################################################################################
# correlation_significative
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

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

def _normalize_data(X: np.ndarray, y: np.ndarray,
                   normalization: str = 'standard',
                   custom_normalization: Optional[Callable] = None) -> tuple:
    """Normalize data according to specified method."""
    if custom_normalization is not None:
        X_norm = custom_normalization(X)
        y_norm = custom_normalization(y.reshape(-1, 1)).flatten()
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
        X_norm = X.copy()
        y_norm = y.copy()
    return X_norm, y_norm

def _compute_correlation(X: np.ndarray, y: np.ndarray,
                        metric: str = 'pearson',
                        custom_metric: Optional[Callable] = None) -> float:
    """Compute correlation between X and y using specified metric."""
    if custom_metric is not None:
        return custom_metric(X, y)
    elif metric == 'pearson':
        cov = np.cov(X.T, y, bias=True)
        stddev_X = np.std(X, axis=0)
        stddev_y = np.std(y)
        return cov[0:-1, -1] / (stddev_X * stddev_y)
    elif metric == 'spearman':
        X_rank = np.argsort(np.argsort(X, axis=0), axis=0)
        y_rank = np.argsort(np.argsort(y))
        return _compute_correlation(X_rank, y_rank, 'pearson')
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_significance(correlation: float, n_samples: int) -> float:
    """Compute significance of correlation coefficient."""
    t_stat = correlation * np.sqrt((n_samples - 2) / (1 - correlation**2))
    p_value = 2 * (1 - _t_cdf(t_stat, n_samples - 2))
    return p_value

def _t_cdf(x: float, df: int) -> float:
    """Cumulative distribution function of Student's t-distribution."""
    from scipy.stats import t
    return t.cdf(x, df)

def correlation_significative_fit(X: np.ndarray,
                                y: np.ndarray,
                                normalization: str = 'standard',
                                metric: str = 'pearson',
                                custom_normalization: Optional[Callable] = None,
                                custom_metric: Optional[Callable] = None) -> Dict:
    """
    Compute significant correlation between features and target.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalization : str or callable, optional
        Normalization method ('standard', 'minmax', 'robust') or custom function
    metric : str, optional
        Correlation metric ('pearson', 'spearman')
    custom_normalization : callable, optional
        Custom normalization function
    custom_metric : callable, optional
        Custom correlation metric function

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': correlation coefficients and p-values
        - 'metrics': computed metrics
        - 'params_used': parameters used in computation
        - 'warnings': any warnings generated

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = correlation_significative_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization, custom_normalization)

    # Compute correlations
    n_samples = X.shape[0]
    correlations = np.zeros(X.shape[1])
    p_values = np.zeros(X.shape[1])

    for i in range(X.shape[1]):
        correlations[i] = _compute_correlation(X_norm[:, i], y_norm, metric, custom_metric)
        p_values[i] = _compute_significance(correlations[i], n_samples)

    # Prepare output
    result = {
        'result': {
            'correlations': correlations,
            'p_values': p_values
        },
        'metrics': {
            'normalization': normalization,
            'metric': metric if custom_metric is None else 'custom'
        },
        'params_used': {
            'normalization_method': normalization,
            'correlation_metric': metric
        },
        'warnings': []
    }

    return result

################################################################################
# test_hypothese_correlation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and preprocess input data.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    normalizer : Optional[Callable]
        Function to normalize the input data

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Normalized X and y arrays

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

    if normalizer is not None:
        X = normalizer(X)
        y = normalizer(y.reshape(-1, 1)).flatten()

    return X, y

def _compute_correlation_statistic(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'pearson'
) -> float:
    """
    Compute correlation statistic between X and y.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    method : str
        Correlation method ('pearson', 'spearman')

    Returns
    -------
    float
        Correlation statistic

    Raises
    ------
    ValueError
        If method is not supported
    """
    if method == 'pearson':
        return np.corrcoef(X.flatten(), y)[0, 1]
    elif method == 'spearman':
        return np.corrcoef(np.argsort(X.flatten()), np.argsort(y))[0, 1]
    else:
        raise ValueError(f"Unsupported correlation method: {method}")

def _compute_p_value(
    statistic: float,
    n_samples: int
) -> float:
    """
    Compute p-value for the correlation statistic.

    Parameters
    ----------
    statistic : float
        Correlation statistic value
    n_samples : int
        Number of samples

    Returns
    -------
    float
        p-value
    """
    # Fisher transformation for Pearson correlation
    z = 0.5 * np.log((1 + statistic) / (1 - statistic))
    se = 1 / np.sqrt(n_samples - 3)
    p_value = 2 * (1 - _standard_normal_cdf(z / se))
    return min(p_value, 2 * (1 - _standard_normal_cdf(-z / se)))

def _standard_normal_cdf(x: float) -> float:
    """
    Approximate standard normal CDF.

    Parameters
    ----------
    x : float
        Value to compute CDF for

    Returns
    -------
    float
        CDF value
    """
    return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

def test_hypothese_correlation_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    method: str = 'pearson',
    alpha: float = 0.05,
    custom_statistic_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_p_value_func: Optional[Callable[[float, int], float]] = None
) -> Dict[str, Union[float, Dict, str]]:
    """
    Test hypothesis of correlation between X and y.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    normalizer : Optional[Callable]
        Function to normalize the input data
    method : str
        Correlation method ('pearson', 'spearman')
    alpha : float
        Significance level for hypothesis test
    custom_statistic_func : Optional[Callable]
        Custom function to compute correlation statistic
    custom_p_value_func : Optional[Callable]
        Custom function to compute p-value

    Returns
    -------
    Dict[str, Union[float, Dict, str]]
        Dictionary containing:
        - result: dict with 'statistic', 'p_value', 'hypothesis'
        - metrics: dict with correlation metric
        - params_used: dict of parameters used
        - warnings: list of warning messages

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = test_hypothese_correlation_fit(X, y)
    """
    warnings = []

    # Validate inputs
    X_valid, y_valid = _validate_inputs(X, y, normalizer)

    # Compute correlation statistic
    if custom_statistic_func is not None:
        statistic = custom_statistic_func(X_valid, y_valid)
    else:
        statistic = _compute_correlation_statistic(X_valid, y_valid, method)

    # Compute p-value
    if custom_p_value_func is not None:
        p_value = custom_p_value_func(statistic, X_valid.shape[0])
    else:
        p_value = _compute_p_value(statistic, X_valid.shape[0])

    # Determine hypothesis result
    if p_value < alpha:
        hypothesis = "Reject H0 (significant correlation)"
    else:
        hypothesis = "Fail to reject H0 (no significant correlation)"

    # Prepare output
    result = {
        "statistic": statistic,
        "p_value": p_value,
        "hypothesis": hypothesis
    }

    metrics = {
        "correlation": statistic
    }

    params_used = {
        "normalizer": normalizer.__name__ if normalizer else None,
        "method": method,
        "alpha": alpha,
        "custom_statistic_func": custom_statistic_func.__name__ if custom_statistic_func else None,
        "custom_p_value_func": custom_p_value_func.__name__ if custom_p_value_func else None
    }

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# outliers_impact
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def outliers_impact_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Estimate the impact of outliers on correlation metrics.

    Parameters
    ----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the input data.
    metric : str
        Metric to use for evaluation ('mse', 'mae', 'r2', 'logloss').
    distance : str
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski').
    solver : str
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : Optional[str]
        Regularization type (None, 'l1', 'l2', 'elasticnet').
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom distance function.

    Returns
    -------
    Dict[str, Union[Dict, float, str]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = outliers_impact_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = normalizer(X)
    y_normalized = normalizer(y.reshape(-1, 1)).flatten()

    # Choose metric
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric(metric)

    # Choose distance
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance(distance)

    # Solve for parameters
    params = _solve(
        X_normalized, y_normalized,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Calculate metrics
    predictions = _predict(X_normalized, params)
    metrics = {
        'metric': metric_func(y_normalized, predictions),
        'distance': distance_func(X_normalized, y_normalized.reshape(-1, 1))
    }

    # Check for outliers
    warnings = _check_outliers(X_normalized, y_normalized)

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': warnings
    }

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

def _get_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the specified metric function."""
    metrics = {
        'mse': _mse,
        'mae': _mae,
        'r2': _r2_score,
        'logloss': _log_loss
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the specified distance function."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': lambda x, y: np.linalg.norm(x - y, ord=3)
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _solve(
    X: np.ndarray,
    y: np.ndarray,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve for parameters using the specified solver."""
    solvers = {
        'closed_form': _closed_form_solution,
        'gradient_descent': lambda x, y: _gradient_descent(x, y, tol, max_iter),
        'newton': lambda x, y: _newton_method(x, y, tol, max_iter),
        'coordinate_descent': lambda x, y: _coordinate_descent(x, y, tol, max_iter)
    }
    if solver not in solvers:
        raise ValueError(f"Unknown solver: {solver}")

    if regularization is not None:
        reg_func = _get_regularizer(regularization)
    else:
        reg_func = lambda x: 0

    return solvers[solver](X, y)

def _get_regularizer(regularization: str) -> Callable[[np.ndarray], float]:
    """Return the specified regularizer function."""
    regularizers = {
        'l1': lambda x: np.linalg.norm(x, ord=1),
        'l2': lambda x: np.linalg.norm(x, ord=2),
        'elasticnet': lambda x: 0.5 * np.linalg.norm(x, ord=2) + 0.5 * np.linalg.norm(x, ord=1)
    }
    if regularization not in regularizers:
        raise ValueError(f"Unknown regularization: {regularization}")
    return regularizers[regularization]

def _closed_form_solution(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form solution for linear regression."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent solver."""
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = -2 * X.T @ (y - X @ beta) / n_samples
        new_beta = beta - learning_rate * gradient

        if np.linalg.norm(new_beta - beta) < tol:
            break
        beta = new_beta

    return beta

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Newton's method solver."""
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)

    for _ in range(max_iter):
        residuals = y - X @ beta
        gradient = -2 * X.T @ residuals / n_samples
        hessian = 2 * X.T @ X / n_samples

        new_beta = beta - np.linalg.inv(hessian) @ gradient

        if np.linalg.norm(new_beta - beta) < tol:
            break
        beta = new_beta

    return beta

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Coordinate descent solver."""
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - X @ beta + beta[j] * X_j
            beta[j] = np.sum(X_j * residuals) / np.sum(X_j ** 2)

        if np.linalg.norm(beta - beta) < tol:
            break

    return beta

def _predict(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Predict using the model parameters."""
    return X @ params

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(X: np.ndarray, y: np.ndarray) -> float:
    """Euclidean distance."""
    return np.linalg.norm(X - y, axis=1).mean()

def _manhattan_distance(X: np.ndarray, y: np.ndarray) -> float:
    """Manhattan distance."""
    return np.sum(np.abs(X - y), axis=1).mean()

def _cosine_distance(X: np.ndarray, y: np.ndarray) -> float:
    """Cosine distance."""
    return 1 - np.mean(np.sum(X * y, axis=1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(y, axis=1)))

def _check_outliers(X: np.ndarray, y: np.ndarray) -> Dict[str, str]:
    """Check for outliers in the data."""
    warnings = {}
    if np.any(np.abs(X) > 3):
        warnings['X_outliers'] = 'Potential outliers detected in X'
    if np.any(np.abs(y) > 3):
        warnings['y_outliers'] = 'Potential outliers detected in y'
    return warnings
