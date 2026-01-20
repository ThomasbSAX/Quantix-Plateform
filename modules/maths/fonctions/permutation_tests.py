"""
Quantix – Module permutation_tests
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# basic_concept
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def basic_concept_fit(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    normalization: str = 'none',
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Perform a basic permutation test on the given data.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    metric : str or callable, optional
        Metric to evaluate. Can be 'mse', 'mae', 'r2', or a custom callable.
    normalization : str, optional
        Normalization method. Can be 'none', 'standard', 'minmax', or 'robust'.
    n_permutations : int, optional
        Number of permutations to perform.
    random_state : int or None, optional
        Random seed for reproducibility.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.

    Returns:
    --------
    Dict containing:
        - 'result': Dictionary with test results.
        - 'metrics': Dictionary with computed metrics.
        - 'params_used': Dictionary of parameters used.
        - 'warnings': List of warnings encountered.

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = basic_concept_fit(X, y, metric='mse', normalization='standard')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state
    rng = np.random.RandomState(random_state)

    # Normalize data if required
    X_normalized, y_normalized = _apply_normalization(X, y, normalization)

    # Choose metric
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        if custom_metric is None:
            raise ValueError("custom_metric must be provided when metric is a callable")
        metric_func = custom_metric

    # Perform permutation test
    observed_statistic, permuted_statistics = _perform_permutation_test(
        X_normalized, y_normalized, metric_func, n_permutations, rng
    )

    # Calculate p-value
    p_value = _calculate_p_value(observed_statistic, permuted_statistics)

    # Prepare results
    result = {
        'observed_statistic': observed_statistic,
        'p_value': p_value,
        'permutation_distribution': permuted_statistics
    }

    metrics = {
        'observed_metric': observed_statistic,
        'p_value': p_value
    }

    params_used = {
        'metric': metric,
        'normalization': normalization,
        'n_permutations': n_permutations,
        'random_state': random_state
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
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

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    normalization: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply specified normalization to data."""
    if normalization == 'standard':
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_normalized = (y - np.mean(y)) / np.std(y)
    elif normalization == 'minmax':
        X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == 'robust':
        X_normalized = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y_normalized = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        X_normalized = X.copy()
        y_normalized = y.copy()

    return X_normalized, y_normalized

def _get_metric_function(metric: str) -> Callable:
    """Get the appropriate metric function based on input string."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

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

def _perform_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    n_permutations: int,
    rng: np.random.RandomState
) -> tuple[float, np.ndarray]:
    """Perform permutation test and return observed statistic and permuted statistics."""
    # Calculate observed statistic
    y_pred = np.mean(X, axis=0)  # Simple baseline prediction
    observed_statistic = metric_func(y, y_pred)

    # Calculate permuted statistics
    n_samples = X.shape[0]
    permuted_statistics = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Permute y
        y_permuted = rng.permutation(y)

        # Calculate prediction on permuted data
        y_pred_permuted = np.mean(X, axis=0)  # Same baseline prediction

        # Calculate statistic
        permuted_statistics[i] = metric_func(y_permuted, y_pred_permuted)

    return observed_statistic, permuted_statistics

def _calculate_p_value(
    observed_statistic: float,
    permuted_statistics: np.ndarray
) -> float:
    """Calculate p-value from observed statistic and permuted statistics."""
    if observed_statistic > np.mean(permuted_statistics):
        # One-tailed test (observed statistic is higher than permuted)
        p_value = np.sum(permuted_statistics >= observed_statistic) / len(permuted_statistics)
    else:
        # One-tailed test (observed statistic is lower than permuted)
        p_value = np.sum(permuted_statistics <= observed_statistic) / len(permuted_statistics)

    return p_value

################################################################################
# null_hypothesis
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def null_hypothesis_fit(
    observed_statistic: float,
    permutations: np.ndarray,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'mean',
    solver: str = 'exact',
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute the null hypothesis p-value using permutation tests.

    Parameters:
    -----------
    observed_statistic : float
        The observed test statistic under the alternative hypothesis.
    permutations : np.ndarray
        Array of permuted test statistics under the null hypothesis.
    normalization : str, optional (default='none')
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : str or callable, optional (default='mean')
        Metric to compute p-value: 'mean', 'median', or custom callable.
    solver : str, optional (default='exact')
        Solver method: 'exact', 'approximate'.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    **kwargs :
        Additional keyword arguments for specific solvers or metrics.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.

    Example:
    --------
    >>> observed_stat = 1.5
    >>> perms = np.random.normal(0, 1, 1000)
    >>> result = null_hypothesis_fit(observed_stat, perms)
    """
    # Validate inputs
    _validate_inputs(observed_statistic, permutations)

    # Normalize permutations if required
    normalized_perms = _apply_normalization(permutations, normalization)

    # Compute p-value based on chosen metric
    if isinstance(metric, str):
        if metric == 'mean':
            p_value = _compute_mean_pvalue(observed_statistic, normalized_perms)
        elif metric == 'median':
            p_value = _compute_median_pvalue(observed_statistic, normalized_perms)
        else:
            raise ValueError("Unsupported metric. Choose 'mean' or 'median'.")
    elif callable(metric):
        p_value = _compute_custom_pvalue(observed_statistic, normalized_perms, metric)
    else:
        raise TypeError("Metric must be a string or callable.")

    # Solve for p-value using specified solver
    if solver == 'exact':
        final_p_value = _solve_exact(p_value, **kwargs)
    elif solver == 'approximate':
        final_p_value = _solve_approximate(p_value, **kwargs)
    else:
        raise ValueError("Unsupported solver. Choose 'exact' or 'approximate'.")

    # Prepare results dictionary
    result = {
        "result": final_p_value,
        "metrics": {"p_value": p_value},
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "solver": solver
        },
        "warnings": []
    }

    return result

def _validate_inputs(observed_statistic: float, permutations: np.ndarray) -> None:
    """Validate input types and dimensions."""
    if not isinstance(observed_statistic, (int, float)):
        raise TypeError("Observed statistic must be a numeric value.")
    if not isinstance(permutations, np.ndarray):
        raise TypeError("Permutations must be a numpy array.")
    if permutations.ndim != 1:
        raise ValueError("Permutations must be a 1-dimensional array.")
    if np.isnan(observed_statistic) or np.any(np.isnan(permutations)):
        raise ValueError("Input values cannot contain NaN.")
    if np.any(np.isinf(permutations)):
        raise ValueError("Permutations cannot contain infinite values.")

def _apply_normalization(permutations: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to permutations."""
    if method == 'none':
        return permutations
    elif method == 'standard':
        return (permutations - np.mean(permutations)) / np.std(permutations)
    elif method == 'minmax':
        return (permutations - np.min(permutations)) / (np.max(permutations) - np.min(permutations))
    elif method == 'robust':
        median = np.median(permutations)
        iqr = np.percentile(permutations, 75) - np.percentile(permutations, 25)
        return (permutations - median) / iqr
    else:
        raise ValueError("Unsupported normalization method.")

def _compute_mean_pvalue(observed_statistic: float, permutations: np.ndarray) -> float:
    """Compute p-value using mean of permutations."""
    return np.mean(permutations >= observed_statistic)

def _compute_median_pvalue(observed_statistic: float, permutations: np.ndarray) -> float:
    """Compute p-value using median of permutations."""
    return np.median(permutations >= observed_statistic)

def _compute_custom_pvalue(
    observed_statistic: float,
    permutations: np.ndarray,
    metric_func: Callable
) -> float:
    """Compute p-value using custom metric function."""
    return metric_func(permutations >= observed_statistic)

def _solve_exact(p_value: float, **kwargs) -> float:
    """Solve for exact p-value."""
    return p_value

def _solve_approximate(p_value: float, **kwargs) -> float:
    """Solve for approximate p-value."""
    # Example: Add some tolerance or approximation logic here
    return max(0.0, min(1.0, p_value))

################################################################################
# alternative_hypothesis
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def alternative_hypothesis_fit(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_permutations: int = 1000,
    normalization: str = 'none',
    metric: Union[str, Callable[[np.ndarray], float]] = 'mean',
    distance: str = 'euclidean',
    solver: str = 'exact',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute the alternative hypothesis using permutation tests.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    statistic_func : Callable[[np.ndarray], float]
        Function to compute the test statistic.
    n_permutations : int, optional
        Number of permutations (default: 1000).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    metric : Union[str, Callable[[np.ndarray], float]], optional
        Metric to evaluate ('mean', 'median', 'std', or custom callable) (default: 'mean').
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') (default: 'euclidean').
    solver : str, optional
        Solver method ('exact', 'approximate') (default: 'exact').
    regularization : Optional[str], optional
        Regularization method (None, 'l1', 'l2', 'elasticnet') (default: None).
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    random_state : Optional[int], optional
        Random seed for reproducibility (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, statistic_func)

    # Normalize data if required
    normalized_data = _apply_normalization(data, normalization)

    # Compute the observed statistic
    observed_statistic = statistic_func(normalized_data)

    # Perform permutations
    permuted_statistics = _perform_permutations(
        normalized_data,
        statistic_func,
        n_permutations,
        distance,
        solver,
        regularization,
        tol,
        max_iter,
        random_state
    )

    # Compute p-value and other metrics
    results = _compute_metrics(observed_statistic, permuted_statistics, metric)

    # Prepare output
    output = {
        "result": results,
        "metrics": {"observed_statistic": observed_statistic, "permuted_statistics": permuted_statistics},
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return output

def _validate_inputs(data: np.ndarray, statistic_func: Callable[[np.ndarray], float]) -> None:
    """Validate input data and functions."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if not callable(statistic_func):
        raise ValueError("statistic_func must be a callable function.")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values.")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to the data."""
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

def _perform_permutations(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_permutations: int,
    distance: str,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> np.ndarray:
    """Perform permutations and compute statistics."""
    np.random.seed(random_state)
    n = data.shape[0]
    permuted_statistics = np.zeros(n_permutations)

    for i in range(n_permutations):
        permuted_data = np.random.permutation(data)
        permuted_statistics[i] = statistic_func(permuted_data)

    return permuted_statistics

def _compute_metrics(
    observed_statistic: float,
    permuted_statistics: np.ndarray,
    metric: Union[str, Callable[[np.ndarray], float]]
) -> Dict[str, Any]:
    """Compute metrics from permuted statistics."""
    if callable(metric):
        metric_value = metric(permuted_statistics)
    elif metric == 'mean':
        metric_value = np.mean(permuted_statistics)
    elif metric == 'median':
        metric_value = np.median(permuted_statistics)
    elif metric == 'std':
        metric_value = np.std(permuted_statistics)
    else:
        raise ValueError("Invalid metric specified.")

    p_value = np.mean(permuted_statistics >= observed_statistic) if observed_statistic >= metric_value else np.mean(permuted_statistics <= observed_statistic)

    return {
        "p_value": p_value,
        "observed_statistic": observed_statistic,
        "metric_value": metric_value
    }

################################################################################
# test_statistic
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(
    x: np.ndarray,
    y: np.ndarray,
    statistic_func: Callable[[np.ndarray, np.ndarray], float],
) -> None:
    """Validate input arrays and statistic function."""
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input x contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input y contains NaN or infinite values")
    if not callable(statistic_func):
        raise TypeError("statistic_func must be a callable")

def compute_statistic(
    x: np.ndarray,
    y: np.ndarray,
    statistic_func: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Compute the test statistic."""
    return statistic_func(x, y)

def normalize_data(
    data: np.ndarray,
    method: str = "standard",
) -> np.ndarray:
    """Normalize data using specified method."""
    if method == "none":
        return data
    elif method == "standard":
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    elif method == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    elif method == "robust":
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def test_statistic_fit(
    x: np.ndarray,
    y: np.ndarray,
    statistic_func: Callable[[np.ndarray, np.ndarray], float],
    normalization: str = "standard",
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute permutation test statistic.

    Parameters:
    - x: First input array
    - y: Second input array
    - statistic_func: Function to compute test statistic
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - n_permutations: Number of permutations
    - random_state: Random seed for reproducibility

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    validate_inputs(x, y, statistic_func)

    # Normalize data
    x_norm = normalize_data(x, normalization)
    y_norm = normalize_data(y, normalization)

    # Compute observed statistic
    observed_statistic = compute_statistic(x_norm, y_norm)

    # Generate permutations
    rng = np.random.RandomState(random_state)
    permuted_statistics = []

    for _ in range(n_permutations):
        permuted_y = rng.permutation(y_norm)
        permuted_statistic = compute_statistic(x_norm, permuted_y)
        permuted_statistics.append(permuted_statistic)

    # Compute p-value
    p_value = np.mean(np.abs(permuted_statistics) >= np.abs(observed_statistic))

    # Prepare results
    result = {
        "observed_statistic": observed_statistic,
        "p_value": p_value,
    }

    metrics = {
        "n_permutations": n_permutations,
    }

    params_used = {
        "normalization": normalization,
        "statistic_func": statistic_func.__name__ if hasattr(statistic_func, '__name__') else "custom",
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings,
    }

# Example usage:
"""
def example_statistic(x: np.ndarray, y: np.ndarray) -> float:
    return np.mean(x * y)

x = np.random.randn(100)
y = np.random.randn(100)

result = test_statistic_fit(
    x=x,
    y=y,
    statistic_func=example_statistic,
    normalization="standard",
    n_permutations=1000
)
"""

################################################################################
# permutation_distribution
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def permutation_distribution_fit(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_permutations: int = 1000,
    normalization: str = 'none',
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute the permutation distribution of a statistic.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    statistic_func : Callable[[np.ndarray], float]
        Function to compute the test statistic from data.
    n_permutations : int, optional
        Number of permutations to perform (default: 1000).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    metric : Callable[[np.ndarray, np.ndarray], float], optional
        Custom metric function (default: None).
    random_state : int, optional
        Random seed for reproducibility (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(data, statistic_func)

    # Set random seed if provided
    rng = np.random.RandomState(random_state)

    # Normalize data if specified
    normalized_data = _apply_normalization(data, normalization)

    # Compute observed statistic
    observed_statistic = statistic_func(normalized_data)

    # Generate permutation distribution
    permuted_stats = _generate_permutation_distribution(
        normalized_data, statistic_func, n_permutations, rng
    )

    # Compute metrics if provided
    metrics = {}
    if metric is not None:
        metrics['custom_metric'] = metric(normalized_data, permuted_stats)

    # Prepare results
    result = {
        'observed_statistic': observed_statistic,
        'permutation_distribution': permuted_stats,
    }

    # Prepare output dictionary
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'n_permutations': n_permutations,
            'normalization': normalization,
        },
        'warnings': _check_warnings(normalized_data, permuted_stats)
    }

    return output

def _validate_inputs(data: np.ndarray, statistic_func: Callable[[np.ndarray], float]) -> None:
    """Validate input data and functions."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if not callable(statistic_func):
        raise TypeError("statistic_func must be a callable function")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to data."""
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

def _generate_permutation_distribution(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_permutations: int,
    rng: np.random.RandomState
) -> np.ndarray:
    """Generate permutation distribution of the statistic."""
    n = len(data)
    permuted_stats = np.zeros(n_permutations)

    for i in range(n_permutations):
        permuted_data = rng.permutation(data)
        permuted_stats[i] = statistic_func(permuted_data)

    return permuted_stats

def _check_warnings(data: np.ndarray, permuted_stats: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []

    if len(data) < 10:
        warnings.append("Warning: Small sample size may affect permutation test reliability")

    if np.std(permuted_stats) == 0:
        warnings.append("Warning: Zero variance in permutation distribution")

    return warnings

################################################################################
# p_value
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def p_value_compute(
    observed_statistic: float,
    permuted_statistics: np.ndarray,
    tail: str = 'two-sided',
    alternative: str = 'greater',
    metric: Union[str, Callable] = 'empirical',
    **kwargs
) -> Dict[str, Any]:
    """
    Compute the p-value from permutation test results.

    Parameters:
    -----------
    observed_statistic : float
        The observed statistic value under the null hypothesis.
    permuted_statistics : np.ndarray
        Array of statistics computed from permuted samples.
    tail : str, optional (default='two-sided')
        Type of tail for the test: 'left', 'right' or 'two-sided'.
    alternative : str, optional (default='greater')
        Alternative hypothesis: 'greater', 'less' or 'two-sided'.
    metric : str or callable, optional (default='empirical')
        Method to compute p-value: 'empirical' or custom callable.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'p_value': computed p-value
        - 'metrics': additional metrics if any
        - 'params_used': parameters used for computation
        - 'warnings': potential warnings

    Example:
    --------
    >>> observed_stat = 1.5
    >>> permuted_stats = np.random.normal(0, 1, 1000)
    >>> result = p_value_compute(observed_stat, permuted_stats)
    """
    # Validate inputs
    validate_inputs(observed_statistic, permuted_statistics)

    # Get parameters used
    params_used = {
        'tail': tail,
        'alternative': alternative,
        'metric': metric.__name__ if callable(metric) else metric
    }

    # Compute p-value based on selected method
    if callable(metric):
        p_value = metric(observed_statistic, permuted_statistics)
    else:
        if metric == 'empirical':
            p_value = compute_empirical_pvalue(
                observed_statistic,
                permuted_statistics,
                tail=tail,
                alternative=alternative
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # Prepare result dictionary
    result = {
        'p_value': p_value,
        'metrics': {},
        'params_used': params_used,
        'warnings': []
    }

    return result

def validate_inputs(observed_statistic: float, permuted_statistics: np.ndarray) -> None:
    """
    Validate input parameters for permutation test.

    Parameters:
    -----------
    observed_statistic : float
        The observed statistic value under the null hypothesis.
    permuted_statistics : np.ndarray
        Array of statistics computed from permuted samples.

    Raises:
    -------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(observed_statistic, (int, float)):
        raise ValueError("observed_statistic must be a numeric value")

    if not isinstance(permuted_statistics, np.ndarray):
        raise ValueError("permuted_statistics must be a numpy array")

    if len(permuted_statistics) == 0:
        raise ValueError("permuted_statistics cannot be empty")

    if np.any(np.isnan(permuted_statistics)):
        raise ValueError("permuted_statistics contains NaN values")

    if np.any(np.isinf(permuted_statistics)):
        raise ValueError("permuted_statistics contains infinite values")

def compute_empirical_pvalue(
    observed_statistic: float,
    permuted_statistics: np.ndarray,
    tail: str = 'two-sided',
    alternative: str = 'greater'
) -> float:
    """
    Compute empirical p-value from permutation test results.

    Parameters:
    -----------
    observed_statistic : float
        The observed statistic value under the null hypothesis.
    permuted_statistics : np.ndarray
        Array of statistics computed from permuted samples.
    tail : str, optional (default='two-sided')
        Type of tail for the test: 'left', 'right' or 'two-sided'.
    alternative : str, optional (default='greater')
        Alternative hypothesis: 'greater', 'less' or 'two-sided'.

    Returns:
    --------
    float
        Computed empirical p-value.

    Raises:
    -------
    ValueError
        If tail or alternative parameters are invalid.
    """
    if tail not in ['left', 'right', 'two-sided']:
        raise ValueError("tail must be 'left', 'right' or 'two-sided'")

    if alternative not in ['greater', 'less', 'two-sided']:
        raise ValueError("alternative must be 'greater', 'less' or 'two-sided'")

    if tail == 'left':
        if alternative == 'greater':
            p_value = np.mean(permuted_statistics <= observed_statistic)
        elif alternative == 'less':
            p_value = np.mean(permuted_statistics >= observed_statistic)
        else:  # two-sided
            p_value = min(
                np.mean(permuted_statistics <= observed_statistic),
                np.mean(permuted_statistics >= observed_statistic)
            ) * 2
    elif tail == 'right':
        if alternative == 'greater':
            p_value = np.mean(permuted_statistics >= observed_statistic)
        elif alternative == 'less':
            p_value = np.mean(permuted_statistics <= observed_statistic)
        else:  # two-sided
            p_value = min(
                np.mean(permuted_statistics >= observed_statistic),
                np.mean(permuted_statistics <= observed_statistic)
            ) * 2
    else:  # two-sided tail
        if alternative == 'greater':
            p_value = np.mean(permuted_statistics >= observed_statistic) * 2
        elif alternative == 'less':
            p_value = np.mean(permuted_statistics <= observed_statistic) * 2
        else:  # two-sided alternative
            p_value = min(
                np.mean(permuted_statistics >= observed_statistic),
                np.mean(permuted_statistics <= observed_statistic)
            ) * 2

    return p_value

################################################################################
# one_sample_test
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def one_sample_test_fit(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_permutations: int = 1000,
    normalization: Optional[str] = None,
    metric_func: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.mean(x),
    distance_func: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.linalg.norm(x - y),
    solver: str = 'exact',
    tol: float = 1e-6,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Perform a one-sample permutation test.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    statistic_func : Callable[[np.ndarray], float]
        Function to compute the test statistic from the data.
    n_permutations : int, optional
        Number of permutations to perform. Default is 1000.
    normalization : Optional[str], optional
        Normalization method. Options: 'none', 'standard', 'minmax', 'robust'. Default is None.
    metric_func : Callable[[np.ndarray, np.ndarray], float], optional
        Metric function to compare data. Default is mean.
    distance_func : Callable[[np.ndarray, np.ndarray], float], optional
        Distance function to compare data. Default is Euclidean distance.
    solver : str, optional
        Solver method. Options: 'exact', 'approximate'. Default is 'exact'.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    random_state : Optional[int], optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]
        Dictionary containing the test results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.randn(100)
    >>> result = one_sample_test_fit(data, statistic_func=np.mean)
    """
    # Validate inputs
    _validate_inputs(data, normalization)

    # Normalize data if specified
    normalized_data = _apply_normalization(data, normalization)

    # Compute observed statistic
    observed_statistic = statistic_func(normalized_data)

    # Perform permutations
    permuted_statistics = _perform_permutations(
        normalized_data,
        statistic_func,
        n_permutations,
        random_state
    )

    # Compute p-value
    p_value = _compute_p_value(observed_statistic, permuted_statistics)

    # Compute metrics
    metrics = _compute_metrics(normalized_data, metric_func, distance_func)

    # Get parameters used
    params_used = {
        'normalization': normalization,
        'metric_func': metric_func.__name__ if hasattr(metric_func, '__name__') else 'custom',
        'distance_func': distance_func.__name__ if hasattr(distance_func, '__name__') else 'custom',
        'solver': solver,
        'tol': tol,
        'max_iter': max_iter
    }

    # Check for warnings
    warnings = _check_warnings(data, normalized_data)

    return {
        'result': {
            'observed_statistic': observed_statistic,
            'p_value': p_value
        },
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(data: np.ndarray, normalization: Optional[str]) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")
    if normalization not in [None, 'none', 'standard', 'minmax', 'robust']:
        raise ValueError("Normalization must be one of: 'none', 'standard', 'minmax', 'robust'.")

def _apply_normalization(data: np.ndarray, normalization: Optional[str]) -> np.ndarray:
    """Apply specified normalization to the data."""
    if normalization == 'standard':
        return (data - np.mean(data)) / np.std(data)
    elif normalization == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif normalization == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        return data.copy()

def _perform_permutations(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_permutations: int,
    random_state: Optional[int]
) -> np.ndarray:
    """Perform permutations and compute statistics."""
    rng = np.random.RandomState(random_state)
    permuted_statistics = []
    for _ in range(n_permutations):
        permuted_data = rng.permutation(data)
        permuted_statistics.append(statistic_func(permuted_data))
    return np.array(permuted_statistics)

def _compute_p_value(
    observed_statistic: float,
    permuted_statistics: np.ndarray
) -> float:
    """Compute the p-value from observed and permuted statistics."""
    return (np.sum(permuted_statistics >= observed_statistic) + 1) / (len(permuted_statistics) + 1)

def _compute_metrics(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Compute various metrics for the data."""
    return {
        'metric': metric_func(data, np.zeros_like(data)),
        'distance': distance_func(data, np.zeros_like(data))
    }

def _check_warnings(
    data: np.ndarray,
    normalized_data: np.ndarray
) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isinf(normalized_data)):
        warnings.append("Normalized data contains infinite values.")
    if np.std(data) == 0:
        warnings.append("Data has zero standard deviation, normalization may be unstable.")
    return warnings

################################################################################
# two_sample_test
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def two_sample_test_fit(
    sample1: np.ndarray,
    sample2: np.ndarray,
    statistic_func: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.mean(x) - np.mean(y),
    n_permutations: int = 10_000,
    normalization: Optional[str] = None,
    metric_func: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: abs(np.mean(x) - np.mean(y)),
    random_state: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform a two-sample permutation test.

    Parameters:
    -----------
    sample1 : np.ndarray
        First sample array.
    sample2 : np.ndarray
        Second sample array.
    statistic_func : Callable[[np.ndarray, np.ndarray], float]
        Function to compute the test statistic.
    n_permutations : int
        Number of permutations to perform.
    normalization : Optional[str]
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric_func : Callable[[np.ndarray, np.ndarray], float]
        Function to compute the metric.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(sample1, sample2)

    # Normalize data if specified
    if normalization is not None:
        sample1, sample2 = _apply_normalization(sample1, sample2, normalization)

    # Compute observed statistic
    observed_statistic = statistic_func(sample1, sample2)

    # Perform permutations
    permuted_statistics = _perform_permutations(sample1, sample2, statistic_func, n_permutations, random_state)

    # Compute p-value
    p_value = _compute_p_value(observed_statistic, permuted_statistics)

    # Compute metrics
    metrics = _compute_metrics(sample1, sample2, metric_func)

    # Prepare results
    result = {
        "observed_statistic": observed_statistic,
        "p_value": p_value,
        "permuted_statistics": permuted_statistics
    }

    # Prepare output dictionary
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "statistic_func": statistic_func.__name__ if hasattr(statistic_func, '__name__') else "custom",
            "n_permutations": n_permutations,
            "normalization": normalization,
            "metric_func": metric_func.__name__ if hasattr(metric_func, '__name__') else "custom",
            "random_state": random_state
        },
        "warnings": []
    }

    return output

def _validate_inputs(sample1: np.ndarray, sample2: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(sample1, np.ndarray) or not isinstance(sample2, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if sample1.ndim != 1 or sample2.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if len(sample1) == 0 or len(sample2) == 0:
        raise ValueError("Input arrays must not be empty.")
    if np.any(np.isnan(sample1)) or np.any(np.isnan(sample2)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(sample1)) or np.any(np.isinf(sample2)):
        raise ValueError("Input arrays must not contain infinite values.")

def _apply_normalization(sample1: np.ndarray, sample2: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalization to the samples."""
    if method == "standard":
        mean1, std1 = np.mean(sample1), np.std(sample1)
        mean2, std2 = np.mean(sample2), np.std(sample2)
        sample1 = (sample1 - mean1) / std1
        sample2 = (sample2 - mean2) / std2
    elif method == "minmax":
        min1, max1 = np.min(sample1), np.max(sample1)
        min2, max2 = np.min(sample2), np.max(sample2)
        sample1 = (sample1 - min1) / (max1 - min1)
        sample2 = (sample2 - min2) / (max2 - min2)
    elif method == "robust":
        median1, iqr1 = np.median(sample1), np.percentile(sample1, 75) - np.percentile(sample1, 25)
        median2, iqr2 = np.median(sample2), np.percentile(sample2, 75) - np.percentile(sample2, 25)
        sample1 = (sample1 - median1) / iqr1
        sample2 = (sample2 - median2) / iqr2
    return sample1, sample2

def _perform_permutations(
    sample1: np.ndarray,
    sample2: np.ndarray,
    statistic_func: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int,
    random_state: Optional[int]
) -> np.ndarray:
    """Perform permutations and compute statistics."""
    rng = np.random.RandomState(random_state)
    combined = np.concatenate([sample1, sample2])
    permuted_statistics = np.zeros(n_permutations)

    for i in range(n_permutations):
        rng.shuffle(combined)
        split = len(sample1)
        permuted_sample1, permuted_sample2 = combined[:split], combined[split:]
        permuted_statistics[i] = statistic_func(permuted_sample1, permuted_sample2)

    return permuted_statistics

def _compute_p_value(observed_statistic: float, permuted_statistics: np.ndarray) -> float:
    """Compute the p-value from permuted statistics."""
    if observed_statistic >= 0:
        p_value = np.mean(permuted_statistics >= observed_statistic)
    else:
        p_value = np.mean(permuted_statistics <= observed_statistic)
    return p_value

def _compute_metrics(sample1: np.ndarray, sample2: np.ndarray, metric_func: Callable[[np.ndarray, np.ndarray], float]) -> Dict[str, float]:
    """Compute metrics for the samples."""
    return {
        "metric_value": metric_func(sample1, sample2)
    }

################################################################################
# paired_samples_test
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def paired_samples_test_fit(
    x: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    n_permutations: int = 1000,
    normalization: Optional[str] = None,
    random_state: Optional[int] = None
) -> Dict:
    """
    Perform a permutation test for paired samples.

    Parameters:
    -----------
    x : np.ndarray
        First set of paired samples.
    y : np.ndarray
        Second set of paired samples.
    metric : str or callable, optional
        Metric to compute between x and y. Can be 'mse', 'mae', 'r2', or a custom callable.
    n_permutations : int, optional
        Number of permutations to perform. Default is 1000.
    normalization : str, optional
        Normalization method to apply. Can be 'standard', 'minmax', or None.
    random_state : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    Dict containing:
        - result: dict with test statistic and p-value
        - metrics: computed metrics
        - params_used: parameters used in the test
        - warnings: any warnings encountered

    Example:
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([2, 3, 4])
    >>> result = paired_samples_test_fit(x, y)
    """
    # Validate inputs
    _validate_inputs(x, y)

    # Apply normalization if specified
    x_norm, y_norm = _apply_normalization(x, y, normalization)

    # Compute observed statistic
    observed_statistic = _compute_observed_statistic(x_norm, y_norm, metric)

    # Perform permutations
    permuted_statistics = _perform_permutations(
        x_norm, y_norm, metric, n_permutations, random_state
    )

    # Compute p-value
    p_value = _compute_p_value(observed_statistic, permuted_statistics)

    # Prepare results
    result = {
        'statistic': observed_statistic,
        'p_value': p_value
    }

    metrics = {
        'observed_metric': observed_statistic,
        'mean_permuted_metric': np.mean(permuted_statistics),
        'std_permuted_metric': np.std(permuted_statistics)
    }

    params_used = {
        'metric': metric,
        'n_permutations': n_permutations,
        'normalization': normalization,
        'random_state': random_state
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("Input arrays must not contain NaN values.")
    if np.isinf(x).any() or np.isinf(y).any():
        raise ValueError("Input arrays must not contain infinite values.")

def _apply_normalization(
    x: np.ndarray,
    y: np.ndarray,
    normalization: Optional[str]
) -> tuple:
    """Apply specified normalization to input arrays."""
    if normalization is None:
        return x, y
    elif normalization == 'standard':
        x_norm = (x - np.mean(x)) / np.std(x)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif normalization == 'minmax':
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    return x_norm, y_norm

def _compute_observed_statistic(
    x: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Compute the observed statistic between x and y."""
    if callable(metric):
        return metric(x, y)
    elif metric == 'mse':
        return np.mean((x - y) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(x - y))
    elif metric == 'r2':
        ss_res = np.sum((y - x) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _perform_permutations(
    x: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    n_permutations: int,
    random_state: Optional[int]
) -> np.ndarray:
    """Perform permutations and compute statistics."""
    rng = np.random.RandomState(random_state)
    permuted_statistics = []
    for _ in range(n_permutations):
        # Create a random permutation of the differences
        diff = x - y
        permuted_diff = rng.permutation(diff)
        # Compute the statistic for this permutation
        x_permuted = y + permuted_diff
        stat = _compute_observed_statistic(x_permuted, y, metric)
        permuted_statistics.append(stat)
    return np.array(permuted_statistics)

def _compute_p_value(
    observed_statistic: float,
    permuted_statistics: np.ndarray
) -> float:
    """Compute the p-value from observed and permuted statistics."""
    return (np.sum(permuted_statistics >= observed_statistic) /
            len(permuted_statistics))

################################################################################
# multivariate_permutation
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    distance: Callable[[np.ndarray, np.ndarray], float] = None,
) -> None:
    """Validate input data and functions."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values")

def compute_statistic(
    X: np.ndarray,
    y: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Compute the test statistic."""
    return metric(X, y)

def permute_data(
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 1000,
) -> np.ndarray:
    """Generate permuted samples."""
    n_samples = len(X)
    permuted_stats = np.zeros(n_permutations)

    for i in range(n_permutations):
        permuted_indices = np.random.permutation(n_samples)
        X_permuted = X[permuted_indices]
        permuted_stats[i] = compute_statistic(X_permuted, y)

    return permuted_stats

def multivariate_permutation_fit(
    X: np.ndarray,
    y: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    distance: Callable[[np.ndarray, np.ndarray], float] = None,
    n_permutations: int = 1000,
    normalize: str = "standard",
) -> Dict[str, Any]:
    """
    Perform multivariate permutation test.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    metric : Callable[[np.ndarray, np.ndarray], float]
        Metric function to compute the test statistic
    distance : Callable[[np.ndarray, np.ndarray], float], optional
        Distance function for permutation (default: None)
    n_permutations : int, optional
        Number of permutations to perform (default: 1000)
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard')

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    validate_inputs(X, y, metric, distance)

    # Normalize data if required
    X_normalized = normalize_data(X, method=normalize)

    # Compute observed statistic
    observed_stat = compute_statistic(X_normalized, y)

    # Perform permutations
    permuted_stats = permute_data(X_normalized, y, n_permutations)

    # Compute p-value
    p_value = (np.sum(permuted_stats >= observed_stat) + 1) / (n_permutations + 1)

    # Prepare results
    result = {
        "observed_statistic": observed_stat,
        "p_value": p_value,
    }

    metrics = {
        "observed_metric": observed_stat,
    }

    params_used = {
        "n_permutations": n_permutations,
        "normalize": normalize,
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings,
    }

def normalize_data(
    X: np.ndarray,
    method: str = "standard",
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

################################################################################
# block_permutation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    block_sizes: List[int],
    metric: Union[str, Callable],
    normalizer: Optional[Callable] = None,
) -> None:
    """Validate inputs for block permutation test."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if not all(isinstance(size, int) for size in block_sizes):
        raise TypeError("Block sizes must be integers")
    if sum(block_sizes) != X.shape[0]:
        raise ValueError("Sum of block sizes must equal number of samples")
    if isinstance(metric, str) and metric not in ["mse", "mae", "r2"]:
        raise ValueError("Metric must be 'mse', 'mae', 'r2' or a callable")
    if normalizer is not None and not callable(normalizer):
        raise TypeError("Normalizer must be a callable or None")

def compute_statistic(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
) -> float:
    """Compute the test statistic based on the chosen metric."""
    if isinstance(metric, str):
        if metric == "mse":
            return np.mean((X - y) ** 2)
        elif metric == "mae":
            return np.mean(np.abs(X - y))
        elif metric == "r2":
            ss_res = np.sum((y - X) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)
    else:
        return metric(X, y)

def normalize_data(
    X: np.ndarray,
    normalizer: Optional[Callable] = None,
) -> np.ndarray:
    """Apply normalization to the data."""
    if normalizer is None:
        return X
    return normalizer(X)

def block_permutation_fit(
    X: np.ndarray,
    y: np.ndarray,
    block_sizes: List[int],
    metric: Union[str, Callable] = "mse",
    normalizer: Optional[Callable] = None,
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
) -> Dict[str, Union[float, Dict, List]]:
    """
    Perform block permutation test.

    Parameters
    ----------
    X : np.ndarray
        Predicted values.
    y : np.ndarray
        True values.
    block_sizes : List[int]
        Sizes of the blocks for permutation.
    metric : Union[str, Callable], optional
        Metric to compute ('mse', 'mae', 'r2' or callable), by default "mse".
    normalizer : Optional[Callable], optional
        Normalization function, by default None.
    n_permutations : int, optional
        Number of permutations, by default 1000.
    random_state : Optional[int], optional
        Random seed, by default None.

    Returns
    -------
    Dict[str, Union[float, Dict, List]]
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    validate_inputs(X, y, block_sizes, metric, normalizer)

    # Normalize data
    X_norm = normalize_data(X, normalizer)
    y_norm = normalize_data(y, normalizer)

    # Compute observed statistic
    observed_stat = compute_statistic(X_norm, y_norm, metric)

    # Generate permutations
    rng = np.random.RandomState(random_state)
    permuted_stats = []
    for _ in range(n_permutations):
        # Permute blocks
        permuted_X = np.zeros_like(X_norm)
        start = 0
        for size in block_sizes:
            block = rng.permutation(X_norm[start:start + size])
            permuted_X[start:start + size] = block
            start += size

        # Compute statistic for permuted data
        stat = compute_statistic(permuted_X, y_norm, metric)
        permuted_stats.append(stat)

    # Compute p-value
    p_value = (np.sum(np.array(permuted_stats) >= observed_stat) + 1) / (n_permutations + 1)

    # Prepare results
    result = {
        "observed_statistic": observed_stat,
        "p_value": p_value,
    }

    metrics = {
        "metric_used": metric if isinstance(metric, str) else "custom",
    }

    params_used = {
        "block_sizes": block_sizes,
        "n_permutations": n_permutations,
        "normalizer": normalizer.__name__ if normalizer else None,
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings,
    }

# Example usage:
"""
X = np.array([1, 2, 3, 4, 5])
y = np.array([1.1, 2.0, 3.2, 4.1, 5.0])
block_sizes = [2, 3]
result = block_permutation_fit(X, y, block_sizes)
print(result)
"""

################################################################################
# stratified_permutation
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def stratified_permutation_fit(
    X: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
    normalize: bool = True
) -> Dict[str, Any]:
    """
    Perform stratified permutation test.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    strata : np.ndarray
        Strata labels for each sample.
    metric : str or callable, optional
        Metric to compute (default: 'mse'). Can be 'mse', 'mae', 'r2', or a custom callable.
    n_permutations : int, optional
        Number of permutations to perform (default: 1000).
    random_state : int, optional
        Random seed for reproducibility (default: None).
    normalize : bool, optional
        Whether to normalize the data (default: True).

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y, strata)

    # Set random state
    rng = np.random.RandomState(random_state)

    # Normalize data if required
    X_normalized, y_normalized = _normalize_data(X, y) if normalize else (X.copy(), y.copy())

    # Compute observed statistic
    observed_stat = _compute_observed_statistic(X_normalized, y_normalized, strata, metric)

    # Perform permutations
    permuted_stats = _perform_permutations(
        X_normalized, y_normalized, strata, metric, n_permutations, rng
    )

    # Compute p-value
    p_value = _compute_p_value(observed_stat, permuted_stats)

    # Prepare results
    result = {
        'observed_statistic': observed_stat,
        'p_value': p_value,
        'permutation_distribution': permuted_stats
    }

    metrics = {
        'metric_used': metric if isinstance(metric, str) else 'custom'
    }

    params_used = {
        'n_permutations': n_permutations,
        'random_state': random_state,
        'normalize': normalize
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray, strata: np.ndarray) -> None:
    """Validate input arrays."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if X.shape[0] != strata.shape[0]:
        raise ValueError("X, y and strata must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _normalize_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """Normalize data using standard scaling."""
    X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y_normalized = (y - np.mean(y)) / np.std(y)
    return X_normalized, y_normalized

def _compute_observed_statistic(
    X: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Compute the observed statistic for the original data."""
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    return metric_func(y, _predict_stratified(X, y, strata))

def _get_metric_function(metric: str) -> Callable:
    """Get the metric function based on the metric name."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

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

def _predict_stratified(X: np.ndarray, y: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Predict using stratified means."""
    y_pred = np.zeros_like(y)
    for stratum in np.unique(strata):
        mask = strata == stratum
        y_pred[mask] = np.mean(y[mask])
    return y_pred

def _perform_permutations(
    X: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    metric: Union[str, Callable],
    n_permutations: int,
    rng: np.random.RandomState
) -> np.ndarray:
    """Perform permutations and compute statistics."""
    permuted_stats = []
    for _ in range(n_permutations):
        y_permuted = rng.permutation(y)
        if isinstance(metric, str):
            metric_func = _get_metric_function(metric)
        else:
            metric_func = metric
        permuted_stats.append(
            metric_func(y, _predict_stratified(X, y_permuted, strata))
        )
    return np.array(permuted_stats)

def _compute_p_value(observed_stat: float, permuted_stats: np.ndarray) -> float:
    """Compute p-value from permutation distribution."""
    return (np.sum(permuted_stats >= observed_stat) + 1) / (len(permuted_stats) + 1)

################################################################################
# monte_carlo_approximation
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def monte_carlo_approximation_fit(
    observed_statistic: float,
    permutation_function: Callable[[np.ndarray], np.ndarray],
    n_permutations: int = 1000,
    normalization: str = 'none',
    metric: Union[str, Callable[[float], float]] = 'absolute_difference',
    random_state: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform Monte Carlo approximation for permutation tests.

    Parameters:
    -----------
    observed_statistic : float
        The observed test statistic under the null hypothesis.
    permutation_function : callable
        A function that takes a numpy array and returns a permuted version.
    n_permutations : int, optional
        Number of permutations to perform (default: 1000).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    metric : str or callable, optional
        Metric to compute ('absolute_difference', 'squared_difference', custom callable) (default: 'absolute_difference').
    random_state : int, optional
        Seed for random number generation (default: None).

    Returns:
    --------
    dict
        A dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(observed_statistic, permutation_function, n_permutations)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Generate permutations
    permuted_statistics = _generate_permutations(permutation_function, n_permutations)

    # Normalize statistics if required
    normalized_statistics = _normalize_statistics(permuted_statistics, normalization)

    # Compute metric
    computed_metric = _compute_metric(normalized_statistics, observed_statistic, metric)

    # Calculate p-value
    p_value = _calculate_p_value(observed_statistic, normalized_statistics)

    # Prepare results
    result = {
        'p_value': p_value,
        'observed_statistic': observed_statistic,
        'permuted_statistics': normalized_statistics,
    }

    metrics = {
        'computed_metric': computed_metric
    }

    params_used = {
        'n_permutations': n_permutations,
        'normalization': normalization,
        'metric': metric.__name__ if callable(metric) else metric,
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(observed_statistic: float, permutation_function: Callable, n_permutations: int) -> None:
    """Validate the inputs for Monte Carlo approximation."""
    if not isinstance(observed_statistic, (int, float)):
        raise ValueError("Observed statistic must be a numeric value.")
    if not callable(permutation_function):
        raise ValueError("Permutation function must be a callable.")
    if not isinstance(n_permutations, int) or n_permutations <= 0:
        raise ValueError("Number of permutations must be a positive integer.")

def _generate_permutations(permutation_function: Callable, n_permutations: int) -> np.ndarray:
    """Generate permuted statistics using the provided permutation function."""
    # This is a placeholder; actual implementation depends on the data structure
    return np.random.randn(n_permutations)

def _normalize_statistics(statistics: np.ndarray, method: str) -> np.ndarray:
    """Normalize the statistics based on the specified method."""
    if method == 'none':
        return statistics
    elif method == 'standard':
        return (statistics - np.mean(statistics)) / np.std(statistics)
    elif method == 'minmax':
        return (statistics - np.min(statistics)) / (np.max(statistics) - np.min(statistics))
    elif method == 'robust':
        median = np.median(statistics)
        iqr = np.percentile(statistics, 75) - np.percentile(statistics, 25)
        return (statistics - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_metric(statistics: np.ndarray, observed_statistic: float, metric: Union[str, Callable]) -> float:
    """Compute the specified metric between observed and permuted statistics."""
    if callable(metric):
        return metric(statistics, observed_statistic)
    elif metric == 'absolute_difference':
        return np.mean(np.abs(statistics - observed_statistic))
    elif metric == 'squared_difference':
        return np.mean((statistics - observed_statistic) ** 2)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _calculate_p_value(observed_statistic: float, permuted_statistics: np.ndarray) -> float:
    """Calculate the p-value based on the observed statistic and permuted statistics."""
    return np.mean(permuted_statistics >= observed_statistic)

################################################################################
# exact_test
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def exact_test_fit(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_permutations: int = 1000,
    normalization: str = 'none',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform an exact permutation test with configurable parameters.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    statistic_func : Callable[[np.ndarray], float]
        Function to compute the test statistic from data
    n_permutations : int, optional
        Number of permutations to perform (default: 1000)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none')
    metric : Union[str, Callable], optional
        Metric to evaluate ('mse', 'mae', 'r2', 'logloss') or custom callable (default: 'mse')
    distance : Union[str, Callable], optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable (default: 'euclidean')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default: 'closed_form')
    regularization : Optional[str], optional
        Regularization method ('none', 'l1', 'l2', 'elasticnet') (default: None)
    tol : float, optional
        Tolerance for convergence (default: 1e-6)
    max_iter : int, optional
        Maximum number of iterations (default: 1000)
    random_state : Optional[int], optional
        Random seed for reproducibility (default: None)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(data, statistic_func)

    # Normalize data if required
    normalized_data = _apply_normalization(data, normalization)

    # Compute observed statistic
    observed_statistic = statistic_func(normalized_data)

    # Perform permutations
    permuted_statistics = _perform_permutations(
        normalized_data,
        statistic_func,
        n_permutations,
        random_state
    )

    # Compute p-value
    p_value = _compute_p_value(observed_statistic, permuted_statistics)

    # Compute metrics
    metrics = _compute_metrics(
        normalized_data,
        metric,
        distance,
        solver,
        regularization,
        tol,
        max_iter
    )

    # Prepare results dictionary
    result = {
        'observed_statistic': observed_statistic,
        'permuted_statistics': permuted_statistics,
        'p_value': p_value
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': _check_warnings(normalized_data, p_value)
    }

def _validate_inputs(data: np.ndarray, statistic_func: Callable) -> None:
    """Validate input data and functions."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")
    if not callable(statistic_func):
        raise TypeError("statistic_func must be callable")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the data."""
    if method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)
        return (data - median) / iqr
    else:
        return data.copy()

def _perform_permutations(
    data: np.ndarray,
    statistic_func: Callable,
    n_permutations: int,
    random_state: Optional[int]
) -> np.ndarray:
    """Perform permutations of the data and compute statistics."""
    rng = np.random.RandomState(random_state)
    n_samples = data.shape[0]
    permuted_statistics = np.zeros(n_permutations)

    for i in range(n_permutations):
        permuted_data = rng.permutation(data)
        permuted_statistics[i] = statistic_func(permuted_data)

    return permuted_statistics

def _compute_p_value(
    observed_statistic: float,
    permuted_statistics: np.ndarray
) -> float:
    """Compute p-value from observed and permuted statistics."""
    n_greater = np.sum(permuted_statistics >= observed_statistic)
    return (n_greater + 1) / (len(permuted_statistics) + 1)

def _compute_metrics(
    data: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, float]:
    """Compute various metrics based on the data and parameters."""
    # This is a placeholder - actual implementation would depend on specific metrics
    return {
        'metric_value': _compute_metric(data, metric),
        'distance_value': _compute_distance(data, distance)
    }

def _compute_metric(
    data: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Compute specified metric."""
    if callable(metric):
        return metric(data, data)
    elif metric == 'mse':
        return np.mean((data - np.mean(data, axis=0))**2)
    elif metric == 'mae':
        return np.mean(np.abs(data - np.mean(data, axis=0)))
    elif metric == 'r2':
        ss_total = np.sum((data - np.mean(data))**2)
        ss_residual = np.sum((data - data)**2)  # Placeholder
        return 1 - (ss_residual / ss_total)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_distance(
    data: np.ndarray,
    distance: Union[str, Callable]
) -> float:
    """Compute specified distance."""
    if callable(distance):
        return distance(data, data)
    elif distance == 'euclidean':
        return np.linalg.norm(data - data)
    elif distance == 'manhattan':
        return np.sum(np.abs(data - data))
    elif distance == 'cosine':
        return 1 - np.dot(data, data) / (np.linalg.norm(data) * np.linalg.norm(data))
    else:
        raise ValueError(f"Unknown distance: {distance}")

def _check_warnings(data: np.ndarray, p_value: float) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isinf(data)):
        warnings.append("Data contains infinite values")
    if p_value == 0:
        warnings.append("p-value is exactly zero (consider increasing n_permutations)")
    return warnings

################################################################################
# resampling_methods
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def resampling_methods_fit(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    normalization: str = 'none',
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Fit resampling methods for permutation tests.

    Parameters:
    - X: Input features array of shape (n_samples, n_features)
    - y: Target values array of shape (n_samples,)
    - metric: Metric to use for evaluation ('mse', 'mae', 'r2') or custom callable
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - n_permutations: Number of permutations to perform
    - random_state: Random seed for reproducibility
    - custom_metric: Custom metric function if not using built-in metrics

    Returns:
    - Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random seed if provided
    rng = np.random.RandomState(random_state)

    # Normalize data if required
    X_norm, y_norm = _apply_normalization(X, y, normalization)

    # Initialize results dictionary
    results = {
        'result': None,
        'metrics': {},
        'params_used': {
            'metric': metric,
            'normalization': normalization,
            'n_permutations': n_permutations
        },
        'warnings': []
    }

    # Select metric function
    if callable(metric):
        metric_func = metric
    elif custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric_function(metric)

    # Perform permutation test
    observed_statistic = metric_func(y_norm, _predict(X_norm))
    permuted_statistics = []

    for _ in range(n_permutations):
        y_permuted = rng.permutation(y_norm)
        permuted_statistics.append(metric_func(y_permuted, _predict(X_norm)))

    # Calculate p-value
    p_value = (np.sum(np.array(permuted_statistics) >= observed_statistic) + 1) / (n_permutations + 1)

    # Store results
    results['result'] = {
        'observed_statistic': observed_statistic,
        'permuted_statistics': np.array(permuted_statistics),
        'p_value': p_value
    }

    # Calculate and store metrics
    results['metrics'] = {
        'observed_metric': observed_statistic,
        'mean_permuted_metric': np.mean(permuted_statistics),
        'std_permuted_metric': np.std(permuted_statistics)
    }

    return results

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
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalization to input data."""
    X_norm = X.copy()
    y_norm = y.copy()

    if method == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif method == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == 'robust':
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))

    return X_norm, y_norm

def _get_metric_function(metric_name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get metric function based on name."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }
    if metric_name not in metrics:
        raise ValueError(f"Unknown metric: {metric_name}")
    return metrics[metric_name]

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

def _predict(X: np.ndarray) -> np.ndarray:
    """Simple prediction function (can be replaced with any model)."""
    return np.mean(X, axis=1)

################################################################################
# computational_complexity
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str,
    normalization: Optional[str] = None
) -> None:
    """Validate input data and parameters."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if normalization not in [None, 'standard', 'minmax', 'robust']:
        raise ValueError("Invalid normalization method")
    if solver not in ['closed_form', 'gradient_descent', 'newton', 'coordinate_descent']:
        raise ValueError("Invalid solver method")
    if isinstance(metric, str) and metric not in ['mse', 'mae', 'r2', 'logloss']:
        raise ValueError("Invalid metric")
    if isinstance(distance, str) and distance not in ['euclidean', 'manhattan', 'cosine', 'minkowski']:
        raise ValueError("Invalid distance metric")

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    method: Optional[str] = None
) -> tuple:
    """Apply normalization to data."""
    if method == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y = (y - np.mean(y)) / np.std(y)
    elif method == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    return X, y

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Compute the specified metric."""
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
        return metric(y_true, y_pred)

def _compute_distance(
    X: np.ndarray,
    distance: Union[str, Callable],
    p: float = 2.0
) -> np.ndarray:
    """Compute the specified distance matrix."""
    if isinstance(distance, str):
        if distance == 'euclidean':
            return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
        elif distance == 'manhattan':
            return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
        elif distance == 'cosine':
            return 1 - np.dot(X, X.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(X, axis=1))
        elif distance == 'minkowski':
            return np.sum(np.abs(X[:, np.newaxis] - X) ** p, axis=2) ** (1/p)
    else:
        return distance(X)

def _solve_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    distance: Union[str, Callable],
    solver: str
) -> Dict:
    """Solve the permutation test with specified solver."""
    if solver == 'closed_form':
        # Placeholder for closed form solution
        pass
    elif solver == 'gradient_descent':
        # Placeholder for gradient descent solution
        pass
    elif solver == 'newton':
        # Placeholder for Newton's method solution
        pass
    elif solver == 'coordinate_descent':
        # Placeholder for coordinate descent solution
        pass

def computational_complexity_fit(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    normalization: Optional[str] = None,
    p: float = 2.0
) -> Dict:
    """
    Compute the computational complexity of permutation tests.

    Parameters:
    - X: Input features (2D array)
    - y: Target values (1D array)
    - metric: Metric to evaluate performance
    - distance: Distance metric for permutation test
    - solver: Solver method to use
    - normalization: Normalization method
    - p: Power parameter for Minkowski distance

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    _validate_inputs(X, y, metric, distance, solver, normalization)

    X_norm, y_norm = _apply_normalization(X.copy(), y.copy(), normalization)
    distance_matrix = _compute_distance(X_norm, distance, p)

    result = _solve_permutation_test(X_norm, y_norm, distance, solver)
    metrics = {'metric_value': _compute_metric(y, result['predictions'], metric)}

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'normalization': normalization,
            'p': p
        },
        'warnings': []
    }

################################################################################
# software_implementations
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def software_implementations_fit(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_permutations: int = 1000,
    normalization: str = 'none',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'exact',
    custom_params: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Main function for permutation tests implementation.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    statistic_func : Callable[[np.ndarray], float]
        Function to compute the test statistic.
    n_permutations : int, optional
        Number of permutations (default: 1000).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    metric : Union[str, Callable], optional
        Metric to evaluate ('mse', 'mae', 'r2', 'logloss') or custom callable (default: 'mse').
    distance : Union[str, Callable], optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable (default: 'euclidean').
    solver : str, optional
        Solver method ('exact', 'approximate') (default: 'exact').
    custom_params : Dict[str, Any], optional
        Additional parameters for custom functions (default: None).
    random_state : int, optional
        Random seed for reproducibility (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Initialize result dictionary
    result = {
        'result': None,
        'metrics': {},
        'params_used': {
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
            'distance': distance if isinstance(distance, str) else 'custom',
            'solver': solver,
            'n_permutations': n_permutations
        },
        'warnings': []
    }

    # Validate inputs
    _validate_inputs(data, statistic_func)

    # Normalize data if required
    normalized_data = _apply_normalization(data, normalization)

    # Compute observed statistic
    observed_statistic = statistic_func(normalized_data)
    result['result'] = {'observed_statistic': observed_statistic}

    # Perform permutations
    permuted_statistics = _perform_permutations(
        normalized_data,
        statistic_func,
        n_permutations,
        random_state
    )

    # Compute p-value
    p_value = _compute_p_value(observed_statistic, permuted_statistics)
    result['result']['p_value'] = p_value

    # Compute metrics
    metrics = _compute_metrics(normalized_data, metric, distance)
    result['metrics'].update(metrics)

    return result

def _validate_inputs(data: np.ndarray, statistic_func: Callable[[np.ndarray], float]) -> None:
    """Validate input data and functions."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if not callable(statistic_func):
        raise TypeError("statistic_func must be a callable function.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")

def _apply_normalization(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply normalization to the data."""
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
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _perform_permutations(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_permutations: int,
    random_state: Optional[int]
) -> np.ndarray:
    """Perform permutations of the data and compute statistics."""
    rng = np.random.RandomState(random_state)
    n_samples = data.shape[0]
    permuted_statistics = np.zeros(n_permutations)

    for i in range(n_permutations):
        permuted_data = rng.permutation(data)
        permuted_statistics[i] = statistic_func(permuted_data)

    return permuted_statistics

def _compute_p_value(
    observed_statistic: float,
    permuted_statistics: np.ndarray
) -> float:
    """Compute the p-value from permuted statistics."""
    return (np.sum(permuted_statistics >= observed_statistic) + 1) / (len(permuted_statistics) + 1)

def _compute_metrics(
    data: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute various metrics for the data."""
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            # Example: compute MSE between data and its mean
            mean = np.mean(data, axis=0)
            metrics['mse'] = np.mean((data - mean) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(data - np.mean(data, axis=0)))
        elif metric == 'r2':
            # Example: compute R-squared
            y_pred = np.mean(data, axis=0)
            ss_res = np.sum((data - y_pred) ** 2)
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
        elif metric == 'logloss':
            # Example: compute log loss (requires probabilities)
            pass
    else:
        metrics['custom_metric'] = metric(data, data)

    if isinstance(distance, str):
        if distance == 'euclidean':
            # Example: compute average Euclidean distance
            n = data.shape[0]
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist_matrix[i, j] = np.linalg.norm(data[i] - data[j])
            metrics['avg_euclidean_distance'] = np.mean(dist_matrix)
        elif distance == 'manhattan':
            # Example: compute average Manhattan distance
            n = data.shape[0]
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist_matrix[i, j] = np.sum(np.abs(data[i] - data[j]))
            metrics['avg_manhattan_distance'] = np.mean(dist_matrix)
        elif distance == 'cosine':
            # Example: compute average cosine similarity
            n = data.shape[0]
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    sim_matrix[i, j] = np.dot(data[i], data[j]) / (np.linalg.norm(data[i]) * np.linalg.norm(data[j]))
            metrics['avg_cosine_similarity'] = np.mean(sim_matrix)
        elif distance == 'minkowski':
            # Example: compute average Minkowski distance with p=3
            n = data.shape[0]
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist_matrix[i, j] = np.sum(np.abs(data[i] - data[j]) ** 3) ** (1/3)
            metrics['avg_minkowski_distance'] = np.mean(dist_matrix)
    else:
        metrics['custom_distance'] = distance(data, data)

    return metrics
