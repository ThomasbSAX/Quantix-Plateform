"""
Quantix – Module jackknife
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# jackknife_resampling
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def jackknife_resampling_fit(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    normalize: str = "none",
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    weights: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform jackknife resampling on the given data and compute statistics.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features).
    statistic_func : Callable[[np.ndarray], float]
        Function to compute the statistic of interest on a subset of data.
    normalize : str, optional
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : str or Callable, optional
        Metric to evaluate the statistic: "mse", "mae", "r2", or custom callable.
    weights : np.ndarray, optional
        Weights for the data points. If None, uniform weights are used.
    random_state : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, statistic_func, normalize, metric, weights)

    # Normalize data if required
    normalized_data = _normalize_data(data, normalize)

    # Initialize results dictionary
    results: Dict[str, Any] = {
        "result": None,
        "metrics": {},
        "params_used": {
            "normalize": normalize,
            "metric": metric if isinstance(metric, str) else "custom",
            "weights": weights is not None
        },
        "warnings": []
    }

    # Perform jackknife resampling
    n_samples = data.shape[0]
    jackknife_stats = np.zeros(n_samples)

    if random_state is not None:
        np.random.seed(random_state)

    for i in range(n_samples):
        # Leave out the ith sample
        jackknife_sample = np.delete(normalized_data, i, axis=0)
        if weights is not None:
            jackknife_weights = np.delete(weights, i)
        else:
            jackknife_weights = None

        # Compute statistic for the current sample
        jackknife_stats[i] = _compute_statistic(jackknife_sample, statistic_func, jackknife_weights)

    # Compute the final statistic
    results["result"] = _compute_final_statistic(jackknife_stats)

    # Compute metrics if a metric is provided
    if isinstance(metric, str):
        results["metrics"] = _compute_metrics(jackknife_stats, metric)
    elif callable(metric):
        results["metrics"] = {"custom_metric": metric(jackknife_stats, data)}

    return results

def _validate_inputs(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    normalize: str,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    weights: Optional[np.ndarray]
) -> None:
    """Validate the inputs for jackknife resampling."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if not callable(statistic_func):
        raise ValueError("statistic_func must be a callable.")
    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("normalize must be one of: 'none', 'standard', 'minmax', 'robust'.")
    if isinstance(metric, str) and metric not in ["mse", "mae", "r2"]:
        raise ValueError("metric must be one of: 'mse', 'mae', 'r2' or a custom callable.")
    if weights is not None:
        if not isinstance(weights, np.ndarray):
            raise ValueError("Weights must be a numpy array.")
        if weights.shape[0] != data.shape[0]:
            raise ValueError("Weights must have the same number of samples as data.")

def _normalize_data(
    data: np.ndarray,
    normalize: str
) -> np.ndarray:
    """Normalize the data according to the specified method."""
    if normalize == "none":
        return data
    elif normalize == "standard":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif normalize == "minmax":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif normalize == "robust":
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError("Invalid normalization method.")

def _compute_statistic(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    weights: Optional[np.ndarray] = None
) -> float:
    """Compute the statistic for a given subset of data."""
    if weights is not None:
        return np.average(data, weights=weights, axis=0)
    else:
        return statistic_func(data)

def _compute_final_statistic(
    jackknife_stats: np.ndarray
) -> float:
    """Compute the final statistic from jackknife resampling."""
    return np.mean(jackknife_stats)

def _compute_metrics(
    jackknife_stats: np.ndarray,
    metric: str
) -> Dict[str, float]:
    """Compute the specified metrics for the jackknife statistics."""
    metrics = {}
    if metric == "mse":
        metrics["mse"] = np.mean((jackknife_stats - np.mean(jackknife_stats)) ** 2)
    elif metric == "mae":
        metrics["mae"] = np.mean(np.abs(jackknife_stats - np.mean(jackknife_stats)))
    elif metric == "r2":
        ss_total = np.sum((jackknife_stats - np.mean(jackknife_stats)) ** 2)
        ss_res = np.sum((jackknife_stats - jackknife_stats.mean()) ** 2)
        metrics["r2"] = 1 - (ss_res / ss_total) if ss_total != 0 else 0.0
    return metrics

# Example usage:
"""
data = np.random.rand(10, 5)
statistic_func = lambda x: np.mean(x, axis=0).mean()
result = jackknife_resampling_fit(data, statistic_func, normalize="standard", metric="mse")
print(result)
"""

################################################################################
# bias_estimation
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    data: np.ndarray,
    model_func: Callable,
    metric_func: Callable,
    normalization: str = "none",
) -> None:
    """Validate input data and functions."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Data contains infinite values.")

def _apply_normalization(
    data: np.ndarray,
    normalization: str = "none",
) -> np.ndarray:
    """Apply selected normalization to the data."""
    if normalization == "standard":
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif normalization == "minmax":
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif normalization == "robust":
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / iqr
    elif normalization == "none":
        return data
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_jackknife_samples(
    data: np.ndarray,
) -> list:
    """Compute jackknife samples by leaving out one observation at a time."""
    n_samples = data.shape[0]
    return [np.delete(data, i, axis=0) for i in range(n_samples)]

def _estimate_bias(
    data: np.ndarray,
    model_func: Callable,
    metric_func: Callable,
    normalization: str = "none",
) -> Dict[str, Union[float, Dict]]:
    """Estimate the bias using jackknife method."""
    normalized_data = _apply_normalization(data, normalization)
    samples = _compute_jackknife_samples(normalized_data)

    original_model = model_func(data)
    original_metric = metric_func(original_model, data)

    jackknife_estimates = []
    for sample in samples:
        model = model_func(sample)
        jackknife_estimates.append(metric_func(model, sample))

    bias = np.mean(jackknife_estimates) - original_metric

    return {
        "result": bias,
        "metrics": {"original_metric": original_metric, "jackknife_mean": np.mean(jackknife_estimates)},
        "params_used": {"normalization": normalization},
        "warnings": [],
    }

def bias_estimation_fit(
    data: np.ndarray,
    model_func: Callable,
    metric_func: Callable,
    normalization: str = "none",
) -> Dict[str, Union[float, Dict]]:
    """
    Estimate the bias of a model using the jackknife method.

    Parameters
    ----------
    data : np.ndarray
        Input data as a 2D numpy array.
    model_func : Callable
        Function that fits the model to the data and returns the model parameters.
    metric_func : Callable
        Function that computes the performance metric of the model on the data.
    normalization : str, optional
        Normalization method to apply to the data. Options: "none", "standard", "minmax", "robust".

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing the bias estimate, metrics, parameters used, and warnings.

    Examples
    --------
    >>> def linear_model(data):
    ...     return np.linalg.lstsq(data[:, :-1], data[:, -1], rcond=None)[0]
    ...
    >>> def mse_metric(model, data):
    ...     predictions = np.dot(data[:, :-1], model)
    ...     return np.mean((predictions - data[:, -1]) ** 2)
    ...
    >>> data = np.random.rand(100, 3)
    >>> bias_estimation_fit(data, linear_model, mse_metric, normalization="standard")
    """
    _validate_inputs(data, model_func, metric_func, normalization)
    return _estimate_bias(data, model_func, metric_func, normalization)

################################################################################
# variance_estimation
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def variance_estimation_fit(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    normalization: str = 'none',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    weights: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Estimate variance using the jackknife method.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    statistic_func : Callable[[np.ndarray], float]
        Function to compute the statistic of interest.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : Union[str, Callable], optional
        Metric to evaluate ('mse', 'mae', 'r2', custom callable).
    weights : np.ndarray, optional
        Weights for the data points.
    random_state : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, statistic_func, normalization, metric, weights)

    # Set random state if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    normalized_data = _apply_normalization(data, normalization)

    # Compute jackknife estimates
    n = len(normalized_data)
    jackknife_stats = np.zeros(n)
    for i in range(n):
        leave_one_out_data = np.delete(normalized_data, i)
        jackknife_stats[i] = statistic_func(leave_one_out_data)

    # Compute variance estimate
    variance_estimate = _compute_jackknife_variance(jackknife_stats, weights)

    # Compute metrics
    metrics = _compute_metrics(data, normalized_data, jackknife_stats, metric)

    # Prepare output
    result = {
        'variance_estimate': variance_estimate,
        'jackknife_statistics': jackknife_stats
    }

    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric.__name__ if callable(metric) else metric,
            'weights': weights is not None
        },
        'warnings': []
    }

    return output

def _validate_inputs(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    normalization: str,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    weights: Optional[np.ndarray]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be 1-dimensional.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")
    if normalization not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("Invalid normalization method.")
    if weights is not None:
        if len(weights) != len(data):
            raise ValueError("Weights must have the same length as data.")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

def _apply_normalization(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply normalization to the data."""
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
        raise ValueError("Invalid normalization method.")

def _compute_jackknife_variance(
    jackknife_stats: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    """Compute the jackknife variance estimate."""
    n = len(jackknife_stats)
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = weights / np.sum(weights)

    mean_stat = np.sum(jackknife_stats * weights)
    variance = np.sum(weights * (jackknife_stats - mean_stat) ** 2) * (n - 1)
    return variance

def _compute_metrics(
    original_data: np.ndarray,
    normalized_data: np.ndarray,
    jackknife_stats: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for the jackknife estimates."""
    metrics = {}
    if callable(metric):
        # Custom metric
        metrics['custom_metric'] = metric(original_data, normalized_data)
    elif metric == 'mse':
        metrics['mse'] = np.mean((original_data - normalized_data) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(original_data - normalized_data))
    elif metric == 'r2':
        ss_res = np.sum((original_data - normalized_data) ** 2)
        ss_tot = np.sum((original_data - np.mean(original_data)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    else:
        raise ValueError("Invalid metric.")
    return metrics

# Example usage:
# data = np.random.rand(100)
# def my_statistic(x: np.ndarray) -> float:
#     return np.mean(x)
# result = variance_estimation_fit(data, my_statistic, normalization='standard', metric='mse')

################################################################################
# confidence_intervals
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def confidence_intervals_fit(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    alpha: float = 0.05,
    normalization: str = 'none',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    n_resamples: int = 100,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute jackknife confidence intervals for a given statistic.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    statistic_func : Callable[[np.ndarray], float]
        Function to compute the statistic of interest
    alpha : float, default=0.05
        Significance level for confidence intervals
    normalization : str, default='none'
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or Callable, default='mse'
        Metric for evaluation ('mse', 'mae', 'r2') or custom callable
    n_resamples : int, default=100
        Number of jackknife resamples
    random_state : Optional[int], default=None
        Random seed for reproducibility

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(data, statistic_func)

    # Set random seed if provided
    rng = np.random.RandomState(random_state)

    # Normalize data if required
    normalized_data, norm_params = _apply_normalization(data, normalization)

    # Compute full statistic
    full_statistic = statistic_func(normalized_data)

    # Compute jackknife statistics
    jackknife_stats = _compute_jackknife_statistics(
        normalized_data, statistic_func, n_resamples, rng
    )

    # Compute confidence intervals
    ci_lower, ci_upper = _compute_confidence_intervals(jackknife_stats, alpha)

    # Compute metrics
    metrics = _compute_metrics(
        full_statistic,
        jackknife_stats,
        metric=metric
    )

    # Prepare results dictionary
    result = {
        'result': {
            'full_statistic': full_statistic,
            'confidence_interval': (ci_lower, ci_upper),
            'jackknife_statistics': jackknife_stats
        },
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
            'n_resamples': n_resamples,
            'random_state': random_state
        },
        'warnings': _check_warnings(data, jackknife_stats)
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float]
) -> None:
    """Validate input data and statistic function."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")

    # Test statistic function
    try:
        test_stat = statistic_func(data[:10, :10])
        if not isinstance(test_stat, (int, float)):
            raise ValueError("Statistic function must return a scalar")
    except Exception as e:
        raise ValueError(f"Statistic function raised an error: {str(e)}")

def _apply_normalization(
    data: np.ndarray,
    method: str
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Apply data normalization."""
    normalized = data.copy()
    params = {}

    if method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / std
        params['mean'] = mean.tolist()
        params['std'] = std.tolist()

    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        params['min'] = min_val.tolist()
        params['max'] = max_val.tolist()

    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        normalized = (data - median) / (iqr + 1e-8)
        params['median'] = median.tolist()
        params['iqr'] = iqr.tolist()

    return normalized, {'method': method, 'params': params}

def _compute_jackknife_statistics(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_resamples: int,
    rng: np.random.RandomState
) -> np.ndarray:
    """Compute jackknife statistics."""
    n_samples = data.shape[0]
    stats = np.zeros(n_resamples)

    for i in range(n_resamples):
        # Randomly select samples to leave out
        mask = rng.choice([True, False], size=n_samples, p=[0.5, 0.5])
        if np.sum(mask) == n_samples:
            mask[-1] = False  # Ensure at least one sample is left out

        stats[i] = statistic_func(data[mask])

    return stats

def _compute_confidence_intervals(
    jackknife_stats: np.ndarray,
    alpha: float
) -> tuple[float, float]:
    """Compute confidence intervals from jackknife statistics."""
    sorted_stats = np.sort(jackknife_stats)
    n = len(sorted_stats)

    # Calculate percentiles
    lower_pct = alpha / 2 * 100
    upper_pct = (1 - alpha / 2) * 100

    # Linear interpolation for percentiles
    ci_lower = np.interp(lower_pct, [100*i/n for i in range(n+1)], np.concatenate([[sorted_stats[0]], sorted_stats, [sorted_stats[-1]]]))
    ci_upper = np.interp(upper_pct, [100*i/n for i in range(n+1)], np.concatenate([[sorted_stats[0]], sorted_stats, [sorted_stats[-1]]]))

    return ci_lower, ci_upper

def _compute_metrics(
    full_statistic: float,
    jackknife_stats: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((jackknife_stats - full_statistic) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(jackknife_stats - full_statistic))
        elif metric == 'r2':
            ss_res = np.sum((jackknife_stats - full_statistic) ** 2)
            ss_tot = np.var(jackknife_stats) * len(jackknife_stats)
            metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    else:
        try:
            metrics['custom'] = metric(jackknife_stats, np.full_like(jackknife_stats, full_statistic))
        except Exception as e:
            raise ValueError(f"Custom metric function raised an error: {str(e)}")

    return metrics

def _check_warnings(
    data: np.ndarray,
    jackknife_stats: np.ndarray
) -> list[str]:
    """Check for potential issues and generate warnings."""
    warnings = []

    if np.any(np.isnan(jackknife_stats)) or np.any(np.isinf(jackknife_stats)):
        warnings.append("Jackknife statistics contain NaN or infinite values")

    if data.shape[0] < 2:
        warnings.append("Small sample size may affect jackknife reliability")

    if np.std(jackknife_stats) == 0:
        warnings.append("Jackknife statistics have zero variance")

    return warnings

################################################################################
# leave_one_out
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def leave_one_out_fit(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable[[np.ndarray, np.ndarray], Any],
    metric_func: Callable[[np.ndarray, np.ndarray], float] = None,
    normalize: str = 'none',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform leave-one-out cross-validation using the jackknife method.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    model_func : Callable[[np.ndarray, np.ndarray], Any]
        Function that fits a model and returns predictions.
    metric_func : Callable[[np.ndarray, np.ndarray], float], optional
        Function to compute the metric between true and predicted values.
    normalize : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    custom_metric : Callable[[np.ndarray, np.ndarray], float], optional
        Custom metric function if not using the default.
    **kwargs : dict
        Additional keyword arguments passed to model_func.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_norm = _normalize_data(X, method=normalize)

    # Initialize results storage
    results = {
        'result': [],
        'metrics': [],
        'params_used': kwargs,
        'warnings': []
    }

    # Perform leave-one-out cross-validation
    for i in range(X.shape[0]):
        X_train, y_train = np.delete(X_norm, i, axis=0), np.delete(y, i)
        X_test, y_test = X_norm[i:i+1], y[i]

        # Fit model and get predictions
        model = model_func(X_train, y_train, **kwargs)
        y_pred = model.predict(X_test)

        # Compute metric
        if custom_metric is not None:
            metric = custom_metric(y_test, y_pred)
        elif metric_func is not None:
            metric = metric_func(y_test, y_pred)
        else:
            results['warnings'].append("No metric function provided. Skipping metric calculation.")
            metric = None

        results['result'].append(y_pred)
        if metric is not None:
            results['metrics'].append(metric)

    return results

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input data dimensions and types.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.

    Raises:
    -------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("X and y must be numpy arrays.")
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

def _normalize_data(X: np.ndarray, method: str = 'none') -> np.ndarray:
    """
    Normalize the input data.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix to normalize.
    method : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.

    Returns:
    --------
    np.ndarray
        Normalized feature matrix.
    """
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

# Example usage
if __name__ == "__main__":
    # Example model function (linear regression)
    def linear_regression(X: np.ndarray, y: np.ndarray) -> Any:
        class Model:
            def __init__(self):
                self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y

            def predict(self, X: np.ndarray) -> np.ndarray:
                return X @ self.coef_
        return Model()

    # Example metric function (MSE)
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    # Example data
    X = np.random.rand(10, 5)
    y = np.random.rand(10)

    # Perform leave-one-out cross-validation
    results = leave_one_out_fit(
        X, y,
        model_func=linear_regression,
        metric_func=mse,
        normalize='standard'
    )

################################################################################
# pseudo_values
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    normalize: str = "none",
) -> None:
    """Validate input data and parameters."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Input data contains NaN or infinite values")
    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("Invalid normalization option")

def normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize data based on the specified method."""
    if method == "none":
        return X, y
    elif method == "standard":
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / X_std
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_normalized = (y - y_mean) / y_std
    elif method == "minmax":
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
        y_min = np.min(y)
        y_max = np.max(y)
        y_normalized = (y - y_min) / (y_max - y_min + 1e-8)
    elif method == "robust":
        X_median = np.median(X, axis=0)
        X_q75, X_q25 = np.percentile(X, [75, 25], axis=0)
        X_iqr = X_q75 - X_q25
        X_normalized = (X - X_median) / (X_iqr + 1e-8)
        y_median = np.median(y)
        y_q75, y_q25 = np.percentile(y, [75, 25])
        y_iqr = y_q75 - y_q25
        y_normalized = (y - y_median) / (y_iqr + 1e-8)
    else:
        raise ValueError("Invalid normalization method")
    return X_normalized, y_normalized

def compute_pseudo_values(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    normalize: str = "none",
) -> Dict[str, Any]:
    """Compute pseudo-values using the jackknife method."""
    n_samples = X.shape[0]
    pseudo_values = np.zeros(n_samples)
    full_metric = metric_func(X, y)

    for i in range(n_samples):
        X_leave_one_out = np.delete(X, i, axis=0)
        y_leave_one_out = np.delete(y, i)

        # Compute metric for the leave-one-out dataset
        metric_loo = metric_func(X_leave_one_out, y_leave_one_out)

        # Compute pseudo-value
        pseudo_values[i] = n_samples * full_metric - (n_samples - 1) * metric_loo

    return {
        "result": pseudo_values,
        "metrics": {"full_metric": full_metric},
        "params_used": {
            "normalize": normalize,
            "metric_func": metric_func.__name__ if hasattr(metric_func, "__name__") else "custom",
        },
        "warnings": [],
    }

def pseudo_values_fit(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    normalize: str = "none",
) -> Dict[str, Any]:
    """Main function to compute pseudo-values with jackknife method."""
    # Validate inputs
    validate_inputs(X, y, metric_func, normalize)

    # Normalize data if specified
    X_normalized, y_normalized = normalize_data(X, y, normalize)

    # Compute pseudo-values
    result = compute_pseudo_values(
        X_normalized,
        y_normalized,
        metric_func,
        normalize,
    )

    return result

# Example usage:
if __name__ == "__main__":
    # Define a simple metric function (MSE)
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    # Generate some random data
    np.random.seed(42)
    X = np.random.rand(10, 5)
    y = np.random.rand(10)

    # Compute pseudo-values
    result = pseudo_values_fit(X, y, mse, normalize="standard")
    print(result)

################################################################################
# jackknife_replication
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def jackknife_replication_fit(
    data: np.ndarray,
    target: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    n_splits: int = 5,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute jackknife replication for a given dataset and metric.

    Parameters:
    -----------
    data : np.ndarray
        Input features array of shape (n_samples, n_features)
    target : np.ndarray
        Target values array of shape (n_samples,)
    metric_func : Callable[[np.ndarray, np.ndarray], float]
        Function to compute the metric between predictions and true values
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]]
        Function to normalize the data, defaults to None
    n_splits : int
        Number of splits for jackknife replication, defaults to 5
    random_state : Optional[int]
        Random seed for reproducibility, defaults to None

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.random.rand(10, 3)
    >>> target = np.random.rand(10)
    >>> def mse(y_true, y_pred): return np.mean((y_true - y_pred)**2)
    >>> result = jackknife_replication_fit(data, target, mse)
    """
    # Validate inputs
    if data.shape[0] != target.shape[0]:
        raise ValueError("Data and target must have the same number of samples")
    if n_splits <= 1 or n_splits > data.shape[0]:
        raise ValueError("n_splits must be between 2 and n_samples")

    # Initialize results dictionary
    results = {
        "result": [],
        "metrics": {},
        "params_used": {
            "n_splits": n_splits,
            "normalizer": normalizer is not None
        },
        "warnings": []
    }

    # Normalize data if normalizer is provided
    if normalizer is not None:
        try:
            data = normalizer(data)
        except Exception as e:
            results["warnings"].append(f"Normalization failed: {str(e)}")
            data = normalizer(data)

    # Generate jackknife splits
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(data.shape[0])
    splits = np.array_split(indices, n_splits)

    # Compute jackknife replicates
    for i in range(n_splits):
        train_idx = np.setdiff1d(indices, splits[i])
        X_train, y_train = data[train_idx], target[train_idx]

        # Here you would typically fit a model and make predictions
        # For this example, we'll just use the mean as a simple predictor
        y_pred = np.mean(y_train)

        # Compute metric
        try:
            metric_value = metric_func(y_train, y_pred)
            results["result"].append(metric_value)
        except Exception as e:
            results["warnings"].append(f"Metric computation failed for split {i}: {str(e)}")

    # Compute final statistics
    if results["result"]:
        results["metrics"]["mean"] = np.mean(results["result"])
        results["metrics"]["std"] = np.std(results["result"])
        results["metrics"]["ci_lower"] = np.percentile(results["result"], 2.5)
        results["metrics"]["ci_upper"] = np.percentile(results["result"], 97.5)

    return results

def _validate_input_data(data: np.ndarray, target: np.ndarray) -> None:
    """
    Validate input data and target arrays.

    Parameters:
    -----------
    data : np.ndarray
        Input features array
    target : np.ndarray
        Target values array

    Raises:
    -------
    ValueError
        If inputs are invalid
    """
    if not isinstance(data, np.ndarray) or not isinstance(target, np.ndarray):
        raise ValueError("Data and target must be numpy arrays")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if target.ndim != 1:
        raise ValueError("Target must be a 1D array")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values")
    if np.isnan(target).any() or np.isinf(target).any():
        raise ValueError("Target contains NaN or infinite values")

def _standard_normalizer(data: np.ndarray) -> np.ndarray:
    """
    Standard normalizer (z-score normalization).

    Parameters:
    -----------
    data : np.ndarray
        Input features array

    Returns:
    --------
    np.ndarray
        Normalized data
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-8)

def _minmax_normalizer(data: np.ndarray) -> np.ndarray:
    """
    Min-max normalizer.

    Parameters:
    -----------
    data : np.ndarray
        Input features array

    Returns:
    --------
    np.ndarray
        Normalized data
    """
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val + 1e-8)

# Example usage:
if __name__ == "__main__":
    # Generate some random data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    # Define a simple metric function
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    # Run jackknife replication
    result = jackknife_replication_fit(
        data=X,
        target=y,
        metric_func=mse,
        normalizer=_standard_normalizer,
        n_splits=10
    )

    print(result)

################################################################################
# standard_error_estimation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def standard_error_estimation_fit(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_resamples: int = 100,
    normalization: str = 'none',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    random_state: Optional[int] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Estimate the standard error of a statistic using the jackknife method.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    statistic_func : Callable[[np.ndarray], float]
        Function to compute the statistic of interest.
    n_resamples : int, optional
        Number of jackknife resamples (default is 100).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default is 'none').
    metric : Union[str, Callable], optional
        Metric to evaluate ('mse', 'mae', 'r2', or custom callable) (default is 'mse').
    random_state : Optional[int], optional
        Random seed for reproducibility (default is None).

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]
        Dictionary containing:
        - "result": Estimated standard error
        - "metrics": Computed metrics
        - "params_used": Parameters used in the computation
        - "warnings": List of warnings encountered

    Examples
    --------
    >>> data = np.random.rand(100, 5)
    >>> def mean_statistic(x): return np.mean(x)
    >>> result = standard_error_estimation_fit(data, mean_statistic)
    """
    # Validate inputs
    _validate_inputs(data, statistic_func)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    normalized_data = _normalize_data(data, normalization)

    # Compute jackknife resamples
    jackknife_stats = _compute_jackknife_resamples(normalized_data, statistic_func, n_resamples)

    # Compute standard error
    std_error = _compute_standard_error(jackknife_stats)

    # Compute metrics
    metrics = _compute_metrics(normalized_data, jackknife_stats, metric)

    # Prepare output
    result = {
        "result": std_error,
        "metrics": metrics,
        "params_used": {
            "n_resamples": n_resamples,
            "normalization": normalization,
            "metric": metric.__name__ if callable(metric) else metric
        },
        "warnings": []
    }

    return result

def _validate_inputs(data: np.ndarray, statistic_func: Callable[[np.ndarray], float]) -> None:
    """Validate input data and statistic function."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if not callable(statistic_func):
        raise TypeError("statistic_func must be a callable function")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data according to specified method."""
    if method == 'none':
        return data
    elif method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / ((max_val - min_val) + 1e-8)
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_jackknife_resamples(data: np.ndarray, statistic_func: Callable[[np.ndarray], float],
                                n_resamples: int) -> np.ndarray:
    """Compute jackknife resamples."""
    n_samples = data.shape[0]
    jackknife_stats = np.zeros(n_resamples)

    for i in range(n_resamples):
        # Leave out one sample
        resampled_data = np.delete(data, i % n_samples, axis=0)
        jackknife_stats[i] = statistic_func(resampled_data)

    return jackknife_stats

def _compute_standard_error(jackknife_stats: np.ndarray) -> float:
    """Compute standard error from jackknife resamples."""
    return np.std(jackknife_stats, ddof=1)

def _compute_metrics(data: np.ndarray, jackknife_stats: np.ndarray,
                     metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]) -> Dict[str, float]:
    """Compute metrics based on jackknife resamples."""
    metrics = {}

    if callable(metric):
        # Compute metric for each jackknife resample
        n_resamples = len(jackknife_stats)
        metric_values = np.zeros(n_resamples)

        for i in range(n_resamples):
            resampled_data = np.delete(data, i % data.shape[0], axis=0)
            metric_values[i] = metric(resampled_data, jackknife_stats[i])

        metrics['custom_metric'] = {
            'mean': np.mean(metric_values),
            'std': np.std(metric_values)
        }
    else:
        # Compute standard metrics
        if metric == 'mse':
            mse_values = np.zeros(len(jackknife_stats))
            for i in range(len(jackknife_stats)):
                resampled_data = np.delete(data, i % data.shape[0], axis=0)
                mse_values[i] = np.mean((resampled_data - jackknife_stats[i])**2)
            metrics['mse'] = {
                'mean': np.mean(mse_values),
                'std': np.std(mse_values)
            }
        elif metric == 'mae':
            mae_values = np.zeros(len(jackknife_stats))
            for i in range(len(jackknife_stats)):
                resampled_data = np.delete(data, i % data.shape[0], axis=0)
                mae_values[i] = np.mean(np.abs(resampled_data - jackknife_stats[i]))
            metrics['mae'] = {
                'mean': np.mean(mae_values),
                'std': np.std(mae_values)
            }
        elif metric == 'r2':
            r2_values = np.zeros(len(jackknife_stats))
            for i in range(len(jackknife_stats)):
                resampled_data = np.delete(data, i % data.shape[0], axis=0)
                ss_res = np.sum((resampled_data - jackknife_stats[i])**2)
                ss_tot = np.sum((resampled_data - np.mean(resampled_data))**2)
                r2_values[i] = 1 - (ss_res / (ss_tot + 1e-8))
            metrics['r2'] = {
                'mean': np.mean(r2_values),
                'std': np.std(r2_values)
            }

    return metrics

################################################################################
# bias_correction
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weights: Optional[np.ndarray] = None
) -> None:
    """Validate input arrays."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if sample_weights is not None:
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have the same length as y_true")
        if np.any(sample_weights < 0):
            raise ValueError("sample_weights cannot contain negative values")

def _apply_normalization(
    y: np.ndarray,
    method: str = "none",
    axis: int = 0
) -> np.ndarray:
    """Apply normalization to input data."""
    if method == "none":
        return y
    elif method == "standard":
        mean = np.mean(y, axis=axis, keepdims=True)
        std = np.std(y, axis=axis, keepdims=True)
        return (y - mean) / std
    elif method == "minmax":
        min_val = np.min(y, axis=axis, keepdims=True)
        max_val = np.max(y, axis=axis, keepdims=True)
        return (y - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(y, axis=axis, keepdims=True)
        iqr = np.percentile(y, 75, axis=axis, keepdims=True) - np.percentile(y, 25, axis=axis, keepdims=True)
        return (y - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    sample_weights: Optional[np.ndarray] = None
) -> float:
    """Compute specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)

    if metric == "mse":
        error = y_true - y_pred
        if sample_weights is not None:
            return np.average(error**2, weights=sample_weights)
        return np.mean(error**2)
    elif metric == "mae":
        error = np.abs(y_true - y_pred)
        if sample_weights is not None:
            return np.average(error, weights=sample_weights)
        return np.mean(error)
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == "logloss":
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _jackknife_resampling(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    sample_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Perform jackknife resampling to compute bias-corrected estimates."""
    n_samples = len(y_true)
    jackknife_metrics = np.zeros(n_samples)

    for i in range(n_samples):
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False

        y_true_sub = y_true[mask]
        y_pred_sub = y_pred[mask]

        if sample_weights is not None:
            weights_sub = sample_weights[mask]
        else:
            weights_sub = None

        jackknife_metrics[i] = _compute_metric(y_true_sub, y_pred_sub, metric, weights_sub)

    return jackknife_metrics

def bias_correction_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable] = "mse",
    normalization: str = "none",
    sample_weights: Optional[np.ndarray] = None,
    return_full_results: bool = False
) -> Dict[str, Union[float, Dict]]:
    """
    Compute bias-corrected estimate using jackknife resampling.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    metric : str or callable, default="mse"
        Metric to compute (can be "mse", "mae", "r2", "logloss" or custom callable).
    normalization : str, default="none"
        Normalization method ("none", "standard", "minmax", "robust").
    sample_weights : np.ndarray, optional
        Array of sample weights.
    return_full_results : bool, default=False
        Whether to return full results dictionary.

    Returns:
    --------
    dict or float
        If return_full_results is True, returns dictionary with results.
        Otherwise returns just the bias-corrected estimate.

    Example:
    --------
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 1.9, 3.2])
    >>> result = bias_correction_fit(y_true, y_pred)
    """
    _validate_inputs(y_true, y_pred, sample_weights)

    # Apply normalization
    y_true_norm = _apply_normalization(y_true, normalization)
    y_pred_norm = _apply_normalization(y_pred, normalization)

    # Compute original metric
    original_metric = _compute_metric(y_true_norm, y_pred_norm, metric, sample_weights)

    # Perform jackknife resampling
    jackknife_metrics = _jackknife_resampling(y_true_norm, y_pred_norm, metric, sample_weights)

    # Compute bias-corrected estimate
    bias_corrected = (n_samples * original_metric) - (n_samples - 1) * np.mean(jackknife_metrics)

    if return_full_results:
        return {
            "result": bias_corrected,
            "metrics": {
                "original_metric": original_metric,
                "jackknife_metrics": jackknife_metrics.tolist(),
                "bias_corrected": bias_corrected
            },
            "params_used": {
                "metric": metric,
                "normalization": normalization
            },
            "warnings": []
        }
    else:
        return bias_corrected

################################################################################
# efficiency_gain
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def efficiency_gain_fit(
    estimator: Callable,
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    normalization: str = 'none',
    n_bootstraps: int = 100,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute the efficiency gain using jackknife resampling.

    Parameters:
    -----------
    estimator : callable
        The model/estimator to evaluate. Must have fit and predict methods.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2', or custom callable).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    n_bootstraps : int, optional
        Number of bootstrap samples.
    random_state : int or None, optional
        Random seed for reproducibility.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_norm = _apply_normalization(X, normalization)
    y_norm = _apply_normalization(y.reshape(-1, 1), normalization).flatten()

    # Fit the full model
    estimator.fit(X_norm, y_norm)
    full_pred = estimator.predict(X_norm)

    # Compute baseline metric
    baseline_metric = _compute_metric(full_pred, y_norm, metric)

    # Jackknife resampling
    n_samples = X.shape[0]
    jackknife_metrics = []
    for i in range(n_samples):
        X_jack = np.delete(X_norm, i, axis=0)
        y_jack = np.delete(y_norm, i)

        estimator.fit(X_jack, y_jack)
        pred_jack = estimator.predict(X_norm)
        jackknife_metrics.append(_compute_metric(pred_jack, y_norm, metric))

    # Compute efficiency gain
    var_full = np.var(full_pred - y_norm)
    var_jack = np.mean(np.array(jackknife_metrics) - baseline_metric)**2
    efficiency_gain = var_full / (n_samples * var_jack)

    return {
        'result': efficiency_gain,
        'metrics': {
            'baseline_metric': baseline_metric,
            'jackknife_metrics': jackknife_metrics
        },
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'n_bootstraps': n_bootstraps
        },
        'warnings': []
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

def _compute_metric(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Compute the specified metric."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_pred - y_true)**2)
    elif metric == 'mae':
        return np.mean(np.abs(y_pred - y_true))
    elif metric == 'r2':
        ss_res = np.sum((y_pred - y_true)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    else:
        raise ValueError(f"Unknown metric: {metric}")
