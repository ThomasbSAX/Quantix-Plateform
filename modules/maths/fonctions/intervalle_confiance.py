"""
Quantix – Module intervalle_confiance
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# intervalle_confiance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def intervalle_confiance_fit(
    data: np.ndarray,
    confidence_level: float = 0.95,
    method: str = 'normal',
    alpha: Optional[float] = None,
    custom_interval_func: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Calculate confidence intervals for given data.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    confidence_level : float, optional
        Confidence level (0 < confidence_level < 1), by default 0.95.
    method : str, optional
        Method to calculate confidence intervals ('normal', 't', 'bootstrap'), by default 'normal'.
    alpha : float, optional
        Significance level (0 < alpha < 1), by default None. If provided, overrides confidence_level.
    custom_interval_func : Callable, optional
        Custom function to calculate confidence intervals, by default None.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - "result": array of confidence intervals
        - "metrics": dictionary of metrics
        - "params_used": dictionary of parameters used
        - "warnings": list of warnings

    Examples
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = intervalle_confiance_fit(data)
    """
    # Validate inputs
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array")
    if len(data) == 0:
        raise ValueError("data cannot be empty")

    # Set alpha if not provided
    if alpha is None:
        alpha = 1 - confidence_level

    # Validate alpha and confidence_level
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")
    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be between 0 and 1")

    # Choose method
    if custom_interval_func is not None:
        result = custom_interval_func(data, alpha)
    else:
        if method == 'normal':
            result = _calculate_normal_interval(data, alpha)
        elif method == 't':
            result = _calculate_t_interval(data, alpha)
        elif method == 'bootstrap':
            result = _calculate_bootstrap_interval(data, alpha)
        else:
            raise ValueError(f"Unknown method: {method}")

    # Calculate metrics
    metrics = {
        'mean': np.mean(data),
        'std': np.std(data, ddof=1),
        'n': len(data)
    }

    # Prepare output
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "confidence_level": confidence_level,
            "method": method,
            "alpha": alpha
        },
        "warnings": []
    }

    return output

def _calculate_normal_interval(data: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calculate confidence intervals using normal distribution.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    alpha : float
        Significance level.

    Returns
    -------
    np.ndarray
        Array of confidence intervals.
    """
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    z_score = _get_z_score(alpha)
    margin_of_error = z_score * std_err
    return np.array([mean - margin_of_error, mean + margin_of_error])

def _calculate_t_interval(data: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calculate confidence intervals using t-distribution.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    alpha : float
        Significance level.

    Returns
    -------
    np.ndarray
        Array of confidence intervals.
    """
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    t_score = _get_t_score(alpha, len(data) - 1)
    margin_of_error = t_score * std_err
    return np.array([mean - margin_of_error, mean + margin_of_error])

def _calculate_bootstrap_interval(data: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calculate confidence intervals using bootstrap method.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    alpha : float
        Significance level.

    Returns
    -------
    np.ndarray
        Array of confidence intervals.
    """
    n_bootstraps = 1000
    bootstrap_means = np.zeros(n_bootstraps)

    for i in range(n_bootstraps):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means[i] = np.mean(sample)

    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    return np.percentile(bootstrap_means, [lower_percentile, upper_percentile])

def _get_z_score(alpha: float) -> float:
    """
    Get z-score for given alpha.

    Parameters
    ----------
    alpha : float
        Significance level.

    Returns
    -------
    float
        Z-score.
    """
    from scipy.stats import norm
    return norm.ppf(1 - alpha / 2)

def _get_t_score(alpha: float, df: int) -> float:
    """
    Get t-score for given alpha and degrees of freedom.

    Parameters
    ----------
    alpha : float
        Significance level.
    df : int
        Degrees of freedom.

    Returns
    -------
    float
        T-score.
    """
    from scipy.stats import t
    return t.ppf(1 - alpha / 2, df)

################################################################################
# marge_erreur
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def marge_erreur_fit(
    data: np.ndarray,
    confidence_level: float = 0.95,
    method: str = 'normal',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'mse',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """
    Calculate the margin of error for a given dataset.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    confidence_level : float, optional
        Confidence level for the interval (default: 0.95).
    method : str, optional
        Method to calculate margin of error ('normal', 't', 'bootstrap') (default: 'normal').
    normalizer : Callable, optional
        Function to normalize the data (default: None).
    metric : str, optional
        Metric to use ('mse', 'mae', 'r2') (default: 'mse').
    custom_metric : Callable, optional
        Custom metric function (default: None).
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent') (default: 'closed_form').
    regularization : str, optional
        Regularization type ('l1', 'l2', 'elasticnet') (default: None).
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, confidence_level)

    # Normalize data if a normalizer is provided
    normalized_data = _apply_normalization(data, normalizer)

    # Calculate the margin of error based on the chosen method
    if method == 'normal':
        result = _marge_erreur_normal(normalized_data, confidence_level)
    elif method == 't':
        result = _marge_erreur_t(normalized_data, confidence_level)
    elif method == 'bootstrap':
        result = _marge_erreur_bootstrap(normalized_data, confidence_level)
    else:
        raise ValueError("Invalid method specified.")

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, result['mean'], metric, custom_metric)

    # Prepare the output dictionary
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "confidence_level": confidence_level,
            "method": method,
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return output

def _validate_inputs(data: np.ndarray, confidence_level: float) -> None:
    """Validate the input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("Confidence level must be between 0 and 1.")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values.")

def _apply_normalization(data: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to the data if a normalizer is provided."""
    if normalizer is not None:
        return normalizer(data)
    return data

def _marge_erreur_normal(data: np.ndarray, confidence_level: float) -> Dict[str, Any]:
    """Calculate margin of error using the normal distribution."""
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    z_score = _get_z_score(confidence_level)
    margin_of_error = z_score * std_err
    return {"mean": mean, "margin_of_error": margin_of_error}

def _marge_erreur_t(data: np.ndarray, confidence_level: float) -> Dict[str, Any]:
    """Calculate margin of error using the t-distribution."""
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    t_score = _get_t_score(confidence_level, len(data) - 1)
    margin_of_error = t_score * std_err
    return {"mean": mean, "margin_of_error": margin_of_error}

def _marge_erreur_bootstrap(data: np.ndarray, confidence_level: float) -> Dict[str, Any]:
    """Calculate margin of error using bootstrap method."""
    n_bootstraps = 1000
    bootstrapped_means = np.array([
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstraps)
    ])
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrapped_means, alpha / 2 * 100)
    upper = np.percentile(bootstrapped_means, (1 - alpha / 2) * 100)
    margin_of_error = (upper - lower) / 2
    return {"mean": np.mean(data), "margin_of_error": margin_of_error}

def _get_z_score(confidence_level: float) -> float:
    """Get the z-score for a given confidence level."""
    from scipy.stats import norm
    return norm.ppf((1 + confidence_level) / 2)

def _get_t_score(confidence_level: float, df: int) -> float:
    """Get the t-score for a given confidence level and degrees of freedom."""
    from scipy.stats import t
    return t.ppf((1 + confidence_level) / 2, df)

def _calculate_metrics(data: np.ndarray, mean: float, metric: str, custom_metric: Optional[Callable]) -> Dict[str, float]:
    """Calculate the specified metrics."""
    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((data - mean) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(data - mean))
    elif metric == 'r2':
        ss_total = np.sum((data - np.mean(data)) ** 2)
        ss_res = np.sum((data - mean) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_total)
    if custom_metric is not None:
        metrics['custom'] = custom_metric(data, mean)
    return metrics

################################################################################
# niveau_confiance
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def niveau_confiance_fit(
    data: np.ndarray,
    confidence_level: float = 0.95,
    method: str = 'normal',
    metric: Union[str, Callable] = 'mean',
    solver: str = 'closed_form',
    normalizer: Optional[Callable] = None,
    custom_func: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Calculate confidence interval for given data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    confidence_level : float, optional
        Confidence level (default: 0.95).
    method : str, optional
        Method to calculate confidence interval ('normal', 't', 'bootstrap') (default: 'normal').
    metric : str or callable, optional
        Metric to use ('mean', 'median', custom callable) (default: 'mean').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent') (default: 'closed_form').
    normalizer : callable, optional
        Normalization function (default: None).
    custom_func : callable, optional
        Custom function for confidence interval calculation (default: None).
    **kwargs : dict
        Additional parameters for the chosen method.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(data, confidence_level)

    # Normalize data if required
    if normalizer is not None:
        data = normalizer(data)

    # Calculate statistic based on metric
    statistic, stat_name = _calculate_statistic(data, metric)

    # Calculate confidence interval
    interval = _calculate_confidence_interval(
        data,
        statistic,
        stat_name,
        confidence_level,
        method,
        solver,
        custom_func,
        **kwargs
    )

    # Prepare results
    result = {
        'interval': interval,
        'statistic': statistic,
        'confidence_level': confidence_level
    }

    metrics = {
        'stat_name': stat_name,
        'method_used': method
    }

    params_used = {
        'metric': metric if isinstance(metric, str) else 'custom',
        'solver': solver,
        'normalizer': normalizer.__name__ if normalizer else None
    }

    warnings = _check_warnings(data, confidence_level)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(data: np.ndarray, confidence_level: float) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if len(data.shape) != 1:
        raise ValueError("Data must be 1-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("Confidence level must be between 0 and 1")

def _calculate_statistic(
    data: np.ndarray,
    metric: Union[str, Callable]
) -> tuple:
    """Calculate statistic based on chosen metric."""
    if isinstance(metric, str):
        if metric == 'mean':
            stat = np.mean(data)
            name = 'mean'
        elif metric == 'median':
            stat = np.median(data)
            name = 'median'
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        stat = metric(data)
        name = 'custom'
    return stat, name

def _calculate_confidence_interval(
    data: np.ndarray,
    statistic: float,
    stat_name: str,
    confidence_level: float,
    method: str,
    solver: str,
    custom_func: Optional[Callable],
    **kwargs
) -> tuple:
    """Calculate confidence interval using specified method."""
    if custom_func is not None:
        return custom_func(data, statistic, confidence_level)

    if method == 'normal':
        std_err = np.std(data) / np.sqrt(len(data))
        z_score = _get_z_score(confidence_level)
        margin = z_score * std_err
    elif method == 't':
        df = len(data) - 1
        t_score = _get_t_score(confidence_level, df)
        std_err = np.std(data, ddof=1) / np.sqrt(len(data))
        margin = t_score * std_err
    elif method == 'bootstrap':
        margin = _bootstrap_confidence_interval(data, statistic, confidence_level)
    else:
        raise ValueError(f"Unknown method: {method}")

    return (statistic - margin, statistic + margin)

def _get_z_score(confidence_level: float) -> float:
    """Get z-score for normal distribution."""
    from scipy.stats import norm
    return norm.ppf((1 + confidence_level) / 2)

def _get_t_score(confidence_level: float, df: int) -> float:
    """Get t-score for Student's t-distribution."""
    from scipy.stats import t
    return t.ppf((1 + confidence_level) / 2, df)

def _bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func: Callable,
    confidence_level: float,
    n_bootstraps: int = 1000
) -> tuple:
    """Calculate bootstrap confidence interval."""
    bootstraps = np.random.choice(data, (n_bootstraps, len(data)))
    stats = np.apply_along_axis(statistic_func, 1, bootstraps)
    alpha = (1 - confidence_level) / 2
    lower = np.percentile(stats, alpha * 100)
    upper = np.percentile(stats, (1 - alpha) * 100)
    return (lower, upper)

def _check_warnings(data: np.ndarray, confidence_level: float) -> list:
    """Check for potential warnings."""
    warnings = []
    if len(data) < 30:
        warnings.append("Sample size is small, consider using t-distribution")
    if confidence_level < 0.9:
        warnings.append("Low confidence level may lead to unreliable intervals")
    return warnings

################################################################################
# ecart_type
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data must not contain NaN or infinite values.")

def _compute_ecart_type(data: np.ndarray, ddof: int = 1) -> float:
    """Compute the standard deviation of the data."""
    mean = np.mean(data)
    squared_diff = (data - mean) ** 2
    variance = np.sum(squared_diff) / (len(data) - ddof)
    return np.sqrt(variance)

def _normalize_data(data: np.ndarray, method: str = "none") -> np.ndarray:
    """Normalize the data using the specified method."""
    if method == "none":
        return data
    elif method == "standard":
        mean = np.mean(data)
        std = np.std(data, ddof=1)
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

def ecart_type_fit(
    data: np.ndarray,
    normalize_method: str = "none",
    ddof: int = 1
) -> Dict[str, Any]:
    """
    Compute the standard deviation of the data with optional normalization.

    Parameters:
    -----------
    data : np.ndarray
        Input data.
    normalize_method : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    ddof : int, optional
        Delta degrees of freedom for variance calculation.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_input(data)

    normalized_data = _normalize_data(data, normalize_method)
    ecart_type_value = _compute_ecart_type(normalized_data, ddof)

    result = {
        "result": ecart_type_value,
        "metrics": {},
        "params_used": {
            "normalize_method": normalize_method,
            "ddof": ddof
        },
        "warnings": []
    }

    return result

# Example usage:
# data = np.array([1, 2, 3, 4, 5])
# result = ecart_type_fit(data, normalize_method="standard")

################################################################################
# taille_echantillon
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(
    margin_error: float,
    confidence_level: float,
    population_std: Optional[float] = None,
    sample_std: Optional[float] = None,
    alpha: Optional[float] = None
) -> None:
    """Validate inputs for sample size calculation."""
    if margin_error <= 0:
        raise ValueError("Margin of error must be positive.")
    if not (0 < confidence_level <= 1):
        raise ValueError("Confidence level must be between 0 and 1.")
    if population_std is not None and population_std <= 0:
        raise ValueError("Population standard deviation must be positive.")
    if sample_std is not None and sample_std <= 0:
        raise ValueError("Sample standard deviation must be positive.")
    if alpha is not None and (not 0 < alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1.")

def _calculate_z_score(confidence_level: float) -> float:
    """Calculate z-score for given confidence level."""
    from scipy.stats import norm
    alpha = 1 - confidence_level
    return norm.ppf(1 - alpha / 2)

def _calculate_t_score(
    confidence_level: float,
    degrees_freedom: int
) -> float:
    """Calculate t-score for given confidence level and degrees of freedom."""
    from scipy.stats import t
    alpha = 1 - confidence_level
    return t.ppf(1 - alpha / 2, degrees_freedom)

def _calculate_sample_size(
    margin_error: float,
    std_dev: float,
    z_score: float
) -> int:
    """Calculate required sample size."""
    return np.ceil((z_score * std_dev / margin_error) ** 2)

def taille_echantillon_fit(
    margin_error: float,
    confidence_level: float,
    population_std: Optional[float] = None,
    sample_std: Optional[float] = None,
    use_t_distribution: bool = False,
    degrees_freedom: Optional[int] = None,
    alpha: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate required sample size for a confidence interval.

    Parameters:
    - margin_error: Desired margin of error
    - confidence_level: Confidence level (0 < x <= 1)
    - population_std: Population standard deviation (if known)
    - sample_std: Sample standard deviation (if known)
    - use_t_distribution: Whether to use t-distribution
    - degrees_freedom: Degrees of freedom for t-distribution
    - alpha: Significance level (alternative to confidence_level)

    Returns:
    Dictionary containing result, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(margin_error, confidence_level, population_std, sample_std, alpha)

    # Set parameters
    params_used = {
        'margin_error': margin_error,
        'confidence_level': confidence_level,
        'use_t_distribution': use_t_distribution
    }

    # Calculate standard deviation to use
    std_dev = population_std if population_std is not None else sample_std
    if std_dev is None:
        raise ValueError("Either population standard deviation or sample standard deviation must be provided.")

    # Calculate appropriate score
    if use_t_distribution:
        if degrees_freedom is None:
            raise ValueError("Degrees of freedom must be provided when using t-distribution.")
        z_score = _calculate_t_score(confidence_level, degrees_freedom)
    else:
        z_score = _calculate_z_score(confidence_level)

    # Calculate sample size
    result = int(_calculate_sample_size(margin_error, std_dev, z_score))

    # Prepare output
    metrics = {
        'z_score': z_score,
        'std_dev_used': std_dev
    }

    warnings = []
    if use_t_distribution and degrees_freedom < 30:
        warnings.append("Small sample size (n < 30) - consider using normal approximation with caution.")

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

# Example usage:
"""
result = taille_echantillon_fit(
    margin_error=0.05,
    confidence_level=0.95,
    population_std=1.0
)
"""

################################################################################
# distribution_normale
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values.")

def _compute_mean(data: np.ndarray) -> float:
    """Compute the mean of the data."""
    return np.mean(data)

def _compute_std(data: np.ndarray) -> float:
    """Compute the standard deviation of the data."""
    return np.std(data, ddof=1)

def _compute_confidence_interval(
    mean: float,
    std: float,
    n_samples: int,
    confidence_level: float = 0.95
) -> tuple:
    """Compute the confidence interval for a normal distribution."""
    from scipy.stats import norm
    z_score = norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z_score * (std / np.sqrt(n_samples))
    return (mean - margin_of_error, mean + margin_of_error)

def distribution_normale_fit(
    data: np.ndarray,
    confidence_level: float = 0.95,
    normalize: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Fit a normal distribution to the data and compute confidence intervals.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    confidence_level : float, optional
        Confidence level for the interval (default is 0.95).
    normalize : str, optional
        Normalization method: 'standard', 'minmax', or None (default is None).
    custom_metric : Callable, optional
        Custom metric function to evaluate the fit.

    Returns
    -------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Example
    -------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = distribution_normale_fit(data)
    """
    _validate_inputs(data)

    # Normalize data if specified
    if normalize == 'standard':
        mean = _compute_mean(data)
        std = _compute_std(data)
        data_normalized = (data - mean) / std
    elif normalize == 'minmax':
        data_min, data_max = np.min(data), np.max(data)
        data_normalized = (data - data_min) / (data_max - data_min)
    else:
        data_normalized = data.copy()

    # Compute mean and standard deviation
    mean = _compute_mean(data_normalized)
    std = _compute_std(data_normalized)

    # Compute confidence interval
    n_samples = len(data_normalized)
    ci_lower, ci_upper = _compute_confidence_interval(mean, std, n_samples, confidence_level)

    # Compute metrics
    metrics = {}
    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(data, data_normalized)
    else:
        # Default metrics
        residuals = data - mean
        metrics['mse'] = np.mean(residuals ** 2)
        metrics['mae'] = np.mean(np.abs(residuals))

    # Prepare output
    result = {
        'result': {
            'mean': mean,
            'std': std,
            'confidence_interval': (ci_lower, ci_upper)
        },
        'metrics': metrics,
        'params_used': {
            'confidence_level': confidence_level,
            'normalize': normalize
        },
        'warnings': []
    }

    return result

################################################################################
# intervalle_confiance_bilaterale
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def intervalle_confiance_bilaterale_fit(
    data: np.ndarray,
    alpha: float = 0.05,
    method: str = 'normal',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'mean',
    solver: str = 'closed_form',
    **kwargs
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Calculate bilateral confidence interval for given data.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    alpha : float, optional
        Significance level (default: 0.05).
    method : str, optional
        Confidence interval method ('normal', 't', 'bootstrap') (default: 'normal').
    normalizer : Callable, optional
        Normalization function (default: None).
    metric : str, optional
        Metric to use ('mean', 'median') (default: 'mean').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent') (default: 'closed_form').
    **kwargs
        Additional method-specific parameters.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': tuple of (lower_bound, upper_bound)
        - 'metrics': dictionary of computed metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings

    Examples
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = intervalle_confiance_bilaterale_fit(data)
    """
    # Validate inputs
    _validate_inputs(data, alpha)

    # Normalize data if normalizer is provided
    normalized_data = _apply_normalization(data, normalizer)

    # Calculate statistic based on method
    statistic = _calculate_statistic(normalized_data, metric)

    # Calculate confidence interval based on method
    lower_bound, upper_bound = _calculate_confidence_interval(
        statistic,
        normalized_data,
        alpha,
        method,
        solver,
        **kwargs
    )

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, statistic)

    return {
        'result': (lower_bound, upper_bound),
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'method': method,
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver
        },
        'warnings': _check_warnings(normalized_data, alpha)
    }

def _validate_inputs(data: np.ndarray, alpha: float) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if len(data.shape) != 1:
        raise ValueError("Data must be one-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")

def _apply_normalization(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to data if normalizer is provided."""
    if normalizer is not None:
        return normalizer(data)
    return data

def _calculate_statistic(
    data: np.ndarray,
    metric: str
) -> float:
    """Calculate statistic based on chosen metric."""
    if metric == 'mean':
        return np.mean(data)
    elif metric == 'median':
        return np.median(data)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _calculate_confidence_interval(
    statistic: float,
    data: np.ndarray,
    alpha: float,
    method: str,
    solver: str,
    **kwargs
) -> tuple:
    """Calculate confidence interval based on chosen method."""
    if method == 'normal':
        return _calculate_normal_confidence_interval(statistic, data, alpha)
    elif method == 't':
        return _calculate_t_confidence_interval(statistic, data, alpha)
    elif method == 'bootstrap':
        return _calculate_bootstrap_confidence_interval(statistic, data, alpha, solver, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

def _calculate_normal_confidence_interval(
    statistic: float,
    data: np.ndarray,
    alpha: float
) -> tuple:
    """Calculate normal confidence interval."""
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    z_score = _get_z_score(alpha)
    return (statistic - z_score * std_err, statistic + z_score * std_err)

def _calculate_t_confidence_interval(
    statistic: float,
    data: np.ndarray,
    alpha: float
) -> tuple:
    """Calculate t-distribution confidence interval."""
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    t_score = _get_t_score(alpha, len(data))
    return (statistic - t_score * std_err, statistic + t_score * std_err)

def _calculate_bootstrap_confidence_interval(
    statistic: float,
    data: np.ndarray,
    alpha: float,
    solver: str,
    **kwargs
) -> tuple:
    """Calculate bootstrap confidence interval."""
    n_bootstraps = kwargs.get('n_bootstraps', 1000)
    bootstrap_stats = _bootstrap_resample(data, statistic, n_bootstraps)

    if solver == 'closed_form':
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        return (
            np.percentile(bootstrap_stats, lower_percentile),
            np.percentile(bootstrap_stats, upper_percentile)
        )
    else:
        raise ValueError(f"Unknown solver for bootstrap: {solver}")

def _bootstrap_resample(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_bootstraps: int
) -> np.ndarray:
    """Perform bootstrap resampling."""
    n = len(data)
    bootstrap_stats = np.zeros(n_bootstraps)

    for i in range(n_bootstraps):
        resampled_data = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic_func(resampled_data)

    return bootstrap_stats

def _calculate_metrics(
    data: np.ndarray,
    statistic: float
) -> Dict[str, float]:
    """Calculate various metrics."""
    return {
        'mean': np.mean(data),
        'std_dev': np.std(data, ddof=1),
        'sample_size': len(data),
        'statistic_value': statistic
    }

def _check_warnings(
    data: np.ndarray,
    alpha: float
) -> list:
    """Check for potential warnings."""
    warnings = []

    if len(data) < 30:
        warnings.append("Sample size is small, consider using t-distribution instead of normal")

    if alpha < 0.01 or alpha > 0.1:
        warnings.append(f"Alpha value {alpha} is outside typical range (0.01-0.1)")

    return warnings

def _get_z_score(alpha: float) -> float:
    """Get z-score for given alpha."""
    from scipy.stats import norm
    return norm.ppf(1 - alpha / 2)

def _get_t_score(alpha: float, df: int) -> float:
    """Get t-score for given alpha and degrees of freedom."""
    from scipy.stats import t
    return t.ppf(1 - alpha / 2, df)

################################################################################
# intervalle_confiance_unilaterale
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def intervalle_confiance_unilaterale_fit(
    data: np.ndarray,
    alpha: float = 0.05,
    method: str = 'normal',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    statistic_func: Optional[Callable[[np.ndarray], float]] = None,
    solver: str = 'closed_form',
    **kwargs
) -> Dict[str, Union[float, Dict, str]]:
    """
    Calculate a one-sided confidence interval for the mean of a dataset.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    alpha : float, optional
        Significance level (default is 0.05).
    method : str, optional
        Method to compute the confidence interval ('normal', 't', or 'bootstrap').
    normalizer : Callable, optional
        Function to normalize the data before computation.
    statistic_func : Callable, optional
        Custom function to compute the statistic of interest.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', etc.).
    **kwargs : dict
        Additional keyword arguments for the solver or method.

    Returns
    -------
    Dict[str, Union[float, Dict, str]]
        Dictionary containing:
        - 'result': The confidence interval (lower or upper bound).
        - 'metrics': Dictionary of computed metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings generated during computation.

    Examples
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = intervalle_confiance_unilaterale_fit(data, alpha=0.05)
    """
    # Validate inputs
    _validate_inputs(data, alpha)

    # Normalize data if a normalizer is provided
    normalized_data = _apply_normalization(data, normalizer)

    # Compute the statistic of interest
    statistic = _compute_statistic(normalized_data, statistic_func)

    # Compute the confidence interval
    ci = _compute_confidence_interval(statistic, normalized_data, alpha, method, solver, **kwargs)

    # Prepare the output dictionary
    result = {
        'result': ci,
        'metrics': {'statistic': statistic},
        'params_used': {
            'alpha': alpha,
            'method': method,
            'normalizer': normalizer.__name__ if normalizer else None,
            'statistic_func': statistic_func.__name__ if statistic_func else None,
            'solver': solver
        },
        'warnings': []
    }

    return result

def _validate_inputs(data: np.ndarray, alpha: float) -> None:
    """Validate the input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if len(data.shape) != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values.")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")

def _apply_normalization(data: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to the data if a normalizer is provided."""
    if normalizer is not None:
        return normalizer(data)
    return data

def _compute_statistic(data: np.ndarray, statistic_func: Optional[Callable]) -> float:
    """Compute the statistic of interest."""
    if statistic_func is not None:
        return statistic_func(data)
    return np.mean(data)

def _compute_confidence_interval(
    statistic: float,
    data: np.ndarray,
    alpha: float,
    method: str,
    solver: str,
    **kwargs
) -> float:
    """Compute the one-sided confidence interval."""
    if method == 'normal':
        std_err = np.std(data, ddof=1) / np.sqrt(len(data))
        z_score = _get_z_score(alpha)
        return statistic + z_score * std_err
    elif method == 't':
        t_score = _get_t_score(len(data) - 1, alpha)
        std_err = np.std(data, ddof=1) / np.sqrt(len(data))
        return statistic + t_score * std_err
    elif method == 'bootstrap':
        return _bootstrap_confidence_interval(data, alpha, statistic, solver, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

def _get_z_score(alpha: float) -> float:
    """Get the z-score for a given alpha level."""
    from scipy.stats import norm
    return norm.ppf(1 - alpha)

def _get_t_score(df: int, alpha: float) -> float:
    """Get the t-score for a given degrees of freedom and alpha level."""
    from scipy.stats import t
    return t.ppf(1 - alpha, df)

def _bootstrap_confidence_interval(
    data: np.ndarray,
    alpha: float,
    statistic_func: Callable,
    solver: str,
    n_bootstrap: int = 1000,
    **kwargs
) -> float:
    """Compute the confidence interval using bootstrap."""
    bootstrap_stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))
    bootstrap_stats = np.array(bootstrap_stats)
    return np.percentile(bootstrap_stats, 100 * (1 - alpha))

################################################################################
# estimation_point
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate input arrays and return standardized versions.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : Optional[np.ndarray]
        Array of predicted values. If None, will be initialized as zeros.
    weights : Optional[np.ndarray]
        Array of observation weights. If None, uniform weights are used.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Validated and standardized (y_true, y_pred, weights) arrays.
    """
    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be a numpy array")
    if y_pred is not None and not isinstance(y_pred, np.ndarray):
        raise TypeError("y_pred must be a numpy array or None")
    if weights is not None and not isinstance(weights, np.ndarray):
        raise TypeError("weights must be a numpy array or None")

    y_true = np.asarray(y_true, dtype=np.float64)
    if y_pred is None:
        y_pred = np.zeros_like(y_true)
    else:
        y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if weights is None:
        weights = np.ones_like(y_true, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != len(y_true):
            raise ValueError("weights must have the same length as y_true")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")

    return y_true, y_pred, weights

def _compute_metric(
    metric: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray
) -> float:
    """
    Compute specified metric between true and predicted values.

    Parameters
    ----------
    metric : str
        Name of the metric to compute.
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    weights : np.ndarray
        Array of observation weights.

    Returns
    -------
    float
        Computed metric value.
    """
    residuals = y_true - y_pred

    if metric == "mse":
        return np.average(residuals**2, weights=weights)
    elif metric == "mae":
        return np.average(np.abs(residuals), weights=weights)
    elif metric == "r2":
        ss_res = np.sum(weights * residuals**2)
        ss_tot = np.sum(weights * (y_true - np.average(y_true, weights=weights))**2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    elif metric == "logloss":
        # For binary classification (example implementation)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.average(weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _estimate_parameters(
    method: str,
    y_true: np.ndarray,
    weights: np.ndarray
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Estimate point parameters using specified method.

    Parameters
    ----------
    method : str
        Method for parameter estimation.
    y_true : np.ndarray
        Array of true values.
    weights : np.ndarray

    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        Dictionary containing estimated parameters.
    """
    if method == "mean":
        return {"point_estimate": np.average(y_true, weights=weights)}
    elif method == "median":
        return {"point_estimate": np.median(y_true)}
    elif method == "regression":
        # Simple linear regression example
        X = np.column_stack([np.ones_like(y_true), y_true])
        W = np.diag(weights)
        beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y_true)
        return {"intercept": beta[0], "slope": beta[1]}
    else:
        raise ValueError(f"Unknown estimation method: {method}")

def _compute_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute confidence interval for point estimates.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    weights : np.ndarray
        Array of observation weights.
    alpha : float
        Significance level for confidence interval.

    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        Dictionary containing confidence interval information.
    """
    residuals = y_true - y_pred
    std_err = np.sqrt(np.average(residuals**2, weights=weights) / len(y_true))
    z_score = 1.96  # For normal distribution, approximate for large n

    return {
        "lower_bound": y_pred - z_score * std_err,
        "upper_bound": y_pred + z_score * std_err,
        "alpha": alpha
    }

def estimation_point_fit(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    metric: str = "mse",
    estimation_method: str = "mean",
    alpha: float = 0.05,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Compute point estimates and confidence intervals with various options.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : Optional[np.ndarray]
        Array of predicted values. If None, will be estimated.
    weights : Optional[np.ndarray]
        Array of observation weights. If None, uniform weights are used.
    metric : str
        Name of the metric to compute ("mse", "mae", "r2", "logloss").
    estimation_method : str
        Method for parameter estimation ("mean", "median", "regression").
    alpha : float
        Significance level for confidence interval.
    custom_metric : Optional[Callable]
        Custom metric function if needed.

    Returns
    -------
    Dict[str, Union[float, np.ndarray, Dict]]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    y_true, y_pred, weights = _validate_inputs(y_true, y_pred, weights)

    # Estimate parameters if needed
    if y_pred is None:
        params = _estimate_parameters(estimation_method, y_true, weights)
        if estimation_method == "mean":
            y_pred = np.full_like(y_true, params["point_estimate"])
        elif estimation_method == "median":
            y_pred = np.full_like(y_true, params["point_estimate"])
        elif estimation_method == "regression":
            y_pred = params["intercept"] + params["slope"] * y_true

    # Compute metrics
    if custom_metric is not None:
        metric_value = custom_metric(y_true, y_pred)
    else:
        metric_value = _compute_metric(metric, y_true, y_pred, weights)

    # Compute confidence interval
    ci = _compute_confidence_interval(y_true, y_pred, weights, alpha)

    # Prepare output
    result = {
        "point_estimate": np.average(y_pred, weights=weights),
        "confidence_interval": ci,
        "metrics": {metric: metric_value}
    }

    params_used = {
        "estimation_method": estimation_method,
        "metric": metric,
        "alpha": alpha
    }

    warnings = []
    if np.any(np.isnan(y_true)):
        warnings.append("NaN values found in y_true and removed")
    if np.any(np.isnan(y_pred)):
        warnings.append("NaN values found in y_pred and removed")

    return {
        "result": result,
        "metrics": {metric: metric_value},
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
"""
y_true = np.array([1.2, 2.3, 3.4, 4.5])
y_pred = np.array([1.0, 2.5, 3.0, 4.8])
result = estimation_point_fit(y_true, y_pred)
print(result)
"""

################################################################################
# statistique_echantillon
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    data: np.ndarray,
    confidence_level: float = 0.95
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Data must be 1-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")

def _calculate_statistic(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float]
) -> float:
    """Calculate the sample statistic."""
    return statistic_func(data)

def _calculate_interval(
    data: np.ndarray,
    statistic: float,
    std_func: Callable[[np.ndarray], float],
    confidence_level: float = 0.95
) -> tuple:
    """Calculate the confidence interval."""
    alpha = 1 - confidence_level
    z_score = _get_z_score(alpha)
    std_error = std_func(data) / np.sqrt(len(data))
    margin_of_error = z_score * std_error
    return (statistic - margin_of_error, statistic + margin_of_error)

def _get_z_score(alpha: float) -> float:
    """Get the z-score for a given alpha level."""
    from scipy.stats import norm
    return norm.ppf(1 - alpha/2)

def statistique_echantillon_fit(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float] = np.mean,
    std_func: Callable[[np.ndarray], float] = np.std,
    confidence_level: float = 0.95
) -> Dict[str, Union[float, Dict]]:
    """
    Calculate a confidence interval for a sample statistic.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    statistic_func : Callable[[np.ndarray], float], optional
        Function to calculate the sample statistic (default: np.mean).
    std_func : Callable[[np.ndarray], float], optional
        Function to calculate the standard deviation (default: np.std).
    confidence_level : float, optional
        Confidence level for the interval (default: 0.95).

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing the result, metrics, parameters used, and warnings.

    Example
    -------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = statistique_echantillon_fit(data)
    """
    _validate_inputs(data, confidence_level)

    statistic = _calculate_statistic(data, statistic_func)
    interval = _calculate_interval(data, statistic, std_func, confidence_level)

    return {
        "result": {
            "statistic": statistic,
            "interval": interval
        },
        "metrics": {},
        "params_used": {
            "statistic_func": statistic_func.__name__,
            "std_func": std_func.__name__,
            "confidence_level": confidence_level
        },
        "warnings": []
    }

################################################################################
# parametre_population
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def parametre_population_fit(
    data: np.ndarray,
    confidence_level: float = 0.95,
    method: str = 'normal',
    metric: Union[str, Callable] = 'mean',
    solver: str = 'closed_form',
    normalizer: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Estimate population parameters with confidence intervals.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    confidence_level : float, default=0.95
        Confidence level for intervals (0 < confidence_level < 1).
    method : str, default='normal'
        Statistical method for interval calculation ('normal', 't', 'bootstrap').
    metric : str or callable, default='mean'
        Metric to estimate ('mean', 'median', 'custom').
    solver : str, default='closed_form'
        Solver method ('closed_form', 'iterative').
    normalizer : callable, optional
        Normalization function to apply to data.
    **kwargs :
        Additional parameters for specific methods.

    Returns:
    --------
    dict
        Dictionary containing results, metrics and parameters used.
    """
    # Validate inputs
    _validate_inputs(data, confidence_level)

    # Normalize data if specified
    normalized_data = _apply_normalization(data, normalizer)

    # Calculate statistic and interval
    result = _calculate_interval(
        normalized_data,
        confidence_level,
        method,
        metric,
        solver,
        **kwargs
    )

    return result

def _validate_inputs(data: np.ndarray, confidence_level: float) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if len(data.shape) != 1:
        raise ValueError("Data must be 1-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")

def _apply_normalization(
    data: np.ndarray,
    normalizer: Optional[Callable]
) -> np.ndarray:
    """Apply normalization to data if specified."""
    if normalizer is None:
        return data
    try:
        return normalizer(data)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _calculate_interval(
    data: np.ndarray,
    confidence_level: float,
    method: str,
    metric: Union[str, Callable],
    solver: str,
    **kwargs
) -> Dict[str, Any]:
    """Calculate confidence interval for population parameter."""
    # Calculate point estimate
    if callable(metric):
        point_estimate = metric(data)
    elif metric == 'mean':
        point_estimate = np.mean(data)
    elif metric == 'median':
        point_estimate = np.median(data)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Calculate interval based on method
    if method == 'normal':
        interval = _calculate_normal_interval(data, point_estimate, confidence_level)
    elif method == 't':
        interval = _calculate_t_interval(data, point_estimate, confidence_level)
    elif method == 'bootstrap':
        interval = _calculate_bootstrap_interval(data, point_estimate, confidence_level, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate metrics
    metrics = _calculate_metrics(data, point_estimate)

    return {
        'result': {
            'point_estimate': point_estimate,
            'confidence_interval': interval
        },
        'metrics': metrics,
        'params_used': {
            'method': method,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

def _calculate_normal_interval(
    data: np.ndarray,
    point_estimate: float,
    confidence_level: float
) -> tuple:
    """Calculate normal distribution based confidence interval."""
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    z_score = _get_z_score(confidence_level)
    margin_of_error = z_score * std_err
    return (point_estimate - margin_of_error, point_estimate + margin_of_error)

def _calculate_t_interval(
    data: np.ndarray,
    point_estimate: float,
    confidence_level: float
) -> tuple:
    """Calculate t-distribution based confidence interval."""
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    t_score = _get_t_score(confidence_level, len(data) - 1)
    margin_of_error = t_score * std_err
    return (point_estimate - margin_of_error, point_estimate + margin_of_error)

def _calculate_bootstrap_interval(
    data: np.ndarray,
    point_estimate: float,
    confidence_level: float,
    n_samples: int = 1000,
    sample_size: Optional[int] = None
) -> tuple:
    """Calculate bootstrap based confidence interval."""
    if sample_size is None:
        sample_size = len(data)

    boot_estimates = []
    for _ in range(n_samples):
        sample = np.random.choice(data, size=sample_size, replace=True)
        if callable(metric):
            boot_estimates.append(metric(sample))
        elif metric == 'mean':
            boot_estimates.append(np.mean(sample))
        elif metric == 'median':
            boot_estimates.append(np.median(sample))

    lower = np.percentile(boot_estimates, (1 - confidence_level)/2 * 100)
    upper = np.percentile(boot_estimates, (1 + confidence_level)/2 * 100)
    return (lower, upper)

def _calculate_metrics(
    data: np.ndarray,
    point_estimate: float
) -> Dict[str, float]:
    """Calculate various metrics for the estimate."""
    return {
        'sample_mean': np.mean(data),
        'sample_std': np.std(data, ddof=1),
        'sample_size': len(data)
    }

def _get_z_score(confidence_level: float) -> float:
    """Get z-score for given confidence level."""
    from scipy.stats import norm
    return norm.ppf((1 + confidence_level) / 2)

def _get_t_score(confidence_level: float, df: int) -> float:
    """Get t-score for given confidence level and degrees of freedom."""
    from scipy.stats import t
    return t.ppf((1 + confidence_level) / 2, df)

# Example usage:
"""
data = np.random.normal(0, 1, 100)
result = parametre_population_fit(
    data,
    confidence_level=0.95,
    method='normal',
    metric='mean'
)
"""

################################################################################
# hypotheses
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def hypotheses_fit(
    data: np.ndarray,
    target: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Compute hypothesis testing results with configurable parameters.

    Parameters
    ----------
    data : np.ndarray
        Input features array of shape (n_samples, n_features).
    target : np.ndarray
        Target values array of shape (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the input data.
    metric : str
        Metric to evaluate the hypothesis. Options: "mse", "mae", "r2", "logloss".
    distance : str
        Distance metric for hypothesis testing. Options: "euclidean", "manhattan", "cosine", "minkowski".
    solver : str
        Solver to use for hypothesis testing. Options: "closed_form", "gradient_descent", "newton", "coordinate_descent".
    regularization : Optional[str]
        Regularization type. Options: None, "l1", "l2", "elasticnet".
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
    >>> data = np.random.rand(100, 5)
    >>> target = np.random.rand(100)
    >>> result = hypotheses_fit(data, target, metric="mse", solver="gradient_descent")
    """
    # Validate inputs
    _validate_inputs(data, target)

    # Normalize data
    normalized_data = normalizer(data)

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

    # Choose solver
    if solver == "closed_form":
        params = _solve_closed_form(normalized_data, target)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(normalized_data, target, metric_func, tol, max_iter)
    elif solver == "newton":
        params = _solve_newton(normalized_data, target, metric_func, tol, max_iter)
    elif solver == "coordinate_descent":
        params = _solve_coordinate_descent(normalized_data, target, metric_func, tol, max_iter)
    else:
        raise ValueError("Invalid solver specified.")

    # Apply regularization if needed
    if regularization is not None:
        params = _apply_regularization(params, normalized_data, target, regularization)

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, target, params, metric_func)

    # Calculate confidence intervals
    confidence_intervals = _calculate_confidence_intervals(normalized_data, target, params)

    # Prepare result dictionary
    result = {
        "result": confidence_intervals,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if hasattr(normalizer, '__name__') else "custom",
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": _check_warnings(normalized_data, target)
    }

    return result

def _validate_inputs(data: np.ndarray, target: np.ndarray) -> None:
    """Validate input data and target."""
    if not isinstance(data, np.ndarray) or not isinstance(target, np.ndarray):
        raise TypeError("Data and target must be numpy arrays.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if target.ndim != 1:
        raise ValueError("Target must be a 1D array.")
    if data.shape[0] != target.shape[0]:
        raise ValueError("Data and target must have the same number of samples.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")
    if np.any(np.isnan(target)) or np.any(np.isinf(target)):
        raise ValueError("Target contains NaN or infinite values.")

def _get_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the metric function based on the specified metric."""
    metrics = {
        "mse": _mean_squared_error,
        "mae": _mean_absolute_error,
        "r2": _r_squared,
        "logloss": _log_loss
    }
    if metric not in metrics:
        raise ValueError(f"Invalid metric: {metric}")
    return metrics[metric]

def _get_distance(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the distance function based on the specified distance."""
    distances = {
        "euclidean": _euclidean_distance,
        "manhattan": _manhattan_distance,
        "cosine": _cosine_distance,
        "minkowski": _minkowski_distance
    }
    if distance not in distances:
        raise ValueError(f"Invalid distance: {distance}")
    return distances[distance]

def _solve_closed_form(data: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve the hypothesis using closed-form solution."""
    # Add intercept term
    data_with_intercept = np.column_stack([np.ones(data.shape[0]), data])
    # Calculate parameters using normal equation
    params = np.linalg.inv(data_with_intercept.T @ data_with_intercept) @ data_with_intercept.T @ target
    return params

def _solve_gradient_descent(
    data: np.ndarray,
    target: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve the hypothesis using gradient descent."""
    # Initialize parameters
    params = np.zeros(data.shape[1] + 1)  # +1 for intercept
    data_with_intercept = np.column_stack([np.ones(data.shape[0]), data])

    for _ in range(max_iter):
        gradients = _compute_gradients(data_with_intercept, target, params, metric_func)
        params -= tol * gradients
        if np.linalg.norm(gradients) < tol:
            break

    return params

def _solve_newton(
    data: np.ndarray,
    target: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve the hypothesis using Newton's method."""
    # Initialize parameters
    params = np.zeros(data.shape[1] + 1)  # +1 for intercept
    data_with_intercept = np.column_stack([np.ones(data.shape[0]), data])

    for _ in range(max_iter):
        gradients = _compute_gradients(data_with_intercept, target, params, metric_func)
        hessian = _compute_hessian(data_with_intercept, target, params, metric_func)
        params -= np.linalg.inv(hessian) @ gradients
        if np.linalg.norm(gradients) < tol:
            break

    return params

def _solve_coordinate_descent(
    data: np.ndarray,
    target: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve the hypothesis using coordinate descent."""
    # Initialize parameters
    params = np.zeros(data.shape[1] + 1)  # +1 for intercept
    data_with_intercept = np.column_stack([np.ones(data.shape[0]), data])

    for _ in range(max_iter):
        for i in range(params.shape[0]):
            # Temporarily remove the ith parameter
            temp_params = params.copy()
            temp_params[i] = 0
            # Compute the optimal value for the ith parameter
            residual = target - data_with_intercept @ temp_params
            numerator = data_with_intercept[:, i] @ residual
            denominator = data_with_intercept[:, i] @ data_with_intercept[:, i]
            params[i] = numerator / denominator
        if np.linalg.norm(_compute_gradients(data_with_intercept, target, params, metric_func)) < tol:
            break

    return params

def _apply_regularization(
    params: np.ndarray,
    data: np.ndarray,
    target: np.ndarray,
    regularization: str
) -> np.ndarray:
    """Apply regularization to the parameters."""
    data_with_intercept = np.column_stack([np.ones(data.shape[0]), data])
    if regularization == "l1":
        # Lasso regression
        params = _solve_lasso(data_with_intercept, target)
    elif regularization == "l2":
        # Ridge regression
        params = _solve_ridge(data_with_intercept, target)
    elif regularization == "elasticnet":
        # Elastic Net regression
        params = _solve_elastic_net(data_with_intercept, target)
    return params

def _calculate_metrics(
    data: np.ndarray,
    target: np.ndarray,
    params: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Calculate various metrics for the hypothesis."""
    data_with_intercept = np.column_stack([np.ones(data.shape[0]), data])
    predictions = data_with_intercept @ params

    metrics = {
        "mse": _mean_squared_error(target, predictions),
        "mae": _mean_absolute_error(target, predictions),
        "r2": _r_squared(target, predictions)
    }

    return metrics

def _calculate_confidence_intervals(
    data: np.ndarray,
    target: np.ndarray,
    params: np.ndarray
) -> Dict[str, float]:
    """Calculate confidence intervals for the parameters."""
    data_with_intercept = np.column_stack([np.ones(data.shape[0]), data])
    n_samples, n_features = data_with_intercept.shape
    residuals = target - data_with_intercept @ params
    mse = np.mean(residuals ** 2)
    variance = mse * np.linalg.inv(data_with_intercept.T @ data_with_intercept)
    std_errors = np.sqrt(np.diag(variance))

    confidence_intervals = {
        "params": params,
        "std_errors": std_errors,
        "confidence_intervals": {
            param: (params[i] - 1.96 * std_errors[i], params[i] + 1.96 * std_errors[i])
            for i, param in enumerate(["intercept"] + [f"feature_{i}" for i in range(n_features - 1)])
        }
    }

    return confidence_intervals

def _check_warnings(data: np.ndarray, target: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if data.shape[1] > data.shape[0]:
        warnings.append("Warning: Number of features exceeds number of samples.")
    if np.linalg.cond(np.column_stack([np.ones(data.shape[0]), data])) > 1e6:
        warnings.append("Warning: Data matrix is ill-conditioned.")
    return warnings

# Metric functions
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

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Log Loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Distance functions
def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _minkowski_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Minkowski distance."""
    return np.sum(np.abs(a - b) ** 3) ** (1/3)

# Regularization functions
def _solve_lasso(data: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve Lasso regression."""
    # Placeholder for actual implementation
    return np.linalg.inv(data.T @ data) @ data.T @ target

def _solve_ridge(data: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve Ridge regression."""
    # Placeholder for actual implementation
    return np.linalg.inv(data.T @ data + 1e-6 * np.eye(data.shape[1])) @ data.T @ target

def _solve_elastic_net(data: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve Elastic Net regression."""
    # Placeholder for actual implementation
    return np.linalg.inv(data.T @ data) @ data.T @ target

# Gradient and Hessian functions
def _compute_gradients(
    data: np.ndarray,
    target: np.ndarray,
    params: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Compute gradients for the given metric."""
    predictions = data @ params
    if metric_func == _mean_squared_error:
        return -2 * (target - predictions) @ data
    elif metric_func == _mean_absolute_error:
        return -np.sign(target - predictions) @ data
    elif metric_func == _r_squared:
        ss_res = np.sum((target - predictions) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        return -2 * (data.T @ (predictions - target) / ss_res) * (ss_tot / (n_samples - 1))
    else:
        raise ValueError("Gradient computation not implemented for the given metric.")

def _compute_hessian(
    data: np.ndarray,
    target: np.ndarray,
    params: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Compute Hessian for the given metric."""
    predictions = data @ params
    if metric_func == _mean_squared_error:
        return 2 * (data.T @ data)
    elif metric_func == _mean_absolute_error:
        return np.zeros((data.shape[1], data.shape[1]))
    elif metric_func == _r_squared:
        ss_res = np.sum((target - predictions) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        return (data.T @ data) / ss_res * (ss_tot / (n_samples - 1))
    else:
        raise ValueError("Hessian computation not implemented for the given metric.")

################################################################################
# test_statistique
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def test_statistique_fit(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    confidence_level: float = 0.95,
    normalization: str = "standard",
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    solver: str = "closed_form",
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_params: Optional[Dict] = None
) -> Dict:
    """
    Compute statistical test results with confidence intervals.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    statistic_func : Callable[[np.ndarray], float]
        Function to compute the test statistic.
    confidence_level : float, optional
        Confidence level for intervals (default: 0.95).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard').
    metric : Union[str, Callable], optional
        Metric for evaluation ('mse', 'mae', 'r2', 'logloss') or custom callable (default: 'mse').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default: 'closed_form').
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
    max_iter : int, optional
        Maximum iterations for iterative solvers (default: 1000).
    custom_params : Dict, optional
        Additional parameters for custom functions.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, statistic_func)

    # Normalize data
    normalized_data = _apply_normalization(data, normalization)

    # Compute test statistic
    statistic_value = statistic_func(normalized_data)

    # Estimate parameters based on solver choice
    params = _estimate_parameters(
        normalized_data,
        statistic_func,
        solver=solver,
        tol=tol,
        max_iter=max_iter
    )

    # Compute confidence interval
    ci = _compute_confidence_interval(
        statistic_value,
        normalized_data,
        confidence_level=confidence_level
    )

    # Compute metrics
    metrics = _compute_metrics(
        normalized_data,
        statistic_func,
        metric=metric
    )

    # Prepare output
    result = {
        "statistic": statistic_value,
        "confidence_interval": ci,
        "result": {
            "lower_bound": ci[0],
            "upper_bound": ci[1]
        },
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric if isinstance(metric, str) else "custom",
            "solver": solver,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": _check_warnings(normalized_data, params)
    }

    return result

def _validate_inputs(data: np.ndarray, statistic_func: Callable) -> None:
    """Validate input data and functions."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if not callable(statistic_func):
        raise TypeError("statistic_func must be a callable.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")

def _apply_normalization(
    data: np.ndarray,
    method: str = "standard"
) -> np.ndarray:
    """Apply normalization to data."""
    if method == "none":
        return data
    elif method == "standard":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _estimate_parameters(
    data: np.ndarray,
    statistic_func: Callable,
    solver: str = "closed_form",
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict:
    """Estimate parameters using specified solver."""
    if solver == "closed_form":
        return _closed_form_solution(data, statistic_func)
    elif solver == "gradient_descent":
        return _gradient_descent(
            data, statistic_func, tol=tol, max_iter=max_iter
        )
    elif solver == "newton":
        return _newton_method(
            data, statistic_func, tol=tol, max_iter=max_iter
        )
    elif solver == "coordinate_descent":
        return _coordinate_descent(
            data, statistic_func, tol=tol, max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _closed_form_solution(
    data: np.ndarray,
    statistic_func: Callable
) -> Dict:
    """Closed form solution for parameter estimation."""
    # Placeholder for actual implementation
    return {"params": np.zeros(data.shape[1])}

def _gradient_descent(
    data: np.ndarray,
    statistic_func: Callable,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict:
    """Gradient descent for parameter estimation."""
    # Placeholder for actual implementation
    return {"params": np.zeros(data.shape[1])}

def _newton_method(
    data: np.ndarray,
    statistic_func: Callable,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict:
    """Newton's method for parameter estimation."""
    # Placeholder for actual implementation
    return {"params": np.zeros(data.shape[1])}

def _coordinate_descent(
    data: np.ndarray,
    statistic_func: Callable,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict:
    """Coordinate descent for parameter estimation."""
    # Placeholder for actual implementation
    return {"params": np.zeros(data.shape[1])}

def _compute_confidence_interval(
    statistic_value: float,
    data: np.ndarray,
    confidence_level: float = 0.95
) -> tuple:
    """Compute confidence interval for the statistic."""
    # Placeholder for actual implementation
    alpha = (1 - confidence_level) / 2
    margin_of_error = 1.96 * np.std(data) / np.sqrt(len(data))
    return (statistic_value - margin_of_error, statistic_value + margin_of_error)

def _compute_metrics(
    data: np.ndarray,
    statistic_func: Callable,
    metric: Union[str, Callable] = "mse"
) -> Dict:
    """Compute evaluation metrics."""
    if isinstance(metric, str):
        if metric == "mse":
            return {"mse": np.mean((data - statistic_func(data))**2)}
        elif metric == "mae":
            return {"mae": np.mean(np.abs(data - statistic_func(data)))}
        elif metric == "r2":
            ss_res = np.sum((data - statistic_func(data))**2)
            ss_tot = np.sum((data - np.mean(data))**2)
            return {"r2": 1 - (ss_res / ss_tot)}
        elif metric == "logloss":
            return {"logloss": -np.mean(data * np.log(statistic_func(data)))}
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        return {"custom": metric(data, statistic_func(data))}

def _check_warnings(
    data: np.ndarray,
    params: Dict
) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isnan(params["params"])):
        warnings.append("Parameters contain NaN values.")
    if np.any(np.isinf(data)):
        warnings.append("Data contains infinite values.")
    return warnings

# Example usage:
"""
data = np.random.randn(100, 5)
statistic_func = lambda x: np.mean(x, axis=0)

result = test_statistique_fit(
    data=data,
    statistic_func=statistic_func,
    confidence_level=0.95,
    normalization="standard",
    metric="mse"
)
"""

################################################################################
# p_value
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(
    observed: np.ndarray,
    expected: Optional[np.ndarray] = None,
    alternative: str = 'two-sided',
    metric: Union[str, Callable] = 'mse'
) -> None:
    """
    Validate input data for p_value computation.

    Parameters
    ----------
    observed : np.ndarray
        Array of observed values.
    expected : Optional[np.ndarray]
        Array of expected values. If None, uses mean of observed.
    alternative : str
        Type of alternative hypothesis ('two-sided', 'less', 'greater').
    metric : Union[str, Callable]
        Metric function or name ('mse', 'mae', etc.).

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if observed.ndim != 1:
        raise ValueError("observed must be a 1D array")
    if expected is not None and observed.shape != expected.shape:
        raise ValueError("observed and expected must have the same shape")
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("alternative must be 'two-sided', 'less' or 'greater'")
    if isinstance(metric, str) and metric not in ['mse', 'mae']:
        raise ValueError("metric must be a callable or one of ['mse', 'mae']")

def _compute_statistic(
    observed: np.ndarray,
    expected: Optional[np.ndarray] = None,
    metric: Union[str, Callable] = 'mse'
) -> float:
    """
    Compute the test statistic based on observed and expected values.

    Parameters
    ----------
    observed : np.ndarray
        Array of observed values.
    expected : Optional[np.ndarray]
        Array of expected values. If None, uses mean of observed.
    metric : Union[str, Callable]
        Metric function or name ('mse', 'mae', etc.).

    Returns
    ------
    float
        Computed test statistic.
    """
    if expected is None:
        expected = np.mean(observed)

    if isinstance(metric, str):
        if metric == 'mse':
            return np.mean((observed - expected) ** 2)
        elif metric == 'mae':
            return np.mean(np.abs(observed - expected))
    else:
        return metric(observed, expected)

def _compute_p_value(
    statistic: float,
    alternative: str = 'two-sided',
    distribution: Callable = np.random.normal
) -> float:
    """
    Compute the p-value based on test statistic and distribution.

    Parameters
    ----------
    statistic : float
        Computed test statistic.
    alternative : str
        Type of alternative hypothesis ('two-sided', 'less', 'greater').
    distribution : Callable
        Distribution function for p-value computation.

    Returns
    ------
    float
        Computed p-value.
    """
    if alternative == 'two-sided':
        return 2 * min(distribution.sf(statistic), distribution.sf(-statistic))
    elif alternative == 'less':
        return distribution.cdf(statistic)
    else:  # 'greater'
        return 1 - distribution.cdf(statistic)

def p_value_fit(
    observed: np.ndarray,
    expected: Optional[np.ndarray] = None,
    alternative: str = 'two-sided',
    metric: Union[str, Callable] = 'mse',
    distribution: Callable = np.random.normal,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute p-value for confidence interval.

    Parameters
    ----------
    observed : np.ndarray
        Array of observed values.
    expected : Optional[np.ndarray]
        Array of expected values. If None, uses mean of observed.
    alternative : str
        Type of alternative hypothesis ('two-sided', 'less', 'greater').
    metric : Union[str, Callable]
        Metric function or name ('mse', 'mae', etc.).
    distribution : Callable
        Distribution function for p-value computation.
    **kwargs :
        Additional keyword arguments passed to distribution.

    Returns
    ------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> observed = np.array([1.0, 2.0, 3.0])
    >>> p_value_fit(observed)
    {
        'result': 0.5,
        'metrics': {'statistic': 1.0},
        'params_used': {
            'alternative': 'two-sided',
            'metric': 'mse',
            'distribution': '<class 'numpy.random._generator.Generator'>'
        },
        'warnings': []
    }
    """
    _validate_inputs(observed, expected, alternative, metric)

    statistic = _compute_statistic(observed, expected, metric)
    p_value = _compute_p_value(statistic, alternative, distribution)

    return {
        'result': p_value,
        'metrics': {'statistic': statistic},
        'params_used': {
            'alternative': alternative,
            'metric': metric if isinstance(metric, str) else 'custom',
            'distribution': str(distribution)
        },
        'warnings': []
    }

################################################################################
# interpretation
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def interpretation_fit(
    data: np.ndarray,
    confidence_level: float = 0.95,
    method: str = 'normal',
    metric: Union[str, Callable] = 'mean',
    solver: str = 'closed_form',
    normalize: bool = True,
    custom_normalization: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute confidence intervals for given data with various customization options.

    Parameters:
    -----------
    data : np.ndarray
        Input data for which to compute confidence intervals.
    confidence_level : float, default=0.95
        Confidence level for the intervals (e.g., 0.95 for 95% confidence).
    method : str, default='normal'
        Method to compute intervals: 'normal', 't', 'bootstrap'.
    metric : str or callable, default='mean'
        Metric to compute intervals for: 'mean', 'median', or custom callable.
    solver : str, default='closed_form'
        Solver method: 'closed_form', 'gradient_descent'.
    normalize : bool, default=True
        Whether to normalize the data before computation.
    custom_normalization : callable, optional
        Custom normalization function if normalize=True and default normalization is not desired.
    **kwargs : dict
        Additional keyword arguments for specific methods.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, confidence_level)

    # Normalize data if required
    normalized_data = _normalize_data(data, normalize, custom_normalization)

    # Choose metric function
    if isinstance(metric, str):
        if metric == 'mean':
            metric_func = np.mean
        elif metric == 'median':
            metric_func = np.median
        else:
            raise ValueError("Metric must be 'mean', 'median', or a callable.")
    else:
        metric_func = metric

    # Compute statistic
    statistic = metric_func(normalized_data)

    # Choose method to compute confidence intervals
    if method == 'normal':
        lower, upper = _compute_normal_interval(normalized_data, statistic, confidence_level)
    elif method == 't':
        lower, upper = _compute_t_interval(normalized_data, statistic, confidence_level)
    elif method == 'bootstrap':
        lower, upper = _compute_bootstrap_interval(normalized_data, statistic, confidence_level, **kwargs)
    else:
        raise ValueError("Method must be 'normal', 't', or 'bootstrap'.")

    # Compute metrics
    metrics = _compute_metrics(normalized_data, statistic)

    # Prepare output
    result = {
        'statistic': statistic,
        'lower_bound': lower,
        'upper_bound': upper
    }

    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'confidence_level': confidence_level,
            'method': method,
            'metric': metric,
            'solver': solver,
            'normalize': normalize
        },
        'warnings': _check_warnings(normalized_data, method)
    }

    return output

def _validate_inputs(data: np.ndarray, confidence_level: float) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if len(data.shape) != 1:
        raise ValueError("Data must be one-dimensional.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")
    if not 0 < confidence_level <= 1:
        raise ValueError("Confidence level must be between 0 and 1.")

def _normalize_data(
    data: np.ndarray,
    normalize: bool,
    custom_normalization: Optional[Callable] = None
) -> np.ndarray:
    """Normalize data if required."""
    if not normalize:
        return data
    if custom_normalization is not None:
        return custom_normalization(data)
    # Default standardization
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def _compute_normal_interval(
    data: np.ndarray,
    statistic: float,
    confidence_level: float
) -> tuple:
    """Compute normal-based confidence interval."""
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    z_score = _get_z_score(confidence_level)
    margin_of_error = z_score * std_err
    lower = statistic - margin_of_error
    upper = statistic + margin_of_error
    return lower, upper

def _compute_t_interval(
    data: np.ndarray,
    statistic: float,
    confidence_level: float
) -> tuple:
    """Compute t-based confidence interval."""
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    t_score = _get_t_score(confidence_level, len(data) - 1)
    margin_of_error = t_score * std_err
    lower = statistic - margin_of_error
    upper = statistic + margin_of_error
    return lower, upper

def _compute_bootstrap_interval(
    data: np.ndarray,
    statistic_func: Callable,
    confidence_level: float,
    n_bootstrap: int = 1000,
    **kwargs
) -> tuple:
    """Compute bootstrap-based confidence interval."""
    n = len(data)
    boot_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats[i] = statistic_func(sample)

    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower = np.percentile(boot_stats, lower_percentile)
    upper = np.percentile(boot_stats, upper_percentile)

    return lower, upper

def _compute_metrics(
    data: np.ndarray,
    statistic: float
) -> Dict[str, float]:
    """Compute various metrics for the confidence interval."""
    return {
        'sample_mean': np.mean(data),
        'sample_std': np.std(data, ddof=1),
        'statistic_value': statistic
    }

def _check_warnings(
    data: np.ndarray,
    method: str
) -> list:
    """Check for potential warnings during computation."""
    warnings = []

    if len(data) < 30 and method == 'normal':
        warnings.append("Sample size is small; consider using t-based intervals.")

    if np.var(data) == 0:
        warnings.append("Data has zero variance; intervals may be trivial.")

    return warnings

def _get_z_score(confidence_level: float) -> float:
    """Get z-score for given confidence level (normal distribution)."""
    from scipy.stats import norm
    return norm.ppf((1 + confidence_level) / 2)

def _get_t_score(confidence_level: float, df: int) -> float:
    """Get t-score for given confidence level and degrees of freedom."""
    from scipy.stats import t
    return t.ppf((1 + confidence_level) / 2, df)

################################################################################
# limites_intervalle
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def limites_intervalle_fit(
    data: np.ndarray,
    confidence_level: float = 0.95,
    method: str = 'normal',
    alpha: Optional[float] = None,
    metric: Union[str, Callable] = 'mean',
    normalization: str = 'none',
    custom_func: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Calculate confidence interval limits for given data.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    confidence_level : float, optional
        Confidence level (0 < confidence_level < 1), by default 0.95.
    method : str, optional
        Method for confidence interval calculation ('normal', 't', 'bootstrap'), by default 'normal'.
    alpha : float, optional
        Significance level (0 < alpha < 1). If None, calculated from confidence_level.
    metric : Union[str, Callable], optional
        Metric to use ('mean', 'median', custom callable), by default 'mean'.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax'), by default 'none'.
    custom_func : Callable, optional
        Custom function for confidence interval calculation.
    **kwargs :
        Additional keyword arguments passed to internal functions.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': tuple of (lower_limit, upper_limit)
        - 'metrics': dictionary of calculated metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warning messages

    Examples
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = limites_intervalle_fit(data, confidence_level=0.95)
    """
    # Validate inputs
    _validate_inputs(data, confidence_level, method, alpha)

    # Prepare parameters dictionary
    params_used = {
        'confidence_level': confidence_level,
        'method': method,
        'metric': metric,
        'normalization': normalization
    }

    # Normalize data if required
    normalized_data = _normalize_data(data, normalization)

    # Calculate metric
    metric_value, metrics = _calculate_metric(normalized_data, metric)

    # Calculate confidence interval
    if custom_func is not None:
        lower_limit, upper_limit = _custom_confidence_interval(
            normalized_data,
            metric_value,
            custom_func,
            **kwargs
        )
    else:
        lower_limit, upper_limit = _calculate_confidence_interval(
            normalized_data,
            metric_value,
            method,
            confidence_level,
            **kwargs
        )

    # Prepare result dictionary
    result = {
        'result': (lower_limit, upper_limit),
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    confidence_level: float,
    method: str,
    alpha: Optional[float]
) -> None:
    """Validate input parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if len(data.shape) != 1:
        raise ValueError("Data must be 1-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")
    if not (0 < confidence_level < 1):
        raise ValueError("Confidence level must be between 0 and 1")
    if alpha is not None and not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1")
    if method not in ['normal', 't', 'bootstrap']:
        raise ValueError("Invalid method specified")

def _normalize_data(
    data: np.ndarray,
    normalization: str
) -> np.ndarray:
    """Normalize data according to specified method."""
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
    else:
        raise ValueError("Invalid normalization method")

def _calculate_metric(
    data: np.ndarray,
    metric: Union[str, Callable]
) -> tuple:
    """Calculate specified metric for the data."""
    metrics = {}

    if callable(metric):
        value = metric(data)
        metrics['custom_metric'] = value
    elif metric == 'mean':
        value = np.mean(data)
        metrics['mean'] = value
    elif metric == 'median':
        value = np.median(data)
        metrics['median'] = value
    else:
        raise ValueError("Invalid metric specified")

    return value, metrics

def _calculate_confidence_interval(
    data: np.ndarray,
    metric_value: float,
    method: str,
    confidence_level: float,
    **kwargs
) -> tuple:
    """Calculate confidence interval using specified method."""
    alpha = 1 - confidence_level
    n = len(data)

    if method == 'normal':
        std_err = np.std(data, ddof=1) / np.sqrt(n)
        z_score = _z_score(alpha/2)
        lower_limit = metric_value - z_score * std_err
        upper_limit = metric_value + z_score * std_err

    elif method == 't':
        std_err = np.std(data, ddof=1) / np.sqrt(n)
        t_score = _t_score(alpha/2, n-1)
        lower_limit = metric_value - t_score * std_err
        upper_limit = metric_value + t_score * std_err

    elif method == 'bootstrap':
        lower_limit, upper_limit = _bootstrap_confidence_interval(
            data,
            metric_value,
            confidence_level,
            **kwargs
        )

    return lower_limit, upper_limit

def _custom_confidence_interval(
    data: np.ndarray,
    metric_value: float,
    custom_func: Callable,
    **kwargs
) -> tuple:
    """Calculate confidence interval using custom function."""
    return custom_func(data, metric_value, **kwargs)

def _z_score(alpha: float) -> float:
    """Calculate z-score for normal distribution."""
    from scipy.stats import norm
    return norm.ppf(1 - alpha/2)

def _t_score(alpha: float, df: int) -> float:
    """Calculate t-score for Student's t-distribution."""
    from scipy.stats import t
    return t.ppf(1 - alpha/2, df)

def _bootstrap_confidence_interval(
    data: np.ndarray,
    metric_value: float,
    confidence_level: float,
    n_bootstrap: int = 1000,
    **kwargs
) -> tuple:
    """Calculate confidence interval using bootstrap method."""
    n = len(data)
    bootstrap_metrics = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_metrics[i] = _calculate_metric(sample, 'mean')[0]

    alpha = (1 - confidence_level) / 2
    lower_percentile = int(alpha * n_bootstrap)
    upper_percentile = int((1 - alpha) * n_bootstrap)

    lower_limit = np.percentile(bootstrap_metrics, lower_percentile)
    upper_limit = np.percentile(bootstrap_metrics, upper_percentile)

    return lower_limit, upper_limit

################################################################################
# biais_estimation
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def biais_estimation_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    normalization: str = 'none',
    confidence_level: float = 0.95,
    solver: str = 'closed_form',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Estimate confidence intervals for bias between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    metric : str or callable, optional
        Metric to use for bias estimation. Can be 'mse', 'mae', 'r2', or a custom callable.
    normalization : str, optional
        Normalization method. Can be 'none', 'standard', 'minmax', or 'robust'.
    confidence_level : float, optional
        Confidence level for the interval (0 < confidence_level < 1).
    solver : str, optional
        Solver method. Can be 'closed_form', 'gradient_descent', or other supported methods.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    **kwargs :
        Additional keyword arguments for specific solvers or metrics.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], str]]
        Dictionary containing:
        - 'result': estimated bias and confidence interval
        - 'metrics': computed metrics
        - 'params_used': parameters used in the computation
        - 'warnings': any warnings generated during computation

    Examples
    --------
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 1.9, 3.2])
    >>> result = biais_estimation_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize data if required
    y_true_norm, y_pred_norm = _apply_normalization(y_true, y_pred, normalization)

    # Choose metric
    if isinstance(metric, str):
        if custom_metric is not None:
            raise ValueError("Cannot use both built-in metric and custom metric.")
        metric_func = _get_metric_function(metric)
    else:
        if callable(metric):
            metric_func = metric
        else:
            raise ValueError("Metric must be a string or callable.")

    # Compute bias and confidence interval
    if solver == 'closed_form':
        bias, ci = _closed_form_solver(y_true_norm, y_pred_norm, metric_func, confidence_level)
    else:
        raise ValueError(f"Solver {solver} not supported.")

    # Compute additional metrics
    metrics = _compute_metrics(y_true_norm, y_pred_norm)

    # Prepare output
    result = {
        'result': {
            'bias': bias,
            'confidence_interval': ci
        },
        'metrics': metrics,
        'params_used': {
            'metric': metric if isinstance(metric, str) else 'custom',
            'normalization': normalization,
            'confidence_level': confidence_level,
            'solver': solver
        },
        'warnings': []
    }

    return result

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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalization to input arrays."""
    if method == 'none':
        return y_true, y_pred
    elif method == 'standard':
        mean = np.mean(y_true)
        std = np.std(y_true)
        if std == 0:
            raise ValueError("Standard deviation is zero, cannot normalize.")
        y_true_norm = (y_true - mean) / std
        y_pred_norm = (y_pred - mean) / std
    elif method == 'minmax':
        min_val = np.min(y_true)
        max_val = np.max(y_true)
        if min_val == max_val:
            raise ValueError("Min and max values are equal, cannot normalize.")
        y_true_norm = (y_true - min_val) / (max_val - min_val)
        y_pred_norm = (y_pred - min_val) / (max_val - min_val)
    elif method == 'robust':
        median = np.median(y_true)
        iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        if iqr == 0:
            raise ValueError("IQR is zero, cannot normalize.")
        y_true_norm = (y_true - median) / iqr
        y_pred_norm = (y_pred - median) / iqr
    else:
        raise ValueError(f"Normalization method {method} not supported.")
    return y_true_norm, y_pred_norm

def _get_metric_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the metric function based on the input string."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }
    if metric not in metrics:
        raise ValueError(f"Metric {metric} not supported.")
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

def _closed_form_solver(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    confidence_level: float
) -> tuple[float, tuple[float, float]]:
    """Closed form solver for bias estimation."""
    residuals = y_true - y_pred
    bias = np.mean(residuals)
    std_error = np.std(residuals) / np.sqrt(len(y_true))
    z_score = _get_z_score(confidence_level)
    ci_lower = bias - z_score * std_error
    ci_upper = bias + z_score * std_error
    return bias, (ci_lower, ci_upper)

def _get_z_score(confidence_level: float) -> float:
    """Get z-score for given confidence level."""
    from scipy.stats import norm
    return norm.ppf((1 + confidence_level) / 2)

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute additional metrics."""
    return {
        'mse': _mean_squared_error(y_true, y_pred),
        'mae': _mean_absolute_error(y_true, y_pred),
        'r2': _r_squared(y_true, y_pred)
    }

################################################################################
# puissance_test
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    sample_size: int,
    effect_size: float,
    alpha: float,
    power: float
) -> None:
    """
    Validate the inputs for the power analysis.

    Parameters
    ----------
    sample_size : int
        The sample size.
    effect_size : float
        The effect size to detect.
    alpha : float
        The significance level (Type I error rate).
    power : float
        The desired statistical power (1 - Type II error rate).

    Raises
    ------
    ValueError
        If any of the inputs are invalid.
    """
    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1.")
    if not (0 < power < 1):
        raise ValueError("Power must be between 0 and 1.")
    if effect_size <= 0:
        raise ValueError("Effect size must be positive.")

def _compute_non_centrality_parameter(
    effect_size: float,
    sample_size: int
) -> float:
    """
    Compute the non-centrality parameter for a given effect size and sample size.

    Parameters
    ----------
    effect_size : float
        The effect size to detect.
    sample_size : int
        The sample size.

    Returns
    ------
    float
        The non-centrality parameter.
    """
    return effect_size * np.sqrt(sample_size)

def _compute_critical_value(
    alpha: float,
    df: int
) -> float:
    """
    Compute the critical value for a given significance level and degrees of freedom.

    Parameters
    ----------
    alpha : float
        The significance level (Type I error rate).
    df : int
        The degrees of freedom.

    Returns
    ------
    float
        The critical value.
    """
    from scipy.stats import t
    return t.ppf(1 - alpha, df)

def _compute_power(
    non_centrality: float,
    df: int
) -> float:
    """
    Compute the statistical power for a given non-centrality parameter and degrees of freedom.

    Parameters
    ----------
    non_centrality : float
        The non-centrality parameter.
    df : int
        The degrees of freedom.

    Returns
    ------
    float
        The statistical power.
    """
    from scipy.stats import noncentral_t
    critical_value = _compute_critical_value(0.05, df)
    return 1 - noncentral_t.cdf(critical_value, df, non_centrality)

def puissance_test_fit(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    sample_size: Optional[int] = None,
    metric: str = 't_test',
    solver: str = 'closed_form',
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute the power of a statistical test or determine the required sample size for a given power.

    Parameters
    ----------
    effect_size : float
        The effect size to detect.
    alpha : float, optional
        The significance level (Type I error rate), by default 0.05.
    power : float, optional
        The desired statistical power (1 - Type II error rate), by default 0.8.
    sample_size : int, optional
        The sample size to use for the power calculation. If None, the function will compute the required sample size.
    metric : str, optional
        The metric to use for the power calculation. Supported values: 't_test', by default 't_test'.
    solver : str, optional
        The solver to use for the power calculation. Supported values: 'closed_form', by default 'closed_form'.
    custom_metric : Callable, optional
        A custom metric function to use for the power calculation.
    **kwargs : dict
        Additional keyword arguments passed to the solver.

    Returns
    ------
    Dict[str, Any]
        A dictionary containing the results of the power analysis.

    Examples
    --------
    >>> puissance_test_fit(effect_size=0.5, alpha=0.05, power=0.8)
    {
        'result': 49,
        'metrics': {'power': 0.8, 'alpha': 0.05},
        'params_used': {'effect_size': 0.5, 'alpha': 0.05, 'power': 0.8},
        'warnings': []
    }
    """
    _validate_inputs(effect_size, alpha, power)

    if sample_size is not None:
        non_centrality = _compute_non_centrality_parameter(effect_size, sample_size)
        computed_power = _compute_power(non_centrality, sample_size - 1)

        return {
            'result': computed_power,
            'metrics': {'power': computed_power, 'alpha': alpha},
            'params_used': {
                'effect_size': effect_size,
                'alpha': alpha,
                'power': power,
                'sample_size': sample_size
            },
            'warnings': []
        }
    else:
        from scipy.optimize import fsolve
        def equation(sample_size):
            non_centrality = _compute_non_centrality_parameter(effect_size, sample_size)
            computed_power = _compute_power(non_centrality, sample_size - 1)
            return computed_power - power

        initial_guess = int((effect_size / (np.sqrt(2) * np.sqrt(power))) ** 2)
        required_sample_size = int(fsolve(equation, initial_guess)[0])

        return {
            'result': required_sample_size,
            'metrics': {'power': power, 'alpha': alpha},
            'params_used': {
                'effect_size': effect_size,
                'alpha': alpha,
                'power': power
            },
            'warnings': []
        }

################################################################################
# intervalle_confiance_bayesien
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def intervalle_confiance_bayesien_fit(
    data: np.ndarray,
    prior_mean: float = 0.0,
    prior_variance: float = 1.0,
    confidence_level: float = 0.95,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = "mse",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    solver: str = "closed_form",
    tolerance: float = 1e-6,
    max_iterations: int = 1000
) -> Dict[str, Any]:
    """
    Calculate Bayesian confidence interval for given data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    prior_mean : float, optional
        Mean of the prior distribution (default is 0.0).
    prior_variance : float, optional
        Variance of the prior distribution (default is 1.0).
    confidence_level : float, optional
        Confidence level for the interval (default is 0.95).
    normalizer : Callable, optional
        Function to normalize the data (default is None).
    metric : str, optional
        Metric to evaluate performance ("mse", "mae", "r2") (default is "mse").
    custom_metric : Callable, optional
        Custom metric function (default is None).
    solver : str, optional
        Solver method ("closed_form", "gradient_descent") (default is "closed_form").
    tolerance : float, optional
        Tolerance for convergence (default is 1e-6).
    max_iterations : int, optional
        Maximum number of iterations (default is 1000).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, prior_variance, confidence_level)

    # Normalize data if a normalizer is provided
    normalized_data = _apply_normalization(data, normalizer)

    # Calculate posterior parameters
    posterior_mean, posterior_variance = _calculate_posterior(
        normalized_data,
        prior_mean,
        prior_variance
    )

    # Calculate confidence interval
    lower_bound, upper_bound = _calculate_confidence_interval(
        posterior_mean,
        posterior_variance,
        confidence_level
    )

    # Calculate metrics
    metrics = _calculate_metrics(
        normalized_data,
        posterior_mean,
        metric,
        custom_metric
    )

    # Prepare results dictionary
    result = {
        "result": {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        },
        "metrics": metrics,
        "params_used": {
            "prior_mean": prior_mean,
            "prior_variance": prior_variance,
            "confidence_level": confidence_level,
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "solver": solver
        },
        "warnings": []
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    prior_variance: float,
    confidence_level: float
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if prior_variance <= 0:
        raise ValueError("Prior variance must be positive.")
    if not (0 < confidence_level < 1):
        raise ValueError("Confidence level must be between 0 and 1.")

def _apply_normalization(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to the data if a normalizer is provided."""
    if normalizer is not None:
        return normalizer(data)
    return data

def _calculate_posterior(
    data: np.ndarray,
    prior_mean: float,
    prior_variance: float
) -> tuple[float, float]:
    """Calculate posterior mean and variance."""
    n = len(data)
    sample_mean = np.mean(data)
    sample_variance = np.var(data, ddof=1)

    posterior_variance = 1 / (1/prior_variance + n/sample_variance)
    posterior_mean = (prior_mean/prior_variance + n*sample_mean/sample_variance) * posterior_variance

    return posterior_mean, posterior_variance

def _calculate_confidence_interval(
    mean: float,
    variance: float,
    confidence_level: float
) -> tuple[float, float]:
    """Calculate the confidence interval."""
    z_score = _get_z_score(confidence_level)
    margin_of_error = z_score * np.sqrt(variance)

    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return lower_bound, upper_bound

def _get_z_score(confidence_level: float) -> float:
    """Get the z-score for a given confidence level."""
    from scipy.stats import norm
    return norm.ppf((1 + confidence_level) / 2)

def _calculate_metrics(
    data: np.ndarray,
    mean: float,
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate metrics based on the chosen metric."""
    predictions = np.full_like(data, mean)

    if custom_metric is not None:
        return {"custom_metric": custom_metric(data, predictions)}

    metrics_dict = {}
    if metric == "mse":
        metrics_dict["mse"] = np.mean((data - predictions) ** 2)
    elif metric == "mae":
        metrics_dict["mae"] = np.mean(np.abs(data - predictions))
    elif metric == "r2":
        ss_res = np.sum((data - predictions) ** 2)
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        metrics_dict["r2"] = 1 - (ss_res / ss_tot)

    return metrics_dict
