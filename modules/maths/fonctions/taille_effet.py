"""
Quantix – Module taille_effet
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# cohen_d
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(
    group1: np.ndarray,
    group2: np.ndarray,
    normalize: str = "standard",
) -> None:
    """Validate input arrays and normalization choice."""
    if not isinstance(group1, np.ndarray) or not isinstance(group2, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if group1.ndim != 1 or group2.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional")
    if len(group1) != len(group2):
        raise ValueError("Groups must have the same length")
    if normalize not in ["none", "standard", "robust"]:
        raise ValueError("normalize must be 'none', 'standard', or 'robust'")

def _apply_normalization(
    group: np.ndarray,
    normalize: str = "standard",
) -> np.ndarray:
    """Apply normalization to a group."""
    if normalize == "none":
        return group
    elif normalize == "standard":
        mean = np.mean(group)
        std = np.std(group)
        if std == 0:
            return group - mean
        return (group - mean) / std
    elif normalize == "robust":
        median = np.median(group)
        iqr = np.percentile(group, 75) - np.percentile(group, 25)
        if iqr == 0:
            return group - median
        return (group - median) / iqr

def _compute_cohen_d(
    group1: np.ndarray,
    group2: np.ndarray,
) -> float:
    """Compute Cohen's d effect size."""
    diff = group1 - group2
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) +
                          (len(group2) - 1) * np.var(group2, ddof=1)) /
                         (len(group1) + len(group2) - 2))
    if pooled_std == 0:
        return 0.0
    return np.mean(diff) / pooled_std

def cohen_d_fit(
    group1: np.ndarray,
    group2: np.ndarray,
    normalize: str = "standard",
) -> Dict[str, Any]:
    """
    Compute Cohen's d effect size between two groups.

    Parameters:
    -----------
    group1 : np.ndarray
        First group of observations.
    group2 : np.ndarray
        Second group of observations.
    normalize : str, optional
        Normalization method ('none', 'standard', or 'robust').

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_inputs(group1, group2, normalize)

    # Normalize groups
    norm_group1 = _apply_normalization(group1, normalize)
    norm_group2 = _apply_normalization(group2, normalize)

    # Compute Cohen's d
    result = _compute_cohen_d(norm_group1, norm_group2)

    # Prepare output
    output = {
        "result": result,
        "metrics": {},
        "params_used": {
            "normalize": normalize
        },
        "warnings": []
    }

    return output

# Example usage:
# group1 = np.array([1, 2, 3])
# group2 = np.array([4, 5, 6])
# result = cohen_d_fit(group1, group2)

################################################################################
# eta_squared
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: str = "none",
) -> None:
    """Validate input arrays and normalization choice."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs must not contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs must not contain infinite values.")
    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("Invalid normalization choice.")

def _normalize_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize input data based on the specified method."""
    if normalize == "none":
        return y_true, y_pred
    elif normalize == "standard":
        mean = np.mean(y_true)
        std = np.std(y_true)
        if std == 0:
            raise ValueError("Standard normalization requires non-zero standard deviation.")
        y_true_norm = (y_true - mean) / std
        y_pred_norm = (y_pred - mean) / std
    elif normalize == "minmax":
        min_val = np.min(y_true)
        max_val = np.max(y_true)
        if min_val == max_val:
            raise ValueError("Min-max normalization requires non-constant values.")
        y_true_norm = (y_true - min_val) / (max_val - min_val)
        y_pred_norm = (y_pred - min_val) / (max_val - min_val)
    elif normalize == "robust":
        median = np.median(y_true)
        iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        if iqr == 0:
            raise ValueError("Robust normalization requires non-zero IQR.")
        y_true_norm = (y_true - median) / iqr
        y_pred_norm = (y_pred - median) / iqr
    return y_true_norm, y_pred_norm

def _compute_eta_squared(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute eta squared (η²) effect size."""
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_model = np.sum((y_pred - np.mean(y_true)) ** 2)
    if ss_total == 0:
        raise ValueError("Total sum of squares cannot be zero.")
    eta_squared = ss_model / ss_total
    return eta_squared

def eta_squared_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: str = "none",
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Compute eta squared (η²) effect size between true and predicted values.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    normalize : str, optional (default="none")
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : Callable, optional
        Custom metric function that takes (y_true, y_pred) and returns a float.

    Returns:
    --------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(y_true, y_pred, normalize)
    y_true_norm, y_pred_norm = _normalize_data(y_true, y_pred, normalize)

    result = _compute_eta_squared(y_true_norm, y_pred_norm)
    metrics = {}
    if metric is not None:
        try:
            metrics["custom_metric"] = metric(y_true, y_pred)
        except Exception as e:
            metrics["custom_metric"] = f"Error computing custom metric: {str(e)}"

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {"normalize": normalize},
        "warnings": [],
    }

# Example usage:
# eta_squared_fit(np.array([1, 2, 3]), np.array([1.1, 2.0, 3.2]))

################################################################################
# omega_squared
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def omega_squared_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalization: str = 'standard',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Calculate omega squared effect size between true and predicted values.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    normalization : str, optional (default='standard')
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : str or callable, optional (default='mse')
        Metric to use: 'mse', 'mae', 'r2', or custom callable.
    solver : str, optional (default='closed_form')
        Solver method: 'closed_form', 'gradient_descent', or 'newton'.
    regularization : str, optional (default=None)
        Regularization type: None, 'l1', 'l2', or 'elasticnet'.
    tol : float, optional (default=1e-6)
        Tolerance for convergence.
    max_iter : int, optional (default=1000)
        Maximum number of iterations.
    custom_metric : callable, optional (default=None)
        Custom metric function if not using built-in metrics.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Example:
    --------
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 1.9, 3.2])
    >>> result = omega_squared_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize data if required
    y_true_norm, y_pred_norm = _normalize_data(y_true, y_pred, normalization)

    # Calculate omega squared
    omega_sq = _calculate_omega_squared(y_true_norm, y_pred_norm)

    # Calculate metrics
    metrics = _calculate_metrics(y_true_norm, y_pred_norm, metric, custom_metric)

    # Prepare output
    return {
        'result': omega_sq,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(y_true_norm, y_pred_norm)
    }

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("Inputs must not contain NaN values.")
    if np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError("Inputs must not contain infinite values.")

def _normalize_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalization: str
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize data based on specified method."""
    if normalization == 'none':
        return y_true, y_pred
    elif normalization == 'standard':
        mean = np.mean(y_true)
        std = np.std(y_true)
        if std == 0:
            return y_true - mean, y_pred - mean
        return (y_true - mean) / std, (y_pred - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(y_true)
        max_val = np.max(y_true)
        if min_val == max_val:
            return y_true, y_pred
        return (y_true - min_val) / (max_val - min_val), (y_pred - min_val) / (max_val - min_val)
    elif normalization == 'robust':
        median = np.median(y_true)
        iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        if iqr == 0:
            return y_true - median, y_pred - median
        return (y_true - median) / iqr, (y_pred - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _calculate_omega_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate omega squared effect size."""
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_error = np.sum((y_pred - y_true) ** 2)
    omega_sq = (ss_total - ss_error) / ss_total
    return max(0.0, omega_sq)

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Calculate metrics based on specified method."""
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        metrics['custom'] = metric(y_true, y_pred)
    else:
        raise TypeError("Metric must be a string or callable.")

    if custom_metric is not None:
        metrics['custom_additional'] = custom_metric(y_true, y_pred)

    return metrics

def _check_warnings(y_true: np.ndarray, y_pred: np.ndarray) -> list[str]:
    """Check for potential warnings."""
    warnings = []
    if np.var(y_true) == 0:
        warnings.append("Variance of y_true is zero, omega squared may be unreliable.")
    if np.all(y_pred == y_true):
        warnings.append("y_pred is identical to y_true, omega squared is zero.")
    return warnings

################################################################################
# hedges_g
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    x1: np.ndarray,
    x2: np.ndarray,
    normalize: str = "standard",
) -> None:
    """Validate input arrays and normalization method."""
    if not isinstance(x1, np.ndarray) or not isinstance(x2, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional")
    if len(x1) != len(x2):
        raise ValueError("Inputs must have the same length")
    if np.any(np.isnan(x1)) or np.any(np.isnan(x2)):
        raise ValueError("Inputs must not contain NaN values")
    if np.any(np.isinf(x1)) or np.any(np.isinf(x2)):
        raise ValueError("Inputs must not contain infinite values")

    if normalize not in ["none", "standard", "robust"]:
        raise ValueError("normalize must be 'none', 'standard', or 'robust'")

def _normalize_data(
    x: np.ndarray,
    method: str = "standard",
) -> np.ndarray:
    """Normalize data using specified method."""
    if method == "none":
        return x
    elif method == "standard":
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        if std == 0:
            return x - mean
        return (x - mean) / std
    elif method == "robust":
        median = np.median(x)
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
        if iqr == 0:
            return x - median
        return (x - median) / iqr

def _compute_pooled_variance(
    x1: np.ndarray,
    x2: np.ndarray,
) -> float:
    """Compute pooled variance of two samples."""
    n1 = len(x1)
    n2 = len(x2)
    var1 = np.var(x1, ddof=1)
    var2 = np.var(x2, ddof=1)
    return ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

def _compute_hedges_g(
    x1: np.ndarray,
    x2: np.ndarray,
) -> float:
    """Compute Hedges' g effect size."""
    mean1 = np.mean(x1)
    mean2 = np.mean(x2)
    pooled_var = _compute_pooled_variance(x1, x2)
    n1 = len(x1)
    n2 = len(x2)

    if pooled_var == 0:
        return 0.0

    g = (mean1 - mean2) / np.sqrt(pooled_var)
    j = 1 - (3.0 / (4 * (n1 + n2) - 9))
    return j * g

def hedges_g_fit(
    x1: np.ndarray,
    x2: np.ndarray,
    normalize: str = "standard",
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute Hedges' g effect size between two samples.

    Parameters:
    -----------
    x1 : np.ndarray
        First sample array.
    x2 : np.ndarray
        Second sample array.
    normalize : str, optional
        Normalization method ('none', 'standard', or 'robust').

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": computed Hedges' g value
        - "metrics": dictionary of additional metrics
        - "params_used": dictionary of parameters used
        - "warnings": dictionary of warnings (empty if none)
    """
    # Validate inputs
    _validate_inputs(x1, x2, normalize)

    # Normalize data if requested
    normalized_x1 = _normalize_data(x1, normalize)
    normalized_x2 = _normalize_data(x2, normalize)

    # Compute Hedges' g
    result = _compute_hedges_g(normalized_x1, normalized_x2)

    # Compute additional metrics
    mean_diff = np.mean(normalized_x1) - np.mean(normalized_x2)
    pooled_var = _compute_pooled_variance(normalized_x1, normalized_x2)
    metrics = {
        "mean_difference": mean_diff,
        "pooled_variance": pooled_var,
    }

    # Record parameters used
    params_used = {
        "normalize": normalize,
    }

    warnings = {}

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings,
    }

# Example usage:
"""
x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([2, 3, 4, 5, 6])
result = hedges_g_fit(x1, x2)
print(result)
"""

################################################################################
# odds_ratio
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(a: np.ndarray, b: np.ndarray) -> None:
    """Validate input arrays for odds ratio calculation."""
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional.")
    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        raise ValueError("Input arrays must not contain NaN or infinite values.")
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("Input arrays must contain non-negative values.")

def compute_odds_ratio(a: np.ndarray, b: np.ndarray,
                      normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> float:
    """Compute the odds ratio between two binary arrays."""
    validate_inputs(a, b)

    # Apply normalizer if provided
    if normalizer is not None:
        a = normalizer(a)
        b = normalizer(b)

    # Calculate contingency table
    a_pos = np.sum(a == 1)
    a_neg = np.sum(a == 0)
    b_pos = np.sum(b == 1)
    b_neg = np.sum(b == 0)

    # Calculate odds ratio
    odds_ratio = (a_pos * b_neg) / (a_neg * b_pos)

    return odds_ratio

def compute_metrics(odds_ratio: float,
                   metric_funcs: Dict[str, Callable[[float], float]]) -> Dict[str, float]:
    """Compute metrics for the odds ratio."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(odds_ratio)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def odds_ratio_fit(a: np.ndarray, b: np.ndarray,
                  normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                  metric_funcs: Dict[str, Callable[[float], float]] = None) -> Dict:
    """Compute odds ratio with configurable options."""
    if metric_funcs is None:
        metric_funcs = {}

    # Compute odds ratio
    result = compute_odds_ratio(a, b, normalizer)

    # Compute metrics if provided
    metrics = compute_metrics(result, metric_funcs) if metric_funcs else {}

    # Prepare output
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
        },
        "warnings": []
    }

    return output

# Example usage:
"""
a = np.array([1, 0, 1, 1, 0])
b = np.array([0, 1, 1, 0, 1])

def standard_normalizer(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / np.std(x)

def log_metric(x: float) -> float:
    return np.log(x)

output = odds_ratio_fit(a, b,
                       normalizer=standard_normalizer,
                       metric_funcs={"log": log_metric})
"""

################################################################################
# relative_risk
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_inputs(
    exposed: np.ndarray,
    unexposed: np.ndarray,
    normalize: str = "none",
) -> None:
    """Validate input arrays and perform basic checks."""
    if exposed.ndim != 1 or unexposed.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if len(exposed) != len(unexposed):
        raise ValueError("Inputs must have the same length.")
    if np.any(np.isnan(exposed)) or np.any(np.isnan(unexposed)):
        raise ValueError("Inputs must not contain NaN values.")
    if np.any(exposed < 0) or np.any(unexposed < 0):
        raise ValueError("Inputs must be non-negative.")

def normalize_data(
    data: np.ndarray,
    method: str = "none",
) -> np.ndarray:
    """Normalize data using specified method."""
    if method == "none":
        return data
    elif method == "standard":
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_relative_risk(
    exposed: np.ndarray,
    unexposed: np.ndarray,
) -> float:
    """Compute the relative risk (risk ratio)."""
    exposed_risk = np.sum(exposed) / len(exposed)
    unexposed_risk = np.sum(unexposed) / len(unexposed)
    return exposed_risk / (unexposed_risk + 1e-8)

def compute_confidence_interval(
    exposed: np.ndarray,
    unexposed: np.ndarray,
    alpha: float = 0.05,
) -> tuple:
    """Compute the confidence interval for the relative risk."""
    exposed_risk = np.sum(exposed) / len(exposed)
    unexposed_risk = np.sum(unexposed) / len(unexposed)

    # Using Wald method for confidence interval
    se = np.sqrt(
        (exposed_risk * (1 - exposed_risk) / len(exposed)) +
        (unexposed_risk * (1 - unexposed_risk) / len(unexposed))
    )
    z = 1.96  # For 95% confidence interval
    lower = exposed_risk / unexposed_risk * np.exp(-z * se)
    upper = exposed_risk / unexposed_risk * np.exp(z * se)

    return (lower, upper)

def relative_risk_fit(
    exposed: np.ndarray,
    unexposed: np.ndarray,
    normalize: str = "none",
    alpha: float = 0.05,
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Compute the relative risk (risk ratio) between exposed and unexposed groups.

    Parameters:
    -----------
    exposed : np.ndarray
        Array of outcomes for the exposed group.
    unexposed : np.ndarray
        Array of outcomes for the unexposed group.
    normalize : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    alpha : float, optional
        Significance level for confidence interval.

    Returns:
    --------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(exposed, unexposed)

    # Normalize data if specified
    exposed_norm = normalize_data(exposed, normalize)
    unexposed_norm = normalize_data(unexposed, normalize)

    # Compute relative risk
    rr = compute_relative_risk(exposed_norm, unexposed_norm)

    # Compute confidence interval
    ci = compute_confidence_interval(exposed, unexposed, alpha)

    # Prepare output
    result = {
        "result": rr,
        "confidence_interval": {"lower": ci[0], "upper": ci[1]},
        "metrics": {},
        "params_used": {
            "normalize": normalize,
            "alpha": alpha,
        },
        "warnings": [],
    }

    return result

# Example usage:
# exposed = np.array([1, 0, 1, 1, 0])
# unexposed = np.array([0, 0, 0, 1, 0])
# result = relative_risk_fit(exposed, unexposed)

################################################################################
# standardized_mean_difference
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def standardized_mean_difference_fit(
    group1: np.ndarray,
    group2: np.ndarray,
    *,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'absolute',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Calculate the standardized mean difference between two groups.

    Parameters:
    -----------
    group1 : np.ndarray
        Array of values for the first group.
    group2 : np.ndarray
        Array of values for the second group.
    normalization : str, optional (default='standard')
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : str or callable, optional (default='absolute')
        Metric to use: 'absolute', 'squared', or custom callable.
    distance : str, optional (default='euclidean')
        Distance metric: 'euclidean', 'manhattan', or custom callable.
    solver : str, optional (default='closed_form')
        Solver method: 'closed_form'.
    tol : float, optional (default=1e-6)
        Tolerance for convergence.
    max_iter : int, optional (default=1000)
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function.
    custom_distance : callable, optional
        Custom distance function.

    Returns:
    --------
    Dict containing:
        - result: float, the standardized mean difference.
        - metrics: dict of computed metrics.
        - params_used: dict of parameters used.
        - warnings: list of warnings.

    Example:
    --------
    >>> group1 = np.array([1, 2, 3])
    >>> group2 = np.array([4, 5, 6])
    >>> result = standardized_mean_difference_fit(group1, group2)
    """
    # Validate inputs
    _validate_inputs(group1, group2)

    # Normalize data
    norm_group1, norm_group2 = _normalize_data(group1, group2, normalization=normalization)

    # Calculate mean difference
    mean_diff = _calculate_mean_difference(norm_group1, norm_group2)

    # Calculate pooled standard deviation
    pooled_std = _calculate_pooled_std(norm_group1, norm_group2)

    # Calculate standardized mean difference
    smd = mean_diff / pooled_std

    # Compute metrics
    metrics = _compute_metrics(norm_group1, norm_group2, mean_diff, pooled_std,
                              metric=metric, distance=distance,
                              custom_metric=custom_metric, custom_distance=custom_distance)

    # Prepare output
    result = {
        'result': smd,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
            'distance': distance,
            'solver': solver
        },
        'warnings': []
    }

    return result

def _validate_inputs(group1: np.ndarray, group2: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(group1, np.ndarray) or not isinstance(group2, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if group1.ndim != 1 or group2.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if len(group1) != len(group2):
        raise ValueError("Input arrays must have the same length.")
    if np.any(np.isnan(group1)) or np.any(np.isnan(group2)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(group1)) or np.any(np.isinf(group2)):
        raise ValueError("Input arrays must not contain infinite values.")

def _normalize_data(
    group1: np.ndarray,
    group2: np.ndarray,
    *,
    normalization: str = 'standard'
) -> tuple:
    """Normalize the input data."""
    if normalization == 'none':
        return group1, group2
    elif normalization == 'standard':
        mean = np.mean(np.concatenate([group1, group2]))
        std = np.std(np.concatenate([group1, group2]), ddof=1)
        norm_group1 = (group1 - mean) / std
        norm_group2 = (group2 - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(np.concatenate([group1, group2]))
        max_val = np.max(np.concatenate([group1, group2]))
        norm_group1 = (group1 - min_val) / (max_val - min_val)
        norm_group2 = (group2 - min_val) / (max_val - min_val)
    elif normalization == 'robust':
        median = np.median(np.concatenate([group1, group2]))
        iqr = np.percentile(np.concatenate([group1, group2]), 75) - np.percentile(np.concatenate([group1, group2]), 25)
        norm_group1 = (group1 - median) / iqr
        norm_group2 = (group2 - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    return norm_group1, norm_group2

def _calculate_mean_difference(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate the mean difference between two groups."""
    return np.mean(group1) - np.mean(group2)

def _calculate_pooled_std(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate the pooled standard deviation."""
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    return np.sqrt(pooled_var)

def _compute_metrics(
    group1: np.ndarray,
    group2: np.ndarray,
    mean_diff: float,
    pooled_std: float,
    *,
    metric: Union[str, Callable] = 'absolute',
    distance: str = 'euclidean',
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """Compute additional metrics."""
    metrics = {}

    if isinstance(metric, str):
        if metric == 'absolute':
            metrics['mean_absolute_difference'] = np.abs(mean_diff)
        elif metric == 'squared':
            metrics['mean_squared_difference'] = mean_diff ** 2
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        metrics['custom_metric'] = metric(group1, group2)

    if isinstance(distance, str):
        if distance == 'euclidean':
            metrics['euclidean_distance'] = np.linalg.norm(group1 - group2)
        elif distance == 'manhattan':
            metrics['manhattan_distance'] = np.sum(np.abs(group1 - group2))
        elif distance == 'cosine':
            metrics['cosine_distance'] = 1 - np.dot(group1, group2) / (np.linalg.norm(group1) * np.linalg.norm(group2))
        elif distance == 'minkowski':
            metrics['minkowski_distance'] = np.sum(np.abs(group1 - group2) ** 3) ** (1/3)
        else:
            raise ValueError(f"Unknown distance: {distance}")
    elif callable(distance):
        metrics['custom_distance'] = distance(group1, group2)

    return metrics

################################################################################
# number_needed_to_treat
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    event_rate_control: np.ndarray,
    event_rate_treatment: np.ndarray,
    alpha: Optional[float] = None
) -> None:
    """Validate inputs for NNT calculation."""
    if not isinstance(event_rate_control, np.ndarray) or not isinstance(event_rate_treatment, np.ndarray):
        raise TypeError("event_rate_control and event_rate_treatment must be numpy arrays")
    if len(event_rate_control.shape) != 1 or len(event_rate_treatment.shape) != 1:
        raise ValueError("Inputs must be 1-dimensional arrays")
    if event_rate_control.shape != event_rate_treatment.shape:
        raise ValueError("Inputs must have the same shape")
    if np.any((event_rate_control < 0) | (event_rate_treatment < 0)):
        raise ValueError("Event rates cannot be negative")
    if np.any((event_rate_control > 1) | (event_rate_treatment > 1)):
        raise ValueError("Event rates cannot exceed 1")
    if alpha is not None and (alpha <= 0 or alpha >= 1):
        raise ValueError("Alpha must be between 0 and 1")

def _calculate_absolute_risk_reduction(
    event_rate_control: np.ndarray,
    event_rate_treatment: np.ndarray
) -> float:
    """Calculate Absolute Risk Reduction (ARR)."""
    return np.mean(event_rate_control) - np.mean(event_rate_treatment)

def _calculate_number_needed_to_treat(
    arr: float,
    alpha: Optional[float] = None
) -> float:
    """Calculate Number Needed to Treat (NNT)."""
    if arr == 0:
        return np.inf
    if alpha is None:
        return 1 / arr
    return 1 / (arr * (1 - alpha))

def number_needed_to_treat_fit(
    event_rate_control: np.ndarray,
    event_rate_treatment: np.ndarray,
    alpha: Optional[float] = None,
    metric_func: Callable[[np.ndarray, np.ndarray], float] = _calculate_absolute_risk_reduction
) -> Dict[str, Union[float, Dict[str, float], Dict[str, Union[str, float]], list]]:
    """
    Calculate Number Needed to Treat (NNT) with configurable options.

    Parameters
    ----------
    event_rate_control : np.ndarray
        Array of event rates for control group.
    event_rate_treatment : np.ndarray
        Array of event rates for treatment group.
    alpha : Optional[float], default=None
        Alpha level for confidence interval adjustment. If None, no adjustment is made.
    metric_func : Callable[[np.ndarray, np.ndarray], float], default=_calculate_absolute_risk_reduction
        Function to calculate the metric used for NNT calculation.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, Union[str, float]], list]]
        Dictionary containing:
        - "result": The calculated NNT value.
        - "metrics": Dictionary of calculated metrics.
        - "params_used": Dictionary of parameters used in the calculation.
        - "warnings": List of any warnings generated during calculation.

    Examples
    --------
    >>> event_rate_control = np.array([0.1, 0.2, 0.3])
    >>> event_rate_treatment = np.array([0.05, 0.15, 0.2])
    >>> result = number_needed_to_treat_fit(event_rate_control, event_rate_treatment)
    """
    warnings = []

    _validate_inputs(event_rate_control, event_rate_treatment, alpha)

    arr = metric_func(event_rate_control, event_rate_treatment)
    nnt = _calculate_number_needed_to_treat(arr, alpha)

    metrics = {
        "absolute_risk_reduction": arr,
        "number_needed_to_treat": nnt
    }

    params_used = {
        "alpha": alpha,
        "metric_function": metric_func.__name__ if hasattr(metric_func, '__name__') else "custom"
    }

    return {
        "result": nnt,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# phi_coefficient
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("Inputs must have the same shape.")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input x contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input y contains NaN or infinite values.")

def _compute_contingency_table(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the contingency table for binary variables."""
    unique_x = np.unique(x)
    unique_y = np.unique(y)

    if len(unique_x) != 2 or len(unique_y) != 2:
        raise ValueError("Inputs must be binary variables.")

    contingency_table = np.zeros((2, 2), dtype=int)
    for i in range(2):
        for j in range(2):
            contingency_table[i, j] = np.sum((x == unique_x[i]) & (y == unique_y[j]))

    return contingency_table

def _compute_phi_coefficient(contingency_table: np.ndarray) -> float:
    """Compute the phi coefficient from a contingency table."""
    n = np.sum(contingency_table)
    chi_square = ((contingency_table * (n - contingency_table)) / n).sum()
    phi = np.sqrt(chi_square) / n
    return phi

def phi_coefficient_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    normalize: Optional[str] = None,
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Compute the phi coefficient between two binary variables.

    Parameters
    ----------
    x : np.ndarray
        First binary variable.
    y : np.ndarray
        Second binary variable.
    normalize : str, optional
        Normalization method (not used for phi coefficient but kept for consistency).
    metric : Callable, optional
        Custom metric function (not used for phi coefficient but kept for consistency).

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]
        Dictionary containing the result, metrics, parameters used, and warnings.

    Examples
    --------
    >>> x = np.array([0, 1, 0, 1])
    >>> y = np.array([0, 0, 1, 1])
    >>> result = phi_coefficient_fit(x, y)
    """
    _validate_inputs(x, y)
    contingency_table = _compute_contingency_table(x, y)
    phi = _compute_phi_coefficient(contingency_table)

    return {
        "result": phi,
        "metrics": {},
        "params_used": {"normalize": normalize, "metric": metric.__name__ if metric else None},
        "warnings": []
    }

################################################################################
# cramers_v
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Cramér's V calculation."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays must have the same length.")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Input arrays must not contain NaN or infinite values.")

def _compute_contingency_table(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the contingency table from two categorical arrays."""
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    contingency_table = np.zeros((len(x_unique), len(y_unique)), dtype=int)

    for i, val_x in enumerate(x_unique):
        for j, val_y in enumerate(y_unique):
            contingency_table[i, j] = np.sum((x == val_x) & (y == val_y))

    return contingency_table

def _compute_chi_square(contingency_table: np.ndarray) -> float:
    """Compute the chi-square statistic from a contingency table."""
    n = np.sum(contingency_table)
    row_sums = np.sum(contingency_table, axis=1)
    col_sums = np.sum(contingency_table, axis=0)

    expected = (row_sums[:, np.newaxis] * col_sums[np.newaxis, :]) / n
    chi_square = np.sum((contingency_table - expected) ** 2 / expected)
    return chi_square

def cramers_v_fit(
    x: np.ndarray,
    y: np.ndarray,
    correction: bool = True,
    normalization: str = "standard",
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Compute Cramér's V association measure between two categorical variables.

    Parameters
    ----------
    x : np.ndarray
        First categorical variable.
    y : np.ndarray
        Second categorical variable.
    correction : bool, optional
        Whether to apply continuity correction (default is True).
    normalization : str, optional
        Normalization method ("standard" or "none", default is "standard").

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]
        Dictionary containing:
        - "result": Cramér's V value
        - "metrics": Dictionary of additional metrics (chi-square, degrees of freedom)
        - "params_used": Dictionary of parameters used
        - "warnings": List of warnings

    Examples
    --------
    >>> x = np.array([1, 2, 1, 3, 2])
    >>> y = np.array([4, 4, 5, 6, 6])
    >>> result = cramers_v_fit(x, y)
    """
    _validate_inputs(x, y)

    contingency_table = _compute_contingency_table(x, y)
    chi_square = _compute_chi_square(contingency_table)

    n = contingency_table.shape[0]
    m = contingency_table.shape[1]

    if correction:
        chi_square -= (n - 1) * (m - 1) / (np.sqrt(n * m))

    min_dim = min(n, m) - 1
    cramers_v = np.sqrt(chi_square / (n * m * min_dim))

    if normalization == "standard":
        pass  # Already normalized
    elif normalization == "none":
        cramers_v = np.sqrt(chi_square / (n * m))
    else:
        raise ValueError("Invalid normalization method.")

    metrics = {
        "chi_square": chi_square,
        "degrees_of_freedom": (n - 1) * (m - 1),
    }

    params_used = {
        "correction": correction,
        "normalization": normalization,
    }

    warnings = []

    return {
        "result": cramers_v,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings,
    }
