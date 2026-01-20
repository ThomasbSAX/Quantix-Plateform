"""
Quantix – Module skewness_kurtosis
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# skewness
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for skewness calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def _calculate_skewness(data: np.ndarray, bias: bool = False) -> float:
    """Calculate skewness of the data."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=0 if bias else 1)

    if std == 0:
        raise ValueError("Standard deviation is zero, cannot calculate skewness")

    skewness = (np.sum((data - mean) ** 3) / n) / (std ** 3)
    if not bias:
        skewness *= np.sqrt(n * (n - 1)) / (n - 2)
    return skewness

def _calculate_metrics(data: np.ndarray, skewness: float) -> Dict[str, float]:
    """Calculate metrics related to skewness."""
    return {
        "skewness": skewness,
        "absolute_skewness": abs(skewness),
    }

def skewness_fit(
    data: np.ndarray,
    bias: bool = False,
    normalize: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, float], Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Calculate skewness of the data with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data for which skewness is calculated.
    bias : bool, optional
        If True, calculates biased skewness. Default is False.
    normalize : str or None, optional
        Normalization method to apply before calculation. Options: 'standard', 'minmax', None.
    custom_metric : callable or None, optional
        Custom metric function to calculate additional metrics.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.

    Example:
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> skewness_fit(data)
    {
        'result': {'skewness': 0.0},
        'metrics': {'skewness': 0.0, 'absolute_skewness': 0.0},
        'params_used': {'bias': False, 'normalize': None},
        'warnings': []
    }
    """
    _validate_input(data)

    if normalize == "standard":
        data = (data - np.mean(data)) / np.std(data)
    elif normalize == "minmax":
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

    skewness = _calculate_skewness(data, bias)
    metrics = _calculate_metrics(data, skewness)

    if custom_metric is not None:
        metrics.update(custom_metric(data, skewness))

    return {
        "result": {"skewness": skewness},
        "metrics": metrics,
        "params_used": {
            "bias": bias,
            "normalize": normalize,
        },
        "warnings": [],
    }

################################################################################
# kurtosis
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_input(data: np.ndarray) -> None:
    """Validate input data for kurtosis calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values.")

def compute_kurtosis(data: np.ndarray, fisher: bool = True, bias: bool = False) -> float:
    """Compute the kurtosis of a dataset."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1 if bias else 0)

    if std == 0:
        raise ValueError("Standard deviation is zero, cannot compute kurtosis.")

    diff = data - mean
    kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(diff**4) / (std**4) - 3 * ((n - 1)**2 / ((n - 2) * (n - 3)))

    if fisher:
        kurtosis -= 3

    return kurtosis

def kurtosis_fit(
    data: np.ndarray,
    fisher: bool = True,
    bias: bool = False,
    normalization: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the kurtosis of a dataset with various options.

    Parameters:
    -----------
    data : np.ndarray
        Input data.
    fisher : bool, optional
        Whether to use Fisher's correction (default is True).
    bias : bool, optional
        Whether to use a biased estimator (default is False).
    normalization : str, optional
        Type of normalization to apply (none, standard, minmax, robust).
    custom_metric : callable, optional
        Custom metric function to compute additional metrics.

    Returns:
    --------
    dict
        A dictionary containing the result, metrics, parameters used, and warnings.
    """
    validate_input(data)

    if normalization == "standard":
        data = (data - np.mean(data)) / np.std(data)
    elif normalization == "minmax":
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
    elif normalization == "robust":
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        data = (data - median) / iqr

    kurtosis_value = compute_kurtosis(data, fisher=fisher, bias=bias)

    metrics = {}
    if custom_metric is not None:
        metrics["custom"] = custom_metric(data)

    result = {
        "result": kurtosis_value,
        "metrics": metrics,
        "params_used": {
            "fisher": fisher,
            "bias": bias,
            "normalization": normalization
        },
        "warnings": []
    }

    return result

# Example usage:
# data = np.random.normal(0, 1, 1000)
# result = kurtosis_fit(data, fisher=True, normalization="standard")

################################################################################
# excess_kurtosis
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for excess kurtosis calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def _compute_moments(data: np.ndarray, order: int) -> float:
    """Compute the nth central moment of the data."""
    mean = np.mean(data)
    return np.mean((data - mean) ** order)

def _excess_kurtosis(data: np.ndarray, bias_corrected: bool = True) -> float:
    """Calculate the excess kurtosis of a dataset."""
    n = len(data)
    m4 = _compute_moments(data, 4)
    m2_squared = _compute_moments(data, 2) ** 2

    kurtosis = m4 / m2_squared - 3
    if bias_corrected:
        kurtosis *= (n * (n + 1)) / ((n - 1) * (n - 2)) - (3 * (n - 1)) / (n - 2)
    return kurtosis

def excess_kurtosis_fit(
    data: np.ndarray,
    bias_corrected: bool = True,
    normalization: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Calculate the excess kurtosis of a dataset with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    bias_corrected : bool, optional
        Whether to apply bias correction (default: True).
    normalization : str or None, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    custom_metric : callable or None, optional
        Custom metric function that takes data as input.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Example:
    --------
    >>> data = np.random.normal(0, 1, 1000)
    >>> result = excess_kurtosis_fit(data)
    """
    _validate_input(data)

    # Normalize data if specified
    normalized_data = data.copy()
    if normalization == 'standard':
        normalized_data = (data - np.mean(data)) / np.std(data)
    elif normalization == 'minmax':
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    elif normalization == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        normalized_data = (data - median) / iqr

    # Calculate excess kurtosis
    ek = _excess_kurtosis(normalized_data, bias_corrected)

    # Calculate custom metric if provided
    metrics = {}
    if custom_metric is not None:
        try:
            metrics['custom_metric'] = custom_metric(normalized_data)
        except Exception as e:
            metrics['custom_metric_error'] = str(e)

    # Prepare output
    result = {
        'result': ek,
        'metrics': metrics,
        'params_used': {
            'bias_corrected': bias_corrected,
            'normalization': normalization
        },
        'warnings': []
    }

    return result

################################################################################
# positive_skewness
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for positive skewness calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _calculate_skewness(data: np.ndarray, bias: bool = False) -> float:
    """Calculate skewness of the data."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1 if not bias else 0)

    if std == 0:
        return 0.0

    skewness = (np.sum((data - mean) ** 3) / n) / (std ** 3)
    return skewness

def _calculate_kurtosis(data: np.ndarray, bias: bool = False) -> float:
    """Calculate kurtosis of the data."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1 if not bias else 0)

    if std == 0:
        return 0.0

    kurtosis = (np.sum((data - mean) ** 4) / n) / (std ** 4)
    return kurtosis

def _normalize_data(data: np.ndarray, method: str = "standard") -> np.ndarray:
    """Normalize the data using specified method."""
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

def positive_skewness_fit(
    data: np.ndarray,
    normalization: str = "standard",
    bias_correction: bool = False,
    custom_metric: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Calculate positive skewness and related statistics.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    bias_correction : bool, optional
        Whether to apply bias correction.
    custom_metric : Callable[[np.ndarray], float], optional
        Custom metric function.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> result = positive_skewness_fit(data)
    """
    _validate_input(data)

    normalized_data = _normalize_data(data, normalization)
    skewness = _calculate_skewness(normalized_data, bias_correction)
    kurtosis = _calculate_kurtosis(normalized_data, bias_correction)

    metrics = {}
    if custom_metric is not None:
        metrics["custom"] = custom_metric(normalized_data)

    warnings = []
    if skewness < 0:
        warnings.append("Negative skewness detected")

    return {
        "result": {
            "skewness": skewness,
            "kurtosis": kurtosis
        },
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "bias_correction": bias_correction
        },
        "warnings": warnings
    }

################################################################################
# negative_skewness
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for negative skewness computation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values.")

def _compute_negative_skewness(data: np.ndarray, normalization: str = 'standard') -> float:
    """Compute negative skewness with optional normalization."""
    mean = np.mean(data)
    std = np.std(data)

    if normalization == 'standard':
        normalized_data = (data - mean) / std
    elif normalization == 'none':
        normalized_data = data
    else:
        raise ValueError("Unsupported normalization method.")

    skewness = np.mean(normalized_data**3)
    return -skewness  # Negative skewness

def _compute_metrics(data: np.ndarray, negative_skewness: float,
                     metric_funcs: Optional[Dict[str, Callable]] = None) -> Dict[str, float]:
    """Compute additional metrics for negative skewness."""
    metrics = {}
    if metric_funcs is None:
        return metrics

    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(data, negative_skewness)
        except Exception as e:
            metrics[f"{name}_error"] = str(e)

    return metrics

def negative_skewness_fit(data: np.ndarray,
                         normalization: str = 'standard',
                         metric_funcs: Optional[Dict[str, Callable]] = None) -> Dict:
    """Compute negative skewness with configurable options.

    Parameters
    ----------
    data : np.ndarray
        Input data for skewness computation.
    normalization : str, optional
        Normalization method ('standard' or 'none'), by default 'standard'.
    metric_funcs : Optional[Dict[str, Callable]], optional
        Dictionary of custom metric functions, by default None.

    Returns
    -------
    Dict
        Structured result containing negative skewness, metrics, and parameters used.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> result = negative_skewness_fit(data)
    """
    _validate_input(data)

    negative_skew = _compute_negative_skewness(data, normalization)
    metrics = _compute_metrics(data, negative_skew, metric_funcs)

    return {
        "result": {"negative_skewness": negative_skew},
        "metrics": metrics,
        "params_used": {
            "normalization": normalization
        },
        "warnings": []
    }

################################################################################
# mesokurtic
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for mesokurtic analysis."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def _compute_skewness(data: np.ndarray) -> float:
    """Compute the skewness of the data."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)
    skewness = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
    return skewness

def _compute_kurtosis(data: np.ndarray) -> float:
    """Compute the kurtosis of the data."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)
    kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((data - mean) / std) ** 4) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return kurtosis

def _normalize_data(data: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize the data using the specified method."""
    if method == 'none':
        return data
    elif method == 'standard':
        mean = np.mean(data)
        std = np.std(data, ddof=1)
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

def _compute_metrics(skewness: float, kurtosis: float, metric_funcs: Optional[Dict[str, Callable]] = None) -> Dict[str, float]:
    """Compute the specified metrics for skewness and kurtosis."""
    metrics = {}
    if metric_funcs is None:
        return metrics

    for name, func in metric_funcs.items():
        try:
            if name == 'skewness':
                metrics[name] = func(skewness)
            elif name == 'kurtosis':
                metrics[name] = func(kurtosis)
            else:
                raise ValueError(f"Unknown metric: {name}")
        except Exception as e:
            metrics[name] = np.nan
            print(f"Warning: Error computing metric {name}: {e}")

    return metrics

def mesokurtic_fit(
    data: np.ndarray,
    normalization_method: str = 'standard',
    metric_funcs: Optional[Dict[str, Callable]] = None,
    **kwargs
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute mesokurtic statistics and metrics for the given data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    normalization_method : str, optional
        Normalization method to apply ('none', 'standard', 'minmax', 'robust').
    metric_funcs : Dict[str, Callable], optional
        Dictionary of custom metric functions to compute.
    **kwargs : dict
        Additional keyword arguments for future extensions.

    Returns:
    --------
    result : Dict[str, Union[float, Dict[str, float], Dict[str, str]]]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    _validate_inputs(data)

    normalized_data = _normalize_data(data, normalization_method)
    skewness = _compute_skewness(normalized_data)
    kurtosis = _compute_kurtosis(normalized_data)

    metrics = _compute_metrics(skewness, kurtosis, metric_funcs)

    result = {
        "result": {
            "skewness": skewness,
            "kurtosis": kurtosis
        },
        "metrics": metrics,
        "params_used": {
            "normalization_method": normalization_method
        },
        "warnings": []
    }

    return result

# Example usage:
# data = np.random.normal(0, 1, 1000)
# result = mesokurtic_fit(data, normalization_method='standard')

################################################################################
# leptokurtic
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional.")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data must not contain NaN or infinite values.")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using specified method."""
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
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_leptokurtic_statistic(data: np.ndarray) -> float:
    """Compute the leptokurtic statistic."""
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    return (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * np.sum((data - mean) ** 4) / std ** 4 - 3

def _compute_metrics(data: np.ndarray, statistic: float, metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute metrics based on provided functions."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(data, statistic)
        except Exception as e:
            metrics[f"{name}_error"] = str(e)
    return metrics

def leptokurtic_fit(
    data: np.ndarray,
    normalization: str = "standard",
    metric_funcs: Optional[Dict[str, Callable]] = None,
    **kwargs
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute leptokurtic statistic and metrics for the given data.

    Parameters:
    - data: Input data as a numpy array.
    - normalization: Normalization method ("none", "standard", "minmax", "robust").
    - metric_funcs: Dictionary of metric functions to compute.

    Returns:
    - A dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(data)
    normalized_data = _normalize_data(data, normalization)
    statistic = _compute_leptokurtic_statistic(normalized_data)

    default_metrics = {
        "leptokurtic": lambda d, s: s,
    }
    if metric_funcs is None:
        metric_funcs = default_metrics
    else:
        metric_funcs.update(default_metrics)

    metrics = _compute_metrics(normalized_data, statistic, metric_funcs)

    return {
        "result": statistic,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric_functions": list(metric_funcs.keys()),
        },
        "warnings": [],
    }

# Example usage:
# data = np.random.normal(0, 1, 1000)
# result = leptokurtic_fit(data, normalization="standard")

################################################################################
# platykurtic
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for platykurtic analysis."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values.")

def _compute_kurtosis(data: np.ndarray, fisher: bool = True) -> float:
    """Compute kurtosis of the data."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1) if n > 1 else 0.0
    if std == 0:
        return 0.0

    diff = data - mean
    kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(diff**4) / (std**4) - 3 * ((n - 1)**2 / ((n - 2) * (n - 3)))
    if fisher:
        kurtosis -= 3
    return kurtosis

def _normalize_data(data: np.ndarray, method: str = "standard") -> np.ndarray:
    """Normalize data using specified method."""
    if method == "none":
        return data
    elif method == "standard":
        mean = np.mean(data)
        std = np.std(data, ddof=1) if len(data) > 1 else 0.0
        return (data - mean) / std if std != 0 else data
    elif method == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val) if max_val != min_val else data
    elif method == "robust":
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        return (data - q1) / iqr if iqr != 0 else data
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_metrics(data: np.ndarray, kurtosis: float, metric_funcs: Optional[Dict[str, Callable]] = None) -> Dict[str, float]:
    """Compute metrics based on kurtosis."""
    metrics = {}
    if metric_funcs is None:
        metric_funcs = {}

    # Default metrics
    default_metrics = {
        "kurtosis": lambda: kurtosis,
        "excess_kurtosis": lambda: kurtosis - 3 if not np.isnan(kurtosis) else np.nan,
        "is_platykurtic": lambda: kurtosis < 0 if not np.isnan(kurtosis) else False
    }

    for name, func in {**default_metrics, **metric_funcs}.items():
        try:
            metrics[name] = func()
        except Exception as e:
            metrics[f"error_{name}"] = str(e)

    return metrics

def platykurtic_fit(
    data: np.ndarray,
    normalization: str = "standard",
    metric_funcs: Optional[Dict[str, Callable]] = None,
    fisher: bool = True
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute platykurtic analysis on the given data.

    Parameters:
    -----------
    data : np.ndarray
        Input data for analysis.
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    metric_funcs : Dict[str, Callable], optional
        Custom metrics to compute.
    fisher : bool, optional
        Whether to use Fisher's definition of kurtosis.

    Returns:
    --------
    Dict containing:
        - "result": float, the computed kurtosis.
        - "metrics": Dict[str, float], computed metrics.
        - "params_used": Dict[str, str], parameters used.
        - "warnings": Dict[str, str], any warnings.

    Example:
    --------
    >>> data = np.random.normal(0, 1, 1000)
    >>> result = platykurtic_fit(data)
    """
    _validate_input(data)

    normalized_data = _normalize_data(data, normalization)
    kurtosis = _compute_kurtosis(normalized_data, fisher)
    metrics = _compute_metrics(data, kurtosis, metric_funcs)

    return {
        "result": kurtosis,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "fisher": str(fisher)
        },
        "warnings": {}
    }

################################################################################
# sample_skewness
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for skewness calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def _calculate_moments(data: np.ndarray, order: int) -> float:
    """Calculate the nth moment of the data."""
    mean = np.mean(data)
    return np.sum((data - mean) ** order) / len(data)

def _standardize_data(data: np.ndarray, normalization: str = 'none') -> np.ndarray:
    """Standardize data based on the chosen normalization method."""
    if normalization == 'none':
        return data
    elif normalization == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            raise ValueError("Standard deviation is zero, cannot standardize")
        return (data - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        if min_val == max_val:
            raise ValueError("Min and max values are equal, cannot normalize")
        return (data - min_val) / (max_val - min_val)
    elif normalization == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        if iqr == 0:
            raise ValueError("IQR is zero, cannot robust standardize")
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def sample_skewness_fit(
    data: np.ndarray,
    normalization: str = 'none',
    bias_correction: bool = True
) -> Dict[str, Any]:
    """
    Calculate the sample skewness of a dataset.

    Parameters:
    -----------
    data : np.ndarray
        Input data for which to calculate skewness.
    normalization : str, optional
        Normalization method to apply before calculation ('none', 'standard', 'minmax', 'robust').
    bias_correction : bool, optional
        Whether to apply bias correction to the skewness estimate.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the skewness result, metrics, parameters used, and warnings.

    Example:
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> result = sample_skewness_fit(data)
    """
    _validate_input(data)

    # Standardize data if required
    standardized_data = _standardize_data(data, normalization)

    # Calculate moments
    mean = np.mean(standardized_data)
    variance = _calculate_moments(standardized_data, 2)
    skewness = _calculate_moments(standardized_data, 3) / (variance ** (3/2))

    # Apply bias correction if required
    n = len(data)
    if bias_correction and n > 3:
        skewness *= np.sqrt((n * (n - 1)) ** 0.5) / (n - 2)

    # Prepare output
    result = {
        "result": skewness,
        "metrics": {
            "mean": mean,
            "variance": variance
        },
        "params_used": {
            "normalization": normalization,
            "bias_correction": bias_correction
        },
        "warnings": []
    }

    return result

################################################################################
# sample_kurtosis
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for kurtosis calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def _compute_moments(data: np.ndarray, n: int) -> float:
    """Compute the nth central moment of the data."""
    mean = np.mean(data)
    return np.mean((data - mean) ** n)

def _sample_kurtosis(
    data: np.ndarray,
    bias_corrected: bool = True,
    fisher_normalized: bool = False
) -> float:
    """Compute sample kurtosis with optional bias correction and Fisher normalization."""
    n = len(data)
    m4 = _compute_moments(data, 4)
    m2 = _compute_moments(data, 2)

    kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * m4 / (m2 ** 2) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

    if bias_corrected:
        kurtosis -= 3
        kurtosis *= (n - 1) * (n + 1) / ((n - 2) * (n - 3))
        kurtosis += 3

    if fisher_normalized:
        kurtosis = (kurtosis - 3) / np.sqrt(24 / n)

    return kurtosis

def sample_kurtosis_fit(
    data: np.ndarray,
    bias_corrected: bool = True,
    fisher_normalized: bool = False
) -> Dict[str, Any]:
    """
    Compute sample kurtosis with various normalization options.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    bias_corrected : bool, optional
        Whether to apply bias correction (default True).
    fisher_normalized : bool, optional
        Whether to apply Fisher normalization (default False).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_input(data)

    result = {
        "result": _sample_kurtosis(data, bias_corrected, fisher_normalized),
        "metrics": {},
        "params_used": {
            "bias_corrected": bias_corrected,
            "fisher_normalized": fisher_normalized
        },
        "warnings": []
    }

    return result

# Example usage:
# data = np.random.normal(0, 1, 1000)
# result = sample_kurtosis_fit(data, bias_corrected=True, fisher_normalized=False)

################################################################################
# population_skewness
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for skewness calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def _calculate_moments(data: np.ndarray, normalized: bool = True) -> Dict[str, float]:
    """Calculate moments needed for skewness calculation."""
    mean = np.mean(data)
    std = np.std(data) if normalized else 1.0
    n = len(data)

    m2 = np.sum((data - mean) ** 2) / n
    m3 = np.sum((data - mean) ** 3) / n

    return {
        'mean': mean,
        'std': std,
        'm2': m2,
        'm3': m3
    }

def population_skewness_fit(
    data: np.ndarray,
    *,
    normalization: str = 'standard',
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'standard',
    custom_metric: Optional[Callable[[float, float], float]] = None
) -> Dict[str, Any]:
    """
    Calculate population skewness with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    normalization : str, optional
        Normalization method ('none', 'standard', 'robust').
    custom_normalization : callable, optional
        Custom normalization function.
    metric : str, optional
        Metric type ('standard', 'biased').
    custom_metric : callable, optional
        Custom metric function.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings.
    """
    _validate_input(data)

    # Handle normalization
    if custom_normalization is not None:
        normalized_data = custom_normalization(data)
    elif normalization == 'standard':
        normalized_data = (data - np.mean(data)) / np.std(data)
    elif normalization == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        normalized_data = (data - median) / iqr
    else:
        normalized_data = data.copy()

    # Calculate moments
    moments = _calculate_moments(normalized_data)

    # Calculate skewness based on metric choice
    if custom_metric is not None:
        skewness = custom_metric(moments['m3'], moments['std'])
    elif metric == 'standard':
        skewness = moments['m3'] / (moments['std'] ** 3)
    else:  # biased
        n = len(data)
        skewness = (moments['m3'] * n) / ((n - 1) * (n - 2)) * (moments['std'] ** 3)

    # Prepare output
    result = {
        'result': skewness,
        'metrics': {
            'mean': moments['mean'],
            'std_dev': moments['std'],
            'third_moment': moments['m3']
        },
        'params_used': {
            'normalization': normalization if custom_normalization is None else 'custom',
            'metric': metric if custom_metric is None else 'custom'
        },
        'warnings': []
    }

    return result

# Example usage:
# skewness_result = population_skewness_fit(
#     np.array([1, 2, 3, 4, 5]),
#     normalization='standard',
#     metric='standard'
# )

################################################################################
# population_kurtosis
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
    """Compute the nth central moment of the data."""
    mean = np.mean(data)
    return np.mean((data - mean) ** order)

def _population_kurtosis_compute(
    data: np.ndarray,
    bias_corrected: bool = True,
    fisher_normalized: bool = False
) -> float:
    """Compute the population kurtosis of a dataset.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    bias_corrected : bool, optional
        Whether to apply bias correction (default True).
    fisher_normalized : bool, optional
        Whether to use Fisher normalization (default False).

    Returns
    -------
    float
        The computed kurtosis value.
    """
    _validate_input(data)

    n = len(data)
    m4 = _compute_moments(data, 4)
    m2 = _compute_moments(data, 2)

    kurtosis = (m4 / (m2 ** 2)) - 3

    if bias_corrected:
        kurtosis = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * kurtosis
        kurtosis -= ((3 * (n - 1) ** 2) / ((n - 2) * (n - 3))) * kurtosis

    if fisher_normalized:
        kurtosis = kurtosis / np.sqrt(24 / n)

    return kurtosis

def population_kurtosis_fit(
    data: np.ndarray,
    bias_corrected: bool = True,
    fisher_normalized: bool = False
) -> Dict[str, Union[float, Dict]]:
    """Compute the population kurtosis with additional metrics and parameters.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    bias_corrected : bool, optional
        Whether to apply bias correction (default True).
    fisher_normalized : bool, optional
        Whether to use Fisher normalization (default False).

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing:
        - "result": the computed kurtosis value
        - "metrics": additional metrics (currently empty)
        - "params_used": parameters used in the computation
        - "warnings": any warnings generated during computation
    """
    _validate_input(data)

    result = _population_kurtosis_compute(
        data,
        bias_corrected=bias_corrected,
        fisher_normalized=fisher_normalized
    )

    return {
        "result": result,
        "metrics": {},
        "params_used": {
            "bias_corrected": bias_corrected,
            "fisher_normalized": fisher_normalized
        },
        "warnings": []
    }

# Example usage:
# data = np.random.normal(0, 1, 1000)
# result = population_kurtosis_fit(data)
