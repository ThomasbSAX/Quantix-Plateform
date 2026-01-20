"""
Quantix – Module quantiles_percentiles
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# quartile
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for quartile calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _calculate_quartile(
    data: np.ndarray,
    method: str = 'linear',
    interpolation: Optional[Callable[[float, float, float], float]] = None
) -> np.ndarray:
    """
    Calculate quartiles using specified method.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    method : str, optional
        Interpolation method ('linear', 'nearest', 'lower', 'higher').
    interpolation : callable, optional
        Custom interpolation function.

    Returns:
    --------
    np.ndarray
        Array of quartile values.
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)

    if method == 'linear':
        def _interpolate(i: float) -> float:
            lower = int(np.floor(i))
            upper = int(np.ceil(i))
            if lower == upper:
                return sorted_data[lower]
            frac = i - lower
            return (1 - frac) * sorted_data[lower] + frac * sorted_data[upper]
        interpolate = _interpolate
    elif method == 'nearest':
        def _interpolate(i: float) -> float:
            return sorted_data[int(round(i))]
        interpolate = _interpolate
    elif method == 'lower':
        def _interpolate(i: float) -> float:
            return sorted_data[int(np.floor(i))]
        interpolate = _interpolate
    elif method == 'higher':
        def _interpolate(i: float) -> float:
            return sorted_data[int(np.ceil(i))]
        interpolate = _interpolate
    elif interpolation is not None:
        interpolate = interpolation
    else:
        raise ValueError("Invalid method or missing custom interpolation function")

    positions = np.array([0.25, 0.5, 0.75]) * (n - 1)
    return np.array([interpolate(pos) for pos in positions])

def quartile_fit(
    data: np.ndarray,
    method: str = 'linear',
    interpolation: Optional[Callable[[float, float, float], float]] = None,
    normalize: bool = False
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Calculate quartiles with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    method : str, optional
        Interpolation method ('linear', 'nearest', 'lower', 'higher').
    interpolation : callable, optional
        Custom interpolation function.
    normalize : bool, optional
        Whether to normalize data before calculation.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_input(data)

    if normalize:
        data = (data - np.mean(data)) / np.std(data)
        warnings = "Data was normalized before quartile calculation"
    else:
        warnings = None

    quartiles = _calculate_quartile(data, method, interpolation)

    metrics = {
        'q1': quartiles[0],
        'median': quartiles[1],
        'q3': quartiles[2]
    }

    params_used = {
        'method': method,
        'interpolation': interpolation.__name__ if interpolation else None,
        'normalize': normalize
    }

    return {
        'result': quartiles,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

# Example usage:
# quartile_fit(np.random.randn(100), method='linear', normalize=True)

################################################################################
# decile
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for decile computation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _compute_decile(
    data: np.ndarray,
    method: str = 'linear',
    interpolation_callable: Optional[Callable] = None
) -> np.ndarray:
    """Compute deciles using specified method."""
    if interpolation_callable is not None:
        return interpolation_callable(data)
    sorted_data = np.sort(data)
    n = len(sorted_data)
    deciles = np.zeros(10)

    for i in range(1, 11):
        p = i / 10
        if method == 'linear':
            h = (n - 1) * p
            j = int(h)
            g = h - j
            if j == n - 1:
                deciles[i-1] = sorted_data[j]
            else:
                deciles[i-1] = (1 - g) * sorted_data[j] + g * sorted_data[j+1]
        elif method == 'nearest':
            h = (n - 1) * p
            j = int(round(h))
            deciles[i-1] = sorted_data[min(max(j, 0), n - 1)]
        elif method == 'lower':
            h = (n + 1) * p
            j = int(h)
            deciles[i-1] = sorted_data[min(max(j - 1, 0), n - 1)]
        elif method == 'higher':
            h = (n + 1) * p
            j = int(h)
            deciles[i-1] = sorted_data[min(max(j, 0), n - 1)]
        elif method == 'midpoint':
            h = (n + 1) * p
            j = int(h)
            if j == n:
                deciles[i-1] = (sorted_data[j - 1] + sorted_data[j]) / 2
            else:
                deciles[i-1] = (sorted_data[j - 1] + sorted_data[j]) / 2
    return deciles

def _compute_metrics(
    data: np.ndarray,
    deciles: np.ndarray
) -> Dict[str, float]:
    """Compute metrics for decile computation."""
    metrics = {}
    # Example metric: mean absolute error between consecutive deciles
    if len(deciles) > 1:
        mae = np.mean(np.abs(np.diff(deciles)))
        metrics['decile_mae'] = mae
    return metrics

def decile_fit(
    data: np.ndarray,
    method: str = 'linear',
    interpolation_callable: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metrics_callables: Optional[Dict[str, Callable[[np.ndarray], float]]] = None
) -> Dict:
    """
    Compute deciles for given data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    method : str, optional
        Interpolation method ('linear', 'nearest', 'lower', 'higher', 'midpoint').
    interpolation_callable : callable, optional
        Custom interpolation function.
    metrics_callables : dict, optional
        Dictionary of custom metric functions.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data)

    # Compute deciles
    deciles = _compute_decile(
        data,
        method=method,
        interpolation_callable=interpolation_callable
    )

    # Compute metrics
    metrics = _compute_metrics(data, deciles)
    if metrics_callables is not None:
        for name, func in metrics_callables.items():
            try:
                metrics[name] = func(data)
            except Exception as e:
                warnings.append(f"Metric {name} computation failed: {str(e)}")

    # Prepare output
    result = {
        'result': deciles,
        'metrics': metrics,
        'params_used': {
            'method': method,
            'interpolation_callable': interpolation_callable is not None
        },
        'warnings': []
    }

    return result

# Example usage:
# data = np.random.randn(100)
# decile_fit(data, method='linear')

################################################################################
# percentile
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def percentile_fit(
    data: np.ndarray,
    percentile: float = 50.0,
    method: str = 'linear',
    interpolation: Optional[Callable[[np.ndarray, float], float]] = None,
    axis: Optional[int] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Calculate the percentile of a dataset.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    percentile : float, optional (default=50.0)
        Percentile to compute (0-100).
    method : str, optional (default='linear')
        Interpolation method ('linear', 'lower', 'higher', 'midpoint', 'nearest').
    interpolation : Callable, optional
        Custom interpolation function.
    axis : int, optional
        Axis along which to compute percentiles. If None, computes over flattened array.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': computed percentile value
        - 'metrics': empty dict (no metrics for this function)
        - 'params_used': parameters used
        - 'warnings': any warnings encountered

    Example:
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> percentile_fit(data, percentile=75)
    """
    # Validate inputs
    _validate_inputs(data, percentile)

    # Set default interpolation if not provided
    if interpolation is None:
        interpolation = _get_interpolation_method(method)

    # Compute percentile
    result = _compute_percentile(data, percentile, interpolation, axis)

    return {
        'result': result,
        'metrics': {},
        'params_used': {
            'percentile': percentile,
            'method': method,
            'axis': axis
        },
        'warnings': _check_warnings(data, percentile)
    }

def _validate_inputs(
    data: np.ndarray,
    percentile: float
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Data must contain numeric values")
    if not (0 <= percentile <= 100):
        raise ValueError("Percentile must be between 0 and 100")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")

def _get_interpolation_method(
    method: str
) -> Callable[[np.ndarray, float], float]:
    """Get interpolation function based on method name."""
    methods = {
        'linear': _linear_interpolation,
        'lower': _lower_bound_interpolation,
        'higher': _higher_bound_interpolation,
        'midpoint': _midpoint_interpolation,
        'nearest': _nearest_interpolation
    }
    if method not in methods:
        raise ValueError(f"Unknown interpolation method: {method}")
    return methods[method]

def _compute_percentile(
    data: np.ndarray,
    percentile: float,
    interpolation: Callable[[np.ndarray, float], float],
    axis: Optional[int]
) -> Union[float, np.ndarray]:
    """Compute percentile using specified interpolation."""
    if axis is not None:
        return np.apply_along_axis(
            lambda x: interpolation(x, percentile),
            axis,
            data
        )
    return interpolation(data.flatten(), percentile)

def _linear_interpolation(
    data: np.ndarray,
    percentile: float
) -> float:
    """Linear interpolation for percentile calculation."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    p = (n - 1) * (percentile / 100.0)
    k = int(p)
    d = p - k
    if k == n - 1:
        return sorted_data[k]
    return sorted_data[k] + d * (sorted_data[k + 1] - sorted_data[k])

def _lower_bound_interpolation(
    data: np.ndarray,
    percentile: float
) -> float:
    """Lower bound interpolation for percentile calculation."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    k = int(np.ceil((n * percentile / 100.0)) - 1)
    return sorted_data[k]

def _higher_bound_interpolation(
    data: np.ndarray,
    percentile: float
) -> float:
    """Higher bound interpolation for percentile calculation."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    k = int(np.floor((n * percentile / 100.0)) - 1)
    return sorted_data[k]

def _midpoint_interpolation(
    data: np.ndarray,
    percentile: float
) -> float:
    """Midpoint interpolation for percentile calculation."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    k = int(np.ceil((n * percentile / 100.0)) - 1)
    if k == n:
        return sorted_data[k - 1]
    if k == 0:
        return sorted_data[0]
    return (sorted_data[k - 1] + sorted_data[k]) / 2

def _nearest_interpolation(
    data: np.ndarray,
    percentile: float
) -> float:
    """Nearest neighbor interpolation for percentile calculation."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    k = int(round((n * percentile / 100.0)) - 1)
    if k < 0:
        return sorted_data[0]
    if k >= n:
        return sorted_data[n - 1]
    return sorted_data[k]

def _check_warnings(
    data: np.ndarray,
    percentile: float
) -> Dict[str, str]:
    """Check for potential warnings."""
    warnings = {}
    if len(data) < 2:
        warnings['small_sample'] = "Input data contains fewer than 2 values"
    if percentile == 0 or percentile == 100:
        warnings['edge_case'] = "Percentile at boundary (0 or 100)"
    return warnings

################################################################################
# quantile_general
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def quantile_general_compute(
    data: np.ndarray,
    quantile: float = 0.5,
    method: str = 'linear',
    interpolation: Optional[str] = None,
    axis: Optional[int] = None,
    nan_policy: str = 'propagate',
    keepdims: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Compute the general quantile of an array along the specified axis.

    Parameters
    ----------
    data : np.ndarray
        Input array.
    quantile : float, default=0.5
        Quantile to compute (0 <= quantile <= 1).
    method : str, default='linear'
        Interpolation method to use. Options: 'linear', 'lower', 'higher', 'midpoint', 'nearest'.
    interpolation : Optional[str], default=None
        Deprecated, use method instead.
    axis : Optional[int], default=None
        Axis along which to compute the quantile. If None, compute over the flattened array.
    nan_policy : str, default='propagate'
        How to handle NaN values. Options: 'propagate', 'raise', 'omit'.
    keepdims : bool, default=False
        If True, the output will broadcast correctly against the input.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Computed quantile value(s).
        - 'metrics': Metrics related to the computation.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings generated during computation.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> quantile_general_compute(data, quantile=0.5)
    {
        'result': 3.0,
        'metrics': {'method': 'linear'},
        'params_used': {
            'quantile': 0.5,
            'method': 'linear',
            'axis': None,
            'nan_policy': 'propagate'
        },
        'warnings': []
    }
    """
    # Validate inputs
    _validate_inputs(data, quantile, method, nan_policy)

    # Set default method if interpolation is provided
    if interpolation is not None:
        method = interpolation

    # Compute quantile
    result, metrics = _compute_quantile(
        data,
        quantile=quantile,
        method=method,
        axis=axis,
        nan_policy=nan_policy,
        keepdims=keepdims
    )

    # Prepare output dictionary
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'quantile': quantile,
            'method': method,
            'axis': axis,
            'nan_policy': nan_policy
        },
        'warnings': []
    }

    return output

def _validate_inputs(
    data: np.ndarray,
    quantile: float,
    method: str,
    nan_policy: str
) -> None:
    """Validate input parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array")
    if not (0 <= quantile <= 1):
        raise ValueError("quantile must be between 0 and 1")
    if method not in ['linear', 'lower', 'higher', 'midpoint', 'nearest']:
        raise ValueError("method must be one of: 'linear', 'lower', 'higher', 'midpoint', 'nearest'")
    if nan_policy not in ['propagate', 'raise', 'omit']:
        raise ValueError("nan_policy must be one of: 'propagate', 'raise', 'omit'")
    if np.any(np.isnan(data)) and nan_policy == 'raise':
        raise ValueError("Input contains NaN values")

def _compute_quantile(
    data: np.ndarray,
    quantile: float,
    method: str,
    axis: Optional[int],
    nan_policy: str,
    keepdims: bool
) -> tuple:
    """Compute the quantile using the specified method."""
    if axis is not None:
        data = np.moveaxis(data, axis, -1)

    if nan_policy == 'omit':
        data = data[~np.isnan(data)]

    sorted_data = np.sort(data, axis=-1)
    n = sorted_data.shape[-1]
    k = (n - 1) * quantile

    if method == 'linear':
        f = np.floor(k).astype(int)
        c = k - f
        result = (1 - c) * sorted_data[..., f] + c * sorted_data[..., f + 1]
    elif method == 'lower':
        result = sorted_data[..., np.floor(k).astype(int)]
    elif method == 'higher':
        result = sorted_data[..., np.ceil(k).astype(int)]
    elif method == 'midpoint':
        f = np.floor(k / 2).astype(int)
        c = (k / 2) - f
        result = (1 - c) * sorted_data[..., f] + c * sorted_data[..., f + 1]
    elif method == 'nearest':
        k_rounded = np.round(k)
        result = sorted_data[..., k_rounded.astype(int)]

    if keepdims:
        shape = list(sorted_data.shape)
        shape[-1] = 1
        result = np.reshape(result, shape)

    metrics = {'method': method}
    return result, metrics

################################################################################
# median
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_input(data: np.ndarray) -> None:
    """Validate input data for median calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array")
    if np.isnan(data).any():
        raise ValueError("Input contains NaN values")
    if not np.isfinite(data).all():
        raise ValueError("Input contains infinite values")

def compute_median(data: np.ndarray, method: str = 'linear') -> float:
    """Compute median using specified interpolation method."""
    if method not in ['linear', 'lower', 'higher', 'midpoint']:
        raise ValueError("Method must be one of: 'linear', 'lower', 'higher', 'midpoint'")

    sorted_data = np.sort(data)
    n = len(sorted_data)

    if n % 2 == 1:
        return float(sorted_data[n//2])
    else:
        if method == 'linear':
            return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        elif method == 'lower':
            return float(sorted_data[n//2 - 1])
        elif method == 'higher':
            return float(sorted_data[n//2])
        else:  # midpoint
            return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2

def median_fit(
    data: np.ndarray,
    method: str = 'linear',
    custom_metric: Optional[Callable[[np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute median with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data array
    method : str, optional
        Interpolation method for even-length arrays (default: 'linear')
    custom_metric : callable, optional
        Custom metric function that takes data and returns a float

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.array([1, 3, 5, 7])
    >>> median_fit(data)
    """
    validate_input(data)

    result = compute_median(data, method=method)

    metrics = {}
    if custom_metric is not None:
        try:
            metrics['custom'] = custom_metric(data)
        except Exception as e:
            warnings.append(f"Custom metric calculation failed: {str(e)}")

    params_used = {
        'method': method,
        'custom_metric': custom_metric.__name__ if custom_metric else None
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# mode
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(data: np.ndarray) -> None:
    """Validate input data for mode computation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array")
    if np.isnan(data).any():
        raise ValueError("Input contains NaN values")

def compute_frequency(data: np.ndarray) -> Dict[float, int]:
    """Compute frequency of each value in the data."""
    unique_values, counts = np.unique(data, return_counts=True)
    return dict(zip(unique_values, counts))

def find_modes(frequencies: Dict[float, int]) -> np.ndarray:
    """Find all modes from frequency dictionary."""
    max_count = max(frequencies.values())
    modes = np.array([k for k, v in frequencies.items() if v == max_count])
    return modes

def mode_fit(
    data: np.ndarray,
    *,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: abs(x - y),
    mode_selection: str = "all",
    return_frequencies: bool = False
) -> Dict[str, Union[np.ndarray, Dict[float, int], Dict]]:
    """
    Compute the mode(s) of a dataset.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    distance_metric : Callable, optional
        Distance metric function (default: absolute difference).
    mode_selection : str, optional
        Whether to return all modes or just one (default: "all").
    return_frequencies : bool, optional
        Whether to return frequencies (default: False).

    Returns
    -------
    dict
        Dictionary containing:
        - "result": array of mode(s)
        - "frequencies": frequency dictionary (if return_frequencies=True)
        - "params_used": parameters used
        - "warnings": any warnings

    Examples
    --------
    >>> data = np.array([1, 2, 2, 3, 4])
    >>> mode_fit(data)
    {
        'result': array([2]),
        'params_used': {'distance_metric': <function mode_fit.<locals>.<lambda>>, 'mode_selection': 'all', 'return_frequencies': False},
        'warnings': []
    }
    """
    validate_input(data)
    frequencies = compute_frequency(data)
    modes = find_modes(frequencies)

    if mode_selection == "all":
        result = modes
    elif mode_selection == "first":
        result = np.array([modes[0]])
    else:
        raise ValueError("mode_selection must be either 'all' or 'first'")

    output = {
        "result": result,
        "params_used": {
            "distance_metric": distance_metric,
            "mode_selection": mode_selection,
            "return_frequencies": return_frequencies
        },
        "warnings": []
    }

    if return_frequencies:
        output["frequencies"] = frequencies

    return output

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
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def _calculate_quantiles(data: np.ndarray, quantiles: tuple = (0.25, 0.75)) -> np.ndarray:
    """Calculate quantiles for the given data."""
    return np.quantile(data, quantiles)

def interquartile_range_compute(
    data: Union[np.ndarray, list],
    quantiles: tuple = (0.25, 0.75),
    normalize: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[float, Dict, str]]:
    """
    Compute the interquartile range (IQR) of a dataset.

    Parameters
    ----------
    data : Union[np.ndarray, list]
        Input data for which to compute the IQR.
    quantiles : tuple, optional
        Quantiles to use for IQR calculation (default is (0.25, 0.75)).
    normalize : Optional[Callable], optional
        Normalization function to apply before calculation (default is None).
    **kwargs : dict
        Additional keyword arguments for normalization function.

    Returns
    -------
    Dict[str, Union[float, Dict, str]]
        Dictionary containing:
        - "result": computed IQR
        - "metrics": dictionary of additional metrics
        - "params_used": parameters used in the calculation
        - "warnings": any warnings generated

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> result = interquartile_range_compute(data)
    """
    # Convert input to numpy array if necessary
    data = np.array(data, dtype=np.float64)

    # Validate input
    _validate_input(data)

    # Apply normalization if specified
    if normalize is not None:
        data = normalize(data, **kwargs)

    # Calculate quantiles
    q1, q3 = _calculate_quantiles(data, quantiles)

    # Compute IQR
    iqr = q3 - q1

    # Prepare output dictionary
    result_dict = {
        "result": float(iqr),
        "metrics": {
            "q1": float(q1),
            "q3": float(q3)
        },
        "params_used": {
            "quantiles": quantiles,
            "normalize": normalize.__name__ if normalize else None
        },
        "warnings": []
    }

    return result_dict

def standard_normalize(data: np.ndarray) -> np.ndarray:
    """Standard normalization (z-score)."""
    return (data - np.mean(data)) / np.std(data)

def robust_normalize(data: np.ndarray) -> np.ndarray:
    """Robust normalization using median and MAD."""
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    return (data - med) / mad

################################################################################
# percentile_rank
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_input(data: np.ndarray) -> None:
    """Validate input data for percentile rank calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")

def default_percentile_rank(data: np.ndarray, percentile: float) -> float:
    """Default implementation of percentile rank calculation."""
    validate_input(data)
    sorted_data = np.sort(data)
    n = len(sorted_data)
    k = (n - 1) * (percentile / 100.0)
    f = np.floor(k)
    c = np.ceil(k)

    if f == c:
        return sorted_data[int(k)]
    d0 = (c - k) * sorted_data[int(f)] + (k - f) * sorted_data[int(c)]
    return d0

def percentile_rank_fit(
    data: np.ndarray,
    percentile: float = 50.0,
    method: str = 'default',
    custom_method: Optional[Callable[[np.ndarray, float], float]] = None,
    **kwargs
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Calculate percentile rank for given data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    percentile : float, optional
        Percentile to calculate (0-100), default 50.0.
    method : str, optional
        Method to use ('default'), default 'default'.
    custom_method : callable, optional
        Custom percentile rank calculation function.
    **kwargs :
        Additional keyword arguments for custom methods.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings.
    """
    result = {}
    metrics = {}
    params_used = {'method': method}
    warnings = []

    if custom_method is not None:
        try:
            result['percentile_value'] = custom_method(data, percentile, **kwargs)
            params_used['custom_method'] = True
        except Exception as e:
            warnings.append(f"Custom method failed: {str(e)}")
            result['percentile_value'] = default_percentile_rank(data, percentile)
    else:
        if method == 'default':
            result['percentile_value'] = default_percentile_rank(data, percentile)
        else:
            warnings.append(f"Unknown method '{method}'. Using default.")
            result['percentile_value'] = default_percentile_rank(data, percentile)

    metrics['percentile'] = percentile
    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

# Example usage:
# data = np.array([1, 2, 3, 4, 5])
# result = percentile_rank_fit(data, percentile=25)

################################################################################
# cumulative_distribution
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def cumulative_distribution_fit(
    data: np.ndarray,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Compute the cumulative distribution function (CDF) of a dataset with various options.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to evaluate the fit ('mse', 'mae', 'r2', 'logloss') or custom callable.
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
        Custom metric function if not using built-in metrics.
    custom_distance : callable, optional
        Custom distance function if not using built-in distances.

    Returns:
    --------
    dict
        Dictionary containing 'result', 'metrics', 'params_used', and 'warnings'.
    """
    # Validate inputs
    _validate_inputs(data, normalization, metric, distance, solver, regularization)

    # Normalize data
    normalized_data = _normalize_data(data, normalization)

    # Choose metric and distance functions
    metric_func = _get_metric_function(metric, custom_metric)
    distance_func = _get_distance_function(distance, custom_distance)

    # Solve for CDF parameters
    cdf_params = _solve_cdf(
        normalized_data,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Compute metrics
    metrics = _compute_metrics(normalized_data, cdf_params, metric_func)

    # Prepare output
    result = {
        'result': _compute_cdf(normalized_data, cdf_params),
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
            'distance': distance if isinstance(distance, str) else 'custom',
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(normalized_data, cdf_params)
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    normalization: str,
    metric: Union[str, Callable],
    distance: str,
    solver: str,
    regularization: Optional[str]
) -> None:
    """Validate input parameters."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")

    valid_normalizations = ['none', 'standard', 'minmax', 'robust']
    if normalization not in valid_normalizations:
        raise ValueError(f"Normalization must be one of {valid_normalizations}.")

    valid_metrics = ['mse', 'mae', 'r2', 'logloss']
    if isinstance(metric, str) and metric not in valid_metrics:
        raise ValueError(f"Metric must be one of {valid_metrics} or a custom callable.")

    valid_distances = ['euclidean', 'manhattan', 'cosine', 'minkowski']
    if distance not in valid_distances:
        raise ValueError(f"Distance must be one of {valid_distances} or a custom callable.")

    valid_solvers = ['closed_form', 'gradient_descent', 'newton', 'coordinate_descent']
    if solver not in valid_solvers:
        raise ValueError(f"Solver must be one of {valid_solvers}.")

    valid_regularizations = [None, 'l1', 'l2', 'elasticnet']
    if regularization not in valid_regularizations:
        raise ValueError(f"Regularization must be one of {valid_regularizations}.")

def _normalize_data(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Normalize data using specified method."""
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
        raise ValueError("Invalid normalization method.")

def _get_metric_function(
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Callable:
    """Get metric function based on input."""
    if isinstance(metric, str):
        if metric == 'mse':
            return _mse
        elif metric == 'mae':
            return _mae
        elif metric == 'r2':
            return _r2
        elif metric == 'logloss':
            return _logloss
    if custom_metric is not None:
        return custom_metric
    raise ValueError("Invalid metric specification.")

def _get_distance_function(
    distance: str,
    custom_distance: Optional[Callable]
) -> Callable:
    """Get distance function based on input."""
    if isinstance(distance, str):
        if distance == 'euclidean':
            return _euclidean_distance
        elif distance == 'manhattan':
            return _manhattan_distance
        elif distance == 'cosine':
            return _cosine_distance
        elif distance == 'minkowski':
            return _minkowski_distance
    if custom_distance is not None:
        return custom_distance
    raise ValueError("Invalid distance specification.")

def _solve_cdf(
    data: np.ndarray,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict:
    """Solve for CDF parameters using specified solver."""
    if solver == 'closed_form':
        return _solve_closed_form(data, regularization)
    elif solver == 'gradient_descent':
        return _solve_gradient_descent(data, regularization, tol, max_iter)
    elif solver == 'newton':
        return _solve_newton(data, regularization, tol, max_iter)
    elif solver == 'coordinate_descent':
        return _solve_coordinate_descent(data, regularization, tol, max_iter)
    else:
        raise ValueError("Invalid solver specification.")

def _solve_closed_form(
    data: np.ndarray,
    regularization: Optional[str]
) -> Dict:
    """Solve for CDF parameters using closed-form solution."""
    sorted_data = np.sort(data)
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return {'x': sorted_data, 'cdf': cdf_values}

def _solve_gradient_descent(
    data: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict:
    """Solve for CDF parameters using gradient descent."""
    # Placeholder implementation
    return {'x': data, 'cdf': np.linspace(0, 1, len(data))}

def _solve_newton(
    data: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict:
    """Solve for CDF parameters using Newton's method."""
    # Placeholder implementation
    return {'x': data, 'cdf': np.linspace(0, 1, len(data))}

def _solve_coordinate_descent(
    data: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict:
    """Solve for CDF parameters using coordinate descent."""
    # Placeholder implementation
    return {'x': data, 'cdf': np.linspace(0, 1, len(data))}

def _compute_metrics(
    data: np.ndarray,
    cdf_params: Dict,
    metric_func: Callable
) -> Dict:
    """Compute metrics for the CDF fit."""
    return {'metric': metric_func(data, cdf_params)}

def _compute_cdf(
    data: np.ndarray,
    params: Dict
) -> np.ndarray:
    """Compute the CDF values."""
    sorted_data = np.sort(data)
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return np.column_stack((sorted_data, cdf_values))

def _check_warnings(
    data: np.ndarray,
    params: Dict
) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isnan(params['cdf'])):
        warnings.append("CDF contains NaN values.")
    if np.any(params['cdf'] < 0) or np.any(params['cdf'] > 1):
        warnings.append("CDF values outside [0, 1] range.")
    return warnings

# Example metric functions
def _mse(data: np.ndarray, params: Dict) -> float:
    """Mean Squared Error."""
    return np.mean((data - params['cdf']) ** 2)

def _mae(data: np.ndarray, params: Dict) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(data - params['cdf']))

def _r2(data: np.ndarray, params: Dict) -> float:
    """R-squared."""
    ss_res = np.sum((data - params['cdf']) ** 2)
    ss_tot = np.sum((data - np.mean(data)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def _logloss(data: np.ndarray, params: Dict) -> float:
    """Log Loss."""
    epsilon = 1e-8
    cdf = np.clip(params['cdf'], epsilon, 1 - epsilon)
    return -np.mean(data * np.log(cdf) + (1 - data) * np.log(1 - cdf))

# Example distance functions
def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _minkowski_distance(a: np.ndarray, b: np.ndarray, p: float = 2) -> float:
    """Minkowski distance."""
    return np.sum(np.abs(a - b) ** p) ** (1 / p)
