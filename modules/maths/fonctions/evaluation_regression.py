"""
Quantix – Module evaluation_regression
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# mse
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("Inputs must not contain NaN values")
    if np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError("Inputs must not contain infinite values")

def _compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def mse_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    metric_func : Optional[Callable]
        Custom metric function. If None, uses default MSE.

    Returns:
    --------
    Dict containing:
        - "result": computed metric value
        - "metrics": dictionary of metrics (only one in this case)
        - "params_used": parameters used
        - "warnings": any warnings encountered

    Example:
    --------
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 2.0, 2.9])
    >>> mse_fit(y_true, y_pred)
    {
        'result': 0.013333333333333334,
        'metrics': {'mse': 0.013333333333333334},
        'params_used': {'metric_func': None},
        'warnings': []
    }
    """
    _validate_inputs(y_true, y_pred)

    warnings = []

    # Use custom metric if provided
    if metric_func is not None:
        result = metric_func(y_true, y_pred)
        metrics = {'custom_metric': result}
    else:
        result = _compute_mse(y_true, y_pred)
        metrics = {'mse': result}

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {'metric_func': metric_func},
        'warnings': warnings
    }

################################################################################
# rmse
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs must not contain NaN values")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs must not contain infinite values")

def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    squared_errors = (y_true - y_pred) ** 2
    mse = np.mean(squared_errors)
    return np.sqrt(mse)

def rmse_compute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute RMSE and optionally other metrics.

    Parameters:
    - y_true: Ground truth values
    - y_pred: Predicted values
    - metric_func: Optional callable for additional metrics

    Returns:
    Dictionary containing results, metrics, and warnings
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Compute RMSE
    rmse_value = _compute_rmse(y_true, y_pred)

    # Prepare results
    result = {
        "result": rmse_value,
        "metrics": {},
        "params_used": {
            "metric_func": metric_func.__name__ if metric_func else None
        },
        "warnings": []
    }

    # Compute additional metrics if provided
    if metric_func is not None:
        try:
            result["metrics"]["custom_metric"] = metric_func(y_true, y_pred)
        except Exception as e:
            result["warnings"].append(f"Custom metric computation failed: {str(e)}")

    return result

# Example usage:
"""
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])
result = rmse_compute(y_true, y_pred)
"""

################################################################################
# mae
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays for MAE calculation."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs must not contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs must not contain infinite values.")

def compute_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """Compute Mean Absolute Error (MAE)."""
    validate_inputs(y_true, y_pred)

    errors = np.abs(y_true - y_pred)
    if sample_weight is not None:
        if len(sample_weight) != len(y_true):
            raise ValueError("sample_weight must have the same length as y_true.")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight must be non-negative.")
        return np.average(errors, weights=sample_weight)
    else:
        return np.mean(errors)

def mae_compute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute Mean Absolute Error (MAE) with additional metrics and diagnostics.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    sample_weight : Optional[np.ndarray], default=None
        Individual weights for each sample.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - "result": float, the MAE value
        - "metrics": dict of additional metrics
        - "params_used": dict of parameters used
        - "warnings": list of warnings (if any)
    """
    result = compute_mae(y_true, y_pred, sample_weight)

    metrics = {
        "mae": result,
        "median_absolute_error": np.median(np.abs(y_true - y_pred))
    }

    params_used = {
        "sample_weight": sample_weight is not None
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
# mae_result = mae_compute(y_true=np.array([1, 2, 3]), y_pred=np.array([1.5, 2.5, 3.5]))

################################################################################
# r_squared
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs must not contain NaN values")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs must not contain infinite values")

def _compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared metric."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def r_squared_compute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute R-squared metric for regression evaluation.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    metric_func : Optional[Callable]
        Custom metric function. If None, uses default R-squared computation.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - "result": computed R-squared value
        - "metrics": dictionary of metrics (currently just R-squared)
        - "params_used": parameters used in computation
        - "warnings": list of warnings (empty if no warnings)
    """
    _validate_inputs(y_true, y_pred)

    params_used = {
        "metric_func": metric_func.__name__ if metric_func else "_compute_r_squared"
    }

    warnings = []

    if metric_func is None:
        result = _compute_r_squared(y_true, y_pred)
    else:
        try:
            result = metric_func(y_true, y_pred)
        except Exception as e:
            raise RuntimeError(f"Custom metric function failed: {str(e)}")

    metrics = {
        "r_squared": result
    }

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
"""
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])
result = r_squared_compute(y_true, y_pred)
"""

################################################################################
# adjusted_r_squared
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs must not contain NaN values")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs must not contain infinite values")

def _compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the R-squared metric."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def adjusted_r_squared_compute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int
) -> Dict[str, Any]:
    """
    Compute the adjusted R-squared metric for regression evaluation.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    n_features : int
        Number of features in the regression model.

    Returns:
    --------
    dict
        Dictionary containing:
        - result: float, the adjusted R-squared value
        - metrics: dict, additional metrics (currently just r_squared)
        - params_used: dict, parameters used in computation
        - warnings: list, any warnings generated

    Example:
    --------
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> adjusted_r_squared_compute(y_true, y_pred, n_features=2)
    """
    _validate_inputs(y_true, y_pred)

    r_squared = _compute_r_squared(y_true, y_pred)
    n_samples = len(y_true)

    if n_features >= n_samples:
        adjusted_r2 = float('-inf')
        warnings = ["Number of features is greater than or equal to number of samples"]
    else:
        adjusted_r2 = 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - n_features - 1)
        warnings = []

    return {
        "result": adjusted_r2,
        "metrics": {"r_squared": r_squared},
        "params_used": {
            "n_samples": n_samples,
            "n_features": n_features
        },
        "warnings": warnings
    }

################################################################################
# mdae
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def mdae_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mae",
    normalize: bool = False,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Compute the Mean Directional Absolute Error (MDAE) between true and predicted values.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    metric : str or callable, optional
        Metric to use for evaluation. Default is "mae".
    normalize : bool, optional
        Whether to normalize the data before computation. Default is False.
    custom_metric : callable, optional
        Custom metric function to use. If provided, overrides the `metric` parameter.

    Returns:
    --------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.

    Example:
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 2.2, 2.9])
    >>> result = mdae_fit(y_true, y_pred)
    """
    # Validate inputs
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_true and y_pred must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("y_true and y_pred must not contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_true and y_pred must not contain infinite values.")

    # Normalize data if required
    if normalize:
        y_true = (y_true - np.mean(y_true)) / np.std(y_true)
        y_pred = (y_pred - np.mean(y_pred)) / np.std(y_pred)

    # Compute MDAE
    directional_errors = np.abs((y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    mdae_value = np.mean(directional_errors)

    # Compute additional metrics if required
    metrics = {}
    if isinstance(metric, str):
        if metric == "mae":
            metrics["mae"] = np.mean(np.abs(y_true - y_pred))
        elif metric == "mse":
            metrics["mse"] = np.mean((y_true - y_pred) ** 2)
        elif metric == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics["r2"] = 1 - (ss_res / ss_tot)
    elif callable(metric):
        metrics["custom"] = metric(y_true, y_pred)

    if custom_metric is not None:
        metrics["custom_metric"] = custom_metric(y_true, y_pred)

    # Prepare the result dictionary
    result_dict = {
        "result": mdae_value,
        "metrics": metrics,
        "params_used": {
            "metric": metric if not callable(metric) else "custom",
            "normalize": normalize,
            "custom_metric": bool(custom_metric)
        },
        "warnings": []
    }

    return result_dict

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Validate the input arrays.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.

    Raises:
    -------
    TypeError
        If inputs are not numpy arrays.
    ValueError
        If shapes, NaN, or infinite values are invalid.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_true and y_pred must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("y_true and y_pred must not contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_true and y_pred must not contain infinite values.")

def _normalize_data(y: np.ndarray, normalize: bool) -> np.ndarray:
    """
    Normalize the data if required.

    Parameters:
    -----------
    y : np.ndarray
        Array of values to normalize.
    normalize : bool
        Whether to normalize the data.

    Returns:
    --------
    np.ndarray
        Normalized array if `normalize` is True, otherwise the original array.
    """
    if normalize:
        return (y - np.mean(y)) / np.std(y)
    return y

def _compute_mdae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Mean Directional Absolute Error (MDAE).

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.

    Returns:
    --------
    float
        The MDAE value.
    """
    directional_errors = np.abs((y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    return np.mean(directional_errors)

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """
    Compute the specified metric.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    metric : str or callable
        Metric to compute.

    Returns:
    --------
    dict
        Dictionary containing the computed metric.
    """
    metrics = {}
    if isinstance(metric, str):
        if metric == "mae":
            metrics["mae"] = np.mean(np.abs(y_true - y_pred))
        elif metric == "mse":
            metrics["mse"] = np.mean((y_true - y_pred) ** 2)
        elif metric == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics["r2"] = 1 - (ss_res / ss_tot)
    elif callable(metric):
        metrics["custom"] = metric(y_true, y_pred)

    return metrics

################################################################################
# mape
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Validate input arrays for MAPE calculation.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if np.any(y_true == 0):
        raise ValueError("MAPE is undefined when y_true contains zeros.")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or infinite values.")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or infinite values.")

def mape_compute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the Mean Absolute Percentage Error (MAPE).

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    metric : Optional[Callable]
        Custom metric function. If None, uses default MAPE.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the result and metrics.

    Examples
    --------
    >>> y_true = np.array([10, 20, 30])
    >>> y_pred = np.array([11, 19, 28])
    >>> result = mape_compute(y_true, y_pred)
    """
    validate_inputs(y_true, y_pred)

    if metric is None:
        absolute_percentage_errors = np.abs((y_true - y_pred) / y_true)
        mape_value = np.mean(absolute_percentage_errors) * 100
    else:
        mape_value = metric(y_true, y_pred)

    return {
        "result": {"mape": mape_value},
        "metrics": {"mape": mape_value},
        "params_used": {},
        "warnings": []
    }

################################################################################
# smape
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays for SMAPE calculation."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or infinite values.")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or infinite values.")
    if np.any(y_true == 0):
        raise ValueError("y_true contains zero values which can lead to division by zero.")

def _calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = (2 * np.mean(numerator / denominator)) * 100
    return smape

def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None) -> Dict[str, float]:
    """Calculate additional metrics."""
    metrics = {
        'smape': _calculate_smape(y_true, y_pred)
    }
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)
    return metrics

def smape_compute(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None) -> Dict[str, Any]:
    """
    Compute the Symmetric Mean Absolute Percentage Error (SMAPE) and additional metrics.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function to compute additional metrics.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.

    Example:
    --------
    >>> y_true = np.array([10, 20, 30])
    >>> y_pred = np.array([12, 18, 25])
    >>> result = smape_compute(y_true, y_pred)
    """
    _validate_inputs(y_true, y_pred)

    result = {
        'result': None,
        'metrics': _calculate_metrics(y_true, y_pred, custom_metric),
        'params_used': {
            'custom_metric': custom_metric is not None
        },
        'warnings': []
    }

    return result

################################################################################
# rmsle
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays for RMSLE calculation."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(y_true <= 0) or np.any(y_pred <= 0):
        raise ValueError("All values in y_true and y_pred must be positive.")

def _log_transform(values: np.ndarray) -> np.ndarray:
    """Apply log transformation to values."""
    return np.log1p(values)

def rmsle_compute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    log_transform: bool = True,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Compute the Root Mean Squared Logarithmic Error (RMSLE).

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    log_transform : bool, optional
        Whether to apply log transformation (default: True).
    custom_metric : Callable, optional
        Custom metric function to compute alongside RMSLE.

    Returns:
    --------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.

    Example:
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 2.1, 3.1])
    >>> rmsle_compute(y_true, y_pred)
    """
    _validate_inputs(y_true, y_pred)

    metrics = {}
    warnings = []

    if log_transform:
        y_true_log = _log_transform(y_true)
        y_pred_log = _log_transform(y_pred)
    else:
        y_true_log = y_true
        y_pred_log = y_pred

    squared_errors = (y_true_log - y_pred_log) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmsle_value = np.sqrt(mean_squared_error)

    metrics["rmsle"] = rmsle_value

    if custom_metric is not None:
        try:
            custom_result = custom_metric(y_true, y_pred)
            metrics["custom_metric"] = custom_result
        except Exception as e:
            warnings.append(f"Custom metric computation failed: {str(e)}")

    return {
        "result": rmsle_value,
        "metrics": metrics,
        "params_used": {
            "log_transform": log_transform
        },
        "warnings": warnings
    }

################################################################################
# max_error
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    check_nan: bool = True
) -> None:
    """Validate input arrays."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if check_nan:
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            raise ValueError("Inputs contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs contain infinite values.")

def _compute_max_error(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """Compute the maximum error between true and predicted values."""
    return np.max(np.abs(y_true - y_pred))

def max_error_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    check_nan: bool = True
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Compute the maximum error between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    check_nan : bool, optional
        Whether to check for NaN values in inputs. Default is True.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], str]]
        Dictionary containing:
        - "result": The computed maximum error.
        - "metrics": Additional metrics (empty in this case).
        - "params_used": Parameters used for computation.
        - "warnings": Any warnings generated during computation.

    Examples
    --------
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> max_error_fit(y_true, y_pred)
    {
        'result': 1.0,
        'metrics': {},
        'params_used': {'check_nan': True},
        'warnings': []
    }
    """
    _validate_inputs(y_true, y_pred, check_nan=check_nan)
    max_err = _compute_max_error(y_true, y_pred)

    return {
        "result": float(max_err),
        "metrics": {},
        "params_used": {"check_nan": check_nan},
        "warnings": []
    }

################################################################################
# explained_variance_score
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs must not contain NaN values")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs must not contain infinite values")

def _compute_explained_variance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True
) -> float:
    """Compute explained variance score."""
    y_true_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_true_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)

    if normalize:
        return 1.0 - (ss_residual / ss_total)
    else:
        return ss_total - ss_residual

def explained_variance_score_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
    metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the explained variance score for regression evaluation.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    normalize : bool, optional
        Whether to normalize the score (default is True).
    metric_func : Callable, optional
        Custom metric function that takes (y_true, y_pred) and returns a float.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.

    Example:
    --------
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> result = explained_variance_score_fit(y_true, y_pred)
    """
    _validate_inputs(y_true, y_pred)

    result = {
        "result": _compute_explained_variance(y_true, y_pred, normalize),
        "metrics": {},
        "params_used": {
            "normalize": normalize,
            "custom_metric": metric_func is not None
        },
        "warnings": []
    }

    if metric_func is not None:
        try:
            custom_metric = metric_func(y_true, y_pred)
            result["metrics"]["custom_metric"] = custom_metric
        except Exception as e:
            result["warnings"].append(f"Custom metric computation failed: {str(e)}")

    return result
