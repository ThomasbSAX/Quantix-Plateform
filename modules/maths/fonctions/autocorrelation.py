"""
Quantix – Module autocorrelation
Généré automatiquement
Date: 2026-01-06
"""

################################################################################
# definition
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    series: np.ndarray,
    max_lag: int = 10,
    normalize: str = "standard",
) -> None:
    """Validate input series and parameters."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input series must be a numpy array.")
    if len(series.shape) != 1:
        raise ValueError("Input series must be one-dimensional.")
    if np.any(np.isnan(series)) or np.any(np.isinf(series)):
        raise ValueError("Input series must not contain NaN or infinite values.")
    if max_lag <= 0:
        raise ValueError("max_lag must be a positive integer.")
    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("normalize must be one of: 'none', 'standard', 'minmax', 'robust'.")

def _normalize_series(
    series: np.ndarray,
    normalize: str = "standard",
) -> np.ndarray:
    """Normalize the input series based on the specified method."""
    if normalize == "none":
        return series
    elif normalize == "standard":
        mean = np.mean(series)
        std = np.std(series)
        return (series - mean) / (std + 1e-8)
    elif normalize == "minmax":
        min_val = np.min(series)
        max_val = np.max(series)
        return (series - min_val) / ((max_val - min_val) + 1e-8)
    elif normalize == "robust":
        median = np.median(series)
        iqr = np.percentile(series, 75) - np.percentile(series, 25)
        return (series - median) / ((iqr + 1e-8))
    else:
        raise ValueError("Invalid normalization method.")

def _compute_autocorrelation(
    series: np.ndarray,
    max_lag: int = 10,
) -> Dict[int, float]:
    """Compute autocorrelation for each lag up to max_lag."""
    n = len(series)
    mean = np.mean(series)
    autocorr = {}
    for lag in range(max_lag + 1):
        numerator = np.sum((series[lag:] - mean) * (series[:-lag] - mean))
        denominator = np.sum((series - mean) ** 2)
        autocorr[lag] = numerator / (denominator + 1e-8)
    return autocorr

def _compute_metrics(
    autocorrelation: Dict[int, float],
    metric: str = "mse",
) -> Dict[str, float]:
    """Compute metrics based on the autocorrelation results."""
    if metric == "mse":
        return {"mse": np.mean([v ** 2 for v in autocorrelation.values()])}
    elif metric == "mae":
        return {"mae": np.mean([abs(v) for v in autocorrelation.values()])}
    elif metric == "r2":
        return {"r2": 1 - np.sum([v ** 2 for v in autocorrelation.values()]) / len(autocorrelation)}
    else:
        raise ValueError("Invalid metric.")

def definition_fit(
    series: np.ndarray,
    max_lag: int = 10,
    normalize: str = "standard",
    metric: str = "mse",
) -> Dict[str, Union[Dict[int, float], Dict[str, float], Dict[str, str]]]:
    """
    Compute autocorrelation for a given time series.

    Parameters:
    -----------
    series : np.ndarray
        Input time series.
    max_lag : int, optional
        Maximum lag to compute autocorrelation for (default: 10).
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard').
    metric : str, optional
        Metric to compute ('mse', 'mae', 'r2') (default: 'mse').

    Returns:
    --------
    dict
        Dictionary containing autocorrelation results, metrics, and parameters used.
    """
    _validate_inputs(series, max_lag, normalize)
    normalized_series = _normalize_series(series, normalize)
    autocorrelation = _compute_autocorrelation(normalized_series, max_lag)
    metrics = _compute_metrics(autocorrelation, metric)

    return {
        "result": autocorrelation,
        "metrics": metrics,
        "params_used": {
            "max_lag": str(max_lag),
            "normalize": normalize,
            "metric": metric,
        },
        "warnings": [],
    }

# Example usage:
# result = definition_fit(np.random.randn(100), max_lag=5, normalize="standard", metric="mse")

################################################################################
# acf_plot
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def validate_input(series: np.ndarray) -> None:
    """Validate the input series for autocorrelation plot."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if series.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array.")
    if np.isnan(series).any() or np.isinf(series).any():
        raise ValueError("Input array must not contain NaN or inf values.")

def compute_acf(series: np.ndarray, n_lags: int = 10) -> np.ndarray:
    """Compute the autocorrelation function for a given series."""
    mean = np.mean(series)
    c0 = np.sum((series - mean) ** 2)

    def r(h: int) -> float:
        if h == 0:
            return 1.0
        c_h = np.sum((series[:-h] - mean) * (series[h:] - mean))
        return c_h / c0

    acf = np.array([r(h) for h in range(n_lags + 1)])
    return acf

def compute_confidence_intervals(acf: np.ndarray, n_lags: int) -> np.ndarray:
    """Compute the confidence intervals for the autocorrelation function."""
    n = len(acf) - 1
    ci = 1.96 / np.sqrt(n)
    return np.array([-ci, ci])

def acf_plot_fit(
    series: np.ndarray,
    n_lags: int = 10,
    normalization: str = "standard",
    confidence_level: float = 0.95,
    custom_metric: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute and plot the autocorrelation function for a given series.

    Parameters:
    -----------
    series : np.ndarray
        Input time series data.
    n_lags : int, optional
        Number of lags to compute (default is 10).
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust' (default is 'standard').
    confidence_level : float, optional
        Confidence level for the intervals (default is 0.95).
    custom_metric : Callable, optional
        Custom metric function to evaluate the autocorrelation.

    Returns:
    --------
    dict
        A dictionary containing the results, metrics, parameters used, and warnings.
    """
    validate_input(series)

    acf = compute_acf(series, n_lags)
    ci = compute_confidence_intervals(acf, n_lags)

    if normalization == "standard":
        acf = (acf - np.mean(acf)) / np.std(acf)
    elif normalization == "minmax":
        acf = (acf - np.min(acf)) / (np.max(acf) - np.min(acf))
    elif normalization == "robust":
        acf = (acf - np.median(acf)) / (np.percentile(acf, 75) - np.percentile(acf, 25))

    metrics = {}
    if custom_metric is not None:
        metrics["custom_metric"] = custom_metric(acf)

    result = {
        "acf": acf,
        "confidence_intervals": ci
    }

    params_used = {
        "n_lags": n_lags,
        "normalization": normalization,
        "confidence_level": confidence_level
    }

    warnings = []
    if n_lags > len(series) - 1:
        warnings.append("Number of lags is too large for the series length.")

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
# result = acf_plot_fit(np.random.randn(100), n_lags=20, normalization="standard")

################################################################################
# pacf_plot
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(series: np.ndarray) -> None:
    """Validate input series for PACF plot."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if series.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array")
    if np.any(np.isnan(series)) or np.any(np.isinf(series)):
        raise ValueError("Input contains NaN or infinite values")

def _compute_pacf(series: np.ndarray, max_lag: int = 10,
                  method: str = 'yw', alpha: float = 0.05) -> np.ndarray:
    """Compute partial autocorrelation function."""
    n = len(series)
    pacf = np.zeros(max_lag + 1)

    for lag in range(1, max_lag + 1):
        if method == 'yw':
            pacf[lag] = _yw_pacf(series, lag)
        elif method == 'ols':
            pacf[lag] = _ols_pacf(series, lag)
        else:
            raise ValueError(f"Unknown method: {method}")

    return pacf

def _yw_pacf(series: np.ndarray, lag: int) -> float:
    """Compute partial autocorrelation using Yule-Walker equations."""
    n = len(series)
    acf = np.correlate(series - np.mean(series), series - np.mean(series),
                       mode='full')[-n:]
    acf = acf / (acf[0] * n)

    if lag == 1:
        return acf[lag]

    # Solve Yule-Walker equations
    R = np.zeros((lag, lag))
    for i in range(lag):
        for j in range(lag - i):
            R[i, j] = acf[abs(i-j)]

    r = np.zeros(lag)
    for i in range(1, lag + 1):
        r[i-1] = acf[i]

    phi = np.linalg.solve(R, r)
    return phi[-1]

def _ols_pacf(series: np.ndarray, lag: int) -> float:
    """Compute partial autocorrelation using OLS regression."""
    X = np.column_stack([series[i:-(lag-i)] for i in range(1, lag + 1)])
    y = series[lag:]

    if X.shape[0] < X.shape[1]:
        raise ValueError("Not enough data points for OLS estimation")

    # Center the data
    X = X - np.mean(X, axis=0)
    y = y - np.mean(y)

    # OLS solution
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return beta[-1]

def _compute_confidence_intervals(pacf: np.ndarray, n: int,
                                 alpha: float = 0.05) -> tuple:
    """Compute confidence intervals for PACF values."""
    critical_value = 1.96 / np.sqrt(n)  # Approximate for large n
    lower = pacf - critical_value
    upper = pacf + critical_value
    return lower, upper

def _plot_pacf(pacf: np.ndarray, lags: np.ndarray,
               lower_ci: Optional[np.ndarray] = None,
               upper_ci: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Generate PACF plot data."""
    result = {
        'lags': lags,
        'pacf_values': pacf,
    }

    if lower_ci is not None and upper_ci is not None:
        result['confidence_intervals'] = {
            'lower': lower_ci,
            'upper': upper_ci
        }

    return result

def pacf_plot_fit(series: np.ndarray, max_lag: int = 10,
                 method: str = 'yw', alpha: float = 0.05) -> Dict[str, Any]:
    """
    Compute and plot partial autocorrelation function.

    Parameters:
    -----------
    series : np.ndarray
        Input time series data.
    max_lag : int, optional
        Maximum lag to compute (default: 10).
    method : str, optional
        Method for PACF computation ('yw' or 'ols', default: 'yw').
    alpha : float, optional
        Significance level for confidence intervals (default: 0.05).

    Returns:
    --------
    dict
        Dictionary containing PACF results, metrics, and parameters used.

    Example:
    --------
    >>> series = np.random.randn(100)
    >>> result = pacf_plot_fit(series, max_lag=5)
    """
    _validate_input(series)

    n = len(series)
    lags = np.arange(max_lag + 1)
    pacf = _compute_pacf(series, max_lag, method, alpha)

    lower_ci, upper_ci = _compute_confidence_intervals(pacf, n, alpha)
    plot_data = _plot_pacf(pacf, lags, lower_ci, upper_ci)

    return {
        'result': plot_data,
        'metrics': {},
        'params_used': {
            'max_lag': max_lag,
            'method': method,
            'alpha': alpha
        },
        'warnings': []
    }

################################################################################
# lag
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(
    series: np.ndarray,
    max_lag: int = 10,
    normalize: str = "none",
    metric: Union[str, Callable] = "mse"
) -> None:
    """Validate input parameters for lag computation."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input series must be a numpy array")
    if len(series.shape) != 1:
        raise ValueError("Input series must be 1-dimensional")
    if np.any(np.isnan(series)):
        raise ValueError("Input series contains NaN values")
    if np.any(np.isinf(series)):
        raise ValueError("Input series contains infinite values")
    if max_lag <= 0:
        raise ValueError("max_lag must be positive")
    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("Invalid normalization method")
    if isinstance(metric, str) and metric not in ["mse", "mae", "r2"]:
        raise ValueError("Invalid metric string")
    if isinstance(metric, Callable):
        try:
            metric(np.array([1.0]), np.array([1.0]))
        except Exception as e:
            raise ValueError(f"Custom metric callable failed: {str(e)}")

def _normalize_series(
    series: np.ndarray,
    method: str = "none"
) -> np.ndarray:
    """Normalize the input series using specified method."""
    if method == "none":
        return series
    elif method == "standard":
        return (series - np.mean(series)) / np.std(series)
    elif method == "minmax":
        return (series - np.min(series)) / (np.max(series) - np.min(series))
    elif method == "robust":
        median = np.median(series)
        iqr = np.percentile(series, 75) - np.percentile(series, 25)
        return (series - median) / iqr
    else:
        raise ValueError("Invalid normalization method")

def _compute_autocorrelation(
    series: np.ndarray,
    lags: int
) -> float:
    """Compute autocorrelation for a given lag."""
    n = len(series)
    mean = np.mean(series)
    numerator = np.sum((series[:-lags] - mean) * (series[lags:] - mean))
    denominator = np.sum((series[:-lags] - mean) ** 2)
    return numerator / denominator

def _compute_metric(
    series: np.ndarray,
    lags: int,
    metric_func: Union[str, Callable]
) -> float:
    """Compute the specified metric for autocorrelation at given lag."""
    if isinstance(metric_func, str):
        if metric_func == "mse":
            return np.mean((series[:-lags] - series[lags:]) ** 2)
        elif metric_func == "mae":
            return np.mean(np.abs(series[:-lags] - series[lags:]))
        elif metric_func == "r2":
            ss_res = np.sum((series[:-lags] - series[lags:]) ** 2)
            ss_tot = np.sum((series[:-lags] - np.mean(series)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    else:
        return metric_func(series[:-lags], series[lags:])

def lag_fit(
    series: np.ndarray,
    max_lag: int = 10,
    normalize: str = "none",
    metric: Union[str, Callable] = "mse"
) -> Dict[str, Any]:
    """
    Compute autocorrelation for multiple lags with configurable options.

    Parameters:
    -----------
    series : np.ndarray
        Input time series data
    max_lag : int, optional
        Maximum lag to compute (default: 10)
    normalize : str, optional
        Normalization method ("none", "standard", "minmax", "robust")
    metric : Union[str, Callable], optional
        Metric to evaluate autocorrelation ("mse", "mae", "r2") or custom callable

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> series = np.random.randn(100)
    >>> result = lag_fit(series, max_lag=5, normalize="standard")
    """
    # Validate inputs
    _validate_inputs(series, max_lag, normalize, metric)

    # Normalize series
    normalized_series = _normalize_series(series.copy(), normalize)

    # Initialize results storage
    results = {}
    metrics = {}

    warnings = []

    if len(normalized_series) <= max_lag:
        warnings.append("Warning: Series length is too short for requested max_lag")

    # Compute autocorrelation and metrics for each lag
    for lag in range(1, min(max_lag + 1, len(normalized_series))):
        try:
            acf = _compute_autocorrelation(normalized_series, lag)
            results[f"lag_{lag}"] = acf
            metrics[f"metric_lag_{lag}"] = _compute_metric(normalized_series, lag, metric)
        except Exception as e:
            warnings.append(f"Error computing lag {lag}: {str(e)}")

    return {
        "result": results,
        "metrics": metrics,
        "params_used": {
            "max_lag": max_lag,
            "normalize": normalize,
            "metric": metric if isinstance(metric, str) else "custom"
        },
        "warnings": warnings
    }

################################################################################
# stationarity
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    series: np.ndarray,
    max_lag: int = 10,
    normalize: str = "standard",
    metric: Union[str, Callable] = "mse"
) -> None:
    """Validate inputs for stationarity analysis."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input series must be a numpy array")
    if len(series.shape) != 1:
        raise ValueError("Input series must be 1-dimensional")
    if np.any(np.isnan(series)) or np.any(np.isinf(series)):
        raise ValueError("Input series contains NaN or infinite values")
    if not isinstance(max_lag, int) or max_lag < 1:
        raise ValueError("max_lag must be a positive integer")
    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("normalize must be one of: 'none', 'standard', 'minmax', 'robust'")
    if isinstance(metric, str) and metric not in ["mse", "mae", "r2"]:
        raise ValueError("metric must be one of: 'mse', 'mae', 'r2' or a callable")

def _normalize_series(
    series: np.ndarray,
    method: str = "standard"
) -> np.ndarray:
    """Normalize the input series."""
    if method == "none":
        return series
    elif method == "standard":
        return (series - np.mean(series)) / np.std(series)
    elif method == "minmax":
        return (series - np.min(series)) / (np.max(series) - np.min(series))
    elif method == "robust":
        median = np.median(series)
        iqr = np.percentile(series, 75) - np.percentile(series, 25)
        return (series - median) / iqr
    else:
        raise ValueError("Unknown normalization method")

def _compute_autocorrelation(
    series: np.ndarray,
    max_lag: int
) -> np.ndarray:
    """Compute autocorrelation for different lags."""
    n = len(series)
    mean = np.mean(series)
    autocorr = np.zeros(max_lag)

    for lag in range(1, max_lag + 1):
        numerator = np.sum((series[:n-lag] - mean) * (series[lag:] - mean))
        denominator = np.sum((series[:n-lag] - mean) ** 2)
        autocorr[lag-1] = numerator / denominator

    return autocorr

def _compute_metrics(
    series: np.ndarray,
    autocorr: np.ndarray,
    metric: Union[str, Callable] = "mse"
) -> Dict[str, float]:
    """Compute metrics for stationarity analysis."""
    metrics = {}

    if isinstance(metric, str):
        if metric == "mse":
            residuals = series[1:] - autocorr * series[:-1]
            metrics["mse"] = np.mean(residuals ** 2)
        elif metric == "mae":
            residuals = series[1:] - autocorr * series[:-1]
            metrics["mae"] = np.mean(np.abs(residuals))
        elif metric == "r2":
            y_pred = autocorr * series[:-1]
            ss_res = np.sum((series[1:] - y_pred) ** 2)
            ss_tot = np.sum((series[1:] - np.mean(series[1:])) ** 2)
            metrics["r2"] = 1 - (ss_res / ss_tot)
    else:
        metrics["custom"] = metric(series, autocorr)

    return metrics

def stationarity_fit(
    series: np.ndarray,
    max_lag: int = 10,
    normalize: str = "standard",
    metric: Union[str, Callable] = "mse"
) -> Dict:
    """
    Perform stationarity analysis on a time series.

    Parameters
    ----------
    series : np.ndarray
        Input time series data.
    max_lag : int, optional
        Maximum lag to consider for autocorrelation, by default 10.
    normalize : str, optional
        Normalization method, by default "standard".
    metric : Union[str, Callable], optional
        Metric to evaluate stationarity, by default "mse".

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> series = np.random.randn(100)
    >>> result = stationarity_fit(series, max_lag=5, normalize="standard", metric="mse")
    """
    # Validate inputs
    _validate_inputs(series, max_lag, normalize, metric)

    # Normalize series
    normalized_series = _normalize_series(series, normalize)

    # Compute autocorrelation
    autocorr = _compute_autocorrelation(normalized_series, max_lag)

    # Compute metrics
    metrics = _compute_metrics(normalized_series, autocorr, metric)

    # Prepare output
    result = {
        "result": {
            "autocorrelation": autocorr,
            "optimal_lag": np.argmax(autocorr) + 1
        },
        "metrics": metrics,
        "params_used": {
            "max_lag": max_lag,
            "normalize": normalize,
            "metric": metric
        },
        "warnings": []
    }

    return result

################################################################################
# white_noise
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    series: np.ndarray,
    max_lag: int = 10,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Union[np.ndarray, int]]:
    """
    Validate and preprocess input series for white noise analysis.

    Parameters
    ----------
    series : np.ndarray
        Input time series data.
    max_lag : int, optional
        Maximum lag to consider for autocorrelation, by default 10.
    normalizer : Optional[Callable], optional
        Normalization function, by default None.

    Returns
    -------
    Dict[str, Union[np.ndarray, int]]
        Validated and processed inputs.

    Raises
    ------
    ValueError
        If input series is invalid.
    """
    if not isinstance(series, np.ndarray):
        raise ValueError("Input series must be a numpy array")
    if len(series.shape) != 1:
        raise ValueError("Input series must be one-dimensional")
    if np.any(np.isnan(series)) or np.any(np.isinf(series)):
        raise ValueError("Input series contains NaN or infinite values")

    if normalizer is not None:
        series = normalizer(series)

    return {
        "series": series,
        "max_lag": max_lag
    }

def _compute_autocorrelation(
    series: np.ndarray,
    max_lag: int
) -> np.ndarray:
    """
    Compute autocorrelation for given lags.

    Parameters
    ----------
    series : np.ndarray
        Input time series data.
    max_lag : int
        Maximum lag to consider.

    Returns
    -------
    np.ndarray
        Autocorrelation values for each lag.
    """
    n = len(series)
    mean = np.mean(series)
    autocorr = np.zeros(max_lag + 1)

    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr[lag] = np.sum((series - mean) ** 2) / n
        else:
            autocorr[lag] = np.sum((series[:-lag] - mean) * (series[lag:] - mean)) / n

    autocorr /= autocorr[0]
    return autocorr

def _compute_metrics(
    autocorrelation: np.ndarray,
    metric_funcs: Dict[str, Callable]
) -> Dict[str, float]:
    """
    Compute metrics for white noise analysis.

    Parameters
    ----------
    autocorrelation : np.ndarray
        Autocorrelation values.
    metric_funcs : Dict[str, Callable]
        Dictionary of metric functions.

    Returns
    -------
    Dict[str, float]
        Computed metrics.
    """
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(autocorrelation)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def white_noise_fit(
    series: np.ndarray,
    max_lag: int = 10,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric_funcs: Dict[str, Callable] = None
) -> Dict:
    """
    Fit white noise model to time series data.

    Parameters
    ----------
    series : np.ndarray
        Input time series data.
    max_lag : int, optional
        Maximum lag to consider for autocorrelation, by default 10.
    normalizer : Optional[Callable], optional
        Normalization function, by default None.
    metric_funcs : Dict[str, Callable], optional
        Dictionary of metric functions to compute, by default None.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import StandardScaler
    >>> def r2_score(y_true, y_pred):
    ...     return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    ...
    >>> series = np.random.normal(0, 1, 100)
    >>> result = white_noise_fit(
    ...     series,
    ...     max_lag=5,
    ...     normalizer=StandardScaler().fit_transform,
    ...     metric_funcs={"r2": r2_score}
    ... )
    """
    # Default metrics if none provided
    if metric_funcs is None:
        def mse(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)
        metric_funcs = {"mse": mse}

    # Validate and preprocess inputs
    validated_inputs = _validate_inputs(series, max_lag, normalizer)
    series = validated_inputs["series"]
    max_lag = validated_inputs["max_lag"]

    # Compute autocorrelation
    autocorrelation = _compute_autocorrelation(series, max_lag)

    # Compute metrics
    metrics = _compute_metrics(autocorrelation, metric_funcs)

    # Prepare results
    result = {
        "result": autocorrelation,
        "metrics": metrics,
        "params_used": {
            "max_lag": max_lag,
            "normalizer": normalizer.__name__ if normalizer else None
        },
        "warnings": []
    }

    return result

# Example normalizers
def standard_normalizer(series: np.ndarray) -> np.ndarray:
    """Standard normalization (z-score)."""
    return (series - np.mean(series)) / np.std(series)

def minmax_normalizer(series: np.ndarray) -> np.ndarray:
    """Min-max normalization."""
    return (series - np.min(series)) / (np.max(series) - np.min(series))

def robust_normalizer(series: np.ndarray) -> np.ndarray:
    """Robust normalization using median and IQR."""
    q1, q3 = np.percentile(series, [25, 75])
    iqr = q3 - q1
    return (series - np.median(series)) / iqr

# Example metrics
def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    """R-squared score."""
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

################################################################################
# random_walk
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def random_walk_fit(
    data: np.ndarray,
    n_steps: int = 100,
    step_size: float = 1.0,
    initial_value: float = 0.0,
    normalize: str = "none",
    metric: Union[str, Callable] = "mse",
    solver: str = "closed_form",
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute the random walk autocorrelation model.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.
    n_steps : int, optional
        Number of steps in the random walk (default: 100).
    step_size : float, optional
        Size of each random walk step (default: 1.0).
    initial_value : float, optional
        Starting value of the random walk (default: 0.0).
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    metric : Union[str, Callable], optional
        Metric to evaluate the model ('mse', 'mae', 'r2', custom callable) (default: 'mse').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton') (default: 'closed_form').
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    custom_metric : Optional[Callable], optional
        Custom metric function (default: None).

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, n_steps, step_size, initial_value)

    # Normalize data if required
    normalized_data = _normalize_data(data, normalize)

    # Generate random walk
    random_walk = _generate_random_walk(n_steps, step_size, initial_value)

    # Compute autocorrelation
    autocorr = _compute_autocorrelation(random_walk, normalized_data)

    # Solve for parameters
    params = _solve_parameters(autocorr, solver, tol, max_iter)

    # Compute metrics
    metrics = _compute_metrics(autocorr, normalized_data, metric, custom_metric)

    # Prepare output
    result = {
        "result": autocorr,
        "metrics": metrics,
        "params_used": params,
        "warnings": _check_warnings(autocorr, metrics)
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    n_steps: int,
    step_size: float,
    initial_value: float
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")

def _normalize_data(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Normalize the input data."""
    if method == "standard":
        return (data - np.mean(data)) / np.std(data)
    elif method == "minmax":
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == "robust":
        return (data - np.median(data)) / (np.percentile(data, 75) - np.percentile(data, 25))
    else:
        return data

def _generate_random_walk(
    n_steps: int,
    step_size: float,
    initial_value: float
) -> np.ndarray:
    """Generate a random walk."""
    steps = np.random.normal(0, step_size, n_steps)
    return initial_value + np.cumsum(steps)

def _compute_autocorrelation(
    random_walk: np.ndarray,
    data: np.ndarray
) -> np.ndarray:
    """Compute autocorrelation between random walk and data."""
    return np.correlate(random_walk, data, mode='full')

def _solve_parameters(
    autocorr: np.ndarray,
    solver: str,
    tol: float,
    max_iter: int
) -> Dict:
    """Solve for model parameters."""
    if solver == "closed_form":
        return _closed_form_solution(autocorr)
    elif solver == "gradient_descent":
        return _gradient_descent_solution(autocorr, tol, max_iter)
    elif solver == "newton":
        return _newton_solution(autocorr, tol, max_iter)
    else:
        raise ValueError("Unsupported solver method.")

def _closed_form_solution(
    autocorr: np.ndarray
) -> Dict:
    """Closed form solution for parameters."""
    return {"method": "closed_form", "parameters": {}}

def _gradient_descent_solution(
    autocorr: np.ndarray,
    tol: float,
    max_iter: int
) -> Dict:
    """Gradient descent solution for parameters."""
    return {"method": "gradient_descent", "parameters": {}}

def _newton_solution(
    autocorr: np.ndarray,
    tol: float,
    max_iter: int
) -> Dict:
    """Newton's method solution for parameters."""
    return {"method": "newton", "parameters": {}}

def _compute_metrics(
    autocorr: np.ndarray,
    data: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute evaluation metrics."""
    if callable(metric):
        return {"custom_metric": metric(autocorr, data)}
    elif custom_metric is not None:
        return {"custom_metric": custom_metric(autocorr, data)}
    else:
        if metric == "mse":
            return {"mse": np.mean((autocorr - data) ** 2)}
        elif metric == "mae":
            return {"mae": np.mean(np.abs(autocorr - data))}
        elif metric == "r2":
            ss_res = np.sum((data - autocorr) ** 2)
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            return {"r2": 1 - (ss_res / ss_tot)}
        else:
            raise ValueError("Unsupported metric method.")

def _check_warnings(
    autocorr: np.ndarray,
    metrics: Dict
) -> list:
    """Check for warnings in the results."""
    warnings = []
    if np.any(np.isnan(autocorr)):
        warnings.append("NaN values detected in autocorrelation.")
    if np.any(np.isinf(autocorr)):
        warnings.append("Infinite values detected in autocorrelation.")
    if metrics.get("mse", float('inf')) > 1e6:
        warnings.append("High MSE value detected.")
    return warnings

################################################################################
# seasonality
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def seasonality_fit(
    series: np.ndarray,
    period: int,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Compute seasonality components from a time series using autocorrelation.

    Parameters
    ----------
    series : np.ndarray
        Input time series data.
    period : int
        The period of seasonality to detect.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the series, by default None.
    metric : str, optional
        Metric for evaluation ('mse', 'mae', 'r2'), by default 'mse'.
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine'), by default 'euclidean'.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent'), by default 'closed_form'.
    regularization : Optional[str], optional
        Regularization type ('none', 'l1', 'l2'), by default None.
    tol : float, optional
        Tolerance for convergence, by default 1e-6.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function, by default None.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function, by default None.

    Returns
    -------
    Dict[str, Union[Dict, float, str]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> series = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> result = seasonality_fit(series, period=2)
    """
    # Validate inputs
    _validate_inputs(series, period)

    # Normalize the series if a normalizer is provided
    normalized_series = _apply_normalization(series, normalizer)

    # Compute seasonality components
    seasonal_components = _compute_seasonal_components(normalized_series, period)

    # Solve for parameters
    params = _solve_seasonality(
        seasonal_components,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Compute metrics
    metrics = _compute_metrics(
        series,
        seasonal_components,
        metric=metric,
        distance=distance,
        custom_metric=custom_metric,
        custom_distance=custom_distance
    )

    # Prepare the result dictionary
    result = {
        "result": seasonal_components,
        "metrics": metrics,
        "params_used": {
            "period": period,
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return result

def _validate_inputs(series: np.ndarray, period: int) -> None:
    """Validate input series and period."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input series must be a numpy array.")
    if not isinstance(period, int) or period <= 0:
        raise ValueError("Period must be a positive integer.")
    if len(series) < period:
        raise ValueError("Series length must be greater than or equal to the period.")

def _apply_normalization(series: np.ndarray, normalizer: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Apply normalization to the series if a normalizer is provided."""
    if normalizer is not None:
        return normalizer(series)
    return series

def _compute_seasonal_components(series: np.ndarray, period: int) -> np.ndarray:
    """Compute seasonal components from the series."""
    n = len(series)
    seasonal_components = np.zeros(n)
    for i in range(n):
        seasonal_components[i] = series[(i + period) % n]
    return seasonal_components

def _solve_seasonality(
    seasonal_components: np.ndarray,
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, float]:
    """Solve for seasonality parameters using the specified solver."""
    if solver == 'closed_form':
        params = _solve_closed_form(seasonal_components, regularization)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(seasonal_components, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")
    return params

def _solve_closed_form(seasonal_components: np.ndarray, regularization: Optional[str]) -> Dict[str, float]:
    """Solve for seasonality parameters using closed-form solution."""
    n = len(seasonal_components)
    X = np.column_stack([np.ones(n), seasonal_components])
    y = seasonal_components
    if regularization == 'l1':
        # L1 regularization (Lasso)
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=0.1)
    elif regularization == 'l2':
        # L2 regularization (Ridge)
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=0.1)
    else:
        # No regularization (Linear Regression)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    model.fit(X, y)
    return {
        'intercept': model.intercept_,
        'coefficient': model.coef_[1]
    }

def _solve_gradient_descent(
    seasonal_components: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, float]:
    """Solve for seasonality parameters using gradient descent."""
    n = len(seasonal_components)
    X = np.column_stack([np.ones(n), seasonal_components])
    y = seasonal_components
    theta = np.zeros(2)
    learning_rate = 0.01
    for _ in range(max_iter):
        gradient = 2/n * X.T.dot(X.dot(theta) - y)
        theta -= learning_rate * gradient
        if np.linalg.norm(gradient) < tol:
            break
    return {
        'intercept': theta[0],
        'coefficient': theta[1]
    }

def _compute_metrics(
    series: np.ndarray,
    seasonal_components: np.ndarray,
    metric: str = 'mse',
    distance: str = 'euclidean',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute metrics for the seasonality fit."""
    if custom_metric is not None:
        return {'custom_metric': custom_metric(series, seasonal_components)}
    if metric == 'mse':
        return {'mse': np.mean((series - seasonal_components) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(series - seasonal_components))}
    elif metric == 'r2':
        ss_res = np.sum((series - seasonal_components) ** 2)
        ss_tot = np.sum((series - np.mean(series)) ** 2)
        return {'r2': 1 - ss_res / ss_tot}
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# partial_acf
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    x: np.ndarray,
    lags: int,
    normalization: str = "standard",
) -> None:
    """Validate input data and parameters."""
    if not isinstance(x, np.ndarray):
        raise TypeError("Input x must be a numpy array.")
    if not isinstance(lags, int) or lags < 0:
        raise ValueError("Lags must be a non-negative integer.")
    if normalization not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("Normalization must be one of: none, standard, minmax, robust.")
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Input array must not contain NaN or inf values.")

def _apply_normalization(
    x: np.ndarray,
    normalization: str = "standard",
) -> np.ndarray:
    """Apply the specified normalization to the input array."""
    if normalization == "none":
        return x
    elif normalization == "standard":
        return (x - np.mean(x)) / np.std(x)
    elif normalization == "minmax":
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    elif normalization == "robust":
        median = np.median(x)
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
        return (x - median) / iqr
    else:
        raise ValueError("Unsupported normalization method.")

def _compute_partial_acf(
    x: np.ndarray,
    lags: int,
    metric: str = "mse",
    solver: str = "closed_form",
) -> np.ndarray:
    """Compute the partial autocorrelation function."""
    n = len(x)
    acf = np.zeros(lags + 1)

    for lag in range(lags + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            y = x[lag:]
            X = np.column_stack([x[i:len(y)] for i in range(1, lag + 1)])
            if solver == "closed_form":
                beta = np.linalg.pinv(X) @ y
            else:
                raise ValueError("Unsupported solver method.")
            residuals = y - X @ beta
            if metric == "mse":
                acf[lag] = residuals.var() / x.var()
            else:
                raise ValueError("Unsupported metric method.")

    return acf

def partial_acf_fit(
    x: np.ndarray,
    lags: int = 10,
    normalization: str = "standard",
    metric: str = "mse",
    solver: str = "closed_form",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Compute the partial autocorrelation function (ACF) for a given time series.

    Parameters:
    -----------
    x : np.ndarray
        Input time series data.
    lags : int, optional
        Number of lags to compute (default is 10).
    normalization : str, optional
        Normalization method (default is "standard").
    metric : str, optional
        Metric to use for ACF computation (default is "mse").
    solver : str, optional
        Solver method to use (default is "closed_form").
    custom_metric : Callable, optional
        Custom metric function (default is None).

    Returns:
    --------
    dict
        A dictionary containing the results, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, lags, normalization)
    x_normalized = _apply_normalization(x, normalization)

    if custom_metric is not None:
        metric = "custom"

    acf_values = _compute_partial_acf(x_normalized, lags, metric, solver)

    metrics = {}
    if metric == "mse":
        metrics["metric"] = "MSE"
    elif metric == "custom":
        metrics["metric"] = "Custom"

    params_used = {
        "normalization": normalization,
        "metric": metric,
        "solver": solver,
    }

    warnings = []
    if np.any(np.isnan(acf_values)):
        warnings.append("NaN values detected in ACF computation.")

    return {
        "result": acf_values,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings,
    }

# Example usage:
# x = np.random.randn(100)
# result = partial_acf_fit(x, lags=5)

################################################################################
# confidence_intervals
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def confidence_intervals_fit(
    data: np.ndarray,
    alpha: float = 0.05,
    method: str = 'bartlett',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'mse',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Calculate confidence intervals for autocorrelation coefficients.

    Parameters
    ----------
    data : np.ndarray
        Input time series data.
    alpha : float, optional
        Significance level (default: 0.05).
    method : str, optional
        Method for confidence interval calculation ('bartlett', 'dwm', 'acf') (default: 'bartlett').
    normalizer : Callable, optional
        Function to normalize data (default: None).
    metric : str, optional
        Metric for evaluation ('mse', 'mae') (default: 'mse').
    custom_metric : Callable, optional
        Custom metric function (default: None).

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.randn(100)
    >>> result = confidence_intervals_fit(data, alpha=0.05)
    """
    # Validate inputs
    _validate_inputs(data, alpha)

    # Normalize data if specified
    if normalizer is not None:
        data = normalizer(data)

    # Calculate autocorrelation coefficients
    acf, conf_int = _calculate_confidence_intervals(data, alpha, method)

    # Calculate metrics
    metrics = _calculate_metrics(acf, conf_int, metric, custom_metric)

    # Prepare output
    result = {
        "result": {
            "acf": acf,
            "confidence_intervals": conf_int
        },
        "metrics": metrics,
        "params_used": {
            "alpha": alpha,
            "method": method
        },
        "warnings": []
    }

    return result

def _validate_inputs(data: np.ndarray, alpha: float) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")

def _calculate_confidence_intervals(
    data: np.ndarray,
    alpha: float,
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate autocorrelation coefficients and confidence intervals."""
    n = len(data)
    acf = np.correlate(data, data, mode='full')[-n:]
    acf = acf / np.arange(n, 0, -1)

    if method == 'bartlett':
        conf_int = _bartlett_confidence_intervals(acf, n, alpha)
    elif method == 'dwm':
        conf_int = _dwm_confidence_intervals(acf, n, alpha)
    elif method == 'acf':
        conf_int = _acf_confidence_intervals(acf, n, alpha)
    else:
        raise ValueError(f"Unknown method: {method}")

    return acf, conf_int

def _bartlett_confidence_intervals(
    acf: np.ndarray,
    n: int,
    alpha: float
) -> np.ndarray:
    """Calculate confidence intervals using Bartlett's formula."""
    z = 1.96  # Approximate for normal distribution
    se = np.sqrt(1 / n)
    return np.column_stack([acf - z * se, acf + z * se])

def _dwm_confidence_intervals(
    acf: np.ndarray,
    n: int,
    alpha: float
) -> np.ndarray:
    """Calculate confidence intervals using Dawson-Meyer formula."""
    z = 1.96
    se = np.sqrt((1 + 2 * np.sum(acf[1:] ** 2)) / n)
    return np.column_stack([acf - z * se, acf + z * se])

def _acf_confidence_intervals(
    acf: np.ndarray,
    n: int,
    alpha: float
) -> np.ndarray:
    """Calculate confidence intervals using ACF formula."""
    z = 1.96
    se = np.sqrt(1 / n)
    return np.column_stack([acf - z * se, acf + z * se])

def _calculate_metrics(
    acf: np.ndarray,
    conf_int: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Calculate metrics for the confidence intervals."""
    metrics = {}

    if metric == 'mse':
        mse = np.mean((acf - (conf_int[:, 0] + conf_int[:, 1]) / 2) ** 2)
        metrics['mse'] = mse
    elif metric == 'mae':
        mae = np.mean(np.abs(acf - (conf_int[:, 0] + conf_int[:, 1]) / 2))
        metrics['mae'] = mae

    if custom_metric is not None:
        metrics['custom'] = custom_metric(acf, conf_int)

    return metrics

################################################################################
# ljung_box_test
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def ljung_box_test_fit(
    series: np.ndarray,
    lags: int = 10,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    custom_metric: Optional[Callable] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict:
    """
    Perform Ljung-Box test for autocorrelation in a time series.

    Parameters
    ----------
    series : np.ndarray
        Input time series data.
    lags : int, optional
        Number of lags to consider (default is 10).
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust' (default is 'standard').
    metric : str or callable, optional
        Metric to use: 'mse', 'mae', 'r2', or custom callable (default is 'mse').
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', or 'newton' (default is 'closed_form').
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    tol : float, optional
        Tolerance for convergence (default is 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default is 1000).

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> series = np.random.randn(100)
    >>> result = ljung_box_test_fit(series, lags=5)
    """
    # Validate inputs
    _validate_inputs(series, lags)

    # Normalize data if required
    normalized_series = _normalize_data(series, normalization)

    # Calculate autocorrelation
    autocorr = _calculate_autocorrelation(normalized_series, lags)

    # Compute Ljung-Box statistic
    statistic = _compute_ljung_box_statistic(autocorr, lags)

    # Calculate p-value
    p_value = _calculate_p_value(statistic, lags)

    # Compute metrics based on user choice
    metrics = _compute_metrics(autocorr, metric, custom_metric)

    # Prepare results
    result = {
        'statistic': statistic,
        'p_value': p_value,
        'autocorrelation': autocorr
    }

    params_used = {
        'lags': lags,
        'normalization': normalization,
        'metric': metric if isinstance(metric, str) else 'custom',
        'solver': solver,
        'tol': tol,
        'max_iter': max_iter
    }

    warnings = _check_warnings(series, lags)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(series: np.ndarray, lags: int) -> None:
    """Validate input series and parameters."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input series must be a numpy array.")
    if len(series) < 2:
        raise ValueError("Input series must have at least 2 elements.")
    if not np.all(np.isfinite(series)):
        raise ValueError("Input series must contain only finite values.")
    if not isinstance(lags, int) or lags < 1:
        raise ValueError("Lags must be a positive integer.")

def _normalize_data(series: np.ndarray, method: str) -> np.ndarray:
    """Normalize the input series based on the specified method."""
    if method == 'none':
        return series
    elif method == 'standard':
        mean = np.mean(series)
        std = np.std(series)
        return (series - mean) / std
    elif method == 'minmax':
        min_val = np.min(series)
        max_val = np.max(series)
        return (series - min_val) / (max_val - min_val)
    elif method == 'robust':
        median = np.median(series)
        iqr = np.percentile(series, 75) - np.percentile(series, 25)
        return (series - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _calculate_autocorrelation(series: np.ndarray, lags: int) -> np.ndarray:
    """Calculate autocorrelation for the given series and lags."""
    n = len(series)
    mean = np.mean(series)
    autocorr = np.zeros(lags + 1)

    for lag in range(lags + 1):
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            numerator = np.sum((series[:n - lag] - mean) * (series[lag:] - mean))
            denominator = np.sum((series[:n - lag] - mean) ** 2)
            autocorr[lag] = numerator / denominator

    return autocorr

def _compute_ljung_box_statistic(autocorr: np.ndarray, lags: int) -> float:
    """Compute the Ljung-Box statistic."""
    n = len(autocorr) - 1
    Q = n * (n + 2) * np.sum((autocorr[1:lags + 1] / n) ** 2)
    return Q

def _calculate_p_value(statistic: float, lags: int) -> float:
    """Calculate the p-value for the Ljung-Box test."""
    from scipy.stats import chi2
    df = lags
    return 1 - chi2.cdf(statistic, df)

def _compute_metrics(
    autocorr: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute metrics based on user choice."""
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((autocorr[1:] - 0) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(autocorr[1:]))
        elif metric == 'r2':
            y_true = np.zeros_like(autocorr[1:])
            y_pred = autocorr[1:]
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        metrics['custom'] = metric(autocorr[1:])
    else:
        raise TypeError("Metric must be a string or callable.")

    if custom_metric is not None and callable(custom_metric):
        metrics['custom_additional'] = custom_metric(autocorr[1:])

    return metrics

def _check_warnings(series: np.ndarray, lags: int) -> Dict:
    """Check for potential warnings."""
    warnings = {}

    if len(series) < 2 * lags:
        warnings['short_series'] = "Input series may be too short for the specified number of lags."

    if np.any(np.isnan(series)):
        warnings['nan_values'] = "Input series contains NaN values."

    return warnings

################################################################################
# durbin_watson_statistic
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(residuals: np.ndarray) -> None:
    """Validate input residuals for Durbin-Watson statistic calculation."""
    if not isinstance(residuals, np.ndarray):
        raise TypeError("Residuals must be a numpy array")
    if residuals.ndim != 1:
        raise ValueError("Residuals must be a 1-dimensional array")
    if np.isnan(residuals).any() or np.isinf(residuals).any():
        raise ValueError("Residuals must not contain NaN or infinite values")

def _compute_durbin_watson(residuals: np.ndarray) -> float:
    """Compute the Durbin-Watson statistic from residuals."""
    n = len(residuals)
    if n < 2:
        raise ValueError("At least two residuals are required")

    diff_squared = np.sum(np.diff(residuals) ** 2)
    var_residuals = np.var(residuals, ddof=1)

    if var_residuals == 0:
        return 4.0

    dw_statistic = diff_squared / (2 * var_residuals)
    return dw_statistic

def durbin_watson_statistic_fit(
    residuals: np.ndarray,
    normalize: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute the Durbin-Watson statistic for testing autocorrelation in residuals.

    Parameters:
    -----------
    residuals : np.ndarray
        Array of residuals from a regression model.
    normalize : str, optional
        Normalization method to apply to residuals. Options: 'standard', 'minmax'.
    custom_metric : callable, optional
        Custom metric function that takes residuals and returns a float.

    Returns:
    --------
    dict
        Dictionary containing the Durbin-Watson statistic, metrics, parameters used,
        and any warnings.

    Example:
    --------
    >>> residuals = np.array([1.0, -0.5, 0.3, -0.2])
    >>> result = durbin_watson_statistic_fit(residuals)
    """
    # Validate inputs
    _validate_inputs(residuals)

    # Normalize residuals if specified
    normalized_residuals = _normalize_residuals(residuals, normalize)

    # Compute Durbin-Watson statistic
    dw_statistic = _compute_durbin_watson(normalized_residuals)

    # Compute additional metrics if custom metric is provided
    metrics = {}
    if custom_metric is not None:
        try:
            metrics['custom_metric'] = custom_metric(normalized_residuals)
        except Exception as e:
            metrics['custom_metric'] = f"Error: {str(e)}"

    # Prepare output
    result = {
        "result": dw_statistic,
        "metrics": metrics,
        "params_used": {
            "normalize": normalize,
            "custom_metric": custom_metric.__name__ if custom_metric else None
        },
        "warnings": []
    }

    return result

def _normalize_residuals(
    residuals: np.ndarray,
    method: Optional[str] = None
) -> np.ndarray:
    """
    Normalize residuals using the specified method.

    Parameters:
    -----------
    residuals : np.ndarray
        Array of residuals to normalize.
    method : str, optional
        Normalization method. Options: 'standard', 'minmax'.

    Returns:
    --------
    np.ndarray
        Normalized residuals.
    """
    if method is None:
        return residuals

    if method == 'standard':
        mean = np.mean(residuals)
        std = np.std(residuals)
        if std == 0:
            return residuals - mean
        return (residuals - mean) / std

    elif method == 'minmax':
        min_val = np.min(residuals)
        max_val = np.max(residuals)
        if min_val == max_val:
            return residuals - min_val
        return (residuals - min_val) / (max_val - min_val)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

################################################################################
# yule_walker_estimates
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for Yule-Walker estimates."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values")

def _compute_autocorrelation(data: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation function up to specified maximum lag."""
    n = len(data)
    mean = np.mean(data)
    autocorr = np.zeros(max_lag + 1)

    for lag in range(max_lag + 1):
        autocorr[lag] = np.sum((data[:n-lag] - mean) * (data[lag:] - mean)) / n

    return autocorr

def _solve_yule_walker(r: np.ndarray, order: int,
                       solver: str = 'levinson',
                       tol: float = 1e-6) -> np.ndarray:
    """Solve Yule-Walker equations for AR coefficients."""
    if solver == 'levinson':
        return _levinson_recursion(r, order, tol)
    elif solver == 'closed_form':
        return _closed_form_solution(r, order)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _levinson_recursion(r: np.ndarray, order: int, tol: float) -> np.ndarray:
    """Levinson recursion algorithm for solving Yule-Walker equations."""
    n = len(r)
    a = np.zeros(order + 1)
    e = r[0]

    for i in range(1, order + 1):
        sum_val = np.sum(a[:i] * r[i:i-1:-1])
        a[i] = (r[i] - sum_val) / e
        if np.abs(a[i]) < tol:
            a = a[:i]
            break

        e_new = e * (1 - np.abs(a[i])**2)
        if e_new <= tol:
            break

        a[:i] -= a[i] * r[i-1:i-1:-1]
        e = e_new

    return a[1:]

def _closed_form_solution(r: np.ndarray, order: int) -> np.ndarray:
    """Closed form solution for Yule-Walker equations."""
    R = np.zeros((order, order))
    r_vec = r[1:order+1]

    for i in range(order):
        R[i, :] = np.array([r[abs(i - j)] for j in range(order)])

    a = np.linalg.solve(R, -r_vec)
    return np.concatenate([[1], a])

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     metric: Union[str, Callable]) -> Dict[str, float]:
    """Compute specified metrics between true and predicted values."""
    metrics = {}

    if metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred)**2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    elif callable(metric):
        metrics['custom'] = metric(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

def yule_walker_estimates_fit(data: np.ndarray,
                             order: int,
                             max_lag: Optional[int] = None,
                             solver: str = 'levinson',
                             metric: Union[str, Callable] = 'mse',
                             tol: float = 1e-6) -> Dict:
    """
    Estimate AR coefficients using Yule-Walker method.

    Parameters
    ----------
    data : np.ndarray
        Input time series data.
    order : int
        Order of the AR model.
    max_lag : Optional[int], default=None
        Maximum lag for autocorrelation computation. If None, set to order.
    solver : str, default='levinson'
        Solver method ('levinson' or 'closed_form').
    metric : Union[str, Callable], default='mse'
        Metric for evaluation ('mse', 'mae', 'r2') or custom callable.
    tol : float, default=1e-6
        Tolerance for solver convergence.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used and warnings.

    Example
    -------
    >>> data = np.random.randn(100)
    >>> result = yule_walker_estimates_fit(data, order=2)
    """
    # Validate input
    _validate_input(data)

    if max_lag is None:
        max_lag = order

    # Compute autocorrelation
    r = _compute_autocorrelation(data, max_lag)

    # Solve Yule-Walker equations
    coefficients = _solve_yule_walker(r, order, solver, tol)

    # Compute predictions (simple AR model)
    n = len(data)
    predictions = np.zeros(n)
    for t in range(order, n):
        predictions[t] = np.sum(coefficients * data[t-order:t])

    # Compute metrics
    metrics = _compute_metrics(data[order:], predictions[order:], metric)

    return {
        'result': {
            'coefficients': coefficients,
            'predictions': predictions
        },
        'metrics': metrics,
        'params_used': {
            'order': order,
            'max_lag': max_lag,
            'solver': solver,
            'metric': metric if not callable(metric) else 'custom',
            'tol': tol
        },
        'warnings': []
    }

################################################################################
# burt_test
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(series: np.ndarray) -> None:
    """Validate input series for Burt test."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input series must be a numpy array")
    if len(series.shape) != 1:
        raise ValueError("Input series must be one-dimensional")
    if np.isnan(series).any() or np.isinf(series).any():
        raise ValueError("Input series contains NaN or infinite values")

def _compute_autocorrelation(series: np.ndarray, lag: int) -> float:
    """Compute autocorrelation for a given lag."""
    mean = np.mean(series)
    covariance = np.sum((series[:-lag] - mean) * (series[lag:] - mean))
    variance = np.sum((series - mean) ** 2)
    return covariance / variance

def _burt_statistic(series: np.ndarray, lag: int) -> float:
    """Compute Burt statistic for autocorrelation test."""
    n = len(series)
    r = _compute_autocorrelation(series, lag)
    return (n - lag) * r ** 2

def _p_value(burt_stat: float, lag: int) -> float:
    """Compute p-value for Burt statistic."""
    from scipy.stats import chi2
    df = 1
    return 1 - chi2.cdf(burt_stat, df)

def burt_test_fit(
    series: np.ndarray,
    lag: int = 1,
    normalization: str = "standard",
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform Burt test for autocorrelation.

    Parameters:
    -----------
    series : np.ndarray
        Input time series data.
    lag : int, optional
        Lag for autocorrelation calculation (default: 1).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : Callable, optional
        Custom metric function (must take two arrays as input).

    Returns:
    --------
    dict
        Dictionary containing test results, metrics, parameters used and warnings.
    """
    _validate_inputs(series)

    # Normalization
    if normalization == "standard":
        series = (series - np.mean(series)) / np.std(series)
    elif normalization == "minmax":
        series = (series - np.min(series)) / (np.max(series) - np.min(series))
    elif normalization == "robust":
        median = np.median(series)
        iqr = np.percentile(series, 75) - np.percentile(series, 25)
        series = (series - median) / iqr

    # Compute Burt statistic
    burt_stat = _burt_statistic(series, lag)
    p_value_result = _p_value(burt_stat, lag)

    # Compute metrics
    metrics = {}
    if metric is not None:
        # Example: compare with some reference (placeholder)
        metrics["custom_metric"] = metric(series, series)

    # Prepare output
    result = {
        "burt_statistic": burt_stat,
        "p_value": p_value_result
    }

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "lag": lag,
            "normalization": normalization
        },
        "warnings": []
    }

################################################################################
# autocorrelation_function
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    series: np.ndarray,
    max_lag: int = 10,
    normalize: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Validate and preprocess input series for autocorrelation computation.

    Parameters
    ----------
    series : np.ndarray
        Input time series data.
    max_lag : int, optional
        Maximum lag to compute autocorrelation for (default: 10).
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').

    Returns
    -------
    dict
        Dictionary containing validated and processed data.

    Raises
    ------
    ValueError
        If input validation fails.
    """
    if not isinstance(series, np.ndarray):
        raise ValueError("Input series must be a numpy array")

    if len(series.shape) != 1:
        raise ValueError("Input series must be one-dimensional")

    if np.any(np.isnan(series)):
        raise ValueError("Input series contains NaN values")

    if np.any(np.isinf(series)):
        raise ValueError("Input series contains infinite values")

    if max_lag < 1:
        raise ValueError("max_lag must be at least 1")

    if normalize not in [None, 'none', 'standard', 'minmax', 'robust']:
        raise ValueError("Invalid normalization method")

    processed = {'series': series.copy(), 'max_lag': max_lag}

    if normalize == 'standard':
        mean = np.mean(series)
        std = np.std(series)
        processed['normalized'] = (series - mean) / std
    elif normalize == 'minmax':
        min_val = np.min(series)
        max_val = np.max(series)
        processed['normalized'] = (series - min_val) / (max_val - min_val)
    elif normalize == 'robust':
        median = np.median(series)
        iqr = np.percentile(series, 75) - np.percentile(series, 25)
        processed['normalized'] = (series - median) / iqr

    return processed

def _compute_autocorrelation(
    series: np.ndarray,
    max_lag: int
) -> Dict[str, np.ndarray]:
    """
    Compute autocorrelation for given series and lags.

    Parameters
    ----------
    series : np.ndarray
        Input time series data.
    max_lag : int
        Maximum lag to compute autocorrelation for.

    Returns
    -------
    dict
        Dictionary containing computed autocorrelations and lags.
    """
    n = len(series)
    mean = np.mean(series)

    autocorr = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr[lag] = np.sum((series - mean) ** 2) / n
        else:
            cov = np.sum((series[:-lag] - mean) * (series[lag:] - mean))
            autocorr[lag] = cov / np.sum((series - mean) ** 2)

    return {
        'lags': np.arange(max_lag + 1),
        'autocorrelation': autocorr
    }

def _compute_metrics(
    autocorr: np.ndarray,
    metric_funcs: Dict[str, Callable]
) -> Dict[str, float]:
    """
    Compute metrics for autocorrelation results.

    Parameters
    ----------
    autocorr : np.ndarray
        Autocorrelation values.
    metric_funcs : dict
        Dictionary of metric functions to compute.

    Returns
    -------
    dict
        Dictionary containing computed metrics.
    """
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(autocorr)
        except Exception as e:
            metrics[f"{name}_error"] = str(e)

    return metrics

def autocorrelation_function_fit(
    series: np.ndarray,
    max_lag: int = 10,
    normalize: Optional[str] = None,
    metric_funcs: Optional[Dict[str, Callable]] = None
) -> Dict:
    """
    Compute autocorrelation function for a time series.

    Parameters
    ----------
    series : np.ndarray
        Input time series data.
    max_lag : int, optional
        Maximum lag to compute autocorrelation for (default: 10).
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric_funcs : dict, optional
        Dictionary of metric functions to compute on results.

    Returns
    -------
    dict
        Structured result containing:
        - 'result': Computed autocorrelation values and lags
        - 'metrics': Computed metrics (if provided)
        - 'params_used': Parameters used in computation
        - 'warnings': Any warnings generated

    Examples
    --------
    >>> series = np.random.randn(100)
    >>> result = autocorrelation_function_fit(series, max_lag=5)
    """
    # Validate inputs
    processed = _validate_inputs(series, max_lag, normalize)
    series_to_use = processed['normalized'] if 'normalized' in processed else processed['series']

    # Compute autocorrelation
    acf_result = _compute_autocorrelation(series_to_use, processed['max_lag'])

    # Compute metrics if provided
    metrics = {}
    warnings = []
    if metric_funcs is not None:
        try:
            metrics = _compute_metrics(acf_result['autocorrelation'], metric_funcs)
        except Exception as e:
            warnings.append(f"Metric computation failed: {str(e)}")

    return {
        'result': acf_result,
        'metrics': metrics,
        'params_used': {
            'max_lag': processed['max_lag'],
            'normalize': normalize,
            'original_series_length': len(series)
        },
        'warnings': warnings
    }

################################################################################
# partial_autocorrelation_function
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(data: np.ndarray) -> None:
    """Validate input data for partial autocorrelation function.

    Args:
        data: Input time series data as numpy array.

    Raises:
        ValueError: If input is invalid.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array")
    if len(data.shape) != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values")

def standardize_data(data: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Standardize data using specified normalization method.

    Args:
        data: Input time series data.
        method: Normalization method ('none', 'standard', 'minmax', 'robust').

    Returns:
        Standardized data.
    """
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

def compute_autocorrelation(data: np.ndarray, lag: int) -> float:
    """Compute autocorrelation for a given lag.

    Args:
        data: Input time series data.
        lag: Lag value to compute autocorrelation for.

    Returns:
        Autocorrelation coefficient.
    """
    n = len(data)
    mean = np.mean(data)
    numerator = np.sum((data[:n-lag] - mean) * (data[lag:] - mean))
    denominator = np.sum((data[:n-lag] - mean) ** 2)
    return numerator / denominator

def partial_autocorrelation_function_fit(
    data: np.ndarray,
    max_lag: int = 10,
    normalization: str = 'standard',
    solver: str = 'closed_form',
    metric: Union[str, Callable] = 'mse',
    **kwargs
) -> Dict:
    """Compute partial autocorrelation function for time series data.

    Args:
        data: Input time series data.
        max_lag: Maximum lag to compute.
        normalization: Normalization method for data.
        solver: Solver method ('closed_form', 'gradient_descent').
        metric: Metric to use for evaluation ('mse', 'mae', custom callable).
        **kwargs: Additional solver-specific parameters.

    Returns:
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate input
    validate_input(data)

    # Standardize data
    standardized_data = standardize_data(data, normalization)
    n = len(standardized_data)

    # Initialize results
    pacf_values = np.zeros(max_lag)
    metrics = {}
    warnings = []

    # Compute partial autocorrelation for each lag
    for lag in range(1, max_lag + 1):
        # Prepare data matrices
        X = np.column_stack([standardized_data[i:-(lag-i)] for i in range(1, lag)])
        y = standardized_data[lag:]

        # Solve for partial autocorrelation
        if solver == 'closed_form':
            pacf = solve_closed_form(X, y)
        elif solver == 'gradient_descent':
            pacf = solve_gradient_descent(X, y, **kwargs)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        pacf_values[lag-1] = pacf

        # Compute metric
        if isinstance(metric, str):
            metrics[f'lag_{lag}'] = compute_metric(y, predict(X, pacf), metric)
        else:
            metrics[f'lag_{lag}'] = metric(y, predict(X, pacf))

    return {
        'result': {'pacf_values': pacf_values},
        'metrics': metrics,
        'params_used': {
            'max_lag': max_lag,
            'normalization': normalization,
            'solver': solver,
            'metric': metric
        },
        'warnings': warnings
    }

def solve_closed_form(X: np.ndarray, y: np.ndarray) -> float:
    """Solve for partial autocorrelation using closed form solution.

    Args:
        X: Design matrix.
        y: Target vector.

    Returns:
        Partial autocorrelation coefficient.
    """
    return np.linalg.inv(X.T @ X) @ X.T @ y

def solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> float:
    """Solve for partial autocorrelation using gradient descent.

    Args:
        X: Design matrix.
        y: Target vector.
        learning_rate: Learning rate for gradient descent.
        max_iter: Maximum number of iterations.
        tol: Tolerance for convergence.

    Returns:
        Partial autocorrelation coefficient.
    """
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ beta - y) / n_samples
        new_beta = beta - learning_rate * gradient

        if np.linalg.norm(new_beta - beta) < tol:
            break
        beta = new_beta

    return beta[-1]

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Compute specified metric between true and predicted values.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        metric: Metric to compute ('mse', 'mae').

    Returns:
        Computed metric value.
    """
    if metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def predict(X: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Make predictions using given coefficients.

    Args:
        X: Design matrix.
        coefficients: Model coefficients.

    Returns:
        Predicted values.
    """
    return X @ coefficients
