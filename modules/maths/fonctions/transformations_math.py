"""
Quantix – Module transformations_math
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# normalisation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for normalization."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _standard_normalization(data: np.ndarray) -> np.ndarray:
    """Apply standard normalization (z-score)."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-8)

def _minmax_normalization(data: np.ndarray) -> np.ndarray:
    """Apply min-max normalization."""
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val + 1e-8)

def _robust_normalization(data: np.ndarray) -> np.ndarray:
    """Apply robust normalization using median and IQR."""
    median = np.median(data, axis=0)
    q75, q25 = np.percentile(data, [75, 25], axis=0)
    iqr = q75 - q25
    return (data - median) / (iqr + 1e-8)

def _compute_metrics(
    original: np.ndarray,
    normalized: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Compute metrics between original and normalized data."""
    return {"metric": metric_func(original, normalized)}

def normalisation_fit(
    data: np.ndarray,
    method: str = "standard",
    metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Apply normalization to the input data.

    Parameters:
    - data: Input 2D numpy array
    - method: Normalization method ("none", "standard", "minmax", "robust")
    - metric_func: Optional callable to compute metrics
    - **kwargs: Additional parameters for the normalization method

    Returns:
    - Dictionary containing "result", "metrics", "params_used", and "warnings"
    """
    _validate_inputs(data)

    # Choose normalization method
    if method == "standard":
        normalized_data = _standard_normalization(data)
    elif method == "minmax":
        normalized_data = _minmax_normalization(data)
    elif method == "robust":
        normalized_data = _robust_normalization(data)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Compute metrics if provided
    metrics = {}
    if metric_func is not None:
        metrics = _compute_metrics(data, normalized_data, metric_func)

    return {
        "result": normalized_data,
        "metrics": metrics,
        "params_used": {"method": method},
        "warnings": []
    }

# Example usage:
"""
data = np.random.rand(10, 5)
result = normalisation_fit(data, method="standard")
print(result["result"])
"""

################################################################################
# standardisation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for standardisation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _standardize_none(data: np.ndarray) -> np.ndarray:
    """No standardisation applied."""
    return data

def _standardize_standard(data: np.ndarray) -> np.ndarray:
    """Standardise data using mean and standard deviation."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

def _standardize_minmax(data: np.ndarray) -> np.ndarray:
    """Standardise data using min-max scaling."""
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val)

def _standardize_robust(data: np.ndarray) -> np.ndarray:
    """Standardise data using median and IQR."""
    median = np.median(data, axis=0)
    q75, q25 = np.percentile(data, [75, 25], axis=0)
    iqr = q75 - q25
    return (data - median) / iqr

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable] = None
) -> float:
    """Compute the specified metric between true and predicted values."""
    if custom_metric is not None:
        return custom_metric(y_true, y_pred)

    if metric == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def standardisation_fit(
    data: np.ndarray,
    method: str = "standard",
    metric: Optional[str] = None,
    custom_metric: Optional[Callable] = None,
    y_true: Optional[np.ndarray] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Standardise data using the specified method.

    Parameters:
    -----------
    data : np.ndarray
        Input data to be standardised.
    method : str, optional (default="standard")
        Standardisation method: "none", "standard", "minmax", or "robust".
    metric : str, optional
        Metric to compute if y_true is provided: "mse", "mae", or "r2".
    custom_metric : Callable, optional
        Custom metric function if a predefined one is not used.
    y_true : np.ndarray, optional
        True values for metric computation.

    Returns:
    --------
    dict
        Dictionary containing the standardised data, metrics, parameters used,
        and any warnings.
    """
    _validate_input(data)

    # Standardise data
    if method == "none":
        standardised_data = _standardize_none(data)
    elif method == "standard":
        standardised_data = _standardize_standard(data)
    elif method == "minmax":
        standardised_data = _standardize_minmax(data)
    elif method == "robust":
        standardised_data = _standardize_robust(data)
    else:
        raise ValueError(f"Unknown standardisation method: {method}")

    # Compute metrics if y_true is provided
    metrics = {}
    warnings = []
    if y_true is not None:
        _validate_input(y_true)
        if data.shape[0] != y_true.shape[0]:
            raise ValueError("Data and y_true must have the same number of samples")
        if metric is not None:
            metrics["metric"] = _compute_metric(y_true, standardised_data, metric, custom_metric)
        elif custom_metric is not None:
            metrics["metric"] = _compute_metric(y_true, standardised_data, "custom", custom_metric)
        else:
            warnings.append("No metric specified. Metrics computation skipped.")

    # Prepare output
    result = {
        "result": standardised_data,
        "metrics": metrics,
        "params_used": {
            "method": method,
            "metric": metric,
            "custom_metric": custom_metric is not None
        },
        "warnings": warnings
    }

    return result

# Example usage:
# data = np.random.randn(100, 5)
# result = standardisation_fit(data, method="standard", metric="mse")

################################################################################
# log_transform
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    normalize: str = "none",
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """
    Validate and preprocess input data.

    Parameters
    ----------
    X : np.ndarray
        Input data to validate and preprocess.
    normalize : str, optional
        Normalization method (default is "none").
    custom_normalize : callable, optional
        Custom normalization function.

    Returns
    -------
    np.ndarray
        Validated and preprocessed data.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input X contains NaN or infinite values.")

    if normalize == "none":
        return X
    elif normalize == "standard":
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalize == "minmax":
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalize == "robust":
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    elif callable(custom_normalize):
        return custom_normalize(X)
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")

def _compute_log_transform(
    X: np.ndarray,
    base: float = 10.0
) -> np.ndarray:
    """
    Compute the log transform of input data.

    Parameters
    ----------
    X : np.ndarray
        Input data to transform.
    base : float, optional
        Base of the logarithm (default is 10.0).

    Returns
    -------
    np.ndarray
        Log-transformed data.
    """
    if base <= 0:
        raise ValueError("Logarithm base must be positive.")
    if np.any(X <= 0):
        raise ValueError("Logarithm is only defined for positive values.")

    return np.log(X) / np.log(base)

def _compute_metrics(
    X: np.ndarray,
    X_transformed: np.ndarray,
    metric: str = "mse",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """
    Compute metrics between original and transformed data.

    Parameters
    ----------
    X : np.ndarray
        Original input data.
    X_transformed : np.ndarray
        Transformed data.
    metric : str, optional
        Metric to compute (default is "mse").
    custom_metric : callable, optional
        Custom metric function.

    Returns
    -------
    Dict[str, float]
        Dictionary of computed metrics.
    """
    metrics = {}

    if callable(custom_metric):
        metrics["custom"] = custom_metric(X, X_transformed)
    else:
        if metric == "mse":
            metrics["mse"] = np.mean((X - X_transformed) ** 2)
        elif metric == "mae":
            metrics["mae"] = np.mean(np.abs(X - X_transformed))
        elif metric == "r2":
            ss_res = np.sum((X - X_transformed) ** 2)
            ss_tot = np.sum((X - np.mean(X)) ** 2)
            metrics["r2"] = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        elif metric == "logloss":
            metrics["logloss"] = -np.mean(X * np.log(X_transformed + 1e-10))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return metrics

def log_transform_fit(
    X: np.ndarray,
    base: float = 10.0,
    normalize: str = "none",
    metric: str = "mse",
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Fit and apply log transform to input data.

    Parameters
    ----------
    X : np.ndarray
        Input data to transform.
    base : float, optional
        Base of the logarithm (default is 10.0).
    normalize : str, optional
        Normalization method (default is "none").
    metric : str, optional
        Metric to compute (default is "mse").
    custom_normalize : callable, optional
        Custom normalization function.
    custom_metric : callable, optional
        Custom metric function.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]
        Dictionary containing:
        - "result": Transformed data.
        - "metrics": Computed metrics.
        - "params_used": Parameters used in the transformation.
        - "warnings": Any warnings generated during processing.
    """
    result = {}
    warnings = []

    # Validate and preprocess input data
    X_validated = _validate_inputs(X, normalize, custom_normalize)

    # Compute log transform
    try:
        X_transformed = _compute_log_transform(X_validated, base)
    except ValueError as e:
        warnings.append(str(e))
        X_transformed = np.zeros_like(X_validated)  # Fallback to zeros

    # Compute metrics
    metrics = _compute_metrics(X_validated, X_transformed, metric, custom_metric)

    # Prepare output
    result["result"] = X_transformed
    result["metrics"] = metrics
    result["params_used"] = {
        "base": base,
        "normalize": normalize,
        "metric": metric
    }
    result["warnings"] = warnings

    return result

# Example usage:
# X = np.array([[1, 2], [3, 4]])
# result = log_transform_fit(X, base=10, normalize="standard", metric="mse")

################################################################################
# sqrt_transform
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for sqrt transform."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array")
    if np.any(np.isnan(data)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input contains infinite values")

def _apply_normalization(
    data: np.ndarray,
    normalization: str = "none",
    custom_norm: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Apply normalization to the data."""
    if custom_norm is not None:
        return custom_norm(data)

    if normalization == "standard":
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-8)
    elif normalization == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif normalization == "robust":
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / (iqr + 1e-8)
    elif normalization == "none":
        return data
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _sqrt_transform(
    data: np.ndarray,
    lambda_: float = 1.0
) -> np.ndarray:
    """Apply square root transform to the data."""
    return lambda_ * np.sqrt(data)

def _inverse_sqrt_transform(
    transformed_data: np.ndarray,
    lambda_: float = 1.0
) -> np.ndarray:
    """Apply inverse square root transform to the data."""
    return (transformed_data / lambda_) ** 2

def _compute_metrics(
    original_data: np.ndarray,
    transformed_data: np.ndarray,
    metric: str = "mse",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute metrics between original and transformed data."""
    if custom_metric is not None:
        return {"custom": custom_metric(original_data, transformed_data)}

    metrics = {}
    if metric == "mse" or "all" in metric:
        mse = np.mean((original_data - transformed_data) ** 2)
        metrics["mse"] = mse
    if metric == "mae" or "all" in metric:
        mae = np.mean(np.abs(original_data - transformed_data))
        metrics["mae"] = mae
    if metric == "r2" or "all" in metric:
        ss_res = np.sum((original_data - transformed_data) ** 2)
        ss_tot = np.sum((original_data - np.mean(original_data)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        metrics["r2"] = r2
    return metrics

def sqrt_transform_fit(
    data: np.ndarray,
    normalization: str = "none",
    metric: str = "mse",
    lambda_: float = 1.0,
    custom_norm: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, float], list]]:
    """
    Apply square root transform to the data.

    Parameters
    ----------
    data : np.ndarray
        Input data to be transformed.
    normalization : str, optional
        Normalization method to apply before transformation. Options: "none", "standard", "minmax", "robust".
    metric : str, optional
        Metric to compute between original and transformed data. Options: "mse", "mae", "r2", "all".
    lambda_ : float, optional
        Scaling factor for the square root transform.
    custom_norm : Callable[[np.ndarray], np.ndarray], optional
        Custom normalization function.
    custom_metric : Callable[[np.ndarray, np.ndarray], float], optional
        Custom metric function.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, float], list]]
        Dictionary containing the transformed data, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.array([1, 4, 9, 16])
    >>> result = sqrt_transform_fit(data)
    """
    _validate_input(data)

    normalized_data = _apply_normalization(data, normalization, custom_norm)
    transformed_data = _sqrt_transform(normalized_data, lambda_)
    metrics = _compute_metrics(data, transformed_data, metric, custom_metric)

    return {
        "result": transformed_data,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "lambda_": lambda_
        },
        "warnings": []
    }

################################################################################
# boxcox_transform
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(
    data: np.ndarray,
    normalize: str = "none",
    metric: Union[str, Callable] = "mse",
    solver: str = "closed_form"
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

    normalize_options = ["none", "standard", "minmax", "robust"]
    if normalize not in normalize_options:
        raise ValueError(f"normalize must be one of {normalize_options}")

    metric_options = ["mse", "mae", "r2", "logloss"]
    if isinstance(metric, str) and metric not in metric_options:
        raise ValueError(f"metric must be one of {metric_options} or a callable")

    solver_options = ["closed_form", "gradient_descent", "newton"]
    if solver not in solver_options:
        raise ValueError(f"solver must be one of {solver_options}")

def _normalize_data(
    data: np.ndarray,
    normalize: str = "none"
) -> np.ndarray:
    """Normalize the input data."""
    if normalize == "standard":
        return (data - np.mean(data)) / np.std(data)
    elif normalize == "minmax":
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif normalize == "robust":
        return (data - np.median(data)) / (np.percentile(data, 75) - np.percentile(data, 25))
    return data.copy()

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable] = "mse"
) -> float:
    """Compute the specified metric."""
    if isinstance(metric, str):
        if metric == "mse":
            return np.mean((y_true - y_pred) ** 2)
        elif metric == "mae":
            return np.mean(np.abs(y_true - y_pred))
        elif metric == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)
        elif metric == "logloss":
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        return metric(y_true, y_pred)

def _boxcox_closed_form(
    data: np.ndarray
) -> float:
    """Compute the Box-Cox transformation parameter using closed form solution."""
    log_data = np.log(data)
    n = len(data)
    sum_log = np.sum(log_data)
    sum_sq_log = np.sum(log_data ** 2)
    sum_x = np.sum(data)
    sum_sq_x = np.sum(data ** 2)

    numerator = n * sum_log - sum_x
    denominator = sum_sq_log - (sum_log ** 2) / n

    return numerator / denominator

def _boxcox_gradient_descent(
    data: np.ndarray,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> float:
    """Compute the Box-Cox transformation parameter using gradient descent."""
    lambda_val = 1.0
    prev_lambda = 0

    for _ in range(max_iter):
        transformed_data = (data ** lambda_val - 1) / lambda_val if lambda_val != 0 else np.log(data)
        log_likelihood = -np.sum(np.log(lambda_val * data ** (lambda_val - 1)))
        gradient = np.sum((transformed_data - np.mean(transformed_data)) ** 2) / (lambda_val * data ** lambda_val)

        prev_lambda = lambda_val
        lambda_val -= learning_rate * gradient

        if abs(lambda_val - prev_lambda) < tol:
            break

    return lambda_val

def _boxcox_newton(
    data: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> float:
    """Compute the Box-Cox transformation parameter using Newton's method."""
    lambda_val = 1.0
    prev_lambda = 0

    for _ in range(max_iter):
        transformed_data = (data ** lambda_val - 1) / lambda_val if lambda_val != 0 else np.log(data)
        log_likelihood = -np.sum(np.log(lambda_val * data ** (lambda_val - 1)))
        gradient = np.sum((transformed_data - np.mean(transformed_data)) ** 2) / (lambda_val * data ** lambda_val)
        hessian = np.sum((transformed_data - np.mean(transformed_data)) ** 2 * (1 - lambda_val) / (lambda_val ** 2))

        prev_lambda = lambda_val
        lambda_val -= gradient / hessian

        if abs(lambda_val - prev_lambda) < tol:
            break

    return lambda_val

def boxcox_transform_fit(
    data: np.ndarray,
    normalize: str = "none",
    metric: Union[str, Callable] = "mse",
    solver: str = "closed_form",
    **kwargs
) -> Dict[str, Any]:
    """
    Fit the Box-Cox transformation to the input data.

    Parameters:
    -----------
    data : np.ndarray
        Input data to transform.
    normalize : str, optional
        Normalization method (default: "none").
    metric : Union[str, Callable], optional
        Metric to evaluate the transformation (default: "mse").
    solver : str, optional
        Solver to use for parameter estimation (default: "closed_form").
    **kwargs :
        Additional solver-specific parameters.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the transformation result, metrics, parameters used, and warnings.
    """
    _validate_inputs(data, normalize, metric, solver)

    normalized_data = _normalize_data(data, normalize)
    positive_data = normalized_data[normalized_data > 0]

    if solver == "closed_form":
        lambda_val = _boxcox_closed_form(positive_data)
    elif solver == "gradient_descent":
        lambda_val = _boxcox_gradient_descent(positive_data, **kwargs)
    elif solver == "newton":
        lambda_val = _boxcox_newton(positive_data, **kwargs)

    transformed_data = (data ** lambda_val - 1) / lambda_val if lambda_val != 0 else np.log(data)
    metric_value = _compute_metric(normalized_data, transformed_data, metric)

    result = {
        "result": transformed_data,
        "metrics": {"value": metric_value, "name": str(metric)},
        "params_used": {
            "normalize": normalize,
            "metric": str(metric),
            "solver": solver,
            **kwargs
        },
        "warnings": []
    }

    if np.any(data <= 0):
        result["warnings"].append("Data contains non-positive values. Box-Cox transformation may not be appropriate.")

    return result

################################################################################
# yeo_johnson_transform
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(
    X: np.ndarray,
    normalize: str = "standard",
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """
    Validate and preprocess input data.

    Parameters
    ----------
    X : np.ndarray
        Input data to validate and normalize.
    normalize : str, optional
        Normalization method (default is "standard").
    custom_normalize : callable, optional
        Custom normalization function.

    Returns
    -------
    np.ndarray
        Validated and normalized data.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 1:
        raise ValueError("Input must be 1-dimensional")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")

    if custom_normalize is not None:
        return custom_normalize(X)

    if normalize == "standard":
        X = (X - np.mean(X)) / np.std(X)
    elif normalize == "minmax":
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
    elif normalize == "robust":
        X = (X - np.median(X)) / (np.percentile(X, 75) - np.percentile(X, 25))
    elif normalize != "none":
        raise ValueError(f"Unknown normalization method: {normalize}")

    return X

def _compute_yeo_johnson(
    X: np.ndarray,
    lambda_param: float = 1.0
) -> np.ndarray:
    """
    Compute Yeo-Johnson transform for given lambda.

    Parameters
    ----------
    X : np.ndarray
        Input data to transform.
    lambda_param : float, optional
        Lambda parameter for the transformation (default is 1.0).

    Returns
    -------
    np.ndarray
        Transformed data.
    """
    X_transformed = np.zeros_like(X)
    mask_pos = X > 0
    mask_neg = X <= 0

    if lambda_param == 1:
        X_transformed[mask_pos] = np.log(X[mask_pos])
    else:
        X_transformed[mask_pos] = ((X[mask_pos]**lambda_param - 1) / lambda_param)

    if lambda_param == -1:
        X_transformed[mask_neg] = -np.log(-X[mask_neg])
    else:
        X_transformed[mask_neg] = -(((-X[mask_neg])**(-lambda_param) - 1) / (-lambda_param))

    return X_transformed

def _optimize_lambda(
    X: np.ndarray,
    metric: str = "mse",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    solver: str = "gradient_descent",
    tol: float = 1e-4,
    max_iter: int = 100
) -> float:
    """
    Optimize lambda parameter for Yeo-Johnson transform.

    Parameters
    ----------
    X : np.ndarray
        Input data to optimize lambda for.
    metric : str, optional
        Metric to optimize (default is "mse").
    custom_metric : callable, optional
        Custom metric function.
    solver : str, optional
        Solver method (default is "gradient_descent").
    tol : float, optional
        Tolerance for convergence (default is 1e-4).
    max_iter : int, optional
        Maximum number of iterations (default is 100).

    Returns
    -------
    float
        Optimized lambda parameter.
    """
    def _mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def _mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def _r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    metrics = {
        "mse": _mse,
        "mae": _mae,
        "r2": _r2
    }

    if custom_metric is not None:
        metric_func = custom_metric
    else:
        try:
            metric_func = metrics[metric]
        except KeyError:
            raise ValueError(f"Unknown metric: {metric}")

    def _gradient_descent(X, metric_func, tol, max_iter):
        lambda_param = 0.5
        prev_metric = float('inf')
        for _ in range(max_iter):
            X_transformed = _compute_yeo_johnson(X, lambda_param)
            current_metric = metric_func(X, X_transformed)

            if abs(prev_metric - current_metric) < tol:
                break

            prev_metric = current_metric
            lambda_param += 0.1 * (np.random.rand() - 0.5)  # Simple random step

        return lambda_param

    def _newton(X, metric_func, tol, max_iter):
        # Simplified Newton method
        lambda_param = 0.5
        for _ in range(max_iter):
            X_transformed = _compute_yeo_johnson(X, lambda_param)
            current_metric = metric_func(X, X_transformed)

            if abs(current_metric) < tol:
                break

            lambda_param -= 0.1 * current_metric  # Simplified update

        return lambda_param

    solvers = {
        "gradient_descent": _gradient_descent,
        "newton": _newton
    }

    try:
        solver_func = solvers[solver]
    except KeyError:
        raise ValueError(f"Unknown solver: {solver}")

    return solver_func(X, metric_func, tol, max_iter)

def yeo_johnson_transform_fit(
    X: np.ndarray,
    normalize: str = "standard",
    metric: str = "mse",
    solver: str = "gradient_descent",
    tol: float = 1e-4,
    max_iter: int = 100,
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, float, Dict[str, float], str]]:
    """
    Fit Yeo-Johnson transform to data.

    Parameters
    ----------
    X : np.ndarray
        Input data to fit the transform.
    normalize : str, optional
        Normalization method (default is "standard").
    metric : str, optional
        Metric to optimize (default is "mse").
    solver : str, optional
        Solver method (default is "gradient_descent").
    tol : float, optional
        Tolerance for convergence (default is 1e-4).
    max_iter : int, optional
        Maximum number of iterations (default is 100).
    custom_normalize : callable, optional
        Custom normalization function.
    custom_metric : callable, optional
        Custom metric function.

    Returns
    -------
    dict
        Dictionary containing:
        - "result": Transformed data
        - "metrics": Dictionary of computed metrics
        - "params_used": Dictionary of parameters used
        - "warnings": List of warnings (if any)
    """
    # Validate and normalize input
    X_normalized = _validate_input(X, normalize, custom_normalize)

    # Optimize lambda parameter
    lambda_param = _optimize_lambda(
        X_normalized,
        metric=metric,
        custom_metric=custom_metric,
        solver=solver,
        tol=tol,
        max_iter=max_iter
    )

    # Compute transform with optimized lambda
    X_transformed = _compute_yeo_johnson(X_normalized, lambda_param)

    # Compute metrics
    def _mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def _mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def _r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    metrics = {
        "mse": _mse(X, X_transformed),
        "mae": _mae(X, X_transformed),
        "r2": _r2(X, X_transformed)
    }

    # Prepare output
    result = {
        "result": X_transformed,
        "metrics": metrics,
        "params_used": {
            "normalize": normalize,
            "metric": metric,
            "solver": solver,
            "tol": tol,
            "max_iter": max_iter,
            "lambda_param": lambda_param
        },
        "warnings": []
    }

    return result

# Example usage:
# result = yeo_johnson_transform_fit(X=np.array([1, 2, 3, 4, 5]))

################################################################################
# minmax_scaling
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for minmax scaling."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2-dimensional array.")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values.")

def _minmax_transform(
    data: np.ndarray,
    feature_range: tuple = (0, 1),
    axis: int = 0
) -> np.ndarray:
    """Apply min-max transformation to the data."""
    min_vals = np.min(data, axis=axis, keepdims=True)
    max_vals = np.max(data, axis=axis, keepdims=True)

    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = np.where(max_vals == min_vals, 0, (data - min_vals) / (max_vals - min_vals))
        scaled_data = scale * (feature_range[1] - feature_range[0]) + feature_range[0]

    return scaled_data

def _compute_metrics(
    original: np.ndarray,
    transformed: np.ndarray,
    metric_funcs: Dict[str, Callable]
) -> Dict[str, float]:
    """Compute metrics for the transformed data."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(original, transformed)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def minmax_scaling_fit(
    data: np.ndarray,
    feature_range: tuple = (0, 1),
    axis: int = 0,
    metric_funcs: Optional[Dict[str, Callable]] = None
) -> Dict:
    """
    Apply min-max scaling to the input data.

    Parameters:
    -----------
    data : np.ndarray
        Input data to be scaled.
    feature_range : tuple, optional
        Desired range of transformed data (default is (0, 1)).
    axis : int, optional
        Axis along which to compute the min and max (default is 0).
    metric_funcs : Dict[str, Callable], optional
        Dictionary of metric functions to compute (default is None).

    Returns:
    --------
    Dict
        Dictionary containing the transformed data, metrics, parameters used, and warnings.
    """
    _validate_input(data)

    if metric_funcs is None:
        metric_funcs = {}

    transformed_data = _minmax_transform(data, feature_range, axis)
    metrics = _compute_metrics(data, transformed_data, metric_funcs)

    result = {
        "result": transformed_data,
        "metrics": metrics,
        "params_used": {
            "feature_range": feature_range,
            "axis": axis
        },
        "warnings": []
    }

    return result

# Example metric functions
def mse(original: np.ndarray, transformed: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((original - transformed) ** 2)

def mae(original: np.ndarray, transformed: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(original - transformed))

# Example usage
if __name__ == "__main__":
    data = np.array([[1, 2], [3, 4], [5, 6]])
    metric_funcs = {
        "mse": mse,
        "mae": mae
    }
    result = minmax_scaling_fit(data, feature_range=(0, 1), metric_funcs=metric_funcs)
    print(result)

################################################################################
# robust_scaling
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def robust_scaling_fit(
    X: np.ndarray,
    normalization: str = 'robust',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit robust scaling transformation on input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    normalization : str, optional
        Type of normalization ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric to evaluate performance ('mse', 'mae', 'r2', 'logloss')
    distance : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics
    custom_distance : callable, optional
        Custom distance function if not using built-in distances

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = robust_scaling_fit(X, normalization='robust', metric='mse')
    """
    # Input validation
    _validate_inputs(X, normalization, metric, distance)

    # Initialize parameters
    params = {
        'normalization': normalization,
        'metric': metric,
        'distance': distance,
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Choose normalization method
    if normalization == 'none':
        X_normalized = X.copy()
    elif normalization == 'standard':
        X_normalized, center, scale = _standard_scaling(X)
    elif normalization == 'minmax':
        X_normalized, min_vals, max_vals = _minmax_scaling(X)
    elif normalization == 'robust':
        X_normalized, median_vals, iqr_vals = _robust_scaling(X)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    # Choose solver
    if solver == 'closed_form':
        coefficients = _closed_form_solver(X_normalized)
    elif solver == 'gradient_descent':
        coefficients = _gradient_descent_solver(X_normalized, tol, max_iter)
    elif solver == 'newton':
        coefficients = _newton_solver(X_normalized, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver method: {solver}")

    # Calculate metrics
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    if custom_metric is not None:
        metrics = {
            'custom': custom_metric(X_normalized, coefficients),
            metric: metric_func(X_normalized, coefficients)
        }
    else:
        metrics = {
            metric: metric_func(X_normalized, coefficients)
        }

    # Prepare output
    result = {
        'result': X_normalized,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

    return result

def _validate_inputs(
    X: np.ndarray,
    normalization: str,
    metric: Union[str, Callable],
    distance: Union[str, Callable]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input X must be 2-dimensional")
    if np.any(np.isnan(X)):
        raise ValueError("Input X contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input X contains infinite values")

    valid_normalizations = ['none', 'standard', 'minmax', 'robust']
    if normalization not in valid_normalizations:
        raise ValueError(f"Normalization must be one of {valid_normalizations}")

    valid_metrics = ['mse', 'mae', 'r2', 'logloss']
    if isinstance(metric, str) and metric not in valid_metrics:
        raise ValueError(f"Metric must be one of {valid_metrics} or a callable")

    valid_distances = ['euclidean', 'manhattan', 'cosine', 'minkowski']
    if isinstance(distance, str) and distance not in valid_distances:
        raise ValueError(f"Distance must be one of {valid_distances} or a callable")

def _standard_scaling(X: np.ndarray) -> tuple:
    """Standard scaling (z-score normalization)."""
    center = X.mean(axis=0)
    scale = X.std(axis=0)
    return (X - center) / scale, center, scale

def _minmax_scaling(X: np.ndarray) -> tuple:
    """Min-max scaling (normalization to [0,1] range)."""
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    return (X - min_vals) / (max_vals - min_vals), min_vals, max_vals

def _robust_scaling(X: np.ndarray) -> tuple:
    """Robust scaling using median and IQR."""
    median_vals = np.median(X, axis=0)
    iqr_vals = np.subtract(*np.percentile(X, [75, 25], axis=0))
    return (X - median_vals) / iqr_vals, median_vals, iqr_vals

def _closed_form_solver(X: np.ndarray) -> np.ndarray:
    """Closed form solution for coefficients."""
    return np.linalg.pinv(X.T @ X) @ X.T @ np.ones(X.shape[0])

def _gradient_descent_solver(
    X: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent solver for coefficients."""
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = -2 * (X @ coefficients - np.ones(n_samples)).sum()
        new_coefficients = coefficients - learning_rate * gradient

        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients

    return coefficients

def _newton_solver(
    X: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Newton's method solver for coefficients."""
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)

    for _ in range(max_iter):
        residual = X @ coefficients - np.ones(n_samples)
        gradient = -2 * X.T @ residual
        hessian = 2 * X.T @ X

        if np.linalg.cond(hessian) < 1e15:
            delta = np.linalg.solve(hessian, gradient)
        else:
            delta = np.linalg.pinv(hessian) @ gradient

        new_coefficients = coefficients - delta

        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients

    return coefficients

def _get_metric_function(metric: str) -> Callable:
    """Get the appropriate metric function based on input string."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared,
        'logloss': _log_loss
    }
    return metrics[metric]

def _mean_squared_error(X: np.ndarray, coefficients: np.ndarray) -> float:
    """Calculate mean squared error."""
    return np.mean((X @ coefficients - 1) ** 2)

def _mean_absolute_error(X: np.ndarray, coefficients: np.ndarray) -> float:
    """Calculate mean absolute error."""
    return np.mean(np.abs(X @ coefficients - 1))

def _r_squared(X: np.ndarray, coefficients: np.ndarray) -> float:
    """Calculate R-squared."""
    y_pred = X @ coefficients
    ss_res = np.sum((y_pred - 1) ** 2)
    ss_tot = np.sum((X.mean(axis=0) @ coefficients - 1) ** 2)
    return 1 - (ss_res / ss_tot)

def _log_loss(X: np.ndarray, coefficients: np.ndarray) -> float:
    """Calculate log loss."""
    y_pred = X @ coefficients
    return -np.mean(1 * np.log(y_pred) + (1-1) * np.log(1-y_pred))

################################################################################
# quantile_transform
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    n_quantiles: int = 100,
    output_distribution: str = 'normal',
    copy: bool = True
) -> np.ndarray:
    """
    Validate input data and parameters for quantile transform.

    Parameters
    ----------
    X : np.ndarray
        Input data to be transformed.
    n_quantiles : int, optional
        Number of quantiles to compute (default=100).
    output_distribution : str, optional
        Desired output distribution ('normal' or 'uniform', default='normal').
    copy : bool, optional
        Whether to make a copy of the input data (default=True).

    Returns
    -------
    np.ndarray
        Validated and potentially copied input data.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("Input X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D array.")
    if np.any(np.isnan(X)):
        raise ValueError("Input X contains NaN values.")
    if np.any(np.isinf(X)):
        raise ValueError("Input X contains infinite values.")

    if copy:
        X = X.copy()

    return X

def _compute_quantiles(
    X: np.ndarray,
    n_quantiles: int = 100
) -> np.ndarray:
    """
    Compute quantile values from input data.

    Parameters
    ----------
    X : np.ndarray
        Input data.
    n_quantiles : int, optional
        Number of quantiles to compute (default=100).

    Returns
    -------
    np.ndarray
        Computed quantile values.
    """
    return np.percentile(X, np.linspace(0, 100, n_quantiles), axis=0)

def _compute_output_values(
    X: np.ndarray,
    quantiles: np.ndarray,
    output_distribution: str = 'normal'
) -> np.ndarray:
    """
    Compute transformed values based on desired output distribution.

    Parameters
    ----------
    X : np.ndarray
        Input data.
    quantiles : np.ndarray
        Computed quantile values.
    output_distribution : str, optional
        Desired output distribution ('normal' or 'uniform', default='normal').

    Returns
    -------
    np.ndarray
        Transformed values.
    """
    if output_distribution == 'normal':
        # Standard normal distribution
        return np.interp(X, quantiles, np.linspace(-1, 1, len(quantiles)))
    elif output_distribution == 'uniform':
        # Uniform distribution
        return np.interp(X, quantiles, np.linspace(0, 1, len(quantiles)))
    else:
        raise ValueError("output_distribution must be 'normal' or 'uniform'")

def _compute_metrics(
    X: np.ndarray,
    transformed_X: np.ndarray,
    metric_func: Optional[Callable] = None
) -> Dict[str, float]:
    """
    Compute metrics for the transformation.

    Parameters
    ----------
    X : np.ndarray
        Original input data.
    transformed_X : np.ndarray
        Transformed data.
    metric_func : Optional[Callable], optional
        Custom metric function (default=None).

    Returns
    -------
    Dict[str, float]
        Dictionary of computed metrics.
    """
    metrics = {}

    if metric_func is not None:
        try:
            custom_metric = metric_func(X, transformed_X)
            metrics['custom'] = float(custom_metric)
        except Exception as e:
            metrics['warning_custom_metric'] = str(e)

    return metrics

def quantile_transform_fit(
    X: np.ndarray,
    n_quantiles: int = 100,
    output_distribution: str = 'normal',
    metric_func: Optional[Callable] = None,
    copy: bool = True
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Fit and apply quantile transform to input data.

    Parameters
    ----------
    X : np.ndarray
        Input data to be transformed.
    n_quantiles : int, optional
        Number of quantiles to compute (default=100).
    output_distribution : str, optional
        Desired output distribution ('normal' or 'uniform', default='normal').
    metric_func : Optional[Callable], optional
        Custom metric function (default=None).
    copy : bool, optional
        Whether to make a copy of the input data (default=True).

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]
        Dictionary containing:
        - 'result': Transformed data
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in the transformation
        - 'warnings': Any warnings encountered

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = quantile_transform_fit(X)
    """
    # Validate inputs
    X_validated = _validate_inputs(X, n_quantiles, output_distribution, copy)

    # Compute quantiles
    quantiles = _compute_quantiles(X_validated, n_quantiles)

    # Compute transformed values
    transformed_X = _compute_output_values(X_validated, quantiles, output_distribution)

    # Compute metrics
    metrics = _compute_metrics(X_validated, transformed_X, metric_func)

    # Prepare output
    result = {
        'result': transformed_X,
        'metrics': metrics,
        'params_used': {
            'n_quantiles': n_quantiles,
            'output_distribution': output_distribution
        },
        'warnings': {}
    }

    return result

################################################################################
# power_transform
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input contains infinite values")

def normalize_data(
    X: np.ndarray,
    method: str = "standard",
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data using specified method or custom function."""
    if custom_normalizer is not None:
        return custom_normalizer(X)

    X_norm = X.copy()
    if method == "standard":
        mean = np.mean(X)
        std = np.std(X)
        X_norm = (X - mean) / std
    elif method == "minmax":
        min_val = np.min(X)
        max_val = np.max(X)
        X_norm = (X - min_val) / (max_val - min_val)
    elif method == "robust":
        median = np.median(X)
        iqr = np.percentile(X, 75) - np.percentile(X, 25)
        X_norm = (X - median) / iqr
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_norm

def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "mse",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> float:
    """Compute specified metric between true and predicted values."""
    if custom_metric is not None:
        return custom_metric(y_true, y_pred)

    if metric == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    elif metric == "logloss":
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def power_transform_fit(
    X: np.ndarray,
    lambda_: float = 1.0,
    normalization: str = "standard",
    metric: str = "mse",
    solver: str = "closed_form",
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Fit power transform to data.

    Parameters
    ----------
    X : np.ndarray
        Input data (1D array)
    lambda_ : float, optional
        Power transform parameter (default=1.0)
    normalization : str or callable, optional
        Normalization method (default="standard")
    metric : str or callable, optional
        Metric to optimize (default="mse")
    solver : str, optional
        Solver method (default="closed_form")
    tol : float, optional
        Tolerance for convergence (default=1e-4)
    max_iter : int, optional
        Maximum number of iterations (default=1000)
    custom_normalizer : callable, optional
        Custom normalization function
    custom_metric : callable, optional
        Custom metric function

    Returns
    -------
    dict
        Dictionary containing:
        - "result": transformed data
        - "metrics": computed metrics
        - "params_used": parameters used
        - "warnings": any warnings

    Example
    -------
    >>> X = np.array([1, 2, 3, 4, 5])
    >>> result = power_transform_fit(X, lambda_=0.5)
    """
    # Validate input
    validate_input(X)

    # Normalize data
    X_norm = normalize_data(X, normalization, custom_normalizer)

    # Initialize parameters
    params_used = {
        "lambda_": lambda_,
        "normalization": normalization,
        "metric": metric,
        "solver": solver
    }

    # Apply power transform
    if lambda_ == 0:
        X_transformed = np.log(X_norm)
    elif lambda_ != 1:
        X_transformed = (X_norm ** lambda_) / lambda_
    else:
        X_transformed = X_norm

    # Compute metrics
    metrics = {
        "mse": compute_metric(X, X_transformed, "mse", custom_metric),
        "mae": compute_metric(X, X_transformed, "mae", custom_metric),
        "r2": compute_metric(X, X_transformed, "r2", custom_metric)
    }

    # Check for warnings
    warnings = []
    if np.any(np.isinf(X_transformed)):
        warnings.append("Transformed data contains infinite values")
    if np.any(np.isnan(X_transformed)):
        warnings.append("Transformed data contains NaN values")

    return {
        "result": X_transformed,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }
