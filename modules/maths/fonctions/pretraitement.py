"""
Quantix – Module pretraitement
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# normalisation
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input contains infinite values")

def _standard_normalization(X: np.ndarray) -> np.ndarray:
    """Standard normalization (z-score)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    return (X - mean) / std

def _minmax_normalization(X: np.ndarray, feature_range: tuple = (0, 1)) -> np.ndarray:
    """Min-max normalization."""
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    return (X - min_val) / (max_val - min_val + 1e-8) * (feature_range[1] - feature_range[0]) + feature_range[0]

def _robust_normalization(X: np.ndarray, quantile_range: tuple = (25, 75)) -> np.ndarray:
    """Robust normalization using quantiles."""
    lower = np.percentile(X, quantile_range[0], axis=0)
    upper = np.percentile(X, quantile_range[1], axis=0)
    return (X - lower) / (upper - lower + 1e-8)

def _compute_metrics(X_normalized: np.ndarray, X_original: np.ndarray,
                     metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute metrics for normalized data."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(X_normalized, X_original)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def normalisation_fit(X: np.ndarray,
                      method: str = 'standard',
                      feature_range: tuple = (0, 1),
                      quantile_range: tuple = (25, 75),
                      metric_funcs: Optional[Dict[str, Callable]] = None,
                      custom_normalization: Optional[Callable] = None) -> Dict:
    """
    Normalize data using specified method.

    Parameters
    ----------
    X : np.ndarray
        Input data to normalize (2D array)
    method : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    feature_range : tuple, optional
        Range for min-max normalization (default: (0, 1))
    quantile_range : tuple, optional
        Quantiles for robust normalization (default: (25, 75))
    metric_funcs : dict, optional
        Dictionary of metric functions to compute
    custom_normalization : callable, optional
        Custom normalization function

    Returns
    -------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> result = normalisation_fit(X, method='standard')
    """
    _validate_input(X)

    # Initialize output dictionary
    output = {
        'result': None,
        'metrics': {},
        'params_used': {
            'method': method,
            'feature_range': feature_range,
            'quantile_range': quantile_range
        },
        'warnings': []
    }

    # Apply normalization
    if method == 'none':
        X_normalized = X.copy()
    elif method == 'standard':
        X_normalized = _standard_normalization(X)
    elif method == 'minmax':
        X_normalized = _minmax_normalization(X, feature_range)
    elif method == 'robust':
        X_normalized = _robust_normalization(X, quantile_range)
    elif custom_normalization is not None:
        X_normalized = custom_normalization(X)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    output['result'] = X_normalized

    # Compute metrics if provided
    if metric_funcs is not None:
        output['metrics'] = _compute_metrics(X_normalized, X, metric_funcs)

    return output

################################################################################
# standardisation
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input contains infinite values")

def _standardize_none(X: np.ndarray) -> np.ndarray:
    """No standardization."""
    return X

def _standardize_standard(X: np.ndarray) -> np.ndarray:
    """Standardization using mean and standard deviation."""
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    return (X - mean) / std

def _standardize_minmax(X: np.ndarray) -> np.ndarray:
    """Min-max normalization."""
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    return (X - min_val) / (max_val - min_val)

def _standardize_robust(X: np.ndarray) -> np.ndarray:
    """Robust standardization using median and IQR."""
    median = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    iqr = q75 - q25
    return (X - median) / iqr

def _compute_metrics(X: np.ndarray, X_transformed: np.ndarray,
                     metric_func: Callable) -> Dict[str, float]:
    """Compute metrics for standardization."""
    if metric_func is None:
        return {}
    return {"metric": metric_func(X, X_transformed)}

def standardisation_fit(
    X: np.ndarray,
    method: str = "standard",
    metric_func: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Standardize data using the specified method.

    Parameters:
    -----------
    X : np.ndarray
        Input data to be standardized.
    method : str, optional (default="standard")
        Standardization method: "none", "standard", "minmax", "robust".
    metric_func : Callable, optional
        Function to compute metrics on standardized data.

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": standardized data
        - "metrics": computed metrics
        - "params_used": parameters used
        - "warnings": any warnings

    Example:
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = standardisation_fit(X, method="standard")
    """
    _validate_inputs(X)

    methods = {
        "none": _standardize_none,
        "standard": _standardize_standard,
        "minmax": _standardize_minmax,
        "robust": _standardize_robust
    }

    if method not in methods:
        raise ValueError(f"Method must be one of {list(methods.keys())}")

    X_transformed = methods[method](X)
    metrics = _compute_metrics(X, X_transformed, metric_func)

    return {
        "result": X_transformed,
        "metrics": metrics,
        "params_used": {"method": method},
        "warnings": []
    }

################################################################################
# encodage_categoriel
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    encoder_type: str = 'one_hot',
    normalize: str = 'none'
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if encoder_type not in ['one_hot', 'ordinal', 'target', 'frequency']:
        raise ValueError("encoder_type must be one of: 'one_hot', 'ordinal', 'target', 'frequency'")
    if normalize not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("normalize must be one of: 'none', 'standard', 'minmax', 'robust'")

def _one_hot_encode(
    X: np.ndarray,
    normalize: str = 'none'
) -> np.ndarray:
    """Perform one-hot encoding."""
    unique_values = np.unique(X)
    encoded = np.zeros((X.size, unique_values.size))
    for i, val in enumerate(unique_values):
        encoded[X == val, i] = 1
    return _normalize_data(encoded, normalize)

def _ordinal_encode(
    X: np.ndarray,
    normalize: str = 'none'
) -> np.ndarray:
    """Perform ordinal encoding."""
    unique_values, indices = np.unique(X, return_inverse=True)
    encoded = indices.reshape(-1, 1)
    return _normalize_data(encoded, normalize)

def _target_encode(
    X: np.ndarray,
    y: np.ndarray,
    smoothing: float = 1.0
) -> np.ndarray:
    """Perform target encoding."""
    unique_values = np.unique(X)
    encoded = np.zeros_like(X, dtype=float)
    for val in unique_values:
        mask = X == val
        mean_y = np.mean(y[mask])
        encoded[mask] = (mean_y * len(mask) + smoothing * np.mean(y)) / (len(mask) + smoothing)
    return encoded.reshape(-1, 1)

def _frequency_encode(
    X: np.ndarray,
    normalize: str = 'none'
) -> np.ndarray:
    """Perform frequency encoding."""
    unique_values, counts = np.unique(X, return_counts=True)
    encoded = np.zeros_like(X, dtype=float)
    for val, count in zip(unique_values, counts):
        encoded[X == val] = count
    return _normalize_data(encoded.reshape(-1, 1), normalize)

def _normalize_data(
    X: np.ndarray,
    method: str = 'none'
) -> np.ndarray:
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
        raise ValueError("Invalid normalization method")

def encodage_categoriel_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    encoder_type: str = 'one_hot',
    normalize: str = 'none',
    smoothing: float = 1.0,
    custom_encoder: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Perform categorical encoding on input data.

    Parameters:
    - X: Input categorical data as numpy array
    - y: Target values (required for target encoding)
    - encoder_type: Type of encoding ('one_hot', 'ordinal', 'target', 'frequency')
    - normalize: Normalization method ('none', 'standard', 'minmax', 'robust')
    - smoothing: Smoothing factor for target encoding
    - custom_encoder: Custom encoding function

    Returns:
    Dictionary containing encoded data, metrics, parameters used and warnings
    """
    _validate_inputs(X, y, encoder_type, normalize)

    if custom_encoder is not None:
        encoded = custom_encoder(X, y)
    elif encoder_type == 'one_hot':
        encoded = _one_hot_encode(X, normalize)
    elif encoder_type == 'ordinal':
        encoded = _ordinal_encode(X, normalize)
    elif encoder_type == 'target' and y is not None:
        encoded = _target_encode(X, y, smoothing)
    elif encoder_type == 'frequency':
        encoded = _frequency_encode(X, normalize)
    else:
        raise ValueError("Invalid encoder type or missing target values")

    # Calculate some basic metrics
    metrics = {
        'input_shape': X.shape,
        'output_shape': encoded.shape,
        'num_unique_values': len(np.unique(X))
    }

    return {
        'result': encoded,
        'metrics': metrics,
        'params_used': {
            'encoder_type': encoder_type,
            'normalize': normalize,
            'smoothing': smoothing
        },
        'warnings': []
    }

# Example usage:
"""
X = np.array(['a', 'b', 'a', 'c'])
y = np.array([1, 0, 1, 0])
result = encodage_categoriel_fit(X, y, encoder_type='target', normalize='standard')
"""

################################################################################
# imputation_missing_values
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(data: np.ndarray) -> None:
    """Validate input data for missing value imputation.

    Args:
        data: Input array with potential missing values (np.nan).

    Raises:
        ValueError: If input is not a numpy array or contains invalid values.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array")
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input data must contain numerical values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")

def normalize_data(
    data: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """Normalize data using specified method or custom function.

    Args:
        data: Input array to normalize.
        method: Normalization method ('none', 'standard', 'minmax', 'robust').
        custom_func: Optional custom normalization function.

    Returns:
        Normalized data array.
    """
    if method == "none":
        return data.copy()
    elif custom_func is not None:
        return custom_func(data)
    elif method == "standard":
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        return (data - mean) / std
    elif method == "minmax":
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.nanmedian(data, axis=0)
        iqr = np.nanpercentile(data, 75, axis=0) - np.nanpercentile(data, 25, axis=0)
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "mse",
    custom_func: Optional[Callable] = None
) -> float:
    """Compute specified metric between true and predicted values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        metric: Metric to compute ('mse', 'mae', 'r2', 'logloss').
        custom_func: Optional custom metric function.

    Returns:
        Computed metric value.
    """
    if custom_func is not None:
        return custom_func(y_true, y_pred)

    mask = ~np.isnan(y_true)
    if metric == "mse":
        return np.mean((y_pred[mask] - y_true[mask])**2)
    elif metric == "mae":
        return np.mean(np.abs(y_pred[mask] - y_true[mask]))
    elif metric == "r2":
        ss_res = np.sum((y_pred[mask] - y_true[mask])**2)
        ss_tot = np.sum((y_true[mask] - np.mean(y_true[mask]))**2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == "logloss":
        return -np.mean(y_true[mask] * np.log(y_pred[mask] + 1e-8) +
                       (1 - y_true[mask]) * np.log(1 - y_pred[mask] + 1e-8))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def impute_missing_values(
    data: np.ndarray,
    method: str = "mean",
    normalize_method: str = "standard",
    distance_metric: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """Impute missing values in a dataset using specified method.

    Args:
        data: Input array with missing values (np.nan).
        method: Imputation method ('mean', 'median', 'knn').
        normalize_method: Data normalization method.
        distance_metric: Distance metric for KNN imputation.
        solver: Solver algorithm.
        regularization: Regularization type (None, 'l1', 'l2').
        tol: Tolerance for convergence.
        max_iter: Maximum iterations.
        random_state: Random seed.

    Returns:
        Dictionary containing imputed data, metrics, parameters used and warnings.
    """
    # Validate input
    validate_input(data)

    # Initialize random state if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    normalized_data = normalize_data(data, method=normalize_method)
    missing_mask = np.isnan(normalized_data)

    # Impute missing values
    if method == "mean":
        imputed = np.nanmean(normalized_data, axis=0)
    elif method == "median":
        imputed = np.nanmedian(normalized_data, axis=0)
    elif method == "knn":
        # KNN imputation implementation would go here
        pass
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    # Fill missing values
    normalized_data[missing_mask] = imputed

    # Denormalize data if needed
    if normalize_method != "none":
        denormalized_data = denormalize_data(normalized_data, normalize_method)
    else:
        denormalized_data = normalized_data

    # Compute metrics (example with MSE)
    original_complete = data[~np.isnan(data)]
    imputed_complete = denormalized_data[~missing_mask]
    mse = compute_metric(original_complete, imputed_complete)

    # Prepare output
    result = {
        "result": denormalized_data,
        "metrics": {"mse": mse},
        "params_used": {
            "method": method,
            "normalize_method": normalize_method,
            "distance_metric": distance_metric,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return result

def denormalize_data(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Denormalize data based on the normalization method used.

    Args:
        data: Normalized data to denormalize.
        method: Original normalization method.

    Returns:
        Denormalized data array.
    """
    if method == "none":
        return data.copy()
    elif method == "standard":
        # For standard normalization, we would need the original mean and std
        # This is a simplified version assuming we can't store them
        return data  # In practice, you'd need to store and use original stats
    elif method == "minmax":
        # For minmax normalization, we would need the original min and max
        return data  # In practice, you'd need to store and use original stats
    elif method == "robust":
        # For robust normalization, we would need the original median and IQR
        return data  # In practice, you'd need to store and use original stats
    else:
        raise ValueError(f"Unknown denormalization method: {method}")

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values
    data = np.array([[1, 2, np.nan],
                    [4, np.nan, 6],
                    [7, 8, 9]])

    # Impute missing values using mean imputation
    result = impute_missing_values(data, method="mean")

    print("Imputed data:")
    print(result["result"])
    print("\nMetrics:")
    print(result["metrics"])

################################################################################
# scaling_minmax
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input contains NaN or infinite values")

def _minmax_scale(X: np.ndarray, feature_range: tuple = (0, 1)) -> np.ndarray:
    """Apply min-max scaling to the input data."""
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    data_range = max_val - min_val
    # Avoid division by zero for constant features
    scale = np.where(data_range == 0, 0, (feature_range[1] - feature_range[0]) / data_range)
    return X * scale + (feature_range[0] - min_val * scale)

def _compute_metrics(
    X: np.ndarray,
    X_scaled: np.ndarray,
    metric_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics between original and scaled data."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(X, X_scaled)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def scaling_minmax_fit(
    X: np.ndarray,
    feature_range: tuple = (0, 1),
    metric_funcs: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
    custom_scaler: Optional[Callable[[np.ndarray, tuple], np.ndarray]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, any]]]:
    """
    Apply min-max scaling to the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    feature_range : tuple, optional
        Desired range of transformed data (default: (0, 1))
    metric_funcs : dict, optional
        Dictionary of metric functions to compute between original and scaled data
    custom_scaler : callable, optional
        Custom scaling function to use instead of min-max

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": scaled data
        - "metrics": computed metrics
        - "params_used": parameters used
        - "warnings": any warnings generated

    Example:
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = scaling_minmax_fit(X)
    """
    _validate_inputs(X)

    warnings = []
    params_used = {
        "feature_range": feature_range,
        "metric_funcs": metric_funcs is not None,
        "custom_scaler_used": custom_scaler is not None
    }

    if custom_scaler is not None:
        X_scaled = custom_scaler(X, feature_range)
    else:
        X_scaled = _minmax_scale(X, feature_range)

    metrics = {}
    if metric_funcs is not None:
        metrics = _compute_metrics(X, X_scaled, metric_funcs)

    return {
        "result": X_scaled,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# reduction_dimensionnelle
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def reduction_dimensionnelle_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalizer: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Perform dimensionality reduction on input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int, optional
        Number of dimensions to reduce to (default: 2)
    normalizer : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard')
    distance_metric : str or callable, optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable (default: 'euclidean')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default: 'closed_form')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet') (default: None)
    tol : float, optional
        Tolerance for convergence (default: 1e-4)
    max_iter : int, optional
        Maximum number of iterations (default: 1000)
    custom_metric : callable, optional
        Custom metric function if needed (default: None)
    **kwargs :
        Additional solver-specific parameters

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': Reduced data matrix of shape (n_samples, n_components)
        - 'metrics': Dictionary of computed metrics
        - 'params_used': Parameters actually used in the computation
        - 'warnings': List of warnings encountered

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> result = reduction_dimensionnelle_fit(X, n_components=2)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Normalize data
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize warnings list
    warnings = []

    # Choose distance metric
    if isinstance(distance_metric, str):
        distance_func = _get_distance_function(distance_metric)
    else:
        distance_func = distance_metric

    # Choose solver
    if solver == 'closed_form':
        result, metrics = _solve_closed_form(X_normalized, n_components, distance_func)
    elif solver == 'gradient_descent':
        result, metrics = _solve_gradient_descent(
            X_normalized,
            n_components,
            distance_func,
            tol=tol,
            max_iter=max_iter,
            regularization=regularization,
            **kwargs
        )
    elif solver == 'newton':
        result, metrics = _solve_newton(
            X_normalized,
            n_components,
            distance_func,
            tol=tol,
            max_iter=max_iter,
            regularization=regularization,
            **kwargs
        )
    elif solver == 'coordinate_descent':
        result, metrics = _solve_coordinate_descent(
            X_normalized,
            n_components,
            distance_func,
            tol=tol,
            max_iter=max_iter,
            regularization=regularization,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute additional metrics if custom metric provided
    if custom_metric is not None:
        try:
            custom_value = custom_metric(X, result)
            metrics['custom_metric'] = custom_value
        except Exception as e:
            warnings.append(f"Custom metric computation failed: {str(e)}")

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'normalizer': normalizer,
            'distance_metric': distance_metric if isinstance(distance_metric, str) else 'custom',
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': warnings
    }

def _validate_inputs(X: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if X.shape[0] < n_components:
        raise ValueError("n_samples must be >= n_components")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to data."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _get_distance_function(metric_name: str) -> Callable:
    """Return distance function based on name."""
    if metric_name == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric_name == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric_name == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric_name == 'minkowski':
        return lambda x, y: np.sum(np.abs(x - y)**3)**(1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric_name}")

def _solve_closed_form(X: np.ndarray, n_components: int, distance_func: Callable) -> tuple:
    """Closed form solution for dimensionality reduction."""
    # This is a placeholder - actual implementation would depend on specific algorithm
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    result = U[:, :n_components] @ np.diag(S[:n_components])

    # Compute metrics
    reconstruction = result @ Vt[:n_components].T
    mse = np.mean((X - reconstruction) ** 2)
    r2 = 1 - mse / np.var(X)

    return result, {'mse': mse, 'r2': r2}

def _solve_gradient_descent(
    X: np.ndarray,
    n_components: int,
    distance_func: Callable,
    tol: float = 1e-4,
    max_iter: int = 1000,
    regularization: Optional[str] = None,
    **kwargs
) -> tuple:
    """Gradient descent solver for dimensionality reduction."""
    # Initialize parameters
    n_samples, n_features = X.shape
    W = np.random.randn(n_features, n_components)

    for i in range(max_iter):
        # Compute gradients (placeholder implementation)
        gradients = _compute_gradients(X, W, distance_func)

        # Apply regularization if needed
        if regularization == 'l1':
            gradients += np.sign(W) * kwargs.get('alpha', 0.1)
        elif regularization == 'l2':
            gradients += kwargs.get('alpha', 0.1) * W

        # Update parameters
        W -= kwargs.get('learning_rate', 0.01) * gradients

        # Check convergence
        if np.linalg.norm(gradients) < tol:
            break

    result = X @ W
    mse = np.mean((X - result @ W.T) ** 2)
    r2 = 1 - mse / np.var(X)

    return result, {'mse': mse, 'r2': r2}

def _compute_gradients(X: np.ndarray, W: np.ndarray, distance_func: Callable) -> np.ndarray:
    """Compute gradients for gradient descent solver."""
    # Placeholder implementation
    return 2 * X.T @ (X @ W - X @ np.linalg.pinv(X @ W) @ X.T @ X @ W)

def _solve_newton(
    X: np.ndarray,
    n_components: int,
    distance_func: Callable,
    tol: float = 1e-4,
    max_iter: int = 1000,
    regularization: Optional[str] = None,
    **kwargs
) -> tuple:
    """Newton method solver for dimensionality reduction."""
    # Initialize parameters
    n_samples, n_features = X.shape
    W = np.random.randn(n_features, n_components)

    for i in range(max_iter):
        # Compute gradients and hessian (placeholder implementation)
        gradients = _compute_gradients(X, W, distance_func)
        hessian = _compute_hessian(X, W)

        # Apply regularization if needed
        if regularization == 'l2':
            hessian += kwargs.get('alpha', 0.1) * np.eye(n_features)

        # Update parameters
        W -= np.linalg.solve(hessian, gradients).T

        # Check convergence
        if np.linalg.norm(gradients) < tol:
            break

    result = X @ W
    mse = np.mean((X - result @ W.T) ** 2)
    r2 = 1 - mse / np.var(X)

    return result, {'mse': mse, 'r2': r2}

def _compute_hessian(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Compute hessian matrix for Newton method."""
    # Placeholder implementation
    return 2 * X.T @ X

def _solve_coordinate_descent(
    X: np.ndarray,
    n_components: int,
    distance_func: Callable,
    tol: float = 1e-4,
    max_iter: int = 1000,
    regularization: Optional[str] = None,
    **kwargs
) -> tuple:
    """Coordinate descent solver for dimensionality reduction."""
    # Initialize parameters
    n_samples, n_features = X.shape
    W = np.random.randn(n_features, n_components)

    for i in range(max_iter):
        # Update each coordinate (placeholder implementation)
        for j in range(n_features):
            W[j, :] = _update_coordinate(X, W, j, distance_func)

        # Check convergence
        if np.linalg.norm(_compute_gradients(X, W, distance_func)) < tol:
            break

    result = X @ W
    mse = np.mean((X - result @ W.T) ** 2)
    r2 = 1 - mse / np.var(X)

    return result, {'mse': mse, 'r2': r2}

def _update_coordinate(X: np.ndarray, W: np.ndarray, j: int, distance_func: Callable) -> np.ndarray:
    """Update a single coordinate in coordinate descent."""
    # Placeholder implementation
    W_copy = W.copy()
    W_copy[j, :] = 0
    residual = X - X @ W_copy

    # Solve least squares for this coordinate
    return np.linalg.pinv(X[:, j:j+1].T @ X[:, j:j+1]) @ (X[:, j:j+1].T @ residual)

################################################################################
# binning
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def binning_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    n_bins: int = 10,
    strategy: str = 'quantile',
    metric: Union[str, Callable] = 'mse',
    normalize: str = 'none',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict:
    """
    Perform binning on numerical data with various configurable options.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values if supervised binning is needed
    n_bins : int, default=10
        Number of bins to create
    strategy : str, default='quantile'
        Binning strategy: 'uniform', 'quantile', or 'custom'
    metric : Union[str, Callable], default='mse'
        Metric to optimize: 'mse', 'mae', 'r2', or custom callable
    normalize : str, default='none'
        Normalization method: 'none', 'standard', 'minmax', or 'robust'
    solver : str, default='closed_form'
        Solver method: 'closed_form', 'gradient_descent', or 'newton'
    regularization : Optional[str], default=None
        Regularization type: None, 'l1', 'l2', or 'elasticnet'
    tol : float, default=1e-4
        Tolerance for convergence
    max_iter : int, default=1000
        Maximum number of iterations
    random_state : Optional[int], default=None
        Random seed for reproducibility

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': binning results
        - 'metrics': computed metrics
        - 'params_used': parameters used
        - 'warnings': any warnings generated

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = binning_fit(X, y, n_bins=5, strategy='quantile', metric='mse')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if needed
    X_normalized = _apply_normalization(X, normalize)

    # Determine bin edges based on strategy
    bin_edges = _determine_bin_edges(X_normalized, n_bins, strategy)

    # Perform binning
    bins = _perform_binning(X_normalized, bin_edges)

    # Calculate metrics if y is provided
    metrics = {}
    if y is not None:
        metrics = _calculate_metrics(bins, y, metric)

    # Prepare output
    result = {
        'result': bins,
        'metrics': metrics,
        'params_used': {
            'n_bins': n_bins,
            'strategy': strategy,
            'metric': metric,
            'normalize': normalize,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if y is not None and np.any(np.isnan(y)):
        raise ValueError("y contains NaN values")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the data."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _determine_bin_edges(X: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Determine bin edges based on the specified strategy."""
    if strategy == 'uniform':
        return np.linspace(np.min(X), np.max(X), n_bins + 1)
    elif strategy == 'quantile':
        return np.nanpercentile(X, np.linspace(0, 100, n_bins + 1))
    elif strategy == 'custom':
        # Implement custom bin edge determination
        raise NotImplementedError("Custom binning strategy not yet implemented")
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

def _perform_binning(X: np.ndarray, bin_edges: np.ndarray) -> Dict:
    """Perform the actual binning operation."""
    bins = {}
    for i in range(X.shape[1]):
        feature_bins = np.digitize(X[:, i], bin_edges) - 1
        bins[f'feature_{i}'] = feature_bins
    return bins

def _calculate_metrics(bins: Dict, y: np.ndarray, metric: Union[str, Callable]) -> Dict:
    """Calculate specified metrics for the binning results."""
    metrics = {}
    if callable(metric):
        # Custom metric function
        for feature, bin_labels in bins.items():
            metrics[f'custom_{feature}'] = metric(y, bin_labels)
    elif metric == 'mse':
        for feature, bin_labels in bins.items():
            metrics[f'mse_{feature}'] = np.mean((y - bin_labels) ** 2)
    elif metric == 'mae':
        for feature, bin_labels in bins.items():
            metrics[f'mae_{feature}'] = np.mean(np.abs(y - bin_labels))
    elif metric == 'r2':
        for feature, bin_labels in bins.items():
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - bin_labels) ** 2)
            metrics[f'r2_{feature}'] = 1 - (ss_res / ss_total)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics

################################################################################
# transformation_log
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    X: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray],
    metric: Union[str, Callable]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if np.any(np.isnan(X)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input data contains infinite values")

def _apply_normalization(
    X: np.ndarray,
    normalization: str
) -> np.ndarray:
    """Apply selected normalization to the data."""
    if normalization == "none":
        return X
    elif normalization == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif normalization == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif normalization == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Compute the selected metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)

    if metric == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == "logloss":
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def transformation_log_fit(
    X: np.ndarray,
    normalization: str = "none",
    metric: Union[str, Callable] = "mse",
    solver: str = "closed_form"
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a log transformation model to the data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    normalization : str, optional
        Normalization method to apply ("none", "standard", "minmax", "robust")
    metric : str or callable, optional
        Metric to evaluate the transformation ("mse", "mae", "r2", "logloss")
    solver : str, optional
        Solver to use for fitting ("closed_form")

    Returns
    -------
    dict
        Dictionary containing:
        - "result": transformed data
        - "metrics": computed metrics
        - "params_used": parameters used for fitting
        - "warnings": any warnings generated

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = transformation_log_fit(X)
    """
    warnings = []

    # Validate inputs
    _validate_inputs(X, lambda x: x, metric)

    # Apply normalization
    X_normalized = _apply_normalization(X.copy(), normalization)

    # Fit the log transformation
    if solver == "closed_form":
        X_transformed = np.log(X_normalized + 1e-8)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    if metric == "logloss":
        warnings.append("Log loss is not appropriate for log transformation")
    metrics = {"metric": _compute_metric(X_normalized, X_transformed, metric)}

    return {
        "result": X_transformed,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "solver": solver
        },
        "warnings": warnings
    }

################################################################################
# scaling_robust
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def scaling_robust_fit(
    X: np.ndarray,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a robust scaling model to the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    normalization : str, optional
        Type of normalization to apply. Options: 'none', 'standard', 'minmax', 'robust'.
    metric : str or callable, optional
        Metric to evaluate the scaling. Options: 'mse', 'mae', 'r2', 'logloss'.
    distance : str, optional
        Distance metric for robust scaling. Options: 'euclidean', 'manhattan', 'cosine', 'minkowski'.
    solver : str, optional
        Solver to use for optimization. Options: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.
    regularization : str, optional
        Type of regularization. Options: 'none', 'l1', 'l2', 'elasticnet'.
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
    Dict
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X)

    # Choose normalization method
    if normalization == 'standard':
        X_normalized, params = _normalize_standard(X)
    elif normalization == 'minmax':
        X_normalized, params = _normalize_minmax(X)
    elif normalization == 'robust':
        X_normalized, params = _normalize_robust(X)
    else:
        X_normalized, params = X, {}

    # Choose metric
    if isinstance(metric, str):
        if metric == 'mse':
            metric_func = _compute_mse
        elif metric == 'mae':
            metric_func = _compute_mae
        elif metric == 'r2':
            metric_func = _compute_r2
        elif metric == 'logloss':
            metric_func = _compute_logloss
        else:
            raise ValueError("Invalid metric specified.")
    else:
        metric_func = metric

    # Choose distance
    if isinstance(distance, str):
        if distance == 'euclidean':
            distance_func = _compute_euclidean
        elif distance == 'manhattan':
            distance_func = _compute_manhattan
        elif distance == 'cosine':
            distance_func = _compute_cosine
        elif distance == 'minkowski':
            distance_func = lambda x, y: np.sum(np.abs(x - y) ** 2, axis=1)
        else:
            raise ValueError("Invalid distance specified.")
    else:
        distance_func = distance

    # Choose solver
    if solver == 'closed_form':
        result, metrics = _solve_closed_form(X_normalized, metric_func)
    elif solver == 'gradient_descent':
        result, metrics = _solve_gradient_descent(X_normalized, metric_func, tol, max_iter)
    elif solver == 'newton':
        result, metrics = _solve_newton(X_normalized, metric_func, tol, max_iter)
    elif solver == 'coordinate_descent':
        result, metrics = _solve_coordinate_descent(X_normalized, metric_func, tol, max_iter)
    else:
        raise ValueError("Invalid solver specified.")

    # Apply regularization if needed
    if regularization == 'l1':
        result = _apply_l1_regularization(result, X_normalized)
    elif regularization == 'l2':
        result = _apply_l2_regularization(result, X_normalized)
    elif regularization == 'elasticnet':
        result = _apply_elasticnet_regularization(result, X_normalized)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values.")

def _normalize_standard(X: np.ndarray) -> tuple:
    """Standard normalization (mean=0, std=1)."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized, {'mean': mean, 'std': std}

def _normalize_minmax(X: np.ndarray) -> tuple:
    """Min-max normalization (range [0, 1])."""
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    X_normalized = (X - min_val) / (max_val - min_val)
    return X_normalized, {'min': min_val, 'max': max_val}

def _normalize_robust(X: np.ndarray) -> tuple:
    """Robust normalization (median and IQR)."""
    median = np.median(X, axis=0)
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
    X_normalized = (X - median) / iqr
    return X_normalized, {'median': median, 'iqr': iqr}

def _compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def _compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _compute_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Log Loss."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _compute_euclidean(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Euclidean distance."""
    return np.linalg.norm(x - y)

def _compute_manhattan(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Manhattan distance."""
    return np.sum(np.abs(x - y))

def _compute_cosine(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cosine distance."""
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def _solve_closed_form(X: np.ndarray, metric_func: Callable) -> tuple:
    """Solve using closed-form solution."""
    # Placeholder for actual implementation
    result = np.linalg.pinv(X.T @ X) @ (X.T @ np.ones(X.shape[0]))
    metrics = {'metric': metric_func(np.ones(X.shape[0]), X @ result)}
    return result, metrics

def _solve_gradient_descent(X: np.ndarray, metric_func: Callable, tol: float, max_iter: int) -> tuple:
    """Solve using gradient descent."""
    # Placeholder for actual implementation
    result = np.zeros(X.shape[1])
    metrics = {'metric': float('inf')}
    for _ in range(max_iter):
        grad = 2 * X.T @ (X @ result - np.ones(X.shape[0]))
        result -= 0.01 * grad
        current_metric = metric_func(np.ones(X.shape[0]), X @ result)
        if abs(metrics['metric'] - current_metric) < tol:
            break
        metrics['metric'] = current_metric
    return result, metrics

def _solve_newton(X: np.ndarray, metric_func: Callable, tol: float, max_iter: int) -> tuple:
    """Solve using Newton's method."""
    # Placeholder for actual implementation
    result = np.zeros(X.shape[1])
    metrics = {'metric': float('inf')}
    for _ in range(max_iter):
        grad = 2 * X.T @ (X @ result - np.ones(X.shape[0]))
        hess = 2 * X.T @ X
        result -= np.linalg.pinv(hess) @ grad
        current_metric = metric_func(np.ones(X.shape[0]), X @ result)
        if abs(metrics['metric'] - current_metric) < tol:
            break
        metrics['metric'] = current_metric
    return result, metrics

def _solve_coordinate_descent(X: np.ndarray, metric_func: Callable, tol: float, max_iter: int) -> tuple:
    """Solve using coordinate descent."""
    # Placeholder for actual implementation
    result = np.zeros(X.shape[1])
    metrics = {'metric': float('inf')}
    for _ in range(max_iter):
        for i in range(X.shape[1]):
            X_i = X[:, i]
            residual = np.ones(X.shape[0]) - X @ result + result[i] * X_i
            result[i] = (X_i.T @ residual) / (X_i.T @ X_i)
        current_metric = metric_func(np.ones(X.shape[0]), X @ result)
        if abs(metrics['metric'] - current_metric) < tol:
            break
        metrics['metric'] = current_metric
    return result, metrics

def _apply_l1_regularization(result: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply L1 regularization."""
    # Placeholder for actual implementation
    return result

def _apply_l2_regularization(result: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply L2 regularization."""
    # Placeholder for actual implementation
    return result

def _apply_elasticnet_regularization(result: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply ElasticNet regularization."""
    # Placeholder for actual implementation
    return result

# Example usage:
# result = scaling_robust_fit(X, normalization='standard', metric='mse', solver='closed_form')

################################################################################
# reduction_outliers
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def reduction_outliers_fit(
    X: np.ndarray,
    method: str = 'median',
    threshold: float = 3.0,
    axis: int = 0,
    normalize: Optional[str] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict:
    """
    Fit a reduction outliers model to the data.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    method : str, optional
        Method for outlier detection ('median', 'mean', 'iqr')
    threshold : float, optional
        Threshold for outlier detection
    axis : int, optional
        Axis along which to compute statistics (0 for columns, 1 for rows)
    normalize : str or None, optional
        Normalization method ('standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric for evaluation ('mse', 'mae', 'r2')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent')
    regularization : str or None, optional
        Regularization method ('l1', 'l2', 'elasticnet')
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    random_state : int or None, optional
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, method, normalize, metric)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalize)

    # Fit the model based on chosen method
    if method == 'median':
        params = _fit_median_outliers(X_normalized, threshold, axis)
    elif method == 'mean':
        params = _fit_mean_outliers(X_normalized, threshold, axis)
    elif method == 'iqr':
        params = _fit_iqr_outliers(X_normalized, threshold, axis)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, params, metric)

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'method': method,
            'threshold': threshold,
            'axis': axis,
            'normalize': normalize,
            'metric': metric,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(X_normalized, params)
    }

def _validate_inputs(
    X: np.ndarray,
    method: str,
    normalize: Optional[str],
    metric: Union[str, Callable]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input data contains infinite values")

    valid_methods = ['median', 'mean', 'iqr']
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")

    valid_normalizations = [None, 'standard', 'minmax', 'robust']
    if normalize not in valid_normalizations:
        raise ValueError(f"Normalization must be one of {valid_normalizations}")

    valid_metrics = ['mse', 'mae', 'r2']
    if isinstance(metric, str) and metric not in valid_metrics:
        raise ValueError(f"Metric must be one of {valid_metrics} or a callable")

def _apply_normalization(
    X: np.ndarray,
    method: Optional[str]
) -> np.ndarray:
    """Apply normalization to the data."""
    if method is None:
        return X

    if method == 'standard':
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        med = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - med) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_normalized

def _fit_median_outliers(
    X: np.ndarray,
    threshold: float,
    axis: int
) -> Dict:
    """Fit outlier detection using median and MAD."""
    med = np.median(X, axis=axis)
    mad = _median_absolute_deviation(X, axis=axis)

    if axis == 0:
        outlier_mask = np.abs((X - med) / mad) > threshold
    else:
        outlier_mask = np.abs((X.T - med) / mad) > threshold

    return {
        'median': med,
        'mad': mad,
        'outlier_mask': outlier_mask
    }

def _fit_mean_outliers(
    X: np.ndarray,
    threshold: float,
    axis: int
) -> Dict:
    """Fit outlier detection using mean and standard deviation."""
    mean = np.mean(X, axis=axis)
    std = np.std(X, axis=axis)

    if axis == 0:
        outlier_mask = np.abs((X - mean) / std) > threshold
    else:
        outlier_mask = np.abs((X.T - mean) / std) > threshold

    return {
        'mean': mean,
        'std': std,
        'outlier_mask': outlier_mask
    }

def _fit_iqr_outliers(
    X: np.ndarray,
    threshold: float,
    axis: int
) -> Dict:
    """Fit outlier detection using IQR."""
    q25 = np.percentile(X, 25, axis=axis)
    q75 = np.percentile(X, 75, axis=axis)
    iqr = q75 - q25

    if axis == 0:
        lower_bound = q25 - threshold * iqr
        upper_bound = q75 + threshold * iqr
        outlier_mask = (X < lower_bound) | (X > upper_bound)
    else:
        lower_bound = q25 - threshold * iqr
        upper_bound = q75 + threshold * iqr
        outlier_mask = (X.T < lower_bound) | (X.T > upper_bound)

    return {
        'q25': q25,
        'q75': q75,
        'iqr': iqr,
        'outlier_mask': outlier_mask
    }

def _median_absolute_deviation(
    X: np.ndarray,
    axis: int
) -> np.ndarray:
    """Compute median absolute deviation."""
    med = np.median(X, axis=axis)
    if axis == 0:
        return np.median(np.abs(X - med), axis=axis)
    else:
        return np.median(np.abs(X.T - med), axis=axis)

def _compute_metrics(
    X: np.ndarray,
    params: Dict,
    metric: Union[str, Callable]
) -> Dict:
    """Compute evaluation metrics."""
    if isinstance(metric, str):
        if metric == 'mse':
            return {'mse': _compute_mse(X, params)}
        elif metric == 'mae':
            return {'mae': _compute_mae(X, params)}
        elif metric == 'r2':
            return {'r2': _compute_r2(X, params)}
    else:
        return {metric.__name__: metric(X, params)}

def _compute_mse(
    X: np.ndarray,
    params: Dict
) -> float:
    """Compute mean squared error."""
    if 'median' in params:
        return np.mean((X - params['median']) ** 2)
    elif 'mean' in params:
        return np.mean((X - params['mean']) ** 2)
    else:
        raise ValueError("Invalid parameters for MSE computation")

def _compute_mae(
    X: np.ndarray,
    params: Dict
) -> float:
    """Compute mean absolute error."""
    if 'median' in params:
        return np.mean(np.abs(X - params['median']))
    elif 'mean' in params:
        return np.mean(np.abs(X - params['mean']))
    else:
        raise ValueError("Invalid parameters for MAE computation")

def _compute_r2(
    X: np.ndarray,
    params: Dict
) -> float:
    """Compute R-squared."""
    if 'median' in params:
        y_pred = np.tile(params['median'], (X.shape[0], 1))
    elif 'mean' in params:
        y_pred = np.tile(params['mean'], (X.shape[0], 1))
    else:
        raise ValueError("Invalid parameters for R2 computation")

    ss_res = np.sum((X - y_pred) ** 2)
    ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
    return 1 - (ss_res / ss_tot)

def _check_warnings(
    X: np.ndarray,
    params: Dict
) -> list:
    """Check for potential warnings."""
    warnings = []

    if np.any(np.isnan(params['outlier_mask'])):
        warnings.append("Outlier mask contains NaN values")

    if np.any(np.isinf(params['outlier_mask'])):
        warnings.append("Outlier mask contains infinite values")

    if np.sum(params['outlier_mask']) == 0:
        warnings.append("No outliers detected")

    if np.sum(params['outlier_mask']) == X.size:
        warnings.append("All points detected as outliers")

    return warnings

################################################################################
# tokenization
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(
    X: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray],
    distance_metric: Callable[[np.ndarray, np.ndarray], float]
) -> None:
    """Validate input data and functions."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("X contains infinite values")

def default_normalizer(data: np.ndarray) -> np.ndarray:
    """Default normalizer (standard scaling)."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-8)

def default_distance_metric(a: np.ndarray, b: np.ndarray) -> float:
    """Default distance metric (Euclidean)."""
    return np.linalg.norm(a - b)

def tokenization_fit(
    X: np.ndarray,
    *,
    normalizer: Callable[[np.ndarray], np.ndarray] = default_normalizer,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = default_distance_metric,
    min_token_size: int = 1,
    max_token_size: Optional[int] = None,
    threshold: float = 0.5
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Tokenize input data based on distance metrics.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the data
    distance_metric : Callable[[np.ndarray, np.ndarray], float]
        Function to compute distance between tokens
    min_token_size : int
        Minimum size of a token
    max_token_size : Optional[int]
        Maximum size of a token. If None, no maximum.
    threshold : float
        Distance threshold for merging tokens

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]
        Dictionary containing:
        - "tokens": array of tokens
        - "metrics": dictionary of computed metrics
        - "params_used": dictionary of parameters used
        - "warnings": list of warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> result = tokenization_fit(X, min_token_size=2, threshold=0.3)
    """
    # Validate inputs
    validate_inputs(X, normalizer, distance_metric)

    # Normalize data
    normalized_X = normalizer(X)
    params_used = {
        "normalizer": str(normalizer.__name__),
        "distance_metric": str(distance_metric.__name__),
        "min_token_size": min_token_size,
        "max_token_size": max_token_size,
        "threshold": threshold
    }

    # Initialize tokens as individual points
    tokens = [np.array([x]) for x in normalized_X]

    # Merge tokens based on distance threshold
    changed = True
    warnings = []
    while changed:
        changed = False
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                dist = distance_metric(np.mean(tokens[i], axis=0), np.mean(tokens[j], axis=0))
                if dist < threshold:
                    # Check token size constraints
                    new_size = len(tokens[i]) + len(tokens[j])
                    if (max_token_size is None or new_size <= max_token_size) and \
                       new_size >= min_token_size:
                        tokens[i] = np.vstack((tokens[i], tokens[j]))
                        del tokens[j]
                        changed = True
                        break

    # Calculate metrics
    avg_token_size = np.mean([len(token) for token in tokens])
    num_tokens = len(tokens)
    metrics = {
        "avg_token_size": float(avg_token_size),
        "num_tokens": num_tokens,
        "coverage": float(num_tokens) / len(normalized_X)
    }

    # Check for warnings
    if max_token_size is not None and any(len(token) > max_token_size for token in tokens):
        warnings.append("Some tokens exceed max_token_size")

    return {
        "tokens": np.array(tokens, dtype=object),
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# stemming
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(texts: list) -> None:
    """Validate input texts for stemming."""
    if not isinstance(texts, list):
        raise TypeError("Input must be a list of strings.")
    if not all(isinstance(text, str) for text in texts):
        raise ValueError("All elements in the input list must be strings.")

def default_normalize(text: str) -> str:
    """Default text normalization."""
    return text.lower()

def stemming_fit(
    texts: list,
    normalizer: Callable[[str], str] = default_normalize,
    stemmer: Optional[Callable[[str], str]] = None,
    **kwargs
) -> Dict[str, Union[list, dict]]:
    """
    Perform stemming on a list of texts.

    Parameters
    ----------
    texts : list
        List of input strings to be stemmed.
    normalizer : callable, optional
        Function to normalize text before stemming. Default is lowercase.
    stemmer : callable, optional
        Function to perform stemming. If None, no stemming is applied.

    Returns
    -------
    dict
        Dictionary containing:
        - "result": list of stemmed texts
        - "metrics": dictionary of metrics (currently empty)
        - "params_used": dictionary of parameters used
        - "warnings": list of warnings

    Examples
    --------
    >>> stemming_fit(["running", "runs"])
    {
        'result': ['run', 'run'],
        'metrics': {},
        'params_used': {'normalizer': '<function default_normalize at ...>', 'stemmer': None},
        'warnings': []
    }
    """
    validate_input(texts)

    stemmed_texts = []
    warnings = []

    for text in texts:
        try:
            normalized_text = normalizer(text)
            if stemmer is not None:
                stemmed_text = stemmer(normalized_text)
            else:
                stemmed_text = normalized_text
            stemmed_texts.append(stemmed_text)
        except Exception as e:
            warnings.append(f"Error processing text '{text}': {str(e)}")
            stemmed_texts.append(text)  # Fallback to original text

    return {
        "result": stemmed_texts,
        "metrics": {},
        "params_used": {"normalizer": normalizer, "stemmer": stemmer},
        "warnings": warnings
    }

################################################################################
# lemmatisation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List
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
    LOGLOS = "logloss"

class Solver(Enum):
    CLOSED_FORM = "closed_form"
    GRADIENT_DESCENT = "gradient_descent"
    NEWTON = "newton"
    COORDINATE_DESCENT = "coordinate_descent"

class Regularization(Enum):
    NONE = "none"
    L1 = "l1"
    L2 = "l2"
    ELASTICNET = "elasticnet"

def validate_input(texts: List[str], lemmatizer: Callable) -> None:
    """Validate input texts and lemmatizer function."""
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ValueError("Texts must be a list of strings.")
    if not callable(lemmatizer):
        raise ValueError("Lemmatizer must be a callable function.")

def apply_normalization(texts: List[str], normalization: Normalization) -> List[str]:
    """Apply selected normalization to texts."""
    if normalization == Normalization.NONE:
        return texts
    # Implement other normalizations as needed
    raise NotImplementedError("Normalization methods not yet implemented.")

def lemmatize_texts(texts: List[str], lemmatizer: Callable) -> List[str]:
    """Apply lemmatization to texts using the provided lemmatizer."""
    return [lemmatizer(text) for text in texts]

def compute_metrics(original_texts: List[str], lemmatized_texts: List[str],
                    metric: Union[Metric, Callable]) -> Dict[str, float]:
    """Compute selected metrics between original and lemmatized texts."""
    if isinstance(metric, Metric):
        if metric == Metric.MSE:
            return {"mse": np.mean([(len(o) - len(l))**2 for o, l in zip(original_texts, lemmatized_texts)])}
        elif metric == Metric.MAE:
            return {"mae": np.mean([abs(len(o) - len(l)) for o, l in zip(original_texts, lemmatized_texts)])}
        # Implement other metrics as needed
    elif callable(metric):
        return {"custom_metric": metric(original_texts, lemmatized_texts)}
    raise ValueError("Invalid metric provided.")

def lemmatization_fit(texts: List[str], lemmatizer: Callable,
                      normalization: Normalization = Normalization.NONE,
                      metric: Union[Metric, Callable] = Metric.MSE) -> Dict:
    """
    Perform lemmatization on input texts with configurable options.

    Parameters:
    - texts: List of strings to be lemmatized
    - lemmatizer: Callable function that performs the actual lemmatization
    - normalization: Normalization method to apply before lemmatization
    - metric: Metric to compute between original and lemmatized texts

    Returns:
    Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    validate_input(texts, lemmatizer)

    # Apply normalization if specified
    normalized_texts = apply_normalization(texts, normalization)

    # Perform lemmatization
    lemmatized_texts = lemmatize_texts(normalized_texts, lemmatizer)

    # Compute metrics
    metrics = compute_metrics(texts, lemmatized_texts, metric)

    # Prepare output
    result = {
        "result": lemmatized_texts,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization.value,
            "metric": metric.value if isinstance(metric, Metric) else "custom"
        },
        "warnings": []
    }

    return result

# Example usage:
"""
from nltk.stem import WordNetLemmatizer
nltk_lemmatizer = WordNetLemmatizer().lemmatize

texts = ["running", "better", "happiness"]
result = lemmatization_fit(
    texts=texts,
    lemmatizer=nltk_lemmatizer,
    normalization=Normalization.NONE,
    metric=Metric.MAE
)
"""

################################################################################
# suppression_stopwords
################################################################################

import numpy as np
from typing import Callable, Dict, Union, List, Optional

def validate_input(texts: List[str], stopwords: List[str]) -> None:
    """
    Validate input data for suppression_stopwords.

    Parameters
    ----------
    texts : List[str]
        List of text documents to process.
    stopwords : List[str]
        List of stopwords to remove.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ValueError("texts must be a list of strings")
    if not isinstance(stopwords, list) or not all(isinstance(s, str) for s in stopwords):
        raise ValueError("stopwords must be a list of strings")

def suppression_stopwords_fit(
    texts: List[str],
    stopwords: List[str],
    case_sensitive: bool = False,
    custom_preprocessor: Optional[Callable[[str], str]] = None
) -> Dict[str, Union[List[str], Dict[str, int]]]:
    """
    Fit the stopword suppression model and return processed texts.

    Parameters
    ----------
    texts : List[str]
        List of text documents to process.
    stopwords : List[str]
        List of stopwords to remove.
    case_sensitive : bool, optional
        Whether the suppression is case sensitive (default False).
    custom_preprocessor : Callable[[str], str], optional
        Custom preprocessing function to apply before stopword removal.

    Returns
    -------
    Dict[str, Union[List[str], Dict[str, int]]]
        Dictionary containing:
        - "processed_texts": List of processed texts
        - "metrics": Dictionary with processing metrics

    Examples
    --------
    >>> texts = ["This is a sample text", "Another example here"]
    >>> stopwords = ["is", "a", "another"]
    >>> result = suppression_stopwords_fit(texts, stopwords)
    """
    validate_input(texts, stopwords)

    processed_texts = []
    word_counts = {"original": 0, "removed": 0}

    for text in texts:
        if custom_preprocessor is not None:
            text = custom_preprocessor(text)

        words = text.split()
        original_count = len(words)
        word_counts["original"] += original_count

        if not case_sensitive:
            words = [word.lower() for word in words]
            stopwords = [sw.lower() for sw in stopwords]

        filtered_words = [word for word in words if word not in stopwords]
        removed_count = original_count - len(filtered_words)
        word_counts["removed"] += removed_count

        processed_texts.append(" ".join(filtered_words))

    return {
        "result": processed_texts,
        "metrics": word_counts,
        "params_used": {
            "case_sensitive": case_sensitive,
            "custom_preprocessor": custom_preprocessor is not None
        },
        "warnings": []
    }

def suppression_stopwords_compute(
    texts: List[str],
    stopwords: List[str],
    case_sensitive: bool = False,
    custom_preprocessor: Optional[Callable[[str], str]] = None
) -> Dict[str, Union[List[str], Dict[str, int]]]:
    """
    Compute stopword suppression on texts.

    Parameters
    ----------
    texts : List[str]
        List of text documents to process.
    stopwords : List[str]
        List of stopwords to remove.
    case_sensitive : bool, optional
        Whether the suppression is case sensitive (default False).
    custom_preprocessor : Callable[[str], str], optional
        Custom preprocessing function to apply before stopword removal.

    Returns
    -------
    Dict[str, Union[List[str], Dict[str, int]]]
        Dictionary containing:
        - "processed_texts": List of processed texts
        - "metrics": Dictionary with processing metrics

    Examples
    --------
    >>> texts = ["This is a sample text", "Another example here"]
    >>> stopwords = ["is", "a", "another"]
    >>> result = suppression_stopwords_compute(texts, stopwords)
    """
    return suppression_stopwords_fit(
        texts=texts,
        stopwords=stopwords,
        case_sensitive=case_sensitive,
        custom_preprocessor=custom_preprocessor
    )

################################################################################
# one_hot_encoding
################################################################################

import numpy as np
from typing import Dict, Any, Callable, Optional, Union

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data for one-hot encoding."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")

def _one_hot_encode(X: np.ndarray, categories: Optional[Dict[int, list]] = None) -> np.ndarray:
    """Perform one-hot encoding on the input data."""
    if categories is None:
        categories = {i: np.unique(X[:, i]) for i in range(X.shape[1])}

    encoded = []
    for i, col in enumerate(X.T):
        unique_categories = categories[i]
        encoded_col = np.zeros((len(col), len(unique_categories)))
        for j, category in enumerate(unique_categories):
            encoded_col[:, j] = (col == category)
        encoded.append(encoded_col)

    return np.hstack(encoded)

def one_hot_encoding_fit(
    X: np.ndarray,
    categories: Optional[Dict[int, list]] = None,
    handle_unknown: str = 'ignore',
    sparse: bool = False
) -> Dict[str, Any]:
    """
    Fit one-hot encoding on the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data to be encoded.
    categories : Optional[Dict[int, list]]
        Dictionary of categories for each feature. If None, categories are inferred from the data.
    handle_unknown : str
        Strategy for handling unknown categories. Options: 'ignore', 'error'.
    sparse : bool
        Whether to return a sparse matrix.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the fitted encoder and metadata.
    """
    _validate_inputs(X)

    if categories is None:
        categories = {i: np.unique(X[:, i]) for i in range(X.shape[1])}

    result = {
        'categories': categories,
        'handle_unknown': handle_unknown,
        'sparse': sparse
    }

    return result

def one_hot_encoding_compute(
    X: np.ndarray,
    encoder_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute one-hot encoding using fitted parameters.

    Parameters:
    -----------
    X : np.ndarray
        Input data to be encoded.
    encoder_params : Dict[str, Any]
        Parameters from one_hot_encoding_fit.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the encoded data and metadata.
    """
    _validate_inputs(X)

    categories = encoder_params.get('categories', None)
    handle_unknown = encoder_params.get('handle_unknown', 'ignore')
    sparse = encoder_params.get('sparse', False)

    encoded_data = _one_hot_encode(X, categories)

    if sparse:
        from scipy.sparse import csr_matrix
        encoded_data = csr_matrix(encoded_data)

    result = {
        'result': encoded_data,
        'params_used': encoder_params,
        'warnings': []
    }

    return result

################################################################################
# label_encoding
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None
) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if X.ndim != 1:
        raise ValueError("X must be a 1D array")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same length")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if y is not None and np.any(np.isnan(y)):
        raise ValueError("y contains NaN values")

def _compute_label_mapping(
    X: np.ndarray,
    encoding_dict: Optional[Dict] = None
) -> Dict:
    """Compute label mapping dictionary."""
    unique_labels = np.unique(X)
    if encoding_dict is None:
        return {label: idx for idx, label in enumerate(unique_labels)}
    else:
        # Validate encoding_dict
        if not isinstance(encoding_dict, dict):
            raise TypeError("encoding_dict must be a dictionary")
        if not all(isinstance(k, type(X[0])) for k in encoding_dict.keys()):
            raise ValueError("Keys of encoding_dict must match X's dtype")
        if not all(isinstance(v, int) for v in encoding_dict.values()):
            raise ValueError("Values of encoding_dict must be integers")
        return encoding_dict

def label_encoding_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    encoding_dict: Optional[Dict] = None
) -> Dict:
    """
    Fit label encoder and return mapping.

    Parameters:
    -----------
    X : np.ndarray
        Input array of labels to be encoded.
    y : Optional[np.ndarray], default=None
        Target values (unused in this function but kept for consistency).
    encoding_dict : Optional[Dict], default=None
        Predefined mapping from labels to integers. If None, will be computed.

    Returns:
    --------
    Dict
        Dictionary containing:
        - "result": label mapping dictionary
        - "params_used": parameters used
        - "warnings": any warnings generated

    Example:
    --------
    >>> X = np.array(['a', 'b', 'a'])
    >>> result = label_encoding_fit(X)
    >>> print(result['result'])
    {'a': 0, 'b': 1}
    """
    _validate_inputs(X, y)

    result = {
        "result": _compute_label_mapping(X, encoding_dict),
        "params_used": {
            "encoding_dict_provided": encoding_dict is not None
        },
        "warnings": []
    }

    return result

def label_encoding_compute(
    X: np.ndarray,
    encoding_dict: Dict
) -> Dict:
    """
    Transform labels using provided mapping.

    Parameters:
    -----------
    X : np.ndarray
        Input array of labels to be encoded.
    encoding_dict : Dict
        Mapping from labels to integers.

    Returns:
    --------
    Dict
        Dictionary containing:
        - "result": encoded array
        - "params_used": parameters used
        - "warnings": any warnings generated

    Example:
    --------
    >>> X = np.array(['a', 'b', 'a'])
    >>> encoding_dict = {'a': 0, 'b': 1}
    >>> result = label_encoding_compute(X, encoding_dict)
    >>> print(result['result'])
    array([0, 1, 0])
    """
    _validate_inputs(X)

    # Validate encoding_dict covers all labels
    missing_labels = set(X) - set(encoding_dict.keys())
    if missing_labels:
        warnings.warn(f"Labels {missing_labels} not found in encoding_dict")
        # Replace missing labels with -1
        result_array = np.array([encoding_dict.get(label, -1) for label in X])
    else:
        result_array = np.array([encoding_dict[label] for label in X])

    return {
        "result": result_array,
        "params_used": {},
        "warnings": [] if not missing_labels else [f"Missing labels: {missing_labels}"]
    }

################################################################################
# feature_engineering
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def feature_engineering_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalizer: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_normalizer: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Main function for feature engineering preprocessing.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values array of shape (n_samples,) or None
    normalizer : str
        Normalization method: 'none', 'standard', 'minmax', 'robust'
    metric : Union[str, Callable]
        Metric to evaluate: 'mse', 'mae', 'r2', 'logloss' or custom callable
    distance : str
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'minkowski'
    solver : str
        Solver method: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'
    regularization : Optional[str]
        Regularization type: None, 'l1', 'l2', 'elasticnet'
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    custom_normalizer : Optional[Callable]
        Custom normalization function
    custom_metric : Optional[Callable]
        Custom metric function
    custom_distance : Optional[Callable]
        Custom distance function

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Choose normalizer
    if custom_normalizer is not None:
        normalizer_func = custom_normalizer
    else:
        normalizer_func = _get_normalizer(normalizer)

    # Normalize features
    X_norm, norm_params = normalizer_func(X)

    # Initialize results dictionary
    result = {
        'result': None,
        'metrics': {},
        'params_used': {
            'normalizer': normalizer,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

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
    if y is not None:
        result['result'], metrics = _solve_regression(
            X_norm, y,
            solver=solver,
            metric=metric_func,
            distance=distance_func,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter
        )
    else:
        result['result'], metrics = _solve_unsupervised(
            X_norm,
            solver=solver,
            distance=distance_func,
            tol=tol,
            max_iter=max_iter
        )

    result['metrics'] = metrics

    return result

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if y is not None and np.any(np.isnan(y)):
        raise ValueError("y contains NaN values")

def _get_normalizer(normalizer: str) -> Callable:
    """Get normalizer function based on parameter."""
    normalizers = {
        'none': _no_normalization,
        'standard': _standard_normalizer,
        'minmax': _minmax_normalizer,
        'robust': _robust_normalizer
    }
    if normalizer not in normalizers:
        raise ValueError(f"Unknown normalizer: {normalizer}")
    return normalizers[normalizer]

def _no_normalization(X: np.ndarray) -> tuple:
    """No normalization."""
    return X, {}

def _standard_normalizer(X: np.ndarray) -> tuple:
    """Standard normalization (mean=0, std=1)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_norm = (X - mean) / std
    return X_norm, {'mean': mean, 'std': std}

def _minmax_normalizer(X: np.ndarray) -> tuple:
    """Min-max normalization (range [0, 1])."""
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    X_norm = (X - min_val) / (max_val - min_val + 1e-8)
    return X_norm, {'min': min_val, 'max': max_val}

def _robust_normalizer(X: np.ndarray) -> tuple:
    """Robust normalization (median and IQR)."""
    median = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    iqr = q75 - q25
    X_norm = (X - median) / (iqr + 1e-8)
    return X_norm, {'median': median, 'iqr': iqr}

def _get_metric(metric: str) -> Callable:
    """Get metric function based on parameter."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared,
        'logloss': _log_loss
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Log loss."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _get_distance(distance: str) -> Callable:
    """Get distance function based on parameter."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': _minkowski_distance
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _euclidean_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Euclidean distance."""
    if Y is None:
        Y = X
    return np.sqrt(np.sum((X[:, np.newaxis] - Y) ** 2, axis=2))

def _manhattan_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Manhattan distance."""
    if Y is None:
        Y = X
    return np.sum(np.abs(X[:, np.newaxis] - Y), axis=2)

def _cosine_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Cosine distance (1 - cosine similarity)."""
    if Y is None:
        Y = X
    dot_products = np.dot(X, Y.T)
    norms = np.sqrt(np.sum(X**2, axis=1))[:, np.newaxis] * np.sqrt(np.sum(Y**2, axis=1))
    return 1 - (dot_products / (norms + 1e-8))

def _minkowski_distance(X: np.ndarray, Y: Optional[np.ndarray] = None, p: float = 2) -> np.ndarray:
    """Minkowski distance."""
    if Y is None:
        Y = X
    return np.sum(np.abs(X[:, np.newaxis] - Y) ** p, axis=2) ** (1/p)

def _solve_regression(
    X: np.ndarray,
    y: np.ndarray,
    solver: str,
    metric: Callable,
    distance: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple:
    """Solve regression problem."""
    if solver == 'closed_form':
        return _closed_form_solution(X, y, regularization)
    elif solver == 'gradient_descent':
        return _gradient_descent(X, y, metric, tol, max_iter)
    elif solver == 'newton':
        return _newton_method(X, y, metric, tol, max_iter)
    elif solver == 'coordinate_descent':
        return _coordinate_descent(X, y, metric, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _closed_form_solution(X: np.ndarray, y: np.ndarray, regularization: Optional[str]) -> tuple:
    """Closed form solution for linear regression."""
    if regularization == 'l2':
        # Ridge regression
        alpha = 1.0  # Default regularization strength
        I = np.eye(X.shape[1])
        coef = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    elif regularization == 'l1':
        # Lasso regression (simplified)
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=1.0)
        model.fit(X, y)
        coef = model.coef_
    else:
        # Ordinary least squares
        coef = np.linalg.inv(X.T @ X) @ X.T @ y

    y_pred = X @ coef
    metrics = {'metric': _mean_squared_error(y, y_pred)}
    return coef, metrics

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric: Callable,
    tol: float,
    max_iter: int
) -> tuple:
    """Gradient descent optimization."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    learning_rate = 0.01

    for i in range(max_iter):
        gradient = (2/n_samples) * X.T @ (X @ coef - y)
        new_coef = coef - learning_rate * gradient
        if np.linalg.norm(new_coef - coef) < tol:
            break
        coef = new_coef

    y_pred = X @ coef
    metrics = {'metric': metric(y, y_pred)}
    return coef, metrics

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    metric: Callable,
    tol: float,
    max_iter: int
) -> tuple:
    """Newton method optimization."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)

    for i in range(max_iter):
        gradient = (2/n_samples) * X.T @ (X @ coef - y)
        hessian = (2/n_samples) * X.T @ X
        new_coef = coef - np.linalg.inv(hessian) @ gradient

        if np.linalg.norm(new_coef - coef) < tol:
            break
        coef = new_coef

    y_pred = X @ coef
    metrics = {'metric': metric(y, y_pred)}
    return coef, metrics

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric: Callable,
    tol: float,
    max_iter: int
) -> tuple:
    """Coordinate descent optimization."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    learning_rate = 0.1

    for i in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residual = y - (X @ coef) + coef[j] * X_j
            coef[j] = np.sum(X_j * residual) / (np.sum(X_j**2) + 1e-8)

        if i % 10 == 0:  # Check convergence every 10 iterations
            new_coef = coef.copy()
            y_pred = X @ new_coef
            if np.linalg.norm(new_coef - coef) < tol:
                break

    y_pred = X @ coef
    metrics = {'metric': metric(y, y_pred)}
    return coef, metrics

def _solve_unsupervised(
    X: np.ndarray,
    solver: str,
    distance: Callable,
    tol: float,
    max_iter: int
) -> tuple:
    """Solve unsupervised problem."""
    if solver == 'closed_form':
        return _pca_closed_form(X)
    elif solver == 'gradient_descent':
        return _kmeans_gradient_descent(X, distance, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver for unsupervised: {solver}")

def _pca_closed_form(X: np.ndarray) -> tuple:
    """Closed form solution for PCA."""
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute covariance matrix
    cov_matrix = X_centered.T @ X_centered / (X.shape[0] - 1)

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    result = {
        'components': eigvecs,
        'explained_variance': eigvals
    }

    metrics = {
        'total_explained_variance': np.sum(eigvals)
    }

    return result, metrics

def _kmeans_gradient_descent(
    X: np.ndarray,
    distance: Callable,
    tol: float,
    max_iter: int
) -> tuple:
    """K-means clustering with gradient descent."""
    n_samples, n_features = X.shape
    k = 3  # Default number of clusters

    # Initialize centroids randomly
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for i in range(max_iter):
        # Assign clusters
        distances = distance(X[:, np.newaxis], centroids)
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        # Check convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    result = {
        'centroids': centroids,
        'labels': labels
    }

    metrics = {
        'inertia': np.sum([np.sum(distance(X[labels == j], centroids[j])**2) for j in range(k)])
    }

    return result, metrics

################################################################################
# splitting_train_test
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional, Any

def splitting_train_test_fit(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    stratification: bool = False,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'random',
    **kwargs
) -> Dict[str, Any]:
    """
    Split data into train and test sets with optional preprocessing.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    test_size : float, optional
        Proportion of data to use for testing (default: 0.2)
    random_state : int, optional
        Random seed for reproducibility (default: None)
    stratification : bool, optional
        Whether to stratify the split (default: False)
    normalizer : callable, optional
        Function to normalize features (default: identity function)
    metric : str or callable, optional
        Metric to evaluate split quality (default: 'mse')
    distance : str or callable, optional
        Distance metric for stratification (default: 'euclidean')
    solver : str, optional
        Solver strategy ('random', 'stratified') (default: 'random')
    **kwargs : dict
        Additional solver-specific parameters

    Returns:
    --------
    dict
        Dictionary containing:
        - 'train_indices': indices for training set
        - 'test_indices': indices for test set
        - 'metrics': evaluation metrics
        - 'params_used': parameters used
        - 'warnings': any warnings generated

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = splitting_train_test_fit(X, y, test_size=0.3,
    ...                                  stratification=True)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Normalize features
    X_norm = normalizer(X)

    # Determine split indices based on solver strategy
    if stratification:
        train_indices, test_indices = _stratified_split(
            X_norm, y, test_size, distance, rng
        )
    else:
        train_indices, test_indices = _random_split(
            X_norm.shape[0], test_size, rng
        )

    # Calculate metrics
    metrics = _calculate_metrics(
        X_norm, y, train_indices, test_indices,
        metric=metric
    )

    # Prepare output dictionary
    result = {
        'train_indices': train_indices,
        'test_indices': test_indices,
        'metrics': metrics,
        'params_used': {
            'test_size': test_size,
            'random_state': random_state,
            'stratification': stratification,
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'metric': metric,
            'distance': distance,
            'solver': solver
        },
        'warnings': []
    }

    return result

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

def _random_split(n_samples: int, test_size: float, rng) -> tuple:
    """Generate random train/test split indices."""
    n_test = int(n_samples * test_size)
    indices = rng.permutation(n_samples)
    return indices[n_test:], indices[:n_test]

def _stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    distance: Union[str, Callable],
    rng
) -> tuple:
    """Generate stratified train/test split indices."""
    classes = np.unique(y)
    train_indices, test_indices = [], []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        n_test = int(len(cls_indices) * test_size)
        permuted = rng.permutation(cls_indices)

        if distance == 'euclidean':
            # Simple example of distance-based stratification
            centroid = np.mean(X[permuted], axis=0)
            distances = np.linalg.norm(X[permuted] - centroid, axis=1)
            sorted_indices = permuted[np.argsort(distances)]
        elif callable(distance):
            # Custom distance function
            centroid = np.mean(X[permuted], axis=0)
            distances = [distance(x, centroid) for x in X[permuted]]
            sorted_indices = permuted[np.argsort(distances)]
        else:
            # Default random stratification
            sorted_indices = permuted

        test_indices.extend(sorted_indices[:n_test])
        train_indices.extend(sorted_indices[n_test:])

    return np.array(train_indices), np.array(test_indices)

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Calculate evaluation metrics for the split."""
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    metrics = {}

    if metric == 'mse':
        metrics['mse'] = np.mean((y_train - y_test)**2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_train - y_test))
    elif metric == 'r2':
        ss_total = np.sum((y_train - np.mean(y))**2)
        ss_res = np.sum((y_train - y_test)**2)
        metrics['r2'] = 1 - (ss_res / ss_total)
    elif callable(metric):
        metrics['custom'] = metric(y_train, y_test)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    metrics['train_size'] = len(train_indices)
    metrics['test_size'] = len(test_indices)

    return metrics
