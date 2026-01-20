"""
Quantix – Module imputation
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# mean_imputation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    strategy: str = 'column_wise',
    copy: bool = True
) -> np.ndarray:
    """
    Validate input data and return a validated copy if needed.

    Parameters
    ----------
    X : np.ndarray
        Input data array with missing values (np.nan)
    strategy : str, optional
        Strategy for handling missing values ('column_wise' or 'row_wise')
    copy : bool, optional
        Whether to return a copy of the input array

    Returns
    -------
    np.ndarray
        Validated input data

    Raises
    ------
    ValueError
        If input contains invalid values or dimensions
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a numpy array")

    if X.ndim != 2:
        raise ValueError("Input must be a 2-dimensional array")

    if np.isinf(X).any():
        raise ValueError("Input contains infinite values")

    if copy:
        X = X.copy()

    return X

def _calculate_mean_imputation(
    X: np.ndarray,
    strategy: str = 'column_wise'
) -> np.ndarray:
    """
    Calculate mean imputation for missing values.

    Parameters
    ----------
    X : np.ndarray
        Input data array with missing values (np.nan)
    strategy : str, optional
        Strategy for handling missing values ('column_wise' or 'row_wise')

    Returns
    -------
    np.ndarray
        Imputed data array

    Raises
    ------
    ValueError
        If invalid strategy is provided
    """
    if strategy == 'column_wise':
        column_means = np.nanmean(X, axis=0)
        X_imputed = np.where(np.isnan(X), column_means, X)
    elif strategy == 'row_wise':
        row_means = np.nanmean(X, axis=1)
        X_imputed = np.where(np.isnan(X), row_means[:, np.newaxis], X)
    else:
        raise ValueError("Invalid strategy. Choose 'column_wise' or 'row_wise'")

    return X_imputed

def _calculate_metrics(
    X: np.ndarray,
    X_imputed: np.ndarray,
    metric_funcs: Optional[Dict[str, Callable]] = None
) -> Dict[str, float]:
    """
    Calculate metrics for imputation quality.

    Parameters
    ----------
    X : np.ndarray
        Original data array with missing values
    X_imputed : np.ndarray
        Imputed data array
    metric_funcs : Dict[str, Callable], optional
        Dictionary of custom metric functions

    Returns
    -------
    Dict[str, float]
        Dictionary of calculated metrics
    """
    metrics = {}

    if metric_funcs is None:
        metric_funcs = {
            'mse': lambda x, y: np.mean((x - y) ** 2),
            'mae': lambda x, y: np.mean(np.abs(x - y)),
            'r2': lambda x, y: 1 - np.sum((x - y) ** 2) / np.sum((x - np.mean(x)) ** 2)
        }

    for name, func in metric_funcs.items():
        metrics[name] = func(X[~np.isnan(X)], X_imputed[~np.isnan(X)])

    return metrics

def mean_imputation_fit(
    X: np.ndarray,
    strategy: str = 'column_wise',
    metric_funcs: Optional[Dict[str, Callable]] = None,
    copy: bool = True
) -> Dict:
    """
    Perform mean imputation on data with missing values.

    Parameters
    ----------
    X : np.ndarray
        Input data array with missing values (np.nan)
    strategy : str, optional
        Strategy for handling missing values ('column_wise' or 'row_wise')
    metric_funcs : Dict[str, Callable], optional
        Dictionary of custom metric functions
    copy : bool, optional
        Whether to return a copy of the input array

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Imputed data array
        - 'metrics': Dictionary of calculated metrics
        - 'params_used': Dictionary of parameters used
        - 'warnings': List of warnings

    Examples
    --------
    >>> X = np.array([[1, 2], [3, np.nan], [np.nan, 6]])
    >>> result = mean_imputation_fit(X)
    """
    # Validate inputs
    X_validated = _validate_inputs(X, strategy, copy)

    # Calculate imputation
    X_imputed = _calculate_mean_imputation(X_validated, strategy)

    # Calculate metrics
    metrics = _calculate_metrics(X_validated, X_imputed, metric_funcs)

    # Prepare output
    result = {
        'result': X_imputed,
        'metrics': metrics,
        'params_used': {
            'strategy': strategy,
            'copy': copy
        },
        'warnings': []
    }

    return result

################################################################################
# median_imputation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for median imputation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")

def _normalize_data(
    data: np.ndarray,
    method: str = "none",
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data using specified method or custom function."""
    if method == "none":
        return data
    elif method == "standard":
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.nanmedian(data, axis=0)
        iqr = np.nanpercentile(data, 75, axis=0) - np.nanpercentile(data, 25, axis=0)
        return (data - median) / (iqr + 1e-8)
    elif custom_normalizer is not None:
        return custom_normalizer(data)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _calculate_median_imputation(
    data: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """Calculate median imputation for missing values."""
    median_values = np.nanmedian(data, axis=0)
    data_imputed = data.copy()
    data_imputed[mask] = median_values[mask.sum(axis=0) != 0]
    return data_imputed

def _calculate_metrics(
    original: np.ndarray,
    imputed: np.ndarray,
    metric: str = "mse",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Calculate metrics for imputation quality."""
    if custom_metric is not None:
        return {"custom": custom_metric(original, imputed)}

    metrics = {}
    if metric == "mse" or "all" in metric:
        mse = np.mean((original - imputed) ** 2)
        metrics["mse"] = mse
    if metric == "mae" or "all" in metric:
        mae = np.mean(np.abs(original - imputed))
        metrics["mae"] = mae
    if metric == "r2" or "all" in metric:
        ss_res = np.sum((original - imputed) ** 2)
        ss_tot = np.sum((original - np.mean(original)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        metrics["r2"] = r2
    return metrics

def median_imputation_fit(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    normalization: str = "none",
    metric: str = "mse",
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Perform median imputation on missing values in the data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array with missing values (NaN)
    mask : Optional[np.ndarray]
        Binary mask indicating missing values (True where data is missing)
    normalization : str
        Normalization method ("none", "standard", "minmax", "robust")
    metric : str
        Metric to calculate ("mse", "mae", "r2", "all")
    custom_normalizer : Optional[Callable]
        Custom normalization function
    custom_metric : Optional[Callable]
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": Imputed data array
        - "metrics": Calculated metrics
        - "params_used": Parameters used in the computation
        - "warnings": Any warnings generated

    Example:
    --------
    >>> data = np.array([[1, 2], [3, np.nan], [np.nan, 6]])
    >>> mask = np.isnan(data)
    >>> result = median_imputation_fit(data, mask, normalization="standard")
    """
    # Validate input
    _validate_input(data)

    # Create mask if not provided
    if mask is None:
        mask = np.isnan(data)

    # Normalize data
    normalized_data = _normalize_data(
        data,
        method=normalization,
        custom_normalizer=custom_normalizer
    )

    # Calculate median imputation
    imputed_data = _calculate_median_imputation(normalized_data, mask)

    # Calculate metrics
    metrics = _calculate_metrics(
        normalized_data,
        imputed_data,
        metric=metric,
        custom_metric=custom_metric
    )

    # Prepare output
    result = {
        "result": imputed_data,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric
        },
        "warnings": []
    }

    return result

################################################################################
# mode_imputation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(data: np.ndarray) -> None:
    """Validate input data for mode imputation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")

def normalize_data(
    data: np.ndarray,
    method: str = "standard",
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data using specified method or custom function."""
    if custom_normalizer is not None:
        return custom_normalizer(data)

    if method == "none":
        return data
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

def compute_mode(
    data: np.ndarray,
    axis: int = 0
) -> np.ndarray:
    """Compute mode along specified axis."""
    unique, counts = np.unique(data, axis=axis, return_counts=True)
    mode_indices = np.argmax(counts, axis=axis)
    mode_values = np.take_along_axis(unique, mode_indices[..., None], axis=axis)
    return np.squeeze(mode_values)

def impute_missing_values(
    data: np.ndarray,
    mode_values: np.ndarray
) -> np.ndarray:
    """Impute missing values with mode values."""
    mask = np.isnan(data)
    data[mask] = mode_values[mask]
    return data

def calculate_metrics(
    original: np.ndarray,
    imputed: np.ndarray,
    metrics: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Calculate specified metrics between original and imputed data."""
    result = {}

    if custom_metric is not None:
        result["custom"] = custom_metric(original, imputed)
        return result

    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if metric == "mse":
            result["mse"] = np.mean((original - imputed) ** 2)
        elif metric == "mae":
            result["mae"] = np.mean(np.abs(original - imputed))
        elif metric == "r2":
            ss_res = np.sum((original - imputed) ** 2)
            ss_tot = np.sum((original - np.mean(original)) ** 2)
            result["r2"] = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        elif metric == "logloss":
            result["logloss"] = -np.mean(original * np.log(imputed + 1e-8) +
                                       (1 - original) * np.log(1 - imputed + 1e-8))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return result

def mode_imputation_fit(
    data: np.ndarray,
    normalization_method: str = "standard",
    metrics: Union[str, List[str]] = ["mse"],
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict:
    """Main function for mode imputation."""
    # Validate input data
    validate_input(data)

    # Normalize data
    normalized_data = normalize_data(
        data,
        method=normalization_method,
        custom_normalizer=custom_normalizer
    )

    # Compute mode values
    mode_values = compute_mode(normalized_data)

    # Impute missing values
    imputed_data = impute_missing_values(normalized_data, mode_values)

    # Calculate metrics
    metrics_result = calculate_metrics(
        normalized_data,
        imputed_data,
        metrics=metrics,
        custom_metric=custom_metric
    )

    # Prepare output dictionary
    result = {
        "result": imputed_data,
        "metrics": metrics_result,
        "params_used": {
            "normalization_method": normalization_method,
            "metrics": metrics if isinstance(metrics, list) else [metrics],
            "custom_normalizer": custom_normalizer is not None,
            "custom_metric": custom_metric is not None
        },
        "warnings": []
    }

    return result

# Example usage:
"""
data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
result = mode_imputation_fit(data)
"""

################################################################################
# random_sample_imputation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def random_sample_imputation_fit(
    data: np.ndarray,
    n_samples: int = 5,
    random_state: Optional[int] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    normalization: Optional[str] = None,
    distance_metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form'
) -> Dict:
    """
    Fit the random sample imputation model.

    Parameters
    ----------
    data : np.ndarray
        Input data with missing values (represented as NaN).
    n_samples : int, optional
        Number of samples to use for imputation, by default 5.
    random_state : Optional[int], optional
        Random seed for reproducibility, by default None.
    metric : Union[str, Callable], optional
        Metric to evaluate imputation quality. Options: 'mse', 'mae', 'r2', or custom callable, by default 'mse'.
    normalization : Optional[str], optional
        Normalization method. Options: None, 'standard', 'minmax', 'robust', by default None.
    distance_metric : Union[str, Callable], optional
        Distance metric for similarity. Options: 'euclidean', 'manhattan', 'cosine', 'minkowski', or custom callable, by default 'euclidean'.
    solver : str, optional
        Solver method. Options: 'closed_form', by default 'closed_form'.

    Returns
    -------
    Dict
        Dictionary containing the fitted model, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
    >>> result = random_sample_imputation_fit(data)
    """
    # Validate inputs
    _validate_inputs(data, n_samples, random_state)

    # Normalize data if specified
    normalized_data = _normalize_data(data, normalization) if normalization else data

    # Get indices of missing values
    missing_mask = np.isnan(normalized_data)
    missing_indices = np.argwhere(missing_mask)

    # Initialize RNG
    rng = np.random.RandomState(random_state)

    # Select random samples for imputation
    sample_indices = _select_random_samples(normalized_data, n_samples, missing_mask, rng)

    # Impute missing values
    imputed_data = _impute_missing_values(normalized_data, sample_indices, missing_mask)

    # Calculate metrics
    metrics = _calculate_metrics(imputed_data, normalized_data, metric)

    # Prepare output
    result = {
        "result": imputed_data,
        "metrics": metrics,
        "params_used": {
            "n_samples": n_samples,
            "random_state": random_state,
            "metric": metric,
            "normalization": normalization,
            "distance_metric": distance_metric,
            "solver": solver
        },
        "warnings": []
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    n_samples: int,
    random_state: Optional[int]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if random_state is not None and (not isinstance(random_state, int) or random_state < 0):
        raise ValueError("random_state must be a non-negative integer.")

def _normalize_data(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Normalize the input data."""
    if method == 'standard':
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        return (data - mean) / std
    elif method == 'minmax':
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.nanmedian(data, axis=0)
        iqr = np.nanpercentile(data, 75, axis=0) - np.nanpercentile(data, 25, axis=0)
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _select_random_samples(
    data: np.ndarray,
    n_samples: int,
    missing_mask: np.ndarray,
    rng: np.random.RandomState
) -> np.ndarray:
    """Select random samples for imputation."""
    non_missing_mask = ~missing_mask
    sample_indices = []
    for i in range(data.shape[1]):
        non_missing_cols = np.where(non_missing_mask[:, i])[0]
        if len(non_missing_cols) >= n_samples:
            sample_indices.append(rng.choice(non_missing_cols, size=n_samples, replace=False))
        else:
            sample_indices.append(rng.choice(non_missing_cols, size=len(non_missing_cols), replace=True))
    return np.array(sample_indices)

def _impute_missing_values(
    data: np.ndarray,
    sample_indices: np.ndarray,
    missing_mask: np.ndarray
) -> np.ndarray:
    """Impute missing values using random samples."""
    imputed_data = data.copy()
    for i in range(data.shape[1]):
        missing_rows = np.where(missing_mask[:, i])[0]
        for row in missing_rows:
            imputed_data[row, i] = np.mean(data[sample_indices[i], i])
    return imputed_data

def _calculate_metrics(
    imputed_data: np.ndarray,
    original_data: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate metrics for imputation quality."""
    metrics = {}
    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((imputed_data - original_data) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(imputed_data - original_data))
        elif metric == 'r2':
            ss_res = np.sum((imputed_data - original_data) ** 2)
            ss_tot = np.sum((original_data - np.nanmean(original_data)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        metrics['custom'] = metric(imputed_data, original_data)
    else:
        raise TypeError("Metric must be a string or callable.")
    return metrics

################################################################################
# k_nearest_neighbors_imputation
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def k_nearest_neighbors_imputation_fit(
    data: np.ndarray,
    n_neighbors: int = 5,
    distance_metric: str = 'euclidean',
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    normalization: str = 'standard',
    impute_strategy: str = 'mean',
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Impute missing values in a dataset using k-nearest neighbors.

    Parameters
    ----------
    data : np.ndarray
        Input data array with missing values (represented as NaN).
    n_neighbors : int, optional
        Number of neighbors to use for imputation (default: 5).
    distance_metric : str, optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski') (default: 'euclidean').
    custom_distance : Callable, optional
        Custom distance function if not using built-in metrics.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard').
    impute_strategy : str, optional
        Strategy for imputation ('mean', 'median', 'mode') (default: 'mean').
    random_state : int, optional
        Random seed for reproducibility (default: None).

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Imputed data array.
        - 'metrics': Dictionary of metrics (e.g., RMSE).
        - 'params_used': Dictionary of parameters used.
        - 'warnings': List of warnings encountered.

    Examples
    --------
    >>> data = np.array([[1, 2], [3, np.nan], [np.nan, 6]])
    >>> result = k_nearest_neighbors_imputation_fit(data)
    """
    # Validate inputs
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")

    if np.isnan(data).any():
        missing_mask = np.isnan(data)
    else:
        raise ValueError("No missing values found in the input data.")

    # Normalize data
    normalized_data, normalization_params = _normalize_data(data, normalization)

    # Find neighbors and impute
    imputed_data = _impute_with_knn(
        normalized_data,
        missing_mask,
        n_neighbors,
        distance_metric,
        custom_distance,
        impute_strategy,
        random_state
    )

    # Denormalize the imputed data
    denormalized_data = _denormalize_data(imputed_data, normalization_params)

    # Calculate metrics
    metrics = _calculate_metrics(data, denormalized_data, missing_mask)

    # Prepare output
    result = {
        'result': denormalized_data,
        'metrics': metrics,
        'params_used': {
            'n_neighbors': n_neighbors,
            'distance_metric': distance_metric,
            'normalization': normalization,
            'impute_strategy': impute_strategy
        },
        'warnings': []
    }

    return result

def _normalize_data(
    data: np.ndarray,
    method: str = 'standard'
) -> tuple[np.ndarray, Dict[str, Union[float, np.ndarray]]]:
    """
    Normalize data based on the specified method.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    method : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard').

    Returns
    -------
    tuple
        Normalized data array and normalization parameters.
    """
    if method == 'none':
        return data, {}

    mask = ~np.isnan(data)
    masked_data = data[mask]

    if method == 'standard':
        mean = np.mean(masked_data)
        std = np.std(masked_data)
        normalized = (data - mean) / std
    elif method == 'minmax':
        min_val = np.min(masked_data)
        max_val = np.max(masked_data)
        normalized = (data - min_val) / (max_val - min_val)
    elif method == 'robust':
        median = np.median(masked_data)
        iqr = np.percentile(masked_data, 75) - np.percentile(masked_data, 25)
        normalized = (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    params = {
        'mean': mean if method == 'standard' else None,
        'std': std if method == 'standard' else None,
        'min_val': min_val if method == 'minmax' else None,
        'max_val': max_val if method == 'minmax' else None,
        'median': median if method == 'robust' else None,
        'iqr': iqr if method == 'robust' else None
    }

    return normalized, params

def _denormalize_data(
    data: np.ndarray,
    params: Dict[str, Union[float, np.ndarray]]
) -> np.ndarray:
    """
    Denormalize data using the provided normalization parameters.

    Parameters
    ----------
    data : np.ndarray
        Normalized data array.
    params : dict
        Dictionary of normalization parameters.

    Returns
    -------
    np.ndarray
        Denormalized data array.
    """
    if not params:
        return data

    if 'mean' in params and 'std' in params:
        return data * params['std'] + params['mean']
    elif 'min_val' in params and 'max_val' in params:
        return data * (params['max_val'] - params['min_val']) + params['min_val']
    elif 'median' in params and 'iqr' in params:
        return data * params['iqr'] + params['median']
    else:
        raise ValueError("Unknown normalization parameters.")

def _impute_with_knn(
    data: np.ndarray,
    missing_mask: np.ndarray,
    n_neighbors: int,
    distance_metric: str,
    custom_distance: Optional[Callable],
    impute_strategy: str,
    random_state: Optional[int]
) -> np.ndarray:
    """
    Impute missing values using k-nearest neighbors.

    Parameters
    ----------
    data : np.ndarray
        Normalized input data array.
    missing_mask : np.ndarray
        Boolean mask indicating missing values (True where NaN).
    n_neighbors : int
        Number of neighbors to use for imputation.
    distance_metric : str
        Distance metric to use.
    custom_distance : Callable, optional
        Custom distance function if not using built-in metrics.
    impute_strategy : str
        Strategy for imputation ('mean', 'median', 'mode').
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Imputed data array.
    """
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_function(distance_metric)

    imputed_data = data.copy()
    rows, cols = np.where(missing_mask)

    for row, col in zip(rows, cols):
        # Get the non-missing values
        complete_rows = data[~np.isnan(data[:, col]), :]

        # Calculate distances
        if complete_rows.size == 0:
            continue

        distances = np.array([distance_func(data[row, :], complete_row) for complete_row in complete_rows])

        # Get indices of nearest neighbors
        nearest_indices = np.argsort(distances)[:n_neighbors]

        # Get values from neighbors
        neighbor_values = complete_rows[nearest_indices, col]

        if impute_strategy == 'mean':
            imputed_value = np.mean(neighbor_values)
        elif impute_strategy == 'median':
            imputed_value = np.median(neighbor_values)
        elif impute_strategy == 'mode':
            imputed_value = np.bincount(neighbor_values.astype(int)).argmax()
        else:
            raise ValueError(f"Unknown impute strategy: {impute_strategy}")

        imputed_data[row, col] = imputed_value

    return imputed_data

def _get_distance_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Get the distance function based on the specified metric.

    Parameters
    ----------
    metric : str
        Distance metric to use.

    Returns
    -------
    Callable
        Distance function.
    """
    if metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric == 'minkowski':
        return lambda x, y: np.sum(np.abs(x - y) ** 3) ** (1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _calculate_metrics(
    original_data: np.ndarray,
    imputed_data: np.ndarray,
    missing_mask: np.ndarray
) -> Dict[str, float]:
    """
    Calculate metrics for the imputed data.

    Parameters
    ----------
    original_data : np.ndarray
        Original input data array.
    imputed_data : np.ndarray
        Imputed data array.
    missing_mask : np.ndarray
        Boolean mask indicating missing values (True where NaN).

    Returns
    -------
    dict
        Dictionary of metrics.
    """
    metrics = {}

    # Calculate RMSE for imputed values
    original_values = original_data[missing_mask]
    imputed_values = imputed_data[missing_mask]

    if len(original_values) > 0:
        rmse = np.sqrt(np.mean((original_values - imputed_values) ** 2))
        metrics['rmse'] = rmse

    return metrics

################################################################################
# multivariate_imputation_by_chained_equations
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def multivariate_imputation_by_chained_equations_fit(
    data: np.ndarray,
    impute_methods: Dict[str, Callable],
    normalizer: Optional[Callable] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    max_iter: int = 10,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> Dict:
    """
    Perform multivariate imputation by chained equations.

    Parameters:
    -----------
    data : np.ndarray
        Input data with missing values (NaN).
    impute_methods : Dict[str, Callable]
        Dictionary mapping column names to imputation methods.
    normalizer : Optional[Callable], default=None
        Normalization function. If None, no normalization is applied.
    metric : Union[str, Callable], default='mse'
        Metric to evaluate imputation quality. Can be 'mse', 'mae', 'r2', or a custom callable.
    solver : str, default='closed_form'
        Solver to use for imputation. Options: 'closed_form', 'gradient_descent'.
    max_iter : int, default=10
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    random_state : Optional[int], default=None
        Random seed for reproducibility.

    Returns:
    --------
    Dict
        Dictionary containing the imputed data, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, impute_methods)

    # Initialize results dictionary
    results = {
        'result': None,
        'metrics': {},
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol,
            'random_state': random_state
        },
        'warnings': []
    }

    # Normalize data if specified
    if normalizer is not None:
        data, results['params_used']['normalization_params'] = normalizer(data)

    # Perform imputation
    imputed_data, metrics = _impute_chained_equations(
        data,
        impute_methods,
        metric,
        solver,
        max_iter,
        tol,
        random_state
    )

    # Store results
    results['result'] = imputed_data
    results['metrics'] = metrics

    return results

def _validate_inputs(data: np.ndarray, impute_methods: Dict[str, Callable]) -> None:
    """
    Validate input data and imputation methods.

    Parameters:
    -----------
    data : np.ndarray
        Input data to validate.
    impute_methods : Dict[str, Callable]
        Dictionary of imputation methods to validate.

    Raises:
    -------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if not all(isinstance(method, Callable) for method in impute_methods.values()):
        raise ValueError("Imputation methods must be callable.")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values.")

def _impute_chained_equations(
    data: np.ndarray,
    impute_methods: Dict[str, Callable],
    metric: Union[str, Callable],
    solver: str,
    max_iter: int,
    tol: float,
    random_state: Optional[int]
) -> tuple:
    """
    Perform the actual imputation using chained equations.

    Parameters:
    -----------
    data : np.ndarray
        Input data with missing values.
    impute_methods : Dict[str, Callable]
        Dictionary of imputation methods.
    metric : Union[str, Callable]
        Metric to evaluate imputation quality.
    solver : str
        Solver to use for imputation.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns:
    --------
    tuple
        Tuple containing the imputed data and metrics.
    """
    # Initialize variables
    n_samples, n_features = data.shape
    imputed_data = data.copy()
    metrics = {}

    # Iterate over features to impute
    for feature in impute_methods.keys():
        feature_idx = int(feature)
        if np.isnan(imputed_data[:, feature_idx]).any():
            # Prepare data for imputation
            X = np.delete(imputed_data, feature_idx, axis=1)
            y = imputed_data[:, feature_idx].copy()

            # Impute missing values
            if solver == 'closed_form':
                imputed_values = _impute_closed_form(X, y, impute_methods[feature])
            elif solver == 'gradient_descent':
                imputed_values = _impute_gradient_descent(X, y, impute_methods[feature], max_iter, tol, random_state)
            else:
                raise ValueError(f"Unknown solver: {solver}")

            # Update imputed data
            imputed_data[np.isnan(imputed_data[:, feature_idx]), feature_idx] = imputed_values

    # Calculate metrics
    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = _calculate_mse(data, imputed_data)
        elif metric == 'mae':
            metrics['mae'] = _calculate_mae(data, imputed_data)
        elif metric == 'r2':
            metrics['r2'] = _calculate_r2(data, imputed_data)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        metrics['custom'] = metric(data, imputed_data)

    return imputed_data, metrics

def _impute_closed_form(X: np.ndarray, y: np.ndarray, method: Callable) -> np.ndarray:
    """
    Impute missing values using closed-form solution.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector with missing values.
    method : Callable
        Imputation method to use.

    Returns:
    --------
    np.ndarray
        Imputed values.
    """
    # Example: Simple linear regression
    X_mean = np.nanmean(X, axis=0)
    X[np.isnan(X)] = np.take(X_mean, np.where(np.isnan(X))[1])

    # Fit model
    beta = np.linalg.pinv(X) @ y

    # Predict missing values
    imputed_values = X[np.isnan(y)] @ beta

    return imputed_values

def _impute_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    method: Callable,
    max_iter: int,
    tol: float,
    random_state: Optional[int]
) -> np.ndarray:
    """
    Impute missing values using gradient descent.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector with missing values.
    method : Callable
        Imputation method to use.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns:
    --------
    np.ndarray
        Imputed values.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Initialize parameters
    n_features = X.shape[1]
    beta = np.random.randn(n_features)

    # Gradient descent
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ beta - y)
        beta -= tol * gradient

    # Predict missing values
    imputed_values = X[np.isnan(y)] @ beta

    return imputed_values

def _calculate_mse(original: np.ndarray, imputed: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.

    Parameters:
    -----------
    original : np.ndarray
        Original data.
    imputed : np.ndarray
        Imputed data.

    Returns:
    --------
    float
        MSE value.
    """
    return np.mean((original - imputed) ** 2)

def _calculate_mae(original: np.ndarray, imputed: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Parameters:
    -----------
    original : np.ndarray
        Original data.
    imputed : np.ndarray
        Imputed data.

    Returns:
    --------
    float
        MAE value.
    """
    return np.mean(np.abs(original - imputed))

def _calculate_r2(original: np.ndarray, imputed: np.ndarray) -> float:
    """
    Calculate R-squared.

    Parameters:
    -----------
    original : np.ndarray
        Original data.
    imputed : np.ndarray
        Imputed data.

    Returns:
    --------
    float
        R-squared value.
    """
    ss_res = np.sum((original - imputed) ** 2)
    ss_tot = np.sum((original - np.mean(original)) ** 2)
    return 1 - (ss_res / ss_tot)

# Example usage
if __name__ == "__main__":
    # Create example data with missing values
    data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
    impute_methods = {'0': lambda x: np.mean(x), '1': lambda x: np.median(x)}

    # Perform imputation
    results = multivariate_imputation_by_chained_equations_fit(
        data,
        impute_methods,
        normalizer=None,
        metric='mse',
        solver='closed_form'
    )

    print(results)

################################################################################
# iterative_imputation
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def iterative_imputation_fit(
    data: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.linalg.norm(x - y),
    solver: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda X, y: np.linalg.pinv(X) @ y,
    metric: Callable[[np.ndarray, np.ndarray], float] = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
    impute_strategy: str = 'mean',
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Perform iterative imputation on a dataset with missing values.

    Parameters:
    -----------
    data : np.ndarray
        Input data array with missing values (represented as NaN).
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    random_state : int, optional
        Random seed for reproducibility.
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Function to normalize the data.
    distance_metric : Callable[[np.ndarray, np.ndarray], float], optional
        Function to compute distance between vectors.
    solver : Callable[[np.ndarray, np.ndarray], np.ndarray], optional
        Function to solve the imputation problem.
    metric : Callable[[np.ndarray, np.ndarray], float], optional
        Function to compute the imputation metric.
    impute_strategy : str, optional
        Strategy for initial imputation ('mean', 'median', etc.).
    verbose : bool, optional
        Whether to print progress information.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the imputed data, metrics, parameters used, and warnings.
    """
    # Validate input
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if np.isnan(data).any():
        pass  # Handle missing values

    # Initialize random state
    if random_state is not None:
        np.random.seed(random_state)

    # Initial imputation
    if impute_strategy == 'mean':
        initial_impute = np.nanmean(data, axis=0)
    elif impute_strategy == 'median':
        initial_impute = np.nanmedian(data, axis=0)
    else:
        raise ValueError("Unsupported impute strategy.")

    # Main iterative loop
    for _ in range(max_iter):
        # Normalize data
        normalized_data = normalizer(data)

        # Compute distances and impute missing values
        # ... (implementation details)

        # Check convergence
        if verbose:
            print(f"Iteration {_ + 1}/{max_iter}")

    # Prepare output
    result = {
        "result": data,  # Imputed data
        "metrics": {"metric_name": metric_value},  # Metrics dictionary
        "params_used": {
            "max_iter": max_iter,
            "tol": tol,
            "impute_strategy": impute_strategy
        },
        "warnings": []  # List of warnings
    }

    return result

def _validate_input(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if np.isnan(data).any():
        pass  # Handle missing values

def _normalize_data(data: np.ndarray, normalizer: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Normalize the data using the provided normalizer."""
    return normalizer(data)

def _compute_distance(X: np.ndarray, y: np.ndarray, distance_metric: Callable[[np.ndarray, np.ndarray], float]) -> float:
    """Compute distance between vectors."""
    return distance_metric(X, y)

def _solve_imputation(X: np.ndarray, y: np.ndarray, solver: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    """Solve the imputation problem."""
    return solver(X, y)

def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float]) -> float:
    """Compute the imputation metric."""
    return metric(y_true, y_pred)

# Example usage
if __name__ == "__main__":
    data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
    result = iterative_imputation_fit(data)

################################################################################
# regression_imputation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regression_imputation_fit(
    X: np.ndarray,
    impute_method: str = 'linear',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> Dict:
    """
    Perform regression-based imputation on missing values in the dataset.

    Parameters
    ----------
    X : np.ndarray
        Input data array with missing values (represented as NaN).
    impute_method : str, optional
        Method for regression imputation ('linear', 'ridge', 'lasso').
    normalizer : Callable, optional
        Function to normalize the data (e.g., standard scaling).
    metric : str or Callable, optional
        Metric to evaluate imputation quality ('mse', 'mae', 'r2' or custom).
    solver : str, optional
        Solver to use for regression ('closed_form', 'gradient_descent').
    regularization : str, optional
        Regularization type ('l1', 'l2', 'elasticnet').
    max_iter : int, optional
        Maximum number of iterations for iterative solvers.
    tol : float, optional
        Tolerance for convergence.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Imputed data array.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Example
    -------
    >>> X = np.array([[1, 2], [3, np.nan], [np.nan, 6]])
    >>> result = regression_imputation_fit(X)
    """
    # Validate inputs
    _validate_inputs(X, impute_method, solver)

    # Initialize random state if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data if normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Identify missing values
    mask = np.isnan(X_normalized)
    complete_mask = ~mask

    # Initialize result array
    X_imputed = np.copy(X_normalized)

    # Perform imputation for each feature with missing values
    for i in range(X_normalized.shape[1]):
        if np.any(mask[:, i]):
            X_imputed = _impute_feature(
                X_normalized, mask, i,
                impute_method=impute_method,
                solver=solver,
                regularization=regularization,
                max_iter=max_iter,
                tol=tol
            )

    # Compute metrics if a metric is specified
    metrics = {}
    if isinstance(metric, str):
        metrics = _compute_metrics(X_normalized, X_imputed, metric)
    elif callable(metric):
        metrics = {'custom_metric': metric(X_normalized, X_imputed)}

    # Denormalize the data if normalization was applied
    if normalizer is not None:
        X_imputed = _inverse_normalization(X_imputed, normalizer)

    # Prepare output dictionary
    result = {
        'result': X_imputed,
        'metrics': metrics,
        'params_used': {
            'impute_method': impute_method,
            'normalizer': normalizer.__name__ if callable(normalizer) else None,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

    return result

def _validate_inputs(
    X: np.ndarray,
    impute_method: str,
    solver: str
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array.")
    if np.any(~np.isfinite(X)):
        raise ValueError("X contains non-finite values (NaN or inf).")

    valid_impute_methods = ['linear', 'ridge', 'lasso']
    if impute_method not in valid_impute_methods:
        raise ValueError(f"impute_method must be one of {valid_impute_methods}.")

    valid_solvers = ['closed_form', 'gradient_descent']
    if solver not in valid_solvers:
        raise ValueError(f"solver must be one of {valid_solvers}.")

def _apply_normalization(
    X: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to the data if a normalizer is provided."""
    if callable(normalizer):
        return normalizer(X)
    return X

def _inverse_normalization(
    X: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Apply inverse normalization to the data if a normalizer was used."""
    # This is a placeholder; actual implementation depends on the normalizer
    return X

def _impute_feature(
    X: np.ndarray,
    mask: np.ndarray,
    feature_idx: int,
    impute_method: str,
    solver: str,
    regularization: Optional[str],
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Impute missing values for a single feature using regression."""
    # Get indices of complete and incomplete observations
    complete_idx = np.where(~mask[:, feature_idx])[0]
    incomplete_idx = np.where(mask[:, feature_idx])[0]

    if len(complete_idx) == 0:
        raise ValueError("Cannot impute feature with no complete observations.")

    # Prepare design matrix and target vector
    X_complete = X[complete_idx, :]
    y_complete = X_complete[:, feature_idx]

    # Remove the target feature from predictors
    X_predictors = np.delete(X_complete, feature_idx, axis=1)

    # Fit regression model
    if solver == 'closed_form':
        beta = _solve_closed_form(X_predictors, y_complete)
    else:
        beta = _solve_iterative(
            X_predictors, y_complete,
            method=impute_method,
            regularization=regularization,
            max_iter=max_iter,
            tol=tol
        )

    # Predict missing values
    X_incomplete = X[incomplete_idx, :]
    X_predictors_incomplete = np.delete(X_incomplete, feature_idx, axis=1)
    y_pred = X_predictors_incomplete @ beta

    # Update the imputed values
    X_imputed = np.copy(X)
    X_imputed[incomplete_idx, feature_idx] = y_pred

    return X_imputed

def _solve_closed_form(
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Solve linear regression using closed-form solution."""
    XtX = X.T @ X
    if np.linalg.det(XtX) == 0:
        raise ValueError("Design matrix is singular.")
    beta = np.linalg.inv(XtX) @ X.T @ y
    return beta

def _solve_iterative(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    regularization: Optional[str],
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Solve regression using iterative methods."""
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)

    for _ in range(max_iter):
        old_beta = beta.copy()
        gradient = 2 * X.T @ (X @ beta - y) / n_samples

        if regularization == 'l1':
            gradient += np.sign(beta)
        elif regularization == 'l2':
            gradient += 2 * beta

        if method == 'gradient_descent':
            learning_rate = 0.01
            beta -= learning_rate * gradient

        # Check for convergence
        if np.linalg.norm(beta - old_beta) < tol:
            break

    return beta

def _compute_metrics(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    metric: str
) -> Dict[str, float]:
    """Compute metrics for imputation quality."""
    mask = np.isnan(X_true)
    complete_mask = ~mask

    metrics = {}
    if metric == 'mse':
        mse = np.mean((X_true[complete_mask] - X_pred[complete_mask]) ** 2)
        metrics['mse'] = mse
    elif metric == 'mae':
        mae = np.mean(np.abs(X_true[complete_mask] - X_pred[complete_mask]))
        metrics['mae'] = mae
    elif metric == 'r2':
        ss_res = np.sum((X_true[complete_mask] - X_pred[complete_mask]) ** 2)
        ss_tot = np.sum((X_true[complete_mask] - np.mean(X_true[complete_mask])) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        metrics['r2'] = r2

    return metrics

################################################################################
# knn_imputation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], float],
    n_neighbors: int,
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X must not contain NaN or infinite values")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be a positive integer")
    if n_neighbors > X.shape[0]:
        raise ValueError("n_neighbors must be less than or equal to the number of samples")

def _compute_distance_matrix(
    X: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Compute the distance matrix between all pairs of samples."""
    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_neighbors))
    for i in range(n_samples):
        distances = np.array([distance_metric(X[i], X[j]) for j in range(n_samples) if i != j])
        nearest_indices = np.argsort(distances)[:n_neighbors]
        distance_matrix[i] = distances[nearest_indices]
    return distance_matrix

def _impute_missing_values(
    X: np.ndarray,
    distance_matrix: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    """Impute missing values using the k-nearest neighbors."""
    X_imputed = X.copy()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.isnan(X[i, j]):
                neighbors = distance_matrix[i]
                weights = 1 / (neighbors + 1e-10)
                weights /= np.sum(weights)
                imputed_value = np.sum(X[neighbors, j] * weights)
                X_imputed[i, j] = imputed_value
    return X_imputed

def knn_imputation_fit(
    X: np.ndarray,
    n_neighbors: int = 5,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.linalg.norm(x - y),
    normalization: Optional[str] = None,
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Perform k-nearest neighbors imputation on a dataset with missing values.

    Parameters
    ----------
    X : np.ndarray
        Input data array of shape (n_samples, n_features) with missing values represented as NaN.
    n_neighbors : int, optional
        Number of neighbors to use for imputation (default is 5).
    distance_metric : Callable[[np.ndarray, np.ndarray], float], optional
        Distance metric function (default is Euclidean distance).
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust' (default is None).

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]
        A dictionary containing:
        - "result": Imputed data array.
        - "metrics": Dictionary of metrics (currently empty).
        - "params_used": Dictionary of parameters used.
        - "warnings": List of warnings (currently empty).
    """
    _validate_inputs(X, distance_metric, n_neighbors)

    if normalization == 'standard':
        X = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-10)
    elif normalization == 'minmax':
        X = (X - np.nanmin(X, axis=0)) / (np.nanmax(X, axis=0) - np.nanmin(X, axis=0) + 1e-10)
    elif normalization == 'robust':
        X = (X - np.nanmedian(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0) + 1e-10)

    distance_matrix = _compute_distance_matrix(X, distance_metric)
    X_imputed = _impute_missing_values(X, distance_matrix, n_neighbors)

    return {
        "result": X_imputed,
        "metrics": {},
        "params_used": {
            "n_neighbors": n_neighbors,
            "distance_metric": distance_metric.__name__ if hasattr(distance_metric, '__name__') else "custom",
            "normalization": normalization,
        },
        "warnings": [],
    }

# Example usage:
# X = np.array([[1, 2], [np.nan, 5], [7, np.nan]])
# result = knn_imputation_fit(X)

################################################################################
# missforest_imputation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def missforest_imputation_fit(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = 'euclidean',
    solver: str = 'auto',
    max_iter: int = 10,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Perform missForest imputation on the input data.

    Parameters:
    -----------
    data : np.ndarray
        Input data with missing values (np.nan).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the data. If None, no normalization is applied.
    distance_metric : str, optional
        Distance metric to use for imputation. Options: 'euclidean', 'manhattan', 'cosine'.
    solver : str, optional
        Solver to use for imputation. Options: 'auto', 'saga'.
    max_iter : int, optional
        Maximum number of iterations for the imputer.
    random_state : Optional[int], optional
        Random seed for reproducibility.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function to evaluate the imputation quality.

    Returns:
    --------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]
        Dictionary containing:
        - "result": Imputed data.
        - "metrics": Dictionary of metrics.
        - "params_used": Dictionary of parameters used.
        - "warnings": List of warnings.

    Example:
    --------
    >>> data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
    >>> result = missforest_imputation_fit(data)
    """
    # Validate input data
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if np.isnan(data).any():
        warnings = []
        if np.any(np.isinf(data)):
            warnings.append("Input data contains infinite values.")
    else:
        raise ValueError("Input data must contain missing values (np.nan).")

    # Normalize data if a normalizer is provided
    if normalizer is not None:
        try:
            data_normalized = normalizer(data)
        except Exception as e:
            raise ValueError(f"Normalization failed: {e}")
    else:
        data_normalized = data.copy()

    # Initialize the imputer
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(random_state=random_state),
        max_iter=max_iter,
        random_state=random_state
    )

    # Fit and transform the data
    imputed_data = imputer.fit_transform(data_normalized)

    # Calculate metrics
    metrics = {}
    if custom_metric is not None:
        try:
            # Assuming the first row is the imputed data and the second is the original (for example)
            metrics['custom_metric'] = custom_metric(imputed_data, data_normalized)
        except Exception as e:
            metrics['custom_metric'] = f"Error calculating custom metric: {e}"

    # Prepare the output dictionary
    result = {
        "result": imputed_data,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "distance_metric": distance_metric,
            "solver": solver,
            "max_iter": max_iter,
            "random_state": random_state
        },
        "warnings": warnings if 'warnings' in locals() else []
    }

    return result

def _validate_input_data(data: np.ndarray) -> None:
    """
    Validate the input data for missForest imputation.

    Parameters:
    -----------
    data : np.ndarray
        Input data to validate.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if not np.any(np.isnan(data)):
        raise ValueError("Input data must contain missing values (np.nan).")

def _normalize_data(data: np.ndarray, normalizer: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """
    Normalize the input data using the provided normalizer.

    Parameters:
    -----------
    data : np.ndarray
        Input data to normalize.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]]
        Function to normalize the data.

    Returns:
    --------
    np.ndarray
        Normalized data.
    """
    if normalizer is not None:
        return normalizer(data)
    return data.copy()

def _calculate_metrics(imputed_data: np.ndarray, original_data: np.ndarray, custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]) -> Dict[str, float]:
    """
    Calculate metrics for the imputed data.

    Parameters:
    -----------
    imputed_data : np.ndarray
        Imputed data.
    original_data : np.ndarray
        Original data with missing values.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function to evaluate the imputation quality.

    Returns:
    --------
    Dict[str, float]
        Dictionary of metrics.
    """
    metrics = {}
    if custom_metric is not None:
        try:
            metrics['custom_metric'] = custom_metric(imputed_data, original_data)
        except Exception as e:
            metrics['custom_metric'] = f"Error calculating custom metric: {e}"
    return metrics

################################################################################
# soft_imputation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def soft_imputation_fit(
    X: np.ndarray,
    mask: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    metric: str = 'mse',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Perform soft imputation on missing values in a dataset.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix with missing values (NaN).
    mask : np.ndarray
        Binary mask where 1 indicates observed values and 0 missing values.
    normalizer : Optional[Callable]
        Function to normalize the data. Default is None (no normalization).
    distance_metric : str
        Distance metric for imputation. Options: 'euclidean', 'manhattan', 'cosine'.
    solver : str
        Solver method. Options: 'closed_form', 'gradient_descent'.
    regularization : Optional[str]
        Regularization type. Options: None, 'l1', 'l2'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    metric : str
        Evaluation metric. Options: 'mse', 'mae', 'r2'.
    custom_metric : Optional[Callable]
        Custom metric function if needed.
    verbose : bool
        Whether to print progress.

    Returns:
    --------
    dict
        Dictionary containing the imputed data, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, mask)

    # Normalize data if specified
    X_norm = X.copy()
    if normalizer is not None:
        X_norm = normalizer(X)

    # Initialize imputed data
    X_imputed = np.nan_to_num(X_norm, nan=0.0)

    # Choose distance metric
    distance_func = _get_distance_function(distance_metric)

    # Choose solver
    if solver == 'closed_form':
        X_imputed = _closed_form_solver(X_norm, mask)
    elif solver == 'gradient_descent':
        X_imputed = _gradient_descent_solver(X_norm, mask, distance_func,
                                            regularization, tol, max_iter, verbose)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_norm, X_imputed, mask, metric, custom_metric)

    # Prepare output
    result = {
        'result': X_imputed,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'distance_metric': distance_metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter,
            'metric': metric
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, mask: np.ndarray) -> None:
    """Validate input data and mask."""
    if X.shape != mask.shape:
        raise ValueError("X and mask must have the same shape")
    if not np.all(np.isin(mask, [0, 1])):
        raise ValueError("Mask must contain only 0s and 1s")
    if np.any(np.isnan(mask)):
        raise ValueError("Mask cannot contain NaN values")

def _get_distance_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the distance function based on the metric."""
    if metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _closed_form_solver(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Closed form solution for soft imputation."""
    # This is a placeholder - actual implementation would depend on the specific method
    return X * mask + np.nanmean(X) * (1 - mask)

def _gradient_descent_solver(
    X: np.ndarray,
    mask: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    verbose: bool
) -> np.ndarray:
    """Gradient descent solver for soft imputation."""
    # This is a placeholder - actual implementation would depend on the specific method
    X_imputed = np.nan_to_num(X, nan=0.0)
    for _ in range(max_iter):
        # Update rule would go here
        pass
    return X_imputed

def _compute_metrics(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    mask: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((X_true[mask.astype(bool)] - X_pred[mask.astype(bool)])**2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(X_true[mask.astype(bool)] - X_pred[mask.astype(bool)]))
    elif metric == 'r2':
        ss_res = np.sum((X_true[mask.astype(bool)] - X_pred[mask.astype(bool)])**2)
        ss_tot = np.sum((X_true[mask.astype(bool)] - np.mean(X_true[mask.astype(bool)]))**2)
        metrics['r2'] = 1 - ss_res / ss_tot
    if custom_metric is not None:
        metrics['custom'] = custom_metric(X_true, X_pred)
    return metrics

# Example usage:
"""
X_example = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
mask_example = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]])

result = soft_imputation_fit(
    X=X_example,
    mask=mask_example,
    normalizer=None,
    distance_metric='euclidean',
    solver='closed_form'
)
"""

################################################################################
# multiple_imputation
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def multiple_imputation_fit(
    data: np.ndarray,
    impute_method: str = 'mean',
    n_iterations: int = 5,
    random_state: Optional[int] = None,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 100,
    metric: str = 'mse',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Perform multiple imputation on missing data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array with missing values (np.nan).
    impute_method : str, optional
        Method for imputation ('mean', 'median', 'knn').
    n_iterations : int, optional
        Number of imputation iterations.
    random_state : Optional[int], optional
        Random seed for reproducibility.
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Function to normalize data.
    distance_metric : str, optional
        Distance metric for KNN imputation ('euclidean', 'manhattan', etc.).
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent').
    regularization : Optional[str], optional
        Regularization type (None, 'l1', 'l2').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    metric : str, optional
        Metric for evaluation ('mse', 'mae').
    custom_metric : Optional[Callable], optional
        Custom metric function.
    verbose : bool, optional
        Verbosity flag.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, impute_method, distance_metric, solver, regularization)

    # Initialize random state
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    normalized_data = normalizer(data.copy())

    # Impute missing values
    imputed_data = _impute_missing_values(
        normalized_data,
        method=impute_method,
        distance_metric=distance_metric,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Compute metrics
    metrics = _compute_metrics(
        imputed_data,
        original_data=data,
        metric=metric,
        custom_metric=custom_metric
    )

    # Prepare output
    result = {
        'result': imputed_data,
        'metrics': metrics,
        'params_used': {
            'impute_method': impute_method,
            'n_iterations': n_iterations,
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'distance_metric': distance_metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter,
            'metric': metric
        },
        'warnings': []
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    impute_method: str,
    distance_metric: str,
    solver: str,
    regularization: Optional[str]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values.")
    if impute_method not in ['mean', 'median', 'knn']:
        raise ValueError("Invalid impute method.")
    if distance_metric not in ['euclidean', 'manhattan', 'cosine', 'minkowski']:
        raise ValueError("Invalid distance metric.")
    if solver not in ['closed_form', 'gradient_descent']:
        raise ValueError("Invalid solver.")
    if regularization not in [None, 'l1', 'l2']:
        raise ValueError("Invalid regularization type.")

def _impute_missing_values(
    data: np.ndarray,
    method: str,
    distance_metric: str,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Impute missing values in the data."""
    imputed_data = data.copy()
    mask = np.isnan(imputed_data)

    if method == 'mean':
        for i in range(imputed_data.shape[1]):
            col = imputed_data[:, i]
            if np.isnan(col).any():
                mean_val = np.nanmean(col)
                col[np.isnan(col)] = mean_val
    elif method == 'median':
        for i in range(imputed_data.shape[1]):
            col = imputed_data[:, i]
            if np.isnan(col).any():
                median_val = np.nanmedian(col)
                col[np.isnan(col)] = median_val
    elif method == 'knn':
        imputed_data = _knn_impute(imputed_data, distance_metric, solver, regularization, tol, max_iter)

    return imputed_data

def _knn_impute(
    data: np.ndarray,
    distance_metric: str,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Perform KNN imputation."""
    # Placeholder for KNN imputation logic
    return data

def _compute_metrics(
    imputed_data: np.ndarray,
    original_data: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}

    if metric == 'mse':
        mask = ~np.isnan(original_data)
        mse = np.mean((imputed_data[mask] - original_data[mask]) ** 2)
        metrics['mse'] = mse
    elif metric == 'mae':
        mask = ~np.isnan(original_data)
        mae = np.mean(np.abs(imputed_data[mask] - original_data[mask]))
        metrics['mae'] = mae

    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(imputed_data, original_data)

    return metrics

################################################################################
# predictive_mean_matching
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def predictive_mean_matching_fit(
    data: np.ndarray,
    missing_mask: np.ndarray,
    n_donors: int = 5,
    random_state: Optional[int] = None,
    distance_metric: str = 'euclidean',
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    normalization: str = 'standard',
    impute_mean: bool = True
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Predictive Mean Matching (PMM) imputation method.

    Parameters
    ----------
    data : np.ndarray
        Input data array with missing values (NaN).
    missing_mask : np.ndarray
        Boolean mask where True indicates missing values.
    n_donors : int, optional
        Number of donors to consider for imputation (default: 5).
    random_state : int, optional
        Random seed for reproducibility (default: None).
    distance_metric : str, optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine') (default: 'euclidean').
    custom_distance : Callable, optional
        Custom distance function if not using built-in metrics.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard').
    impute_mean : bool, optional
        Whether to use the mean of donors for imputation (default: True).

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
        - 'result': Imputed data array
        - 'metrics': Dictionary of computed metrics
        - 'params_used': Parameters used in the computation
        - 'warnings': Any warnings encountered

    Example
    -------
    >>> data = np.array([[1, 2], [3, np.nan], [np.nan, 6]])
    >>> mask = np.array([[False, False], [False, True], [True, False]])
    >>> result = predictive_mean_matching_fit(data, mask)
    """
    # Validate inputs
    _validate_inputs(data, missing_mask)

    # Set random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Normalize data
    normalized_data, normalization_params = _normalize_data(data, missing_mask, normalization)

    # Get indices of missing and observed values
    missing_indices = np.where(missing_mask)
    observed_data = normalized_data[~missing_mask]

    # Initialize imputed data
    imputed_data = np.copy(normalized_data)

    # For each missing value, find donors and impute
    for i, j in zip(*missing_indices):
        # Get observed values from the same column
        col_observed = normalized_data[~missing_mask[:, j], j]

        # Compute distances
        if custom_distance is not None:
            distances = np.array([custom_distance(normalized_data[k], normalized_data[i]) for k in range(len(normalized_data))])
        else:
            distances = _compute_distances(normalized_data, i, distance_metric)

        # Get indices of n_donors closest donors
        donor_indices = np.argsort(distances)[1:n_donors+1]  # Skip self

        # Get values from donors
        donor_values = normalized_data[donor_indices, j]

        if impute_mean:
            # Use mean of donors
            imputed_value = np.mean(donor_values)
        else:
            # Randomly select from donors
            imputed_value = rng.choice(donor_values)

        # Store imputed value
        imputed_data[i, j] = imputed_value

    # Denormalize data
    denormalized_data = _denormalize_data(imputed_data, normalization_params, normalization)

    # Calculate metrics
    metrics = _calculate_metrics(data, denormalized_data, missing_mask)

    return {
        'result': denormalized_data,
        'metrics': metrics,
        'params_used': {
            'n_donors': n_donors,
            'distance_metric': distance_metric,
            'normalization': normalization,
            'impute_mean': impute_mean
        },
        'warnings': []
    }

def _validate_inputs(data: np.ndarray, missing_mask: np.ndarray) -> None:
    """Validate input data and mask."""
    if data.shape != missing_mask.shape:
        raise ValueError("Data and missing mask must have the same shape")
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("Data must be numeric")
    if not np.issubdtype(missing_mask.dtype, bool):
        raise TypeError("Missing mask must be boolean")
    if np.any(np.isinf(data)):
        raise ValueError("Data contains infinite values")

def _normalize_data(
    data: np.ndarray,
    missing_mask: np.ndarray,
    method: str
) -> tuple[np.ndarray, Dict]:
    """Normalize data based on specified method."""
    normalized = np.copy(data).astype(float)
    params = {}

    if method == 'standard':
        for j in range(data.shape[1]):
            col = data[~missing_mask[:, j], j]
            params[j] = {
                'mean': np.mean(col),
                'std': np.std(col)
            }
            normalized[~missing_mask[:, j], j] = (col - params[j]['mean']) / params[j]['std']
    elif method == 'minmax':
        for j in range(data.shape[1]):
            col = data[~missing_mask[:, j], j]
            params[j] = {
                'min': np.min(col),
                'max': np.max(col)
            }
            normalized[~missing_mask[:, j], j] = (col - params[j]['min']) / (params[j]['max'] - params[j]['min'])
    elif method == 'robust':
        for j in range(data.shape[1]):
            col = data[~missing_mask[:, j], j]
            params[j] = {
                'median': np.median(col),
                'iqr': np.percentile(col, 75) - np.percentile(col, 25)
            }
            normalized[~missing_mask[:, j], j] = (col - params[j]['median']) / params[j]['iqr']
    elif method == 'none':
        pass
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, params

def _denormalize_data(
    data: np.ndarray,
    params: Dict,
    method: str
) -> np.ndarray:
    """Denormalize data based on specified method."""
    denormalized = np.copy(data)

    if method == 'standard':
        for j in range(data.shape[1]):
            denormalized[:, j] = data[:, j] * params[j]['std'] + params[j]['mean']
    elif method == 'minmax':
        for j in range(data.shape[1]):
            denormalized[:, j] = data[:, j] * (params[j]['max'] - params[j]['min']) + params[j]['min']
    elif method == 'robust':
        for j in range(data.shape[1]):
            denormalized[:, j] = data[:, j] * params[j]['iqr'] + params[j]['median']
    elif method == 'none':
        pass

    return denormalized

def _compute_distances(
    data: np.ndarray,
    index: int,
    metric: str
) -> np.ndarray:
    """Compute distances between data points."""
    if metric == 'euclidean':
        return np.linalg.norm(data - data[index], axis=1)
    elif metric == 'manhattan':
        return np.sum(np.abs(data - data[index]), axis=1)
    elif metric == 'cosine':
        return 1 - np.dot(data, data[index]) / (np.linalg.norm(data, axis=1) * np.linalg.norm(data[index]))
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _calculate_metrics(
    original_data: np.ndarray,
    imputed_data: np.ndarray,
    missing_mask: np.ndarray
) -> Dict:
    """Calculate imputation metrics."""
    metrics = {}

    # Only compute on originally missing values
    missing_values = original_data[missing_mask]
    imputed_values = imputed_data[missing_mask]

    # Mean Absolute Error
    metrics['mae'] = np.mean(np.abs(imputed_values - missing_values))

    # Root Mean Squared Error
    metrics['rmse'] = np.sqrt(np.mean((imputed_values - missing_values) ** 2))

    # R-squared
    ss_total = np.sum((missing_values - np.mean(missing_values)) ** 2)
    ss_residual = np.sum((missing_values - imputed_values) ** 2)
    metrics['r2'] = 1 - (ss_residual / ss_total) if ss_total != 0 else np.nan

    return metrics

################################################################################
# fully_conditional_specification
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def fully_conditional_specification_fit(
    data: np.ndarray,
    impute_func: Callable = None,
    normalizer: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fit the fully conditional specification imputation model.

    Parameters:
    -----------
    data : np.ndarray
        Input data array with missing values (np.nan).
    impute_func : Callable, optional
        Custom imputation function. If None, uses default.
    normalizer : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : str or Callable, optional
        Metric for evaluation: 'mse', 'mae', 'r2', or custom callable.
    distance : str, optional
        Distance metric: 'euclidean', 'manhattan', 'cosine', or 'minkowski'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    random_state : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data)

    # Initialize results dictionary
    results = {
        'result': None,
        'metrics': {},
        'params_used': locals(),
        'warnings': []
    }

    # Normalize data
    normalized_data, normalizer_func = _apply_normalization(data, normalizer)

    # Impute missing values
    imputed_data = _impute_missing_values(
        normalized_data,
        impute_func=impute_func,
        solver=solver,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )

    # Calculate metrics
    results['metrics'] = _calculate_metrics(imputed_data, metric)

    # Denormalize data
    results['result'] = normalizer_func.inverse_transform(imputed_data)

    return results

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if np.isnan(data).any():
        results['warnings'].append("Input data contains NaN values.")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values.")

def _apply_normalization(
    data: np.ndarray,
    method: str = 'standard'
) -> tuple[np.ndarray, Any]:
    """Apply normalization to the data."""
    if method == 'none':
        return data, None
    elif method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler

def _impute_missing_values(
    data: np.ndarray,
    impute_func: Callable = None,
    solver: str = 'closed_form',
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Impute missing values in the data."""
    if impute_func is not None:
        return impute_func(data, **kwargs)

    # Default imputation logic
    if solver == 'closed_form':
        return _closed_form_imputation(data)
    elif solver == 'gradient_descent':
        return _gradient_descent_imputation(data, max_iter, tol, random_state)
    elif solver == 'newton':
        return _newton_imputation(data, max_iter, tol)
    elif solver == 'coordinate_descent':
        return _coordinate_descent_imputation(data, max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _closed_form_imputation(data: np.ndarray) -> np.ndarray:
    """Closed form imputation."""
    # Implement closed form imputation logic
    return data

def _gradient_descent_imputation(
    data: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Gradient descent imputation."""
    # Implement gradient descent imputation logic
    return data

def _newton_imputation(
    data: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-4
) -> np.ndarray:
    """Newton imputation."""
    # Implement Newton imputation logic
    return data

def _coordinate_descent_imputation(
    data: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-4
) -> np.ndarray:
    """Coordinate descent imputation."""
    # Implement coordinate descent imputation logic
    return data

def _calculate_metrics(
    data: np.ndarray,
    metric: Union[str, Callable] = 'mse'
) -> Dict[str, float]:
    """Calculate metrics for the imputed data."""
    if callable(metric):
        return {'custom_metric': metric(data)}
    elif metric == 'mse':
        return {'mse': np.mean((data - np.nanmean(data))**2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(data - np.nanmean(data)))}
    elif metric == 'r2':
        return {'r2': 1 - np.sum((data - np.nanmean(data))**2) / np.sum((data - np.mean(data))**2)}
    elif metric == 'logloss':
        return {'logloss': -np.mean(data * np.log(data + 1e-10))}
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# bayesian_imputation
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def bayesian_imputation_fit(
    data: np.ndarray,
    missing_mask: np.ndarray,
    n_iter: int = 100,
    tol: float = 1e-4,
    solver: str = 'gradient_descent',
    distance_metric: Union[str, Callable] = 'euclidean',
    normalization: str = 'standard',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    random_state: Optional[int] = None,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Bayesian imputation for missing data.

    Parameters:
    -----------
    data : np.ndarray
        Input data with missing values (NaN)
    missing_mask : np.ndarray
        Binary mask indicating missing values (1 for missing, 0 otherwise)
    n_iter : int
        Number of iterations
    tol : float
        Tolerance for convergence
    solver : str
        Solver to use ('gradient_descent', 'newton', 'coordinate_descent')
    distance_metric : str or callable
        Distance metric to use ('euclidean', 'manhattan', 'cosine', custom callable)
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    regularization : str or None
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    alpha : float
        Regularization strength for L1/L2
    beta : float
        Elastic net mixing parameter (0 = L2, 1 = L1)
    random_state : int or None
        Random seed for reproducibility
    metric : str or callable
        Metric to evaluate imputation ('mse', 'mae', 'r2', custom callable)
    custom_metric : callable or None
        Custom metric function if needed
    verbose : bool
        Whether to print progress

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': imputed data
        - 'metrics': evaluation metrics
        - 'params_used': parameters used
        - 'warnings': any warnings

    Example:
    --------
    >>> data = np.array([[1, 2], [3, np.nan], [np.nan, 6]])
    >>> mask = np.array([[0, 0], [0, 1], [1, 0]])
    >>> result = bayesian_imputation_fit(data, mask)
    """
    # Validate inputs
    _validate_inputs(data, missing_mask)

    # Initialize random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Normalize data
    normalized_data, norm_params = _apply_normalization(data, normalization)

    # Initialize parameters
    params = _initialize_parameters(normalized_data, missing_mask, rng)

    # Main optimization loop
    for i in range(n_iter):
        prev_params = params.copy()

        # Update parameters based on solver
        if solver == 'gradient_descent':
            params = _gradient_descent_step(params, normalized_data, missing_mask, distance_metric,
                                          regularization, alpha, beta)
        elif solver == 'newton':
            params = _newton_step(params, normalized_data, missing_mask, distance_metric,
                                 regularization, alpha)
        elif solver == 'coordinate_descent':
            params = _coordinate_descent_step(params, normalized_data, missing_mask,
                                             distance_metric, regularization, alpha)

        # Check convergence
        if np.linalg.norm(params - prev_params) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break

    # Impute missing values
    imputed_data = _impute_values(normalized_data, params, missing_mask)

    # Denormalize data
    imputed_data = _denormalize(imputed_data, norm_params, normalization)

    # Calculate metrics
    metrics = _calculate_metrics(data, imputed_data, missing_mask, metric, custom_metric)

    return {
        'result': imputed_data,
        'metrics': metrics,
        'params_used': {
            'n_iter': i + 1,
            'solver': solver,
            'distance_metric': distance_metric,
            'normalization': normalization,
            'regularization': regularization,
            'alpha': alpha,
            'beta': beta
        },
        'warnings': _check_warnings(data, imputed_data)
    }

def _validate_inputs(data: np.ndarray, missing_mask: np.ndarray) -> None:
    """Validate input data and mask."""
    if not isinstance(data, np.ndarray) or not isinstance(missing_mask, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if data.shape != missing_mask.shape:
        raise ValueError("Data and mask must have the same shape")
    if np.any(missing_mask < 0) or np.any(missing_mask > 1):
        raise ValueError("Mask must contain only 0s and 1s")
    if np.isnan(data).any() and not missing_mask[np.isnan(data)].all():
        raise ValueError("NaN values must be marked in the mask")

def _apply_normalization(data: np.ndarray, method: str) -> tuple:
    """Apply normalization to data."""
    if method == 'none':
        return data, None
    elif method == 'standard':
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        normalized = (data - mean) / std
    elif method == 'minmax':
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.nanmedian(data, axis=0)
        iqr = np.nanpercentile(data, 75, axis=0) - np.nanpercentile(data, 25, axis=0)
        normalized = (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return normalized, {'mean': mean, 'std': std} if method == 'standard' else \
           {'min': min_val, 'max': max_val} if method == 'minmax' else \
           {'median': median, 'iqr': iqr}

def _initialize_parameters(data: np.ndarray, missing_mask: np.ndarray,
                          rng: np.random.RandomState) -> np.ndarray:
    """Initialize parameters for Bayesian imputation."""
    n_features = data.shape[1]
    return rng.randn(n_features, 2)  # Mean and precision parameters

def _gradient_descent_step(params: np.ndarray, data: np.ndarray,
                          missing_mask: np.ndarray, distance_metric: Union[str, Callable],
                          regularization: Optional[str], alpha: float,
                          beta: float) -> np.ndarray:
    """Perform one gradient descent step."""
    # Implementation of gradient descent update
    return params  # Placeholder

def _newton_step(params: np.ndarray, data: np.ndarray,
                 missing_mask: np.ndarray, distance_metric: Union[str, Callable],
                 regularization: Optional[str], alpha: float) -> np.ndarray:
    """Perform one Newton step."""
    # Implementation of Newton update
    return params  # Placeholder

def _coordinate_descent_step(params: np.ndarray, data: np.ndarray,
                            missing_mask: np.ndarray, distance_metric: Union[str, Callable],
                            regularization: Optional[str], alpha: float) -> np.ndarray:
    """Perform one coordinate descent step."""
    # Implementation of coordinate descent update
    return params  # Placeholder

def _impute_values(data: np.ndarray, params: np.ndarray,
                   missing_mask: np.ndarray) -> np.ndarray:
    """Impute missing values using current parameters."""
    imputed = data.copy()
    # Implementation of imputation
    return imputed

def _denormalize(data: np.ndarray, norm_params: Dict[str, Any],
                 method: str) -> np.ndarray:
    """Denormalize data."""
    if method == 'none':
        return data
    elif method == 'standard':
        return data * norm_params['std'] + norm_params['mean']
    elif method == 'minmax':
        return data * (norm_params['max'] - norm_params['min']) + norm_params['min']
    elif method == 'robust':
        return data * norm_params['iqr'] + norm_params['median']
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _calculate_metrics(true_data: np.ndarray, imputed_data: np.ndarray,
                       missing_mask: np.ndarray, metric: Union[str, Callable],
                       custom_metric: Optional[Callable]) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((true_data[missing_mask] - imputed_data[missing_mask])**2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(true_data[missing_mask] - imputed_data[missing_mask]))
    elif metric == 'r2':
        ss_res = np.sum((true_data[missing_mask] - imputed_data[missing_mask])**2)
        ss_tot = np.sum((true_data[missing_mask] - np.mean(true_data[missing_mask]))**2)
        metrics['r2'] = 1 - ss_res / (ss_tot + 1e-8)
    elif callable(metric):
        metrics['custom'] = metric(true_data[missing_mask], imputed_data[missing_mask])
    if custom_metric is not None:
        metrics['custom_additional'] = custom_metric(true_data[missing_mask],
                                                    imputed_data[missing_mask])
    return metrics

def _check_warnings(data: np.ndarray, imputed_data: np.ndarray) -> list:
    """Check for potential issues and return warnings."""
    warnings = []
    if np.isnan(imputed_data).any():
        warnings.append("Imputed data contains NaN values")
    if np.isinf(imputed_data).any():
        warnings.append("Imputed data contains infinite values")
    return warnings
