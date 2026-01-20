"""
Quantix – Module normalisation_scaling
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# min_max_scaling
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(X: np.ndarray) -> None:
    """Validate input array."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input contains infinite values")

def _min_max_scaling_compute(
    X: np.ndarray,
    feature_range: tuple = (0, 1),
    copy: bool = True
) -> np.ndarray:
    """Compute min-max scaling for a given array."""
    if copy:
        X = X.copy()
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    data_range = max_vals - min_vals
    # Avoid division by zero for constant features
    data_range[data_range == 0] = 1
    X_scaled = (X - min_vals) / data_range
    X_scaled *= (feature_range[1] - feature_range[0])
    X_scaled += feature_range[0]
    return X_scaled

def min_max_scaling_fit(
    X: np.ndarray,
    feature_range: tuple = (0, 1),
    copy: bool = True
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Fit min-max scaling and return transformation parameters.

    Parameters
    ----------
    X : np.ndarray
        Input data to be scaled.
    feature_range : tuple, optional
        Desired range of transformed data (default is (0, 1)).
    copy : bool, optional
        If True, the input array is copied (default is True).

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Transformed data
        - 'metrics': Scaling parameters (min, max)
        - 'params_used': Parameters used for scaling
        - 'warnings': Any warnings generated during processing

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = min_max_scaling_fit(X)
    """
    _validate_input(X)

    warnings: Dict[str, str] = {}
    if np.any(np.std(X, axis=0) == 0):
        warnings['constant_features'] = 'Some features are constant and will be set to 0'

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    result = _min_max_scaling_compute(X, feature_range, copy)

    return {
        'result': result,
        'metrics': {
            'min_values': min_vals,
            'max_values': max_vals
        },
        'params_used': {
            'feature_range': feature_range,
            'copy': copy
        },
        'warnings': warnings
    }

################################################################################
# standard_scaling
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2-dimensional array")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input contains infinite values")

def _standard_scaling_compute(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute standard scaling."""
    if mean is None or std is None:
        raise ValueError("Mean and standard deviation must be provided for computation")
    return (X - mean) / std

def _standard_scaling_fit(
    X: np.ndarray,
    with_mean: bool = True,
    with_std: bool = True
) -> Dict[str, np.ndarray]:
    """Fit standard scaling parameters."""
    _validate_input(X)
    params = {}
    if with_mean:
        params['mean'] = np.mean(X, axis=0)
    else:
        params['mean'] = np.zeros(X.shape[1])
    if with_std:
        params['std'] = np.std(X, axis=0)
        # Avoid division by zero
        params['std'][params['std'] == 0] = 1.0
    else:
        params['std'] = np.ones(X.shape[1])
    return params

def standard_scaling_fit(
    X: np.ndarray,
    with_mean: bool = True,
    with_std: bool = True
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Fit standard scaling parameters.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    with_mean : bool, optional
        If True, center the data before scaling (default: True)
    with_std : bool, optional
        If True, scale the data to unit variance (default: True)

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': fitted parameters (mean and std)
        - 'metrics': empty dict
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = standard_scaling_fit(X)
    """
    warnings = []
    params_used = {
        'with_mean': with_mean,
        'with_std': with_std
    }

    try:
        params = _standard_scaling_fit(X, with_mean=with_mean, with_std=with_std)
        result = {
            'mean': params['mean'],
            'std': params['std']
        }
    except Exception as e:
        warnings.append(str(e))
        result = {}

    return {
        'result': result,
        'metrics': {},
        'params_used': params_used,
        'warnings': warnings
    }

def standard_scaling_compute(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Compute standard scaling.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    mean : np.ndarray, optional
        Mean values for each feature (default: None)
    std : np.ndarray, optional
        Standard deviation values for each feature (default: None)

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': scaled data
        - 'metrics': empty dict
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> mean = np.array([2, 3])
    >>> std = np.array([1, 1])
    >>> result = standard_scaling_compute(X, mean=mean, std=std)
    """
    warnings = []
    params_used = {
        'mean': mean,
        'std': std
    }

    try:
        _validate_input(X)
        scaled_data = _standard_scaling_compute(X, mean=mean, std=std)
    except Exception as e:
        warnings.append(str(e))
        scaled_data = np.array([])

    return {
        'result': scaled_data,
        'metrics': {},
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# robust_scaling
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def robust_scaling_fit(
    X: np.ndarray,
    normalization_type: str = 'robust',
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
    Fit robust scaling parameters to the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    normalization_type : str
        Type of normalization ('none', 'standard', 'minmax', 'robust')
    metric : str or callable
        Metric to evaluate scaling ('mse', 'mae', 'r2', custom callable)
    distance : str or callable
        Distance metric ('euclidean', 'manhattan', 'cosine', custom callable)
    solver : str
        Solver method ('closed_form', 'gradient_descent', 'newton')
    regularization : str or None
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    custom_metric : callable or None
        Custom metric function if needed
    custom_distance : callable or None
        Custom distance function if needed

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X)

    # Choose normalization method
    if normalization_type == 'none':
        scaling_params = _no_scaling(X)
    elif normalization_type == 'standard':
        scaling_params = _standard_scaling_fit(X)
    elif normalization_type == 'minmax':
        scaling_params = _minmax_scaling_fit(X)
    elif normalization_type == 'robust':
        scaling_params = _robust_scaling_fit(X)
    else:
        raise ValueError("Invalid normalization type")

    # Choose metric
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        if callable(metric):
            metric_func = metric
        else:
            raise ValueError("Metric must be a string or callable")

    # Choose distance
    if isinstance(distance, str):
        distance_func = _get_distance_function(distance)
    else:
        if callable(distance):
            distance_func = distance
        else:
            raise ValueError("Distance must be a string or callable")

    # Choose solver
    if solver == 'closed_form':
        params = _closed_form_solver(X, scaling_params)
    elif solver == 'gradient_descent':
        params = _gradient_descent_solver(X, scaling_params, tol, max_iter)
    elif solver == 'newton':
        params = _newton_solver(X, scaling_params, tol, max_iter)
    else:
        raise ValueError("Invalid solver type")

    # Calculate metrics
    metrics = _calculate_metrics(X, params, metric_func)

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalization_type': normalization_type,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be 2-dimensional")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input contains infinite values")

def _no_scaling(X: np.ndarray) -> Dict:
    """No scaling applied."""
    return {'scale': 1.0, 'center': 0.0}

def _standard_scaling_fit(X: np.ndarray) -> Dict:
    """Fit standard scaling parameters."""
    center = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    return {'scale': scale, 'center': center}

def _minmax_scaling_fit(X: np.ndarray) -> Dict:
    """Fit min-max scaling parameters."""
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    scale = max_val - min_val
    center = min_val
    return {'scale': scale, 'center': center}

def _robust_scaling_fit(X: np.ndarray) -> Dict:
    """Fit robust scaling parameters using median and IQR."""
    center = np.median(X, axis=0)
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
    scale = iqr
    return {'scale': scale, 'center': center}

def _get_metric_function(metric: str) -> Callable:
    """Get metric function based on string input."""
    metrics = {
        'mse': _mse,
        'mae': _mae,
        'r2': _r2
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance_function(distance: str) -> Callable:
    """Get distance function based on string input."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _closed_form_solver(X: np.ndarray, scaling_params: Dict) -> np.ndarray:
    """Closed form solution for scaling parameters."""
    # Placeholder implementation
    return np.zeros(X.shape[1])

def _gradient_descent_solver(
    X: np.ndarray,
    scaling_params: Dict,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent solver for scaling parameters."""
    # Placeholder implementation
    return np.zeros(X.shape[1])

def _newton_solver(
    X: np.ndarray,
    scaling_params: Dict,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Newton's method solver for scaling parameters."""
    # Placeholder implementation
    return np.zeros(X.shape[1])

def _calculate_metrics(
    X: np.ndarray,
    params: np.ndarray,
    metric_func: Callable
) -> Dict:
    """Calculate metrics for the scaling."""
    # Placeholder implementation
    return {'metric_value': 0.0}

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

################################################################################
# max_abs_scaling
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

def _max_abs_scaling_transform(X: np.ndarray, scale_params: Dict[str, float]) -> np.ndarray:
    """Apply max-abs scaling transformation."""
    return X / scale_params['max_abs']

def _compute_scale_params(X: np.ndarray) -> Dict[str, float]:
    """Compute max-abs scaling parameters."""
    return {'max_abs': np.max(np.abs(X), axis=0)}

def max_abs_scaling_fit(
    X: np.ndarray,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Fit max-abs scaling to data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    custom_metric : Callable, optional
        Custom metric function to evaluate scaling performance

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Transformed data
        - 'metrics': Computed metrics
        - 'params_used': Parameters used for scaling
        - 'warnings': Any warnings generated during processing

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = max_abs_scaling_fit(X)
    """
    _validate_inputs(X)

    # Compute scaling parameters
    scale_params = _compute_scale_params(X)

    # Apply transformation
    X_scaled = _max_abs_scaling_transform(X, scale_params)

    # Compute metrics if custom metric is provided
    metrics = {}
    if custom_metric is not None:
        try:
            metrics['custom'] = custom_metric(X, X_scaled)
        except Exception as e:
            metrics['custom_error'] = str(e)

    return {
        'result': X_scaled,
        'metrics': metrics,
        'params_used': scale_params,
        'warnings': []
    }

################################################################################
# normalization_l1
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input array dimensions and types."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input array contains NaN or infinite values")

def _normalize_l1(X: np.ndarray, axis: int = 1) -> np.ndarray:
    """Compute L1 normalization along specified axis."""
    norms = np.sum(np.abs(X), axis=axis, keepdims=True)
    return X / (norms + 1e-10)  # Avoid division by zero

def _compute_metrics(
    X_normalized: np.ndarray,
    metric_funcs: Dict[str, Callable[[np.ndarray], float]],
    **kwargs
) -> Dict[str, float]:
    """Compute specified metrics on normalized data."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(X_normalized, **kwargs)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def normalization_l1_fit(
    X: np.ndarray,
    normalize_func: Callable[[np.ndarray], np.ndarray] = _normalize_l1,
    metric_funcs: Optional[Dict[str, Callable[[np.ndarray], float]]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fit L1 normalization to data with optional metrics computation.

    Parameters:
        X: Input data array of shape (n_samples, n_features)
        normalize_func: Function to apply normalization
        metric_funcs: Dictionary of metric functions to compute on normalized data
        **kwargs: Additional arguments for normalization and metrics

    Returns:
        Dictionary containing:
            - "result": Normalized data
            - "metrics": Computed metrics if provided
            - "params_used": Parameters used in computation
            - "warnings": Any warnings generated

    Example:
        >>> X = np.random.rand(10, 5)
        >>> result = normalization_l1_fit(X)
    """
    _validate_inputs(X)

    # Apply normalization
    X_normalized = normalize_func(X, **kwargs)

    # Compute metrics if provided
    metrics = {}
    if metric_funcs is not None:
        metrics = _compute_metrics(X_normalized, metric_funcs, **kwargs)

    return {
        "result": X_normalized,
        "metrics": metrics,
        "params_used": {
            "normalize_func": normalize_func.__name__,
            **kwargs
        },
        "warnings": []
    }

################################################################################
# normalization_l2
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2-dimensional array")
    if np.any(np.isnan(X)):
        raise ValueError("Input X contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input X contains infinite values")

def _compute_l2_norm(X: np.ndarray) -> np.ndarray:
    """Compute L2 norm for each feature."""
    return np.linalg.norm(X, axis=0)

def _normalize_l2(X: np.ndarray) -> np.ndarray:
    """Normalize data using L2 norm."""
    norms = _compute_l2_norm(X)
    if np.any(norms == 0):
        raise ValueError("Zero norm encountered, cannot normalize")
    return X / norms

def normalization_l2_fit(
    X: np.ndarray,
    normalize: bool = True,
    metric: Union[str, Callable] = "euclidean",
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Fit L2 normalization to data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    normalize : bool, optional
        Whether to apply normalization, by default True
    metric : str or callable, optional
        Metric to use for evaluation, by default "euclidean"
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
        - "result": normalized data or original if normalize=False
        - "metrics": computed metrics
        - "params_used": parameters used
        - "warnings": any warnings encountered

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = normalization_l2_fit(X)
    """
    _validate_inputs(X)

    params_used = {
        "normalize": normalize,
        "metric": metric,
        **kwargs
    }

    warnings = []

    if not normalize:
        result = X.copy()
    else:
        try:
            result = _normalize_l2(X)
        except ValueError as e:
            warnings.append(str(e))
            result = X.copy()

    # Compute metrics
    metrics = {}
    if metric == "euclidean":
        norms_before = _compute_l2_norm(X)
        norms_after = _compute_l2_norm(result) if normalize else norms_before
        metrics["euclidean_norm"] = {
            "before": norms_before,
            "after": norms_after
        }
    elif callable(metric):
        metrics["custom_metric"] = metric(X, result)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# quantile_transform
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def quantile_transform_fit(
    X: np.ndarray,
    n_quantiles: int = 100,
    output_distribution: str = 'normal',
    subsample: Optional[int] = None,
    random_state: Optional[int] = None,
    copy: bool = True
) -> Dict:
    """
    Fit the quantile transform to X.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data used to compute the quantiles.
    n_quantiles : int, default=100
        Number of quantiles to be computed.
    output_distribution : str, default='normal'
        Desired distribution of the transformed data. Can be 'normal' or 'uniform'.
    subsample : int, default=None
        Number of samples to use for computing quantiles. If None, uses all samples.
    random_state : int, default=None
        Random seed for reproducibility when subsampling.
    copy : bool, default=True
        Whether to copy the input data.

    Returns
    -------
    dict
        Dictionary containing:
        - 'quantiles_': array of shape (n_quantiles, n_features)
            The computed quantiles.
        - 'reference_': array of shape (n_quantiles,)
            The reference values to transform the data to.
        - 'params_used': dict
            Parameters used for fitting.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = quantile_transform_fit(X)
    """
    # Input validation
    X = _validate_input(X, copy=copy)

    if n_quantiles <= 0:
        raise ValueError("n_quantiles must be positive.")
    if output_distribution not in ['normal', 'uniform']:
        raise ValueError("output_distribution must be either 'normal' or 'uniform'.")

    # Subsample if needed
    if subsample is not None:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(X.shape[0], subsample, replace=False)
        X = X[indices]

    # Compute quantiles
    quantiles = np.linspace(0, 1, n_quantiles)
    quantiles_X = np.percentile(X, quantiles * 100, axis=0)

    # Compute reference values
    if output_distribution == 'normal':
        reference = _compute_normal_reference(quantiles)
    else:
        reference = quantiles

    return {
        'quantiles_': quantiles_X,
        'reference_': reference,
        'params_used': {
            'n_quantiles': n_quantiles,
            'output_distribution': output_distribution,
            'subsample': subsample,
            'random_state': random_state
        }
    }

def quantile_transform_compute(
    X: np.ndarray,
    quantiles_: np.ndarray,
    reference_: np.ndarray,
    copy: bool = True
) -> Dict:
    """
    Transform X using precomputed quantiles and reference values.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data to be transformed.
    quantiles_ : array-like of shape (n_quantiles, n_features)
        The precomputed quantiles.
    reference_ : array-like of shape (n_quantiles,)
        The precomputed reference values.
    copy : bool, default=True
        Whether to copy the input data.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': array of shape (n_samples, n_features)
            The transformed data.
        - 'metrics': dict
            Metrics computed on the transformed data.
        - 'params_used': dict
            Parameters used for transformation.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> quantiles_ = np.array([[0.1, 0.2], [0.3, 0.4]])
    >>> reference_ = np.array([-1.28, -0.84])
    >>> result = quantile_transform_compute(X, quantiles_, reference_)
    """
    # Input validation
    X = _validate_input(X, copy=copy)
    quantiles_ = np.asarray(quantiles_)
    reference_ = np.asarray(reference_)

    if quantiles_.shape[0] != reference_.shape[0]:
        raise ValueError("quantiles_ and reference_ must have the same number of quantiles.")
    if quantiles_.shape[1] != X.shape[1]:
        raise ValueError("quantiles_ and X must have the same number of features.")

    # Transform data
    transformed_X = _transform_data(X, quantiles_, reference_)

    # Compute metrics
    metrics = {
        'mean': np.mean(transformed_X),
        'std': np.std(transformed_X),
        'min': np.min(transformed_X),
        'max': np.max(transformed_X)
    }

    return {
        'result': transformed_X,
        'metrics': metrics,
        'params_used': {
            'copy': copy
        }
    }

def _validate_input(X: np.ndarray, copy: bool = True) -> np.ndarray:
    """Validate and prepare input data."""
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if copy:
        X = X.copy()
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input data contains NaN or infinite values.")
    return X

def _compute_normal_reference(quantiles: np.ndarray) -> np.ndarray:
    """Compute reference values for normal distribution."""
    return np.sqrt(2) * scipy.special.erfinv(2 * quantiles - 1)

def _transform_data(
    X: np.ndarray,
    quantiles_: np.ndarray,
    reference_: np.ndarray
) -> np.ndarray:
    """Transform data using precomputed quantiles and reference values."""
    # Linear interpolation to find the corresponding reference value for each data point
    transformed_X = np.zeros_like(X)
    for i in range(X.shape[1]):
        transformed_X[:, i] = np.interp(
            X[:, i],
            quantiles_[:, i],
            reference_
        )
    return transformed_X

################################################################################
# power_transform
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2-dimensional array.")
    if np.any(np.isnan(X)):
        raise ValueError("Input X contains NaN values.")
    if np.any(np.isinf(X)):
        raise ValueError("Input X contains infinite values.")

def _power_transform(
    X: np.ndarray,
    method: str = 'box-cox',
    lambda_param: Optional[float] = None,
    standardize: bool = True
) -> np.ndarray:
    """Apply power transformation to the input data."""
    if method == 'box-cox':
        if lambda_param is None:
            raise ValueError("Lambda parameter must be provided for Box-Cox transformation.")
        return (X ** lambda_param - 1) / lambda_param if lambda_param != 0 else np.log(X)
    elif method == 'yeo-johnson':
        sign = np.sign(X)
        log_X = np.log(np.abs(X) + 1)
        return sign * ((np.abs(X) + 1) ** (2 - lambda_param) - 1) * (2 - lambda_param) / 2 if lambda_param != 2 else sign * np.log(np.abs(X) + 1)
    elif method == 'none':
        return X
    else:
        raise ValueError(f"Unknown transformation method: {method}")

def _compute_metrics(
    X_transformed: np.ndarray,
    metric_funcs: Dict[str, Callable[[np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for the transformed data."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(X_transformed)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def power_transform_fit(
    X: np.ndarray,
    method: str = 'box-cox',
    lambda_param: Optional[float] = None,
    standardize: bool = True,
    metric_funcs: Dict[str, Callable[[np.ndarray], float]] = None
) -> Dict:
    """
    Fit power transformation to the input data.

    Parameters:
    - X: Input data (2D numpy array).
    - method: Transformation method ('box-cox', 'yeo-johnson', or 'none').
    - lambda_param: Lambda parameter for the transformation.
    - standardize: Whether to standardize the data before transformation.
    - metric_funcs: Dictionary of metric functions to compute.

    Returns:
    - A dictionary containing the transformed data, metrics, parameters used, and warnings.
    """
    _validate_inputs(X)

    if metric_funcs is None:
        metric_funcs = {}

    X_transformed = _power_transform(X, method, lambda_param, standardize)
    metrics = _compute_metrics(X_transformed, metric_funcs)

    result = {
        'result': X_transformed,
        'metrics': metrics,
        'params_used': {
            'method': method,
            'lambda_param': lambda_param,
            'standardize': standardize
        },
        'warnings': []
    }

    return result

def power_transform_compute(
    X: np.ndarray,
    lambda_param: float,
    method: str = 'box-cox',
    standardize: bool = True
) -> np.ndarray:
    """
    Compute power transformation for the input data with given lambda parameter.

    Parameters:
    - X: Input data (2D numpy array).
    - lambda_param: Lambda parameter for the transformation.
    - method: Transformation method ('box-cox' or 'yeo-johnson').
    - standardize: Whether to standardize the data before transformation.

    Returns:
    - Transformed data as a numpy array.
    """
    _validate_inputs(X)
    return _power_transform(X, method, lambda_param, standardize)

# Example usage:
# X = np.array([[1.0, 2.0], [3.0, 4.0]])
# result = power_transform_fit(X, method='box-cox', lambda_param=0.5)

################################################################################
# unit_vector_scaling
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(X: np.ndarray) -> None:
    """Validate input array."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

def _compute_unit_vectors(X: np.ndarray) -> np.ndarray:
    """Compute unit vectors for each row of the input array."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms

def _compute_metrics(
    X: np.ndarray,
    scaled_X: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
) -> Dict[str, float]:
    """Compute metrics between original and scaled data."""
    return {"metric": metric_func(X, scaled_X)}

def unit_vector_scaling_fit(
    X: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.mean((x - y) ** 2),
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Fit unit vector scaling to the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    metric_func : Callable[[np.ndarray, np.ndarray], float]
        Function to compute the metric between original and scaled data.
    **kwargs
        Additional keyword arguments for future extensions.

    Returns:
    --------
    dict
        Dictionary containing the scaled data, metrics, parameters used, and warnings.
    """
    _validate_input(X)
    scaled_X = _compute_unit_vectors(X)
    metrics = _compute_metrics(X, scaled_X, metric_func)

    return {
        "result": scaled_X,
        "metrics": metrics,
        "params_used": {"metric_func": metric_func.__name__},
        "warnings": [],
    }
