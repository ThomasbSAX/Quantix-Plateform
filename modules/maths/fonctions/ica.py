"""
Quantix – Module ica
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# independent_component_analysis
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def independent_component_analysis_fit(
    X: np.ndarray,
    n_components: int = None,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'gradient_descent',
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    custom_normalization: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Perform Independent Component Analysis (ICA) on the input data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of independent components to extract. If None, uses all features.
    normalization : str or callable, optional
        Normalization method: 'none', 'standard', 'minmax', 'robust' or custom callable.
    metric : str or callable, optional
        Metric to optimize: 'mse', 'mae', 'r2', 'logloss' or custom callable.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    random_state : int, optional
        Random seed for reproducibility.
    custom_normalization : callable, optional
        Custom normalization function.
    custom_metric : callable, optional
        Custom metric function.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Estimated independent components.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = independent_component_analysis_fit(X, n_components=3)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Set random state
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    X_normalized, norm_warning = _apply_normalization(X, normalization, custom_normalization)

    # Initialize parameters
    params_used = {
        'n_components': n_components if n_components is not None else X.shape[1],
        'normalization': normalization,
        'metric': metric,
        'solver': solver,
        'max_iter': max_iter,
        'tol': tol
    }

    # Select solver
    if solver == 'closed_form':
        components = _closed_form_solver(X_normalized, params_used['n_components'])
    elif solver == 'gradient_descent':
        components = _gradient_descent_solver(
            X_normalized, params_used['n_components'], max_iter, tol,
            metric if isinstance(metric, str) else custom_metric
        )
    elif solver == 'newton':
        components = _newton_solver(X_normalized, params_used['n_components'], max_iter, tol)
    elif solver == 'coordinate_descent':
        components = _coordinate_descent_solver(X_normalized, params_used['n_components'], max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, components, metric if isinstance(metric, str) else custom_metric)

    # Prepare output
    result = {
        'result': components,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': norm_warning if norm_warning else []
    }

    return result

def _validate_inputs(X: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if n_components is not None and (n_components <= 0 or n_components > X.shape[1]):
        raise ValueError("Invalid number of components")

def _apply_normalization(
    X: np.ndarray,
    normalization: str,
    custom_normalization: Optional[Callable]
) -> tuple:
    """Apply specified normalization to the data."""
    warnings = []

    if custom_normalization is not None:
        X_norm = custom_normalization(X)
    elif normalization == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalization == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalization == 'robust':
        median = np.median(X, axis=0)
        iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        X_norm = (X - median) / iqr
    elif normalization == 'none':
        X_norm = X.copy()
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    # Check for division by zero or invalid values
    if np.any(np.isnan(X_norm)) or np.any(np.isinf(X_norm)):
        warnings.append("Normalization resulted in NaN or infinite values")

    return X_norm, warnings

def _closed_form_solver(X: np.ndarray, n_components: int) -> np.ndarray:
    """Closed-form solution for ICA."""
    # This is a placeholder - actual implementation would use eigendecomposition or similar
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :n_components]

def _gradient_descent_solver(
    X: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float,
    metric_func: Callable
) -> np.ndarray:
    """Gradient descent solver for ICA."""
    n_samples, n_features = X.shape
    W = np.random.randn(n_components, n_features)

    for _ in range(max_iter):
        # Update rule (placeholder)
        W_new = W + 0.01 * np.random.randn(*W.shape)

        # Check convergence
        if np.linalg.norm(W_new - W) < tol:
            break

        W = W_new

    return W.T

def _newton_solver(X: np.ndarray, n_components: int, max_iter: int, tol: float) -> np.ndarray:
    """Newton's method solver for ICA."""
    # Placeholder implementation
    return np.random.randn(X.shape[1], n_components)

def _coordinate_descent_solver(X: np.ndarray, n_components: int, max_iter: int, tol: float) -> np.ndarray:
    """Coordinate descent solver for ICA."""
    # Placeholder implementation
    return np.random.randn(X.shape[1], n_components)

def _compute_metrics(
    X: np.ndarray,
    components: np.ndarray,
    metric_func: Union[str, Callable]
) -> Dict:
    """Compute metrics for the ICA results."""
    if isinstance(metric_func, str):
        if metric_func == 'mse':
            return {'mse': np.mean((X - X @ components) ** 2)}
        elif metric_func == 'mae':
            return {'mae': np.mean(np.abs(X - X @ components))}
        elif metric_func == 'r2':
            ss_res = np.sum((X - X @ components) ** 2)
            ss_tot = np.sum(X ** 2)
            return {'r2': 1 - (ss_res / ss_tot)}
        elif metric_func == 'logloss':
            return {'logloss': -np.mean(X * np.log(X @ components + 1e-10))}
        else:
            raise ValueError(f"Unknown metric: {metric_func}")
    else:
        return {'custom_metric': metric_func(X, components)}

################################################################################
# fast_ica
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def fast_ica_fit(
    X: np.ndarray,
    n_components: int = 2,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    normalization: str = 'standard',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'gradient_descent',
    fun: Optional[Callable[[np.ndarray], float]] = None,
    fun_grad: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    FastICA algorithm for Independent Component Analysis.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of components to extract (default: 2).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    tol : float, optional
        Tolerance for stopping criterion (default: 1e-4).
    random_state : Optional[int], optional
        Random seed for reproducibility (default: None).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard').
    metric : Union[str, Callable], optional
        Metric for evaluation ('mse', 'mae', 'r2', custom callable) (default: 'mse').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default: 'gradient_descent').
    fun : Optional[Callable], optional
        Custom objective function (default: None).
    fun_grad : Optional[Callable], optional
        Gradient of custom objective function (default: None).

    Returns:
    --------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]
        Dictionary containing:
        - 'result': Estimated components.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings encountered.

    Example:
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = fast_ica_fit(X, n_components=3)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Set random state
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    X_norm = _normalize_data(X, normalization)

    # Initialize components
    W = np.random.randn(n_components, X_norm.shape[1])

    # Choose solver
    if solver == 'closed_form':
        W = _closed_form_solver(X_norm, n_components)
    elif solver == 'gradient_descent':
        W = _gradient_descent_solver(X_norm, n_components, max_iter, tol, fun, fun_grad)
    elif solver == 'newton':
        W = _newton_solver(X_norm, n_components, max_iter, tol)
    elif solver == 'coordinate_descent':
        W = _coordinate_descent_solver(X_norm, n_components, max_iter, tol)
    else:
        raise ValueError("Invalid solver specified.")

    # Compute components
    S = np.dot(W, X_norm.T)

    # Compute metrics
    metrics = _compute_metrics(X_norm, S, metric)

    # Prepare output
    result_dict = {
        'result': W,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'max_iter': max_iter,
            'tol': tol,
            'normalization': normalization,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

    return result_dict

def _validate_inputs(X: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if n_components <= 0 or n_components > X.shape[1]:
        raise ValueError("n_components must be between 1 and the number of features.")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data according to specified method."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError("Invalid normalization method specified.")

def _closed_form_solver(X: np.ndarray, n_components: int) -> np.ndarray:
    """Closed form solution for FastICA."""
    # Placeholder for closed form implementation
    return np.random.randn(n_components, X.shape[1])

def _gradient_descent_solver(
    X: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float,
    fun: Optional[Callable[[np.ndarray], float]],
    fun_grad: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Gradient descent solver for FastICA."""
    # Placeholder for gradient descent implementation
    return np.random.randn(n_components, X.shape[1])

def _newton_solver(X: np.ndarray, n_components: int, max_iter: int, tol: float) -> np.ndarray:
    """Newton solver for FastICA."""
    # Placeholder for Newton method implementation
    return np.random.randn(n_components, X.shape[1])

def _coordinate_descent_solver(X: np.ndarray, n_components: int, max_iter: int, tol: float) -> np.ndarray:
    """Coordinate descent solver for FastICA."""
    # Placeholder for coordinate descent implementation
    return np.random.randn(n_components, X.shape[1])

def _compute_metrics(X: np.ndarray, S: np.ndarray, metric: Union[str, Callable]) -> Dict[str, float]:
    """Compute metrics for FastICA results."""
    if callable(metric):
        return {'custom_metric': metric(X, S)}
    elif metric == 'mse':
        return {'mse': np.mean((X - S.T) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(X - S.T))}
    elif metric == 'r2':
        ss_res = np.sum((X - S.T) ** 2)
        ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
        return {'r2': 1 - ss_res / ss_tot}
    else:
        raise ValueError("Invalid metric specified.")

################################################################################
# non_gaussianity
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for non-gaussianity computation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values")

def _normalize_data(
    data: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data using specified method or custom function."""
    if custom_func is not None:
        return custom_func(data)

    if method == "none":
        return data
    elif method == "standard":
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == "minmax":
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == "robust":
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_non_gaussianity(
    data: np.ndarray,
    metric: str = "kurtosis",
    custom_metric: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Any]:
    """Compute non-gaussianity using specified metric or custom function."""
    if custom_metric is not None:
        return {"value": custom_metric(data)}

    if metric == "kurtosis":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return {"value": (np.mean((data - mean) ** 4, axis=0) / (std ** 4)) - 3}
    elif metric == "skewness":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return {"value": np.mean((data - mean) ** 3, axis=0) / (std ** 3)}
    elif metric == "entropy":
        # Example implementation - could be more sophisticated
        hist, _ = np.histogram(data, bins=10)
        prob = hist / np.sum(hist)
        return {"value": -np.sum(prob * np.log(prob + 1e-10))}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def non_gaussianity_fit(
    data: np.ndarray,
    normalization_method: str = "standard",
    metric: str = "kurtosis",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute non-gaussianity of data with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    normalization_method : str, optional
        Normalization method to apply (default: "standard")
    metric : str, optional
        Metric to use for non-gaussianity computation (default: "kurtosis")
    custom_normalization : callable, optional
        Custom normalization function
    custom_metric : callable, optional
        Custom metric computation function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.random.randn(100, 5)
    >>> result = non_gaussianity_fit(data)
    """
    _validate_input(data)

    # Normalize data
    normalized_data = _normalize_data(
        data,
        method=normalization_method,
        custom_func=custom_normalization
    )

    # Compute non-gaussianity
    result = _compute_non_gaussianity(
        normalized_data,
        metric=metric,
        custom_metric=custom_metric
    )

    return {
        "result": result["value"],
        "metrics": {"non_gaussianity": result["value"]},
        "params_used": {
            "normalization_method": normalization_method,
            "metric": metric
        },
        "warnings": []
    }

################################################################################
# central_limit_theorem
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def central_limit_theorem_fit(
    data: np.ndarray,
    n_samples: int = 1000,
    sample_size: int = 30,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_normalization: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit the Central Limit Theorem to sample data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    n_samples : int, optional
        Number of samples to draw (default: 1000).
    sample_size : int, optional
        Size of each sample (default: 30).
    normalization : str or callable, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') or custom callable (default: 'standard').
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2') or custom callable (default: 'mse').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent') (default: 'closed_form').
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    custom_normalization : callable, optional
        Custom normalization function.
    custom_metric : callable, optional
        Custom metric function.

    Returns:
    --------
    Dict containing:
        - 'result': Estimated parameters.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used in the fit.
        - 'warnings': Any warnings encountered.

    Example:
    --------
    >>> data = np.random.normal(0, 1, 1000)
    >>> result = central_limit_theorem_fit(data)
    """
    # Validate inputs
    _validate_inputs(data, n_samples, sample_size)

    # Normalize data
    normalized_data = _apply_normalization(data, normalization, custom_normalization)

    # Draw samples
    samples = _draw_samples(normalized_data, n_samples, sample_size)

    # Estimate parameters
    params = _estimate_parameters(samples, solver, tol, max_iter)

    # Compute metrics
    metrics = _compute_metrics(samples, params, metric, custom_metric)

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'n_samples': n_samples,
            'sample_size': sample_size,
            'normalization': normalization,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

    return result

def _validate_inputs(data: np.ndarray, n_samples: int, sample_size: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if len(data.shape) != 1:
        raise ValueError("Input data must be 1-dimensional.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values.")
    if n_samples <= 0:
        raise ValueError("Number of samples must be positive.")
    if sample_size <= 1:
        raise ValueError("Sample size must be greater than 1.")
    if sample_size > len(data):
        raise ValueError("Sample size cannot be larger than the input data size.")

def _apply_normalization(
    data: np.ndarray,
    normalization: str,
    custom_normalization: Optional[Callable]
) -> np.ndarray:
    """Apply normalization to the data."""
    if custom_normalization is not None:
        return custom_normalization(data)

    if normalization == 'none':
        return data
    elif normalization == 'standard':
        return (data - np.mean(data)) / np.std(data)
    elif normalization == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif normalization == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _draw_samples(data: np.ndarray, n_samples: int, sample_size: int) -> np.ndarray:
    """Draw random samples from the data."""
    return np.random.choice(data, size=(n_samples, sample_size), replace=True)

def _estimate_parameters(
    samples: np.ndarray,
    solver: str,
    tol: float,
    max_iter: int
) -> Dict:
    """Estimate parameters using the specified solver."""
    if solver == 'closed_form':
        return _closed_form_solver(samples, tol)
    elif solver == 'gradient_descent':
        return _gradient_descent_solver(samples, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _closed_form_solver(samples: np.ndarray, tol: float) -> Dict:
    """Closed-form solver for parameter estimation."""
    mean = np.mean(samples, axis=1)
    variance = np.var(samples, axis=1)
    return {'mean': mean, 'variance': variance}

def _gradient_descent_solver(
    samples: np.ndarray,
    tol: float,
    max_iter: int
) -> Dict:
    """Gradient descent solver for parameter estimation."""
    # Placeholder for gradient descent implementation
    mean = np.mean(samples, axis=1)
    variance = np.var(samples, axis=1)
    return {'mean': mean, 'variance': variance}

def _compute_metrics(
    samples: np.ndarray,
    params: Dict,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute metrics based on the estimated parameters."""
    if custom_metric is not None:
        return {'custom': custom_metric(samples, params)}

    if metric == 'mse':
        return {'mse': _compute_mse(samples, params)}
    elif metric == 'mae':
        return {'mae': _compute_mae(samples, params)}
    elif metric == 'r2':
        return {'r2': _compute_r2(samples, params)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_mse(samples: np.ndarray, params: Dict) -> float:
    """Compute Mean Squared Error."""
    predicted = np.tile(params['mean'], (samples.shape[1], 1)).T
    return np.mean((samples - predicted) ** 2)

def _compute_mae(samples: np.ndarray, params: Dict) -> float:
    """Compute Mean Absolute Error."""
    predicted = np.tile(params['mean'], (samples.shape[1], 1)).T
    return np.mean(np.abs(samples - predicted))

def _compute_r2(samples: np.ndarray, params: Dict) -> float:
    """Compute R-squared."""
    predicted = np.tile(params['mean'], (samples.shape[1], 1)).T
    ss_res = np.sum((samples - predicted) ** 2)
    ss_tot = np.sum((samples - np.mean(samples)) ** 2)
    return 1 - (ss_res / ss_tot)

################################################################################
# whitening
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def whitening_fit(
    X: np.ndarray,
    *,
    normalization: str = 'standard',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    metric: str = 'mse',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Perform whitening transformation on input data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton')
    regularization : str, optional
        Regularization method ('none', 'l1', 'l2', 'elasticnet')
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    metric : str, optional
        Metric for evaluation ('mse', 'mae', 'r2')
    custom_metric : callable, optional
        Custom metric function
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': whitened data matrix
        - 'metrics': dictionary of computed metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings

    Example
    -------
    >>> X = np.random.randn(100, 5)
    >>> result = whitening_fit(X, normalization='standard', solver='closed_form')
    """
    # Validate inputs
    X = _validate_input(X)

    # Normalize data
    X_norm, norm_params = _apply_normalization(X, normalization)

    # Choose solver
    if solver == 'closed_form':
        W = _closed_form_solver(X_norm, regularization)
    elif solver == 'gradient_descent':
        W = _gradient_descent_solver(X_norm, tol, max_iter, random_state)
    elif solver == 'newton':
        W = _newton_solver(X_norm, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply whitening
    X_white = _apply_whitening(X_norm, W)

    # Compute metrics
    metrics = _compute_metrics(X_white, X_norm, metric, custom_metric)

    return {
        'result': X_white,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

def _validate_input(X: np.ndarray) -> np.ndarray:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")
    return X

def _apply_normalization(X: np.ndarray, method: str) -> tuple:
    """Apply normalization to input data."""
    warnings = []
    if method == 'none':
        return X, {}
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_norm, {
        'mean': mean if method == 'standard' else None,
        'std': std if method == 'standard' else None,
        'min_val': min_val if method == 'minmax' else None,
        'max_val': max_val if method == 'minmax' else None,
        'median': median if method == 'robust' else None,
        'iqr': iqr if method == 'robust' else None
    }

def _closed_form_solver(X: np.ndarray, regularization: Optional[str]) -> np.ndarray:
    """Closed form solution for whitening."""
    C = np.cov(X, rowvar=False)
    if regularization == 'l2':
        C += 1e-6 * np.eye(C.shape[0])
    eigvals, eigvecs = np.linalg.eigh(C)
    D = np.diag(1.0 / np.sqrt(eigvals + 1e-8))
    return eigvecs @ D @ eigvecs.T

def _gradient_descent_solver(
    X: np.ndarray,
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> np.ndarray:
    """Gradient descent solver for whitening."""
    if random_state is not None:
        np.random.seed(random_state)
    n_features = X.shape[1]
    W = np.eye(n_features)

    for _ in range(max_iter):
        grad = 2 * (np.cov(X @ W, rowvar=False) - np.eye(n_features)) @ W
        W -= tol * grad

    return W

def _newton_solver(X: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Newton solver for whitening."""
    n_features = X.shape[1]
    W = np.eye(n_features)

    for _ in range(max_iter):
        C = np.cov(X @ W, rowvar=False)
        grad = 2 * (C - np.eye(n_features)) @ W
        hess = 2 * np.kron(np.eye(n_features), C) + 2 * np.kron(C, np.eye(n_features))
        W -= np.linalg.solve(hess, grad).reshape(W.shape)

    return W

def _apply_whitening(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply whitening transformation."""
    return X @ W

def _compute_metrics(
    X_white: np.ndarray,
    X_norm: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}

    if metric == 'mse':
        mse = np.mean((X_white @ X_white.T - np.eye(X_white.shape[0]))**2)
        metrics['mse'] = mse
    elif metric == 'mae':
        mae = np.mean(np.abs(X_white @ X_white.T - np.eye(X_white.shape[0])))
        metrics['mae'] = mae
    elif metric == 'r2':
        ss_res = np.sum((X_white @ X_white.T - np.eye(X_white.shape[0]))**2)
        ss_tot = np.sum((X_norm @ X_norm.T)**2) - np.mean(X_norm @ X_norm.T)**2
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        metrics['r2'] = r2

    if custom_metric is not None:
        metrics['custom'] = custom_metric(X_white, X_norm)

    return metrics

################################################################################
# preprocessing
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def preprocessing_fit(
    X: np.ndarray,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Main preprocessing function for ICA (Independent Component Analysis).

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric to evaluate preprocessing ('mse', 'mae', 'r2', custom callable)
    distance : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', custom callable)
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
    >>> result = preprocessing_fit(X, normalize='standard', metric='mse')
    """
    # Validate inputs
    _validate_inputs(X, normalize)

    # Normalize data
    X_normalized = _apply_normalization(X, normalize)

    # Initialize parameters
    params_used = {
        'normalize': normalize,
        'metric': metric if isinstance(metric, str) else 'custom',
        'distance': distance if isinstance(distance, str) else 'custom',
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Select solver
    if solver == 'closed_form':
        result = _solve_closed_form(X_normalized, regularization)
    elif solver == 'gradient_descent':
        result = _solve_gradient_descent(
            X_normalized, metric, distance,
            regularization, tol, max_iter,
            custom_metric, custom_distance
        )
    elif solver == 'newton':
        result = _solve_newton(
            X_normalized, metric, distance,
            regularization, tol, max_iter,
            custom_metric, custom_distance
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(X_normalized, result['components'], metric, custom_metric)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': _check_warnings(result)
    }

    return output

def _validate_inputs(X: np.ndarray, normalize: str) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input data contains infinite values")

    valid_normalizations = ['none', 'standard', 'minmax', 'robust']
    if normalize not in valid_normalizations:
        raise ValueError(f"normalize must be one of {valid_normalizations}")

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

def _solve_closed_form(X: np.ndarray, regularization: Optional[str]) -> Dict[str, Any]:
    """Closed form solution for ICA preprocessing."""
    # This is a placeholder - actual implementation would use proper ICA methods
    components = np.linalg.eig(np.cov(X.T))[1].T
    return {'components': components}

def _solve_gradient_descent(
    X: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Any]:
    """Gradient descent solver for ICA preprocessing."""
    # Initialize components
    n_components = X.shape[1]
    W = np.random.randn(n_components, n_components)

    for _ in range(max_iter):
        # Update components (placeholder implementation)
        W_new = _update_components_gradient_descent(
            X, W, metric, distance,
            regularization, custom_metric, custom_distance
        )

        # Check convergence
        if np.linalg.norm(W_new - W) < tol:
            break

        W = W_new

    return {'components': W}

def _update_components_gradient_descent(
    X: np.ndarray,
    W: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> np.ndarray:
    """Update components using gradient descent."""
    # Placeholder implementation - actual ICA update rule would go here
    return W + 0.1 * np.random.randn(*W.shape)

def _solve_newton(
    X: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Any]:
    """Newton method solver for ICA preprocessing."""
    # Initialize components
    n_components = X.shape[1]
    W = np.random.randn(n_components, n_components)

    for _ in range(max_iter):
        # Update components (placeholder implementation)
        W_new = _update_components_newton(
            X, W, metric, distance,
            regularization, custom_metric, custom_distance
        )

        # Check convergence
        if np.linalg.norm(W_new - W) < tol:
            break

        W = W_new

    return {'components': W}

def _update_components_newton(
    X: np.ndarray,
    W: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> np.ndarray:
    """Update components using Newton method."""
    # Placeholder implementation - actual ICA update rule would go here
    return W + 0.1 * np.random.randn(*W.shape)

def _calculate_metrics(
    X: np.ndarray,
    components: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Calculate metrics for preprocessing results."""
    if isinstance(metric, str):
        if metric == 'mse':
            return {'mse': np.mean((X - X @ components) ** 2)}
        elif metric == 'mae':
            return {'mae': np.mean(np.abs(X - X @ components))}
        elif metric == 'r2':
            ss_res = np.sum((X - X @ components) ** 2)
            ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
            return {'r2': 1 - ss_res / ss_tot}
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        return {'custom_metric': custom_metric(X, components)}
    else:
        raise TypeError("metric must be either a string or callable")

def _check_warnings(result: Dict[str, Any]) -> list:
    """Check for potential warnings in the results."""
    warnings = []
    if np.any(np.isnan(result['components'])):
        warnings.append("Result contains NaN values")
    if np.any(np.isinf(result['components'])):
        warnings.append("Result contains infinite values")
    return warnings

################################################################################
# dimensionality_reduction
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def dimensionality_reduction_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Perform dimensionality reduction using Independent Component Analysis (ICA).

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of components to keep. Default is 2.
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'. Default is 'standard'.
    metric : str or callable, optional
        Metric to optimize: 'mse', 'mae', 'r2', or custom callable. Default is 'mse'.
    distance : str or callable, optional
        Distance metric: 'euclidean', 'manhattan', 'cosine', or custom callable. Default is 'euclidean'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'. Default is 'gradient_descent'.
    regularization : str, optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'. Default is None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric : callable, optional
        Custom metric function. Default is None.
    custom_distance : callable, optional
        Custom distance function. Default is None.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Reduced data matrix of shape (n_samples, n_components).
        - 'metrics': Dictionary of computed metrics.
        - 'params_used': Dictionary of parameters used.
        - 'warnings': List of warnings encountered.

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = dimensionality_reduction_fit(X, n_components=2)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Normalize data
    X_normalized = _normalize_data(X, normalization)

    # Initialize parameters
    params_used = {
        'n_components': n_components,
        'normalization': normalization,
        'metric': metric,
        'distance': distance,
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Choose metric and distance functions
    metric_func = _get_metric_function(metric, custom_metric)
    distance_func = _get_distance_function(distance, custom_distance)

    # Initialize warnings
    warnings = []

    # Perform dimensionality reduction based on solver choice
    if solver == 'closed_form':
        result = _closed_form_solution(X_normalized, n_components)
    elif solver == 'gradient_descent':
        result = _gradient_descent_solution(
            X_normalized, n_components, metric_func, distance_func,
            regularization, tol, max_iter, warnings
        )
    elif solver == 'newton':
        result = _newton_solution(
            X_normalized, n_components, metric_func, distance_func,
            regularization, tol, max_iter, warnings
        )
    elif solver == 'coordinate_descent':
        result = _coordinate_descent_solution(
            X_normalized, n_components, metric_func, distance_func,
            regularization, tol, max_iter, warnings
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, result, metric_func)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
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
    if n_components > X.shape[1]:
        raise ValueError("n_components cannot be greater than the number of features")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data based on the specified method."""
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

def _get_metric_function(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Get the metric function based on the specified metric."""
    if custom_metric is not None:
        return custom_metric
    elif metric == 'mse':
        return _mean_squared_error
    elif metric == 'mae':
        return _mean_absolute_error
    elif metric == 'r2':
        return _r_squared
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _get_distance_function(distance: Union[str, Callable], custom_distance: Optional[Callable]) -> Callable:
    """Get the distance function based on the specified distance."""
    if custom_distance is not None:
        return custom_distance
    elif distance == 'euclidean':
        return _euclidean_distance
    elif distance == 'manhattan':
        return _manhattan_distance
    elif distance == 'cosine':
        return _cosine_distance
    else:
        raise ValueError(f"Unknown distance: {distance}")

def _closed_form_solution(X: np.ndarray, n_components: int) -> np.ndarray:
    """Compute the closed-form solution for dimensionality reduction."""
    # This is a placeholder for the actual implementation
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :n_components] @ np.diag(S[:n_components])

def _gradient_descent_solution(
    X: np.ndarray,
    n_components: int,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    warnings: list
) -> np.ndarray:
    """Compute the solution using gradient descent."""
    # This is a placeholder for the actual implementation
    return np.random.randn(X.shape[0], n_components)

def _newton_solution(
    X: np.ndarray,
    n_components: int,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    warnings: list
) -> np.ndarray:
    """Compute the solution using Newton's method."""
    # This is a placeholder for the actual implementation
    return np.random.randn(X.shape[0], n_components)

def _coordinate_descent_solution(
    X: np.ndarray,
    n_components: int,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    warnings: list
) -> np.ndarray:
    """Compute the solution using coordinate descent."""
    # This is a placeholder for the actual implementation
    return np.random.randn(X.shape[0], n_components)

def _compute_metrics(
    X: np.ndarray,
    result: np.ndarray,
    metric_func: Callable
) -> Dict:
    """Compute the metrics for the dimensionality reduction."""
    return {
        'metric_value': metric_func(X, result),
        'reconstruction_error': _mean_squared_error(X, np.dot(result, result.T))
    }

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the R-squared metric."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    return 1 - (ss_res / ss_tot)

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

################################################################################
# source_separation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def source_separation_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalizer: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Perform Independent Component Analysis (ICA) for source separation.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of components to extract (default: 2).
    normalizer : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard').
    metric : str or callable, optional
        Metric to evaluate performance ('mse', 'mae', 'r2', 'logloss') or custom callable (default: 'mse').
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') (default: 'euclidean').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default: 'gradient_descent').
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet') (default: None).
    tol : float, optional
        Tolerance for convergence (default: 1e-4).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    random_state : int, optional
        Random seed for reproducibility (default: None).
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics (default: None).
    custom_distance : callable, optional
        Custom distance function if not using built-in distances (default: None).

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Estimated sources and mixing matrix.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = source_separation_fit(X, n_components=3, normalizer='standard', metric='mse')
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Set random state if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize parameters
    W = _initialize_parameters(n_components, X_normalized.shape[1], random_state)

    # Choose solver
    if solver == 'closed_form':
        W = _solve_closed_form(X_normalized, n_components)
    elif solver == 'gradient_descent':
        W = _solve_gradient_descent(X_normalized, n_components, tol, max_iter,
                                   distance, regularization)
    elif solver == 'newton':
        W = _solve_newton(X_normalized, n_components, tol, max_iter)
    elif solver == 'coordinate_descent':
        W = _solve_coordinate_descent(X_normalized, n_components, tol, max_iter,
                                     regularization)
    else:
        raise ValueError("Unsupported solver method")

    # Estimate sources
    S = np.dot(W, X_normalized.T).T

    # Compute metrics
    metrics = _compute_metrics(X, X_normalized, S, W,
                              metric if not custom_metric else custom_metric)

    # Prepare output
    result = {
        'result': {'sources': S, 'mixing_matrix': W},
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'normalizer': normalizer,
            'metric': metric if not custom_metric else 'custom',
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if n_components <= 0 or n_components > X.shape[1]:
        raise ValueError("n_components must be between 1 and n_features")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the data."""
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
        raise ValueError("Unsupported normalization method")

def _initialize_parameters(n_components: int, n_features: int,
                          random_state: Optional[int] = None) -> np.ndarray:
    """Initialize parameters for ICA."""
    if random_state is not None:
        np.random.seed(random_state)
    return np.random.randn(n_components, n_features)

def _solve_closed_form(X: np.ndarray, n_components: int) -> np.ndarray:
    """Solve ICA using closed-form solution."""
    # This is a placeholder for actual implementation
    return np.linalg.pinv(X.T @ X) @ X.T

def _solve_gradient_descent(X: np.ndarray, n_components: int,
                           tol: float, max_iter: int,
                           distance: str, regularization: Optional[str]) -> np.ndarray:
    """Solve ICA using gradient descent."""
    # This is a placeholder for actual implementation
    W = _initialize_parameters(n_components, X.shape[1])
    for _ in range(max_iter):
        # Update W using gradient descent
        pass
    return W

def _solve_newton(X: np.ndarray, n_components: int,
                  tol: float, max_iter: int) -> np.ndarray:
    """Solve ICA using Newton's method."""
    # This is a placeholder for actual implementation
    W = _initialize_parameters(n_components, X.shape[1])
    for _ in range(max_iter):
        # Update W using Newton's method
        pass
    return W

def _solve_coordinate_descent(X: np.ndarray, n_components: int,
                             tol: float, max_iter: int,
                             regularization: Optional[str]) -> np.ndarray:
    """Solve ICA using coordinate descent."""
    # This is a placeholder for actual implementation
    W = _initialize_parameters(n_components, X.shape[1])
    for _ in range(max_iter):
        # Update W using coordinate descent
        pass
    return W

def _compute_metrics(X: np.ndarray, X_normalized: np.ndarray,
                     S: np.ndarray, W: np.ndarray,
                     metric: Union[str, Callable]) -> Dict:
    """Compute metrics for ICA results."""
    if callable(metric):
        return {'custom_metric': metric(X, S)}
    elif metric == 'mse':
        return {'mse': np.mean((X - S @ W) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(X - S @ W))}
    elif metric == 'r2':
        ss_res = np.sum((X - S @ W) ** 2)
        ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
        return {'r2': 1 - ss_res / ss_tot}
    elif metric == 'logloss':
        return {'logloss': -np.mean(X * np.log(S @ W + 1e-10))}
    else:
        raise ValueError("Unsupported metric")

################################################################################
# blind_source_separation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def blind_source_separation_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalizer: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'gradient_descent',
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    **kwargs
) -> Dict:
    """
    Perform blind source separation using Independent Component Analysis (ICA).

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of components to extract. Default is 2.
    normalizer : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'. Default is 'standard'.
    metric : str or callable, optional
        Metric to evaluate performance: 'mse', 'mae', 'r2', or custom callable. Default is 'mse'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'. Default is 'gradient_descent'.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    tol : float, optional
        Tolerance for stopping criterion. Default is 1e-4.
    random_state : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Estimated sources.
        - 'metrics': Performance metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings generated during computation.

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = blind_source_separation_fit(X, n_components=3)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    X_normalized, norm_params = _apply_normalization(X, normalizer)

    # Initialize parameters
    W = _initialize_parameters(n_components, X_normalized.shape[1], random_state)

    # Choose solver
    if solver == 'closed_form':
        W = _solve_closed_form(X_normalized, n_components)
    elif solver == 'gradient_descent':
        W = _solve_gradient_descent(X_normalized, n_components, max_iter, tol)
    elif solver == 'newton':
        W = _solve_newton(X_normalized, n_components, max_iter, tol)
    elif solver == 'coordinate_descent':
        W = _solve_coordinate_descent(X_normalized, n_components, max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Estimate sources
    S = np.dot(W, X_normalized.T).T

    # Compute metrics
    metrics = _compute_metrics(X, S, metric)

    # Prepare output
    result_dict = {
        'result': S,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'normalizer': normalizer,
            'metric': metric,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

    return result_dict

def _validate_inputs(X: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if X.shape[1] < n_components:
        raise ValueError("n_components cannot be greater than the number of features")

def _apply_normalization(X: np.ndarray, method: str) -> tuple:
    """Apply normalization to the input data."""
    norm_params = {}
    X_normalized = X.copy()

    if method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / std
        norm_params['mean'] = mean
        norm_params['std'] = std
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_normalized = (X - min_val) / (max_val - min_val + 1e-8)
        norm_params['min'] = min_val
        norm_params['max'] = max_val
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - median) / iqr
        norm_params['median'] = median
        norm_params['iqr'] = iqr

    return X_normalized, norm_params

def _initialize_parameters(n_components: int, n_features: int, random_state: Optional[int]) -> np.ndarray:
    """Initialize the mixing matrix."""
    if random_state is not None:
        np.random.seed(random_state)
    return np.random.randn(n_components, n_features)

def _solve_closed_form(X: np.ndarray, n_components: int) -> np.ndarray:
    """Solve ICA using closed-form solution."""
    # Placeholder for actual implementation
    return np.linalg.pinv(X.T @ X) @ X.T

def _solve_gradient_descent(
    X: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Solve ICA using gradient descent."""
    # Placeholder for actual implementation
    W = _initialize_parameters(n_components, X.shape[1], None)
    for _ in range(max_iter):
        # Update W using gradient descent
        pass
    return W

def _solve_newton(
    X: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Solve ICA using Newton's method."""
    # Placeholder for actual implementation
    W = _initialize_parameters(n_components, X.shape[1], None)
    for _ in range(max_iter):
        # Update W using Newton's method
        pass
    return W

def _solve_coordinate_descent(
    X: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Solve ICA using coordinate descent."""
    # Placeholder for actual implementation
    W = _initialize_parameters(n_components, X.shape[1], None)
    for _ in range(max_iter):
        # Update W using coordinate descent
        pass
    return W

def _compute_metrics(
    X: np.ndarray,
    S: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Compute performance metrics."""
    metrics = {}

    if metric == 'mse':
        metrics['mse'] = np.mean((X - S) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(X - S))
    elif metric == 'r2':
        ss_res = np.sum((X - S) ** 2)
        ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    elif callable(metric):
        metrics['custom'] = metric(X, S)

    return metrics

################################################################################
# linear_ica
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def linear_ica_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Perform Linear Independent Component Analysis (ICA) on the input data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of components to extract, by default 2.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the data, by default None.
    metric : str, optional
        Metric to evaluate the performance, by default 'mse'.
    distance : str, optional
        Distance metric for the solver, by default 'euclidean'.
    solver : str, optional
        Solver to use for optimization, by default 'closed_form'.
    regularization : Optional[str], optional
        Type of regularization to apply, by default None.
    tol : float, optional
        Tolerance for convergence, by default 1e-4.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    random_state : Optional[int], optional
        Random seed for reproducibility, by default None.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function, by default None.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]
        Dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = linear_ica_fit(X, n_components=3)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Normalize data if a normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize parameters
    params_used = {
        'n_components': n_components,
        'normalizer': normalizer.__name__ if normalizer else None,
        'metric': metric,
        'distance': distance,
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Choose solver and compute ICA components
    if solver == 'closed_form':
        W = _closed_form_solver(X_normalized, n_components)
    elif solver == 'gradient_descent':
        W = _gradient_descent_solver(X_normalized, n_components, distance, tol, max_iter, random_state)
    else:
        raise ValueError(f"Solver {solver} not supported.")

    # Apply regularization if specified
    if regularization:
        W = _apply_regularization(W, regularization)

    # Compute metrics
    metrics = _compute_metrics(X_normalized, W, metric, custom_metric)

    # Prepare the result dictionary
    result = {
        'result': W,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, n_components: int) -> None:
    """Validate the input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if n_components <= 0 or n_components > X.shape[1]:
        raise ValueError("n_components must be between 1 and the number of features in X.")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Apply normalization to the input data."""
    if normalizer is not None:
        X_normalized = normalizer(X)
    else:
        X_normalized = X
    return X_normalized

def _closed_form_solver(X: np.ndarray, n_components: int) -> np.ndarray:
    """Compute ICA components using closed-form solution."""
    # Placeholder for actual implementation
    return np.random.randn(X.shape[1], n_components)

def _gradient_descent_solver(
    X: np.ndarray,
    n_components: int,
    distance: str,
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> np.ndarray:
    """Compute ICA components using gradient descent."""
    if random_state is not None:
        np.random.seed(random_state)
    # Placeholder for actual implementation
    return np.random.randn(X.shape[1], n_components)

def _apply_regularization(W: np.ndarray, regularization: str) -> np.ndarray:
    """Apply regularization to the weight matrix."""
    # Placeholder for actual implementation
    return W

def _compute_metrics(
    X: np.ndarray,
    W: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for the ICA results."""
    metrics = {}
    if custom_metric:
        metrics['custom'] = custom_metric(X, W)
    else:
        if metric == 'mse':
            metrics['mse'] = np.mean((X - X @ W) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(X - X @ W))
        elif metric == 'r2':
            ss_res = np.sum((X - X @ W) ** 2)
            ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
        else:
            raise ValueError(f"Metric {metric} not supported.")
    return metrics

################################################################################
# non_linear_ica
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def non_linear_ica_fit(
    X: np.ndarray,
    n_components: int = 2,
    max_iter: int = 1000,
    tol: float = 1e-4,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    learning_rate: float = 0.01,
    batch_size: Optional[int] = None,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict:
    """
    Perform Non-Linear Independent Component Analysis (ICA).

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of components to extract. Default is 2.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    tol : float, optional
        Tolerance for stopping criterion. Default is 1e-4.
    normalizer : Callable, optional
        Function to normalize the data. Default is None (no normalization).
    metric : str or Callable, optional
        Metric to evaluate the performance. Default is 'mse'.
    distance : str or Callable, optional
        Distance metric for the solver. Default is 'euclidean'.
    solver : str, optional
        Solver to use. Options are 'gradient_descent', 'newton', 'coordinate_descent'. Default is 'gradient_descent'.
    regularization : str, optional
        Regularization type. Options are None, 'l1', 'l2', 'elasticnet'. Default is None.
    learning_rate : float, optional
        Learning rate for gradient descent. Default is 0.01.
    batch_size : int, optional
        Batch size for mini-batch gradient descent. Default is None (full batch).
    random_state : int, optional
        Random seed for reproducibility. Default is None.
    verbose : bool, optional
        Whether to print progress information. Default is False.

    Returns
    -------
    Dict
        A dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    if normalizer is not None:
        X = normalizer(X)

    # Initialize parameters
    W = np.random.randn(n_components, X.shape[1])

    # Choose solver
    if solver == 'gradient_descent':
        W = _gradient_descent_ica(X, W, n_components, max_iter, tol,
                                 metric, distance, regularization,
                                 learning_rate, batch_size, verbose)
    elif solver == 'newton':
        W = _newton_ica(X, W, n_components, max_iter, tol,
                        metric, distance, regularization, verbose)
    elif solver == 'coordinate_descent':
        W = _coordinate_descent_ica(X, W, n_components, max_iter, tol,
                                    metric, distance, regularization, verbose)
    else:
        raise ValueError("Unknown solver: {}".format(solver))

    # Compute metrics
    metrics = _compute_metrics(X, W, metric)

    # Prepare output
    result = {
        'result': {'W': W},
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'max_iter': max_iter,
            'tol': tol,
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'random_state': random_state
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if n_components <= 0 or n_components > X.shape[1]:
        raise ValueError("n_components must be between 1 and the number of features.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X must not contain NaN or Inf values.")

def _gradient_descent_ica(
    X: np.ndarray,
    W: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    learning_rate: float,
    batch_size: Optional[int],
    verbose: bool
) -> np.ndarray:
    """Perform ICA using gradient descent."""
    for iter_ in range(max_iter):
        if batch_size is None:
            X_batch = X
        else:
            idx = np.random.choice(X.shape[0], batch_size, replace=False)
            X_batch = X[idx]

        # Update W using gradient descent
        grad_W = _compute_gradient(X_batch, W, distance)
        if regularization == 'l1':
            grad_W += np.sign(W) * learning_rate
        elif regularization == 'l2':
            grad_W += W * learning_rate

        W -= learning_rate * grad_W

        # Check convergence
        if iter_ > 0 and np.linalg.norm(grad_W) < tol:
            if verbose:
                print(f"Converged at iteration {iter_}")
            break

    return W

def _newton_ica(
    X: np.ndarray,
    W: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    verbose: bool
) -> np.ndarray:
    """Perform ICA using Newton's method."""
    for iter_ in range(max_iter):
        # Compute Hessian and gradient
        grad_W = _compute_gradient(X, W, distance)
        hessian = _compute_hessian(X, W, distance)

        # Update W using Newton's method
        if regularization == 'l1':
            hessian += np.eye(W.shape[0]) * 0.1
        elif regularization == 'l2':
            hessian += np.eye(W.shape[0]) * 0.1

        W -= np.linalg.solve(hessian, grad_W)

        # Check convergence
        if iter_ > 0 and np.linalg.norm(grad_W) < tol:
            if verbose:
                print(f"Converged at iteration {iter_}")
            break

    return W

def _coordinate_descent_ica(
    X: np.ndarray,
    W: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    verbose: bool
) -> np.ndarray:
    """Perform ICA using coordinate descent."""
    for iter_ in range(max_iter):
        for i in range(W.shape[0]):
            # Update each component one by one
            W[i, :] = _update_component(X, W, i, distance, regularization)

        # Check convergence
        if iter_ > 0 and np.linalg.norm(_compute_gradient(X, W, distance)) < tol:
            if verbose:
                print(f"Converged at iteration {iter_}")
            break

    return W

def _compute_gradient(
    X: np.ndarray,
    W: np.ndarray,
    distance: Union[str, Callable]
) -> np.ndarray:
    """Compute the gradient of the objective function."""
    if callable(distance):
        dist = distance(X @ W.T)
    elif distance == 'euclidean':
        dist = np.linalg.norm(X @ W.T, axis=1)
    elif distance == 'manhattan':
        dist = np.sum(np.abs(X @ W.T), axis=1)
    elif distance == 'cosine':
        dist = 1 - np.sum(X @ W.T * X @ W.T, axis=1) / (np.linalg.norm(X @ W.T, axis=1) * np.linalg.norm(X @ W.T, axis=1))
    else:
        raise ValueError("Unknown distance metric: {}".format(distance))

    grad_W = -2 * X.T @ (X @ W.T) / dist[:, np.newaxis]
    return grad_W

def _compute_hessian(
    X: np.ndarray,
    W: np.ndarray,
    distance: Union[str, Callable]
) -> np.ndarray:
    """Compute the Hessian of the objective function."""
    if callable(distance):
        dist = distance(X @ W.T)
    elif distance == 'euclidean':
        dist = np.linalg.norm(X @ W.T, axis=1)
    elif distance == 'manhattan':
        dist = np.sum(np.abs(X @ W.T), axis=1)
    elif distance == 'cosine':
        dist = 1 - np.sum(X @ W.T * X @ W.T, axis=1) / (np.linalg.norm(X @ W.T, axis=1) * np.linalg.norm(X @ W.T, axis=1))
    else:
        raise ValueError("Unknown distance metric: {}".format(distance))

    hessian = 2 * X.T @ (X / dist[:, np.newaxis, np.newaxis]) @ X
    return hessian

def _update_component(
    X: np.ndarray,
    W: np.ndarray,
    i: int,
    distance: Union[str, Callable],
    regularization: Optional[str]
) -> np.ndarray:
    """Update a single component using coordinate descent."""
    W_i = W[i, :]
    X_proj = X @ W.T
    if callable(distance):
        dist = distance(X_proj)
    elif distance == 'euclidean':
        dist = np.linalg.norm(X_proj, axis=1)
    elif distance == 'manhattan':
        dist = np.sum(np.abs(X_proj), axis=1)
    elif distance == 'cosine':
        dist = 1 - np.sum(X_proj * X_proj, axis=1) / (np.linalg.norm(X_proj, axis=1) * np.linalg.norm(X_proj, axis=1))
    else:
        raise ValueError("Unknown distance metric: {}".format(distance))

    grad_W_i = -2 * X.T @ (X_proj[:, i] / dist) / np.sum(dist)
    if regularization == 'l1':
        grad_W_i += np.sign(W_i) * 0.1
    elif regularization == 'l2':
        grad_W_i += W_i * 0.1

    return W_i - grad_W_i

def _compute_metrics(
    X: np.ndarray,
    W: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Compute the metrics for the ICA results."""
    X_proj = X @ W.T
    if callable(metric):
        return {'custom_metric': metric(X_proj)}
    elif metric == 'mse':
        return {'mse': np.mean((X - X_proj @ W) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(X - X_proj @ W))}
    elif metric == 'r2':
        ss_res = np.sum((X - X_proj @ W) ** 2)
        ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
        return {'r2': 1 - ss_res / ss_tot}
    elif metric == 'logloss':
        return {'logloss': -np.mean(X * np.log(X_proj) + (1 - X) * np.log(1 - X_proj))}
    else:
        raise ValueError("Unknown metric: {}".format(metric))

################################################################################
# contrast_function
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def contrast_function_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit the contrast function for Independent Component Analysis (ICA).

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of components to extract. Default is 2.
    normalize : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'. Default is 'standard'.
    metric : Union[str, Callable], optional
        Metric to optimize: 'mse', 'mae', 'r2', or custom callable. Default is 'mse'.
    distance : Union[str, Callable], optional
        Distance metric: 'euclidean', 'manhattan', 'cosine', or custom callable. Default is 'euclidean'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'. Default is 'gradient_descent'.
    regularization : Optional[str], optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'. Default is None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    learning_rate : float, optional
        Learning rate for gradient-based solvers. Default is 0.01.
    custom_metric : Optional[Callable], optional
        Custom metric function if not using built-in metrics. Default is None.
    custom_distance : Optional[Callable], optional
        Custom distance function if not using built-in distances. Default is None.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Estimated components.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = contrast_function_fit(X, n_components=3, normalize='standard', metric='mse')
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Normalize data
    X_normalized = _normalize_data(X, normalize)

    # Initialize parameters
    W = np.random.randn(n_components, X_normalized.shape[1])

    # Choose solver
    if solver == 'closed_form':
        W = _solve_closed_form(X_normalized, n_components)
    elif solver == 'gradient_descent':
        W = _solve_gradient_descent(
            X_normalized, n_components, metric, distance,
            regularization, tol, max_iter, learning_rate,
            custom_metric, custom_distance
        )
    elif solver == 'newton':
        W = _solve_newton(X_normalized, n_components)
    elif solver == 'coordinate_descent':
        W = _solve_coordinate_descent(X_normalized, n_components)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, W, metric, custom_metric)

    # Prepare output
    result = {
        'result': W,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'normalize': normalize,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter,
            'learning_rate': learning_rate
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if n_components <= 0 or n_components > X.shape[1]:
        raise ValueError("n_components must be between 1 and n_features")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize the input data."""
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

def _solve_closed_form(X: np.ndarray, n_components: int) -> np.ndarray:
    """Solve using closed-form solution."""
    # Placeholder for actual implementation
    return np.linalg.svd(X, full_matrices=False)[:n_components]

def _solve_gradient_descent(
    X: np.ndarray,
    n_components: int,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    learning_rate: float,
    custom_metric: Optional[Callable],
    custom_distance: Optional[Callable]
) -> np.ndarray:
    """Solve using gradient descent."""
    W = np.random.randn(n_components, X.shape[1])
    for _ in range(max_iter):
        gradient = _compute_gradient(X, W, metric, distance, custom_metric, custom_distance)
        if regularization == 'l1':
            gradient += np.sign(W) * learning_rate
        elif regularization == 'l2':
            gradient += 2 * W * learning_rate
        W -= gradient * learning_rate
    return W

def _solve_newton(X: np.ndarray, n_components: int) -> np.ndarray:
    """Solve using Newton's method."""
    # Placeholder for actual implementation
    return np.random.randn(n_components, X.shape[1])

def _solve_coordinate_descent(X: np.ndarray, n_components: int) -> np.ndarray:
    """Solve using coordinate descent."""
    # Placeholder for actual implementation
    return np.random.randn(n_components, X.shape[1])

def _compute_gradient(
    X: np.ndarray,
    W: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    custom_metric: Optional[Callable],
    custom_distance: Optional[Callable]
) -> np.ndarray:
    """Compute the gradient for the given metric and distance."""
    if custom_metric is not None:
        return _custom_gradient(X, W, custom_metric, custom_distance)
    elif metric == 'mse':
        return _mse_gradient(X, W, distance)
    elif metric == 'mae':
        return _mae_gradient(X, W, distance)
    elif metric == 'r2':
        return _r2_gradient(X, W, distance)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _custom_gradient(
    X: np.ndarray,
    W: np.ndarray,
    metric: Callable,
    distance: Optional[Callable]
) -> np.ndarray:
    """Compute gradient using custom metric."""
    # Placeholder for actual implementation
    return np.random.randn(*W.shape)

def _mse_gradient(X: np.ndarray, W: np.ndarray, distance: Union[str, Callable]) -> np.ndarray:
    """Compute MSE gradient."""
    # Placeholder for actual implementation
    return 2 * np.dot(W, X.T) / X.shape[0]

def _mae_gradient(X: np.ndarray, W: np.ndarray, distance: Union[str, Callable]) -> np.ndarray:
    """Compute MAE gradient."""
    # Placeholder for actual implementation
    return np.sign(np.dot(W, X.T)) / X.shape[0]

def _r2_gradient(X: np.ndarray, W: np.ndarray, distance: Union[str, Callable]) -> np.ndarray:
    """Compute R2 gradient."""
    # Placeholder for actual implementation
    return 2 * np.dot(W, X.T) / (X.shape[0] * np.var(X))

def _compute_metrics(
    X: np.ndarray,
    W: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute metrics for the given components."""
    if custom_metric is not None:
        return {'custom': custom_metric(X, W)}
    elif metric == 'mse':
        return {'mse': np.mean((np.dot(X, W.T) - X) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(np.dot(X, W.T) - X))}
    elif metric == 'r2':
        ss_res = np.sum((X - np.dot(X, W.T)) ** 2)
        ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
        return {'r2': 1 - ss_res / ss_tot}
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# gradient_ascent
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(W: np.ndarray, X: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if not isinstance(W, np.ndarray) or not isinstance(X, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if W.ndim != 2 or X.ndim != 2:
        raise ValueError("Inputs must be 2-dimensional arrays")
    if W.shape[1] != X.shape[0]:
        raise ValueError("Incompatible dimensions between W and X")
    if np.any(np.isnan(W)) or np.any(np.isinf(W)):
        raise ValueError("W contains NaN or infinite values")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")

def normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'none':
        return X
    elif method == 'standard':
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        return (X - mean) / std
    elif method == 'minmax':
        min_val = np.min(X, axis=1, keepdims=True)
        max_val = np.max(X, axis=1, keepdims=True)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=1, keepdims=True)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=1))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_metric(W: np.ndarray, X: np.ndarray, metric: str = 'mse', custom_metric: Optional[Callable] = None) -> float:
    """Compute specified metric between W and X."""
    if custom_metric is not None:
        return custom_metric(W, X)
    if metric == 'mse':
        return np.mean((W @ X) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(W @ X))
    elif metric == 'r2':
        y_pred = W @ X
        ss_res = np.sum((y_pred - np.mean(y_pred)) ** 2)
        ss_tot = np.sum((X - np.mean(X, axis=1, keepdims=True)) ** 2)
        return 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def gradient_ascent_step(W: np.ndarray, X: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
    """Perform a single gradient ascent step."""
    gradient = 2 * (W @ X) / X.shape[1]
    return W + learning_rate * gradient

def check_convergence(metric_history: np.ndarray, tol: float = 1e-4) -> bool:
    """Check if the algorithm has converged."""
    return np.abs(np.diff(metric_history[-2:])) < tol

def gradient_ascent_fit(
    X: np.ndarray,
    n_components: int = 2,
    max_iter: int = 1000,
    tol: float = 1e-4,
    learning_rate: float = 0.01,
    normalization: str = 'standard',
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Perform Independent Component Analysis using gradient ascent.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_features, n_samples)
    n_components : int, optional
        Number of components to extract, by default 2
    max_iter : int, optional
        Maximum number of iterations, by default 1000
    tol : float, optional
        Tolerance for convergence, by default 1e-4
    learning_rate : float, optional
        Learning rate for gradient ascent, by default 0.01
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), by default 'standard'
    metric : str, optional
        Metric to optimize ('mse', 'mae', 'r2'), by default 'mse'
    custom_metric : Callable, optional
        Custom metric function, by default None
    random_state : int, optional
        Random seed for reproducibility, by default None

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Initialize random state if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Validate inputs
    validate_inputs(np.zeros((n_components, X.shape[0])), X)

    # Normalize data
    X_norm = normalize_data(X, normalization)

    # Initialize W randomly
    W = np.random.randn(n_components, X.shape[0])

    # Initialize metric history
    metric_history = []

    for _ in range(max_iter):
        # Compute current metric
        current_metric = compute_metric(W, X_norm, metric, custom_metric)
        metric_history.append(current_metric)

        # Check convergence
        if len(metric_history) > 1 and check_convergence(np.array(metric_history)):
            break

        # Perform gradient ascent step
        W = gradient_ascent_step(W, X_norm, learning_rate)

    # Compute final metrics
    final_metric = compute_metric(W, X_norm, metric, custom_metric)

    return {
        "result": W,
        "metrics": {"final_metric": final_metric, "metric_history": metric_history},
        "params_used": {
            "n_components": n_components,
            "max_iter": max_iter,
            "tol": tol,
            "learning_rate": learning_rate,
            "normalization": normalization,
            "metric": metric
        },
        "warnings": []
    }

# Example usage:
"""
X = np.random.randn(10, 100)
result = gradient_ascent_fit(X, n_components=3, max_iter=500, learning_rate=0.01)
"""

################################################################################
# orthogonalization
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def orthogonalization_fit(
    X: np.ndarray,
    normalize: str = 'standard',
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
    Perform orthogonalization of input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix (n_samples, n_features)
    normalize : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable
        Metric to optimize ('mse', 'mae', 'r2', custom callable)
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
        Custom metric function if using 'custom'
    custom_distance : callable or None
        Custom distance function if using 'custom'

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X)

    # Normalize data
    X_norm = _apply_normalization(X, normalize)

    # Initialize parameters
    params = {
        'normalize': normalize,
        'metric': metric,
        'distance': distance,
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Choose solver
    if solver == 'closed_form':
        W = _closed_form_solver(X_norm)
    elif solver == 'gradient_descent':
        W = _gradient_descent_solver(X_norm, metric, distance,
                                    regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(X_norm, W, metric)

    return {
        'result': {'components': W},
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

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

def _apply_normalization(
    X: np.ndarray,
    method: str
) -> np.ndarray:
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

def _closed_form_solver(
    X: np.ndarray
) -> np.ndarray:
    """Closed form solution for orthogonalization."""
    return np.linalg.qr(X.T)[0].T

def _gradient_descent_solver(
    X: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent solver for orthogonalization."""
    n_features = X.shape[1]
    W = np.eye(n_features)

    for _ in range(max_iter):
        # Update rule would go here
        pass

    return W

def _calculate_metrics(
    X: np.ndarray,
    W: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate metrics for orthogonalization."""
    if callable(metric):
        return {'custom_metric': metric(X, W)}
    elif metric == 'mse':
        return {'mse': np.mean((X - X @ W.T) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(X - X @ W.T))}
    elif metric == 'r2':
        ss_res = np.sum((X - X @ W.T) ** 2)
        ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
        return {'r2': 1 - ss_res / ss_tot}
    else:
        raise ValueError(f"Unknown metric: {metric}")

# Example usage
"""
X = np.random.randn(100, 5)
result = orthogonalization_fit(X,
                             normalize='standard',
                             metric='mse',
                             solver='closed_form')
"""

################################################################################
# stability
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def stability_fit(
    X: np.ndarray,
    n_components: int = 2,
    n_bootstrap: int = 100,
    metric: Union[str, Callable] = 'correlation',
    normalization: str = 'standard',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'fastica',
    random_state: Optional[int] = None,
    **kwargs
) -> Dict:
    """
    Compute stability of ICA components across bootstrap samples.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of ICA components to extract.
    n_bootstrap : int, optiona
        Number of bootstrap samples to generate.
    metric : str or callable, optional
        Metric for stability computation ('correlation', 'mse', custom callable).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    distance : str or callable, optional
        Distance metric for stability computation.
    solver : str, optional
        ICA solver to use ('fastica', 'picard').
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict containing:
        - 'result': Stability scores
        - 'metrics': Computed metrics
        - 'params_used': Parameters used
        - 'warnings': Potential warnings

    Example
    -------
    >>> X = np.random.randn(100, 20)
    >>> result = stability_fit(X, n_components=5)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Set random state
    rng = np.random.RandomState(random_state)

    # Normalize data
    X_norm, norm_params = _apply_normalization(X, normalization)

    # Initialize results dictionary
    results = {
        'result': None,
        'metrics': {},
        'params_used': {
            'n_components': n_components,
            'n_bootstrap': n_bootstrap,
            'metric': metric,
            'normalization': normalization,
            'distance': distance,
            'solver': solver
        },
        'warnings': []
    }

    # Compute stability scores
    stability_scores = _compute_stability(
        X_norm,
        n_components=n_components,
        n_bootstrap=n_bootstrap,
        metric=metric,
        distance=distance,
        solver=solver,
        rng=rng
    )

    # Store results
    results['result'] = stability_scores

    return results

def _validate_inputs(X: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if n_components > X.shape[1]:
        raise ValueError("n_components cannot exceed number of features")

def _apply_normalization(X: np.ndarray, method: str) -> tuple:
    """Apply specified normalization to data."""
    X_norm = X.copy()
    norm_params = {}

    if method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
        norm_params['mean'] = mean
        norm_params['std'] = std

    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
        norm_params['min'] = min_val
        norm_params['max'] = max_val

    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
        norm_params['median'] = median
        norm_params['iqr'] = iqr

    return X_norm, norm_params

def _compute_stability(
    X: np.ndarray,
    n_components: int,
    n_bootstrap: int,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str,
    rng: np.random.RandomState
) -> np.ndarray:
    """Compute stability scores across bootstrap samples."""
    # Generate bootstrap samples
    bootstrap_samples = _generate_bootstrap_samples(X, n_bootstrap, rng)

    # Compute ICA components for each sample
    components_list = []
    for sample in bootstrap_samples:
        if solver == 'fastica':
            comps = _fastica_solver(sample, n_components)
        elif solver == 'picard':
            comps = _picard_solver(sample, n_components)
        components_list.append(comps)

    # Compute stability scores
    if callable(metric):
        return metric(components_list, distance)
    elif metric == 'correlation':
        return _compute_correlation_stability(components_list, distance)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _generate_bootstrap_samples(X: np.ndarray, n_samples: int, rng: np.random.RandomState) -> list:
    """Generate bootstrap samples from input data."""
    n_samples, n_features = X.shape
    return [X[rng.choice(n_samples, size=n_samples, replace=True)] for _ in range(n_samples)]

def _fastica_solver(X: np.ndarray, n_components: int) -> np.ndarray:
    """FastICA solver implementation."""
    # Placeholder for actual FastICA implementation
    return np.random.randn(n_components, X.shape[1])

def _picard_solver(X: np.ndarray, n_components: int) -> np.ndarray:
    """Picard solver implementation."""
    # Placeholder for actual Picard implementation
    return np.random.randn(n_components, X.shape[1])

def _compute_correlation_stability(components_list: list, distance: Union[str, Callable]) -> np.ndarray:
    """Compute stability based on correlation between components."""
    n_components = len(components_list[0])
    stability_scores = np.zeros(n_components)

    for i in range(n_components):
        # Collect all components for this index across bootstrap samples
        comps = np.array([comp[i] for comp in components_list])

        # Compute pairwise distances
        if callable(distance):
            dist_matrix = distance(comps)
        elif distance == 'euclidean':
            dist_matrix = np.sqrt(np.sum((comps[:, None] - comps) ** 2, axis=2))
        elif distance == 'cosine':
            dot_products = np.dot(comps, comps.T)
            norms = np.linalg.norm(comps, axis=1)[:, None]
            dist_matrix = 1 - dot_products / (norms * norms.T)
        else:
            raise ValueError(f"Unknown distance: {distance}")

        # Convert distances to similarities
        similarity_matrix = 1 / (1 + dist_matrix)

        # Compute average stability for this component
        stability_scores[i] = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])

    return stability_scores

################################################################################
# convergence_criteria
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def convergence_criteria_fit(
    X: np.ndarray,
    W_prev: np.ndarray,
    W_current: np.ndarray,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Compute convergence criteria for ICA algorithms.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    W_prev : np.ndarray
        Previous unmixing matrix of shape (n_components, n_features).
    W_current : np.ndarray
        Current unmixing matrix of shape (n_components, n_features).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to compute convergence ('mse', 'mae', 'r2', custom callable).
    distance : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', custom callable).
    solver : str, optional
        Solver method ('gradient_descent', 'newton', 'coordinate_descent').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    custom_distance : callable, optional
        Custom distance function if not using built-in distances.

    Returns
    -------
    Dict[str, Union[float, np.ndarray, Dict]]
        Dictionary containing:
        - 'result': Convergence result (bool)
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in computation
        - 'warnings': Any warnings encountered

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> W_prev = np.eye(3, 5)
    >>> W_current = np.random.randn(3, 5)
    >>> result = convergence_criteria_fit(X, W_prev, W_current)
    """
    # Validate inputs
    _validate_inputs(X, W_prev, W_current)

    # Normalize data if required
    X_normalized = _apply_normalization(X, normalization)

    # Compute convergence criteria based on solver
    if solver == 'gradient_descent':
        result, metrics = _compute_gradient_descent_criteria(
            X_normalized, W_prev, W_current, metric, distance,
            custom_metric=custom_metric, custom_distance=custom_distance
        )
    elif solver == 'newton':
        result, metrics = _compute_newton_criteria(
            X_normalized, W_prev, W_current, metric, distance,
            custom_metric=custom_metric, custom_distance=custom_distance
        )
    elif solver == 'coordinate_descent':
        result, metrics = _compute_coordinate_descent_criteria(
            X_normalized, W_prev, W_current, metric, distance,
            custom_metric=custom_metric, custom_distance=custom_distance
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Check convergence
    converged = metrics['value'] < tol or max_iter == 0

    return {
        'result': converged,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, W_prev: np.ndarray, W_current: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(W_prev, np.ndarray) or not isinstance(W_current, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")

    if X.ndim != 2 or W_prev.ndim != 2 or W_current.ndim != 2:
        raise ValueError("Inputs must be 2-dimensional arrays")

    if W_prev.shape[0] != W_current.shape[0]:
        raise ValueError("W_prev and W_current must have the same number of components")

    if X.shape[1] != W_prev.shape[1]:
        raise ValueError("Number of features in X must match number of features in W_prev")

    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to the input data."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        return (X - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_gradient_descent_criteria(
    X: np.ndarray,
    W_prev: np.ndarray,
    W_current: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> tuple:
    """Compute convergence criteria for gradient descent solver."""
    if custom_metric is not None:
        metric_value = custom_metric(W_prev, W_current)
    elif isinstance(metric, str):
        if metric == 'mse':
            metric_value = np.mean((W_prev - W_current) ** 2)
        elif metric == 'mae':
            metric_value = np.mean(np.abs(W_prev - W_current))
        elif metric == 'r2':
            ss_res = np.sum((W_prev - W_current) ** 2)
            ss_tot = np.sum(W_prev ** 2)
            metric_value = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.inf
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        raise TypeError("Metric must be either a string or callable")

    if custom_distance is not None:
        distance_value = custom_distance(W_prev, W_current)
    elif isinstance(distance, str):
        if distance == 'euclidean':
            distance_value = np.linalg.norm(W_prev - W_current)
        elif distance == 'manhattan':
            distance_value = np.sum(np.abs(W_prev - W_current))
        elif distance == 'cosine':
            distance_value = 1 - np.dot(W_prev.flatten(), W_current.flatten()) / (
                np.linalg.norm(W_prev) * np.linalg.norm(W_current)
            )
        elif distance == 'minkowski':
            p = 3
            distance_value = np.sum(np.abs(W_prev - W_current) ** p) ** (1/p)
        else:
            raise ValueError(f"Unknown distance: {distance}")
    else:
        raise TypeError("Distance must be either a string or callable")

    return True, {
        'value': metric_value,
        'distance': distance_value,
        'metric_type': str(metric),
        'distance_type': str(distance)
    }

def _compute_newton_criteria(
    X: np.ndarray,
    W_prev: np.ndarray,
    W_current: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> tuple:
    """Compute convergence criteria for Newton solver."""
    return _compute_gradient_descent_criteria(
        X, W_prev, W_current, metric, distance,
        custom_metric=custom_metric, custom_distance=custom_distance
    )

def _compute_coordinate_descent_criteria(
    X: np.ndarray,
    W_prev: np.ndarray,
    W_current: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> tuple:
    """Compute convergence criteria for coordinate descent solver."""
    return _compute_gradient_descent_criteria(
        X, W_prev, W_current, metric, distance,
        custom_metric=custom_metric, custom_distance=custom_distance
    )

################################################################################
# initialization_methods
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def initialization_methods_fit(
    data: np.ndarray,
    n_components: int = 2,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Main function for ICA initialization methods.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int, optional
        Number of components to extract (default: 2)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard')
    metric : str or callable, optional
        Metric to optimize ('mse', 'mae', 'r2', custom callable) (default: 'mse')
    distance : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', custom callable) (default: 'euclidean')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent') (default: 'closed_form')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2') (default: None)
    tol : float, optional
        Tolerance for convergence (default: 1e-4)
    max_iter : int, optional
        Maximum number of iterations (default: 1000)
    random_state : int, optional
        Random seed for reproducibility (default: None)
    custom_metric : callable, optional
        Custom metric function if needed (default: None)
    custom_distance : callable, optional
        Custom distance function if needed (default: None)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.random.randn(100, 5)
    >>> result = initialization_methods_fit(data, n_components=3)
    """
    # Validate inputs
    _validate_inputs(data, n_components)

    # Set random state if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    normalized_data = _apply_normalization(data, normalization)

    # Initialize components
    if solver == 'closed_form':
        components = _closed_form_initialization(normalized_data, n_components)
    else:
        components = _gradient_descent_initialization(
            normalized_data,
            n_components,
            metric if isinstance(metric, str) else custom_metric,
            distance if isinstance(distance, str) else custom_distance,
            tol,
            max_iter
        )

    # Apply regularization if needed
    if regularization is not None:
        components = _apply_regularization(components, regularization)

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, components, metric)

    return {
        'result': components,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
            'distance': distance if isinstance(distance, str) else 'custom',
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(normalized_data, components)
    }

def _validate_inputs(data: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if n_components <= 0 or n_components > data.shape[1]:
        raise ValueError("n_components must be between 1 and number of features")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to data."""
    if method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _closed_form_initialization(data: np.ndarray, n_components: int) -> np.ndarray:
    """Closed form initialization for ICA components."""
    # This is a placeholder - actual implementation would use PCA or similar
    cov = np.cov(data, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs[:, -n_components:]

def _gradient_descent_initialization(
    data: np.ndarray,
    n_components: int,
    metric: Callable,
    distance: Callable,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent initialization for ICA components."""
    # Initialize random components
    components = np.random.randn(data.shape[1], n_components)

    for _ in range(max_iter):
        # Update components using gradient descent
        # This is a simplified version - actual implementation would be more complex
        grad = _compute_gradient(data, components, metric, distance)
        components -= tol * grad

    return components

def _compute_gradient(
    data: np.ndarray,
    components: np.ndarray,
    metric: Callable,
    distance: Callable
) -> np.ndarray:
    """Compute gradient for ICA components."""
    # Placeholder implementation
    return np.random.randn(*components.shape)

def _apply_regularization(components: np.ndarray, method: str) -> np.ndarray:
    """Apply regularization to components."""
    if method == 'l1':
        return np.sign(components) * np.maximum(np.abs(components) - 0.1, 0)
    elif method == 'l2':
        return components / (np.sqrt(np.sum(components**2, axis=0)) + 1e-8)
    else:
        raise ValueError(f"Unknown regularization method: {method}")

def _calculate_metrics(
    data: np.ndarray,
    components: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate metrics for the initialization."""
    if isinstance(metric, str):
        if metric == 'mse':
            return {'mse': np.mean((data @ components) ** 2)}
        elif metric == 'mae':
            return {'mae': np.mean(np.abs(data @ components))}
        elif metric == 'r2':
            ss_total = np.sum((data - np.mean(data, axis=0))**2)
            ss_residual = np.sum((data - (data @ components))**2)
            return {'r2': 1 - ss_residual / ss_total}
    else:
        return {metric.__name__: metric(data, components)}

def _check_warnings(data: np.ndarray, components: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isnan(data)):
        warnings.append("Input data contains NaN values")
    if np.any(np.isinf(components)):
        warnings.append("Components contain infinite values")
    return warnings

################################################################################
# learning_rate
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def learning_rate_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Compute the optimal learning rate for ICA using various optimization methods.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the input data.
    metric : str
        Metric to optimize ('mse', 'mae', 'r2', 'logloss').
    distance : str
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski').
    solver : str
        Optimization solver ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : Optional[str]
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    learning_rate : float
        Initial learning rate.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.
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
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = learning_rate_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = normalizer(X)

    # Initialize parameters
    params = {
        'learning_rate': learning_rate,
        'max_iter': max_iter,
        'tol': tol,
        'solver': solver,
        'metric': metric,
        'distance': distance,
        'regularization': regularization
    }

    # Choose metric function
    metric_func = _get_metric_function(metric, custom_metric)

    # Choose distance function
    distance_func = _get_distance_function(distance, custom_distance)

    # Solve for optimal learning rate
    result = _solve_learning_rate(
        X_normalized, y,
        metric_func, distance_func,
        solver, regularization,
        learning_rate, max_iter, tol
    )

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, result['params'], metric_func)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

    return output

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
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

def _get_metric_function(
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the appropriate metric function based on user input."""
    if custom_metric is not None:
        return custom_metric
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared,
        'logloss': _log_loss
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance_function(
    distance: str,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the appropriate distance function based on user input."""
    if custom_distance is not None:
        return custom_distance
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': _minkowski_distance
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _solve_learning_rate(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    solver: str,
    regularization: Optional[str],
    learning_rate: float,
    max_iter: int,
    tol: float
) -> Dict[str, Union[np.ndarray, float]]:
    """Solve for optimal learning rate using the specified solver."""
    solvers = {
        'closed_form': _closed_form_solution,
        'gradient_descent': _gradient_descent,
        'newton': _newton_method,
        'coordinate_descent': _coordinate_descent
    }
    if solver not in solvers:
        raise ValueError(f"Unknown solver: {solver}")
    return solvers[solver](
        X, y,
        metric_func, distance_func,
        regularization,
        learning_rate, max_iter, tol
    )

def _closed_form_solution(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    learning_rate: float,
    max_iter: int,
    tol: float
) -> Dict[str, Union[np.ndarray, float]]:
    """Closed form solution for learning rate optimization."""
    # Implement closed form solution logic
    pass

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    learning_rate: float,
    max_iter: int,
    tol: float
) -> Dict[str, Union[np.ndarray, float]]:
    """Gradient descent for learning rate optimization."""
    # Implement gradient descent logic
    pass

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    learning_rate: float,
    max_iter: int,
    tol: float
) -> Dict[str, Union[np.ndarray, float]]:
    """Newton's method for learning rate optimization."""
    # Implement Newton's method logic
    pass

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    learning_rate: float,
    max_iter: int,
    tol: float
) -> Dict[str, Union[np.ndarray, float]]:
    """Coordinate descent for learning rate optimization."""
    # Implement coordinate descent logic
    pass

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, float],
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Compute various metrics for the optimized learning rate."""
    # Compute and return metrics
    pass

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

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _minkowski_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Minkowski distance."""
    return np.sum(np.abs(a - b) ** 3) ** (1/3)

################################################################################
# regularization
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regularization_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: str = 'none',
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Fit a regularized model using Independent Component Analysis (ICA) approach.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the input data
    metric : str
        Metric to evaluate model performance ('mse', 'mae', 'r2', 'logloss')
    distance : str
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    solver : str
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    regularization : str
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    alpha : float
        Regularization strength
    l1_ratio : float
        ElasticNet mixing parameter (0 = L2, 1 = L1)
    tol : float
        Tolerance for stopping criteria
    max_iter : int
        Maximum number of iterations
    custom_metric : Optional[Callable]
        Custom metric function if needed
    custom_distance : Optional[Callable]
        Custom distance function if needed

    Returns:
    --------
    Dict containing:
        - 'result': Estimated parameters
        - 'metrics': Performance metrics
        - 'params_used': Parameters used in fitting
        - 'warnings': Any warnings generated during fitting

    Example:
    --------
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randn(100)
    >>> result = regularization_fit(X, y, metric='mse', solver='gradient_descent')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = normalizer(X)
    y_normalized = normalizer(y.reshape(-1, 1)).flatten()

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
    if solver == 'closed_form':
        params = _closed_form_solver(X_normalized, y_normalized)
    elif solver == 'gradient_descent':
        params = _gradient_descent_solver(
            X_normalized, y_normalized,
            metric_func, distance_func,
            regularization, alpha, l1_ratio,
            tol, max_iter
        )
    elif solver == 'newton':
        params = _newton_solver(
            X_normalized, y_normalized,
            metric_func, distance_func,
            regularization, alpha, l1_ratio,
            tol, max_iter
        )
    elif solver == 'coordinate_descent':
        params = _coordinate_descent_solver(
            X_normalized, y_normalized,
            metric_func, distance_func,
            regularization, alpha, l1_ratio,
            tol, max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(y_normalized, X_normalized @ params)

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'alpha': alpha,
            'l1_ratio': l1_ratio
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _get_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the appropriate metric function."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared,
        'logloss': _log_loss
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the appropriate distance function."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': lambda x, y: np.sum(np.abs(x - y)**2, axis=1)**(1/3)
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form solution for regularized linear regression."""
    return np.linalg.pinv(X) @ y

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: str,
    alpha: float,
    l1_ratio: float,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent solver with regularization."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradients = _compute_gradients(X, y, params, metric_func, distance_func)
        if regularization == 'l1':
            gradients += alpha * np.sign(params)
        elif regularization == 'l2':
            gradients += 2 * alpha * params
        elif regularization == 'elasticnet':
            gradients += alpha * (l1_ratio * np.sign(params) + (1 - l1_ratio) * 2 * params)

        params -= learning_rate * gradients

        if np.linalg.norm(gradients) < tol:
            break

    return params

def _newton_solver(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: str,
    alpha: float,
    l1_ratio: float,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Newton's method solver with regularization."""
    n_features = X.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        gradients = _compute_gradients(X, y, params, metric_func, distance_func)
        hessian = _compute_hessian(X, y, params, metric_func)

        if regularization == 'l1':
            hessian += alpha * np.eye(n_features)
        elif regularization == 'l2':
            hessian += 2 * alpha * np.eye(n_features)
        elif regularization == 'elasticnet':
            hessian += alpha * (l1_ratio * np.eye(n_features) + (1 - l1_ratio) * 2 * np.eye(n_features))

        params -= np.linalg.solve(hessian, gradients)

        if np.linalg.norm(gradients) < tol:
            break

    return params

def _coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: str,
    alpha: float,
    l1_ratio: float,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Coordinate descent solver with regularization."""
    n_features = X.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residual = y - (X @ params) + params[j] * X_j

            if regularization == 'l1':
                params[j] = _soft_threshold(residual @ X_j, alpha)
            elif regularization == 'l2':
                params[j] = (residual @ X_j) / (X_j @ X_j + 2 * alpha)
            elif regularization == 'elasticnet':
                params[j] = _elasticnet_threshold(residual @ X_j, alpha, l1_ratio)
            else:
                params[j] = (residual @ X_j) / (X_j @ X_j)

        if np.linalg.norm(_compute_gradients(X, y, params, metric_func, distance_func)) < tol:
            break

    return params

def _compute_gradients(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Compute gradients for the given parameters."""
    predictions = X @ params
    if metric_func == _mean_squared_error:
        return -2 * (y - predictions) @ X
    elif metric_func == _mean_absolute_error:
        return -np.sign(y - predictions) @ X
    elif metric_func == _r_squared:
        return -2 * (y - predictions) @ X
    elif metric_func == _log_loss:
        return (-y / (1 + np.exp(y * predictions)) + (1 - y) / (1 + np.exp(-y * predictions))) @ X
    else:
        raise ValueError("Unknown metric function")

def _compute_hessian(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Compute Hessian matrix for the given parameters."""
    if metric_func == _mean_squared_error:
        return 2 * X.T @ X
    elif metric_func == _mean_absolute_error:
        return np.zeros((X.shape[1], X.shape[1]))
    elif metric_func == _r_squared:
        return 2 * X.T @ X
    else:
        raise ValueError("Hessian computation not implemented for this metric")

def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate various metrics for model evaluation."""
    return {
        'mse': _mean_squared_error(y_true, y_pred),
        'mae': _mean_absolute_error(y_true, y_pred),
        'r2': _r_squared(y_true, y_pred)
    }

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
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.sqrt(np.sum((x - y) ** 2, axis=1))

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(x - y), axis=1)

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cosine distance."""
    return 1 - (x @ y.T) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))

def _soft_threshold(value: float, threshold: float) -> float:
    """Soft thresholding function for L1 regularization."""
    if value > threshold:
        return value - threshold
    elif value < -threshold:
        return value + threshold
    else:
        return 0

def _elasticnet_threshold(value: float, alpha: float, l1_ratio: float) -> float:
    """Thresholding function for ElasticNet regularization."""
    l1 = alpha * l1_ratio
    l2 = alpha * (1 - l1_ratio)
    if value > l1:
        return (value - l1) / (1 + 2 * l2)
    elif value < -l1:
        return (value + l1) / (1 + 2 * l2)
    else:
        return 0

################################################################################
# noise_modeling
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def noise_modeling_fit(
    data: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Fit a noise modeling algorithm to the given data.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Function to normalize the data. Default is identity function.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to evaluate the model. Can be 'mse', 'mae', 'r2', or a custom callable.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Distance metric for the solver. Can be 'euclidean', 'manhattan', 'cosine', or a custom callable.
    solver : str, optional
        Solver to use. Options are 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : Optional[str], optional
        Regularization type. Options are 'l1', 'l2', or None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function. If provided, overrides the `metric` parameter.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function. If provided, overrides the `distance` parameter.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]
        Dictionary containing:
        - "result": Estimated parameters.
        - "metrics": Computed metrics.
        - "params_used": Parameters used during fitting.
        - "warnings": List of warnings encountered.

    Examples
    --------
    >>> data = np.random.randn(100, 5)
    >>> result = noise_modeling_fit(data, metric='mse', solver='gradient_descent')
    """
    # Validate inputs
    _validate_inputs(data, normalizer, metric, distance, solver, regularization)

    # Normalize data
    normalized_data = normalizer(data)

    # Prepare metric and distance functions
    metric_func, distance_func = _prepare_functions(metric, distance, custom_metric, custom_distance)

    # Initialize parameters
    params = _initialize_parameters(normalized_data.shape[1])

    # Solve the problem
    result, metrics, warnings = _solve(
        normalized_data,
        params,
        metric_func,
        distance_func,
        solver,
        regularization,
        tol,
        max_iter
    )

    return {
        "result": result,
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
        "warnings": warnings
    }

def _validate_inputs(
    data: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray],
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    solver: str,
    regularization: Optional[str]
) -> None:
    """Validate the inputs."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Data contains infinite values.")

    valid_solvers = ['gradient_descent', 'newton', 'coordinate_descent']
    if solver not in valid_solvers:
        raise ValueError(f"Solver must be one of {valid_solvers}.")

    valid_regularizations = [None, 'l1', 'l2']
    if regularization not in valid_regularizations:
        raise ValueError(f"Regularization must be one of {valid_regularizations}.")

def _prepare_functions(
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> tuple:
    """Prepare metric and distance functions."""
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric_function(metric)

    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_function(distance)

    return metric_func, distance_func

def _get_metric_function(metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the metric function based on the input."""
    if isinstance(metric, str):
        if metric == 'mse':
            return _mse
        elif metric == 'mae':
            return _mae
        elif metric == 'r2':
            return _r2
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        return metric
    else:
        raise ValueError("Metric must be a string or callable.")

def _get_distance_function(distance: Union[str, Callable[[np.ndarray, np.ndarray], float]]) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the distance function based on the input."""
    if isinstance(distance, str):
        if distance == 'euclidean':
            return _euclidean_distance
        elif distance == 'manhattan':
            return _manhattan_distance
        elif distance == 'cosine':
            return _cosine_distance
        else:
            raise ValueError(f"Unknown distance: {distance}")
    elif callable(distance):
        return distance
    else:
        raise ValueError("Distance must be a string or callable.")

def _initialize_parameters(n_features: int) -> np.ndarray:
    """Initialize parameters."""
    return np.zeros(n_features)

def _solve(
    data: np.ndarray,
    params: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple:
    """Solve the noise modeling problem."""
    warnings = []
    metrics = {}

    if solver == 'gradient_descent':
        result, metrics, warnings = _gradient_descent(
            data,
            params,
            metric_func,
            distance_func,
            regularization,
            tol,
            max_iter
        )
    elif solver == 'newton':
        result, metrics, warnings = _newton_method(
            data,
            params,
            metric_func,
            distance_func,
            regularization,
            tol,
            max_iter
        )
    elif solver == 'coordinate_descent':
        result, metrics, warnings = _coordinate_descent(
            data,
            params,
            metric_func,
            distance_func,
            regularization,
            tol,
            max_iter
        )

    return result, metrics, warnings

def _gradient_descent(
    data: np.ndarray,
    params: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple:
    """Gradient descent solver."""
    # Placeholder for actual implementation
    result = params.copy()
    metrics = {'metric': metric_func(data, result)}
    warnings = []
    return result, metrics, warnings

def _newton_method(
    data: np.ndarray,
    params: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple:
    """Newton method solver."""
    # Placeholder for actual implementation
    result = params.copy()
    metrics = {'metric': metric_func(data, result)}
    warnings = []
    return result, metrics, warnings

def _coordinate_descent(
    data: np.ndarray,
    params: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple:
    """Coordinate descent solver."""
    # Placeholder for actual implementation
    result = params.copy()
    metrics = {'metric': metric_func(data, result)}
    warnings = []
    return result, metrics, warnings

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
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
