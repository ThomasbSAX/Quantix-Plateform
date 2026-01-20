"""
Quantix – Module kernel_methods
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# linear_kernel
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
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

def _normalize_data(X: np.ndarray, method: str = 'none') -> np.ndarray:
    """Normalize data based on specified method."""
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
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_linear_kernel(X: np.ndarray, y: Optional[np.ndarray] = None,
                          metric: str = 'euclidean',
                          custom_metric: Optional[Callable] = None) -> np.ndarray:
    """Compute the linear kernel matrix."""
    if custom_metric is not None:
        return custom_metric(X, y)
    if metric == 'euclidean':
        return np.dot(X, X.T)
    elif metric == 'cosine':
        norms = np.linalg.norm(X, axis=1)
        return np.dot(X, X.T) / (np.outer(norms, norms) + 1e-8)
    elif metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
    elif metric == 'minkowski':
        p = 3
        return np.sum(np.abs(X[:, np.newaxis] - X)**p, axis=2)**(1/p)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _solve_closed_form(K: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve the linear kernel problem using closed form solution."""
    return np.linalg.solve(K + 1e-8 * np.eye(K.shape[0]), y)

def _solve_gradient_descent(K: np.ndarray, y: np.ndarray,
                           learning_rate: float = 0.01,
                           max_iter: int = 1000,
                           tol: float = 1e-4) -> np.ndarray:
    """Solve the linear kernel problem using gradient descent."""
    w = np.zeros_like(y)
    for _ in range(max_iter):
        grad = 2 * (np.dot(K, w) - y)
        new_w = w - learning_rate * grad
        if np.linalg.norm(new_w - w) < tol:
            break
        w = new_w
    return w

def linear_kernel_fit(X: np.ndarray, y: Optional[np.ndarray] = None,
                     normalization: str = 'none',
                     metric: str = 'euclidean',
                     solver: str = 'closed_form',
                     custom_metric: Optional[Callable] = None,
                     **solver_kwargs) -> Dict:
    """Fit a linear kernel model.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values of shape (n_samples,) or None
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str
        Distance metric ('euclidean', 'cosine', 'manhattan', 'minkowski')
    solver : str
        Solver method ('closed_form', 'gradient_descent')
    custom_metric : Optional[Callable]
        Custom metric function
    **solver_kwargs :
        Additional solver-specific parameters

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used and warnings

    Examples
    --------
    >>> X = np.random.rand(10, 5)
    >>> y = np.random.rand(10)
    >>> result = linear_kernel_fit(X, y)
    """
    _validate_inputs(X, y)

    X_norm = _normalize_data(X, normalization)
    K = _compute_linear_kernel(X_norm, y, metric, custom_metric)

    if solver == 'closed_form':
        w = _solve_closed_form(K, y)
    elif solver == 'gradient_descent':
        w = _solve_gradient_descent(K, y, **solver_kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    if y is not None:
        predictions = np.dot(K, w)
        mse = np.mean((predictions - y)**2)
    else:
        predictions = None
        mse = None

    return {
        'result': {'weights': w, 'kernel_matrix': K},
        'metrics': {'mse': mse} if y is not None else {},
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

################################################################################
# polynomial_kernel
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if Y is not None and not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)) or (Y is not None and np.any(np.isnan(Y))):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or (Y is not None and np.any(np.isinf(Y))):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, method: str = 'none') -> np.ndarray:
    """Normalize input data."""
    if method == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X

def _polynomial_kernel_matrix(X: np.ndarray, Y: Optional[np.ndarray] = None,
                             degree: int = 3, gamma: float = 1.0) -> np.ndarray:
    """Compute the polynomial kernel matrix."""
    if Y is None:
        Y = X
    K = np.dot(X, Y.T)
    return (gamma * K + 1) ** degree

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     metric_functions: Dict[str, Callable]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}
    for name, func in metric_functions.items():
        try:
            metrics[name] = func(y_true, y_pred)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def polynomial_kernel_fit(X: np.ndarray, Y: Optional[np.ndarray] = None,
                         degree: int = 3, gamma: float = 1.0,
                         normalization: str = 'none',
                         metric_functions: Optional[Dict[str, Callable]] = None) -> Dict:
    """
    Compute the polynomial kernel matrix and evaluate metrics.

    Parameters:
    - X: Input data matrix of shape (n_samples, n_features)
    - Y: Optional target data matrix. If None, uses X for both inputs.
    - degree: Degree of the polynomial kernel
    - gamma: Kernel coefficient
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - metric_functions: Dictionary of metric functions to compute

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data
    X_norm = _normalize_data(X, normalization)
    if Y is not None:
        Y_norm = _normalize_data(Y, normalization)

    # Compute kernel matrix
    K = _polynomial_kernel_matrix(X_norm, Y_norm if Y is not None else None,
                                 degree=degree, gamma=gamma)

    # Compute metrics if provided
    metrics = {}
    if Y is not None and metric_functions:
        # For demonstration, we'll use a simple prediction (kernel diagonal)
        y_pred = np.diag(K) if Y.ndim == 1 else K
        metrics = _compute_metrics(Y, y_pred, metric_functions)

    # Prepare output
    result = {
        'kernel_matrix': K,
        'result': 'success',
        'metrics': metrics,
        'params_used': {
            'degree': degree,
            'gamma': gamma,
            'normalization': normalization
        },
        'warnings': []
    }

    return result

# Example metric functions
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Example usage:
"""
X = np.random.rand(10, 5)
Y = np.random.rand(10)

metrics = {
    'mse': mse,
    'r2': r2
}

result = polynomial_kernel_fit(X, Y, degree=2, gamma=0.5,
                             normalization='standard',
                             metric_functions=metrics)
"""

################################################################################
# gaussian_rbf_kernel
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if Y is not None and not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if Y is not None and np.any(np.isnan(Y)):
        raise ValueError("Y contains NaN values")

def _normalize_data(X: np.ndarray, method: str = 'none') -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X

def _compute_kernel_matrix(X: np.ndarray, Y: Optional[np.ndarray] = None,
                          gamma: float = 1.0) -> np.ndarray:
    """Compute the Gaussian RBF kernel matrix."""
    if Y is None:
        Y = X
    pairwise_sq_dists = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(Y**2, axis=1)[np.newaxis, :] - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * pairwise_sq_dists)

def gaussian_rbf_kernel_fit(X: np.ndarray, Y: Optional[np.ndarray] = None,
                           gamma: float = 1.0,
                           normalization_method: str = 'none',
                           distance_metric: Callable[[np.ndarray, np.ndarray], float] = None) -> Dict:
    """
    Compute the Gaussian RBF kernel matrix.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target data matrix of shape (n_samples, n_features). If None, Y = X.
    gamma : float
        Kernel coefficient for RBF kernel. Default is 1.0.
    normalization_method : str
        Normalization method for input data ('none', 'standard', 'minmax', 'robust'). Default is 'none'.
    distance_metric : Callable
        Custom distance metric function. If None, uses squared Euclidean distance.

    Returns:
    --------
    Dict
        Dictionary containing:
        - 'result': Computed kernel matrix
        - 'metrics': Dictionary of metrics (currently empty)
        - 'params_used': Dictionary of parameters used
        - 'warnings': List of warnings (if any)
    """
    _validate_inputs(X, Y)

    # Normalize data
    X_norm = _normalize_data(X, normalization_method)
    Y_norm = None if Y is None else _normalize_data(Y, normalization_method)

    # Compute kernel matrix
    K = _compute_kernel_matrix(X_norm, Y_norm, gamma)

    return {
        'result': K,
        'metrics': {},
        'params_used': {
            'gamma': gamma,
            'normalization_method': normalization_method
        },
        'warnings': []
    }

# Example usage:
"""
X = np.array([[1, 2], [3, 4], [5, 6]])
Y = np.array([[1, 2], [3, 4]])

result = gaussian_rbf_kernel_fit(X, Y, gamma=0.5, normalization_method='standard')
print(result['result'])
"""

################################################################################
# laplacian_kernel
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """
    Validate input arrays for the Laplacian kernel.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features).
    Y : Optional[np.ndarray]
        Target values if available. Default is None.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array.")
    if Y is not None and (not isinstance(Y, np.ndarray) or Y.ndim != 1):
        raise ValueError("Y must be a 1D numpy array if provided.")
    if np.any(np.isnan(X)) or (Y is not None and np.any(np.isnan(Y))):
        raise ValueError("Input arrays must not contain NaN values.")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")

def laplacian_kernel_compute(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    distance_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
    normalization: str = 'none',
    **kwargs
) -> Dict[str, Any]:
    """
    Compute the Laplacian kernel between samples.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features).
    Y : Optional[np.ndarray]
        Target values if available. Default is None.
    gamma : float, optional
        Bandwidth parameter for the Laplacian kernel. Default is 1.0.
    distance_metric : Callable, optional
        Custom distance metric function. Default is None (uses Euclidean).
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'. Default is 'none'.

    Returns
    ------
    Dict[str, Any]
        Dictionary containing the kernel matrix and other relevant information.
    """
    validate_inputs(X, Y)

    if distance_metric is None:
        distance_metric = lambda a, b: np.sqrt(np.sum((a - b) ** 2, axis=1))

    n_samples = X.shape[0]
    kernel_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            kernel_matrix[i, j] = np.exp(-gamma * distance_metric(X[i], X[j]))

    if normalization == 'standard':
        kernel_matrix = (kernel_matrix - np.mean(kernel_matrix)) / np.std(kernel_matrix)
    elif normalization == 'minmax':
        kernel_matrix = (kernel_matrix - np.min(kernel_matrix)) / (np.max(kernel_matrix) - np.min(kernel_matrix))
    elif normalization == 'robust':
        kernel_matrix = (kernel_matrix - np.median(kernel_matrix)) / (np.percentile(kernel_matrix, 75) - np.percentile(kernel_matrix, 25))

    result = {
        "kernel_matrix": kernel_matrix,
        "params_used": {
            "gamma": gamma,
            "distance_metric": distance_metric.__name__ if hasattr(distance_metric, '__name__') else "custom",
            "normalization": normalization
        },
        "warnings": []
    }

    return result

def laplacian_kernel_fit(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    distance_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
    normalization: str = 'none',
    solver: str = 'closed_form',
    **kwargs
) -> Dict[str, Any]:
    """
    Fit a model using the Laplacian kernel.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features).
    Y : Optional[np.ndarray]
        Target values if available. Default is None.
    gamma : float, optional
        Bandwidth parameter for the Laplacian kernel. Default is 1.0.
    distance_metric : Callable, optional
        Custom distance metric function. Default is None (uses Euclidean).
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'. Default is 'none'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', etc. Default is 'closed_form'.

    Returns
    ------
    Dict[str, Any]
        Dictionary containing the fitted model and other relevant information.
    """
    validate_inputs(X, Y)

    kernel_result = laplacian_kernel_compute(
        X, Y, gamma, distance_metric, normalization
    )

    if solver == 'closed_form':
        kernel_matrix = kernel_result["kernel_matrix"]
        if Y is not None:
            alpha = np.linalg.solve(kernel_matrix + 1e-6 * np.eye(len(Y)), Y)
            predictions = kernel_matrix @ alpha
        else:
            predictions = None

    result = {
        "kernel_result": kernel_result,
        "predictions": predictions,
        "params_used": {
            "gamma": gamma,
            "distance_metric": distance_metric.__name__ if hasattr(distance_metric, '__name__') else "custom",
            "normalization": normalization,
            "solver": solver
        },
        "warnings": []
    }

    return result

# Example usage:
if __name__ == "__main__":
    X = np.random.rand(10, 5)
    Y = np.random.rand(10)

    # Compute the Laplacian kernel
    kernel_result = laplacian_kernel_compute(X, Y, gamma=0.5)
    print(kernel_result)

    # Fit a model using the Laplacian kernel
    fit_result = laplacian_kernel_fit(X, Y, gamma=0.5)
    print(fit_result)

################################################################################
# sigmoid_kernel
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if Y is not None and not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if Y is not None and np.any(np.isnan(Y)):
        raise ValueError("Y contains NaN values")

def _normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize input data."""
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
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_sigmoid_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None,
                           alpha: float = 1.0) -> np.ndarray:
    """Compute the sigmoid kernel."""
    if Y is None:
        Y = X
    XY = np.dot(X, Y.T)
    return np.tanh(alpha * XY)

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute metrics for the kernel."""
    results = {}
    for name, func in metric_funcs.items():
        try:
            results[name] = func(y_true, y_pred)
        except Exception as e:
            results[name] = np.nan
    return results

def sigmoid_kernel_fit(X: np.ndarray, Y: Optional[np.ndarray] = None,
                      alpha: float = 1.0,
                      normalize_method: str = 'standard',
                      metric_funcs: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
    """
    Compute the sigmoid kernel between two sets of vectors.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target data matrix of shape (n_samples, n_features). If None, Y = X.
    alpha : float
        Parameter of the sigmoid kernel.
    normalize_method : str
        Normalization method for input data ('none', 'standard', 'minmax', 'robust').
    metric_funcs : Optional[Dict[str, Callable]]
        Dictionary of metric functions to compute. Keys are metric names,
        values are callables that take (y_true, y_pred) and return a float.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'result': Computed kernel matrix
        - 'metrics': Dictionary of computed metrics (if metric_funcs provided)
        - 'params_used': Parameters used in the computation
        - 'warnings': List of warnings encountered

    Example:
    --------
    >>> X = np.random.rand(10, 5)
    >>> result = sigmoid_kernel_fit(X)
    """
    warnings = []

    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data
    X_norm = _normalize_data(X, normalize_method)
    Y_norm = None if Y is None else _normalize_data(Y, normalize_method)

    # Compute kernel
    kernel_matrix = _compute_sigmoid_kernel(X_norm, Y_norm, alpha)

    # Compute metrics if provided
    metrics = {}
    if metric_funcs is not None and Y is not None:
        try:
            metrics = _compute_metrics(Y, kernel_matrix.diagonal(), metric_funcs)
        except Exception as e:
            warnings.append(f"Error computing metrics: {str(e)}")

    return {
        'result': kernel_matrix,
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'normalize_method': normalize_method
        },
        'warnings': warnings
    }

################################################################################
# cosine_similarity_kernel
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if Y is not None:
        if not isinstance(Y, np.ndarray):
            raise TypeError("Y must be a numpy array")
        if Y.ndim != 1:
            raise ValueError("Y must be a 1-dimensional array")
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same number of samples")
        if np.any(np.isnan(Y)):
            raise ValueError("Y contains NaN values")

def _normalize_data(X: np.ndarray, method: str = 'none') -> np.ndarray:
    """Normalize data according to specified method."""
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
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_cosine_similarity(X: np.ndarray) -> np.ndarray:
    """Compute cosine similarity kernel matrix."""
    X_normalized = _normalize_data(X, 'standard')
    return np.dot(X_normalized, X_normalized.T)

def cosine_similarity_kernel_fit(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    normalize: str = 'none',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute cosine similarity kernel and optionally fit a model.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target values if fitting a model
    normalize : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_metric : Optional[Callable]
        Custom metric function
    **kwargs :
        Additional parameters for model fitting

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data if specified
    X_normalized = _normalize_data(X, normalize)

    # Compute cosine similarity kernel
    K = np.dot(X_normalized, X_normalized.T)

    # Prepare output dictionary
    result = {
        'kernel_matrix': K,
        'metrics': {},
        'params_used': {
            'normalize': normalize
        },
        'warnings': []
    }

    # If Y is provided, compute metrics
    if Y is not None:
        if custom_metric is not None:
            result['metrics']['custom'] = custom_metric(K, Y)
        else:
            # Default metrics can be added here
            pass

    return result

# Example usage:
"""
X = np.random.rand(10, 5)
Y = np.random.rand(10)

result = cosine_similarity_kernel_fit(
    X=X,
    Y=Y,
    normalize='standard'
)
"""

################################################################################
# chi2_kernel
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if Y is not None and not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)) or (Y is not None and np.any(np.isnan(Y))):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or (Y is not None and np.any(np.isinf(Y))):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, normalization: str = 'none') -> np.ndarray:
    """Normalize data according to specified method."""
    if normalization == 'none':
        return X
    elif normalization == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif normalization == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif normalization == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_chi2_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None,
                         metric: str = 'euclidean',
                         custom_metric: Optional[Callable] = None) -> np.ndarray:
    """Compute the Chi-squared kernel between X and Y."""
    if custom_metric is not None:
        if Y is None:
            return custom_metric(X, X)
        else:
            return custom_metric(X, Y)

    if metric == 'euclidean':
        def chi2_kernel(a: np.ndarray, b: np.ndarray) -> float:
            return np.sum((a - b)**2 / (a + b + 1e-8))
    elif metric == 'manhattan':
        def chi2_kernel(a: np.ndarray, b: np.ndarray) -> float:
            return np.sum(np.abs(a - b) / (a + b + 1e-8))
    elif metric == 'cosine':
        def chi2_kernel(a: np.ndarray, b: np.ndarray) -> float:
            return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    n = X.shape[0]
    if Y is None:
        Y = X
        m = n
    else:
        m = Y.shape[0]

    kernel_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            kernel_matrix[i, j] = chi2_kernel(X[i], Y[j])

    return kernel_matrix

def _compute_metrics(kernel_matrix: np.ndarray, Y: Optional[np.ndarray] = None,
                     metric_funcs: Dict[str, Callable] = None) -> Dict[str, float]:
    """Compute evaluation metrics."""
    if metric_funcs is None:
        metric_funcs = {}

    metrics = {}
    for name, func in metric_funcs.items():
        try:
            if Y is not None:
                metrics[name] = func(kernel_matrix, Y)
            else:
                metrics[name] = func(kernel_matrix)
        except Exception as e:
            metrics[f"{name}_error"] = str(e)

    return metrics

def chi2_kernel_fit(X: np.ndarray, Y: Optional[np.ndarray] = None,
                    normalization: str = 'none',
                    metric: str = 'euclidean',
                    custom_metric: Optional[Callable] = None,
                    metric_funcs: Dict[str, Callable] = None) -> Dict[str, Any]:
    """
    Compute the Chi-squared kernel between X and Y.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    Y : np.ndarray, optional
        Target data matrix of shape (m_samples, n_features)
    normalization : str, default='none'
        Normalization method: 'none', 'standard', 'minmax', or 'robust'
    metric : str, default='euclidean'
        Distance metric: 'euclidean', 'manhattan', or 'cosine'
    custom_metric : callable, optional
        Custom metric function that takes two arrays and returns a distance
    metric_funcs : dict, optional
        Dictionary of metric functions to compute on the kernel matrix

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': The computed kernel matrix
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in the computation
        - 'warnings': Any warnings generated during computation

    Example:
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = chi2_kernel_fit(X)
    """
    _validate_inputs(X, Y)

    X_normalized = _normalize_data(X, normalization)
    if Y is not None:
        Y_normalized = _normalize_data(Y, normalization)

    kernel_matrix = _compute_chi2_kernel(X_normalized, Y_normalized if Y is not None else None,
                                        metric, custom_metric)

    metrics = _compute_metrics(kernel_matrix, Y, metric_funcs)

    return {
        'result': kernel_matrix,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric if custom_metric is None else 'custom',
        },
        'warnings': []
    }

################################################################################
# histogram_intersection_kernel
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """
    Validate input arrays for histogram intersection kernel.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Optional input array of shape (n_samples, n_features)

    Raises
    ------
    ValueError
        If inputs are invalid (NaN, inf, wrong dimensions)
    """
    if not isinstance(X, np.ndarray) or (Y is not None and not isinstance(Y, np.ndarray)):
        raise ValueError("Inputs must be numpy arrays")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if Y is not None and Y.ndim != 2:
        raise ValueError("Y must be a 2D array or None")

    if Y is not None and X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")

    if np.any(np.isnan(X)) or (Y is not None and np.any(np.isnan(Y))):
        raise ValueError("Input arrays must not contain NaN values")

    if np.any(np.isinf(X)) or (Y is not None and np.any(np.isinf(Y))):
        raise ValueError("Input arrays must not contain infinite values")

def normalize_data(X: np.ndarray, Y: Optional[np.ndarray] = None,
                  normalization: str = 'none') -> tuple:
    """
    Normalize input data according to specified method.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Optional input array of shape (n_samples, n_features)
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')

    Returns
    -------
    tuple
        Tuple of normalized X and Y (if provided)
    """
    if normalization == 'none':
        return X, Y

    # Standard normalization
    if normalization == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)

    # Min-Max normalization
    elif normalization == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / ((max_val - min_val + 1e-8))

    # Robust normalization
    elif normalization == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)

    if Y is not None:
        if normalization == 'standard':
            Y_norm = (Y - mean) / (std + 1e-8)
        elif normalization == 'minmax':
            Y_norm = (Y - min_val) / ((max_val - min_val + 1e-8))
        elif normalization == 'robust':
            Y_norm = (Y - median) / (iqr + 1e-8)
        return X_norm, Y_norm
    else:
        return X_norm, None

def compute_histogram_intersection(X: np.ndarray, Y: Optional[np.ndarray] = None,
                                 bins: int = 10) -> np.ndarray:
    """
    Compute histogram intersection kernel between X and Y.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Optional input array of shape (n_samples, n_features)
    bins : int
        Number of bins for histogram computation

    Returns
    -------
    np.ndarray
        Kernel matrix of shape (n_samples_X, n_samples_Y)
    """
    if Y is None:
        Y = X

    kernel_matrix = np.zeros((X.shape[0], Y.shape[0]))

    for i in range(X.shape[1]):
        # Compute histograms
        x_hist, _ = np.histogram(X[:, i], bins=bins, density=True)
        y_hist, _ = np.histogram(Y[:, i], bins=bins, density=True)

        # Compute intersection for each pair
        intersections = np.minimum(x_hist[:, np.newaxis], y_hist[np.newaxis, :])
        kernel_matrix += intersections

    return kernel_matrix / X.shape[1]

def histogram_intersection_kernel_fit(X: np.ndarray, Y: Optional[np.ndarray] = None,
                                    normalization: str = 'none',
                                    bins: int = 10) -> Dict[str, Any]:
    """
    Compute histogram intersection kernel with configurable options.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Optional input array of shape (n_samples, n_features)
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    bins : int
        Number of bins for histogram computation

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'result': computed kernel matrix
        - 'metrics': dictionary of metrics
        - 'params_used': parameters used in computation
        - 'warnings': list of warnings

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> result = histogram_intersection_kernel_fit(X)
    """
    # Validate inputs
    validate_inputs(X, Y)

    # Normalize data
    X_norm, Y_norm = normalize_data(X, Y, normalization)

    # Compute kernel
    kernel_matrix = compute_histogram_intersection(X_norm, Y_norm, bins)

    # Prepare output
    result = {
        'result': kernel_matrix,
        'metrics': {},
        'params_used': {
            'normalization': normalization,
            'bins': bins
        },
        'warnings': []
    }

    return result

# Example usage:
# X = np.random.rand(10, 5)
# result = histogram_intersection_kernel_fit(X)

################################################################################
# anova_rbf_kernel
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, y: np.ndarray,
                   normalization: str = 'standard',
                   custom_norm: Optional[Callable] = None) -> tuple:
    """Normalize input data."""
    if custom_norm is not None:
        X_norm = custom_norm(X)
        y_norm = custom_norm(y.reshape(-1, 1)).flatten()
    elif normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / (X_std + 1e-8)
        y_norm = (y - np.mean(y)) / (np.std(y) + 1e-8)
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        y_min = np.min(y)
        y_max = np.max(y)
        y_norm = (y - y_min) / (y_max - y_min + 1e-8)
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_q75 = np.percentile(X, 75, axis=0)
        X_q25 = np.percentile(X, 25, axis=0)
        X_norm = (X - X_median) / (X_q75 - X_q25 + 1e-8)
        y_median = np.median(y)
        y_q75 = np.percentile(y, 75)
        y_q25 = np.percentile(y, 25)
        y_norm = (y - y_median) / (y_q75 - y_q25 + 1e-8)
    else:
        X_norm = X.copy()
        y_norm = y.copy()

    return X_norm, y_norm

def _compute_rbf_kernel(X: np.ndarray,
                       gamma: float = 1.0) -> np.ndarray:
    """Compute RBF kernel matrix."""
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + \
               np.sum(X**2, axis=1).reshape(1, -1) - \
               2 * np.dot(X, X.T)
    return np.exp(-gamma * sq_dists)

def _anova_rbf_kernel_statistic(X: np.ndarray,
                              y: np.ndarray,
                              gamma: float = 1.0) -> float:
    """Compute ANOVA statistic for RBF kernel."""
    K = _compute_rbf_kernel(X, gamma)
    n = X.shape[0]
    y_mean = np.mean(y)

    # Center the kernel matrix
    H = np.eye(n) - np.ones((n, n)) / n
    K_centered = H @ K @ H

    # Compute the ANOVA statistic
    numerator = y.T @ K_centered @ y / (n - 1)
    denominator = np.trace(K_centered) / (n * (n - 1))

    return numerator / denominator

def _compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    metric: str = 'mse',
                    custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)
    else:
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred)**2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            metrics['r2'] = 1 - ss_res / (ss_tot + 1e-8)
        elif metric == 'logloss':
            y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
            metrics['logloss'] = -np.mean(y_true * np.log(y_pred) +
                                        (1 - y_true) * np.log(1 - y_pred))

    return metrics

def anova_rbf_kernel_fit(X: np.ndarray,
                        y: np.ndarray,
                        gamma: float = 1.0,
                        normalization: str = 'standard',
                        metric: str = 'mse',
                        custom_norm: Optional[Callable] = None,
                        custom_metric: Optional[Callable] = None) -> Dict:
    """
    Fit ANOVA RBF kernel method.

    Parameters
    ----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    gamma : float, optional
        RBF kernel parameter, by default 1.0
    normalization : str or Callable, optional
        Normalization method ('standard', 'minmax', 'robust'), by default 'standard'
    metric : str or Callable, optional
        Evaluation metric ('mse', 'mae', 'r2'), by default 'mse'
    custom_norm : Callable, optional
        Custom normalization function
    custom_metric : Callable, optional
        Custom metric function

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used and warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = anova_rbf_kernel_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y,
                                   normalization=normalization,
                                   custom_norm=custom_norm)

    # Compute ANOVA statistic
    stat = _anova_rbf_kernel_statistic(X_norm, y_norm, gamma)

    # Compute metrics (using predicted values as target for demonstration)
    metrics = _compute_metrics(y, y * 0 + np.mean(y),  # Dummy prediction
                             metric=metric,
                             custom_metric=custom_metric)

    return {
        'result': stat,
        'metrics': metrics,
        'params_used': {
            'gamma': gamma,
            'normalization': normalization if custom_norm is None else 'custom',
            'metric': metric if custom_metric is None else 'custom'
        },
        'warnings': []
    }

################################################################################
# multiquadric_kernel
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or (y is not None and np.any(np.isnan(y))):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or (y is not None and np.any(np.isinf(y))):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, y: Optional[np.ndarray] = None,
                   normalization: str = 'none') -> tuple:
    """Normalize input data."""
    if normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / (X_std + 1e-8)
        if y is not None:
            y_mean = np.mean(y)
            y_std = np.std(y)
            return X_normalized, (y - y_mean) / (y_std + 1e-8), {'X_mean': X_mean, 'X_std': X_std,
                                                                'y_mean': y_mean, 'y_std': y_std}
        return X_normalized, None, {'X_mean': X_mean, 'X_std': X_std}
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
        if y is not None:
            y_min = np.min(y)
            y_max = np.max(y)
            return X_normalized, (y - y_min) / (y_max - y_min + 1e-8), {'X_min': X_min, 'X_max': X_max,
                                                                    'y_min': y_min, 'y_max': y_max}
        return X_normalized, None, {'X_min': X_min, 'X_max': X_max}
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - X_median) / (X_iqr + 1e-8)
        if y is not None:
            y_median = np.median(y)
            y_iqr = np.subtract(*np.percentile(y, [75, 25]))
            return X_normalized, (y - y_median) / (y_iqr + 1e-8), {'X_median': X_median, 'X_iqr': X_iqr,
                                                                 'y_median': y_median, 'y_iqr': y_iqr}
        return X_normalized, None, {'X_median': X_median, 'X_iqr': X_iqr}
    return X.copy(), y.copy() if y is not None else None, {}

def _compute_multiquadric_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None,
                               c: float = 1.0) -> np.ndarray:
    """Compute the multiquadric kernel matrix."""
    if Y is None:
        Y = X
    distances = np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :])**2, axis=-1)
    return np.sqrt(distances + c**2)

def _closed_form_solver(K: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form solution for kernel ridge regression."""
    return np.linalg.solve(K + 1e-8 * np.eye(K.shape[0]), y)

def _gradient_descent_solver(K: np.ndarray, y: np.ndarray,
                           learning_rate: float = 0.01,
                           n_iter: int = 1000) -> np.ndarray:
    """Gradient descent solver for kernel ridge regression."""
    alpha = np.zeros(K.shape[0])
    for _ in range(n_iter):
        gradient = 2 * K @ alpha - 2 * y
        alpha -= learning_rate * gradient
    return alpha

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric_functions: Dict[str, Callable]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    return {name: func(y_true, y_pred) for name, func in metric_functions.items()}

def multiquadric_kernel_fit(X: np.ndarray, y: Optional[np.ndarray] = None,
                          c: float = 1.0,
                          normalization: str = 'none',
                          solver: str = 'closed_form',
                          learning_rate: float = 0.01,
                          n_iter: int = 1000,
                          metric_functions: Optional[Dict[str, Callable]] = None) -> Dict:
    """
    Fit a multiquadric kernel model.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values of shape (n_samples,) or None for unsupervised learning
    c : float, optional
        Shape parameter for the multiquadric kernel (default=1.0)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default='none')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent') (default='closed_form')
    learning_rate : float, optional
        Learning rate for gradient descent solver (default=0.01)
    n_iter : int, optional
        Number of iterations for gradient descent solver (default=1000)
    metric_functions : Optional[Dict[str, Callable]], optional
        Dictionary of metric functions to compute (default=None)

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': fitted model parameters or kernel matrix
        - 'metrics': computed metrics if y is provided and metric_functions are given
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> y = np.random.rand(10)
    >>> result = multiquadric_kernel_fit(X, y, c=2.0, normalization='standard')
    """
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm, norm_params = _normalize_data(X, y, normalization)

    # Compute kernel matrix
    K = _compute_multiquadric_kernel(X_norm, None if y is None else X_norm, c)

    result = {}
    metrics = {}
    warnings = []

    if y is not None:
        # Solve for coefficients
        if solver == 'closed_form':
            alpha = _closed_form_solver(K, y_norm)
        elif solver == 'gradient_descent':
            alpha = _gradient_descent_solver(K, y_norm, learning_rate, n_iter)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        result['alpha'] = alpha
        y_pred = K @ alpha

        # Compute metrics if requested
        if metric_functions is not None:
            metrics = _compute_metrics(y_norm, y_pred, metric_functions)

    result['kernel_matrix'] = K
    params_used = {
        'c': c,
        'normalization': normalization,
        'solver': solver,
        'learning_rate': learning_rate,
        'n_iter': n_iter
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# inverse_multiquadric_kernel
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def inverse_multiquadric_kernel_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    bandwidth: float = 1.0,
    normalization: str = 'none',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    reg_param: float = 1e-6,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    verbose: bool = False
) -> Dict:
    """
    Fit an inverse multiquadric kernel model.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values. If None, only kernel matrix is computed.
    bandwidth : float
        Bandwidth parameter for the kernel
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    distance_metric : Union[str, Callable]
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski')
    solver : str
        Solver to use ('closed_form', 'gradient_descent', 'newton')
    regularization : Optional[str]
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    reg_param : float
        Regularization parameter
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    custom_metric : Optional[Callable]
        Custom metric function if needed
    verbose : bool
        Whether to print progress information

    Returns:
    --------
    Dict containing:
        - 'result': fitted model or kernel matrix
        - 'metrics': computed metrics
        - 'params_used': parameters used in fitting
        - 'warnings': any warnings generated

    Example:
    --------
    >>> X = np.random.rand(10, 2)
    >>> y = np.random.rand(10)
    >>> result = inverse_multiquadric_kernel_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if needed
    X_norm = _apply_normalization(X, normalization)

    # Compute kernel matrix
    K = _compute_kernel_matrix(X_norm, bandwidth=bandwidth,
                              distance_metric=distance_metric)

    # Prepare output
    result = {
        'result': K,
        'metrics': {},
        'params_used': {
            'bandwidth': bandwidth,
            'normalization': normalization,
            'distance_metric': distance_metric,
            'solver': solver,
            'regularization': regularization,
            'reg_param': reg_param
        },
        'warnings': []
    }

    if y is not None:
        # Fit model with specified solver
        if solver == 'closed_form':
            result['result'] = _fit_closed_form(K, y, regularization, reg_param)
        elif solver == 'gradient_descent':
            result['result'] = _fit_gradient_descent(K, y, regularization,
                                                   reg_param, tol, max_iter)
        elif solver == 'newton':
            result['result'] = _fit_newton(K, y, regularization,
                                         reg_param, tol, max_iter)

        # Compute metrics
        result['metrics'] = _compute_metrics(result['result'], y, K,
                                           custom_metric=custom_metric)

    return result

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    """Validate input data."""
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
    """Apply specified normalization to data."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) -
                                           np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_kernel_matrix(X: np.ndarray, bandwidth: float,
                          distance_metric: Union[str, Callable]) -> np.ndarray:
    """Compute inverse multiquadric kernel matrix."""
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            if isinstance(distance_metric, str):
                dist = _compute_distance(X[i], X[j], distance_metric)
            else:
                dist = distance_metric(X[i], X[j])

            K[i, j] = 1 / np.sqrt(dist**2 + bandwidth**2)

    return K

def _compute_distance(x1: np.ndarray, x2: np.ndarray,
                     metric: str) -> float:
    """Compute specified distance between two vectors."""
    if metric == 'euclidean':
        return np.linalg.norm(x1 - x2)
    elif metric == 'manhattan':
        return np.sum(np.abs(x1 - x2))
    elif metric == 'cosine':
        return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    elif metric == 'minkowski':
        return np.sum(np.abs(x1 - x2)**3)**(1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _fit_closed_form(K: np.ndarray, y: np.ndarray,
                    regularization: Optional[str], reg_param: float):
    """Fit model using closed form solution."""
    if regularization == 'l2':
        K_reg = K + reg_param * np.eye(K.shape[0])
    elif regularization == 'l1':
        # For L1, we would typically use coordinate descent
        raise NotImplementedError("L1 regularization not implemented for closed form")
    else:
        K_reg = K

    return np.linalg.solve(K_reg, y)

def _fit_gradient_descent(K: np.ndarray, y: np.ndarray,
                         regularization: Optional[str], reg_param: float,
                         tol: float, max_iter: int):
    """Fit model using gradient descent."""
    n_samples = K.shape[0]
    beta = np.zeros(n_samples)

    for _ in range(max_iter):
        gradient = 2 * K @ (K @ beta - y)

        if regularization == 'l2':
            gradient += 2 * reg_param * beta
        elif regularization == 'l1':
            gradient += reg_param * np.sign(beta)

        beta_new = beta - 0.01 * gradient

        if np.linalg.norm(beta_new - beta) < tol:
            break

        beta = beta_new

    return beta

def _fit_newton(K: np.ndarray, y: np.ndarray,
               regularization: Optional[str], reg_param: float,
               tol: float, max_iter: int):
    """Fit model using Newton's method."""
    n_samples = K.shape[0]
    beta = np.zeros(n_samples)

    for _ in range(max_iter):
        residual = K @ beta - y
        gradient = 2 * K.T @ residual

        if regularization == 'l2':
            hessian = 2 * (K.T @ K + reg_param * np.eye(n_samples))
        else:
            hessian = 2 * (K.T @ K)

        beta_new = beta - np.linalg.solve(hessian, gradient)

        if np.linalg.norm(beta_new - beta) < tol:
            break

        beta = beta_new

    return beta

def _compute_metrics(result: np.ndarray, y_true: np.ndarray,
                    K: np.ndarray, custom_metric: Optional[Callable]) -> Dict:
    """Compute various metrics for the fitted model."""
    y_pred = K @ result
    metrics = {}

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)

    # Add other standard metrics as needed
    return metrics

################################################################################
# cauchy_kernel
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if Y is not None and not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)) or (Y is not None and np.any(np.isnan(Y))):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or (Y is not None and np.any(np.isinf(Y))):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, normalization: str = 'none') -> np.ndarray:
    """Normalize the input data."""
    if normalization == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalization == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalization == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X

def _compute_cauchy_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None,
                          sigma: float = 1.0) -> np.ndarray:
    """Compute the Cauchy kernel matrix."""
    if Y is None:
        Y = X
    pairwise_sq_dists = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(Y**2, axis=1) - 2 * X.dot(Y.T)
    return 1 / (1 + pairwise_sq_dists / sigma**2)

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(y_true, y_pred)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def cauchy_kernel_fit(X: np.ndarray, Y: Optional[np.ndarray] = None,
                     normalization: str = 'none',
                     sigma: float = 1.0,
                     metric_funcs: Optional[Dict[str, Callable]] = None) -> Dict:
    """
    Fit the Cauchy kernel and compute results.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target data matrix of shape (n_samples, n_features) or None
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    sigma : float
        Bandwidth parameter for the Cauchy kernel
    metric_funcs : Optional[Dict[str, Callable]]
        Dictionary of metric functions to compute

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': computed kernel matrix
        - 'metrics': computed metrics (if metric_funcs provided)
        - 'params_used': parameters used
        - 'warnings': any warnings generated

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> Y = np.random.rand(10, 3)
    >>> result = cauchy_kernel_fit(X, Y, normalization='standard', sigma=0.5)
    """
    warnings = []

    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data
    X_norm = _normalize_data(X, normalization)
    if Y is not None:
        Y_norm = _normalize_data(Y, normalization)

    # Compute kernel
    kernel_matrix = _compute_cauchy_kernel(X_norm, Y_norm if Y is not None else None, sigma)

    # Compute metrics if provided
    metrics = {}
    if Y is not None and metric_funcs is not None:
        try:
            metrics = _compute_metrics(Y, kernel_matrix.diagonal(), metric_funcs)
        except Exception as e:
            warnings.append(f"Error computing metrics: {str(e)}")

    return {
        'result': kernel_matrix,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'sigma': sigma
        },
        'warnings': warnings
    }

################################################################################
# exponential_kernel
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def exponential_kernel_fit(
    X: np.ndarray,
    Y: np.ndarray,
    bandwidth: float = 1.0,
    normalize_X: bool = True,
    distance_metric: Union[str, Callable] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    reg_param: float = 1e-6,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Compute the exponential kernel fit for given data.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    Y : np.ndarray
        Target values of shape (n_samples,)
    bandwidth : float, optional
        Bandwidth parameter for the exponential kernel (default=1.0)
    normalize_X : bool, optional
        Whether to normalize input features (default=True)
    distance_metric : str or callable, optional
        Distance metric to use ("euclidean", "manhattan", etc.) or custom function (default="euclidean")
    solver : str, optional
        Solver to use ("closed_form", "gradient_descent") (default="closed_form")
    regularization : str or None, optional
        Regularization type ("l1", "l2") (default=None)
    reg_param : float, optional
        Regularization parameter (default=1e-6)
    tol : float, optional
        Tolerance for convergence (default=1e-4)
    max_iter : int, optional
        Maximum number of iterations (default=1000)
    custom_metric : callable or None, optional
        Custom metric function (default=None)

    Returns:
    --------
    Dict containing:
        - "result": fitted model parameters
        - "metrics": computed metrics
        - "params_used": dictionary of used parameters
        - "warnings": list of warnings

    Example:
    --------
    >>> X = np.random.rand(10, 2)
    >>> Y = np.random.rand(10)
    >>> result = exponential_kernel_fit(X, Y)
    """
    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize features if requested
    X_normalized = _normalize_features(X) if normalize_X else X

    # Prepare parameters dictionary
    params_used = {
        "bandwidth": bandwidth,
        "normalize_X": normalize_X,
        "distance_metric": distance_metric,
        "solver": solver,
        "regularization": regularization,
        "reg_param": reg_param
    }

    # Compute kernel matrix
    K = _compute_kernel_matrix(X_normalized, bandwidth, distance_metric)

    # Solve for parameters based on chosen solver
    if solver == "closed_form":
        params = _solve_closed_form(K, Y, regularization, reg_param)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(K, Y, regularization, reg_param,
                                       tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, Y, params, K,
                             custom_metric=custom_metric)

    # Prepare warnings
    warnings = _check_warnings(K, params)

    return {
        "result": params,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

def _validate_inputs(X: np.ndarray, Y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Y.ndim != 1:
        raise ValueError("Y must be a 1D array")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
        raise ValueError("Y contains NaN or infinite values")

def _normalize_features(X: np.ndarray) -> np.ndarray:
    """Normalize features using standard scaling."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0  # Avoid division by zero
    return (X - mean) / std

def _compute_kernel_matrix(
    X: np.ndarray,
    bandwidth: float,
    distance_metric: Union[str, Callable]
) -> np.ndarray:
    """Compute the exponential kernel matrix."""
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    if isinstance(distance_metric, str):
        if distance_metric == "euclidean":
            dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        elif distance_metric == "manhattan":
            dist = np.sum(np.abs(X[:, None, :] - X[None, :, :]), axis=-1)
        elif distance_metric == "cosine":
            dist = 1 - np.dot(X, X.T) / (np.linalg.norm(X, axis=1)[:, None] *
                                       np.linalg.norm(X, axis=1)[None, :])
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
    else:
        dist = distance_metric(X)

    K = np.exp(-dist / (2 * bandwidth**2))
    return K

def _solve_closed_form(
    K: np.ndarray,
    Y: np.ndarray,
    regularization: Optional[str],
    reg_param: float
) -> np.ndarray:
    """Solve for parameters using closed form solution."""
    if regularization == "l2":
        K_reg = K + reg_param * np.eye(K.shape[0])
    elif regularization == "l1":
        raise NotImplementedError("L1 regularization not implemented for closed form")
    else:
        K_reg = K

    return np.linalg.solve(K_reg, Y)

def _solve_gradient_descent(
    K: np.ndarray,
    Y: np.ndarray,
    regularization: Optional[str],
    reg_param: float,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve for parameters using gradient descent."""
    n_samples = K.shape[0]
    params = np.zeros(n_samples)

    for _ in range(max_iter):
        grad = -2 * K @ (Y - K @ params) / n_samples

        if regularization == "l2":
            grad += 2 * reg_param * params
        elif regularization == "l1":
            grad += reg_param * np.sign(params)

        params_new = params - tol * grad

        if np.linalg.norm(params_new - params) < tol:
            break

        params = params_new

    return params

def _compute_metrics(
    X: np.ndarray,
    Y: np.ndarray,
    params: np.ndarray,
    K: np.ndarray,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute various metrics for the model."""
    Y_pred = K @ params

    metrics = {
        "mse": np.mean((Y - Y_pred) ** 2),
        "mae": np.mean(np.abs(Y - Y_pred)),
        "r2": 1 - np.sum((Y - Y_pred) ** 2) / np.sum((Y - np.mean(Y)) ** 2)
    }

    if custom_metric is not None:
        metrics["custom"] = custom_metric(Y, Y_pred)

    return metrics

def _check_warnings(
    K: np.ndarray,
    params: np.ndarray
) -> list:
    """Check for potential warnings in the model."""
    warnings = []

    if np.any(np.isnan(params)):
        warnings.append("Parameters contain NaN values")
    if np.any(np.isinf(K)):
        warnings.append("Kernel matrix contains infinite values")

    return warnings

################################################################################
# spline_kernel
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def spline_kernel_fit(
    X: np.ndarray,
    y: np.ndarray,
    kernel_type: str = 'linear',
    degree: int = 3,
    gamma: float = 1.0,
    coef0: float = 1.0,
    normalize: bool = True,
    metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Fit a spline kernel model to the data.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, n_outputs).
    kernel_type : str
        Type of spline kernel ('linear', 'poly', 'rbf', 'sigmoid').
    degree : int
        Degree of the polynomial kernel.
    gamma : float
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    coef0 : float
        Independent term in kernel function.
    normalize : bool
        Whether to normalize the data.
    metric : str or callable
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable.
    solver : str
        Solver type ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : str or None
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    alpha : float
        Regularization strength.
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    custom_metric : callable or None
        Custom metric function if needed.
    custom_distance : callable or None
        Custom distance function if needed.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_normalized = _normalize_data(X) if normalize else X

    # Compute kernel matrix
    K = _compute_kernel_matrix(X_normalized, kernel_type, degree, gamma, coef0)

    # Solve for parameters
    params = _solve_kernel_equation(K, y, solver, regularization, alpha, tol, max_iter)

    # Compute metrics
    metrics = _compute_metrics(y, params['y_pred'], metric, custom_metric)

    # Prepare output
    result = {
        'result': params['y_pred'],
        'metrics': metrics,
        'params_used': {
            'kernel_type': kernel_type,
            'degree': degree,
            'gamma': gamma,
            'coef0': coef0,
            'normalize': normalize,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'alpha': alpha
        },
        'warnings': params.get('warnings', [])
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1 and y.ndim != 2:
        raise ValueError("y must be a 1D or 2D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values.")

def _normalize_data(X: np.ndarray) -> np.ndarray:
    """Normalize the input data."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + 1e-8)

def _compute_kernel_matrix(
    X: np.ndarray,
    kernel_type: str,
    degree: int,
    gamma: float,
    coef0: float
) -> np.ndarray:
    """Compute the kernel matrix."""
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    if kernel_type == 'linear':
        K = X @ X.T
    elif kernel_type == 'poly':
        K = (gamma * X @ X.T + coef0) ** degree
    elif kernel_type == 'rbf':
        pairwise_sq_dists = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(X**2, axis=1) - 2 * X @ X.T
        K = np.exp(-gamma * pairwise_sq_dists)
    elif kernel_type == 'sigmoid':
        K = np.tanh(gamma * X @ X.T + coef0)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return K

def _solve_kernel_equation(
    K: np.ndarray,
    y: np.ndarray,
    solver: str,
    regularization: Optional[str],
    alpha: float,
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve the kernel equation for parameters."""
    n_samples = K.shape[0]
    params = {}

    if solver == 'closed_form':
        if regularization is None:
            K_inv = np.linalg.pinv(K)
            params['coef'] = K_inv @ y
        else:
            if regularization == 'l2':
                K_reg = K + alpha * np.eye(n_samples)
            elif regularization == 'l1':
                # Use coordinate descent for L1
                params['coef'], _ = _coordinate_descent(K, y, alpha, tol, max_iter)
            elif regularization == 'elasticnet':
                params['coef'], _ = _coordinate_descent(K, y, alpha, tol, max_iter, l1_ratio=0.5)
            else:
                raise ValueError(f"Unknown regularization type: {regularization}")
            params['coef'] = np.linalg.solve(K_reg, y)
    else:
        raise ValueError(f"Unknown solver type: {solver}")

    params['y_pred'] = K @ params['coef']
    return params

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Compute metrics for the model."""
    metrics = {}

    if metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    elif callable(metric):
        metrics['custom'] = metric(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric type: {metric}")

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)

    return metrics

def _coordinate_descent(
    K: np.ndarray,
    y: np.ndarray,
    alpha: float,
    tol: float,
    max_iter: int,
    l1_ratio: float = 1.0
) -> tuple:
    """Coordinate descent for L1 or elasticnet regularization."""
    n_samples = K.shape[0]
    coef = np.zeros(n_samples)
    for _ in range(max_iter):
        old_coef = coef.copy()
        for i in range(n_samples):
            residuals = y - K @ coef + K[:, i] * coef[i]
            r = K[:, i].T @ residuals
            if l1_ratio == 1.0:
                coef[i] = _soft_threshold(r, alpha)
            else:
                coef[i] = _elasticnet_threshold(r, alpha * l1_ratio, alpha * (1 - l1_ratio))
        if np.linalg.norm(coef - old_coef) < tol:
            break
    return coef, None

def _soft_threshold(r: float, alpha: float) -> float:
    """Soft thresholding operator for L1 regularization."""
    if r > alpha:
        return r - alpha
    elif r < -alpha:
        return r + alpha
    else:
        return 0.0

def _elasticnet_threshold(r: float, alpha1: float, alpha2: float) -> float:
    """Thresholding operator for elasticnet regularization."""
    if r > (alpha1 + alpha2):
        return r - alpha1
    elif r < -(alpha1 + alpha2):
        return r + alpha1
    else:
        return 0.0

# Example usage:
"""
X = np.random.rand(10, 2)
y = np.random.rand(10)

result = spline_kernel_fit(
    X, y,
    kernel_type='rbf',
    gamma=0.1,
    normalize=True,
    metric='mse',
    solver='closed_form'
)

print(result)
"""

################################################################################
# bessel_kernel
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if Y is not None and not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)) or (Y is not None and np.any(np.isnan(Y))):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or (Y is not None and np.any(np.isinf(Y))):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, Y: Optional[np.ndarray] = None,
                    normalization: str = 'none') -> tuple:
    """Normalize input data."""
    if normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / (X_std + 1e-8)
        if Y is not None:
            return X_norm, (Y - np.mean(Y)) / (np.std(Y) + 1e-8)
        return X_norm, None
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        if Y is not None:
            return X_norm, (Y - np.min(Y)) / (np.max(Y) - np.min(Y) + 1e-8)
        return X_norm, None
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_q75, X_q25 = np.percentile(X, [75, 25], axis=0)
        X_iqr = X_q75 - X_q25
        X_norm = (X - X_median) / (X_iqr + 1e-8)
        if Y is not None:
            y_median = np.median(Y)
            y_q75, y_q25 = np.percentile(Y, [75, 25])
            y_iqr = y_q75 - y_q25
            return X_norm, (Y - y_median) / (y_iqr + 1e-8)
        return X_norm, None
    else:
        return X, Y

def _bessel_kernel_matrix(X: np.ndarray, nu: float = 1.0) -> np.ndarray:
    """Compute the Bessel kernel matrix."""
    pairwise_dists = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :])**2, axis=-1))
    return np.sqrt(2 * nu / (np.pi * pairwise_dists)) * np.sin(pairwise_dists)

def _compute_metrics(Y_true: np.ndarray, Y_pred: np.ndarray,
                     metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(Y_true, Y_pred)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def bessel_kernel_fit(X: np.ndarray, Y: Optional[np.ndarray] = None,
                      nu: float = 1.0,
                      normalization: str = 'none',
                      metric_funcs: Optional[Dict[str, Callable]] = None,
                      **kwargs) -> Dict[str, Any]:
    """
    Fit a Bessel kernel model.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target values if available
    nu : float
        Parameter for the Bessel kernel (default=1.0)
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric_funcs : Optional[Dict[str, Callable]]
        Dictionary of metric functions to compute
    **kwargs :
        Additional parameters for the kernel

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(10, 5)
    >>> Y = np.random.rand(10)
    >>> result = bessel_kernel_fit(X, Y, nu=2.0, normalization='standard')
    """
    _validate_inputs(X, Y)

    # Normalize data
    X_norm, Y_norm = _normalize_data(X, Y, normalization)

    # Compute kernel matrix
    K = _bessel_kernel_matrix(X_norm, nu=nu)

    # Prepare results dictionary
    result = {
        'result': K,
        'params_used': {
            'nu': nu,
            'normalization': normalization
        },
        'warnings': []
    }

    # Compute metrics if Y is provided and metric functions are given
    if Y is not None:
        if metric_funcs is None:
            result['warnings'].append("No metrics provided")
        else:
            Y_pred = np.mean(K, axis=0)  # Simple prediction example
            metrics = _compute_metrics(Y_norm if Y is not None else Y,
                                      Y_pred, metric_funcs)
            result['metrics'] = metrics

    return result

################################################################################
# dot_product_kernel
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Union

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or (y is not None and np.any(np.isnan(y))):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or (y is not None and np.any(np.isinf(y))):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, normalization: str = 'none') -> np.ndarray:
    """Normalize the input data."""
    if normalization == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalization == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalization == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X

def _compute_kernel_matrix(X: np.ndarray, kernel_func: Callable) -> np.ndarray:
    """Compute the kernel matrix."""
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_func(X[i], X[j])
    return K

def _solve_closed_form(K: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve the kernel problem using closed form solution."""
    return np.linalg.solve(K + 1e-6 * np.eye(K.shape[0]), y)

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute the metrics."""
    metrics = {}
    for name, func in metric_funcs.items():
        metrics[name] = func(y_true, y_pred)
    return metrics

def dot_product_kernel_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalization: str = 'none',
    kernel_func: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.dot(x, y),
    solver: str = 'closed_form',
    metric_funcs: Dict[str, Callable] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Fit a dot product kernel model.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    y : Optional[np.ndarray]
        Target values of shape (n_samples,) or None.
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust').
    kernel_func : Callable
        Kernel function that takes two vectors and returns a scalar.
    solver : str
        Solver method ('closed_form').
    metric_funcs : Dict[str, Callable]
        Dictionary of metric functions to compute.
    **kwargs
        Additional keyword arguments for the solver.

    Returns:
    --------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(X, y)
    X_normalized = _normalize_data(X, normalization)
    K = _compute_kernel_matrix(X_normalized, kernel_func)

    if solver == 'closed_form':
        params = _solve_closed_form(K, y) if y is not None else np.zeros(X.shape[0])
    else:
        raise ValueError(f"Solver {solver} not implemented")

    result = {'params': params}
    metrics = {}
    if y is not None and metric_funcs is not None:
        y_pred = np.dot(K, params)
        metrics = _compute_metrics(y, y_pred, metric_funcs)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'kernel_func': kernel_func.__name__ if hasattr(kernel_func, '__name__') else 'custom',
            'solver': solver
        },
        'warnings': []
    }

# Example usage:
"""
X = np.random.rand(10, 5)
y = np.random.rand(10)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

metric_funcs = {'mse': mse}

result = dot_product_kernel_fit(
    X=X,
    y=y,
    normalization='standard',
    kernel_func=lambda x, y: np.dot(x, y),
    solver='closed_form',
    metric_funcs=metric_funcs
)
"""

################################################################################
# tanimoto_kernel
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """
    Validate input arrays for Tanimoto kernel computation.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Optional input array of shape (n_samples, n_features) or None

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if not isinstance(X, np.ndarray) or (Y is not None and not isinstance(Y, np.ndarray)):
        raise ValueError("Inputs must be numpy arrays")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if Y is not None and Y.ndim != 2:
        raise ValueError("Y must be a 2D array or None")

    if Y is not None and X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")

    if np.any(np.isnan(X)) or (Y is not None and np.any(np.isnan(Y))):
        raise ValueError("Input arrays must not contain NaN values")

    if np.any(np.isinf(X)) or (Y is not None and np.any(np.isinf(Y))):
        raise ValueError("Input arrays must not contain infinite values")

def tanimoto_kernel_compute(X: np.ndarray,
                           Y: Optional[np.ndarray] = None,
                           normalize: str = 'none',
                           distance_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
                           custom_kernel: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None) -> Dict[str, Any]:
    """
    Compute Tanimoto kernel between input arrays.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Optional input array of shape (n_samples, n_features) or None
    normalize : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    distance_metric : Callable
        Custom distance metric function
    custom_kernel : Optional[Callable]
        Custom kernel function

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'kernel_matrix': Computed kernel matrix
        - 'params_used': Parameters used in computation
        - 'warnings': Any warnings generated

    Examples
    --------
    >>> X = np.array([[1, 0, 1], [0, 1, 0]])
    >>> result = tanimoto_kernel_compute(X)
    """
    warnings: list = []

    # Validate inputs
    validate_inputs(X, Y)

    if Y is None:
        Y = X

    # Apply normalization
    normalized_X, normalized_Y = apply_normalization(X, Y, normalize)
    warnings.extend(normalized_X.get('warnings', []))
    warnings.extend(normalized_Y.get('warnings', []))
    normalized_X = normalized_X['normalized']
    normalized_Y = normalized_Y['normalized']

    # Compute kernel matrix
    if custom_kernel is not None:
        kernel_matrix = custom_kernel(normalized_X, normalized_Y)
    else:
        if distance_metric is None:
            # Default to Euclidean distance
            def euclidean_distance(a, b):
                return np.sqrt(np.sum((a - b)**2, axis=1))
            distance_metric = euclidean_distance

        kernel_matrix = compute_tanimoto_kernel(normalized_X, normalized_Y, distance_metric)

    return {
        'kernel_matrix': kernel_matrix,
        'params_used': {
            'normalize': normalize,
            'distance_metric': distance_metric.__name__ if callable(distance_metric) else str(distance_metric),
            'custom_kernel': custom_kernel.__name__ if callable(custom_kernel) else str(custom_kernel)
        },
        'warnings': warnings
    }

def apply_normalization(X: np.ndarray,
                       Y: Optional[np.ndarray] = None,
                       method: str = 'none') -> Dict[str, Any]:
    """
    Apply normalization to input arrays.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Optional input array of shape (n_samples, n_features) or None
    method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'normalized': Normalized array(s)
        - 'warnings': Any warnings generated
    """
    if Y is None:
        Y = X

    normalized_X = X.copy()
    normalized_Y = Y.copy()

    if method == 'standard':
        mean_X = np.mean(normalized_X, axis=0)
        std_X = np.std(normalized_X, axis=0)
        normalized_X = (normalized_X - mean_X) / std_X

        if Y is not X:
            mean_Y = np.mean(normalized_Y, axis=0)
            std_Y = np.std(normalized_Y, axis=0)
            normalized_Y = (normalized_Y - mean_Y) / std_Y

    elif method == 'minmax':
        min_X = np.min(normalized_X, axis=0)
        max_X = np.max(normalized_X, axis=0)
        normalized_X = (normalized_X - min_X) / (max_X - min_X + 1e-8)

        if Y is not X:
            min_Y = np.min(normalized_Y, axis=0)
            max_Y = np.max(normalized_Y, axis=0)
            normalized_Y = (normalized_Y - min_Y) / (max_Y - min_Y + 1e-8)

    elif method == 'robust':
        median_X = np.median(normalized_X, axis=0)
        iqr_X = np.subtract(*np.percentile(normalized_X, [75, 25], axis=0))
        normalized_X = (normalized_X - median_X) / (iqr_X + 1e-8)

        if Y is not X:
            median_Y = np.median(normalized_Y, axis=0)
            iqr_Y = np.subtract(*np.percentile(normalized_Y, [75, 25], axis=0))
            normalized_Y = (normalized_Y - median_Y) / (iqr_Y + 1e-8)

    return {
        'normalized': normalized_X if Y is X else (normalized_X, normalized_Y),
        'warnings': []
    }

def compute_tanimoto_kernel(X: np.ndarray,
                           Y: Optional[np.ndarray] = None,
                           distance_metric: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None) -> np.ndarray:
    """
    Compute Tanimoto kernel matrix.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Optional input array of shape (n_samples, n_features) or None
    distance_metric : Callable
        Distance metric function

    Returns
    -------
    np.ndarray
        Computed kernel matrix of shape (n_samples_X, n_samples_Y)
    """
    if Y is None:
        Y = X

    if distance_metric is None:
        distance_metric = lambda a, b: np.sqrt(np.sum((a - b) ** 2, axis=-1))

    # Compute pairwise distances
    dist_XY = distance_metric(X[:, np.newaxis, :], Y[np.newaxis, :, :])

    # Compute Tanimoto kernel
    norm_X = np.sum(X**2, axis=1)[:, np.newaxis]
    norm_Y = np.sum(Y**2, axis=1)[np.newaxis, :]

    kernel_matrix = 1 - (dist_XY**2) / (norm_X + norm_Y - dist_XY**2 + 1e-8)

    return kernel_matrix
