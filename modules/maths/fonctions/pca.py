"""
Quantix – Module pca
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# dimension_reduction
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

def normalize_data(
    X: np.ndarray,
    method: str = "standard",
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data according to specified method."""
    if custom_normalizer is not None:
        return custom_normalizer(X)

    X_norm = X.copy()
    if method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
    return X_norm

def compute_covariance_matrix(X: np.ndarray) -> np.ndarray:
    """Compute covariance matrix."""
    return np.cov(X, rowvar=False)

def compute_eigendecomposition(
    cov_matrix: np.ndarray,
    n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigendecomposition of covariance matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][:n_components]
    eigenvectors = eigenvectors[:, idx][:, :n_components]
    return eigenvalues, eigenvectors

def transform_data(
    X: np.ndarray,
    components: np.ndarray
) -> np.ndarray:
    """Transform data to lower dimensional space."""
    return X @ components

def compute_metrics(
    X: np.ndarray,
    X_transformed: np.ndarray,
    metric_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute various metrics."""
    return {name: func(X, X_transformed) for name, func in metric_funcs.items()}

def dimension_reduction_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalization_method: str = "standard",
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Perform dimensionality reduction using PCA.

    Parameters:
        X: Input data matrix of shape (n_samples, n_features)
        n_components: Number of components to keep
        normalization_method: Normalization method ('standard', 'minmax', 'robust')
        custom_normalizer: Custom normalization function
        metric_funcs: Dictionary of metric functions to compute

    Returns:
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate input
    validate_input(X)

    # Normalize data
    X_norm = normalize_data(
        X,
        method=normalization_method,
        custom_normalizer=custom_normalizer
    )

    # Compute covariance matrix
    cov_matrix = compute_covariance_matrix(X_norm)

    # Eigendecomposition
    eigenvalues, eigenvectors = compute_eigendecomposition(
        cov_matrix,
        n_components
    )

    # Transform data
    X_transformed = transform_data(X_norm, eigenvectors)

    # Compute metrics if provided
    metrics = {}
    if metric_funcs is not None:
        metrics = compute_metrics(X_norm, X_transformed, metric_funcs)

    # Prepare output
    result = {
        "components": eigenvectors,
        "explained_variance": eigenvalues,
        "transformed_data": X_transformed
    }

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalization_method": normalization_method,
            "n_components": n_components
        },
        "warnings": []
    }

# Example usage:
"""
X = np.random.rand(100, 5)
result = dimension_reduction_fit(
    X,
    n_components=2,
    normalization_method="standard",
    metric_funcs={
        "mse": lambda X, X_t: np.mean((X - X_t @ X_t.T) ** 2)
    }
)
"""

################################################################################
# principal_components
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def principal_components_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalization: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute principal components using PCA.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of principal components to compute (default: 2).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard').
    distance_metric : Union[str, Callable], optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable (default: 'euclidean').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default: 'closed_form').
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
    _validate_inputs(X, n_components)

    # Normalize data
    X_normalized = _apply_normalization(X, normalization)

    # Compute covariance matrix or distance matrix
    if solver == 'closed_form':
        cov_matrix = _compute_covariance(X_normalized)
    else:
        distance_matrix = _compute_distance_matrix(X_normalized, distance_metric)

    # Solve for principal components
    if solver == 'closed_form':
        components, explained_variance = _solve_closed_form(cov_matrix, n_components)
    else:
        components, explained_variance = _solve_iterative(
            distance_matrix,
            n_components,
            solver,
            tol,
            max_iter
        )

    # Compute metrics
    metrics = _compute_metrics(X_normalized, components, custom_metric)

    # Prepare output
    result = {
        'components': components,
        'explained_variance': explained_variance
    }

    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'normalization': normalization,
            'distance_metric': distance_metric,
            'solver': solver
        },
        'warnings': []
    }

    return output

def _validate_inputs(X: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if n_components <= 0 or n_components > X.shape[1]:
        raise ValueError("n_components must be between 1 and n_features")

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
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_covariance(X: np.ndarray) -> np.ndarray:
    """Compute covariance matrix."""
    return np.cov(X, rowvar=False)

def _compute_distance_matrix(X: np.ndarray, metric: Union[str, Callable]) -> np.ndarray:
    """Compute distance matrix using specified metric."""
    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    if callable(metric):
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance_matrix[i, j] = metric(X[i], X[j])
                distance_matrix[j, i] = distance_matrix[i, j]
    elif metric == 'euclidean':
        for i in range(n_samples):
            distance_matrix[i] = np.linalg.norm(X - X[i], axis=1)
    elif metric == 'manhattan':
        for i in range(n_samples):
            distance_matrix[i] = np.sum(np.abs(X - X[i]), axis=1)
    elif metric == 'cosine':
        for i in range(n_samples):
            distance_matrix[i] = 1 - np.dot(X, X[i]) / (np.linalg.norm(X, axis=1) * np.linalg.norm(X[i]))
    elif metric == 'minkowski':
        p = 3
        for i in range(n_samples):
            distance_matrix[i] = np.sum(np.abs(X - X[i])**p, axis=1)**(1/p)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    return distance_matrix

def _solve_closed_form(cov_matrix: np.ndarray, n_components: int) -> tuple:
    """Solve PCA using closed-form solution."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    components = eigenvectors[:, :n_components]
    explained_variance = eigenvalues[:n_components]

    return components, explained_variance

def _solve_iterative(
    distance_matrix: np.ndarray,
    n_components: int,
    solver: str,
    tol: float,
    max_iter: int
) -> tuple:
    """Solve PCA using iterative methods."""
    n_samples = distance_matrix.shape[0]
    components = np.random.rand(n_samples, n_components)

    for _ in range(max_iter):
        # Update components based on solver
        if solver == 'gradient_descent':
            components = _gradient_descent_step(components, distance_matrix)
        elif solver == 'newton':
            components = _newton_step(components, distance_matrix)
        elif solver == 'coordinate_descent':
            components = _coordinate_descent_step(components, distance_matrix)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        # Check convergence
        if _check_convergence(components, tol):
            break

    explained_variance = np.zeros(n_components)
    return components, explained_variance

def _gradient_descent_step(components: np.ndarray, distance_matrix: np.ndarray) -> np.ndarray:
    """Perform one gradient descent step."""
    # Placeholder for actual implementation
    return components

def _newton_step(components: np.ndarray, distance_matrix: np.ndarray) -> np.ndarray:
    """Perform one Newton step."""
    # Placeholder for actual implementation
    return components

def _coordinate_descent_step(components: np.ndarray, distance_matrix: np.ndarray) -> np.ndarray:
    """Perform one coordinate descent step."""
    # Placeholder for actual implementation
    return components

def _check_convergence(components: np.ndarray, tol: float) -> bool:
    """Check if components have converged."""
    # Placeholder for actual implementation
    return False

def _compute_metrics(
    X: np.ndarray,
    components: np.ndarray,
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute metrics for PCA results."""
    metrics = {}

    if custom_metric is not None:
        metrics['custom'] = custom_metric(X, components)

    return metrics

# Example usage:
"""
X = np.random.rand(100, 5)
result = principal_components_fit(X, n_components=2, normalization='standard')
"""

################################################################################
# eigenvalues
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def eigenvalues_fit(
    matrix: np.ndarray,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Compute eigenvalues of a matrix with configurable options.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix for eigenvalue computation.
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to evaluate performance ('mse', 'mae', 'r2', custom callable).
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    custom_distance : callable, optional
        Custom distance function for solver.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing eigenvalues, metrics, parameters used, and warnings.

    Examples
    --------
    >>> matrix = np.array([[1, 2], [3, 4]])
    >>> result = eigenvalues_fit(matrix)
    """
    # Validate inputs
    _validate_inputs(matrix, normalize)

    # Normalize the matrix if required
    normalized_matrix = _apply_normalization(matrix, normalize)

    # Choose solver and compute eigenvalues
    if solver == 'closed_form':
        eigenvalues, metrics = _closed_form_solver(normalized_matrix)
    elif solver == 'gradient_descent':
        eigenvalues, metrics = _gradient_descent_solver(
            normalized_matrix,
            tol=tol,
            max_iter=max_iter,
            custom_metric=custom_metric,
            custom_distance=custom_distance
        )
    elif solver == 'newton':
        eigenvalues, metrics = _newton_solver(
            normalized_matrix,
            tol=tol,
            max_iter=max_iter,
            custom_metric=custom_metric
        )
    else:
        raise ValueError("Unsupported solver method.")

    # Prepare output dictionary
    result = {
        "result": eigenvalues,
        "metrics": metrics,
        "params_used": {
            "normalize": normalize,
            "metric": metric,
            "solver": solver,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": _check_warnings(eigenvalues)
    }

    return result

def _validate_inputs(matrix: np.ndarray, normalize: str) -> None:
    """Validate input matrix and normalization method."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    if normalize not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("Invalid normalization method.")

def _apply_normalization(matrix: np.ndarray, normalize: str) -> np.ndarray:
    """Apply normalization to the input matrix."""
    if normalize == 'none':
        return matrix
    elif normalize == 'standard':
        return (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)
    elif normalize == 'minmax':
        return (matrix - np.min(matrix, axis=0)) / (np.max(matrix, axis=0) - np.min(matrix, axis=0))
    elif normalize == 'robust':
        return (matrix - np.median(matrix, axis=0)) / (np.percentile(matrix, 75, axis=0) - np.percentile(matrix, 25, axis=0))
    else:
        raise ValueError("Invalid normalization method.")

def _closed_form_solver(matrix: np.ndarray) -> tuple[np.ndarray, Dict[str, float]]:
    """Compute eigenvalues using closed-form solution."""
    eigenvalues = np.linalg.eigvals(matrix)
    metrics = {"mse": _compute_mse(eigenvalues, matrix)}
    return eigenvalues, metrics

def _gradient_descent_solver(
    matrix: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> tuple[np.ndarray, Dict[str, float]]:
    """Compute eigenvalues using gradient descent."""
    # Placeholder for gradient descent implementation
    eigenvalues = np.zeros(matrix.shape[0])
    metrics = {"mse": 0.0}
    return eigenvalues, metrics

def _newton_solver(
    matrix: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None
) -> tuple[np.ndarray, Dict[str, float]]:
    """Compute eigenvalues using Newton's method."""
    # Placeholder for Newton's method implementation
    eigenvalues = np.zeros(matrix.shape[0])
    metrics = {"mse": 0.0}
    return eigenvalues, metrics

def _compute_mse(eigenvalues: np.ndarray, matrix: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return np.mean((np.linalg.eigvals(matrix) - eigenvalues) ** 2)

def _check_warnings(eigenvalues: np.ndarray) -> list[str]:
    """Check for warnings in the eigenvalues computation."""
    warnings = []
    if np.any(np.isnan(eigenvalues)):
        warnings.append("NaN values detected in eigenvalues.")
    if np.any(np.isinf(eigenvalues)):
        warnings.append("Infinite values detected in eigenvalues.")
    return warnings

################################################################################
# eigenvectors
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def eigenvectors_fit(
    data: np.ndarray,
    n_components: int = 2,
    normalize: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Compute the principal eigenvectors of a dataset using PCA.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of principal components to compute, by default 2.
    normalize : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust', by default 'standard'.
    distance_metric : Union[str, Callable], optional
        Distance metric to use: 'euclidean', 'manhattan', 'cosine', or custom callable, by default 'euclidean'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', or 'newton', by default 'closed_form'.
    tol : float, optional
        Tolerance for convergence, by default 1e-6.
    max_iter : int, optional
        Maximum number of iterations for iterative solvers, by default 1000.
    custom_metric : Optional[Callable], optional
        Custom metric function, by default None.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Computed eigenvectors.
        - 'metrics': Performance metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings generated during computation.

    Examples
    --------
    >>> data = np.random.rand(100, 5)
    >>> result = eigenvectors_fit(data, n_components=3)
    """
    # Validate inputs
    _validate_inputs(data, n_components)

    # Normalize data
    normalized_data = _normalize_data(data, normalize)

    # Compute covariance matrix
    cov_matrix = _compute_covariance(normalized_data)

    # Solve for eigenvectors
    eigenvectors = _solve_eigenvectors(
        cov_matrix,
        n_components=n_components,
        solver=solver,
        tol=tol,
        max_iter=max_iter
    )

    # Compute metrics
    metrics = _compute_metrics(normalized_data, eigenvectors, distance_metric, custom_metric)

    # Prepare output
    result = {
        'result': eigenvectors,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'normalize': normalize,
            'distance_metric': distance_metric,
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(data: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")
    if n_components <= 0:
        raise ValueError("n_components must be a positive integer.")
    if n_components > data.shape[1]:
        raise ValueError("n_components cannot be greater than the number of features.")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize the input data."""
    if method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_covariance(data: np.ndarray) -> np.ndarray:
    """Compute the covariance matrix of the data."""
    return np.cov(data, rowvar=False)

def _solve_eigenvectors(
    cov_matrix: np.ndarray,
    n_components: int,
    solver: str,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve for the principal eigenvectors."""
    if solver == 'closed_form':
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, idx[:n_components]]
    elif solver == 'gradient_descent':
        return _gradient_descent_eigenvectors(cov_matrix, n_components, tol, max_iter)
    elif solver == 'newton':
        return _newton_eigenvectors(cov_matrix, n_components, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _gradient_descent_eigenvectors(
    cov_matrix: np.ndarray,
    n_components: int,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Compute eigenvectors using gradient descent."""
    # Placeholder for gradient descent implementation
    return np.random.rand(cov_matrix.shape[0], n_components)

def _newton_eigenvectors(
    cov_matrix: np.ndarray,
    n_components: int,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Compute eigenvectors using Newton's method."""
    # Placeholder for Newton's method implementation
    return np.random.rand(cov_matrix.shape[0], n_components)

def _compute_metrics(
    data: np.ndarray,
    eigenvectors: np.ndarray,
    distance_metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute performance metrics."""
    if callable(distance_metric):
        metric_func = distance_metric
    elif distance_metric == 'euclidean':
        metric_func = lambda x, y: np.linalg.norm(x - y)
    elif distance_metric == 'manhattan':
        metric_func = lambda x, y: np.sum(np.abs(x - y))
    elif distance_metric == 'cosine':
        metric_func = lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    # Compute explained variance
    projected = data @ eigenvectors
    total_variance = np.var(data, axis=0).sum()
    explained_variance = np.var(projected, axis=0).sum() / total_variance

    metrics = {
        'explained_variance': explained_variance,
        'distance_metric': distance_metric
    }

    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(data, eigenvectors)

    return metrics

################################################################################
# covariance_matrix
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def covariance_matrix_fit(
    X: np.ndarray,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_solver: Optional[Callable] = None
) -> Dict:
    """
    Compute the covariance matrix with configurable options.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    normalize : str, optional
        Normalization method: 'none', 'standard', 'minmax', 'robust'
    metric : str or callable, optional
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'minkowski'
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton'
    regularization : str, optional
        Regularization type: 'none', 'l1', 'l2', 'elasticnet'
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    custom_metric : callable, optional
        Custom metric function if needed
    custom_solver : callable, optional
        Custom solver function if needed

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(10, 5)
    >>> result = covariance_matrix_fit(X, normalize='standard', metric='euclidean')
    """
    # Validate inputs
    _validate_inputs(X, normalize, metric, solver)

    # Normalize data if needed
    X_normalized = _apply_normalization(X, normalize)

    # Select metric function
    distance_func = _get_metric_function(metric, custom_metric)

    # Select solver function
    solver_func = _get_solver_function(solver, custom_solver)

    # Compute covariance matrix
    cov_matrix = solver_func(X_normalized, distance_func, regularization, tol, max_iter)

    # Calculate metrics
    metrics = _calculate_metrics(X_normalized, cov_matrix, metric)

    # Prepare output
    result = {
        'result': {'covariance_matrix': cov_matrix},
        'metrics': metrics,
        'params_used': {
            'normalize': normalize,
            'metric': metric if isinstance(metric, str) else 'custom',
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

    return result

def _validate_inputs(
    X: np.ndarray,
    normalize: str,
    metric: Union[str, Callable],
    solver: str
) -> None:
    """Validate input parameters and data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if np.any(np.isnan(X)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input data contains infinite values")

    valid_normalizations = ['none', 'standard', 'minmax', 'robust']
    if normalize not in valid_normalizations:
        raise ValueError(f"normalize must be one of {valid_normalizations}")

    valid_metrics = ['euclidean', 'manhattan', 'cosine', 'minkowski']
    if isinstance(metric, str) and metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics} or a callable")

    valid_solvers = ['closed_form', 'gradient_descent', 'newton']
    if solver not in valid_solvers:
        raise ValueError(f"solver must be one of {valid_solvers}")

def _apply_normalization(
    X: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply selected normalization to the data."""
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
        raise ValueError("Invalid normalization method")

def _get_metric_function(
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Callable:
    """Return the appropriate metric function."""
    if callable(metric):
        return metric
    elif custom_metric is not None:
        return custom_metric
    elif metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))
    elif metric == 'minkowski':
        return lambda x, y: np.sum(np.abs(x - y)**3)**(1/3)
    else:
        raise ValueError("Invalid metric specified")

def _get_solver_function(
    solver: str,
    custom_solver: Optional[Callable] = None
) -> Callable:
    """Return the appropriate solver function."""
    if custom_solver is not None:
        return custom_solver
    elif solver == 'closed_form':
        return _solve_closed_form
    elif solver == 'gradient_descent':
        return _solve_gradient_descent
    elif solver == 'newton':
        return _solve_newton
    else:
        raise ValueError("Invalid solver specified")

def _solve_closed_form(
    X: np.ndarray,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve covariance matrix using closed form solution."""
    return np.cov(X, rowvar=False)

def _solve_gradient_descent(
    X: np.ndarray,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve covariance matrix using gradient descent."""
    # Placeholder implementation
    n_features = X.shape[1]
    cov_matrix = np.eye(n_features)

    for _ in range(max_iter):
        # Update rule would go here
        pass

    return cov_matrix

def _solve_newton(
    X: np.ndarray,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve covariance matrix using Newton's method."""
    # Placeholder implementation
    n_features = X.shape[1]
    cov_matrix = np.eye(n_features)

    for _ in range(max_iter):
        # Update rule would go here
        pass

    return cov_matrix

def _calculate_metrics(
    X: np.ndarray,
    cov_matrix: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate various metrics for the covariance matrix."""
    metrics = {
        'trace': np.trace(cov_matrix),
        'determinant': np.linalg.det(cov_matrix),
        'condition_number': np.linalg.cond(cov_matrix)
    }

    if isinstance(metric, str):
        metrics[f'{metric}_distance'] = _calculate_distance_metric(X, cov_matrix, metric)

    return metrics

def _calculate_distance_metric(
    X: np.ndarray,
    cov_matrix: np.ndarray,
    metric: str
) -> float:
    """Calculate distance-based metrics."""
    if metric == 'euclidean':
        return np.mean(np.linalg.norm(X @ cov_matrix, axis=1))
    elif metric == 'manhattan':
        return np.mean(np.sum(np.abs(X @ cov_matrix), axis=1))
    elif metric == 'cosine':
        return np.mean(1 - (X @ cov_matrix) / (np.linalg.norm(X, axis=1) * np.linalg.norm(cov_matrix)))
    elif metric == 'minkowski':
        return np.mean(np.sum(np.abs(X @ cov_matrix)**3, axis=1)**(1/3))
    else:
        raise ValueError("Invalid metric for distance calculation")

################################################################################
# variance_explained
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def variance_explained_fit(
    X: np.ndarray,
    n_components: int = None,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'variance_ratio',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_normalize: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Compute the explained variance for PCA.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of principal components to compute. If None, all components are computed.
    normalize : str or callable
        Normalization method: 'none', 'standard', 'minmax', 'robust' or custom callable.
    metric : str or callable
        Metric to compute explained variance: 'variance_ratio' or custom callable.
    solver : str
        Solver method: 'closed_form', 'gradient_descent', etc.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_normalize : callable, optional
        Custom normalization function.
    custom_metric : callable, optional
        Custom metric function.

    Returns:
    --------
    Dict containing:
        - 'result': Explained variance.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used.
        - 'warnings': Any warnings generated.

    Example:
    --------
    >>> X = np.random.rand(10, 5)
    >>> result = variance_explained_fit(X, n_components=2)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Normalize data
    X_normalized = _normalize_data(X, normalize, custom_normalize)

    # Compute PCA
    if solver == 'closed_form':
        components, explained_variance = _pca_closed_form(X_normalized, n_components)
    else:
        raise ValueError(f"Solver {solver} not implemented.")

    # Compute metrics
    if metric == 'variance_ratio':
        variance_ratio = _compute_variance_ratio(explained_variance)
    elif callable(metric):
        variance_ratio = metric(explained_variance)
    else:
        raise ValueError(f"Metric {metric} not recognized.")

    # Prepare output
    result = {
        'result': variance_ratio,
        'metrics': {'explained_variance': explained_variance},
        'params_used': {
            'n_components': n_components,
            'normalize': normalize,
            'metric': metric,
            'solver': solver
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
    if n_components is not None and (n_components <= 0 or n_components > X.shape[1]):
        raise ValueError("Invalid number of components.")

def _normalize_data(
    X: np.ndarray,
    normalize: str,
    custom_normalize: Optional[Callable]
) -> np.ndarray:
    """Normalize the input data."""
    if custom_normalize is not None:
        return custom_normalize(X)
    elif normalize == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalize == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalize == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    elif normalize == 'none':
        return X
    else:
        raise ValueError(f"Normalization method {normalize} not recognized.")

def _pca_closed_form(X: np.ndarray, n_components: int) -> tuple:
    """Compute PCA using closed-form solution."""
    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]

    return eigenvectors, eigenvalues

def _compute_variance_ratio(explained_variance: np.ndarray) -> np.ndarray:
    """Compute the explained variance ratio."""
    total_variance = np.sum(explained_variance)
    return explained_variance / total_variance

################################################################################
# explained_variance_ratio
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def explained_variance_ratio_fit(
    X: np.ndarray,
    n_components: int = None,
    normalize: str = 'standard',
    solver: str = 'closed_form',
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_solver: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Compute the explained variance ratio for PCA.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of principal components to compute. If None, all components are computed.
    normalize : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', or custom callable.
    custom_normalize : Callable, optional
        Custom normalization function.
    custom_solver : Callable, optional
        Custom solver function.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
        - "result": Explained variance ratio.
        - "metrics": Metrics dictionary.
        - "params_used": Parameters used.
        - "warnings": Warnings dictionary.

    Examples
    --------
    >>> X = np.random.rand(10, 5)
    >>> result = explained_variance_ratio_fit(X, n_components=2)
    """
    # Validate inputs
    X = _validate_input(X)

    # Normalize data
    if custom_normalize is not None:
        X_normalized = custom_normalize(X)
    else:
        X_normalized = _apply_normalization(X, normalize)

    # Compute PCA
    if custom_solver is not None:
        U, S, Vt = custom_solver(X_normalized)
    else:
        U, S, Vt = _apply_solver(X_normalized, solver)

    # Compute explained variance ratio
    explained_variance = _compute_explained_variance(S, n_components)

    # Prepare output
    metrics = {
        'explained_variance': explained_variance,
        'total_explained_variance': np.sum(explained_variance)
    }

    params_used = {
        'normalize': normalize,
        'solver': solver,
        'n_components': n_components
    }

    warnings = {}

    return {
        'result': explained_variance,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_input(X: np.ndarray) -> np.ndarray:
    """
    Validate input data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix.

    Returns
    -------
    np.ndarray
        Validated input data.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D array.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input X must not contain NaN or Inf values.")
    return X

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """
    Apply normalization to the input data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix.
    method : str
        Normalization method.

    Returns
    -------
    np.ndarray
        Normalized data.
    """
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

def _apply_solver(X: np.ndarray, method: str) -> tuple:
    """
    Apply solver to compute PCA components.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix.
    method : str
        Solver method.

    Returns
    -------
    tuple
        U, S, Vt: PCA components.
    """
    if method == 'closed_form':
        return np.linalg.svd(X, full_matrices=False)
    elif method == 'gradient_descent':
        raise NotImplementedError("Gradient descent solver not implemented.")
    else:
        raise ValueError(f"Unknown solver method: {method}")

def _compute_explained_variance(S: np.ndarray, n_components: int = None) -> np.ndarray:
    """
    Compute explained variance ratio.

    Parameters
    ----------
    S : np.ndarray
        Singular values from SVD.
    n_components : int, optional
        Number of components to consider.

    Returns
    -------
    np.ndarray
        Explained variance ratio.
    """
    if n_components is None:
        n_components = len(S)
    total_variance = np.sum(S**2)
    explained_variance = (S[:n_components]**2) / total_variance
    return explained_variance

################################################################################
# singular_value_decomposition
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(X: np.ndarray) -> None:
    """Validate input matrix X."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input must be a 2-dimensional array.")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input contains NaN or infinite values.")

def _normalize_data(X: np.ndarray, normalization: str) -> np.ndarray:
    """Normalize the input data based on the specified method."""
    if normalization == "none":
        return X
    elif normalization == "standard":
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalization == "minmax":
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalization == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_svd(X: np.ndarray, solver: str) -> tuple:
    """Compute the singular value decomposition using the specified solver."""
    if solver == "closed_form":
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    elif solver == "randomized":
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    else:
        raise ValueError(f"Unknown solver: {solv}")
    return U, S, Vt

def _compute_metrics(U: np.ndarray, S: np.ndarray, Vt: np.ndarray,
                     X: np.ndarray, metric: Union[str, Callable]) -> Dict:
    """Compute the specified metrics for the SVD results."""
    X_reconstructed = U @ np.diag(S) @ Vt
    metrics = {}

    if metric == "mse":
        mse = np.mean((X - X_reconstructed) ** 2)
        metrics["mse"] = mse
    elif metric == "mae":
        mae = np.mean(np.abs(X - X_reconstructed))
        metrics["mae"] = mae
    elif metric == "r2":
        ss_res = np.sum((X - X_reconstructed) ** 2)
        ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        metrics["r2"] = r2
    elif callable(metric):
        metrics["custom"] = metric(X, X_reconstructed)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

def singular_value_decomposition_fit(
    X: np.ndarray,
    normalization: str = "standard",
    solver: str = "closed_form",
    metric: Union[str, Callable] = "mse"
) -> Dict:
    """
    Perform Singular Value Decomposition (SVD) on the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    normalization : str, optional
        Normalization method to apply to the data. Options: "none", "standard", "minmax", "robust".
    solver : str, optional
        Solver to use for SVD. Options: "closed_form", "randomized".
    metric : str or callable, optional
        Metric to compute for the SVD results. Options: "mse", "mae", "r2", or a custom callable.

    Returns:
    --------
    Dict
        A dictionary containing the results, metrics, parameters used, and any warnings.
    """
    _validate_input(X)
    X_normalized = _normalize_data(X, normalization)
    U, S, Vt = _compute_svd(X_normalized, solver)
    metrics = _compute_metrics(U, S, Vt, X_normalized, metric)

    return {
        "result": {
            "U": U,
            "S": S,
            "Vt": Vt
        },
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "solver": solver,
            "metric": metric
        },
        "warnings": []
    }

# Example usage:
# result = singular_value_decomposition_fit(X, normalization="standard", solver="closed_form", metric="mse")

################################################################################
# whitening
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or Inf values")

def _normalize_data(
    X: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """Normalize data based on specified method."""
    if custom_func is not None:
        return custom_func(X)

    X_norm = X.copy()
    if method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
    return X_norm

def _compute_whitening_matrix(
    X: np.ndarray,
    method: str = "closed_form",
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """Compute whitening matrix based on specified method."""
    if custom_func is not None:
        return custom_func(X)

    cov = np.cov(X, rowvar=False)
    if method == "closed_form":
        eigvals, eigvecs = np.linalg.eigh(cov)
        D = np.diag(1.0 / np.sqrt(eigvals + 1e-8))
        W = eigvecs @ D @ eigvecs.T
    return W

def _compute_metrics(
    X: np.ndarray,
    X_white: np.ndarray,
    metric_funcs: Dict[str, Callable]
) -> Dict[str, float]:
    """Compute specified metrics."""
    metrics = {}
    for name, func in metric_funcs.items():
        if name == "mse":
            metrics[name] = np.mean((X - X_white) ** 2)
        elif name == "mae":
            metrics[name] = np.mean(np.abs(X - X_white))
        elif name == "r2":
            ss_total = np.sum((X - np.mean(X, axis=0)) ** 2)
            ss_res = np.sum((X - X_white) ** 2)
            metrics[name] = 1 - (ss_res / ss_total)
        else:
            metrics[name] = func(X, X_white)
    return metrics

def whitening_fit(
    X: np.ndarray,
    normalization_method: str = "standard",
    custom_normalization: Optional[Callable] = None,
    whitening_method: str = "closed_form",
    custom_whitening: Optional[Callable] = None,
    metrics: Dict[str, Union[str, Callable]] = {"mse": "mse", "r2": "r2"},
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Perform whitening on input data.

    Parameters:
    - X: Input data (2D numpy array)
    - normalization_method: Normalization method ("standard", "minmax", "robust")
    - custom_normalization: Custom normalization function
    - whitening_method: Whitening method ("closed_form")
    - custom_whitening: Custom whitening function
    - metrics: Dictionary of metric names and functions

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    _validate_input(X)

    # Normalize data
    X_norm = _normalize_data(
        X,
        method=normalization_method,
        custom_func=custom_normalization
    )

    # Compute whitening matrix
    W = _compute_whitening_matrix(
        X_norm,
        method=whitening_method,
        custom_func=custom_whitening
    )

    # Apply whitening
    X_white = X_norm @ W

    # Compute metrics
    metric_funcs = {}
    for name, func in metrics.items():
        if isinstance(func, str):
            if func == "mse":
                metric_funcs[name] = lambda x, y: np.mean((x - y) ** 2)
            elif func == "mae":
                metric_funcs[name] = lambda x, y: np.mean(np.abs(x - y))
            elif func == "r2":
                metric_funcs[name] = lambda x, y: 1 - np.sum((x - y) ** 2) / np.sum((x - np.mean(x, axis=0)) ** 2)
        else:
            metric_funcs[name] = func

    metrics_result = _compute_metrics(X_norm, X_white, metric_funcs)

    return {
        "result": X_white,
        "metrics": metrics_result,
        "params_used": {
            "normalization_method": normalization_method,
            "whitening_method": whitening_method
        },
        "warnings": []
    }

# Example usage:
"""
X = np.random.randn(100, 5)
result = whitening_fit(
    X,
    normalization_method="standard",
    metrics={"mse": "mse", "r2": "r2"}
)
"""

################################################################################
# kernel_pca
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, kernel: Callable) -> None:
    """Validate input data and kernel function."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("X contains infinite values")

def _center_kernel_matrix(K: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Center the kernel matrix."""
    n_samples = K.shape[0]
    one_n = np.ones((n_samples, n_samples)) / n_samples
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    if normalize:
        K_centered /= n_samples
    return K_centered

def _eigendecomposition(K: np.ndarray, n_components: int) -> tuple:
    """Perform eigendecomposition on the kernel matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][:n_components]
    eigenvectors = eigenvectors[:, idx][:, :n_components]
    return eigenvalues, eigenvectors

def kernel_pca_fit(
    X: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], float],
    n_components: int = 2,
    normalize: bool = True,
    eigen_solver: str = 'auto',
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Perform Kernel PCA on the input data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    kernel : Callable[[np.ndarray, np.ndarray], float]
        Kernel function to use.
    n_components : int, optional
        Number of principal components to return, by default 2.
    normalize : bool, optional
        Whether to normalize the centered kernel matrix, by default True.
    eigen_solver : str, optional
        Solver to use for the eigendecomposition, by default 'auto'.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    _validate_inputs(X, kernel)

    # Compute the kernel matrix
    K = np.array([[kernel(xi, xj) for xi in X] for xj in X])

    # Center the kernel matrix
    K_centered = _center_kernel_matrix(K, normalize)

    # Perform eigendecomposition
    eigenvalues, eigenvectors = _eigendecomposition(K_centered, n_components)

    # Prepare the output
    result = {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
    }

    metrics = {
        'explained_variance_ratio': eigenvalues / np.sum(eigenvalues),
    }

    params_used = {
        'n_components': n_components,
        'normalize': normalize,
        'eigen_solver': eigen_solver,
    }

    warnings = {
        'no_warnings': True,
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings,
    }

# Example usage:
"""
X = np.random.rand(100, 5)
def rbf_kernel(x, y):
    gamma = 1.0
    return np.exp(-gamma * np.linalg.norm(x - y)**2)

result = kernel_pca_fit(X, rbf_kernel, n_components=3)
"""

################################################################################
# sparse_pca
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def sparse_pca_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalize: str = 'standard',
    solver: str = 'coordinate_descent',
    penalty: str = 'l1',
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    **kwargs
) -> Dict:
    """
    Perform Sparse Principal Component Analysis (PCA).

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of sparse principal components to extract.
    normalize : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    solver : str, optional
        Solver method: 'coordinate_descent', 'gradient_descent'.
    penalty : str, optional
        Penalty type: 'none', 'l1', 'l2', or 'elasticnet'.
    alpha : float, optional
        Regularization strength.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    random_state : int, optional
        Random seed for reproducibility.
    metric : str or callable, optional
        Metric to evaluate performance: 'mse', 'mae', 'r2', or custom callable.
    **kwargs : dict
        Additional solver-specific parameters.

    Returns:
    --------
    Dict containing:
        - 'result': Dictionary with components and explained variance.
        - 'metrics': Dictionary of performance metrics.
        - 'params_used': Dictionary of parameters used.
        - 'warnings': List of warnings encountered.

    Example:
    --------
    >>> X = np.random.randn(100, 20)
    >>> result = sparse_pca_fit(X, n_components=3, normalize='standard')
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Normalize data
    X_normalized = _normalize_data(X, normalize)

    # Initialize components
    components = np.zeros((n_components, X_normalized.shape[1]))

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Solve for each component
    for i in range(n_components):
        components[i], _ = _solve_sparse_pca(
            X_normalized,
            solver=solver,
            penalty=penalty,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            **kwargs
        )
        X_normalized = _deflate(X_normalized, components[i])

    # Calculate explained variance
    explained_variance = _calculate_explained_variance(X, components)

    # Calculate metrics
    metrics = _calculate_metrics(
        X,
        components,
        metric=metric
    )

    # Prepare output
    result = {
        'components': components,
        'explained_variance': explained_variance
    }

    params_used = {
        'n_components': n_components,
        'normalize': normalize,
        'solver': solver,
        'penalty': penalty,
        'alpha': alpha,
        'max_iter': max_iter,
        'tol': tol,
        'random_state': random_state
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if n_components > X.shape[1]:
        raise ValueError("n_components cannot be greater than number of features")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data according to specified method."""
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

def _solve_sparse_pca(
    X: np.ndarray,
    solver: str,
    penalty: str,
    alpha: float,
    max_iter: int,
    tol: float,
    **kwargs
) -> tuple:
    """Solve sparse PCA problem using specified solver."""
    if solver == 'coordinate_descent':
        return _coordinate_descent_solver(X, penalty, alpha, max_iter, tol)
    elif solver == 'gradient_descent':
        return _gradient_descent_solver(X, penalty, alpha, max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _coordinate_descent_solver(
    X: np.ndarray,
    penalty: str,
    alpha: float,
    max_iter: int,
    tol: float
) -> tuple:
    """Coordinate descent solver for sparse PCA."""
    n_features = X.shape[1]
    w = np.random.randn(n_features)

    for _ in range(max_iter):
        old_w = w.copy()

        for j in range(n_features):
            # Compute gradient
            grad = 2 * np.dot(X.T, X @ w - X[:, j] * w[j])

            # Update weight
            if penalty == 'l1':
                w[j] = np.sign(grad) * np.maximum(np.abs(grad) - alpha, 0)
            elif penalty == 'l2':
                w[j] = grad / (2 * alpha + 1e-6)
            elif penalty == 'elasticnet':
                w[j] = np.sign(grad) * np.maximum(np.abs(grad) - alpha, 0)
                w[j] = grad / (2 * alpha + 1e-6) if np.abs(grad) < alpha else w[j]
            elif penalty == 'none':
                w[j] = grad

        # Normalize
        w = w / np.linalg.norm(w)

        if np.linalg.norm(w - old_w) < tol:
            break

    return w, None

def _gradient_descent_solver(
    X: np.ndarray,
    penalty: str,
    alpha: float,
    max_iter: int,
    tol: float
) -> tuple:
    """Gradient descent solver for sparse PCA."""
    n_features = X.shape[1]
    w = np.random.randn(n_features)

    for _ in range(max_iter):
        old_w = w.copy()

        # Compute gradient
        grad = 2 * X.T @ (X @ w - np.diag(X @ w))

        # Apply penalty
        if penalty == 'l1':
            grad += alpha * np.sign(w)
        elif penalty == 'l2':
            grad += 2 * alpha * w
        elif penalty == 'elasticnet':
            grad += alpha * (np.sign(w) + 2 * w)
        elif penalty == 'none':
            pass

        # Update weights
        w -= grad

        # Normalize
        w = w / np.linalg.norm(w)

        if np.linalg.norm(w - old_w) < tol:
            break

    return w, None

def _deflate(X: np.ndarray, component: np.ndarray) -> np.ndarray:
    """Deflate the data by subtracting the current component."""
    return X - X @ component[:, np.newaxis] * component

def _calculate_explained_variance(X: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Calculate explained variance for each component."""
    total_variance = np.var(X, axis=0).sum()
    explained_variance = np.zeros(components.shape[0])

    for i in range(components.shape[0]):
        projected = X @ components[i]
        explained_variance[i] = np.var(projected) / total_variance

    return explained_variance

def _calculate_metrics(
    X: np.ndarray,
    components: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate performance metrics."""
    if callable(metric):
        return {'custom_metric': metric(X, components)}

    metrics = {}
    if metric == 'mse':
        projected = X @ components.T
        mse = np.mean((X - projected) ** 2)
        metrics['mse'] = mse
    elif metric == 'mae':
        projected = X @ components.T
        mae = np.mean(np.abs(X - projected))
        metrics['mae'] = mae
    elif metric == 'r2':
        projected = X @ components.T
        ss_res = np.sum((X - projected) ** 2)
        ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        metrics['r2'] = r2
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

################################################################################
# incremental_pca
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def incremental_pca_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalize: str = 'standard',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    metric: str = 'mse',
    distance: str = 'euclidean',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Perform Incremental Principal Component Analysis (PCA).

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of principal components to keep. Default is 2.
    normalize : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'. Default is 'standard'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'. Default is 'closed_form'.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    metric : str, optional
        Metric for evaluation: 'mse', 'mae', 'r2', or custom callable. Default is 'mse'.
    distance : str, optional
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'minkowski', or custom callable. Default is 'euclidean'.
    regularization : str, optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'. Default is None.
    alpha : float, optional
        Regularization strength. Default is 1.0.
    l1_ratio : float, optional
        ElasticNet mixing parameter. Default is 0.5.
    custom_metric : Callable, optional
        Custom metric function. Default is None.
    custom_distance : Callable, optional
        Custom distance function. Default is None.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Computed principal components.
        - 'metrics': Evaluation metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings generated during computation.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> result = incremental_pca_fit(X, n_components=3)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Normalize data
    X_normalized = _normalize_data(X, normalize)

    # Choose solver
    if solver == 'closed_form':
        components = _closed_form_pca(X_normalized, n_components)
    elif solver == 'gradient_descent':
        components = _gradient_descent_pca(X_normalized, n_components, tol, max_iter)
    elif solver == 'newton':
        components = _newton_pca(X_normalized, n_components, tol, max_iter)
    elif solver == 'coordinate_descent':
        components = _coordinate_descent_pca(X_normalized, n_components, tol, max_iter)
    else:
        raise ValueError("Invalid solver specified.")

    # Apply regularization if needed
    if regularization is not None:
        components = _apply_regularization(components, regularization, alpha, l1_ratio)

    # Compute metrics
    metrics = _compute_metrics(X_normalized, components, metric, custom_metric)

    # Prepare output
    result = {
        'result': components,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'normalize': normalize,
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter,
            'metric': metric,
            'distance': distance,
            'regularization': regularization,
            'alpha': alpha,
            'l1_ratio': l1_ratio
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

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize the input data."""
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

def _closed_form_pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """Compute PCA using closed-form solution."""
    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvectors[:, :n_components]

def _gradient_descent_pca(X: np.ndarray, n_components: int, tol: float, max_iter: int) -> np.ndarray:
    """Compute PCA using gradient descent."""
    n_samples, n_features = X.shape
    components = np.random.randn(n_features, n_components)
    for _ in range(max_iter):
        gradients = 2 * X.T @ (X @ components - X @ components @ components.T @ components)
        components -= tol * gradients
        if np.linalg.norm(gradients) < tol:
            break
    return components

def _newton_pca(X: np.ndarray, n_components: int, tol: float, max_iter: int) -> np.ndarray:
    """Compute PCA using Newton's method."""
    n_samples, n_features = X.shape
    components = np.random.randn(n_features, n_components)
    for _ in range(max_iter):
        hessian = 2 * X.T @ X
        gradients = 2 * X.T @ (X @ components - X @ components @ components.T @ components)
        components -= np.linalg.inv(hessian) @ gradients
        if np.linalg.norm(gradients) < tol:
            break
    return components

def _coordinate_descent_pca(X: np.ndarray, n_components: int, tol: float, max_iter: int) -> np.ndarray:
    """Compute PCA using coordinate descent."""
    n_samples, n_features = X.shape
    components = np.random.randn(n_features, n_components)
    for _ in range(max_iter):
        for i in range(n_features):
            for j in range(n_components):
                components[i, j] -= tol * (2 * X[:, i].T @ (X @ components) - 2 * X[:, i].T @ (X @ components @ components.T @ components))[i, j]
        if np.linalg.norm(2 * X.T @ (X @ components - X @ components @ components.T @ components)) < tol:
            break
    return components

def _apply_regularization(components: np.ndarray, method: str, alpha: float, l1_ratio: float) -> np.ndarray:
    """Apply regularization to the components."""
    if method == 'l1':
        return np.sign(components) * (np.abs(components) - alpha)
    elif method == 'l2':
        return components / (1 + alpha * np.linalg.norm(components, axis=0))
    elif method == 'elasticnet':
        l1_penalty = l1_ratio * alpha
        l2_penalty = (1 - l1_ratio) * alpha
        return np.sign(components) * (np.abs(components) - l1_penalty) / (1 + l2_penalty)
    else:
        return components

def _compute_metrics(X: np.ndarray, components: np.ndarray, metric: str, custom_metric: Optional[Callable]) -> Dict:
    """Compute evaluation metrics."""
    if custom_metric is not None:
        return {'custom': custom_metric(X, components)}

    metrics = {}
    if metric == 'mse':
        reconstructed = X @ components @ components.T
        metrics['mse'] = np.mean((X - reconstructed) ** 2)
    elif metric == 'mae':
        reconstructed = X @ components @ components.T
        metrics['mae'] = np.mean(np.abs(X - reconstructed))
    elif metric == 'r2':
        reconstructed = X @ components @ components.T
        ss_res = np.sum((X - reconstructed) ** 2)
        ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    else:
        raise ValueError("Invalid metric specified.")

    return metrics

################################################################################
# randomized_pca
################################################################################

import numpy as np
from typing import Optional, Callable, Dict, Any

def randomized_pca_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    random_state: Optional[int] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Perform Randomized PCA on input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of principal components to keep.
    normalizer : Optional[Callable], optional
        Function to normalize the data. If None, no normalization is applied.
    distance_metric : str, optional
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine'.
    solver : str, optional
        Solver to use. Options: 'closed_form', 'gradient_descent'.
    random_state : Optional[int], optional
        Random seed for reproducibility.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    custom_metric : Optional[Callable], optional
        Custom metric function. If provided, overrides distance_metric.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Set random state
    rng = np.random.RandomState(random_state)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Perform PCA based on solver choice
    if solver == 'closed_form':
        components, explained_variance = _randomized_pca_closed_form(
            X_normalized, n_components, rng
        )
    elif solver == 'gradient_descent':
        components, explained_variance = _randomized_pca_gradient_descent(
            X_normalized, n_components, rng, max_iter, tol
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(X_normalized, components, custom_metric)

    return {
        'result': {
            'components': components,
            'explained_variance': explained_variance
        },
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'normalizer': normalizer.__name__ if normalizer else None,
            'distance_metric': distance_metric,
            'solver': solver,
            'random_state': random_state
        },
        'warnings': []
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
        raise ValueError("n_components cannot be greater than number of features")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to the data."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _randomized_pca_closed_form(
    X: np.ndarray,
    n_components: int,
    rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    """Perform Randomized PCA using closed-form solution."""
    n_samples, n_features = X.shape

    # Step 1: Random projection
    Omega = rng.normal(size=(n_features, n_components))

    # Step 2: Compute Q
    Y = X @ Omega
    Q, _ = np.linalg.qr(Y)

    # Step 3: Compute B
    B = Q.T @ X

    # Step 4: Compute SVD of B
    U, S, Vt = np.linalg.svd(B)

    # Step 5: Compute final components
    components = Q @ U[:, :n_components]
    explained_variance = S[:n_components] ** 2 / (n_samples - 1)

    return components, explained_variance

def _randomized_pca_gradient_descent(
    X: np.ndarray,
    n_components: int,
    rng: np.random.RandomState,
    max_iter: int,
    tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Perform Randomized PCA using gradient descent."""
    n_samples, n_features = X.shape
    components = rng.normal(size=(n_features, n_components))

    for _ in range(max_iter):
        old_components = components.copy()

        # Update components using gradient descent
        gradients = _compute_gradient(X, components)
        components -= tol * gradients

        # Normalize components
        norms = np.linalg.norm(components, axis=0)
        components /= norms

        # Check convergence
        if np.linalg.norm(components - old_components) < tol:
            break

    # Compute explained variance
    projected = X @ components
    total_variance = np.var(X, axis=0).sum()
    explained_variance = np.var(projected, axis=0)

    return components, explained_variance

def _compute_gradient(X: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Compute gradient for PCA."""
    n_samples = X.shape[0]
    projected = X @ components
    residuals = X - projected @ components.T
    gradient = (2 / n_samples) * (residuals.T @ X)
    return gradient

def _calculate_metrics(
    X: np.ndarray,
    components: np.ndarray,
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Calculate metrics for PCA."""
    projected = X @ components
    residuals = X - projected

    metrics = {}

    if custom_metric is not None:
        try:
            metrics['custom'] = custom_metric(X, projected)
        except Exception as e:
            raise ValueError(f"Custom metric failed: {str(e)}")
    else:
        metrics['mse'] = np.mean(residuals ** 2)
        metrics['mae'] = np.mean(np.abs(residuals))
        metrics['r2'] = 1 - (np.sum(residuals ** 2) / np.sum((X - np.mean(X, axis=0)) ** 2))

    return metrics

# Example usage:
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

# Generate random data
X = np.random.randn(100, 5)

# Define a custom normalizer
def robust_normalizer(X):
    return (X - np.median(X, axis=0)) / (2 * np.median(np.abs(X - np.median(X, axis=0)), axis=0))

# Perform randomized PCA
result = randomized_pca_fit(
    X,
    n_components=2,
    normalizer=StandardScaler(),
    distance_metric='euclidean',
    solver='closed_form',
    random_state=42
)

print(result)
"""
