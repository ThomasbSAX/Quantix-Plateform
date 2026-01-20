"""
Quantix – Module nmf
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# factorization
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def factorization_fit(
    X: np.ndarray,
    n_components: int,
    *,
    normalization: str = "none",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "gradient_descent",
    regularization: Optional[str] = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict:
    """
    Perform Non-negative Matrix Factorization (NMF) on input data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int
        Number of components to factorize into
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2', 'logloss') or custom callable
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    regularization : str or None, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Tolerance for stopping criterion
    random_state : int or None, optional
        Random seed for reproducibility
    verbose : bool, optional
        Whether to print progress information

    Returns
    -------
    Dict containing:
        - 'result': Dictionary with factors W and H
        - 'metrics': Dictionary of computed metrics
        - 'params_used': Dictionary of parameters actually used
        - 'warnings': List of warnings encountered

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> result = factorization_fit(X, n_components=3)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Initialize parameters
    params_used = {
        'normalization': normalization,
        'metric': metric,
        'distance': distance,
        'solver': solver,
        'regularization': regularization,
        'max_iter': max_iter,
        'tol': tol
    }

    # Normalize data if requested
    X_norm, norm_params = _apply_normalization(X, normalization)

    # Initialize factors
    W, H = _initialize_factors(n_components, X_norm.shape[1], random_state)

    # Get solver function
    solver_func = _get_solver(solver, regularization)

    # Main optimization loop
    for iteration in range(max_iter):
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}")

        # Update factors
        W, H = solver_func(X_norm, W, H)

        # Check convergence
        if _check_convergence(W, H, iteration, tol):
            break

    # Calculate metrics
    metrics = _calculate_metrics(X_norm, W, H, metric)

    # Prepare result
    result = {
        'W': W,
        'H': H,
        'norm_params': norm_params
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
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")

def _apply_normalization(
    X: np.ndarray,
    method: str
) -> tuple[np.ndarray, Optional[Dict]]:
    """Apply specified normalization to data."""
    if method == "none":
        return X, None
    elif method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
        return X_norm, {'mean': mean, 'std': std}
    elif method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
        return X_norm, {'min': min_val, 'max': max_val}
    elif method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
        return X_norm, {'median': median, 'iqr': iqr}
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _initialize_factors(
    n_components: int,
    n_features: int,
    random_state: Optional[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize factor matrices with random values."""
    if random_state is not None:
        np.random.seed(random_state)
    W = np.abs(np.random.randn(n_features, n_components))
    H = np.abs(np.random.randn(n_components, n_features))
    return W, H

def _get_solver(
    method: str,
    regularization: Optional[str]
) -> Callable:
    """Return appropriate solver function based on parameters."""
    if method == "closed_form":
        return _solve_closed_form
    elif method == "gradient_descent":
        if regularization is None:
            return _solve_gradient_descent
        elif regularization == "l1":
            return _solve_lasso
        elif regularization == "l2":
            return _solve_ridge
        elif regularization == "elasticnet":
            return _solve_elasticnet
    elif method == "newton":
        return _solve_newton
    elif method == "coordinate_descent":
        if regularization is None:
            return _solve_coordinate_descent
        elif regularization == "l1":
            return _solve_lasso_cd
    else:
        raise ValueError(f"Unknown solver method: {method}")

def _solve_gradient_descent(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Update factors using gradient descent."""
    # Update H
    grad_H = 2 * (W.T @ W @ H - W.T @ X)
    H -= 0.01 * grad_H
    H = np.abs(H)  # Ensure non-negativity

    # Update W
    grad_W = 2 * (W @ H @ H.T - X @ H.T)
    W -= 0.01 * grad_W
    W = np.abs(W)  # Ensure non-negativity

    return W, H

def _calculate_metrics(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate specified metrics."""
    X_reconstructed = W @ H

    if callable(metric):
        return {'custom': metric(X, X_reconstructed)}

    metrics = {}
    if metric == "mse" or 'mse' in metric:
        mse = np.mean((X - X_reconstructed) ** 2)
        metrics['mse'] = mse
    if metric == "mae" or 'mae' in metric:
        mae = np.mean(np.abs(X - X_reconstructed))
        metrics['mae'] = mae
    if metric == "r2" or 'r2' in metric:
        ss_res = np.sum((X - X_reconstructed) ** 2)
        ss_tot = np.sum((X - np.mean(X, axis=1, keepdims=True)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        metrics['r2'] = r2
    if metric == "logloss" or 'logloss' in metric:
        epsilon = 1e-8
        logloss = -np.mean(X * np.log(X_reconstructed + epsilon) +
                          (1 - X) * np.log(1 - X_reconstructed + epsilon))
        metrics['logloss'] = logloss

    return metrics

def _check_convergence(
    W: np.ndarray,
    H: np.ndarray,
    iteration: int,
    tol: float
) -> bool:
    """Check if optimization has converged."""
    # Simple convergence check based on parameter changes
    return iteration > 0 and np.max(np.abs(W - W_prev)) < tol and \
           np.max(np.abs(H - H_prev)) < tol

################################################################################
# non_negative_constraints
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def non_negative_constraints_fit(
    X: np.ndarray,
    n_components: int,
    solver: str = 'gradient_descent',
    normalization: str = 'none',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 200,
    custom_solver: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Applies non-negative constraints to a matrix factorization problem.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int
        Number of components for the factorization.
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to evaluate the solution ('mse', 'mae', 'r2', 'logloss').
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski').
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for stopping criterion.
    max_iter : int, optional
        Maximum number of iterations.
    custom_solver : callable, optional
        Custom solver function.
    **kwargs :
        Additional solver-specific parameters.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Factorized matrices (W, H)
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in the computation
        - 'warnings': Any warnings generated

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> result = non_negative_constraints_fit(X, n_components=2)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Normalize data if required
    X_normalized = _apply_normalization(X, normalization)

    # Initialize parameters
    W, H = _initialize_parameters(X_normalized.shape[0], X_normalized.shape[1], n_components)

    # Choose solver
    if custom_solver is not None:
        W, H = custom_solver(X_normalized, W, H, **kwargs)
    else:
        solver_func = _get_solver(solver)
        W, H = solver_func(X_normalized, W, H, n_components, tol, max_iter, **kwargs)

    # Apply non-negative constraints
    W = np.maximum(W, 0)
    H = np.maximum(H, 0)

    # Compute metrics
    metrics = _compute_metrics(X_normalized, W, H, metric)

    # Prepare output
    result = {
        'result': {'W': W, 'H': H},
        'metrics': metrics,
        'params_used': {
            'solver': solver,
            'normalization': normalization,
            'metric': metric,
            'distance': distance,
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
        raise ValueError("X must be a 2-dimensional array")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X must not contain NaN or Inf values")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to the input data."""
    if method == 'standard':
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        X_normalized = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        X_normalized = X.copy()
    return X_normalized

def _initialize_parameters(n_samples: int, n_features: int, n_components: int) -> tuple:
    """Initialize W and H matrices."""
    W = np.random.rand(n_samples, n_components)
    H = np.random.rand(n_components, n_features)
    return W, H

def _get_solver(solver: str) -> Callable:
    """Get the appropriate solver function."""
    solvers = {
        'closed_form': _closed_form_solver,
        'gradient_descent': _gradient_descent_solver,
        'newton': _newton_solver,
        'coordinate_descent': _coordinate_descent_solver
    }
    if solver not in solvers:
        raise ValueError(f"Unknown solver: {solver}")
    return solvers[solver]

def _closed_form_solver(X: np.ndarray, W: np.ndarray, H: np.ndarray, n_components: int,
                        tol: float, max_iter: int, **kwargs) -> tuple:
    """Closed form solver for NMF."""
    # Implement closed form solution
    return W, H

def _gradient_descent_solver(X: np.ndarray, W: np.ndarray, H: np.ndarray, n_components: int,
                             tol: float, max_iter: int, **kwargs) -> tuple:
    """Gradient descent solver for NMF."""
    # Implement gradient descent solution
    return W, H

def _newton_solver(X: np.ndarray, W: np.ndarray, H: np.ndarray, n_components: int,
                   tol: float, max_iter: int, **kwargs) -> tuple:
    """Newton solver for NMF."""
    # Implement Newton's method solution
    return W, H

def _coordinate_descent_solver(X: np.ndarray, W: np.ndarray, H: np.ndarray, n_components: int,
                               tol: float, max_iter: int, **kwargs) -> tuple:
    """Coordinate descent solver for NMF."""
    # Implement coordinate descent solution
    return W, H

def _compute_metrics(X: np.ndarray, W: np.ndarray, H: np.ndarray,
                     metric: Union[str, Callable]) -> Dict:
    """Compute metrics for the factorization."""
    reconstruction = np.dot(W, H)
    metrics_dict = {}

    if metric == 'mse':
        mse = np.mean((X - reconstruction) ** 2)
        metrics_dict['mse'] = mse
    elif metric == 'mae':
        mae = np.mean(np.abs(X - reconstruction))
        metrics_dict['mae'] = mae
    elif metric == 'r2':
        ss_res = np.sum((X - reconstruction) ** 2)
        ss_tot = np.sum((X - np.mean(X)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        metrics_dict['r2'] = r2
    elif metric == 'logloss':
        logloss = -np.mean(X * np.log(reconstruction + 1e-10))
        metrics_dict['logloss'] = logloss
    elif callable(metric):
        custom_metric = metric(X, reconstruction)
        metrics_dict['custom'] = custom_metric

    return metrics_dict

################################################################################
# matrix_decomposition
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def matrix_decomposition_fit(
    X: np.ndarray,
    n_components: int,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = "mse",
    solver: str = "gradient_descent",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 200,
    random_state: Optional[int] = None
) -> Dict:
    """
    Perform Non-negative Matrix Factorization (NMF) on input matrix X.

    Parameters:
    -----------
    X : np.ndarray
        Input non-negative matrix to decompose (shape: n_samples, n_features)
    n_components : int
        Number of components in the decomposition
    normalizer : Optional[Callable]
        Function to normalize X before decomposition. If None, no normalization.
    metric : Union[str, Callable]
        Metric to evaluate decomposition quality. Can be "mse", "mae", or custom callable.
    solver : str
        Solver to use for optimization. Options: "gradient_descent", "coordinate_descent"
    regularization : Optional[str]
        Regularization type. Options: None, "l1", "l2"
    tol : float
        Tolerance for stopping criterion
    max_iter : int
        Maximum number of iterations
    random_state : Optional[int]
        Random seed for reproducibility

    Returns:
    --------
    Dict containing:
        - "result": dict with factors W and H
        - "metrics": dict of computed metrics
        - "params_used": dict of parameters actually used
        - "warnings": list of warnings encountered

    Example:
    --------
    >>> X = np.random.rand(10, 5)
    >>> result = matrix_decomposition_fit(X, n_components=3)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Initialize factors
    W = _initialize_factors(X.shape[0], n_components)
    H = _initialize_factors(n_components, X.shape[1])

    # Normalize if requested
    X_normalized = _apply_normalization(X, normalizer)

    # Choose solver
    if solver == "gradient_descent":
        W, H = _gradient_descent_solver(X_normalized, W, H,
                                      metric=metric,
                                      regularization=regularization,
                                      tol=tol,
                                      max_iter=max_iter)
    elif solver == "coordinate_descent":
        W, H = _coordinate_descent_solver(X_normalized, W, H,
                                        metric=metric,
                                        regularization=regularization,
                                        tol=tol,
                                        max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, W, H, metric)

    return {
        "result": {"W": W, "H": H},
        "metrics": metrics,
        "params_used": {
            "n_components": n_components,
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, n_components: int) -> None:
    """Validate input matrix and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if np.any(X < 0):
        raise ValueError("NMF requires non-negative input matrix")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input matrix contains NaN or Inf values")

def _initialize_factors(n_rows: int, n_cols: int) -> np.ndarray:
    """Initialize factors with random non-negative values."""
    return np.abs(np.random.randn(n_rows, n_cols))

def _apply_normalization(X: np.ndarray,
                        normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization if requested."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _gradient_descent_solver(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    *,
    metric: Union[str, Callable],
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Perform NMF using gradient descent."""
    for _ in range(max_iter):
        # Update H
        H = _update_h(X, W, H, regularization)

        # Update W
        W = _update_w(X, W, H, regularization)

        # Check convergence
        if _check_convergence(X, W, H, metric, tol):
            break

    return W, H

def _coordinate_descent_solver(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    *,
    metric: Union[str, Callable],
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Perform NMF using coordinate descent."""
    for _ in range(max_iter):
        # Update H
        H = _update_h_coordinate(X, W, H, regularization)

        # Update W
        W = _update_w_coordinate(X, W, H, regularization)

        # Check convergence
        if _check_convergence(X, W, H, metric, tol):
            break

    return W, H

def _update_h(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Update H matrix using gradient descent."""
    WH = W @ H
    grad_H = (W.T @ (WH - X)) / X.shape[0]

    if regularization == "l1":
        grad_H += np.sign(H)
    elif regularization == "l2":
        grad_H += 2 * H

    return np.maximum(H - 0.1 * grad_H, 0)

def _update_w(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Update W matrix using gradient descent."""
    WH = W @ H
    grad_W = (WH - X) @ H.T / X.shape[0]

    if regularization == "l1":
        grad_W += np.sign(W)
    elif regularization == "l2":
        grad_W += 2 * W

    return np.maximum(W - 0.1 * grad_W, 0)

def _update_h_coordinate(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Update H matrix using coordinate descent."""
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            # Compute gradient
            grad = (W.T @ (W @ H - X))[i, j]

            # Update with regularization
            if regularization == "l1":
                grad += 1
            elif regularization == "l2":
                grad += 2 * H[i, j]

            # Update value
            H_new = np.maximum(H[i, j] - 0.1 * grad, 0)
            H[i, j] = H_new

    return H

def _update_w_coordinate(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Update W matrix using coordinate descent."""
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            # Compute gradient
            grad = ((W @ H - X) @ H.T)[i, j]

            # Update with regularization
            if regularization == "l1":
                grad += 1
            elif regularization == "l2":
                grad += 2 * W[i, j]

            # Update value
            W_new = np.maximum(W[i, j] - 0.1 * grad, 0)
            W[i, j] = W_new

    return W

def _check_convergence(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    metric: Union[str, Callable],
    tol: float
) -> bool:
    """Check if convergence criteria is met."""
    current_metric = _compute_single_metric(X, W @ H, metric)
    return current_metric < tol

def _compute_single_metric(
    X: np.ndarray,
    X_reconstructed: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Compute a single metric between original and reconstructed matrix."""
    if callable(metric):
        return metric(X, X_reconstructed)

    if metric == "mse":
        return np.mean((X - X_reconstructed) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(X - X_reconstructed))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_metrics(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Compute all requested metrics."""
    X_reconstructed = W @ H
    return {
        "reconstruction_error": _compute_single_metric(X, X_reconstructed, metric),
        "sparsity_W": _compute_sparsity(W),
        "sparsity_H": _compute_sparsity(H)
    }

def _compute_sparsity(matrix: np.ndarray) -> float:
    """Compute sparsity of a matrix."""
    return 1 - (np.count_nonzero(matrix) / matrix.size)

################################################################################
# latent_features
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def latent_features_fit(
    X: np.ndarray,
    n_components: int = 2,
    init: str = 'random',
    solver: str = 'gradient_descent',
    beta_loss: str = 'frobenius',
    tol: float = 1e-4,
    max_iter: int = 200,
    random_state: Optional[int] = None,
    verbose: bool = False,
    normalize_factors: bool = True,
    regularization: Optional[str] = None,
    alpha: float = 0.0,
    l1_ratio: float = 0.5,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None,
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Compute Non-negative Matrix Factorization (NMF) to find latent features.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int, optional
        Number of latent features to extract (default=2)
    init : str, optional
        Initialization method ('random', 'nndsvd') (default='random')
    solver : str, optional
        Solver to use ('gradient_descent', 'mu', 'cd') (default='gradient_descent')
    beta_loss : str, optional
        Loss function ('frobenius', 'kullback-leibler') (default='frobenius')
    tol : float, optional
        Tolerance for stopping condition (default=1e-4)
    max_iter : int, optional
        Maximum number of iterations (default=200)
    random_state : Optional[int], optional
        Random seed for reproducibility (default=None)
    verbose : bool, optional
        Whether to print progress messages (default=False)
    normalize_factors : bool, optional
        Whether to normalize factors (default=True)
    regularization : Optional[str], optional
        Regularization type ('l1', 'l2', 'elasticnet') (default=None)
    alpha : float, optional
        Regularization strength (default=0.0)
    l1_ratio : float, optional
        ElasticNet mixing parameter (default=0.5)
    metric : str, optional
        Metric to compute ('mse', 'mae', 'r2') (default='mse')
    custom_metric : Optional[Callable], optional
        Custom metric function (default=None)

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': dict with 'W' and 'H' factors
        - 'metrics': computed metrics
        - 'params_used': parameters used
        - 'warnings': any warnings

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> result = latent_features_fit(X, n_components=3)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Set random seed if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Initialize factors
    W, H = _initialize_factors(X, n_components, init, rng)

    # Prepare output dictionary
    output = {
        'result': {'W': None, 'H': None},
        'metrics': {},
        'params_used': {
            'n_components': n_components,
            'init': init,
            'solver': solver,
            'beta_loss': beta_loss,
            'tol': tol,
            'max_iter': max_iter,
            'random_state': random_state,
            'normalize_factors': normalize_factors,
            'regularization': regularization,
            'alpha': alpha,
            'l1_ratio': l1_ratio,
        },
        'warnings': []
    }

    # Choose solver
    if solver == 'gradient_descent':
        W, H = _gradient_descent_solver(X, W, H, beta_loss, tol, max_iter,
                                      regularization, alpha, l1_ratio, verbose)
    elif solver == 'mu':
        W, H = _multiplicative_update_solver(X, W, H, beta_loss, tol, max_iter,
                                           regularization, alpha, l1_ratio, verbose)
    elif solver == 'cd':
        W, H = _coordinate_descent_solver(X, W, H, beta_loss, tol, max_iter,
                                        regularization, alpha, l1_ratio, verbose)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Normalize factors if requested
    if normalize_factors:
        W, H = _normalize_factors(W, H)

    # Store results
    output['result']['W'] = W
    output['result']['H'] = H

    # Compute metrics
    _compute_metrics(X, W, H, metric, custom_metric, output)

    return output

def _validate_inputs(X: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if X.shape[1] < n_components:
        raise ValueError("n_components cannot be greater than number of features")
    if np.any(np.isnan(X)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input data contains infinite values")

def _initialize_factors(
    X: np.ndarray,
    n_components: int,
    init: str,
    rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize W and H factors."""
    n_samples = X.shape[0]

    if init == 'random':
        W = rng.rand(n_samples, n_components)
        H = rng.rand(n_components, X.shape[1])
    elif init == 'nndsvd':
        W, H = _nndsvd_init(X, n_components)
    else:
        raise ValueError(f"Unknown initialization method: {init}")

    return W, H

def _nndsvd_init(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """NNDSVD initialization."""
    # Implementation of NNDSVD algorithm
    pass

def _gradient_descent_solver(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    beta_loss: str,
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    alpha: float,
    l1_ratio: float,
    verbose: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient descent solver for NMF."""
    # Implementation of gradient descent algorithm
    pass

def _multiplicative_update_solver(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    beta_loss: str,
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    alpha: float,
    l1_ratio: float,
    verbose: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Multiplicative update solver for NMF."""
    # Implementation of multiplicative updates algorithm
    pass

def _coordinate_descent_solver(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    beta_loss: str,
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    alpha: float,
    l1_ratio: float,
    verbose: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate descent solver for NMF."""
    # Implementation of coordinate descent algorithm
    pass

def _normalize_factors(W: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize factors."""
    # Implementation of factor normalization
    pass

def _compute_metrics(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable],
    output: Dict
) -> None:
    """Compute metrics for the NMF solution."""
    if custom_metric is not None:
        output['metrics']['custom'] = custom_metric(X, W @ H)
    else:
        if metric == 'mse':
            output['metrics']['mse'] = np.mean((X - W @ H) ** 2)
        elif metric == 'mae':
            output['metrics']['mae'] = np.mean(np.abs(X - W @ H))
        elif metric == 'r2':
            ss_res = np.sum((X - W @ H) ** 2)
            ss_tot = np.sum((X - np.mean(X)) ** 2)
            output['metrics']['r2'] = 1 - ss_res / ss_tot
        else:
            raise ValueError(f"Unknown metric: {metric}")

################################################################################
# reconstruction_error
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def reconstruction_error_fit(
    V: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    normalize: bool = False,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Calculate the reconstruction error for Non-negative Matrix Factorization (NMF).

    Parameters
    ----------
    V : np.ndarray
        The original matrix to be reconstructed.
    W : np.ndarray
        The basis matrix from NMF decomposition.
    H : np.ndarray
        The coefficient matrix from NMF decomposition.
    metric : str or callable, optional
        Metric to use for error calculation. Options: 'mse', 'mae', 'r2', or custom callable.
    distance : str or callable, optional
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine', or custom callable.
    normalize : bool, optional
        Whether to normalize the matrices before calculation.
    custom_metric : callable, optional
        Custom metric function if not using built-in options.
    **kwargs :
        Additional parameters for the chosen metric or distance.

    Returns
    -------
    Dict
        A dictionary containing:
        - 'result': The reconstruction error value.
        - 'metrics': Additional metrics if applicable.
        - 'params_used': Parameters used in the calculation.
        - 'warnings': Any warnings generated during computation.

    Examples
    --------
    >>> V = np.random.rand(10, 5)
    >>> W = np.random.rand(10, 3)
    >>> H = np.random.rand(3, 5)
    >>> error = reconstruction_error_fit(V, W, H, metric='mse')
    """
    # Validate inputs
    _validate_inputs(V, W, H)

    # Normalize if required
    if normalize:
        V = _normalize_matrix(V)
        W = _normalize_matrix(W)
        H = _normalize_matrix(H)

    # Compute reconstructed matrix
    V_reconstructed = np.dot(W, H)

    # Choose distance metric
    if isinstance(distance, str):
        distance_func = _get_distance_function(distance)
    else:
        distance_func = distance

    # Choose metric
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    # Calculate error
    error_value = _calculate_error(V, V_reconstructed, metric_func, distance_func, **kwargs)

    # Prepare output
    result_dict = {
        'result': error_value,
        'metrics': {},
        'params_used': {
            'metric': metric,
            'distance': distance,
            'normalize': normalize
        },
        'warnings': []
    }

    return result_dict

def _validate_inputs(V: np.ndarray, W: np.ndarray, H: np.ndarray) -> None:
    """Validate the input matrices."""
    if V.ndim != 2 or W.ndim != 2 or H.ndim != 2:
        raise ValueError("All inputs must be 2-dimensional arrays.")
    if V.shape[0] != W.shape[0]:
        raise ValueError("V and W must have the same number of rows.")
    if V.shape[1] != H.shape[1]:
        raise ValueError("V and H must have the same number of columns.")
    if W.shape[1] != H.shape[0]:
        raise ValueError("W and H must have compatible dimensions for matrix multiplication.")

def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize a matrix using min-max normalization."""
    min_val = np.nanmin(matrix)
    max_val = np.nanmax(matrix)
    if max_val == min_val:
        return matrix
    return (matrix - min_val) / (max_val - min_val)

def _get_distance_function(distance: str) -> Callable:
    """Get the distance function based on the input string."""
    distance_functions = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance
    }
    if distance not in distance_functions:
        raise ValueError(f"Unknown distance metric: {distance}")
    return distance_functions[distance]

def _get_metric_function(metric: str) -> Callable:
    """Get the metric function based on the input string."""
    metric_functions = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }
    if metric not in metric_functions:
        raise ValueError(f"Unknown metric: {metric}")
    return metric_functions[metric]

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two matrices."""
    return np.sqrt(np.sum((a - b) ** 2))

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Manhattan distance between two matrices."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine distance between two matrices."""
    dot_product = np.sum(a * b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1 - (dot_product / (norm_a * norm_b))

def _mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Mean Squared Error between two matrices."""
    return np.mean((a - b) ** 2)

def _mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Mean Absolute Error between two matrices."""
    return np.mean(np.abs(a - b))

def _r_squared(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate R-squared between two matrices."""
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1 - (ss_res / ss_tot)

def _calculate_error(
    V: np.ndarray,
    V_reconstructed: np.ndarray,
    metric_func: Callable,
    distance_func: Optional[Callable] = None,
    **kwargs
) -> float:
    """Calculate the reconstruction error using the specified metric and distance."""
    if distance_func is not None:
        distance = distance_func(V, V_reconstructed)
        return metric_func(distance, **kwargs)
    else:
        return metric_func(V, V_reconstructed, **kwargs)

################################################################################
# sparsity
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def sparsity_fit(
    W: np.ndarray,
    H: np.ndarray,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'l1_ratio',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute sparsity measures for NMF factors W and H.

    Parameters
    ----------
    W : np.ndarray
        Factor matrix of shape (n_samples, n_components)
    H : np.ndarray
        Factor matrix of shape (n_components, n_features)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Sparsity metric ('l1_ratio', 'nnz_ratio', 'entropy') or custom callable
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2')
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics

    Returns
    -------
    Dict containing:
        - 'result': computed sparsity measures
        - 'metrics': additional metrics if applicable
        - 'params_used': parameters actually used
        - 'warnings': any warnings generated

    Example
    -------
    >>> W = np.random.rand(10, 5)
    >>> H = np.random.rand(5, 8)
    >>> result = sparsity_fit(W, H, normalization='standard', metric='l1_ratio')
    """
    # Validate inputs
    _validate_inputs(W, H)

    # Normalize matrices if required
    W_norm, H_norm = _apply_normalization(W, H, normalization)

    # Choose solver
    if solver == 'closed_form':
        sparsity = _compute_sparsity_closed_form(W_norm, H_norm, metric, custom_metric)
    elif solver == 'gradient_descent':
        sparsity = _compute_sparsity_gradient_descent(
            W_norm, H_norm, metric, custom_metric,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute additional metrics if needed
    metrics = {}
    if metric in ['l1_ratio', 'nnz_ratio']:
        metrics['entropy'] = _compute_entropy(W_norm, H_norm)

    return {
        'result': sparsity,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

def _validate_inputs(W: np.ndarray, H: np.ndarray) -> None:
    """Validate input matrices."""
    if not isinstance(W, np.ndarray) or not isinstance(H, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if W.ndim != 2 or H.ndim != 2:
        raise ValueError("Inputs must be 2-dimensional")
    if W.shape[0] != H.shape[0]:
        raise ValueError("Incompatible matrix dimensions")
    if np.any(np.isnan(W)) or np.any(np.isinf(W)):
        raise ValueError("W contains NaN or infinite values")
    if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        raise ValueError("H contains NaN or infinite values")

def _apply_normalization(
    W: np.ndarray,
    H: np.ndarray,
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalization to matrices."""
    if method == 'none':
        return W, H
    elif method == 'standard':
        W_norm = (W - np.mean(W)) / np.std(W)
        H_norm = (H - np.mean(H)) / np.std(H)
    elif method == 'minmax':
        W_norm = (W - np.min(W)) / (np.max(W) - np.min(W))
        H_norm = (H - np.min(H)) / (np.max(H) - np.min(H))
    elif method == 'robust':
        W_norm = (W - np.median(W)) / (np.percentile(W, 75) - np.percentile(W, 25))
        H_norm = (H - np.median(H)) / (np.percentile(H, 75) - np.percentile(H, 25))
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return W_norm, H_norm

def _compute_sparsity_closed_form(
    W: np.ndarray,
    H: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute sparsity using closed-form solution."""
    if callable(metric):
        return {'W': metric(W), 'H': metric(H)}
    elif custom_metric is not None:
        return {'W': custom_metric(W), 'H': custom_metric(H)}
    elif metric == 'l1_ratio':
        return {
            'W': _compute_l1_ratio(W),
            'H': _compute_l1_ratio(H)
        }
    elif metric == 'nnz_ratio':
        return {
            'W': _compute_nnz_ratio(W),
            'H': _compute_nnz_ratio(H)
        }
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_sparsity_gradient_descent(
    W: np.ndarray,
    H: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None,
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict:
    """Compute sparsity using gradient descent."""
    W_current = W.copy()
    H_current = H.copy()

    for _ in range(max_iter):
        # Update W and H (simplified example)
        if regularization == 'l1':
            W_current = np.maximum(0, W_current - tol)
        elif regularization == 'l2':
            W_current = np.clip(W_current, -1, 1)

        # Check convergence
        if _check_convergence(W_current, H_current, tol):
            break

    return _compute_sparsity_closed_form(W_current, H_current, metric, custom_metric)

def _compute_l1_ratio(X: np.ndarray) -> float:
    """Compute L1 ratio sparsity measure."""
    return np.sum(np.abs(X)) / (X.shape[0] * X.shape[1])

def _compute_nnz_ratio(X: np.ndarray) -> float:
    """Compute non-zero ratio sparsity measure."""
    return np.count_nonzero(X) / (X.shape[0] * X.shape[1])

def _compute_entropy(X: np.ndarray) -> float:
    """Compute entropy sparsity measure."""
    prob = X / np.sum(X)
    return -np.sum(prob * np.log(prob + 1e-10))

def _check_convergence(W: np.ndarray, H: np.ndarray, tol: float) -> bool:
    """Check if matrices have converged."""
    return np.max(np.abs(W)) < tol and np.max(np.abs(H)) < tol

################################################################################
# regularization
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Literal

def regularization_fit(
    X: np.ndarray,
    W_init: Optional[np.ndarray] = None,
    H_init: Optional[np.ndarray] = None,
    n_components: int = 10,
    max_iter: int = 200,
    tol: float = 1e-4,
    solver: Literal['cd', 'mu'] = 'cd',
    beta_loss: Union[Literal['frobenius', 'kullback-leibler'], float] = 'frobenius',
    regularization: Literal['none', 'l1', 'l2', 'elasticnet'] = 'none',
    alpha: float = 0.0,
    l1_ratio: float = 0.5,
    normalize_factors: bool = False,
    init: Literal['random', 'nndsvd'] = 'nndsvd',
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict:
    """
    Perform Non-negative Matrix Factorization (NMF) with regularization.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    W_init : Optional[np.ndarray]
        Initial guess for the W matrix
    H_init : Optional[np.ndarray]
        Initial guess for the H matrix
    n_components : int, default=10
        Number of components in the decomposition
    max_iter : int, default=200
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for stopping condition
    solver : {'cd', 'mu'}, default='cd'
        Solver to use ('cd' for coordinate descent, 'mu' for multiplicative update)
    beta_loss : {'frobenius', 'kullback-leibler'} or float, default='frobenius'
        Beta divergence to minimize
    regularization : {'none', 'l1', 'l2', 'elasticnet'}, default='none'
        Type of regularization to apply
    alpha : float, default=0.0
        Regularization strength
    l1_ratio : float, default=0.5
        Ratio of L1 regularization in elasticnet
    normalize_factors : bool, default=False
        Whether to normalize the factors
    init : {'random', 'nndsvd'}, default='nndsvd'
        Initialization method
    random_state : Optional[int], default=None
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print progress messages

    Returns
    -------
    Dict containing:
        - 'result': dict with 'W' and 'H' matrices
        - 'metrics': dict of computed metrics
        - 'params_used': dict of parameters actually used
        - 'warnings': list of warnings encountered

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> result = regularization_fit(X, n_components=3, solver='cd', regularization='l1')
    """
    # Validate inputs
    _validate_inputs(X, W_init, H_init, n_components)

    # Set random seed if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Initialize factors
    W, H = _initialize_factors(X.shape[0], X.shape[1], n_components,
                              W_init, H_init, init, rng)

    # Prepare output dictionary
    result = {
        'result': {'W': W, 'H': H},
        'metrics': {},
        'params_used': {
            'n_components': n_components,
            'max_iter': max_iter,
            'tol': tol,
            'solver': solver,
            'beta_loss': beta_loss,
            'regularization': regularization,
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'normalize_factors': normalize_factors,
            'init': init
        },
        'warnings': []
    }

    # Main optimization loop
    for i in range(max_iter):
        prev_W, prev_H = W.copy(), H.copy()

        if solver == 'cd':
            W, H = _coordinate_descent(X, W, H, beta_loss, regularization,
                                      alpha, l1_ratio, normalize_factors)
        else:  # mu
            W, H = _multiplicative_update(X, W, H, beta_loss, regularization,
                                         alpha, l1_ratio)

        # Check convergence
        if _check_convergence(W, prev_W, H, prev_H, tol):
            if verbose:
                print(f"Converged at iteration {i}")
            break

    # Compute final metrics
    result['metrics'] = _compute_metrics(X, W @ H, beta_loss)

    return result

def _validate_inputs(
    X: np.ndarray,
    W_init: Optional[np.ndarray],
    H_init: Optional[np.ndarray],
    n_components: int
) -> None:
    """Validate input matrices and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")

    n_samples, n_features = X.shape
    if W_init is not None:
        if W_init.shape != (n_samples, n_components):
            raise ValueError(f"W_init must have shape ({n_samples}, {n_components})")
    if H_init is not None:
        if H_init.shape != (n_components, n_features):
            raise ValueError(f"H_init must have shape ({n_components}, {n_features})")
    if n_components <= 0:
        raise ValueError("n_components must be positive")

def _initialize_factors(
    n_samples: int,
    n_features: int,
    n_components: int,
    W_init: Optional[np.ndarray],
    H_init: Optional[np.ndarray],
    init: str,
    rng: np.random.RandomState
) -> tuple:
    """Initialize W and H matrices."""
    if W_init is not None and H_init is not None:
        return W_init, H_init

    if init == 'random':
        W = rng.rand(n_samples, n_components)
        H = rng.rand(n_components, n_features)
    else:  # nndsvd
        W = _nndsvd_init(n_samples, n_components, rng)
        H = _nndsvd_init(n_features, n_components, rng)

    return W, H

def _nndsvd_init(n: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    """Non-negative double singular value decomposition initialization."""
    # Implementation of NNDSVD would go here
    return rng.rand(n, k)

def _coordinate_descent(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    beta_loss: Union[str, float],
    regularization: str,
    alpha: float,
    l1_ratio: float,
    normalize_factors: bool
) -> tuple:
    """Coordinate descent solver for NMF."""
    # Implementation of coordinate descent would go here
    return W, H

def _multiplicative_update(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    beta_loss: Union[str, float],
    regularization: str,
    alpha: float,
    l1_ratio: float
) -> tuple:
    """Multiplicative update solver for NMF."""
    # Implementation of multiplicative updates would go here
    return W, H

def _check_convergence(
    W_new: np.ndarray,
    W_prev: np.ndarray,
    H_new: np.ndarray,
    H_prev: np.ndarray,
    tol: float
) -> bool:
    """Check if the algorithm has converged."""
    delta_W = np.linalg.norm(W_new - W_prev, 'fro') / max(np.linalg.norm(W_prev, 'fro'), 1e-6)
    delta_H = np.linalg.norm(H_new - H_prev, 'fro') / max(np.linalg.norm(H_prev, 'fro'), 1e-6)
    return delta_W < tol and delta_H < tol

def _compute_metrics(
    X: np.ndarray,
    X_reconstructed: np.ndarray,
    beta_loss: Union[str, float]
) -> Dict:
    """Compute reconstruction metrics."""
    metrics = {}

    if beta_loss == 'frobenius':
        metrics['mse'] = np.mean((X - X_reconstructed) ** 2)
        metrics['r2'] = 1 - metrics['mse'] / np.var(X)
    elif beta_loss == 'kullback-leibler':
        metrics['kl_divergence'] = np.sum(X * (np.log(X) - np.log(X_reconstructed + 1e-10)))
    elif isinstance(beta_loss, (int, float)):
        # General beta divergence
        pass

    return metrics

################################################################################
# initialization_methods
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def initialization_methods_fit(
    X: np.ndarray,
    n_components: int,
    init_type: str = 'random',
    normalization: Optional[str] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'frobenius',
    solver: str = 'nndsvd',
    random_state: Optional[int] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Initialize matrices for Non-negative Matrix Factorization (NMF).

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int
        Number of components.
    init_type : str, optional
        Initialization type ('random', 'nndsvd', 'custom').
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax').
    metric : str or callable, optional
        Metric to evaluate initialization quality.
    solver : str, optional
        Solver method for initialization ('nndsvd', 'random').
    random_state : int, optional
        Random seed for reproducibility.
    **kwargs :
        Additional solver-specific parameters.

    Returns
    -------
    dict
        Dictionary containing:
        - 'W': Initial W matrix.
        - 'H': Initial H matrix.
        - 'metrics': Dictionary of metrics.
        - 'params_used': Parameters used for initialization.

    Examples
    --------
    >>> X = np.random.rand(10, 5)
    >>> result = initialization_methods_fit(X, n_components=3)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Set random state
    rng = np.random.RandomState(random_state)

    # Normalize data if specified
    X_normalized, norm_params = _apply_normalization(X, normalization)

    # Initialize W and H based on init_type
    if init_type == 'random':
        W, H = _random_initialization(X_normalized.shape[0], n_components, rng)
    elif init_type == 'nndsvd':
        W, H = _nndsvd_initialization(X_normalized, n_components)
    elif init_type == 'custom':
        W, H = _custom_initialization(X_normalized.shape[0], n_components, rng)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    # Compute metrics
    metrics = _compute_metrics(W, H, X_normalized, metric)

    # Prepare output
    result = {
        'W': W,
        'H': H,
        'metrics': metrics,
        'params_used': {
            'init_type': init_type,
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
            'solver': solver,
            'random_state': random_state
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
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if X.shape[1] < n_components:
        raise ValueError("n_components cannot be greater than the number of features")

def _apply_normalization(X: np.ndarray, method: Optional[str]) -> tuple:
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

def _random_initialization(n_samples: int, n_components: int, rng: np.random.RandomState) -> tuple:
    """Random initialization of W and H matrices."""
    W = rng.rand(n_samples, n_components)
    H = rng.rand(n_components, n_samples)
    return W, H

def _nndsvd_initialization(X: np.ndarray, n_components: int) -> tuple:
    """NNDSVD initialization of W and H matrices."""
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    W = np.abs(U[:, :n_components]) * np.sqrt(S[:n_components])
    H = np.abs(Vt[:n_components, :]) * np.sqrt(S[:n_components])
    return W, H

def _custom_initialization(n_samples: int, n_components: int, rng: np.random.RandomState) -> tuple:
    """Custom initialization of W and H matrices."""
    raise NotImplementedError("Custom initialization not yet implemented")

def _compute_metrics(W: np.ndarray, H: np.ndarray, X: np.ndarray, metric: Union[str, Callable]) -> Dict:
    """Compute metrics for initialization quality."""
    reconstruction = W @ H
    metrics_dict = {}

    if metric == 'frobenius':
        error = np.linalg.norm(X - reconstruction, 'fro')
        metrics_dict['frobenius_norm'] = error
    elif callable(metric):
        metrics_dict['custom_metric'] = metric(X, reconstruction)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics_dict

################################################################################
# convergence_criteria
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def convergence_criteria_fit(
    W: np.ndarray,
    H: np.ndarray,
    V: np.ndarray,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: str = 'none',
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Compute convergence criteria for NMF.

    Parameters:
    -----------
    W : np.ndarray
        Factor matrix of shape (n_components, n_features)
    H : np.ndarray
        Factor matrix of shape (n_components, n_samples)
    V : np.ndarray
        Data matrix to approximate of shape (n_features, n_samples)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric to use ('mse', 'mae', 'r2', 'logloss') or custom callable
    distance : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    regularization : str, optional
        Regularization method ('none', 'l1', 'l2', 'elasticnet')
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
    >>> W = np.random.rand(5, 10)
    >>> H = np.random.rand(5, 20)
    >>> V = W @ H
    >>> result = convergence_criteria_fit(W, H, V)
    """
    # Validate inputs
    _validate_inputs(W, H, V)

    # Normalize data if required
    W_norm, H_norm = _apply_normalization(W, H, normalization)

    # Initialize parameters
    params_used = {
        'normalization': normalization,
        'metric': metric if isinstance(metric, str) else 'custom',
        'distance': distance if isinstance(distance, str) else 'custom',
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Compute convergence criteria based on solver choice
    if solver == 'closed_form':
        result, metrics = _compute_closed_form(W_norm, H_norm, V)
    elif solver == 'gradient_descent':
        result, metrics = _compute_gradient_descent(W_norm, H_norm, V, tol, max_iter)
    elif solver == 'newton':
        result, metrics = _compute_newton(W_norm, H_norm, V, tol, max_iter)
    elif solver == 'coordinate_descent':
        result, metrics = _compute_coordinate_descent(W_norm, H_norm, V, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    if isinstance(distance, str):
        distance_func = _get_distance_function(distance)
    else:
        distance_func = distance

    metrics.update({
        'final_metric': metric_func(W_norm @ H_norm, V),
        'distance': distance_func(W_norm @ H_norm, V)
    })

    # Check for warnings
    warnings = _check_warnings(W_norm @ H_norm, V)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(W: np.ndarray, H: np.ndarray, V: np.ndarray) -> None:
    """Validate input matrices dimensions and types."""
    if not (isinstance(W, np.ndarray) and isinstance(H, np.ndarray) and isinstance(V, np.ndarray)):
        raise TypeError("Inputs must be numpy arrays")
    if W.ndim != 2 or H.ndim != 2 or V.ndim != 2:
        raise ValueError("Inputs must be 2-dimensional arrays")
    if W.shape[0] != H.shape[0]:
        raise ValueError("Number of components in W and H must match")
    if V.shape != (W.shape[1], H.shape[1]):
        raise ValueError("V dimensions must match W and H product")
    if np.any(np.isnan(W)) or np.any(np.isnan(H)) or np.any(np.isnan(V)):
        raise ValueError("Input arrays contain NaN values")
    if np.any(np.isinf(W)) or np.any(np.isinf(H)) or np.any(np.isinf(V)):
        raise ValueError("Input arrays contain infinite values")

def _apply_normalization(W: np.ndarray, H: np.ndarray, method: str) -> tuple:
    """Apply normalization to input matrices."""
    W_norm = W.copy()
    H_norm = H.copy()

    if method == 'standard':
        W_norm = (W - np.mean(W, axis=0)) / np.std(W, axis=0)
        H_norm = (H - np.mean(H, axis=0)) / np.std(H, axis=0)
    elif method == 'minmax':
        W_norm = (W - np.min(W, axis=0)) / (np.max(W, axis=0) - np.min(W, axis=0))
        H_norm = (H - np.min(H, axis=0)) / (np.max(H, axis=0) - np.min(H, axis=0))
    elif method == 'robust':
        W_norm = (W - np.median(W, axis=0)) / (np.percentile(W, 75, axis=0) - np.percentile(W, 25, axis=0))
        H_norm = (H - np.median(H, axis=0)) / (np.percentile(H, 75, axis=0) - np.percentile(H, 25, axis=0))
    elif method != 'none':
        raise ValueError(f"Unknown normalization method: {method}")

    return W_norm, H_norm

def _get_metric_function(metric_name: str) -> Callable:
    """Get metric function based on name."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared,
        'logloss': _log_loss
    }
    if metric_name not in metrics:
        raise ValueError(f"Unknown metric: {metric_name}")
    return metrics[metric_name]

def _get_distance_function(distance_name: str) -> Callable:
    """Get distance function based on name."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': _minkowski_distance
    }
    if distance_name not in distances:
        raise ValueError(f"Unknown distance: {distance_name}")
    return distances[distance_name]

def _compute_closed_form(W: np.ndarray, H: np.ndarray, V: np.ndarray) -> tuple:
    """Compute convergence criteria using closed form solution."""
    # Placeholder for actual implementation
    result = W @ H
    metrics = {
        'iterations': 0,
        'convergence_time': 0.0
    }
    return result, metrics

def _compute_gradient_descent(W: np.ndarray, H: np.ndarray, V: np.ndarray,
                             tol: float, max_iter: int) -> tuple:
    """Compute convergence criteria using gradient descent."""
    # Placeholder for actual implementation
    result = W @ H
    metrics = {
        'iterations': max_iter,
        'convergence_time': 0.1
    }
    return result, metrics

def _compute_newton(W: np.ndarray, H: np.ndarray, V: np.ndarray,
                    tol: float, max_iter: int) -> tuple:
    """Compute convergence criteria using Newton's method."""
    # Placeholder for actual implementation
    result = W @ H
    metrics = {
        'iterations': max_iter // 2,
        'convergence_time': 0.05
    }
    return result, metrics

def _compute_coordinate_descent(W: np.ndarray, H: np.ndarray, V: np.ndarray,
                               tol: float, max_iter: int) -> tuple:
    """Compute convergence criteria using coordinate descent."""
    # Placeholder for actual implementation
    result = W @ H
    metrics = {
        'iterations': max_iter // 4,
        'convergence_time': 0.2
    }
    return result, metrics

def _mean_squared_error(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((X - Y) ** 2)

def _mean_absolute_error(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(X - Y))

def _r_squared(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((Y - X) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    return 1 - (ss_res / ss_tot)

def _log_loss(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute log loss."""
    epsilon = 1e-15
    X = np.clip(X, epsilon, 1 - epsilon)
    return -np.mean(Y * np.log(X) + (1 - Y) * np.log(1 - X))

def _euclidean_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Euclidean distance."""
    return np.linalg.norm(X - Y)

def _manhattan_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Manhattan distance."""
    return np.sum(np.abs(X - Y))

def _cosine_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute cosine distance."""
    return 1 - (X @ Y.T) / (np.linalg.norm(X) * np.linalg.norm(Y))

def _minkowski_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Minkowski distance."""
    return np.sum(np.abs(X - Y) ** 3) ** (1/3)

def _check_warnings(X: np.ndarray, Y: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isnan(X)):
        warnings.append("Result contains NaN values")
    if np.any(np.isinf(X)):
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
    normalizer: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 200,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Perform dimensionality reduction using Non-negative Matrix Factorization (NMF).

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of components to keep, by default 2.
    normalizer : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), by default 'standard'.
    metric : Union[str, Callable], optional
        Metric to optimize ('mse', 'mae', 'r2', 'logloss'), by default 'mse'.
    distance : Union[str, Callable], optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski'), by default 'euclidean'.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent'), by default 'gradient_descent'.
    regularization : Optional[str], optional
        Regularization method ('none', 'l1', 'l2', 'elasticnet'), by default None.
    tol : float, optional
        Tolerance for stopping criterion, by default 1e-4.
    max_iter : int, optional
        Maximum number of iterations, by default 200.
    custom_metric : Optional[Callable], optional
        Custom metric function, by default None.
    custom_distance : Optional[Callable], optional
        Custom distance function, by default None.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Reduced data matrix of shape (n_samples, n_components)
        - 'metrics': Dictionary of computed metrics
        - 'params_used': Dictionary of parameters used
        - 'warnings': List of warnings encountered

    Examples
    --------
    >>> X = np.random.rand(100, 50)
    >>> result = dimensionality_reduction_fit(X, n_components=2)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Normalize data
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize parameters
    W, H = _initialize_parameters(X_normalized.shape[0], X_normalized.shape[1], n_components)

    # Choose solver
    if solver == 'closed_form':
        W, H = _solve_closed_form(X_normalized, n_components)
    elif solver == 'gradient_descent':
        W, H = _solve_gradient_descent(X_normalized, n_components, metric, distance,
                                      regularization, tol, max_iter)
    elif solver == 'newton':
        W, H = _solve_newton(X_normalized, n_components, metric, distance,
                            regularization, tol, max_iter)
    elif solver == 'coordinate_descent':
        W, H = _solve_coordinate_descent(X_normalized, n_components, metric, distance,
                                        regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, W @ H, metric, custom_metric)

    # Prepare output
    result = {
        'result': W @ H,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'normalizer': normalizer,
            'metric': metric,
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
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if X.shape[1] < n_components:
        raise ValueError("n_components cannot be greater than the number of features")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to the input data."""
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

def _initialize_parameters(n_samples: int, n_features: int, n_components: int) -> tuple:
    """Initialize W and H matrices."""
    np.random.seed(42)
    W = np.abs(np.random.randn(n_samples, n_components))
    H = np.abs(np.random.randn(n_components, n_features))
    return W, H

def _solve_closed_form(X: np.ndarray, n_components: int) -> tuple:
    """Solve NMF using closed-form solution."""
    # This is a placeholder for the actual implementation
    return np.zeros((X.shape[0], n_components)), np.zeros((n_components, X.shape[1]))

def _solve_gradient_descent(
    X: np.ndarray,
    n_components: int,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple:
    """Solve NMF using gradient descent."""
    # This is a placeholder for the actual implementation
    return np.zeros((X.shape[0], n_components)), np.zeros((n_components, X.shape[1]))

def _solve_newton(
    X: np.ndarray,
    n_components: int,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple:
    """Solve NMF using Newton's method."""
    # This is a placeholder for the actual implementation
    return np.zeros((X.shape[0], n_components)), np.zeros((n_components, X.shape[1]))

def _solve_coordinate_descent(
    X: np.ndarray,
    n_components: int,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple:
    """Solve NMF using coordinate descent."""
    # This is a placeholder for the actual implementation
    return np.zeros((X.shape[0], n_components)), np.zeros((n_components, X.shape[1]))

def _compute_metrics(
    X: np.ndarray,
    X_reconstructed: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute metrics for the reconstruction."""
    metrics = {}

    if custom_metric is not None:
        metrics['custom'] = custom_metric(X, X_reconstructed)
    else:
        if metric == 'mse':
            metrics['mse'] = np.mean((X - X_reconstructed) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(X - X_reconstructed))
        elif metric == 'r2':
            ss_res = np.sum((X - X_reconstructed) ** 2)
            ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
        elif metric == 'logloss':
            epsilon = 1e-15
            metrics['logloss'] = -np.mean(X * np.log(X_reconstructed + epsilon) +
                                         (1 - X) * np.log(1 - X_reconstructed + epsilon))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return metrics

################################################################################
# topic_modeling
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def topic_modeling_fit(
    X: np.ndarray,
    n_topics: int,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 200,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Perform Non-Negative Matrix Factorization (NMF) for topic modeling.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_topics : int
        Number of topics to extract
    normalizer : Optional[Callable]
        Function to normalize the input data. Default is None.
    metric : Union[str, Callable]
        Metric to evaluate the model. Can be 'mse', 'mae', 'r2', or a custom callable.
    distance : Union[str, Callable]
        Distance metric for the solver. Can be 'euclidean', 'manhattan', 'cosine', or a custom callable.
    solver : str
        Solver to use. Options are 'gradient_descent', 'coordinate_descent'.
    regularization : Optional[str]
        Regularization type. Options are 'l1', 'l2', or None.
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    random_state : Optional[int]
        Random seed for reproducibility.
    custom_metric : Optional[Callable]
        Custom metric function if needed.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Example:
    --------
    >>> X = np.random.rand(100, 50)
    >>> result = topic_modeling_fit(X, n_topics=10)
    """
    # Validate inputs
    _validate_inputs(X, n_topics)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    if normalizer is not None:
        X = normalizer(X)

    # Initialize W and H matrices
    W, H = _initialize_matrices(X.shape[0], X.shape[1], n_topics)

    # Select solver
    if solver == 'gradient_descent':
        W, H = _gradient_descent_solver(X, W, H, n_topics, distance, regularization, tol, max_iter)
    elif solver == 'coordinate_descent':
        W, H = _coordinate_descent_solver(X, W, H, n_topics, distance, regularization, tol, max_iter)
    else:
        raise ValueError("Unsupported solver. Choose 'gradient_descent' or 'coordinate_descent'.")

    # Compute metrics
    metrics = _compute_metrics(X, W @ H, metric, custom_metric)

    # Prepare output
    result = {
        'W': W,
        'H': H,
        'metrics': metrics,
        'params_used': {
            'n_topics': n_topics,
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, n_topics: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if n_topics <= 0:
        raise ValueError("n_topics must be a positive integer.")
    if n_topics > min(X.shape):
        raise ValueError("n_topics cannot be greater than the smaller dimension of X.")

def _initialize_matrices(n_samples: int, n_features: int, n_topics: int) -> tuple:
    """Initialize W and H matrices with random non-negative values."""
    W = np.abs(np.random.randn(n_samples, n_topics))
    H = np.abs(np.random.randn(n_topics, n_features))
    return W, H

def _gradient_descent_solver(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    n_topics: int,
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple:
    """Perform NMF using gradient descent."""
    for _ in range(max_iter):
        # Update W and H matrices
        grad_W = _compute_gradient(W, H, X, distance)
        grad_H = _compute_gradient(H.T, W.T, X.T, distance).T

        if regularization == 'l1':
            grad_W += np.sign(W)
            grad_H += np.sign(H)
        elif regularization == 'l2':
            grad_W += 2 * W
            grad_H += 2 * H

        W -= tol * grad_W
        H -= tol * grad_H

        # Ensure non-negativity
        W = np.maximum(W, 0)
        H = np.maximum(H, 0)

    return W, H

def _coordinate_descent_solver(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    n_topics: int,
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple:
    """Perform NMF using coordinate descent."""
    for _ in range(max_iter):
        # Update W and H matrices
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i, j] = _update_coordinate(W, H, X, i, j, distance, regularization)

        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                H[i, j] = _update_coordinate(H.T, W.T, X.T, i, j, distance, regularization).T

        # Ensure non-negativity
        W = np.maximum(W, 0)
        H = np.maximum(H, 0)

    return W, H

def _compute_gradient(
    matrix: np.ndarray,
    other_matrix: np.ndarray,
    X: np.ndarray,
    distance: Union[str, Callable]
) -> np.ndarray:
    """Compute gradient based on the chosen distance metric."""
    if callable(distance):
        return distance(matrix, other_matrix, X)
    elif distance == 'euclidean':
        return 2 * (matrix @ other_matrix - X) @ other_matrix.T
    elif distance == 'manhattan':
        return np.sign(matrix @ other_matrix - X) @ other_matrix.T
    elif distance == 'cosine':
        return (matrix @ other_matrix - X) @ other_matrix.T / np.linalg.norm(matrix @ other_matrix, axis=1)[:, None]
    else:
        raise ValueError("Unsupported distance metric.")

def _update_coordinate(
    matrix: np.ndarray,
    other_matrix: np.ndarray,
    X: np.ndarray,
    i: int,
    j: int,
    distance: Union[str, Callable],
    regularization: Optional[str]
) -> float:
    """Update a single coordinate in the matrix."""
    if callable(distance):
        grad = distance(matrix, other_matrix, X)
    elif distance == 'euclidean':
        grad = 2 * (matrix @ other_matrix - X)[:, j]
    elif distance == 'manhattan':
        grad = np.sign(matrix @ other_matrix - X)[:, j]
    elif distance == 'cosine':
        grad = (matrix @ other_matrix - X)[:, j] / np.linalg.norm(matrix @ other_matrix, axis=1)

    if regularization == 'l1':
        grad += np.sign(matrix[i, j])
    elif regularization == 'l2':
        grad += 2 * matrix[i, j]

    return np.maximum(0, -grad)

def _compute_metrics(
    X: np.ndarray,
    X_reconstructed: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute metrics based on the chosen metric."""
    metrics = {}

    if custom_metric is not None:
        metrics['custom'] = custom_metric(X, X_reconstructed)
    elif metric == 'mse':
        metrics['mse'] = np.mean((X - X_reconstructed) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(X - X_reconstructed))
    elif metric == 'r2':
        ss_res = np.sum((X - X_reconstructed) ** 2)
        ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
        metrics['r2'] = 1 - ss_res / ss_tot
    elif metric == 'logloss':
        epsilon = 1e-10
        metrics['logloss'] = -np.mean(X * np.log(X_reconstructed + epsilon) - (1 - X) * np.log(1 - X_reconstructed + epsilon))
    else:
        raise ValueError("Unsupported metric.")

    return metrics

################################################################################
# feature_extraction
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def feature_extraction_fit(
    X: np.ndarray,
    n_components: int,
    *,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 200,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Perform Non-negative Matrix Factorization (NMF) for feature extraction.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int
        Number of components to extract
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric to evaluate the reconstruction error ('mse', 'mae', 'r2')
    distance : str or callable, optional
        Distance metric for similarity ('euclidean', 'manhattan', 'cosine')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'coordinate_descent')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    tol : float, optional
        Tolerance for stopping criterion
    max_iter : int, optional
        Maximum number of iterations
    random_state : int, optional
        Random seed for reproducibility
    custom_metric : callable, optional
        Custom metric function if needed
    custom_distance : callable, optional
        Custom distance function if needed

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': dict with factors W and H
        - 'metrics': dict of computed metrics
        - 'params_used': dict of parameters used
        - 'warnings': list of warnings

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> result = feature_extraction_fit(X, n_components=3)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Initialize parameters
    params_used = {
        'normalization': normalization,
        'metric': metric if isinstance(metric, str) else 'custom',
        'distance': distance if isinstance(distance, str) else 'custom',
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Normalize data if needed
    X_norm, norm_params = _apply_normalization(X, normalization)

    # Initialize factors
    W, H = _initialize_factors(X_norm.shape[0], X_norm.shape[1], n_components)

    # Choose solver
    if solver == 'closed_form':
        W, H = _solve_closed_form(X_norm, n_components)
    elif solver == 'gradient_descent':
        W, H = _solve_gradient_descent(X_norm, n_components,
                                      tol=tol, max_iter=max_iter)
    elif solver == 'coordinate_descent':
        W, H = _solve_coordinate_descent(X_norm, n_components,
                                        tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if needed
    if regularization is not None:
        W, H = _apply_regularization(W, H, X_norm, regularization)

    # Compute metrics
    metrics = _compute_metrics(X_norm, W, H,
                              metric=metric,
                              distance=distance,
                              custom_metric=custom_metric,
                              custom_distance=custom_distance)

    # Prepare result
    result = {
        'W': W,
        'H': H,
        'norm_params': norm_params
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
        raise ValueError("X must be a 2D array")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")

def _apply_normalization(
    X: np.ndarray,
    method: str
) -> tuple[np.ndarray, Optional[Dict]]:
    """Apply normalization to the input data."""
    norm_params = None

    if method == 'standard':
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        norm_params = {'mean': mean, 'std': std}
        X_norm = (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)
        norm_params = {'min': min_val, 'max': max_val}
        X_norm = (X - min_val) / ((max_val - min_val + 1e-8))
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        norm_params = {'median': median, 'iqr': iqr}
        X_norm = (X - median) / (iqr + 1e-8)
    else:
        X_norm = X.copy()

    return X_norm, norm_params

def _initialize_factors(
    n_samples: int,
    n_features: int,
    n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize NMF factors with random non-negative values."""
    W = np.abs(np.random.randn(n_samples, n_components))
    H = np.abs(np.random.randn(n_components, n_features))
    return W, H

def _solve_closed_form(
    X: np.ndarray,
    n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Solve NMF using closed-form solution."""
    # This is a placeholder - actual implementation would use SVD or similar
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    W = U[:, :n_components] @ np.diag(S[:n_components])
    H = Vt[:n_components, :]
    return np.abs(W), np.abs(H)

def _solve_gradient_descent(
    X: np.ndarray,
    n_components: int,
    *,
    tol: float = 1e-4,
    max_iter: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Solve NMF using gradient descent."""
    n_samples, n_features = X.shape
    W, H = _initialize_factors(n_samples, n_features, n_components)

    for _ in range(max_iter):
        # Update H
        numerator = W.T @ X
        denominator = W.T @ W @ H + 1e-8
        H *= numerator / denominator

        # Update W
        numerator = X @ H.T
        denominator = W @ H @ H.T + 1e-8
        W *= numerator / denominator

        # Check convergence
        if np.linalg.norm(X - W @ H) < tol:
            break

    return W, H

def _solve_coordinate_descent(
    X: np.ndarray,
    n_components: int,
    *,
    tol: float = 1e-4,
    max_iter: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Solve NMF using coordinate descent."""
    n_samples, n_features = X.shape
    W, H = _initialize_factors(n_samples, n_features, n_components)

    for _ in range(max_iter):
        # Update H
        for j in range(n_components):
            numerator = X.T @ W[:, j:j+1]
            denominator = W.T @ W[:, j:j+1] @ H[j:j+1, :] + 1e-8
            H[j:j+1, :] *= numerator / denominator

        # Update W
        for i in range(n_components):
            numerator = X @ H[i:i+1, :]
            denominator = W[:, i:i+1].T @ W[:, i:i+1] @ H[i:i+1, :] + 1e-8
            W[:, i:i+1] *= numerator / denominator

        # Check convergence
        if np.linalg.norm(X - W @ H) < tol:
            break

    return W, H

def _apply_regularization(
    W: np.ndarray,
    H: np.ndarray,
    X: np.ndarray,
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply regularization to NMF factors."""
    if method == 'l1':
        W -= 0.1 * np.sign(W)
        H -= 0.1 * np.sign(H)
    elif method == 'l2':
        W *= (1 - 0.1 * np.abs(W))
        H *= (1 - 0.1 * np.abs(H))
    elif method == 'elasticnet':
        W -= 0.05 * np.sign(W)
        W *= (1 - 0.05 * np.abs(W))
        H -= 0.05 * np.sign(H)
        H *= (1 - 0.05 * np.abs(H))
    return W, H

def _compute_metrics(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    *,
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """Compute metrics for NMF results."""
    X_reconstructed = W @ H

    if custom_metric is not None:
        metric_value = custom_metric(X, X_reconstructed)
    elif isinstance(metric, str):
        if metric == 'mse':
            metric_value = np.mean((X - X_reconstructed) ** 2)
        elif metric == 'mae':
            metric_value = np.mean(np.abs(X - X_reconstructed))
        elif metric == 'r2':
            ss_res = np.sum((X - X_reconstructed) ** 2)
            ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
            metric_value = 1 - ss_res / (ss_tot + 1e-8)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        metric_value = 0

    if custom_distance is not None:
        distance_value = custom_distance(X, X_reconstructed)
    elif isinstance(distance, str):
        if distance == 'euclidean':
            distance_value = np.linalg.norm(X - X_reconstructed, axis=1).mean()
        elif distance == 'manhattan':
            distance_value = np.sum(np.abs(X - X_reconstructed), axis=1).mean()
        elif distance == 'cosine':
            norm_X = np.linalg.norm(X, axis=1)
            norm_recon = np.linalg.norm(X_reconstructed, axis=1)
            dot_products = (X * X_reconstructed).sum(axis=1)
            distance_value = 1 - dot_products / (norm_X * norm_recon + 1e-8)
        elif distance == 'minkowski':
            distance_value = np.mean(np.sum(np.abs(X - X_reconstructed) ** 3, axis=1) ** (1/3))
        else:
            raise ValueError(f"Unknown distance: {distance}")
    else:
        distance_value = 0

    return {
        'reconstruction_error': metric_value,
        'distance_metric': distance_value
    }

################################################################################
# alternating_least_squares
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def alternating_least_squares_fit(
    V: np.ndarray,
    n_components: int,
    max_iter: int = 1000,
    tol: float = 1e-6,
    normalize: str = 'none',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    alpha: float = 0.1,
    beta: float = 0.1,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict:
    """
    Perform Non-negative Matrix Factorization using Alternating Least Squares.

    Parameters
    ----------
    V : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int
        Number of components to factorize into
    max_iter : int, optional
        Maximum number of iterations, by default 1000
    tol : float, optional
        Tolerance for stopping criterion, by default 1e-6
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), by default 'none'
    metric : Union[str, Callable], optional
        Metric to evaluate ('mse', 'mae', 'r2', 'logloss') or custom callable, by default 'mse'
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'coordinate_descent'), by default 'closed_form'
    regularization : Optional[str], optional
        Regularization type (None, 'l1', 'l2', 'elasticnet'), by default None
    alpha : float, optional
        L1 regularization parameter, by default 0.1
    beta : float, optional
        L2 regularization parameter, by default 0.1
    random_state : Optional[int], optional
        Random seed for reproducibility, by default None
    verbose : bool, optional
        Whether to print progress information, by default False

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': tuple of (W, H) matrices
        - 'metrics': dictionary of computed metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings encountered

    Example
    -------
    >>> V = np.random.rand(10, 5)
    >>> result = alternating_least_squares_fit(V, n_components=3)
    """
    # Initialize warnings
    warnings = []

    # Validate inputs
    validate_inputs(V, n_components)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Initialize matrices
    n_samples, n_features = V.shape
    W = np.abs(np.random.randn(n_samples, n_components))
    H = np.abs(np.random.randn(n_components, n_features))

    # Normalize data if required
    V_normalized = apply_normalization(V, normalize)
    if normalize != 'none':
        warnings.append(f"Data was normalized using {normalize} method")

    # Initialize metrics
    metric_func = get_metric_function(metric)
    current_metric = compute_metric(V_normalized, W @ H, metric_func)

    # Main ALS loop
    for iteration in range(max_iter):
        # Update H matrix
        H = update_matrix(V_normalized, W, H, solver, regularization, alpha, beta)

        # Update W matrix
        W = update_matrix(V_normalized.T, H.T, W.T, solver, regularization, alpha, beta).T

        # Compute current metric
        new_metric = compute_metric(V_normalized, W @ H, metric_func)
        improvement = abs(current_metric - new_metric)

        if verbose:
            print(f"Iteration {iteration + 1}/{max_iter}, Metric: {new_metric:.4f}")

        # Check convergence
        if improvement < tol:
            break

        current_metric = new_metric

    # Compute final metrics
    reconstruction_error = compute_reconstruction_error(V_normalized, W @ H)
    metrics = {
        'final_metric': current_metric,
        'reconstruction_error': reconstruction_error
    }

    # Prepare output
    result = {
        'result': (W, H),
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'max_iter': iteration + 1,
            'tol': tol,
            'normalize': normalize,
            'metric': metric if isinstance(metric, str) else 'custom',
            'solver': solver,
            'regularization': regularization,
            'alpha': alpha,
            'beta': beta
        },
        'warnings': warnings
    }

    return result

def validate_inputs(V: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(V, np.ndarray):
        raise TypeError("Input V must be a numpy array")
    if V.ndim != 2:
        raise ValueError("Input V must be a 2-dimensional array")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if V.shape[1] < n_components:
        raise ValueError("n_components cannot be larger than number of features")
    if np.any(np.isnan(V)):
        raise ValueError("Input matrix contains NaN values")
    if np.any(np.isinf(V)):
        raise ValueError("Input matrix contains infinite values")

def apply_normalization(V: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the input matrix."""
    if method == 'none':
        return V
    elif method == 'standard':
        return (V - np.mean(V, axis=0)) / np.std(V, axis=0)
    elif method == 'minmax':
        return (V - np.min(V, axis=0)) / (np.max(V, axis=0) - np.min(V, axis=0))
    elif method == 'robust':
        median = np.median(V, axis=0)
        iqr = np.percentile(V, 75, axis=0) - np.percentile(V, 25, axis=0)
        return (V - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def get_metric_function(metric: Union[str, Callable]) -> Callable:
    """Get the appropriate metric function based on input."""
    if isinstance(metric, str):
        if metric == 'mse':
            return mean_squared_error
        elif metric == 'mae':
            return mean_absolute_error
        elif metric == 'r2':
            return r_squared
        elif metric == 'logloss':
            return log_loss
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        return metric
    else:
        raise TypeError("Metric must be either a string or callable")

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric_func: Callable) -> float:
    """Compute specified metric between true and predicted values."""
    return metric_func(y_true, y_pred)

def compute_reconstruction_error(V: np.ndarray, reconstruction: np.ndarray) -> float:
    """Compute reconstruction error."""
    return mean_squared_error(V, reconstruction)

def update_matrix(
    V: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    solver: str,
    regularization: Optional[str],
    alpha: float,
    beta: float
) -> np.ndarray:
    """Update one matrix in the ALS algorithm."""
    if solver == 'closed_form':
        return closed_form_update(V, A, B, regularization, alpha, beta)
    elif solver == 'gradient_descent':
        return gradient_descent_update(V, A, B, regularization, alpha, beta)
    elif solver == 'coordinate_descent':
        return coordinate_descent_update(V, A, B, regularization, alpha, beta)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def closed_form_update(
    V: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    regularization: Optional[str],
    alpha: float,
    beta: float
) -> np.ndarray:
    """Update matrix using closed-form solution."""
    if regularization is None or regularization == 'none':
        return np.linalg.solve(A.T @ A, A.T @ V)
    elif regularization == 'l1':
        # For L1, we would typically use coordinate descent
        return coordinate_descent_update(V, A, B, regularization, alpha, beta)
    elif regularization == 'l2':
        return np.linalg.solve(A.T @ A + beta * np.eye(B.shape[0]), A.T @ V)
    elif regularization == 'elasticnet':
        return coordinate_descent_update(V, A, B, regularization, alpha, beta)
    else:
        raise ValueError(f"Unknown regularization: {regularization}")

def gradient_descent_update(
    V: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    regularization: Optional[str],
    alpha: float,
    beta: float
) -> np.ndarray:
    """Update matrix using gradient descent."""
    # Implementation of gradient descent would go here
    raise NotImplementedError("Gradient descent solver not yet implemented")

def coordinate_descent_update(
    V: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    regularization: Optional[str],
    alpha: float,
    beta: float
) -> np.ndarray:
    """Update matrix using coordinate descent."""
    # Implementation of coordinate descent would go here
    raise NotImplementedError("Coordinate descent solver not yet implemented")

################################################################################
# multiplicative_update_rules
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def multiplicative_update_rules_fit(
    V: np.ndarray,
    n_components: int,
    max_iter: int = 200,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    normalization: str = 'none',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'frobenius',
    solver: str = 'multiplicative',
    regularization: Optional[str] = None,
    alpha: float = 0.1,
    l1_ratio: float = 0.5
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Fit Non-negative Matrix Factorization (NMF) using multiplicative update rules.

    Parameters:
    -----------
    V : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int
        Number of components in the decomposition
    max_iter : int, optional
        Maximum number of iterations (default: 200)
    tol : float, optional
        Tolerance for stopping criteria (default: 1e-4)
    random_state : int, optional
        Random seed for reproducibility (default: None)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none')
    metric : str or callable, optional
        Metric to evaluate ('frobenius', 'kl', custom callable) (default: 'frobenius')
    solver : str, optional
        Solver method ('multiplicative') (default: 'multiplicative')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet') (default: None)
    alpha : float, optional
        Regularization strength (default: 0.1)
    l1_ratio : float, optional
        ElasticNet mixing parameter (default: 0.5)

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': tuple of (W, H) matrices
        - 'metrics': dictionary of computed metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings encountered

    Example:
    --------
    >>> V = np.random.rand(10, 5)
    >>> result = multiplicative_update_rules_fit(V, n_components=3)
    """
    # Validate inputs
    _validate_inputs(V, n_components)

    # Initialize random state
    rng = np.random.RandomState(random_state)

    # Normalize data if required
    V_normalized, normalization_params = _apply_normalization(V, normalization)

    # Initialize W and H
    W = rng.rand(V.shape[0], n_components)
    H = rng.rand(n_components, V.shape[1])

    # Normalize initial factors
    W /= np.sum(W, axis=1, keepdims=True)
    H *= np.sum(V_normalized) / np.sum(H)

    # Prepare output
    metrics = {}
    warnings = []

    # Main optimization loop
    for i in range(max_iter):
        # Update rules
        W, H = _multiplicative_update(V_normalized, W, H)

        # Apply regularization if needed
        if regularization:
            W, H = _apply_regularization(W, H, regularization, alpha, l1_ratio)

        # Compute current metric
        current_metric = _compute_metric(V_normalized, W, H, metric)

        # Check convergence
        if i > 0 and abs(prev_metric - current_metric) < tol:
            break

        prev_metric = current_metric
        metrics[f'iteration_{i}'] = current_metric

    # Store results
    result = {
        'result': (W, H),
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'max_iter': max_iter,
            'tol': tol,
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
            'solver': solver,
            'regularization': regularization,
            'alpha': alpha,
            'l1_ratio': l1_ratio
        },
        'warnings': warnings
    }

    return result

def _validate_inputs(V: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(V, np.ndarray):
        raise TypeError("Input V must be a numpy array")
    if V.ndim != 2:
        raise ValueError("Input V must be a 2D array")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if V.shape[0] < n_components or V.shape[1] < n_components:
        raise ValueError("n_components cannot be larger than input dimensions")

def _apply_normalization(V: np.ndarray, method: str) -> tuple:
    """Apply specified normalization to the input data."""
    if method == 'none':
        return V, None
    elif method == 'standard':
        mean = np.mean(V, axis=0)
        std = np.std(V, axis=0)
        V_normalized = (V - mean) / std
    elif method == 'minmax':
        min_val = np.min(V, axis=0)
        max_val = np.max(V, axis=0)
        V_normalized = (V - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(V, axis=0)
        iqr = np.subtract(*np.percentile(V, [75, 25], axis=0))
        V_normalized = (V - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return V_normalized, None

def _multiplicative_update(V: np.ndarray, W: np.ndarray, H: np.ndarray) -> tuple:
    """Perform multiplicative update for NMF."""
    # Update H
    numerator = W.T @ V
    denominator = W.T @ W @ H + 1e-8
    H *= numerator / denominator

    # Update W
    numerator = V @ H.T
    denominator = W @ H @ H.T + 1e-8
    W *= numerator / denominator

    return W, H

def _apply_regularization(W: np.ndarray, H: np.ndarray,
                         method: str, alpha: float, l1_ratio: float) -> tuple:
    """Apply regularization to the factors."""
    if method == 'l1':
        W -= alpha * np.sign(W)
        H -= alpha * np.sign(H)
    elif method == 'l2':
        W *= (1 - 2 * alpha)
        H *= (1 - 2 * alpha)
    elif method == 'elasticnet':
        W -= alpha * l1_ratio * np.sign(W)
        W *= (1 - alpha * (1 - l1_ratio))
        H -= alpha * l1_ratio * np.sign(H)
        H *= (1 - alpha * (1 - l1_ratio))
    return W, H

def _compute_metric(V: np.ndarray, W: np.ndarray, H: np.ndarray,
                    metric: Union[str, Callable]) -> float:
    """Compute the specified metric between V and WH."""
    reconstruction = W @ H

    if isinstance(metric, str):
        if metric == 'frobenius':
            return np.linalg.norm(V - reconstruction, 'fro')
        elif metric == 'kl':
            return np.sum(V * (np.log(V + 1e-8) - np.log(reconstruction + 1e-8)) -
                         V + reconstruction)
    else:
        return metric(V, reconstruction)

    raise ValueError(f"Unknown metric: {metric}")

################################################################################
# frobenius_norm
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def frobenius_norm_fit(
    matrix: np.ndarray,
    *,
    normalization: str = "none",
    metric: Union[str, Callable] = "frobenius",
    solver: str = "closed_form",
    tol: float = 1e-6,
    max_iter: int = 1000,
    regularization: Optional[str] = None,
    l1_ratio: float = 0.5,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute the Frobenius norm of a matrix with various options.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix for which to compute the Frobenius norm.
    normalization : str, optional
        Type of normalization to apply. Options: "none", "standard", "minmax", "robust".
    metric : str or callable, optional
        Metric to use. Options: "frobenius", custom callable.
    solver : str, optional
        Solver to use. Options: "closed_form", "gradient_descent".
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    regularization : str or None, optional
        Type of regularization. Options: "l1", "l2", "elasticnet", None.
    l1_ratio : float, optional
        Ratio of L1 regularization in elasticnet.
    custom_metric : callable or None, optional
        Custom metric function.

    Returns
    -------
    Dict
        Dictionary containing the result, metrics, parameters used, and warnings.

    Examples
    --------
    >>> matrix = np.array([[1, 2], [3, 4]])
    >>> result = frobenius_norm_fit(matrix)
    """
    # Validate inputs
    _validate_inputs(matrix, normalization, metric, solver, regularization)

    # Normalize the matrix
    normalized_matrix = _apply_normalization(matrix, normalization)

    # Choose solver
    if solver == "closed_form":
        result = _closed_form_solution(normalized_matrix)
    elif solver == "gradient_descent":
        result = _gradient_descent_solution(normalized_matrix, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(result, normalized_matrix, metric, custom_metric)

    # Prepare output
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "solver": solver,
            "tol": tol,
            "max_iter": max_iter,
            "regularization": regularization,
            "l1_ratio": l1_ratio
        },
        "warnings": []
    }

    return output

def _validate_inputs(
    matrix: np.ndarray,
    normalization: str,
    metric: Union[str, Callable],
    solver: str,
    regularization: Optional[str]
) -> None:
    """Validate input parameters."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input matrix must be a numpy array.")
    if normalization not in ["none", "standard", "minmax", "robust"]:
        raise ValueError(f"Unknown normalization: {normalization}")
    if isinstance(metric, str) and metric != "frobenius":
        raise ValueError(f"Unknown metric: {metric}")
    if solver not in ["closed_form", "gradient_descent"]:
        raise ValueError(f"Unknown solver: {solver}")
    if regularization not in [None, "l1", "l2", "elasticnet"]:
        raise ValueError(f"Unknown regularization: {regularization}")

def _apply_normalization(
    matrix: np.ndarray,
    normalization: str
) -> np.ndarray:
    """Apply normalization to the matrix."""
    if normalization == "none":
        return matrix
    elif normalization == "standard":
        return (matrix - np.mean(matrix)) / np.std(matrix)
    elif normalization == "minmax":
        return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    elif normalization == "robust":
        return (matrix - np.median(matrix)) / (np.percentile(matrix, 75) - np.percentile(matrix, 25))
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

def _closed_form_solution(
    matrix: np.ndarray
) -> float:
    """Compute the Frobenius norm using closed-form solution."""
    return np.sqrt(np.sum(matrix ** 2))

def _gradient_descent_solution(
    matrix: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> float:
    """Compute the Frobenius norm using gradient descent."""
    # Initialize parameters
    norm_estimate = np.sum(matrix ** 2)
    prev_norm = 0.0
    iter_count = 0

    # Gradient descent loop
    while abs(norm_estimate - prev_norm) > tol and iter_count < max_iter:
        prev_norm = norm_estimate
        # Update norm estimate (simplified for illustration)
        norm_estimate -= 0.01 * (norm_estimate - np.sum(matrix ** 2))
        iter_count += 1

    return np.sqrt(norm_estimate)

def _compute_metrics(
    result: float,
    matrix: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute metrics based on the result."""
    metrics = {}

    if isinstance(metric, str) and metric == "frobenius":
        metrics["frobenius_norm"] = result
    elif callable(custom_metric):
        metrics["custom_metric"] = custom_metric(matrix)

    return metrics

################################################################################
# kl_divergence
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    normalize_X: bool = False,
    normalization_method: str = 'none'
) -> None:
    """Validate input matrices and parameters."""
    if not isinstance(X, np.ndarray) or not isinstance(W, np.ndarray) or not isinstance(H, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2 or W.ndim != 2 or H.ndim != 2:
        raise ValueError("Inputs must be 2-dimensional arrays")
    if X.shape[0] != W.shape[0]:
        raise ValueError("X and W must have the same number of rows")
    if H.shape[1] != X.shape[1]:
        raise ValueError("H and X must have the same number of columns")
    if normalize_X:
        if normalization_method not in ['none', 'standard', 'minmax', 'robust']:
            raise ValueError("Invalid normalization method")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(W)) or np.any(np.isinf(W)):
        raise ValueError("W contains NaN or infinite values")
    if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        raise ValueError("H contains NaN or infinite values")

def _normalize_matrix(
    matrix: np.ndarray,
    method: str = 'none'
) -> np.ndarray:
    """Normalize a matrix using the specified method."""
    if method == 'none':
        return matrix
    elif method == 'standard':
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        return (matrix - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(matrix, axis=0)
        max_val = np.max(matrix, axis=0)
        return (matrix - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(matrix, axis=0)
        iqr = np.subtract(*np.percentile(matrix, [75, 25], axis=0))
        return (matrix - median) / (iqr + 1e-8)
    else:
        raise ValueError("Invalid normalization method")

def _compute_kl_divergence(
    X: np.ndarray,
    WH: np.ndarray
) -> float:
    """Compute the KL divergence between X and WH."""
    epsilon = 1e-10
    X_positive = np.maximum(X, epsilon)
    WH_positive = np.maximum(WH, epsilon)
    return np.sum(X_positive * (np.log(X_positive / WH_positive) - 1) + WH_positive)

def _compute_metrics(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    metric_functions: Dict[str, Callable]
) -> Dict[str, float]:
    """Compute various metrics based on user-provided functions."""
    WH = W @ H
    metrics = {}
    for name, func in metric_functions.items():
        if name == 'kl_divergence':
            metrics[name] = _compute_kl_divergence(X, WH)
        else:
            metrics[name] = func(X, WH)
    return metrics

def kl_divergence_fit(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    normalize_X: bool = False,
    normalization_method: str = 'none',
    metric_functions: Optional[Dict[str, Callable]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute the KL divergence and other metrics between X and W*H.

    Parameters:
    - X: Input data matrix of shape (n_samples, n_features)
    - W: Factor matrix of shape (n_samples, n_components)
    - H: Factor matrix of shape (n_components, n_features)
    - normalize_X: Whether to normalize X before computation
    - normalization_method: Method for normalizing X ('none', 'standard', 'minmax', 'robust')
    - metric_functions: Dictionary of metric functions to compute
    - **kwargs: Additional parameters for metric functions

    Returns:
    - Dictionary containing results, metrics, and other information
    """
    # Validate inputs
    _validate_inputs(X, W, H, normalize_X, normalization_method)

    # Normalize X if requested
    if normalize_X:
        X = _normalize_matrix(X, normalization_method)

    # Compute WH
    WH = W @ H

    # Set default metric functions if none provided
    if metric_functions is None:
        metric_functions = {
            'kl_divergence': _compute_kl_divergence
        }

    # Compute metrics
    metrics = _compute_metrics(X, W, H, metric_functions)

    # Prepare output
    result = {
        "result": WH,
        "metrics": metrics,
        "params_used": {
            "normalize_X": normalize_X,
            "normalization_method": normalization_method
        },
        "warnings": []
    }

    return result

def kl_divergence_compute(
    X: np.ndarray,
    WH: np.ndarray
) -> float:
    """
    Compute the KL divergence between X and WH.

    Parameters:
    - X: Input data matrix of shape (n_samples, n_features)
    - WH: Product matrix W*H of shape (n_samples, n_features)

    Returns:
    - KL divergence value
    """
    _validate_inputs(X, np.zeros((X.shape[0], 1)), WH.reshape(WH.shape[0], -1))
    return _compute_kl_divergence(X, WH)

################################################################################
# beta_divergence
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def beta_divergence_fit(
    X: np.ndarray,
    W: Optional[np.ndarray] = None,
    H: Optional[np.ndarray] = None,
    beta: float = 2.0,
    solver: str = 'gradient_descent',
    max_iter: int = 1000,
    tol: float = 1e-6,
    normalization: str = 'none',
    metric: str = 'mse',
    regularization: Optional[str] = None,
    l1_ratio: float = 0.5,
    alpha: float = 1.0,
    beta_divergence_func: Optional[Callable] = None,
    distance_func: Optional[Callable] = None,
    metric_func: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Compute the beta divergence between matrices X and WH.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    W : Optional[np.ndarray]
        Basis matrix of shape (n_samples, n_components), default None
    H : Optional[np.ndarray]
        Coefficient matrix of shape (n_components, n_features), default None
    beta : float
        Beta parameter for the divergence, default 2.0 (Euclidean distance)
    solver : str
        Solver to use, default 'gradient_descent'
    max_iter : int
        Maximum number of iterations, default 1000
    tol : float
        Tolerance for convergence, default 1e-6
    normalization : str
        Normalization method, default 'none'
    metric : str
        Metric to compute, default 'mse'
    regularization : Optional[str]
        Regularization type, default None
    l1_ratio : float
        Elastic net mixing parameter, default 0.5
    alpha : float
        Regularization strength, default 1.0
    beta_divergence_func : Optional[Callable]
        Custom beta divergence function, default None
    distance_func : Optional[Callable]
        Custom distance function, default None
    metric_func : Optional[Callable]
        Custom metric function, default None

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, W, H)

    # Normalize data if needed
    X_norm = _apply_normalization(X, normalization)

    # Initialize W and H if not provided
    if W is None or H is None:
        W, H = _initialize_matrices(X_norm)

    # Choose beta divergence function
    if beta_divergence_func is None:
        beta_divergence_func = _beta_divergence

    # Choose distance function
    if distance_func is None:
        distance_func = _euclidean_distance

    # Choose metric function
    if metric_func is None:
        metric_func = _get_metric_function(metric)

    # Choose solver
    if solver == 'gradient_descent':
        W, H = _gradient_descent_solver(X_norm, W, H, beta_divergence_func,
                                       distance_func, metric_func,
                                       max_iter, tol, regularization,
                                       l1_ratio, alpha)
    elif solver == 'coordinate_descent':
        W, H = _coordinate_descent_solver(X_norm, W, H, beta_divergence_func,
                                         distance_func, metric_func,
                                         max_iter, tol, regularization,
                                         l1_ratio, alpha)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute final metrics
    divergence = beta_divergence_func(X_norm, W @ H, beta)
    metric_value = metric_func(X_norm, W @ H)

    return {
        'result': {'W': W, 'H': H},
        'metrics': {'divergence': divergence, metric: metric_value},
        'params_used': {
            'beta': beta,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol,
            'normalization': normalization,
            'metric': metric,
            'regularization': regularization,
            'l1_ratio': l1_ratio,
            'alpha': alpha
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, W: Optional[np.ndarray], H: Optional[np.ndarray]) -> None:
    """Validate input matrices dimensions and types."""
    if not isinstance(X, np.ndarray) or (W is not None and not isinstance(W, np.ndarray)) or (H is not None and not isinstance(H, np.ndarray)):
        raise TypeError("Inputs must be numpy arrays")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if W is not None and (W.ndim != 2 or X.shape[0] != W.shape[0]):
        raise ValueError("W must have the same number of rows as X")

    if H is not None and (H.ndim != 2 or W.shape[1] != H.shape[0]):
        raise ValueError("H must have compatible dimensions with W")

    if np.any(np.isnan(X)) or (W is not None and np.any(np.isnan(W))) or (H is not None and np.any(np.isnan(H))):
        raise ValueError("Input arrays must not contain NaN values")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to the input matrix."""
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

def _initialize_matrices(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Initialize W and H matrices."""
    n_samples, n_features = X.shape
    n_components = min(n_samples, n_features) // 2

    W = np.abs(np.random.randn(n_samples, n_components))
    H = np.abs(np.random.randn(n_components, n_features))

    return W, H

def _beta_divergence(X: np.ndarray, WH: np.ndarray, beta: float) -> float:
    """Compute the beta divergence between X and WH."""
    if beta == 0:
        return np.sum(X * np.log(X / WH) - X + WH)
    elif beta == 1:
        return np.sum(WH - X * np.log(WH))
    elif beta == 2:
        return 0.5 * np.sum((X - WH) ** 2)
    else:
        return (np.sum(WH**beta / beta + (beta - 1) * X**beta / beta - X * WH**(beta - 1)))

def _euclidean_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Euclidean distance between X and Y."""
    return np.sum((X - Y) ** 2)

def _get_metric_function(metric: str) -> Callable:
    """Get the appropriate metric function."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared,
        'logloss': _log_loss
    }
    return metrics.get(metric, lambda X, Y: np.nan)

def _mean_squared_error(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return np.mean((X - Y) ** 2)

def _mean_absolute_error(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(X - Y))

def _r_squared(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((X - Y) ** 2)
    ss_tot = np.sum((X - np.mean(X)) ** 2)
    return 1 - ss_res / ss_tot

def _log_loss(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Log Loss."""
    epsilon = 1e-15
    Y_pred = np.clip(Y, epsilon, 1 - epsilon)
    return -np.mean(X * np.log(Y_pred) + (1 - X) * np.log(1 - Y_pred))

def _gradient_descent_solver(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    beta_divergence_func: Callable,
    distance_func: Callable,
    metric_func: Callable,
    max_iter: int,
    tol: float,
    regularization: Optional[str],
    l1_ratio: float,
    alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient descent solver for NMF."""
    prev_divergence = float('inf')
    for _ in range(max_iter):
        # Update H
        grad_H = _compute_gradient_H(X, W, H, beta_divergence_func)
        if regularization:
            grad_H += _compute_regularization(H, regularization, l1_ratio, alpha)
        H -= 0.01 * grad_H

        # Update W
        grad_W = _compute_gradient_W(X, W, H, beta_divergence_func)
        if regularization:
            grad_W += _compute_regularization(W, regularization, l1_ratio, alpha)
        W -= 0.01 * grad_W

        # Check convergence
        current_divergence = beta_divergence_func(X, W @ H)
        if abs(prev_divergence - current_divergence) < tol:
            break
        prev_divergence = current_divergence

    return W, H

def _coordinate_descent_solver(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    beta_divergence_func: Callable,
    distance_func: Callable,
    metric_func: Callable,
    max_iter: int,
    tol: float,
    regularization: Optional[str],
    l1_ratio: float,
    alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate descent solver for NMF."""
    prev_divergence = float('inf')
    for _ in range(max_iter):
        # Update H
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                H_ij = _update_H_ij(X, W, H, i, j, beta_divergence_func)
                if regularization:
                    H_ij = _apply_regularization(H_ij, i, j, regularization, l1_ratio, alpha)
                H[i, j] = H_ij

        # Update W
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_ij = _update_W_ij(X, W, H, i, j, beta_divergence_func)
                if regularization:
                    W_ij = _apply_regularization(W_ij, i, j, regularization, l1_ratio, alpha)
                W[i, j] = W_ij

        # Check convergence
        current_divergence = beta_divergence_func(X, W @ H)
        if abs(prev_divergence - current_divergence) < tol:
            break
        prev_divergence = current_divergence

    return W, H

def _compute_gradient_H(X: np.ndarray, W: np.ndarray, H: np.ndarray, beta_divergence_func: Callable) -> np.ndarray:
    """Compute gradient of H for beta divergence."""
    WH = W @ H
    if beta_divergence_func == _beta_divergence and beta == 2:
        return W.T @ (WH - X) / WH.shape[0]
    else:
        # General case for other beta values
        return W.T @ (beta_divergence_func.gradient(X, WH)) / WH.shape[0]

def _compute_gradient_W(X: np.ndarray, W: np.ndarray, H: np.ndarray, beta_divergence_func: Callable) -> np.ndarray:
    """Compute gradient of W for beta divergence."""
    WH = W @ H
    if beta_divergence_func == _beta_divergence and beta == 2:
        return (WH - X) @ H.T / WH.shape[0]
    else:
        # General case for other beta values
        return (beta_divergence_func.gradient(X, WH)) @ H.T / WH.shape[0]

def _compute_regularization(matrix: np.ndarray, method: str, l1_ratio: float, alpha: float) -> np.ndarray:
    """Compute regularization term."""
    if method == 'l1':
        return alpha * np.sign(matrix)
    elif method == 'l2':
        return 2 * alpha * matrix
    elif method == 'elasticnet':
        l1 = l1_ratio * alpha * np.sign(matrix)
        l2 = (1 - l1_ratio) * 2 * alpha * matrix
        return l1 + l2
    else:
        raise ValueError(f"Unknown regularization method: {method}")

def _apply_regularization(value: float, i: int, j: int, method: str, l1_ratio: float, alpha: float) -> float:
    """Apply regularization to a single value."""
    if method == 'l1':
        return np.sign(value) * (np.abs(value) - alpha)
    elif method == 'l2':
        return value / (1 + 2 * alpha)
    elif method == 'elasticnet':
        if np.abs(value) < l1_ratio * alpha / (1 - l1_ratio + 2 * alpha):
            return 0
        else:
            return (value - l1_ratio * np.sign(value)) / (1 - l1_ratio + 2 * alpha)
    else:
        raise ValueError(f"Unknown regularization method: {method}")

def _update_H_ij(X: np.ndarray, W: np.ndarray, H: np.ndarray, i: int, j: int, beta_divergence_func: Callable) -> float:
    """Update a single element of H."""
    # This is a simplified version - actual implementation would need to compute
    # the optimal value for H_ij given current W and other H values
    return 0.5

def _update_W_ij(X: np.ndarray, W: np.ndarray, H: np.ndarray, i: int, j: int, beta_divergence_func: Callable) -> float:
    """Update a single element of W."""
    # This is a simplified version - actual implementation would need to compute
    # the optimal value for W_ij given current H and other W values
    return 0.5

# Example usage:
"""
X = np.random.rand(10, 5)
result = beta_divergence_fit(X, beta=2.0, solver='gradient_descent', max_iter=100)
"""

################################################################################
# sparse_coding
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def sparse_coding_fit(
    X: np.ndarray,
    D: np.ndarray,
    n_iter: int = 1000,
    tol: float = 1e-4,
    solver: str = 'gradient_descent',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    normalization: Optional[str] = None,
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict:
    """
    Perform sparse coding on input data using a dictionary.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    D : np.ndarray
        Dictionary matrix of shape (n_features, n_components)
    n_iter : int, optional
        Maximum number of iterations, default 1000
    tol : float, optional
        Tolerance for stopping criterion, default 1e-4
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'coordinate_descent'), default 'gradient_descent'
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2'), default 'mse'
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine'), default 'euclidean'
    normalization : str, optional
        Normalization method (None, 'standard', 'minmax'), default None
    regularization : str, optional
        Regularization type (None, 'l1', 'l2'), default None
    alpha : float, optional
        Regularization strength for L1 penalty, default 1.0
    beta : float, optional
        Regularization strength for L2 penalty, default 1.0
    random_state : int, optional
        Random seed for reproducibility, default None
    verbose : bool, optional
        Whether to print progress information, default False

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': Coefficients matrix
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in the computation
        - 'warnings': Any warnings generated during computation

    Example:
    --------
    >>> X = np.random.rand(10, 5)
    >>> D = np.random.rand(5, 3)
    >>> result = sparse_coding_fit(X, D, n_iter=100, solver='gradient_descent')
    """
    # Validate inputs
    _validate_inputs(X, D)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Initialize coefficients
    C = _initialize_coefficients(X.shape[0], D.shape[1])

    # Normalize data if required
    X_norm, D_norm = _apply_normalization(X, D, normalization)

    # Select solver
    if solver == 'closed_form':
        C = _solve_closed_form(X_norm, D_norm)
    elif solver == 'gradient_descent':
        C = _solve_gradient_descent(X_norm, D_norm, n_iter, tol, regularization, alpha, beta)
    elif solver == 'coordinate_descent':
        C = _solve_coordinate_descent(X_norm, D_norm, n_iter, tol, regularization, alpha, beta)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_norm, D_norm, C, metric)

    # Prepare output
    result = {
        'result': C,
        'metrics': metrics,
        'params_used': {
            'n_iter': n_iter,
            'tol': tol,
            'solver': solver,
            'metric': metric,
            'distance': distance,
            'normalization': normalization,
            'regularization': regularization,
            'alpha': alpha,
            'beta': beta
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, D: np.ndarray) -> None:
    """Validate input matrices."""
    if not isinstance(X, np.ndarray) or not isinstance(D, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2 or D.ndim != 2:
        raise ValueError("Inputs must be 2-dimensional")
    if X.shape[1] != D.shape[0]:
        raise ValueError("Number of features in X must match number of rows in D")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(D)) or np.any(np.isinf(D)):
        raise ValueError("D contains NaN or infinite values")

def _initialize_coefficients(n_samples: int, n_components: int) -> np.ndarray:
    """Initialize coefficients matrix."""
    return np.random.randn(n_samples, n_components)

def _apply_normalization(X: np.ndarray, D: np.ndarray, method: Optional[str]) -> tuple:
    """Apply normalization to input matrices."""
    if method == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        D = (D - np.mean(D, axis=0)) / np.std(D, axis=0)
    elif method == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        D = (D - np.min(D, axis=0)) / (np.max(D, axis=0) - np.min(D, axis=0))
    return X, D

def _solve_closed_form(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Solve sparse coding using closed form solution."""
    return np.linalg.pinv(D.T @ D) @ D.T @ X

def _solve_gradient_descent(
    X: np.ndarray,
    D: np.ndarray,
    n_iter: int,
    tol: float,
    regularization: Optional[str],
    alpha: float,
    beta: float
) -> np.ndarray:
    """Solve sparse coding using gradient descent."""
    C = _initialize_coefficients(X.shape[0], D.shape[1])
    prev_loss = float('inf')

    for _ in range(n_iter):
        # Compute gradient
        grad = 2 * (D @ C - X) @ D.T

        # Apply regularization
        if regularization == 'l1':
            grad += alpha * np.sign(C)
        elif regularization == 'l2':
            grad += 2 * beta * C

        # Update coefficients
        C -= 0.01 * grad

        # Check convergence
        current_loss = np.linalg.norm(D @ C - X, 'fro')
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return C

def _solve_coordinate_descent(
    X: np.ndarray,
    D: np.ndarray,
    n_iter: int,
    tol: float,
    regularization: Optional[str],
    alpha: float,
    beta: float
) -> np.ndarray:
    """Solve sparse coding using coordinate descent."""
    C = _initialize_coefficients(X.shape[0], D.shape[1])
    prev_loss = float('inf')

    for _ in range(n_iter):
        for i in range(C.shape[1]):
            # Compute residual
            r = X - D @ C

            # Update coefficient for feature i
            if regularization == 'l1':
                C[:, i] = _soft_threshold(r @ D[:, i], alpha)
            elif regularization == 'l2':
                C[:, i] = (r @ D[:, i]) / (D[:, i].T @ D[:, i] + beta)
            else:
                C[:, i] = (r @ D[:, i]) / (D[:, i].T @ D[:, i])

        # Check convergence
        current_loss = np.linalg.norm(D @ C - X, 'fro')
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return C

def _soft_threshold(x: np.ndarray, alpha: float) -> np.ndarray:
    """Soft thresholding operator."""
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

def _compute_metrics(
    X: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Compute metrics for sparse coding."""
    reconstruction = D @ C
    metrics = {}

    if metric == 'mse':
        metrics['mse'] = np.mean((X - reconstruction) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(X - reconstruction))
    elif metric == 'r2':
        ss_res = np.sum((X - reconstruction) ** 2)
        ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
        metrics['r2'] = 1 - ss_res / ss_tot
    elif callable(metric):
        metrics['custom'] = metric(X, reconstruction)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

################################################################################
# dictionary_learning
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def dictionary_learning_fit(
    X: np.ndarray,
    n_components: int,
    max_iter: int = 1000,
    tol: float = 1e-4,
    solver: str = 'gradient_descent',
    metric: Union[str, Callable] = 'mse',
    normalization: str = 'none',
    distance: Union[str, Callable] = 'euclidean',
    regularization: str = 'none',
    l1_ratio: float = 0.5,
    alpha: float = 1.0,
    beta: float = 1.0,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict:
    """
    Perform dictionary learning on input data X.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int
        Number of components to learn
    max_iter : int, optional
        Maximum number of iterations (default: 1000)
    tol : float, optional
        Tolerance for stopping criterion (default: 1e-4)
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'coordinate_descent') (default: 'gradient_descent')
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2') or custom callable (default: 'mse')
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none')
    distance : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine') or custom callable (default: 'euclidean')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet') (default: 'none')
    l1_ratio : float, optional
        ElasticNet mixing parameter (default: 0.5)
    alpha : float, optional
        L1 regularization parameter (default: 1.0)
    beta : float, optional
        L2 regularization parameter (default: 1.0)
    random_state : int, optional
        Random seed for reproducibility (default: None)
    verbose : bool, optional
        Whether to print progress information (default: False)

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': Learned dictionary and coefficients
        - 'metrics': Computed metrics
        - 'params_used': Parameters actually used
        - 'warnings': Any warnings generated

    Example:
    --------
    >>> X = np.random.rand(100, 20)
    >>> result = dictionary_learning_fit(X, n_components=10)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Initialize dictionary and coefficients
    D = _initialize_dictionary(X.shape[1], n_components, random_state)
    C = np.zeros((X.shape[0], n_components))

    # Normalize data if specified
    X_norm, norm_params = _normalize_data(X, normalization)

    # Select solver function
    if solver == 'closed_form':
        solve_func = _solve_closed_form
    elif solver == 'gradient_descent':
        solve_func = _solve_gradient_descent
    elif solver == 'coordinate_descent':
        solve_func = _solve_coordinate_descent
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Main optimization loop
    for iteration in range(max_iter):
        # Update coefficients
        C = _update_coefficients(X_norm, D, distance, regularization,
                                l1_ratio, alpha, beta)

        # Update dictionary
        D = solve_func(X_norm, C, distance, regularization,
                      l1_ratio, alpha, beta)

        # Compute metrics
        current_metric = _compute_metric(X_norm, D, C, metric)

        # Check convergence
        if iteration > 0 and abs(prev_metric - current_metric) < tol:
            break

        prev_metric = current_metric
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}, metric: {current_metric}")

    # Compute final metrics
    metrics = _compute_final_metrics(X_norm, D, C, metric)

    # Prepare output
    result = {
        'dictionary': D,
        'coefficients': C,
        'metrics': metrics,
        'params_used': {
            'solver': solver,
            'metric': metric,
            'normalization': normalization,
            'distance': distance,
            'regularization': regularization
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
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if X.shape[1] < n_components:
        raise ValueError("n_components cannot be greater than number of features")

def _initialize_dictionary(n_features: int, n_components: int,
                          random_state: Optional[int]) -> np.ndarray:
    """Initialize dictionary with random values."""
    if random_state is not None:
        np.random.seed(random_state)
    return np.random.randn(n_features, n_components)

def _normalize_data(X: np.ndarray, method: str) -> tuple:
    """Normalize data according to specified method."""
    norm_params = {}
    X_norm = X.copy()

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

def _update_coefficients(X: np.ndarray, D: np.ndarray,
                         distance: Union[str, Callable],
                         regularization: str,
                         l1_ratio: float,
                         alpha: float,
                         beta: float) -> np.ndarray:
    """Update coefficients using specified distance and regularization."""
    # Implement coefficient update logic based on parameters
    pass

def _solve_closed_form(X: np.ndarray, C: np.ndarray,
                       distance: Union[str, Callable],
                       regularization: str,
                       l1_ratio: float,
                       alpha: float,
                       beta: float) -> np.ndarray:
    """Solve dictionary learning using closed form solution."""
    # Implement closed form solution
    pass

def _solve_gradient_descent(X: np.ndarray, C: np.ndarray,
                            distance: Union[str, Callable],
                            regularization: str,
                            l1_ratio: float,
                            alpha: float,
                            beta: float) -> np.ndarray:
    """Solve dictionary learning using gradient descent."""
    # Implement gradient descent solution
    pass

def _solve_coordinate_descent(X: np.ndarray, C: np.ndarray,
                              distance: Union[str, Callable],
                              regularization: str,
                              l1_ratio: float,
                              alpha: float,
                              beta: float) -> np.ndarray:
    """Solve dictionary learning using coordinate descent."""
    # Implement coordinate descent solution
    pass

def _compute_metric(X: np.ndarray, D: np.ndarray, C: np.ndarray,
                    metric: Union[str, Callable]) -> float:
    """Compute specified metric between X and DC."""
    if callable(metric):
        return metric(X, D @ C)
    elif metric == 'mse':
        return np.mean((X - D @ C) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(X - D @ C))
    elif metric == 'r2':
        ss_res = np.sum((X - D @ C) ** 2)
        ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-8)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_final_metrics(X: np.ndarray, D: np.ndarray,
                           C: np.ndarray,
                           metric: Union[str, Callable]) -> Dict:
    """Compute final set of metrics."""
    return {
        'primary_metric': _compute_metric(X, D, C, metric),
        'reconstruction_error': np.mean((X - D @ C) ** 2)
    }

################################################################################
# applications
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def applications_fit(
    X: np.ndarray,
    n_components: int,
    normalizer: str = 'none',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'gradient_descent',
    max_iter: int = 1000,
    tol: float = 1e-4,
    alpha: float = 0.0,
    l1_ratio: float = 0.5,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit NMF applications model with configurable parameters.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int
        Number of components for NMF decomposition
    normalizer : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2', 'logloss') or custom function
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'coordinate_descent')
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Tolerance for stopping criterion
    alpha : float, optional
        Regularization strength (0 means no regularization)
    l1_ratio : float, optional
        ElasticNet mixing parameter (0 = L2, 1 = L1)
    random_state : int or None, optional
        Random seed for reproducibility
    custom_metric : callable, optional
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(10, 5)
    >>> result = applications_fit(X, n_components=3)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Initialize random state
    rng = np.random.RandomState(random_state)

    # Normalize data if required
    X_norm, norm_params = _apply_normalization(X, normalizer)

    # Initialize NMF components
    W = rng.rand(n_components, X_norm.shape[1])
    H = rng.rand(X_norm.shape[0], n_components)

    # Select solver
    if solver == 'closed_form':
        W, H = _closed_form_solver(X_norm, n_components)
    elif solver == 'gradient_descent':
        W, H = _gradient_descent_solver(
            X_norm, n_components, max_iter, tol,
            alpha, l1_ratio, rng
        )
    elif solver == 'coordinate_descent':
        W, H = _coordinate_descent_solver(
            X_norm, n_components, max_iter, tol,
            alpha, l1_ratio
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate reconstruction
    X_reconstructed = np.dot(W, H.T)

    # Calculate metrics
    metrics = _calculate_metrics(
        X_norm, X_reconstructed,
        metric, custom_metric
    )

    # Prepare output
    result = {
        'W': W,
        'H': H,
        'X_reconstructed': X_reconstructed
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer,
            'metric': metric,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol,
            'alpha': alpha,
            'l1_ratio': l1_ratio
        },
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
    if np.any(np.isnan(X)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input data contains infinite values")

def _apply_normalization(
    X: np.ndarray,
    method: str
) -> tuple[np.ndarray, Optional[Dict]]:
    """Apply data normalization."""
    norm_params = None
    if method == 'standard':
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        norm_params = {'mean': mean, 'std': std}
        X_norm = (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)
        norm_params = {'min': min_val, 'max': max_val}
        X_norm = (X - min_val) / ((max_val - min_val) + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        norm_params = {'median': median, 'iqr': iqr}
        X_norm = (X - median) / ((iqr + 1e-8))
    elif method != 'none':
        raise ValueError(f"Unknown normalization method: {method}")
    else:
        X_norm = X.copy()
    return X_norm, norm_params

def _closed_form_solver(
    X: np.ndarray,
    n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Closed form solution for NMF."""
    # This is a placeholder - actual implementation would use
    # appropriate closed form solution for NMF
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    W = U[:, :n_components] @ np.diag(S[:n_components])
    H = Vt[:n_components, :]
    return W, H

def _gradient_descent_solver(
    X: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float,
    alpha: float,
    l1_ratio: float,
    rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient descent solver for NMF."""
    n_samples, n_features = X.shape
    W = rng.rand(n_components, n_features)
    H = rng.rand(n_samples, n_components)

    for _ in range(max_iter):
        # Update H
        numerator = X @ W.T - W @ H.T @ H
        denominator = (W.T @ W) + alpha * _elasticnet_regularization(H, l1_ratio)
        H = np.maximum(H - (H * numerator) / denominator, 0)

        # Update W
        numerator = X.T @ H - W @ H.T @ H
        denominator = (H.T @ H) + alpha * _elasticnet_regularization(W, l1_ratio)
        W = np.maximum(W - (W * numerator) / denominator, 0)

    return W, H

def _coordinate_descent_solver(
    X: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float,
    alpha: float,
    l1_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate descent solver for NMF."""
    n_samples, n_features = X.shape
    W = np.random.rand(n_components, n_features)
    H = np.random.rand(n_samples, n_components)

    for _ in range(max_iter):
        # Update H using coordinate descent
        for i in range(n_samples):
            for j in range(n_components):
                # Compute gradient
                grad = - (X[i, :] @ W[j, :]) + np.sum(W[:, :] * H[i, j])

                # Update with regularization
                if alpha > 0:
                    grad += alpha * _elasticnet_derivative(H[i, j], l1_ratio)

                # Update parameter
                H_new = np.maximum(H[i, j] - grad / (np.sum(W[:, :]**2) + 1e-8), 0)
                H[i, j] = H_new

        # Update W using coordinate descent
        for i in range(n_components):
            for j in range(n_features):
                # Compute gradient
                grad = - (X[:, j].T @ H[:, i]) + np.sum(H[:, :] * W[i, j])

                # Update with regularization
                if alpha > 0:
                    grad += alpha * _elasticnet_derivative(W[i, j], l1_ratio)

                # Update parameter
                W_new = np.maximum(W[i, j] - grad / (np.sum(H[:, :]**2) + 1e-8), 0)
                W[i, j] = W_new

    return W, H

def _elasticnet_regularization(
    X: np.ndarray,
    l1_ratio: float
) -> np.ndarray:
    """Compute ElasticNet regularization."""
    return l1_ratio * np.abs(X) + (1 - l1_ratio) * X**2

def _elasticnet_derivative(
    x: float,
    l1_ratio: float
) -> float:
    """Compute ElasticNet derivative."""
    if x > 0:
        return l1_ratio + (1 - l1_ratio) * 2 * x
    elif x < 0:
        return -l1_ratio + (1 - l1_ratio) * 2 * x
    else:
        return np.sign(x) * l1_ratio

def _calculate_metrics(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate evaluation metrics."""
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((X_true - X_pred)**2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(X_true - X_pred))
        elif metric == 'r2':
            ss_res = np.sum((X_true - X_pred)**2)
            ss_tot = np.sum((X_true - np.mean(X_true))**2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
        elif metric == 'logloss':
            # For NMF, we use a modified version of log loss
            epsilon = 1e-8
            metrics['logloss'] = -np.mean(X_true * np.log(X_pred + epsilon))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        metrics['custom_metric'] = metric(X_true, X_pred)

    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(X_true, X_pred)

    return metrics
