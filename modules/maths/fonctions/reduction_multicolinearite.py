"""
Quantix – Module reduction_multicolinearite
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# variance_inflation_factor
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def variance_inflation_factor_fit(
    X: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    reg_param: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Compute Variance Inflation Factor (VIF) for each feature in the dataset.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the data. Default is identity function.
    metric : str
        Metric used for evaluation ('mse', 'mae', 'r2'). Default is 'mse'.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function if needed. Overrides `metric` parameter.
    solver : str
        Solver to use ('closed_form', 'gradient_descent'). Default is 'closed_form'.
    regularization : Optional[str]
        Regularization type ('l1', 'l2'). Default is None.
    reg_param : float
        Regularization parameter. Default is 1.0.
    tol : float
        Tolerance for convergence. Default is 1e-6.
    max_iter : int
        Maximum number of iterations. Default is 1000.
    random_state : Optional[int]
        Random seed for reproducibility. Default is None.

    Returns:
    --------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]
        Dictionary containing:
        - "result": VIF values for each feature
        - "metrics": Computed metrics
        - "params_used": Parameters used in the computation
        - "warnings": Any warnings encountered

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> result = variance_inflation_factor_fit(X)
    """
    # Validate inputs
    _validate_inputs(X, normalizer)

    # Normalize data
    X_norm = normalizer(X)

    # Initialize results dictionary
    results = {
        "result": np.zeros(X.shape[1]),
        "metrics": {},
        "params_used": {
            "normalizer": normalizer.__name__,
            "metric": metric,
            "solver": solver,
            "regularization": regularization,
            "reg_param": reg_param
        },
        "warnings": []
    }

    # Compute VIF for each feature
    for i in range(X.shape[1]):
        try:
            vif = _compute_vif(
                X_norm,
                i,
                metric=metric,
                custom_metric=custom_metric,
                solver=solver,
                regularization=regularization,
                reg_param=reg_param,
                tol=tol,
                max_iter=max_iter
            )
            results["result"][i] = vif
        except Exception as e:
            results["warnings"].append(f"Feature {i} failed: {str(e)}")
            results["result"][i] = np.nan

    return results

def _validate_inputs(X: np.ndarray, normalizer: Callable[[np.ndarray], np.ndarray]) -> None:
    """Validate input data and normalizer function."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("X contains infinite values")

    # Test normalizer function
    try:
        test_data = np.array([[1, 2], [3, 4]])
        normalizer(test_data)
    except Exception as e:
        raise ValueError(f"Normalizer function failed: {str(e)}")

def _compute_vif(
    X: np.ndarray,
    feature_idx: int,
    metric: str = 'mse',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    reg_param: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> float:
    """
    Compute VIF for a specific feature.

    Parameters:
    -----------
    X : np.ndarray
        Normalized input data matrix
    feature_idx : int
        Index of the feature to compute VIF for
    metric : str
        Metric used for evaluation
    custom_metric : Optional[Callable]
        Custom metric function
    solver : str
        Solver to use
    regularization : Optional[str]
        Regularization type
    reg_param : float
        Regularization parameter
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations

    Returns:
    --------
    float
        VIF value for the specified feature
    """
    # Get target and features
    y = X[:, feature_idx]
    X_regressors = np.delete(X, feature_idx, axis=1)

    # Choose solver
    if solver == 'closed_form':
        coefs = _closed_form_solver(X_regressors, y, regularization, reg_param)
    elif solver == 'gradient_descent':
        coefs = _gradient_descent_solver(
            X_regressors, y, metric, custom_metric,
            regularization, reg_param, tol, max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate R-squared
    r_squared = _calculate_r_squared(X_regressors, y, coefs)

    # Compute VIF
    vif = 1.0 / (1.0 - r_squared)

    return float(vif)

def _closed_form_solver(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    reg_param: float
) -> np.ndarray:
    """Closed form solution for linear regression."""
    if regularization == 'l2':
        # Ridge regression
        n_features = X.shape[1]
        identity = np.eye(n_features)
        XTX = X.T @ X
        coefs = np.linalg.inv(XTX + reg_param * identity) @ X.T @ y
    elif regularization == 'l1':
        # Lasso regression (simplified version)
        raise NotImplementedError("L1 regularization not implemented in closed form")
    else:
        # Ordinary least squares
        coefs = np.linalg.inv(X.T @ X) @ X.T @ y

    return coefs

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable],
    regularization: Optional[str],
    reg_param: float,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent solver for linear regression."""
    n_features = X.shape[1]
    coefs = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        gradients = _compute_gradients(X, y, coefs, metric, custom_metric, regularization, reg_param)
        coefs -= gradients

        current_loss = _compute_loss(X, y, coefs, metric, custom_metric)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return coefs

def _compute_gradients(
    X: np.ndarray,
    y: np.ndarray,
    coefs: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable],
    regularization: Optional[str],
    reg_param: float
) -> np.ndarray:
    """Compute gradients for gradient descent."""
    predictions = X @ coefs

    if custom_metric is not None:
        # Use custom metric
        gradient = -X.T @ (custom_metric(y, predictions) * (y - predictions)) / len(y)
    elif metric == 'mse':
        gradient = -X.T @ (y - predictions) / len(y)
    elif metric == 'mae':
        gradient = -X.T @ np.sign(y - predictions) / len(y)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Add regularization term
    if regularization == 'l2':
        gradient += 2 * reg_param * coefs
    elif regularization == 'l1':
        gradient += reg_param * np.sign(coefs)

    return gradient

def _compute_loss(
    X: np.ndarray,
    y: np.ndarray,
    coefs: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable]
) -> float:
    """Compute loss based on specified metric."""
    predictions = X @ coefs

    if custom_metric is not None:
        return custom_metric(y, predictions)
    elif metric == 'mse':
        return np.mean((y - predictions) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y - predictions))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _calculate_r_squared(
    X: np.ndarray,
    y: np.ndarray,
    coefs: np.ndarray
) -> float:
    """Calculate R-squared value."""
    predictions = X @ coefs
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        return 1.0

    return 1 - (ss_res / ss_tot)

################################################################################
# correlation_matrix
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(X: np.ndarray) -> None:
    """Validate input matrix dimensions and types."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input contains NaN or infinite values")

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

def compute_correlation_matrix(
    X: np.ndarray,
    method: str = "pearson",
    custom_correlation_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> np.ndarray:
    """Compute correlation matrix using specified method or custom function."""
    if custom_correlation_func is not None:
        n = X.shape[0]
        corr_matrix = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                corr = custom_correlation_func(X[:, i], X[:, j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        return corr_matrix

    if method == "pearson":
        cov_matrix = np.cov(X, rowvar=False)
        stddev = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(stddev, stddev)
    elif method == "spearman":
        ranks = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), axis=0, arr=X)
        corr_matrix = compute_correlation_matrix(ranks, method="pearson")
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    return corr_matrix

def compute_metrics(
    corr_matrix: np.ndarray,
    metric_functions: Dict[str, Callable[[np.ndarray], float]],
    custom_metrics: Optional[Dict[str, Callable[[np.ndarray], float]]] = None
) -> Dict[str, float]:
    """Compute specified metrics on correlation matrix."""
    metrics = {}
    for name, func in metric_functions.items():
        metrics[name] = func(corr_matrix)

    if custom_metrics is not None:
        for name, func in custom_metrics.items():
            metrics[name] = func(corr_matrix)

    return metrics

def correlation_matrix_fit(
    X: np.ndarray,
    normalize_method: str = "standard",
    corr_method: str = "pearson",
    metric_functions: Optional[Dict[str, Callable[[np.ndarray], float]]] = None,
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_correlation_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_metrics: Optional[Dict[str, Callable[[np.ndarray], float]]] = None
) -> Dict:
    """Compute correlation matrix with various options."""
    # Validate input
    validate_input(X)

    # Default metrics if none provided
    default_metrics = {
        "max_correlation": lambda m: np.max(np.abs(m - np.eye(m.shape[0]))),
        "mean_correlation": lambda m: np.mean(np.abs(m - np.eye(m.shape[0])))
    }
    if metric_functions is None:
        metric_functions = default_metrics
    else:
        metric_functions.update(default_metrics)

    # Normalize data
    X_norm = normalize_data(X, method=normalize_method, custom_normalizer=custom_normalizer)

    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(
        X_norm,
        method=corr_method,
        custom_correlation_func=custom_correlation_func
    )

    # Compute metrics
    metrics = compute_metrics(corr_matrix, metric_functions, custom_metrics)

    # Prepare output
    result = {
        "result": corr_matrix,
        "metrics": metrics,
        "params_used": {
            "normalize_method": normalize_method,
            "corr_method": corr_method
        },
        "warnings": []
    }

    return result

# Example usage:
"""
X = np.random.randn(100, 5)
result = correlation_matrix_fit(X, normalize_method="standard", corr_method="pearson")
print(result)
"""

################################################################################
# principal_component_analysis
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2-dimensional array.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input X must not contain NaN or infinite values.")

def _normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
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
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_covariance_matrix(X: np.ndarray) -> np.ndarray:
    """Compute covariance matrix."""
    return np.cov(X, rowvar=False)

def _eig_decomposition(cov_matrix: np.ndarray) -> tuple:
    """Perform eigenvalue decomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    return eigenvalues, eigenvectors

def _select_components(eigenvalues: np.ndarray,
                       eigenvectors: np.ndarray,
                       n_components: int) -> tuple:
    """Select principal components."""
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][:n_components]
    eigenvectors = eigenvectors[:, idx][:, :n_components]
    return eigenvalues, eigenvectors

def principal_component_analysis_fit(X: np.ndarray,
                                    n_components: int = 2,
                                    normalization: str = 'standard',
                                    solver: str = 'closed_form') -> Dict:
    """Perform Principal Component Analysis (PCA).

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of principal components to keep.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    solver : str, optional
        Solver method ('closed_form').

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_input(X)
    X_normalized = _normalize_data(X, method=normalization)
    cov_matrix = _compute_covariance_matrix(X_normalized)
    eigenvalues, eigenvectors = _eig_decomposition(cov_matrix)
    eigenvalues, eigenvectors = _select_components(eigenvalues, eigenvectors, n_components)

    result = {
        'components': eigenvectors,
        'explained_variance': eigenvalues,
        'explained_variance_ratio': eigenvalues / np.sum(eigenvalues)
    }

    metrics = {
        'total_variance': np.sum(eigenvalues),
        'cumulative_variance_ratio': np.cumsum(eigenvalues) / np.sum(eigenvalues)
    }

    params_used = {
        'n_components': n_components,
        'normalization': normalization,
        'solver': solver
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# factor_analysis
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def factor_analysis_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    solver: str = 'closed_form',
    max_iter: int = 1000,
    tol: float = 1e-6,
    random_state: Optional[int] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Perform factor analysis to reduce multicollinearity in the data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of factors to extract, by default 2.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the data, by default None.
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent'), by default 'closed_form'.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    tol : float, optional
        Tolerance for convergence, by default 1e-6.
    random_state : Optional[int], optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Estimated factors and loadings
        - 'metrics': Performance metrics
        - 'params_used': Parameters used in the fit
        - 'warnings': Any warnings encountered

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> result = factor_analysis_fit(X, n_components=3)
    """
    # Validate inputs
    X = _validate_input(X)

    # Normalize data if specified
    if normalizer is not None:
        X = normalizer(X)

    # Initialize parameters
    params_used = {
        'n_components': n_components,
        'solver': solver,
        'max_iter': max_iter,
        'tol': tol
    }

    # Choose solver
    if solver == 'closed_form':
        factors, loadings = _closed_form_solution(X, n_components)
    elif solver == 'gradient_descent':
        factors, loadings = _gradient_descent_solution(
            X, n_components, max_iter, tol, random_state
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(X, factors, loadings)

    return {
        'result': {'factors': factors, 'loadings': loadings},
        'metrics': metrics,
        'params_used': params_used,
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

def _closed_form_solution(X: np.ndarray, n_components: int) -> tuple:
    """Closed form solution for factor analysis."""
    # Calculate covariance matrix
    cov = np.cov(X, rowvar=False)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top n_components
    loadings = eigenvectors[:, :n_components]
    factors = X @ loadings

    return factors, loadings

def _gradient_descent_solution(
    X: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float,
    random_state: Optional[int]
) -> tuple:
    """Gradient descent solution for factor analysis."""
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    loadings = np.random.randn(n_features, n_components)

    for _ in range(max_iter):
        # Update factors
        factors = X @ loadings

        # Update loadings via gradient descent
        grad = 2 * (X.T @ factors - n_samples * loadings)
        loadings -= tol * grad

    return factors, loadings

def _calculate_metrics(
    X: np.ndarray,
    factors: np.ndarray,
    loadings: np.ndarray
) -> Dict[str, float]:
    """Calculate performance metrics."""
    reconstructed = factors @ loadings.T
    mse = np.mean((X - reconstructed) ** 2)
    r2 = 1 - mse / np.var(X, ddof=1)

    return {
        'mse': float(mse),
        'r2': float(r2)
    }

################################################################################
# ridge_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or Inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or Inf values")

def _normalize_data(X: np.ndarray, y: np.ndarray,
                   normalization: str = 'standard',
                   custom_normalize: Optional[Callable] = None) -> tuple:
    """Normalize data according to specified method."""
    if custom_normalize is not None:
        X_norm, y_norm = custom_normalize(X, y)
    elif normalization == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif normalization == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == 'robust':
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        X_norm, y_norm = X, y
    return X_norm, y_norm

def _compute_ridge_solution(X: np.ndarray, y: np.ndarray,
                          alpha: float = 1.0) -> np.ndarray:
    """Compute ridge regression solution using closed form."""
    I = np.eye(X.shape[1])
    return np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric: str = 'mse',
                    custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute specified metrics."""
    if custom_metric is not None:
        return {'custom': custom_metric(y_true, y_pred)}

    metrics = {}
    if metric == 'mse' or 'all' in metric:
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    if metric == 'mae' or 'all' in metric:
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric == 'r2' or 'all' in metric:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

def ridge_regression_fit(X: np.ndarray, y: np.ndarray,
                       alpha: float = 1.0,
                       normalization: str = 'standard',
                       metric: Union[str, Callable] = 'mse',
                       custom_normalize: Optional[Callable] = None,
                       custom_metric: Optional[Callable] = None) -> Dict:
    """
    Perform ridge regression with configurable options.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    alpha : float, optional
        Regularization strength, by default 1.0
    normalization : str or Callable, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), by default 'standard'
    metric : str or Callable, optional
        Metric to compute ('mse', 'mae', 'r2', 'all'), by default 'mse'
    custom_normalize : Callable, optional
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
    >>> result = ridge_regression_fit(X, y, alpha=0.5, normalization='standard')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization, custom_normalize)

    # Compute solution
    coef = _compute_ridge_solution(X_norm, y_norm, alpha)

    # Make predictions
    y_pred = X_norm @ coef

    # Compute metrics
    if isinstance(metric, str):
        metrics = _compute_metrics(y_norm, y_pred, metric)
    else:
        metrics = {'custom': metric(y_norm, y_pred)}

    return {
        'result': {
            'coefficients': coef,
            'intercept': 0,  # Ridge regression with centered data
            'predictions': y_pred
        },
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'normalization': normalization if custom_normalize is None else 'custom',
            'metric': metric
        },
        'warnings': []
    }

################################################################################
# lasso_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input matrices for Lasso regression."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be a 2D array and y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def normalize_data(X: np.ndarray, y: np.ndarray,
                  normalization: str = 'standard',
                  custom_normalize: Optional[Callable] = None) -> tuple:
    """Normalize input data based on specified method."""
    if custom_normalize is not None:
        X_norm, y_norm = custom_normalize(X, y)
    elif normalization == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif normalization == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == 'robust':
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        X_norm, y_norm = X, y
    return X_norm, y_norm

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   metric: str = 'mse',
                   custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute specified metrics between true and predicted values."""
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)
    if metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    return metrics

def lasso_regression_fit(X: np.ndarray, y: np.ndarray,
                        alpha: float = 1.0,
                        max_iter: int = 1000,
                        tol: float = 1e-4,
                        solver: str = 'coordinate_descent',
                        normalization: str = 'standard',
                        metric: str = 'mse',
                        custom_normalize: Optional[Callable] = None,
                        custom_metric: Optional[Callable] = None) -> Dict:
    """Fit Lasso regression model with specified parameters."""
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization, custom_normalize)

    # Initialize coefficients
    n_features = X_norm.shape[1]
    coef = np.zeros(n_features)

    # Solve using specified solver
    if solver == 'coordinate_descent':
        coef = coordinate_descent_solver(X_norm, y_norm, alpha, max_iter, tol)
    else:
        raise ValueError(f"Solver {solver} not implemented")

    # Predict and compute metrics
    y_pred = X_norm @ coef
    metrics = compute_metrics(y_norm, y_pred, metric, custom_metric)

    # Return results
    return {
        'result': {'coefficients': coef},
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'max_iter': max_iter,
            'tol': tol,
            'solver': solver,
            'normalization': normalization,
            'metric': metric
        },
        'warnings': []
    }

def coordinate_descent_solver(X: np.ndarray, y: np.ndarray,
                            alpha: float, max_iter: int, tol: float) -> np.ndarray:
    """Coordinate descent solver for Lasso regression."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    for _ in range(max_iter):
        old_coef = coef.copy()
        for j in range(n_features):
            # Compute residuals without feature j
            r = y - X @ coef + X[:, j] * coef[j]

            # Compute correlation
            corr = np.dot(X[:, j], r) / n_samples

            if corr < -alpha/2:
                coef[j] = (corr - alpha/2)
            elif corr > alpha/2:
                coef[j] = (corr + alpha/2)
            else:
                coef[j] = 0

        # Check convergence
        if np.linalg.norm(coef - old_coef) < tol:
            break
    return coef

# Example usage:
"""
X = np.random.rand(100, 5)
y = X @ np.array([2.0, -1.5, 0.8, 0.3, -0.2]) + np.random.normal(0, 0.1, 100)

result = lasso_regression_fit(X, y,
                             alpha=0.5,
                             solver='coordinate_descent',
                             normalization='standard')
"""

################################################################################
# elastic_net
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def elastic_net_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    normalize: str = 'standard',
    solver: str = 'coordinate_descent',
    max_iter: int = 1000,
    tol: float = 1e-4,
    metric: str = 'mse',
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Fit an Elastic Net model to reduce multicollinearity in the features.

    Parameters:
    -----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    alpha : float, default=1.0
        Regularization strength.
    l1_ratio : float, default=0.5
        Mixing parameter between L1 and L2 regularization.
    normalize : str, default='standard'
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    solver : str, default='coordinate_descent'
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    metric : str, default='mse'
        Evaluation metric: 'mse', 'mae', 'r2', or custom callable.
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    custom_metric : Optional[Callable], default=None
        Custom metric function.

    Returns:
    --------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]
        Dictionary containing:
        - 'result': Coefficients array.
        - 'metrics': Evaluation metrics dictionary.
        - 'params_used': Parameters used in the fitting process.
        - 'warnings': Warnings dictionary.

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = elastic_net_fit(X, y, alpha=0.5, l1_ratio=0.3)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _normalize_data(X, method=normalize)

    # Initialize coefficients
    n_features = X_normalized.shape[1]
    coefs = np.zeros(n_features)

    # Choose solver
    if solver == 'closed_form':
        coefs = _closed_form_solution(X_normalized, y, alpha, l1_ratio)
    elif solver == 'gradient_descent':
        coefs = _gradient_descent(X_normalized, y, alpha, l1_ratio,
                                 max_iter=max_iter, tol=tol,
                                 random_state=random_state)
    elif solver == 'newton':
        coefs = _newton_method(X_normalized, y, alpha, l1_ratio,
                              max_iter=max_iter, tol=tol)
    elif solver == 'coordinate_descent':
        coefs = _coordinate_descent(X_normalized, y, alpha, l1_ratio,
                                   max_iter=max_iter, tol=tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    y_pred = X_normalized @ coefs
    metrics = _calculate_metrics(y, y_pred, metric=metric,
                                custom_metric=custom_metric)

    # Prepare output
    result = {
        'result': coefs,
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'normalize': normalize,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': {}
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
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
        raise ValueError(f"Unknown normalization method: {method}")

def _closed_form_solution(X: np.ndarray, y: np.ndarray,
                         alpha: float, l1_ratio: float) -> np.ndarray:
    """Closed form solution for Elastic Net."""
    # This is a placeholder - actual implementation would need to handle
    # the non-differentiable L1 penalty properly
    XtX = X.T @ X
    Xty = X.T @ y

    # Add L2 penalty
    l2_penalty = alpha * (1 - l1_ratio) * np.eye(X.shape[1])
    XtX += l2_penalty

    # Solve linear system
    coefs = np.linalg.solve(XtX, Xty)

    return coefs

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Gradient descent solver for Elastic Net."""
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    coefs = np.random.randn(n_features)

    for _ in range(max_iter):
        gradient = X.T @ (X @ coefs - y) / n_samples

        # Add L1 and L2 penalties
        l1_penalty = alpha * l1_ratio * np.sign(coefs)
        l2_penalty = 2 * alpha * (1 - l1_ratio) * coefs
        gradient += l1_penalty + l2_penalty

        # Update coefficients
        old_coefs = coefs.copy()
        coefs -= gradient

        # Check convergence
        if np.linalg.norm(coefs - old_coefs) < tol:
            break

    return coefs

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> np.ndarray:
    """Newton method solver for Elastic Net."""
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)

    for _ in range(max_iter):
        # Compute gradient
        gradient = X.T @ (X @ coefs - y) / n_samples

        # Add L1 and L2 penalties
        l1_penalty = alpha * l1_ratio * np.sign(coefs)
        l2_penalty = 2 * alpha * (1 - l1_ratio) * coefs
        gradient += l1_penalty + l2_penalty

        # Compute Hessian (approximate)
        hessian = 2 * X.T @ X / n_samples + 2 * alpha * (1 - l1_ratio) * np.eye(n_features)

        # Update coefficients
        old_coefs = coefs.copy()
        if np.linalg.det(hessian) != 0:
            coefs -= np.linalg.solve(hessian, gradient)

        # Check convergence
        if np.linalg.norm(coefs - old_coefs) < tol:
            break

    return coefs

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> np.ndarray:
    """Coordinate descent solver for Elastic Net."""
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)

    for _ in range(max_iter):
        old_coefs = coefs.copy()

        for j in range(n_features):
            # Compute residual without feature j
            r = y - X @ coefs + X[:, j] * coefs[j]

            # Compute correlation
            corr = X[:, j].T @ r

            # Compute step size
            xj_sq = np.sum(X[:, j] ** 2)
            step_size = corr / (xj_sq + alpha * (1 - l1_ratio) + 1e-8)

            # Soft-thresholding for L1 penalty
            l1_penalty = alpha * l1_ratio
            coefs[j] = np.sign(step_size) * max(abs(step_size) - l1_penalty, 0)

        # Check convergence
        if np.linalg.norm(coefs - old_coefs) < tol:
            break

    return coefs

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = 'mse',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    metrics = {}

    if metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - ss_res / (ss_tot + 1e-8)
    elif custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

################################################################################
# partial_least_squares
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input matrices for PLS regression."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be a 2D array and y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _normalize_data(X: np.ndarray, y: np.ndarray,
                   X_norm_type: str = 'standard',
                   y_norm_type: str = 'standard') -> tuple:
    """Normalize input data according to specified methods."""
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)

    y_mean = np.mean(y)
    y_std = np.std(y)

    if X_norm_type == 'standard':
        X_normalized = (X - X_mean) / (X_std + 1e-8)
    elif X_norm_type == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_normalized = (X - X_min) / ((X_max - X_min) + 1e-8)
    elif X_norm_type == 'robust':
        X_median = np.median(X, axis=0)
        X_q75, X_q25 = np.percentile(X, [75, 25], axis=0)
        X_iqr = X_q75 - X_q25
        X_normalized = (X - X_median) / (X_iqr + 1e-8)
    else:
        X_normalized = X.copy()

    if y_norm_type == 'standard':
        y_normalized = (y - y_mean) / (y_std + 1e-8)
    elif y_norm_type == 'minmax':
        y_min = np.min(y)
        y_max = np.max(y)
        y_normalized = (y - y_min) / ((y_max - y_min) + 1e-8)
    elif y_norm_type == 'robust':
        y_median = np.median(y)
        y_q75, y_q25 = np.percentile(y, [75, 25])
        y_iqr = y_q75 - y_q25
        y_normalized = (y - y_median) / (y_iqr + 1e-8)
    else:
        y_normalized = y.copy()

    return X_normalized, y_normalized

def _compute_pls_components(X: np.ndarray, y: np.ndarray,
                          n_components: int = 2) -> tuple:
    """Compute PLS components using NIPALS algorithm."""
    X_residual = X.copy()
    y_residual = y.copy()

    W = np.zeros((X.shape[1], n_components))
    P = np.zeros((X.shape[1], n_components))
    Q = np.zeros((n_components, 1))

    for i in range(n_components):
        # Step 1: Compute weight vector w
        cov = np.cov(X_residual.T, y_residual, bias=True)
        w = cov[:X.shape[1], X.shape[1]:].sum(axis=1)
        w = w / np.linalg.norm(w)

        # Step 2: Compute score vector t
        t = X_residual @ w

        # Step 3: Compute loading vectors p and q
        p = (X_residual.T @ t) / (t.T @ t)
        q = (y_residual.T @ t) / (t.T @ t)

        # Step 4: Update residuals
        X_residual -= t[:, np.newaxis] @ p[np.newaxis, :]
        y_residual -= t @ q.T

        W[:, i] = w
        P[:, i] = p.flatten()
        Q[i, 0] = q[0]

    return W, P, Q

def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Calculate specified metrics between true and predicted values."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(y_true, y_pred)
        except Exception as e:
            metrics[f"{name}_error"] = str(e)

    return metrics

def partial_least_squares_fit(X: np.ndarray, y: np.ndarray,
                            n_components: int = 2,
                            X_norm_type: str = 'standard',
                            y_norm_type: str = 'standard',
                            metric_funcs: Optional[Dict[str, Callable]] = None,
                            custom_metric: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Perform Partial Least Squares (PLS) regression.

    Parameters:
    -----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    n_components : int, optional
        Number of PLS components to compute (default: 2)
    X_norm_type : str, optional
        Normalization type for X ('none', 'standard', 'minmax', 'robust')
    y_norm_type : str, optional
        Normalization type for y ('none', 'standard', 'minmax', 'robust')
    metric_funcs : dict, optional
        Dictionary of metric functions to compute (default: None)
    custom_metric : callable, optional
        Custom metric function to compute (default: None)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = partial_least_squares_fit(X, y, n_components=3)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize default metrics if none provided
    if metric_funcs is None:
        def mse(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        def r2(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)

        metric_funcs = {
            'mse': mse,
            'r2': r2
        }

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, X_norm_type, y_norm_type)

    # Compute PLS components
    W, P, Q = _compute_pls_components(X_norm, y_norm, n_components)

    # Calculate predictions
    T = X_norm @ W
    y_pred = T @ Q

    # Calculate metrics
    metrics = _calculate_metrics(y, y_pred, metric_funcs)

    # Add custom metric if provided
    if custom_metric is not None:
        try:
            metrics['custom'] = custom_metric(y, y_pred)
        except Exception as e:
            metrics['custom_error'] = str(e)

    # Prepare results dictionary
    result = {
        'result': {
            'weights': W,
            'loadings_X': P,
            'loadings_y': Q.flatten(),
            'scores': T
        },
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'X_norm_type': X_norm_type,
            'y_norm_type': y_norm_type
        },
        'warnings': []
    }

    return result

################################################################################
# forward_selection
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_input(X: np.ndarray) -> None:
    """Validate input matrix X."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X must not contain NaN or infinite values")

def normalize_data(
    X: np.ndarray,
    method: str = "standard",
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize the input data."""
    if custom_normalizer is not None:
        return custom_normalizer(X)

    if method == "none":
        return X
    elif method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "mse",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> float:
    """Compute the specified metric."""
    if custom_metric is not None:
        return custom_metric(y_true, y_pred)

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

def forward_selection_fit(
    X: np.ndarray,
    y: np.ndarray,
    metric: str = "mse",
    normalizer: str = "standard",
    max_features: int = None,
    tol: float = 1e-4,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Perform forward selection to reduce multicollinearity.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - metric: Metric to optimize ("mse", "mae", "r2", "logloss")
    - normalizer: Normalization method ("none", "standard", "minmax", "robust")
    - max_features: Maximum number of features to select
    - tol: Tolerance for stopping criterion
    - custom_metric: Custom metric function
    - custom_normalizer: Custom normalizer function

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    validate_input(X)
    if y.ndim != 1:
        raise ValueError("y must be a 1-dimensional array")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")

    X_norm = normalize_data(X, method=normalizer, custom_normalizer=custom_normalizer)
    n_features = X_norm.shape[1]
    if max_features is None:
        max_features = n_features
    elif max_features > n_features:
        raise ValueError("max_features cannot be greater than the number of features")

    selected_features = []
    best_metric = float('inf')
    if metric in ["mse", "mae"]:
        best_metric = float('inf')
    else:
        best_metric = -float('inf')

    for _ in range(max_features):
        metrics = []
        for feature_idx in range(n_features):
            if feature_idx not in selected_features:
                current_features = selected_features + [feature_idx]
                X_current = X_norm[:, current_features]

                # Simple linear regression for demonstration
                beta = np.linalg.pinv(X_current) @ y
                y_pred = X_current @ beta

                current_metric = compute_metric(y, y_pred, metric, custom_metric)
                metrics.append((feature_idx, current_metric))

        if not metrics:
            break

        _, current_best_metric = min(metrics, key=lambda x: x[1]) if metric in ["mse", "mae"] else max(metrics, key=lambda x: x[1])
        best_feature_idx = [x[0] for x in metrics if x[1] == current_best_metric][0]

        if (metric in ["mse", "mae"] and current_best_metric >= best_metric - tol) or \
           (metric in ["r2", "logloss"] and current_best_metric <= best_metric + tol):
            break

        selected_features.append(best_feature_idx)
        best_metric = current_best_metric

    result = {
        "selected_features": selected_features,
        "metrics": {"best_metric": best_metric},
        "params_used": {
            "metric": metric,
            "normalizer": normalizer,
            "max_features": max_features,
            "tol": tol
        },
        "warnings": []
    }

    return result

# Example usage:
# X = np.random.rand(100, 5)
# y = np.random.rand(100)
# result = forward_selection_fit(X, y, metric="r2", normalizer="standard")

################################################################################
# backward_elimination
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
    CUSTOM = "custom"

class Solver(Enum):
    OLS = "ols"
    GRADIENT_DESCENT = "gradient_descent"

def validate_input(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """Validate input matrices."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y is not None and y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or (y is not None and np.any(np.isnan(y))):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or (y is not None and np.any(np.isinf(y))):
        raise ValueError("Input arrays must not contain infinite values")

def normalize_data(X: np.ndarray, method: Normalization) -> np.ndarray:
    """Normalize data according to specified method."""
    if method == Normalization.NONE:
        return X
    elif method == Normalization.STANDARD:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == Normalization.MINMAX:
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif method == Normalization.ROBUST:
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def calculate_metric(y_true: np.ndarray, y_pred: np.ndarray,
                    metric: Union[Metric, Callable]) -> float:
    """Calculate specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == Metric.MSE:
        return np.mean((y_true - y_pred) ** 2)
    elif metric == Metric.MAE:
        return np.mean(np.abs(y_true - y_pred))
    elif metric == Metric.R2:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def backward_elimination_fit(X: np.ndarray, y: np.ndarray,
                           metric: Union[Metric, Callable] = Metric.R2,
                           solver: Solver = Solver.OLS,
                           normalization: Normalization = Normalization.NONE,
                           threshold: float = 0.01,
                           max_iter: int = None) -> Dict:
    """Perform backward elimination to reduce multicollinearity.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        metric: Metric to optimize or custom callable
        solver: Solver to use for regression
        normalization: Normalization method to apply
        threshold: Threshold for feature elimination
        max_iter: Maximum number of iterations

    Returns:
        Dictionary containing results, metrics, parameters used and warnings
    """
    validate_input(X, y)

    # Initialize variables
    X_normalized = normalize_data(X.copy(), normalization)
    selected_features = list(range(X.shape[1]))
    best_metric = -np.inf
    results = []
    warnings_list = []

    # Initial fit with all features
    current_model = _fit_regression(X_normalized[:, selected_features], y, solver)
    current_metric = calculate_metric(y, current_model.predict(X_normalized[:, selected_features]), metric)
    best_metric = current_metric
    results.append({
        'features': selected_features.copy(),
        'metric': current_metric,
        'model': current_model
    })

    # Iterative feature elimination
    iteration = 0
    while len(selected_features) > 1 and (max_iter is None or iteration < max_iter):
        iteration += 1
        metrics = []
        temp_features = selected_features.copy()

        for i in range(len(temp_features)):
            # Temporarily remove feature
            temp_features.remove(temp_features[i])
            model = _fit_regression(X_normalized[:, temp_features], y, solver)
            metrics.append(calculate_metric(y, model.predict(X_normalized[:, temp_features]), metric))
            temp_features.insert(i, temp_features[i])  # Put back the feature

        # Find best feature to remove
        best_feature_to_remove = np.argmin(metrics)
        new_metric = metrics[best_feature_to_remove]

        if best_metric - new_metric < threshold:
            break

        # Update results
        selected_features.remove(selected_features[best_feature_to_remove])
        model = _fit_regression(X_normalized[:, selected_features], y, solver)
        current_metric = calculate_metric(y, model.predict(X_normalized[:, selected_features]), metric)
        best_metric = current_metric

        results.append({
            'features': selected_features.copy(),
            'metric': current_metric,
            'model': model
        })

    # Prepare output
    output = {
        'result': results[-1],
        'metrics': [r['metric'] for r in results],
        'params_used': {
            'normalization': normalization.value,
            'metric': metric.value if isinstance(metric, Metric) else "custom",
            'solver': solver.value,
            'threshold': threshold,
            'max_iter': max_iter
        },
        'warnings': warnings_list
    }

    return output

class BaseModel:
    """Base class for regression models."""
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model not fitted yet")
        return X.dot(self.coef_) + self.intercept_

def _fit_regression(X: np.ndarray, y: np.ndarray,
                   solver: Solver) -> BaseModel:
    """Fit regression model using specified solver."""
    if solver == Solver.OLS:
        return _fit_ols(X, y)
    elif solver == Solver.GRADIENT_DESCENT:
        return _fit_gradient_descent(X, y)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _fit_ols(X: np.ndarray, y: np.ndarray) -> BaseModel:
    """Fit OLS regression model."""
    model = BaseModel()
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    coef, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
    model.intercept_ = coef[0]
    model.coef_ = coef[1:]
    return model

def _fit_gradient_descent(X: np.ndarray, y: np.ndarray,
                         learning_rate: float = 0.01,
                         n_iter: int = 1000) -> BaseModel:
    """Fit regression model using gradient descent."""
    model = BaseModel()
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    intercept = 0

    for _ in range(n_iter):
        y_pred = X.dot(coef) + intercept
        error = y_pred - y

        # Update coefficients
        coef -= learning_rate * (2/n_samples) * X.T.dot(error)
        intercept -= learning_rate * (2/n_samples) * np.sum(error)

    model.coef_ = coef
    model.intercept_ = intercept
    return model

################################################################################
# stepwise_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def stepwise_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    direction: str = 'both',
    criterion: str = 'aic',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'r2',
    max_features: Optional[int] = None,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Perform stepwise regression for feature selection.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    direction : str, optional
        Direction of stepwise selection ('forward', 'backward', or 'both')
    criterion : str, optional
        Criterion for feature selection ('aic', 'bic', or custom callable)
    normalizer : Callable, optional
        Function to normalize features (e.g., StandardScaler)
    metric : str or Callable, optional
        Metric for model evaluation ('mse', 'mae', 'r2', or custom callable)
    max_features : int, optional
        Maximum number of features to select
    tol : float, optional
        Tolerance for stopping criterion
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'selected_features': Indices of selected features
        - 'coefficients': Model coefficients
        - 'metrics': Evaluation metrics
        - 'params_used': Parameters used in the fit
        - 'warnings': Any warnings encountered

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = stepwise_regression_fit(X, y, direction='forward', criterion='aic')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize parameters
    params_used = {
        'direction': direction,
        'criterion': criterion,
        'normalizer': normalizer.__name__ if normalizer else None,
        'metric': metric if isinstance(metric, str) else 'custom',
        'max_features': max_features,
        'tol': tol,
        'random_state': random_state
    }

    # Normalize data if specified
    X_norm = _apply_normalization(X, normalizer)

    # Initialize results
    selected_features = np.array([], dtype=int)
    coefficients = np.zeros(X.shape[1])
    best_metric = -np.inf if isinstance(metric, str) and metric in ['r2'] else np.inf
    warnings = []

    # Set random state if specified
    if random_state is not None:
        np.random.seed(random_state)

    # Stepwise selection
    if direction in ['forward', 'both']:
        selected_features = _forward_selection(
            X_norm, y, criterion, metric,
            max_features, tol, warnings
        )

    if direction in ['backward', 'both'] and (max_features is None or len(selected_features) < max_features):
        selected_features = _backward_selection(
            X_norm, y, criterion, metric,
            max_features, tol, warnings
        )

    # Fit final model with selected features
    if len(selected_features) > 0:
        X_selected = X_norm[:, selected_features]
        coefficients[selected_features], metrics = _fit_model(X_selected, y, metric)
    else:
        metrics = {'metric': np.nan}

    return {
        'selected_features': selected_features,
        'coefficients': coefficients,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _forward_selection(
    X: np.ndarray,
    y: np.ndarray,
    criterion: Union[str, Callable],
    metric: Union[str, Callable],
    max_features: Optional[int],
    tol: float,
    warnings: list
) -> np.ndarray:
    """Perform forward stepwise selection."""
    n_features = X.shape[1]
    selected_features = np.array([], dtype=int)
    best_criterion_value = float('inf')

    for _ in range(n_features):
        if max_features is not None and len(selected_features) >= max_features:
            break

        candidate_features = np.setdiff1d(np.arange(n_features), selected_features)
        best_feature = None
        best_value = float('inf')

        for feature in candidate_features:
            current_features = np.concatenate([selected_features, [feature]])
            X_current = X[:, current_features]

            try:
                coefs, current_metric = _fit_model(X_current, y, metric)
            except Exception as e:
                warnings.append(f"Feature {feature} failed: {str(e)}")
                continue

            if isinstance(criterion, str):
                current_value = _calculate_criterion(X_current, y, coefs, criterion)
            else:
                current_value = criterion(coefs)

            if current_value < best_value - tol:
                best_feature = feature
                best_value = current_value

        if best_feature is None:
            break

        selected_features = np.append(selected_features, best_feature)
        best_criterion_value = best_value

    return selected_features

def _backward_selection(
    X: np.ndarray,
    y: np.ndarray,
    criterion: Union[str, Callable],
    metric: Union[str, Callable],
    max_features: Optional[int],
    tol: float,
    warnings: list
) -> np.ndarray:
    """Perform backward stepwise selection."""
    selected_features = np.arange(X.shape[1])
    best_criterion_value = float('inf')

    while len(selected_features) > 0:
        if max_features is not None and len(selected_features) <= max_features:
            break

        candidate_features = selected_features.copy()
        best_feature = None
        best_value = float('inf')

        for feature in candidate_features:
            current_features = np.setdiff1d(selected_features, [feature])
            X_current = X[:, current_features]

            try:
                coefs, current_metric = _fit_model(X_current, y, metric)
            except Exception as e:
                warnings.append(f"Feature {feature} removal failed: {str(e)}")
                continue

            if isinstance(criterion, str):
                current_value = _calculate_criterion(X_current, y, coefs, criterion)
            else:
                current_value = criterion(coefs)

            if current_value < best_value - tol:
                best_feature = feature
                best_value = current_value

        if best_feature is None:
            break

        selected_features = np.setdiff1d(selected_features, [best_feature])
        best_criterion_value = best_value

    return selected_features

def _fit_model(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable]
) -> Tuple[np.ndarray, Dict]:
    """Fit linear regression model and calculate metrics."""
    # Using closed-form solution for simplicity
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    coefs = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
    intercept, coefficients = coefs[0], coefs[1:]

    # Calculate metrics
    y_pred = X_with_intercept @ coefs

    if isinstance(metric, str):
        metrics = _calculate_metrics(y_pred, y)
    else:
        metrics = {'metric': metric(y, y_pred)}

    return coefficients, metrics

def _calculate_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray
) -> Dict[str, float]:
    """Calculate common regression metrics."""
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)

    return {
        'mse': ss_residual / len(y_true),
        'mae': np.mean(np.abs(y_true - y_pred)),
        'r2': 1 - (ss_residual / ss_total)
    }

def _calculate_criterion(
    X: np.ndarray,
    y: np.ndarray,
    coefs: np.ndarray,
    criterion: str
) -> float:
    """Calculate model selection criterion."""
    n_samples, n_features = X.shape
    y_pred = X @ coefs

    if criterion == 'aic':
        mse = np.mean((y - y_pred)**2)
        return n_samples * np.log(mse) + 2 * (n_features + 1)
    elif criterion == 'bic':
        mse = np.mean((y - y_pred)**2)
        return n_samples * np.log(mse) + (n_features + 1) * np.log(n_samples)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

################################################################################
# random_forest_feature_importance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import BaseEstimator

def validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """
    Validate input data for random forest feature importance.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values array of shape (n_samples,)

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise ValueError("y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y is not None and len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if y is not None and np.any(np.isnan(y)):
        raise ValueError("y contains NaN values")

def normalize_features(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize features using specified method.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')

    Returns
    ------
    np.ndarray
        Normalized features array
    """
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

def compute_feature_importance(
    model: BaseEstimator,
    X: np.ndarray
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute feature importance from a trained model.

    Parameters
    ----------
    model : BaseEstimator
        Trained model with feature_importances_ attribute
    X : np.ndarray
        Input features array

    Returns
    ------
    Dict[str, Union[np.ndarray, float]]
        Dictionary containing feature importances and related metrics
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model must have feature_importances_ attribute")

    importances = model.feature_importances_
    normalized_importances = importances / np.sum(importances)

    return {
        'feature_importances': importances,
        'normalized_feature_importances': normalized_importances,
        'total_importance': np.sum(importances)
    }

def random_forest_feature_importance_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    normalization_method: str = 'standard',
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Union[int, float, str] = 'auto',
    bootstrap: bool = True,
    random_state: Optional[int] = None,
    regression: bool = True
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute feature importance using random forest.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values array of shape (n_samples,)
    normalization_method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    n_estimators : int
        Number of trees in the forest
    max_depth : Optional[int]
        Maximum depth of the tree
    min_samples_split : int
        Minimum number of samples required to split a node
    min_samples_leaf : int
        Minimum number of samples required at each leaf node
    max_features : Union[int, float, str]
        Number of features to consider when looking for the best split
    bootstrap : bool
        Whether bootstrap samples are used when building trees
    random_state : Optional[int]
        Controls both the randomness of the bootstrapping of the samples
        and the sampling of the features at each split
    regression : bool
        Whether to use random forest for regression or classification

    Returns
    ------
    Dict[str, Union[np.ndarray, float]]
        Dictionary containing:
        - 'result': Feature importances
        - 'metrics': Computed metrics
        - 'params_used': Parameters used for computation
        - 'warnings': Any warnings generated

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = random_forest_feature_importance_fit(X, y)
    """
    # Validate inputs
    validate_inputs(X, y)

    warnings = []

    # Normalize features
    X_normalized = normalize_features(X, normalization_method)

    # Determine model type based on regression flag
    if y is None:
        raise ValueError("y must be provided for feature importance computation")

    model_class = RandomForestRegressor if regression else RandomForestClassifier

    # Initialize and fit the model
    model = model_class(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=random_state
    )

    model.fit(X_normalized, y)

    # Compute feature importance
    importance_results = compute_feature_importance(model, X_normalized)

    return {
        'result': importance_results['normalized_feature_importances'],
        'metrics': {
            'total_importance': importance_results['total_importance']
        },
        'params_used': {
            'normalization_method': normalization_method,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'random_state': random_state
        },
        'warnings': warnings
    }

################################################################################
# gradient_boosting_feature_importance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> None:
    """Validate input data and normalizer."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")
    if normalizer is not None:
        try:
            X_normalized = normalizer(X)
        except Exception as e:
            raise ValueError(f"Normalizer function failed: {str(e)}")

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> float:
    """Compute the specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _fit_gradient_boosting(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """Fit gradient boosting model and compute feature importance."""
    _validate_inputs(X, y, normalizer)

    if normalizer is not None:
        X = normalizer(X)

    n_samples, n_features = X.shape
    feature_importance = np.zeros(n_features)
    y_pred = np.zeros_like(y)

    for _ in range(n_estimators):
        residuals = y - y_pred
        best_feature = None
        best_score = float('inf')

        for feature_idx in range(n_features):
            X_feature = X[:, feature_idx].reshape(-1, 1)
            model = np.linalg.lstsq(X_feature, residuals, rcond=None)[0][0]
            y_pred_feature = X_feature * model
            score = _compute_metric(residuals, y_pred_feature, metric)

            if score < best_score:
                best_score = score
                best_feature = feature_idx

        if best_feature is not None:
            X_best = X[:, best_feature].reshape(-1, 1)
            model = np.linalg.lstsq(X_best, residuals, rcond=None)[0][0]
            y_pred += learning_rate * (X_best * model)
            feature_importance[best_feature] += 1

    return {
        'result': {
            'feature_importance': feature_importance / n_estimators,
            'y_pred': y_pred
        },
        'metrics': {
            'final_metric': _compute_metric(y, y_pred, metric)
        },
        'params_used': {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'metric': metric,
            'normalizer': normalizer.__name__ if normalizer else None
        },
        'warnings': []
    }

def gradient_boosting_feature_importance_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Compute feature importance using gradient boosting.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    n_estimators : int, optional
        Number of boosting iterations.
    learning_rate : float, optional
        Shrinkage factor for each iteration.
    metric : str or callable, optional
        Metric to evaluate performance. Can be 'mse', 'mae', 'r2', or a custom callable.
    normalizer : callable, optional
        Function to normalize the input features.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': {'feature_importance', 'y_pred'}
        - 'metrics': {'final_metric'}
        - 'params_used': Dictionary of parameters used
        - 'warnings': List of warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = gradient_boosting_feature_importance_fit(X, y)
    """
    return _fit_gradient_boosting(
        X=X,
        y=y,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        metric=metric,
        normalizer=normalizer
    )
