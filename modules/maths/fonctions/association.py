"""
Quantix – Module association
Généré automatiquement
Date: 2026-01-06
"""

################################################################################
# correlation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def correlation_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'pearson',
    distance: Optional[Union[str, Callable]] = None,
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    weights: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute correlation between variables with configurable options.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Correlation metric ('pearson', 'spearman', 'kendall') or custom callable
    distance : str or callable, optional
        Distance metric for correlation computation (None by default)
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2')
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    custom_metric : callable, optional
        Custom metric function if needed
    weights : np.ndarray, optional
        Sample weights

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = correlation_fit(X, y, normalization='standard', metric='pearson')
    """
    # Input validation
    _validate_inputs(X, y, weights)

    # Normalize data
    X_norm, y_norm = _apply_normalization(X, y, normalization)

    # Prepare weights
    if weights is None:
        weights = np.ones_like(y)
    else:
        _validate_weights(weights)

    # Compute correlation based on parameters
    if callable(metric):
        corr_values = _compute_custom_correlation(X_norm, y_norm, metric)
    else:
        if distance is not None:
            corr_values = _compute_distance_based_correlation(X_norm, y_norm, distance)
        else:
            corr_values = _compute_standard_correlation(X_norm, y_norm, metric)

    # Solve for correlation coefficients
    if solver == 'closed_form':
        coefs = _closed_form_solution(X_norm, y_norm)
    else:
        coefs = _iterative_solver(
            X_norm, y_norm,
            solver=solver,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter
        )

    # Compute metrics
    metrics = _compute_metrics(y, coefs, X_norm, metric, custom_metric)

    return {
        'result': corr_values,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(X_norm, y_norm)
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
    """Validate input arrays dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if weights is not None:
        _validate_weights(weights)

def _validate_weights(weights: np.ndarray) -> None:
    """Validate weights array."""
    if not isinstance(weights, np.ndarray):
        raise TypeError("Weights must be a numpy array")
    if weights.ndim != 1:
        raise ValueError("Weights must be a 1D array")
    if len(weights) != X.shape[0]:
        raise ValueError("Weights must have same length as samples")

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply specified normalization to data."""
    if method == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif method == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == 'robust':
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        X_norm, y_norm = X.copy(), y.copy()
    return X_norm, y_norm

def _compute_standard_correlation(
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> np.ndarray:
    """Compute standard correlation metrics."""
    if method == 'pearson':
        return np.corrcoef(X, y, rowvar=False)[-1, :-1]
    elif method == 'spearman':
        return np.corrcoef(_rank_data(X), _rank_data(y), rowvar=False)[-1, :-1]
    elif method == 'kendall':
        return np.array([_kendall_tau(x, y) for x in X.T])
    else:
        raise ValueError(f"Unknown correlation method: {method}")

def _compute_distance_based_correlation(
    X: np.ndarray,
    y: np.ndarray,
    distance_metric: Union[str, Callable]
) -> np.ndarray:
    """Compute correlation based on distance metrics."""
    if callable(distance_metric):
        dist = np.array([[distance_metric(x, y) for x in X.T] for y in [y]])
    else:
        dist = _compute_distance(X, y, distance_metric)
    return 1 - dist / np.max(dist)

def _compute_custom_correlation(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable
) -> np.ndarray:
    """Compute correlation using custom metric function."""
    return np.array([metric_func(x, y) for x in X.T])

def _closed_form_solution(
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Compute correlation coefficients using closed form solution."""
    X_tX = X.T @ X
    if np.linalg.det(X_tX) == 0:
        raise ValueError("Matrix is singular, cannot compute closed form solution")
    return np.linalg.inv(X_tX) @ X.T @ y

def _iterative_solver(
    X: np.ndarray,
    y: np.ndarray,
    *,
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Iterative solver for correlation coefficients."""
    coefs = np.zeros(X.shape[1])
    prev_coefs = None

    for _ in range(max_iter):
        if solver == 'gradient_descent':
            coefs = _gradient_step(X, y, coefs, regularization)
        elif solver == 'newton':
            coefs = _newton_step(X, y, coefs)
        elif solver == 'coordinate_descent':
            coefs = _coordinate_step(X, y, coefs)

        if prev_coefs is not None and np.linalg.norm(coefs - prev_coefs) < tol:
            break
        prev_coefs = coefs.copy()

    return coefs

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray,
    metric_type: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute various metrics for correlation."""
    metrics = {}

    if metric_type == 'pearson':
        metrics['r_squared'] = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    elif metric_type == 'spearman':
        metrics['spearman_rho'] = np.corrcoef(_rank_data(y_true), _rank_data(y_pred))[0, 1]
    elif metric_type == 'kendall':
        metrics['kendall_tau'] = _kendall_tau(y_true, y_pred)

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)

    return metrics

def _check_warnings(
    X: np.ndarray,
    y: np.ndarray
) -> list:
    """Check for potential issues and return warnings."""
    warnings = []

    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        warnings.append("Input X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        warnings.append("Input y contains NaN or infinite values")

    return warnings

# Helper functions
def _rank_data(data: np.ndarray) -> np.ndarray:
    """Convert data to ranks."""
    return np.argsort(np.argsort(data))

def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Kendall's tau correlation."""
    n = len(x)
    concordant = discordant = 0

    for i in range(n):
        for j in range(i+1, n):
            if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                concordant += 1
            elif (x[i] - x[j]) * (y[i] - y[j]) < 0:
                discordant += 1

    return (concordant - discordant) / (n * (n-1) / 2)

def _compute_distance(
    X: np.ndarray,
    y: np.ndarray,
    metric: str
) -> np.ndarray:
    """Compute distance between samples."""
    if metric == 'euclidean':
        return np.sqrt(np.sum((X - y) ** 2, axis=1))
    elif metric == 'manhattan':
        return np.sum(np.abs(X - y), axis=1)
    elif metric == 'cosine':
        return 1 - np.dot(X, y) / (np.linalg.norm(X, axis=1) * np.linalg.norm(y))
    elif metric == 'minkowski':
        return np.sum(np.abs(X - y) ** 3, axis=1) ** (1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _gradient_step(
    X: np.ndarray,
    y: np.ndarray,
    coefs: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Perform one gradient descent step."""
    residuals = y - X @ coefs
    grad = -2 * X.T @ residuals

    if regularization == 'l1':
        grad += np.sign(coefs)
    elif regularization == 'l2':
        grad += 2 * coefs

    return coefs - 0.01 * grad

def _newton_step(
    X: np.ndarray,
    y: np.ndarray,
    coefs: np.ndarray
) -> np.ndarray:
    """Perform one Newton step."""
    residuals = y - X @ coefs
    hessian = 2 * X.T @ X
    grad = -2 * X.T @ residuals

    if np.linalg.det(hessian) == 0:
        raise ValueError("Hessian matrix is singular")

    return coefs - np.linalg.inv(hessian) @ grad

def _coordinate_step(
    X: np.ndarray,
    y: np.ndarray,
    coefs: np.ndarray
) -> np.ndarray:
    """Perform one coordinate descent step."""
    for i in range(X.shape[1]):
        X_i = X[:, i]
        coefs[i] = np.linalg.lstsq(X_i.reshape(-1, 1), y - X @ coefs + coefs[i] * X_i.reshape(-1, 1), rcond=None)[0]
    return coefs

################################################################################
# covariance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def covariance_fit(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    normalization: str = "none",
    metric: Union[str, Callable] = "mse",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Compute covariance between variables with configurable options.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target data matrix of shape (n_samples, n_targets). If None, computes auto-covariance.
    normalization : str
        Normalization method: "none", "standard", "minmax", or "robust"
    metric : Union[str, Callable]
        Metric to evaluate: "mse", "mae", "r2", or custom callable
    solver : str
        Solver method: "closed_form", "gradient_descent", or "coordinate_descent"
    regularization : Optional[str]
        Regularization type: None, "l1", "l2", or "elasticnet"
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    custom_metric : Optional[Callable]
        Custom metric function if needed
    custom_distance : Optional[Callable]
        Custom distance function if needed

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> Y = np.random.rand(100, 2)
    >>> result = covariance_fit(X, Y, normalization="standard", metric="r2")
    """
    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data
    X_norm = _apply_normalization(X, normalization)
    Y_norm = _apply_normalization(Y, normalization) if Y is not None else None

    # Choose solver
    if solver == "closed_form":
        cov_matrix = _closed_form_solver(X_norm, Y_norm)
    elif solver == "gradient_descent":
        cov_matrix = _gradient_descent_solver(X_norm, Y_norm, tol, max_iter)
    elif solver == "coordinate_descent":
        cov_matrix = _coordinate_descent_solver(X_norm, Y_norm, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if needed
    if regularization:
        cov_matrix = _apply_regularization(cov_matrix, regularization)

    # Compute metrics
    metrics = _compute_metrics(X_norm, Y_norm, cov_matrix, metric, custom_metric)

    return {
        "result": cov_matrix,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": _check_warnings(X_norm, Y_norm)
    }

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray]) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if Y is not None and not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array or None")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if Y is not None and np.any(np.isnan(Y)):
        raise ValueError("Y contains NaN values")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to data."""
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

def _closed_form_solver(X: np.ndarray, Y: Optional[np.ndarray]) -> np.ndarray:
    """Compute covariance using closed form solution."""
    if Y is None:
        return np.cov(X, rowvar=False)
    else:
        X_centered = X - np.mean(X, axis=0)
        Y_centered = Y - np.mean(Y, axis=0)
        return X_centered.T @ Y_centered / (X.shape[0] - 1)

def _gradient_descent_solver(X: np.ndarray, Y: Optional[np.ndarray], tol: float, max_iter: int) -> np.ndarray:
    """Compute covariance using gradient descent."""
    # Implementation of gradient descent solver
    pass

def _coordinate_descent_solver(X: np.ndarray, Y: Optional[np.ndarray], tol: float, max_iter: int) -> np.ndarray:
    """Compute covariance using coordinate descent."""
    # Implementation of coordinate descent solver
    pass

def _apply_regularization(cov_matrix: np.ndarray, method: str) -> np.ndarray:
    """Apply regularization to covariance matrix."""
    if method == "l1":
        return cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-4
    elif method == "l2":
        return cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-4
    elif method == "elasticnet":
        return cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-4
    else:
        raise ValueError(f"Unknown regularization method: {method}")

def _compute_metrics(X: np.ndarray, Y: Optional[np.ndarray], cov_matrix: np.ndarray,
                    metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Dict:
    """Compute specified metrics."""
    metrics = {}

    if callable(metric):
        metrics["custom_metric"] = metric(X, Y)
    elif metric == "mse":
        metrics["mse"] = np.mean((X @ cov_matrix - Y) ** 2)
    elif metric == "mae":
        metrics["mae"] = np.mean(np.abs(X @ cov_matrix - Y))
    elif metric == "r2":
        ss_res = np.sum((Y - X @ cov_matrix) ** 2)
        ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2)
        metrics["r2"] = 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if custom_metric is not None:
        metrics["custom_metric"] = custom_metric(X, Y)

    return metrics

def _check_warnings(X: np.ndarray, Y: Optional[np.ndarray]) -> list:
    """Check for potential warnings."""
    warnings = []

    if np.any(np.isclose(X.std(axis=0), 0)):
        warnings.append("Some features in X have zero variance")

    if Y is not None and np.any(np.isclose(Y.std(axis=0), 0)):
        warnings.append("Some targets in Y have zero variance")

    return warnings

################################################################################
# chi_square_test
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(contingency_table: np.ndarray) -> None:
    """Validate the input contingency table."""
    if not isinstance(contingency_table, np.ndarray):
        raise TypeError("Contingency table must be a numpy array.")
    if contingency_table.ndim != 2:
        raise ValueError("Contingency table must be 2-dimensional.")
    if np.any(contingency_table < 0):
        raise ValueError("Contingency table must contain non-negative values.")
    if np.any(np.isnan(contingency_table)):
        raise ValueError("Contingency table must not contain NaN values.")
    if np.any(np.isinf(contingency_table)):
        raise ValueError("Contingency table must not contain infinite values.")

def _calculate_expected(contingency_table: np.ndarray) -> np.ndarray:
    """Calculate the expected frequencies under the null hypothesis."""
    row_sums = contingency_table.sum(axis=1, keepdims=True)
    col_sums = contingency_table.sum(axis=0, keepdims=True)
    total = contingency_table.sum()
    expected = (row_sums @ col_sums) / total
    return expected

def _calculate_chi_square_statistic(observed: np.ndarray, expected: np.ndarray) -> float:
    """Calculate the chi-square statistic."""
    with np.errstate(divide='ignore', invalid='ignore'):
        chi_square = np.sum((observed - expected) ** 2 / expected)
    return float(chi_square)

def _calculate_p_value(chi_square_statistic: float, degrees_of_freedom: int) -> float:
    """Calculate the p-value for the chi-square statistic."""
    from scipy.stats import chi2
    return 1 - chi2.cdf(chi_square_statistic, degrees_of_freedom)

def _calculate_degrees_of_freedom(contingency_table: np.ndarray) -> int:
    """Calculate the degrees of freedom for the chi-square test."""
    return (contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1)

def chi_square_test_fit(
    contingency_table: np.ndarray,
    correction: Optional[str] = None,
    custom_statistic_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_p_value_func: Optional[Callable[[float, int], float]] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Perform a chi-square test for independence on a contingency table.

    Parameters
    ----------
    contingency_table : np.ndarray
        The observed contingency table.
    correction : str, optional
        Type of correction to apply ('yates' for Yates' correction).
    custom_statistic_func : callable, optional
        Custom function to calculate the test statistic.
    custom_p_value_func : callable, optional
        Custom function to calculate the p-value.

    Returns
    -------
    dict
        A dictionary containing the test results, metrics, and parameters used.

    Examples
    --------
    >>> contingency_table = np.array([[10, 20], [30, 40]])
    >>> result = chi_square_test_fit(contingency_table)
    """
    _validate_inputs(contingency_table)

    expected = _calculate_expected(contingency_table)
    degrees_of_freedom = _calculate_degrees_of_freedom(contingency_table)

    if custom_statistic_func is not None:
        chi_square_statistic = custom_statistic_func(contingency_table, expected)
    else:
        chi_square_statistic = _calculate_chi_square_statistic(contingency_table, expected)

    if correction == 'yates':
        chi_square_statistic = (np.abs(contingency_table - expected) - 0.5) ** 2 / expected
        chi_square_statistic = np.sum(chi_square_statistic)

    if custom_p_value_func is not None:
        p_value = custom_p_value_func(chi_square_statistic, degrees_of_freedom)
    else:
        p_value = _calculate_p_value(chi_square_statistic, degrees_of_freedom)

    result = {
        "result": {
            "chi_square_statistic": chi_square_statistic,
            "p_value": p_value,
            "degrees_of_freedom": degrees_of_freedom
        },
        "metrics": {},
        "params_used": {
            "correction": correction,
            "custom_statistic_func": custom_statistic_func is not None,
            "custom_p_value_func": custom_p_value_func is not None
        },
        "warnings": []
    }

    return result

################################################################################
# mutual_information
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Inputs must have the same length.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Inputs must not contain NaN values.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Inputs must not contain infinite values.")

def _compute_joint_probability(x: np.ndarray, y: np.ndarray,
                              bins: int = 10) -> tuple:
    """Compute joint probability distribution."""
    x_hist, x_edges = np.histogram(x, bins=bins, density=True)
    y_hist, y_edges = np.histogram(y, bins=bins, density=True)

    joint_hist, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges],
                                     density=True)

    return joint_hist, x_hist, y_hist

def _compute_mutual_information(joint_prob: np.ndarray,
                              x_prob: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute mutual information from probability distributions."""
    joint_prob = np.where(joint_prob == 0, 1e-12, joint_prob)
    x_prob = np.where(x_prob == 0, 1e-12, x_prob)
    y_prob = np.where(y_prob == 0, 1e-12, y_prob)

    mi = np.sum(joint_prob * np.log(joint_prob / (x_prob[:, None] * y_prob[None, :])))
    return mi

def mutual_information_fit(x: np.ndarray, y: np.ndarray,
                          bins: int = 10) -> Dict[str, Union[float, Dict]]:
    """
    Compute mutual information between two variables.

    Parameters
    ----------
    x : np.ndarray
        First input variable.
    y : np.ndarray
        Second input variable.
    bins : int, optional
        Number of bins for histogram computation (default: 10).

    Returns
    -------
    dict
        Dictionary containing:
        - "result": computed mutual information
        - "metrics": dictionary of additional metrics
        - "params_used": parameters used in computation
        - "warnings": list of warnings

    Example
    -------
    >>> x = np.random.randn(100)
    >>> y = np.random.randn(100)
    >>> result = mutual_information_fit(x, y)
    """
    _validate_inputs(x, y)

    joint_prob, x_prob, y_prob = _compute_joint_probability(x, y, bins=bins)
    mi = _compute_mutual_information(joint_prob, x_prob, y_prob)

    return {
        "result": mi,
        "metrics": {},
        "params_used": {"bins": bins},
        "warnings": []
    }

################################################################################
# pearson_correlation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def pearson_correlation_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'r2',
    solver: str = 'closed_form',
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute Pearson correlation between two variables with configurable options.

    Parameters:
    -----------
    x : np.ndarray
        First variable array.
    y : np.ndarray
        Second variable array.
    normalization : str, optional (default='standard')
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : str or callable, optional (default='r2')
        Metric to compute: 'mse', 'mae', 'r2', or custom callable.
    solver : str, optional (default='closed_form')
        Solver method: 'closed_form' or other future options.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    **kwargs :
        Additional solver-specific parameters.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, used parameters, and warnings.

    Example:
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> result = pearson_correlation_fit(x, y)
    """
    # Validate inputs
    _validate_inputs(x, y)

    # Normalize data
    x_norm, y_norm = _apply_normalization(x, y, normalization)

    # Compute correlation
    correlation = _compute_pearson_correlation(x_norm, y_norm)

    # Compute metrics
    metrics = _compute_metrics(correlation, x_norm, y_norm, metric, custom_metric)

    # Prepare output
    result = {
        'result': correlation,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver
        },
        'warnings': _check_warnings(x_norm, y_norm)
    }

    return result

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _apply_normalization(
    x: np.ndarray,
    y: np.ndarray,
    method: str
) -> tuple:
    """Apply selected normalization to input arrays."""
    if method == 'none':
        return x, y
    elif method == 'standard':
        x_norm = (x - np.mean(x)) / np.std(x)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif method == 'minmax':
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == 'robust':
        x_norm = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return x_norm, y_norm

def _compute_pearson_correlation(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """Compute Pearson correlation coefficient."""
    covariance = np.cov(x, y)[0, 1]
    std_x = np.std(x)
    std_y = np.std(y)
    return covariance / (std_x * std_y)

def _compute_metrics(
    correlation: float,
    x: np.ndarray,
    y: np.ndarray,
    metric_type: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute selected metrics."""
    metrics = {}

    if metric_type == 'r2':
        metrics['r_squared'] = correlation ** 2
    elif metric_type == 'mse':
        metrics['mse'] = np.mean((x - y) ** 2)
    elif metric_type == 'mae':
        metrics['mae'] = np.mean(np.abs(x - y))
    elif callable(metric_type):
        metrics['custom_metric'] = metric_type(x, y)
    elif custom_metric is not None:
        metrics['custom_metric'] = custom_metric(x, y)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

    return metrics

def _check_warnings(
    x: np.ndarray,
    y: np.ndarray
) -> list:
    """Check for potential warnings."""
    warnings = []

    if np.std(x) == 0 or np.std(y) == 0:
        warnings.append("Warning: One of the variables has zero standard deviation.")

    return warnings

################################################################################
# spearman_rank_correlation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def spearman_rank_correlation_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    normalize_x: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    normalize_y: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = 'euclidean',
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    tolerance: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Compute the Spearman rank correlation between two variables.

    Parameters
    ----------
    x : np.ndarray
        First input array.
    y : np.ndarray
        Second input array.
    normalize_x : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize x, by default None.
    normalize_y : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize y, by default None.
    distance_metric : str, optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine'), by default 'euclidean'.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function, by default None.
    tolerance : float, optional
        Tolerance for convergence, by default 1e-6.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]
        Dictionary containing the result, metrics, parameters used, and warnings.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([5, 4, 3, 2, 1])
    >>> result = spearman_rank_correlation_fit(x, y)
    """
    # Validate inputs
    _validate_inputs(x, y)

    # Normalize data if required
    x_normalized = _apply_normalization(x, normalize_x)
    y_normalized = _apply_normalization(y, normalize_y)

    # Compute ranks
    rank_x = _compute_ranks(x_normalized)
    rank_y = _compute_ranks(y_normalized)

    # Compute Spearman correlation
    spearman_corr = _compute_spearman_correlation(rank_x, rank_y)

    # Compute metrics
    metrics = _compute_metrics(x_normalized, y_normalized, rank_x, rank_y)

    # Prepare output
    result = {
        "result": spearman_corr,
        "metrics": metrics,
        "params_used": {
            "normalize_x": normalize_x.__name__ if normalize_x else None,
            "normalize_y": normalize_y.__name__ if normalize_y else None,
            "distance_metric": distance_metric,
            "custom_distance": custom_distance.__name__ if custom_distance else None,
            "tolerance": tolerance,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return result

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional.")
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _apply_normalization(
    data: np.ndarray,
    normalization_func: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to the data if a function is provided."""
    if normalization_func is not None:
        return normalization_func(data)
    return data

def _compute_ranks(data: np.ndarray) -> np.ndarray:
    """Compute ranks of the data."""
    sorted_indices = np.argsort(data)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(data) + 1)
    return ranks

def _compute_spearman_correlation(rank_x: np.ndarray, rank_y: np.ndarray) -> float:
    """Compute the Spearman rank correlation coefficient."""
    n = len(rank_x)
    d_squared = (rank_x - rank_y) ** 2
    spearman_corr = 1 - (6 * np.sum(d_squared)) / (n * (n ** 2 - 1))
    return spearman_corr

def _compute_metrics(
    x: np.ndarray,
    y: np.ndarray,
    rank_x: np.ndarray,
    rank_y: np.ndarray
) -> Dict[str, float]:
    """Compute various metrics."""
    return {
        "pearson_correlation": np.corrcoef(x, y)[0, 1],
        "spearman_correlation": _compute_spearman_correlation(rank_x, rank_y),
        "mean_rank_difference": np.mean(np.abs(rank_x - rank_y))
    }

################################################################################
# kendall_tau
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Kendall's tau calculation."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional.")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Input arrays must not contain NaN or inf values.")

def _count_concordant_discordant_pairs(x: np.ndarray, y: np.ndarray) -> Dict[str, int]:
    """Count concordant and discordant pairs for Kendall's tau."""
    n = len(x)
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            if x[i] != x[j] and y[i] != y[j]:
                if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                    concordant += 1
                else:
                    discordant += 1

    return {"concordant": concordant, "discordant": discordant}

def _compute_kendall_tau(concordant: int, discordant: int) -> float:
    """Compute Kendall's tau coefficient."""
    n = concordant + discordant
    if n == 0:
        return np.nan
    tau = (concordant - discordant) / n
    return tau

def kendall_tau_fit(
    x: np.ndarray,
    y: np.ndarray,
    normalization: str = "none",
    metric: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, Any]:
    """
    Compute Kendall's tau coefficient between two variables.

    Parameters:
    -----------
    x : np.ndarray
        First input array.
    y : np.ndarray
        Second input array.
    normalization : str, optional
        Normalization method (not used for Kendall's tau but kept for consistency).
    metric : str, optional
        Metric to compute (not used for Kendall's tau but kept for consistency).
    custom_metric : Callable, optional
        Custom metric function (not used for Kendall's tau but kept for consistency).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, y)

    pairs = _count_concordant_discordant_pairs(x, y)
    tau = _compute_kendall_tau(pairs["concordant"], pairs["discordant"])

    result = {
        "result": tau,
        "metrics": {"concordant_pairs": pairs["concordant"], "discordant_pairs": pairs["discordant"]},
        "params_used": {
            "normalization": normalization,
            "metric": metric if metric else "kendall_tau",
        },
        "warnings": [],
    }

    return result

# Example usage:
# x = np.array([1, 2, 3, 4])
# y = np.array([4, 3, 2, 1])
# result = kendall_tau_fit(x, y)

################################################################################
# point_biserial_correlation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def point_biserial_correlation_fit(
    x: np.ndarray,
    y: np.ndarray,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'pearson',
    solver: str = 'closed_form',
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute the point-biserial correlation between two variables.

    Parameters
    ----------
    x : np.ndarray
        Continuous variable.
    y : np.ndarray
        Binary variable (0 or 1).
    normalization : str, optional
        Normalization method for x. Options: 'none', 'standard', 'minmax', 'robust'.
    metric : str or callable, optional
        Metric to use for correlation. Options: 'pearson', 'spearman'. Default is 'pearson'.
    solver : str, optional
        Solver method. Options: 'closed_form'. Default is 'closed_form'.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    **kwargs :
        Additional keyword arguments for the solver or metric.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': The point-biserial correlation coefficient.
        - 'metrics': Dictionary of computed metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': List of warnings encountered.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([0, 1, 0, 1, 1])
    >>> result = point_biserial_correlation_fit(x, y)
    """
    # Validate inputs
    _validate_inputs(x, y)

    # Normalize x if required
    normalized_x = _normalize(x, normalization)

    # Compute the point-biserial correlation
    r_pb = _compute_point_biserial_correlation(normalized_x, y, metric, custom_metric)

    # Prepare the output
    output = {
        'result': r_pb,
        'metrics': {'correlation': r_pb},
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

    return output

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate the input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional.")
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Binary variable y must contain only 0 and 1.")

def _normalize(x: np.ndarray, method: str) -> np.ndarray:
    """Normalize the input array."""
    if method == 'none':
        return x
    elif method == 'standard':
        return (x - np.mean(x)) / np.std(x)
    elif method == 'minmax':
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    elif method == 'robust':
        return (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_point_biserial_correlation(
    x: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> float:
    """Compute the point-biserial correlation coefficient."""
    if metric == 'pearson':
        return np.corrcoef(x, y)[0, 1]
    elif metric == 'spearman':
        return np.corrcoef(_rank_data(x), _rank_data(y))[0, 1]
    elif callable(metric) or custom_metric:
        if custom_metric:
            return custom_metric(x, y)
        else:
            return metric(x, y)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _rank_data(data: np.ndarray) -> np.ndarray:
    """Rank the data for Spearman's correlation."""
    return np.argsort(np.argsort(data))

def _solve_closed_form(x: np.ndarray, y: np.ndarray) -> float:
    """Solve the point-biserial correlation using closed-form solution."""
    return np.corrcoef(x, y)[0, 1]

################################################################################
# phi_coefficient
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("Inputs must have the same shape.")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Inputs must not contain NaN or inf values.")

def _compute_contingency_table(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the contingency table for binary variables."""
    unique_x = np.unique(x)
    unique_y = np.unique(y)

    if len(unique_x) != 2 or len(unique_y) != 2:
        raise ValueError("Inputs must be binary variables.")

    contingency_table = np.zeros((2, 2), dtype=int)
    for i in range(2):
        for j in range(2):
            contingency_table[i, j] = np.sum((x == unique_x[i]) & (y == unique_y[j]))

    return contingency_table

def _compute_phi_coefficient(contingency_table: np.ndarray) -> float:
    """Compute the phi coefficient from a contingency table."""
    n = np.sum(contingency_table)
    chi_square = ((contingency_table * (n - contingency_table)) / n).sum()
    phi = np.sqrt(chi_square) / n
    return phi

def _normalize_data(x: np.ndarray, y: np.ndarray, normalization: str) -> tuple:
    """Normalize the input data."""
    if normalization == "standard":
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    elif normalization == "minmax":
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == "robust":
        x = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    return x, y

def phi_coefficient_fit(
    x: np.ndarray,
    y: np.ndarray,
    normalization: str = "none",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Compute the phi coefficient between two binary variables.

    Parameters
    ----------
    x : np.ndarray
        First binary variable.
    y : np.ndarray
        Second binary variable.
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    custom_metric : Callable, optional
        Custom metric function.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, y)

    x_norm, y_norm = _normalize_data(x, y, normalization)
    contingency_table = _compute_contingency_table(x_norm, y_norm)
    phi = _compute_phi_coefficient(contingency_table)

    metrics = {}
    if custom_metric is not None:
        metrics["custom_metric"] = custom_metric(x, y)

    return {
        "result": phi,
        "metrics": metrics,
        "params_used": {"normalization": normalization},
        "warnings": [],
    }

################################################################################
# cramers_v
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Cramér's V calculation."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays must have the same length.")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input x contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input y contains NaN or infinite values.")

def _compute_contingency_table(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the contingency table from two categorical arrays."""
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    contingency_table = np.zeros((len(x_unique), len(y_unique)), dtype=int)

    for i, x_val in enumerate(x_unique):
        for j, y_val in enumerate(y_unique):
            contingency_table[i, j] = np.sum((x == x_val) & (y == y_val))

    return contingency_table

def _compute_chi_square(contingency_table: np.ndarray) -> float:
    """Compute the chi-square statistic from a contingency table."""
    n = np.sum(contingency_table)
    row_sums = np.sum(contingency_table, axis=1)
    col_sums = np.sum(contingency_table, axis=0)

    expected = (row_sums[:, np.newaxis] * col_sums[np.newaxis, :]) / n
    chi_square = np.sum((contingency_table - expected) ** 2 / expected)
    return chi_square

def cramers_v_fit(
    x: np.ndarray,
    y: np.ndarray,
    normalization: str = "standard",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Compute Cramér's V association measure between two categorical variables.

    Parameters
    ----------
    x : np.ndarray
        First categorical variable.
    y : np.ndarray
        Second categorical variable.
    normalization : str, optional
        Normalization method for Cramér's V. Options: "standard", "none".
    custom_metric : Callable, optional
        Custom metric function to compute additional metrics.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]
        Dictionary containing:
        - "result": Cramér's V value
        - "metrics": Additional metrics if custom_metric is provided
        - "params_used": Parameters used in the computation
        - "warnings": Any warnings generated during computation

    Example
    -------
    >>> x = np.array([1, 2, 1, 3, 2])
    >>> y = np.array([4, 5, 4, 6, 5])
    >>> result = cramers_v_fit(x, y)
    """
    _validate_inputs(x, y)
    contingency_table = _compute_contingency_table(x, y)
    chi_square = _compute_chi_square(contingency_table)

    n = np.sum(contingency_table)
    phi_square = chi_square / n
    r, k = contingency_table.shape

    if normalization == "standard":
        phi = np.sqrt(phi_square / min(r - 1, k - 1))
    elif normalization == "none":
        phi = np.sqrt(phi_square)
    else:
        raise ValueError("Invalid normalization method. Choose 'standard' or 'none'.")

    result = {"result": phi}
    metrics = {}
    if custom_metric is not None:
        metrics["custom"] = custom_metric(x, y)

    return {
        "result": phi,
        "metrics": metrics,
        "params_used": {"normalization": normalization},
        "warnings": [],
    }

################################################################################
# anovas_test
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(
    y: np.ndarray,
    X: np.ndarray,
    groups: Optional[np.ndarray] = None
) -> None:
    """Validate input arrays."""
    if not isinstance(y, np.ndarray) or not isinstance(X, np.ndarray):
        raise TypeError("y and X must be numpy arrays")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if len(y) != X.shape[0]:
        raise ValueError("y and X must have the same number of samples")
    if groups is not None:
        if not isinstance(groups, np.ndarray):
            raise TypeError("groups must be a numpy array")
        if groups.ndim != 1:
            raise ValueError("groups must be a 1D array")
        if len(groups) != len(y):
            raise ValueError("groups must have the same length as y")

def _normalize_data(
    data: np.ndarray,
    method: str = "standard"
) -> np.ndarray:
    """Normalize data using specified method."""
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
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _calculate_ss(
    y: np.ndarray,
    X: np.ndarray,
    groups: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate sum of squares for ANOVA test."""
    if groups is None:
        # One-way ANOVA
        group_ids = np.unique(X, axis=0)
    else:
        # Two-way ANOVA
        group_ids = np.unique(groups)

    n_groups = len(group_ids)
    n_samples = len(y)

    # Total sum of squares
    ss_total = np.sum((y - np.mean(y))**2)

    # Between-group sum of squares
    ss_between = 0.0

    for group in group_ids:
        if groups is None:
            mask = np.all(X == group, axis=1)
        else:
            mask = groups == group
        y_group = y[mask]
        ss_between += len(y_group) * (np.mean(y_group) - np.mean(y))**2

    # Within-group sum of squares
    ss_within = ss_total - ss_between

    return {
        "ss_total": float(ss_total),
        "ss_between": float(ss_between),
        "ss_within": float(ss_within)
    }

def _calculate_f_statistic(
    ss_between: float,
    ss_within: float,
    n_groups: int,
    n_samples: int
) -> float:
    """Calculate F-statistic for ANOVA test."""
    df_between = n_groups - 1
    df_within = n_samples - n_groups

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    f_statistic = ms_between / (ms_within + 1e-8)
    return float(f_statistic)

def _calculate_p_value(
    f_statistic: float,
    df_between: int,
    df_within: int
) -> float:
    """Calculate p-value for ANOVA test."""
    from scipy.stats import f as f_dist
    return float(1 - f_dist.cdf(f_statistic, df_between, df_within))

def anovas_test_fit(
    y: np.ndarray,
    X: np.ndarray,
    groups: Optional[np.ndarray] = None,
    normalization: str = "standard",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform ANOVA test to assess association between variables.

    Parameters:
    -----------
    y : np.ndarray
        Dependent variable (1D array)
    X : np.ndarray
        Independent variables (2D array)
    groups : Optional[np.ndarray]
        Group labels for two-way ANOVA
    normalization : str
        Normalization method ("none", "standard", "minmax", "robust")
    custom_metric : Optional[Callable]
        Custom metric function if needed

    Returns:
    --------
    Dict containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(y, X, groups)

    # Normalize data
    y_norm = _normalize_data(y, normalization)
    X_norm = _normalize_data(X, normalization)

    # Calculate sum of squares
    ss_results = _calculate_ss(y_norm, X_norm, groups)

    # Calculate F-statistic
    if groups is None:
        n_groups = len(np.unique(X_norm, axis=0))
    else:
        n_groups = len(np.unique(groups))

    f_statistic = _calculate_f_statistic(
        ss_results["ss_between"],
        ss_results["ss_within"],
        n_groups,
        len(y)
    )

    # Calculate p-value
    df_between = n_groups - 1
    df_within = len(y) - n_groups

    p_value = _calculate_p_value(f_statistic, df_between, df_within)

    # Prepare results
    result = {
        "f_statistic": f_statistic,
        "p_value": p_value,
        "df_between": df_between,
        "df_within": df_within
    }

    metrics = {
        "ss_total": ss_results["ss_total"],
        "ss_between": ss_results["ss_between"],
        "ss_within": ss_results["ss_within"]
    }

    params_used = {
        "normalization": normalization,
        "custom_metric": custom_metric is not None
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
"""
y = np.array([1, 2, 3, 4, 5])
X = np.array([[0], [1], [0], [1], [0]])
result = anovas_test_fit(y, X)
"""

################################################################################
# t_test
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def t_test_fit(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = 'two-sided',
    equal_var: bool = True,
    nan_policy: str = 'propagate',
    *,
    normalization: Optional[str] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    **kwargs
) -> Dict[str, Any]:
    """
    Perform a t-test for the mean of one or two independent samples.

    Parameters
    ----------
    x : np.ndarray
        First sample data.
    y : np.ndarray
        Second sample data (for two-sample test) or None for one-sample test.
    alternative : str, optional
        Defines the alternative hypothesis. Must be 'two-sided', 'less' or 'greater'.
    equal_var : bool, optional
        If True, perform a standard independent 2-sample test assuming equal variances.
    nan_policy : str, optional
        Defines how to handle input NaN values. Must be 'propagate', 'raise' or 'omit'.
    normalization : str, optional
        Normalization method to apply. Must be None, 'standard', 'minmax' or 'robust'.
    metric : str or callable, optional
        Metric to evaluate. Must be 'mse', 'mae', 'r2' or a custom callable.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'result': dict with test results (statistic, pvalue)
        - 'metrics': dict with computed metrics
        - 'params_used': dict with parameters used
        - 'warnings': list of warnings

    Examples
    --------
    >>> x = np.random.normal(0, 1, 100)
    >>> y = np.random.normal(0.5, 1, 100)
    >>> result = t_test_fit(x, y, alternative='two-sided')
    """
    # Validate inputs
    _validate_inputs(x, y, nan_policy)

    # Normalize data if requested
    x_norm = _apply_normalization(x, normalization)
    y_norm = _apply_normalization(y, normalization) if y is not None else None

    # Compute test statistic and p-value
    result = _compute_t_test(x_norm, y_norm, alternative, equal_var)

    # Compute metrics
    metrics = _compute_metrics(x_norm, y_norm, metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'alternative': alternative,
            'equal_var': equal_var,
            'nan_policy': nan_policy,
            'normalization': normalization
        },
        'warnings': []
    }

def _validate_inputs(
    x: np.ndarray,
    y: Optional[np.ndarray],
    nan_policy: str
) -> None:
    """Validate input arrays and handle NaN values."""
    if not isinstance(x, np.ndarray) or (y is not None and not isinstance(y, np.ndarray)):
        raise TypeError("Inputs must be numpy arrays")

    if x.ndim != 1 or (y is not None and y.ndim != 1):
        raise ValueError("Inputs must be 1-dimensional")

    if nan_policy not in ('propagate', 'raise', 'omit'):
        raise ValueError("nan_policy must be 'propagate', 'raise' or 'omit'")

    if nan_policy == 'raise':
        if np.isnan(x).any() or (y is not None and np.isnan(y).any()):
            raise ValueError("Input contains NaN values")
    elif nan_policy == 'omit':
        x = x[~np.isnan(x)]
        if y is not None:
            y = y[~np.isnan(y)]

def _apply_normalization(
    data: np.ndarray,
    method: Optional[str]
) -> np.ndarray:
    """Apply normalization to data."""
    if method is None:
        return data

    if method == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    elif method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    elif method == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_t_test(
    x: np.ndarray,
    y: Optional[np.ndarray],
    alternative: str,
    equal_var: bool
) -> Dict[str, float]:
    """Compute t-test statistic and p-value."""
    if y is None:
        # One-sample t-test
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        n = len(x)

        t_stat = (mean - 0) / (std / np.sqrt(n))

    else:
        # Two-sample t-test
        x_mean, y_mean = np.mean(x), np.mean(y)
        x_std, y_std = np.std(x, ddof=1), np.std(y, ddof=1)
        n_x, n_y = len(x), len(y)

        if equal_var:
            pooled_std = np.sqrt(((n_x - 1) * x_std**2 + (n_y - 1) * y_std**2) / (n_x + n_y - 2))
            t_stat = (x_mean - y_mean) / (pooled_std * np.sqrt(1/n_x + 1/n_y))
        else:
            t_stat = (x_mean - y_mean) / np.sqrt(x_std**2/n_x + y_std**2/n_y)

    # Calculate p-value based on alternative hypothesis
    df = n_x + n_y - 2 if y is not None else n_x - 1
    p_value = _calculate_p_value(t_stat, df, alternative)

    return {
        'statistic': t_stat,
        'pvalue': p_value
    }

def _calculate_p_value(
    t_stat: float,
    df: int,
    alternative: str
) -> float:
    """Calculate p-value from t-statistic."""
    from scipy.stats import t

    if alternative == 'two-sided':
        p_value = 2 * (1 - t.cdf(np.abs(t_stat), df))
    elif alternative == 'less':
        p_value = t.cdf(t_stat, df)
    elif alternative == 'greater':
        p_value = 1 - t.cdf(t_stat, df)
    else:
        raise ValueError("alternative must be 'two-sided', 'less' or 'greater'")

    return p_value

def _compute_metrics(
    x: np.ndarray,
    y: Optional[np.ndarray],
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for the test."""
    if y is None:
        return {}

    metrics = {}
    x_mean, y_mean = np.mean(x), np.mean(y)

    if metric == 'mse':
        metrics['mse'] = np.mean((x - y) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(x - y))
    elif metric == 'r2':
        ss_total = np.sum((x - x_mean) ** 2)
        ss_residual = np.sum((x - y) ** 2)
        metrics['r2'] = 1 - (ss_residual / ss_total) if ss_total != 0 else np.nan
    elif callable(metric):
        metrics['custom'] = metric(x, y)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

################################################################################
# fisher_exact_test
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_contingency_table(table: np.ndarray) -> None:
    """Validate the contingency table for Fisher's exact test.

    Args:
        table: 2x2 numpy array representing the contingency table

    Raises:
        ValueError: If table is not 2x2 or contains invalid values
    """
    if table.shape != (2, 2):
        raise ValueError("Contingency table must be 2x2")
    if np.any(table < 0):
        raise ValueError("Table values must be non-negative")
    if np.any(np.isnan(table)):
        raise ValueError("Table contains NaN values")

def calculate_fisher_statistic(table: np.ndarray) -> float:
    """Calculate the Fisher exact test statistic.

    Args:
        table: 2x2 numpy array representing the contingency table

    Returns:
        The Fisher exact test statistic
    """
    a, b = table[0]
    c, d = table[1]

    # Calculate the hypergeometric probability
    n = a + b + c + d
    k1 = a + c
    k2 = a + b

    # Using log probabilities for numerical stability
    log_p = np.log(np.math.factorial(n)) - np.log(np.math.factorial(k1)) \
            - np.log(np.math.factorial(n - k1)) - np.log(np.math.factorial(k2)) \
            - np.log(np.math.factorial(n - k2)) + np.log(np.math.factorial(a)) \
            + np.log(np.math.factorial(n - k1 - k2 + a))

    return np.exp(log_p)

def fisher_exact_test_fit(
    table: np.ndarray,
    alternative_hypothesis: str = 'two-sided',
    method: str = 'midp',
    *,
    custom_statistic_func: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Perform Fisher's exact test for a 2x2 contingency table.

    Args:
        table: 2x2 numpy array representing the contingency table
        alternative_hypothesis: 'two-sided', 'less', or 'greater'
        method: 'midp' for mid-p-value or 'exact' for exact p-value
        custom_statistic_func: Optional callable to compute custom statistic

    Returns:
        Dictionary containing:
            - result: p-value
            - metrics: dictionary of additional metrics
            - params_used: parameters used in the computation
            - warnings: any warnings generated

    Example:
        >>> table = np.array([[10, 5], [2, 8]])
        >>> result = fisher_exact_test_fit(table)
    """
    # Validate inputs
    validate_contingency_table(table)

    # Use custom statistic function if provided, otherwise use default
    if custom_statistic_func is not None:
        statistic = custom_statistic_func(table)
    else:
        statistic = calculate_fisher_statistic(table)

    # Calculate p-value based on alternative hypothesis
    a, b = table[0]
    c, d = table[1]

    # Calculate all possible tables with same margins
    n = a + b + c + d
    k1 = a + c
    k2 = a + b

    # Generate all possible tables with same margins
    tables = []
    for i in range(min(k1, k2) + 1):
        j = k2 - i
        if j >= 0 and (k1 - i) >= 0:
            tables.append(np.array([[i, j], [k1 - i, k2 - j]]))

    # Calculate probabilities for all tables
    probs = []
    for t in tables:
        ti, tj = t[0]
        tk1_mi, tk2_mj = t[1]

        # Using log probabilities for numerical stability
        log_p = np.log(np.math.factorial(n)) - np.log(np.math.factorial(k1)) \
                - np.log(np.math.factorial(n - k1)) - np.log(np.math.factorial(k2)) \
                - np.log(np.math.factorial(n - k2)) + np.log(np.math.factorial(ti)) \
                + np.log(np.math.factorial(tk1_mi)) + np.log(np.math.factorial(tj)) \
                + np.log(np.math.factorial(tk2_mj))

        probs.append(np.exp(log_p))

    # Normalize probabilities
    total_prob = sum(probs)
    probs = [p / total_prob for p in probs]

    # Calculate p-value based on alternative hypothesis
    if alternative_hypothesis == 'two-sided':
        # Find all tables with probability <= original table
        if method == 'midp':
            # Mid-p-value calculation
            p_value = sum(p for t, p in zip(tables, probs) if calculate_fisher_statistic(t) <= statistic)
            p_value = min(p_value * 2, 1.0)
        else:
            # Exact p-value calculation
            p_value = sum(p for t, p in zip(tables, probs) if calculate_fisher_statistic(t) <= statistic)
            p_value = min(p_value * 2, 1.0)
    elif alternative_hypothesis == 'less':
        p_value = sum(p for t, p in zip(tables, probs) if calculate_fisher_statistic(t) <= statistic)
    elif alternative_hypothesis == 'greater':
        p_value = sum(p for t, p in zip(tables, probs) if calculate_fisher_statistic(t) >= statistic)
    else:
        raise ValueError("alternative_hypothesis must be 'two-sided', 'less', or 'greater'")

    # Prepare results
    result = {
        "result": p_value,
        "metrics": {
            "statistic": statistic,
            "odds_ratio": (a * d) / (b * c) if b != 0 and c != 0 else np.nan
        },
        "params_used": {
            "alternative_hypothesis": alternative_hypothesis,
            "method": method
        },
        "warnings": []
    }

    return result

################################################################################
# logistic_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def logistic_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = "logloss",
    solver: str = "gradient_descent",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    penalty: float = 1.0,
    custom_metric: Optional[Callable] = None,
    verbose: bool = False
) -> Dict:
    """
    Fit a logistic regression model to the given data.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, 1).
    normalizer : Optional[Callable], default=None
        Function to normalize the input features.
    metric : Union[str, Callable], default="logloss"
        Metric to evaluate the model. Can be "mse", "mae", "r2", "logloss" or a custom callable.
    solver : str, default="gradient_descent"
        Solver to use. Options: "closed_form", "gradient_descent", "newton".
    regularization : Optional[str], default=None
        Regularization type. Options: "l1", "l2", "elasticnet".
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    max_iter : int, default=1000
        Maximum number of iterations.
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    penalty : float, default=1.0
        Regularization strength.
    custom_metric : Optional[Callable], default=None
        Custom metric function.
    verbose : bool, default=False
        Whether to print progress information.

    Returns:
    --------
    Dict
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if a normalizer is provided
    if normalizer is not None:
        X = normalizer(X)

    # Initialize parameters
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    intercept = 0.0

    # Choose solver
    if solver == "closed_form":
        coefficients, intercept = _closed_form_solution(X, y)
    elif solver == "gradient_descent":
        coefficients, intercept = _gradient_descent(
            X, y, learning_rate, max_iter, tol, regularization, penalty
        )
    elif solver == "newton":
        coefficients, intercept = _newton_method(X, y, tol, max_iter)
    else:
        raise ValueError("Invalid solver specified.")

    # Calculate metrics
    metrics = _calculate_metrics(X, y, coefficients, intercept, metric, custom_metric)

    # Prepare results
    result = {
        "coefficients": coefficients,
        "intercept": intercept,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "penalty": penalty
        },
        "warnings": []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate the input data."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1 and y.ndim != 2:
        raise ValueError("y must be a 1D or 2D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _closed_form_solution(X: np.ndarray, y: np.ndarray) -> tuple:
    """Calculate the closed-form solution for logistic regression."""
    # This is a placeholder; actual implementation would involve iterative methods
    return np.zeros(X.shape[1]), 0.0

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    max_iter: int,
    tol: float,
    regularization: Optional[str],
    penalty: float
) -> tuple:
    """Perform gradient descent to find the coefficients."""
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    intercept = 0.0

    for _ in range(max_iter):
        # Calculate predictions
        linear_model = np.dot(X, coefficients) + intercept
        predictions = _sigmoid(linear_model)

        # Calculate gradients
        gradient = np.dot(X.T, (predictions - y)) / n_samples

        # Apply regularization if specified
        if regularization == "l1":
            gradient += penalty * np.sign(coefficients)
        elif regularization == "l2":
            gradient += 2 * penalty * coefficients
        elif regularization == "elasticnet":
            gradient += penalty * (np.sign(coefficients) + 2 * coefficients)

        # Update coefficients
        coefficients -= learning_rate * gradient

        # Check for convergence
        if np.linalg.norm(gradient) < tol:
            break

    return coefficients, intercept

def _newton_method(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> tuple:
    """Perform Newton's method to find the coefficients."""
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    intercept = 0.0

    for _ in range(max_iter):
        # Calculate predictions
        linear_model = np.dot(X, coefficients) + intercept
        predictions = _sigmoid(linear_model)

        # Calculate Hessian and gradient
        hessian = np.dot(X.T * (predictions * (1 - predictions)), X) / n_samples
        gradient = np.dot(X.T, (predictions - y)) / n_samples

        # Update coefficients
        coefficients -= np.linalg.solve(hessian, gradient)

        # Check for convergence
        if np.linalg.norm(gradient) < tol:
            break

    return coefficients, intercept

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate the metrics for the model."""
    linear_model = np.dot(X, coefficients) + intercept
    predictions = _sigmoid(linear_model)

    metrics_dict = {}

    if metric == "logloss":
        log_loss = -np.mean(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))
        metrics_dict["logloss"] = log_loss
    elif metric == "mse":
        mse = np.mean((y - predictions) ** 2)
        metrics_dict["mse"] = mse
    elif metric == "mae":
        mae = np.mean(np.abs(y - predictions))
        metrics_dict["mae"] = mae
    elif metric == "r2":
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        metrics_dict["r2"] = r2
    elif callable(metric):
        metrics_dict["custom_metric"] = metric(y, predictions)

    if custom_metric is not None:
        metrics_dict["custom_metric"] = custom_metric(y, predictions)

    return metrics_dict

################################################################################
# association_rules
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def association_rules_fit(
    data: np.ndarray,
    min_support: float = 0.1,
    metric: str = 'confidence',
    normalization: Optional[str] = None,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = np.linalg.norm,
    solver: str = 'apriori',
    max_rules: int = 100,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, np.ndarray, str]]:
    """
    Compute association rules from transactional data.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix where each row represents a transaction and columns represent items.
    min_support : float, optional
        Minimum support threshold for itemsets (default: 0.1).
    metric : str, optional
        Metric to evaluate rules ('confidence', 'lift', 'support') (default: 'confidence').
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax') (default: None).
    distance_metric : Callable, optional
        Custom distance metric function (default: Euclidean norm).
    solver : str, optional
        Solver algorithm ('apriori', 'fp_growth') (default: 'apriori').
    max_rules : int, optional
        Maximum number of rules to return (default: 100).
    custom_metric : Callable, optional
        Custom metric function for rule evaluation.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': List of association rules.
        - 'metrics': Computed metrics for each rule.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings generated during computation.

    Example
    -------
    >>> data = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    >>> result = association_rules_fit(data)
    """
    # Validate inputs
    _validate_inputs(data, min_support, max_rules)

    # Normalize data if specified
    normalized_data = _normalize(data, normalization) if normalization else data

    # Compute frequent itemsets
    frequent_itemsets = _compute_frequent_itemsets(normalized_data, min_support, solver)

    # Generate association rules
    rules = _generate_rules(frequent_itemsets, metric, custom_metric)

    # Select top rules
    selected_rules = _select_top_rules(rules, max_rules)

    # Compute metrics for selected rules
    metrics = _compute_metrics(selected_rules, metric, custom_metric)

    return {
        'result': selected_rules,
        'metrics': metrics,
        'params_used': {
            'min_support': min_support,
            'metric': metric,
            'normalization': normalization,
            'solver': solver
        },
        'warnings': []
    }

def _validate_inputs(data: np.ndarray, min_support: float, max_rules: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if min_support <= 0 or min_support > 1:
        raise ValueError("min_support must be between 0 and 1.")
    if max_rules <= 0:
        raise ValueError("max_rules must be positive.")

def _normalize(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data based on specified method."""
    if method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_frequent_itemsets(data: np.ndarray, min_support: float, solver: str) -> Dict:
    """Compute frequent itemsets using specified solver."""
    if solver == 'apriori':
        return _apriori(data, min_support)
    elif solver == 'fp_growth':
        return _fp_growth(data, min_support)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _apriori(data: np.ndarray, min_support: float) -> Dict:
    """Apriori algorithm for frequent itemset mining."""
    # Implementation of Apriori algorithm
    pass

def _fp_growth(data: np.ndarray, min_support: float) -> Dict:
    """FP-Growth algorithm for frequent itemset mining."""
    # Implementation of FP-Growth algorithm
    pass

def _generate_rules(frequent_itemsets: Dict, metric: str, custom_metric: Optional[Callable]) -> list:
    """Generate association rules from frequent itemsets."""
    rules = []
    for itemset, support in frequent_itemsets.items():
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                antecedent = tuple(sorted(list(itemset)[:i]))
                consequent = tuple(sorted(list(itemset)[i:]))
                rules.append((antecedent, consequent))
    return rules

def _select_top_rules(rules: list, max_rules: int) -> list:
    """Select top rules based on specified metric."""
    # Placeholder for rule selection logic
    return rules[:max_rules]

def _compute_metrics(rules: list, metric: str, custom_metric: Optional[Callable]) -> Dict:
    """Compute metrics for association rules."""
    metrics = {}
    for rule in rules:
        antecedent, consequent = rule
        if custom_metric:
            metrics[rule] = custom_metric(antecedent, consequent)
        elif metric == 'confidence':
            metrics[rule] = _compute_confidence(antecedent, consequent)
        elif metric == 'lift':
            metrics[rule] = _compute_lift(antecedent, consequent)
        elif metric == 'support':
            metrics[rule] = _compute_support(antecedent, consequent)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return metrics

def _compute_confidence(antecedent: tuple, consequent: tuple) -> float:
    """Compute confidence for a rule."""
    # Placeholder for confidence computation
    return 0.5

def _compute_lift(antecedent: tuple, consequent: tuple) -> float:
    """Compute lift for a rule."""
    # Placeholder for lift computation
    return 1.0

def _compute_support(antecedent: tuple, consequent: tuple) -> float:
    """Compute support for a rule."""
    # Placeholder for support computation
    return 0.1

################################################################################
# lift
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def lift_fit(
    contingency_table: np.ndarray,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'standard',
    solver: str = 'closed_form',
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute the lift statistic for a contingency table.

    Parameters:
    -----------
    contingency_table : np.ndarray
        2D array representing the contingency table.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to use ('standard', 'custom') or custom callable.
    solver : str, optional
        Solver method ('closed_form').
    custom_metric : callable, optional
        Custom metric function if metric='custom'.
    **kwargs :
        Additional keyword arguments for specific methods.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    contingency_table = _validate_contingency_table(contingency_table)

    # Normalize data
    normalized_data = _normalize_contingency_table(contingency_table, normalization)

    # Compute lift
    result = _compute_lift(normalized_data, solver, **kwargs)

    # Compute metrics
    metrics = _compute_metrics(result, metric, custom_metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver
        },
        'warnings': _check_warnings(contingency_table, result)
    }

def _validate_contingency_table(table: np.ndarray) -> np.ndarray:
    """
    Validate the contingency table.

    Parameters:
    -----------
    table : np.ndarray
        Contingency table to validate.

    Returns:
    --------
    np.ndarray
        Validated contingency table.
    """
    if not isinstance(table, np.ndarray) or len(table.shape) != 2:
        raise ValueError("Contingency table must be a 2D numpy array.")
    if np.any(table < 0):
        raise ValueError("Contingency table must contain non-negative values.")
    if np.any(np.isnan(table)):
        raise ValueError("Contingency table must not contain NaN values.")
    return table

def _normalize_contingency_table(
    table: np.ndarray,
    method: str = 'none'
) -> np.ndarray:
    """
    Normalize the contingency table.

    Parameters:
    -----------
    table : np.ndarray
        Contingency table to normalize.
    method : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').

    Returns:
    --------
    np.ndarray
        Normalized contingency table.
    """
    if method == 'none':
        return table
    elif method == 'standard':
        mean = np.mean(table)
        std = np.std(table)
        return (table - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(table)
        max_val = np.max(table)
        return (table - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(table)
        iqr = np.percentile(table, 75) - np.percentile(table, 25)
        return (table - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_lift(
    table: np.ndarray,
    solver: str = 'closed_form',
    **kwargs
) -> float:
    """
    Compute the lift statistic.

    Parameters:
    -----------
    table : np.ndarray
        Normalized contingency table.
    solver : str, optional
        Solver method ('closed_form').

    Returns:
    --------
    float
        Computed lift statistic.
    """
    if solver == 'closed_form':
        total = np.sum(table)
        row_sums = np.sum(table, axis=1, keepdims=True)
        col_sums = np.sum(table, axis=0, keepdims=True)
        expected = (row_sums * col_sums) / total
        lift = table / expected
        return np.sum(lift)
    else:
        raise ValueError(f"Unknown solver method: {solver}")

def _compute_metrics(
    result: float,
    metric: Union[str, Callable] = 'standard',
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Compute metrics for the lift statistic.

    Parameters:
    -----------
    result : float
        Computed lift statistic.
    metric : str or callable, optional
        Metric to use ('standard', 'custom') or custom callable.
    custom_metric : callable, optional
        Custom metric function if metric='custom'.

    Returns:
    --------
    dict
        Dictionary of computed metrics.
    """
    if metric == 'standard':
        return {'lift': result}
    elif callable(metric):
        return {f'custom_metric_{id(metric)}': metric(result)}
    elif custom_metric is not None:
        return {f'custom_metric_{id(custom_metric)}': custom_metric(result)}
    else:
        raise ValueError("Invalid metric specification.")

def _check_warnings(
    table: np.ndarray,
    result: float
) -> list:
    """
    Check for warnings during computation.

    Parameters:
    -----------
    table : np.ndarray
        Original contingency table.
    result : float
        Computed lift statistic.

    Returns:
    --------
    list
        List of warning messages.
    """
    warnings = []
    if np.any(table == 0):
        warnings.append("Contingency table contains zeros, which may affect lift computation.")
    if np.isnan(result):
        warnings.append("Result is NaN, check input data and normalization method.")
    return warnings

################################################################################
# confidence
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def confidence_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalization: str = "standard",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Compute confidence metrics for association between X and y.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    normalization : str, optional
        Normalization method: "none", "standard", "minmax", or "robust"
    metric : str or callable, optional
        Metric to evaluate: "mse", "mae", "r2", "logloss" or custom callable
    distance : str, optional
        Distance metric: "euclidean", "manhattan", "cosine", or "minkowski"
    solver : str, optional
        Solver method: "closed_form", "gradient_descent", "newton", or "coordinate_descent"
    regularization : str, optional
        Regularization type: None, "l1", "l2", or "elasticnet"
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
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = confidence_fit(X, y, normalization="standard", metric="r2")
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = _normalize_data(X, method=normalization)
    y_norm = _normalize_data(y.reshape(-1, 1), method=normalization).flatten()

    # Prepare solver parameters
    solver_params = {
        "tol": tol,
        "max_iter": max_iter,
        "regularization": regularization
    }

    # Solve for parameters
    params = _solve_association(
        X_norm, y_norm,
        solver=solver,
        distance=distance,
        custom_distance=custom_distance,
        **solver_params
    )

    # Calculate metrics
    metrics = _calculate_metrics(
        X_norm, y_norm,
        params=params,
        metric=metric,
        custom_metric=custom_metric
    )

    # Prepare results dictionary
    result = {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric if isinstance(metric, str) else "custom",
            "distance": distance,
            "solver": solver,
            "regularization": regularization
        },
        "warnings": _check_warnings(X_norm, y_norm)
    }

    return result

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

def _normalize_data(data: np.ndarray, method: str = "standard") -> np.ndarray:
    """Normalize data using specified method."""
    if method == "none":
        return data
    elif method == "standard":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _solve_association(
    X: np.ndarray,
    y: np.ndarray,
    *,
    solver: str = "closed_form",
    distance: str = "euclidean",
    custom_distance: Optional[Callable] = None,
    **kwargs
) -> np.ndarray:
    """Solve for association parameters using specified solver."""
    if solver == "closed_form":
        return _solve_closed_form(X, y)
    elif solver == "gradient_descent":
        return _solve_gradient_descent(X, y, distance=distance, custom_distance=custom_distance, **kwargs)
    elif solver == "newton":
        return _solve_newton(X, y, distance=distance, custom_distance=custom_distance, **kwargs)
    elif solver == "coordinate_descent":
        return _solve_coordinate_descent(X, y, distance=distance, custom_distance=custom_distance, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    *,
    params: np.ndarray,
    metric: Union[str, Callable] = "mse",
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Calculate metrics for the association model."""
    if custom_metric is not None:
        return {"custom": custom_metric(X, y, params)}

    metrics = {}
    if metric == "mse":
        metrics["mse"] = np.mean((y - X @ params) ** 2)
    elif metric == "mae":
        metrics["mae"] = np.mean(np.abs(y - X @ params))
    elif metric == "r2":
        ss_res = np.sum((y - X @ params) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics["r2"] = 1 - (ss_res / ss_tot)
    elif metric == "logloss":
        y_pred = 1 / (1 + np.exp(-X @ params))
        metrics["logloss"] = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

def _check_warnings(X: np.ndarray, y: np.ndarray) -> list:
    """Check for potential warnings in the association model."""
    warnings = []

    if np.any(np.std(X, axis=0) == 0):
        warnings.append("Some features have zero variance")

    if np.any(np.abs(X) > 1e6):
        warnings.append("Features contain very large values")

    if np.any(np.abs(y) > 1e6):
        warnings.append("Target contains very large values")

    return warnings

# Example internal solver implementations
def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form solution for linear association."""
    return np.linalg.pinv(X) @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    distance: str = "euclidean",
    custom_distance: Optional[Callable] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Gradient descent solver for association."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        grad = -2 * X.T @ (y - X @ params) / len(y)
        new_params = params - learning_rate * grad

        if np.linalg.norm(new_params - params) < tol:
            break

        params = new_params

    return params

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    *,
    distance: str = "euclidean",
    custom_distance: Optional[Callable] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Newton's method solver for association."""
    n_features = X.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        residuals = y - X @ params
        grad = -2 * X.T @ residuals / len(y)
        hessian = 2 * X.T @ X / len(y)

        new_params = params - np.linalg.pinv(hessian) @ grad

        if np.linalg.norm(new_params - params) < tol:
            break

        params = new_params

    return params

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    distance: str = "euclidean",
    custom_distance: Optional[Callable] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Coordinate descent solver for association."""
    n_features = X.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        for i in range(n_features):
            X_i = X[:, i]
            residuals = y - (X @ params - X_i * params[i])

            # Compute correlation
            corr = np.sum(X_i * residuals) / (np.linalg.norm(X_i) * np.linalg.norm(residuals))

            # Update parameter
            params[i] = corr

        if np.linalg.norm(params) < tol:
            break

    return params

################################################################################
# support
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def support_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalization: str = 'standard',
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
    Compute support for association analysis between features and target.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'
    metric : str or callable, optional
        Metric to evaluate: 'mse', 'mae', 'r2', or custom callable
    distance : str or callable, optional
        Distance metric: 'euclidean', 'manhattan', 'cosine', or custom callable
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', or 'coordinate_descent'
    regularization : str, optional
        Regularization type: None, 'l1', 'l2', or 'elasticnet'
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
        Dictionary containing results, metrics, parameters used, and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = support_fit(X, y, normalization='standard', metric='r2')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = _normalize_data(X, method=normalization)

    # Prepare solver parameters
    solver_params = {
        'tol': tol,
        'max_iter': max_iter,
        'regularization': regularization
    }

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(X_norm, y)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(X_norm, y, **solver_params)
    elif solver == 'coordinate_descent':
        params = _solve_coordinate_descent(X_norm, y, **solver_params)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_norm, y, params, metric=metric)

    # Compute support
    result = _compute_support(X_norm, y, params, distance=distance)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
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

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> Dict:
    """Solve using closed form solution."""
    XtX = X.T @ X
    Xty = X.T @ y
    params = np.linalg.solve(XtX, Xty)
    return {'coefficients': params}

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000,
    regularization: Optional[str] = None
) -> Dict:
    """Solve using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y)
        if regularization == 'l1':
            gradient += np.sign(params)
        elif regularization == 'l2':
            gradient += 2 * params
        params -= learning_rate * gradient

        if np.linalg.norm(gradient) < tol:
            break

    return {'coefficients': params}

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000,
    regularization: Optional[str] = None
) -> Dict:
    """Solve using coordinate descent."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - X @ params + params[j] * X_j
            numerator = X_j.T @ residuals
            denominator = X_j.T @ X_j

            if regularization == 'l1':
                params[j] = np.sign(numerator) * np.maximum(
                    np.abs(numerator) - 1, 0
                ) / denominator
            elif regularization == 'l2':
                params[j] = numerator / (denominator + 1)
            else:
                params[j] = numerator / denominator

        if np.linalg.norm(X @ params - y) < tol:
            break

    return {'coefficients': params}

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict,
    metric: Union[str, Callable] = 'mse'
) -> Dict:
    """Compute metrics based on specified method."""
    y_pred = X @ params['coefficients']

    if callable(metric):
        return {'custom_metric': metric(y, y_pred)}

    if metric == 'mse':
        return {'mse': np.mean((y - y_pred) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(y - y_pred))}
    elif metric == 'r2':
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return {'r2': 1 - ss_res / (ss_tot + 1e-8)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_support(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict,
    distance: Union[str, Callable] = 'euclidean'
) -> np.ndarray:
    """Compute support between features and target."""
    y_pred = X @ params['coefficients']

    if callable(distance):
        return distance(y, y_pred)

    if distance == 'euclidean':
        return np.sqrt(np.sum((y - y_pred) ** 2))
    elif distance == 'manhattan':
        return np.sum(np.abs(y - y_pred))
    elif distance == 'cosine':
        return 1 - np.dot(y, y_pred) / (np.linalg.norm(y) * np.linalg.norm(y_pred))
    else:
        raise ValueError(f"Unknown distance: {distance}")
