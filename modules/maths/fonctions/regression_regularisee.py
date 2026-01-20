"""
Quantix – Module regression_regularisee
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# ridge_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def ridge_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    normalize: str = 'standard',
    solver: str = 'closed_form',
    metric: Union[str, Callable] = 'mse',
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> Dict:
    """
    Fit a ridge regression model with configurable parameters.

    Parameters:
    -----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    alpha : float, default=1.0
        Regularization strength
    normalize : str, default='standard'
        Normalization method: 'none', 'standard', 'minmax', or 'robust'
    solver : str, default='closed_form'
        Solver method: 'closed_form', 'gradient_descent', or 'newton'
    metric : str or callable, default='mse'
        Evaluation metric: 'mse', 'mae', 'r2', or custom callable
    max_iter : int, default=1000
        Maximum number of iterations for iterative solvers
    tol : float, default=1e-4
        Tolerance for stopping criteria
    random_state : int or None, default=None
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': fitted coefficients
        - 'metrics': computed metrics
        - 'params_used': parameters used in fitting
        - 'warnings': any warnings encountered

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = ridge_regression_fit(X, y, alpha=0.5, normalize='standard')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else None

    # Normalize data
    X_norm, y_norm = _apply_normalization(X, y, normalize)

    # Choose solver
    if solver == 'closed_form':
        coefs = _ridge_closed_form(X_norm, y_norm, alpha)
    elif solver == 'gradient_descent':
        coefs = _ridge_gradient_descent(X_norm, y_norm, alpha, max_iter, tol, rng)
    elif solver == 'newton':
        coefs = _ridge_newton(X_norm, y_norm, alpha, max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(y, y_norm @ coefs, metric)

    # Prepare output
    result = {
        'result': coefs,
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'normalize': normalize,
            'solver': solver,
            'metric': metric if isinstance(metric, str) else 'custom',
            'max_iter': max_iter,
            'tol': tol
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

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> tuple:
    """Apply specified normalization to features and target."""
    if method == 'none':
        return X, y
    elif method == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif method == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == 'robust':
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_norm, y_norm

def _ridge_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float
) -> np.ndarray:
    """Compute ridge regression coefficients using closed-form solution."""
    n_features = X.shape[1]
    I = np.eye(n_features)
    coefs = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return coefs

def _ridge_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
    rng: Optional[np.random.RandomState]
) -> np.ndarray:
    """Compute ridge regression coefficients using gradient descent."""
    n_features = X.shape[1]
    coefs = rng.randn(n_features) if rng is not None else np.random.randn(n_features)
    learning_rate = 0.1

    for _ in range(max_iter):
        gradient = X.T @ (X @ coefs - y) + alpha * coefs
        new_coefs = coefs - learning_rate * gradient

        if np.linalg.norm(new_coefs - coefs) < tol:
            break

        coefs = new_coefs

    return coefs

def _ridge_newton(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Compute ridge regression coefficients using Newton's method."""
    n_features = X.shape[1]
    coefs = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = X.T @ (X @ coefs - y) + alpha * coefs
        hessian = X.T @ X + alpha * np.eye(n_features)
        delta = np.linalg.solve(hessian, -gradient)
        new_coefs = coefs + delta

        if np.linalg.norm(delta) < tol:
            break

        coefs = new_coefs

    return coefs

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Compute specified metrics between true and predicted values."""
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - ss_res / ss_tot
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        metrics['custom'] = metric(y_true, y_pred)

    return metrics

################################################################################
# lasso_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def lasso_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    solver: str = 'coordinate_descent',
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    penalty: str = 'l1',
    random_state: Optional[int] = None,
    **kwargs
) -> Dict:
    """
    Fit a Lasso regression model with configurable parameters.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    alpha : float, default=1.0
        Regularization strength
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for stopping criteria
    solver : str, default='coordinate_descent'
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    normalization : str, default='standard'
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, default='mse'
        Metric to evaluate ('mse', 'mae', 'r2', custom callable)
    penalty : str, default='l1'
        Penalty type ('none', 'l1', 'l2', 'elasticnet')
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': fitted coefficients
        - 'metrics': computed metrics
        - 'params_used': parameters used in fitting
        - 'warnings': any warnings generated

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = lasso_regression_fit(X, y, alpha=0.1)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    X_norm, y_norm = _apply_normalization(X, y, normalization)

    # Initialize coefficients
    n_features = X_norm.shape[1]
    coef = np.zeros(n_features)

    # Choose solver
    if solver == 'coordinate_descent':
        coef = _coordinate_descent_solver(X_norm, y_norm, alpha, max_iter, tol)
    elif solver == 'gradient_descent':
        coef = _gradient_descent_solver(X_norm, y_norm, alpha, max_iter, tol)
    elif solver == 'closed_form':
        coef = _closed_form_solver(X_norm, y_norm, alpha)
    elif solver == 'newton':
        coef = _newton_solver(X_norm, y_norm, alpha, max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_norm, y_norm, coef, metric)

    # Prepare output
    result = {
        'result': {'coefficients': coef},
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'max_iter': max_iter,
            'tol': tol,
            'solver': solver,
            'normalization': normalization,
            'metric': metric,
            'penalty': penalty
        },
        'warnings': []
    }

    return result

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

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> tuple:
    """Apply normalization to features and target."""
    if method == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif method == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == 'robust':
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    elif method == 'none':
        X_norm = X.copy()
        y_norm = y.copy()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_norm, y_norm

def _coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Coordinate descent solver for Lasso regression."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    prev_coef = coef.copy()

    for _ in range(max_iter):
        for j in range(n_features):
            # Compute residuals
            r = y - X.dot(coef) + coef[j] * X[:, j]

            # Compute correlation
            corr = X[:, j].dot(r)

            if corr < -alpha / 2:
                coef[j] = (corr - alpha / 2) / np.sum(X[:, j]**2)
            elif corr > alpha / 2:
                coef[j] = (corr + alpha / 2) / np.sum(X[:, j]**2)
            else:
                coef[j] = 0

        # Check convergence
        if np.linalg.norm(coef - prev_coef) < tol:
            break

        prev_coef = coef.copy()

    return coef

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Gradient descent solver for Lasso regression."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    prev_coef = coef.copy()
    learning_rate = 0.1

    for _ in range(max_iter):
        # Compute gradient
        grad = X.T.dot(X.dot(coef) - y)
        l1_grad = alpha * np.sign(coef)

        # Update coefficients
        coef -= learning_rate * (grad + l1_grad)

        # Check convergence
        if np.linalg.norm(coef - prev_coef) < tol:
            break

        prev_coef = coef.copy()

    return coef

def _closed_form_solver(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float
) -> np.ndarray:
    """Closed form solution for Lasso regression (when possible)."""
    # Note: Closed form is not generally available for Lasso
    raise NotImplementedError("Closed form solution not implemented for Lasso regression")

def _newton_solver(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Newton's method solver for Lasso regression."""
    # Note: Newton's method is not typically used for Lasso
    raise NotImplementedError("Newton's method not implemented for Lasso regression")

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Compute evaluation metrics."""
    y_pred = X.dot(coef)
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((y - y_pred)**2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            metrics['r2'] = 1 - ss_res / ss_tot
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        metrics['custom'] = metric(y, y_pred)
    else:
        raise ValueError("Metric must be a string or callable")

    return metrics

################################################################################
# elastic_net
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def elastic_net_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha_l1: float = 1.0,
    alpha_l2: float = 1.0,
    normalize: str = 'standard',
    solver: str = 'coordinate_descent',
    max_iter: int = 1000,
    tol: float = 1e-4,
    metric: str = 'mse',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    penalty: str = 'elasticnet',
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit an Elastic Net regression model.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    alpha_l1 : float, optional
        L1 regularization strength.
    alpha_l2 : float, optional
        L2 regularization strength.
    normalize : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for stopping criteria.
    metric : str, optional
        Metric to evaluate: 'mse', 'mae', 'r2', or 'logloss'.
    custom_metric : Callable, optional
        Custom metric function.
    penalty : str, optional
        Penalty type: 'none', 'l1', 'l2', or 'elasticnet'.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Fitted coefficients.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = elastic_net_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_normalized, y_normalized = _normalize_data(X, y, normalize)

    # Initialize parameters
    n_features = X_normalized.shape[1]
    coefs = np.zeros(n_features)

    # Choose solver
    if solver == 'coordinate_descent':
        coefs = _coordinate_descent(X_normalized, y_normalized, alpha_l1, alpha_l2, max_iter, tol)
    elif solver == 'gradient_descent':
        coefs = _gradient_descent(X_normalized, y_normalized, alpha_l1, alpha_l2, max_iter, tol)
    elif solver == 'newton':
        coefs = _newton_method(X_normalized, y_normalized, alpha_l1, alpha_l2, max_iter, tol)
    elif solver == 'closed_form':
        coefs = _closed_form_solution(X_normalized, y_normalized, alpha_l1, alpha_l2)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    y_pred = X_normalized @ coefs
    metrics = _compute_metrics(y_normalized, y_pred, metric, custom_metric)

    # Prepare output
    result = {
        'result': coefs,
        'metrics': metrics,
        'params_used': {
            'alpha_l1': alpha_l1,
            'alpha_l2': alpha_l2,
            'normalize': normalize,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol,
            'metric': metric
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'standard'
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize data based on the specified method."""
    if method == 'none':
        return X, y
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_normalized = (X - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_normalized, y

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha_l1: float,
    alpha_l2: float,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Coordinate descent solver for Elastic Net."""
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)
    for _ in range(max_iter):
        old_coefs = coefs.copy()
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - np.dot(X, coefs) + coefs[j] * X_j
            corr = np.dot(X_j, residuals)
            rss = np.sum(residuals**2)
            if alpha_l1 > 0:
                coefs[j] = _soft_threshold(corr, alpha_l1)
            else:
                coefs[j] = corr / (np.sum(X_j**2) + alpha_l2)
        if np.linalg.norm(coefs - old_coefs) < tol:
            break
    return coefs

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha_l1: float,
    alpha_l2: float,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Gradient descent solver for Elastic Net."""
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)
    learning_rate = 0.1
    for _ in range(max_iter):
        old_coefs = coefs.copy()
        gradient = 2 * np.dot(X.T, np.dot(X, coefs) - y)
        if alpha_l1 > 0:
            gradient += alpha_l1 * np.sign(coefs)
        if alpha_l2 > 0:
            gradient += 2 * alpha_l2 * coefs
        coefs -= learning_rate * gradient
        if np.linalg.norm(coefs - old_coefs) < tol:
            break
    return coefs

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    alpha_l1: float,
    alpha_l2: float,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Newton's method solver for Elastic Net."""
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)
    for _ in range(max_iter):
        old_coefs = coefs.copy()
        gradient = 2 * np.dot(X.T, np.dot(X, coefs) - y)
        if alpha_l1 > 0:
            gradient += alpha_l1 * np.sign(coefs)
        if alpha_l2 > 0:
            gradient += 2 * alpha_l2 * coefs
        hessian = 2 * np.dot(X.T, X)
        if alpha_l2 > 0:
            hessian += 2 * alpha_l2 * np.eye(n_features)
        coefs -= np.linalg.solve(hessian, gradient)
        if np.linalg.norm(coefs - old_coefs) < tol:
            break
    return coefs

def _closed_form_solution(
    X: np.ndarray,
    y: np.ndarray,
    alpha_l1: float,
    alpha_l2: float
) -> np.ndarray:
    """Closed form solution for Elastic Net (when possible)."""
    if alpha_l1 > 0:
        raise ValueError("Closed form solution not available for L1 regularization.")
    XtX = np.dot(X.T, X)
    if alpha_l2 > 0:
        XtX += alpha_l2 * np.eye(X.shape[1])
    coefs = np.linalg.solve(XtX, np.dot(X.T, y))
    return coefs

def _soft_threshold(rho: float, alpha: float) -> float:
    """Soft thresholding operator."""
    if rho < -alpha:
        return rho + alpha
    elif rho > alpha:
        return rho - alpha
    else:
        return 0.0

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = 'mse',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute metrics for the regression model."""
    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - ss_res / (ss_tot + 1e-8)
    elif metric == 'logloss':
        metrics['logloss'] = -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)
    return metrics

################################################################################
# penalisation_l2
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def penalisation_l2_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    normalizer: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_normalizer: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit a L2 regularized regression model.

    Parameters:
    -----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    alpha : float, optional
        Regularization strength (default=1.0)
    normalizer : str or callable, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') or custom callable (default='standard')
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2') or custom callable (default='mse')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent') (default='closed_form')
    tol : float, optional
        Tolerance for stopping criteria (default=1e-4)
    max_iter : int, optional
        Maximum number of iterations (default=1000)
    custom_normalizer : callable, optional
        Custom normalizer function (default=None)
    custom_metric : callable, optional
        Custom metric function (default=None)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if needed
    X_norm, normalizer_used = _apply_normalization(X, y, normalizer, custom_normalizer)

    # Choose solver
    if solver == 'closed_form':
        coefs = _closed_form_solution(X_norm, y, alpha)
    elif solver == 'gradient_descent':
        coefs = _gradient_descent(X_norm, y, alpha, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(X_norm, y, coefs, metric, custom_metric)

    return {
        'result': {'coefficients': coefs},
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'normalizer': normalizer_used,
            'solver': solver
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
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

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: str,
    custom_normalizer: Optional[Callable]
) -> tuple:
    """Apply normalization to data."""
    if custom_normalizer is not None:
        X_norm = custom_normalizer(X)
        y_norm = y.copy()  # Custom normalizers should handle y if needed
        return X_norm, 'custom'

    if normalizer == 'none':
        return X.copy(), 'none'
    elif normalizer == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif normalizer == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalizer == 'robust':
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        raise ValueError(f"Unknown normalizer: {normalizer}")

    return X_norm, y_norm, normalizer

def _closed_form_solution(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Compute closed form solution for L2 regularized regression."""
    n_features = X.shape[1]
    identity = np.eye(n_features)
    coefs = np.linalg.inv(X.T @ X + alpha * identity) @ X.T @ y
    return coefs

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent solver for L2 regularized regression."""
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)
    learning_rate = 1.0 / (alpha * n_samples)

    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ coefs - y) + 2 * alpha * coefs
        new_coefs = coefs - learning_rate * gradient

        if np.linalg.norm(new_coefs - coefs) < tol:
            break

        coefs = new_coefs

    return coefs

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coefs: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate evaluation metrics."""
    y_pred = X @ coefs

    if custom_metric is not None:
        return {'custom': custom_metric(y, y_pred)}

    metrics = {}
    if metric == 'mse' or 'mse' in metric:
        metrics['mse'] = np.mean((y - y_pred) ** 2)
    if metric == 'mae' or 'mae' in metric:
        metrics['mae'] = np.mean(np.abs(y - y_pred))
    if metric == 'r2' or 'r2' in metric:
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.random.rand(100)

result = penalisation_l2_fit(
    X, y,
    alpha=0.1,
    normalizer='standard',
    metric='mse',
    solver='closed_form'
)
"""

################################################################################
# penalisation_l1
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def penalisation_l1_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'coordinate_descent',
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit a L1-penalized regression model.

    Parameters:
    -----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    alpha : float, default=1.0
        Regularization strength
    normalize : str, default='standard'
        Normalization method: 'none', 'standard', 'minmax', or 'robust'
    metric : str or callable, default='mse'
        Metric to evaluate: 'mse', 'mae', 'r2', or custom callable
    solver : str, default='coordinate_descent'
        Solver method: 'closed_form', 'gradient_descent', or 'coordinate_descent'
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for stopping criteria
    random_state : int or None, default=None
        Random seed for reproducibility
    custom_metric : callable or None, default=None
        Custom metric function if needed

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = penalisation_l1_fit(X, y, alpha=0.5)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if needed
    X_norm, y_norm = _normalize_data(X, y, normalize)

    # Initialize parameters
    n_features = X_norm.shape[1]
    coefs = np.zeros(n_features)

    # Choose solver
    if solver == 'closed_form':
        coefs = _closed_form_solution(X_norm, y_norm, alpha)
    elif solver == 'gradient_descent':
        coefs = _gradient_descent(
            X_norm, y_norm, alpha,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
    elif solver == 'coordinate_descent':
        coefs = _coordinate_descent(
            X_norm, y_norm, alpha,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(
        X_norm, y_norm,
        coefs,
        metric=metric,
        custom_metric=custom_metric
    )

    # Prepare output
    result = {
        'result': coefs,
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'normalize': normalize,
            'metric': metric,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol
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

def _normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'standard'
) -> tuple:
    """Normalize data according to specified method."""
    if method == 'none':
        return X, y

    X_norm = np.copy(X)
    y_norm = np.copy(y)

    if method == 'standard':
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_norm = (X - X_mean) / (X_std + 1e-8)
        y_norm = (y - y.mean()) / (y.std() + 1e-8)

    elif method == 'minmax':
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)

    elif method == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - X_median) / (X_iqr + 1e-8)
        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        y_norm = (y - y_median) / (y_iqr + 1e-8)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_norm, y_norm

def _closed_form_solution(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float
) -> np.ndarray:
    """Closed form solution for L1 penalized regression."""
    # This is a placeholder - actual implementation would use coordinate descent
    # or other methods since L1 doesn't have a closed form solution
    raise NotImplementedError("Closed form solution not implemented for L1")

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Gradient descent solver for L1 penalized regression."""
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    coefs = np.random.randn(n_features)

    for _ in range(max_iter):
        gradient = X.T @ (X @ coefs - y) / n_samples
        # Subgradient for L1 penalty
        subgrad = alpha * np.sign(coefs)
        # Update coefficients
        coefs_new = coefs - 0.1 * (gradient + subgrad)

        if np.linalg.norm(coefs_new - coefs) < tol:
            break
        coefs = coefs_new

    return coefs

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Coordinate descent solver for L1 penalized regression."""
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)
    prev_coefs = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            r = y - X @ coefs + X_j * coefs[j]

            # Soft-thresholding operator
            rho = np.dot(X_j, r)
            coefs[j] = _soft_threshold(rho / np.dot(X_j, X_j), alpha)

        if np.linalg.norm(coefs - prev_coefs) < tol:
            break
        prev_coefs = coefs.copy()

    return coefs

def _soft_threshold(rho: float, alpha: float) -> float:
    """Soft-thresholding operator."""
    if rho < -alpha:
        return rho + alpha
    elif rho > alpha:
        return rho - alpha
    else:
        return 0.0

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coefs: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Calculate metrics for the regression."""
    y_pred = X @ coefs
    metrics = {}

    if metric == 'mse' or (custom_metric is None and metric == 'mse'):
        metrics['mse'] = np.mean((y - y_pred) ** 2)
    if metric == 'mae' or (custom_metric is None and metric == 'mae'):
        metrics['mae'] = np.mean(np.abs(y - y_pred))
    if metric == 'r2' or (custom_metric is None and metric == 'r2'):
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics['r2'] = 1 - ss_res / (ss_tot + 1e-8)

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y, y_pred)

    return metrics

################################################################################
# hyperparametre_alpha
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def hyperparametre_alpha_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha_range: np.ndarray,
    solver: str = 'closed_form',
    metric: Union[str, Callable] = 'mse',
    normalization: str = 'standard',
    regularization: str = 'l2',
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Find the optimal alpha hyperparameter for regularized regression.

    Parameters:
    -----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    alpha_range : np.ndarray
        Array of alpha values to test
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', etc.)
    metric : str or callable, optional
        Metric to optimize ('mse', 'mae', etc.) or custom function
    normalization : str, optional
        Normalization method ('none', 'standard', etc.)
    regularization : str, optional
        Regularization type ('l1', 'l2', etc.)
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    custom_metric : callable, optional
        Custom metric function if needed

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> alphas = np.logspace(-4, 2, 50)
    >>> result = hyperparametre_alpha_fit(X, y, alphas)
    """
    # Input validation
    _validate_inputs(X, y, alpha_range)

    # Normalize data if needed
    X_norm, y_norm = _apply_normalization(X, y, normalization)

    # Initialize results storage
    results = {
        'result': {},
        'metrics': {},
        'params_used': {
            'solver': solver,
            'metric': metric,
            'normalization': normalization,
            'regularization': regularization
        },
        'warnings': []
    }

    # Get metric function
    metric_func = _get_metric_function(metric, custom_metric)

    # Find optimal alpha
    best_alpha = _find_optimal_alpha(
        X_norm, y_norm,
        alpha_range,
        solver,
        metric_func,
        regularization,
        tol,
        max_iter
    )

    # Store results
    results['result']['optimal_alpha'] = best_alpha
    results['metrics']['best_metric_value'] = metric_func(
        y_norm,
        _predict(X_norm, best_alpha, solver, regularization)
    )

    return results

def _validate_inputs(X: np.ndarray, y: np.ndarray, alpha_range: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")
    if np.any(alpha_range <= 0):
        raise ValueError("Alpha values must be positive")

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> tuple:
    """Apply selected normalization to data."""
    if method == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif method == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == 'robust':
        X_norm = (X - np.median(X, axis=0)) / (
            np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        )
        y_norm = (y - np.median(y)) / (
            np.percentile(y, 75) - np.percentile(y, 25)
        )
    else:
        X_norm, y_norm = X.copy(), y.copy()
    return X_norm, y_norm

def _get_metric_function(
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Callable:
    """Get the appropriate metric function."""
    if custom_metric is not None:
        return custom_metric

    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }

    if isinstance(metric, str):
        return metrics.get(metric.lower(), _mean_squared_error)
    else:
        raise ValueError("Invalid metric specified")

def _find_optimal_alpha(
    X: np.ndarray,
    y: np.ndarray,
    alpha_range: np.ndarray,
    solver: str,
    metric_func: Callable,
    regularization: str,
    tol: float,
    max_iter: int
) -> float:
    """Find the alpha that minimizes the selected metric."""
    best_alpha = None
    best_metric = float('inf')

    for alpha in alpha_range:
        try:
            coefs = _fit_model(X, y, alpha, solver, regularization, tol, max_iter)
            predictions = _predict(X, coefs, solver, regularization)
            current_metric = metric_func(y, predictions)

            if current_metric < best_metric:
                best_metric = current_metric
                best_alpha = alpha

        except Exception as e:
            continue

    if best_alpha is None:
        raise RuntimeError("No valid alpha found")

    return best_alpha

def _fit_model(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    solver: str,
    regularization: str,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Fit the regularized regression model."""
    if solver == 'closed_form':
        return _closed_form_solution(X, y, alpha, regularization)
    elif solver == 'gradient_descent':
        return _gradient_descent(X, y, alpha, regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _closed_form_solution(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    regularization: str
) -> np.ndarray:
    """Closed form solution for ridge regression."""
    if regularization == 'l2':
        I = np.eye(X.shape[1])
        coefs = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    elif regularization == 'l1':
        # For Lasso, we would typically use coordinate descent
        raise NotImplementedError("L1 regularization not implemented in closed form")
    else:
        coefs = np.linalg.inv(X.T @ X) @ X.T @ y
    return coefs

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    regularization: str,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent solver for regularized regression."""
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)
    learning_rate = 0.1

    for _ in range(max_iter):
        gradient = X.T @ (X @ coefs - y) / n_samples

        if regularization == 'l2':
            gradient += 2 * alpha * coefs
        elif regularization == 'l1':
            gradient += alpha * np.sign(coefs)

        new_coefs = coefs - learning_rate * gradient

        if np.linalg.norm(new_coefs - coefs) < tol:
            break

        coefs = new_coefs

    return coefs

def _predict(
    X: np.ndarray,
    coefs: np.ndarray,
    solver: str,
    regularization: str
) -> np.ndarray:
    """Make predictions using the fitted model."""
    return X @ coefs

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

################################################################################
# selection_de_variables
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def selection_de_variables_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict:
    """
    Perform variable selection in regularized regression.

    Parameters:
    -----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalizer : callable, optional
        Function to normalize features. Default is None.
    metric : str or callable
        Metric to evaluate model performance. Options: 'mse', 'mae', 'r2'.
        Can also be a custom callable.
    solver : str
        Solver to use. Options: 'closed_form', 'gradient_descent'.
    regularization : str, optional
        Type of regularization. Options: 'l1', 'l2'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    **kwargs :
        Additional solver-specific parameters.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, used parameters and warnings.

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = selection_de_variables_fit(X, y, metric='r2', solver='gradient_descent')
    """
    # Input validation
    _validate_inputs(X, y)

    # Normalize data if requested
    X_norm = _apply_normalization(X, normalizer)

    # Prepare metric function
    metric_func = _prepare_metric(metric, custom_metric)

    # Select solver and run
    if solver == 'closed_form':
        result = _solve_closed_form(X_norm, y, regularization, **kwargs)
    elif solver == 'gradient_descent':
        result = _solve_gradient_descent(X_norm, y, metric_func,
                                        regularization, tol, max_iter, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(y, result['predictions'], metric_func)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or Inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or Inf values")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _prepare_metric(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Prepare metric function."""
    if callable(metric):
        return metric
    if custom_metric is not None:
        return custom_metric

    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }

    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _solve_closed_form(X: np.ndarray, y: np.ndarray,
                      regularization: Optional[str], **kwargs) -> Dict:
    """Solve using closed form solution."""
    if regularization == 'l2':
        alpha = kwargs.get('alpha', 1.0)
        I = np.eye(X.shape[1])
        coefficients = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    elif regularization == 'l1':
        # This would typically use a specialized solver like coordinate descent
        raise NotImplementedError("L1 regularization not implemented in closed form")
    else:
        coefficients = np.linalg.pinv(X) @ y

    predictions = X @ coefficients
    return {
        'coefficients': coefficients,
        'predictions': predictions
    }

def _solve_gradient_descent(X: np.ndarray, y: np.ndarray,
                           metric_func: Callable,
                           regularization: Optional[str],
                           tol: float, max_iter: int, **kwargs) -> Dict:
    """Solve using gradient descent."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    learning_rate = kwargs.get('learning_rate', 0.01)

    for _ in range(max_iter):
        gradients = _compute_gradients(X, y, coefficients, regularization)

        new_coefficients = coefficients - learning_rate * gradients
        predictions = X @ new_coefficients

        if metric_func(y, predictions) < tol:
            break

        coefficients = new_coefficients

    return {
        'coefficients': coefficients,
        'predictions': X @ coefficients
    }

def _compute_gradients(X: np.ndarray, y: np.ndarray,
                      coefficients: np.ndarray,
                      regularization: Optional[str]) -> np.ndarray:
    """Compute gradients for gradient descent."""
    residuals = X @ coefficients - y
    gradients = 2 * (X.T @ residuals) / len(y)

    if regularization == 'l1':
        gradients += np.sign(coefficients)
    elif regularization == 'l2':
        gradients += 2 * coefficients

    return gradients

def _calculate_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      metric_func: Callable) -> Dict:
    """Calculate evaluation metrics."""
    return {
        'metric': metric_func(y_true, y_pred),
        'mse': _mean_squared_error(y_true, y_pred),
        'mae': _mean_absolute_error(y_true, y_pred)
    }

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

################################################################################
# shrinkage
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input matrices and vectors."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _normalize_data(X: np.ndarray, y: np.ndarray,
                   normalization: str = 'standard',
                   custom_norm: Optional[Callable] = None) -> tuple:
    """Normalize data according to specified method."""
    if custom_norm is not None:
        X_norm, y_norm = custom_norm(X, y)
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
        X_norm, y_norm = X.copy(), y.copy()
    return X_norm, y_norm

def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
                   metric: str = 'mse',
                   custom_metric: Optional[Callable] = None) -> float:
    """Compute specified metric between true and predicted values."""
    if custom_metric is not None:
        return custom_metric(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    elif metric == 'logloss':
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _shrinkage_closed_form(X: np.ndarray, y: np.ndarray,
                          alpha: float = 1.0) -> np.ndarray:
    """Compute shrinkage coefficients using closed-form solution."""
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.solve(XtX + alpha * np.eye(X.shape[1]), Xty)

def _shrinkage_gradient_descent(X: np.ndarray, y: np.ndarray,
                              alpha: float = 1.0,
                              lr: float = 0.01,
                              max_iter: int = 1000,
                              tol: float = 1e-4) -> np.ndarray:
    """Compute shrinkage coefficients using gradient descent."""
    n_features = X.shape[1]
    coefs = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * (X @ coefs - y) + 2 * alpha * coefs
        new_coefs = coefs - lr * gradient
        if np.linalg.norm(new_coefs - coefs) < tol:
            break
        coefs = new_coefs
    return coefs

def shrinkage_fit(X: np.ndarray, y: np.ndarray,
                 solver: str = 'closed_form',
                 normalization: str = 'standard',
                 metric: str = 'mse',
                 alpha: float = 1.0,
                 custom_norm: Optional[Callable] = None,
                 custom_metric: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Perform shrinkage regression with specified parameters.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    solver : str or callable
        Solver to use ('closed_form', 'gradient_descent') or custom function
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str
        Metric to compute ('mse', 'mae', 'r2', 'logloss')
    alpha : float
        Regularization parameter
    custom_norm : callable, optional
        Custom normalization function
    custom_metric : callable, optional
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = shrinkage_fit(X, y, solver='closed_form', alpha=1.0)
    """
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization, custom_norm)

    # Choose solver
    if callable(solver):
        coefs = solver(X_norm, y_norm, alpha)
    elif solver == 'closed_form':
        coefs = _shrinkage_closed_form(X_norm, y_norm, alpha)
    elif solver == 'gradient_descent':
        coefs = _shrinkage_gradient_descent(X_norm, y_norm, alpha)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute predictions and metrics
    y_pred = X_norm @ coefs
    main_metric = _compute_metric(y_norm, y_pred, metric, custom_metric)

    return {
        'result': {
            'coefficients': coefs,
            'predictions': y_pred
        },
        'metrics': {
            metric: main_metric
        },
        'params_used': {
            'solver': solver,
            'normalization': normalization,
            'metric': metric,
            'alpha': alpha
        },
        'warnings': []
    }

################################################################################
# bias_variance_tradeoff
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def bias_variance_tradeoff_fit(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    metric: Union[str, Callable] = 'mse',
    n_splits: int = 5,
    normalizer: Optional[Callable] = None,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Compute bias-variance tradeoff for a given model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    model : Callable
        A callable that implements fit and predict methods.
    metric : Union[str, Callable], optional
        Metric to evaluate the model. Can be 'mse', 'mae', 'r2' or a custom callable.
        Default is 'mse'.
    n_splits : int, optional
        Number of splits for cross-validation. Default is 5.
    normalizer : Optional[Callable], optional
        A callable that normalizes the data. Default is None.
    random_state : Optional[int], optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        A dictionary containing:
        - "result": Dictionary with bias and variance estimates.
        - "metrics": Dictionary of computed metrics.
        - "params_used": Dictionary of parameters used.
        - "warnings": List of warnings encountered.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = bias_variance_tradeoff_fit(X, y, LinearRegression())
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize warnings list
    warnings = []

    # Normalize data if normalizer is provided
    X_normalized, y_normalized = _apply_normalization(X, y, normalizer)

    # Initialize metrics
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    # Perform cross-validation
    bias, variance = _compute_bias_variance(
        X_normalized,
        y_normalized,
        model,
        metric_func,
        n_splits=n_splits,
        random_state=random_state
    )

    # Compute additional metrics if needed
    metrics = {
        'bias': bias,
        'variance': variance,
        'metric_used': str(metric)
    }

    # Prepare output
    result = {
        "result": {
            "bias": bias,
            "variance": variance
        },
        "metrics": metrics,
        "params_used": {
            "n_splits": n_splits,
            "metric": str(metric),
            "normalizer": normalizer.__name__ if normalizer else None
        },
        "warnings": warnings
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable]
) -> tuple:
    """Apply normalization to the data."""
    if normalizer is not None:
        X = normalizer(X)
        y = normalizer(y.reshape(-1, 1)).flatten()
    return X, y

def _get_metric_function(metric: str) -> Callable:
    """Get the metric function based on the input string."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

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

def _compute_bias_variance(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    metric_func: Callable,
    n_splits: int = 5,
    random_state: Optional[int] = None
) -> tuple:
    """Compute bias and variance using cross-validation."""
    np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    # Initialize arrays to store predictions
    y_preds = np.zeros((n_splits, n_samples))

    for i in range(n_splits):
        # Split data into train and test sets
        test_idx = np.random.choice(indices, size=n_samples // n_splits, replace=False)
        train_idx = np.setdiff1d(indices, test_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Fit model and predict
        model.fit(X_train, y_train)
        y_preds[i, test_idx] = model.predict(X_test)

    # Compute bias and variance
    y_mean = np.mean(y_preds, axis=0)
    bias_squared = metric_func(y, y_mean)
    variance = np.mean(np.var(y_preds, axis=0))

    return bias_squared, variance

################################################################################
# coefficients_parsimonieux
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def coefficients_parsimonieux_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalisation: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularisation: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    alpha: float = 1.0,
    l1_ratio: Optional[float] = None,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Estimate parsimonious coefficients for regularized regression.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalisation : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to evaluate model performance ('mse', 'mae', 'r2', 'logloss').
    distance : str, optional
        Distance metric for regularization ('euclidean', 'manhattan', 'cosine').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton').
    regularisation : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for stopping criteria.
    max_iter : int, optional
        Maximum number of iterations.
    alpha : float, optional
        Regularization strength.
    l1_ratio : float, optional
        ElasticNet mixing parameter (0 <= l1_ratio <= 1).
    custom_metric : callable, optional
        Custom metric function.
    custom_distance : callable, optional
        Custom distance function.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Estimated coefficients
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in fitting
        - 'warnings': Any warnings generated

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = coefficients_parsimonieux_fit(X, y,
    ...                                       normalisation='standard',
    ...                                       regularisation='l1')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _apply_normalisation(X, y, normalisation)

    # Initialize parameters
    params_used = {
        'normalisation': normalisation,
        'metric': metric,
        'distance': distance,
        'solver': solver,
        'regularisation': regularisation,
        'tol': tol,
        'max_iter': max_iter,
        'alpha': alpha
    }

    if regularisation == 'elasticnet':
        params_used['l1_ratio'] = l1_ratio

    # Select metric function
    metric_func = _get_metric_function(metric, custom_metric)

    # Select distance function
    distance_func = _get_distance_function(distance, custom_distance)

    # Solve for coefficients
    if solver == 'closed_form':
        coef = _solve_closed_form(X_norm, y_norm,
                                 regularisation=regularisation,
                                 alpha=alpha,
                                 l1_ratio=l1_ratio)
    elif solver == 'gradient_descent':
        coef = _solve_gradient_descent(X_norm, y_norm,
                                      metric_func=metric_func,
                                      distance_func=distance_func,
                                      regularisation=regularisation,
                                      alpha=alpha,
                                      tol=tol,
                                      max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(y, y_norm @ coef, metric_func)

    # Prepare output
    result = {
        'result': coef,
        'metrics': metrics,
        'params_used': params_used,
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

def _apply_normalisation(X: np.ndarray, y: np.ndarray, method: str) -> tuple:
    """Apply specified normalisation to data."""
    if method == 'none':
        return X, y
    elif method == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif method == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == 'robust':
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        raise ValueError(f"Unknown normalisation method: {method}")
    return X_norm, y_norm

def _get_metric_function(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Get metric function based on input."""
    if custom_metric is not None:
        return custom_metric

    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared,
        'logloss': _log_loss
    }

    if isinstance(metric, str):
        return metrics.get(metric.lower(), _mean_squared_error)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _get_distance_function(distance: str, custom_distance: Optional[Callable]) -> Callable:
    """Get distance function based on input."""
    if custom_distance is not None:
        return custom_distance

    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance
    }

    return distances.get(distance.lower(), _euclidean_distance)

def _solve_closed_form(X: np.ndarray, y: np.ndarray,
                      regularisation: Optional[str],
                      alpha: float,
                      l1_ratio: Optional[float]) -> np.ndarray:
    """Solve for coefficients using closed-form solution."""
    n_features = X.shape[1]
    identity = np.eye(n_features)

    if regularisation == 'l2':
        coef = np.linalg.inv(X.T @ X + alpha * identity) @ X.T @ y
    elif regularisation == 'l1':
        # For simplicity, using coordinate descent for L1
        coef = _solve_coordinate_descent(X, y,
                                        regularisation='l1',
                                        alpha=alpha)
    elif regularisation == 'elasticnet':
        if l1_ratio is None:
            raise ValueError("l1_ratio must be specified for elasticnet")
        coef = _solve_coordinate_descent(X, y,
                                        regularisation='elasticnet',
                                        alpha=alpha,
                                        l1_ratio=l1_ratio)
    else:
        coef = np.linalg.inv(X.T @ X) @ X.T @ y

    return coef

def _solve_gradient_descent(X: np.ndarray, y: np.ndarray,
                           metric_func: Callable,
                           distance_func: Callable,
                           regularisation: Optional[str],
                           alpha: float,
                           tol: float,
                           max_iter: int) -> np.ndarray:
    """Solve for coefficients using gradient descent."""
    n_features = X.shape[1]
    coef = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ coef - y)

        if regularisation == 'l1':
            gradient += alpha * np.sign(coef)
        elif regularisation == 'l2':
            gradient += 2 * alpha * coef
        elif regularisation == 'elasticnet':
            gradient += alpha * (l1_ratio * np.sign(coef) + (1 - l1_ratio) * 2 * coef)

        coef -= gradient

        current_loss = metric_func(y, X @ coef)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return coef

def _solve_coordinate_descent(X: np.ndarray, y: np.ndarray,
                             regularisation: str,
                             alpha: float,
                             l1_ratio: Optional[float] = None) -> np.ndarray:
    """Solve for coefficients using coordinate descent."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)

    for _ in range(10 * n_features):  # Arbitrary number of passes
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - np.dot(X, coef) + coef[j] * X_j

            if regularisation == 'l1':
                rho = alpha
            elif regularisation == 'elasticnet' and l1_ratio is not None:
                rho = alpha * l1_ratio
            else:
                rho = 0

            coef[j] = _soft_threshold(np.dot(X_j, residuals), rho)

    return coef

def _soft_threshold(rho: float, alpha: float) -> float:
    """Soft thresholding operator."""
    if rho < -alpha:
        return rho + alpha
    elif rho > alpha:
        return rho - alpha
    else:
        return 0

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
    return 1 - ss_res / ss_tot

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log loss."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
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

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric_func: Callable) -> Dict:
    """Compute all metrics."""
    return {
        'mse': _mean_squared_error(y_true, y_pred),
        'mae': _mean_absolute_error(y_true, y_pred),
        'r2': _r_squared(y_true, y_pred),
        'metric_custom': metric_func(y_true, y_pred)
    }
