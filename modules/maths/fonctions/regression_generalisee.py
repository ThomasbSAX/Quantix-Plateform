"""
Quantix – Module regression_generalisee
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# modele_lineaire
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
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

def normalize_data(X: np.ndarray, y: np.ndarray,
                   normalization: str = 'standard',
                   custom_normalize: Optional[Callable] = None) -> tuple:
    """Normalize input data."""
    if custom_normalize is not None:
        X_norm = custom_normalize(X)
        y_norm = custom_normalize(y.reshape(-1, 1)).flatten()
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
        X_norm = X.copy()
        y_norm = y.copy()
    return X_norm, y_norm

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metrics: Union[str, list] = 'mse',
                    custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute regression metrics."""
    result = {}
    if isinstance(metrics, str):
        metrics = [metrics]

    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    def logloss(y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    metric_functions = {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'logloss': logloss
    }

    for metric in metrics:
        if metric in metric_functions:
            result[metric] = metric_functions[metric](y_true, y_pred)
        elif custom_metric is not None and metric == 'custom':
            result['custom'] = custom_metric(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return result

def closed_form_solver(X: np.ndarray, y: np.ndarray,
                      regularization: str = 'none',
                      alpha: float = 1.0) -> np.ndarray:
    """Closed form solution for linear regression."""
    if regularization == 'l1':
        # Lasso - would need a different solver
        raise NotImplementedError("L1 regularization not implemented in closed form.")
    elif regularization == 'l2':
        # Ridge
        I = np.eye(X.shape[1])
        coefficients = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    elif regularization == 'elasticnet':
        # ElasticNet - would need a different solver
        raise NotImplementedError("ElasticNet regularization not implemented in closed form.")
    else:
        coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
    return coefficients

def gradient_descent_solver(X: np.ndarray, y: np.ndarray,
                           learning_rate: float = 0.01,
                           n_iterations: int = 1000,
                           tol: float = 1e-4,
                           regularization: str = 'none',
                           alpha: float = 1.0) -> np.ndarray:
    """Gradient descent solver for linear regression."""
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    prev_cost = float('inf')

    for _ in range(n_iterations):
        gradients = (2/n_samples) * X.T @ (X @ coefficients - y)

        if regularization == 'l2':
            gradients += (2 * alpha / n_samples) * coefficients
        elif regularization == 'l1':
            gradients += (alpha / n_samples) * np.sign(coefficients)

        coefficients -= learning_rate * gradients

        current_cost = np.mean((X @ coefficients - y) ** 2)
        if abs(prev_cost - current_cost) < tol:
            break
        prev_cost = current_cost

    return coefficients

def modele_lineaire_fit(X: np.ndarray, y: np.ndarray,
                       normalization: str = 'standard',
                       metrics: Union[str, list] = 'mse',
                       solver: str = 'closed_form',
                       regularization: str = 'none',
                       alpha: float = 1.0,
                       custom_normalize: Optional[Callable] = None,
                       custom_metric: Optional[Callable] = None) -> Dict:
    """
    Fit a linear regression model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalization : str or callable, optional
        Normalization method ('standard', 'minmax', 'robust') or custom function
    metrics : str, list of str, optional
        Metrics to compute ('mse', 'mae', 'r2', 'logloss') or custom function
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    alpha : float, optional
        Regularization strength
    custom_normalize : callable, optional
        Custom normalization function
    custom_metric : callable, optional
        Custom metric function

    Returns
    -------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = modele_lineaire_fit(X, y)
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization, custom_normalize)

    # Solve for coefficients
    if solver == 'closed_form':
        coefficients = closed_form_solver(X_norm, y_norm, regularization, alpha)
    elif solver == 'gradient_descent':
        coefficients = gradient_descent_solver(X_norm, y_norm,
                                             regularization=regularization,
                                             alpha=alpha)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Make predictions
    y_pred = X_norm @ coefficients

    # Compute metrics
    computed_metrics = compute_metrics(y_norm, y_pred, metrics, custom_metric)

    # Prepare result
    result = {
        'result': {
            'coefficients': coefficients,
            'predictions': y_pred
        },
        'metrics': computed_metrics,
        'params_used': {
            'normalization': normalization if custom_normalize is None else 'custom',
            'metrics': metrics,
            'solver': solver,
            'regularization': regularization,
            'alpha': alpha
        },
        'warnings': []
    }

    return result

################################################################################
# modele_logistique
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def modele_logistique_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalisation: str = 'standard',
    solver: str = 'gradient_descent',
    metric: Union[str, Callable] = 'logloss',
    penalty: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a logistic regression model with configurable options.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, 1)
    normalisation : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'
    solver : str
        Solver method: 'gradient_descent', 'newton', or 'coordinate_descent'
    metric : str or callable
        Evaluation metric: 'mse', 'mae', 'r2', 'logloss', or custom callable
    penalty : str, optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'
    tol : float
        Tolerance for stopping criteria
    max_iter : int
        Maximum number of iterations
    learning_rate : float
        Learning rate for gradient descent
    custom_metric : callable, optional
        Custom metric function
    custom_distance : callable, optional
        Custom distance function

    Returns:
    --------
    Dict containing:
        - 'result': fitted model parameters
        - 'metrics': computed metrics
        - 'params_used': configuration used
        - 'warnings': any warnings encountered

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = modele_logistique_fit(X, y, normalisation='standard', solver='gradient_descent')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalization(X, normalisation)

    # Initialize parameters
    n_features = X_normalized.shape[1]
    coefs = np.zeros(n_features)
    intercept = 0.0

    # Choose solver
    if solver == 'gradient_descent':
        coefs, intercept = _gradient_descent(
            X_normalized, y,
            penalty=penalty,
            tol=tol,
            max_iter=max_iter,
            learning_rate=learning_rate
        )
    elif solver == 'newton':
        coefs, intercept = _newton_method(
            X_normalized, y,
            penalty=penalty,
            tol=tol,
            max_iter=max_iter
        )
    elif solver == 'coordinate_descent':
        coefs, intercept = _coordinate_descent(
            X_normalized, y,
            penalty=penalty,
            tol=tol,
            max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    y_pred = _sigmoid(np.dot(X_normalized, coefs) + intercept)
    metrics = _compute_metrics(y, y_pred, metric=metric, custom_metric=custom_metric)

    return {
        'result': {'coefs': coefs, 'intercept': intercept},
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'solver': solver,
            'metric': metric,
            'penalty': penalty,
            'tol': tol,
            'max_iter': max_iter,
            'learning_rate': learning_rate
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if y.ndim not in (1, 2) or (y.ndim == 2 and y.shape[1] != 1):
        raise ValueError("y must be 1-dimensional or (n_samples, 1)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to input data."""
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

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-z))

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    penalty: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01
) -> tuple:
    """Gradient descent optimization for logistic regression."""
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)
    intercept = 0.0

    for _ in range(max_iter):
        # Compute predictions
        z = np.dot(X, coefs) + intercept
        y_pred = _sigmoid(z)

        # Compute gradients
        gradient = np.dot(X.T, (y_pred - y)) / n_samples

        # Add regularization if needed
        if penalty == 'l1':
            gradient += np.sign(coefs)
        elif penalty == 'l2':
            gradient += 2 * coefs
        elif penalty == 'elasticnet':
            gradient += np.sign(coefs) + 2 * coefs

        # Update parameters
        old_coefs = coefs.copy()
        coefs -= learning_rate * gradient
        intercept -= learning_rate * np.mean(y_pred - y)

        # Check convergence
        if np.linalg.norm(coefs - old_coefs) < tol:
            break

    return coefs, intercept

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    *,
    penalty: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> tuple:
    """Newton's method optimization for logistic regression."""
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)
    intercept = 0.0

    for _ in range(max_iter):
        # Compute predictions
        z = np.dot(X, coefs) + intercept
        y_pred = _sigmoid(z)

        # Compute gradient and Hessian
        gradient = np.dot(X.T, (y_pred - y)) / n_samples

        # Add regularization if needed
        if penalty == 'l2':
            gradient += 2 * coefs

        hessian = np.dot(X.T * (y_pred * (1 - y_pred)), X) / n_samples

        if penalty == 'l2':
            hessian += 2 * np.eye(n_features)

        # Update parameters
        old_coefs = coefs.copy()
        coefs -= np.linalg.solve(hessian, gradient)
        intercept -= np.mean(y_pred - y)

        # Check convergence
        if np.linalg.norm(coefs - old_coefs) < tol:
            break

    return coefs, intercept

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    penalty: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> tuple:
    """Coordinate descent optimization for logistic regression."""
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)
    intercept = 0.0

    for _ in range(max_iter):
        old_coefs = coefs.copy()

        for j in range(n_features):
            # Compute residual without feature j
            z = np.dot(X, coefs) + intercept - X[:, j] * coefs[j]
            y_pred = _sigmoid(z)

            # Compute gradient for feature j
            r_j = np.dot(X[:, j], (y_pred - y))

            # Update coefficient for feature j
            if penalty == 'l1':
                coefs[j] = np.sign(r_j) * max(0, abs(r_j) - 1)
            elif penalty == 'l2':
                coefs[j] = r_j / (1 + 2)
            else:
                coefs[j] = r_j

        # Update intercept
        z = np.dot(X, coefs) + intercept
        y_pred = _sigmoid(z)
        intercept -= np.mean(y_pred - y)

        # Check convergence
        if np.linalg.norm(coefs - old_coefs) < tol:
            break

    return coefs, intercept

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: Union[str, Callable] = 'logloss',
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute specified metrics."""
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
        metrics['logloss'] = -np.mean(y_true * np.log(y_pred + 1e-8) +
                                    (1 - y_true) * np.log(1 - y_pred + 1e-8))
    elif callable(metric):
        metrics['custom'] = metric(y_true, y_pred)
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)

    return metrics

################################################################################
# regression_poisson
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input matrices and vectors."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X and y must not contain NaN values")
    if np.any(y < 0):
        raise ValueError("y must contain non-negative values for Poisson regression")

def _normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize the input data."""
    if method == 'none':
        return X
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1.0
        return (X - mean) / std
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        return (X - min_val) / range_val
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        iqr[iqr == 0] = 1.0
        return (X - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _poisson_log_likelihood(y: np.ndarray, mu: np.ndarray) -> float:
    """Compute the Poisson log-likelihood."""
    return np.sum(y * np.log(mu) - mu - np.linalg.norm(y))

def _poisson_gradient(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute the gradient of the Poisson log-likelihood."""
    mu = np.exp(X @ beta)
    return X.T @ (y - mu)

def _poisson_hessian(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute the Hessian of the Poisson log-likelihood."""
    mu = np.exp(X @ beta)
    return X.T @ (mu[:, None] * X)

def _newton_raphson(X: np.ndarray, y: np.ndarray,
                    tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    """Newton-Raphson solver for Poisson regression."""
    n_features = X.shape[1]
    beta = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = _poisson_gradient(X, y, beta)
        hessian = _poisson_hessian(X, beta)
        delta = np.linalg.solve(hessian, gradient)
        beta -= delta

        if np.linalg.norm(delta) < tol:
            break

    return beta

def _gradient_descent(X: np.ndarray, y: np.ndarray,
                     learning_rate: float = 0.01, tol: float = 1e-6,
                     max_iter: int = 100) -> np.ndarray:
    """Gradient descent solver for Poisson regression."""
    n_features = X.shape[1]
    beta = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = _poisson_gradient(X, y, beta)
        beta -= learning_rate * gradient

        if np.linalg.norm(gradient) < tol:
            break

    return beta

def regression_poisson_fit(X: np.ndarray, y: np.ndarray,
                          solver: str = 'newton',
                          normalization: str = 'standard',
                          tol: float = 1e-6,
                          max_iter: int = 100) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a Poisson regression model.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    solver : str, optional
        Solver to use ('newton' or 'gradient_descent'), by default 'newton'.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), by default 'standard'.
    tol : float, optional
        Tolerance for convergence, by default 1e-6.
    max_iter : int, optional
        Maximum number of iterations, by default 100.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Estimated coefficients.
        - 'metrics': Dictionary of metrics.
        - 'params_used': Parameters used for fitting.
        - 'warnings': Any warnings encountered.

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.poisson(lam=2, size=100)
    >>> result = regression_poisson_fit(X, y)
    """
    _validate_inputs(X, y)
    X_normalized = _normalize_data(X, normalization)

    if solver == 'newton':
        beta = _newton_raphson(X_normalized, y, tol=tol, max_iter=max_iter)
    elif solver == 'gradient_descent':
        beta = _gradient_descent(X_normalized, y, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    mu = np.exp(X_normalized @ beta)
    log_likelihood = _poisson_log_likelihood(y, mu)

    metrics = {
        'log_likelihood': log_likelihood,
        'deviance': 2 * (np.sum(y * np.log(y / mu)) - (y - mu))
    }

    return {
        'result': beta,
        'metrics': metrics,
        'params_used': {
            'solver': solver,
            'normalization': normalization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

################################################################################
# regression_gamma
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regression_gamma_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = "mse",
    solver: str = "newton",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    weights: Optional[np.ndarray] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit a Gamma regression model to the data.

    Parameters:
    -----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalizer : Optional[Callable]
        Function to normalize the design matrix. Default is None.
    metric : Union[str, Callable]
        Metric to evaluate the model. Can be "mse", "mae", or a custom callable.
    solver : str
        Solver to use. Options are "newton", "gradient_descent".
    regularization : Optional[str]
        Regularization type. Options are "l1", "l2", or None.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    weights : Optional[np.ndarray]
        Weights for the samples. Default is None.
    custom_metric : Optional[Callable]
        Custom metric function.

    Returns:
    --------
    Dict
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y, weights)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize parameters
    params = _initialize_parameters(X_normalized.shape[1])

    # Choose solver
    if solver == "newton":
        params = _newton_solver(X_normalized, y, params, tol, max_iter, regularization)
    elif solver == "gradient_descent":
        params = _gradient_descent_solver(X_normalized, y, params, tol, max_iter, regularization)
    else:
        raise ValueError("Unsupported solver. Choose 'newton' or 'gradient_descent'.")

    # Calculate metrics
    metrics = _calculate_metrics(X_normalized, y, params, metric, custom_metric)

    # Prepare output
    result = {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray]) -> None:
    """Validate the input data."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if weights is not None and X.shape[0] != weights.shape[0]:
        raise ValueError("Weights must have the same number of samples as X and y.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values.")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to the design matrix."""
    if normalizer is not None:
        return normalizer(X)
    return X

def _initialize_parameters(n_features: int) -> np.ndarray:
    """Initialize the parameters."""
    return np.zeros(n_features)

def _newton_solver(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    tol: float,
    max_iter: int,
    regularization: Optional[str]
) -> np.ndarray:
    """Newton's method solver for Gamma regression."""
    for _ in range(max_iter):
        gradient, hessian = _compute_gradient_hessian(X, y, params)
        if regularization == "l2":
            hessian += np.eye(X.shape[1]) * 0.1
        params_new = params - np.linalg.solve(hessian, gradient)
        if np.linalg.norm(params_new - params) < tol:
            break
        params = params_new
    return params

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    tol: float,
    max_iter: int,
    regularization: Optional[str]
) -> np.ndarray:
    """Gradient descent solver for Gamma regression."""
    learning_rate = 0.01
    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, params)
        if regularization == "l1":
            gradient += np.sign(params) * 0.1
        elif regularization == "l2":
            gradient += params * 0.1
        params_new = params - learning_rate * gradient
        if np.linalg.norm(params_new - params) < tol:
            break
        params = params_new
    return params

def _compute_gradient_hessian(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the gradient and Hessian for Newton's method."""
    eta = X @ params
    gradient = X.T @ (y * np.exp(-eta) - 1)
    hessian = X.T @ np.diag(y * np.exp(-eta))
    return gradient, hessian

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray
) -> np.ndarray:
    """Compute the gradient for gradient descent."""
    eta = X @ params
    return X.T @ (y * np.exp(-eta) - 1)

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate the metrics for the model."""
    eta = X @ params
    predictions = np.exp(eta)

    metrics_dict = {}
    if metric == "mse":
        metrics_dict["mse"] = np.mean((y - predictions) ** 2)
    elif metric == "mae":
        metrics_dict["mae"] = np.mean(np.abs(y - predictions))
    elif callable(metric):
        metrics_dict["custom_metric"] = metric(y, predictions)
    if custom_metric is not None:
        metrics_dict["custom_metric"] = custom_metric(y, predictions)

    return metrics_dict

################################################################################
# regression_tweedie
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input matrices and vectors."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X and y must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("X and y must not contain infinite values")

def normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize the input data."""
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

def tweedie_deviance(y: np.ndarray, mu: np.ndarray, p: float) -> np.ndarray:
    """Compute the Tweedie deviance."""
    if p == 1:
        return 2 * np.sum(y * np.log(mu + 1e-8) - (y + 1) * np.log(y + 1e-8))
    elif p == 2:
        return np.sum((y - mu) ** 2)
    else:
        term1 = (np.power(mu, 2 - p) / (2 - p)) * np.exp((p - 2) / (p - 1) * (np.log(y + 1e-8) - np.log(mu + 1e-8)))
        term2 = y * (mu ** (p - 1) / (p - 1)) - mu
        return 2 * np.sum(term1 + term2)

def closed_form_solver(X: np.ndarray, y: np.ndarray, p: float) -> np.ndarray:
    """Closed form solution for Tweedie regression."""
    if p == 2:
        XtX = np.dot(X.T, X)
        Xty = np.dot(X.T, y)
        return np.linalg.solve(XtX + 1e-8 * np.eye(X.shape[1]), Xty)
    else:
        raise NotImplementedError("Closed form solution not implemented for p != 2")

def gradient_descent_solver(X: np.ndarray, y: np.ndarray, p: float,
                           learning_rate: float = 0.01, max_iter: int = 1000,
                           tol: float = 1e-4) -> np.ndarray:
    """Gradient descent solver for Tweedie regression."""
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    for _ in range(max_iter):
        mu = np.exp(np.dot(X, beta))
        gradient = np.dot(X.T, (mu ** (2 - p) - y * mu ** (p - 1))) / n_samples
        beta_new = beta - learning_rate * gradient
        if np.linalg.norm(beta_new - beta) < tol:
            break
        beta = beta_new
    return beta

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   metric_functions: Dict[str, Callable]) -> Dict[str, float]:
    """Compute metrics for the regression model."""
    metrics = {}
    for name, func in metric_functions.items():
        try:
            metrics[name] = func(y_true, y_pred)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def regression_tweedie_fit(X: np.ndarray, y: np.ndarray,
                          p: float = 1.5,
                          normalization: str = 'standard',
                          solver: str = 'gradient_descent',
                          learning_rate: float = 0.01,
                          max_iter: int = 1000,
                          tol: float = 1e-4,
                          metric_functions: Optional[Dict[str, Callable]] = None) -> Dict:
    """
    Fit a Tweedie regression model.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - p: Power parameter for Tweedie distribution
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - solver: Solver method ('closed_form', 'gradient_descent')
    - learning_rate: Learning rate for gradient descent
    - max_iter: Maximum number of iterations
    - tol: Tolerance for convergence
    - metric_functions: Dictionary of metric functions

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm = normalize_data(X, normalization)

    # Initialize metric functions if not provided
    if metric_functions is None:
        metric_functions = {
            'mse': lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
            'mae': lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
            'r2': lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        }

    # Fit model
    if solver == 'closed_form':
        beta = closed_form_solver(X_norm, y, p)
    elif solver == 'gradient_descent':
        beta = gradient_descent_solver(X_norm, y, p, learning_rate, max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Predictions
    mu = np.exp(np.dot(X_norm, beta))

    # Compute metrics
    metrics = compute_metrics(y, mu, metric_functions)

    return {
        'result': {
            'coefficients': beta,
            'predictions': mu
        },
        'metrics': metrics,
        'params_used': {
            'p': p,
            'normalization': normalization,
            'solver': solver,
            'learning_rate': learning_rate,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

################################################################################
# regression_quantile
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regression_quantile_fit(
    X: np.ndarray,
    y: np.ndarray,
    quantile: float = 0.5,
    solver: str = 'coordinate_descent',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'mae',
    max_iter: int = 1000,
    tol: float = 1e-4,
    alpha: float = 0.0,
    l1_ratio: Optional[float] = None,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a quantile regression model.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    quantile : float, default=0.5
        Quantile to estimate (must be between 0 and 1).
    solver : str, default='coordinate_descent'
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    normalizer : Optional[Callable], default=None
        Normalization function to apply to X.
    metric : str, default='mae'
        Metric to compute ('mse', 'mae', 'r2', custom callable).
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    alpha : float, default=0.0
        Regularization strength (L1 for lasso, L2 for ridge).
    l1_ratio : Optional[float], default=None
        ElasticNet mixing parameter (0 <= l1_ratio <= 1).
    random_state : Optional[int], default=None
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - "result": Estimated coefficients
        - "metrics": Computed metrics
        - "params_used": Parameters used for fitting
        - "warnings": Any warnings generated

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = regression_quantile_fit(X, y, quantile=0.75)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize parameters
    n_features = X_normalized.shape[1]
    coef = np.zeros(n_features)
    intercept = 0.0

    # Choose solver
    if solver == 'closed_form':
        coef, intercept = _closed_form_solver(X_normalized, y, quantile)
    elif solver == 'gradient_descent':
        coef, intercept = _gradient_descent_solver(
            X_normalized, y, quantile, max_iter, tol,
            alpha, l1_ratio, random_state
        )
    elif solver == 'newton':
        coef, intercept = _newton_solver(
            X_normalized, y, quantile, max_iter, tol,
            alpha, l1_ratio
        )
    elif solver == 'coordinate_descent':
        coef, intercept = _coordinate_descent_solver(
            X_normalized, y, quantile, max_iter, tol,
            alpha, l1_ratio
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, coef, intercept, metric)

    # Prepare output
    result = {
        'result': {'coef': coef, 'intercept': intercept},
        'metrics': metrics,
        'params_used': {
            'quantile': quantile,
            'solver': solver,
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'max_iter': max_iter,
            'tol': tol,
            'alpha': alpha,
            'l1_ratio': l1_ratio
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

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    return normalizer(X)

def _closed_form_solver(
    X: np.ndarray,
    y: np.ndarray,
    quantile: float
) -> tuple[np.ndarray, float]:
    """Closed-form solution for quantile regression."""
    n_samples, n_features = X.shape
    # This is a simplified version - actual implementation would need to handle quantile regression properly
    X_tilde = np.column_stack([X, np.ones(n_samples)])
    y_tilde = y.copy()

    # Solve linear program (simplified for demonstration)
    coef = np.linalg.pinv(X_tilde) @ y_tilde
    intercept = coef[-1]
    coef = coef[:-1]

    return coef, intercept

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    quantile: float,
    max_iter: int,
    tol: float,
    alpha: float,
    l1_ratio: Optional[float],
    random_state: Optional[int]
) -> tuple[np.ndarray, float]:
    """Gradient descent solver for quantile regression."""
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    coef = np.random.randn(n_features) * 0.1
    intercept = 0.0

    for _ in range(max_iter):
        # Compute residuals
        residuals = y - (X @ coef + intercept)

        # Quantile loss gradient
        grad_coef = -quantile * X.T @ (residuals < 0) / n_samples
        grad_intercept = -quantile * np.sum(residuals < 0) / n_samples

        # Add regularization
        if alpha > 0:
            if l1_ratio is None or l1_ratio == 1:
                grad_coef += alpha * np.sign(coef)
            elif l1_ratio == 0:
                grad_coef += 2 * alpha * coef
            else:
                grad_coef += alpha * (l1_ratio * np.sign(coef) + (1 - l1_ratio) * 2 * coef)

        # Update parameters
        old_coef = coef.copy()
        coef -= tol * grad_coef
        intercept -= tol * grad_intercept

        # Check convergence
        if np.linalg.norm(coef - old_coef) < tol:
            break

    return coef, intercept

def _newton_solver(
    X: np.ndarray,
    y: np.ndarray,
    quantile: float,
    max_iter: int,
    tol: float,
    alpha: float,
    l1_ratio: Optional[float]
) -> tuple[np.ndarray, float]:
    """Newton's method solver for quantile regression."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    intercept = 0.0

    for _ in range(max_iter):
        # Compute residuals
        residuals = y - (X @ coef + intercept)

        # Quantile loss gradient and hessian
        grad_coef = -quantile * X.T @ (residuals < 0) / n_samples
        hessian_coef = quantile * X.T @ (residuals < 0) / n_samples

        # Add regularization
        if alpha > 0:
            if l1_ratio is None or l1_ratio == 1:
                grad_coef += alpha * np.sign(coef)
            elif l1_ratio == 0:
                grad_coef += 2 * alpha * coef
                hessian_coef += 2 * alpha * np.eye(n_features)
            else:
                grad_coef += alpha * (l1_ratio * np.sign(coef) + (1 - l1_ratio) * 2 * coef)
                hessian_coef += (1 - l1_ratio) * 2 * alpha * np.eye(n_features)

        # Update parameters
        old_coef = coef.copy()
        coef -= np.linalg.pinv(hessian_coef) @ grad_coef
        intercept -= (quantile * np.sum(residuals < 0) / n_samples)

        # Check convergence
        if np.linalg.norm(coef - old_coef) < tol:
            break

    return coef, intercept

def _coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    quantile: float,
    max_iter: int,
    tol: float,
    alpha: float,
    l1_ratio: Optional[float]
) -> tuple[np.ndarray, float]:
    """Coordinate descent solver for quantile regression."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    intercept = 0.0

    for _ in range(max_iter):
        # Update each coefficient one at a time
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - (X @ coef + intercept)

            # Compute gradient for this coefficient
            grad_j = -quantile * np.sum(X_j[(residuals < 0) & (X_j != 0)]) / n_samples

            # Add regularization
            if alpha > 0:
                if l1_ratio is None or l1_ratio == 1:
                    grad_j += alpha * np.sign(coef[j])
                elif l1_ratio == 0:
                    grad_j += 2 * alpha * coef[j]
                else:
                    grad_j += alpha * (l1_ratio * np.sign(coef[j]) + (1 - l1_ratio) * 2 * coef[j])

            # Update coefficient
            old_coef = coef[j]
            if X_j.std() > 0:
                coef[j] -= tol * grad_j / (X_j.std())
            else:
                coef[j] = 0

        # Update intercept
        residuals = y - (X @ coef + intercept)
        grad_intercept = -quantile * np.sum(residuals < 0) / n_samples
        intercept -= tol * grad_intercept

        # Check convergence
        if np.linalg.norm(coef - old_coef) < tol:
            break

    return coef, intercept

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Compute metrics for the regression model."""
    y_pred = X @ coef + intercept

    if callable(metric):
        return {'custom': metric(y, y_pred)}

    metrics = {}
    if metric == 'mse' or 'all' in metric:
        metrics['mse'] = np.mean((y - y_pred) ** 2)
    if metric == 'mae' or 'all' in metric:
        metrics['mae'] = np.mean(np.abs(y - y_pred))
    if metric == 'r2' or 'all' in metric:
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics['r2'] = 1 - ss_res / ss_tot

    return metrics

################################################################################
# regression_ridge
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input matrices for regression."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def normalize_data(X: np.ndarray, y: np.ndarray,
                  normalization: str = 'standard',
                  custom_norm: Optional[Callable] = None) -> tuple:
    """Normalize features and target."""
    if custom_norm is not None:
        X_norm = custom_norm(X)
        y_norm = custom_norm(y.reshape(-1, 1)).flatten()
    elif normalization == 'standard':
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_norm = (X - X_mean) / X_std
        y_mean = y.mean()
        y_std = y.std()
        y_norm = (y - y_mean) / y_std
    elif normalization == 'minmax':
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        y_min = y.min()
        y_max = y.max()
        y_norm = (y - y_min) / (y_max - y_min + 1e-8)
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - X_median) / (X_iqr + 1e-8)
        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        y_norm = (y - y_median) / (y_iqr + 1e-8)
    else:
        X_norm = X.copy()
        y_norm = y.copy()

    return X_norm, y_norm

def compute_ridge_closed_form(X: np.ndarray, y: np.ndarray,
                             alpha: float = 1.0) -> np.ndarray:
    """Compute Ridge coefficients using closed-form solution."""
    I = np.eye(X.shape[1])
    coefs = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return coefs

def compute_ridge_gradient_descent(X: np.ndarray, y: np.ndarray,
                                  alpha: float = 1.0,
                                  lr: float = 0.01,
                                  max_iter: int = 1000,
                                  tol: float = 1e-4) -> np.ndarray:
    """Compute Ridge coefficients using gradient descent."""
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * (X.T @ (X @ coefs - y) + alpha * coefs)
        new_coefs = coefs - lr * gradient
        if np.linalg.norm(new_coefs - coefs) < tol:
            break
        coefs = new_coefs
    return coefs

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   metric_funcs: Dict[str, Callable] = None) -> Dict[str, float]:
    """Compute regression metrics."""
    if metric_funcs is None:
        metric_funcs = {
            'mse': lambda yt, yp: np.mean((yt - yp) ** 2),
            'mae': lambda yt, yp: np.mean(np.abs(yt - yp)),
            'r2': lambda yt, yp: 1 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2)
        }

    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(y_true, y_pred)
        except Exception as e:
            metrics[f"{name}_error"] = str(e)

    return metrics

def regression_ridge_fit(X: np.ndarray, y: np.ndarray,
                        alpha: float = 1.0,
                        solver: str = 'closed_form',
                        normalization: str = 'standard',
                        custom_norm: Optional[Callable] = None,
                        metric_funcs: Optional[Dict[str, Callable]] = None) -> Dict:
    """Fit Ridge regression model."""
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization, custom_norm)

    # Choose solver
    if solver == 'closed_form':
        coefs = compute_ridge_closed_form(X_norm, y_norm, alpha)
    elif solver == 'gradient_descent':
        coefs = compute_ridge_gradient_descent(X_norm, y_norm, alpha)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Make predictions
    y_pred = X_norm @ coefs

    # Compute metrics
    metrics = compute_metrics(y_norm, y_pred, metric_funcs)

    # Prepare output
    result = {
        'result': {'coefficients': coefs},
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'solver': solver,
            'normalization': normalization
        },
        'warnings': []
    }

    return result

################################################################################
# regression_lasso
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regression_lasso_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    fit_intercept: bool = True,
    normalize: Optional[str] = None,
    solver: str = 'coordinate_descent',
    metric: Union[str, Callable] = 'mse',
    random_state: Optional[int] = None,
    **kwargs
) -> Dict:
    """
    Fit a Lasso regression model.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    alpha : float, default=1.0
        Regularization strength.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    normalize : str or None, default=None
        Normalization method: 'standard', 'minmax', 'robust', or None.
    solver : str, default='coordinate_descent'
        Solver to use: 'closed_form', 'gradient_descent', 'coordinate_descent'.
    metric : str or callable, default='mse'
        Metric to evaluate the model: 'mse', 'mae', 'r2', or custom callable.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Returns:
    --------
    Dict containing:
        - 'result': Fitted coefficients.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used for fitting.
        - 'warnings': Any warnings encountered.

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = regression_lasso_fit(X, y, alpha=0.5, solver='coordinate_descent')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _normalize_data(X, method=normalize)

    # Initialize parameters
    n_samples, n_features = X_normalized.shape
    coef_ = np.zeros(n_features)
    intercept_ = 0.0 if fit_intercept else None

    # Choose solver
    if solver == 'coordinate_descent':
        coef_, intercept_ = _coordinate_descent(
            X_normalized, y, alpha=alpha,
            max_iter=max_iter, tol=tol,
            fit_intercept=fit_intercept
        )
    elif solver == 'gradient_descent':
        coef_, intercept_ = _gradient_descent(
            X_normalized, y, alpha=alpha,
            max_iter=max_iter, tol=tol,
            fit_intercept=fit_intercept
        )
    else:
        raise ValueError("Unsupported solver. Choose from 'coordinate_descent' or 'gradient_descent'.")

    # Compute metrics
    y_pred = _predict(X_normalized, coef_, intercept_)
    metrics = _compute_metrics(y, y_pred, metric=metric)

    # Prepare output
    result = {
        'result': {'coef_': coef_, 'intercept_': intercept_},
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'max_iter': max_iter,
            'tol': tol,
            'fit_intercept': fit_intercept,
            'normalize': normalize,
            'solver': solver
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or infinite values.")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("y contains NaN or infinite values.")

def _normalize_data(X: np.ndarray, method: Optional[str] = None) -> np.ndarray:
    """Normalize data based on specified method."""
    if method is None:
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
        raise ValueError("Unsupported normalization method.")

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
    fit_intercept: bool
) -> tuple:
    """Coordinate descent solver for Lasso regression."""
    n_samples, n_features = X.shape
    coef_ = np.zeros(n_features)
    intercept_ = 0.0 if fit_intercept else None

    for _ in range(max_iter):
        old_coef = coef_.copy()

        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - np.dot(X, coef_) if not fit_intercept else y - (np.dot(X, coef_) + intercept_)

            if fit_intercept:
                residuals -= intercept_

            rho_j = np.dot(X_j, X_j)
            if rho_j == 0:
                continue

            c_j = np.dot(X_j, residuals) + coef_[j] * rho_j
            if c_j < -alpha / 2:
                coef_[j] = (c_j + alpha / 2) / rho_j
            elif c_j > alpha / 2:
                coef_[j] = (c_j - alpha / 2) / rho_j
            else:
                coef_[j] = 0

        if fit_intercept:
            intercept_ = np.mean(y - np.dot(X, coef_))

        if np.linalg.norm(coef_ - old_coef) < tol:
            break

    return coef_, intercept_

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
    fit_intercept: bool
) -> tuple:
    """Gradient descent solver for Lasso regression."""
    n_samples, n_features = X.shape
    coef_ = np.zeros(n_features)
    intercept_ = 0.0 if fit_intercept else None
    learning_rate = 0.1

    for _ in range(max_iter):
        old_coef = coef_.copy()

        # Compute gradients
        y_pred = _predict(X, coef_, intercept_)
        error = y_pred - y

        if fit_intercept:
            intercept_ -= learning_rate * np.mean(error)

        gradient = (1 / n_samples) * np.dot(X.T, error)
        coef_ -= learning_rate * (gradient + alpha * np.sign(coef_))

        if np.linalg.norm(coef_ - old_coef) < tol:
            break

    return coef_, intercept_

def _predict(X: np.ndarray, coef_: np.ndarray, intercept_: Optional[float] = None) -> np.ndarray:
    """Predict using the fitted model."""
    y_pred = np.dot(X, coef_)
    if intercept_ is not None:
        y_pred += intercept_
    return y_pred

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable] = 'mse'
) -> Dict:
    """Compute metrics based on specified method."""
    if callable(metric):
        return {'custom': metric(y_true, y_pred)}

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
) -> Dict[str, Union[Dict, np.ndarray, str]]:
    """
    Fit an Elastic Net regression model.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
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
    metric : str or callable, default='mse'
        Metric to evaluate: 'mse', 'mae', 'r2', or custom callable.
    random_state : int, optional
        Random seed for reproducibility.
    custom_metric : callable, optional
        Custom metric function.

    Returns
    -------
    Dict[str, Union[Dict, np.ndarray, str]]
        Dictionary containing:
        - 'result': Fitted coefficients.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used for fitting.
        - 'warnings': Any warnings encountered.

    Examples
    --------
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
    coef = np.zeros(n_features)

    # Choose solver
    if solver == 'closed_form':
        coef = _solve_closed_form(X_normalized, y_normalized, alpha, l1_ratio)
    elif solver == 'gradient_descent':
        coef = _solve_gradient_descent(X_normalized, y_normalized, alpha, l1_ratio,
                                      max_iter, tol, random_state)
    elif solver == 'newton':
        coef = _solve_newton(X_normalized, y_normalized, alpha, l1_ratio,
                            max_iter, tol)
    elif solver == 'coordinate_descent':
        coef = _solve_coordinate_descent(X_normalized, y_normalized, alpha, l1_ratio,
                                        max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    y_pred = X_normalized @ coef
    metrics = _compute_metrics(y_normalized, y_pred, metric, custom_metric)

    # Prepare output
    result = {
        'result': coef,
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'l1_ratio': l1_ratio,
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

    # Normalize y if required (for some metrics)
    if method != 'none':
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_normalized = (y - y_mean) / (y_std + 1e-8)
    else:
        y_normalized = y

    return X_normalized, y_normalized

def _solve_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float
) -> np.ndarray:
    """Solve Elastic Net using closed-form solution (for small problems)."""
    n_samples, n_features = X.shape
    if alpha == 0:
        coef = np.linalg.pinv(X) @ y
    else:
        # This is a simplified version; actual implementation would need to handle the L1/L2 mix
        coef = np.linalg.pinv(X.T @ X + alpha * np.eye(n_features)) @ X.T @ y
    return coef

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    random_state: Optional[int]
) -> np.ndarray:
    """Solve Elastic Net using gradient descent."""
    if random_state is not None:
        np.random.seed(random_state)
    n_features = X.shape[1]
    coef = np.random.randn(n_features) * 0.01

    for _ in range(max_iter):
        gradient = X.T @ (X @ coef - y) + alpha * (
            l1_ratio * np.sign(coef) +
            (1 - l1_ratio) * 2 * coef
        )
        coef -= gradient * 0.01

    return coef

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Solve Elastic Net using Newton's method."""
    n_features = X.shape[1]
    coef = np.zeros(n_features)

    for _ in range(max_iter):
        residual = X @ coef - y
        gradient = X.T @ residual + alpha * (
            l1_ratio * np.sign(coef) +
            (1 - l1_ratio) * 2 * coef
        )
        hessian = X.T @ X + alpha * (1 - l1_ratio) * 2 * np.eye(n_features)
        coef -= np.linalg.pinv(hessian) @ gradient

    return coef

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Solve Elastic Net using coordinate descent."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residual = y - (X @ coef) + coef[j] * X_j

            # Soft-thresholding for Elastic Net
            rho = l1_ratio * alpha
            sigma = (1 - l1_ratio) * alpha

            numerator = X_j.T @ residual
            denominator = X_j.T @ X_j + sigma

            coef[j] = _soft_threshold(numerator / denominator, rho)

    return coef

def _soft_threshold(value: float, threshold: float) -> float:
    """Soft-thresholding function."""
    if value > threshold:
        return value - threshold
    elif value < -threshold:
        return value + threshold
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
        metrics['r2'] = 1 - (ss_res / ss_tot)
    elif custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

################################################################################
# regression_polynomiale
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regression_polynomiale_fit(
    X: np.ndarray,
    y: np.ndarray,
    degree: int = 2,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    solver: str = 'closed_form',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Fit a polynomial regression model to the given data.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    degree : int, optional
        Degree of the polynomial features, by default 2.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the features, by default None.
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent'), by default 'closed_form'.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to evaluate the model ('mse', 'mae', 'r2'), by default 'mse'.
    regularization : Optional[str], optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet'), by default None.
    alpha : float, optional
        Regularization strength, by default 1.0.
    l1_ratio : float, optional
        ElasticNet mixing parameter, by default 0.5.
    tol : float, optional
        Tolerance for stopping criteria, by default 1e-4.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    random_state : Optional[int], optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]
        Dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.array([[1], [2], [3]])
    >>> y = np.array([1, 4, 9])
    >>> result = regression_polynomiale_fit(X, y, degree=2)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if a normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Create polynomial features
    X_poly = _create_polynomial_features(X_normalized, degree)

    # Choose solver
    if solver == 'closed_form':
        coefficients = _solve_closed_form(X_poly, y, regularization, alpha, l1_ratio)
    elif solver == 'gradient_descent':
        coefficients = _solve_gradient_descent(X_poly, y, tol, max_iter, random_state,
                                             regularization, alpha, l1_ratio)
    else:
        raise ValueError("Invalid solver specified. Choose 'closed_form' or 'gradient_descent'.")

    # Calculate metrics
    y_pred = _predict(X_poly, coefficients)
    metrics = _calculate_metrics(y, y_pred, metric)

    # Prepare output
    result = {
        'result': coefficients,
        'metrics': metrics,
        'params_used': {
            'degree': degree,
            'normalizer': normalizer.__name__ if normalizer else None,
            'solver': solver,
            'metric': metric if isinstance(metric, str) else 'custom',
            'regularization': regularization,
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate the input data."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values.")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Apply normalization to the features."""
    if normalizer is not None:
        X_normalized = normalizer(X)
    else:
        X_normalized = X
    return X_normalized

def _create_polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Create polynomial features."""
    n_samples = X.shape[0]
    X_poly = np.ones((n_samples, 1))
    for d in range(1, degree + 1):
        X_poly = np.hstack((X_poly, X ** d))
    return X_poly

def _solve_closed_form(X: np.ndarray, y: np.ndarray,
                      regularization: Optional[str], alpha: float, l1_ratio: float) -> np.ndarray:
    """Solve the polynomial regression using closed-form solution."""
    if regularization is None or regularization == 'none':
        coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
    elif regularization == 'l2':
        I = np.eye(X.shape[1])
        coefficients = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    elif regularization == 'l1':
        # Simplified L1 solution (in practice, use coordinate descent or similar)
        coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
    elif regularization == 'elasticnet':
        # Simplified ElasticNet solution (in practice, use coordinate descent or similar)
        I = np.eye(X.shape[1])
        coefficients = np.linalg.inv(X.T @ X + alpha * (l1_ratio * np.eye(X.shape[1]) +
                                                      (1 - l1_ratio) * I)) @ X.T @ y
    else:
        raise ValueError("Invalid regularization type specified.")
    return coefficients

def _solve_gradient_descent(X: np.ndarray, y: np.ndarray,
                           tol: float, max_iter: int, random_state: Optional[int],
                           regularization: Optional[str], alpha: float, l1_ratio: float) -> np.ndarray:
    """Solve the polynomial regression using gradient descent."""
    if random_state is not None:
        np.random.seed(random_state)
    n_features = X.shape[1]
    coefficients = np.random.randn(n_features)

    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, coefficients, regularization, alpha, l1_ratio)
        coefficients -= tol * gradient
    return coefficients

def _compute_gradient(X: np.ndarray, y: np.ndarray,
                     coefficients: np.ndarray,
                     regularization: Optional[str], alpha: float, l1_ratio: float) -> np.ndarray:
    """Compute the gradient for gradient descent."""
    n_samples = X.shape[0]
    residuals = y - X @ coefficients
    gradient = -(2 / n_samples) * (X.T @ residuals)

    if regularization == 'l2':
        gradient += 2 * alpha * coefficients
    elif regularization == 'l1':
        gradient += alpha * np.sign(coefficients)
    elif regularization == 'elasticnet':
        gradient += alpha * (l1_ratio * np.sign(coefficients) + (1 - l1_ratio) * 2 * coefficients)

    return gradient

def _predict(X: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Predict using the polynomial regression model."""
    return X @ coefficients

def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]) -> Dict[str, float]:
    """Calculate the specified metrics."""
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
        else:
            raise ValueError("Invalid metric specified.")
    else:
        metrics['custom'] = metric(y_true, y_pred)

    return metrics

################################################################################
# spline_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def spline_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    knots: Optional[np.ndarray] = None,
    degree: int = 3,
    solver: str = 'closed_form',
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    penalty: Optional[str] = None,
    alpha: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_solver: Optional[Callable] = None
) -> Dict:
    """
    Fit a spline regression model to the given data.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    knots : Optional[np.ndarray], default=None
        Locations of the knots. If None, uses quantiles of X.
    degree : int, default=3
        Degree of the spline basis functions.
    solver : str or Callable, default='closed_form'
        Solver to use. Options: 'closed_form', 'gradient_descent', 'newton'.
    normalization : str, default='standard'
        Normalization method. Options: 'none', 'standard', 'minmax', 'robust'.
    metric : str or Callable, default='mse'
        Metric to evaluate the model. Options: 'mse', 'mae', 'r2'.
    penalty : Optional[str], default=None
        Penalty type. Options: 'l1', 'l2', 'elasticnet'.
    alpha : float, default=1.0
        Regularization strength.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    max_iter : int, default=1000
        Maximum number of iterations for iterative solvers.
    custom_metric : Optional[Callable], default=None
        Custom metric function if not using built-in metrics.
    custom_solver : Optional[Callable], default=None
        Custom solver function if not using built-in solvers.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Fitted model parameters.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used for fitting.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> X = np.random.rand(100, 2)
    >>> y = np.random.rand(100)
    >>> result = spline_regression_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization)

    # Set up spline basis
    if knots is None:
        knots = np.quantile(X, np.linspace(0, 1, degree + 2))

    basis = _create_spline_basis(X_norm, knots, degree)

    # Choose solver
    if custom_solver is not None:
        solver_func = custom_solver
    else:
        solver_func = _get_solver(solver)

    # Fit model
    params = solver_func(basis, y_norm, penalty=penalty, alpha=alpha,
                         tol=tol, max_iter=max_iter)

    # Compute metrics
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric(metric)

    metrics = metric_func(y_norm, basis @ params)

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'knots': knots,
            'degree': degree,
            'solver': solver,
            'normalization': normalization,
            'metric': metric,
            'penalty': penalty,
            'alpha': alpha
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
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
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize data according to specified method."""
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

def _create_spline_basis(
    X: np.ndarray,
    knots: np.ndarray,
    degree: int
) -> np.ndarray:
    """Create spline basis matrix."""
    n_samples = X.shape[0]
    n_basis = len(knots) + degree
    basis = np.zeros((n_samples, n_basis))

    for i in range(n_samples):
        basis[i] = _evaluate_spline_basis(X[i], knots, degree)

    return basis

def _evaluate_spline_basis(
    x: np.ndarray,
    knots: np.ndarray,
    degree: int
) -> np.ndarray:
    """Evaluate spline basis functions at given points."""
    n_basis = len(knots) + degree
    basis = np.zeros(n_basis)

    # Constant term
    basis[0] = 1.0

    # Linear and higher degree terms
    for d in range(1, degree + 1):
        basis[d] = x ** d

    # Spline terms
    for k in range(degree + 1, n_basis):
        basis[k] = _evaluate_bspline(x, knots, k - degree)

    return basis

def _evaluate_bspline(
    x: float,
    knots: np.ndarray,
    degree: int
) -> float:
    """Evaluate a single B-spline basis function."""
    if degree == 0:
        return 1.0 if knots[0] <= x < knots[1] else 0.0

    coeff1 = (x - knots[degree]) / (knots[degree + degree] - knots[degree])
    coeff2 = (knots[degree + degree + 1] - x) / (knots[degree + degree + 1] - knots[degree + 1])

    return coeff1 * _evaluate_bspline(x, knots[:-1], degree - 1) + coeff2 * _evaluate_bspline(x, knots[1:], degree - 1)

def _get_solver(solver: str) -> Callable:
    """Get solver function based on name."""
    solvers = {
        'closed_form': _closed_form_solver,
        'gradient_descent': _gradient_descent_solver,
        'newton': _newton_solver
    }
    if solver not in solvers:
        raise ValueError(f"Unknown solver: {solver}")
    return solvers[solver]

def _closed_form_solver(
    basis: np.ndarray,
    y: np.ndarray,
    penalty: Optional[str] = None,
    alpha: float = 1.0,
    **kwargs
) -> np.ndarray:
    """Closed form solution for spline regression."""
    if penalty is None:
        params = np.linalg.pinv(basis) @ y
    elif penalty == 'l2':
        identity = np.eye(basis.shape[1])
        params = np.linalg.inv(basis.T @ basis + alpha * identity) @ basis.T @ y
    else:
        raise ValueError(f"Penalty {penalty} not supported for closed form solver")
    return params

def _gradient_descent_solver(
    basis: np.ndarray,
    y: np.ndarray,
    penalty: Optional[str] = None,
    alpha: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Gradient descent solver for spline regression."""
    n_params = basis.shape[1]
    params = np.zeros(n_params)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = -2 * basis.T @ (y - basis @ params)

        if penalty == 'l1':
            gradient += alpha * np.sign(params)
        elif penalty == 'l2':
            gradient += 2 * alpha * params

        new_params = params - learning_rate * gradient
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params

    return params

def _newton_solver(
    basis: np.ndarray,
    y: np.ndarray,
    penalty: Optional[str] = None,
    alpha: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Newton's method solver for spline regression."""
    n_params = basis.shape[1]
    params = np.zeros(n_params)

    for _ in range(max_iter):
        residuals = y - basis @ params
        gradient = -2 * basis.T @ residuals

        hessian = 2 * basis.T @ basis
        if penalty == 'l2':
            hessian += 2 * alpha * np.eye(n_params)

        delta = np.linalg.solve(hessian, gradient)
        new_params = params - delta

        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params

    return params

def _get_metric(metric: str) -> Callable:
    """Get metric function based on name."""
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

################################################################################
# generalized_additive_models
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def generalized_additive_models_fit(
    X: np.ndarray,
    y: np.ndarray,
    basis_functions: Dict[str, Callable],
    link_function: Callable,
    family: str = 'gaussian',
    solver: str = 'gradient_descent',
    normalization: str = 'standard',
    metric: str = 'mse',
    max_iter: int = 100,
    tol: float = 1e-4,
    alpha: float = 0.0,
    l1_ratio: float = 0.5,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Fit a Generalized Additive Model (GAM).

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    basis_functions : Dict[str, Callable]
        Dictionary mapping feature names to their respective basis functions.
    link_function : Callable
        The link function to use for the model.
    family : str, optional
        Distribution family ('gaussian', 'binomial', etc.), default='gaussian'.
    solver : str, optional
        Solver to use ('gradient_descent', 'newton', etc.), default='gradient_descent'.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), default='standard'.
    metric : str, optional
        Metric to evaluate the model ('mse', 'mae', 'r2', etc.), default='mse'.
    max_iter : int, optional
        Maximum number of iterations, default=100.
    tol : float, optional
        Tolerance for stopping criteria, default=1e-4.
    alpha : float, optional
        Regularization strength, default=0.0.
    l1_ratio : float, optional
        ElasticNet mixing parameter, default=0.5.
    random_state : Optional[int], optional
        Random seed for reproducibility, default=None.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _normalize_data(X, normalization)

    # Initialize parameters
    params = _initialize_parameters(basis_functions, X_normalized.shape[1])

    # Choose solver
    if solver == 'gradient_descent':
        params = _gradient_descent(
            X_normalized, y, basis_functions, link_function,
            family, params, max_iter, tol, alpha, l1_ratio
        )
    elif solver == 'newton':
        params = _newton_method(
            X_normalized, y, basis_functions, link_function,
            family, params, max_iter, tol
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(y, _predict(X_normalized, params, basis_functions, link_function), metric)

    return {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "solver": solver,
            "normalization": normalization,
            "metric": metric,
            "max_iter": max_iter,
            "tol": tol,
            "alpha": alpha,
            "l1_ratio": l1_ratio
        },
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize the input data."""
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

def _initialize_parameters(basis_functions: Dict[str, Callable], n_features: int) -> np.ndarray:
    """Initialize model parameters."""
    return np.zeros(n_features)

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    basis_functions: Dict[str, Callable],
    link_function: Callable,
    family: str,
    params: np.ndarray,
    max_iter: int,
    tol: float,
    alpha: float,
    l1_ratio: float
) -> np.ndarray:
    """Gradient descent solver for GAM."""
    for _ in range(max_iter):
        # Update parameters
        gradient = _compute_gradient(X, y, basis_functions, link_function, family, params)
        if alpha > 0:
            gradient += _compute_regularization(gradient, params, l1_ratio)
        params -= tol * gradient
    return params

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    basis_functions: Dict[str, Callable],
    link_function: Callable,
    family: str,
    params: np.ndarray,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Newton method solver for GAM."""
    for _ in range(max_iter):
        # Update parameters using Newton's method
        gradient = _compute_gradient(X, y, basis_functions, link_function, family, params)
        hessian = _compute_hessian(X, y, basis_functions, link_function, family, params)
        params -= np.linalg.solve(hessian + 1e-8 * np.eye(len(params)), gradient)
    return params

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    basis_functions: Dict[str, Callable],
    link_function: Callable,
    family: str,
    params: np.ndarray
) -> np.ndarray:
    """Compute the gradient of the loss function."""
    # Placeholder for actual gradient computation
    return np.zeros_like(params)

def _compute_hessian(
    X: np.ndarray,
    y: np.ndarray,
    basis_functions: Dict[str, Callable],
    link_function: Callable,
    family: str,
    params: np.ndarray
) -> np.ndarray:
    """Compute the Hessian matrix."""
    # Placeholder for actual Hessian computation
    return np.eye(X.shape[1])

def _compute_regularization(
    gradient: np.ndarray,
    params: np.ndarray,
    l1_ratio: float
) -> np.ndarray:
    """Compute the regularization term."""
    if l1_ratio == 0:
        return alpha * params
    elif l1_ratio == 1:
        return alpha * np.sign(params)
    else:
        return alpha * (l1_ratio * np.sign(params) + (1 - l1_ratio) * params)

def _predict(
    X: np.ndarray,
    params: np.ndarray,
    basis_functions: Dict[str, Callable],
    link_function: Callable
) -> np.ndarray:
    """Make predictions using the fitted model."""
    # Placeholder for actual prediction computation
    return np.zeros(X.shape[0])

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    if metric == 'mse':
        return {'mse': np.mean((y_true - y_pred) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(y_true - y_pred))}
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return {'r2': 1 - ss_res / (ss_tot + 1e-8)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# regression_bayesienne
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regression_bayesienne_fit(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: Optional[np.ndarray] = None,
    prior_cov: Optional[np.ndarray] = None,
    likelihood_var: float = 1.0,
    solver: str = 'closed_form',
    max_iter: int = 1000,
    tol: float = 1e-6,
    metric: str = 'mse',
    normalize: bool = True,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a Bayesian regression model.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    prior_mean : Optional[np.ndarray]
        Prior mean for the coefficients
    prior_cov : Optional[np.ndarray]
        Prior covariance matrix for the coefficients
    likelihood_var : float
        Variance of the likelihood (noise variance)
    solver : str
        Solver to use ('closed_form', 'gradient_descent')
    max_iter : int
        Maximum number of iterations for iterative solvers
    tol : float
        Tolerance for convergence
    metric : str
        Metric to compute ('mse', 'mae', 'r2')
    normalize : bool
        Whether to normalize the features
    random_state : Optional[int]
        Random seed for reproducibility

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Estimated coefficients
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in the fit
        - 'warnings': Any warnings encountered

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = regression_bayesienne_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize features if requested
    X_norm, y_norm = _normalize_data(X, y) if normalize else (X, y)

    # Set default prior
    n_features = X_norm.shape[1]
    if prior_mean is None:
        prior_mean = np.zeros(n_features)
    if prior_cov is None:
        prior_cov = np.eye(n_features)

    # Choose solver
    if solver == 'closed_form':
        coefficients = _bayesian_closed_form(X_norm, y_norm, prior_mean, prior_cov, likelihood_var)
    elif solver == 'gradient_descent':
        coefficients = _bayesian_gradient_descent(
            X_norm, y_norm, prior_mean, prior_cov, likelihood_var,
            max_iter=max_iter, tol=tol
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(y, X @ coefficients, metric)

    return {
        'result': coefficients,
        'metrics': metrics,
        'params_used': {
            'prior_mean': prior_mean,
            'prior_cov': prior_cov,
            'likelihood_var': likelihood_var,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol,
            'metric': metric,
            'normalize': normalize
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """Normalize features and target."""
    X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y_norm = (y - np.mean(y)) / np.std(y)
    return X_norm, y_norm

def _bayesian_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    likelihood_var: float
) -> np.ndarray:
    """Closed form solution for Bayesian regression."""
    n_features = X.shape[1]
    posterior_cov = np.linalg.inv(np.linalg.inv(prior_cov) + (X.T @ X) / likelihood_var)
    posterior_mean = posterior_cov @ (
        np.linalg.inv(prior_cov) @ prior_mean + (X.T @ y) / likelihood_var
    )
    return posterior_mean

def _bayesian_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    likelihood_var: float,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> np.ndarray:
    """Gradient descent solver for Bayesian regression."""
    n_samples, n_features = X.shape
    coefficients = prior_mean.copy()

    for _ in range(max_iter):
        gradient = (X.T @ (X @ coefficients - y)) / likelihood_var
        hessian = (X.T @ X) / likelihood_var + np.linalg.inv(prior_cov)
        coefficients_new = coefficients - np.linalg.solve(hessian, gradient)

        if np.linalg.norm(coefficients_new - coefficients) < tol:
            break
        coefficients = coefficients_new

    return coefficients

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str
) -> Dict[str, float]:
    """Compute regression metrics."""
    metrics = {}

    if metric in ['mse', 'mae', 'r2'] or isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

################################################################################
# regularisation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regularisation_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalisation: str = "standard",
    metric: Union[str, Callable] = "mse",
    solver: str = "closed_form",
    regularisation_type: str = "none",
    alpha: float = 1.0,
    l1_ratio: Optional[float] = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit a regularized regression model.

    Parameters:
    -----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalisation : str, optional
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : str or callable, optional
        Metric to evaluate: "mse", "mae", "r2", or custom callable.
    solver : str, optional
        Solver method: "closed_form", "gradient_descent", or "coordinate_descent".
    regularisation_type : str, optional
        Regularization type: "none", "l1", "l2", or "elasticnet".
    alpha : float, optional
        Regularization strength.
    l1_ratio : float, optional
        ElasticNet mixing parameter (0 <= l1_ratio <= 1).
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for stopping criteria.
    random_state : int, optional
        Random seed for reproducibility.
    custom_metric : callable, optional
        Custom metric function.

    Returns:
    --------
    Dict containing:
        - "result": fitted coefficients
        - "metrics": computed metrics
        - "params_used": parameters used in fitting
        - "warnings": potential warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = regularisation_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _apply_normalisation(X, y, normalisation)

    # Initialize parameters
    params_used = {
        "normalisation": normalisation,
        "metric": metric,
        "solver": solver,
        "regularisation_type": regularisation_type,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "max_iter": max_iter,
        "tol": tol
    }

    # Choose solver
    if solver == "closed_form":
        coefficients = _solve_closed_form(X_norm, y_norm, regularisation_type, alpha, l1_ratio)
    elif solver == "gradient_descent":
        coefficients = _solve_gradient_descent(X_norm, y_norm, regularisation_type,
                                             alpha, l1_ratio, max_iter, tol, random_state)
    elif solver == "coordinate_descent":
        coefficients = _solve_coordinate_descent(X_norm, y_norm, regularisation_type,
                                               alpha, l1_ratio, max_iter, tol, random_state)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_norm, y_norm, coefficients, metric, custom_metric)

    # Prepare output
    result = {
        "result": coefficients,
        "metrics": metrics,
        "params_used": params_used,
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

def _apply_normalisation(X: np.ndarray, y: np.ndarray, method: str) -> tuple:
    """Apply data normalisation."""
    if method == "standard":
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y = (y - np.mean(y)) / np.std(y)
    elif method == "minmax":
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == "robust":
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    return X, y

def _solve_closed_form(X: np.ndarray, y: np.ndarray,
                      reg_type: str, alpha: float, l1_ratio: Optional[float]) -> np.ndarray:
    """Closed form solution for regularized regression."""
    n_features = X.shape[1]
    I = np.eye(n_features)

    if reg_type == "l2":
        coefficients = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    elif reg_type == "l1":
        # This is a simplified version - in practice you'd use coordinate descent
        coefficients = np.linalg.pinv(X) @ y
    elif reg_type == "elasticnet" and l1_ratio is not None:
        # This is a simplified version - in practice you'd use coordinate descent
        coefficients = np.linalg.pinv(X) @ y
    else:
        coefficients = np.linalg.pinv(X) @ y

    return coefficients

def _solve_gradient_descent(X: np.ndarray, y: np.ndarray,
                           reg_type: str, alpha: float, l1_ratio: Optional[float],
                           max_iter: int, tol: float, random_state: Optional[int]) -> np.ndarray:
    """Gradient descent solver for regularized regression."""
    n_samples, n_features = X.shape
    np.random.seed(random_state)
    coefficients = np.random.randn(n_features)

    for _ in range(max_iter):
        gradient = X.T @ (X @ coefficients - y) / n_samples

        if reg_type == "l2":
            gradient += alpha * coefficients
        elif reg_type == "l1":
            gradient += alpha * np.sign(coefficients)
        elif reg_type == "elasticnet" and l1_ratio is not None:
            gradient += alpha * (l1_ratio * np.sign(coefficients) + (1 - l1_ratio) * coefficients)

        # Update coefficients
        new_coefficients = coefficients - tol * gradient

        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break

        coefficients = new_coefficients

    return coefficients

def _solve_coordinate_descent(X: np.ndarray, y: np.ndarray,
                             reg_type: str, alpha: float, l1_ratio: Optional[float],
                             max_iter: int, tol: float, random_state: Optional[int]) -> np.ndarray:
    """Coordinate descent solver for regularized regression."""
    n_samples, n_features = X.shape
    np.random.seed(random_state)
    coefficients = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - X @ coefficients + coefficients[j] * X_j

            if reg_type == "l2":
                numerator = X_j.T @ residuals
                denominator = X_j.T @ X_j + alpha
            elif reg_type == "l1":
                numerator = X_j.T @ residuals
                denominator = 1.0

                if numerator > alpha / 2:
                    coefficients[j] = (numerator - alpha / 2) / denominator
                elif numerator < -alpha / 2:
                    coefficients[j] = (numerator + alpha / 2) / denominator
                else:
                    coefficients[j] = 0.0
            elif reg_type == "elasticnet" and l1_ratio is not None:
                numerator = X_j.T @ residuals
                denominator = X_j.T @ X_j + alpha * (1 - l1_ratio)

                if numerator > alpha * l1_ratio / 2:
                    coefficients[j] = (numerator - alpha * l1_ratio / 2) / denominator
                elif numerator < -alpha * l1_ratio / 2:
                    coefficients[j] = (numerator + alpha * l1_ratio / 2) / denominator
                else:
                    coefficients[j] = 0.0

        if np.linalg.norm(coefficients) < tol:
            break

    return coefficients

def _compute_metrics(X: np.ndarray, y: np.ndarray,
                    coefficients: np.ndarray,
                    metric: Union[str, Callable],
                    custom_metric: Optional[Callable]) -> Dict:
    """Compute regression metrics."""
    y_pred = X @ coefficients
    metrics = {}

    if metric == "mse" or custom_metric is None:
        metrics["mse"] = np.mean((y - y_pred) ** 2)
    if metric == "mae" or custom_metric is None:
        metrics["mae"] = np.mean(np.abs(y - y_pred))
    if metric == "r2" or custom_metric is None:
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics["r2"] = 1 - ss_res / ss_tot

    if custom_metric is not None:
        metrics["custom"] = custom_metric(y, y_pred)

    return metrics

def _check_warnings(X: np.ndarray, y: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []

    if X.shape[1] > X.shape[0]:
        warnings.append("Warning: Number of features exceeds number of samples")

    if np.any(np.std(X, axis=0) == 0):
        warnings.append("Warning: Zero-variance features detected")

    return warnings

################################################################################
# validation_croisee
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validation_croisee_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform cross-validation for generalized regression.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - n_splits: Number of cross-validation splits
    - normalizer: Callable for normalization or None
    - metric: Metric name ('mse', 'mae', 'r2') or custom callable
    - solver: Solver type ('closed_form', 'gradient_descent', etc.)
    - regularization: Regularization type (None, 'l1', 'l2', 'elasticnet')
    - tol: Tolerance for convergence
    - max_iter: Maximum iterations
    - random_state: Random seed

    Returns:
    Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize results dictionary
    results = {
        'result': {},
        'metrics': {},
        'params_used': locals(),
        'warnings': []
    }

    # Normalize data if specified
    X_norm, y_norm = _apply_normalization(X, y, normalizer)

    # Perform cross-validation
    fold_sizes = np.full(n_splits, len(X) // n_splits, dtype=int)
    fold_sizes[:len(X) % n_splits] += 1

    current = 0
    for fold in range(n_splits):
        # Split data into train and test sets
        val_indices = np.arange(current, current + fold_sizes[fold])
        train_indices = np.setdiff1d(np.arange(len(X)), val_indices)

        X_train, y_train = X_norm[train_indices], y_norm[train_indices]
        X_val, y_val = X_norm[val_indices], y_norm[val_indices]

        # Fit model
        model = _fit_model(
            X_train, y_train,
            solver=solver,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter
        )

        # Evaluate on validation set
        y_pred = _predict(model, X_val)
        metrics = _compute_metrics(y_val, y_pred, metric)

        # Store results
        results['result'][f'fold_{fold}'] = {
            'model': model,
            'predictions': y_pred
        }
        results['metrics'][f'fold_{fold}'] = metrics

        current += fold_sizes[fold]

    return results

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
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> tuple:
    """Apply normalization to data."""
    if normalizer is None:
        return X, y

    X_norm = normalizer(X)
    if hasattr(normalizer, 'fit_transform') and callable(getattr(normalizer, 'fit_transform')):
        X_norm = normalizer.fit_transform(X)

    return X_norm, y

def _fit_model(
    X: np.ndarray,
    y: np.ndarray,
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """Fit regression model."""
    if solver == 'closed_form':
        return _fit_closed_form(X, y, regularization)
    elif solver == 'gradient_descent':
        return _fit_gradient_descent(X, y, regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _fit_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str]
) -> Dict[str, Any]:
    """Fit model using closed form solution."""
    if regularization is None:
        # Simple linear regression
        XTX = np.dot(X.T, X)
        if not np.allclose(XTX, XTX.T):
            raise ValueError("X^T X is not symmetric positive definite")
        coeffs = np.linalg.solve(XTX, np.dot(X.T, y))
    else:
        # Regularized regression
        coeffs = _fit_regularized(X, y, regularization)

    return {'coeffs': coeffs}

def _fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """Fit model using gradient descent."""
    n_features = X.shape[1]
    coeffs = np.zeros(n_features)

    for _ in range(max_iter):
        # Compute gradient
        grad = np.dot(X.T, (np.dot(X, coeffs) - y)) / len(y)

        if regularization == 'l2':
            grad += 2 * coeffs
        elif regularization == 'l1':
            grad += np.sign(coeffs)

        # Update coefficients
        coeffs -= tol * grad

    return {'coeffs': coeffs}

def _fit_regularized(
    X: np.ndarray,
    y: np.ndarray,
    regularization: str
) -> np.ndarray:
    """Fit regularized regression model."""
    if regularization == 'l1':
        return _fit_lasso(X, y)
    elif regularization == 'l2':
        return _fit_ridge(X, y)
    else:
        raise ValueError(f"Unknown regularization: {regularization}")

def _fit_lasso(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit L1 regularized regression."""
    # Simplified implementation
    return _fit_regularized_general(X, y, 'l1')

def _fit_ridge(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit L2 regularized regression."""
    # Simplified implementation
    return _fit_regularized_general(X, y, 'l2')

def _fit_regularized_general(
    X: np.ndarray,
    y: np.ndarray,
    reg_type: str
) -> np.ndarray:
    """General regularized regression."""
    # This is a placeholder for actual implementation
    return np.linalg.solve(np.dot(X.T, X) + 1e-4 * np.eye(X.shape[1]), np.dot(X.T, y))

def _predict(model: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Make predictions using fitted model."""
    return np.dot(X, model['coeffs'])

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    if callable(metric):
        return {'custom': metric(y_true, y_pred)}

    metrics = {}
    if metric == 'mse' or isinstance(metric, str):
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    if metric == 'mae' or isinstance(metric, str):
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric == 'r2' or isinstance(metric, str):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

################################################################################
# metriques_evaluation
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def metriques_evaluation_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Union[str, list, Callable] = 'mse',
    normalizations: Optional[Union[str, list]] = None,
    custom_metrics: Optional[Dict[str, Callable]] = None
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for generalized regression models.

    Parameters:
    -----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    metrics : Union[str, list, Callable], optional
        Metrics to compute. Can be a string ('mse', 'mae', 'r2'), list of strings,
        or custom callable function. Default is 'mse'.
    normalizations : Optional[Union[str, list]], optional
        Normalization methods to apply. Can be 'none', 'standard', 'minmax',
        or 'robust'. Default is None.
    custom_metrics : Optional[Dict[str, Callable]], optional
        Dictionary of custom metrics where keys are names and values are callables.
        Default is None.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples:
    ---------
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 1.9, 3.2])
    >>> result = metriques_evaluation_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Apply normalizations if specified
    normalized_y_true, normalized_y_pred = _apply_normalization(y_true, y_pred, normalizations)

    # Prepare metrics
    metric_functions = _prepare_metrics(metrics, custom_metrics)

    # Compute metrics
    computed_metrics = _compute_metrics(normalized_y_true, normalized_y_pred, metric_functions)

    return {
        "result": "success",
        "metrics": computed_metrics,
        "params_used": {
            "normalizations": normalizations,
            "metrics": metrics
        },
        "warnings": _check_warnings(y_true, y_pred)
    }

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("Inputs contain NaN values.")
    if np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError("Inputs contain infinite values.")

def _apply_normalization(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalizations: Optional[Union[str, list]] = None
) -> tuple:
    """Apply specified normalizations to the input arrays."""
    if normalizations is None or normalizations == 'none':
        return y_true, y_pred

    if isinstance(normalizations, str):
        normalizations = [normalizations]

    for norm in normalizations:
        if norm == 'standard':
            mean = np.mean(y_true)
            std = np.std(y_true)
            y_true = (y_true - mean) / std
            y_pred = (y_pred - mean) / std
        elif norm == 'minmax':
            min_val = np.min(y_true)
            max_val = np.max(y_true)
            y_true = (y_true - min_val) / (max_val - min_val)
            y_pred = (y_pred - min_val) / (max_val - min_val)
        elif norm == 'robust':
            median = np.median(y_true)
            iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
            y_true = (y_true - median) / iqr
            y_pred = (y_pred - median) / iqr
        else:
            raise ValueError(f"Unknown normalization method: {norm}")

    return y_true, y_pred

def _prepare_metrics(
    metrics: Union[str, list, Callable],
    custom_metrics: Optional[Dict[str, Callable]] = None
) -> Dict[str, Callable]:
    """Prepare metric functions based on user input."""
    built_in_metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }

    if custom_metrics is None:
        custom_metrics = {}

    metric_functions = {}

    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if callable(metric):
            metric_functions[metric.__name__] = metric
        elif metric in built_in_metrics:
            metric_functions[metric] = built_in_metrics[metric]
        elif metric in custom_metrics:
            metric_functions[metric] = custom_metrics[metric]
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return metric_functions

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_functions: Dict[str, Callable]
) -> Dict[str, float]:
    """Compute specified metrics."""
    computed_metrics = {}
    for name, func in metric_functions.items():
        try:
            computed_metrics[name] = func(y_true, y_pred)
        except Exception as e:
            raise RuntimeError(f"Error computing metric {name}: {str(e)}")
    return computed_metrics

def _check_warnings(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if len(y_true) < 2:
        warnings.append("Warning: Small sample size may affect metric reliability.")
    if np.std(y_true) == 0:
        warnings.append("Warning: Zero variance in target variable.")
    return warnings

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
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')

################################################################################
# scaling_standardisation
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

def standard_scale(X: np.ndarray) -> np.ndarray:
    """Standardize features by removing the mean and scaling to unit variance."""
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def minmax_scale(X: np.ndarray) -> np.ndarray:
    """Scale features to a given range, usually [0, 1]."""
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

def robust_scale(X: np.ndarray) -> np.ndarray:
    """Scale features using statistics that are robust to outliers."""
    median = np.median(X, axis=0)
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
    return (X - median) / iqr

def scaling_standardisation_fit(
    X: np.ndarray,
    method: str = 'standard',
    custom_scaler: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Fit scaling/standardisation to data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    method : str, optional
        Scaling method ('none', 'standard', 'minmax', 'robust'), default='standard'
    custom_scaler : callable, optional
        Custom scaling function that takes a numpy array and returns scaled data

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Scaled data
        - 'metrics': Dictionary of metrics (empty for scaling)
        - 'params_used': Parameters used
        - 'warnings': List of warnings

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = scaling_standardisation_fit(X, method='standard')
    """
    validate_input(X)

    warnings = []
    params_used = {'method': method}

    if custom_scaler is not None:
        try:
            scaled_X = custom_scaler(X)
        except Exception as e:
            warnings.append(f"Custom scaler failed: {str(e)}")
            scaled_X = X.copy()
    else:
        if method == 'none':
            scaled_X = X.copy()
        elif method == 'standard':
            scaled_X = standard_scale(X)
        elif method == 'minmax':
            scaled_X = minmax_scale(X)
        elif method == 'robust':
            scaled_X = robust_scale(X)
        else:
            raise ValueError(f"Unknown method: {method}")

    return {
        'result': scaled_X,
        'metrics': {},
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# feature_engineering
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def feature_engineering_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Perform feature engineering for generalized regression.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - normalizer: Callable for feature normalization
    - metric: Metric to evaluate performance ('mse', 'mae', 'r2', or custom callable)
    - distance: Distance metric for feature transformations ('euclidean', 'manhattan', etc.)
    - solver: Solver method ('closed_form', 'gradient_descent', etc.)
    - regularization: Regularization type (None, 'l1', 'l2', 'elasticnet')
    - tol: Tolerance for convergence
    - max_iter: Maximum iterations
    - custom_weights: Custom weights for samples

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Apply normalization if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Prepare weights if specified
    sample_weights = _prepare_sample_weights(y.shape[0], custom_weights)

    # Solve the regression problem
    result = _solve_regression(
        X_normalized, y, solver, regularization,
        metric=metric, distance=distance,
        tol=tol, max_iter=max_iter
    )

    # Calculate metrics
    metrics = _calculate_metrics(y, result['predictions'], metric)

    # Prepare output
    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
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

def _apply_normalization(
    X: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _prepare_sample_weights(
    n_samples: int,
    custom_weights: Optional[np.ndarray]
) -> np.ndarray:
    """Prepare sample weights."""
    if custom_weights is not None:
        if len(custom_weights) != n_samples:
            raise ValueError("custom_weights must match number of samples")
        return custom_weights
    return np.ones(n_samples)

def _solve_regression(
    X: np.ndarray,
    y: np.ndarray,
    solver: str,
    regularization: Optional[str],
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve the regression problem with specified solver."""
    if solver == 'closed_form':
        return _solve_closed_form(X, y, regularization)
    elif solver == 'gradient_descent':
        return _solve_gradient_descent(X, y, regularization, metric, distance, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _solve_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str]
) -> Dict[str, Any]:
    """Solve regression using closed form solution."""
    if regularization is None:
        coefficients = np.linalg.pinv(X) @ y
    elif regularization == 'l2':
        coefficients = _ridge_regression(X, y)
    else:
        raise ValueError(f"Unsupported regularization for closed form: {regularization}")

    predictions = X @ coefficients
    return {
        'coefficients': coefficients,
        'predictions': predictions
    }

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve regression using gradient descent."""
    # Initialize coefficients
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)

    # Gradient descent parameters
    learning_rate = 0.01

    for _ in range(max_iter):
        gradients = _compute_gradients(X, y, coefficients, regularization)
        coefficients -= learning_rate * gradients

    predictions = X @ coefficients
    return {
        'coefficients': coefficients,
        'predictions': predictions
    }

def _compute_gradients(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Compute gradients for gradient descent."""
    residuals = X @ coefficients - y
    gradients = (X.T @ residuals) / X.shape[0]

    if regularization == 'l2':
        gradients += 2 * coefficients

    return gradients

def _ridge_regression(
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Perform ridge regression."""
    alpha = 1.0  # Default regularization strength
    identity = np.eye(X.shape[1])
    coefficients = np.linalg.inv(X.T @ X + alpha * identity) @ X.T @ y
    return coefficients

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    if callable(metric):
        return {'custom_metric': metric(y_true, y_pred)}

    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

################################################################################
# selection_variables
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def selection_variables_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Perform variable selection in generalized regression.

    Parameters:
    -----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalizer : Optional[Callable], default=None
        Function to normalize the design matrix.
    metric : Union[str, Callable], default='mse'
        Metric to evaluate model performance. Can be 'mse', 'mae', 'r2', or a custom callable.
    distance : Union[str, Callable], default='euclidean'
        Distance metric for feature selection. Can be 'euclidean', 'manhattan', or a custom callable.
    solver : str, default='closed_form'
        Solver to use for optimization. Options: 'closed_form', 'gradient_descent', etc.
    regularization : Optional[str], default=None
        Regularization type. Options: 'l1', 'l2', 'elasticnet'.
    tol : float, default=1e-4
        Tolerance for convergence.
    max_iter : int, default=1000
        Maximum number of iterations.
    custom_metric : Optional[Callable], default=None
        Custom metric function if not using built-in metrics.
    custom_distance : Optional[Callable], default=None
        Custom distance function if not using built-in distances.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Select solver and compute parameters
    if solver == 'closed_form':
        params = _solve_closed_form(X_normalized, y, regularization)
    else:
        params = _solve_iterative(
            X_normalized, y,
            solver=solver,
            metric=metric,
            distance=distance,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter,
            custom_metric=custom_metric,
            custom_distance=custom_distance
        )

    # Compute metrics
    y_pred = X_normalized @ params['coefficients']
    metrics = _compute_metrics(y, y_pred, metric, custom_metric)

    # Prepare results
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric if isinstance(metric, str) else custom_metric.__name__ if custom_metric else None,
            'distance': distance if isinstance(distance, str) else custom_distance.__name__ if custom_distance else None,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(params)
    }

    return result

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

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to the design matrix."""
    if normalizer is None:
        return X
    return normalizer(X)

def _solve_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str]
) -> Dict[str, Any]:
    """Solve regression using closed-form solution."""
    if regularization is None:
        coefficients = np.linalg.pinv(X) @ y
    elif regularization == 'l2':
        coefficients = _ridge_regression(X, y)
    else:
        raise ValueError(f"Unsupported regularization: {regularization}")

    return {
        'coefficients': coefficients,
        'method': 'closed_form',
        'regularization': regularization
    }

def _solve_iterative(
    X: np.ndarray,
    y: np.ndarray,
    *,
    solver: str,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric: Optional[Callable],
    custom_distance: Optional[Callable]
) -> Dict[str, Any]:
    """Solve regression using iterative methods."""
    if solver == 'gradient_descent':
        return _gradient_descent(
            X, y,
            metric=metric,
            distance=distance,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter,
            custom_metric=custom_metric,
            custom_distance=custom_distance
        )
    else:
        raise ValueError(f"Unsupported solver: {solver}")

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric: Optional[Callable],
    custom_distance: Optional[Callable]
) -> Dict[str, Any]:
    """Perform gradient descent for regression."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        gradients = _compute_gradients(X, y, coefficients, distance, custom_distance)
        if regularization == 'l2':
            gradients += 2 * coefficients
        coefficients -= gradients

        current_loss = _compute_loss(y, X @ coefficients, metric, custom_metric)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return {
        'coefficients': coefficients,
        'method': 'gradient_descent',
        'regularization': regularization
    }

def _compute_gradients(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    distance: Union[str, Callable],
    custom_distance: Optional[Callable]
) -> np.ndarray:
    """Compute gradients for the given distance metric."""
    if isinstance(distance, str):
        if distance == 'euclidean':
            residuals = X @ coefficients - y
            return 2 * X.T @ residuals / len(y)
        else:
            raise ValueError(f"Unsupported distance: {distance}")
    elif custom_distance is not None:
        return _custom_gradient(X, y, coefficients, custom_distance)
    else:
        raise ValueError("No valid distance metric provided")

def _custom_gradient(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    distance_func: Callable
) -> np.ndarray:
    """Compute gradients for a custom distance function."""
    # This is a placeholder; actual implementation depends on the distance function
    return np.zeros(X.shape[1])

def _compute_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> float:
    """Compute loss using the specified metric."""
    if isinstance(metric, str):
        if metric == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - ss_res / ss_tot
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    elif custom_metric is not None:
        return custom_metric(y_true, y_pred)
    else:
        raise ValueError("No valid metric provided")

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Compute multiple metrics."""
    metrics = {}
    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        if metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        if metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - ss_res / ss_tot
    elif custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)

    return metrics

def _check_warnings(params: Dict[str, Any]) -> list:
    """Check for potential warnings in the results."""
    warnings = []
    if np.any(np.isnan(params['coefficients'])):
        warnings.append("Coefficients contain NaN values")
    if np.any(np.isinf(params['coefficients'])):
        warnings.append("Coefficients contain infinite values")
    return warnings

def _ridge_regression(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Perform ridge regression."""
    I = np.eye(X.shape[1])
    coefficients = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return coefficients

# Example usage:
"""
X = np.random.rand(100, 5)
y = X @ np.array([1.2, -0.8, 0.5, 0.3, -0.1]) + np.random.normal(0, 0.1, 100)

result = selection_variables_fit(
    X, y,
    normalizer=None,
    metric='mse',
    distance='euclidean',
    solver='closed_form',
    regularization=None
)
"""

################################################################################
# interaction_terms
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def interaction_terms_fit(
    X: np.ndarray,
    y: np.ndarray,
    interaction_pairs: List[tuple],
    normalizer: Callable = None,
    metric: str = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit a generalized linear model with interaction terms.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    interaction_pairs : List[tuple]
        List of tuples specifying feature pairs to create interactions
    normalizer : Callable, optional
        Function for normalizing features (default: None)
    metric : str or Callable, optional
        Metric to evaluate model performance (default: 'mse')
    solver : str, optional
        Solver to use for optimization (default: 'closed_form')
    regularization : str, optional
        Type of regularization to apply (default: None)
    alpha : float, optional
        Regularization strength (default: 1.0)
    max_iter : int, optional
        Maximum number of iterations (default: 1000)
    tol : float, optional
        Tolerance for convergence (default: 1e-4)
    random_state : int, optional
        Random seed for reproducibility (default: None)
    custom_metric : Callable, optional
        Custom metric function (default: None)

    Returns:
    --------
    Dict containing:
        - 'result': Model coefficients
        - 'metrics': Dictionary of computed metrics
        - 'params_used': Parameters used in fitting
        - 'warnings': List of warnings encountered

    Example:
    --------
    >>> X = np.random.rand(100, 3)
    >>> y = np.random.rand(100)
    >>> pairs = [(0, 1), (1, 2)]
    >>> result = interaction_terms_fit(X, y, pairs)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Create interaction terms
    X_with_interactions = _create_interaction_terms(X_normalized, interaction_pairs)

    # Prepare solver parameters
    solver_params = {
        'max_iter': max_iter,
        'tol': tol,
        'random_state': random_state
    }

    # Fit model based on solver choice
    if solver == 'closed_form':
        coefficients = _closed_form_solver(X_with_interactions, y)
    elif solver == 'gradient_descent':
        coefficients = _gradient_descent_solver(
            X_with_interactions, y,
            regularization=regularization,
            alpha=alpha,
            **solver_params
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(
        X_with_interactions, y,
        coefficients,
        metric=metric,
        custom_metric=custom_metric
    )

    # Prepare output dictionary
    result = {
        'result': coefficients,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'alpha': alpha
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

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X

    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _create_interaction_terms(X: np.ndarray, interaction_pairs: List[tuple]) -> np.ndarray:
    """Create interaction terms from specified feature pairs."""
    n_samples = X.shape[0]
    original_features = X.copy()

    for i, j in interaction_pairs:
        if i >= X.shape[1] or j >= X.shape[1]:
            raise ValueError(f"Invalid interaction pair ({i}, {j})")
        original_features = np.hstack((original_features, X[:, i] * X[:, j]))

    return original_features

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve using closed-form solution."""
    X_tx = X.T @ X
    if np.linalg.det(X_tx) == 0:
        raise ValueError("Matrix is singular, cannot compute closed-form solution")
    return np.linalg.inv(X_tx) @ X.T @ y

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Solve using gradient descent."""
    if random_state is not None:
        np.random.seed(random_state)

    n_features = X.shape[1]
    coefficients = np.random.randn(n_features)
    learning_rate = 0.01
    prev_loss = float('inf')

    for _ in range(max_iter):
        gradients = 2 * X.T @ (X @ coefficients - y) / len(y)

        if regularization == 'l1':
            gradients += alpha * np.sign(coefficients)
        elif regularization == 'l2':
            gradients += 2 * alpha * coefficients
        elif regularization == 'elasticnet':
            gradients += alpha * (np.sign(coefficients) + 2 * coefficients)

        coefficients -= learning_rate * gradients
        current_loss = np.mean((X @ coefficients - y) ** 2)

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return coefficients

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Calculate model metrics."""
    y_pred = X @ coefficients

    metrics_dict = {}

    if metric == 'mse':
        metrics_dict['mse'] = np.mean((y_pred - y) ** 2)
    elif metric == 'mae':
        metrics_dict['mae'] = np.mean(np.abs(y_pred - y))
    elif metric == 'r2':
        ss_res = np.sum((y_pred - y) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics_dict['r2'] = 1 - (ss_res / ss_tot)

    if custom_metric is not None:
        try:
            metrics_dict['custom'] = custom_metric(y, y_pred)
        except Exception as e:
            metrics_dict['custom_error'] = str(e)

    return metrics_dict
