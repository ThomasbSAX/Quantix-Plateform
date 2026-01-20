"""
Quantix – Module regression_bayesienne
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# priors
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def priors_fit(
    X: np.ndarray,
    y: np.ndarray,
    prior_type: str = 'normal',
    prior_params: Optional[Dict[str, Union[float, np.ndarray]]] = None,
    solver: str = 'closed_form',
    metric: Union[str, Callable] = 'mse',
    normalize: bool = True,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict:
    """
    Fit Bayesian regression priors to data.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    prior_type : str
        Type of prior distribution ('normal', 'laplace', 'uniform')
    prior_params : dict
        Dictionary of parameters for the prior distribution
    solver : str
        Solver method ('closed_form', 'gradient_descent')
    metric : str or callable
        Evaluation metric ('mse', 'mae', custom function)
    normalize : bool
        Whether to normalize features
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    random_state : int or None
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = priors_fit(X, y, prior_type='normal', solver='closed_form')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_norm = _normalize_data(X) if normalize else X

    # Set default prior parameters
    if prior_params is None:
        prior_params = _get_default_prior_params(prior_type)

    # Select solver
    if solver == 'closed_form':
        params = _solve_closed_form(X_norm, y, prior_type, prior_params)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(
            X_norm, y, prior_type, prior_params,
            metric=metric, tol=tol, max_iter=max_iter,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(X_norm, y, params, metric)

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'prior_type': prior_type,
            'solver': solver,
            'normalize': normalize
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
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

def _normalize_data(X: np.ndarray) -> np.ndarray:
    """Normalize features to zero mean and unit variance."""
    return (X - X.mean(axis=0)) / X.std(axis=0)

def _get_default_prior_params(prior_type: str) -> Dict[str, Union[float, np.ndarray]]:
    """Get default parameters for different prior types."""
    if prior_type == 'normal':
        return {'mu': 0.0, 'sigma': 1.0}
    elif prior_type == 'laplace':
        return {'mu': 0.0, 'b': 1.0}
    elif prior_type == 'uniform':
        return {'low': -1.0, 'high': 1.0}
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

def _solve_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    prior_type: str,
    prior_params: Dict[str, Union[float, np.ndarray]]
) -> np.ndarray:
    """Solve Bayesian regression using closed-form solution."""
    # This is a simplified version - actual implementation would depend on prior type
    XtX = X.T @ X
    Xty = X.T @ y

    if prior_type == 'normal':
        # Add regularization term based on prior
        reg_term = np.diag(prior_params['sigma'] ** -2)
        params = np.linalg.inv(XtX + reg_term) @ Xty
    else:
        # For other priors, might need different approach
        params = np.linalg.pinv(X) @ y

    return params

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    prior_type: str,
    prior_params: Dict[str, Union[float, np.ndarray]],
    metric: Union[str, Callable],
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> np.ndarray:
    """Solve Bayesian regression using gradient descent."""
    if random_state is not None:
        np.random.seed(random_state)

    n_features = X.shape[1]
    params = np.random.randn(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        grad = _compute_gradient(X, y, params, prior_type, prior_params)
        params -= 0.01 * grad

        current_loss = _compute_loss(X, y, params, prior_type, prior_params)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return params

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    prior_type: str,
    prior_params: Dict[str, Union[float, np.ndarray]]
) -> np.ndarray:
    """Compute gradient of the loss function."""
    residuals = y - X @ params

    # Data fit term
    grad_data = -2 * X.T @ residuals

    # Prior term (simplified for normal prior)
    if prior_type == 'normal':
        grad_prior = 2 * (params - prior_params['mu']) / (prior_params['sigma'] ** 2)
    else:
        grad_prior = np.zeros_like(params)

    return grad_data + grad_prior

def _compute_loss(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    prior_type: str,
    prior_params: Dict[str, Union[float, np.ndarray]]
) -> float:
    """Compute the loss function."""
    residuals = y - X @ params
    data_loss = np.sum(residuals ** 2)

    if prior_type == 'normal':
        prior_loss = np.sum((params - prior_params['mu'])**2) / (prior_params['sigma'] ** 2)
    elif prior_type == 'laplace':
        prior_loss = np.sum(np.abs(params - prior_params['mu'])) / prior_params['b']
    else:
        prior_loss = 0.0

    return data_loss + prior_loss

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate evaluation metrics."""
    y_pred = X @ params
    metrics_dict = {}

    if metric == 'mse' or isinstance(metric, str) and metric.lower() == 'mse':
        metrics_dict['mse'] = np.mean((y - y_pred) ** 2)
    elif metric == 'mae' or isinstance(metric, str) and metric.lower() == 'mae':
        metrics_dict['mae'] = np.mean(np.abs(y - y_pred))
    elif metric == 'r2' or isinstance(metric, str) and metric.lower() == 'r2':
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics_dict['r2'] = 1 - (ss_res / ss_tot)
    elif callable(metric):
        metrics_dict['custom'] = metric(y, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics_dict

################################################################################
# likelihood
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for likelihood computation."""
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
    """Normalize input data based on specified method."""
    if custom_norm is not None:
        X_norm = custom_norm(X)
        y_norm = custom_norm(y.reshape(-1, 1)).flatten()
    elif normalization == 'standard':
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_norm = (X - X_mean) / (X_std + 1e-8)
        y_norm = (y - y.mean()) / (y.std() + 1e-8)
    elif normalization == 'minmax':
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_norm = (X - X_min) / ((X_max - X_min + 1e-8))
        y_norm = (y - y.min()) / ((y.max() - y.min() + 1e-8))
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

def compute_likelihood(X: np.ndarray, y: np.ndarray,
                      prior_mean: float = 0.0,
                      prior_var: float = 1.0,
                      noise_var: float = 1.0) -> np.ndarray:
    """Compute the likelihood for Bayesian regression."""
    n_samples, n_features = X.shape
    beta_posterior_mean = np.linalg.solve(
        X.T @ X + (prior_var / noise_var) * np.eye(n_features),
        X.T @ y + (prior_mean / noise_var) * np.ones(n_features)
    )
    beta_posterior_cov = np.linalg.inv(
        X.T @ X + (prior_var / noise_var) * np.eye(n_features)
    )
    likelihood = (
        (1 / ((2 * np.pi * noise_var) ** (n_samples / 2))) *
        np.exp(-0.5 * np.sum((y - X @ beta_posterior_mean) ** 2 / noise_var))
    )
    return likelihood

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     metric: str = 'mse',
                     custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Calculate specified metrics between true and predicted values."""
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)
    else:
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
        elif metric == 'logloss':
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            metrics['logloss'] = -np.mean(y_true * np.log(y_pred) +
                                         (1 - y_true) * np.log(1 - y_pred))
    return metrics

def likelihood_fit(X: np.ndarray, y: np.ndarray,
                  normalization: str = 'standard',
                  metric: str = 'mse',
                  prior_mean: float = 0.0,
                  prior_var: float = 1.0,
                  noise_var: float = 1.0,
                  custom_norm: Optional[Callable] = None,
                  custom_metric: Optional[Callable] = None) -> Dict:
    """Main function to compute Bayesian likelihood with configurable options."""
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y,
                                  normalization=normalization,
                                  custom_norm=custom_norm)

    # Compute likelihood
    likelihood_value = compute_likelihood(X_norm, y_norm,
                                        prior_mean=prior_mean,
                                        prior_var=prior_var,
                                        noise_var=noise_var)

    # Calculate metrics
    y_pred = X_norm @ np.linalg.solve(
        X_norm.T @ X_norm + (prior_var / noise_var) * np.eye(X_norm.shape[1]),
        X_norm.T @ y_norm + (prior_mean / noise_var) * np.ones(X_norm.shape[1])
    )
    metrics = calculate_metrics(y_norm, y_pred,
                              metric=metric,
                              custom_metric=custom_metric)

    # Prepare output
    result = {
        'result': float(likelihood_value),
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'prior_mean': prior_mean,
            'prior_var': prior_var,
            'noise_var': noise_var
        },
        'warnings': []
    }

    return result

# Example usage:
"""
X = np.random.randn(100, 5)
y = np.random.randn(100)

result = likelihood_fit(X, y,
                      normalization='standard',
                      metric='mse')
"""

################################################################################
# posterior
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def posterior_fit(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: Optional[np.ndarray] = None,
    prior_cov: Optional[np.ndarray] = None,
    likelihood_var: float = 1.0,
    solver: str = 'closed_form',
    metric: Union[str, Callable] = 'mse',
    normalize: str = 'standard',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_prior: Optional[Callable] = None,
    custom_likelihood: Optional[Callable] = None
) -> Dict:
    """
    Compute the posterior distribution in Bayesian regression.

    Parameters:
    -----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    prior_mean : Optional[np.ndarray]
        Mean of the prior distribution
    prior_cov : Optional[np.ndarray]
        Covariance matrix of the prior distribution
    likelihood_var : float
        Variance of the likelihood (noise variance)
    solver : str
        Solver method ('closed_form', 'gradient_descent')
    metric : Union[str, Callable]
        Metric to evaluate ('mse', 'mae', custom callable)
    normalize : str
        Normalization method ('none', 'standard', 'minmax')
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    custom_prior : Optional[Callable]
        Custom prior distribution function
    custom_likelihood : Optional[Callable]
        Custom likelihood function

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_norm, y_norm = _apply_normalization(X, y, normalize)

    # Set default prior if not provided
    if custom_prior is None:
        prior_mean, prior_cov = _set_default_prior(X.shape[1], prior_mean, prior_cov)
    else:
        prior_mean, prior_cov = custom_prior(X.shape[1])

    # Solve for posterior
    if solver == 'closed_form':
        result = _solve_closed_form(X_norm, y_norm, prior_mean, prior_cov, likelihood_var)
    else:
        result = _solve_iterative(X_norm, y_norm, prior_mean, prior_cov,
                                likelihood_var, solver, tol, max_iter)

    # Compute metrics
    metrics = _compute_metrics(X_norm, y_norm, result['posterior_mean'], metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'solver': solver,
            'metric': metric,
            'normalize': normalize
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input contains infinite values")

def _apply_normalization(X: np.ndarray, y: np.ndarray,
                        method: str) -> tuple[np.ndarray, np.ndarray]:
    """Apply data normalization."""
    if method == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif method == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    else:
        X_norm, y_norm = X.copy(), y.copy()
    return X_norm, y_norm

def _set_default_prior(n_features: int,
                      prior_mean: Optional[np.ndarray] = None,
                      prior_cov: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """Set default prior distribution."""
    if prior_mean is None:
        prior_mean = np.zeros(n_features)
    else:
        if len(prior_mean) != n_features:
            raise ValueError("prior_mean must match number of features")

    if prior_cov is None:
        prior_cov = np.eye(n_features)
    else:
        if prior_cov.shape != (n_features, n_features):
            raise ValueError("prior_cov must be square matrix matching number of features")
    return prior_mean, prior_cov

def _solve_closed_form(X: np.ndarray,
                      y: np.ndarray,
                      prior_mean: np.ndarray,
                      prior_cov: np.ndarray,
                      likelihood_var: float) -> Dict:
    """Closed form solution for posterior."""
    # Compute posterior mean and covariance
    post_cov = np.linalg.inv(np.linalg.inv(prior_cov) + (1/likelihood_var) * X.T @ X)
    post_mean = post_cov @ (np.linalg.inv(prior_cov) @ prior_mean + (1/likelihood_var) * X.T @ y)

    return {
        'posterior_mean': post_mean,
        'posterior_cov': post_cov
    }

def _solve_iterative(X: np.ndarray,
                    y: np.ndarray,
                    prior_mean: np.ndarray,
                    prior_cov: np.ndarray,
                    likelihood_var: float,
                    solver: str,
                    tol: float,
                    max_iter: int) -> Dict:
    """Iterative solution for posterior."""
    n_features = X.shape[1]
    post_mean = prior_mean.copy()
    prev_mean = None

    for _ in range(max_iter):
        if solver == 'gradient_descent':
            grad = (1/likelihood_var) * X.T @ (X @ post_mean - y)
            grad += np.linalg.inv(prior_cov) @ (post_mean - prior_mean)
            post_mean -= 0.1 * grad

        elif solver == 'newton':
            hess = (1/likelihood_var) * X.T @ X + np.linalg.inv(prior_cov)
            grad = (1/likelihood_var) * X.T @ (X @ post_mean - y)
            grad += np.linalg.inv(prior_cov) @ (post_mean - prior_mean)
            post_mean -= np.linalg.inv(hess) @ grad

        if prev_mean is not None and np.linalg.norm(post_mean - prev_mean) < tol:
            break
        prev_mean = post_mean.copy()

    # Compute posterior covariance (approximation)
    post_cov = np.linalg.inv((1/likelihood_var) * X.T @ X + np.linalg.inv(prior_cov))

    return {
        'posterior_mean': post_mean,
        'posterior_cov': post_cov
    }

def _compute_metrics(X: np.ndarray,
                    y: np.ndarray,
                    post_mean: np.ndarray,
                    metric: Union[str, Callable]) -> Dict:
    """Compute evaluation metrics."""
    y_pred = X @ post_mean
    metrics = {}

    if metric == 'mse':
        metrics['mse'] = np.mean((y - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    elif callable(metric):
        metrics['custom'] = metric(y, y_pred)

    return metrics

################################################################################
# bayesian_linear_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def bayesian_linear_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    prior_mean: Optional[np.ndarray] = None,
    prior_covariance: Optional[np.ndarray] = None,
    likelihood_variance: float = 1.0,
    normalize: str = 'standard',
    solver: str = 'closed_form',
    metric: Union[str, Callable] = 'mse',
    max_iter: int = 1000,
    tol: float = 1e-6,
    random_state: Optional[int] = None
) -> Dict:
    """
    Perform Bayesian Linear Regression.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    prior_mean : Optional[np.ndarray]
        Prior mean for the coefficients. If None, uses zero vector.
    prior_covariance : Optional[np.ndarray]
        Prior covariance matrix for the coefficients. If None, uses identity matrix.
    likelihood_variance : float
        Variance of the likelihood (noise variance).
    normalize : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    solver : str
        Solver method: 'closed_form' or 'gradient_descent'.
    metric : Union[str, Callable]
        Metric to evaluate: 'mse', 'mae', 'r2', or custom callable.
    max_iter : int
        Maximum number of iterations for iterative solvers.
    tol : float
        Tolerance for convergence.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Estimated coefficients.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used in the fitting process.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = bayesian_linear_regression_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    X_norm, y_norm, normalize_params = _normalize_data(X, y, normalize)

    # Set default prior if not provided
    n_features = X_norm.shape[1]
    if prior_mean is None:
        prior_mean = np.zeros(n_features)
    if prior_covariance is None:
        prior_covariance = np.eye(n_features)

    # Choose solver
    if solver == 'closed_form':
        coefficients = _closed_form_solver(X_norm, y_norm, prior_mean, prior_covariance, likelihood_variance)
    elif solver == 'gradient_descent':
        coefficients = _gradient_descent_solver(X_norm, y_norm, prior_mean, prior_covariance,
                                               likelihood_variance, max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(y_norm, X_norm @ coefficients, metric)

    # Prepare output
    result = {
        'result': coefficients,
        'metrics': metrics,
        'params_used': {
            'prior_mean': prior_mean.tolist(),
            'prior_covariance': prior_covariance.tolist(),
            'likelihood_variance': likelihood_variance,
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
    method: str
) -> tuple[np.ndarray, np.ndarray, Dict]:
    """Normalize data based on the specified method."""
    params = {}
    X_norm = X.copy()
    y_norm = y.copy()

    if method == 'standard':
        mean_X = X.mean(axis=0)
        std_X = X.std(axis=0)
        params['mean_X'] = mean_X.tolist()
        params['std_X'] = std_X.tolist()
        X_norm = (X - mean_X) / (std_X + 1e-8)

        mean_y = y.mean()
        std_y = y.std()
        params['mean_y'] = mean_y
        params['std_y'] = std_y
        y_norm = (y - mean_y) / (std_y + 1e-8)

    elif method == 'minmax':
        min_X = X.min(axis=0)
        max_X = X.max(axis=0)
        params['min_X'] = min_X.tolist()
        params['max_X'] = max_X.tolist()
        X_norm = (X - min_X) / ((max_X - min_X + 1e-8))

        min_y = y.min()
        max_y = y.max()
        params['min_y'] = min_y
        params['max_y'] = max_y
        y_norm = (y - min_y) / ((max_y - min_y + 1e-8))

    elif method == 'robust':
        median_X = np.median(X, axis=0)
        iqr_X = np.subtract(*np.percentile(X, [75, 25], axis=0))
        params['median_X'] = median_X.tolist()
        params['iqr_X'] = iqr_X.tolist()
        X_norm = (X - median_X) / (iqr_X + 1e-8)

        median_y = np.median(y)
        iqr_y = np.subtract(*np.percentile(y, [75, 25]))
        params['median_y'] = median_y
        params['iqr_y'] = iqr_y
        y_norm = (y - median_y) / (iqr_y + 1e-8)

    return X_norm, y_norm, params

def _closed_form_solver(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    likelihood_variance: float
) -> np.ndarray:
    """Closed-form solution for Bayesian Linear Regression."""
    n_features = X.shape[1]
    posterior_covariance = np.linalg.inv(X.T @ X / likelihood_variance + np.linalg.inv(prior_covariance))
    posterior_mean = posterior_covariance @ (X.T @ y / likelihood_variance + np.linalg.inv(prior_covariance) @ prior_mean)
    return posterior_mean

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    likelihood_variance: float,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Gradient descent solver for Bayesian Linear Regression."""
    n_features = X.shape[1]
    coefficients = prior_mean.copy()
    prev_coefficients = np.zeros_like(coefficients)

    for _ in range(max_iter):
        gradient = (X.T @ (X @ coefficients - y) / likelihood_variance +
                   np.linalg.inv(prior_covariance) @ (coefficients - prior_mean))

        coefficients -= gradient

        if np.linalg.norm(coefficients - prev_coefficients) < tol:
            break

        prev_coefficients = coefficients.copy()

    return coefficients

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Compute metrics based on the specified method."""
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - ss_res / (ss_tot + 1e-8)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        metrics['custom'] = metric(y_true, y_pred)

    return metrics

################################################################################
# ridge_regression_bayesienne
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def ridge_regression_bayesienne_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict:
    """
    Fit a Bayesian Ridge Regression model.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalizer : Optional[Callable], optional
        Function to normalize the input features. Default is None.
    metric : Union[str, Callable], optional
        Metric to evaluate the model. Can be 'mse', 'mae', 'r2', or a custom callable.
        Default is 'mse'.
    solver : str, optional
        Solver to use. Can be 'closed_form', 'gradient_descent', or 'newton'.
        Default is 'closed_form'.
    alpha : float, optional
        Regularization strength. Default is 1.0.
    max_iter : int, optional
        Maximum number of iterations for iterative solvers. Default is 1000.
    tol : float, optional
        Tolerance for stopping criteria. Default is 1e-4.
    random_state : Optional[int], optional
        Random seed for reproducibility. Default is None.
    verbose : bool, optional
        Whether to print progress information. Default is False.

    Returns
    -------
    Dict
        A dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = ridge_regression_bayesienne_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if a normalizer is provided
    X_norm = _apply_normalization(X, normalizer)

    # Initialize parameters
    n_samples, n_features = X_norm.shape

    # Choose solver
    if solver == 'closed_form':
        weights, _ = _ridge_regression_closed_form(X_norm, y, alpha)
    elif solver == 'gradient_descent':
        weights, _ = _ridge_regression_gradient_descent(
            X_norm, y, alpha, max_iter, tol, random_state, verbose
        )
    elif solver == 'newton':
        weights, _ = _ridge_regression_newton(
            X_norm, y, alpha, max_iter, tol, random_state, verbose
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(y, X_norm @ weights, metric)

    # Prepare output
    result = {
        'result': {'weights': weights},
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver,
            'alpha': alpha,
            'max_iter': max_iter,
            'tol': tol,
            'random_state': random_state
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

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to the input data."""
    if normalizer is not None:
        return normalizer(X)
    return X

def _ridge_regression_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float
) -> tuple:
    """Closed-form solution for Bayesian Ridge Regression."""
    n_samples, n_features = X.shape
    I = np.eye(n_features)
    weights = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return weights, None

def _ridge_regression_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
    random_state: Optional[int],
    verbose: bool
) -> tuple:
    """Gradient descent solver for Bayesian Ridge Regression."""
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features)

    for i in range(max_iter):
        gradient = 2 * X.T @ (X @ weights - y) + 2 * alpha * weights
        weights -= gradient

        if verbose and i % 100 == 0:
            print(f"Iteration {i}, weights: {weights}")

        if np.linalg.norm(gradient) < tol:
            break

    return weights, None

def _ridge_regression_newton(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
    random_state: Optional[int],
    verbose: bool
) -> tuple:
    """Newton's method solver for Bayesian Ridge Regression."""
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features)

    for i in range(max_iter):
        gradient = 2 * X.T @ (X @ weights - y) + 2 * alpha * weights
        hessian = 2 * (X.T @ X + alpha * np.eye(n_features))
        weights -= np.linalg.inv(hessian) @ gradient

        if verbose and i % 100 == 0:
            print(f"Iteration {i}, weights: {weights}")

        if np.linalg.norm(gradient) < tol:
            break

    return weights, None

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate the specified metrics."""
    metrics = {}

    if metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - ss_res / ss_tot
    elif callable(metric):
        metrics['custom'] = metric(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

################################################################################
# lasso_regression_bayesienne
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def lasso_regression_bayesienne_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'coordinate_descent',
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Fit Bayesian Lasso regression model.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the design matrix.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric to evaluate the model. Can be 'mse', 'mae', 'r2' or a custom callable.
    solver : str
        Solver to use. Options: 'coordinate_descent'.
    alpha : float
        Regularization strength.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for stopping criteria.
    random_state : Optional[int]
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]
        Dictionary containing:
        - 'result': Estimated coefficients.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': List of warnings encountered.

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randn(100)
    >>> result = lasso_regression_bayesienne_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = normalizer(X)
    y_norm = normalizer(y.reshape(-1, 1)).flatten()

    # Initialize parameters
    n_samples, n_features = X_norm.shape
    beta = np.zeros(n_features)
    if random_state is not None:
        np.random.seed(random_state)

    # Choose solver
    if solver == 'coordinate_descent':
        beta, metrics = _coordinate_descent_solver(
            X_norm, y_norm, alpha, max_iter, tol, verbose
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    y_pred = X_norm @ beta
    metrics_result = {'metric': metric_func(y_norm, y_pred)}

    # Prepare output
    return {
        'result': beta,
        'metrics': metrics_result,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'metric': metric,
            'solver': solver,
            'alpha': alpha,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

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

def _coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
    verbose: bool
) -> tuple[np.ndarray, Dict[str, float]]:
    """Coordinate descent solver for Bayesian Lasso."""
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    metrics_history = []

    for iteration in range(max_iter):
        beta_old = beta.copy()

        for j in range(n_features):
            X_j = X[:, j]
            r = y - np.dot(X, beta) + beta[j] * X_j

            if np.linalg.norm(X_j) == 0:
                beta[j] = 0
            else:
                rho_j = np.dot(X_j, X_j)
                c_j = np.dot(X_j, r) / rho_j
                beta[j] = _soft_threshold(c_j, alpha / rho_j)

        # Check convergence
        diff = np.linalg.norm(beta - beta_old)
        if diff < tol:
            if verbose:
                print(f"Converged at iteration {iteration}")
            break

        # Compute metrics
        y_pred = X @ beta
        mse = np.mean((y - y_pred) ** 2)
        metrics_history.append({'iteration': iteration, 'mse': mse})

    return beta, {'history': metrics_history}

def _soft_threshold(rho: float, lambda_: float) -> float:
    """Soft thresholding operator."""
    if rho < -lambda_:
        return rho + lambda_
    elif rho > lambda_:
        return rho - lambda_
    else:
        return 0.0

def _get_metric_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get metric function based on string identifier."""
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
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

################################################################################
# bayesian_ridge_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def bayesian_ridge_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1.0,
    lambda_: float = 1.0,
    normalize: str = 'standard',
    solver: str = 'closed_form',
    metric: Union[str, Callable] = 'mse',
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Bayesian Ridge Regression implementation.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    alpha : float, optional
        Regularization parameter for the prior on weights (default=1.0).
    lambda_ : float, optional
        Regularization parameter for the prior on noise (default=1.0).
    normalize : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust' (default='standard').
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', or 'coordinate_descent' (default='closed_form').
    metric : str or callable, optional
        Metric to evaluate: 'mse', 'mae', 'r2', or custom callable (default='mse').
    tol : float, optional
        Tolerance for stopping criteria (default=1e-4).
    max_iter : int, optional
        Maximum number of iterations (default=1000).
    random_state : int, optional
        Random seed for reproducibility (default=None).

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Estimated weights.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used in the fit.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = bayesian_ridge_regression_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_norm, y_norm, normalizer = _normalize_data(X, y, normalize)

    # Initialize parameters
    n_samples, n_features = X_norm.shape

    # Choose solver
    if solver == 'closed_form':
        weights = _closed_form_solution(X_norm, y_norm, alpha, lambda_)
    elif solver == 'gradient_descent':
        weights = _gradient_descent_solver(X_norm, y_norm, alpha, lambda_, tol, max_iter, random_state)
    elif solver == 'coordinate_descent':
        weights = _coordinate_descent_solver(X_norm, y_norm, alpha, lambda_, tol, max_iter, random_state)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_norm, y_norm, weights, metric)

    # Prepare output
    result = {
        'result': weights,
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'lambda_': lambda_,
            'normalize': normalize,
            'solver': solver,
            'metric': metric,
            'tol': tol,
            'max_iter': max_iter
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

def _normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    """Normalize data based on the specified method."""
    if method == 'none':
        return X, y, None
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
        normalizer = {'mean': mean, 'std': std}
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
        normalizer = {'min': min_val, 'max': max_val}
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
        normalizer = {'median': median, 'iqr': iqr}
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Normalize y if needed (could be extended)
    return X_norm, y, normalizer

def _closed_form_solution(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    lambda_: float
) -> np.ndarray:
    """Compute the closed-form solution for Bayesian Ridge Regression."""
    n_samples, n_features = X.shape
    I = np.eye(n_features)
    weights = np.linalg.inv(X.T @ X + alpha * I) @ (X.T @ y)
    return weights

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    lambda_: float,
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> np.ndarray:
    """Gradient descent solver for Bayesian Ridge Regression."""
    if random_state is not None:
        np.random.seed(random_state)
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features)

    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ weights - y) + 2 * alpha * weights
        weights -= gradient

        if np.linalg.norm(gradient) < tol:
            break

    return weights

def _coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    lambda_: float,
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> np.ndarray:
    """Coordinate descent solver for Bayesian Ridge Regression."""
    if random_state is not None:
        np.random.seed(random_state)
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)

    for _ in range(max_iter):
        old_weights = weights.copy()
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - np.dot(X, weights) + weights[j] * X_j
            numerator = X_j.T @ residuals
            denominator = X_j.T @ X_j + alpha
            weights[j] = numerator / denominator

        if np.linalg.norm(weights - old_weights) < tol:
            break

    return weights

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Compute metrics for the regression."""
    y_pred = X @ weights
    metrics = {}

    if metric == 'mse':
        metrics['mse'] = np.mean((y - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    elif callable(metric):
        metrics['custom'] = metric(y, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

################################################################################
# bayesian_lasso_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def bayesian_lasso_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    solver: str = "coordinate_descent",
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Bayesian Lasso Regression with configurable parameters.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize features. Default is identity.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric to evaluate performance. Can be "mse", "mae", "r2" or custom callable.
    solver : str
        Solver to use. Options: "coordinate_descent".
    alpha : float
        Regularization strength.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for stopping criteria.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - "result": Estimated coefficients.
        - "metrics": Computed metrics.
        - "params_used": Parameters used in the fit.
        - "warnings": Any warnings encountered.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = bayesian_lasso_regression_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_norm = normalizer(X)
    y_norm = normalizer(y.reshape(-1, 1)).flatten()

    # Initialize parameters
    n_features = X_norm.shape[1]
    coefs = np.zeros(n_features)

    # Solve using the specified solver
    if solver == "coordinate_descent":
        coefs = _coordinate_descent_solver(X_norm, y_norm, alpha, max_iter, tol, random_state)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    y_pred = X_norm @ coefs
    metrics = _compute_metrics(y_norm, y_pred, metric)

    # Prepare output
    result_dict = {
        "result": coefs,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if hasattr(normalizer, "__name__") else "custom",
            "metric": metric,
            "solver": solver,
            "alpha": alpha,
            "max_iter": max_iter,
            "tol": tol
        },
        "warnings": []
    }

    return result_dict

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or Inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or Inf values")

def _coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
    random_state: Optional[int]
) -> np.ndarray:
    """Coordinate descent solver for Bayesian Lasso."""
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)
    rng = np.random.RandomState(random_state)

    for _ in range(max_iter):
        old_coefs = coefs.copy()

        for j in range(n_features):
            # Compute residuals without feature j
            r = y - X[:, :j] @ coefs[:j] - X[:, j+1:] @ coefs[j+1:]

            # Compute correlation
            corr = X[:, j] @ r

            # Update coefficient
            if np.abs(corr) < alpha / 2:
                coefs[j] = 0
            else:
                coefs[j] = np.sign(corr) * (np.abs(corr) - alpha / 2) / (X[:, j] @ X[:, j])

        # Check convergence
        if np.linalg.norm(coefs - old_coefs) < tol:
            break

    return coefs

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics based on the specified metric."""
    metrics_dict = {}

    if isinstance(metric, str):
        if metric == "mse":
            metrics_dict["mse"] = np.mean((y_true - y_pred) ** 2)
        elif metric == "mae":
            metrics_dict["mae"] = np.mean(np.abs(y_true - y_pred))
        elif metric == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics_dict["r2"] = 1 - ss_res / ss_tot
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        metrics_dict["custom"] = metric(y_true, y_pred)

    return metrics_dict

################################################################################
# markov_chain_monte_carlo
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def markov_chain_monte_carlo_fit(
    likelihood: Callable,
    prior: Callable,
    n_samples: int = 10000,
    burn_in: int = 1000,
    thinning: int = 1,
    initial_state: Optional[np.ndarray] = None,
    proposal_scale: float = 1.0,
    metric: str = 'mse',
    normalizer: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None,
    tol: float = 1e-6,
    max_iter: int = 100000
) -> Dict[str, Any]:
    """
    Perform Markov Chain Monte Carlo sampling for Bayesian regression.

    Parameters:
    -----------
    likelihood : callable
        Likelihood function that takes parameters and data, returns log-likelihood.
    prior : callable
        Prior function that takes parameters and returns log-prior.
    n_samples : int, optional
        Number of samples to draw (default: 10000).
    burn_in : int, optional
        Number of initial samples to discard (default: 1000).
    thinning : int, optional
        Thinning interval for samples (default: 1).
    initial_state : np.ndarray, optional
        Initial state of the Markov chain (default: None).
    proposal_scale : float, optional
        Scale parameter for proposal distribution (default: 1.0).
    metric : str, optional
        Metric to evaluate ('mse', 'mae', 'r2') (default: 'mse').
    normalizer : callable, optional
        Normalization function (default: None).
    custom_metric : callable, optional
        Custom metric function (default: None).
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default: 100000).

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(likelihood, prior, n_samples, burn_in, thinning,
                    initial_state, proposal_scale)

    # Initialize chain
    if initial_state is None:
        initial_state = np.zeros(likelihood.__code__.co_argcount - 1)

    chain = np.zeros((n_samples, initial_state.shape[0]))
    current_state = initial_state.copy()

    # MCMC sampling
    for i in range(n_samples + burn_in):
        proposed_state = _propose_state(current_state, proposal_scale)
        log_acceptance_ratio = (
            likelihood(proposed_state) + prior(proposed_state) -
            likelihood(current_state) - prior(current_state)
        )
        acceptance_prob = np.minimum(1, np.exp(log_acceptance_ratio))

        if np.random.rand() < acceptance_prob:
            current_state = proposed_state

        if i >= burn_in and (i - burn_in) % thinning == 0:
            chain[(i - burn_in) // thinning - 1] = current_state

    # Calculate metrics
    metrics = _calculate_metrics(chain, metric, custom_metric)

    return {
        'result': chain,
        'metrics': metrics,
        'params_used': {
            'n_samples': n_samples,
            'burn_in': burn_in,
            'thinning': thinning,
            'proposal_scale': proposal_scale
        },
        'warnings': []
    }

def _validate_inputs(
    likelihood: Callable,
    prior: Callable,
    n_samples: int,
    burn_in: int,
    thinning: int,
    initial_state: Optional[np.ndarray],
    proposal_scale: float
) -> None:
    """Validate input parameters."""
    if not callable(likelihood):
        raise ValueError("Likelihood must be a callable function.")
    if not callable(prior):
        raise ValueError("Prior must be a callable function.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if burn_in < 0:
        raise ValueError("burn_in cannot be negative.")
    if thinning <= 0:
        raise ValueError("thinning must be positive.")
    if initial_state is not None and len(initial_state.shape) != 1:
        raise ValueError("initial_state must be a 1D array.")
    if proposal_scale <= 0:
        raise ValueError("proposal_scale must be positive.")

def _propose_state(
    current_state: np.ndarray,
    proposal_scale: float
) -> np.ndarray:
    """Propose a new state using symmetric Gaussian proposal."""
    return current_state + proposal_scale * np.random.randn(len(current_state))

def _calculate_metrics(
    chain: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Calculate metrics from the MCMC chain."""
    metrics = {}

    if custom_metric is not None:
        metrics['custom'] = custom_metric(chain)

    if metric == 'mse':
        metrics['mse'] = np.mean(np.var(chain, axis=0))
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(chain - np.median(chain, axis=0)))
    elif metric == 'r2':
        # Placeholder for R-squared calculation
        metrics['r2'] = 0.0

    return metrics

# Example usage:
"""
def example_likelihood(params):
    # Example likelihood function
    return -0.5 * np.sum(params**2)

def example_prior(params):
    # Example prior function
    return -np.sum(np.abs(params))

result = markov_chain_monte_carlo_fit(
    likelihood=example_likelihood,
    prior=example_prior,
    n_samples=5000,
    burn_in=1000
)
"""

################################################################################
# metropolis_hastings
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def metropolis_hastings_fit(
    log_posterior: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    proposal_distribution: Callable[[np.ndarray, float], np.ndarray],
    steps: int = 10000,
    burn_in: int = 1000,
    thinning: int = 5,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'mse',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    tol: float = 1e-6,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform Metropolis-Hastings algorithm for Bayesian regression.

    Parameters
    ----------
    log_posterior : callable
        Function that computes the log posterior distribution.
    initial_params : np.ndarray
        Initial parameters for the Markov chain.
    proposal_distribution : callable
        Function that generates a new candidate from current state and step size.
    steps : int, optional
        Number of MCMC steps to perform (default: 10000).
    burn_in : int, optional
        Number of initial steps to discard (default: 1000).
    thinning : int, optional
        Thinning interval for the chain (default: 5).
    normalizer : callable, optional
        Function to normalize parameters before evaluation.
    metric : str or callable, optional
        Metric to evaluate the performance ('mse', 'mae', 'r2') or custom callable.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing results, metrics, parameters used and warnings.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Validate inputs
    _validate_inputs(log_posterior, initial_params, proposal_distribution)

    # Initialize chain storage
    chain = np.zeros((steps - burn_in // thinning, initial_params.shape[0]))
    current_params = initial_params.copy()
    accepted = 0

    # Metropolis-Hastings algorithm
    for i in range(steps):
        # Propose new parameters
        proposed_params = proposal_distribution(current_params, 1.0)

        # Normalize if required
        if normalizer is not None:
            proposed_params = normalizer(proposed_params)
            current_params = normalizer(current_params)

        # Compute acceptance ratio
        log_ratio = (log_posterior(proposed_params) -
                     log_posterior(current_params))

        # Accept or reject
        if np.log(np.random.rand()) < log_ratio:
            current_params = proposed_params.copy()
            accepted += 1

        # Store parameters with thinning
        if i >= burn_in and (i - burn_in) % thinning == 0:
            chain[(i - burn_in) // thinning] = current_params.copy()

    # Calculate acceptance rate
    acceptance_rate = accepted / steps

    # Compute metrics if data is available (example with dummy data)
    metrics = _compute_metrics(chain, metric, custom_metric)

    return {
        'result': chain,
        'metrics': metrics,
        'params_used': {
            'steps': steps,
            'burn_in': burn_in,
            'thinning': thinning,
            'acceptance_rate': acceptance_rate
        },
        'warnings': []
    }

def _validate_inputs(
    log_posterior: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    proposal_distribution: Callable[[np.ndarray, float], np.ndarray]
) -> None:
    """Validate input functions and parameters."""
    if not callable(log_posterior):
        raise ValueError("log_posterior must be a callable function")
    if not isinstance(initial_params, np.ndarray):
        raise ValueError("initial_params must be a numpy array")
    if not callable(proposal_distribution):
        raise ValueError("proposal_distribution must be a callable function")

def _compute_metrics(
    chain: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute metrics from the MCMC chain."""
    metrics = {}

    if custom_metric is not None:
        # Example with dummy data - in practice, you would use actual data
        metrics['custom_metric'] = custom_metric(chain[-1], np.zeros_like(chain[-1]))
    else:
        if metric == 'mse':
            # Example MSE calculation
            metrics['mse'] = np.mean((chain[-1] - np.zeros_like(chain[-1]))**2)
        elif metric == 'mae':
            # Example MAE calculation
            metrics['mae'] = np.mean(np.abs(chain[-1] - np.zeros_like(chain[-1])))
        elif metric == 'r2':
            # Example R2 calculation
            ss_res = np.sum((chain[-1] - np.zeros_like(chain[-1]))**2)
            ss_tot = np.sum((np.zeros_like(chain[-1]) - np.mean(np.zeros_like(chain[-1])))**2)
            metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

# Example usage:
"""
def example_log_posterior(params):
    # Example log posterior (Gaussian likelihood with flat prior)
    return -0.5 * np.sum(params**2)

def example_proposal(current, step_size):
    # Example proposal distribution (Gaussian random walk)
    return current + np.random.normal(0, step_size, size=current.shape)

chain_result = metropolis_hastings_fit(
    log_posterior=example_log_posterior,
    initial_params=np.zeros(10),
    proposal_distribution=example_proposal,
    steps=5000,
    burn_in=1000,
    thinning=2
)
"""

################################################################################
# gibbs_sampling
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def gibbs_sampling_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 1000,
    burn_in: int = 500,
    thin: int = 1,
    prior_mean: Optional[np.ndarray] = None,
    prior_cov: np.ndarray = np.eye(1),
    likelihood_var: float = 1.0,
    random_state: Optional[int] = None,
    metric: str = 'mse',
    normalize: bool = True
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Perform Gibbs sampling for Bayesian linear regression.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    n_iter : int, optional
        Number of iterations to run the Gibbs sampler.
    burn_in : int, optional
        Number of initial iterations to discard as burn-in.
    thin : int, optional
        Thinning interval for the samples.
    prior_mean : np.ndarray, optional
        Prior mean for the regression coefficients. If None, uses zero mean.
    prior_cov : np.ndarray, optional
        Prior covariance matrix for the regression coefficients.
    likelihood_var : float, optional
        Variance of the likelihood (noise variance).
    random_state : int, optional
        Random seed for reproducibility.
    metric : str, optional
        Metric to compute ('mse', 'mae', 'r2').
    normalize : bool, optional
        Whether to normalize the features.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - "result": Array of sampled coefficients.
        - "metrics": Computed metrics.
        - "params_used": Parameters used in the fitting process.
        - "warnings": Any warnings encountered during fitting.

    Example
    -------
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randn(100)
    >>> result = gibbs_sampling_fit(X, y, n_iter=2000)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if required
    X_norm, y_norm = _normalize_data(X, y) if normalize else (X, y)

    # Initialize parameters
    n_samples, n_features = X_norm.shape
    if prior_mean is None:
        prior_mean = np.zeros(n_features)

    # Initialize storage for samples
    beta_samples = np.zeros((n_iter - burn_in // thin, n_features))

    # Set random seed
    rng = np.random.RandomState(random_state)

    # Initialize beta with prior mean
    beta = prior_mean.copy()

    for i in range(n_iter):
        # Sample beta given y and sigma^2
        beta = _sample_beta(X_norm, y_norm, prior_mean, prior_cov, likelihood_var, rng)

        # Sample sigma^2 given y and beta
        likelihood_var = _sample_likelihood_var(y_norm, X_norm @ beta, rng)

        # Store samples after burn-in and thinning
        if i >= burn_in and (i - burn_in) % thin == 0:
            idx = (i - burn_in) // thin
            beta_samples[idx, :] = beta

    # Compute metrics
    metrics = _compute_metrics(X_norm, y_norm, beta_samples, metric)

    return {
        "result": beta_samples,
        "metrics": metrics,
        "params_used": {
            "n_iter": n_iter,
            "burn_in": burn_in,
            "thin": thin,
            "prior_mean": prior_mean,
            "prior_cov": prior_cov,
            "likelihood_var": likelihood_var,
            "random_state": random_state,
            "metric": metric,
            "normalize": normalize
        },
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X and y must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("X and y must not contain infinite values.")

def _normalize_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """Normalize features and target."""
    X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y_norm = (y - np.mean(y)) / np.std(y)
    return X_norm, y_norm

def _sample_beta(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    likelihood_var: float,
    rng: np.random.RandomState
) -> np.ndarray:
    """Sample beta from its posterior distribution."""
    n_samples, n_features = X.shape
    post_cov = np.linalg.inv(np.linalg.inv(prior_cov) + (X.T @ X) / likelihood_var)
    post_mean = post_cov @ (np.linalg.inv(prior_cov) @ prior_mean + (X.T @ y) / likelihood_var)
    beta = rng.multivariate_normal(post_mean, post_cov)
    return beta

def _sample_likelihood_var(
    y: np.ndarray,
    y_pred: np.ndarray,
    rng: np.random.RandomState
) -> float:
    """Sample likelihood variance from its posterior distribution."""
    n_samples = y.shape[0]
    sigma_sq = np.sum((y - y_pred) ** 2) / rng.chisquare(n_samples)
    return sigma_sq

def _compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    beta_samples: np.ndarray,
    metric: str
) -> Dict[str, float]:
    """Compute metrics for the sampled coefficients."""
    n_samples = X.shape[0]
    y_pred_samples = np.dot(X, beta_samples.T)
    metrics = {}

    if metric == 'mse':
        y_pred_mean = np.mean(y_pred_samples, axis=1)
        metrics['mse'] = _compute_mse(y, y_pred_mean)
    elif metric == 'mae':
        y_pred_mean = np.mean(y_pred_samples, axis=1)
        metrics['mae'] = _compute_mae(y, y_pred_mean)
    elif metric == 'r2':
        y_pred_mean = np.mean(y_pred_samples, axis=1)
        metrics['r2'] = _compute_r2(y, y_pred_mean)

    return metrics

################################################################################
# hamiltonian_monte_carlo
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int = 1000,
    step_size: float = 0.01,
    n_steps: int = 10
) -> None:
    """
    Validate input data and parameters for Hamiltonian Monte Carlo.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    n_samples : int
        Number of samples to generate.
    step_size : float
        Step size for the leapfrog integrator.
    n_steps : int
        Number of steps in the leapfrog integrator.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

def compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    log_posterior: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """
    Compute the gradient of the log posterior distribution.

    Parameters
    ----------
    X : np.ndarray
        Design matrix.
    y : np.ndarray
        Target vector.
    params : np.ndarray
        Current parameters.
    log_posterior : Callable[[np.ndarray, np.ndarray], float]
        Function to compute the log posterior.

    Returns
    ------
    np.ndarray
        Gradient of the log posterior.
    """
    epsilon = 1e-8
    gradient = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += epsilon
        params_minus = params.copy()
        params_minus[i] -= epsilon
        gradient[i] = (log_posterior(params_plus, y) - log_posterior(params_minus, y)) / (2 * epsilon)
    return gradient

def leapfrog_integrator(
    current_params: np.ndarray,
    current_momentum: np.ndarray,
    step_size: float,
    n_steps: int,
    gradient_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> tuple:
    """
    Perform the leapfrog integration step.

    Parameters
    ----------
    current_params : np.ndarray
        Current parameters.
    current_momentum : np.ndarray
        Current momentum.
    step_size : float
        Step size.
    n_steps : int
        Number of steps.
    gradient_func : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Function to compute the gradient.

    Returns
    ------
    tuple
        New parameters and momentum.
    """
    params = current_params.copy()
    momentum = current_momentum.copy()

    for _ in range(n_steps):
        momentum -= step_size * gradient_func(params, None) / 2
        params += step_size * momentum
        momentum -= step_size * gradient_func(params, None) / 2

    return params, momentum

def hamiltonian_monte_carlo_fit(
    X: np.ndarray,
    y: np.ndarray,
    initial_params: Optional[np.ndarray] = None,
    n_samples: int = 1000,
    step_size: float = 0.01,
    n_steps: int = 10,
    log_posterior: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    normalize: bool = True
) -> Dict[str, Any]:
    """
    Perform Hamiltonian Monte Carlo sampling for Bayesian regression.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    initial_params : Optional[np.ndarray]
        Initial parameters. If None, random initialization.
    n_samples : int
        Number of samples to generate.
    step_size : float
        Step size for the leapfrog integrator.
    n_steps : int
        Number of steps in the leapfrog integrator.
    log_posterior : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Function to compute the log posterior. If None, default is used.
    normalize : bool
        Whether to normalize the data.

    Returns
    ------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    validate_inputs(X, y, n_samples, step_size, n_steps)

    if normalize:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y = (y - np.mean(y)) / np.std(y)

    n_features = X.shape[1]
    if initial_params is None:
        initial_params = np.random.randn(n_features)

    if log_posterior is None:
        def default_log_posterior(params: np.ndarray, y_data: np.ndarray) -> float:
            residuals = y_data - X @ params
            log_likelihood = -0.5 * np.sum(residuals**2)
            log_prior = -0.5 * np.sum(params**2)  # Assuming Gaussian prior
            return log_likelihood + log_prior

        log_posterior = default_log_posterior

    samples = []
    current_params = initial_params.copy()
    current_momentum = np.random.randn(n_features)

    for _ in range(n_samples):
        # Propose new parameters and momentum
        proposed_params, proposed_momentum = leapfrog_integrator(
            current_params,
            current_momentum,
            step_size,
            n_steps,
            lambda params, _: compute_gradient(X, y, params, log_posterior)
        )

        # Metropolis-Hastings acceptance step
        current_log_posterior = log_posterior(current_params, y)
        proposed_log_posterior = log_posterior(proposed_params, y)

        log_acceptance_ratio = proposed_log_posterior - current_log_posterior
        if np.log(np.random.rand()) < log_acceptance_ratio:
            current_params = proposed_params
            current_momentum = proposed_momentum

        samples.append(current_params.copy())

    # Calculate metrics
    samples_array = np.array(samples)
    mean_params = np.mean(samples_array, axis=0)
    std_params = np.std(samples_array, axis=0)

    # Example metric: Mean Squared Error
    y_pred = X @ mean_params
    mse = np.mean((y - y_pred)**2)

    return {
        "result": samples_array,
        "metrics": {"mse": mse},
        "params_used": {
            "n_samples": n_samples,
            "step_size": step_size,
            "n_steps": n_steps,
            "normalize": normalize
        },
        "warnings": []
    }

################################################################################
# variational_inference
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def variational_inference_fit(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: Optional[np.ndarray] = None,
    prior_cov: Optional[np.ndarray] = None,
    likelihood_var: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    l1_ratio: float = 0.5,
    verbose: bool = False
) -> Dict:
    """
    Perform variational inference for Bayesian regression.

    Parameters:
    -----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    prior_mean : Optional[np.ndarray]
        Prior mean for the weights. If None, zero vector is used.
    prior_cov : Optional[np.ndarray]
        Prior covariance matrix for the weights. If None, identity matrix is used.
    likelihood_var : float
        Variance of the likelihood (noise variance)
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence
    normalize : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : Union[str, Callable]
        Metric to evaluate ('mse', 'mae', 'r2', custom callable)
    solver : str
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    regularization : Optional[str]
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    l1_ratio : float
        Mixing parameter for elastic net (0 <= l1_ratio <= 1)
    verbose : bool
        Whether to print progress information

    Returns:
    --------
    Dict containing:
        - 'result': Optimization result
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in the optimization
        - 'warnings': Any warnings encountered

    Example:
    --------
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randn(100)
    >>> result = variational_inference_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize parameters
    params = {
        'prior_mean': prior_mean if prior_mean is not None else np.zeros(X.shape[1]),
        'prior_cov': prior_cov if prior_cov is not None else np.eye(X.shape[1]),
        'likelihood_var': likelihood_var,
        'max_iter': max_iter,
        'tol': tol,
        'normalize': normalize,
        'metric': metric,
        'solver': solver,
        'regularization': regularization,
        'l1_ratio': l1_ratio
    }

    # Normalize data if needed
    X_norm, y_norm = _normalize_data(X, y, params['normalize'])

    # Initialize variables
    n_samples, n_features = X_norm.shape
    posterior_mean = np.zeros(n_features)
    posterior_cov = np.eye(n_features)

    # Choose solver
    if params['solver'] == 'closed_form':
        posterior_mean, posterior_cov = _closed_form_solution(X_norm, y_norm, params)
    elif params['solver'] == 'gradient_descent':
        posterior_mean, posterior_cov = _gradient_descent(
            X_norm, y_norm, params, max_iter, tol
        )
    elif params['solver'] == 'newton':
        posterior_mean, posterior_cov = _newton_method(
            X_norm, y_norm, params, max_iter, tol
        )
    elif params['solver'] == 'coordinate_descent':
        posterior_mean, posterior_cov = _coordinate_descent(
            X_norm, y_norm, params, max_iter, tol
        )
    else:
        raise ValueError(f"Unknown solver: {params['solver']}")

    # Compute metrics
    metrics = _compute_metrics(
        X_norm, y_norm, posterior_mean, params['metric']
    )

    # Prepare output
    result = {
        'posterior_mean': posterior_mean,
        'posterior_cov': posterior_cov
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': warnings
    }

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
    method: str
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
        X_norm = (X - np.median(X, axis=0)) / (
            np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        )
        y_norm = (y - np.median(y)) / (
            np.percentile(y, 75) - np.percentile(y, 25)
        )
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return X_norm, y_norm

def _closed_form_solution(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict
) -> tuple[np.ndarray, np.ndarray]:
    """Closed form solution for variational inference."""
    n_samples, n_features = X.shape
    prior_mean = params['prior_mean']
    prior_cov = params['prior_cov']
    likelihood_var = params['likelihood_var']

    # Compute posterior covariance
    posterior_cov = np.linalg.inv(
        np.linalg.inv(prior_cov) + (1/likelihood_var) * X.T @ X
    )

    # Compute posterior mean
    posterior_mean = posterior_cov @ (
        (1/likelihood_var) * X.T @ y + np.linalg.inv(prior_cov) @ prior_mean
    )

    return posterior_mean, posterior_cov

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict,
    max_iter: int,
    tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient descent solver for variational inference."""
    n_samples, n_features = X.shape
    posterior_mean = np.zeros(n_features)
    posterior_cov = np.eye(n_features)

    for _ in range(max_iter):
        # Update posterior mean
        grad = (1/params['likelihood_var']) * X.T @ (X @ posterior_mean - y)
        if params['regularization'] == 'l1':
            grad += np.sign(posterior_mean)
        elif params['regularization'] == 'l2':
            grad += posterior_mean
        elif params['regularization'] == 'elasticnet':
            grad += params['l1_ratio'] * np.sign(posterior_mean) + (
                1 - params['l1_ratio']
            ) * posterior_mean

        # Update rule
        posterior_mean -= 0.01 * grad

        # Check convergence
        if np.linalg.norm(grad) < tol:
            break

    return posterior_mean, posterior_cov

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict,
    max_iter: int,
    tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Newton method solver for variational inference."""
    n_samples, n_features = X.shape
    posterior_mean = np.zeros(n_features)
    posterior_cov = np.eye(n_features)

    for _ in range(max_iter):
        # Compute gradient
        grad = (1/params['likelihood_var']) * X.T @ (X @ posterior_mean - y)

        # Compute Hessian
        hess = (1/params['likelihood_var']) * X.T @ X

        # Update rule
        posterior_mean -= np.linalg.inv(hess) @ grad

        # Check convergence
        if np.linalg.norm(grad) < tol:
            break

    return posterior_mean, posterior_cov

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict,
    max_iter: int,
    tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate descent solver for variational inference."""
    n_samples, n_features = X.shape
    posterior_mean = np.zeros(n_features)
    posterior_cov = np.eye(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            # Compute residual
            r = y - X @ posterior_mean + X[:, j] * posterior_mean[j]

            # Update coefficient
            x_j = X[:, j]
            numerator = x_j.T @ r
            denominator = x_j.T @ x_j

            if params['regularization'] == 'l1':
                posterior_mean[j] = np.sign(numerator) * np.maximum(
                    0, abs(numerator) - params['l1_ratio']
                ) / denominator
            elif params['regularization'] == 'l2':
                posterior_mean[j] = numerator / (denominator + 1)
            elif params['regularization'] == 'elasticnet':
                posterior_mean[j] = np.sign(numerator) * np.maximum(
                    0, abs(numerator) - params['l1_ratio']
                ) / (denominator + (1 - params['l1_ratio']))
            else:
                posterior_mean[j] = numerator / denominator

        # Check convergence
        if np.linalg.norm(X @ posterior_mean - y) < tol:
            break

    return posterior_mean, posterior_cov

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    posterior_mean: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Compute evaluation metrics."""
    y_pred = X @ posterior_mean
    metrics = {}

    if metric == 'mse' or isinstance(metric, Callable):
        mse = np.mean((y - y_pred) ** 2)
        if isinstance(metric, str):
            metrics['mse'] = mse
    elif metric == 'mae':
        mae = np.mean(np.abs(y - y_pred))
        metrics['mae'] = mae
    elif metric == 'r2':
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        metrics['r2'] = r2
    elif callable(metric):
        metrics['custom_metric'] = metric(y, y_pred)

    return metrics

################################################################################
# conjugate_priors
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def conjugate_priors_fit(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: Optional[np.ndarray] = None,
    prior_precision: Optional[np.ndarray] = None,
    likelihood_variance: float = 1.0,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit Bayesian regression with conjugate priors.

    Parameters:
    -----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    prior_mean : Optional[np.ndarray]
        Prior mean for the coefficients
    prior_precision : Optional[np.ndarray]
        Prior precision matrix (inverse of covariance)
    likelihood_variance : float
        Variance of the likelihood
    normalize : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : Union[str, Callable]
        Metric to evaluate ('mse', 'mae', 'r2') or custom callable
    solver : str
        Solver method ('closed_form', 'gradient_descent')
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    custom_metric : Optional[Callable]
        Custom metric function

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = conjugate_priors_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if needed
    X_norm, y_norm = _normalize_data(X, y, normalize)

    # Set default priors if not provided
    prior_mean = _set_default_prior_mean(X_norm.shape[1], prior_mean)
    prior_precision = _set_default_prior_precision(X_norm.shape[1], prior_precision)

    # Choose solver
    if solver == 'closed_form':
        coefficients = _closed_form_solver(X_norm, y_norm, prior_mean, prior_precision, likelihood_variance)
    elif solver == 'gradient_descent':
        coefficients = _gradient_descent_solver(
            X_norm, y_norm, prior_mean, prior_precision,
            likelihood_variance, tol, max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    if isinstance(metric, str):
        metrics = _calculate_metrics(X_norm, y_norm, coefficients, metric)
    else:
        metrics = {'custom': metric(X_norm @ coefficients, y_norm)}

    # Add custom metric if provided
    if custom_metric is not None:
        metrics['custom'] = custom_metric(X_norm @ coefficients, y_norm)

    return {
        'result': {
            'coefficients': coefficients,
            'prior_mean': prior_mean,
            'prior_precision': prior_precision
        },
        'metrics': metrics,
        'params_used': {
            'normalize': normalize,
            'metric': metric,
            'solver': solver
        },
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

def _normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'standard'
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize data according to specified method."""
    if method == 'none':
        return X, y
    elif method == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # Avoid division by zero
        X_norm = (X - X_mean) / X_std
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std == 0:
            y_std = 1
        y_norm = (y - y_mean) / y_std
    elif method == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        y_min = np.min(y)
        y_max = np.max(y)
        y_norm = (y - y_min) / (y_max - y_min + 1e-8)
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

def _set_default_prior_mean(n_features: int, prior_mean: Optional[np.ndarray]) -> np.ndarray:
    """Set default prior mean if not provided."""
    if prior_mean is None:
        return np.zeros(n_features)
    return prior_mean

def _set_default_prior_precision(
    n_features: int,
    prior_precision: Optional[np.ndarray]
) -> np.ndarray:
    """Set default prior precision if not provided."""
    if prior_precision is None:
        return np.eye(n_features)
    return prior_precision

def _closed_form_solver(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    likelihood_variance: float
) -> np.ndarray:
    """Closed form solution for conjugate priors."""
    posterior_precision = prior_precision + (X.T @ X) / likelihood_variance
    posterior_mean = np.linalg.solve(
        posterior_precision,
        prior_precision @ prior_mean + (X.T @ y) / likelihood_variance
    )
    return posterior_mean

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    likelihood_variance: float,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent solver for conjugate priors."""
    n_features = X.shape[1]
    coefficients = prior_mean.copy()
    learning_rate = 0.01
    prev_loss = np.inf

    for _ in range(max_iter):
        # Compute gradient
        gradient = (
            -prior_precision @ coefficients +
            (X.T @ X @ coefficients - X.T @ y) / likelihood_variance
        )

        # Update coefficients
        coefficients -= learning_rate * gradient

        # Check convergence
        current_loss = np.linalg.norm(gradient)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return coefficients

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    metric_type: str
) -> Dict[str, float]:
    """Calculate specified metrics."""
    y_pred = X @ coefficients
    metrics = {}

    if metric_type == 'mse':
        metrics['mse'] = np.mean((y_pred - y) ** 2)
    elif metric_type == 'mae':
        metrics['mae'] = np.mean(np.abs(y_pred - y))
    elif metric_type == 'r2':
        ss_res = np.sum((y_pred - y) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    else:
        raise ValueError(f"Unknown metric: {metric_type}")

    return metrics

################################################################################
# bayesian_hyperparameter_tuning
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def bayesian_hyperparameter_tuning_fit(
    X: np.ndarray,
    y: np.ndarray,
    objective_function: Callable[[np.ndarray, Dict[str, Any]], float],
    prior_distribution: Callable[[], np.ndarray],
    n_iterations: int = 100,
    n_initial_points: int = 5,
    acquisition_function: Callable[[np.ndarray, np.ndarray], float] = None,
    bounds: Optional[Dict[str, tuple]] = None,
    normalize: str = 'standard',
    metric: Callable[[np.ndarray, np.ndarray], float] = None,
    solver: str = 'gradient_descent',
    regularization: str = 'none',
    tol: float = 1e-4,
    max_evals: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform Bayesian hyperparameter tuning for regression models.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    objective_function : Callable[[np.ndarray, Dict[str, Any]], float]
        Function to optimize. Takes hyperparameters and returns a score.
    prior_distribution : Callable[[], np.ndarray]
        Function returning initial hyperparameters from prior distribution.
    n_iterations : int, optional
        Number of iterations for optimization (default: 100).
    n_initial_points : int, optional
        Number of initial points to sample (default: 5).
    acquisition_function : Callable[[np.ndarray, np.ndarray], float], optional
        Acquisition function for Bayesian optimization (default: None).
    bounds : Dict[str, tuple], optional
        Dictionary of hyperparameter bounds (default: None).
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard').
    metric : Callable[[np.ndarray, np.ndarray], float], optional
        Custom metric function (default: None).
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default: 'gradient_descent').
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet') (default: 'none').
    tol : float, optional
        Tolerance for convergence (default: 1e-4).
    max_evals : int, optional
        Maximum number of evaluations (default: 1000).
    random_state : int, optional
        Random seed for reproducibility (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _normalize_data(X, method=normalize)

    # Initialize results dictionary
    results = {
        'result': None,
        'metrics': {},
        'params_used': {},
        'warnings': []
    }

    # Perform Bayesian hyperparameter tuning
    best_params, best_score = _bayesian_optimization(
        X_normalized,
        y,
        objective_function,
        prior_distribution,
        n_iterations,
        n_initial_points,
        acquisition_function,
        bounds,
        metric,
        solver,
        regularization,
        tol,
        max_evals,
        random_state
    )

    # Store results
    results['result'] = best_score
    results['params_used'] = best_params

    # Calculate metrics if custom metric is provided
    if metric is not None:
        results['metrics']['custom_metric'] = metric(X_normalized, y)

    return results

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

def _bayesian_optimization(
    X: np.ndarray,
    y: np.ndarray,
    objective_function: Callable[[np.ndarray, Dict[str, Any]], float],
    prior_distribution: Callable[[], np.ndarray],
    n_iterations: int,
    n_initial_points: int,
    acquisition_function: Callable[[np.ndarray, np.ndarray], float],
    bounds: Dict[str, tuple],
    metric: Callable[[np.ndarray, np.ndarray], float],
    solver: str,
    regularization: str,
    tol: float,
    max_evals: int,
    random_state: Optional[int]
) -> tuple:
    """Perform Bayesian optimization."""
    if random_state is not None:
        np.random.seed(random_state)

    # Initialize with random points
    initial_params = [prior_distribution() for _ in range(n_initial_points)]
    initial_scores = [objective_function(X, {'params': params}) for params in initial_params]

    best_score = min(initial_scores)
    best_params = initial_params[np.argmin(initial_scores)]

    # Bayesian optimization loop
    for _ in range(n_iterations):
        if acquisition_function is None:
            # Default to expected improvement
            acquisition_function = _expected_improvement

        next_params = _suggest_next_point(
            X,
            y,
            objective_function,
            prior_distribution,
            acquisition_function,
            bounds,
            metric,
            solver,
            regularization
        )

        score = objective_function(X, {'params': next_params})

        if score < best_score:
            best_score = score
            best_params = next_params

    return best_params, best_score

def _suggest_next_point(
    X: np.ndarray,
    y: np.ndarray,
    objective_function: Callable[[np.ndarray, Dict[str, Any]], float],
    prior_distribution: Callable[[], np.ndarray],
    acquisition_function: Callable[[np.ndarray, np.ndarray], float],
    bounds: Dict[str, tuple],
    metric: Callable[[np.ndarray, np.ndarray], float],
    solver: str,
    regularization: str
) -> np.ndarray:
    """Suggest the next point to evaluate."""
    # This is a placeholder for the actual implementation
    return prior_distribution()

def _expected_improvement(
    X: np.ndarray,
    y: np.ndarray
) -> float:
    """Expected improvement acquisition function."""
    # This is a placeholder for the actual implementation
    return 0.0

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.random.rand(100)

def objective_function(X: np.ndarray, params: Dict[str, Any]) -> float:
    return np.mean((X @ params['params'] - y) ** 2)

def prior_distribution() -> np.ndarray:
    return np.random.rand(5)

result = bayesian_hyperparameter_tuning_fit(
    X, y,
    objective_function=objective_function,
    prior_distribution=prior_distribution
)
"""

################################################################################
# bayesian_model_selection
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def bayesian_model_selection_fit(
    X: np.ndarray,
    y: np.ndarray,
    models: List[Dict],
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict:
    """
    Perform Bayesian model selection by fitting multiple models and selecting the best one.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    models : List[Dict]
        List of model configurations. Each dict should contain:
        - 'prior': prior distribution parameters
        - 'likelihood': likelihood function parameters
    normalizer : Optional[Callable]
        Function to normalize the data. If None, no normalization is applied.
    metric : Union[str, Callable]
        Metric to evaluate models. Can be 'mse', 'mae', 'r2' or a custom callable.
    solver : str
        Solver to use for model fitting. Options: 'closed_form', 'gradient_descent'.
    regularization : Optional[str]
        Regularization type. Options: None, 'l1', 'l2'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': best model results
        - 'metrics': evaluation metrics for all models
        - 'params_used': parameters used for the best model
        - 'warnings': any warnings encountered

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> models = [{'prior': 'normal', 'likelihood': 'gaussian'}, {'prior': 'laplace', 'likelihood': 'gaussian'}]
    >>> result = bayesian_model_selection_fit(X, y, models)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_norm = _apply_normalization(X, normalizer)

    # Initialize results dictionary
    results = {
        'result': None,
        'metrics': {},
        'params_used': {},
        'warnings': []
    }

    # Evaluate each model
    for i, model in enumerate(models):
        try:
            # Fit the model
            params = _fit_model(
                X_norm, y,
                model['prior'], model['likelihood'],
                solver=solver,
                regularization=regularization,
                tol=tol,
                max_iter=max_iter,
                random_state=random_state
            )

            # Calculate metrics
            metrics = _calculate_metrics(X_norm, y, params, metric)

            # Store results
            results['metrics'][f'model_{i}'] = metrics
            results['params_used'][f'model_{i}'] = params

        except Exception as e:
            results['warnings'].append(f"Model {i} failed: {str(e)}")
            continue

    # Select the best model
    if results['metrics']:
        best_model = max(results['metrics'].items(), key=lambda x: x[1]['score'])
        results['result'] = {
            'model': best_model[0],
            'metrics': best_model[1]
        }

    return results

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data."""
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
    """Apply normalization to the data."""
    if normalizer is not None:
        return normalizer(X)
    return X

def _fit_model(
    X: np.ndarray,
    y: np.ndarray,
    prior: Dict,
    likelihood: Dict,
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict:
    """Fit a Bayesian regression model."""
    if solver == 'closed_form':
        return _fit_closed_form(X, y, prior, likelihood, regularization)
    elif solver == 'gradient_descent':
        return _fit_gradient_descent(X, y, prior, likelihood, regularization, tol, max_iter, random_state)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _fit_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    prior: Dict,
    likelihood: Dict,
    regularization: Optional[str]
) -> Dict:
    """Fit model using closed-form solution."""
    # Implement closed-form solution
    return {'coefficients': np.random.rand(X.shape[1])}  # Placeholder

def _fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    prior: Dict,
    likelihood: Dict,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> Dict:
    """Fit model using gradient descent."""
    # Implement gradient descent
    return {'coefficients': np.random.rand(X.shape[1])}  # Placeholder

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate evaluation metrics."""
    y_pred = _predict(X, params)

    if isinstance(metric, str):
        if metric == 'mse':
            score = np.mean((y - y_pred) ** 2)
        elif metric == 'mae':
            score = np.mean(np.abs(y - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            score = 1 - (ss_res / ss_tot)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        score = metric(y, y_pred)

    return {
        'score': score,
        'mse': np.mean((y - y_pred) ** 2),
        'mae': np.mean(np.abs(y - y_pred)),
        'r2': 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
    }

def _predict(X: np.ndarray, params: Dict) -> np.ndarray:
    """Make predictions using the fitted model."""
    return X @ params['coefficients']  # Placeholder

################################################################################
# bayesian_neural_networks
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def bayesian_neural_networks_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_hidden_layers: int = 1,
    hidden_units: Union[int, List[int]] = 32,
    activation: str = 'relu',
    prior_mean: float = 0.0,
    prior_std: float = 1.0,
    likelihood_std: float = 1.0,
    optimizer: str = 'adam',
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
    batch_size: Optional[int] = None,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict:
    """
    Fit a Bayesian Neural Network model.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, n_outputs).
    n_hidden_layers : int, optional
        Number of hidden layers in the network.
    hidden_units : Union[int, List[int]], optional
        Number of units in each hidden layer. If int, same number for all layers.
    activation : str, optional
        Activation function for hidden layers ('relu', 'tanh', 'sigmoid').
    prior_mean : float, optional
        Mean of the Gaussian prior for weights.
    prior_std : float, optional
        Standard deviation of the Gaussian prior for weights.
    likelihood_std : float, optional
        Standard deviation of the Gaussian likelihood.
    optimizer : str, optional
        Optimization algorithm ('adam', 'sgd').
    learning_rate : float, optional
        Learning rate for the optimizer.
    n_iterations : int, optional
        Number of training iterations.
    batch_size : Optional[int], optional
        Batch size for mini-batch optimization. If None, uses full batch.
    tol : float, optional
        Tolerance for stopping criterion.
    random_state : Optional[int], optional
        Random seed for reproducibility.
    verbose : bool, optional
        Whether to print progress information.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Fitted model parameters.
        - 'metrics': Training metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = bayesian_neural_networks_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize random state
    if random_state is not None:
        np.random.seed(random_state)

    # Set default hidden units if single integer provided
    if isinstance(hidden_units, int):
        hidden_units = [hidden_units] * n_hidden_layers

    # Initialize model parameters
    params = _initialize_parameters(X.shape[1], y.shape[1],
                                  n_hidden_layers, hidden_units,
                                  prior_mean, prior_std)

    # Prepare optimizer
    if optimizer == 'adam':
        optimizer_func = _adam_optimizer
    elif optimizer == 'sgd':
        optimizer_func = _sgd_optimizer
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # Training loop
    metrics = _train_bnn(X, y, params, activation,
                        likelihood_std, optimizer_func,
                        learning_rate, n_iterations,
                        batch_size, tol, verbose)

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'n_hidden_layers': n_hidden_layers,
            'hidden_units': hidden_units,
            'activation': activation,
            'prior_mean': prior_mean,
            'prior_std': prior_std,
            'likelihood_std': likelihood_std,
            'optimizer': optimizer,
            'learning_rate': learning_rate,
            'n_iterations': n_iterations,
            'batch_size': batch_size,
            'tol': tol
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim not in (1, 2):
        raise ValueError("y must be a 1D or 2D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _initialize_parameters(
    n_features: int,
    n_outputs: int,
    n_hidden_layers: int,
    hidden_units: List[int],
    prior_mean: float,
    prior_std: float
) -> Dict:
    """Initialize model parameters with Gaussian priors."""
    params = {}

    # Input to first hidden layer
    params['W1'] = np.random.normal(
        loc=prior_mean,
        scale=prior_std,
        size=(n_features, hidden_units[0])
    )

    # Hidden layers
    for i in range(1, n_hidden_layers):
        params[f'W{i+1}'] = np.random.normal(
            loc=prior_mean,
            scale=prior_std,
            size=(hidden_units[i-1], hidden_units[i])
        )

    # Last layer to output
    params[f'W{n_hidden_layers+1}'] = np.random.normal(
        loc=prior_mean,
        scale=prior_std,
        size=(hidden_units[-1], n_outputs)
    )

    # Biases
    for i in range(n_hidden_layers + 1):
        params[f'b{i+1}'] = np.zeros(hidden_units[i] if i < n_hidden_layers else n_outputs)

    return params

def _forward_pass(
    X: np.ndarray,
    params: Dict,
    activation: str
) -> np.ndarray:
    """Perform forward pass through the network."""
    layer_input = X
    for i in range(1, len(params) // 2 + 1):
        W = params[f'W{i}']
        b = params[f'b{i}']

        if i == len(params) // 2:
            # Output layer (linear activation)
            layer_output = np.dot(layer_input, W) + b
        else:
            # Hidden layer with specified activation
            if activation == 'relu':
                layer_output = np.maximum(0, np.dot(layer_input, W) + b)
            elif activation == 'tanh':
                layer_output = np.tanh(np.dot(layer_input, W) + b)
            elif activation == 'sigmoid':
                layer_output = 1 / (1 + np.exp(-(np.dot(layer_input, W) + b)))
            else:
                raise ValueError(f"Unknown activation function: {activation}")

        layer_input = layer_output

    return layer_output

def _compute_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    likelihood_std: float
) -> float:
    """Compute the negative log likelihood loss."""
    n_samples = y_true.shape[0]
    return 0.5 * (n_samples * np.log(2 * np.pi) +
                 n_samples * np.log(likelihood_std**2) +
                 np.sum((y_true - y_pred)**2) / (likelihood_std**2))

def _adam_optimizer(
    params: Dict,
    grads: Dict,
    learning_rate: float
) -> None:
    """Adam optimizer implementation."""
    if not hasattr(_adam_optimizer, 'm'):
        _adam_optimizer.m = {k: np.zeros_like(v) for k, v in params.items()}
        _adam_optimizer.v = {k: np.zeros_like(v) for k, v in params.items()}
        _adam_optimizer.t = 0

    _adam_optimizer.t += 1
    beta1, beta2 = 0.9, 0.999
    epsilon = 1e-8

    for key in params:
        _adam_optimizer.m[key] = beta1 * _adam_optimizer.m[key] + (1 - beta1) * grads[key]
        _adam_optimizer.v[key] = beta2 * _adam_optimizer.v[key] + (1 - beta2) * (grads[key]**2)

        m_hat = _adam_optimizer.m[key] / (1 - beta1**_adam_optimizer.t)
        v_hat = _adam_optimizer.v[key] / (1 - beta2**_adam_optimizer.t)

        params[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

def _sgd_optimizer(
    params: Dict,
    grads: Dict,
    learning_rate: float
) -> None:
    """Stochastic Gradient Descent optimizer implementation."""
    for key in params:
        params[key] -= learning_rate * grads[key]

def _compute_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    params: Dict,
    activation: str
) -> Dict:
    """Compute gradients of the loss with respect to parameters."""
    # Forward pass
    layer_activations = [X]
    for i in range(1, len(params) // 2 + 1):
        W = params[f'W{i}']
        b = params[f'b{i}']

        if i == len(params) // 2:
            # Output layer
            z = np.dot(layer_activations[-1], W) + b
        else:
            # Hidden layer
            if activation == 'relu':
                z = np.maximum(0, np.dot(layer_activations[-1], W) + b)
            elif activation == 'tanh':
                z = np.tanh(np.dot(layer_activations[-1], W) + b)
            elif activation == 'sigmoid':
                z = 1 / (1 + np.exp(-(np.dot(layer_activations[-1], W) + b)))
            else:
                raise ValueError(f"Unknown activation function: {activation}")

        layer_activations.append(z)

    y_pred = layer_activations[-1]

    # Backward pass
    grads = {}
    m = y_true.shape[0]

    # Output layer gradients
    delta = (y_pred - y_true) / m
    grads[f'W{len(params)//2}'] = np.dot(layer_activations[-2].T, delta)
    grads[f'b{len(params)//2}'] = np.sum(delta, axis=0)

    # Hidden layers gradients
    for i in range(len(params)//2 - 1, 0, -1):
        delta = np.dot(delta, params[f'W{i+1}'].T)

        if activation == 'relu':
            delta = delta * (layer_activations[i] > 0)
        elif activation == 'tanh':
            delta = delta * (1 - layer_activations[i]**2)
        elif activation == 'sigmoid':
            delta = delta * (layer_activations[i] * (1 - layer_activations[i]))

        grads[f'W{i}'] = np.dot(layer_activations[i-1].T, delta)
        grads[f'b{i}'] = np.sum(delta, axis=0)

    return grads

def _train_bnn(
    X: np.ndarray,
    y_true: np.ndarray,
    params: Dict,
    activation: str,
    likelihood_std: float,
    optimizer_func: Callable,
    learning_rate: float,
    n_iterations: int,
    batch_size: Optional[int],
    tol: float,
    verbose: bool
) -> Dict:
    """Train the Bayesian Neural Network."""
    metrics = {
        'loss': [],
        'mse': []
    }

    prev_loss = float('inf')

    if batch_size is None:
        batch_size = X.shape[0]

    for iteration in range(n_iterations):
        # Mini-batch selection
        indices = np.random.permutation(X.shape[0])
        X_batch = X[indices[:batch_size]]
        y_batch = y_true[indices[:batch_size]]

        # Forward pass
        y_pred = _forward_pass(X_batch, params, activation)

        # Compute loss and metrics
        current_loss = _compute_loss(y_batch, y_pred, likelihood_std)
        mse = np.mean((y_batch - y_pred)**2)

        metrics['loss'].append(current_loss)
        metrics['mse'].append(mse)

        # Check for convergence
        if abs(prev_loss - current_loss) < tol:
            if verbose:
                print(f"Converged at iteration {iteration}")
            break

        prev_loss = current_loss

        # Compute gradients
        grads = _compute_gradients(X_batch, y_batch, params, activation)

        # Update parameters
        optimizer_func(params, grads, learning_rate)

        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {current_loss:.4f}, MSE: {mse:.4f}")

    return metrics

################################################################################
# bayesian_optimization
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def bayesian_optimization_fit(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_iter: int = 100,
    acquisition_function: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = None,
    kernel_function: Callable[[np.ndarray, np.ndarray], float] = None,
    normalization: str = 'standard',
    metric: str = 'mse',
    solver: str = 'closed_form',
    tol: float = 1e-4,
    max_evals: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Perform Bayesian optimization to find the minimum of an objective function.

    Parameters
    ----------
    objective_function : Callable[[np.ndarray], float]
        The objective function to minimize.
    bounds : np.ndarray
        Array of shape (n_params, 2) defining the bounds for each parameter.
    n_iter : int, optional
        Number of iterations to perform, by default 100.
    acquisition_function : Callable[[np.ndarray, np.ndarray, np.ndarray], float], optional
        Acquisition function to use. If None, uses expected improvement.
    kernel_function : Callable[[np.ndarray, np.ndarray], float], optional
        Kernel function for the Gaussian process. If None, uses squared exponential.
    normalization : str, optional
        Normalization method for the objective function values. Options: 'none', 'standard', 'minmax', by default 'standard'.
    metric : str, optional
        Metric to use for optimization. Options: 'mse', 'mae', 'r2', by default 'mse'.
    solver : str, optional
        Solver to use for optimization. Options: 'closed_form', 'gradient_descent', by default 'closed_form'.
    tol : float, optional
        Tolerance for convergence, by default 1e-4.
    max_evals : int, optional
        Maximum number of evaluations, by default 1000.
    random_state : Optional[int], optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing the optimization results.
    """
    # Initialize random state
    if random_state is not None:
        np.random.seed(random_state)

    # Validate inputs
    _validate_inputs(objective_function, bounds, n_iter, tol, max_evals)

    # Initialize Gaussian process
    gp = _initialize_gaussian_process(bounds, kernel_function)

    # Normalize objective function values
    y_normalized = _normalize_objective(objective_function, bounds, normalization)

    # Perform optimization
    x_opt, y_opt = _optimize(
        gp,
        objective_function,
        bounds,
        n_iter,
        acquisition_function,
        solver,
        tol,
        max_evals
    )

    # Calculate metrics
    metrics = _calculate_metrics(y_normalized, metric)

    return {
        'result': x_opt,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

def _validate_inputs(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_iter: int,
    tol: float,
    max_evals: int
) -> None:
    """
    Validate the inputs for Bayesian optimization.

    Parameters
    ----------
    objective_function : Callable[[np.ndarray], float]
        The objective function to minimize.
    bounds : np.ndarray
        Array of shape (n_params, 2) defining the bounds for each parameter.
    n_iter : int
        Number of iterations to perform.
    tol : float
        Tolerance for convergence.
    max_evals : int
        Maximum number of evaluations.

    Raises
    ------
    ValueError
        If any input is invalid.
    """
    if not callable(objective_function):
        raise ValueError("objective_function must be a callable.")
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must be a 2D array of shape (n_params, 2).")
    if n_iter <= 0:
        raise ValueError("n_iter must be a positive integer.")
    if tol <= 0:
        raise ValueError("tol must be a positive float.")
    if max_evals <= 0:
        raise ValueError("max_evals must be a positive integer.")

def _initialize_gaussian_process(
    bounds: np.ndarray,
    kernel_function: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict:
    """
    Initialize the Gaussian process for Bayesian optimization.

    Parameters
    ----------
    bounds : np.ndarray
        Array of shape (n_params, 2) defining the bounds for each parameter.
    kernel_function : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Kernel function for the Gaussian process.

    Returns
    -------
    Dict
        Dictionary containing the initialized Gaussian process.
    """
    if kernel_function is None:
        kernel_function = _squared_exponential_kernel

    return {
        'kernel': kernel_function,
        'X': np.array([]),
        'y': np.array([])
    }

def _squared_exponential_kernel(
    x1: np.ndarray,
    x2: np.ndarray
) -> float:
    """
    Squared exponential kernel function.

    Parameters
    ----------
    x1 : np.ndarray
        First input vector.
    x2 : np.ndarray
        Second input vector.

    Returns
    -------
    float
        Kernel value.
    """
    return np.exp(-0.5 * np.linalg.norm(x1 - x2)**2)

def _normalize_objective(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    normalization: str
) -> np.ndarray:
    """
    Normalize the objective function values.

    Parameters
    ----------
    objective_function : Callable[[np.ndarray], float]
        The objective function to minimize.
    bounds : np.ndarray
        Array of shape (n_params, 2) defining the bounds for each parameter.
    normalization : str
        Normalization method. Options: 'none', 'standard', 'minmax'.

    Returns
    -------
    np.ndarray
        Normalized objective function values.
    """
    if normalization == 'none':
        return np.array([])
    elif normalization == 'standard':
        # Sample points within bounds
        n_samples = 100
        X = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, bounds.shape[0]))
        y = np.array([objective_function(x) for x in X])
        return (y - np.mean(y)) / np.std(y)
    elif normalization == 'minmax':
        n_samples = 100
        X = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, bounds.shape[0]))
        y = np.array([objective_function(x) for x in X])
        return (y - np.min(y)) / (np.max(y) - np.min(y))
    else:
        raise ValueError("Invalid normalization method.")

def _optimize(
    gp: Dict,
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_iter: int,
    acquisition_function: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], float]],
    solver: str,
    tol: float,
    max_evals: int
) -> Tuple[np.ndarray, float]:
    """
    Perform the optimization using Bayesian methods.

    Parameters
    ----------
    gp : Dict
        Gaussian process dictionary.
    objective_function : Callable[[np.ndarray], float]
        The objective function to minimize.
    bounds : np.ndarray
        Array of shape (n_params, 2) defining the bounds for each parameter.
    n_iter : int
        Number of iterations to perform.
    acquisition_function : Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], float]]
        Acquisition function to use.
    solver : str
        Solver to use for optimization. Options: 'closed_form', 'gradient_descent'.
    tol : float
        Tolerance for convergence.
    max_evals : int
        Maximum number of evaluations.

    Returns
    -------
    Tuple[np.ndarray, float]
        Optimal parameters and corresponding objective value.
    """
    if acquisition_function is None:
        acquisition_function = _expected_improvement

    for _ in range(n_iter):
        # Update Gaussian process
        gp = _update_gaussian_process(gp, objective_function, bounds)

        # Find next point to evaluate
        x_next = _find_next_point(gp, bounds, acquisition_function, solver)

        # Evaluate objective function
        y_next = objective_function(x_next)
        gp['X'] = np.vstack([gp['X'], x_next])
        gp['y'] = np.append(gp['y'], y_next)

        # Check convergence
        if _check_convergence(gp, tol):
            break

    x_opt = gp['X'][np.argmin(gp['y'])]
    y_opt = np.min(gp['y'])
    return x_opt, y_opt

def _update_gaussian_process(
    gp: Dict,
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray
) -> Dict:
    """
    Update the Gaussian process with new observations.

    Parameters
    ----------
    gp : Dict
        Gaussian process dictionary.
    objective_function : Callable[[np.ndarray], float]
        The objective function to minimize.
    bounds : np.ndarray
        Array of shape (n_params, 2) defining the bounds for each parameter.

    Returns
    -------
    Dict
        Updated Gaussian process dictionary.
    """
    # This is a placeholder for the actual update logic
    return gp

def _find_next_point(
    gp: Dict,
    bounds: np.ndarray,
    acquisition_function: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    solver: str
) -> np.ndarray:
    """
    Find the next point to evaluate using the acquisition function.

    Parameters
    ----------
    gp : Dict
        Gaussian process dictionary.
    bounds : np.ndarray
        Array of shape (n_params, 2) defining the bounds for each parameter.
    acquisition_function : Callable[[np.ndarray, np.ndarray, np.ndarray], float]
        Acquisition function to use.
    solver : str
        Solver to use for optimization. Options: 'closed_form', 'gradient_descent'.

    Returns
    -------
    np.ndarray
        Next point to evaluate.
    """
    # This is a placeholder for the actual logic
    return np.random.uniform(bounds[:, 0], bounds[:, 1])

def _expected_improvement(
    x: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Expected improvement acquisition function.

    Parameters
    ----------
    x : np.ndarray
        Point to evaluate.
    X : np.ndarray
        Observed points.
    y : np.ndarray
        Observed objective values.

    Returns
    -------
    float
        Expected improvement value.
    """
    # This is a placeholder for the actual logic
    return 0.0

def _check_convergence(
    gp: Dict,
    tol: float
) -> bool:
    """
    Check for convergence of the optimization.

    Parameters
    ----------
    gp : Dict
        Gaussian process dictionary.
    tol : float
        Tolerance for convergence.

    Returns
    -------
    bool
        True if converged, False otherwise.
    """
    # This is a placeholder for the actual logic
    return False

def _calculate_metrics(
    y_normalized: np.ndarray,
    metric: str
) -> Dict[str, float]:
    """
    Calculate the optimization metrics.

    Parameters
    ----------
    y_normalized : np.ndarray
        Normalized objective function values.
    metric : str
        Metric to use. Options: 'mse', 'mae', 'r2'.

    Returns
    -------
    Dict[str, float]
        Dictionary of calculated metrics.
    """
    if metric == 'mse':
        return {'mse': np.mean(y_normalized**2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(y_normalized))}
    elif metric == 'r2':
        return {'r2': 1 - np.sum(y_normalized**2) / len(y_normalized)}
    else:
        raise ValueError("Invalid metric.")

################################################################################
# bayesian_structural_time_series
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def bayesian_structural_time_series_fit(
    y: np.ndarray,
    X: Optional[np.ndarray] = None,
    normalizer: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a Bayesian Structural Time Series model.

    Parameters:
    -----------
    y : np.ndarray
        Target values.
    X : Optional[np.ndarray], default=None
        Feature matrix. If None, only intercept is used.
    normalizer : str, default='standard'
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : Union[str, Callable], default='mse'
        Metric to evaluate model performance: 'mse', 'mae', 'r2', or custom callable.
    solver : str, default='gradient_descent'
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : Optional[str], default=None
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float, default=1e-4
        Tolerance for convergence.
    max_iter : int, default=1000
        Maximum number of iterations.
    custom_metric : Optional[Callable], default=None
        Custom metric function if not using built-in metrics.
    custom_distance : Optional[Callable], default=None
        Custom distance function for solver.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(y, X)

    # Normalize data
    y_norm, X_norm = _normalize_data(y, X, normalizer)

    # Initialize parameters
    params = _initialize_parameters(X_norm)

    # Choose solver and fit model
    if solver == 'closed_form':
        params = _solve_closed_form(X_norm, y_norm)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(
            X_norm, y_norm, metric, tol, max_iter,
            regularization, custom_metric, custom_distance
        )
    elif solver == 'newton':
        params = _solve_newton(
            X_norm, y_norm, metric, tol, max_iter,
            regularization, custom_metric
        )
    elif solver == 'coordinate_descent':
        params = _solve_coordinate_descent(
            X_norm, y_norm, metric, tol, max_iter,
            regularization, custom_metric
        )
    else:
        raise ValueError("Invalid solver specified.")

    # Calculate metrics
    metrics = _calculate_metrics(y_norm, X_norm, params, metric, custom_metric)

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(y: np.ndarray, X: Optional[np.ndarray]) -> None:
    """Validate input arrays."""
    if not isinstance(y, np.ndarray) or (X is not None and not isinstance(X, np.ndarray)):
        raise TypeError("Inputs must be numpy arrays.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X is not None and X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X is not None and y.shape[0] != X.shape[0]:
        raise ValueError("y and X must have the same number of samples.")

def _normalize_data(
    y: np.ndarray,
    X: Optional[np.ndarray],
    method: str
) -> tuple:
    """Normalize data based on specified method."""
    if method == 'standard':
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_norm = (y - y_mean) / y_std
    elif method == 'minmax':
        y_min = np.min(y)
        y_max = np.max(y)
        y_norm = (y - y_min) / (y_max - y_min)
    elif method == 'robust':
        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        y_norm = (y - y_median) / y_iqr
    else:
        y_norm = y.copy()

    if X is not None:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        if method == 'standard':
            X_norm = (X - X_mean) / X_std
        elif method == 'minmax':
            X_min = np.min(X, axis=0)
            X_max = np.max(X, axis=0)
            X_norm = (X - X_min) / (X_max - X_min)
        elif method == 'robust':
            X_median = np.median(X, axis=0)
            X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
            X_norm = (X - X_median) / X_iqr
        else:
            X_norm = X.copy()
    else:
        X_norm = None

    return y_norm, X_norm

def _initialize_parameters(X: Optional[np.ndarray]) -> np.ndarray:
    """Initialize model parameters."""
    if X is None:
        return np.array([0.0])
    else:
        n_features = X.shape[1]
        return np.zeros(n_features)

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve using closed-form solution."""
    XTX = np.dot(X.T, X)
    if not np.allclose(XTX, XTX.T):
        raise ValueError("X^T X is not symmetric positive definite.")
    params = np.linalg.solve(XTX, np.dot(X.T, y))
    return params

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    custom_metric: Optional[Callable],
    custom_distance: Optional[Callable]
) -> np.ndarray:
    """Solve using gradient descent."""
    params = _initialize_parameters(X)
    for _ in range(max_iter):
        grad = _compute_gradient(X, y, params, metric, custom_metric)
        if regularization == 'l1':
            grad += np.sign(params)  # L1 regularization
        elif regularization == 'l2':
            grad += 2 * params  # L2 regularization
        elif regularization == 'elasticnet':
            grad += np.sign(params) + 2 * params  # ElasticNet regularization

        if custom_distance is not None:
            step_size = _compute_step_size(X, y, params, grad, custom_distance)
        else:
            step_size = 0.1

        params -= step_size * grad
    return params

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> np.ndarray:
    """Compute gradient based on specified metric."""
    if custom_metric is not None:
        grad = custom_metric(X, y, params)
    elif metric == 'mse':
        residuals = np.dot(X, params) - y
        grad = 2 * np.dot(X.T, residuals) / len(y)
    elif metric == 'mae':
        grad = np.dot(X.T, np.sign(np.dot(X, params) - y)) / len(y)
    else:
        raise ValueError("Unsupported metric for gradient computation.")
    return grad

def _compute_step_size(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    grad: np.ndarray,
    distance_func: Callable
) -> float:
    """Compute step size using custom distance function."""
    return distance_func(X, y, params, grad)

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    custom_metric: Optional[Callable]
) -> np.ndarray:
    """Solve using Newton's method."""
    params = _initialize_parameters(X)
    for _ in range(max_iter):
        grad = _compute_gradient(X, y, params, metric, custom_metric)
        hessian = _compute_hessian(X, y, params, metric)

        if regularization == 'l1':
            hessian += np.eye(len(params))  # L1 regularization
        elif regularization == 'l2':
            hessian += 2 * np.eye(len(params))  # L2 regularization

        if not np.allclose(hessian, hessian.T):
            raise ValueError("Hessian is not symmetric.")

        params -= np.linalg.solve(hessian, grad)
    return params

def _compute_hessian(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: str
) -> np.ndarray:
    """Compute Hessian matrix based on specified metric."""
    if metric == 'mse':
        hessian = 2 * np.dot(X.T, X) / len(y)
    else:
        raise ValueError("Unsupported metric for Hessian computation.")
    return hessian

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    custom_metric: Optional[Callable]
) -> np.ndarray:
    """Solve using coordinate descent."""
    params = _initialize_parameters(X)
    for _ in range(max_iter):
        for i in range(len(params)):
            X_i = X[:, i]
            params[i] = _update_parameter(X_i, y, params, i, metric, regularization, custom_metric)
    return params

def _update_parameter(
    X_i: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    i: int,
    metric: Union[str, Callable],
    regularization: Optional[str],
    custom_metric: Optional[Callable]
) -> float:
    """Update a single parameter using coordinate descent."""
    if custom_metric is not None:
        return custom_metric(X_i, y, params, i)
    elif metric == 'mse':
        residuals = y - np.dot(X[:, :i], params[:i]) - np.dot(X[:, i+1:], params[i+1:])
        numerator = np.dot(X_i, residuals)
        denominator = np.dot(X_i, X_i) + (1.0 if regularization == 'l2' else 0.0)
        return numerator / denominator
    else:
        raise ValueError("Unsupported metric for coordinate descent.")

def _calculate_metrics(
    y: np.ndarray,
    X: Optional[np.ndarray],
    params: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate metrics based on specified metric."""
    if X is None:
        y_pred = np.full_like(y, params[0])
    else:
        y_pred = np.dot(X, params)

    metrics_dict = {}
    if custom_metric is not None:
        metrics_dict['custom'] = custom_metric(y, y_pred)
    elif metric == 'mse':
        metrics_dict['mse'] = np.mean((y - y_pred) ** 2)
    elif metric == 'mae':
        metrics_dict['mae'] = np.mean(np.abs(y - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics_dict['r2'] = 1 - (ss_res / ss_tot)

    return metrics_dict

################################################################################
# bayesian_networks
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def bayesian_networks_fit(
    data: np.ndarray,
    target: np.ndarray,
    structure: Optional[np.ndarray] = None,
    prior_mean: Optional[np.ndarray] = None,
    prior_covariance: Optional[np.ndarray] = None,
    likelihood: str = 'gaussian',
    inference_method: str = 'exact',
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    normalize_data: bool = True,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a Bayesian Network model to the given data.

    Parameters
    ----------
    data : np.ndarray
        Input features of shape (n_samples, n_features).
    target : np.ndarray
        Target values of shape (n_samples,).
    structure : Optional[np.ndarray], default=None
        Adjacency matrix representing the network structure.
    prior_mean : Optional[np.ndarray], default=None
        Prior mean for the parameters.
    prior_covariance : Optional[np.ndarray], default=None
        Prior covariance for the parameters.
    likelihood : str, default='gaussian'
        Likelihood function to use ('gaussian', 'bernoulli').
    inference_method : str, default='exact'
        Inference method ('exact', 'variational', 'mcmc').
    max_iterations : int, default=100
        Maximum number of iterations.
    tolerance : float, default=1e-6
        Tolerance for convergence.
    normalize_data : bool, default=True
        Whether to normalize the input data.
    metric : Union[str, Callable], default='mse'
        Metric to evaluate the model ('mse', 'mae', 'r2').
    custom_metric : Optional[Callable], default=None
        Custom metric function.
    verbose : bool, default=False
        Whether to print progress information.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.rand(100, 5)
    >>> target = np.random.rand(100)
    >>> result = bayesian_networks_fit(data, target)
    """
    # Validate inputs
    _validate_inputs(data, target)

    # Normalize data if required
    if normalize_data:
        data = _normalize_data(data)

    # Initialize parameters
    params = _initialize_parameters(
        data.shape[1],
        prior_mean=prior_mean,
        prior_covariance=prior_covariance
    )

    # Fit the model based on the chosen inference method
    if inference_method == 'exact':
        result = _exact_inference(
            data, target, structure, params,
            likelihood, max_iterations, tolerance
        )
    elif inference_method == 'variational':
        result = _variational_inference(
            data, target, structure, params,
            likelihood, max_iterations, tolerance
        )
    elif inference_method == 'mcmc':
        result = _mcmc_inference(
            data, target, structure, params,
            likelihood, max_iterations, tolerance
        )
    else:
        raise ValueError("Invalid inference method specified.")

    # Calculate metrics
    metrics = _calculate_metrics(
        result['predictions'], target,
        metric=metric, custom_metric=custom_metric
    )

    # Prepare the output dictionary
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

    return output

def _validate_inputs(data: np.ndarray, target: np.ndarray) -> None:
    """Validate the input data and target."""
    if not isinstance(data, np.ndarray) or not isinstance(target, np.ndarray):
        raise TypeError("Data and target must be numpy arrays.")
    if data.shape[0] != target.shape[0]:
        raise ValueError("Data and target must have the same number of samples.")
    if np.any(np.isnan(data)) or np.any(np.isnan(target)):
        raise ValueError("Data and target must not contain NaN values.")
    if np.any(np.isinf(data)) or np.any(np.isinf(target)):
        raise ValueError("Data and target must not contain infinite values.")

def _normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalize the input data."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-8)

def _initialize_parameters(
    n_features: int,
    prior_mean: Optional[np.ndarray] = None,
    prior_covariance: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """Initialize the parameters for the Bayesian Network."""
    if prior_mean is None:
        prior_mean = np.zeros(n_features)
    if prior_covariance is None:
        prior_covariance = np.eye(n_features)
    return {
        'prior_mean': prior_mean,
        'prior_covariance': prior_covariance
    }

def _exact_inference(
    data: np.ndarray,
    target: np.ndarray,
    structure: Optional[np.ndarray],
    params: Dict[str, np.ndarray],
    likelihood: str,
    max_iterations: int,
    tolerance: float
) -> Dict[str, Union[np.ndarray, float]]:
    """Perform exact inference for the Bayesian Network."""
    # Placeholder for exact inference implementation
    predictions = np.random.rand(data.shape[0])
    return {
        'predictions': predictions,
        'posterior_mean': np.random.rand(data.shape[1]),
        'posterior_covariance': np.eye(data.shape[1])
    }

def _variational_inference(
    data: np.ndarray,
    target: np.ndarray,
    structure: Optional[np.ndarray],
    params: Dict[str, np.ndarray],
    likelihood: str,
    max_iterations: int,
    tolerance: float
) -> Dict[str, Union[np.ndarray, float]]:
    """Perform variational inference for the Bayesian Network."""
    # Placeholder for variational inference implementation
    predictions = np.random.rand(data.shape[0])
    return {
        'predictions': predictions,
        'variational_params': np.random.rand(data.shape[1])
    }

def _mcmc_inference(
    data: np.ndarray,
    target: np.ndarray,
    structure: Optional[np.ndarray],
    params: Dict[str, np.ndarray],
    likelihood: str,
    max_iterations: int,
    tolerance: float
) -> Dict[str, Union[np.ndarray, float]]:
    """Perform MCMC inference for the Bayesian Network."""
    # Placeholder for MCMC inference implementation
    predictions = np.random.rand(data.shape[0])
    return {
        'predictions': predictions,
        'mcmc_samples': np.random.rand(10, data.shape[1])
    }

def _calculate_metrics(
    predictions: np.ndarray,
    target: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Calculate the metrics for the model."""
    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((predictions - target) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(predictions - target))
    elif metric == 'r2':
        ss_res = np.sum((predictions - target) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    elif callable(metric):
        metrics['custom'] = metric(predictions, target)
    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(predictions, target)
    return metrics
