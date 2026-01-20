"""
Quantix – Module inference_bayesienne
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# priors
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def priors_fit(
    data: np.ndarray,
    prior_type: str = 'uniform',
    hyperparameters: Optional[Dict[str, float]] = None,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_prior: Optional[Callable] = None
) -> Dict:
    """
    Fit Bayesian priors to data with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    prior_type : str, optional
        Type of prior distribution ('uniform', 'normal', 'gamma', etc.)
    hyperparameters : dict, optional
        Dictionary of hyperparameters for the prior distribution
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric for evaluation ('mse', 'mae', etc.) or custom callable
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', etc.)
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    custom_prior : callable, optional
        Custom prior distribution function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.random.randn(100, 5)
    >>> result = priors_fit(data, prior_type='normal', hyperparameters={'mean': 0, 'std': 1})
    """
    # Validate inputs
    _validate_inputs(data, prior_type, hyperparameters, normalization, metric)

    # Normalize data if required
    normalized_data = _normalize_data(data, normalization)

    # Prepare prior parameters
    prior_params = _prepare_prior_params(prior_type, hyperparameters, custom_prior)

    # Choose solver
    if solver == 'closed_form':
        result = _solve_closed_form(normalized_data, prior_params)
    elif solver == 'gradient_descent':
        result = _solve_gradient_descent(normalized_data, prior_params, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(result, normalized_data, metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'prior_type': prior_type,
            'hyperparameters': hyperparameters or {},
            'normalization': normalization,
            'metric': metric,
            'solver': solver
        },
        'warnings': _check_warnings(result, metrics)
    }

def _validate_inputs(
    data: np.ndarray,
    prior_type: str,
    hyperparameters: Optional[Dict[str, float]],
    normalization: str,
    metric: Union[str, Callable]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")

    valid_priors = ['uniform', 'normal', 'gamma']
    if prior_type not in valid_priors and not callable(metric):
        raise ValueError(f"Unknown prior type: {prior_type}")

    valid_normalizations = ['none', 'standard', 'minmax', 'robust']
    if normalization not in valid_normalizations:
        raise ValueError(f"Unknown normalization: {normalization}")

    if callable(metric):
        try:
            metric(np.array([0]), np.array([1]))
        except Exception as e:
            raise ValueError(f"Custom metric callable failed: {str(e)}")

def _normalize_data(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Normalize data according to specified method."""
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

def _prepare_prior_params(
    prior_type: str,
    hyperparameters: Optional[Dict[str, float]],
    custom_prior: Optional[Callable]
) -> Dict:
    """Prepare prior parameters based on type and hyperparameters."""
    if custom_prior is not None:
        return {'type': 'custom', 'function': custom_prior}

    params = {'type': prior_type}
    if hyperparameters is None:
        return params

    if prior_type == 'uniform':
        params.update({'low': hyperparameters.get('low', 0), 'high': hyperparameters.get('high', 1)})
    elif prior_type == 'normal':
        params.update({'mean': hyperparameters.get('mean', 0), 'std': hyperparameters.get('std', 1)})
    elif prior_type == 'gamma':
        params.update({'shape': hyperparameters.get('shape', 1), 'scale': hyperparameters.get('scale', 1)})

    return params

def _solve_closed_form(
    data: np.ndarray,
    prior_params: Dict
) -> Dict:
    """Solve for priors using closed-form solution."""
    # Placeholder for actual implementation
    return {
        'posterior_mean': np.mean(data, axis=0),
        'posterior_variance': np.var(data, axis=0)
    }

def _solve_gradient_descent(
    data: np.ndarray,
    prior_params: Dict,
    tol: float,
    max_iter: int
) -> Dict:
    """Solve for priors using gradient descent."""
    # Placeholder for actual implementation
    return {
        'posterior_mean': np.mean(data, axis=0),
        'posterior_variance': np.var(data, axis=0)
    }

def _calculate_metrics(
    result: Dict,
    data: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate metrics for the prior fitting."""
    if callable(metric):
        return {'custom_metric': metric(result['posterior_mean'], data)}

    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((data - result['posterior_mean']) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(data - result['posterior_mean']))
    elif metric == 'r2':
        ss_total = np.sum((data - np.mean(data, axis=0)) ** 2)
        ss_res = np.sum((data - result['posterior_mean']) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_total)

    return metrics

def _check_warnings(
    result: Dict,
    metrics: Dict
) -> list:
    """Check for potential warnings in the results."""
    warnings = []
    if np.any(np.isnan(result['posterior_mean'])):
        warnings.append("Posterior mean contains NaN values")
    if np.any(np.isnan(result['posterior_variance'])):
        warnings.append("Posterior variance contains NaN values")
    if 'mse' in metrics and metrics['mse'] > 1e6:
        warnings.append("High MSE value detected")
    return warnings

################################################################################
# likelihood
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def likelihood_fit(
    data: np.ndarray,
    prior: Callable[[np.ndarray], float],
    likelihood_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    normalizer: str = 'none',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'gradient_descent',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compute the likelihood of given data under a Bayesian framework.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    prior : Callable[[np.ndarray], float]
        Prior distribution function.
    likelihood_func : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom likelihood function. If None, uses default Gaussian likelihood.
    normalizer : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric to evaluate: 'mse', 'mae', 'r2', or custom callable.
    solver : str
        Solver method: 'gradient_descent', 'newton', or 'coordinate_descent'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_params : Optional[Dict[str, Any]]
        Additional parameters for the solver.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, prior, likelihood_func, normalizer, metric)

    # Normalize data
    normalized_data = _normalize(data, normalizer)

    # Set default likelihood function if not provided
    if likelihood_func is None:
        likelihood_func = _default_gaussian_likelihood

    # Set default metric if not provided
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    # Set default solver if not provided
    solver_func = _get_solver_function(solver)

    # Estimate parameters
    params, history = solver_func(
        normalized_data,
        prior,
        likelihood_func,
        tol=tol,
        max_iter=max_iter,
        custom_params=custom_params
    )

    # Compute metrics
    metrics = _compute_metrics(normalized_data, params, metric_func)

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer,
            'metric': metric,
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    prior: Callable[[np.ndarray], float],
    likelihood_func: Optional[Callable[[np.ndarray, np.ndarray], float]],
    normalizer: str,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Data contains infinite values.")

    valid_normalizers = ['none', 'standard', 'minmax', 'robust']
    if normalizer not in valid_normalizers:
        raise ValueError(f"Normalizer must be one of {valid_normalizers}.")

    if isinstance(metric, str):
        valid_metrics = ['mse', 'mae', 'r2']
        if metric not in valid_metrics:
            raise ValueError(f"Metric must be one of {valid_metrics} or a callable.")

def _normalize(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Normalize the input data."""
    if method == 'none':
        return data
    elif method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError("Invalid normalization method.")

def _default_gaussian_likelihood(
    data: np.ndarray,
    params: np.ndarray
) -> float:
    """Default Gaussian likelihood function."""
    residuals = data - params[:-1]
    variance = np.exp(params[-1])
    return (-0.5 * np.sum(residuals**2 / variance) -
            0.5 * len(data) * np.log(2 * np.pi * variance))

def _get_metric_function(
    metric: str
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the metric function based on the input string."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }
    return metrics[metric]

def _mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """Compute mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _get_solver_function(
    solver: str
) -> Callable:
    """Get the solver function based on the input string."""
    solvers = {
        'gradient_descent': _gradient_descent,
        'newton': _newton_method,
        'coordinate_descent': _coordinate_descent
    }
    return solvers[solver]

def _gradient_descent(
    data: np.ndarray,
    prior: Callable[[np.ndarray], float],
    likelihood_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_params: Optional[Dict[str, Any]] = None
) -> tuple:
    """Gradient descent solver."""
    # Initialize parameters (example: mean and log variance)
    params = np.zeros(data.shape[1] + 1)

    for _ in range(max_iter):
        grad = _compute_gradient(data, params, likelihood_func)
        params -= 0.01 * grad

        if np.linalg.norm(grad) < tol:
            break

    return params, None

def _compute_gradient(
    data: np.ndarray,
    params: np.ndarray,
    likelihood_func: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Compute the gradient of the likelihood function."""
    epsilon = 1e-8
    grad = np.zeros_like(params)

    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += epsilon
        params_minus = params.copy()
        params_minus[i] -= epsilon

        grad[i] = (likelihood_func(data, params_plus) -
                   likelihood_func(data, params_minus)) / (2 * epsilon)

    return grad

def _newton_method(
    data: np.ndarray,
    prior: Callable[[np.ndarray], float],
    likelihood_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_params: Optional[Dict[str, Any]] = None
) -> tuple:
    """Newton method solver."""
    # Placeholder for Newton method implementation
    return _gradient_descent(data, prior, likelihood_func, tol, max_iter)

def _coordinate_descent(
    data: np.ndarray,
    prior: Callable[[np.ndarray], float],
    likelihood_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_params: Optional[Dict[str, Any]] = None
) -> tuple:
    """Coordinate descent solver."""
    # Placeholder for coordinate descent implementation
    return _gradient_descent(data, prior, likelihood_func, tol, max_iter)

def _compute_metrics(
    data: np.ndarray,
    params: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Compute metrics for the given data and parameters."""
    predictions = np.dot(data, params[:-1]) + params[-1]
    return {'metric': metric_func(data, predictions)}

################################################################################
# posterior
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def posterior_fit(
    prior: np.ndarray,
    likelihood: Callable[[np.ndarray], float],
    evidence: Optional[float] = None,
    normalization: str = 'standard',
    solver: str = 'gradient_descent',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **solver_kwargs
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Compute the posterior distribution given a prior and likelihood.

    Parameters
    ----------
    prior : np.ndarray
        Prior probability distribution.
    likelihood : callable
        Function that computes the likelihood given parameters.
    evidence : float, optional
        Evidence (marginal likelihood), by default None (computed automatically).
    normalization : str, optional
        Normalization method ('none', 'standard', 'robust'), by default 'standard'.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton'), by default 'gradient_descent'.
    tol : float, optional
        Tolerance for convergence, by default 1e-6.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    custom_metric : callable, optional
        Custom metric function, by default None.
    **solver_kwargs : dict
        Additional solver-specific keyword arguments.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Posterior distribution.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings generated.

    Examples
    --------
    >>> prior = np.array([0.1, 0.2, 0.7])
    >>> def likelihood(params):
    ...     return np.exp(-np.sum(params**2))
    >>> result = posterior_fit(prior, likelihood)
    """
    # Validate inputs
    _validate_inputs(prior, likelihood)

    # Normalize prior if needed
    normalized_prior = _normalize_prior(prior, normalization)

    # Compute evidence if not provided
    if evidence is None:
        evidence = _compute_evidence(normalized_prior, likelihood)

    # Choose solver
    if solver == 'closed_form':
        posterior = _solve_closed_form(normalized_prior, likelihood)
    elif solver == 'gradient_descent':
        posterior = _solve_gradient_descent(
            normalized_prior, likelihood, tol=tol, max_iter=max_iter, **solver_kwargs
        )
    elif solver == 'newton':
        posterior = _solve_newton(
            normalized_prior, likelihood, tol=tol, max_iter=max_iter, **solver_kwargs
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(posterior, normalized_prior, custom_metric)

    return {
        'result': posterior,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

def _validate_inputs(prior: np.ndarray, likelihood: Callable) -> None:
    """Validate input parameters."""
    if not isinstance(prior, np.ndarray):
        raise TypeError("Prior must be a numpy array")
    if not callable(likelihood):
        raise TypeError("Likelihood must be a callable function")
    if np.any(np.isnan(prior)) or np.any(np.isinf(prior)):
        raise ValueError("Prior contains NaN or inf values")

def _normalize_prior(prior: np.ndarray, method: str) -> np.ndarray:
    """Normalize the prior distribution."""
    if method == 'none':
        return prior
    elif method == 'standard':
        return prior / np.sum(prior)
    elif method == 'robust':
        return (prior - np.min(prior)) / (np.max(prior) - np.min(prior))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_evidence(prior: np.ndarray, likelihood: Callable) -> float:
    """Compute the evidence (marginal likelihood)."""
    return np.sum(prior * likelihood(prior))

def _solve_closed_form(prior: np.ndarray, likelihood: Callable) -> np.ndarray:
    """Solve for posterior using closed-form solution."""
    return prior * likelihood(prior) / _compute_evidence(prior, likelihood)

def _solve_gradient_descent(
    prior: np.ndarray,
    likelihood: Callable,
    tol: float = 1e-6,
    max_iter: int = 1000,
    **kwargs
) -> np.ndarray:
    """Solve for posterior using gradient descent."""
    # Initialize parameters
    params = prior.copy()
    evidence = _compute_evidence(prior, likelihood)

    for _ in range(max_iter):
        # Compute gradient
        grad = _compute_gradient(params, likelihood, evidence)

        # Update parameters
        params -= kwargs.get('learning_rate', 0.01) * grad

        # Check convergence
        if np.linalg.norm(grad) < tol:
            break

    return params * likelihood(params) / evidence

def _solve_newton(
    prior: np.ndarray,
    likelihood: Callable,
    tol: float = 1e-6,
    max_iter: int = 1000,
    **kwargs
) -> np.ndarray:
    """Solve for posterior using Newton's method."""
    # Initialize parameters
    params = prior.copy()
    evidence = _compute_evidence(prior, likelihood)

    for _ in range(max_iter):
        # Compute gradient and Hessian
        grad = _compute_gradient(params, likelihood, evidence)
        hess = _compute_hessian(params, likelihood)

        # Update parameters
        params -= np.linalg.solve(hess, grad)

        # Check convergence
        if np.linalg.norm(grad) < tol:
            break

    return params * likelihood(params) / evidence

def _compute_gradient(
    params: np.ndarray,
    likelihood: Callable,
    evidence: float
) -> np.ndarray:
    """Compute the gradient of the posterior."""
    # This is a placeholder; actual implementation depends on likelihood
    return np.zeros_like(params)

def _compute_hessian(
    params: np.ndarray,
    likelihood: Callable
) -> np.ndarray:
    """Compute the Hessian of the posterior."""
    # This is a placeholder; actual implementation depends on likelihood
    return np.eye(len(params))

def _compute_metrics(
    posterior: np.ndarray,
    prior: np.ndarray,
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Compute metrics for the posterior."""
    metrics = {
        'kl_divergence': _compute_kl_divergence(posterior, prior)
    }

    if custom_metric is not None:
        metrics['custom'] = custom_metric(posterior, prior)

    return metrics

def _compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence between two distributions."""
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

################################################################################
# evidence
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def evidence_fit(
    log_likelihood: Callable[[np.ndarray], float],
    log_prior: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    solver: str = 'gradient_descent',
    max_iter: int = 1000,
    tol: float = 1e-6,
    normalize: str = 'none',
    metric: Union[str, Callable[[np.ndarray], float]] = 'mse',
    regularization: str = 'none',
    alpha: float = 1.0,
    beta: float = 1.0,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the evidence (marginal likelihood) for Bayesian inference.

    Parameters:
    -----------
    log_likelihood : callable
        Function that computes the log-likelihood of the data given parameters.
    log_prior : callable
        Function that computes the log-prior of the parameters.
    initial_params : np.ndarray
        Initial guess for the parameters.
    solver : str, optional
        Solver to use ('gradient_descent', 'newton', 'coordinate_descent').
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to use ('mse', 'mae', 'r2', 'logloss') or custom callable.
    regularization : str, optional
        Regularization method ('none', 'l1', 'l2', 'elasticnet').
    alpha : float, optional
        Regularization strength for L1 or elastic net.
    beta : float, optional
        Regularization strength for L2 or elastic net.
    custom_distance : callable, optional
        Custom distance function.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(log_likelihood, log_prior, initial_params)

    # Normalize parameters if needed
    normalized_params = apply_normalization(initial_params, normalize)

    # Choose solver and compute evidence
    if solver == 'gradient_descent':
        result, metrics = gradient_descent_solver(
            log_likelihood, log_prior, normalized_params,
            max_iter=max_iter, tol=tol, metric=metric
        )
    elif solver == 'newton':
        result, metrics = newton_solver(
            log_likelihood, log_prior, normalized_params,
            max_iter=max_iter, tol=tol, metric=metric
        )
    elif solver == 'coordinate_descent':
        result, metrics = coordinate_descent_solver(
            log_likelihood, log_prior, normalized_params,
            max_iter=max_iter, tol=tol, metric=metric
        )
    else:
        raise ValueError("Unsupported solver. Choose from 'gradient_descent', 'newton', 'coordinate_descent'.")

    # Apply regularization if needed
    result = apply_regularization(result, regularization, alpha, beta)

    # Compute metrics
    if isinstance(metric, str):
        metrics = compute_metrics(result, metric)
    else:
        metrics['custom'] = metric(result)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': normalized_params,
        'warnings': []
    }

    return output

def validate_inputs(
    log_likelihood: Callable[[np.ndarray], float],
    log_prior: Callable[[np.ndarray], float],
    initial_params: np.ndarray
) -> None:
    """
    Validate the inputs for evidence computation.

    Parameters:
    -----------
    log_likelihood : callable
        Function that computes the log-likelihood of the data given parameters.
    log_prior : callable
        Function that computes the log-prior of the parameters.
    initial_params : np.ndarray
        Initial guess for the parameters.

    Raises:
    -------
    ValueError
        If inputs are invalid.
    """
    if not callable(log_likelihood):
        raise ValueError("log_likelihood must be a callable.")
    if not callable(log_prior):
        raise ValueError("log_prior must be a callable.")
    if not isinstance(initial_params, np.ndarray):
        raise ValueError("initial_params must be a numpy array.")
    if np.any(np.isnan(initial_params)) or np.any(np.isinf(initial_params)):
        raise ValueError("initial_params must not contain NaN or inf values.")

def apply_normalization(
    params: np.ndarray,
    method: str
) -> np.ndarray:
    """
    Apply normalization to the parameters.

    Parameters:
    -----------
    params : np.ndarray
        Parameters to normalize.
    method : str
        Normalization method ('none', 'standard', 'minmax', 'robust').

    Returns:
    --------
    np.ndarray
        Normalized parameters.
    """
    if method == 'none':
        return params
    elif method == 'standard':
        mean = np.mean(params)
        std = np.std(params)
        return (params - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(params)
        max_val = np.max(params)
        return (params - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(params)
        iqr = np.percentile(params, 75) - np.percentile(params, 25)
        return (params - median) / (iqr + 1e-8)
    else:
        raise ValueError("Unsupported normalization method.")

def gradient_descent_solver(
    log_likelihood: Callable[[np.ndarray], float],
    log_prior: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
    metric: Union[str, Callable[[np.ndarray], float]] = 'mse'
) -> tuple[float, Dict[str, float]]:
    """
    Solve for evidence using gradient descent.

    Parameters:
    -----------
    log_likelihood : callable
        Function that computes the log-likelihood of the data given parameters.
    log_prior : callable
        Function that computes the log-prior of the parameters.
    initial_params : np.ndarray
        Initial guess for the parameters.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    metric : str or callable, optional
        Metric to use ('mse', 'mae', 'r2', 'logloss') or custom callable.

    Returns:
    --------
    tuple
        Tuple containing the evidence result and metrics.
    """
    params = initial_params.copy()
    learning_rate = 0.01
    for _ in range(max_iter):
        grad_log_likelihood = compute_gradient(log_likelihood, params)
        grad_log_prior = compute_gradient(log_prior, params)
        gradient = grad_log_likelihood + grad_log_prior
        params -= learning_rate * gradient
        if np.linalg.norm(gradient) < tol:
            break
    evidence = compute_evidence(log_likelihood, log_prior, params)
    metrics = {'gradient_descent': evidence}
    return evidence, metrics

def newton_solver(
    log_likelihood: Callable[[np.ndarray], float],
    log_prior: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
    metric: Union[str, Callable[[np.ndarray], float]] = 'mse'
) -> tuple[float, Dict[str, float]]:
    """
    Solve for evidence using Newton's method.

    Parameters:
    -----------
    log_likelihood : callable
        Function that computes the log-likelihood of the data given parameters.
    log_prior : callable
        Function that computes the log-prior of the parameters.
    initial_params : np.ndarray
        Initial guess for the parameters.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    metric : str or callable, optional
        Metric to use ('mse', 'mae', 'r2', 'logloss') or custom callable.

    Returns:
    --------
    tuple
        Tuple containing the evidence result and metrics.
    """
    params = initial_params.copy()
    for _ in range(max_iter):
        grad_log_likelihood = compute_gradient(log_likelihood, params)
        grad_log_prior = compute_gradient(log_prior, params)
        hessian_log_likelihood = compute_hessian(log_likelihood, params)
        hessian_log_prior = compute_hessian(log_prior, params)
        hessian = hessian_log_likelihood + hessian_log_prior
        update = np.linalg.solve(hessian, -(grad_log_likelihood + grad_log_prior))
        params += update
        if np.linalg.norm(update) < tol:
            break
    evidence = compute_evidence(log_likelihood, log_prior, params)
    metrics = {'newton': evidence}
    return evidence, metrics

def coordinate_descent_solver(
    log_likelihood: Callable[[np.ndarray], float],
    log_prior: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
    metric: Union[str, Callable[[np.ndarray], float]] = 'mse'
) -> tuple[float, Dict[str, float]]:
    """
    Solve for evidence using coordinate descent.

    Parameters:
    -----------
    log_likelihood : callable
        Function that computes the log-likelihood of the data given parameters.
    log_prior : callable
        Function that computes the log-prior of the parameters.
    initial_params : np.ndarray
        Initial guess for the parameters.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    metric : str or callable, optional
        Metric to use ('mse', 'mae', 'r2', 'logloss') or custom callable.

    Returns:
    --------
    tuple
        Tuple containing the evidence result and metrics.
    """
    params = initial_params.copy()
    for _ in range(max_iter):
        for i in range(len(params)):
            other_params = params.copy()
            other_params[i] = 0
            grad = compute_gradient(lambda p: log_likelihood(p) + log_prior(p), params)[i]
            step = grad / compute_hessian(lambda p: log_likelihood(p) + log_prior(p), params)[i, i]
            params[i] -= step
        if np.linalg.norm(grad) < tol:
            break
    evidence = compute_evidence(log_likelihood, log_prior, params)
    metrics = {'coordinate_descent': evidence}
    return evidence, metrics

def compute_gradient(
    func: Callable[[np.ndarray], float],
    params: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Compute the gradient of a function using finite differences.

    Parameters:
    -----------
    func : callable
        Function to compute the gradient of.
    params : np.ndarray
        Parameters at which to compute the gradient.
    epsilon : float, optional
        Step size for finite differences.

    Returns:
    --------
    np.ndarray
        Gradient of the function.
    """
    gradient = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += epsilon
        params_minus = params.copy()
        params_minus[i] -= epsilon
        gradient[i] = (func(params_plus) - func(params_minus)) / (2 * epsilon)
    return gradient

def compute_hessian(
    func: Callable[[np.ndarray], float],
    params: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Compute the Hessian of a function using finite differences.

    Parameters:
    -----------
    func : callable
        Function to compute the Hessian of.
    params : np.ndarray
        Parameters at which to compute the Hessian.
    epsilon : float, optional
        Step size for finite differences.

    Returns:
    --------
    np.ndarray
        Hessian of the function.
    """
    n = len(params)
    hessian = np.zeros((n, n))
    for i in range(n):
        params_plus_i = params.copy()
        params_plus_i[i] += epsilon
        grad_plus_i = compute_gradient(func, params_plus_i)
        params_minus_i = params.copy()
        params_minus_i[i] -= epsilon
        grad_minus_i = compute_gradient(func, params_minus_i)
        hessian[:, i] = (grad_plus_i - grad_minus_i) / (2 * epsilon)
    return hessian

def compute_evidence(
    log_likelihood: Callable[[np.ndarray], float],
    log_prior: Callable[[np.ndarray], float],
    params: np.ndarray
) -> float:
    """
    Compute the evidence (marginal likelihood).

    Parameters:
    -----------
    log_likelihood : callable
        Function that computes the log-likelihood of the data given parameters.
    log_prior : callable
        Function that computes the log-prior of the parameters.
    params : np.ndarray
        Parameters at which to compute the evidence.

    Returns:
    --------
    float
        Evidence (marginal likelihood).
    """
    return log_likelihood(params) + log_prior(params)

def apply_regularization(
    result: float,
    method: str,
    alpha: float = 1.0,
    beta: float = 1.0
) -> float:
    """
    Apply regularization to the result.

    Parameters:
    -----------
    result : float
        Result to regularize.
    method : str
        Regularization method ('none', 'l1', 'l2', 'elasticnet').
    alpha : float, optional
        Regularization strength for L1 or elastic net.
    beta : float, optional
        Regularization strength for L2 or elastic net.

    Returns:
    --------
    float
        Regularized result.
    """
    if method == 'none':
        return result
    elif method == 'l1':
        return result - alpha * np.sum(np.abs(result))
    elif method == 'l2':
        return result - beta * np.sum(result**2)
    elif method == 'elasticnet':
        return result - alpha * np.sum(np.abs(result)) - beta * np.sum(result**2)
    else:
        raise ValueError("Unsupported regularization method.")

def compute_metrics(
    result: float,
    metric: str
) -> Dict[str, float]:
    """
    Compute metrics for the result.

    Parameters:
    -----------
    result : float
        Result to compute metrics for.
    metric : str
        Metric to use ('mse', 'mae', 'r2', 'logloss').

    Returns:
    --------
    dict
        Dictionary containing the computed metrics.
    """
    if metric == 'mse':
        return {'mse': result**2}
    elif metric == 'mae':
        return {'mae': np.abs(result)}
    elif metric == 'r2':
        return {'r2': 1 - result}
    elif metric == 'logloss':
        return {'logloss': -np.log(result + 1e-8)}
    else:
        raise ValueError("Unsupported metric.")

################################################################################
# bayes_theorem
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def bayes_theorem_fit(
    prior: np.ndarray,
    likelihood: Callable[[np.ndarray], np.ndarray],
    evidence: Optional[float] = None,
    normalization: str = 'none',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Compute the posterior distribution using Bayes' theorem.

    Parameters
    ----------
    prior : np.ndarray
        Prior probability distribution.
    likelihood : Callable[[np.ndarray], np.ndarray]
        Likelihood function that takes data and returns likelihood distribution.
    evidence : Optional[float], default=None
        Evidence (marginal likelihood). If None, it will be computed.
    normalization : str, default='none'
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    solver : str, default='closed_form'
        Solver method: 'closed_form', 'gradient_descent', or 'newton'.
    tol : float, default=1e-6
        Tolerance for convergence.
    max_iter : int, default=1000
        Maximum number of iterations for iterative solvers.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], default=None
        Custom metric function to evaluate the result.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        Dictionary containing:
        - "result": Posterior distribution.
        - "metrics": Dictionary of computed metrics.
        - "params_used": Parameters used in the computation.
        - "warnings": Any warnings generated during computation.

    Examples
    --------
    >>> prior = np.array([0.2, 0.3, 0.5])
    >>> likelihood = lambda data: np.array([0.1, 0.6, 0.3])
    >>> result = bayes_theorem_fit(prior, likelihood)
    """
    # Validate inputs
    _validate_inputs(prior, likelihood)

    # Normalize prior if needed
    normalized_prior = _normalize(prior, normalization)

    # Compute evidence if not provided
    if evidence is None:
        evidence = _compute_evidence(normalized_prior, likelihood)

    # Compute posterior
    posterior = _compute_posterior(normalized_prior, likelihood, evidence, solver, tol, max_iter)

    # Compute metrics
    metrics = _compute_metrics(posterior, custom_metric)

    return {
        "result": posterior,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "solver": solver,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

def _validate_inputs(prior: np.ndarray, likelihood: Callable[[np.ndarray], np.ndarray]) -> None:
    """Validate the inputs for Bayes' theorem computation."""
    if not isinstance(prior, np.ndarray):
        raise TypeError("Prior must be a numpy array.")
    if not isinstance(likelihood, Callable):
        raise TypeError("Likelihood must be a callable function.")
    if np.any(prior < 0):
        raise ValueError("Prior probabilities must be non-negative.")
    if not np.isclose(np.sum(prior), 1.0, atol=1e-6):
        raise ValueError("Prior probabilities must sum to 1.")

def _normalize(prior: np.ndarray, method: str) -> np.ndarray:
    """Normalize the prior distribution."""
    if method == 'none':
        return prior
    elif method == 'standard':
        return (prior - np.mean(prior)) / np.std(prior)
    elif method == 'minmax':
        return (prior - np.min(prior)) / (np.max(prior) - np.min(prior))
    elif method == 'robust':
        return (prior - np.median(prior)) / (np.percentile(prior, 75) - np.percentile(prior, 25))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_evidence(prior: np.ndarray, likelihood: Callable[[np.ndarray], np.ndarray]) -> float:
    """Compute the evidence (marginal likelihood)."""
    likelihood_values = likelihood(prior)
    return np.sum(prior * likelihood_values)

def _compute_posterior(
    prior: np.ndarray,
    likelihood: Callable[[np.ndarray], np.ndarray],
    evidence: float,
    solver: str,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Compute the posterior distribution."""
    likelihood_values = likelihood(prior)
    if solver == 'closed_form':
        return (prior * likelihood_values) / evidence
    elif solver == 'gradient_descent':
        return _gradient_descent_solver(prior, likelihood_values, evidence, tol, max_iter)
    elif solver == 'newton':
        return _newton_solver(prior, likelihood_values, evidence, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver method: {solver}")

def _gradient_descent_solver(
    prior: np.ndarray,
    likelihood_values: np.ndarray,
    evidence: float,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve for posterior using gradient descent."""
    posterior = prior * likelihood_values / evidence
    for _ in range(max_iter):
        gradient = (prior * likelihood_values - posterior) / evidence
        posterior += gradient
        if np.linalg.norm(gradient) < tol:
            break
    return posterior

def _newton_solver(
    prior: np.ndarray,
    likelihood_values: np.ndarray,
    evidence: float,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve for posterior using Newton's method."""
    posterior = prior * likelihood_values / evidence
    for _ in range(max_iter):
        hessian = np.diag(prior * likelihood_values)
        gradient = (prior * likelihood_values - posterior) / evidence
        posterior -= np.linalg.solve(hessian, gradient)
        if np.linalg.norm(gradient) < tol:
            break
    return posterior

def _compute_metrics(
    posterior: np.ndarray,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for the posterior distribution."""
    metrics = {
        'entropy': _compute_entropy(posterior),
        'kl_divergence': _compute_kl_divergence(posterior)
    }
    if custom_metric is not None:
        metrics['custom'] = custom_metric(posterior, posterior)
    return metrics

def _compute_entropy(probabilities: np.ndarray) -> float:
    """Compute the entropy of a probability distribution."""
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def _compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute the Kullback-Leibler divergence between two distributions."""
    return np.sum(p * (np.log2(p + 1e-10) - np.log2(q + 1e-10)))

################################################################################
# conjugate_priors
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def conjugate_priors_fit(
    data: np.ndarray,
    prior_params: Dict[str, Any],
    likelihood_type: str = 'gaussian',
    solver: str = 'closed_form',
    metric: Union[str, Callable] = 'mse',
    normalization: str = 'none',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fit conjugate priors for Bayesian inference.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    prior_params : dict
        Dictionary containing prior parameters.
    likelihood_type : str, optional
        Type of likelihood (default: 'gaussian').
    solver : str, optional
        Solver method (default: 'closed_form').
    metric : Union[str, Callable], optional
        Metric to evaluate (default: 'mse').
    normalization : str, optional
        Normalization method (default: 'none').
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    custom_metric : Optional[Callable], optional
        Custom metric function (default: None).

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, prior_params)

    # Normalize data if required
    normalized_data = _apply_normalization(data, normalization)

    # Choose solver
    if solver == 'closed_form':
        posterior_params = _solve_closed_form(normalized_data, prior_params, likelihood_type)
    else:
        posterior_params = _iterative_solver(normalized_data, prior_params, likelihood_type,
                                            solver, tol, max_iter)

    # Compute metrics
    metrics = _compute_metrics(normalized_data, posterior_params, metric, custom_metric)

    # Prepare output
    result = {
        'result': posterior_params,
        'metrics': metrics,
        'params_used': {
            'likelihood_type': likelihood_type,
            'solver': solver,
            'metric': metric,
            'normalization': normalization
        },
        'warnings': []
    }

    return result

def _validate_inputs(data: np.ndarray, prior_params: Dict[str, Any]) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if not isinstance(prior_params, dict):
        raise TypeError("Prior parameters must be a dictionary")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to data."""
    if method == 'standard':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        return data

def _solve_closed_form(data: np.ndarray, prior_params: Dict[str, Any], likelihood_type: str) -> Dict[str, Any]:
    """Solve for posterior parameters using closed-form solution."""
    if likelihood_type == 'gaussian':
        n = data.shape[0]
        alpha_prior, beta_prior = prior_params['alpha'], prior_params['beta']
        mu_prior, sigma_prior_sq = prior_params.get('mu', 0), prior_params.get('sigma_sq', 1)

        alpha_post = alpha_prior + n / 2
        beta_post = beta_prior + np.sum((data - mu_prior)**2) / 2
        mu_post = (beta_prior * mu_prior + n * np.mean(data)) / (beta_prior + n)
        sigma_post_sq = 1 / (alpha_post * (1 / beta_prior + np.sum((data - mu_post)**2) / (beta_post * n)))

        return {
            'alpha': alpha_post,
            'beta': beta_post,
            'mu': mu_post,
            'sigma_sq': sigma_post_sq
        }
    else:
        raise NotImplementedError(f"Closed-form solution not implemented for {likelihood_type} likelihood")

def _iterative_solver(data: np.ndarray, prior_params: Dict[str, Any], likelihood_type: str,
                     solver: str, tol: float, max_iter: int) -> Dict[str, Any]:
    """Solve for posterior parameters using iterative methods."""
    # Placeholder for iterative solvers
    raise NotImplementedError("Iterative solver not implemented yet")

def _compute_metrics(data: np.ndarray, posterior_params: Dict[str, Any],
                    metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    if callable(metric):
        return {'custom_metric': metric(data, posterior_params)}
    elif custom_metric is not None:
        return {'custom_metric': custom_metric(data, posterior_params)}
    else:
        if metric == 'mse':
            return {'mse': np.mean((data - posterior_params['mu'])**2)}
        elif metric == 'mae':
            return {'mae': np.mean(np.abs(data - posterior_params['mu']))}
        elif metric == 'r2':
            ss_res = np.sum((data - posterior_params['mu'])**2)
            ss_tot = np.sum((data - np.mean(data))**2)
            return {'r2': 1 - ss_res / ss_tot}
        else:
            raise ValueError(f"Unknown metric: {metric}")

################################################################################
# markov_chain_monte_carlo
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(
    target_distribution: Callable,
    proposal_distribution: Callable,
    initial_state: np.ndarray,
    n_iterations: int,
    burn_in: int = 0
) -> None:
    """
    Validate the inputs for MCMC simulation.

    Parameters
    ----------
    target_distribution : Callable
        The target distribution to sample from.
    proposal_distribution : Callable
        The proposal distribution for the Markov chain.
    initial_state : np.ndarray
        Initial state of the Markov chain.
    n_iterations : int
        Number of iterations to run the MCMC algorithm.
    burn_in : int, optional
        Number of initial samples to discard (default is 0).

    Raises
    ------
    ValueError
        If any input validation fails.
    """
    if not callable(target_distribution):
        raise ValueError("target_distribution must be a callable function.")
    if not callable(proposal_distribution):
        raise ValueError("proposal_distribution must be a callable function.")
    if not isinstance(initial_state, np.ndarray):
        raise ValueError("initial_state must be a numpy array.")
    if not isinstance(n_iterations, int) or n_iterations <= 0:
        raise ValueError("n_iterations must be a positive integer.")
    if not isinstance(burn_in, int) or burn_in < 0:
        raise ValueError("burn_in must be a non-negative integer.")
    if burn_in >= n_iterations:
        raise ValueError("burn_in must be less than n_iterations.")

def metropolis_hastings_step(
    current_state: np.ndarray,
    target_distribution: Callable,
    proposal_distribution: Callable
) -> np.ndarray:
    """
    Perform a single Metropolis-Hastings step.

    Parameters
    ----------
    current_state : np.ndarray
        Current state of the Markov chain.
    target_distribution : Callable
        The target distribution to sample from.
    proposal_distribution : Callable
        The proposal distribution for the Markov chain.

    Returns
    ------
    np.ndarray
        New state of the Markov chain.
    """
    proposed_state = proposal_distribution(current_state)
    current_prob = target_distribution(current_state)
    proposed_prob = target_distribution(proposed_state)

    acceptance_ratio = proposed_prob / current_prob
    if np.random.rand() < acceptance_ratio:
        return proposed_state
    else:
        return current_state

def markov_chain_monte_carlo_fit(
    target_distribution: Callable,
    proposal_distribution: Callable,
    initial_state: np.ndarray,
    n_iterations: int = 10000,
    burn_in: int = 1000,
    thin: int = 1
) -> Dict[str, Any]:
    """
    Run a Markov Chain Monte Carlo simulation using the Metropolis-Hastings algorithm.

    Parameters
    ----------
    target_distribution : Callable
        The target distribution to sample from.
    proposal_distribution : Callable
        The proposal distribution for the Markov chain.
    initial_state : np.ndarray
        Initial state of the Markov chain.
    n_iterations : int, optional
        Number of iterations to run the MCMC algorithm (default is 10000).
    burn_in : int, optional
        Number of initial samples to discard (default is 1000).
    thin : int, optional
        Thinning interval to reduce autocorrelation (default is 1).

    Returns
    ------
    Dict[str, Any]
        A dictionary containing the results of the MCMC simulation.
    """
    validate_inputs(target_distribution, proposal_distribution, initial_state, n_iterations, burn_in)

    chain = []
    current_state = initial_state

    for _ in range(n_iterations):
        current_state = metropolis_hastings_step(current_state, target_distribution, proposal_distribution)
        if _ >= burn_in and (_ - burn_in) % thin == 0:
            chain.append(current_state)

    return {
        "result": np.array(chain),
        "params_used": {
            "n_iterations": n_iterations,
            "burn_in": burn_in,
            "thin": thin
        },
        "warnings": []
    }

def calculate_metrics(
    chain: np.ndarray,
    target_distribution: Callable
) -> Dict[str, float]:
    """
    Calculate metrics for the MCMC results.

    Parameters
    ----------
    chain : np.ndarray
        The Markov chain samples.
    target_distribution : Callable
        The target distribution to sample from.

    Returns
    ------
    Dict[str, float]
        A dictionary containing the calculated metrics.
    """
    acceptance_rate = np.mean(chain[:-1] != chain[1:])
    mean_estimate = np.mean(chain, axis=0)
    variance_estimate = np.var(chain, axis=0)

    return {
        "acceptance_rate": acceptance_rate,
        "mean_estimate": mean_estimate,
        "variance_estimate": variance_estimate
    }

def markov_chain_monte_carlo_compute(
    target_distribution: Callable,
    proposal_distribution: Callable,
    initial_state: np.ndarray,
    n_iterations: int = 10000,
    burn_in: int = 1000,
    thin: int = 1
) -> Dict[str, Any]:
    """
    Run a Markov Chain Monte Carlo simulation and compute metrics.

    Parameters
    ----------
    target_distribution : Callable
        The target distribution to sample from.
    proposal_distribution : Callable
        The proposal distribution for the Markov chain.
    initial_state : np.ndarray
        Initial state of the Markov chain.
    n_iterations : int, optional
        Number of iterations to run the MCMC algorithm (default is 10000).
    burn_in : int, optional
        Number of initial samples to discard (default is 1000).
    thin : int, optional
        Thinning interval to reduce autocorrelation (default is 1).

    Returns
    ------
    Dict[str, Any]
        A dictionary containing the results and metrics of the MCMC simulation.
    """
    result = markov_chain_monte_carlo_fit(
        target_distribution, proposal_distribution, initial_state,
        n_iterations, burn_in, thin
    )

    metrics = calculate_metrics(result["result"], target_distribution)

    return {
        "result": result["result"],
        "metrics": metrics,
        "params_used": result["params_used"],
        "warnings": result["warnings"]
    }

# Example usage:
if __name__ == "__main__":
    # Define a simple target distribution (e.g., Gaussian)
    def gaussian_distribution(x: np.ndarray) -> float:
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    # Define a simple proposal distribution (e.g., Gaussian random walk)
    def gaussian_proposal(current_state: np.ndarray) -> np.ndarray:
        return current_state + np.random.normal(0, 1)

    # Initial state
    initial_state = np.array([0.0])

    # Run MCMC
    result = markov_chain_monte_carlo_compute(
        target_distribution=gaussian_distribution,
        proposal_distribution=gaussian_proposal,
        initial_state=initial_state
    )

    print(result)

################################################################################
# metropolis_hastings
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(
    target_distribution: Callable[[np.ndarray], float],
    proposal_distribution: Callable[[np.ndarray, np.ndarray], float],
    initial_state: np.ndarray,
    n_samples: int,
    burn_in: int = 0
) -> None:
    """
    Validate the inputs for Metropolis-Hastings algorithm.

    Parameters
    ----------
    target_distribution : Callable[[np.ndarray], float]
        The target distribution to sample from.
    proposal_distribution : Callable[[np.ndarray, np.ndarray], float]
        The proposal distribution for generating candidate samples.
    initial_state : np.ndarray
        Initial state of the Markov chain.
    n_samples : int
        Number of samples to generate.
    burn_in : int, optional
        Number of initial samples to discard (burn-in period), by default 0

    Raises
    ------
    ValueError
        If any input is invalid.
    """
    if not callable(target_distribution):
        raise ValueError("target_distribution must be a callable function.")
    if not callable(proposal_distribution):
        raise ValueError("proposal_distribution must be a callable function.")
    if not isinstance(initial_state, np.ndarray):
        raise ValueError("initial_state must be a numpy array.")
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if not isinstance(burn_in, int) or burn_in < 0:
        raise ValueError("burn_in must be a non-negative integer.")

def metropolis_hastings_fit(
    target_distribution: Callable[[np.ndarray], float],
    proposal_distribution: Callable[[np.ndarray, np.ndarray], float],
    initial_state: np.ndarray,
    n_samples: int,
    burn_in: int = 0,
    thinning: int = 1,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform Metropolis-Hastings algorithm to sample from a target distribution.

    Parameters
    ----------
    target_distribution : Callable[[np.ndarray], float]
        The target distribution to sample from.
    proposal_distribution : Callable[[np.ndarray, np.ndarray], float]
        The proposal distribution for generating candidate samples.
    initial_state : np.ndarray
        Initial state of the Markov chain.
    n_samples : int
        Number of samples to generate.
    burn_in : int, optional
        Number of initial samples to discard (burn-in period), by default 0
    thinning : int, optional
        Thinning interval for the samples, by default 1
    random_seed : Optional[int], optional
        Random seed for reproducibility, by default None

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the results, metrics, parameters used, and warnings.
    """
    validate_inputs(target_distribution, proposal_distribution, initial_state, n_samples, burn_in)

    if random_seed is not None:
        np.random.seed(random_seed)

    current_state = initial_state.copy()
    samples = []
    acceptance_rate = 0

    for i in range(n_samples + burn_in):
        # Generate a candidate sample
        candidate_state = np.random.normal(current_state, scale=1.0)  # Simplified proposal

        # Calculate the acceptance probability
        current_prob = target_distribution(current_state)
        candidate_prob = target_distribution(candidate_state)

        acceptance_prob = min(1, candidate_prob / current_prob)

        # Accept or reject the candidate
        if np.random.rand() < acceptance_prob:
            current_state = candidate_state
            acceptance_rate += 1

        # Store the sample if it's after burn-in and thinning
        if i >= burn_in and (i - burn_in) % thinning == 0:
            samples.append(current_state.copy())

    acceptance_rate /= (n_samples + burn_in)

    return {
        "result": np.array(samples),
        "metrics": {"acceptance_rate": acceptance_rate},
        "params_used": {
            "target_distribution": target_distribution.__name__,
            "proposal_distribution": proposal_distribution.__name__,
            "initial_state": initial_state.tolist(),
            "n_samples": n_samples,
            "burn_in": burn_in,
            "thinning": thinning,
            "random_seed": random_seed
        },
        "warnings": []
    }

# Example usage:
if __name__ == "__main__":
    # Define a simple target distribution (e.g., Gaussian)
    def gaussian_distribution(x: np.ndarray) -> float:
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    # Define a simple proposal distribution (e.g., Gaussian)
    def gaussian_proposal(current: np.ndarray, candidate: np.ndarray) -> float:
        return np.exp(-0.5 * (candidate - current)**2) / np.sqrt(2 * np.pi)

    # Initial state
    initial_state = np.array([0.0])

    # Run Metropolis-Hastings
    result = metropolis_hastings_fit(
        target_distribution=gaussian_distribution,
        proposal_distribution=gaussian_proposal,
        initial_state=initial_state,
        n_samples=1000,
        burn_in=100,
        thinning=1,
        random_seed=42
    )

    print(result)

################################################################################
# gibbs_sampling
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def gibbs_sampling_fit(
    data: np.ndarray,
    n_iter: int = 1000,
    burn_in: int = 500,
    thin: int = 1,
    initial_state: Optional[np.ndarray] = None,
    target_distribution: Callable[[np.ndarray], float] = lambda x: np.prod(np.exp(-0.5 * np.sum(x**2))),
    proposal_distribution: Callable[[np.ndarray], float] = lambda x: np.prod(np.random.normal(0, 1, size=x.shape)),
    acceptance_criterion: Callable[[float, float], bool] = lambda old, new: new >= old,
    metrics: Optional[Dict[str, Callable[[np.ndarray], float]]] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform Gibbs sampling for Bayesian inference.

    Parameters:
    -----------
    data : np.ndarray
        Input data for the sampling process.
    n_iter : int, optional
        Number of iterations to run (default: 1000).
    burn_in : int, optional
        Number of iterations to discard as burn-in (default: 500).
    thin : int, optional
        Thinning interval (default: 1).
    initial_state : np.ndarray, optional
        Initial state for the Markov chain (default: None).
    target_distribution : Callable[[np.ndarray], float], optional
        Target distribution function (default: standard normal).
    proposal_distribution : Callable[[np.ndarray], float], optional
        Proposal distribution function (default: standard normal).
    acceptance_criterion : Callable[[float, float], bool], optional
        Acceptance criterion function (default: Metropolis-Hastings).
    metrics : Dict[str, Callable[[np.ndarray], float]], optional
        Dictionary of metrics to compute (default: None).
    random_state : Optional[int], optional
        Random seed for reproducibility (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    if random_state is not None:
        np.random.seed(random_state)

    if initial_state is None:
        initial_state = np.zeros(data.shape[1])

    n_samples = int((n_iter - burn_in) / thin)
    samples = np.zeros((n_samples, data.shape[1]))
    current_state = initial_state.copy()

    for i in range(n_iter):
        proposed_state = proposal_distribution(current_state)
        old_log_prob = np.log(target_distribution(current_state))
        new_log_prob = np.log(target_distribution(proposed_state))

        if acceptance_criterion(old_log_prob, new_log_prob):
            current_state = proposed_state

        if i >= burn_in and (i - burn_in) % thin == 0:
            samples[int((i - burn_in) / thin)] = current_state

    result = {
        "samples": samples,
        "acceptance_rate": np.mean([acceptance_criterion(np.log(target_distribution(current_state)), np.log(target_distribution(proposal_distribution(current_state)))) for _ in range(100)])
    }

    metrics_result = {}
    if metrics is not None:
        for name, metric_func in metrics.items():
            metrics_result[name] = metric_func(samples)

    params_used = {
        "n_iter": n_iter,
        "burn_in": burn_in,
        "thin": thin,
        "initial_state": initial_state.tolist(),
        "random_state": random_state
    }

    warnings = []
    if np.any(np.isnan(samples)):
        warnings.append("NaN values detected in samples.")

    return {
        "result": result,
        "metrics": metrics_result,
        "params_used": params_used,
        "warnings": warnings
    }

def validate_input(data: np.ndarray) -> None:
    """
    Validate the input data.

    Parameters:
    -----------
    data : np.ndarray
        Input data to validate.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values.")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values.")

def compute_metrics(samples: np.ndarray, metrics: Dict[str, Callable[[np.ndarray], float]]) -> Dict[str, float]:
    """
    Compute the specified metrics on the samples.

    Parameters:
    -----------
    samples : np.ndarray
        Samples from the Gibbs sampling process.
    metrics : Dict[str, Callable[[np.ndarray], float]]
        Dictionary of metrics to compute.

    Returns:
    --------
    Dict[str, float]
        Dictionary of computed metrics.
    """
    return {name: metric_func(samples) for name, metric_func in metrics.items()}

def check_convergence(samples: np.ndarray, threshold: float = 0.1) -> bool:
    """
    Check for convergence of the samples.

    Parameters:
    -----------
    samples : np.ndarray
        Samples from the Gibbs sampling process.
    threshold : float, optional
        Threshold for convergence (default: 0.1).

    Returns:
    --------
    bool
        True if the samples have converged, False otherwise.
    """
    mean_samples = np.mean(samples, axis=0)
    std_samples = np.std(samples, axis=0)
    return np.all(std_samples / mean_samples < threshold) if np.any(mean_samples != 0) else True

################################################################################
# hamiltonian_monte_carlo
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def hamiltonian_monte_carlo_fit(
    log_probability: Callable[[np.ndarray], float],
    initial_position: np.ndarray,
    step_size: float = 0.1,
    num_steps: int = 1000,
    burn_in: int = 500,
    thin: int = 1,
    metric: str = 'euclidean',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    solver: str = 'leapfrog',
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """
    Perform Hamiltonian Monte Carlo sampling.

    Parameters:
    -----------
    log_probability : callable
        The log probability function of the target distribution.
    initial_position : np.ndarray
        Initial position in parameter space.
    step_size : float, optional
        Step size for the integrator (default: 0.1).
    num_steps : int, optional
        Number of steps to take (default: 1000).
    burn_in : int, optional
        Number of steps to discard as burn-in (default: 500).
    thin : int, optional
        Thinning interval (default: 1).
    metric : str or callable, optional
        Metric for the leapfrog integrator ('euclidean', 'manhattan') or custom callable (default: 'euclidean').
    normalizer : callable, optional
        Normalization function for the parameters (default: None).
    custom_metric : callable, optional
        Custom metric function for the leapfrog integrator (default: None).
    solver : str, optional
        Solver for the Hamiltonian dynamics ('leapfrog', 'rk4') (default: 'leapfrog').
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(log_probability, initial_position)

    # Initialize samples
    samples = []
    current_position = initial_position.copy()

    for step in range(num_steps + burn_in):
        # Sample momentum
        current_momentum = np.random.normal(size=current_position.shape)

        # Perform Hamiltonian dynamics
        new_position, new_momentum = _hamiltonian_dynamics(
            log_probability,
            current_position,
            current_momentum,
            step_size,
            solver,
            metric,
            custom_metric
        )

        # Metropolis-Hastings acceptance step
        log_acceptance = _metropolis_hastings(
            log_probability,
            current_position,
            new_position,
            current_momentum
        )

        if np.log(np.random.rand()) < log_acceptance:
            current_position = new_position

        # Store samples after burn-in and thinning
        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(current_position.copy())

    # Calculate metrics
    metrics = _calculate_metrics(samples)

    return {
        'result': np.array(samples),
        'metrics': metrics,
        'params_used': {
            'step_size': step_size,
            'num_steps': num_steps,
            'burn_in': burn_in,
            'thin': thin,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

def _validate_inputs(log_probability: Callable[[np.ndarray], float], initial_position: np.ndarray) -> None:
    """
    Validate the inputs for Hamiltonian Monte Carlo.

    Parameters:
    -----------
    log_probability : callable
        The log probability function of the target distribution.
    initial_position : np.ndarray
        Initial position in parameter space.

    Raises:
    -------
    ValueError
        If inputs are invalid.
    """
    if not callable(log_probability):
        raise ValueError("log_probability must be a callable function.")
    if not isinstance(initial_position, np.ndarray):
        raise ValueError("initial_position must be a numpy array.")
    if np.any(np.isnan(initial_position)) or np.any(np.isinf(initial_position)):
        raise ValueError("initial_position must not contain NaN or inf values.")

def _hamiltonian_dynamics(
    log_probability: Callable[[np.ndarray], float],
    position: np.ndarray,
    momentum: np.ndarray,
    step_size: float,
    solver: str,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> tuple:
    """
    Perform Hamiltonian dynamics using the specified solver.

    Parameters:
    -----------
    log_probability : callable
        The log probability function of the target distribution.
    position : np.ndarray
        Current position in parameter space.
    momentum : np.ndarray
        Current momentum.
    step_size : float
        Step size for the integrator.
    solver : str
        Solver for the Hamiltonian dynamics ('leapfrog', 'rk4').
    metric : str or callable
        Metric for the leapfrog integrator ('euclidean', 'manhattan') or custom callable.
    custom_metric : callable, optional
        Custom metric function for the leapfrog integrator (default: None).

    Returns:
    --------
    tuple
        New position and momentum after Hamiltonian dynamics.
    """
    if solver == 'leapfrog':
        return _leapfrog_integrator(log_probability, position, momentum, step_size, metric, custom_metric)
    elif solver == 'rk4':
        return _rk4_integrator(log_probability, position, momentum, step_size)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _leapfrog_integrator(
    log_probability: Callable[[np.ndarray], float],
    position: np.ndarray,
    momentum: np.ndarray,
    step_size: float,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> tuple:
    """
    Perform leapfrog integration for Hamiltonian dynamics.

    Parameters:
    -----------
    log_probability : callable
        The log probability function of the target distribution.
    position : np.ndarray
        Current position in parameter space.
    momentum : np.ndarray
        Current momentum.
    step_size : float
        Step size for the integrator.
    metric : str or callable
        Metric for the leapfrog integrator ('euclidean', 'manhattan') or custom callable.
    custom_metric : callable, optional
        Custom metric function for the leapfrog integrator (default: None).

    Returns:
    --------
    tuple
        New position and momentum after leapfrog integration.
    """
    # Half step for momentum
    gradient = _compute_gradient(log_probability, position)
    if custom_metric is not None:
        momentum += 0.5 * step_size * custom_metric(position, gradient)
    elif isinstance(metric, str):
        if metric == 'euclidean':
            momentum += 0.5 * step_size * gradient
        elif metric == 'manhattan':
            momentum += 0.5 * step_size * np.sign(gradient)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        momentum += 0.5 * step_size * metric(position, gradient)

    # Full step for position
    position += step_size * momentum

    # Half step for momentum
    gradient = _compute_gradient(log_probability, position)
    if custom_metric is not None:
        momentum += 0.5 * step_size * custom_metric(position, gradient)
    elif isinstance(metric, str):
        if metric == 'euclidean':
            momentum += 0.5 * step_size * gradient
        elif metric == 'manhattan':
            momentum += 0.5 * step_size * np.sign(gradient)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        momentum += 0.5 * step_size * metric(position, gradient)

    return position, momentum

def _rk4_integrator(
    log_probability: Callable[[np.ndarray], float],
    position: np.ndarray,
    momentum: np.ndarray,
    step_size: float
) -> tuple:
    """
    Perform RK4 integration for Hamiltonian dynamics.

    Parameters:
    -----------
    log_probability : callable
        The log probability function of the target distribution.
    position : np.ndarray
        Current position in parameter space.
    momentum : np.ndarray
        Current momentum.
    step_size : float
        Step size for the integrator.

    Returns:
    --------
    tuple
        New position and momentum after RK4 integration.
    """
    k1_momentum = _compute_gradient(log_probability, position)
    k1_position = momentum

    k2_momentum = _compute_gradient(log_probability, position + 0.5 * step_size * k1_position)
    k2_position = momentum + 0.5 * step_size * k1_momentum

    k3_momentum = _compute_gradient(log_probability, position + 0.5 * step_size * k2_position)
    k3_position = momentum + 0.5 * step_size * k2_momentum

    k4_momentum = _compute_gradient(log_probability, position + step_size * k3_position)
    k4_position = momentum + step_size * k3_momentum

    new_momentum = momentum + (step_size / 6) * (k1_momentum + 2 * k2_momentum + 2 * k3_momentum + k4_momentum)
    new_position = position + (step_size / 6) * (k1_position + 2 * k2_position + 2 * k3_position + k4_position)

    return new_position, new_momentum

def _compute_gradient(log_probability: Callable[[np.ndarray], float], position: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the log probability function.

    Parameters:
    -----------
    log_probability : callable
        The log probability function of the target distribution.
    position : np.ndarray
        Current position in parameter space.

    Returns:
    --------
    np.ndarray
        Gradient of the log probability function at the given position.
    """
    epsilon = 1e-6
    gradient = np.zeros_like(position)
    for i in range(len(position)):
        position_plus = position.copy()
        position_plus[i] += epsilon
        position_minus = position.copy()
        position_minus[i] -= epsilon
        gradient[i] = (log_probability(position_plus) - log_probability(position_minus)) / (2 * epsilon)
    return gradient

def _metropolis_hastings(
    log_probability: Callable[[np.ndarray], float],
    current_position: np.ndarray,
    new_position: np.ndarray,
    momentum: np.ndarray
) -> float:
    """
    Perform Metropolis-Hastings acceptance step.

    Parameters:
    -----------
    log_probability : callable
        The log probability function of the target distribution.
    current_position : np.ndarray
        Current position in parameter space.
    new_position : np.ndarray
        New position proposed by Hamiltonian dynamics.
    momentum : np.ndarray
        Current momentum.

    Returns:
    --------
    float
        Log acceptance probability.
    """
    current_log_prob = log_probability(current_position)
    new_log_prob = log_probability(new_position)

    # Hamiltonian energy
    current_energy = -current_log_prob + 0.5 * np.dot(momentum, momentum)
    new_energy = -new_log_prob + 0.5 * np.dot(momentum, momentum)

    log_acceptance = new_log_prob - current_log_prob
    return log_acceptance

def _calculate_metrics(samples: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics for the samples.

    Parameters:
    -----------
    samples : np.ndarray
        Array of samples from Hamiltonian Monte Carlo.

    Returns:
    --------
    Dict[str, float]
        Dictionary of metrics.
    """
    samples = np.array(samples)
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)

    return {
        'mean': mean.tolist(),
        'std': std.tolist()
    }

################################################################################
# variational_inference
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(data: np.ndarray, prior: np.ndarray,
                   likelihood_func: Callable, metric_func: Callable) -> None:
    """
    Validate input data and functions.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    prior : np.ndarray
        Prior distribution parameters.
    likelihood_func : Callable
        Likelihood function callable.
    metric_func : Callable
        Metric function callable.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")
    if not callable(likelihood_func) or not callable(metric_func):
        raise ValueError("Likelihood and metric functions must be callables.")

def compute_criterion(data: np.ndarray, params: Dict,
                     likelihood_func: Callable) -> float:
    """
    Compute the variational inference criterion.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    params : Dict
        Current parameters dictionary.
    likelihood_func : Callable
        Likelihood function callable.

    Returns
    ------
    float
        Computed criterion value.
    """
    return -np.mean(likelihood_func(data, **params))

def estimate_parameters(data: np.ndarray, prior: np.ndarray,
                        likelihood_func: Callable, solver: str,
                        tol: float = 1e-6, max_iter: int = 1000) -> Dict:
    """
    Estimate parameters using the specified solver.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    prior : np.ndarray
        Prior distribution parameters.
    likelihood_func : Callable
        Likelihood function callable.
    solver : str
        Solver method ('gradient_descent', 'newton', etc.).
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.

    Returns
    ------
    Dict
        Estimated parameters.
    """
    params = {'mean': np.mean(data, axis=0), 'cov': np.cov(data.T)}
    criterion_history = []

    for _ in range(max_iter):
        old_params = params.copy()
        if solver == 'gradient_descent':
            grad = compute_gradient(data, params, likelihood_func)
            params['mean'] -= 0.01 * grad['mean']
            params['cov'] -= 0.01 * grad['cov']
        elif solver == 'newton':
            hess = compute_hessian(data, params, likelihood_func)
            grad = compute_gradient(data, params, likelihood_func)
            params['mean'] -= np.linalg.solve(hess['mean'], grad['mean'])
            params['cov'] -= np.linalg.solve(hess['cov'], grad['cov'])

        criterion = compute_criterion(data, params, likelihood_func)
        criterion_history.append(criterion)

        if np.linalg.norm(np.array(list(params.values())) -
                         np.array(list(old_params.values()))) < tol:
            break

    return {'params': params, 'criterion_history': criterion_history}

def compute_metrics(data: np.ndarray, params: Dict,
                    metric_func: Callable) -> Dict:
    """
    Compute metrics for the estimated parameters.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    params : Dict
        Estimated parameters dictionary.
    metric_func : Callable
        Metric function callable.

    Returns
    ------
    Dict
        Computed metrics.
    """
    return {'metric': metric_func(data, **params)}

def variational_inference_fit(data: np.ndarray, prior: np.ndarray,
                             likelihood_func: Callable = None,
                             metric_func: Callable = None,
                             solver: str = 'gradient_descent',
                             normalization: str = 'none',
                             tol: float = 1e-6,
                             max_iter: int = 1000) -> Dict:
    """
    Perform variational inference with configurable parameters.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    prior : np.ndarray
        Prior distribution parameters.
    likelihood_func : Callable, optional
        Likelihood function callable. Default is Gaussian likelihood.
    metric_func : Callable, optional
        Metric function callable. Default is MSE.
    solver : str, optional
        Solver method ('gradient_descent', 'newton'). Default is 'gradient_descent'.
    normalization : str, optional
        Normalization method ('none', 'standard'). Default is 'none'.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.

    Returns
    ------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Default functions if not provided
    if likelihood_func is None:
        def likelihood_func(data, mean, cov):
            n = data.shape[1]
            det_cov = np.linalg.det(cov)
            inv_cov = np.linalg.inv(cov)
            diff = data - mean
            return -0.5 * (n * np.log(2 * np.pi) + np.log(det_cov) +
                          np.sum(diff @ inv_cov * diff, axis=1))

    if metric_func is None:
        def metric_func(data, mean, cov):
            return np.mean((data - mean) ** 2)

    # Validate inputs
    validate_inputs(data, prior, likelihood_func, metric_func)

    # Normalize data if specified
    if normalization == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        data = (data - mean) / std
    elif normalization == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        data = (data - median) / iqr

    # Estimate parameters
    result = estimate_parameters(data, prior, likelihood_func,
                                solver, tol, max_iter)

    # Compute metrics
    metrics = compute_metrics(data, result['params'], metric_func)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'solver': solver,
            'normalization': normalization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

def compute_gradient(data: np.ndarray, params: Dict,
                    likelihood_func: Callable) -> Dict:
    """
    Compute the gradient of the criterion with respect to parameters.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    params : Dict
        Current parameters dictionary.
    likelihood_func : Callable
        Likelihood function callable.

    Returns
    ------
    Dict
        Gradient values for each parameter.
    """
    eps = 1e-8
    grad_mean = np.zeros_like(params['mean'])
    grad_cov = np.zeros_like(params['cov'])

    for i in range(len(params['mean'])):
        params_plus = params.copy()
        params_plus['mean'][i] += eps
        criterion_plus = compute_criterion(data, params_plus, likelihood_func)

        params_minus = params.copy()
        params_minus['mean'][i] -= eps
        criterion_minus = compute_criterion(data, params_minus, likelihood_func)

        grad_mean[i] = (criterion_plus - criterion_minus) / (2 * eps)

    for i in range(params['cov'].shape[0]):
        for j in range(params['cov'].shape[1]):
            params_plus = params.copy()
            params_plus['cov'][i, j] += eps
            criterion_plus = compute_criterion(data, params_plus, likelihood_func)

            params_minus = params.copy()
            params_minus['cov'][i, j] -= eps
            criterion_minus = compute_criterion(data, params_minus, likelihood_func)

            grad_cov[i, j] = (criterion_plus - criterion_minus) / (2 * eps)

    return {'mean': grad_mean, 'cov': grad_cov}

def compute_hessian(data: np.ndarray, params: Dict,
                   likelihood_func: Callable) -> Dict:
    """
    Compute the Hessian of the criterion with respect to parameters.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    params : Dict
        Current parameters dictionary.
    likelihood_func : Callable
        Likelihood function callable.

    Returns
    ------
    Dict
        Hessian values for each parameter.
    """
    eps = 1e-8
    hess_mean = np.zeros((len(params['mean']), len(params['mean'])))
    hess_cov = np.zeros((params['cov'].shape[0], params['cov'].shape[1],
                        params['cov'].shape[0], params['cov'].shape[1]))

    for i in range(len(params['mean'])):
        for j in range(len(params['mean'])):
            params_plus_i = params.copy()
            params_plus_i['mean'][i] += eps
            grad_plus_i = compute_gradient(data, params_plus_i, likelihood_func)['mean']

            params_minus_i = params.copy()
            params_minus_i['mean'][i] -= eps
            grad_minus_i = compute_gradient(data, params_minus_i, likelihood_func)['mean']

            hess_mean[i, j] = (grad_plus_i[j] - grad_minus_i[j]) / (2 * eps)

    for i in range(params['cov'].shape[0]):
        for j in range(params['cov'].shape[1]):
            for k in range(params['cov'].shape[0]):
                for l in range(params['cov'].shape[1]):
                    params_plus_ij = params.copy()
                    params_plus_ij['cov'][i, j] += eps
                    grad_plus_ij = compute_gradient(data, params_plus_ij,
                                                   likelihood_func)['cov']

                    params_minus_ij = params.copy()
                    params_minus_ij['cov'][i, j] -= eps
                    grad_minus_ij = compute_gradient(data, params_minus_ij,
                                                    likelihood_func)['cov']

                    hess_cov[i, j, k, l] = (grad_plus_ij[k, l] -
                                           grad_minus_ij[k, l]) / (2 * eps)

    return {'mean': hess_mean, 'cov': hess_cov}

################################################################################
# bayesian_networks
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def bayesian_networks_fit(
    data: np.ndarray,
    structure: np.ndarray,
    prior: Optional[np.ndarray] = None,
    likelihood: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.prod(np.exp(-0.5 * (x - y)**2)),
    inference_method: str = 'exact',
    max_iter: int = 1000,
    tol: float = 1e-6,
    normalize_data: bool = True,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Fit a Bayesian Network to the given data.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    structure : np.ndarray
        Adjacency matrix representing the network structure.
    prior : Optional[np.ndarray], default=None
        Prior distribution over parameters.
    likelihood : Callable, default=gaussian_likelihood
        Likelihood function to use.
    inference_method : str, default='exact'
        Inference method: 'exact', 'variational', or 'mc'.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-6
        Tolerance for convergence.
    normalize_data : bool, default=True
        Whether to normalize the input data.
    custom_metric : Optional[Callable], default=None
        Custom metric function to evaluate the fit.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, structure)

    # Normalize data if required
    if normalize_data:
        data = _normalize_data(data)

    # Initialize parameters
    params = _initialize_parameters(data, structure, prior)

    # Perform inference based on the chosen method
    if inference_method == 'exact':
        result = _exact_inference(data, structure, params, likelihood, max_iter, tol)
    elif inference_method == 'variational':
        result = _variational_inference(data, structure, params, likelihood, max_iter, tol)
    elif inference_method == 'mc':
        result = _markov_chain_inference(data, structure, params, likelihood, max_iter, tol)
    else:
        raise ValueError("Invalid inference method specified.")

    # Calculate metrics
    metrics = _calculate_metrics(data, result, custom_metric)

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params,
        "warnings": []
    }

def _validate_inputs(data: np.ndarray, structure: np.ndarray) -> None:
    """Validate the input data and network structure."""
    if not isinstance(data, np.ndarray) or not isinstance(structure, np.ndarray):
        raise TypeError("Data and structure must be numpy arrays.")
    if data.ndim != 2 or structure.ndim != 2:
        raise ValueError("Data must be 2D and structure must be a square matrix.")
    if data.shape[1] != structure.shape[0]:
        raise ValueError("Number of features in data must match the size of the structure matrix.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")

def _normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalize the input data."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-8)

def _initialize_parameters(data: np.ndarray, structure: np.ndarray, prior: Optional[np.ndarray]) -> Dict[str, Any]:
    """Initialize the parameters for the Bayesian Network."""
    n_features = data.shape[1]
    params = {
        'weights': np.zeros((n_features, n_features)),
        'biases': np.zeros(n_features)
    }
    if prior is not None:
        params['prior'] = prior
    return params

def _exact_inference(data: np.ndarray, structure: np.ndarray, params: Dict[str, Any], likelihood: Callable,
                     max_iter: int, tol: float) -> np.ndarray:
    """Perform exact inference for the Bayesian Network."""
    # Placeholder for exact inference logic
    return np.zeros(data.shape[1])

def _variational_inference(data: np.ndarray, structure: np.ndarray, params: Dict[str, Any], likelihood: Callable,
                           max_iter: int, tol: float) -> np.ndarray:
    """Perform variational inference for the Bayesian Network."""
    # Placeholder for variational inference logic
    return np.zeros(data.shape[1])

def _markov_chain_inference(data: np.ndarray, structure: np.ndarray, params: Dict[str, Any], likelihood: Callable,
                            max_iter: int, tol: float) -> np.ndarray:
    """Perform Markov Chain Monte Carlo inference for the Bayesian Network."""
    # Placeholder for MCMC inference logic
    return np.zeros(data.shape[1])

def _calculate_metrics(data: np.ndarray, result: np.ndarray, custom_metric: Optional[Callable]) -> Dict[str, float]:
    """Calculate the metrics for the Bayesian Network fit."""
    metrics = {
        'mse': np.mean((data - result) ** 2),
        'mae': np.mean(np.abs(data - result))
    }
    if custom_metric is not None:
        metrics['custom'] = custom_metric(data, result)
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
    noise_variance: float = 1.0,
    solver: str = 'closed_form',
    max_iter: int = 1000,
    tol: float = 1e-6,
    metric: str = 'mse',
    normalize: bool = True,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Perform Bayesian Linear Regression with configurable options.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    prior_mean : Optional[np.ndarray]
        Prior mean for the coefficients
    prior_covariance : Optional[np.ndarray]
        Prior covariance matrix for the coefficients
    noise_variance : float
        Assumed variance of the noise
    solver : str
        Solver to use ('closed_form', 'gradient_descent')
    max_iter : int
        Maximum number of iterations for iterative solvers
    tol : float
        Tolerance for convergence
    metric : str
        Metric to compute ('mse', 'mae')
    normalize : bool
        Whether to normalize the features
    random_state : Optional[int]
        Random seed for reproducibility

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Posterior mean of coefficients
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in the computation
        - 'warnings': Any warnings encountered

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = bayesian_linear_regression_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Normalize features if requested
    X_normalized, X_mean, X_std = _normalize_features(X) if normalize else (X, None, None)

    # Set default prior if not provided
    n_features = X.shape[1]
    if prior_mean is None:
        prior_mean = np.zeros(n_features)
    if prior_covariance is None:
        prior_covariance = np.eye(n_features)

    # Choose solver
    if solver == 'closed_form':
        posterior_mean = _closed_form_solution(X_normalized, y, prior_mean, prior_covariance, noise_variance)
    elif solver == 'gradient_descent':
        posterior_mean = _gradient_descent_solution(
            X_normalized, y, prior_mean, prior_covariance, noise_variance,
            max_iter=max_iter, tol=tol, rng=rng
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    y_pred = X_normalized @ posterior_mean
    metrics = _compute_metrics(y, y_pred, metric=metric)

    # Return results
    return {
        'result': posterior_mean,
        'metrics': metrics,
        'params_used': {
            'prior_mean': prior_mean.tolist(),
            'prior_covariance': prior_covariance.tolist(),
            'noise_variance': noise_variance,
            'solver': solver,
            'max_iter': max_iter if solver == 'gradient_descent' else None,
            'tol': tol if solver == 'gradient_descent' else None,
            'metric': metric,
            'normalize': normalize
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

def _normalize_features(X: np.ndarray) -> tuple:
    """Normalize features to zero mean and unit variance."""
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0  # Avoid division by zero
    X_normalized = (X - X_mean) / X_std
    return X_normalized, X_mean, X_std

def _closed_form_solution(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    noise_variance: float
) -> np.ndarray:
    """Compute posterior mean using closed-form solution."""
    n_features = X.shape[1]
    posterior_covariance = np.linalg.inv(
        np.linalg.inv(prior_covariance) + (1/noise_variance) * X.T @ X
    )
    posterior_mean = posterior_covariance @ (
        (1/noise_variance) * X.T @ y + np.linalg.inv(prior_covariance) @ prior_mean
    )
    return posterior_mean

def _gradient_descent_solution(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    noise_variance: float,
    *,
    max_iter: int = 1000,
    tol: float = 1e-6,
    rng: np.random.RandomState
) -> np.ndarray:
    """Compute posterior mean using gradient descent."""
    n_features = X.shape[1]
    beta = rng.normal(size=n_features)
    prev_beta = np.zeros_like(beta)

    for _ in range(max_iter):
        gradient = (1/noise_variance) * X.T @ (X @ beta - y) + np.linalg.inv(prior_covariance) @ (beta - prior_mean)
        beta -= gradient  # Simple gradient descent step

        if np.linalg.norm(beta - prev_beta) < tol:
            break
        prev_beta = beta.copy()

    return beta

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = 'mse'
) -> Dict[str, float]:
    """Compute specified metrics."""
    metrics = {}

    if metric in ['mse', 'all']:
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    if metric in ['mae', 'all']:
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric in ['r2', 'all']:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

################################################################################
# bayesian_logistic_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def bayesian_logistic_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    prior_mean: Optional[np.ndarray] = None,
    prior_cov: Optional[np.ndarray] = None,
    n_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    normalize: str = 'standard',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'logloss',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Perform Bayesian logistic regression.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,) in {0, 1}
    prior_mean : np.ndarray, optional
        Prior mean for the coefficients
    prior_cov : np.ndarray, optional
        Prior covariance matrix for the coefficients
    n_iter : int, default=1000
        Number of iterations for optimization
    tol : float, default=1e-4
        Tolerance for convergence
    random_state : int, optional
        Random seed for reproducibility
    normalize : str, default='standard'
        Normalization method: 'none', 'standard', 'minmax', 'robust'
    metric : str or callable, default='logloss'
        Evaluation metric: 'mse', 'mae', 'r2', 'logloss' or custom callable
    solver : str, default='gradient_descent'
        Optimization algorithm: 'gradient_descent', 'newton', 'coordinate_descent'
    regularization : str, optional
        Regularization type: 'none', 'l1', 'l2', 'elasticnet'
    alpha : float, default=1.0
        Regularization strength
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': Estimated coefficients
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in the fit
        - 'warnings': Any warnings encountered

    Example:
    --------
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = bayesian_logistic_regression_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Normalize data
    X_normalized, normalizer = _normalize_data(X, method=normalize)

    # Initialize parameters
    n_features = X_normalized.shape[1]
    if prior_mean is None:
        prior_mean = np.zeros(n_features)
    if prior_cov is None:
        prior_cov = alpha * np.eye(n_features)

    # Initialize coefficients
    coef_ = rng.randn(n_features) if random_state is not None else np.random.randn(n_features)

    # Choose solver
    if solver == 'gradient_descent':
        coef_ = _gradient_descent(X_normalized, y, prior_mean, prior_cov,
                                 coef_, n_iter, tol, rng)
    elif solver == 'newton':
        coef_ = _newton_method(X_normalized, y, prior_mean, prior_cov,
                              coef_, n_iter, tol)
    elif solver == 'coordinate_descent':
        coef_ = _coordinate_descent(X_normalized, y, prior_mean, prior_cov,
                                   coef_, n_iter, tol, rng)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization == 'l1':
        coef_ = _apply_l1_penalty(coef_, alpha)
    elif regularization == 'l2':
        coef_ = _apply_l2_penalty(coef_, alpha)
    elif regularization == 'elasticnet':
        coef_ = _apply_elasticnet_penalty(coef_, alpha, l1_ratio)

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, coef_, metric)

    # Prepare output
    result = {
        'result': {'coef_': coef_, 'intercept_': 0.0},  # Intercept handled in prediction
        'metrics': metrics,
        'params_used': {
            'prior_mean': prior_mean.tolist(),
            'prior_cov': prior_cov.tolist(),
            'n_iter': n_iter,
            'tol': tol,
            'normalize': normalize,
            'metric': metric if isinstance(metric, str) else 'custom',
            'solver': solver,
            'regularization': regularization,
            'alpha': alpha,
            'l1_ratio': l1_ratio
        },
        'warnings': []
    }

    return result

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
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only 0 and 1 values")

def _normalize_data(X: np.ndarray, method: str = 'standard') -> tuple:
    """Normalize the input data."""
    if method == 'none':
        return X, None
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / std
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

    return X_normalized, {'mean': mean, 'std': std} if method == 'standard' else None

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    coef_init: np.ndarray,
    n_iter: int,
    tol: float,
    rng: np.random.RandomState
) -> np.ndarray:
    """Gradient descent optimization."""
    coef = coef_init.copy()
    learning_rate = 0.1
    prev_loss = float('inf')

    for _ in range(n_iter):
        # Compute gradient
        linear_pred = X @ coef
        prob = 1 / (1 + np.exp(-linear_pred))
        gradient = X.T @ (prob - y) + prior_cov @ (coef - prior_mean)

        # Update coefficients
        coef -= learning_rate * gradient

        # Check convergence
        current_loss = _compute_logloss(X, y, coef)
        if np.abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return coef

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    coef_init: np.ndarray,
    n_iter: int,
    tol: float
) -> np.ndarray:
    """Newton's method optimization."""
    coef = coef_init.copy()
    prev_loss = float('inf')

    for _ in range(n_iter):
        # Compute gradient and hessian
        linear_pred = X @ coef
        prob = 1 / (1 + np.exp(-linear_pred))
        gradient = X.T @ (prob - y) + prior_cov @ (coef - prior_mean)

        W = np.diag(prob * (1 - prob))
        hessian = X.T @ W @ X + prior_cov

        # Update coefficients
        coef -= np.linalg.solve(hessian, gradient)

        # Check convergence
        current_loss = _compute_logloss(X, y, coef)
        if np.abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return coef

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    coef_init: np.ndarray,
    n_iter: int,
    tol: float,
    rng: np.random.RandomState
) -> np.ndarray:
    """Coordinate descent optimization."""
    coef = coef_init.copy()
    prev_loss = float('inf')

    for _ in range(n_iter):
        for j in rng.permutation(X.shape[1]):
            # Compute residual
            linear_pred = X @ coef
            prob = 1 / (1 + np.exp(-linear_pred))
            residual = y - prob

            # Compute gradient for feature j
            X_j = X[:, j]
            gradient_j = -X_j.T @ residual + prior_cov[j, j] * (coef[j] - prior_mean[j])

            # Compute second derivative for feature j
            W_j = prob * (1 - prob)
            hessian_j = X_j.T @ W_j + prior_cov[j, j]

            # Update coefficient j
            coef[j] -= gradient_j / hessian_j

        # Check convergence
        current_loss = _compute_logloss(X, y, coef)
        if np.abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return coef

def _apply_l1_penalty(coef: np.ndarray, alpha: float) -> np.ndarray:
    """Apply L1 penalty."""
    return np.sign(coef) * np.maximum(np.abs(coef) - alpha, 0)

def _apply_l2_penalty(coef: np.ndarray, alpha: float) -> np.ndarray:
    """Apply L2 penalty."""
    return coef / (1 + alpha)

def _apply_elasticnet_penalty(coef: np.ndarray, alpha: float, l1_ratio: float) -> np.ndarray:
    """Apply ElasticNet penalty."""
    coef_l1 = _apply_l1_penalty(coef, alpha * l1_ratio)
    coef_l2 = _apply_l2_penalty(coef, alpha * (1 - l1_ratio))
    return coef_l1 + coef_l2 - coef

def _compute_logloss(X: np.ndarray, y: np.ndarray, coef: np.ndarray) -> float:
    """Compute log loss."""
    linear_pred = X @ coef
    prob = 1 / (1 + np.exp(-linear_pred))
    return -np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    linear_pred = X @ coef
    prob = 1 / (1 + np.exp(-linear_pred))
    y_pred = (prob >= 0.5).astype(int)

    metrics_dict = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics_dict['mse'] = np.mean((y - prob) ** 2)
        elif metric == 'mae':
            metrics_dict['mae'] = np.mean(np.abs(y - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y - prob) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            metrics_dict['r2'] = 1 - ss_res / (ss_tot + 1e-8)
        elif metric == 'logloss':
            metrics_dict['logloss'] = _compute_logloss(X, y, coef)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        metrics_dict['custom'] = metric(y, prob)

    return metrics_dict

################################################################################
# bayesian_optimization
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def bayesian_optimization_fit(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_iter: int = 100,
    acquisition_function: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = None,
    kernel_function: Callable[[np.ndarray, np.ndarray], float] = None,
    normalizer: str = 'standard',
    metric: str = 'mse',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_evals: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform Bayesian optimization to find the minimum of an objective function.

    Parameters:
    -----------
    objective_function : callable
        The objective function to minimize.
    bounds : np.ndarray
        Array of shape (n_features, 2) defining the search space for each parameter.
    n_iter : int, optional
        Number of iterations to run the optimization (default: 100).
    acquisition_function : callable, optional
        The acquisition function to use (default: None, uses expected improvement).
    kernel_function : callable, optional
        The kernel function for the Gaussian process (default: None, uses squared exponential).
    normalizer : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard').
    metric : str, optional
        Metric to evaluate the optimization ('mse', 'mae', 'r2', 'logloss') (default: 'mse').
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default: 'closed_form').
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
    max_evals : int, optional
        Maximum number of evaluations (default: 1000).
    random_state : int, optional
        Random seed for reproducibility (default: None).

    Returns:
    --------
    dict
        A dictionary containing the optimization results, metrics, parameters used, and warnings.
    """
    # Initialize random state
    if random_state is not None:
        np.random.seed(random_state)

    # Validate inputs
    _validate_inputs(objective_function, bounds, n_iter, acquisition_function, kernel_function)

    # Initialize Gaussian Process
    gp = _initialize_gaussian_process(kernel_function)

    # Normalize bounds
    normalized_bounds = _normalize_bounds(bounds, normalizer)

    # Run optimization
    results = _run_optimization(
        objective_function,
        normalized_bounds,
        n_iter,
        acquisition_function,
        gp,
        metric,
        solver,
        tol,
        max_evals
    )

    return results

def _validate_inputs(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_iter: int,
    acquisition_function: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], float]],
    kernel_function: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> None:
    """
    Validate the inputs for Bayesian optimization.

    Parameters:
    -----------
    objective_function : callable
        The objective function to minimize.
    bounds : np.ndarray
        Array of shape (n_features, 2) defining the search space for each parameter.
    n_iter : int
        Number of iterations to run the optimization.
    acquisition_function : callable, optional
        The acquisition function to use.
    kernel_function : callable, optional
        The kernel function for the Gaussian process.

    Raises:
    -------
    ValueError
        If any of the inputs are invalid.
    """
    if not callable(objective_function):
        raise ValueError("objective_function must be a callable.")
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must be a 2D array of shape (n_features, 2).")
    if n_iter <= 0:
        raise ValueError("n_iter must be a positive integer.")
    if acquisition_function is not None and not callable(acquisition_function):
        raise ValueError("acquisition_function must be a callable or None.")
    if kernel_function is not None and not callable(kernel_function):
        raise ValueError("kernel_function must be a callable or None.")

def _initialize_gaussian_process(
    kernel_function: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, Any]:
    """
    Initialize the Gaussian Process with the given kernel function.

    Parameters:
    -----------
    kernel_function : callable, optional
        The kernel function for the Gaussian process.

    Returns:
    --------
    dict
        A dictionary representing the initialized Gaussian Process.
    """
    if kernel_function is None:
        kernel_function = _squared_exponential_kernel
    return {'kernel': kernel_function, 'mean': 0.0}

def _squared_exponential_kernel(
    x1: np.ndarray,
    x2: np.ndarray
) -> float:
    """
    Squared exponential kernel function.

    Parameters:
    -----------
    x1 : np.ndarray
        First input vector.
    x2 : np.ndarray
        Second input vector.

    Returns:
    --------
    float
        The kernel value.
    """
    squared_dist = np.sum((x1 - x2) ** 2)
    return np.exp(-0.5 * squared_dist)

def _normalize_bounds(
    bounds: np.ndarray,
    normalizer: str
) -> np.ndarray:
    """
    Normalize the bounds based on the specified normalizer.

    Parameters:
    -----------
    bounds : np.ndarray
        Array of shape (n_features, 2) defining the search space for each parameter.
    normalizer : str
        Normalization method ('none', 'standard', 'minmax', 'robust').

    Returns:
    --------
    np.ndarray
        Normalized bounds.
    """
    if normalizer == 'none':
        return bounds
    elif normalizer == 'standard':
        mean = np.mean(bounds, axis=1, keepdims=True)
        std = np.std(bounds, axis=1, keepdims=True)
        return (bounds - mean) / std
    elif normalizer == 'minmax':
        min_val = np.min(bounds, axis=1, keepdims=True)
        max_val = np.max(bounds, axis=1, keepdims=True)
        return (bounds - min_val) / (max_val - min_val + 1e-8)
    elif normalizer == 'robust':
        median = np.median(bounds, axis=1, keepdims=True)
        iqr = np.subtract(*np.percentile(bounds, [75, 25], axis=1))
        return (bounds - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalizer: {normalizer}")

def _run_optimization(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_iter: int,
    acquisition_function: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], float]],
    gp: Dict[str, Any],
    metric: str,
    solver: str,
    tol: float,
    max_evals: int
) -> Dict[str, Any]:
    """
    Run the Bayesian optimization process.

    Parameters:
    -----------
    objective_function : callable
        The objective function to minimize.
    bounds : np.ndarray
        Normalized bounds for the search space.
    n_iter : int
        Number of iterations to run the optimization.
    acquisition_function : callable, optional
        The acquisition function to use.
    gp : dict
        Initialized Gaussian Process.
    metric : str
        Metric to evaluate the optimization.
    solver : str
        Solver to use.
    tol : float
        Tolerance for convergence.
    max_evals : int
        Maximum number of evaluations.

    Returns:
    --------
    dict
        A dictionary containing the optimization results, metrics, parameters used, and warnings.
    """
    # Initialize variables
    X = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_iter, bounds.shape[0]))
    y = np.array([objective_function(x) for x in X])

    # Initialize acquisition function
    if acquisition_function is None:
        acquisition_function = _expected_improvement

    # Run optimization iterations
    for i in range(n_iter):
        # Update Gaussian Process
        gp = _update_gaussian_process(gp, X[:i+1], y[:i+1])

        # Find next point to evaluate
        next_point = _find_next_point(gp, bounds, acquisition_function, solver)

        # Evaluate objective function
        y_next = objective_function(next_point)
        X = np.vstack([X, next_point])
        y = np.append(y, y_next)

        # Check convergence
        if i > 0 and np.abs(y[i] - y[i-1]) < tol:
            break

    # Calculate metrics
    metrics = _calculate_metrics(y, metric)

    return {
        'result': {'x_opt': X[-1], 'y_opt': y[-1]},
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

def _update_gaussian_process(
    gp: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, Any]:
    """
    Update the Gaussian Process with new data.

    Parameters:
    -----------
    gp : dict
        Initialized Gaussian Process.
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    y : np.ndarray
        Output data of shape (n_samples,).

    Returns:
    --------
    dict
        Updated Gaussian Process.
    """
    # Placeholder for actual GP update logic
    return gp

def _find_next_point(
    gp: Dict[str, Any],
    bounds: np.ndarray,
    acquisition_function: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    solver: str
) -> np.ndarray:
    """
    Find the next point to evaluate using the acquisition function.

    Parameters:
    -----------
    gp : dict
        Initialized Gaussian Process.
    bounds : np.ndarray
        Normalized bounds for the search space.
    acquisition_function : callable
        The acquisition function to use.
    solver : str
        Solver to use.

    Returns:
    --------
    np.ndarray
        The next point to evaluate.
    """
    # Placeholder for actual next point finding logic
    return np.random.uniform(bounds[:, 0], bounds[:, 1])

def _expected_improvement(
    x: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Expected improvement acquisition function.

    Parameters:
    -----------
    x : np.ndarray
        Point to evaluate.
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    y : np.ndarray
        Output data of shape (n_samples,).

    Returns:
    --------
    float
        The expected improvement value.
    """
    # Placeholder for actual expected improvement calculation
    return 0.0

def _calculate_metrics(
    y: np.ndarray,
    metric: str
) -> Dict[str, float]:
    """
    Calculate the specified metrics for the optimization results.

    Parameters:
    -----------
    y : np.ndarray
        Output data of shape (n_samples,).
    metric : str
        Metric to calculate ('mse', 'mae', 'r2', 'logloss').

    Returns:
    --------
    dict
        A dictionary containing the calculated metrics.
    """
    if metric == 'mse':
        return {'mse': np.mean((y - np.mean(y)) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(y - np.mean(y)))}
    elif metric == 'r2':
        ss_res = np.sum((y - np.mean(y)) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + np.sum((np.mean(y) - np.mean(y)) ** 2)
        return {'r2': 1 - ss_res / (ss_tot + 1e-8)}
    elif metric == 'logloss':
        return {'logloss': -np.mean(y * np.log(y + 1e-8) + (1 - y) * np.log(1 - y + 1e-8))}
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# empirical_bayes
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def validate_inputs(
    data: np.ndarray,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    likelihood_var: float
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Data contains infinite values")

    if not isinstance(prior_mean, np.ndarray):
        raise TypeError("Prior mean must be a numpy array")
    if prior_mean.ndim != 1:
        raise ValueError("Prior mean must be a 1D array")

    if not isinstance(prior_precision, np.ndarray):
        raise TypeError("Prior precision must be a numpy array")
    if prior_precision.ndim != 2:
        raise ValueError("Prior precision must be a 2D array")
    if prior_precision.shape[0] != prior_precision.shape[1]:
        raise ValueError("Prior precision must be a square matrix")
    if np.linalg.det(prior_precision) == 0:
        raise ValueError("Prior precision matrix must be invertible")

    if not isinstance(likelihood_var, (int, float)) or likelihood_var <= 0:
        raise ValueError("Likelihood variance must be a positive number")

def compute_posterior(
    data: np.ndarray,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    likelihood_var: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the posterior mean and precision."""
    n_samples, n_features = data.shape

    # Posterior precision
    posterior_precision = prior_precision + (1 / likelihood_var) * np.eye(n_features)

    # Posterior mean
    posterior_mean = np.linalg.solve(
        posterior_precision,
        prior_precision @ prior_mean + (1 / likelihood_var) * data.T @ np.ones(n_samples)
    )

    return posterior_mean, posterior_precision

def compute_metrics(
    data: np.ndarray,
    posterior_mean: np.ndarray,
    metric_func: Callable = None
) -> Dict[str, float]:
    """Compute metrics for the empirical Bayes results."""
    if metric_func is None:
        def default_metric(y_true, y_pred):
            return {
                'mse': np.mean((y_true - y_pred) ** 2),
                'mae': np.mean(np.abs(y_true - y_pred))
            }
        metric_func = default_metric

    predictions = data @ posterior_mean
    return metric_func(data, predictions)

def empirical_bayes_fit(
    data: np.ndarray,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    likelihood_var: float = 1.0,
    metric_func: Optional[Callable] = None,
    normalize_data: bool = False
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, object]]]:
    """
    Perform empirical Bayes inference.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    prior_mean : np.ndarray
        Prior mean vector of shape (n_features,)
    prior_precision : np.ndarray
        Prior precision matrix of shape (n_features, n_features)
    likelihood_var : float, optional
        Likelihood variance (default=1.0)
    metric_func : Callable, optional
        Custom metric function (default=None)
    normalize_data : bool, optional
        Whether to normalize the data (default=False)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(data, prior_mean, prior_precision, likelihood_var)

    # Normalize data if required
    if normalize_data:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        data = (data - mean) / std

    # Compute posterior
    posterior_mean, posterior_precision = compute_posterior(
        data, prior_mean, prior_precision, likelihood_var
    )

    # Compute metrics
    metrics = compute_metrics(data, posterior_mean, metric_func)

    return {
        'result': {
            'posterior_mean': posterior_mean,
            'posterior_precision': posterior_precision
        },
        'metrics': metrics,
        'params_used': {
            'likelihood_var': likelihood_var,
            'normalize_data': normalize_data
        },
        'warnings': []
    }

# Example usage:
"""
data = np.random.randn(100, 5)
prior_mean = np.zeros(5)
prior_precision = np.eye(5)

result = empirical_bayes_fit(
    data=data,
    prior_mean=prior_mean,
    prior_precision=prior_precision
)
"""

################################################################################
# hierarchical_models
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List, Any

def hierarchical_models_fit(
    data: np.ndarray,
    prior_means: Optional[np.ndarray] = None,
    prior_vars: Optional[np.ndarray] = None,
    likelihood_var: float = 1.0,
    solver: str = 'closed_form',
    metric: Union[str, Callable] = 'mse',
    normalization: str = 'none',
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Fit hierarchical Bayesian models with configurable parameters.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    prior_means : Optional[np.ndarray]
        Prior means for each feature
    prior_vars : Optional[np.ndarray]
        Prior variances for each feature
    likelihood_var : float
        Variance of the likelihood (default 1.0)
    solver : str
        Solver to use ('closed_form', 'gradient_descent')
    metric : Union[str, Callable]
        Metric to evaluate ('mse', 'mae', custom callable)
    normalization : str
        Normalization method ('none', 'standard', 'minmax')
    max_iter : int
        Maximum number of iterations (default 1000)
    tol : float
        Tolerance for convergence (default 1e-6)
    verbose : bool
        Whether to print progress (default False)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(data, prior_means, prior_vars)

    # Normalize data if needed
    normalized_data = _apply_normalization(data, normalization)

    # Set default priors if not provided
    prior_means = _set_default_priors(prior_means, data.shape[1])
    prior_vars = _set_default_priors(prior_vars, data.shape[1])

    # Initialize parameters
    params = _initialize_parameters(data.shape[1], prior_means, prior_vars)

    # Solve the model
    if solver == 'closed_form':
        result = _solve_closed_form(normalized_data, params, likelihood_var)
    elif solver == 'gradient_descent':
        result = _solve_gradient_descent(
            normalized_data, params, likelihood_var,
            max_iter=max_iter, tol=tol, verbose=verbose
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(
        normalized_data, result['posterior_means'],
        metric=metric
    )

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'solver': solver,
            'metric': metric,
            'normalization': normalization
        },
        'warnings': _check_warnings(result)
    }

    return output

def _validate_inputs(
    data: np.ndarray,
    prior_means: Optional[np.ndarray],
    prior_vars: Optional[np.ndarray]
) -> None:
    """Validate input data and priors."""
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array")

    if prior_means is not None and len(prior_means) != data.shape[1]:
        raise ValueError("Prior means must match number of features")

    if prior_vars is not None and len(prior_vars) != data.shape[1]:
        raise ValueError("Prior variances must match number of features")

def _apply_normalization(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply normalization to the data."""
    if method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _set_default_priors(
    priors: Optional[np.ndarray],
    n_features: int
) -> np.ndarray:
    """Set default priors if not provided."""
    if priors is None:
        return np.zeros(n_features)
    return priors

def _initialize_parameters(
    n_features: int,
    prior_means: np.ndarray,
    prior_vars: np.ndarray
) -> Dict[str, Any]:
    """Initialize model parameters."""
    return {
        'prior_means': prior_means,
        'prior_vars': prior_vars,
        'posterior_means': np.zeros(n_features),
        'posterior_vars': np.ones(n_features)
    }

def _solve_closed_form(
    data: np.ndarray,
    params: Dict[str, Any],
    likelihood_var: float
) -> Dict[str, Any]:
    """Solve hierarchical model using closed-form solution."""
    n_samples = data.shape[0]
    prior_means = params['prior_means']
    prior_vars = params['prior_vars']

    # Calculate posterior means and variances
    posterior_means = (prior_vars * np.sum(data, axis=0) + likelihood_var * prior_means * n_samples) / (
        prior_vars + likelihood_var * n_samples
    )
    posterior_vars = 1 / (1/prior_vars + n_samples/likelihood_var)

    return {
        'posterior_means': posterior_means,
        'posterior_vars': posterior_vars
    }

def _solve_gradient_descent(
    data: np.ndarray,
    params: Dict[str, Any],
    likelihood_var: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False
) -> Dict[str, Any]:
    """Solve hierarchical model using gradient descent."""
    n_samples = data.shape[0]
    prior_means = params['prior_means']
    prior_vars = params['prior_vars']

    posterior_means = np.zeros_like(prior_means)
    posterior_vars = np.ones_like(prior_vars)

    for i in range(max_iter):
        # Update posterior means
        new_means = (prior_vars * np.sum(data, axis=0) + likelihood_var * prior_means * n_samples) / (
            prior_vars + likelihood_var * n_samples
        )

        # Check convergence
        if np.allclose(posterior_means, new_means, atol=tol):
            if verbose:
                print(f"Converged after {i} iterations")
            break

        posterior_means = new_means
        posterior_vars = 1 / (1/prior_vars + n_samples/likelihood_var)

    return {
        'posterior_means': posterior_means,
        'posterior_vars': posterior_vars
    }

def _calculate_metrics(
    data: np.ndarray,
    posterior_means: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Calculate metrics for the model."""
    predictions = np.mean(data, axis=0)  # Simple prediction for example

    if metric == 'mse':
        return {'mse': np.mean((predictions - posterior_means)**2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(predictions - posterior_means))}
    elif callable(metric):
        return {'custom_metric': metric(predictions, posterior_means)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _check_warnings(
    result: Dict[str, Any]
) -> List[str]:
    """Check for potential warnings in the results."""
    warnings = []

    if np.any(np.isnan(result['posterior_means'])):
        warnings.append("NaN values detected in posterior means")

    if np.any(np.isinf(result['posterior_vars'])):
        warnings.append("Infinite values detected in posterior variances")

    return warnings

################################################################################
# bayesian_nonparametrics
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def bayesian_nonparametrics_fit(
    data: np.ndarray,
    prior: Optional[np.ndarray] = None,
    likelihood: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.sum((x - y) ** 2),
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.linalg.norm(x - y),
    solver: str = 'gradient_descent',
    max_iter: int = 1000,
    tol: float = 1e-6,
    normalize: str = 'standard',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Perform Bayesian non-parametric inference.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    prior : Optional[np.ndarray], default=None
        Prior distribution. If None, a uniform prior is assumed.
    likelihood : Callable[[np.ndarray, np.ndarray], float], default=lambda x, y: np.sum((x - y) ** 2)
        Likelihood function.
    distance_metric : Callable[[np.ndarray, np.ndarray], float], default=lambda x, y: np.linalg.norm(x - y)
        Distance metric function.
    solver : str, default='gradient_descent'
        Solver to use. Options: 'gradient_descent', 'newton', 'coordinate_descent'.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-6
        Tolerance for convergence.
    normalize : str, default='standard'
        Normalization method. Options: 'none', 'standard', 'minmax', 'robust'.
    regularization : Optional[str], default=None
        Regularization method. Options: 'none', 'l1', 'l2', 'elasticnet'.
    alpha : float, default=1.0
        Regularization strength for L1 or elasticnet.
    beta : float, default=1.0
        Regularization strength for L2 or elasticnet.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], default=None
        Custom metric function.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.randn(100, 2)
    >>> result = bayesian_nonparametrics_fit(data)
    """
    # Validate inputs
    _validate_inputs(data, prior, normalize)

    # Normalize data if required
    normalized_data = _normalize_data(data, normalize)

    # Initialize parameters
    params = _initialize_parameters(normalized_data, prior)

    # Choose solver and optimize
    if solver == 'gradient_descent':
        optimized_params = _gradient_descent(
            params, normalized_data, likelihood, distance_metric,
            max_iter, tol, regularization, alpha, beta
        )
    elif solver == 'newton':
        optimized_params = _newton_method(
            params, normalized_data, likelihood, distance_metric,
            max_iter, tol, regularization, alpha, beta
        )
    elif solver == 'coordinate_descent':
        optimized_params = _coordinate_descent(
            params, normalized_data, likelihood, distance_metric,
            max_iter, tol, regularization, alpha, beta
        )
    else:
        raise ValueError("Invalid solver specified.")

    # Calculate metrics
    metrics = _calculate_metrics(
        optimized_params, normalized_data,
        likelihood, distance_metric, custom_metric
    )

    # Prepare output
    result = {
        'result': optimized_params,
        'metrics': metrics,
        'params_used': {
            'solver': solver,
            'normalize': normalize,
            'regularization': regularization,
            'alpha': alpha,
            'beta': beta
        },
        'warnings': _check_warnings(optimized_params, metrics)
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    prior: Optional[np.ndarray],
    normalize: str
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if prior is not None and not isinstance(prior, np.ndarray):
        raise TypeError("Prior must be a numpy array or None.")
    if normalize not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("Invalid normalization method specified.")

def _normalize_data(
    data: np.ndarray,
    normalize: str
) -> np.ndarray:
    """Normalize the input data."""
    if normalize == 'none':
        return data
    elif normalize == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif normalize == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif normalize == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError("Invalid normalization method specified.")

def _initialize_parameters(
    data: np.ndarray,
    prior: Optional[np.ndarray]
) -> np.ndarray:
    """Initialize parameters for optimization."""
    if prior is not None:
        return prior.copy()
    else:
        return np.zeros(data.shape[1])

def _gradient_descent(
    params: np.ndarray,
    data: np.ndarray,
    likelihood: Callable[[np.ndarray, np.ndarray], float],
    distance_metric: Callable[[np.ndarray, np.ndarray], float],
    max_iter: int,
    tol: float,
    regularization: Optional[str],
    alpha: float,
    beta: float
) -> np.ndarray:
    """Perform gradient descent optimization."""
    for _ in range(max_iter):
        grad = _compute_gradient(params, data, likelihood, distance_metric)
        if regularization == 'l1':
            grad += alpha * np.sign(params)
        elif regularization == 'l2':
            grad += 2 * beta * params
        elif regularization == 'elasticnet':
            grad += alpha * np.sign(params) + 2 * beta * params
        params -= tol * grad
    return params

def _newton_method(
    params: np.ndarray,
    data: np.ndarray,
    likelihood: Callable[[np.ndarray, np.ndarray], float],
    distance_metric: Callable[[np.ndarray, np.ndarray], float],
    max_iter: int,
    tol: float,
    regularization: Optional[str],
    alpha: float,
    beta: float
) -> np.ndarray:
    """Perform Newton method optimization."""
    for _ in range(max_iter):
        grad = _compute_gradient(params, data, likelihood, distance_metric)
        hessian = _compute_hessian(params, data, likelihood, distance_metric)
        if regularization == 'l1':
            hessian += alpha * np.eye(len(params))
        elif regularization == 'l2':
            hessian += 2 * beta * np.eye(len(params))
        elif regularization == 'elasticnet':
            hessian += alpha * np.eye(len(params)) + 2 * beta * np.eye(len(params))
        params -= np.linalg.solve(hessian, grad) * tol
    return params

def _coordinate_descent(
    params: np.ndarray,
    data: np.ndarray,
    likelihood: Callable[[np.ndarray, np.ndarray], float],
    distance_metric: Callable[[np.ndarray, np.ndarray], float],
    max_iter: int,
    tol: float,
    regularization: Optional[str],
    alpha: float,
    beta: float
) -> np.ndarray:
    """Perform coordinate descent optimization."""
    for _ in range(max_iter):
        for i in range(len(params)):
            params[i] -= tol * _compute_partial_gradient(i, params, data, likelihood, distance_metric)
            if regularization == 'l1':
                params[i] = np.sign(params[i]) * max(0, abs(params[i]) - alpha * tol)
            elif regularization == 'l2':
                params[i] = params[i] / (1 + 2 * beta * tol)
            elif regularization == 'elasticnet':
                params[i] = np.sign(params[i]) * max(0, abs(params[i]) - alpha * tol) / (1 + 2 * beta * tol)
    return params

def _compute_gradient(
    params: np.ndarray,
    data: np.ndarray,
    likelihood: Callable[[np.ndarray, np.ndarray], float],
    distance_metric: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Compute the gradient of the objective function."""
    return -2 * (data @ params - np.mean(data @ params)) / len(data)

def _compute_hessian(
    params: np.ndarray,
    data: np.ndarray,
    likelihood: Callable[[np.ndarray, np.ndarray], float],
    distance_metric: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Compute the Hessian matrix of the objective function."""
    return 2 * (data.T @ data) / len(data)

def _compute_partial_gradient(
    i: int,
    params: np.ndarray,
    data: np.ndarray,
    likelihood: Callable[[np.ndarray, np.ndarray], float],
    distance_metric: Callable[[np.ndarray, np.ndarray], float]
) -> float:
    """Compute the partial gradient for coordinate descent."""
    return -2 * (data[:, i] @ params) / len(data)

def _calculate_metrics(
    params: np.ndarray,
    data: np.ndarray,
    likelihood: Callable[[np.ndarray, np.ndarray], float],
    distance_metric: Callable[[np.ndarray, np.ndarray], float],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate various metrics."""
    mse = np.mean((data @ params - data) ** 2)
    mae = np.mean(np.abs(data @ params - data))
    r2 = 1 - mse / np.var(data)
    metrics = {'mse': mse, 'mae': mae, 'r2': r2}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(data, params)
    return metrics

def _check_warnings(
    params: np.ndarray,
    metrics: Dict[str, float]
) -> str:
    """Check for warnings."""
    if np.any(np.isnan(params)):
        return "Warning: NaN values in parameters."
    if metrics['mse'] > 1e6:
        return "Warning: High MSE value."
    return ""

################################################################################
# bayesian_structural_time_series
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def bayesian_structural_time_series_fit(
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, List, str]]:
    """
    Fit a Bayesian structural time series model.

    Parameters
    ----------
    y : np.ndarray
        Input time series data.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the input data. Default is None.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to evaluate the model. Can be 'mse', 'mae', 'r2', or a custom callable.
    solver : str, optional
        Solver to use. Options are 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : Optional[str], optional
        Regularization method. Options are 'l1', 'l2', or None.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.
    random_state : Optional[int], optional
        Random seed for reproducibility. Default is None.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function. Default is None.

    Returns
    -------
    Dict[str, Union[Dict, List, str]]
        Dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> y = np.array([1.0, 2.0, 3.0, 4.0])
    >>> result = bayesian_structural_time_series_fit(y)
    """
    # Validate inputs
    _validate_inputs(y, normalizer)

    # Normalize data if a normalizer is provided
    y_normalized = _apply_normalization(y, normalizer)

    # Initialize parameters
    params = _initialize_parameters(y_normalized.shape[0], random_state)

    # Choose solver
    if solver == 'gradient_descent':
        params = _gradient_descent(y_normalized, params, max_iter, tol, regularization)
    elif solver == 'newton':
        params = _newton_method(y_normalized, params, max_iter, tol)
    elif solver == 'coordinate_descent':
        params = _coordinate_descent(y_normalized, params, max_iter, tol)
    else:
        raise ValueError("Invalid solver specified.")

    # Calculate metrics
    metrics = _calculate_metrics(y_normalized, params, metric, custom_metric)

    # Prepare the result dictionary
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

    return result

def _validate_inputs(y: np.ndarray, normalizer: Optional[Callable[[np.ndarray], np.ndarray]]) -> None:
    """Validate the input data and normalizer."""
    if not isinstance(y, np.ndarray):
        raise TypeError("Input y must be a numpy array.")
    if normalizer is not None and not callable(normalizer):
        raise TypeError("Normalizer must be a callable function.")

def _apply_normalization(y: np.ndarray, normalizer: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Apply normalization to the input data."""
    if normalizer is None:
        return y
    return normalizer(y)

def _initialize_parameters(n: int, random_state: Optional[int]) -> np.ndarray:
    """Initialize parameters for the model."""
    if random_state is not None:
        np.random.seed(random_state)
    return np.random.randn(n)

def _gradient_descent(
    y: np.ndarray,
    params: np.ndarray,
    max_iter: int,
    tol: float,
    regularization: Optional[str]
) -> np.ndarray:
    """Perform gradient descent optimization."""
    for _ in range(max_iter):
        # Update parameters
        params = _update_parameters_gradient_descent(y, params, regularization)
    return params

def _newton_method(
    y: np.ndarray,
    params: np.ndarray,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Perform Newton's method optimization."""
    for _ in range(max_iter):
        # Update parameters
        params = _update_parameters_newton(y, params)
    return params

def _coordinate_descent(
    y: np.ndarray,
    params: np.ndarray,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Perform coordinate descent optimization."""
    for _ in range(max_iter):
        # Update parameters
        params = _update_parameters_coordinate_descent(y, params)
    return params

def _calculate_metrics(
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate the metrics for the model."""
    if custom_metric is not None:
        return {'custom': custom_metric(y, params)}

    metrics_dict = {}
    if metric == 'mse':
        metrics_dict['mse'] = np.mean((y - params) ** 2)
    elif metric == 'mae':
        metrics_dict['mae'] = np.mean(np.abs(y - params))
    elif metric == 'r2':
        ss_res = np.sum((y - params) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics_dict['r2'] = 1 - (ss_res / ss_tot)
    else:
        raise ValueError("Invalid metric specified.")

    return metrics_dict

def _update_parameters_gradient_descent(
    y: np.ndarray,
    params: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Update parameters using gradient descent."""
    gradient = 2 * (params - y)
    if regularization == 'l1':
        gradient += np.sign(params)
    elif regularization == 'l2':
        gradient += 2 * params
    return params - 0.01 * gradient

def _update_parameters_newton(
    y: np.ndarray,
    params: np.ndarray
) -> np.ndarray:
    """Update parameters using Newton's method."""
    hessian = 2 * np.eye(len(params))
    gradient = 2 * (params - y)
    return params - np.linalg.inv(hessian) @ gradient

def _update_parameters_coordinate_descent(
    y: np.ndarray,
    params: np.ndarray
) -> np.ndarray:
    """Update parameters using coordinate descent."""
    for i in range(len(params)):
        params[i] = y[i]
    return params

################################################################################
# approximate_bayesian_computation
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def approximate_bayesian_computation_fit(
    simulator: Callable,
    distance_metric: Union[str, Callable],
    observed_data: np.ndarray,
    prior_distributions: Dict[str, Any],
    n_simulations: int = 1000,
    epsilon: float = 0.1,
    normalization: str = 'none',
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Approximate Bayesian Computation (ABC) fit function.

    Parameters:
    -----------
    simulator : callable
        Function that simulates data given parameters.
    distance_metric : str or callable
        Distance metric to compare simulated and observed data. Can be 'euclidean', 'manhattan',
        'cosine', or a custom callable.
    observed_data : np.ndarray
        Observed data to compare against simulated data.
    prior_distributions : dict
        Dictionary of prior distributions for each parameter.
    n_simulations : int, optional
        Number of simulations to run (default: 1000).
    epsilon : float, optional
        Threshold for accepting simulations (default: 0.1).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    custom_metric : callable, optional
        Custom metric function to evaluate simulations (default: None).
    **kwargs : dict
        Additional keyword arguments for the simulator.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(simulator, distance_metric, observed_data, prior_distributions)

    # Normalize data if required
    normalized_observed = _normalize_data(observed_data, normalization)

    # Initialize results
    accepted_params = []
    distances = []

    for _ in range(n_simulations):
        # Sample parameters from prior distributions
        params = _sample_prior(prior_distributions)

        # Simulate data
        simulated_data = simulator(params, **kwargs)

        # Normalize simulated data if required
        normalized_simulated = _normalize_data(simulated_data, normalization)

        # Calculate distance
        if isinstance(distance_metric, str):
            distance = _calculate_distance(normalized_simulated, normalized_observed, distance_metric)
        else:
            distance = distance_metric(normalized_simulated, normalized_observed)

        # Check if simulation is accepted
        if distance <= epsilon:
            accepted_params.append(params)
            distances.append(distance)

    # Calculate metrics
    metrics = _calculate_metrics(accepted_params, distances, custom_metric)

    # Prepare results
    result = {
        'result': accepted_params,
        'metrics': metrics,
        'params_used': prior_distributions,
        'warnings': []
    }

    return result

def _validate_inputs(
    simulator: Callable,
    distance_metric: Union[str, Callable],
    observed_data: np.ndarray,
    prior_distributions: Dict[str, Any]
) -> None:
    """
    Validate input parameters.

    Parameters:
    -----------
    simulator : callable
        Function that simulates data given parameters.
    distance_metric : str or callable
        Distance metric to compare simulated and observed data.
    observed_data : np.ndarray
        Observed data to compare against simulated data.
    prior_distributions : dict
        Dictionary of prior distributions for each parameter.

    Raises:
    -------
    ValueError
        If any input is invalid.
    """
    if not callable(simulator):
        raise ValueError("simulator must be a callable function.")
    if isinstance(distance_metric, str) and distance_metric not in ['euclidean', 'manhattan', 'cosine']:
        raise ValueError("distance_metric must be 'euclidean', 'manhattan', 'cosine', or a custom callable.")
    if not isinstance(observed_data, np.ndarray):
        raise ValueError("observed_data must be a numpy array.")
    if not isinstance(prior_distributions, dict):
        raise ValueError("prior_distributions must be a dictionary.")
    if np.any(np.isnan(observed_data)) or np.any(np.isinf(observed_data)):
        raise ValueError("observed_data contains NaN or infinite values.")

def _normalize_data(
    data: np.ndarray,
    method: str = 'none'
) -> np.ndarray:
    """
    Normalize data using the specified method.

    Parameters:
    -----------
    data : np.ndarray
        Data to normalize.
    method : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').

    Returns:
    --------
    np.ndarray
        Normalized data.
    """
    if method == 'none':
        return data
    elif method == 'standard':
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

def _sample_prior(
    prior_distributions: Dict[str, Any]
) -> Dict[str, float]:
    """
    Sample parameters from prior distributions.

    Parameters:
    -----------
    prior_distributions : dict
        Dictionary of prior distributions for each parameter.

    Returns:
    --------
    dict
        Dictionary of sampled parameters.
    """
    params = {}
    for param, dist in prior_distributions.items():
        if isinstance(dist, tuple) and len(dist) == 2:
            # Assume uniform distribution
            low, high = dist
            params[param] = np.random.uniform(low, high)
        else:
            raise ValueError(f"Unsupported prior distribution for parameter {param}: {dist}")
    return params

def _calculate_distance(
    simulated_data: np.ndarray,
    observed_data: np.ndarray,
    metric: str
) -> float:
    """
    Calculate distance between simulated and observed data.

    Parameters:
    -----------
    simulated_data : np.ndarray
        Simulated data.
    observed_data : np.ndarray
        Observed data.
    metric : str
        Distance metric ('euclidean', 'manhattan', 'cosine').

    Returns:
    --------
    float
        Calculated distance.
    """
    if metric == 'euclidean':
        return np.linalg.norm(simulated_data - observed_data)
    elif metric == 'manhattan':
        return np.sum(np.abs(simulated_data - observed_data))
    elif metric == 'cosine':
        return 1 - np.dot(simulated_data, observed_data) / (np.linalg.norm(simulated_data) * np.linalg.norm(observed_data))
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _calculate_metrics(
    accepted_params: list,
    distances: list,
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """
    Calculate metrics for the accepted parameters.

    Parameters:
    -----------
    accepted_params : list
        List of accepted parameter sets.
    distances : list
        List of distances for accepted simulations.
    custom_metric : callable, optional
        Custom metric function to evaluate simulations (default: None).

    Returns:
    --------
    dict
        Dictionary of calculated metrics.
    """
    metrics = {
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'n_accepted': len(accepted_params)
    }

    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(accepted_params)

    return metrics
