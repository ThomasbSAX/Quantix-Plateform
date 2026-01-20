"""
Quantix – Module tests_bayesiens
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# test_rapport_vraisemblance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List, Tuple

def test_rapport_vraisemblance_fit(
    data: np.ndarray,
    model: Callable,
    prior: Optional[Callable] = None,
    likelihood: Optional[Callable] = None,
    normalizer: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'gradient_descent',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_params: Optional[Dict] = None
) -> Dict:
    """
    Perform a Bayesian likelihood ratio test with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    model : Callable
        The statistical model to fit (e.g., logistic regression).
    prior : Optional[Callable], default=None
        Prior distribution function.
    likelihood : Optional[Callable], default=None
        Likelihood function. If None, uses a default Gaussian likelihood.
    normalizer : str, default='standard'
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : Union[str, Callable], default='mse'
        Metric for evaluation: 'mse', 'mae', 'r2', or custom callable.
    solver : str, default='gradient_descent'
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    tol : float, default=1e-6
        Tolerance for convergence.
    max_iter : int, default=1000
        Maximum number of iterations.
    custom_params : Optional[Dict], default=None
        Additional parameters for the model or solver.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, model)

    # Normalize data
    normalized_data = _normalize_data(data, normalizer)

    # Initialize parameters
    params = _initialize_parameters(model, custom_params)

    # Fit model using specified solver
    fitted_model = _fit_model(
        normalized_data,
        model,
        prior,
        likelihood,
        solver,
        tol,
        max_iter,
        params
    )

    # Compute metrics
    metrics = _compute_metrics(normalized_data, fitted_model, metric)

    # Prepare output
    result = {
        'result': fitted_model,
        'metrics': metrics,
        'params_used': params,
        'warnings': _check_warnings(normalized_data, fitted_model)
    }

    return result

def _validate_inputs(data: np.ndarray, model: Callable) -> None:
    """Validate input data and model."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if not callable(model):
        raise TypeError("Model must be a callable function.")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data based on specified method."""
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

def _initialize_parameters(model: Callable, custom_params: Optional[Dict]) -> Dict:
    """Initialize model parameters."""
    params = {}
    if custom_params is not None:
        params.update(custom_params)
    return params

def _fit_model(
    data: np.ndarray,
    model: Callable,
    prior: Optional[Callable],
    likelihood: Optional[Callable],
    solver: str,
    tol: float,
    max_iter: int,
    params: Dict
) -> Dict:
    """Fit the model using specified solver."""
    if solver == 'closed_form':
        return _fit_closed_form(data, model, params)
    elif solver == 'gradient_descent':
        return _fit_gradient_descent(data, model, prior, likelihood, tol, max_iter, params)
    elif solver == 'newton':
        return _fit_newton(data, model, prior, likelihood, tol, max_iter, params)
    elif solver == 'coordinate_descent':
        return _fit_coordinate_descent(data, model, prior, likelihood, tol, max_iter, params)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _fit_closed_form(data: np.ndarray, model: Callable, params: Dict) -> Dict:
    """Fit model using closed-form solution."""
    # Placeholder for actual implementation
    return {'params': params, 'converged': True}

def _fit_gradient_descent(
    data: np.ndarray,
    model: Callable,
    prior: Optional[Callable],
    likelihood: Optional[Callable],
    tol: float,
    max_iter: int,
    params: Dict
) -> Dict:
    """Fit model using gradient descent."""
    # Placeholder for actual implementation
    return {'params': params, 'converged': True}

def _fit_newton(
    data: np.ndarray,
    model: Callable,
    prior: Optional[Callable],
    likelihood: Optional[Callable],
    tol: float,
    max_iter: int,
    params: Dict
) -> Dict:
    """Fit model using Newton's method."""
    # Placeholder for actual implementation
    return {'params': params, 'converged': True}

def _fit_coordinate_descent(
    data: np.ndarray,
    model: Callable,
    prior: Optional[Callable],
    likelihood: Optional[Callable],
    tol: float,
    max_iter: int,
    params: Dict
) -> Dict:
    """Fit model using coordinate descent."""
    # Placeholder for actual implementation
    return {'params': params, 'converged': True}

def _compute_metrics(data: np.ndarray, model_result: Dict, metric: Union[str, Callable]) -> Dict:
    """Compute metrics based on specified method."""
    if callable(metric):
        return {'custom_metric': metric(data, model_result)}
    elif metric == 'mse':
        return {'mse': np.mean((data - model_result['prediction']) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(data - model_result['prediction']))}
    elif metric == 'r2':
        ss_res = np.sum((data - model_result['prediction']) ** 2)
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        return {'r2': 1 - (ss_res / ss_tot)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _check_warnings(data: np.ndarray, model_result: Dict) -> List[str]:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isnan(data)):
        warnings.append("Input data contains NaN values.")
    if not model_result.get('converged', False):
        warnings.append("Solver did not converge.")
    return warnings

################################################################################
# test_posteriori
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(
    data: np.ndarray,
    prior: np.ndarray,
    likelihood_func: Callable,
    normalizer: Optional[Callable] = None
) -> None:
    """
    Validate inputs for posterior test.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    prior : np.ndarray
        Prior distribution.
    likelihood_func : Callable
        Likelihood function.
    normalizer : Optional[Callable], default=None
        Normalization function.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if prior.ndim != 1:
        raise ValueError("Prior must be a 1D array.")
    if len(prior) != data.shape[1]:
        raise ValueError("Prior length must match number of features.")
    if normalizer is not None and not callable(normalizer):
        raise ValueError("Normalizer must be a callable function.")

def compute_posterior(
    data: np.ndarray,
    prior: np.ndarray,
    likelihood_func: Callable,
    normalizer: Optional[Callable] = None
) -> np.ndarray:
    """
    Compute posterior distribution.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    prior : np.ndarray
        Prior distribution.
    likelihood_func : Callable
        Likelihood function.
    normalizer : Optional[Callable], default=None
        Normalization function.

    Returns
    ------
    np.ndarray
        Posterior distribution.
    """
    if normalizer is not None:
        data = normalizer(data)

    likelihoods = np.array([likelihood_func(data, param) for param in prior])
    posterior = prior * likelihoods
    return posterior / np.sum(posterior)

def compute_metrics(
    data: np.ndarray,
    prior: np.ndarray,
    posterior: np.ndarray,
    metric_funcs: Dict[str, Callable]
) -> Dict[str, float]:
    """
    Compute metrics for posterior test.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    prior : np.ndarray
        Prior distribution.
    posterior : np.ndarray
        Posterior distribution.
    metric_funcs : Dict[str, Callable]
        Dictionary of metric functions.

    Returns
    ------
    Dict[str, float]
        Computed metrics.
    """
    metrics = {}
    for name, func in metric_funcs.items():
        metrics[name] = func(data, prior, posterior)
    return metrics

def test_posteriori_fit(
    data: np.ndarray,
    prior: np.ndarray,
    likelihood_func: Callable,
    normalizer: Optional[Callable] = None,
    metric_funcs: Dict[str, Callable] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Fit posterior test.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    prior : np.ndarray
        Prior distribution.
    likelihood_func : Callable
        Likelihood function.
    normalizer : Optional[Callable], default=None
        Normalization function.
    metric_funcs : Dict[str, Callable], default=None
        Dictionary of metric functions.

    Returns
    ------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]
        Dictionary containing results, metrics, and parameters used.
    """
    validate_inputs(data, prior, likelihood_func, normalizer)

    posterior = compute_posterior(data, prior, likelihood_func, normalizer)
    metrics = {} if metric_funcs is None else compute_metrics(data, prior, posterior, metric_funcs)

    return {
        "result": posterior,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric_funcs": list(metric_funcs.keys()) if metric_funcs else []
        },
        "warnings": []
    }

# Example usage:
"""
data = np.random.rand(100, 5)
prior = np.ones(5) / 5
likelihood_func = lambda data, param: np.exp(-np.sum((data - param) ** 2, axis=1))
normalizer = lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0)
metric_funcs = {
    "mse": lambda data, prior, posterior: np.mean((data - np.dot(data, posterior)) ** 2),
    "mae": lambda data, prior, posterior: np.mean(np.abs(data - np.dot(data, posterior)))
}

result = test_posteriori_fit(data, prior, likelihood_func, normalizer, metric_funcs)
"""

################################################################################
# test_prior
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def test_prior_fit(
    data: np.ndarray,
    prior: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Perform a Bayesian prior test with configurable parameters.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    prior : np.ndarray
        Prior distribution vector of shape (n_features,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the data. Default is identity function.
    metric : str
        Metric to evaluate the test ('mse', 'mae', 'r2', 'logloss').
    distance : str
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski').
    solver : str
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : Optional[str]
        Regularization type (None, 'l1', 'l2', 'elasticnet').
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function if needed.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom distance function if needed.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, prior)

    # Normalize data
    normalized_data = normalizer(data)

    # Choose metric and distance functions
    metric_func, distance_func = _get_metric_distance_functions(metric, distance,
                                                               custom_metric, custom_distance)

    # Solve the problem
    result = _solve_prior_problem(normalized_data, prior, solver, regularization,
                                 tol, max_iter, distance_func)

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, result['params'], metric_func)

    # Prepare output
    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

def _validate_inputs(data: np.ndarray, prior: np.ndarray) -> None:
    """Validate input data and prior."""
    if not isinstance(data, np.ndarray) or not isinstance(prior, np.ndarray):
        raise TypeError("Data and prior must be numpy arrays.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if prior.ndim != 1:
        raise ValueError("Prior must be a 1D array.")
    if data.shape[1] != prior.shape[0]:
        raise ValueError("Data features and prior dimensions must match.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")
    if np.any(np.isnan(prior)) or np.any(np.isinf(prior)):
        raise ValueError("Prior contains NaN or infinite values.")

def _get_metric_distance_functions(
    metric: str,
    distance: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> tuple:
    """Get metric and distance functions based on user choices."""
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric_function(metric)

    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_function(distance)

    return metric_func, distance_func

def _get_metric_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
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

def _get_distance_function(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the appropriate distance function."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': lambda x, y: np.sum(np.abs(x - y)**3)**(1/3)
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _solve_prior_problem(
    data: np.ndarray,
    prior: np.ndarray,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, Any]:
    """Solve the prior problem using the specified solver."""
    solvers = {
        'closed_form': _solve_closed_form,
        'gradient_descent': lambda d, p, r, t, m: _solve_gradient_descent(d, p, r, t, m),
        'newton': lambda d, p, r, t, m: _solve_newton(d, p, r, t, m),
        'coordinate_descent': lambda d, p, r, t, m: _solve_coordinate_descent(d, p, r, t, m)
    }
    if solver not in solvers:
        raise ValueError(f"Unknown solver: {solver}")
    return solvers[solver](data, prior, regularization, tol, max_iter)

def _solve_closed_form(
    data: np.ndarray,
    prior: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve the problem in closed form."""
    # Placeholder for actual implementation
    params = np.linalg.pinv(data.T @ data) @ (data.T @ prior)
    return {
        'params': params,
        'converged': True,
        'iterations': 0
    }

def _solve_gradient_descent(
    data: np.ndarray,
    prior: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve the problem using gradient descent."""
    # Placeholder for actual implementation
    params = np.zeros(data.shape[1])
    for i in range(max_iter):
        gradient = -2 * data.T @ (prior - data @ params)
        if regularization == 'l2':
            gradient += 2 * np.linalg.norm(params)
        params -= 0.01 * gradient
    return {
        'params': params,
        'converged': np.linalg.norm(gradient) < tol,
        'iterations': max_iter
    }

def _solve_newton(
    data: np.ndarray,
    prior: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve the problem using Newton's method."""
    # Placeholder for actual implementation
    params = np.zeros(data.shape[1])
    for i in range(max_iter):
        gradient = -2 * data.T @ (prior - data @ params)
        hessian = 2 * data.T @ data
        if regularization == 'l2':
            hessian += 2 * np.eye(data.shape[1])
        params -= np.linalg.pinv(hessian) @ gradient
    return {
        'params': params,
        'converged': np.linalg.norm(gradient) < tol,
        'iterations': max_iter
    }

def _solve_coordinate_descent(
    data: np.ndarray,
    prior: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve the problem using coordinate descent."""
    # Placeholder for actual implementation
    params = np.zeros(data.shape[1])
    for i in range(max_iter):
        for j in range(data.shape[1]):
            residual = prior - data @ params + data[:, j] * params[j]
            params[j] = np.sum(data[:, j] * residual) / (data[:, j].T @ data[:, j])
    return {
        'params': params,
        'converged': True,
        'iterations': max_iter
    }

def _calculate_metrics(
    data: np.ndarray,
    params: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Calculate metrics for the solution."""
    predictions = data @ params
    return {
        'metric_value': metric_func(data, predictions)
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

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate log loss."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.linalg.norm(x - y)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(x - y))

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate cosine distance."""
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

################################################################################
# test_likelihood
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    data: np.ndarray,
    prior: np.ndarray,
    likelihood_func: Callable[[np.ndarray, np.ndarray], float],
    normalize: str = "none",
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray) or not isinstance(prior, np.ndarray):
        raise TypeError("Data and prior must be numpy arrays")
    if data.ndim != 2 or prior.ndim != 1:
        raise ValueError("Data must be 2D and prior must be 1D")
    if data.shape[1] != prior.size:
        raise ValueError("Data columns must match prior size")
    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("Invalid normalization option")
    if any(np.isnan(data).any()) or any(np.isinf(data).any()):
        raise ValueError("Data contains NaN or infinite values")
    if any(np.isnan(prior).any()) or any(np.isinf(prior).any()):
        raise ValueError("Prior contains NaN or infinite values")

def _normalize_data(
    data: np.ndarray,
    normalize: str = "none",
) -> np.ndarray:
    """Normalize data according to specified method."""
    if normalize == "none":
        return data
    elif normalize == "standard":
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif normalize == "minmax":
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif normalize == "robust":
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / iqr
    else:
        raise ValueError("Invalid normalization option")

def _compute_likelihood(
    data: np.ndarray,
    prior: np.ndarray,
    likelihood_func: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Compute the likelihood of data given prior using provided function."""
    return likelihood_func(data, prior)

def _compute_metrics(
    data: np.ndarray,
    prior: np.ndarray,
    likelihood: float,
    metric_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
) -> Dict[str, float]:
    """Compute various metrics based on data and prior."""
    return {name: func(data, prior) for name, func in metric_funcs.items()}

def test_likelihood_fit(
    data: np.ndarray,
    prior: np.ndarray,
    likelihood_func: Callable[[np.ndarray, np.ndarray], float],
    normalize: str = "none",
    metric_funcs: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
    **kwargs
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Perform a likelihood test with configurable options.

    Parameters:
    - data: Input data as 2D numpy array
    - prior: Prior distribution as 1D numpy array
    - likelihood_func: Callable that computes likelihood
    - normalize: Normalization method (none, standard, minmax, robust)
    - metric_funcs: Dictionary of metric functions to compute
    - **kwargs: Additional keyword arguments for future extensions

    Returns:
    Dictionary containing results, metrics, parameters used, and warnings
    """
    _validate_inputs(data, prior, likelihood_func, normalize)
    normalized_data = _normalize_data(data, normalize)

    result = {
        "result": _compute_likelihood(normalized_data, prior, likelihood_func),
        "metrics": _compute_metrics(
            normalized_data,
            prior,
            result["result"],
            metric_funcs or {}
        ),
        "params_used": {
            "normalize": normalize,
            "likelihood_func": likelihood_func.__name__ if hasattr(likelihood_func, "__name__") else "custom",
            **kwargs
        },
        "warnings": []
    }

    return result

def _default_likelihood_func(data: np.ndarray, prior: np.ndarray) -> float:
    """Default likelihood function (product of probabilities)."""
    return np.prod(np.sum(data * prior, axis=1))

# Example usage:
if __name__ == "__main__":
    data = np.random.rand(10, 3)
    prior = np.array([0.2, 0.3, 0.5])
    metrics = {
        "mse": lambda x, y: np.mean((x - y) ** 2),
        "mae": lambda x, y: np.mean(np.abs(x - y))
    }
    result = test_likelihood_fit(
        data,
        prior,
        _default_likelihood_func,
        normalize="standard",
        metric_funcs=metrics
    )

################################################################################
# test_bayes_factor
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(data: np.ndarray, prior_odds: float = 1.0) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")
    if prior_odds <= 0:
        raise ValueError("Prior odds must be positive")

def _compute_likelihood(data: np.ndarray, model: Callable) -> float:
    """Compute the likelihood of the data under a given model."""
    return np.prod(model(data))

def _compute_bayes_factor(likelihood_alt: float, likelihood_null: float,
                         prior_odds: float = 1.0) -> Dict[str, Union[float, str]]:
    """Compute the Bayes factor."""
    bayes_factor = (likelihood_alt / likelihood_null) * prior_odds
    return {
        "bayes_factor": bayes_factor,
        "interpretation": _interpret_bayes_factor(bayes_factor)
    }

def _interpret_bayes_factor(bayes_factor: float) -> str:
    """Interpret the Bayes factor."""
    if bayes_factor > 100:
        return "Very strong evidence for alternative hypothesis"
    elif bayes_factor > 10:
        return "Strong evidence for alternative hypothesis"
    elif bayes_factor > 3:
        return "Moderate evidence for alternative hypothesis"
    elif bayes_factor > 1:
        return "Anecdotal evidence for alternative hypothesis"
    elif bayes_factor == 1:
        return "No evidence"
    else:
        return "Evidence for null hypothesis"

def test_bayes_factor_compute(
    data: np.ndarray,
    model_alt: Callable[[np.ndarray], float],
    model_null: Callable[[np.ndarray], float] = lambda x: np.ones(x.shape[0]),
    prior_odds: float = 1.0,
    normalize_data: bool = False
) -> Dict[str, Union[float, str]]:
    """
    Compute the Bayes factor between two models.

    Parameters
    ----------
    data : np.ndarray
        Input data (2D array)
    model_alt : callable
        Alternative model likelihood function
    model_null : callable, optional
        Null model likelihood function (default: uniform)
    prior_odds : float, optional
        Prior odds ratio (default: 1.0)
    normalize_data : bool, optional
        Whether to normalize data (default: False)

    Returns
    -------
    dict
        Dictionary containing Bayes factor and interpretation

    Example
    -------
    >>> data = np.random.randn(100, 2)
    >>> model_alt = lambda x: np.exp(-np.sum(x**2, axis=1))
    >>> test_bayes_factor_compute(data, model_alt)
    """
    _validate_inputs(data, prior_odds)

    if normalize_data:
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    likelihood_alt = _compute_likelihood(data, model_alt)
    likelihood_null = _compute_likelihood(data, model_null)

    result = _compute_bayes_factor(likelihood_alt, likelihood_null, prior_odds)

    return {
        "result": result,
        "metrics": {},
        "params_used": {
            "prior_odds": prior_odds,
            "normalize_data": normalize_data
        },
        "warnings": []
    }

################################################################################
# test_odds_ratio
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(a: np.ndarray, b: np.ndarray) -> None:
    """Validate input arrays for odds ratio test."""
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional.")
    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        raise ValueError("Input arrays must contain only finite values.")

def _compute_odds_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the odds ratio between two binary arrays."""
    a_pos = np.sum(a == 1)
    a_neg = np.sum(a == 0)
    b_pos = np.sum(b == 1)
    b_neg = np.sum(b == 0)

    odds_ratio = (a_pos / a_neg) / (b_pos / b_neg)
    return odds_ratio

def _compute_log_odds_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the log odds ratio between two binary arrays."""
    return np.log(_compute_odds_ratio(a, b))

def _compute_p_value(odds_ratio: float) -> float:
    """Compute the p-value for the odds ratio test."""
    # Placeholder implementation - replace with actual statistical computation
    return 2 * (1 - np.sqrt(odds_ratio))

def _compute_confidence_interval(odds_ratio: float, alpha: float = 0.05) -> tuple:
    """Compute the confidence interval for the odds ratio."""
    # Placeholder implementation - replace with actual statistical computation
    lower = odds_ratio * np.exp(-1.96 * np.sqrt(1 / (odds_ratio - 1)))
    upper = odds_ratio * np.exp(1.96 * np.sqrt(1 / (odds_ratio - 1)))
    return lower, upper

def test_odds_ratio_fit(
    a: np.ndarray,
    b: np.ndarray,
    *,
    normalization: str = "none",
    metric: Union[str, Callable] = "logloss",
    solver: str = "closed_form",
    alpha: float = 0.05,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Union[float, Dict, str]]:
    """
    Perform a Bayesian odds ratio test between two binary arrays.

    Parameters:
    -----------
    a : np.ndarray
        First binary array.
    b : np.ndarray
        Second binary array.
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    metric : str or callable, optional
        Metric to use ("mse", "mae", "r2", "logloss") or custom callable.
    solver : str, optional
        Solver method ("closed_form", "gradient_descent", "newton").
    alpha : float, optional
        Significance level for confidence intervals.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_inputs(a, b)

    # Normalization (placeholder - implement based on normalization choice)
    a_norm = a.copy()
    b_norm = b.copy()

    # Compute odds ratio and related statistics
    odds_ratio = _compute_odds_ratio(a_norm, b_norm)
    log_odds_ratio = _compute_log_odds_ratio(a_norm, b_norm)
    p_value = _compute_p_value(odds_ratio)
    conf_interval = _compute_confidence_interval(odds_ratio, alpha)

    # Compute metrics (placeholder - implement based on metric choice)
    if isinstance(metric, str):
        if metric == "logloss":
            log_loss = - (a_norm * np.log(odds_ratio) + (1 - a_norm) * np.log(1 / odds_ratio)).mean()
            metrics = {"log_loss": log_loss}
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        metrics = {"custom_metric": metric(a_norm, b_norm)}
    else:
        raise ValueError("Metric must be a string or callable.")

    # Prepare output
    result = {
        "odds_ratio": odds_ratio,
        "log_odds_ratio": log_odds_ratio,
        "p_value": p_value,
        "confidence_interval": conf_interval
    }

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric if not custom_metric else "custom",
            "solver": solver,
            "alpha": alpha
        },
        "warnings": []
    }

# Example usage:
# a = np.array([1, 0, 1, 1, 0])
# b = np.array([0, 1, 0, 1, 1])
# result = test_odds_ratio_fit(a, b)

################################################################################
# test_marginal_likelihood
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def test_marginal_likelihood_fit(
    data: np.ndarray,
    prior: Callable[[np.ndarray], float],
    likelihood: Callable[[np.ndarray, np.ndarray], float],
    normalizer: str = 'standard',
    solver: str = 'gradient_descent',
    max_iter: int = 1000,
    tol: float = 1e-6,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[float, Dict, str]]:
    """
    Compute the marginal likelihood using Bayesian inference.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    prior : Callable[[np.ndarray], float]
        Prior probability function.
    likelihood : Callable[[np.ndarray, np.ndarray], float]
        Likelihood function.
    normalizer : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), by default 'standard'.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent'), by default 'gradient_descent'.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    tol : float, optional
        Tolerance for convergence, by default 1e-6.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function, by default None.
    **kwargs : dict
        Additional solver-specific parameters.

    Returns
    -------
    Dict[str, Union[float, Dict, str]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.randn(100, 2)
    >>> prior_func = lambda params: np.prod(np.exp(-params**2 / 2))
    >>> likelihood_func = lambda params, data: np.prod(np.exp(-np.sum((data - params)**2, axis=1) / 2))
    >>> result = test_marginal_likelihood_fit(data, prior_func, likelihood_func)
    """
    # Validate inputs
    _validate_inputs(data, prior, likelihood)

    # Normalize data if required
    normalized_data = _apply_normalization(data, normalizer)

    # Choose solver and compute marginal likelihood
    if solver == 'closed_form':
        result = _solve_closed_form(normalized_data, prior, likelihood)
    elif solver == 'gradient_descent':
        result = _solve_gradient_descent(normalized_data, prior, likelihood, max_iter, tol, **kwargs)
    elif solver == 'newton':
        result = _solve_newton(normalized_data, prior, likelihood, max_iter, tol, **kwargs)
    elif solver == 'coordinate_descent':
        result = _solve_coordinate_descent(normalized_data, prior, likelihood, max_iter, tol, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(normalized_data, result['params'], custom_metric)

    # Prepare output
    output = {
        'result': result['marginal_likelihood'],
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': result.get('warnings', [])
    }

    return output

def _validate_inputs(data: np.ndarray, prior: Callable[[np.ndarray], float], likelihood: Callable[[np.ndarray, np.ndarray], float]) -> None:
    """Validate input data and functions."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")
    if not callable(prior):
        raise TypeError("Prior must be a callable function.")
    if not callable(likelihood):
        raise TypeError("Likelihood must be a callable function.")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to the data."""
    if method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _solve_closed_form(data: np.ndarray, prior: Callable[[np.ndarray], float], likelihood: Callable[[np.ndarray, np.ndarray], float]) -> Dict[str, Union[float, list]]:
    """Solve using closed-form solution."""
    # Placeholder for closed-form solution
    params = np.mean(data, axis=0)
    marginal_likelihood = prior(params) * likelihood(params, data)
    return {'marginal_likelihood': float(marginal_likelihood), 'params': params, 'warnings': []}

def _solve_gradient_descent(
    data: np.ndarray,
    prior: Callable[[np.ndarray], float],
    likelihood: Callable[[np.ndarray, np.ndarray], float],
    max_iter: int,
    tol: float,
    **kwargs
) -> Dict[str, Union[float, list]]:
    """Solve using gradient descent."""
    # Placeholder for gradient descent solution
    params = np.zeros(data.shape[1])
    for _ in range(max_iter):
        # Update parameters using gradient descent
        pass  # Actual implementation would go here
    marginal_likelihood = prior(params) * likelihood(params, data)
    return {'marginal_likelihood': float(marginal_likelihood), 'params': params, 'warnings': []}

def _solve_newton(
    data: np.ndarray,
    prior: Callable[[np.ndarray], float],
    likelihood: Callable[[np.ndarray, np.ndarray], float],
    max_iter: int,
    tol: float,
    **kwargs
) -> Dict[str, Union[float, list]]:
    """Solve using Newton's method."""
    # Placeholder for Newton's method solution
    params = np.zeros(data.shape[1])
    for _ in range(max_iter):
        # Update parameters using Newton's method
        pass  # Actual implementation would go here
    marginal_likelihood = prior(params) * likelihood(params, data)
    return {'marginal_likelihood': float(marginal_likelihood), 'params': params, 'warnings': []}

def _solve_coordinate_descent(
    data: np.ndarray,
    prior: Callable[[np.ndarray], float],
    likelihood: Callable[[np.ndarray, np.ndarray], float],
    max_iter: int,
    tol: float,
    **kwargs
) -> Dict[str, Union[float, list]]:
    """Solve using coordinate descent."""
    # Placeholder for coordinate descent solution
    params = np.zeros(data.shape[1])
    for _ in range(max_iter):
        # Update parameters using coordinate descent
        pass  # Actual implementation would go here
    marginal_likelihood = prior(params) * likelihood(params, data)
    return {'marginal_likelihood': float(marginal_likelihood), 'params': params, 'warnings': []}

def _compute_metrics(data: np.ndarray, params: np.ndarray, custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None) -> Dict[str, float]:
    """Compute metrics for the results."""
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(data, params)
    # Add other default metrics as needed
    return metrics

################################################################################
# test_bayesian_p_value
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def test_bayesian_p_value_fit(
    data: np.ndarray,
    prior: np.ndarray,
    likelihood_func: Callable[[np.ndarray, np.ndarray], float],
    posterior_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    normalization: str = 'standard',
    distance_metric: str = 'euclidean',
    solver: str = 'gradient_descent',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Perform a Bayesian p-value test with configurable parameters.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    prior : np.ndarray
        Prior distribution parameters.
    likelihood_func : callable
        Function to compute the likelihood given data and parameters.
    posterior_func : callable, optional
        Function to compute the posterior distribution. If None, will be inferred.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    distance_metric : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, prior)

    # Normalize data if required
    normalized_data = _normalize_data(data, normalization)

    # Compute posterior if not provided
    if posterior_func is None:
        posterior = _infer_posterior(normalized_data, prior, likelihood_func)
    else:
        posterior = posterior_func(normalized_data, prior)

    # Solve for parameters
    params = _solve_parameters(
        normalized_data,
        prior,
        posterior,
        likelihood_func,
        solver=solver,
        tol=tol,
        max_iter=max_iter
    )

    # Compute metrics
    metrics = _compute_metrics(normalized_data, params, custom_metric)

    # Compute Bayesian p-value
    result = _compute_bayesian_p_value(normalized_data, params, posterior)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

def _validate_inputs(data: np.ndarray, prior: np.ndarray) -> None:
    """Validate input data and prior."""
    if not isinstance(data, np.ndarray) or not isinstance(prior, np.ndarray):
        raise TypeError("Data and prior must be numpy arrays.")
    if data.ndim != 2 or prior.ndim != 1:
        raise ValueError("Data must be 2D and prior must be 1D.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")
    if np.any(np.isnan(prior)) or np.any(np.isinf(prior)):
        raise ValueError("Prior contains NaN or infinite values.")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data based on the specified method."""
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

def _infer_posterior(data: np.ndarray, prior: np.ndarray, likelihood_func: Callable) -> np.ndarray:
    """Infer the posterior distribution from data and prior."""
    # Placeholder for actual inference logic
    return np.ones_like(data)

def _solve_parameters(
    data: np.ndarray,
    prior: np.ndarray,
    posterior: np.ndarray,
    likelihood_func: Callable,
    solver: str = 'gradient_descent',
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """Solve for parameters using the specified solver."""
    if solver == 'closed_form':
        return _solve_closed_form(data, prior)
    elif solver == 'gradient_descent':
        return _solve_gradient_descent(data, prior, posterior, likelihood_func, tol, max_iter)
    elif solver == 'newton':
        return _solve_newton(data, prior, posterior, likelihood_func, tol, max_iter)
    elif solver == 'coordinate_descent':
        return _solve_coordinate_descent(data, prior, posterior, likelihood_func, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _compute_metrics(data: np.ndarray, params: Dict[str, Any], custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute metrics for the Bayesian test."""
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(data, params)
    else:
        # Placeholder for built-in metrics
        pass
    return metrics

def _compute_bayesian_p_value(data: np.ndarray, params: Dict[str, Any], posterior: np.ndarray) -> float:
    """Compute the Bayesian p-value."""
    # Placeholder for actual computation
    return 0.5

# Example usage:
# data = np.random.randn(100, 2)
# prior = np.ones(2) / 2
# def likelihood_func(data, params):
#     return np.prod(np.random.rand(*data.shape))
# result = test_bayesian_p_value_fit(data, prior, likelihood_func)

################################################################################
# test_credible_interval
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def _validate_inputs(
    data: np.ndarray,
    credible_level: float = 0.95,
    prior: Optional[np.ndarray] = None,
    likelihood_func: Callable = lambda x, y: np.sum((x - y) ** 2),
    credible_method: str = 'hpd'
) -> None:
    """
    Validate the inputs for the credible interval test.

    Parameters
    ----------
    data : np.ndarray
        The observed data.
    credible_level : float, optional
        The credible level (e.g., 0.95 for 95% credible interval), by default 0.95.
    prior : Optional[np.ndarray], optional
        The prior distribution, by default None.
    likelihood_func : Callable, optional
        The likelihood function, by default lambda x, y: np.sum((x - y) ** 2).
    credible_method : str, optional
        The method to compute the credible interval ('hpd' or 'etm'), by default 'hpd'.

    Raises
    ------
    ValueError
        If any of the inputs are invalid.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if not (0 < credible_level < 1):
        raise ValueError("Credible level must be between 0 and 1.")
    if prior is not None and len(prior) != len(data):
        raise ValueError("Prior must have the same length as data.")
    if credible_method not in ['hpd', 'etm']:
        raise ValueError("Credible method must be either 'hpd' or 'etm'.")

def _compute_posterior(
    data: np.ndarray,
    prior: Optional[np.ndarray] = None,
    likelihood_func: Callable = lambda x, y: np.sum((x - y) ** 2)
) -> np.ndarray:
    """
    Compute the posterior distribution.

    Parameters
    ----------
    data : np.ndarray
        The observed data.
    prior : Optional[np.ndarray], optional
        The prior distribution, by default None.
    likelihood_func : Callable, optional
        The likelihood function, by default lambda x, y: np.sum((x - y) ** 2).

    Returns
    ------
    np.ndarray
        The posterior distribution.
    """
    if prior is None:
        prior = np.ones_like(data) / len(data)
    likelihood = np.array([likelihood_func(d, data) for d in data])
    posterior = prior * np.exp(-likelihood)
    return posterior / np.sum(posterior)

def _compute_credible_interval(
    posterior: np.ndarray,
    credible_level: float = 0.95,
    method: str = 'hpd'
) -> Tuple[float, float]:
    """
    Compute the credible interval.

    Parameters
    ----------
    posterior : np.ndarray
        The posterior distribution.
    credible_level : float, optional
        The credible level (e.g., 0.95 for 95% credible interval), by default 0.95.
    method : str, optional
        The method to compute the credible interval ('hpd' or 'etm'), by default 'hpd'.

    Returns
    ------
    Tuple[float, float]
        The lower and upper bounds of the credible interval.
    """
    sorted_indices = np.argsort(posterior)[::-1]
    cumulative_prob = np.cumsum(posterior[sorted_indices])
    credible_mass = (1 - credible_level) / 2

    if method == 'hpd':
        lower_idx = np.argmax(cumulative_prob >= credible_mass)
        upper_idx = len(posterior) - np.argmax(cumulative_prob[::-1] >= credible_mass) - 1
    else:  # etm
        lower_idx = np.argmax(cumulative_prob >= credible_mass)
        upper_idx = len(posterior) - np.argmax(cumulative_prob[::-1] >= (1 - credible_mass)) - 1

    return posterior[sorted_indices[lower_idx]], posterior[sorted_indices[upper_idx]]

def test_credible_interval_fit(
    data: np.ndarray,
    credible_level: float = 0.95,
    prior: Optional[np.ndarray] = None,
    likelihood_func: Callable = lambda x, y: np.sum((x - y) ** 2),
    credible_method: str = 'hpd'
) -> Dict[str, Union[Tuple[float, float], Dict[str, float], Dict[str, str], List[str]]]:
    """
    Compute the credible interval for Bayesian analysis.

    Parameters
    ----------
    data : np.ndarray
        The observed data.
    credible_level : float, optional
        The credible level (e.g., 0.95 for 95% credible interval), by default 0.95.
    prior : Optional[np.ndarray], optional
        The prior distribution, by default None.
    likelihood_func : Callable, optional
        The likelihood function, by default lambda x, y: np.sum((x - y) ** 2).
    credible_method : str, optional
        The method to compute the credible interval ('hpd' or 'etm'), by default 'hpd'.

    Returns
    ------
    Dict[str, Union[Tuple[float, float], Dict[str, float], Dict[str, str], List[str]]]
        A dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(data, credible_level, prior, likelihood_func, credible_method)
    posterior = _compute_posterior(data, prior, likelihood_func)
    interval = _compute_credible_interval(posterior, credible_level, credible_method)

    result = {
        "result": interval,
        "metrics": {"credible_level": credible_level},
        "params_used": {
            "prior": prior is not None,
            "likelihood_func": likelihood_func.__name__ if hasattr(likelihood_func, '__name__') else "custom",
            "credible_method": credible_method
        },
        "warnings": []
    }

    return result

# Example usage:
# data = np.random.normal(0, 1, 1000)
# result = test_credible_interval_fit(data, credible_level=0.95, prior=None)

################################################################################
# test_hypothesis_testing
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(data: np.ndarray, prior: np.ndarray) -> None:
    """Validate input data and prior probabilities."""
    if not isinstance(data, np.ndarray) or not isinstance(prior, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if prior.ndim != 1:
        raise ValueError("Prior must be a 1D array")
    if len(prior) != data.shape[1]:
        raise ValueError("Prior length must match number of features")
    if not np.allclose(np.sum(prior), 1.0):
        raise ValueError("Prior probabilities must sum to 1")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")
    if np.any(np.isnan(prior)) or np.any(np.isinf(prior)):
        raise ValueError("Prior contains NaN or infinite values")

def _compute_likelihood(data: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Compute likelihood of each observation given the prior."""
    return np.exp(np.log(prior) + np.mean(data * np.log(data), axis=0))

def _compute_posterior(likelihood: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Compute posterior probabilities."""
    return (likelihood * prior) / np.sum(likelihood * prior)

def _compute_statistic(data: np.ndarray, posterior: np.ndarray) -> float:
    """Compute test statistic."""
    return np.sum(posterior * np.mean(data, axis=0))

def _compute_p_value(statistic: float, alternative: str = 'two-sided') -> float:
    """Compute p-value based on test statistic."""
    if alternative == 'two-sided':
        return 2 * min(statistic, 1 - statistic)
    elif alternative == 'less':
        return statistic
    elif alternative == 'greater':
        return 1 - statistic
    else:
        raise ValueError("Alternative must be 'two-sided', 'less' or 'greater'")

def test_hypothesis_testing_fit(
    data: np.ndarray,
    prior: np.ndarray,
    normalization: str = 'none',
    statistic_func: Optional[Callable] = None,
    p_value_func: Optional[Callable] = None,
    alternative: str = 'two-sided',
    **kwargs
) -> Dict[str, Any]:
    """
    Perform Bayesian hypothesis testing.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    prior : np.ndarray
        Prior probabilities of shape (n_features,)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    statistic_func : Callable, optional
        Custom function to compute test statistic
    p_value_func : Callable, optional
        Custom function to compute p-value
    alternative : str, optional
        Alternative hypothesis ('two-sided', 'less', 'greater')
    **kwargs : dict
        Additional keyword arguments for custom functions

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.random.rand(100, 5)
    >>> prior = np.ones(5) / 5
    >>> result = test_hypothesis_testing_fit(data, prior)
    """
    _validate_inputs(data, prior)

    # Normalization
    if normalization == 'standard':
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif normalization == 'minmax':
        data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif normalization == 'robust':
        data = (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))

    # Compute likelihood and posterior
    likelihood = _compute_likelihood(data, prior)
    posterior = _compute_posterior(likelihood, prior)

    # Compute statistic
    if statistic_func is not None:
        statistic = statistic_func(data, posterior, **kwargs)
    else:
        statistic = _compute_statistic(data, posterior)

    # Compute p-value
    if p_value_func is not None:
        p_value = p_value_func(statistic, alternative=alternative, **kwargs)
    else:
        p_value = _compute_p_value(statistic, alternative=alternative)

    return {
        'result': {
            'statistic': statistic,
            'p_value': p_value
        },
        'metrics': {
            'posterior': posterior,
            'likelihood': likelihood
        },
        'params_used': {
            'normalization': normalization,
            'alternative': alternative
        },
        'warnings': []
    }

################################################################################
# test_model_comparison
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def test_model_comparison_fit(
    models: List[Callable],
    data: np.ndarray,
    prior_weights: Optional[np.ndarray] = None,
    metric: Union[str, Callable] = 'mse',
    normalization: str = 'standard',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric_kwargs: Optional[Dict] = None
) -> Dict:
    """
    Compare multiple models using Bayesian model comparison.

    Parameters:
    -----------
    models : List[Callable]
        List of callable model functions to compare.
    data : np.ndarray
        Input data for model fitting.
    prior_weights : Optional[np.ndarray]
        Prior weights for each model. If None, uniform priors are used.
    metric : Union[str, Callable]
        Metric to use for model comparison. Can be 'mse', 'mae', 'r2', or a custom callable.
    normalization : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    solver : str
        Solver method: 'closed_form', 'gradient_descent', or 'newton'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric_kwargs : Optional[Dict]
        Additional keyword arguments for custom metric functions.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(models, data, prior_weights)

    # Normalize data if required
    normalized_data = _normalize_data(data, normalization)

    # Initialize results dictionary
    results = {
        'result': {},
        'metrics': {},
        'params_used': {},
        'warnings': []
    }

    # Set default prior weights if not provided
    if prior_weights is None:
        prior_weights = np.ones(len(models)) / len(models)

    # Compare models
    for i, model in enumerate(models):
        try:
            # Fit model
            params = _fit_model(model, normalized_data, solver, tol, max_iter)

            # Compute metrics
            metrics = _compute_metrics(model, normalized_data, metric, custom_metric_kwargs)

            # Store results
            results['result'][f'model_{i}'] = {
                'params': params,
                'posterior_weight': _compute_posterior_weight(metrics, prior_weights[i])
            }
            results['metrics'][f'model_{i}'] = metrics
            results['params_used'][f'model_{i}'] = {
                'solver': solver,
                'normalization': normalization
            }
        except Exception as e:
            results['warnings'].append(f"Model {i} failed: {str(e)}")

    return results

def _validate_inputs(models: List[Callable], data: np.ndarray, prior_weights: Optional[np.ndarray]) -> None:
    """Validate input parameters."""
    if not isinstance(models, list) or not all(callable(m) for m in models):
        raise ValueError("Models must be a list of callable functions.")
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array.")
    if prior_weights is not None:
        if len(prior_weights) != len(models):
            raise ValueError("Length of prior_weights must match number of models.")
        if not np.isclose(np.sum(prior_weights), 1.0):
            raise ValueError("Prior weights must sum to 1.")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _fit_model(model: Callable, data: np.ndarray, solver: str, tol: float, max_iter: int) -> Dict:
    """Fit model using specified solver."""
    if solver == 'closed_form':
        return _fit_closed_form(model, data)
    elif solver == 'gradient_descent':
        return _fit_gradient_descent(model, data, tol, max_iter)
    elif solver == 'newton':
        return _fit_newton(model, data, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _fit_closed_form(model: Callable, data: np.ndarray) -> Dict:
    """Fit model using closed-form solution."""
    # Implement closed-form fitting logic
    return {'params': np.random.rand(data.shape[1])}  # Placeholder

def _fit_gradient_descent(model: Callable, data: np.ndarray, tol: float, max_iter: int) -> Dict:
    """Fit model using gradient descent."""
    # Implement gradient descent fitting logic
    return {'params': np.random.rand(data.shape[1])}  # Placeholder

def _fit_newton(model: Callable, data: np.ndarray, tol: float, max_iter: int) -> Dict:
    """Fit model using Newton's method."""
    # Implement Newton's method fitting logic
    return {'params': np.random.rand(data.shape[1])}  # Placeholder

def _compute_metrics(model: Callable, data: np.ndarray, metric: Union[str, Callable], kwargs: Optional[Dict]) -> Dict:
    """Compute metrics for model comparison."""
    if isinstance(metric, str):
        if metric == 'mse':
            return {'mse': _compute_mse(model, data)}
        elif metric == 'mae':
            return {'mae': _compute_mae(model, data)}
        elif metric == 'r2':
            return {'r2': _compute_r2(model, data)}
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        return {'custom': metric(model, data, **(kwargs or {}))}
    else:
        raise ValueError("Metric must be a string or callable.")

def _compute_mse(model: Callable, data: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    # Implement MSE calculation
    return np.random.rand()  # Placeholder

def _compute_mae(model: Callable, data: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    # Implement MAE calculation
    return np.random.rand()  # Placeholder

def _compute_r2(model: Callable, data: np.ndarray) -> float:
    """Compute R-squared."""
    # Implement R2 calculation
    return np.random.rand()  # Placeholder

def _compute_posterior_weight(metrics: Dict, prior_weight: float) -> float:
    """Compute posterior weight for model."""
    # Implement Bayesian model comparison logic
    return prior_weight * np.random.rand()  # Placeholder

################################################################################
# test_bayesian_networks
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List, Any

def _validate_inputs(
    data: np.ndarray,
    target: Optional[np.ndarray] = None,
    network_structure: Optional[List[List[int]]] = None
) -> Dict[str, Any]:
    """
    Validate the inputs for Bayesian network testing.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    target : Optional[np.ndarray]
        Target values if available. Shape (n_samples,).
    network_structure : Optional[List[List[int]]]
        Adjacency matrix representing the network structure.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing validation results and warnings.
    """
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")

    warnings = []
    if target is not None:
        if len(data) != len(target):
            raise ValueError("Data and target must have the same number of samples.")
        if np.any(np.isnan(target)):
            raise ValueError("Target contains NaN values.")

    if network_structure is not None:
        if len(network_structure) != data.shape[1]:
            raise ValueError("Network structure must match the number of features.")
        for row in network_structure:
            if len(row) != data.shape[1]:
                raise ValueError("Network structure must be a square matrix.")

    return {"warnings": warnings}

def _normalize_data(
    data: np.ndarray,
    method: str = "standard",
    custom_normalizer: Optional[Callable] = None
) -> np.ndarray:
    """
    Normalize the input data using the specified method.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    method : str, optional
        Normalization method: 'none', 'standard', 'minmax', 'robust'.
    custom_normalizer : Optional[Callable]
        Custom normalization function.

    Returns
    -------
    np.ndarray
        Normalized data.
    """
    if custom_normalizer is not None:
        return custom_normalizer(data)

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

def _compute_metrics(
    predictions: np.ndarray,
    target: np.ndarray,
    metrics: List[str],
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """
    Compute the specified metrics for the predictions.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted values of shape (n_samples,).
    target : np.ndarray
        True target values of shape (n_samples,).
    metrics : List[str]
        List of metric names: 'mse', 'mae', 'r2', 'logloss'.
    custom_metric : Optional[Callable]
        Custom metric function.

    Returns
    -------
    Dict[str, float]
        Dictionary of computed metrics.
    """
    result = {}
    if custom_metric is not None:
        result["custom"] = custom_metric(predictions, target)

    for metric in metrics:
        if metric == "mse":
            result["mse"] = np.mean((predictions - target) ** 2)
        elif metric == "mae":
            result["mae"] = np.mean(np.abs(predictions - target))
        elif metric == "r2":
            ss_res = np.sum((predictions - target) ** 2)
            ss_tot = np.sum((target - np.mean(target)) ** 2)
            result["r2"] = 1 - (ss_res / (ss_tot + 1e-8))
        elif metric == "logloss":
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            result["logloss"] = -np.mean(target * np.log(predictions) + (1 - target) * np.log(1 - predictions))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return result

def _estimate_parameters(
    data: np.ndarray,
    target: Optional[np.ndarray] = None,
    network_structure: Optional[List[List[int]]] = None,
    solver: str = "closed_form",
    custom_solver: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Estimate the parameters of the Bayesian network.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    target : Optional[np.ndarray]
        Target values if available. Shape (n_samples,).
    network_structure : Optional[List[List[int]]]
        Adjacency matrix representing the network structure.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.
    custom_solver : Optional[Callable]
        Custom solver function.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the estimated parameters and solver details.
    """
    if custom_solver is not None:
        return custom_solver(data, target)

    if solver == "closed_form":
        # Placeholder for closed form solution
        params = np.linalg.pinv(data.T @ data) @ (data.T @ target)
    elif solver == "gradient_descent":
        # Placeholder for gradient descent
        params = np.random.rand(data.shape[1])
    elif solver == "newton":
        # Placeholder for Newton's method
        params = np.random.rand(data.shape[1])
    elif solver == "coordinate_descent":
        # Placeholder for coordinate descent
        params = np.random.rand(data.shape[1])
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return {"params": params, "solver": solver}

def test_bayesian_networks_fit(
    data: np.ndarray,
    target: Optional[np.ndarray] = None,
    network_structure: Optional[List[List[int]]] = None,
    normalization: str = "standard",
    metrics: List[str] = ["mse"],
    solver: str = "closed_form",
    custom_normalizer: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None,
    custom_solver: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Fit a Bayesian network model to the data and compute metrics.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    target : Optional[np.ndarray]
        Target values if available. Shape (n_samples,).
    network_structure : Optional[List[List[int]]]
        Adjacency matrix representing the network structure.
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', 'robust'.
    metrics : List[str], optional
        List of metric names: 'mse', 'mae', 'r2', 'logloss'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.
    custom_normalizer : Optional[Callable]
        Custom normalization function.
    custom_metric : Optional[Callable]
        Custom metric function.
    custom_solver : Optional[Callable]
        Custom solver function.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validation = _validate_inputs(data, target, network_structure)

    # Normalize data
    normalized_data = _normalize_data(data, normalization, custom_normalizer)

    # Estimate parameters
    params_result = _estimate_parameters(normalized_data, target, network_structure, solver, custom_solver)

    # Compute predictions (placeholder)
    predictions = np.random.rand(len(data))

    # Compute metrics
    metrics_result = _compute_metrics(predictions, target if target is not None else predictions, metrics, custom_metric)

    return {
        "result": {"predictions": predictions},
        "metrics": metrics_result,
        "params_used": params_result,
        "warnings": validation["warnings"]
    }
